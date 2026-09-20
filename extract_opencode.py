#!/usr/bin/env python3
"""
Extract ALL OpenCode conversation data
Supports: CLI (JSON files) and Desktop (Tauri .dat files)

Storage locations:
- CLI: ~/.local/share/opencode/ (Linux/macOS)
- Desktop: Platform-specific Tauri app data directories

Features:
- Extracts conversations from sessions WITH and WITHOUT metadata files
- Reconstructs session metadata (directory, title, timestamps) from message content
- Assembles complete messages from message metadata + parts
- Handles sessions where session files are missing or corrupted
"""

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import struct
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from source_manifest import (
    SourceAdmissionError,
    SourceManifestError,
    SourceManifestIndex,
    validate_source_admission,
)

OPENCODE_MAX_RECORD_CHARS = int(os.environ.get("OPENCODE_MAX_RECORD_CHARS", "250000"))
# The canonical builder materializes messages and normalized events.  Keep the
# provider record well below its final bound so tool observations are not able
# to turn a safe ingress record into an oversized canonical episode.
OPENCODE_INGRESS_TARGET_FRACTION = 0.25
OPENCODE_TEXT_PAYLOAD_MAX_CHARS = 64_000
OPENCODE_OBSERVATION_PAYLOAD_MAX_CHARS = 16_000
OPENCODE_ARTIFACT_PAYLOAD_MAX_CHARS = 64_000
OPENCODE_EXTERNAL_OUTPUT_MAX_BYTES = 64 * 1024 * 1024
SOURCE_MANIFEST_SCHEMA = 'ai-data-extraction/opencode-source-manifest/v1'
SOURCE_MANIFEST_VERSION = '1.1.0'

def find_opencode_installations(home=None):
    """Find all OpenCode installation directories"""
    system = platform.system()
    home = (home or Path.home()).expanduser()
    
    locations = []
    
    # CLI storage locations (XDG Base Directory)
    if system == "Darwin":  # macOS
        cli_dirs = [
            home / "Library/Application Support/opencode",
            home / '.local/share/opencode'
        ]
    elif system == "Linux":
        cli_dirs = [
            home / '.local/share/opencode'
        ]
    elif system == "Windows":
        cli_dirs = [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')) / 'opencode'
        ]
    else:
        cli_dirs = [home / '.local/share/opencode']
    
    for cli_dir in cli_dirs:
        if cli_dir.exists():
            locations.append(('cli', cli_dir))
    
    # Desktop storage locations (Tauri app data)
    if system == "Darwin":  # macOS
        desktop_dirs = [
            home / "Library/Application Support/ai.opencode.app"
        ]
    elif system == "Linux":
        desktop_dirs = [
            home / ".local/share/ai.opencode.app"
        ]
    elif system == "Windows":
        desktop_dirs = [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')) / 'ai.opencode.app'
        ]
    else:
        desktop_dirs = []
    
    for desktop_dir in desktop_dirs:
        if desktop_dir.exists():
            locations.append(('desktop', desktop_dir))
    
    return locations

def read_tauri_store(dat_file):
    """
    Parse Tauri store .dat files
    Format: Simple key-value pairs with length prefixes
    """
    try:
        with open(dat_file, 'rb') as f:
            data = f.read()
        
        store = {}
        offset = 0
        
        while offset < len(data):
            # Try to read key length (4 bytes, little-endian)
            if offset + 4 > len(data):
                break
            
            key_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # Sanity check
            if key_len > 10000 or offset + key_len > len(data):
                break
            
            # Read key
            key = data[offset:offset+key_len].decode('utf-8', errors='ignore')
            offset += key_len
            
            # Read value length
            if offset + 4 > len(data):
                break
            
            value_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # Sanity check
            if value_len > 1000000 or offset + value_len > len(data):
                break
            
            # Read value
            try:
                value_bytes = data[offset:offset+value_len]
                value = json.loads(value_bytes.decode('utf-8'))
                store[key] = value
            except:
                pass
            
            offset += value_len
        
        return store
    
    except Exception as e:
        print(f"Error reading Tauri store {dat_file}: {e}")
        return {}

def extract_directory_from_content(text):
    """
    Try to extract a directory path from text content (e.g., tool commands).
    Looks for common patterns like 'cd /path/to/dir' or paths in commands.
    """
    if not text:
        return None
    
    import re
    
    # Pattern 1: cd command followed by path
    cd_pattern = r'cd\s+(["\']?)([^\s\'"]+)\1'
    matches = re.findall(cd_pattern, text)
    for match in matches:
        path = match[1] if isinstance(match, tuple) else match
        if path and (path.startswith('/') or path.startswith('~') or path[1:].startswith(':')):
            return path
    
    # Pattern 2: Common working directory indicators
    cwd_pattern = r'(?:working\s+)?directory[:\s]+(["\']?)([^\s\'"]+)\1'
    matches = re.findall(cwd_pattern, text)
    for match in matches:
        path = match[1] if isinstance(match, tuple) else match
        if path and (path.startswith('/') or path.startswith('~') or path[1:].startswith(':')):
            return path
    
    # Pattern 3: Extract absolute paths (Unix-style)
    abs_path_pattern = r'(?:^|\s|/)(/[^/\s\'"]{2,})'
    matches = re.findall(abs_path_pattern, text)
    for path in matches:
        if path and len(path) > 3 and not path.endswith('.') and not path.endswith('..'):
            return path
    
    return None


def extract_project_id_from_content(text):
    """
    Try to extract a project ID from text content.
    Often appears in tool commands or git operations.
    """
    if not text:
        return None
    
    import re
    
    # Pattern: project IDs in commands
    project_pattern = r'(?:project[-_]?id|project)[=:\s]+([a-zA-Z0-9_-]+)'
    match = re.search(project_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None

def load_sidecar_json(storage_dir, category, session_id):
    sidecar_file = storage_dir / 'storage' / category / f'{session_id}.json'
    if not sidecar_file.exists():
        return None
    try:
        with open(sidecar_file) as f:
            return json.load(f)
    except Exception:
        return None

def apply_part_to_message(part_data, message, all_content):
    part_type = part_data.get('type')
    part_text = part_data.get('text', '')

    if part_text:
        all_content.append(part_text)

    if part_type == 'text':
        message.setdefault('_content_parts', []).append(part_text)
    elif part_type == 'tool' or part_type == 'tool-call':
        state = part_data.get('state', {})
        tool_name = part_data.get('tool', part_data.get('name'))

        tool_call = {
            'id': part_data.get('callID', part_data.get('id')),
            'name': tool_name,
            'input': state.get('input', part_data.get('input'))
        }

        message.setdefault('tool_calls', []).append(tool_call)

        if state.get('status') == 'completed' and 'output' in state:
            message.setdefault('tool_results', []).append({
                'tool_call_id': part_data.get('callID'),
                'tool': tool_name,
                'output': state['output']
            })
    elif part_type == 'tool-result':
        message.setdefault('tool_results', []).append({
            'tool_call_id': part_data.get('toolCallID'),
            'output': part_data.get('output')
        })
    elif part_type == 'code':
        code_text = part_data.get('text', '')
        language = part_data.get('language', '')
        message.setdefault('_content_parts', []).append(f"```{language}\n{code_text}\n```")
    elif part_type == 'reasoning':
        reasoning_text = part_data.get('text', '')
        if reasoning_text:
            message.setdefault('_reasoning_parts', []).append(reasoning_text)
    elif part_text and part_type not in {'step-start', 'step-finish'}:
        message.setdefault('_content_parts', []).append(part_text)

def finalize_message(message):
    content_parts = message.pop('_content_parts', [])
    # Reasoning is retained by the original OpenCode store.  Derived raw
    # exports intentionally omit it; the canonical builder also enforces this
    # boundary for every provider.
    message.pop('_reasoning_parts', [])
    message['content'] = '\n'.join(content_parts)
    return message

def _json_digest(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


def _json_size(value: Any) -> int:
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return sum(len(chunk) for chunk in encoder.iterencode(value))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path):
    try:
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
    except (FileNotFoundError, OSError):
        return None
    stable = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    return {
        'file_name': path.name,
        'bytes': after.st_size,
        'mtime_ns': after.st_mtime_ns,
        'sha256': digest,
        'hash_stable': stable,
    }


def _sqlite_source_origin(db_path):
    wal_path = Path(f'{db_path}-wal')
    database = _file_fingerprint(db_path)
    wal = _file_fingerprint(wal_path) if wal_path.exists() else None
    source_fingerprint = _json_digest({
        'database_sha256': database.get('sha256') if database else None,
        'wal_sha256': wal.get('sha256') if wal else None,
    })
    return {
        'store_type': 'sqlite',
        'database': database,
        'wal': wal,
        'source_fingerprint': f'sha256:{source_fingerprint}',
        'read_mode': 'read_only_with_wal',
        'ordering': 'message.time_created,message.id,part.time_created,part.id',
        'content_policy': 'reasoning_omitted; large_payloads_bounded',
    }


def _sidecar_fingerprints(storage_dir, session_id):
    sidecars = []
    for category in (
        'session_diff',
        'directory-readme',
        'agent-usage-reminder',
        'rules-injector',
    ):
        path = storage_dir / 'storage' / category / f'{session_id}.json'
        if not path.exists():
            continue
        fingerprint = _file_fingerprint(path)
        if fingerprint is not None:
            sidecars.append({
                'category': category,
                'file_name': fingerprint['file_name'],
                'bytes': fingerprint['bytes'],
                'mtime_ns': fingerprint['mtime_ns'],
                'sha256': fingerprint['sha256'],
            })
    return sidecars


def _session_sidecar_fingerprints(storage_dir, session_id):
    """Fingerprint session metadata in every legacy storage layout."""
    sidecars = []
    seen = set()
    for category in ('session', 'session_diff'):
        root = storage_dir / 'storage' / category
        if not root.exists():
            continue
        for path in sorted(root.rglob(f'{session_id}.json')):
            fingerprint = _file_fingerprint(path)
            if fingerprint is None or path in seen:
                continue
            seen.add(path)
            sidecars.append({
                'category': category,
                'file_name': fingerprint['file_name'],
                'relative_path_sha256': _json_digest(
                    path.relative_to(storage_dir).as_posix()
                ),
                'bytes': fingerprint['bytes'],
                'mtime_ns': fingerprint['mtime_ns'],
                'sha256': fingerprint['sha256'],
            })
    return sidecars


def _read_json_source(
    path,
    *,
    storage_dir,
    source_manifest,
    root_label,
    source_class,
):
    """Read one bounded JSON source and optionally bind its exact ledger row."""

    if source_manifest is None:
        with path.open(encoding='utf-8') as source:
            return json.load(source), None

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    relative_path_sha256 = _source_relative_path_sha256(path, storage_dir)
    admission = source_manifest.admit_path(
        source_sha256=digest,
        relative_path_sha256=relative_path_sha256,
        provider='opencode',
        root_label=root_label,
        source_class=source_class,
    )
    admission = validate_source_admission(
        admission,
        source_sha256=digest,
        source_bytes=len(payload),
    )
    return json.loads(payload), admission


def _admit_json_file(
    path,
    *,
    storage_dir,
    source_manifest,
    root_label,
    source_class,
):
    """Bind a non-record JSON file without retaining its contents."""

    if source_manifest is None:
        return None
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    admission = source_manifest.admit_path(
        source_sha256=digest,
        relative_path_sha256=_source_relative_path_sha256(path, storage_dir),
        provider='opencode',
        root_label=root_label,
        source_class=source_class,
    )
    return validate_source_admission(
        admission,
        source_sha256=digest,
        source_bytes=len(payload),
    )


def _json_sidecar_index(storage_dir, *, source_manifest=None, root_label='cli'):
    """Index session sidecars once, retaining metadata and digest lineage only."""

    index = {}
    for category in ('session', 'session_diff'):
        root = storage_dir / 'storage' / category
        if not root.exists():
            continue
        for path in sorted(root.rglob('*.json')):
            session_id = path.stem
            entry = index.setdefault(
                session_id,
                {'metadata': None, 'source_refs': set(), 'sidecar_count': 0},
            )
            admission = _admit_json_file(
                path,
                storage_dir=storage_dir,
                source_manifest=source_manifest,
                root_label=root_label,
                source_class='conversation_sidecar',
            )
            if admission is not None:
                entry['source_refs'].add(admission['source_ref_sha256'])
            entry['sidecar_count'] += 1
            if category != 'session' or entry['metadata'] is not None:
                continue
            try:
                value, _admission = _read_json_source(
                    path,
                    storage_dir=storage_dir,
                    source_manifest=None,
                    root_label=root_label,
                    source_class='conversation_sidecar',
                )
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                entry['metadata'] = value
    return index


def _json_tool_output_index(storage_dir, *, source_manifest=None, root_label='cli'):
    """Index tool-output files once so each referenced artifact is O(n)."""

    index = {}
    root = storage_dir / 'tool-output'
    if not root.exists():
        return index
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        fingerprint = _file_fingerprint(path)
        if fingerprint is None:
            continue
        admission = None
        if source_manifest is not None:
            admission = source_manifest.admit_path(
                source_sha256=fingerprint['sha256'],
                relative_path_sha256=_source_relative_path_sha256(
                    path, storage_dir
                ),
                provider='opencode',
                root_label=root_label,
                source_class='tool_output',
            )
            admission = validate_source_admission(
                admission,
                source_sha256=fingerprint['sha256'],
                source_bytes=fingerprint['bytes'],
            )
        index[path.name] = {
            'sha256': fingerprint['sha256'],
            'admission': admission,
        }
    return index


def _source_relative_path_sha256(path, root):
    return hashlib.sha256(
        path.relative_to(root).as_posix().encode('utf-8')
    ).hexdigest()


def _output_path_references_in_json_parts(storage_dir):
    """Return external tool-output basenames referenced by legacy parts."""
    references = set()
    part_root = storage_dir / 'storage' / 'part'
    if not part_root.exists():
        return references
    for path in part_root.rglob('*.json'):
        try:
            with path.open('r', encoding='utf-8') as source:
                value = json.load(source)
        except (OSError, json.JSONDecodeError):
            continue
        metadata = (
            value.get('state', {}).get('metadata', {})
            if isinstance(value, dict)
            else {}
        )
        output_path = metadata.get('outputPath') if isinstance(metadata, dict) else None
        if output_path:
            references.add(Path(str(output_path)).name)
    return references


def _sqlite_read_uri(db_path, *, immutable=False):
    query = 'mode=ro&immutable=1' if immutable else 'mode=ro'
    return f'file:{db_path}?{query}'


def _output_path_references_in_database(db_path, output_names, *, immutable=False):
    """Find external tool-output references without materializing DB content."""
    if not db_path.exists() or not output_names:
        return set()
    references = set()
    try:
        uri = _sqlite_read_uri(db_path, immutable=immutable)
        conn = sqlite3.connect(uri, uri=True)
        try:
            for name in output_names:
                row = conn.execute(
                    'SELECT 1 FROM part WHERE data LIKE ? LIMIT 1',
                    (f'%{name}%',),
                ).fetchone()
                if row:
                    references.add(name)
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return references
    return references


def build_source_manifest(
    storage_dir,
    *,
    emitted_db_session_ids=None,
    emitted_storage_session_ids=None,
    source_manifest: SourceManifestIndex | None = None,
    root_label='cli',
    immutable_snapshot=False,
):
    """Hash and classify every OpenCode conversation candidate file.

    This is intentionally metadata-only.  It does not export source content;
    it records whether the current adapter emitted, linked, or could not link
    each candidate file to a session or external tool observation.
    """
    storage_dir = Path(storage_dir)
    emitted_db_session_ids = set(emitted_db_session_ids or ())
    emitted_storage_session_ids = set(emitted_storage_session_ids or ())
    message_root = storage_dir / 'storage' / 'message'
    part_root = storage_dir / 'storage' / 'part'
    session_root = storage_dir / 'storage' / 'session'
    session_diff_root = storage_dir / 'storage' / 'session_diff'
    tool_output_root = storage_dir / 'tool-output'

    message_to_session = {}
    known_session_ids = set(emitted_db_session_ids) | set(emitted_storage_session_ids)
    parse_errors = 0
    if message_root.exists():
        for path in sorted(message_root.rglob('msg_*.json')):
            try:
                with path.open('r', encoding='utf-8') as source:
                    value = json.load(source)
                message_id = value.get('id') if isinstance(value, dict) else None
                session_id = value.get('sessionID') if isinstance(value, dict) else None
                session_id = session_id or (
                    path.parent.name if path.parent.name.startswith('ses_') else None
                )
                if message_id and session_id:
                    message_to_session[str(message_id)] = str(session_id)
                    known_session_ids.add(str(session_id))
            except (OSError, json.JSONDecodeError):
                parse_errors += 1

    json_output_refs = _output_path_references_in_json_parts(storage_dir)
    output_names = {
        path.name
        for path in tool_output_root.iterdir()
        if path.is_file()
    } if tool_output_root.exists() else set()
    db_path = storage_dir / 'opencode.db'
    db_output_refs = _output_path_references_in_database(
        db_path,
        output_names,
        immutable=immutable_snapshot,
    )
    output_refs = json_output_refs | db_output_refs

    candidates = []
    if db_path.exists():
        candidates.append((db_path, 'conversation_store', None, 'database'))
    wal_path = Path(f'{db_path}-wal')
    if wal_path.exists() and not immutable_snapshot:
        candidates.append((wal_path, 'conversation_store', None, 'database_wal'))
    for path in sorted(message_root.rglob('*.json')) if message_root.exists() else []:
        candidates.append((path, 'conversation_store', path.parent.name, 'message'))
    for path in sorted(part_root.rglob('*.json')) if part_root.exists() else []:
        message_id = path.parent.name
        candidates.append((path, 'conversation_store', message_to_session.get(message_id), 'part'))
    for root, kind in ((session_root, 'session'), (session_diff_root, 'session_diff')):
        if not root.exists():
            continue
        for path in sorted(root.rglob('*.json')):
            candidates.append((path, 'conversation_sidecar', path.stem, kind))
    if tool_output_root.exists():
        for path in sorted(tool_output_root.iterdir()):
            if path.is_file():
                candidates.append((path, 'tool_output', None, 'tool_output'))

    entries = []
    for path, source_class, session_hint, kind in candidates:
        try:
            fingerprint = _file_fingerprint(path)
            relative_hash = _source_relative_path_sha256(path, storage_dir)
        except OSError as exc:
            entries.append({
                'provider': 'opencode-cli',
                'source_class': source_class,
                'source_file_name': path.name,
                'source_relative_path_sha256': None,
                'source_bytes': None,
                'status': 'read_error',
                'error_type': type(exc).__name__,
            })
            continue

        session_id = str(session_hint) if session_hint else None
        if kind == 'database':
            status = 'emitted' if emitted_db_session_ids else 'available_secondary'
        elif kind == 'database_wal':
            status = 'linked_to_database'
        elif kind in {'message', 'part'}:
            status = (
                'emitted_via_json_storage'
                if session_id in emitted_storage_session_ids
                else 'linked_to_session' if session_id in known_session_ids else 'unlinked'
            )
        elif kind in {'session', 'session_diff'}:
            session_id = path.stem
            status = 'linked_to_session' if session_id in known_session_ids else 'unlinked'
        else:
            status = (
                'referenced_by_store'
                if path.name in output_refs
                else 'not_referenced_by_known_store'
            )

        entry = {
            'provider': 'opencode-cli',
            'source_class': source_class,
            'source_kind': kind,
            'source_file_name': path.name,
            'source_relative_path_sha256': relative_hash,
            'source_file_sha256': fingerprint['sha256'],
            'source_bytes': fingerprint['bytes'],
            'status': status,
        }
        if source_manifest is not None:
            admission = source_manifest.admit_path(
                source_sha256=fingerprint['sha256'],
                relative_path_sha256=relative_hash,
                provider='opencode',
                root_label=root_label,
                source_class=source_class,
            )
            admission = validate_source_admission(
                admission,
                source_sha256=fingerprint['sha256'],
                source_bytes=fingerprint['bytes'],
            )
            entry.update({
                'source_manifest_revision': admission['source_manifest_revision'],
                'source_ref_sha256': admission['source_ref_sha256'],
                'source_snapshot_status': admission['source_snapshot_status'],
            })
        if session_id:
            entry['session_id_sha256'] = hashlib.sha256(
                session_id.encode('utf-8')
            ).hexdigest()
        if source_class == 'tool_output':
            entry['referenced_by_json_parts'] = path.name in json_output_refs
            entry['referenced_by_database'] = path.name in db_output_refs
        entries.append(entry)

    return {
        'schema_version': SOURCE_MANIFEST_SCHEMA,
        'extractor_version': SOURCE_MANIFEST_VERSION,
        'provider': 'opencode-cli',
        'source_files': entries,
        'counts': {
            'source_files': len(entries),
            'source_parse_errors': parse_errors,
            'emitted_or_linked': sum(
                entry.get('status') in {
                    'emitted', 'emitted_via_json_storage', 'linked_to_session',
                    'linked_to_database', 'referenced_by_store',
                }
                for entry in entries
            ),
            'unlinked_or_unreferenced': sum(
                entry.get('status') in {'unlinked', 'not_referenced_by_known_store'}
                for entry in entries
            ),
            'read_errors': sum(entry.get('status') == 'read_error' for entry in entries),
        },
    }


def write_source_manifest(output_dir, timestamp, manifest):
    manifest_file = Path(output_dir) / f'opencode_source_manifest_{timestamp}.json'
    with manifest_file.open('w', encoding='utf-8') as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    return manifest_file


def _decode_sql_json(value):
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped.startswith(('{', '[')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _bounded_payload(
    value,
    *,
    max_chars,
    truncations: list[dict[str, Any]] | None,
    kind,
    message_index,
    part_id=None,
    call_id=None,
):
    if value is None:
        return None
    if isinstance(value, str):
        original_chars = len(value)
        if original_chars <= max_chars:
            return value
        digest = hashlib.sha256(value.encode('utf-8')).hexdigest()
        marker = (
            '\n<OPENCODE_PAYLOAD_TRUNCATED>'
            f' original_chars={original_chars} sha256={digest}\n'
        )
        available = max(2, max_chars - len(marker))
        head = max(1, int(available * 0.7))
        tail = max(1, available - head)
        bounded = value[:head] + marker + value[-tail:]
        policy = 'head_tail'
        kept_chars = len(bounded)
        original_digest = digest
    else:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
        original_chars = len(serialized)
        if original_chars <= max_chars:
            return value
        original_digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        bounded = {
            'truncated': True,
            'original_chars': original_chars,
            'sha256': original_digest,
        }
        policy = 'digest_only'
        kept_chars = _json_size(bounded)
    entry = {
        'kind': kind,
        'source_message_index': message_index,
        'original_chars': original_chars,
        'kept_chars': kept_chars,
        'sha256': original_digest,
        'policy': policy,
    }
    if part_id is not None:
        entry['source_part_id_sha256'] = hashlib.sha256(
            str(part_id).encode('utf-8')
        ).hexdigest()
    if call_id is not None:
        entry['call_id_sha256'] = hashlib.sha256(
            str(call_id).encode('utf-8')
        ).hexdigest()
    if truncations is not None:
        truncations.append(entry)
    return bounded


def _opencode_payload_limits(max_record_chars):
    target = max(128, int(max_record_chars * OPENCODE_INGRESS_TARGET_FRACTION))
    return (
        target,
        max(128, min(OPENCODE_TEXT_PAYLOAD_MAX_CHARS, target // 2)),
        max(64, min(OPENCODE_OBSERVATION_PAYLOAD_MAX_CHARS, target // 8)),
        max(128, min(OPENCODE_ARTIFACT_PAYLOAD_MAX_CHARS, target // 2)),
    )


def _external_tool_output(tool_output_dir, metadata):
    """Load an explicitly referenced tool output, never an arbitrary path."""
    if tool_output_dir is None or not isinstance(metadata, dict):
        return None
    output_path = metadata.get('outputPath')
    if not output_path:
        return None
    root = Path(tool_output_dir)
    candidate = root / Path(str(output_path)).name
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
        if candidate_resolved.parent != root_resolved or not candidate_resolved.is_file():
            return None
        stat = candidate_resolved.stat()
        fingerprint = _file_fingerprint(candidate_resolved)
        if fingerprint is None or stat.st_size > OPENCODE_EXTERNAL_OUTPUT_MAX_BYTES:
            return None
        with candidate_resolved.open('r', encoding='utf-8', errors='replace') as source:
            value = source.read()
    except OSError:
        return None
    return {
        'value': value,
        'sha256': fingerprint['sha256'],
        'bytes': fingerprint['bytes'],
    }


def _apply_db_part(
    part_data,
    message,
    *,
    truncations,
    omitted_reasoning,
    snapshots,
    message_index,
    part_id,
    text_limit,
    observation_limit,
    artifact_limit,
    tool_output_dir=None,
):
    if not isinstance(part_data, dict):
        return 0
    part_type = part_data.get('type')
    if part_type in {'step-start', 'step-finish'} and part_data.get('snapshot'):
        snapshots.add(str(part_data['snapshot']))
    if part_type == 'reasoning':
        # Do not copy hidden reasoning into the derived record.  The count is
        # emitted as source metadata so coverage remains auditable.
        omitted_reasoning.append(message_index)
        return 1

    state = part_data.get('state')
    if not isinstance(state, dict):
        state = {}
    call_id = part_data.get('callID', part_data.get('id'))
    tool_name = part_data.get('tool', part_data.get('name'))

    if part_type in {'tool', 'tool-call'}:
        tool_call = {
            'id': call_id,
            'name': tool_name,
            'input': _bounded_payload(
                state.get('input', part_data.get('input')),
                max_chars=observation_limit,
                truncations=truncations,
                kind='tool_input',
                message_index=message_index,
                part_id=part_id,
                call_id=call_id,
            ),
        }
        message.setdefault('tool_calls', []).append(tool_call)
        external_output = _external_tool_output(
            tool_output_dir,
            state.get('metadata'),
        )
        if 'output' in state or 'error' in state or external_output:
            output = state.get('output', state.get('error'))
            status = state.get('status', 'unknown')
            if state.get('error') is not None:
                status = 'error'
            result = {
                'tool_call_id': call_id,
                'tool': tool_name,
                'status': status,
                'output': _bounded_payload(
                    external_output['value'] if external_output else output,
                    max_chars=observation_limit,
                    truncations=truncations,
                    kind='tool_observation',
                    message_index=message_index,
                    part_id=part_id,
                    call_id=call_id,
                ),
            }
            if external_output:
                result['output_source'] = 'external_tool_output'
                result['output_file_sha256'] = external_output['sha256']
                result['output_file_bytes'] = external_output['bytes']
            else:
                result['output_source'] = 'message_part_state'
            message.setdefault('tool_results', []).append(result)
        return 0

    if part_type == 'tool-result':
        result_call_id = part_data.get('toolCallID', part_data.get('callID'))
        external_output = _external_tool_output(
            tool_output_dir,
            part_data.get('metadata'),
        )
        message.setdefault('tool_results', []).append({
            'tool_call_id': result_call_id,
            'status': part_data.get('status', 'unknown'),
            'output': _bounded_payload(
                external_output['value'] if external_output else part_data.get(
                    'output', part_data.get('text')
                ),
                max_chars=observation_limit,
                truncations=truncations,
                kind='tool_observation',
                message_index=message_index,
                part_id=part_id,
                call_id=result_call_id,
            ),
            **({
                'output_source': 'external_tool_output',
                'output_file_sha256': external_output['sha256'],
                'output_file_bytes': external_output['bytes'],
            } if external_output else {'output_source': 'part_state'}),
        })
        return 0

    if part_type == 'patch':
        patch = {
            'hash': part_data.get('hash'),
            'files': part_data.get('files', []),
        }
        message.setdefault('diffs', []).append(
            _bounded_payload(
                patch,
                max_chars=artifact_limit,
                truncations=truncations,
                kind='patch_artifact',
                message_index=message_index,
                part_id=part_id,
            )
        )
        return 0

    if part_type in {'text', 'code'}:
        text = part_data.get('text', '')
        if part_type == 'code':
            text = f"```{part_data.get('language', '')}\n{text}\n```"
        if text:
            message.setdefault('_content_parts', []).append(
                _bounded_payload(
                    text,
                    max_chars=text_limit,
                    truncations=truncations,
                    kind='message_text',
                    message_index=message_index,
                    part_id=part_id,
                )
            )
        return 0

    text = part_data.get('text')
    if text and part_type not in {'step-start', 'step-finish', 'compaction'}:
        message.setdefault('_content_parts', []).append(
            _bounded_payload(
                text,
                max_chars=text_limit,
                truncations=truncations,
                kind='message_text',
                message_index=message_index,
                part_id=part_id,
            )
        )
    return 0


def _new_db_message(row, message_index):
    role = row['role'] or 'assistant'
    message = {
        'role': role,
        'content': '',
        'timestamp': row['message_time'] or row['time_created'],
        '_content_parts': [],
    }
    fields = {
        'modelID': row['model_id'],
        'providerID': row['provider_id'],
        'mode': row['mode'],
        'agent': row['agent'],
        'path': row['path'],
        'cost': row['cost'],
        'tokens': _decode_sql_json(row['tokens_json']),
        'variant': row['variant'],
        'finish': row['finish'],
        'error': _decode_sql_json(row['error_json']),
        'model': _decode_sql_json(row['model_json']),
        'parent_id': row['parent_id'],
    }
    for key, value in fields.items():
        if value is not None:
            message[key] = value
    message['_source_message_index'] = message_index
    return message


def _finalize_db_message(message, *, truncations, text_limit, message_index):
    content_parts = message.pop('_content_parts', [])
    content = '\n'.join(part for part in content_parts if isinstance(part, str))
    if content:
        content = _bounded_payload(
            content,
            max_chars=text_limit,
            truncations=truncations,
            kind='message_text_combined',
            message_index=message_index,
        )
    message['content'] = content
    message.pop('_source_message_index', None)
    return message


def _iter_db_messages(
    conn,
    session_id,
    *,
    truncations,
    omitted_reasoning,
    snapshots,
    text_limit,
    observation_limit,
    artifact_limit,
    tool_output_dir=None,
):
    query = '''
        SELECT
            m.id AS message_id,
            m.time_created AS time_created,
            m.time_updated AS time_updated,
            json_extract(m.data, '$.role') AS role,
            json_extract(m.data, '$.time.created') AS message_time,
            json_extract(m.data, '$.modelID') AS model_id,
            json_extract(m.data, '$.providerID') AS provider_id,
            json_extract(m.data, '$.mode') AS mode,
            json_extract(m.data, '$.agent') AS agent,
            json_extract(m.data, '$.path') AS path,
            json_extract(m.data, '$.cost') AS cost,
            json_extract(m.data, '$.tokens') AS tokens_json,
            json_extract(m.data, '$.variant') AS variant,
            json_extract(m.data, '$.finish') AS finish,
            json_extract(m.data, '$.error') AS error_json,
            json_extract(m.data, '$.model') AS model_json,
            json_extract(m.data, '$.parentID') AS parent_id,
            p.id AS part_id,
            p.data AS part_data
        FROM message AS m
        LEFT JOIN part AS p ON p.message_id = m.id
        WHERE m.session_id = ?
        ORDER BY m.time_created, m.id, p.time_created, p.id
    '''
    current_id = None
    current = None
    message_index = -1
    for row in conn.execute(query, (session_id,)):
        if row['message_id'] != current_id:
            if current is not None:
                yield _finalize_db_message(
                    current,
                    truncations=truncations,
                    text_limit=text_limit,
                    message_index=message_index,
                )
            current_id = row['message_id']
            message_index += 1
            current = _new_db_message(row, message_index)
        if row['part_data']:
            try:
                part_data = json.loads(row['part_data'])
            except (TypeError, json.JSONDecodeError):
                continue
            _apply_db_part(
                part_data,
                current,
                truncations=truncations,
                omitted_reasoning=omitted_reasoning,
                snapshots=snapshots,
                message_index=message_index,
                part_id=row['part_id'],
                text_limit=text_limit,
                observation_limit=observation_limit,
                artifact_limit=artifact_limit,
                tool_output_dir=tool_output_dir,
            )
    if current is not None:
        yield _finalize_db_message(
            current,
            truncations=truncations,
            text_limit=text_limit,
            message_index=message_index,
        )


def _iter_opencode_chunk_groups(messages, *, target):
    """Yield bounded message groups without retaining a whole session."""

    current = []
    current_size = 0
    current_has_user = False
    current_has_assistant = False
    latest_user = None

    def flush(anchor_index=None):
        nonlocal current, current_size, current_has_user, current_has_assistant
        if not current:
            return None
        actual_indexes = [
            index for index, _message in current if index is not None
        ]
        start = min(actual_indexes) if actual_indexes else 0
        end = max(actual_indexes) if actual_indexes else start
        group = (
            [message for _index, message in current],
            start,
            end,
            anchor_index,
        )
        current = []
        current_size = 0
        current_has_user = False
        current_has_assistant = False
        return group

    for source_index, message in enumerate(messages):
        role = message.get('role')
        message_size = _json_size(message) + 1
        should_cut = (
            current
            and current_has_user
            and current_has_assistant
            and (
                current_size + message_size > target
                or (role == 'user' and current_size >= int(target * 0.7))
            )
        )
        if should_cut:
            if role == 'user':
                group = flush()
                if group is not None:
                    yield group
            else:
                anchor = latest_user[0] if latest_user is not None else None
                group = flush(anchor)
                if group is not None:
                    yield group
                if latest_user is not None:
                    anchor_message = dict(latest_user[1])
                    current = [(latest_user[0], anchor_message)]
                    current_size = _json_size(anchor_message) + 1
                    current_has_user = True
                else:
                    current = []
                current_has_assistant = False
        current.append((source_index, message))
        current_size += message_size
        current_has_user = current_has_user or role == 'user'
        current_has_assistant = current_has_assistant or role == 'assistant'
        if role == 'user':
            latest_user = (source_index, message)
    group = flush()
    if group is not None:
        yield group


def _opencode_parent_hash(base_record):
    """Hash session identity/provenance without hashing its message payload."""

    return _json_digest({
        key: value
        for key, value in base_record.items()
        if key != 'messages'
    })


def _opencode_chunk_candidate(
    base_record,
    chunk_messages,
    truncations,
    *,
    index,
    chunk_count,
    start,
    end,
    anchor_index,
    parent_hash,
):
    candidate = dict(base_record)
    candidate['messages'] = chunk_messages
    candidate['observation_truncations'] = [
        item for item in truncations
        if start <= item.get('source_message_index', start) <= end
    ]
    candidate['_chunk_parent_record_sha256'] = parent_hash
    candidate['_chunk_index'] = index
    candidate['_chunk_count'] = chunk_count
    candidate['_chunk_message_start'] = start
    candidate['_chunk_message_end'] = end
    candidate['_chunk_cut_reason'] = 'opencode_ingress_record_budget'
    if anchor_index is not None:
        candidate['_chunk_anchor_message_index'] = anchor_index
    candidate['source_origin'] = dict(base_record.get('source_origin', {}))
    candidate['source_origin']['message_index_range'] = {
        'start': start,
        'end': end,
    }
    return candidate


def _iter_opencode_chunks(
    base_record,
    messages,
    truncations,
    *,
    max_record_chars,
    chunk_count,
):
    target, _text_limit, _observation_limit, _artifact_limit = _opencode_payload_limits(
        max_record_chars
    )
    parent_hash = _opencode_parent_hash(base_record)
    for index, (chunk_messages, start, end, anchor_index) in enumerate(
        _iter_opencode_chunk_groups(messages, target=target)
    ):
        yield _opencode_chunk_candidate(
            base_record,
            chunk_messages,
            truncations,
            index=index,
            chunk_count=chunk_count,
            start=start,
            end=end,
            anchor_index=anchor_index,
            parent_hash=parent_hash,
        )


def _opencode_chunks(base_record, messages, truncations, *, max_record_chars):
    """Compatibility wrapper for callers that explicitly provide a list."""

    target, _text_limit, _observation_limit, _artifact_limit = _opencode_payload_limits(
        max_record_chars
    )
    groups = list(_iter_opencode_chunk_groups(messages, target=target))
    parent_hash = _opencode_parent_hash(base_record)
    return [
        _opencode_chunk_candidate(
            base_record,
            chunk_messages,
            truncations,
            index=index,
            chunk_count=len(groups),
            start=start,
            end=end,
            anchor_index=anchor_index,
            parent_hash=parent_hash,
        )
        for index, (chunk_messages, start, end, anchor_index) in enumerate(groups)
    ]


def iter_cli_conversations_db(
    install_dir,
    *,
    max_record_chars=OPENCODE_MAX_RECORD_CHARS,
    source_manifest: SourceManifestIndex | None = None,
    source_admission: Mapping[str, Any] | None = None,
    root_label='cli',
    immutable_snapshot=False,
):
    """Stream bounded SQLite sessions without loading the database into RAM.

    OpenCode keeps message summaries and patches in the same SQLite rows.  A
    row can therefore be hundreds of megabytes even when its visible turn is
    tiny.  The query projects only message metadata, joins parts once per
    session, omits reasoning, bounds payloads, and emits linked chunks.
    """
    db_path = install_dir / 'opencode.db'
    if not db_path.exists():
        return

    if source_manifest is not None and source_admission is not None:
        raise SourceAdmissionError(
            'provide source_manifest or source_admission, not both'
        )
    source_origin = _sqlite_source_origin(db_path)
    database_fingerprint = source_origin.get('database') or {}
    source_sha256 = database_fingerprint.get('sha256')
    if source_manifest is not None:
        if not database_fingerprint.get('hash_stable'):
            raise SourceAdmissionError(
                'cannot admit an OpenCode database whose fingerprint is unstable'
            )
        source_admission = source_manifest.admit(
            source_sha256=source_sha256,
            provider='opencode',
            root_label=root_label,
            source_class='conversation_store',
        )
    if source_admission is not None:
        if source_sha256 is None or database_fingerprint.get('bytes') is None:
            raise SourceAdmissionError(
                'OpenCode database fingerprint is incomplete for source admission'
            )
        source_admission = validate_source_admission(
            source_admission,
            source_sha256=source_sha256,
            source_bytes=database_fingerprint['bytes'],
        )
        source_origin.update({
            'source_manifest_revision': source_admission['source_manifest_revision'],
            'source_ref_sha256': source_admission['source_ref_sha256'],
            'source_snapshot_status': source_admission['source_snapshot_status'],
            'source_sha256': source_admission['source_sha256'],
            'source_bytes': source_admission['source_bytes'],
        })
    _target, text_limit, observation_limit, artifact_limit = _opencode_payload_limits(
        max_record_chars
    )
    storage_dir = install_dir
    uri = _sqlite_read_uri(db_path, immutable=immutable_snapshot)
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        session_rows = conn.execute(
            'SELECT id, project_id, parent_id, slug, directory, title, version, '
            'share_url, summary_additions, summary_deletions, summary_files, '
            'revert, permission, time_created, time_updated, time_compacting, '
            'time_archived, workspace_id, agent, model '
            'FROM session ORDER BY time_created, id'
        )
        mode = 'immutable snapshot' if immutable_snapshot else 'WAL-aware read-only'
        print(f'  Streaming sessions from opencode.db ({mode} mode)')
        session_count = 0
        emitted_count = 0
        for session_row in session_rows:
            session_id = session_row['id']
            truncations = []
            omitted_reasoning = []
            snapshots = set()

            def message_stream(
                *,
                stream_truncations,
                stream_omitted_reasoning,
                stream_snapshots,
            ):
                return _iter_db_messages(
                    conn,
                    session_id,
                    truncations=stream_truncations,
                    omitted_reasoning=stream_omitted_reasoning,
                    snapshots=stream_snapshots,
                    text_limit=text_limit,
                    observation_limit=observation_limit,
                    artifact_limit=artifact_limit,
                    tool_output_dir=storage_dir / 'tool-output',
                )

            message_count = 0

            def counted_messages():
                nonlocal message_count
                for message in message_stream(
                    stream_truncations=truncations,
                    stream_omitted_reasoning=omitted_reasoning,
                    stream_snapshots=snapshots,
                ):
                    message_count += 1
                    yield message

            chunk_count = sum(
                1 for _ in _iter_opencode_chunk_groups(
                    counted_messages(),
                    target=_target,
                )
            )
            session_count += 1
            if message_count == 0:
                continue
            origin = dict(source_origin)
            origin.update({
                'session_id_sha256': hashlib.sha256(
                    str(session_id).encode('utf-8')
                ).hexdigest(),
                'message_count': message_count,
                'reasoning_parts_omitted': len(omitted_reasoning),
                'snapshot_ref_count': len(snapshots),
                'snapshot_refs_sha256': (
                    _json_digest(sorted(snapshots)) if snapshots else None
                ),
                'sidecars': (
                    _sidecar_fingerprints(storage_dir, session_id)
                    + _session_sidecar_fingerprints(storage_dir, session_id)
                ),
            })
            base_record = {
                'source': 'opencode-cli',
                'source_class': 'conversation_database',
                'source_origin': origin,
                'session_id': session_id,
                'project_id': session_row['project_id'],
                'parent_session_id': session_row['parent_id'],
                'slug': session_row['slug'],
                'directory': session_row['directory'],
                'title': session_row['title'],
                'version': session_row['version'],
                'share_url': session_row['share_url'],
                'created_at': session_row['time_created'],
                'updated_at': session_row['time_updated'],
                'time_archived': session_row['time_archived'],
                'workspace_id': session_row['workspace_id'],
                'agent': session_row['agent'],
                'model': _decode_sql_json(session_row['model']),
                'summary': {
                    'additions': session_row['summary_additions'],
                    'deletions': session_row['summary_deletions'],
                    'files': session_row['summary_files'],
                },
            }
            for candidate in _iter_opencode_chunks(
                base_record,
                message_stream(
                    stream_truncations=None,
                    stream_omitted_reasoning=[],
                    stream_snapshots=set(),
                ),
                truncations,
                max_record_chars=max_record_chars,
                chunk_count=chunk_count,
            ):
                emitted_count += 1
                yield candidate
        print(
            f'  SQLite sessions inspected: {session_count}; '
            f'bounded records emitted: {emitted_count}'
        )
    finally:
        conn.close()


def extract_cli_conversations_db(install_dir, **kwargs):
    """Compatibility wrapper for callers that still expect a list."""
    return list(iter_cli_conversations_db(install_dir, **kwargs))


def _iter_json_session_messages(
    storage_dir,
    session_id,
    message_files,
    *,
    truncations,
    omitted_reasoning,
    snapshots,
    text_limit,
    observation_limit,
    artifact_limit,
    source_manifest=None,
    root_label='cli',
    source_refs=None,
    tool_output_index=None,
):
    """Yield one bounded JSON-store message at a time."""

    part_dir = storage_dir / 'storage' / 'part'
    for message_index, message_file in enumerate(message_files):
        message_data, admission = _read_json_source(
            message_file,
            storage_dir=storage_dir,
            source_manifest=source_manifest,
            root_label=root_label,
            source_class='conversation_store',
        )
        if admission is not None and source_refs is not None:
            source_refs.add(admission['source_ref_sha256'])
        if not isinstance(message_data, dict):
            continue

        time_data = message_data.get('time', {})
        timestamp = time_data.get('created') if isinstance(time_data, dict) else None
        message = {
            'role': message_data.get('role', 'assistant'),
            'content': '',
            'timestamp': timestamp,
            '_content_parts': [],
        }
        fields = {
            'model': message_data.get('modelID'),
            'provider': message_data.get('providerID'),
            'mode': message_data.get('mode'),
            'agent': message_data.get('agent'),
            'path': message_data.get('path'),
            'cost': message_data.get('cost'),
            'tokens': message_data.get('tokens'),
            'variant': message_data.get('variant'),
            'finish': message_data.get('finish'),
            'error': message_data.get('error'),
            'parent_id': message_data.get('parentID'),
        }
        for key, value in fields.items():
            if value is not None:
                message[key] = value
        message['_source_message_index'] = message_index

        message_id = message_data.get('id') or message_file.stem
        message_part_dir = part_dir / str(message_id)
        if message_part_dir.exists():
            for part_file in sorted(message_part_dir.glob('*.json')):
                part_data, part_admission = _read_json_source(
                    part_file,
                    storage_dir=storage_dir,
                    source_manifest=source_manifest,
                    root_label=root_label,
                    source_class='conversation_store',
                )
                if part_admission is not None and source_refs is not None:
                    source_refs.add(part_admission['source_ref_sha256'])
                part_id = (
                    part_data.get('id', part_file.stem)
                    if isinstance(part_data, dict)
                    else part_file.stem
                )
                _apply_db_part(
                    part_data,
                    message,
                    truncations=truncations,
                    omitted_reasoning=omitted_reasoning,
                    snapshots=snapshots,
                    message_index=message_index,
                    part_id=part_id,
                    text_limit=text_limit,
                    observation_limit=observation_limit,
                    artifact_limit=artifact_limit,
                    tool_output_dir=storage_dir / 'tool-output',
                )

        message = _finalize_db_message(
            message,
            truncations=truncations,
            text_limit=text_limit,
            message_index=message_index,
        )
        if source_refs is not None and tool_output_index:
            for result in message.get('tool_results', []):
                output_sha256 = result.get('output_file_sha256')
                metadata = tool_output_index.get(output_sha256)
                if metadata is not None:
                    admission = metadata.get('admission')
                    if admission is not None:
                        source_refs.add(admission['source_ref_sha256'])
        yield message


def _iter_json_session_specs(storage_dir):
    message_root = storage_dir / 'storage' / 'message'
    if not message_root.exists():
        return
    for session_dir in sorted(message_root.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith('ses_'):
            continue
        message_files = sorted(session_dir.glob('*.json'))
        if message_files:
            yield session_dir.name, message_files


def _json_session_metadata(session_data, messages):
    first_message_time = None
    last_message_time = None
    directory = None
    project_id = None
    title = None
    for message in messages:
        timestamp = message.get('timestamp')
        if timestamp is not None:
            if first_message_time is None or timestamp < first_message_time:
                first_message_time = timestamp
            if last_message_time is None or timestamp > last_message_time:
                last_message_time = timestamp
        content = message.get('content')
        if not isinstance(content, str):
            continue
        directory = directory or extract_directory_from_content(content)
        project_id = project_id or extract_project_id_from_content(content)
        if title is None and message.get('role') == 'user' and content:
            title = content[:100].strip()
            if len(content) > 100:
                title += '...'

    if isinstance(session_data, dict):
        time_data = session_data.get('time', {})
        if not isinstance(time_data, dict):
            time_data = {}
        return {
            'title': session_data.get('title'),
            'created_at': time_data.get('created', first_message_time),
            'updated_at': time_data.get('updated', last_message_time),
            'project_id': session_data.get('projectID', project_id),
            'directory': session_data.get('directory', directory),
            'version': session_data.get('version', 'unknown'),
            'summary': session_data.get('summary'),
            'parent_session_id': session_data.get('parentID'),
        }
    return {
        'title': title,
        'created_at': first_message_time,
        'updated_at': last_message_time,
        'project_id': project_id,
        'directory': directory,
        'version': 'unknown',
    }


def iter_cli_conversations_json(
    storage_dir,
    *,
    max_record_chars=OPENCODE_MAX_RECORD_CHARS,
    source_manifest: SourceManifestIndex | None = None,
    root_label='cli',
):
    """Stream disjoint OpenCode JSON sessions through the bounded adapter."""

    message_root = storage_dir / 'storage' / 'message'
    if not message_root.exists():
        return

    _target, text_limit, observation_limit, artifact_limit = _opencode_payload_limits(
        max_record_chars
    )
    sidecars = _json_sidecar_index(
        storage_dir,
        source_manifest=source_manifest,
        root_label=root_label,
    )
    tool_output_index_by_name = _json_tool_output_index(
        storage_dir,
        source_manifest=source_manifest,
        root_label=root_label,
    )
    tool_output_index = {
        metadata['sha256']: metadata
        for metadata in tool_output_index_by_name.values()
    }
    session_count = 0
    emitted_count = 0
    for session_id, message_files in _iter_json_session_specs(storage_dir):
        session_count += 1
        truncations = []
        omitted_reasoning = []
        snapshots = set()
        source_refs = set()
        metadata_entry = sidecars.get(
            session_id,
            {'metadata': None, 'source_refs': set(), 'sidecar_count': 0},
        )
        source_refs.update(metadata_entry['source_refs'])

        def first_pass_messages():
            yield from _iter_json_session_messages(
                storage_dir,
                session_id,
                message_files,
                truncations=truncations,
                omitted_reasoning=omitted_reasoning,
                snapshots=snapshots,
                text_limit=text_limit,
                observation_limit=observation_limit,
                artifact_limit=artifact_limit,
                source_manifest=source_manifest,
                root_label=root_label,
                source_refs=source_refs,
                tool_output_index=tool_output_index,
            )

        message_count = 0
        chunk_count = 0
        first_message_time = None
        last_message_time = None
        fallback_directory = None
        fallback_project_id = None
        fallback_title = None

        def inspected_messages():
            nonlocal first_message_time, last_message_time
            nonlocal fallback_directory, fallback_project_id, fallback_title
            nonlocal message_count
            for message in first_pass_messages():
                message_count += 1
                timestamp = message.get('timestamp')
                if timestamp is not None:
                    if first_message_time is None or timestamp < first_message_time:
                        first_message_time = timestamp
                    if last_message_time is None or timestamp > last_message_time:
                        last_message_time = timestamp
                content = message.get('content')
                if isinstance(content, str):
                    fallback_directory = (
                        fallback_directory or extract_directory_from_content(content)
                    )
                    fallback_project_id = (
                        fallback_project_id or extract_project_id_from_content(content)
                    )
                    if fallback_title is None and message.get('role') == 'user' and content:
                        fallback_title = content[:100].strip()
                        if len(content) > 100:
                            fallback_title += '...'
                yield message

        for _group in _iter_opencode_chunk_groups(
            inspected_messages(),
            target=_target,
        ):
            chunk_count += 1
        if message_count == 0:
            continue

        session_data = metadata_entry.get('metadata')
        metadata = _json_session_metadata(
            session_data,
            (),
        )
        if not isinstance(session_data, dict):
            metadata.update({
                'created_at': first_message_time,
                'updated_at': last_message_time,
            })
        metadata.update({
            'created_at': metadata.get('created_at') or first_message_time,
            'updated_at': metadata.get('updated_at') or last_message_time,
            'directory': metadata.get('directory') or fallback_directory,
            'project_id': metadata.get('project_id') or fallback_project_id,
            'title': metadata.get('title') or fallback_title,
        })
        origin = {
            'store_type': 'json_files',
            'message_root': 'storage/message',
            'part_root': 'storage/part',
            'session_metadata_root': 'storage/session',
            'session_id_sha256': hashlib.sha256(
                str(session_id).encode('utf-8')
            ).hexdigest(),
            'message_count': message_count,
            'reasoning_parts_omitted': len(omitted_reasoning),
            'snapshot_ref_count': len(snapshots),
            'snapshot_refs_sha256': (
                _json_digest(sorted(snapshots)) if snapshots else None
            ),
            'sidecar_count': metadata_entry['sidecar_count'],
            'content_policy': 'reasoning_omitted; large_payloads_bounded',
        }
        if source_manifest is not None:
            origin.update({
                'source_manifest_revision': source_manifest.revision,
                'source_snapshot_status': 'bound',
                'source_file_count': len(source_refs),
                'source_ref_set_sha256': _json_digest(sorted(source_refs)),
            })
        base_record = {
            'source': 'opencode-cli',
            'source_class': 'conversation_storage',
            'source_origin': origin,
            'session_id': session_id,
            'title': metadata.get('title'),
            'created_at': metadata.get('created_at'),
            'updated_at': metadata.get('updated_at'),
            'project_id': metadata.get('project_id'),
            'directory': metadata.get('directory'),
            'version': metadata.get('version'),
        }
        if metadata.get('summary') is not None:
            base_record['summary'] = metadata['summary']
        if metadata.get('parent_session_id') is not None:
            base_record['parent_session_id'] = metadata['parent_session_id']

        def second_pass_messages():
            yield from _iter_json_session_messages(
                storage_dir,
                session_id,
                message_files,
                truncations=None,
                omitted_reasoning=[],
                snapshots=set(),
                text_limit=text_limit,
                observation_limit=observation_limit,
                artifact_limit=artifact_limit,
                source_manifest=None,
                root_label=root_label,
                source_refs=None,
                tool_output_index=tool_output_index,
            )

        for candidate in _iter_opencode_chunks(
            base_record,
            second_pass_messages(),
            truncations,
            max_record_chars=max_record_chars,
            chunk_count=chunk_count,
        ):
            emitted_count += 1
            yield candidate
    print(
        f'  JSON sessions inspected: {session_count}; '
        f'bounded records emitted: {emitted_count}'
    )


def _iter_payload_groups(payloads, *, target):
    """Yield bounded artifact groups without retaining an unbounded group."""

    current = []
    current_size = 0
    start = 0
    for index, payload in enumerate(payloads):
        payload_size = _json_size(payload) + 1
        if current and current_size + payload_size > target:
            yield current, start, index - 1
            current = []
            current_size = 0
            start = index
        current.append(payload)
        current_size += payload_size
    if current:
        yield current, start, start + len(current) - 1


def iter_cli_session_diff_records(
    storage_dir,
    *,
    max_record_chars=OPENCODE_MAX_RECORD_CHARS,
    source_manifest: SourceManifestIndex | None = None,
    root_label='cli',
):
    """Emit bounded, reasoning-free session-diff artifacts as non-chat records."""

    root = storage_dir / 'storage' / 'session_diff'
    if not root.exists():
        return
    target, _text_limit, _observation_limit, artifact_limit = _opencode_payload_limits(
        max_record_chars
    )
    file_count = 0
    emitted_count = 0
    for path in sorted(root.rglob('*.json')):
        file_count += 1
        value, admission = _read_json_source(
            path,
            storage_dir=storage_dir,
            source_manifest=source_manifest,
            root_label=root_label,
            source_class='conversation_sidecar',
        )
        if isinstance(value, list):
            raw_artifacts = value
        else:
            raw_artifacts = [value]
        truncations = []
        bounded_artifacts = []
        for artifact_index, artifact in enumerate(raw_artifacts):
            if artifact is None:
                continue
            bounded_artifacts.append(
                _bounded_payload(
                    artifact,
                    max_chars=artifact_limit,
                    truncations=truncations,
                    kind='session_diff_artifact',
                    message_index=artifact_index,
                )
            )
        if not bounded_artifacts:
            bounded_groups = [([], 0, 0)]
        else:
            bounded_groups = list(
                _iter_payload_groups(bounded_artifacts, target=target)
            )
        parent_hash = _json_digest({
            'source': 'opencode-cli',
            'source_class': 'conversation_sidecar',
            'session_id': path.stem,
            'source_ref_sha256': (
                admission['source_ref_sha256'] if admission else None
            ),
        })
        for chunk_index, (artifacts, start, end) in enumerate(bounded_groups):
            source_origin = {
                'store_type': 'json_session_diff',
                'session_id_sha256': hashlib.sha256(
                    path.stem.encode('utf-8')
                ).hexdigest(),
                'artifact_count': len(raw_artifacts),
                'content_policy': 'reasoning_omitted; large_payloads_bounded',
                'artifact_index_range': {'start': start, 'end': end},
            }
            if admission is not None:
                source_origin.update({
                    'source_manifest_revision': admission['source_manifest_revision'],
                    'source_ref_sha256': admission['source_ref_sha256'],
                    'source_snapshot_status': admission['source_snapshot_status'],
                    'source_sha256': admission['source_sha256'],
                    'source_bytes': admission['source_bytes'],
                })
            record = {
                'source': 'opencode-cli',
                'source_class': 'conversation_sidecar',
                'source_origin': source_origin,
                'session_id': path.stem,
                'messages': [],
                'diffs': artifacts,
                'observation_truncations': [
                    item for item in truncations
                    if start <= item.get('source_message_index', start) <= end
                ],
                '_chunk_parent_record_sha256': parent_hash,
                '_chunk_index': chunk_index,
                '_chunk_count': len(bounded_groups),
                '_chunk_message_start': start,
                '_chunk_message_end': end,
                '_chunk_cut_reason': 'opencode_session_diff_artifact_budget',
            }
            emitted_count += 1
            yield record
    print(
        f'  Session-diff files inspected: {file_count}; '
        f'bounded artifact records emitted: {emitted_count}'
    )


def extract_cli_conversations(storage_dir):
    """
    Extract conversations from CLI JSON storage.
    
    Handles sessions both WITH and WITHOUT session metadata files.
    For sessions without metadata, reconstructs session info from messages/parts.
    """
    conversations = []
    
    message_dir = storage_dir / 'storage' / 'message'
    part_dir = storage_dir / 'storage' / 'part'
    
    if not message_dir.exists():
        db_conversations = extract_cli_conversations_db(storage_dir)
        if db_conversations:
            return db_conversations
        print(f"  Message directory not found: {message_dir}")
        return conversations
    
    # Find all session directories (each is a directory named ses_xxx)
    session_dirs = [d for d in message_dir.iterdir() if d.is_dir() and d.name.startswith('ses_')]
    
    print(f"  Found {len(session_dirs)} session directories")
    
    processed_sessions = set()
    
    for session_dir_path in session_dirs:
        try:
            session_id = session_dir_path.name
            
            # Skip if already processed (deduplication)
            if session_id in processed_sessions:
                continue
            processed_sessions.add(session_id)
            
            # Try to load session metadata if available
            session_data = None
            session_file = storage_dir / 'storage' / 'session' / 'global' / f'{session_id}.json'
            
            if session_file.exists():
                with open(session_file) as f:
                    session_data = json.load(f)
            
            # Collect all messages for this session
            message_files = sorted(session_dir_path.glob('msg_*.json'))
            
            if not message_files:
                continue
            
            messages = []
            all_content = []  # For reconstructing metadata
            first_message_time = None
            last_message_time = None
            
            for msg_file in message_files:
                try:
                    with open(msg_file) as f:
                        msg_data = json.load(f)
                    
                    message_id = msg_data.get('id')
                    role = msg_data.get('role', 'assistant')
                    msg_time = msg_data.get('time', {}).get('created')
                    
                    # Track timestamps
                    if msg_time:
                        if not first_message_time or msg_time < first_message_time:
                            first_message_time = msg_time
                        if not last_message_time or msg_time > last_message_time:
                            last_message_time = msg_time
                    
                    # Build the message
                    message = {
                        'role': role,
                        'content': '',
                        'timestamp': msg_time
                    }
                    
                    # Add metadata
                    if 'modelID' in msg_data:
                        message['model'] = msg_data['modelID']
                    if 'providerID' in msg_data:
                        message['provider'] = msg_data['providerID']
                    if 'agent' in msg_data:
                        message['agent'] = msg_data['agent']
                    if 'mode' in msg_data:
                        message['mode'] = msg_data['mode']
                    
                    # Add token usage
                    if 'tokens' in msg_data:
                        message['tokens'] = msg_data['tokens']
                    if 'cost' in msg_data:
                        message['cost'] = msg_data['cost']
                    
                    # Find all parts for this message
                    message_part_dir = part_dir / message_id
                    
                    if message_part_dir.exists():
                        part_files = sorted(message_part_dir.glob('prt_*.json'))
                        content_parts = []
                        tool_calls = []
                        tool_results = []
                        reasoning_parts = []
                        
                        for part_file in part_files:
                            try:
                                with open(part_file) as f:
                                    part_data = json.load(f)
                                
                                part_type = part_data.get('type')
                                part_text = part_data.get('text', '')
                                
                                # Collect content for metadata reconstruction
                                if part_text:
                                    all_content.append(part_text)
                                
                                if part_type == 'text':
                                    content_parts.append(part_text)
                                elif part_type == 'tool' or part_type == 'tool-call':
                                    # OpenCode uses 'tool' type with state containing input/output
                                    state = part_data.get('state', {})
                                    tool_name = part_data.get('tool', part_data.get('name'))
                                    
                                    tool_call = {
                                        'id': part_data.get('callID', part_data.get('id')),
                                        'name': tool_name,
                                        'input': state.get('input', part_data.get('input'))
                                    }

                                    # If completed, also add to tool_results
                                    external_output = _external_tool_output(
                                        storage_dir / 'tool-output',
                                        state.get('metadata'),
                                    )
                                    if state.get('status') == 'completed' and (
                                        'output' in state or external_output
                                    ):
                                        result = {
                                            'tool_call_id': part_data.get('callID'),
                                            'tool': tool_name,
                                            'output': external_output['value'] if external_output else state.get('output'),
                                        }
                                        if external_output:
                                            result['output_source'] = 'external_tool_output'
                                            result['output_file_sha256'] = external_output['sha256']
                                            result['output_file_bytes'] = external_output['bytes']
                                        else:
                                            result['output_source'] = 'message_part_state'
                                        tool_results.append(result)
                                    
                                    tool_calls.append(tool_call)
                                elif part_type == 'tool-result':
                                    external_output = _external_tool_output(
                                        storage_dir / 'tool-output',
                                        part_data.get('metadata'),
                                    )
                                    result = {
                                        'tool_call_id': part_data.get('toolCallID'),
                                        'output': external_output['value'] if external_output else part_data.get('output'),
                                    }
                                    if external_output:
                                        result['output_source'] = 'external_tool_output'
                                        result['output_file_sha256'] = external_output['sha256']
                                        result['output_file_bytes'] = external_output['bytes']
                                    else:
                                        result['output_source'] = 'part_state'
                                    tool_results.append(result)
                                elif part_type == 'code':
                                    # Code blocks
                                    code_text = part_data.get('text', '')
                                    language = part_data.get('language', '')
                                    content_parts.append(f"```{language}\n{code_text}\n```")
                                elif part_type == 'reasoning':
                                    # Reasoning is available in the original
                                    # store but is intentionally not copied to
                                    # a derived export.
                                    continue
                                
                            except Exception as e:
                                print(f"    Error reading part {part_file}: {e}")
                                continue
                        
                        message['content'] = '\n'.join(content_parts)
                        
                        if tool_calls:
                            message['tool_calls'] = tool_calls
                        if tool_results:
                            message['tool_results'] = tool_results
                        if reasoning_parts:
                            message['reasoning'] = '\n'.join(reasoning_parts)
                    
                    messages.append(message)
                
                except Exception as e:
                    print(f"    Error reading message {msg_file}: {e}")
                    continue
            
            if not messages:
                continue
            
            # Build conversation - use session data if available, otherwise reconstruct
            combined_content = '\n'.join(all_content)
            
            conversation = {
                'messages': messages,
                'source': 'opencode-cli',
                'session_id': session_id,
            }
            
            if session_data:
                # Use metadata from session file
                conversation['title'] = session_data.get('title')
                conversation['created_at'] = session_data.get('time', {}).get('created')
                conversation['updated_at'] = session_data.get('time', {}).get('updated')
                conversation['project_id'] = session_data.get('projectID')
                conversation['directory'] = session_data.get('directory')
                conversation['version'] = session_data.get('version')
                
                # Add summary stats if available
                if 'summary' in session_data:
                    conversation['summary'] = session_data['summary']
                
                # Add parent session if it's a child session
                if 'parentID' in session_data:
                    conversation['parent_session_id'] = session_data['parentID']
            else:
                # RECONSTRUCT metadata from messages/parts
                conversation['created_at'] = first_message_time
                conversation['updated_at'] = last_message_time
                
                # Try to extract directory from content
                conversation['directory'] = extract_directory_from_content(combined_content)
                
                # Try to extract project ID from content
                conversation['project_id'] = extract_project_id_from_content(combined_content)
                
                # Generate a title from first user message
                for msg in messages:
                    if msg.get('role') == 'user' and msg.get('content'):
                        # Take first 100 chars of first user message as title
                        title = msg['content'][:100].strip()
                        if len(msg['content']) > 100:
                            title += '...'
                        conversation['title'] = title
                        break
                
                # Set default version
                conversation['version'] = 'unknown'
            
            conversations.append(conversation)
        
        except Exception as e:
            print(f"  Error processing session {session_dir_path}: {e}")
            continue
    
    return conversations


def iter_cli_conversations(
    storage_dir,
    *,
    max_record_chars=OPENCODE_MAX_RECORD_CHARS,
    source_manifest: SourceManifestIndex | None = None,
    root_label='cli',
    immutable_snapshot=False,
):
    """Yield every CLI source class present in an installation.

    SQLite and JSON storage are not assumed to be mirrors.  They commonly
    contain disjoint session populations, so both are emitted with explicit
    provenance and allowed to deduplicate only at the canonical boundary.
    """
    db_path = storage_dir / 'opencode.db'
    if db_path.exists():
        yield from iter_cli_conversations_db(
            storage_dir,
            max_record_chars=max_record_chars,
            source_manifest=source_manifest,
            root_label=root_label,
            immutable_snapshot=immutable_snapshot,
        )

    message_dir = storage_dir / 'storage' / 'message'
    if not message_dir.exists():
        if not db_path.exists():
            print(f"  Message directory not found: {message_dir}")
        return

    yield from iter_cli_conversations_json(
        storage_dir,
        max_record_chars=max_record_chars,
        source_manifest=source_manifest,
        root_label=root_label,
    )
    yield from iter_cli_session_diff_records(
        storage_dir,
        max_record_chars=max_record_chars,
        source_manifest=source_manifest,
        root_label=root_label,
    )

def extract_desktop_conversations(desktop_dir):
    """Extract conversations from Desktop Tauri store files"""
    conversations = []
    
    # Look for .dat files
    dat_files = list(desktop_dir.rglob('*.dat'))
    
    if not dat_files:
        return conversations
    
    print(f"  Found {len(dat_files)} .dat store files")
    
    for dat_file in dat_files:
        store = read_tauri_store(dat_file)
        
        if not store:
            continue
        
        # Look for session/conversation data in the store
        # Keys might be like "session:ses_xxxxx" or similar
        for key, value in store.items():
            if not isinstance(value, dict):
                continue
            
            # Check if this looks like a conversation/session
            if 'messages' in value or 'history' in value:
                try:
                    messages = value.get('messages', value.get('history', []))
                    
                    if not messages:
                        continue
                    
                    conversation = {
                        'messages': messages,
                        'source': 'opencode-desktop',
                        'store_key': key,
                        'store_file': str(dat_file.name)
                    }
                    
                    # Add any additional metadata
                    for meta_key in ['session_id', 'title', 'created_at', 'workspace']:
                        if meta_key in value:
                            conversation[meta_key] = value[meta_key]
                    
                    conversations.append(conversation)
                
                except Exception:
                    continue
    
    return conversations

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-home',
        type=Path,
        help='explicit immutable synthetic HOME or live home containing provider roots',
    )
    parser.add_argument(
        '--source-manifest',
        type=Path,
        help='bind the SQLite source to an immutable source-manifest artifact',
    )
    parser.add_argument(
        '--max-record-chars',
        type=int,
        default=OPENCODE_MAX_RECORD_CHARS,
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(os.environ.get('EXTRACTED_DATA_DIR', 'extracted_data')),
    )
    args = parser.parse_args(argv)
    try:
        source_manifest_index = (
            SourceManifestIndex.from_path(args.source_manifest)
            if args.source_manifest is not None
            else None
        )
    except (OSError, SourceAdmissionError, SourceManifestError) as exc:
        print(f'❌ Source manifest failed: {exc}')
        return 2

    print("="*80)
    print("OPENCODE EXTRACTION")
    print("="*80)
    print()
    
    installations = find_opencode_installations(args.source_home)
    
    if not installations:
        print("❌ No OpenCode installations found!")
        print()
        print("Searched locations:")
        print("  CLI: ~/.local/share/opencode (Linux)")
        print("       ~/Library/Application Support/opencode (macOS)")
        print("  Desktop: ~/.local/share/ai.opencode.app (Linux)")
        print("           ~/Library/Application Support/ai.opencode.app (macOS)")
        return
    
    print(f"✅ Found {len(installations)} installation(s)")
    print()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'opencode_conversations_{timestamp}.jsonl'
    temporary_output = None

    total_conversations = 0
    total_messages = 0
    with_tools = 0
    with_models = 0
    with_reasoning = 0
    with_session_file = 0
    emitted_db_sessions = {}
    emitted_storage_sessions = {}

    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=output_dir,
            prefix=f'.{output_file.name}.',
            suffix='.tmp',
            delete=False,
        ) as output:
            temporary_output = Path(output.name)
            for install_type, install_dir in installations:
                print(f"Processing {install_type} installation: {install_dir}")
                if install_type == 'cli':
                    conversations = iter_cli_conversations(
                        install_dir,
                        max_record_chars=args.max_record_chars,
                        source_manifest=source_manifest_index,
                        root_label=install_type,
                        immutable_snapshot=args.source_home is not None,
                    )
                else:  # desktop
                    conversations = iter(extract_desktop_conversations(install_dir))

                db_sessions = emitted_db_sessions.setdefault(install_dir, set())
                storage_sessions = emitted_storage_sessions.setdefault(install_dir, set())

                installation_count = 0
                for conversation in conversations:
                    installation_count += 1
                    total_conversations += 1
                    messages = conversation.get('messages', [])
                    total_messages += len(messages)
                    if any(
                        'tool_calls' in message or 'tool_results' in message
                        for message in messages
                    ):
                        with_tools += 1
                    if any('model' in message for message in messages):
                        with_models += 1
                    if any('reasoning' in message for message in messages):
                        with_reasoning += 1
                    if conversation.get('directory'):
                        with_session_file += 1
                    session_id = conversation.get('session_id')
                    if session_id:
                        if conversation.get('source_class') == 'conversation_database':
                            db_sessions.add(str(session_id))
                        elif conversation.get('source_class') == 'conversation_storage':
                            storage_sessions.add(str(session_id))
                    output.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                print(f"  Extracted {installation_count} conversations")
                print()

        source_manifest_files = []
        for install_type, install_dir in installations:
            if install_type != 'cli':
                continue
            source_manifest = build_source_manifest(
                install_dir,
                emitted_db_session_ids=emitted_db_sessions.get(install_dir, set()),
                emitted_storage_session_ids=emitted_storage_sessions.get(install_dir, set()),
                source_manifest=source_manifest_index,
                root_label=install_type,
                immutable_snapshot=args.source_home is not None,
            )
            source_manifest_file = write_source_manifest(
                output_dir,
                timestamp,
                source_manifest,
            )
            source_manifest_files.append(source_manifest_file)

        if total_conversations == 0:
            print("❌ No conversation data found!")
            return
        os.replace(temporary_output, output_file)
        temporary_output = None
    finally:
        if temporary_output is not None:
            try:
                temporary_output.unlink()
            except FileNotFoundError:
                pass

    print(f"✅ Total bounded conversations extracted: {total_conversations}")
    print(f"Total messages: {total_messages}")
    print(f"With tool use: {with_tools}")
    print(f"With model info: {with_models}")
    print(f"With reasoning in derived output: {with_reasoning}")
    print(f"With directory metadata: {with_session_file}")
    print()
    file_size = output_file.stat().st_size / 1024
    print(f"✅ Saved to: {output_file}")
    print(f"   Size: {file_size:.2f} KB")
    for source_manifest_file in source_manifest_files:
        print(f"✅ Source manifest: {source_manifest_file}")

if __name__ == '__main__':
    main()
	
