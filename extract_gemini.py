#!/usr/bin/env python3
"""
Extract ALL Google Gemini CLI chat data
Includes: messages, thoughts (reasoning), token usage, model info
Auto-discovers Gemini CLI installations on the device
"""

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from inventory_sources import discover_roots
from source_manifest import SourceManifestIndex, validate_source_admission

EXTRACTOR_VERSION = '1.2.0'


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value):
    """Represent a structured tool response without losing its shape."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _message_id(message):
    value = message.get('id') if isinstance(message, dict) else None
    return str(value) if value is not None else None


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _content_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _json_text(content) if content is not None else ''

    text_parts = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get('text') is not None:
            text_parts.append(str(block['text']))
    return ''.join(text_parts)


def _tool_call_id(payload):
    if not isinstance(payload, dict):
        return None
    for key in ('id', 'callId', 'callID', 'call_id', 'toolCallId', 'tool_call_id'):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return None


def _normalize_tool_call(tool_call):
    if not isinstance(tool_call, dict):
        return None
    call_id = _tool_call_id(tool_call)
    name = tool_call.get('name') or tool_call.get('displayName') or 'unknown_tool'
    arguments = tool_call.get('args', tool_call.get('arguments', tool_call.get('input', {})))
    return {
        'id': call_id or str(name),
        'type': 'function',
        'function': {
            'name': str(name),
            'arguments': arguments,
        },
    }


def _normalize_function_response(function_response):
    if not isinstance(function_response, dict):
        return None
    call_id = _tool_call_id(function_response)
    name = function_response.get('name') or 'unknown_tool'
    response = function_response.get('response')
    if response is None:
        response = function_response.get('result', function_response)
    return {
        'role': 'tool',
        'tool_call_id': call_id or str(name),
        'name': str(name),
        'content': _json_text(response),
    }


def _load_gemini_session_data(session_file):
    """Load JSON or event-sourced JSONL without duplicating snapshot messages."""
    parse_errors = 0
    metadata = {}
    fallback_messages = []

    if session_file.suffix.lower() != '.jsonl':
        with open(session_file, 'r', encoding='utf-8') as source:
            data = json.load(source)
        return data, list(data.get('messages') or []), parse_errors

    latest_messages = None
    historical_tool_calls = {}
    with open(session_file, 'r', encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            if not isinstance(event, dict):
                continue
            for key in ('sessionId', 'projectHash', 'startTime', 'lastUpdated'):
                if event.get(key) is not None:
                    metadata[key] = event[key]

            set_data = event.get('$set')
            if isinstance(set_data, dict):
                for key in ('sessionId', 'projectHash', 'startTime', 'lastUpdated'):
                    if set_data.get(key) is not None:
                        metadata[key] = set_data[key]
                if isinstance(set_data.get('messages'), list):
                    latest_messages = set_data['messages']
                    for message in latest_messages:
                        if not isinstance(message, dict):
                            continue
                        message_id = _message_id(message)
                        tool_calls = message.get('toolCalls')
                        if message_id and tool_calls:
                            historical_tool_calls[message_id] = tool_calls
                continue

            if event.get('type') in {'user', 'gemini'}:
                fallback_messages.append(event)

    raw_messages = latest_messages if latest_messages is not None else fallback_messages
    if historical_tool_calls:
        enriched_messages = []
        for message in raw_messages:
            if not isinstance(message, dict):
                enriched_messages.append(message)
                continue
            enriched = dict(message)
            message_id = _message_id(message)
            if message_id and not enriched.get('toolCalls') and message_id in historical_tool_calls:
                enriched['toolCalls'] = historical_tool_calls[message_id]
            enriched_messages.append(enriched)
        raw_messages = enriched_messages

    data = dict(metadata)
    data['messages'] = raw_messages
    return data, raw_messages, parse_errors


def _normalize_gemini_messages(raw_messages):
    """Turn Gemini chat messages and function responses into canonical turns."""
    messages = []
    emitted_response_ids = set()
    response_ids = {
        response['tool_call_id']
        for raw in raw_messages
        if isinstance(raw, dict) and raw.get('type') == 'user'
        for block in _as_list(raw.get('content', ''))
        if isinstance(block, dict) and isinstance(block.get('functionResponse'), dict)
        for response in [_normalize_function_response(block['functionResponse'])]
        if response
    }
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        if raw.get('type') == 'user':
            content = raw.get('content', '')
            text = _content_text(content)
            if text:
                messages.append({
                    'role': 'user',
                    'content': text,
                    'timestamp': raw.get('timestamp'),
                })
            for block in _as_list(content):
                if not isinstance(block, dict) or not isinstance(block.get('functionResponse'), dict):
                    continue
                response = _normalize_function_response(block['functionResponse'])
                if response:
                    if response['tool_call_id'] in emitted_response_ids:
                        continue
                    emitted_response_ids.add(response['tool_call_id'])
                    messages.append(response)
            continue

        if raw.get('type') != 'gemini':
            continue

        normalized = {
            'role': 'assistant',
            'content': _content_text(raw.get('content', '')),
            'timestamp': raw.get('timestamp'),
        }
        for field in ('model', 'thoughts', 'tokens'):
            if raw.get(field):
                normalized[field] = raw[field]

        tool_calls = [
            normalized_call
            for normalized_call in (
                _normalize_tool_call(tool_call)
                for tool_call in _as_list(raw.get('toolCalls'))
            )
            if normalized_call
        ]
        if tool_calls:
            normalized['tool_calls'] = tool_calls
        messages.append(normalized)

        # Some snapshots retain a completed result only on toolCalls.  Add it
        # only when no corresponding functionResponse exists, avoiding a
        # second observation for the same call.
        for tool_call in _as_list(raw.get('toolCalls')):
            if not isinstance(tool_call, dict):
                continue
            call_id = _tool_call_id(tool_call)
            if not call_id or call_id in response_ids or 'result' not in tool_call:
                continue
            messages.append({
                'role': 'tool',
                'tool_call_id': call_id,
                'name': str(tool_call.get('name') or tool_call.get('displayName') or 'unknown_tool'),
                'content': _json_text(tool_call.get('result')),
            })

    return messages

def _admission_metadata(admission):
    if admission is None:
        return {}
    return {
        'source_manifest_revision': admission['source_manifest_revision'],
        'source_ref_sha256': admission['source_ref_sha256'],
        'source_snapshot_status': admission['source_snapshot_status'],
    }


def _append_source_manifest(source_manifest, entry, admission):
    if source_manifest is None:
        return
    entry.update(_admission_metadata(admission))
    source_manifest.append(entry)


def extract_gemini_session(
    session_file,
    *,
    source_manifest=None,
    source_ledger=None,
    root_label='primary',
):
    """Extract conversation from a Gemini CLI session file"""
    source_sha256 = None
    source_bytes = None
    source_admission = None
    try:
        source_sha256 = _sha256_file(session_file)
        source_bytes = session_file.stat().st_size
        if source_ledger is not None:
            source_admission = source_ledger.admit(
                source_sha256=source_sha256,
                provider='gemini-cli',
                root_label=root_label,
                source_class='session_active',
            )
            source_admission = validate_source_admission(
                source_admission,
                source_sha256=source_sha256,
                source_bytes=source_bytes,
            )
    except OSError:
        pass
    try:
        data, raw_messages, parse_errors = _load_gemini_session_data(session_file)
        messages = _normalize_gemini_messages(raw_messages)

        common_manifest = {
            'provider': 'gemini-cli',
            'source_class': 'session_active',
            'training_lane': 'primary',
            'source_file_name': session_file.name,
            'source_file_sha256': source_sha256,
            'source_bytes': source_bytes,
            'source_root_label': root_label,
            'session_id': data.get('sessionId'),
            'source_parse_errors': parse_errors,
        }

        if not raw_messages:
            _append_source_manifest(source_manifest, {
                **common_manifest,
                'status': 'empty_or_missing_messages',
                'emitted_records': 0,
            }, source_admission)
            return None

        if not messages:
            _append_source_manifest(source_manifest, {
                **common_manifest,
                'status': 'no_supported_messages',
                'emitted_records': 0,
            }, source_admission)
            return None

        conv = {
            'messages': messages,
            'source': 'gemini-cli',
            'session_id': data.get('sessionId'),
            'project_hash': data.get('projectHash'),
            'start_time': data.get('startTime'),
            'last_updated': data.get('lastUpdated'),
            'source_file': str(session_file),
            'source_class': 'session_active',
            'training_lane': 'primary',
            'source_origin': {
                'provider': 'gemini-cli',
                'source_class': 'session_active',
                'source_file_name': session_file.name,
                'source_file_sha256': source_sha256,
                'source_root_label': root_label,
                'extractor_version': EXTRACTOR_VERSION,
                'content_policy': 'thoughts_retained_in_private_ingress; canonical_builder_removes_reasoning',
                'source_parse_errors': parse_errors,
            }
        }
        conv['source_origin'].update(_admission_metadata(source_admission))
        if parse_errors:
            conv['source_parse_errors'] = parse_errors

        _append_source_manifest(source_manifest, {
            **common_manifest,
            'status': 'emitted',
            'emitted_records': 1,
        }, source_admission)

        return conv

    except Exception as e:
        if source_manifest is not None:
            _append_source_manifest(source_manifest, {
                'provider': 'gemini-cli',
                'source_class': 'session_active',
                'training_lane': 'primary',
                'source_file_name': session_file.name,
                'source_file_sha256': source_sha256,
                'source_bytes': source_bytes,
                'source_root_label': root_label,
                'status': 'read_error',
                'error_type': type(e).__name__,
                'emitted_records': 0,
            }, source_admission)
        if source_ledger is not None:
            raise
        return None

def find_gemini_installations(home=None):
    """Find all Gemini CLI installation directories."""
    system = platform.system()
    home = (home or Path.home()).expanduser()
    locations = []
    for candidate in (home / '.gemini', home / '.config/gemini', home / '.local/share/gemini'):
        if candidate.exists():
            locations.append(candidate)
    if system == 'Windows':
        user_profile = Path(os.environ.get('USERPROFILE', home))
        local_app_data = Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local'))
        for candidate in (user_profile / '.gemini', local_app_data / 'gemini'):
            if candidate.exists():
                locations.append(candidate)
    return list(set(locations))


def find_all_gemini_sessions(installation):
    """Find all Gemini CLI session files in an installation"""
    session_files = []

    # Search for session files in tmp/[hash]/chats/session-*.json pattern
    tmp_dir = installation / 'tmp'
    if tmp_dir.exists():
        # Gemini has used both JSON and JSONL session files.  Keep the
        # filename contract narrow while admitting both formats.
        session_files.extend(
            path
            for path in tmp_dir.rglob('chats/session-*')
            if path.is_file() and path.suffix.lower() in {'.json', '.jsonl'}
        )

    return sorted(set(session_files))


def write_source_manifest(output_dir, timestamp, source_manifest):
    """Publish metadata for every attempted Gemini session file."""
    manifest_file = output_dir / f'gemini_source_manifest_{timestamp}.json'
    payload = {
        'schema_version': 'ai-data-extraction/gemini-source-manifest/v1',
        'extractor_version': EXTRACTOR_VERSION,
        'provider': 'gemini-cli',
        'source_files': source_manifest,
        'counts': {
            'source_files': len(source_manifest),
            'emitted_files': sum(item.get('emitted_records', 0) > 0 for item in source_manifest),
            'empty_files': sum(item.get('status') in {'empty_or_missing_messages', 'no_supported_messages'} for item in source_manifest),
            'read_errors': sum(item.get('status') == 'read_error' for item in source_manifest),
        },
    }
    with manifest_file.open('w', encoding='utf-8') as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    return manifest_file

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-home',
        type=Path,
        help='explicit immutable synthetic HOME or live home containing provider roots',
    )
    parser.add_argument(
        '--source-manifest',
        type=Path,
        help='validated source ledger used for fail-closed byte admission',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(os.environ.get('EXTRACTED_DATA_DIR', 'extracted_data')),
    )
    return parser


def _installation_specs(source_home):
    if source_home is not None:
        return [
            (root.path, root.label)
            for root in discover_roots(source_home)
            if root.provider == 'gemini'
        ]
    return [(path, 'primary') for path in find_gemini_installations()]


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("="*80)
    print("GOOGLE GEMINI CLI DATA EXTRACTION")
    print("="*80)
    print()

    # Find all Gemini installations
    print("🔍 Searching for Gemini CLI installations...")
    installations = _installation_specs(args.source_home)

    if not installations:
        print("❌ No Gemini CLI installations found!")
        return

    print(f"✅ Found {len(installations)} installation(s):")
    for inst in installations:
        print(f"   - {inst}")
    print()

    # Extract from all installations
    all_conversations = []
    source_manifest = []
    source_ledger = (
        SourceManifestIndex.from_path(args.source_manifest)
        if args.source_manifest is not None
        else None
    )
    installation_stats = {}

    for installation, root_label in installations:
        print(f"📂 Processing: {installation}")

        session_files = find_all_gemini_sessions(installation)
        print(f"   Found {len(session_files)} session files")

        conversations = []
        for session_file in session_files:
            conv = extract_gemini_session(
                session_file,
                source_manifest=source_manifest,
                source_ledger=source_ledger,
                root_label=root_label,
            )
            if conv:
                conv['installation'] = str(installation)
                conversations.append(conv)

        if conversations:
            all_conversations.extend(conversations)
            installation_stats[str(installation)] = len(conversations)
            print(f"   ✅ {len(conversations)} conversations")
        else:
            print("   ⚠️  No conversations found")

    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Total conversations: {len(all_conversations):,}")

    if not all_conversations:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        manifest_file = write_source_manifest(output_dir, timestamp, source_manifest)
        print(f"✅ Source manifest: {manifest_file}")
        print("No conversations found!")
        return

    # Statistics
    total_messages = sum(len(c['messages']) for c in all_conversations)
    with_thoughts = sum(1 for c in all_conversations
                       if any('thoughts' in m for m in c['messages']))
    complete = sum(1 for c in all_conversations
                   if any(m['role'] == 'assistant' for m in c['messages']))

    print(f"Complete conversations: {complete:,}")
    print(f"Total messages: {total_messages:,}")
    print(f"With thoughts: {with_thoughts:,}")
    print()

    print("Breakdown by installation:")
    for inst, count in sorted(installation_stats.items(), key=lambda x: -x[1]):
        print(f"  {Path(inst).name:20} {count:5,} conversations")
    print()

    # Save to organized JSONL
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'gemini_conversations_{timestamp}.jsonl'

    manifest_file = write_source_manifest(output_dir, timestamp, source_manifest)

    with open(output_file, 'w') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"✅ Saved to: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print("   Format: JSONL (one conversation per line)")
    print(f"✅ Source manifest: {manifest_file}")

if __name__ == '__main__':
    main()
