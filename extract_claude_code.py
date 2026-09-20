#!/usr/bin/env python3
"""
Extract ALL Claude Code chat data from all projects
Includes: messages, code context, diffs, file references
Auto-discovers Claude Code installations on the device
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

def find_claude_installations(home=None):
    """Find all Claude Code installation directories"""
    system = platform.system()
    home = (home or Path.home()).expanduser()

    # Common installation locations by OS
    locations = []

    if system == "Darwin":  # macOS
        base_dirs = [
            home / "Library/Application Support",
            home / ".config"
        ]
    elif system == "Linux":
        base_dirs = [
            home / ".config",
            home / ".local/share"
        ]
    elif system == "Windows":
        base_dirs = [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')),
            Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local'))
        ]
    else:
        base_dirs = [home / ".config"]

    # Search for Claude-related directories
    claude_patterns = [
        'claude', 'claude-code', 'claude-local', 'claude-m2', 'claude-zai',
        '.claude', '.claude-code', '.claude-local', '.claude-m2', '.claude-zai'
    ]

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        # Check direct children
        for pattern in claude_patterns:
            claude_dir = base_dir / pattern
            if claude_dir.exists():
                locations.append(claude_dir)

        # Also check home directory directly
        for pattern in claude_patterns:
            home_dir = home / pattern
            if home_dir.exists():
                locations.append(home_dir)

    return list(set(locations))  # Remove duplicates


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

def extract_claude_project_conversations(
    project_dir,
    *,
    source_manifest=None,
    source_ledger=None,
    root_label='primary',
):
    """Extract conversations from a Claude project directory with full context"""
    conversations = []

    # Find all JSONL session files
    jsonl_files = []
    if (project_dir / 'projects').exists():
        # New structure: projects/project-name/session.jsonl
        for proj in (project_dir / 'projects').iterdir():
            if proj.is_dir():
                # Sidechain transcripts live below project/subagents and are
                # first-class source sessions for an optional lane.
                jsonl_files.extend(list(proj.rglob('*.jsonl')))
    else:
        # Old structure: direct JSONL files
        jsonl_files = list(project_dir.glob('*.jsonl'))

    jsonl_files = sorted(set(jsonl_files))

    for jsonl_file in jsonl_files:
        is_subagent = jsonl_file.name.startswith('agent-')
        source_class = 'subagent_session' if is_subagent else 'session_active'
        training_lane = 'optional_alt' if is_subagent else 'primary'
        source_sha256 = None
        source_bytes = None
        source_admission = None
        try:
            source_sha256 = _sha256_file(jsonl_file)
            source_bytes = jsonl_file.stat().st_size
            if source_ledger is not None:
                source_admission = source_ledger.admit(
                    source_sha256=source_sha256,
                    provider='claude-code',
                    root_label=root_label,
                    source_class=source_class,
                )
                source_admission = validate_source_admission(
                    source_admission,
                    source_sha256=source_sha256,
                    source_bytes=source_bytes,
                )
        except OSError:
            pass
        try:
            messages = []
            session_id = jsonl_file.stem
            project_path = None
            project_name = jsonl_file.parent.name if jsonl_file.parent.name != 'projects' else None
            parse_errors = 0
            parent_session_id = None
            agent_id = None
            sidechain = is_subagent

            with open(jsonl_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        obj = json.loads(line)
                        if parent_session_id is None and obj.get('sessionId'):
                            parent_session_id = obj.get('sessionId')
                        if agent_id is None and obj.get('agentId'):
                            agent_id = obj.get('agentId')
                        sidechain = sidechain or bool(obj.get('isSidechain'))
                        msg_type = obj.get('type')

                        if msg_type == 'user':
                            message = obj.get('message', {})
                            content = message.get('content', '')

                            if content:
                                msg = {
                                    'role': 'user',
                                    'content': content,
                                    'timestamp': obj.get('timestamp')
                                }

                                # Extract tool use (code context, diffs, etc.)
                                if 'toolUse' in obj:
                                    msg['tool_use'] = obj['toolUse']

                                messages.append(msg)

                            # Extract working directory
                            if 'cwd' in obj:
                                project_path = obj['cwd']

                        elif msg_type == 'assistant':
                            message = obj.get('message', {})
                            content = message.get('content', [])

                            # Extract text from content array
                            text_parts = []
                            tool_uses = []

                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        if item.get('type') == 'text':
                                            text_parts.append(item.get('text', ''))
                                        elif item.get('type') == 'tool_use':
                                            # Code execution, file edits, etc.
                                            tool_uses.append(item)
                            elif isinstance(content, str):
                                text_parts.append(content)

                            full_text = '\n'.join(text_parts)
                            if full_text or tool_uses:
                                msg = {
                                    'role': 'assistant',
                                    'content': full_text,
                                    'model': message.get('model'),
                                    'timestamp': obj.get('timestamp')
                                }

                                if tool_uses:
                                    msg['tool_uses'] = tool_uses

                                messages.append(msg)

                        elif msg_type == 'tool_result':
                            # Capture tool results (diffs, file reads, etc.)
                            tool_result = obj.get('toolResult', {})
                            if tool_result and messages:
                                # Add to last assistant message
                                if 'tool_results' not in messages[-1]:
                                    messages[-1]['tool_results'] = []
                                messages[-1]['tool_results'].append(tool_result)

                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue

            emitted_records = 0
            if messages:
                conversation = {
                    'messages': messages,
                    'source': 'claude-code',
                    'source_class': source_class,
                    'training_lane': training_lane,
                    'session_id': session_id,
                    'project_path': project_path,
                    'project_name': project_name,
                    'source_file': str(jsonl_file),
                    'installation': str(project_dir),
                    'source_origin': {
                        'provider': 'claude-code',
                        'source_class': source_class,
                        'source_file_name': jsonl_file.name,
                        'source_file_sha256': source_sha256,
                        'source_root_label': root_label,
                        'parent_session_id': parent_session_id,
                        'agent_id': agent_id,
                        'sidechain': sidechain,
                        'extractor_version': EXTRACTOR_VERSION,
                        'content_policy': 'reasoning_omitted_at_adapter; canonical_builder_required',
                    },
                }
                conversation['source_origin'].update(_admission_metadata(source_admission))
                if parse_errors:
                    conversation['source_parse_errors'] = parse_errors
                conversations.append(conversation)
                emitted_records = 1

            _append_source_manifest(source_manifest, {
                    'provider': 'claude-code',
                    'source_class': source_class,
                    'training_lane': training_lane,
                    'source_file_name': jsonl_file.name,
                    'source_file_sha256': source_sha256,
                    'source_bytes': source_bytes,
                    'source_root_label': root_label,
                    'session_id': session_id,
                    'parent_session_id': parent_session_id,
                    'agent_id': agent_id,
                    'sidechain': sidechain,
                    'parse_errors': parse_errors,
                    'emitted_records': emitted_records,
                    'status': 'emitted' if emitted_records else 'no_emitted_messages',
                }, source_admission)

        except Exception as e:
            if source_manifest is not None:
                _append_source_manifest(source_manifest, {
                    'provider': 'claude-code',
                    'source_class': source_class,
                    'training_lane': training_lane,
                    'source_file_name': jsonl_file.name,
                    'source_file_sha256': source_sha256,
                    'source_bytes': source_bytes,
                    'source_root_label': root_label,
                    'status': 'read_error',
                    'error_type': type(e).__name__,
                }, source_admission)
            if source_ledger is not None:
                raise
            print(f"Error processing {jsonl_file}: {e}")
            continue

    return conversations

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
            if root.provider == 'claude'
        ]
    home = Path.home()
    labels = {
        home / name: f'root-{index}'
        for index, name in enumerate(
            ('.claude', '.claude-code', '.claude-local', '.claude-m2', '.claude-zai')
        )
    }
    return [(path, labels.get(path, 'primary')) for path in find_claude_installations()]


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("="*80)
    print("CLAUDE CODE COMPLETE DATA EXTRACTION")
    print("="*80)
    print()

    # Find all Claude installations
    print("🔍 Searching for Claude Code installations...")
    installations = _installation_specs(args.source_home)

    if not installations:
        print("❌ No Claude Code installations found!")
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

        conversations = extract_claude_project_conversations(
            installation,
            source_manifest=source_manifest,
            source_ledger=source_ledger,
            root_label=root_label,
        )

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
        print("No conversations found!")
        return

    # Statistics
    total_messages = sum(len(c['messages']) for c in all_conversations)
    with_tools = sum(1 for c in all_conversations
                     if any('tool_use' in m or 'tool_uses' in m or 'tool_results' in m
                           for m in c['messages']))
    complete = sum(1 for c in all_conversations
                   if any(m['role'] == 'assistant' for m in c['messages']))

    print(f"Complete conversations: {complete:,}")
    print(f"Total messages: {total_messages:,}")
    print(f"With tool use/diffs: {with_tools:,}")
    print()

    print("Breakdown by installation:")
    for inst, count in sorted(installation_stats.items(), key=lambda x: -x[1]):
        print(f"  {Path(inst).name:20} {count:5,} conversations")
    print()

    # Save to organized JSONL
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'claude_code_conversations_{timestamp}.jsonl'

    with open(output_file, 'w') as f:
        f.writelines(json.dumps(conv, ensure_ascii=False) + '\n' for conv in all_conversations)

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"✅ Saved to: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print("   Format: JSONL (one conversation per line)")

    manifest_file = output_dir / f'claude_code_source_manifest_{timestamp}.json'
    manifest = {
        'schema_version': 'ai-data-extraction/claude-source-manifest/v1',
        'extractor_version': EXTRACTOR_VERSION,
        'provider': 'claude-code',
        'source_files': source_manifest,
        'counts': {
            'source_files': len(source_manifest),
            'emitted_files': sum(item.get('emitted_records', 0) > 0 for item in source_manifest),
            'empty_files': sum(item.get('status') == 'no_emitted_messages' for item in source_manifest),
            'read_errors': sum(item.get('status') == 'read_error' for item in source_manifest),
        },
    }
    with manifest_file.open('w', encoding='utf-8') as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write('\n')
    print(f"✅ Source manifest: {manifest_file}")

if __name__ == '__main__':
    main()
