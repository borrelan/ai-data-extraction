#!/usr/bin/env python3
"""
Extract ALL Codex chat data from all projects
Includes: messages, code context, diffs, file references
Auto-discovers Codex installations on the device
"""

import hashlib
import json
import os
import platform
import tempfile
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from inventory_sources import discover_roots
from source_manifest import (
    SourceManifestIndex,
    validate_source_admission,
    write_manifest as write_metadata_manifest,
)

def find_codex_installations(home=None):
    """Find all Codex installation directories"""
    system = platform.system()
    home = (home or Path.home()).expanduser()

    locations = []

    # Search patterns for Codex directories
    codex_patterns = [
        'codex', 'codex-local', '.codex', '.codex-local'
    ]

    if system == "Darwin":  # macOS
        base_dirs = [
            home / "Library/Application Support",
            home / ".config",
            home
        ]
    elif system == "Linux":
        base_dirs = [
            home / ".config",
            home / ".local/share",
            home
        ]
    elif system == "Windows":
        base_dirs = [
            Path(os.environ.get('APPDATA', home / 'AppData/Roaming')),
            Path(os.environ.get('LOCALAPPDATA', home / 'AppData/Local')),
            home
        ]
    else:
        base_dirs = [home / ".config", home]

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        for pattern in codex_patterns:
            codex_dir = base_dir / pattern
            if codex_dir.exists():
                locations.append(codex_dir)

    return list(set(locations))

EXTRACTOR_VERSION = '1.2.0'
CODEX_CHECKPOINT_SCHEMA = 'ai-data-extraction/codex-checkpoint/v1'
MAX_OUTPUT_RECORD_CHARS = 250_000
# The canonical builder materializes both messages and event projections.  A
# deliberately conservative ingress target leaves room for that second view,
# lineage, and context anchors without turning a valid source episode into an
# oversized normalized rejection.
OUTPUT_TARGET_FRACTION = 0.25
NATIVE_RESPONSE_TYPES = frozenset(
    {
        'message',
        'function_call',
        'custom_tool_call',
        'function_call_output',
        'custom_tool_call_output',
    }
)
MESSAGE_ROLES = frozenset({'system', 'user', 'assistant', 'tool'})


@dataclass(frozen=True)
class CodexSessionInspection:
    source_sha256: str
    mode: str
    session_meta: dict[str, Any]
    parse_errors: int
    unmatched_tool_call_ids: frozenset[str]
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class CodexChunkSpec:
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    message_start: int
    message_end: int
    source_line_start: int
    source_line_end: int
    open_tool_call_ids: tuple[str, ...]
    observation_truncations: list[dict[str, Any]]
    anchor_message_index: int | None


@dataclass(frozen=True)
class CodexSourceSpec:
    """One immutable source stream and its installation/root provenance."""

    path: Path
    installation: Path
    root_label: str


def _source_class(session_file: Path) -> str:
    return 'session_backup' if session_file.name.endswith('.jsonl.backup') else 'session_active'


def _training_lane(session_file: Path) -> str:
    """Keep backup provenance out of the primary mixture by default."""
    return 'optional_alt' if _source_class(session_file) == 'session_backup' else 'primary'


def _response_tool_call(payload: dict[str, Any]) -> dict[str, Any]:
    call_id = payload.get('call_id') or payload.get('callID') or payload.get('id')
    name = payload.get('name') or payload.get('tool_name') or 'unknown'
    arguments = payload.get('arguments', payload.get('input', {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))
    return {
        'id': call_id,
        'type': 'function',
        'function': {
            'name': name,
            'arguments': arguments,
        },
    }


def _response_tool_output(payload: dict[str, Any], timestamp: Any) -> dict[str, Any]:
    output = payload.get('output', payload.get('result', ''))
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, separators=(',', ':'))
    return {
        'role': 'tool',
        'content': output,
        'tool_call_id': payload.get('call_id') or payload.get('callID') or payload.get('id'),
        'timestamp': timestamp,
    }


def _inspect_codex_session(session_file: Path) -> CodexSessionInspection:
    """Hash and classify a source stream without retaining its messages."""
    stat_before = session_file.stat()
    digest = hashlib.sha256()
    session_meta: dict[str, Any] = {}
    native = False
    parse_errors = 0
    tool_call_ids: set[str] = set()
    tool_output_ids: set[str] = set()

    with session_file.open('rb') as source:
        for raw_line in source:
            digest.update(raw_line)
            try:
                obj = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parse_errors += 1
                continue
            if not isinstance(obj, dict):
                continue
            event_type = obj.get('type')
            payload = obj.get('payload')
            if event_type == 'session_meta' and isinstance(payload, dict):
                session_meta = payload
            if (
                event_type == 'response_item'
                and isinstance(payload, dict)
                and payload.get('type') in NATIVE_RESPONSE_TYPES
            ):
                native = True
                payload_type = payload.get('type')
                call_id = payload.get('call_id') or payload.get('callID') or payload.get('id')
                if call_id is not None:
                    if payload_type in {'function_call', 'custom_tool_call'}:
                        tool_call_ids.add(str(call_id))
                    elif payload_type in {
                        'function_call_output',
                        'custom_tool_call_output',
                    }:
                        tool_output_ids.add(str(call_id))

    stat_after = session_file.stat()
    if stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
        raise RuntimeError(f'source changed during inspection: {session_file.name}')
    return CodexSessionInspection(
        source_sha256=digest.hexdigest(),
        mode='native_response_items' if native else 'legacy_event_messages',
        session_meta=session_meta,
        parse_errors=parse_errors,
        unmatched_tool_call_ids=frozenset(tool_call_ids - tool_output_ids),
        size=stat_after.st_size,
        mtime_ns=stat_after.st_mtime_ns,
    )


def _assert_codex_session_stable(session_file: Path, inspection: CodexSessionInspection) -> None:
    stat = session_file.stat()
    if stat.st_size != inspection.size or stat.st_mtime_ns != inspection.mtime_ns:
        raise RuntimeError(f'source changed during bounded extraction: {session_file.name}')


def _iter_codex_message_events(
    session_file: Path,
    mode: str,
) -> Iterator[tuple[str, dict[str, Any], int]]:
    """Yield one source-ordered message or detached legacy attachment."""
    pending_assistant: dict[str, Any] | None = None
    pending_line: int | None = None

    def take_pending() -> tuple[dict[str, Any], int] | None:
        nonlocal pending_assistant, pending_line
        if pending_assistant is None or pending_line is None:
            return None
        result = (pending_assistant, pending_line)
        pending_assistant = None
        pending_line = None
        return result

    with session_file.open('rb') as source:
        for source_line, raw_line in enumerate(source, start=1):
            try:
                obj = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            event_type = obj.get('type')
            payload = obj.get('payload')
            if not isinstance(payload, dict):
                continue

            if mode == 'native_response_items':
                if event_type != 'response_item':
                    continue
                payload_type = payload.get('type')
                timestamp = obj.get('timestamp')
                if payload_type == 'message':
                    role = payload.get('role')
                    if role not in MESSAGE_ROLES:
                        continue
                    flushed = take_pending()
                    if flushed is not None:
                        yield 'message', flushed[0], flushed[1]
                    message = {
                        'role': role,
                        'content': payload.get('content', ''),
                        'timestamp': timestamp,
                    }
                    for key in ('tool_calls', 'tool_call_id'):
                        if key in payload:
                            message[key] = payload[key]
                    pending_assistant = message
                    pending_line = source_line
                elif payload_type in {'function_call', 'custom_tool_call'}:
                    if (
                        pending_assistant is None
                        or pending_assistant.get('role') != 'assistant'
                    ):
                        flushed = take_pending()
                        if flushed is not None:
                            yield 'message', flushed[0], flushed[1]
                        pending_assistant = {
                            'role': 'assistant',
                            'content': '',
                            'timestamp': timestamp,
                            'tool_calls': [],
                        }
                        pending_line = source_line
                    pending_assistant.setdefault('tool_calls', []).append(
                        _response_tool_call(payload)
                    )
                elif payload_type in {'function_call_output', 'custom_tool_call_output'}:
                    flushed = take_pending()
                    if flushed is not None:
                        yield 'message', flushed[0], flushed[1]
                    yield 'message', _response_tool_output(payload, timestamp), source_line
                continue

            if event_type != 'event_msg':
                continue
            payload_type = payload.get('type')
            timestamp = obj.get('timestamp')
            if payload_type in {'user_message', 'agent_message'}:
                message_text = payload.get('message', '')
                if not isinstance(message_text, str):
                    continue
                message_text = message_text.strip()
                if not message_text:
                    continue
                role = 'user' if payload_type == 'user_message' else 'assistant'
                message: dict[str, Any] = {
                    'role': role,
                    'content': message_text,
                    'timestamp': timestamp,
                }
                if role == 'user' and 'context' in payload:
                    message['context'] = payload['context']
                if role == 'assistant' and 'model' in payload:
                    message['model'] = payload['model']
                yield 'message', message, source_line
            elif payload_type == 'tool_use':
                yield 'attachment', {
                    'type': 'tool_use',
                    'tool': payload.get('tool'),
                    'input': payload.get('input'),
                    'timestamp': timestamp,
                }, source_line
            elif payload_type == 'tool_result':
                yield 'attachment', {
                    'type': 'tool_result',
                    'tool': payload.get('tool'),
                    'output': payload.get('output'),
                    'timestamp': timestamp,
                }, source_line
            elif payload_type == 'diff':
                yield 'attachment', {
                    'type': 'diff',
                    'file': payload.get('file'),
                    'diff': payload.get('diff'),
                    'timestamp': timestamp,
                }, source_line

    flushed = take_pending()
    if flushed is not None:
        yield 'message', flushed[0], flushed[1]


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) + 1


def _observation_budget(max_chars: int) -> int:
    """Reserve room for the turn, event projection, and lineage metadata."""
    return max(256, int(max_chars * 0.20))


def _bounded_observation_value(
    value: Any,
    *,
    field: str,
    max_chars: int,
    source_line: int,
    call_id: Any = None,
    budget_chars: int | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Keep a large observation useful while retaining a verifiable omission."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            return value, None
    budget = budget_chars if budget_chars is not None else _observation_budget(max_chars)
    if len(text) <= budget:
        return value, None

    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    marker = (
        f'\n<OBSERVATION_TRUNCATED original_chars={len(text)} '
        f'sha256={digest} policy=head_tail>\n'
    )
    available = max(0, budget - len(marker))
    head_chars = int(available * 0.70)
    tail_chars = available - head_chars
    bounded = text[:head_chars] + marker + (text[-tail_chars:] if tail_chars else '')
    return bounded, {
        'field': field,
        'source_line': source_line,
        'call_id': str(call_id) if call_id is not None else None,
        'original_chars': len(text),
        'original_sha256': digest,
        'kept_chars': len(bounded),
        'policy': 'head_tail',
    }


def _bound_tool_message(
    message: dict[str, Any],
    *,
    max_chars: int,
    source_line: int,
    budget_chars: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if message.get('role') != 'tool' or 'content' not in message:
        return message, None
    bounded, truncation = _bounded_observation_value(
        message.get('content'),
        field='content',
        max_chars=max_chars,
        source_line=source_line,
        call_id=message.get('tool_call_id'),
        budget_chars=budget_chars,
    )
    if truncation is None:
        return message, None
    bounded_message = dict(message)
    bounded_message['content'] = bounded
    return bounded_message, truncation


def _bound_attachment(
    attachment: dict[str, Any],
    *,
    max_chars: int,
    source_line: int,
    budget_chars: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    for field in ('output', 'diff'):
        if field not in attachment:
            continue
        bounded, truncation = _bounded_observation_value(
            attachment[field],
            field=field,
            max_chars=max_chars,
            source_line=source_line,
            call_id=attachment.get('call_id') or attachment.get('id'),
            budget_chars=budget_chars,
        )
        if truncation is None:
            continue
        bounded_attachment = dict(attachment)
        bounded_attachment[field] = bounded
        return bounded_attachment, truncation
    return attachment, None


def _message_action_ids(message: dict[str, Any]) -> set[str]:
    values = message.get('tool_calls')
    if not isinstance(values, list):
        return set()
    result: set[str] = set()
    for call in values:
        if isinstance(call, dict):
            call_id = call.get('id') or call.get('call_id') or call.get('callID')
            if call_id is not None:
                result.add(str(call_id))
    return result


def _message_observation_id(message: dict[str, Any]) -> str | None:
    value = message.get('tool_call_id') or message.get('call_id') or message.get('callID')
    return str(value) if value is not None else None


def _iter_codex_chunk_specs(
    session_file: Path,
    inspection: CodexSessionInspection,
    max_chars: int,
) -> Iterator[CodexChunkSpec]:
    """Segment a source stream while retaining only one bounded chunk."""
    if max_chars <= 0:
        raise ValueError('max_chars must be positive')
    target_size = max(1, int(max_chars * OUTPUT_TARGET_FRACTION))
    prefix: list[dict[str, Any]] = []
    prefix_size = 0
    seen_non_system = False
    current: list[dict[str, Any]] = []
    current_size = 0
    current_start: int | None = None
    current_end: int | None = None
    current_line_start: int | None = None
    current_line_end: int | None = None
    current_has_user = False
    current_has_assistant = False
    current_last_role: str | None = None
    current_last_user: tuple[dict[str, Any], int] | None = None
    current_anchor_index: int | None = None
    pending_tool_ids: set[str] = set()
    pending_tool_batch_size = 0
    tool_results: list[dict[str, Any]] = []
    observation_truncations: list[dict[str, Any]] = []
    message_index = 0

    def reset_current(
        continuation_user: tuple[dict[str, Any], int] | None = None,
    ) -> None:
        nonlocal current, current_size, current_start, current_end
        nonlocal current_line_start, current_line_end, current_has_user
        nonlocal current_has_assistant, current_last_role
        nonlocal current_last_user, current_anchor_index
        nonlocal pending_tool_ids, pending_tool_batch_size
        nonlocal tool_results, observation_truncations
        current = list(prefix)
        current_size = prefix_size
        current_start = None
        current_end = None
        current_line_start = None
        current_line_end = None
        current_has_user = False
        current_has_assistant = False
        current_last_role = 'system' if prefix else None
        current_last_user = None
        current_anchor_index = None
        pending_tool_ids = set()
        pending_tool_batch_size = 0
        tool_results = []
        observation_truncations = []
        if continuation_user is not None:
            anchor_message, anchor_index = continuation_user
            current.append(anchor_message)
            current_size += _json_size(anchor_message)
            current_has_user = True
            current_last_role = 'user'
            current_last_user = continuation_user
            current_anchor_index = anchor_index

    def take_current(
        continuation_user: tuple[dict[str, Any], int] | None = None,
    ) -> CodexChunkSpec:
        nonlocal current, current_size, current_start, current_end
        nonlocal current_line_start, current_line_end, tool_results
        nonlocal observation_truncations
        start = current_start if current_start is not None else 0
        end = current_end if current_end is not None else start
        line_start = current_line_start if current_line_start is not None else 1
        line_end = current_line_end if current_line_end is not None else line_start
        result = CodexChunkSpec(
            messages=list(current),
            tool_results=list(tool_results),
            message_start=start,
            message_end=end,
            source_line_start=line_start,
            source_line_end=line_end,
            open_tool_call_ids=tuple(sorted(pending_tool_ids)),
            observation_truncations=list(observation_truncations),
            anchor_message_index=current_anchor_index,
        )
        reset_current(continuation_user)
        return result

    reset_current()
    for kind, value, source_line in _iter_codex_message_events(session_file, inspection.mode):
        if kind == 'attachment':
            value, truncation = _bound_attachment(
                value,
                max_chars=max_chars,
                source_line=source_line,
                budget_chars=(
                    _observation_budget(max_chars)
                    // max(1, pending_tool_batch_size)
                ),
            )
            if truncation is not None:
                observation_truncations.append(truncation)
            if current_line_start is None:
                current_line_start = source_line
            current_line_end = source_line
            tool_results.append(value)
            current_size += _json_size(value)
            continue

        message, truncation = _bound_tool_message(
            value,
            max_chars=max_chars,
            source_line=source_line,
            budget_chars=(
                _observation_budget(max_chars)
                // max(1, pending_tool_batch_size)
            ),
        )
        if truncation is not None:
            observation_truncations.append(truncation)
        role = message.get('role')
        piece_size = _json_size(message)
        if not seen_non_system and role == 'system':
            prefix.append(message)
            prefix_size += piece_size
            current.append(message)
            current_size += piece_size
            current_line_start = source_line if current_line_start is None else current_line_start
            current_line_end = source_line
            message_index += 1
            continue
        seen_non_system = True

        split_before = False
        if current_has_user and current_has_assistant and current_size + piece_size > target_size:
            only_unmatched_tools = pending_tool_ids and pending_tool_ids.issubset(
                inspection.unmatched_tool_call_ids
            )
            if role == 'user':
                split_before = not pending_tool_ids or only_unmatched_tools
            elif role != 'tool' and (not pending_tool_ids or only_unmatched_tools):
                split_before = current_last_role in {'assistant', 'tool'}
        if split_before and current:
            continuation_user = current_last_user if role != 'user' else None
            yield take_current(continuation_user)

        if current_start is None:
            current_start = message_index
        current_end = message_index
        current_line_start = source_line if current_line_start is None else current_line_start
        current_line_end = source_line
        current.append(message)
        current_size += piece_size
        current_has_user = current_has_user or role == 'user'
        current_has_assistant = current_has_assistant or role == 'assistant'
        current_last_role = role
        if role == 'user':
            current_last_user = (message, message_index)
        observation_id = _message_observation_id(message)
        if observation_id is not None:
            pending_tool_ids.discard(observation_id)
        action_ids = _message_action_ids(message)
        if action_ids:
            pending_tool_batch_size = max(
                pending_tool_batch_size,
                len(pending_tool_ids) + len(action_ids),
            )
        pending_tool_ids.update(action_ids)
        if not pending_tool_ids:
            pending_tool_batch_size = 0
        message_index += 1

    if current and (len(current) > len(prefix) or tool_results or not prefix):
        yield take_current()


def _codex_chunk_record(
    session_file: Path,
    inspection: CodexSessionInspection,
    spec: CodexChunkSpec,
    *,
    chunk_index: int,
    chunk_count: int,
    installation: Path | None = None,
    source_admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_class = _source_class(session_file)
    record: dict[str, Any] = {
        'messages': spec.messages,
        'session_id': inspection.session_meta.get('id'),
        'cwd': inspection.session_meta.get('cwd'),
        'source': 'codex',
        'source_class': source_class,
        'training_lane': _training_lane(session_file),
        'session_file': str(session_file),
        'timestamp': inspection.session_meta.get('timestamp'),
        'source_origin': {
            'provider': 'codex',
            'source_class': source_class,
            'training_lane': _training_lane(session_file),
            'source_file_name': session_file.name,
            'source_file_sha256': inspection.source_sha256,
            'source_event_line_range': {
                'start': spec.source_line_start,
                'end': spec.source_line_end,
            },
        },
        '_chunk_parent_record_sha256': inspection.source_sha256,
        '_chunk_index': chunk_index,
        '_chunk_count': chunk_count,
        '_chunk_message_start': spec.message_start,
        '_chunk_message_end': spec.message_end,
        '_chunk_cut_reason': (
            'ingress_record_budget_with_unmatched_tool_call'
            if spec.open_tool_call_ids
            else 'ingress_record_budget'
        ),
        '_ingress_mode': inspection.mode,
    }
    for key in ('model_provider', 'model', 'model_id', 'modelID', 'model_name'):
        value = inspection.session_meta.get(key)
        if value not in (None, ''):
            record[key] = value
    if source_admission is not None:
        record['source_origin'].update(
            {
                'source_manifest_revision': source_admission['source_manifest_revision'],
                'source_ref_sha256': source_admission['source_ref_sha256'],
                'source_snapshot_status': source_admission['source_snapshot_status'],
            }
        )
    if spec.anchor_message_index is not None:
        record['_chunk_anchor_message_index'] = spec.anchor_message_index
    if spec.open_tool_call_ids:
        record['_open_tool_call_ids'] = list(spec.open_tool_call_ids)
    if spec.observation_truncations:
        record['observation_truncations'] = spec.observation_truncations
    if inspection.parse_errors:
        record['source_parse_errors'] = inspection.parse_errors
    if spec.tool_results:
        record['tool_results'] = spec.tool_results
    if installation is not None:
        record['installation'] = str(installation)
    return record


def iter_codex_session_records(
    session_file: Path,
    *,
    max_chars: int = MAX_OUTPUT_RECORD_CHARS,
    installation: Path | None = None,
    root_label: str = 'primary',
    source_manifest: SourceManifestIndex | None = None,
    source_admission: dict[str, Any] | None = None,
    inspection: CodexSessionInspection | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield bounded provider records without retaining a whole session.

    The source is inspected once, then streamed twice: one pass counts
    segments so lineage has an exact count, and one pass emits them.  Source
    size/mtime are checked between passes; a changing source fails closed.
    """
    inspection = inspection or _inspect_codex_session(session_file)
    if source_manifest is not None:
        if source_admission is not None:
            raise ValueError('source_manifest and source_admission are mutually exclusive')
        source_admission = source_manifest.admit(
            source_sha256=inspection.source_sha256,
            provider='codex',
            root_label=root_label,
            source_class=_source_class(session_file),
        )
    if source_admission is not None:
        source_admission = validate_source_admission(
            source_admission,
            source_sha256=inspection.source_sha256,
            source_bytes=inspection.size,
        )
    _assert_codex_session_stable(session_file, inspection)
    chunk_count = sum(
        1
        for _ in _iter_codex_chunk_specs(session_file, inspection, max_chars)
    )
    _assert_codex_session_stable(session_file, inspection)
    for chunk_index, spec in enumerate(
        _iter_codex_chunk_specs(session_file, inspection, max_chars)
    ):
        yield _codex_chunk_record(
            session_file,
            inspection,
            spec,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            installation=installation,
            source_admission=source_admission,
        )
    _assert_codex_session_stable(session_file, inspection)


def extract_codex_session(session_file):
    """Return the compatibility full-session projection.

    The orchestration path uses :func:`iter_codex_session_records` instead so
    normal extraction never materializes an entire long session.  This helper
    remains for callers that explicitly need the historical single-record
    shape and should only be used on bounded inputs.
    """
    inspection = _inspect_codex_session(session_file)
    messages: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for kind, value, _source_line in _iter_codex_message_events(session_file, inspection.mode):
        if kind == 'message':
            messages.append(value)
        else:
            tool_results.append(value)
    _assert_codex_session_stable(session_file, inspection)
    if not messages:
        return None

    conv: dict[str, Any] = {
        'messages': messages,
        'session_id': inspection.session_meta.get('id'),
        'cwd': inspection.session_meta.get('cwd'),
        'source': 'codex',
        'source_class': _source_class(session_file),
        'training_lane': _training_lane(session_file),
        'session_file': str(session_file),
        'timestamp': inspection.session_meta.get('timestamp'),
    }
    for key in ('model_provider', 'model', 'model_id', 'modelID', 'model_name'):
        value = inspection.session_meta.get(key)
        if value not in (None, ''):
            conv[key] = value
    if tool_results:
        conv['tool_results'] = tool_results
    if inspection.parse_errors:
        conv['source_parse_errors'] = inspection.parse_errors
    return conv

def find_all_codex_sessions(installation, *, include_backups=False):
    """Find active Codex sessions, with backups only when explicitly enabled."""
    session_files = []

    # Check for sessions directory
    sessions_dir = installation / 'sessions'
    if sessions_dir.exists():
        # Sessions are organized by date: YYYY/MM/DD/rollout-*.jsonl
        session_files.extend(list(sessions_dir.rglob('rollout-*.jsonl')))
        if include_backups:
            session_files.extend(list(sessions_dir.rglob('rollout-*.jsonl.backup')))

    # Also check for project-based structure
    projects_dir = installation / 'projects'
    if projects_dir.exists():
        session_files.extend(list(projects_dir.rglob('*.jsonl')))

    return sorted(
        set(session_files),
        key=lambda path: (path.name.endswith('.jsonl.backup'), path.as_posix()),
    )


def _codex_source_key(
    source: CodexSourceSpec,
    inspection: CodexSessionInspection,
) -> tuple[str, str, str, str]:
    return (
        source.root_label,
        _source_class(source.path),
        'codex',
        inspection.source_sha256,
    )


def _load_codex_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'cannot load Codex checkpoint {path}: {exc}') from exc
    if not isinstance(value, dict) or value.get('schema_version') != CODEX_CHECKPOINT_SCHEMA:
        raise ValueError(f'unsupported Codex checkpoint: {path}')
    return value


def _file_digest_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _prefix_digest(path: Path, limit: int) -> tuple[str, int, Any]:
    """Hash exactly the committed prefix and return the live digest state."""
    if limit < 0:
        raise ValueError('prefix limit must be non-negative')
    digest = hashlib.sha256()
    remaining = limit
    size = 0
    with path.open('rb') as source:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            remaining -= len(chunk)
    return digest.hexdigest(), size, digest


def write_codex_records(
    sources: Iterator[CodexSourceSpec] | list[CodexSourceSpec],
    *,
    output_file: Path,
    timestamp: str,
    max_chars: int = MAX_OUTPUT_RECORD_CHARS,
    source_manifest: SourceManifestIndex | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Write Codex records with source-granular checkpoint/resume semantics."""

    if max_chars <= 0:
        raise ValueError('max_chars must be positive')
    output_file = Path(output_file)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint_path is None:
        raise ValueError('resume requires checkpoint_path')
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and checkpoint_path.exists():
            raise FileExistsError(
                f'Refusing to overwrite Codex checkpoint {checkpoint_path}; pass resume'
            )

    checkpoint: dict[str, Any] | None = None
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f'Codex checkpoint does not exist: {checkpoint_path}')
        checkpoint = _load_codex_checkpoint(checkpoint_path)
        expected_revision = source_manifest.revision if source_manifest is not None else None
        if checkpoint.get('source_manifest_revision') != expected_revision:
            raise ValueError('Codex checkpoint source-manifest revision does not match')
        target_name = checkpoint.get('target_name')
        if target_name != output_file.name:
            raise ValueError('Codex checkpoint target does not match output_file')
        if output_file.exists():
            raise ValueError('Codex output already exists for an unfinished checkpoint')
        temporary_name = checkpoint.get('temporary_name')
        if not isinstance(temporary_name, str) or Path(temporary_name).name != temporary_name:
            raise ValueError('Codex checkpoint temporary output name is invalid')
        temporary = output_file.parent / temporary_name
        expected_bytes = checkpoint.get('temporary_bytes')
        expected_digest = checkpoint.get('temporary_sha256')
        if (
            not temporary.is_file()
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_digest, str)
        ):
            raise ValueError('Codex checkpoint temporary output is not resumable')
        temporary_size = temporary.stat().st_size
        if temporary_size < expected_bytes:
            raise ValueError('Codex checkpoint temporary output is not resumable')
        actual_digest, actual_bytes, temporary_digest = _prefix_digest(
            temporary, expected_bytes
        )
        if actual_bytes != expected_bytes or actual_digest != expected_digest.removeprefix('sha256:'):
            raise ValueError('Codex checkpoint temporary output digest does not match')
        if temporary_size > expected_bytes:
            with temporary.open('r+b') as destination:
                destination.truncate(expected_bytes)
                destination.flush()
                os.fsync(destination.fileno())
    else:
        if output_file.exists():
            raise FileExistsError(f'Refusing to overwrite Codex output {output_file}')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f'.{output_file.name}.', suffix='.tmp', dir=output_file.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary_digest = hashlib.sha256()

    source_entries: list[dict[str, Any]] = []
    completed_entries: list[dict[str, str]] = []
    completed_keys: set[tuple[str, str, str, str]] = set()
    records = 0
    total_messages = 0
    with_tools = 0
    complete = 0
    if checkpoint is not None:
        raw_entries = checkpoint.get('source_sessions')
        raw_completed = checkpoint.get('completed_sources')
        if not isinstance(raw_entries, list) or not isinstance(raw_completed, list):
            raise ValueError('Codex checkpoint source state is invalid')
        source_entries = [dict(item) for item in raw_entries if isinstance(item, dict)]
        if len(source_entries) != len(raw_entries):
            raise ValueError('Codex checkpoint source entries are invalid')
        for item in raw_completed:
            if not isinstance(item, dict):
                raise ValueError('Codex checkpoint completion entries are invalid')
            fields = (
                item.get('root_label'),
                item.get('source_class'),
                item.get('provider'),
                item.get('source_sha256'),
            )
            if not all(isinstance(field, str) and field for field in fields):
                raise ValueError('Codex checkpoint source identity is invalid')
            completed_keys.add(fields)  # type: ignore[arg-type]
            completed_entries.append(dict(item))
        records = int(checkpoint.get('records', 0))
        total_messages = int(checkpoint.get('messages', 0))
        with_tools = int(checkpoint.get('with_tools', 0))
        complete = int(checkpoint.get('complete', 0))

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        state = {
            'schema_version': CODEX_CHECKPOINT_SCHEMA,
            'extractor_version': EXTRACTOR_VERSION,
            'source_manifest_revision': (
                source_manifest.revision if source_manifest is not None else None
            ),
            'target_name': output_file.name,
            'temporary_name': temporary.name,
            'temporary_bytes': temporary.stat().st_size,
            'temporary_sha256': temporary_digest.hexdigest(),
            'records': records,
            'messages': total_messages,
            'with_tools': with_tools,
            'complete': complete,
            'source_sessions': source_entries,
            'completed_sources': completed_entries,
        }
        write_metadata_manifest(checkpoint_path, state, overwrite=True)

    seen_keys: set[tuple[str, str, str, str]] = set()
    try:
        with temporary.open('a', encoding='utf-8') as destination:
            for source in sources:
                inspection = _inspect_codex_session(source.path)
                source_key = _codex_source_key(source, inspection)
                seen_keys.add(source_key)
                if source_key in completed_keys:
                    continue

                admission = None
                if source_manifest is not None:
                    admission = source_manifest.admit(
                        source_sha256=inspection.source_sha256,
                        provider='codex',
                        root_label=source.root_label,
                        source_class=_source_class(source.path),
                    )
                session_record_count = 0
                session_messages = 0
                session_with_tools = 0
                session_complete = 0
                for record in iter_codex_session_records(
                    source.path,
                    max_chars=max_chars,
                    installation=source.installation,
                    root_label=source.root_label,
                    source_admission=admission,
                    inspection=inspection,
                ):
                    line = (
                        json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                        + '\n'
                    )
                    destination.write(line)
                    temporary_digest.update(line.encode('utf-8'))
                    session_record_count += 1
                    session_messages += len(record.get('messages', []))
                    has_tools = 'tool_results' in record or any(
                        isinstance(message, dict)
                        and ('tool_calls' in message or message.get('role') == 'tool')
                        for message in record.get('messages', [])
                    )
                    session_with_tools += int(has_tools)
                    session_complete += int(
                        any(
                            message.get('role') == 'assistant'
                            for message in record.get('messages', [])
                            if isinstance(message, dict)
                        )
                    )
                destination.flush()
                os.fsync(destination.fileno())

                source_entry: dict[str, Any] = {
                    'provider': 'codex',
                    'installation': str(source.installation),
                    'root_label': source.root_label,
                    'source_class': _source_class(source.path),
                    'training_lane': _training_lane(source.path),
                    'source_file_name': source.path.name,
                    'source_file_sha256': inspection.source_sha256,
                    'source_bytes': inspection.size,
                    'session_id': inspection.session_meta.get('id'),
                    'parse_errors': inspection.parse_errors,
                    'unmatched_tool_calls': len(inspection.unmatched_tool_call_ids),
                    'emitted_records': session_record_count,
                    'status': (
                        'emitted' if session_record_count else 'no_emitted_messages'
                    ),
                }
                for key in ('model_provider', 'model', 'model_id', 'modelID', 'model_name'):
                    value = inspection.session_meta.get(key)
                    if value not in (None, ''):
                        source_entry[key] = value
                if admission is not None:
                    source_entry.update(
                        {
                            'source_manifest_revision': admission[
                                'source_manifest_revision'
                            ],
                            'source_ref_sha256': admission['source_ref_sha256'],
                            'source_snapshot_status': admission[
                                'source_snapshot_status'
                            ],
                        }
                    )
                source_entries.append(source_entry)
                records += session_record_count
                total_messages += session_messages
                with_tools += session_with_tools
                complete += session_complete
                completed_keys.add(source_key)
                completed_entries.append(
                    {
                        'root_label': source_key[0],
                        'source_class': source_key[1],
                        'provider': source_key[2],
                        'source_sha256': source_key[3],
                    }
                )
                save_checkpoint()

        missing_completed = completed_keys - seen_keys
        if missing_completed:
            raise ValueError('Codex checkpoint contains sources absent from resumed input')

        output_sha256, output_bytes = _file_digest_and_size(temporary)
        source_manifest_file = write_source_manifest(
            output_file.parent,
            timestamp,
            source_entries,
            source_manifest_revision=(
                source_manifest.revision if source_manifest is not None else None
            ),
            output_info={
                'name': output_file.name,
                'sha256': f'sha256:{output_sha256}',
                'bytes': output_bytes,
                'records': records,
            },
        )
        os.replace(temporary, output_file)
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)
    except BaseException:
        if checkpoint_path is None or not checkpoint_path.exists():
            temporary.unlink(missing_ok=True)
        raise

    return {
        'schema_version': 'ai-data-extraction/codex-ingress/v1',
        'extractor_version': EXTRACTOR_VERSION,
        'records': records,
        'messages': total_messages,
        'with_tools': with_tools,
        'complete': complete,
        'source_sessions': source_entries,
        'outputs': {
            output_file.name: {
                'sha256': f'sha256:{output_sha256}',
                'bytes': output_bytes,
                'records': records,
            }
        },
        'source_manifest_file': source_manifest_file.name,
        **(
            {'source_manifest_revision': source_manifest.revision}
            if source_manifest is not None
            else {}
        ),
    }


def write_source_manifest(
    output_dir: Path,
    timestamp: str,
    entries: list[dict[str, Any]],
    *,
    source_manifest_revision: str | None = None,
    output_info: dict[str, Any] | None = None,
) -> Path:
    """Publish metadata for every attempted source session, including empty ones."""
    manifest_file = output_dir / f'codex_source_manifest_{timestamp}.json'
    manifest = {
        'schema_version': 'ai-data-extraction/codex-source-manifest/v1',
        'extractor_version': EXTRACTOR_VERSION,
        'provider': 'codex',
        'include_backups': any(
            entry.get('source_class') == 'session_backup' for entry in entries
        ),
        'source_sessions': entries,
        'counts': {
            'source_files': len(entries),
            'emitted_files': sum(entry.get('emitted_records', 0) > 0 for entry in entries),
            'empty_files': sum(entry.get('status') == 'no_emitted_messages' for entry in entries),
            'read_errors': sum(entry.get('status') == 'read_error' for entry in entries),
        },
    }
    if source_manifest_revision is not None:
        manifest['source_manifest_revision'] = source_manifest_revision
    if output_info is not None:
        manifest['outputs'] = {output_info['name']: dict(output_info)}
    fd, temporary_name = tempfile.mkstemp(
        prefix=f'.{manifest_file.name}.', suffix='.tmp', dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
            destination.write('\n')
        os.replace(temporary, manifest_file)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest_file


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-home',
        type=Path,
        help='explicit immutable synthetic HOME or live home containing provider roots',
    )
    parser.add_argument(
        '--source-manifest',
        type=Path,
        help='validated source ledger required to admit Codex source files',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='output JSONL path; defaults to EXTRACTED_DATA_DIR/codex_conversations_<timestamp>.jsonl',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='checkpoint path; defaults to <output>.checkpoint.json',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume an unfinished checkpoint instead of creating a new output',
    )
    args = parser.parse_args(argv)
    ledger_index = (
        SourceManifestIndex.from_path(args.source_manifest)
        if args.source_manifest is not None
        else None
    )
    print("="*80)
    print("CODEX COMPLETE DATA EXTRACTION")
    print("="*80)
    print()

    # Find all Codex installations
    print("🔍 Searching for Codex installations...")
    if args.source_home is not None:
        installations = [
            (root.path, root.label)
            for root in discover_roots(args.source_home)
            if root.provider == 'codex'
        ]
    else:
        home_installations = find_codex_installations()
        installations = [
            (
                installation,
                'local' if installation.name == '.codex-local' else 'primary',
            )
            for installation in home_installations
        ]
    include_backups = os.environ.get('INCLUDE_CODEX_BACKUPS', '').lower() in {
        '1', 'true', 'yes', 'on'
    }

    if not installations:
        print("❌ No Codex installations found!")
        return

    print(f"✅ Found {len(installations)} installation(s):")
    for inst, _root_label in installations:
        print(f"   - {inst}")
    print()
    print(f"Backup sessions: {'included (explicit mode)' if include_backups else 'excluded (default)'}")

    try:
        max_output_chars = int(
            os.environ.get('CODEX_MAX_OUTPUT_RECORD_CHARS', str(MAX_OUTPUT_RECORD_CHARS))
        )
    except ValueError:
        print('❌ CODEX_MAX_OUTPUT_RECORD_CHARS must be an integer')
        return 2
    if max_output_chars <= 0:
        print('❌ CODEX_MAX_OUTPUT_RECORD_CHARS must be positive')
        return 2

    output_dir = Path(os.environ.get('EXTRACTED_DATA_DIR', 'extracted_data'))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or (
        output_dir / f'codex_conversations_{timestamp}.jsonl'
    )
    if args.resume and args.output is None:
        print('❌ --resume requires --output so the checkpoint target is unambiguous')
        return 2
    checkpoint_file = args.checkpoint or Path(
        str(output_file) + '.checkpoint.json'
    )
    if not args.resume and args.checkpoint is None and checkpoint_file.exists():
        print(f'❌ Checkpoint already exists: {checkpoint_file}; pass --resume or choose another output')
        return 2

    sources: list[CodexSourceSpec] = []
    seen_source_paths: set[Path] = set()
    installation_stats: dict[str, dict[str, int]] = {}
    for installation, root_label in installations:
        session_files = find_all_codex_sessions(
            installation,
            include_backups=include_backups,
        )
        for session_file in session_files:
            resolved = session_file.resolve()
            if resolved in seen_source_paths:
                continue
            seen_source_paths.add(resolved)
            sources.append(
                CodexSourceSpec(
                    path=session_file,
                    installation=installation,
                    root_label=root_label,
                )
            )
        installation_stats[str(installation)] = {
            'sessions': len(session_files),
            'records': 0,
        }

    try:
        print(f'📂 Processing {len(sources)} Codex source files')
        result = write_codex_records(
            sources,
            output_file=output_file,
            timestamp=timestamp,
            max_chars=max_output_chars,
            source_manifest=ledger_index,
            checkpoint_path=checkpoint_file,
            resume=args.resume,
        )
    except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
        print(f'❌ Extraction failed: {exc}')
        return 1

    total_records = result['records']
    total_messages = result['messages']
    with_tools = result['with_tools']
    complete = result['complete']
    source_sessions = sum(
        entry.get('emitted_records', 0) > 0
        for entry in result['source_sessions']
    )
    for entry in result['source_sessions']:
        installation = entry.get('installation')
        if installation in installation_stats:
            installation_stats[installation]['records'] += entry.get('emitted_records', 0)
    manifest_file = output_file.parent / result['source_manifest_file']

    print()
    print("="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Total conversations: {total_records:,}")
    print(f"Source sessions with data: {source_sessions:,}")
    print(f"Complete conversations: {complete:,}")
    print(f"Total messages: {total_messages:,}")
    print(f"With tool use/diffs: {with_tools:,}")
    print()

    print("Breakdown by installation:")
    for inst, counts in sorted(
        installation_stats.items(), key=lambda item: -item[1]['records']
    ):
        print(
            f"  {Path(inst).name:20} {counts['sessions']:5,} sessions -> "
            f"{counts['records']:5,} records"
        )
    print()

    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"✅ Saved to: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print(
        "   Format: bounded JSONL records (one or more per source session); "
        f"target {max_output_chars:,} characters"
    )
    print(f"✅ Source manifest: {manifest_file}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
