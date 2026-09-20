#!/usr/bin/env python3
"""Extract Prime Agent, Pi, and Oh My Pi JSONL sessions.

These stores use the same general event-log shape, but they are not safe to
copy directly into a trainer.  This adapter keeps visible user/assistant/tool
messages, removes thinking blocks at ingress, bounds tool payloads, and keeps
harness/advisor telemetry as non-training metadata.

Source-selection lanes and quality gates are deliberately independent.  Prime
and ordinary Pi/Oh My Pi sessions keep an ``optional_alt`` provenance lane by
default, but the quality gate is assessed per session from structural and
harness evidence.  A local model is a provenance fact and a review flag, not a
global quality verdict.  Files named ``__advisor.jsonl`` are quarantined for
contamination review because advisor output can alter the trajectory; they
remain available for a separately manifested experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from source_manifest import (
    SourceAdmissionError,
    SourceManifestIndex,
    validate_source_admission,
)


EXTRACTOR_VERSION = "1.2.0"
DEFAULT_MAX_RECORD_CHARS = int(os.environ.get("AGENT_SESSION_MAX_RECORD_CHARS", "250000"))
INGRESS_TARGET_FRACTION = 0.25
TEXT_MAX_CHARS = 32_000
TOOL_ARGUMENT_MAX_CHARS = 8_000
TOOL_OBSERVATION_MAX_CHARS = 12_000
QUALITY_GATES = frozenset({"candidate", "review_required", "quarantine", "unassessed"})
SESSION_CHECKPOINT_SCHEMA = "ai-data-extraction/agent-session-checkpoint/v1"


@dataclass(frozen=True)
class SessionSpec:
    provider: str
    source_class: str
    training_lane: str
    quality_gate: str | None
    path: Path
    root_label: str


@dataclass
class Inspection:
    source_sha256: str
    session_id: str | None
    session_metadata: dict[str, Any]
    model_counts: Counter[str] = field(default_factory=Counter)
    provider_counts: Counter[str] = field(default_factory=Counter)
    event_counts: Counter[str] = field(default_factory=Counter)
    control_counts: Counter[str] = field(default_factory=Counter)
    tool_names: Counter[str] = field(default_factory=Counter)
    parse_errors: int = 0
    message_count: int = 0
    omitted_reasoning_blocks: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    tool_message_count: int = 0
    tool_call_count: int = 0
    tool_observation_count: int = 0
    matched_tool_observation_count: int = 0
    unmatched_tool_observation_count: int = 0
    unmatched_tool_call_count: int = 0
    tool_error_count: int = 0
    truncated_payload_count: int = 0
    terminal_signal_count: int = 0
    size: int = 0
    mtime_ns: int = 0


@dataclass
class MessageItem:
    message: dict[str, Any]
    message_index: int
    source_line: int
    truncations: list[dict[str, Any]]
    omitted_reasoning: int = 0


@dataclass
class Chunk:
    messages: list[dict[str, Any]]
    message_start: int
    message_end: int
    source_line_start: int
    source_line_end: int
    open_tool_call_ids: tuple[str, ...]
    observation_truncations: list[dict[str, Any]]
    omitted_reasoning: int
    anchor_message_index: int | None
    cut_reason: str | None


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _sha256_file(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return digest.hexdigest(), stat.st_size, stat.st_mtime_ns


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _short_token(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _bounded(
    value: Any,
    *,
    max_chars: int,
    kind: str,
    source_line: int,
    truncations: list[dict[str, Any]],
    call_id: Any = None,
) -> Any:
    """Bound visible payloads while retaining a recoverable source digest."""
    if value is None:
        return None
    if isinstance(value, str):
        serialized = value
    else:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    original_chars = len(serialized)
    if original_chars <= max_chars:
        return value

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if isinstance(value, str):
        marker = f"\n<AGENT_PAYLOAD_TRUNCATED original_chars={original_chars} sha256={digest} policy=head_tail>\n"
        available = max(2, max_chars - len(marker))
        head = max(1, int(available * 0.70))
        tail = max(1, available - head)
        bounded = value[:head] + marker + value[-tail:]
        kept_chars = len(bounded)
        policy = "head_tail"
    else:
        bounded = {"truncated": True, "original_chars": original_chars, "sha256": digest}
        kept_chars = _json_size(bounded)
        policy = "digest_only"
    entry: dict[str, Any] = {
        "kind": kind,
        "source_line": source_line,
        "original_chars": original_chars,
        "kept_chars": kept_chars,
        "sha256": digest,
        "policy": policy,
    }
    if call_id is not None:
        entry["call_id_sha256"] = hashlib.sha256(str(call_id).encode("utf-8")).hexdigest()
    truncations.append(entry)
    return bounded


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, dict):
        return [content]
    return []


def _visible_text(content: Any) -> str:
    """Return only visible text; never copy thinking/analysis blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                part_type = str(item.get("type", "")).lower().replace("-", "_")
                if part_type in {"thinking", "analysis", "reasoning", "thought"}:
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        part_type = str(content.get("type", "")).lower().replace("-", "_")
        if part_type in {"thinking", "analysis", "reasoning", "thought"}:
            return ""
        for key in ("text", "content", "message", "value"):
            if key in content:
                return _visible_text(content[key])
    return ""


def _tool_call_block(
    block: dict[str, Any],
    *,
    source_line: int,
    truncations: list[dict[str, Any]],
) -> dict[str, Any]:
    call_id = block.get("id") or block.get("callId") or block.get("call_id")
    name = block.get("name") or block.get("toolName") or "unknown"
    arguments = block.get("arguments", block.get("input", block.get("partialArgs", {})))
    bounded_arguments = _bounded(
        arguments,
        max_chars=TOOL_ARGUMENT_MAX_CHARS,
        kind="tool_input",
        source_line=source_line,
        truncations=truncations,
        call_id=call_id,
    )
    if not isinstance(bounded_arguments, str):
        bounded_arguments = json.dumps(
            bounded_arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": str(name), "arguments": bounded_arguments},
    }


def _normalize_message(
    payload: dict[str, Any],
    *,
    source_line: int,
) -> MessageItem | None:
    raw_role = payload.get("role")
    role = {"toolResult": "tool", "agent": "assistant", "model": "assistant"}.get(
        raw_role, raw_role
    )
    if role not in {"user", "assistant", "tool", "system", "developer"}:
        return None
    content = payload.get("content", "")
    truncations: list[dict[str, Any]] = []
    omitted_reasoning = 0
    blocks = _content_blocks(content)
    tool_calls: list[dict[str, Any]] = []
    visible_parts: list[str] = []
    for block in blocks:
        part_type = str(block.get("type", "")).lower().replace("-", "_")
        if part_type in {"thinking", "analysis", "reasoning", "thought"}:
            omitted_reasoning += 1
            continue
        if part_type in {"toolcall", "tool_call", "tool_use", "function_call"}:
            tool_calls.append(
                _tool_call_block(block, source_line=source_line, truncations=truncations)
            )
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            visible_parts.append(text)
    if not blocks:
        text = _visible_text(content)
        if text:
            visible_parts.append(text)
    text = "\n".join(part for part in visible_parts if part).strip()
    if role == "tool":
        text = _visible_text(content)
    if text:
        text = _bounded(
            text,
            max_chars=TOOL_OBSERVATION_MAX_CHARS if role == "tool" else TEXT_MAX_CHARS,
            kind="tool_observation" if role == "tool" else "message_text",
            source_line=source_line,
            truncations=truncations,
            call_id=payload.get("toolCallId"),
        )
    if role == "tool":
        message: dict[str, Any] = {
            "role": "tool",
            "content": text or "",
            "tool_call_id": payload.get("toolCallId") or payload.get("tool_call_id"),
        }
        if payload.get("toolName"):
            message["tool_name"] = payload["toolName"]
        if payload.get("isError") is True:
            message["status"] = "error"
        else:
            message["status"] = "success"
    else:
        message = {"role": "system" if role == "developer" else role, "content": text or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if role == "assistant":
            for key in ("model", "provider", "api", "stopReason"):
                if payload.get(key) not in (None, ""):
                    message[key] = payload[key]
    if not message.get("content") and not message.get("tool_calls") and role != "tool":
        return None
    return MessageItem(
        message=message,
        message_index=-1,
        source_line=source_line,
        truncations=truncations,
        omitted_reasoning=omitted_reasoning,
    )


def _iter_json_objects(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, exc.msg
                continue
            yield line_number, value if isinstance(value, dict) else None, None


def _metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _metadata_value(v) for k, v in value.items() if str(k).lower() not in {"prompt", "objective", "rationale", "summary", "message"}}
    if isinstance(value, list):
        return [_metadata_value(item) for item in value[:32]]
    return str(value)


def inspect_session(path: Path) -> Inspection:
    digest = hashlib.sha256()
    stat_before = path.stat()
    session_id: str | None = None
    session_metadata: dict[str, Any] = {}
    inspection = Inspection("", None, {})
    pending_tool_call_ids: set[str] = set()
    observed_tool_call_ids: set[str] = set()
    with path.open("rb") as source:
        for raw_line in source:
            digest.update(raw_line)
            try:
                obj = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                inspection.parse_errors += 1
                continue
            if not isinstance(obj, dict):
                continue
            event_type = _short_token(obj.get("type")) or "unknown"
            inspection.event_counts[event_type] += 1
            if event_type in {"custom", "custom_message", "agent_status", "git_state", "session_state", "compaction", "thinking_level_change", "model_change", "service_tier_change", "child_usage_attributed"}:
                inspection.control_counts[event_type] += 1
            if event_type == "session":
                header_id = obj.get("id")
                if header_id is not None:
                    session_id = str(header_id)
                for key in ("id", "timestamp", "version", "rlmDepth"):
                    if obj.get(key) not in (None, ""):
                        session_metadata[key] = _metadata_value(obj[key])
            if event_type in {"model", "model_change", "session_init"}:
                model = obj.get("modelId") or obj.get("model") or obj.get("modelName")
                if model:
                    inspection.model_counts[str(model)] += 1
                provider = obj.get("provider") or obj.get("api")
                if provider:
                    inspection.provider_counts[str(provider)] += 1
            if event_type == "message" and isinstance(obj.get("message"), dict):
                payload = obj["message"]
                role = payload.get("role")
                inspection.event_counts[f"message:{role}"] += 1
                if role in {"user", "assistant", "toolResult", "tool", "developer"}:
                    inspection.message_count += 1
                normalized = _normalize_message(
                    payload,
                    source_line=0,
                )
                if normalized is not None:
                    normalized_role = normalized.message.get("role")
                    if normalized_role == "user":
                        inspection.user_message_count += 1
                    elif normalized_role == "assistant":
                        inspection.assistant_message_count += 1
                    elif normalized_role == "tool":
                        inspection.tool_message_count += 1
                    inspection.truncated_payload_count += len(normalized.truncations)
                    action_ids = _action_ids(normalized.message)
                    inspection.tool_call_count += len(action_ids)
                    observed = _observation_id(normalized.message)
                    if observed is not None:
                        inspection.tool_observation_count += 1
                        if observed in pending_tool_call_ids:
                            pending_tool_call_ids.discard(observed)
                            observed_tool_call_ids.add(observed)
                            inspection.matched_tool_observation_count += 1
                        else:
                            inspection.unmatched_tool_observation_count += 1
                    pending_tool_call_ids.update(action_ids)
                    if normalized.message.get("status") == "error":
                        inspection.tool_error_count += 1
                model = payload.get("model") or payload.get("responseModel")
                if model:
                    inspection.model_counts[str(model)] += 1
                provider = payload.get("provider") or payload.get("api")
                if provider:
                    inspection.provider_counts[str(provider)] += 1
                for block in _content_blocks(payload.get("content")):
                    part_type = str(block.get("type", "")).lower().replace("-", "_")
                    if part_type in {"thinking", "analysis", "reasoning", "thought"}:
                        inspection.omitted_reasoning_blocks += 1
                    if part_type in {"toolcall", "tool_call", "tool_use", "function_call"}:
                        name = block.get("name") or block.get("toolName")
                        if name:
                            inspection.tool_names[str(name)] += 1
            if event_type == "custom":
                custom_type = _short_token(obj.get("customType")) or "unknown"
                inspection.control_counts[f"custom:{custom_type}"] += 1
                if custom_type in {"session_exit", "session_end", "turn_end", "task_complete"}:
                    inspection.terminal_signal_count += 1
                data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
                if custom_type == "tool_execution_start" and isinstance(data, dict):
                    name = data.get("toolName") or data.get("name")
                    if name:
                        inspection.tool_names[str(name)] += 1
            if event_type in {"session_end", "session_exit", "turn_end", "task_complete"}:
                inspection.terminal_signal_count += 1
    stat_after = path.stat()
    if stat_before.st_size != stat_after.st_size or stat_before.st_mtime_ns != stat_after.st_mtime_ns:
        raise RuntimeError(f"source changed during inspection: {path.name}")
    inspection.source_sha256 = digest.hexdigest()
    inspection.session_id = session_id
    inspection.session_metadata = session_metadata
    inspection.size = stat_after.st_size
    inspection.mtime_ns = stat_after.st_mtime_ns
    inspection.unmatched_tool_call_count = len(
        pending_tool_call_ids - observed_tool_call_ids
    )
    return inspection


def _iter_message_items(path: Path) -> Iterator[MessageItem]:
    message_index = 0
    for source_line, obj, _error in _iter_json_objects(path):
        if not isinstance(obj, dict) or obj.get("type") != "message":
            continue
        payload = obj.get("message")
        if not isinstance(payload, dict):
            continue
        item = _normalize_message(payload, source_line=source_line)
        if item is None:
            continue
        item.message_index = message_index
        message_index += 1
        yield item


def _action_ids(message: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
        if isinstance(call, dict):
            value = call.get("id") or call.get("call_id")
            if value is not None:
                result.add(str(value))
    return result


def _observation_id(message: dict[str, Any]) -> str | None:
    value = message.get("tool_call_id") or message.get("call_id")
    return str(value) if value is not None else None


def _chunk_stream(path: Path, *, max_record_chars: int) -> Iterator[Chunk]:
    target = max(1, int(max_record_chars * INGRESS_TARGET_FRACTION))
    # A pending call with no matching observation must not hold an entire
    # session hostage.  This ceiling is deliberately below the final bound;
    # the lineage exposes the open IDs so the cut is reviewable.
    hard_limit = max(target, int(max_record_chars * 0.55))
    prefix: list[dict[str, Any]] = []
    prefix_size = 0
    seen_non_system = False
    current: list[dict[str, Any]] = []
    current_size = 0
    current_start: int | None = None
    current_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    has_user = False
    has_assistant = False
    last_role: str | None = None
    last_user: tuple[dict[str, Any], int] | None = None
    anchor: int | None = None
    pending: set[str] = set()
    truncations: list[dict[str, Any]] = []
    omitted_reasoning = 0

    def reset(continuation: tuple[dict[str, Any], int] | None = None) -> None:
        nonlocal current, current_size, current_start, current_end, line_start, line_end
        nonlocal has_user, has_assistant, last_role, last_user, anchor, pending
        nonlocal truncations, omitted_reasoning
        current = list(prefix)
        current_size = prefix_size
        current_start = None
        current_end = None
        line_start = None
        line_end = None
        has_user = False
        has_assistant = False
        last_role = "system" if prefix else None
        last_user = None
        anchor = None
        pending = set()
        truncations = []
        omitted_reasoning = 0
        if continuation is not None:
            message, index = continuation
            current.append(message)
            current_size += _json_size(message)
            has_user = True
            last_role = "user"
            last_user = continuation
            anchor = index

    def take(cut_reason: str | None = None) -> Chunk:
        nonlocal current, current_size, current_start, current_end, line_start, line_end
        start = current_start if current_start is not None else 0
        end = current_end if current_end is not None else start
        result = Chunk(
            messages=list(current),
            message_start=start,
            message_end=end,
            source_line_start=line_start if line_start is not None else 1,
            source_line_end=line_end if line_end is not None else (line_start if line_start is not None else 1),
            open_tool_call_ids=tuple(sorted(pending)),
            observation_truncations=list(truncations),
            omitted_reasoning=omitted_reasoning,
            anchor_message_index=anchor,
            cut_reason=cut_reason,
        )
        reset()
        return result

    reset()
    for item in _iter_message_items(path):
        message = item.message
        role = message.get("role")
        size = _json_size(message)
        if not seen_non_system and role == "system":
            prefix.append(message)
            prefix_size += size
            current.append(message)
            current_size += size
            line_start = item.source_line if line_start is None else line_start
            line_end = item.source_line
            omitted_reasoning += item.omitted_reasoning
            continue
        seen_non_system = True
        split_before = False
        cut_reason: str | None = None
        if current and current_size + size > target and not pending:
            if role == "user":
                split_before = True
            elif role != "tool" and last_role in {"assistant", "tool"}:
                split_before = True
            elif role == "tool" and last_role == "tool":
                # Subagent and recovery logs can contain a tool-only stream.
                # Once no call is pending, each completed observation is a
                # safe bounded cut even without a nearby user turn.
                split_before = True
            if split_before:
                cut_reason = "token_budget"
        if not split_before and current and current_size + size > hard_limit:
            # The normal path keeps calls adjacent to observations.  If a
            # provider leaves calls open indefinitely, bounded lineage is the
            # safer invariant; the next segment explicitly carries the
            # continuation and the previous segment's open IDs.
            split_before = True
            cut_reason = "pending_tool_size_bound" if pending else "hard_size_bound"
        if split_before and current:
            continuation = last_user if role == "assistant" and last_user is not None else None
            yield take(cut_reason)
            reset(continuation)
        if current_start is None:
            current_start = item.message_index
        current_end = item.message_index
        line_start = item.source_line if line_start is None else line_start
        line_end = item.source_line
        current.append(message)
        current_size += size
        truncations.extend(item.truncations)
        omitted_reasoning += item.omitted_reasoning
        has_user = has_user or role == "user"
        has_assistant = has_assistant or role == "assistant"
        last_role = role
        if role == "user":
            last_user = (message, item.message_index)
        observed = _observation_id(message)
        if observed is not None:
            pending.discard(observed)
        pending.update(_action_ids(message))
    if current and (len(current) > len(prefix) or not prefix):
        yield take()


def _model_provenance(inspection: Inspection) -> str:
    names = [name.lower() for name in inspection.model_counts]
    providers = [name.lower() for name in inspection.provider_counts]
    local_tokens = (
        "local",
        "ollama",
        "llama.cpp",
        "llamacpp",
        "lmstudio",
        "self-host",
        "self_host",
    )
    if any(token in value for value in (*names, *providers) for token in local_tokens):
        return "local_or_self_hosted"
    if len(inspection.model_counts) > 1 or len(inspection.provider_counts) > 1:
        return "mixed_or_routed"
    if names or providers:
        return "identified_non_local_or_unknown"
    return "unknown"


def assess_session_quality(
    spec: SessionSpec,
    inspection: Inspection,
    *,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess one session without using provider identity as a quality label.

    ``candidate`` means structurally usable for review; it does not mean
    verified success or privacy approval.  Tool-trace and outcome evidence
    remain separate dimensions so a session can be useful for SFT while being
    ineligible for verifier-backed RL.
    """

    flags: list[str] = []
    if spec.source_class in {"pi_advisor_overlay", "pi_advisor_overlay_backup"}:
        flags.append("advisor_overlay_contamination")
    if inspection.parse_errors:
        flags.append("source_parse_errors")
    if inspection.user_message_count == 0:
        flags.append("missing_user_objective")
    if inspection.assistant_message_count == 0:
        flags.append("missing_visible_assistant_response")
    if inspection.tool_call_count and inspection.tool_observation_count == 0:
        flags.append("missing_tool_observations")
    if inspection.unmatched_tool_call_count:
        flags.append("unmatched_tool_calls")
    if inspection.unmatched_tool_observation_count:
        flags.append("unmatched_tool_observations")
    if inspection.truncated_payload_count:
        flags.append("payload_truncated")
    if inspection.tool_error_count:
        flags.append("tool_error_observed")
    if inspection.terminal_signal_count == 0:
        flags.append("outcome_unverified")

    model_provenance = _model_provenance(inspection)
    if model_provenance == "local_or_self_hosted":
        flags.append("model_provenance_local_or_self_hosted")
    elif model_provenance == "unknown":
        flags.append("model_provenance_unknown")

    if inspection.message_count == 0:
        gate = "quarantine"
    elif inspection.parse_errors or "advisor_overlay_contamination" in flags:
        gate = "quarantine"
    elif any(
        flag in flags
        for flag in (
            "missing_user_objective",
            "missing_visible_assistant_response",
            "missing_tool_observations",
            "unmatched_tool_calls",
            "unmatched_tool_observations",
            "payload_truncated",
        )
    ):
        gate = "review_required"
    else:
        gate = "candidate"

    assessment: dict[str, Any] = {
        "assessment_version": "session-quality/v1",
        "gate": gate,
        "flags": sorted(set(flags)),
        "dimensions": {
            "model_provenance": {
                "status": model_provenance,
                "models": dict(sorted(inspection.model_counts.items())),
                "providers": dict(sorted(inspection.provider_counts.items())),
            },
            "dialogue": {
                "user_messages": inspection.user_message_count,
                "assistant_messages": inspection.assistant_message_count,
                "tool_messages": inspection.tool_message_count,
            },
            "tool_trace_integrity": {
                "calls": inspection.tool_call_count,
                "observations": inspection.tool_observation_count,
                "matched_observations": inspection.matched_tool_observation_count,
                "unmatched_calls": inspection.unmatched_tool_call_count,
                "unmatched_observations": inspection.unmatched_tool_observation_count,
                "errors": inspection.tool_error_count,
            },
            "payload_integrity": {
                "truncated_payloads": inspection.truncated_payload_count,
            },
            "outcome_evidence": {
                "terminal_signals": inspection.terminal_signal_count,
                "status": "observed" if inspection.terminal_signal_count else "unverified",
            },
        },
        "method": "deterministic_structural_session_v1",
        "provider_neutral": True,
    }

    requested_gate = None
    if isinstance(override, dict):
        requested_gate = override.get("quality_gate") or override.get("gate")
    if requested_gate is None and spec.quality_gate in QUALITY_GATES:
        requested_gate = spec.quality_gate
    if requested_gate is not None:
        if requested_gate not in QUALITY_GATES:
            raise ValueError(
                "quality override must be one of: " + ", ".join(sorted(QUALITY_GATES))
            )
        assessment["automatic_gate"] = assessment["gate"]
        assessment["gate"] = requested_gate
        assessment["override"] = {
            key: _metadata_value(value)
            for key, value in (override or {}).items()
            if key in {"quality_gate", "gate", "reason", "reviewer", "ticket"}
            and value not in (None, "")
        }
        if not assessment["override"] and spec.quality_gate is not None:
            assessment["override"] = {
                "quality_gate": requested_gate,
                "reason": "explicit_session_spec_override",
            }
    return assessment


def _quality_summary(
    spec: SessionSpec,
    inspection: Inspection,
    assessment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_adapter": "prime-pi-jsonl",
        "extractor_version": EXTRACTOR_VERSION,
        "training_lane": spec.training_lane,
        "quality_gate": assessment["gate"],
        "quality_flags": assessment["flags"],
        "quality_assessment": assessment,
        "advisor_overlay": spec.source_class in {"pi_advisor_overlay", "pi_advisor_overlay_backup"},
        "model_counts": dict(sorted(inspection.model_counts.items())),
        "provider_counts": dict(sorted(inspection.provider_counts.items())),
        "event_counts": dict(sorted(inspection.event_counts.items())),
        "control_event_counts": dict(sorted(inspection.control_counts.items())),
        "tool_names": sorted(inspection.tool_names),
        "parse_errors": inspection.parse_errors,
        "source_message_count": inspection.message_count,
        "reasoning_blocks_omitted_at_ingress": inspection.omitted_reasoning_blocks,
        "tool_call_count": inspection.tool_call_count,
        "tool_observation_count": inspection.tool_observation_count,
        "matched_tool_observation_count": inspection.matched_tool_observation_count,
        "unmatched_tool_call_count": inspection.unmatched_tool_call_count,
        "truncated_payload_count": inspection.truncated_payload_count,
    }


def iter_session_records(
    spec: SessionSpec,
    *,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
    inspection: Inspection | None = None,
    quality_assessment: dict[str, Any] | None = None,
    quality_override: dict[str, Any] | None = None,
    source_manifest: SourceManifestIndex | None = None,
    source_admission: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    if max_record_chars <= 0:
        raise ValueError("max_record_chars must be positive")
    inspection = inspection or inspect_session(spec.path)
    if source_manifest is not None:
        if source_admission is not None:
            raise ValueError("source_manifest and source_admission are mutually exclusive")
        source_admission = source_manifest.admit(
            source_sha256=inspection.source_sha256,
            provider=spec.provider,
            root_label=spec.root_label,
            source_class=spec.source_class,
        )
    if source_admission is not None:
        source_admission = _validate_source_admission(inspection, source_admission)
    quality_assessment = quality_assessment or assess_session_quality(
        spec,
        inspection,
        override=quality_override,
    )
    quality_summary = _quality_summary(spec, inspection, quality_assessment)
    # Count first, then stream the same bounded segmentation a second time.
    # This keeps a year-long session from being materialized merely to stamp
    # deterministic chunk_count metadata.
    chunk_count = sum(1 for _ in _chunk_stream(spec.path, max_record_chars=max_record_chars))
    if chunk_count == 0:
        return
    parent_hash = _sha256_json(
        {
            "source_sha256": inspection.source_sha256,
            "session_id": inspection.session_id or spec.path.stem,
            "provider": spec.provider,
            "source_class": spec.source_class,
            "extractor_version": EXTRACTOR_VERSION,
        }
    )
    model_names = sorted(inspection.model_counts)
    record_model = model_names[0] if len(model_names) == 1 else ("mixed" if model_names else None)
    for index, chunk in enumerate(
        _chunk_stream(spec.path, max_record_chars=max_record_chars)
    ):
        origin = {
            "store_type": "jsonl_session",
            "source_file_name": spec.path.name,
            "source_file_sha256": f"sha256:{inspection.source_sha256}",
            "source_size_bytes": inspection.size,
            "source_root_label": spec.root_label,
            "read_mode": "streaming_stable",
            "ordering": "source_file_line_order",
            "content_policy": "thinking_omitted; tool_payloads_bounded; control_events_metadata_only",
            "message_line_range": {
                "start": chunk.source_line_start,
                "end": chunk.source_line_end,
            },
            "source_kind": spec.source_class,
            "session_header": inspection.session_metadata,
            "quality_summary": quality_summary,
        }
        if source_admission is not None:
            origin.update(
                {
                    "source_manifest_revision": source_admission[
                        "source_manifest_revision"
                    ],
                    "source_ref_sha256": source_admission["source_ref_sha256"],
                    "source_snapshot_status": source_admission["source_snapshot_status"],
                }
            )
        record: dict[str, Any] = {
            "source": spec.provider,
            "source_class": spec.source_class,
            "training_lane": spec.training_lane,
            "quality_gate": quality_assessment["gate"],
            "quality_flags": quality_assessment["flags"],
            "quality_assessment": quality_assessment,
            "session_id": inspection.session_id or spec.path.stem,
            "messages": chunk.messages,
            "source_origin": origin,
            "harness_summary": quality_summary,
            "_chunk_parent_record_sha256": parent_hash,
            "_chunk_index": index,
            "_chunk_count": chunk_count,
            "_chunk_message_start": chunk.message_start,
            "_chunk_message_end": chunk.message_end,
            "_chunk_cut_reason": "token_budget" if chunk_count > 1 else None,
            "_open_tool_call_ids": list(chunk.open_tool_call_ids),
            "observation_truncations": chunk.observation_truncations,
            "reasoning_omitted_count": chunk.omitted_reasoning,
        }
        if record_model is not None:
            record["model"] = record_model
        if chunk.anchor_message_index is not None:
            record["_chunk_anchor_message_index"] = chunk.anchor_message_index
        if chunk.cut_reason is not None:
            record["_chunk_cut_reason"] = chunk.cut_reason
        yield record
    digest, size, mtime_ns = _sha256_file(spec.path)
    if digest != inspection.source_sha256 or size != inspection.size or mtime_ns != inspection.mtime_ns:
        raise RuntimeError(f"source changed during bounded extraction: {spec.path.name}")


def _is_advisor(path: Path) -> bool:
    return path.name == "__advisor.jsonl"


def discover_session_specs(home: Path | None = None) -> list[SessionSpec]:
    home = (home or Path.home()).expanduser()
    specs: list[SessionSpec] = []
    seen: set[Path] = set()

    def add(path: Path, spec: SessionSpec) -> None:
        if not path.is_file() or path.suffix.lower() != ".jsonl":
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        specs.append(spec)

    prime_sessions = home / ".prime" / "agent" / "sessions"
    if prime_sessions.is_dir():
        for path in sorted(prime_sessions.glob("*.jsonl")):
            add(
                path,
                SessionSpec("prime-agent", "prime_session_active", "optional_alt", None, path, "primary"),
            )
    prime_artifacts = home / ".prime" / "agent" / "session-artifacts"
    if prime_artifacts.is_dir():
        for path in sorted(prime_artifacts.rglob("*.jsonl")):
            add(
                path,
                SessionSpec("prime-agent", "prime_subagent_session", "optional_alt", None, path, "session-artifacts"),
            )

    omp = home / ".omp" / "agent" / "sessions"
    if omp.is_dir():
        for path in sorted(omp.rglob("*.jsonl")):
            advisor = _is_advisor(path)
            add(
                path,
                SessionSpec(
                    "oh-my-pi",
                    "pi_advisor_overlay" if advisor else "pi_session_active",
                    "quarantine" if advisor else "optional_alt",
                    None,
                    path,
                    "primary",
                ),
            )

    backups = home / ".omp-backups"
    if backups.is_dir():
        for path in sorted(backups.rglob("*.jsonl")):
            parts = path.parts
            try:
                session_root = parts.index(".omp")
            except ValueError:
                continue
            if tuple(parts[session_root + 1 : session_root + 3]) != ("agent", "sessions"):
                continue
            advisor = _is_advisor(path)
            add(
                path,
                SessionSpec(
                    "oh-my-pi",
                    "pi_advisor_overlay_backup" if advisor else "pi_session_backup",
                    "quarantine" if advisor else "optional_alt",
                    None,
                    path,
                    "backup",
                ),
            )

    # Keep a separate Pi path available if the user has installed Pi without
    # the Oh My Pi wrapper.  These narrow locations avoid walking unrelated
    # configuration trees.
    for candidate in (
        home / ".pi" / "agent" / "sessions",
        home / ".pi" / "sessions",
        home / ".local" / "share" / "pi" / "sessions",
    ):
        if not candidate.is_dir():
            continue
        for path in sorted(candidate.rglob("*.jsonl")):
            add(
                path,
                SessionSpec("pi-agent", "pi_session_active", "optional_alt", None, path, "primary"),
            )
    return specs


def load_quality_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load explicit per-session quality decisions without changing lanes."""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    entries = value.get("overrides") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise ValueError("quality override file must contain an overrides list")
    overrides: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each quality override must be an object")
        gate = entry.get("quality_gate") or entry.get("gate")
        if gate not in QUALITY_GATES:
            raise ValueError(
                "quality override gate must be one of: "
                + ", ".join(sorted(QUALITY_GATES))
            )
        keys = [
            entry.get("source_file_sha256"),
            entry.get("session_id"),
        ]
        keys = [str(key) for key in keys if key not in (None, "")]
        if not keys:
            raise ValueError("quality override needs source_file_sha256 or session_id")
        override = {
            key: _metadata_value(item)
            for key, item in entry.items()
            if key in {"quality_gate", "gate", "reason", "reviewer", "ticket"}
            and item not in (None, "")
        }
        for key in keys:
            overrides[key] = override
            if key.startswith("sha256:"):
                overrides[key[7:]] = override
    return overrides


def _quality_override_for(
    overrides: dict[str, dict[str, Any]],
    inspection: Inspection,
) -> dict[str, Any] | None:
    return overrides.get(f"sha256:{inspection.source_sha256}") or overrides.get(
        inspection.source_sha256
    ) or (
        overrides.get(inspection.session_id) if inspection.session_id else None
    )


def _validate_source_admission(
    inspection: Inspection,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_source_admission(
        admission,
        source_sha256=inspection.source_sha256,
        source_bytes=inspection.size,
    )


def _source_key(spec: SessionSpec, inspection: Inspection) -> tuple[str, str, str, str]:
    return (
        spec.provider,
        spec.root_label,
        spec.source_class,
        inspection.source_sha256,
    )


def _load_session_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load session checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SESSION_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported session checkpoint: {path}")
    return value


def write_records(
    specs: Iterable[SessionSpec],
    *,
    output_dir: Path,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
    timestamp: str | None = None,
    quality_overrides: dict[str, dict[str, Any]] | None = None,
    source_manifest: SourceManifestIndex | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if resume and checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_state: dict[str, Any] | None = None
    if checkpoint_path is not None:
        if resume:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"session checkpoint does not exist: {checkpoint_path}")
            checkpoint_state = _load_session_checkpoint(checkpoint_path)
            expected_revision = source_manifest.revision if source_manifest is not None else None
            if checkpoint_state.get("source_manifest_revision") != expected_revision:
                raise ValueError("session checkpoint source-manifest revision does not match")
        elif checkpoint_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite session checkpoint {checkpoint_path}; pass resume"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    files: dict[str, Any] = {}
    by_lane: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    records = 0
    sessions = 0
    errors: list[dict[str, str]] = []
    handles: dict[str, tuple[Path, Path]] = {}
    source_sessions: list[dict[str, Any]] = []
    quality_overrides = quality_overrides or {}
    completed_keys: set[tuple[str, str, str, str]] = set()
    if checkpoint_state is not None:
        raw_outputs = checkpoint_state.get("outputs")
        raw_sessions = checkpoint_state.get("source_sessions")
        raw_completed = checkpoint_state.get("completed_sources")
        if not isinstance(raw_outputs, dict) or not isinstance(raw_sessions, list) or not isinstance(raw_completed, list):
            raise ValueError("session checkpoint has invalid state")
        for name, raw_output in raw_outputs.items():
            if not isinstance(name, str) or not isinstance(raw_output, dict):
                raise ValueError("session checkpoint output state is invalid")
            target_name = raw_output.get("target_name", name)
            temporary_name = raw_output.get("temporary_name")
            if (
                not isinstance(target_name, str)
                or not isinstance(temporary_name, str)
                or Path(target_name).name != target_name
                or Path(temporary_name).name != temporary_name
            ):
                raise ValueError("session checkpoint output paths are invalid")
            target = output_dir / target_name
            temporary = output_dir / temporary_name
            if target.exists() or not temporary.is_file():
                raise ValueError("session checkpoint output files are not resumable")
            expected_bytes = raw_output.get("temporary_bytes")
            if not isinstance(expected_bytes, int) or temporary.stat().st_size != expected_bytes:
                raise ValueError("session checkpoint temporary output changed")
            handles[name] = (target, temporary)
        source_sessions = [dict(item) for item in raw_sessions if isinstance(item, dict)]
        if len(source_sessions) != len(raw_sessions):
            raise ValueError("session checkpoint source-session state is invalid")
        for item in raw_completed:
            if not isinstance(item, dict):
                raise ValueError("session checkpoint completion state is invalid")
            fields = (item.get("provider"), item.get("root_label"), item.get("source_class"), item.get("source_sha256"))
            if not all(isinstance(field, str) and field for field in fields):
                raise ValueError("session checkpoint completion identity is invalid")
            completed_keys.add(fields)  # type: ignore[arg-type]
        records = int(checkpoint_state.get("records", 0))
        sessions = int(checkpoint_state.get("sessions", 0))
        by_lane.update(checkpoint_state.get("training_lane_records", {}))
        by_class.update(checkpoint_state.get("source_class_records", {}))

    def save_checkpoint(completed: list[dict[str, str]]) -> None:
        if checkpoint_path is None:
            return
        checkpoint = {
            "schema_version": SESSION_CHECKPOINT_SCHEMA,
            "extractor_version": EXTRACTOR_VERSION,
            "source_manifest_revision": source_manifest.revision if source_manifest is not None else None,
            "records": records,
            "sessions": sessions,
            "training_lane_records": dict(sorted(by_lane.items())),
            "source_class_records": dict(sorted(by_class.items())),
            "source_sessions": source_sessions,
            "completed_sources": completed,
            "outputs": {
                name: {
                    "target_name": target.name,
                    "temporary_name": temporary.name,
                    "temporary_bytes": temporary.stat().st_size,
                }
                for name, (target, temporary) in handles.items()
            },
        }
        write_manifest(checkpoint_path, checkpoint)

    completed_records: list[dict[str, str]] = []
    if checkpoint_state is not None:
        completed_records = [
            dict(item)
            for item in checkpoint_state.get("completed_sources", [])
            if isinstance(item, dict)
        ]
    seen_keys: set[tuple[str, str, str, str]] = set()
    try:
        for spec in specs:
            if spec.training_lane == "quarantine":
                name = f"pi_advisor_quarantine_{stamp}.jsonl"
            elif spec.provider == "prime-agent":
                name = f"prime_agent_sessions_{stamp}.jsonl"
            else:
                name = f"pi_sessions_{stamp}.jsonl"
            if name not in handles:
                target = output_dir / name
                fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=output_dir)
                os.close(fd)
                handles[name] = (target, Path(temporary_name))
            target, temporary = handles[name]
            try:
                inspection = inspect_session(spec.path)
                source_key = _source_key(spec, inspection)
                seen_keys.add(source_key)
                if source_key in completed_keys:
                    continue
                source_admission = None
                if source_manifest is not None:
                    source_admission = source_manifest.admit(
                        source_sha256=inspection.source_sha256,
                        provider=spec.provider,
                        root_label=spec.root_label,
                        source_class=spec.source_class,
                    )
                quality_override = _quality_override_for(quality_overrides, inspection)
                quality_assessment = assess_session_quality(
                    spec,
                    inspection,
                    override=quality_override,
                )
                session_record_count = 0
                with temporary.open("a", encoding="utf-8") as destination:
                    for record in iter_session_records(
                        spec,
                        max_record_chars=max_record_chars,
                        inspection=inspection,
                        quality_assessment=quality_assessment,
                        quality_override=quality_override,
                        source_admission=source_admission,
                    ):
                        destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        records += 1
                        session_record_count += 1
                        by_lane[spec.training_lane] += 1
                        by_class[spec.source_class] += 1
                    sessions += 1
                source_sha256, source_bytes, source_mtime_ns = _sha256_file(spec.path)
                source_sessions.append(
                    {
                        "provider": spec.provider,
                        "root_label": spec.root_label,
                        "source_class": spec.source_class,
                        "training_lane": spec.training_lane,
                        "quality_gate": quality_assessment["gate"],
                        "quality_flags": quality_assessment["flags"],
                        "quality_assessment": quality_assessment,
                        "source_file_name": spec.path.name,
                        "source_file_sha256": f"sha256:{source_sha256}",
                        "source_bytes": source_bytes,
                        "source_mtime_ns": source_mtime_ns,
                        "emitted_records": session_record_count,
                    }
                )
                if source_admission is not None:
                    source_sessions[-1].update(
                        {
                            "source_manifest_revision": source_admission[
                                "source_manifest_revision"
                            ],
                            "source_ref_sha256": source_admission["source_ref_sha256"],
                            "source_snapshot_status": source_admission[
                                "source_snapshot_status"
                            ],
                        }
                    )
                completion = {
                    "provider": source_key[0],
                    "root_label": source_key[1],
                    "source_class": source_key[2],
                    "source_sha256": source_key[3],
                }
                completed_keys.add(source_key)
                completed_records.append(completion)
                save_checkpoint(completed_records)
                # The file is reopened per session so a single source cannot
                # leave a descriptor or partial temp file behind.
            except Exception as exc:
                errors.append({"file": spec.path.name, "error": str(exc)})
                if checkpoint_path is None or not checkpoint_path.exists():
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
        missing_completed = completed_keys - seen_keys
        if missing_completed:
            raise ValueError("session checkpoint contains sources absent from the resumed input")
        for target, temporary in handles.values():
            if temporary.exists():
                os.replace(temporary, target)
                files[target.name] = {"bytes": target.stat().st_size, "records": 0}
        # Count output records without retaining them; this also binds the
        # manifest to the exact bytes that were atomically installed.
        for name, info in files.items():
            info["sha256"], _size, _mtime = _sha256_file(output_dir / name)
            with (output_dir / name).open("r", encoding="utf-8") as source:
                info["records"] = sum(1 for line in source if line.strip())
    except Exception:
        if checkpoint_path is None or not checkpoint_path.exists():
            for _target, temporary in handles.values():
                try:
                    if temporary.exists():
                        temporary.unlink()
                except OSError:
                    pass
        elif checkpoint_path.exists():
            saved_state = checkpoint_state or _load_session_checkpoint(checkpoint_path)
            checkpoint_outputs = saved_state.get("outputs", {})
            keep = {
                item.get("temporary_name")
                for item in checkpoint_outputs.values()
                if isinstance(item, dict)
            }
            for _target, temporary in handles.values():
                if temporary.name in keep:
                    continue
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        raise
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    manifest = {
        "schema_version": "ai-data-extraction/agent-session-ingress/v1",
        "extractor_version": EXTRACTOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": records,
        "sessions": sessions,
        "training_lane_records": dict(sorted(by_lane.items())),
        "source_class_records": dict(sorted(by_class.items())),
        "source_sessions": source_sessions,
        "errors": errors,
        "outputs": files,
    }
    if source_manifest is not None:
        manifest["source_manifest_revision"] = source_manifest.revision
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            destination.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-home",
        type=Path,
        help="explicit immutable synthetic HOME or live home containing agent roots",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("EXTRACTED_DATA_DIR", "extracted_data")))
    parser.add_argument("--max-record-chars", type=int, default=DEFAULT_MAX_RECORD_CHARS)
    parser.add_argument(
        "--quality-overrides",
        type=Path,
        help="JSON overrides keyed by source_file_sha256 or session_id; does not change training lanes",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="validated source ledger required to admit source files before extraction",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="atomic source-completion checkpoint retained for a resumable run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from --checkpoint after a prior bounded interruption",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = discover_session_specs(args.source_home)
    print(f"Found {len(specs)} Prime/Pi session files")
    overrides = load_quality_overrides(args.quality_overrides)
    source_manifest = (
        SourceManifestIndex.from_path(args.source_manifest)
        if args.source_manifest is not None
        else None
    )
    manifest = write_records(
        specs,
        output_dir=args.output_dir,
        max_record_chars=args.max_record_chars,
        quality_overrides=overrides,
        source_manifest=source_manifest,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    manifest_path = args.output_dir / "agent_session_ingress_manifest.json"
    write_manifest(args.output_dir / "agent_session_ingress_manifest.json", manifest)
    print(f"Total conversations extracted: {manifest['records']} records from {manifest['sessions']} sessions")
    print(f"Training lanes: {manifest['training_lane_records']}")
    print(f"Outputs: {', '.join(sorted(manifest['outputs'])) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
