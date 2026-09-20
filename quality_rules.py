"""Shared row-level quality rules for release and audit tooling.

The functions in this module return counts, structural paths, and status
values. They never return matched text or session content.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


HIDDEN_KEY_RE = re.compile(
    r"(?:^|[_-])(thoughts?|thinking|reasoning|analysis|deliberation|"
    r"scratchpad|cot|chain[_-]?of[_-]?thought|internal[_-]?monologue)(?:$|[_-])",
    re.IGNORECASE,
)
HIDDEN_MARKER_RE = re.compile(
    r"<\s*/?\s*(?:thinking|analysis|reasoning|deliberation|scratchpad|"
    r"chain[_ -]?of[_ -]?thought)\b|"
    r"\b(?:chain of thought|hidden reasoning|internal reasoning|"
    r"private scratchpad)\b",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"-----BEGIN [^-]{2,80} PRIVATE KEY-----|"
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|"
    r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b",
    re.IGNORECASE,
)
SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(api[_-]?key|access[_-]?token|auth[_-]?token|password|"
    r"passwd|secret|private[_-]?key|client[_-]?secret)(?:$|[_-])",
    re.IGNORECASE,
)
PRIVATE_VALUE_RE = re.compile(
    r"(?:^|\s)/(?:home|root|Users|private|data|tmp|var|opt)/[^\s\"']+|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|"
    r"\b[A-Fa-f0-9]{32,}\b"
)
PRIVATE_KEY_RE = re.compile(
    r"(?:^|[_-])(cwd|directory|project[_-]?path|source[_-]?file|"
    r"session[_-]?file|installation|share[_-]?url|session[_-]?id|"
    r"project[_-]?id|workspace[_-]?id|parent[_-]?session[_-]?id|"
    r"agent[_-]?id|source[_-]?uri|source[_-]?path)(?:$|[_-])",
    re.IGNORECASE,
)

RAW_KEYS = frozenset({"raw", "extra", "debug", "stack_trace", "stacktrace"})
METADATA_CONTAINERS = frozenset(
    {
        "database",
        "event_counts",
        "control_event_counts",
        "model",
        "source_origin",
        "harness_summary",
        "quality_assessment",
        "quality_summary",
        "session_header",
        "tokens",
    }
)
CALL_TYPES = frozenset(
    {
        "tool_use",
        "tool_call",
        "toolcall",
        "function_call",
        "custom_tool_call",
        "functioncall",
    }
)
RESULT_TYPES = frozenset(
    {
        "tool_result",
        "toolresult",
        "function_call_output",
        "custom_tool_call_output",
        "functioncalloutput",
    }
)
CALL_KEYS = (
    "tool_calls",
    "tool_call",
    "tool_use",
    "toolUses",
    "toolUse",
    "function_calls",
    "function_call",
    "custom_tool_calls",
)
RESULT_KEYS = (
    "tool_results",
    "tool_result",
    "toolResult",
    "function_call_outputs",
    "function_call_output",
    "custom_tool_call_outputs",
)
LINEAGE_RANGE_KEYS = (
    "source_event_line_range",
    "message_line_range",
    "message_index_range",
    "source_message_range",
    "byte_range",
    "byte_start",
    "_chunk_message_start",
)
UNIT_ID_KEYS = (
    "unit_id",
    "record_id",
    "example_id",
    "window_id",
    "episode_id",
    "session_id",
    "_chunk_parent_record_sha256",
)


def _normalized_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def _short_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_")


def _is_omission_key(key: str) -> bool:
    return bool(
        re.search(
            r"^(?:reasoning|thoughts?|thinking)(?:[_-][a-z0-9]+)*[_-]omitted"
            r"(?:[_-][a-z0-9]+)*$",
            key,
            re.IGNORECASE,
        )
    )


def _is_structural_hidden_key(key: str, path: str, value: Any) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if path.startswith("$.quality.") and normalized in {
        "reasoning_removed",
        "reasoning_fields_removed",
        "reasoning_blocks_removed",
    }:
        return isinstance(value, (bool, int, float))
    if _is_omission_key(normalized):
        return True
    if normalized == "thinking_level_change":
        return True
    if normalized in {"reasoning", "thoughts"} and ".tokens" in path:
        return isinstance(value, (int, float))
    return False


@dataclass(slots=True)
class RecordFindings:
    hidden_keys: Counter[str] = field(default_factory=Counter)
    hidden_key_paths: Counter[str] = field(default_factory=Counter)
    marker_paths: Counter[str] = field(default_factory=Counter)
    marker_contexts: Counter[str] = field(default_factory=Counter)
    raw_keys: Counter[str] = field(default_factory=Counter)
    secret_keys: Counter[str] = field(default_factory=Counter)
    secret_values: int = 0
    private_keys: Counter[str] = field(default_factory=Counter)
    private_values: int = 0

    @property
    def has_hard_privacy_issue(self) -> bool:
        return bool(
            self.hidden_keys
            or self.raw_keys
            or self.secret_keys
            or self.secret_values
            or self.private_values
        )

    @property
    def has_marker(self) -> bool:
        return bool(self.marker_paths)

    def summary(self) -> dict[str, Any]:
        return {
            "hidden_key_hits": sum(self.hidden_keys.values()),
            "hidden_key_paths": dict(self.hidden_key_paths),
            "marker_hits": sum(self.marker_paths.values()),
            "marker_paths": dict(self.marker_paths),
            "marker_contexts": dict(self.marker_contexts),
            "raw_key_hits": dict(self.raw_keys),
            "secret_key_hits": dict(self.secret_keys),
            "secret_value_hits": self.secret_values,
            "private_key_hits": dict(self.private_keys),
            "private_value_hits": self.private_values,
        }


def _scan_value(
    value: Any,
    path: str,
    findings: RecordFindings,
    *,
    metadata: bool,
    role: str | None,
) -> None:
    if isinstance(value, dict):
        container_name = path.rsplit(".", 1)[-1]
        child_metadata = metadata or container_name in METADATA_CONTAINERS
        declared_role = value.get("role")
        current_role = (
            declared_role.lower()
            if isinstance(declared_role, str) and declared_role
            else role
        )
        for key, child in value.items():
            key_text = str(key).strip()
            key_path = f"{path}.{key_text}"
            if HIDDEN_KEY_RE.search(key_text) and not _is_structural_hidden_key(
                key_text, key_path, child
            ):
                findings.hidden_keys[key_text] += 1
                findings.hidden_key_paths[_normalized_path(key_path)] += 1
            normalized_key = key_text.lower()
            if normalized_key in RAW_KEYS:
                findings.raw_keys[normalized_key] += 1
            if SECRET_KEY_RE.search(key_text):
                findings.secret_keys[key_text] += 1
            if PRIVATE_KEY_RE.search(key_text):
                findings.private_keys[key_text] += 1
            next_metadata = (
                child_metadata
                or normalized_key in METADATA_CONTAINERS
                or (path == "$" and normalized_key != "messages")
            )
            _scan_value(
                child,
                key_path,
                findings,
                metadata=next_metadata,
                role=current_role,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_value(
                child,
                f"{path}[{index}]",
                findings,
                metadata=metadata,
                role=role,
            )
        return
    if not isinstance(value, str) or not value:
        return

    marker_count = len(HIDDEN_MARKER_RE.findall(value))
    if marker_count:
        marker_path = _normalized_path(path)
        findings.marker_paths[marker_path] += marker_count
        if metadata:
            context = "metadata"
        elif role in {"assistant", "user", "system"}:
            context = "trainer_blocker"
        elif role in {"tool", "tool_result"}:
            context = "tool_observation_review"
        else:
            context = "unknown_context_review"
        findings.marker_contexts[context] += marker_count
    if SECRET_VALUE_RE.search(value):
        findings.secret_values += 1
    if not metadata and PRIVATE_VALUE_RE.search(value):
        findings.private_values += 1


def scan_record(record: dict[str, Any]) -> RecordFindings:
    findings = RecordFindings()
    _scan_value(record, "$", findings, metadata=False, role=None)
    return findings


def _containers(record: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [record]
    for key in ("metadata", "provenance", "lineage", "quality", "source_origin"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def merged_lineage(record: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for container in _containers(record):
        combined.update(container)
    return combined


def lineage_issues(record: dict[str, Any]) -> list[str]:
    combined = merged_lineage(record)
    issues: list[str] = []
    if not any(
        isinstance(combined.get(key), str) and combined[key]
        for key in ("source_sha256", "source_file_sha256", "source_fingerprint")
    ):
        issues.append("source_digest_missing")
    snapshot_bound = (
        isinstance(combined.get("source_manifest_revision"), str)
        and bool(combined.get("source_manifest_revision"))
        and combined.get("source_snapshot_status") in {"bound", "immutable"}
    )
    if not snapshot_bound and not any(
        isinstance(combined.get(key), str) and combined[key]
        for key in ("snapshot_revision", "snapshot_id", "snapshot_ref")
    ):
        issues.append("snapshot_revision_missing")
    if not any(
        isinstance(combined.get(key), str) and combined[key]
        for key in ("parser_revision", "parser_rev")
    ):
        issues.append("parser_revision_missing")
    if not any(
        isinstance(combined.get(key), (dict, list, int, str))
        for key in LINEAGE_RANGE_KEYS
    ):
        issues.append("source_or_message_range_missing")
    if not any(
        isinstance(combined.get(key), str) and combined[key] for key in UNIT_ID_KEYS
    ):
        issues.append("unit_identity_missing")
    return issues


def quality_gate(record: dict[str, Any]) -> str:
    for container in _containers(record):
        for key in ("quality_gate", "gate", "session_quality_gate"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
        assessment = container.get("quality_assessment")
        if isinstance(assessment, dict) and isinstance(assessment.get("gate"), str):
            return assessment["gate"]
    return "unspecified"


def model_tier(record: dict[str, Any]) -> str:
    for container in _containers(record):
        value = container.get("model_tier")
        if isinstance(value, str) and value:
            return value
    return "unclassified"


def training_lane(record: dict[str, Any]) -> str:
    for container in _containers(record):
        value = container.get("training_lane")
        if isinstance(value, str) and value:
            return value
    return "unclassified"


def unit_id(record: dict[str, Any]) -> str:
    for container in _containers(record):
        for key in UNIT_ID_KEYS:
            value = container.get(key)
            if isinstance(value, (str, int)) and str(value):
                return str(value)
    digest = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"sha256:{digest}"


def _id_for_action(value: dict[str, Any]) -> str | None:
    for key in ("id", "call_id", "callID", "tool_call_id", "toolCallId"):
        item = value.get(key)
        if isinstance(item, (str, int)) and str(item):
            return str(item)
    nested = value.get("function")
    if isinstance(nested, dict):
        for key in ("id", "call_id", "callID"):
            item = nested.get(key)
            if isinstance(item, (str, int)) and str(item):
                return str(item)
    return None


def _classify_action(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ("type", "kind", "part_type", "partType"):
        item = _short_type(value.get(key))
        if item in CALL_TYPES:
            return "call"
        if item in RESULT_TYPES:
            return "result"
    return None


@dataclass(frozen=True, slots=True)
class ToolEdgeStatus:
    calls: int
    results: int
    missing_call_ids: int
    missing_result_ids: int
    unmatched_calls: int
    unmatched_results: int
    duplicate_call_ids: int
    duplicate_result_ids: int

    @property
    def has_actions(self) -> bool:
        return bool(self.calls or self.results)

    @property
    def complete(self) -> bool:
        return bool(self.has_actions) and not any(
            (
                self.missing_call_ids,
                self.missing_result_ids,
                self.unmatched_calls,
                self.unmatched_results,
                self.duplicate_call_ids,
                self.duplicate_result_ids,
            )
        )

    def summary(self) -> dict[str, int | bool]:
        return {
            "calls": self.calls,
            "results": self.results,
            "missing_call_ids": self.missing_call_ids,
            "missing_result_ids": self.missing_result_ids,
            "unmatched_calls": self.unmatched_calls,
            "unmatched_results": self.unmatched_results,
            "duplicate_call_ids": self.duplicate_call_ids,
            "duplicate_result_ids": self.duplicate_result_ids,
            "complete": self.complete,
        }


def tool_edge_status(record: dict[str, Any]) -> ToolEdgeStatus:
    call_ids: Counter[str] = Counter()
    result_ids: Counter[str] = Counter()
    missing_call_ids = 0
    missing_result_ids = 0
    calls = 0
    results = 0

    def inspect(value: Any, kind: str | None = None) -> None:
        nonlocal calls, results, missing_call_ids, missing_result_ids
        if isinstance(value, list):
            for item in value:
                inspect(item, kind)
            return
        if not isinstance(value, dict):
            return
        detected = kind or _classify_action(value)
        if detected == "call":
            calls += 1
            identity = _id_for_action(value)
            if identity is None:
                missing_call_ids += 1
            else:
                call_ids[identity] += 1
        elif detected == "result":
            results += 1
            identity = _id_for_action(value)
            if identity is None:
                missing_result_ids += 1
            else:
                result_ids[identity] += 1

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            for key in CALL_KEYS:
                if key in message:
                    inspect(message[key], "call")
            for key in RESULT_KEYS:
                if key in message:
                    inspect(message[key], "result")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    inspect(part)
            if str(message.get("role", "")).lower() == "tool":
                inspect(message, "result")

    return ToolEdgeStatus(
        calls=calls,
        results=results,
        missing_call_ids=missing_call_ids,
        missing_result_ids=missing_result_ids,
        unmatched_calls=sum(
            max(count - result_ids.get(identity, 0), 0)
            for identity, count in call_ids.items()
        ),
        unmatched_results=sum(
            max(count - call_ids.get(identity, 0), 0)
            for identity, count in result_ids.items()
        ),
        duplicate_call_ids=sum(max(count - 1, 0) for count in call_ids.values()),
        duplicate_result_ids=sum(max(count - 1, 0) for count in result_ids.values()),
    )


def prompt_group_id(record: dict[str, Any]) -> str | None:
    explicit = record.get("prompt_group_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    prompts = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict)
        and str(message.get("role", "")).lower() == "user"
    ]
    if not prompts:
        return None
    encoded = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
