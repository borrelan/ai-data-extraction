#!/usr/bin/env python3
"""Record observable, verifier-bound harness events for agent episodes.

This module is deliberately execution-adjacent rather than model-specific.  A
provider adapter may report what happened, but the harness is the owner of
skill-read, registry, permission, execution, verification, and loop-boundary
truth.  Hidden reasoning fields are rejected instead of being silently copied
into the trace.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "ai-data-extraction/harness-trace/v1"
TRACE_VERSION = "1.0.0"

EVENT_TYPES = {
    "skill_preflight",
    "tool_registry",
    "mcp_health",
    "decision",
    "tool_call",
    "tool_observation",
    "state_delta",
    "verification",
    "terminal",
    "loop_guard",
}
DECISIONS = {"use", "skip", "defer", "ask"}
READ_RESULTS = {"read", "failed", "not_observed"}
SCOPE_DECISIONS = {"in_scope", "out_of_scope", "unknown"}
PERMISSION_DECISIONS = {"granted", "denied", "not_required"}
OBSERVATION_STATUSES = {
    "success",
    "partial",
    "failure",
    "error",
    "timeout",
    "permission_denied",
    "unmatched",
}
VERIFICATION_RESULTS = {"pass", "fail", "partial", "not_run", "unknown"}
TERMINAL_STATUSES = {"success", "failure", "partial", "unknown", "bounded_stop"}
PRIVACY_STATES = {"unreviewed", "heuristic", "approved", "quarantine"}
REQUIRED_OPTIONAL_SKIP_REASONS = {
    "unavailable",
    "out_of_scope",
    "measured_regression",
    "not_triggered",
}
DESTRUCTIVE_SIDE_EFFECT_CLASSES = {
    "destructive",
    "write",
    "external_side_effect",
}
FORBIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "hidden_reasoning",
    "model_thoughts",
    "private_deliberation",
    "raw_thoughts",
    "scratchpad",
}


class TraceContractError(ValueError):
    """Raised when a harness event would violate the trace contract."""


@dataclass(frozen=True)
class TraceLimits:
    """Independent bounds used by the harness, not learned model behavior."""

    max_calls: int = 64
    max_output_bytes: int = 1_048_576
    repeated_signature_limit: int = 3
    no_progress_limit: int = 3

    def __post_init__(self) -> None:
        if any(
            value < 1
            for value in (
                self.max_calls,
                self.max_output_bytes,
                self.repeated_signature_limit,
                self.no_progress_limit,
            )
        ):
            raise ValueError("trace limits must be positive")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_json_value(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return False
    return True


def _check_observable(value: Any, *, path: str = "payload") -> None:
    """Reject hidden-deliberation fields at any nested payload level."""

    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in FORBIDDEN_REASONING_KEYS:
                raise TraceContractError(
                    f"hidden reasoning field is not allowed: {path}.{key}"
                )
            _check_observable(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _check_observable(nested, path=f"{path}[{index}]")
    elif not _is_json_value(value):
        raise TraceContractError(f"non-JSON observable value at {path}")


def _require_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TraceContractError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    value = _require_string(value, field)
    if value.startswith("sha256:"):
        digest = value[7:]
    else:
        digest = value
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise TraceContractError(f"{field} must be a SHA-256 digest")
    return "sha256:" + digest.lower()


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TraceContractError(f"{field} must be an object")
    return value


def _bounded_text(value: Any, max_bytes: int) -> tuple[str, str, int, bool]:
    if isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = _canonical_json(value).encode("utf-8")
    digest = _sha256(raw)
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace"), digest, len(raw), False
    head_bytes = max(1, max_bytes // 2)
    tail_bytes = max(1, max_bytes - head_bytes)
    bounded = (
        raw[:head_bytes].decode("utf-8", errors="replace")
        + "\n...[truncated by harness]...\n"
        + raw[-tail_bytes:].decode("utf-8", errors="replace")
    )
    return bounded, digest, len(raw), True


def _bounded_list(value: Iterable[Any], field: str) -> list[Any]:
    result = list(value)
    _check_observable(result, path=field)
    return result


class HarnessTrace:
    """Single-funnel recorder and stateful contract guard for one episode."""

    def __init__(
        self,
        *,
        episode_id: str,
        source: str | dict[str, Any],
        registry_revision: str = "not_applicable",
        skill_revision: str = "not_applicable",
        environment_revision: str = "not_applicable",
        verifier_revision: str = "not_applicable",
        privacy_state: str = "unreviewed",
        required_skills: Iterable[str] = (),
        limits: TraceLimits | None = None,
    ) -> None:
        self.episode_id = _require_string(episode_id, "episode_id")
        if not isinstance(source, (str, dict)):
            raise TraceContractError("source must be a string or object")
        _check_observable(source, path="source")
        self.source = source
        self.registry_revision = _require_string(registry_revision, "registry_revision")
        self.skill_revision = _require_string(skill_revision, "skill_revision")
        self.environment_revision = _require_string(
            environment_revision, "environment_revision"
        )
        self.verifier_revision = _require_string(verifier_revision, "verifier_revision")
        if privacy_state not in PRIVACY_STATES:
            raise TraceContractError(
                f"privacy_state must be one of: {', '.join(sorted(PRIVACY_STATES))}"
            )
        required_skills = list(required_skills)
        if any(
            not isinstance(skill, str) or not skill.strip()
            for skill in required_skills
        ):
            raise TraceContractError("required_skills must contain non-empty strings")
        if len(set(required_skills)) != len(required_skills):
            raise TraceContractError("required_skills must not contain duplicates")
        self.privacy_state = privacy_state
        self.required_skills = tuple(sorted(required_skills))
        self.limits = limits or TraceLimits()
        self._events: list[dict[str, Any]] = []
        self._skills: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._registry_recorded = False
        self._decisions: list[dict[str, Any]] = []
        self._calls: dict[str, dict[str, Any]] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        self._verifications: list[tuple[int, dict[str, Any]]] = []
        self._signatures: Counter[str] = Counter()
        self._no_progress: Counter[str] = Counter()
        self._terminal = False

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def _append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise TraceContractError(f"unsupported event type: {event_type}")
        if self._terminal:
            raise TraceContractError("episode already has a terminal event")
        _check_observable(payload)
        ordinal = len(self._events)
        envelope = {
            "schema_version": TRACE_SCHEMA,
            "trace_version": TRACE_VERSION,
            "episode_id": self.episode_id,
            "event_id": None,
            "ordinal": ordinal,
            "event_type": event_type,
            "registry_revision": self.registry_revision,
            "skill_revision": self.skill_revision,
            "environment_revision": self.environment_revision,
            "verifier_revision": self.verifier_revision,
            "privacy_state": self.privacy_state,
            "required_skills": list(self.required_skills),
            "source": self.source,
            "payload": payload,
        }
        identity = dict(envelope)
        identity.pop("event_id")
        envelope["event_id"] = _sha256(_canonical_json(identity))
        self._events.append(envelope)
        return dict(envelope)

    def _missing_required_skills(self) -> list[str]:
        return [
            skill
            for skill in self.required_skills
            if not (
                self._skills.get(skill, {}).get("read_result") == "read"
                and self._skills.get(skill, {}).get("scope_decision") == "in_scope"
            )
        ]

    def record_skill_preflight(
        self,
        *,
        skill: str,
        skill_revision: str,
        mandatory: bool,
        trigger: str,
        read_result: str,
        scope_decision: str,
        content_sha256: str | None = None,
        skip_reason: str | None = None,
    ) -> dict[str, Any]:
        skill = _require_string(skill, "skill")
        skill_revision = _require_string(skill_revision, "skill_revision")
        trigger = _require_string(trigger, "trigger")
        if read_result not in READ_RESULTS:
            raise TraceContractError("invalid skill read result")
        if scope_decision not in SCOPE_DECISIONS:
            raise TraceContractError("invalid skill scope decision")
        if read_result == "read":
            content_sha256 = _require_sha256(content_sha256, "content_sha256")
        elif content_sha256 is not None:
            raise TraceContractError("content_sha256 requires read_result=read")
        if read_result != "read" and mandatory:
            skip_reason = skip_reason or "mandatory_skill_not_read"
        if (
            read_result != "read"
            and not mandatory
            and skip_reason not in REQUIRED_OPTIONAL_SKIP_REASONS
        ):
            raise TraceContractError(
                "optional skill skips require an approved negative-return reason"
            )
        payload = {
            "skill": skill,
            "skill_revision": skill_revision,
            "mandatory": mandatory,
            "trigger": trigger,
            "read_result": read_result,
            "scope_decision": scope_decision,
            "content_sha256": content_sha256,
            "skip_reason": skip_reason,
        }
        event = self._append("skill_preflight", payload)
        self._skills[skill] = payload
        if mandatory and read_result != "read":
            raise TraceContractError(f"mandatory skill was not read: {skill}")
        return event

    def record_tool_registry(
        self,
        *,
        registry_revision: str,
        tools: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        registry_revision = _require_string(registry_revision, "registry_revision")
        if registry_revision != self.registry_revision:
            raise TraceContractError("tool registry revision does not match trace context")
        if self._registry_recorded:
            raise TraceContractError("tool registry is single-assignment per episode")
        tool_list = _bounded_list(tools, "tools")
        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for tool in tool_list:
            tool = _require_mapping(tool, "tool")
            name = _require_string(tool.get("name"), "tool.name")
            if name in seen_names:
                raise TraceContractError(f"duplicate tool name in registry: {name}")
            seen_names.add(name)
            schema = tool.get("schema", tool.get("inputSchema", {}))
            if not isinstance(schema, dict):
                raise TraceContractError("tool schema must be an object")
            _check_observable(schema, path=f"tool[{name}].schema")
            trust_class = _require_string(
                tool.get("trust_class", "unknown"), "tool.trust_class"
            )
            side_effect_class = _require_string(
                tool.get("side_effect_class", "read_only"),
                "tool.side_effect_class",
            )
            item = {
                "name": name,
                "schema": schema,
                "trust_class": trust_class,
                "side_effect_class": side_effect_class,
            }
            normalized.append(item)
        normalized.sort(key=lambda item: item["name"])
        registry_sha256 = _sha256(
            _canonical_json(
                {
                    "registry_revision": registry_revision,
                    "tools": normalized,
                }
            )
        )
        event = self._append(
            "tool_registry",
            {
                "registry_revision": registry_revision,
                "registry_sha256": registry_sha256,
                "tools": normalized,
            },
        )
        self._tools = {item["name"]: item for item in normalized}
        self._registry_recorded = True
        return event

    def record_mcp_health(
        self,
        *,
        server: str,
        revision: str,
        health_result: str,
        capabilities: Iterable[str] = (),
        timeout: bool = False,
        error: str | None = None,
    ) -> dict[str, Any]:
        server = _require_string(server, "server")
        revision = _require_string(revision, "revision")
        health_result = _require_string(health_result, "health_result")
        capabilities = list(capabilities)
        if any(not isinstance(item, str) or not item.strip() for item in capabilities):
            raise TraceContractError("MCP capabilities must be non-empty strings")
        if error is not None:
            error = _require_string(error, "error")
        return self._append(
            "mcp_health",
            {
                "server": server,
                "revision": revision,
                "health_result": health_result,
                "capabilities": capabilities,
                "timeout": bool(timeout),
                "error": error,
            },
        )

    def record_decision(
        self,
        *,
        decision: str,
        decision_basis: str,
        required_gate_status: str,
        tool_name: str | None = None,
        skill: str | None = None,
    ) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise TraceContractError("decision must be use, skip, defer, or ask")
        decision_basis = _require_string(decision_basis, "decision_basis")
        if len(decision_basis) > 256:
            raise TraceContractError("decision_basis is limited to 256 characters")
        required_gate_status = _require_string(
            required_gate_status, "required_gate_status"
        )
        if tool_name is not None:
            tool_name = _require_string(tool_name, "tool_name")
        if skill is not None:
            skill = _require_string(skill, "skill")
        if decision == "use" and tool_name is not None:
            missing_skills = self._missing_required_skills()
            if missing_skills:
                raise TraceContractError(
                    "tool use requires required skills to be read in scope: "
                    + ", ".join(missing_skills)
                )
        payload = {
            "decision": decision,
            "decision_basis": decision_basis,
            "required_gate_status": required_gate_status,
            "tool_name": tool_name,
            "skill": skill,
        }
        event = self._append("decision", payload)
        self._decisions.append(payload)
        return event

    def _last_tool_decision(self, tool_name: str) -> dict[str, Any] | None:
        for decision in reversed(self._decisions):
            if decision.get("tool_name") == tool_name:
                return decision
        return None

    def record_tool_call(
        self,
        *,
        tool_name: str,
        call_id: str,
        arguments: Any,
        permission_decision: str,
    ) -> dict[str, Any]:
        if self._terminal:
            raise TraceContractError("episode already has a terminal event")
        tool_name = _require_string(tool_name, "tool_name")
        call_id = _require_string(call_id, "call_id")
        if call_id in self._calls:
            raise TraceContractError(f"duplicate tool call ID: {call_id}")
        if not _is_json_value(arguments):
            raise TraceContractError("tool arguments must be JSON-compatible")
        if permission_decision not in PERMISSION_DECISIONS:
            raise TraceContractError("invalid permission decision")
        if len(self._calls) >= self.limits.max_calls:
            guard_event = self._append(
                "loop_guard",
                {
                    "repeated_signature": _sha256(f"call-limit:{tool_name}"),
                    "no_progress_hash": _sha256(f"call-count:{len(self._calls)}"),
                    "repeated_count": len(self._calls),
                    "no_progress_count": len(self._calls),
                    "limit": self.limits.max_calls,
                    "action": "terminate",
                    "reason": "call_limit",
                },
            )
            self.record_terminal(
                status="bounded_stop",
                evidence=[guard_event["event_id"]],
            )
            raise TraceContractError("tool call limit reached")
        if not self._registry_recorded:
            raise TraceContractError("tool registry must be recorded before tool calls")
        if tool_name not in self._tools:
            raise TraceContractError(f"tool is not exposed by the registry: {tool_name}")
        decision = self._last_tool_decision(tool_name)
        if decision is None or decision.get("decision") != "use":
            raise TraceContractError("tool call requires a preceding use decision")
        if decision.get("required_gate_status") not in {"passed", "not_required"}:
            raise TraceContractError("tool call requires passed required gates")
        side_effect_class = self._tools[tool_name]["side_effect_class"]
        if (
            side_effect_class in DESTRUCTIVE_SIDE_EFFECT_CLASSES
            and permission_decision != "granted"
        ):
            raise TraceContractError(
                "side-effecting tool execution requires granted permission"
            )
        payload = {
            "tool_name": tool_name,
            "call_id": call_id,
            "arguments": arguments,
            "permission_decision": permission_decision,
            "side_effect_class": side_effect_class,
        }
        event = self._append("tool_call", payload)
        self._calls[call_id] = payload
        return event

    def record_tool_observation(
        self,
        *,
        call_id: str,
        status: str,
        output: Any = "",
        error: str | None = None,
        max_output_bytes: int | None = None,
    ) -> dict[str, Any]:
        call_id = _require_string(call_id, "call_id")
        if status not in OBSERVATION_STATUSES:
            raise TraceContractError("invalid tool observation status")
        if call_id not in self._calls and status != "unmatched":
            raise TraceContractError(
                "unknown call ID must be recorded with status=unmatched"
            )
        if call_id in self._observations:
            raise TraceContractError(f"duplicate tool observation ID: {call_id}")
        limit = max_output_bytes or self.limits.max_output_bytes
        if limit < 1:
            raise TraceContractError("max_output_bytes must be positive")
        bounded, output_sha256, output_bytes, truncated = _bounded_text(output, limit)
        if error is not None:
            error = _require_string(error, "error")
        payload = {
            "call_id": call_id,
            "status": status,
            "output": bounded,
            "output_sha256": output_sha256,
            "output_bytes": output_bytes,
            "output_truncated": truncated,
            "error": error,
        }
        event = self._append("tool_observation", payload)
        self._observations[call_id] = payload
        return event

    def record_state_delta(
        self,
        *,
        before: dict[str, str | None],
        after: dict[str, str | None],
        changed: Iterable[str] = (),
    ) -> dict[str, Any]:
        before = _require_mapping(before, "before")
        after = _require_mapping(after, "after")
        for label, values in (("before", before), ("after", after)):
            for key, value in values.items():
                if value is not None:
                    _require_sha256(value, f"{label}.{key}")
        changed = list(changed)
        if any(not isinstance(item, str) for item in changed):
            raise TraceContractError("changed state labels must be strings")
        return self._append(
            "state_delta",
            {"before": before, "after": after, "changed": changed},
        )

    def record_verification(
        self,
        *,
        verifier_revision: str,
        checks: Iterable[Any],
        result: str,
        durable_evidence: Iterable[Any] = (),
    ) -> dict[str, Any]:
        verifier_revision = _require_string(verifier_revision, "verifier_revision")
        if verifier_revision != self.verifier_revision:
            raise TraceContractError("verifier revision does not match trace context")
        if result not in VERIFICATION_RESULTS:
            raise TraceContractError("invalid verification result")
        checks = _bounded_list(checks, "checks")
        durable_evidence = _bounded_list(durable_evidence, "durable_evidence")
        if result == "pass" and (not checks or not durable_evidence):
            raise TraceContractError(
                "a passing verification requires checks and durable evidence"
            )
        event = self._append(
            "verification",
            {
                "verifier_revision": verifier_revision,
                "checks": checks,
                "result": result,
                "durable_evidence": durable_evidence,
            },
        )
        self._verifications.append((event["ordinal"], event["payload"]))
        return event

    def record_terminal(
        self,
        *,
        status: str,
        evidence: Iterable[Any] = (),
    ) -> dict[str, Any]:
        if status not in TERMINAL_STATUSES:
            raise TraceContractError("invalid terminal status")
        if self._terminal:
            raise TraceContractError("episode already has a terminal event")
        if status == "success":
            open_calls = sorted(set(self._calls).difference(self._observations))
            if open_calls:
                raise TraceContractError(
                    "successful terminal state has unobserved calls: "
                    + ", ".join(open_calls)
                )
            if not self._verifications:
                raise TraceContractError(
                    "successful terminal state requires verifier evidence"
                )
            last_call_ordinal = max(
                (
                    event["ordinal"]
                    for event in self._events
                    if event["event_type"] == "tool_call"
                ),
                default=-1,
            )
            verification_ordinal, verification = self._verifications[-1]
            if verification_ordinal < last_call_ordinal:
                raise TraceContractError(
                    "successful terminal state requires verification after the last call"
                )
            if verification.get("result") != "pass":
                raise TraceContractError(
                    "successful terminal state requires a passing verification"
                )
        evidence = _bounded_list(evidence, "evidence")
        event = self._append("terminal", {"status": status, "evidence": evidence})
        self._terminal = True
        return event

    def record_loop_guard(
        self,
        *,
        signature: str,
        state_hash: str,
        limit: int | None = None,
        action: str = "terminate",
        reason: str = "repeated_signature_or_no_progress",
    ) -> dict[str, Any]:
        signature = _require_string(signature, "signature")
        state_hash = _require_sha256(state_hash, "state_hash")
        limit = limit or self.limits.repeated_signature_limit
        if limit < 1:
            raise TraceContractError("loop guard limit must be positive")
        signature_hash = _sha256(signature)
        self._signatures[signature_hash] += 1
        self._no_progress[state_hash] += 1
        repeated_count = self._signatures[signature_hash]
        no_progress_count = self._no_progress[state_hash]
        triggered = (
            repeated_count >= limit
            or no_progress_count >= self.limits.no_progress_limit
        )
        effective_action = action if triggered else "observe"
        payload = {
            "repeated_signature": signature_hash,
            "no_progress_hash": state_hash,
            "repeated_count": repeated_count,
            "no_progress_count": no_progress_count,
            "limit": limit,
            "action": _require_string(effective_action, "action"),
            "reason": _require_string(reason, "reason"),
        }
        event = self._append("loop_guard", payload)
        if triggered and effective_action == "terminate" and not self._terminal:
            self.record_terminal(
                status="bounded_stop",
                evidence=[event["event_id"]],
            )
        return event

    def to_jsonl(self) -> str:
        return "".join(_canonical_json(event) + "\n" for event in self._events)

    def write_jsonl(self, path: Path, *, overwrite: bool = False) -> None:
        path = Path(path)
        if not self._terminal:
            raise TraceContractError("cannot write a non-terminal episode")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass overwrite=True")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                destination.write(self.to_jsonl())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


__all__ = [
    "DESTRUCTIVE_SIDE_EFFECT_CLASSES",
    "EVENT_TYPES",
    "PRIVACY_STATES",
    "TRACE_SCHEMA",
    "TRACE_VERSION",
    "HarnessTrace",
    "TraceContractError",
    "TraceLimits",
]
