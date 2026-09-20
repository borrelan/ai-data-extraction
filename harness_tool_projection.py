#!/usr/bin/env python3
"""Project one sanitized episode plus a harness trace into review-only tool SFT.

The harness trace owns execution truth.  The episode owns the sanitized
messages and trainer tool schemas.  This module joins them only when their
call IDs, tool names/schemas, revisions, skill gates, and verifier closure
agree.  It never invents a registry, reward, or outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from build_training_data import no_reasoning_content, trainer_marker_count
from quality_rules import scan_record
from trainer_export import (
    TRAINER_EXAMPLE_SCHEMA,
    _has_tool_activity,
    _parent_id,
    _tool_schema_name,
    _trainer_view,
)
from harness_trace import EVENT_TYPES, TRACE_SCHEMA, TRACE_VERSION


PROJECTION_SCHEMA = "ai-data-extraction/harness-tool-projection/v1"


class HarnessProjectionError(ValueError):
    """Raised when a trace cannot prove a trainer-safe tool episode."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise HarnessProjectionError(f"{field} must be a SHA-256 string")
    digest = value[7:] if value.startswith("sha256:") else value
    if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise HarnessProjectionError(f"{field} must be a SHA-256 string")
    return digest.lower()


def _load_one_episode(path: Path) -> tuple[dict[str, Any], str, int]:
    row: dict[str, Any] | None = None
    row_raw = b""
    records = 0
    with path.open("rb") as source:
        for raw in source:
            if not raw.strip():
                continue
            records += 1
            if records > 1:
                raise HarnessProjectionError("episode input must contain exactly one row")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise HarnessProjectionError("episode input row must be an object")
            row = value
            row_raw = raw
    if row is None:
        raise HarnessProjectionError("episode input is empty")
    return row, _sha256_bytes(row_raw), records


def _load_trace(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise HarnessProjectionError(f"trace line {line_number} is not an object")
            events.append(value)
    if not events:
        raise HarnessProjectionError("trace is empty")

    base: dict[str, Any] | None = None
    for ordinal, event in enumerate(events):
        if event.get("schema_version") != TRACE_SCHEMA:
            raise HarnessProjectionError("trace schema mismatch")
        if event.get("trace_version") != TRACE_VERSION:
            raise HarnessProjectionError("trace version mismatch")
        if event.get("ordinal") != ordinal:
            raise HarnessProjectionError("trace ordinals are not contiguous")
        if event.get("event_type") not in EVENT_TYPES:
            raise HarnessProjectionError("trace contains an unknown event type")
        event_id = event.get("event_id")
        identity = dict(event)
        identity.pop("event_id", None)
        expected_id = "sha256:" + _sha256_bytes(_canonical_bytes(identity))
        if event_id != expected_id:
            raise HarnessProjectionError(f"trace event {ordinal} identity mismatch")
        common = {
            key: event.get(key)
            for key in (
                "episode_id",
                "registry_revision",
                "skill_revision",
                "environment_revision",
                "verifier_revision",
                "privacy_state",
                "required_skills",
                "source",
            )
        }
        if base is None:
            base = common
        elif common != base:
            raise HarnessProjectionError(f"trace event {ordinal} context mismatch")
        if not isinstance(event.get("payload"), dict):
            raise HarnessProjectionError(f"trace event {ordinal} payload is not an object")
    return events


def _message_calls(record: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    calls: dict[str, dict[str, Any]] = {}
    observations: set[str] = set()
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise HarnessProjectionError("episode messages are missing")
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                raise HarnessProjectionError("episode tool call is not an object")
            call_id = call.get("id") or call.get("call_id")
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name") or call.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise HarnessProjectionError("episode tool call has no ID")
            if not isinstance(name, str) or not name:
                raise HarnessProjectionError("episode tool call has no name")
            if call_id in calls:
                raise HarnessProjectionError(f"duplicate episode tool call ID: {call_id}")
            arguments = function.get("arguments", call.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise HarnessProjectionError(
                        f"episode tool arguments are not JSON: {call_id}"
                    ) from exc
            calls[call_id] = {"name": name, "arguments": arguments}
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise HarnessProjectionError("episode tool observation has no call ID")
            if call_id in observations:
                raise HarnessProjectionError(f"duplicate episode observation ID: {call_id}")
            observations.add(call_id)
    return calls, observations


def _trainer_tool_schemas(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        raise HarnessProjectionError("tool episode has no trainer tool schemas")
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = _tool_schema_name(tool)
        if name is None or name in result:
            raise HarnessProjectionError("tool schemas have missing or duplicate names")
        if not isinstance(tool, dict):
            raise HarnessProjectionError("tool schema is not an object")
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        parameters = function.get("parameters", tool.get("parameters", tool.get("inputSchema")))
        if not isinstance(parameters, dict):
            raise HarnessProjectionError(f"tool schema has no parameters: {name}")
        result[name] = parameters
    return result


def _join_trace(record: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    if not _has_tool_activity(record):
        raise HarnessProjectionError("episode has no tool activity")
    view = _trainer_view(record, source_dataset="tool_trace")
    if not no_reasoning_content(view) or trainer_marker_count(view):
        raise HarnessProjectionError("trainer view contains reasoning or markers")
    findings = scan_record(view)
    if findings.has_hard_privacy_issue or findings.has_marker:
        raise HarnessProjectionError("trainer view fails privacy/marker firewall")

    registry_events = [event for event in events if event["event_type"] == "tool_registry"]
    terminal_events = [event for event in events if event["event_type"] == "terminal"]
    if len(registry_events) != 1:
        raise HarnessProjectionError("tool episode requires exactly one registry event")
    if len(terminal_events) != 1 or events[-1] is not terminal_events[0]:
        raise HarnessProjectionError("tool episode requires one final terminal event")
    terminal = terminal_events[0]["payload"]
    if terminal.get("status") != "success":
        raise HarnessProjectionError("only verifier-backed success is projected to tool SFT")

    registry_event = registry_events[0]
    registry_payload = registry_event["payload"]
    registry_tools = registry_payload.get("tools")
    if not isinstance(registry_tools, list) or not registry_tools:
        raise HarnessProjectionError("registry event has no tools")
    registry_identity = {
        "registry_revision": registry_payload.get("registry_revision"),
        "tools": registry_tools,
    }
    expected_registry_sha = "sha256:" + _sha256_bytes(_canonical_bytes(registry_identity))
    if registry_payload.get("registry_sha256") != expected_registry_sha:
        raise HarnessProjectionError("registry digest does not match its contents")

    trainer_schemas = _trainer_tool_schemas(record)
    harness_schemas = {}
    for tool in registry_tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise HarnessProjectionError("registry tool is malformed")
        name = tool["name"]
        if name in harness_schemas:
            raise HarnessProjectionError("registry has duplicate tool names")
        schema = tool.get("schema")
        if not isinstance(schema, dict):
            raise HarnessProjectionError(f"registry schema is missing: {name}")
        harness_schemas[name] = schema
    if trainer_schemas != harness_schemas:
        raise HarnessProjectionError("episode tool schemas do not match harness registry")

    message_calls, message_observations = _message_calls(record)
    trace_calls: dict[str, dict[str, Any]] = {}
    trace_observations: dict[str, dict[str, Any]] = {}
    decision_ordinals: dict[str, list[int]] = {}
    verification_events: list[dict[str, Any]] = []
    required_skills = events[0].get("required_skills")
    if not isinstance(required_skills, list):
        raise HarnessProjectionError("trace required_skills is not a list")
    skill_reads = {
        event["payload"].get("skill")
        for event in events
        if event["event_type"] == "skill_preflight"
        and event["payload"].get("read_result") == "read"
        and event["payload"].get("scope_decision") == "in_scope"
    }
    missing_skills = sorted(set(required_skills).difference(skill_reads))
    if missing_skills:
        raise HarnessProjectionError(
            "required skills were not read in scope: " + ", ".join(missing_skills)
        )

    for event in events:
        payload = event["payload"]
        if event["event_type"] == "decision":
            if payload.get("decision") == "use" and isinstance(payload.get("tool_name"), str):
                decision_ordinals.setdefault(payload["tool_name"], []).append(event["ordinal"])
        elif event["event_type"] == "tool_call":
            call_id = payload.get("call_id")
            name = payload.get("tool_name")
            if not isinstance(call_id, str) or not isinstance(name, str) or call_id in trace_calls:
                raise HarnessProjectionError("trace calls have invalid or duplicate IDs")
            trace_calls[call_id] = payload | {"ordinal": event["ordinal"]}
        elif event["event_type"] == "tool_observation":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or call_id in trace_observations:
                raise HarnessProjectionError("trace observations have invalid or duplicate IDs")
            trace_observations[call_id] = payload | {"ordinal": event["ordinal"]}
        elif event["event_type"] == "verification":
            verification_events.append(event)

    if set(message_calls) != set(trace_calls) or message_observations != set(trace_observations):
        raise HarnessProjectionError("episode and harness call/observation IDs do not match")
    if set(trace_calls) != set(trace_observations):
        raise HarnessProjectionError("success trace has an unobserved call")
    for call_id, call in trace_calls.items():
        message_call = message_calls[call_id]
        if message_call["name"] != call["tool_name"]:
            raise HarnessProjectionError(f"tool name mismatch for call {call_id}")
        if message_call["arguments"] != call.get("arguments"):
            raise HarnessProjectionError(f"tool arguments mismatch for call {call_id}")
        if not any(
            ordinal < call["ordinal"]
            for ordinal in decision_ordinals.get(call["tool_name"], [])
        ):
            raise HarnessProjectionError(f"call lacks preceding use decision: {call_id}")

    if not verification_events:
        raise HarnessProjectionError("success trace has no verification")
    last_call_ordinal = max(call["ordinal"] for call in trace_calls.values())
    verification = verification_events[-1]
    verification_payload = verification["payload"]
    if verification["ordinal"] < last_call_ordinal:
        raise HarnessProjectionError("verification precedes the last tool call")
    if verification_payload.get("result") != "pass":
        raise HarnessProjectionError("latest verification is not a pass")
    if not verification_payload.get("checks") or not verification_payload.get("durable_evidence"):
        raise HarnessProjectionError("passing verification lacks checks or durable evidence")

    return {
        "view": view,
        "episode_id": events[0]["episode_id"],
        "registry_revision": events[0]["registry_revision"],
        "registry_sha256": registry_payload["registry_sha256"],
        "skill_revision": events[0]["skill_revision"],
        "environment_revision": events[0]["environment_revision"],
        "verifier_revision": events[0]["verifier_revision"],
        "required_skills": sorted(required_skills),
        "terminal_status": terminal["status"],
        "verification_result": verification_payload["result"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    with path.open("wb") as destination:
        for row in rows:
            raw = _canonical_bytes(row) + b"\n"
            destination.write(raw)
            digest.update(raw)
            count += 1
            byte_count += len(raw)
    return {"records": count, "bytes": byte_count, "sha256": digest.hexdigest()}


def project_harness_tool_sft(
    episode_path: Path,
    trace_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    episode_path = episode_path.resolve()
    trace_path = trace_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to use existing output directory: {output_dir}")
    if not episode_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError("episode and trace inputs must exist")
    record, source_row_sha256, source_records = _load_one_episode(episode_path)
    if source_records != 1:
        raise HarnessProjectionError("episode input must contain one record")
    parent = _parent_id(record)
    if parent is None:
        raise HarnessProjectionError("episode is missing parent identity")
    events = _load_trace(trace_path)
    joined = _join_trace(record, events)
    trace_sha256 = _sha256_file(trace_path)
    episode_sha256 = _sha256_file(episode_path)
    view = joined["view"]
    lineage = {
        "schema_version": "ai-data-extraction/harness-tool-lineage/v1",
        "example_id": view["example_id"],
        "split": view["split"],
        "source_dataset": record.get("dataset"),
        "source_row_sha256": source_row_sha256,
        "source_file_sha256": episode_sha256,
        "parent_record_sha256": parent,
        "trace_file_sha256": trace_sha256,
        "episode_id": joined["episode_id"],
        "registry_revision": joined["registry_revision"],
        "registry_sha256": joined["registry_sha256"],
        "skill_revision": joined["skill_revision"],
        "required_skills": joined["required_skills"],
        "environment_revision": joined["environment_revision"],
        "verifier_revision": joined["verifier_revision"],
        "terminal_status": joined["terminal_status"],
        "verification_result": joined["verification_result"],
        "quality_tier": "harness_verified_tool_sft",
        "limitations": ["training_authorized_false", "reward_not_exported"],
    }
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError(f"refusing to use existing staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        files = {
            "tool_sft.jsonl": _write_jsonl(staging / "tool_sft.jsonl", [view]),
            "lineage.jsonl": _write_jsonl(staging / "lineage.jsonl", [lineage]),
        }
        manifest = {
            "schema_version": PROJECTION_SCHEMA,
            "status": "review_only",
            "training_authorized": False,
            "trainer_loadable": True,
            "format": "messages_jsonl_with_tools",
            "source": {
                "episode_file_sha256": episode_sha256,
                "episode_records": source_records,
                "trace_file_sha256": trace_sha256,
                "episode_id": joined["episode_id"],
            },
            "contract": {
                "registry_revision": joined["registry_revision"],
                "registry_sha256": joined["registry_sha256"],
                "skill_revision": joined["skill_revision"],
                "environment_revision": joined["environment_revision"],
                "verifier_revision": joined["verifier_revision"],
                "terminal_status": joined["terminal_status"],
                "verification_result": joined["verification_result"],
                "rewards": "not_exported",
            },
            "quality": {
                "tier": "harness_verified_tool_sft",
                "limitations": ["training_authorized_false", "reward_not_exported"],
            },
            "counts": {"input_records": 1, "tool_sft": 1},
            "validation": {
                "trace_identity": "passed",
                "registry_join": "passed",
                "skill_gate": "passed",
                "call_observation_join": "passed",
                "verification": "passed",
                "privacy_reasoning_firewall": "passed",
                "reward": "not_present",
            },
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = _sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", type=Path)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            project_harness_tool_sft(args.episode_path, args.trace_path, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
