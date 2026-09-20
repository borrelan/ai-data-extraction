#!/usr/bin/env python3
"""Stream the bounded candidate tool-trace partition into review trajectories.

This is an archival salvage projection, not tool-SFT authorization.  It keeps
messages and normalized action/observation events when they are structurally
recoverable, binds each row to its source and parent session, and records the
missing schema/verifier boundary explicitly instead of guessing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from build_training_data import no_reasoning_content, trainer_marker_count
from quality_rules import scan_record


PILOT_SCHEMA = "ai-data-extraction/historical-tool-trajectory-pilot/v1"
DECISION_SCHEMA = "ai-data-extraction/historical-tool-trajectory-decision/v1"
EVENT_SCHEMA = "ai-data-extraction/event/v1"
FINAL_ANSWER_SCHEMA = "ai-data-extraction/trajectory-final-answer-sft/v1"
FINAL_ANSWER_DECISION_SCHEMA = (
    "ai-data-extraction/trajectory-final-answer-decision/v1"
)
SOURCE_DATASET = "tool_traces"
MIN_FINAL_ANSWER_CHARS = 200
MAX_FINAL_ANSWER_CHARS = 32_768
SUGGESTION_MODE_MARKER = "[suggestion mode:"
INCOMPLETE_ANSWER_RE = re.compile(
    r"\b(?:"
    r"interrupted|"
    r"unable to (?:fully )?complete|"
    r"cannot complete|"
    r"could not complete|"
    r"must terminate|"
    r"have to terminate|"
    r"cut short|"
    r"reach(?:ed|ing) the (?:maximum (?:number of )?)?turn limit|"
    r"maximum (?:number of )?turns"
    r")\b",
    re.IGNORECASE,
)
PROVIDER_ARTIFACT_RE = re.compile(
    r"\[(?:thought|tool)\s*:|api error:\s*claude's response exceeded",
    re.IGNORECASE,
)


class TrajectoryPilotError(ValueError):
    """Raised when the source partition cannot be bound safely."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    raw = canonical_json(value) + b"\n"
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return {"bytes": len(raw), "sha256": sha256_bytes(raw)}


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None
        self.digest = hashlib.sha256()
        self.records = 0
        self.bytes = 0

    def __enter__(self) -> "JsonlWriter":
        self.handle = self.path.open("wb")
        return self

    def write(self, value: Any) -> None:
        if self.handle is None:
            raise RuntimeError("writer is not open")
        raw = canonical_json(value) + b"\n"
        self.handle.write(raw)
        self.digest.update(raw)
        self.records += 1
        self.bytes += len(raw)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.close()

    def descriptor(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryPilotError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_child(root: Path, relative: str, field: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise TrajectoryPilotError(f"{field} escapes release directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_source_spec(release_dir: Path) -> tuple[Path, dict[str, Any], str]:
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TrajectoryPilotError("release manifest must be an object")
    dataset = manifest.get("datasets", {}).get(SOURCE_DATASET)
    if not isinstance(dataset, dict):
        raise TrajectoryPilotError("release manifest has no tool_traces dataset")
    partitions = dataset.get("partitions")
    if not isinstance(partitions, dict) or not isinstance(partitions.get("candidate"), dict):
        raise TrajectoryPilotError("release manifest has no candidate tool partition")
    spec = partitions["candidate"]
    path = _safe_child(release_dir, _non_empty(spec.get("path"), "candidate.path"), "candidate.path")
    return path, spec, manifest_sha256


def _parent_id(row: dict[str, Any]) -> str:
    lineage = row.get("lineage")
    metadata = row.get("metadata")
    for container in (lineage, metadata):
        if not isinstance(container, dict):
            continue
        for key in ("parent_record_sha256", "_chunk_parent_record_sha256", "parent_unit_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    raise TrajectoryPilotError("source row has no parent identity")


def _source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise TrajectoryPilotError("source row has no metadata")
    quality = row.get("quality")
    if not isinstance(quality, dict):
        raise TrajectoryPilotError("source row has no quality metadata")
    return {
        "provider": metadata.get("provider"),
        "agent": metadata.get("source_label") or metadata.get("provider"),
        "source_file_sha256": metadata.get("source_file_sha256"),
        "source_file_name": metadata.get("source_file_name"),
        "source_line": metadata.get("source_line"),
        "source_record_sha256": metadata.get("source_record_sha256"),
        "segment_record_sha256": metadata.get("segment_record_sha256"),
        "parser_revision": metadata.get("parser_revision"),
        "model_tier": quality.get("model_tier"),
        "model_tier_basis": quality.get("model_tier_basis"),
        "model_tier_registry_revision": quality.get("model_tier_registry_revision"),
        "session_quality_id": quality.get("session_quality_id"),
        "session_quality_scope": quality.get("session_quality_scope"),
    }


def _event_summary(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    events = row.get("events")
    if not isinstance(events, list) or not events:
        return {}, ["no_events"]
    event_ids: set[str] = set()
    ordinals: set[int] = set()
    calls: set[str] = set()
    observations: set[str] = set()
    actions = 0
    observation_count = 0
    reasons: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            reasons.append("event_not_object")
            continue
        if event.get("schema_version") != EVENT_SCHEMA:
            reasons.append("event_schema_mismatch")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            reasons.append("event_identity_missing")
        elif event_id in event_ids:
            reasons.append("duplicate_event_identity")
        else:
            event_ids.add(event_id)
        ordinal = event.get("ordinal")
        if not isinstance(ordinal, int) or ordinal in ordinals:
            reasons.append("event_ordinal_invalid_or_duplicate")
        else:
            ordinals.add(ordinal)
        kind = event.get("kind")
        call_id = event.get("call_id")
        if kind == "action":
            actions += 1
            if not isinstance(call_id, str) or not call_id:
                reasons.append("action_call_id_missing")
            elif call_id in calls:
                reasons.append("duplicate_action_call_id")
            else:
                calls.add(call_id)
        elif kind == "observation":
            observation_count += 1
            if not isinstance(call_id, str) or not call_id:
                reasons.append("observation_call_id_missing")
            elif call_id in observations:
                reasons.append("duplicate_observation_call_id")
            else:
                observations.add(call_id)
    if calls != observations:
        reasons.append("event_call_observation_mismatch")
    if actions == 0:
        reasons.append("no_actions")
    return {
        "event_count": len(events),
        "action_count": actions,
        "observation_count": observation_count,
        "matched_call_count": len(calls & observations),
        "open_action_count": len(calls - observations),
        "unmatched_observation_count": len(observations - calls),
    }, sorted(set(reasons))


def _messages_valid(row: dict[str, Any]) -> tuple[bool, str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, "messages_missing"
    call_ids: set[str] = set()
    observation_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            return False, "message_role_invalid"
        if not isinstance(message.get("content"), str):
            return False, "message_content_invalid"
        for call in message.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                return False, "tool_call_invalid"
            call_id = call.get("id") or call.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                return False, "tool_call_identity_invalid_or_duplicate"
            call_ids.add(call_id)
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id or call_id in observation_ids:
                return False, "tool_observation_identity_invalid_or_duplicate"
            observation_ids.add(call_id)
    if call_ids != observation_ids:
        return False, "message_call_observation_mismatch"
    return True, None


def _split_for_parent(parent_id: str) -> str:
    bucket = int(sha256_bytes(parent_id.encode("utf-8"))[:8], 16) % 100
    return "validation" if bucket >= 90 else "train"


def _text_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_json(value).decode("utf-8")


def _loader_safe_messages(messages: list[Any]) -> list[Any]:
    """Keep tool calls, but make arguments one stable loader type."""

    safe: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            safe.append(message)
            continue
        copied = dict(message)
        calls = copied.get("tool_calls")
        if isinstance(calls, list):
            safe_calls: list[Any] = []
            for call in calls:
                if not isinstance(call, dict):
                    safe_calls.append(call)
                    continue
                safe_call = dict(call)
                function = safe_call.get("function")
                if isinstance(function, dict):
                    safe_function = dict(function)
                    if "arguments" in safe_function:
                        safe_function["arguments"] = _json_text(safe_function["arguments"])
                    safe_call["function"] = safe_function
                safe_calls.append(safe_call)
            copied["tool_calls"] = safe_calls
        safe.append(copied)
    return safe


def _loader_safe_events(events: list[Any]) -> list[str]:
    """Preserve each exact normalized event as one JSON-text list element."""

    return [canonical_json(event).decode("utf-8") for event in events]


def _final_answer_candidate(
    row: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    split: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Project one tool-bearing trajectory into tool-free final-answer SFT.

    Historical actions remain evidence only.  This projection uses every
    non-empty user turn before the terminal assistant answer as prompt context,
    and only the assistant text after the last tool call/observation as the
    supervised completion.
    """

    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, ["messages_missing"]

    last_tool_index = -1
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            last_tool_index = index

    terminal_index = -1
    completion = ""
    for index, message in enumerate(messages):
        if index <= last_tool_index or not isinstance(message, dict):
            continue
        content = message.get("content")
        if (
            message.get("role") == "assistant"
            and isinstance(content, str)
            and content.strip()
        ):
            terminal_index = index
            completion = content.strip()

    reasons: list[str] = []
    if terminal_index < 0:
        reasons.append("terminal_assistant_answer_missing")
        return None, reasons

    user_turns = [
        message["content"].strip()
        for message in messages[:terminal_index]
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if not user_turns:
        reasons.append("user_task_missing")
    if any(SUGGESTION_MODE_MARKER in turn.lower() for turn in user_turns):
        reasons.append("provider_suggestion_mode_task")
    if len(completion) < MIN_FINAL_ANSWER_CHARS:
        reasons.append("completion_signal_too_small")
    if len(completion) > MAX_FINAL_ANSWER_CHARS:
        reasons.append("completion_size_outlier")
    if INCOMPLETE_ANSWER_RE.search(completion):
        reasons.append("explicitly_incomplete_or_interrupted_answer")
    if PROVIDER_ARTIFACT_RE.search(completion):
        reasons.append("provider_runtime_artifact_in_answer")
    if trajectory.get("lineage", {}).get("continuation_status") == "middle":
        reasons.append("middle_fragment_not_terminal_episode")

    projected_messages = [
        {"role": "user", "content": "\n\n".join(user_turns)},
        {"role": "assistant", "content": completion},
    ]
    trainer_view = {"messages": projected_messages}
    if not no_reasoning_content(trainer_view) or trainer_marker_count(trainer_view):
        reasons.append("reasoning_or_trainer_marker_in_projection")
    findings = scan_record(trainer_view)
    if findings.has_hard_privacy_issue:
        reasons.append("hard_privacy_finding_in_projection")
    if findings.has_marker:
        reasons.append("privacy_or_reasoning_marker_in_projection")
    if reasons:
        return None, sorted(set(reasons))

    parent_id = trajectory["lineage"]["parent_record_sha256"]
    source_example_id = trajectory["example_id"]
    example_id = "sha256:" + sha256_bytes(
        canonical_json(
            {
                "schema_version": FINAL_ANSWER_SCHEMA,
                "parent_record_sha256": parent_id,
                "source_example_id": source_example_id,
                "messages": projected_messages,
            }
        )
    )
    source_task_tags = [
        tag
        for tag in trajectory.get("tags", [])
        if isinstance(tag, str) and tag.startswith("task:")
    ]
    candidate = {
        "schema_version": FINAL_ANSWER_SCHEMA,
        "example_id": example_id,
        "split": split,
        "messages": projected_messages,
        "provider": trajectory.get("provider"),
        "agent": trajectory.get("agent"),
        "model_tier": trajectory.get("model_tier"),
        "quality_tier": "candidate",
        "quality_reason": [
            "final_answer_projected_from_tool_trajectory",
            "not_human_adjudicated",
            "outcome_unverified",
            "tool_history_excluded",
        ],
        "training_authorized": False,
        "privacy": {
            "state": "review_required",
            "eligible_for_training": False,
            "source_reason": trajectory.get("privacy", {}).get("source_reason", ""),
            "structural_redactions": trajectory.get("privacy", {}).get(
                "structural_redactions", 0
            ),
        },
        "projection": {
            "kind": "terminal_answer_without_tool_history",
            "prompt_construction": "all_user_turns_before_terminal_answer_joined_with_blank_line",
            "source_tool_schema": "not_observed_not_required_for_final_answer_sft",
            "completion_chars": len(completion),
            "prompt_chars": sum(len(turn) for turn in user_turns),
            "source_user_turns": len(user_turns),
        },
        "tags": sorted(
            set(source_task_tags)
            | {
                "lane:ordinary-sft",
                "outcome:unknown",
                "projection:trajectory-final-answer",
                "source:tool-bearing-trajectory",
                f"tier:{str(trajectory.get('model_tier', '')).replace('_', '-')}",
            }
        ),
        "lineage": {
            **trajectory["lineage"],
            "source_trajectory_example_id": source_example_id,
            "projection_schema_version": FINAL_ANSWER_SCHEMA,
            "tool_history_retained_in_source_artifact": True,
            "tool_history_in_trainer_messages": False,
        },
    }
    return candidate, []


def _candidate_row(
    row: dict[str, Any],
    *,
    split: str,
    event_summary: dict[str, Any],
) -> dict[str, Any]:
    metadata = _source_metadata(row)
    quality = row["quality"]
    privacy = row.get("privacy") if isinstance(row.get("privacy"), dict) else {}
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    parent_id = _parent_id(row)
    quality_flags = quality.get("session_quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []
    tool_families = quality.get("tool_families")
    if not isinstance(tool_families, list):
        tool_families = []
    source_tags = row.get("tags")
    if not isinstance(source_tags, list):
        source_tags = []
    return {
        "schema_version": PILOT_SCHEMA,
        "example_id": _non_empty(row.get("example_id"), "example_id"),
        "split": split,
        "provider": metadata["provider"],
        "agent": metadata["agent"],
        "model_tier": metadata["model_tier"],
        "quality_tier": "candidate",
        "quality_reason": sorted(
            set(str(flag) for flag in quality_flags)
            | {"outcome_unverified", "tool_schema_not_observed", "not_human_adjudicated"}
        ),
        "session_quality": {
            "gate": _text_or_empty(quality.get("session_quality_gate")),
            "flags": quality_flags,
            "id": _text_or_empty(quality.get("session_quality_id")),
            "scope": _text_or_empty(quality.get("session_quality_scope")),
            "assessment_json": _json_text(quality.get("session_quality_assessment", {})),
        },
        "privacy": {
            "state": "review_required",
            "eligible_for_training": False,
            "source_reason": _text_or_empty(privacy.get("reason")),
            "structural_redactions": int(privacy.get("structural_redactions") or 0),
        },
        "tool_contract": {
            "schema_status": "not_observed",
            "registry_revision": "",
            "verification_status": "not_observed",
            "reward_status": "not_exported",
        },
        "tool_families": sorted(set(str(item) for item in tool_families)),
        "tags": sorted(set(str(tag) for tag in source_tags) | {"trajectory:historical-review"}),
        "lineage": {
            "source_dataset": SOURCE_DATASET,
            "source_example_id": row.get("source_example_id") or row.get("example_id"),
            "source_file_sha256": _text_or_empty(metadata["source_file_sha256"]),
            "source_file_name": _text_or_empty(metadata["source_file_name"]),
            "source_line": int(metadata["source_line"] or 0),
            "source_record_sha256": _text_or_empty(metadata["source_record_sha256"]),
            "segment_record_sha256": _text_or_empty(metadata["segment_record_sha256"]),
            "parent_record_sha256": parent_id,
            "parser_revision": _text_or_empty(metadata["parser_revision"]),
            "model_tier_basis": _text_or_empty(metadata["model_tier_basis"]),
            "model_tier_registry_revision": _text_or_empty(metadata["model_tier_registry_revision"]),
            "source_message_range_json": _json_text(lineage.get("source_message_range", {})),
            "chunk_index": int(lineage.get("chunk_index") or 0),
            "chunk_count": int(lineage.get("chunk_count") or 0),
            "continuation_status": _text_or_empty(lineage.get("continuation_status")),
            "cut_reason": _text_or_empty(lineage.get("cut_reason")),
            "previous_example_id": _text_or_empty(lineage.get("previous_example_id")),
            "next_example_id": _text_or_empty(lineage.get("next_example_id")),
        },
        "events_summary": event_summary,
        "messages": _loader_safe_messages(row["messages"]),
        "events": _loader_safe_events(row["events"]),
    }


def export_historical_tool_trajectory_pilot(
    *,
    release_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    source_path, source_spec, release_manifest_sha256 = _load_source_spec(release_dir)
    expected_records = source_spec.get("records")
    expected_sha256 = _non_empty(source_spec.get("sha256"), "candidate.sha256")
    if not isinstance(expected_records, int) or expected_records < 1:
        raise TrajectoryPilotError("candidate partition record count is invalid")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    decisions = Counter()
    provider_counts = Counter()
    tier_counts = Counter()
    family_counts = Counter()
    parent_splits: dict[str, str] = {}
    source_examples: set[str] = set()
    rows_read = 0
    selected = 0
    excluded = 0
    final_answer_selected = 0
    final_answer_decisions = Counter()
    final_answer_provider_counts = Counter()
    final_answer_parents: set[str] = set()
    final_answer_message_hashes: set[str] = set()
    source_digest = hashlib.sha256()
    source_bytes = 0
    train_writer = JsonlWriter(staging / "train.jsonl")
    validation_writer = JsonlWriter(staging / "validation.jsonl")
    decisions_writer = JsonlWriter(staging / "decisions.jsonl")
    final_train_writer = JsonlWriter(staging / "final_answer_sft_train.jsonl")
    final_validation_writer = JsonlWriter(staging / "final_answer_sft_validation.jsonl")
    final_decisions_writer = JsonlWriter(staging / "final_answer_decisions.jsonl")
    try:
        with (
            source_path.open("rb") as source,
            train_writer as train,
            validation_writer as validation,
            decisions_writer as decision_file,
            final_train_writer as final_train,
            final_validation_writer as final_validation,
            final_decisions_writer as final_decision_file,
        ):
            for line_number, raw in enumerate(source, 1):
                rows_read += 1
                source_bytes += len(raw)
                source_digest.update(raw)
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise TrajectoryPilotError(f"source line {line_number} is invalid JSON") from exc
                if not isinstance(row, dict):
                    raise TrajectoryPilotError(f"source line {line_number} is not an object")
                source_example_id = row.get("example_id")
                if not isinstance(source_example_id, str) or not source_example_id:
                    raise TrajectoryPilotError(f"source line {line_number} has no example_id")
                if source_example_id in source_examples:
                    raise TrajectoryPilotError(f"duplicate source example_id: {source_example_id}")
                source_examples.add(source_example_id)
                reasons: list[str] = []
                if row.get("dataset") != "tool_trace":
                    reasons.append("dataset_not_tool_trace")
                try:
                    parent_id = _parent_id(row)
                except TrajectoryPilotError:
                    parent_id = ""
                    reasons.append("parent_identity_missing")
                message_ok, message_reason = _messages_valid(row)
                if not message_ok and message_reason:
                    reasons.append(message_reason)
                event_summary, event_reasons = _event_summary(row)
                reasons.extend(event_reasons)
                if not no_reasoning_content(row):
                    reasons.append("reasoning_content_present")
                if trainer_marker_count(row):
                    reasons.append("trainer_marker_present")
                findings = scan_record(row)
                if findings.has_hard_privacy_issue:
                    reasons.append("hard_privacy_finding")
                if findings.has_marker:
                    reasons.append("privacy_marker_finding")
                split = _split_for_parent(parent_id) if parent_id else "unassigned"
                if parent_id:
                    prior_split = parent_splits.setdefault(parent_id, split)
                    if prior_split != split:
                        raise TrajectoryPilotError("deterministic parent split collision")
                selected_row: dict[str, Any] | None = None
                if not reasons:
                    selected_row = _candidate_row(row, split=split, event_summary=event_summary)
                    destination = train if split == "train" else validation
                    destination.write(selected_row)
                    selected += 1
                    metadata = _source_metadata(row)
                    provider_counts[str(metadata["provider"])] += 1
                    tier_counts[str(metadata["model_tier"])] += 1
                    for family in selected_row["tool_families"]:
                        family_counts[family] += 1
                    decisions["selected"] += 1
                else:
                    excluded += 1
                    for reason in set(reasons):
                        decisions[reason] += 1
                decision_file.write(
                    {
                        "schema_version": DECISION_SCHEMA,
                        "source_example_id": source_example_id,
                        "parent_record_sha256": parent_id or None,
                        "source_line": line_number,
                        "selected": selected_row is not None,
                        "split": split if selected_row is not None else None,
                        "reasons": sorted(set(reasons))
                        if reasons
                        else ["candidate_structurally_recoverable", "schema_not_observed_review_only"],
                        "quality_gate": row.get("quality", {}).get("session_quality_gate"),
                        "model_tier": row.get("quality", {}).get("model_tier"),
                    }
                )

                final_candidate: dict[str, Any] | None = None
                if selected_row is None:
                    final_reasons = ["source_trajectory_not_structurally_recoverable"]
                    final_reasons.extend(sorted(set(reasons)))
                else:
                    final_candidate, final_reasons = _final_answer_candidate(
                        row,
                        selected_row,
                        split=split,
                    )
                    if final_candidate is not None:
                        message_hash = sha256_bytes(
                            canonical_json(final_candidate["messages"])
                        )
                        if message_hash in final_answer_message_hashes:
                            final_candidate = None
                            final_reasons = ["duplicate_prompt_completion"]
                        else:
                            final_answer_message_hashes.add(message_hash)

                if final_candidate is not None:
                    destination = (
                        final_train if split == "train" else final_validation
                    )
                    destination.write(final_candidate)
                    final_answer_selected += 1
                    final_answer_decisions["selected"] += 1
                    final_answer_provider_counts[
                        str(final_candidate.get("provider"))
                    ] += 1
                    final_answer_parents.add(parent_id)
                    final_decision_reasons = ["candidate_requires_review"]
                else:
                    for reason in set(final_reasons):
                        final_answer_decisions[reason] += 1
                    final_decision_reasons = sorted(set(final_reasons))
                final_decision_file.write(
                    {
                        "schema_version": FINAL_ANSWER_DECISION_SCHEMA,
                        "source_example_id": source_example_id,
                        "parent_record_sha256": parent_id or None,
                        "source_line": line_number,
                        "selected": final_candidate is not None,
                        "example_id": (
                            final_candidate["example_id"]
                            if final_candidate is not None
                            else None
                        ),
                        "split": split if final_candidate is not None else None,
                        "reasons": final_decision_reasons,
                        "training_authorized": False,
                    }
                )
        actual_sha256 = source_digest.hexdigest()
        if rows_read != expected_records:
            raise TrajectoryPilotError(
                f"source record count mismatch: expected {expected_records}, got {rows_read}"
            )
        if actual_sha256 != expected_sha256:
            raise TrajectoryPilotError(
                f"source SHA mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        if source_bytes != source_spec.get("bytes"):
            raise TrajectoryPilotError(
                f"source byte count mismatch: expected {source_spec.get('bytes')}, got {source_bytes}"
            )
        if selected == 0:
            raise TrajectoryPilotError("no structurally recoverable candidate trajectories")
        files = {
            "train.jsonl": train.descriptor(),
            "validation.jsonl": validation.descriptor(),
            "decisions.jsonl": decisions_writer.descriptor(),
            "final_answer_sft_train.jsonl": final_train_writer.descriptor(),
            "final_answer_sft_validation.jsonl": final_validation_writer.descriptor(),
            "final_answer_decisions.jsonl": final_decisions_writer.descriptor(),
        }
        manifest = {
            "schema_version": PILOT_SCHEMA,
            "status": "review_only",
            "trainer_loadable": True,
            "training_authorized": False,
            "format": "messages_plus_events_jsonl",
            "source": {
                "release_manifest_sha256": release_manifest_sha256,
                "release_builder": "1.3.6",
                "partition": SOURCE_DATASET + ":candidate",
                "path": str(source_path),
                "records": rows_read,
                "bytes": source_bytes,
                "sha256": actual_sha256,
            },
            "counts": {
                "input_records": rows_read,
                "selected_records": selected,
                "excluded_records": excluded,
                "train": train.records,
                "validation": validation.records,
                "unique_parent_sessions": len(parent_splits),
                "final_answer_sft": final_answer_selected,
                "final_answer_sft_train": final_train_writer.records,
                "final_answer_sft_validation": final_validation_writer.records,
                "final_answer_unique_parent_sessions": len(final_answer_parents),
            },
            "quality": {
                "quality_tier": "candidate",
                "model_tier_counts": dict(sorted(tier_counts.items())),
                "provider_counts": dict(sorted(provider_counts.items())),
                "tool_family_counts": dict(sorted(family_counts.items())),
                "limitations": [
                    "privacy_approval_not_granted",
                    "outcome_unverified",
                    "tool_schema_not_observed",
                    "verifier_not_observed",
                    "rewards_not_exported",
                ],
            },
            "decisions": {
                "records": decisions_writer.records,
                "reason_counts": dict(sorted(decisions.items())),
            },
            "final_answer_sft": {
                "schema_version": FINAL_ANSWER_SCHEMA,
                "status": "review_only",
                "training_authorized": False,
                "format": "messages_jsonl_without_tools",
                "provider_counts": dict(sorted(final_answer_provider_counts.items())),
                "decision_records": final_decisions_writer.records,
                "decision_reason_counts": dict(
                    sorted(final_answer_decisions.items())
                ),
                "quality_limitations": [
                    "not_human_adjudicated",
                    "outcome_unverified",
                    "privacy_approval_not_granted",
                ],
            },
            "validation": {
                "source_partition_binding": "passed",
                "streaming_projection": "passed",
                "parent_disjoint": "pending_loader_audit",
                "privacy_reasoning_firewall": "passed_for_selected_rows",
                "tool_schema": "not_observed_not_inferred",
                "reward": "not_present",
                "loader": "pending",
                "final_answer_parent_disjoint": "passed_by_deterministic_parent_split",
                "final_answer_tool_history": "excluded_from_trainer_messages",
                "final_answer_privacy_reasoning_firewall": "passed_for_selected_rows",
            },
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_historical_tool_trajectory_pilot(
        release_dir=args.release_dir,
        output_dir=args.output_dir,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_SCHEMA",
    "FINAL_ANSWER_DECISION_SCHEMA",
    "FINAL_ANSWER_SCHEMA",
    "PILOT_SCHEMA",
    "TrajectoryPilotError",
    "export_historical_tool_trajectory_pilot",
]
