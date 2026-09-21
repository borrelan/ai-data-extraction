#!/usr/bin/env python3
"""Build a Qwen/TRL SFT projection from provider-neutral tool trajectories.

The source chat contract stores function arguments as JSON text. Qwen's chat
template expects a JSON object so it can render named parameters. Normalize
that difference once at this target-specific boundary; never rewrite source
artifacts or guess malformed arguments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from build_training_data import (
    ACTION_EVIDENCE_SCHEMA_VERSION,
    ACTION_WINDOW_SCHEMA_VERSION,
)


ADAPTER_SCHEMA = "ai-data-extraction/trl-qwen-tool-sft-adapter/v1"
EXAMPLE_SCHEMA = "ai-data-extraction/qwen-tool-sft-example/v1"
ACTION_WINDOW_EXAMPLE_SCHEMA = "ai-data-extraction/qwen-action-window-sft-example/v1"
ACTION_WINDOW_BUNDLE_SCHEMA = "ai-data-extraction/qwen-action-window-sft-candidate/v1"
VERIFIED_POSITIVE_STATUS = "verified_positive"
VERIFIED_QUALITY_STAGES = frozenset({"verified", "verifier_backed"})
VERIFIED_SOURCES = frozenset({"adjudication", "executable_replay", "harness"})


class NormalizationIssue(str, Enum):
    MESSAGES_NOT_LIST = "messages_not_list"
    MESSAGE_NOT_OBJECT = "message_not_object"
    TOOL_CALLS_NOT_LIST = "tool_calls_not_list"
    TOOL_CALLS_NOT_ASSISTANT = "tool_calls_not_assistant"
    TOOL_CALL_NOT_OBJECT = "tool_call_not_object"
    FUNCTION_NOT_OBJECT = "function_not_object"
    FUNCTION_NAME_MISSING = "function_name_missing"
    ARGUMENTS_MISSING = "arguments_missing"
    ARGUMENTS_INVALID_JSON = "arguments_invalid_json"
    ARGUMENTS_DUPLICATE_KEYS = "arguments_duplicate_keys"
    ARGUMENTS_NOT_OBJECT = "arguments_not_object"
    NO_TOOL_CALLS = "no_tool_calls"


class ActionWindowIssue(str, Enum):
    ROW_NOT_OBJECT = "row_not_object"
    SOURCE_SCHEMA_UNSUPPORTED = "source_schema_unsupported"
    EVIDENCE_SCHEMA_UNSUPPORTED = "evidence_schema_unsupported"
    POSITIVE_TARGET_NOT_VERIFIED = "positive_target_not_verified"
    VERIFICATION_SOURCE_UNSUPPORTED = "verification_source_unsupported"
    PRIVACY_NOT_TRAINING_ELIGIBLE = "privacy_not_training_eligible"
    WINDOW_ID_MISSING = "window_id_missing"
    EPISODE_ID_MISSING = "episode_id_missing"
    PARENT_ID_MISSING = "parent_id_missing"
    ACTION_NOT_USE = "action_not_use"
    CONTEXT_MESSAGES_INVALID = "context_messages_invalid"
    MESSAGE_ROLE_INVALID = "message_role_invalid"
    MESSAGE_CONTENT_INVALID = "message_content_invalid"
    PROMPT_USER_MISSING = "prompt_user_missing"
    TARGET_NOT_ASSISTANT = "target_not_assistant"
    TARGET_CALL_ID_MISSING = "target_call_id_missing"
    TARGET_CALL_ID_MISMATCH = "target_call_id_mismatch"
    TARGET_TOOL_NAME_MISMATCH = "target_tool_name_mismatch"
    TARGET_ARGUMENTS_INVALID = "target_arguments_invalid"
    TARGET_ARGUMENTS_MISMATCH = "target_arguments_mismatch"
    OBSERVATION_MISSING = "observation_missing"
    OBSERVATION_CALL_ID_MISMATCH = "observation_call_id_mismatch"
    QUALITY_NOT_VERIFIED = "quality_not_verified"
    MODEL_TIER_NOT_SELECTED = "model_tier_not_selected"
    TAGS_INVALID = "tags_invalid"
    PROVIDER_NOT_IDENTIFIED = "provider_not_identified"


@dataclass(frozen=True)
class NormalizationResult:
    messages: list[dict[str, Any]] | None
    tool_call_count: int
    normalized_argument_count: int
    issues: tuple[NormalizationIssue, ...]


class _DuplicateJSONKey(ValueError):
    pass


class _NonJSONConstant(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise _NonJSONConstant


def _object_arguments(
    value: Any,
) -> tuple[dict[str, Any] | None, NormalizationIssue | None]:
    if isinstance(value, dict):
        return copy.deepcopy(value), None
    if not isinstance(value, str):
        return None, NormalizationIssue.ARGUMENTS_NOT_OBJECT
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateJSONKey:
        return None, NormalizationIssue.ARGUMENTS_DUPLICATE_KEYS
    except (json.JSONDecodeError, _NonJSONConstant, ValueError):
        return None, NormalizationIssue.ARGUMENTS_INVALID_JSON
    if not isinstance(parsed, dict):
        return None, NormalizationIssue.ARGUMENTS_NOT_OBJECT
    return parsed, None


def normalize_tool_messages(messages: Any) -> NormalizationResult:
    """Convert valid JSON-string tool arguments to objects without mutating input.

    A row is rejected as a unit if any tool call is malformed. This avoids
    teaching an incomplete or invalid call while preserving the original row
    and its provenance in the source artifact.
    """

    if not isinstance(messages, list):
        return NormalizationResult(
            messages=None,
            tool_call_count=0,
            normalized_argument_count=0,
            issues=(NormalizationIssue.MESSAGES_NOT_LIST,),
        )

    normalized = copy.deepcopy(messages)
    issues: list[NormalizationIssue] = []
    tool_call_count = 0
    normalized_argument_count = 0

    for message in normalized:
        if not isinstance(message, dict):
            issues.append(NormalizationIssue.MESSAGE_NOT_OBJECT)
            continue

        calls = message.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list):
            issues.append(NormalizationIssue.TOOL_CALLS_NOT_LIST)
            continue
        if calls and message.get("role") != "assistant":
            issues.append(NormalizationIssue.TOOL_CALLS_NOT_ASSISTANT)

        for call in calls:
            tool_call_count += 1
            if not isinstance(call, dict):
                issues.append(NormalizationIssue.TOOL_CALL_NOT_OBJECT)
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                issues.append(NormalizationIssue.FUNCTION_NOT_OBJECT)
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                issues.append(NormalizationIssue.FUNCTION_NAME_MISSING)
            if "arguments" not in function:
                issues.append(NormalizationIssue.ARGUMENTS_MISSING)
                continue

            parsed, argument_issue = _object_arguments(function["arguments"])
            if argument_issue is not None:
                issues.append(argument_issue)
                continue
            function["arguments"] = parsed
            normalized_argument_count += 1

    if tool_call_count == 0:
        issues.append(NormalizationIssue.NO_TOOL_CALLS)

    return NormalizationResult(
        messages=normalized if not issues else None,
        tool_call_count=tool_call_count,
        normalized_argument_count=normalized_argument_count,
        issues=tuple(issues),
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_jsonl(value)).hexdigest()


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    return sorted({tag[len(prefix) :] for tag in tags if tag.startswith(prefix)})


def _action_window_prompt_completion(
    row: Any,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    reasons: set[str] = set()
    if not isinstance(row, dict):
        return None, (ActionWindowIssue.ROW_NOT_OBJECT.value,)

    if row.get("schema_version") != ACTION_WINDOW_SCHEMA_VERSION:
        reasons.add(ActionWindowIssue.SOURCE_SCHEMA_UNSUPPORTED.value)
    evidence = row.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    if evidence.get("schema_version") != ACTION_EVIDENCE_SCHEMA_VERSION:
        reasons.add(ActionWindowIssue.EVIDENCE_SCHEMA_UNSUPPORTED.value)
    if evidence.get("positive_target_status") != VERIFIED_POSITIVE_STATUS:
        reasons.add(ActionWindowIssue.POSITIVE_TARGET_NOT_VERIFIED.value)
    verification = row.get("verification")
    verification = verification if isinstance(verification, dict) else {}
    if verification.get("source") not in VERIFIED_SOURCES:
        reasons.add(ActionWindowIssue.VERIFICATION_SOURCE_UNSUPPORTED.value)
    privacy = row.get("privacy")
    privacy = privacy if isinstance(privacy, dict) else {}
    if privacy.get("eligible_for_training") is not True:
        reasons.add(ActionWindowIssue.PRIVACY_NOT_TRAINING_ELIGIBLE.value)

    window_id = row.get("window_id")
    episode_id = row.get("episode_id")
    parent_sha = _window_parent_sha(row)
    if not isinstance(window_id, str) or not window_id:
        reasons.add(ActionWindowIssue.WINDOW_ID_MISSING.value)
    if not isinstance(episode_id, str) or not episode_id:
        reasons.add(ActionWindowIssue.EPISODE_ID_MISSING.value)
    if parent_sha is None:
        reasons.add(ActionWindowIssue.PARENT_ID_MISSING.value)

    decision = row.get("decision")
    if not isinstance(decision, dict) or decision.get("action") != "use":
        reasons.add(ActionWindowIssue.ACTION_NOT_USE.value)

    context = row.get("context")
    messages = context.get("messages") if isinstance(context, dict) else None
    if not isinstance(messages, list) or len(messages) < 2:
        reasons.add(ActionWindowIssue.CONTEXT_MESSAGES_INVALID.value)
        messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            reasons.add(ActionWindowIssue.MESSAGE_ROLE_INVALID.value)
            continue
        if not isinstance(message.get("content"), str):
            reasons.add(ActionWindowIssue.MESSAGE_CONTENT_INVALID.value)

    normalized = normalize_tool_messages(messages)
    reasons.update(issue.value for issue in normalized.issues)
    normalized_messages = normalized.messages

    source_call = row.get("tool_call")
    source_call = source_call if isinstance(source_call, dict) else {}
    call_id = source_call.get("call_id")
    tool_name = source_call.get("name")
    if not isinstance(call_id, str) or not call_id:
        reasons.add(ActionWindowIssue.TARGET_CALL_ID_MISSING.value)

    if normalized_messages:
        target_message = normalized_messages[-1]
        if target_message.get("role") != "assistant":
            reasons.add(ActionWindowIssue.TARGET_NOT_ASSISTANT.value)
        prompt = normalized_messages[:-1]
        if not any(message.get("role") == "user" for message in prompt):
            reasons.add(ActionWindowIssue.PROMPT_USER_MISSING.value)

        target_calls = target_message.get("tool_calls")
        target_calls = target_calls if isinstance(target_calls, list) else []
        matching_calls = [
            tool_call
            for tool_call in target_calls
            if isinstance(tool_call, dict) and tool_call.get("id") == call_id
        ]
        if len(matching_calls) != 1:
            reasons.add(ActionWindowIssue.TARGET_CALL_ID_MISMATCH.value)
        else:
            function = matching_calls[0].get("function")
            function = function if isinstance(function, dict) else {}
            if function.get("name") != tool_name:
                reasons.add(ActionWindowIssue.TARGET_TOOL_NAME_MISMATCH.value)
            event_arguments, argument_issue = _object_arguments(
                source_call.get("arguments")
            )
            if argument_issue is not None:
                reasons.add(ActionWindowIssue.TARGET_ARGUMENTS_INVALID.value)
            elif function.get("arguments") != event_arguments:
                reasons.add(ActionWindowIssue.TARGET_ARGUMENTS_MISMATCH.value)
    else:
        prompt = []
        target_message = None

    observation = row.get("observation")
    if not isinstance(observation, dict):
        reasons.add(ActionWindowIssue.OBSERVATION_MISSING.value)
    elif observation.get("call_id") != call_id:
        reasons.add(ActionWindowIssue.OBSERVATION_CALL_ID_MISMATCH.value)

    quality = row.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    if quality.get("stage") not in VERIFIED_QUALITY_STAGES:
        reasons.add(ActionWindowIssue.QUALITY_NOT_VERIFIED.value)
    if quality.get("model_tier") != "tier1_frontier":
        reasons.add(ActionWindowIssue.MODEL_TIER_NOT_SELECTED.value)

    tags = row.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        reasons.add(ActionWindowIssue.TAGS_INVALID.value)
        tags = []
    providers = _tag_values(tags, "provider:")
    if len(providers) != 1:
        reasons.add(ActionWindowIssue.PROVIDER_NOT_IDENTIFIED.value)

    if reasons or normalized_messages is None or target_message is None:
        return None, tuple(sorted(reasons or {"normalization_failed"}))

    agents = _tag_values(tags, "agent:")
    provider = providers[0]
    agent = agents[0] if len(agents) == 1 else "not_observed"
    trainer_tags = set(tags) | {
        "trainer:qwen-action-window-sft-candidate",
        "loss:completion-only",
    }
    output = {
        "schema_version": ACTION_WINDOW_EXAMPLE_SCHEMA,
        "example_id": window_id,
        "window_id": window_id,
        "episode_id": episode_id,
        "parent_record_sha256": parent_sha,
        "split": None,
        "provider": provider,
        "agent": agent,
        "model_tier": quality["model_tier"],
        "quality_tier": quality["stage"],
        "quality": copy.deepcopy(quality),
        "privacy": copy.deepcopy(row.get("privacy") or {}),
        "prompt": prompt,
        "completion": [target_message],
        "chat_template_kwargs": {"enable_thinking": False},
        "skill_trace": copy.deepcopy((decision or {}).get("skill") or {}),
        "observation_trace": {
            "call_id": observation.get("call_id"),
            "event_id": observation.get("event_id"),
            "status": observation.get("status", "unknown"),
            "payload_policy": quality.get("observation_output_policy", "unknown"),
        },
        "source_event": {
            "event_id": source_call.get("event_id"),
            "observation_event_id": observation.get("event_id"),
        },
        "tags": sorted(trainer_tags),
    }
    return output, ()


def _window_parent_sha(row: dict[str, Any]) -> str | None:
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("parent_record_sha256")
        if isinstance(value, str) and value:
            return value
    lineage = row.get("lineage")
    if isinstance(lineage, dict):
        value = lineage.get("parent_record_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _stable_parent_split(parent_sha: str, validation_fraction: float) -> str:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and below 0.5")
    bucket = int(hashlib.sha256(parent_sha.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < round(validation_fraction * 100) else "train"


def build_qwen_action_window_sft_candidate(
    source_release_dir: Path,
    output_dir: Path,
    *,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Create completion-only SFT rows from one existing action-window release."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and below 0.5")
    source_release_dir = source_release_dir.resolve()
    source_manifest_path = source_release_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha = _sha256_file(source_manifest_path)
    action_manifest = source_manifest.get("datasets", {}).get("action_windows", {})
    binding = action_manifest.get("partitions", {}).get("candidate")
    if not isinstance(binding, dict):
        raise ValueError("release manifest has no action_windows:candidate binding")
    relative_source_path = binding.get("path")
    if not isinstance(relative_source_path, str) or not relative_source_path:
        raise ValueError("candidate file binding has no relative path")
    source_path = (source_release_dir / relative_source_path).resolve()
    if not source_path.is_relative_to(source_release_dir):
        raise ValueError("candidate file binding escapes the release directory")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    before = source_path.stat()
    first_digest = hashlib.sha256()
    input_bytes = 0
    input_rows = 0
    prompt_completions: dict[str, Counter[str]] = defaultdict(Counter)
    first_pass_reasons: Counter[str] = Counter()
    first_pass_window_ids: set[str] = set()
    duplicate_window_ids = 0
    with source_path.open("rb") as source:
        for line_no, raw in enumerate(source, start=1):
            first_digest.update(raw)
            input_bytes += len(raw)
            if not raw.strip():
                raise ValueError(f"blank source line at {line_no}")
            input_rows += 1
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                first_pass_reasons["invalid_json"] += 1
                continue
            projection, reasons = _action_window_prompt_completion(row)
            first_pass_reasons.update(reasons)
            if projection is None:
                continue
            window_id = projection["window_id"]
            if window_id in first_pass_window_ids:
                duplicate_window_ids += 1
            first_pass_window_ids.add(window_id)
            prompt_sha = _json_sha256(projection["prompt"])
            completion_sha = _json_sha256(projection["completion"])
            prompt_completions[prompt_sha][completion_sha] += 1

    input_sha = first_digest.hexdigest()
    if input_sha != binding.get("sha256"):
        raise ValueError("action-window candidate digest differs from release manifest")
    if input_bytes != binding.get("bytes") or input_rows != binding.get("records"):
        raise ValueError("action-window candidate bytes/records differ from release manifest")

    conflicting_prompts = {
        prompt_sha
        for prompt_sha, completions in prompt_completions.items()
        if len(completions) > 1
    }
    conflicting_rows = sum(
        sum(prompt_completions[prompt_sha].values())
        for prompt_sha in conflicting_prompts
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    published = False
    try:
        file_meta: dict[str, dict[str, Any]] = {}
        split_counts: Counter[str] = Counter()
        provider_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        privacy_counts: Counter[str] = Counter()
        rejection_counts: Counter[str] = Counter()
        selected_parent_splits: dict[str, set[str]] = defaultdict(set)
        selected_parent_counts: Counter[str] = Counter()
        seen_window_ids: set[str] = set()
        seen_prompt_completions: set[tuple[str, str]] = set()
        decisions_hash = hashlib.sha256()
        decisions_bytes = 0
        decisions_count = 0
        lineage_hash = hashlib.sha256()
        lineage_bytes = 0
        lineage_count = 0
        selected_count = 0
        duplicate_count = 0
        second_digest = hashlib.sha256()
        second_bytes = 0
        second_rows = 0
        data_hashes = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
        data_bytes = Counter()

        decisions_path = stage_dir / "decisions.jsonl"
        lineage_path = stage_dir / "lineage.jsonl"
        with decisions_path.open("wb") as decisions_out, lineage_path.open("wb") as lineage_out:
            output_paths = {
                split: (stage_dir / f"{split}.jsonl").open("wb")
                for split in ("train", "validation")
            }
            try:
                with source_path.open("rb") as source:
                    for line_no, raw in enumerate(source, start=1):
                        second_digest.update(raw)
                        second_bytes += len(raw)
                        if not raw.strip():
                            raise ValueError(f"blank source line at {line_no}")
                        second_rows += 1
                        row_sha = hashlib.sha256(raw).hexdigest()
                        reasons: set[str] = set()
                        try:
                            row = json.loads(raw)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            row = None
                            reasons.add("invalid_json")

                        projection = None
                        prompt_sha = None
                        completion_sha = None
                        if isinstance(row, dict):
                            projection, projection_reasons = _action_window_prompt_completion(row)
                            reasons.update(projection_reasons)
                        else:
                            reasons.add(ActionWindowIssue.ROW_NOT_OBJECT.value)

                        window_id = row.get("window_id") if isinstance(row, dict) else None
                        parent_sha = _window_parent_sha(row) if isinstance(row, dict) else None
                        provider = "unknown"
                        if projection is not None:
                            provider = projection["provider"]
                            prompt_sha = _json_sha256(projection["prompt"])
                            completion_sha = _json_sha256(projection["completion"])
                            pair = (prompt_sha, completion_sha)
                            if prompt_sha in conflicting_prompts:
                                reasons.add("conflicting_completion_for_prompt")
                            elif window_id in seen_window_ids:
                                reasons.add("duplicate_window_id")
                            elif pair in seen_prompt_completions:
                                reasons.add("duplicate_prompt_completion")
                                duplicate_count += 1
                            else:
                                split = _stable_parent_split(parent_sha, validation_fraction)
                                projection["split"] = split
                                encoded = _canonical_jsonl(projection)
                                output_paths[split].write(encoded)
                                data_hashes[split].update(encoded)
                                data_bytes[split] += len(encoded)
                                split_counts[split] += 1
                                provider_counts[provider] += 1
                                quality_counts[str(projection["quality_tier"])] += 1
                                privacy = projection["privacy"]
                                privacy_counts[str(privacy.get("reason", "missing"))] += 1
                                selected_parent_splits[parent_sha].add(split)
                                selected_parent_counts[parent_sha] += 1
                                selected_prompt = prompt_sha
                                selected_completion = completion_sha
                                seen_prompt_completions.add(pair)
                                seen_window_ids.add(window_id)
                                selected_count += 1

                                lineage = {
                                    "example_id": window_id,
                                    "window_id": window_id,
                                    "episode_id": projection["episode_id"],
                                    "parent_record_sha256": parent_sha,
                                    "split": split,
                                    "provider": provider,
                                    "source_partition": "action_windows:candidate",
                                    "source_line": line_no,
                                    "source_row_sha256": row_sha,
                                    "prompt_sha256": selected_prompt,
                                    "completion_sha256": selected_completion,
                                    "source_event_id": projection["source_event"]["event_id"],
                                    "source_manifest_sha256": source_manifest_sha,
                                }
                                encoded_lineage = _canonical_jsonl(lineage)
                                lineage_out.write(encoded_lineage)
                                lineage_hash.update(encoded_lineage)
                                lineage_bytes += len(encoded_lineage)
                                lineage_count += 1

                        if projection is not None:
                            seen_window_ids.add(window_id)
                        if reasons:
                            rejection_counts.update(reasons)
                            decision = {
                                "window_id": window_id if isinstance(window_id, str) else None,
                                "decision": "review_only",
                                "reasons": sorted(reasons),
                                "source_row_sha256": row_sha,
                                "prompt_sha256": prompt_sha,
                                "completion_sha256": completion_sha,
                            }
                        else:
                            decision = {
                                "window_id": window_id,
                                "decision": "selected_candidate_sft",
                                "reasons": [],
                                "source_row_sha256": row_sha,
                                "prompt_sha256": prompt_sha,
                                "completion_sha256": completion_sha,
                            }
                        encoded_decision = _canonical_jsonl(decision)
                        decisions_out.write(encoded_decision)
                        decisions_hash.update(encoded_decision)
                        decisions_bytes += len(encoded_decision)
                        decisions_count += 1
            finally:
                for output in output_paths.values():
                    output.close()

        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("action-window candidate changed while building projection")
        if second_digest.hexdigest() != input_sha or second_bytes != input_bytes or second_rows != input_rows:
            raise RuntimeError("action-window source changed between streaming passes")
        if decisions_count != input_rows or lineage_count != selected_count:
            raise RuntimeError("candidate, decisions, and lineage counts do not reconcile")
        parent_overlaps = sum(len(splits) > 1 for splits in selected_parent_splits.values())
        if parent_overlaps:
            raise RuntimeError("parent split leakage in action-window SFT output")
        if selected_count == 0:
            raise ValueError("no action-window rows passed the strict SFT projection")

        for split in ("train", "validation"):
            file_meta[f"{split}.jsonl"] = {
                "records": split_counts[split],
                "bytes": data_bytes[split],
                "sha256": data_hashes[split].hexdigest(),
            }
        file_meta["decisions.jsonl"] = {
            "records": decisions_count,
            "bytes": decisions_bytes,
            "sha256": decisions_hash.hexdigest(),
        }
        file_meta["lineage.jsonl"] = {
            "records": lineage_count,
            "bytes": lineage_bytes,
            "sha256": lineage_hash.hexdigest(),
        }
        manifest = {
            "schema_version": ACTION_WINDOW_BUNDLE_SCHEMA,
            "status": "candidate_built_pending_trl_validation",
            "target": "Qwen chat template via Transformers/TRL",
            "format": "conversational_prompt_completion_jsonl",
            "source": {
                "release_dir": str(source_release_dir),
                "release_manifest_sha256": source_manifest_sha,
                "release_schema": source_manifest.get("schema_version"),
                "canonical_manifest_sha256": (
                    source_manifest.get("canonical_manifest", {}).get("sha256")
                    if isinstance(source_manifest.get("canonical_manifest"), dict)
                    else None
                ),
                "partition": "action_windows:candidate",
                "file": str(source_path),
                "bytes": input_bytes,
                "records": input_rows,
                "sha256": input_sha,
            },
            "files": file_meta,
            "counts": {
                "input_rows": input_rows,
                "selected_candidate_sft": selected_count,
                "selected_by_split": dict(sorted(split_counts.items())),
                "review_only_rows": input_rows - selected_count,
                "decisions": decisions_count,
                "lineage": lineage_count,
                "unique_selected_parents": len(selected_parent_splits),
                "parent_split_overlap": parent_overlaps,
                "exact_duplicate_prompt_completion_rows": duplicate_count,
                "conflicting_prompt_groups": len(conflicting_prompts),
                "rows_in_conflicting_prompt_groups": conflicting_rows,
                "source_duplicate_window_ids": duplicate_window_ids,
                "max_selected_rows_per_parent": max(selected_parent_counts.values(), default=0),
                "median_selected_rows_per_parent": (
                    sorted(selected_parent_counts.values())[len(selected_parent_counts) // 2]
                    if selected_parent_counts
                    else 0
                ),
                "provider_counts_selected": dict(sorted(provider_counts.items())),
                "quality_tier_counts_selected": dict(sorted(quality_counts.items())),
                "privacy_state_counts_selected": dict(sorted(privacy_counts.items())),
                "first_pass_reason_counts": dict(sorted(first_pass_reasons.items())),
                "review_reason_counts": dict(sorted(rejection_counts.items())),
            },
            "policy": {
                "loss": "completion_only; prompt tokens masked by TRL",
                "normalization": "strict JSON-object tool arguments only; no schema inference",
                "duplicate_policy": "exact prompt/completion duplicates are review-only",
                "conflict_policy": "all rows for a prompt with differing completions are review-only",
                "split_policy": f"stable parent-hash split; validation_fraction={validation_fraction:.4f}",
                "quality": "candidate; source outcome remains unverified",
                "tool_schema": "not observed; no schema inferred",
                "privacy": "preserved from source; review_required is not upgraded",
                "reasoning": "use existing canonical structural marker gate; do not add rationale or inferred reasoning",
                "rl": "no reward or RL data emitted",
            },
            "training_authorized": False,
            "quality_release": False,
            "target_model_training": False,
            "rl_data": False,
            "source_mutation": False,
            "trl_validation": "pending",
        }
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        (stage_dir / "manifest.json").write_bytes(manifest_bytes)
        os.replace(stage_dir, output_dir)
        published = True
        return manifest
    finally:
        if not published and stage_dir.exists():
            shutil.rmtree(stage_dir)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_jsonl(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _parent_sha(row: dict[str, Any]) -> str | None:
    lineage = row.get("lineage")
    if isinstance(lineage, dict):
        value = lineage.get("parent_record_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _trainer_example(
    row: dict[str, Any],
    *,
    normalized_messages: list[dict[str, Any]],
    parent_sha: str,
) -> dict[str, Any]:
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    trainer_tags = {
        tag for tag in tags if isinstance(tag, str)
    } | {"trainer:qwen-tool-sft-candidate", "tool-arguments:json-object"}
    example: dict[str, Any] = {
        "schema_version": EXAMPLE_SCHEMA,
        "example_id": row["example_id"],
        "split": row["split"],
        "parent_record_sha256": parent_sha,
        "messages": normalized_messages,
        "tags": sorted(trainer_tags),
    }
    for field in (
        "provider",
        "agent",
        "model_tier",
        "quality_tier",
        "quality_reason",
        "session_quality",
        "privacy",
        "tool_contract",
        "tool_families",
    ):
        if field in row:
            example[field] = copy.deepcopy(row[field])
    return example


def build_qwen_tool_sft_candidate(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build split JSONL, lineage, and row decisions from an existing pilot."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha = _sha256_file(source_manifest_path)
    source_files = source_manifest.get("files")
    if not isinstance(source_files, dict):
        raise ValueError("source manifest has no file bindings")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    published = False
    try:
        file_meta: dict[str, dict[str, Any]] = {}
        input_meta: dict[str, dict[str, Any]] = {}
        decisions_count = 0
        input_count = 0
        selected_count = 0
        rejected_count = 0
        normalized_calls = 0
        all_calls = 0
        selected_calls = 0
        selected_normalized_calls = 0
        review_calls = 0
        review_normalized_calls = 0
        system_messages = 0
        selected_provider_counts: Counter[str] = Counter()
        selected_tier_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        privacy_counts: Counter[str] = Counter()
        rejection_counts: Counter[str] = Counter()
        selected_by_split: dict[str, int] = {}
        parent_splits: dict[str, set[str]] = defaultdict(set)
        seen_example_ids: set[str] = set()

        decisions_path = stage_dir / "decisions.jsonl"
        lineage_path = stage_dir / "lineage.jsonl"
        decisions_hash = hashlib.sha256()
        decisions_bytes = 0
        lineage_hash = hashlib.sha256()
        lineage_bytes = 0
        lineage_count = 0

        with decisions_path.open("wb") as decisions_out, lineage_path.open(
            "wb"
        ) as lineage_out:
            for split in ("train", "validation"):
                source_name = f"{split}.jsonl"
                source_path = source_dir / source_name
                binding = source_files.get(source_name)
                if not isinstance(binding, dict):
                    raise ValueError(f"source manifest missing {source_name}")

                input_digest = hashlib.sha256()
                input_bytes = 0
                split_input_count = 0
                split_selected_count = 0
                output_name = f"{split}.jsonl"
                output_path = stage_dir / output_name
                output_digest = hashlib.sha256()
                output_bytes = 0
                output_count = 0

                before = source_path.stat()
                with source_path.open("rb") as source, output_path.open("wb") as dest:
                    for line_no, raw in enumerate(source, start=1):
                        input_digest.update(raw)
                        input_bytes += len(raw)
                        if not raw.strip():
                            continue
                        split_input_count += 1
                        input_count += 1
                        row_sha = hashlib.sha256(raw).hexdigest()
                        reasons: set[str] = set()
                        try:
                            row = json.loads(raw)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            row = None
                            reasons.add("invalid_json")
                        if not isinstance(row, dict):
                            reasons.add("row_not_object")

                        result: NormalizationResult | None = None
                        parent_sha: str | None = None
                        example_id: str | None = None
                        if isinstance(row, dict):
                            example_id_value = row.get("example_id")
                            if isinstance(example_id_value, str) and example_id_value:
                                example_id = example_id_value
                            else:
                                reasons.add("example_id_missing")
                            if row.get("split") != split:
                                reasons.add("source_split_mismatch")
                            parent_sha = _parent_sha(row)
                            if parent_sha is None:
                                reasons.add("parent_identity_missing")
                            if example_id is not None and example_id in seen_example_ids:
                                reasons.add("duplicate_example_id")
                            if example_id is not None:
                                seen_example_ids.add(example_id)
                            messages = row.get("messages")
                            if isinstance(messages, list):
                                system_messages += sum(
                                    isinstance(message, dict)
                                    and message.get("role") == "system"
                                    for message in messages
                                )
                            result = normalize_tool_messages(messages)
                            reasons.update(issue.value for issue in result.issues)
                            all_calls += result.tool_call_count
                            normalized_calls += result.normalized_argument_count

                        if not reasons and isinstance(row, dict) and result is not None:
                            if result.messages is None or example_id is None or parent_sha is None:
                                raise RuntimeError("eligible row lost normalized fields")
                            trainer_row = _trainer_example(
                                row,
                                normalized_messages=result.messages,
                                parent_sha=parent_sha,
                            )
                            encoded = _canonical_jsonl(trainer_row)
                            dest.write(encoded)
                            output_digest.update(encoded)
                            output_bytes += len(encoded)
                            output_count += 1
                            selected_count += 1
                            split_selected_count += 1
                            selected_calls += result.tool_call_count
                            selected_normalized_calls += result.normalized_argument_count
                            parent_splits[parent_sha].add(split)
                            selected_provider_counts[str(row.get("provider") or "<missing>")] += 1
                            selected_tier_counts[str(row.get("model_tier") or "<missing>")] += 1
                            quality_counts[str(row.get("quality_tier") or "<missing>")] += 1
                            privacy = row.get("privacy")
                            privacy_state = (
                                privacy.get("state")
                                if isinstance(privacy, dict)
                                else None
                            )
                            privacy_counts[str(privacy_state or "<missing>")] += 1
                            lineage_row = {
                                "example_id": example_id,
                                "split": split,
                                "parent_record_sha256": parent_sha,
                                "source_artifact_manifest_sha256": source_manifest_sha,
                                "source_partition": source_name,
                                "source_line": line_no,
                                "source_row_sha256": row_sha,
                            }
                            encoded_lineage = _canonical_jsonl(lineage_row)
                            lineage_out.write(encoded_lineage)
                            lineage_hash.update(encoded_lineage)
                            lineage_bytes += len(encoded_lineage)
                            lineage_count += 1
                            decision = {
                                "example_id": example_id,
                                "split": split,
                                "decision": "selected_candidate_sft",
                                "reasons": [],
                                "source_row_sha256": row_sha,
                                "normalized_tool_calls": result.normalized_argument_count,
                            }
                        else:
                            rejected_count += 1
                            if result is not None:
                                review_calls += result.tool_call_count
                                review_normalized_calls += result.normalized_argument_count
                            rejection_counts.update(reasons or {"normalization_failed"})
                            decision = {
                                "example_id": example_id,
                                "split": split,
                                "decision": "review_only",
                                "reasons": sorted(reasons or {"normalization_failed"}),
                                "source_row_sha256": row_sha,
                                "tool_calls": result.tool_call_count if result else 0,
                            }
                        encoded_decision = _canonical_jsonl(decision)
                        decisions_out.write(encoded_decision)
                        decisions_hash.update(encoded_decision)
                        decisions_bytes += len(encoded_decision)
                        decisions_count += 1

                after = source_path.stat()
                actual_sha = input_digest.hexdigest()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise RuntimeError(f"source changed while reading: {source_path}")
                if actual_sha != binding.get("sha256"):
                    raise ValueError(f"source digest mismatch: {source_name}")
                if input_bytes != binding.get("bytes"):
                    raise ValueError(f"source size mismatch: {source_name}")
                if split_input_count != binding.get("records"):
                    raise ValueError(f"source record-count mismatch: {source_name}")

                input_meta[source_name] = {
                    "records": split_input_count,
                    "bytes": input_bytes,
                    "sha256": actual_sha,
                }
                file_meta[output_name] = {
                    "records": output_count,
                    "bytes": output_bytes,
                    "sha256": output_digest.hexdigest(),
                }
                selected_by_split[split] = split_selected_count

        if decisions_count != input_count:
            raise RuntimeError("not every source row received a decision")
        if lineage_count != selected_count:
            raise RuntimeError("lineage/output row counts differ")
        overlap = [parent for parent, splits in parent_splits.items() if len(splits) > 1]
        if overlap:
            raise ValueError("selected parents overlap train and validation splits")

        file_meta["decisions.jsonl"] = {
            "records": decisions_count,
            "bytes": decisions_bytes,
            "sha256": decisions_hash.hexdigest(),
        }
        file_meta["lineage.jsonl"] = {
            "records": lineage_count,
            "bytes": lineage_bytes,
            "sha256": lineage_hash.hexdigest(),
        }
        manifest = {
            "schema_version": ADAPTER_SCHEMA,
            "status": "candidate_built_pending_trl_validation",
            "target": "Qwen3.5 chat template via Transformers/TRL",
            "format": "messages_jsonl",
            "source": {
                "manifest_path": str(source_manifest_path),
                "manifest_schema": source_manifest.get("schema_version"),
                "manifest_sha256": source_manifest_sha,
                "source_partition": source_manifest.get("source", {}).get("partition"),
                "files": input_meta,
            },
            "files": file_meta,
            "counts": {
                "input_rows": input_count,
                "selected_candidate_sft": selected_count,
                "selected_by_split": selected_by_split,
                "review_only_rows": rejected_count,
                "decisions": decisions_count,
                "lineage": lineage_count,
                "unique_selected_parents": len(parent_splits),
                "parent_split_overlap": 0,
                "tool_calls_input": all_calls,
                "tool_arguments_valid_in_input": normalized_calls,
                "tool_calls_selected": selected_calls,
                "tool_arguments_selected": selected_normalized_calls,
                "tool_calls_review_only": review_calls,
                "tool_arguments_valid_in_review_rows": review_normalized_calls,
                "provider_counts_selected": dict(sorted(selected_provider_counts.items())),
                "model_tier_counts_selected": dict(sorted(selected_tier_counts.items())),
                "quality_tier_counts_selected": dict(sorted(quality_counts.items())),
                "privacy_state_counts_selected": dict(sorted(privacy_counts.items())),
                "source_system_messages": system_messages,
                "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            },
            "policy": {
                "normalization": "strict JSON object strings to mappings; no argument guessing",
                "invalid_row_policy": "preserve source, emit review-only decision, do not train row",
                "quality_tier": "candidate",
                "outcome_status": "unverified; SFT only",
                "tool_schema_status": "not_observed; runtime harness supplies schemas",
                "reward_status": "not_present; no RL data emitted",
                "privacy_status": "preserved from source; not upgraded to approved",
                "system_prompt_policy": "source had no system-role messages in this lane",
            },
            "qwen_argument_contract_valid": True,
            "trainer_compatible": "pending_trl_validation",
            "quality_release": False,
            "raw_session_access": False,
            "reward_invention": False,
            "source_mutation": False,
            "trl_validation": "pending",
        }
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        (stage_dir / "manifest.json").write_bytes(manifest_bytes)
        os.replace(stage_dir, output_dir)
        published = True
        return manifest
    finally:
        if not published and stage_dir.exists():
            shutil.rmtree(stage_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Qwen/TRL SFT candidates from archived tool data."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path)
    source.add_argument("--action-window-release-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.source_dir is not None:
        manifest = build_qwen_tool_sft_candidate(args.source_dir, args.output_dir)
    else:
        manifest = build_qwen_action_window_sft_candidate(
            args.action_window_release_dir,
            args.output_dir,
            validation_fraction=args.validation_fraction,
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
