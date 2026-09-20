#!/usr/bin/env python3
"""Compose a bounded agentic SFT release from qualified source partitions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from runtime.sft.dataset import (
    EXAMPLE_SCHEMA,
    RELEASE_SCHEMA,
    iter_release_rows,
    load_manifest,
    sha256_file,
    validate_example,
)
from runtime.sft.filter_release import canonical_bytes, iter_bound_jsonl, write_jsonl


MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
COMPOSITE_LINEAGE_SCHEMA = "ai-data-extraction/agent-sft-composite-lineage/v1"
DECISION_SCHEMA = "ai-data-extraction/agent-sft-composition-decision/v1"
TOOLCALL = re.compile(r"^\s*<TOOLCALL>(.*?)</TOOLCALL>\s*$", re.S)
FORBIDDEN_MODEL_TEXT = re.compile(
    r"<instructions>|\bTHOUGHT section\b|exactly ONE bash command|patch\.txt|"
    r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|\bOpenAI\b|\bAnthropic\b|"
    r"\bClaude\b|\bGemini\b|\bCodex\b",
    re.I,
)
VISIBLE_REASONING = re.compile(r"<(?:think|analysis)>|```(?:reasoning|analysis)", re.I)


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_split(parent_id: str, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(parent_id.encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def assert_model_text_clean(
    value: Any, blocked_text_patterns: tuple[re.Pattern[str], ...] = ()
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {
                "reasoning",
                "reasoning_content",
                "thought",
                "thoughts",
                "chain_of_thought",
            }:
                raise ValueError("reasoning_field")
            assert_model_text_clean(child, blocked_text_patterns)
    elif isinstance(value, list):
        for child in value:
            assert_model_text_clean(child, blocked_text_patterns)
    elif isinstance(value, str):
        if VISIBLE_REASONING.search(value):
            raise ValueError("visible_reasoning")
        if FORBIDDEN_MODEL_TEXT.search(value):
            raise ValueError("forbidden_model_text")
        if any(pattern.search(value) for pattern in blocked_text_patterns):
            raise ValueError("blocked_model_text")


def release_rows(
    path: Path,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    path = path.resolve()
    manifest = load_manifest(path)
    lineage = {
        row["example_id"]: row
        for row in iter_bound_jsonl(path, manifest, "lineage.jsonl")
    }
    result: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for split, _, row in iter_release_rows(path, manifest):
        validate_example(row, expected_split=split)
        example_id = row["example_id"]
        if example_id in seen:
            raise ValueError(f"duplicate source example: {example_id}")
        seen.add(example_id)
        source_lineage = lineage.get(example_id)
        if source_lineage is None:
            raise ValueError(f"missing source lineage: {example_id}")
        result.append((split, row, source_lineage))
    if set(lineage) != seen:
        raise ValueError("source release has detached lineage")
    return manifest, result


def parse_json_object(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}_invalid_json") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}_not_object")
    return value


def normalize_type(raw_type: Any) -> dict[str, Any]:
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ValueError("tool_property_type_missing")
    value = raw_type.strip()
    value = re.sub(r",\s*optional.*$", "", value, flags=re.I).strip()
    scalar = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
    }
    if value.lower() in scalar:
        return {"type": scalar[value.lower()]}
    if value.lower() in {"dict", "object"}:
        return {"type": "object"}
    if value.lower() in {"list", "set"}:
        return {"type": "array", "items": {}}
    list_match = re.fullmatch(r"List\[(.*)]", value, re.I)
    if list_match:
        return {"type": "array", "items": normalize_type(list_match.group(1))}
    tuple_match = re.fullmatch(r"Tuple\[(.*)]", value, re.I)
    if tuple_match:
        parts = [part.strip() for part in tuple_match.group(1).split(",")]
        normalized = [normalize_type(part) for part in parts]
        if all(item == normalized[0] for item in normalized):
            return {"type": "array", "items": normalized[0]}
        return {"type": "array", "items": {}}
    union_match = re.fullmatch(r"Union\[(.*)]", value, re.I)
    if union_match:
        options = [normalize_type(part.strip()) for part in union_match.group(1).split(",")]
        types = {option.get("type") for option in options}
        if types <= {"integer", "number"}:
            return {"type": "number"}
        return {"anyOf": options}
    raise ValueError("tool_property_type_unsupported")


def normalize_property(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("tool_property_not_object")
    normalized = normalize_type(raw.get("type"))
    description = raw.get("description")
    if isinstance(description, str) and description:
        normalized["description"] = description
    enum = raw.get("enum")
    if isinstance(enum, list) and enum:
        normalized["enum"] = enum
    return normalized


def normalize_when2call_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("tools_missing")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_tools):
        tool = parse_json_object(raw, f"tool_{index}")
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("tool_name_invalid_or_duplicate")
        if not isinstance(parameters, dict):
            raise ValueError("tool_parameters_missing")
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("tool_properties_missing")
        required = tool.get("required", parameters.get("required", []))
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise ValueError("tool_required_invalid")
        if not set(required).issubset(properties):
            raise ValueError("tool_required_unknown_property")
        function = {
            "name": name,
            "description": str(tool.get("description") or "Call the selected tool."),
            "parameters": {
                "type": "object",
                "properties": {
                    key: normalize_property(value)
                    for key, value in sorted(properties.items())
                },
                "required": sorted(set(required)),
                "additionalProperties": False,
            },
        }
        normalized.append({"type": "function", "function": function})
        names.add(name)
    return normalized


def value_matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    if "anyOf" in schema:
        return any(value_matches_schema(value, item) for item in schema["anyOf"])
    expected = schema.get("type")
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list) and all(
            value_matches_schema(item, schema.get("items", {})) for item in value
        )
    if expected == "object":
        return isinstance(value, dict)
    return schema == {}


def normalize_when2call_target(
    content: Any, tools: list[dict[str, Any]], source_row: int
) -> tuple[str, dict[str, Any]]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chosen_response_missing")
    match = TOOLCALL.fullmatch(content)
    if match:
        try:
            calls = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError("chosen_toolcall_invalid_json") from exc
        if not isinstance(calls, list) or not calls:
            raise ValueError("chosen_toolcall_not_list")
        schemas = {tool["function"]["name"]: tool["function"] for tool in tools}
        normalized_calls = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                raise ValueError("chosen_toolcall_invalid")
            name = call.get("name")
            schema = schemas.get(name)
            if schema is None:
                raise ValueError("chosen_toolcall_unknown_tool")
            arguments = call["arguments"]
            parameters = schema["parameters"]
            if not set(parameters["required"]).issubset(arguments):
                raise ValueError("chosen_toolcall_missing_required")
            if not set(arguments).issubset(parameters["properties"]):
                raise ValueError("chosen_toolcall_unknown_argument")
            if any(
                not value_matches_schema(value, parameters["properties"][key])
                for key, value in arguments.items()
            ):
                raise ValueError("chosen_toolcall_argument_type")
            normalized_calls.append(
                {
                    "id": f"call_w2c_{source_row:05d}_{index:02d}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return "tool_call", {
            "role": "assistant",
            "content": "",
            "tool_calls": normalized_calls,
        }
    if "<TOOLCALL>" in content or len(content) > 600:
        raise ValueError("chosen_text_invalid_or_verbose")
    category = "clarify" if content.rstrip().endswith("?") else "no_tool"
    return category, {"role": "assistant", "content": content.strip()}


def project_release_row(
    *,
    row: dict[str, Any],
    lineage: dict[str, Any],
    source_name: str,
    source_manifest_sha256: str,
    blocked_text_patterns: tuple[re.Pattern[str], ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    projected = copy.deepcopy(row)
    source_example_id = projected["example_id"]
    projected["example_id"] = "sha256:" + digest_value(
        {"source_manifest": source_manifest_sha256, "example_id": source_example_id}
    )
    assert_model_text_clean(projected, blocked_text_patterns)
    validate_example(projected, expected_split=projected["split"])
    parent = lineage.get("parent_id")
    if not isinstance(parent, str) or not parent:
        raise ValueError("source parent identity missing")
    projected_lineage = {
        "schema_version": COMPOSITE_LINEAGE_SCHEMA,
        "example_id": projected["example_id"],
        "parent_id": f"{source_name}:{parent}",
        "source_partition": source_name,
        "source_manifest_sha256": source_manifest_sha256,
        "source_example_id": source_example_id,
        "source_lane": row["lane"],
        "training_role": "positive_sft",
        "outcome_verification": lineage.get("outcome_verification", "source_bound"),
    }
    return projected, projected_lineage


def build_when2call_replay(
    *,
    source_path: Path,
    source_revision: str,
    cap_per_category: int,
    validation_percent: int,
    blocked_text_patterns: tuple[re.Pattern[str], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    seen_projection: set[str] = set()
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row_sha256 = digest_value(raw)
            decision: dict[str, Any] = {
                "schema_version": DECISION_SCHEMA,
                "source_partition": "when2call_train_pref",
                "source_row": line_number,
                "source_row_sha256": row_sha256,
            }
            try:
                tools = normalize_when2call_tools(raw.get("tools"))
                messages = raw.get("messages")
                if (
                    not isinstance(messages, list)
                    or len(messages) != 1
                    or messages[0].get("role") != "user"
                    or not isinstance(messages[0].get("content"), str)
                ):
                    raise ValueError("source_prompt_invalid")
                chosen = raw.get("chosen_response")
                if not isinstance(chosen, dict) or chosen.get("role") != "assistant":
                    raise ValueError("chosen_response_invalid")
                category, target = normalize_when2call_target(
                    chosen.get("content"), tools, line_number
                )
                parent_id = f"when2call:{row_sha256}"
                split = stable_split(parent_id, validation_percent)
                example = {
                    "schema_version": EXAMPLE_SCHEMA,
                    "example_id": "sha256:"
                    + digest_value(
                        {
                            "dataset_revision": source_revision,
                            "source_row_sha256": row_sha256,
                            "role": "tool_policy_replay",
                        }
                    ),
                    "split": split,
                    "lane": "tool_policy_replay",
                    "messages": [
                        {"role": "user", "content": messages[0]["content"]},
                        target,
                    ],
                    "tools": tools,
                }
                assert_model_text_clean(example, blocked_text_patterns)
                validate_example(example, expected_split=split)
                projection_hash = digest_value(
                    {"messages": example["messages"], "tools": example["tools"]}
                )
                if projection_hash in seen_projection:
                    raise ValueError("duplicate_projected_example")
                seen_projection.add(projection_hash)
                candidate = {
                    "rank": digest_value({"category": category, "row": row_sha256}),
                    "category": category,
                    "example": example,
                    "lineage": {
                        "schema_version": COMPOSITE_LINEAGE_SCHEMA,
                        "example_id": example["example_id"],
                        "parent_id": parent_id,
                        "source_partition": "when2call_train_pref",
                        "source_revision": source_revision,
                        "source_row": line_number,
                        "source_row_sha256": row_sha256,
                        "source_lane": category,
                        "training_role": "capability_replay_sft",
                        "outcome_verification": "automated_preference_label_not_executed",
                    },
                    "decision": decision,
                }
                candidates[category].append(candidate)
                decision.update({"category": category, "decision": "eligible"})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                decision.update({"decision": "excluded", "reason": str(exc)})
            decisions.append(decision)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, str]] = set()
    for category in ("tool_call", "clarify", "no_tool"):
        ranked = sorted(candidates[category], key=lambda item: item["rank"])
        if len(ranked) < cap_per_category:
            raise ValueError(f"insufficient When2Call {category} candidates")
        for item in ranked[:cap_per_category]:
            selected.append(item)
            selected_keys.add((item["decision"]["source_row"], category))
    for decision in decisions:
        key = (decision["source_row"], decision.get("category", ""))
        if decision.get("decision") == "eligible":
            decision["decision"] = "selected" if key in selected_keys else "quota_excluded"
    selected.sort(key=lambda item: item["example"]["example_id"])
    return (
        [item["example"] for item in selected],
        [item["lineage"] for item in selected],
        decisions,
        {
            "source_rows": len(decisions),
            "eligible_by_category": {
                category: len(candidates[category])
                for category in ("tool_call", "clarify", "no_tool")
            },
            "selected_by_category": {
                category: cap_per_category
                for category in ("tool_call", "clarify", "no_tool")
            },
            "excluded_reasons": dict(
                sorted(
                    Counter(
                        item.get("reason", "unknown")
                        for item in decisions
                        if item["decision"] == "excluded"
                    ).items()
                )
            ),
        },
    )


def build_curriculum(
    *,
    open_release: Path,
    internal_release: Path,
    when2call_source: Path,
    when2call_revision: str,
    output_dir: Path,
    replay_cap_per_category: int = 96,
    validation_percent: int = 20,
    blocked_text_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    open_manifest, open_rows = release_rows(open_release)
    internal_manifest, internal_rows = release_rows(internal_release)
    open_manifest_sha = sha256_file(open_release / "manifest.json")
    internal_manifest_sha = sha256_file(internal_release / "manifest.json")
    compiled_blocked_text = tuple(
        re.compile(pattern, re.I) for pattern in blocked_text_patterns if pattern
    )
    if len(compiled_blocked_text) != len(blocked_text_patterns):
        raise ValueError("blocked text patterns must be non-empty")

    examples: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    partition_counts = Counter()
    for _, row, source_lineage in open_rows:
        projected, projected_lineage = project_release_row(
            row=row,
            lineage=source_lineage,
            source_name="open_swe_verified",
            source_manifest_sha256=open_manifest_sha,
            blocked_text_patterns=compiled_blocked_text,
        )
        examples.append(projected)
        lineage.append(projected_lineage)
        partition_counts["open_swe_verified"] += 1

    allowed_internal_lanes = {"skill_policy", "reviewed_final_answer"}
    for _, row, source_lineage in internal_rows:
        decision = {
            "schema_version": DECISION_SCHEMA,
            "source_partition": "internal_pilot_v3",
            "source_example_id": row["example_id"],
            "source_lane": row["lane"],
        }
        if row["lane"] not in allowed_internal_lanes:
            decision.update({"decision": "excluded", "reason": "unverified_action_lane"})
            decisions.append(decision)
            continue
        try:
            projected, projected_lineage = project_release_row(
                row=row,
                lineage=source_lineage,
                source_name="internal_policy",
                source_manifest_sha256=internal_manifest_sha,
                blocked_text_patterns=compiled_blocked_text,
            )
        except ValueError as exc:
            decision.update({"decision": "excluded", "reason": str(exc)})
            decisions.append(decision)
            continue
        examples.append(projected)
        lineage.append(projected_lineage)
        partition_counts[f"internal_{row['lane']}"] += 1
        decision["decision"] = "selected"
        decisions.append(decision)

    replay, replay_lineage, replay_decisions, replay_report = build_when2call_replay(
        source_path=when2call_source.resolve(),
        source_revision=when2call_revision,
        cap_per_category=replay_cap_per_category,
        validation_percent=validation_percent,
        blocked_text_patterns=compiled_blocked_text,
    )
    examples.extend(replay)
    lineage.extend(replay_lineage)
    decisions.extend(replay_decisions)
    partition_counts["when2call_replay"] = len(replay)

    ids = [row["example_id"] for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate composite example identity")
    lineage_ids = [row["example_id"] for row in lineage]
    if len(lineage_ids) != len(set(lineage_ids)) or set(lineage_ids) != set(ids):
        raise ValueError("composite lineage identity mismatch")
    split_by_id = {row["example_id"]: row["split"] for row in examples}
    parent_splits: defaultdict[str, set[str]] = defaultdict(set)
    for item in lineage:
        parent_splits[item["parent_id"]].add(split_by_id[item["example_id"]])
    if any(len(splits) != 1 for splits in parent_splits.values()):
        raise ValueError("composite parent split overlap")

    examples.sort(key=lambda row: (row["split"], row["example_id"]))
    lineage.sort(key=lambda row: row["example_id"])
    decisions.sort(
        key=lambda row: (
            row["source_partition"],
            int(row.get("source_row", 0)),
            row.get("source_example_id", ""),
        )
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        files = {
            "train.jsonl": write_jsonl(
                staging / "train.jsonl",
                (row for row in examples if row["split"] == "train"),
            ),
            "validation.jsonl": write_jsonl(
                staging / "validation.jsonl",
                (row for row in examples if row["split"] == "validation"),
            ),
            "lineage.jsonl": write_jsonl(staging / "lineage.jsonl", lineage),
            "decisions.jsonl": write_jsonl(staging / "decisions.jsonl", decisions),
        }
        lane_counts = Counter(f"{row['split']}:{row['lane']}" for row in examples)
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "status": "ready_for_exact_tokenizer_preflight_not_training_authorized",
            "purpose": "balanced_long_horizon_agentic_sft",
            "training_authorized": False,
            "model": {
                "repo_id": "Qwen/Qwen3.5-9B",
                "revision": MODEL_REVISION,
                "local_path": "/data-120/models/Qwen3.5-9B",
                "chat_template_kwargs": {"enable_thinking": False},
                "max_sequence_tokens": 8192,
                "loss": "final_assistant_turn_only",
            },
            "sources": {
                "open_swe_verified": {
                    "path": str(open_release.resolve()),
                    "manifest_sha256": open_manifest_sha,
                    "source_manifest_status": open_manifest.get("status"),
                    "role": "source_executed_positive_sft",
                },
                "internal_policy": {
                    "path": str(internal_release.resolve()),
                    "manifest_sha256": internal_manifest_sha,
                    "source_manifest_status": internal_manifest.get("status"),
                    "role": "source_bound_skill_and_stop_policy",
                },
                "when2call": {
                    "dataset_id": "nvidia/When2Call",
                    "revision": when2call_revision,
                    "path": str(when2call_source.resolve()),
                    "sha256": sha256_file(when2call_source),
                    "license": "cc-by-4.0",
                    "role": "automated_capability_replay_not_expert_trajectory",
                    "test_split_used": False,
                },
            },
            "selection": {
                "one_copy_per_example": True,
                "open_swe_source_protocol_removed": True,
                "internal_lanes": sorted(allowed_internal_lanes),
                "blocked_model_text": "reject",
                "blocked_text_pattern_sha256": sorted(
                    hashlib.sha256(pattern.encode()).hexdigest()
                    for pattern in blocked_text_patterns
                ),
                "unverified_internal_action_windows": "excluded",
                "when2call_replay_cap_per_category": replay_cap_per_category,
                "when2call": replay_report,
                "parent_disjoint": True,
                "validation_percent_for_replay": validation_percent,
            },
            "counts": {
                "total": len(examples),
                "train": sum(row["split"] == "train" for row in examples),
                "validation": sum(row["split"] == "validation" for row in examples),
                "unique_parents": len(parent_splits),
                "partitions": dict(sorted(partition_counts.items())),
                "lanes": dict(sorted(lane_counts.items())),
            },
            "quality": {
                "hidden_reasoning": "removed_or_rejected",
                "open_swe_outcome": "source_execution_label_not_locally_replayed",
                "internal_policy": "source_bound_authored_examples",
                "when2call": "automated_preference_labels_not_executed",
                "rewards": "not_present_not_rl_eligible",
                "tokenizer_preflight": "pending",
            },
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-release", type=Path, required=True)
    parser.add_argument("--internal-release", type=Path, required=True)
    parser.add_argument("--when2call-source", type=Path, required=True)
    parser.add_argument("--when2call-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-cap-per-category", type=int, default=96)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--blocked-text-pattern", action="append", default=[])
    args = parser.parse_args()
    if args.replay_cap_per_category <= 0:
        raise SystemExit("replay cap must be positive")
    result = build_curriculum(
        open_release=args.open_release,
        internal_release=args.internal_release,
        when2call_source=args.when2call_source,
        when2call_revision=args.when2call_revision,
        output_dir=args.output_dir,
        replay_cap_per_category=args.replay_cap_per_category,
        validation_percent=args.validation_percent,
        blocked_text_patterns=tuple(args.blocked_text_pattern),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
