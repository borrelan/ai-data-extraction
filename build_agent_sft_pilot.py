#!/usr/bin/env python3
"""Build the bounded Qwen3.5-9B agent SFT pilot from existing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from quality_rules import scan_record
from runtime.sft.skill_curriculum import build_skill_curriculum


SCHEMA = "ai-data-extraction/agent-sft-pilot/v1"
EXAMPLE_SCHEMA = "ai-data-extraction/agent-sft-example/v1"
DEFAULT_REVIEWED = Path(".tmp/final_answer_sft_reviewed_qwen3_8_20260919_v1")
DEFAULT_ACTIONS = Path(".tmp/trl_action_window_sft_candidate_20260919_v2")
DEFAULT_SKILL_ROOT = Path.home() / ".codex" / "skills" / "personal"
DEFAULT_TOOL_SCHEMAS = Path("runtime/sft/tool_schemas.json")
DEFAULT_EVAL_CASES = Path(".tmp/qwen3_8_behavior_gate_20260919_v1/cases.jsonl")
DEFAULT_BENCHMARK_REGISTRY = Path(
    "/home/borrelan/Projects/Personal/ai-agent-benchmark/registry/tasks.jsonl"
)
DEFAULT_MODEL_DIR = Path("/data-120/models/Qwen3.5-9B")
FORBIDDEN_VISIBLE = {
    "legacy_brand": re.compile(r"spatial\s*chat", re.IGNORECASE),
    "provider_brand": re.compile(r"\b(?:openai|claude|gemini|codex)\b", re.IGNORECASE),
    "suggestion_mode": re.compile(r"\bSUGGESTION MODE\b", re.IGNORECASE),
}
REASONING_KEYS = frozenset(
    {"reasoning", "reasoning_content", "thought", "thoughts", "chain_of_thought"}
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptor(path: Path, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield line_number, raw, value


def _walk_has_reasoning(value: Any) -> bool:
    if isinstance(value, dict):
        if any(str(key).lower() in REASONING_KEYS for key in value):
            return True
        return any(_walk_has_reasoning(child) for child in value.values())
    if isinstance(value, list):
        return any(_walk_has_reasoning(child) for child in value)
    return False


def _visible_rejection(view: dict[str, Any], *, max_characters: int) -> str | None:
    raw = canonical_bytes(view)
    if len(raw) > max_characters:
        return "visible_character_limit"
    text = raw.decode("utf-8")
    for reason, pattern in FORBIDDEN_VISIBLE.items():
        if pattern.search(text):
            return reason
    if _walk_has_reasoning(view):
        return "reasoning_field"
    findings = scan_record(view)
    if findings.has_hard_privacy_issue:
        return "privacy_firewall"
    if findings.has_marker:
        return "reasoning_marker"
    return None


def _validate_messages(messages: Any) -> str | None:
    if not isinstance(messages, list) or len(messages) < 2:
        return "messages_invalid"
    for message in messages:
        if not isinstance(message, dict):
            return "message_not_object"
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            return "message_role_invalid"
        if not isinstance(message.get("content"), str):
            return "message_content_invalid"
        calls = message.get("tool_calls")
        if calls is None:
            continue
        if message.get("role") != "assistant" or not isinstance(calls, list) or not calls:
            return "tool_calls_invalid"
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                return "tool_function_invalid"
            if not isinstance(function.get("name"), str) or not function["name"]:
                return "tool_name_invalid"
            if not isinstance(function.get("arguments"), dict):
                return "tool_arguments_invalid"
    if messages[-1].get("role") != "assistant":
        return "target_not_assistant"
    return None


def _tool_names(messages: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            function["name"]
            for message in messages
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
            and isinstance((function := call.get("function")), dict)
            and isinstance(function.get("name"), str)
        }
    )


def _target_tool(messages: list[dict[str, Any]]) -> str:
    calls = messages[-1].get("tool_calls") or []
    names = [call["function"]["name"] for call in calls]
    return "+".join(sorted(names))


def _action_refs(
    action_dir: Path,
    *,
    max_characters: int,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, dict[str, Any]]]:
    refs: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    descriptors: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        path = action_dir / f"{split}.jsonl"
        descriptors[path.name] = file_descriptor(path)
        for line_number, raw, row in iter_jsonl(path):
            decisions["input"] += 1
            if row.get("model_tier") != "tier1_frontier":
                decisions["model_tier_not_frontier"] += 1
                continue
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list) or len(completion) != 1:
                decisions["prompt_completion_invalid"] += 1
                continue
            messages = [*prompt, *completion]
            reason = _validate_messages(messages)
            if reason is None and not completion[0].get("tool_calls"):
                reason = "target_has_no_tool_call"
            view = {"messages": messages}
            reason = reason or _visible_rejection(view, max_characters=max_characters)
            example_id = row.get("example_id")
            parent = row.get("parent_record_sha256")
            if reason is None and (
                not isinstance(example_id, str)
                or not example_id
                or not isinstance(parent, str)
                or not parent
            ):
                reason = "lineage_identity_missing"
            if reason is not None:
                decisions[reason] += 1
                continue
            refs.append(
                {
                    "example_id": example_id,
                    "parent_id": parent,
                    "split": split,
                    "source_file": path,
                    "source_line": line_number,
                    "source_row_sha256": sha256_bytes(raw),
                    "target_tool": _target_tool(messages),
                }
            )
            decisions["eligible"] += 1
    return refs, decisions, descriptors


def _diverse_parent_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tool[row["target_tool"]].append(row)
    for values in by_tool.values():
        values.sort(key=lambda item: item["example_id"])
    ordered: list[dict[str, Any]] = []
    while by_tool:
        for name in sorted(tuple(by_tool)):
            ordered.append(by_tool[name].pop(0))
            if not by_tool[name]:
                del by_tool[name]
    return ordered


def select_action_refs(
    refs: list[dict[str, Any]], *, total_caps: dict[str, int], parent_caps: dict[str, int]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for split in ("train", "validation"):
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in refs:
            if row["split"] == split:
                by_parent[row["parent_id"]].append(row)
        queues = {
            parent: _diverse_parent_order(rows)[: parent_caps[split]]
            for parent, rows in sorted(by_parent.items())
        }
        decisions[f"{split}_parent_cap_excluded"] = sum(
            max(0, len(by_parent[parent]) - len(queues[parent])) for parent in queues
        )
        while queues and sum(row["split"] == split for row in selected) < total_caps[split]:
            for parent in sorted(tuple(queues)):
                if sum(row["split"] == split for row in selected) >= total_caps[split]:
                    break
                selected.append(queues[parent].pop(0))
                if not queues[parent]:
                    del queues[parent]
        decisions[f"{split}_total_cap_excluded"] = sum(len(rows) for rows in queues.values())
        decisions[f"{split}_selected"] = sum(row["split"] == split for row in selected)
    return selected, decisions


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"unsupported tool argument type: {type(value).__name__}")


def _compatibility_schemas(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "present": Counter(), "types": defaultdict(set)}
    )
    for item in rows:
        for message in item["messages"]:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                name = function["name"]
                arguments = function["arguments"]
                stats[name]["calls"] += 1
                for key, value in arguments.items():
                    stats[name]["present"][key] += 1
                    stats[name]["types"][key].add(_json_type(value))
    schemas: dict[str, dict[str, Any]] = {}
    type_order = {"null": 0, "boolean": 1, "integer": 2, "number": 3, "string": 4, "array": 5, "object": 6}
    for name, observed in sorted(stats.items()):
        properties: dict[str, Any] = {}
        for key, types in sorted(observed["types"].items()):
            ordered = sorted(types, key=type_order.__getitem__)
            properties[key] = {"type": ordered[0] if len(ordered) == 1 else ordered}
        required = sorted(
            key for key, count in observed["present"].items() if count == observed["calls"]
        )
        schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": "Compatibility schema reconstructed from validated calls in the bound source artifact; the runtime schema remains authoritative.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    return schemas


def _load_selected_actions(
    refs: list[dict[str, Any]], *, max_characters: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = {row["example_id"]: row for row in refs}
    loaded: list[dict[str, Any]] = []
    for path in sorted({row["source_file"] for row in refs}):
        for line_number, _raw, source in iter_jsonl(path):
            ref = wanted.get(source.get("example_id"))
            if ref is None:
                continue
            if line_number != ref["source_line"]:
                raise ValueError(f"source line drift for {ref['example_id']}")
            messages = [*(source["prompt"]), *(source["completion"])]
            loaded.append({"ref": ref, "source": source, "messages": messages})
    if len(loaded) != len(refs):
        raise ValueError("selected action rows could not be reloaded exactly")
    schemas = _compatibility_schemas(loaded)
    examples: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for item in sorted(loaded, key=lambda value: value["ref"]["example_id"]):
        ref = item["ref"]
        names = _tool_names(item["messages"])
        tools = [schemas[name] for name in names]
        view = {"messages": item["messages"], "tools": tools}
        # The source-message character cap is an inexpensive streaming ingress
        # bound. Derived schemas are admitted by the exact tokenizer gate below.
        reason = _visible_rejection(view, max_characters=1_000_000_000)
        if reason is not None:
            raise ValueError(f"selected action failed final firewall: {reason}")
        examples.append(
            {
                "schema_version": EXAMPLE_SCHEMA,
                "example_id": ref["example_id"],
                "split": ref["split"],
                "lane": "frontier_action_window",
                **view,
            }
        )
        source = item["source"]
        lineage.append(
            {
                "example_id": ref["example_id"],
                "parent_id": ref["parent_id"],
                "lane": "frontier_action_window",
                "source_file": str(ref["source_file"].resolve()),
                "source_line": ref["source_line"],
                "source_row_sha256": ref["source_row_sha256"],
                "provider": source.get("provider"),
                "model_tier": source.get("model_tier"),
                "outcome_verification": "not_independently_replayed",
                "tool_schema_basis": "compatibility_schema_from_observed_valid_calls",
            }
        )
    return examples, lineage


def _reviewed_examples(
    reviewed_dir: Path, *, max_characters: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], dict[str, Any]]:
    manifest_path = reviewed_dir / "manifest.json"
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != "ai-data-extraction/reviewed-final-answer-sft-pilot/v1":
        raise ValueError("unexpected reviewed final-answer schema")
    lineage_map: dict[str, dict[str, Any]] = {}
    for _, _, row in iter_jsonl(reviewed_dir / "lineage.jsonl"):
        lineage_map[row["source_example_id"]] = row
    examples: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    files: dict[str, Any] = {"manifest.json": file_descriptor(manifest_path)}
    for split in ("train", "validation"):
        path = reviewed_dir / f"{split}.jsonl"
        files[path.name] = file_descriptor(path)
        for line_number, raw, row in iter_jsonl(path):
            decisions["input"] += 1
            messages = row.get("messages")
            reason = _validate_messages(messages)
            reason = reason or _visible_rejection(
                {"messages": messages}, max_characters=max_characters
            )
            source_id = sha256_bytes(raw)
            matching = [
                value for key, value in lineage_map.items() if value.get("trainer_line") == line_number and value.get("split") == split
            ]
            if reason is None and len(matching) != 1:
                reason = "reviewed_lineage_missing_or_ambiguous"
            if reason is not None:
                decisions[reason] += 1
                continue
            source_lineage = matching[0]
            original_id = source_lineage["source_example_id"]
            example_id = f"sha256:{sha256_bytes(canonical_bytes({'lane': 'reviewed_final_answer', 'source': original_id}))}"
            examples.append(
                {
                    "schema_version": EXAMPLE_SCHEMA,
                    "example_id": example_id,
                    "split": split,
                    "lane": "reviewed_final_answer",
                    "messages": messages,
                }
            )
            lineage.append(
                {
                    "example_id": example_id,
                    "parent_id": source_lineage["parent_record_sha256"],
                    "lane": "reviewed_final_answer",
                    "source_file": str(path.resolve()),
                    "source_line": line_number,
                    "source_row_sha256": source_id,
                    "source_example_id": original_id,
                    "quality_tier": source_lineage.get("quality_tier"),
                    "outcome_verification": source_lineage.get("outcome_verification"),
                }
            )
            decisions["selected"] += 1
    return examples, lineage, decisions, files


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            raw = canonical_bytes(row) + b"\n"
            output.write(raw)
            digest.update(raw)
            count += 1
        output.flush()
        os.fsync(output.fileno())
    return {"records": count, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _prompt_hashes(examples: Iterable[dict[str, Any]]) -> set[str]:
    return {
        sha256_bytes(message["content"].encode("utf-8"))
        for row in examples
        for message in row["messages"]
        if message["role"] == "user"
    }


def resolve_parent_split_conflicts(
    examples: list[dict[str, Any]], lineage: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep the highest-confidence lane's split and drop opposing lower lanes."""
    priority = {
        "reviewed_final_answer": 0,
        "skill_policy": 1,
        "frontier_action_window": 2,
    }
    example_by_id = {row["example_id"]: row for row in examples}
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in lineage:
        example = example_by_id.get(source["example_id"])
        if example is None:
            raise ValueError("lineage refers to an unknown example")
        by_parent[source["parent_id"]].append(example)
    excluded: set[str] = set()
    excluded_lanes: Counter[str] = Counter()
    conflict_parents = 0
    for rows in by_parent.values():
        splits = {row["split"] for row in rows}
        if len(splits) <= 1:
            continue
        conflict_parents += 1
        best_priority = min(priority[row["lane"]] for row in rows)
        authoritative = {
            row["split"] for row in rows if priority[row["lane"]] == best_priority
        }
        if len(authoritative) != 1:
            raise ValueError("same-priority lane has contradictory parent splits")
        keep_split = next(iter(authoritative))
        for row in rows:
            if row["split"] != keep_split:
                excluded.add(row["example_id"])
                excluded_lanes[row["lane"]] += 1
    selected_examples = [row for row in examples if row["example_id"] not in excluded]
    selected_lineage = [row for row in lineage if row["example_id"] not in excluded]
    return selected_examples, selected_lineage, {
        "policy": "reviewed_final_answer_then_skill_policy_then_frontier_action_window",
        "conflicting_parents": conflict_parents,
        "excluded_rows": len(excluded),
        "excluded_by_lane": dict(sorted(excluded_lanes.items())),
    }


def _tokenizer_gate(
    examples: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    *,
    model_dir: Path,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer

    from runtime.sft.dataset import tokenize_final_assistant, validate_example

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    selected: list[dict[str, Any]] = []
    excluded: set[str] = set()
    lengths: list[int] = []
    lane_exclusions: Counter[str] = Counter()
    for row in examples:
        validate_example(row, expected_split=row["split"])
        try:
            tokenized = tokenize_final_assistant(row, tokenizer, max_length=max_length)
        except ValueError as exc:
            if not str(exc).startswith("sequence exceeds"):
                raise
            excluded.add(row["example_id"])
            lane_exclusions[row["lane"]] += 1
            continue
        selected.append(row)
        lengths.append(tokenized["sequence_tokens"])
    selected_lineage = [row for row in lineage if row["example_id"] not in excluded]
    if len(selected_lineage) != len(selected):
        raise ValueError("tokenizer gate broke example/lineage identity")
    lengths.sort()
    return selected, selected_lineage, {
        "status": "passed",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "model_dir": str(model_dir.resolve()),
        "max_length": max_length,
        "truncation": False,
        "selected": len(selected),
        "excluded_over_length": len(excluded),
        "excluded_by_lane": dict(sorted(lane_exclusions.items())),
        "sequence_tokens": {
            "min": min(lengths, default=0),
            "p50": lengths[len(lengths) // 2] if lengths else 0,
            "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0,
            "max": max(lengths, default=0),
        },
    }


def _contamination_check(
    examples: list[dict[str, Any]], *, eval_cases: Path, benchmark_registry: Path
) -> dict[str, Any]:
    prompt_hashes = _prompt_hashes(examples)
    eval_hashes: set[str] = set()
    for _, _, row in iter_jsonl(eval_cases):
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            eval_hashes.add(sha256_bytes(prompt.encode("utf-8")))
    benchmark_hashes: set[str] = set()
    benchmark_tasks = 0
    for _, _, row in iter_jsonl(benchmark_registry):
        path = Path(str(row.get("harbor_task_path", ""))) / "instruction.md"
        expected = row.get("instruction_sha256")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"benchmark instruction binding failed: {path}")
        benchmark_hashes.add(expected)
        benchmark_tasks += 1
    return {
        "method": "exact_sha256_of_individual_user_message_or_benchmark_instruction",
        "training_user_message_hashes": len(prompt_hashes),
        "diagnostic_case_count": len(eval_hashes),
        "diagnostic_exact_overlap": len(prompt_hashes & eval_hashes),
        "benchmark_task_count": benchmark_tasks,
        "benchmark_exact_overlap": len(prompt_hashes & benchmark_hashes),
        "eval_cases": file_descriptor(eval_cases),
        "benchmark_registry": file_descriptor(benchmark_registry, benchmark_tasks),
    }


def build_pilot(
    *,
    reviewed_dir: Path,
    action_dir: Path,
    skill_root: Path,
    tool_schema_path: Path,
    eval_cases: Path,
    benchmark_registry: Path,
    model_dir: Path,
    output_dir: Path,
    train_cap: int = 1024,
    validation_cap: int = 128,
    train_parent_cap: int = 24,
    validation_parent_cap: int = 16,
    max_characters: int = 24000,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    reviewed, reviewed_lineage, reviewed_decisions, reviewed_files = _reviewed_examples(
        reviewed_dir.resolve(), max_characters=max_characters
    )
    action_refs, action_decisions, action_files = _action_refs(
        action_dir.resolve(), max_characters=max_characters
    )
    selected_refs, cap_decisions = select_action_refs(
        action_refs,
        total_caps={"train": train_cap, "validation": validation_cap},
        parent_caps={"train": train_parent_cap, "validation": validation_parent_cap},
    )
    actions, action_lineage = _load_selected_actions(
        selected_refs, max_characters=max_characters
    )
    skills, skill_lineage, skill_bindings = build_skill_curriculum(
        skill_root=skill_root.resolve(), tool_schema_path=tool_schema_path.resolve()
    )
    all_examples = [*reviewed, *actions, *skills]
    all_lineage = [*reviewed_lineage, *action_lineage, *skill_lineage]
    all_examples, all_lineage, split_resolution = resolve_parent_split_conflicts(
        all_examples, all_lineage
    )
    all_examples, all_lineage, tokenizer_gate = _tokenizer_gate(
        all_examples,
        all_lineage,
        model_dir=model_dir,
        max_length=8192,
    )
    ids = [row["example_id"] for row in all_examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example identity")
    lineage_ids = {row["example_id"] for row in all_lineage}
    if lineage_ids != set(ids):
        raise ValueError("example/lineage identity mismatch")
    parent_splits: dict[str, set[str]] = defaultdict(set)
    split_by_id = {row["example_id"]: row["split"] for row in all_examples}
    for row in all_lineage:
        parent_splits[row["parent_id"]].add(split_by_id[row["example_id"]])
    overlap = sorted(parent for parent, splits in parent_splits.items() if len(splits) > 1)
    if overlap:
        raise ValueError(f"parent split overlap: {overlap[:5]}")
    contamination = _contamination_check(
        all_examples, eval_cases=eval_cases.resolve(), benchmark_registry=benchmark_registry.resolve()
    )
    if contamination["diagnostic_exact_overlap"] or contamination["benchmark_exact_overlap"]:
        raise ValueError("exact evaluation contamination detected")

    sorted_examples = sorted(all_examples, key=lambda row: (row["split"], row["lane"], row["example_id"]))
    sorted_lineage = sorted(all_lineage, key=lambda row: row["example_id"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        files = {
            "train.jsonl": _write_jsonl(staging / "train.jsonl", (row for row in sorted_examples if row["split"] == "train")),
            "validation.jsonl": _write_jsonl(staging / "validation.jsonl", (row for row in sorted_examples if row["split"] == "validation")),
            "lineage.jsonl": _write_jsonl(staging / "lineage.jsonl", sorted_lineage),
        }
        lane_counts = Counter((row["split"], row["lane"]) for row in all_examples)
        manifest = {
            "schema_version": SCHEMA,
            "status": "ready_for_internal_sft_pilot",
            "purpose": "bounded_behavior_qualification_not_release_or_rl",
            "model": {
                "repo_id": "Qwen/Qwen3.5-9B",
                "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "local_path": "/data-120/models/Qwen3.5-9B",
                "chat_template_kwargs": {"enable_thinking": False},
                "max_sequence_tokens": 8192,
                "loss": "final_assistant_turn_only",
            },
            "inputs": {
                "reviewed_final_answers": {
                    "directory": str(reviewed_dir.resolve()),
                    "files": reviewed_files,
                },
                "frontier_action_windows": {
                    "directory": str(action_dir.resolve()),
                    "manifest": file_descriptor(action_dir.resolve() / "manifest.json"),
                    "files": action_files,
                },
                "skill_policy": skill_bindings,
            },
            "selection": {
                "frontier_only": True,
                "no_raw_session_rebuild": True,
                "action_total_caps": {"train": train_cap, "validation": validation_cap},
                "action_parent_caps": {"train": train_parent_cap, "validation": validation_parent_cap},
                "max_historical_message_characters": max_characters,
                "parent_disjoint": True,
                "parent_split_conflict_resolution": split_resolution,
                "legacy_brand_excluded": True,
                "provider_brand_prompts_excluded": True,
                "reasoning_excluded": True,
                "hard_privacy_firewall": True,
                "tool_schema_policy": "runtime schemas for curated policy; compatibility schemas reconstructed from observed valid historical calls",
            },
            "counts": {
                "total": len(all_examples),
                "train": sum(row["split"] == "train" for row in all_examples),
                "validation": sum(row["split"] == "validation" for row in all_examples),
                "lanes": {
                    f"{split}:{lane}": count
                    for (split, lane), count in sorted(lane_counts.items())
                },
                "unique_parents": len(parent_splits),
            },
            "decisions": {
                "reviewed": dict(sorted(reviewed_decisions.items())),
                "actions": dict(sorted(action_decisions.items())),
                "caps": dict(sorted(cap_decisions.items())),
            },
            "quality": {
                "historical_action_outcomes": "not_independently_replayed",
                "historical_tool_schemas": "compatibility_only_runtime_schema_is_authoritative",
                "skill_policy": "source_bound_authored_examples",
                "rewards": "not_present_not_rl_eligible",
                "tokenizer_selection_gate": tokenizer_gate,
                "full_release_preflight": "pending",
                "runtime_weight_update": "pending",
            },
            "contamination": contamination,
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reviewed-dir", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--action-dir", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL_ROOT)
    parser.add_argument("--tool-schemas", type=Path, default=DEFAULT_TOOL_SCHEMAS)
    parser.add_argument("--eval-cases", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--benchmark-registry", type=Path, default=DEFAULT_BENCHMARK_REGISTRY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-cap", type=int, default=1024)
    parser.add_argument("--validation-cap", type=int, default=128)
    parser.add_argument("--train-parent-cap", type=int, default=24)
    parser.add_argument("--validation-parent-cap", type=int, default=16)
    parser.add_argument("--max-characters", type=int, default=24000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_pilot(
        reviewed_dir=args.reviewed_dir,
        action_dir=args.action_dir,
        skill_root=args.skill_root,
        tool_schema_path=args.tool_schemas,
        eval_cases=args.eval_cases,
        benchmark_registry=args.benchmark_registry,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        train_cap=args.train_cap,
        validation_cap=args.validation_cap,
        train_parent_cap=args.train_parent_cap,
        validation_parent_cap=args.validation_parent_cap,
        max_characters=args.max_characters,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
