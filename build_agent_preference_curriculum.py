#!/usr/bin/env python3
"""Build same-state agent preferences from qualified policy and tool data."""

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

from build_agent_sft_curriculum import (
    assert_model_text_clean,
    digest_value,
    normalize_when2call_target,
    normalize_when2call_tools,
    release_rows,
)
from runtime.dpo.dataset import PAIR_SCHEMA, RELEASE_SCHEMA, validate_pair
from runtime.sft.dataset import sha256_file
from runtime.sft.filter_release import canonical_bytes, iter_bound_jsonl, write_jsonl


MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LINEAGE_SCHEMA = "ai-data-extraction/agent-preference-lineage/v1"
DECISION_SCHEMA = "ai-data-extraction/agent-preference-decision/v1"


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _copy_call_with_new_id(message: dict[str, Any], call_id: str) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("negative source action must contain one tool call")
    rejected = copy.deepcopy(message)
    rejected["tool_calls"][0]["id"] = call_id
    return rejected


def _tool_names(example: dict[str, Any]) -> set[str]:
    return {tool["function"]["name"] for tool in example.get("tools") or []}


def skill_rejected_action(
    example: dict[str, Any], source_kind: str
) -> tuple[dict[str, Any], str]:
    messages = example["messages"]
    names = _tool_names(example)
    if source_kind == "skill_route":
        return (
            _tool_call(
                "search_code", {"query": "implement the requested change"}, "call-rejected"
            ),
            "bypasses_required_skill",
        )
    if source_kind == "skill_apply":
        first_action = next(
            message for message in messages[:-1] if message.get("role") == "assistant"
        )
        return (
            _copy_call_with_new_id(first_action, "call-rejected"),
            "repeats_completed_skill_read",
        )
    if source_kind == "skill_skip_low_return":
        return (
            _tool_call("read_skill", {"name": "core-principles"}, "call-rejected"),
            "unnecessary_skill_overhead",
        )
    if source_kind == "bounded_termination":
        return (
            _tool_call("read_skill", {"name": "core-principles"}, "call-rejected"),
            "delays_required_decision",
        )

    prior_actions = [
        message
        for message in messages[:-1]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    if prior_actions:
        return (
            _copy_call_with_new_id(prior_actions[-1], "call-rejected"),
            "repeats_observed_action_instead_of_follow_through",
        )
    if "read_skill" in names:
        return (
            _tool_call("read_skill", {"name": "core-principles"}, "call-rejected"),
            "unnecessary_skill_overhead",
        )
    raise ValueError(f"no controlled negative for skill policy kind: {source_kind}")


def gate_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"invalid evaluation prompt at line {line_number}")
            prompts.add(prompt.strip())
    return prompts


def initial_user_text(prompt: list[dict[str, Any]]) -> str:
    for message in prompt:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"].strip()
    raise ValueError("preference prompt has no user message")


def make_pair(
    *,
    split: str,
    lane: str,
    prompt: list[dict[str, Any]],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    tools: list[dict[str, Any]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    pair = {
        "schema_version": PAIR_SCHEMA,
        "pair_id": "sha256:"
        + digest_value(
            {
                "identity": identity,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tools": tools,
            }
        ),
        "split": split,
        "lane": lane,
        "prompt": copy.deepcopy(prompt),
        "chosen": [copy.deepcopy(chosen)],
        "rejected": [copy.deepcopy(rejected)],
        "tools": copy.deepcopy(tools),
    }
    validate_pair(pair, expected_split=split)
    return pair


def build_skill_preferences(
    *,
    source_release: Path,
    excluded_prompts: set[str],
    blocked_text_patterns: tuple[re.Pattern[str], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest, rows = release_rows(source_release)
    manifest_sha = sha256_file(source_release / "manifest.json")
    lineage = {
        row["example_id"]: row
        for row in iter_bound_jsonl(source_release, manifest, "lineage.jsonl")
    }
    pairs: list[dict[str, Any]] = []
    pair_lineage: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for split, row, _ in rows:
        if row["lane"] != "skill_policy":
            continue
        source = lineage[row["example_id"]]
        source_kind = source.get("source_kind")
        decision = {
            "schema_version": DECISION_SCHEMA,
            "source_partition": "skill_policy",
            "source_example_id": row["example_id"],
            "source_kind": source_kind,
        }
        try:
            if initial_user_text(row["messages"][:-1]) in excluded_prompts:
                raise ValueError("evaluation_prompt_overlap")
            rejected, reason = skill_rejected_action(row, str(source_kind))
            pair = make_pair(
                split=split,
                lane="skill_policy_preference",
                prompt=row["messages"][:-1],
                chosen=row["messages"][-1],
                rejected=rejected,
                tools=row["tools"],
                identity={
                    "source_manifest_sha256": manifest_sha,
                    "source_example_id": row["example_id"],
                },
            )
            assert_model_text_clean(pair, blocked_text_patterns)
            pairs.append(pair)
            pair_lineage.append(
                {
                    "schema_version": LINEAGE_SCHEMA,
                    "pair_id": pair["pair_id"],
                    "parent_id": source["parent_id"],
                    "source_partition": "skill_policy",
                    "source_manifest_sha256": manifest_sha,
                    "source_example_id": row["example_id"],
                    "source_kind": source_kind,
                    "skill": source.get("skill"),
                    "skill_sha256": source.get("skill_sha256"),
                    "preference_basis": reason,
                    "label_strength": "deterministic_policy_contract",
                }
            )
            reasons[reason] += 1
            decision.update(
                {"decision": "selected", "pair_id": pair["pair_id"], "reason": reason}
            )
        except (KeyError, TypeError, ValueError) as exc:
            decision.update({"decision": "excluded", "reason": str(exc)})
        decisions.append(decision)
    return pairs, pair_lineage, decisions, {
        "source_manifest_sha256": manifest_sha,
        "source_status": manifest.get("status"),
        "selected": len(pairs),
        "preference_basis": dict(sorted(reasons.items())),
    }


def build_when2call_preferences(
    *,
    sft_release: Path,
    raw_source: Path,
    source_revision: str,
    excluded_prompts: set[str],
    blocked_text_patterns: tuple[re.Pattern[str], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest, rows = release_rows(sft_release)
    manifest_sha = sha256_file(sft_release / "manifest.json")
    selected: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for _, row, lineage in rows:
        if lineage.get("source_partition") != "when2call_train_pref":
            continue
        source_row = lineage.get("source_row")
        if not isinstance(source_row, int) or source_row in selected:
            raise ValueError("invalid or duplicate selected When2Call source row")
        selected[source_row] = (row, lineage)

    pairs: list[dict[str, Any]] = []
    pair_lineage: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    selected_rows: set[int] = set()
    categories: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    with raw_source.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line_number not in selected:
                continue
            row, lineage = selected[line_number]
            raw = json.loads(line)
            decision = {
                "schema_version": DECISION_SCHEMA,
                "source_partition": "when2call_train_pref",
                "source_row": line_number,
                "source_row_sha256": lineage["source_row_sha256"],
            }
            try:
                if digest_value(raw) != lineage["source_row_sha256"]:
                    raise ValueError("source_row_hash_mismatch")
                tools = normalize_when2call_tools(raw.get("tools"))
                if tools != row.get("tools"):
                    raise ValueError("tool_projection_mismatch")
                messages = raw.get("messages")
                if not isinstance(messages, list) or len(messages) != 1:
                    raise ValueError("source_prompt_invalid")
                if messages[0] != row["messages"][0]:
                    raise ValueError("prompt_projection_mismatch")
                if initial_user_text(messages) in excluded_prompts:
                    raise ValueError("evaluation_prompt_overlap")
                chosen_category, chosen = normalize_when2call_target(
                    raw["chosen_response"]["content"], tools, line_number
                )
                if chosen != row["messages"][-1]:
                    raise ValueError("chosen_projection_mismatch")
                _, rejected = normalize_when2call_target(
                    raw["rejected_response"]["content"], tools, line_number
                )
                pair = make_pair(
                    split=row["split"],
                    lane="tool_decision_preference",
                    prompt=row["messages"][:-1],
                    chosen=chosen,
                    rejected=rejected,
                    tools=tools,
                    identity={
                        "dataset_revision": source_revision,
                        "source_row_sha256": lineage["source_row_sha256"],
                    },
                )
                assert_model_text_clean(pair, blocked_text_patterns)
                pairs.append(pair)
                pair_lineage.append(
                    {
                        "schema_version": LINEAGE_SCHEMA,
                        "pair_id": pair["pair_id"],
                        "parent_id": lineage["parent_id"],
                        "source_partition": "when2call_train_pref",
                        "source_revision": source_revision,
                        "source_row": line_number,
                        "source_row_sha256": lineage["source_row_sha256"],
                        "source_category": chosen_category,
                        "preference_basis": "source_chosen_over_rejected",
                        "label_strength": "automated_preference_label_not_executed",
                    }
                )
                categories[chosen_category] += 1
                selected_rows.add(line_number)
                decision.update(
                    {
                        "decision": "selected",
                        "pair_id": pair["pair_id"],
                        "category": chosen_category,
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = str(exc)
                exclusions[reason] += 1
                decision.update({"decision": "excluded", "reason": reason})
            decisions.append(decision)
    if set(selected) - selected_rows != {
        item["source_row"] for item in decisions if item.get("decision") == "excluded"
    }:
        raise ValueError("selected When2Call rows did not reconcile")
    return pairs, pair_lineage, decisions, {
        "selected_sft_manifest_sha256": manifest_sha,
        "selected_sft_status": manifest.get("status"),
        "requested": len(selected),
        "selected": len(pairs),
        "selected_by_category": dict(sorted(categories.items())),
        "excluded_reasons": dict(sorted(exclusions.items())),
    }


def _verify_parent_splits(
    pairs: list[dict[str, Any]], lineage: list[dict[str, Any]]
) -> int:
    split_by_id = {row["pair_id"]: row["split"] for row in pairs}
    parent_splits: defaultdict[str, set[str]] = defaultdict(set)
    for item in lineage:
        parent_splits[item["parent_id"]].add(split_by_id[item["pair_id"]])
    if any(len(splits) != 1 for splits in parent_splits.values()):
        raise ValueError("preference parent split overlap")
    return len(parent_splits)


def build_curriculum(
    *,
    skill_release: Path,
    sft_release: Path,
    when2call_source: Path,
    when2call_revision: str,
    evaluation_cases: Path,
    output_dir: Path,
    blocked_text_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    compiled_blocked = tuple(
        re.compile(pattern, re.I) for pattern in blocked_text_patterns if pattern
    )
    if len(compiled_blocked) != len(blocked_text_patterns):
        raise ValueError("blocked text patterns must be non-empty")
    excluded_prompts = gate_prompts(evaluation_cases.resolve())

    skill = build_skill_preferences(
        source_release=skill_release.resolve(),
        excluded_prompts=excluded_prompts,
        blocked_text_patterns=compiled_blocked,
    )
    tool = build_when2call_preferences(
        sft_release=sft_release.resolve(),
        raw_source=when2call_source.resolve(),
        source_revision=when2call_revision,
        excluded_prompts=excluded_prompts,
        blocked_text_patterns=compiled_blocked,
    )
    pairs = skill[0] + tool[0]
    lineage = skill[1] + tool[1]
    decisions = skill[2] + tool[2]
    pair_ids = [row["pair_id"] for row in pairs]
    lineage_ids = [row["pair_id"] for row in lineage]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate preference pair identity")
    if len(lineage_ids) != len(set(lineage_ids)) or set(lineage_ids) != set(pair_ids):
        raise ValueError("preference lineage identity mismatch")
    unique_parents = _verify_parent_splits(pairs, lineage)
    if any(initial_user_text(row["prompt"]) in excluded_prompts for row in pairs):
        raise ValueError("evaluation prompt leaked into preference release")

    pairs.sort(key=lambda row: (row["split"], row["pair_id"]))
    lineage.sort(key=lambda row: row["pair_id"])
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
                (row for row in pairs if row["split"] == "train"),
            ),
            "validation.jsonl": write_jsonl(
                staging / "validation.jsonl",
                (row for row in pairs if row["split"] == "validation"),
            ),
            "lineage.jsonl": write_jsonl(staging / "lineage.jsonl", lineage),
            "decisions.jsonl": write_jsonl(staging / "decisions.jsonl", decisions),
        }
        lanes = Counter(f"{row['split']}:{row['lane']}" for row in pairs)
        partitions = Counter(row["source_partition"] for row in lineage)
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "status": "ready_for_exact_tokenizer_preflight",
            "purpose": "same_state_long_horizon_agent_preference_optimization",
            "model": {
                "repo_id": "Qwen/Qwen3.5-9B",
                "revision": MODEL_REVISION,
                "local_path": "/data-120/models/Qwen3.5-9B",
                "reference_adapter": "/data-120/models/adapters/Qwen3.5-9B-agent-sft-curriculum-v1",
                "chat_template_kwargs": {"enable_thinking": False},
                "max_sequence_tokens": 8192,
            },
            "sources": {
                "skill_policy": {
                    "path": str(skill_release.resolve()),
                    **skill[3],
                },
                "when2call": {
                    "dataset_id": "nvidia/When2Call",
                    "revision": when2call_revision,
                    "path": str(when2call_source.resolve()),
                    "sha256": sha256_file(when2call_source),
                    "license": "cc-by-4.0",
                    **tool[3],
                },
            },
            "selection": {
                "same_observable_prompt_state": True,
                "one_copy_per_pair": True,
                "hidden_reasoning": "not_present",
                "evaluation_cases_path": str(evaluation_cases.resolve()),
                "evaluation_cases_sha256": sha256_file(evaluation_cases),
                "evaluation_prompts_excluded": len(excluded_prompts),
                "evaluation_prompt_overlap": 0,
                "parent_disjoint": True,
                "invalid_rejected_tool_calls": "excluded_not_reformatted",
                "open_swe_outcome_only_preferences": "audit_only_not_selected",
                "blocked_text_pattern_sha256": sorted(
                    hashlib.sha256(pattern.encode()).hexdigest()
                    for pattern in blocked_text_patterns
                ),
            },
            "counts": {
                "total": len(pairs),
                "train": sum(row["split"] == "train" for row in pairs),
                "validation": sum(row["split"] == "validation" for row in pairs),
                "unique_parents": unique_parents,
                "partitions": dict(sorted(partitions.items())),
                "lanes": dict(sorted(lanes.items())),
            },
            "quality": {
                "skill_policy_labels": "deterministic_contract",
                "when2call_labels": "automated_preference_not_executed",
                "rewards": "not_invented",
                "tokenizer_preflight": "pending",
                "training_role": "offline_preference_optimization_not_rl",
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
    parser.add_argument("--skill-release", type=Path, required=True)
    parser.add_argument("--sft-release", type=Path, required=True)
    parser.add_argument("--when2call-source", type=Path, required=True)
    parser.add_argument("--when2call-revision", required=True)
    parser.add_argument("--evaluation-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocked-text-pattern", action="append", default=[])
    args = parser.parse_args()
    manifest = build_curriculum(
        skill_release=args.skill_release,
        sft_release=args.sft_release,
        when2call_source=args.when2call_source,
        when2call_revision=args.when2call_revision,
        evaluation_cases=args.evaluation_cases,
        output_dir=args.output_dir,
        blocked_text_patterns=tuple(args.blocked_text_pattern),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
