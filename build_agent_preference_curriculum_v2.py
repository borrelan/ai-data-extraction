#!/usr/bin/env python3
"""Build a state-transition-heavy preference release for agentic DPO."""

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
from typing import Any, Callable

from build_agent_preference_curriculum import (
    DECISION_SCHEMA,
    LINEAGE_SCHEMA,
    build_when2call_preferences,
    gate_prompts,
    initial_user_text,
    make_pair,
)
from build_agent_sft_curriculum import assert_model_text_clean, digest_value, release_rows
from runtime.dpo.dataset import RELEASE_SCHEMA, render_pair
from runtime.sft.dataset import sha256_file
from runtime.sft.filter_release import canonical_bytes, write_jsonl


MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
REQUIRED_TRAIN_FAMILIES = {
    "permission_read_only_boundary",
    "post_error_changed_recovery",
    "premature_stop_before_required_work",
    "productive_follow_through_after_observation",
    "required_skill_routing",
    "verified_finish_over_unnecessary_continuation",
}
PROMOTION_COVERAGE = {
    "tool_schema": (
        "generic_tool_decision_replay",
        "productive_follow_through_after_observation",
        "required_skill_routing",
    ),
    "loop_recovery": (
        "post_error_changed_recovery",
        "productive_follow_through_after_observation",
    ),
    "skill_routing": (
        "required_skill_routing",
        "unnecessary_skill_overhead",
    ),
    "completion_stop": (
        "blocker_over_false_completion",
        "permission_read_only_boundary",
        "premature_stop_before_required_work",
        "verified_finish_over_unnecessary_continuation",
    ),
}


def _tool_names(example: dict[str, Any]) -> set[str]:
    return {tool["function"]["name"] for tool in example.get("tools") or []}


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


def _copy_action(message: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("negative source action has no tool call")
    result = copy.deepcopy(message)
    for index, call in enumerate(result["tool_calls"]):
        call["id"] = f"call-rejected-{index:02d}"
    return result


def _last_prior_action(messages: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return index, message
    raise ValueError("prior_tool_action_missing")


def _action_signature(message: dict[str, Any]) -> tuple[tuple[str, bytes], ...]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("tool_action_missing")
    return tuple(
        (
            call["function"]["name"],
            canonical_bytes(call["function"]["arguments"]),
        )
        for call in calls
    )


def strict_observation_status(
    prompt: list[dict[str, Any]], prior_action_index: int
) -> str:
    """Classify only explicit structured return codes; never infer from prose."""

    observations = prompt[prior_action_index + 1 :]
    if not observations or any(item.get("role") != "tool" for item in observations):
        return "unknown"
    returncodes: list[int] = []
    for observation in observations:
        content = observation.get("content")
        if not isinstance(content, str):
            return "unknown"
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return "unknown"
        if not isinstance(value, dict) or not isinstance(value.get("returncode"), int):
            return "unknown"
        returncodes.append(value["returncode"])
    return "explicit_failure" if any(returncodes) else "explicit_success"


def _unauthorized_patch() -> dict[str, Any]:
    return _tool_call(
        "apply_patch",
        {
            "path": "src/implementation.py",
            "patch": "Modify source despite the read-only task.",
        },
        "call-rejected",
    )


def skill_negative_v2(
    example: dict[str, Any], source_kind: str
) -> tuple[dict[str, Any], str]:
    """Return one policy-invalid action and its behavior family."""

    messages = example["messages"]
    names = _tool_names(example)
    chosen = messages[-1]
    if source_kind == "skill_route":
        return (
            _tool_call(
                "search_code", {"query": "implement the requested change"}, "call-rejected"
            ),
            "required_skill_routing",
        )
    if source_kind == "skill_skip_low_return":
        return (
            _tool_call("read_skill", {"name": "core-principles"}, "call-rejected"),
            "unnecessary_skill_overhead",
        )
    if source_kind == "skill_apply":
        return (
            {"role": "assistant", "content": "The task is complete."},
            "premature_stop_before_required_work",
        )
    if source_kind in {
        "bounded_command_report",
        "diagnose_without_repair",
        "diagnostic_report_holdout",
        "read_only_report",
        "skill_bounded_read_only",
        "untrusted_observation_holdout",
        "untrusted_observation_report",
        "verified_noop_report",
    }:
        if "apply_patch" not in names:
            raise ValueError("read_only_negative_requires_apply_patch")
        return _unauthorized_patch(), "permission_read_only_boundary"
    if source_kind in {"evidence_bound_blocker", "blocker_holdout"}:
        if "finish_task" not in names:
            raise ValueError("blocker_negative_requires_finish_task")
        return (
            _tool_call(
                "finish_task",
                {
                    "summary": "The task is complete.",
                    "evidence": "No blocking prerequisite remains.",
                },
                "call-rejected",
            ),
            "blocker_over_false_completion",
        )
    if source_kind in {"verified_finish", "verified_finish_holdout", "verified_mutation_finish"}:
        _, prior = _last_prior_action(messages[:-1])
        return _copy_action(prior), "verified_finish_over_unnecessary_continuation"
    if source_kind == "single_inspection_report":
        _, prior = _last_prior_action(messages[:-1])
        return _copy_action(prior), "verified_finish_over_unnecessary_continuation"
    if source_kind in {"no_repeat_alternate", "skill_fallback_report"}:
        _, prior = _last_prior_action(messages[:-1])
        return _copy_action(prior), "post_error_changed_recovery"
    if source_kind == "bounded_termination":
        chosen_calls = chosen.get("tool_calls") or []
        chosen_name = chosen_calls[0]["function"]["name"] if chosen_calls else ""
        if chosen_name == "search_code":
            return (
                {"role": "assistant", "content": "The task is complete."},
                "premature_stop_before_required_work",
            )
        return (
            _tool_call(
                "search_code", {"query": "continue investigating"}, "call-rejected"
            ),
            "verified_finish_over_unnecessary_continuation",
        )
    raise ValueError(f"no_v2_policy_negative:{source_kind}")


def build_skill_preferences_v2(
    *,
    source_release: Path,
    excluded_prompts: set[str],
    blocked_text_patterns: tuple[re.Pattern[str], ...],
    pair_gate: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest, rows = release_rows(source_release)
    manifest_sha = sha256_file(source_release / "manifest.json")
    pairs: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    families: Counter[str] = Counter()
    for split, row, source in rows:
        if row["lane"] != "skill_policy":
            continue
        source_kind = str(source.get("source_kind"))
        decision = {
            "schema_version": DECISION_SCHEMA,
            "source_partition": "skill_policy",
            "source_example_id": row["example_id"],
            "source_kind": source_kind,
        }
        try:
            if initial_user_text(row["messages"][:-1]) in excluded_prompts:
                raise ValueError("evaluation_prompt_overlap")
            rejected, family = skill_negative_v2(row, source_kind)
            pair = make_pair(
                split=split,
                lane="skill_policy_preference",
                prompt=row["messages"][:-1],
                chosen=row["messages"][-1],
                rejected=rejected,
                tools=row["tools"],
                identity={
                    "curriculum": "v2",
                    "source_manifest_sha256": manifest_sha,
                    "source_example_id": row["example_id"],
                    "preference_family": family,
                },
            )
            assert_model_text_clean(pair, blocked_text_patterns)
            if not pair_gate(pair):
                raise ValueError("exact_token_limit")
            pairs.append(pair)
            lineage.append(
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
                    "preference_family": family,
                    "preference_basis": family,
                    "label_strength": "deterministic_policy_contract",
                }
            )
            families[family] += 1
            decision.update(
                {"decision": "selected", "pair_id": pair["pair_id"], "reason": family}
            )
        except (KeyError, TypeError, ValueError) as exc:
            decision.update({"decision": "excluded", "reason": str(exc)})
        decisions.append(decision)
    return pairs, lineage, decisions, {
        "source_manifest_sha256": manifest_sha,
        "source_status": manifest.get("status"),
        "selected": len(pairs),
        "preference_families": dict(sorted(families.items())),
    }


def build_transition_preferences(
    *,
    source_release: Path,
    excluded_prompts: set[str],
    blocked_text_patterns: tuple[re.Pattern[str], ...],
    caps: dict[tuple[str, str], int],
    pair_gate: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest, rows = release_rows(source_release)
    manifest_sha = sha256_file(source_release / "manifest.json")
    candidates: defaultdict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for split, row, source in rows:
        if row["lane"] != "verified_open_swe_action":
            continue
        decision = {
            "schema_version": DECISION_SCHEMA,
            "source_partition": "open_swe_transition",
            "source_example_id": row["example_id"],
        }
        try:
            prompt = row["messages"][:-1]
            if initial_user_text(prompt) in excluded_prompts:
                raise ValueError("evaluation_prompt_overlap")
            prior_index, prior = _last_prior_action(prompt)
            status = strict_observation_status(prompt, prior_index)
            if status == "unknown":
                raise ValueError("observation_status_not_explicit")
            chosen = row["messages"][-1]
            if _action_signature(prior) == _action_signature(chosen):
                raise ValueError("chosen_repeats_observed_action")
            family = (
                "post_error_changed_recovery"
                if status == "explicit_failure"
                else "productive_follow_through_after_observation"
            )
            rejected = _copy_action(prior)
            pair = make_pair(
                split=split,
                lane="state_transition_preference",
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                tools=row["tools"],
                identity={
                    "curriculum": "v2",
                    "source_manifest_sha256": manifest_sha,
                    "source_example_id": row["example_id"],
                    "preference_family": family,
                },
            )
            assert_model_text_clean(pair, blocked_text_patterns)
            if not pair_gate(pair):
                raise ValueError("exact_token_limit")
            item_lineage = {
                "schema_version": LINEAGE_SCHEMA,
                "pair_id": pair["pair_id"],
                "parent_id": source["parent_id"],
                "source_partition": "open_swe_transition",
                "source_manifest_sha256": manifest_sha,
                "source_example_id": row["example_id"],
                "preference_family": family,
                "preference_basis": "changed_action_after_explicit_tool_observation",
                "observation_status": status,
                "label_strength": "source_executed_success_transition_not_locally_replayed",
            }
            candidates[(split, family)].append((pair, item_lineage, decision))
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
            exclusions[reason] += 1
            decision.update({"decision": "excluded", "reason": reason})
            decisions.append(decision)

    selected_pairs: list[dict[str, Any]] = []
    selected_lineage: list[dict[str, Any]] = []
    available: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    for key, items in sorted(candidates.items()):
        split, family = key
        items.sort(key=lambda item: item[0]["pair_id"])
        available[f"{split}:{family}"] = len(items)
        seen_parents: set[str] = set()
        eligible: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for item in items:
            parent_id = item[1]["parent_id"]
            if parent_id in seen_parents:
                item[2].update({"decision": "excluded", "reason": "family_parent_cap"})
                decisions.append(item[2])
                continue
            seen_parents.add(parent_id)
            eligible.append(item)
        cap = caps[key]
        for index, (pair, item_lineage, decision) in enumerate(eligible):
            if index < cap:
                selected_pairs.append(pair)
                selected_lineage.append(item_lineage)
                selected[f"{split}:{family}"] += 1
                decision.update(
                    {"decision": "selected", "pair_id": pair["pair_id"], "reason": family}
                )
            else:
                decision.update({"decision": "excluded", "reason": "split_family_cap"})
            decisions.append(decision)
    return selected_pairs, selected_lineage, decisions, {
        "source_manifest_sha256": manifest_sha,
        "source_status": manifest.get("status"),
        "caps": {f"{split}:{family}": cap for (split, family), cap in sorted(caps.items())},
        "available": dict(sorted(available.items())),
        "selected": dict(sorted(selected.items())),
        "excluded_reasons": dict(sorted(exclusions.items())),
        "outcome_evidence": "source_executed_success_not_locally_replayed",
    }


def cap_when2call(
    result: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    *,
    split_caps: dict[str, int],
    pair_gate: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs, lineage, decisions, stats = result
    pair_by_id = {row["pair_id"]: row for row in pairs}
    lineage_by_id = {row["pair_id"]: row for row in lineage}
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    token_excluded: set[str] = set()
    for item in lineage:
        if not pair_gate(pair_by_id[item["pair_id"]]):
            token_excluded.add(item["pair_id"])
            continue
        split = pair_by_id[item["pair_id"]]["split"]
        groups[(split, item["source_category"])].append(item["pair_id"])
    selected_ids: set[str] = set()
    categories = sorted({category for _, category in groups})
    for split, total_cap in split_caps.items():
        base, remainder = divmod(total_cap, len(categories))
        for index, category in enumerate(categories):
            cap = base + (1 if index < remainder else 0)
            selected_ids.update(sorted(groups[(split, category)])[:cap])
    for item in decisions:
        pair_id = item.get("pair_id")
        if item.get("decision") == "selected" and pair_id not in selected_ids:
            item["decision"] = "excluded"
            item["reason"] = (
                "exact_token_limit"
                if pair_id in token_excluded
                else "v2_balanced_split_cap"
            )
            item.pop("pair_id", None)
    selected_pairs = [pair_by_id[pair_id] for pair_id in sorted(selected_ids)]
    selected_lineage = []
    for pair_id in sorted(selected_ids):
        item = copy.deepcopy(lineage_by_id[pair_id])
        item["preference_family"] = "generic_tool_decision_replay"
        selected_lineage.append(item)
    stats = copy.deepcopy(stats)
    stats["v2_split_caps"] = dict(sorted(split_caps.items()))
    stats["v2_selected"] = len(selected_pairs)
    stats["v2_exact_token_excluded"] = len(token_excluded)
    stats["v2_selected_by_split_category"] = dict(
        sorted(
            Counter(
                f"{pair_by_id[pair_id]['split']}:{lineage_by_id[pair_id]['source_category']}"
                for pair_id in selected_ids
            ).items()
        )
    )
    return selected_pairs, selected_lineage, decisions, stats


def _verify_release_constraints(
    pairs: list[dict[str, Any]], lineage: list[dict[str, Any]], excluded_prompts: set[str]
) -> dict[str, Any]:
    pair_by_id = {row["pair_id"]: row for row in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("duplicate_preference_pair_identity")
    lineage_by_id = {row["pair_id"]: row for row in lineage}
    if len(lineage_by_id) != len(lineage) or set(lineage_by_id) != set(pair_by_id):
        raise ValueError("preference_lineage_identity_mismatch")
    prompt_states: set[str] = set()
    parent_splits: defaultdict[str, set[str]] = defaultdict(set)
    families: Counter[str] = Counter()
    partitions_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for pair_id, pair in pair_by_id.items():
        item = lineage_by_id[pair_id]
        prompt_state = digest_value({"prompt": pair["prompt"], "tools": pair["tools"]})
        if prompt_state in prompt_states:
            raise ValueError("duplicate_prompt_tool_state")
        prompt_states.add(prompt_state)
        if initial_user_text(pair["prompt"]) in excluded_prompts:
            raise ValueError("evaluation_prompt_overlap")
        split = pair["split"]
        parent_splits[item["parent_id"]].add(split)
        family = item["preference_family"]
        families[f"{split}:{family}"] += 1
        partitions_by_split[split][item["source_partition"]] += 1
    if any(len(splits) != 1 for splits in parent_splits.values()):
        raise ValueError("preference_parent_split_overlap")
    train_families = {
        key.removeprefix("train:") for key in families if key.startswith("train:")
    }
    missing = sorted(REQUIRED_TRAIN_FAMILIES - train_families)
    if missing:
        raise ValueError(f"required_train_preference_family_missing:{','.join(missing)}")
    for split, partitions in partitions_by_split.items():
        total = sum(partitions.values())
        if any(count * 2 > total for count in partitions.values()):
            raise ValueError(f"source_partition_exceeds_half:{split}")
        generic = partitions.get("when2call_train_pref", 0)
        if generic >= total - generic:
            raise ValueError(f"generic_replay_not_below_stateful:{split}")
    coverage = {
        check: {
            family: families.get(f"train:{family}", 0) for family in required_families
        }
        for check, required_families in PROMOTION_COVERAGE.items()
    }
    if any(sum(values.values()) < 2 for values in coverage.values()):
        raise ValueError("promotion_check_has_insufficient_non_eval_coverage")
    return {
        "unique_prompt_tool_states": len(prompt_states),
        "unique_parents": len(parent_splits),
        "families": dict(sorted(families.items())),
        "partitions_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(partitions_by_split.items())
        },
        "promotion_coverage": coverage,
    }


def build_curriculum_v2(
    *,
    skill_release: Path,
    sft_release: Path,
    when2call_source: Path,
    when2call_revision: str,
    evaluation_cases: Path,
    model_dir: Path,
    output_dir: Path,
    blocked_text_patterns: tuple[str, ...] = (),
    open_swe_train_cap_per_family: int = 35,
    open_swe_validation_cap_per_family: int = 8,
    when2call_train_cap: int = 36,
    when2call_validation_cap: int = 12,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    compiled_blocked = tuple(re.compile(pattern, re.I) for pattern in blocked_text_patterns)
    excluded_prompts = gate_prompts(evaluation_cases.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    token_exclusions: dict[str, str] = {}

    def pair_gate(pair: dict[str, Any]) -> bool:
        try:
            render_pair(pair, tokenizer, max_length=8192)
            return True
        except ValueError as exc:
            if "exceeds 8192 tokens" not in str(exc):
                raise
            token_exclusions[pair["pair_id"]] = str(exc)
            return False

    skill = build_skill_preferences_v2(
        source_release=skill_release.resolve(),
        excluded_prompts=excluded_prompts,
        blocked_text_patterns=compiled_blocked,
        pair_gate=pair_gate,
    )
    transition = build_transition_preferences(
        source_release=sft_release.resolve(),
        excluded_prompts=excluded_prompts,
        blocked_text_patterns=compiled_blocked,
        pair_gate=pair_gate,
        caps={
            (split, family): cap
            for split, cap in (
                ("train", open_swe_train_cap_per_family),
                ("validation", open_swe_validation_cap_per_family),
            )
            for family in (
                "post_error_changed_recovery",
                "productive_follow_through_after_observation",
            )
        },
    )
    generic = cap_when2call(
        build_when2call_preferences(
            sft_release=sft_release.resolve(),
            raw_source=when2call_source.resolve(),
            source_revision=when2call_revision,
            excluded_prompts=excluded_prompts,
            blocked_text_patterns=compiled_blocked,
        ),
        split_caps={"train": when2call_train_cap, "validation": when2call_validation_cap},
        pair_gate=pair_gate,
    )
    pairs = skill[0] + transition[0] + generic[0]
    lineage = skill[1] + transition[1] + generic[1]
    decisions = skill[2] + transition[2] + generic[2]
    constraints = _verify_release_constraints(pairs, lineage, excluded_prompts)
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
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        files = {
            "train.jsonl": write_jsonl(
                staging / "train.jsonl", (row for row in pairs if row["split"] == "train")
            ),
            "validation.jsonl": write_jsonl(
                staging / "validation.jsonl",
                (row for row in pairs if row["split"] == "validation"),
            ),
            "lineage.jsonl": write_jsonl(staging / "lineage.jsonl", lineage),
            "decisions.jsonl": write_jsonl(staging / "decisions.jsonl", decisions),
        }
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "status": "ready_for_exact_tokenizer_preflight",
            "purpose": "state_transition_heavy_long_horizon_agent_preference_optimization",
            "builder": {
                "path": Path(__file__).name,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "model": {
                "repo_id": "Qwen/Qwen3.5-9B",
                "revision": MODEL_REVISION,
                "local_path": "/data-120/models/Qwen3.5-9B",
                "reference_adapter": "/data-120/models/adapters/Qwen3.5-9B-agent-sft-curriculum-v1",
                "chat_template_kwargs": {"enable_thinking": False},
                "max_sequence_tokens": 8192,
            },
            "sources": {
                "skill_policy": {"path": str(skill_release.resolve()), **skill[3]},
                "open_swe_transition": {"path": str(sft_release.resolve()), **transition[3]},
                "when2call": {
                    "dataset_id": "nvidia/When2Call",
                    "revision": when2call_revision,
                    "path": str(when2call_source.resolve()),
                    "sha256": sha256_file(when2call_source),
                    "license": "cc-by-4.0",
                    **generic[3],
                },
            },
            "selection": {
                "same_observable_prompt_state": True,
                "one_pair_per_prompt_tool_state": True,
                "hidden_reasoning": "not_present",
                "evaluation_cases_path": str(evaluation_cases.resolve()),
                "evaluation_cases_sha256": sha256_file(evaluation_cases),
                "evaluation_prompts_excluded": len(excluded_prompts),
                "evaluation_prompt_overlap": 0,
                "parent_disjoint": True,
                "explicit_observation_status": "structured_returncode_only",
                "open_swe_parent_cap_per_split_family": 1,
                "generic_replay_below_stateful": True,
                "source_partition_max_fraction": 0.5,
                "required_train_families": sorted(REQUIRED_TRAIN_FAMILIES),
                "blocked_text_pattern_sha256": sorted(
                    hashlib.sha256(pattern.encode()).hexdigest()
                    for pattern in blocked_text_patterns
                ),
                "exact_tokenizer_filter": {
                    "model_dir": str(model_dir.resolve()),
                    "tokenizer_class": type(tokenizer).__name__,
                    "tokenizer_length": len(tokenizer),
                    "chat_template_sha256": hashlib.sha256(
                        str(tokenizer.chat_template).encode("utf-8")
                    ).hexdigest(),
                    "max_sequence_tokens": 8192,
                    "excluded": len(token_exclusions),
                    "excluded_reasons": dict(
                        sorted(Counter(token_exclusions.values()).items())
                    ),
                },
            },
            "coverage": constraints,
            "counts": {
                "total": len(pairs),
                "train": sum(row["split"] == "train" for row in pairs),
                "validation": sum(row["split"] == "validation" for row in pairs),
                "lanes": dict(
                    sorted(Counter(f"{row['split']}:{row['lane']}" for row in pairs).items())
                ),
            },
            "quality": {
                "skill_policy_labels": "deterministic_contract",
                "open_swe_labels": "source_executed_transition_not_locally_replayed",
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocked-text-pattern", action="append", default=[])
    parser.add_argument("--open-swe-train-cap-per-family", type=int, default=35)
    parser.add_argument("--open-swe-validation-cap-per-family", type=int, default=8)
    parser.add_argument("--when2call-train-cap", type=int, default=36)
    parser.add_argument("--when2call-validation-cap", type=int, default=12)
    args = parser.parse_args()
    result = build_curriculum_v2(
        skill_release=args.skill_release,
        sft_release=args.sft_release,
        when2call_source=args.when2call_source,
        when2call_revision=args.when2call_revision,
        evaluation_cases=args.evaluation_cases,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        blocked_text_patterns=tuple(args.blocked_text_pattern),
        open_swe_train_cap_per_family=args.open_swe_train_cap_per_family,
        open_swe_validation_cap_per_family=args.open_swe_validation_cap_per_family,
        when2call_train_cap=args.when2call_train_cap,
        when2call_validation_cap=args.when2call_validation_cap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
