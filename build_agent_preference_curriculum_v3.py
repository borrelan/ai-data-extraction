#!/usr/bin/env python3
"""Build cycle-aware agent preferences from the proven v2 release boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_agent_preference_curriculum import (
    DECISION_SCHEMA,
    LINEAGE_SCHEMA,
    gate_prompts,
    initial_user_text,
    make_pair,
)
from build_agent_preference_curriculum_v2 import (
    MODEL_REVISION,
    REQUIRED_TRAIN_FAMILIES,
    _action_signature,
    _copy_action,
    _verify_release_constraints,
)
from build_agent_sft_curriculum import assert_model_text_clean, digest_value, release_rows
from runtime.dpo.dataset import RELEASE_SCHEMA, iter_release_rows, load_manifest, render_pair
from runtime.sft.dataset import sha256_file
from runtime.sft.filter_release import canonical_bytes, iter_bound_jsonl, write_jsonl


COVERAGE_SCHEMA = "ai-data-extraction/cycle-preference-coverage/v1"
CYCLE_FAMILIES = {
    1: "immediate_repeat_after_observed_read_only_action",
    2: "period_2_cycle_break_after_observed_read_only_path",
    3: "period_3_cycle_break_after_observed_read_only_path",
}
INHERITED_REQUIRED_TRAIN_FAMILIES = REQUIRED_TRAIN_FAMILIES - {
    "post_error_changed_recovery",
    "productive_follow_through_after_observation",
}
V3_REQUIRED_TRAIN_FAMILIES = INHERITED_REQUIRED_TRAIN_FAMILIES | set(
    CYCLE_FAMILIES.values()
)
V3_PROMOTION_COVERAGE = {
    "tool_schema": (
        "generic_tool_decision_replay",
        "required_skill_routing",
    ),
    "loop_recovery": tuple(CYCLE_FAMILIES.values()),
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
ALLOWED_INHERITED_PARTITIONS = frozenset(
    {"skill_policy", "when2call_train_pref"}
)
ALLOWED_SOURCE_PARTITIONS = ALLOWED_INHERITED_PARTITIONS | {"open_swe_cycle"}
PROGRESS_CATEGORIES = {
    "first_mutation",
    "post_failure_recovery",
    "post_mutation_verification",
}
READ_ONLY_SHELL = {
    "cat",
    "find",
    "grep",
    "head",
    "jq",
    "ls",
    "pwd",
    "rg",
    "sort",
    "stat",
    "tail",
    "test",
    "uniq",
    "wc",
}
READ_ONLY_GIT = {"branch", "diff", "log", "rev-parse", "show", "status"}
DEV_NULL_REDIRECT = re.compile(r"(?:\d*>>?|&>)\s*/dev/null\b")
SHELL_SPLIT = re.compile(r"(?:&&|\|\||[;|])")


def action_payload(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    value = function if isinstance(function, dict) else call
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {"raw": arguments}
    return {"name": str(value.get("name") or "unknown"), "arguments": arguments}


def unwrap_command(command: Any) -> str | None:
    if isinstance(command, list):
        values = [str(item) for item in command]
        if (
            len(values) >= 3
            and Path(values[0]).name in {"bash", "dash", "sh", "zsh"}
            and values[1] in {"-c", "-lc"}
        ):
            return values[2].strip()
        return " ".join(values).strip()
    return command.strip() if isinstance(command, str) else None


def shell_is_read_only(command: str) -> bool:
    command = DEV_NULL_REDIRECT.sub("", command)
    if ">" in command or "<" in command or "`" in command or "$(" in command:
        return False
    segments = [segment.strip() for segment in SHELL_SPLIT.split(command) if segment.strip()]
    if not segments:
        return False
    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        while tokens and "=" in tokens[0] and not tokens[0].startswith(("=", "-")):
            tokens.pop(0)
        if not tokens:
            continue
        executable = Path(tokens[0]).name
        if executable == "cd":
            continue
        if executable == "sed":
            if "-i" in tokens or any(token.startswith("--in-place") for token in tokens):
                return False
            continue
        if executable == "git":
            if len(tokens) < 2 or tokens[1] not in READ_ONLY_GIT:
                return False
            continue
        if executable not in READ_ONLY_SHELL:
            return False
    return True


def call_is_read_only(call: dict[str, Any]) -> bool:
    payload = action_payload(call)
    if payload["name"] != "exec_command":
        return False
    command = unwrap_command(payload["arguments"].get("cmd"))
    return bool(command) and shell_is_read_only(command)


def split_prompt(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one normalized prompt into base messages and observed action turns."""

    base: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    index = 0
    while index < len(messages) and messages[index].get("role") in {"system", "user"}:
        base.append(messages[index])
        index += 1
    if not base or not any(message.get("role") == "user" for message in base):
        raise ValueError("cycle_prompt_base_invalid")
    while index < len(messages):
        assistant = messages[index]
        calls = assistant.get("tool_calls") or []
        if assistant.get("role") != "assistant" or not calls:
            raise ValueError("cycle_prompt_action_order_invalid")
        call_ids = {str(call.get("id")) for call in calls if call.get("id") is not None}
        chunk = [assistant]
        observed_ids: set[str] = set()
        index += 1
        while index < len(messages) and messages[index].get("role") == "tool":
            observation = messages[index]
            observed_ids.add(str(observation.get("tool_call_id")))
            chunk.append(observation)
            index += 1
        turns.append(
            {
                "action": assistant,
                "chunk": chunk,
                "observation_complete": bool(call_ids) and call_ids <= observed_ids,
                "read_only": all(call_is_read_only(call) for call in calls),
            }
        )
    return base, turns


def prompt_state(pair: dict[str, Any]) -> str:
    return digest_value({"prompt": pair["prompt"], "tools": pair["tools"]})


def nearest_prior_action_distance(
    prompt: list[dict[str, Any]], action: dict[str, Any]
) -> int | None:
    """Return distance in assistant action turns, not raw message positions."""

    target = _action_signature(action)
    action_signatures = [
        _action_signature(message)
        for message in prompt
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("tool_calls")
    ]
    return next(
        (
            len(action_signatures) - index
            for index in range(len(action_signatures) - 1, -1, -1)
            if action_signatures[index] == target
        ),
        None,
    )


def validate_coverage_evidence(
    coverage: dict[str, Any], *, source_release: Path, evaluation_cases: Path
) -> dict[str, Any]:
    """Bind the audit to current inputs without accepting its scheduling decision."""

    if coverage.get("schema_version") != COVERAGE_SCHEMA:
        raise ValueError("cycle_coverage_schema_mismatch")
    if coverage.get("status") != "completed":
        raise ValueError("cycle_coverage_incomplete")
    sources = coverage.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("cycle_coverage_sources_invalid")
    open_swe = sources.get("open_swe")
    if not isinstance(open_swe, dict):
        raise ValueError("cycle_coverage_open_swe_source_invalid")
    source_manifest_sha = sha256_file(source_release / "manifest.json")
    if open_swe.get("manifest_sha256") != source_manifest_sha:
        raise ValueError("cycle_coverage_source_manifest_mismatch")
    evaluation_sha = sha256_file(evaluation_cases)
    evaluation = coverage.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("cycle_coverage_evaluation_invalid")
    if evaluation.get("cases_sha256") != evaluation_sha:
        raise ValueError("cycle_coverage_evaluation_cases_mismatch")
    return {
        "schema_version": COVERAGE_SCHEMA,
        "status": "bound_evidence_only",
        "source_manifest_sha256": source_manifest_sha,
        "evaluation_cases_sha256": evaluation_sha,
        "scheduling_decision_ignored": True,
    }


def verify_exact_token_contract(
    pairs: list[dict[str, Any]], tokenizer: Any, *, max_sequence_tokens: int
) -> dict[str, Any]:
    maxima: Counter[str] = Counter()
    for pair in pairs:
        _, token_counts, _ = render_pair(
            pair, tokenizer, max_length=max_sequence_tokens
        )
        for key, value in token_counts.items():
            maxima[key] = max(maxima[key], value)
    return {
        "status": "passed",
        "pairs": len(pairs),
        "max_sequence_tokens": max_sequence_tokens,
        "maxima": dict(sorted(maxima.items())),
    }


def fit_cycle_pair(
    *,
    split: str,
    row: dict[str, Any],
    source: dict[str, Any],
    source_manifest_sha256: str,
    distance: int,
    tokenizer: Any,
    max_sequence_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = row["messages"][:-1]
    chosen = row["messages"][-1]
    base, turns = split_prompt(prompt)
    if len(turns) < distance:
        raise ValueError("insufficient_prior_actions")
    cycle_path = turns[-distance:]
    if not all(turn["observation_complete"] for turn in cycle_path):
        raise ValueError("cycle_path_observation_incomplete")
    if not all(turn["read_only"] for turn in cycle_path):
        raise ValueError("cycle_path_contains_mutation")
    rejected_source = cycle_path[0]["action"]
    if _action_signature(rejected_source) == _action_signature(chosen):
        raise ValueError("chosen_revisits_cycle_action")
    nearest_distance = nearest_prior_action_distance(prompt, rejected_source)
    if nearest_distance != distance:
        raise ValueError("rejected_action_nearest_distance_mismatch")
    progress_categories = sorted(
        set(source.get("selection_categories") or []) & PROGRESS_CATEGORIES
    )
    if not progress_categories:
        raise ValueError("chosen_action_not_progress_backed")
    if source.get("outcome") != "resolved":
        raise ValueError("source_outcome_not_resolved")
    family = CYCLE_FAMILIES[distance]
    final_error: str | None = None
    for retained_turns in range(len(turns), distance - 1, -1):
        bounded_prompt = copy.deepcopy(base)
        for turn in turns[-retained_turns:]:
            bounded_prompt.extend(copy.deepcopy(turn["chunk"]))
        pair = make_pair(
            split=split,
            lane="state_transition_preference",
            prompt=bounded_prompt,
            chosen=chosen,
            rejected=_copy_action(rejected_source),
            tools=row["tools"],
            identity={
                "curriculum": "v3",
                "source_manifest_sha256": source_manifest_sha256,
                "source_example_id": row["example_id"],
                "cycle_distance": distance,
                "retained_action_turns": retained_turns,
                "preference_family": family,
            },
        )
        try:
            _, token_counts, _ = render_pair(
                pair, tokenizer, max_length=max_sequence_tokens
            )
        except ValueError as exc:
            if f"exceeds {max_sequence_tokens} tokens" not in str(exc):
                raise
            final_error = str(exc)
            continue
        return pair, {
            "schema_version": LINEAGE_SCHEMA,
            "pair_id": pair["pair_id"],
            "parent_id": source["parent_id"],
            "source_partition": "open_swe_cycle",
            "source_manifest_sha256": source_manifest_sha256,
            "source_example_id": row["example_id"],
            "source_action_ordinal": source["action_ordinal"],
            "source_selection_categories": list(source.get("selection_categories") or []),
            "chosen_progress_categories": progress_categories,
            "source_outcome": source.get("outcome"),
            "preference_family": family,
            "preference_basis": "older_consumed_action_rejected_after_fully_observed_read_only_path",
            "label_strength": "source_executed_progress_heuristic_not_locally_replayed",
            "cycle_distance": distance,
            "rejected_nearest_prior_action_distance": nearest_distance,
            "source_action_turns": len(turns),
            "retained_action_turns": retained_turns,
            "context_reduced_at_action_boundary": retained_turns < len(turns),
            "tokens": token_counts,
        }
    raise ValueError(final_error or "cycle_pair_does_not_fit")


def build_cycle_candidates(
    *,
    source_release: Path,
    excluded_prompts: set[str],
    blocked_text_patterns: tuple[re.Pattern[str], ...],
    tokenizer: Any,
    max_sequence_tokens: int,
) -> tuple[
    dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest, rows = release_rows(source_release)
    manifest_sha = sha256_file(source_release / "manifest.json")
    candidates: defaultdict[
        tuple[str, int],
        list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for split, row, source in rows:
        if row["lane"] != "verified_open_swe_action":
            continue
        categories = set(source.get("selection_categories") or [])
        for distance in sorted(CYCLE_FAMILIES):
            decision = {
                "schema_version": DECISION_SCHEMA,
                "source_partition": "open_swe_cycle",
                "source_example_id": row["example_id"],
                "cycle_distance": distance,
            }
            try:
                if initial_user_text(row["messages"][:-1]) in excluded_prompts:
                    raise ValueError("evaluation_prompt_overlap")
                if not categories & PROGRESS_CATEGORIES:
                    raise ValueError("chosen_action_not_progress_backed")
                if source.get("outcome") != "resolved":
                    raise ValueError("source_outcome_not_resolved")
                pair, lineage = fit_cycle_pair(
                    split=split,
                    row=row,
                    source=source,
                    source_manifest_sha256=manifest_sha,
                    distance=distance,
                    tokenizer=tokenizer,
                    max_sequence_tokens=max_sequence_tokens,
                )
                assert_model_text_clean(pair, blocked_text_patterns)
                candidates[(split, distance)].append((pair, lineage, decision))
            except (KeyError, TypeError, ValueError) as exc:
                reason = str(exc)
                exclusions[f"{split}:distance_{distance}:{reason}"] += 1
                decision.update({"decision": "excluded", "reason": reason})
                decisions.append(decision)
    for items in candidates.values():
        items.sort(key=lambda item: item[0]["pair_id"])
    return candidates, decisions, {
        "path": str(source_release),
        "source_manifest_sha256": manifest_sha,
        "source_status": manifest.get("status"),
        "available": {
            f"{split}:distance_{distance}": len(items)
            for (split, distance), items in sorted(candidates.items())
        },
        "excluded_reasons": dict(sorted(exclusions.items())),
    }


def load_base_release(
    source_release: Path,
    *,
    excluded_prompts: set[str],
    blocked_text_patterns: tuple[re.Pattern[str], ...],
    max_sequence_tokens: int,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    dict[str, Any],
]:
    manifest = load_manifest(source_release)
    if manifest.get("model", {}).get("max_sequence_tokens") != max_sequence_tokens:
        raise ValueError("base_preference_sequence_contract_mismatch")
    manifest_sha = sha256_file(source_release / "manifest.json")
    lineage = {
        row["pair_id"]: row
        for row in iter_bound_jsonl(source_release, manifest, "lineage.jsonl")
    }
    inherited: list[tuple[dict[str, Any], dict[str, Any]]] = []
    discarded_transitions = 0
    for split, _, pair in iter_release_rows(source_release, manifest):
        item = lineage.get(pair["pair_id"])
        if item is None:
            raise ValueError("base_preference_lineage_missing")
        if initial_user_text(pair["prompt"]) in excluded_prompts:
            raise ValueError("base_preference_evaluation_overlap")
        assert_model_text_clean(pair, blocked_text_patterns)
        copied_lineage = copy.deepcopy(item)
        copied_lineage["inherited_from_manifest_sha256"] = manifest_sha
        if item["source_partition"] == "open_swe_transition":
            discarded_transitions += 1
            continue
        if item["source_partition"] not in ALLOWED_INHERITED_PARTITIONS:
            raise ValueError("base_preference_source_partition_unsupported")
        inherited.append((copy.deepcopy(pair), copied_lineage))
    return inherited, {
        "path": str(source_release),
        "manifest_sha256": manifest_sha,
        "status": manifest.get("status"),
        "inherited_non_transition": len(inherited),
        "discarded_rejected_control_transitions": discarded_transitions,
    }


def select_cycle_pairs(
    candidates: dict[
        tuple[str, int],
        list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    ],
    *,
    caps: dict[tuple[str, int], int],
    used_prompt_states: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    used_parents: defaultdict[str, set[str]] = defaultdict(set)
    selected: Counter[str] = Counter()
    for split in ("train", "validation"):
        for distance in (3, 2, 1):
            items = sorted(
                candidates.get((split, distance), []),
                key=lambda item: item[0]["pair_id"],
            )
            cap = caps[(split, distance)]
            for pair, item, decision in items:
                state = prompt_state(pair)
                parent = item["parent_id"]
                if selected[f"{split}:distance_{distance}"] >= cap:
                    reason = "cycle_split_distance_cap"
                elif parent in used_parents[split]:
                    reason = "cycle_parent_already_selected"
                elif state in used_prompt_states:
                    reason = "duplicate_prompt_tool_state"
                else:
                    pairs.append(pair)
                    lineage.append(item)
                    used_parents[split].add(parent)
                    used_prompt_states.add(state)
                    selected[f"{split}:distance_{distance}"] += 1
                    decision.update(
                        {
                            "decision": "selected",
                            "pair_id": pair["pair_id"],
                            "reason": CYCLE_FAMILIES[distance],
                        }
                    )
                    decisions.append(decision)
                    continue
                decision.update({"decision": "excluded", "reason": reason})
                decisions.append(decision)
    return pairs, lineage, decisions, {
        "caps": {
            f"{split}:distance_{distance}": cap
            for (split, distance), cap in sorted(caps.items())
        },
        "selected": dict(sorted(selected.items())),
        "unique_parents": {split: len(parents) for split, parents in sorted(used_parents.items())},
    }


def verify_v3_constraints(
    pairs: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    excluded_prompts: set[str],
    *,
    cycle_caps: dict[tuple[str, int], int],
) -> dict[str, Any]:
    result = _verify_release_constraints(
        pairs,
        lineage,
        excluded_prompts,
        required_train_families=V3_REQUIRED_TRAIN_FAMILIES,
        promotion_coverage=V3_PROMOTION_COVERAGE,
    )
    pair_by_id = {row["pair_id"]: row for row in pairs}
    cycle_counts: Counter[str] = Counter()
    open_swe: Counter[str] = Counter()
    total: Counter[str] = Counter()
    open_swe_source_examples: set[str] = set()
    for item in lineage:
        pair = pair_by_id[item["pair_id"]]
        split = pair["split"]
        total[split] += 1
        source_partition = item.get("source_partition")
        if source_partition not in ALLOWED_SOURCE_PARTITIONS:
            raise ValueError("trainer_source_partition_unsupported")
        if source_partition == "open_swe_cycle":
            open_swe[split] += 1
            source_example = item.get("source_example_id")
            if not isinstance(source_example, str) or not source_example:
                raise ValueError("open_swe_source_example_missing")
            if source_example in open_swe_source_examples:
                raise ValueError("open_swe_source_example_reused")
            open_swe_source_examples.add(source_example)
            distance = int(item["cycle_distance"])
            if distance not in CYCLE_FAMILIES:
                raise ValueError("cycle_distance_unsupported")
            if item.get("preference_family") != CYCLE_FAMILIES[distance]:
                raise ValueError("cycle_preference_family_mismatch")
            if item["retained_action_turns"] < distance:
                raise ValueError("cycle_context_dropped_required_path")
            source_categories = item.get("source_selection_categories")
            chosen_categories = item.get("chosen_progress_categories")
            if not isinstance(source_categories, list) or not isinstance(
                chosen_categories, list
            ):
                raise ValueError("cycle_chosen_progress_evidence_missing")
            if not chosen_categories or not set(chosen_categories) <= (
                set(source_categories) & PROGRESS_CATEGORIES
            ):
                raise ValueError("cycle_chosen_progress_evidence_invalid")
            if item.get("source_outcome") != "resolved":
                raise ValueError("cycle_source_outcome_not_resolved")
            if item.get("label_strength") != (
                "source_executed_progress_heuristic_not_locally_replayed"
            ):
                raise ValueError("cycle_label_strength_invalid")
            if item.get("preference_basis") != (
                "older_consumed_action_rejected_after_fully_observed_read_only_path"
            ):
                raise ValueError("cycle_preference_basis_invalid")
            rejected = pair.get("rejected")
            if not isinstance(rejected, list) or len(rejected) != 1:
                raise ValueError("cycle_rejected_action_invalid")
            nearest_distance = nearest_prior_action_distance(
                pair["prompt"], rejected[0]
            )
            if (
                nearest_distance != distance
                or item.get("rejected_nearest_prior_action_distance") != distance
            ):
                raise ValueError("cycle_rejected_action_distance_mismatch")
            chosen = pair.get("chosen")
            if (
                not isinstance(chosen, list)
                or len(chosen) != 1
                or _action_signature(chosen[0]) == _action_signature(rejected[0])
            ):
                raise ValueError("cycle_chosen_action_invalid")
            cycle_counts[f"{split}:distance_{distance}"] += 1
    for (split, distance), cap in sorted(cycle_caps.items()):
        if cycle_counts[f"{split}:distance_{distance}"] != cap:
            raise ValueError(f"{split}_cycle_distance_{distance}_coverage_mismatch")
    for split in total:
        if open_swe[split] * 2 > total[split]:
            raise ValueError(f"aggregate_open_swe_exceeds_half:{split}")
    train_families = {
        item["preference_family"]
        for item in lineage
        if pair_by_id[item["pair_id"]]["split"] == "train"
    }
    required = V3_REQUIRED_TRAIN_FAMILIES
    missing = sorted(required - train_families)
    if missing:
        raise ValueError(f"v3_required_train_family_missing:{','.join(missing)}")
    result["cycle_families"] = dict(sorted(cycle_counts.items()))
    result["aggregate_open_swe"] = {
        split: {"rows": open_swe[split], "total": total[split]}
        for split in sorted(total)
    }
    result["required_train_families_v3"] = sorted(required)
    return result


def build_curriculum_v3(
    *,
    base_preference_release: Path,
    sft_release: Path,
    coverage_report: Path,
    evaluation_cases: Path,
    model_dir: Path,
    output_dir: Path,
    blocked_text_patterns: tuple[str, ...] = (),
    max_sequence_tokens: int = 6144,
    train_cycle_cap_per_distance: int = 20,
    validation_period_1_cap: int = 8,
    validation_period_2_cap: int = 8,
    validation_period_3_cap: int = 4,
) -> dict[str, Any]:
    if max_sequence_tokens <= 0:
        raise ValueError("max_sequence_tokens must be positive")
    if train_cycle_cap_per_distance < 20:
        raise ValueError("train_cycle_cap_per_distance must be at least 20")
    if min(
        validation_period_1_cap,
        validation_period_2_cap,
        validation_period_3_cap,
    ) <= 0:
        raise ValueError("validation cycle caps must be positive")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    coverage = json.loads(coverage_report.read_text())
    coverage_binding = validate_coverage_evidence(
        coverage,
        source_release=sft_release.resolve(),
        evaluation_cases=evaluation_cases.resolve(),
    )
    compiled_blocked = tuple(re.compile(pattern, re.I) for pattern in blocked_text_patterns)
    excluded_prompts = gate_prompts(evaluation_cases.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    inherited, base_stats = load_base_release(
        base_preference_release.resolve(),
        excluded_prompts=excluded_prompts,
        blocked_text_patterns=compiled_blocked,
        max_sequence_tokens=max_sequence_tokens,
    )
    pairs = [pair for pair, _ in inherited]
    lineage = [item for _, item in inherited]
    used_prompt_states = {prompt_state(pair) for pair in pairs}
    cycle_candidates, cycle_decisions, cycle_source_stats = build_cycle_candidates(
        source_release=sft_release.resolve(),
        excluded_prompts=excluded_prompts,
        blocked_text_patterns=compiled_blocked,
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )
    cycle_caps = {
        ("train", 1): train_cycle_cap_per_distance,
        ("train", 2): train_cycle_cap_per_distance,
        ("train", 3): train_cycle_cap_per_distance,
        ("validation", 1): validation_period_1_cap,
        ("validation", 2): validation_period_2_cap,
        ("validation", 3): validation_period_3_cap,
    }
    cycles = select_cycle_pairs(
        cycle_candidates,
        caps=cycle_caps,
        used_prompt_states=used_prompt_states,
    )
    pairs.extend(cycles[0])
    lineage.extend(cycles[1])
    decisions = cycle_decisions + cycles[2]
    constraints = verify_v3_constraints(
        pairs, lineage, excluded_prompts, cycle_caps=cycle_caps
    )
    exact_token_contract = verify_exact_token_contract(
        pairs, tokenizer, max_sequence_tokens=max_sequence_tokens
    )
    pairs.sort(key=lambda row: (row["split"], row["pair_id"]))
    lineage.sort(key=lambda row: row["pair_id"])
    decisions.sort(
        key=lambda row: (
            row["source_partition"],
            row.get("source_example_id", ""),
            int(row.get("cycle_distance", 0)),
            row.get("source_pair_id", ""),
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
            "status": "ready_for_independent_exact_tokenizer_preflight",
            "purpose": "cycle_aware_long_horizon_agent_preference_optimization",
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
                "max_sequence_tokens": max_sequence_tokens,
            },
            "sources": {
                "base_preference_release": base_stats,
                "open_swe_cycle": {**cycle_source_stats, **cycles[3]},
                "coverage_report": {
                    "path": str(coverage_report.resolve()),
                    "sha256": sha256_file(coverage_report),
                    "schema_version": coverage.get("schema_version"),
                    "binding": coverage_binding,
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
                "cycle_parent_cap_across_distances": 1,
                "cycle_distances": sorted(CYCLE_FAMILIES),
                "cycle_context_reduction": "oldest_complete_action_observation_turns_only",
                "minimum_retained_action_turns": "cycle_distance",
                "cycle_intervening_path": "fully_observed_and_read_only",
                "chosen_progress": "resolved_source_trajectory_and_selected_progress_category",
                "aggregate_open_swe_max_fraction": 0.5,
                "blocked_text_pattern_sha256": sorted(
                    hashlib.sha256(pattern.encode()).hexdigest()
                    for pattern in blocked_text_patterns
                ),
                "exact_tokenizer_filter": exact_token_contract,
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
                "inherited_labels": "non_transition_pairs_only_from_rejected_v2.1_control",
                "cycle_labels": "source_executed_progress_heuristic_not_locally_replayed",
                "rewards": "not_invented",
                "tokenizer_preflight": "builder_passed_independent_runtime_preflight_pending",
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
    parser.add_argument("--base-preference-release", type=Path, required=True)
    parser.add_argument("--sft-release", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--evaluation-cases", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocked-text-pattern", action="append", default=[])
    parser.add_argument("--max-sequence-tokens", type=int, default=6144)
    args = parser.parse_args()
    result = build_curriculum_v3(
        base_preference_release=args.base_preference_release,
        sft_release=args.sft_release,
        coverage_report=args.coverage_report,
        evaluation_cases=args.evaluation_cases,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        blocked_text_patterns=tuple(args.blocked_text_pattern),
        max_sequence_tokens=args.max_sequence_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
