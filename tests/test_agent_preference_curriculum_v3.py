import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_agent_preference_curriculum_v3 as curriculum
from build_agent_preference_curriculum import LINEAGE_SCHEMA, make_pair


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a source patch.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
]


def action(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": copy.deepcopy(arguments)},
            }
        ],
    }


def observed_action(command, call_id):
    call = action("exec_command", {"cmd": command}, call_id)
    observation = {
        "role": "tool",
        "content": json.dumps({"returncode": 0, "stdout": call_id}),
        "tool_call_id": call_id,
    }
    return call, observation


def cycle_source_row(distance, *, suffix="source", mutation_at=None):
    commands = ("rg needle src", "git status --short", "sed -n '1,20p' src/a.py")
    messages = [{"role": "user", "content": f"repair issue {suffix} distance {distance}"}]
    prior_actions = []
    for index in range(distance):
        call_id = f"call-{suffix}-{index}"
        if mutation_at == index:
            prior = action("apply_patch", {"patch": f"mutation-{suffix}"}, call_id)
        else:
            prior, _ = observed_action(commands[index], call_id)
        observation = {
            "role": "tool",
            "content": json.dumps({"returncode": 0, "stdout": call_id}),
            "tool_call_id": call_id,
        }
        prior_actions.append(prior)
        messages.extend((prior, observation))
    messages.append(
        action("apply_patch", {"patch": f"progress-{suffix}"}, f"chosen-{suffix}")
    )
    return {
        "example_id": f"example-{suffix}",
        "lane": "verified_open_swe_action",
        "messages": messages,
        "tools": TOOLS,
    }, {
        "parent_id": f"parent-{suffix}",
        "action_ordinal": distance,
        "selection_categories": ["first_mutation"],
        "outcome": "resolved",
    }, prior_actions


def policy_pair(split, family, partition, suffix):
    pair = make_pair(
        split=split,
        lane=(
            "tool_decision_preference"
            if partition == "when2call_train_pref"
            else "skill_policy_preference"
        ),
        prompt=[{"role": "user", "content": f"policy request {suffix}"}],
        chosen={"role": "assistant", "content": f"correct action {suffix}"},
        rejected={"role": "assistant", "content": f"wrong action {suffix}"},
        tools=None,
        identity={"fixture": suffix, "family": family},
    )
    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "pair_id": pair["pair_id"],
        "parent_id": f"parent-{suffix}",
        "source_partition": partition,
        "preference_family": family,
    }
    return pair, lineage


def cycle_pair(split, distance, suffix):
    row, source, prior_actions = cycle_source_row(distance, suffix=suffix)
    prompt = row["messages"][:-1]
    rejected = copy.deepcopy(prior_actions[0])
    rejected["tool_calls"][0]["id"] = f"rejected-{suffix}"
    pair = make_pair(
        split=split,
        lane="state_transition_preference",
        prompt=prompt,
        chosen=row["messages"][-1],
        rejected=rejected,
        tools=TOOLS,
        identity={"fixture": suffix, "distance": distance},
    )
    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "pair_id": pair["pair_id"],
        "parent_id": source["parent_id"],
        "source_partition": "open_swe_cycle",
        "source_manifest_sha256": "a" * 64,
        "source_example_id": row["example_id"],
        "source_action_ordinal": distance,
        "source_selection_categories": ["first_mutation"],
        "chosen_progress_categories": ["first_mutation"],
        "source_outcome": "resolved",
        "preference_family": curriculum.CYCLE_FAMILIES[distance],
        "preference_basis": (
            "older_consumed_action_rejected_after_fully_observed_read_only_path"
        ),
        "label_strength": "source_executed_progress_heuristic_not_locally_replayed",
        "cycle_distance": distance,
        "rejected_nearest_prior_action_distance": distance,
        "source_action_turns": distance,
        "retained_action_turns": distance,
    }
    return pair, lineage


def valid_release_fixture():
    pairs = []
    lineage = []

    train_skill_families = [
        "permission_read_only_boundary",
        "premature_stop_before_required_work",
        "required_skill_routing",
        "verified_finish_over_unnecessary_continuation",
        "unnecessary_skill_overhead",
    ]
    for index, family in enumerate(train_skill_families):
        pair, item = policy_pair("train", family, "skill_policy", f"train-skill-{index}")
        pairs.append(pair)
        lineage.append(item)
    for index in range(2):
        pair, item = policy_pair(
            "train",
            "generic_tool_decision_replay",
            "when2call_train_pref",
            f"train-generic-{index}",
        )
        pairs.append(pair)
        lineage.append(item)
    for distance in curriculum.CYCLE_FAMILIES:
        pair, item = cycle_pair("train", distance, f"train-cycle-{distance}")
        pairs.append(pair)
        lineage.append(item)

    for index, family in enumerate(
        ("permission_read_only_boundary", "required_skill_routing")
    ):
        pair, item = policy_pair(
            "validation", family, "skill_policy", f"validation-skill-{index}"
        )
        pairs.append(pair)
        lineage.append(item)
    pair, item = policy_pair(
        "validation",
        "generic_tool_decision_replay",
        "when2call_train_pref",
        "validation-generic",
    )
    pairs.append(pair)
    lineage.append(item)
    for distance in curriculum.CYCLE_FAMILIES:
        pair, item = cycle_pair(
            "validation", distance, f"validation-cycle-{distance}"
        )
        pairs.append(pair)
        lineage.append(item)
    return pairs, lineage


class CyclePairContractTests(unittest.TestCase):
    def test_nearest_prior_distance_is_counted_in_action_turns(self):
        row, _, actions = cycle_source_row(3, suffix="distance")
        prompt = row["messages"][:-1]

        self.assertEqual(curriculum.nearest_prior_action_distance(prompt, actions[0]), 3)
        self.assertEqual(curriculum.nearest_prior_action_distance(prompt, actions[1]), 2)
        self.assertEqual(curriculum.nearest_prior_action_distance(prompt, actions[2]), 1)

    def test_fit_cycle_pair_records_progress_and_exact_distance(self):
        row, source, _ = cycle_source_row(2, suffix="fit")
        rendered = ({}, {"prompt": 100, "chosen_sequence": 120}, {})

        with patch.object(curriculum, "render_pair", return_value=rendered):
            pair, lineage = curriculum.fit_cycle_pair(
                split="train",
                row=row,
                source=source,
                source_manifest_sha256="b" * 64,
                distance=2,
                tokenizer=object(),
                max_sequence_tokens=6144,
            )

        self.assertEqual(lineage["cycle_distance"], 2)
        self.assertEqual(lineage["rejected_nearest_prior_action_distance"], 2)
        self.assertEqual(lineage["chosen_progress_categories"], ["first_mutation"])
        self.assertNotEqual(pair["chosen"], pair["rejected"])

    def test_fit_cycle_pair_rejects_nearer_duplicate(self):
        row, source, actions = cycle_source_row(3, suffix="nearer")
        duplicate = copy.deepcopy(actions[0])
        duplicate["tool_calls"][0]["id"] = "call-nearer-2"
        row["messages"][-3] = duplicate

        with self.assertRaisesRegex(
            ValueError, "rejected_action_nearest_distance_mismatch"
        ):
            curriculum.fit_cycle_pair(
                split="train",
                row=row,
                source=source,
                source_manifest_sha256="b" * 64,
                distance=3,
                tokenizer=object(),
                max_sequence_tokens=6144,
            )

    def test_fit_cycle_pair_rejects_mutation_in_observed_path(self):
        row, source, _ = cycle_source_row(2, suffix="mutation", mutation_at=1)

        with self.assertRaisesRegex(ValueError, "cycle_path_contains_mutation"):
            curriculum.fit_cycle_pair(
                split="train",
                row=row,
                source=source,
                source_manifest_sha256="b" * 64,
                distance=2,
                tokenizer=object(),
                max_sequence_tokens=6144,
            )


class EvidenceBindingTests(unittest.TestCase):
    def test_coverage_report_is_digest_bound_but_cannot_authorize_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "source"
            release.mkdir()
            manifest = release / "manifest.json"
            manifest.write_text('{"status":"ready"}\n', encoding="utf-8")
            cases = root / "cases.jsonl"
            cases.write_text('{"prompt":"holdout"}\n', encoding="utf-8")
            coverage = {
                "schema_version": curriculum.COVERAGE_SCHEMA,
                "status": "completed",
                "sources": {
                    "open_swe": {
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
                    }
                },
                "evaluation": {
                    "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest()
                },
                "decision": {"sufficient": False},
            }

            result = curriculum.validate_coverage_evidence(
                coverage, source_release=release, evaluation_cases=cases
            )
            self.assertTrue(result["scheduling_decision_ignored"])

            manifest.write_text('{"status":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "cycle_coverage_source_manifest_mismatch"
            ):
                curriculum.validate_coverage_evidence(
                    coverage, source_release=release, evaluation_cases=cases
                )

    def test_exact_token_contract_renders_every_final_pair_and_fails_overflow(self):
        pairs = [{"pair_id": "one"}, {"pair_id": "two"}, {"pair_id": "three"}]

        def render(pair, _tokenizer, *, max_length):
            self.assertEqual(max_length, 6144)
            if pair["pair_id"] == "three":
                raise ValueError("chosen branch exceeds 6144 tokens: 6145")
            return {}, {"chosen_sequence": 100, "rejected_sequence": 90}, {}

        with patch.object(curriculum, "render_pair", side_effect=render) as mocked:
            with self.assertRaisesRegex(ValueError, "exceeds 6144 tokens"):
                curriculum.verify_exact_token_contract(
                    pairs, object(), max_sequence_tokens=6144
                )

        self.assertEqual(mocked.call_count, 3)


class ReleaseConstraintTests(unittest.TestCase):
    def test_valid_cycle_aware_release_passes_all_static_constraints(self):
        pairs, lineage = valid_release_fixture()
        caps = {(split, distance): 1 for split in ("train", "validation") for distance in (1, 2, 3)}

        result = curriculum.verify_v3_constraints(
            pairs, lineage, set(), cycle_caps=caps
        )

        self.assertEqual(result["aggregate_open_swe"]["train"], {"rows": 3, "total": 10})
        self.assertEqual(result["aggregate_open_swe"]["validation"], {"rows": 3, "total": 6})

    def test_rejects_duplicate_prompt_parent_split_and_historical_partition(self):
        caps = {(split, distance): 1 for split in ("train", "validation") for distance in (1, 2, 3)}

        pairs, lineage = valid_release_fixture()
        pairs[1]["prompt"] = copy.deepcopy(pairs[0]["prompt"])
        pairs[1]["tools"] = copy.deepcopy(pairs[0]["tools"])
        with self.assertRaisesRegex(ValueError, "duplicate_prompt_tool_state"):
            curriculum.verify_v3_constraints(pairs, lineage, set(), cycle_caps=caps)

        pairs, lineage = valid_release_fixture()
        validation = next(
            item
            for item in lineage
            if next(pair for pair in pairs if pair["pair_id"] == item["pair_id"])["split"]
            == "validation"
        )
        validation["parent_id"] = lineage[0]["parent_id"]
        with self.assertRaisesRegex(ValueError, "preference_parent_split_overlap"):
            curriculum.verify_v3_constraints(pairs, lineage, set(), cycle_caps=caps)

        pairs, lineage = valid_release_fixture()
        generic_validation = next(
            item
            for item in lineage
            if item["source_partition"] == "when2call_train_pref"
            and next(pair for pair in pairs if pair["pair_id"] == item["pair_id"])["split"]
            == "validation"
        )
        generic_validation["source_partition"] = "action_windows:candidate"
        with self.assertRaisesRegex(ValueError, "trainer_source_partition_unsupported"):
            curriculum.verify_v3_constraints(pairs, lineage, set(), cycle_caps=caps)

    def test_rejects_source_imbalance_eval_overlap_and_distance_mismatch(self):
        caps = {(split, distance): 1 for split in ("train", "validation") for distance in (1, 2, 3)}

        pairs, lineage = valid_release_fixture()
        for item in lineage:
            pair = next(pair for pair in pairs if pair["pair_id"] == item["pair_id"])
            if pair["split"] == "train" and item["source_partition"] == "when2call_train_pref":
                item["source_partition"] = "skill_policy"
        with self.assertRaisesRegex(ValueError, "source_partition_exceeds_half:train"):
            curriculum.verify_v3_constraints(pairs, lineage, set(), cycle_caps=caps)

        pairs, lineage = valid_release_fixture()
        excluded = {curriculum.initial_user_text(pairs[0]["prompt"])}
        with self.assertRaisesRegex(ValueError, "evaluation_prompt_overlap"):
            curriculum.verify_v3_constraints(pairs, lineage, excluded, cycle_caps=caps)

        pairs, lineage = valid_release_fixture()
        cycle = next(item for item in lineage if item.get("cycle_distance") == 2)
        cycle["rejected_nearest_prior_action_distance"] = 1
        with self.assertRaisesRegex(ValueError, "cycle_rejected_action_distance_mismatch"):
            curriculum.verify_v3_constraints(pairs, lineage, set(), cycle_caps=caps)

    def test_cycle_candidate_builder_excludes_blocked_and_evaluation_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_release = Path(temporary)
            (source_release / "manifest.json").write_text("{}\n", encoding="utf-8")
            row, source, _ = cycle_source_row(1, suffix="blocked-token")
            rendered = ({}, {"prompt": 100, "chosen_sequence": 120}, {})
            with patch.object(
                curriculum,
                "release_rows",
                return_value=({"status": "ready"}, [("train", row, source)]),
            ), patch.object(curriculum, "render_pair", return_value=rendered):
                candidates, decisions, _ = curriculum.build_cycle_candidates(
                    source_release=source_release,
                    excluded_prompts=set(),
                    blocked_text_patterns=(re.compile("blocked-token"),),
                    tokenizer=object(),
                    max_sequence_tokens=6144,
                )
            self.assertFalse(candidates)
            self.assertIn("blocked_model_text", {item["reason"] for item in decisions})

            excluded_prompt = curriculum.initial_user_text(row["messages"][:-1])
            with patch.object(
                curriculum,
                "release_rows",
                return_value=({"status": "ready"}, [("train", row, source)]),
            ):
                candidates, decisions, _ = curriculum.build_cycle_candidates(
                    source_release=source_release,
                    excluded_prompts={excluded_prompt},
                    blocked_text_patterns=(),
                    tokenizer=object(),
                    max_sequence_tokens=6144,
                )
            self.assertFalse(candidates)
            self.assertEqual(
                {item["reason"] for item in decisions}, {"evaluation_prompt_overlap"}
            )


if __name__ == "__main__":
    unittest.main()
