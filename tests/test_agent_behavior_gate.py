import json
import tempfile
import unittest
from pathlib import Path

from agent_behavior_compare import compare_run_pairs
from agent_behavior_gate import (
    RESULT_SCHEMA,
    RUNTIME_SCHEMA,
    SUMMARY_SCHEMA,
    build_run_identity,
    detect_cycles,
    digest_value,
    file_binding,
    make_turn_event,
    score,
)


def turn(number, name, result, *, state_after=None, suffix=""):
    event = {
        "turn": number,
        "name": name,
        "arguments": {"value": suffix},
        "signature": f"{name}:{suffix}",
        "result": result,
    }
    evaluation = {} if state_after is None else {"state_after": state_after}
    return make_turn_event(
        turn=number,
        state_before=f"before-{number}",
        calls=[event],
        evaluations=[evaluation],
    )


def write_run(path, *, evaluation_contract, subject, passed, failures=()):
    contract = digest_value(evaluation_contract)
    subject_sha = digest_value(subject)
    row = {
        "schema_version": RESULT_SCHEMA,
        "case_id": "case-1",
        "slice": "loop_recovery",
        "passed": passed,
        "failures": list(failures),
        "call_count": 1,
        "usage": {"total_tokens": 10},
        "evaluation_contract_sha256": contract,
        "subject_sha256": subject_sha,
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": contract,
        "subject": subject,
        "subject_sha256": subject_sha,
        "case_count": 1,
        "results": file_binding(path),
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


class CycleDetectionTests(unittest.TestCase):
    def test_detects_immediate_period_2_and_period_3_cycles(self):
        immediate = [
            turn(1, "A", {"value": "same"}),
            turn(2, "A", {"value": "same"}),
        ]
        period_2 = [
            turn(1, "A", {"value": "a"}),
            turn(2, "B", {"value": "b"}),
            turn(3, "A", {"value": "a"}),
            turn(4, "B", {"value": "b"}),
        ]
        period_3 = [
            turn(1, "A", {"value": "a"}),
            turn(2, "B", {"value": "b"}),
            turn(3, "C", {"value": "c"}),
            turn(4, "A", {"value": "a"}),
            turn(5, "B", {"value": "b"}),
            turn(6, "C", {"value": "c"}),
        ]

        self.assertEqual({item["period"] for item in detect_cycles(immediate)}, {1})
        self.assertIn(2, {item["period"] for item in detect_cycles(period_2)})
        self.assertIn(3, {item["period"] for item in detect_cycles(period_3)})

    def test_changed_observation_or_explicit_state_allows_legitimate_revisit(self):
        changed_observation = [
            turn(1, "A", {"value": "before"}),
            turn(2, "B", {"value": "middle"}),
            turn(3, "A", {"value": "after"}),
            turn(4, "B", {"value": "done"}),
        ]
        changed_hidden_state = [
            turn(1, "A", {"ok": True}, state_after="state-1"),
            turn(2, "B", {"ok": True}, state_after="state-2"),
            turn(3, "A", {"ok": True}, state_after="state-3"),
            turn(4, "B", {"ok": True}, state_after="state-4"),
        ]

        self.assertEqual(detect_cycles(changed_observation), [])
        self.assertEqual(detect_cycles(changed_hidden_state), [])

    def test_parallel_calls_are_one_ordered_action_turn(self):
        calls = [
            {
                "turn": 1,
                "name": "A",
                "arguments": {},
                "signature": "A:{}",
                "result": {"a": 1},
            },
            {
                "turn": 1,
                "name": "B",
                "arguments": {},
                "signature": "B:{}",
                "result": {"b": 1},
            },
        ]
        event = make_turn_event(
            turn=1, state_before="initial", calls=calls, evaluations=[{}, {}]
        )

        self.assertEqual(event["action_signatures"], ["A:{}", "B:{}"])
        self.assertEqual(event["state_source"], "observation_digest")

    def test_score_exposes_cycle_period_and_malformed_failures(self):
        case = {
            "oracle": {
                "required": ["A"],
                "forbidden": [],
                "forbid_cycles_up_to": 3,
            }
        }
        events = [
            {"name": "A", "arguments": {}, "signature": "A:{}"},
            {"name": "A", "arguments": {}, "signature": "A:{}"},
        ]
        turns = [
            turn(1, "A", {"value": "same"}),
            turn(2, "A", {"value": "same"}),
        ]

        result = score(case, events, turns, ["bad arguments"])

        self.assertFalse(result["passed"])
        self.assertIn("cycle_period_1", result["failures"])
        self.assertIn("malformed:bad arguments", result["failures"])


class RuntimeIdentityTests(unittest.TestCase):
    def test_runtime_files_are_bound_and_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "server"
            model = root / "model.gguf"
            cases = root / "cases.jsonl"
            tools = root / "tools.json"
            runtime = root / "runtime.json"
            server.write_bytes(b"server-v1")
            model.write_bytes(b"model-v1")
            cases.write_text('{"case_id":"one"}\n', encoding="utf-8")
            tools.write_text("[]\n", encoding="utf-8")
            runtime.write_text(
                json.dumps(
                    {
                        "schema_version": RUNTIME_SCHEMA,
                        "comparison_contract": {
                            "server_binary": file_binding(server),
                            "launch": {"context_tokens": 32768, "device": "device-0"},
                        },
                        "subject": {
                            "served_alias": "student",
                            "model_artifact": file_binding(model),
                            "precision": "Q8_0",
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            identity = build_run_identity(
                cases=cases,
                tools=tools,
                runtime_manifest=runtime,
                endpoint="http://127.0.0.1:8080/v1/chat/completions",
                model="student",
                seed=7,
            )
            self.assertEqual(identity["subject"]["precision"], "Q8_0")

            server.write_bytes(b"server-v2")
            with self.assertRaisesRegex(ValueError, "server_binary_binding_mismatch"):
                build_run_identity(
                    cases=cases,
                    tools=tools,
                    runtime_manifest=runtime,
                    endpoint="http://127.0.0.1:8080/v1/chat/completions",
                    model="student",
                    seed=7,
                )


class ComparisonContractTests(unittest.TestCase):
    def test_subject_only_difference_can_promote_a_real_task_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            write_run(
                baseline,
                evaluation_contract={"contract": "one"},
                subject={"served_alias": "base", "precision": "Q8_0"},
                passed=False,
                failures=("cycle_period_2",),
            )
            write_run(
                candidate,
                evaluation_contract={"contract": "one"},
                subject={"served_alias": "candidate", "precision": "Q8_0"},
                passed=True,
            )

            report = compare_run_pairs(
                [baseline],
                [candidate],
                minimum_pass_delta=1,
                maximum_regressions=0,
            )

            self.assertEqual(report["verdict"], "promote_to_real_task_pilot")
            self.assertTrue(report["next_gate"]["real_task_pilot_authorized"])
            self.assertFalse(report["next_gate"]["full_registry_authorized"])

    def test_contract_or_precision_mismatch_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            write_run(
                baseline,
                evaluation_contract={"contract": "one"},
                subject={"served_alias": "base", "precision": "Q8_0"},
                passed=False,
            )
            write_run(
                candidate,
                evaluation_contract={"contract": "two"},
                subject={"served_alias": "candidate", "precision": "Q8_0"},
                passed=True,
            )
            with self.assertRaisesRegex(ValueError, "paired_evaluation_contract_mismatch"):
                compare_run_pairs(
                    [baseline],
                    [candidate],
                    minimum_pass_delta=1,
                    maximum_regressions=0,
                )

            write_run(
                candidate,
                evaluation_contract={"contract": "one"},
                subject={"served_alias": "candidate", "precision": "Q4_K_M"},
                passed=True,
            )
            with self.assertRaisesRegex(ValueError, "paired_precision_mismatch"):
                compare_run_pairs(
                    [baseline],
                    [candidate],
                    minimum_pass_delta=1,
                    maximum_regressions=0,
                )


if __name__ == "__main__":
    unittest.main()
