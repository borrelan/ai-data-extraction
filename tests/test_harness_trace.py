import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from harness_trace import HarnessTrace, TraceContractError, TraceLimits


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


class HarnessTraceTests(unittest.TestCase):
    def make_trace(self, **kwargs):
        return HarnessTrace(
            episode_id="episode-1",
            source={"agent": "test", "provider": "fixture"},
            registry_revision="registry-1",
            environment_revision="env-1",
            verifier_revision="verifier-1",
            privacy_state="heuristic",
            **kwargs,
        )

    def expose_shell(self, trace):
        trace.record_tool_registry(
            registry_revision="registry-1",
            tools=[
                {
                    "name": "shell.exec",
                    "schema": {"type": "object"},
                    "trust_class": "sandboxed",
                    "side_effect_class": "read_only",
                },
                {
                    "name": "filesystem.write",
                    "schema": {"type": "object"},
                    "trust_class": "sandboxed",
                    "side_effect_class": "write",
                },
            ],
        )

    def read_required_skill(self, trace):
        trace.record_skill_preflight(
            skill="contract-enforcement",
            skill_revision="skill-sha",
            mandatory=True,
            trigger="schema change",
            read_result="read",
            scope_decision="in_scope",
            content_sha256=digest("skill contents"),
        )

    def test_skill_and_tool_funnel_records_only_observable_events(self):
        trace = self.make_trace()
        self.read_required_skill(trace)
        self.expose_shell(trace)
        trace.record_decision(
            decision="use",
            decision_basis="capability",
            required_gate_status="passed",
            tool_name="shell.exec",
        )
        trace.record_tool_call(
            tool_name="shell.exec",
            call_id="call-1",
            arguments={"command": "true"},
            permission_decision="not_required",
        )
        observation = trace.record_tool_observation(
            call_id="call-1",
            status="success",
            output="ok",
        )
        trace.record_verification(
            verifier_revision="verifier-1",
            checks=[{"name": "exit_code", "result": "pass"}],
            result="pass",
            durable_evidence=["artifact:1"],
        )
        trace.record_terminal(status="success", evidence=["verification:pass"])

        self.assertEqual(len(trace.events), 7)
        self.assertEqual(observation["payload"]["output_sha256"], digest("ok"))
        self.assertNotIn("reasoning", trace.to_jsonl().lower())
        for ordinal, event in enumerate(trace.events):
            self.assertEqual(event["ordinal"], ordinal)
            self.assertTrue(event["event_id"].startswith("sha256:"))
        registry = trace.events[1]
        self.assertTrue(registry["payload"]["registry_sha256"].startswith("sha256:"))

    def test_required_skill_gate_blocks_tool_use_until_read(self):
        trace = self.make_trace(required_skills=("contract-enforcement",))
        self.expose_shell(trace)
        with self.assertRaises(TraceContractError):
            trace.record_decision(
                decision="use",
                decision_basis="capability",
                required_gate_status="passed",
                tool_name="shell.exec",
            )
        self.read_required_skill(trace)
        trace.record_decision(
            decision="use",
            decision_basis="capability",
            required_gate_status="passed",
            tool_name="shell.exec",
        )
        self.assertEqual(trace.events[-1]["event_type"], "decision")
        self.assertEqual(trace.events[-1]["required_skills"], ["contract-enforcement"])

    def test_mandatory_skill_failure_is_recorded_and_blocks(self):
        trace = self.make_trace()
        with self.assertRaises(TraceContractError):
            trace.record_skill_preflight(
                skill="core-principles",
                skill_revision="skill-sha",
                mandatory=True,
                trigger="implementation",
                read_result="failed",
                scope_decision="in_scope",
            )
        self.assertEqual(trace.events[0]["event_type"], "skill_preflight")
        self.assertEqual(trace.events[0]["payload"]["read_result"], "failed")

    def test_registry_rejects_duplicate_names_and_non_object_schemas(self):
        trace = self.make_trace()
        with self.assertRaises(TraceContractError):
            trace.record_tool_registry(
                registry_revision="registry-1",
                tools=[
                    {"name": "read", "schema": {}},
                    {"name": "read", "schema": {}},
                ],
            )

        with self.assertRaises(TraceContractError):
            trace.record_tool_registry(
                registry_revision="registry-1",
                tools=[{"name": "read", "schema": "not-a-schema"}],
            )

    def test_registry_digest_is_order_independent(self):
        first = self.make_trace()
        second = self.make_trace()
        tools = [
            {"name": "zeta", "schema": {"type": "object"}},
            {"name": "alpha", "inputSchema": {"type": "object"}},
        ]
        first.record_tool_registry(registry_revision="registry-1", tools=tools)
        second.record_tool_registry(registry_revision="registry-1", tools=list(reversed(tools)))

        self.assertEqual(
            first.events[0]["payload"]["registry_sha256"],
            second.events[0]["payload"]["registry_sha256"],
        )

    def test_optional_skip_requires_negative_return(self):
        trace = self.make_trace()
        with self.assertRaises(TraceContractError):
            trace.record_skill_preflight(
                skill="code-indexer-ops",
                skill_revision="skill-sha",
                mandatory=False,
                trigger="semantic lookup",
                read_result="not_observed",
                scope_decision="in_scope",
                skip_reason="model did not want to read it",
            )
        trace.record_skill_preflight(
            skill="code-indexer-ops",
            skill_revision="skill-sha",
            mandatory=False,
            trigger="semantic lookup",
            read_result="not_observed",
            scope_decision="in_scope",
            skip_reason="unavailable",
        )
        self.assertEqual(len(trace.events), 1)

    def test_side_effect_requires_permission_and_call_decision(self):
        trace = self.make_trace()
        self.expose_shell(trace)
        trace.record_decision(
            decision="use",
            decision_basis="required change",
            required_gate_status="passed",
            tool_name="filesystem.write",
        )
        with self.assertRaises(TraceContractError):
            trace.record_tool_call(
                tool_name="filesystem.write",
                call_id="call-1",
                arguments={"path": "x"},
                permission_decision="denied",
            )
        with self.assertRaises(TraceContractError):
            trace.record_tool_call(
                tool_name="shell.exec",
                call_id="call-2",
                arguments={"command": "true"},
                permission_decision="not_required",
            )

    def test_observation_is_bounded_but_hash_preserves_original_identity(self):
        trace = self.make_trace(limits=TraceLimits(max_output_bytes=16))
        self.expose_shell(trace)
        trace.record_decision(
            decision="use",
            decision_basis="capability",
            required_gate_status="passed",
            tool_name="shell.exec",
        )
        trace.record_tool_call(
            tool_name="shell.exec",
            call_id="call-1",
            arguments={},
            permission_decision="not_required",
        )
        event = trace.record_tool_observation(
            call_id="call-1",
            status="partial",
            output="0123456789abcdef0123456789",
        )
        self.assertTrue(event["payload"]["output_truncated"])
        self.assertEqual(
            event["payload"]["output_sha256"], digest("0123456789abcdef0123456789")
        )
        self.assertIn("truncated by harness", event["payload"]["output"])

    def test_success_terminal_requires_closed_calls_and_fresh_verification(self):
        trace = self.make_trace()
        self.expose_shell(trace)
        trace.record_decision(
            decision="use",
            decision_basis="capability",
            required_gate_status="passed",
            tool_name="shell.exec",
        )
        trace.record_tool_call(
            tool_name="shell.exec",
            call_id="call-1",
            arguments={},
            permission_decision="not_required",
        )
        with self.assertRaises(TraceContractError):
            trace.record_terminal(status="success", evidence=[])
        trace.record_tool_observation(call_id="call-1", status="success", output="ok")
        with self.assertRaises(TraceContractError):
            trace.record_terminal(status="success", evidence=[])
        trace.record_verification(
            verifier_revision="verifier-1",
            checks=[{"name": "exit", "result": "pass"}],
            result="pass",
            durable_evidence=["artifact:1"],
        )
        trace.record_terminal(status="success", evidence=["verification:pass"])

    def test_registry_and_observation_are_single_assignment(self):
        trace = self.make_trace()
        self.expose_shell(trace)
        with self.assertRaises(TraceContractError):
            self.expose_shell(trace)
        trace.record_decision(
            decision="use",
            decision_basis="capability",
            required_gate_status="passed",
            tool_name="shell.exec",
        )
        trace.record_tool_call(
            tool_name="shell.exec",
            call_id="call-1",
            arguments={},
            permission_decision="not_required",
        )
        trace.record_tool_observation(call_id="call-1", status="success", output="ok")
        with self.assertRaises(TraceContractError):
            trace.record_tool_observation(call_id="call-1", status="success", output="again")

    def test_non_terminal_write_and_verifier_revision_are_rejected(self):
        trace = self.make_trace()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TraceContractError):
                trace.write_jsonl(Path(directory) / "trace.jsonl")
        with self.assertRaises(TraceContractError):
            trace.record_verification(
                verifier_revision="wrong-revision",
                checks=[{"name": "x", "result": "pass"}],
                result="pass",
                durable_evidence=["artifact:1"],
            )

    def test_hidden_reasoning_and_unmatched_calls_are_not_silent(self):
        trace = self.make_trace()
        with self.assertRaises(TraceContractError):
            trace.record_decision(
                decision="use",
                decision_basis="capability",
                required_gate_status="passed",
                skill="x",
            )
            trace._append("decision", {"chain_of_thought": "private"})

        event = trace.record_tool_observation(
            call_id="missing",
            status="unmatched",
            output="orphan observation",
        )
        self.assertEqual(event["payload"]["status"], "unmatched")

    def test_loop_guard_and_atomic_write(self):
        trace = self.make_trace()
        event = trace.record_loop_guard(
            signature="same-call",
            state_hash=digest("state-1"),
            limit=2,
            action="terminate",
        )
        self.assertEqual(event["payload"]["repeated_count"], 1)
        self.assertEqual(event["payload"]["action"], "observe")
        trace.record_loop_guard(
            signature="same-call",
            state_hash=digest("state-1"),
            limit=2,
            action="terminate",
        )
        self.assertEqual(trace.events[-1]["event_type"], "terminal")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace.write_jsonl(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[0]["event_type"], "loop_guard")


if __name__ == "__main__":
    unittest.main()
