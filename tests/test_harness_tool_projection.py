import json
import tempfile
import unittest
from pathlib import Path

from build_training_data import SCHEMA_VERSION
from harness_trace import HarnessTrace, TraceContractError
from harness_tool_projection import HarnessProjectionError, project_harness_tool_sft
from trainer_export import canonical_json_bytes


def _digest(value: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _episode(call_id: str = "call-1") -> dict[str, object]:
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": _digest("example"),
        "dataset": "tool_trace",
        "split": "train",
        "tags": ["tool:search"],
        "messages": [
            {"role": "user", "content": "Find the owner."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "code-indexer.search",
                            "arguments": {"query": "owner"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "one match", "tool_call_id": call_id},
            {"role": "assistant", "content": "The owner is recorded."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "code-indexer.search",
                    "description": "Search bounded code evidence.",
                    "parameters": parameters,
                },
            }
        ],
        "quality": {"status": "review", "has_tools": True},
        "privacy": {"eligible_for_training": False},
        "lineage": {"parent_record_sha256": _digest("parent")},
    }


def _write_trace(path: Path, call_id: str = "call-1") -> None:
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    trace = HarnessTrace(
        episode_id="episode-1",
        source={"fixture": "projection"},
        registry_revision="registry-1",
        skill_revision="skills-1",
        environment_revision="environment-1",
        verifier_revision="verifier-1",
        privacy_state="heuristic",
        required_skills=("core-principles",),
    )
    trace.record_skill_preflight(
        skill="core-principles",
        skill_revision="skill-1",
        mandatory=True,
        trigger="tool episode",
        read_result="read",
        scope_decision="in_scope",
        content_sha256=_digest("skill"),
    )
    trace.record_tool_registry(
        registry_revision="registry-1",
        tools=[
            {
                "name": "code-indexer.search",
                "schema": parameters,
                "trust_class": "derived_read_only",
                "side_effect_class": "read_only",
            }
        ],
    )
    trace.record_decision(
        decision="use",
        decision_basis="required semantic lookup",
        required_gate_status="passed",
        tool_name="code-indexer.search",
    )
    trace.record_tool_call(
        tool_name="code-indexer.search",
        call_id=call_id,
        arguments={"query": "owner"},
        permission_decision="not_required",
    )
    trace.record_tool_observation(call_id=call_id, status="success", output="one match")
    trace.record_verification(
        verifier_revision="verifier-1",
        checks=[{"name": "matched", "result": "pass"}],
        result="pass",
        durable_evidence=["artifact:bounded"],
    )
    trace.record_terminal(status="success", evidence=["verification:pass"])
    trace.write_jsonl(path)


class HarnessToolProjectionTests(unittest.TestCase):
    def test_projection_requires_matching_registry_and_verifier_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode.jsonl"
            trace = root / "trace.jsonl"
            episode.write_bytes(canonical_json_bytes(_episode()) + b"\n")
            _write_trace(trace)

            manifest = project_harness_tool_sft(episode, trace, root / "output")

            self.assertFalse(manifest["training_authorized"])
            self.assertTrue(manifest["trainer_loadable"])
            self.assertEqual(manifest["counts"], {"input_records": 1, "tool_sft": 1})
            self.assertEqual(manifest["validation"]["registry_join"], "passed")
            row = json.loads((root / "output" / "tool_sft.jsonl").read_text())
            self.assertEqual(
                set(row), {"schema_version", "example_id", "split", "messages", "tools"}
            )
            lineage = json.loads((root / "output" / "lineage.jsonl").read_text())
            self.assertEqual(lineage["registry_revision"], "registry-1")
            self.assertEqual(lineage["verification_result"], "pass")

    def test_projection_rejects_call_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode.jsonl"
            trace = root / "trace.jsonl"
            episode.write_bytes(canonical_json_bytes(_episode("call-1")) + b"\n")
            _write_trace(trace, "call-2")
            with self.assertRaises(HarnessProjectionError):
                project_harness_tool_sft(episode, trace, root / "output")


if __name__ == "__main__":
    unittest.main()
