import json
import unittest
from unittest.mock import patch

from build_agent_preference_curriculum_v2 import (
    pair_within_token_limit,
    skill_negative_v2,
    strict_observation_status,
)


def call(name, arguments, call_id):
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


def tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Use {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class AgentPreferenceCurriculumV2Tests(unittest.TestCase):
    def test_pair_gate_uses_requested_sequence_contract(self):
        pair = {"pair_id": "pair-1"}
        exclusions = {}
        with patch(
            "build_agent_preference_curriculum_v2.render_pair",
            side_effect=ValueError("chosen branch exceeds 6144 tokens: 6200"),
        ) as render:
            self.assertFalse(
                pair_within_token_limit(
                    pair,
                    object(),
                    max_sequence_tokens=6144,
                    token_exclusions=exclusions,
                )
            )
        render.assert_called_once_with(pair, unittest.mock.ANY, max_length=6144)
        self.assertEqual(
            exclusions,
            {"pair-1": "chosen branch exceeds 6144 tokens: 6200"},
        )

    def test_pair_gate_does_not_hide_non_limit_validation_errors(self):
        with patch(
            "build_agent_preference_curriculum_v2.render_pair",
            side_effect=ValueError("prompt prefix mismatch"),
        ):
            with self.assertRaisesRegex(ValueError, "prefix mismatch"):
                pair_within_token_limit(
                    {"pair_id": "pair-1"},
                    object(),
                    max_sequence_tokens=6144,
                    token_exclusions={},
                )

    def test_observation_status_uses_returncode_not_error_words(self):
        prompt = [
            {"role": "user", "content": "Continue."},
            call("exec_command", {"cmd": "test"}, "call-1"),
            {
                "role": "tool",
                "name": "exec_command",
                "tool_call_id": "call-1",
                "content": json.dumps(
                    {"returncode": 0, "output": "expected error text in fixture"}
                ),
            },
        ]
        self.assertEqual(strict_observation_status(prompt, 1), "explicit_success")
        prompt[-1]["content"] = json.dumps({"returncode": 2, "output": ""})
        self.assertEqual(strict_observation_status(prompt, 1), "explicit_failure")

    def test_skill_apply_penalizes_premature_completion(self):
        example = {
            "messages": [
                {"role": "user", "content": "Use the required skill, then inspect."},
                call("read_skill", {"name": "core-principles"}, "call-skill"),
                {
                    "role": "tool",
                    "name": "read_skill",
                    "tool_call_id": "call-skill",
                    "content": "skill text",
                },
                call("read_file", {"path": "a.py"}, "call-read"),
            ],
            "tools": [tool("read_skill"), tool("read_file")],
        }
        rejected, family = skill_negative_v2(example, "skill_apply")
        self.assertEqual(family, "premature_stop_before_required_work")
        self.assertEqual(rejected, {"role": "assistant", "content": "The task is complete."})

    def test_read_only_policy_penalizes_mutation(self):
        example = {
            "messages": [
                {"role": "user", "content": "Inspect only."},
                call("read_file", {"path": "a.py"}, "call-read"),
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "call-read",
                    "content": "contents",
                },
                {"role": "assistant", "content": "The file is valid."},
            ],
            "tools": [tool("read_file"), tool("apply_patch")],
        }
        rejected, family = skill_negative_v2(example, "read_only_report")
        self.assertEqual(family, "permission_read_only_boundary")
        self.assertEqual(rejected["tool_calls"][0]["function"]["name"], "apply_patch")

    def test_verified_finish_penalizes_repeating_verification(self):
        example = {
            "messages": [
                {"role": "user", "content": "Test and finish."},
                call("run_tests", {"command": "pytest -q"}, "call-test"),
                {
                    "role": "tool",
                    "name": "run_tests",
                    "tool_call_id": "call-test",
                    "content": "passed",
                },
                call("finish_task", {"summary": "done", "evidence": "passed"}, "call-finish"),
            ],
            "tools": [tool("run_tests"), tool("finish_task")],
        }
        rejected, family = skill_negative_v2(example, "verified_finish")
        self.assertEqual(family, "verified_finish_over_unnecessary_continuation")
        self.assertEqual(rejected["tool_calls"][0]["function"]["name"], "run_tests")
        self.assertEqual(rejected["tool_calls"][0]["id"], "call-rejected-00")


if __name__ == "__main__":
    unittest.main()
