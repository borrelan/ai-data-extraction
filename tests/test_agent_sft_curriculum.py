import json
import tempfile
import unittest
from pathlib import Path

from build_agent_sft_curriculum import (
    build_curriculum,
    normalize_when2call_target,
    normalize_when2call_tools,
)
from runtime.sft.dataset import validate_example
from runtime.sft.filter_release import canonical_bytes, write_jsonl


def native_tool(name="exec_command"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Run a tool.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    }


def example(example_id, split, lane, prompt, target, tools=None):
    row = {
        "schema_version": "ai-data-extraction/agent-sft-example/v1",
        "example_id": example_id,
        "split": split,
        "lane": lane,
        "messages": [
            {"role": "user", "content": prompt},
            target,
        ],
    }
    if tools:
        row["tools"] = tools
    return row


def make_release(root, rows):
    root.mkdir()
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    lineage = [
        {
            "example_id": row["example_id"],
            "parent_id": f"parent-{row['example_id']}",
            "outcome_verification": "fixture",
        }
        for row in rows
    ]
    files = {
        "train.jsonl": write_jsonl(root / "train.jsonl", train),
        "validation.jsonl": write_jsonl(root / "validation.jsonl", validation),
        "lineage.jsonl": write_jsonl(root / "lineage.jsonl", lineage),
    }
    manifest = {
        "schema_version": "ai-data-extraction/agent-sft-pilot/v1",
        "status": "fixture",
        "counts": {"total": len(rows)},
        "files": files,
    }
    (root / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")


def when2call_tool():
    return json.dumps(
        {
            "name": "lookup",
            "description": "Look up an item.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "query": {"type": "str", "description": "Search query."},
                    "limit": {"type": "int, optional"},
                },
            },
            "required": ["query"],
        }
    )


class AgentSftCurriculumTests(unittest.TestCase):
    def test_when2call_projection_uses_native_tool_contract(self):
        tools = normalize_when2call_tools([when2call_tool()])
        category, target = normalize_when2call_target(
            '<TOOLCALL>[{"name":"lookup","arguments":{"query":"x"}}]</TOOLCALL>',
            tools,
            7,
        )
        self.assertEqual(category, "tool_call")
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")
        self.assertEqual(
            tools[0]["function"]["parameters"]["properties"]["limit"]["type"],
            "integer",
        )
        self.assertEqual(target["tool_calls"][0]["function"]["name"], "lookup")

    def test_composer_excludes_blocked_and_unverified_rows_and_balances_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            open_release = root / "open"
            internal_release = root / "internal"
            make_release(
                open_release,
                [
                    example(
                        "open-1",
                        "train",
                        "verified_open_swe_action",
                        "Fix the parser.",
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-open",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": {"cmd": "pytest -q"},
                                    },
                                }
                            ],
                        },
                        [native_tool()],
                    )
                ],
            )
            make_release(
                internal_release,
                [
                    example(
                        "skill-1",
                        "validation",
                        "skill_policy",
                        "A relevant skill governs this task.",
                        {"role": "assistant", "content": "Read the skill first."},
                    ),
                    example(
                        "blocked-1",
                        "train",
                        "reviewed_final_answer",
                        "Diagnose the retired product marker.",
                        {"role": "assistant", "content": "Retired product answer."},
                    ),
                    example(
                        "unverified-1",
                        "train",
                        "frontier_action_window",
                        "Historical action.",
                        {"role": "assistant", "content": "Unverified."},
                    ),
                ],
            )
            when2call = root / "when2call.jsonl"
            rows = [
                {
                    "tools": [when2call_tool()],
                    "messages": [{"role": "user", "content": "Find item x."}],
                    "chosen_response": {
                        "role": "assistant",
                        "content": '<TOOLCALL>[{"name":"lookup","arguments":{"query":"x"}}]</TOOLCALL>',
                    },
                    "rejected_response": {"role": "assistant", "content": "No."},
                },
                {
                    "tools": [when2call_tool()],
                    "messages": [{"role": "user", "content": "Find an item."}],
                    "chosen_response": {
                        "role": "assistant",
                        "content": "Which item should I find?",
                    },
                    "rejected_response": {"role": "assistant", "content": "No."},
                },
                {
                    "tools": [when2call_tool()],
                    "messages": [{"role": "user", "content": "Write a poem."}],
                    "chosen_response": {
                        "role": "assistant",
                        "content": "The available lookup tool cannot help with that request.",
                    },
                    "rejected_response": {"role": "assistant", "content": "No."},
                },
            ]
            when2call.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            output = root / "output"
            manifest = build_curriculum(
                open_release=open_release,
                internal_release=internal_release,
                when2call_source=when2call,
                when2call_revision="fixture-revision",
                output_dir=output,
                replay_cap_per_category=1,
                blocked_text_patterns=("retired product",),
            )
            self.assertEqual(manifest["counts"]["total"], 5)
            self.assertEqual(
                manifest["counts"]["partitions"],
                {
                    "internal_skill_policy": 1,
                    "open_swe_verified": 1,
                    "when2call_replay": 3,
                },
            )
            emitted = []
            for split in ("train", "validation"):
                for line in (output / f"{split}.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    validate_example(row, expected_split=split)
                    emitted.append(row)
            encoded = json.dumps(emitted).lower()
            self.assertNotIn("retired product", encoded)
            self.assertNotIn("unverified", encoded)
            self.assertEqual(
                sum(row["lane"] == "tool_policy_replay" for row in emitted), 3
            )


if __name__ == "__main__":
    unittest.main()
