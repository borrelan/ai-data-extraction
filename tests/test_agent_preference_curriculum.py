import json
import tempfile
import unittest
from pathlib import Path

from build_agent_preference_curriculum import (
    build_curriculum,
    make_pair,
    skill_rejected_action,
)
from runtime.dpo.dataset import preflight_release, validate_pair
from runtime.sft.filter_release import canonical_bytes, write_jsonl


def tool(name, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Use {name}.",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = {
    "read_skill": tool(
        "read_skill", {"name": {"type": "string"}}, ["name"]
    ),
    "search_code": tool(
        "search_code", {"query": {"type": "string"}}, ["query"]
    ),
    "read_file": tool("read_file", {"path": {"type": "string"}}, ["path"]),
}


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


def make_release(path, rows, lineage):
    path.mkdir()
    files = {
        "train.jsonl": write_jsonl(
            path / "train.jsonl", (row for row in rows if row["split"] == "train")
        ),
        "validation.jsonl": write_jsonl(
            path / "validation.jsonl",
            (row for row in rows if row["split"] == "validation"),
        ),
        "lineage.jsonl": write_jsonl(path / "lineage.jsonl", lineage),
    }
    manifest = {
        "schema_version": "ai-data-extraction/agent-sft-pilot/v1",
        "status": "fixture",
        "counts": {"total": len(rows)},
        "files": files,
    }
    (path / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")


def sft_row(example_id, split, lane, messages, tools):
    return {
        "schema_version": "ai-data-extraction/agent-sft-example/v1",
        "example_id": example_id,
        "split": split,
        "lane": lane,
        "messages": messages,
        "tools": tools,
    }


def raw_tool(name="lookup"):
    return json.dumps(
        {
            "name": name,
            "description": "Look up an item.",
            "parameters": {
                "type": "dict",
                "properties": {"query": {"type": "str"}},
            },
            "required": ["query"],
        }
    )


class FakeTokenizer:
    chat_template = "fixture-template"

    def __len__(self):
        return 256

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        **_,
    ):
        chunks = ["TOOLS:" + ",".join(t["function"]["name"] for t in tools or [])]
        for message in messages:
            chunks.append(f"<{message['role']}>{message.get('content', '')}")
            for item in message.get("tool_calls") or []:
                chunks.append(json.dumps(item, sort_keys=True))
        if add_generation_prompt:
            chunks.append("<assistant>")
        value = "\n".join(chunks)
        return [ord(char) for char in value] if tokenize else value


class AgentPreferenceCurriculumTests(unittest.TestCase):
    def test_skill_negative_repeats_prior_action_after_observation(self):
        example = sft_row(
            "skill-1",
            "train",
            "skill_policy",
            [
                {"role": "user", "content": "Read the file and report."},
                call("read_file", {"path": "a.txt"}, "call-read"),
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "call-read",
                    "content": "done",
                },
                {"role": "assistant", "content": "The file is done."},
            ],
            [TOOLS["read_file"]],
        )
        rejected, reason = skill_rejected_action(example, "read_only_report")
        self.assertEqual(reason, "repeats_observed_action_instead_of_follow_through")
        self.assertEqual(rejected["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(rejected["tool_calls"][0]["id"], "call-rejected")

    def test_pair_validation_and_exact_rendering(self):
        pair = make_pair(
            split="train",
            lane="skill_policy_preference",
            prompt=[{"role": "user", "content": "Use the governing skill."}],
            chosen=call("read_skill", {"name": "core-principles"}, "call-chosen"),
            rejected=call("search_code", {"query": "change it"}, "call-rejected"),
            tools=[TOOLS["read_skill"], TOOLS["search_code"]],
            identity={"fixture": 1},
        )
        validate_pair(pair, expected_split="train")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "train.jsonl": write_jsonl(root / "train.jsonl", [pair]),
                "validation.jsonl": write_jsonl(root / "validation.jsonl", []),
            }
            manifest = {
                "schema_version": "ai-data-extraction/agent-preference-release/v1",
                "model": {"max_sequence_tokens": 8192},
                "counts": {"total": 1},
                "files": files,
            }
            (root / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
            report, tokenized = preflight_release(
                root, FakeTokenizer(), max_length=8192, retain_tokens=True
            )
            self.assertEqual(report["counts"]["total"], 1)
            self.assertEqual(len(tokenized["train"]), 1)
            self.assertNotEqual(
                tokenized["train"][0]["chosen_input_ids"],
                tokenized["train"][0]["rejected_input_ids"],
            )

    def test_composer_keeps_same_state_pairs_and_excludes_invalid_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill_release = root / "skills"
            skill_rows = [
                sft_row(
                    "skill-route",
                    "train",
                    "skill_policy",
                    [
                        {"role": "user", "content": "Trace the owner before editing."},
                        call(
                            "read_skill",
                            {"name": "core-principles"},
                            "call-skill",
                        ),
                    ],
                    [TOOLS["read_skill"], TOOLS["search_code"]],
                ),
                sft_row(
                    "skill-direct",
                    "validation",
                    "skill_policy",
                    [
                        {"role": "user", "content": "Return lowercase READY."},
                        {"role": "assistant", "content": "ready"},
                    ],
                    [TOOLS["read_skill"], TOOLS["search_code"]],
                ),
            ]
            make_release(
                skill_release,
                skill_rows,
                [
                    {
                        "example_id": "skill-route",
                        "parent_id": "skill-parent-1",
                        "lane": "skill_policy",
                        "source_kind": "skill_route",
                        "skill": "core-principles",
                        "skill_sha256": "abc",
                    },
                    {
                        "example_id": "skill-direct",
                        "parent_id": "skill-parent-2",
                        "lane": "skill_policy",
                        "source_kind": "skill_skip_low_return",
                        "skill": None,
                        "skill_sha256": None,
                    },
                ],
            )

            raw_rows = [
                {
                    "tools": [raw_tool()],
                    "messages": [{"role": "user", "content": "Find item x."}],
                    "chosen_response": {
                        "role": "assistant",
                        "content": '<TOOLCALL>[{"name":"lookup","arguments":{"query":"x"}}]</TOOLCALL>',
                    },
                    "rejected_response": {
                        "role": "assistant",
                        "content": "Which item should I find?",
                    },
                },
                {
                    "tools": [raw_tool()],
                    "messages": [{"role": "user", "content": "Find item y."}],
                    "chosen_response": {
                        "role": "assistant",
                        "content": '<TOOLCALL>[{"name":"lookup","arguments":{"query":"y"}}]</TOOLCALL>',
                    },
                    "rejected_response": {
                        "role": "assistant",
                        "content": '<TOOLCALL>[{"name":"missing","arguments":{"query":"y"}}]</TOOLCALL>',
                    },
                },
            ]
            raw_source = root / "when2call.jsonl"
            raw_source.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
            )
            from build_agent_sft_curriculum import (
                digest_value,
                normalize_when2call_target,
                normalize_when2call_tools,
            )

            sft_rows = []
            sft_lineage = []
            for index, raw in enumerate(raw_rows, 1):
                tools = normalize_when2call_tools(raw["tools"])
                _, chosen = normalize_when2call_target(
                    raw["chosen_response"]["content"], tools, index
                )
                item = sft_row(
                    f"w2c-{index}",
                    "train" if index == 1 else "validation",
                    "tool_policy_replay",
                    raw["messages"] + [chosen],
                    tools,
                )
                sft_rows.append(item)
                row_hash = digest_value(raw)
                sft_lineage.append(
                    {
                        "example_id": item["example_id"],
                        "parent_id": f"when2call:{row_hash}",
                        "source_partition": "when2call_train_pref",
                        "source_revision": "fixture-revision",
                        "source_row": index,
                        "source_row_sha256": row_hash,
                    }
                )
            sft_release = root / "sft"
            make_release(sft_release, sft_rows, sft_lineage)
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps({"case_id": "eval-1", "prompt": "Held out prompt."}) + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            manifest = build_curriculum(
                skill_release=skill_release,
                sft_release=sft_release,
                when2call_source=raw_source,
                when2call_revision="fixture-revision",
                evaluation_cases=cases,
                output_dir=output,
            )
            self.assertEqual(manifest["counts"]["total"], 3)
            self.assertEqual(
                manifest["counts"]["partitions"],
                {"skill_policy": 2, "when2call_train_pref": 1},
            )
            self.assertEqual(
                manifest["sources"]["when2call"]["excluded_reasons"],
                {"chosen_toolcall_unknown_tool": 1},
            )
            emitted = [
                json.loads(line)
                for name in ("train.jsonl", "validation.jsonl")
                for line in (output / name).read_text().splitlines()
                if line
            ]
            self.assertTrue(all(row["chosen"] != row["rejected"] for row in emitted))
            self.assertTrue(
                all(
                    row["prompt"][0]["content"] != "Held out prompt."
                    for row in emitted
                )
            )


if __name__ == "__main__":
    unittest.main()
