import json
import tempfile
import unittest
from pathlib import Path

from build_agent_sft_pilot import (
    _compatibility_schemas,
    _visible_rejection,
    resolve_parent_split_conflicts,
    select_action_refs,
)
from runtime.sft.skill_curriculum import SKILLS, build_skill_curriculum


ROOT = Path(__file__).resolve().parents[1]


class AgentSftPilotTests(unittest.TestCase):
    def test_firewall_rejects_legacy_and_provider_branding(self):
        self.assertEqual(
            _visible_rejection(
                {"messages": [{"role": "user", "content": "legacy SpatialChat task"}]},
                max_characters=1000,
            ),
            "legacy_brand",
        )
        self.assertEqual(
            _visible_rejection(
                {"messages": [{"role": "user", "content": "ask Codex to choose"}]},
                max_characters=1000,
            ),
            "provider_brand",
        )

    def test_parent_balanced_selection_preserves_tool_diversity(self):
        refs = []
        for parent in ("a", "b"):
            for index, tool in enumerate(("shell", "shell", "read", "search")):
                refs.append(
                    {
                        "example_id": f"{parent}-{index}",
                        "parent_id": parent,
                        "split": "train",
                        "target_tool": tool,
                    }
                )
        selected, decisions = select_action_refs(
            refs,
            total_caps={"train": 4, "validation": 0},
            parent_caps={"train": 2, "validation": 0},
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["parent_id"] for row in selected}, {"a", "b"})
        self.assertEqual(decisions["train_parent_cap_excluded"], 4)
        self.assertGreaterEqual(len({row["target_tool"] for row in selected}), 2)

    def test_compatibility_schema_requires_only_always_present_arguments(self):
        rows = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read",
                                    "arguments": {"path": "a", "offset": 1},
                                }
                            }
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read", "arguments": {"path": "b"}}}
                        ],
                    }
                ]
            },
        ]
        schema = _compatibility_schemas(rows)["read"]["function"]["parameters"]
        self.assertEqual(schema["required"], ["path"])
        self.assertEqual(schema["properties"]["path"]["type"], "string")
        self.assertEqual(schema["properties"]["offset"]["type"], "integer")
        self.assertFalse(schema["additionalProperties"])

    def test_reviewed_split_wins_over_opposing_action_window(self):
        examples = [
            {"example_id": "gold", "split": "validation", "lane": "reviewed_final_answer"},
            {"example_id": "action", "split": "train", "lane": "frontier_action_window"},
            {"example_id": "other", "split": "train", "lane": "frontier_action_window"},
        ]
        lineage = [
            {"example_id": "gold", "parent_id": "shared"},
            {"example_id": "action", "parent_id": "shared"},
            {"example_id": "other", "parent_id": "other"},
        ]
        selected, selected_lineage, report = resolve_parent_split_conflicts(examples, lineage)
        self.assertEqual({row["example_id"] for row in selected}, {"gold", "other"})
        self.assertEqual({row["example_id"] for row in selected_lineage}, {"gold", "other"})
        self.assertEqual(report["conflicting_parents"], 1)
        self.assertEqual(report["excluded_by_lane"], {"frontier_action_window": 1})

    def test_skill_curriculum_is_source_bound_and_parent_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = root / "skills"
            for name in SKILLS:
                path = skills / name / "SKILL.md"
                path.parent.mkdir(parents=True)
                path.write_text(f"# {name}\n\nBounded instructions.\n", encoding="utf-8")
            examples, lineage, bindings = build_skill_curriculum(
                skill_root=skills,
                tool_schema_path=ROOT / "runtime" / "sft" / "tool_schemas.json",
            )
            self.assertEqual(bindings["counts"], {"train": 24, "validation": 8})
            self.assertEqual(len(examples), 32)
            self.assertEqual(len({row["example_id"] for row in examples}), 32)
            self.assertEqual(len({row["parent_id"] for row in lineage}), 32)
            self.assertTrue(
                all(
                    len([tool["function"]["name"] for tool in row["tools"]])
                    == len({tool["function"]["name"] for tool in row["tools"]})
                    for row in examples
                )
            )
            visible = json.dumps(examples).lower()
            self.assertNotIn("spatialchat", visible)
            self.assertTrue(all(item["sha256"] for item in bindings["skills"].values()))


if __name__ == "__main__":
    unittest.main()
