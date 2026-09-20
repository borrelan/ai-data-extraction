import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_open_swe_curriculum import (
    build_open_swe_curriculum,
    normalize_trajectory,
    transition_signatures,
)
from runtime.sft.dataset import validate_example


def tool_schema():
    return json.dumps(
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    )


def action(command, suffix):
    return {
        "role": "assistant",
        "content": "Verbose teacher narration that must not be supervised.",
        "reasoning_content": "private hidden reasoning",
        "tool_calls": [
            {
                "id": f"provider-{suffix}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def observation(output, returncode=0):
    return {
        "role": "tool",
        "content": json.dumps({"returncode": returncode, "output": output}),
        "reasoning_content": "",
        "tool_calls": [],
    }


def source_row(
    *,
    teacher,
    resolved,
    trajectory_id,
    instance_id="repo__project-pr1",
    repo="repo/project",
    first_command="ls",
    wrapped_task=True,
):
    user_content = "Fix the failing parser and run its tests."
    if wrapped_task:
        user_content = f"""<pr_description>
{user_content}
</pr_description>

<instructions>
Write a THOUGHT section and provide exactly ONE bash command.
Create and verify patch.txt, then run echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
</instructions>"""
    return {
        "instance_id": instance_id,
        "repo": repo,
        "license": "MIT",
        "language": "python",
        "trajectory_id": trajectory_id,
        "messages": [
            {
                "role": "system",
                "content": "You are an engineering agent.",
                "reasoning_content": "",
                "tool_calls": [],
            },
            {
                "role": "user",
                "content": user_content,
                "reasoning_content": "",
                "tool_calls": [],
            },
            action(first_command, "one"),
            observation("files"),
            action("python -c \"open('parser.py','w').write('fixed')\"", "two"),
            observation(""),
            action("pytest -q", "three"),
            observation("1 passed" if resolved == 1 else "1 failed", 0 if resolved == 1 else 1),
            action("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "four"),
        ],
        "tools": [tool_schema()],
        "resolved": resolved,
        "metadata": {
            "category": "bug-fix",
            "teacher_model": {
                "name": teacher,
                "enable_thinking": teacher == "Qwen3.8-27B",
                "reasoning_effort": "xhigh" if teacher == "Qwen3.8-27B" else None,
            },
            "reference_patch": {
                "patch": "diff --git a/parser.py b/parser.py",
                "num_modified_files": 1,
                "num_modified_lines": 1,
            },
            "model_patch": {
                "patch": "diff --git a/parser.py b/parser.py",
                "num_modified_files": 1,
                "num_modified_lines": 1,
            },
        },
        "hf_dataset_name": "fixture",
    }


class OpenSweCurriculumTests(unittest.TestCase):
    def test_ambiguous_positional_join_is_rejected(self):
        row = source_row(
            teacher="Qwen3.8-27B", resolved=1, trajectory_id="trajectory"
        )
        row["messages"][2]["tool_calls"].append(
            {
                "id": "second",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "pwd"}),
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "observation_join_ambiguous"):
            normalize_trajectory(row)

    def test_unobserved_nonterminal_call_is_rejected(self):
        row = source_row(
            teacher="Qwen3.8-27B", resolved=1, trajectory_id="trajectory"
        )
        row["messages"] = row["messages"][:7]
        with self.assertRaisesRegex(ValueError, "observation_missing"):
            normalize_trajectory(row)

    def test_terminal_sentinel_is_removed_not_supervised(self):
        row = source_row(
            teacher="Qwen3.8-27B", resolved=1, trajectory_id="trajectory"
        )
        trajectory = normalize_trajectory(row)
        self.assertTrue(trajectory.terminal_sentinel_removed)
        self.assertTrue(trajectory.source_harness_instructions_removed)
        self.assertEqual(
            trajectory.base_messages[-1]["content"],
            "Repository task:\n\nFix the failing parser and run its tests.",
        )
        self.assertNotIn(
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            json.dumps([turn.target for turn in trajectory.turns]),
        )

    def test_transition_state_chain_matches_equal_history_then_diverges(self):
        first = normalize_trajectory(
            source_row(
                teacher="Qwen3.8-27B",
                resolved=1,
                trajectory_id="first",
                first_command="ls",
            )
        )
        second = normalize_trajectory(
            source_row(
                teacher="Qwen3.6-27B",
                resolved=0,
                trajectory_id="second",
                first_command="pwd",
            )
        )
        first_states = list(transition_signatures(first))
        second_states = list(transition_signatures(second))
        self.assertEqual(first_states[0][0], second_states[0][0])
        self.assertNotEqual(first_states[0][1], second_states[0][1])
        self.assertNotEqual(first_states[1][0], second_states[1][0])

    def test_streaming_build_strips_reasoning_and_excludes_eval_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qwen38 = root / "qwen38.parquet"
            qwen36 = root / "qwen36.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        source_row(
                            teacher="Qwen3.8-27B",
                            resolved=1,
                            trajectory_id="chosen",
                        ),
                        source_row(
                            teacher="Qwen3.8-27B",
                            resolved=1,
                            trajectory_id="held",
                            instance_id="held__repo-pr2",
                            repo="held/repo",
                        ),
                        source_row(
                            teacher="Qwen3.8-27B",
                            resolved=-1,
                            trajectory_id="unknown",
                            instance_id="other__repo-pr3",
                            repo="other/repo",
                        ),
                        source_row(
                            teacher="Qwen3.8-27B",
                            resolved=1,
                            trajectory_id="submission-tail",
                            instance_id="tail__repo-pr4",
                            repo="tail/repo",
                            first_command="git diff -- parser.py > patch.txt",
                        ),
                    ]
                ),
                qwen38,
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        source_row(
                            teacher="Qwen3.6-27B",
                            resolved=0,
                            trajectory_id="rejected",
                            first_command="pwd",
                        )
                    ]
                ),
                qwen36,
            )
            registry = root / "registry.jsonl"
            registry.write_text(
                json.dumps(
                    {
                        "repository": "held/repo",
                        "source_instance_id": "held__repo-pr2",
                        "instruction_sha256": "a" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            eval_cases = root / "cases.jsonl"
            eval_cases.write_text(
                json.dumps({"prompt": "Unrelated diagnostic prompt"}) + "\n",
                encoding="utf-8",
            )
            output = root / "candidate"
            manifest = build_open_swe_curriculum(
                source_paths=[qwen38, qwen36],
                output_dir=output,
                benchmark_registry=registry,
                eval_cases=eval_cases,
                dataset_id="fixture/open-swe",
                dataset_revision="fixture-revision",
                language_caps={"python": 16},
                preference_cap=8,
            )

            self.assertFalse(manifest["training_authorized"])
            self.assertEqual(manifest["counts"]["source_decisions"]["input_rows"], 5)
            self.assertEqual(
                manifest["counts"]["source_decisions"][
                    "reason:evaluation_repository_overlap"
                ],
                1,
            )
            examples = []
            for split in ("train", "validation"):
                with (output / f"{split}.jsonl").open() as source:
                    for line in source:
                        row = json.loads(line)
                        validate_example(row, expected_split=split)
                        examples.append(row)
            self.assertGreaterEqual(len(examples), 3)
            encoded = json.dumps(examples).lower()
            self.assertNotIn("reasoning_content", encoded)
            self.assertNotIn("private hidden reasoning", encoded)
            self.assertNotIn("verbose teacher narration", encoded)
            self.assertNotIn("complete_task_and_submit_final_output", encoded)
            self.assertNotIn("<instructions>", encoded)
            self.assertNotIn("thought section", encoded)
            self.assertNotIn("patch.txt", encoded)
            self.assertNotIn("held/repo", encoded)
            self.assertEqual(
                manifest["selection"]["selected_source_harness_removals"],
                len(examples),
            )
            self.assertGreaterEqual(
                manifest["selection"]["materialization_rejections"].get(
                    "source harness protocol survived projection", 0
                ),
                1,
            )
            self.assertTrue(
                all(
                    call["function"]["name"] == "exec_command"
                    and "cmd" in call["function"]["arguments"]
                    for row in examples
                    for message in row["messages"]
                    for call in message.get("tool_calls") or []
                )
            )
            preferences = [
                json.loads(line)
                for line in (output / "preference_candidates.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
            self.assertEqual(len(preferences), 1)
            self.assertEqual(preferences[0]["training_role"], "audit_only")
            self.assertEqual(
                preferences[0]["preference_status"], "requires_interactive_replay"
            )


if __name__ == "__main__":
    unittest.main()
