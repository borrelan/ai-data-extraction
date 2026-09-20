import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trl_qwen_tool_sft import (
    ActionWindowIssue,
    NormalizationIssue,
    _action_window_prompt_completion,
    build_qwen_action_window_sft_candidate,
    build_qwen_tool_sft_candidate,
    normalize_tool_messages,
)


def _messages(arguments):
    return [
        {"role": "user", "content": "request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "observed", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "done"},
    ]


def _action_window(
    window_id,
    parent_id,
    prompt_text,
    *,
    name="read_file",
    arguments=None,
    call_id=None,
):
    call_id = call_id or f"call-{window_id}"
    arguments = {"path": "src/main.py"} if arguments is None else arguments
    message = {
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
    return {
        "schema_version": "ai-data-extraction/action-window/v1",
        "window_id": window_id,
        "episode_id": f"episode-{window_id}",
        "context": {
            "messages": [
                {"role": "user", "content": prompt_text},
                message,
            ],
            "state_hash": f"state-{window_id}",
        },
        "decision": {"action": "use", "skill": {"status": "not_observed"}},
        "tool_call": {
            "name": name,
            "call_id": call_id,
            "arguments": copy.deepcopy(arguments),
            "event_id": f"event-{window_id}",
        },
        "observation": {
            "status": "unknown",
            "call_id": call_id,
            "event_id": f"observation-{window_id}",
            "output": "must not be copied into the completion",
        },
        "quality": {
            "stage": "candidate",
            "session_quality_gate": "candidate",
            "session_quality_flags": ["outcome_unverified"],
            "model_tier": "tier1_frontier",
            "tool_contract": "review",
            "verification": "absent",
            "observation_match": "call-id",
            "observation_output_policy": "full",
        },
        "provenance": {
            "parent_record_sha256": parent_id,
            "source_event_id": f"event-{window_id}",
        },
        "lineage": {"parent_record_sha256": parent_id},
        "tags": ["provider:codex", "tier:tier1-frontier", "quality-gate:candidate"],
        "privacy": {
            "mode": "heuristic",
            "reason": "review-required",
            "eligible_for_training": False,
        },
    }


def _write_action_window_release(root, rows):
    release_dir = root / "release"
    release_dir.mkdir()
    data = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    candidate_path = release_dir / "action_windows.candidate.jsonl"
    candidate_path.write_bytes(data)
    manifest = {
        "schema_version": "ai-data-extraction/release-gate/v1",
        "canonical_manifest": {"sha256": "c" * 64},
        "datasets": {
            "action_windows": {
                "partitions": {
                    "candidate": {
                        "path": candidate_path.name,
                        "records": len(rows),
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                }
            }
        },
        "privacy": {"training_authorized": False},
    }
    (release_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return release_dir, candidate_path


class NormalizeToolMessagesTests(unittest.TestCase):
    def test_parses_json_object_string_without_mutating_source(self):
        source = _messages('{"path":"src/main.py","line":4}')
        before = copy.deepcopy(source)

        result = normalize_tool_messages(source)

        self.assertEqual(result.issues, ())
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.normalized_argument_count, 1)
        self.assertEqual(
            result.messages[1]["tool_calls"][0]["function"]["arguments"],
            {"path": "src/main.py", "line": 4},
        )
        self.assertEqual(source, before)
        self.assertIsInstance(source[1]["tool_calls"][0]["function"]["arguments"], str)

    def test_rejects_malformed_json_without_returning_a_trainable_row(self):
        result = normalize_tool_messages(_messages("<partial tool args>"))

        self.assertIsNone(result.messages)
        self.assertIn(NormalizationIssue.ARGUMENTS_INVALID_JSON, result.issues)

    def test_rejects_non_object_json(self):
        result = normalize_tool_messages(_messages('["not", "an", "object"]'))

        self.assertIsNone(result.messages)
        self.assertIn(NormalizationIssue.ARGUMENTS_NOT_OBJECT, result.issues)

    def test_rejects_duplicate_json_keys(self):
        result = normalize_tool_messages(_messages('{"path":"a","path":"b"}'))

        self.assertIsNone(result.messages)
        self.assertIn(NormalizationIssue.ARGUMENTS_DUPLICATE_KEYS, result.issues)

    def test_accepts_already_mapped_arguments_without_mutating_source(self):
        source = _messages({"path": "src/main.py"})
        before = copy.deepcopy(source)

        result = normalize_tool_messages(source)

        self.assertEqual(result.issues, ())
        self.assertEqual(source, before)
        self.assertEqual(
            result.messages[1]["tool_calls"][0]["function"]["arguments"],
            {"path": "src/main.py"},
        )

    def test_rejects_rows_without_tool_calls(self):
        result = normalize_tool_messages([{"role": "user", "content": "hello"}])

        self.assertIsNone(result.messages)
        self.assertEqual(result.issues, (NormalizationIssue.NO_TOOL_CALLS,))

    def test_builder_accounts_for_source_rows_and_preserves_invalid_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            train_row = {
                "example_id": "example-train",
                "split": "train",
                "messages": _messages('{"path":"src/main.py"}'),
                "lineage": {"parent_record_sha256": "parent-train"},
                "provider": "codex",
                "agent": "codex",
                "model_tier": "tier1_frontier",
                "quality_tier": "candidate",
                "quality_reason": ["outcome_unverified"],
                "privacy": {"state": "review_required", "structural_redactions": 1},
                "tool_contract": {"schema_status": "not_observed"},
                "tool_families": ["filesystem"],
                "tags": ["tier:tier1-frontier", "quality-gate:candidate"],
            }
            validation_row = {
                **train_row,
                "example_id": "example-validation",
                "split": "validation",
                "messages": _messages("<partial>"),
                "lineage": {"parent_record_sha256": "parent-validation"},
            }
            bindings = {}
            for split, row in (("train", train_row), ("validation", validation_row)):
                payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                path = source / f"{split}.jsonl"
                path.write_bytes(payload)
                bindings[path.name] = {
                    "records": 1,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "fixture/v1",
                        "source": {"partition": "tool_traces:candidate"},
                        "files": bindings,
                    }
                )
            )
            original_validation = (source / "validation.jsonl").read_bytes()

            output = root / "candidate"
            manifest = build_qwen_tool_sft_candidate(source, output)

            self.assertEqual(manifest["counts"]["input_rows"], 2)
            self.assertEqual(manifest["counts"]["selected_candidate_sft"], 1)
            self.assertEqual(manifest["counts"]["review_only_rows"], 1)
            self.assertEqual((source / "validation.jsonl").read_bytes(), original_validation)
            selected = json.loads((output / "train.jsonl").read_text().strip())
            self.assertEqual(
                selected["messages"][1]["tool_calls"][0]["function"]["arguments"],
                {"path": "src/main.py"},
            )
            decisions = [
                json.loads(line)
                for line in (output / "decisions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(decisions), 2)
            self.assertIn("arguments_invalid_json", decisions[1]["reasons"])
            self.assertNotIn("messages", decisions[1])


class ActionWindowSftTests(unittest.TestCase):
    def test_action_projection_targets_exact_call_and_drops_observation_payload(self):
        source = _action_window("window-1", "parent-1", "open the source file")
        before = copy.deepcopy(source)

        projected, reasons = _action_window_prompt_completion(source)

        self.assertEqual(reasons, ())
        self.assertEqual(source, before)
        self.assertEqual(projected["prompt"], [{"role": "user", "content": "open the source file"}])
        self.assertEqual(len(projected["completion"]), 1)
        self.assertEqual(
            projected["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(projected["completion"][0]["role"], "assistant")
        self.assertEqual(
            projected["completion"][0]["tool_calls"][0]["function"]["arguments"],
            {"path": "src/main.py"},
        )
        self.assertNotIn("output", projected)
        self.assertNotIn("must not be copied", json.dumps(projected))

    def test_action_projection_rejects_event_message_argument_mismatch(self):
        source = _action_window("window-1", "parent-1", "open the source file")
        source["tool_call"]["arguments"] = {"path": "different.py"}

        projected, reasons = _action_window_prompt_completion(source)

        self.assertIsNone(projected)
        self.assertIn(ActionWindowIssue.TARGET_ARGUMENTS_MISMATCH.value, reasons)

    def test_builder_deduplicates_conflicts_and_malformed_rows(self):
        rows = [
            _action_window("window-1", "parent-1", "request one", call_id="call-one"),
            _action_window("window-2", "parent-2", "request one", call_id="call-one"),
            _action_window("window-3", "parent-3", "request two", name="read_file"),
            _action_window("window-4", "parent-4", "request two", name="shell", arguments={"command": "pwd"}),
            _action_window("window-5", "parent-5", "request three", arguments="<partial-arguments>"),
            _action_window("window-6", "parent-6", "request four"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, source_path = _write_action_window_release(root, rows)
            source_bytes = source_path.read_bytes()
            output = root / "candidate"

            manifest = build_qwen_action_window_sft_candidate(release, output)

            counts = manifest["counts"]
            self.assertEqual(counts["input_rows"], 6)
            self.assertEqual(counts["selected_candidate_sft"], 2)
            self.assertEqual(counts["review_only_rows"], 4)
            self.assertEqual(counts["exact_duplicate_prompt_completion_rows"], 1)
            self.assertEqual(counts["conflicting_prompt_groups"], 1)
            self.assertEqual(counts["rows_in_conflicting_prompt_groups"], 2)
            self.assertEqual(counts["parent_split_overlap"], 0)
            self.assertEqual(source_path.read_bytes(), source_bytes)

            selected = [
                json.loads(line)
                for name in ("train.jsonl", "validation.jsonl")
                for line in (output / name).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(selected), 2)
            self.assertTrue(all(row["tags"] and "loss:completion-only" in row["tags"] for row in selected))
            self.assertTrue(all(row["privacy"]["reason"] == "review-required" for row in selected))
            self.assertTrue(all("must not be copied" not in json.dumps(row) for row in selected))

            decisions = [
                json.loads(line)
                for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(decisions), 6)
            reason_sets = [set(item["reasons"]) for item in decisions]
            self.assertIn("duplicate_prompt_completion", set.union(*reason_sets))
            self.assertIn("conflicting_completion_for_prompt", set.union(*reason_sets))
            self.assertIn(NormalizationIssue.ARGUMENTS_INVALID_JSON.value, set.union(*reason_sets))


if __name__ == "__main__":
    unittest.main()
