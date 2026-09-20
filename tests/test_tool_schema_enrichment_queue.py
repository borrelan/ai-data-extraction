import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_tool_schema_enrichment_queue import build_queue


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(raw)
    return {
        "records": len(rows),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _row(
    example_id: str,
    parent: str,
    action_name: str = "shell",
    split: str = "train",
) -> dict[str, object]:
    action = {
        "call_id": f"call-{example_id}",
        "event_id": f"event-action-{example_id}",
        "input": {"command": ["true"]},
        "kind": "action",
        "message_index": 1,
        "name": action_name,
        "ordinal": 0,
        "schema_version": "ai-data-extraction/event/v1",
    }
    observation = {
        "call_id": f"call-{example_id}",
        "event_id": f"event-observation-{example_id}",
        "kind": "observation",
        "message_index": 2,
        "name": "unknown",
        "ordinal": 1,
        "output": "ok",
        "schema_version": "ai-data-extraction/event/v1",
    }
    return {
        "schema_version": "ai-data-extraction/historical-tool-trajectory-pilot/v1",
        "example_id": example_id,
        "split": split,
        "provider": "codex",
        "agent": "codex",
        "model_tier": "tier1_frontier",
        "quality_tier": "candidate",
        "quality_reason": ["outcome_unverified"],
        "tool_families": ["shell"],
        "tags": ["trajectory:historical-review"],
        "messages": [
            {"role": "user", "content": "Run the check."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{example_id}",
                        "type": "function",
                        "function": {"name": action_name, "arguments": "{\"command\":[\"true\"]}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call-{example_id}", "content": "ok"},
        ],
        "events": [action, observation],
        "metadata": {
            "provider": "codex",
            "source_label": "codex",
            "source_file_name": "sessions.jsonl",
            "source_file_sha256": "source-file",
            "source_line": 1,
            "source_record_sha256": parent,
            "segment_record_sha256": example_id,
            "parser_revision": "test/parser-v1",
        },
        "lineage": {
            "parent_record_sha256": parent,
            "segment_record_sha256": example_id,
            "source_message_range": {"start": 0, "end": 2},
        },
        "privacy": {
            "state": "review_required",
            "eligible_for_training": False,
            "structural_redactions": 0,
        },
    }


class ToolSchemaEnrichmentQueueTests(unittest.TestCase):
    def test_builds_parent_diverse_review_queue_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = root / "pilot"
            pilot.mkdir()
            train = [_row("example-1", "parent-1"), _row("example-2", "parent-1")]
            validation = [_row("example-3", "parent-2", "read_file", "validation")]
            train_descriptor = _write_jsonl(pilot / "train.jsonl", train)
            validation_descriptor = _write_jsonl(pilot / "validation.jsonl", validation)
            (pilot / "manifest.json").write_bytes(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/historical-tool-trajectory-pilot/v1",
                        "training_authorized": False,
                        "files": {
                            "train.jsonl": train_descriptor,
                            "validation.jsonl": validation_descriptor,
                        },
                    },
                    sort_keys=True,
                ).encode()
                + b"\n"
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "registry_revision": "test-registry-v1",
                        "source_revision": "test-source",
                        "definitions": [{"name": "shell"}],
                    }
                ),
                encoding="utf-8",
            )

            output = root / "queue"
            result = build_queue(
                pilot_dir=pilot,
                output_dir=output,
                registry_path=registry,
                replay_limit=2,
            )

            self.assertEqual(result["counts"]["input_records"], 3)
            self.assertEqual(result["counts"]["replay_records"], 2)
            rows = [json.loads(line) for line in (output / "queue.jsonl").read_text().splitlines()]
            tasks = [
                json.loads(line)
                for line in (output / "replay_tasks.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["parent_record_sha256"] for row in tasks}), 2)
            self.assertTrue(all(row["training_authorized"] is False for row in rows))
            self.assertTrue(all(row["schema_status"] == "not_observed" for row in rows))
            self.assertTrue(all(row["verification_status"] == "not_observed" for row in rows))
            self.assertTrue(all(row["reward_status"] == "not_exported" for row in rows))
            self.assertEqual(rows[0]["registry_status"], "name_candidate_unbound")
            pairs = json.loads(rows[0]["event_pairs_json"])
            self.assertEqual(pairs[0]["action_name"], "shell")
            self.assertNotIn("_queue_event_index", pairs[0]["action_event_json"])
            decisions = [
                json.loads(line)
                for line in (output / "decisions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(decisions), 3)
            self.assertEqual(sum(1 for row in decisions if row["replay_selected"]), 2)


if __name__ == "__main__":
    unittest.main()
