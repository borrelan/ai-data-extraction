import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_replay_adjudication import build_replay_adjudication


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


def _task(queue_id: str, parent: str, action_name: str) -> dict[str, object]:
    pairs = [
        {
            "call_id": f"call-{queue_id}",
            "action_name": action_name,
            "action_input_json": '{"workdir":"/workspace"}',
            "observation_output_json": '"ok"',
            "observation_present": True,
        }
    ]
    return {
        "schema_version": "ai-data-extraction/tool-replay-adjudication-task/v1",
        "queue_id": queue_id,
        "parent_record_sha256": parent,
        "source_example_id": f"example-{queue_id}",
        "provider": "codex",
        "agent": "codex",
        "model_tier": "tier1_frontier",
        "quality_tier": "candidate",
        "quality_reason_json": '["outcome_unverified"]',
        "privacy_eligible_for_training": False,
        "privacy_state": "review_required",
        "action_names_json": json.dumps([action_name]),
        "tool_families_json": '["shell"]',
        "event_pairs_json": json.dumps(pairs),
        "action_count": 1,
        "observation_count": 1,
        "messages_json": '[{"role":"user","content":"run"}]',
        "lineage_json": "{}",
        "tags_json": "[]",
        "source_metadata_json": "{}",
        "registry_candidate_names_json": "[]",
        "replay_blockers_json": '["registry_status"]',
        "registry_status": "name_candidate_unbound",
        "replay_selection_rank": 1,
        "replay_status": "selected",
        "reward_status": "not_exported",
        "schema_status": "not_observed",
        "verification_status": "not_observed",
        "split": "train",
        "source_pilot_file": "train.jsonl",
        "source_pilot_line": 1,
        "source_pilot_manifest_sha256": "pilot",
        "source_pilot_row_sha256": parent,
        "segment_record_sha256": parent,
        "training_authorized": False,
    }


class ReplayAdjudicationTests(unittest.TestCase):
    def test_preserves_payload_and_blocks_unbound_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue"
            queue.mkdir()
            tasks = [
                _task("queue-1", "parent-1", "shell"),
                _task("queue-2", "parent-2", "unknown_tool"),
            ]
            descriptor = _write_jsonl(queue / "replay_tasks.jsonl", tasks)
            (queue / "manifest.json").write_bytes(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/tool-schema-enrichment-queue/v1",
                        "training_authorized": False,
                        "counts": {"replay_records": 2},
                        "files": {"replay_tasks.jsonl": descriptor},
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
                        "scope": "test",
                        "definitions": [{"name": "shell"}],
                    }
                ),
                encoding="utf-8",
            )

            output = root / "packet"
            manifest = build_replay_adjudication(
                queue_dir=queue,
                registry_path=registry,
                output_dir=output,
            )

            self.assertEqual(manifest["counts"]["packet_tasks"], 2)
            self.assertEqual(manifest["counts"]["unique_parent_sessions"], 2)
            self.assertEqual(
                manifest["decision_counts"],
                {
                    "blocked_call_identity_unbound": 1,
                    "blocked_no_exact_registry_match": 1,
                },
            )
            rows = [
                json.loads(line)
                for line in (output / "replay_tasks.jsonl").read_text().splitlines()
            ]
            self.assertEqual(json.loads(rows[0]["event_pairs_json"])[0]["action_name"], "shell")
            self.assertTrue(all(row["training_authorized"] is False for row in rows))
            self.assertTrue(all(row["replay_status"] == "blocked" for row in rows))
            self.assertTrue(all(row["verifier_status"] == "not_provided" for row in rows))


if __name__ == "__main__":
    unittest.main()
