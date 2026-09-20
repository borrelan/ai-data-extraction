import json
import tempfile
import unittest
from pathlib import Path

from trainer_export import canonical_json_bytes, export_silver_sft_pilot


def _row(
    example_id: str,
    parent: str,
    *,
    tool: bool = False,
    redactions: int = 0,
    quality_flags: list[str] | None = None,
) -> dict[str, object]:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "request"},
    ]
    if tool:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "content": "observed", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "done"},
            ]
        )
    else:
        messages.append({"role": "assistant", "content": "response"})
    return {
        "schema_version": "ai-data-extraction/v1",
        "example_id": example_id,
        "dataset": "sft",
        "split": "train",
        "tags": [],
        "messages": messages,
        "metadata": {
            "provider": "codex",
            "model_tier": "tier1_frontier",
            "training_lane": "primary",
            "source_file_name": "source.jsonl",
            "source_file_sha256": "source-sha",
            "source_line": 1,
            "parent_record_sha256": parent,
            "source_origin": {
                "source_file_name": "raw.jsonl",
                "source_file_sha256": "raw-sha",
            },
        },
        "lineage": {
            "parent_record_sha256": parent,
            "source_message_range": {"start": 0, "end": len(messages) - 1},
        },
        "quality": {
            "status": "review",
            "has_tools": tool,
            "session_quality_gate": "candidate",
            "session_quality_flags": quality_flags or [],
        },
        "privacy": {
            "eligible_for_training": False,
            "structural_redactions": redactions,
        },
    }


class SilverSftPilotTests(unittest.TestCase):
    def test_materializer_is_streamed_parent_disjoint_and_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "sft.candidate.jsonl"
            rows = [
                _row("a-long", "parent-a"),
                _row("a-short", "parent-a"),
                _row("b", "parent-b"),
                _row("redacted", "parent-c", redactions=1),
                _row("tool", "parent-d", tool=True),
            ]
            candidate.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
            output = root / "pilot"

            manifest = export_silver_sft_pilot(candidate, output)

            self.assertFalse(manifest["training_authorized"])
            self.assertTrue(manifest["trainer_loadable"])
            self.assertEqual(manifest["counts"]["input_records"], 5)
            self.assertEqual(manifest["counts"]["eligible_before_parent_dedup"], 3)
            self.assertEqual(manifest["counts"]["selected_records"], 2)
            self.assertEqual(manifest["counts"]["parents_selected"], 2)
            self.assertEqual(manifest["counts"]["excluded_reason_counts"], {
                "duplicate_parent_selection": 1,
                "privacy_redactions_present": 1,
                "tool_activity_not_silver_sft": 1,
            })
            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            validation = [
                json.loads(line)
                for line in (output / "validation.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train), 1)
            self.assertEqual(len(validation), 1)
            self.assertEqual(set(train[0]), {"schema_version", "example_id", "split", "messages"})
            self.assertEqual(set(validation[0]), {"schema_version", "example_id", "split", "messages"})
            lineage = [json.loads(line) for line in (output / "lineage.jsonl").read_text().splitlines()]
            self.assertEqual({row["parent_record_sha256"] for row in lineage}, {"parent-a", "parent-b"})
            decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
            self.assertEqual(len(decisions), 5)

    def test_materializer_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "sft.candidate.jsonl"
            candidate.write_bytes(canonical_json_bytes(_row("one", "parent-one")) + b"\n")
            output = root / "pilot"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export_silver_sft_pilot(candidate, output)

    def test_materializer_blocks_configured_source_quality_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "sft.quarantine.jsonl"
            rows = [
                _row("clean", "parent-clean"),
                _row(
                    "truncated",
                    "parent-truncated",
                    quality_flags=["outcome_unverified", "payload_truncated"],
                ),
            ]
            candidate.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
            manifest = export_silver_sft_pilot(
                candidate,
                root / "pilot",
                source_partition="sft.quarantine",
                blocked_quality_flags=("payload_truncated",),
                quality_limitations=("source_quality_denylist_applied",),
            )
            self.assertEqual(manifest["counts"]["selected_records"], 1)
            self.assertEqual(
                manifest["counts"]["excluded_reason_counts"],
                {"source_quality_flag:payload_truncated": 1},
            )
            self.assertEqual(manifest["selection"]["source_partition"], "sft.quarantine")
            self.assertEqual(
                manifest["selection"]["blocked_quality_flags"], ["payload_truncated"]
            )


if __name__ == "__main__":
    unittest.main()
