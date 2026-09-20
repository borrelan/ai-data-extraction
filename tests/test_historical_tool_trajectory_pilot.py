import json
import tempfile
import unittest
from pathlib import Path

from build_historical_tool_trajectory_pilot import (
    export_historical_tool_trajectory_pilot,
)


def _source_row(
    example_id: str,
    parent: str,
    *,
    malformed: bool = False,
    suggestion_mode: bool = False,
) -> dict:
    event = {
        "kind": "action",
        "call_id": "call-1",
        "name": "shell",
        "input": {"command": ["true"]},
        "schema_version": "ai-data-extraction/event/v1",
        "event_id": "sha256:event-1",
        "ordinal": 0,
    }
    observation = {
        "kind": "observation",
        "call_id": "call-1",
        "name": "unknown",
        "status": "unknown",
        "output": "ok",
        "schema_version": "ai-data-extraction/event/v1",
        "event_id": "sha256:event-2",
        "ordinal": 1,
    }
    if malformed:
        observation["call_id"] = "call-missing"
    return {
        "schema_version": "ai-data-extraction/v1",
        "example_id": example_id,
        "source_example_id": example_id,
        "dataset": "tool_trace",
        "split": "train",
        "tags": ["provider:codex", "tool:shell"],
        "messages": [
            {
                "role": "user",
                "content": (
                    "[SUGGESTION MODE: predict the next user message]"
                    if suggestion_mode
                    else "Find the answer."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": {"command": ["true"]}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {
                "role": "assistant",
                "content": (
                    "The requested definition was found and verified against the "
                    "repository source. The result includes the exact owning file, "
                    "symbol, and line, and it distinguishes observed source evidence "
                    "from any remaining unverified behavior."
                ),
            },
        ],
        "events": [event, observation],
        "metadata": {
            "provider": "codex",
            "source_label": "codex",
            "source_file_sha256": "source-file",
            "source_file_name": "sessions.jsonl",
            "source_line": 1,
            "source_record_sha256": parent,
            "segment_record_sha256": example_id,
            "parser_revision": "parser/v1",
            "parent_record_sha256": parent,
            "model_tier": "tier1_frontier",
            "model_tier_basis": "test",
            "model_tier_registry_revision": "test/v1",
            "session_quality_id": "quality-1",
            "session_quality_scope": "source_session",
        },
        "lineage": {
            "parent_record_sha256": parent,
            "source_message_range": {"start": 0, "end": 3},
            "chunk_index": 0,
            "chunk_count": 1,
            "continuation_status": "complete",
        },
        "quality": {
            "model_tier": "tier1_frontier",
            "model_tier_basis": "test",
            "model_tier_registry_revision": "test/v1",
            "session_quality_id": "quality-1",
            "session_quality_scope": "source_session",
            "session_quality_gate": "candidate",
            "session_quality_flags": ["outcome_unverified"],
            "tool_families": ["shell"],
        },
        "privacy": {
            "eligible_for_training": False,
            "reason": "review-required",
            "structural_redactions": 0,
        },
    }


class HistoricalTrajectoryPilotTests(unittest.TestCase):
    def test_streams_partition_and_records_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            source = release / "tool_traces.candidate.jsonl"
            rows = [
                _source_row("example-1", "parent-1"),
                _source_row("example-2", "parent-1"),
                _source_row("example-3", "parent-2", malformed=True),
            ]
            raw = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
            source.write_bytes(raw)
            manifest = {
                "datasets": {
                    "tool_traces": {
                        "partitions": {
                            "candidate": {
                                "path": source.name,
                                "records": len(rows),
                                "bytes": len(raw),
                                "sha256": __import__("hashlib").sha256(raw).hexdigest(),
                            }
                        }
                    }
                }
            }
            (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "pilot"
            result = export_historical_tool_trajectory_pilot(
                release_dir=release,
                output_dir=output,
            )
            self.assertEqual(result["counts"]["input_records"], 3)
            self.assertEqual(result["counts"]["selected_records"], 2)
            self.assertEqual(result["counts"]["excluded_records"], 1)
            self.assertEqual(result["counts"]["final_answer_sft"], 1)
            self.assertEqual(result["final_answer_sft"]["decision_records"], 3)
            self.assertEqual(
                result["final_answer_sft"]["decision_reason_counts"][
                    "duplicate_prompt_completion"
                ],
                1,
            )
            self.assertEqual(result["decisions"]["reason_counts"]["event_call_observation_mismatch"], 1)
            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines() if line]
            validation = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines() if line]
            self.assertEqual(len(train) + len(validation), 2)
            self.assertEqual({row["lineage"]["parent_record_sha256"] for row in train + validation}, {"parent-1"})
            self.assertFalse(any(row["lineage"]["parent_record_sha256"] == "parent-2" for row in train + validation))

    def test_source_rows_keep_event_rich_review_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            source = release / "tool_traces.candidate.jsonl"
            row = _source_row("example-1", "parent-1")
            raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            source.write_bytes(raw)
            manifest = {"datasets": {"tool_traces": {"partitions": {"candidate": {
                "path": source.name, "records": 1, "bytes": len(raw),
                "sha256": __import__("hashlib").sha256(raw).hexdigest(),
            }}}}}
            (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "pilot"
            export_historical_tool_trajectory_pilot(release_dir=release, output_dir=output)
            emitted = json.loads((output / "train.jsonl").read_text().splitlines()[0])
            self.assertEqual(emitted["tool_contract"]["schema_status"], "not_observed")
            self.assertEqual(emitted["events_summary"]["action_count"], 1)
            self.assertEqual(emitted["events_summary"]["observation_count"], 1)
            self.assertIsInstance(emitted["events"][0], str)
            self.assertIsInstance(
                emitted["messages"][1]["tool_calls"][0]["function"]["arguments"],
                str,
            )
            self.assertFalse(emitted["privacy"]["eligible_for_training"])

            final_rows = [
                json.loads(line)
                for name in (
                    "final_answer_sft_train.jsonl",
                    "final_answer_sft_validation.jsonl",
                )
                for line in (output / name).read_text().splitlines()
                if line
            ]
            self.assertEqual(len(final_rows), 1)
            self.assertEqual(
                [message["role"] for message in final_rows[0]["messages"]],
                ["user", "assistant"],
            )
            self.assertFalse(any("tool_calls" in message for message in final_rows[0]["messages"]))
            self.assertFalse(final_rows[0]["training_authorized"])

    def test_final_answer_projection_excludes_provider_suggestion_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            source = release / "tool_traces.candidate.jsonl"
            rows = [
                _source_row("example-1", "parent-1"),
                _source_row("example-2", "parent-2", suggestion_mode=True),
            ]
            raw = b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
                for row in rows
            )
            source.write_bytes(raw)
            manifest = {
                "datasets": {
                    "tool_traces": {
                        "partitions": {
                            "candidate": {
                                "path": source.name,
                                "records": len(rows),
                                "bytes": len(raw),
                                "sha256": __import__("hashlib").sha256(raw).hexdigest(),
                            }
                        }
                    }
                }
            }
            (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "pilot"
            result = export_historical_tool_trajectory_pilot(
                release_dir=release,
                output_dir=output,
            )
            self.assertEqual(result["counts"]["selected_records"], 2)
            self.assertEqual(result["counts"]["final_answer_sft"], 1)
            self.assertEqual(
                result["final_answer_sft"]["decision_reason_counts"][
                    "provider_suggestion_mode_task"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
