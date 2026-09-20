import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_multi_provider_salvage_pilot import (
    INPUT_SCHEMA,
    build_multi_provider_salvage_pilot,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _row(example_id: str, parent: str, *, tier: str, tool: bool = False) -> dict[str, object]:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "Inspect the bounded change."},
    ]
    events: list[dict[str, object]] = []
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
                {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
                {"role": "assistant", "content": "The result is recorded."},
            ]
        )
        events = [
            {
                "schema_version": "ai-data-extraction/event/v1",
                "event_id": "event-action-1",
                "ordinal": 0,
                "kind": "action",
                "call_id": "call-1",
                "name": "read_file",
                "input": {},
            },
            {
                "schema_version": "ai-data-extraction/event/v1",
                "event_id": "event-observation-1",
                "ordinal": 1,
                "kind": "observation",
                "call_id": "call-1",
                "name": "read_file",
                "output": "ok",
            },
        ]
    else:
        messages.append({"role": "assistant", "content": "The result is recorded."})
    return {
        "schema_version": "ai-data-extraction/v1",
        "example_id": example_id,
        "dataset": "sft",
        "split": "train",
        "tags": ["task:debugging"],
        "messages": messages,
        "events": events,
        "metadata": {
            "provider": "fixture-provider",
            "source_label": "fixture-agent",
            "model_tier": tier,
            "training_lane": "primary" if tier == "tier1_frontier" else "optional_alt",
            "source_file_name": "sessions.jsonl",
            "source_file_sha256": "fixture-source",
            "source_origin": {
                "source_file_name": "session-001.jsonl",
                "source_file_sha256": "fixture-origin-source",
                "session_id": "session-001",
            },
            "source_line": 1,
            "parent_record_sha256": parent,
            "session_quality_id": f"quality-{parent}",
            "continuation_status": "complete",
            "chunk_index": 0,
            "chunk_count": 1,
        },
        "lineage": {
            "parent_record_sha256": parent,
            "source_message_range": {"start": 0, "end": len(messages) - 1},
            "continuation_status": "complete",
            "chunk_index": 0,
            "chunk_count": 1,
        },
        "quality": {
            "status": "review",
            "has_tools": tool,
            "tool_families": ["filesystem"] if tool else [],
            "session_quality_gate": "candidate",
            "session_quality_flags": ["outcome_unverified"],
            "session_quality_id": f"quality-{parent}",
            "model_tier": tier,
        },
        "privacy": {"eligible_for_training": False, "structural_redactions": 0},
    }


def _write_source(root: Path, rows: list[dict[str, object]]) -> Path:
    source = root / "sessions.jsonl"
    raw = b"".join(_canonical(row) + b"\n" for row in rows)
    source.write_bytes(raw)
    manifest = root / "source-manifest.json"
    manifest.write_bytes(
        _canonical(
            {
                "schema_version": "fixture/source/v1",
                "outputs": {
                    "sft.jsonl": {
                        "path": "sessions.jsonl",
                        "records": len(rows),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                },
            }
        )
        + b"\n"
    )
    return source


class MultiProviderSalvagePilotTests(unittest.TestCase):
    def test_emits_tiered_sft_and_separate_tool_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(
                root,
                [
                    _row("dialogue-long", "parent-1", tier="tier1_frontier"),
                    _row("dialogue-duplicate", "parent-1", tier="tier1_frontier"),
                    _row("tool-local", "parent-2", tier="tier3_local", tool=True),
                ],
            )
            raw = source.read_bytes()
            manifest = source.with_name("source-manifest.json")
            spec = root / "inputs.json"
            spec.write_bytes(
                _canonical(
                    {
                        "schema_version": INPUT_SCHEMA,
                        "sources": [
                            {
                                "source_id": "fixture",
                                "path": str(source),
                                "manifest_path": str(manifest),
                                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                "manifest_keys": ["outputs", "sft.jsonl"],
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "records": 3,
                                "bytes": len(raw),
                                "provider_scope": ["fixture-provider"],
                            }
                        ],
                    }
                )
                + b"\n"
            )

            output = root / "pilot"
            result = build_multi_provider_salvage_pilot(input_spec=spec, output_dir=output)

            self.assertFalse(result["training_authorized"])
            self.assertEqual(result["counts"]["sft_selected"], 1)
            self.assertEqual(result["counts"]["sft_tier1_train"] + result["counts"]["sft_tier1_validation"], 1)
            self.assertEqual(result["counts"]["tool_review_selected"], 1)
            self.assertEqual(result["counts"]["decision_records"], 3)
            sft_files = list(output.glob("sft_*.jsonl"))
            sft_rows = [
                json.loads(line)
                for path in sft_files
                for line in path.read_text().splitlines()
                if line
            ]
            self.assertEqual(len(sft_rows), 1)
            self.assertEqual(set(sft_rows[0]), {"schema_version", "example_id", "split", "messages"})
            tool_rows = [
                json.loads(line)
                for path in (output / "tool_review_train.jsonl", output / "tool_review_validation.jsonl")
                for line in path.read_text().splitlines()
                if line
            ]
            self.assertEqual(len(tool_rows), 1)
            self.assertEqual(tool_rows[0]["tool_schema_status"], "not_observed")
            self.assertEqual(json.loads(tool_rows[0]["events_summary_json"])["matched_call_count"], 1)
            lineage = [
                json.loads(line)
                for line in (output / "lineage.jsonl").read_text().splitlines()
                if line
            ]
            self.assertTrue(lineage)
            self.assertEqual(lineage[0]["source_origin_file_name"], "session-001.jsonl")
            self.assertEqual(lineage[0]["source_origin_file_sha256"], "fixture-origin-source")
            self.assertEqual(json.loads(lineage[0]["source_origin_json"])["session_id"], "session-001")


if __name__ == "__main__":
    unittest.main()
