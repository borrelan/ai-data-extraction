import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_archived_salvage_pilot import build_archived_salvage_pilot


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


def _trainer_row(example_id: str, split: str, *, tool: bool = False) -> dict[str, object]:
    messages: list[dict[str, object]] = [{"role": "user", "content": "request"}]
    row: dict[str, object] = {
        "schema_version": "ai-data-extraction/trainer-example/v1",
        "example_id": example_id,
        "split": split,
        "messages": messages,
    }
    if tool:
        row["messages"] = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-tool",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-tool", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        row["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    else:
        row["messages"].append({"role": "assistant", "content": "response"})
    return row


def _lineage(example_id: str, parent: str, dataset: str, split: str) -> dict[str, object]:
    return {
        "schema_version": "ai-data-extraction/unified-trainer-lineage/v1",
        "dataset": dataset,
        "example_id": example_id,
        "split": split,
        "parent_record_sha256": parent,
        "provider": "codex",
        "quality_tier": "silver_sft" if dataset == "sft" else "gold",
        "source_artifact": "fixture",
        "source_artifact_manifest_sha256": "fixture-manifest",
    }


def _historical_row(example_id: str, parent: str, split: str) -> dict[str, object]:
    action = {
        "schema_version": "ai-data-extraction/event/v1",
        "event_id": f"action-{example_id}",
        "call_id": f"call-{example_id}",
        "kind": "action",
        "name": "shell",
        "input": {"command": ["true"]},
        "ordinal": 0,
    }
    observation = {
        "schema_version": "ai-data-extraction/event/v1",
        "event_id": f"observation-{example_id}",
        "call_id": f"call-{example_id}",
        "kind": "observation",
        "name": "unknown",
        "output": "ok",
        "ordinal": 1,
    }
    return {
        "schema_version": "ai-data-extraction/historical-tool-trajectory-pilot/v1",
        "example_id": example_id,
        "split": split,
        "provider": "codex",
        "agent": "codex",
        "model_tier": "tier1_frontier",
        "quality_tier": "candidate",
        "quality_reason": ["outcome_unverified", "tool_schema_not_observed"],
        "privacy": {"eligible_for_training": False, "state": "review_required"},
        "tool_contract": {
            "schema_status": "not_observed",
            "verification_status": "not_observed",
            "reward_status": "not_exported",
        },
        "session_quality": {"id": "quality-1", "scope": "source_session"},
        "messages": [{"role": "user", "content": "run"}],
        "events": [json.dumps(action), json.dumps(observation)],
        "events_summary": {
            "event_count": 2,
            "action_count": 1,
            "observation_count": 1,
            "matched_call_count": 1,
        },
        "lineage": {
            "source_dataset": "tool_traces",
            "source_example_id": example_id,
            "parent_record_sha256": parent,
            "source_record_sha256": parent,
            "segment_record_sha256": example_id,
            "source_file_name": "sessions.jsonl",
            "source_file_sha256": "source",
            "source_line": 1,
            "source_message_range_json": "{\"start\":0,\"end\":1}",
            "chunk_index": 0,
            "chunk_count": 1,
            "continuation_status": "complete",
        },
        "tags": ["trajectory:historical-review"],
        "tool_families": ["shell"],
    }


def _write_unified(root: Path) -> Path:
    root.mkdir()
    files: dict[str, dict[str, object]] = {}
    files["train.jsonl"] = _write_jsonl(
        root / "train.jsonl", [_trainer_row("sft-1", "train")]
    )
    files["validation.jsonl"] = _write_jsonl(root / "validation.jsonl", [])
    files["tool_train.jsonl"] = _write_jsonl(
        root / "tool_train.jsonl", [_trainer_row("tool-1", "train", tool=True)]
    )
    files["tool_validation.jsonl"] = _write_jsonl(root / "tool_validation.jsonl", [])
    files["lineage.jsonl"] = _write_jsonl(
        root / "lineage.jsonl",
        [
            _lineage("sft-1", "parent-1", "sft", "train"),
            _lineage("tool-1", "parent-2", "tool_sft", "train"),
        ],
    )
    files["decisions.jsonl"] = _write_jsonl(
        root / "decisions.jsonl",
        [
            {"example_id": "sft-1", "decision": "selected", "reasons": ["selected"]},
            {"example_id": "tool-1", "decision": "selected", "reasons": ["selected"]},
        ],
    )
    (root / "manifest.json").write_bytes(
        json.dumps(
            {
                "schema_version": "ai-data-extraction/unified-trainer-pilot/v1",
                "training_authorized": False,
                "files": files,
                "counts": {"input_rows": 2, "selected_rows": 2},
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    return root


def _write_historical(root: Path) -> Path:
    root.mkdir()
    train = [_historical_row("hist-1", "parent-2", "train")]
    validation = [_historical_row("hist-2", "parent-3", "validation")]
    files = {
        "train.jsonl": _write_jsonl(root / "train.jsonl", train),
        "validation.jsonl": _write_jsonl(root / "validation.jsonl", validation),
        "decisions.jsonl": _write_jsonl(
            root / "decisions.jsonl",
            [
                {"source_example_id": "hist-1", "selected": True, "reasons": ["selected"]},
                {"source_example_id": "hist-2", "selected": True, "reasons": ["selected"]},
            ],
        ),
    }
    (root / "manifest.json").write_bytes(
        json.dumps(
            {
                "schema_version": "ai-data-extraction/historical-tool-trajectory-pilot/v1",
                "training_authorized": False,
                "files": files,
                "counts": {"input_records": 2, "selected_records": 2},
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    return root


class ArchivedSalvagePilotTests(unittest.TestCase):
    def test_composes_trainer_and_historical_lanes_with_global_parent_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unified = _write_unified(root / "unified")
            historical = _write_historical(root / "historical")
            output = root / "output"
            manifest = build_archived_salvage_pilot(
                unified_dir=unified,
                historical_dir=historical,
                output_dir=output,
            )
            self.assertEqual(manifest["counts"]["sft_train"], 1)
            self.assertEqual(manifest["counts"]["tool_sft_train"], 1)
            self.assertEqual(manifest["counts"]["tool_trajectory_train"] + manifest["counts"]["tool_trajectory_validation"], 2)
            self.assertEqual(manifest["counts"]["decision_records"], 4)
            historical_rows = [
                json.loads(line)
                for line in (output / "tool_trajectory_train.jsonl").read_text().splitlines()
                + (output / "tool_trajectory_validation.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["split"] for row in historical_rows if row["example_id"] == "hist-1"}, {"train"})
            self.assertEqual(
                json.loads((output / "lineage.jsonl").read_text().splitlines()[-1])["schema_version"],
                "ai-data-extraction/archived-salvage-lineage/v1",
            )
            self.assertFalse(manifest["training_authorized"])


if __name__ == "__main__":
    unittest.main()
