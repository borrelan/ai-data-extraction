import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trainer_export import (
    TRAINER_EXAMPLE_SCHEMA,
    UNIFIED_PILOT_SCHEMA,
    canonical_json_bytes,
    export_unified_training_pilot,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _trainer_row(example_id: str, *, tool: bool = False) -> dict[str, object]:
    messages: list[dict[str, object]] = [{"role": "user", "content": "request"}]
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
                            "function": {"name": "read", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "content": "observed", "tool_call_id": "call-1"},
            ]
        )
    messages.append({"role": "assistant", "content": "done"})
    row: dict[str, object] = {
        "schema_version": TRAINER_EXAMPLE_SCHEMA,
        "example_id": example_id,
        "split": "train",
        "messages": messages,
    }
    if tool:
        row["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    return {
        "records": len(rows),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_artifact(
    root: Path,
    *,
    schema: str,
    authorized: bool,
    rows_by_file: dict[str, list[dict[str, object]]],
    quality_tier: str | None = None,
) -> Path:
    root.mkdir()
    files: dict[str, dict[str, object]] = {}
    lineage: list[dict[str, object]] = []
    for filename, rows in rows_by_file.items():
        files[filename] = _write_jsonl(root / filename, rows)
        if filename in {"train.jsonl", "validation.jsonl", "sft.jsonl", "tool_sft.jsonl"}:
            for row in rows:
                lineage.append(
                    {
                        "example_id": row["example_id"],
                        "parent_record_sha256": _digest(f"parent:{row['example_id']}"),
                        "source_row_sha256": _digest(f"row:{row['example_id']}"),
                    }
                )
    files["lineage.jsonl"] = _write_jsonl(root / "lineage.jsonl", lineage)
    manifest: dict[str, object] = {
        "schema_version": schema,
        "training_authorized": authorized,
        "files": files,
    }
    if quality_tier is not None:
        manifest["quality"] = {"tier": quality_tier}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return root


class UnifiedTrainerPilotTests(unittest.TestCase):
    def test_combines_sources_with_global_parent_split_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_a = _write_artifact(
                root / "silver-a",
                schema="ai-data-extraction/silver-sft-pilot/v1",
                authorized=False,
                quality_tier="silver_sft",
                rows_by_file={
                    "train.jsonl": [_trainer_row("silver-a")],
                    "validation.jsonl": [_trainer_row("silver-b")],
                    "decisions.jsonl": [],
                },
            )
            silver_b = _write_artifact(
                root / "silver-b",
                schema="ai-data-extraction/silver-sft-pilot/v1",
                authorized=False,
                quality_tier="silver_sft",
                rows_by_file={
                    "train.jsonl": [_trainer_row("silver-c")],
                    "validation.jsonl": [],
                    "decisions.jsonl": [],
                },
            )
            gold_dialogue = _write_artifact(
                root / "gold-dialogue",
                schema="ai-data-extraction/trainer-export/v1",
                authorized=True,
                rows_by_file={
                    "sft.jsonl": [_trainer_row("gold-sft")],
                    "tool_sft.jsonl": [],
                },
            )
            gold_tool = _write_artifact(
                root / "gold-tool",
                schema="ai-data-extraction/trainer-export/v1",
                authorized=True,
                rows_by_file={
                    "sft.jsonl": [],
                    "tool_sft.jsonl": [_trainer_row("gold-tool", tool=True)],
                },
            )

            manifest = export_unified_training_pilot(
                silver_pilot_dirs=[silver_a, silver_b],
                gold_dialogue_dir=gold_dialogue,
                gold_tool_dir=gold_tool,
                output_dir=root / "unified",
            )

            self.assertEqual(manifest["schema_version"], UNIFIED_PILOT_SCHEMA)
            self.assertFalse(manifest["training_authorized"])
            self.assertEqual(manifest["counts"]["sft_selected_rows"], 4)
            self.assertEqual(manifest["counts"]["tool_selected_rows"], 1)
            self.assertEqual(manifest["counts"]["excluded_duplicate_rows"], 0)
            lineage = [
                json.loads(line)
                for line in (root / "unified" / "lineage.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({row["parent_record_sha256"] for row in lineage}), 5)
            self.assertEqual(
                {row["quality_tier"] for row in lineage}, {"silver_sft", "gold"}
            )
            decisions = [
                json.loads(line)
                for line in (root / "unified" / "decisions.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(row["decision"] == "selected" for row in decisions))

    def test_refuses_tampered_input_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = _write_artifact(
                root / "silver",
                schema="ai-data-extraction/silver-sft-pilot/v1",
                authorized=False,
                quality_tier="silver_sft",
                rows_by_file={
                    "train.jsonl": [_trainer_row("silver")],
                    "validation.jsonl": [],
                    "decisions.jsonl": [],
                },
            )
            (silver / "train.jsonl").write_text(
                (silver / "train.jsonl").read_text() + "\n",
                encoding="utf-8",
            )
            gold_dialogue = _write_artifact(
                root / "gold-dialogue",
                schema="ai-data-extraction/trainer-export/v1",
                authorized=True,
                rows_by_file={"sft.jsonl": [], "tool_sft.jsonl": []},
            )
            gold_tool = _write_artifact(
                root / "gold-tool",
                schema="ai-data-extraction/trainer-export/v1",
                authorized=True,
                rows_by_file={"sft.jsonl": [], "tool_sft.jsonl": []},
            )
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                export_unified_training_pilot(
                    silver_pilot_dirs=[silver],
                    gold_dialogue_dir=gold_dialogue,
                    gold_tool_dir=gold_tool,
                    output_dir=root / "unified",
                )


if __name__ == "__main__":
    unittest.main()
