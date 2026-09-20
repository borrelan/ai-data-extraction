import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from release_gate import DATASET_FILES, gate_corpus


def canonical_record(*, marker: bool = False, secret: bool = False) -> dict[str, object]:
    assistant = "answer"
    if marker:
        assistant = "answer with <analysis> marker"
    if secret:
        assistant = "Bearer abcdefghijklmnopqrstuvwxyz123456"
    return {
        "schema_version": "ai-data-extraction/v1",
        "example_id": "example-1",
        "dataset": "sft",
        "split": "train",
        "messages": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source_file_sha256": "source-digest",
            "source_manifest_revision": "manifest-revision",
            "source_snapshot_status": "bound",
            "parser_revision": "parser-revision",
            "source_message_range": {"start": 0, "end": 1},
            "session_id": "session-1",
            "model_tier": "tier1_frontier",
            "training_lane": "primary",
            "quality_gate": "candidate",
        },
        "quality": {
            "session_quality_gate": "candidate",
            "model_tier": "tier1_frontier",
            "has_tools": False,
        },
        "privacy": {"eligible_for_training": False},
    }


def write_canonical(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    raw = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    path.write_bytes(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "records": len(rows)}


class ReleaseGateTests(unittest.TestCase):
    def test_gate_partitions_rows_and_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "canonical"
            output_dir = root / "release"
            input_dir.mkdir()
            output_entries: dict[str, object] = {}
            rows = [
                canonical_record(),
                canonical_record(marker=True),
                canonical_record(secret=True),
            ]
            for label, filename in DATASET_FILES.items():
                entries = rows if label == "sft" else []
                output_entries[filename] = write_canonical(input_dir / filename, entries)
            manifest = {
                "schema_version": "ai-data-extraction/v1",
                "builder_version": "1.3.5",
                "policy": {"privacy_approved": False, "parser_revision": "parser-revision"},
                "outputs": output_entries,
            }
            (input_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            result = gate_corpus(input_dir, output_dir, datasets=["sft"])

            self.assertFalse(result["privacy"]["training_authorized"])
            self.assertEqual(result["counts"]["output_records"], 3)
            self.assertEqual(result["datasets"]["sft"]["partition_counts"], {
                "candidate": 1,
                "quarantine": 1,
                "review_required": 1,
            })
            self.assertEqual(
                len((output_dir / "sft.candidate.jsonl").read_bytes().splitlines()), 1
            )
            self.assertEqual(
                len((output_dir / "sft.review_required.jsonl").read_bytes().splitlines()), 1
            )
            self.assertEqual(
                len((output_dir / "sft.quarantine.jsonl").read_bytes().splitlines()), 1
            )
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "decisions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
