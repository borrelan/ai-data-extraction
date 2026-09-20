import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from release_gate import (
    REQUIRED_REVIEW_DIMENSIONS,
    canonical_json_bytes,
    materialize_reviewed_pilot,
)


def _row(unit: str, parent: str, dataset: str) -> dict[str, object]:
    return {
        "schema_version": "ai-data-extraction/v1",
        "example_id": unit,
        "dataset": dataset,
        "split": "train",
        "messages": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "response"},
        ],
        "metadata": {"parent_record_sha256": parent, "quality_gate": "candidate"},
        "lineage": {"parent_record_sha256": parent},
    }


def _write_release(root: Path, rows_by_dataset: dict[str, list[dict[str, object]]]) -> tuple[Path, str]:
    release = root / "release"
    release.mkdir()
    datasets: dict[str, object] = {}
    for dataset, rows in rows_by_dataset.items():
        path = release / f"{dataset}.candidate.jsonl"
        raw = b"".join(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            for row in rows
        )
        path.write_bytes(raw)
        datasets[dataset] = {
            "partitions": {
                "candidate": {
                    "path": path.name,
                    "bytes": len(raw),
                    "records": len(rows),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            }
        }
    manifest = {
        "schema_version": "ai-data-extraction/release-gate/v1",
        "privacy": {"training_authorized": False},
        "datasets": datasets,
    }
    manifest_path = release / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return release, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _review(unit: str, release_sha: str, decision: str = "accepted") -> dict[str, object]:
    dimensions = {
        name: ("not_applicable" if name == "contamination" else "pass")
        for name in REQUIRED_REVIEW_DIMENSIONS
    }
    return {
        "unit_id": unit,
        "review_id": f"review-{unit}",
        "manifest_revision": "reviews/test/v1",
        "snapshot_revision": "snapshot-test",
        "reviewer_id": "test-reviewer",
        "rubric_revision": "quality/v1",
        "reviewed_at": "2026-09-17T00:00:00Z",
        "decision": decision,
        "dimensions": dimensions,
        "evidence_ids": [unit, f"sha256:{release_sha}"],
    }


def _write_review_manifest(path: Path, reviews: list[dict[str, object]]) -> None:
    manifest = {
        "manifest_revision": "reviews/test/v1",
        "snapshot_revision": "snapshot-test",
        "rubric_revision": "quality/v1",
        "reviews": reviews,
    }
    artifact = {
        "schema_version": "agentir/hardening/quality-review-manifest/v1",
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "manifest": manifest,
    }
    path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")


class ReviewedPilotTests(unittest.TestCase):
    def test_materializes_accepted_rows_byte_for_byte_and_preserves_reviews(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sft = _row("sft-unit", "parent-sft", "sft")
            tool = _row("tool-unit", "parent-tool", "tool_traces")
            release, release_sha = _write_release(
                root, {"sft": [sft], "tool_traces": [tool]}
            )
            review_path = root / "review_manifest.json"
            _write_review_manifest(
                review_path,
                [_review("sft-unit", release_sha), _review("tool-unit", release_sha)],
            )
            output = root / "pilot"

            result = materialize_reviewed_pilot(release, review_path, output)

            self.assertFalse(result["privacy"]["training_authorized"])
            self.assertTrue(result["parent_disjoint"])
            self.assertEqual(result["counts"]["pilot_rows"], 2)
            self.assertEqual(
                (output / "sft.jsonl").read_bytes(),
                (release / "sft.candidate.jsonl").read_bytes(),
            )
            self.assertEqual(
                (output / "tool_traces.jsonl").read_bytes(),
                (release / "tool_traces.candidate.jsonl").read_bytes(),
            )
            self.assertEqual(len((output / "reviews.jsonl").read_text().splitlines()), 2)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertFalse((output / ".staging").exists())

    def test_tampered_review_manifest_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, release_sha = _write_release(
                root, {"sft": [_row("sft-unit", "parent-sft", "sft")]}
            )
            review_path = root / "review_manifest.json"
            _write_review_manifest(review_path, [_review("sft-unit", release_sha)])
            artifact = json.loads(review_path.read_text())
            artifact["manifest_sha256"] = "0" * 64
            review_path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "digest"):
                materialize_reviewed_pilot(release, review_path, root / "pilot")
            self.assertFalse((root / "pilot").exists())

    def test_missing_accepted_unit_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, release_sha = _write_release(
                root, {"sft": [_row("present", "parent-sft", "sft")]}
            )
            review_path = root / "review_manifest.json"
            _write_review_manifest(review_path, [_review("missing", release_sha)])

            with self.assertRaisesRegex(ValueError, "not found"):
                materialize_reviewed_pilot(
                    release, review_path, root / "pilot", datasets=("sft",)
                )
            self.assertFalse((root / "pilot" / "manifest.json").exists())

    def test_parent_collision_is_not_a_valid_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, release_sha = _write_release(
                root,
                {
                    "sft": [_row("sft-unit", "same-parent", "sft")],
                    "tool_traces": [_row("tool-unit", "same-parent", "tool_traces")],
                },
            )
            review_path = root / "review_manifest.json"
            _write_review_manifest(
                review_path,
                [_review("sft-unit", release_sha), _review("tool-unit", release_sha)],
            )

            with self.assertRaisesRegex(ValueError, "parent-disjoint"):
                materialize_reviewed_pilot(release, review_path, root / "pilot")


if __name__ == "__main__":
    unittest.main()
