import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from coverage_report import build_coverage_report, write_report


class CoverageReportTests(unittest.TestCase):
    def test_reconciles_direct_source_hashes_and_reports_unmatched_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            hash_a = hashlib.sha256(b"source-a").hexdigest()
            hash_b = hashlib.sha256(b"source-b").hexdigest()
            source.write_text(
                json.dumps(
                    {
                        "source": "provider-a",
                        "source_class": "session_active",
                        "source_origin": {"source_file_sha256": f"sha256:{hash_a}"},
                        "messages": [
                            {"role": "user", "content": "secret prompt"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/source-inventory/v1",
                        "hash_mode": "candidates",
                        "counts": {"files": 2, "bytes": 2, "hashed_files": 2, "errors": 0},
                        "errors": [],
                        "files": [
                            {
                                "provider": "provider-a",
                                "source_class": "session_active",
                                "treatment": "candidate",
                                "bytes": 1,
                                "file_sha256": hash_a,
                            },
                            {
                                "provider": "provider-b",
                                "source_class": "session_backup",
                                "treatment": "candidate",
                                "bytes": 1,
                                "file_sha256": hash_b,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            preflight = root / "preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/preflight/v1",
                        "status": "complete",
                        "counts": {"files": 1, "records": 1, "valid_json": 1},
                        "inputs": [
                            {
                                "name": source.name,
                                "sha256": raw_hash,
                                "bytes": source.stat().st_size,
                                "parser_status": "complete",
                                "stable_size": True,
                                "counts": {"records": 1, "valid_json": 1},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            build = root / "build-manifest.json"
            build.write_text(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/v1",
                        "builder_version": "test",
                        "counts": {
                            "input_records": 1,
                            "normalized_records": 1,
                            "rejected": 0,
                            "skipped_training_lane_records": 0,
                        },
                        "policy": {"privacy_mode": "heuristic", "privacy_approved": False},
                        "inputs": [
                            {"name": source.name, "sha256": raw_hash, "bytes": source.stat().st_size}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build_coverage_report(
                inventory=inventory,
                raw_artifacts=[source],
                preflight_manifests=[preflight],
                build_manifests=[build],
            )
            output = root / "coverage.json"
            write_report(output, report)

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["inventory"]["candidate"]["matched_files"], 1)
        self.assertEqual(report["inventory"]["candidate"]["unmatched_files"], 1)
        self.assertTrue(report["raw_preflight_bindings"][0]["preflight_bound"])
        self.assertTrue(report["build_manifests"][0]["all_inputs_bound"])
        encoded = json.dumps(report)
        self.assertNotIn("secret prompt", encoded)

    def test_source_manifest_hash_can_bind_without_reading_source_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "a" * 64
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/source-inventory/v1",
                        "errors": [],
                        "files": [
                            {
                                "provider": "pi-agent",
                                "source_class": "session_active",
                                "treatment": "candidate",
                                "bytes": 7,
                                "file_sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ingress = root / "ingress.json"
            ingress.write_text(
                json.dumps(
                    {
                        "schema_version": "ai-data-extraction/agent-session-ingress/v1",
                        "source_sessions": [
                            {
                                "provider": "pi-agent",
                                "source_class": "session_active",
                                "source_file_sha256": f"sha256:{digest}",
                                "training_lane": "optional_alt",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = build_coverage_report(
                inventory=inventory,
                ingress_manifests=[ingress],
            )

        self.assertEqual(report["inventory"]["candidate"]["matched_files"], 1)
        self.assertEqual(report["inventory"]["candidate"]["unmatched_files"], 0)
        self.assertEqual(report["ingress_manifests"][0]["entries_with_source_hash"], 1)


if __name__ == "__main__":
    unittest.main()
