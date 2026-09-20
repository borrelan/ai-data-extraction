import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from adapter_coverage import reconcile_adapter_coverage
from source_manifest import build_source_manifest


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def entry(
    relative: str,
    *,
    content: str,
    provider: str,
    root_label: str = "primary",
    source_class: str = "session_active",
):
    return {
        "provider": provider,
        "root_label": root_label,
        "source_class": source_class,
        "treatment": "candidate",
        "relative_path_sha256": digest(relative),
        "bytes": len(content.encode()),
        "file_sha256": digest(content),
    }


def snapshot_entry(
    relative: str,
    *,
    content: str,
    provider: str,
    root_label: str = "primary",
    source_class: str = "session_active",
):
    return {
        "provider": provider,
        "root_label": root_label,
        "source_class": source_class,
        "treatment": "candidate",
        "relative_path_sha256": digest(relative),
        "snapshot_relative_path_sha256": digest(f"home/{relative}"),
        "status": "stable",
        "bytes": len(content.encode()),
        "sha256": digest(content),
    }


class AdapterCoverageTests(unittest.TestCase):
    def test_exact_source_ref_wins_over_ambiguous_content_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = {
                "schema_version": "ai-data-extraction/source-inventory/v1",
                "files": [
                    entry("a.jsonl", content="same", provider="codex"),
                    entry("b.jsonl", content="same", provider="codex"),
                ],
            }
            snapshot = {
                "schema_version": "ai-data-extraction/source-snapshot/v1",
                "snapshot_revision": "snapshot-1",
                "files": [
                    snapshot_entry("a.jsonl", content="same", provider="codex"),
                    snapshot_entry("b.jsonl", content="same", provider="codex"),
                ],
            }
            inventory_path = root / "inventory.json"
            snapshot_path = root / "snapshot.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            manifest = build_source_manifest(inventory_path, snapshot_path)
            manifest_path = root / "source-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first = manifest["files"][0]
            ingress = root / "ingress.jsonl"
            ingress.write_text(
                json.dumps(
                    {
                        "source": "codex",
                        "source_class": "session_active",
                        "source_origin": {
                            "source_ref_sha256": first["source_ref_sha256"],
                            "source_file_sha256": f"sha256:{first['source_sha256']}",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = reconcile_adapter_coverage(
                source_manifest=manifest_path,
                ingress_jsonl=[ingress],
            )

        self.assertEqual(report["coverage"]["exact_source_ref_files"], 1)
        self.assertEqual(
            report["coverage"]["status_counts"]["ambiguous_digest_only"], 1
        )
        self.assertEqual(report["status"], "partial")

    def test_unrouted_and_identity_drift_are_not_called_unparsed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = {
                "schema_version": "ai-data-extraction/source-inventory/v1",
                "files": [
                    entry("unknown.jsonl", content="unknown", provider="new-agent", source_class="new_class"),
                    entry("drift.jsonl", content="same", provider="codex"),
                ],
            }
            snapshot = {
                "schema_version": "ai-data-extraction/source-snapshot/v1",
                "snapshot_revision": "snapshot-1",
                "files": [
                    snapshot_entry("unknown.jsonl", content="unknown", provider="new-agent", source_class="new_class"),
                    snapshot_entry("drift.jsonl", content="same", provider="codex", source_class="session_backup"),
                ],
            }
            inventory_path = root / "inventory.json"
            snapshot_path = root / "snapshot.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            manifest_path = root / "source-manifest.json"
            manifest_path.write_text(
                json.dumps(build_source_manifest(inventory_path, snapshot_path)),
                encoding="utf-8",
            )
            report = reconcile_adapter_coverage(source_manifest=manifest_path)

        statuses = report["coverage"]["status_counts"]
        self.assertEqual(statuses["unsupported_source_class"], 1)
        self.assertEqual(statuses["blocked_source_identity_status"], 1)
        self.assertEqual(statuses.get("unparsed_routed_source", 0), 0)


if __name__ == "__main__":
    unittest.main()
