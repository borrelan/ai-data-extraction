import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from source_manifest import (
    SourceAdmissionError,
    SourceManifestError,
    SourceManifestIndex,
    build_source_manifest,
    write_manifest,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def inventory_entry(
    relative: str,
    *,
    content: str,
    provider: str = "codex",
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
    provider: str = "codex",
    root_label: str = "primary",
    source_class: str = "session_active",
):
    return {
        "provider": provider,
        "root_label": root_label,
        "source_class": source_class,
        "treatment": "candidate",
        "relative_path_sha256": digest(relative),
        "snapshot_relative_path_sha256": digest(f"home/.codex/{relative}"),
        "status": "stable",
        "bytes": len(content.encode()),
        "sha256": digest(content),
    }


class SourceManifestTests(unittest.TestCase):
    def write_inputs(self, root: Path, inventory_files, snapshot_files):
        inventory = {
            "schema_version": "ai-data-extraction/source-inventory/v1",
            "files": inventory_files,
        }
        snapshot = {
            "schema_version": "ai-data-extraction/source-snapshot/v1",
            "snapshot_revision": "snapshot-1",
            "files": snapshot_files,
        }
        inventory_path = root / "inventory.json"
        snapshot_path = root / "snapshot.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
        return inventory_path, snapshot_path

    def test_snapshot_is_authoritative_and_drift_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_path, snapshot_path = self.write_inputs(
                root,
                [inventory_entry("same.jsonl", content="same"), inventory_entry("changed.jsonl", content="old")],
                [snapshot_entry("same.jsonl", content="same"), snapshot_entry("changed.jsonl", content="new")],
            )

            manifest = build_source_manifest(inventory_path, snapshot_path)

        self.assertEqual(manifest["authority"], "stable_snapshot")
        self.assertEqual(manifest["counts"]["joined_files"], 2)
        self.assertEqual(manifest["counts"]["bound_files"], 1)
        self.assertEqual(manifest["counts"]["changed_since_inventory"], 1)
        changed = next(
            row
            for row in manifest["files"]
            if row["snapshot_status"] != "bound"
        )
        self.assertEqual(changed["source_bytes"], len(b"new"))
        self.assertEqual(changed["source_sha256"], digest("new"))

    def test_unmatched_source_identity_is_not_silently_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_path, snapshot_path = self.write_inputs(
                root,
                [inventory_entry("inventory-only.jsonl", content="old")],
                [snapshot_entry("snapshot-only.jsonl", content="new")],
            )
            manifest = build_source_manifest(inventory_path, snapshot_path)

        self.assertEqual(manifest["counts"]["unaccounted_files"], 2)
        self.assertEqual(manifest["counts"]["inventory_only"], 1)
        self.assertEqual(manifest["counts"]["snapshot_only"], 1)

    def test_duplicate_identity_fails_closed_and_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = inventory_entry("duplicate.jsonl", content="one")
            inventory_path, snapshot_path = self.write_inputs(
                root,
                [first, dict(first)],
                [snapshot_entry("duplicate.jsonl", content="one")],
            )
            with self.assertRaises(SourceManifestError):
                build_source_manifest(inventory_path, snapshot_path)

            output = root / "nested" / "source-manifest.json"
            write_manifest(output, {"schema_version": "test"})
            self.assertEqual(json.loads(output.read_text())["schema_version"], "test")

    def test_index_accepts_legacy_adapter_labels_and_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = "new session bytes"
            inventory_path, snapshot_path = self.write_inputs(
                root,
                [
                    inventory_entry(
                        "prime/session.jsonl",
                        content="old session bytes",
                        provider="prime-agent",
                        source_class="session_active",
                    )
                ],
                [
                    snapshot_entry(
                        "prime/session.jsonl",
                        content=content,
                        provider="prime-agent",
                        source_class="session_active",
                    )
                ],
            )
            index = SourceManifestIndex(build_source_manifest(inventory_path, snapshot_path))
            admission = index.admit(
                source_sha256=digest(content),
                provider="prime-agent",
                root_label="primary",
                source_class="prime_session_active",
            )

        self.assertEqual(
            admission["source_snapshot_status"],
            "snapshot_authoritative_changed_since_inventory",
        )
        self.assertEqual(admission["source_class"], "session_active")

    def test_index_admits_exact_relative_path_when_content_digest_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = "same bytes"
            inventory_path, snapshot_path = self.write_inputs(
                root,
                [
                    inventory_entry(
                        "first/session.jsonl",
                        content=content,
                        provider="prime-agent",
                    ),
                    inventory_entry(
                        "second/session.jsonl",
                        content=content,
                        provider="prime-agent",
                    ),
                ],
                [
                    snapshot_entry(
                        "first/session.jsonl",
                        content=content,
                        provider="prime-agent",
                    ),
                    snapshot_entry(
                        "second/session.jsonl",
                        content=content,
                        provider="prime-agent",
                    ),
                ],
            )
            index = SourceManifestIndex(build_source_manifest(inventory_path, snapshot_path))

            admission = index.admit_path(
                source_sha256=digest(content),
                relative_path_sha256=digest("second/session.jsonl"),
                provider="prime-agent",
                root_label="primary",
                source_class="session_active",
            )

        self.assertEqual(
            admission["relative_path_sha256"],
            digest("second/session.jsonl"),
        )
        with self.assertRaises(SourceAdmissionError):
            index.admit(
                source_sha256=digest(content),
                provider="prime-agent",
                root_label="primary",
                source_class="session_active",
            )

    def test_index_rejects_historical_class_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = "same bytes"
            inventory_path, snapshot_path = self.write_inputs(
                root,
                [
                    inventory_entry(
                        "prime/session.jsonl",
                        content=content,
                        provider="prime-agent",
                        source_class="prime_session_active",
                    )
                ],
                [
                    snapshot_entry(
                        "prime/session.jsonl",
                        content=content,
                        provider="prime-agent",
                        source_class="session_active",
                    )
                ],
            )
            index = SourceManifestIndex(build_source_manifest(inventory_path, snapshot_path))

            with self.assertRaises(SourceAdmissionError):
                index.admit(
                    source_sha256=digest(content),
                    provider="prime-agent",
                    root_label="primary",
                    source_class="prime_session_active",
                )


if __name__ == "__main__":
    unittest.main()
