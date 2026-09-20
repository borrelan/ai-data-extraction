import json
import tempfile
import unittest
from pathlib import Path

from inventory_sources import (
    RootSpec,
    classify_file,
    classify_root_file,
    inventory_roots,
    write_manifest,
)


class InventoryClassificationTests(unittest.TestCase):
    def test_classifies_active_backup_and_snapshot_boundaries(self):
        self.assertEqual(
            classify_file("codex", Path("sessions/2026/rollout-a.jsonl")),
            ("session_active", "candidate"),
        )
        self.assertEqual(
            classify_file("codex", Path("sessions/.backups/rollout-a.jsonl.backup")),
            ("session_backup", "candidate"),
        )
        self.assertEqual(
            classify_file("codex", Path("packages/standalone/model.bin")),
            ("binary_or_package", "excluded"),
        )
        self.assertEqual(
            classify_file("opencode", Path("snapshot/abc/file.txt")),
            ("workspace_snapshot", "review"),
        )
        self.assertEqual(
            classify_file("opencode", Path("storage/message/ses/msg.json")),
            ("conversation_store", "candidate"),
        )
        self.assertEqual(
            classify_file(
                "claude",
                Path("projects/project-a/subagents/agent-side.jsonl"),
            ),
            ("subagent_session", "candidate"),
        )

    def test_root_aware_classification_is_shared_by_inventory_and_snapshot(self):
        self.assertEqual(
            classify_root_file(
                "oh-my-pi",
                "backups",
                Path("agent/sessions/2026/rollout.jsonl"),
            ),
            ("session_backup", "candidate"),
        )


class InventoryManifestTests(unittest.TestCase):
    def test_inventory_is_metadata_only_and_hashes_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "codex"
            (root / "sessions/.backups").mkdir(parents=True)
            (root / "sessions/rollout-a.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
            (root / "sessions/.backups/rollout-b.jsonl.backup").write_text(
                "private content\n", encoding="utf-8"
            )
            (root / "cache.bin").write_bytes(b"cache")
            manifest = inventory_roots(
                [RootSpec("codex", "test", root)],
                hash_mode="candidates",
            )
            self.assertEqual(manifest["counts"]["files"], 3)
            self.assertEqual(manifest["counts"]["hashed_files"], 2)
            encoded = json.dumps(manifest)
            self.assertNotIn("private content", encoded)
            self.assertNotIn(str(root), encoded)

    def test_write_manifest_creates_parent_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "inventory.json"
            write_manifest(path, {"schema_version": "test"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], "test")


if __name__ == "__main__":
    unittest.main()
