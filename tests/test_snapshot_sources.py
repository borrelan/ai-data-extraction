import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from inventory_sources import RootSpec
from snapshot_sources import snapshot_roots


class SnapshotSourceTests(unittest.TestCase):
    def test_snapshot_copies_candidates_and_keeps_manifest_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_home = root / "home"
            session = source_home / ".codex" / "sessions" / "2026" / "09" / "rollout-test.jsonl"
            session.parent.mkdir(parents=True)
            content = '{"messages":[{"role":"user","content":"secret"}]}\n'
            session.write_text(content, encoding="utf-8")

            output_dir = root / "snapshot"
            manifest = snapshot_roots(
                [RootSpec("codex", "primary", source_home / ".codex")],
                source_home=source_home,
                output_dir=output_dir,
            )

            self.assertEqual(manifest["counts"]["candidate_files"], 1)
            self.assertEqual(manifest["counts"]["stable_files"], 1)
            self.assertEqual(manifest["counts"]["unstable_files"], 0)
            entry = manifest["files"][0]
            self.assertEqual(entry["sha256"], hashlib.sha256(content.encode()).hexdigest())
            self.assertNotIn("secret", json.dumps(manifest))
            self.assertNotIn(str(source_home), json.dumps(manifest))
            copied = output_dir / "home" / ".codex" / "sessions" / "2026" / "09" / "rollout-test.jsonl"
            self.assertEqual(copied.read_text(encoding="utf-8"), content)

    def test_backup_root_keeps_backup_source_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_home = root / "home"
            session = source_home / ".omp-backups" / "agent" / "sessions" / "rollout.jsonl"
            session.parent.mkdir(parents=True)
            session.write_text("{}\n", encoding="utf-8")

            manifest = snapshot_roots(
                [RootSpec("oh-my-pi", "backups", source_home / ".omp-backups")],
                source_home=source_home,
                output_dir=root / "snapshot",
            )

        self.assertEqual(manifest["files"][0]["source_class"], "session_backup")

    def test_snapshot_refuses_to_overwrite_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_home = root / "home"
            source_root = source_home / ".codex"
            source_root.mkdir(parents=True)
            (source_root / "sessions" / "rollout-test.jsonl").parent.mkdir(parents=True)
            (source_root / "sessions" / "rollout-test.jsonl").write_text("{}\n", encoding="utf-8")
            output_dir = root / "snapshot"
            output_dir.mkdir()
            (output_dir / "existing").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                snapshot_roots(
                    [RootSpec("codex", "primary", source_root)],
                    source_home=source_home,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
