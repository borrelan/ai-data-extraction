import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from preflight_sources import discover_line_files, preflight_sources, scan_jsonl, write_manifest


class PreflightTests(unittest.TestCase):
    def test_streaming_scan_bounds_oversize_line_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.jsonl"
            large_content = b'{"secret":"' + (b"x" * 120) + b'"}'
            path.write_bytes(b'{"ok":true}\nnot-json\n' + large_content + b"\n\n")

            report = scan_jsonl(
                path,
                max_record_chars=32,
                parse_limit_bytes=64,
                chunk_bytes=5,
            )

            self.assertEqual(report["parser_status"], "complete")
            self.assertEqual(report["counts"]["records"], 3)
            self.assertEqual(report["counts"]["valid_json"], 1)
            self.assertEqual(report["counts"]["invalid_json"], 1)
            self.assertEqual(report["counts"]["unparsed_oversize"], 1)
            self.assertEqual(report["counts"]["raw_oversize_lines"], 1)
            self.assertEqual(report["oversized_lines"][0]["line"], 3)
            self.assertEqual(
                report["oversized_lines"][0]["sha256"],
                hashlib.sha256(large_content).hexdigest(),
            )

            encoded = json.dumps(report)
            self.assertNotIn("secret", encoded)
            self.assertNotIn("x" * 120, encoded)

    def test_discovers_explicit_backup_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active.jsonl").write_text('{"a":1}\n', encoding="utf-8")
            (root / "backup.jsonl.backup").write_text('{"b":2}\n', encoding="utf-8")
            names = [path.name for path in discover_line_files([root])]
            self.assertEqual(names, ["active.jsonl", "backup.jsonl.backup"])

    def test_manifest_is_metadata_only_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "nested" / "preflight.json"
            source.write_text('{"private":"do-not-export"}\n', encoding="utf-8")

            manifest = preflight_sources([source], parse_limit_bytes=1024)
            write_manifest(output, manifest, overwrite=False)

            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "ai-data-extraction/preflight/v1")
            self.assertEqual(saved["policy"]["content_policy"], "metadata_only")
            self.assertNotIn("do-not-export", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
