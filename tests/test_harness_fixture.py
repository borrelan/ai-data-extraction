import json
import tempfile
import unittest
from pathlib import Path

from harness_fixture import replay_fixture


class HarnessFixtureTests(unittest.TestCase):
    def test_replay_uses_quality_metadata_and_covers_success_and_bounded_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sft.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "provider": "oh-my-pi",
                            "source_class": "session_active",
                        },
                        "quality": {
                            "session_quality_id": "quality-1",
                            "session_quality_gate": "candidate",
                            "session_quality_flags": [],
                        },
                        "messages": [
                            {"role": "user", "content": "private prompt"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "fixture"

            manifest = replay_fixture(source, output_dir)

            success = [
                json.loads(line)
                for line in (output_dir / "success.jsonl").read_text().splitlines()
            ]
            bounded = [
                json.loads(line)
                for line in (output_dir / "bounded_stop.jsonl").read_text().splitlines()
            ]

        self.assertEqual(manifest["source"]["quality"]["session_quality_id"], "quality-1")
        self.assertEqual(len(success), 12)
        self.assertEqual(len(bounded), 4)
        self.assertEqual(success[-1]["event_type"], "terminal")
        self.assertEqual(bounded[-1]["event_type"], "terminal")
        self.assertEqual(bounded[-1]["payload"]["status"], "bounded_stop")
        self.assertNotIn("private prompt", "\n".join(json.dumps(row) for row in success))


if __name__ == "__main__":
    unittest.main()
