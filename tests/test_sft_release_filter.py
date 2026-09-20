import json
import tempfile
import unittest
from pathlib import Path

from runtime.sft.filter_release import canonical_bytes, filter_release, write_jsonl


class FakeTokenizer:
    chat_template = "fake-template"

    def __len__(self):
        return 100

    def apply_chat_template(self, messages, **kwargs):
        size = 1 + sum(len(message.get("content", "")) for message in messages)
        return list(range(size))


class SftReleaseFilterTests(unittest.TestCase):
    def test_filter_drops_whole_overlength_row_and_preserves_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            short = {
                "schema_version": "ai-data-extraction/agent-sft-example/v1",
                "example_id": "short",
                "split": "train",
                "lane": "verified_open_swe_action",
                "messages": [
                    {"role": "user", "content": "small"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
            long = {
                "schema_version": "ai-data-extraction/agent-sft-example/v1",
                "example_id": "long",
                "split": "validation",
                "lane": "verified_open_swe_action",
                "messages": [
                    {"role": "user", "content": "x" * 100},
                    {"role": "assistant", "content": "ok"},
                ],
            }
            files = {
                "train.jsonl": write_jsonl(input_dir / "train.jsonl", [short]),
                "validation.jsonl": write_jsonl(
                    input_dir / "validation.jsonl", [long]
                ),
                "lineage.jsonl": write_jsonl(
                    input_dir / "lineage.jsonl",
                    [
                        {"example_id": "short", "parent_id": "parent-short"},
                        {"example_id": "long", "parent_id": "parent-long"},
                    ],
                ),
                "decisions.jsonl": write_jsonl(
                    input_dir / "decisions.jsonl", [{"decision": "fixture"}]
                ),
            }
            manifest = {
                "schema_version": "ai-data-extraction/agent-sft-pilot/v1",
                "status": "pending",
                "training_authorized": False,
                "model": {"max_sequence_tokens": 50},
                "selection": {},
                "quality": {},
                "counts": {"total": 2},
                "files": files,
            }
            (input_dir / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")

            output = root / "output"
            result = filter_release(
                input_dir=input_dir,
                output_dir=output,
                tokenizer=FakeTokenizer(),
                max_length=50,
            )
            self.assertEqual(result["counts"]["total"], 1)
            self.assertEqual(result["counts"]["excluded_over_token_limit"], 1)
            self.assertFalse(result["training_authorized"])
            self.assertEqual(
                result["status"],
                "exact_tokenizer_filtered_pending_independent_preflight_not_training_authorized",
            )
            self.assertEqual(
                result["quality"]["tokenizer_preflight"],
                "pending_independent_verifier",
            )
            self.assertEqual(
                json.loads((output / "train.jsonl").read_text())["example_id"],
                "short",
            )
            self.assertEqual((output / "validation.jsonl").read_text(), "")
            exclusion = json.loads((output / "token_exclusions.jsonl").read_text())
            self.assertEqual(exclusion["example_id"], "long")
            lineage = json.loads((output / "lineage.jsonl").read_text())
            self.assertEqual(lineage["example_id"], "short")
            self.assertTrue((output / "decisions.jsonl").is_file())

    def test_filter_rejects_duplicate_example_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            row = {
                "schema_version": "ai-data-extraction/agent-sft-example/v1",
                "example_id": "duplicate",
                "split": "train",
                "lane": "verified_open_swe_action",
                "messages": [
                    {"role": "user", "content": "small"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
            files = {
                "train.jsonl": write_jsonl(input_dir / "train.jsonl", [row, row]),
                "validation.jsonl": write_jsonl(
                    input_dir / "validation.jsonl", []
                ),
                "lineage.jsonl": write_jsonl(
                    input_dir / "lineage.jsonl",
                    [{"example_id": "duplicate", "parent_id": "parent"}],
                ),
            }
            manifest = {
                "schema_version": "ai-data-extraction/agent-sft-pilot/v1",
                "status": "pending",
                "training_authorized": False,
                "model": {"max_sequence_tokens": 50},
                "selection": {},
                "quality": {},
                "counts": {"total": 2},
                "files": files,
            }
            (input_dir / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")

            with self.assertRaisesRegex(ValueError, "duplicate trainer example"):
                filter_release(
                    input_dir=input_dir,
                    output_dir=root / "output",
                    tokenizer=FakeTokenizer(),
                    max_length=50,
                )


if __name__ == "__main__":
    unittest.main()
