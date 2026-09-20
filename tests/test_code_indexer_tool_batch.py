import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from build_code_indexer_tool_batch import build_code_indexer_tool_batch
from capture_code_indexer_episode import build_episode_and_trace
from harness_tool_projection import project_harness_tool_sft


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _make_capture(root: Path, query: str, line: int) -> Path:
    capture_dir = root / query
    capture_dir.mkdir()
    result = {
        "capability_stage": "semantic_ready",
        "hits": [
            {
                "source": {
                    "chunk": {
                        "file_path": "build_training_data.py",
                        "start_line": line,
                        "symbol_name": query,
                    }
                }
            }
        ],
    }
    record, trace, details = build_episode_and_trace(
        result=result,
        raw_result_sha256=(query + "-raw").encode().hex()[:64].ljust(64, "a"),
        sanitized_result_sha256=(query + "-sanitized").encode().hex()[:64].ljust(64, "b"),
        binary_sha256="c" * 64,
        project_revision="d" * 40,
        skill_records=[
            {
                "skill": "code-indexer-ops",
                "skill_revision": "sha256:" + "e" * 64,
                "content_sha256": "sha256:" + "e" * 64,
            },
            {
                "skill": "core-principles",
                "skill_revision": "sha256:" + "f" * 64,
                "content_sha256": "sha256:" + "f" * 64,
            },
        ],
        skill_revision="runtime-skill-bundle/v1:sha256:" + "1" * 64,
        verifier_revision="verifier/v1:sha256:" + "2" * 64,
        project_root=root,
        query=query,
        limit=5,
    )
    episode = capture_dir / "episode.jsonl"
    trace_path = capture_dir / "trace.jsonl"
    episode.write_text(_canonical(record) + "\n", encoding="utf-8")
    trace.write_jsonl(trace_path)
    (capture_dir / "cli_result.json").write_text(_canonical(result) + "\n", encoding="utf-8")
    runtime_status = {"capability_stage": "semantic_ready", "published_view": {"complete": True, "revision": 1}}
    (capture_dir / "runtime_status.json").write_text(
        _canonical(runtime_status) + "\n", encoding="utf-8"
    )
    projection = project_harness_tool_sft(episode, trace_path, capture_dir / "projection")
    capture_manifest = {
        "schema_version": "ai-data-extraction/runtime-tool-capture/v1",
        "status": "review_only",
        "training_authorized": False,
        "trainer_projection": "review_only",
        "source": {
            "repository": "ai-data-extraction",
            "project_revision": "d" * 40,
            "binary_sha256": "c" * 64,
            "command": ["code-indexer", "search", query, "--root", ".", "--limit", 5],
            "exit_code": 0,
            "raw_result_sha256": "a" * 64,
            "sanitized_result_sha256": "b" * 64,
            "worktree_identity": {
                "head": "d" * 40,
                "dirty": True,
                "status_sha256": "c" * 64,
                "tracked_diff_sha256": "e" * 64,
                "tracked_diff_bytes": 1,
                "untracked_files_sha256": "sha256:" + "f" * 64,
                "untracked_file_count": 1,
            },
            "runtime_status_identity": {
                "capability_stage": "semantic_ready",
                "published_complete": True,
                "published_revision": 1,
                "published_fence": None,
                "raw_status_sha256": "a" * 64,
                "sanitized_status_sha256": hashlib.sha256(
                    (_canonical(runtime_status) + "\n").encode()
                ).hexdigest(),
            },
            "runtime_status_raw_sha256": "a" * 64,
            "runtime_status_sanitized_sha256": hashlib.sha256(
                (_canonical(runtime_status) + "\n").encode()
            ).hexdigest(),
        },
        "contract": {
            "tool_name": "code-indexer.search",
            "registry_revision": details["registry_revision"],
            "skill_revision": "runtime-skill-bundle/v1:sha256:" + "1" * 64,
            "verifier_revision": "verifier/v1:sha256:" + "2" * 64,
            "required_skills": ["code-indexer-ops", "core-principles"],
            "rewards": "not_exported",
        },
        "counts": {"episodes": 1, "projected_tool_sft": 1},
        "validation": {
            "actual_cli_invocation": "passed",
            "trace_terminal": "success",
            "registry_join": projection["validation"]["registry_join"],
            "skill_gate": projection["validation"]["skill_gate"],
            "call_observation_join": projection["validation"]["call_observation_join"],
            "verification": projection["validation"]["verification"],
            "privacy_reasoning_firewall": projection["validation"]["privacy_reasoning_firewall"],
            "reward": "not_present",
        },
        "definition": details["definition"],
        "query": query,
        "limit": 5,
        "files": {},
    }
    file_paths = [
        "episode.jsonl",
        "trace.jsonl",
        "cli_result.json",
        "runtime_status.json",
        "projection/tool_sft.jsonl",
        "projection/lineage.jsonl",
        "projection/manifest.json",
    ]
    for relative in file_paths:
        path = capture_dir / relative
        capture_manifest["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (capture_dir / "capture_manifest.json").write_text(
        _canonical(capture_manifest) + "\n", encoding="utf-8"
    )
    return capture_dir


class CodeIndexerToolBatchTests(unittest.TestCase):
    def test_joins_verified_captures_with_parent_disjoint_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _make_capture(root, "model_tier_for", 833)
            second = _make_capture(root, "normalize_record", 3286)
            output = root / "batch"
            manifest = build_code_indexer_tool_batch([second, first], output)

            self.assertEqual(manifest["counts"]["captures"], 2)
            self.assertEqual(manifest["counts"]["tool_sft"], 2)
            self.assertEqual(manifest["counts"]["tool_sft_train"], 1)
            self.assertEqual(manifest["counts"]["tool_sft_validation"], 1)
            self.assertFalse(manifest["training_authorized"])
            self.assertEqual(manifest["validation"]["exact_registry_binding"], "passed")

            train = [json.loads(line) for line in (output / "tool_sft_train.jsonl").read_text().splitlines()]
            validation = [
                json.loads(line)
                for line in (output / "tool_sft_validation.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["split"] for row in train}, {"train"})
            self.assertEqual({row["split"] for row in validation}, {"validation"})
            self.assertNotEqual(train[0]["example_id"], validation[0]["example_id"])
            self.assertNotIn("reasoning", json.dumps(train + validation).lower())

            lineage = [
                json.loads(line) for line in (output / "lineage.jsonl").read_text().splitlines()
            ]
            rows_by_id = {row["example_id"]: row for row in train + validation}
            self.assertTrue(
                all(
                    item["split"] == rows_by_id[item["example_id"]]["split"]
                    and item["batch_split"] == item["split"]
                    for item in lineage
                )
            )
            self.assertTrue(all("source_projection_split" in item for item in lineage))

            decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
            self.assertEqual({decision["decision"] for decision in decisions}, {"review_only"})
            self.assertEqual({decision["reward_status"] for decision in decisions}, {"not_exported"})


if __name__ == "__main__":
    unittest.main()
