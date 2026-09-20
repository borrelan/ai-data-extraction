import json
import tempfile
import unittest
from pathlib import Path

from capture_code_indexer_episode import (
    TOOL_NAME,
    build_episode_and_trace,
)
from harness_tool_projection import project_harness_tool_sft


class CodeIndexerCaptureTests(unittest.TestCase):
    def test_builds_and_projects_verifier_bound_episode(self):
        result = {
            "capability_stage": "semantic_partial",
            "hits": [
                {
                    "source": {
                        "chunk": {
                            "file_path": "build_training_data.py",
                            "start_line": 2409,
                            "symbol_name": "build_session_quality_index",
                        }
                    }
                }
            ],
            "published_revision": 25,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record, trace, details = build_episode_and_trace(
                result=result,
                raw_result_sha256="a" * 64,
                sanitized_result_sha256="b" * 64,
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
            )
            episode_path = root / "episode.jsonl"
            trace_path = root / "trace.jsonl"
            episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            trace.write_jsonl(trace_path)
            projection = project_harness_tool_sft(
                episode_path,
                trace_path,
                root / "projection",
            )
            self.assertEqual(details["definition"]["symbol"], "build_session_quality_index")
            self.assertEqual(projection["counts"], {"input_records": 1, "tool_sft": 1})
            self.assertEqual(projection["contract"]["registry_revision"], details["registry_revision"])
            exported = json.loads((root / "projection/tool_sft.jsonl").read_text())
            self.assertEqual(exported["tools"][0]["function"]["name"], TOOL_NAME)
            self.assertNotIn("reasoning", json.dumps(exported).lower())

    def test_trace_has_single_registry_and_terminal_event(self):
        result = {
            "hits": [
                {
                    "source": {
                        "chunk": {
                            "file_path": "x.py",
                            "start_line": 4,
                            "symbol_name": "build_session_quality_index",
                        }
                    }
                }
            ]
        }
        record, trace, _ = build_episode_and_trace(
            result=result,
            raw_result_sha256="a" * 64,
            sanitized_result_sha256="b" * 64,
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
            project_root=Path("."),
        )
        events = trace.events
        self.assertEqual(sum(event["event_type"] == "tool_registry" for event in events), 1)
        self.assertEqual(events[-1]["event_type"], "terminal")
        self.assertEqual(events[-1]["payload"]["status"], "success")
        self.assertEqual(record["dataset"], "tool_trace")

    def test_parameterized_query_is_bound_into_call_and_lineage(self):
        result = {
            "hits": [
                {
                    "source": {
                        "chunk": {
                            "file_path": "build_training_data.py",
                            "start_line": 833,
                            "symbol_name": "model_tier_for",
                        }
                    }
                }
            ]
        }
        record, trace, details = build_episode_and_trace(
            result=result,
            raw_result_sha256="a" * 64,
            sanitized_result_sha256="b" * 64,
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
            project_root=Path("."),
            query="model_tier_for",
            limit=3,
            worktree_identity={"head": "d" * 40, "dirty": True, "status_sha256": "a" * 64},
            runtime_status_identity={
                "capability_stage": "semantic_ready",
                "published_revision": 25,
                "sanitized_status_sha256": "b" * 64,
            },
        )
        call = next(
            call
            for message in record["messages"]
            for call in message.get("tool_calls", [])
        )
        self.assertEqual(call["function"]["arguments"], {"limit": 3, "query": "model_tier_for", "root": "."})
        self.assertEqual(record["lineage"]["query"], "model_tier_for")
        self.assertEqual(record["lineage"]["limit"], 3)
        self.assertTrue(record["lineage"]["worktree_identity"]["dirty"])
        self.assertEqual(record["lineage"]["runtime_status_identity"]["published_revision"], 25)
        self.assertEqual(details["definition"]["symbol"], "model_tier_for")
        self.assertEqual(trace.events[-1]["payload"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
