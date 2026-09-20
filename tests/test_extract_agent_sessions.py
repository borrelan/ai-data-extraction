import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from extract_agent_sessions import (
    Inspection,
    SessionSpec,
    assess_session_quality,
    iter_session_records,
    write_records,
)
from source_manifest import SourceAdmissionError, SourceManifestIndex, build_source_manifest


class AgentSessionExtractorTests(unittest.TestCase):
    def _write_ledger(self, root: Path, source: Path) -> SourceManifestIndex:
        content = source.read_text(encoding="utf-8")
        source_digest = hashlib.sha256(content.encode()).hexdigest()
        relative_digest = hashlib.sha256(b"prime/session.jsonl").hexdigest()
        inventory = {
            "schema_version": "ai-data-extraction/source-inventory/v1",
            "files": [
                {
                    "provider": "prime-agent",
                    "root_label": "primary",
                    "source_class": "session_active",
                    "treatment": "candidate",
                    "relative_path_sha256": relative_digest,
                    "bytes": len(content.encode()),
                    "file_sha256": source_digest,
                }
            ],
        }
        snapshot = {
            "schema_version": "ai-data-extraction/source-snapshot/v1",
            "snapshot_revision": "snapshot-1",
            "files": [
                {
                    "provider": "prime-agent",
                    "root_label": "primary",
                    "source_class": "session_active",
                    "treatment": "candidate",
                    "relative_path_sha256": relative_digest,
                    "snapshot_relative_path_sha256": hashlib.sha256(
                        b"home/.prime/prime/session.jsonl"
                    ).hexdigest(),
                    "status": "stable",
                    "bytes": len(content.encode()),
                    "sha256": source_digest,
                }
            ],
        }
        inventory_path = root / "inventory.json"
        snapshot_path = root / "snapshot.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
        return SourceManifestIndex(build_source_manifest(inventory_path, snapshot_path))

    def test_omits_thinking_preserves_tool_contract_and_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            events = [
                {"type": "session", "id": "session-1", "timestamp": "now"},
                {"type": "model_change", "modelId": "local-model"},
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Inspect the change."}],
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "private"},
                            {
                                "type": "toolCall",
                                "id": "call-1",
                                "name": "bash",
                                "arguments": {"command": "pytest"},
                            },
                        ],
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "call-1",
                        "toolName": "bash",
                        "isError": False,
                        "content": [{"type": "text", "text": "passed"}],
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "The test passed."}],
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            spec = SessionSpec(
                "prime-agent",
                "prime_session_active",
                "optional_alt",
                "harness_governed_review",
                path,
                "test",
            )
            records = list(iter_session_records(spec, max_record_chars=800))

        self.assertGreater(len(records), 1)
        self.assertTrue(all("private" not in json.dumps(record) for record in records))
        first = records[0]
        assistant = next(message for message in first["messages"] if message["role"] == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "bash")
        tool = next(message for message in first["messages"] if message["role"] == "tool")
        self.assertEqual(tool["tool_call_id"], "call-1")
        self.assertEqual(first["training_lane"], "optional_alt")
        self.assertEqual(first["quality_gate"], "candidate")
        self.assertIn("model_provenance_local_or_self_hosted", first["quality_flags"])
        self.assertTrue(first["quality_assessment"]["provider_neutral"])
        self.assertEqual(first["source_origin"]["store_type"], "jsonl_session")
        self.assertEqual(first["source_origin"]["source_root_label"], "test")
        self.assertEqual(first["_chunk_count"], len(records))

    def test_unparseable_lines_are_recorded_without_leaking_control_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "advisor.jsonl"
            path.write_text(
                '{"type":"session","id":"s"}\nnot-json\n'
                '{"type":"message","message":{"role":"user","content":"hello"}}\n'
                '{"type":"message","message":{"role":"assistant","content":"ok"}}\n'
            )
            spec = SessionSpec(
                "oh-my-pi",
                "pi_advisor_overlay",
                "quarantine",
                None,
                path,
                "test",
            )
            record = next(iter_session_records(spec))
        self.assertEqual(record["training_lane"], "quarantine")
        self.assertEqual(record["quality_gate"], "quarantine")
        self.assertIn("advisor_overlay_contamination", record["quality_flags"])
        self.assertEqual(record["harness_summary"]["parse_errors"], 1)
        self.assertNotIn("not-json", json.dumps(record))

    def test_quality_gate_is_session_evidence_not_provider_identity(self):
        inspection = Inspection("a" * 64, "session-1", {})
        inspection.user_message_count = 1
        inspection.assistant_message_count = 1
        inspection.message_count = 2
        inspection.model_counts["local-core"] = 1

        prime = assess_session_quality(
            SessionSpec("prime-agent", "prime_session_active", "optional_alt", None, Path("prime"), "test"),
            inspection,
        )
        pi = assess_session_quality(
            SessionSpec("oh-my-pi", "pi_session_active", "optional_alt", None, Path("pi"), "test"),
            inspection,
        )

        self.assertEqual(prime["gate"], "candidate")
        self.assertEqual(pi["gate"], "candidate")
        self.assertIn("model_provenance_local_or_self_hosted", prime["flags"])
        self.assertEqual(prime["flags"], pi["flags"])

    def test_explicit_quality_override_changes_gate_without_changing_lane(self):
        inspection = Inspection("b" * 64, "session-2", {})
        inspection.user_message_count = 1
        inspection.assistant_message_count = 1
        inspection.message_count = 2
        assessment = assess_session_quality(
            SessionSpec("pi-agent", "pi_session_active", "optional_alt", None, Path("pi"), "test"),
            inspection,
            override={"quality_gate": "review_required", "reviewer": "human", "reason": "needs outcome review"},
        )
        self.assertEqual(assessment["gate"], "review_required")
        self.assertEqual(assessment["automatic_gate"], "candidate")
        self.assertEqual(assessment["override"]["reviewer"], "human")

    def test_streaming_admission_stamps_source_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session", "id": "s"}),
                        json.dumps(
                            {
                                "type": "message",
                                "message": {"role": "user", "content": "hello"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "message",
                                "message": {"role": "assistant", "content": "world"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ledger = self._write_ledger(root, source)
            spec = SessionSpec(
                "prime-agent",
                "prime_session_active",
                "optional_alt",
                None,
                source,
                "primary",
            )
            records = list(iter_session_records(spec, source_manifest=ledger))

        self.assertEqual(len(records), 1)
        origin = records[0]["source_origin"]
        self.assertTrue(origin["source_manifest_revision"].startswith("sha256:"))
        self.assertTrue(origin["source_ref_sha256"].startswith("sha256:"))
        self.assertEqual(origin["source_snapshot_status"], "bound")

    def test_batch_admission_stamps_manifest_and_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.jsonl"
            source.write_text(
                '{"type":"message","message":{"role":"user","content":"hello"}}\n'
                '{"type":"message","message":{"role":"assistant","content":"ok"}}\n',
                encoding="utf-8",
            )
            ledger = self._write_ledger(root, source)
            spec = SessionSpec(
                "prime-agent",
                "prime_session_active",
                "optional_alt",
                None,
                source,
                "primary",
            )
            output = root / "out"
            manifest = write_records(
                [spec],
                output_dir=output,
                timestamp="test",
                source_manifest=ledger,
            )

            self.assertEqual(manifest["source_manifest_revision"], ledger.revision)
            self.assertEqual(
                manifest["source_sessions"][0]["source_snapshot_status"], "bound"
            )
            with (output / "prime_agent_sessions_test.jsonl").open(
                encoding="utf-8"
            ) as emitted:
                record = json.loads(emitted.readline())
            self.assertEqual(
                record["source_origin"]["source_manifest_revision"], ledger.revision
            )

            wrong_spec = SessionSpec(
                "pi-agent",
                "pi_session_active",
                "optional_alt",
                None,
                source,
                "primary",
            )
            with self.assertRaises(SourceAdmissionError):
                write_records(
                    [wrong_spec],
                    output_dir=root / "wrong",
                    source_manifest=ledger,
                )

    def test_checkpoint_resume_does_not_duplicate_completed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.jsonl"
            source.write_text(
                '{"type":"message","message":{"role":"user","content":"hello"}}\n'
                '{"type":"message","message":{"role":"assistant","content":"ok"}}\n',
                encoding="utf-8",
            )
            ledger = self._write_ledger(root, source)
            spec = SessionSpec(
                "prime-agent",
                "prime_session_active",
                "optional_alt",
                None,
                source,
                "primary",
            )
            wrong_spec = SessionSpec(
                "pi-agent",
                "pi_session_active",
                "optional_alt",
                None,
                source,
                "primary",
            )
            output = root / "resumed"
            checkpoint = root / "resumed.checkpoint.json"
            with self.assertRaises(SourceAdmissionError):
                write_records(
                    [spec, wrong_spec],
                    output_dir=output,
                    timestamp="test",
                    source_manifest=ledger,
                    checkpoint_path=checkpoint,
                )
            self.assertTrue(checkpoint.is_file())

            resumed = write_records(
                [spec],
                output_dir=output,
                timestamp="test",
                source_manifest=ledger,
                checkpoint_path=checkpoint,
                resume=True,
            )
            clean_output = root / "clean"
            clean = write_records(
                [spec],
                output_dir=clean_output,
                timestamp="test",
                source_manifest=ledger,
            )

            resumed_path = output / "prime_agent_sessions_test.jsonl"
            clean_path = clean_output / "prime_agent_sessions_test.jsonl"
            self.assertFalse(checkpoint.exists())
            self.assertEqual(resumed["records"], clean["records"])
            self.assertEqual(
                resumed["outputs"][resumed_path.name]["sha256"],
                clean["outputs"][clean_path.name]["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
