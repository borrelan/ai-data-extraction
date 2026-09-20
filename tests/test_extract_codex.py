import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from extract_codex import (
    CodexSourceSpec,
    extract_codex_session,
    find_all_codex_sessions,
    iter_codex_session_records,
    write_codex_records,
)
from source_manifest import SourceAdmissionError, SourceManifestIndex, build_source_manifest


class CodexExtractionTests(unittest.TestCase):
    def _ledger_for(self, root: Path, source: Path) -> SourceManifestIndex:
        content = source.read_text(encoding="utf-8")
        source_digest = hashlib.sha256(content.encode()).hexdigest()
        relative_digest = hashlib.sha256(b"sessions/rollout.jsonl").hexdigest()
        inventory = {
            "schema_version": "ai-data-extraction/source-inventory/v1",
            "files": [
                {
                    "provider": "codex",
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
                    "provider": "codex",
                    "root_label": "primary",
                    "source_class": "session_active",
                    "treatment": "candidate",
                    "relative_path_sha256": relative_digest,
                    "snapshot_relative_path_sha256": hashlib.sha256(
                        b"home/.codex/sessions/rollout.jsonl"
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

    def test_backup_mode_is_explicit_and_source_class_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            installation = Path(directory)
            sessions = installation / "sessions/2026/09/16"
            sessions.mkdir(parents=True)
            active = sessions / "rollout-active.jsonl"
            backup = installation / "sessions/.backups/rollout-backup.jsonl.backup"
            backup.parent.mkdir(parents=True)
            payload = {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            }
            active.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            backup.write_text(
                json.dumps(payload)
                + "\n"
                + json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "world"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(find_all_codex_sessions(installation), [active])
            self.assertEqual(
                find_all_codex_sessions(installation, include_backups=True),
                [active, backup],
            )
            conversation = extract_codex_session(backup)

        self.assertEqual(conversation["source_class"], "session_backup")

    def test_session_model_provider_is_preserved_for_tier_provenance(self):
        events = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "model-provenance",
                    "model_provider": "openai",
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "Inspect"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "Done"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            record = next(iter_codex_session_records(source))

        self.assertEqual(record["model_provider"], "openai")

    def test_streaming_records_stamp_source_ledger_lineage(self):
        events = [
            {"type": "session_meta", "payload": {"id": "ledger-codex"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "Inspect"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "Done"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            ledger = self._ledger_for(root, source)
            records = list(
                iter_codex_session_records(
                    source,
                    source_manifest=ledger,
                    root_label="primary",
                )
            )

        self.assertEqual(len(records), 1)
        origin = records[0]["source_origin"]
        self.assertEqual(origin["source_snapshot_status"], "bound")
        self.assertEqual(origin["source_manifest_revision"], ledger.revision)
        self.assertTrue(origin["source_ref_sha256"].startswith("sha256:"))

    def test_streaming_admission_rejects_wrong_codex_root(self):
        events = [
            {"type": "session_meta", "payload": {"id": "wrong-root"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "Inspect"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            ledger = self._ledger_for(root, source)
            with self.assertRaises(SourceAdmissionError):
                list(
                    iter_codex_session_records(
                        source,
                        source_manifest=ledger,
                        root_label="local",
                    )
                )

    def test_native_response_items_preserve_tool_call_and_output(self):
        events = [
            {
                "type": "session_meta",
                "payload": {"id": "session-1", "cwd": "/home/alice/repo"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run the tests."}],
                },
            },
            # The event stream mirrors these response messages.  It must not
            # cause duplicate conversation turns when native messages exist.
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Run the tests."},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I will run them."}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": '{"command":"pytest"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "2 passed",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The tests pass."}],
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "The tests pass."},
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            conversation = extract_codex_session(source)

        self.assertIsNotNone(conversation)
        messages = conversation["messages"]
        self.assertEqual([message["role"] for message in messages], [
            "user",
            "assistant",
            "tool",
            "assistant",
        ])
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "shell")
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")
        self.assertEqual(messages[2]["content"], "2 passed")
        self.assertNotIn("tool_results", conversation)

    def test_large_tool_observation_is_bounded_with_hash_metadata(self):
        events = [
            {"type": "session_meta", "payload": {"id": "large-observation"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "Inspect"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "I will inspect it."},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "large-result",
                    "name": "shell",
                    "arguments": {"command": "dump"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "large-result",
                    "output": "header\n" + ("payload\n" * 20_000) + "footer",
                },
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large-observation.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            records = list(iter_codex_session_records(source, max_chars=2_000))

        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["observation_truncations"]), 1)
        truncation = records[0]["observation_truncations"][0]
        self.assertEqual(truncation["call_id"], "large-result")
        self.assertGreater(truncation["original_chars"], truncation["kept_chars"])
        tool_message = next(
            message for message in records[0]["messages"] if message["role"] == "tool"
        )
        self.assertIn("<OBSERVATION_TRUNCATED", tool_message["content"])
        self.assertLess(
            len(json.dumps(records[0], ensure_ascii=False, separators=(",", ":"))),
            5_000,
        )

    def test_native_tool_call_after_user_item_becomes_assistant_action(self):
        events = [
            {"type": "session_meta", "payload": {"id": "role-boundary"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "Inspect"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "role-call",
                    "name": "shell",
                    "arguments": {"command": "pwd"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "role-call",
                    "output": "/repo",
                },
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "role-boundary.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            conversation = extract_codex_session(source)

        self.assertEqual(
            [message["role"] for message in conversation["messages"]],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(
            conversation["messages"][1]["tool_calls"][0]["id"],
            "role-call",
        )

    def test_legacy_event_messages_remain_compatible(self):
        events = [
            {"type": "session_meta", "payload": {"id": "legacy-1"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "Run tests"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "Done"}},
            {"type": "event_msg", "payload": {"type": "tool_use", "tool": "shell", "input": {"command": "pytest"}}},
            {"type": "event_msg", "payload": {"type": "tool_result", "tool": "shell", "output": "ok"}},
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy-rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            conversation = extract_codex_session(source)

        self.assertEqual(len(conversation["messages"]), 2)
        self.assertEqual(len(conversation["tool_results"]), 2)

    def test_streaming_records_bound_long_native_session_and_keep_tool_pair(self):
        events = [
            {
                "type": "session_meta",
                "payload": {"id": "long-session", "cwd": "/tmp/repo"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "system",
                    "content": "Use the repository contract.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "Investigate the failure. " + ("context " * 8),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": "I will inspect the service.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": {"command": "pytest"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "1 failed",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": "The first signal is collected. " + ("evidence " * 8),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "Now summarize the bounded result. " + ("detail " * 8),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": "The failure is isolated.",
                },
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            records = list(iter_codex_session_records(source, max_chars=420))

        self.assertGreater(len(records), 1)
        self.assertEqual(
            [record["_chunk_index"] for record in records],
            list(range(len(records))),
        )
        self.assertTrue(
            all(record["_chunk_count"] == len(records) for record in records)
        )
        self.assertEqual(
            len({record["_chunk_parent_record_sha256"] for record in records}),
            1,
        )
        tool_records = [
            record
            for record in records
            if any(
                isinstance(message, dict)
                and message.get("tool_calls")
                for message in record["messages"]
            )
        ]
        self.assertEqual(len(tool_records), 1)
        self.assertEqual(
            next(
                message["tool_call_id"]
                for message in tool_records[0]["messages"]
                if message.get("role") == "tool"
            ),
            "call-1",
        )
        self.assertEqual(
            tool_records[0]["source_origin"]["source_event_line_range"],
            {"start": 2, "end": 6},
        )

    def test_unmatched_tool_call_cannot_hold_the_session_unbounded(self):
        events = [
            {
                "type": "session_meta",
                "payload": {"id": "unmatched-tool-session"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "Start"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "I will inspect it."},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "never-returned",
                    "name": "shell",
                    "arguments": {"command": "tail"},
                },
            },
        ]
        for index in range(40):
            events.extend(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": f"Continue investigation {index}",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": f"Evidence {index}",
                        },
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unmatched-rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            records = list(iter_codex_session_records(source, max_chars=420))

        self.assertGreater(len(records), 1)
        self.assertEqual(
            sum("never-returned" in record.get("_open_tool_call_ids", []) for record in records),
            1,
        )
        open_record = next(
            record for record in records if record.get("_open_tool_call_ids")
        )
        self.assertEqual(
            open_record["_chunk_cut_reason"],
            "ingress_record_budget_with_unmatched_tool_call",
        )
        self.assertTrue(
            all(
                len(json.dumps(record, ensure_ascii=False, separators=(",", ":"))) < 4_000
                for record in records
            )
        )

    def test_checkpoint_resume_matches_clean_output_and_publishes_lineage(self):
        events = [
            {"type": "session_meta", "payload": {"id": "checkpoint-session"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "Inspect the failing path.",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": "I will inspect the evidence.",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            ledger = self._ledger_for(root, source)
            valid = CodexSourceSpec(
                path=source,
                installation=root,
                root_label="primary",
            )
            wrong_root = CodexSourceSpec(
                path=source,
                installation=root,
                root_label="local",
            )
            resumed_output = root / "resumed.jsonl"
            checkpoint = root / "resumed.checkpoint.json"

            with self.assertRaises(SourceAdmissionError):
                write_codex_records(
                    [valid, wrong_root],
                    output_file=resumed_output,
                    timestamp="resume",
                    source_manifest=ledger,
                    checkpoint_path=checkpoint,
                )

            self.assertTrue(checkpoint.is_file())
            checkpoint_state = json.loads(checkpoint.read_text(encoding="utf-8"))
            temporary = root / checkpoint_state["temporary_name"]
            self.assertTrue(temporary.is_file())
            self.assertGreater(temporary.stat().st_size, 0)
            with temporary.open("ab") as destination:
                destination.write(b"uncommitted tail\n")

            resumed = write_codex_records(
                [valid],
                output_file=resumed_output,
                timestamp="resumed-final",
                source_manifest=ledger,
                checkpoint_path=checkpoint,
                resume=True,
            )

            clean_output = root / "clean.jsonl"
            clean = write_codex_records(
                [valid],
                output_file=clean_output,
                timestamp="clean",
                source_manifest=ledger,
                checkpoint_path=root / "clean.checkpoint.json",
            )

            self.assertFalse(checkpoint.exists())
            self.assertEqual(
                hashlib.sha256(resumed_output.read_bytes()).hexdigest(),
                hashlib.sha256(clean_output.read_bytes()).hexdigest(),
            )
            self.assertEqual(resumed["records"], clean["records"])
            source_manifest = json.loads(
                (root / resumed["source_manifest_file"]).read_text(encoding="utf-8")
            )
            self.assertEqual(source_manifest["extractor_version"], "1.2.0")
            self.assertEqual(source_manifest["source_manifest_revision"], ledger.revision)
            self.assertEqual(
                source_manifest["outputs"][resumed_output.name]["sha256"],
                "sha256:" + hashlib.sha256(resumed_output.read_bytes()).hexdigest(),
            )

    def test_checkpoint_resume_rejects_tampered_temporary_output(self):
        events = [
            {"type": "session_meta", "payload": {"id": "tamper-session"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "Inspect.",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rollout.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            ledger = self._ledger_for(root, source)
            valid = CodexSourceSpec(path=source, installation=root, root_label="primary")
            wrong_root = CodexSourceSpec(path=source, installation=root, root_label="local")
            output = root / "tamper.jsonl"
            checkpoint = root / "tamper.checkpoint.json"

            with self.assertRaises(SourceAdmissionError):
                write_codex_records(
                    [valid, wrong_root],
                    output_file=output,
                    timestamp="tamper",
                    source_manifest=ledger,
                    checkpoint_path=checkpoint,
                )
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            temporary = root / state["temporary_name"]
            with temporary.open("r+b") as destination:
                destination.seek(0)
                destination.write(b"X")

            with self.assertRaisesRegex(ValueError, "temporary output digest does not match"):
                write_codex_records(
                    [valid],
                    output_file=output,
                    timestamp="tamper-resume",
                    source_manifest=ledger,
                    checkpoint_path=checkpoint,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
