import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from extract_opencode import (
    build_source_manifest,
    iter_cli_conversations,
    iter_cli_conversations_db,
    iter_cli_conversations_json,
    iter_cli_session_diff_records,
)
from source_manifest import SourceManifestIndex, _digest_value


def create_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            workspace_id TEXT,
            parent_id TEXT,
            slug TEXT,
            directory TEXT,
            title TEXT,
            version TEXT,
            share_url TEXT,
            summary_additions INTEGER,
            summary_deletions INTEGER,
            summary_files INTEGER,
            summary_diffs TEXT,
            revert TEXT,
            permission TEXT,
            time_created INTEGER,
            time_updated INTEGER,
            time_compacting INTEGER,
            time_archived INTEGER,
            agent TEXT,
            model TEXT
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            time_created INTEGER,
            time_updated INTEGER,
            data TEXT
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            session_id TEXT,
            time_created INTEGER,
            time_updated INTEGER,
            data TEXT
        );
        """
    )
    connection.execute(
        """INSERT INTO session VALUES
        (?, ?, NULL, NULL, ?, ?, ?, ?, NULL, 1, 2, 3, NULL, NULL, NULL,
         1, 2, NULL, NULL, ?, NULL)""",
        (
            "ses_db",
            "project_db",
            "slug",
            "/private/project",
            "Database session",
            "v1",
            "assistant",
        ),
    )
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            "msg_user",
            "ses_db",
            1,
            1,
            json.dumps({
                "role": "user",
                "time": {"created": 1},
                "summary": {"diffs": ["discard-me"] * 10000},
            }),
        ),
    )
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (
            "msg_assistant",
            "ses_db",
            2,
            2,
            json.dumps({
                "role": "assistant",
                "time": {"created": 2},
                "modelID": "model-1",
                "providerID": "provider-1",
            }),
        ),
    )
    parts = [
        (
            "part_user",
            "msg_user",
            1,
            {"type": "text", "text": "Please inspect this change."},
        ),
        (
            "part_reasoning",
            "msg_assistant",
            2,
            {"type": "reasoning", "text": "private chain of thought"},
        ),
        (
            "part_tool",
            "msg_assistant",
            3,
            {
                "type": "tool",
                "tool": "shell",
                "callID": "call-1",
                "state": {
                    "status": "completed",
                    "input": {"command": "printf hi"},
                    "output": "o" * 5000,
                },
            },
        ),
        (
            "part_patch",
            "msg_assistant",
            4,
            {"type": "patch", "hash": "patch-hash", "files": ["src/a.py"]},
        ),
        (
            "part_text",
            "msg_assistant",
            5,
            {"type": "text", "text": "The command completed."},
        ),
    ]
    connection.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        [
            (part_id, message_id, "ses_db", time, time, json.dumps(data))
            for part_id, message_id, time, data in parts
        ],
    )
    connection.commit()
    connection.close()


def source_manifest_for_database(path: Path) -> SourceManifestIndex:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "provider": "opencode",
        "root_label": "cli",
        "source_class": "conversation_store",
        "treatment": "candidate",
        "relative_path_sha256": "a" * 64,
        "snapshot_relative_path_sha256": "b" * 64,
        "source_sha256": f"sha256:{digest}",
        "source_bytes": path.stat().st_size,
        "snapshot_status": "bound",
        "inventory_sha256": f"sha256:{digest}",
        "inventory_bytes": path.stat().st_size,
        "inventory_status": "observed",
    }
    row["source_ref_sha256"] = f"sha256:{_digest_value(row)}"
    manifest = {
        "schema_version": "ai-data-extraction/source-manifest/v1",
        "authority": "stable_snapshot",
        "files": [row],
    }
    manifest["source_manifest_revision"] = f"sha256:{_digest_value(manifest)}"
    return SourceManifestIndex(manifest)


def source_manifest_for_files(root: Path, files: list[tuple[Path, str]]) -> SourceManifestIndex:
    rows = []
    for path, source_class in files:
        payload = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(payload).hexdigest()
        row = {
            "provider": "opencode",
            "root_label": "cli",
            "source_class": source_class,
            "treatment": "candidate",
            "relative_path_sha256": hashlib.sha256(relative.encode()).hexdigest(),
            "snapshot_relative_path_sha256": hashlib.sha256(
                f"snapshot/{relative}".encode()
            ).hexdigest(),
            "source_sha256": f"sha256:{digest}",
            "source_bytes": len(payload),
            "snapshot_status": "bound",
            "inventory_sha256": f"sha256:{digest}",
            "inventory_bytes": len(payload),
            "inventory_status": "observed",
        }
        row["source_ref_sha256"] = f"sha256:{_digest_value(row)}"
        rows.append(row)
    manifest = {
        "schema_version": "ai-data-extraction/source-manifest/v1",
        "authority": "stable_snapshot",
        "files": rows,
    }
    manifest["source_manifest_revision"] = f"sha256:{_digest_value(manifest)}"
    return SourceManifestIndex(manifest)


class OpenCodeExtractorTests(unittest.TestCase):
    def test_database_admission_binds_exact_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "opencode.db")
            source_manifest = source_manifest_for_database(root / "opencode.db")
            records = list(
                iter_cli_conversations_db(root, source_manifest=source_manifest)
            )

        origin = records[0]["source_origin"]
        self.assertTrue(origin["source_manifest_revision"].startswith("sha256:"))
        self.assertTrue(origin["source_ref_sha256"].startswith("sha256:"))
        self.assertEqual(origin["source_snapshot_status"], "bound")
        self.assertEqual(origin["source_bytes"], origin["database"]["bytes"])

    def test_database_projection_omits_summary_reasoning_and_bounds_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "opencode.db")
            records = list(iter_cli_conversations_db(root, max_record_chars=20_000))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["source_class"], "conversation_database")
        self.assertNotIn("summary", record["messages"][0])
        self.assertTrue(
            all("reasoning" not in message for message in record["messages"])
        )
        self.assertEqual(record["source_origin"]["store_type"], "sqlite")
        self.assertTrue(record["source_origin"]["database"]["hash_stable"])
        self.assertTrue(record["observation_truncations"])
        self.assertEqual(
            record["observation_truncations"][0]["kind"],
            "tool_observation",
        )
        tool_result = record["messages"][1]["tool_results"][0]
        self.assertLess(len(tool_result["output"]), 1000)

    def test_large_session_is_replayed_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "opencode.db")
            connection = sqlite3.connect(root / "opencode.db")
            connection.executemany(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        f"msg_long_{index}",
                        "ses_db",
                        index + 10,
                        index + 10,
                        json.dumps({
                            "role": "user" if index % 2 == 0 else "assistant",
                            "time": {"created": index + 10},
                        }),
                    )
                    for index in range(100)
                ],
            )
            connection.commit()
            connection.close()

            records = list(iter_cli_conversations_db(root, max_record_chars=800))

        self.assertGreater(len(records), 1)
        self.assertEqual(
            {record["source_origin"]["message_count"] for record in records},
            {102},
        )
        self.assertEqual(
            {record["_chunk_count"] for record in records},
            {len(records)},
        )
        self.assertEqual(
            len({record["_chunk_parent_record_sha256"] for record in records}),
            1,
        )

    def test_json_store_admission_binds_message_part_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            message_dir = root / "storage" / "message" / "ses_json"
            part_dir = root / "storage" / "part" / "msg_json"
            session_dir = root / "storage" / "session" / "project"
            message_dir.mkdir(parents=True)
            part_dir.mkdir(parents=True)
            session_dir.mkdir(parents=True)
            message = message_dir / "msg_json.json"
            part = part_dir / "prt_json.json"
            sidecar = session_dir / "ses_json.json"
            message.write_text(
                json.dumps({
                    "id": "msg_json",
                    "role": "user",
                    "time": {"created": 1},
                }),
                encoding="utf-8",
            )
            part.write_text(
                json.dumps({"id": "prt_json", "type": "text", "text": "hello"}),
                encoding="utf-8",
            )
            sidecar.write_text(
                json.dumps({
                    "title": "JSON session",
                    "time": {"created": 1, "updated": 2},
                }),
                encoding="utf-8",
            )
            source_manifest = source_manifest_for_files(
                root,
                [
                    (message, "conversation_store"),
                    (part, "conversation_store"),
                    (sidecar, "conversation_sidecar"),
                ],
            )

            records = list(
                iter_cli_conversations_json(
                    root,
                    source_manifest=source_manifest,
                )
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        origin = record["source_origin"]
        self.assertEqual(origin["source_snapshot_status"], "bound")
        self.assertEqual(origin["source_file_count"], 3)
        self.assertTrue(origin["source_ref_set_sha256"])
        self.assertEqual(record["messages"][0]["content"], "hello")

    def test_json_store_long_session_emits_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            message_dir = root / "storage" / "message" / "ses_long"
            message_dir.mkdir(parents=True)
            for index in range(100):
                (message_dir / f"msg_{index:04d}.json").write_text(
                    json.dumps({
                        "id": f"msg_{index:04d}",
                        "role": "user" if index % 2 == 0 else "assistant",
                        "time": {"created": index},
                    }),
                    encoding="utf-8",
                )

            records = list(iter_cli_conversations_json(root, max_record_chars=800))

        self.assertGreater(len(records), 1)
        self.assertEqual(
            {record["source_origin"]["message_count"] for record in records},
            {100},
        )
        self.assertEqual(
            {record["_chunk_count"] for record in records},
            {len(records)},
        )

    def test_session_diff_artifacts_are_bounded_and_source_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diff_path = root / "storage" / "session_diff" / "ses_diff.json"
            diff_path.parent.mkdir(parents=True)
            diff_path.write_text(
                json.dumps([
                    {
                        "file": "src/a.py",
                        "status": "modified",
                        "before": "a" * 5000,
                        "after": "b" * 5000,
                        "additions": 5,
                        "deletions": 2,
                    },
                    {
                        "file": "src/b.py",
                        "status": "added",
                        "before": "",
                        "after": "c" * 5000,
                        "additions": 5,
                        "deletions": 0,
                    },
                ]),
                encoding="utf-8",
            )
            source_manifest = source_manifest_for_files(
                root,
                [(diff_path, "conversation_sidecar")],
            )

            records = list(
                iter_cli_session_diff_records(
                    root,
                    max_record_chars=20_000,
                    source_manifest=source_manifest,
                )
            )

        self.assertGreaterEqual(len(records), 1)
        self.assertTrue(
            all(
                record["source_origin"]["source_ref_sha256"].startswith("sha256:")
                for record in records
            )
        )
        self.assertTrue(all(record["messages"] == [] for record in records))
        self.assertTrue(
            all(
                len(json.dumps(record, ensure_ascii=False)) < 20_000
                for record in records
            )
        )

    def test_database_projection_uses_referenced_external_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "opencode.db")
            output_dir = root / "tool-output"
            output_dir.mkdir()
            external_output = "complete external observation"
            (output_dir / "tool_external").write_text(
                external_output,
                encoding="utf-8",
            )
            connection = sqlite3.connect(root / "opencode.db")
            connection.execute(
                "UPDATE part SET data = ? WHERE id = ?",
                (
                    json.dumps(
                        {
                            "type": "tool",
                            "tool": "shell",
                            "callID": "call-1",
                            "state": {
                                "status": "completed",
                                "input": {"command": "printf hi"},
                                "output": "preview",
                                "metadata": {"outputPath": "/tmp/tool_external"},
                            },
                        }
                    ),
                    "part_tool",
                ),
            )
            connection.commit()
            connection.close()

            records = list(iter_cli_conversations_db(root, max_record_chars=20_000))

        tool_result = records[0]["messages"][1]["tool_results"][0]
        self.assertEqual(tool_result["output"], external_output)
        self.assertEqual(tool_result["output_source"], "external_tool_output")
        self.assertEqual(tool_result["output_file_bytes"], len(external_output))
        self.assertTrue(tool_result["output_file_sha256"])

    def test_database_and_json_storage_are_both_emitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "opencode.db")
            message_dir = root / "storage" / "message" / "ses_json"
            part_dir = root / "storage" / "part" / "msg_json"
            message_dir.mkdir(parents=True)
            part_dir.mkdir(parents=True)
            (message_dir / "msg_json.json").write_text(
                json.dumps({"id": "msg_json", "role": "user", "time": {"created": 1}}),
                encoding="utf-8",
            )
            (part_dir / "prt_json.json").write_text(
                json.dumps({"type": "text", "text": "storage prompt"}),
                encoding="utf-8",
            )
            records = list(iter_cli_conversations(root))

        self.assertEqual(
            {record["source_class"] for record in records},
            {"conversation_database", "conversation_storage"},
        )

    def test_source_manifest_links_child_stores_and_external_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "opencode.db")
            message_dir = root / "storage" / "message" / "ses_json"
            part_dir = root / "storage" / "part" / "msg_json"
            message_dir.mkdir(parents=True)
            part_dir.mkdir(parents=True)
            (message_dir / "msg_json.json").write_text(
                json.dumps({"id": "msg_json", "role": "user"}),
                encoding="utf-8",
            )
            (part_dir / "prt_json.json").write_text(
                json.dumps(
                    {
                        "type": "tool",
                        "state": {
                            "metadata": {"outputPath": "/tmp/tool_tool1"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            sidecar = root / "storage" / "session" / "project" / "ses_json.json"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text("{}", encoding="utf-8")
            (root / "storage" / "session_diff").mkdir(parents=True)
            (root / "storage" / "session_diff" / "ses_json.json").write_text(
                "{}", encoding="utf-8"
            )
            output_dir = root / "tool-output"
            output_dir.mkdir()
            (output_dir / "tool_tool1").write_text("tool output", encoding="utf-8")

            manifest = build_source_manifest(
                root,
                emitted_db_session_ids={"ses_db"},
                emitted_storage_session_ids={"ses_json"},
            )

        entries = manifest["source_files"]
        self.assertEqual(len(entries), 6)
        self.assertTrue(all(entry.get("source_file_sha256") for entry in entries))
        by_kind = {(entry["source_class"], entry["source_file_name"]): entry for entry in entries}
        self.assertEqual(by_kind[("conversation_store", "opencode.db")]["status"], "emitted")
        self.assertEqual(
            by_kind[("conversation_store", "msg_json.json")]["status"],
            "emitted_via_json_storage",
        )
        self.assertEqual(
            by_kind[("conversation_sidecar", "ses_json.json")]["status"],
            "linked_to_session",
        )
        self.assertTrue(by_kind[("tool_output", "tool_tool1")]["referenced_by_json_parts"])


if __name__ == "__main__":
    unittest.main()
