import json
import tempfile
import unittest
from pathlib import Path

from extract_claude_code import extract_claude_project_conversations


class ClaudeExtractionTests(unittest.TestCase):
    def test_sidechain_sessions_are_ingested_as_optional_and_empty_files_are_manifested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "projects" / "project-a"
            project.mkdir(parents=True)

            def write(name, rows):
                path = project / name
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return path

            write(
                "session-main.jsonl",
                [
                    {"type": "user", "message": {"content": "Inspect this."}},
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "Done."}]},
                    },
                ],
            )
            write(
                "agent-side.jsonl",
                [
                    {
                        "type": "user",
                        "sessionId": "parent-session",
                        "agentId": "agent-side",
                        "isSidechain": True,
                        "message": {"content": "Check the narrow path."},
                    },
                    {
                        "type": "assistant",
                        "sessionId": "parent-session",
                        "agentId": "agent-side",
                        "isSidechain": True,
                        "message": {"content": [{"type": "text", "text": "Checked."}]},
                    },
                ],
            )
            write("agent-empty.jsonl", [{"type": "progress", "sessionId": "empty"}])

            manifest = []
            conversations = extract_claude_project_conversations(
                root,
                source_manifest=manifest,
            )

        self.assertEqual(len(conversations), 2)
        by_class = {conversation["source_class"]: conversation for conversation in conversations}
        self.assertEqual(by_class["session_active"]["training_lane"], "primary")
        self.assertEqual(by_class["subagent_session"]["training_lane"], "optional_alt")
        self.assertEqual(by_class["subagent_session"]["source_origin"]["sidechain"], True)
        self.assertEqual(by_class["subagent_session"]["source_origin"]["parent_session_id"], "parent-session")

        statuses = {entry["source_file_name"]: entry["status"] for entry in manifest}
        self.assertEqual(statuses["session-main.jsonl"], "emitted")
        self.assertEqual(statuses["agent-side.jsonl"], "emitted")
        self.assertEqual(statuses["agent-empty.jsonl"], "no_emitted_messages")
        self.assertTrue(all(entry.get("source_file_sha256") for entry in manifest))


if __name__ == "__main__":
    unittest.main()
