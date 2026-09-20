import json
import tempfile
import unittest
from pathlib import Path

from extract_gemini import extract_gemini_session, find_all_gemini_sessions


class GeminiExtractionTests(unittest.TestCase):
    def test_json_session_preserves_supported_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "session-json.json"
            source.write_text(
                json.dumps(
                    {
                        "sessionId": "json-session",
                        "messages": [
                            {"type": "user", "content": "Inspect this."},
                            {
                                "type": "gemini",
                                "content": "Done.",
                                "model": "test-model",
                                "thoughts": "hidden reasoning",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = []
            conversation = extract_gemini_session(source, source_manifest=manifest)

        self.assertEqual(conversation["session_id"], "json-session")
        self.assertEqual([message["role"] for message in conversation["messages"]], [
            "user",
            "assistant",
        ])
        self.assertEqual(conversation["messages"][1]["model"], "test-model")
        self.assertEqual(manifest[0]["status"], "emitted")
        self.assertTrue(manifest[0]["source_file_sha256"])

    def test_event_sourced_jsonl_uses_latest_snapshot_without_losing_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chats = root / "tmp" / "project" / "chats"
            chats.mkdir(parents=True)
            source = chats / "session-jsonl.jsonl"
            first_snapshot = [
                {"type": "user", "id": "u1", "content": [{"text": "Run tests."}]},
                {
                    "type": "gemini",
                    "id": "a1",
                    "content": "I will run them.",
                    "toolCalls": [
                        {
                            "id": "call-1",
                            "name": "run_shell_command",
                            "args": {"command": "pytest"},
                            "result": [{"output": "passed"}],
                            "status": "success",
                        }
                    ],
                },
            ]
            final_snapshot = [
                first_snapshot[0],
                {"type": "gemini", "id": "a1", "content": "I will run them."},
                {
                    "type": "user",
                    "id": "r1",
                    "content": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "run_shell_command",
                                "response": {"output": "passed"},
                            }
                        },
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "run_shell_command",
                                "response": {"output": "passed"},
                            }
                        },
                    ],
                },
            ]
            source.write_text(
                "".join(
                    json.dumps(event) + "\n"
                    for event in [
                        {
                            "kind": "session",
                            "sessionId": "jsonl-session",
                            "projectHash": "project-1",
                        },
                        {"$set": {"messages": first_snapshot}},
                        {"$set": {"messages": final_snapshot}},
                    ]
                ),
                encoding="utf-8",
            )

            manifest = []
            conversation = extract_gemini_session(source, source_manifest=manifest)

        messages = conversation["messages"]
        self.assertEqual([message["role"] for message in messages], [
            "user",
            "assistant",
            "tool",
        ])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "run_shell_command")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")
        self.assertIn("passed", messages[2]["content"])
        self.assertEqual(conversation.get("source_parse_errors", 0), 0)
        self.assertEqual(manifest[0]["status"], "emitted")

    def test_session_discovery_admits_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            chats = Path(directory) / "tmp" / "project" / "chats"
            chats.mkdir(parents=True)
            json_file = chats / "session-one.json"
            jsonl_file = chats / "session-two.jsonl"
            json_file.write_text("{}", encoding="utf-8")
            jsonl_file.write_text("{}\n", encoding="utf-8")

            found = find_all_gemini_sessions(Path(directory))

        self.assertEqual(found, [json_file, jsonl_file])


if __name__ == "__main__":
    unittest.main()
