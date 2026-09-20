import unittest

from quality_rules import lineage_issues, scan_record, tool_edge_status


def complete_metadata() -> dict[str, object]:
    return {
        "source_file_sha256": "source-digest",
        "source_manifest_revision": "manifest-revision",
        "source_snapshot_status": "bound",
        "parser_revision": "parser-revision",
        "source_message_range": {"start": 0, "end": 1},
        "session_id": "session-1",
    }


class QualityRulesTests(unittest.TestCase):
    def test_scan_reports_findings_without_returning_content(self):
        record = {
            "messages": [
                {"role": "user", "content": "visible <analysis> marker"},
                {
                    "role": "assistant",
                    "content": "Bearer abcdefghijklmnopqrstuvwxyz123456",
                    "thinking": "must not survive",
                },
                {"role": "tool", "content": "/home/private/file"},
            ],
            "raw": {"debug": "discard"},
        }

        findings = scan_record(record)

        self.assertIn("thinking", findings.hidden_keys)
        self.assertTrue(findings.marker_paths)
        self.assertGreater(findings.secret_values, 0)
        self.assertGreater(findings.private_values, 0)
        self.assertIn("raw", findings.raw_keys)
        self.assertNotIn("must not survive", str(findings.summary()))

    def test_lineage_requires_snapshot_and_parser_binding(self):
        record = {"metadata": {"source_file_sha256": "digest", "session_id": "s"}}
        self.assertEqual(
            lineage_issues(record),
            [
                "snapshot_revision_missing",
                "parser_revision_missing",
                "source_or_message_range_missing",
            ],
        )

    def test_tool_edges_require_ids_and_matching_results(self):
        complete = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "type": "function_call"}],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            ]
        }
        incomplete = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "type": "function_call"}],
                }
            ]
        }

        self.assertTrue(tool_edge_status(complete).complete)
        self.assertFalse(tool_edge_status(incomplete).complete)
        self.assertEqual(tool_edge_status(incomplete).unmatched_calls, 1)


if __name__ == "__main__":
    unittest.main()
