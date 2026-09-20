import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from trainer_export import (
    AUTHORIZATION_SCHEMA,
    TRAINER_EXAMPLE_SCHEMA,
    canonical_json_bytes,
    export_trainer_release,
    inspect_pilot,
)


def _row(unit: str, parent: str, *, tool: bool = False, schema: bool = True) -> dict[str, object]:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "request"},
    ]
    if tool:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "content": "observed", "tool_call_id": "call-1"},
            ]
        )
        messages.append({"role": "assistant", "content": "done"})
    else:
        messages.append({"role": "assistant", "content": "response"})
    row: dict[str, object] = {
        "schema_version": "ai-data-extraction/v1",
        "example_id": unit,
        "dataset": "tool_trace" if tool else "sft",
        "split": "train",
        "tags": [],
        "messages": messages,
        "quality": {"status": "review", "session_quality_gate": "candidate"},
        "privacy": {"eligible_for_training": False},
        "metadata": {"parent_record_sha256": parent},
        "lineage": {"parent_record_sha256": parent},
    }
    if tool and schema:
        row["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    return {"records": len(rows), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write_pilot(root: Path, rows: dict[str, list[dict[str, object]]]) -> tuple[Path, str]:
    pilot = root / "pilot"
    pilot.mkdir()
    dataset_specs: dict[str, object] = {}
    for dataset, dataset_rows in rows.items():
        name = f"{dataset}.jsonl"
        path = pilot / name
        spec = _write_jsonl(path, dataset_rows)
        spec["path"] = name
        dataset_specs[dataset] = spec
    reviews = []
    for dataset_rows in rows.values():
        for row in dataset_rows:
            reviews.append(
                {
                    "unit_id": row["example_id"],
                    "review_id": f"review-{row['example_id']}",
                    "decision": "accepted",
                    "dimensions": {
                        name: "not_applicable" if name == "contamination" else "pass"
                        for name in (
                            "structural",
                            "tool_integrity",
                            "tool_correctness",
                            "observation_grounding",
                            "privacy",
                            "reasoning_exclusion",
                            "provenance",
                            "task_quality",
                            "human_review",
                            "contamination",
                            "deduplication",
                        )
                    },
                    "review_id": f"review-{row['example_id']}",
                }
            )
    review_path = pilot / "reviews.jsonl"
    review_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in reviews)
    review_path.write_bytes(review_raw)
    manifest = {
        "schema_version": "ai-data-extraction/reviewed-pilot/v1",
        "privacy": {"training_authorized": False},
        "review_manifest": {"sha256": "review-manifest"},
        "datasets": dataset_specs,
        "reviews": {
            "path": "reviews.jsonl",
            "records": len(reviews),
            "bytes": len(review_raw),
            "sha256": hashlib.sha256(review_raw).hexdigest(),
        },
    }
    manifest_path = pilot / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return pilot, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _write_authorization(path: Path, pilot_sha: str) -> None:
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "pilot_manifest_sha256": pilot_sha,
        "training_authorized": True,
        "authorized_at": "2026-09-18T00:00:00Z",
        "privacy_approval": {
            "approved": True,
            "authority_basis": "test-fixture-only",
            "approved_at": "2026-09-18T00:00:00Z",
        },
        "trainer_certification": {
            "status": "passed",
            "cert_revision": "trainer-cert/test-v1",
        },
    }
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


class TrainerExportTests(unittest.TestCase):
    def test_audit_rejects_tool_rows_without_schemas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot, _ = _write_pilot(
                root,
                {"sft": [_row("sft-unit", "parent-sft")], "tool_traces": [_row("tool-unit", "parent-tool", tool=True, schema=False)]},
            )
            audit = inspect_pilot(pilot)
            self.assertEqual(audit["accepted_reviews"], 2)
            self.assertEqual(audit["exportable_rows"], 1)
            self.assertEqual(audit["reason_counts"], {"tool_schema_missing_or_invalid": 1})

    def test_export_requires_explicit_authorization_and_emits_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot, pilot_sha = _write_pilot(
                root,
                {"sft": [_row("sft-unit", "parent-sft")], "tool_traces": [_row("tool-unit", "parent-tool", tool=True)]},
            )
            authorization = root / "authorization.json"
            _write_authorization(authorization, pilot_sha)
            output = root / "trainer"
            result = export_trainer_release(pilot, authorization, output)
            self.assertTrue(result["training_authorized"])
            self.assertEqual(result["counts"], {"accepted_reviews": 2, "sft": 1, "tool_sft": 1})
            exported = [json.loads(line) for line in (output / "tool_sft.jsonl").read_text().splitlines()]
            self.assertEqual(exported[0]["schema_version"], TRAINER_EXAMPLE_SCHEMA)
            self.assertEqual(set(exported[0]), {"schema_version", "example_id", "split", "messages", "tools"})
            self.assertFalse((output / "tool_sft.jsonl").read_text().find("metadata") >= 0)

    def test_export_refuses_non_exportable_accepted_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot, pilot_sha = _write_pilot(
                root,
                {"tool_traces": [_row("tool-unit", "parent-tool", tool=True, schema=False)]},
            )
            authorization = root / "authorization.json"
            _write_authorization(authorization, pilot_sha)
            with self.assertRaisesRegex(ValueError, "not all trainer-exportable"):
                export_trainer_release(pilot, authorization, root / "trainer")


if __name__ == "__main__":
    unittest.main()
