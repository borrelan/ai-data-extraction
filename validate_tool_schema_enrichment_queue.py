#!/usr/bin/env python3
"""Validate the schema-enrichment queue with the actual JSON loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from datasets import __version__ as datasets_version
from datasets import load_dataset

from build_tool_schema_enrichment_queue import QUEUE_SCHEMA, TASK_SCHEMA


VALIDATION_SCHEMA = "ai-data-extraction/tool-schema-enrichment-loader-validation/v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(raw)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_object(text: Any, field: str) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"{field} is not stable JSON text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is invalid JSON text") from exc


def _validate_rows(
    dataset: Any,
    *,
    expected_rows: int,
    expected_schema: str,
    split_name: str,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    if len(dataset) != expected_rows:
        raise ValueError(f"{split_name} row count mismatch: {len(dataset)} != {expected_rows}")
    examples: set[str] = set()
    parents: set[str] = set()
    action_count = 0
    observation_count = 0
    for row in dataset:
        if row.get("schema_version") != expected_schema:
            raise ValueError(f"{split_name} schema mismatch")
        queue_id = row.get("queue_id")
        if not isinstance(queue_id, str) or not queue_id or queue_id in examples:
            raise ValueError(f"{split_name} contains a missing or duplicate queue_id")
        examples.add(queue_id)
        parent = row.get("parent_record_sha256")
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"{split_name} contains incomplete parent lineage")
        parents.add(parent)
        if row.get("training_authorized") is not False:
            raise ValueError(f"{split_name} contains training authorization")
        if row.get("schema_status") != "not_observed":
            raise ValueError(f"{split_name} contains an inferred tool schema")
        if row.get("verification_status") != "not_observed":
            raise ValueError(f"{split_name} contains a verifier claim")
        if row.get("reward_status") != "not_exported":
            raise ValueError(f"{split_name} contains a reward claim")
        if not isinstance(row.get("messages_json"), str):
            raise ValueError(f"{split_name} messages are not stable JSON text")
        if not isinstance(_json_object(row["messages_json"], "messages_json"), list):
            raise ValueError(f"{split_name} messages are not a list")
        pairs = _json_object(row.get("event_pairs_json"), "event_pairs_json")
        if not isinstance(pairs, list):
            raise ValueError(f"{split_name} event pairs are not a list")
        if int(row.get("action_count", -1)) != len(pairs):
            raise ValueError(f"{split_name} action count does not match event pairs")
        if int(row.get("observation_count", -1)) != sum(
            1 for pair in pairs if isinstance(pair, dict) and pair.get("observation_present") is True
        ):
            raise ValueError(f"{split_name} observation count does not match event pairs")
        action_count += int(row["action_count"])
        observation_count += int(row["observation_count"])
        if not isinstance(row.get("action_names_json"), str):
            raise ValueError(f"{split_name} action names are not stable JSON text")
        if not isinstance(_json_object(row["action_names_json"], "action_names_json"), list):
            raise ValueError(f"{split_name} action names are not a list")
        for field in (
            "quality_reason_json",
            "tool_families_json",
            "tags_json",
            "registry_candidate_names_json",
            "replay_blockers_json",
            "source_metadata_json",
            "lineage_json",
        ):
            _json_object(row.get(field), field)
        messages_blob = row["messages_json"].lower()
        events_blob = row["event_pairs_json"].lower()
        if any(
            marker in messages_blob or marker in events_blob
            for marker in ("<think>", "<analysis>", "chain_of_thought", "hidden_reasoning")
        ):
            raise ValueError(f"{split_name} contains a reasoning marker")
        if selected_ids is not None and queue_id not in selected_ids:
            raise ValueError(f"{split_name} is not a subset of queue rows")
    if selected_ids is not None and examples != selected_ids:
        raise ValueError(f"{split_name} selection set mismatch")
    return {
        "rows": len(dataset),
        "columns": sorted(dataset.column_names),
        "unique_parents": len(parents),
        "action_events": action_count,
        "observation_events": observation_count,
        "queue_ids": examples,
        "parents": parents,
    }


def validate_queue(queue_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    queue_dir = queue_dir.resolve()
    manifest_path = queue_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unexpected queue schema")
    if manifest.get("training_authorized") is not False:
        raise ValueError("queue is training-authorized")
    before_manifest_sha = _sha256_file(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("queue manifest has no file descriptors")
    for filename, descriptor in files.items():
        path = queue_dir / filename
        if not path.is_file() or _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"queue file digest mismatch: {filename}")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("queue has no source binding")
    pilot_manifest_path = Path(str(source.get("pilot_dir"))) / "manifest.json"
    if not pilot_manifest_path.is_file():
        raise ValueError("bound pilot manifest is missing")
    if _sha256_file(pilot_manifest_path) != source.get("pilot_manifest_sha256"):
        raise ValueError("bound pilot manifest changed")

    counts = manifest.get("counts") or {}
    # These are intentionally different schemas: replay tasks add selection
    # fields and use TASK_SCHEMA. Load them separately so the JSON loader does
    # not try to cast one schema into the other.
    queue_dataset = load_dataset(
        "json", data_files=str(queue_dir / "queue.jsonl"), split="train"
    )
    replay_dataset = load_dataset(
        "json", data_files=str(queue_dir / "replay_tasks.jsonl"), split="train"
    )
    queue_details = _validate_rows(
        queue_dataset,
        expected_rows=int(counts["queue_records"]),
        expected_schema=QUEUE_SCHEMA,
        split_name="queue",
    )
    replay_details = _validate_rows(
        replay_dataset,
        expected_rows=int(counts["replay_records"]),
        expected_schema=TASK_SCHEMA,
        split_name="replay",
    )
    if len(replay_details["parents"]) != replay_details["rows"]:
        raise ValueError("replay tasks are not parent-diverse")
    if replay_details["queue_ids"] - queue_details["queue_ids"]:
        raise ValueError("replay contains a queue id outside queue.jsonl")
    queue_train = {
        row["parent_record_sha256"] for row in queue_dataset if row["split"] == "train"
    }
    queue_validation = {
        row["parent_record_sha256"]
        for row in queue_dataset
        if row["split"] == "validation"
    }
    if queue_train & queue_validation:
        raise ValueError("queue parent sessions cross train/validation")

    decision_records = 0
    decision_ids: set[str] = set()
    replay_decision_ids: set[str] = set()
    with (queue_dir / "decisions.jsonl").open(encoding="utf-8") as decisions:
        for line in decisions:
            if not line.strip():
                continue
            row = json.loads(line)
            decision_records += 1
            queue_id = row.get("queue_id")
            if not isinstance(queue_id, str) or queue_id in decision_ids:
                raise ValueError("decision queue ids are missing or duplicated")
            decision_ids.add(queue_id)
            if row.get("training_authorized") is not False:
                raise ValueError("decision contains training authorization")
            if row.get("replay_selected") is True:
                replay_decision_ids.add(queue_id)
    if decision_records != int(counts["decision_records"]):
        raise ValueError("decision count mismatch")
    if decision_ids != queue_details["queue_ids"]:
        raise ValueError("decision ids do not cover queue rows")
    if replay_decision_ids != replay_details["queue_ids"]:
        raise ValueError("decision replay selection does not match replay tasks")

    report = {
        "schema_version": VALIDATION_SCHEMA,
        "queue_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "format": "jsonl",
        "loaded": {
            "queue": {
                key: value for key, value in queue_details.items() if key not in {"queue_ids", "parents"}
            },
            "replay": {
                key: value for key, value in replay_details.items() if key not in {"queue_ids", "parents"}
            },
        },
        "decision_records": decision_records,
        "parent_disjoint": True,
        "replay_parent_diverse": True,
        "schema_inference": "none",
        "verifier_inference": "none",
        "reward_export": "none",
        "training_authorized": False,
        "result": "passed",
    }
    if report_path is None:
        report_path = queue_dir / "loader_validation.json"
    report_path = report_path.resolve()
    _atomic_write(report_path, _canonical_bytes(report) + b"\n")
    report_sha = _sha256_file(report_path)
    validation = manifest.setdefault("validation", {})
    validation.update(
        {
            "loader": "passed",
            "parent_disjoint": "passed",
            "replay_parent_diverse": "passed",
        }
    )
    validation["loader_validation"] = {
        "status": "passed",
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "report": report_path.name,
        "report_sha256": report_sha,
        "counts": {
            "queue": report["loaded"]["queue"],
            "replay": report["loaded"]["replay"],
        },
        "parent_disjoint": True,
        "replay_parent_diverse": True,
    }
    _atomic_write(manifest_path, _canonical_bytes(manifest) + b"\n")
    report["report_sha256"] = report_sha
    report["queue_manifest_sha256_after"] = _sha256_file(manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_queue(args.queue_dir, report_path=args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VALIDATION_SCHEMA", "validate_queue"]
