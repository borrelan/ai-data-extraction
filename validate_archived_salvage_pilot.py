#!/usr/bin/env python3
"""Validate the composed archived salvage pilot with the JSON dataset loader."""

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

from build_archived_salvage_pilot import (
    ARCHIVED_SALVAGE_SCHEMA,
    HISTORICAL_SCHEMA,
    LINEAGE_SCHEMA,
    TRAINER_EXAMPLE_SCHEMA,
)


VALIDATION_SCHEMA = "ai-data-extraction/archived-salvage-loader-validation/v1"


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _verify_descriptor(root: Path, filename: str, descriptor: dict[str, Any]) -> None:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_file(path) != descriptor.get("sha256"):
        raise ValueError(f"file digest mismatch: {path}")
    expected_records = descriptor.get("records")
    if not isinstance(expected_records, int):
        raise ValueError(f"file record count missing: {path}")
    records = sum(1 for line in path.open("rb") if line.strip())
    if records != expected_records:
        raise ValueError(f"file record count mismatch: {path}: {records} != {expected_records}")


def _verify_inputs(manifest: dict[str, Any]) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("manifest has no input bindings")
    for name, binding in inputs.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid input binding: {name}")
        root = Path(str(binding.get("path")))
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if _sha256_file(manifest_path) != binding.get("manifest_sha256"):
            raise ValueError(f"input manifest changed: {name}")
        source_manifest = _load_json(manifest_path)
        if source_manifest.get("schema_version") != binding.get("schema_version"):
            raise ValueError(f"input schema changed: {name}")
        if source_manifest.get("training_authorized") is not False:
            raise ValueError(f"input authorization changed: {name}")
        files = binding.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"input file bindings missing: {name}")
        for filename, descriptor in files.items():
            if not isinstance(descriptor, dict):
                raise ValueError(f"invalid input descriptor: {name}/{filename}")
            _verify_descriptor(root, filename, descriptor)


def _load_partition(path: Path, expected_rows: int) -> Any:
    if expected_rows == 0:
        if path.stat().st_size != 0:
            raise ValueError(f"zero-row partition is not empty: {path}")
        return None
    return load_dataset("json", data_files=str(path), split="train")


def _validate_trainer_partition(
    dataset: Any,
    *,
    name: str,
    expected_rows: int,
    has_tools: bool,
) -> dict[str, Any]:
    expected_columns = {"schema_version", "example_id", "split", "messages"}
    if has_tools:
        expected_columns.add("tools")
    if len(dataset) != expected_rows:
        raise ValueError(f"{name} row count mismatch")
    if set(dataset.column_names) != expected_columns:
        raise ValueError(f"{name} columns mismatch: {sorted(dataset.column_names)}")
    ids: set[str] = set()
    parents_not_available = 0
    for row in dataset:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in ids:
            raise ValueError(f"{name} example identity missing or duplicated")
        ids.add(example_id)
        if row.get("schema_version") != TRAINER_EXAMPLE_SCHEMA:
            raise ValueError(f"{name} schema mismatch")
        if row.get("split") not in {"train", "validation"}:
            raise ValueError(f"{name} split invalid")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{name} messages missing")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"{name} message invalid")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{name} message content invalid")
        if has_tools and (not isinstance(row.get("tools"), list) or not row["tools"]):
            raise ValueError(f"{name} tool schema missing")
        blob = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in blob for marker in ("<think>", "<analysis>", "chain_of_thought", "hidden_reasoning")):
            raise ValueError(f"{name} reasoning marker present")
    return {"rows": len(dataset), "columns": sorted(dataset.column_names), "ids": ids}


def _validate_tool_review_partition(dataset: Any, *, name: str, expected_rows: int) -> dict[str, Any]:
    if len(dataset) != expected_rows:
        raise ValueError(f"{name} row count mismatch")
    expected_columns = {
        "agent",
        "events",
        "events_summary",
        "example_id",
        "lineage",
        "messages",
        "model_tier",
        "privacy",
        "provider",
        "quality_reason",
        "quality_tier",
        "schema_version",
        "session_quality",
        "split",
        "tags",
        "tool_contract",
        "tool_families",
    }
    if set(dataset.column_names) != expected_columns:
        raise ValueError(f"{name} columns mismatch: {sorted(dataset.column_names)}")
    ids: set[str] = set()
    action_events = 0
    observation_events = 0
    for row in dataset:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in ids:
            raise ValueError(f"{name} example identity missing or duplicated")
        ids.add(example_id)
        if row.get("schema_version") != HISTORICAL_SCHEMA:
            raise ValueError(f"{name} schema mismatch")
        if row.get("split") not in {"train", "validation"}:
            raise ValueError(f"{name} split invalid")
        events = row.get("events")
        summary = row.get("events_summary")
        if not isinstance(events, list) or not isinstance(summary, dict):
            raise ValueError(f"{name} events missing")
        if summary.get("event_count") != len(events):
            raise ValueError(f"{name} event count mismatch")
        if any(not isinstance(event, str) for event in events):
            raise ValueError(f"{name} event representation changed")
        action_events += int(summary.get("action_count") or 0)
        observation_events += int(summary.get("observation_count") or 0)
        contract = row.get("tool_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"{name} tool contract missing")
        if contract.get("schema_status") != "not_observed":
            raise ValueError(f"{name} contains inferred schema")
        if contract.get("verification_status") != "not_observed":
            raise ValueError(f"{name} contains verifier claim")
        if contract.get("reward_status") != "not_exported":
            raise ValueError(f"{name} contains reward claim")
        privacy = row.get("privacy")
        if not isinstance(privacy, dict) or privacy.get("eligible_for_training") is not False:
            raise ValueError(f"{name} privacy gate changed")
        blob = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in blob for marker in ("<think>", "<analysis>", "chain_of_thought", "hidden_reasoning")):
            raise ValueError(f"{name} reasoning marker present")
    return {
        "rows": len(dataset),
        "columns": sorted(dataset.column_names),
        "ids": ids,
        "action_events": action_events,
        "observation_events": observation_events,
    }


def _validate_lineage_and_decisions(
    pilot_dir: Path,
    manifest: dict[str, Any],
    partition_ids: set[str],
) -> dict[str, Any]:
    lineage_ids: set[str] = set()
    parent_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    lineage_path = pilot_dir / "lineage.jsonl"
    for line in lineage_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != LINEAGE_SCHEMA:
            raise ValueError("lineage schema mismatch")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or example_id in lineage_ids:
            raise ValueError("lineage identity missing or duplicated")
        lineage_ids.add(example_id)
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("lineage split invalid")
        parent = row.get("parent_record_sha256")
        if not isinstance(parent, str) or not parent:
            raise ValueError("lineage parent missing")
        parent_by_split[row["split"]].add(parent)
    if lineage_ids != partition_ids:
        raise ValueError("lineage does not cover output partitions exactly")
    if parent_by_split["train"] & parent_by_split["validation"]:
        raise ValueError("parent crosses global train/validation split")

    decision_count = 0
    exclusion_count = 0
    with (pilot_dir / "decisions.jsonl").open(encoding="utf-8") as decisions:
        for line in decisions:
            if not line.strip():
                continue
            row = json.loads(line)
            decision_count += 1
            if row.get("schema_version") != "ai-data-extraction/archived-salvage-decision/v1":
                raise ValueError("decision schema mismatch")
            if row.get("training_authorized") is not False:
                raise ValueError("decision contains training authorization")
            if row.get("decision") != "selected":
                exclusion_count += 1
                reasons = row.get("reasons")
                if not isinstance(reasons, list) or not reasons:
                    raise ValueError("excluded decision has no reason")
    expected_decisions = int(manifest["counts"]["decision_records"])
    if decision_count != expected_decisions:
        raise ValueError("decision count mismatch")
    return {
        "records": len(lineage_ids),
        "unique_parents": len(parent_by_split["train"] | parent_by_split["validation"]),
        "train_parents": len(parent_by_split["train"]),
        "validation_parents": len(parent_by_split["validation"]),
        "decision_records": decision_count,
        "excluded_decisions": exclusion_count,
    }


def validate_pilot(pilot_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    pilot_dir = pilot_dir.resolve()
    manifest_path = pilot_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != ARCHIVED_SALVAGE_SCHEMA:
        raise ValueError("unexpected archived salvage schema")
    if manifest.get("training_authorized") is not False:
        raise ValueError("archived salvage pilot is training-authorized")
    before_manifest_sha = _sha256_file(manifest_path)
    _verify_inputs(manifest)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("pilot has no output file descriptors")
    for filename, descriptor in files.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid output descriptor: {filename}")
        _verify_descriptor(pilot_dir, filename, descriptor)

    counts = manifest.get("counts") or {}
    loaded: dict[str, dict[str, Any]] = {}
    partition_ids: set[str] = set()
    for name, filename, count_key, has_tools in (
        ("sft_train", "sft_train.jsonl", "sft_train", False),
        ("sft_validation", "sft_validation.jsonl", "sft_validation", False),
        ("tool_sft_train", "tool_sft_train.jsonl", "tool_sft_train", True),
    ):
        dataset = _load_partition(pilot_dir / filename, int(counts[count_key]))
        if dataset is None:
            raise ValueError(f"non-empty partition unexpectedly empty: {name}")
        details = _validate_trainer_partition(
            dataset,
            name=name,
            expected_rows=int(counts[count_key]),
            has_tools=has_tools,
        )
        loaded[name] = {key: value for key, value in details.items() if key != "ids"}
        partition_ids.update(details["ids"])

    if int(counts["tool_sft_validation"]) == 0:
        if (pilot_dir / "tool_sft_validation.jsonl").stat().st_size != 0:
            raise ValueError("empty tool SFT validation partition is not empty")
        loaded["tool_sft_validation"] = {"rows": 0, "columns": []}
    else:
        dataset = _load_partition(
            pilot_dir / "tool_sft_validation.jsonl", int(counts["tool_sft_validation"])
        )
        assert dataset is not None
        details = _validate_trainer_partition(
            dataset,
            name="tool_sft_validation",
            expected_rows=int(counts["tool_sft_validation"]),
            has_tools=True,
        )
        loaded["tool_sft_validation"] = {key: value for key, value in details.items() if key != "ids"}
        partition_ids.update(details["ids"])

    for name, filename, count_key in (
        ("tool_trajectory_train", "tool_trajectory_train.jsonl", "tool_trajectory_train"),
        ("tool_trajectory_validation", "tool_trajectory_validation.jsonl", "tool_trajectory_validation"),
    ):
        dataset = _load_partition(pilot_dir / filename, int(counts[count_key]))
        if dataset is None:
            raise ValueError(f"non-empty partition unexpectedly empty: {name}")
        details = _validate_tool_review_partition(
            dataset, name=name, expected_rows=int(counts[count_key])
        )
        loaded[name] = {key: value for key, value in details.items() if key != "ids"}
        partition_ids.update(details["ids"])

    lineage = _validate_lineage_and_decisions(pilot_dir, manifest, partition_ids)
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "pilot_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "format": "partitioned_jsonl",
        "loaded": loaded,
        "lineage": lineage,
        "input_artifact_hashes": "passed",
        "parent_disjoint": True,
        "trainer_projection": "passed",
        "tool_review_projection": "passed",
        "schema_inference": "none",
        "verifier_inference": "none",
        "reward_export": "none",
        "training_authorized": False,
        "result": "passed",
    }
    if report_path is None:
        report_path = pilot_dir / "loader_validation.json"
    report_path = report_path.resolve()
    _atomic_write(report_path, _canonical_bytes(report) + b"\n")
    report_sha = _sha256_file(report_path)
    validation = manifest.setdefault("validation", {})
    validation.update(
        {
            "input_artifact_hashes": "passed",
            "trainer_projection": "passed",
            "tool_review_projection": "passed",
            "parent_disjoint": "passed",
            "reasoning_firewall": "passed",
            "schema_inference": "none",
            "verifier_inference": "none",
            "reward": "not_present",
            "loader": "passed",
        }
    )
    validation["loader_validation"] = {
        "status": "passed",
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "report": report_path.name,
        "report_sha256": report_sha,
        "loaded": loaded,
    }
    _atomic_write(manifest_path, _canonical_bytes(manifest) + b"\n")
    report["report_sha256"] = report_sha
    report["pilot_manifest_sha256_after"] = _sha256_file(manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_pilot(args.pilot_dir, report_path=args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VALIDATION_SCHEMA", "validate_pilot"]
