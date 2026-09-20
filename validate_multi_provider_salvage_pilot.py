#!/usr/bin/env python3
"""Validate the multi-provider salvage package with the pinned JSON loader."""

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

from build_multi_provider_salvage_pilot import (
    DECISION_SCHEMA,
    LINEAGE_SCHEMA,
    PACKAGE_SCHEMA,
    TOOL_REVIEW_SCHEMA,
    TRAINER_SCHEMA,
    _load_source_bindings,
)
from build_training_data import no_reasoning_content, trainer_marker_count


VALIDATION_SCHEMA = "ai-data-extraction/multi-provider-salvage-loader-validation/v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


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


def _json_text(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{field} is not JSON text")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is invalid JSON text") from exc


def _verify_sources(manifest: dict[str, Any]) -> dict[str, Any]:
    input_spec = manifest.get("input_spec")
    if not isinstance(input_spec, dict):
        raise ValueError("input spec binding is missing")
    spec_path = Path(str(input_spec.get("path"))).resolve()
    if not spec_path.is_file() or _sha256_file(spec_path) != input_spec.get("sha256"):
        raise ValueError("input spec changed")
    _spec, bindings = _load_source_bindings(spec_path)
    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, list):
        raise ValueError("manifest has no source bindings")
    by_id = {str(item.get("source_id")): item for item in manifest_sources if isinstance(item, dict)}
    if len(by_id) != len(bindings):
        raise ValueError("source binding count mismatch")
    source_records = 0
    source_keys: set[tuple[str, int]] = set()
    verified: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        source_id = binding["source_id"]
        manifest_binding = by_id.get(source_id)
        if not isinstance(manifest_binding, dict):
            raise ValueError(f"source binding missing from package: {source_id}")
        for key in ("sha256", "records", "bytes", "manifest_sha256"):
            if manifest_binding.get(key) != binding.get(key):
                raise ValueError(f"source binding mismatch: {source_id}.{key}")
        digest = hashlib.sha256()
        bytes_seen = 0
        records_seen = 0
        with binding["path"].open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                digest.update(raw)
                bytes_seen += len(raw)
                if raw.strip():
                    records_seen += 1
                    source_keys.add((source_id, line_number))
        if records_seen != binding["records"] or bytes_seen != binding["bytes"]:
            raise ValueError(f"source count/byte mismatch: {source_id}")
        if digest.hexdigest() != binding["sha256"]:
            raise ValueError(f"source digest mismatch: {source_id}")
        source_records += records_seen
        verified[source_id] = {
            "records": records_seen,
            "bytes": bytes_seen,
            "sha256": digest.hexdigest(),
        }
    return {"records": source_records, "keys": source_keys, "sources": verified}


def _load_partition(
    path: Path,
    *,
    expected_records: int,
    expected_columns: set[str],
    split: str,
) -> tuple[Any | None, dict[str, Any]]:
    if expected_records == 0:
        if path.stat().st_size != 0:
            raise ValueError(f"empty partition is not empty: {path.name}")
        return None, {"rows": 0, "columns": sorted(expected_columns)}
    dataset = load_dataset("json", data_files=str(path), split="train")
    if len(dataset) != expected_records:
        raise ValueError(f"{path.name} row count mismatch")
    if set(dataset.column_names) != expected_columns:
        raise ValueError(f"{path.name} columns mismatch: {sorted(dataset.column_names)}")
    if set(dataset["split"]) != {split}:
        raise ValueError(f"{path.name} split field mismatch")
    return dataset, {"rows": len(dataset), "columns": sorted(dataset.column_names)}


def _validate_sft(dataset: Any | None, *, partition: str) -> int:
    if dataset is None:
        return 0
    for row in dataset:
        if row.get("schema_version") != TRAINER_SCHEMA:
            raise ValueError(f"{partition} trainer schema mismatch")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{partition} messages missing")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"{partition} message invalid")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{partition} message content invalid")
            if message.get("role") == "tool" or message.get("tool_calls"):
                raise ValueError(f"{partition} contains tool activity")
        if not no_reasoning_content(row) or trainer_marker_count(row):
            raise ValueError(f"{partition} reasoning/marker firewall failed")
    return len(dataset)


TOOL_COLUMNS_LEGACY = {
    "schema_version",
    "example_id",
    "split",
    "provider",
    "agent",
    "model_tier",
    "model_tier_basis",
    "quality_tier",
    "quality_gate",
    "quality_reason_json",
    "privacy_state",
    "privacy_structural_redactions",
    "tool_schema_status",
    "verifier_status",
    "reward_status",
    "parent_record_sha256",
    "source_example_id",
    "source_file_sha256",
    "source_file_name",
    "source_line",
    "source_row_sha256",
    "source_message_range_json",
    "chunk_index",
    "chunk_count",
    "continuation_status",
    "previous_example_id",
    "next_example_id",
    "tool_families_json",
    "tags_json",
    "events_summary_json",
    "messages",
    "events",
}
TOOL_COLUMNS = TOOL_COLUMNS_LEGACY | {
    "source_origin_file_sha256",
    "source_origin_file_name",
    "source_origin_json",
}


def _validate_tool(dataset: Any | None, *, partition: str) -> tuple[int, set[str], set[str]]:
    if dataset is None:
        return 0, set(), set()
    parents: set[str] = set()
    examples: set[str] = set()
    for row in dataset:
        if row.get("schema_version") != TOOL_REVIEW_SCHEMA:
            raise ValueError(f"{partition} tool schema mismatch")
        example_id = row.get("example_id")
        parent = row.get("parent_record_sha256")
        if not isinstance(example_id, str) or not example_id or example_id in examples:
            raise ValueError(f"{partition} example identity missing or duplicated")
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"{partition} parent identity missing")
        examples.add(example_id)
        parents.add(parent)
        if row.get("tool_schema_status") != "not_observed":
            raise ValueError(f"{partition} schema was promoted")
        if row.get("verifier_status") != "not_observed" or row.get("reward_status") != "not_exported":
            raise ValueError(f"{partition} verifier/reward state was promoted")
        messages = row.get("messages")
        events = row.get("events")
        if not isinstance(messages, list) or not messages or not isinstance(events, list) or not events:
            raise ValueError(f"{partition} payload missing")
        if not all(isinstance(event, str) for event in events):
            raise ValueError(f"{partition} events are not JSON text")
        for event in events:
            if not isinstance(json.loads(event), dict):
                raise ValueError(f"{partition} event text is not an object")
        _json_text(row.get("quality_reason_json"), "quality_reason_json")
        _json_text(row.get("events_summary_json"), "events_summary_json")
        _json_text(row.get("source_message_range_json"), "source_message_range_json")
        _json_text(row.get("tool_families_json"), "tool_families_json")
        _json_text(row.get("tags_json"), "tags_json")
        if not no_reasoning_content({"messages": messages, "events": events}) or trainer_marker_count(
            {"messages": messages, "events": events}
        ):
            raise ValueError(f"{partition} reasoning/marker firewall failed")
    return len(dataset), parents, examples


def validate_package(package_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("unexpected multi-provider package schema")
    if manifest.get("training_authorized") is not False:
        raise ValueError("package is training-authorized")
    before_manifest_sha = _sha256_file(manifest_path)
    source_state = _verify_sources(manifest)
    files = manifest.get("files")
    counts = manifest.get("counts")
    if not isinstance(files, dict) or not isinstance(counts, dict):
        raise ValueError("package has no files/counts")
    for filename, descriptor in files.items():
        path = package_dir / filename
        if not path.is_file() or _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"package file digest mismatch: {filename}")
        if path.stat().st_size != descriptor.get("bytes"):
            raise ValueError(f"package file byte mismatch: {filename}")

    partition_specs = (
        ("sft_tier1_train.jsonl", "sft_tier1_train", "train"),
        ("sft_tier1_validation.jsonl", "sft_tier1_validation", "validation"),
        ("sft_optional_train.jsonl", "sft_optional_train", "train"),
        ("sft_optional_validation.jsonl", "sft_optional_validation", "validation"),
    )
    loaded: dict[str, Any] = {}
    all_sft_examples: set[str] = set()
    for filename, count_key, split in partition_specs:
        dataset, info = _load_partition(
            package_dir / filename,
            expected_records=int(counts[count_key]),
            expected_columns={"schema_version", "example_id", "split", "messages"},
            split=split,
        )
        _validate_sft(dataset, partition=filename)
        loaded[count_key] = info
        if dataset is not None:
            for example_id in dataset["example_id"]:
                if example_id in all_sft_examples:
                    raise ValueError("duplicate SFT example across partitions")
                all_sft_examples.add(example_id)

    tool_specs = (
        ("tool_review_train.jsonl", "tool_review_train", "train"),
        ("tool_review_validation.jsonl", "tool_review_validation", "validation"),
    )
    tool_columns = TOOL_COLUMNS if isinstance(manifest.get("lineage_contract"), dict) else TOOL_COLUMNS_LEGACY
    all_tool_parents: dict[str, str] = {}
    all_tool_examples: set[str] = set()
    for filename, count_key, split in tool_specs:
        dataset, info = _load_partition(
            package_dir / filename,
            expected_records=int(counts[count_key]),
            expected_columns=tool_columns,
            split=split,
        )
        tool_count, parents, examples = _validate_tool(dataset, partition=filename)
        loaded[count_key] = info
        if tool_count != int(counts[count_key]):
            raise ValueError(f"{filename} tool count mismatch")
        for example_id in examples:
            if example_id in all_sft_examples or example_id in all_tool_examples:
                raise ValueError("duplicate output example across package")
            all_tool_examples.add(example_id)
        for parent in parents:
            if parent in all_tool_parents and all_tool_parents[parent] != split:
                raise ValueError("tool parent crosses train/validation")
            all_tool_parents[parent] = split

    lineage_path = package_dir / "lineage.jsonl"
    lineage_rows: list[dict[str, Any]] = []
    parent_splits: dict[str, str] = {}
    lineage_examples: set[str] = set()
    with lineage_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != LINEAGE_SCHEMA:
                raise ValueError("lineage schema mismatch")
            example_id = row.get("example_id")
            parent = row.get("parent_record_sha256")
            split = row.get("split")
            if not isinstance(example_id, str) or example_id in lineage_examples:
                raise ValueError("lineage example identity missing or duplicated")
            if not isinstance(parent, str) or not parent or split not in {"train", "validation"}:
                raise ValueError("lineage parent/split missing")
            prior = parent_splits.setdefault(parent, split)
            if prior != split:
                raise ValueError("parent crosses package train/validation")
            lineage_examples.add(example_id)
            lineage_rows.append(row)
    if len(lineage_rows) != int(files["lineage.jsonl"]["records"]):
        raise ValueError("lineage record count mismatch")
    if lineage_examples != all_sft_examples | all_tool_examples:
        raise ValueError("lineage does not cover output examples")

    decision_keys: set[tuple[str, int]] = set()
    decision_count = 0
    with (package_dir / "decisions.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            decision_count += 1
            if row.get("schema_version") != DECISION_SCHEMA:
                raise ValueError("decision schema mismatch")
            key = (str(row.get("source_binding_id")), int(row.get("source_line", -1)))
            if key in decision_keys:
                raise ValueError("duplicate decision source key")
            decision_keys.add(key)
            if row.get("rl_decision") != "not_exported" or row.get("rl_reason") != "executable_reward_not_observed":
                raise ValueError("decision contains RL promotion")
            if not isinstance(row.get("sft_reasons"), list) or not isinstance(row.get("tool_review_reasons"), list):
                raise ValueError("decision reasons are not lists")
    if decision_count != int(counts["decision_records"]):
        raise ValueError("decision record count mismatch")
    if decision_keys != source_state["keys"]:
        raise ValueError("decisions do not cover every source row")

    report = {
        "schema_version": VALIDATION_SCHEMA,
        "package_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "source_bindings": source_state["sources"],
        "source_records": source_state["records"],
        "loaded": loaded,
        "sft_examples": len(all_sft_examples),
        "tool_review_examples": len(all_tool_examples),
        "lineage_records": len(lineage_rows),
        "unique_global_parents": len(parent_splits),
        "decision_records": decision_count,
        "parent_split": "passed",
        "reasoning_firewall": "passed",
        "schema_inference": "none",
        "verifier_inference": "none",
        "reward_export": "none",
        "training_authorized": False,
        "result": "passed",
    }
    if report_path is None:
        report_path = package_dir / "loader_validation.json"
    report_path = report_path.resolve()
    _atomic_write(report_path, _canonical(report) + b"\n")
    report_sha = _sha256_file(report_path)
    manifest.setdefault("validation", {}).update(
        {
            "source_bindings": "passed",
            "loader": "passed",
            "parent_split": "passed",
            "trainer_projection": "passed",
            "reasoning_firewall": "passed",
            "tool_schema": "not_observed_not_inferred",
            "verifier": "not_observed",
            "reward": "not_present",
            "loader_validation": {
                "status": "passed",
                "loader": "datasets.load_dataset",
                "datasets_version": datasets_version,
                "report": report_path.name,
                "report_sha256": report_sha,
                "source_records": source_state["records"],
                "sft_examples": len(all_sft_examples),
                "tool_review_examples": len(all_tool_examples),
            },
        }
    )
    _atomic_write(manifest_path, _canonical(manifest) + b"\n")
    report["report_sha256"] = report_sha
    report["package_manifest_sha256_after"] = _sha256_file(manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_package(args.package_dir, report_path=args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VALIDATION_SCHEMA", "validate_package"]
