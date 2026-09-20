#!/usr/bin/env python3
"""Validate the historical trajectory pilot with the Hugging Face JSON loader."""

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


PILOT_SCHEMA = "ai-data-extraction/historical-tool-trajectory-pilot/v1"
VALIDATION_SCHEMA = "ai-data-extraction/historical-tool-trajectory-loader-validation/v1"


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


def _validate_split(dataset: Any, *, split: str, expected_rows: int) -> dict[str, Any]:
    if len(dataset) != expected_rows:
        raise ValueError(f"{split} row count mismatch: {len(dataset)} != {expected_rows}")
    if set(dataset["split"]) != {split}:
        raise ValueError(f"{split} contains another split value")
    parent_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    examples: set[str] = set()
    action_count = 0
    observation_count = 0
    for row in dataset:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in examples:
            raise ValueError(f"{split} contains a missing or duplicate example_id")
        examples.add(example_id)
        lineage = row.get("lineage")
        if not isinstance(lineage, dict) or not isinstance(lineage.get("parent_record_sha256"), str):
            raise ValueError(f"{split} contains incomplete lineage")
        parent_by_split[split].add(lineage["parent_record_sha256"])
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{split} contains empty messages")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"{split} contains an invalid message role")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{split} contains non-string message content")
        events = row.get("events")
        summary = row.get("events_summary")
        if not isinstance(events, list) or not events or not isinstance(summary, dict):
            raise ValueError(f"{split} contains incomplete event trajectory")
        if summary.get("event_count") != len(events):
            raise ValueError(f"{split} event summary count mismatch")
        if any(not isinstance(event, str) for event in events):
            raise ValueError(f"{split} events are not stable JSON text")
        action_count += int(summary.get("action_count", 0))
        observation_count += int(summary.get("observation_count", 0))
        for message in messages:
            for call in message.get("tool_calls", []) or []:
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and not isinstance(function.get("arguments"), str):
                    raise ValueError(f"{split} tool arguments are not stable JSON text")
        contract = row.get("tool_contract")
        if not isinstance(contract, dict) or contract.get("schema_status") != "not_observed":
            raise ValueError(f"{split} changed the missing-schema contract")
        if contract.get("reward_status") != "not_exported":
            raise ValueError(f"{split} contains a reward claim")
        privacy = row.get("privacy")
        if not isinstance(privacy, dict) or privacy.get("eligible_for_training") is not False:
            raise ValueError(f"{split} is not review-only privacy state")
        blob = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in blob for marker in ("<think>", "<analysis>", "chain_of_thought", "hidden_reasoning")):
            raise ValueError(f"{split} contains a reasoning marker")
    return {
        "rows": len(dataset),
        "columns": sorted(dataset.column_names),
        "unique_parents": len(parent_by_split[split]),
        "action_events": action_count,
        "observation_events": observation_count,
    }


def validate_pilot(pilot_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    pilot_dir = pilot_dir.resolve()
    manifest_path = pilot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PILOT_SCHEMA:
        raise ValueError("unexpected historical trajectory pilot schema")
    before_manifest_sha = _sha256_file(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("pilot manifest has no file descriptors")
    for filename, descriptor in files.items():
        path = pilot_dir / filename
        if not path.is_file() or _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"pilot file digest mismatch: {filename}")

    counts = manifest.get("counts") or {}
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(pilot_dir / "train.jsonl"),
            "validation": str(pilot_dir / "validation.jsonl"),
        },
    )
    loaded = {
        "train": _validate_split(dataset["train"], split="train", expected_rows=int(counts["train"])),
        "validation": _validate_split(
            dataset["validation"],
            split="validation",
            expected_rows=int(counts["validation"]),
        ),
    }
    train_parents = set()
    validation_parents = set()
    for split_name, details in loaded.items():
        # The parent sets are reconstructed from the loaded rows so this check
        # is independent of the exporter-side map.
        rows = dataset[split_name]
        target = train_parents if split_name == "train" else validation_parents
        target.update(row["lineage"]["parent_record_sha256"] for row in rows)
    if train_parents & validation_parents:
        raise ValueError("parent sessions cross train/validation splits")

    decision_rows = sum(1 for line in (pilot_dir / "decisions.jsonl").open() if line.strip())
    if decision_rows != int(counts["input_records"]):
        raise ValueError("decision row count does not match source input count")
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "pilot_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "format": "jsonl",
        "loaded": loaded,
        "decision_records": decision_rows,
        "parent_disjoint": True,
        "result": "passed",
    }
    if report_path is None:
        report_path = pilot_dir / "loader_validation.json"
    report_path = report_path.resolve()
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    report_sha = _sha256_file(report_path)
    manifest.setdefault("validation", {})["loader_validation"] = {
        "status": "passed",
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "report": report_path.name,
        "report_sha256": report_sha,
        "counts": loaded,
        "parent_disjoint": True,
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
