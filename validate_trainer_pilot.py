#!/usr/bin/env python3
"""Validate the unified pilot with the Hugging Face JSON dataset loader."""

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


PILOT_SCHEMA = "ai-data-extraction/unified-trainer-pilot/v1"


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


def _validate_split(dataset: Any, *, split: str, expected_rows: int, has_tools: bool) -> dict[str, Any]:
    expected_columns = {"schema_version", "example_id", "split", "messages"}
    if has_tools:
        expected_columns.add("tools")
    if len(dataset) != expected_rows:
        raise ValueError(f"{split} row count mismatch: {len(dataset)} != {expected_rows}")
    if set(dataset.column_names) != expected_columns:
        raise ValueError(
            f"{split} columns mismatch: {sorted(dataset.column_names)} != {sorted(expected_columns)}"
        )
    split_values = set(dataset["split"])
    if split_values != {split}:
        raise ValueError(f"{split} contains split values {sorted(split_values)}")
    for messages in dataset["messages"]:
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{split} contains an empty messages field")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"{split} contains an invalid message")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{split} contains non-string message content")
    if has_tools and any(not isinstance(tools, list) or not tools for tools in dataset["tools"]):
        raise ValueError(f"{split} contains an empty tool schema field")
    return {"rows": len(dataset), "columns": dataset.column_names}


def validate_pilot(pilot_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    pilot_dir = pilot_dir.resolve()
    manifest_path = pilot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PILOT_SCHEMA:
        raise ValueError("unexpected unified pilot schema")
    before_manifest_sha = _sha256_file(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("pilot manifest has no file descriptors")
    for filename, descriptor in files.items():
        path = pilot_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"pilot file digest mismatch: {filename}")

    counts = manifest.get("counts") or {}
    sft = load_dataset(
        "json",
        data_files={
            "train": str(pilot_dir / "train.jsonl"),
            "validation": str(pilot_dir / "validation.jsonl"),
        },
    )
    tool = load_dataset("json", data_files={"train": str(pilot_dir / "tool_train.jsonl")})
    loaded = {
        "sft_train": _validate_split(
            sft["train"],
            split="train",
            expected_rows=int(counts["sft_train"]),
            has_tools=False,
        ),
        "sft_validation": _validate_split(
            sft["validation"],
            split="validation",
            expected_rows=int(counts["sft_validation"]),
            has_tools=False,
        ),
        "tool_train": _validate_split(
            tool["train"],
            split="train",
            expected_rows=int(counts["tool_train"]),
            has_tools=True,
        ),
    }
    report = {
        "schema_version": "ai-data-extraction/trainer-loader-validation/v1",
        "pilot_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "format": "jsonl",
        "loaded": loaded,
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
