#!/usr/bin/env python3
"""Validate a live Code Indexer tool batch with the pinned HF JSON loader."""

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

from build_code_indexer_tool_batch import BATCH_SCHEMA


LOADER_SCHEMA = "ai-data-extraction/trainer-loader-validation/v1"


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
    expected_columns = {"schema_version", "example_id", "split", "messages", "tools"}
    if len(dataset) != expected_rows:
        raise ValueError(f"{split} row count mismatch: {len(dataset)} != {expected_rows}")
    if set(dataset.column_names) != expected_columns:
        raise ValueError(
            f"{split} columns mismatch: {sorted(dataset.column_names)} != {sorted(expected_columns)}"
        )
    if set(dataset["split"]) != {split}:
        raise ValueError(f"{split} contains an unexpected split value")
    for messages, tools in zip(dataset["messages"], dataset["tools"]):
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{split} contains empty messages")
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"{split} contains empty tool schemas")
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"{split} contains an invalid message")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{split} contains non-string message content")
    return {"rows": len(dataset), "columns": dataset.column_names}


def validate_batch(batch_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    manifest_path = batch_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BATCH_SCHEMA:
        raise ValueError("unexpected Code Indexer batch schema")
    if manifest.get("training_authorized") is not False:
        raise ValueError("batch is training-authorized")
    before_manifest_sha = _sha256_file(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("batch manifest has no file descriptors")
    for filename, descriptor in files.items():
        path = batch_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"batch file digest mismatch: {filename}")
        if path.stat().st_size != descriptor.get("bytes"):
            raise ValueError(f"batch file byte count mismatch: {filename}")

    counts = manifest.get("counts") or {}
    loaded = load_dataset(
        "json",
        data_files={
            "train": str(batch_dir / "tool_sft_train.jsonl"),
            "validation": str(batch_dir / "tool_sft_validation.jsonl"),
        },
    )
    result = {
        "train": _validate_split(
            loaded["train"], split="train", expected_rows=int(counts["tool_sft_train"])
        ),
        "validation": _validate_split(
            loaded["validation"],
            split="validation",
            expected_rows=int(counts["tool_sft_validation"]),
        ),
    }
    example_ids = list(loaded["train"]["example_id"]) + list(loaded["validation"]["example_id"])
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("duplicate example IDs across splits")
    report = {
        "schema_version": LOADER_SCHEMA,
        "batch_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "format": "jsonl",
        "loaded": result,
        "result": "passed",
    }
    if report_path is None:
        report_path = batch_dir / "loader_validation.json"
    report_path = report_path.resolve()
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    report_sha = _sha256_file(report_path)
    manifest.setdefault("validation", {})["loader_validation"] = {
        "status": "passed",
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "report": report_path.name,
        "report_sha256": report_sha,
        "counts": result,
    }
    _atomic_write(manifest_path, _canonical_bytes(manifest) + b"\n")
    report["report_sha256"] = report_sha
    report["batch_manifest_sha256_after"] = _sha256_file(manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_batch(args.batch_dir, report_path=args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate_batch"]
