#!/usr/bin/env python3
"""Derive an immutable SFT release by dropping whole over-token examples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .dataset import (
        iter_release_rows,
        load_manifest,
        sha256_file,
        tokenize_final_assistant,
        validate_example,
    )
except ImportError:  # Direct script execution inside the training container.
    from dataset import (
        iter_release_rows,
        load_manifest,
        sha256_file,
        tokenize_final_assistant,
        validate_example,
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            raw = canonical_bytes(row) + b"\n"
            output.write(raw)
            digest.update(raw)
            count += 1
        output.flush()
        os.fsync(output.fileno())
    return {"records": count, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def descriptor(path: Path, records: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if records is not None:
        value["records"] = records
    return value


def iter_bound_jsonl(
    input_dir: Path, manifest: dict[str, Any], name: str
) -> Iterable[dict[str, Any]]:
    spec = manifest.get("files", {}).get(name)
    path = input_dir / name
    if not isinstance(spec, dict) or not path.is_file():
        raise ValueError(f"release file binding is missing: {name}")
    if path.stat().st_size != spec.get("bytes") or sha256_file(path) != spec.get("sha256"):
        raise ValueError(f"release file binding changed: {name}")
    count = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row: {path}:{line_number}")
            count += 1
            yield row
    if spec.get("records") != count:
        raise ValueError(f"release record count changed: {name}")


def copy_bound_extra(
    input_dir: Path,
    staging: Path,
    manifest: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    spec = manifest.get("files", {}).get(name)
    source = input_dir / name
    if not isinstance(spec, dict) or not source.is_file():
        raise ValueError(f"release file binding is missing: {name}")
    if source.stat().st_size != spec.get("bytes") or sha256_file(source) != spec.get("sha256"):
        raise ValueError(f"release file binding changed: {name}")
    shutil.copyfile(source, staging / name)
    copied = dict(spec)
    copied.pop("path", None)
    return copied


def filter_release(
    *,
    input_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    manifest = load_manifest(input_dir)
    if manifest.get("model", {}).get("max_sequence_tokens") != max_length:
        raise ValueError("runtime max length differs from the release contract")

    kept: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    exclusions: list[dict[str, Any]] = []
    lengths: list[int] = []
    targets: list[int] = []
    lane_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    input_rows = 0
    for split, line_number, row in iter_release_rows(input_dir, manifest):
        validate_example(row, expected_split=split)
        input_rows += 1
        if row["example_id"] in seen_ids:
            raise ValueError(f"duplicate trainer example: {row['example_id']}")
        seen_ids.add(row["example_id"])
        tokenized = tokenize_final_assistant(row, tokenizer, max_length=10**9)
        sequence_tokens = tokenized["sequence_tokens"]
        if sequence_tokens > max_length:
            exclusions.append(
                {
                    "example_id": row["example_id"],
                    "split": split,
                    "source_line": line_number,
                    "sequence_tokens": sequence_tokens,
                    "max_sequence_tokens": max_length,
                    "decision": "excluded_whole_example_no_truncation",
                }
            )
            continue
        kept[split].append(row)
        lengths.append(sequence_tokens)
        targets.append(tokenized["target_tokens"])
        lane_counts[f"{split}:{row['lane']}"] += 1
    if manifest.get("counts", {}).get("total") != input_rows:
        raise ValueError("release total does not reconcile with trainer rows")
    retained_ids = {
        row["example_id"] for rows in kept.values() for row in rows
    }
    if not retained_ids:
        raise ValueError("no trainer examples survived the exact tokenizer gate")

    lineage = [
        row
        for row in iter_bound_jsonl(input_dir, manifest, "lineage.jsonl")
        if row.get("example_id") in retained_ids
    ]
    if {row.get("example_id") for row in lineage} != retained_ids:
        raise ValueError("token filter broke example/lineage identity")
    split_by_id = {
        row["example_id"]: split for split, rows in kept.items() for row in rows
    }
    parent_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in lineage:
        parent = row.get("parent_id")
        if not isinstance(parent, str) or not parent:
            raise ValueError("lineage parent identity missing")
        parent_splits[parent].add(split_by_id[row["example_id"]])
    if any(len(splits) != 1 for splits in parent_splits.values()):
        raise ValueError("parent split overlap after tokenizer filtering")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        files = {
            "train.jsonl": write_jsonl(staging / "train.jsonl", kept["train"]),
            "validation.jsonl": write_jsonl(
                staging / "validation.jsonl", kept["validation"]
            ),
            "lineage.jsonl": write_jsonl(staging / "lineage.jsonl", lineage),
            "token_exclusions.jsonl": write_jsonl(
                staging / "token_exclusions.jsonl", exclusions
            ),
        }
        for name in sorted(manifest.get("files", {})):
            if name in {"train.jsonl", "validation.jsonl", "lineage.jsonl"}:
                continue
            files[name] = copy_bound_extra(input_dir, staging, manifest, name)

        report = {
            "schema_version": "ai-data-extraction/agent-sft-token-filter/v1",
            "status": "passed",
            "input_manifest_sha256": sha256_file(input_dir / "manifest.json"),
            "max_sequence_tokens": max_length,
            "truncation": False,
            "retained": len(retained_ids),
            "excluded": len(exclusions),
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "length": len(tokenizer),
                "chat_template_sha256": hashlib.sha256(
                    str(tokenizer.chat_template).encode("utf-8")
                ).hexdigest(),
            },
            "sequence_tokens": {
                "sum": sum(lengths),
                "min": min(lengths),
                "max": max(lengths),
            },
            "target_tokens": {
                "sum": sum(targets),
                "min": min(targets),
                "max": max(targets),
            },
        }
        (staging / "token_filter_report.json").write_bytes(
            json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        files["token_filter_report.json"] = descriptor(
            staging / "token_filter_report.json"
        )

        child = copy.deepcopy(manifest)
        child["status"] = (
            "exact_tokenizer_filtered_pending_independent_preflight_not_training_authorized"
        )
        child["training_authorized"] = False
        child["parent_release"] = {
            "path": str(input_dir),
            "manifest_sha256": report["input_manifest_sha256"],
        }
        child.setdefault("selection", {})["exact_tokenizer_filter"] = report
        child["counts"] = {
            "total": len(retained_ids),
            "train": len(kept["train"]),
            "validation": len(kept["validation"]),
            "unique_parents": len(parent_splits),
            "lanes": dict(sorted(lane_counts.items())),
            "excluded_over_token_limit": len(exclusions),
        }
        child.setdefault("quality", {})["tokenizer_filter"] = "passed"
        child["quality"]["tokenizer_preflight"] = "pending_independent_verifier"
        child["files"] = files
        (staging / "manifest.json").write_bytes(canonical_bytes(child) + b"\n")
        staging.replace(output_dir)
        child["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return child
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    result = filter_release(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
