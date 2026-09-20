#!/usr/bin/env python3
"""Join bounded live Code Indexer captures into one review-only tool release.

The capture adapter owns execution truth.  This module only verifies the
capture manifests and joins their already-projected rows.  It does not infer
schemas, score assistant text, or export rewards.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_training_data import no_reasoning_content
from capture_code_indexer_episode import CAPTURE_SCHEMA, TOOL_NAME
from quality_rules import scan_record


BATCH_SCHEMA = "ai-data-extraction/runtime-tool-capture-batch/v1"
DECISION_SCHEMA = "ai-data-extraction/runtime-tool-capture-decision/v1"


class ToolBatchError(ValueError):
    """Raised when a live capture cannot be joined without losing evidence."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolBatchError(f"could not load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ToolBatchError(f"JSON object required: {path}")
    return value


def _load_one_jsonl(path: Path, *, label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw in enumerate(source, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ToolBatchError(f"invalid {label} JSONL at line {line_number}") from exc
                if not isinstance(value, dict):
                    raise ToolBatchError(f"{label} line {line_number} is not an object")
                rows.append(value)
    except OSError as exc:
        raise ToolBatchError(f"could not read {label}: {path}") from exc
    if len(rows) != 1:
        raise ToolBatchError(f"{label} must contain exactly one row: {path}")
    return rows[0]


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_capture_files(capture_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ToolBatchError(f"capture manifest has no file descriptors: {capture_dir}")
    for relative, descriptor in files.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ToolBatchError(f"capture file path is not relative: {relative!r}")
        path = (capture_dir / relative).resolve()
        try:
            path.relative_to(capture_dir.resolve())
        except ValueError as exc:
            raise ToolBatchError(f"capture file escapes capture directory: {relative}") from exc
        if not path.is_file() or not isinstance(descriptor, dict):
            raise ToolBatchError(f"capture file is missing: {relative}")
        actual = _file_descriptor(path)
        if actual != {
            "bytes": descriptor.get("bytes"),
            "sha256": descriptor.get("sha256"),
        }:
            raise ToolBatchError(f"capture file digest mismatch: {capture_dir}/{relative}")


def _query_and_limit(manifest: dict[str, Any]) -> tuple[str, int]:
    query = manifest.get("query")
    limit = manifest.get("limit")
    source = manifest.get("source")
    command = source.get("command") if isinstance(source, dict) else None
    if not isinstance(query, str) or not query.strip():
        if isinstance(command, list) and len(command) >= 3 and isinstance(command[2], str):
            query = command[2]
        else:
            raise ToolBatchError("capture manifest has no query")
    if not isinstance(limit, int):
        if isinstance(command, list) and len(command) >= 7 and isinstance(command[6], int):
            limit = command[6]
        else:
            raise ToolBatchError("capture manifest has no integer limit")
    if not 1 <= limit <= 200:
        raise ToolBatchError("capture limit is outside the callable schema")
    return query, limit


def _verify_projection(
    capture_dir: Path,
    capture_manifest: dict[str, Any],
) -> dict[str, Any]:
    projection_dir = capture_dir / "projection"
    projection_manifest = _load_json(projection_dir / "manifest.json")
    if projection_manifest.get("status") != "review_only":
        raise ToolBatchError("projection is not review-only")
    if projection_manifest.get("training_authorized") is not False:
        raise ToolBatchError("projection is training-authorized")
    if projection_manifest.get("trainer_loadable") is not True:
        raise ToolBatchError("projection is not trainer-loadable")
    validation = projection_manifest.get("validation")
    if not isinstance(validation, dict):
        raise ToolBatchError("projection has no validation record")
    for key in (
        "trace_identity",
        "registry_join",
        "skill_gate",
        "call_observation_join",
        "verification",
        "privacy_reasoning_firewall",
    ):
        if validation.get(key) != "passed":
            raise ToolBatchError(f"projection validation did not pass: {key}")
    if validation.get("reward") != "not_present":
        raise ToolBatchError("projection contains a reward status")

    projection_files = projection_manifest.get("files")
    if not isinstance(projection_files, dict):
        raise ToolBatchError("projection has no file descriptors")
    for relative, descriptor in projection_files.items():
        path = projection_dir / relative
        if not path.is_file() or not isinstance(descriptor, dict):
            raise ToolBatchError(f"projection file is missing: {relative}")
        actual = _file_descriptor(path)
        if actual != {
            "bytes": descriptor.get("bytes"),
            "sha256": descriptor.get("sha256"),
        } or descriptor.get("records") != 1:
            raise ToolBatchError(f"projection file digest mismatch: {relative}")

    row = _load_one_jsonl(projection_dir / "tool_sft.jsonl", label="tool SFT")
    lineage = _load_one_jsonl(projection_dir / "lineage.jsonl", label="lineage")
    if row.get("example_id") != lineage.get("example_id"):
        raise ToolBatchError("projection row and lineage example IDs differ")
    parent = lineage.get("parent_record_sha256")
    if not isinstance(parent, str) or not parent:
        raise ToolBatchError("projection lineage has no parent identity")
    if not no_reasoning_content(row):
        raise ToolBatchError("projection row contains reasoning content")
    findings = scan_record(row)
    if findings.has_hard_privacy_issue or findings.has_marker:
        raise ToolBatchError("projection row fails privacy/marker firewall")

    query, limit = _query_and_limit(capture_manifest)
    source = capture_manifest.get("source")
    if not isinstance(source, dict):
        raise ToolBatchError("capture manifest has no source block")
    command = source.get("command")
    expected_command = ["code-indexer", "search", query, "--root", ".", "--limit", limit]
    if command != expected_command:
        raise ToolBatchError("capture command does not match query/limit")
    definition = capture_manifest.get("definition")
    if not isinstance(definition, dict) or definition.get("symbol") != query:
        raise ToolBatchError("capture definition is not an exact query match")
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ToolBatchError("projection row has no messages")
    calls = [
        call
        for message in messages
        if isinstance(message, dict)
        for call in (message.get("tool_calls") or [])
        if isinstance(call, dict)
    ]
    if len(calls) != 1:
        raise ToolBatchError("projection row must contain one tool call")
    function = calls[0].get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
        raise ToolBatchError("projection row has the wrong tool name")
    if arguments != {"limit": limit, "query": query, "root": "."}:
        raise ToolBatchError("projection arguments do not match capture command")

    return {
        "row": row,
        "lineage": lineage,
        "parent": parent,
        "query": query,
        "limit": limit,
        "capture_manifest": capture_manifest,
        "capture_manifest_sha256": _sha256_file(capture_dir / "capture_manifest.json"),
        "projection_manifest_sha256": _sha256_file(projection_dir / "manifest.json"),
    }


def _load_capture(capture_dir: Path) -> dict[str, Any]:
    capture_dir = capture_dir.resolve()
    manifest_path = capture_dir / "capture_manifest.json"
    if not capture_dir.is_dir() or not manifest_path.is_file():
        raise ToolBatchError(f"capture directory is incomplete: {capture_dir}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != CAPTURE_SCHEMA:
        raise ToolBatchError(f"unexpected capture schema: {capture_dir}")
    if manifest.get("status") != "review_only":
        raise ToolBatchError(f"capture is not review-only: {capture_dir}")
    if manifest.get("training_authorized") is not False:
        raise ToolBatchError(f"capture is training-authorized: {capture_dir}")
    counts = manifest.get("counts")
    if counts != {"episodes": 1, "projected_tool_sft": 1}:
        raise ToolBatchError(f"capture counts are not one verified episode: {capture_dir}")
    _verify_capture_files(capture_dir, manifest)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ToolBatchError("capture manifest has no source block")
    worktree_identity = source.get("worktree_identity")
    runtime_status_identity = source.get("runtime_status_identity")
    if not isinstance(worktree_identity, dict) or not isinstance(runtime_status_identity, dict):
        raise ToolBatchError("capture lacks dirty-worktree or runtime-status identity")
    if worktree_identity.get("head") != source.get("project_revision"):
        raise ToolBatchError("worktree identity does not match project revision")
    status_descriptor = manifest.get("files", {}).get("runtime_status.json")
    if not isinstance(status_descriptor, dict):
        raise ToolBatchError("capture has no persisted runtime status")
    if status_descriptor.get("sha256") != runtime_status_identity.get("sanitized_status_sha256"):
        raise ToolBatchError("runtime status file hash does not match its identity")
    if source.get("runtime_status_raw_sha256") != runtime_status_identity.get("raw_status_sha256"):
        raise ToolBatchError("runtime raw status hash does not match its identity")
    verified = _verify_projection(capture_dir, manifest)
    verified["capture_dir_label"] = capture_dir.name
    verified["worktree_identity"] = worktree_identity
    verified["runtime_status_identity"] = runtime_status_identity
    return verified


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    with path.open("wb") as destination:
        for row in rows:
            raw = _canonical_bytes(row) + b"\n"
            destination.write(raw)
            digest.update(raw)
            count += 1
            byte_count += len(raw)
    return {"records": count, "bytes": byte_count, "sha256": digest.hexdigest()}


def _copy_with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result["split"] = split
    return result


def build_code_indexer_tool_batch(
    capture_dirs: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Verify and join captures without changing their source directories."""

    capture_paths = [Path(path).resolve() for path in capture_dirs]
    if not capture_paths:
        raise ToolBatchError("at least one capture directory is required")
    if len(set(capture_paths)) != len(capture_paths):
        raise ToolBatchError("capture directories must be unique")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite batch directory: {output_dir}")

    entries = [_load_capture(path) for path in capture_paths]
    parents = [entry["parent"] for entry in entries]
    if len(set(parents)) != len(parents):
        raise ToolBatchError("capture parents are not unique")
    ordered = sorted(entries, key=lambda entry: entry["parent"])
    validation_count = max(1, len(ordered) // 5) if len(ordered) > 1 else 0
    validation_parents = {entry["parent"] for entry in ordered[:validation_count]}

    baseline = ordered[0]["capture_manifest"]
    baseline_source = baseline.get("source")
    baseline_contract = baseline.get("contract")
    if not isinstance(baseline_source, dict) or not isinstance(baseline_contract, dict):
        raise ToolBatchError("capture manifest lacks source or contract")
    shared_fields = (
        ("source", "project_revision"),
        ("source", "binary_sha256"),
        ("contract", "registry_revision"),
        ("contract", "skill_revision"),
        ("contract", "verifier_revision"),
    )
    for entry in ordered[1:]:
        manifest = entry["capture_manifest"]
        for block_name, field in shared_fields:
            expected_block = baseline.get(block_name)
            actual_block = manifest.get(block_name)
            if not isinstance(expected_block, dict) or not isinstance(actual_block, dict):
                raise ToolBatchError(f"capture lacks shared identity block: {block_name}")
            if actual_block.get(field) != expected_block.get(field):
                raise ToolBatchError(f"capture identity drifted: {block_name}.{field}")
        if entry["worktree_identity"] != ordered[0]["worktree_identity"]:
            raise ToolBatchError("capture identity drifted: worktree")
        baseline_status = ordered[0]["runtime_status_identity"]
        current_status = entry["runtime_status_identity"]
        stable_status_fields = (
            "capability_stage",
            "chunks",
            "edges",
            "embedded_chunks",
            "symbols",
            "semantic",
            "published_complete",
            "published_revision",
            "published_fence",
            "worktree_id",
        )
        if any(current_status.get(field) != baseline_status.get(field) for field in stable_status_fields):
            raise ToolBatchError("capture identity drifted: runtime status")

    tool_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    lineage_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    capture_summaries: list[dict[str, Any]] = []
    for entry in ordered:
        parent = entry["parent"]
        split = "validation" if parent in validation_parents else "train"
        row = _copy_with_split(entry["row"], split)
        lineage = copy.deepcopy(entry["lineage"])
        source_projection_split = lineage.get("split")
        lineage.update(
            {
                "batch_schema_version": BATCH_SCHEMA,
                "source_projection_split": source_projection_split,
                "split": split,
                "batch_split": split,
                "capture_manifest_sha256": entry["capture_manifest_sha256"],
                "source_projection_manifest_sha256": entry["projection_manifest_sha256"],
            }
        )
        decision_rows.append(
            {
                "schema_version": DECISION_SCHEMA,
                "decision": "review_only",
                "reason": "verified_runtime_capture_not_authorized_for_training",
                "promotable": False,
                "example_id": row["example_id"],
                "parent_record_sha256": parent,
                "split": split,
                "query": entry["query"],
                "tool_name": TOOL_NAME,
                "quality_tier": "harness_verified_tool_sft",
                "verifier_result": "pass",
                "reward_status": "not_exported",
                "capture_manifest_sha256": entry["capture_manifest_sha256"],
            }
        )
        tool_rows[split].append(row)
        lineage_rows.append(lineage)
        capture_manifest = entry["capture_manifest"]
        source = capture_manifest["source"]
        contract = capture_manifest["contract"]
        capture_summaries.append(
            {
                "capture_manifest_sha256": entry["capture_manifest_sha256"],
                "capture_label": entry["capture_dir_label"],
                "query": entry["query"],
                "limit": entry["limit"],
                "definition": capture_manifest["definition"],
                "project_revision": source["project_revision"],
                "binary_sha256": source["binary_sha256"],
                "registry_revision": contract["registry_revision"],
                "worktree_identity": entry["worktree_identity"],
                "runtime_status_identity": entry["runtime_status_identity"],
            }
        )

    batch_identity = {
        "schema_version": BATCH_SCHEMA,
        "capture_manifests": [summary["capture_manifest_sha256"] for summary in capture_summaries],
        "parents": [entry["parent"] for entry in ordered],
    }
    batch_id = "sha256:" + _sha256_bytes(_canonical_bytes(batch_identity))
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        files = {
            "tool_sft_train.jsonl": _write_jsonl(staging / "tool_sft_train.jsonl", tool_rows["train"]),
            "tool_sft_validation.jsonl": _write_jsonl(
                staging / "tool_sft_validation.jsonl", tool_rows["validation"]
            ),
            "lineage.jsonl": _write_jsonl(staging / "lineage.jsonl", lineage_rows),
            "decisions.jsonl": _write_jsonl(staging / "decisions.jsonl", decision_rows),
        }
        manifest = {
            "schema_version": BATCH_SCHEMA,
            "batch_id": batch_id,
            "status": "review_only",
            "training_authorized": False,
            "trainer_loadable": True,
            "format": "messages_jsonl_with_tools",
            "source": {
                "capture_count": len(entries),
                "captures": capture_summaries,
                "project_revision": baseline_source["project_revision"],
                "binary_sha256": baseline_source["binary_sha256"],
            },
            "contract": {
                "tool_name": TOOL_NAME,
                "registry_revision": baseline_contract["registry_revision"],
                "skill_revision": baseline_contract["skill_revision"],
                "verifier_revision": baseline_contract["verifier_revision"],
                "schema_binding": "exact_runtime_registry",
                "verifier": "passed",
                "rewards": "not_exported",
            },
            "quality": {
                "tier": "harness_verified_tool_sft",
                "limitations": ["training_authorized_false", "reward_not_exported"],
            },
            "split_policy": {
                "name": "global_parent_sha256_sorted_v1",
                "key": "parent_record_sha256",
                "validation_count": validation_count,
                "parent_disjoint": True,
            },
            "counts": {
                "captures": len(entries),
                "tool_sft": len(ordered),
                "tool_sft_train": len(tool_rows["train"]),
                "tool_sft_validation": len(tool_rows["validation"]),
                "lineage": len(lineage_rows),
                "decisions": len(decision_rows),
                "rewards": 0,
            },
            "decision_counts": dict(sorted(Counter(row["decision"] for row in decision_rows).items())),
            "validation": {
                "capture_manifest_files": "passed",
                "projection_joins": "passed",
                "exact_registry_binding": "passed",
                "parent_disjoint_split": "passed",
                "privacy_reasoning_firewall": "passed",
                "reward": "not_present",
                "training_authorization": "false",
            },
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = _sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("capture_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_code_indexer_tool_batch(args.capture_dirs, args.output_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_SCHEMA",
    "DECISION_SCHEMA",
    "ToolBatchError",
    "build_code_indexer_tool_batch",
]
