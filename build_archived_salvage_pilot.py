#!/usr/bin/env python3
"""Compose the verified archived SFT and tool-review pilots.

This is a trainer-export boundary adapter. It does not read the source corpus,
rewrite either input pilot, infer tool schemas, or create rewards. The output
contains trainer-shaped SFT/tool-SFT partitions and a separate event-rich
historical tool-review partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


ARCHIVED_SALVAGE_SCHEMA = "ai-data-extraction/archived-salvage-pilot/v1"
LINEAGE_SCHEMA = "ai-data-extraction/archived-salvage-lineage/v1"
DECISION_SCHEMA = "ai-data-extraction/archived-salvage-decision/v1"
UNIFIED_SCHEMA = "ai-data-extraction/unified-trainer-pilot/v1"
HISTORICAL_SCHEMA = "ai-data-extraction/historical-tool-trajectory-pilot/v1"
TRAINER_EXAMPLE_SCHEMA = "ai-data-extraction/trainer-example/v1"


class SalvagePilotError(ValueError):
    """Raised when an input pilot cannot be safely composed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SalvagePilotError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_file(root: Path, relative: str, field: str) -> Path:
    candidate = (root / _required_string(relative, field)).resolve()
    if root.resolve() not in candidate.parents:
        raise SalvagePilotError(f"{field} escapes input directory")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None
        self.digest = hashlib.sha256()
        self.records = 0
        self.bytes = 0

    def __enter__(self) -> "JsonlWriter":
        self.handle = self.path.open("wb")
        return self

    def write(self, value: Any) -> None:
        if self.handle is None:
            raise RuntimeError("writer is not open")
        raw = canonical_json(value) + b"\n"
        self.handle.write(raw)
        self.digest.update(raw)
        self.records += 1
        self.bytes += len(raw)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is not None:
            self.handle.close()

    def descriptor(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }


def _load_manifest(root: Path, expected_schema: str) -> tuple[dict[str, Any], str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != expected_schema:
        raise SalvagePilotError(f"unexpected input schema in {root}")
    if manifest.get("training_authorized") is not False:
        raise SalvagePilotError(f"input is training-authorized or missing authorization state: {root}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SalvagePilotError(f"input has no file descriptors: {root}")
    for filename, descriptor in files.items():
        if not isinstance(descriptor, dict):
            raise SalvagePilotError(f"invalid file descriptor: {root}/{filename}")
        path = _safe_file(root, filename, f"input file {filename}")
        if sha256_file(path) != descriptor.get("sha256"):
            raise SalvagePilotError(f"input file digest mismatch: {path}")
        expected_records = descriptor.get("records")
        if not isinstance(expected_records, int):
            raise SalvagePilotError(f"input record count missing: {path}")
    return manifest, manifest_sha


def _read_jsonl(path: Path) -> Iterable[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SalvagePilotError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise SalvagePilotError(f"row is not an object at {path}:{line_number}")
            yield line_number, raw, value


def _count_non_empty_lines(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for line in source if line.strip())


def _parent_from_lineage(lineage: dict[str, Any], field: str) -> str:
    return _required_string(lineage.get("parent_record_sha256"), field)


def _assign_parent_split(parent: str, parent_splits: dict[str, str]) -> str:
    existing = parent_splits.get(parent)
    if existing is not None:
        return existing
    bucket = int(hashlib.sha256(parent.encode("utf-8")).hexdigest()[:8], 16) % 100
    split = "validation" if bucket >= 90 else "train"
    parent_splits[parent] = split
    return split


def _record_manifest_input(root: Path, manifest: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    files = manifest.get("files")
    assert isinstance(files, dict)
    return {
        "path": str(root.resolve()),
        "manifest_sha256": manifest_sha,
        "schema_version": manifest.get("schema_version"),
        "training_authorized": manifest.get("training_authorized"),
        "files": {
            filename: {
                "records": descriptor.get("records"),
                "bytes": descriptor.get("bytes"),
                "sha256": descriptor.get("sha256"),
            }
            for filename, descriptor in sorted(files.items())
            if isinstance(descriptor, dict)
        },
        "counts": manifest.get("counts", {}),
        "quality": manifest.get("quality", {}),
        "source": manifest.get("source", {}),
    }


def _write_copy(
    *,
    source_path: Path,
    destination: JsonlWriter,
    expected_schema: str,
    parent_splits: dict[str, str],
    parent_seen: set[str],
    example_ids: set[str],
    lane: str,
    lineage_by_id: dict[str, dict[str, Any]],
    provider_counts: Counter[str],
    tier_counts: Counter[str],
    split_counts: Counter[str],
) -> None:
    for source_line, _raw, row in _read_jsonl(source_path):
        if row.get("schema_version") != expected_schema:
            raise SalvagePilotError(f"unexpected row schema at {source_path}:{source_line}")
        example_id = _required_string(row.get("example_id"), "row.example_id")
        if example_id in example_ids:
            raise SalvagePilotError(f"duplicate example_id across composed lanes: {example_id}")
        example_ids.add(example_id)
        split = _required_string(row.get("split"), "row.split")
        if split not in {"train", "validation", "test"}:
            raise SalvagePilotError(f"invalid row split at {source_path}:{source_line}")
        destination.write(row)
        split_counts[f"{lane}:{split}"] += 1
        lineage = lineage_by_id.get(example_id)
        if not isinstance(lineage, dict):
            raise SalvagePilotError(f"missing lineage for copied row: {example_id}")
        parent = _parent_from_lineage(lineage, f"lineage.parent_record_sha256:{example_id}")
        if parent in parent_seen:
            raise SalvagePilotError(f"duplicate parent in selected trainer lanes: {parent}")
        parent_seen.add(parent)
        prior_split = parent_splits.get(parent)
        if prior_split is not None and prior_split != split:
            raise SalvagePilotError(f"trainer row split conflicts with parent assignment: {parent}")
        parent_splits[parent] = split
        provider_counts[str(lineage.get("provider") or "unknown")] += 1
        tier_counts[str(lineage.get("quality_tier") or "unknown")] += 1


def _unified_lineage(
    *,
    unified_root: Path,
    unified_manifest_sha: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    result: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    path = unified_root / "lineage.jsonl"
    for _line, _raw, lineage in _read_jsonl(path):
        example_id = _required_string(lineage.get("example_id"), "unified lineage.example_id")
        if example_id in result:
            raise SalvagePilotError(f"duplicate unified lineage example_id: {example_id}")
        normalized = dict(lineage)
        normalized["schema_version"] = LINEAGE_SCHEMA
        normalized["source_lineage_schema_version"] = lineage.get("schema_version")
        normalized["source_pilot_manifest_sha256"] = unified_manifest_sha
        normalized["package_dataset"] = str(lineage.get("dataset") or "unknown")
        result[example_id] = normalized
        rows.append(normalized)
    return result, rows


def _historical_lineage(
    *,
    row: dict[str, Any],
    raw: bytes,
    source_line: int,
    source_split: str,
    historical_manifest_sha: str,
) -> dict[str, Any]:
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    quality = row.get("session_quality") if isinstance(row.get("session_quality"), dict) else {}
    contract = row.get("tool_contract") if isinstance(row.get("tool_contract"), dict) else {}
    return {
        "schema_version": LINEAGE_SCHEMA,
        "example_id": _required_string(row.get("example_id"), "historical row.example_id"),
        "package_dataset": "tool_trajectory_review",
        "split": source_split,
        "source_pilot_manifest_sha256": historical_manifest_sha,
        "source_pilot_row_sha256": sha256_bytes(raw),
        "source_pilot_line": source_line,
        "source_pilot_split": source_split,
        "source_dataset": lineage.get("source_dataset"),
        "source_example_id": lineage.get("source_example_id"),
        "parent_record_sha256": _required_string(
            lineage.get("parent_record_sha256"), "historical lineage.parent_record_sha256"
        ),
        "source_record_sha256": lineage.get("source_record_sha256"),
        "segment_record_sha256": lineage.get("segment_record_sha256"),
        "source_file_name": lineage.get("source_file_name"),
        "source_file_sha256": lineage.get("source_file_sha256"),
        "source_line_in_original": lineage.get("source_line"),
        "source_message_range_json": lineage.get("source_message_range_json"),
        "chunk_index": lineage.get("chunk_index"),
        "chunk_count": lineage.get("chunk_count"),
        "continuation_status": lineage.get("continuation_status"),
        "provider": row.get("provider"),
        "agent": row.get("agent"),
        "model_tier": row.get("model_tier"),
        "quality_tier": row.get("quality_tier"),
        "quality_reason": row.get("quality_reason", []),
        "session_quality_id": quality.get("id"),
        "session_quality_scope": quality.get("scope"),
        "tool_schema_status": contract.get("schema_status"),
        "verification_status": contract.get("verification_status"),
        "reward_status": contract.get("reward_status"),
        "limitations": [
            *[str(item) for item in row.get("quality_reason", []) if isinstance(item, str)],
            "tool_schema_not_observed",
            "verifier_not_observed",
            "training_authorized_false",
        ],
    }


def _source_decision(
    *,
    row: dict[str, Any],
    lane: str,
    manifest_sha: str,
    package_split: str | None,
) -> dict[str, Any]:
    decision = dict(row)
    decision["schema_version"] = DECISION_SCHEMA
    decision["source_lane"] = lane
    decision["source_manifest_sha256"] = manifest_sha
    decision["package_split"] = package_split
    decision["training_authorized"] = False
    return decision


def build_archived_salvage_pilot(
    *,
    unified_dir: Path,
    historical_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    unified_dir = unified_dir.resolve()
    historical_dir = historical_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    unified_manifest, unified_manifest_sha = _load_manifest(unified_dir, UNIFIED_SCHEMA)
    historical_manifest, historical_manifest_sha = _load_manifest(historical_dir, HISTORICAL_SCHEMA)
    unified_lineage, lineage_rows = _unified_lineage(
        unified_root=unified_dir,
        unified_manifest_sha=unified_manifest_sha,
    )
    parent_splits: dict[str, str] = {}
    for lineage in lineage_rows:
        parent = _parent_from_lineage(lineage, "unified lineage.parent_record_sha256")
        split = _required_string(lineage.get("split"), "unified lineage.split")
        prior = parent_splits.setdefault(parent, split)
        if prior != split:
            raise SalvagePilotError(f"unified input parent split conflict: {parent}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    example_ids: set[str] = set()
    selected_parent_seen: set[str] = set()
    provider_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tool_review_action_events = 0
    tool_review_observation_events = 0
    historical_lineage_by_id: dict[str, dict[str, Any]] = {}
    historical_lineage_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    try:
        # Copy trainer-shaped inputs through one allowlisted boundary.
        for source_name, output_name, expected_schema, lane in (
            ("train.jsonl", "sft_train.jsonl", TRAINER_EXAMPLE_SCHEMA, "sft"),
            ("validation.jsonl", "sft_validation.jsonl", TRAINER_EXAMPLE_SCHEMA, "sft"),
            ("tool_train.jsonl", "tool_sft_train.jsonl", TRAINER_EXAMPLE_SCHEMA, "tool_sft"),
            ("tool_validation.jsonl", "tool_sft_validation.jsonl", TRAINER_EXAMPLE_SCHEMA, "tool_sft"),
        ):
            lineage_by_id = unified_lineage
            with JsonlWriter(staging / output_name) as destination:
                _write_copy(
                    source_path=_safe_file(unified_dir, source_name, f"unified.{source_name}"),
                    destination=destination,
                    expected_schema=expected_schema,
                    parent_splits=parent_splits,
                    parent_seen=selected_parent_seen,
                    example_ids=example_ids,
                    lane=lane,
                    lineage_by_id=lineage_by_id,
                    provider_counts=provider_counts,
                    tier_counts=tier_counts,
                    split_counts=split_counts,
                )

        for _line, _raw, decision in _read_jsonl(unified_dir / "decisions.jsonl"):
            example_id = decision.get("example_id")
            package_split = None
            if isinstance(example_id, str) and example_id in unified_lineage:
                package_split = str(unified_lineage[example_id].get("split") or "")
            decisions.append(
                _source_decision(
                    row=decision,
                    lane="unified_trainer_pilot",
                    manifest_sha=unified_manifest_sha,
                    package_split=package_split,
                )
            )

        with (
            JsonlWriter(staging / "tool_trajectory_train.jsonl") as tool_train,
            JsonlWriter(staging / "tool_trajectory_validation.jsonl") as tool_validation,
        ):
            for source_name in ("train.jsonl", "validation.jsonl"):
                source_path = _safe_file(historical_dir, source_name, f"historical.{source_name}")
                for source_line, raw, row in _read_jsonl(source_path):
                    if row.get("schema_version") != HISTORICAL_SCHEMA:
                        raise SalvagePilotError(f"unexpected historical row schema: {source_path}:{source_line}")
                    example_id = _required_string(row.get("example_id"), "historical row.example_id")
                    if example_id in example_ids:
                        raise SalvagePilotError(f"duplicate example_id across composed lanes: {example_id}")
                    example_ids.add(example_id)
                    source_split = _required_string(row.get("split"), "historical row.split")
                    lineage = _historical_lineage(
                        row=row,
                        raw=raw,
                        source_line=source_line,
                        source_split=source_split,
                        historical_manifest_sha=historical_manifest_sha,
                    )
                    historical_lineage_by_id[example_id] = lineage
                    historical_lineage_rows.append(lineage)
                    parent = _parent_from_lineage(lineage, "historical lineage.parent_record_sha256")
                    package_split = _assign_parent_split(parent, parent_splits)
                    lineage["split"] = package_split
                    output_row = dict(row)
                    output_row["split"] = package_split
                    (tool_train if package_split == "train" else tool_validation).write(output_row)
                    split_counts[f"tool_trajectory_review:{package_split}"] += 1
                    provider_counts[str(row.get("provider") or "unknown")] += 1
                    tier_counts[str(row.get("quality_tier") or "unknown")] += 1
                    selected_parent_seen.add(parent)
                    summary = row.get("events_summary")
                    if isinstance(summary, dict):
                        tool_review_action_events += int(summary.get("action_count") or 0)
                        tool_review_observation_events += int(summary.get("observation_count") or 0)
                    decisions.append(
                        _source_decision(
                            row={
                                "source_example_id": example_id,
                                "example_id": example_id,
                                "parent_record_sha256": parent,
                                "source_pilot_line": source_line,
                                "source_pilot_split": source_split,
                                "decision": "selected",
                                "reasons": ["selected_archived_tool_review"],
                            },
                            lane="historical_tool_trajectory_pilot",
                            manifest_sha=historical_manifest_sha,
                            package_split=package_split,
                        )
                    )

        lineage_rows.extend(historical_lineage_rows)
        with JsonlWriter(staging / "lineage.jsonl") as lineage_file:
            for lineage in lineage_rows:
                lineage_file.write(lineage)
        with JsonlWriter(staging / "decisions.jsonl") as decisions_file:
            for decision in decisions:
                decisions_file.write(decision)

        files: dict[str, dict[str, Any]] = {}
        for filename in (
            "sft_train.jsonl",
            "sft_validation.jsonl",
            "tool_sft_train.jsonl",
            "tool_sft_validation.jsonl",
            "tool_trajectory_train.jsonl",
            "tool_trajectory_validation.jsonl",
            "lineage.jsonl",
            "decisions.jsonl",
        ):
            path = staging / filename
            files[filename] = {
                "records": _count_non_empty_lines(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": ARCHIVED_SALVAGE_SCHEMA,
            "status": "review_only",
            "trainer_loadable": True,
            "training_authorized": False,
            "format": "partitioned_jsonl",
            "inputs": {
                "unified_trainer_pilot": _record_manifest_input(
                    unified_dir, unified_manifest, unified_manifest_sha
                ),
                "historical_tool_trajectory_pilot": _record_manifest_input(
                    historical_dir, historical_manifest, historical_manifest_sha
                ),
            },
            "partitions": {
                "sft": "trainer-shaped dialogue; silver/gold lineage retained",
                "tool_sft": "schema-bound tool SFT only",
                "tool_trajectory_review": "event-rich historical review; schema/verifier absent rows retained but not trainer-authorized",
            },
            "counts": {
                "sft_train": split_counts["sft:train"],
                "sft_validation": split_counts["sft:validation"],
                "tool_sft_train": split_counts["tool_sft:train"],
                "tool_sft_validation": split_counts["tool_sft:validation"],
                "tool_trajectory_train": split_counts["tool_trajectory_review:train"],
                "tool_trajectory_validation": split_counts["tool_trajectory_review:validation"],
                "lineage_records": len(lineage_rows),
                "decision_records": len(decisions),
                "unique_parent_sessions": len(parent_splits),
                "selected_parent_sessions": len(selected_parent_seen),
                "tool_review_action_events": tool_review_action_events,
                "tool_review_observation_events": tool_review_observation_events,
            },
            "quality": {
                "provider_counts": dict(sorted(provider_counts.items())),
                "quality_tier_counts": dict(sorted(tier_counts.items())),
                "training_tier_policy": "preserve source tier; quality gate per row/session",
                "limitations": [
                    "silver_rows_outcome_unknown",
                    "silver_rows_not_human_adjudicated",
                    "historical_tool_schema_not_observed",
                    "historical_tool_verifier_not_observed",
                    "privacy_approval_not_granted",
                    "rewards_not_exported",
                ],
            },
            "rl": {
                "records": 0,
                "status": "not_exported",
                "reason": "no executable verifier-backed rewards in inputs",
            },
            "split_policy": {
                "name": "global parent hash assignment",
                "parent_disjoint_across_all_partitions": True,
                "validation_bucket": 10,
                "validation_threshold": 90,
                "historical_source_split_preserved": False,
            },
            "validation": {
                "input_artifact_hashes": "passed",
                "trainer_projection": "passed_for_sft_and_schema_bound_tool_sft",
                "tool_review_projection": "passed_structural_copy",
                "parent_disjoint": "pending_loader_validation",
                "reasoning_firewall": "inherited_from_inputs_pending_validator",
                "schema_inference": "none",
                "verifier_inference": "none",
                "reward": "not_present",
                "loader": "pending",
            },
            "files": files,
        }
        temporary_manifest = staging / "manifest.json"
        temporary_manifest.write_bytes(canonical_json(manifest) + b"\n")
        staging.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unified_dir", type=Path)
    parser.add_argument("historical_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_archived_salvage_pilot(
                unified_dir=args.unified_dir,
                historical_dir=args.historical_dir,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARCHIVED_SALVAGE_SCHEMA", "SalvagePilotError", "build_archived_salvage_pilot"]
