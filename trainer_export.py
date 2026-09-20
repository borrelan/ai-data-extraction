#!/usr/bin/env python3
"""Export an explicitly authorized reviewed pilot to trainer-neutral JSONL.

The release gate owns admission.  This module is the narrow consumer adapter
after that gate: it removes audit-only columns, preserves the conversational
messages/tool schemas required by TRL-compatible consumers, and writes a
lineage sidecar.  It never infers privacy approval, tool schemas, rewards, or
quality from a provider name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_training_data import (
    SCHEMA_VERSION,
    no_reasoning_content,
    trainer_marker_count,
    validate_dataset_record,
)
from quality_rules import scan_record, unit_id
from release_gate import REQUIRED_REVIEW_DIMENSIONS


TRAINER_EXPORT_SCHEMA = "ai-data-extraction/trainer-export/v1"
TRAINER_EXAMPLE_SCHEMA = "ai-data-extraction/trainer-example/v1"
AUTHORIZATION_SCHEMA = "ai-data-extraction/trainer-authorization/v1"
SILVER_PILOT_SCHEMA = "ai-data-extraction/silver-sft-pilot/v1"
UNIFIED_PILOT_SCHEMA = "ai-data-extraction/unified-trainer-pilot/v1"
DATASET_FILES = {"sft": "sft.jsonl", "tool_traces": "tool_traces.jsonl"}
TOOL_ROLES = frozenset({"tool"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_child(root: Path, relative: str, field: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{field} escapes its input directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parent_id(record: dict[str, Any]) -> str | None:
    for container_name in ("lineage", "metadata", "provenance"):
        container = record.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in ("parent_record_sha256", "_chunk_parent_record_sha256", "parent_unit_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _has_tool_activity(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") in TOOL_ROLES or message.get("tool_calls"):
                return True
    events = record.get("events")
    return isinstance(events, list) and any(
        isinstance(event, dict)
        and event.get("kind") in {"action", "observation"}
        for event in events
    )


def _tool_schema_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    name = tool.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _valid_tool_schemas(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    names: set[str] = set()
    for tool in value:
        if not isinstance(tool, dict):
            return False
        name = _tool_schema_name(tool)
        if name is None or name in names:
            return False
        names.add(name)
        function = tool.get("function")
        parameters = (
            function.get("parameters")
            if isinstance(function, dict)
            else tool.get("parameters", tool.get("inputSchema"))
        )
        if not isinstance(parameters, dict):
            return False
    return True


def _trainer_view(record: dict[str, Any], *, source_dataset: str) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    dataset = record.get("dataset")
    if dataset not in {"sft", "tool_trace"}:
        raise ValueError(f"unexpected canonical dataset: {dataset!r}")

    # Validate the canonical row before reducing it to the trainer allowlist.
    validate_dataset_record(record, str(dataset))
    view: dict[str, Any] = {
        "schema_version": TRAINER_EXAMPLE_SCHEMA,
        "example_id": _non_empty(record.get("example_id"), "example_id"),
        "split": record.get("split"),
        "messages": messages,
    }
    if view["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if "tools" in record:
        view["tools"] = record["tools"]
    if _has_tool_activity(record):
        if not _valid_tool_schemas(view.get("tools")):
            raise ValueError("tool_schema_missing_or_invalid")
    return view


def _review_map(pilot_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    review_spec = manifest.get("reviews")
    if not isinstance(review_spec, dict):
        raise ValueError("reviewed pilot has no review file descriptor")
    relative = _non_empty(review_spec.get("path"), "reviews.path")
    path = _safe_child(pilot_dir, relative, "reviews.path")
    expected_sha = _non_empty(review_spec.get("sha256"), "reviews.sha256")
    if sha256_file(path) != expected_sha:
        raise ValueError("review file digest does not match the pilot manifest")
    expected_records = review_spec.get("records")
    reviews: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        count = 0
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            count += 1
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"review row {line_number} is not an object")
            unit = _non_empty(value.get("unit_id"), f"review[{line_number}].unit_id")
            if unit in reviews:
                raise ValueError(f"duplicate review unit: {unit}")
            decision = value.get("decision")
            if decision not in {"accepted", "quarantined", "rejected"}:
                raise ValueError(f"invalid review decision for {unit}")
            dimensions = value.get("dimensions")
            if not isinstance(dimensions, dict):
                raise ValueError(f"review dimensions missing for {unit}")
            if decision == "accepted":
                missing = REQUIRED_REVIEW_DIMENSIONS.difference(dimensions)
                if missing:
                    raise ValueError(f"accepted review lacks dimensions for {unit}")
                if any(
                    dimensions[name] not in {"pass", "not_applicable"}
                    for name in REQUIRED_REVIEW_DIMENSIONS
                ):
                    raise ValueError(f"accepted review has failing dimensions for {unit}")
            reviews[unit] = value
    if expected_records != count:
        raise ValueError("review record count does not match the pilot manifest")
    return reviews


def _pilot_inputs(
    pilot_dir: Path,
    manifest: dict[str, Any],
    selected: Iterable[str],
) -> list[tuple[str, Path, dict[str, Any]]]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("reviewed pilot has no datasets")
    result = []
    for source_dataset in selected:
        spec = datasets.get(source_dataset)
        if not isinstance(spec, dict):
            continue
        relative = _non_empty(spec.get("path"), f"{source_dataset}.path")
        path = _safe_child(pilot_dir, relative, f"{source_dataset}.path")
        if sha256_file(path) != _non_empty(spec.get("sha256"), f"{source_dataset}.sha256"):
            raise ValueError(f"{source_dataset} digest does not match the pilot manifest")
        result.append((source_dataset, path, spec))
    if not result:
        raise ValueError("no selected trainer datasets exist in the reviewed pilot")
    return result


def inspect_pilot(
    pilot_dir: Path,
    *,
    datasets: Iterable[str] = ("sft", "tool_traces"),
) -> dict[str, Any]:
    """Audit accepted rows without authorizing or writing trainer data."""
    pilot_dir = pilot_dir.resolve()
    manifest_path = pilot_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "ai-data-extraction/reviewed-pilot/v1":
        raise ValueError("unexpected reviewed-pilot schema")
    pilot_sha256 = sha256_file(manifest_path)
    reviews = _review_map(pilot_dir, manifest)
    accepted = {unit for unit, review in reviews.items() if review["decision"] == "accepted"}
    seen_units: set[str] = set()
    parent_ids: set[str] = set()
    duplicate_units: list[str] = []
    exportable = 0
    reason_counts: Counter[str] = Counter()
    rows_by_dataset: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for source_dataset, path, spec in _pilot_inputs(pilot_dir, manifest, datasets):
        count = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                count += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{source_dataset}:{line_number} is not an object")
                unit = unit_id(record)
                if not isinstance(unit, str) or not unit:
                    raise ValueError(f"{source_dataset}:{line_number} has no unit ID")
                if unit not in accepted:
                    reason_counts["row_without_accepted_review"] += 1
                    continue
                if unit in seen_units:
                    duplicate_units.append(unit)
                    reason_counts["duplicate_accepted_unit"] += 1
                    continue
                seen_units.add(unit)
                parent = _parent_id(record)
                reasons: list[str] = []
                if parent is None:
                    reasons.append("parent_identity_missing")
                elif parent in parent_ids:
                    reasons.append("parent_collision")
                else:
                    parent_ids.add(parent)
                try:
                    view = _trainer_view(record, source_dataset=source_dataset)
                except ValueError as exc:
                    reasons.append(str(exc))
                    view = None
                if view is not None:
                    if not no_reasoning_content(view) or trainer_marker_count(view):
                        reasons.append("reasoning_or_marker_in_trainer_view")
                    findings = scan_record(view)
                    if findings.has_hard_privacy_issue or findings.has_marker:
                        reasons.append("trainer_view_firewall_finding")
                if reasons:
                    for reason in reasons:
                        reason_counts[reason] += 1
                else:
                    exportable += 1
                rows_by_dataset[source_dataset] += 1
                rows.append(
                    {
                        "unit_id": unit,
                        "source_dataset": source_dataset,
                        "source_line": line_number,
                        "source_row_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                        "parent_record_sha256": parent,
                        "exportable": not reasons,
                        "reasons": sorted(set(reasons)),
                    }
                )
        if count != spec.get("records"):
            raise ValueError(f"{source_dataset} record count does not match the pilot manifest")

    missing = sorted(accepted - seen_units)
    if missing:
        reason_counts["accepted_review_missing_from_selected_rows"] += len(missing)
    if missing or duplicate_units:
        raise ValueError("accepted review joins are not one-to-one")
    return {
        "schema_version": TRAINER_EXPORT_SCHEMA,
        "pilot_manifest_sha256": pilot_sha256,
        "pilot_training_authorized": bool(
            (manifest.get("privacy") or {}).get("training_authorized") is True
        ),
        "accepted_reviews": len(accepted),
        "accepted_rows_seen": len(seen_units),
        "exportable_rows": exportable,
        "rows_by_dataset": dict(sorted(rows_by_dataset.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
    }


def _load_authorization(path: Path, pilot_manifest_sha256: str) -> dict[str, Any]:
    authorization = _load_json(path)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unexpected trainer-authorization schema")
    if authorization.get("pilot_manifest_sha256") != pilot_manifest_sha256:
        raise ValueError("authorization is bound to a different pilot manifest")
    if authorization.get("training_authorized") is not True:
        raise ValueError("authorization does not set training_authorized=true")
    privacy = authorization.get("privacy_approval")
    if not isinstance(privacy, dict) or privacy.get("approved") is not True:
        raise ValueError("privacy approval is missing")
    _non_empty(privacy.get("authority_basis"), "privacy_approval.authority_basis")
    _non_empty(privacy.get("approved_at"), "privacy_approval.approved_at")
    certification = authorization.get("trainer_certification")
    if not isinstance(certification, dict) or certification.get("status") != "passed":
        raise ValueError("trainer certification is missing")
    _non_empty(certification.get("cert_revision"), "trainer_certification.cert_revision")
    return authorization


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    with path.open("wb") as destination:
        for row in rows:
            raw = canonical_json_bytes(row) + b"\n"
            destination.write(raw)
            digest.update(raw)
            count += 1
            byte_count += len(raw)
    return count, byte_count, digest.hexdigest()


def export_silver_sft_pilot(
    candidate_path: Path,
    output_dir: Path,
    *,
    required_model_tier: str = "tier1_frontier",
    required_training_lane: str = "primary",
    source_partition: str = "sft.candidate",
    blocked_quality_flags: Iterable[str] = (),
    quality_limitations: Iterable[str] = (),
) -> dict[str, Any]:
    """Materialize a bounded, review-only dialogue salvage pilot.

    This is deliberately narrower than ``export_trainer_release``.  It can
    emit privacy-firewall-clean, tool-free dialogue with an explicit
    ``outcome_unknown`` limitation, but it can never authorize training.  Tool
    rows, redacted rows, missing lineage, and trainer-contract failures remain
    in the decision ledger with reasons rather than being silently dropped.
    """
    candidate_path = candidate_path.resolve()
    output_dir = output_dir.resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to use existing output directory: {output_dir}")

    before = candidate_path.stat()
    digest = hashlib.sha256()
    input_bytes = 0
    input_records = 0
    decisions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    blocked_quality_flag_set = frozenset(
        flag for flag in blocked_quality_flags if isinstance(flag, str) and flag
    )

    with candidate_path.open("rb") as source:
        for input_line, raw in enumerate(source, 1):
            digest.update(raw)
            input_bytes += len(raw)
            if not raw.strip():
                continue
            input_records += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                decisions.append(
                    {
                        "input_line": input_line,
                        "decision": "excluded",
                        "reasons": ["invalid_json"],
                        "error": exc.msg,
                    }
                )
                continue
            if not isinstance(record, dict):
                decisions.append(
                    {
                        "input_line": input_line,
                        "decision": "excluded",
                        "reasons": ["row_not_object"],
                    }
                )
                continue

            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
            privacy = record.get("privacy") if isinstance(record.get("privacy"), dict) else {}
            origin = metadata.get("source_origin") if isinstance(metadata.get("source_origin"), dict) else {}
            source_quality_flags = quality.get("session_quality_flags")
            if not isinstance(source_quality_flags, list):
                source_quality_flags = []
            source_quality_flags = sorted(
                {flag for flag in source_quality_flags if isinstance(flag, str) and flag}
            )
            parent = _parent_id(record)
            example_id = record.get("example_id")
            row_sha = hashlib.sha256(raw).hexdigest()
            base = {
                "input_line": input_line,
                "source_row_sha256": row_sha,
                "example_id": example_id,
                "provider": metadata.get("provider"),
                "model_tier": metadata.get("model_tier"),
                "training_lane": metadata.get("training_lane"),
                "parent_record_sha256": parent,
                "source_file_name": metadata.get("source_file_name"),
                "source_line": metadata.get("source_line"),
                "source_quality_flags": source_quality_flags,
            }
            reasons: list[str] = []
            if record.get("dataset") != "sft":
                reasons.append("dataset_not_sft")
            if metadata.get("model_tier") != required_model_tier:
                reasons.append("model_tier_not_selected")
            if metadata.get("training_lane") != required_training_lane:
                reasons.append("training_lane_not_selected")
            if quality.get("has_tools") is not False or _has_tool_activity(record):
                reasons.append("tool_activity_not_silver_sft")
            for flag in sorted(blocked_quality_flag_set.intersection(source_quality_flags)):
                reasons.append(f"source_quality_flag:{flag}")
            if int(privacy.get("structural_redactions") or 0) > 0:
                reasons.append("privacy_redactions_present")
            if parent is None:
                reasons.append("parent_identity_missing")

            view: dict[str, Any] | None = None
            if not reasons:
                try:
                    view = _trainer_view(record, source_dataset="sft")
                except ValueError as exc:
                    reasons.append(f"trainer_contract:{exc}")
            if view is not None:
                view.pop("tools", None)
                if not no_reasoning_content(view) or trainer_marker_count(view):
                    reasons.append("reasoning_or_marker_in_trainer_view")
                findings = scan_record(view)
                if findings.has_hard_privacy_issue:
                    reasons.append("trainer_view_firewall_privacy")
                if findings.has_marker:
                    reasons.append("trainer_view_firewall_marker")

            decision = dict(base)
            decision["decision"] = "eligible" if not reasons else "excluded"
            decision["reasons"] = sorted(set(reasons))
            decisions.append(decision)
            if not reasons and view is not None:
                eligible.append(
                    {
                        "view": view,
                        "record": record,
                        "input_line": input_line,
                        "source_row_sha256": row_sha,
                        "parent_record_sha256": parent,
                    }
                )

    after = candidate_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"candidate source changed during pilot: {candidate_path}")

    input_sha256 = digest.hexdigest()
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_parent.setdefault(str(item["parent_record_sha256"]), []).append(item)

    selected: list[dict[str, Any]] = []
    selected_rows: dict[int, dict[str, Any]] = {}
    for parent, items in by_parent.items():
        items.sort(
            key=lambda item: (
                -len(item["view"].get("messages", [])),
                item["input_line"],
                str(item["view"].get("example_id")),
            )
        )
        winner = items[0]
        selected.append(winner)
        selected_rows[winner["input_line"]] = winner
        for duplicate in items[1:]:
            for decision in decisions:
                if decision.get("input_line") == duplicate["input_line"]:
                    decision["decision"] = "excluded"
                    decision["reasons"] = ["duplicate_parent_selection"]
                    break

    selected.sort(key=lambda item: str(item["parent_record_sha256"]))
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        split = "validation" if len(selected) > 1 and index % 5 == 0 else "train"
        view = dict(item["view"])
        view["split"] = split
        (validation_rows if split == "validation" else train_rows).append(view)
        record = item["record"]
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        origin = metadata.get("source_origin") if isinstance(metadata.get("source_origin"), dict) else {}
        lineage_rows.append(
            {
                "schema_version": "ai-data-extraction/silver-sft-lineage/v1",
                "example_id": view["example_id"],
                "split": split,
                "source_dataset": source_partition,
                "source_line": item["input_line"],
                "source_row_sha256": item["source_row_sha256"],
                "parent_record_sha256": item["parent_record_sha256"],
                "source_file_name": metadata.get("source_file_name"),
                "source_file_sha256": metadata.get("source_file_sha256"),
                "source_origin_file_name": origin.get("source_file_name"),
                "source_origin_file_sha256": origin.get("source_file_sha256"),
                "source_message_range": (
                    record.get("lineage", {}).get("source_message_range")
                    if isinstance(record.get("lineage"), dict)
                    else None
                ),
                "provider": metadata.get("provider"),
                "model_tier": metadata.get("model_tier"),
                "training_lane": metadata.get("training_lane"),
                "source_quality_flags": item["record"].get("quality", {}).get(
                    "session_quality_flags", []
                )
                if isinstance(item["record"].get("quality"), dict)
                else [],
                "quality_tier": "silver_sft",
                "limitations": [
                    "outcome_unknown",
                    "not_human_adjudicated",
                    "training_authorized_false",
                    *quality_limitations,
                ],
            }
        )

    for decision in decisions:
        if decision.get("decision") == "eligible":
            decision["decision"] = "selected" if decision.get("input_line") in selected_rows else "excluded"
            if decision["decision"] == "excluded":
                decision["reasons"] = ["duplicate_parent_selection"]

    excluded_reasons: Counter[str] = Counter()
    for decision in decisions:
        if decision.get("decision") == "excluded":
            excluded_reasons.update(decision.get("reasons", []))

    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError(f"refusing to use existing staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        files: dict[str, dict[str, Any]] = {}
        for filename, rows in (
            ("train.jsonl", train_rows),
            ("validation.jsonl", validation_rows),
            ("lineage.jsonl", lineage_rows),
            ("decisions.jsonl", decisions),
        ):
            count, byte_count, file_sha256 = _write_jsonl(staging / filename, rows)
            files[filename] = {
                "records": count,
                "bytes": byte_count,
                "sha256": file_sha256,
            }

        manifest = {
            "schema_version": SILVER_PILOT_SCHEMA,
            "status": "review_only",
            "training_authorized": False,
            "trainer_loadable": True,
            "format": "messages_jsonl",
            "source": {
                "path": str(candidate_path),
                "sha256": input_sha256,
                "bytes": input_bytes,
                "records": input_records,
            },
            "selection": {
                "source_partition": source_partition,
                "model_tier": required_model_tier,
                "training_lane": required_training_lane,
                "tool_activity": "excluded",
                "privacy_redactions": "excluded",
                "blocked_quality_flags": sorted(blocked_quality_flag_set),
                "parent_policy": "one_selected_row_per_parent",
                "split_policy": "parent-sorted; every fifth selected parent is validation",
            },
            "quality": {
                "tier": "silver_sft",
                "limitations": [
                    "outcome_unknown",
                    "not_human_adjudicated",
                    "privacy_approval_not_granted",
                    "historical_tool_rows_not_included",
                    *quality_limitations,
                ],
            },
            "counts": {
                "input_records": input_records,
                "eligible_before_parent_dedup": len(eligible),
                "selected_records": len(selected),
                "parents_selected": len({item["parent_record_sha256"] for item in selected}),
                "train": len(train_rows),
                "validation": len(validation_rows),
                "excluded": sum(1 for decision in decisions if decision.get("decision") == "excluded"),
                "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
            },
            "validation": {
                "trainer_projection": "passed",
                "reasoning_firewall": "passed",
                "privacy_firewall": "passed_for_selected_rows",
                "parent_disjoint": "passed",
                "split_deterministic": "passed",
                "reward": "not_present",
            },
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        raise


def _validate_trainer_row(row: dict[str, Any], *, has_tools: bool) -> None:
    expected = {
        "schema_version",
        "example_id",
        "split",
        "messages",
        *(("tools",) if has_tools else ()),
    }
    if set(row) != expected:
        raise ValueError(
            f"trainer row keys mismatch: expected {sorted(expected)}, got {sorted(row)}"
        )
    if row.get("schema_version") != TRAINER_EXAMPLE_SCHEMA:
        raise ValueError("trainer row schema mismatch")
    if not isinstance(row.get("example_id"), str) or not row["example_id"]:
        raise ValueError("trainer row has no example ID")
    if row.get("split") not in {"train", "validation", "test"}:
        raise ValueError("trainer row has invalid split")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("trainer row messages are missing")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("trainer row message is not an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("trainer row has an invalid message role")
        if not isinstance(message.get("content"), str):
            raise ValueError("trainer row message content is not a string")
    if not no_reasoning_content(row) or trainer_marker_count(row):
        raise ValueError("trainer row contains reasoning or trainer markers")
    findings = scan_record(row)
    if findings.has_hard_privacy_issue or findings.has_marker:
        raise ValueError("trainer row fails privacy/marker firewall")
    if has_tools:
        if not _has_tool_activity(row) or not _valid_tool_schemas(row.get("tools")):
            raise ValueError("tool trainer row has invalid activity or schemas")


def _verified_artifact_files(
    artifact_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"artifact has no file descriptors: {artifact_dir}")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for filename, descriptor in files.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid file descriptor: {filename}")
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = _non_empty(descriptor.get("sha256"), f"{filename}.sha256")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"artifact file digest mismatch: {path}")
        expected_records = descriptor.get("records")
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(value)
        if expected_records != len(rows):
            raise ValueError(f"artifact record count mismatch: {path}")
        loaded[filename] = rows
    return loaded


def _load_unified_input_artifact(
    artifact_dir: Path,
    *,
    artifact_kind: str,
    expected_manifest_schema: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != expected_manifest_schema:
        raise ValueError(f"unexpected artifact schema: {manifest_path}")
    files = _verified_artifact_files(artifact_dir, manifest)
    lineage_rows = files.get("lineage.jsonl")
    if lineage_rows is None:
        raise ValueError(f"artifact has no lineage sidecar: {artifact_dir}")
    lineage_by_id: dict[str, dict[str, Any]] = {}
    for lineage in lineage_rows:
        example_id = lineage.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in lineage_by_id:
            raise ValueError(f"artifact lineage identity is missing or duplicated: {artifact_dir}")
        lineage_by_id[example_id] = lineage

    if artifact_kind == "silver":
        if manifest.get("training_authorized") is not False:
            raise ValueError("silver artifact must remain unauthorized")
        row_files = ("train.jsonl", "validation.jsonl")
    elif artifact_kind == "gold_dialogue":
        if manifest.get("training_authorized") is not True:
            raise ValueError("gold dialogue artifact is not authorized")
        row_files = ("sft.jsonl",)
    elif artifact_kind == "gold_tool":
        if manifest.get("training_authorized") is not True:
            raise ValueError("gold tool artifact is not authorized")
        row_files = ("tool_sft.jsonl",)
    else:
        raise ValueError(f"unknown artifact kind: {artifact_kind}")

    rows: list[dict[str, Any]] = []
    for filename in row_files:
        for row in files.get(filename, []):
            has_tools = artifact_kind == "gold_tool"
            _validate_trainer_row(row, has_tools=has_tools)
            lineage = lineage_by_id.get(row["example_id"])
            if lineage is None:
                raise ValueError(f"row has no lineage: {artifact_dir}/{filename}")
            parent = lineage.get("parent_record_sha256")
            if not isinstance(parent, str) or not parent:
                raise ValueError(f"row lineage has no parent identity: {artifact_dir}/{filename}")
            rows.append(
                {
                    "row": row,
                    "lineage": lineage,
                    "artifact_dir": artifact_dir,
                    "artifact_name": artifact_dir.name,
                    "artifact_manifest_sha256": sha256_file(manifest_path),
                    "artifact_kind": artifact_kind,
                    "quality_tier": (
                        manifest.get("quality", {}).get("tier", "silver_sft")
                        if artifact_kind == "silver"
                        else "gold"
                    ),
                    "source_training_authorized": manifest.get("training_authorized") is True,
                    "dataset": "tool_sft" if has_tools else "sft",
                    "parent_record_sha256": parent,
                }
            )
    return manifest, rows, lineage_rows


def _select_one_per_parent(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        groups.setdefault(item["parent_record_sha256"], []).append(item)
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for parent in sorted(groups):
        candidates = sorted(
            groups[parent],
            key=lambda item: (
                0 if item["source_training_authorized"] else 1,
                item["artifact_name"],
                str(item["row"].get("example_id")),
            ),
        )
        selected.append(candidates[0])
        excluded.extend(candidates[1:])
    return selected, excluded


def _assign_deterministic_splits(rows: list[dict[str, Any]]) -> None:
    parents = sorted({item["parent_record_sha256"] for item in rows})
    split_by_parent = {
        parent: "validation" if len(parents) > 1 and index % 5 == 0 else "train"
        for index, parent in enumerate(parents)
    }
    for item in rows:
        item["split"] = split_by_parent[item["parent_record_sha256"]]


def export_unified_training_pilot(
    *,
    silver_pilot_dirs: Iterable[Path],
    gold_dialogue_dir: Path,
    gold_tool_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Combine already-verified pilot/release artifacts without reading source archives."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to use existing output directory: {output_dir}")

    inputs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for directory in silver_pilot_dirs:
        manifest, rows, _ = _load_unified_input_artifact(
            Path(directory),
            artifact_kind="silver",
            expected_manifest_schema=SILVER_PILOT_SCHEMA,
        )
        inputs.extend(rows)
        artifacts.append(
            {
                "name": Path(directory).resolve().name,
                "path": str(Path(directory).resolve()),
                "manifest_sha256": sha256_file(Path(directory).resolve() / "manifest.json"),
                "schema_version": manifest["schema_version"],
                "training_authorized": manifest["training_authorized"],
            }
        )

    for directory, artifact_kind in (
        (gold_dialogue_dir, "gold_dialogue"),
        (gold_tool_dir, "gold_tool"),
    ):
        manifest, rows, _ = _load_unified_input_artifact(
            Path(directory),
            artifact_kind=artifact_kind,
            expected_manifest_schema=TRAINER_EXPORT_SCHEMA,
        )
        inputs.extend(rows)
        artifacts.append(
            {
                "name": Path(directory).resolve().name,
                "path": str(Path(directory).resolve()),
                "manifest_sha256": sha256_file(Path(directory).resolve() / "manifest.json"),
                "schema_version": manifest["schema_version"],
                "training_authorized": manifest["training_authorized"],
            }
        )

    dialogue = [item for item in inputs if item["dataset"] == "sft"]
    tool_rows = [item for item in inputs if item["dataset"] == "tool_sft"]
    selected_dialogue, excluded_dialogue = _select_one_per_parent(dialogue)
    selected_tools, excluded_tools = _select_one_per_parent(tool_rows)
    selected_all = [*selected_dialogue, *selected_tools]
    selected_example_ids = [item["row"]["example_id"] for item in selected_all]
    if len(selected_example_ids) != len(set(selected_example_ids)):
        raise ValueError("unified pilot has duplicate selected example IDs")
    _assign_deterministic_splits(selected_all)

    decisions: list[dict[str, Any]] = []
    selected_ids = {
        id(item)
        for item in (*selected_dialogue, *selected_tools)
    }
    excluded_by_id = {
        id(item): item for item in (*excluded_dialogue, *excluded_tools)
    }
    for item in inputs:
        is_selected = id(item) in selected_ids
        decision = {
            "dataset": item["dataset"],
            "artifact": item["artifact_name"],
            "artifact_manifest_sha256": item["artifact_manifest_sha256"],
            "example_id": item["row"]["example_id"],
            "parent_record_sha256": item["parent_record_sha256"],
            "source_row_sha256": item["lineage"].get("source_row_sha256"),
            "quality_tier": item["quality_tier"],
            "source_training_authorized": item["source_training_authorized"],
            "decision": "selected" if is_selected else "excluded",
            "reasons": [
                "selected_unified_pilot"
                if is_selected
                else "duplicate_parent_selection"
            ],
        }
        if is_selected:
            decision["split"] = item["split"]
        elif id(item) not in excluded_by_id:
            raise ValueError("unclassified unified input row")
        decisions.append(decision)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    tool_train_rows: list[dict[str, Any]] = []
    tool_validation_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for item in selected_all:
        row = dict(item["row"])
        row["split"] = item["split"]
        if item["dataset"] == "sft":
            (validation_rows if item["split"] == "validation" else train_rows).append(row)
        else:
            (tool_validation_rows if item["split"] == "validation" else tool_train_rows).append(row)
        lineage = dict(item["lineage"])
        lineage.update(
            {
                "schema_version": "ai-data-extraction/unified-trainer-lineage/v1",
                "dataset": item["dataset"],
                "split": item["split"],
                "source_artifact": item["artifact_name"],
                "source_artifact_manifest_sha256": item["artifact_manifest_sha256"],
                "source_training_authorized": item["source_training_authorized"],
                "quality_tier": item["quality_tier"],
                "parent_record_sha256": item["parent_record_sha256"],
                "limitations": (
                    ["outcome_unknown", "not_human_adjudicated"]
                    if item["quality_tier"] == "silver_sft"
                    else ["reward_not_exported"]
                ),
            }
        )
        lineage_rows.append(lineage)

    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError(f"refusing to use existing staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        file_rows = (
            ("train.jsonl", train_rows),
            ("validation.jsonl", validation_rows),
            ("tool_train.jsonl", tool_train_rows),
            ("tool_validation.jsonl", tool_validation_rows),
            ("lineage.jsonl", lineage_rows),
            ("decisions.jsonl", decisions),
        )
        files: dict[str, dict[str, Any]] = {}
        for filename, rows in file_rows:
            count, byte_count, file_sha256 = _write_jsonl(staging / filename, rows)
            files[filename] = {
                "records": count,
                "bytes": byte_count,
                "sha256": file_sha256,
            }
        tier_counts = Counter(item["quality_tier"] for item in (*selected_dialogue, *selected_tools))
        dataset_counts = Counter(item["dataset"] for item in (*selected_dialogue, *selected_tools))
        manifest = {
            "schema_version": UNIFIED_PILOT_SCHEMA,
            "status": "review_only",
            "training_authorized": False,
            "trainer_loadable": True,
            "format": "messages_jsonl",
            "inputs": artifacts,
            "selection": {
                "parent_policy": "one_selected_row_per_parent; gold_over_silver",
                "split_policy": "parent-sorted; every fifth selected parent is validation",
                "dialogue_source": "verified silver pilots plus authorized gold dialogue",
                "tool_source": "authorized gold tool release only",
            },
            "quality": {
                "tier_counts": dict(sorted(tier_counts.items())),
                "dataset_counts": dict(sorted(dataset_counts.items())),
                "limitations": [
                    "silver_rows_outcome_unknown",
                    "silver_rows_not_human_adjudicated",
                    "training_authorized_false",
                    "rewards_not_exported",
                ],
            },
            "counts": {
                "input_rows": len(inputs),
                "selected_rows": len(selected_dialogue) + len(selected_tools),
                "excluded_duplicate_rows": len(excluded_dialogue) + len(excluded_tools),
                "sft_input_rows": len(dialogue),
                "sft_selected_rows": len(selected_dialogue),
                "sft_train": len(train_rows),
                "sft_validation": len(validation_rows),
                "tool_input_rows": len(tool_rows),
                "tool_selected_rows": len(selected_tools),
                "tool_train": len(tool_train_rows),
                "tool_validation": len(tool_validation_rows),
            },
            "validation": {
                "source_artifact_hashes": "passed",
                "trainer_projection": "passed",
                "privacy_reasoning_firewall": "passed",
                "parent_disjoint": "passed",
                "split_deterministic": "passed",
                "explicit_decisions": "passed",
                "loader_validation": "pending_external_loader",
                "reward": "not_present",
            },
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        raise


def export_trainer_release(
    pilot_dir: Path,
    authorization_path: Path,
    output_dir: Path,
    *,
    datasets: Iterable[str] = ("sft", "tool_traces"),
) -> dict[str, Any]:
    pilot_dir = pilot_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to use non-empty output directory: {output_dir}")
    audit = inspect_pilot(pilot_dir, datasets=datasets)
    authorization = _load_authorization(
        authorization_path.resolve(), audit["pilot_manifest_sha256"]
    )
    if audit["exportable_rows"] != audit["accepted_reviews"]:
        raise ValueError(
            "accepted pilot rows are not all trainer-exportable: "
            + json.dumps(audit["reason_counts"], sort_keys=True)
        )

    manifest = _load_json(pilot_dir / "manifest.json")
    reviews = _review_map(pilot_dir, manifest)
    accepted = {unit for unit, review in reviews.items() if review["decision"] == "accepted"}
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        raise FileExistsError(f"refusing to use existing staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        output_rows: dict[str, list[dict[str, Any]]] = {"sft": [], "tool_sft": []}
        lineage_rows: list[dict[str, Any]] = []
        for source_dataset, path, _spec in _pilot_inputs(pilot_dir, manifest, datasets):
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    unit = unit_id(record)
                    if unit not in accepted:
                        continue
                    view = _trainer_view(record, source_dataset=source_dataset)
                    has_tool = _has_tool_activity(record)
                    output_rows["tool_sft" if has_tool else "sft"].append(view)
                    lineage_rows.append(
                        {
                            "example_id": view["example_id"],
                            "source_dataset": source_dataset,
                            "source_line": line_number,
                            "source_row_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                            "parent_record_sha256": _parent_id(record),
                            "review_id": reviews[unit]["review_id"],
                        }
                    )

        files: dict[str, dict[str, Any]] = {}
        for name, rows in (
            ("sft.jsonl", output_rows["sft"]),
            ("tool_sft.jsonl", output_rows["tool_sft"]),
            ("lineage.jsonl", lineage_rows),
        ):
            target = staging / name
            count, byte_count, digest = _write_jsonl(target, rows)
            files[name] = {
                "records": count,
                "bytes": byte_count,
                "sha256": digest,
            }

        export_manifest = {
            "schema_version": TRAINER_EXPORT_SCHEMA,
            "generated_at": authorization.get("authorized_at") or "not_recorded",
            "source": {
                "pilot_manifest_sha256": audit["pilot_manifest_sha256"],
                "pilot_dir_name": pilot_dir.name,
                "review_manifest_sha256": manifest["review_manifest"]["sha256"],
            },
            "authorization": {
                "path": authorization_path.name,
                "sha256": sha256_file(authorization_path.resolve()),
                "authority_basis": authorization["privacy_approval"]["authority_basis"],
                "cert_revision": authorization["trainer_certification"]["cert_revision"],
            },
            "policy": {
                "backend_neutral": True,
                "trainer_contract": "messages_plus_optional_tools",
                "reasoning": "allowlist_projection_and_firewall",
                "tool_schema": "required_for_any_tool_activity",
                "rewards": "not_exported",
            },
            "training_authorized": True,
            "counts": {
                "accepted_reviews": len(accepted),
                "sft": len(output_rows["sft"]),
                "tool_sft": len(output_rows["tool_sft"]),
            },
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(export_manifest) + b"\n")
        staging.replace(output_dir)
        export_manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return export_manifest
    except BaseException:
        # Preserve a failed staging directory for diagnosis; it has no final
        # manifest and therefore cannot be mistaken for a trainer release.
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true", help="Audit without writing trainer data")
    mode.add_argument("--authorization", type=Path, help="Explicit trainer authorization JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_FILES),
        default=list(DATASET_FILES),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.audit:
            print(json.dumps(inspect_pilot(args.pilot_dir, datasets=args.datasets), indent=2, sort_keys=True))
            return 0
        if args.authorization is None or args.output_dir is None:
            raise ValueError("--authorization and --output-dir are required unless --audit is used")
        result = export_trainer_release(
            args.pilot_dir,
            args.authorization,
            args.output_dir,
            datasets=args.datasets,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
