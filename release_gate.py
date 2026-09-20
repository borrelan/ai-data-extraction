"""Stream canonical datasets into review partitions without authorizing training.

This is a row-level release gate, not a trainer. It preserves the canonical
inputs, copies rows byte-for-byte into candidate/review/quarantine partitions,
and writes a decision ledger containing identifiers, counts, and reason codes
only. The manifest is intentionally never training-authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from quality_rules import (
    RecordFindings,
    lineage_issues,
    model_tier,
    prompt_group_id,
    quality_gate,
    scan_record,
    tool_edge_status,
    training_lane,
    unit_id,
)


SCHEMA_VERSION = "ai-data-extraction/release-gate/v1"
REVIEW_MANIFEST_SCHEMA = "agentir/hardening/quality-review-manifest/v1"
REVIEWED_PILOT_SCHEMA = "ai-data-extraction/reviewed-pilot/v1"
REQUIRED_REVIEW_DIMENSIONS = frozenset(
    {
        "structural",
        "tool_integrity",
        "tool_correctness",
        "observation_grounding",
        "privacy",
        "reasoning_exclusion",
        "provenance",
        "task_quality",
        "human_review",
        "contamination",
        "deduplication",
    }
)
PARTITIONS = ("candidate", "review_required", "quarantine")
DATASET_FILES = {
    "sft": "sft.jsonl",
    "trajectories": "trajectories.jsonl",
    "tool_traces": "tool_traces.jsonl",
    "action_windows": "action_windows.jsonl",
    "preferences": "preferences.jsonl",
    "rl_prompts": "rl_prompts.jsonl",
    "rejected": "rejected.jsonl",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _quality_value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    quality = record.get("quality")
    if isinstance(quality, dict) and key in quality:
        return quality[key]
    return record.get(key, default)


def _base_partition(label: str, record: dict[str, Any]) -> tuple[str, list[str]]:
    if label == "rejected":
        return "quarantine", ["source_rejected"]
    gate = quality_gate(record)
    if gate == "candidate":
        return "candidate", []
    if gate == "review_required":
        return "review_required", ["source_quality_review_required"]
    if gate in {"quarantine", "unassessed", "unspecified"}:
        return "quarantine", [f"source_quality_{gate}"]
    return "quarantine", ["source_quality_unknown"]


def _privacy_reasons(findings: RecordFindings) -> list[str]:
    reasons: list[str] = []
    if findings.hidden_keys:
        reasons.append("hidden_trainer_key")
    if findings.raw_keys:
        reasons.append("raw_or_debug_field")
    if findings.secret_keys or findings.secret_values:
        reasons.append("secret_indicator")
    if findings.private_values:
        reasons.append("private_value_indicator")
    return reasons


def _tool_reasons(label: str, record: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if label not in {"sft", "trajectories", "tool_traces"}:
        return [], {}
    status = tool_edge_status(record)
    if not status.has_actions:
        if label == "tool_traces" and _quality_value(record, "has_tools", False):
            return ["tool_trace_without_actions"], status.summary()
        return [], status.summary()
    if status.complete:
        return [], status.summary()
    return ["tool_edges_incomplete"], status.summary()


def _action_window_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _quality_value(record, "observation_present") is not True:
        reasons.append("tool_observation_missing")
    if _quality_value(record, "observation_match") != "call-id":
        reasons.append("tool_observation_unmatched")
    if _quality_value(record, "source_episode_status") != "accepted":
        reasons.append("source_episode_not_accepted")
    return reasons


def _rl_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _quality_value(record, "rl_ready") is not True:
        reasons.append("rl_not_ready")
    reward_status = record.get("reward_status") or _quality_value(
        record, "reward_status"
    )
    if reward_status != "scored":
        reasons.append("rl_reward_unscored")
    return reasons


def decide_row(
    label: str,
    record: dict[str, Any],
    *,
    allowed_model_tiers: set[str] | None,
    duplicate_counts: Counter[str],
    duplicate_cap: int,
) -> tuple[str, list[str], RecordFindings, dict[str, Any], str | None]:
    partition, reasons = _base_partition(label, record)
    findings = scan_record(record)
    hard_reasons = _privacy_reasons(findings)
    if hard_reasons:
        partition = "quarantine"
        reasons.extend(hard_reasons)

    marker_reasons = []
    if findings.has_marker:
        marker_reasons.append("visible_reasoning_marker")
        if partition == "candidate":
            partition = "review_required"
    reasons.extend(marker_reasons)

    missing_lineage = lineage_issues(record)
    if missing_lineage:
        partition = "quarantine"
        reasons.extend(f"lineage_{issue.removesuffix('_missing')}" for issue in missing_lineage)

    tier = model_tier(record)
    if allowed_model_tiers is not None and tier not in allowed_model_tiers:
        if partition == "candidate":
            partition = "review_required"
        reasons.append("model_tier_not_selected")

    tool_reasons, tool_summary = _tool_reasons(label, record)
    if tool_reasons:
        if partition == "candidate":
            partition = "review_required"
        reasons.extend(tool_reasons)

    if label == "action_windows":
        action_reasons = _action_window_reasons(record)
        if action_reasons and partition == "candidate":
            partition = "review_required"
        reasons.extend(action_reasons)
        tool_summary = {
            **tool_summary,
            "observation_match": _quality_value(record, "observation_match"),
            "observation_present": _quality_value(record, "observation_present"),
        }

    if label == "rl_prompts":
        rl_reasons = _rl_reasons(record)
        if rl_reasons and partition == "candidate":
            partition = "review_required"
        reasons.extend(rl_reasons)

    group = prompt_group_id(record) if label == "rl_prompts" else None
    if group is not None:
        prior = duplicate_counts[group]
        duplicate_counts[group] += 1
        if prior >= duplicate_cap:
            if partition == "candidate":
                partition = "review_required"
            reasons.append("duplicate_prompt_group_cap")

    # Keep the ledger deterministic when several rules identify the same row.
    reasons = list(dict.fromkeys(reasons))
    return partition, reasons, findings, tool_summary, group


@dataclass(slots=True)
class WriterStats:
    final_path: Path
    temporary_path: Path
    handle: Any
    digest: Any = field(default_factory=hashlib.sha256)
    bytes_written: int = 0
    records: int = 0

    def write(self, raw_line: bytes) -> None:
        self.handle.write(raw_line)
        self.digest.update(raw_line)
        self.bytes_written += len(raw_line)
        self.records += 1

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def sha256(self) -> str:
        return self.digest.hexdigest()


def _new_writer(staging_dir: Path, final_path: Path) -> WriterStats:
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=staging_dir,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    return WriterStats(final_path=final_path, temporary_path=temporary_path, handle=handle)


def _write_decision(writer: WriterStats, decision: dict[str, Any]) -> None:
    raw = (json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    writer.write(raw)


def _validate_manifest(manifest: dict[str, Any], input_dir: Path) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Canonical manifest has no outputs map")
    for filename in DATASET_FILES.values():
        entry = outputs.get(filename)
        path = input_dir / filename
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"Canonical manifest lacks hash for {filename}")
        if not path.is_file():
            raise FileNotFoundError(path)


def gate_corpus(
    input_dir: Path,
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    datasets: Iterable[str] = DATASET_FILES,
    allowed_model_tiers: set[str] | None = None,
    duplicate_cap: int = 1,
) -> dict[str, Any]:
    if duplicate_cap < 1:
        raise ValueError("duplicate_cap must be at least 1")
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the canonical input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".staging"
    staging_dir.mkdir()

    manifest_path = (manifest_path or input_dir / "manifest.json").resolve()
    manifest = load_json(manifest_path)
    _validate_manifest(manifest, input_dir)
    selected = list(dict.fromkeys(datasets))
    unknown = set(selected) - set(DATASET_FILES)
    if unknown:
        raise ValueError(f"Unknown dataset labels: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("At least one dataset is required")

    decisions_writer = _new_writer(staging_dir, output_dir / "decisions.jsonl")
    dataset_results: dict[str, Any] = {}
    reason_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    total_input_records = 0
    total_output_records = 0

    try:
        for label in selected:
            filename = DATASET_FILES[label]
            input_path = input_dir / filename
            expected = manifest["outputs"][filename]
            writers = {
                partition: _new_writer(
                    staging_dir, output_dir / f"{label}.{partition}.jsonl"
                )
                for partition in PARTITIONS
            }
            input_digest = hashlib.sha256()
            input_bytes = 0
            input_records = 0
            invalid_lines = 0
            label_reason_counts: Counter[str] = Counter()
            label_partition_counts: Counter[str] = Counter()

            with input_path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    input_digest.update(raw_line)
                    input_bytes += len(raw_line)
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        invalid_lines += 1
                        raise ValueError(
                            f"Invalid canonical JSONL in {input_path} at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        invalid_lines += 1
                        raise ValueError(
                            f"Canonical JSONL row is not an object in {input_path} "
                            f"at line {line_number}"
                        )
                    input_records += 1
                    partition, reasons, findings, tool_summary, group = decide_row(
                        label,
                        record,
                        allowed_model_tiers=allowed_model_tiers,
                        duplicate_counts=duplicate_counts,
                        duplicate_cap=duplicate_cap,
                    )
                    writers[partition].write(raw_line)
                    label_partition_counts[partition] += 1
                    partition_counts[f"{label}:{partition}"] += 1
                    total_output_records += 1
                    for reason in reasons:
                        label_reason_counts[reason] += 1
                        reason_counts[f"{label}:{reason}"] += 1
                    _write_decision(
                        decisions_writer,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "dataset": label,
                            "input_file": filename,
                            "input_line": line_number,
                            "unit_id": unit_id(record),
                            "partition": partition,
                            "reason_codes": reasons,
                            "quality_gate": quality_gate(record),
                            "training_lane": training_lane(record),
                            "model_tier": model_tier(record),
                            "prompt_group_id": group,
                            "findings": findings.summary(),
                            "tool_edges": tool_summary,
                        },
                    )

            actual_input_sha256 = input_digest.hexdigest()
            if actual_input_sha256 != expected["sha256"]:
                raise ValueError(
                    f"Canonical input hash mismatch for {filename}: "
                    f"expected {expected['sha256']}, got {actual_input_sha256}"
                )
            if input_records != expected.get("records"):
                raise ValueError(
                    f"Canonical input record count mismatch for {filename}: "
                    f"expected {expected.get('records')}, got {input_records}"
                )
            for writer in writers.values():
                writer.close()
            total_input_records += input_records
            dataset_results[label] = {
                "input_file": filename,
                "input_sha256": actual_input_sha256,
                "input_bytes": input_bytes,
                "input_records": input_records,
                "invalid_lines": invalid_lines,
                "partitions": {
                    partition: {
                        "path": writer.final_path.name,
                        "records": writer.records,
                        "bytes": writer.bytes_written,
                        "sha256": writer.sha256(),
                    }
                    for partition, writer in writers.items()
                },
                "partition_counts": dict(sorted(label_partition_counts.items())),
                "reason_counts": dict(sorted(label_reason_counts.items())),
            }

        decisions_writer.close()
        os.replace(decisions_writer.temporary_path, decisions_writer.final_path)
        for dataset in dataset_results.values():
            for partition in dataset["partitions"].values():
                staged = next(
                    path
                    for path in staging_dir.iterdir()
                    if path.name.startswith(f".{partition['path']}.")
                    and path.name.endswith(".tmp")
                )
                os.replace(staged, output_dir / partition["path"])

        policy = manifest.get("policy") if isinstance(manifest.get("policy"), dict) else {}
        privacy_approved = policy.get("privacy_approved") is True
        release_manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "canonical_manifest": {
                "path": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "builder_version": manifest.get("builder_version"),
                "parser_revision": policy.get("parser_revision"),
            },
            "policy": {
                "partitions": list(PARTITIONS),
                "duplicate_prompt_cap": duplicate_cap,
                "allowed_model_tiers": (
                    sorted(allowed_model_tiers)
                    if allowed_model_tiers is not None
                    else "all"
                ),
                "row_content": "copied byte-for-byte from canonical input",
                "decision_content": "identifiers, structural findings, and reason codes only",
            },
            "privacy": {
                "canonical_privacy_approved": privacy_approved,
                "training_authorized": False,
                "blocked_reasons": [
                    "privacy_approval_missing" if not privacy_approved else None,
                    "trainer_certification_not_implemented",
                ],
            },
            "counts": {
                "input_records": total_input_records,
                "partition_records": dict(sorted(partition_counts.items())),
                "output_records": total_output_records,
                "decision_records": decisions_writer.records,
            },
            "datasets": dataset_results,
            "reason_counts": dict(sorted(reason_counts.items())),
            "decisions": {
                "path": decisions_writer.final_path.name,
                "records": decisions_writer.records,
                "bytes": decisions_writer.bytes_written,
                "sha256": sha256_file(decisions_writer.final_path),
            },
        }
        manifest_target = output_dir / "manifest.json"
        manifest_temp = output_dir / ".manifest.json.tmp"
        manifest_temp.write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_temp, manifest_target)
        release_manifest["manifest_sha256"] = sha256_file(manifest_target)
        return release_manifest
    except BaseException:
        # Keep the staging directory for posterity; it has no release manifest
        # and therefore cannot be mistaken for a valid release.
        try:
            decisions_writer.close()
        except OSError:
            pass
        raise


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Review manifest field {field} must be a non-empty string")
    return value.strip()


def load_review_manifest_artifact(
    path: Path,
    *,
    expected_release_manifest_sha256: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and validate the external typed review-manifest contract.

    The upstream review writer is intentionally not a runtime dependency of
    this standard-library release gate.  This function is the narrow ingress
    adapter: it verifies the published digest and the fields this owner needs,
    then exposes plain provider-neutral review decisions internally.
    """

    path = path.resolve()
    artifact = load_json(path)
    if artifact.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
        raise ValueError("Unexpected review manifest schema")
    declared_digest = _non_empty_text(artifact.get("manifest_sha256"), "manifest_sha256")
    manifest = artifact.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Review manifest has no manifest object")
    actual_digest = sha256_bytes(canonical_json_bytes(manifest))
    if declared_digest != actual_digest:
        raise ValueError("Review manifest digest does not match its content")

    manifest_revision = _non_empty_text(manifest.get("manifest_revision"), "manifest_revision")
    snapshot_revision = _non_empty_text(manifest.get("snapshot_revision"), "snapshot_revision")
    rubric_revision = _non_empty_text(manifest.get("rubric_revision"), "rubric_revision")
    reviews = manifest.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Review manifest must contain at least one review")

    by_unit: dict[str, dict[str, Any]] = {}
    review_ids: set[str] = set()
    release_evidence = (
        f"sha256:{expected_release_manifest_sha256}"
        if expected_release_manifest_sha256
        else None
    )
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("Review manifest contains a non-object review")
        unit = _non_empty_text(review.get("unit_id"), "review.unit_id")
        review_id = _non_empty_text(review.get("review_id"), "review.review_id")
        if unit in by_unit or review_id in review_ids:
            raise ValueError("Review manifest contains duplicate unit or review identity")
        if review.get("manifest_revision") != manifest_revision:
            raise ValueError("Review manifest revision does not match its review")
        if review.get("snapshot_revision") != snapshot_revision:
            raise ValueError("Review snapshot revision does not match its review")
        if review.get("rubric_revision") != rubric_revision:
            raise ValueError("Review rubric revision does not match its review")
        decision = review.get("decision")
        if decision not in {"accepted", "quarantined", "rejected"}:
            raise ValueError(f"Unsupported review decision for {unit}")
        evidence_ids = review.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or not all(
            isinstance(item, str) and item for item in evidence_ids
        ):
            raise ValueError(f"Review evidence is missing for {unit}")
        dimensions = review.get("dimensions")
        if not isinstance(dimensions, dict):
            raise ValueError(f"Review dimensions are missing for {unit}")
        if decision == "accepted":
            missing = REQUIRED_REVIEW_DIMENSIONS.difference(dimensions)
            if missing:
                raise ValueError(
                    f"Accepted review lacks dimensions for {unit}: {', '.join(sorted(missing))}"
                )
            invalid = {
                name
                for name in REQUIRED_REVIEW_DIMENSIONS
                if dimensions[name] not in {"pass", "not_applicable"}
            }
            if invalid:
                raise ValueError(
                    f"Accepted review has non-passing dimensions for {unit}: "
                    f"{', '.join(sorted(invalid))}"
                )
            if unit not in evidence_ids:
                raise ValueError(f"Accepted review does not cite its unit: {unit}")
        if release_evidence is not None and release_evidence not in evidence_ids:
            raise ValueError(f"Review is not bound to the release manifest: {unit}")
        by_unit[unit] = review
        review_ids.add(review_id)
    return declared_digest, manifest, by_unit


def _candidate_partition(
    release_dir: Path,
    release_manifest: dict[str, Any],
    dataset: str,
) -> tuple[Path, dict[str, Any]]:
    datasets = release_manifest.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get(dataset), dict):
        raise ValueError(f"Release manifest has no dataset: {dataset}")
    partitions = datasets[dataset].get("partitions")
    if not isinstance(partitions, dict) or not isinstance(partitions.get("candidate"), dict):
        raise ValueError(f"Release manifest has no candidate partition: {dataset}")
    entry = partitions["candidate"]
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Release manifest has no candidate path: {dataset}")
    path = Path(raw_path)
    path = (path if path.is_absolute() else release_dir / path).resolve()
    if release_dir not in path.parents:
        raise ValueError(f"Candidate partition escapes release directory: {dataset}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("records"), int):
        raise ValueError(f"Candidate partition lacks digest/count: {dataset}")
    if not isinstance(entry.get("bytes"), int):
        raise ValueError(f"Candidate partition lacks byte count: {dataset}")
    return path, entry


def _parent_record_id(record: dict[str, Any]) -> str | None:
    for container_name in ("lineage", "metadata", "provenance"):
        container = record.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in ("parent_record_sha256", "_chunk_parent_record_sha256", "parent_unit_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def materialize_reviewed_pilot(
    release_dir: Path,
    review_manifest_path: Path,
    output_dir: Path,
    *,
    datasets: Iterable[str] = ("sft", "tool_traces"),
) -> dict[str, Any]:
    """Materialize accepted rows from an immutable release into a pilot.

    Candidate rows are copied byte-for-byte.  The review manifest is emitted
    alongside them as evidence; it never changes a row's quality/privacy flags
    and never authorizes training.
    """

    release_dir = release_dir.resolve()
    review_manifest_path = review_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {output_dir}")
    selected = list(dict.fromkeys(datasets))
    unknown = set(selected) - {"sft", "tool_traces", "trajectories"}
    if unknown:
        raise ValueError(f"Unsupported reviewed-pilot dataset: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("At least one reviewed-pilot dataset is required")

    release_manifest_path = release_dir / "manifest.json"
    release_manifest_sha256 = sha256_file(release_manifest_path)
    release_manifest = load_json(release_manifest_path)
    review_manifest_sha256, review_manifest, reviews = load_review_manifest_artifact(
        review_manifest_path,
        expected_release_manifest_sha256=release_manifest_sha256,
    )
    accepted = {
        unit: review for unit, review in reviews.items() if review["decision"] == "accepted"
    }
    if not accepted:
        raise ValueError("Review manifest contains no accepted rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".staging"
    staging_dir.mkdir()
    writers: dict[str, WriterStats] = {}
    matched: dict[str, dict[str, Any]] = {}
    dataset_results: dict[str, Any] = {}
    try:
        for label in selected:
            input_path, expected = _candidate_partition(release_dir, release_manifest, label)
            writer = _new_writer(staging_dir, output_dir / f"{label}.jsonl")
            writers[label] = writer
            input_digest = hashlib.sha256()
            input_bytes = 0
            input_records = 0
            with input_path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    input_digest.update(raw_line)
                    input_bytes += len(raw_line)
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid candidate JSONL in {input_path} at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"Candidate row is not an object at {input_path}:{line_number}")
                    input_records += 1
                    candidate_unit = unit_id(record)
                    if candidate_unit not in accepted:
                        continue
                    if candidate_unit in matched:
                        raise ValueError(f"Accepted review unit matched more than once: {candidate_unit}")
                    parent_id = _parent_record_id(record)
                    if parent_id is None:
                        raise ValueError(f"Accepted row lacks parent identity: {candidate_unit}")
                    matched[candidate_unit] = {
                        "dataset": label,
                        "partition_line": line_number,
                        "parent_record_sha256": parent_id,
                    }
                    writer.write(raw_line)

            actual_input_sha256 = input_digest.hexdigest()
            if actual_input_sha256 != expected["sha256"]:
                raise ValueError(
                    f"Candidate partition hash mismatch for {label}: "
                    f"expected {expected['sha256']}, got {actual_input_sha256}"
                )
            if input_bytes != expected["bytes"] or input_records != expected["records"]:
                raise ValueError(f"Candidate partition size/count mismatch for {label}")
            writer.close()
            dataset_results[label] = {
                "path": writer.final_path.name,
                "source_partition": input_path.name,
                "source_partition_sha256": actual_input_sha256,
                "source_partition_records": input_records,
                "records": writer.records,
                "bytes": writer.bytes_written,
                "sha256": writer.sha256(),
                "accepted_units": sorted(
                    unit for unit, match in matched.items() if match["dataset"] == label
                ),
            }

        missing = sorted(set(accepted) - set(matched))
        if missing:
            raise ValueError(
                "Accepted review units were not found in the selected candidate partitions: "
                + ", ".join(missing)
            )
        parent_ids = [match["parent_record_sha256"] for match in matched.values()]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("Accepted pilot rows are not parent-disjoint")

        reviews_writer = _new_writer(staging_dir, output_dir / "reviews.jsonl")
        for unit in sorted(reviews):
            _write_decision(reviews_writer, reviews[unit])
        reviews_writer.close()
        review_result = {
            "path": reviews_writer.final_path.name,
            "records": reviews_writer.records,
            "bytes": reviews_writer.bytes_written,
            "sha256": reviews_writer.sha256(),
        }

        for writer in (*writers.values(), reviews_writer):
            os.replace(writer.temporary_path, output_dir / writer.final_path.name)

        privacy = release_manifest.get("privacy")
        source_training_authorized = (
            privacy.get("training_authorized") is True if isinstance(privacy, dict) else False
        )
        pilot_manifest = {
            "schema_version": REVIEWED_PILOT_SCHEMA,
            "generated_at": utc_now(),
            "source_release": {
                "manifest_path": release_manifest_path.name,
                "manifest_sha256": release_manifest_sha256,
                "source_training_authorized": source_training_authorized,
            },
            "review_manifest": {
                "path": review_manifest_path.name,
                "sha256": review_manifest_sha256,
                "manifest_revision": review_manifest["manifest_revision"],
                "snapshot_revision": review_manifest["snapshot_revision"],
                "rubric_revision": review_manifest["rubric_revision"],
            },
            "policy": {
                "selection": "accepted_review_decisions_only",
                "row_content": "copied_byte_for_byte_from_release_candidate_partitions",
                "review_evidence": "all_review_decisions_preserved_in_reviews.jsonl",
            },
            "counts": {
                "review_decisions": len(reviews),
                "accepted_reviews": len(accepted),
                "quarantined_reviews": sum(
                    review["decision"] == "quarantined" for review in reviews.values()
                ),
                "rejected_reviews": sum(
                    review["decision"] == "rejected" for review in reviews.values()
                ),
                "pilot_rows": len(matched),
                "parent_count": len(set(parent_ids)),
            },
            "privacy": {
                "training_authorized": False,
                "blocked_reasons": [
                    "privacy_approval_missing",
                    "trainer_certification_not_implemented",
                ],
            },
            "parent_disjoint": len(parent_ids) == len(set(parent_ids)),
            "datasets": dataset_results,
            "reviews": review_result,
        }
        manifest_target = output_dir / "manifest.json"
        manifest_temp = output_dir / ".manifest.json.tmp"
        manifest_temp.write_text(
            json.dumps(pilot_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_temp, manifest_target)
        staging_dir.rmdir()
        pilot_manifest["manifest_sha256"] = sha256_file(manifest_target)
        return pilot_manifest
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except OSError:
                pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_FILES),
        default=list(DATASET_FILES),
    )
    parser.add_argument(
        "--model-tiers",
        nargs="+",
        help="Optional model tiers allowed in the candidate partition",
    )
    parser.add_argument(
        "--duplicate-cap",
        type=int,
        default=1,
        help="Maximum first-seen RL prompt rows per prompt group before review",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = gate_corpus(
            args.input_dir,
            args.output_dir,
            manifest_path=args.manifest,
            datasets=args.datasets,
            allowed_model_tiers=set(args.model_tiers) if args.model_tiers else None,
            duplicate_cap=args.duplicate_cap,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        "Wrote release partitions: "
        f"{manifest['counts']['output_records']:,} rows; "
        f"training_authorized={manifest['privacy']['training_authorized']}"
    )
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
