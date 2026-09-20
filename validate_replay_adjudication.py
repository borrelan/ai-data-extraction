#!/usr/bin/env python3
"""Validate a replay/adjudication packet with the JSON dataset loader."""

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

from build_replay_adjudication import DECISION_SCHEMA, PACKET_SCHEMA, TASK_SCHEMA


VALIDATION_SCHEMA = "ai-data-extraction/replay-adjudication-loader-validation/v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def _verify_source(manifest: dict[str, Any]) -> None:
    source = manifest.get("source")
    registry = manifest.get("registry")
    if not isinstance(source, dict) or not isinstance(registry, dict):
        raise ValueError("packet source/registry binding missing")
    queue_dir = Path(str(source.get("queue_dir")))
    queue_manifest = queue_dir / "manifest.json"
    if not queue_manifest.is_file() or _sha256_file(queue_manifest) != source.get("queue_manifest_sha256"):
        raise ValueError("queue manifest changed")
    registry_path = Path(str(registry.get("path")))
    if not registry_path.is_file() or _sha256_file(registry_path) != registry.get("sha256"):
        raise ValueError("registry artifact changed")


def validate_packet(packet_dir: Path, *, report_path: Path | None = None) -> dict[str, Any]:
    packet_dir = packet_dir.resolve()
    manifest_path = packet_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("unexpected replay packet schema")
    if manifest.get("training_authorized") is not False:
        raise ValueError("replay packet is training-authorized")
    before_manifest_sha = _sha256_file(manifest_path)
    _verify_source(manifest)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("packet has no file descriptors")
    for filename, descriptor in files.items():
        path = packet_dir / filename
        if not path.is_file() or _sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"packet file digest mismatch: {filename}")

    expected = int(manifest.get("counts", {}).get("packet_tasks", -1))
    dataset = load_dataset("json", data_files=str(packet_dir / "replay_tasks.jsonl"), split="train")
    if len(dataset) != expected:
        raise ValueError("packet task count mismatch")
    expected_columns = {
        "action_count",
        "action_names_json",
        "agent",
        "event_pairs_json",
        "execution_status",
        "lineage_json",
        "messages_json",
        "model_tier",
        "observation_count",
        "parent_record_sha256",
        "privacy_eligible_for_training",
        "privacy_state",
        "provider",
        "quality_reason_json",
        "quality_tier",
        "queue_id",
        "registry_candidate_names_json",
        "registry_name_matches_json",
        "registry_status",
        "registry_artifact_sha256",
        "registry_revision",
        "replay_blockers_json",
        "replay_selection_rank",
        "replay_status",
        "reward_status",
        "schema_binding_status",
        "schema_status",
        "schema_version",
        "segment_record_sha256",
        "source_example_id",
        "source_metadata_json",
        "source_pilot_file",
        "source_pilot_line",
        "source_pilot_manifest_sha256",
        "source_pilot_row_sha256",
        "split",
        "tags_json",
        "tool_families_json",
        "training_authorized",
        "verification_status",
        "verifier_status",
        "workspace_binding_status",
        "workspace_signals_json",
    }
    if set(dataset.column_names) != expected_columns:
        raise ValueError(f"packet columns mismatch: {sorted(dataset.column_names)}")
    ids: set[str] = set()
    parents: set[str] = set()
    action_events = 0
    observation_events = 0
    for row in dataset:
        queue_id = row.get("queue_id")
        if not isinstance(queue_id, str) or not queue_id or queue_id in ids:
            raise ValueError("packet queue identity missing or duplicated")
        ids.add(queue_id)
        parent = row.get("parent_record_sha256")
        if not isinstance(parent, str) or not parent or parent in parents:
            raise ValueError("packet parent identity missing or duplicated")
        parents.add(parent)
        if row.get("schema_version") != TASK_SCHEMA:
            raise ValueError("packet task schema mismatch")
        if row.get("schema_binding_status") not in {
            "blocked_no_exact_registry_match",
            "blocked_call_identity_unbound",
        }:
            raise ValueError("packet contains a schema promotion")
        if row.get("workspace_binding_status") != "unbound":
            raise ValueError("packet contains a workspace binding")
        if row.get("verifier_status") != "not_provided":
            raise ValueError("packet contains a verifier claim")
        if row.get("execution_status") != "not_run":
            raise ValueError("packet contains execution")
        if row.get("replay_status") != "blocked":
            raise ValueError("packet contains replay readiness")
        if row.get("training_authorized") is not False or row.get("reward_status") != "not_exported":
            raise ValueError("packet contains training/reward authorization")
        pairs = _json_text(row.get("event_pairs_json"), "event_pairs_json")
        if not isinstance(pairs, list) or len(pairs) != int(row.get("action_count", -1)):
            raise ValueError("packet event pairs do not match action count")
        if int(row.get("observation_count", -1)) != sum(
            1 for pair in pairs if isinstance(pair, dict) and pair.get("observation_present") is True
        ):
            raise ValueError("packet observations do not match count")
        action_events += int(row["action_count"])
        observation_events += int(row["observation_count"])
        for field in (
            "messages_json",
            "action_names_json",
            "quality_reason_json",
            "tool_families_json",
            "tags_json",
            "registry_candidate_names_json",
            "registry_name_matches_json",
            "replay_blockers_json",
            "source_metadata_json",
            "lineage_json",
            "workspace_signals_json",
        ):
            _json_text(row.get(field), field)
        blob = json.dumps(row, ensure_ascii=False).lower()
        if any(marker in blob for marker in ("<think>", "<analysis>", "chain_of_thought", "hidden_reasoning")):
            raise ValueError("packet contains a reasoning marker")

    decision_count = 0
    decision_ids: set[str] = set()
    with (packet_dir / "decisions.jsonl").open(encoding="utf-8") as decisions:
        for line in decisions:
            if not line.strip():
                continue
            decision = json.loads(line)
            decision_count += 1
            if decision.get("schema_version") != DECISION_SCHEMA:
                raise ValueError("decision schema mismatch")
            task_id = decision.get("packet_task_id")
            if not isinstance(task_id, str) or task_id in decision_ids or task_id not in ids:
                raise ValueError("decision task identity missing, duplicated, or unknown")
            decision_ids.add(task_id)
            if decision.get("decision") != "blocked" or not isinstance(decision.get("reasons"), list) or not decision["reasons"]:
                raise ValueError("blocked decision lacks reasons")
            if decision.get("training_authorized") is not False:
                raise ValueError("decision contains authorization")
    if decision_count != int(manifest.get("counts", {}).get("decision_records", -1)):
        raise ValueError("decision count mismatch")
    if decision_ids != ids:
        raise ValueError("decisions do not cover packet tasks")

    report = {
        "schema_version": VALIDATION_SCHEMA,
        "packet_manifest_sha256_before": before_manifest_sha,
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "loaded": {
            "rows": len(dataset),
            "columns": sorted(dataset.column_names),
            "unique_parents": len(parents),
            "action_events": action_events,
            "observation_events": observation_events,
        },
        "decision_records": decision_count,
        "parent_unique": True,
        "schema_inference": "none",
        "verifier_inference": "none",
        "execution": "not_run",
        "reward_export": "none",
        "training_authorized": False,
        "result": "passed",
    }
    if report_path is None:
        report_path = packet_dir / "loader_validation.json"
    report_path = report_path.resolve()
    _atomic_write(report_path, _canonical_bytes(report) + b"\n")
    report_sha = _sha256_file(report_path)
    validation = manifest.setdefault("validation", {})
    validation.update(
        {
            "input_binding": "passed",
            "loader": "passed",
            "parent_unique": "passed",
            "schema_inference": "none",
            "verifier_inference": "none",
            "execution": "not_run",
            "reward": "not_present",
        }
    )
    validation["loader_validation"] = {
        "status": "passed",
        "loader": "datasets.load_dataset",
        "datasets_version": datasets_version,
        "report": report_path.name,
        "report_sha256": report_sha,
        "rows": len(dataset),
        "decision_records": decision_count,
    }
    _atomic_write(manifest_path, _canonical_bytes(manifest) + b"\n")
    report["report_sha256"] = report_sha
    report["packet_manifest_sha256_after"] = _sha256_file(manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_packet(args.packet_dir, report_path=args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VALIDATION_SCHEMA", "validate_packet"]
