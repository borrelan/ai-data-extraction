#!/usr/bin/env python3
"""Build a bounded, replay-ready queue from historical tool trajectories.

The queue is an adjudication artifact, not SFT or RL data. It preserves the
visible prompt/tool/observation payload as JSON text, binds every row to the
already validated historical pilot, and refuses to infer a tool schema,
verifier, reward, or training authorization. A small parent-diverse subset is
emitted for the next bounded replay/adjudication pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO


QUEUE_SCHEMA = "ai-data-extraction/tool-schema-enrichment-queue/v1"
DECISION_SCHEMA = "ai-data-extraction/tool-schema-enrichment-decision/v1"
TASK_SCHEMA = "ai-data-extraction/tool-replay-adjudication-task/v1"
PILOT_SCHEMA = "ai-data-extraction/historical-tool-trajectory-pilot/v1"
EVENT_SCHEMA = "ai-data-extraction/event/v1"


class QueueError(ValueError):
    """Raised when the queue cannot be bound to its pilot evidence."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_text(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    raw = canonical_json(value) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return {"records": 1, "bytes": len(raw), "sha256": sha256_bytes(raw)}


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


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise QueueError(f"{field} must be a non-empty string")
    return value


def _safe_child(root: Path, relative: str, field: str) -> Path:
    candidate = (root / _required_string(relative, field)).resolve()
    if root.resolve() not in candidate.parents:
        raise QueueError(f"{field} escapes pilot directory")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_pilot(pilot_dir: Path) -> tuple[dict[str, Any], str, list[tuple[str, Path, dict[str, Any]]]]:
    manifest_path = pilot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PILOT_SCHEMA:
        raise QueueError("pilot is not the historical trajectory pilot schema")
    if manifest.get("training_authorized") is not False:
        raise QueueError("refusing a pilot with training_authorized != false")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise QueueError("pilot manifest has no file descriptors")
    inputs: list[tuple[str, Path, dict[str, Any]]] = []
    for split in ("train", "validation"):
        descriptor = files.get(f"{split}.jsonl")
        if not isinstance(descriptor, dict):
            raise QueueError(f"pilot has no {split}.jsonl descriptor")
        path = _safe_child(pilot_dir, f"{split}.jsonl", f"{split}.path")
        if sha256_file(path) != descriptor.get("sha256"):
            raise QueueError(f"pilot digest mismatch: {split}.jsonl")
        if not isinstance(descriptor.get("records"), int):
            raise QueueError(f"pilot record count missing: {split}.jsonl")
        inputs.append((split, path, descriptor))
    return manifest, manifest_sha, inputs


def _load_registry(registry_path: Path | None) -> dict[str, Any]:
    if registry_path is None:
        return {
            "status": "not_provided",
            "path": "",
            "sha256": "",
            "revision": "",
            "source_revision": "",
            "names": [],
        }
    registry_path = registry_path.resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise QueueError("registry artifact must be an object")
    definitions = registry.get("definitions")
    if not isinstance(definitions, list):
        raise QueueError("registry artifact has no definitions list")
    names = sorted(
        {
            definition.get("name")
            for definition in definitions
            if isinstance(definition, dict) and isinstance(definition.get("name"), str)
        }
    )
    return {
        "status": "provided",
        "path": str(registry_path),
        "sha256": sha256_file(registry_path),
        "revision": str(registry.get("registry_revision") or ""),
        "source_revision": str(registry.get("source_revision") or ""),
        "scope": str(registry.get("scope") or ""),
        "names": names,
    }


def _event_pairs(events_value: Any) -> tuple[list[dict[str, Any]], list[str], int, int]:
    if not isinstance(events_value, list):
        raise QueueError("pilot row events must be a list")
    events: list[dict[str, Any]] = []
    for event_index, raw_event in enumerate(events_value):
        if isinstance(raw_event, str):
            event = json.loads(raw_event)
        else:
            event = raw_event
        if not isinstance(event, dict):
            raise QueueError(f"event {event_index} is not an object")
        if event.get("schema_version") != EVENT_SCHEMA:
            raise QueueError(f"event {event_index} has an unexpected schema")
        events.append(dict(event))

    actions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    observations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    action_names: list[str] = []
    action_count = 0
    observation_count = 0
    for event in events:
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        if event.get("kind") == "action":
            actions[call_id].append(event)
            action_names.append(str(event.get("name") or ""))
            action_count += 1
        elif event.get("kind") == "observation":
            observations[call_id].append(event)
            observation_count += 1

    pairs: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "action":
            continue
        call_id = str(event.get("call_id") or "")
        action = actions[call_id].pop(0)
        observation = observations[call_id].pop(0) if observations[call_id] else None
        pairs.append(
            {
                "call_id": call_id,
                "action_name": str(action.get("name") or ""),
                "action_event_id": str(action.get("event_id") or ""),
                "observation_event_id": (
                    str(observation.get("event_id") or "") if observation else ""
                ),
                "action_ordinal": action.get("ordinal"),
                "observation_ordinal": observation.get("ordinal") if observation else None,
                "action_message_index": action.get("message_index"),
                "observation_message_index": (
                    observation.get("message_index") if observation else None
                ),
                "action_input_json": json_text(action.get("input")),
                "observation_output_json": (
                    json_text(observation.get("output")) if observation else ""
                ),
                "action_event_json": json_text(action),
                "observation_event_json": json_text(observation) if observation else "",
                "observation_present": observation is not None,
            }
        )
    return pairs, action_names, action_count, observation_count


def _queue_id(pilot_manifest_sha: str, source_row_sha: str, example_id: str) -> str:
    return "sha256:" + sha256_bytes(
        canonical_json(
            {
                "schema_version": QUEUE_SCHEMA,
                "pilot_manifest_sha256": pilot_manifest_sha,
                "source_row_sha256": source_row_sha,
                "example_id": example_id,
            }
        )
    )


def _replay_blockers(
    *,
    privacy: dict[str, Any],
    structural_ok: bool,
) -> list[str]:
    blockers: list[str] = []
    if not structural_ok:
        blockers.append("event_pairing_incomplete")
    blockers.extend(
        [
            "tool_schema_not_observed",
            "verifier_not_observed",
            "workspace_snapshot_unbound",
        ]
    )
    if privacy.get("eligible_for_training") is not True:
        blockers.append("privacy_review_required")
    return sorted(set(blockers))


def _queue_row(
    *,
    row: dict[str, Any],
    raw_line: bytes,
    pilot_manifest_sha: str,
    split: str,
    source_line: int,
    registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    example_id = _required_string(row.get("example_id"), "pilot.example_id")
    source_row_sha = sha256_bytes(raw_line)
    queue_id = _queue_id(pilot_manifest_sha, source_row_sha, example_id)
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    privacy = row.get("privacy") if isinstance(row.get("privacy"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    pairs, action_names, action_count, observation_count = _event_pairs(row.get("events"))
    structural_ok = action_count == observation_count and all(
        pair["observation_present"] for pair in pairs
    )
    action_name_set = sorted(set(action_names))
    registry_candidates = sorted(set(action_name_set) & set(registry["names"]))
    if registry["status"] == "not_provided":
        registry_status = "not_provided"
    elif registry_candidates:
        registry_status = "name_candidate_unbound"
    else:
        registry_status = "no_name_candidate"
    blockers = _replay_blockers(privacy=privacy, structural_ok=structural_ok)
    source_metadata = {
        "provider": metadata.get("provider"),
        "agent": metadata.get("agent") or metadata.get("source_label"),
        "source_file_name": metadata.get("source_file_name"),
        "source_file_sha256": metadata.get("source_file_sha256"),
        "source_line": metadata.get("source_line"),
        "source_record_sha256": metadata.get("source_record_sha256"),
        "segment_record_sha256": metadata.get("segment_record_sha256"),
        "parser_revision": metadata.get("parser_revision"),
    }
    row_value = {
        "schema_version": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "source_pilot_manifest_sha256": pilot_manifest_sha,
        "source_pilot_file": f"{split}.jsonl",
        "source_pilot_line": source_line,
        "source_pilot_row_sha256": source_row_sha,
        "source_example_id": example_id,
        "parent_record_sha256": str(lineage.get("parent_record_sha256") or ""),
        "segment_record_sha256": str(lineage.get("segment_record_sha256") or ""),
        "split": split,
        "provider": str(row.get("provider") or metadata.get("provider") or ""),
        "agent": str(row.get("agent") or metadata.get("agent") or ""),
        "model_tier": str(row.get("model_tier") or ""),
        "quality_tier": str(row.get("quality_tier") or "candidate"),
        "quality_reason_json": json_text(row.get("quality_reason") or []),
        "tool_families_json": json_text(row.get("tool_families") or []),
        "tags_json": json_text(row.get("tags") or []),
        "privacy_state": str(privacy.get("state") or "review_required"),
        "privacy_eligible_for_training": privacy.get("eligible_for_training") is True,
        "messages_json": json_text(row.get("messages") or []),
        "event_pairs_json": json_text(pairs),
        "action_names_json": json_text(action_name_set),
        "action_count": action_count,
        "observation_count": observation_count,
        "registry_status": registry_status,
        "registry_candidate_names_json": json_text(registry_candidates),
        "schema_status": "not_observed",
        "verification_status": "not_observed",
        "reward_status": "not_exported",
        "replay_status": "not_run",
        "replay_blockers_json": json_text(blockers),
        "training_authorized": False,
        "source_metadata_json": json_text(source_metadata),
        "lineage_json": json_text(lineage),
    }
    ref = {
        "queue_id": queue_id,
        "parent_record_sha256": row_value["parent_record_sha256"],
        "provider": row_value["provider"],
        "tool_families": json.loads(row_value["tool_families_json"]),
        "action_names": action_name_set,
        "split": split,
    }
    return row_value, ref


def _select_replay_tasks(refs: list[dict[str, Any]], limit: int) -> dict[str, int]:
    if limit < 1:
        raise QueueError("replay limit must be positive")
    selected: dict[str, int] = {}
    selected_parents: set[str] = set()
    covered_names: set[str] = set()
    covered_families: set[str] = set()
    covered_providers: set[str] = set()
    while len(selected) < limit:
        candidates = [
            ref
            for ref in refs
            if ref["queue_id"] not in selected
            and ref["parent_record_sha256"] not in selected_parents
        ]
        if not candidates:
            break

        def score(ref: dict[str, Any]) -> tuple[int, str]:
            names = set(ref["action_names"])
            families = set(ref["tool_families"])
            value = (
                (1000 if ref["provider"] not in covered_providers else 0)
                + 100 * len(families - covered_families)
                + 10 * len(names - covered_names)
                + len(names)
            )
            return value, ref["queue_id"]

        chosen = max(candidates, key=score)
        selected[chosen["queue_id"]] = len(selected) + 1
        selected_parents.add(chosen["parent_record_sha256"])
        covered_names.update(chosen["action_names"])
        covered_families.update(chosen["tool_families"])
        covered_providers.add(chosen["provider"])
    return selected


def build_queue(
    *,
    pilot_dir: Path,
    output_dir: Path,
    registry_path: Path | None = None,
    replay_limit: int = 64,
) -> dict[str, Any]:
    pilot_dir = pilot_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    pilot_manifest, pilot_manifest_sha, inputs = _load_pilot(pilot_dir)
    registry = _load_registry(registry_path)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    refs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    provider_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    registry_status_counts: Counter[str] = Counter()
    parent_ids: set[str] = set()
    split_parents: dict[str, set[str]] = {"train": set(), "validation": set()}
    source_counts: Counter[str] = Counter()
    queue_rows = 0
    action_count = 0
    observation_count = 0
    try:
        with JsonlWriter(staging / "queue.jsonl") as queue_file:
            for split, path, descriptor in inputs:
                digest = hashlib.sha256()
                bytes_read = 0
                records = 0
                with path.open("rb") as source:
                    for source_line, raw_line in enumerate(source, 1):
                        digest.update(raw_line)
                        bytes_read += len(raw_line)
                        records += 1
                        try:
                            row = json.loads(raw_line)
                        except json.JSONDecodeError as exc:
                            raise QueueError(f"invalid pilot JSON: {split}:{source_line}") from exc
                        if not isinstance(row, dict):
                            raise QueueError(f"pilot row is not an object: {split}:{source_line}")
                        if row.get("split") != split:
                            raise QueueError(f"pilot split mismatch: {split}:{source_line}")
                        queue_row, ref = _queue_row(
                            row=row,
                            raw_line=raw_line,
                            pilot_manifest_sha=pilot_manifest_sha,
                            split=split,
                            source_line=source_line,
                            registry=registry,
                        )
                        queue_file.write(queue_row)
                        refs.append(ref)
                        queue_rows += 1
                        source_counts[split] += 1
                        parent = ref["parent_record_sha256"]
                        parent_ids.add(parent)
                        split_parents[split].add(parent)
                        provider_counts[ref["provider"]] += 1
                        family_counts.update(ref["tool_families"])
                        registry_status_counts[queue_row["registry_status"]] += 1
                        action_count += queue_row["action_count"]
                        observation_count += queue_row["observation_count"]
                        decisions.append(
                            {
                                "schema_version": DECISION_SCHEMA,
                                "queue_id": queue_row["queue_id"],
                                "source_example_id": queue_row["source_example_id"],
                                "parent_record_sha256": parent,
                                "source_pilot_file": queue_row["source_pilot_file"],
                                "source_pilot_line": source_line,
                                "structural_status": (
                                    "paired"
                                    if queue_row["action_count"] == queue_row["observation_count"]
                                    else "unpaired"
                                ),
                                "schema_status": queue_row["schema_status"],
                                "verification_status": queue_row["verification_status"],
                                "registry_status": queue_row["registry_status"],
                                "replay_status": queue_row["replay_status"],
                                "training_authorized": False,
                                "reasons": json.loads(queue_row["replay_blockers_json"]),
                            }
                        )
                if records != descriptor["records"] or bytes_read != descriptor["bytes"]:
                    raise QueueError(f"pilot descriptor count mismatch: {split}.jsonl")
                if digest.hexdigest() != descriptor["sha256"]:
                    raise QueueError(f"pilot descriptor SHA mismatch: {split}.jsonl")

        selected = _select_replay_tasks(refs, replay_limit)
        with JsonlWriter(staging / "replay_tasks.jsonl") as replay_file:
            with (staging / "queue.jsonl").open("rb") as queue_source:
                for raw_line in queue_source:
                    row = json.loads(raw_line)
                    rank = selected.get(row["queue_id"])
                    if rank is None:
                        continue
                    row["schema_version"] = TASK_SCHEMA
                    row["replay_selection_rank"] = rank
                    row["replay_status"] = "selected_for_adjudication"
                    replay_file.write(row)
        for decision in decisions:
            rank = selected.get(decision["queue_id"])
            decision["replay_selected"] = rank is not None
            decision["replay_selection_rank"] = rank
        with JsonlWriter(staging / "decisions.jsonl") as decisions_file:
            for decision in decisions:
                decisions_file.write(decision)

        queue_descriptor = {
            "records": queue_rows,
            "bytes": (staging / "queue.jsonl").stat().st_size,
            "sha256": sha256_file(staging / "queue.jsonl"),
        }
        replay_descriptor = {
            "records": len(selected),
            "bytes": (staging / "replay_tasks.jsonl").stat().st_size,
            "sha256": sha256_file(staging / "replay_tasks.jsonl"),
        }
        decision_descriptor = {
            "records": len(decisions),
            "bytes": (staging / "decisions.jsonl").stat().st_size,
            "sha256": sha256_file(staging / "decisions.jsonl"),
        }
        manifest = {
            "schema_version": QUEUE_SCHEMA,
            "status": "review_only",
            "loader_loadable": True,
            "training_authorized": False,
            "format": "jsonl_with_json_text_payloads",
            "source": {
                "pilot_dir": str(pilot_dir),
                "pilot_manifest_sha256": pilot_manifest_sha,
                "pilot_schema_version": pilot_manifest.get("schema_version"),
                "pilot_records": sum(item[2]["records"] for item in inputs),
                "pilot_files": {
                    split: {
                        "path": str(path),
                        "records": descriptor["records"],
                        "bytes": descriptor["bytes"],
                        "sha256": descriptor["sha256"],
                    }
                    for split, path, descriptor in inputs
                },
            },
            "registry_evidence": {
                "binding_policy": "exact source call identity plus temporally bound registry revision; name-only matches never bind",
                "artifact_path": registry["path"],
                "artifact_sha256": registry["sha256"],
                "registry_revision": registry["revision"],
                "source_revision": registry["source_revision"],
                "scope": registry.get("scope", ""),
                "definition_count": len(registry["names"]),
                "status_counts": dict(sorted(registry_status_counts.items())),
                "exact_bindings": 0,
            },
            "counts": {
                "input_records": queue_rows,
                "queue_records": queue_rows,
                "replay_records": len(selected),
                "decision_records": len(decisions),
                "unique_parent_sessions": len(parent_ids),
                "train_parent_sessions": len(split_parents["train"]),
                "validation_parent_sessions": len(split_parents["validation"]),
                "action_events": action_count,
                "observation_events": observation_count,
            },
            "quality": {
                "provider_counts": dict(sorted(provider_counts.items())),
                "tool_family_counts": dict(sorted(family_counts.items())),
                "quality_policy": "provider-neutral; preserve source tier and gate per row",
                "promotion_status": "blocked",
                "promotion_blockers": [
                    "tool_schema_not_observed",
                    "verifier_not_observed",
                    "workspace_snapshot_unbound",
                    "privacy_review_required",
                ],
            },
            "selection": {
                "method": "deterministic greedy coverage over action names and tool families, one row per parent",
                "replay_limit": replay_limit,
                "replay_parent_unique": True,
                "replay_manifest_status": "adjudication_only",
            },
            "validation": {
                "pilot_binding": "passed",
                "streaming_projection": "passed",
                "parent_disjoint": "inherited_from_pilot_pending_validator",
                "schema": "not_observed_not_inferred",
                "verifier": "not_observed_not_inferred",
                "reward": "not_exported",
                "loader": "pending",
            },
            "files": {
                "queue.jsonl": queue_descriptor,
                "replay_tasks.jsonl": replay_descriptor,
                "decisions.jsonl": decision_descriptor,
            },
        }
        atomic_json(staging / "manifest.json", manifest)
        staging.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--replay-limit", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            build_queue(
                pilot_dir=args.pilot_dir,
                output_dir=args.output_dir,
                registry_path=args.registry,
                replay_limit=args.replay_limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_SCHEMA",
    "QUEUE_SCHEMA",
    "TASK_SCHEMA",
    "QueueError",
    "build_queue",
]
