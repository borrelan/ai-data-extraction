#!/usr/bin/env python3
"""Materialize a machine-readable replay/adjudication packet.

This packet is the handoff between historical trajectory salvage and a real
execution harness. It preserves the selected task payloads and records why a
task is or is not executable. It never infers schemas, verifiers, rewards, or
success from historical text.
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
from typing import Any, TextIO


PACKET_SCHEMA = "ai-data-extraction/replay-adjudication-packet/v1"
TASK_SCHEMA = "ai-data-extraction/replay-adjudication-task/v1"
DECISION_SCHEMA = "ai-data-extraction/replay-adjudication-decision/v1"
QUEUE_SCHEMA = "ai-data-extraction/tool-schema-enrichment-queue/v1"


class ReplayPacketError(ValueError):
    """Raised when the packet cannot be bound to its source evidence."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayPacketError(f"{field} must be non-empty")
    return value.strip()


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
        return {"records": self.records, "bytes": self.bytes, "sha256": self.digest.hexdigest()}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayPacketError(f"expected object: {path}")
    return value


def _load_registry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    registry = _load_json(path)
    definitions = registry.get("definitions")
    if not isinstance(definitions, list):
        raise ReplayPacketError("registry has no definitions")
    names = sorted(
        {
            definition.get("name")
            for definition in definitions
            if isinstance(definition, dict) and isinstance(definition.get("name"), str)
        }
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "revision": str(registry.get("registry_revision") or ""),
        "source_revision": str(registry.get("source_revision") or ""),
        "scope": str(registry.get("scope") or ""),
        "names": names,
    }


def _load_queue(queue_dir: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    manifest_path = queue_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest.get("schema_version") != QUEUE_SCHEMA:
        raise ReplayPacketError("unexpected queue schema")
    if manifest.get("training_authorized") is not False:
        raise ReplayPacketError("queue is training-authorized")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ReplayPacketError("queue has no file descriptors")
    for filename, descriptor in files.items():
        path = queue_dir / filename
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise ReplayPacketError(f"queue file changed: {filename}")
    tasks: list[dict[str, Any]] = []
    with (queue_dir / "replay_tasks.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ReplayPacketError("queue replay row is not an object")
                tasks.append(row)
    expected = int(manifest.get("counts", {}).get("replay_records", -1))
    if len(tasks) != expected:
        raise ReplayPacketError(f"queue replay count mismatch: {len(tasks)} != {expected}")
    return manifest, manifest_sha, tasks


def _action_names(task: dict[str, Any]) -> list[str]:
    raw = task.get("action_names_json")
    if not isinstance(raw, str):
        raise ReplayPacketError("task action names are not JSON text")
    names = json.loads(raw)
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ReplayPacketError("task action names are invalid")
    return names


def _event_pairs(task: dict[str, Any]) -> list[dict[str, Any]]:
    raw = task.get("event_pairs_json")
    if not isinstance(raw, str):
        raise ReplayPacketError("task event pairs are not JSON text")
    pairs = json.loads(raw)
    if not isinstance(pairs, list) or not all(isinstance(pair, dict) for pair in pairs):
        raise ReplayPacketError("task event pairs are invalid")
    return pairs


def _workspace_signals(pairs: list[dict[str, Any]]) -> list[str]:
    signals: set[str] = set()
    for pair in pairs:
        raw_input = pair.get("action_input_json")
        if not isinstance(raw_input, str):
            continue
        try:
            payload = json.loads(raw_input)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            workdir = payload.get("workdir")
            if isinstance(workdir, str) and workdir:
                signals.add(workdir)
            path = payload.get("path")
            if isinstance(path, str) and path:
                signals.add(path)
    return sorted(signals)


def _task_row(task: dict[str, Any], registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_id = _required(task.get("queue_id"), "queue_id")
    parent = _required(task.get("parent_record_sha256"), "parent_record_sha256")
    names = _action_names(task)
    pairs = _event_pairs(task)
    if len(pairs) != int(task.get("action_count", -1)):
        raise ReplayPacketError(f"action count mismatch: {queue_id}")
    if int(task.get("observation_count", -1)) != sum(
        1 for pair in pairs if pair.get("observation_present") is True
    ):
        raise ReplayPacketError(f"observation count mismatch: {queue_id}")
    name_matches = sorted(set(names) & set(registry["names"]))
    workspaces = _workspace_signals(pairs)
    reasons: list[str] = []
    if not name_matches:
        reasons.append("registry_name_unmatched")
    else:
        reasons.append("registry_call_identity_unbound")
    reasons.append("workspace_snapshot_unbound")
    reasons.append("verifier_not_provided")
    reasons.append("privacy_review_required")
    row = dict(task)
    row.update(
        {
            "schema_version": TASK_SCHEMA,
            "registry_artifact_sha256": registry["sha256"],
            "registry_revision": registry["revision"],
            "registry_name_matches_json": json.dumps(name_matches, separators=(",", ":")),
            "schema_binding_status": "blocked_no_exact_registry_match"
            if not name_matches
            else "blocked_call_identity_unbound",
            "workspace_signals_json": json.dumps(workspaces, ensure_ascii=False, separators=(",", ":")),
            "workspace_binding_status": "unbound",
            "verifier_status": "not_provided",
            "execution_status": "not_run",
            "replay_status": "blocked",
            "replay_blockers_json": json.dumps(sorted(set(reasons)), separators=(",", ":")),
            "training_authorized": False,
            "reward_status": "not_exported",
        }
    )
    decision = {
        "schema_version": DECISION_SCHEMA,
        "packet_task_id": queue_id,
        "source_example_id": task.get("source_example_id"),
        "parent_record_sha256": parent,
        "provider": task.get("provider"),
        "model_tier": task.get("model_tier"),
        "quality_tier": task.get("quality_tier"),
        "registry_revision": registry["revision"],
        "registry_name_match_count": len(name_matches),
        "schema_binding_status": row["schema_binding_status"],
        "workspace_binding_status": "unbound",
        "verifier_status": "not_provided",
        "execution_status": "not_run",
        "decision": "blocked",
        "reasons": sorted(set(reasons)),
        "training_authorized": False,
        "reward_status": "not_exported",
    }
    return row, decision


def build_replay_adjudication(
    *, queue_dir: Path, registry_path: Path, output_dir: Path
) -> dict[str, Any]:
    queue_dir = queue_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    queue_manifest, queue_manifest_sha, tasks = _load_queue(queue_dir)
    registry = _load_registry(registry_path)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    decisions: list[dict[str, Any]] = []
    parents: set[str] = set()
    match_counts: Counter[str] = Counter()
    try:
        with JsonlWriter(staging / "replay_tasks.jsonl") as task_file:
            for task in tasks:
                row, decision = _task_row(task, registry)
                parent = _required(row.get("parent_record_sha256"), "parent_record_sha256")
                if parent in parents:
                    raise ReplayPacketError(f"replay packet is not parent-diverse: {parent}")
                parents.add(parent)
                task_file.write(row)
                decisions.append(decision)
                match_counts[decision["schema_binding_status"]] += 1
        with JsonlWriter(staging / "decisions.jsonl") as decision_file:
            for decision in decisions:
                decision_file.write(decision)
        files = {}
        for filename, records in (
            ("replay_tasks.jsonl", len(tasks)),
            ("decisions.jsonl", len(decisions)),
        ):
            path = staging / filename
            files[filename] = {
                "records": records,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": PACKET_SCHEMA,
            "status": "review_only",
            "trainer_loadable": True,
            "training_authorized": False,
            "format": "jsonl_with_preserved_json_text_payloads",
            "source": {
                "queue_dir": str(queue_dir),
                "queue_manifest_sha256": queue_manifest_sha,
                "queue_schema_version": queue_manifest.get("schema_version"),
                "queue_replay_records": len(tasks),
            },
            "registry": {
                "path": registry["path"],
                "sha256": registry["sha256"],
                "revision": registry["revision"],
                "source_revision": registry["source_revision"],
                "scope": registry["scope"],
                "definition_count": len(registry["names"]),
                "binding_policy": "exact call identity plus temporal registry revision; name-only match never binds",
            },
            "counts": {
                "input_tasks": len(tasks),
                "packet_tasks": len(tasks),
                "decision_records": len(decisions),
                "unique_parent_sessions": len(parents),
                "exact_schema_bindings": 0,
                "workspace_bound": 0,
                "verifier_bound": 0,
                "executed": 0,
                "blocked": len(decisions),
            },
            "decision_counts": dict(sorted(match_counts.items())),
            "gates": {
                "schema": "blocked_until_exact_registry_join",
                "workspace": "blocked_until_snapshot_binding",
                "verifier": "blocked_until_executable_contract",
                "execution": "not_run",
                "reward": "not_exported",
            },
            "validation": {"input_binding": "passed", "loader": "pending"},
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        staging.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("registry_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_replay_adjudication(queue_dir=args.queue_dir, registry_path=args.registry_path, output_dir=args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PACKET_SCHEMA", "ReplayPacketError", "build_replay_adjudication"]
