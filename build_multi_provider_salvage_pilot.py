#!/usr/bin/env python3
"""Build one bounded, multi-provider trainer/review salvage package.

The inputs are already-normalized provider partitions.  This adapter verifies
their source manifests and file digests, emits tool-free trainer JSONL for
explicitly eligible rows, and preserves structurally recoverable tool rows in
a separate review partition.  It never infers quality from provider name,
tool schemas from calls, or rewards from assistant text.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

from build_historical_tool_trajectory_pilot import (
    _event_summary,
    _loader_safe_events,
    _loader_safe_messages,
    _messages_valid,
)
from build_training_data import (
    MODEL_TIER_REGISTRY_REVISION,
    model_tier_for,
    no_reasoning_content,
    normalize_model_tier,
    trainer_marker_count,
)
from quality_rules import scan_record
from trainer_export import _has_tool_activity, _parent_id, _trainer_view


PACKAGE_SCHEMA = "ai-data-extraction/multi-provider-salvage-pilot/v1"
INPUT_SCHEMA = "ai-data-extraction/multi-provider-salvage-inputs/v1"
LINEAGE_SCHEMA = "ai-data-extraction/multi-provider-salvage-lineage/v1"
DECISION_SCHEMA = "ai-data-extraction/multi-provider-salvage-decision/v1"
TOOL_REVIEW_SCHEMA = "ai-data-extraction/multi-provider-tool-review/v1"
TRAINER_SCHEMA = "ai-data-extraction/trainer-example/v1"
SUPPORTED_TIERS = frozenset({"tier1_frontier", "tier2_open_source", "tier3_local"})
TIER_RANK = {
    "tier1_frontier": 0,
    "tier2_open_source": 1,
    "tier3_local": 2,
    "unclassified": 3,
}
QUALITY_GATE_ALIASES = {"review": "review_required"}


class SalvagePilotError(ValueError):
    """Raised when a source or output contract cannot be proven."""


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


def json_text(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def _required(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SalvagePilotError(f"{field} must be a non-empty string")
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
        return {
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SalvagePilotError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SalvagePilotError(f"expected JSON object: {path}")
    return value


def _manifest_value(manifest: dict[str, Any], keys: list[str]) -> Any:
    value: Any = manifest
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise SalvagePilotError(f"manifest key path missing: {'.'.join(keys)}")
        value = value[key]
    return value


def _load_source_bindings(spec_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec_path = spec_path.resolve()
    spec = _read_json(spec_path)
    if spec.get("schema_version") != INPUT_SCHEMA:
        raise SalvagePilotError("unexpected multi-provider input schema")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SalvagePilotError("input spec has no sources")
    seen: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for entry in sources:
        if not isinstance(entry, dict):
            raise SalvagePilotError("source binding is not an object")
        source_id = _required(entry.get("source_id"), "source_id")
        if source_id in seen:
            raise SalvagePilotError(f"duplicate source_id: {source_id}")
        seen.add(source_id)
        path = Path(_required(entry.get("path"), f"{source_id}.path")).resolve()
        manifest_path = Path(
            _required(entry.get("manifest_path"), f"{source_id}.manifest_path")
        ).resolve()
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else manifest_path)
        manifest_sha = sha256_file(manifest_path)
        expected_manifest_sha = _required(
            entry.get("manifest_sha256"), f"{source_id}.manifest_sha256"
        )
        if manifest_sha != expected_manifest_sha:
            raise SalvagePilotError(f"source manifest changed: {source_id}")
        manifest = _read_json(manifest_path)
        keys = entry.get("manifest_keys")
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
            raise SalvagePilotError(f"{source_id}.manifest_keys is invalid")
        descriptor = _manifest_value(manifest, keys)
        if not isinstance(descriptor, dict):
            raise SalvagePilotError(f"{source_id}.manifest_keys does not select a descriptor")
        expected_sha = _required(entry.get("sha256"), f"{source_id}.sha256")
        expected_records = entry.get("records")
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_records, int) or expected_records < 0:
            raise SalvagePilotError(f"{source_id}.records is invalid")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise SalvagePilotError(f"{source_id}.bytes is invalid")
        if descriptor.get("sha256") != expected_sha or descriptor.get("records") != expected_records:
            raise SalvagePilotError(f"source descriptor mismatch: {source_id}")
        if descriptor.get("bytes") is not None and descriptor.get("bytes") != expected_bytes:
            raise SalvagePilotError(f"source byte descriptor mismatch: {source_id}")
        descriptor_path = descriptor.get("path")
        if descriptor_path not in (None, "") and Path(str(descriptor_path)).name != path.name:
            raise SalvagePilotError(f"source path descriptor mismatch: {source_id}")
        bindings.append(
            {
                "source_id": source_id,
                "path": path,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_sha,
                "manifest_keys": list(keys),
                "schema_version": manifest.get("schema_version"),
                "sha256": expected_sha,
                "records": expected_records,
                "bytes": expected_bytes,
                "provider_scope": entry.get("provider_scope") or [],
            }
        )
    return spec, bindings


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("quality")
    return value if isinstance(value, dict) else {}


def _quality_assessment(row: dict[str, Any]) -> dict[str, Any]:
    value = _metadata(row).get("quality_assessment")
    return value if isinstance(value, dict) else {}


def _quality_gate(row: dict[str, Any]) -> str:
    assessment = _quality_assessment(row)
    quality = _quality(row)
    raw = (
        assessment.get("gate")
        or assessment.get("automatic_gate")
        or quality.get("session_quality_gate")
        or quality.get("status")
        or "unassessed"
    )
    value = str(raw).strip().lower()
    return QUALITY_GATE_ALIASES.get(value, value)


def _quality_flags(row: dict[str, Any]) -> list[str]:
    assessment = _quality_assessment(row)
    quality = _quality(row)
    values: list[Any] = []
    for container, key in ((assessment, "flags"), (quality, "session_quality_flags")):
        value = container.get(key)
        if isinstance(value, list):
            values.extend(value)
    return sorted({str(value) for value in values if value not in (None, "")})


def _training_lane(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    quality = _quality(row)
    value = metadata.get("training_lane") or quality.get("training_lane") or row.get("training_lane")
    return str(value) if value not in (None, "") else "unclassified"


def _provider(row: dict[str, Any], binding: dict[str, Any]) -> str:
    metadata = _metadata(row)
    value = metadata.get("provider") or metadata.get("source_label") or row.get("provider")
    if value in (None, "") and binding.get("provider_scope"):
        value = binding["provider_scope"][0]
    return str(value or "unknown")


def _agent(row: dict[str, Any], provider: str) -> str:
    metadata = _metadata(row)
    return str(metadata.get("agent") or metadata.get("source_label") or provider)


def _parent(row: dict[str, Any]) -> tuple[str | None, str]:
    value = _parent_id(row)
    if value:
        return value, "parent_record_sha256"
    metadata = _metadata(row)
    for key in ("session_id", "session_quality_id", "parent_unit_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value, key
    quality = _quality(row)
    value = quality.get("session_quality_id")
    if isinstance(value, str) and value:
        return value, "quality.session_quality_id"
    return None, "missing"


def _model_tokens(row: dict[str, Any]) -> list[str]:
    metadata = _metadata(row)
    quality = _quality(row)
    assessment = _quality_assessment(row)
    dimensions = assessment.get("dimensions")
    provenance = dimensions.get("model_provenance") if isinstance(dimensions, dict) else None
    values: list[Any] = []
    for container in (row, metadata, quality):
        if not isinstance(container, dict):
            continue
        for key in (
            "model",
            "model_id",
            "modelID",
            "model_provider",
            "providerID",
            "provider_id",
            "deployment_provider",
            "deployment",
            "runtime",
        ):
            value = container.get(key)
            if isinstance(value, str) and value:
                values.append(value)
            elif isinstance(value, dict):
                values.extend(
                    str(item) for item in value.values() if isinstance(item, (str, int, float))
                )
    if isinstance(provenance, dict):
        for key in ("models", "providers"):
            value = provenance.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value if item not in (None, ""))
            elif isinstance(value, dict):
                values.extend(str(item) for item in value.values())
    return sorted({str(value) for value in values if value not in (None, "")})


def _model_tier(row: dict[str, Any], provider: str) -> tuple[str, str, str]:
    metadata = _metadata(row)
    quality = _quality(row)
    explicit = metadata.get("model_tier") or quality.get("model_tier")
    normalized = normalize_model_tier(explicit)
    if normalized is not None:
        return normalized, "row_declared", "declared"
    results = []
    for token in _model_tokens(row):
        result = model_tier_for(
            {"source": provider, "model": token, "providerID": token}
        )
        results.append(result)
    classified = [result for result in results if result["tier"] in SUPPORTED_TIERS]
    tiers = {result["tier"] for result in classified}
    if len(tiers) == 1:
        tier = next(iter(tiers))
        bases = sorted({str(result["basis"]) for result in classified})
        confidence = "registry" if all(result["confidence"] == "registry" for result in classified) else "derived"
        return tier, ";".join(bases), confidence
    if len(tiers) > 1:
        return "unclassified", "conflicting_model_identities", "conflict"
    return "unclassified", "missing_authoritative_model_tier", "unknown"


def _split_for_parent(parent: str) -> str:
    bucket = int(sha256_bytes(parent.encode("utf-8"))[:8], 16) % 10
    return "validation" if bucket >= 8 else "train"


def _lineage_values(
    row: dict[str, Any],
    *,
    binding: dict[str, Any],
    source_line: int,
    row_sha256: str,
    parent: str | None,
    parent_basis: str,
) -> dict[str, Any]:
    metadata = _metadata(row)
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    source_origin = metadata.get("source_origin")
    if not isinstance(source_origin, dict):
        source_origin = {}
    source_file_sha = metadata.get("source_file_sha256") or binding["sha256"]
    source_file_name = metadata.get("source_file_name") or binding["path"].name
    source_record_sha = metadata.get("source_record_sha256") or row_sha256
    segment_sha = metadata.get("segment_record_sha256") or row_sha256
    source_range = lineage.get("source_message_range")
    if source_range is None:
        source_range = metadata.get("source_message_range")
    return {
        "source_file_sha256": str(source_file_sha),
        "source_file_name": str(source_file_name),
        # Keep the validated aggregate input binding separate from the
        # original session/file identity nested by the ingress adapter.  The
        # two hashes answer different provenance questions and must not be
        # collapsed into one field.
        "source_origin_file_sha256": str(source_origin.get("source_file_sha256") or ""),
        "source_origin_file_name": str(source_origin.get("source_file_name") or ""),
        "source_origin_json": json_text(source_origin),
        "source_line": int(metadata.get("source_line") or source_line),
        "source_record_sha256": str(source_record_sha),
        "segment_record_sha256": str(segment_sha),
        "source_message_range_json": json_text(source_range or {}),
        "parent_record_sha256": parent,
        "parent_identity_basis": parent_basis,
        "chunk_index": int(lineage.get("chunk_index") or metadata.get("chunk_index") or 0),
        "chunk_count": int(lineage.get("chunk_count") or metadata.get("chunk_count") or 1),
        "continuation_status": str(
            lineage.get("continuation_status")
            or metadata.get("continuation_status")
            or "unknown"
        ),
        "previous_example_id": str(lineage.get("previous_example_id") or ""),
        "next_example_id": str(lineage.get("next_example_id") or ""),
        "source_row_sha256": row_sha256,
        "source_example_id": row.get("example_id"),
        "source_dataset": row.get("dataset"),
        "source_binding_id": binding["source_id"],
    }


def _base_context(
    row: dict[str, Any],
    *,
    binding: dict[str, Any],
    source_line: int,
    row_sha256: str,
) -> dict[str, Any]:
    provider = _provider(row, binding)
    parent, parent_basis = _parent(row)
    tier, tier_basis, tier_confidence = _model_tier(row, provider)
    quality = _quality(row)
    return {
        "provider": provider,
        "agent": _agent(row, provider),
        "model_tier": tier,
        "model_tier_basis": tier_basis,
        "model_tier_confidence": tier_confidence,
        "model_tier_registry_revision": MODEL_TIER_REGISTRY_REVISION,
        "quality_gate": _quality_gate(row),
        "quality_flags": _quality_flags(row),
        "training_lane": _training_lane(row),
        "quality_id": str(
            quality.get("session_quality_id")
            or _quality_assessment(row).get("session_quality_id")
            or ""
        ),
        "parent": parent,
        "parent_basis": parent_basis,
        "lineage": _lineage_values(
            row,
            binding=binding,
            source_line=source_line,
            row_sha256=row_sha256,
            parent=parent,
            parent_basis=parent_basis,
        ),
    }


def _quality_reasons(context: dict[str, Any]) -> list[str]:
    return sorted(
        set(context["quality_flags"])
        | {f"quality_gate:{context['quality_gate']}"}
        | {"outcome_unverified", "not_human_adjudicated"}
    )


def _decision_base(
    row: dict[str, Any],
    *,
    binding: dict[str, Any],
    source_line: int,
    row_sha256: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DECISION_SCHEMA,
        "source_binding_id": binding["source_id"],
        "source_line": source_line,
        "source_row_sha256": row_sha256,
        "source_example_id": row.get("example_id"),
        "parent_record_sha256": context["parent"],
        "provider": context["provider"],
        "agent": context["agent"],
        "model_tier": context["model_tier"],
        "model_tier_basis": context["model_tier_basis"],
        "quality_gate": context["quality_gate"],
        "quality_flags_json": json_text(context["quality_flags"]),
        "training_lane": context["training_lane"],
        "sft_decision": "excluded",
        "sft_reasons": [],
        "tool_review_decision": "excluded",
        "tool_review_reasons": [],
        "rl_decision": "not_exported",
        "rl_reason": "executable_reward_not_observed",
    }


def _sft_candidate(
    row: dict[str, Any],
    *,
    context: dict[str, Any],
    source_id: str,
    source_line: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if row.get("dataset") != "sft":
        reasons.append("dataset_not_sft")
    if _has_tool_activity(row):
        reasons.append("tool_activity_not_sft")
    if context["quality_gate"] != "candidate":
        reasons.append("quality_gate_not_candidate")
    if context["model_tier"] not in SUPPORTED_TIERS:
        reasons.append("model_tier_unclassified")
    if context["parent"] is None:
        reasons.append("parent_identity_missing")
    if int((_metadata(row).get("structural_redactions") or 0)) > 0:
        reasons.append("privacy_redactions_present")
    privacy = row.get("privacy") if isinstance(row.get("privacy"), dict) else {}
    if int(privacy.get("structural_redactions") or 0) > 0:
        reasons.append("privacy_redactions_present")
    if not no_reasoning_content(row):
        reasons.append("reasoning_content_present")
    if trainer_marker_count(row):
        reasons.append("trainer_marker_present")
    findings = scan_record(row)
    if findings.has_hard_privacy_issue:
        reasons.append("hard_privacy_finding")
    if findings.has_marker:
        reasons.append("privacy_marker_finding")
    if reasons:
        return None, sorted(set(reasons))
    try:
        view = _trainer_view(row, source_dataset="sft")
    except ValueError as exc:
        return None, [f"trainer_contract:{exc}"]
    if not no_reasoning_content(view) or trainer_marker_count(view):
        return None, ["trainer_view_reasoning_or_marker"]
    view["split"] = "pending"
    return view, ["sft_candidate_eligible"]


def _tool_review_row(
    row: dict[str, Any],
    *,
    context: dict[str, Any],
    binding: dict[str, Any],
    source_line: int,
    row_sha256: str,
    split: str,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if row.get("dataset") != "sft":
        reasons.append("dataset_not_sft")
    if not _has_tool_activity(row):
        reasons.append("no_tool_activity")
    if context["parent"] is None:
        reasons.append("parent_identity_missing")
    messages_ok, message_reason = _messages_valid(row)
    if not messages_ok and message_reason:
        reasons.append(message_reason)
    event_summary, event_reasons = _event_summary(row)
    reasons.extend(event_reasons)
    if not no_reasoning_content(row):
        reasons.append("reasoning_content_present")
    if trainer_marker_count(row):
        reasons.append("trainer_marker_present")
    findings = scan_record(row)
    if findings.has_hard_privacy_issue:
        reasons.append("hard_privacy_finding")
    if findings.has_marker:
        reasons.append("privacy_marker_finding")
    if reasons:
        return None, sorted(set(reasons)), event_summary
    quality = _quality(row)
    privacy = row.get("privacy") if isinstance(row.get("privacy"), dict) else {}
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    families = quality.get("tool_families") if isinstance(quality.get("tool_families"), list) else []
    output = {
        "schema_version": TOOL_REVIEW_SCHEMA,
        "example_id": _required(row.get("example_id"), "example_id"),
        "split": split,
        "provider": context["provider"],
        "agent": context["agent"],
        "model_tier": context["model_tier"],
        "model_tier_basis": context["model_tier_basis"],
        "quality_tier": context["quality_gate"],
        "quality_gate": context["quality_gate"],
        "quality_reason_json": json_text(_quality_reasons(context) + ["tool_schema_not_observed"]),
        "privacy_state": "review_required",
        "privacy_structural_redactions": int(privacy.get("structural_redactions") or 0),
        "tool_schema_status": "not_observed",
        "verifier_status": "not_observed",
        "reward_status": "not_exported",
        "parent_record_sha256": context["parent"],
        "source_example_id": row.get("example_id"),
        "source_file_sha256": context["lineage"]["source_file_sha256"],
        "source_file_name": context["lineage"]["source_file_name"],
        "source_origin_file_sha256": context["lineage"]["source_origin_file_sha256"],
        "source_origin_file_name": context["lineage"]["source_origin_file_name"],
        "source_origin_json": context["lineage"]["source_origin_json"],
        "source_line": context["lineage"]["source_line"],
        "source_row_sha256": row_sha256,
        "source_message_range_json": context["lineage"]["source_message_range_json"],
        "chunk_index": context["lineage"]["chunk_index"],
        "chunk_count": context["lineage"]["chunk_count"],
        "continuation_status": context["lineage"]["continuation_status"],
        "previous_example_id": context["lineage"]["previous_example_id"],
        "next_example_id": context["lineage"]["next_example_id"],
        "tool_families_json": json_text(sorted({str(item) for item in families})),
        "tags_json": json_text(sorted({str(item) for item in tags} | {"trajectory:historical-review"})),
        "events_summary_json": json_text(event_summary),
        "messages": _loader_safe_messages(row["messages"]),
        "events": _loader_safe_events(row["events"]),
    }
    return output, ["tool_review_structurally_recoverable", "tool_schema_not_observed"], event_summary


def _lineage_row(
    *,
    kind: str,
    record: dict[str, Any],
    context: dict[str, Any],
    split: str,
    source_row_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": LINEAGE_SCHEMA,
        "record_kind": kind,
        "example_id": record.get("example_id"),
        "split": split,
        "parent_record_sha256": context["parent"],
        "provider": context["provider"],
        "agent": context["agent"],
        "model_tier": context["model_tier"],
        "model_tier_basis": context["model_tier_basis"],
        "model_tier_confidence": context["model_tier_confidence"],
        "quality_tier": record.get("quality_tier") or context["quality_gate"],
        "quality_gate": context["quality_gate"],
        "quality_reason_json": json_text(_quality_reasons(context)),
        "training_lane": context["training_lane"],
        "source_row_sha256": source_row_sha256,
        **context["lineage"],
    }


def build_multi_provider_salvage_pilot(*, input_spec: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_spec = input_spec.resolve()
    spec, bindings = _load_source_bindings(input_spec)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    writer_names = (
        "sft_tier1_train.jsonl",
        "sft_tier1_validation.jsonl",
        "sft_optional_train.jsonl",
        "sft_optional_validation.jsonl",
        "tool_review_train.jsonl",
        "tool_review_validation.jsonl",
        "lineage.jsonl",
        "decisions.jsonl",
    )
    writers = {name: JsonlWriter(staging / name) for name in writer_names}
    decisions: list[dict[str, Any]] = []
    decision_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    sft_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_examples: set[str] = set()
    parent_splits: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    input_tier_counts: Counter[str] = Counter()
    selected_tier_counts: Counter[str] = Counter()
    selected_provider_counts: Counter[str] = Counter()
    selected_tool_provider_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    input_digest: dict[str, hashlib._Hash] = {}
    input_bytes: Counter[str] = Counter()
    input_records: Counter[str] = Counter()
    try:
        with contextlib.ExitStack() as stack:
            active_writers = {name: stack.enter_context(writer) for name, writer in writers.items()}
            for binding in bindings:
                source_id = binding["source_id"]
                digest = hashlib.sha256()
                with binding["path"].open("rb") as source:
                    for source_line, raw in enumerate(source, 1):
                        digest.update(raw)
                        input_bytes[source_id] += len(raw)
                        if not raw.strip():
                            continue
                        input_records[source_id] += 1
                        source_counts["input_records"] += 1
                        row_sha256 = sha256_bytes(raw)
                        try:
                            row = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            decision = {
                                "schema_version": DECISION_SCHEMA,
                                "source_binding_id": source_id,
                                "source_line": source_line,
                                "source_row_sha256": row_sha256,
                                "source_example_id": None,
                                "parent_record_sha256": None,
                                "provider": None,
                                "agent": None,
                                "model_tier": "unclassified",
                                "model_tier_basis": "invalid_json",
                                "quality_gate": "unassessed",
                                "quality_flags_json": "[]",
                                "training_lane": "unclassified",
                                "sft_decision": "excluded",
                                "sft_reasons": ["invalid_json"],
                                "tool_review_decision": "excluded",
                                "tool_review_reasons": ["invalid_json"],
                                "rl_decision": "not_exported",
                                "rl_reason": "executable_reward_not_observed",
                                "error": exc.msg,
                            }
                            decisions.append(decision)
                            decision_by_key[(source_id, source_line)] = decision
                            exclusion_counts["invalid_json"] += 1
                            continue
                        if not isinstance(row, dict):
                            reasons = ["row_not_object"]
                            decision = {
                                "schema_version": DECISION_SCHEMA,
                                "source_binding_id": source_id,
                                "source_line": source_line,
                                "source_row_sha256": row_sha256,
                                "source_example_id": None,
                                "parent_record_sha256": None,
                                "provider": None,
                                "agent": None,
                                "model_tier": "unclassified",
                                "model_tier_basis": "row_not_object",
                                "quality_gate": "unassessed",
                                "quality_flags_json": "[]",
                                "training_lane": "unclassified",
                                "sft_decision": "excluded",
                                "sft_reasons": reasons,
                                "tool_review_decision": "excluded",
                                "tool_review_reasons": reasons,
                                "rl_decision": "not_exported",
                                "rl_reason": "executable_reward_not_observed",
                            }
                            decisions.append(decision)
                            decision_by_key[(source_id, source_line)] = decision
                            exclusion_counts.update(reasons)
                            continue
                        context = _base_context(
                            row,
                            binding=binding,
                            source_line=source_line,
                            row_sha256=row_sha256,
                        )
                        provider = context["provider"]
                        gate_counts[context["quality_gate"]] += 1
                        input_tier_counts[context["model_tier"]] += 1
                        decision = _decision_base(
                            row,
                            binding=binding,
                            source_line=source_line,
                            row_sha256=row_sha256,
                            context=context,
                        )
                        example_id = row.get("example_id")
                        duplicate_example = isinstance(example_id, str) and example_id in seen_examples
                        if isinstance(example_id, str) and example_id:
                            seen_examples.add(example_id)
                        if duplicate_example:
                            decision["sft_reasons"].append("duplicate_source_example_id")
                            decision["tool_review_reasons"].append("duplicate_source_example_id")
                        sft_view, sft_reasons = _sft_candidate(
                            row,
                            context=context,
                            source_id=source_id,
                            source_line=source_line,
                        )
                        if duplicate_example:
                            sft_view = None
                            sft_reasons = ["duplicate_source_example_id"]
                        if sft_view is not None and context["parent"] is not None:
                            sft_candidates[context["parent"]].append(
                                {
                                    "view": sft_view,
                                    "row": row,
                                    "context": context,
                                    "source_id": source_id,
                                    "source_line": source_line,
                                    "row_sha256": row_sha256,
                                }
                            )
                            decision["sft_decision"] = "eligible_pending_parent_selection"
                            decision["sft_reasons"] = ["sft_candidate_eligible"]
                        else:
                            decision["sft_reasons"] = sorted(set(sft_reasons))
                        tool_row, tool_reasons, event_summary = _tool_review_row(
                            row,
                            context=context,
                            binding=binding,
                            source_line=source_line,
                            row_sha256=row_sha256,
                            split=_split_for_parent(context["parent"]) if context["parent"] else "unassigned",
                        )
                        if duplicate_example:
                            tool_row = None
                            tool_reasons = ["duplicate_source_example_id"]
                        if tool_row is not None:
                            split = tool_row["split"]
                            destination = "tool_review_validation.jsonl" if split == "validation" else "tool_review_train.jsonl"
                            active_writers[destination].write(tool_row)
                            active_writers["lineage.jsonl"].write(
                                _lineage_row(
                                    kind="tool_review",
                                    record=tool_row,
                                    context=context,
                                    split=split,
                                    source_row_sha256=row_sha256,
                                )
                            )
                            decision["tool_review_decision"] = "selected"
                            decision["tool_review_reasons"] = sorted(set(tool_reasons))
                            selected_tool_provider_counts[provider] += 1
                            source_counts["tool_review_selected"] += 1
                        else:
                            decision["tool_review_reasons"] = sorted(set(tool_reasons))
                        decision_by_key[(source_id, source_line)] = decision
                        decisions.append(decision)
            for source_id, binding in zip(
                (binding["source_id"] for binding in bindings), bindings
            ):
                if input_records[source_id] != binding["records"]:
                    raise SalvagePilotError(
                        f"source record count mismatch: {source_id}: "
                        f"{input_records[source_id]} != {binding['records']}"
                    )
                if input_bytes[source_id] != binding["bytes"]:
                    raise SalvagePilotError(
                        f"source byte count mismatch: {source_id}: "
                        f"{input_bytes[source_id]} != {binding['bytes']}"
                    )
                # Re-opened source digest is intentionally checked after the
                # stream so no derived artifact can hide a changed source.
                actual_sha = sha256_file(binding["path"])
                if actual_sha != binding["sha256"]:
                    raise SalvagePilotError(f"source SHA mismatch: {source_id}")
            selected_parents: set[str] = set()
            for parent in sorted(sft_candidates):
                candidates = sorted(
                    sft_candidates[parent],
                    key=lambda item: (
                        TIER_RANK[item["context"]["model_tier"]],
                        -len(item["view"].get("messages", [])),
                        item["source_id"],
                        item["source_line"],
                        str(item["view"].get("example_id")),
                    ),
                )
                winner = candidates[0]
                selected_parents.add(parent)
                split = _split_for_parent(parent)
                tier = winner["context"]["model_tier"]
                lane = "sft_tier1" if tier == "tier1_frontier" else "sft_optional"
                filename = f"{lane}_{split}.jsonl"
                view = dict(winner["view"])
                view["split"] = split
                active_writers[filename].write(view)
                lineage_record = dict(view)
                lineage_record["quality_tier"] = "silver_sft"
                active_writers["lineage.jsonl"].write(
                    _lineage_row(
                        kind=lane,
                        record=lineage_record,
                        context=winner["context"],
                        split=split,
                        source_row_sha256=winner["row_sha256"],
                    )
                )
                decision = decision_by_key[(winner["source_id"], winner["source_line"])]
                decision["sft_decision"] = "selected"
                decision["sft_split"] = split
                decision["sft_reasons"] = ["selected_one_row_per_parent"]
                selected_tier_counts[tier] += 1
                selected_provider_counts[winner["context"]["provider"]] += 1
                source_counts["sft_selected"] += 1
                for loser in candidates[1:]:
                    loser_decision = decision_by_key[(loser["source_id"], loser["source_line"])]
                    loser_decision["sft_decision"] = "excluded"
                    loser_decision["sft_reasons"] = ["duplicate_parent_selection"]
            for decision in decisions:
                if decision["sft_decision"] == "excluded":
                    exclusion_counts.update(decision.get("sft_reasons", []))
                if decision["tool_review_decision"] == "excluded":
                    exclusion_counts.update(decision.get("tool_review_reasons", []))
            for decision in decisions:
                active_writers["decisions.jsonl"].write(decision)
            files = {name: writer.descriptor() for name, writer in active_writers.items()}
            manifest = {
                "schema_version": PACKAGE_SCHEMA,
                "status": "review_only",
                "trainer_loadable": source_counts["sft_selected"] > 0,
                "training_authorized": False,
                "format": "tiered_messages_jsonl_plus_tool_review_jsonl",
                "input_spec": {
                    "path": str(input_spec),
                    "sha256": sha256_file(input_spec),
                    "schema_version": spec.get("schema_version"),
                },
                "sources": [
                    {
                        key: value
                        for key, value in binding.items()
                        if key not in {"path", "manifest_path"}
                    }
                    | {
                        "path": str(binding["path"]),
                        "manifest_path": str(binding["manifest_path"]),
                    }
                    for binding in bindings
                ],
                "counts": {
                    "input_records": source_counts["input_records"],
                    "decision_records": len(decisions),
                    "sft_selected": source_counts["sft_selected"],
                    "sft_tier1_train": active_writers["sft_tier1_train.jsonl"].records,
                    "sft_tier1_validation": active_writers["sft_tier1_validation.jsonl"].records,
                    "sft_optional_train": active_writers["sft_optional_train.jsonl"].records,
                    "sft_optional_validation": active_writers["sft_optional_validation.jsonl"].records,
                    "tool_review_selected": source_counts["tool_review_selected"],
                    "tool_review_train": active_writers["tool_review_train.jsonl"].records,
                    "tool_review_validation": active_writers["tool_review_validation.jsonl"].records,
                    "unique_sft_parents": len(selected_parents),
                    "unique_tool_review_parents": len(
                        {
                            decision["parent_record_sha256"]
                            for decision in decisions
                            if decision["tool_review_decision"] == "selected"
                            and decision["parent_record_sha256"]
                        }
                    ),
                    "rl_records": 0,
                },
                "quality": {
                    "input_gate_counts": dict(sorted(gate_counts.items())),
                    "input_model_tier_counts": dict(sorted(input_tier_counts.items())),
                    "selected_model_tier_counts": dict(sorted(selected_tier_counts.items())),
                    "selected_provider_counts": dict(sorted(selected_provider_counts.items())),
                    "selected_tool_provider_counts": dict(sorted(selected_tool_provider_counts.items())),
                    "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
                    "limitations": [
                        "outcome_unknown_for_sft",
                        "not_human_adjudicated",
                        "privacy_authorization_not_granted",
                        "tool_review_schema_not_observed",
                        "verifier_not_observed",
                        "rewards_not_exported",
                    ],
                },
                "policy": {
                    "sft": "candidate, tool-free, privacy/firewall-clean, reasoning-free, one row per parent",
                    "tool_review": "recoverable matched calls/observations, privacy/reasoning firewall, preserve source gate",
                    "model_tier": "explicit row declaration or existing model identity registry; provider alone never binds",
                    "split": "sha256(parent) final digit 0-7 train, 8-9 validation",
                    "rl": "not_exported_without_executable_reward",
                },
                "lineage_contract": {
                    "outer_source_file": "validated input-artifact binding",
                    "source_origin_file": "nested original session/file identity when ingress provides it",
                    "source_origin_json": "metadata-only nested origin snapshot; no conversation content",
                },
                "validation": {
                    "source_bindings": "passed",
                    "streaming_projection": "passed",
                    "parent_split": "pending_loader_audit",
                    "trainer_projection": "passed_for_selected_sft",
                    "reasoning_firewall": "passed_for_selected_rows",
                    "tool_schema": "not_observed_not_inferred",
                    "reward": "not_present",
                    "loader": "pending",
                },
                "files": files,
            }
            (staging / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_spec", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_multi_provider_salvage_pilot(
                input_spec=args.input_spec,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INPUT_SCHEMA",
    "PACKAGE_SCHEMA",
    "SalvagePilotError",
    "build_multi_provider_salvage_pilot",
]
