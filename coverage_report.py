#!/usr/bin/env python3
"""Reconcile source inventory, ingress, preflight, and canonical build stages.

This is a metadata-only loss report.  It never copies conversation text into
the report and it deliberately distinguishes a direct source-file hash from a
weak derived fingerprint.  A non-empty downstream dataset is not coverage
proof: the report is only complete when every hashed inventory candidate is
bound to direct ingress evidence and every supplied raw artifact has a
matching preflight/build identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


COVERAGE_SCHEMA = "ai-data-extraction/source-coverage/v1"
COVERAGE_VERSION = "1.0.0"
DEFAULT_PARSE_LIMIT_BYTES = 8 * 1024 * 1024
SCAN_CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# These are the only record fields treated as source identity.  In particular,
# arbitrary ``sha256`` fields in observations or artifacts are not source
# evidence.
SOURCE_HASH_FIELDS = {
    "source_file_sha256": "source_origin_file_hash",
    "source_sha256": "source_origin_file_hash",
    "source_fingerprint": "source_fingerprint",
}
SOURCE_PATH_FIELDS = ("source_file", "session_file")
PREFIX_FIELD_RE = re.compile(
    rb'"(?P<key>source|source_class|source_file|session_file)"\s*:\s*'
    rb'"(?P<value>(?:\\.|[^"\\])*)"'
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith("sha256:"):
        value = value[7:]
    if not SHA256_RE.fullmatch(value):
        return None
    return value.lower()


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _prefix_fields(prefix: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in PREFIX_FIELD_RE.finditer(prefix):
        key = match.group("key").decode("ascii")
        if key in fields:
            continue
        try:
            decoded = json.loads(b'"' + match.group("value") + b'"')
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(decoded, str):
            fields[key] = decoded
    return fields


def _source_hashes_from_origin(
    origin: Any,
    *,
    add_evidence: Any,
    provider: str | None,
    source_class: str | None,
) -> None:
    """Collect only fields with an explicit source-provenance meaning."""

    if not isinstance(origin, dict):
        return
    for key, kind in SOURCE_HASH_FIELDS.items():
        value = _digest(origin.get(key))
        if value:
            add_evidence(value, kind=kind, provider=provider, source_class=source_class)

    # OpenCode records place direct file hashes under database and wal.  Do not
    # recurse over arbitrary objects: nested content digests are not file
    # identity.
    for container_key in ("database", "wal", "source_file"):
        container = origin.get(container_key)
        if not isinstance(container, dict):
            continue
        value = _digest(container.get("sha256"))
        if value:
            add_evidence(
                value,
                kind="source_origin_file_hash",
                provider=provider,
                source_class=source_class,
            )


def _source_hashes_from_record(
    record: dict[str, Any],
    *,
    add_evidence: Any,
    hash_source_paths: bool,
    source_path_cache: dict[str, tuple[str, int] | None],
    source_path_stats: Counter[str],
) -> None:
    provider = record.get("source") if isinstance(record.get("source"), str) else None
    source_class = (
        record.get("source_class")
        if isinstance(record.get("source_class"), str)
        else None
    )
    origin = record.get("source_origin")
    _source_hashes_from_origin(
        origin,
        add_evidence=add_evidence,
        provider=provider,
        source_class=source_class,
    )

    if not hash_source_paths:
        return
    for field in SOURCE_PATH_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            continue
        source_path = Path(value).expanduser()
        if not source_path.is_absolute():
            continue
        cache_key = str(source_path)
        if cache_key not in source_path_cache:
            try:
                source_path_cache[cache_key] = _sha256_file(source_path)
            except (OSError, ValueError):
                source_path_cache[cache_key] = None
        fingerprint = source_path_cache[cache_key]
        if fingerprint is None:
            source_path_stats["missing_or_unreadable"] += 1
            continue
        source_path_stats["hashed"] += 1
        add_evidence(
            fingerprint[0],
            kind="source_path_hashed",
            provider=provider,
            source_class=source_class,
        )
        # A path is retained only in the local cache for the duration of the
        # scan.  It is never copied to the report.


def _source_from_prefix(
    prefix: bytes,
    *,
    add_evidence: Any,
    hash_source_paths: bool,
    source_path_cache: dict[str, tuple[str, int] | None],
    source_path_stats: Counter[str],
) -> None:
    fields = _prefix_fields(prefix)
    record: dict[str, Any] = {
        key: value for key, value in fields.items() if key in {"source", "source_class"}
    }
    for field in SOURCE_PATH_FIELDS:
        if field in fields:
            record[field] = fields[field]
    _source_hashes_from_record(
        record,
        add_evidence=add_evidence,
        hash_source_paths=hash_source_paths,
        source_path_cache=source_path_cache,
        source_path_stats=source_path_stats,
    )


def scan_raw_artifact(
    path: Path,
    *,
    add_evidence: Any,
    parse_limit_bytes: int = DEFAULT_PARSE_LIMIT_BYTES,
    hash_source_paths: bool = False,
    source_path_cache: dict[str, tuple[str, int] | None] | None = None,
    source_path_stats: Counter[str] | None = None,
) -> dict[str, Any]:
    """Hash and count a JSONL artifact without retaining oversized records."""

    if parse_limit_bytes < 0:
        raise ValueError("parse_limit_bytes cannot be negative")
    if not path.is_file():
        raise FileNotFoundError(path)
    source_path_cache = source_path_cache if source_path_cache is not None else {}
    source_path_stats = source_path_stats if source_path_stats is not None else Counter()
    source_path_stats_before = Counter(source_path_stats)

    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    captured = bytearray()
    line_bytes = 0
    line_number = 1
    scanned_bytes = 0
    stat_before = path.stat()

    def consume(piece: bytes) -> None:
        nonlocal line_bytes
        if not piece:
            return
        line_bytes += len(piece)
        remaining = parse_limit_bytes + 1 - len(captured)
        if remaining > 0:
            captured.extend(piece[:remaining])

    def finish_line() -> None:
        nonlocal captured, line_bytes, line_number
        if line_bytes == 0 or bytes(captured).strip() == b"":
            counts["empty_lines"] += 1
        else:
            counts["records"] += 1
            if line_bytes > parse_limit_bytes:
                counts["unparsed_oversize"] += 1
                _source_from_prefix(
                    bytes(captured),
                    add_evidence=add_evidence,
                    hash_source_paths=hash_source_paths,
                    source_path_cache=source_path_cache,
                    source_path_stats=source_path_stats,
                )
            else:
                try:
                    value = json.loads(bytes(captured))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    counts["invalid_json"] += 1
                else:
                    if isinstance(value, dict):
                        counts["valid_json"] += 1
                        _source_hashes_from_record(
                            value,
                            add_evidence=add_evidence,
                            hash_source_paths=hash_source_paths,
                            source_path_cache=source_path_cache,
                            source_path_stats=source_path_stats,
                        )
                    else:
                        counts["valid_json_non_object"] += 1
        captured = bytearray()
        line_bytes = 0
        line_number += 1

    with path.open("rb") as source:
        for block in iter(lambda: source.read(SCAN_CHUNK_BYTES), b""):
            digest.update(block)
            scanned_bytes += len(block)
            start = 0
            while True:
                newline = block.find(b"\n", start)
                if newline < 0:
                    consume(block[start:])
                    break
                consume(block[start:newline])
                finish_line()
                start = newline + 1
    if line_bytes:
        finish_line()

    try:
        stat_after = path.stat()
    except OSError:
        stat_after = None
    path_stats_delta = source_path_stats - source_path_stats_before
    return {
        "name": path.name,
        "sha256": digest.hexdigest(),
        "bytes": scanned_bytes,
        "counts": _counter_dict(counts),
        "source_path_stats": _counter_dict(path_stats_delta),
        "stable_size": (
            stat_after is not None
            and stat_before.st_size == stat_after.st_size
            and stat_before.st_mtime_ns == stat_after.st_mtime_ns
            and stat_after.st_size == scanned_bytes
        ),
    }


def _add_evidence_factory() -> tuple[dict[str, dict[str, Any]], Any]:
    evidence: dict[str, dict[str, Any]] = {}

    def add(
        value: str,
        *,
        kind: str,
        provider: str | None,
        source_class: str | None,
    ) -> None:
        item = evidence.setdefault(
            value,
            {
                "evidence_count": 0,
                "evidence_kinds": Counter(),
                "providers": Counter(),
                "source_classes": Counter(),
            },
        )
        item["evidence_count"] += 1
        item["evidence_kinds"][kind] += 1
        if provider:
            item["providers"][provider] += 1
        if source_class:
            item["source_classes"][source_class] += 1

    return evidence, add


def _manifest_source_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in ("source_sessions", "source_files", "source_origins"):
        value = manifest.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    return entries


def collect_ingress_manifest(
    path: Path,
    *,
    add_evidence: Any,
) -> dict[str, Any]:
    manifest = _load_json(path)
    entries = _manifest_source_entries(manifest)
    by_lane: Counter[str] = Counter()
    by_provider: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    known_hashes = 0
    for entry in entries:
        provider = entry.get("provider") if isinstance(entry.get("provider"), str) else None
        source_class = (
            entry.get("source_class")
            if isinstance(entry.get("source_class"), str)
            else None
        )
        lane = entry.get("training_lane") if isinstance(entry.get("training_lane"), str) else None
        if provider:
            by_provider[provider] += 1
        if source_class:
            by_class[source_class] += 1
        if lane:
            by_lane[lane] += 1
        found = False
        for key in ("source_file_sha256", "source_sha256"):
            value = _digest(entry.get(key))
            if value:
                found = True
                add_evidence(
                    value,
                    kind="source_manifest_file_hash",
                    provider=provider,
                    source_class=source_class,
                )
        origin = entry.get("source_origin")
        if isinstance(origin, dict):
            origin_hash_present = any(
                _digest(origin.get(key)) for key in ("source_file_sha256", "source_sha256")
            )
            origin_hash_present = origin_hash_present or any(
                isinstance(origin.get(container), dict)
                and _digest(origin[container].get("sha256"))
                for container in ("database", "wal", "source_file")
            )
            _source_hashes_from_origin(
                origin,
                add_evidence=add_evidence,
                provider=provider,
                source_class=source_class,
            )
            found = found or origin_hash_present
        known_hashes += int(found)
    return {
        "name": path.name,
        "schema_version": manifest.get("schema_version"),
        "entries": len(entries),
        "entries_with_source_hash": known_hashes,
        "providers": _counter_dict(by_provider),
        "source_classes": _counter_dict(by_class),
        "training_lanes": _counter_dict(by_lane),
    }


def _inventory_summary(
    manifest: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Inventory manifest has no files list")
    treatment_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    candidate_files = 0
    candidate_bytes = 0
    candidate_hashes: set[str] = set()
    unmatched_class_files: Counter[str] = Counter()
    unmatched_class_bytes: Counter[str] = Counter()
    weak_matches = 0
    strong_matches = 0
    unhashed = 0
    ambiguous_hashes = 0
    hash_to_entries: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for entry in files:
        if not isinstance(entry, dict):
            continue
        treatment = str(entry.get("treatment", "unknown"))
        source_class = str(entry.get("source_class", "unknown"))
        class_key = f"{entry.get('provider', 'unknown')}:{source_class}"
        treatment_counts[treatment] += 1
        class_counts[class_key] += 1
        if treatment != "candidate":
            continue
        candidate_files += 1
        candidate_bytes += int(entry.get("bytes", 0) or 0)
        digest = _digest(entry.get("file_sha256"))
        if not digest:
            unhashed += 1
            unmatched_class_files[class_key] += 1
            unmatched_class_bytes[class_key] += int(entry.get("bytes", 0) or 0)
            continue
        candidate_hashes.add(digest)
        hash_to_entries[digest].append(entry)

    for digest, entries in hash_to_entries.items():
        if len(entries) > 1:
            ambiguous_hashes += 1
        source_evidence = evidence.get(digest)
        kinds = set(source_evidence["evidence_kinds"]) if source_evidence else set()
        strong = bool(
            kinds
            & {
                "source_manifest_file_hash",
                "source_origin_file_hash",
                "source_path_hashed",
            }
        )
        if strong:
            strong_matches += len(entries)
        elif source_evidence:
            weak_matches += len(entries)
            for entry in entries:
                class_key = f"{entry.get('provider', 'unknown')}:{entry.get('source_class', 'unknown')}"
                unmatched_class_files[class_key] += 1
                unmatched_class_bytes[class_key] += int(entry.get("bytes", 0) or 0)
        else:
            for entry in entries:
                class_key = f"{entry.get('provider', 'unknown')}:{entry.get('source_class', 'unknown')}"
                unmatched_class_files[class_key] += 1
                unmatched_class_bytes[class_key] += int(entry.get("bytes", 0) or 0)

    matched_files = strong_matches
    unmatched_files = candidate_files - matched_files
    matched_bytes = candidate_bytes - sum(unmatched_class_bytes.values())
    return {
        "schema_version": manifest.get("schema_version"),
        "hash_mode": manifest.get("hash_mode"),
        "counts": manifest.get("counts", {}),
        "treatment_counts": _counter_dict(treatment_counts),
        "class_file_counts": _counter_dict(class_counts),
        "candidate": {
            "files": candidate_files,
            "bytes": candidate_bytes,
            "unique_hashes": len(candidate_hashes),
            "hashed_files": candidate_files - unhashed,
            "unhashed_files": unhashed,
            "matched_files": matched_files,
            "matched_bytes": matched_bytes,
            "weakly_matched_files": weak_matches,
            "unmatched_files": unmatched_files,
            "unmatched_bytes": sum(unmatched_class_bytes.values()),
            "ambiguous_hashes": ambiguous_hashes,
            "coverage_by_file": (
                matched_files / candidate_files if candidate_files else 1.0
            ),
            "coverage_by_bytes": (
                matched_bytes / candidate_bytes if candidate_bytes else 1.0
            ),
        },
        "unmatched_candidate_classes": {
            key: {
                "files": unmatched_class_files[key],
                "bytes": unmatched_class_bytes[key],
            }
            for key in sorted(unmatched_class_files)
        },
        "inventory_errors": len(manifest.get("errors", []))
        if isinstance(manifest.get("errors"), list)
        else None,
    }


def _load_preflight(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    inputs = []
    for entry in manifest.get("inputs", []):
        if not isinstance(entry, dict):
            continue
        inputs.append(
            {
                "name": entry.get("name"),
                "sha256": _digest(entry.get("sha256")),
                "bytes": entry.get("bytes"),
                "parser_status": entry.get("parser_status"),
                "stable_size": entry.get("stable_size"),
                "records": (entry.get("counts") or {}).get("records", 0),
                "valid_json": (entry.get("counts") or {}).get("valid_json", 0),
                "unparsed_oversize": (entry.get("counts") or {}).get(
                    "unparsed_oversize", 0
                ),
            }
        )
    return {
        "name": path.name,
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "counts": manifest.get("counts", {}),
        "inputs": inputs,
    }


def _load_build(path: Path, raw_artifacts: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    manifest = _load_json(path)
    bound_inputs = []
    for entry in manifest.get("inputs", []):
        if not isinstance(entry, dict):
            continue
        digest = _digest(entry.get("sha256"))
        name = entry.get("name")
        match = raw_artifacts.get((name, digest)) if isinstance(name, str) and digest else None
        bound_inputs.append(
            {
                "name": name,
                "sha256": digest,
                "bytes": entry.get("bytes"),
                "raw_artifact_bound": match is not None,
            }
        )
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        policy = {}
    return {
        "name": path.name,
        "schema_version": manifest.get("schema_version"),
        "builder_version": manifest.get("builder_version"),
        "counts": manifest.get("counts", {}),
        "training_lanes": manifest.get("training_lanes", {}),
        "privacy": {
            "mode": policy.get("privacy_mode"),
            "approved": policy.get("privacy_approved"),
        },
        "inputs": bound_inputs,
        "all_inputs_bound": all(item["raw_artifact_bound"] for item in bound_inputs),
    }


def _preflight_bindings(
    raw_artifacts: list[dict[str, Any]],
    preflights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preflight_by_identity: defaultdict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for manifest in preflights:
        for entry in manifest["inputs"]:
            preflight_by_identity[(entry.get("name"), entry.get("sha256"))].append(manifest)
    result = []
    for artifact in raw_artifacts:
        identity = (artifact["name"], artifact["sha256"])
        matches = preflight_by_identity.get(identity, [])
        result.append(
            {
                "name": artifact["name"],
                "sha256": artifact["sha256"],
                "matching_preflights": [item["name"] for item in matches],
                "preflight_bound": bool(matches),
                "preflight_complete": all(item.get("status") == "complete" for item in matches),
            }
        )
    return result


def _projection_accounting(build: dict[str, Any]) -> dict[str, Any]:
    counts = build.get("counts") if isinstance(build.get("counts"), dict) else {}
    input_records = int(counts.get("input_records", 0) or 0)
    normalized = int(counts.get("normalized_records", 0) or 0)
    rejected = int(counts.get("rejected", 0) or 0)
    skipped = int(counts.get("skipped_training_lane_records", 0) or 0)
    skipped_quality = int(counts.get("skipped_quality_gate_records", 0) or 0)
    selected = input_records - skipped
    gap = selected - normalized - rejected - skipped_quality
    if gap == 0:
        status = "balanced"
    elif counts.get("chunk_records", 0) or counts.get("chunked_parent_records", 0):
        status = "requires_chunk_accounting"
    else:
        status = "unreconciled"
    return {
        "input_records": input_records,
        "skipped_training_lane_records": skipped,
        "selected_input_records": selected,
        "normalized_records": normalized,
        "rejected_records": rejected,
        "skipped_quality_gate_records": skipped_quality,
        "projection_gap": gap,
        "status": status,
    }


def build_coverage_report(
    *,
    inventory: Path,
    ingress_manifests: Iterable[Path] = (),
    raw_artifacts: Iterable[Path] = (),
    preflight_manifests: Iterable[Path] = (),
    build_manifests: Iterable[Path] = (),
    parse_limit_bytes: int = DEFAULT_PARSE_LIMIT_BYTES,
    hash_source_paths: bool = False,
) -> dict[str, Any]:
    evidence, add_evidence = _add_evidence_factory()
    source_path_cache: dict[str, tuple[str, int] | None] = {}
    source_path_stats: Counter[str] = Counter()

    ingress_reports = [
        collect_ingress_manifest(path, add_evidence=add_evidence)
        for path in ingress_manifests
    ]
    raw_reports = []
    for path in raw_artifacts:
        raw_reports.append(
            scan_raw_artifact(
                path,
                add_evidence=add_evidence,
                parse_limit_bytes=parse_limit_bytes,
                hash_source_paths=hash_source_paths,
                source_path_cache=source_path_cache,
                source_path_stats=source_path_stats,
            )
        )
    preflight_reports = [_load_preflight(path) for path in preflight_manifests]
    raw_by_identity = {
        (entry["name"], entry["sha256"]): entry for entry in raw_reports
    }
    build_reports = [
        _load_build(path, raw_by_identity) for path in build_manifests
    ]
    inventory_manifest = _load_json(inventory)
    inventory_report = _inventory_summary(inventory_manifest, evidence)
    raw_preflight = _preflight_bindings(raw_reports, preflight_reports)

    strong_source_coverage = inventory_report["candidate"]["unmatched_files"] == 0 and not inventory_report["candidate"]["unhashed_files"]
    raw_preflight_complete = all(
        item["preflight_bound"] and item["preflight_complete"] for item in raw_preflight
    )
    builders_bound = all(item["all_inputs_bound"] for item in build_reports)
    report_status = (
        "complete"
        if strong_source_coverage
        and raw_preflight_complete
        and builders_bound
        and not inventory_report.get("inventory_errors")
        else "partial"
    )

    serialized_evidence = {}
    for digest, item in evidence.items():
        serialized_evidence[digest] = {
            "evidence_count": item["evidence_count"],
            "evidence_kinds": _counter_dict(item["evidence_kinds"]),
            "providers": _counter_dict(item["providers"]),
            "source_classes": _counter_dict(item["source_classes"]),
        }

    return {
        "schema_version": COVERAGE_SCHEMA,
        "coverage_version": COVERAGE_VERSION,
        "status": report_status,
        "privacy": "metadata_only_no_source_content_or_paths",
        "policies": {
            "direct_source_match": "inventory candidate file_sha256 must match source manifest, source_origin file hash, or explicitly hashed legacy source path",
            "weak_evidence": "derived source_fingerprint is reported but does not count as coverage",
            "raw_scan": "streaming JSONL scan; records above parse_limit_bytes are counted and prefix-inspected only",
            "hash_source_paths": hash_source_paths,
            "parse_limit_bytes": parse_limit_bytes,
        },
        "inventory": inventory_report,
        "ingress_manifests": ingress_reports,
        "raw_artifacts": raw_reports,
        "preflight_manifests": preflight_reports,
        "raw_preflight_bindings": raw_preflight,
        "build_manifests": [
            {**item, "projection_accounting": _projection_accounting(item)}
            for item in build_reports
        ],
        "source_evidence": {
            "unique_hashes": len(serialized_evidence),
            "hashes": serialized_evidence,
            "legacy_source_path_stats": _counter_dict(source_path_stats),
        },
        "gates": {
            "inventory_has_no_errors": not bool(inventory_report.get("inventory_errors")),
            "all_candidate_files_directly_bound": strong_source_coverage,
            "all_raw_artifacts_preflight_complete": raw_preflight_complete,
            "all_build_inputs_bound_to_raw_artifacts": builders_bound,
            "privacy_approved": all(
                item["privacy"].get("approved") is True for item in build_reports
            )
            if build_reports
            else False,
        },
    }


def write_report(path: Path, report: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--ingress-manifest", action="append", type=Path, default=[])
    parser.add_argument("--raw", action="append", type=Path, default=[])
    parser.add_argument("--preflight", action="append", type=Path, default=[])
    parser.add_argument("--build", action="append", type=Path, default=[])
    parser.add_argument("--parse-limit-bytes", type=int, default=DEFAULT_PARSE_LIMIT_BYTES)
    parser.add_argument(
        "--hash-source-paths",
        action="store_true",
        help="Hash legacy source_file/session_file paths found in raw metadata",
    )
    parser.add_argument("--output", type=Path, default=Path(".tmp/source_coverage.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_coverage_report(
            inventory=args.inventory,
            ingress_manifests=args.ingress_manifest,
            raw_artifacts=args.raw,
            preflight_manifests=args.preflight,
            build_manifests=args.build,
            parse_limit_bytes=args.parse_limit_bytes,
            hash_source_paths=args.hash_source_paths,
        )
        write_report(args.output, report, overwrite=args.overwrite)
    except (FileNotFoundError, OSError, ValueError, FileExistsError) as exc:
        print(f"Coverage report failed: {exc}", file=sys.stderr)
        return 2
    candidate = report["inventory"]["candidate"]
    print(
        f"Status: {report['status']}; candidate files directly bound: "
        f"{candidate['matched_files']}/{candidate['files']}; "
        f"unmatched: {candidate['unmatched_files']}"
    )
    print(f"Report: {args.output}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
