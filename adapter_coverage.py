#!/usr/bin/env python3
"""Reconcile source-ledger candidates with provider adapter evidence.

The report is metadata-only.  Exact source-ref evidence is closure-grade;
content-digest-only evidence is retained but downgraded because duplicate
source files can share bytes.  A routed but unseen source is ``unparsed``;
identity drift and unsupported classes are separate statuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from source_manifest import ADMISSIBLE_SNAPSHOT_STATUSES, SourceManifestIndex


COVERAGE_SCHEMA = "ai-data-extraction/adapter-coverage/v1"
COVERAGE_VERSION = "1.0.0"
SHA256_LENGTH = 64

# This is a routing contract, not a claim that the adapter has successfully
# parsed every row.  Successful coverage is established only by ingress
# evidence joined to the source-ref emitted by that adapter.
ADAPTER_ROUTES: dict[tuple[str, str], str] = {
    ("claude", "session_active"): "extract_claude_code.py",
    ("codex", "session_active"): "extract_codex.py",
    ("codex", "session_backup"): "extract_codex.py",
    ("gemini", "session_active"): "extract_gemini.py",
    ("oh-my-pi", "advisor_overlay"): "extract_agent_sessions.py",
    ("oh-my-pi", "session_active"): "extract_agent_sessions.py",
    ("opencode", "conversation_sidecar"): "extract_opencode.py",
    ("opencode", "conversation_store"): "extract_opencode.py",
    ("opencode", "tool_output"): "extract_opencode.py",
    ("prime-agent", "session_active"): "extract_agent_sessions.py",
    ("prime-agent", "subagent_session"): "extract_agent_sessions.py",
}


def _digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.removeprefix("sha256:").lower()
    if len(value) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        return None
    return value


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _add_ingress_value(
    value: Any,
    *,
    as_ref: bool,
    refs: set[str],
    digests: set[str],
    ref_counts: Counter[str],
    digest_counts: Counter[str],
) -> None:
    digest = _digest(value)
    if digest is None:
        return
    if as_ref:
        refs.add(digest)
        ref_counts[digest] += 1
    else:
        digests.add(digest)
        digest_counts[digest] += 1


def _scan_ingress_jsonl(path: Path, evidence: dict[str, Any]) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            evidence["records"] += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                evidence["invalid_records"] += 1
                continue
            if not isinstance(value, dict):
                continue
            origin = value.get("source_origin")
            if not isinstance(origin, dict):
                continue
            _add_ingress_value(
                origin.get("source_ref_sha256"),
                as_ref=True,
                refs=evidence["refs"],
                digests=evidence["digests"],
                ref_counts=evidence["ref_counts"],
                digest_counts=evidence["digest_counts"],
            )
            _add_ingress_value(
                origin.get("source_file_sha256"),
                as_ref=False,
                refs=evidence["refs"],
                digests=evidence["digests"],
                ref_counts=evidence["ref_counts"],
                digest_counts=evidence["digest_counts"],
            )


def _scan_ingress_manifest(path: Path, evidence: dict[str, Any]) -> None:
    manifest = _load_object(path)
    entries: list[dict[str, Any]] = []
    for key in (
        "source_sessions",
        "source_files",
        "source_origins",
        "parsed_source_files",
    ):
        value = manifest.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    for entry in entries:
        evidence["records"] += 1
        _add_ingress_value(
            entry.get("source_ref_sha256"),
            as_ref=True,
            refs=evidence["refs"],
            digests=evidence["digests"],
            ref_counts=evidence["ref_counts"],
            digest_counts=evidence["digest_counts"],
        )
        _add_ingress_value(
            entry.get("source_file_sha256") or entry.get("source_sha256"),
            as_ref=False,
            refs=evidence["refs"],
            digests=evidence["digests"],
            ref_counts=evidence["ref_counts"],
            digest_counts=evidence["digest_counts"],
        )


def _new_evidence() -> dict[str, Any]:
    return {
        "refs": set(),
        "digests": set(),
        "ref_counts": Counter(),
        "digest_counts": Counter(),
        "records": 0,
        "invalid_records": 0,
    }


def _route(provider: str, source_class: str) -> str | None:
    return ADAPTER_ROUTES.get((provider, source_class))


def _load_source_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_object(path)
    SourceManifestIndex.from_path(path)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("source manifest files must be a list")
    rows = [item for item in files if isinstance(item, dict)]
    return manifest, rows


def reconcile_adapter_coverage(
    *,
    source_manifest: Path,
    ingress_jsonl: Iterable[Path] = (),
    ingress_manifests: Iterable[Path] = (),
) -> dict[str, Any]:
    manifest, rows = _load_source_rows(source_manifest)
    evidence = _new_evidence()
    ingress_artifacts: list[dict[str, Any]] = []
    for path in ingress_jsonl:
        path = Path(path)
        before_records = evidence["records"]
        before_invalid = evidence["invalid_records"]
        _scan_ingress_jsonl(path, evidence)
        digest, size = _file_digest(path)
        ingress_artifacts.append(
            {
                "name": path.name,
                "kind": "jsonl",
                "sha256": f"sha256:{digest}",
                "bytes": size,
                "records": evidence["records"] - before_records,
                "invalid_records": evidence["invalid_records"] - before_invalid,
            }
        )
    for path in ingress_manifests:
        path = Path(path)
        before_records = evidence["records"]
        before_invalid = evidence["invalid_records"]
        _scan_ingress_manifest(path, evidence)
        digest, size = _file_digest(path)
        ingress_artifacts.append(
            {
                "name": path.name,
                "kind": "manifest",
                "sha256": f"sha256:{digest}",
                "bytes": size,
                "records": evidence["records"] - before_records,
                "invalid_records": evidence["invalid_records"] - before_invalid,
            }
        )

    digest_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        digest = _digest(row.get("source_sha256"))
        if digest:
            digest_rows[digest].append(row)

    status_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    reconciled: list[dict[str, Any]] = []
    for row in rows:
        provider = str(row.get("provider", "unknown"))
        root_label = str(row.get("root_label", "unknown"))
        source_class = str(row.get("source_class", "unknown"))
        route = _route(provider, source_class)
        source_ref = _digest(row.get("source_ref_sha256"))
        source_digest = _digest(row.get("source_sha256"))
        if source_ref in evidence["refs"]:
            status = "parsed_exact_source_ref"
            evidence_kind = "source_ref"
        elif source_digest in evidence["digests"] and len(digest_rows[source_digest]) == 1:
            status = "parsed_digest_only_non_closure"
            evidence_kind = "source_digest_unique"
        elif source_digest in evidence["digests"]:
            status = "ambiguous_digest_only"
            evidence_kind = "source_digest_ambiguous"
        elif row.get("snapshot_status") not in ADMISSIBLE_SNAPSHOT_STATUSES:
            status = "blocked_source_identity_status"
            evidence_kind = None
        elif route is None:
            status = "unsupported_source_class"
            evidence_kind = None
        else:
            status = "unparsed_routed_source"
            evidence_kind = None
        status_counts[status] += 1
        route_counts[route or "unsupported"] += 1
        provider_counts[provider] += 1
        byte_counts[status] += int(row.get("source_bytes") or 0)
        reconciled.append(
            {
                "provider": provider,
                "root_label": root_label,
                "source_class": source_class,
                "treatment": row.get("treatment"),
                "source_bytes": row.get("source_bytes"),
                "source_sha256": row.get("source_sha256"),
                "source_ref_sha256": row.get("source_ref_sha256"),
                "snapshot_status": row.get("snapshot_status"),
                "adapter_route": route,
                "status": status,
                "evidence_kind": evidence_kind,
            }
        )

    exact_files = status_counts["parsed_exact_source_ref"]
    candidate_files = len(rows)
    report = {
        "schema_version": COVERAGE_SCHEMA,
        "coverage_version": COVERAGE_VERSION,
        "status": "complete" if exact_files == candidate_files else "partial",
        "privacy": "metadata_only_no_source_paths_or_content",
        "source_manifest_revision": manifest.get("source_manifest_revision"),
        "source_manifest_counts": manifest.get("counts", {}),
        "ingress_artifacts": ingress_artifacts,
        "ingress_evidence": {
            "records": evidence["records"],
            "invalid_records": evidence["invalid_records"],
            "unique_source_refs": len(evidence["refs"]),
            "unique_source_digests": len(evidence["digests"]),
        },
        "routing": {
            "route_counts": dict(sorted(route_counts.items())),
            "provider_counts": dict(sorted(provider_counts.items())),
            "routes": {
                f"{provider}:{source_class}": route
                for (provider, source_class), route in sorted(ADAPTER_ROUTES.items())
            },
        },
        "coverage": {
            "candidate_files": candidate_files,
            "candidate_bytes": sum(int(row.get("source_bytes") or 0) for row in rows),
            "status_counts": dict(sorted(status_counts.items())),
            "status_bytes": dict(sorted(byte_counts.items())),
            "exact_source_ref_files": exact_files,
            "exact_source_ref_bytes": byte_counts["parsed_exact_source_ref"],
            "exact_source_ref_fraction": exact_files / candidate_files if candidate_files else 1.0,
        },
        "files": reconciled,
    }
    report["coverage_revision"] = "sha256:" + hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


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
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--ingress-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--ingress-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = reconcile_adapter_coverage(
        source_manifest=args.source_manifest,
        ingress_jsonl=args.ingress_jsonl,
        ingress_manifests=args.ingress_manifest,
    )
    write_report(args.output, report, overwrite=args.overwrite)
    coverage = report["coverage"]
    print(
        f"Status: {report['status']}; exact source-ref coverage: "
        f"{coverage['exact_source_ref_files']}/{coverage['candidate_files']}"
    )
    print(f"Report: {args.output}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTER_ROUTES",
    "COVERAGE_SCHEMA",
    "COVERAGE_VERSION",
    "reconcile_adapter_coverage",
    "write_report",
]
