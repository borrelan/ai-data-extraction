#!/usr/bin/env python3
"""Preflight JSONL sources without normalizing or exporting message content.

The canonical builder must parse complete JSON records, which can make one
very large JSONL line consume several gigabytes of memory.  This command is a
bounded admission check: it hashes each source in one pass, measures every
non-empty line, parses only lines under a configurable cap, and records
metadata for oversized/unparseable lines so a later streaming adapter or
quarantine decision has recoverable provenance.

It deliberately does not produce training rows and never writes source text
to the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PREFLIGHT_SCHEMA = "ai-data-extraction/preflight/v1"
PREFLIGHT_VERSION = "1.0.0"
DEFAULT_MAX_RECORD_CHARS = 250_000
DEFAULT_PARSE_LIMIT_BYTES = 8 * 1024 * 1024
SCAN_CHUNK_BYTES = 1024 * 1024
JSONL_SUFFIXES = (".jsonl", ".jsonl.backup")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def discover_line_files(paths: Iterable[Path], *, excluded: Path | None = None) -> list[Path]:
    """Discover JSONL and explicit JSONL backup files without reading them."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    excluded_resolved = excluded.resolve() if excluded is not None else None

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.name.lower().endswith(JSONL_SUFFIXES)
            )
        else:
            continue

        for candidate in candidates:
            resolved = candidate.resolve()
            if excluded_resolved is not None and resolved == excluded_resolved:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)

    if not discovered:
        raise ValueError("No JSONL or JSONL backup inputs found")
    return discovered


def _line_entry(
    *,
    line_number: int,
    line_bytes: int,
    line_sha256: str,
    status: str,
) -> dict[str, Any]:
    return {
        "line": line_number,
        "bytes": line_bytes,
        "sha256": line_sha256,
        "status": status,
    }


def scan_jsonl(
    path: Path,
    *,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
    parse_limit_bytes: int = DEFAULT_PARSE_LIMIT_BYTES,
    chunk_bytes: int = SCAN_CHUNK_BYTES,
) -> dict[str, Any]:
    """Scan one JSONL source while bounding retained line data.

    The reported ``raw_oversize`` flag is conservative: it compares raw
    record bytes with the builder's normalized character limit.  It is an
    admission hint, not a substitute for normalized-size measurement.
    """

    if max_record_chars < 0:
        raise ValueError("max_record_chars cannot be negative")
    if parse_limit_bytes < 0:
        raise ValueError("parse_limit_bytes cannot be negative")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    oversized_lines: list[dict[str, Any]] = []
    largest_line_bytes = 0
    largest_line_number: int | None = None
    scanned_bytes = 0
    line_number = 1
    line_bytes = 0
    line_digest = hashlib.sha256()
    captured = bytearray()

    def consume(piece: bytes) -> None:
        nonlocal line_bytes
        if not piece:
            return
        line_bytes += len(piece)
        line_digest.update(piece)
        remaining = parse_limit_bytes + 1 - len(captured)
        if remaining > 0:
            captured.extend(piece[:remaining])

    def finish_line() -> None:
        nonlocal captured, line_bytes, line_digest
        nonlocal largest_line_bytes, largest_line_number

        is_empty = line_bytes == 0 or bytes(captured).strip() == b""
        if is_empty:
            counts["empty_lines"] += 1
            status = "empty"
        elif line_bytes > parse_limit_bytes:
            counts["unparsed_oversize"] += 1
            status = "unparsed_oversize"
        else:
            try:
                json.loads(bytes(captured))
            except (json.JSONDecodeError, UnicodeDecodeError):
                counts["invalid_json"] += 1
                status = "invalid_json"
            else:
                counts["valid_json"] += 1
                status = "valid_json"

        if line_bytes > 0 and not is_empty:
            counts["records"] += 1
            if line_bytes > max_record_chars:
                counts["raw_oversize_lines"] += 1
                counts["raw_oversize_bytes"] += line_bytes
                oversized_lines.append(
                    _line_entry(
                        line_number=line_number,
                        line_bytes=line_bytes,
                        line_sha256=line_digest.hexdigest(),
                        status=status,
                    )
                )
            if line_bytes > largest_line_bytes:
                largest_line_bytes = line_bytes
                largest_line_number = line_number

        captured = bytearray()
        line_bytes = 0
        line_digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            scanned_bytes += len(chunk)
            cursor = 0
            while True:
                newline = chunk.find(b"\n", cursor)
                if newline < 0:
                    consume(chunk[cursor:])
                    break
                consume(chunk[cursor:newline])
                finish_line()
                line_number += 1
                cursor = newline + 1

    if line_bytes or captured:
        finish_line()

    stat = path.stat()
    stable = stat.st_size == scanned_bytes
    result: dict[str, Any] = {
        "name": path.name,
        "sha256": digest.hexdigest(),
        "bytes": scanned_bytes,
        "stable_size": stable,
        "parser_status": "complete" if stable else "changed_during_scan",
        "counts": dict(sorted(counts.items())),
        "largest_line": {
            "bytes": largest_line_bytes,
            "line": largest_line_number,
        },
        "oversized_lines": oversized_lines,
    }
    if not stable:
        result["error"] = "file size changed during scan; rerun against a stable snapshot"
    return result


def preflight_sources(
    inputs: Iterable[Path],
    *,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
    parse_limit_bytes: int = DEFAULT_PARSE_LIMIT_BYTES,
    excluded: Path | None = None,
) -> dict[str, Any]:
    files = discover_line_files(inputs, excluded=excluded)
    file_reports: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            report = scan_jsonl(
                path,
                max_record_chars=max_record_chars,
                parse_limit_bytes=parse_limit_bytes,
            )
        except (OSError, ValueError) as exc:
            errors.append({"name": path.name, "error": str(exc)})
            continue
        file_reports.append(report)
        totals.update(report["counts"])
        if report["parser_status"] != "complete":
            errors.append({"name": path.name, "error": report["error"]})

    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "preflight_version": PREFLIGHT_VERSION,
        "generated_at": utc_now(),
        "status": "complete" if not errors else "incomplete",
        "policy": {
            "max_record_chars": max_record_chars,
            "parse_limit_bytes": parse_limit_bytes,
            "raw_size_comparison": "raw UTF-8 bytes versus normalized character limit; conservative hint",
            "oversize_parse_policy": "do not retain or parse lines above parse_limit_bytes",
            "content_policy": "metadata_only",
        },
        "counts": {
            "files": len(file_reports),
            **dict(sorted(totals.items())),
            "errors": len(errors),
        },
        "inputs": file_reports,
        "errors": errors,
    }


def write_manifest(path: Path, manifest: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure JSONL source lines without normalizing or exporting content."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/source_preflight.json"),
        help="Metadata manifest path (default: .tmp/source_preflight.json)",
    )
    parser.add_argument(
        "--max-record-chars",
        type=int,
        default=DEFAULT_MAX_RECORD_CHARS,
        help="Builder normalized-size threshold used for a conservative raw-line flag",
    )
    parser.add_argument(
        "--parse-limit-bytes",
        type=int,
        default=DEFAULT_PARSE_LIMIT_BYTES,
        help="Do not retain or parse a raw line above this byte size",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = preflight_sources(
            args.inputs,
            max_record_chars=args.max_record_chars,
            parse_limit_bytes=args.parse_limit_bytes,
            excluded=args.output,
        )
        write_manifest(args.output, manifest, overwrite=args.overwrite)
    except (FileNotFoundError, OSError, ValueError, FileExistsError) as exc:
        print(f"Preflight failed: {exc}")
        return 2

    counts = manifest["counts"]
    print(f"Files: {counts['files']}")
    print(f"Non-empty records: {counts.get('records', 0)}")
    print(f"Valid JSON records: {counts.get('valid_json', 0)}")
    print(f"Unparsed oversize records: {counts.get('unparsed_oversize', 0)}")
    print(f"Raw-oversize records: {counts.get('raw_oversize_lines', 0)}")
    print(f"Manifest: {args.output}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
