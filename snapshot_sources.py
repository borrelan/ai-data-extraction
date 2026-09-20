#!/usr/bin/env python3
"""Create an immutable, metadata-bound snapshot for assistant-source ingress.

Live assistant stores are append-only only by convention: the active session
and database-backed stores can change while an extractor is reading them.  A
raw manifest from one read cannot therefore be compared with a later live
inventory.  This command copies only files classified as candidate source
evidence into a synthetic ``HOME`` tree, verifies that each source file stayed
stable during the copy, and writes a metadata-only snapshot manifest.

Existing provider adapters remain the parsers.  Run them with ``HOME`` set to
the emitted ``home`` directory so their normal discovery paths operate against
one immutable source state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inventory_sources import RootSpec, classify_root_file, discover_roots, iter_files

SNAPSHOT_SCHEMA = "ai-data-extraction/source-snapshot/v1"
SNAPSHOT_VERSION = "1.0.0"
DEFAULT_RETRIES = 3
COPY_CHUNK_BYTES = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_stat(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _synthetic_root(root: RootSpec, source_home: Path, snapshot_home: Path) -> Path:
    """Map a live root below source_home into the synthetic HOME tree."""

    source_home = source_home.resolve()
    root_path = root.path.resolve()
    try:
        relative_root = root_path.relative_to(source_home)
    except ValueError:
        relative_root = Path("roots") / root.provider / root.label
    return snapshot_home / relative_root


def _safe_snapshot_path(path: Path, snapshot_home: Path) -> Path:
    candidate = path.resolve()
    root = snapshot_home.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"snapshot path escapes synthetic HOME: {path}")
    return candidate


def _copy_with_digest(source: Path, temporary: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as origin, temporary.open("wb") as destination:
        while True:
            chunk = origin.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            destination.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    return digest.hexdigest(), copied


def _stable_copy(
    source: Path,
    target: Path,
    *,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Copy one file only when its size/mtime is stable for the copy pass."""

    if retries < 1:
        raise ValueError("retries must be positive")
    target.parent.mkdir(parents=True, exist_ok=True)
    last_before: tuple[int, int] | None = None
    last_after: tuple[int, int] | None = None

    for attempt in range(1, retries + 1):
        before = _file_stat(source)
        last_before = before
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{source.name}.", suffix=".snapshot.tmp", dir=target.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            digest, copied_bytes = _copy_with_digest(source, temporary)
            after = _file_stat(source)
            last_after = after
            if before != after or copied_bytes != after[0]:
                temporary.unlink(missing_ok=True)
                continue
            os.replace(temporary, target)
            return {
                "status": "stable",
                "attempts": attempt,
                "bytes": copied_bytes,
                "source_size_before": before[0],
                "source_size_after": after[0],
                "source_mtime_before_ns": before[1],
                "source_mtime_after_ns": after[1],
                "sha256": digest,
            }
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    return {
        "status": "unstable",
        "attempts": retries,
        "bytes": 0,
        "source_size_before": last_before[0] if last_before else None,
        "source_size_after": last_after[0] if last_after else None,
        "source_mtime_before_ns": last_before[1] if last_before else None,
        "source_mtime_after_ns": last_after[1] if last_after else None,
        "sha256": None,
        "reason": "source_changed_during_snapshot_copy",
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(value, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def snapshot_roots(
    roots: Iterable[RootSpec],
    *,
    source_home: Path,
    output_dir: Path,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Snapshot candidate files from roots into ``output_dir/home``."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"snapshot output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_home = output_dir / "home"
    snapshot_home.mkdir(parents=True, exist_ok=True)
    source_home = source_home.resolve()
    roots = list(roots)
    entries: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    bytes_by_status: Counter[str] = Counter()

    for root in roots:
        synthetic_root = _synthetic_root(root, source_home, snapshot_home)
        for source in iter_files(root.path):
            relative = source.relative_to(root.path)
            source_class, treatment = classify_root_file(root.provider, root.label, relative)
            if treatment != "candidate":
                continue
            target = _safe_snapshot_path(synthetic_root / relative, snapshot_home)
            copy_result = _stable_copy(source, target, retries=retries)
            counts[copy_result["status"]] += 1
            bytes_by_status[copy_result["status"]] += copy_result["bytes"]
            entries.append(
                {
                    "provider": root.provider,
                    "root_label": root.label,
                    "source_class": source_class,
                    "treatment": treatment,
                    "relative_path_sha256": sha256_text(relative.as_posix()),
                    "snapshot_relative_path_sha256": sha256_text(
                        target.relative_to(output_dir).as_posix()
                    ),
                    **copy_result,
                }
            )

    stable_hashes = sorted(
        entry["sha256"] for entry in entries if entry["status"] == "stable"
    )
    snapshot_revision = sha256_text("\n".join(stable_hashes))
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "snapshot_version": SNAPSHOT_VERSION,
        "generated_at": utc_now(),
        "source_home": "live_home",
        "snapshot_home": "home",
        "privacy": "metadata_only_no_source_content_or_absolute_paths",
        "policy": {
            "included_treatment": "candidate",
            "stability_check": "size_and_mtime_before_after_copy",
            "retries": retries,
            "unstable_files_are_not_published": True,
        },
        "snapshot_revision": snapshot_revision,
        "roots": [
            {
                "provider": root.provider,
                "label": root.label,
                "relative_root_sha256": sha256_text(
                    root.path.resolve().relative_to(source_home).as_posix()
                )
                if root.path.resolve().is_relative_to(source_home)
                else None,
            }
            for root in roots
        ],
        "counts": {
            "candidate_files": len(entries),
            "stable_files": counts["stable"],
            "unstable_files": counts["unstable"],
            "stable_bytes": bytes_by_status["stable"],
            "unstable_bytes": bytes_by_status["unstable"],
        },
        "files": entries,
    }
    _atomic_json(output_dir / "source_snapshot_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new, empty output directory containing the synthetic HOME snapshot",
    )
    parser.add_argument(
        "--source-home",
        type=Path,
        default=Path.home(),
        help="live home whose discovered provider roots should be copied",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="stable-copy attempts per candidate file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = discover_roots(args.source_home)
    if not roots:
        raise SystemExit("no supported provider roots found")
    manifest = snapshot_roots(
        roots,
        source_home=args.source_home,
        output_dir=args.output_dir,
        retries=args.retries,
    )
    counts = manifest["counts"]
    print(
        f"Snapshot wrote {args.output_dir}: {counts['stable_files']} stable candidate files, "
        f"{counts['stable_bytes']} bytes, {counts['unstable_files']} unstable files"
    )
    return 0 if counts["unstable_files"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
