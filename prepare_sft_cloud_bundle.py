#!/usr/bin/env python3
"""Create a deterministic private cloud bundle without copying base-model weights."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


RUNTIME_FILES = (
    "dataset.py",
    "preflight.py",
    "runpod_remote.sh",
    "runtime_contract.json",
    "smoke.py",
    "tool_schemas.json",
    "train.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def add_tree(archive: tarfile.TarFile, root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        arcname = Path("pilot") / path.relative_to(root)
        archive.add(path, arcname=arcname.as_posix(), recursive=False, filter=normalized_tar_info)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    release = args.release.resolve()
    preflight_path = args.preflight.resolve()
    acquisition = args.acquisition_manifest.resolve()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"cloud bundle output already exists: {output}")
    release_manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    release_sha = sha256_file(release / "manifest.json")
    if preflight.get("status") != "passed" or preflight.get("release", {}).get("manifest_sha256") != release_sha:
        raise ValueError("preflight does not bind the selected release")

    stage_parent = repo_root / ".tmp"
    stage_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="sft-cloud-bundle-", dir=stage_parent))
    try:
        runtime_target = staging / "runtime" / "sft"
        runtime_target.mkdir(parents=True)
        for name in RUNTIME_FILES:
            shutil.copy2(repo_root / "runtime" / "sft" / name, runtime_target / name)
        shutil.copytree(release, staging / "input")
        artifacts = staging / "artifacts"
        artifacts.mkdir()
        shutil.copy2(acquisition, artifacts / acquisition.name)
        shutil.copy2(preflight_path, artifacts / "dataset-preflight.json")
        source_files = {
            path.relative_to(staging).as_posix(): descriptor(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        bundle_manifest = {
            "schema_version": "ai-data-extraction/sft-cloud-bundle/v1",
            "status": "ready",
            "base_model": {
                "repo_id": "Qwen/Qwen3.5-9B",
                "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "weights_in_bundle": False,
            },
            "release_manifest_sha256": release_sha,
            "release_counts": release_manifest["counts"],
            "preflight_sha256": sha256_file(preflight_path),
            "files": source_files,
        }
        manifest_path = staging / "bundle-manifest.json"
        manifest_path.write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(f".{output.name}.partial")
        with temporary_output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    add_tree(archive, staging)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_output, output)
        result = {
            **bundle_manifest,
            "archive": {"path": str(output), **descriptor(output)},
        }
        sidecar = output.with_suffix(output.suffix + ".manifest.json")
        sidecar.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
