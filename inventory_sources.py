#!/usr/bin/env python3
"""Inventory local assistant stores without exporting their content.

The inventory is intentionally metadata-only. It classifies source files so a
large cache, workspace snapshot, backup, or package tree cannot be mistaken
for conversation training data. Candidate session files are hashed by default;
the manifest never records absolute paths or path text from a source store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

INVENTORY_SCHEMA = "ai-data-extraction/source-inventory/v1"
INVENTORY_VERSION = "1.0.0"


@dataclass(frozen=True)
class RootSpec:
    provider: str
    label: str
    path: Path


def discover_roots(home: Path | None = None) -> list[RootSpec]:
    home = (home or Path.home()).expanduser()
    system = platform.system()
    roots: list[RootSpec] = []

    def add(provider: str, label: str, path: Path) -> None:
        path = path.expanduser()
        if path.exists() and path.is_dir():
            roots.append(RootSpec(provider, label, path))

    add("codex", "primary", home / ".codex")
    add("codex", "local", home / ".codex-local")

    for index, name in enumerate(
        (".claude", ".claude-code", ".claude-local", ".claude-m2", ".claude-zai")
    ):
        add("claude", f"root-{index}", home / name)

    add("gemini", "primary", home / ".gemini")
    add("continue", "primary", home / ".continue")
    add("prime-agent", "primary", home / ".prime")
    add("oh-my-pi", "primary", home / ".omp")
    add("oh-my-pi", "backups", home / ".omp-backups")
    add("trae", "home", home / ".trae")
    add("windsurf", "home", home / ".windsurf")
    add("cursor", "home", home / ".cursor")

    if system == "Darwin":
        add("cursor", "app-support", home / "Library/Application Support/Cursor")
        add("trae", "app-support", home / "Library/Application Support/Trae")
        add("windsurf", "app-support", home / "Library/Application Support/Windsurf")
        add("opencode", "cli", home / "Library/Application Support/opencode")
        add("opencode", "desktop", home / "Library/Application Support/ai.opencode.app")
    elif system == "Windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        add("cursor", "appdata", appdata / "Cursor")
        add("trae", "appdata", appdata / "Trae")
        add("windsurf", "appdata", appdata / "Windsurf")
        add("opencode", "cli", appdata / "opencode")
        add("opencode", "desktop", appdata / "ai.opencode.app")
    else:
        data_home = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share"))
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
        add("opencode", "cli", data_home / "opencode")
        add("opencode", "desktop", data_home / "ai.opencode.app")
        add("cursor", "config", config_home / "Cursor")
        add("windsurf", "config", config_home / "Windsurf")

    unique: dict[Path, RootSpec] = {}
    for root in roots:
        try:
            resolved = root.path.resolve()
        except OSError:
            resolved = root.path.absolute()
        unique.setdefault(resolved, root)
    return list(unique.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def classify_file(provider: str, relative: Path) -> tuple[str, str]:
    """Return (source_class, treatment) without inspecting file contents."""
    path = relative.as_posix().lower()
    wrapped_path = f"/{path}/"
    name = relative.name.lower()

    if provider == "codex":
        if name.startswith("rollout-") and name.endswith(".jsonl.backup"):
            return "session_backup", "candidate"
        if name.startswith("rollout-") and name.endswith(".jsonl"):
            return "session_active", "candidate"
        if "/.backups/" in wrapped_path or name.endswith(".backup"):
            return "backup_state", "review"
        if path.startswith("sessions/"):
            return "session_auxiliary", "review"
        if "/attachments/" in wrapped_path:
            return "attachment", "review"
        if any(
            token in wrapped_path
            for token in (
                "/cache/",
                "/tmp/",
                "/packages/",
                "/plugins/",
                "/memories/",
                "/skills/",
                "/rules/",
                "/shell_snapshots/",
                "/log/",
                "/policy/",
                "/mcp-oauth-locks/",
                "/thread-writer-locks/",
            )
        ):
            if "/packages/" in wrapped_path:
                return "binary_or_package", "excluded"
            return "support_state", "excluded"

    if provider == "claude":
        if name.endswith(".jsonl") and ("/projects/" in f"/{path}" or "/sessions/" in f"/{path}"):
            if name.startswith("agent-"):
                return "subagent_session", "candidate"
            return "session_active", "candidate"
        if any(token in path for token in ("/backups/", "/cache/", "/logs/", "/debug/")):
            return "support_state", "excluded"

    if provider == "gemini":
        if "/tmp/" in f"/{path}" and "/chats/" in f"/{path}" and name.endswith((".json", ".jsonl")):
            return "session_active", "candidate"
        if "/history/" in f"/{path}":
            return "support_state", "review"

    if provider == "continue":
        if "/sessions/" in f"/{path}" and name.endswith((".json", ".jsonl")):
            return "session_active", "candidate"
        if any(token in path for token in ("/cache/", "/logs/")):
            return "support_state", "excluded"

    if provider == "opencode":
        if path == "opencode.db" or "/storage/message/" in f"/{path}" or "/storage/part/" in f"/{path}":
            return "conversation_store", "candidate"
        if "/storage/session/" in f"/{path}" or "/storage/session_diff/" in f"/{path}":
            return "conversation_sidecar", "candidate"
        if "/tool-output/" in f"/{path}":
            return "tool_output", "candidate"
        if "/snapshot/" in f"/{path}":
            return "workspace_snapshot", "review"
        if "/bin/" in f"/{path}" or "/node_modules/" in f"/{path}":
            return "binary_or_package", "excluded"
        if "/log/" in f"/{path}":
            return "support_state", "excluded"

    if provider == "prime-agent":
        if path.startswith("agent/sessions/") and name.endswith(".jsonl"):
            return "session_active", "candidate"
        if path.startswith("agent/session-artifacts/") and name.endswith(".jsonl"):
            return "subagent_session", "candidate"
        if any(
            token in wrapped_path
            for token in (
                "/agent/logs/",
                "/agent/command-journal/",
                "/agent/recovery/",
                "/agent/harness/",
                "/agent/extensions/",
            )
        ):
            return "harness_state", "review"
        if name.endswith((".dill", ".db", ".sqlite", ".sqlite3")):
            return "agent_state", "review"

    if provider == "oh-my-pi":
        if "/agent/sessions/" in wrapped_path and name.endswith(".jsonl"):
            if name == "__advisor.jsonl":
                return "advisor_overlay", "candidate"
            return (
                "session_backup" if ".omp-backups/" in wrapped_path else "session_active",
                "candidate",
            )
        if "/agent/" in wrapped_path and name.endswith((".db", ".sqlite", ".sqlite3")):
            return "agent_database", "review"
        if any(
            token in wrapped_path
            for token in (
                "/logs/",
                "/natives/",
                "/webcache/",
                "/node_modules/",
                "/extensions/",
            )
        ):
            return "support_state", "excluded"

    if provider == "cursor":
        if name.endswith((".db", ".vscdb")) or "cursordiskkv" in name:
            return "conversation_store", "candidate"

    if provider in {"trae", "windsurf"}:
        if name.endswith((".db", ".vscdb", ".json", ".jsonl")):
            return "conversation_store", "candidate"

    if name.endswith(".jsonl"):
        return "jsonl_candidate", "review"
    if name.endswith((".json", ".db", ".sqlite", ".sqlite3")):
        return "structured_candidate", "review"
    if name.endswith((".log", ".lock", ".tmp")):
        return "support_state", "excluded"
    return "unknown", "review"


def classify_root_file(
    provider: str,
    root_label: str,
    relative: Path,
) -> tuple[str, str]:
    """Classify a file with root-level provenance normalization applied."""

    source_class, treatment = classify_file(provider, relative)
    if (
        provider == "oh-my-pi"
        and root_label == "backups"
        and source_class == "session_active"
    ):
        source_class = "session_backup"
    return source_class, treatment


def iter_files(root: Path, *, excluded: Path | None = None) -> Iterable[Path]:
    """Walk a root without following symlinks and without reading contents."""
    excluded_resolved = excluded.resolve() if excluded and excluded.exists() else None
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if excluded_resolved is not None:
            kept_dirs: list[str] = []
            for dirname in dirnames:
                candidate = (directory_path / dirname).resolve()
                if candidate != excluded_resolved and excluded_resolved not in candidate.parents:
                    kept_dirs.append(dirname)
            dirnames[:] = kept_dirs
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink():
                continue
            if excluded_resolved is not None:
                try:
                    if path.resolve() == excluded_resolved or excluded_resolved in path.resolve().parents:
                        continue
                except OSError:
                    continue
            yield path


def inventory_roots(
    roots: Iterable[RootSpec],
    *,
    output: Path | None = None,
    hash_mode: str = "candidates",
) -> dict[str, object]:
    if hash_mode not in {"none", "candidates", "all"}:
        raise ValueError("hash_mode must be none, candidates, or all")
    roots = list(roots)
    output_resolved = output.resolve() if output and output.parent.exists() else None
    files: list[dict[str, object]] = []
    aggregates: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, str]] = []

    for root in roots:
        try:
            root_resolved = root.path.resolve()
        except OSError as exc:
            errors.append({"root": root.label, "error": str(exc)})
            continue
        for path in iter_files(root_resolved, excluded=output_resolved):
            try:
                relative = path.relative_to(root_resolved)
                size = path.stat().st_size
            except OSError as exc:
                errors.append({"root": root.label, "error": str(exc)})
                continue
            source_class, treatment = classify_root_file(root.provider, root.label, relative)
            should_hash = hash_mode == "all" or (hash_mode == "candidates" and treatment == "candidate")
            entry: dict[str, object] = {
                "provider": root.provider,
                "root_label": root.label,
                "relative_path_sha256": sha256_text(relative.as_posix()),
                "extension": path.suffix.lower(),
                "source_class": source_class,
                "treatment": treatment,
                "bytes": size,
                "parser_status": "not_parsed",
            }
            if should_hash:
                try:
                    entry["file_sha256"] = sha256_file(path)
                except OSError as exc:
                    entry["parser_status"] = "hash_error"
                    errors.append({"root": root.label, "error": str(exc)})
            files.append(entry)
            aggregates[root.provider]["files"] += 1
            aggregates[root.provider]["bytes"] += size
            aggregates[f"{root.provider}:{source_class}"]["files"] += 1
            aggregates[f"{root.provider}:{source_class}"]["bytes"] += size

    provider_summary: dict[str, dict[str, int]] = {}
    for key, counts in sorted(aggregates.items()):
        if ":" in key:
            continue
        provider_summary[key] = dict(counts)
    class_summary = {
        key: dict(counts)
        for key, counts in sorted(aggregates.items())
        if ":" in key
    }
    manifest: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA,
        "inventory_version": INVENTORY_VERSION,
        "hash_mode": hash_mode,
        "privacy": "metadata_only_no_source_content",
        "roots": [
            {"provider": root.provider, "label": root.label, "present": root.path.exists()}
            for root in roots
        ],
        "counts": {
            "files": len(files),
            "bytes": sum(int(entry["bytes"]) for entry in files),
            "hashed_files": sum("file_sha256" in entry for entry in files),
            "errors": len(errors),
        },
        "providers": provider_summary,
        "classes": class_summary,
        "errors": errors,
        "files": sorted(
            files,
            key=lambda entry: (
                str(entry["provider"]),
                str(entry["root_label"]),
                str(entry["relative_path_sha256"]),
            ),
        ),
    }
    return manifest


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("source_inventory.json"),
        help="metadata-only JSON manifest path",
    )
    parser.add_argument(
        "--hash-mode",
        choices=("none", "candidates", "all"),
        default="candidates",
        help="hash no files, candidate session evidence, or every file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = discover_roots()
    manifest = inventory_roots(roots, output=args.output, hash_mode=args.hash_mode)
    write_manifest(args.output, manifest)
    counts = manifest["counts"]
    print(
        f"Inventory wrote {args.output}: {counts['files']} files, "
        f"{counts['bytes']} bytes, {counts['hashed_files']} hashes, "
        f"{counts['errors']} errors"
    )
    return 0 if counts["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
