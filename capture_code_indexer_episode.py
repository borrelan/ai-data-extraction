#!/usr/bin/env python3
"""Capture one real read-only Code Indexer CLI call as a harness episode.

The CLI is the execution boundary.  This adapter records the exact installed
binary digest, callable schema, required skill reads, bounded result, and
verifier-backed terminal state, then projects the episode through the existing
review-only tool-SFT join.  It does not infer a provider schema or create a
reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from build_training_data import SCHEMA_VERSION
from harness_trace import HarnessTrace
from harness_tool_projection import project_harness_tool_sft


CAPTURE_SCHEMA = "ai-data-extraction/runtime-tool-capture/v1"
TOOL_NAME = "code-indexer.search"
QUERY = "build_session_quality_index"
DEFAULT_LIMIT = 5
DEFAULT_BINARY = Path("/home/borrelan/.local/bin/code-indexer")
REQUIRED_SKILLS = (
    "code-indexer-ops",
    "core-principles",
)
SKILL_PATHS = {
    "code-indexer-ops": Path("/home/borrelan/.codex/skills/personal/code-indexer-ops/SKILL.md"),
    "core-principles": Path("/home/borrelan/.codex/skills/personal/core-principles/SKILL.md"),
}
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        "query": {"type": "string"},
        "root": {"type": "string"},
    },
    "required": ["query", "root"],
}
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "description": "Search the repository for code definitions.",
        "name": TOOL_NAME,
        "parameters": TOOL_PARAMETERS,
    },
}


class CaptureError(ValueError):
    """Raised when a runtime capture cannot prove a trainer-safe episode."""


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


def prefixed_sha256(value: bytes) -> str:
    return "sha256:" + sha256_bytes(value)


def _run_checked(args: list[str], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(f"command timed out after {timeout}s: {args[0]}") from exc
    return result


def _git_revision(project_root: Path) -> str:
    result = _run_checked(
        ["git", "rev-parse", "HEAD"], cwd=project_root, timeout=5.0
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise CaptureError("could not resolve project Git revision")
    return result.stdout.strip()


def _git_worktree_identity(project_root: Path, *, head: str) -> dict[str, Any]:
    """Bind dirty tracked and untracked content without copying it into output."""

    status = _run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root,
        timeout=10.0,
    )
    diff = _run_checked(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=project_root,
        timeout=10.0,
    )
    untracked = _run_checked(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=project_root,
        timeout=10.0,
    )
    if status.returncode != 0 or diff.returncode != 0 or untracked.returncode != 0:
        raise CaptureError("could not compute Git worktree identity")

    untracked_files: list[dict[str, Any]] = []
    for relative in sorted(item for item in untracked.stdout.split("\0") if item):
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise CaptureError(f"untracked path escapes project root: {relative}") from exc
        if not path.is_file():
            raise CaptureError(f"untracked path is not a regular file: {relative}")
        untracked_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "head": head,
        "dirty": bool(status.stdout.strip() or diff.stdout),
        "status_sha256": sha256_bytes(status.stdout.encode("utf-8")),
        "tracked_diff_sha256": sha256_bytes(diff.stdout.encode("utf-8")),
        "tracked_diff_bytes": len(diff.stdout.encode("utf-8")),
        "untracked_files_sha256": prefixed_sha256(canonical_json(untracked_files)),
        "untracked_file_count": len(untracked_files),
    }


def _code_indexer_status(
    *,
    binary: Path,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Capture the exact pre-search published/index status and its hashes."""

    result = _run_checked(
        [str(binary), "status", "--root", "."],
        cwd=project_root,
        timeout=10.0,
    )
    raw_bytes = result.stdout.encode("utf-8")
    raw_sha256 = sha256_bytes(raw_bytes)
    if result.returncode != 0:
        raise CaptureError(
            f"Code Indexer status failed with exit {result.returncode}: "
            + result.stderr.strip()[:500]
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CaptureError("Code Indexer status did not return JSON") from exc
    if not isinstance(value, dict):
        raise CaptureError("Code Indexer status result must be an object")
    sanitized = _sanitize_cli_result(value, project_root)
    sanitized_bytes = canonical_json(sanitized)
    published = value.get("published_view")
    if not isinstance(published, dict):
        published = {}
    sanitized_file_sha256 = sha256_bytes(sanitized_bytes + b"\n")
    identity = {
        "capability_stage": value.get("capability_stage"),
        "chunks": value.get("chunks"),
        "edges": value.get("edges"),
        "embedded_chunks": value.get("embedded_chunks"),
        "symbols": value.get("symbols"),
        "semantic": value.get("semantic"),
        "published_complete": published.get("complete"),
        "published_revision": value.get("published_revision", published.get("revision")),
        "published_fence": published.get("fence"),
        "worktree_id": value.get("worktree_id") or published.get("members"),
        "raw_status_sha256": raw_sha256,
        "sanitized_status_sha256": sanitized_file_sha256,
    }
    return sanitized, identity, raw_sha256, sanitized_file_sha256


def _read_skill_records() -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for skill in REQUIRED_SKILLS:
        path = SKILL_PATHS[skill]
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CaptureError(f"required skill could not be read: {skill}") from exc
        if not raw:
            raise CaptureError(f"required skill is empty: {skill}")
        digest = sha256_bytes(raw)
        records.append(
            {
                "skill": skill,
                "skill_revision": "sha256:" + digest,
                "content_sha256": "sha256:" + digest,
                "path_label": f".codex/skills/{skill}/SKILL.md",
            }
        )
    bundle = prefixed_sha256(canonical_json(records))
    return records, "runtime-skill-bundle/v1:" + bundle


def _sanitize_cli_result(value: Any, project_root: Path) -> Any:
    """Remove the absolute checkout path while preserving the result shape."""

    absolute_root = str(project_root.resolve())
    if isinstance(value, str):
        return value.replace(absolute_root, "<PROJECT_ROOT>")
    if isinstance(value, list):
        return [_sanitize_cli_result(item, project_root) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_cli_result(item, project_root)
            for key, item in value.items()
        }
    return value


def _extract_definition(
    result: dict[str, Any],
    *,
    query: str = QUERY,
) -> tuple[str, str, int]:
    hits = result.get("hits")
    if not isinstance(hits, list):
        raise CaptureError("Code Indexer result has no hits list")
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get("source")
        chunk = source.get("chunk") if isinstance(source, dict) else None
        if not isinstance(chunk, dict):
            chunk = hit
        symbol = chunk.get("symbol_name") or chunk.get("symbol")
        file_path = chunk.get("file_path") or chunk.get("file")
        start_line = chunk.get("start_line")
        if (
            symbol == query
            and isinstance(file_path, str)
            and file_path
            and isinstance(start_line, int)
            and start_line > 0
        ):
            return symbol, file_path, start_line
    raise CaptureError(f"Code Indexer result has no exact definition hit for {query}")


def _tool_registry_revision(binary_sha256: str) -> str:
    contract = {
        "binary_sha256": binary_sha256,
        "tool_name": TOOL_NAME,
        "parameters": TOOL_PARAMETERS,
        "cli_surface": "code-indexer search <query> --root <root> --limit <limit>",
    }
    return "code-indexer-cli-contract/v1:" + prefixed_sha256(canonical_json(contract))


def _capture_identity(
    *,
    binary_sha256: str,
    project_revision: str,
    raw_result_sha256: str,
    sanitized_result_sha256: str,
    registry_revision: str,
    query: str,
    limit: int,
    worktree_identity: dict[str, Any],
    runtime_status_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "binary_sha256": binary_sha256,
        "project_revision": project_revision,
        "raw_result_sha256": raw_result_sha256,
        "sanitized_result_sha256": sanitized_result_sha256,
        "registry_revision": registry_revision,
        "tool_name": TOOL_NAME,
        "query": query,
        "limit": limit,
        "worktree_identity": worktree_identity,
        "runtime_status_identity": runtime_status_identity,
    }


def build_episode_and_trace(
    *,
    result: dict[str, Any],
    raw_result_sha256: str,
    sanitized_result_sha256: str,
    binary_sha256: str,
    project_revision: str,
    skill_records: Iterable[dict[str, Any]],
    skill_revision: str,
    verifier_revision: str,
    project_root: Path,
    query: str = QUERY,
    limit: int = DEFAULT_LIMIT,
    worktree_identity: dict[str, Any] | None = None,
    runtime_status_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], HarnessTrace, dict[str, Any]]:
    """Build one canonical source row and its execution trace from a real result."""

    if not query.strip():
        raise CaptureError("Code Indexer query must not be empty")
    if not 1 <= limit <= 200:
        raise CaptureError("Code Indexer limit must be between 1 and 200")
    worktree_identity = dict(worktree_identity or {})
    runtime_status_identity = dict(runtime_status_identity or {})
    symbol, file_path, start_line = _extract_definition(result, query=query)
    registry_revision = _tool_registry_revision(binary_sha256)
    capture_identity = _capture_identity(
        binary_sha256=binary_sha256,
        project_revision=project_revision,
        raw_result_sha256=raw_result_sha256,
        sanitized_result_sha256=sanitized_result_sha256,
        registry_revision=registry_revision,
        query=query,
        limit=limit,
        worktree_identity=worktree_identity,
        runtime_status_identity=runtime_status_identity,
    )
    parent_id = prefixed_sha256(canonical_json(capture_identity))
    call_id = "call-" + parent_id.removeprefix("sha256:")[:24]
    arguments = {"limit": limit, "query": query, "root": "."}
    tool_content = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    final_answer = f"The function `{symbol}` is defined at `{file_path}:{start_line}`."
    runtime_status_evidence_sha = runtime_status_identity.get(
        "sanitized_status_sha256"
    ) or prefixed_sha256(canonical_json(runtime_status_identity))

    trace = HarnessTrace(
        episode_id=parent_id,
        source={
            "channel": "cli",
            "provider": "code-indexer",
            "repository": "ai-data-extraction",
            "capture_kind": "runtime_read_only",
            "binary_sha256": binary_sha256,
            "project_revision": project_revision,
            "raw_result_sha256": raw_result_sha256,
            "sanitized_result_sha256": sanitized_result_sha256,
            "worktree_identity": worktree_identity,
            "runtime_status_identity": runtime_status_identity,
        },
        registry_revision=registry_revision,
        skill_revision=skill_revision,
        environment_revision="code-indexer-runtime/v1:" + prefixed_sha256(
            canonical_json(
                {
                    "binary_sha256": binary_sha256,
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "project_revision": project_revision,
                    "worktree_identity": worktree_identity,
                    "runtime_status_identity": runtime_status_identity,
                }
            )
        ),
        verifier_revision=verifier_revision,
        privacy_state="heuristic",
        required_skills=REQUIRED_SKILLS,
    )
    for skill in skill_records:
        trace.record_skill_preflight(
            skill=skill["skill"],
            skill_revision=skill["skill_revision"],
            mandatory=True,
            trigger="read-only code-indexer search",
            read_result="read",
            scope_decision="in_scope",
            content_sha256=skill["content_sha256"],
        )
    trace.record_tool_registry(
        registry_revision=registry_revision,
        tools=[
            {
                "name": TOOL_NAME,
                "schema": TOOL_PARAMETERS,
                "trust_class": "local_read_only_indexer",
                "side_effect_class": "read_only",
            }
        ],
    )
    trace.record_decision(
        decision="use",
        decision_basis="definition lookup requires indexed repository evidence",
        required_gate_status="passed",
        tool_name=TOOL_NAME,
    )
    trace.record_tool_call(
        tool_name=TOOL_NAME,
        call_id=call_id,
        arguments=arguments,
        permission_decision="not_required",
    )
    trace.record_tool_observation(
        call_id=call_id,
        status="success",
        output=tool_content,
    )
    trace.record_verification(
        verifier_revision=verifier_revision,
        result="pass",
        checks=[
            {"name": "process_exit", "result": "pass", "exit_code": 0},
            {"name": "json_result", "result": "pass"},
            {
                "name": "exact_definition_hit",
                "result": "pass",
                "symbol": symbol,
                "file": file_path,
                "start_line": start_line,
            },
            {"name": "project_root_redaction", "result": "pass"},
        ],
        durable_evidence=[
            {"kind": "cli_result", "name": "cli_result.json", "sha256": sanitized_result_sha256},
            {
                "kind": "runtime_status",
                "name": "runtime_status.json",
                "sha256": runtime_status_evidence_sha,
            },
            {"kind": "binary", "name": "code-indexer", "sha256": binary_sha256},
            {"kind": "project_revision", "value": project_revision},
            {
                "kind": "worktree_identity",
                "value": prefixed_sha256(canonical_json(worktree_identity)),
            },
        ],
    )
    trace.record_terminal(
        status="success",
        evidence=[{"kind": "verifier", "result": "pass"}],
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Find the definition of {query} with "
                "code-indexer.search. After the tool result, answer exactly one "
                "sentence containing the function name and precise file:line "
                "location. Do not omit the line number."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "arguments": arguments,
                        "name": TOOL_NAME,
                    },
                    "id": call_id,
                    "type": "function",
                }
            ],
        },
        {
            "role": "tool",
            "content": tool_content,
            "tool_call_id": call_id,
        },
        {"role": "assistant", "content": final_answer},
    ]
    example_seed = {
        "messages": messages,
        "parent_record_sha256": parent_id,
        "tools": [TOOL_SCHEMA],
    }
    example_id = prefixed_sha256(canonical_json(example_seed))
    record = {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": "tool_trace",
        "split": "train",
        "tags": [
            "capture:runtime",
            "outcome:verifier-passed",
            "tool-family:code-indexer",
            "tool:code-indexer.search",
        ],
        "quality": {
            "tier": "harness_verified_tool_sft",
            "source_quality": "runtime_cli_capture",
            "training_authorized": False,
        },
        "privacy": {
            "mode": "heuristic",
            "state": "heuristic",
            "redactions": ["absolute_project_root"],
        },
        "lineage": {
            "parent_record_sha256": parent_id,
            "capture_schema": CAPTURE_SCHEMA,
            "source": "code-indexer-cli",
            "project_revision": project_revision,
            "binary_sha256": binary_sha256,
            "raw_result_sha256": raw_result_sha256,
            "sanitized_result_sha256": sanitized_result_sha256,
            "worktree_identity": worktree_identity,
            "runtime_status_identity": runtime_status_identity,
            "registry_revision": registry_revision,
            "published_revision": result.get("published_revision"),
            "project_root": "<PROJECT_ROOT>",
            "query": query,
            "limit": limit,
        },
        "messages": messages,
        "tools": [TOOL_SCHEMA],
    }
    return record, trace, {
        "parent_record_sha256": parent_id,
        "example_id": example_id,
        "registry_revision": registry_revision,
        "definition": {"symbol": symbol, "file": file_path, "start_line": start_line},
    }


def _write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)
    return sha256_file(path)


def _write_json(path: Path, value: Any) -> str:
    return _write_bytes(path, canonical_json(value) + b"\n")


def capture_code_indexer_episode(
    *,
    project_root: Path,
    output_dir: Path,
    binary: Path = DEFAULT_BINARY,
    query: str = QUERY,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    binary = binary.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite capture directory: {output_dir}")
    if not project_root.is_dir() or not binary.is_file():
        raise CaptureError("project root and Code Indexer binary must exist")
    if not query.strip():
        raise CaptureError("Code Indexer query must not be empty")
    if not 1 <= limit <= 200:
        raise CaptureError("Code Indexer limit must be between 1 and 200")

    skill_records, skill_revision = _read_skill_records()
    project_revision = _git_revision(project_root)
    worktree_identity = _git_worktree_identity(project_root, head=project_revision)
    binary_sha256 = sha256_file(binary)
    (
        runtime_status,
        runtime_status_identity,
        runtime_status_raw_sha256,
        runtime_status_sanitized_sha256,
    ) = _code_indexer_status(binary=binary, project_root=project_root)
    command = [str(binary), "search", query, "--root", ".", "--limit", str(limit)]
    result = _run_checked(command, cwd=project_root, timeout=10.0)
    raw_stdout = result.stdout.encode("utf-8")
    raw_result_sha256 = sha256_bytes(raw_stdout)
    if result.returncode != 0:
        raise CaptureError(
            f"Code Indexer search failed with exit {result.returncode}: "
            + result.stderr.strip()[:500]
        )
    try:
        raw_value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CaptureError("Code Indexer search did not return JSON") from exc
    if not isinstance(raw_value, dict):
        raise CaptureError("Code Indexer search result must be an object")
    sanitized_value = _sanitize_cli_result(raw_value, project_root)
    sanitized_bytes = canonical_json(sanitized_value)
    sanitized_result_sha256 = sha256_bytes(sanitized_bytes)
    verifier_revision = "code-indexer-capture-verifier/v1:" + sha256_file(Path(__file__))

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        record, trace, details = build_episode_and_trace(
            result=sanitized_value,
            raw_result_sha256=raw_result_sha256,
            sanitized_result_sha256=sanitized_result_sha256,
            binary_sha256=binary_sha256,
            project_revision=project_revision,
            skill_records=skill_records,
            skill_revision=skill_revision,
            verifier_revision=verifier_revision,
            project_root=project_root,
            query=query,
            limit=limit,
            worktree_identity=worktree_identity,
            runtime_status_identity=runtime_status_identity,
        )
        episode_path = staging / "episode.jsonl"
        trace_path = staging / "trace.jsonl"
        _write_bytes(episode_path, canonical_json(record) + b"\n")
        trace.write_jsonl(trace_path)
        _write_json(staging / "cli_result.json", sanitized_value)
        _write_json(staging / "runtime_status.json", runtime_status)
        projection = project_harness_tool_sft(
            episode_path,
            trace_path,
            staging / "projection",
        )
        file_descriptors = {}
        for relative in (
            "episode.jsonl",
            "trace.jsonl",
            "cli_result.json",
            "runtime_status.json",
            "projection/tool_sft.jsonl",
            "projection/lineage.jsonl",
            "projection/manifest.json",
        ):
            path = staging / relative
            file_descriptors[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": CAPTURE_SCHEMA,
            "status": "review_only",
            "training_authorized": False,
            "trainer_projection": "review_only",
            "source": {
                "repository": "ai-data-extraction",
                "project_revision": project_revision,
                "binary_sha256": binary_sha256,
                "command": ["code-indexer", "search", query, "--root", ".", "--limit", limit],
                "exit_code": result.returncode,
                "raw_result_sha256": raw_result_sha256,
                "sanitized_result_sha256": sanitized_result_sha256,
                "published_revision": runtime_status_identity.get("published_revision"),
                "worktree_identity": worktree_identity,
                "runtime_status_identity": runtime_status_identity,
                "runtime_status_raw_sha256": runtime_status_raw_sha256,
                "runtime_status_sanitized_sha256": runtime_status_sanitized_sha256,
            },
            "contract": {
                "tool_name": TOOL_NAME,
                "registry_revision": details["registry_revision"],
                "skill_revision": skill_revision,
                "verifier_revision": verifier_revision,
                "required_skills": list(REQUIRED_SKILLS),
                "rewards": "not_exported",
            },
            "counts": {"episodes": 1, "projected_tool_sft": 1},
            "validation": {
                "actual_cli_invocation": "passed",
                "trace_terminal": "success",
                "registry_join": projection["validation"]["registry_join"],
                "skill_gate": projection["validation"]["skill_gate"],
                "call_observation_join": projection["validation"]["call_observation_join"],
                "verification": projection["validation"]["verification"],
                "privacy_reasoning_firewall": projection["validation"]["privacy_reasoning_firewall"],
                "reward": "not_present",
            },
            "definition": details["definition"],
            "query": query,
            "limit": limit,
            "files": file_descriptors,
        }
        _write_json(staging / "capture_manifest.json", manifest)
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "capture_manifest.json")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--query", default=QUERY)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    print(
        json.dumps(
            capture_code_indexer_episode(
                project_root=args.project_root,
                output_dir=args.output_dir,
                binary=args.binary,
                query=args.query,
                limit=args.limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPTURE_SCHEMA",
    "TOOL_NAME",
    "TOOL_PARAMETERS",
    "TOOL_SCHEMA",
    "CaptureError",
    "build_episode_and_trace",
    "capture_code_indexer_episode",
]
