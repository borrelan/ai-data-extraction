#!/usr/bin/env python3
"""Replay a deterministic harness fixture against one canonical session row.

The fixture consumes only metadata from a canonical JSONL row.  It creates a
successful tool episode and a bounded-stop episode so the harness contract can
be tested without copying private prompt, response, or reasoning content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from harness_trace import HarnessTrace, TraceLimits

FIXTURE_SCHEMA = "ai-data-extraction/harness-fixture/v1"


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_quality_metadata(path: Path) -> dict[str, Any]:
    """Read one row and retain only source/quality metadata."""
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            quality = record.get("quality")
            metadata = record.get("metadata")
            if not isinstance(quality, dict):
                continue
            session_quality_id = quality.get("session_quality_id")
            if not isinstance(session_quality_id, str) or not session_quality_id:
                continue
            return {
                "provider": (metadata or {}).get("provider", "unknown"),
                "source_class": (metadata or {}).get("source_class", "unknown"),
                "session_quality_id": session_quality_id,
                "session_quality_gate": quality.get(
                    "session_quality_gate", "unassessed"
                ),
                "session_quality_flags": quality.get("session_quality_flags", []),
            }
    raise ValueError(f"no row with session quality metadata found: {path}")


def _source(quality: dict[str, Any], episode_suffix: str) -> dict[str, Any]:
    return {
        "provider": quality["provider"],
        "source_class": quality["source_class"],
        "session_quality_id": quality["session_quality_id"],
        "session_quality_gate": quality["session_quality_gate"],
        "session_quality_flags": quality["session_quality_flags"],
        "fixture": episode_suffix,
    }


def _record_success_trace(quality: dict[str, Any]) -> HarnessTrace:
    trace = HarnessTrace(
        episode_id="fixture-success",
        source=_source(quality, "success"),
        registry_revision="fixture-registry-v1",
        skill_revision="fixture-skills-v1",
        environment_revision="fixture-environment-v1",
        verifier_revision="fixture-verifier-v1",
        privacy_state="heuristic",
        required_skills=(
            "core-principles",
            "contract-enforcement",
            "systematic-debugging",
            "code-indexer-ops",
        ),
    )
    for skill in (
        "core-principles",
        "contract-enforcement",
        "systematic-debugging",
        "code-indexer-ops",
    ):
        trace.record_skill_preflight(
            skill=skill,
            skill_revision="fixture-skill-v1",
            mandatory=True,
            trigger="fixture episode",
            read_result="read",
            scope_decision="in_scope",
            content_sha256=_digest(f"{skill}:fixture"),
        )
    trace.record_tool_registry(
        registry_revision="fixture-registry-v1",
        tools=[
            {
                "name": "code-indexer.search",
                "schema": {"type": "object", "required": ["query"]},
                "trust_class": "derived_read_only",
                "side_effect_class": "read_only",
            }
        ],
    )
    trace.record_mcp_health(
        server="code-indexer",
        revision="fixture-mcp-v1",
        health_result="healthy",
        capabilities=["search", "references"],
    )
    trace.record_decision(
        decision="use",
        decision_basis="required semantic owner lookup",
        required_gate_status="passed",
        tool_name="code-indexer.search",
    )
    trace.record_tool_call(
        tool_name="code-indexer.search",
        call_id="fixture-call-1",
        arguments={"query": "owner boundary", "limit": 5},
        permission_decision="not_required",
    )
    trace.record_tool_observation(
        call_id="fixture-call-1",
        status="success",
        output={"matches": 2, "evidence_scope": "bounded_fixture"},
    )
    trace.record_state_delta(
        before={"source_revision": _digest("before")},
        after={"source_revision": _digest("after")},
        changed=["source_revision"],
    )
    trace.record_verification(
        verifier_revision="fixture-verifier-v1",
        checks=[
            {"name": "tool_observation_matched", "result": "pass"},
            {"name": "state_delta_recorded", "result": "pass"},
        ],
        result="pass",
        durable_evidence=["fixture:bounded-observation"],
    )
    trace.record_terminal(status="success", evidence=["verification:pass"])
    return trace


def _record_bounded_stop_trace(quality: dict[str, Any]) -> HarnessTrace:
    trace = HarnessTrace(
        episode_id="fixture-bounded-stop",
        source=_source(quality, "bounded-stop"),
        registry_revision="fixture-registry-v1",
        skill_revision="fixture-skills-v1",
        environment_revision="fixture-environment-v1",
        verifier_revision="fixture-verifier-v1",
        privacy_state="heuristic",
        required_skills=(
            "core-principles",
            "contract-enforcement",
            "systematic-debugging",
            "code-indexer-ops",
        ),
        limits=TraceLimits(repeated_signature_limit=3, no_progress_limit=3),
    )
    for _ in range(3):
        trace.record_loop_guard(
            signature="same-decision",
            state_hash=_digest("unchanged-state"),
            limit=3,
            reason="fixture_repeated_signature",
        )
    return trace


def replay_fixture(
    input_jsonl: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    quality = _load_quality_metadata(input_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "success.jsonl"
    bounded_stop_path = output_dir / "bounded_stop.jsonl"
    _record_success_trace(quality).write_jsonl(success_path, overwrite=overwrite)
    _record_bounded_stop_trace(quality).write_jsonl(
        bounded_stop_path,
        overwrite=overwrite,
    )
    manifest = {
        "schema_version": FIXTURE_SCHEMA,
        "source": {
            "name": input_jsonl.name,
            "sha256": _file_sha256(input_jsonl),
            "quality": quality,
        },
        "outputs": {
            "success": {"file": success_path.name, "events": 12},
            "bounded_stop": {"file": bounded_stop_path.name, "events": 4},
        },
        "content_policy": "metadata_only_fixture; no prompt,response,or reasoning content",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temporary, manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/harness-fixture"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = replay_fixture(args.input_jsonl, args.output_dir, overwrite=args.overwrite)
    print(f"Fixture replayed for session quality {manifest['source']['quality']['session_quality_id']}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
