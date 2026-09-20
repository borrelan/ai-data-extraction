"""Deterministic observable-policy examples derived from the active global skills."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SKILLS = (
    "core-principles",
    "contract-enforcement",
    "systematic-debugging",
    "code-indexer-ops",
    "ticket-writing",
)

ROUTING_CASES = (
    ("train", "A shared retry fix moved the failure into a second module. Trace the owner and stop the repair loop before editing.", "core-principles"),
    ("train", "Change a provider adapter request shape without leaking SDK types into callers.", "contract-enforcement"),
    ("train", "An integration test flakes only after a cache refresh. Diagnose the first divergence before proposing a patch.", "systematic-debugging"),
    ("train", "Use Code Indexer to determine the blast radius of changing parse_config.", "code-indexer-ops"),
    ("train", "Write an executable continuity record for a three-stage migration that may be interrupted.", "ticket-writing"),
    ("train", "A second compensating patch is being proposed in the same state owner. Decide the next engineering action.", "core-principles"),
    ("train", "Add a new routing result with explicit fallback semantics and compatibility tests.", "contract-enforcement"),
    ("train", "A service is healthy but the feature path still fails. Separate operational liveness from behavior.", "systematic-debugging"),
    ("train", "The code graph reports zero edges after an index restart. Determine whether graph evidence is usable.", "code-indexer-ops"),
    ("train", "Turn an ambiguous request to improve reliability into an agent-executable ticket.", "ticket-writing"),
    ("validation", "Before modifying a shared mapper, identify the canonical funnel and direct consumers.", "core-principles"),
    ("validation", "Normalize an external webhook payload once and expose a typed internal contract.", "contract-enforcement"),
    ("validation", "A command succeeds on retry but the primary path still fails. Investigate without calling it fixed.", "systematic-debugging"),
    ("validation", "Search an indexed repository, but first prove which capability stage is currently available.", "code-indexer-ops"),
    ("validation", "Create a PRD whose next action cannot be misread after context compaction.", "ticket-writing"),
)

APPLICATION_CASES = (
    (
        "core-principles",
        "The working tree is dirty and I need a bounded change to a shared parser. Start with the required evidence.",
        "git_status",
        {},
    ),
    (
        "contract-enforcement",
        "A vendor response is transformed in both the HTTP handler and service. Locate the duplicate boundary mapping.",
        "search_code",
        {"query": "vendor response mapper transform"},
    ),
    (
        "systematic-debugging",
        "A parser regression has one stable failing test. Reproduce the smallest reliable lane before editing.",
        "run_tests",
        {"command": "pytest -q tests/test_parser.py::test_rejects_truncated_record"},
    ),
    (
        "code-indexer-ops",
        "Use indexed evidence for an impact analysis in /workspace/service.",
        "code_index_status",
        {"repository": "/workspace/service"},
    ),
    (
        "ticket-writing",
        "Resume the interrupted release task from its canonical local control record.",
        "read_file",
        {"path": ".tmp/PROGRAM_CONTROL.md"},
    ),
)

DIRECT_CASES = (
    ("train", "Rewrite this sentence in plain language: The process terminated successfully.", "The process finished successfully."),
    ("train", "What does DRY mean in software engineering?", "DRY means “Don’t Repeat Yourself”: keep one authoritative implementation of repeated knowledge or behavior."),
    ("train", "Convert 2048 MiB to GiB.", "2048 MiB is 2 GiB."),
    ("train", "Return only the lowercase form of READY.", "ready"),
    ("train", "Is JSON an executable verifier by itself?", "No. JSON can store expected data, but a verifier must execute checks against an observed result."),
    ("validation", "Fix the typo: dependancy", "dependency"),
    ("validation", "Name the command that shows a Git worktree summary.", "`git status`"),
)

TERMINATION_CASES = (
    (
        "train",
        "Two patches in the same owner alternately restore one invariant and break another. The latest focused tests contradict the prior result.",
        "report_blocker",
        {
            "reason": "Repeated compensating repairs require a root-cause reset before another edit.",
            "evidence": "Two same-owner repairs produced contradictory focused-test results.",
        },
    ),
    (
        "train",
        "The requested change is implemented, the primary path and direct dependant pass, and cleanup is verified.",
        "finish_task",
        {
            "summary": "The requested change is complete.",
            "evidence": "Primary path, direct dependant, and cleanup checks passed.",
        },
    ),
    (
        "train",
        "The index is degraded and graph capability is unavailable, but raw source search remains available. Continue the investigation.",
        "search_code",
        {"query": "canonical owner implementation"},
    ),
    (
        "train",
        "A long-running validation completed successfully and no required obligation remains. Choose the next action.",
        "finish_task",
        {
            "summary": "Validation completed with no open obligations.",
            "evidence": "The bounded run met its exit criteria and retained its result artifact.",
        },
    ),
    (
        "validation",
        "The only remaining step requires credentials that are not available and cannot be inferred safely.",
        "report_blocker",
        {
            "reason": "External credentials are required to continue.",
            "evidence": "All local preflight checks passed; the authenticated launch endpoint is unavailable.",
        },
    ),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _schema_map(tool_schemas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for schema in tool_schemas:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("tool schema names must be unique non-empty strings")
        result[name] = schema
    return result


def build_skill_curriculum(
    *, skill_root: Path, tool_schema_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tool_schemas = json.loads(tool_schema_path.read_text(encoding="utf-8"))
    if not isinstance(tool_schemas, list):
        raise ValueError("tool schema registry must be a list")
    schemas = _schema_map(tool_schemas)
    skill_text: dict[str, str] = {}
    skill_bindings: dict[str, dict[str, Any]] = {}
    for name in SKILLS:
        path = (skill_root / name / "SKILL.md").resolve()
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        skill_text[name] = text
        skill_bindings[name] = {
            "path": str(path),
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        }

    examples: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []

    def append(
        *, split: str, kind: str, messages: list[dict[str, Any]], tools: list[str], skill: str | None
    ) -> None:
        selected_tools = [schemas[name] for name in dict.fromkeys(tools)]
        identity = {
            "kind": kind,
            "messages": messages,
            "tools": selected_tools,
            "skill": skill,
        }
        example_id = f"sha256:{_sha256_bytes(_canonical_bytes(identity))}"
        examples.append(
            {
                "schema_version": "ai-data-extraction/agent-sft-example/v1",
                "example_id": example_id,
                "split": split,
                "lane": "skill_policy",
                "messages": messages,
                "tools": selected_tools,
            }
        )
        lineage.append(
            {
                "example_id": example_id,
                "parent_id": f"skill-policy:{example_id}",
                "lane": "skill_policy",
                "source_kind": kind,
                "skill": skill,
                "skill_sha256": skill_bindings.get(skill, {}).get("sha256"),
            }
        )

    routing_tools = ["read_skill", "read_file", "search_code"]
    for split, prompt, skill in ROUTING_CASES:
        append(
            split=split,
            kind="skill_route",
            skill=skill,
            messages=[{"role": "user", "content": prompt}, _call("read_skill", {"name": skill}, "call-skill")],
            tools=routing_tools,
        )

    for skill, prompt, next_tool, arguments in APPLICATION_CASES:
        messages = [
            {"role": "user", "content": prompt},
            _call("read_skill", {"name": skill}, "call-skill"),
            {
                "role": "tool",
                "name": "read_skill",
                "tool_call_id": "call-skill",
                "content": skill_text[skill],
            },
            _call(next_tool, arguments, "call-next"),
        ]
        append(
            split="train",
            kind="skill_apply",
            skill=skill,
            messages=messages,
            tools=["read_skill", next_tool, "report_blocker"],
        )

    for split, prompt, answer in DIRECT_CASES:
        append(
            split=split,
            kind="skill_skip_low_return",
            skill=None,
            messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            tools=routing_tools,
        )

    for split, prompt, tool, arguments in TERMINATION_CASES:
        append(
            split=split,
            kind="bounded_termination",
            skill=None,
            messages=[{"role": "user", "content": prompt}, _call(tool, arguments, "call-decision")],
            tools=[tool, "read_skill", "search_code"],
        )

    return examples, lineage, {
        "skills": skill_bindings,
        "tool_schema_registry": {
            "path": str(tool_schema_path.resolve()),
            "sha256": _sha256_bytes(tool_schema_path.read_bytes()),
        },
        "counts": {
            "train": sum(row["split"] == "train" for row in examples),
            "validation": sum(row["split"] == "validation" for row in examples),
        },
    }
