#!/usr/bin/env python3
"""Build privacy-gated SFT, preference, and trajectory datasets.

The extractor scripts are intentionally format-specific.  This module is the
single boundary where their records become training data.  It removes private
reasoning fields, normalizes messages and tool traces, assigns deterministic
identities/splits/tags, and refuses to invent rewards or preferences.

The output is JSONL and uses conversational records so it can be adapted to
Unsloth, TRL, or another trainer without coupling extraction to one trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "ai-data-extraction/v1"
EVENT_SCHEMA_VERSION = "ai-data-extraction/event/v1"
ACTION_WINDOW_SCHEMA_VERSION = "ai-data-extraction/action-window/v2"
ACTION_EVIDENCE_SCHEMA_VERSION = "ai-data-extraction/action-evidence/v1"
BUILDER_VERSION = "1.4.0"
PARSER_REVISION = f"ai-data-extraction/build_training_data@{BUILDER_VERSION}"
CHUNK_STRATEGIES = frozenset({"chunk", "reject"})
TRAINING_LANES = frozenset({"primary", "optional_alt", "quarantine"})
QUALITY_GATES = frozenset({"candidate", "review_required", "quarantine", "unassessed"})
QUALITY_GATE_RANK = {
    "candidate": 0,
    "review_required": 1,
    "quarantine": 2,
    "unassessed": 3,
}
MODEL_TIER_SCHEMA_VERSION = "ai-data-extraction/model-tier/v1"
MODEL_TIERS = frozenset(
    {"tier1_frontier", "tier2_open_source", "tier3_local", "unclassified"}
)
MODEL_TIER_ALIASES = {
    "tier1": "tier1_frontier",
    "tier1_frontier": "tier1_frontier",
    "frontier": "tier1_frontier",
    "tier2": "tier2_open_source",
    "tier2_open_source": "tier2_open_source",
    "open_source": "tier2_open_source",
    "opensource": "tier2_open_source",
    "tier3": "tier3_local",
    "tier3_local": "tier3_local",
    "local": "tier3_local",
    "self_hosted": "tier3_local",
    "self-hosted": "tier3_local",
    "unclassified": "unclassified",
    "unknown": "unclassified",
}
MODEL_TIER_TAGS = {
    "tier1_frontier": "tier:tier1-frontier",
    "tier2_open_source": "tier:tier2-open-source",
    "tier3_local": "tier:tier3-local",
    "unclassified": "tier:unclassified",
}
MODEL_TIER_REGISTRY_REVISION = "builtin-20260917-v2"
MODEL_TIER_LOCAL_MARKERS = (
    "local",
    "self_host",
    "self-host",
    "ollama",
    "llama.cpp",
    "llamacpp",
    "lmstudio",
    "llm-studio",
    "openarc",
)
FRONTIER_MODEL_PATTERN = re.compile(
    r"(?i)(?:^|[/.:_-])(?:claude-(?:opus|sonnet|haiku)|gemini-[0-9]|gpt-[0-9]|o[1-9](?:[-.]|$)|codex(?:[-.]|$))"
)
OPEN_SOURCE_MODEL_PATTERN = re.compile(
    r"(?i)(?:^|[/.:_-])(?:qwen[0-9]*|deepseek|glm|chatglm|llama|mistral|hy[0-9]|zai/glm)(?:[-_./:]|$)"
)
CHUNK_TARGET_FRACTION = 0.55
ACTION_WINDOW_PAYLOAD_MAX_CHARS = 96_000
OUTPUT_FILES = (
    "sft.jsonl",
    "trajectories.jsonl",
    "tool_traces.jsonl",
    "action_windows.jsonl",
    "preferences.jsonl",
    "rl_prompts.jsonl",
    "rejected.jsonl",
)

ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
TASK_FAMILIES = (
    "code-edit",
    "debugging",
    "testing",
    "review",
    "documentation",
    "configuration",
    "question",
    "unknown",
)
OUTCOMES = frozenset({"success", "failure", "partial", "unknown"})
PRIVACY_MODES = frozenset({"heuristic", "filtered", "none"})

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "developer": "system",
    "system": "system",
    "assistant": "assistant",
    "agent": "assistant",
    "model": "assistant",
    "gemini": "assistant",
    "bot": "assistant",
    "tool": "tool",
    "function": "tool",
    "tool_result": "tool",
    "tool-result": "tool",
}

PROVIDER_ALIASES = {
    "claude": "claude",
    "claude-code": "claude",
    "claude-desktop": "claude",
    "codex": "codex",
    "cursor": "cursor",
    "cursor-agent-cli": "cursor",
    "cursor-aiservice": "cursor",
    "cursor-chat": "cursor",
    "cursor-global-composer": "cursor",
    "cursor-workspace-composer": "cursor",
    "continue": "continue",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "opencode": "opencode",
    "opencode-cli": "opencode",
    "opencode-desktop": "opencode",
    "prime-agent": "prime-agent",
    "prime": "prime-agent",
    "pi-agent": "pi-agent",
    "pi": "pi-agent",
    "oh-my-pi": "oh-my-pi",
    "ohmypi": "oh-my-pi",
    "trae": "trae",
    "windsurf": "windsurf",
    "windsurf-agent": "windsurf",
    "windsurf-chat": "windsurf",
}
PROVIDER_MENTION_PATTERNS = {
    "openai": re.compile(r"\b(?:openai|chatgpt)\b", re.I),
    "anthropic": re.compile(r"\b(?:anthropic|claude)\b", re.I),
    "google": re.compile(r"\b(?:google|gemini)\b", re.I),
    "qwen": re.compile(r"\bqwen\b", re.I),
    "cursor": re.compile(r"\bcursor\b", re.I),
    "opencode": re.compile(r"\bopencode\b", re.I),
    "codex": re.compile(r"\bcodex\b", re.I),
    "prime": re.compile(r"\bprime(?:\s+agent)?\b", re.I),
    "pi": re.compile(r"\boh[ -]?my[ -]?pi\b|\bpi(?:\s+agent)?\b", re.I),
}

REASONING_KEYS = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "chain-of-thought",
        "cot",
        "deliberation",
        "internal_reasoning",
        "reasoning",
        "scratchpad",
        "thinking",
        "thought",
        "thoughts",
    }
)
REASONING_CONTROL_KEYS = frozenset(
    {
        "enable_thinking",
        "preserve_thinking",
        "reasoning_budget",
        "reasoning_budget_message",
        "reasoning_effort",
        "thinking_budget",
        "thinking_level",
    }
)
HIDDEN_TRAINER_KEYS = REASONING_KEYS | REASONING_CONTROL_KEYS

TOOL_CALL_KEYS = (
    "tool_calls",
    "tool_uses",
    "tool_use",
    "toolCalls",
    "toolCall",
    "function_calls",
    "custom_tool_calls",
)
TOOL_RESULT_KEYS = (
    "tool_results",
    "tool_result",
    "toolResult",
    "toolCallStates",
    "function_call_outputs",
    "custom_tool_call_outputs",
)
ARTIFACT_KEYS = (
    "diff",
    "diffs",
    "suggested_diffs",
    "suggestedDiffs",
    "diff_histories",
    "diffHistories",
    "edits",
    "suggested_code_blocks",
    "suggestedCodeBlocks",
)
CONTEXT_KEYS = ("code_context", "context_items", "context", "selections")
TOOL_SCHEMA_KEYS = ("tools", "tool_definitions", "toolDefinitions")
TOOL_FAMILY_PATTERNS = {
    "code-indexer": (
        "code-indexer",
        "code_indexer",
        "codeintel",
        "code_intel",
        "serena",
        "ast-grep",
        "ast_grep",
    ),
    "lsp": (
        "lsp",
        "language-server",
        "language_server",
        "go-to-definition",
        "go_to_definition",
        "find-references",
        "find_references",
        "diagnostic",
        "hover",
    ),
    "devops": (
        "kubectl",
        "kubernetes",
        "k8s",
        "helm",
        "terraform",
        "docker",
        "podman",
        "ansible",
        "aws",
        "gcloud",
        "azure",
        "firebase",
        "ssh",
        "systemctl",
    ),
    "research": (
        "web",
        "search",
        "browser",
        "fetch",
        "http",
        "arxiv",
        "paper",
        "url",
        "finance",
        "weather",
    ),
    "filesystem": (
        "read",
        "write",
        "edit",
        "patch",
        "file",
        "glob",
        "grep",
        "rg",
        "find",
    ),
    "shell": ("bash", "shell", "terminal", "exec", "command", "run"),
    "source-control": (
        "git",
        "github",
        "gitlab",
        "pull-request",
        "pull_request",
        "commit",
    ),
    "testing": ("pytest", "vitest", "unittest", "test", "lint", "coverage"),
    "package-management": ("npm", "yarn", "pnpm", "pip", "cargo", "go-mod", "go_mod"),
    "mcp": ("mcp", "model-context-protocol", "model_context_protocol"),
}
TOOL_FAMILIES = tuple(TOOL_FAMILY_PATTERNS) + ("unclassified",)
PATH_KEYS = frozenset(
    {
        "cwd",
        "directory",
        "installation",
        "project_path",
        "source_file",
        "workspace",
    }
)
IDENTIFIER_KEYS = frozenset(
    {
        "agent_id",
        "chat_id",
        "composer_id",
        "id",
        "parent_id",
        "parent_session_id",
        "project_hash",
        "project_id",
        "session_id",
        "tab_id",
        "workspace_id",
    }
)

REASONING_BLOCK_PATTERNS = (
    # Require an XML-style tag boundary.  A bare ``<think|<analysis`` inside
    # a grep/rg expression is a literal search pattern, not a reasoning block.
    re.compile(r"(?is)<think(?:ing)?(?:\s[^>]*)?>.*?(?:</think(?:ing)?\s*>|$)"),
    re.compile(r"(?is)<analysis(?:\s[^>]*)?>.*?(?:</analysis\s*>|$)"),
    re.compile(r"(?is)```(?:thinking|reasoning|analysis)\s*\n.*?(?:```|$)"),
    re.compile(
        r"(?is)<\|im_start\|>\s*(?:think|analysis)\b.*?(?:<\|im_end\|>|$)"
    ),
)
TRAINER_MARKER_PATTERN = re.compile(
    r"<\s*/?\s*(?:thinking|analysis|reasoning|deliberation|scratchpad|"
    r"chain[_ -]?of[_ -]?thought)\b|"
    r"\b(?:chain of thought|hidden reasoning|internal reasoning|"
    r"private scratchpad)\b",
    re.IGNORECASE,
)
HIDDEN_REASONING_TOOL_MARKERS = frozenset(
    {
        "chainofthought",
        "hiddenreasoning",
        "scratchpad",
        "sequentialthinking",
    }
)

SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}"), "<SECRET>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "<SECRET>"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b"), "<SECRET>"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "<SECRET>"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer <SECRET>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "<PRIVATE_KEY>"),
    (re.compile(r"(?i)(\b(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*)[^\s,;]+"), r"\1<SECRET>"),
)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
UNIX_HOME_PATTERN = re.compile(r"(?<![A-Za-z0-9])/(?:home|Users)/[^/\s'\"]+")
WINDOWS_HOME_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]Users[\\/][^\\/\s'\"]+")


@dataclass
class NormalizationState:
    reasoning_fields_removed: int = 0
    reasoning_blocks_removed: int = 0
    redactions: int = 0
    dropped_roles: list[str] = field(default_factory=list)
    multimodal_parts: int = 0
    tool_arguments_unparsed: int = 0


@dataclass
class BuildStats:
    input_records: int = 0
    normalized_records: int = 0
    sft_records: int = 0
    trajectory_records: int = 0
    tool_trace_records: int = 0
    action_window_records: int = 0
    recovered_action_window_records: int = 0
    preference_records: int = 0
    rl_prompt_records: int = 0
    rejected_records: int = 0
    duplicate_records: int = 0
    source_counts: Counter[str] = field(default_factory=Counter)
    task_counts: Counter[str] = field(default_factory=Counter)
    outcome_counts: Counter[str] = field(default_factory=Counter)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    reward_status_counts: Counter[str] = field(default_factory=Counter)
    tool_family_counts: Counter[str] = field(default_factory=Counter)
    provider_mention_counts: Counter[str] = field(default_factory=Counter)
    rl_prompt_duplicate_groups: int = 0
    rl_prompt_duplicate_rows: int = 0
    chunked_parent_records: int = 0
    chunk_records: int = 0
    skipped_lane_records: int = 0
    skipped_quality_gate_records: int = 0
    training_lane_counts: Counter[str] = field(default_factory=Counter)
    quality_gate_counts: Counter[str] = field(default_factory=Counter)
    quality_session_gate_counts: Counter[str] = field(default_factory=Counter)
    quality_session_lane_counts: Counter[str] = field(default_factory=Counter)
    model_tier_counts: Counter[str] = field(default_factory=Counter)
    skipped_model_tier_records: int = 0
    quality_session_count: int = 0
    quality_session_conflict_count: int = 0


class RejectionReasons(list[str]):
    """List-compatible rejection reasons carrying the derived quality gate."""

    def __init__(self, reasons: Iterable[str], *, quality_gate: str, quality_flags: Iterable[str]):
        super().__init__(reasons)
        self.quality_gate = quality_gate
        self.quality_flags = list(quality_flags)


@dataclass
class SessionQualityAccumulator:
    """Bounded evidence accumulator for one source session.

    A source file can contain many bounded segments from one parent session.
    This object deliberately keeps only structural evidence and unresolved
    tool IDs; it never retains message text or reasoning.  The pending-ID
    sets are pruned whenever a call and observation pair is closed, so a
    long healthy session does not require memory proportional to its whole
    tool history.
    """

    session_key: str
    session_quality_id: str
    record_count: int = 0
    message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    tool_message_count: int = 0
    action_count: int = 0
    observation_count: int = 0
    tool_error_count: int = 0
    terminal_signal_count: int = 0
    truncated_payload_count: int = 0
    parse_error_count: int = 0
    source_parse_error_count: int = 0
    nonempty_message_count: int = 0
    pending_action_ids: set[str] = field(default_factory=set)
    pending_observation_ids: set[str] = field(default_factory=set)
    model_values: set[str] = field(default_factory=set)
    provider_values: set[str] = field(default_factory=set)
    explicit_gates: set[str] = field(default_factory=set)
    explicit_flags: set[str] = field(default_factory=set)
    source_classes: set[str] = field(default_factory=set)
    training_lanes: set[str] = field(default_factory=set)
    model_tiers: set[str] = field(default_factory=set)
    model_tier_bases: set[str] = field(default_factory=set)
    advisor_overlay: bool = False
    explicit_outcome_observed: bool = False


def quality_session_key(record: Any, source_file_hash: str) -> str:
    """Return a provider-neutral key for grouping segments into a session.

    The source file hash prevents an ID reused by two exports from merging.
    A stable session ID takes precedence over a builder-created chunk parent
    because chunking happens after this preflight.  When no session ID exists,
    the extractor parent hash is the next strongest identity.
    """

    if not isinstance(record, dict):
        identity: Any = {"record": canonical_json_digest(record)}
    else:
        identifier = first_value(
            record,
            ("session_id", "composer_id", "chat_id", "agent_id", "tab_id"),
        )
        parent = record.get("_chunk_parent_record_sha256")
        if identifier not in (None, ""):
            identity = {"session_id": str(identifier)}
        elif isinstance(parent, str) and re.fullmatch(r"[0-9a-f]{64}", parent):
            identity = {"parent_record_sha256": parent}
        else:
            identity = {"record": canonical_json_digest(record)}
    return canonical_json_digest(
        {"source_file_sha256": source_file_hash, "identity": identity}
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_CANONICAL_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)


def canonical_json_chunks(value: Any) -> Iterator[str]:
    """Yield canonical JSON without materializing a potentially huge string."""
    return _CANONICAL_ENCODER.iterencode(value)


def canonical_json_digest(value: Any) -> str:
    digest = hashlib.sha256()
    for chunk in canonical_json_chunks(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def canonical_json_size(value: Any) -> int:
    """Return canonical JSON character size without building the full string."""
    return sum(len(chunk) for chunk in canonical_json_chunks(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: Any) -> str:
    return f"sha256:{canonical_json_digest(value)}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
    else:
        token = str(value).strip()
    return token or None


def is_hidden_reasoning_tool(value: Any) -> bool:
    token = normalize_token(value) or ""
    compact = re.sub(r"[^a-z0-9]+", "", token.lower())
    return any(marker in compact for marker in HIDDEN_REASONING_TOOL_MARKERS)


def normalized_key(value: Any) -> str:
    """Normalize field names for conservative policy checks."""
    return str(value).lower().replace("-", "_").replace(" ", "_")


def redact_text(value: str, state: NormalizationState, *, enabled: bool) -> str:
    """Apply conservative structural redaction; this is not an anonymity proof."""
    if not enabled:
        return value
    cleaned = value
    for pattern, replacement in SECRET_PATTERNS:
        cleaned, count = pattern.subn(replacement, cleaned)
        state.redactions += count
    cleaned, count = EMAIL_PATTERN.subn("<PRIVATE_EMAIL>", cleaned)
    state.redactions += count
    cleaned, count = UNIX_HOME_PATTERN.subn("<PRIVATE_PATH>", cleaned)
    state.redactions += count
    cleaned, count = WINDOWS_HOME_PATTERN.subn("<PRIVATE_PATH>", cleaned)
    state.redactions += count
    return cleaned


def strip_reasoning_blocks(value: str, state: NormalizationState) -> str:
    cleaned = value
    for pattern in REASONING_BLOCK_PATTERNS:
        cleaned, count = pattern.subn("", cleaned)
        state.reasoning_blocks_removed += count
    return cleaned.strip()


def clean_text(value: str, state: NormalizationState, *, privacy_enabled: bool) -> str:
    without_reasoning = strip_reasoning_blocks(value, state)
    return redact_text(without_reasoning, state, enabled=privacy_enabled).strip()


def clean_value(value: Any, state: NormalizationState, *, privacy_enabled: bool) -> Any:
    """Copy JSON values while removing reasoning fields and redacting text."""
    if isinstance(value, str):
        return clean_text(value, state, privacy_enabled=privacy_enabled)
    if isinstance(value, list):
        return [clean_value(item, state, privacy_enabled=privacy_enabled) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if normalized_key(key_text) in HIDDEN_TRAINER_KEYS:
                state.reasoning_fields_removed += 1
                continue
            cleaned[key_text] = clean_value(item, state, privacy_enabled=privacy_enabled)
        return cleaned
    return value


def trainer_marker_count(value: Any) -> int:
    """Count visible reasoning-marker text without retaining its content."""
    if isinstance(value, str):
        return len(TRAINER_MARKER_PATTERN.findall(value))
    if isinstance(value, list):
        return sum(trainer_marker_count(item) for item in value)
    if isinstance(value, dict):
        return sum(trainer_marker_count(item) for item in value.values())
    return 0
    return value


def text_from_content(
    value: Any,
    state: NormalizationState,
    *,
    privacy_enabled: bool,
) -> str:
    """Extract user-visible text and skip structured thinking/multimodal parts."""
    if isinstance(value, str):
        return clean_text(value, state, privacy_enabled=privacy_enabled)
    if isinstance(value, list):
        parts = [
            text_from_content(item, state, privacy_enabled=privacy_enabled)
            for item in value
        ]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        part_type = normalize_token(value.get("type"))
        if part_type and normalized_key(part_type) in HIDDEN_TRAINER_KEYS | {"thinking", "analysis"}:
            state.reasoning_fields_removed += 1
            return ""
        if part_type in {"image", "image_url", "audio", "video"}:
            state.multimodal_parts += 1
            return ""
        for key in ("text", "content", "rawText", "message", "value"):
            if key in value:
                return text_from_content(
                    value[key], state, privacy_enabled=privacy_enabled
                )
    return ""


def structured_content_parts(value: Any) -> list[dict[str, Any]]:
    """Return structured content blocks without treating ordinary text as a block."""
    if isinstance(value, list):
        return [part for part in value if isinstance(part, dict)]
    if isinstance(value, dict) and value.get("type") is not None:
        return [value]
    return []


def normalized_tool_schemas(
    record: dict[str, Any], state: NormalizationState, *, privacy_enabled: bool
) -> list[Any]:
    """Preserve optional function schemas for trainers that support tool SFT."""
    for key in TOOL_SCHEMA_KEYS:
        value = record.get(key)
        if value is None:
            continue
        cleaned = clean_value(value, state, privacy_enabled=privacy_enabled)
        schemas = cleaned if isinstance(cleaned, list) else [cleaned]
        visible_schemas = [
            schema
            for schema in schemas
            if not is_hidden_reasoning_tool(
                tool_name(schema.get("function", schema))
                if isinstance(schema, dict)
                else None
            )
        ]
        state.reasoning_fields_removed += len(schemas) - len(visible_schemas)
        return visible_schemas
    return []


def tool_families(events: list[dict[str, Any]], tools: list[Any]) -> list[str]:
    """Assign conservative surface tags from names and sanitized payloads.

    These tags are routing/analysis hints, not a learned semantic classifier.
    An unclassified tool remains visible so a missing catalog mapping cannot be
    mistaken for absence of tool data.
    """
    if not events and not tools:
        return []
    haystacks: list[str] = []
    for event in events:
        name = event.get("name") if isinstance(event, dict) else None
        if name:
            haystacks.append(str(name).lower())
        if isinstance(event, dict) and event.get("kind") == "action":
            # Tool-family tagging is an audit hint, not a payload classifier.
            # A multi-megabyte observation/argument must not be serialized and
            # regex-scanned once for every family; use shallow keys and bounded
            # scalar signals instead.
            payload = event.get("input")
            if isinstance(payload, dict):
                signal_parts: list[str] = [str(key) for key in payload]
                for value in payload.values():
                    if isinstance(value, (str, int, float, bool)):
                        signal_parts.append(str(value)[:2048])
                haystacks.append(" ".join(signal_parts).lower())
            elif isinstance(payload, (str, int, float, bool)):
                haystacks.append(str(payload)[:4096].lower())
    for tool in tools:
        if isinstance(tool, dict):
            function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            name = function.get("name") or tool.get("name")
            if name:
                haystacks.append(str(name).lower())
        else:
            haystacks.append(str(tool).lower())

    def matches(haystack: str, pattern: str) -> bool:
        # Treat underscores and hyphens as token boundaries so ``thread`` does
        # not accidentally become a filesystem trace because it contains
        # ``read``.
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])",
                haystack,
            )
        )

    matched: set[str] = set()
    strong_families = {
        "code-indexer",
        "lsp",
        "devops",
        "filesystem",
        "shell",
        "source-control",
        "testing",
        "package-management",
    }
    for haystack in haystacks:
        local_matches = {
            family
            for family, patterns in TOOL_FAMILY_PATTERNS.items()
            if any(matches(haystack, pattern) for pattern in patterns)
        }
        # Generic words such as ``search`` and ``find`` are not enough to
        # relabel a known code-indexer/LSP surface as research/filesystem.
        if local_matches & {"code-indexer", "lsp"}:
            local_matches.discard("research")
            local_matches.discard("filesystem")
        elif local_matches & strong_families:
            local_matches.discard("research")
        matched.update(local_matches)
    return sorted(matched or {"unclassified"})


def provider_mentions(messages: list[dict[str, Any]]) -> list[str]:
    """Tag provider names mentioned in visible text without rewriting them."""
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    return sorted(
        provider
        for provider, pattern in PROVIDER_MENTION_PATTERNS.items()
        if pattern.search(text)
    )


def normalize_role(value: Any) -> str | None:
    token = normalize_token(value)
    if token is None:
        return None
    lowered = token.lower().replace(" ", "_")
    return ROLE_ALIASES.get(lowered)


def provider_for(record: dict[str, Any]) -> tuple[str, str]:
    raw = normalize_token(record.get("source") or record.get("provider")) or "unknown"
    lowered = raw.lower().replace("_", "-")
    return PROVIDER_ALIASES.get(lowered, "unknown"), raw


def training_lane_for(record: Any) -> str:
    """Return the explicit source-selection lane, defaulting legacy rows safely."""
    if not isinstance(record, dict):
        return "quarantine"
    value = normalize_token(record.get("training_lane"))
    return value if value in TRAINING_LANES else "quarantine" if value else "primary"


def normalize_model_tier(value: Any) -> str | None:
    """Normalize an explicit model-tier label without guessing its quality."""
    token = normalize_token(value)
    if token is None:
        return None
    normalized = token.lower().replace(" ", "_").replace("-", "_")
    return MODEL_TIER_ALIASES.get(normalized)


def _model_tier_values(
    record: Any, messages: Iterable[dict[str, Any]] | None = None
) -> tuple[set[str], set[str]]:
    """Collect explicit tier/deployment values from adapter-visible metadata."""
    tier_values: set[str] = set()
    deployment_values: set[str] = set()
    candidates: list[dict[str, Any]] = []
    if isinstance(record, dict):
        candidates.append(record)
    if messages is not None:
        candidates.extend(message for message in messages if isinstance(message, dict))
    for candidate in candidates:
        for key in ("model_tier", "modelTier"):
            tier = normalize_model_tier(candidate.get(key))
            if tier is not None:
                tier_values.add(tier)
        for key in ("model_deployment", "deployment", "serving_mode", "runtime"):
            value = normalize_token(candidate.get(key))
            if value:
                deployment_values.add(value.lower())
    return tier_values, deployment_values


def model_tier_for(
    record: Any,
    *,
    messages: Iterable[dict[str, Any]] | None = None,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve model provenance tier with an explicit, fail-closed policy.

    Tier is a mixture/provenance dimension, not a quality verdict.  A reviewed
    override or adapter-declared tier is authoritative.  Local deployment can
    be inferred from an unambiguous runtime marker because it changes the
    tier, but missing model identity is never promoted to frontier by source
    provider name alone.
    """
    if isinstance(override, dict):
        tier = normalize_model_tier(override.get("model_tier") or override.get("tier"))
        if tier is None:
            raise ValueError("model-tier override must contain a supported model_tier")
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": tier,
            "basis": str(override.get("basis") or "reviewed_manifest"),
            "confidence": "reviewed",
            "conflict": False,
            "override": True,
        }

    # The existing adapters often expose model/provider fields but not a
    # trustworthy deployment or open-weight registry assertion.  Preserve the
    # model identity for audit, while refusing to infer Tier 1 or Tier 2 from
    # a name that may be a routed alias.
    tier_values, deployment_values = _model_tier_values(record, messages)
    provenance_tokens: list[str] = []
    candidates: list[dict[str, Any]] = []
    if isinstance(record, dict):
        candidates.append(record)
    if messages is not None:
        candidates.extend(message for message in messages if isinstance(message, dict))
    for candidate in candidates:
        for key in (
            "model",
            "modelID",
            "model_id",
            "modelId",
            "modelName",
            "provider",
            "providerID",
            "provider_id",
            "api",
            "model_provider",
            "deployment_provider",
        ):
            value = candidate.get(key)
            if value not in (None, ""):
                provenance_tokens.append(str(value).lower())
    deployment_text = " ".join(sorted(deployment_values))
    local_provenance = any(
        marker in deployment_text
        for marker in MODEL_TIER_LOCAL_MARKERS
    ) or any(
        marker in value
        for value in provenance_tokens
        for marker in MODEL_TIER_LOCAL_MARKERS
    )

    if len(tier_values) > 1:
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": "unclassified",
            "basis": "conflicting_adapter_declarations",
            "confidence": "conflict",
            "conflict": True,
            "override": False,
        }
    if len(tier_values) == 1:
        declared_tier = next(iter(tier_values))
        if local_provenance and declared_tier != "tier3_local":
            return {
                "schema_version": MODEL_TIER_SCHEMA_VERSION,
                "tier": "unclassified",
                "basis": "conflicting_declared_and_local_provenance",
                "confidence": "conflict",
                "conflict": True,
                "override": False,
            }
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": declared_tier,
            "basis": "adapter_declared",
            "confidence": "declared",
            "conflict": False,
            "override": False,
        }
    if local_provenance:
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": "tier3_local",
            "basis": "inferred_local_provenance_marker",
            "confidence": "heuristic",
            "conflict": False,
            "override": False,
        }

    source_token = ""
    if isinstance(record, dict):
        source_token = (normalize_token(record.get("source") or record.get("provider")) or "").lower()

    # These are versioned registry rules, not a provider-wide quality score:
    # the exact model family and serving evidence determine the tier.  The
    # rules intentionally leave aliases such as ``auto`` and ``mixed``
    # unclassified unless their runtime explicitly proves local serving.
    if source_token == "codex" and any("openai" in value for value in provenance_tokens):
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": "tier1_frontier",
            "basis": "model_registry:codex_openai_frontier_runtime",
            "confidence": "registry",
            "conflict": False,
            "override": False,
        }
    if any(FRONTIER_MODEL_PATTERN.search(value) for value in provenance_tokens):
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": "tier1_frontier",
            "basis": "model_registry:frontier_model_family",
            "confidence": "registry",
            "conflict": False,
            "override": False,
        }
    if any(OPEN_SOURCE_MODEL_PATTERN.search(value) for value in provenance_tokens):
        return {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": "tier2_open_source",
            "basis": "model_registry:open_source_model_family",
            "confidence": "registry",
            "conflict": False,
            "override": False,
        }
    return {
        "schema_version": MODEL_TIER_SCHEMA_VERSION,
        "tier": "unclassified",
        "basis": "missing_authoritative_tier",
        "confidence": "unknown",
        "conflict": False,
        "override": False,
    }


def quality_assessment_for(
    record: Any,
    *,
    messages: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    quality_override: dict[str, Any] | None = None,
    session_quality_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return provider-neutral session quality metadata.

    Quality is intentionally not inferred from ``source`` or ``provider``.
    Older exports that carry a provider-specific quality string are retained
    as an unmapped hint.  Their structural session evidence is still assessed
    here, but the old string is never treated as a quality verdict.
    """

    messages = messages or []
    events = events or []

    if isinstance(session_quality_assessment, dict):
        assessment = dict(session_quality_assessment)
        tier_info = model_tier_for(record, messages=messages)
        assessment.setdefault("model_tier", tier_info["tier"])
        assessment.setdefault("model_tier_basis", tier_info["basis"])
        assessment.setdefault("model_tier_confidence", tier_info["confidence"])
        assessment.setdefault(
            "model_tier_registry_revision", MODEL_TIER_REGISTRY_REVISION
        )
        assessment.setdefault("scope", "source_session")
        assessment.setdefault("provider_neutral", True)
        return apply_quality_override(assessment, quality_override)
    if not isinstance(record, dict):
        assessment = {
            "assessment_version": "session-quality/v1",
            "gate": "unassessed",
            "flags": ["quality_assessment_missing"],
            "method": "builder_default",
            "provider_neutral": True,
            "model_tier": "unclassified",
            "model_tier_basis": "missing_record",
            "model_tier_registry_revision": MODEL_TIER_REGISTRY_REVISION,
        }
        return apply_quality_override(assessment, quality_override)
    value = record.get("quality_assessment")
    if isinstance(value, dict):
        gate = value.get("gate") or value.get("quality_gate")
        if gate in QUALITY_GATES:
            assessment = dict(value)
            assessment["gate"] = gate
            flags = assessment.get("flags")
            assessment["flags"] = sorted(
                {str(flag) for flag in flags} if isinstance(flags, list) else set()
            )
            tier_info = model_tier_for(record, messages=messages)
            assessment.setdefault("model_tier", tier_info["tier"])
            assessment.setdefault("model_tier_basis", tier_info["basis"])
            assessment.setdefault("model_tier_confidence", tier_info["confidence"])
            assessment.setdefault(
                "model_tier_registry_revision", MODEL_TIER_REGISTRY_REVISION
            )
            assessment.setdefault("scope", "source_session")
            assessment.setdefault("provider_neutral", True)
            return apply_quality_override(assessment, quality_override)
    legacy_gate = record.get("quality_gate")
    user_count = sum(1 for message in messages if message.get("role") == "user")
    assistant_count = sum(
        1 for message in messages if message.get("role") == "assistant"
    )
    action_events = [event for event in events if event.get("kind") == "action"]
    observation_events = [
        event for event in events if event.get("kind") == "observation"
    ]
    action_ids = {
        event.get("call_id") for event in action_events if event.get("call_id")
    }
    observation_ids = {
        event.get("call_id")
        for event in observation_events
        if event.get("call_id")
    }
    flags: list[str] = []
    if not messages:
        flags.append("no_normalized_messages")
    if user_count == 0:
        flags.append("missing_user_objective")
    if assistant_count == 0:
        flags.append("missing_visible_assistant_response")
    if action_ids - observation_ids:
        flags.append("unmatched_tool_calls")
    if observation_ids - action_ids:
        flags.append("unmatched_tool_observations")
    if isinstance(record.get("observation_truncations"), list) and record["observation_truncations"]:
        flags.append("payload_truncated")
    model_token_keys = (
        "model",
        "modelID",
        "model_id",
        "modelId",
        "modelName",
        "model_provider",
        "provider",
        "providerID",
        "provider_id",
        "api",
    )
    model_tokens = [
        str(record.get(key)).lower()
        for key in model_token_keys
        if record.get(key) not in (None, "")
    ]
    # Older adapters commonly attach model/provider identity to an assistant
    # message rather than to the parent record.  Read that observable
    # provenance here so a local-model flag is not lost merely because the
    # source adapter chose a different envelope.  It remains a flag only; it
    # never changes the quality gate by itself.
    for message in messages:
        if not isinstance(message, dict):
            continue
        for key in model_token_keys:
            value = message.get(key)
            if value not in (None, ""):
                model_tokens.append(str(value).lower())
    local_tokens = (
        "local",
        "ollama",
        "llama.cpp",
        "llamacpp",
        "lmstudio",
        "self-host",
        "self_host",
        "openarc",
    )
    model_provenance = (
        "local_or_self_hosted"
        if any(token in value for value in model_tokens for token in local_tokens)
        else "identified_or_unknown"
        if model_tokens
        else "unknown"
    )
    if model_provenance == "local_or_self_hosted":
        flags.append("model_provenance_local_or_self_hosted")
    elif model_provenance == "unknown":
        flags.append("model_provenance_unknown")
    outcome, _outcome_source = explicit_outcome(record)
    if outcome == "unknown":
        flags.append("outcome_unverified")
    tier_info = model_tier_for(record, messages=messages)
    if tier_info["tier"] == "unclassified":
        flags.append("model_tier_unclassified")
    if tier_info["conflict"]:
        flags.append("conflicting_model_tiers")
    assessment: dict[str, Any] = {
        "assessment_version": "session-quality/v1",
        "gate": "review_required",
        "flags": sorted(set(flags)),
        "dimensions": {
            "dialogue": {
                "user_messages": user_count,
                "assistant_messages": assistant_count,
                "tool_messages": sum(
                    1 for message in messages if message.get("role") == "tool"
                ),
            },
            "tool_trace_integrity": {
                "actions": len(action_events),
                "observations": len(observation_events),
                "matched_call_ids": len(action_ids & observation_ids),
                "unmatched_call_ids": len(action_ids - observation_ids),
                "unmatched_observation_ids": len(observation_ids - action_ids),
            },
            "outcome_evidence": {
                "status": "observed" if outcome != "unknown" else "unverified",
            },
            "model_provenance": {
                "status": model_provenance,
                "identified_values": model_tokens,
                "model_tier": tier_info["tier"],
                "model_tier_basis": tier_info["basis"],
            },
        },
        "model_tier": tier_info["tier"],
        "model_tier_basis": tier_info["basis"],
        "model_tier_confidence": tier_info["confidence"],
        "model_tier_registry_revision": MODEL_TIER_REGISTRY_REVISION,
        "method": "builder_structural_session_v1",
        "provider_neutral": True,
    }
    blocking_flags = {
        "no_normalized_messages",
        "missing_user_objective",
        "missing_visible_assistant_response",
        "unmatched_tool_calls",
        "unmatched_tool_observations",
        "payload_truncated",
        "conflicting_model_tiers",
    }
    if messages and not (set(assessment["flags"]) & blocking_flags):
        assessment["gate"] = "candidate"
    elif not messages:
        assessment["gate"] = "quarantine"
    if legacy_gate not in (None, ""):
        assessment["legacy_quality_hint"] = str(legacy_gate)
        assessment["flags"] = sorted(set(assessment["flags"]) | {"legacy_quality_hint"})
    return apply_quality_override(assessment, quality_override)


def apply_quality_override(
    assessment: dict[str, Any],
    quality_override: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply an explicit review decision without changing provenance policy."""

    if not isinstance(quality_override, dict):
        return assessment
    requested_gate = quality_override.get("quality_gate") or quality_override.get("gate")
    if requested_gate not in QUALITY_GATES:
        raise ValueError(
            "quality override gate must be one of: "
            + ", ".join(sorted(QUALITY_GATES))
        )
    result = dict(assessment)
    result["automatic_gate"] = result.get("gate", "unassessed")
    result["gate"] = requested_gate
    result["override"] = {
        key: value
        for key, value in quality_override.items()
        if key in {"quality_gate", "gate", "reason", "reviewer", "ticket"}
        and value not in (None, "")
    }
    return result


def load_quality_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load provider-neutral per-session quality decisions."""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    entries = value.get("overrides") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise ValueError("quality override file must contain an overrides list")
    overrides: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each quality override must be an object")
        gate = entry.get("quality_gate") or entry.get("gate")
        if gate not in QUALITY_GATES:
            raise ValueError(
                "quality override gate must be one of: "
                + ", ".join(sorted(QUALITY_GATES))
            )
        keys = [
            entry.get("source_file_sha256"),
            entry.get("source_sha256"),
            entry.get("session_id"),
        ]
        keys = [str(key) for key in keys if key not in (None, "")]
        if not keys:
            raise ValueError("quality override needs a source hash or session_id")
        override = {
            key: value
            for key, value in entry.items()
            if key in {"quality_gate", "gate", "reason", "reviewer", "ticket"}
            and value not in (None, "")
        }
        for key in keys:
            overrides[key] = override
            if key.startswith("sha256:"):
                overrides[key[7:]] = override
    return overrides


def quality_override_for(
    record: Any,
    *,
    source_file_hash: str,
    quality_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find a record/session override without using provider identity."""

    if not quality_overrides or not isinstance(record, dict):
        return None
    origin = record.get("source_origin")
    origin_hashes: list[str] = []
    if isinstance(origin, dict):
        for key in ("source_file_sha256", "source_sha256"):
            value = origin.get(key)
            if isinstance(value, str) and value:
                origin_hashes.extend((value, value.removeprefix("sha256:")))
    record_session_id = record.get("session_id")
    keys = origin_hashes + [
        str(record_session_id) if record_session_id not in (None, "") else "",
        source_file_hash,
        source_file_hash.removeprefix("sha256:"),
    ]
    for key in keys:
        if key and key in quality_overrides:
            return quality_overrides[key]
    return None


def load_model_tier_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load reviewed model-tier assignments keyed to immutable identities."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    entries = value.get("overrides") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise ValueError("model-tier override file must contain an overrides list")
    overrides: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each model-tier override must be an object")
        tier = normalize_model_tier(entry.get("model_tier") or entry.get("tier"))
        if tier is None or tier == "unclassified":
            raise ValueError(
                "model-tier override must assign tier1_frontier, "
                "tier2_open_source, or tier3_local"
            )
        keys = [
            entry.get("source_file_sha256"),
            entry.get("source_sha256"),
            entry.get("session_id"),
        ]
        keys = [str(key) for key in keys if key not in (None, "")]
        if not keys:
            raise ValueError("model-tier override needs a source hash or session_id")
        override = {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "model_tier": tier,
            "basis": entry.get("basis") or "reviewed_manifest",
            "reviewer": entry.get("reviewer"),
            "reason": entry.get("reason"),
            "ticket": entry.get("ticket"),
        }
        override = {key: item for key, item in override.items() if item not in (None, "")}
        for key in keys:
            overrides[key] = override
            if key.startswith("sha256:"):
                overrides[key[7:]] = override
    return overrides


def model_tier_override_for(
    record: Any,
    *,
    source_file_hash: str,
    model_tier_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find a tier override using source/session identity, never provider name."""
    if not model_tier_overrides or not isinstance(record, dict):
        return None
    origin = record.get("source_origin")
    origin_hashes: list[str] = []
    if isinstance(origin, dict):
        for key in ("source_file_sha256", "source_sha256"):
            value = origin.get(key)
            if isinstance(value, str) and value:
                origin_hashes.extend((value, value.removeprefix("sha256:")))
    record_session_id = record.get("session_id")
    keys = origin_hashes + [
        str(record_session_id) if record_session_id not in (None, "") else "",
        source_file_hash,
        source_file_hash.removeprefix("sha256:"),
    ]
    for key in keys:
        if key and key in model_tier_overrides:
            return model_tier_overrides[key]
    return None


def first_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def message_list(record: dict[str, Any]) -> list[Any] | None:
    for key in ("messages", "history", "conversation"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    return None


def message_key(record: dict[str, Any]) -> str | None:
    for key in ("messages", "history", "conversation"):
        if isinstance(record.get(key), list):
            return key
    return None


CHUNK_INTERNAL_KEYS = frozenset(
    {
        "_chunk_parent_record_sha256",
        "_chunk_index",
        "_chunk_count",
        "_chunk_message_start",
        "_chunk_message_end",
        "_chunk_cut_reason",
        "_chunk_anchor_message_index",
        "_open_tool_call_ids",
    }
)


def _raw_message_role(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    return normalize_role(value.get("role") or value.get("type"))


def _raw_message_has_tool_call(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(key in value for key in TOOL_CALL_KEYS):
        return True
    for part in structured_content_parts(value.get("content")):
        if normalized_key(part.get("type")) in {"tool_use", "tool_call"}:
            return True
    return False


def _raw_event_ids(value: Any, keys: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    if not isinstance(value, dict):
        return ids
    key_set = set(keys)
    include_actions = bool(key_set & set(TOOL_CALL_KEYS))
    include_observations = bool(key_set & set(TOOL_RESULT_KEYS)) or bool(
        key_set & {"tool_call_id", "toolCallId", "tool_callID", "tool_use_id"}
    )
    for key in keys:
        for entry in candidate_entries(value.get(key)) if key in value else []:
            if not isinstance(entry, dict):
                continue
            identifier = first_value(
                entry,
                ("id", "callID", "call_id", "toolCallId", "tool_call_id", "tool_use_id"),
            )
            if identifier is not None:
                ids.add(str(identifier))
    if include_observations:
        for key in ("tool_call_id", "toolCallId", "tool_callID", "tool_use_id"):
            if value.get(key) is not None:
                ids.add(str(value[key]))
    for part in structured_content_parts(value.get("content")):
        part_type = normalized_key(part.get("type"))
        if include_actions and part_type in {"tool_use", "tool_call"} and part.get("id") is not None:
            ids.add(str(part["id"]))
        if include_observations and part_type in {"tool_result", "tool_response"}:
            identifier = first_value(part, ("tool_use_id", "tool_call_id", "call_id", "id"))
            if identifier is not None:
                ids.add(str(identifier))
    return ids


def _raw_action_ids(value: Any) -> set[str]:
    return _raw_event_ids(value, TOOL_CALL_KEYS)


def _raw_observation_ids(value: Any) -> set[str]:
    ids = _raw_event_ids(value, TOOL_RESULT_KEYS)
    if isinstance(value, dict) and _raw_message_role(value) == "tool":
        ids.update(
            _raw_event_ids(
                value,
                ("tool_call_id", "toolCallId", "tool_callID", "tool_use_id"),
            )
        )
    return ids


def _source_record_hash(record: Any) -> str:
    """Return the immutable parent hash for a raw or chunked candidate."""
    if isinstance(record, dict):
        parent = record.get("_chunk_parent_record_sha256")
        if isinstance(parent, str) and re.fullmatch(r"[0-9a-f]{64}", parent):
            return parent
    return canonical_json_digest(record)


def _segment_record_hash(record: Any) -> str:
    """Hash the exact derived segment without private chunk bookkeeping."""
    if not isinstance(record, dict):
        return canonical_json_digest(record)
    visible = {
        key: value for key, value in record.items() if key not in CHUNK_INTERNAL_KEYS
    }
    return canonical_json_digest(visible)


def record_lineage(record: dict[str, Any], source_record_hash: str) -> dict[str, Any]:
    """Read deterministic segmentation metadata without exposing control keys."""
    parent_value = record.get("_chunk_parent_record_sha256")
    parent_hash = (
        parent_value
        if isinstance(parent_value, str) and re.fullmatch(r"[0-9a-f]{64}", parent_value)
        else source_record_hash
    )
    chunk_index = record.get("_chunk_index", 0)
    chunk_count = record.get("_chunk_count", 1)
    try:
        chunk_index = int(chunk_index)
    except (TypeError, ValueError):
        chunk_index = 0
    try:
        chunk_count = int(chunk_count)
    except (TypeError, ValueError):
        chunk_count = 1
    chunk_count = max(chunk_count, 1)
    chunk_index = min(max(chunk_index, 0), chunk_count - 1)
    if chunk_count == 1:
        continuation_status = "complete"
    elif chunk_index == 0:
        continuation_status = "start"
    elif chunk_index == chunk_count - 1:
        continuation_status = "end"
    else:
        continuation_status = "middle"
    start = record.get("_chunk_message_start")
    end = record.get("_chunk_message_end")
    anchor_index = record.get("_chunk_anchor_message_index")
    open_tool_call_ids = record.get("_open_tool_call_ids")
    if isinstance(open_tool_call_ids, (list, tuple)):
        open_tool_call_ids = sorted(
            {str(value) for value in open_tool_call_ids if value is not None}
        )
    else:
        open_tool_call_ids = []
    source_message_range = None
    if isinstance(start, int) and isinstance(end, int):
        source_message_range = {"start": start, "end": end}
    return {
        "parent_record_sha256": parent_hash,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "continuation_status": continuation_status,
        "cut_reason": record.get("_chunk_cut_reason") if chunk_count > 1 else None,
        "open_tool_call_ids": open_tool_call_ids,
        "source_message_range": source_message_range,
        "context_overlap_message_index": anchor_index if isinstance(anchor_index, int) else None,
        "previous_example_id": None,
        "next_example_id": None,
    }


def _approx_message_size(value: Any) -> int:
    size = canonical_json_size(value) + 1
    if not isinstance(value, dict):
        return size
    role = _raw_message_role(value)
    # Tool payloads are present once in the conversational message and again
    # in the normalized event projection.  Budget that duplication before the
    # final validator sees it; otherwise adding correct observations can turn
    # every formerly accepted chunk into an oversized rejection.
    if role == "tool":
        content = value.get("content", value.get("message", value.get("text", "")))
        size += canonical_json_size(content) + 1
    for key in TOOL_CALL_KEYS + TOOL_RESULT_KEYS + ARTIFACT_KEYS + CONTEXT_KEYS:
        if key in value:
            size += canonical_json_size(value[key]) + 1
    return size


def chunk_record_variants(record: Any, max_record_chars: int) -> list[Any]:
    """Split a long parent record at observable conversation boundaries.

    The parent stays immutable in the source JSONL.  Derived candidates retain
    only a bounded message range and carry private bookkeeping used to emit
    public lineage metadata.  We never split between a tool call and its tool
    observation, and an indivisible message/exchange is left intact so the
    normal size validator can reject it with provenance.
    """
    if not isinstance(record, dict) or max_record_chars <= 0:
        return [record]
    messages = message_list(record)
    key = message_key(record)
    if not messages or key is None or len(messages) < 2:
        return [record]

    # Leave room for normalized roles, tool events, and lineage metadata.  The
    # final validator remains authoritative; this is only a cut heuristic.
    # Event IDs, role=tool observations, and normalized tool payloads are
    # emitted alongside messages.  A conservative 55% pre-normalization target
    # keeps the final serialized segment below the safety bound on tool-heavy
    # transcripts while still allowing long ordinary turns to remain intact.
    target_size = max(1, int(max_record_chars * CHUNK_TARGET_FRACTION))
    whole_size = sum(_approx_message_size(message) for message in messages)
    if whole_size <= target_size:
        return [record]

    prefix_count = 0
    while prefix_count < len(messages) and _raw_message_role(messages[prefix_count]) == "system":
        prefix_count += 1
    prefix = list(messages[:prefix_count])
    prefix_size = sum(_approx_message_size(message) for message in prefix)

    chunks: list[tuple[list[Any], int, int, int | None]] = []
    current = list(prefix)
    current_size = prefix_size
    current_start = prefix_count
    current_has_user = False
    current_has_assistant = False
    current_last_user: tuple[Any, int] | None = None
    current_anchor_index: int | None = None
    pending_tool_ids: set[str] = set()

    def flush(
        end_index: int,
        *,
        continuation_anchor: tuple[Any, int] | None = None,
    ) -> None:
        nonlocal current, current_size, current_start
        nonlocal current_has_user, current_has_assistant, current_last_user
        nonlocal current_anchor_index, pending_tool_ids
        if current:
            chunks.append((current, current_start, end_index, current_anchor_index))
        current = list(prefix)
        current_size = prefix_size
        current_start = end_index + 1
        current_has_user = False
        current_has_assistant = False
        current_last_user = None
        current_anchor_index = None
        pending_tool_ids = set()
        if continuation_anchor is not None:
            anchor_message, anchor_index = continuation_anchor
            current.append(anchor_message)
            current_size += _approx_message_size(anchor_message)
            current_has_user = _raw_message_role(anchor_message) == "user"
            current_last_user = continuation_anchor
            current_anchor_index = anchor_index

    for index in range(prefix_count, len(messages)):
        message = messages[index]
        role = _raw_message_role(message)
        piece_size = _approx_message_size(message)
        split_before = False
        if current_has_user and current_has_assistant and current_size + piece_size > target_size:
            if role == "user":
                # A fresh objective/follow-up is the preferred cut point.
                split_before = not pending_tool_ids
            elif role != "tool" and current:
                last_role = _raw_message_role(current[-1])
                # Keep user -> assistant and assistant tool-call -> tool
                # observation pairs intact.  A completed assistant response is
                # a safe fallback boundary when the next user has not arrived.
                # When a long tool loop has no user boundary, carry the latest
                # user as explicit context into the continuation segment.
                if not pending_tool_ids and last_role == "assistant" and not _raw_message_has_tool_call(current[-1]):
                    split_before = True
                elif not pending_tool_ids and last_role == "tool" and role == "assistant":
                    split_before = True
        if split_before:
            anchor = None
            if role == "assistant" and current_last_user is not None:
                anchor = current_last_user
            flush(index - 1, continuation_anchor=anchor)
        current.append(message)
        current_size += piece_size
        current_has_user = current_has_user or role == "user"
        current_has_assistant = current_has_assistant or role == "assistant"
        pending_tool_ids.difference_update(_raw_observation_ids(message))
        pending_tool_ids.update(_raw_action_ids(message))
        if role == "user":
            current_last_user = (message, index)

    if current:
        chunks.append((current, current_start, len(messages) - 1, current_anchor_index))
    if len(chunks) <= 1:
        return [record]

    parent_hash = _source_record_hash(record)
    chunk_count = len(chunks)
    variants: list[dict[str, Any]] = []
    for index, (chunk_messages, start, end, anchor_index) in enumerate(chunks):
        candidate = dict(record)
        candidate[key] = chunk_messages
        # Detached session-level tool/artifact fields cannot be safely assigned
        # to every segment.  Keep them on the first segment instead of silently
        # duplicating observations and rewards across all descendants.
        if index > 0:
            for field_name in (*TOOL_CALL_KEYS, *TOOL_RESULT_KEYS, *ARTIFACT_KEYS, *CONTEXT_KEYS):
                candidate.pop(field_name, None)
        candidate.update(
            {
                "_chunk_parent_record_sha256": parent_hash,
                "_chunk_index": index,
                "_chunk_count": chunk_count,
                "_chunk_message_start": start,
                "_chunk_message_end": end,
                "_chunk_cut_reason": "token_budget",
            }
        )
        if anchor_index is not None:
            candidate["_chunk_anchor_message_index"] = anchor_index
        variants.append(candidate)
    return variants


def candidate_entries(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def clean_tool_arguments(
    value: Any,
    state: NormalizationState,
    *,
    privacy_enabled: bool,
) -> Any:
    """Keep tool arguments as JSON when possible for current trainer formats."""
    if not isinstance(value, str):
        return clean_value(value, state, privacy_enabled=privacy_enabled)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        state.tool_arguments_unparsed += 1
        return clean_text(value, state, privacy_enabled=privacy_enabled)
    # Parse first, then recurse through the structured value.  Cleaning the
    # JSON-encoded string first cannot remove nested reasoning keys such as
    # ``thought`` from a sequential-thinking tool payload.
    return clean_value(parsed, state, privacy_enabled=privacy_enabled)


def standard_tool_call(value: Any, state: NormalizationState, *, privacy_enabled: bool) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {"input": value}
    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    direct_name = first_value(raw, ("name", "tool_name"))
    nested_tool = raw.get("tool") if isinstance(raw.get("tool"), dict) else {}
    name = direct_name or function.get("name") or nested_tool.get("name")
    if not name and isinstance(nested_tool.get("function"), dict):
        name = nested_tool["function"].get("name")
    name = name or raw.get("tool") or "unknown"
    call_id = first_value(
        raw,
        ("id", "callID", "call_id", "toolCallId", "tool_call_id", "tool_use_id"),
    )
    arguments = first_value(raw, ("arguments", "input", "parameters"))
    if arguments is None:
        arguments = function.get("arguments")
    cleaned_arguments = clean_tool_arguments(
        arguments,
        state,
        privacy_enabled=privacy_enabled,
    )
    if cleaned_arguments is None:
        cleaned_arguments = {}
    return {
        "id": normalize_token(call_id),
        "type": "function",
        "function": {
            "name": normalize_token(name) or "unknown",
            "arguments": cleaned_arguments,
        },
    }


def normalize_messages(
    record: dict[str, Any],
    state: NormalizationState,
    *,
    privacy_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_messages = message_list(record)
    if raw_messages is None:
        return [], []

    messages: list[dict[str, Any]] = []
    raw_kept: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            state.dropped_roles.append("non-object")
            continue
        role = normalize_role(raw.get("role") or raw.get("type"))
        if role is None:
            state.dropped_roles.append(normalize_token(raw.get("role") or raw.get("type")) or "missing")
            continue
        content_value = raw.get("content")
        if content_value is None:
            content_value = raw.get("message", raw.get("text", ""))
        content = text_from_content(
            content_value, state, privacy_enabled=privacy_enabled
        )
        message: dict[str, Any] = {"role": role, "content": content}

        raw_tool_calls: list[Any] = []
        for key in TOOL_CALL_KEYS:
            if key in raw:
                raw_tool_calls.extend(candidate_entries(raw[key]))
        raw_tool_calls.extend(
            part
            for part in structured_content_parts(content_value)
            if normalized_key(part.get("type")) in {"tool_use", "tool_call"}
        )
        normalized_tool_calls = [
            standard_tool_call(item, state, privacy_enabled=privacy_enabled)
            for item in raw_tool_calls
        ]
        visible_tool_calls = [
            item
            for item in normalized_tool_calls
            if not is_hidden_reasoning_tool(item["function"].get("name"))
        ]
        state.reasoning_fields_removed += len(normalized_tool_calls) - len(visible_tool_calls)
        if visible_tool_calls:
            message["tool_calls"] = visible_tool_calls

        tool_call_id = first_value(
            raw,
            ("tool_call_id", "toolCallId", "tool_callID", "tool_use_id"),
        )
        if role == "tool" and tool_call_id is not None:
            message["tool_call_id"] = normalize_token(tool_call_id)

        if content or visible_tool_calls or role == "tool":
            messages.append(message)
            raw_kept.append(raw)
    return messages, raw_kept


def event_status(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("status") or value.get("state")
    token = (normalize_token(value) or "").lower()
    if token in {"success", "succeeded", "complete", "completed", "done", "ok"}:
        return "success"
    if token in {"failure", "failed", "error", "errored", "cancelled", "canceled"}:
        return "failure"
    return "unknown"


def tool_name(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    direct_name = first_value(value, ("name", "tool_name"))
    nested_tool = value.get("tool") if isinstance(value.get("tool"), dict) else {}
    name = direct_name or function.get("name") or nested_tool.get("name")
    if not name and isinstance(nested_tool.get("function"), dict):
        name = nested_tool["function"].get("name")
    if not name and value.get("tool") is not None:
        name = value.get("tool")
    return normalize_token(name) or "unknown"


def explicit_result_code(raw: dict[str, Any], payload: Any) -> tuple[int | None, str]:
    """Preserve only structured process result codes; never parse output prose."""

    values: set[int] = set()
    for candidate in (raw, payload, raw.get("metadata"), raw.get("details")):
        if not isinstance(candidate, dict):
            continue
        for key in ("returncode", "return_code", "exit_code", "exitCode"):
            value = candidate.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                values.add(value)
    if len(values) == 1:
        return next(iter(values)), "structured_field"
    if len(values) > 1:
        return None, "conflicting_structured_fields"
    return None, "absent"


def make_tool_event(
    value: Any,
    *,
    kind: str,
    message_index: int | None,
    state: NormalizationState,
    privacy_enabled: bool,
) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {"output": value}
    call_id = first_value(
        raw,
        ("id", "callID", "call_id", "toolCallId", "tool_call_id", "tool_use_id"),
    )
    status = event_status(raw.get("status") or raw.get("state"))
    event: dict[str, Any] = {
        "kind": kind,
        "message_index": message_index,
        "call_id": normalize_token(call_id),
        "name": tool_name(raw),
        "status": status,
        "status_source": "structured_status" if status != "unknown" else "absent",
    }
    if kind == "action":
        payload = first_value(raw, ("input", "arguments", "parameters"))
        if payload is None and isinstance(raw.get("function"), dict):
            payload = first_value(raw["function"], ("input", "arguments", "parameters"))
        event["input"] = clean_tool_arguments(
            payload,
            state,
            privacy_enabled=privacy_enabled,
        )
    else:
        payload = first_value(raw, ("output", "result", "content", "error"))
        event["output"] = clean_value(payload, state, privacy_enabled=privacy_enabled)
        result_code, result_code_source = explicit_result_code(raw, payload)
        event["result_code"] = result_code
        event["result_code_source"] = result_code_source
        if raw.get("error") is not None:
            event["status"] = "failure"
            event["status_source"] = "structured_error"
        elif result_code is not None:
            result_status = "success" if result_code == 0 else "failure"
            if event["status"] == "unknown":
                event["status"] = result_status
                event["status_source"] = "structured_result_code"
            elif event["status"] != result_status:
                event["status"] = "unknown"
                event["status_source"] = "conflicting_structured_evidence"
    return event


def normalize_tool_events(
    record: dict[str, Any],
    raw_messages: list[dict[str, Any]],
    state: NormalizationState,
    *,
    privacy_enabled: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def add(value: Any, *, kind: str, message_index: int | None) -> None:
        for entry in candidate_entries(value):
            effective_kind = kind
            if isinstance(entry, dict):
                entry_type = (normalize_token(entry.get("type")) or "").lower()
                if entry_type in {
                    "tool_use",
                    "tool-call",
                    "tool_call",
                    "tool",
                    "function_call",
                    "custom_tool_call",
                }:
                    effective_kind = "action"
                elif entry_type in {
                    "tool_result",
                    "tool-response",
                    "tool_response",
                    "function_call_output",
                    "custom_tool_call_output",
                }:
                    effective_kind = "observation"
                elif entry_type in {"diff", "edit"}:
                    effective_kind = "artifact"
            if effective_kind == "action" and is_hidden_reasoning_tool(tool_name(entry)):
                # Dedicated reasoning tools are provider control machinery,
                # not observable engineering actions.  Remove their payload
                # and call rather than teaching a student to invoke them.
                state.reasoning_fields_removed += 1
                continue
            event = make_tool_event(
                entry,
                kind=effective_kind,
                message_index=message_index,
                state=state,
                privacy_enabled=privacy_enabled,
            )
            if effective_kind == "artifact":
                event = {
                    "kind": "artifact",
                    "message_index": message_index,
                    "artifact": clean_value(entry, state, privacy_enabled=privacy_enabled),
                }
            if event not in events:
                events.append(event)

    def add_artifacts(value: Any, *, message_index: int | None) -> None:
        for entry in candidate_entries(value):
            events.append(
                {
                    "kind": "artifact",
                    "message_index": message_index,
                    "artifact": clean_value(entry, state, privacy_enabled=privacy_enabled),
                }
            )

    for key in TOOL_CALL_KEYS:
        if key in record:
            add(record[key], kind="action", message_index=None)
    for key in TOOL_RESULT_KEYS:
        if key in record:
            add(record[key], kind="observation", message_index=None)
    for key in ARTIFACT_KEYS:
        if key in record:
            add_artifacts(record[key], message_index=None)
    for key in CONTEXT_KEYS:
        if key in record:
            events.append(
                {
                    "kind": "context",
                    "message_index": None,
                    "context": clean_value(record[key], state, privacy_enabled=privacy_enabled),
                }
            )

    for message_index, raw in enumerate(raw_messages):
        if normalize_role(raw.get("role") or raw.get("type")) == "tool":
            has_explicit_result = any(key in raw for key in TOOL_RESULT_KEYS)
            has_structured_result = any(
                normalized_key(part.get("type")) in {"tool_result", "tool_response"}
                for part in structured_content_parts(raw.get("content"))
            )
            if not has_explicit_result and not has_structured_result:
                # Several adapters represent an observation as an ordinary
                # role=tool message.  Promote that message to an event while
                # retaining its tool_call_id; otherwise the call survives but
                # its observation silently disappears from tool training.
                observation = dict(raw)
                observation["output"] = raw.get(
                    "content", raw.get("message", raw.get("text", ""))
                )
                add(observation, kind="observation", message_index=message_index)
        for key in TOOL_CALL_KEYS:
            if key in raw:
                add(raw[key], kind="action", message_index=message_index)
        for key in TOOL_RESULT_KEYS:
            if key in raw:
                add(raw[key], kind="observation", message_index=message_index)
        for key in ARTIFACT_KEYS:
            if key in raw:
                add_artifacts(raw[key], message_index=message_index)
        for key in CONTEXT_KEYS:
            if key in raw:
                events.append(
                    {
                        "kind": "context",
                        "message_index": message_index,
                        "context": clean_value(raw[key], state, privacy_enabled=privacy_enabled),
                    }
                )
        for part in structured_content_parts(raw.get("content")):
            part_type = normalized_key(part.get("type"))
            if part_type == "tool_use":
                add(part, kind="action", message_index=message_index)
            elif part_type in {"tool_result", "tool_response"}:
                add(part, kind="observation", message_index=message_index)

    return events


def explicit_outcome(record: dict[str, Any]) -> tuple[str, str]:
    value = first_value(record, ("outcome", "result", "status"))
    if isinstance(value, bool):
        return ("success" if value else "failure"), "explicit"
    token = (normalize_token(value) or "").lower()
    if token in OUTCOMES:
        return token, "explicit"
    success = record.get("success")
    if isinstance(success, bool):
        return ("success" if success else "failure"), "explicit"
    return "unknown", "unscored"


def explicit_reward(record: dict[str, Any]) -> tuple[float | int | None, str]:
    value = record.get("reward")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value, "source"
    if value is not None:
        return None, "invalid"
    return None, "unscored"


def _raw_visible_action_ids(value: Any) -> set[str]:
    """Collect observable tool-call IDs without counting hidden thought tools."""

    if not isinstance(value, dict):
        return set()
    result: set[str] = set()

    def add_entry(entry: Any) -> None:
        if not isinstance(entry, dict) or is_hidden_reasoning_tool(tool_name(entry)):
            return
        identifier = first_value(
            entry,
            ("id", "callID", "call_id", "toolCallId", "tool_call_id", "tool_use_id"),
        )
        if identifier is not None:
            result.add(str(identifier))

    for key in TOOL_CALL_KEYS:
        if key in value:
            for entry in candidate_entries(value[key]):
                add_entry(entry)
    for part in structured_content_parts(value.get("content")):
        if normalized_key(part.get("type")) in {"tool_use", "tool_call"}:
            add_entry(part)
    return result


def _raw_quality_event_ids(record: Any) -> tuple[set[str], set[str], int, int, int]:
    """Return action/observation IDs and error counts from one raw record.

    This is intentionally a structural pass.  It does not retain text and it
    uses the same visible tool-call policy as canonical normalization so a
    hidden reasoning tool cannot downgrade an otherwise valid session.
    """

    if not isinstance(record, dict):
        return set(), set(), 0, 0, 0
    values = [record]
    messages = message_list(record)
    if messages:
        values.extend(message for message in messages if isinstance(message, dict))
    action_ids: set[str] = set()
    observation_ids: set[str] = set()
    action_count = 0
    observation_count = 0
    tool_error_count = 0
    for value in values:
        local_actions = _raw_visible_action_ids(value)
        local_observations = _raw_observation_ids(value)
        action_ids.update(local_actions)
        observation_ids.update(local_observations)
        action_count += len(local_actions)
        observation_count += len(local_observations)
        if (
            value.get("isError") is True
            or str(value.get("status", "")).lower() in {"error", "failed", "failure"}
            or value.get("error") not in (None, "", False)
        ):
            tool_error_count += 1
    for event in record.get("events", []) if isinstance(record.get("events"), list) else []:
        if not isinstance(event, dict):
            continue
        kind = normalize_token(event.get("kind"))
        call_id = first_value(event, ("call_id", "tool_call_id", "id"))
        if call_id is None:
            continue
        if kind == "action" and not is_hidden_reasoning_tool(event.get("name")):
            action_ids.add(str(call_id))
            action_count += 1
        elif kind == "observation":
            observation_ids.add(str(call_id))
            observation_count += 1
        if event_status(event.get("status")) == "failure":
            tool_error_count += 1
    return action_ids, observation_ids, action_count, observation_count, tool_error_count


def _harness_outcome_evidence(record: Any) -> tuple[bool, int]:
    """Read aggregate harness outcome evidence without promoting tool calls.

    Some ingress adapters keep the session-level harness summary under
    ``harness_summary`` (and older snapshots may also retain it under
    ``source_origin.quality_summary``). The quality index owns the session
    gate, so it must consume that evidence. This deliberately returns only
    aggregate outcome presence and a bounded signal count: it never claims
    that an individual action window was verified.
    """

    if not isinstance(record, dict):
        return False, 0
    candidates: list[Any] = []
    for key in ("harness_summary", "quality_assessment"):
        value = record.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    source_origin = record.get("source_origin")
    if isinstance(source_origin, dict):
        value = source_origin.get("quality_summary")
        if isinstance(value, dict):
            candidates.append(value)

    observed = False
    terminal_signals = 0
    for candidate in candidates:
        evidence = candidate.get("outcome_evidence")
        if not isinstance(evidence, dict):
            dimensions = candidate.get("dimensions")
            evidence = (
                dimensions.get("outcome_evidence")
                if isinstance(dimensions, dict)
                else None
            )
        if not isinstance(evidence, dict):
            continue
        status = normalize_token(evidence.get("status"))
        if status in {"observed", "verified", "success", "pass"}:
            observed = True
        try:
            terminal_signals = max(
                terminal_signals,
                int(evidence.get("terminal_signals") or 0),
            )
        except (TypeError, ValueError):
            continue
    return observed or terminal_signals > 0, terminal_signals


def _raw_visible_message(value: Any) -> bool:
    """Detect visible message content without retaining or exporting it."""

    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_raw_visible_message(item) for item in value)
    if isinstance(value, dict):
        part_type = normalized_key(value.get("type", ""))
        if part_type in HIDDEN_TRAINER_KEYS | {"thinking", "analysis", "reasoning", "thought"}:
            return False
        if part_type in {"tool_use", "tool_call"}:
            return not is_hidden_reasoning_tool(value.get("name"))
        for key in ("text", "content", "rawText", "message", "value"):
            if key in value and _raw_visible_message(value[key]):
                return True
        return any(
            _raw_visible_message(entry)
            for key in TOOL_CALL_KEYS
            for entry in (candidate_entries(value[key]) if key in value else [])
            if isinstance(entry, dict) and not is_hidden_reasoning_tool(tool_name(entry))
        )
    return bool(value) if value is not None else False


def _raw_model_values(record: Any) -> tuple[set[str], set[str]]:
    models: set[str] = set()
    providers: set[str] = set()
    if not isinstance(record, dict):
        return models, providers
    values = [record]
    messages = message_list(record)
    if messages:
        values.extend(message for message in messages if isinstance(message, dict))
    for value in values:
        for key in ("model", "modelID", "model_id", "modelId", "modelName"):
            item = value.get(key)
            if item not in (None, ""):
                models.add(str(item))
        for key in (
            "provider",
            "providerID",
            "provider_id",
            "api",
            "model_provider",
            "deployment_provider",
        ):
            item = value.get(key)
            if item not in (None, ""):
                providers.add(str(item))
    return models, providers


def _merge_pending_tool_ids(
    pending_actions: set[str],
    pending_observations: set[str],
    action_ids: set[str],
    observation_ids: set[str],
) -> None:
    """Reconcile one record while retaining only currently unmatched IDs."""

    matched = action_ids & observation_ids
    action_ids = action_ids - matched
    observation_ids = observation_ids - matched
    for call_id in action_ids:
        if call_id in pending_observations:
            pending_observations.remove(call_id)
        else:
            pending_actions.add(call_id)
    for call_id in observation_ids:
        if call_id in pending_actions:
            pending_actions.remove(call_id)
        else:
            pending_observations.add(call_id)


def _quality_override_for_session(
    record: Any,
    *,
    source_file_hash: str,
    quality_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Use the existing provider-neutral override lookup during preflight."""

    return quality_override_for(
        record,
        source_file_hash=source_file_hash,
        quality_overrides=quality_overrides,
    )


def _finalize_session_quality(
    accumulator: SessionQualityAccumulator,
    *,
    quality_override: dict[str, Any] | None = None,
    model_tier_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the one gate inherited by every segment of a source session."""

    flags = set(accumulator.explicit_flags)
    if accumulator.advisor_overlay:
        flags.add("advisor_overlay_contamination")
    if accumulator.parse_error_count or accumulator.source_parse_error_count:
        flags.add("source_parse_errors")
    if accumulator.message_count == 0:
        flags.add("no_normalized_messages")
    if accumulator.user_message_count == 0:
        flags.add("missing_user_objective")
    if accumulator.assistant_message_count == 0:
        flags.add("missing_visible_assistant_response")
    if accumulator.pending_action_ids:
        flags.add("unmatched_tool_calls")
    if accumulator.pending_observation_ids:
        flags.add("unmatched_tool_observations")
    if accumulator.truncated_payload_count:
        flags.add("payload_truncated")
    if accumulator.tool_error_count:
        flags.add("tool_error_observed")
    if not accumulator.explicit_outcome_observed:
        flags.add("outcome_unverified")

    model_values = sorted(accumulator.model_values)
    provider_values = sorted(accumulator.provider_values)
    provenance_values = [value.lower() for value in (*model_values, *provider_values)]
    local_tokens = (
        "local",
        "ollama",
        "llama.cpp",
        "llamacpp",
        "lmstudio",
        "self-host",
        "self_host",
    )
    if any(token in value for value in provenance_values for token in local_tokens):
        model_provenance = "local_or_self_hosted"
        flags.add("model_provenance_local_or_self_hosted")
    elif provenance_values:
        model_provenance = "identified_or_unknown"
    else:
        model_provenance = "unknown"
        flags.add("model_provenance_unknown")

    if model_tier_override is not None:
        tier_info = model_tier_for(accumulator, override=model_tier_override)
    elif len(accumulator.model_tiers) == 1:
        tier_info = {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": next(iter(accumulator.model_tiers)),
            "basis": next(iter(accumulator.model_tier_bases), "session_aggregate"),
            "confidence": "declared" if accumulator.model_tier_bases == {"adapter_declared"} else "aggregate",
            "conflict": False,
            "override": False,
        }
    elif len(accumulator.model_tiers) > 1:
        tier_info = {
            "schema_version": MODEL_TIER_SCHEMA_VERSION,
            "tier": "unclassified",
            "basis": "conflicting_session_tiers",
            "confidence": "conflict",
            "conflict": True,
            "override": False,
        }
    else:
        tier_info = model_tier_for(None)
    if tier_info["tier"] == "unclassified":
        flags.add("model_tier_unclassified")
    if tier_info["conflict"]:
        flags.add("conflicting_model_tiers")

    automatic_gate = "candidate"
    if accumulator.message_count == 0:
        automatic_gate = "quarantine"
    elif (
        accumulator.advisor_overlay
        or accumulator.parse_error_count
        or accumulator.source_parse_error_count
    ):
        automatic_gate = "quarantine"
    elif flags & {
        "missing_user_objective",
        "missing_visible_assistant_response",
        "unmatched_tool_calls",
        "unmatched_tool_observations",
        "payload_truncated",
        "no_normalized_messages",
        "conflicting_model_tiers",
        "conflicting_model_tier_overrides",
    }:
        automatic_gate = "review_required"

    if len(accumulator.explicit_gates) > 1:
        flags.add("conflicting_session_quality_assessments")
        automatic_gate = max(
            (automatic_gate, "review_required"),
            key=lambda gate: QUALITY_GATE_RANK[gate],
        )
    elif accumulator.explicit_gates:
        explicit_gate = next(iter(accumulator.explicit_gates))
        automatic_gate = max(
            (automatic_gate, explicit_gate),
            key=lambda gate: QUALITY_GATE_RANK[gate],
        )

    assessment: dict[str, Any] = {
        "assessment_version": "session-quality/v1",
        "scope": "source_session",
        "session_quality_id": accumulator.session_quality_id,
        "gate": automatic_gate,
        "automatic_gate": automatic_gate,
        "flags": sorted(flags),
        "dimensions": {
            "dialogue": {
                "source_records": accumulator.record_count,
                "messages": accumulator.message_count,
                "user_messages": accumulator.user_message_count,
                "assistant_messages": accumulator.assistant_message_count,
                "tool_messages": accumulator.tool_message_count,
                "visible_messages": accumulator.nonempty_message_count,
            },
            "tool_trace_integrity": {
                "actions": accumulator.action_count,
                "observations": accumulator.observation_count,
                "unmatched_calls": len(accumulator.pending_action_ids),
                "unmatched_observations": len(accumulator.pending_observation_ids),
                "errors": accumulator.tool_error_count,
            },
            "payload_integrity": {
                "truncated_payloads": accumulator.truncated_payload_count,
                "parse_errors": accumulator.parse_error_count
                + accumulator.source_parse_error_count,
            },
            "outcome_evidence": {
                "status": "observed" if accumulator.explicit_outcome_observed else "unverified",
                "terminal_signals": accumulator.terminal_signal_count,
            },
            "model_provenance": {
                "status": model_provenance,
                "models": model_values,
                "providers": provider_values,
                "model_tier": tier_info["tier"],
                "model_tier_basis": tier_info["basis"],
            },
        },
        "model_tier": tier_info["tier"],
        "model_tier_basis": tier_info["basis"],
        "model_tier_confidence": tier_info["confidence"],
        "model_tier_registry_revision": MODEL_TIER_REGISTRY_REVISION,
        "method": (
            "adapter_plus_builder_session_v1"
            if accumulator.explicit_gates
            else "builder_structural_session_v1"
        ),
        "provider_neutral": True,
    }
    if len(accumulator.explicit_gates) > 1:
        assessment["explicit_gates"] = sorted(accumulator.explicit_gates)
    if quality_override is not None:
        assessment = apply_quality_override(assessment, quality_override)
    return assessment


def build_session_quality_index(
    files: Iterable[Path],
    *,
    quality_overrides: dict[str, dict[str, Any]] | None = None,
    model_tier_overrides: dict[str, dict[str, Any]] | None = None,
    max_records_per_file: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Preflight every input and return provider-neutral session assessments.

    This pass is intentionally independent of lane selection.  A session is
    assessed even when its provenance lane is excluded from the current
    training build, so a later optional-lane experiment does not silently
    inherit a provider-wide verdict.
    """

    quality_overrides = quality_overrides or {}
    model_tier_overrides = model_tier_overrides or {}
    accumulators: dict[str, SessionQualityAccumulator] = {}
    file_hashes: dict[str, str] = {}
    override_by_session: dict[str, dict[str, Any]] = {}
    override_conflicts: set[str] = set()
    tier_override_by_session: dict[str, dict[str, Any]] = {}
    tier_override_conflicts: set[str] = set()
    for source_file in files:
        source_hash = sha256_file(source_file)
        file_hashes[str(source_file.resolve())] = source_hash
        file_record_count = 0
        for _source_line, record in iter_jsonl(source_file):
            if (
                max_records_per_file is not None
                and file_record_count >= max_records_per_file
            ):
                break
            file_record_count += 1
            session_key = quality_session_key(record, source_hash)
            session_quality_id = stable_id(
                {"source_file_sha256": source_hash, "session_key": session_key}
            )
            accumulator = accumulators.setdefault(
                session_key,
                SessionQualityAccumulator(
                    session_key=session_key,
                    session_quality_id=session_quality_id,
                ),
            )
            accumulator.record_count += 1
            if isinstance(record, dict):
                source_class = normalize_token(record.get("source_class"))
                if source_class:
                    accumulator.source_classes.add(source_class)
                lane = training_lane_for(record)
                accumulator.training_lanes.add(lane)
                record_tier_override = model_tier_override_for(
                    record,
                    source_file_hash=source_hash,
                    model_tier_overrides=model_tier_overrides,
                )
                tier_info = model_tier_for(
                    record,
                    messages=message_list(record) or [],
                    override=record_tier_override,
                )
                accumulator.model_tiers.add(tier_info["tier"])
                accumulator.model_tier_bases.add(tier_info["basis"])
                if tier_info["conflict"]:
                    accumulator.explicit_flags.add("conflicting_model_tiers")
                if record_tier_override is not None:
                    existing_tier_override = tier_override_by_session.get(session_key)
                    if existing_tier_override is not None and existing_tier_override != record_tier_override:
                        tier_override_conflicts.add(session_key)
                    else:
                        tier_override_by_session[session_key] = record_tier_override
                accumulator.advisor_overlay = accumulator.advisor_overlay or bool(
                    record.get("advisor_overlay")
                    or "advisor" in (source_class or "").lower()
                )
                try:
                    accumulator.parse_error_count += int(record.get("parse_errors") or 0)
                except (TypeError, ValueError):
                    accumulator.parse_error_count += 1
                try:
                    accumulator.source_parse_error_count += int(
                        record.get("source_parse_errors") or 0
                    )
                except (TypeError, ValueError):
                    accumulator.source_parse_error_count += 1
                truncations = record.get("observation_truncations")
                if isinstance(truncations, list):
                    accumulator.truncated_payload_count += len(truncations)
                models, providers = _raw_model_values(record)
                accumulator.model_values.update(models)
                accumulator.provider_values.update(providers)
                value = record.get("quality_assessment")
                if isinstance(value, dict):
                    gate = value.get("gate") or value.get("quality_gate")
                    if gate in QUALITY_GATES:
                        accumulator.explicit_gates.add(gate)
                    flags = value.get("flags")
                    if isinstance(flags, list):
                        accumulator.explicit_flags.update(str(flag) for flag in flags)
                harness_outcome_observed, harness_terminal_signals = _harness_outcome_evidence(
                    record
                )
                if harness_outcome_observed:
                    accumulator.explicit_outcome_observed = True
                if harness_terminal_signals:
                    # Chunked ingress repeats the session summary on each
                    # segment; retain the session-level maximum instead of
                    # counting the same terminal event once per chunk.
                    accumulator.terminal_signal_count = max(
                        accumulator.terminal_signal_count,
                        harness_terminal_signals,
                    )
                messages = message_list(record) or []
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    role = _raw_message_role(message)
                    if role is None:
                        continue
                    accumulator.message_count += 1
                    if _raw_visible_message(message.get("content", message.get("message", message.get("text", "")))) or _raw_message_has_tool_call(message):
                        accumulator.nonempty_message_count += 1
                    if role == "user":
                        accumulator.user_message_count += 1
                    elif role == "assistant":
                        accumulator.assistant_message_count += 1
                    elif role == "tool":
                        accumulator.tool_message_count += 1
                action_ids, observation_ids, action_count, observation_count, error_count = _raw_quality_event_ids(record)
                accumulator.action_count += action_count
                accumulator.observation_count += observation_count
                accumulator.tool_error_count += error_count
                _merge_pending_tool_ids(
                    accumulator.pending_action_ids,
                    accumulator.pending_observation_ids,
                    action_ids,
                    observation_ids,
                )
                outcome, _outcome_source = explicit_outcome(record)
                if outcome != "unknown":
                    accumulator.explicit_outcome_observed = True
                    accumulator.terminal_signal_count += 1
                try:
                    accumulator.terminal_signal_count += int(
                        record.get("terminal_signal_count") or 0
                    )
                except (TypeError, ValueError):
                    pass
                override = _quality_override_for_session(
                    record,
                    source_file_hash=source_hash,
                    quality_overrides=quality_overrides,
                )
                if override is not None:
                    existing = override_by_session.get(session_key)
                    if existing is not None and existing != override:
                        override_conflicts.add(session_key)
                    else:
                        override_by_session[session_key] = override

    assessments: dict[str, dict[str, Any]] = {}
    session_gate_counts: Counter[str] = Counter()
    session_lane_counts: Counter[str] = Counter()
    for session_key, accumulator in accumulators.items():
        if session_key in override_conflicts:
            accumulator.explicit_flags.add("conflicting_quality_overrides")
            override_by_session.pop(session_key, None)
        if session_key in tier_override_conflicts:
            accumulator.explicit_flags.add("conflicting_model_tier_overrides")
            tier_override_by_session.pop(session_key, None)
        assessment = _finalize_session_quality(
            accumulator,
            quality_override=override_by_session.get(session_key),
            model_tier_override=tier_override_by_session.get(session_key),
        )
        assessments[session_key] = assessment
        session_gate_counts[assessment["gate"]] += 1
        for lane in accumulator.training_lanes:
            session_lane_counts[lane] += 1
    return assessments, {
        "schema_version": "ai-data-extraction/session-quality-index/v1",
        "scope": "all_selected_input_files",
        "file_hashes": file_hashes,
        "sessions": len(assessments),
        "gate_counts": dict(sorted(session_gate_counts.items())),
        "lane_counts": dict(sorted(session_lane_counts.items())),
        "override_conflicts": len(override_conflicts),
        "model_tier_counts": dict(
            sorted(
                Counter(
                    assessment.get("model_tier", "unclassified")
                    for assessment in assessments.values()
                ).items()
            )
        ),
        "model_tier_override_conflicts": len(tier_override_conflicts),
    }


def task_family(user_text: str) -> str:
    text = user_text.lower()
    if any(token in text for token in ("debug", "bug", "error", "exception", "crash", "failing")):
        return "debugging"
    if any(token in text for token in ("test", "pytest", "vitest", "unittest", "coverage")):
        return "testing"
    if any(token in text for token in ("review", "audit", "security", "blast radius")):
        return "review"
    if any(token in text for token in ("readme", "documentation", "docs", "explain")):
        return "documentation"
    if any(token in text for token in ("deploy", "config", "environment", "kubernetes", "helm")):
        return "configuration"
    if any(token in text for token in ("how do", "what is", "why does", "can you explain")):
        return "question"
    if any(token in text for token in ("implement", "add ", "change ", "fix ", "refactor", "edit")):
        return "code-edit"
    return "unknown"


def metadata_for(
    record: dict[str, Any],
    *,
    provider: str,
    raw_source: str,
    source_file: Path,
    source_file_hash: str,
    source_line: int,
    source_record_hash: str,
    segment_record_hash: str,
    state: NormalizationState,
    privacy_enabled: bool,
    quality_assessment: dict[str, Any],
) -> dict[str, Any]:
    lineage = record_lineage(record, source_record_hash)
    metadata: dict[str, Any] = {
        "parser_revision": PARSER_REVISION,
        "provider": provider,
        "source_label": clean_text(raw_source, state, privacy_enabled=privacy_enabled),
        "source_file_name": clean_text(source_file.name, state, privacy_enabled=privacy_enabled),
        "source_file_sha256": source_file_hash,
        "source_line": source_line,
        "source_record_sha256": source_record_hash,
        "segment_record_sha256": segment_record_hash,
        "parent_record_sha256": lineage["parent_record_sha256"],
        "chunk_index": lineage["chunk_index"],
        "chunk_count": lineage["chunk_count"],
        "continuation_status": lineage["continuation_status"],
        "open_tool_call_ids": lineage["open_tool_call_ids"],
        "quality_gate": quality_assessment["gate"],
        "quality_flags": quality_assessment["flags"],
        "session_quality_id": quality_assessment.get("session_quality_id"),
        "session_quality_scope": quality_assessment.get("scope", "source_session"),
        "model_tier": quality_assessment.get("model_tier", "unclassified"),
        "model_tier_basis": quality_assessment.get(
            "model_tier_basis", "missing_authoritative_tier"
        ),
        "model_tier_registry_revision": quality_assessment.get(
            "model_tier_registry_revision", MODEL_TIER_REGISTRY_REVISION
        ),
        "quality_assessment": clean_value(
            quality_assessment, state, privacy_enabled=privacy_enabled
        ),
    }
    if lineage["source_message_range"] is not None:
        metadata["source_message_range"] = lineage["source_message_range"]
    source_class = record.get("source_class")
    if source_class is not None and source_class != "":
        metadata["source_class"] = clean_value(
            source_class, state, privacy_enabled=privacy_enabled
        )
    identifier = first_value(
        record,
        ("session_id", "composer_id", "chat_id", "agent_id", "tab_id"),
    )
    metadata["session_id"] = stable_id(identifier if identifier is not None else source_record_hash)
    for key in (
        "model",
        "modelID",
        "model_id",
        "modelId",
        "modelName",
        "model_provider",
        "provider",
        "providerID",
        "provider_id",
        "deployment_provider",
        "model_deployment",
        "mode",
        "agent",
        "title",
        "name",
        "version",
        "created_at",
        "updated_at",
        "start_time",
        "last_updated",
        "timestamp",
        "training_lane",
        "quality_gate",
    ):
        value = record.get(key)
        if value is not None and value != "":
            metadata[key] = clean_value(value, state, privacy_enabled=privacy_enabled)
    source_origin = record.get("source_origin")
    if isinstance(source_origin, dict):
        metadata["source_origin"] = clean_value(
            source_origin, state, privacy_enabled=privacy_enabled
        )
        source_manifest_revision = source_origin.get("source_manifest_revision")
        if isinstance(source_manifest_revision, str) and source_manifest_revision:
            metadata["source_manifest_revision"] = source_manifest_revision
        source_snapshot_status = source_origin.get("source_snapshot_status")
        if isinstance(source_snapshot_status, str) and source_snapshot_status:
            metadata["source_snapshot_status"] = source_snapshot_status
    for key in ("harness_summary",):
        value = record.get(key)
        if isinstance(value, (dict, list)):
            metadata[key] = clean_value(value, state, privacy_enabled=privacy_enabled)
    observation_truncations = record.get("observation_truncations")
    if isinstance(observation_truncations, list) and observation_truncations:
        metadata["observation_truncations"] = clean_value(
            observation_truncations, state, privacy_enabled=privacy_enabled
        )
    for key in PATH_KEYS:
        value = record.get(key)
        if value is not None and value != "":
            metadata[f"{key}_sha256"] = stable_id(str(value))
    for key in IDENTIFIER_KEYS:
        if key == "session_id":
            continue
        value = record.get(key)
        if value is not None and value != "":
            metadata[f"{key}_sha256"] = stable_id(str(value))
    return metadata


def base_tags(
    *,
    provider: str,
    mode: Any,
    family: str,
    has_tools: bool,
    has_diffs: bool,
    has_context: bool,
    has_code: bool,
    tool_family_tags: list[str],
    content_provider_mentions: list[str],
    privacy_mode: str,
    privacy_approved: bool,
    outcome: str,
    training_lane: str,
    quality_gate: Any,
    model_tier: Any,
    quality_flags: Iterable[Any] = (),
) -> list[str]:
    tags = {f"provider:{provider}", f"task:{family}", f"outcome:{outcome}"}
    if normalize_token(mode):
        mode_token = re.sub(r"[^a-z0-9_-]+", "-", str(mode).lower()).strip("-")
        if mode_token:
            tags.add(f"mode:{mode_token[:48]}")
    if has_tools:
        tags.add("has:tools")
    if has_diffs:
        tags.add("has:diffs")
    if has_context:
        tags.add("has:context")
    if has_code:
        tags.add("has:code")
    for tool_family in tool_family_tags:
        tags.add(f"tool:{tool_family}")
    for mentioned_provider in content_provider_mentions:
        tags.add(f"provider-mention:{mentioned_provider}")
    if content_provider_mentions:
        tags.add("content:provider-specific")
    tags.add("has:reasoning-removed")
    tags.add(f"privacy:{privacy_mode}")
    tags.add("privacy:approved" if privacy_approved else "privacy:review")
    tags.add(f"lane:{training_lane}")
    tier = normalize_model_tier(model_tier) or "unclassified"
    tags.add(MODEL_TIER_TAGS[tier])
    tags.add("quality-scope:source-session")
    quality_token = normalize_token(quality_gate)
    if quality_token:
        quality_token = re.sub(r"[^a-z0-9_-]+", "-", quality_token.lower()).strip("-")
        if quality_token:
            tags.add(f"quality-gate:{quality_token[:64]}")
    for flag in quality_flags:
        flag_token = normalize_token(flag)
        if flag_token:
            flag_token = re.sub(r"[^a-z0-9_-]+", "-", flag_token.lower()).strip("-")
            if flag_token:
                tags.add(f"quality-flag:{flag_token[:64]}")
    return sorted(tags)


def privacy_eligibility(privacy_mode: str, privacy_approved: bool) -> tuple[bool, str]:
    if privacy_mode not in PRIVACY_MODES:
        raise ValueError(f"Unsupported privacy mode: {privacy_mode}")
    if privacy_mode == "none":
        return False, "unfiltered"
    if privacy_approved:
        return True, "user-approved-after-filter"
    return False, "review-required"


def split_for(example_id: str) -> str:
    bucket = int(example_id.removeprefix("sha256:")[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def no_reasoning_content(value: Any) -> bool:
    if isinstance(value, str):
        return not any(pattern.search(value) for pattern in REASONING_BLOCK_PATTERNS)
    if isinstance(value, list):
        return all(no_reasoning_content(item) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if normalized_key(key) in HIDDEN_TRAINER_KEYS:
                return False
            if not no_reasoning_content(item):
                return False
    return True


def validate_dataset_record(record: dict[str, Any], dataset: str) -> None:
    required = {"schema_version", "example_id", "dataset", "split", "tags", "quality", "privacy"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{dataset} record missing fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION or record["dataset"] != dataset:
        raise ValueError(f"Invalid schema or dataset for {dataset} record")
    if record["split"] not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split in {dataset} record")
    if not isinstance(record["tags"], list) or not all(isinstance(tag, str) for tag in record["tags"]):
        raise ValueError(f"Invalid tags in {dataset} record")
    if not no_reasoning_content(record):
        raise ValueError(f"Reasoning content survived normalization in {dataset} record")
    if dataset in {"sft", "trajectory", "tool_trace"}:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{dataset} record must contain messages")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ALLOWED_ROLES:
                raise ValueError(f"Invalid message role in {dataset} record")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"Message content must be a string in {dataset} record")
        if "tools" in record and not isinstance(record["tools"], list):
            raise ValueError(f"Tools must be a list in {dataset} record")
    elif dataset == "preference":
        for key in ("prompt", "chosen", "rejected"):
            if not isinstance(record.get(key), list) or not record[key]:
                raise ValueError(f"Preference record must contain non-empty {key}")
            for message in record[key]:
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise ValueError(f"Preference {key} contains an invalid message")
        for key in ("chosen", "rejected"):
            if not any(
                message.get("role") == "assistant"
                for message in record[key]
                if isinstance(message, dict)
            ):
                raise ValueError(f"Preference {key} must contain an assistant response")
    elif dataset == "rl_prompt":
        prompt = record.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("RL prompt record must contain a non-empty prompt")
        if not all(
            isinstance(message, dict)
            and message.get("role") in ALLOWED_ROLES
            and isinstance(message.get("content"), str)
            for message in prompt
        ):
            raise ValueError("RL prompt contains an invalid message")
        if not any(message.get("role") == "user" for message in prompt):
            raise ValueError("RL prompt record must contain a user message")
        if record.get("reward") is not None or record.get("reward_status") != "unscored":
            raise ValueError("RL prompt records must remain unscored")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _action_signature(event: dict[str, Any]) -> str:
    return stable_id({"name": event.get("name") or "unknown", "input": event.get("input")})


def _match_action_observations(
    events: list[dict[str, Any]],
) -> dict[int, tuple[int | None, dict[str, Any] | None, str]]:
    """Assign each observation to at most one action using explicit evidence."""

    actions = [
        (event_index, event)
        for event_index, event in enumerate(events)
        if isinstance(event, dict) and event.get("kind") == "action"
    ]
    observations = [
        (event_index, event)
        for event_index, event in enumerate(events)
        if isinstance(event, dict) and event.get("kind") == "observation"
    ]
    matches: dict[int, tuple[int | None, dict[str, Any] | None, str]] = {
        event_index: (None, None, "unmatched") for event_index, _event in actions
    }
    consumed_observations: set[int] = set()
    actions_by_call_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    observations_by_call_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for event_index, event in actions:
        call_id = event.get("call_id")
        if isinstance(call_id, str) and call_id:
            actions_by_call_id.setdefault(call_id, []).append((event_index, event))
    for event_index, event in observations:
        call_id = event.get("call_id")
        if isinstance(call_id, str) and call_id:
            observations_by_call_id.setdefault(call_id, []).append((event_index, event))

    # A call ID is exact only when it identifies one action and one observation.
    for call_id, action_candidates in actions_by_call_id.items():
        observation_candidates = observations_by_call_id.get(call_id, [])
        if len(action_candidates) != 1 or len(observation_candidates) != 1:
            continue
        action_index, _action = action_candidates[0]
        observation_index, observation = observation_candidates[0]
        matches[action_index] = (observation_index, observation, "call-id")
        consumed_observations.add(observation_index)

    # Event intervals are disjoint. A sole non-conflicting observation between
    # this action and the next action is ordered evidence; multiple candidates
    # remain unmatched instead of being guessed or reused.
    observation_cursor = 0
    for action_ordinal, (action_index, action_event) in enumerate(actions):
        next_action_index = (
            actions[action_ordinal + 1][0]
            if action_ordinal + 1 < len(actions)
            else len(events)
        )
        interval_candidates: list[tuple[int, dict[str, Any]]] = []
        while (
            observation_cursor < len(observations)
            and observations[observation_cursor][0] <= action_index
        ):
            observation_cursor += 1
        scan_cursor = observation_cursor
        while (
            scan_cursor < len(observations)
            and observations[scan_cursor][0] < next_action_index
        ):
            observation_index, observation = observations[scan_cursor]
            if observation_index not in consumed_observations:
                action_call_id = action_event.get("call_id")
                observation_call_id = observation.get("call_id")
                explicit_conflict = (
                    action_call_id
                    and observation_call_id
                    and action_call_id != observation_call_id
                )
                if not explicit_conflict:
                    interval_candidates.append((observation_index, observation))
            scan_cursor += 1
        observation_cursor = scan_cursor
        if matches[action_index][1] is None and len(interval_candidates) == 1:
            observation_index, observation = interval_candidates[0]
            matches[action_index] = (observation_index, observation, "event-order")
            consumed_observations.add(observation_index)

    # Detached legacy fields can place the sole observation before the sole
    # action. Keep that join explicitly heuristic and reject conflicting IDs.
    if len(actions) == 1 and len(observations) == 1:
        action_index, action_event = actions[0]
        observation_index, observation = observations[0]
        action_call_id = action_event.get("call_id")
        observation_call_id = observation.get("call_id")
        explicit_conflict = (
            action_call_id
            and observation_call_id
            and action_call_id != observation_call_id
        )
        if matches[action_index][1] is None and not explicit_conflict:
            matches[action_index] = (
                observation_index,
                observation,
                "singleton-fallback",
            )
    return matches


def _action_turn_index(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Group parallel calls from one assistant message into one sequence turn."""

    groups: list[dict[str, Any]] = []
    by_message: dict[int, dict[str, Any]] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("kind") != "action":
            continue
        message_index = event.get("message_index")
        group = by_message.get(message_index) if isinstance(message_index, int) else None
        if group is None:
            group = {"event_indices": [], "events": []}
            groups.append(group)
            if isinstance(message_index, int):
                by_message[message_index] = group
        group["event_indices"].append(event_index)
        group["events"].append(event)

    result: dict[int, dict[str, Any]] = {}
    signatures: list[str] = []
    signature_counts: Counter[str] = Counter()
    last_signature_index: dict[str, int] = {}
    for ordinal, group in enumerate(groups):
        signature = stable_id(
            [
                {"name": event.get("name") or "unknown", "input": event.get("input")}
                for event in group["events"]
            ]
        )
        prior_occurrences = signature_counts[signature]
        prior_index = last_signature_index.get(signature)
        nearest_distance = ordinal - prior_index if prior_index is not None else None
        current = [*signatures, signature]
        periods = [
            period
            for period in (2, 3)
            if len(current) >= period * 2
            and current[-period * 2 : -period] == current[-period:]
        ]
        next_action_index = (
            groups[ordinal + 1]["event_indices"][0]
            if ordinal + 1 < len(groups)
            else len(events)
        )
        artifact_hashes = sorted(
            {
                stable_id(candidate.get("artifact"))
                for candidate in events[group["event_indices"][-1] + 1 : next_action_index]
                if isinstance(candidate, dict) and candidate.get("kind") == "artifact"
            }
        )
        shared = {
            "turn_ordinal": ordinal,
            "turn_signature": signature,
            "calls_in_turn": len(group["events"]),
            "prior_turn_occurrences": prior_occurrences,
            "nearest_prior_turn_distance": nearest_distance,
            "immediate_repeat": nearest_distance == 1,
            "complete_cycle_periods": periods,
            "artifact_hashes_before_next_action": artifact_hashes,
            "families": tool_families(group["events"], []),
        }
        for event_index in group["event_indices"]:
            result[event_index] = shared
        signatures.append(signature)
        signature_counts[signature] += 1
        last_signature_index[signature] = ordinal
    return result


def _action_evidence(
    *,
    trajectory: dict[str, Any],
    event: dict[str, Any],
    observation: dict[str, Any] | None,
    observation_match: str,
    turn: dict[str, Any],
    prior_observation_digests: list[str],
) -> dict[str, Any]:
    action_signature = _action_signature(event)
    output_digest = (
        stable_id(
            {
                "status": observation.get("status"),
                "status_source": observation.get("status_source"),
                "result_code": observation.get("result_code"),
                "output": observation.get("output"),
            }
        )
        if observation is not None
        else None
    )
    novelty = (
        output_digest != prior_observation_digests[-1]
        if output_digest is not None and prior_observation_digests
        else None
    )
    trajectory_outcome = trajectory.get("trajectory", {})
    if not isinstance(trajectory_outcome, dict):
        trajectory_outcome = {}
    return {
        "schema_version": ACTION_EVIDENCE_SCHEMA_VERSION,
        "positive_target_status": "not_adjudicated",
        "action": {
            "signature": action_signature,
            "turn_signature": turn["turn_signature"],
            "turn_ordinal": turn["turn_ordinal"],
            "calls_in_turn": turn["calls_in_turn"],
            "families": turn["families"],
        },
        "observation": {
            "joined": observation is not None,
            "match": observation_match,
            "match_strength": {
                "call-id": "exact",
                "event-order": "ordered",
                "singleton-fallback": "heuristic",
                "unmatched": "absent",
            }[observation_match],
            "status": observation.get("status", "unknown") if observation is not None else "unknown",
            "status_source": observation.get("status_source", "absent")
            if observation is not None
            else "absent",
            "result_code": observation.get("result_code") if observation is not None else None,
            "result_code_source": observation.get("result_code_source", "absent")
            if observation is not None
            else "absent",
            "output_digest": output_digest,
            "novel_for_same_action": novelty,
        },
        "sequence": {
            "prior_turn_occurrences": turn["prior_turn_occurrences"],
            "nearest_prior_turn_distance": turn["nearest_prior_turn_distance"],
            "immediate_repeat": turn["immediate_repeat"],
            "complete_cycle_periods": turn["complete_cycle_periods"],
        },
        "artifacts": {
            "hashes_before_next_action": turn["artifact_hashes_before_next_action"],
            "content_included": False,
        },
        "episode_outcome": {
            "value": trajectory_outcome.get("outcome", "unknown"),
            "source": trajectory_outcome.get("outcome_source", "unscored"),
            "step_credit": "absent",
        },
    }


def validate_action_window(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "window_id",
        "episode_id",
        "context",
        "decision",
        "tool_call",
        "observation",
        "evidence",
        "verification",
        "quality",
        "provenance",
        "privacy",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Action window missing fields: {sorted(missing)}")
    if record["schema_version"] != ACTION_WINDOW_SCHEMA_VERSION:
        raise ValueError("Invalid action-window schema")
    if not isinstance(record["window_id"], str) or not isinstance(record["episode_id"], str):
        raise ValueError("Action-window identities must be strings")
    context = record["context"]
    if not isinstance(context, dict) or not isinstance(context.get("messages"), list):
        raise ValueError("Action-window context must contain messages")
    for message in context["messages"]:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ALLOWED_ROLES
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Action-window context contains an invalid message")
    decision = record["decision"]
    if not isinstance(decision, dict) or decision.get("action") not in {"use", "skip", "defer", "ask"}:
        raise ValueError("Action-window decision is invalid")
    if decision["action"] == "use":
        call = record["tool_call"]
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("Tool-use action window must contain a named call")
    if record["observation"] is not None and not isinstance(record["observation"], dict):
        raise ValueError("Action-window observation must be an object or null")
    evidence = record["evidence"]
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != ACTION_EVIDENCE_SCHEMA_VERSION
        or evidence.get("positive_target_status") != "not_adjudicated"
    ):
        raise ValueError("Action-window evidence contract is invalid")
    action = evidence.get("action")
    sequence = evidence.get("sequence")
    observation_evidence = evidence.get("observation")
    artifacts = evidence.get("artifacts")
    episode_outcome = evidence.get("episode_outcome")
    if (
        not isinstance(action, dict)
        or not isinstance(action.get("signature"), str)
        or not isinstance(action.get("turn_signature"), str)
        or not isinstance(action.get("turn_ordinal"), int)
        or action["turn_ordinal"] < 0
        or not isinstance(action.get("calls_in_turn"), int)
        or action["calls_in_turn"] <= 0
        or not isinstance(action.get("families"), list)
        or not all(isinstance(family, str) for family in action["families"])
    ):
        raise ValueError("Action-window action evidence is invalid")
    cycle_periods = sequence.get("complete_cycle_periods") if isinstance(sequence, dict) else None
    if (
        not isinstance(sequence, dict)
        or not isinstance(sequence.get("prior_turn_occurrences"), int)
        or sequence["prior_turn_occurrences"] < 0
        or not isinstance(sequence.get("immediate_repeat"), bool)
        or not isinstance(cycle_periods, list)
        or not all(
            isinstance(period, int)
            and not isinstance(period, bool)
            and period in {2, 3}
            for period in cycle_periods
        )
        or len(cycle_periods) != len(set(cycle_periods))
    ):
        raise ValueError("Action-window sequence evidence is invalid")
    nearest = sequence.get("nearest_prior_turn_distance")
    if nearest is not None and (not isinstance(nearest, int) or nearest <= 0):
        raise ValueError("Action-window recurrence distance is invalid")
    if sequence["immediate_repeat"] != (nearest == 1):
        raise ValueError("Action-window immediate-repeat evidence is inconsistent")
    if sequence["prior_turn_occurrences"] == 0 and nearest is not None:
        raise ValueError("Action-window prior-occurrence evidence is inconsistent")
    observation_joined = record["observation"] is not None
    expected_matches = {
        "call-id": "exact",
        "event-order": "ordered",
        "singleton-fallback": "heuristic",
        "unmatched": "absent",
    }
    if (
        not isinstance(observation_evidence, dict)
        or observation_evidence.get("joined") != observation_joined
        or observation_evidence.get("match") not in expected_matches
        or observation_evidence.get("match_strength")
        != expected_matches[observation_evidence["match"]]
        or observation_joined == (observation_evidence["match"] == "unmatched")
        or observation_evidence.get("status") not in {"success", "failure", "unknown"}
        or observation_evidence.get("status_source")
        not in {
            "structured_status",
            "structured_error",
            "structured_result_code",
            "conflicting_structured_evidence",
            "absent",
        }
        or observation_evidence.get("result_code_source")
        not in {"structured_field", "conflicting_structured_fields", "absent"}
        or (
            observation_evidence.get("result_code") is not None
            and (
                not isinstance(observation_evidence["result_code"], int)
                or isinstance(observation_evidence["result_code"], bool)
            )
        )
        or (
            observation_evidence.get("output_digest") is not None
            and not isinstance(observation_evidence["output_digest"], str)
        )
        or observation_evidence.get("novel_for_same_action")
        not in (True, False, None)
    ):
        raise ValueError("Action-window observation evidence is inconsistent")
    if not observation_joined and any(
        observation_evidence.get(key) is not None
        for key in ("result_code", "output_digest", "novel_for_same_action")
    ):
        raise ValueError("Missing observations cannot carry derived evidence")
    artifact_hashes = (
        artifacts.get("hashes_before_next_action") if isinstance(artifacts, dict) else None
    )
    if (
        not isinstance(artifacts, dict)
        or not isinstance(artifact_hashes, list)
        or not all(isinstance(value, str) for value in artifact_hashes)
        or artifact_hashes != sorted(set(artifact_hashes))
        or artifacts.get("content_included") is not False
    ):
        raise ValueError("Action-window artifact evidence is invalid")
    if (
        not isinstance(episode_outcome, dict)
        or episode_outcome.get("value") not in OUTCOMES
        or not isinstance(episode_outcome.get("source"), str)
        or episode_outcome.get("step_credit") != "absent"
    ):
        raise ValueError("Action-window episode outcome evidence is invalid")
    if not no_reasoning_content(record):
        raise ValueError("Reasoning content survived action-window normalization")


def bounded_window_payload(value: Any, *, source_status: str) -> tuple[Any, str]:
    """Keep action-window payloads bounded and identify every omission."""
    if source_status != "accepted":
        return None, f"omitted_{source_status}"
    size = canonical_json_size(value)
    if size <= ACTION_WINDOW_PAYLOAD_MAX_CHARS:
        return value, "full"
    if isinstance(value, str):
        marker = "\n<OBSERVATION_TRUNCATED_FOR_ACTION_WINDOW>\n"
        available = max(2, ACTION_WINDOW_PAYLOAD_MAX_CHARS - len(marker))
        head = max(1, int(available * 0.7))
        tail = max(1, available - head)
        return value[:head] + marker + value[-tail:], "truncated"
    return {
        "truncated": True,
        "sha256": stable_id(value),
        "serialized_chars": size,
    }, "digest_only"


def action_windows_for(
    trajectory: dict[str, Any], *, source_status: str = "accepted"
) -> list[dict[str, Any]]:
    """Project observed tool transitions into reviewable action windows.

    This projection deliberately does not claim that a skill was read, that a
    tool was necessary, or that an outcome was verified.  Those are harness
    annotations added later when the execution environment can prove them.
    """
    events = trajectory.get("events") if isinstance(trajectory.get("events"), list) else []
    messages = trajectory.get("messages") if isinstance(trajectory.get("messages"), list) else []
    windows: list[dict[str, Any]] = []
    observation_matches = _match_action_observations(events)
    turn_index = _action_turn_index(events)
    observation_digests_by_action: dict[str, list[str]] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("kind") != "action":
            continue
        call_id = event.get("call_id")
        observation_index, observation, observation_match = observation_matches[
            event_index
        ]

        message_index = event.get("message_index")
        context_end = (
            min(message_index + 1, len(messages))
            if isinstance(message_index, int) and message_index >= 0
            else len(messages)
        )
        bounded_start = max(0, context_end - 8)
        context_messages = [dict(message) for message in messages[bounded_start:context_end]]
        latest_user_index = max(
            (
                index
                for index, message in enumerate(messages[:context_end])
                if message.get("role") == "user"
            ),
            default=-1,
        )
        if latest_user_index >= 0 and not any(
            index == latest_user_index
            for index in range(bounded_start, context_end)
        ):
            recent_start = max(latest_user_index + 1, context_end - 7)
            context_messages = [
                dict(messages[latest_user_index]),
                *[dict(message) for message in messages[recent_start:context_end]],
            ]
        prior_events = events[:event_index]
        prior_event_ids = [
            prior_event.get("event_id")
            for prior_event in prior_events
            if isinstance(prior_event, dict) and prior_event.get("event_id")
        ]
        tool_registry = trajectory.get("tools", [])
        window_id = stable_id(
            {
                "dataset": "action_window",
                "episode_id": trajectory["example_id"],
                "event_index": event_index,
                "action_event_id": event.get("event_id"),
                "observation_event_id": (
                    observation.get("event_id") if observation is not None else None
                ),
            }
        )
        quality_stage = "candidate"
        if source_status != "accepted":
            quality_stage = f"{source_status}-candidate"
        if observation is not None and trajectory.get("trajectory", {}).get("outcome_source") == "explicit":
            quality_stage = "source-outcome-candidate"
        bounded_arguments, argument_policy = bounded_window_payload(
            event.get("input"), source_status=source_status
        )
        bounded_output = None
        output_policy = "absent"
        if observation is not None:
            bounded_output, output_policy = bounded_window_payload(
                observation.get("output"), source_status=source_status
            )
        action_signature = _action_signature(event)
        evidence = _action_evidence(
            trajectory=trajectory,
            event=event,
            observation=observation,
            observation_match=observation_match,
            turn=turn_index[event_index],
            prior_observation_digests=observation_digests_by_action.get(
                action_signature, []
            ),
        )
        source_metadata = trajectory.get("metadata", {})
        provenance = {
            "parser_revision": source_metadata.get("parser_revision", PARSER_REVISION),
            "episode_id": trajectory["example_id"],
            "source_file_name": source_metadata.get("source_file_name"),
            "source_file_sha256": source_metadata.get("source_file_sha256"),
            "source_record_sha256": source_metadata.get("source_record_sha256"),
            "segment_record_sha256": source_metadata.get("segment_record_sha256"),
            "parent_record_sha256": source_metadata.get("parent_record_sha256"),
            "source_manifest_revision": source_metadata.get("source_manifest_revision"),
            "source_snapshot_status": source_metadata.get("source_snapshot_status"),
            "source_class": source_metadata.get("source_class"),
            "source_line": source_metadata.get("source_line"),
            "source_event_index": event_index,
            "source_event_id": event.get("event_id"),
            "observation_event_index": observation_index,
            "observation_event_id": (
                observation.get("event_id") if observation is not None else None
            ),
            "episode_status": source_status,
        }
        session_quality = trajectory.get("quality", {})
        if isinstance(session_quality, dict):
            provenance["session_quality_id"] = session_quality.get("session_quality_id")
        lineage = dict(trajectory.get("lineage", {}))
        lineage["episode_id"] = trajectory["example_id"]
        record = {
            "schema_version": ACTION_WINDOW_SCHEMA_VERSION,
            "window_id": window_id,
            "episode_id": trajectory["example_id"],
            "context": {
                "messages": context_messages,
                # Event IDs already commit to each normalized action,
                # observation, or artifact payload.  Hash the bounded context
                # plus the ID prefix instead of serializing all prior event
                # payloads for every action window.
                "state_hash": stable_id(
                    {"messages": context_messages, "prior_event_ids": prior_event_ids}
                ),
                "available_tools_revision": stable_id(tool_registry) if tool_registry else None,
                "available_skills_revision": None,
            },
            "decision": {
                "action": "use",
                "basis": "source-tool-event",
                "skill": {
                    "status": "not_observed",
                    "triggered": None,
                    "read": None,
                    "decision": "not_observed",
                    "skip_reason": None,
                    "revision_sha256": None,
                },
            },
            "tool_call": {
                "name": event.get("name") or "unknown",
                "call_id": call_id,
                "arguments": bounded_arguments,
                "event_id": event.get("event_id"),
            },
            "observation": (
                {
                    "status": observation.get("status", "unknown"),
                    "call_id": observation.get("call_id"),
                    "output": bounded_output,
                    "event_id": observation.get("event_id"),
                }
                if observation is not None
                else None
            ),
            "evidence": evidence,
            "verification": {
                "state_delta_hash": None,
                "artifact_hashes": [],
                "tests": [],
                "diagnostics": [],
                "source": "unscored",
            },
            "quality": {
                "stage": quality_stage,
                "tool_contract": "review",
                "verification": "absent",
                "observation_present": observation is not None,
                "observation_output_policy": output_policy,
                "tool_argument_policy": argument_policy,
                "observation_match": observation_match,
                "evidence_schema_version": ACTION_EVIDENCE_SCHEMA_VERSION,
                "source_episode_status": source_status,
                "session_quality_gate": session_quality.get("session_quality_gate")
                if isinstance(session_quality, dict)
                else None,
                "session_quality_id": session_quality.get("session_quality_id")
                if isinstance(session_quality, dict)
                else None,
                "session_quality_flags": session_quality.get("session_quality_flags", [])
                if isinstance(session_quality, dict)
                else [],
                "model_tier": session_quality.get("model_tier", "unclassified")
                if isinstance(session_quality, dict)
                else "unclassified",
            },
            "provenance": provenance,
            "lineage": lineage,
            "tags": sorted(
                set(trajectory.get("tags", []))
                | {"action-window:tool-use"}
                | ({f"action-window:source-{source_status}"} if source_status != "accepted" else set())
                | ({"action-window:observed-observation"} if observation is not None else {"action-window:missing-observation"})
            ),
            "privacy": trajectory.get("privacy", {}),
        }
        validate_action_window(record)
        windows.append(record)
        output_digest = evidence["observation"]["output_digest"]
        if output_digest is not None:
            observation_digests_by_action.setdefault(action_signature, []).append(
                output_digest
            )
    return windows


def latest_user_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the conversation state ending at the latest user turn."""
    latest_index = max(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        default=-1,
    )
    if latest_index < 0:
        return []
    return [dict(message) for message in messages[: latest_index + 1]]


def preference_candidate(
    value: Any,
    state: NormalizationState,
    *,
    privacy_enabled: bool,
) -> list[dict[str, Any]]:
    if isinstance(value, str):
        content = clean_text(value, state, privacy_enabled=privacy_enabled)
        return [{"role": "assistant", "content": content}] if content else []
    if isinstance(value, dict):
        if isinstance(value.get("messages"), list):
            candidate_record = {"messages": value["messages"]}
            messages, _ = normalize_messages(
                candidate_record, state, privacy_enabled=privacy_enabled
            )
            return messages
        content = text_from_content(value, state, privacy_enabled=privacy_enabled)
        return [{"role": "assistant", "content": content}] if content else []
    return []


def build_preference(
    record: dict[str, Any],
    *,
    common: dict[str, Any],
    state: NormalizationState,
    privacy_enabled: bool,
    privacy_mode: str,
    privacy_approved: bool,
    source_session_key: Any,
) -> dict[str, Any] | None:
    preference = record.get("preference") if isinstance(record.get("preference"), dict) else record
    chosen = preference.get("chosen")
    rejected = preference.get("rejected")
    if chosen is None:
        chosen = preference.get("chosen_response")
    if rejected is None:
        rejected = preference.get("rejected_response")
    if chosen is None or rejected is None:
        return None

    raw_messages = message_list(record) or []
    prompt_value = preference.get("prompt")
    if prompt_value is not None:
        prompt_messages = preference_candidate(prompt_value, state, privacy_enabled=privacy_enabled)
        if prompt_messages and prompt_messages[0].get("role") == "assistant":
            prompt_messages = [{"role": "user", "content": prompt_messages[0]["content"]}]
    else:
        prompt_messages, _ = normalize_messages(
            {"messages": raw_messages}, state, privacy_enabled=privacy_enabled
        )
        prompt_messages = [
            message for message in prompt_messages if message["role"] in {"system", "user"}
        ]
    chosen_messages = preference_candidate(chosen, state, privacy_enabled=privacy_enabled)
    rejected_messages = preference_candidate(rejected, state, privacy_enabled=privacy_enabled)
    if not prompt_messages or not chosen_messages or not rejected_messages:
        return None
    eligible, privacy_reason = privacy_eligibility(privacy_mode, privacy_approved)
    example_id = stable_id(
        {
            "dataset": "preference",
            "session": source_session_key,
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
        }
    )
    return {
        **common,
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": "preference",
        "split": split_for(example_id),
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "tags": sorted(set(common["tags"]) | {"preference:explicit", "quality:review"}),
        "quality": {
            "status": "eligible" if eligible else "review",
            "label_source": "explicit-preference-field",
            "session_quality_gate": common["metadata"].get("quality_gate"),
            "session_quality_id": common["metadata"].get("session_quality_id"),
            "model_tier": common["metadata"].get("model_tier", "unclassified"),
            "model_tier_registry_revision": common["metadata"].get(
                "model_tier_registry_revision", MODEL_TIER_REGISTRY_REVISION
            ),
            "reward_status": "unscored",
        },
        "privacy": {
            "mode": privacy_mode,
            "eligible_for_training": eligible,
            "reason": privacy_reason,
            "structural_redactions": state.redactions,
        },
    }


def normalize_record(
    record: Any,
    *,
    source_file: Path,
    source_file_hash: str,
    source_line: int,
    privacy_mode: str,
    privacy_approved: bool,
    max_record_chars: int,
    quality_override: dict[str, Any] | None = None,
    session_quality_assessment: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[str] | None,
    dict[str, Any] | None,
]:
    source_record_hash = _source_record_hash(record)
    segment_record_hash = _segment_record_hash(record)
    lineage = record_lineage(record, source_record_hash) if isinstance(record, dict) else {
        "parent_record_sha256": source_record_hash,
        "chunk_index": 0,
        "chunk_count": 1,
        "continuation_status": "complete",
        "cut_reason": None,
        "open_tool_call_ids": [],
        "source_message_range": None,
        "previous_example_id": None,
        "next_example_id": None,
    }
    provenance = {
        "parser_revision": PARSER_REVISION,
        "source_file_name": source_file.name,
        "source_file_sha256": source_file_hash,
        "source_line": source_line,
        "source_record_sha256": source_record_hash,
        "segment_record_sha256": segment_record_hash,
        "parent_record_sha256": lineage["parent_record_sha256"],
        "chunk_index": lineage["chunk_index"],
        "chunk_count": lineage["chunk_count"],
        "continuation_status": lineage["continuation_status"],
        "open_tool_call_ids": lineage["open_tool_call_ids"],
    }
    if lineage["source_message_range"] is not None:
        provenance["source_message_range"] = lineage["source_message_range"]
    if not isinstance(record, dict):
        return None, None, None, ["record_not_object"], None

    privacy_enabled = privacy_mode != "none"
    state = NormalizationState()
    source_origin = record.get("source_origin")
    if isinstance(source_origin, dict):
        provenance["source_origin"] = clean_value(
            source_origin, state, privacy_enabled=privacy_enabled
        )
    messages, raw_messages = normalize_messages(
        record, state, privacy_enabled=privacy_enabled
    )
    provider, raw_source = provider_for(record)
    events = normalize_tool_events(
        record, raw_messages, state, privacy_enabled=privacy_enabled
    )
    tools = normalized_tool_schemas(record, state, privacy_enabled=privacy_enabled)
    for ordinal, event in enumerate(events):
        event_payload = dict(event)
        event["schema_version"] = EVENT_SCHEMA_VERSION
        event["event_id"] = stable_id(
            {
                "segment_record_sha256": segment_record_hash,
                "ordinal": ordinal,
                "event": event_payload,
            }
        )
        event["ordinal"] = ordinal
        event["parent_record_sha256"] = source_record_hash
    has_user = any(message["role"] == "user" and message["content"] for message in messages)
    has_assistant = any(
        message["role"] == "assistant"
        and (message["content"] or message.get("tool_calls"))
        for message in messages
    )
    if not message_list(record):
        common_reasons = ["messages_missing_or_not_list"]
    else:
        common_reasons = []
    if not has_user:
        common_reasons.append("user_turn_missing")
    if not has_assistant:
        common_reasons.append("assistant_turn_missing")

    joined_user_text = "\n".join(
        message["content"] for message in messages if message["role"] == "user"
    )
    family = task_family(joined_user_text)
    outcome, outcome_source = explicit_outcome(record)
    reward, reward_status = explicit_reward(record)
    training_lane = training_lane_for(record)
    has_tools = any(event["kind"] in {"action", "observation"} for event in events)
    has_diffs = any(event["kind"] == "artifact" for event in events)
    has_context = any(event["kind"] == "context" for event in events)
    tool_family_tags = tool_families(events, tools)
    content_provider_mentions = provider_mentions(messages)
    observation_truncation_count = len(
        record.get("observation_truncations", [])
        if isinstance(record.get("observation_truncations"), list)
        else []
    )
    has_code = has_diffs or has_context or "```" in "\n".join(
        message["content"] for message in messages
    )
    quality_assessment = quality_assessment_for(
        record,
        messages=messages,
        events=events,
        quality_override=quality_override,
        session_quality_assessment=session_quality_assessment,
    )
    visible_marker_count = trainer_marker_count(messages) + trainer_marker_count(events)
    if visible_marker_count:
        quality_assessment = dict(quality_assessment)
        quality_flags = set(quality_assessment.get("flags", []))
        quality_flags.add("visible_trainer_marker_review")
        quality_assessment["flags"] = sorted(quality_flags)
        if quality_assessment.get("gate") == "candidate":
            quality_assessment["gate"] = "review_required"
    # The session-quality index is structural, but it still carries source
    # model/provider values for auditability.  Do not place that raw assessment
    # directly inside trainer-facing rows: it is part of the no-reasoning and
    # privacy boundary just like messages, events, and tool schemas.
    cleaned_quality_assessment = clean_value(
        quality_assessment, state, privacy_enabled=privacy_enabled
    )
    quality_gate = quality_assessment["gate"]
    quality_flags = quality_assessment["flags"]
    eligible, privacy_reason = privacy_eligibility(privacy_mode, privacy_approved)
    common = {
        "tags": base_tags(
            provider=provider,
            mode=first_value(record, ("mode", "agent")),
            family=family,
            has_tools=has_tools,
            has_diffs=has_diffs,
            has_context=has_context,
            has_code=has_code,
            tool_family_tags=tool_family_tags,
            content_provider_mentions=content_provider_mentions,
            privacy_mode=privacy_mode,
            privacy_approved=privacy_approved,
            outcome=outcome,
            training_lane=training_lane,
            quality_gate=quality_gate,
            model_tier=quality_assessment.get("model_tier", "unclassified"),
            quality_flags=quality_flags,
        ),
        "metadata": metadata_for(
            record,
            provider=provider,
            raw_source=raw_source,
            source_file=source_file,
            source_file_hash=source_file_hash,
            source_line=source_line,
            source_record_hash=source_record_hash,
            segment_record_hash=segment_record_hash,
            state=state,
            privacy_enabled=privacy_enabled,
            quality_assessment=quality_assessment,
        ),
        "lineage": lineage,
    }
    if lineage["chunk_count"] > 1:
        common["tags"] = sorted(
            set(common["tags"]) | {"trace:chunked", "lineage:episode"}
        )
    if observation_truncation_count:
        common["tags"] = sorted(
            set(common["tags"]) | {"quality:observation-truncated"}
        )
    preference = build_preference(
        record,
        common=common,
        state=state,
        privacy_enabled=privacy_enabled,
        privacy_mode=privacy_mode,
        privacy_approved=privacy_approved,
        source_session_key=common["metadata"]["session_id"],
    )

    identity = {
        "provider": provider,
        "session_id": common["metadata"]["session_id"],
        "messages": messages,
        "events": events,
        "tools": tools,
    }
    example_id = stable_id(identity)
    serialized_size = canonical_json_size({"messages": messages, "events": events})
    if max_record_chars > 0 and serialized_size > max_record_chars:
        common_reasons.append("record_exceeds_max_chars")

    if common_reasons:
        if preference is not None and set(common_reasons).issubset(
            {"messages_missing_or_not_list", "user_turn_missing", "assistant_turn_missing"}
        ):
            return None, None, preference, None, None
        rejection = {
            "schema_version": SCHEMA_VERSION,
            "record_id": stable_id({"source": provenance, "record": source_record_hash}),
            "status": "rejected",
            "reasons": sorted(set(common_reasons)),
            "provenance": provenance,
        }
        oversized_trajectory = None
        if common_reasons == ["record_exceeds_max_chars"]:
            # Reuse the normalized payload already in memory.  The caller can
            # project action windows from this rejected episode without
            # recursively scrubbing every large observation a second time.
            oversized_trajectory = {
                **common,
                "schema_version": SCHEMA_VERSION,
                "example_id": example_id,
                "dataset": "trajectory",
                "split": split_for(example_id),
                "messages": messages,
                "events": events,
                "trajectory": {
                    "outcome": outcome,
                    "outcome_source": outcome_source,
                    "reward": reward,
                    "reward_status": reward_status,
                    "terminal": outcome != "unknown" and outcome_source == "explicit",
                    "ordering_note": "Message order is source order; detached tool events retain null message_index.",
                    "lineage": lineage,
                },
                "quality": {
                    "status": "rejected_oversize",
                    "event_count": len(events),
                    "payload_chars": serialized_size,
                    "token_count": None,
                    "token_count_status": "not_tokenized",
                    "model_tier": quality_assessment.get("model_tier", "unclassified"),
                    "model_tier_basis": quality_assessment.get(
                        "model_tier_basis", "missing_authoritative_tier"
                    ),
                    "model_tier_registry_revision": quality_assessment.get(
                        "model_tier_registry_revision", MODEL_TIER_REGISTRY_REVISION
                    ),
                },
                "privacy": {
                    "mode": privacy_mode,
                    "eligible_for_training": eligible,
                    "reason": privacy_reason,
                    "structural_redactions": state.redactions,
                },
            }
            if tools:
                oversized_trajectory["tools"] = tools
        return (
            None,
            oversized_trajectory,
            preference,
            RejectionReasons(
                rejection["reasons"],
                quality_gate=quality_gate,
                quality_flags=quality_flags,
            ),
            None,
        )

    quality_status = "eligible" if eligible else "review"
    quality = {
        "status": quality_status,
        "session_quality_gate": quality_gate,
        "session_quality_flags": quality_flags,
        "session_quality_id": quality_assessment.get("session_quality_id"),
        "session_quality_scope": quality_assessment.get("scope", "source_session"),
        "session_quality_assessment": cleaned_quality_assessment,
        "model_tier": quality_assessment.get("model_tier", "unclassified"),
        "model_tier_basis": quality_assessment.get(
            "model_tier_basis", "missing_authoritative_tier"
        ),
        "model_tier_confidence": quality_assessment.get(
            "model_tier_confidence", "unknown"
        ),
        "model_tier_registry_revision": quality_assessment.get(
            "model_tier_registry_revision", MODEL_TIER_REGISTRY_REVISION
        ),
        "has_user": has_user,
        "has_assistant": has_assistant,
        "assistant_turns": sum(1 for message in messages if message["role"] == "assistant"),
        "message_count": len(messages),
        "reasoning_removed": bool(state.reasoning_fields_removed or state.reasoning_blocks_removed),
        "reasoning_fields_removed": state.reasoning_fields_removed,
        "reasoning_blocks_removed": state.reasoning_blocks_removed,
        "dropped_roles": sorted(set(state.dropped_roles)),
        "multimodal_parts_dropped": state.multimodal_parts,
        "tool_arguments_unparsed": state.tool_arguments_unparsed,
        "has_tools": has_tools,
        "has_diffs": has_diffs,
        "has_code": has_code,
        "tool_families": tool_family_tags,
        "tool_family_tagging": "heuristic-name-and-sanitized-action-payload",
        "provider_mentions": content_provider_mentions,
        "outcome_source": outcome_source,
        "reward_status": reward_status,
        "event_count": len(events),
        "payload_chars": serialized_size,
        "token_count": None,
        "token_count_status": "not_tokenized",
        "chunked": lineage["chunk_count"] > 1,
        "chunk_index": lineage["chunk_index"],
        "chunk_count": lineage["chunk_count"],
        "continuation_status": lineage["continuation_status"],
        "cut_reason": lineage["cut_reason"],
        "source_message_range": lineage["source_message_range"],
        "observation_truncation_count": observation_truncation_count,
        "observation_payload_status": (
            "truncated" if observation_truncation_count else "complete_or_not_observed"
        ),
    }
    privacy = {
        "mode": privacy_mode,
        "eligible_for_training": eligible,
        "reason": privacy_reason,
        "structural_redactions": state.redactions,
    }
    sft_messages = []
    for message in messages:
        sft_message = {"role": message["role"], "content": message["content"]}
        if message.get("tool_calls"):
            sft_message["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            sft_message["tool_call_id"] = message["tool_call_id"]
        sft_messages.append(sft_message)
    sft = {
        **common,
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": "sft",
        "split": split_for(example_id),
        "messages": sft_messages,
        "events": events,
        "quality": quality,
        "privacy": privacy,
    }
    if tools:
        sft["tools"] = tools
    trajectory = {
        **common,
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": "trajectory",
        "split": split_for(example_id),
        "messages": messages,
        "events": events,
        "trajectory": {
            "outcome": outcome,
            "outcome_source": outcome_source,
            "reward": reward,
            "reward_status": reward_status,
            "terminal": outcome != "unknown" and outcome_source == "explicit",
            "ordering_note": "Message order is source order; detached tool events retain null message_index.",
        },
        "quality": quality,
        "privacy": privacy,
    }
    trajectory["trajectory"]["lineage"] = lineage
    if tools:
        trajectory["tools"] = tools
    tool_trace = None
    if events or tools:
        tool_trace_id = stable_id(
            {
                "dataset": "tool_trace",
                "source_example_id": example_id,
                "messages": messages,
                "events": events,
                "tools": tools,
            }
        )
        tool_trace = {
            **common,
            "schema_version": SCHEMA_VERSION,
            "example_id": tool_trace_id,
            "dataset": "tool_trace",
            "split": split_for(tool_trace_id),
            "source_example_id": example_id,
            "messages": messages,
            "events": events,
            "quality": {
                **quality,
                "trace_ready": bool(events),
                "trace_contract": "action-observation-artifact-context",
            },
            "privacy": privacy,
        }
        if tools:
            tool_trace["tools"] = tools
    validate_dataset_record(sft, "sft")
    validate_dataset_record(trajectory, "trajectory")
    if tool_trace is not None:
        validate_dataset_record(tool_trace, "tool_trace")
    return sft, trajectory, preference, None, tool_trace


def discover_jsonl(paths: Iterable[Path], *, excluded_dir: Path | None = None) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    excluded = excluded_dir.resolve() if excluded_dir else None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("*.jsonl"))
        else:
            continue
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if excluded and (resolved == excluded or excluded in resolved.parents):
                continue
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(candidate)
    if not discovered:
        raise ValueError("No JSONL inputs found")
    return discovered


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                yield line_number, json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest(path: Path, manifest: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_privacy_manifest(
    manifest_path: Path, files: Iterable[Path]
) -> dict[str, str]:
    """Verify that every builder input is an exact output of Privacy Filter."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid privacy manifest {manifest_path}: {exc.msg}") from exc
    if manifest.get("schema_version") != "privacy-filter/v1":
        raise ValueError(f"Unsupported privacy manifest schema in {manifest_path}")
    if manifest.get("model") != "openai/privacy-filter":
        raise ValueError(f"Unexpected privacy filter model in {manifest_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"Privacy manifest outputs must be a list: {manifest_path}")

    output_hashes: dict[str, str] = {}
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError(f"Invalid output entry in privacy manifest {manifest_path}")
        name = output.get("name")
        digest = output.get("sha256")
        relative = Path(name) if isinstance(name, str) else Path()
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"Invalid output identity in privacy manifest {manifest_path}")
        output_hashes[relative.as_posix()] = digest

    root = manifest_path.parent.resolve()
    verified: dict[str, str] = {}
    for file in files:
        try:
            relative = file.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Privacy-manifest input is outside its output directory: {file}"
            ) from exc
        expected = output_hashes.get(relative)
        if expected is None:
            raise ValueError(f"Input is absent from privacy manifest: {relative}")
        actual = sha256_file(file)
        if actual != expected:
            raise ValueError(f"Privacy-manifest hash mismatch for {relative}")
        verified[relative] = actual
    return verified


def link_chunk_lineage(datasets: dict[str, list[dict[str, Any]]]) -> None:
    """Add deterministic adjacent IDs after segment identities are known."""
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for record in datasets.get("sft", []):
        lineage = record.get("lineage")
        if not isinstance(lineage, dict) or lineage.get("chunk_count", 1) <= 1:
            continue
        parent = lineage.get("parent_record_sha256")
        index = lineage.get("chunk_index")
        if isinstance(parent, str) and isinstance(index, int):
            grouped.setdefault(parent, {})[index] = record

    links: dict[str, dict[str, str | None]] = {}
    for records in grouped.values():
        for index, record in records.items():
            previous = records.get(index - 1)
            following = records.get(index + 1)
            links[record["example_id"]] = {
                "previous_example_id": previous["example_id"] if previous else None,
                "next_example_id": following["example_id"] if following else None,
            }

    for dataset in ("sft", "trajectory"):
        for record in datasets.get(dataset, []):
            link = links.get(record.get("example_id"))
            if link is None:
                continue
            lineage = record.get("lineage")
            if isinstance(lineage, dict):
                lineage.update(link)
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                metadata.update(link)
            trajectory = record.get("trajectory")
            if isinstance(trajectory, dict) and isinstance(trajectory.get("lineage"), dict):
                trajectory["lineage"].update(link)

    for record in datasets.get("tool_trace", []):
        link = links.get(record.get("source_example_id"))
        if link is None:
            continue
        lineage = record.get("lineage")
        if isinstance(lineage, dict):
            lineage.update(link)
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metadata.update(link)

    for record in datasets.get("action_window", []):
        link = links.get(record.get("episode_id"))
        if link is None:
            continue
        lineage = record.get("lineage")
        if isinstance(lineage, dict):
            lineage.update(link)

    for record in datasets.get("rl_prompt", []):
        link = links.get(record.get("source_example_id"))
        if link is None:
            continue
        lineage = record.get("lineage")
        if isinstance(lineage, dict):
            lineage.update(link)
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metadata.update(link)


def build_datasets(
    inputs: Iterable[Path],
    *,
    output_dir: Path,
    privacy_mode: str,
    privacy_approved: bool,
    max_record_chars: int,
    overwrite: bool,
    oversize_strategy: str = "chunk",
    max_records_per_file: int | None = None,
    privacy_manifest: Path | None = None,
    training_lanes: Iterable[str] = ("primary",),
    quality_gates: Iterable[str] = ("candidate",),
    quality_overrides: dict[str, dict[str, Any]] | None = None,
    model_tiers: Iterable[str] | None = None,
    model_tier_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if privacy_mode not in PRIVACY_MODES:
        raise ValueError(f"--privacy-mode must be one of: {', '.join(sorted(PRIVACY_MODES))}")
    if max_record_chars < 0:
        raise ValueError("--max-record-chars cannot be negative")
    if oversize_strategy not in CHUNK_STRATEGIES:
        raise ValueError(
            "--oversize-strategy must be one of: "
            + ", ".join(sorted(CHUNK_STRATEGIES))
        )
    if max_records_per_file is not None and max_records_per_file <= 0:
        raise ValueError("--max-records-per-file must be positive when provided")
    selected_lanes = {
        normalize_token(lane) for lane in training_lanes if normalize_token(lane)
    }
    unknown_lanes = selected_lanes - TRAINING_LANES
    if unknown_lanes:
        raise ValueError(
            "--training-lanes contains unsupported values: "
            + ", ".join(sorted(unknown_lanes))
        )
    if not selected_lanes:
        raise ValueError("--training-lanes must select at least one lane")
    selected_quality_gates = {
        normalize_token(gate) for gate in quality_gates if normalize_token(gate)
    }
    unknown_quality_gates = selected_quality_gates - QUALITY_GATES
    if unknown_quality_gates:
        raise ValueError(
            "--quality-gates contains unsupported values: "
            + ", ".join(sorted(unknown_quality_gates))
        )
    if not selected_quality_gates:
        raise ValueError("--quality-gates must select at least one gate")
    quality_overrides = quality_overrides or {}
    model_tier_overrides = model_tier_overrides or {}
    raw_model_tiers = list(model_tiers or MODEL_TIERS)
    normalized_model_tiers = [normalize_model_tier(tier) for tier in raw_model_tiers]
    unknown_model_tiers = {
        str(tier)
        for tier, normalized in zip(raw_model_tiers, normalized_model_tiers)
        if normalized is None
    }
    selected_model_tiers = {
        tier for tier in normalized_model_tiers if tier is not None
    }
    if unknown_model_tiers:
        raise ValueError(
            "--model-tiers contains unsupported values: "
            + ", ".join(sorted(unknown_model_tiers))
        )
    if not selected_model_tiers:
        raise ValueError("--model-tiers must select at least one tier")
    files = discover_jsonl(inputs, excluded_dir=output_dir)
    quality_index, quality_index_summary = build_session_quality_index(
        files,
        quality_overrides=quality_overrides,
        model_tier_overrides=model_tier_overrides,
        max_records_per_file=max_records_per_file,
    )
    privacy_manifest_info: dict[str, str] | None = None
    if privacy_manifest is not None:
        if privacy_mode != "filtered":
            raise ValueError("--privacy-manifest requires --privacy-mode filtered")
        verify_privacy_manifest(privacy_manifest, files)
        privacy_manifest_info = {
            "file_name": privacy_manifest.name,
            "sha256": sha256_file(privacy_manifest),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in OUTPUT_FILES] + [output_dir / "manifest.json"]
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing training outputs: "
                + ", ".join(str(path) for path in existing)
            )

    datasets: dict[str, list[dict[str, Any]]] = {
        "sft": [],
        "trajectory": [],
        "tool_trace": [],
        "action_window": [],
        "preference": [],
        "rl_prompt": [],
        "rejected": [],
    }
    stats = BuildStats()
    stats.quality_session_count = quality_index_summary["sessions"]
    stats.quality_session_gate_counts.update(quality_index_summary["gate_counts"])
    stats.quality_session_lane_counts.update(quality_index_summary["lane_counts"])
    stats.model_tier_counts.update(quality_index_summary["model_tier_counts"])
    stats.quality_session_conflict_count = sum(
        "conflicting_session_quality_assessments" in assessment.get("flags", [])
        or "conflicting_quality_overrides" in assessment.get("flags", [])
        for assessment in quality_index.values()
    )
    seen_ids: set[str] = set()
    seen_tool_trace_ids: set[str] = set()
    seen_action_window_ids: set[str] = set()
    seen_preference_ids: set[str] = set()
    input_manifest: list[dict[str, Any]] = []

    for source_file in files:
        source_hash = sha256_file(source_file)
        indexed_hash = quality_index_summary["file_hashes"].get(str(source_file.resolve()))
        if indexed_hash != source_hash:
            raise RuntimeError(
                f"source changed between quality preflight and build: {source_file.name}"
            )
        input_manifest.append(
            {"name": source_file.name, "sha256": source_hash, "bytes": source_file.stat().st_size}
        )
        file_record_count = 0
        for source_line, parent_record in iter_jsonl(source_file):
            if (
                max_records_per_file is not None
                and file_record_count >= max_records_per_file
            ):
                break
            file_record_count += 1
            stats.input_records += 1
            training_lane = training_lane_for(parent_record)
            stats.training_lane_counts[training_lane] += 1
            if training_lane not in selected_lanes:
                stats.skipped_lane_records += 1
                continue
            parent_quality_assessment = quality_index.get(
                quality_session_key(parent_record, source_hash)
            )
            parent_model_tier = (
                parent_quality_assessment.get("model_tier", "unclassified")
                if isinstance(parent_quality_assessment, dict)
                else "unclassified"
            )
            if parent_model_tier not in selected_model_tiers:
                stats.skipped_model_tier_records += 1
                parent_record_hash = _source_record_hash(parent_record)
                datasets["rejected"].append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_id": stable_id(
                            {
                                "source_file": source_file.name,
                                "line": source_line,
                                "record": parent_record_hash,
                                "reason": "model_tier_not_selected",
                            }
                        ),
                        "status": "skipped_model_tier",
                        "reasons": ["model_tier_not_selected"],
                        "model_tier": parent_model_tier,
                        "session_quality_id": (
                            parent_quality_assessment.get("session_quality_id")
                            if isinstance(parent_quality_assessment, dict)
                            else None
                        ),
                        "provenance": {
                            "parser_revision": PARSER_REVISION,
                            "source_file_name": source_file.name,
                            "source_file_sha256": source_hash,
                            "source_line": source_line,
                            "source_record_sha256": parent_record_hash,
                            "session_quality_id": (
                                parent_quality_assessment.get("session_quality_id")
                                if isinstance(parent_quality_assessment, dict)
                                else None
                            ),
                        },
                    }
                )
                stats.rejected_records += 1
                stats.rejection_reasons["model_tier_not_selected"] += 1
                continue
            candidates = (
                chunk_record_variants(parent_record, max_record_chars)
                if oversize_strategy == "chunk"
                else [parent_record]
            )
            if len(candidates) > 1:
                stats.chunked_parent_records += 1
                stats.chunk_records += len(candidates)
            for record in candidates:
                session_quality_assessment = quality_index.get(
                    quality_session_key(record, source_hash)
                )
                quality_override = None if session_quality_assessment is not None else quality_override_for(
                    record,
                    source_file_hash=source_hash,
                    quality_overrides=quality_overrides,
                )
                sft, trajectory, preference, rejection, tool_trace = normalize_record(
                    record,
                    source_file=source_file,
                    source_file_hash=source_hash,
                    source_line=source_line,
                    privacy_mode=privacy_mode,
                    privacy_approved=privacy_approved,
                    max_record_chars=max_record_chars,
                    quality_override=quality_override,
                    session_quality_assessment=session_quality_assessment,
                )
                if sft is not None:
                    quality_gate = sft["quality"]["session_quality_gate"]
                    quality_flags = sft["quality"]["session_quality_flags"]
                elif rejection is not None:
                    quality_gate = getattr(rejection, "quality_gate", "unassessed")
                    quality_flags = getattr(rejection, "quality_flags", [])
                else:
                    quality_gate = "unassessed"
                    quality_flags = []
                stats.quality_gate_counts[quality_gate] += 1
                if quality_gate not in selected_quality_gates:
                    stats.skipped_quality_gate_records += 1
                    session_quality_id = (
                        session_quality_assessment.get("session_quality_id")
                        if isinstance(session_quality_assessment, dict)
                        else None
                    )
                    datasets["rejected"].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_id": stable_id(
                                {
                                    "source_file": source_file.name,
                                    "line": source_line,
                                    "record": _source_record_hash(record),
                                    "segment": _segment_record_hash(record),
                                    "reason": "quality_gate_not_selected",
                                }
                            ),
                            "status": "skipped_quality_gate",
                            "reasons": ["quality_gate_not_selected"],
                            "quality_gate": quality_gate,
                            "quality_flags": quality_flags,
                            "session_quality_id": session_quality_id,
                            "provenance": {
                                "parser_revision": PARSER_REVISION,
                                "source_file_name": source_file.name,
                                "source_file_sha256": source_hash,
                                "source_line": source_line,
                                "source_record_sha256": _source_record_hash(record),
                                "segment_record_sha256": _segment_record_hash(record),
                                "session_quality_id": session_quality_id,
                            },
                        }
                    )
                    continue
                if preference is not None and preference["example_id"] not in seen_preference_ids:
                    datasets["preference"].append(preference)
                    seen_preference_ids.add(preference["example_id"])
                    stats.preference_records += 1
                if rejection is not None:
                    source_hash_for_rejection = _source_record_hash(record)
                    segment_hash_for_rejection = _segment_record_hash(record)
                    rejection_lineage = (
                        record_lineage(record, source_hash_for_rejection)
                        if isinstance(record, dict)
                        else None
                    )
                    rejection_provenance = {
                        "parser_revision": PARSER_REVISION,
                        "source_file_name": source_file.name,
                        "source_file_sha256": source_hash,
                        "source_line": source_line,
                        "source_record_sha256": source_hash_for_rejection,
                        "segment_record_sha256": segment_hash_for_rejection,
                        "session_quality_id": (
                            session_quality_assessment.get("session_quality_id")
                            if isinstance(session_quality_assessment, dict)
                            else None
                        ),
                    }
                    if rejection_lineage is not None:
                        rejection_provenance.update(
                            {
                                "parent_record_sha256": rejection_lineage["parent_record_sha256"],
                                "chunk_index": rejection_lineage["chunk_index"],
                                "chunk_count": rejection_lineage["chunk_count"],
                                "continuation_status": rejection_lineage["continuation_status"],
                            }
                        )
                        if rejection_lineage["source_message_range"] is not None:
                            rejection_provenance["source_message_range"] = rejection_lineage[
                                "source_message_range"
                            ]
                    if "record_exceeds_max_chars" in rejection:
                        # Keep bounded tool transitions even when the enclosing
                        # episode is rejected for size.  This is explicitly a
                        # candidate projection from the already-normalized
                        # rejected segment, never an accepted SFT/trajectory
                        # row.  Do not normalize a multi-GB observation twice.
                        if trajectory is not None:
                            for action_window in action_windows_for(
                                trajectory,
                                source_status="rejected_oversize",
                            ):
                                if action_window["window_id"] in seen_action_window_ids:
                                    continue
                                datasets["action_window"].append(action_window)
                                seen_action_window_ids.add(action_window["window_id"])
                                stats.action_window_records += 1
                                stats.recovered_action_window_records += 1
                    datasets["rejected"].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_id": stable_id(
                                {
                                    "source_file": source_file.name,
                                    "line": source_line,
                                    "record": source_hash_for_rejection,
                                    "segment": segment_hash_for_rejection,
                                    "chunk_index": (
                                        rejection_lineage["chunk_index"]
                                        if rejection_lineage is not None
                                        else 0
                                    ),
                                }
                            ),
                            "status": "rejected",
                            "reasons": rejection,
                            "provenance": rejection_provenance,
                        }
                    )
                    stats.rejected_records += 1
                    stats.rejection_reasons.update(rejection)
                    continue
                if sft is None or trajectory is None:
                    continue
                if sft["example_id"] in seen_ids:
                    stats.duplicate_records += 1
                    datasets["rejected"].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_id": sft["example_id"],
                            "status": "rejected",
                            "reasons": ["duplicate_example_id"],
                            "provenance": sft["metadata"],
                        }
                    )
                    stats.rejected_records += 1
                    stats.rejection_reasons["duplicate_example_id"] += 1
                    continue
                seen_ids.add(sft["example_id"])
                datasets["sft"].append(sft)
                datasets["trajectory"].append(trajectory)
                for action_window in action_windows_for(trajectory):
                    if action_window["window_id"] in seen_action_window_ids:
                        continue
                    datasets["action_window"].append(action_window)
                    seen_action_window_ids.add(action_window["window_id"])
                    stats.action_window_records += 1
                if tool_trace is not None and tool_trace["example_id"] not in seen_tool_trace_ids:
                    datasets["tool_trace"].append(tool_trace)
                    seen_tool_trace_ids.add(tool_trace["example_id"])
                    stats.tool_trace_records += 1
                prompt_messages = latest_user_prompt(trajectory["messages"])
                if prompt_messages:
                    prompt_group_id = stable_id({"prompt": prompt_messages})
                    prompt_provider_mentions = provider_mentions(prompt_messages)
                    rl_tags = set(sft["tags"]) | {"rl:prompt-only", "reward:unscored"}
                    if prompt_provider_mentions:
                        rl_tags.add("prompt:provider-specific")
                    rl_prompt_id = stable_id(
                        {
                            "dataset": "rl_prompt",
                            "source_example_id": trajectory["example_id"],
                            "prompt": prompt_messages,
                        }
                    )
                    rl_prompt = {
                        "schema_version": SCHEMA_VERSION,
                        "example_id": rl_prompt_id,
                        "dataset": "rl_prompt",
                        "split": split_for(rl_prompt_id),
                        "prompt": prompt_messages,
                        "prompt_group_id": prompt_group_id,
                        "source_example_id": trajectory["example_id"],
                        "reward": None,
                        "reward_status": "unscored",
                        "tags": sorted(rl_tags),
                        "metadata": {
                            **sft["metadata"],
                            "source_example_id": trajectory["example_id"],
                        },
                        "quality": {
                            "status": sft["quality"]["status"],
                            "session_quality_gate": sft["quality"]["session_quality_gate"],
                            "session_quality_id": sft["quality"].get("session_quality_id"),
                            "session_quality_flags": sft["quality"]["session_quality_flags"],
                            "model_tier": sft["quality"].get("model_tier", "unclassified"),
                            "rl_ready": False,
                            "reason": "A task environment and reward function are required.",
                            "provider_mentions": prompt_provider_mentions,
                            "prompt_group_size": 1,
                        },
                        "privacy": sft["privacy"],
                    }
                    validate_dataset_record(rl_prompt, "rl_prompt")
                    datasets["rl_prompt"].append(rl_prompt)
                    stats.rl_prompt_records += 1
                stats.normalized_records += 1
                stats.sft_records += 1
                stats.trajectory_records += 1
                provider = sft["metadata"]["provider"]
                family = next(tag.removeprefix("task:") for tag in sft["tags"] if tag.startswith("task:"))
                stats.source_counts[provider] += 1
                stats.task_counts[family] += 1
                stats.outcome_counts[next(tag.removeprefix("outcome:") for tag in sft["tags"] if tag.startswith("outcome:"))] += 1
                stats.reward_status_counts[trajectory["trajectory"]["reward_status"]] += 1
                stats.tool_family_counts.update(sft["quality"]["tool_families"])
                stats.provider_mention_counts.update(sft["quality"]["provider_mentions"])

    link_chunk_lineage(datasets)
    prompt_groups = Counter(record["prompt_group_id"] for record in datasets["rl_prompt"])
    stats.rl_prompt_duplicate_groups = sum(1 for size in prompt_groups.values() if size > 1)
    stats.rl_prompt_duplicate_rows = sum(
        size - 1 for size in prompt_groups.values() if size > 1
    )
    for record in datasets["rl_prompt"]:
        group_size = prompt_groups[record["prompt_group_id"]]
        record["quality"]["prompt_group_size"] = group_size
        if group_size > 1:
            record["tags"] = sorted(set(record["tags"]) | {"rl:duplicate-prompt"})

    for dataset in ("sft", "trajectory", "tool_trace", "preference", "rl_prompt"):
        for record in datasets[dataset]:
            validate_dataset_record(record, dataset)
    for record in datasets["action_window"]:
        validate_action_window(record)

    output_records = {
        "sft.jsonl": datasets["sft"],
        "trajectories.jsonl": datasets["trajectory"],
        "tool_traces.jsonl": datasets["tool_trace"],
        "action_windows.jsonl": datasets["action_window"],
        "preferences.jsonl": datasets["preference"],
        "rl_prompts.jsonl": datasets["rl_prompt"],
        "rejected.jsonl": datasets["rejected"],
    }
    for name, records in output_records.items():
        write_jsonl(output_dir / name, records, overwrite=True)

    quality_manifest_summary = {
        key: value
        for key, value in quality_index_summary.items()
        if key != "file_hashes"
    }
    quality_manifest_summary["files"] = [
        {"name": Path(path).name, "sha256": digest}
        for path, digest in sorted(quality_index_summary["file_hashes"].items())
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "generated_at": utc_now(),
        "inputs": input_manifest,
        "policy": {
            "parser_revision": PARSER_REVISION,
            "privacy_mode": privacy_mode,
            "privacy_approved": privacy_approved,
            "privacy_manifest": privacy_manifest_info,
            "oversize_strategy": oversize_strategy,
            "max_record_chars": max_record_chars,
            "chunk_target_fraction": CHUNK_TARGET_FRACTION,
            "max_records_per_file": max_records_per_file,
            "training_lanes": sorted(selected_lanes),
            "quality_gates": sorted(selected_quality_gates),
            "model_tiers": sorted(selected_model_tiers),
            "model_tier_registry_revision": MODEL_TIER_REGISTRY_REVISION,
            "quality_overrides": {
                "provided": bool(quality_overrides),
                "keys": len(quality_overrides),
            },
            "model_tier_overrides": {
                "provided": bool(model_tier_overrides),
                "keys": len(model_tier_overrides),
            },
            "quality_index": quality_manifest_summary,
            "oversize_policy": (
                "split at bounded conversation boundaries; reject only an "
                "indivisible segment"
                if oversize_strategy == "chunk"
                else "reject normalized records over the safety bound"
            ),
            "training_eligibility": "eligible only when privacy_approved is true",
            "reasoning": "reasoning fields and tagged blocks are removed; no chain-of-thought is exported",
            "reward": "only numeric source rewards are preserved; otherwise reward is null and unscored",
            "preference": "only explicit chosen/rejected source pairs are exported",
        },
        "counts": {
            "input_records": stats.input_records,
            "normalized_records": stats.normalized_records,
            "sft": stats.sft_records,
            "trajectories": stats.trajectory_records,
            "tool_traces": stats.tool_trace_records,
            "action_windows": stats.action_window_records,
            "recovered_action_windows": stats.recovered_action_window_records,
            "preferences": stats.preference_records,
            "rl_prompts": stats.rl_prompt_records,
            "rejected": stats.rejected_records,
            "duplicates": stats.duplicate_records,
            "chunked_parent_records": stats.chunked_parent_records,
            "chunk_records": stats.chunk_records,
            "skipped_training_lane_records": stats.skipped_lane_records,
            "skipped_quality_gate_records": stats.skipped_quality_gate_records,
            "skipped_model_tier_records": stats.skipped_model_tier_records,
            "quality_sessions": stats.quality_session_count,
            "quality_session_conflicts": stats.quality_session_conflict_count,
        },
        "sources": dict(sorted(stats.source_counts.items())),
        "training_lanes": dict(sorted(stats.training_lane_counts.items())),
        "quality_gates": dict(sorted(stats.quality_gate_counts.items())),
        "quality_sessions": dict(sorted(stats.quality_session_gate_counts.items())),
        "quality_session_lanes": dict(sorted(stats.quality_session_lane_counts.items())),
        "model_tiers": dict(sorted(stats.model_tier_counts.items())),
        "task_families": dict(sorted(stats.task_counts.items())),
        "outcomes": dict(sorted(stats.outcome_counts.items())),
        "reward_status": dict(sorted(stats.reward_status_counts.items())),
        "tool_families": dict(sorted(stats.tool_family_counts.items())),
        "provider_mentions": dict(sorted(stats.provider_mention_counts.items())),
        "rl_prompt_duplicates": {
            "groups": stats.rl_prompt_duplicate_groups,
            "extra_rows": stats.rl_prompt_duplicate_rows,
            "policy": "retain for provenance; cap or sample by prompt_group_id during RL",
        },
        "rejection_reasons": dict(sorted(stats.rejection_reasons.items())),
        "outputs": {},
    }
    for name in (*OUTPUT_FILES,):
        target = output_dir / name
        manifest["outputs"][name] = {
            "sha256": sha256_file(target),
            "records": len(output_records[name]),
        }
    write_manifest(output_dir / "manifest.json", manifest, overwrite=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize extracted AI sessions into reasoning-free training datasets"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL files or directories")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training_data"), help="Output directory"
    )
    parser.add_argument(
        "--privacy-mode",
        choices=sorted(PRIVACY_MODES),
        default="heuristic",
        help="Input privacy state: heuristic structural scrub, external filtered corpus, or none",
    )
    parser.add_argument(
        "--privacy-approved",
        action="store_true",
        help="Record human approval after privacy review; required for training eligibility",
    )
    parser.add_argument(
        "--max-record-chars",
        type=int,
        default=250_000,
        help="Final safety bound per normalized segment; 0 disables the bound",
    )
    parser.add_argument(
        "--oversize-strategy",
        choices=sorted(CHUNK_STRATEGIES),
        default="chunk",
        help="Split oversized sessions at safe boundaries (default) or reject them",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        help="Bounded smoke mode: process at most this many source records per input file",
    )
    parser.add_argument(
        "--privacy-manifest",
        type=Path,
        help=(
            "Verify filtered inputs against filter_privacy.py's privacy_manifest.json; "
            "requires --privacy-mode filtered"
        ),
    )
    parser.add_argument(
        "--training-lanes",
        default="primary",
        help=(
            "Comma-separated source lanes to include (primary, optional_alt, "
            "quarantine); default keeps optional/advisor data out of the main build"
        ),
    )
    parser.add_argument(
        "--quality-gates",
        default="candidate",
        help=(
            "Comma-separated session quality gates to include (candidate, "
            "review_required, quarantine, unassessed); default is candidate only"
        ),
    )
    parser.add_argument(
        "--quality-overrides",
        type=Path,
        help=(
            "Provider-neutral JSON override file keyed by session/source hash; "
            "changes only the quality gate, never the training lane"
        ),
    )
    parser.add_argument(
        "--model-tiers",
        default="tier1_frontier",
        help=(
            "Comma-separated model provenance tiers to include; the CLI default "
            "is tier1_frontier. Use tier1_frontier,tier2_open_source,tier3_local,"
            "unclassified for an audit build."
        ),
    )
    parser.add_argument(
        "--model-tier-overrides",
        type=Path,
        help=(
            "Reviewed JSON assignments keyed by source/session identity; "
            "required to promote legacy rows without authoritative tier metadata"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_datasets(
            args.inputs,
            output_dir=args.output_dir,
            privacy_mode=args.privacy_mode,
            privacy_approved=args.privacy_approved,
            max_record_chars=args.max_record_chars,
            overwrite=args.overwrite,
            oversize_strategy=args.oversize_strategy,
            max_records_per_file=args.max_records_per_file,
            privacy_manifest=args.privacy_manifest,
            training_lanes=(
                lane.strip() for lane in args.training_lanes.split(",")
            ),
            quality_gates=(
                gate.strip() for gate in args.quality_gates.split(",")
            ),
            quality_overrides=load_quality_overrides(args.quality_overrides),
            model_tiers=(
                tier.strip() for tier in args.model_tiers.split(",")
            ),
            model_tier_overrides=load_model_tier_overrides(args.model_tier_overrides),
        )
        counts = manifest["counts"]
        print(
            "Built training data: "
            f"{counts['sft']:,} SFT, {counts['trajectories']:,} trajectories, "
            f"{counts['tool_traces']:,} tool traces, "
            f"{counts['action_windows']:,} action windows, "
            f"{counts['preferences']:,} preferences, {counts['rl_prompts']:,} RL prompts, "
            f"{counts['rejected']:,} rejected"
        )
        print(f"Manifest: {args.output_dir / 'manifest.json'}")
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
