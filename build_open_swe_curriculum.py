#!/usr/bin/env python3
"""Stream pinned Open-SWE parquet shards into bounded training candidates.

The adapter removes hidden reasoning, verifies positional action/observation
joins, excludes evaluation repositories and tasks, and emits a trainer-ready
positive SFT partition plus audit-only same-state preference candidates.  It
never treats a provider or teacher identity as a quality label.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


RELEASE_SCHEMA = "ai-data-extraction/agent-sft-pilot/v1"
EXAMPLE_SCHEMA = "ai-data-extraction/agent-sft-example/v1"
PREFERENCE_SCHEMA = "ai-data-extraction/agent-preference-candidate/v1"
DECISION_SCHEMA = "ai-data-extraction/open-swe-source-decision/v1"
LINEAGE_SCHEMA = "ai-data-extraction/open-swe-lineage/v1"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TARGET_TEACHER = "Qwen3.8-27B"
COMPARISON_TEACHER = "Qwen3.6-27B"
DEFAULT_MODEL_DIR = Path("/data-120/models/Qwen3.5-9B")
DEFAULT_REGISTRY = Path(
    "/home/borrelan/Projects/Personal/ai-agent-benchmark/registry/tasks.jsonl"
)
DEFAULT_EVAL_CASES = Path(".tmp/qwen3_8_behavior_gate_20260919_v1/cases.jsonl")

VISIBLE_REASONING = re.compile(r"<(?:think|analysis)>|```(?:reasoning|analysis)", re.I)
TERMINAL_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
PR_DESCRIPTION = re.compile(
    r"<pr_description>\s*(.*?)\s*</pr_description>", re.I | re.S
)
SOURCE_PROTOCOL_MARKER = re.compile(
    r"<instructions>|\bTHOUGHT section\b|exactly ONE bash command|"
    r"patch\.txt|COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    re.I,
)
STATE_CHAIN_DOMAIN = b"ai-data-extraction/transition-state-chain/v1\x00"
MUTATION_COMMAND = re.compile(
    r"(?:apply_patch|patch\s+-p|sed\s+-i|perl\s+-pi|git\s+apply|"
    r"(?:^|[;&|]\s*)(?:rm|mv|cp|mkdir|touch|chmod|chown)\b|"
    r"(?:^|[;&|]\s*)(?:cat|printf|echo)\b[^\n]*(?:>|>>)|"
    r"python[^\n]*(?:write_text|write_bytes|open\([^)]*['\"]w))",
    re.I,
)
VERIFY_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|cargo\s+test|go\s+test|npm\s+(?:test|run\s+test)|"
    r"pnpm\s+(?:test|run\s+test)|yarn\s+test|make\s+(?:test|check)|"
    r"ctest|meson\s+test|ninja\s+test|mvn\s+test|gradle\s+test|"
    r"python\s+-m\s+(?:pytest|unittest)|ruff\s+check|mypy|tsc\b)",
    re.I,
)
FAILURE_TEXT = re.compile(
    r"(?:traceback|\bfailed\b|\berror\b|command not found|no such file|"
    r"permission denied|timed? out)",
    re.I,
)
DEFAULT_LANGUAGE_CAPS = {
    "python": 160,
    "ts": 80,
    "js": 80,
    "rust": 120,
    "c": 80,
    "cpp": 80,
    "go": 80,
    "java": 80,
    "php": 40,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptor(path: Path, records: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        descriptor["records"] = records
    return descriptor


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield line_number, value


def iter_parquet(path: Path, *, batch_size: int = 16) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment qualification
        raise RuntimeError("pyarrow is required for Open-SWE parquet ingress") from exc

    row_number = 0
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise ValueError(f"non-object parquet row at {path}:{row_number}")
            yield row_number, row
            row_number += 1


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            raw = canonical_bytes(row) + b"\n"
            output.write(raw)
            digest.update(raw)
            count += 1
        output.flush()
        os.fsync(output.fileno())
    return {"records": count, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def append_jsonl(output: Any, row: dict[str, Any]) -> None:
    output.write(canonical_bytes(row) + b"\n")


def stable_split(parent_id: str, *, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def stable_call_id(action_ordinal: int, call_ordinal: int) -> str:
    return f"call_a{action_ordinal:04d}_c{call_ordinal:02d}"


def parse_json_object(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field}_invalid_json") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field}_not_object")
    return value


def normalize_tools(source_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(source_tools, list) or not source_tools:
        raise ValueError("tools_missing")
    normalized: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(source_tools):
        tool = parse_json_object(raw, f"tool_{index}")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool_function_missing")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, dict):
            raise ValueError("tool_contract_invalid")
        if name != "bash":
            raise ValueError("unsupported_source_tool")
        properties = parameters.get("properties")
        if not isinstance(properties, dict) or "command" not in properties:
            raise ValueError("bash_command_contract_missing")
        normalized["exec_command"] = {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Run a shell command in the active task environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "Shell command to execute.",
                        }
                    },
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
        }
    return [normalized[name] for name in sorted(normalized)]


def normalize_call(call: Any, *, action_ordinal: int, call_ordinal: int) -> dict[str, Any]:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        raise ValueError("tool_call_invalid")
    function = call["function"]
    if function.get("name") != "bash":
        raise ValueError("unsupported_source_call")
    arguments = parse_json_object(function.get("arguments"), "tool_arguments")
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError("bash_command_missing")
    return {
        "id": stable_call_id(action_ordinal, call_ordinal),
        "type": "function",
        "function": {"name": "exec_command", "arguments": {"cmd": command}},
    }


def contains_reasoning_marker(messages: Any) -> bool:
    return isinstance(messages, list) and any(
        isinstance(message, dict)
        and isinstance(message.get("content"), str)
        and VISIBLE_REASONING.search(message["content"])
        for message in messages
    )


def normalize_base_message(role: str, content: str) -> tuple[dict[str, str], bool]:
    """Remove only the identified mini-SWE harness wrapper from user context."""
    if (
        role == "user"
        and "<instructions>" in content.lower()
        and TERMINAL_SENTINEL in content
    ):
        match = PR_DESCRIPTION.search(content)
        if match is None:
            raise ValueError("source_harness_pr_description_missing")
        task = match.group(1).strip()
        if not task:
            raise ValueError("source_harness_pr_description_empty")
        return {"role": role, "content": f"Repository task:\n\n{task}"}, True
    return {"role": role, "content": content}, False


def tool_result_failed(content: str) -> bool:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return bool(FAILURE_TEXT.search(content))
    if not isinstance(parsed, dict):
        return bool(FAILURE_TEXT.search(content))
    returncode = parsed.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        return True
    output = parsed.get("output")
    return isinstance(output, str) and bool(FAILURE_TEXT.search(output))


@dataclass(frozen=True)
class ActionTurn:
    ordinal: int
    target: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    mutates: bool
    verifies: bool
    failed: bool

    @property
    def chunk(self) -> list[dict[str, Any]]:
        return [self.target, *self.observations]


@dataclass(frozen=True)
class NormalizedTrajectory:
    base_messages: tuple[dict[str, Any], ...]
    turns: tuple[ActionTurn, ...]
    tools: tuple[dict[str, Any], ...]
    terminal_sentinel_removed: bool
    source_harness_instructions_removed: bool

    def prompt_for(self, action_ordinal: int) -> list[dict[str, Any]]:
        prompt = list(self.base_messages)
        for turn in self.turns[:action_ordinal]:
            prompt.extend(turn.chunk)
        return prompt


def normalize_trajectory(row: dict[str, Any]) -> NormalizedTrajectory:
    source_messages = row.get("messages")
    if not isinstance(source_messages, list) or len(source_messages) < 3:
        raise ValueError("messages_invalid")
    if contains_reasoning_marker(source_messages):
        raise ValueError("visible_reasoning_marker")
    tools = normalize_tools(row.get("tools"))
    base: list[dict[str, Any]] = []
    turns: list[ActionTurn] = []
    terminal_sentinel_removed = False
    source_harness_instructions_removed = False
    index = 0
    while index < len(source_messages) and source_messages[index].get("role") in {
        "system",
        "user",
    }:
        message = source_messages[index]
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("base_message_content_invalid")
        normalized, removed = normalize_base_message(message["role"], content)
        base.append(normalized)
        source_harness_instructions_removed |= removed
        index += 1
    if not any(message["role"] == "user" for message in base):
        raise ValueError("initial_user_missing")

    while index < len(source_messages):
        source_action = source_messages[index]
        if not isinstance(source_action, dict) or source_action.get("role") != "assistant":
            raise ValueError("unexpected_message_order")
        source_calls = source_action.get("tool_calls") or []
        if not source_calls:
            # Open-SWE terminal text is verbose teacher narration. It remains
            # audit evidence but is not a positive target for this curriculum.
            index += 1
            if index != len(source_messages):
                raise ValueError("nonterminal_text_action")
            break
        ordinal = len(turns)
        calls = [
            normalize_call(call, action_ordinal=ordinal, call_ordinal=call_index)
            for call_index, call in enumerate(source_calls)
        ]
        target = {"role": "assistant", "content": "", "tool_calls": calls}
        index += 1
        commands = [call["function"]["arguments"]["cmd"] for call in calls]
        if (
            index == len(source_messages)
            and len(commands) == 1
            and TERMINAL_SENTINEL in commands[0]
        ):
            terminal_sentinel_removed = True
            break
        observations: list[dict[str, Any]] = []
        for call_index, call in enumerate(calls):
            if index >= len(source_messages):
                raise ValueError("observation_missing")
            source_observation = source_messages[index]
            if (
                not isinstance(source_observation, dict)
                or source_observation.get("role") != "tool"
                or not isinstance(source_observation.get("content"), str)
            ):
                raise ValueError("observation_join_ambiguous")
            observations.append(
                {
                    "role": "tool",
                    "content": source_observation["content"],
                    "name": call["function"]["name"],
                    "tool_call_id": call["id"],
                }
            )
            index += 1
        if index < len(source_messages) and source_messages[index].get("role") == "tool":
            raise ValueError("observation_join_ambiguous")
        turns.append(
            ActionTurn(
                ordinal=ordinal,
                target=target,
                observations=tuple(observations),
                mutates=any(MUTATION_COMMAND.search(command) for command in commands),
                verifies=any(VERIFY_COMMAND.search(command) for command in commands),
                failed=any(tool_result_failed(item["content"]) for item in observations),
            )
        )
    if not turns:
        raise ValueError("tool_actions_missing")
    return NormalizedTrajectory(
        tuple(base),
        tuple(turns),
        tuple(tools),
        terminal_sentinel_removed,
        source_harness_instructions_removed,
    )


def selected_action_categories(trajectory: NormalizedTrajectory) -> dict[int, list[str]]:
    selected: defaultdict[int, list[str]] = defaultdict(list)
    selected[0].append("first_grounded_action")
    for index in range(1, len(trajectory.turns)):
        if trajectory.turns[index - 1].failed:
            selected[index].append("post_failure_recovery")
            break
    mutation_indices = [turn.ordinal for turn in trajectory.turns if turn.mutates]
    if mutation_indices:
        selected[mutation_indices[0]].append("first_mutation")
        last_mutation = mutation_indices[-1]
        for turn in trajectory.turns[last_mutation + 1 :]:
            if turn.verifies:
                selected[turn.ordinal].append("post_mutation_verification")
                break
    return dict(selected)


def bounded_messages(
    trajectory: NormalizedTrajectory,
    action_ordinal: int,
    *,
    max_characters: int,
) -> tuple[list[dict[str, Any]], bool]:
    target = trajectory.turns[action_ordinal].target
    fixed = [*trajectory.base_messages, target]
    if len(canonical_bytes({"messages": fixed, "tools": trajectory.tools})) > max_characters:
        raise ValueError("window_fixed_context_too_large")
    retained: list[ActionTurn] = []
    for turn in reversed(trajectory.turns[:action_ordinal]):
        candidate = [*trajectory.base_messages]
        for prior in [turn, *retained]:
            candidate.extend(prior.chunk)
        candidate.append(target)
        if len(canonical_bytes({"messages": candidate, "tools": trajectory.tools})) > max_characters:
            break
        retained.insert(0, turn)
    messages = list(trajectory.base_messages)
    for turn in retained:
        messages.extend(turn.chunk)
    messages.append(target)
    return messages, len(retained) != action_ordinal


@dataclass(frozen=True)
class WindowRef:
    rank: int
    source_index: int
    row_number: int
    parent_id: str
    row_sha256: str
    action_ordinal: int
    language: str
    categories: tuple[str, ...]


class BoundedLanguageSelector:
    def __init__(self, caps: dict[str, int]) -> None:
        self.caps = dict(caps)
        self.heaps: dict[str, list[tuple[int, str, WindowRef]]] = defaultdict(list)
        self.rejected_by_quota: Counter[str] = Counter()

    def add(self, ref: WindowRef) -> None:
        cap = self.caps.get(ref.language, 0)
        if cap <= 0:
            self.rejected_by_quota[ref.language] += 1
            return
        heap = self.heaps[ref.language]
        tie_breaker = f"{ref.parent_id}:{ref.action_ordinal}"
        item = (-ref.rank, tie_breaker, ref)
        if len(heap) < cap:
            heapq.heappush(heap, item)
            return
        if item > heap[0]:
            heapq.heapreplace(heap, item)
        else:
            self.rejected_by_quota[ref.language] += 1

    def selected(self) -> list[WindowRef]:
        return sorted(
            (item[2] for heap in self.heaps.values() for item in heap),
            key=lambda ref: (ref.source_index, ref.row_number, ref.action_ordinal),
        )


def load_evaluation_exclusions(
    benchmark_registry: Path, eval_cases: Path
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    repositories: set[str] = set()
    instances: set[str] = set()
    prompt_hashes: set[str] = set()
    registry_count = 0
    for _, row in iter_jsonl(benchmark_registry):
        registry_count += 1
        repository = row.get("repository")
        instance = row.get("source_instance_id")
        instruction_hash = row.get("instruction_sha256")
        if isinstance(repository, str):
            repositories.add(repository)
        if isinstance(instance, str):
            instances.add(instance)
        if isinstance(instruction_hash, str):
            prompt_hashes.add(instruction_hash)
    diagnostic_count = 0
    for _, row in iter_jsonl(eval_cases):
        diagnostic_count += 1
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            prompt_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    evidence = {
        "benchmark_registry": file_descriptor(benchmark_registry, registry_count),
        "diagnostic_cases": file_descriptor(eval_cases, diagnostic_count),
        "repository_count": len(repositories),
        "source_instance_count": len(instances),
        "prompt_hash_count": len(prompt_hashes),
    }
    return repositories, instances, prompt_hashes, evidence


def teacher_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    teacher = metadata.get("teacher_model") if isinstance(metadata, dict) else None
    name = teacher.get("name") if isinstance(teacher, dict) else None
    return name if isinstance(name, str) else "unclassified"


def source_identity(row: dict[str, Any], path: Path, row_number: int) -> tuple[str, str, str, str]:
    trajectory_id = row.get("trajectory_id")
    instance_id = row.get("instance_id")
    repository = row.get("repo")
    language = row.get("language")
    if not all(isinstance(value, str) and value for value in (trajectory_id, instance_id, repository, language)):
        raise ValueError(f"source identity missing at {path}:{row_number}")
    return trajectory_id, instance_id, repository, language


def first_user_hash(trajectory: NormalizedTrajectory) -> str:
    user = next(message for message in trajectory.base_messages if message["role"] == "user")
    return hashlib.sha256(user["content"].encode("utf-8")).hexdigest()


def create_transition_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE transitions (
            instance_id TEXT NOT NULL,
            repository TEXT NOT NULL,
            language TEXT NOT NULL,
            teacher TEXT NOT NULL,
            outcome INTEGER NOT NULL,
            state_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            row_number INTEGER NOT NULL,
            action_ordinal INTEGER NOT NULL,
            PRIMARY KEY (teacher, source_index, row_number, action_ordinal)
        )
        """
    )
    connection.execute(
        "CREATE INDEX transition_join ON transitions(instance_id, state_hash, outcome, teacher)"
    )
    return connection


def transition_signatures(
    trajectory: NormalizedTrajectory,
) -> Iterator[tuple[str, str]]:
    """Yield exact-history index keys in O(total normalized trajectory bytes)."""

    state = hashlib.sha256(
        STATE_CHAIN_DOMAIN
        + canonical_bytes(
            {
                "base_messages": trajectory.base_messages,
                "tools": trajectory.tools,
            }
        )
    ).digest()
    for turn in trajectory.turns:
        yield state.hex(), digest_value(turn.target)
        state = hashlib.sha256(
            STATE_CHAIN_DOMAIN + state + canonical_bytes(turn.chunk)
        ).digest()


def index_transitions(
    connection: sqlite3.Connection,
    *,
    source_index: int,
    row_number: int,
    trajectory: NormalizedTrajectory,
    instance_id: str,
    repository: str,
    language: str,
    teacher: str,
    outcome: int,
) -> None:
    records: list[tuple[Any, ...]] = []
    for turn, (state_hash, target_hash) in zip(
        trajectory.turns, transition_signatures(trajectory), strict=True
    ):
        records.append(
            (
                instance_id,
                repository,
                language,
                teacher,
                outcome,
                state_hash,
                target_hash,
                source_index,
                row_number,
                turn.ordinal,
            )
        )
    connection.executemany(
        "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records
    )


def preference_refs(connection: sqlite3.Connection, cap: int) -> list[tuple[int, ...]]:
    query = """
        SELECT c.source_index, c.row_number, c.action_ordinal,
               r.source_index, r.row_number, r.action_ordinal
        FROM transitions AS c
        JOIN transitions AS r
          ON c.instance_id = r.instance_id
         AND c.state_hash = r.state_hash
        WHERE c.teacher = ? AND c.outcome = 1
          AND r.teacher = ? AND r.outcome = 0
          AND c.target_hash != r.target_hash
          AND c.repository = r.repository
        ORDER BY c.instance_id, c.state_hash, c.action_ordinal, r.action_ordinal
        LIMIT ?
    """
    return [tuple(int(value) for value in row) for row in connection.execute(
        query, (TARGET_TEACHER, COMPARISON_TEACHER, cap)
    )]


def load_rows_by_ref(
    source_paths: list[Path], refs: Iterable[tuple[int, int]]
) -> dict[tuple[int, int], dict[str, Any]]:
    wanted_by_source: defaultdict[int, set[int]] = defaultdict(set)
    for source_index, row_number in refs:
        wanted_by_source[source_index].add(row_number)
    loaded: dict[tuple[int, int], dict[str, Any]] = {}
    for source_index, wanted in sorted(wanted_by_source.items()):
        for row_number, row in iter_parquet(source_paths[source_index]):
            if row_number in wanted:
                loaded[(source_index, row_number)] = row
    if len(loaded) != sum(len(values) for values in wanted_by_source.values()):
        raise ValueError("selected parquet rows could not be reloaded")
    return loaded


def assert_reasoning_absent(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {
                "reasoning",
                "reasoning_content",
                "thought",
                "thoughts",
                "chain_of_thought",
            }:
                raise ValueError("reasoning field survived projection")
            assert_reasoning_absent(child)
    elif isinstance(value, list) or isinstance(value, tuple):
        for child in value:
            assert_reasoning_absent(child)
    elif isinstance(value, str) and VISIBLE_REASONING.search(value):
        raise ValueError("visible reasoning marker survived projection")


def assert_source_protocol_absent(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_source_protocol_absent(child)
    elif isinstance(value, list) or isinstance(value, tuple):
        for child in value:
            assert_source_protocol_absent(child)
    elif isinstance(value, str) and SOURCE_PROTOCOL_MARKER.search(value):
        raise ValueError("source harness protocol survived projection")


def build_open_swe_curriculum(
    *,
    source_paths: list[Path],
    output_dir: Path,
    benchmark_registry: Path,
    eval_cases: Path,
    dataset_id: str,
    dataset_revision: str,
    validation_percent: int = 20,
    max_characters: int = 32_000,
    max_windows_per_parent: int = 6,
    language_caps: dict[str, int] | None = None,
    preference_cap: int = 256,
) -> dict[str, Any]:
    if not source_paths:
        raise ValueError("at least one source parquet is required")
    if not 1 <= validation_percent <= 50:
        raise ValueError("validation_percent must be between 1 and 50")
    if not 1 <= max_windows_per_parent <= 6:
        raise ValueError("max_windows_per_parent must be between 1 and 6")
    source_paths = [path.resolve() for path in source_paths]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    benchmark_registry = benchmark_registry.resolve()
    eval_cases = eval_cases.resolve()
    excluded_repos, excluded_instances, excluded_prompts, exclusion_evidence = (
        load_evaluation_exclusions(benchmark_registry, eval_cases)
    )
    caps = dict(language_caps or DEFAULT_LANGUAGE_CAPS)
    selector = BoundedLanguageSelector(caps)
    decisions = Counter()
    teacher_counts = Counter()
    outcome_counts = Counter()
    language_counts = Counter()
    source_descriptors = [file_descriptor(path) for path in source_paths]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    transition_db = staging / "transition-index.sqlite"
    connection = create_transition_index(transition_db)
    decisions_path = staging / "decisions.jsonl"
    try:
        with decisions_path.open("wb") as decision_output:
            for source_index, path in enumerate(source_paths):
                source_sha256 = source_descriptors[source_index]["sha256"]
                for row_number, row in iter_parquet(path):
                    decisions["input_rows"] += 1
                    row_sha256 = digest_value(row)
                    reason = None
                    trajectory: NormalizedTrajectory | None = None
                    try:
                        parent_id, instance_id, repository, language = source_identity(
                            row, path, row_number
                        )
                        teacher = teacher_identity(row)
                        outcome = row.get("resolved")
                        if outcome not in {-1, 0, 1}:
                            raise ValueError("outcome_invalid")
                        teacher_counts[teacher] += 1
                        outcome_counts[str(outcome)] += 1
                        language_counts[language] += 1
                        if repository in excluded_repos:
                            reason = "evaluation_repository_overlap"
                        elif instance_id in excluded_instances:
                            reason = "evaluation_instance_overlap"
                        else:
                            trajectory = normalize_trajectory(row)
                            if first_user_hash(trajectory) in excluded_prompts:
                                reason = "evaluation_prompt_overlap"
                    except ValueError as exc:
                        reason = str(exc)
                        parent_id = str(row.get("trajectory_id") or f"row-{source_index}-{row_number}")
                        instance_id = str(row.get("instance_id") or "missing")
                        repository = str(row.get("repo") or "missing")
                        language = str(row.get("language") or "missing")
                        teacher = teacher_identity(row)
                        outcome = row.get("resolved") if row.get("resolved") in {-1, 0, 1} else -1

                    candidate_windows = 0
                    if reason is None and trajectory is not None:
                        index_transitions(
                            connection,
                            source_index=source_index,
                            row_number=row_number,
                            trajectory=trajectory,
                            instance_id=instance_id,
                            repository=repository,
                            language=language,
                            teacher=teacher,
                            outcome=outcome,
                        )
                        if teacher == TARGET_TEACHER and outcome == 1:
                            categories = selected_action_categories(trajectory)
                            for action_ordinal, labels in list(categories.items())[
                                :max_windows_per_parent
                            ]:
                                rank = int(
                                    hashlib.sha256(
                                        f"{parent_id}:{action_ordinal}".encode("utf-8")
                                    ).hexdigest(),
                                    16,
                                )
                                selector.add(
                                    WindowRef(
                                        rank=rank,
                                        source_index=source_index,
                                        row_number=row_number,
                                        parent_id=parent_id,
                                        row_sha256=row_sha256,
                                        action_ordinal=action_ordinal,
                                        language=language,
                                        categories=tuple(labels),
                                    )
                                )
                                candidate_windows += 1
                            decision = "eligible_positive_source"
                        elif outcome == 0:
                            decision = "negative_or_rl_source"
                        elif outcome == -1:
                            decision = "audit_only_unknown_outcome"
                        else:
                            decision = "audit_only_non_target_teacher"
                    else:
                        decision = "excluded"
                    decisions[decision] += 1
                    if reason:
                        decisions[f"reason:{reason}"] += 1
                    append_jsonl(
                        decision_output,
                        {
                            "schema_version": DECISION_SCHEMA,
                            "source_file_sha256": source_sha256,
                            "source_row": row_number,
                            "source_row_sha256": row_sha256,
                            "trajectory_id": parent_id,
                            "instance_id": instance_id,
                            "repository": repository,
                            "language": language,
                            "teacher": teacher,
                            "outcome": outcome,
                            "decision": decision,
                            "reason": reason,
                            "candidate_windows": candidate_windows,
                        },
                    )
            decision_output.flush()
            os.fsync(decision_output.fileno())
        connection.commit()

        selected_refs = selector.selected()
        selected_keys = {(ref.source_index, ref.row_number) for ref in selected_refs}
        loaded = load_rows_by_ref(source_paths, selected_keys)
        examples: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        materialization_rejections = Counter()
        selected_by_language = Counter()
        selected_by_category = Counter()
        source_harness_removals = 0
        for ref in selected_refs:
            row = loaded[(ref.source_index, ref.row_number)]
            trajectory = normalize_trajectory(row)
            try:
                messages, context_truncated = bounded_messages(
                    trajectory,
                    ref.action_ordinal,
                    max_characters=max_characters,
                )
            except ValueError as exc:
                materialization_rejections[str(exc)] += 1
                continue
            example_id = "sha256:" + digest_value(
                {
                    "dataset_revision": dataset_revision,
                    "parent": ref.parent_id,
                    "action_ordinal": ref.action_ordinal,
                }
            )
            split = stable_split(ref.parent_id, validation_percent=validation_percent)
            example = {
                "schema_version": EXAMPLE_SCHEMA,
                "example_id": example_id,
                "split": split,
                "lane": "verified_open_swe_action",
                "messages": messages,
                "tools": list(trajectory.tools),
            }
            try:
                assert_reasoning_absent(example)
                assert_source_protocol_absent(example)
            except ValueError as exc:
                materialization_rejections[str(exc)] += 1
                continue
            examples.append(example)
            selected_by_language[ref.language] += 1
            selected_by_category.update(ref.categories)
            source_harness_removals += int(
                trajectory.source_harness_instructions_removed
            )
            lineage.append(
                {
                    "schema_version": LINEAGE_SCHEMA,
                    "example_id": example_id,
                    "parent_id": ref.parent_id,
                    "source_file": str(source_paths[ref.source_index]),
                    "source_file_sha256": source_descriptors[ref.source_index]["sha256"],
                    "source_row": ref.row_number,
                    "source_row_sha256": ref.row_sha256,
                    "action_ordinal": ref.action_ordinal,
                    "selection_categories": list(ref.categories),
                    "language": ref.language,
                    "training_role": "positive_sft",
                    "outcome": "resolved",
                    "outcome_verification": "source_execution_label_not_locally_replayed",
                    "context_truncated_at_message_boundary": context_truncated,
                    "reasoning_removed": True,
                    "tool_mapping": "bash.command_to_exec_command.cmd",
                    "source_terminal_sentinel_removed": trajectory.terminal_sentinel_removed,
                    "source_harness_instructions_removed": (
                        trajectory.source_harness_instructions_removed
                    ),
                }
            )

        ids = [row["example_id"] for row in examples]
        if not examples:
            raise ValueError("no trainer examples survived source qualification")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate trainer example identity")
        parent_splits: defaultdict[str, set[str]] = defaultdict(set)
        split_by_id = {row["example_id"]: row["split"] for row in examples}
        for row in lineage:
            parent_splits[row["parent_id"]].add(split_by_id[row["example_id"]])
        if any(len(splits) != 1 for splits in parent_splits.values()):
            raise ValueError("parent split overlap")

        pair_refs = preference_refs(connection, preference_cap)
        pair_row_refs = {
            (item[0], item[1]) for item in pair_refs
        } | {(item[3], item[4]) for item in pair_refs}
        pair_rows = load_rows_by_ref(source_paths, pair_row_refs) if pair_row_refs else {}
        preference_candidates: list[dict[str, Any]] = []
        preference_rejections = Counter()
        for chosen_source, chosen_row, chosen_action, rejected_source, rejected_row, rejected_action in pair_refs:
            chosen_trajectory = normalize_trajectory(pair_rows[(chosen_source, chosen_row)])
            rejected_trajectory = normalize_trajectory(pair_rows[(rejected_source, rejected_row)])
            chosen_prompt = chosen_trajectory.prompt_for(chosen_action)
            rejected_prompt = rejected_trajectory.prompt_for(rejected_action)
            if canonical_bytes(chosen_prompt) != canonical_bytes(rejected_prompt):
                raise ValueError("transition index state hash collision")
            if canonical_bytes(chosen_trajectory.tools) != canonical_bytes(rejected_trajectory.tools):
                raise ValueError("same-state pair tool contract mismatch")
            if len(
                canonical_bytes(
                    {
                        "prompt": chosen_prompt,
                        "chosen": chosen_trajectory.turns[chosen_action].target,
                        "rejected": rejected_trajectory.turns[rejected_action].target,
                        "tools": chosen_trajectory.tools,
                    }
                )
            ) > max_characters:
                continue
            candidate = {
                "schema_version": PREFERENCE_SCHEMA,
                "candidate_id": "sha256:" + digest_value(
                    {
                        "prompt": chosen_prompt,
                        "chosen": chosen_trajectory.turns[chosen_action].target,
                        "rejected": rejected_trajectory.turns[rejected_action].target,
                    }
                ),
                "training_role": "audit_only",
                "preference_status": "requires_interactive_replay",
                "reason": "same_observable_state_but_trajectory_outcome_only_no_step_credit",
                "prompt": chosen_prompt,
                "chosen": [chosen_trajectory.turns[chosen_action].target],
                "rejected": [rejected_trajectory.turns[rejected_action].target],
                "tools": list(chosen_trajectory.tools),
                "evidence": {
                    "chosen_teacher": TARGET_TEACHER,
                    "chosen_terminal_outcome": "resolved",
                    "rejected_teacher": COMPARISON_TEACHER,
                    "rejected_terminal_outcome": "unresolved",
                    "state_hash": digest_value(
                        {"messages": chosen_prompt, "tools": chosen_trajectory.tools}
                    ),
                },
            }
            try:
                assert_reasoning_absent(candidate)
                assert_source_protocol_absent(candidate)
            except ValueError as exc:
                preference_rejections[str(exc)] += 1
                continue
            preference_candidates.append(candidate)

        examples.sort(key=lambda row: (row["split"], row["example_id"]))
        lineage.sort(key=lambda row: row["example_id"])
        preference_candidates.sort(key=lambda row: row["candidate_id"])
        files = {
            "train.jsonl": write_jsonl(
                staging / "train.jsonl",
                (row for row in examples if row["split"] == "train"),
            ),
            "validation.jsonl": write_jsonl(
                staging / "validation.jsonl",
                (row for row in examples if row["split"] == "validation"),
            ),
            "lineage.jsonl": write_jsonl(staging / "lineage.jsonl", lineage),
            "preference_candidates.jsonl": write_jsonl(
                staging / "preference_candidates.jsonl", preference_candidates
            ),
        }
        decision_descriptor = file_descriptor(decisions_path, decisions["input_rows"])
        decision_descriptor.pop("path")
        files["decisions.jsonl"] = decision_descriptor
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "status": "ready_for_exact_tokenizer_preflight_not_training_authorized",
            "purpose": "bounded_open_swe_long_horizon_curriculum_partition",
            "training_authorized": False,
            "model": {
                "repo_id": "Qwen/Qwen3.5-9B",
                "revision": MODEL_REVISION,
                "local_path": str(DEFAULT_MODEL_DIR),
                "chat_template_kwargs": {"enable_thinking": False},
                "max_sequence_tokens": 8192,
                "loss": "final_assistant_turn_only",
            },
            "source": {
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "files": source_descriptors,
                "ingress": "pyarrow_parquet_iter_batches",
                "batch_size": 16,
            },
            "selection": {
                "target_teacher": TARGET_TEACHER,
                "positive_outcome": 1,
                "language_row_caps": caps,
                "max_windows_per_parent": max_windows_per_parent,
                "max_window_characters": max_characters,
                "validation_percent": validation_percent,
                "parent_disjoint": True,
                "reasoning_removed": True,
                "assistant_tool_narration_removed": True,
                "source_harness_instructions_removed": True,
                "selected_source_harness_removals": source_harness_removals,
                "tool_mapping": "bash.command_to_exec_command.cmd",
                "ambiguous_action_observation_join": "reject",
                "quota_backfill": False,
                "selected_by_language": dict(sorted(selected_by_language.items())),
                "selected_by_category": dict(sorted(selected_by_category.items())),
                "quota_rejections": dict(sorted(selector.rejected_by_quota.items())),
                "materialization_rejections": dict(sorted(materialization_rejections.items())),
                "preference_rejections": dict(sorted(preference_rejections.items())),
            },
            "counts": {
                "total": len(examples),
                "train": sum(row["split"] == "train" for row in examples),
                "validation": sum(row["split"] == "validation" for row in examples),
                "unique_parents": len(parent_splits),
                "source_decisions": dict(sorted(decisions.items())),
                "source_teachers": dict(sorted(teacher_counts.items())),
                "source_outcomes": dict(sorted(outcome_counts.items())),
                "source_languages": dict(sorted(language_counts.items())),
                "preference_candidates": len(preference_candidates),
            },
            "contamination": {
                **exclusion_evidence,
                "policy": "exclude_repository_then_instance_then_exact_prompt",
                "trainer_overlap": 0,
            },
            "quality": {
                "positive_outcomes": "source_execution_label_not_locally_replayed",
                "preference_candidates": "audit_only_pending_interactive_replay",
                "rewards": "not_present",
                "hidden_reasoning": "removed_and_asserted_absent",
                "tool_observation_join": "positional_exact_count_or_reject",
                "tokenizer_preflight": "pending",
            },
            "files": files,
        }
        (staging / "manifest.json").write_bytes(canonical_bytes(manifest) + b"\n")
        connection.close()
        transition_db.unlink(missing_ok=True)
        Path(str(transition_db) + "-wal").unlink(missing_ok=True)
        Path(str(transition_db) + "-shm").unlink(missing_ok=True)
        staging.replace(output_dir)
        manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        return manifest
    except BaseException:
        connection.close()
        shutil.rmtree(staging)
        raise


def parse_language_caps(value: str) -> dict[str, int]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not parsed:
        raise argparse.ArgumentTypeError("language caps must be a non-empty JSON object")
    caps: dict[str, int] = {}
    for language, cap in parsed.items():
        if not isinstance(language, str) or not isinstance(cap, int) or cap < 0:
            raise argparse.ArgumentTypeError("language caps require string keys and non-negative integers")
        caps[language] = cap
    return caps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--dataset-id", default="nvidia/Open-SWE-Traces")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--benchmark-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--eval-cases", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--max-characters", type=int, default=32_000)
    parser.add_argument("--max-windows-per-parent", type=int, default=6)
    parser.add_argument(
        "--language-caps", type=parse_language_caps, default=DEFAULT_LANGUAGE_CAPS
    )
    parser.add_argument("--preference-cap", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_open_swe_curriculum(
        source_paths=args.source,
        output_dir=args.output_dir,
        benchmark_registry=args.benchmark_registry,
        eval_cases=args.eval_cases,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        validation_percent=args.validation_percent,
        max_characters=args.max_characters,
        max_windows_per_parent=args.max_windows_per_parent,
        language_caps=args.language_caps,
        preference_cap=args.preference_cap,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
