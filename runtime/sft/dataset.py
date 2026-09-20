"""Shared immutable-release validation and final-assistant tokenization."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RELEASE_SCHEMA = "ai-data-extraction/agent-sft-pilot/v1"
EXAMPLE_SCHEMA = "ai-data-extraction/agent-sft-example/v1"


def resolve_text_tokenizer(processing_class: Any) -> Any:
    """Return the text tokenizer owned by a tokenizer or multimodal processor."""
    candidate = getattr(processing_class, "tokenizer", processing_class)
    if not callable(getattr(candidate, "apply_chat_template", None)):
        raise TypeError("processing class does not expose a text chat tokenizer")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(f"unexpected trainer release schema: {path}")
    return value


def iter_release_rows(
    input_dir: Path, manifest: dict[str, Any]
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("release manifest has no file bindings")
    for split in ("train", "validation"):
        name = f"{split}.jsonl"
        descriptor = files.get(name)
        path = input_dir / name
        if not isinstance(descriptor, dict) or not path.is_file():
            raise ValueError(f"release file binding is missing: {name}")
        if path.stat().st_size != descriptor.get("bytes") or sha256_file(path) != descriptor.get("sha256"):
            raise ValueError(f"release file binding changed: {name}")
        count = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object release row: {path}:{line_number}")
                count += 1
                yield split, line_number, row
        if count != descriptor.get("records"):
            raise ValueError(f"release record count changed: {name}")


def validate_example(row: dict[str, Any], *, expected_split: str) -> None:
    allowed = {"schema_version", "example_id", "split", "lane", "messages", "tools"}
    if not set(row).issubset(allowed) or set(row) - allowed:
        raise ValueError("trainer example contains unsupported columns")
    if row.get("schema_version") != EXAMPLE_SCHEMA:
        raise ValueError("trainer example schema mismatch")
    if row.get("split") != expected_split:
        raise ValueError("trainer example split mismatch")
    if not isinstance(row.get("example_id"), str) or not row["example_id"]:
        raise ValueError("trainer example identity missing")
    if row.get("lane") not in {
        "reviewed_final_answer",
        "frontier_action_window",
        "skill_policy",
        "tool_policy_replay",
        "verified_open_swe_action",
    }:
        raise ValueError("trainer example lane is invalid")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("trainer messages are invalid")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("trainer message is not an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("trainer message role is invalid")
        if not isinstance(message.get("content"), str):
            raise ValueError("trainer message content is invalid")
    if messages[-1].get("role") != "assistant":
        raise ValueError("trainer target is not an assistant turn")
    tools = row.get("tools")
    schema_by_name: dict[str, dict[str, Any]] = {}
    if tools is not None:
        if not isinstance(tools, list) or not tools:
            raise ValueError("trainer tool schemas are invalid")
        names: set[str] = set()
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            parameters = function.get("parameters") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name or name in names or not isinstance(parameters, dict):
                raise ValueError("trainer tool schema is malformed or duplicated")
            names.add(name)
            schema_by_name[name] = function
    call_ids: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            call_id = call.get("id") if isinstance(call, dict) else None
            if (
                not isinstance(name, str)
                or not isinstance(arguments, dict)
                or not isinstance(call_id, str)
                or not call_id
            ):
                raise ValueError("trainer tool call is malformed")
            if call_id in call_ids:
                raise ValueError("trainer tool call ID is duplicated")
            call_ids.add(call_id)
            schema = schema_by_name.get(name)
            if schema is None:
                raise ValueError("trainer tool call has no matching schema")
            parameters = schema["parameters"]
            properties = parameters.get("properties")
            required = parameters.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                raise ValueError("trainer tool parameter contract is invalid")
            if not set(required).issubset(arguments):
                raise ValueError("trainer tool call is missing a required argument")
            if parameters.get("additionalProperties") is False and not set(arguments).issubset(properties):
                raise ValueError("trainer tool call has an undeclared argument")


def tokenize_final_assistant(
    row: dict[str, Any], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    messages = row["messages"]
    tools = row.get("tools")
    kwargs = {
        "tools": tools,
        "tokenize": True,
        # Transformers 5 defaults this API to BatchEncoding; pin the list
        # contract used for exact prefix comparison and pre-tokenized TRL rows.
        "return_dict": False,
        "enable_thinking": False,
    }
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, **kwargs
    )
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, **kwargs
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        raise ValueError("chat template did not return token ID lists")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("full conversation does not preserve the exact generation prefix")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("assistant target token range is empty")
    if len(full_ids) > max_length:
        raise ValueError(f"sequence exceeds {max_length} tokens: {len(full_ids)}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(labels) != len(full_ids) or all(value == -100 for value in labels):
        raise ValueError("completion-only labels are invalid")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "example_id": row["example_id"],
        "lane": row["lane"],
        "sequence_tokens": len(full_ids),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(full_ids) - len(prompt_ids),
    }


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    index = math.ceil(percentile * len(values)) - 1
    return sorted(values)[max(0, min(index, len(values) - 1))]


def preflight_release(
    input_dir: Path,
    tokenizer: Any,
    *,
    max_length: int,
    retain_tokens: bool = False,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    input_dir = input_dir.resolve()
    manifest = load_manifest(input_dir)
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("max_sequence_tokens") != max_length:
        raise ValueError("runtime max length differs from the release contract")
    seen: set[str] = set()
    parentless_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    lengths: list[int] = []
    targets: list[int] = []
    lane_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    target_tool_counts: Counter[str] = Counter()
    tool_calls = 0
    tool_schema_rows = 0
    for split, line_number, row in iter_release_rows(input_dir, manifest):
        validate_example(row, expected_split=split)
        if row["example_id"] in seen:
            raise ValueError(f"duplicate trainer example: {row['example_id']}")
        seen.add(row["example_id"])
        tokenized = tokenize_final_assistant(row, tokenizer, max_length=max_length)
        lengths.append(tokenized["sequence_tokens"])
        targets.append(tokenized["target_tokens"])
        lane_counts[f"{split}:{row['lane']}"] += 1
        split_counts[split] += 1
        if row.get("tools"):
            tool_schema_rows += 1
        for message in row["messages"]:
            calls = message.get("tool_calls") or []
            tool_calls += len(calls)
        for call in row["messages"][-1].get("tool_calls") or []:
            target_tool_counts[call["function"]["name"]] += 1
        if retain_tokens:
            parentless_rows[split].append(
                {key: value for key, value in tokenized.items() if key not in {"sequence_tokens", "prompt_tokens", "target_tokens"}}
            )
    expected_total = manifest.get("counts", {}).get("total")
    if expected_total != len(seen):
        raise ValueError("release total does not reconcile with trainer rows")
    report = {
        "schema_version": "ai-data-extraction/agent-sft-preflight/v1",
        "status": "passed",
        "release": {
            "path": str(input_dir),
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "length": len(tokenizer),
            "chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template).encode("utf-8")
            ).hexdigest(),
        },
        "contract": {
            "max_length": max_length,
            "enable_thinking": False,
            "loss": "final_assistant_turn_only",
            "truncation": False,
            "packing": False,
        },
        "counts": {
            "total": len(seen),
            "splits": dict(sorted(split_counts.items())),
            "lanes": dict(sorted(lane_counts.items())),
            "tool_schema_rows": tool_schema_rows,
            "tool_calls": tool_calls,
            "target_tools": dict(sorted(target_tool_counts.items())),
        },
        "tokens": {
            "sequence": {
                "sum": sum(lengths),
                "min": min(lengths, default=0),
                "p50": _percentile(lengths, 0.50),
                "p95": _percentile(lengths, 0.95),
                "max": max(lengths, default=0),
            },
            "target": {
                "sum": sum(targets),
                "min": min(targets, default=0),
                "p50": _percentile(targets, 0.50),
                "p95": _percentile(targets, 0.95),
                "max": max(targets, default=0),
            },
        },
    }
    return report, parentless_rows
