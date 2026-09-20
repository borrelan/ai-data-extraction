"""Validation and exact-tokenizer rendering for agent preference releases."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from runtime.sft.dataset import EXAMPLE_SCHEMA, sha256_file, validate_example


PAIR_SCHEMA = "ai-data-extraction/agent-preference-pair/v1"
RELEASE_SCHEMA = "ai-data-extraction/agent-preference-release/v1"


def load_manifest(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != RELEASE_SCHEMA:
        raise ValueError(f"unexpected preference release schema: {path}")
    return value


def iter_release_rows(
    input_dir: Path, manifest: dict[str, Any]
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("preference manifest has no file bindings")
    for split in ("train", "validation"):
        name = f"{split}.jsonl"
        descriptor = files.get(name)
        path = input_dir / name
        if not isinstance(descriptor, dict) or not path.is_file():
            raise ValueError(f"preference file binding is missing: {name}")
        if (
            path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise ValueError(f"preference file binding changed: {name}")
        count = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object preference row: {path}:{line_number}")
                count += 1
                yield split, line_number, row
        if count != descriptor.get("records"):
            raise ValueError(f"preference record count changed: {name}")


def validate_pair(row: dict[str, Any], *, expected_split: str) -> None:
    allowed = {
        "schema_version",
        "pair_id",
        "split",
        "lane",
        "prompt",
        "chosen",
        "rejected",
        "tools",
    }
    if not set(row).issubset(allowed):
        raise ValueError("preference pair contains unsupported columns")
    if row.get("schema_version") != PAIR_SCHEMA:
        raise ValueError("preference pair schema mismatch")
    if row.get("split") != expected_split:
        raise ValueError("preference pair split mismatch")
    if not isinstance(row.get("pair_id"), str) or not row["pair_id"]:
        raise ValueError("preference pair identity missing")
    if row.get("lane") not in {"skill_policy_preference", "tool_decision_preference"}:
        raise ValueError("preference pair lane is invalid")
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("preference prompt is invalid")
    if not isinstance(chosen, list) or len(chosen) != 1:
        raise ValueError("preference chosen completion is invalid")
    if not isinstance(rejected, list) or len(rejected) != 1:
        raise ValueError("preference rejected completion is invalid")
    if chosen == rejected:
        raise ValueError("preference completions are identical")

    # Reuse the canonical SFT message/tool-call contract on each branch. This
    # also proves both alternatives originate from the exact same prompt state.
    for side, completion in (("chosen", chosen), ("rejected", rejected)):
        validate_example(
            {
                "schema_version": EXAMPLE_SCHEMA,
                "example_id": f"{row['pair_id']}:{side}",
                "split": expected_split,
                "lane": "skill_policy"
                if row["lane"] == "skill_policy_preference"
                else "tool_policy_replay",
                "messages": prompt + completion,
                "tools": row.get("tools"),
            },
            expected_split=expected_split,
        )


def _template_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("chat template did not return token IDs")
    return value


def _template_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    add_generation_prompt: bool,
) -> str:
    value = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if not isinstance(value, str):
        raise ValueError("chat template did not return text")
    return value


def render_pair(
    row: dict[str, Any], tokenizer: Any, *, max_length: int
) -> tuple[dict[str, str], dict[str, int], dict[str, list[int]]]:
    prompt = row["prompt"]
    chosen_messages = prompt + row["chosen"]
    rejected_messages = prompt + row["rejected"]
    tools = row.get("tools")

    prompt_ids = _template_ids(
        tokenizer, prompt, tools, add_generation_prompt=True
    )
    chosen_ids = _template_ids(
        tokenizer, chosen_messages, tools, add_generation_prompt=False
    )
    rejected_ids = _template_ids(
        tokenizer, rejected_messages, tools, add_generation_prompt=False
    )
    for label, full_ids in (("chosen", chosen_ids), ("rejected", rejected_ids)):
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{label} branch does not preserve the exact prompt prefix")
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(f"{label} completion token range is empty")
        if len(full_ids) > max_length:
            raise ValueError(
                f"{label} branch exceeds {max_length} tokens: {len(full_ids)}"
            )

    prompt_text = _template_text(
        tokenizer, prompt, tools, add_generation_prompt=True
    )
    chosen_text = _template_text(
        tokenizer, chosen_messages, tools, add_generation_prompt=False
    )
    rejected_text = _template_text(
        tokenizer, rejected_messages, tools, add_generation_prompt=False
    )
    if not chosen_text.startswith(prompt_text) or not rejected_text.startswith(prompt_text):
        raise ValueError("rendered preference branch does not preserve prompt text")
    chosen_completion = chosen_text[len(prompt_text) :]
    rejected_completion = rejected_text[len(prompt_text) :]
    if not chosen_completion or not rejected_completion or chosen_completion == rejected_completion:
        raise ValueError("rendered preference completions are empty or identical")
    return (
        {
            "prompt": prompt_text,
            "chosen": chosen_completion,
            "rejected": rejected_completion,
        },
        {
            "prompt": len(prompt_ids),
            "chosen_completion": len(chosen_ids) - len(prompt_ids),
            "rejected_completion": len(rejected_ids) - len(prompt_ids),
            "chosen_sequence": len(chosen_ids),
            "rejected_sequence": len(rejected_ids),
        },
        {
            "prompt_input_ids": prompt_ids,
            "chosen_input_ids": chosen_ids[len(prompt_ids) :],
            "rejected_input_ids": rejected_ids[len(prompt_ids) :],
        },
    )


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
) -> tuple[dict[str, Any], dict[str, list[dict[str, list[int]]]]]:
    input_dir = input_dir.resolve()
    manifest = load_manifest(input_dir)
    if manifest.get("model", {}).get("max_sequence_tokens") != max_length:
        raise ValueError("runtime max length differs from preference release contract")
    seen: set[str] = set()
    tokenized: dict[str, list[dict[str, list[int]]]] = {
        "train": [],
        "validation": [],
    }
    lengths: Counter[str] = Counter()
    values: dict[str, list[int]] = {
        "prompt": [],
        "chosen_completion": [],
        "rejected_completion": [],
        "chosen_sequence": [],
        "rejected_sequence": [],
    }
    lanes: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for split, _, row in iter_release_rows(input_dir, manifest):
        validate_pair(row, expected_split=split)
        pair_id = row["pair_id"]
        if pair_id in seen:
            raise ValueError(f"duplicate preference pair: {pair_id}")
        seen.add(pair_id)
        _, token_counts, trainer_row = render_pair(
            row, tokenizer, max_length=max_length
        )
        for key, count in token_counts.items():
            values[key].append(count)
            lengths[key] += count
        splits[split] += 1
        lanes[f"{split}:{row['lane']}"] += 1
        if retain_tokens:
            tokenized[split].append(trainer_row)
    if len(seen) != manifest.get("counts", {}).get("total"):
        raise ValueError("preference total does not reconcile with trainer rows")
    token_report = {
        key: {
            "sum": lengths[key],
            "min": min(items, default=0),
            "p50": _percentile(items, 0.50),
            "p95": _percentile(items, 0.95),
            "max": max(items, default=0),
        }
        for key, items in values.items()
    }
    report = {
        "schema_version": "ai-data-extraction/agent-preference-preflight/v1",
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
            "same_prompt_state": True,
            "truncation": False,
            "packing": False,
        },
        "counts": {
            "total": len(seen),
            "splits": dict(sorted(splits.items())),
            "lanes": dict(sorted(lanes.items())),
        },
        "tokens": token_report,
    }
    return report, tokenized
