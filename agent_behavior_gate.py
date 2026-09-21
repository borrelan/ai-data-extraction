#!/usr/bin/env python3
"""Run and score a hash-bound, provider-neutral agent behavior diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "ai-data-extraction/agent-behavior-result/v1"
SUMMARY_SCHEMA = "ai-data-extraction/agent-behavior-summary/v1"
RUNTIME_SCHEMA = "ai-data-extraction/agent-behavior-runtime/v1"
SYSTEM_PROMPT = """You are an engineering agent operating under a strict tool contract.
Use a tool only when it materially advances the request. Emit schema-valid calls.
Do not repeat an action after an unchanged result; change approach or report the
blocker. Read every explicitly required skill before the action it governs. Never
claim an unverified side effect. Finish only after the requested outcome is
verified. Do not reveal hidden reasoning."""
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "max_tokens": 512,
    "chat_template_kwargs": {
        "enable_thinking": False,
        "preserve_thinking": False,
    },
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL object required: {path}:{line_number}")
            rows.append(row)
    return rows


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_binding(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}_binding_invalid")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}_path_invalid")
    actual = file_binding(Path(path_value))
    if value != actual:
        raise ValueError(f"{label}_binding_mismatch")
    return actual


def validate_runtime_manifest(path: Path, *, model: str) -> dict[str, Any]:
    manifest = load_json(path.resolve())
    if manifest.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError("runtime_manifest_schema_mismatch")
    comparison = manifest.get("comparison_contract")
    subject = manifest.get("subject")
    if not isinstance(comparison, dict) or not isinstance(subject, dict):
        raise ValueError("runtime_manifest_contract_invalid")
    server = verify_file_binding(comparison.get("server_binary"), label="server_binary")
    launch = comparison.get("launch")
    if not isinstance(launch, dict) or not launch:
        raise ValueError("runtime_launch_contract_invalid")
    served_alias = subject.get("served_alias")
    if served_alias != model:
        raise ValueError("runtime_served_alias_mismatch")
    model_artifact = verify_file_binding(
        subject.get("model_artifact"), label="model_artifact"
    )
    precision = subject.get("precision")
    if not isinstance(precision, str) or not precision:
        raise ValueError("runtime_subject_precision_missing")
    return {
        "manifest": file_binding(path),
        "comparison_contract": {**comparison, "server_binary": server},
        "subject": {**subject, "model_artifact": model_artifact},
    }


def build_run_identity(
    *,
    cases: Path,
    tools: Path,
    runtime_manifest: Path,
    endpoint: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    runtime = validate_runtime_manifest(runtime_manifest, model=model)
    evaluation_contract = {
        "runner": file_binding(Path(__file__)),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "cases": file_binding(cases),
        "tools": file_binding(tools),
        "sampling": SAMPLING,
        "seed": seed,
        "endpoint": endpoint,
        "runtime": runtime["comparison_contract"],
    }
    return {
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": digest_value(evaluation_contract),
        "subject": runtime["subject"],
        "subject_sha256": digest_value(runtime["subject"]),
        "runtime_manifest": runtime["manifest"],
    }


def normalized_call(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    function = call.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool call function is not an object")
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call has no function name")
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ValueError("tool arguments are not JSON text or object")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments are not an object")
    signature_arguments = dict(arguments)
    if name == "read_skill" and isinstance(signature_arguments.get("name"), str):
        signature_arguments["name"] = signature_arguments["name"].casefold()
    signature = f"{name}:{canonical_bytes(signature_arguments).decode('utf-8')}"
    return name, arguments, signature


def scripted_result(
    case: dict[str, Any], name: str, call_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = case.get("responses", {}).get(name, [])
    entry: Any = (
        configured[min(call_count - 1, len(configured) - 1)]
        if configured
        else {"ok": False, "error": f"no scripted result for {name}"}
    )
    evaluation: dict[str, Any] = {}
    if isinstance(entry, dict) and "tool_result" in entry:
        result = entry.get("tool_result")
        raw_evaluation = entry.get("evaluation", {})
        if not isinstance(raw_evaluation, dict):
            raise ValueError("scripted evaluation metadata is not an object")
        evaluation = raw_evaluation
    else:
        result = entry
    if not isinstance(result, dict):
        raise ValueError("scripted tool result is not an object")
    state_after = evaluation.get("state_after")
    if state_after is not None and (
        not isinstance(state_after, str) or not state_after
    ):
        raise ValueError("scripted state_after is invalid")
    return result, evaluation


def make_turn_event(
    *,
    turn: int,
    state_before: str,
    calls: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    explicit_states = {
        item["state_after"] for item in evaluations if item.get("state_after")
    }
    if len(explicit_states) > 1:
        raise ValueError("parallel_turn_state_after_conflict")
    observations = [item["result"] for item in calls]
    observation_sha256 = digest_value(observations)
    state_after = next(iter(explicit_states), observation_sha256)
    action_signatures = [item["signature"] for item in calls]
    cycle_key = digest_value(
        {
            "action_signatures": action_signatures,
            "observation_sha256": observation_sha256,
            "state_after": state_after,
        }
    )
    return {
        "turn": turn,
        "state_before": state_before,
        "state_after": state_after,
        "state_source": "explicit" if explicit_states else "observation_digest",
        "action_signatures": action_signatures,
        "observation_sha256": observation_sha256,
        "cycle_key": cycle_key,
    }


def detect_cycles(
    turn_events: list[dict[str, Any]], *, maximum_period: int = 3
) -> list[dict[str, Any]]:
    if maximum_period < 1:
        return []
    keys = [event["cycle_key"] for event in turn_events]
    cycles: list[dict[str, Any]] = []
    for end in range(2, len(keys) + 1):
        for period in range(1, min(maximum_period, end // 2) + 1):
            if keys[end - 2 * period : end - period] != keys[end - period : end]:
                continue
            cycles.append(
                {
                    "period": period,
                    "first_turn": turn_events[end - 2 * period]["turn"],
                    "repeat_turn": turn_events[end - period]["turn"],
                    "detected_at_turn": turn_events[end - 1]["turn"],
                }
            )
            break
    return cycles


def score(
    case: dict[str, Any],
    events: list[dict[str, Any]],
    turn_events: list[dict[str, Any]],
    malformed: list[str],
) -> dict[str, Any]:
    oracle = case.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError("case oracle is not an object")
    names = [event["name"] for event in events]
    failures: list[str] = []
    for required in oracle.get("required", []):
        if required not in names:
            failures.append(f"missing_required:{required}")
    required_any = oracle.get("required_any", [])
    if required_any and not any(name in names for name in required_any):
        failures.append("missing_required_any:" + ",".join(required_any))
    for forbidden in oracle.get("forbidden", []):
        if forbidden in names:
            failures.append(f"forbidden_call:{forbidden}")
    allowed_arguments = oracle.get("allowed_arguments", {})
    for event in events:
        allowed = allowed_arguments.get(event["name"])
        if allowed is not None and event["arguments"] not in allowed:
            failures.append(
                f"unexpected_arguments:{event['name']}:{json.dumps(event['arguments'], sort_keys=True)}"
            )
    for before, after, minimum_before_count in oracle.get("required_before", []):
        try:
            after_index = names.index(after)
        except ValueError:
            continue
        if names[:after_index].count(before) < int(minimum_before_count):
            failures.append(f"order:{before}<{after}:need{minimum_before_count}")
    required_skills = [str(value).casefold() for value in oracle.get("required_skills", [])]
    if oracle.get("required_skill"):
        required_skills.append(str(oracle["required_skill"]).casefold())
    read_skills = {
        str(event["arguments"].get("name")).casefold()
        for event in events
        if event["name"] == "read_skill"
        and isinstance(event["arguments"].get("name"), str)
    }
    for skill in required_skills:
        if skill not in read_skills:
            failures.append(f"missing_skill:{skill}")
    maximum_period = int(oracle.get("forbid_cycles_up_to", 3))
    cycles = detect_cycles(turn_events, maximum_period=maximum_period)
    failures.extend(f"cycle_period_{item['period']}" for item in cycles)
    failures.extend(f"malformed:{item}" for item in malformed)
    failure_counts = Counter(failure.split(":", 1)[0] for failure in failures)
    return {
        "passed": not failures,
        "failures": failures,
        "failure_counts": dict(sorted(failure_counts.items())),
        "call_count": len(events),
        "turn_count": len(turn_events),
        "tool_names": names,
        "unique_action_signatures": len({event["signature"] for event in events}),
        "cycles": cycles,
    }


def post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=canonical_bytes(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:2000]}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("server response is not a JSON object")
    return payload


def get_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:2000]}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("server response is not a JSON object")
    return payload


def run_case(
    case: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    *,
    endpoint: str,
    model: str,
    seed: int,
    timeout: float,
    identity: dict[str, Any],
) -> dict[str, Any]:
    selected_tools = [registry[name] for name in case["tools"]]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case["prompt"]},
    ]
    events: list[dict[str, Any]] = []
    turn_events: list[dict[str, Any]] = []
    malformed: list[str] = []
    assistant_messages: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    current_state = str(case.get("oracle", {}).get("initial_state") or digest_value(case["prompt"]))
    started = time.monotonic()
    max_turns = int(case.get("max_turns", 5))

    for turn in range(1, max_turns + 1):
        body = {
            "model": model,
            "messages": messages,
            "tools": selected_tools,
            "tool_choice": "auto",
            "seed": seed,
            **SAMPLING,
        }
        payload = post_json(endpoint, body, timeout)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("server response has no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("server response has no assistant message")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise RuntimeError("assistant tool_calls is not a list")
        assistant_messages.append(
            {
                "turn": turn,
                "content": message.get("content"),
                "finish_reason": choice.get("finish_reason"),
                "tool_call_count": len(raw_calls),
            }
        )
        response_usage = payload.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(response_usage.get(key) or 0)
        messages.append(message)
        if not raw_calls:
            break

        terminal = False
        turn_malformed = False
        turn_calls: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        per_name_count = Counter(event["name"] for event in events)
        for raw_call in raw_calls:
            call_id = raw_call.get("id") or f"call-{turn}-{len(events)}"
            try:
                name, arguments, signature = normalized_call(raw_call)
                per_name_count[name] += 1
                result, evaluation = scripted_result(case, name, per_name_count[name])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                malformed.append(str(error))
                turn_malformed = True
                continue
            event = {
                "turn": turn,
                "name": name,
                "arguments": arguments,
                "signature": signature,
                "result": result,
            }
            events.append(event)
            turn_calls.append(event)
            evaluations.append(evaluation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": canonical_bytes(result).decode("utf-8"),
                }
            )
            if name in {"finish_task", "report_blocker"}:
                terminal = True
        if turn_calls:
            turn_event = make_turn_event(
                turn=turn,
                state_before=current_state,
                calls=turn_calls,
                evaluations=evaluations,
            )
            turn_events.append(turn_event)
            current_state = turn_event["state_after"]
        if terminal or turn_malformed:
            break

    result = score(case, events, turn_events, malformed)
    result.update(
        {
            "schema_version": RESULT_SCHEMA,
            "case_id": case["case_id"],
            "slice": case["slice"],
            "seed": seed,
            "evaluation_contract_sha256": identity["evaluation_contract_sha256"],
            "subject_sha256": identity["subject_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "usage": dict(usage),
            "events": events,
            "turn_events": turn_events,
            "malformed": malformed,
            "assistant_messages": assistant_messages,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-case-count", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    case_ids = [case.get("case_id") for case in cases]
    if not case_ids or any(not isinstance(value, str) or not value for value in case_ids):
        raise SystemExit("every case requires a non-empty case_id")
    if len(set(case_ids)) != len(case_ids):
        raise SystemExit("case IDs are not unique")
    if args.expected_case_count and len(cases) != args.expected_case_count:
        raise SystemExit("case count does not match --expected-case-count")
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in cases}
        if missing:
            raise SystemExit(f"unknown case IDs: {sorted(missing)}")
    if args.limit:
        cases = cases[: args.limit]

    tool_rows = json.loads(args.tools.read_text(encoding="utf-8"))
    if not isinstance(tool_rows, list):
        raise SystemExit("tool registry must be a JSON array")
    registry = {
        tool["function"]["name"]: tool
        for tool in tool_rows
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    missing_tools = sorted(
        {name for case in cases for name in case.get("tools", [])} - registry.keys()
    )
    if missing_tools:
        raise SystemExit(f"missing tool definitions: {missing_tools}")

    identity = build_run_identity(
        cases=args.cases.resolve(),
        tools=args.tools.resolve(),
        runtime_manifest=args.runtime_manifest.resolve(),
        endpoint=args.endpoint,
        model=args.model,
        seed=args.seed,
    )
    models_url = args.endpoint.rsplit("/v1/chat/completions", 1)[0] + "/v1/models"
    served_models = get_json(models_url, min(args.timeout, 10.0))
    aliases = {
        row.get("id")
        for row in served_models.get("data", [])
        if isinstance(row, dict)
    }
    if args.model not in aliases:
        raise SystemExit("runtime model alias is not served")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    results: list[dict[str, Any]] = []
    mode = "w"
    if args.resume:
        if not args.output.is_file() or summary_path.exists():
            raise SystemExit("resume requires an unfinished result file")
        results = load_jsonl(args.output)
        expected_ids = [case["case_id"] for case in cases[: len(results)]]
        if [row.get("case_id") for row in results] != expected_ids:
            raise SystemExit("resume rows are not the exact case prefix")
        for row in results:
            if (
                row.get("schema_version") != RESULT_SCHEMA
                or row.get("seed") != args.seed
                or row.get("evaluation_contract_sha256")
                != identity["evaluation_contract_sha256"]
                or row.get("subject_sha256") != identity["subject_sha256"]
            ):
                raise SystemExit("resume row contract mismatch")
        mode = "a"
    elif args.output.exists() or summary_path.exists():
        raise SystemExit("refusing to overwrite an existing run")

    with args.output.open(mode, encoding="utf-8") as handle:
        for case in cases[len(results) :]:
            result = run_case(
                case,
                registry,
                endpoint=args.endpoint,
                model=args.model,
                seed=args.seed,
                timeout=args.timeout,
                identity=identity,
            )
            results.append(result)
            handle.write(canonical_bytes(result).decode("utf-8") + "\n")
            handle.flush()
            print(
                f"{result['case_id']} passed={result['passed']} "
                f"calls={result['call_count']} elapsed={result['elapsed_seconds']}s"
            )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        **identity,
        "case_count": len(results),
        "passed": sum(bool(row["passed"]) for row in results),
        "failed": sum(not row["passed"] for row in results),
        "by_slice": {
            name: {
                "cases": sum(row["slice"] == name for row in results),
                "passed": sum(row["slice"] == name and row["passed"] for row in results),
            }
            for name in sorted({row["slice"] for row in results})
        },
        "results": file_binding(args.output),
        "total_elapsed_seconds": round(sum(row["elapsed_seconds"] for row in results), 3),
        "total_usage": dict(
            sum((Counter(row["usage"]) for row in results), Counter())
        ),
    }
    temporary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
