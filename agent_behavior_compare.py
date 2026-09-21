#!/usr/bin/env python3
"""Compare paired agent behavior runs only after proving contract equivalence."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from agent_behavior_gate import (
    RESULT_SCHEMA,
    SUMMARY_SCHEMA,
    digest_value,
    file_binding,
    load_json,
    load_jsonl,
    sha256_file,
)


COMPARISON_SCHEMA = "ai-data-extraction/agent-behavior-comparison/v1"


def load_run(result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = result_path.resolve()
    summary_path = result_path.with_suffix(".summary.json")
    summary = load_json(summary_path)
    rows = load_jsonl(result_path)
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError("run_summary_schema_mismatch")
    evaluation_contract = summary.get("evaluation_contract")
    subject = summary.get("subject")
    if not isinstance(evaluation_contract, dict) or (
        digest_value(evaluation_contract)
        != summary.get("evaluation_contract_sha256")
    ):
        raise ValueError("run_evaluation_contract_digest_mismatch")
    if not isinstance(subject, dict) or (
        digest_value(subject) != summary.get("subject_sha256")
    ):
        raise ValueError("run_subject_digest_mismatch")
    if summary.get("results") != file_binding(result_path):
        raise ValueError("run_result_binding_mismatch")
    if summary.get("case_count") != len(rows):
        raise ValueError("run_result_count_mismatch")
    case_ids = [row.get("case_id") for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("run_case_identity_duplicate")
    for row in rows:
        if (
            row.get("schema_version") != RESULT_SCHEMA
            or row.get("evaluation_contract_sha256")
            != summary.get("evaluation_contract_sha256")
            or row.get("subject_sha256") != summary.get("subject_sha256")
        ):
            raise ValueError("run_row_contract_mismatch")
    return summary, rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        failure.split(":", 1)[0]
        for row in rows
        for failure in row.get("failures", [])
    )
    by_slice: dict[str, dict[str, int]] = {}
    for name in sorted({row["slice"] for row in rows}):
        selected = [row for row in rows if row["slice"] == name]
        by_slice[name] = {
            "attempted": len(selected),
            "passed": sum(bool(row["passed"]) for row in selected),
        }
    return {
        "attempted": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "tool_calls": sum(int(row.get("call_count", 0)) for row in rows),
        "malformed_failures": sum(
            count for reason, count in reasons.items() if reason == "malformed"
        ),
        "forbidden_call_failures": reasons["forbidden_call"],
        "cycle_failures": {
            str(period): reasons[f"cycle_period_{period}"] for period in (1, 2, 3)
        },
        "total_tokens": sum(
            int(row.get("usage", {}).get("total_tokens", 0)) for row in rows
        ),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "by_slice": by_slice,
    }


def compare_run_pairs(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    *,
    minimum_pass_delta: int,
    maximum_regressions: int,
) -> dict[str, Any]:
    if not baseline_paths or len(baseline_paths) != len(candidate_paths):
        raise ValueError("paired_run_count_mismatch")
    all_rows: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    run_bindings: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    transitions: list[dict[str, Any]] = []
    precisions: dict[str, set[str]] = {"baseline": set(), "candidate": set()}
    for baseline_path, candidate_path in zip(baseline_paths, candidate_paths, strict=True):
        baseline_summary, baseline_rows = load_run(baseline_path)
        candidate_summary, candidate_rows = load_run(candidate_path)
        if (
            baseline_summary["evaluation_contract_sha256"]
            != candidate_summary["evaluation_contract_sha256"]
        ):
            raise ValueError("paired_evaluation_contract_mismatch")
        baseline_ids = [row["case_id"] for row in baseline_rows]
        candidate_ids = [row["case_id"] for row in candidate_rows]
        if baseline_ids != candidate_ids:
            raise ValueError("paired_case_order_mismatch")
        if baseline_summary["subject_sha256"] == candidate_summary["subject_sha256"]:
            raise ValueError("paired_subjects_are_identical")
        for role, summary, rows, path in (
            ("baseline", baseline_summary, baseline_rows, baseline_path),
            ("candidate", candidate_summary, candidate_rows, candidate_path),
        ):
            precision = summary.get("subject", {}).get("precision")
            if not isinstance(precision, str) or not precision:
                raise ValueError("run_subject_precision_missing")
            precisions[role].add(precision)
            all_rows[role].extend(rows)
            run_bindings[role].append(
                {
                    "results": file_binding(path),
                    "summary": file_binding(path.with_suffix(".summary.json")),
                    "evaluation_contract_sha256": summary["evaluation_contract_sha256"],
                    "subject_sha256": summary["subject_sha256"],
                }
            )
        for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
            if baseline["passed"] == candidate["passed"]:
                continue
            transitions.append(
                {
                    "case_id": baseline["case_id"],
                    "slice": baseline["slice"],
                    "direction": "win" if candidate["passed"] else "loss",
                    "baseline_failures": baseline["failures"],
                    "candidate_failures": candidate["failures"],
                }
            )
    if len(precisions["baseline"]) != 1 or precisions["baseline"] != precisions["candidate"]:
        raise ValueError("paired_precision_mismatch")

    baseline = aggregate(all_rows["baseline"])
    candidate = aggregate(all_rows["candidate"])
    regressions = sum(item["direction"] == "loss" for item in transitions)
    pass_delta = candidate["passed"] - baseline["passed"]
    checks = {
        "minimum_pass_delta": pass_delta >= minimum_pass_delta,
        "maximum_regressions": regressions <= maximum_regressions,
        "malformed_not_worse": candidate["malformed_failures"] <= baseline["malformed_failures"],
        "forbidden_not_worse": candidate["forbidden_call_failures"] <= baseline["forbidden_call_failures"],
        "cycles_not_worse": all(
            candidate["cycle_failures"][str(period)]
            <= baseline["cycle_failures"][str(period)]
            for period in (1, 2, 3)
        ),
    }
    verdict = "promote_to_real_task_pilot" if all(checks.values()) else "reject"
    return {
        "schema_version": COMPARISON_SCHEMA,
        "verdict": verdict,
        "policy": {
            "minimum_pass_delta": minimum_pass_delta,
            "maximum_regressions": maximum_regressions,
            "structural_diagnostic_only": True,
        },
        "baseline": {"runs": run_bindings["baseline"], "metrics": baseline},
        "candidate": {"runs": run_bindings["candidate"], "metrics": candidate},
        "comparison": {
            "pass_delta": pass_delta,
            "wins": sum(item["direction"] == "win" for item in transitions),
            "losses": regressions,
            "checks": checks,
            "transitions": transitions,
        },
        "next_gate": {
            "real_task_pilot_authorized": verdict == "promote_to_real_task_pilot",
            "full_registry_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", action="append", type=Path, required=True)
    parser.add_argument("--candidate-result", action="append", type=Path, required=True)
    parser.add_argument("--minimum-pass-delta", type=int, default=1)
    parser.add_argument("--maximum-regressions", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_run_pairs(
        args.baseline_result,
        args.candidate_result,
        minimum_pass_delta=args.minimum_pass_delta,
        maximum_regressions=args.maximum_regressions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
