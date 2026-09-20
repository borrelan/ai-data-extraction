#!/usr/bin/env python3
"""Materialize a reviewed, parent-disjoint pilot from release partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_gate import materialize_reviewed_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("sft", "tool_traces", "trajectories"),
        default=["sft", "tool_traces"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = materialize_reviewed_pilot(
            args.release_dir,
            args.review_manifest,
            args.output_dir,
            datasets=args.datasets,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
