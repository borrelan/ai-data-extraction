#!/usr/bin/env python3
"""Build a review-only, tool-free silver SFT salvage pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer_export import export_silver_sft_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-tier", default="tier1_frontier")
    parser.add_argument("--training-lane", default="primary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = export_silver_sft_pilot(
            args.candidate_path,
            args.output_dir,
            required_model_tier=args.model_tier,
            required_training_lane=args.training_lane,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
