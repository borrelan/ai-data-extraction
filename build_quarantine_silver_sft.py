#!/usr/bin/env python3
"""Build the strict archived-quarantine Tier-1 silver SFT salvage pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer_export import export_silver_sft_pilot


BLOCKED_QUALITY_FLAGS = (
    "advisor_overlay_contamination",
    "payload_truncated",
    "source_parse_errors",
    "tool_error_observed",
    "unmatched_tool_calls",
    "unmatched_tool_observations",
    "visible_trainer_marker_review",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = export_silver_sft_pilot(
        args.input_path,
        args.output_dir,
        required_model_tier="tier1_frontier",
        required_training_lane="primary",
        source_partition="sft.quarantine",
        blocked_quality_flags=BLOCKED_QUALITY_FLAGS,
        quality_limitations=[
            "source_quality_denylist_applied",
            "archived_quarantine_source",
        ],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
