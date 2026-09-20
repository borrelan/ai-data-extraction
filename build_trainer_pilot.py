#!/usr/bin/env python3
"""Build one trainer-facing pilot from verified archived artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer_export import export_unified_training_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silver-dir",
        action="append",
        dest="silver_dirs",
        type=Path,
        required=True,
        help="Existing silver pilot directory; repeat for each verified partition.",
    )
    parser.add_argument("--gold-dialogue-dir", type=Path, required=True)
    parser.add_argument("--gold-tool-dir", type=Path, required=True)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = export_unified_training_pilot(
        silver_pilot_dirs=args.silver_dirs,
        gold_dialogue_dir=args.gold_dialogue_dir,
        gold_tool_dir=args.gold_tool_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
