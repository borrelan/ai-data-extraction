#!/usr/bin/env python3
"""Validate preference rows with the exact Qwen tokenizer and chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.sft.train import write_json_atomic

try:
    from .dataset import preflight_release
except ImportError:
    from dataset import preflight_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    report, _ = preflight_release(
        args.input_dir.resolve(), tokenizer, max_length=args.max_length
    )
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
