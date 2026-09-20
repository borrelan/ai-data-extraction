#!/usr/bin/env python3
"""Validate every trainer row against the exact Qwen3.5 tokenizer and loss boundary."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

try:
    from .dataset import preflight_release
except ImportError:  # Direct script execution inside the container.
    from dataset import preflight_release


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
