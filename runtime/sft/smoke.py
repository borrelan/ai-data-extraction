#!/usr/bin/env python3
"""Prove the Qwen3.5 text-only SFT API and PEFT target contract without training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = Path(__file__).with_name("runtime_contract.json")
CRITICAL_LOCAL_HASH_FILES = {
    "chat_template.jinja",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_versions(
    packages: dict[str, Any], *, policy: str = "exact"
) -> dict[str, str]:
    if policy not in {"exact", "record"}:
        raise ValueError(f"invalid package version policy: {policy}")
    versions: dict[str, str] = {}
    for package, expected in packages.items():
        if not isinstance(expected, str):
            raise ValueError(f"invalid package pin for {package}")
        actual = importlib.metadata.version(package)
        if policy == "exact" and actual != expected:
            raise RuntimeError(f"{package} version {actual} != pinned {expected}")
        versions[package] = actual
    return versions


def _verify_acquisition(
    acquisition_path: Path,
    model_dir: Path,
    *,
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    acquisition = _load_object(acquisition_path)
    candidates = [
        model
        for model in acquisition.get("models", [])
        if isinstance(model, dict)
        and model.get("repo_id") == repo_id
        and model.get("revision") == revision
    ]
    if len(candidates) != 1:
        raise ValueError("acquisition manifest does not bind exactly one target model")
    binding = candidates[0]
    files = binding.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("target model has no acquisition file bindings")

    critical_hashes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("invalid acquisition file binding")
        name = item.get("name")
        expected_bytes = item.get("bytes")
        expected_sha = item.get("sha256")
        if not isinstance(name, str) or not isinstance(expected_bytes, int):
            raise ValueError("invalid acquisition name/size binding")
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"model file missing or size mismatch: {name}")
        if name in CRITICAL_LOCAL_HASH_FILES:
            if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
                raise RuntimeError(f"model file hash mismatch: {name}")
            critical_hashes += 1

    if len(list(model_dir.iterdir())) < len(files):
        raise RuntimeError("model directory has fewer entries than its acquisition binding")
    return {
        "manifest": str(acquisition_path),
        "file_count": len(files),
        "critical_hashes_verified": critical_hashes,
        "weight_hashes": "trusted_from_verified_acquisition_manifest",
    }


def _render_tool_example(tokenizer: Any) -> dict[str, Any]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Read README.md and report its first heading."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-smoke-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-smoke-1",
            "name": "read_file",
            "content": "# Example",
        },
        {"role": "assistant", "content": "The first heading is Example."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or "read_file" not in rendered or "README.md" not in rendered:
        raise RuntimeError("Qwen template did not preserve the tool call")
    thought_blocks = re.findall(r"<think>(.*?)</think>", rendered, flags=re.DOTALL)
    if any(block.strip() for block in thought_blocks):
        raise RuntimeError("template introduced non-empty reasoning content")
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise RuntimeError("rendered tool example tokenized to an empty sequence")
    return {
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "characters": len(rendered),
        "tokens": len(token_ids),
        "empty_think_blocks": len(thought_blocks),
    }


def _verify_meta_peft(config: Any, peft_contract: dict[str, Any]) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(f"unexpected text model class: {type(model).__name__}")
    if any(".visual." in f".{name}." for name, _ in model.named_modules()):
        raise RuntimeError("text-only auto class instantiated vision modules")

    lora = LoraConfig(
        r=int(peft_contract["smoke_rank"]),
        lora_alpha=int(peft_contract["smoke_alpha"]),
        target_modules=peft_contract["target_modules"],
        task_type=peft_contract["task_type"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora, low_cpu_mem_usage=True)
    targeted = sorted(
        name
        for name, module in peft_model.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A")
    )
    for fragment in peft_contract["required_target_fragments"]:
        if not any(fragment in f".{name}" for name in targeted):
            raise RuntimeError(f"PEFT target fragment was not resolved: {fragment}")
    forbidden = peft_contract["forbidden_target_fragment"]
    if any(forbidden in f".{name}." for name in targeted):
        raise RuntimeError(f"PEFT targeted forbidden module family: {forbidden}")

    with tempfile.TemporaryDirectory(prefix="sft-peft-config-") as temporary:
        lora.save_pretrained(temporary)
        saved = _load_object(Path(temporary) / "adapter_config.json")
    if saved.get("target_modules") != peft_contract["target_modules"]:
        raise RuntimeError("saved PEFT config changed the all-linear target contract")
    return {
        "model_class": type(model).__name__,
        "target_modules": peft_contract["target_modules"],
        "resolved_lora_module_count": len(targeted),
        "required_target_fragments": peft_contract["required_target_fragments"],
        "adapter_config_serialized": True,
        "base_parameter_device": str(next(model.parameters()).device),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_smoke(
    contract_path: Path,
    model_dir: Path,
    acquisition_manifest: Path,
    *,
    package_version_policy: str = "exact",
    preload_packages: tuple[str, ...] = (),
) -> dict[str, Any]:
    contract = _load_object(contract_path)
    if contract.get("schema") != "ai-data-extraction/sft-runtime-contract/v1":
        raise ValueError("unexpected runtime contract schema")
    model_contract = contract["model"]
    for package in preload_packages:
        importlib.import_module(package)
    versions = _verify_versions(
        contract["packages"], policy=package_version_policy
    )
    acquisition = _verify_acquisition(
        acquisition_manifest,
        model_dir,
        repo_id=model_contract["repo_id"],
        revision=model_contract["revision"],
    )

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != model_contract["expected_parent_model_type"]:
        raise RuntimeError(f"unexpected parent model type: {config.model_type}")
    text_config = config.get_text_config(decoder=True)
    if text_config.model_type != model_contract["expected_text_model_type"]:
        raise RuntimeError(f"unexpected text model type: {text_config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tool_template = _render_tool_example(tokenizer)
    peft = _verify_meta_peft(config, contract["peft"])

    return {
        "schema": "ai-data-extraction/sft-runtime-smoke/v1",
        "status": "passed",
        "contract": str(contract_path),
        "model_dir": str(model_dir),
        "runtime": {
            "package_version_policy": package_version_policy,
            "preloaded_packages": list(preload_packages),
        },
        "versions": versions,
        "acquisition": acquisition,
        "model": {
            "parent_model_type": config.model_type,
            "text_model_type": text_config.model_type,
            "architectures": config.architectures,
            "text_layers": text_config.num_hidden_layers,
            "layer_types": sorted(set(text_config.layer_types)),
        },
        "tool_template": tool_template,
        "peft": peft,
        "side_effects": {
            "model_weights_loaded": False,
            "optimizer_created": False,
            "training_started": False,
            "network_used": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--package-version-policy",
        choices=("exact", "record"),
        default="exact",
        help="Require contract versions or record versions from an immutable image.",
    )
    parser.add_argument(
        "--preload-package",
        action="append",
        default=[],
        help="Import a runtime patch package before Transformers/PEFT checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_smoke(
        args.contract.resolve(),
        args.model_dir.resolve(),
        args.acquisition_manifest.resolve(),
        package_version_policy=args.package_version_policy,
        preload_packages=tuple(args.preload_package),
    )
    if args.output:
        _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
