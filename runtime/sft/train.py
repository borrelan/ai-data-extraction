#!/usr/bin/env python3
"""Run one bounded Qwen3.5-9B 4-bit LoRA pilot and save a standard PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

try:
    from .dataset import preflight_release, resolve_text_tokenizer, sha256_file
except ImportError:  # Direct script execution inside the container.
    from dataset import preflight_release, resolve_text_tokenizer, sha256_file


UNSLOTH_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
]


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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


def package_versions(names: list[str]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def ensure_new_directory(path: Path, *, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, processing_class = FastLanguageModel.from_pretrained(
            model_name=str(args.model_dir),
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
            fast_inference=False,
            device_map="xpu:0",
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = resolve_text_tokenizer(processing_class)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=UNSLOTH_TARGET_MODULES,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        return model, tokenizer, {
            "backend": "unsloth",
            "processing_class": type(processing_class).__name__,
            "text_tokenizer_class": type(tokenizer).__name__,
            "target_modules": UNSLOTH_TARGET_MODULES,
            "quantization": "4bit_unsloth",
        }

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    return model, tokenizer, {
        "backend": "transformers",
        "target_modules": "all-linear",
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
    }


def trainable_parameter_report(model: Any) -> dict[str, Any]:
    trainable = 0
    total = 0
    targeted: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            if "lora_" in name:
                targeted.append(name)
    if trainable <= 0 or not targeted:
        raise RuntimeError("model has no trainable LoRA parameters")
    forbidden = [
        name for name in targeted if "visual" in name.lower() or "vision" in name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"text-only run targeted vision parameters: {forbidden[:5]}")
    return {
        "trainable": trainable,
        "total": total,
        "percent": 100.0 * trainable / total,
        "lora_parameter_tensors": len(targeted),
        "vision_lora_parameter_tensors": 0,
        "sample_lora_parameters": targeted[:20],
    }


def verify_adapter(output_dir: Path, base_model: Any) -> dict[str, Any]:
    from peft import PeftConfig, PeftModel
    from safetensors import safe_open

    config_path = output_dir / "adapter_config.json"
    weights_path = output_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("standard PEFT adapter files were not saved")
    config = PeftConfig.from_pretrained(output_dir, local_files_only=True)
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        if not keys or not all("lora_" in key for key in keys):
            raise RuntimeError("saved adapter tensor set is empty or not LoRA-only")
        non_finite = []
        for key in keys:
            tensor = weights.get_tensor(key)
            if not tensor.isfinite().all().item():
                non_finite.append(key)
    if non_finite:
        raise RuntimeError(f"adapter contains non-finite tensors: {non_finite[:5]}")
    reloaded = PeftModel.from_pretrained(
        base_model,
        output_dir,
        is_trainable=False,
        local_files_only=True,
    )
    if not getattr(reloaded, "peft_config", None):
        raise RuntimeError("saved adapter did not reload through PEFT")
    return {
        "config_class": type(config).__name__,
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "tensor_count": len(keys),
        "all_tensors_finite": True,
        "serialization_reloaded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("unsloth", "transformers"), default="unsloth")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.model_dir.is_dir() or not args.input_dir.is_dir():
        raise FileNotFoundError("model or input directory is missing")
    ensure_new_directory(args.output_dir, label="adapter output")
    ensure_new_directory(args.run_dir, label="run directory")
    args.run_dir.mkdir(parents=True)

    random.seed(args.seed)
    model, tokenizer, backend_report = load_model_and_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    preflight, tokenized = preflight_release(
        args.input_dir, tokenizer, max_length=args.max_length, retain_tokens=True
    )
    write_json_atomic(args.run_dir / "dataset-preflight.json", preflight)

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    class CompletionOnlyCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = pad_token_id

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            width = max(len(row["input_ids"]) for row in features)
            input_ids = []
            attention = []
            labels = []
            for row in features:
                padding = width - len(row["input_ids"])
                input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
                attention.append(row["attention_mask"] + [0] * padding)
                labels.append(row["labels"] + [-100] * padding)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    train_dataset = Dataset.from_list(tokenized["train"])
    validation_dataset = Dataset.from_list(tokenized["validation"])
    parameter_report = trainable_parameter_report(model)
    trainer_args = SFTConfig(
        output_dir=str(args.run_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        packing=False,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    numeric_metrics = [
        value for value in metrics.values() if isinstance(value, (int, float))
    ]
    if not numeric_metrics or any(not math.isfinite(float(value)) for value in numeric_metrics):
        raise RuntimeError("training returned missing or non-finite metrics")
    log_history = trainer.state.log_history
    for entry in log_history:
        for key in ("loss", "eval_loss", "grad_norm"):
            value = entry.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite {key} in trainer history")

    args.output_dir.mkdir(parents=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    base_model = model.unload()
    adapter = verify_adapter(args.output_dir, base_model)
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
    versions = package_versions(
        ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes"]
        + (["triton-xpu", "unsloth", "unsloth_zoo"] if args.backend == "unsloth" else [])
    )
    accelerator = (
        {
            "type": "xpu",
            "name": torch.xpu.get_device_name(0),
            "total_memory_bytes": torch.xpu.get_device_properties(0).total_memory,
            "peak_allocated_bytes": torch.xpu.max_memory_allocated(0),
            "peak_reserved_bytes": torch.xpu.max_memory_reserved(0),
        }
        if torch.xpu.is_available()
        else {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
    )
    run_manifest = {
        "schema_version": "ai-data-extraction/agent-sft-run/v1",
        "status": "completed",
        "base_model": {
            "path": str(args.model_dir),
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "config_sha256": sha256_file(args.model_dir / "config.json"),
        },
        "dataset": preflight["release"],
        "runtime": {
            "versions": versions,
            "container_image_id": os.environ.get("SFT_CONTAINER_IMAGE_ID"),
            "accelerator": accelerator,
            **backend_report,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "max_length": args.max_length,
            "packing": False,
            "seed": args.seed,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "metrics": metrics,
            "global_steps": trainer.state.global_step,
            "log_history": log_history,
        },
        "parameters": parameter_report,
        "adapter": {"path": str(args.output_dir), **adapter},
    }
    write_json_atomic(args.run_dir / "run-manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
