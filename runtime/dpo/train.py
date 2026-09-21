#!/usr/bin/env python3
"""Continue a proven Qwen3.5-9B SFT adapter with bounded robust DPO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from runtime.sft.dataset import resolve_text_tokenizer, sha256_file
from runtime.sft.train import ensure_new_directory, verify_adapter, write_json_atomic

try:
    from .dataset import preflight_release
except ImportError:
    from dataset import preflight_release


def package_versions(names: list[str]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


TRAINER_TOKEN_FIELDS = (
    "prompt_input_ids",
    "chosen_input_ids",
    "rejected_input_ids",
)


def token_lengths(row: dict[str, Any]) -> dict[str, int]:
    prompt = len(row["prompt_input_ids"])
    chosen_completion = len(row["chosen_input_ids"])
    rejected_completion = len(row["rejected_input_ids"])
    return {
        "prompt": prompt,
        "chosen_completion": chosen_completion,
        "rejected_completion": rejected_completion,
        "chosen_sequence": prompt + chosen_completion,
        "rejected_sequence": prompt + rejected_completion,
    }


def select_training_rows(
    rows: list[dict[str, Any]], *, one_step_canary: bool
) -> tuple[list[dict[str, list[int]]], dict[str, Any] | None]:
    """Strip provenance fields and make a one-step canary exercise the longest pair."""
    if not rows:
        raise ValueError("DPO training split is empty")
    selected = rows
    selection = None
    if one_step_canary:
        def longest_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, str]:
            lengths = token_lengths(item[1])
            return (
                max(lengths["chosen_sequence"], lengths["rejected_sequence"]),
                lengths["prompt"],
                item[1]["pair_id"],
            )

        source_index, longest = max(
            enumerate(rows),
            key=longest_key,
        )
        selected = [longest]
        selection = {
            "schema_version": "ai-data-extraction/dpo-canary-selection/v1",
            "strategy": "longest_train_pair",
            "source_index": source_index,
            "pair_id": longest["pair_id"],
            "lane": longest["lane"],
            "tokens": token_lengths(longest),
        }
    payload = [
        {field: row[field] for field in TRAINER_TOKEN_FIELDS} for row in selected
    ]
    return payload, selection


def adapter_parameter_sha256(model: Any, adapter_name: str) -> tuple[str, int, int]:
    """Hash one in-memory adapter while normalizing its local PEFT name."""
    import torch

    digest = hashlib.sha256()
    count = 0
    elements = 0
    marker = f".{adapter_name}."
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if marker not in name or "lora_" not in name:
            continue
        normalized_name = name.replace(marker, ".<adapter>.")
        digest.update(normalized_name.encode("utf-8"))
        value = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(value.numpy().tobytes())
        count += 1
        elements += parameter.numel()
    if count == 0:
        raise RuntimeError(f"adapter has no LoRA parameters: {adapter_name}")
    return digest.hexdigest(), count, elements


def trainable_parameter_report(model: Any) -> dict[str, Any]:
    trainable = 0
    total = 0
    names: list[str] = []
    forbidden: list[str] = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        names.append(name)
        if ".default." not in name or "lora_" not in name:
            forbidden.append(name)
    if trainable <= 0 or not names:
        raise RuntimeError("DPO policy has no trainable parameters")
    if forbidden:
        raise RuntimeError(f"non-policy parameters are trainable: {forbidden[:5]}")
    return {
        "trainable": trainable,
        "total": total,
        "percent": 100.0 * trainable / total,
        "parameter_tensors": len(names),
        "sample_parameters": names[:20],
        "only_default_lora_trainable": True,
    }


def load_policy_and_reference(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    # Unsloth must patch Transformers, PEFT, and TRL before those libraries are
    # imported by this process.
    from unsloth import FastLanguageModel

    model, processing_class = FastLanguageModel.from_pretrained(
        model_name=str(args.reference_adapter),
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
    if "default" not in model.peft_config:
        raise RuntimeError("SFT policy adapter was not loaded as the default adapter")
    model.load_adapter(
        str(args.reference_adapter),
        adapter_name="reference",
        is_trainable=False,
    )
    model.set_adapter("default")
    model = FastLanguageModel.for_training(
        model, use_gradient_checkpointing=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_hash, policy_tensors, policy_elements = adapter_parameter_sha256(
        model, "default"
    )
    reference_hash, reference_tensors, reference_elements = adapter_parameter_sha256(
        model, "reference"
    )
    if (policy_hash, policy_tensors, policy_elements) != (
        reference_hash,
        reference_tensors,
        reference_elements,
    ):
        raise RuntimeError("policy and reference adapters differ before DPO")
    return model, processing_class, tokenizer, {
        "backend": "unsloth",
        "processing_class": type(processing_class).__name__,
        "text_tokenizer_class": type(tokenizer).__name__,
        "quantization": "4bit_unsloth",
        "policy_adapter_name": "default",
        "reference_adapter_name": "reference",
        "initial_adapter_parameter_sha256": policy_hash,
        "adapter_parameter_tensors": policy_tensors,
        "adapter_parameter_elements": policy_elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    if args.max_steps not in (-1, 1):
        raise ValueError("max_steps must be -1 for a full release or 1 for a canary")
    for name in ("model_dir", "reference_adapter", "input_dir", "output_dir", "run_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.model_dir, args.reference_adapter, args.input_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"required input directory is missing: {path}")
    ensure_new_directory(args.output_dir, label="DPO adapter output")
    ensure_new_directory(args.run_dir, label="DPO run directory")
    args.run_dir.mkdir(parents=True)
    random.seed(args.seed)

    model, _, tokenizer, backend_report = load_policy_and_reference(args)
    preflight, tokenized = preflight_release(
        args.input_dir,
        tokenizer,
        max_length=args.max_length,
        retain_tokens=True,
    )
    write_json_atomic(args.run_dir / "dataset-preflight.json", preflight)

    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    canary = args.max_steps > 0
    train_rows, canary_selection = select_training_rows(
        tokenized["train"], one_step_canary=canary
    )
    validation_rows, _ = select_training_rows(
        tokenized["validation"], one_step_canary=False
    )
    if canary_selection is not None:
        write_json_atomic(args.run_dir / "canary-selection.json", canary_selection)
    train_dataset = Dataset.from_list(train_rows)
    validation_dataset = Dataset.from_list(validation_rows)
    parameter_report = trainable_parameter_report(model)
    trainer_args = DPOConfig(
        output_dir=str(args.run_dir / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        # A one-step canary cannot spend its only optimizer step at zero LR.
        # The full 33-step release uses one explicit warmup step.
        warmup_steps=0 if canary else 1,
        beta=args.beta,
        loss_type="robust",
        label_smoothing=args.label_smoothing,
        bf16=True,
        fp16=False,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="no" if canary else "epoch",
        save_strategy="no" if canary else "steps",
        save_steps=10,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode="keep_end",
        gradient_checkpointing=True,
        optim="adamw_8bit",
        remove_unused_columns=True,
        dataset_num_proc=1,
        model_adapter_name=None,
        ref_adapter_name="reference",
        precompute_ref_log_probs=False,
        # DPO scores completion labels only. Avoid materializing the 248K-vocab
        # LM head over thousands of prompt tokens in each chosen/rejected pair.
        use_logits_to_keep=True,
    )
    class PretokenizedDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, config, dataset_name):
            required = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
            if not required.issubset(dataset.column_names):
                raise ValueError(
                    f"pretokenized {dataset_name} dataset lacks DPO columns"
                )
            return dataset

    trainer = PretokenizedDPOTrainer(
        model=model,
        ref_model=None,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=None if canary else validation_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    numeric_metrics = [
        value for value in metrics.values() if isinstance(value, (int, float))
    ]
    if not numeric_metrics or any(not math.isfinite(float(value)) for value in numeric_metrics):
        raise RuntimeError("DPO returned missing or non-finite metrics")
    for entry in trainer.state.log_history:
        for key, value in entry.items():
            if isinstance(value, (int, float)) and (
                "loss" in key or key in {"grad_norm", "rewards/accuracies", "rewards/margins"}
            ):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"non-finite DPO metric: {key}")

    unwrapped = trainer.accelerator.unwrap_model(model)
    policy_hash, policy_tensors, policy_elements = adapter_parameter_sha256(
        unwrapped, "default"
    )
    reference_hash, reference_tensors, reference_elements = adapter_parameter_sha256(
        unwrapped, "reference"
    )
    if policy_hash == backend_report["initial_adapter_parameter_sha256"]:
        raise RuntimeError("DPO policy adapter did not change")
    if reference_hash != backend_report["initial_adapter_parameter_sha256"]:
        raise RuntimeError("frozen DPO reference adapter changed")
    if (policy_tensors, policy_elements) != (reference_tensors, reference_elements):
        raise RuntimeError("policy/reference adapter structures diverged")

    args.output_dir.mkdir(parents=True)
    unwrapped.save_pretrained(
        args.output_dir,
        selected_adapters=["default"],
        safe_serialization=True,
    )
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    base_model = unwrapped.unload()
    adapter_report = verify_adapter(args.output_dir, base_model)
    if torch.xpu.is_available():
        torch.xpu.synchronize()
        accelerator = {
            "type": "xpu",
            "name": torch.xpu.get_device_name(0),
            "total_memory_bytes": torch.xpu.get_device_properties(0).total_memory,
            "peak_allocated_bytes": torch.xpu.max_memory_allocated(0),
            "peak_reserved_bytes": torch.xpu.max_memory_reserved(0),
        }
    else:
        raise RuntimeError("the local DPO contract requires an Intel XPU")

    run_manifest = {
        "schema_version": "ai-data-extraction/agent-dpo-run/v1",
        "status": "completed",
        "mode": "one_step_canary" if canary else "bounded_full_release",
        "base_model": {
            "path": str(args.model_dir),
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "config_sha256": sha256_file(args.model_dir / "config.json"),
        },
        "reference_adapter": {
            "path": str(args.reference_adapter),
            "config_sha256": sha256_file(args.reference_adapter / "adapter_config.json"),
            "weights_sha256": sha256_file(
                args.reference_adapter / "adapter_model.safetensors"
            ),
            "in_memory_parameter_sha256": reference_hash,
            "unchanged": True,
        },
        "dataset": preflight["release"],
        "runtime": {
            "versions": package_versions(
                [
                    "torch",
                    "transformers",
                    "trl",
                    "peft",
                    "accelerate",
                    "datasets",
                    "bitsandbytes",
                    "triton-xpu",
                    "unsloth",
                    "unsloth_zoo",
                ]
            ),
            "container_image_id": os.environ.get("DPO_CONTAINER_IMAGE_ID"),
            "accelerator": accelerator,
            **backend_report,
        },
        "training": {
            "method": "robust_dpo",
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "beta": args.beta,
            "label_smoothing": args.label_smoothing,
            "max_length": args.max_length,
            "truncation": False,
            "logits_scope": "completion_tail",
            "use_logits_to_keep": True,
            "canary_selection": canary_selection,
            "seed": args.seed,
            "metrics": metrics,
            "global_steps": trainer.state.global_step,
            "log_history": trainer.state.log_history,
        },
        "parameters": parameter_report,
        "adapter": {
            "path": str(args.output_dir),
            "final_parameter_sha256": policy_hash,
            "changed_from_reference": True,
            **adapter_report,
        },
    }
    write_json_atomic(args.run_dir / "run-manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
