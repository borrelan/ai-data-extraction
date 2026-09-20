# Qwen3.5-9B local SFT runtime

This directory owns the local Intel Arc Pro B70 training path for the bounded
Qwen3.5-9B QLoRA pilot. RunPod is not a fallback for this lane; cloud execution
is reserved for a separately specified 27B experiment.

`runtime_contract.json` binds the exact model, dataset, image ID, package
versions, training parameters, output owners, and completed memory canaries.
`Dockerfile` rebuilds the local Intel image from a digest-pinned Intel PyTorch
base and the exact Unsloth source revision. Do not substitute an NVIDIA image,
a mutable tag, BF16 full-weight training, or a different model checkpoint.

## Storage and mount contract

| Host owner | Container path | Mode | Purpose |
| --- | --- | --- | --- |
| `/data-120/models/Qwen3.5-9B` | `/models/Qwen3.5-9B` | read-only | Exact base checkpoint |
| `/data-120/ai-training/datasets/qwen3.5-9b-agent-sft-pilot-v2` | `/input` | read-only | Immutable 732-row release |
| `/data-120/models/adapters` | `/model-output` | read-write | Standard PEFT adapter |
| `/data-120/ai-training/runs` | `/run-output` | read-write | Checkpoints, metrics, and run manifest |
| `/data-120/ai-training/cache/qwen3.5-9b-agent-sft-pilot-v2` | `/cache` | read-write | Hugging Face and compiler caches |

No model checkpoint or training cache belongs in the repository or local root
filesystem. The launcher verifies the local image ID and disables container
network access.

## Run the pilot

The immutable release contains 646 training and 86 validation examples. Every
row passed exact Qwen3.5 chat-template prefix validation at 8,192 tokens without
truncation. Training uses final-assistant-only loss, rank-16 NF4 QLoRA, one
epoch, batch size 1, gradient accumulation 8, and seed 20260920.

The baseline `qwen3.5-9b-baseline` llama.cpp service normally owns the B70.
`launch_pilot.sh` verifies its executable and model, stops only that service,
runs training, and restores the exact captured command through a user systemd
unit on success, failure, or interruption.

```bash
runtime/sft/launch_pilot.sh
```

Durable outputs are:

- adapter: `/data-120/models/adapters/Qwen3.5-9B-agent-sft-pilot-v2`;
- run evidence: `/data-120/ai-training/runs/qwen3.5-9b-agent-sft-pilot-v2-seed20260920`.

The trainer revalidates all input hashes, saves checkpoints every 10 optimizer
steps, rejects non-finite metrics or tensors, and reloads the final adapter
through PEFT before marking the run complete.

## Build the proven image

Build only when the image ID in `runtime_contract.json` is intentionally being
requalified:

```bash
docker build -t ai-data-extraction/unsloth-xpu:torch2120-768d644 runtime/sft
```

After a rebuild, the new image is not qualified merely because it builds. Run
the short and longest-sequence optimizer canaries, record their artifact hashes
and peak XPU memory, then deliberately update the contract image ID.
