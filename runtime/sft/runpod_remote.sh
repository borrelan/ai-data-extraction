#!/usr/bin/env bash
set -euo pipefail

echo "disabled: the Qwen3.5-9B pilot is local-B70-only; RunPod is reserved for a later 27B lane" >&2
exit 64

SFT_WORKSPACE="/workspace"
SFT_PILOT_ROOT="${SFT_WORKSPACE}/pilot"
SFT_MODEL_DIR="${SFT_WORKSPACE}/models/Qwen3.5-9B"
SFT_ADAPTER_DIR="${SFT_WORKSPACE}/models/adapters/Qwen3.5-9B-agent-sft-pilot-v2"
SFT_RUN_DIR="${SFT_WORKSPACE}/runs/qwen3.5-9b-agent-sft-pilot-v2-seed20260920"
SFT_RETURN_ARCHIVE="${SFT_WORKSPACE}/qwen3.5-9b-agent-sft-pilot-v2-return.tar.gz"
SFT_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"

if [[ ! -f "${SFT_PILOT_ROOT}/bundle-manifest.json" ]]; then
  echo "pilot bundle is not extracted under ${SFT_PILOT_ROOT}" >&2
  exit 1
fi
nvidia-smi -L
mkdir -p "${SFT_WORKSPACE}/models" "${SFT_WORKSPACE}/models/adapters" \
  "${SFT_WORKSPACE}/runs" "${SFT_WORKSPACE}/cache/huggingface"

export SFT_MODEL_DIR SFT_REVISION
export HF_HOME="${SFT_WORKSPACE}/cache/huggingface"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    revision=os.environ["SFT_REVISION"],
    local_dir=os.environ["SFT_MODEL_DIR"],
)
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python "${SFT_PILOT_ROOT}/runtime/sft/smoke.py" \
  --model-dir "${SFT_MODEL_DIR}" \
  --acquisition-manifest "${SFT_PILOT_ROOT}/artifacts/acquisition-manifest-20260919.json" \
  --package-version-policy record \
  --preload-package unsloth \
  --output "${SFT_WORKSPACE}/sft-runtime-smoke.json"

python "${SFT_PILOT_ROOT}/runtime/sft/train.py" \
  --backend unsloth \
  --model-dir "${SFT_MODEL_DIR}" \
  --input-dir "${SFT_PILOT_ROOT}/input" \
  --output-dir "${SFT_ADAPTER_DIR}" \
  --run-dir "${SFT_RUN_DIR}" \
  --max-length 8192 \
  --epochs 1 \
  --learning-rate 0.0001 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --seed 20260920

cp "${SFT_WORKSPACE}/sft-runtime-smoke.json" "${SFT_RUN_DIR}/runtime-smoke.json"
tar -czf "${SFT_RETURN_ARCHIVE}" \
  -C "${SFT_WORKSPACE}" \
  models/adapters/Qwen3.5-9B-agent-sft-pilot-v2 \
  runs/qwen3.5-9b-agent-sft-pilot-v2-seed20260920
sha256sum "${SFT_RETURN_ARCHIVE}"
