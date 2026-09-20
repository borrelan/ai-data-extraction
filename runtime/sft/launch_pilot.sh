#!/usr/bin/env bash
set -euo pipefail

SFT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SFT_DATA_ROOT="/data-120/ai-training"
SFT_MODEL_ROOT="/data-120/models"
SFT_RELEASE="${SFT_DATA_ROOT}/datasets/qwen3.5-9b-agent-sft-pilot-v2"
SFT_BASE_MODEL="${SFT_MODEL_ROOT}/Qwen3.5-9B"
SFT_ADAPTER_ROOT="${SFT_MODEL_ROOT}/adapters"
SFT_ADAPTER_NAME="Qwen3.5-9B-agent-sft-pilot-v2"
SFT_RUN_ROOT="${SFT_DATA_ROOT}/runs"
SFT_RUN_NAME="qwen3.5-9b-agent-sft-pilot-v2-seed20260920"
SFT_CACHE="${SFT_DATA_ROOT}/cache/qwen3.5-9b-agent-sft-pilot-v2"
SFT_IMAGE="ai-data-extraction/unsloth-xpu:torch2120-768d644"
SFT_IMAGE_ID="sha256:5f0361b760c438c41ea17416c01ebcf3ed4d57b86d4cf9618f83a3db2121093c"
SFT_BASELINE_UNIT="qwen35-9b-baseline.service"
SFT_BASELINE_MODEL="/data-120/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"
SFT_BASELINE_EXECUTABLE="/home/borrelan/Projects/Personal/llama.cpp-master/build-sycl-bmg/bin/llama-server"

baseline_healthy() {
  curl --fail --silent --max-time 5 http://127.0.0.1:8080/v1/models \
    | grep --fixed-strings --quiet "qwen3.5-9b-baseline"
}

for SFT_REQUIRED in "${SFT_RELEASE}" "${SFT_BASE_MODEL}" "${SFT_REPO_ROOT}/runtime/sft"; do
  if [[ ! -d "${SFT_REQUIRED}" ]]; then
    echo "required directory is missing: ${SFT_REQUIRED}" >&2
    exit 1
  fi
done
if [[ -e "${SFT_ADAPTER_ROOT}/${SFT_ADAPTER_NAME}" ]]; then
  echo "adapter output already exists: ${SFT_ADAPTER_ROOT}/${SFT_ADAPTER_NAME}" >&2
  exit 1
fi
if [[ -e "${SFT_RUN_ROOT}/${SFT_RUN_NAME}" ]]; then
  echo "run output already exists: ${SFT_RUN_ROOT}/${SFT_RUN_NAME}" >&2
  exit 1
fi
if [[ ! -c /dev/dri/renderD128 ]]; then
  echo "Intel XPU render device is unavailable" >&2
  exit 1
fi
SFT_ACTUAL_IMAGE_ID="$(docker image inspect "${SFT_IMAGE}" --format '{{.Id}}')"
if [[ "${SFT_ACTUAL_IMAGE_ID}" != "${SFT_IMAGE_ID}" ]]; then
  echo "training image identity mismatch: ${SFT_ACTUAL_IMAGE_ID}" >&2
  exit 1
fi
if ! systemctl --user is-active --quiet "${SFT_BASELINE_UNIT}" || ! baseline_healthy; then
  echo "expected baseline service is not healthy: ${SFT_BASELINE_UNIT}" >&2
  exit 1
fi

SFT_SERVER_PID="$(systemctl --user show "${SFT_BASELINE_UNIT}" --property MainPID --value)"
SFT_SERVER_EXECUTABLE="$(readlink -f "/proc/${SFT_SERVER_PID}/exe")"
mapfile -d '' -t SFT_SERVER_ARGV < "/proc/${SFT_SERVER_PID}/cmdline"
if [[ "${SFT_SERVER_EXECUTABLE}" != "${SFT_BASELINE_EXECUTABLE}" ]]; then
  echo "unexpected baseline executable: ${SFT_SERVER_EXECUTABLE}" >&2
  exit 1
fi
if [[ " ${SFT_SERVER_ARGV[*]} " != *" --model ${SFT_BASELINE_MODEL} "* ]]; then
  echo "baseline service does not own the expected model" >&2
  exit 1
fi

set +eu
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
SFT_ONEAPI_STATUS=$?
set -eu
if [[ "${SFT_ONEAPI_STATUS}" -ne 0 ]]; then
  echo "oneAPI runtime initialization failed: ${SFT_ONEAPI_STATUS}" >&2
  exit 1
fi
if ldd "${SFT_BASELINE_EXECUTABLE}" | grep --quiet "not found"; then
  echo "baseline executable has unresolved oneAPI libraries" >&2
  exit 1
fi

mkdir -p \
  "${SFT_ADAPTER_ROOT}" \
  "${SFT_RUN_ROOT}" \
  "${SFT_CACHE}/home" \
  "${SFT_CACHE}/work"

if [[ "${1-}" == "--preflight-only" ]]; then
  echo "local XPU pilot preflight passed"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi

SFT_SERVER_STOPPED=0
restore_baseline() {
  local restore_deadline
  if [[ "${SFT_SERVER_STOPPED}" -ne 1 ]]; then
    return 0
  fi
  if systemctl --user is-active --quiet "${SFT_BASELINE_UNIT}"; then
    if ! baseline_healthy; then
      echo "baseline service is active but unhealthy" >&2
      return 1
    fi
    SFT_SERVER_STOPPED=0
    return 0
  fi
  systemctl --user reset-failed "${SFT_BASELINE_UNIT}" 2>/dev/null || true
  systemd-run --user \
    --unit "${SFT_BASELINE_UNIT%.service}" \
    --property Type=exec \
    --setenv "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
    --setenv "PATH=${PATH}" \
    "${SFT_SERVER_ARGV[@]}"
  restore_deadline=$((SECONDS + 240))
  while (( SECONDS < restore_deadline )); do
    if systemctl --user is-failed --quiet "${SFT_BASELINE_UNIT}"; then
      echo "restored baseline service failed before health check" >&2
      return 1
    fi
    if baseline_healthy; then
      SFT_SERVER_STOPPED=0
      return 0
    fi
    sleep 2
  done
  echo "restored baseline service did not become healthy" >&2
  return 1
}

finish() {
  local status="$?"
  trap - EXIT INT TERM
  if ! restore_baseline; then
    status=97
  fi
  exit "${status}"
}
trap finish EXIT INT TERM

SFT_SERVER_STOPPED=1
systemctl --user stop "${SFT_BASELINE_UNIT}"
for _ in $(seq 1 60); do
  if ! systemctl --user is-active --quiet "${SFT_BASELINE_UNIT}"; then
    break
  fi
  sleep 1
done
if systemctl --user is-active --quiet "${SFT_BASELINE_UNIT}"; then
  echo "baseline service did not stop within 60 seconds" >&2
  exit 1
fi
if curl --fail --silent --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "port 8080 is still served after stopping the baseline" >&2
  exit 1
fi

docker run --rm \
  --name qwen35-9b-sft-pilot-v2 \
  --user "$(id -u):$(id -g)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --device /dev/dri \
  --ipc=host \
  --network=none \
  --workdir /cache/work \
  --env HOME=/cache/home \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env XDG_CACHE_HOME=/cache \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  --env TRITON_CACHE_DIR=/cache/triton \
  --env "SFT_CONTAINER_IMAGE_ID=${SFT_IMAGE_ID}" \
  --mount "type=bind,src=${SFT_BASE_MODEL},dst=/models/Qwen3.5-9B,readonly" \
  --mount "type=bind,src=${SFT_RELEASE},dst=/input,readonly" \
  --mount "type=bind,src=${SFT_ADAPTER_ROOT},dst=/model-output" \
  --mount "type=bind,src=${SFT_RUN_ROOT},dst=/run-output" \
  --mount "type=bind,src=${SFT_CACHE},dst=/cache" \
  --mount "type=bind,src=${SFT_REPO_ROOT}/runtime/sft,dst=/workspace/runtime/sft,readonly" \
  --entrypoint python \
  "${SFT_IMAGE}" \
  /workspace/runtime/sft/train.py \
  --backend unsloth \
  --model-dir /models/Qwen3.5-9B \
  --input-dir /input \
  --output-dir "/model-output/${SFT_ADAPTER_NAME}" \
  --run-dir "/run-output/${SFT_RUN_NAME}" \
  --max-length 8192 \
  --epochs 1 \
  --learning-rate 0.0001 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --seed 20260920

restore_baseline
trap - EXIT INT TERM
baseline_healthy
echo "local XPU pilot completed and baseline service restored"
