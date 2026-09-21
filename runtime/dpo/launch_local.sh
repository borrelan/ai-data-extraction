#!/usr/bin/env bash
set -euo pipefail

DPO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DPO_DATA_ROOT="/data-120/ai-training"
DPO_MODEL_ROOT="/data-120/models"
DPO_RELEASE="${DPO_RELEASE:-${DPO_DATA_ROOT}/datasets/qwen3.5-9b-agent-preference-v1}"
DPO_BASE_MODEL="${DPO_MODEL_ROOT}/Qwen3.5-9B"
DPO_REFERENCE_ADAPTER="${DPO_REFERENCE_ADAPTER:-${DPO_MODEL_ROOT}/adapters/Qwen3.5-9B-agent-sft-curriculum-v1}"
DPO_ADAPTER_ROOT="${DPO_MODEL_ROOT}/adapters"
DPO_ADAPTER_NAME="${DPO_ADAPTER_NAME:-Qwen3.5-9B-agent-dpo-v1}"
DPO_RUN_ROOT="${DPO_DATA_ROOT}/runs"
DPO_RUN_NAME="${DPO_RUN_NAME:-qwen3.5-9b-agent-dpo-v1-seed20260920}"
DPO_CACHE="${DPO_CACHE:-${DPO_DATA_ROOT}/cache/qwen3.5-9b-agent-dpo-v1}"
DPO_CONTAINER_NAME="${DPO_CONTAINER_NAME:-qwen35-9b-agent-dpo-v1}"
DPO_MAX_STEPS="${DPO_MAX_STEPS:--1}"
DPO_MAX_LENGTH="${DPO_MAX_LENGTH:-8192}"
DPO_IMAGE="ai-data-extraction/unsloth-xpu-dpo:trl028"
DPO_IMAGE_ID="sha256:38b247740a8114ab55624cd076f52b39d15b533d0a376f2be3f1efc1710f4c4a"
DPO_BASELINE_UNIT="qwen35-9b-baseline.service"
DPO_BASELINE_MODEL="/data-120/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"
DPO_BASELINE_EXECUTABLE="/home/borrelan/Projects/Personal/llama.cpp-master/build-sycl-bmg/bin/llama-server"

baseline_healthy() {
  curl --fail --silent --max-time 5 http://127.0.0.1:8080/v1/models \
    | grep --fixed-strings --quiet "qwen3.5-9b-baseline"
}

for DPO_REQUIRED in "${DPO_RELEASE}" "${DPO_BASE_MODEL}" "${DPO_REFERENCE_ADAPTER}" "${DPO_REPO_ROOT}/runtime/dpo"; do
  if [[ ! -d "${DPO_REQUIRED}" ]]; then
    echo "required directory is missing: ${DPO_REQUIRED}" >&2
    exit 1
  fi
done
for DPO_REQUIRED_FILE in \
  "${DPO_RELEASE}/manifest.json" \
  "${DPO_REFERENCE_ADAPTER}/adapter_config.json" \
  "${DPO_REFERENCE_ADAPTER}/adapter_model.safetensors"; do
  if [[ ! -f "${DPO_REQUIRED_FILE}" ]]; then
    echo "required file is missing: ${DPO_REQUIRED_FILE}" >&2
    exit 1
  fi
done
if [[ -e "${DPO_ADAPTER_ROOT}/${DPO_ADAPTER_NAME}" ]]; then
  echo "adapter output already exists: ${DPO_ADAPTER_ROOT}/${DPO_ADAPTER_NAME}" >&2
  exit 1
fi
if [[ -e "${DPO_RUN_ROOT}/${DPO_RUN_NAME}" ]]; then
  echo "run output already exists: ${DPO_RUN_ROOT}/${DPO_RUN_NAME}" >&2
  exit 1
fi
if [[ "${DPO_MAX_STEPS}" != "-1" && "${DPO_MAX_STEPS}" != "1" ]]; then
  echo "DPO_MAX_STEPS must be -1 for a full release or 1 for a longest-pair canary" >&2
  exit 1
fi
if [[ ! "${DPO_MAX_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DPO_MAX_LENGTH must be a positive integer" >&2
  exit 1
fi
if [[ ! -c /dev/dri/renderD128 ]]; then
  echo "Intel XPU render device is unavailable" >&2
  exit 1
fi
DPO_ACTUAL_IMAGE_ID="$(docker image inspect "${DPO_IMAGE}" --format '{{.Id}}')"
if [[ "${DPO_ACTUAL_IMAGE_ID}" != "${DPO_IMAGE_ID}" ]]; then
  echo "DPO image identity mismatch: ${DPO_ACTUAL_IMAGE_ID}" >&2
  exit 1
fi
if ! systemctl --user is-active --quiet "${DPO_BASELINE_UNIT}" || ! baseline_healthy; then
  echo "expected baseline service is not healthy: ${DPO_BASELINE_UNIT}" >&2
  exit 1
fi

DPO_SERVER_PID="$(systemctl --user show "${DPO_BASELINE_UNIT}" --property MainPID --value)"
DPO_SERVER_EXECUTABLE="$(readlink -f "/proc/${DPO_SERVER_PID}/exe")"
mapfile -d '' -t DPO_SERVER_ARGV < "/proc/${DPO_SERVER_PID}/cmdline"
if [[ "${DPO_SERVER_EXECUTABLE}" != "${DPO_BASELINE_EXECUTABLE}" ]]; then
  echo "unexpected baseline executable: ${DPO_SERVER_EXECUTABLE}" >&2
  exit 1
fi
if [[ " ${DPO_SERVER_ARGV[*]} " != *" --model ${DPO_BASELINE_MODEL} "* ]]; then
  echo "baseline service does not own the expected model" >&2
  exit 1
fi

set +eu
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
DPO_ONEAPI_STATUS=$?
set -eu
if [[ "${DPO_ONEAPI_STATUS}" -ne 0 ]]; then
  echo "oneAPI runtime initialization failed: ${DPO_ONEAPI_STATUS}" >&2
  exit 1
fi
if ldd "${DPO_BASELINE_EXECUTABLE}" | grep --quiet "not found"; then
  echo "baseline executable has unresolved oneAPI libraries" >&2
  exit 1
fi

mkdir -p \
  "${DPO_ADAPTER_ROOT}" \
  "${DPO_RUN_ROOT}" \
  "${DPO_CACHE}/home" \
  "${DPO_CACHE}/tmp" \
  "${DPO_CACHE}/work"

if [[ "${1-}" == "--preflight-only" ]]; then
  echo "local XPU preference preflight passed"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi

DPO_SERVER_STOPPED=0
restore_baseline() {
  local restore_deadline
  if [[ "${DPO_SERVER_STOPPED}" -ne 1 ]]; then
    return 0
  fi
  if systemctl --user is-active --quiet "${DPO_BASELINE_UNIT}"; then
    if ! baseline_healthy; then
      echo "baseline service is active but unhealthy" >&2
      return 1
    fi
    DPO_SERVER_STOPPED=0
    return 0
  fi
  systemctl --user reset-failed "${DPO_BASELINE_UNIT}" 2>/dev/null || true
  systemd-run --user \
    --unit "${DPO_BASELINE_UNIT%.service}" \
    --property Type=exec \
    --setenv "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
    --setenv "PATH=${PATH}" \
    "${DPO_SERVER_ARGV[@]}"
  restore_deadline=$((SECONDS + 240))
  while (( SECONDS < restore_deadline )); do
    if systemctl --user is-failed --quiet "${DPO_BASELINE_UNIT}"; then
      echo "restored baseline service failed before health check" >&2
      return 1
    fi
    if baseline_healthy; then
      DPO_SERVER_STOPPED=0
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

DPO_SERVER_STOPPED=1
systemctl --user stop "${DPO_BASELINE_UNIT}"
for _ in $(seq 1 60); do
  if ! systemctl --user is-active --quiet "${DPO_BASELINE_UNIT}"; then
    break
  fi
  sleep 1
done
if systemctl --user is-active --quiet "${DPO_BASELINE_UNIT}"; then
  echo "baseline service did not stop within 60 seconds" >&2
  exit 1
fi
if curl --fail --silent --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "port 8080 is still served after stopping the baseline" >&2
  exit 1
fi

docker run --rm \
  --name "${DPO_CONTAINER_NAME}" \
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
  --env PYTHONPATH=/workspace \
  --env XDG_CACHE_HOME=/cache \
  --env TMPDIR=/cache/tmp \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  --env TRITON_CACHE_DIR=/cache/triton \
  --env "DPO_CONTAINER_IMAGE_ID=${DPO_IMAGE_ID}" \
  --mount "type=bind,src=${DPO_BASE_MODEL},dst=/models/Qwen3.5-9B,readonly" \
  --mount "type=bind,src=${DPO_REFERENCE_ADAPTER},dst=/adapters/reference,readonly" \
  --mount "type=bind,src=${DPO_RELEASE},dst=/input,readonly" \
  --mount "type=bind,src=${DPO_ADAPTER_ROOT},dst=/model-output" \
  --mount "type=bind,src=${DPO_RUN_ROOT},dst=/run-output" \
  --mount "type=bind,src=${DPO_CACHE},dst=/cache" \
  --mount "type=bind,src=${DPO_REPO_ROOT},dst=/workspace,readonly" \
  --entrypoint python \
  "${DPO_IMAGE}" \
  /workspace/runtime/dpo/train.py \
  --model-dir /models/Qwen3.5-9B \
  --reference-adapter /adapters/reference \
  --input-dir /input \
  --output-dir "/model-output/${DPO_ADAPTER_NAME}" \
  --run-dir "/run-output/${DPO_RUN_NAME}" \
  --max-length "${DPO_MAX_LENGTH}" \
  --epochs 1 \
  --learning-rate 0.000005 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --beta 0.1 \
  --label-smoothing 0.05 \
  --seed 20260920 \
  --max-steps "${DPO_MAX_STEPS}"

restore_baseline
trap - EXIT INT TERM
baseline_healthy
echo "local XPU preference run completed and baseline service restored"
