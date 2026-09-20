#!/usr/bin/env bash
set -euo pipefail

echo "disabled: the Qwen3.5-9B pilot is local-B70-only; RunPod is reserved for a later 27B lane" >&2
exit 64

SFT_RUNPODCTL="/data-120/ai-training/tools/runpodctl/v2.8.0/runpodctl"
SFT_BUNDLE="/data-120/ai-training/cloud-bundles/qwen3.5-9b-agent-sft-pilot-v2.tar.gz"
SFT_IMAGE="docker.io/unsloth/unsloth:core-nightly-2026.09.19@sha256:d081bd90a17ed224e93117f87d0eeba98af49a045f62331da7a3038e5f082d00"
SFT_GPU="NVIDIA A100 80GB PCIe"
SFT_ADAPTER_TARGET="/data-120/models/adapters/Qwen3.5-9B-agent-sft-pilot-v2"
SFT_RUN_TARGET="/data-120/ai-training/runs/qwen3.5-9b-agent-sft-pilot-v2-seed20260920"
SFT_RETURN_ROOT="/data-120/ai-training/cloud-returns"
SFT_LOG_ROOT="/data-120/ai-training/cloud-runs"
SFT_REMOTE_ARCHIVE="/workspace/qwen3.5-9b-agent-sft-pilot-v2-return.tar.gz"
SFT_POD_ID=""
SFT_RETURN_STAGE=""

mkdir -p "${SFT_LOG_ROOT}" "${SFT_RETURN_ROOT}"
SFT_LOG="${SFT_LOG_ROOT}/qwen3.5-9b-agent-sft-pilot-v2-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${SFT_LOG}") 2>&1

cleanup() {
  SFT_STATUS=$?
  if [[ -n "${SFT_POD_ID}" ]]; then
    "${SFT_RUNPODCTL}" pod delete "${SFT_POD_ID}" || true
  fi
  if [[ -n "${SFT_RETURN_STAGE}" && -d "${SFT_RETURN_STAGE}" ]]; then
    find "${SFT_RETURN_STAGE}" -depth -delete
  fi
  exit "${SFT_STATUS}"
}
trap cleanup EXIT INT TERM

for SFT_COMMAND in jq ssh scp tar python3; do
  command -v "${SFT_COMMAND}" >/dev/null || {
    echo "required command is missing: ${SFT_COMMAND}" >&2
    exit 1
  }
done
for SFT_REQUIRED in "${SFT_RUNPODCTL}" "${SFT_BUNDLE}"; do
  [[ -f "${SFT_REQUIRED}" ]] || {
    echo "required file is missing: ${SFT_REQUIRED}" >&2
    exit 1
  }
done
if [[ -e "${SFT_ADAPTER_TARGET}" || -e "${SFT_RUN_TARGET}" ]]; then
  echo "canonical adapter or run target already exists" >&2
  exit 1
fi

if ! "${SFT_RUNPODCTL}" user >/dev/null; then
  echo "RunPod authentication failed; set RUNPOD_API_KEY or configure runpodctl" >&2
  exit 1
fi
"${SFT_RUNPODCTL}" ssh list-keys >/dev/null
SFT_TERMINATE_AT="$(date -u -d '+6 hours' +%Y-%m-%dT%H:%M:%SZ)"
SFT_CREATE_JSON="$("${SFT_RUNPODCTL}" pod create \
  --name qwen35-9b-agent-sft-pilot-v2 \
  --image "${SFT_IMAGE}" \
  --gpu-id "${SFT_GPU}" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --container-disk-in-gb 60 \
  --volume-in-gb 80 \
  --volume-mount-path /workspace \
  --ports 22/tcp \
  --ssh \
  --docker-args "sleep infinity" \
  --min-cuda-version 12.6 \
  --terminate-after "${SFT_TERMINATE_AT}" \
  -o json)"
SFT_POD_ID="$(jq -er '.id // .pod.id // .data.id' <<<"${SFT_CREATE_JSON}")"
echo "created auto-terminating pod ${SFT_POD_ID} on ${SFT_GPU}"

SFT_IP=""
SFT_PORT=""
SFT_KEY=""
for _ in $(seq 1 90); do
  if SFT_SSH_JSON="$("${SFT_RUNPODCTL}" ssh info "${SFT_POD_ID}" -o json 2>/dev/null)"; then
    SFT_IP="$(jq -r '.ip // empty' <<<"${SFT_SSH_JSON}")"
    SFT_PORT="$(jq -r '.port // empty' <<<"${SFT_SSH_JSON}")"
    SFT_KEY="$(jq -r '.ssh_key.path // empty' <<<"${SFT_SSH_JSON}")"
    if [[ -n "${SFT_IP}" && -n "${SFT_PORT}" && -f "${SFT_KEY}" ]]; then
      if timeout 5 bash -c "</dev/tcp/${SFT_IP}/${SFT_PORT}" 2>/dev/null; then
        break
      fi
    fi
  fi
  sleep 10
done
if [[ -z "${SFT_IP}" || -z "${SFT_PORT}" || ! -f "${SFT_KEY}" ]]; then
  echo "pod did not expose usable SSH within 15 minutes" >&2
  exit 1
fi

SFT_SSH_OPTIONS=(
  -i "${SFT_KEY}"
  -p "${SFT_PORT}"
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)
SFT_SCP_OPTIONS=(
  -i "${SFT_KEY}"
  -P "${SFT_PORT}"
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

scp "${SFT_SCP_OPTIONS[@]}" "${SFT_BUNDLE}" "root@${SFT_IP}:/workspace/pilot.tar.gz"
ssh "${SFT_SSH_OPTIONS[@]}" "root@${SFT_IP}" \
  'set -euo pipefail; mkdir -p /workspace/pilot; tar -xzf /workspace/pilot.tar.gz -C /workspace; bash /workspace/pilot/runtime/sft/runpod_remote.sh'
SFT_RETURN_ARCHIVE="${SFT_RETURN_ROOT}/qwen3.5-9b-agent-sft-pilot-v2-return-${SFT_POD_ID}.tar.gz"
scp "${SFT_SCP_OPTIONS[@]}" "root@${SFT_IP}:${SFT_REMOTE_ARCHIVE}" "${SFT_RETURN_ARCHIVE}"

SFT_RETURN_STAGE="$(mktemp -d "${SFT_RETURN_ROOT}/qwen3.5-9b-return.XXXXXX")"
tar -xzf "${SFT_RETURN_ARCHIVE}" -C "${SFT_RETURN_STAGE}"
SFT_ADAPTER_STAGE="${SFT_RETURN_STAGE}/models/adapters/Qwen3.5-9B-agent-sft-pilot-v2"
SFT_RUN_STAGE="${SFT_RETURN_STAGE}/runs/qwen3.5-9b-agent-sft-pilot-v2-seed20260920"
export SFT_ADAPTER_STAGE SFT_RUN_STAGE
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

adapter = Path(os.environ["SFT_ADAPTER_STAGE"])
run = Path(os.environ["SFT_RUN_STAGE"])
manifest = json.loads((run / "run-manifest.json").read_text())
if manifest.get("status") != "completed":
    raise SystemExit("returned training run is not complete")
for name, field in (
    ("adapter_config.json", "config_sha256"),
    ("adapter_model.safetensors", "weights_sha256"),
):
    path = adapter / name
    if not path.is_file() or digest(path) != manifest["adapter"][field]:
        raise SystemExit(f"returned adapter binding failed: {name}")
if not (run / "runtime-smoke.json").is_file() or not (run / "dataset-preflight.json").is_file():
    raise SystemExit("returned run is missing smoke or preflight evidence")
print(json.dumps({"status": "verified", "global_steps": manifest["training"]["global_steps"]}))
PY

mkdir -p "$(dirname "${SFT_ADAPTER_TARGET}")" "$(dirname "${SFT_RUN_TARGET}")"
mv "${SFT_ADAPTER_STAGE}" "${SFT_ADAPTER_TARGET}"
mv "${SFT_RUN_STAGE}" "${SFT_RUN_TARGET}"
echo "adapter: ${SFT_ADAPTER_TARGET}"
echo "run: ${SFT_RUN_TARGET}"
echo "orchestrator log: ${SFT_LOG}"

"${SFT_RUNPODCTL}" pod delete "${SFT_POD_ID}"
SFT_POD_ID=""
