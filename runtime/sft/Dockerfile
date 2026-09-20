FROM docker.io/intel/pytorch@sha256:7366f264bd7aaa7f0df7d1bfba43be7c84c126df21ba448ae5445a34c2def4f9

ARG UNSLOTH_COMMIT=768d644036a948441ba80b626c9e747c1fecfbae

LABEL org.opencontainers.image.title="ai-data-extraction local Intel XPU QLoRA runtime"
LABEL org.opencontainers.image.source="https://github.com/unslothai/unsloth"
LABEL org.opencontainers.image.revision="${UNSLOTH_COMMIT}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

RUN dpkg-query -W libze1 libze-intel-gpu1 intel-gpu-compute

RUN python -m pip install \
      "unsloth[intel-gpu-torch2120] @ https://github.com/unslothai/unsloth/archive/${UNSLOTH_COMMIT}.zip" \
    && python -m pip check

# Generic Triton and triton-xpu publish the same Python module. Restore the
# hash-pinned XPU wheel last so it owns the shared native library.
RUN python -m pip install --force-reinstall --no-deps --no-cache-dir \
      "https://download.pytorch.org/whl/triton_xpu-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=a31c058c5c2e78ebe490a2e69f2f50caec6b1307ac096e944f116fdc06819d9a" \
    && python -m pip check \
    && python - <<'PY'
import triton

assert triton.__version__ == "3.7.1", triton.__version__
PY

WORKDIR /workspace

ENTRYPOINT ["python"]
