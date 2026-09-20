FROM ai-data-extraction/unsloth-xpu:torch2120-768d644

ARG TRL_VERSION=0.28.0

LABEL org.opencontainers.image.title="ai-data-extraction local Intel XPU preference runtime"
LABEL org.opencontainers.image.version="${TRL_VERSION}"

COPY trl-transformers5.patch /tmp/trl-transformers5.patch
COPY requirements.lock /tmp/dpo-requirements.lock

RUN python -m pip install --no-deps --require-hashes -r /tmp/dpo-requirements.lock \
    && cd /opt/venv/lib/python3.12/site-packages/trl \
    && patch -p0 < /tmp/trl-transformers5.patch \
    && rm /tmp/trl-transformers5.patch /tmp/dpo-requirements.lock \
    && python -m pip check \
    && python - <<'PY'
from trl import DPOConfig, DPOTrainer

assert DPOConfig is not None
assert DPOTrainer is not None
PY
