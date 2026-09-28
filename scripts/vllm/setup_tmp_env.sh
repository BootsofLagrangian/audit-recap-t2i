#!/usr/bin/env bash
# Create a vLLM environment under /tmp using uv, keeping the venv and compile caches off shared storage.
set -euo pipefail

VENV_DIR="${1:-/tmp/vllm-env-qwen35}"
export TMPDIR="${TMPDIR:-/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch-home}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch-extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export PYTHONDONTWRITEBYTECODE=1

uv venv "${VENV_DIR}" --python python3.11
uv pip install --python "${VENV_DIR}/bin/python" "vllm==0.20.1" openai pillow aiohttp
bash "$(dirname "$0")/install_deepgemm.sh" "${VENV_DIR}"
"${VENV_DIR}/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3.5-35B-A3B-FP8", max_workers=8)
PY
"${VENV_DIR}/bin/python" -c "import vllm; print(f'vLLM {vllm.__version__} ready at ${VENV_DIR}')"
