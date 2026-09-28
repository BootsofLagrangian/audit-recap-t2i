#!/usr/bin/env bash
# Setup isolated vLLM venv — separate from project .venv.
# Equivalent to scripts/vllm/setup_tmp_env.sh with progress messages; the venv
# defaults to /tmp so it stays off shared network storage.
# Usage: bash scripts/vllm/setup_env.sh [VENV_DIR]
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

echo "=== Creating vLLM venv at ${VENV_DIR} ==="
uv venv "${VENV_DIR}" --python python3.11

echo "=== Installing vLLM + dependencies ==="
uv pip install --python "${VENV_DIR}/bin/python" "vllm==0.20.1" openai pillow aiohttp
bash "$(dirname "$0")/install_deepgemm.sh" "${VENV_DIR}"

echo "=== Downloading model weights ==="
# FP8 variant: ~35GB, fits single H200 with room to spare
"${VENV_DIR}/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B-FP8', max_workers=8)
print('Model download complete.')
"

echo "=== Verifying ==="
"${VENV_DIR}/bin/python" -c "import vllm; print(f'vLLM {vllm.__version__} ready')"
echo "vLLM env ready at: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
