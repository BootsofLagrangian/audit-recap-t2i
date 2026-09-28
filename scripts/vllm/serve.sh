#!/usr/bin/env bash
# Usage:
#   bash scripts/vllm/serve.sh                                          # defaults
#   bash scripts/vllm/serve.sh --config configs/recap/vllm_serve.yaml   # custom config
#   bash scripts/vllm/serve.sh --config ... --data-parallel-size 4      # CLI overrides win
set -euo pipefail

VENV_DIR="${VLLM_VENV:-/tmp/vllm-env-qwen35}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/recap/vllm_serve.yaml"

# Activate isolated venv
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
else
  echo "WARNING: venv not found at ${VENV_DIR}, using current env"
fi

# Env vars for throughput
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# If no --config passed, inject default
HAS_CONFIG=false
for arg in "$@"; do [[ "$arg" == "--config" ]] && HAS_CONFIG=true; done

VLLM_BIN="${VENV_DIR}/bin/vllm"
if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vllm binary not found at ${VLLM_BIN}"
  exit 1
fi

if [[ "${HAS_CONFIG}" == "false" ]]; then
  exec "${VLLM_BIN}" serve --config "${DEFAULT_CONFIG}" "$@"
else
  exec "${VLLM_BIN}" serve "$@"
fi
