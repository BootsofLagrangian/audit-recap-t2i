#!/usr/bin/env bash
# Launch Qwen3.5-397B-A17B-FP8 as a single TP=8 vLLM server.
#
# Usage:
#   bash scripts/vllm/serve_397b.sh start
#   bash scripts/vllm/serve_397b.sh stop
#   bash scripts/vllm/serve_397b.sh status
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VLLM_VENV:-/tmp/vllm-env-qwen35}"
VLLM_BIN="${VENV_DIR}/bin/vllm"
CONFIG="${VLLM_CONFIG:-${PROJECT_ROOT}/configs/recap/vllm_serve_397b_fp8.yaml}"
PORT="${VLLM_PORT:-8000}"
LOG="${VLLM_LOG:-/tmp/vllm_397b.log}"

export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch-home}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch-extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

# Hopper FP8/FlashInfer toggles. vLLM 0.19.1 selects FlashAttention3 for the
# main attention path by default on this model, while still using FP8 KV cache.
# Override VLLM_ATTENTION_BACKEND=FLASHINFER at launch time only if stress tests
# show the default backend is the bottleneck or fails with FP8 KV workloads.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-1}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-1073741824}"

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vllm binary not found at ${VLLM_BIN}" >&2
  exit 1
fi

status() {
  if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM 397B :${PORT} ready"
    curl -fsS "http://localhost:${PORT}/v1/models"
  else
    echo "vLLM 397B :${PORT} not ready"
    return 1
  fi
}

# stop() also force-kills every compute process that nvidia-smi reports on this
# host; run it only on a node dedicated to this server.
stop() {
  pkill -f "vllm serve.*${PORT}" 2>/dev/null || true
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null || true
  rm -f /dev/shm/vllm* 2>/dev/null || true
  echo "stopped vLLM 397B on :${PORT}"
}

start() {
  mkdir -p "$(dirname "${LOG}")" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  echo "starting vLLM 397B"
  echo "  config: ${CONFIG}"
  echo "  log:    ${LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    setsid "${VLLM_BIN}" serve --config "${CONFIG}" > "${LOG}" 2>&1 < /dev/null &
  echo "  pid: $!"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
