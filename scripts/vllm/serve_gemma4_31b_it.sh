#!/usr/bin/env bash
# Launch google/gemma-4-31B-it as DP=8 vLLM server for cross-family audits.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VLLM_VENV:-/tmp/vllm-env-qwen35}"
VLLM_BIN="${VENV_DIR}/bin/vllm"
CONFIG="${VLLM_CONFIG:-${PROJECT_ROOT}/configs/recap/vllm_serve_gemma4_31b_it.yaml}"
PORT="${VLLM_PORT:-8000}"
LOG="${VLLM_LOG:-/tmp/vllm_gemma4_31b_it.log}"
PID_FILE="${VLLM_PID_FILE:-/tmp/vllm_gemma4_31b_it.pid}"

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

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vllm binary not found at ${VLLM_BIN}" >&2
  exit 1
fi

status() {
  if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM gemma-4-31B-it :${PORT} ready"
    curl -fsS "http://localhost:${PORT}/v1/models"
  else
    echo "vLLM gemma-4-31B-it :${PORT} not ready"
    return 1
  fi
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && ps -p "${pid}" -o command= 2>/dev/null | grep -q "vllm serve"; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  pgrep -f "vllm serve --config ${CONFIG}" 2>/dev/null | xargs -r kill 2>/dev/null || true
  rm -f /dev/shm/vllm* 2>/dev/null || true
  echo "stopped vLLM gemma-4-31B-it on :${PORT}"
}

start() {
  mkdir -p "$(dirname "${LOG}")" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  echo "starting vLLM gemma-4-31B-it"
  echo "  config: ${CONFIG}"
  echo "  log:    ${LOG}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    setsid "${VLLM_BIN}" serve --config "${CONFIG}" > "${LOG}" 2>&1 < /dev/null &
  echo "$!" > "${PID_FILE}"
  echo "  pid: $!"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
