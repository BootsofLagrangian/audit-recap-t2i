#!/usr/bin/env bash
# Start/stop a 2-GPU Qwen3.5-35B-A3B-FP8 vLLM server from a /tmp environment.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VLLM_VENV:-/tmp/vllm-env-qwen35}"
CONFIG="${VLLM_CONFIG:-${PROJECT_ROOT}/configs/recap/vllm_serve_qwen35_dp2_tmp.yaml}"
LOG="${VLLM_LOG:-/tmp/vllm_qwen35_dp2.log}"
PID_FILE="${VLLM_PID_FILE:-/tmp/vllm_qwen35_dp2.pid}"
PORT="${VLLM_PORT:-8000}"

export TMPDIR="${TMPDIR:-/tmp}"
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
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1

VLLM_BIN="${VENV_DIR}/bin/vllm"

status() {
  if [[ -f "${PID_FILE}" ]] && ps -p "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "running pid=$(cat "${PID_FILE}")"
  else
    echo "stopped"
  fi
  curl -fsS "http://localhost:${PORT}/v1/models" 2>/dev/null || true
}

start() {
  if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: vllm binary not found at ${VLLM_BIN}; run scripts/vllm/setup_tmp_env.sh first" >&2
    exit 1
  fi
  if [[ -f "${PID_FILE}" ]] && ps -p "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "already running pid=$(cat "${PID_FILE}")"
    exit 0
  fi
  rm -f /dev/shm/vllm* 2>/dev/null || true
  setsid "${VLLM_BIN}" serve --config "${CONFIG}" > "${LOG}" 2>&1 < /dev/null &
  echo "$!" > "${PID_FILE}"
  echo "started pid=$(cat "${PID_FILE}") log=${LOG}"
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      kill "${pid}" 2>/dev/null || true
      sleep 5
      ps -p "${pid}" >/dev/null 2>&1 && kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  pkill -f "vllm serve.*${PORT}" 2>/dev/null || true
  rm -f /dev/shm/vllm* 2>/dev/null || true
  echo "stopped"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
