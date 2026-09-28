#!/usr/bin/env bash
# Install DeepGEMM from a pinned source commit into the vLLM uv environment.
set -euo pipefail

VENV_DIR="${1:-${VLLM_VENV:-/tmp/vllm-env-qwen35}}"
DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-891d57b4db1071624b5c8fa0d1e51cb317fa709f}"
SRC_DIR="${DEEPGEMM_SRC_DIR:-/tmp/DeepGEMM-${DEEPGEMM_COMMIT}}"

export TMPDIR="${TMPDIR:-/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-/tmp/vllm-cache/deep_gemm}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "vLLM venv not found: ${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  rm -rf "${SRC_DIR}"
  git clone --recursive "${DEEPGEMM_REPO}" "${SRC_DIR}"
fi

git -C "${SRC_DIR}" fetch --tags origin "${DEEPGEMM_COMMIT}"
git -C "${SRC_DIR}" checkout --detach "${DEEPGEMM_COMMIT}"
git -C "${SRC_DIR}" submodule update --init --recursive

SITE_PACKAGES="$("${VENV_DIR}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
CUDA_INCLUDE="${SITE_PACKAGES}/nvidia/cu13/include"
CUDA_LIB="${SITE_PACKAGES}/nvidia/cu13/lib"
CUSPARSELT_INCLUDE="${SITE_PACKAGES}/nvidia/cusparselt/include"
CUSPARSELT_LIB="${SITE_PACKAGES}/nvidia/cusparselt/lib"

if [[ -e "${CUDA_LIB}/libnvrtc.so.13" && ! -e "${CUDA_LIB}/libnvrtc.so" ]]; then
  ln -s libnvrtc.so.13 "${CUDA_LIB}/libnvrtc.so"
fi
if [[ -e "${CUDA_LIB}/libcudart.so.13" && ! -e "${CUDA_LIB}/libcudart.so" ]]; then
  ln -s libcudart.so.13 "${CUDA_LIB}/libcudart.so"
fi

export CPATH="${CUDA_INCLUDE}:${CUSPARSELT_INCLUDE}${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_LIB}:${CUSPARSELT_LIB}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_LIB}:${CUSPARSELT_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

uv pip install --python "${VENV_DIR}/bin/python" "${SRC_DIR}" --force-reinstall --no-build-isolation
"${VENV_DIR}/bin/python" - <<'PY'
import deep_gemm
print(f"DeepGEMM installed from pinned source: {deep_gemm.__file__}")
PY
