#!/usr/bin/env bash
# Reproduce Qwen397 judge metrics for the 2026-05-02 DataComp naive-Qwen35
# policy ablation. Assumes a Qwen397 vLLM server is already running.
#
# The Gemma Judge answers the same VQA request file built here from the Qwen397
# claims (no Gemma claim extraction), with the Gemma server running:
#   uv run python scripts/run_cbu_vqa_requests.py --input "${VQA_REQ}" \
#     --output <gemma responses>.jsonl --model google/gemma-4-31B-it --concurrency 512 \
#     --max-tokens 2048 --image-mode file --structured-json --no-evidence --resume --resume-ok-only
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch-home}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch-extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"

RUN_ROOT="${RUN_ROOT:-artifacts/recap-ed/datacomp-naive-qwen35-baseline-2026-05-02}"
CBU_ROOT="${CBU_ROOT:-artifacts/cbu/datacomp-naive-qwen35-baseline-2026-05-02}"
GROUND_ROOT="${GROUND_ROOT:-artifacts/grounded-cbu/datacomp-naive-qwen35-baseline-2026-05-02}"
VQA_ROOT="${VQA_ROOT:-artifacts/vqa-cbu/datacomp-naive-qwen35-baseline-2026-05-02}"
TABLE_ROOT="${TABLE_ROOT:-${RUN_ROOT}/qwen397_metric_tables}"

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
URLS="${URLS:-http://localhost:8000}"
CBU_CONCURRENCY="${CBU_CONCURRENCY:-424}"
GROUND_CONCURRENCY="${GROUND_CONCURRENCY:-408}"
VQA_CONCURRENCY="${VQA_CONCURRENCY:-420}"
CBU_MAX_TOKENS="${CBU_MAX_TOKENS:-4096}"
GROUND_MAX_TOKENS="${GROUND_MAX_TOKENS:-4096}"
VQA_MAX_TOKENS="${VQA_MAX_TOKENS:-2048}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"

CAPTIONS="${RUN_ROOT}/naive_qwen35_datacomp.jsonl"
CBU_REQ="${CBU_ROOT}/claimed_cbu_v2_naive_qwen35_datacomp_b64.requests.jsonl"
CBU_RESP="${CBU_ROOT}/claimed_cbu_v2_naive_qwen35_datacomp_b64.responses.qwen397_c${CBU_CONCURRENCY}_mt${CBU_MAX_TOKENS}.jsonl"
CBU_SUMMARY="${CBU_RESP%.jsonl}.summary.json"
GROUND_REQ="${GROUND_ROOT}/grounded_verify_v2_naive_qwen35_datacomp_b64.requests.qwen397_local_file.jsonl"
GROUND_RESP="${GROUND_ROOT}/grounded_verify_v2_naive_qwen35_datacomp_b64.responses.qwen397_local_c${GROUND_CONCURRENCY}_file_mt${GROUND_MAX_TOKENS}.jsonl"
GROUND_SUMMARY="${GROUND_RESP%.jsonl}.summary.json"
VQA_REQ="${VQA_ROOT}/cbu_vqa_naive_qwen35_datacomp_b64.requests.qwen397_local_file.jsonl"
VQA_RESP="${VQA_ROOT}/cbu_vqa_naive_qwen35_datacomp_b64.responses.qwen397_local_c${VQA_CONCURRENCY}_file_mt${VQA_MAX_TOKENS}_compact.jsonl"
VQA_SUMMARY="${VQA_RESP%.jsonl}.summary.json"

mkdir -p "${CBU_ROOT}" "${GROUND_ROOT}" "${VQA_ROOT}" "${TABLE_ROOT}"

uv run python scripts/run_text_json_requests.py \
  --input "${CBU_REQ}" \
  --output "${CBU_RESP}" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${CBU_CONCURRENCY}" \
  --max-tokens "${CBU_MAX_TOKENS}" \
  --timeout-sec "${TIMEOUT_SEC}" \
  --structured-json \
  --resume \
  --resume-ok-only

uv run python scripts/summarize_cbu_responses.py \
  --latest-by-request \
  --mode claimed \
  --input "${CBU_RESP}" \
  --output "${CBU_SUMMARY}"

uv run python scripts/build_grounded_cbu_verify_requests.py \
  --claimed-responses "${CBU_RESP}" \
  --source-jsonl "${CAPTIONS}" \
  --output "${GROUND_REQ}" \
  --require-local-image

uv run python scripts/run_grounded_cbu_verify_requests.py \
  --input "${GROUND_REQ}" \
  --output "${GROUND_RESP}" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${GROUND_CONCURRENCY}" \
  --max-tokens "${GROUND_MAX_TOKENS}" \
  --timeout-sec "${TIMEOUT_SEC}" \
  --image-mode file \
  --structured-json \
  --resume \
  --resume-ok-only

uv run python scripts/summarize_grounded_cbu_verify.py \
  --latest-by-request \
  --input "${GROUND_RESP}" \
  --output "${GROUND_SUMMARY}"

uv run python scripts/build_cbu_vqa_requests.py \
  --input "${GROUND_REQ}" \
  --output "${VQA_REQ}"

uv run python scripts/run_cbu_vqa_requests.py \
  --input "${VQA_REQ}" \
  --output "${VQA_RESP}" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${VQA_CONCURRENCY}" \
  --max-tokens "${VQA_MAX_TOKENS}" \
  --timeout-sec "${TIMEOUT_SEC}" \
  --image-mode file \
  --structured-json \
  --no-evidence \
  --resume \
  --resume-ok-only

uv run python scripts/summarize_cbu_vqa_responses.py \
  --latest-by-request \
  --input "${VQA_RESP}" \
  --output "${VQA_SUMMARY}"

uv run python scripts/export_cbu_metric_tables.py \
  --claimed naive_qwen35_datacomp="${CBU_RESP}" \
  --grounded naive_qwen35_datacomp="${GROUND_RESP}" \
  --output-dir "${TABLE_ROOT}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-2000}" \
  --seed "${BOOTSTRAP_SEED:-0}"

uv run python scripts/export_cbu_vqa_tables.py \
  --summary "${VQA_SUMMARY}" \
  --output-md "${TABLE_ROOT}/cbu_vqa_qwen397_table.md" \
  --output-tex "${TABLE_ROOT}/cbu_vqa_qwen397_table.tex"
