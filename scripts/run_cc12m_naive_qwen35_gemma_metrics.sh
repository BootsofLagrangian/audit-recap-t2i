#!/usr/bin/env bash
# Reproduce Gemma-judge metrics for the CC12M naive Qwen35 caption baseline.
#
# Assumes a Gemma judge vLLM server is already running, for example:
#   CUDA_VISIBLE_DEVICES=0,1 \
#   VLLM_CONFIG=configs/recap/vllm_serve_gemma4_31b_it_dp2_tmp.yaml \
#   VLLM_LOG=/tmp/vllm_gemma4_31b_it_dp2.log \
#   VLLM_PID_FILE=/tmp/vllm_gemma4_31b_it_dp2.pid \
#   bash scripts/vllm/serve_gemma4_31b_it.sh start
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

RUN_ROOT="${RUN_ROOT:-artifacts/recap-ed/cc12m-naive-qwen35-baseline-2026-05-01}"
CBU_ROOT="${CBU_ROOT:-artifacts/cbu/cc12m-naive-qwen35-baseline-2026-05-01}"
GROUND_ROOT="${GROUND_ROOT:-artifacts/grounded-cbu/cc12m-naive-qwen35-baseline-2026-05-01}"
VQA_ROOT="${VQA_ROOT:-artifacts/vqa-cbu/cc12m-naive-qwen35-baseline-2026-05-01}"
LONGCLIP_ROOT="${LONGCLIP_ROOT:-artifacts/longclip/cc12m-naive-qwen35-baseline-2026-05-01}"
TABLE_ROOT="${TABLE_ROOT:-${RUN_ROOT}/gemma4_metric_tables}"

MODEL="${MODEL:-google/gemma-4-31B-it}"
URLS="${URLS:-http://localhost:8000}"

CAPTIONS="${RUN_ROOT}/naive_qwen35_cc12m.jsonl"
CBU_REQ="${CBU_ROOT}/claimed_cbu_v2_naive_qwen35_cc12m_b64_4494.requests.jsonl"
CBU_RESP="${CBU_ROOT}/claimed_cbu_v2_naive_qwen35_cc12m_b64_4494.responses.gemma4_31b_it_c128_mt1024.jsonl"
CBU_SUMMARY="${CBU_RESP%.jsonl}.summary.json"
GROUND_REQ="${GROUND_ROOT}/grounded_verify_v2_naive_qwen35_cc12m_b64_4494.requests.gemma4.jsonl"
VQA_REQ="${VQA_ROOT}/cbu_vqa_naive_qwen35_cc12m_b64_4494.requests.gemma4_image_local.jsonl"
VQA_RESP="${VQA_ROOT}/cbu_vqa_naive_qwen35_cc12m_b64_4494.responses.gemma4_31b_it_c64_file_mt2048.jsonl"
VQA_SUMMARY="${VQA_RESP%.jsonl}.summary.json"

mkdir -p "${CBU_ROOT}" "${GROUND_ROOT}" "${VQA_ROOT}" "${LONGCLIP_ROOT}" "${TABLE_ROOT}"

uv run python scripts/build_caption_cbu_requests.py \
  --input "${CAPTIONS}" \
  --output "${CBU_REQ}" \
  --surface naive_qwen35_cc12m \
  --id-field caption_id \
  --token-budget 64

uv run python scripts/run_text_json_requests.py \
  --input "${CBU_REQ}" \
  --output "${CBU_RESP}" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${CBU_CONCURRENCY:-128}" \
  --max-tokens "${CBU_MAX_TOKENS:-1024}" \
  --timeout-sec "${CBU_TIMEOUT_SEC:-1800}" \
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

uv run python scripts/build_cbu_vqa_requests.py \
  --input "${GROUND_REQ}" \
  --output "${VQA_REQ}"

uv run python scripts/run_cbu_vqa_requests.py \
  --input "${VQA_REQ}" \
  --output "${VQA_RESP}" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${VQA_CONCURRENCY:-256}" \
  --max-tokens "${VQA_MAX_TOKENS:-2048}" \
  --timeout-sec "${VQA_TIMEOUT_SEC:-1800}" \
  --image-mode file \
  --structured-json \
  --no-evidence \
  --resume \
  --resume-ok-only

uv run python scripts/summarize_cbu_vqa_responses.py \
  --latest-by-request \
  --input "${VQA_RESP}" \
  --output "${VQA_SUMMARY}"

uv run python scripts/compute_longclip_retrieval_margin.py \
  --surface naive_qwen35_cc12m="${CAPTIONS}" \
  --output-dir "${LONGCLIP_ROOT}" \
  --model "${LONGCLIP_MODEL:-zer0int/LongCLIP-GmP-ViT-L-14}" \
  --batch-size "${LONGCLIP_BATCH_SIZE:-64}" \
  --retrieval-block-size 512 \
  --max-length 248 \
  --device "${LONGCLIP_DEVICE:-cuda}" \
  --dtype "${LONGCLIP_DTYPE:-float16}" \
  --bootstrap-reps "${LONGCLIP_BOOTSTRAP_REPS:-1000}" \
  --trust-remote-code

uv run python scripts/export_cbu_metric_tables.py \
  --claimed naive_qwen35_cc12m="${CBU_RESP}" \
  --output-dir "${TABLE_ROOT}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-2000}" \
  --seed "${BOOTSTRAP_SEED:-0}"

uv run python scripts/export_cbu_vqa_tables.py \
  --summary "${VQA_SUMMARY}" \
  --output-md "${TABLE_ROOT}/cbu_vqa_gemma4_table.md" \
  --output-tex "${TABLE_ROOT}/cbu_vqa_gemma4_table.tex"

uv run python scripts/export_cc12m_naive_qwen35_comparison_tables.py \
  --output-dir "${RUN_ROOT}/comparison_tables"
