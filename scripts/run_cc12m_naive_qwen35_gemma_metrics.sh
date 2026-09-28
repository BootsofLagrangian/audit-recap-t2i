#!/usr/bin/env bash
# Gemma-judge metrics for a CC12M naive-policy caption surface.
#
# The Gemma Judge re-asks the claims that the Qwen3.5-397B-A17B-FP8 extractor
# took from the naive captions: it answers the same VQA request file as the Qwen
# Judge and runs no claim extraction of its own. Inputs:
#   CAPTIONS       naive captions written by scripts/run_cc12m_naive_qwen35_baseline.sh
#   QWEN_CBU_RESP  Qwen397 claimed-CBU responses for those captions, produced with the
#                  397B server from the request file the baseline driver writes:
#                    uv run python scripts/run_text_json_requests.py \
#                      --input <RUN_ROOT>/claimed_cbu_v2_<SURFACE>_b64.requests.jsonl \
#                      --output <QWEN_CBU_RESP> --urls http://localhost:8000 \
#                      --model Qwen/Qwen3.5-397B-A17B-FP8 --concurrency 128 --max-tokens 2048 \
#                      --structured-json --resume --resume-ok-only
# If VQA_REQ does not exist yet, it is built here from QWEN_CBU_RESP; run the Qwen
# Judge (run_cbu_vqa_requests.py with --model Qwen/Qwen3.5-397B-A17B-FP8) on the same
# VQA_REQ so that both judges answer the identical question set.
#
# For the matched-decoding surface set SURFACE=naive_qwen35_sampled_cc12m and the
# RUN_ROOT used for its captions.
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

SURFACE="${SURFACE:-naive_qwen35_cc12m}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
URLS="${URLS:-http://localhost:8000}"
VQA_CONCURRENCY="${VQA_CONCURRENCY:-512}"
VQA_MAX_TOKENS="${VQA_MAX_TOKENS:-2048}"

CAPTIONS="${CAPTIONS:-${RUN_ROOT}/${SURFACE}.jsonl}"
QWEN_CBU_RESP="${QWEN_CBU_RESP:-${CBU_ROOT}/claimed_cbu_v2_${SURFACE}_b64_4494.responses.qwen397_c128_mt2048.jsonl}"
GROUND_REQ="${GROUND_REQ:-${GROUND_ROOT}/grounded_verify_v2_${SURFACE}_b64_4494.requests.qwen397.jsonl}"
VQA_REQ="${VQA_REQ:-${VQA_ROOT}/cbu_vqa_${SURFACE}_b64_4494.requests.qwen397_image_local.jsonl}"
VQA_RESP="${VQA_RESP:-${VQA_ROOT}/cbu_vqa_${SURFACE}_b64_4494.responses.qwen397_claims.gemma4_31b_it_c${VQA_CONCURRENCY}_file_mt${VQA_MAX_TOKENS}.jsonl}"
VQA_SUMMARY="${VQA_RESP%.jsonl}.summary.json"

for required in "${CAPTIONS}" "${QWEN_CBU_RESP}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing input: ${required}" >&2
    echo "Run the baseline driver and the Qwen3.5-397B-A17B-FP8 claim extraction first (see the header of this script)." >&2
    exit 1
  fi
done

mkdir -p "${GROUND_ROOT}" "${VQA_ROOT}" "${LONGCLIP_ROOT}" "${TABLE_ROOT}"

# VQA questions come from the Qwen397 claims, never from a Gemma extraction.
if [[ ! -f "${VQA_REQ}" ]]; then
  uv run python scripts/build_grounded_cbu_verify_requests.py \
    --claimed-responses "${QWEN_CBU_RESP}" \
    --source-jsonl "${CAPTIONS}" \
    --output "${GROUND_REQ}" \
    --require-local-image

  uv run python scripts/build_cbu_vqa_requests.py \
    --input "${GROUND_REQ}" \
    --output "${VQA_REQ}"
fi

uv run python scripts/run_cbu_vqa_requests.py \
  --input "${VQA_REQ}" \
  --output "${VQA_RESP}" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${VQA_CONCURRENCY}" \
  --max-tokens "${VQA_MAX_TOKENS}" \
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
  --surface "${SURFACE}=${CAPTIONS}" \
  --output-dir "${LONGCLIP_ROOT}" \
  --model "${LONGCLIP_MODEL:-zer0int/LongCLIP-GmP-ViT-L-14}" \
  --batch-size "${LONGCLIP_BATCH_SIZE:-64}" \
  --retrieval-block-size 512 \
  --max-length 248 \
  --device "${LONGCLIP_DEVICE:-cuda}" \
  --dtype "${LONGCLIP_DTYPE:-float16}" \
  --bootstrap-reps "${LONGCLIP_BOOTSTRAP_REPS:-1000}" \
  --trust-remote-code

# Claimed-CBU tables use the Qwen397 extraction shared by both judges.
uv run python scripts/export_cbu_metric_tables.py \
  --claimed "${SURFACE}=${QWEN_CBU_RESP}" \
  --output-dir "${TABLE_ROOT}" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-2000}" \
  --seed "${BOOTSTRAP_SEED:-0}"

uv run python scripts/export_cbu_vqa_tables.py \
  --summary "${VQA_SUMMARY}" \
  --output-md "${TABLE_ROOT}/cbu_vqa_gemma4_table.md" \
  --output-tex "${TABLE_ROOT}/cbu_vqa_gemma4_table.tex"

# The five-surface comparison export is keyed to the greedy surface name.
if [[ "${SURFACE}" == "naive_qwen35_cc12m" ]]; then
  uv run python scripts/export_cc12m_naive_qwen35_comparison_tables.py \
    --naive-vqa "${VQA_SUMMARY}" \
    --naive-captions "${CAPTIONS}" \
    --naive-longclip "${LONGCLIP_ROOT}/longclip_retrieval_summary.tsv" \
    --output-dir "${RUN_ROOT}/comparison_tables"
fi
