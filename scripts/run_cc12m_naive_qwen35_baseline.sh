#!/usr/bin/env bash
# Reproduce the CC12M no-system-prompt Qwen3.5-35B-A3B caption baseline.
#
# Two decodings of the naive policy are used in the paper:
#   greedy (default):   SURFACE=naive_qwen35_cc12m, temperature 0
#   matched decoding:   SURFACE=naive_qwen35_sampled_cc12m CAPTION_TEMPERATURE=1.0 \
#                       CAPTION_TOP_K=20 CAPTION_TOP_P=0.95 \
#                       RUN_ROOT=artifacts/recap-ed/cc12m-naive-qwen35-sampled \
#                       bash scripts/run_cc12m_naive_qwen35_baseline.sh
# The matched decoding uses the release sampling defaults of the captioner, the
# same decoding as the released captions.
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
SRC_VQA="${SRC_VQA:-artifacts/vqa-cbu/cc12m-four-caption-llava-url-bridge-5k-local/cbu_vqa_cc12m_four_caption_llava_url_bridge_b64_4494.requests.qwen397_image_local.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-data/local-images/cc12m-naive-qwen35-baseline-2026-05-01/images}"
PROMPT="${PROMPT:-Please generate a detailed caption of this image. Please be as descriptive as possible.}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
URLS="${URLS:-http://localhost:8000}"
SURFACE="${SURFACE:-naive_qwen35_cc12m}"
CAPTION_TEMPERATURE="${CAPTION_TEMPERATURE:-0}"

# Optional sampling parameters; each is sent only when set.
SAMPLING_ARGS=(--temperature "${CAPTION_TEMPERATURE}")
if [[ -n "${CAPTION_TOP_K:-}" ]]; then
  SAMPLING_ARGS+=(--top-k "${CAPTION_TOP_K}")
fi
if [[ -n "${CAPTION_TOP_P:-}" ]]; then
  SAMPLING_ARGS+=(--top-p "${CAPTION_TOP_P}")
fi

mkdir -p "${RUN_ROOT}"

uv run python scripts/build_naive_vlm_caption_requests.py \
  --input "${SRC_VQA}" \
  --output "${RUN_ROOT}/naive_qwen35_caption.requests.raw.jsonl" \
  --surface "${SURFACE}" \
  --prompt "${PROMPT}"

uv run python scripts/materialize_cc12m_images_from_requests.py \
  --input "${RUN_ROOT}/naive_qwen35_caption.requests.raw.jsonl" \
  --output "${RUN_ROOT}/naive_qwen35_caption.requests.local.jsonl" \
  --image-dir "${IMAGE_DIR}" \
  --workers "${MATERIALIZE_WORKERS:-32}"

uv run python scripts/run_naive_vlm_caption_requests.py \
  --input "${RUN_ROOT}/naive_qwen35_caption.requests.local.jsonl" \
  --output "${RUN_ROOT}/naive_qwen35_caption.responses.jsonl" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${CAPTION_CONCURRENCY:-256}" \
  --max-tokens "${CAPTION_MAX_TOKENS:-512}" \
  "${SAMPLING_ARGS[@]}" \
  --image-mode file \
  --resume

# Retry any oversized or transient failures with bounded data-URI images. This
# preserves the same no-system user prompt while keeping image tokens within the
# 2048-token Qwen35 captioning context.
uv run python scripts/run_naive_vlm_caption_requests.py \
  --input "${RUN_ROOT}/naive_qwen35_caption.requests.local.jsonl" \
  --output "${RUN_ROOT}/naive_qwen35_caption.responses.jsonl" \
  --urls "${URLS}" \
  --model "${MODEL}" \
  --concurrency "${CAPTION_RETRY_CONCURRENCY:-64}" \
  --max-tokens "${CAPTION_MAX_TOKENS:-512}" \
  "${SAMPLING_ARGS[@]}" \
  --image-mode data \
  --target-resolution "${CAPTION_RETRY_TARGET_RESOLUTION:-512}" \
  --jpeg-quality "${CAPTION_RETRY_JPEG_QUALITY:-92}" \
  --resume

uv run python scripts/summarize_naive_vlm_captions.py \
  --responses "${RUN_ROOT}/naive_qwen35_caption.responses.jsonl" \
  --output-jsonl "${RUN_ROOT}/${SURFACE}.jsonl" \
  --summary "${RUN_ROOT}/naive_qwen35_caption.summary.json" \
  --surface "${SURFACE}"

uv run python scripts/build_caption_cbu_requests.py \
  --input "${RUN_ROOT}/${SURFACE}.jsonl" \
  --output "${RUN_ROOT}/claimed_cbu_v2_${SURFACE}_b64.requests.jsonl" \
  --surface "${SURFACE}" \
  --id-field caption_id \
  --token-budget 64

echo "Captioning complete. Stop the Qwen35 caption server before the judge stages."
echo "Next: extract claims with the Qwen3.5-397B-A17B-FP8 server from"
echo "  ${RUN_ROOT}/claimed_cbu_v2_${SURFACE}_b64.requests.jsonl"
echo "then run the Qwen and Gemma judges on the VQA requests built from those claims:"
echo "  bash scripts/run_cc12m_naive_qwen35_gemma_metrics.sh"
