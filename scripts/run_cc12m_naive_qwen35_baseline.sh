#!/usr/bin/env bash
# Reproduce the CC12M no-system-prompt Qwen3.5-35B-A3B caption baseline.
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

mkdir -p "${RUN_ROOT}"

uv run python scripts/build_naive_vlm_caption_requests.py \
  --input "${SRC_VQA}" \
  --output "${RUN_ROOT}/naive_qwen35_caption.requests.raw.jsonl" \
  --surface naive_qwen35_cc12m \
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
  --image-mode data \
  --target-resolution "${CAPTION_RETRY_TARGET_RESOLUTION:-512}" \
  --jpeg-quality "${CAPTION_RETRY_JPEG_QUALITY:-92}" \
  --resume

uv run python scripts/summarize_naive_vlm_captions.py \
  --responses "${RUN_ROOT}/naive_qwen35_caption.responses.jsonl" \
  --output-jsonl "${RUN_ROOT}/naive_qwen35_cc12m.jsonl" \
  --summary "${RUN_ROOT}/naive_qwen35_caption.summary.json" \
  --surface naive_qwen35_cc12m

uv run python scripts/build_caption_cbu_requests.py \
  --input "${RUN_ROOT}/naive_qwen35_cc12m.jsonl" \
  --output "${RUN_ROOT}/claimed_cbu_v2_naive_qwen35_cc12m_b64.requests.jsonl" \
  --surface naive_qwen35_cc12m \
  --id-field caption_id \
  --token-budget 64

echo "Captioning complete. Stop the Qwen35 caption server before judge metrics."
echo "For Gemma judge metrics, start the Gemma vLLM server and run:"
echo "bash scripts/run_cc12m_naive_qwen35_gemma_metrics.sh"
