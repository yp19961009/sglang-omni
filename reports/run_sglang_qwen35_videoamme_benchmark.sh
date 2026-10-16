#!/usr/bin/env bash
set -euo pipefail

# Full-chain Video-AMME benchmark for Qwen3.5-Omni.
#
# This intentionally does not run the old RTC pre-run/actual flow. Each request
# is a normal Video-AMME chat completion with the full video/audio question
# payload, matching benchmarks/eval/benchmark_omni_videoamme.py.
#
# Example:
#   cd /myapp/sglang-omni
#   MAX_CONCURRENCY=8 MAX_SAMPLES=50 \
#     bash reports/run_sglang_qwen35_videoamme_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8162}"
MODEL="${MODEL:-qwen3_5-omni}"
REPO_ID="${VIDEOAMME_REPO_ID:-zhaochenyang20/Video_AMME_ci}"
SPLIT="${VIDEOAMME_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-401408}"
ENABLE_AUDIO="${ENABLE_AUDIO:-1}"
AUDIO_FORMAT="${AUDIO_FORMAT:-wav}"
AUDIO_VOICE="${AUDIO_VOICE:-f245}"
STREAM="${STREAM:-1}"
SKIP_WER="${SKIP_WER:-1}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-results/videoamme_sg_fullchain_c${MAX_CONCURRENCY}_n${MAX_SAMPLES}_${RUN_STAMP}}"

cd "$REPO"

args=(
  benchmarks/eval/benchmark_omni_videoamme.py
  --base-url "$BASE_URL"
  --model "$MODEL"
  --repo-id "$REPO_ID"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --max-concurrency "$MAX_CONCURRENCY"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --video-fps "$VIDEO_FPS"
  --video-max-frames "$VIDEO_MAX_FRAMES"
  --video-max-pixels "$VIDEO_MAX_PIXELS"
)

case "$ENABLE_AUDIO" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--enable-audio --audio-format "$AUDIO_FORMAT" --audio-voice "$AUDIO_VOICE")
    ;;
esac

case "$STREAM" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--stream)
    ;;
esac

case "$SKIP_WER" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--skip-wer)
    ;;
esac

case "$DISABLE_TQDM" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--disable-tqdm)
    ;;
esac

echo "--- Video-AMME full-chain benchmark ---"
echo "base_url=$BASE_URL"
echo "model=$MODEL"
echo "max_concurrency=$MAX_CONCURRENCY"
echo "max_samples=$MAX_SAMPLES"
echo "enable_audio=$ENABLE_AUDIO"
echo "stream=$STREAM"
echo "output_dir=$OUTPUT_DIR"

HF_HOME="${HF_HOME:-/myapp/data/videoamme}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/myapp/data/videoamme/datasets}" \
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/myapp/data/videoamme/hub}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
PYTHONPATH=. "$PYTHON_BIN" "${args[@]}"
