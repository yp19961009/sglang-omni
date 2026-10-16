#!/usr/bin/env bash
# Run one SGLang Qwen3.5-Omni 40-chunk RTC C1 output/latency check.
#
# Assumes the SGLang speech server is already running. See the runbook:
#   benchmarks/reports/qwen35_omni_rtc_c1_accuracy_runbook_20260625.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_URL="${BASE_URL:-http://127.0.0.1:8161}"
MODEL="${MODEL:-qwen3_5-omni}"
RTC_TEST_DIR="${RTC_TEST_DIR:-/home/gangouyu/data/share-data-6batch}"
RUN_ROOT="${RUN_ROOT:-results/qwen35_rtc_c1_accuracy_sglang_$(date +%Y%m%d_%H%M%S)}"
TRUNK_SIZE="${TRUNK_SIZE:-40}"
BATCH_IDX="${BATCH_IDX:-0}"
SIL_START_IDX="${SIL_START_IDX:-0}"
VIDEO_START_IDX="${VIDEO_START_IDX:-0}"
QUESTION_IDX="${QUESTION_IDX:-0}"
VOICE="${VOICE:-f245}"
TALKER_SEED="${TALKER_SEED-3408}"
SUBTALKER_SEED="${SUBTALKER_SEED-3408}"
TALKER_TEMPERATURE="${TALKER_TEMPERATURE-0.9}"
SUBTALKER_TEMPERATURE="${SUBTALKER_TEMPERATURE-0.1}"
RUN_ASR="${RUN_ASR:-0}"
ASR_MODEL="${ASR_MODEL:-large-v3}"
EXPECTED_TEXT="${EXPECTED_TEXT:-我叫千问，是阿里巴巴集团旗下的通义实验室自主研发的多模态超大规模语言模型，有什么我可以帮助你的吗？}"

mkdir -p "$RUN_ROOT/events"

if ! curl -fsS "$BASE_URL/v1/models" >/dev/null; then
  echo "ERROR: SGLang server is not reachable at $BASE_URL/v1/models" >&2
  echo "Start qwen35-omni SGLang server first, then rerun this script." >&2
  exit 1
fi

OUT_DIR="$RUN_ROOT/sglang_trunk${TRUNK_SIZE}_c1"
talker_args=()
if [[ -n "$TALKER_TEMPERATURE" ]]; then
  talker_args+=(--talker-temperature "$TALKER_TEMPERATURE")
fi
if [[ -n "$TALKER_SEED" ]]; then
  talker_args+=(--talker-seed "$TALKER_SEED")
fi
if [[ -n "$SUBTALKER_TEMPERATURE" ]]; then
  talker_args+=(--subtalker-temperature "$SUBTALKER_TEMPERATURE")
fi
if [[ -n "$SUBTALKER_SEED" ]]; then
  talker_args+=(--subtalker-seed "$SUBTALKER_SEED")
fi

python3 -m benchmarks.eval.qwen35_omni_sglang_rtc_profile \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --rtc-test-dir "$RTC_TEST_DIR" \
  --output-dir "$OUT_DIR" \
  --event-dir "$RUN_ROOT/events" \
  --run-id "sglang_rtc_c1_accuracy" \
  --trunk-size "$TRUNK_SIZE" \
  --batch-idx "$BATCH_IDX" \
  --sil-start-idx "$SIL_START_IDX" \
  --video-start-idx "$VIDEO_START_IDX" \
  --question-idx "$QUESTION_IDX" \
  --video-fps 1 \
  --visual-mode video_frames \
  --max-tokens 8192 \
  --prerun-max-tokens 0 \
  --temperature 1.0 \
  "${talker_args[@]}" \
  --voice "$VOICE" \
  --profile

python3 -m benchmarks.eval.qwen35_omni_rtc_compare_outputs \
  --sglang "$OUT_DIR/per_request.json" \
  --out "$RUN_ROOT/output_check.md" \
  --json-out "$RUN_ROOT/output_check.json" \
  --print

if [[ "$RUN_ASR" == "1" ]]; then
  python3 -m benchmarks.eval.qwen35_omni_rtc_asr_check \
    --per-request "$OUT_DIR/per_request.json" \
    --expected "$EXPECTED_TEXT" \
    --model "$ASR_MODEL" \
    --out "$RUN_ROOT/asr_check.json" \
    --print
fi

echo
echo "RUN_ROOT=$RUN_ROOT"
echo "SGLANG_PER_REQUEST=$OUT_DIR/per_request.json"
echo "SGLANG_METRICS=$OUT_DIR/metrics.json"
echo "OUTPUT_CHECK=$RUN_ROOT/output_check.md"
if [[ "$RUN_ASR" == "1" ]]; then
  echo "ASR_CHECK=$RUN_ROOT/asr_check.json"
fi
