#!/usr/bin/env bash
set -euo pipefail

# Container-native SGLang single-example breakdown measurement.
# Run inside the SGLang container:
#   cd /myapp/sglang-omni
#   benchmarks/eval/qwen35_omni_single_breakdown_sglang.sh

ROOT="${ROOT:-/myapp/sglang-omni}"
cd "${ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8161}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-qwen3_5-omni}"
REPO_ID="${REPO_ID:-/myapp/sglang-omni/results/videoamme_pair_warm0142_measure0111_local}"
RUN_ROOT="${RUN_ROOT:-results/qwen35_single_breakdown_sglang_$(date +%Y%m%d_%H%M%S)}"
EVENT_DIR="${EVENT_DIR:-${RUN_ROOT}/events}"
WARMUP_OUT="${RUN_ROOT}/warmup_014_2_stream"
MEASURE_OUT="${RUN_ROOT}/measure_011_1_stream_profile"
RUN_ID="${RUN_ID:-single_0111_stream_rerun}"

MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-401408}"
AUDIO_VOICE="${AUDIO_VOICE:-m02}"

export HF_HOME="${HF_HOME:-/myapp/data/videoamme}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

abs_path() {
  case "$1" in
    /*) printf "%s\n" "$1" ;;
    *) printf "%s/%s\n" "$(pwd)" "$1" ;;
  esac
}

EVENT_DIR_ABS="$(abs_path "${EVENT_DIR}")"
RUN_ROOT_ABS="$(abs_path "${RUN_ROOT}")"

mkdir -p "${RUN_ROOT_ABS}" "${EVENT_DIR_ABS}" "${WARMUP_OUT}" "${MEASURE_OUT}"

if ! curl -fsS "${BASE_URL}/v1/models" >/dev/null; then
  echo "ERROR: SGLang server is not reachable at ${BASE_URL}/v1/models" >&2
  exit 1
fi

echo "RUN_ROOT=${RUN_ROOT}"
echo "REPO_ID=${REPO_ID}"
echo "BASE_URL=${BASE_URL}"
echo "RUN_ID=${RUN_ID}"

echo
echo "=== Warmup sample 014-2, excluded from profile/statistics ==="
python -m benchmarks.eval.benchmark_omni_videoamme \
  --model "${MODEL}" --host "${HOST}" --port "${PORT}" \
  --repo-id "${REPO_ID}" \
  --output-dir "${WARMUP_OUT}" \
  --max-samples 1 --sample-offset 0 --max-concurrency 1 \
  --max-tokens "${MAX_TOKENS}" --temperature "${TEMPERATURE}" \
  --video-fps "${VIDEO_FPS}" --video-max-frames "${VIDEO_MAX_FRAMES}" \
  --video-max-pixels "${VIDEO_MAX_PIXELS}" \
  --enable-audio --audio-voice "${AUDIO_VOICE}" \
  --stream --skip-wer --disable-tqdm \
  2>&1 | tee "${WARMUP_OUT}/run.log"

echo
echo "=== Measured sample 011-1 with request profile ==="
curl -fsS "${BASE_URL}/start_request_profile" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"${RUN_ID}\",\"event_dir\":\"${EVENT_DIR_ABS}\"}" \
  >/dev/null

set +e
python -m benchmarks.eval.benchmark_omni_videoamme \
  --model "${MODEL}" --host "${HOST}" --port "${PORT}" \
  --repo-id "${REPO_ID}" \
  --output-dir "${MEASURE_OUT}" \
  --max-samples 1 --sample-offset 1 --max-concurrency 1 \
  --max-tokens "${MAX_TOKENS}" --temperature "${TEMPERATURE}" \
  --video-fps "${VIDEO_FPS}" --video-max-frames "${VIDEO_MAX_FRAMES}" \
  --video-max-pixels "${VIDEO_MAX_PIXELS}" \
  --enable-audio --audio-voice "${AUDIO_VOICE}" \
  --stream --skip-wer --disable-tqdm \
  2>&1 | tee "${MEASURE_OUT}/run.log"
bench_status=${PIPESTATUS[0]}
set -e

curl -fsS "${BASE_URL}/stop_request_profile" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"${RUN_ID}\"}" \
  >/dev/null || true

if [[ "${bench_status}" -ne 0 ]]; then
  echo "ERROR: measured benchmark failed; see ${MEASURE_OUT}/run.log" >&2
  exit "${bench_status}"
fi

python -m sglang_omni.profiler \
  "${EVENT_DIR_ABS}" \
  --format json \
  --out "${RUN_ROOT_ABS}/request_profile_${RUN_ID}.json"

echo
echo "SGLANG_SINGLE_BREAKDOWN_RUN_ROOT=${RUN_ROOT}"
