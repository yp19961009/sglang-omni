#!/usr/bin/env bash
set -euo pipefail

# Run from the server/container repo, not from the Mac SSH workspace:
#   cd /myapp/sglang-omni
#   CONCURRENCY=1 TOTAL_SAMPLES=1 RUN_LABEL=c1 \
#     bash reports/run_sglang_qwen35_stable_c12_benchmark.sh
#   CONCURRENCY=12 TOTAL_SAMPLES=12 RUN_LABEL=c12 \
#     bash reports/run_sglang_qwen35_stable_c12_benchmark.sh
#
# This follows the vLLM run_rtc_profile concurrency shape by default: each
# worker incrementally sends pre-run chunks 1..TRUNK_SIZE, then immediately
# streams the measured actual request for the same TRUNK_SIZE. Set
# BARRIER_PREFIX=1 to use the older all-prefixes-first barrier shape. See
# reports/README_qwen35_realtime_benchmark_20260701.md for reference numbers.
# Request profiling is enabled by default so summaries include vLLM-style
# profile_* metrics. Set PROFILE_REQUESTS=0 to skip request profiling.

REPO="${REPO:-/myapp/sglang-omni}"
PORT="${PORT:-8162}"
SIL_OFFSET="${SIL_OFFSET:-700}"
CONCURRENCY="${CONCURRENCY:-12}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-$CONCURRENCY}"
TRUNK_SIZE="${TRUNK_SIZE:-40}"
STAGGER_MS="${STAGGER_MS:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
VOICE="${VOICE:-m02}"
BARRIER_PREFIX="${BARRIER_PREFIX:-${BARRIER_PRERUN:-0}}"
PREFIX_MAX_TOKENS="${PREFIX_MAX_TOKENS:-${PRERUN_MAX_TOKENS:-2}}"
PROFILE_REQUESTS="${PROFILE_REQUESTS:-1}"
RUN_LABEL="${RUN_LABEL:-c${CONCURRENCY}}"
RUN_DIR="${RUN_DIR:-}"

cd "$REPO"

case "$BARRIER_PREFIX" in
  1|true|TRUE|yes|YES)
    barrier_args=(--barrier-prefix); RUN_SHAPE="barrier_prefix" ;;
  0|false|FALSE|no|NO)
    barrier_args=(); RUN_SHAPE="vllm_pipeline" ;;
  *)
    echo "BARRIER_PREFIX must be 1/0, true/false, or yes/no; got: $BARRIER_PREFIX" >&2
    exit 1
    ;;
esac

case "$PROFILE_REQUESTS" in
  1|true|TRUE|yes|YES)
    PROFILE_RUN_ID="sg_${RUN_LABEL}_c${CONCURRENCY}_t${TRUNK_SIZE}_$(date +%H%M%S)"
    profile_args=(--profile-actual-run-id "$PROFILE_RUN_ID") ;;
  0|false|FALSE|no|NO)
    PROFILE_RUN_ID=""
    profile_args=() ;;
  *)
    echo "PROFILE_REQUESTS must be 1/0, true/false, or yes/no; got: $PROFILE_REQUESTS" >&2
    exit 1
    ;;
esac

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(ls -td results/sg_realtime_stablefast_*_${PORT}_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_${PORT}_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_${PORT}_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_${PORT}_* 2>/dev/null | head -1)"
fi
if [ -z "$RUN_DIR" ]; then
  echo "No stable run directory found. Start the server script first." >&2
  exit 1
fi

echo "Using RUN_DIR=$RUN_DIR"
echo "Checking active $PORT service..."
ps -eo pid,etimes,args | grep 'sglang_omni.cli serve' | grep -- "--port $PORT" | grep -v grep

OUT_DIR="$RUN_DIR/client_c${CONCURRENCY}_rtcflow_${RUN_SHAPE}_sil${SIL_OFFSET}_trunk${TRUNK_SIZE}_samples${TOTAL_SAMPLES}_stagger${STAGGER_MS}_temp${TEMPERATURE}_prefixmt${PREFIX_MAX_TOKENS}_${RUN_LABEL}_$(date +%H%M%S)"
echo "$OUT_DIR" > "$RUN_DIR/latest_validation_client_dir.txt"

echo "--- benchmark config ---"
echo "base_url=http://127.0.0.1:${PORT}"
echo "concurrency=$CONCURRENCY"
echo "total_samples=$TOTAL_SAMPLES"
echo "trunk_size=$TRUNK_SIZE"
echo "stagger_ms=$STAGGER_MS"
echo "sil_offset=$SIL_OFFSET"
echo "temperature=$TEMPERATURE"
echo "voice=$VOICE"
echo "barrier_prefix=$BARRIER_PREFIX"
echo "realtime_shape=$RUN_SHAPE: per-worker prefix chunks 1..$TRUNK_SIZE, then measured actual chunk $TRUNK_SIZE"
echo "prefix_max_tokens=$PREFIX_MAX_TOKENS"
echo "profile_requests=$PROFILE_REQUESTS"
if [ -n "$PROFILE_RUN_ID" ]; then
  echo "profile_run_id=$PROFILE_RUN_ID"
fi
echo "out_dir=$OUT_DIR"

PYTHONPATH=. python benchmarks/eval/qwen35_omni_sglang_rtc_concurrency.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --model qwen3_5-omni \
  --output-dir "$OUT_DIR" \
  --trunk-size "$TRUNK_SIZE" \
  --concurrency "$CONCURRENCY" \
  --total-samples "$TOTAL_SAMPLES" \
  --stagger-ms "$STAGGER_MS" \
  --sil-offset "$SIL_OFFSET" \
  --temperature "$TEMPERATURE" \
  --voice "$VOICE" \
  --prefix-max-tokens "$PREFIX_MAX_TOKENS" \
  "${barrier_args[@]}" \
  "${profile_args[@]}"

echo "--- metrics ---"
python - <<PY
import json, glob, os
out = "$OUT_DIR"
m = json.load(open(os.path.join(out, "metrics.json")))
for k in [
    "completed", "failed", "actual_elapsed_s",
    "concurrency_shape", "prefix_max_tokens",
    "ttft_semantics", "ttfa_semantics",
    "ttft_avg_ms", "ttft_p99_ms",
    "ttfa_avg_ms", "ttfa_p99_ms",
    "first_output_avg_ms", "first_output_p99_ms", "first_output_type_counts",
    "profile_num_requests", "profile_stats_source",
    "profile_ttft_avg_ms", "profile_ttft_p99_ms",
    "profile_ttfa_avg_ms", "profile_ttfa_p99_ms",
    "profile_ttfa_thinker_prefill_avg_ms",
    "profile_ttfa_hf_preproc_avg_ms",
    "profile_ttfa_talker_prefill_avg_ms",
    "profile_ttfa_code2wav_first_chunk_avg_ms",
    "first_text_event_avg_ms", "first_text_event_p99_ms",
    "first_audio_event_avg_ms", "first_audio_event_p99_ms",
    "audio_before_text_event_count", "audio_before_text_sample_indices",
    "last_audio_avg_ms", "last_audio_p99_ms",
    "e2e_avg_ms", "e2e_p99_ms",
    "audio_duration_avg_s", "bang_count", "errors",
]:
    print(f"{k}={m.get(k)}")
print("wav_count=", len(glob.glob(os.path.join(out, "sample_*", "*.wav"))))
print("result_count=", len(glob.glob(os.path.join(out, "sample_*", "result.json"))))
PY

echo "--- recent server error counters ---"
SERVER_LOG="$RUN_DIR/server.log"
echo -n "500="; grep -c 'HTTP/1.1" 500' "$SERVER_LOG" || true
echo -n "oom="; grep -ci 'out of memory' "$SERVER_LOG" || true
echo -n "mismatch="; grep -ci 'feature/token mismatch' "$SERVER_LOG" || true
echo -n "omitted_payload_cache_miss="; grep -c 'Visual item payload was omitted but encoder item cache missed' "$SERVER_LOG" || true
