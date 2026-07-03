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
# worker incrementally sends pre-run chunks 1..TRUNK_SIZE-1, then immediately
# streams the measured actual request for TRUNK_SIZE. Set
# BARRIER_PREFIX=1 to use the older all-prefixes-first barrier shape. See
# reports/README_qwen35_realtime_benchmark_20260701.md for reference numbers.
# Request profiling is enabled by default so summaries include vLLM-style
# profile_* metrics. Set PROFILE_REQUESTS=0 to skip request profiling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi
PORT="${PORT:-8162}"
SIL_OFFSET="${SIL_OFFSET:-0}"
CONCURRENCY="${CONCURRENCY:-12}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-$CONCURRENCY}"
TRUNK_SIZE="${TRUNK_SIZE:-40}"
STAGGER_MS="${STAGGER_MS:-0}"
TEMPERATURE="${TEMPERATURE:-0.000001}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-1}"
MIN_P="${MIN_P:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SEED="${SEED:-3408}"
TALKER_TEMPERATURE="${TALKER_TEMPERATURE:-0.9}"
TALKER_TOP_K="${TALKER_TOP_K:-50}"
TALKER_TOP_P="${TALKER_TOP_P:-1.0}"
TALKER_REPETITION_PENALTY="${TALKER_REPETITION_PENALTY:-1.05}"
TALKER_SEED="${TALKER_SEED:-3408}"
SUBTALKER_TEMPERATURE="${SUBTALKER_TEMPERATURE:-0.1}"
SUBTALKER_TOP_K="${SUBTALKER_TOP_K:-5}"
SUBTALKER_TOP_P="${SUBTALKER_TOP_P:-1.0}"
SUBTALKER_REPETITION_PENALTY="${SUBTALKER_REPETITION_PENALTY:-1.05}"
SUBTALKER_SEED="${SUBTALKER_SEED:-3408}"
VOICE="${VOICE:-f245}"
BARRIER_PREFIX="${BARRIER_PREFIX:-${BARRIER_PRERUN:-0}}"
PREFIX_MAX_TOKENS="${PREFIX_MAX_TOKENS:-${PRERUN_MAX_TOKENS:-2}}"
PROFILE_REQUESTS="${PROFILE_REQUESTS:-1}"
THINKER_ONLY="${THINKER_ONLY:-${SGLANG_OMNI_THINKER_ONLY:-0}}"
TEXT_ONLY="${TEXT_ONLY:-0}"
case "$THINKER_ONLY" in
  1|true|TRUE|yes|YES) TEXT_ONLY=1 ;;
  0|false|FALSE|no|NO) ;;
  *)
    echo "THINKER_ONLY must be 1/0, true/false, or yes/no; got: $THINKER_ONLY" >&2
    exit 1
    ;;
esac
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

case "$TEXT_ONLY" in
  1|true|TRUE|yes|YES)
    text_args=(--text-only)
    if [ "$THINKER_ONLY" = "1" ] || [ "$THINKER_ONLY" = "true" ] || [ "$THINKER_ONLY" = "TRUE" ] || [ "$THINKER_ONLY" = "yes" ] || [ "$THINKER_ONLY" = "YES" ]; then
      MODE_LABEL="thinkeronly"
    else
      MODE_LABEL="textonly"
    fi
    ;;
  0|false|FALSE|no|NO)
    text_args=(); MODE_LABEL="textaudio" ;;
  *)
    echo "TEXT_ONLY must be 1/0, true/false, or yes/no; got: $TEXT_ONLY" >&2
    exit 1
    ;;
esac

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$($PYTHON_BIN - "$PORT" <<'PY'
import glob
import os
import subprocess
import sys

port = sys.argv[1]
patterns = [
    f"results/sg_realtime_stablefast_*_{port}_*",
    f"results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_{port}_*",
    f"results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_{port}_*",
    f"results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_{port}_*",
]
dirs = []
for pattern in patterns:
    dirs.extend(path for path in glob.glob(pattern) if os.path.isdir(path))
active_pid = None
try:
    ps = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    for line in ps.splitlines():
        if "sglang_omni.cli serve" in line and f"--port {port}" in line:
            active_pid = line.strip().split(None, 1)[0]
            break
except Exception:
    active_pid = None
if active_pid:
    pid_dirs = []
    for path in dirs:
        pid_path = os.path.join(path, "server.pid")
        try:
            if open(pid_path).read().strip() == active_pid:
                pid_dirs.append(path)
        except OSError:
            pass
    if pid_dirs:
        print(max(pid_dirs, key=os.path.getmtime))
        raise SystemExit(0)
if dirs:
    print(max(dirs, key=os.path.getmtime))
PY
)"
fi
if [ -z "$RUN_DIR" ]; then
  echo "No stable run directory found. Start the server script first." >&2
  exit 1
fi

echo "Using RUN_DIR=$RUN_DIR"
echo "Checking active $PORT service..."
ps -eo pid,etimes,args | grep 'sglang_omni.cli serve' | grep -- "--port $PORT" | grep -v grep

OUT_DIR="$RUN_DIR/client_c${CONCURRENCY}_rtcflow_${RUN_SHAPE}_${MODE_LABEL}_sil${SIL_OFFSET}_trunk${TRUNK_SIZE}_samples${TOTAL_SAMPLES}_stagger${STAGGER_MS}_temp${TEMPERATURE}_topk${TOP_K}_topp${TOP_P}_prefixmt${PREFIX_MAX_TOKENS}_${RUN_LABEL}_$(date +%H%M%S)"
echo "$OUT_DIR" > "$RUN_DIR/latest_validation_client_dir.txt"

echo "--- benchmark config ---"
echo "base_url=http://127.0.0.1:${PORT}"
echo "concurrency=$CONCURRENCY"
echo "total_samples=$TOTAL_SAMPLES"
echo "trunk_size=$TRUNK_SIZE"
echo "stagger_ms=$STAGGER_MS"
echo "sil_offset=$SIL_OFFSET"
echo "temperature=$TEMPERATURE"
echo "top_p=$TOP_P"
echo "top_k=$TOP_K"
echo "min_p=$MIN_P"
echo "repetition_penalty=$REPETITION_PENALTY"
echo "seed=$SEED"
echo "talker_temperature=$TALKER_TEMPERATURE"
echo "talker_top_k=$TALKER_TOP_K"
echo "talker_top_p=$TALKER_TOP_P"
echo "talker_repetition_penalty=$TALKER_REPETITION_PENALTY"
echo "talker_seed=$TALKER_SEED"
echo "subtalker_temperature=$SUBTALKER_TEMPERATURE"
echo "subtalker_top_k=$SUBTALKER_TOP_K"
echo "subtalker_top_p=$SUBTALKER_TOP_P"
echo "subtalker_repetition_penalty=$SUBTALKER_REPETITION_PENALTY"
echo "subtalker_seed=$SUBTALKER_SEED"
echo "voice=$VOICE"
echo "barrier_prefix=$BARRIER_PREFIX"
echo "realtime_shape=$RUN_SHAPE: per-worker prefix chunks 1..$((TRUNK_SIZE - 1)), then measured actual chunk $TRUNK_SIZE"
echo "prefix_max_tokens=$PREFIX_MAX_TOKENS"
echo "text_only=$TEXT_ONLY"
echo "thinker_only=$THINKER_ONLY"
echo "profile_requests=$PROFILE_REQUESTS"
if [ -n "$PROFILE_RUN_ID" ]; then
  echo "profile_run_id=$PROFILE_RUN_ID"
fi
echo "out_dir=$OUT_DIR"

PYTHONPATH=. "$PYTHON_BIN" benchmarks/eval/qwen35_omni_sglang_rtc_concurrency.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --model qwen3_5-omni \
  --output-dir "$OUT_DIR" \
  --trunk-size "$TRUNK_SIZE" \
  --concurrency "$CONCURRENCY" \
  --total-samples "$TOTAL_SAMPLES" \
  --stagger-ms "$STAGGER_MS" \
  --sil-offset "$SIL_OFFSET" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p "$MIN_P" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --seed "$SEED" \
  --voice "$VOICE" \
  --talker-temperature "$TALKER_TEMPERATURE" \
  --talker-top-k "$TALKER_TOP_K" \
  --talker-top-p "$TALKER_TOP_P" \
  --talker-repetition-penalty "$TALKER_REPETITION_PENALTY" \
  --talker-seed "$TALKER_SEED" \
  --subtalker-temperature "$SUBTALKER_TEMPERATURE" \
  --subtalker-top-k "$SUBTALKER_TOP_K" \
  --subtalker-top-p "$SUBTALKER_TOP_P" \
  --subtalker-repetition-penalty "$SUBTALKER_REPETITION_PENALTY" \
  --subtalker-seed "$SUBTALKER_SEED" \
  --prefix-max-tokens "$PREFIX_MAX_TOKENS" \
  "${text_args[@]}" \
  "${barrier_args[@]}" \
  "${profile_args[@]}"

echo "--- metrics ---"
"$PYTHON_BIN" - <<PY
import json, glob, os
out = "$OUT_DIR"
m = json.load(open(os.path.join(out, "metrics.json")))
for k in [
    "completed", "failed", "actual_elapsed_s",
    "concurrency_shape", "prefix_max_tokens", "mode",
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed",
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
    "client_first_text_event_avg_ms", "client_first_text_event_p99_ms",
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
