#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-H20-ecs-gangouyu}"
CONTAINER="${CONTAINER:-b5f665f3d883}"
REMOTE_REPO="${REMOTE_REPO:-/myapp/sglang-omni}"
PORT="${PORT:-8162}"
SIL_OFFSET="${SIL_OFFSET:-700}"
CONCURRENCY="${CONCURRENCY:-12}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-$CONCURRENCY}"
TRUNK_SIZE="${TRUNK_SIZE:-40}"
STAGGER_MS="${STAGGER_MS:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
VOICE="${VOICE:-m02}"
BARRIER_PRERUN="${BARRIER_PRERUN:-1}"
RUN_LABEL="${RUN_LABEL:-c${CONCURRENCY}}"

ssh "$REMOTE_HOST" "docker exec -i $CONTAINER bash" <<SH
set -euo pipefail
cd "$REMOTE_REPO"

PORT="$PORT"
SIL_OFFSET="$SIL_OFFSET"
CONCURRENCY="$CONCURRENCY"
TOTAL_SAMPLES="$TOTAL_SAMPLES"
TRUNK_SIZE="$TRUNK_SIZE"
STAGGER_MS="$STAGGER_MS"
TEMPERATURE="$TEMPERATURE"
VOICE="$VOICE"
BARRIER_PRERUN="$BARRIER_PRERUN"
RUN_LABEL="$RUN_LABEL"

case "\$BARRIER_PRERUN" in
  1|true|TRUE|yes|YES) barrier_args=(--barrier-prerun) ;;
  0|false|FALSE|no|NO) barrier_args=() ;;
  *)
    echo "BARRIER_PRERUN must be 1/0, true/false, or yes/no; got: \$BARRIER_PRERUN" >&2
    exit 1
    ;;
esac

RUN_DIR="\$(ls -td results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_\${PORT}_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_\${PORT}_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_\${PORT}_* 2>/dev/null | head -1)"
if [ -z "\$RUN_DIR" ]; then
  echo "No stable mem0.80+c2w4 run directory found." >&2
  exit 1
fi

echo "Using RUN_DIR=\$RUN_DIR"
echo "Checking active \$PORT service..."
ps -eo pid,etimes,args | grep 'sglang_omni.cli serve' | grep -- "--port \$PORT" | grep -v grep

OUT_DIR="\$RUN_DIR/client_c\${CONCURRENCY}_realtime_audio_vllmstyle_sil\${SIL_OFFSET}_trunk\${TRUNK_SIZE}_samples\${TOTAL_SAMPLES}_stagger\${STAGGER_MS}_temp\${TEMPERATURE}_barrier\${BARRIER_PRERUN}_\${RUN_LABEL}_\$(date +%H%M%S)"
echo "\$OUT_DIR" > "\$RUN_DIR/latest_validation_client_dir.txt"

echo "--- benchmark config ---"
echo "base_url=http://127.0.0.1:\${PORT}"
echo "concurrency=\$CONCURRENCY"
echo "total_samples=\$TOTAL_SAMPLES"
echo "trunk_size=\$TRUNK_SIZE"
echo "stagger_ms=\$STAGGER_MS"
echo "sil_offset=\$SIL_OFFSET"
echo "temperature=\$TEMPERATURE"
echo "voice=\$VOICE"
echo "barrier_prerun=\$BARRIER_PRERUN"
echo "out_dir=\$OUT_DIR"

PYTHONPATH=. python benchmarks/eval/qwen35_omni_sglang_rtc_concurrency.py \\
  --base-url "http://127.0.0.1:\${PORT}" \\
  --model qwen3_5-omni \\
  --output-dir "\$OUT_DIR" \\
  --trunk-size "\$TRUNK_SIZE" \\
  --concurrency "\$CONCURRENCY" \\
  --total-samples "\$TOTAL_SAMPLES" \\
  --stagger-ms "\$STAGGER_MS" \\
  --sil-offset "\$SIL_OFFSET" \\
  --temperature "\$TEMPERATURE" \\
  --voice "\$VOICE" \\
  "\${barrier_args[@]}"

echo "--- metrics ---"
python - <<PY
import json, glob, os
out = "\$OUT_DIR"
m = json.load(open(os.path.join(out, "metrics.json")))
for k in [
    "completed", "failed", "actual_elapsed_s",
    "ttft_avg_ms", "ttft_p99_ms",
    "ttfa_avg_ms", "ttfa_p99_ms",
    "last_audio_avg_ms", "last_audio_p99_ms",
    "e2e_avg_ms", "e2e_p99_ms",
    "audio_duration_avg_s", "bang_count", "errors",
]:
    print(f"{k}={m.get(k)}")
print("wav_count=", len(glob.glob(os.path.join(out, "sample_*", "*.wav"))))
print("result_count=", len(glob.glob(os.path.join(out, "sample_*", "result.json"))))
PY

echo "--- recent server error counters ---"
SERVER_LOG="\$RUN_DIR/server.log"
echo -n "500="; grep -c 'HTTP/1.1" 500' "\$SERVER_LOG" || true
echo -n "oom="; grep -ci 'out of memory' "\$SERVER_LOG" || true
echo -n "mismatch="; grep -ci 'feature/token mismatch' "\$SERVER_LOG" || true
echo -n "omitted_payload_cache_miss="; grep -c 'Visual item payload was omitted but encoder item cache missed' "\$SERVER_LOG" || true
SH
