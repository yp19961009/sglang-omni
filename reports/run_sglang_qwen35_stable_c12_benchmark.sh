#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-H20-ecs-gangouyu}"
CONTAINER="${CONTAINER:-b5f665f3d883}"
REMOTE_REPO="${REMOTE_REPO:-/myapp/sglang-omni}"
PORT="${PORT:-8162}"
SIL_OFFSET="${SIL_OFFSET:-700}"

ssh "$REMOTE_HOST" "docker exec -i $CONTAINER bash" <<SH
set -euo pipefail
cd "$REMOTE_REPO"

RUN_DIR="\$(ls -td results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_* results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_8162_* 2>/dev/null | head -1)"
if [ -z "\$RUN_DIR" ]; then
  echo "No stable mem0.80+c2w4 run directory found." >&2
  exit 1
fi

echo "Using RUN_DIR=\$RUN_DIR"
echo "Checking active 8162 service..."
ps -eo pid,etimes,args | grep 'sglang_omni.cli serve' | grep -- '--port 8162' | grep -v grep

OUT_DIR="\$RUN_DIR/client_c12_realtime_audio_vllmstyle_sil${SIL_OFFSET}_stagger0_temp1_barrier_validation_\$(date +%H%M%S)"
echo "\$OUT_DIR" > "\$RUN_DIR/latest_validation_client_dir.txt"

PYTHONPATH=. python benchmarks/eval/qwen35_omni_sglang_rtc_concurrency.py \\
  --base-url "http://127.0.0.1:${PORT}" \\
  --model qwen3_5-omni \\
  --output-dir "\$OUT_DIR" \\
  --trunk-size 40 \\
  --concurrency 12 \\
  --total-samples 12 \\
  --stagger-ms 0 \\
  --sil-offset "${SIL_OFFSET}" \\
  --temperature 1.0 \\
  --voice m02 \\
  --barrier-prerun

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
