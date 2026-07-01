#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-H20-ecs-gangouyu}"
CONTAINER="${CONTAINER:-b5f665f3d883}"
REMOTE_REPO="${REMOTE_REPO:-/myapp/sglang-omni}"
PORT="${PORT:-8162}"
MODE="${MODE:-auto}"
FORCE_RESTART="${FORCE_RESTART:-0}"

if [ "$MODE" = "auto" ]; then
  if [ -d "$REMOTE_REPO" ]; then
    MODE="local"
  elif command -v docker >/dev/null 2>&1 && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    MODE="docker"
  else
    MODE="ssh"
  fi
fi

case "$MODE" in
  local)
    runner=(bash)
    ;;
  docker)
    runner=(docker exec -i "$CONTAINER" bash)
    ;;
  ssh)
    runner=(ssh "$REMOTE_HOST" "docker exec -i $CONTAINER bash")
    ;;
  *)
    echo "MODE must be auto, local, docker, or ssh; got: $MODE" >&2
    exit 1
    ;;
esac

echo "mode=$MODE repo=$REMOTE_REPO port=$PORT force_restart=$FORCE_RESTART" >&2
"${runner[@]}" <<SH
set -euo pipefail
cd "$REMOTE_REPO"

PORT="$PORT"
FORCE_RESTART="$FORCE_RESTART"
MODEL_PATH="/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315"
CODE2WAV_PATH="\$MODEL_PATH/qwen3_5_omni_codec_decode_online_0306"

existing_pid="\$(ps -eo pid,args | awk -v port="\$PORT" '/sglang_omni.cli serve/ && index(\$0, "--port " port) && !/awk/ {print \$1; exit}')"
if [ -n "\$existing_pid" ] && [ "\$FORCE_RESTART" != "1" ]; then
  echo "Stable server is already running on port \$PORT: pid=\$existing_pid"
  curl -fsS "http://127.0.0.1:\$PORT/v1/models" >/tmp/qwen35_stable_models.json
  cat /tmp/qwen35_stable_models.json
  echo
  exit 0
fi

if [ -n "\$existing_pid" ] && [ "\$FORCE_RESTART" = "1" ]; then
  echo "Stopping existing stable server on port \$PORT: pid=\$existing_pid"
  stop_pids="\$(ps -eo pid,ppid,args | awk -v p="\$existing_pid" '\$1==p || \$2==p {print \$1}')"
  if [ -n "\$stop_pids" ]; then
    kill \$stop_pids || true
  fi
  for _ in \$(seq 1 30); do
    remaining="\$(ps -eo pid,ppid,args | awk -v p="\$existing_pid" '\$1==p || \$2==p {print \$1}')"
    if [ -z "\$remaining" ]; then
      break
    fi
    sleep 1
  done
  remaining="\$(ps -eo pid,ppid,args | awk -v p="\$existing_pid" '\$1==p || \$2==p {print \$1}')"
  if [ -n "\$remaining" ]; then
    echo "Force stopping existing stable server process tree on port \$PORT: \$remaining"
    kill -9 \$remaining || true
  fi
fi

python - "\$PORT" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
for _ in range(30):
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError:
        sock.close()
        break
    sock.close()
    time.sleep(1)
else:
    raise SystemExit(f"port {port} is still busy")
PY

RUN_DIR="results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_\${PORT}_\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$RUN_DIR"
cat > "\$RUN_DIR/config.env" <<'EOF'
CUDA_VISIBLE_DEVICES=3,4,5
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE=8
SGLANG_OMNI_TALKER_READY_SUBSET_MIN_SIZE=1
SGLANG_OMNI_RELAY_SLOT_SIZE_MB=1024
SGLANG_OMNI_RELAY_CREDITS=2
SGLANG_OMNI_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES=268435456
SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=2048
SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=68719476736
SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1
QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
QWEN35_TALKER_ALLOW_PARTIAL_TEXT_CHUNK_BEFORE_DONE=0
QWEN35_TALKER_PARTIAL_TEXT_CHUNK_WAIT_SKIPS=0
QWEN35_TALKER_ALLOW_UNBOUNDED_EMPTY_TEXT_FEEDBACK=0
thinker_mem_fraction_static=0.80
thinker_max_running_requests=12
talker_max_running_requests=12
code2wav_stream_chunk_size=4
commit=6115ffd
EOF

(
  export CUDA_VISIBLE_DEVICES=3,4,5
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  export SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE=8
  export SGLANG_OMNI_TALKER_READY_SUBSET_MIN_SIZE=1
  export SGLANG_OMNI_RELAY_SLOT_SIZE_MB=1024
  export SGLANG_OMNI_RELAY_CREDITS=2
  export SGLANG_OMNI_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES=268435456
  export SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=2048
  export SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=68719476736
  export SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0
  export SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1
  export QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
  export QWEN35_TALKER_ALLOW_PARTIAL_TEXT_CHUNK_BEFORE_DONE=0
  export QWEN35_TALKER_PARTIAL_TEXT_CHUNK_WAIT_SKIPS=0
  export QWEN35_TALKER_ALLOW_UNBOUNDED_EMPTY_TEXT_FEEDBACK=0
  export PYTHONPATH=.
  nohup python -m sglang_omni.cli serve \\
    --model-path "\$MODEL_PATH" \\
    --model-name qwen3_5-omni \\
    --host 127.0.0.1 \\
    --port "\$PORT" \\
    --voice-type m02 \\
    --max-tokens 512 \\
    --seed 3408 \\
    --thinker-gpus 0 \\
    --talker-gpu 1 \\
    --code2wav-gpu 2 \\
    --thinker-max-seq-len 192000 \\
    --code2wav-model-path "\$CODE2WAV_PATH" \\
    --prefix-caching on \\
    --thinker-mem-fraction-static 0.80 \\
    --relay-backend nixl \\
    --code2wav-stream-chunk-size 4 \\
    --thinker-cuda-graph off \\
    --talker-cuda-graph on \\
    --talker-torch-compile on \\
    --thinker-max-running-requests 12 \\
    --talker-max-running-requests 12 \\
    > "\$RUN_DIR/server.log" 2>&1 &
  echo \$! > "\$RUN_DIR/server.pid"
)

echo "\$PORT" > "\$RUN_DIR/port.txt"
echo "RUN_DIR=\$RUN_DIR"
echo "PID=\$(cat "\$RUN_DIR/server.pid")"

for i in \$(seq 1 240); do
  if grep -q 'Using port .* instead' "\$RUN_DIR/server.log" 2>/dev/null; then
    echo "Stable server fell back to a different port; expected \$PORT" >&2
    tail -n 40 "\$RUN_DIR/server.log" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:\$PORT/v1/models" >/tmp/qwen35_stable_models.json 2>/tmp/qwen35_stable_curl.err; then
    echo "ready_after=\${i}s"
    cat /tmp/qwen35_stable_models.json
    echo
    exit 0
  fi
  if ! kill -0 "\$(cat "\$RUN_DIR/server.pid")" 2>/dev/null; then
    echo "Stable server exited during startup" >&2
    tail -n 160 "\$RUN_DIR/server.log" >&2
    exit 1
  fi
  if [ \$((i % 30)) -eq 0 ]; then
    echo "waiting \${i}s"
    tail -n 12 "\$RUN_DIR/server.log" || true
  fi
  sleep 1
done

echo "Stable server did not become ready within timeout" >&2
tail -n 200 "\$RUN_DIR/server.log" >&2
exit 1
SH
