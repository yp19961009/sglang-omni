#!/usr/bin/env bash
# Start the Qwen3.5-Omni speech server from inside the SGLang container.
#
# This script is intentionally container-native:
#   - run it inside container b5f665f3d883, or any container with the same
#     /myapp layout
#   - do not wrap it with docker exec
#   - keep all commonly tuned values overridable through environment variables
#
# Common usage inside the container:
#   cd /myapp/sglang-omni
#   bash examples/launch_qwen35_omni_speech_server_container.sh
#
# Useful overrides:
#   PORT=8162 VOICE_TYPE=f245 bash examples/launch_qwen35_omni_speech_server_container.sh
#   GPU_THINKER=0 GPU_TALKER=1 GPU_CODE2WAV=1 bash examples/launch_qwen35_omni_speech_server_container.sh
#   PREFIX_CACHING=off bash examples/launch_qwen35_omni_speech_server_container.sh
#   PREPROCESSING_MAX_CONCURRENCY=2 bash examples/launch_qwen35_omni_speech_server_container.sh
#   SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=8589934592 bash examples/launch_qwen35_omni_speech_server_container.sh
#   EXTRA_ARGS="--code2wav-stream-chunk-size 4" bash examples/launch_qwen35_omni_speech_server_container.sh
#
# 40-trunk realtime performance profile:
#   RELAY_BACKEND=nixl TALKER_PARTIAL_START_MIN_CHUNKS=4 \
#   NO_CODE2WAV_TORCH_COMPILE=0 TORCHDYNAMO_DISABLE=0 \
#   SGLANG_OMNI_RELAY_PAYLOAD_PREP_EXECUTOR=1 \
#   SGLANG_OMNI_COLOCATE_PREPROCESSING_WITH_THINKER=1 \
#   SGLANG_OMNI_COLOCATE_IMAGE_ENCODER_WITH_THINKER=1 \
#   SGLANG_OMNI_COLOCATE_MM_AGGREGATE_WITH_THINKER=1 \
#   SGLANG_OMNI_ENCODER_MAX_BATCH_WAIT_MS=0 \
#   SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=34359738368 \
#   SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=4096 \
#   SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1 \
#   SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0 \
#   SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_THINKER_GPU=1 \
#   THINKER_MEM_FRACTION_STATIC=0.72 \
#   EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
#   bash examples/launch_qwen35_omni_speech_server_container.sh
#
# Performance benchmark example:
#   AUTO_WARMUP=1 WARMUP_SAMPLES=8 WARMUP_REPO_ID=/path/to/holdout_videoamme WARMUP_CONCURRENCIES="1 8" \
#   WARMUP_MEASURED_REPO_ID=zhaochenyang20/Video_AMME_ci \
#   NO_CODE2WAV_TORCH_COMPILE=0 TORCHDYNAMO_DISABLE=0 \
#   SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
#   SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
#   RELAY_BACKEND=nixl TALKER_PARTIAL_START_MIN_CHUNKS=4 \
#   EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
#   bash examples/launch_qwen35_omni_speech_server_container.sh

set -euo pipefail

# Resolve the repo root from this script location so the script can be launched
# from any current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths for the current container environment.
MODEL_PATH="${MODEL_PATH:-/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315}"
CODE2WAV_PATH="${CODE2WAV_PATH:-$MODEL_PATH/qwen3_5_omni_codec_decode_online_0306}"

# The OpenAI-compatible model name exposed by the server.
MODEL_NAME="${MODEL_NAME:-qwen3_5-omni}"
PYTHON="${PYTHON:-python3}"

# Server listen address. Use HOST=0.0.0.0 if another machine/container needs to
# reach this server through a mapped port.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8161}"

# Generation defaults used by the current Qwen3.5-Omni smoke runs.
VOICE_TYPE="${VOICE_TYPE:-m02}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SEED="${SEED:-3408}"

# Two-GPU layout:
#   GPU 0: thinker and multimodal preprocessing
#   GPU 1: talker and code2wav
# Override these when testing another topology.
GPU_THINKER="${GPU_THINKER:-0}"
GPU_TALKER="${GPU_TALKER:-1}"
GPU_CODE2WAV="${GPU_CODE2WAV:-1}"

# Qwen3.5-Omni needs a longer context than the default SGLang limit.
THINKER_MAX_SEQ_LEN="${THINKER_MAX_SEQ_LEN:-192000}"

# Leave enough headroom on the thinker GPU for colocated audio/image encoders
# and large video prefill batches when CUDA graph is enabled. Set this to an
# empty string to use the pipeline default.
THINKER_MEM_FRACTION_STATIC="${THINKER_MEM_FRACTION_STATIC:-0.84}"

# Keep this at 4 for the current speech profile. It matches the 50 Hz
# Qwen3.5-Omni code2wav streaming cadence.
CODE2WAV_STREAM_CHUNK_SIZE="${CODE2WAV_STREAM_CHUNK_SIZE:-4}"

# Keep prefix/radix cache enabled by default. Qwen3.5 uses Qwen3-style stable
# media pad tokens plus Mamba-aware radix cache for safe hybrid-SSM prefix hits.
PREFIX_CACHING="${PREFIX_CACHING:-on}"

# Optional data-plane relay override. NIXL is useful for large cross-process
# realtime video payloads; leave empty to use the pipeline default.
RELAY_BACKEND="${RELAY_BACKEND:-}"

# Optional partial-start threshold override. Qwen3.5 external-text handoff uses
# 4-token chunks; use at least 4 so the first talker chunk is complete.
TALKER_PARTIAL_START_MIN_CHUNKS="${TALKER_PARTIAL_START_MIN_CHUNKS:-}"

# Server-side preprocessing is serial by default. Higher concurrency is useful
# for experiments but can regress video benchmarks through CPU/memory contention.
PREPROCESSING_MAX_CONCURRENCY="${PREPROCESSING_MAX_CONCURRENCY:-1}"

# Disable code2wav torch.compile by default. Startup is faster while reviewing
# functionality. Set NO_CODE2WAV_TORCH_COMPILE=0 for performance runs.
NO_CODE2WAV_TORCH_COMPILE="${NO_CODE2WAV_TORCH_COMPILE:-1}"

# Extra CLI flags are appended as-is. Keep this for quick one-off experiments
# without editing the script.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Optional startup warmup for performance runs. The default remains off so the
# launcher can still be used as a pure serving entrypoint. Set AUTO_WARMUP=1 to
# start the server in the background, wait for readiness, run discarded
# Video-AMME requests, then keep the server process in the foreground.
AUTO_WARMUP="${AUTO_WARMUP:-0}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-8}"
WARMUP_SAMPLE_OFFSET="${WARMUP_SAMPLE_OFFSET:-}"
WARMUP_ALLOW_MEASURED_OVERLAP="${WARMUP_ALLOW_MEASURED_OVERLAP:-0}"
WARMUP_MEASURED_REPO_ID="${WARMUP_MEASURED_REPO_ID:-zhaochenyang20/Video_AMME_ci}"
WARMUP_MEASURED_SAMPLE_OFFSET="${WARMUP_MEASURED_SAMPLE_OFFSET:-0}"
WARMUP_MEASURED_MAX_SAMPLES="${WARMUP_MEASURED_MAX_SAMPLES:-50}"
WARMUP_CONCURRENCIES="${WARMUP_CONCURRENCIES:-1 8}"
WARMUP_REPO_ID="${WARMUP_REPO_ID:-zhaochenyang20/Video_AMME_ci}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-256}"
WARMUP_VIDEO_FPS="${WARMUP_VIDEO_FPS:-2}"
WARMUP_VIDEO_MAX_FRAMES="${WARMUP_VIDEO_MAX_FRAMES:-128}"
WARMUP_VIDEO_MAX_PIXELS="${WARMUP_VIDEO_MAX_PIXELS:-401408}"
WARMUP_OUTPUT_ROOT="${WARMUP_OUTPUT_ROOT:-results/qwen35_server_warmup_$(date +%Y%m%d_%H%M%S)}"
WARMUP_WAIT_ATTEMPTS="${WARMUP_WAIT_ATTEMPTS:-180}"
WARMUP_WAIT_INTERVAL_S="${WARMUP_WAIT_INTERVAL_S:-10}"
WARMUP_HF_HOME="${WARMUP_HF_HOME:-/myapp/data/videoamme}"
WARMUP_HF_DATASETS_CACHE="${WARMUP_HF_DATASETS_CACHE:-$WARMUP_HF_HOME/datasets}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[qwen35] model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$CODE2WAV_PATH" ]]; then
  echo "[qwen35] code2wav path does not exist: $CODE2WAV_PATH" >&2
  exit 1
fi

cd "$REPO_ROOT"

# Make local source imports win over installed packages, and allow the Qwen3.5
# model to request its longer max sequence length during startup.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"

server_args=(
  "$PYTHON"
  -m
  sglang_omni.cli
  serve
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --voice-type "$VOICE_TYPE"
  --max-tokens "$MAX_TOKENS"
  --seed "$SEED"
  --thinker-gpus "$GPU_THINKER"
  --talker-gpu "$GPU_TALKER"
  --code2wav-gpu "$GPU_CODE2WAV"
  --thinker-max-seq-len "$THINKER_MAX_SEQ_LEN"
  --code2wav-model-path "$CODE2WAV_PATH"
  --code2wav-stream-chunk-size "$CODE2WAV_STREAM_CHUNK_SIZE"
  --prefix-caching "$PREFIX_CACHING"
)

if [[ -n "$THINKER_MEM_FRACTION_STATIC" ]]; then
  server_args+=(--thinker-mem-fraction-static "$THINKER_MEM_FRACTION_STATIC")
fi

if [[ -n "$RELAY_BACKEND" ]]; then
  server_args+=(--relay-backend "$RELAY_BACKEND")
fi

if [[ -n "$TALKER_PARTIAL_START_MIN_CHUNKS" ]]; then
  server_args+=(--talker-partial-start-min-chunks "$TALKER_PARTIAL_START_MIN_CHUNKS")
fi

if [[ "$NO_CODE2WAV_TORCH_COMPILE" == "1" ]]; then
  server_args+=(--no-code2wav-torch-compile)
fi

if [[ -n "$PREPROCESSING_MAX_CONCURRENCY" && "$PREPROCESSING_MAX_CONCURRENCY" != "1" ]]; then
  server_args+=(--stages.0.factory_args.max_concurrency "$PREPROCESSING_MAX_CONCURRENCY")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  server_args+=("${extra_args_array[@]}")
fi

wait_for_server_ready() {
  local url="http://$HOST:$PORT/v1/models"
  local attempt
  for ((attempt = 1; attempt <= WARMUP_WAIT_ATTEMPTS; attempt++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[qwen35] server ready: $url"
      return 0
    fi
    echo "[qwen35] waiting for server warmup readiness at $url ($attempt/$WARMUP_WAIT_ATTEMPTS)"
    sleep "$WARMUP_WAIT_INTERVAL_S"
  done
  echo "[qwen35] server did not become ready before warmup timeout: $url" >&2
  return 1
}

run_startup_warmup() {
  mkdir -p "$WARMUP_OUTPUT_ROOT"
  echo "[qwen35] startup warmup enabled"
  echo "[qwen35] warmup samples=$WARMUP_SAMPLES concurrencies=$WARMUP_CONCURRENCIES"
  echo "[qwen35] warmup repo=$WARMUP_REPO_ID measured_repo=$WARMUP_MEASURED_REPO_ID"
  echo "[qwen35] warmup sample_offset=${WARMUP_SAMPLE_OFFSET:-unset} allow_measured_overlap=$WARMUP_ALLOW_MEASURED_OVERLAP"
  echo "[qwen35] warmup output root=$WARMUP_OUTPUT_ROOT"

  if [[ "$WARMUP_ALLOW_MEASURED_OVERLAP" != "1" && "$WARMUP_REPO_ID" == "$WARMUP_MEASURED_REPO_ID" ]]; then
    if [[ -z "$WARMUP_SAMPLE_OFFSET" ]]; then
      echo "[qwen35] refusing startup warmup without WARMUP_SAMPLE_OFFSET." >&2
      echo "[qwen35] Use a non-overlapping warmup slice/repo, or set WARMUP_ALLOW_MEASURED_OVERLAP=1 only for cache-hot experiments." >&2
      return 1
    fi

    local measured_start measured_end warmup_start warmup_end
    measured_start=$((WARMUP_MEASURED_SAMPLE_OFFSET))
    measured_end=$((WARMUP_MEASURED_SAMPLE_OFFSET + WARMUP_MEASURED_MAX_SAMPLES))
    warmup_start=$((WARMUP_SAMPLE_OFFSET))
    warmup_end=$((WARMUP_SAMPLE_OFFSET + WARMUP_SAMPLES))
    if (( warmup_start < measured_end && measured_start < warmup_end )); then
      echo "[qwen35] refusing startup warmup because it overlaps the measured slice." >&2
      echo "[qwen35] measured=[${measured_start}, ${measured_end}) warmup=[${warmup_start}, ${warmup_end})" >&2
      echo "[qwen35] Use a non-overlapping WARMUP_SAMPLE_OFFSET or set WARMUP_ALLOW_MEASURED_OVERLAP=1 only for cache-hot experiments." >&2
      return 1
    fi
  fi

  export HF_HOME="$WARMUP_HF_HOME"
  export HF_DATASETS_CACHE="$WARMUP_HF_DATASETS_CACHE"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

  local concurrency
  for concurrency in $WARMUP_CONCURRENCIES; do
    local out_dir="$WARMUP_OUTPUT_ROOT/warmup_audio_${WARMUP_SAMPLES}_c${concurrency}_skipwer"
    mkdir -p "$out_dir"
    echo "[qwen35] warmup Video-AMME c=$concurrency samples=$WARMUP_SAMPLES"
    python -m benchmarks.eval.benchmark_omni_videoamme \
      --model "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --repo-id "$WARMUP_REPO_ID" \
      --output-dir "$out_dir" \
      --max-samples "$WARMUP_SAMPLES" \
      --sample-offset "${WARMUP_SAMPLE_OFFSET:-0}" \
      --max-concurrency "$concurrency" \
      --max-tokens "$WARMUP_MAX_TOKENS" \
      --temperature 0.0 \
      --video-fps "$WARMUP_VIDEO_FPS" \
      --video-max-frames "$WARMUP_VIDEO_MAX_FRAMES" \
      --video-max-pixels "$WARMUP_VIDEO_MAX_PIXELS" \
      --enable-audio \
      --audio-voice "$VOICE_TYPE" \
      --skip-wer \
      --disable-tqdm \
      2>&1 | tee "$out_dir/run.log"
  done
  echo "[qwen35] startup warmup complete; measured benchmark can start now"
}

echo "[qwen35] repo root: $REPO_ROOT"
echo "[qwen35] model: $MODEL_PATH"
echo "[qwen35] code2wav: $CODE2WAV_PATH"
echo "[qwen35] listen: http://$HOST:$PORT"
echo "[qwen35] voice=$VOICE_TYPE seed=$SEED max_tokens=$MAX_TOKENS prefix_caching=$PREFIX_CACHING"
echo "[qwen35] gpu_thinker=$GPU_THINKER gpu_talker=$GPU_TALKER gpu_code2wav=$GPU_CODE2WAV"
echo "[qwen35] thinker_mem_fraction_static=${THINKER_MEM_FRACTION_STATIC:-pipeline_default}"
echo "[qwen35] relay_backend=${RELAY_BACKEND:-pipeline_default}"
echo "[qwen35] talker_partial_start_min_chunks=${TALKER_PARTIAL_START_MIN_CHUNKS:-pipeline_default}"
echo "[qwen35] preprocessing_max_concurrency=$PREPROCESSING_MAX_CONCURRENCY"
echo "[qwen35] auto_warmup=$AUTO_WARMUP"
echo "[qwen35] starting server..."

if [[ "$AUTO_WARMUP" == "1" ]]; then
  "${server_args[@]}" &
  server_pid=$!

  cleanup() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup INT TERM EXIT

  wait_for_server_ready
  run_startup_warmup
  echo "[qwen35] server pid=$server_pid remains active after warmup"
  wait "$server_pid"
else
  exec "${server_args[@]}"
fi
