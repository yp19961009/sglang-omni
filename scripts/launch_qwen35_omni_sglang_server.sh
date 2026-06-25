#!/usr/bin/env bash
# Launch the Qwen3.5-Omni speech server inside the shared development
# container.
#
# Common usage:
#   bash scripts/launch_qwen35_omni_sglang_server.sh
#
# Useful overrides:
#   PORT=8161 VOICE_TYPE=m02 bash scripts/launch_qwen35_omni_sglang_server.sh
#   PREFIX_CACHING=off bash scripts/launch_qwen35_omni_sglang_server.sh
#   EXTRA_ARGS="--code2wav-stream-chunk-size 4" bash scripts/launch_...

set -euo pipefail

CONTAINER="${CONTAINER:-b5f665f3d883}"
WORKDIR="${WORKDIR:-/myapp/sglang-omni}"
MODEL_PATH="${MODEL_PATH:-/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315}"
CODE2WAV_PATH="${CODE2WAV_PATH:-$MODEL_PATH/qwen3_5_omni_codec_decode_online_0306}"
MODEL_NAME="${MODEL_NAME:-qwen3_5-omni}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8161}"
VOICE_TYPE="${VOICE_TYPE:-m02}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SEED="${SEED:-3408}"
GPU_THINKER="${GPU_THINKER:-0}"
GPU_TALKER="${GPU_TALKER:-1}"
GPU_CODE2WAV="${GPU_CODE2WAV:-1}"
THINKER_MAX_SEQ_LEN="${THINKER_MAX_SEQ_LEN:-192000}"
PREFIX_CACHING="${PREFIX_CACHING:-on}"
NO_CODE2WAV_TORCH_COMPILE="${NO_CODE2WAV_TORCH_COMPILE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
THINKER_MEM_FRACTION_STATIC="${THINKER_MEM_FRACTION_STATIC:-}"
TALKER_MEM_FRACTION_STATIC="${TALKER_MEM_FRACTION_STATIC:-}"
ENCODER_MEM_RESERVE="${ENCODER_MEM_RESERVE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Qwen3.5 RTC emits tiny decode-token stream chunks alongside larger talker
# chunks. Keep those CPU-only chunks on the control plane to avoid relay queue
# tails without biasing the whole stage control plane away from audio work.
export SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES="${SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES:-4096}"

server_args=(
  python
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
  --prefix-caching "$PREFIX_CACHING"
)

if [[ "$NO_CODE2WAV_TORCH_COMPILE" == "1" ]]; then
  # Disabling code2wav torch.compile makes bring-up runs start faster. Set this
  # env var to 0 for a closer performance run.
  server_args+=(--no-code2wav-torch-compile)
fi

if [[ -n "$MEM_FRACTION_STATIC" ]]; then
  server_args+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
fi

if [[ -n "$THINKER_MEM_FRACTION_STATIC" ]]; then
  server_args+=(--thinker-mem-fraction-static "$THINKER_MEM_FRACTION_STATIC")
fi

if [[ -n "$TALKER_MEM_FRACTION_STATIC" ]]; then
  server_args+=(--talker-mem-fraction-static "$TALKER_MEM_FRACTION_STATIC")
fi

if [[ -n "$ENCODER_MEM_RESERVE" ]]; then
  server_args+=(--encoder-mem-reserve "$ENCODER_MEM_RESERVE")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  server_args+=("${extra_args_array[@]}")
fi

docker_env_args=()
for env_name in $(env | sed -n 's/^\([^=][A-Za-z0-9_]*\)=.*/\1/p'); do
  case "$env_name" in
    QWEN35_* | SGLANG_OMNI_* | RELAY_BACKEND | PREPROCESSING_MAX_CONCURRENCY | TALKER_PARTIAL_START_MIN_CHUNKS | TORCHDYNAMO_DISABLE | PYTORCH_CUDA_ALLOC_CONF)
      docker_env_args+=(-e "$env_name=${!env_name}")
      ;;
  esac
done

printf -v quoted_workdir "%q" "$WORKDIR"
printf -v quoted_server_cmd "%q " "${server_args[@]}"

echo "[qwen35] launching SGLang server in container=$CONTAINER"
echo "[qwen35] model=$MODEL_PATH"
echo "[qwen35] listen=http://$HOST:$PORT voice=$VOICE_TYPE prefix_caching=$PREFIX_CACHING"
echo "[qwen35] stream_inline_cpu_max_bytes=$SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES"

docker exec "${docker_env_args[@]}" "$CONTAINER" bash -lc \
  "cd $quoted_workdir && \
   export PYTHONPATH=$quoted_workdir:\${PYTHONPATH:-} && \
   export TORCHDYNAMO_DISABLE=\${TORCHDYNAMO_DISABLE:-1} && \
   export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=\${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1} && \
   exec $quoted_server_cmd"
