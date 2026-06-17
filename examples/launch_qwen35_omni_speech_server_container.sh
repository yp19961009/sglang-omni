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
#   EXTRA_ARGS="--code2wav-stream-chunk-size 4" bash examples/launch_qwen35_omni_speech_server_container.sh

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

# Keep this at 4 for the current speech profile. It matches the 50 Hz
# Qwen3.5-Omni code2wav streaming cadence.
CODE2WAV_STREAM_CHUNK_SIZE="${CODE2WAV_STREAM_CHUNK_SIZE:-4}"

# Keep prefix/radix cache enabled by default. Qwen3.5 uses Qwen3-style stable
# media pad tokens plus Mamba-aware radix cache for safe hybrid-SSM prefix hits.
PREFIX_CACHING="${PREFIX_CACHING:-on}"

# Disable code2wav torch.compile by default. Startup is faster while reviewing
# functionality. Set NO_CODE2WAV_TORCH_COMPILE=0 for performance runs.
NO_CODE2WAV_TORCH_COMPILE="${NO_CODE2WAV_TORCH_COMPILE:-1}"

# Extra CLI flags are appended as-is. Keep this for quick one-off experiments
# without editing the script.
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
  --code2wav-stream-chunk-size "$CODE2WAV_STREAM_CHUNK_SIZE"
  --prefix-caching "$PREFIX_CACHING"
)

if [[ "$NO_CODE2WAV_TORCH_COMPILE" == "1" ]]; then
  server_args+=(--no-code2wav-torch-compile)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  server_args+=("${extra_args_array[@]}")
fi

echo "[qwen35] repo root: $REPO_ROOT"
echo "[qwen35] model: $MODEL_PATH"
echo "[qwen35] code2wav: $CODE2WAV_PATH"
echo "[qwen35] listen: http://$HOST:$PORT"
echo "[qwen35] voice=$VOICE_TYPE seed=$SEED max_tokens=$MAX_TOKENS prefix_caching=$PREFIX_CACHING"
echo "[qwen35] gpu_thinker=$GPU_THINKER gpu_talker=$GPU_TALKER gpu_code2wav=$GPU_CODE2WAV"
echo "[qwen35] starting server..."

exec "${server_args[@]}"
