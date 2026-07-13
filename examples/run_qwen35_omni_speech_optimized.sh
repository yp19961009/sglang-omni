#!/usr/bin/env bash
set -euo pipefail

# Two-GPU Qwen3.5-Omni Thinker+Talker launcher for H20/Hopper.
# Physical GPUs are selected by CUDA_VISIBLE_DEVICES and must stay within 0-3.
# Inside that visibility mask, logical GPU 0 hosts the encoders/Thinker and
# logical GPU 1 hosts Talker/Code2Wav.
#
# Known-good single-request launch:
#   CUDA_VISIBLE_DEVICES=2,3 \
#     ./examples/run_qwen35_omni_speech_optimized.sh
#
# Throughput launch with graph/compile batch ranges aligned through 8:
#   CUDA_VISIBLE_DEVICES=0,1 MAX_RUNNING_REQUESTS=8 \
#     ./examples/run_qwen35_omni_speech_optimized.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples/configs/qwen35_omni_speech_h20.yaml}"
MODEL_PATH="${MODEL_PATH:-/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315}"
MODEL_NAME="${MODEL_NAME:-qwen35-omni}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
ENCODER_MEM_RESERVE="${ENCODER_MEM_RESERVE:-0.30}"
LOG_LEVEL="${LOG_LEVEL:-info}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_OMNI_PREPROCESSED_MEDIA_CACHE_SIZE="${SGLANG_OMNI_PREPROCESSED_MEDIA_CACHE_SIZE:-4}"

if [[ ! "${MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer, got: ${MAX_RUNNING_REQUESTS}" >&2
  exit 2
fi

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_gpus[@]} < 2 )); then
  echo "Qwen3.5-Omni speech mode needs two visible GPUs, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
for gpu in "${visible_gpus[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-3]$ ]]; then
    echo "Only physical GPUs 0-3 are allowed, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Pipeline config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 2
fi

echo "Starting ${MODEL_NAME} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  API: http://${HOST}:${PORT}"
echo "  batch range: 1..${MAX_RUNNING_REQUESTS}"
echo "  Thinker/Talker: torch.compile=on, CUDA graph=on"
echo "  partial-start=on, image encoder same-batch dedup=on"
echo "  radix cache=off, prepared-media cache=${SGLANG_OMNI_PREPROCESSED_MEDIA_CACHE_SIZE}"

exec "${PYTHON_BIN}" -m sglang_omni.cli serve \
  --config "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level "${LOG_LEVEL}" \
  --thinker-gpus 0 \
  --talker-gpu 1 \
  --code2wav-gpu 1 \
  --thinker-cuda-graph on \
  --talker-cuda-graph on \
  --thinker-torch-compile on \
  --talker-torch-compile on \
  --thinker-torch-compile-max-bs "${MAX_RUNNING_REQUESTS}" \
  --talker-torch-compile-max-bs "${MAX_RUNNING_REQUESTS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --cuda-graph-max-bs "${MAX_RUNNING_REQUESTS}" \
  --talker-partial-start on \
  --encoder-mem-reserve "${ENCODER_MEM_RESERVE}" \
  --stages.4.runtime.sglang_server_args.max_running_requests "${MAX_RUNNING_REQUESTS}" \
  --stages.4.runtime.sglang_server_args.disable_radix_cache true \
  "$@"
