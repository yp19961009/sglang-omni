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
#
# Run with --help for prepared-media client examples and request semantics.

print_usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=2,3 [OPTIONS...] \
    ./examples/run_qwen35_omni_speech_optimized.sh [SGLANG_SERVE_ARGS...]

This launcher must run in the sglang-omni-dev container. On the host,
/home/gangouyu is mounted at /myapp in the container:

  docker exec -it sglang-omni-dev bash
  cd /myapp/sglang-omni
  CUDA_VISIBLE_DEVICES=2,3 MAX_RUNNING_REQUESTS=1 \
    ./examples/run_qwen35_omni_speech_optimized.sh

Important options (environment variables):
  MODEL_PATH             Model directory. Default: Qwen3.5-Omni local model.
  MODEL_NAME             Served model name. Default: qwen35-omni.
  HOST / PORT            Listen address. Default: 127.0.0.1:8011.
  MAX_RUNNING_REQUESTS   Runtime, compile, and CUDA graph max batch size.
                         Default: 1.
  ENCODER_MEM_RESERVE    Encoder-stage memory reserve ratio. Default: 0.30.
  SGLANG_OMNI_PREPROCESSED_MEDIA_CACHE_SIZE
                         Number of path-backed prepared media objects retained
                         by the preprocessing process. Default: 4.

Single prepared-media S2T request (sample 001-1):

  python3 benchmarks/eval/qwen35_omni_single_s2t.py client \
    --engine sglang \
    --base-url http://127.0.0.1:8011 \
    --model-name qwen35-omni \
    --data-root /myapp/data/videoamme \
    --output /myapp/benchmarks/qwen35_s2t_align/manual-preprocessed/results.json \
    --suite custom \
    --sample-id 001-1 \
    --no-cache-probes \
    --video-fps 1 \
    --video-max-frames 128 \
    --video-max-pixels 401408 \
    --preprocessed-video-dir \
      /myapp/data/qwen35_s2t_predecoded_fps1_factor32/videos \
    --preprocessed-audio-dir \
      /myapp/data/qwen35_s2t_predecoded_fps1/audios

Prepared-media Thinker+Talker request and audio output:

  python3 benchmarks/eval/benchmark_omni_videoamme.py \
    --base-url http://127.0.0.1:8011 \
    --model qwen35-omni \
    --max-samples 1 \
    --max-concurrency 1 \
    --max-tokens 256 \
    --video-fps 1 \
    --video-max-frames 128 \
    --video-max-pixels 401408 \
    --preprocessed-video-dir \
      /myapp/data/qwen35_s2t_predecoded_fps1_factor32/videos \
    --preprocessed-audio-dir \
      /myapp/data/qwen35_s2t_predecoded_fps1/audios \
    --reuse-preprocessed-media \
    --enable-audio \
    --skip-wer \
    --output-dir /myapp/benchmarks/qwen35_speech_manual

The HTTP request contract is equivalent to:

  "preprocessed_videos": [
    {"path": "/myapp/data/.../001_fps1p0_frames128_px401408.pt",
     "reuse_loaded": true}
  ],
  "preprocessed_audios": [
    {"path": "/myapp/data/.../001-1_sr16000.pt", "reuse_loaded": true}
  ]

Prepared-media behavior:
  - Video .pt files already contain decoded, sampled, resized frames and FPS.
  - Audio .pt files already contain a decoded 16 kHz waveform.
  - The server skips video decode/frame sampling/resize and audio decode/resample,
    then starts at prepared-media materialization followed by the HF processor.
  - Paths are resolved by the server inside the container, so use /myapp paths,
    not host-side /home/gangouyu paths.
  - Do not send videos together with preprocessed_videos, or audios together
    with preprocessed_audios. The API rejects either combination with HTTP 400.
  - reuse_loaded=true retains path-backed objects for repeated requests. The
    benchmark flag --reuse-preprocessed-media adds it automatically.
  - The video-fps/max-frames/max-pixels client options select the matching cache
    filename; they are not reapplied to already prepared tensors.

All extra command-line arguments are forwarded to `python -m sglang_omni.cli
serve` after the optimized defaults in this script.
EOF
}

case "${1:-}" in
  -h|--help)
    print_usage
    exit 0
    ;;
esac

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
