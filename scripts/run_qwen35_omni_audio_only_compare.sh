#!/usr/bin/env bash
# Reproduce the audio-only Qwen3.5-Omni comparison used during bring-up.
#
# The script does four things:
#   1. Generate a Chinese input wav in the container.
#   2. Transcribe that input with Whisper to verify what the model will hear.
#   3. Run vLLM and SGLang on the same audio-only request.
#   4. Print a compact summary of text, audio paths, durations, and ASR.
#
# The user message intentionally contains only audio. If PROMPT is empty, the
# alignment driver now sends no text item in the user content.
#
# Common usage:
#   bash scripts/run_qwen35_omni_audio_only_compare.sh
#
# Faster rerun without regenerating input audio:
#   GENERATE_INPUT=0 bash scripts/run_qwen35_omni_audio_only_compare.sh
#
# Only summarize an existing run:
#   RUN_COMPARE=0 GENERATE_INPUT=0 bash scripts/run_qwen35_omni_audio_only_compare.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONTAINER="${CONTAINER:-b5f665f3d883}"
MODEL_PATH="${MODEL_PATH:-/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315}"
MODEL_NAME="${MODEL_NAME:-qwen3_5-omni}"
VOICE_TYPE="${VOICE_TYPE:-m02}"
SEED="${SEED:-3408}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PORT="${PORT:-8161}"

HOST_OUT="${HOST_OUT:-$ROOT/results/qwen35_audio_only_joke_m02}"
CONTAINER_OUT="${CONTAINER_OUT:-/myapp/sglang-omni/results/qwen35_audio_only_joke_m02}"
HOST_AUDIO="$HOST_OUT/inputs/user_audio.wav"
CONTAINER_AUDIO="$CONTAINER_OUT/inputs/user_audio.wav"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是一个智能手机女助手，用温柔的声音和用户聊天}"
AUDIO_TEXT="${AUDIO_TEXT:-今天我心情不好，请给我讲个笑话。}"
PROMPT="${PROMPT:-}"
TTS_VOICE="${TTS_VOICE:-zh-CN-XiaoxiaoNeural}"

VLLM_DOCKER_IMAGE="${VLLM_DOCKER_IMAGE:-tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589}"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages}"
VLLM_DOCKER_MOUNT="${VLLM_DOCKER_MOUNT:-/home/gangouyu:/myapp}"

GENERATE_INPUT="${GENERATE_INPUT:-1}"
RUN_INPUT_ASR="${RUN_INPUT_ASR:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
ALLOW_ALIGNMENT_FAILURE="${ALLOW_ALIGNMENT_FAILURE:-1}"

mkdir -p "$HOST_OUT/inputs"

if [[ "$GENERATE_INPUT" == "1" ]]; then
  echo "[qwen35] generating input audio: $AUDIO_TEXT"
  docker exec \
    -e AUDIO_TEXT="$AUDIO_TEXT" \
    -e TTS_VOICE="$TTS_VOICE" \
    -e OUT_DIR="$CONTAINER_OUT" \
    "$CONTAINER" bash -lc '
set -euo pipefail
mkdir -p "$OUT_DIR/inputs"
if ! python -c "import edge_tts" >/dev/null 2>&1; then
  python -m pip install -q edge-tts
fi
edge-tts \
  --voice "$TTS_VOICE" \
  --text "$AUDIO_TEXT" \
  --write-media "$OUT_DIR/inputs/user_audio.mp3"
ffmpeg -y -hide_banner -loglevel error \
  -i "$OUT_DIR/inputs/user_audio.mp3" \
  -ac 1 -ar 24000 "$OUT_DIR/inputs/user_audio.wav"
python - <<PY
import json
from pathlib import Path
import soundfile as sf

p = Path("$OUT_DIR/inputs/user_audio.wav")
audio, sr = sf.read(p)
print(json.dumps({
    "path": str(p),
    "sample_rate": sr,
    "samples": int(len(audio)),
    "duration_sec": float(len(audio) / sr),
}, ensure_ascii=False))
PY'
fi

if [[ "$RUN_INPUT_ASR" == "1" ]]; then
  echo "[qwen35] transcribing generated input audio"
  docker exec \
    -e AUDIO_PATH="$CONTAINER_AUDIO" \
    -e OUT_DIR="$CONTAINER_OUT" \
    "$CONTAINER" bash -lc '
set -euo pipefail
python - <<PY
import json
from pathlib import Path
import whisper

audio_path = Path("$AUDIO_PATH")
model = whisper.load_model("large-v3")
result = model.transcribe(str(audio_path), language="zh", fp16=True, verbose=False)
payload = {
    "input_audio": str(audio_path),
    "asr": (result.get("text") or "").strip(),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
Path("$OUT_DIR/input_asr.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY'
fi

if [[ "$RUN_COMPARE" == "1" ]]; then
  echo "[qwen35] running vLLM + SGLang compare"
  set +e
  python3 scripts/qwen35_omni_alignment.py \
    --backend compare \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --output-dir "$HOST_OUT/compare" \
    --audio-path "$HOST_AUDIO" \
    --prompt "$PROMPT" \
    --system-prompt "$SYSTEM_PROMPT" \
    --voice-type "$VOICE_TYPE" \
    --seed "$SEED" \
    --max-tokens "$MAX_TOKENS" \
    --vllm-root "$VLLM_ROOT" \
    --vllm-docker-image "$VLLM_DOCKER_IMAGE" \
    --vllm-docker-mount "$VLLM_DOCKER_MOUNT" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_SKIP_TEXT_ENCODE=True" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_T2C_USE_ZMQ=False" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_T2T_USE_ZMQ=False" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS=True" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING=True" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_USE_V35_RTC_PROMPT_STYLE=True" \
    --vllm-docker-extra-arg "-e VLLM_OMNI_REALTIME_MM_METADATA=True" \
    --vllm-docker-extra-arg "-e VLLM_ENABLE_TORCH_COMPILE=False" \
    --disable-vllm-mtp \
    --vllm-enforce-eager \
    --vllm-default-talker-params \
    --sglang-container "$CONTAINER" \
    --sglang-workdir /myapp/sglang-omni \
    --launch-sglang \
    --sglang-port "$PORT" \
    --sglang-gpu-thinker 0 \
    --sglang-gpu-talker 1 \
    --sglang-gpu-code2wav 1 \
    --no-code2wav-torch-compile \
    --asr-backend whisper \
    --asr-container "$CONTAINER" \
    --asr-model large-v3 \
    --asr-language zh
  compare_rc=$?
  set -e
else
  compare_rc=0
fi

HOST_OUT="$HOST_OUT" python3 - <<'PY'
import json
from pathlib import Path
import os

def compact_text(text, width=220):
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 4] + " ..."

root = Path(os.environ["HOST_OUT"])
summary_path = root / "compare" / "alignment_summary.json"
input_asr_path = root / "input_asr.json"

print("\n[qwen35] summary")
if input_asr_path.exists():
    input_asr = json.loads(input_asr_path.read_text(encoding="utf-8"))
    print(f"- input audio: {root / 'inputs' / 'user_audio.wav'}")
    print(f"- input ASR: {input_asr.get('asr', '')}")

if not summary_path.exists():
    print(f"- no alignment summary found: {summary_path}")
    raise SystemExit(0)

data = json.loads(summary_path.read_text(encoding="utf-8"))
decision = data.get("decision") or {}
print(f"- decision: {decision.get('status')} ({decision.get('reason')})")

for name in ("vllm", "sglang"):
    result = (data.get("results") or {}).get(name) or {}
    metrics = result.get("audio_metrics") or {}
    text = compact_text(result.get("text") or "")
    print(f"- {name} text: {text}")
    print(f"- {name} audio: {result.get('audio_path')}")
    print(f"- {name} duration: {metrics.get('duration_sec')}")

asr = data.get("asr") or {}
for name, item in (asr.get("items") or {}).items():
    text = compact_text(item.get("text") or "")
    print(f"- {name} ASR: {text}")

comparison = asr.get("comparison") or {}
if comparison:
    print(f"- ASR CER: {comparison.get('cer')}")

print(f"- report: {root / 'compare' / 'alignment_report.md'}")
PY

if [[ "$compare_rc" -ne 0 && "$ALLOW_ALIGNMENT_FAILURE" != "1" ]]; then
  exit "$compare_rc"
fi
