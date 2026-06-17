#!/usr/bin/env bash
# Send one audio-only chat request to the Qwen3.5-Omni SGLang speech server.
#
# Run this inside the container after starting:
#   bash examples/launch_qwen35_omni_speech_server_container.sh
#
# Basic usage:
#   cd /myapp/sglang-omni
#   AUDIO_PATH=/path/to/input.wav bash examples/request_qwen35_omni_audio_only.sh
#
# Or pass the wav path as the first argument:
#   bash examples/request_qwen35_omni_audio_only.sh /path/to/input.wav
#
# The script writes:
#   request.json   - exact request body sent to /v1/chat/completions
#   response.json  - raw server response
#   output.wav     - decoded assistant audio, when present
#   output.txt     - assistant text, when present
#   asr/output.txt - Whisper transcript of output.wav, when RUN_ASR=1
#   asr_compare.json - text/audio transcript similarity scores, when RUN_ASR=1

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8161}"
MODEL="${MODEL:-qwen3_5-omni}"
VOICE="${VOICE:-m02}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是一个智能手机女助手，用温柔的声音和用户聊天}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.000001}"
TOP_K="${TOP_K:-1}"
TOP_P="${TOP_P:-0.8}"
SEED="${SEED:-3408}"
TALKER_MAX_TOKENS="${TALKER_MAX_TOKENS:-2048}"
TALKER_TEMPERATURE="${TALKER_TEMPERATURE:-0.0}"
TALKER_TOP_K="${TALKER_TOP_K:-1}"
TALKER_TOP_P="${TALKER_TOP_P:-1.0}"
TALKER_REPETITION_PENALTY="${TALKER_REPETITION_PENALTY:-1.05}"
TALKER_SEED="${TALKER_SEED:-$SEED}"
SUBTALKER_TEMPERATURE="${SUBTALKER_TEMPERATURE:-0.9}"
SUBTALKER_TOP_K="${SUBTALKER_TOP_K:-50}"
SUBTALKER_TOP_P="${SUBTALKER_TOP_P:-1.0}"
SUBTALKER_REPETITION_PENALTY="${SUBTALKER_REPETITION_PENALTY:-1.05}"
SUBTALKER_SEED="${SUBTALKER_SEED:-$SEED}"
RUN_ASR="${RUN_ASR:-1}"
ASR_MODEL="${ASR_MODEL:-base}"
ASR_DEVICE="${ASR_DEVICE:-cpu}"
ASR_LANGUAGE="${ASR_LANGUAGE:-en}"
ASR_MIN_WORD_F1="${ASR_MIN_WORD_F1:-0.80}"
ASR_FAIL_ON_MISMATCH="${ASR_FAIL_ON_MISMATCH:-0}"

AUDIO_PATH="${1:-${AUDIO_PATH:-}}"
OUT_DIR="${OUT_DIR:-/tmp/qwen35_omni_audio_only_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$AUDIO_PATH" ]]; then
  echo "usage: AUDIO_PATH=/path/to/input.wav bash $0" >&2
  echo "   or: bash $0 /path/to/input.wav" >&2
  exit 2
fi

if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "[qwen35] input audio not found: $AUDIO_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

REQUEST_JSON="$OUT_DIR/request.json"
RESPONSE_JSON="$OUT_DIR/response.json"
OUTPUT_WAV="$OUT_DIR/output.wav"
OUTPUT_TXT="$OUT_DIR/output.txt"
ASR_DIR="$OUT_DIR/asr"
ASR_COMPARE_JSON="$OUT_DIR/asr_compare.json"
AUDIO_B64_FILE="$OUT_DIR/input_audio.b64"

echo "[qwen35] audio: $AUDIO_PATH"
echo "[qwen35] url: $BASE_URL/v1/chat/completions"
echo "[qwen35] output dir: $OUT_DIR"
echo "[qwen35] asr check: $RUN_ASR model=$ASR_MODEL device=$ASR_DEVICE language=$ASR_LANGUAGE"

base64 -w0 "$AUDIO_PATH" > "$AUDIO_B64_FILE"

export MODEL VOICE SYSTEM_PROMPT MAX_TOKENS TEMPERATURE TOP_K TOP_P SEED
export TALKER_MAX_TOKENS TALKER_TEMPERATURE TALKER_TOP_K TALKER_TOP_P
export TALKER_REPETITION_PENALTY TALKER_SEED
export SUBTALKER_TEMPERATURE SUBTALKER_TOP_K SUBTALKER_TOP_P
export SUBTALKER_REPETITION_PENALTY SUBTALKER_SEED AUDIO_B64_FILE
python - <<'PY' > "$REQUEST_JSON"
import json
import os
import sys

with open(os.environ["AUDIO_B64_FILE"], encoding="ascii") as f:
    audio_b64 = f.read()

body = {
    "model": os.environ["MODEL"],
    "messages": [
        {
            "role": "system",
            "content": os.environ["SYSTEM_PROMPT"],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                }
            ],
        },
    ],
    "modalities": ["text", "audio"],
    "audio": {
        "voice": os.environ["VOICE"],
        "format": "wav",
    },
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "top_k": int(os.environ["TOP_K"]),
    "top_p": float(os.environ["TOP_P"]),
    "seed": int(os.environ["SEED"]),
    "stream": False,
    "talker_params": {
        "max_tokens": int(os.environ["TALKER_MAX_TOKENS"]),
        "temperature": float(os.environ["TALKER_TEMPERATURE"]),
        "top_k": int(os.environ["TALKER_TOP_K"]),
        "top_p": float(os.environ["TALKER_TOP_P"]),
        "repetition_penalty": float(os.environ["TALKER_REPETITION_PENALTY"]),
        "seed": int(os.environ["TALKER_SEED"]),
    },
    "subtalker_params": {
        "temperature": float(os.environ["SUBTALKER_TEMPERATURE"]),
        "top_k": int(os.environ["SUBTALKER_TOP_K"]),
        "top_p": float(os.environ["SUBTALKER_TOP_P"]),
        "repetition_penalty": float(os.environ["SUBTALKER_REPETITION_PENALTY"]),
        "seed": int(os.environ["SUBTALKER_SEED"]),
    },
}

json.dump(body, sys.stdout, ensure_ascii=False)
PY

HTTP_STATUS="$(
  curl -sS -w "%{http_code}" "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$REQUEST_JSON" \
  -o "$RESPONSE_JSON"
)"

if (( HTTP_STATUS < 200 || HTTP_STATUS >= 300 )); then
  echo "[qwen35] request failed with HTTP $HTTP_STATUS" >&2
  echo "[qwen35] response: $RESPONSE_JSON" >&2
  sed -n '1,120p' "$RESPONSE_JSON" >&2 || true
  exit 1
fi

export RESPONSE_JSON OUTPUT_WAV OUTPUT_TXT
python - <<'PY'
import base64
import json
import os
import sys
from pathlib import Path

response_path = Path(os.environ["RESPONSE_JSON"])
output_wav = Path(os.environ["OUTPUT_WAV"])
output_txt = Path(os.environ["OUTPUT_TXT"])

try:
    resp = json.loads(response_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    print(f"[qwen35] response is not valid JSON: {exc}", file=sys.stderr)
    print(f"[qwen35] response: {response_path}", file=sys.stderr)
    raise SystemExit(1) from exc

if resp.get("error") is not None:
    print(f"[qwen35] server returned error: {resp['error']}", file=sys.stderr)
    raise SystemExit(1)

message = (resp.get("choices") or [{}])[0].get("message") or {}
text = message.get("content") or ""

audio = message.get("audio") or resp.get("audio") or {}
audio_data = audio.get("data")

if text:
    output_txt.write_text(text, encoding="utf-8")

if audio_data:
    output_wav.write_bytes(base64.b64decode(audio_data))

print("[qwen35] text:")
print(text)
print(f"[qwen35] response: {response_path}")
if audio_data:
    print(f"[qwen35] audio: {output_wav}")
else:
    print("[qwen35] audio: <missing>", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$RUN_ASR" =~ ^(1|true|True|yes|YES|on|ON)$ ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("whisper") else 1)
PY
  then
    echo "[qwen35] asr: python package 'whisper' is not installed; skip audio/text check" >&2
    exit 0
  fi

  mkdir -p "$ASR_DIR"
  python -m whisper "$OUTPUT_WAV" \
    --model "$ASR_MODEL" \
    --device "$ASR_DEVICE" \
    --language "$ASR_LANGUAGE" \
    --fp16 False \
    --temperature 0 \
    --output_dir "$ASR_DIR" \
    --output_format txt

  export OUTPUT_TXT ASR_TRANSCRIPT="$ASR_DIR/output.txt" ASR_COMPARE_JSON ASR_MIN_WORD_F1 ASR_FAIL_ON_MISMATCH
  python - <<'PY'
import collections
import difflib
import json
import os
import re
import sys
from pathlib import Path

text_path = Path(os.environ["OUTPUT_TXT"])
asr_path = Path(os.environ["ASR_TRANSCRIPT"])
compare_path = Path(os.environ["ASR_COMPARE_JSON"])
min_word_f1 = float(os.environ["ASR_MIN_WORD_F1"])
fail_on_mismatch = os.environ["ASR_FAIL_ON_MISMATCH"] in {
    "1",
    "true",
    "True",
    "yes",
    "YES",
    "on",
    "ON",
}

text = text_path.read_text(encoding="utf-8")
asr = asr_path.read_text(encoding="utf-8")

def words(value: str) -> list[str]:
    value = value.lower()
    value = re.sub(r"[*_`#>\[\](){}]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return value.split()

text_words = words(text)
asr_words = words(asr)
text_counter = collections.Counter(text_words)
asr_counter = collections.Counter(asr_words)
overlap = sum((text_counter & asr_counter).values())

precision = overlap / len(asr_words) if asr_words else 0.0
recall = overlap / len(text_words) if text_words else 0.0
word_f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall
    else 0.0
)
word_sequence_ratio = difflib.SequenceMatcher(
    None, text_words, asr_words, autojunk=False
).ratio()

result = {
    "text_words": len(text_words),
    "asr_words": len(asr_words),
    "overlap_words": overlap,
    "precision": round(precision, 6),
    "recall": round(recall, 6),
    "word_f1": round(word_f1, 6),
    "word_sequence_ratio": round(word_sequence_ratio, 6),
    "min_word_f1": min_word_f1,
    "pass": word_f1 >= min_word_f1,
    "text_path": str(text_path),
    "asr_path": str(asr_path),
}
compare_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("[qwen35] asr text:")
print(asr)
print(f"[qwen35] asr compare: {compare_path}")
print(
    "[qwen35] asr scores: "
    f"word_f1={word_f1:.4f} "
    f"word_sequence_ratio={word_sequence_ratio:.4f} "
    f"precision={precision:.4f} "
    f"recall={recall:.4f}"
)

if word_f1 < min_word_f1:
    message = (
        f"[qwen35] asr mismatch: word_f1={word_f1:.4f} "
        f"< ASR_MIN_WORD_F1={min_word_f1:.4f}"
    )
    if fail_on_mismatch:
        print(message, file=sys.stderr)
        raise SystemExit(1)
    print(message, file=sys.stderr)
PY
fi
