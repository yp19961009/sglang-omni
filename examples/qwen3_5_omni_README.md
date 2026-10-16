# Qwen3.5-Omni Native SGLang Usage

This page documents the native `sglang_omni.cli serve` path for Qwen3.5-Omni.
The historical example launchers are kept only as thin wrappers around the same
CLI, so all new scripts should call the CLI directly.

## Speech Server

```bash
MODEL=/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315

python -m sglang_omni.cli serve \
  --model-path "$MODEL" \
  --model-name qwen3_5-omni \
  --host 127.0.0.1 \
  --port 8161 \
  --voice-type m02 \
  --max-tokens 512 \
  --seed 3408 \
  --thinker-gpus 0 \
  --talker-gpu 1 \
  --code2wav-gpu 1 \
  --thinker-max-seq-len 192000 \
  --code2wav-model-path "$MODEL/qwen3_5_omni_codec_decode_online_0306" \
  --prefix-caching on \
  --no-code2wav-torch-compile
```

The container helper in `examples/launch_qwen35_omni_speech_server_container.sh`
uses the same entrypoint and flags.

For Qwen3.5 speech/audio requests, prefix cache is enabled by default. The
request builder keeps Qwen3-style stable media-key pad tokens, and the thinker
uses Mamba-aware radix cache for safe hybrid-SSM prefix hits.

## Text-Only Server

```bash
python -m sglang_omni.cli serve \
  --text-only \
  --model-path /myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315 \
  --host 127.0.0.1 \
  --port 8161 \
  --thinker-gpus 0
```

## Request Shape

Use the OpenAI chat-completions shape supported by sglang-omni. Audio output is
requested with `modalities` plus the `audio` object, and returned in
`choices[0].message.audio`.

```bash
AUDIO_B64=$(base64 -w0 /path/to/input.wav)

curl http://127.0.0.1:8161/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3_5-omni\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"你是一个智能手机女助手，用温柔的声音和用户聊天\"},
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"input_audio\",
            \"input_audio\": {
              \"data\": \"$AUDIO_B64\",
              \"format\": \"wav\"
            }
          }
        ]
      }
    ],
    \"modalities\": [\"text\", \"audio\"],
    \"audio\": {\"voice\": \"m02\", \"format\": \"wav\"},
    \"max_tokens\": 512,
    \"temperature\": 0.000001,
    \"top_k\": 1,
    \"top_p\": 0.8,
    \"seed\": 3408,
    \"stream\": false
  }"
```

For native sglang-omni clients, top-level `audios`, `images`, and `videos` are
also supported by the preprocessing stage.

## Preflight

Run the local checkpoint preflight before launch when bringing up a new model
directory:

```bash
MODEL=/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315

python scripts/qwen35_omni_preflight.py \
  --model-path "$MODEL" \
  --code2wav-model-path "$MODEL/qwen3_5_omni_codec_decode_online_0306"
```

## Useful Native Flags

- `--max-running-requests`, `--thinker-max-running-requests`
- `--talker-max-running-requests`
- `--max-prefill-tokens`
- `--page-size`
- `--prefix-caching default|on|off`
- `--chunked-prefill default|on|off`
- `--thinker-cuda-graph default|on|off`
- `--talker-cuda-graph default|on|off`
- `--mem-fraction-static`, `--thinker-mem-fraction-static`
- `--talker-mem-fraction-static`
- `--limit-mm-per-prompt`, `--limit-mm-per-prompt-audio`
- `--limit-mm-per-prompt-image`, `--limit-mm-per-prompt-video`
- `--image-min-pixels`, `--image-max-pixels`, `--video-fps`
- `--video-max-frames`, `--video-max-pixels`
