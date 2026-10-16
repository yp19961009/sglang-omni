# Qwen3.5-Omni Startup And Test Notes

This note records the startup and test flow used for the Qwen3.5-Omni 23B
bring-up in this workspace.

## Default Environment

- Host workspace: `/home/gangouyu/sglang-omni`
- Container: `b5f665f3d883`
- Container workspace: `/myapp/sglang-omni`
- Model: `/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`
- Default voice: `m02`

## SGLang Startup

SGLang is started as an OpenAI-compatible speech server in the container. The
server entrypoint is:

```bash
examples/run_qwen3_5_omni_speech_server.py
```

For the current two-GPU layout:

- thinker/audio/image stages use GPU 0
- talker/code2wav stages use GPU 1
- code2wav torch.compile is disabled for faster startup while reviewing
- code2wav uses 4-codec streaming chunks to match the Qwen3.5 50 Hz output
- prefix/radix cache stays enabled by default; media pad tokens follow Qwen3's
  stable media-key hashing, and Qwen3.5 uses Mamba-aware radix cache for safe
  hybrid-SSM prefix hits

Use the wrapper:

```bash
bash scripts/launch_qwen35_omni_sglang_server.sh
```

Equivalent important arguments:

```bash
python -m sglang_omni.cli serve \
  --model-path /myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315 \
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
  --code2wav-model-path /myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315/qwen3_5_omni_codec_decode_online_0306 \
  --prefix-caching on \
  --no-code2wav-torch-compile
```

## Audio-Only Test

The current audio-only smoke test uses:

- System prompt: `你是一个智能手机女助手，用温柔的声音和用户聊天`
- User prompt: empty
- User content: audio only
- Input audio: `/myapp/data/hzeskbz6.wav`
- Input audio ASR: `What are the names of some famous actors that started their careers on Broadway?`

Run:

```bash
cd /myapp/sglang-omni
OUT_DIR=/tmp/qwen35_user_hzeskbz6 \
AUDIO_PATH=/myapp/data/hzeskbz6.wav \
bash examples/request_qwen35_omni_audio_only.sh
```

The request script generates:

- `/tmp/qwen35_user_hzeskbz6/request.json`
- `/tmp/qwen35_user_hzeskbz6/response.json`
- `/tmp/qwen35_user_hzeskbz6/output.txt`
- `/tmp/qwen35_user_hzeskbz6/output.wav`

## Quick Test Commands

Run focused unit tests in the container:

```bash
docker exec b5f665f3d883 bash -lc '
cd /myapp/sglang-omni &&
pytest -q \
  tests/unit_test/qwen3_5_omni/test_request_builders.py \
  tests/unit_test/qwen3_5_omni/test_talker.py \
  tests/unit_test/qwen3_omni/test_talker.py \
  tests/unit_test/qwen3_5_omni/test_preprocessor.py
'
```

Run the broader Qwen3.5 validation script when the local environment has
`pytest` and all test dependencies:

```bash
bash scripts/validate_qwen35_omni.sh
```

## What To Inspect

For each request run, check:

- `request.json`: exact OpenAI-compatible request body
- `response.json`: raw server response
- `output.txt`: assistant text
- `output.wav`: assistant speech
