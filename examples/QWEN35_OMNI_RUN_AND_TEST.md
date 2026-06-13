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

For the current two-GPU debug layout:

- thinker/audio/image stages use GPU 0
- talker/code2wav stages use GPU 1
- code2wav torch.compile is disabled for faster debug startup
- code2wav uses 4-codec streaming chunks to match vLLM Qwen3.5 50 Hz output

Use the wrapper:

```bash
bash scripts/launch_qwen35_omni_sglang_server.sh
```

Equivalent important arguments:

```bash
python examples/run_qwen3_5_omni_speech_server.py \
  --model-path /myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315 \
  --model-name qwen3_5-omni \
  --host 127.0.0.1 \
  --port 8161 \
  --voice-type m02 \
  --max-tokens 256 \
  --seed 3408 \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --thinker-max-seq-len 192000 \
  --code2wav-model-path /myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315/qwen3_5_omni_codec_decode_online_0306 \
  --no-code2wav-torch-compile
```

## vLLM Startup

vLLM is not kept as a long-running server in these tests. The alignment driver
starts a Docker worker, builds `OmniLLMEngine`, runs one request, writes
artifacts, then exits.

The worker is launched by:

```bash
scripts/qwen35_omni_alignment.py --backend vllm
```

The compare wrapper passes the vLLM runtime flags used during alignment:

- `VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS=True`
- `VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING=True`
- `VLLM_OMNI_USE_V35_RTC_PROMPT_STYLE=True`
- `VLLM_OMNI_REALTIME_MM_METADATA=True`
- `VLLM_OMNI_T2C_USE_ZMQ=False`
- `VLLM_OMNI_T2T_USE_ZMQ=False`
- `VLLM_ENABLE_TORCH_COMPILE=False`
- `--disable-vllm-mtp`
- `--vllm-enforce-eager`
- `--vllm-default-talker-params`

## Audio-Only Test

The audio-only test uses:

- System prompt: `你是一个智能手机女助手，用温柔的声音和用户聊天`
- User prompt: empty
- User content: audio only
- Input audio text: `今天我心情不好，请给我讲个笑话。`

Run:

```bash
bash scripts/run_qwen35_omni_audio_only_compare.sh
```

The wrapper generates:

- `results/qwen35_audio_only_joke_m02/inputs/user_audio.wav`
- `results/qwen35_audio_only_joke_m02/input_asr.json`
- `results/qwen35_audio_only_joke_m02/compare/vllm_output.wav`
- `results/qwen35_audio_only_joke_m02/compare/sglang_output.wav`
- `results/qwen35_audio_only_joke_m02/compare/alignment_summary.json`
- `results/qwen35_audio_only_joke_m02/compare/alignment_report.md`

The wrapper allows alignment failure by default because open-ended joke
responses can be valid but wording/length can diverge. Set
`ALLOW_ALIGNMENT_FAILURE=0` if a nonzero alignment decision should fail CI.

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

For each compare run, check:

- `alignment_summary.json`: machine-readable inputs, texts, durations, ASR
- `alignment_report.md`: human-readable summary
- `vllm_docker_worker.log`: vLLM engine startup and talker/code2wav traces
- `sglang_server.log`: SGLang stage startup and decode progress
- `vllm_output.wav` and `sglang_output.wav`: listen to the generated speech

