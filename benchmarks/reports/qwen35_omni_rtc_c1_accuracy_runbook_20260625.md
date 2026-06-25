# Qwen3.5-Omni RTC C1 Accuracy/Output Runbook

Date: 2026-06-25

This runbook validates the 40-chunk realtime single-concurrency path with
output capture.  The acceptance target is not exact token/waveform equality:
the prompt is open-ended, so the check is non-empty non-degenerate text,
readable generated audio, and comparable TTFT/TTFA/E2E metrics.

## Data

```bash
RTC_TEST_DIR=/home/gangouyu/data/share-data-6batch
```

For the default C1 check:

- measured batch: `video_batches/video_b0`
- chunks: `40`
- each chunk: one 2-second video frame group at `video_fps=1` plus one 2-second
  silence/noise audio segment
- final question audio: `audio_batches/video_input_b0/question_with_noise/2s_question_with_noise_0.wav`
- warmup/cache population: pre-run trunk `1..40`; only the final trunk-40
  request is measured

## Start SGLang-Omni

Run inside the SGLang-Omni container.  The launcher is container-native and
expects the `/myapp` layout.  This starts an OpenAI-compatible server on
`127.0.0.1:8161`.

From the host, enter the container first:

```bash
docker exec -it sglang-omni-dev bash
```

Then start the server:

```bash
cd /myapp/sglang-omni

MODEL_PATH=/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315 \
CODE2WAV_PATH=/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315/qwen3_5_omni_codec_decode_online_0306 \
PORT=8161 \
GPU_THINKER=0 \
GPU_TALKER=1 \
GPU_CODE2WAV=1 \
RELAY_BACKEND=nixl \
TALKER_PARTIAL_START_MIN_CHUNKS=4 \
NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_RELAY_PAYLOAD_PREP_EXECUTOR=1 \
SGLANG_OMNI_COLOCATE_PREPROCESSING_WITH_THINKER=1 \
SGLANG_OMNI_COLOCATE_IMAGE_ENCODER_WITH_THINKER=1 \
SGLANG_OMNI_COLOCATE_MM_AGGREGATE_WITH_THINKER=1 \
SGLANG_OMNI_ENCODER_MAX_BATCH_WAIT_MS=0 \
SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=34359738368 \
SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=4096 \
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1 \
SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0 \
SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_THINKER_GPU=1 \
QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA=0 \
QWEN35_MAMBA_MEDIA_BRANCH_CACHE=1 \
QWEN35_RTC_PRERUN_PREFILL_ONLY=1 \
QWEN35_RTC_ISOLATE_PRERUN_PREFILL=1 \
THINKER_MEM_FRACTION_STATIC=0.72 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

## Run SGLang C1 Check

Use a second terminal after `/v1/models` is reachable.

```bash
cd /home/gangouyu/sglang-omni

RTC_TEST_DIR=/home/gangouyu/data/share-data-6batch \
BASE_URL=http://127.0.0.1:8161 \
RUN_ASR=1 \
benchmarks/eval/qwen35_omni_rtc_c1_accuracy_check.sh
```

Artifacts:

- `sglang_trunk40_c1/metrics.json`: TTFT/TTFA/E2E/stage summary
- `sglang_trunk40_c1/per_request.json`: generated text, wav path, token counts
- `sglang_trunk40_c1/audio/*.wav`: generated audio
- `output_check.md`: output sanity report
- `asr_check.json`: Whisper ASR text, CER, and silence/RMS check when
  `RUN_ASR=1`

## Run vLLM C1 Check

The vLLM example now supports `CAPTURE_OUTPUT=true`, which saves generated
text/audio into `trunk_40_conc_1/per_request.json`.

Important: do not run this from `/home/gangouyu/vllm`.  That source directory
shadows the compiled `vllm` package in the image and can fail with
`No module named vllm._C`.  Run from `/tmp` and call the script by absolute
path.

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v /home/gangouyu:/home/gangouyu \
  vllm-rtc:cuda129_cp312_triton330 \
  bash -lc '
    cd /tmp &&
    OUTPUT_DIR=/home/gangouyu/sglang-omni/results/qwen35_vllm_rtc_c1_capture_$(date +%Y%m%d_%H%M%S) \
    RTC_TEST_DIR=/home/gangouyu/data/share-data-6batch \
    model_path=/home/gangouyu/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315 \
    CODE2WAV_MODEL=/home/gangouyu/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315/qwen3_5_omni_codec_decode_online_0306 \
    TRUNK_SIZES="40" \
    CONCURRENCY="1" \
    TOTAL_SAMPLES=1 \
    STAGGER_MS=0 \
    AUDIO_ONLY=false \
    CAPTURE_OUTPUT=true \
    DO_WARMUP=true \
    WARMUP_TRUNK_SIZE=3 \
    WARMUP_BATCH_IDX=1 \
    WARMUP_SIL_START_IDX=240 \
    WARMUP_VIDEO_START_IDX=480 \
    WARMUP_QUESTION_IDX=4 \
    WARMUP_AUDIO_ONLY=false \
    THINKER_QUANT= \
    MM_CACHE_SHM_GB=30 \
    ENCODER_CACHE_SIZE=0 \
    CODE2WAV_TORCH_COMPILE=true \
    VLLM_ENABLE_TORCH_COMPILE=1 \
    VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=1 \
    VLLM_FLASH_ATTN_USE_UPSTREAM=0 \
    bash /home/gangouyu/vllm/examples/offline_inference/run_rtc_profile.sh
  '
```

Artifacts:

- `rtc_*/comparison.json`: aggregate vLLM timing
- `rtc_*/trunk_40_conc_1/metrics.json`: TTFT/TTFA breakdown
- `rtc_*/trunk_40_conc_1/per_request.json`: profile row plus captured text/wav
- `rtc_*/trunk_40_conc_1/audio/*.wav`: generated audio

## Compare Captured Outputs

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.qwen35_omni_rtc_compare_outputs \
  --sglang /path/to/sglang_trunk40_c1/per_request.json \
  --vllm /path/to/rtc_xxx/trunk_40_conc_1/per_request.json \
  --out /path/to/rtc_c1_output_compare.md \
  --json-out /path/to/rtc_c1_output_compare.json \
  --print
```

## Acceptance

- `status=ok` for both runtimes.
- Text is non-empty and not pure `!`.
- `wav_path` exists and can be opened as WAV.
- ASR CER is <= 0.10 and `silent_seconds=0` for the canonical C1 prompt.
- TTFT/TTFA are in the same latency band as the C1 fast-path validation report.

Current reference from `qwen35_omni_rtc_c1_fast_validation_20260625.md`:

| Runtime | TTFT ms | TTFA ms | E2E ms |
| --- | ---: | ---: | ---: |
| SGLang-Omni | 427.62 | 650.03 | 3174.48 |
| vLLM | 356.22 | 542.60 | 2167.80 |

## 2026-06-25 Fixed C1 Output/ASR Result

This is the local validation after aligning Qwen3.5 external-text chunking with
vLLM: feedback stride is 4, and non-final external text chunks are consumed only
when 4 text rows are available.  The first measured request after a fresh server
start is still discarded because it pays runtime talker/code2wav compile cost.

| Runtime | status | TTFT ms | TTFA ms | E2E ms | audio s | ASR CER | silent s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang-Omni fixed steady | ok | 427.62 | 650.03 | 3174.48 | 8.960 | 0.0217 | 0 |
| SGLang-Omni fixed repeat | ok | 582.17 | 586.35 | 3167.80 | 8.720 | 0.0217 | 0 |
| vLLM reference | ok | 343.39 | 523.22 | 2120.74 | 9.040 | 0.0217 | 0 |

Both runtimes generated the same text:

```text
我叫千问，是阿里巴巴集团旗下的通义实验室自主研发的多模态超大规模语言模型，有什么我可以帮助你的吗？
```

Artifacts from this run:

- SGLang fixed steady metrics: `results/qwen35_rtc_c1_chunkfull_ps4_steady_20260625_122201/sglang_trunk40_c1/metrics.json`
- SGLang fixed steady per request: `results/qwen35_rtc_c1_chunkfull_ps4_steady_20260625_122201/sglang_trunk40_c1/per_request.json`
- SGLang fixed steady audio: `results/qwen35_rtc_c1_chunkfull_ps4_steady_20260625_122201/sglang_trunk40_c1/audio/sg_ck40_4ebf41db-2800-4545-af8e-3cc09cf954f0.wav`
- SGLang fixed steady ASR: `results/qwen35_rtc_c1_chunkfull_ps4_steady_20260625_122201/asr_large_v3.json`
- SGLang fixed repeat metrics: `results/qwen35_rtc_c1_chunkfull_ps4_repeat_20260625_122256/sglang_trunk40_c1/metrics.json`
- SGLang fixed repeat per request: `results/qwen35_rtc_c1_chunkfull_ps4_repeat_20260625_122256/sglang_trunk40_c1/per_request.json`
- SGLang fixed repeat audio: `results/qwen35_rtc_c1_chunkfull_ps4_repeat_20260625_122256/sglang_trunk40_c1/audio/sg_ck40_e370c1cc-ac89-4f50-ac52-53b202c05c1b.wav`
- SGLang fixed repeat ASR: `results/qwen35_rtc_c1_chunkfull_ps4_repeat_20260625_122256/asr_large_v3.json`
- vLLM metrics: `results/qwen35_vllm_rtc_c1_capture_20260625_191113/rtc_20260625_192316/trunk_40_conc_1/metrics.json`
- vLLM per request: `results/qwen35_vllm_rtc_c1_capture_20260625_191113/rtc_20260625_192316/trunk_40_conc_1/per_request.json`
- vLLM audio: `results/qwen35_vllm_rtc_c1_capture_20260625_191113/rtc_20260625_192316/trunk_40_conc_1/audio/av_ck40_96b1e6c1-5396-4798-9189-e69268baa96b.wav`

SGLang first-request note: the first measured audio request after a fresh server
start was intentionally discarded as warmup evidence because it paid runtime
talker/code2wav compile cost.  In the fixed run it measured TTFT 606.86ms,
TTFA 18294.86ms, and E2E 18548.35ms.  The steady-state SGLang rows above are
the comparable C1 results.

vLLM startup note: this run used the local bf16 model because no local fp8
thinker directory was available, so the command sets `THINKER_QUANT=`.  It also
sets `VLLM_FLASH_ATTN_USE_UPSTREAM=0`; with upstream flash-attention enabled,
full CUDA graph capture failed in this image.

## Stage Breakdown Snapshot

| Runtime | component | avg ms | scope |
| --- | --- | ---: | --- |
| SGLang-Omni | thinker lifecycle | 2042.80 | request-wide stage |
| SGLang-Omni | talker AR lifecycle | 2430.42 | request-wide stage |
| SGLang-Omni | preprocessing lifecycle | 31.54 | request-wide stage |
| SGLang-Omni | image encoder lifecycle | 71.43 | request-wide stage |
| SGLang-Omni | mm aggregate lifecycle | 41.57 | request-wide stage |
| SGLang-Omni | code2wav lifecycle | 10.14 | request-wide stage |
| vLLM | thinker prefill | 343.39 | 65.6% of TTFA |
| vLLM | hidden-state queue | 48.52 | 9.3% of TTFA |
| vLLM | embed build | 6.32 | 1.2% of TTFA |
| vLLM | talker prefill | 74.28 | 14.2% of TTFA |
| vLLM | codec accumulation | 36.77 | 7.0% of TTFA |
| vLLM | first code2wav | 13.96 | 2.7% of TTFA |

The SGLang lifecycle stages are request-wide stages from
`request_profile_sglang_rtc_c1_accuracy.json`; the vLLM rows are its explicit
TTFA L1 breakdown from `trunk_40_conc_1/metrics.json`.  They are useful together
for bottleneck diagnosis, but they are not one-to-one stage definitions.
