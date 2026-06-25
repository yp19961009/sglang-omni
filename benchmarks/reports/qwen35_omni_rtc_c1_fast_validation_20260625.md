# Qwen3.5-Omni RTC C1 Fast-Path Validation

Date: 2026-06-25

## Summary

The Qwen3.5-Omni 40-trunk realtime C1 fast path was validated against the
existing optimized vLLM RTC C1 run. SGLang-Omni is in the same latency band as
vLLM for first text and first audio under this single-session cache-reuse
profile.

| Runtime | Run | TTFT ms | TTFA ms | E2E ms |
| --- | --- | ---: | ---: | ---: |
| SGLang-Omni | `results/sglang_rtc_trunk40_c1_fast_validate_20260625_094658/measure_batch0` | 443.38 | 591.16 | 2368.27 |
| vLLM | `results/vllm_rtc_trunk40_c1_rerun_wheel_20260625_171326/rtc_output/rtc_20260625_172544/trunk_40_conc_1` | 356.22 | 542.60 | 2167.80 |

## SGLang Fast-Path Configuration

```bash
RELAY_BACKEND=nixl
TALKER_PARTIAL_START_MIN_CHUNKS=3
NO_CODE2WAV_TORCH_COMPILE=0
TORCHDYNAMO_DISABLE=0
SGLANG_OMNI_RELAY_PAYLOAD_PREP_EXECUTOR=1
SGLANG_OMNI_COLOCATE_PREPROCESSING_WITH_THINKER=1
SGLANG_OMNI_COLOCATE_IMAGE_ENCODER_WITH_THINKER=1
SGLANG_OMNI_COLOCATE_MM_AGGREGATE_WITH_THINKER=1
SGLANG_OMNI_ENCODER_MAX_BATCH_WAIT_MS=0
SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=34359738368
SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=4096
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1
SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0
SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_THINKER_GPU=1
QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA=0
QWEN35_MAMBA_MEDIA_BRANCH_CACHE=1
QWEN35_RTC_PRERUN_PREFILL_ONLY=1
QWEN35_RTC_ISOLATE_PRERUN_PREFILL=1
THINKER_MEM_FRACTION_STATIC=0.72
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8"
```

## Measured SGLang Breakdown

| Stage | Time ms |
| --- | ---: |
| preprocessing | 70.05 |
| image_encoder | 141.39 |
| audio_encoder | 63.36 |
| mm_aggregate | 115.17 |
| thinker total | 1922.35 |
| thinker prefill to first emit | 113.05 |
| talker_ar total | 1942.35 |
| talker prefill to first chunk | 85.80 |
| code2wav final payload | 10.03 |

The server log for the measured request showed a large cache hit:
`#cached-token: 29376` for the final 40-trunk request. That is the expected C1
fast-path behavior and distinguishes this run from the conservative
`limit_prefix_cache_before_media` path.

## Caveat

This validation covers C1/single-session RTC cache reuse. Multi-session
concurrency correctness remains a separate follow-up and should keep using an
explicit safe or auto policy until the fast path is proven with `completed=12`,
`failed=0`, and `bang_count=0`.
