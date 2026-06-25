# Qwen3.5-Omni Video-AMME Performance Report

Date: 2026-06-19

This report captures the local Qwen3.5-Omni Video-AMME ci-50 validation used to
compare the vLLM baseline with the SGLang-Omni implementation. It is an internal
engineering report, not a public benchmark claim.

## Scope

- Model: `qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`
- Task: Video-AMME ci-50, video input + spoken audio question, text + audio output
- Dataset repo: `zhaochenyang20/Video_AMME_ci`
- Sample count: 50
- Benchmark concurrency: 4
- Decode: `max_tokens=256`, `temperature=0.0`
- SGLang endpoint: OpenAI-compatible `/v1/chat/completions`

## vLLM Baseline Environment

The Qwen3.5-capable vLLM image used for the optimized baseline was:

```text
tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589
```

The stock image contained `torch 2.8.0a0+nv25.06` and `triton 3.6.0`. That
combination failed a minimal CUDA `torch.compile` test with:

```text
ImportError: cannot import name 'triton_key' from triton.compiler.compiler
```

For compile-enabled vLLM benchmarking, `triton==3.3.1` was installed before
engine creation. With that pin, the vLLM logs confirmed:

- `torch.compile ok`
- `enforce_eager=False`
- `VLLM_COMPILE`
- `cudagraph_mode=FULL_AND_PIECEWISE`
- talker code predictor CUDA graph enabled
- code2wav `torch.compile` enabled

The full ci-50 vLLM baseline artifact root is:

```text
/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106
```

## SGLang Optimized Configuration

The optimized SGLang run used the container-mounted repo at `/myapp/sglang-omni`
with:

```bash
NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 4 --talker-max-running-requests 4" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

The stable/profile benchmark was rerun after the subtalker seed-sentinel fix and
a warmup pass. The artifact root is:

```text
/home/gangouyu/sglang-omni/results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046
```

## Primary Result

Warmup-excluded ci-50 comparison, skipping the first 4 requests:

| Runtime | n | Accuracy | Latency mean | Latency p95 | RTF mean | RTF p95 | WER corpus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM optimized | 46 | 63.0% | 2.093s | 3.525s | 1.4677 | 3.0717 | 7.44% |
| SGLang optimized | 46 | 67.4% | 1.743s | 3.328s | 1.3536 | 2.4023 | 4.12% |

Full stable SGLang ci-50 result:

| Runtime | n | Accuracy | Latency mean | Latency p95 | RTF mean | RTF p95 | WER corpus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang optimized | 50 | 70.0% | 1.801s | 3.615s | 1.4109 | 2.4998 | 4.12% |

The optimized SGLang run is performance-aligned with the optimized vLLM baseline
on the warmed ci-50 slice, with slightly better latency mean/p95 and comparable
audio quality by Whisper large-v3 WER.

## SGLang Bottleneck Breakdown

Stable SGLang profile, 50 requests:

| Stage / interval | Avg | P95 | Interpretation |
| --- | ---: | ---: | --- |
| `talker_ar stage_input_received->stage_complete` | 601ms | 2189ms | Dominant steady-state stage, driven by longer generated speech. |
| `preprocessing stage_input_received->stage_complete` | 450ms | 988ms | Includes server-side video/audio preprocessing. |
| `thinker stage_input_received->stage_complete` | 115ms | 201ms | Not the main mean-latency cost. |
| `talker_code_predictor` | 53ms | 67ms | Improved by subtalker `torch.compile`. |
| `code2wav_window_collect` | 43ms | 126ms | Mostly waiting/backpressure for talker chunks. |
| `code2wav_decode` | 17ms | 23ms | Actual vocoder decode is small after compile. |
| `code2wav stage_input_received->stage_complete` | 14ms | 32ms | Not a bottleneck. |

Per-request timeline correlation with end-to-end profile time:

| Feature | Correlation |
| --- | ---: |
| `talker_ar stage_input_received->stage_complete` | 0.939 |
| code2wav windows/request | 0.922 |
| talker emitted chunks/request | 0.921 |
| thinker text chunks/request | 0.891 |
| `thinker stage_input_received->stage_complete` | 0.805 |
| `preprocessing stage_input_received->stage_complete` | 0.396 |
| `code2wav stage_input_received->stage_complete` | 0.200 |

Conclusion: the remaining steady-state bottleneck is talker AR for longer
generated speech. Code2wav compute is already small; `code2wav_window_collect`
mainly reflects downstream waiting for incoming codec chunks.

## Caveat

The SGLang subtalker `torch.compile` path improves warmed steady-state
performance, but it has high cold-start and first-request compile cost. The
current production recipe should include explicit warmup or a narrower
compile/capture strategy before enabling this path by default in latency-sensitive
deployments.

## Verification

Relevant unit tests:

```bash
docker exec b5f665f3d883 bash -lc \
  'cd /myapp/sglang-omni && python -m pytest tests/unit_test/qwen3_5_omni tests/unit_test/qwen3_omni/test_code2wav.py tests/unit_test/profiler/test_views.py -q'
```

Result:

```text
567 passed, 2 warnings
```

Detailed local report:

```text
/home/gangouyu/sglang-omni/results/qwen35_videoamme_perf_alignment_report_20260618.md
```
