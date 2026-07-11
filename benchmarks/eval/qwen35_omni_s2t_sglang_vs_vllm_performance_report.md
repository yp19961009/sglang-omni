# Qwen3.5-Omni S2T: SGLang-Omni vs vLLM Performance Report

Date: 2026-07-11

## 1. Executive summary

This report compares one 001-1 request and 50 repeated 001-1 requests at
rolling concurrency 8.

Main findings:

1. For one request, SGLang reaches the first model token in 0.912 s versus
   1.137 s for vLLM. Using the closest post-media boundary, SGLang is 20.7%
   lower latency.
2. For 50 requests at rolling concurrency 8, SGLang finishes in 35.753 s
   versus 50.323 s for vLLM. SGLang throughput is 1.399 req/s versus
   0.994 req/s, or 40.8% higher.
3. The single-request gap is primarily the interval after HF processing and
   before encoder execution. It is about 64 ms in SGLang and 339 ms in vLLM.
   The vLLM gap is about 275 ms larger.
4. The concurrency gap is primarily architectural. vLLM runs the multimodal
   encoder inside _preprocess() and then runs the thinker forward in the same
   GPU runner step. SGLang exposes image encoder, audio encoder, and thinker as
   independent stages and overlaps them.
5. Encoder kernels alone do not explain the result. For one request, vLLM's
   combined image+audio encoder is slightly faster than SGLang's critical audio
   encoder, 330 ms versus 346 ms.
6. SGLang still has a correctness blocker under concurrency. The repeated
   deterministic request returns C only 33/50 times, while vLLM returns C
   50/50 times.

## 2. Test alignment

Both tests use:

- sample 001-1
- one excluded 001-1 warmup request
- predecoded, sampled, resized media loaded once and reused
- HF processor and encoder execution on every measured request
- thinker-only generation
- one NVIDIA H20 GPU
- FPS=1 and max_tokens=256
- compile and CUDA graph enabled
- prefix/radix cache disabled
- encoder cache hits disabled
- both runtimes configured for up to 8 active sequences; the single-request
  test submits only one measured request
- rolling concurrency, where a completed request is immediately replaced

SGLang additionally disables image-encoder same-batch dedup. vLLM gives each
measured request unique MM UUIDs. All 50 vLLM requests report:

    cached_tokens = 0
    HFPREP cached = 0, miss = 2
    encoder cache hits = 0

The vLLM MM processor cache still has a 4 GB allocation because setting it to
zero triggers an audio-placeholder bug in this branch. Unique UUIDs avoid both
hits and large tensor hashing. The vLLM encoder cache manager also remains
allocated and records 100 misses, 0 hits, evictions, and allocations. This is
not result reuse, but its management overhead remains part of the vLLM result.

## 3. Remaining input and configuration differences

The video tensor is aligned:

| Item | SGLang | vLLM |
| --- | ---: | ---: |
| Shape | [74, 3, 352, 640] | [74, 3, 352, 640] |
| Dtype before HF processing | FP32 | FP32 |
| Value sum | 6,042,146,304 | 6,042,146,304 |
| Video FPS | 1 | 1 |

The source WAV and MP4 hashes match, but the two native audio loaders produce
slightly different waveforms. The resulting prompt lengths are:

| Engine | Prompt tokens |
| --- | ---: |
| SGLang | 8,698 |
| vLLM | 8,763 |

vLLM has 65 more tokens, a 0.75% difference. This is too small to explain a
29% wall-time difference, but the comparison is not token-for-token identical.

Other implementation-level differences retained deliberately:

- SGLang max_seq_len=32768 and its observed thinker prefill chunk is 8192
  tokens.
- vLLM hard-codes max_model_len=192000 and uses
  max_num_batched_tokens=32768.
- SGLang reserves about 64.2 GiB after startup; vLLM reserves about 87.9 GiB.
- The tests run on H20 GPUs on different hosts.

## 4. Single-request result

### 4.1 End-to-end metrics

| Metric | SGLang | vLLM | SGLang change |
| --- | ---: | ---: | ---: |
| Request to first token | 912.405 ms | 1,136.627 ms | -19.7% |
| Post-media to first token | 901.214 ms | 1,136.627 ms | -20.7% |
| Request E2E | 937.600 ms | 1,156.171 ms | -18.9% |
| Prompt tokens | 8,698 | 8,763 | -0.75% |
| Completion tokens | 6 | 4 | +2 tokens |
| Response | Answer: (C) | Answer: C | both correct |

The vLLM benchmark loads the shared .pt prompt in 120.020 ms before its active
timer. SGLang's formal request spends 10.236 ms materializing the already
warmed path-backed media. Therefore post-media to first token is the closest
engine-level comparison.

### 4.2 Critical-path reconstruction

| Segment | SGLang | vLLM | vLLM minus SGLang |
| --- | ---: | ---: | ---: |
| HF processor | 113.808 ms | 117.159 ms | +3.351 ms |
| HF end to critical encoder start | 63.703 ms | about 338.8 ms | about +275.1 ms |
| Critical/combined encoder | 345.883 ms | 330.267 ms | -15.616 ms |
| Encoder end to first token | 377.820 ms | about 350.3 ms | about -27.5 ms |
| Post-media to first token | 901.214 ms | 1,136.627 ms | +235.413 ms |

SGLang's critical encoder is audio. Image encoding takes 305.318 ms and
finishes earlier, while audio takes 345.883 ms. They run in parallel.

The vLLM pre-encoder interval can be split approximately as:

- HF processor completion to EngineCore Request added: 53 ms
- EngineCore Request added to encoder start: 285 ms

The second interval includes scheduler dispatch, runner input preparation,
multimodal metadata preparation, and related process handoff. It is the
dominant single-request regression. HF processing, encoder compute, and
post-encoder thinker prefill are already comparable or slightly faster in
vLLM.

### 4.3 SGLang single-request timeline

All times are relative to preprocess_hf_processor_start:

| Event | Relative time |
| --- | ---: |
| HF processor end | 113.808 ms |
| Preprocessing stage complete | 147.435 ms |
| Image encoder start | 174.479 ms |
| Audio encoder start | 177.511 ms |
| Image encoder end | 479.798 ms |
| Audio encoder end | 523.394 ms |
| Thinker prefill start | 533.094 ms |
| 8192-token prefill chunk end | 853.610 ms |
| Final 506-token prefill chunk end | 901.150 ms |
| First sampled token | 901.214 ms |
| Terminal response | 924.572 ms |

The roughly 30 ms from preprocessing completion to encoder input includes the
100 MB BF16 image payload relay write and stage delivery. It is measurable but
is much smaller than vLLM's roughly 339 ms post-HF pre-encoder interval.

## 5. Fifty requests at rolling concurrency 8

### 5.1 Overall result

| Metric | SGLang | vLLM | SGLang change |
| --- | ---: | ---: | ---: |
| Completed / failed | 50 / 0 | 50 / 0 | equal |
| Active wall time | 35.753 s | 50.323 s | -29.0% |
| Throughput | 1.399 req/s | 0.994 req/s | +40.8% |
| E2E mean | 5.552 s | 7.801 s | -28.8% |
| E2E P50 | 5.670 s | 8.019 s | -29.3% |
| E2E P95 | 7.812 s | 8.121 s | -3.8% |
| Request TTFT mean | 5.060 s | 5.635 s | -10.2% |
| Post-media TTFT mean | 4.852 s | 5.635 s | -13.9% |
| TTFT P95 | 6.453 s | 7.884 s | -18.2% |
| Prompt tokens/request | 8,698 | 8,763 | -0.75% |
| Completion tokens total | 224 | 200 | SGLang did 12% more output work |

The last two vLLM requests run during queue drain and have much lower latency.
For the first 48 requests:

| Metric | SGLang | vLLM |
| --- | ---: | ---: |
| E2E mean | 5.612 s | 8.037 s |

### 5.2 HF processor

| Metric | SGLang | vLLM |
| --- | ---: | ---: |
| HF processor mean | 116.822 ms | 127.444 ms |
| HF processor P95 | 131.966 ms | 138.200 ms |
| Total HF processor time | 5.841 s | 6.372 s |

SGLang is about 10.6 ms or 8.3% faster per request. This helps TTFT but cannot
explain the 14.57 s active-wall difference.

vLLM also serializes the eight frontend HFPREP calls. In the first wave, all
eight benchmark workers call add_request within about 4 ms, but the processor
starts and finishes them one at a time over about 0.98 s. The raw processor
kernel remains about 127 ms; the observed vLLM add_request overhead grows to:

| Metric | vLLM add_request overhead |
| --- | ---: |
| Mean | 733.837 ms |
| P50 | 937.047 ms |
| P95 | 1,017.275 ms |

SGLang preprocessing stage wall time averages 357.268 ms under the same
concurrency. These boundaries are not identical, but they show substantially
less frontend queuing in SGLang.

### 5.3 SGLang stage capacity and overlap

Unique batch intervals reconstructed from server events:

| Stage | Unique batch wall total | Effective time/request | Capacity |
| --- | ---: | ---: | ---: |
| Image encoder | 23.335 s | 0.467 s | 2.143 req/s |
| Audio encoder | 34.475 s | 0.689 s | 1.450 req/s |
| Thinker prefill | 29.157 s | 0.583 s | 1.715 req/s |
| Thinker decode | 0.509 s | 0.010 s | not limiting |

SGLang audio batching distribution:

| Batch size | Number of batches |
| ---: | ---: |
| 1 | 4 |
| 2 | 4 |
| 3 | 7 |
| 4 | 3 |
| 5 | 1 |

The audio encoder is the SGLang throughput bottleneck. Its predicted capacity
of 1.450 req/s is close to the measured 1.399 req/s.

The image, audio, prefill, and decode intervals add to 87.474 s but occupy a
35.550 s union:

| Concurrent active stages | Wall time |
| ---: | ---: |
| 1 | 2.445 s |
| 2 | 14.286 s |
| 3 | 18.819 s |

The mean is 2.461 simultaneously active stage intervals. This overlap is why
SGLang throughput follows the slowest stage instead of the sum of all stages.
The intervals include contention and are not independent CUDA-kernel time, but
the event ordering directly confirms pipeline overlap.

### 5.4 vLLM serialized runner pattern

vLLM records 26 combined MM encoder executions:

| Requests in encoder step | Steps | Mean step time | Total |
| ---: | ---: | ---: | ---: |
| 1 | 14 | 334.872 ms | 4.688 s |
| 3 | 12 | 984.862 ms | 11.818 s |
| Total | 26 | - | 16.507 s |

The 3-request encoder step costs 2.94 times the 1-request step. With
VLLM_OMNI_ENABLE_ENCODER_BATCH=False, it provides almost no cross-request
encoder speedup.

The first and steady request waves use a recurring 1 + 3 + 3 + 1 pattern. The
main reason for groups of at most 3 is the 32,768 token scheduler budget:

    3 * 8,763 = 26,289 <= 32,768
    4 * 8,763 = 35,052 > 32,768

The first request enters EngineCore while the other seven requests are still
passing through serialized HFPREP, causing the underfilled first step.

Across the measured run:

- encoder execution totals 16.507 s, or 0.330 s/request
- non-encoder gaps total about 33.816 s, or 0.676 s/request
- encoder and non-encoder GPU-runner work therefore consume about
  1.006 s/request serially
- 1 / 1.006 = 0.994 req/s, matching measured throughput

The non-encoder residual includes thinker prefill, scheduler/input preparation,
decode, and small idle intervals. This vLLM branch does not expose a reliable
prefill-only CUDA duration separately, so it must not be labeled pure prefill.

For steady 3-request steps, a roughly 985 ms encoder execution is followed by
about 1,026 ms before the next encoder execution. In contrast, SGLang overlaps
the next encoder batches with thinker prefill from earlier requests.

## 6. Code-level explanation

### SGLang

- sglang_omni/models/qwen35_omni/config.py:21-38 fans preprocessing out to
  image encoder, audio encoder, and multimodal aggregation.
- sglang_omni/models/qwen35_omni/config.py:74-83 waits for encoder outputs
  before sending the request to thinker.
- sglang_omni/models/qwen35_omni/stages.py:120-150 dynamically batches image
  encoder requests.
- sglang_omni/models/qwen35_omni/stages.py:185-212 dynamically batches audio
  encoder requests.
- sglang_omni/scheduling/simple_scheduler.py:101-130 collects immediately
  available requests into a batch.
- sglang_omni/models/qwen3_omni/components/preprocessor.py:228-243 casts
  vision encoder inputs to BF16 before transport.
- sglang_omni/relay/shm.py:222-248 maps shared-memory payloads without a
  second CPU byte copy.

### vLLM

- /work/vllm/vllm/v1/worker/gpu_model_runner.py:2810-2980 executes the
  multimodal encoder and writes its outputs into encoder cache storage.
- /work/vllm/vllm/v1/worker/gpu_model_runner.py:2911-2961 is the active
  non-reordered encoder path when VLLM_OMNI_ENABLE_ENCODER_BATCH=False.
- /work/vllm/vllm/v1/worker/gpu_model_runner.py:3632-3639 executes the MM
  encoder inside _preprocess().
- /work/vllm/vllm/v1/worker/gpu_model_runner.py:4710-4799 completes
  _preprocess() and only then calls thinker _model_forward().

This ordering makes encoder and thinker prefill serial within one runner step.
It is the central implementation difference behind concurrency throughput.

## 7. Correctness result

| Engine | C | A | B | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| SGLang | 33 | 9 | 8 | 66% |
| vLLM | 50 | 0 | 0 | 100% |

SGLang raw responses:

- Answer: C: 21
- Answer: (C): 12
- Answer: A: 9
- Answer: B: 8

vLLM returns Answer: C for all 50 requests. Both use greedy/top-k-1 style
sampling. SGLang's concurrent nondeterminism remains a correctness blocker, so
the throughput advantage must not be accepted without fixing or explaining it.

## 8. Optimization priorities

### vLLM

1. Decouple MM encoder execution from thinker forward so encoder work for later
   requests can overlap current thinker prefill.
2. Validate and enable modality reordering/batching behind
   VLLM_OMNI_ENABLE_ENCODER_BATCH; batch-3 currently scales almost linearly.
3. Remove frontend HFPREP serialization or move it to a bounded worker pool.
4. Profile the 285 ms single-request Request added -> encoder start interval
   inside scheduler dispatch, _prepare_inputs, metadata creation, and IPC.
5. Make disable_encoder_cache avoid allocation, insertion, and eviction rather
   than only preventing useful hits.
6. Align the audio loader and processor output to SGLang's 8,698-token input.

### SGLang

1. Fix repeated-request correctness under concurrency before further
   performance acceptance.
2. Optimize audio encoder batching first. It is the current 1.450 req/s stage
   ceiling.
3. After audio exceeds about 1.7 req/s, thinker prefill becomes the next likely
   bottleneck.
4. HF processor and SHM relay are secondary: together they are visible in
   single-request latency but do not limit current concurrency throughput.

## 9. Result artifacts

SGLang single:

    /home/gangouyu/benchmarks/qwen35_s2t_align/sglang-aligned-single-001-1-20260711-131910/

SGLang 50x8:

    /home/gangouyu/benchmarks/qwen35_s2t_align/sglang-final-repeat-001-1-c8-50-20260711-122636/

vLLM single, inside lichenhao-ecs/gangouyu_pai_omni:

    /work/vllm_gangouyu/data/videoamme/vllm_compare_001-1_single_aligned_seq8_20260711_132850/

vLLM 50x8, inside lichenhao-ecs/gangouyu_pai_omni:

    /work/vllm_gangouyu/data/videoamme/vllm_compare_001-1_repeat50_8c_aligned_20260711_125804/

## 10. Confidence and limitations

- Overall latency, TTFT, wall time, throughput, cache misses, response
  distributions, and SGLang stage timings come directly from result JSON and
  server events.
- vLLM encoder time comes directly from per-step encoder logs.
- vLLM prefill-only CUDA time is not independently available. The report uses
  a clearly labeled non-encoder residual rather than presenting an inferred
  value as exact prefill time.
- Single-request results are one aligned measured request after warmup. A
  previous vLLM run produced nearly the same result, but this report does not
  present a multi-run confidence interval.
- Different host CPUs and the 65-token processor difference remain small
  uncontrolled variables.
