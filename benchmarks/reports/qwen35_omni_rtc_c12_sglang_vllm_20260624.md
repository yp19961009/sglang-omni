# Qwen3.5-Omni RTC C=12 Performance Check

Date: 2026-06-24

Dataset: `/myapp/data/share-data-6batch`

Scenario: 40 video/audio chunks per request, 12 concurrent sessions, `max_tokens=64`, `prerun_max_tokens=2`, `video_fps=1`, `visual_mode=video_frames`.

## 2026-06-24 Post-Fix C=12 Update

Latest SGLang fixes tested here:

- Decode scheduler stream inbox and stream outbox priority are enabled by default.
- Decode profiler now records stream enqueue/dequeue and first text delta build.
- Non-stream terminal completes are deferred to a low-priority complete path.
- Stage control-plane loops yield every message via `SGLANG_OMNI_STAGE_IO_YIELD_EVERY_MESSAGES=1`, preventing input handling from starving outbox stream sends.

Strict burst comparison, 40 chunks, 12 concurrent sessions, stagger 0 ms:

| Engine | Run | Completed | Avg TTFT | P95/P99 TTFT | Avg TTFA | P95/P99 TTFA | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | `results/vllm_rtc_trunk40_c12_mt64_compile_fa3off_20260624_114111/rtc_20260624_115330/trunk_40_conc_12` | 12/12 | 1.566 s | n/a / 2.786 s | 3.168 s | n/a / 5.666 s | Current optimized baseline |
| SGLang text-only | `results/qwen35_sglang_rtc_trunk40_c12_textonly_stage_yield_20260624_132101/rtc_trunk40_c12_textonly_mt64` | 12/12 | 19.150 s | 26.055 s / 27.261 s | n/a | n/a | Latest post-yield text path |
| SGLang text+audio | `results/qwen35_sglang_rtc_trunk40_c12_audio_stage_yield_20260624_132608/rtc_trunk40_c12_audio_mt64` | 12/12 | 19.117 s | 25.897 s / 27.126 s | 46.195 s | 48.504 s / 50.686 s | Latest post-yield audio path |

Latest SGLang audio profile: `results/qwen35_sglang_rtc_trunk40_c12_audio_stage_yield_20260624_132608/request_profile_sglang_c12_audio_stage_yield.json`

| Component | Avg | P95 | Interpretation |
| --- | ---: | ---: | --- |
| Request to first text | 19.114 s | 26.448 s | Matches latest benchmark TTFT |
| Request to first audio | 46.193 s | 49.483 s | Matches latest benchmark TTFA |
| Admission to preprocessing receive | 4.175 s | 7.792 s | Burst admission/front-door queue still visible |
| Preprocessing stage total | 4.431 s | 8.106 s | Preprocessing queue dominates preprocessing compute |
| Thinker prefill to first emit | 0.200 s | 0.228 s | LLM first-token compute is not the bottleneck |
| Thinker first stream sent to decode received | 8.858 s | 16.174 s | Largest remaining TTFT connector bottleneck |
| Decode first text build to coordinator send | 0.097 s | 0.310 s | Fixed by event-loop yield/outbox fairness |
| Decode to talker first stream receive | 0.034 s | 0.105 s | No longer a major connector bottleneck |
| Talker first receive to first code | 29.862 s | 33.638 s | Main TTFA bottleneck after text-path fix |
| Code2wav first receive to first audio | 6.226 s | 14.047 s | Secondary TTFA tail |

Post-fix conclusion: SGLang C=12 strict burst is stable and substantially better than the earlier 34.2 s TTFT strict run, but it is still not vLLM-level. The remaining text gap is mostly `thinker -> decode` control-plane FIFO/backpressure: stream chunks share the same stage endpoint with many pre-run terminal payloads. The remaining audio gap is dominated by talker first-code latency and code2wav first-audio tail.

## 2026-06-24 Final C=12 Update

The strict burst comparison is now vLLM C=12 stagger0 vs SGLang C=12 stagger0. SGLang completes 12/12, but is not vLLM-level for this RTC workload yet.

| Engine | Run | Stagger | Temp / sampling note | Completed | Avg TTFT | P99 TTFT | Avg TTFA | P99 TTFA | Diagnosis |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | `results/vllm_rtc_trunk40_c12_mt64_compile_fa3off_20260624_114111/rtc_20260624_115330/trunk_40_conc_12` | 0 ms | thinker near-greedy, talker sampling | 12/12 | 1.566 s | 2.786 s | 3.168 s | 5.666 s | Current optimized baseline |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_final_temp0_stagger0_20260624_123221/rtc_trunk40_c12_audio_mt64` | 0 ms | `temperature=0.0`; final recommended server knobs, no decode-priority experiment | 12/12 | 34.231 s | 38.524 s | 47.745 s | 51.250 s | Strict burst C=12 is head-of-line/backpressure limited |
| SGLang probe | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_decode_priority_temp0_20260624_122253/rtc_trunk40_c12_audio_mt64` | 300 ms | `temperature=0.0`; decode-priority experiment enabled | 12/12 | 12.329 s | 17.231 s | 5.375 s | 9.821 s | Shows deterministic talker/code2wav can approach vLLM TTFA, but text path still lags |
| SGLang probe | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_first_stream_priority_20260624_111951/rtc_trunk40_c12_audio_mt64` | 300 ms | default temperature; first-stream priority | 12/12 | 4.821 s | 6.939 s | 36.589 s | 46.344 s | Best 300ms-stagger TTFT, but not burst parity and audio tail is poor |

For the strict stagger0 run, actual-run-only stage profile:

| Component | Avg | P95 | Interpretation |
| --- | ---: | ---: | --- |
| Request to first text | 34.213 s | 38.170 s | Matches benchmark TTFT |
| Request to first audio | 47.743 s | 50.864 s | Matches benchmark TTFA |
| Admission to preprocessing receive | 4.101 s | 7.865 s | Burst C=12 enters with several seconds of front-door queueing |
| Preprocessing stage total | 4.473 s | 8.175 s | Queue dominates; actual HF processor is only 0.111 s avg |
| Preprocessing payload hops complete | 0.683 s | 0.698 s | Large visual/audio payload fanout is visible but not the main tail |
| Image encoder compute | 0.120 s | 0.137 s | Encoder cache keeps repeated chunks under control |
| Audio encoder compute | 0.107 s | 0.170 s | Not the primary bottleneck |
| Thinker prefill to first emit | 0.208 s | 0.239 s | Thinker GPU first token is short after prefix/cache hits |
| Thinker first emit to decode receive | 12.228 s | 18.278 s | Largest TTFT connector bottleneck under burst |
| Decode receive to first text | 11.901 s | 23.165 s | Decode scheduling / first displayable text backlog dominates the rest of TTFT |
| Talker receive to build | 11.146 s | 15.246 s | Talker waits behind upstream/backlog before useful AR work |
| Talker code predictor | 12.779 s | 19.600 s | Major TTFA tail; compile/cudagraph still does not cover this burst shape well |
| Talker stage total | 36.039 s | 39.447 s | Main audio path bottleneck |
| Code2wav first collect+decode to first audio | 4.588 s | 13.548 s | First-audio tail after talker emits first code chunk |

Strict conclusion: SGLang C=12 stability is fixed, but vLLM remains much faster for the 40-trunk realtime burst. The next optimization target is not encoder compute; it is first-stream delivery under burst (`thinker -> decode` stream backlog, decode first-text backlog) and talker AR/code-predictor batch-shape coverage.

## Result Summary

| Engine | Run | C | Samples | Cache state | Stagger | Completed | Avg TTFT | P99 TTFT | Avg TTFA | P99 TTFA | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM | `results/vllm_rtc_trunk40_c12_mt64_compile_fa3off_20260624_114111/rtc_20260624_115330/trunk_40_conc_12` | 12 | 12 | benchmark warm/prerun | 0 ms | 12 | 1.566 s | 2.786 s | 3.168 s | 5.666 s | Current best baseline |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_final_temp0_stagger0_20260624_123221/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 0 ms | 12 | 34.231 s | 38.524 s | 47.745 s | 51.250 s | Final strict burst run; stable but not performant |
| SGLang probe | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_decode_priority_temp0_20260624_122253/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 12.329 s | 17.231 s | 5.375 s | 9.821 s | Deterministic temp=0 probe; decode-priority experiment was enabled |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_recommended_20260624_120813/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 19.711 s | 28.291 s | 36.880 s | 51.325 s | Final default-temperature probe after disabling synthetic warmup |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_first_stream_priority_20260624_111951/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 4.821 s | 6.939 s | 36.589 s | 46.344 s | Best stable SGLang run; first-stream priority enabled |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_priority_preproc2_20260624_113002/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 5.929 s | 10.765 s | 40.821 s | 43.656 s | Stable, but slower than preproc=1 due to downstream pressure |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_priority_preproc4_stagger0_20260624_112542/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 0 ms | 0 | n/a | n/a | n/a | n/a | Unstable burst: talker prefill index/CUDA assert, client payload incomplete |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_busy_yield_20260624_105835/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 15.890 s | 28.017 s | 34.011 s | 47.054 s | Before first-stream priority; AR busy-yield enabled |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_runtime_drain_20260624_104955/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 19.628 s | 31.708 s | 35.933 s | 46.131 s | Runtime outbox drain optimized |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_itemcache_noomit_20260624_103523/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 21.217 s | 36.546 s | 32.696 s | 47.432 s | Stable post-fix run |
| SGLang | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_itemcache_noomit_stagger0_20260624_104050/rtc_trunk40_c12_audio_mt64` | 12 | 12 | warm same server/data | 0 ms | 12 | 23.913 s | 30.133 s | 15.262 s | 24.605 s | Burst smoke only; reuses previous run cache |
| SGLang old | `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_nixl_detok_fix_20260624_100816/rtc_trunk40_c12_audio_mt64` | 12 | 12 | fresh server | 300 ms | 12 | 33.620 s | 45.556 s | 35.138 s | 53.525 s | Before item-cache capacity/full-cache optimization |

For strict burst parity, compare vLLM stagger0 with the final SGLang stagger0 row. On that point, SGLang is stable but not competitive: avg TTFT is 21.9x vLLM, P99 TTFT is 13.8x vLLM, avg TTFA is 15.1x vLLM. The older 300ms-stagger SGLang rows are useful for bottleneck diagnosis and capacity exploration, not for claiming burst C=12 parity.

## What Was Fixed

1. NIXL relay notifications were shared as a constant `done` token. Under concurrency, one waiter could drain another request's completion notification, leaking credits and eventually blocking stage-to-stage transfer. The fix gives each put operation a unique notification token and caches unrelated notifications for the right waiter.

2. Streaming detokenization could crash on out-of-vocab media/pad token ids, for example very large ids introduced for multimodal prefix-cache placeholders. The fix filters non-decodable token ids before tokenizer decode.

3. Encoder item cache was present but ineffective for 12x40 realtime because the same LRU stored both per-item chunk outputs and whole-prefix outputs. Whole-prefix outputs grow roughly O(N^2) over trunk 1..40 and evicted useful chunk entries. The new benchmark recipe uses:

```bash
SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=2048
SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=17179869184
SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=0
```

`SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1` did reduce payload and showed item hits, but in C=12 formal audio it caused a `talker_ar` CUDA device-side assert in this environment, so it is not part of the stable recipe.

4. Stage runtime outbox drain previously used `run_in_executor(... queue.get(timeout=0.1))`, which could starve under high concurrent stage traffic. It now uses non-blocking `get_nowait()` plus a short async sleep. This reduced `decode -> coordinator` stream hop from second-level tails to avg 7.2 ms / p95 39.4 ms in the `runtime_drain` run.

5. AR scheduler busy loops could starve the same-process asyncio stage IO loop after `scheduler_first_emit`. A configurable busy-yield (`SGLANG_OMNI_AR_BUSY_YIELD_EVERY_BATCHES=1`) releases the GIL after busy batches. In the latest C=12 run, avg TTFT improved from 19.628 s to 15.890 s.

6. Thinker first-stream batches were still stuck behind normal stream traffic in FIFO outbox order. The scheduler now marks the first emitted stream batch as priority and the outbox promotes it ahead of already-queued non-first stream chunks. This reduced `thinker_outbox_put_to_dequeue` from avg 8.8 s / p95 18.1 s in the pre-priority run to avg 64 ms / p95 138 ms in the best stable run, improving avg TTFT from 14.075 s to 4.821 s.

## Latest TTFT Breakdown

Profile: `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_first_stream_priority_20260624_111951/request_profile_sglang_c12_audio_mt64_first_stream_priority.json`

Formal audio requests only:

| Component | Avg | P50 | P95 | Max | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Request to first text chunk at coordinator | 4.819 s | 6.492 s | 6.922 s | 6.938 s | Matches benchmark TTFT; low values are requests that entered preprocessing before the queue filled |
| Request to first audio chunk at coordinator | 36.587 s | 42.644 s | 46.070 s | 46.338 s | Long-output audio tail still dominates |
| Request to terminal response | 39.365 s | 45.594 s | 49.452 s | 50.062 s | End-to-end streaming completion |
| Preprocessing queue | 3.004 s | 5.038 s | 5.166 s | 5.184 s | Main remaining TTFT contributor after priority fix |
| Preprocessing compute | 0.506 s | 0.514 s | 0.519 s | 0.522 s | HF processor/video/audio packaging |
| HF processor | 0.472 s | 0.471 s | 0.477 s | 0.477 s | Mostly video frame processing/token replacement |
| Processor video | 0.454 s | 0.454 s | 0.459 s | 0.459 s | Video decode/frame packaging for 40 chunks |
| Image encoder compute | 0.190 s | 0.174 s | 0.228 s | 0.229 s | Item cache keeps repeated chunk encode under control |
| Audio encoder compute | 0.076 s | 0.077 s | 0.106 s | 0.108 s | Not a bottleneck |
| Thinker build + queue | 0.030 s | 0.029 s | 0.035 s | 0.036 s | Not a bottleneck |
| Thinker prefill to first emit | 0.187 s | 0.189 s | 0.190 s | 0.190 s | GPU prefill/first emit is short after cache hits |
| Thinker first emit to outbox dequeue | 0.064 s | 0.066 s | 0.138 s | 0.153 s | Fixed by first-stream priority |
| Thinker stream send to decode receive | 0.562 s | 0.608 s | 0.649 s | 0.654 s | Remaining stream transfer overhead |
| Decode receive to first text send | 0.236 s | 0.113 s | 0.700 s | 0.815 s | Detokenizer / scheduling before first text chunk |
| Decode send to coordinator | 0.003 s | 0.000 s | 0.017 s | 0.037 s | No longer a bottleneck |
| Talker receive to first code chunk | 26.466 s | 32.116 s | 34.605 s | 35.410 s | Main TTFA gap against vLLM |
| Code2wav receive to first audio | 6.338 s | 4.615 s | 14.056 s | 25.426 s | First-audio tail, partly waiting for enough talker chunks |

## SGLang Preprocessing Concurrency Check

Profile: `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_priority_preproc2_20260624_113002/request_profile_sglang_c12_audio_mt64_priority_preproc2.json`

Increasing preprocessing concurrency from 1 to 2 did reduce preprocessing queue avg from 3.004 s to 1.638 s, but total TTFT regressed from 4.821 s to 5.929 s. The extra parallel preprocessing increased downstream pressure: image encoder compute avg rose from 0.190 s to 0.466 s, audio encoder compute avg rose from 0.076 s to 0.194 s, thinker queue/prefill rose from 0.193 s combined to 1.458 s combined, and talker receive-to-first-code rose from 26.466 s to 32.929 s. For this machine and C=12 RTC workload, `PREPROCESSING_MAX_CONCURRENCY=1` remains the best stable serving point.

## Historical SGLang Fresh-Server Stage Breakdown Before First-Stream Priority

Profile: `results/qwen35_sglang_rtc_trunk40_c12_audio_mt64_itemcache_noomit_20260624_103523/request_profile_sglang_c12_audio_mt64_itemcache_noomit.json`

Formal audio requests only:

| Component | Avg | P50 | P95 | Max | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Request to first text chunk at coordinator | 21.215 s | 18.835 s | 36.007 s | 36.545 s | Matches benchmark TTFT |
| Request to first audio chunk at coordinator | 32.694 s | 42.980 s | 46.612 s | 47.427 s | Matches benchmark TTFA |
| Preprocessing lifecycle | 3.854 s | 5.613 s | 5.730 s | 5.747 s | Mostly single-stage queueing; actual compute is much lower |
| Preprocessing compute | 0.523 s | 0.527 s | 0.539 s | 0.541 s | HF processor/video/audio packaging |
| Image encoder lifecycle | 0.242 s | 0.244 s | 0.290 s | 0.294 s | Item cache fixed repeated video encode |
| Audio encoder lifecycle | 0.083 s | 0.080 s | 0.106 s | 0.107 s | Not a bottleneck |
| MM aggregate lifecycle | 0.218 s | 0.221 s | 0.268 s | 0.275 s | Not a bottleneck |
| Thinker lifecycle | 11.464 s | 10.566 s | 21.457 s | 21.697 s | Main TTFT contributor after preprocessing |
| Thinker prefill to first stream | 7.733 s | 5.446 s | 17.457 s | 18.814 s | Prefill/cache/generation tail |
| Talker lifecycle | 40.678 s | 37.506 s | 82.993 s | 83.875 s | Dominates long-output E2E and TTFA tail |

Full-profile top stage changes:

| Stage interval | Previous avg | New avg | Change |
| --- | ---: | ---: | ---: |
| image_encoder `stage_input_received->stage_complete` | 26.349 s | 0.177 s | 149x lower lifecycle average |
| mm_aggregate `stage_input_received->stage_complete` | 26.235 s | small in formal path | Queue pressure removed after encoder fix |
| SGLang elapsed for C=12 | 753.9 s | 229.5 s | 3.29x faster |

## vLLM Breakdown

Profile: `results/vllm_rtc_trunk40_c12_mt64_compile_fa3off_20260624_114111/rtc_20260624_115330/comparison.txt`

vLLM avg TTFA is 3.168 s:

| Component | Avg | Share |
| --- | ---: | ---: |
| Thinker prefill | 1.566 s | 49.4% |
| Hidden-state queue | 1.343 s | 42.4% |
| Embed build | 0.016 s | 0.5% |
| Talker prefill | 0.107 s | 3.4% |
| Codec accumulation | 0.082 s | 2.6% |
| First code2wav | 0.055 s | 1.7% |

## Current Conclusion

SGLang C=12 is now stable and no longer blocked by NIXL relay, detokenizer overflow, repeated video encoder computation, decode coordinator outbox starvation, or FIFO starvation of the first thinker stream. That is a real stability fix, but the strict stagger0 burst run shows SGLang is still far behind vLLM for RTC 40-trunk.

The remaining gap has moved away from raw encoder compute. In the final stagger0 run, image/audio encoder compute is around 0.12 s / 0.11 s, and thinker prefill-to-first-emit is around 0.21 s. The large losses are stage connection and scheduling effects: `thinker -> decode` first-stream transfer averages 12.23 s, decode receive-to-first-text averages 11.90 s, talker code predictor averages 12.78 s, and code2wav first-audio collect/decode adds another 4.59 s on average.

Recommended next optimization target: make first-stream delivery preemptive across `thinker -> decode` under burst, reduce decode first-visible-text backlog, and add real talker code-predictor shape coverage for concurrent first-code batches. Do not use `SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1` in the current C=12 audio recipe; it is promising for payload reduction but caused a CUDA device-side assert in this environment.
