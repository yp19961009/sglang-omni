# Qwen3.5-Omni realtime benchmark notes

Last validated: 2026-07-02 CST

These scripts are intended to run inside the server/container repo:

```bash
cd /myapp/sglang-omni
```

Do not run them as Mac-side SSH wrapper scripts. Codex may connect to the
server over SSH for development, but the benchmark itself should start and
request the local service from the server/container.

## Start the stable service

Start or check the stable service on port 8162:

```bash
bash reports/run_sglang_qwen35_stable_server.sh
```

Force a clean restart:

```bash
TRACE_CACHE=0 FORCE_RESTART=1 \
  bash reports/run_sglang_qwen35_stable_server.sh
```

Enable cache summary logs when debugging cache behavior:

```bash
TRACE_CACHE=1 FORCE_RESTART=1 \
  bash reports/run_sglang_qwen35_stable_server.sh
```

`TRACE_CACHE=1` defaults to `TRACE_CACHE_SCOPE=actual`, so only measured
requests emit cache summaries. This keeps prefix-extension requests off the
hot logging path while still showing the final-chunk `39 hits + 1 miss`
vLLM-style behavior. Use `TRACE_CACHE_SCOPE=all` only when you need the old full-prefix
log stream.

Use `TRACE_CACHE_DETAIL=1` only for detailed item-level cache debugging. It is
noisy and should not be used for pure performance numbers.

The stable service defaults to `QWEN35_RTC_DISABLE_ACTUAL_MAMBA_TRACK=0`.
That means measured actual requests still track the request-local Mamba/SSM
state for the final chunk before decode. This is required for correctness: the
actual request reuses the first 39 chunks from prefix cache, then must update
its local state with chunk 40 before generating text/audio.

Measured actual requests still default to
`SGLANG_OMNI_SKIP_RTC_ACTUAL_MAMBA_CACHE_INSERT=1`, so that final actual state
is not written back into the radix cache. Keep these two controls separate:
track is local state for decode; cache insert/writeback is reusable prefix cache
state for future requests.

## Run C1 and C12

Single concurrency:

```bash
CONCURRENCY=1 TOTAL_SAMPLES=1 RUN_LABEL=c1 \
  bash reports/run_sglang_qwen35_stable_c12_benchmark.sh
```

Twelve-way concurrency:

```bash
CONCURRENCY=12 TOTAL_SAMPLES=12 RUN_LABEL=c12 \
  bash reports/run_sglang_qwen35_stable_c12_benchmark.sh
```

vLLM-style request-profiler metrics are enabled by default. Disable them only
when you need pure client-SSE timing without profiler overhead:

```bash
PROFILE_REQUESTS=0 CONCURRENCY=1 TOTAL_SAMPLES=1 RUN_LABEL=c1_no_profile \
  bash reports/run_sglang_qwen35_stable_c12_benchmark.sh
```

Both commands default to `TRUNK_SIZE=40`, `SIL_OFFSET=700`, `TEMPERATURE=1.0`,
`BARRIER_PREFIX=0`, `PREFIX_MAX_TOKENS=2`, and `PROFILE_REQUESTS=1`. The
profiler post-processes only actual requests, matching vLLM's pre-run filtering
style.

## Realtime measurement shape

The benchmark does not send a separate warmup request. By default it mirrors
the vLLM `run_rtc_profile.sh` concurrency shape: each concurrency worker runs
one realtime session, sends incremental `pre_run` requests for chunks
`1..TRUNK_SIZE-1`, then immediately sends the measured actual request for the
same `TRUNK_SIZE`.

For `CONCURRENCY=12`, that means 12 workers execute this sequence in parallel:

```text
session N: pre_run chunk 1 -> ... -> pre_run chunk 39 -> actual chunk 40
```

The default `PREFIX_MAX_TOKENS=2` also mirrors vLLM's current pre-run behavior.
Set `PREFIX_MAX_TOKENS=0` if you specifically want pure cache-extension
pre-runs, or set `BARRIER_PREFIX=1` to use the older all-prefixes-first barrier
shape where measured actual requests start together after every prefix is done.

The older barrier shape can make the final 40th chunk look more stable or
faster in some runs, but it is not the same concurrency pattern as vLLM's
`run_rtc_profile.sh`. Use the default `BARRIER_PREFIX=0` numbers when comparing
against vLLM.

For `TRUNK_SIZE=40`, the expected final measured request cache behavior is:

```text
processor audio/video: 39 hits + 1 miss
encoder audio/video:   39 hits + 1 miss
```

The last prefix request is `pre_run trunk=39`. The measured actual
`trunk=40` request reuses those 39 chunks and appends the question audio/video
chunk, so one new processor/encoder item is expected.

## Metric semantics

`ttft_ms` / `ttft_*` means first token by default. When request profiling is
enabled, it is measured from request admission to `thinker.scheduler_first_emit`,
matching the vLLM-style TTFT definition. If the profiler is unavailable, the
benchmark falls back to the first streamed text event and records that fallback
in `ttft_semantics`. The client-observed first text event is kept separately as
`client_first_text_event_ms` and as the backward-compatible `first_text_event_ms`.

`ttfa_ms` is the first streamed audio event. `first_output_ms` records whichever
streamed output event arrives first, with `first_output_type` showing whether it
was text or audio. Raw SSE timing also includes `first_audio_event_ms` and
`text_audio_event_gap_ms = first_audio_event_ms - first_text_event_ms`.

## Current reference numbers

These were measured on 2026-07-02 after a forced restart of the stablefast
service on port 8162, with `TRACE_CACHE=0`,
`QWEN35_RTC_DISABLE_ACTUAL_MAMBA_TRACK=0`,
`SGLANG_OMNI_SKIP_RTC_ACTUAL_MAMBA_CACHE_INSERT=1`,
`QWEN35_RTC_LIMIT_ACTUAL_PREFIX_TO_COMPLETE_TURN=1`,
`SGLANG_OMNI_SKIP_NAN_MAMBA_CACHE=1`, `CUDA_VISIBLE_DEVICES=3,4,5`, thinker on
GPU 0, talker on GPU 1, and code2wav on GPU 2.

Reference RUN_DIR:

```text
results/sg_realtime_stablefast_mem072_videocache17g_cache4096_trimpartial_cg_on_run12_c2w4_relay1024_cvd345_8162_20260702_075248
```

The important stability guards are:

- measured actual RTC requests reuse prefix/Mamba state only through the last
  complete prefix limit captured from `pre_run trunk=39`;
- measured actual RTC requests keep Mamba tracking enabled, so chunk 40 updates
  the request-local state used by decode;
- measured actual RTC requests skip radix-cache insertion/writeback, so chunk 40
  does not become reusable prefix state for later sessions.

Processor and encoder item caches still reuse the first 39 realtime chunks and
miss only the new 40th item.

C1, 40 chunks, vLLM pipeline shape, immediately after restart:

```text
client dir:          client_c1_rtcflow_vllm_pipeline_sil700_trunk40_samples1_stagger0_temp1.0_prefixmt2_c1_track_on_075444
completed / failed: 1 / 0
TTFT avg/p99:       713.3 / 713.3 ms
TTFA avg/p99:       1232.6 / 1232.6 ms
first text avg/p99: 1117.5 / 1117.5 ms
last_audio avg/p99: 2514.4 / 2514.4 ms
E2E avg/p99:        3119.6 / 3119.6 ms
audio avg:          9.20 s
audio chunks avg:   29.0
breakdown avg:      thinker 713.3 ms, hf_preproc 190.4 ms,
                    talker_prefill 190.3 ms, code2wav_first_chunk 158.1 ms
output sanity:      bang_count=0, errors=[]
server counters:    500=0, oom=0, mismatch=0, omitted_payload_cache_miss=0
```

C12, 40 chunks, vLLM pipeline shape, same restarted service:

```text
client dir:          client_c12_rtcflow_vllm_pipeline_sil700_trunk40_samples12_stagger0_temp1.0_prefixmt2_c12_track_on_075513
completed / failed: 12 / 0
TTFT avg/p99:       4634.1 / 6406.2 ms
TTFA avg/p99:       5966.5 / 7566.6 ms
first text avg/p99: 5663.6 / 8527.5 ms
last_audio avg/p99: 9615.1 / 12033.5 ms
E2E avg/p99:        25028.2 / 30198.6 ms
audio avg:          9.17 s
audio chunks avg:   29.08
breakdown avg:      thinker 4634.1 ms, hf_preproc 175.7 ms,
                    talker_prefill 103.4 ms, code2wav_first_chunk 199.9 ms
output sanity:      bang_count=0, errors=[]
server counters:    500=0, oom=0, mismatch=0, omitted_payload_cache_miss=0
```

C12 repeat on the same restarted service, validating later runs are not poisoned:

```text
client dir:          client_c12_rtcflow_vllm_pipeline_sil700_trunk40_samples12_stagger0_temp1.0_prefixmt2_c12_track_on_rerun_075752
completed / failed: 12 / 0
TTFT avg/p99:       3688.6 / 4706.7 ms
TTFA avg/p99:       4826.7 / 6087.4 ms
first text avg/p99: 4245.8 / 5554.0 ms
last_audio avg/p99: 10274.3 / 12799.7 ms
E2E avg/p99:        25012.0 / 28358.2 ms
audio avg:          8.77 s
audio chunks avg:   27.67
breakdown avg:      thinker 3688.6 ms, hf_preproc 128.0 ms,
                    talker_prefill 103.4 ms, code2wav_first_chunk 180.0 ms
output sanity:      bang_count=0, errors=[]
server counters:    500=0, oom=0, mismatch=0, omitted_payload_cache_miss=0
```

The C12 repeat sample texts were all normal self-introductions such as
"I am Qwen" / "I am Tongyi Qianwen Omni" in Chinese. Wav durations were
5.52 s to 10.24 s. No sample produced `!!!!`, system prompt leakage, an
over-15-second wav, or an unreadable wav.

Server logs can still contain occasional `invalid codec row` warnings where
code2wav drops an EOS or malformed codec row. The validated runs above did not
turn those warnings into user-visible bad text/audio.

The scripts write output under the active server `RUN_DIR`, with wav files in
`sample_*/`.
