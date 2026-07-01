# Qwen3.5-Omni realtime benchmark notes

Date: 2026-07-01 CST

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
hot logging path while still showing the final-chunk `40 hits + 0 misses`
vLLM-style behavior. Use `TRACE_CACHE_SCOPE=all` only when you need the old full-prefix
log stream.

Use `TRACE_CACHE_DETAIL=1` only for detailed item-level cache debugging. It is
noisy and should not be used for pure performance numbers.

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
`1..TRUNK_SIZE`, then immediately sends the measured actual request for the
same `TRUNK_SIZE`.

For `CONCURRENCY=12`, that means 12 workers execute this sequence in parallel:

```text
session N: pre_run chunk 1 -> ... -> pre_run chunk 40 -> actual chunk 40
```

The default `PREFIX_MAX_TOKENS=2` also mirrors vLLM's current pre-run behavior.
Set `PREFIX_MAX_TOKENS=0` if you specifically want pure cache-extension
pre-runs, or set `BARRIER_PREFIX=1` to use the older all-prefixes-first barrier
shape where measured actual requests start together after every prefix is done.

For `TRUNK_SIZE=40`, the expected final chunk cache behavior is:

```text
processor audio/video: 40 hits + 0 misses
encoder audio/video:   40 hits + 0 misses
```

Because the vLLM-style sequence pre-runs chunk 40 before the actual chunk 40,
`40 hits + 0 misses` is the normal cache-hit result for the measured request.

## Metric semantics

`ttft_ms` is the first streamed text event observed by the client, and
`ttfa_ms` is the first streamed audio event. `first_output_ms` records whichever
streamed output event arrives first, with `first_output_type` showing whether it
was text or audio. The raw SSE event timings are also saved as
`first_text_event_ms` and `first_audio_event_ms`, with
`text_audio_event_gap_ms = first_audio_event_ms - first_text_event_ms`. A
negative gap means the server delivered an audio SSE event before a text SSE
event; in that case `first_output_ms` is the best client-visible first-output
latency, while `ttft_ms` remains strict text-event latency.

## Current reference numbers

These were measured on 2026-07-01 with the stablefast service on port 8162,
`TRACE_CACHE=0`, `CUDA_VISIBLE_DEVICES=3,4,5`, thinker on GPU 0, talker on GPU
1, and code2wav on GPU 2.

C1, 40 chunks:

```text
completed / failed: 1 / 0
actual_elapsed_s:   2.962
TTFT avg/p99:       606.7 / 606.7 ms
TTFA avg/p99:       645.6 / 645.6 ms
last_audio avg/p99: 2084.9 / 2084.9 ms
E2E avg/p99:        2960.0 / 2960.0 ms
audio avg:          9.76 s
audio chunks avg:   31.0
```

C12, 40 chunks:

```text
completed / failed: 12 / 0
actual_elapsed_s:   27.823
TTFT avg/p99:       5852.1 / 8534.0 ms
TTFA avg/p99:       6957.0 / 8472.3 ms
last_audio avg/p99: 16123.1 / 21987.0 ms
E2E avg/p99:        21493.3 / 27811.5 ms
audio avg:          9.07 s
audio chunks avg:   28.67
```

Both runs completed with:

```text
bang_count=0
errors=[]
500=0
oom=0
mismatch=0
omitted_payload_cache_miss=0
```

The scripts write output under the active server `RUN_DIR`, with wav files in
`sample_*/`.
