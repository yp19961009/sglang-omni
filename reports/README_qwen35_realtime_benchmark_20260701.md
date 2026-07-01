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

Both commands default to `TRUNK_SIZE=40`, `SIL_OFFSET=700`, `TEMPERATURE=1.0`,
and `BARRIER_PREFIX=1`.

## Realtime measurement shape

The benchmark does not send a separate warmup request. For each sample it
extends one realtime session prefix chunk by chunk, then measures the final
streamed chunk.

For `TRUNK_SIZE=40`, the expected final chunk cache behavior is:

```text
processor audio/video: 39 hits + 1 miss
encoder audio/video:   39 hits + 1 miss
```

If a final request shows `40 hits + 0 misses`, it is probably reusing cache
from a previous benchmark run and is not the correct realtime final-chunk
measurement.

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
