# Qwen3.5-Omni 40 Trunk Realtime: SGLang-Omni vs vLLM Gap Closure

日期：2026-06-23

## 结论

40 trunk 视频 realtime 单并发场景中，SGLang-Omni 原先距离 vLLM 的首音频延迟主要差在跨 stage 大 payload 传输和 talker partial-start 等待策略。将 relay backend 从默认 shm 切到 NIXL，并将 talker partial-start 阈值从 5 个 thinker chunk 降到 3 个 chunk 后，SGLang-Omni 的 TTFA 从 564.46 ms 降到 529.98 ms，距离 vLLM 522.60 ms 只剩 7.38 ms（1.41%）。

| Runtime | 配置 | TTFT ms | TTFA ms | E2E ms | Audio RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM | 40 trunk C=1 isolated warmup | 348.34 | 522.60 | - | - |
| SGLang-Omni | old best, shm relay | 346.61 | 564.46 | 1715.89 | 0.244 |
| SGLang-Omni | NIXL + partial_start_min_chunks=3 | 360.60 | 529.98 | 1785.52 | 0.232 |

## 定位

最关键的证据来自 request profile。40 trunk 最后一轮正式请求中，`mm_aggregate -> talker_ar` 的 payload 为 8 个 tensor，约 120.8 MB。

| 路径 | 写出 ms | 读取 ms | 合计 ms |
| --- | ---: | ---: | ---: |
| shm old best | 140.77 | 73.13 | 213.90 |
| NIXL + partial_start=3 | 16.23 | 8.72 | 24.95 |

NIXL 后，大 payload 传输基本不再是 TTFA 主瓶颈。剩余差距主要来自前半段 image/thinker 波动、talker prefill、code2wav first-audio 路径。`partial_start_min_chunks=3` 将 talker 收到首个 hidden chunk 到 request build 的等待从约 35.00 ms 降到约 16.76 ms。

已验证反例：`SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_TALKER_GPU=1` 会让 thinker request build 明显变慢，TTFA 退化到 614.01 ms，不建议作为优化配置。

## 推荐启动配置

在容器内：

```bash
cd /myapp/sglang-omni

RELAY_BACKEND=nixl \
TALKER_PARTIAL_START_MIN_CHUNKS=3 \
NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_RELAY_PAYLOAD_PREP_EXECUTOR=1 \
SGLANG_OMNI_COLOCATE_PREPROCESSING_WITH_THINKER=1 \
SGLANG_OMNI_COLOCATE_IMAGE_ENCODER_WITH_THINKER=1 \
SGLANG_OMNI_COLOCATE_MM_AGGREGATE_WITH_THINKER=1 \
SGLANG_OMNI_ENCODER_MAX_BATCH_WAIT_MS=0 \
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1 \
SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_THINKER_GPU=1 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
THINKER_MEM_FRACTION_STATIC=0.72 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

正式 benchmark：

```bash
RUN_ROOT="results/sglang_rtc_trunk40_c1_nixl_ps3_$(date +%Y%m%d_%H%M%S)"
COMMON_ARGS=(
  --base-url http://127.0.0.1:8161
  --rtc-test-dir /myapp/data/share-data-6batch
  --trunk-size 40
  --video-fps 1
  --max-chunks-per-turn 6
  --video-min-pixels 4096
  --video-max-pixels 786432
  --video-override-max-pixels
  --timeout-s 1800
)

python -m benchmarks.eval.qwen35_omni_sglang_rtc_profile "${COMMON_ARGS[@]}" \
  --batch-idx 5 \
  --run-id warmup_batch5_trunk40_c1_nixl_ps3 \
  --output-dir "$RUN_ROOT/warmup_batch5_trunk40_c1" \
  --event-dir "$RUN_ROOT/events" \
  --no-profile

python -m benchmarks.eval.qwen35_omni_sglang_rtc_profile "${COMMON_ARGS[@]}" \
  --batch-idx 0 \
  --run-id measure_batch0_trunk40_c1_nixl_ps3 \
  --output-dir "$RUN_ROOT/measure_batch0_trunk40_c1" \
  --event-dir "$RUN_ROOT/events" \
  --profile
```

## 证据路径

- 最优 SGLang run：`results/sglang_rtc_trunk40_c1_nixl_ps3_20260623_115640/measure_batch0_trunk40_c1/metrics.json`
- 最优 SGLang profile：`results/sglang_rtc_trunk40_c1_nixl_ps3_20260623_115640/measure_batch0_trunk40_c1/request_profile_measure_batch0_trunk40_c1_nixl_ps3.json`
- vLLM baseline：`results/vllm_rtc_trunk40_c1_isolated_warmup_20260623_145232/rtc_20260623_150321/trunk_40_conc_1/metrics.json`

## 代码与验证

新增/固化内容：

- `--talker-partial-start-min-chunks` CLI 参数，避免使用脆弱的 `stages.6.factory_args...`。
- `RELAY_BACKEND` 和 `TALKER_PARTIAL_START_MIN_CHUNKS` 启动脚本环境变量。
- relay profile 元数据：`tensor_count`、`tensor_bytes`、`relay_bytes`、`payload_pickle_bytes`。

已跑测试：

```text
tests/unit_test/qwen3_5_omni/test_cli.py
tests/unit_test/qwen3_5_omni/test_config.py
tests/unit_test/pipeline/test_compile.py
tests/unit_test/pipeline/test_stage.py
=> 115 passed

tests/unit_test/qwen3_omni/test_cli.py
tests/unit_test/qwen3_omni/test_talker.py
=> 92 passed
```
