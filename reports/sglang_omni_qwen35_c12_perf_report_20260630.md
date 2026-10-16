# sglang-omni Qwen3.5-Omni 12 并发 40 chunk 性能报告

生成时间：2026-06-30 23:00 CST  
远端目录：`/myapp/sglang-omni`  
代码提交：`6115ffd fix qwen35 rtc partial visual feature trim`

本地交付附件：

- Benchmark manifest：`reports/sglang_omni_qwen35_c12_benchmark_manifest_20260630.json`
- 稳定服务启动脚本：`reports/run_sglang_qwen35_stable_server.sh`
- 标准复跑脚本：`reports/run_sglang_qwen35_stable_c12_benchmark.sh`

## 结论

当前稳定性能最优版本建议使用 `mem0.80 + code2wav-stream-chunk-size=4 + QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1`。

这个配置在 12 并发、40 chunk、barrier-prerun 口径下已多次 12/12 成功，没有 500、OOM、feature/token mismatch、cache miss，输出文本正常，音频文件完整。首音频不是最快，但整体 actual elapsed 和 last_audio 更稳、更好。

TTFA 优先变体是 `mem0.80 + code2wav-stream-chunk-size=2`，TTFA 从约 10.0s 降到约 8.8s，但 last_audio 和 overall actual elapsed 变差，不建议作为默认稳定最优。

`SGLANG_OMNI_PRIORITY_PREFILL_BATCH_WAIT_MS=20` 能在同 offset 上小幅改善 last_audio/e2e，但会拉高 TTFT/TTFA，并且在 `sil672` 复跑时触发 thinker scheduler crash：`Committed KV cache already freed`，12/12 actual 请求断流失败。因此它不能进入稳定推荐版本。

## 推荐启动配置

```bash
CUDA_VISIBLE_DEVICES=3,4,5
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE=8
SGLANG_OMNI_TALKER_READY_SUBSET_MIN_SIZE=1
SGLANG_OMNI_RELAY_SLOT_SIZE_MB=1024
SGLANG_OMNI_RELAY_CREDITS=2
SGLANG_OMNI_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES=268435456
SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES=2048
SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES=68719476736
SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE=0
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1
QWEN35_TALKER_ALLOW_PARTIAL_TEXT_CHUNK_BEFORE_DONE=0
QWEN35_TALKER_PARTIAL_TEXT_CHUNK_WAIT_SKIPS=0
QWEN35_TALKER_ALLOW_UNBOUNDED_EMPTY_TEXT_FEEDBACK=0
QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
--thinker-mem-fraction-static 0.80
--thinker-max-running-requests 12
--talker-max-running-requests 12
--code2wav-stream-chunk-size 4
--thinker-cuda-graph off
--talker-cuda-graph on
--talker-torch-compile on
```

## Benchmark 对比

| run | 稳定性 | actual_s | TTFT avg / p99 | TTFA avg / p99 | last_audio avg / p99 | e2e avg / p99 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| `sg_mem080_c2w4_best_run12` | 12/12 | 36.19 | 7.81s / 12.81s | 10.01s / 10.07s | 24.28s / 27.47s | 33.79s / 36.18s | 当前整体最优 |
| `sg_mem080_c2w4_profile` | 12/12 | 36.64 | 7.70s / 16.89s | 10.01s / 10.06s | 28.35s / 32.86s | 34.47s / 36.63s | 同配置 profile 复跑，有 profiling/样本抖动 |
| `sg_mem080_c2w4_trimpartial_sil712` | 12/12 | 36.64 | 9.56s / 11.65s | 12.01s / 12.47s | 27.34s / 31.41s | 34.50s / 36.63s | 新提交修复后同 offset 验证通过 |
| `sg_mem080_c2w4_trimpartial_sil724` | 12/12 | 37.31 | 10.57s / 17.90s | 10.79s / 12.76s | 28.88s / 33.98s | 33.42s / 37.30s | 新 offset 独立复验，稳定但不替换最优 |
| `sg_mem080_c2w4_trimpartial_sil732` | 12/12 | 36.78 | 8.45s / 13.94s | 9.38s / 12.43s | 26.64s / 32.69s | 31.03s / 36.77s | wait20 同 offset 对照 |
| `prefillwait20_sil732` | 12/12 | 36.70 | 10.41s / 14.95s | 11.50s / 13.97s | 25.22s / 31.08s | 30.54s / 36.69s | tail 略好，但首包变差 |
| `prefillwait20_sil672` | 0/12 | 失败 | - | - | - | - | thinker scheduler crash，排除 |
| `sg_mem080_c2w2_ttfa` | 12/12 | 37.75 | 7.59s / 12.33s | 8.77s / 9.82s | 26.80s / 31.03s | 34.11s / 37.74s | 首音频更好，尾部变差 |
| `sg_mem084_c2w4` | 12/12 | 36.83 | 7.26s / 9.39s | 10.28s / 10.34s | 27.26s / 30.63s | 34.51s / 36.82s | TTFT p99 好，但尾部差 |
| `sg_partial_text` | 12/12 | 38.98 | 7.46s / 10.19s | 11.67s / 11.72s | 30.62s / 35.32s | 36.93s / 38.97s | 负收益 |
| `mem084+c2w2` | 失败 | - | - | - | - | - | prerun OOM，排除 |
| `code2wav dynamic chunk` | 无效 | - | - | - | - | - | actual 阶段触发 inductor 编译，客户端长时间无 result/wav |

参考 vLLM 结果：

| run | TTFT avg / p99 | TTFA avg / p99 |
|---|---:|---:|
| `vllm_rtc_trunk40_c12_rerun_fa3off_20260624_214308` | 1.91s / 2.67s | 2.51s / 3.41s |
| `vllm_rtc_trunk40_c12_mt64_compile_fa3off_20260624_114111` | 1.57s / 2.79s | 3.17s / 5.67s |
| `vllm_rtc_trunk40_c12_current_fa3off_20260626_211730` | 1.99s / 3.13s | 2.62s / 3.65s |

当前 SGLang 稳定最优相对 vLLM 仍主要慢在 TTFT 和 talker/code2wav 首音频门槛：TTFA 约 10.0s，对 vLLM 2.5-3.2s 仍有约 3-4 倍差距。

## 关键结果目录

稳定最优：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_8162_20260630_103902/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_104103
```

带 profiler 的复跑：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_20260630_122447/client_c12_realtime_audio_vllmstyle_sil700_stagger0_temp1_barrier_profile_stable_c12_c2w4_122641
```

修复后稳定性验证：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_124917/client_c12_realtime_audio_vllmstyle_sil712_stagger0_temp1_barrier_validation_125117
```

新 offset 独立复验：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_124917/client_c12_realtime_audio_vllmstyle_sil724_stagger0_temp1_barrier_validation_130217
```

当前 8162 post-commit 复验：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_142334
```

同 offset warmed 上界：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_143808
```

不同 offset 负向复验：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil680_stagger0_temp1_barrier_validation_144121
```

同 offset wait20 对照：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_124917/client_c12_realtime_audio_vllmstyle_sil732_stagger0_temp1_barrier_validation_132400
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_prefillwait20_relay1024_cvd012_8163_20260630_131428/client_c12_realtime_audio_vllmstyle_sil732_stagger0_temp1_barrier_wait20_131821
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_prefillwait20_relay1024_cvd012_8163_20260630_131428/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_wait20_132958
```

Profile JSON：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_20260630_122447/client_c12_realtime_audio_vllmstyle_sil700_stagger0_temp1_barrier_profile_stable_c12_c2w4_122641/request_profile_stable_c12_c2w4_122641.json
```

## Stage 分析

Profile 覆盖 actual 阶段，12 个 request。

关键路径里程碑，单位 ms，均从 request admission 起算：

| milestone | avg | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| preprocess done | 1199.8 | 1283.6 | 2083.9 | 2261.3 |
| image encoder done | 4624.7 | 4770.4 | 6613.8 | 7030.1 |
| audio encoder done | 1555.1 | 1684.4 | 2423.8 | 2608.1 |
| mm aggregate done | 5914.7 | 6224.0 | 7867.9 | 8270.4 |
| thinker prefill start | 6336.2 | 6542.1 | 8174.2 | 8576.6 |
| thinker first emit | 6733.6 | 6919.1 | 8553.2 | 8762.0 |
| decode first text | 7690.2 | 7193.7 | 8851.8 | 16819.0 |
| talker prefill start | 9457.9 | 9483.4 | 9485.5 | 9485.8 |
| talker first code | 9595.9 | 9608.4 | 9644.3 | 9652.2 |
| first audio | 10001.9 | 10006.7 | 10045.0 | 10054.1 |
| thinker done | 27242.7 | 27617.4 | 30854.7 | 31584.5 |
| talker done | 28340.3 | 28658.8 | 32253.7 | 32848.1 |
| code2wav done | 28354.6 | 28676.0 | 32264.7 | 32862.2 |
| last client stream | 34378.0 | 34298.7 | 35988.5 | 36586.7 |

Stage input->complete duration，单位 ms：

| stage | avg | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| preprocessing | 1199.3 | 1283.2 | 2083.4 | 2260.9 |
| image_encoder | 3247.4 | 3288.0 | 4349.8 | 4591.1 |
| audio_encoder | 227.5 | 233.2 | 246.5 | 258.0 |
| mm_aggregate | 4566.8 | 4756.7 | 5645.7 | 5869.3 |
| thinker | 21062.5 | 20550.1 | 23991.8 | 24094.5 |
| decode | 5861.2 | 5906.5 | 8258.2 | 8840.7 |
| talker_ar | 21960.2 | 21357.4 | 24968.6 | 25170.5 |
| code2wav | 7.4 | 10.2 | 10.9 | 11.0 |

需要注意：`code2wav` 的 stage input->complete 只表示 terminal payload 完成，不代表 streaming decode 总成本。真正的 streaming 音频窗口在下面：

| interval | count | avg | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| code2wav window collect | 352 | 575.8ms | 40.7ms | 366.5ms | 20093.5ms |
| code2wav decode | 352 | 9.3ms | 9.0ms | 9.4ms | 92.4ms |
| talker_ar -> code2wav stream hop | 1418 | 8.3ms | 7.8ms | 16.0ms | 39.9ms |
| thinker -> decode stream hop | 60 | 941.4ms | 1144.4ms | 1386.4ms | 1535.9ms |

服务端 prefill 合批分布也支持这个判断。以下为 run 目录 server.log 全量统计，包含 prerun 与 actual：

| run log | prefill rows | #new-seq 分布 | avg new-seq | 说明 |
|---|---:|---:|---:|---|
| `best_sil672` | 333 | `1:238, 2:63, 3:16, 4:9, 5:3, 9:1, 10:1, 11:2` | 1.51 | 当前最优 run，仍以单 seq prefill 为主 |
| `profile_sil700` | 717 | `1:560, 2:110, 3:19, 4:12, 5:3, 6:5, 7:1, 8:1, 9:2, 10:1, 11:3` | 1.40 | profiler run，同样合批不足 |
| `trimpartial_8162` | 1119 | `1:904, 2:147, 3:25, 4:24, 5:5, 6:5, 8:2, 9:1, 10:3, 11:3` | 1.35 | 新提交稳定 run，同样没有形成 12 路大 prefill |
| `prefillwait20_8163` | 866 | `1:732, 2:88, 3:19, 4:8, 5:6, 6:3, 7:1, 9:3, 10:3, 11:3` | 1.32 | 未稳定提升合批，且后续 scheduler crash |

## 性能判断

1. 首音频固定约 10s，主要来自 talker 启动门槛。`talker_prefill_start` 在约 9.46s，`first_audio` 在约 10.00s，说明 first audio 不是 code2wav decode 算子慢，而是前面 thinker/mm/talker 的串流门槛。
2. `code2wav decode` 本身均值只有 9.3ms，不是主瓶颈。固定 c2w2 可以更早触发 first audio，但会把 audio chunk 数从约 29 提到约 57，尾部 overhead 变大。
3. image encoder 内核均值约 1.03s，但 `image_encoder stage` 均值 3.25s，说明排队/relay/等待聚合占比较明显。
4. thinker 和 talker 都有 21-22s 级别 streaming stage duration，是 last_audio 与 e2e 的主要来源。
5. SGLang 日志里 actual 阶段多数 prefill 是 `#new-seq: 1`，偶尔 `2`，合批不足；vLLM 的 TTFT 优势很大，下一步应优先看 thinker prefill batching/调度，而不是继续只调 code2wav chunk。
6. `SGLANG_OMNI_PRIORITY_PREFILL_BATCH_WAIT_MS=20` 命中了这个方向，但当前实现不稳定，且全量日志没有显示稳定的合批提升：`sil732` 同 offset tail 有收益，`sil672` 复跑触发 scheduler crash，所以只能作为后续代码修复方向，不能作为交付配置。

## 已验证的负向实验

- `QWEN35_TALKER_ALLOW_PARTIAL_TEXT_CHUNK_BEFORE_DONE=1`：12/12 成功，但 TTFA 到 11.67s、last_audio 到 30.62s，负收益。
- `code2wav-stream-chunk-size=2`：TTFA 到 8.77s，但 last_audio 到 26.80s、actual 到 37.75s，适合首包优先，不适合整体最优。
- `thinker_mem_fraction_static=0.84 + c2w2`：prerun OOM，排除。
- `code2wav dynamic chunk on`：actual 阶段触发 torch/inductor 编译，长时间无 result/wav，当前不可用。
- `SGLANG_OMNI_PRIORITY_PREFILL_BATCH_WAIT_MS=20`：`sil732` 12/12 成功，actual 36.70s、last_audio 25.22s，略好于同 offset 默认 36.78s/26.64s；但 `sil672` 0/12，server 侧 `Committed KV cache already freed`，排除。
- `same-offset warmed sil672`：12/12 成功，actual 29.85s、TTFA 7.44s、last_audio 15.31s，是同请求缓存/热服务上界；随后不同 offset `sil680` 0/12 并触发 GPU0 OOM，因此不作为稳定 best。
- 复用旧 profile 服务直接跑 `sil712` 曾出现 1 个流式响应不完整，server 侧定位到 `video_embeds rows=35200` vs `prompt video_token_id count=34738`。已在 `6115ffd` 加入默认关闭、RTC 配置开启的 partial visual feature trim 兜底；clean restart 后同 offset 12/12 通过，0 mismatch。

## 输出质量

稳定/profile/新 offset/post-commit/same-offset warmed 复验均为 12 个 result.json + 12 个 wav。文本输出正常，主要为“我叫千问 / 通义千问 Omni / Qwen3.5 ...”一类回答；`bang_count=0`，没有纯感叹号异常。不同 offset `sil680` 0/12 无输出，作为负向实验排除。当前 8162 active run `20260630_145321` 日志确认：

```text
500=0
oom=0
mismatch=0
cache_miss_payload_omitted=0
```

## 明早交付前建议

1. 保持默认交付版本为 `mem0.80+c2w4`。
2. 如需强调 TTFA，可同时提供 `mem0.80+c2w2` 作为可选变体，但明确它牺牲尾部。
3. 如果今晚继续优化，优先方向是 thinker actual prefill 合批和缓存命中路径，而不是 code2wav decode。
4. 若修改代码，必须再跑 12/12 barrier-prerun，并确认 0 OOM/500/mismatch/cache miss、12 wav、bang_count=0。
