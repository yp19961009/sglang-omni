# sglang-omni Qwen3.5-Omni C12 Stable Lock

锁定时间：2026-06-30 23:00 CST  
远端代码：`/myapp/sglang-omni`  
代码提交：`6115ffd fix qwen35 rtc partial visual feature trim`

## 稳定推荐

当前稳定性能最优版本：

```text
mem0.80 + code2wav-stream-chunk-size=4 + QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
```

推荐理由：

- 12 并发、40 chunk、barrier-prerun 口径下多次 12/12 成功。
- 当前最好 actual：`36.19s`。
- 当前最好 last_audio avg：`24.28s`。
- 当前最好 TTFA avg：`10.01s`。
- 已验证 0 OOM、0 HTTP 500、0 feature/token mismatch、0 omitted-payload cache miss。
- 输出质量正常：12 wav + 12 result，`bang_count=0`。

## 推荐配置

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
QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
QWEN35_TALKER_ALLOW_PARTIAL_TEXT_CHUNK_BEFORE_DONE=0
QWEN35_TALKER_PARTIAL_TEXT_CHUNK_WAIT_SKIPS=0
QWEN35_TALKER_ALLOW_UNBOUNDED_EMPTY_TEXT_FEEDBACK=0
--thinker-mem-fraction-static 0.80
--thinker-max-running-requests 12
--talker-max-running-requests 12
--code2wav-stream-chunk-size 4
--thinker-cuda-graph off
--talker-cuda-graph on
--talker-torch-compile on
```

当前 8162 服务已按该配置从 `6115ffd` 干净启动，健康检查通过，active run log 干净；该配置 post-commit `sil672` 复验 12/12 成功。

## 核心 Benchmark

| run | status | actual | TTFT avg/p99 | TTFA avg/p99 | last_audio avg/p99 | e2e avg/p99 |
|---|---:|---:|---:|---:|---:|---:|
| `sg_mem080_c2w4_best_run12` | 12/12 | 36.19s | 7.81s / 12.81s | 10.01s / 10.07s | 24.28s / 27.47s | 33.79s / 36.18s |
| `sg_mem080_c2w4_profile` | 12/12 | 36.64s | 7.70s / 16.89s | 10.01s / 10.06s | 28.35s / 32.86s | 34.47s / 36.63s |
| `sg_mem080_c2w4_trimpartial_sil712` | 12/12 | 36.64s | 9.56s / 11.65s | 12.01s / 12.47s | 27.34s / 31.41s | 34.50s / 36.63s |
| `sg_mem080_c2w4_trimpartial_sil724` | 12/12 | 37.31s | 10.57s / 17.90s | 10.79s / 12.76s | 28.88s / 33.98s | 33.42s / 37.30s |
| `sg_mem080_c2w4_post_commit_sil672` | 12/12 | 38.78s | 9.96s / 16.50s | 11.90s / 13.91s | 25.31s / 30.54s | 31.69s / 38.76s |
| `sg_mem080_c2w4_warmed_sil672_same_offset` | 12/12 | 29.85s | 7.23s / 11.23s | 7.44s / 10.47s | 15.31s / 18.95s | 21.09s / 29.84s |
| `sg_mem080_c2w4_trimpartial_sil732` | 12/12 | 36.78s | 8.45s / 13.94s | 9.38s / 12.43s | 26.64s / 32.69s | 31.03s / 36.77s |

## 明确排除

| variant | result | reason |
|---|---:|---|
| `mem0.80+c2w2` | 12/12 | TTFA 改善到 8.77s，但 actual/last_audio 变差，不作为整体最优 |
| `mem0.84+c2w4` | 12/12 | TTFT p99 好，但尾部差且显存余量更小 |
| `partial_text` | 12/12 | TTFA/last_audio/actual 全部变差 |
| `mem0.84+c2w2` | failed | prerun OOM |
| `code2wav dynamic chunk` | invalid | actual 阶段触发 inductor 编译，无稳定 metrics/wav |
| `SGLANG_OMNI_PRIORITY_PREFILL_BATCH_WAIT_MS=20` | unstable | `sil732` tail 略好，但 `sil672` 0/12，thinker scheduler crash: `Committed KV cache already freed` |
| `same-offset warmed sil672 29.85s` | not selected | 同请求缓存/热服务上界；随后不同 offset `sil680` 0/12 且 GPU0 OOM，不能作为稳定 best |

## Stage 判断

主要瓶颈：

- first audio 约 10s，主要卡在 thinker/mm/talker 启动链路，不是 code2wav decode。
- code2wav decode 均值约 9.3ms，不是主瓶颈。
- image encoder stage 均值约 3.25s，内核约 1.03s，说明排队/relay/聚合等待占比较明显。
- thinker 与 talker streaming stage 都是 21-22s 级别，是 last_audio/e2e 的主要来源。
- 服务端 prefill 合批不足：稳定 run 的 avg new-seq 约 1.35-1.51，远没有形成 12 路大 prefill。

vLLM 参考 TTFA 约 2.5-3.2s；当前 SGLang 稳定最优 TTFA 约 10.0s，差距主要在 thinker prefill batching/调度和 talker 首音频门槛。

## 复跑入口

本地脚本：

```bash
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/check_sglang_qwen35_delivery_ready.sh
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_server.sh
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_c12_benchmark.sh
```

核心证据：

- 交付入口 README：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/README_qwen35_c12_delivery_20260630.md`
- 最终交付结论：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_final_delivery_note_20260701.md`
- 交付 bundle：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz`
- 交付 bundle SHA256：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz.sha256`
- 最近 ready check 快照：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/delivery_ready_snapshot_20260630.txt`
- 完整报告：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_perf_report_20260630.md`
- Manifest：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_manifest_20260630.json`
- Benchmark summary CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_summary_20260630.csv`
- Stage summary CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stage_summary_20260630.csv`
- 音频输出索引：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_index_20260630.md`
- 音频合法性验证：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_validation_20260630.md`
- 音频播放页：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_stable_best_audio_player_20260630.html`
- 稳定最优本地音频：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/audio/stable_best_sil672`
- 交付就绪检查脚本：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/check_sglang_qwen35_delivery_ready.sh`
- 稳定服务启动脚本：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_server.sh`
- Benchmark 复跑脚本：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_c12_benchmark.sh`
- 稳定最优结果：`/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_8162_20260630_103902/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_104103`
- Post-commit active run：`/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_142334`
- Same-offset warmed upper bound：`/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_143808`
- Failed independent offset：`/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil680_stagger0_temp1_barrier_validation_144121`
- Profile JSON：`/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_20260630_122447/client_c12_realtime_audio_vllmstyle_sil700_stagger0_temp1_barrier_profile_stable_c12_c2w4_122641/request_profile_stable_c12_c2w4_122641.json`
