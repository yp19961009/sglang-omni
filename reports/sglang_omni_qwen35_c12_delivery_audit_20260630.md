# sglang-omni Qwen3.5-Omni C12 Delivery Audit

审计时间：2026-06-30 23:00 CST  
目标：Qwen3.5-Omni，12 并发，40 chunk，稳定性能最优版本、benchmark、stage 分析、性能报告。

## 审计结论

当前交付证据已覆盖目标里的核心要求：

- 稳定性能最优版本已锁定：`mem0.80 + code2wav-stream-chunk-size=4 + QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1`
- 代码已提交：`6115ffd fix qwen35 rtc partial visual feature trim`
- 远端仓库干净，当前 8162 稳定服务健康，active run `20260630_145321` 的 `config.env` 已记录 `commit=6115ffd`。
- benchmark 证据覆盖 12 并发、40 chunk、barrier-prerun。
- stage 分析覆盖关键 path、stage duration、stream interval、prefill 合批分布。
- 负向实验已记录，包含 wait20、c2w2、mem0.84、partial text、dynamic chunk。

## Requirement Audit

| requirement | status | evidence |
|---|---:|---|
| 12 并发 | pass | manifest `benchmark_shape.concurrency=12`，所有核心 run `completed=12` |
| 40 chunk | pass | manifest `benchmark_shape.trunk_size=40` |
| Qwen3.5-Omni | pass | manifest `model=qwen3_5-omni`，当前 8162 `/v1/models` 返回 `qwen3_5-omni` |
| 稳定性能最优版本 | pass | stable lock 选定 `mem0.80+c2w4+trimpartial`，最优 actual `36.19s` |
| benchmark | pass | manifest、report、远端 results 路径、复跑脚本均存在 |
| stage 性能分析 | pass | profile JSON 与报告中 milestone/stage/stream interval/prefill batching 表 |
| 性能分析报告 | pass | `sglang_omni_qwen35_c12_perf_report_20260630.md` |
| 可复跑入口 | pass | `run_sglang_qwen35_stable_server.sh` 与 `run_sglang_qwen35_stable_c12_benchmark.sh` |
| 输出质量 | pass | 稳定/profile/验证/post-commit/same-offset warmed run 均 12 wav + 12 result，`bang_count=0`；`sil680` 失败 run 无输出并已排除 |
| 失败/负向实验记录 | pass | c2w2、mem0.84、partial text、dynamic chunk、wait20 均已记录 |

## Current Stable Version

远端目录：`/myapp/sglang-omni`  
提交：`6115ffd fix qwen35 rtc partial visual feature trim`  
当前服务：`http://127.0.0.1:8162`  

推荐配置摘要：

```text
CUDA_VISIBLE_DEVICES=3,4,5
thinker_mem_fraction_static=0.80
thinker_max_running_requests=12
talker_max_running_requests=12
code2wav_stream_chunk_size=4
SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE=8
SGLANG_OMNI_TALKER_READY_SUBSET_MIN_SIZE=1
SGLANG_OMNI_RELAY_SLOT_SIZE_MB=1024
SGLANG_OMNI_RELAY_CREDITS=2
SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS=1
QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
```

## Best Metrics

稳定最优结果目录：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_8162_20260630_103902/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_104103
```

核心指标：

```text
completed=12
failed=0
actual_elapsed_s=36.188
ttft_avg_ms=7808.845
ttft_p99_ms=12813.277
ttfa_avg_ms=10013.029
ttfa_p99_ms=10067.803
last_audio_avg_ms=24275.350
last_audio_p99_ms=27465.785
e2e_avg_ms=33794.737
e2e_p99_ms=36180.092
audio_duration_avg_s=8.967
bang_count=0
```

Post-commit C12 复验证据：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_142334
completed=12
failed=0
actual_elapsed_s=38.776
ttfa_avg_ms=11899.373
last_audio_avg_ms=25306.965
e2e_avg_ms=31685.279
bang_count=0
```

同 offset warmed 上界与独立 offset 负向证据：

```text
same_offset_warmed_sil672:
completed=12
failed=0
actual_elapsed_s=29.848
ttfa_avg_ms=7438.401
last_audio_avg_ms=15308.464

independent_sil680:
completed=0
failed=12
failure=ClientPayloadError after GPU0 OOM
```

## Stage Evidence

Profile JSON：

```text
/myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_20260630_122447/client_c12_realtime_audio_vllmstyle_sil700_stagger0_temp1_barrier_profile_stable_c12_c2w4_122641/request_profile_stable_c12_c2w4_122641.json
```

主要判断：

- first audio 约 10s，主要来自 thinker/mm/talker 启动链路。
- code2wav decode 均值约 9.3ms，不是主瓶颈。
- image encoder stage 约 3.25s，encoder kernel 约 1.03s，存在排队/relay/聚合等待。
- thinker/talker streaming stage 约 21-22s，是 last_audio/e2e 的主要来源。
- 服务端 prefill 合批均值只有约 1.35-1.51 new-seq，未形成 12 路大 prefill。

## Remaining Risk

- 当前 8162 服务已在 `sil680` 负向实验后重新从 `6115ffd` 干净启动，active log 为 0 OOM/500/mismatch/cache miss；历史 best 仍作为性能最优稳定样本保留，post-commit 和 warmed runs 作为补充证据。
- SGLang 相对 vLLM 仍有明显 TTFA 差距；报告判断后续方向是 thinker prefill batching/调度和 talker 首音频门槛。
- 明早 09:00 前如修改任何代码或配置，必须重新跑 12/12 barrier-prerun，并确认 0 OOM/500/mismatch/cache miss、12 wav、`bang_count=0`。

## Artifacts

- Delivery README：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/README_qwen35_c12_delivery_20260630.md`
- Final delivery note：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_final_delivery_note_20260701.md`
- Delivery bundle：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz`
- Delivery bundle SHA256：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz.sha256`
- Delivery ready snapshot：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/delivery_ready_snapshot_20260630.txt`
- Stable lock：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stable_lock_20260630.md`
- Full report：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_perf_report_20260630.md`
- Manifest：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_manifest_20260630.json`
- Benchmark summary CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_summary_20260630.csv`
- Stage summary CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stage_summary_20260630.csv`
- Audio index：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_index_20260630.md`
- Audio validation：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_validation_20260630.md`
- Audio player HTML：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_stable_best_audio_player_20260630.html`
- Stable best local audio：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/audio/stable_best_sil672`
- Delivery ready check：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/check_sglang_qwen35_delivery_ready.sh`
- Stable server script：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_server.sh`
- Benchmark script：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_c12_benchmark.sh`
