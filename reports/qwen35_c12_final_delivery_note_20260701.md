# Qwen3.5-Omni 12 并发 40 chunk 最终交付结论

交付时间目标：2026-07-01 09:00 CST  
当前锁定时间：2026-06-30 23:00 CST  
远端代码：`/myapp/sglang-omni`  
提交：`6115ffd fix qwen35 rtc partial visual feature trim`

## 结论

当前稳定性能最优版本建议使用：

```text
mem0.80 + code2wav-stream-chunk-size=4 + QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
```

当前 8162 服务已按该配置从 `6115ffd` 干净启动，健康检查通过，active log 仍为 0 HTTP 500 / 0 OOM / 0 mismatch / 0 cache miss。该代码和配置已经完成多轮同口径 12/12 `sil672` 复验。

## 最优指标

12 并发、40 chunk、barrier-prerun、text+audio：

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

稳定性验证：

- 多轮 12/12 成功。
- 当前 active run `config.env` 已记录 `commit=6115ffd`，服务健康且 active log 干净。
- Post-commit 复验 `completed=12 failed=0 actual=38.776s bang_count=0`。
- 同 offset warmed 复跑 `actual=29.848s` 只作为缓存/热服务上界；不同 offset `sil680` 0/12 并触发 GPU0 OOM，因此不把 29.848s 作为稳定 best。
- 0 HTTP 500。
- 0 OOM。
- 0 feature/token mismatch。
- 0 omitted-payload cache miss。
- 输出为 12 wav + 12 result，文本正常。

## 为什么选它

- `c2w2` 能把 TTFA 降到约 8.77s，但 actual 和 last_audio 变差，不适合作为整体稳定最优。
- `mem0.84+c2w4` TTFT p99 更好，但尾部更差且显存余量更小。
- `partial_text` 是负收益。
- `dynamic chunk` 当前不可用，会卡在 inductor 编译。
- `same-offset warmed 29.848s` 是缓存/热服务上界，独立 `sil680` 验证失败，不作为稳定 best。
- `SGLANG_OMNI_PRIORITY_PREFILL_BATCH_WAIT_MS=20` 在一个 offset 上 tail 略好，但另一个 offset 0/12，thinker scheduler crash，因此排除。

## Stage 判断

主要瓶颈不在 code2wav decode：

- first audio 约 10s，主要来自 thinker/mm/talker 启动链路。
- code2wav decode 均值约 9.3ms。
- image encoder stage 约 3.25s，其中 encoder kernel 约 1.03s，存在排队/relay/聚合等待。
- thinker/talker streaming stage 约 21-22s，是 last_audio/e2e 主要来源。
- prefill 合批不足，稳定 run 平均 new-seq 约 1.35-1.51，远未形成 12 路大 prefill。

## 交付入口

先跑：

```bash
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/check_sglang_qwen35_delivery_ready.sh
```

期望输出：`delivery_ready=1`。

主要文件：

- README：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/README_qwen35_c12_delivery_20260630.md`
- 稳定锁定：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stable_lock_20260630.md`
- 完整报告：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_perf_report_20260630.md`
- 交付审计：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_delivery_audit_20260630.md`
- Benchmark CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_summary_20260630.csv`
- Stage CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stage_summary_20260630.csv`
- 音频索引：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_index_20260630.md`
- 音频合法性验证：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_validation_20260630.md`
- 音频播放页：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_stable_best_audio_player_20260630.html`
- Bundle：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz`
