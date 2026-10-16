# Qwen3.5-Omni C12 Delivery Entry

更新时间：2026-06-30 23:00 CST  
目标：sglang-omni 跑 Qwen3.5-Omni，12 并发，40 chunk，交付当前稳定性能最优版本、benchmark、stage 分析和性能报告。

## 先跑这个

```bash
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/check_sglang_qwen35_delivery_ready.sh
```

当前检查结果：`delivery_ready=1`。

这个检查覆盖本地交付文件、CSV 结构、音频合法性、脚本语法、远程 commit、8162 服务健康、benchmark 证据、profile JSON、当前 active run 的配置和 log 错误计数。

## 最终结论

当前稳定性能最优版本：

```text
mem0.80 + code2wav-stream-chunk-size=4 + QWEN35_TRIM_PARTIAL_TRAILING_VISUAL_FEATURES=1
```

远端代码：

```text
/myapp/sglang-omni
6115ffd fix qwen35 rtc partial visual feature trim
```

当前稳定服务：

```text
http://127.0.0.1:8162
active run: /myapp/sglang-omni/results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_145321
config.env: commit=6115ffd
active log: 0 HTTP 500 / 0 OOM / 0 mismatch / 0 cache miss
```

最优指标：

```text
completed=12
failed=0
actual_elapsed_s=36.188
ttfa_avg_ms=10013.029
last_audio_avg_ms=24275.350
e2e_avg_ms=33794.737
bang_count=0
```

当前 8162 干净重启后复验：

```text
completed=12
failed=0
actual_elapsed_s=38.776
ttfa_avg_ms=11899.373
last_audio_avg_ms=25306.965
e2e_avg_ms=31685.279
bang_count=0
```

补充：同 offset 连续 warmed 复跑曾得到 `actual_elapsed_s=29.848`、`ttfa_avg_ms=7438.401`、`last_audio_avg_ms=15308.464`，这是同请求缓存/热服务上界，不作为稳定 best。随后不同 offset `sil680` 复验 0/12 并触发 GPU0 OOM，因此稳定结论仍锁定历史多轮 12/12 的 `36.188s`。

## 读文件顺序

1. 最终交付结论：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_final_delivery_note_20260701.md`
2. 一页稳定结论：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stable_lock_20260630.md`
3. 完整性能报告：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_perf_report_20260630.md`
4. 逐项交付审计：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_delivery_audit_20260630.md`
5. 机器可读 manifest：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_manifest_20260630.json`
6. Benchmark 表格：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_benchmark_summary_20260630.csv`
7. Stage 表格：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_stage_summary_20260630.csv`
8. 音频输出索引：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_index_20260630.md`
9. 音频合法性验证：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_validation_20260630.md`
10. 音频播放页：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_stable_best_audio_player_20260630.html`
11. 最近 ready check 快照：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/delivery_ready_snapshot_20260630.txt`

稳定最优 12 个 wav 已复制到本地：

```text
/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/audio/stable_best_sil672
```

打包交付：

```text
/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz
/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/qwen35_c12_delivery_bundle_20260630.tar.gz.sha256
```

## 复跑命令

只检查/启动稳定服务：

```bash
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_server.sh
```

复跑 12 并发 40 chunk benchmark：

```bash
bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_c12_benchmark.sh
```

默认不会重启 8162；如需强制重启稳定服务：

```bash
FORCE_RESTART=1 bash /Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/run_sglang_qwen35_stable_server.sh
```

## 负向结论

- `mem0.80+c2w2`：TTFA 更快，但 actual/last_audio 变差。
- `mem0.84+c2w4`：TTFT p99 好，但尾部变差且显存余量小。
- `partial_text`：TTFA/last_audio/actual 都变差。
- `mem0.84+c2w2`：prerun OOM。
- `code2wav dynamic chunk`：actual 阶段卡在 inductor 编译，无稳定结果。
- `SGLANG_OMNI_PRIORITY_PREFILL_BATCH_WAIT_MS=20`：`sil732` tail 略好，但 `sil672` 0/12，thinker scheduler crash，不能作为稳定配置。

## Stage 结论

- first audio 约 10s，主要卡在 thinker/mm/talker 启动链路，不是 code2wav decode。
- code2wav decode 均值约 9.3ms，不是主瓶颈。
- image encoder stage 约 3.25s，encoder kernel 约 1.03s，存在排队/relay/聚合等待。
- thinker/talker streaming stage 约 21-22s，是 last_audio/e2e 的主要来源。
- prefill 合批不足，稳定 run avg new-seq 约 1.35-1.51，远没有形成 12 路大 prefill。

## 明早 09:00 Checklist

1. 跑 `check_sglang_qwen35_delivery_ready.sh`，确认 `delivery_ready=1`。
2. 如检查失败，优先看脚本输出的本地文件、远程服务或结果目录错误。
3. 如 8162 不健康，先跑 stable server 脚本恢复服务。
4. 如改过代码或配置，必须重新跑 12/12 benchmark。
5. 没有新代码/配置变更时，直接交付 stable lock、完整报告、manifest 和 CSV。
