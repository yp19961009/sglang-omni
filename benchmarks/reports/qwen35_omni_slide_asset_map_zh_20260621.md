# Qwen3.5-Omni 分享 Deck 图表资产映射

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：把 deck 提纲中的每个核心 claim 映射到可直接放 PPT 的 SVG/CSV 或现场证据 JSON，避免手工挑图或手改数字。

## 1. 使用规则

1. PPT 中优先嵌入 SVG；需要表格或复核数字时打开同名 CSV。
2. 不要手工改图里的数字；图表来自 `share_charts/chart_pack_manifest.json` 记录的审计 JSON。
3. 没有图的复现/追问页使用 JSON 或 Markdown 证据，不伪造图。
4. 若重跑后数字变化，先重建 share chart pack、slide asset map、share bundle，再跑 package validation。

## 2. 当前 Gate

| Gate | Value |
| --- | ---: |
| Ready | True |
| Rows | 10 |
| Checks | 5/5 |
| Chart assets | 14 |
| Missing assets | 0 |
| Missing from chart manifest | 0 |

## 3. Slide 资产映射

| Deck section | Claim | Primary asset | Data / proof asset | Speaker note |
| --- | --- | --- | --- | --- |
| 3. Headline: warmed c=4 严格横向对比 | SGLang latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。 | `results/qwen35_report_audit_20260619/share_charts/strict_c4_latency_rtf.svg` | `results/qwen35_report_audit_20260619/share_charts/strict_c4_runtime_comparison.csv` | 先讲 warmed skip-first-4，再讲 latency/RTF 四项优势和 WER 边界。 |
| 4. SGLang 压测曲线：c=1/2/4/8/16 | c=8 是当前吞吐峰值，c=16 是 saturation boundary。 | `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_qps.svg` | `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_sweep.csv` | 用 QPS 图讲峰值，再用表格解释 c=16 不推荐。 |
| 4. SGLang 压测曲线：latency tail | c=16 tail 变差是 admission/queueing 饱和信号。 | `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_latency.svg` | `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_sweep.csv` | 不要把 c=16 说成失败；它是压力边界证据。 |
| 5. Stage breakdown：哪里慢 | c=8/c=16 preprocessing lifecycle 增长主要是 queue/admission。 | `results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget_pct.svg` | `results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget.csv` | 强调 lifecycle 和 actual compute 的区别。 |
| 6. Stage 连接：是不是卡在 stage 之间 | talker_ar -> code2wav handoff 健康，code2wav decode 不是主瓶颈。 | `results/qwen35_report_audit_20260619/share_charts/sglang_handoff_decode_ms.svg` | `results/qwen35_report_audit_20260619/share_charts/stage_connection_health.csv` | 用 hop p95 和 decode ms 回答 stage boundary 追问。 |
| 7. 短/长文本输入 + 语音输出 guardrail | short/long c=1/4/8 均覆盖，long c=8 仍快于实时。 | `results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_rtf.svg` | `results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv` | 同步说清 synthetic long 不是 official SeedTTS full-set headline。 |
| 8. 负优化：为什么不直接放大 preprocessing 并发 | preproc=2 回退，preproc=4 OOM/失败，朴素加并发不是当前 recipe。 | `results/qwen35_report_audit_20260619/share_charts/preprocessing_antirecipe.csv` | `results/qwen35_report_audit_20260619/share_charts/preprocessing_antirecipe.csv` | 这一页用 CSV 表即可；重点讲共享资源争用。 |
| 9. vLLM c=8：offline diagnostic 边界 | prebuild w4 是当前最强 offline diagnostic，不是 online serving parity。 | `results/qwen35_report_audit_20260619/share_charts/vllm_c8_diagnostic_qps.svg` | `results/qwen35_report_audit_20260619/share_charts/vllm_admission_diagnosis.csv` | 分清 runner QPS、engine QPS 和 admission span。 |
| 11. 复现路径 | 先 full audit，再 receiver smoke/extracted-only，再重跑性能。 | `results/qwen35_report_audit_20260619/audit_run_summary.json` | `results/qwen35_report_audit_20260619/repro_command_manifest.json` | 这页不放性能图；放命令和 expected gates。 |
| 14. 被追问时的证据跳转 | 复跑数字偏移时先用 delta triage，不要直接改 headline。 | `benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md` | `results/qwen35_report_audit_20260619/rerun_delta_triage.json` | 现场把症状映射到 stage/boundary、裁决和下一步动作。 |

## 4. 发送前检查

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"
python3 -m benchmarks.eval.build_qwen35_omni_slide_asset_map \
  --root "$HOST_REPO" \
  --strict \
  --output benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/slide_asset_map.json
python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```
