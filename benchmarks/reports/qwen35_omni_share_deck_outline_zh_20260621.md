# Qwen3.5-Omni 性能分析分享 Deck 提纲

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
主报告：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
最终分享交付说明：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md`
一页式核心数字 scorecard：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`
分享 Deck 图表资产映射：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md`
外部复现 handoff runbook：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
合作方复跑验收表：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
PPT 图表包：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/`
机器审计：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`

这份提纲用于给合作高校做 15-25 分钟技术分享。它不是替代完整报告，而是把
headline、stage breakdown、瓶颈解释、vLLM 边界和复现路径整理成可直接做 PPT 的顺序。

## 1. 开场：我们要证明什么

主标题：

SGLang-Omni Qwen3.5 在语音输出 workload 上的性能、瓶颈和可复现证据

要讲的三句话：

- 在 8x H20、Video-AMME ci-50、warmed c=4 严格横向对比中，SGLang-Omni
  latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。
- SGLang-Omni 当前推荐窗口是 c=4-c=8；c=8 是吞吐峰值，c=16 是 admission
  饱和边界。
- 当前主要瓶颈不是 stage handoff，也不是 code2wav decode；后续应优先优化
  talker AR 和 c=8 附近 admission 策略。

证据入口：

- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- `results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json`
- `benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md`
- `results/qwen35_report_audit_20260619/confidence_ledger.json`

边界提醒：

- 不把官方 SeedTTS full-set 作为 headline。
- 不把 vLLM c=8 prebuild w4 说成 online serving parity。

## 2. 实验环境和对比边界

推荐展示：

| 项 | 值 |
| --- | --- |
| GPU | 8x NVIDIA H20, 97.9GB |
| SGLang image | `frankleeeee/sglang-omni:dev` |
| vLLM image | `tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589` |
| Model | `qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315` |
| Main workload | Video-AMME ci-50, text+audio output |

讲法：

- 先强调 vLLM baseline 不是弱 baseline：使用 Qwen3.5-capable 镜像、compile
  mode、CUDA graph、code2wav compile、prefix caching/chunked prefill 等优化路径。
- SGLang 同样开了 thinker/talker CUDA graph、talker torch compile、code2wav
  compile，并固定当前安全的 `PREPROCESSING_MAX_CONCURRENCY=1`。

证据入口：

- `results/qwen35_report_audit_20260619/environment_snapshot.json`
- 完整报告 section 3 和 section 4.1

## 3. Headline：warmed c=4 严格横向对比

推荐展示：

| Runtime | Lat Mean | Lat P95 | RTF Mean | RTF P95 | Accuracy | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang | 1.743s | 3.328s | 1.3536 | 2.4023 | 67.4% | 4.12% |
| vLLM | 2.093s | 3.525s | 1.4677 | 3.0717 | 63.0% | 7.44% |

讲法：

- warmed skip-first-4 避免首次 compile / CUDA graph capture 污染稳态性能。
- SGLang latency mean 低 16.7%，latency p95 低 5.6%，RTF mean 低 7.8%，RTF
  p95 低 21.8%。
- accuracy 和 WER 同时不退化，因此不是用质量换速度。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/strict_c4_latency_rtf.svg`
- `results/qwen35_report_audit_20260619/share_charts/strict_c4_runtime_comparison.csv`
- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- `results/qwen35_report_audit_20260619/claims_verification.json`

## 4. SGLang 压测曲线：c=1/2/4/8/16

推荐展示：

| c | Accuracy | Latency Mean | Latency P95 | QPS | WER | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 70.0% | 1.316s | 2.406s | 0.760 | 3.85% | talker tail |
| 2 | 70.0% | 1.508s | 3.124s | 1.315 | 3.85% | talker tail |
| 4 | 70.0% | 1.929s | 3.633s | 2.036 | 3.85% | 推荐窗口 |
| 8 | 70.0% | 3.064s | 5.853s | 2.540 | 3.23% | 吞吐峰值 |
| 16 | 70.0% | 6.066s | 7.846s | 2.407 | 2.88% | 饱和边界 |

讲法：

- c=8 是当前 recipe 的最高吞吐点。
- c=16 的 accuracy/WER 没崩，但 QPS 下降、RTF 和 queue tail 变差，所以不是推荐运行点。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_qps.svg`
- `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_latency.svg`
- `results/qwen35_report_audit_20260619/share_charts/sglang_pressure_sweep.csv`
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/tables_summary.json`

## 5. Stage breakdown：哪里慢

推荐展示：

| Regime | 主要限制 | 证据 | 优化方向 |
| --- | --- | --- | --- |
| c=1/c=2/c=4 | `talker_ar` tail | top stage 是 talker；请求耗时随生成音频长度变长 | talker AR 效率、batching、每步 overhead |
| c=8 | preprocessing admission + talker tail | preprocessing lifecycle 1.23s，实际 compute 0.29s | admission 策略，避免过度塞满 thinker |
| c=16 | queueing/saturation | QPS 低于 c=8，短输出 tail 也被 queue 主导 | 当前不推荐；降低 admission 或分片 |
| long speech | talker AR compute | long c=8 talker 25.6s avg，decode 约 14ms/window | talker AR、chunk cadence |

讲法：

- 不要只看 stage lifecycle，要区分实际 compute 和排队/admission。
- c=8/c=16 的 preprocessing 变长主要是队列，不是 video preprocessing 算子突然变慢。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget_pct.svg`
- `results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget.csv`
- 完整报告 section 7.1、12、13
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`

## 6. Stage 连接：是不是卡在 stage 之间

推荐展示：

| Boundary | 证据 | 判断 |
| --- | --- | --- |
| request -> preprocessing | c=8/c=16 lifecycle 增长，actual compute 稳定 | admission/queue 压力 |
| preprocessing -> encoder/thinker | preproc=2 让 media/encoder/thinker 都变慢 | 共享资源争用 |
| thinker -> talker | vLLM thinker-to-talker feed p95 约 1ms | handoff 不是主瓶颈 |
| talker -> code2wav | SGLang hop p95 约 15-24ms | 连接健康 |
| code2wav collect -> decode | decode 14-17ms/window，collect 更大 | 等 codec chunks，不是 vocoder 算不动 |

讲法：

- 当前 stage 之间的连接不是主瓶颈。
- code2wav 不应作为第一优化方向；真正优先的是 talker AR 和 admission。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/sglang_handoff_decode_ms.svg`
- `results/qwen35_report_audit_20260619/share_charts/stage_connection_health.csv`
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `results/qwen35_report_audit_20260619/claims_verification.json`

## 7. 短/长文本输入 + 语音输出 guardrail

推荐展示：

| Scenario | Text Input | c | Audio Mean | Latency Mean | RTF Mean | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| short | 74 chars / 12 words | 1 | 4.2s | 0.866s | 0.2052 | 快于实时 |
| short | 74 chars / 12 words | 4 | 4.3s | 1.768s | 0.4105 | 快于实时 |
| short | 74 chars / 12 words | 8 | 4.3s | 2.638s | 0.6257 | 快于实时 |
| long | 944 chars / 139 words | 1 | 51.9s | 9.168s | 0.1766 | 快于实时 |
| long | 944 chars / 139 words | 4 | 52.6s | 17.551s | 0.3338 | 快于实时 |
| long | 944 chars / 139 words | 8 | 52.3s | 25.799s | 0.4932 | 快于实时 |

讲法：

- 这页回答“长文本输入/长语音输出会不会拖垮”。
- long c=8 仍明显快于实时，但它是 synthetic speech guardrail，不替代官方 SeedTTS full-set。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_rtf.svg`
- `results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv`
- `results/qwen35_report_audit_20260619/tables_summary.json`
- `results/qwen35_report_audit_20260619/confidence_ledger.json`

## 8. 负优化：为什么不直接放大 preprocessing 并发

推荐展示：

| Setting | Completed | Failed | Latency Mean | QPS | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| preproc=1 baseline | 50 | 0 | 3.064s | 2.540 | 当前安全点 |
| preproc=2 | 50 | 0 | 4.579s | 1.642 | QPS 下降 35.4% |
| preproc=4 | 43/50 useful | 7 | n/a | n/a | OOM/失败风险 |

讲法：

- 这页回答“既然 c=8/c=16 有 preprocessing queue，为什么不加 preprocessing worker”。
- 当前 H20 布局下，朴素加并发会把 queue 问题变成 GPU0/encoder/thinker 争用。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/preprocessing_antirecipe.csv`
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- 完整报告 section 9

## 9. vLLM c=8：为什么不能直接拿 offline wall QPS 做结论

推荐展示：

| vLLM c=8 Artifact | Prompt Build Wall | Runner Wall | Runner QPS | Engine QPS | Admission Span | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original c8 | n/a | 308.2s | 0.1622 | 0.1622 | 33.3s / 44.0s | prompt-feed limited |
| prebuild w1 | 249.3s | 352.1s | 0.1420 | 0.5391 | 4.44s / 5.43s | admission 被移除 |
| prebuild w4 | 129.2s | 235.1s | 0.2127 | 0.5360 | 4.09s / 4.89s | 最强 offline 诊断 |

讲法：

- 原始 vLLM c=8 主要慢在 host prompt build/feed admission，不是 engine stage 边界。
- prebuild w4 是更强的 offline diagnostic，但不能当 online serving parity。

证据入口：

- `results/qwen35_report_audit_20260619/share_charts/vllm_c8_diagnostic_qps.svg`
- `results/qwen35_report_audit_20260619/share_charts/vllm_admission_diagnosis.csv`
- `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- `results/qwen35_report_audit_20260619/vllm_log_stage_summary.json`

## 10. 可说的话和不能说的话

可高置信说：

- warmed c=4 上 SGLang-Omni latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。
- SGLang c=8 是当前 recipe 的吞吐峰值，c=16 是饱和边界。
- 当前 stage handoff 和 code2wav decode 不是主瓶颈。
- preproc=2/4 是当前 recipe 的 anti-recipe。

必须带边界说：

- 官方 SeedTTS full-set 当前没有本地完整 headline。
- vLLM c=8 prebuild w4 是 offline diagnostic，不是 online serving parity。
- ci-50 结果不能直接外推为全量线上流量结论；ci-50/stress/synthetic 证据不能直接外推到完整线上流量。

证据入口：

- `results/qwen35_report_audit_20260619/confidence_ledger.json`

## 11. 复现路径

推荐展示：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

期望 gate：

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- SGLang optimization lock `26/26`
- vLLM optimization lock `22/22`
- repro command manifest `63` commands / `7` phases
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`
- final readiness `49/49`
- share package validation `17/17`
- receiver smoke validation `ready=true`
- extracted-only package validation `13/13`

证据入口：

- `benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- `results/qwen35_report_audit_20260619/manifest.json`

## 12. 收尾：后续优化路线

推荐展示：

| 优先级 | 方向 | 原因 |
| --- | --- | --- |
| P0 | 保持当前 code2wav compile 路径 | decode 不是瓶颈，别破坏稳定路径 |
| P1 | talker AR efficiency | c=1/c=4/long speech tail 都指向 talker |
| P1 | c=8 admission 策略 | c=8 是峰值但 queue 已出现 |
| P2 | preprocessing placement/admission 联合调参 | 单独放大 preprocessing 已证实是负优化 |
| P2 | vLLM online ingress + WER/ASR | 严格 c=8 parity 需要在线路径证据 |
| P2 | 官方 SeedTTS full-set | 补自然语音 full-set headline |

最后一句：

当前证据已经足以支持“优化后的 SGLang-Omni Qwen3.5 在该语音输出压力场景下至少与
优化版 vLLM 相当，并在 warmed c=4 严格对比中更快”的对外表述；后续优化重点应从
code2wav 转向 talker AR 和 admission。

## 13. 15 分钟讲稿节奏

如果时间只有 15 分钟，按下面节奏讲，避免在单个表格里停太久：

| 时间 | 内容 | 讲法 |
| ---: | --- | --- |
| 0-2 min | 目标和边界 | 先讲 workload、8x H20、warmed c=4 严格对比、c=8/c=16 压力边界 |
| 2-5 min | Headline 对比 | 用 SGLang/vLLM c=4 表说明 latency/RTF 全赢且 WER/accuracy 不退化 |
| 5-8 min | SGLang 压测曲线 | 讲 c=8 是吞吐峰值，c=16 是饱和边界，不把高并发 tail 美化成推荐点 |
| 8-11 min | Stage breakdown | 讲 talker AR、admission/queueing、code2wav decode、stage handoff 四件事 |
| 11-13 min | vLLM baseline 和 caveat | 说明 vLLM baseline 已优化，c=8 prebuild w4 是 offline diagnostic，不是 online parity |
| 13-15 min | 复现和验收 | 指向 full audit、repro command manifest、handoff runbook、receiver smoke 和 extracted-only validation |

这一版讲稿的重心是“证据链可信”，不是只展示一个 headline 数字。每页只做一个 claim：
SGLang 在严格 c=4 对比中更快；SGLang c=8 是当前峰值；stage 连接健康；vLLM
baseline 已优化但 c=8 仍只能讲 offline diagnostic。

## 14. 被追问时的证据跳转

| 追问 | 先回答 | 立即打开的证据 |
| --- | --- | --- |
| vLLM baseline 会不会太弱？ | 不是弱 baseline；镜像、compile、CUDA graph、prebuild w4 都锁了 | `qwen35_omni_vllm_optimization_lock_zh_20260621.md`；`runtime_image_contract.json` |
| 为什么只把 c=4 当 strict headline？ | c=4 是 warmed apples-to-apples；c=8 vLLM 目前是 offline diagnostic | `qwen35_omni_runtime_comparison_contract_zh_20260621.md` |
| c=8 高并发谁更强？ | SGLang c=8 是当前吞吐峰值；vLLM c=8 需要 online ingress 后才能 strict parity | `vllm_online_parity_protocol.json`；`vllm_admission_diagnosis.json` |
| stage 之间是不是卡住？ | 当前证据显示 handoff 健康，主要压力在 talker AR 和 admission/queueing | `stage_boundary_bottleneck_ledger.json`；`stage_interaction_summary.json` |
| code2wav 是不是瓶颈？ | 不是当前主瓶颈；decode 约十几毫秒，collect wait 多数是在等 talker chunk | `qwen35_omni_stage_metric_dictionary_zh_20260621.md` |
| 长文本/长语音会不会失控？ | long synthetic c=8 仍快于实时，但它是 guardrail，不替代官方 full-set | `tables_summary.json`；`confidence_ledger.json` |
| PPT 该插哪张图？ | 直接按 slide asset map，不手工改 SVG/CSV 数字 | `qwen35_omni_slide_asset_map_zh_20260621.md`；`chart_pack_manifest.json` |
| 能不能复现？ | 可以，从 full audit 到 63 条命令 manifest，再到 tarball receiver smoke 和 standalone validation 都有路径 | `qwen35_omni_external_handoff_runbook_zh_20260621.md`；`repro_command_manifest.json` |
| 复跑数字什么时候能替换主报告？ | 只有同硬件/同 image/同模型/同数据且 all gates green 才能替换 | `qwen35_omni_rerun_acceptance_contract_zh_20260621.md` |
| 复跑数字偏了先看哪里？ | 先按 delta triage 映射到 stage/boundary，不直接改 headline | `qwen35_omni_rerun_delta_triage_zh_20260621.md`；`rerun_delta_triage.json` |
