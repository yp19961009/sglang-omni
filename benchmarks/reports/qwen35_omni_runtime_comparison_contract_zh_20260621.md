# Qwen3.5-Omni Runtime 公平对比合同

生成时间 UTC：`2026-06-21T02:00:21.390464+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页用于把 SGLang-Omni 与 vLLM 的对比边界固定下来：哪些数字可以做
strict apples-to-apples headline，哪些只能做同 runtime scaling，哪些只是
offline diagnostic。它不新增 benchmark 口径，只汇总当前 audit JSON。

## 1. 合同 Gate

| Gate | 当前值 | 判定 |
| --- | --- | --- |
| 硬件/环境 | `8`x NVIDIA H20 / 97871 MiB；SGLang `frankleeeee/sglang-omni:dev / sha256:be7e72126f52`；vLLM `tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589 / sha256:e71dc281e088` | 同一 checkpoint 内可比 |
| 严格横向 workload | `Video-AMME ci-50, warmed skip-first-4, c=4` | 作为唯一 cross-runtime headline |
| headline gate | c=4 latency/RTF win=`True`；quality preserved=`True` | PASS |
| c=8 separation gate | vLLM original prompt-feed limited=`True`；prebuild w4 improves runner wall=`True` | 只能诊断，不能当 online parity |
| confidence gate | high `9`，medium `3`，unsupported `0` | 对外话术受 confidence ledger 约束 |
| final readiness | ready=`True`，contract=`True`，manifest=`196` records，commands=`63` | 复现入口完整 |

## 2. 允许和禁止的比较

| 比较对象 | 可否作为结论 | 允许说法 | 禁止说法 | 证据入口 |
| --- | --- | --- | --- | --- |
| SGLang warmed c=4 vs vLLM warmed c=4 | 可以做主 headline | SGLang latency/RTF 更优，accuracy/WER 不退化 | 把该结论外推到所有并发和所有流量 | `headline_scorecard.json` strict_c4_comparison |
| SGLang c=1/2/4/8/16 | 可以做 SGLang 内部 scaling | c=4-c=8 是推荐窗口，c=8 是吞吐峰值，c=16 是压力边界 | 把 c=16 当默认服务点 | `acceptance_matrix.json`；`tables_summary.json` |
| SGLang c=8 vs vLLM original offline c=8 | 只能做诊断 | vLLM original c=8 受 prompt build/feed admission 限制 | 用 offline runner wall QPS 证明 online parity 或 non-parity | `vllm_admission_diagnosis.json` row vLLM-c8 |
| SGLang c=8 vs vLLM c=8 prebuild w4 | 只能做优化后 offline diagnostic | prebuild w4 缓解 prompt/admission，暴露后续 engine/talker-side tail | 把 prebuild w4 说成严格 online serving-throughput parity | `vllm_admission_diagnosis.json` row vLLM-c8-prebuild-w4 |
| synthetic short/long speech vs Video-AMME | 不能混成同一 headline | short/long 用于守住文本输入和语音输出路径 | 用 synthetic RTF 代替 Video-AMME accuracy/WER | `tables_summary.json` synthetic_speech |
| preproc=2/4 vs baseline c=8 | 可以做 anti-recipe | 朴素 preprocessing 并发放大回退或失败 | 把 preproc=2/4 当当前最优 recipe | `acceptance_matrix.json` anti_recipe rows |

## 3. Strict c=4 横向数字

| Runtime | n | Accuracy | WER | Latency mean/p95 | RTF mean/p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang-Omni | 46 | 67.4% | 4.1% | 1.743s / 3.328s | 1.3536 / 2.4023 |
| vLLM optimized | 46 | 63.0% | 7.4% | 2.093s / 3.525s | 1.4677 / 3.0717 |

SGLang 相对 vLLM 更低：latency mean `16.7%`，latency p95 `5.6%`，RTF mean `7.8%`，RTF p95 `21.8%`。

## 4. vLLM c=8 诊断边界

| vLLM c=8 case | Runner QPS | Engine QPS | Admission avg/p95 | Runner overhead | 诊断 | 对比合同 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| original | 0.1622 | 0.1622 | 33314.0ms / 43972.7ms | 81.8% | prompt_feed_limited | diagnostic_prompt_feed_limited |
| prebuild w4 | 0.2127 | 0.5360 | 4089.0ms / 4891.5ms | 65.6% | engine_or_workload_limited | optimized_offline_diagnostic |

vLLM c=8 prebuild w4 是 offline diagnostic：它证明 prompt/admission 问题被缓解，
但严格 c=8 online serving parity 仍需要 online ingress、同口径 WER/ASR 和
engine/talker boundary 复核。

## 5. Reviewer 使用规则

1. 若要一句话 headline，只引用 warmed c=4 strict runtime comparison。
2. 若讨论高并发服务点，只在 SGLang 内部说 c=4-c=8 推荐窗口和 c=8 吞吐峰值。
3. 若讨论 vLLM c=8，只说 offline runner prompt-feed/admission 诊断和 prebuild w4 改善，不说 online parity 已证明。
4. 若合作方复跑结果硬件、image、模型、数据或 ASR 变化，先按 rerun validation sheet 标注差异，再决定是否替换数字。
5. 若需要新 headline，必须重跑 full audit，且本页、regime decision matrix、confidence ledger、final readiness 全部重新通过。

## 6. 机器证据

- acceptance status counts：`recommended_strict_baseline=1, recommended_serving_window=3, recommended_peak_throughput=1, not_recommended_saturation=1, optimized_offline_diagnostic=1, diagnostic_prompt_feed_limited=1, anti_recipe_regression=1, anti_recipe_failure=1`
- strict c=4 source：`results/qwen35_report_audit_20260619/headline_scorecard.json`
- vLLM c=8 diagnosis source：`results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- regime decisions：`benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md`
- regime decisions JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`
- confidence ledger：`results/qwen35_report_audit_20260619/confidence_ledger.json`
- reproduction command manifest：`results/qwen35_report_audit_20260619/repro_command_manifest.json`
- runtime comparison contract JSON：`results/qwen35_report_audit_20260619/runtime_comparison_contract.json`
