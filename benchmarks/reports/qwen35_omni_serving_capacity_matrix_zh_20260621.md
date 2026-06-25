# Qwen3.5-Omni Serving/Capacity 决策矩阵

生成时间 UTC：`2026-06-21T02:00:23.122395+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份矩阵把已审计的压力测试结果翻译成运行选择：哪些是服务窗口，哪些只是压力边界或诊断证据。
所有数字来自 `tail_confidence_appendix.json`、`stage_latency_budget.json` 和 `stage_boundary_bottleneck_ledger.json`。

## 1. Gate

- ready：`True`，checks：`10/10`，rows：`7`。
- SGLang c8 QPS：`2.54`；SGLang c16 QPS：`2.407`。
- long c8 RTF p95：`0.50008`；vLLM w4 scope：`offline_diagnostic_only`。

## 2. 决策矩阵

| 压力/场景 | Runtime | 运行选择 | 可承诺指标 | Stage guard | 不要做 | Rerun IDs |
| --- | --- | --- | --- | --- | --- | --- |
| Video-AMME c=1-c2 | sglang | latency-first / single-to-low concurrency | QPS=0.760-1.315; lat_p95=2.406s-3.124s; RTF_p95=2.0199-1.9308 | Talker AR tail dominates; handoff/decode p95 remains small. | Do not describe low-concurrency talker tail as high-concurrency admission saturation. | `sglang_videoamme_stress`, `build_tail_confidence_appendix` |
| Video-AMME c=4 | sglang | balanced serving / strict-headline reference | QPS=2.036; lat_p95=3.633s; RTF_p95=2.4982; queue=191.1ms / 9.9% | Talker AR remains primary; queue is still bounded. | Do not mix stress c4 and strict warmed c4 as one metric slice. | `sglang_videoamme_stress`, `vllm_c4_original`, `build_tail_confidence_appendix` |
| Video-AMME c=8 | sglang | throughput edge / current high-concurrency sweet spot | QPS=2.540; lat_p95=5.853s; RTF_p95=4.3924; queue=937.4ms / 30.6% | Admission queue becomes visible, but QPS is still the current peak and handoff is not the bottleneck. | Do not widen preprocessing concurrency; preproc=2/4 are measured anti-recipes. | `sglang_videoamme_stress`, `build_stage_latency_budget` |
| Video-AMME c=16 | sglang | saturation evidence / pressure boundary | QPS=2.407; lat_p95=7.846s; RTF_p95=10.4087; queue=4090.5ms / 67.4% | Throughput is below c8 while queue share and RTF tail rise sharply. | Do not present c16 as better high-concurrency serving. | `sglang_videoamme_stress`, `build_stage_boundary_bottleneck_ledger` |
| Synthetic short text c=8 | sglang | short-text speech high-concurrency guard | QPS=2.983; lat_p95=2.828s; RTF_p95=0.7440; audio=4.3s; hop_p95=21.2ms | Short text remains faster than real time; code2wav decode share is small. | Do not extrapolate short-text throughput to long text. | `sglang_synthetic_text_to_speech`, `build_tail_confidence_appendix` |
| Synthetic long text c=8 | sglang | long-text/long-speech realtime guard | QPS=0.303; lat_p95=26.318s; RTF_p95=0.5001; audio=52.3s; hop_p95=24.0ms | Long c8 RTF p95 remains below 1; pressure maps to Talker cadence. | Do not call vocoder decode the primary bottleneck. | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget` |
| vLLM c=8 prebuild w4 | vllm | optimized offline diagnostic | runner_QPS=0.2127; engine_QPS=0.5360; admission_p95=4891.5ms | Prebuild removes most prompt-feed admission and exposes engine/talker tail. | Do not promote this to online serving parity without online ingress plus WER/ASR; 不要升级为 online serving parity. | `vllm_c8_prebuild_w4`, `diagnose_vllm_admission` |

## 3. Pressure transition 证据

| ID | Pressure transition evidence |
| --- | --- |
| capacity-videoamme-c8 | QPS +24.8%; latency_p95 +61.1%; RTF_p95 +75.8%; queue_delta +746.3ms; queue_share_after 76.4%; hop_p95_delta +2.6ms; decode_p95_delta +4.1ms |
| capacity-videoamme-c16 | QPS -5.2%; latency_p95 +34.1%; RTF_p95 +137.0%; queue_delta +3153.1ms; queue_share_after 93.1%; hop_p95_delta -0.7ms; decode_p95_delta -2.0ms |
| capacity-synthetic-short-c8 | QPS +34.5%; latency_p95 +37.5%; RTF_p95_after 0.7; talker_p95_delta +472.8ms; hop_p95_delta +1.1ms; decode_avg_delta -1.2ms |
| capacity-synthetic-long-c8 | QPS +33.5%; latency_p95 +46.0%; RTF_p95_after 0.5; talker_p95_delta +8251.4ms; hop_p95_delta +3.6ms; decode_avg_delta +0.5ms |
| capacity-vllm-c8-prebuild-w4 | wall_qps +31.1%; engine_qps +230.4%; admission_p95 -88.9%; runner_overhead -16.2pp; talker_drain_p95_delta +107.4ms; after_diagnosis=engine_or_workload_limited |

## 4. Check 明细

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | source summaries are ready | tail={'ready': True, 'rows_total': 18, 'bootstrap_rows_total': 9, 'bootstrap_draws': 5000, 'checks_total': 13, 'checks_passed': 13, 'required_failures': 0, 'strict_c4_sglang_latency_p95_s': 3.3280250000000002, 'strict_c4_vllm_latency_p95_s': 3.5251249999999996, 'strict_c4_sglang_rtf_p95': 2.4022750000000004, 'strict_c4_vllm_rtf_p95': 3.0716750000000004, 'sglang_c8_qps': 2.54, 'sglang_c16_qps': 2.407, 'long_c8_rtf_p95': 0.50008, 'vllm_w4_latency_p95_s': 7.729534999999999, 'strict_c4_latency_mean_advantage_ci95_low_s': 0.024740869565217122, 'strict_c4_latency_mean_advantage_ci95_high_s': 0.6964290760869567, 'strict_c4_rtf_mean_advantage_ci95_low': -0.1587658152173914, 'long_c8_rtf_p95_ci95_high': 0.5005, 'share_scope': 'Per-sample tail and distribution appendix for strict c4, SGLang stress, synthetic short/long speech, vLLM diagnostics, and deterministic bootstrap uncertainty checks.'}; stage_budget={'ready': True, 'checks_total': 12, 'checks_passed': 12, 'required_failures': 0, 'sglang_budget_rows': 5, 'synthetic_budget_rows': 6, 'vllm_budget_rows': 4, 'c8_queue_pct_of_latency': 30.594223237597905, 'c16_queue_pct_of_latency': 67.43293768545993, 'long_c8_rtf_mean': 0.4932, 'vllm_c8_diagnosis': 'prompt_feed_limited', 'vllm_w4_scope': 'offline_diagnostic_only'}; stage_ledger={'ready': True, 'checks_total': 12, 'checks_passed': 12, 'required_failures': 0, 'ledger_rows': 37, 'pressure_transition_rows': 11, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'verdict_counts': {'not_current_bottleneck': 28, 'admission_queue_bottleneck': 2, 'negative_optimization': 1, 'offline_runner_prompt_feed_bottleneck': 2, 'observed_bottleneck': 1, 'offline_diagnostic_not_parity': 2, 'tail_watch': 1}, 'recommended_sglang_window': 'c4-c8', 'saturation_boundary': 'c16', 'vllm_c8_scope': 'offline_diagnostic_until_online_ingress_artifacts'} |
| PASS | yes | required tail cases are present | missing= |
| PASS | yes | stage budgets cover serving regimes | sglang=5, synthetic=6, vllm=4 |
| PASS | yes | SGLang c8 is the measured throughput edge | c4=2.036, c8=2.54, c16=2.407 |
| PASS | yes | SGLang c16 is a saturation boundary | c8_qps=2.54, c16_qps=2.407, c8_queue=30.594223237597905, c16_queue=67.43293768545993 |
| PASS | yes | synthetic c8 speech guards remain faster than real time | short_rtf_p95=0.743975, long_rtf_p95=0.50008 |
| PASS | yes | vLLM c8 prebuild w4 remains offline diagnostic | scope=offline_diagnostic_only, runner=0.21268902737307782, engine=0.5359849581181354 |
| PASS | yes | pressure transitions are attached | pressure_transition_rows=11 |
| PASS | yes | all capacity rows have decisions and guardrails | rows=7 |
| PASS | yes | capacity rows are reproducible | missing_command_rows= |

## 5. 机器证据

- `results/qwen35_report_audit_20260619/serving_capacity_matrix.json`
- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`
- `results/qwen35_report_audit_20260619/stage_latency_budget.json`
- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- Rebuild command ID: `build_serving_capacity_matrix`
