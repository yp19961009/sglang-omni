# Qwen3.5-Omni 优化候选 Ledger

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。

## 1. 裁决摘要

- ready: `True`
- candidates: `8`
- current measured best: `sglang_current_best_measured_recipe`
- anti-recipes / vLLM diagnostics: `2` / `2`
- checks: `10/10`
- required_failures: `0`

当前结论是 measured-best，不是对未来 kernel、placement、admission policy 或 online vLLM ingress 的全局数学最优声明。替换当前 best recipe 前，必须同时重跑 strict c=4 SGLang/vLLM、SGLang c=1/2/4/8/16、quality/WER、stage causal graph、acceptance matrix、rerun acceptance contract 和 final readiness。

## 2. Machine Gate

| Gate | Status |
| --- | --- |
| `source_crosswalk_contract_available` | `PASS` |
| `candidate_rows_total` | `PASS` |
| `current_best_recipe_locked` | `PASS` |
| `anti_recipes_locked` | `PASS` |
| `vllm_baseline_and_diagnostic_locked` | `PASS` |
| `metric_rows_present` | `PASS` |
| `evidence_files_present` | `PASS` |
| `rerun_commands_present` | `PASS` |
| `not_global_optimum_boundary_documented` | `PASS` |
| `final_completion_gate_contract_available` | `PASS` |

## 3. 候选裁决总表

| Candidate | Runtime | Decision class | Decision | Ready | Key metrics | Evidence | Commands |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sglang_current_best_measured_recipe` | sglang | `current_best_measured` | `accept_current_recipe` | `True` | strict_c4_sglang_lower_latency_mean_pct=16.72115280766322 percent; strict_c4_sglang_lower_latency_p95_pct=5.591291088968471 percent; strict_c4_sglang_lower_rtf_mean_pct=7.775356855612155 percent; strict_c4_sglang_lower_rtf_p95_pct=21.792670122978492 percent; +2 metrics | results/qwen35_report_audit_20260619/headline_scorecard.json; results/qwen35_report_audit_20260619/acceptance_matrix.json; results/qwen35_report_audit_20260619/sglang_optimization_lock.json; +2 files | build_headline_scorecard; build_acceptance_matrix; build_sglang_optimization_lock; build_stage_latency_budget; +2 commands |
| `sglang_c16_saturation_boundary` | sglang | `not_recommended_default` | `keep_as_saturation_boundary` | `True` | sglang_videoamme_c8_throughput_qps=2.54 requests_per_second; sglang_videoamme_c16_throughput_qps=2.407 requests_per_second; sglang_videoamme_c16_vs_c8_qps_delta_pct=-5.2362204724409445 percent; stage_drilldown_stage-boundary-13=Use c4-c8 as serving window; treat c16 as saturation boundary and avoid claiming more concurrency as better. | results/qwen35_report_audit_20260619/acceptance_matrix.json; results/qwen35_report_audit_20260619/stage_latency_budget.json; results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | sglang_videoamme_stress; build_acceptance_matrix; build_stage_latency_budget; build_stage_boundary_bottleneck_ledger |
| `sglang_preproc2_widening` | sglang | `rejected_anti_recipe` | `reject_current_recipe_change` | `True` | acceptance_row_15_negative_optimization=baseline_QPS=2.540, preproc2_QPS=1.642, baseline_lat=3.064s, preproc2_lat=4.579s; stage_drilldown_stage-boundary-16=Keep preprocessing concurrency at 1 unless placement/admission is redesigned; preproc=2 is a negative recipe. | results/qwen35_report_audit_20260619/acceptance_matrix.json; results/qwen35_report_audit_20260619/sglang_optimization_lock.json; results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | build_acceptance_matrix; build_sglang_optimization_lock; build_stage_boundary_bottleneck_ledger |
| `sglang_preproc4_widening` | sglang | `rejected_anti_recipe` | `reject_current_recipe_change` | `True` | acceptance_row_16_negative_optimization=failed=7, accuracy=60.0% | results/qwen35_report_audit_20260619/acceptance_matrix.json; results/qwen35_report_audit_20260619/sglang_optimization_lock.json | build_acceptance_matrix; build_sglang_optimization_lock |
| `vllm_optimized_c4_baseline` | vllm | `accepted_strict_baseline` | `accept_as_strong_baseline` | `True` | strict_c4_vllm_latency_mean_s=2.0933391304347824 seconds; strict_c4_vllm_latency_p95_s=3.525125 seconds; strict_c4_vllm_rtf_mean=1.4676804347826087 ratio; strict_c4_vllm_rtf_p95=3.071675 ratio; +2 metrics | results/qwen35_report_audit_20260619/runtime_image_contract.json; results/qwen35_report_audit_20260619/vllm_optimization_lock.json; results/qwen35_report_audit_20260619/runtime_comparison_contract.json; +1 files | vllm_c4_original; build_runtime_image_contract; build_vllm_optimization_lock; build_runtime_comparison_contract; +1 commands |
| `vllm_original_c8_offline` | vllm | `diagnostic_only` | `keep_as_prompt_feed_diagnostic` | `True` | vllm_c8_original_runner_overhead_pct_wall=81.77382174832061 percent; vllm_c8_original_engine_qps=0.16222100015735438 requests_per_second; vllm_c8_original_admission_batch_admission_span_avg_ms=33313.99997075399 milliseconds; vllm_c8_original_admission_prompt_feed_limited=True value; +1 metrics | results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json; results/qwen35_report_audit_20260619/vllm_log_stage_summary.json; results/qwen35_report_audit_20260619/runtime_comparison_contract.json | vllm_c8_original; diagnose_vllm_admission; summarize_vllm_log_stages; build_runtime_comparison_contract |
| `vllm_c8_prebuild_w4_offline` | vllm | `optimized_offline_diagnostic` | `accept_as_strongest_current_offline_diagnostic` | `True` | vllm_c8_prebuild_w4_runner_qps=0.21268902737307782 requests_per_second; vllm_c8_prebuild_w4_engine_qps=0.5359849581181354 requests_per_second; vllm_c8_prebuild_w4_admission_batch_admission_span_avg_ms=4089.0000263849893 milliseconds; vllm_c8_w4_vs_w1_prompt_build_wall_delta_pct=-48.17600654520078 percent; +2 metrics | results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json; results/qwen35_report_audit_20260619/vllm_optimization_lock.json; results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json | vllm_c8_prebuild_w4; diagnose_vllm_admission; build_vllm_optimization_lock; build_vllm_online_parity_protocol |
| `code2wav_first_tuning` | sglang | `deprioritized_not_current_bottleneck` | `do_not_optimize_first` | `True` | stage_drilldown_stage-boundary-03=Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window.; stage_drilldown_stage-boundary-09=Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window.; stage_drilldown_stage-boundary-12=Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window.; acceptance_row_17_stage_connection_health=sglang_talker_to_code2wav_healthy=True, sglang_code2wav_decode_not_bottleneck=True, vllm_original_c8_prompt_feed_limited=True, preprocessing_parallelism_regresses=True | results/qwen35_report_audit_20260619/stage_causal_graph.json; results/qwen35_report_audit_20260619/stage_interaction_summary.json; results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | build_stage_causal_graph; build_stage_interactions; build_stage_boundary_bottleneck_ledger |

## 4. 解释口径

- `sglang_current_best_measured_recipe` 是当前可分享的 SGLang measured-best：strict warmed c=4 优于优化版 vLLM baseline，c=8 是当前 SGLang serving peak，c=16 保留为 saturation boundary。
- `sglang_preproc2_widening` 和 `sglang_preproc4_widening` 是反例，不应作为当前 recipe 的优化方向；前者吞吐/延迟退化，后者触发失败样本和质量风险。
- `vllm_optimized_c4_baseline` 是严格 headline 对比中的强基线；vLLM c=8 original/prebuild w4 只用于 offline diagnostic，不能写成 online serving parity。
- `code2wav_first_tuning` 当前不优先，因为 stage graph 显示 handoff 健康、code2wav decode 不是当前主计算瓶颈；应先看 admission/Talker cadence。

## 5. 复核和刷新入口

```bash
cd /home/gangouyu/sglang-omni
python3 -m benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json
python3 -m benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/optimization_candidate_ledger.json
```

机器证据：

- `results/qwen35_report_audit_20260619/optimization_candidate_ledger.json`
- `results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- `results/qwen35_report_audit_20260619/sglang_optimization_lock.json`
- `results/qwen35_report_audit_20260619/vllm_optimization_lock.json`
- `results/qwen35_report_audit_20260619/runtime_comparison_contract.json`
