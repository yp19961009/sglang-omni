# Qwen3.5-Omni 复跑验收阈值合同

生成时间 UTC：`2026-06-21T02:00:23.169414+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份 contract 把当前 checkpoint 的核心数字转换成合作方复跑验收阈值。
它不替代完整报告；用途是判断复跑结果是“确认当前形态”、
“只能作为外部附录”，还是“具备替换 headline 数字的资格”。

## 1. 当前状态

| Gate | Value |
| --- | ---: |
| Ready | True |
| Checks | 17/17 |
| Required failures | 0 |
| Rules | 18 |
| SGLang stress rules | 5 |
| Synthetic rules | 6 |
| vLLM rules | 4 |
| Return evidence files | 34 |
| Command return evidence rows | 27 |
| Replacement scope | `same hardware/image/model/data plus all gates green` |
| Default decision | `confirm_shape_unless_environment_and_all_gates_match` |

## 2. Gate 明细

| Status | Required | Gate | Evidence |
| --- | --- | --- | --- |
| PASS | yes | final readiness summary is available | final_readiness=ready=True, checks=49/49, required_failures=0 |
| PASS | yes | runtime image contract is ready | runtime_image_contract={'ready': True, 'checks_total': 12, 'checks_passed': 12, 'required_failures': 0, 'sglang_image': 'frankleeeee/sglang-omni:dev', 'sglang_image_id': 'sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84', 'vllm_image': 'tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589', 'vllm_image_id': 'sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba', 'gpu_contract': '8x NVIDIA H20 / CUDA 12.8', 'sglang_scope': 'c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary', 'vllm_strict_scope': 'optimized warmed c4 apples-to-apples headline only', 'vllm_c8_scope': 'prebuild w4 is optimized offline diagnostic, not online parity', 'environment_pending_in_full_audit': False} |
| PASS | yes | acceptance matrix remains green | acceptance={'ready': True, 'rows_total': 17, 'rows_passed': 17, 'rows_failed': 0, 'evidence_status_counts': {'PASS': 17}, 'serving_status_counts': {'anti_recipe_failure': 1, 'anti_recipe_regression': 1, 'cross_stage_guardrail': 1, 'diagnostic_prompt_feed_limited': 1, 'not_recommended_saturation': 1, 'optimized_offline_diagnostic': 1, 'recommended_peak_throughput': 1, 'recommended_serving_window': 3, 'recommended_strict_baseline': 1, 'speech_generation_regression_guard': 6}} |
| PASS | yes | stress rows cover c1/c2/c4/c8/c16 | stress_concurrency=[1, 2, 4, 8, 16] |
| PASS | yes | synthetic rows cover short/long c1/c4/c8 | synthetic_rows=[('short', 1), ('short', 4), ('short', 8), ('long', 1), ('long', 4), ('long', 8)] |
| PASS | yes | c8 is current SGLang throughput peak | c8_qps=2.54, all_qps=[0.76, 1.315, 2.036, 2.54, 2.407] |
| PASS | yes | c16 remains saturation boundary | c8={'concurrency': 8, 'n': 50, 'accuracy': 0.7, 'latency_mean_s': 3.064, 'latency_p95_s': 5.853, 'rtf_mean': 2.2141, 'rtf_p95': 4.3925, 'throughput_qps': 2.54, 'audio_throughput_s_per_s': 5.372, 'wer_corpus': 0.03225806451612903, 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json', 'wer_json': '/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/whisper_large_v3_local_wer.json'}, c16={'concurrency': 16, 'n': 50, 'accuracy': 0.7, 'latency_mean_s': 6.066, 'latency_p95_s': 7.846, 'rtf_mean': 4.8489, 'rtf_p95': 10.4087, 'throughput_qps': 2.407, 'audio_throughput_s_per_s': 4.759, 'wer_corpus': 0.028846153846153848, 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json', 'wer_json': '/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/whisper_large_v3_local_wer.json'} |
| PASS | yes | long c8 remains faster than real time | long_c8={'scenario': 'long', 'concurrency': 8, 'n': 8, 'target_chars': 944.0, 'target_words': 139.0, 'audio_duration_mean_s': 52.33, 'latency_mean_s': 25.799, 'latency_p95_s': 26.318, 'rtf_mean': 0.4932, 'rtf_p95': 0.5001, 'throughput_qps': 0.303, 'audio_throughput_s_per_s': 15.87, 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_synthetic_speech_20260619/long_c8/synthetic_speech_results.json'} |
| PASS | yes | vLLM original c8 remains prompt-feed diagnostic | vllm_c8={'label': 'vLLM-c8', 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json', 'run_log': '/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log', 'skip_first_requests': 8, 'concurrency': 8, 'requests': 50, 'wall_time_s': 308.2215, 'runner_wall_time_s': 308.2215, 'engine_wall_time_s': 308.2215, 'prompt_build_wall_s': 0.0, 'runner_overhead_s': 252.0445, 'runner_overhead_pct_wall': 81.77382174832061, 'engine_overhead_s': 252.0445, 'engine_overhead_pct_wall': 81.77382174832061, 'wall_qps': 0.16222100015735438, 'runner_qps': 0.16222100015735438, 'engine_qps': 0.16222100015735438, 'batch_max_qps': 0.8900439681720277, 'batch_admission_span_avg_ms': 33313.99997075399, 'batch_admission_span_p95_ms': 43972.74994850159, 'batch_last_engine_lag_avg_ms': 38005.49999872843, 'batch_last_engine_lag_p95_ms': 46880.99992275238, 'encoder_p95_ms': 43.763220214843756, 'talker_to_code2wav_drain_p95_ms': 16.000032424926758, 'prompt_feed_limited': True, 'engine_boundaries_clean': True, 'diagnosis': 'prompt_feed_limited', 'recommendation': 'prebuild or parallelize multimodal prompt construction before timed engine admission; then rerun wall-QPS comparison'} |
| PASS | yes | vLLM prebuild w4 improves offline runner QPS | vllm_c8={'label': 'vLLM-c8', 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json', 'run_log': '/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log', 'skip_first_requests': 8, 'concurrency': 8, 'requests': 50, 'wall_time_s': 308.2215, 'runner_wall_time_s': 308.2215, 'engine_wall_time_s': 308.2215, 'prompt_build_wall_s': 0.0, 'runner_overhead_s': 252.0445, 'runner_overhead_pct_wall': 81.77382174832061, 'engine_overhead_s': 252.0445, 'engine_overhead_pct_wall': 81.77382174832061, 'wall_qps': 0.16222100015735438, 'runner_qps': 0.16222100015735438, 'engine_qps': 0.16222100015735438, 'batch_max_qps': 0.8900439681720277, 'batch_admission_span_avg_ms': 33313.99997075399, 'batch_admission_span_p95_ms': 43972.74994850159, 'batch_last_engine_lag_avg_ms': 38005.49999872843, 'batch_last_engine_lag_p95_ms': 46880.99992275238, 'encoder_p95_ms': 43.763220214843756, 'talker_to_code2wav_drain_p95_ms': 16.000032424926758, 'prompt_feed_limited': True, 'engine_boundaries_clean': True, 'diagnosis': 'prompt_feed_limited', 'recommendation': 'prebuild or parallelize multimodal prompt construction before timed engine admission; then rerun wall-QPS comparison'}, vllm_w4={'label': 'vLLM-c8-prebuild-w4', 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json', 'run_log': '/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log', 'skip_first_requests': 8, 'concurrency': 8, 'requests': 50, 'wall_time_s': 235.085, 'runner_wall_time_s': 235.085, 'engine_wall_time_s': 93.2862, 'prompt_build_wall_s': 129.2195, 'runner_overhead_s': 154.2568, 'runner_overhead_pct_wall': 65.61745751536678, 'engine_overhead_s': 12.457999999999998, 'engine_overhead_pct_wall': 13.35460121647146, 'wall_qps': 0.21268902737307782, 'runner_qps': 0.21268902737307782, 'engine_qps': 0.5359849581181354, 'batch_max_qps': 0.6185959850646185, 'batch_admission_span_avg_ms': 4089.0000263849893, 'batch_admission_span_p95_ms': 4891.499996185303, 'batch_last_engine_lag_avg_ms': 20597.166736920673, 'batch_last_engine_lag_p95_ms': 24887.250006198883, 'encoder_p95_ms': 46.17030029296876, 'talker_to_code2wav_drain_p95_ms': 123.3500719070435, 'prompt_feed_limited': False, 'engine_boundaries_clean': False, 'diagnosis': 'engine_or_workload_limited', 'recommendation': 'runner admission is not the dominant signal; inspect engine stage latency and generated-audio length distribution'} |
| PASS | yes | stage interaction guardrails are true | stage_interactions={'total_interactions': 37, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'sglang_talker_to_code2wav_healthy': True, 'sglang_code2wav_decode_not_bottleneck': True, 'vllm_original_c8_prompt_feed_limited': True, 'preprocessing_parallelism_regresses': True} |
| PASS | yes | stage boundary ledger is ready | stage_boundary_ledger={'ready': True, 'checks_total': 12, 'checks_passed': 12, 'required_failures': 0, 'ledger_rows': 37, 'pressure_transition_rows': 11, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'verdict_counts': {'not_current_bottleneck': 28, 'admission_queue_bottleneck': 2, 'negative_optimization': 1, 'offline_runner_prompt_feed_bottleneck': 2, 'observed_bottleneck': 1, 'offline_diagnostic_not_parity': 2, 'tail_watch': 1}, 'recommended_sglang_window': 'c4-c8', 'saturation_boundary': 'c16', 'vllm_c8_scope': 'offline_diagnostic_until_online_ingress_artifacts'} |
| PASS | yes | strict c4 headline claims are machine-verified | claims_verification strict c4 latency/RTF and quality checks |
| PASS | yes | rerun rule inventory is complete | rules=18 |
| PASS | yes | collaborator return evidence contract is complete | return_files=34, missing_current=[], missing_sheet=[], runtime_comparison={'ready': True, 'checks_total': 9, 'checks_passed': 9, 'required_failures': 0, 'strict_scope': 'Video-AMME ci-50, warmed skip-first-4, c=4', 'strict_c4_lower_latency_mean_pct': 16.72115280766322, 'strict_c4_lower_rtf_p95_pct': 21.792670122978492, 'vllm_c8_contract': 'offline_diagnostic_not_online_parity', 'allowed_cross_runtime_headline': 'warmed c=4 only', 'baseline_strength': 'optimized image plus compile/graph/cache/shared-memory/encoder/prebuild evidence'} |
| PASS | yes | command return evidence matrix is complete | command_rows=27, missing_commands=[], missing_required_files=[], without_evidence=[] |
| PASS | yes | silent replacement and protocol drift guards are documented | rerun_delta_triage={'ready': True, 'rows_total': 19, 'required_failures': 0, 'checks_total': 8, 'checks_passed': 8, 'stage_routes_total': 19, 'replacement_scopes': ['不得替换', '可进入评审', '附录趋势', '需替换评审'], 'usage': 'Use after collaborator reruns to map metric deltas to stage evidence before changing headline numbers.'}, silent_rows=2, protocol_rows=1, regeneration_rows=1 |

## 3. 合作方回传证据合同

如果复跑方希望进入 headline 数字替换评审，除 raw SGLang/vLLM artifact 外，
必须把下列机器证据一起回传；缺任一项时只能作为附录趋势或需先补证据。

| Required evidence | Current file | Listed in sheet | Purpose |
| --- | --- | --- | --- |
| `results/qwen35_report_audit_20260619/audit_run_summary.json` | True | True | full audit pass/fail and step order |
| `results/qwen35_report_audit_20260619/environment_snapshot.json` | True | True | hardware, image digest, model/data path |
| `results/qwen35_report_audit_20260619/manifest.json` | True | True | raw and packaged evidence inventory |
| `results/qwen35_report_audit_20260619/coverage_matrix.json` | True | True | original requirement coverage |
| `results/qwen35_report_audit_20260619/claims_verification.json` | True | True | numeric claim gate |
| `results/qwen35_report_audit_20260619/headline_scorecard.json` | True | True | strict headline scorecard |
| `results/qwen35_report_audit_20260619/acceptance_matrix.json` | True | True | pressure/regime acceptance rows |
| `results/qwen35_report_audit_20260619/confidence_ledger.json` | True | True | safe wording and unsupported-claim guard |
| `results/qwen35_report_audit_20260619/runtime_comparison_contract.json` | True | True | warmed c4-only cross-runtime headline and c8 diagnostic boundary |
| `results/qwen35_report_audit_20260619/runtime_image_contract.json` | True | True | SGLang/vLLM image and optimization-scope lock |
| `results/qwen35_report_audit_20260619/rerun_acceptance_contract.json` | True | True | replacement thresholds |
| `results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json` | True | True | receiver quickcheck contract and public-doc route evidence |
| `results/qwen35_report_audit_20260619/repro_command_manifest.json` | True | True | exact rerun commands and expected evidence |
| `results/qwen35_report_audit_20260619/final_readiness_audit.json` | True | True | send/no-send gate after rerun |
| `results/qwen35_report_audit_20260619/share_bundle_manifest.json` | True | True | share bundle evidence inventory |
| `results/qwen35_report_audit_20260619/share_bundle_package_manifest.json` | True | True | tarball packaging and checksum creation |
| `results/qwen35_report_audit_20260619/metric_provenance_index.json` | True | True | metric-to-raw-artifact provenance |
| `results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json` | True | True | original requirement and optimization verdict crosswalk |
| `results/qwen35_report_audit_20260619/sglang_optimization_lock.json` | True | True | SGLang recipe lock and anti-recipes |
| `results/qwen35_report_audit_20260619/vllm_optimization_lock.json` | True | True | optimized vLLM baseline lock |
| `results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json` | True | True | online parity upgrade gate |
| `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` | True | True | vLLM offline admission and prompt-feed diagnosis |
| `results/qwen35_report_audit_20260619/vllm_log_stage_summary.json` | True | True | vLLM log-derived stage timing summary |
| `results/qwen35_report_audit_20260619/stage_interaction_summary.json` | True | True | stage connection flags |
| `results/qwen35_report_audit_20260619/stage_latency_budget.json` | True | True | stage latency proportions |
| `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json` | True | True | boundary bottleneck verdicts |
| `results/qwen35_report_audit_20260619/stage_causal_graph.json` | True | True | stage causal edges and raw drilldown |
| `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json` | True | True | stage row to jq/artifact/command map |
| `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json` | True | True | route-level bottleneck decisions |
| `results/qwen35_report_audit_20260619/rerun_delta_triage.json` | True | True | symptom-to-stage rerun delta triage |
| `results/qwen35_report_audit_20260619/share_package_validation.json` | True | True | tarball validation evidence |
| `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | True | True | receiver smoke evidence |
| `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | True | True | extracted-only validation evidence |
| `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | True | True | repo-independent standalone validation evidence |

## 4. 命令到回传证据矩阵

这张表把关键复跑命令和必须随同回传的 gate 文件绑定起来；
raw/runtime artifact 用来复查原始输出，required return files 用来决定能否替换报告数字。

| Command ID | Phase | Raw/runtime artifacts | Required return files | Ready |
| --- | --- | --- | --- | --- |
| `run_full_audit` | audit_first |  | results/qwen35_report_audit_20260619/audit_run_summary.json, results/qwen35_report_audit_20260619/manifest.json, results/qwen35_report_audit_20260619/coverage_matrix.json, ... (+3) | True |
| `sglang_videoamme_stress` | sglang_stress | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c*_profile_skipwer.json | results/qwen35_report_audit_20260619/headline_scorecard.json, results/qwen35_report_audit_20260619/acceptance_matrix.json, results/qwen35_report_audit_20260619/stage_interaction_summary.json, ... (+5) | True |
| `sglang_synthetic_text_to_speech` | sglang_stress | results/qwen35_synthetic_speech_20260619/*/synthetic_speech_results.json, results/qwen35_synthetic_speech_20260619/request_profile_*_profile.json | results/qwen35_report_audit_20260619/acceptance_matrix.json, results/qwen35_report_audit_20260619/stage_latency_budget.json, results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json, ... (+1) | True |
| `sglang_recompute_wer` | quality_validation | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_*/whisper_large_v3_local_wer.json | results/qwen35_report_audit_20260619/claims_verification.json, results/qwen35_report_audit_20260619/headline_scorecard.json, results/qwen35_report_audit_20260619/acceptance_matrix.json | True |
| `vllm_c4_original` | vllm_baseline | results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_*/run.log, results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_*/benchmark_audio_50_c4_offline_compile/videoamme_results.json | results/qwen35_report_audit_20260619/runtime_comparison_contract.json, results/qwen35_report_audit_20260619/headline_scorecard.json, results/qwen35_report_audit_20260619/claims_verification.json, ... (+4) | True |
| `vllm_c8_original` | vllm_baseline | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_*/run.log, results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_*/benchmark_audio_50_c8_offline_compile/videoamme_results.json | results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json, results/qwen35_report_audit_20260619/vllm_log_stage_summary.json, results/qwen35_report_audit_20260619/vllm_optimization_lock.json, ... (+2) | True |
| `vllm_c8_prebuild_w4` | vllm_baseline | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_*/run.log, results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_*/benchmark_audio_50_c8_offline_compile/videoamme_results.json | results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json, results/qwen35_report_audit_20260619/vllm_log_stage_summary.json, results/qwen35_report_audit_20260619/vllm_optimization_lock.json, ... (+1) | True |
| `build_headline_scorecard` | audit_regeneration |  | results/qwen35_report_audit_20260619/headline_scorecard.json | True |
| `build_acceptance_matrix` | audit_regeneration |  | results/qwen35_report_audit_20260619/acceptance_matrix.json | True |
| `build_stage_latency_budget` | audit_regeneration | benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md | results/qwen35_report_audit_20260619/stage_latency_budget.json | True |
| `build_stage_boundary_bottleneck_ledger` | audit_regeneration | benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md | results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | True |
| `build_stage_causal_graph` | audit_regeneration | benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md | results/qwen35_report_audit_20260619/stage_causal_graph.json | True |
| `build_stage_reproduction_drilldown` | audit_regeneration | benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md | results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json | True |
| `build_stage_route_decision_matrix` | audit_regeneration | benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md, benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md, ... (+1) | results/qwen35_report_audit_20260619/stage_route_decision_matrix.json | True |
| `build_runtime_comparison_contract` | audit_regeneration | benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md | results/qwen35_report_audit_20260619/runtime_comparison_contract.json | True |
| `build_runtime_image_contract` | audit_regeneration | benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md | results/qwen35_report_audit_20260619/runtime_image_contract.json | True |
| `build_rerun_acceptance_contract` | audit_regeneration | benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md | results/qwen35_report_audit_20260619/rerun_acceptance_contract.json | True |
| `build_rerun_delta_triage` | audit_regeneration | benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md | results/qwen35_report_audit_20260619/rerun_delta_triage.json | True |
| `build_metric_provenance_index` | audit_regeneration |  | results/qwen35_report_audit_20260619/metric_provenance_index.json | True |
| `build_objective_requirement_crosswalk` | audit_regeneration |  | results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json | True |
| `build_final_readiness_audit` | audit_regeneration |  | results/qwen35_report_audit_20260619/final_readiness_audit.json | True |
| `build_share_bundle_manifest` | audit_regeneration |  | results/qwen35_report_audit_20260619/share_bundle_manifest.json | True |
| `build_share_bundle_package` | audit_regeneration | results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz, results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256 | results/qwen35_report_audit_20260619/share_bundle_package_manifest.json | True |
| `validate_share_bundle_package` | audit_regeneration |  | results/qwen35_report_audit_20260619/share_package_validation.json | True |
| `validate_share_bundle_receiver_smoke` | audit_regeneration |  | results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json | True |
| `validate_extracted_share_bundle` | audit_regeneration |  | results/qwen35_report_audit_20260619/share_package_validation_extracted.json | True |
| `validate_external_standalone_share_bundle` | audit_regeneration |  | results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json | True |

## 5. 复跑阈值表

| Rule | Category | Scope | Baseline | Acceptance / Replacement |
| --- | --- | --- | --- | --- |
| `sglang_videoamme_c1` | sglang_stress | Video-AMME ci-50 c=1 / serving_window | throughput_qps=0.76; latency_p95_s=2.406; rtf_mean=1.049; wer_corpus=0.038461538461538464 | completed_min=50; accuracy_min=0.68; wer_corpus_max=0.0585; throughput_qps_min_for_same_hardware=0.646; Can replace checkpoint numbers only when environment/image/model/data match and all full-audit gates pass. |
| `sglang_videoamme_c2` | sglang_stress | Video-AMME ci-50 c=2 / serving_window | throughput_qps=1.315; latency_p95_s=3.124; rtf_mean=1.0816; wer_corpus=0.038461538461538464 | completed_min=50; accuracy_min=0.68; wer_corpus_max=0.0585; throughput_qps_min_for_same_hardware=1.1178; Can replace checkpoint numbers only when environment/image/model/data match and all full-audit gates pass. |
| `sglang_videoamme_c4` | sglang_stress | Video-AMME ci-50 c=4 / serving_window | throughput_qps=2.036; latency_p95_s=3.633; rtf_mean=1.4015; wer_corpus=0.038461538461538464 | completed_min=50; accuracy_min=0.68; wer_corpus_max=0.0585; throughput_qps_min_for_same_hardware=1.7306; Can replace checkpoint numbers only when environment/image/model/data match and all full-audit gates pass. |
| `sglang_videoamme_c8` | sglang_stress | Video-AMME ci-50 c=8 / peak_throughput | throughput_qps=2.54; latency_p95_s=5.853; rtf_mean=2.2141; wer_corpus=0.03225806451612903 | completed_min=50; accuracy_min=0.68; wer_corpus_max=0.0523; throughput_qps_min_for_same_hardware=2.159; Can replace checkpoint numbers only when environment/image/model/data match and all full-audit gates pass. |
| `sglang_videoamme_c16` | sglang_stress | Video-AMME ci-50 c=16 / saturation_boundary | throughput_qps=2.407; latency_p95_s=7.846; rtf_mean=4.8489; wer_corpus=0.028846153846153848 | completed_min=50; accuracy_min=0.68; wer_corpus_max=0.0488; throughput_qps_min_for_same_hardware=2.0459; Can replace checkpoint numbers only when environment/image/model/data match and all full-audit gates pass. |
| `synthetic_short_c1` | synthetic_speech | short text input + speech output c=1 | throughput_qps=1.154; latency_p95_s=0.924; rtf_mean=0.2052 | target_chars_exact=74.0; target_words_exact=12.0; completed_min=16; rtf_mean_max=1.0; Use as shape confirmation unless the full audit and environment match. |
| `synthetic_short_c4` | synthetic_speech | short text input + speech output c=4 | throughput_qps=2.218; latency_p95_s=2.056; rtf_mean=0.4105 | target_chars_exact=74.0; target_words_exact=12.0; completed_min=16; rtf_mean_max=1.0; Use as shape confirmation unless the full audit and environment match. |
| `synthetic_short_c8` | synthetic_speech | short text input + speech output c=8 | throughput_qps=2.983; latency_p95_s=2.828; rtf_mean=0.6257 | target_chars_exact=74.0; target_words_exact=12.0; completed_min=16; rtf_mean_max=1.0; Use as shape confirmation unless the full audit and environment match. |
| `synthetic_long_c1` | synthetic_speech | long text input + speech output c=1 | throughput_qps=0.109; latency_p95_s=9.465; rtf_mean=0.1766 | target_chars_exact=944.0; target_words_exact=139.0; completed_min=8; rtf_mean_max=1.0; Use as shape confirmation unless the full audit and environment match. |
| `synthetic_long_c4` | synthetic_speech | long text input + speech output c=4 | throughput_qps=0.227; latency_p95_s=18.025; rtf_mean=0.3338 | target_chars_exact=944.0; target_words_exact=139.0; completed_min=8; rtf_mean_max=1.0; Use as shape confirmation unless the full audit and environment match. |
| `synthetic_long_c8` | synthetic_speech | long text input + speech output c=8 | throughput_qps=0.303; latency_p95_s=26.318; rtf_mean=0.4932 | target_chars_exact=944.0; target_words_exact=139.0; completed_min=8; rtf_mean_max=1.0; Long c8 must stay faster than real time before the long-text speech guardrail can remain a high-confidence claim. |
| `vllm-c4` | vllm_strict_baseline | vLLM-c4 | diagnosis=prompt_feed_limited | completed_min=50; diagnosis_expected=prompt_feed_limited; admission_p95_ms_reference=19135.75005531311; runner_qps_min_for_same_hardware=0.1229; c4 can participate in strict warmed headline comparison when claims pass. |
| `vllm-c8` | vllm_c8_diagnostic | vLLM-c8 | diagnosis=prompt_feed_limited | completed_min=50; diagnosis_expected=prompt_feed_limited; admission_p95_ms_reference=43972.74994850159; runner_qps_min_for_same_hardware=0.1298; Keep as offline diagnostic until online ingress and same-scope WER/ASR exist. |
| `vllm-c8-prebuild-w1` | vllm_c8_diagnostic | vLLM-c8-prebuild-w1 | diagnosis=engine_or_workload_limited | completed_min=50; diagnosis_expected=engine_or_workload_limited; admission_p95_ms_reference=5425.000071525574; runner_qps_min_for_same_hardware=0.1136; Keep as offline diagnostic until online ingress and same-scope WER/ASR exist. |
| `vllm-c8-prebuild-w4` | vllm_c8_diagnostic | vLLM-c8-prebuild-w4 | diagnosis=engine_or_workload_limited | completed_min=50; diagnosis_expected=engine_or_workload_limited; admission_p95_ms_reference=4891.499996185303; runner_qps_min_for_same_hardware=0.1702; Keep as offline diagnostic until online ingress and same-scope WER/ASR exist. |
| `sglang_c8_peak_shape` | cross_condition_shape | SGLang c8 remains the best current high-concurrency point | c8_qps=2.54; c16_qps=2.407; c8_latency_mean_s=3.064 | c8_qps_should_be_highest_or_within_noise=True; c16_latency_mean_should_exceed_c8=True; c16_is_not_default_serving_point=True; If c16 beats c8 materially, rerun stage ledger and update serving-window claims before replacing numbers. |
| `preprocessing_parallelism_antirecipe` | negative_optimization_guardrail | PREPROCESSING_MAX_CONCURRENCY=2 remains a measured anti-recipe | preproc1_qps=2.54; preproc2_qps=1.642; qps_delta_pct=-35.35433070866142 | preproc2_should_not_be_promoted_unless_qps_exceeds_baseline=True; preproc2_current_qps_max_for_antirecipe=2.286; Do not change the recommended recipe to preproc=2 unless a new placement/admission design beats baseline with quality intact. |
| `stage_connection_flags` | stage_connection_guardrail | Stage connections and bottleneck attribution | total_interactions=37; status_counts={'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1} | sglang_talker_to_code2wav_healthy=True; sglang_code2wav_decode_not_bottleneck=True; vllm_original_c8_prompt_feed_limited=True; preprocessing_parallelism_regresses=True; Any changed flag requires rebuilding stage latency budget, boundary ledger, confidence ledger, and final readiness. |

## 6. 是否可以替换报告数字

| Condition | Decision |
| --- | --- |
| same 8x H20, same image digests, same model/data, full audit green | may replace report numbers after reviewer signoff |
| full audit green but hardware or image differs | external validation appendix only; do not replace headline |
| claims, acceptance, confidence, stage ledger, or runtime image contract fails | do not replace; diagnose the failed gate first |
| raw benchmark changed but regeneration/full audit or share seal was not rerun | do not replace; this is a silent-replacement risk |
| vLLM c8 prebuild improves but online ingress/WER is absent | update offline diagnostic only; do not claim online parity |

## 7. 使用方式

1. 合作方先按 handoff runbook 和 reproduction checklist 复跑。
2. 复跑后重跑 full audit、preflight、manifest、final readiness 和本 contract，并按第 3 节回传机器证据。
3. 若硬件、image digest、模型、数据路径任一不同，结果只能作为外部复核附录。
4. 若要替换 headline，必须同时满足本 contract、claims、acceptance、confidence、stage ledger 和 runtime image contract。
5. vLLM c=8 prebuild w4 即使变好，也只能更新 offline diagnostic，不能自动升级为 online parity。
6. raw benchmark、公开 Markdown 或图表任一被替换时，必须同步重跑 regeneration/full audit、share bundle、tarball seal 和 receiver quickcheck，否则按 silent-replacement 风险处理。

## 8. 机器证据

- tables_summary：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tables_summary.json`
- acceptance_matrix：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json`
- claims_verification：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claims_verification.json`
- stage_interaction_summary：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- stage_boundary_bottleneck_ledger：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- runtime_comparison_contract：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/runtime_comparison_contract.json`
- runtime_image_contract：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/runtime_image_contract.json`
- repro_command_manifest：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/repro_command_manifest.json`
- rerun_delta_triage：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/rerun_delta_triage.json`
- final_readiness：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/final_readiness_audit.json`
- collaborator_rerun_validation_sheet：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
