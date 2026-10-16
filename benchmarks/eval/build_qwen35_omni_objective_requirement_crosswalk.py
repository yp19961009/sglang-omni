# SPDX-License-Identifier: Apache-2.0
"""Build original-objective-to-evidence crosswalk for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "objective_requirement_crosswalk.json"

OBJECTIVE_COMPLETION = AUDIT_DIR / "objective_completion_audit.json"
AUDIT_SUMMARY = AUDIT_DIR / "audit_run_summary.json"
FINAL_CHECKPOINT = AUDIT_DIR / "final_checkpoint_watchlist.json"
CLAIM_METRIC_CROSSWALK = AUDIT_DIR / "claim_metric_crosswalk.json"
METRIC_PROVENANCE = AUDIT_DIR / "metric_provenance_index.json"
REPRO_COMMANDS = AUDIT_DIR / "repro_command_manifest.json"

COMMON_MACHINE_EVIDENCE = [
    OBJECTIVE_COMPLETION.as_posix(),
    FINAL_CHECKPOINT.as_posix(),
    CLAIM_METRIC_CROSSWALK.as_posix(),
    METRIC_PROVENANCE.as_posix(),
    REPRO_COMMANDS.as_posix(),
]

OBJECTIVE_REQUIREMENTS = [
    {
        "requirement_id": "sglang_at_least_vllm_performance",
        "category": "performance",
        "original_user_intent": "Optimize SGLang-Omni Qwen3.5 so performance is at least comparable to vLLM.",
        "normalized_requirement": "Strict warmed c=4 SGLang-vs-vLLM headline must show latency/RTF parity or better without quality regression.",
        "source_objective_row_ids": ["sglang_vs_vllm_at_least_comparable"],
        "claim_ids": [
            "sglang_warmed_c4_beats_optimized_vllm",
            "wer_quality_not_traded_for_speed",
        ],
        "metric_row_ids": [
            "strict_c4_sglang_latency_mean_s",
            "strict_c4_sglang_latency_p95_s",
            "strict_c4_sglang_rtf_mean",
            "strict_c4_sglang_rtf_p95",
            "strict_c4_sglang_accuracy",
            "strict_c4_sglang_wer_corpus",
            "strict_c4_vllm_latency_mean_s",
            "strict_c4_vllm_latency_p95_s",
            "strict_c4_vllm_rtf_mean",
            "strict_c4_vllm_rtf_p95",
            "strict_c4_vllm_accuracy",
            "strict_c4_vllm_wer_corpus",
            "strict_c4_sglang_lower_latency_mean_pct",
            "strict_c4_sglang_lower_rtf_mean_pct",
            "acceptance_row_01_strict_runtime_comparison",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/headline_scorecard.json",
            "results/qwen35_report_audit_20260619/claims_verification.json",
            "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md",
        ],
        "rerun_command_ids": [
            "run_full_audit",
            "build_headline_scorecard",
            "verify_report_claims",
            "build_runtime_comparison_contract",
        ],
        "status": "PASS",
        "required_for_share": True,
        "reviewer_hook": "Answer with strict warmed c=4 SGLang/vLLM latency, RTF, accuracy, and WER rows side by side.",
        "caveat": "",
    },
    {
        "requirement_id": "vllm_baseline_strong",
        "category": "vllm_baseline",
        "original_user_intent": "Use the corresponding vLLM image and enable all relevant optimizations; the baseline must be strong.",
        "normalized_requirement": "vLLM baseline must be locked to the Qwen3.5-capable image, optimized flags, c=4 strict headline, and c=8 offline diagnostic boundary.",
        "source_objective_row_ids": [
            "vllm_reproduction_path",
            "vllm_online_parity_boundary",
        ],
        "claim_ids": [
            "vllm_baseline_is_optimized",
            "vllm_c8_prebuild_w4_is_offline_diagnostic",
        ],
        "metric_row_ids": [
            "strict_c4_vllm_latency_mean_s",
            "strict_c4_vllm_latency_p95_s",
            "strict_c4_vllm_rtf_mean",
            "strict_c4_vllm_rtf_p95",
            "vllm_c8_original_runner_overhead_pct_wall",
            "vllm_c8_original_engine_qps",
            "vllm_c8_original_admission_batch_admission_span_avg_ms",
            "vllm_c8_original_admission_prompt_feed_limited",
            "vllm_c8_prebuild_w4_runner_qps",
            "vllm_c8_prebuild_w4_engine_qps",
            "vllm_c8_w4_vs_w1_prompt_build_wall_delta_pct",
            "vllm_c8_w4_vs_w1_runner_qps_delta_pct",
            "acceptance_row_13_vllm_offline_diagnostic",
            "acceptance_row_14_vllm_offline_diagnostic",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
            "results/qwen35_report_audit_20260619/runtime_image_contract.json",
            "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
            "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
            "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
            "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "build_runtime_image_contract",
            "build_vllm_optimization_lock",
            "build_vllm_online_parity_protocol",
            "summarize_vllm_log_stages",
            "diagnose_vllm_admission",
            "vllm_c1_original",
            "vllm_c8_original",
            "vllm_c8_prebuild_w4",
        ],
        "status": "PASS_WITH_CAVEAT",
        "required_for_share": True,
        "reviewer_hook": "Defend baseline strength with image/flag locks first, then keep vLLM c=8 prebuild w4 scoped as offline diagnostic.",
        "caveat": "vLLM c=8 prebuild w4 is optimized offline diagnostic evidence, not online serving parity.",
    },
    {
        "requirement_id": "single_and_high_concurrency",
        "category": "concurrency",
        "original_user_intent": "Cover single concurrency and high concurrency pressure.",
        "normalized_requirement": "SGLang Video-AMME pressure must cover c=1/2/4/8/16, including c=8 serving peak and c=16 saturation boundary.",
        "source_objective_row_ids": [
            "single_concurrency",
            "high_concurrency",
        ],
        "claim_ids": ["sglang_c8_current_high_concurrency_peak"],
        "metric_row_ids": [
            "sglang_videoamme_c1_latency_mean_s",
            "sglang_videoamme_c1_rtf_mean",
            "sglang_videoamme_c1_throughput_qps",
            "sglang_videoamme_c2_latency_mean_s",
            "sglang_videoamme_c2_rtf_mean",
            "sglang_videoamme_c2_throughput_qps",
            "sglang_videoamme_c4_latency_mean_s",
            "sglang_videoamme_c4_rtf_mean",
            "sglang_videoamme_c4_throughput_qps",
            "sglang_videoamme_c8_latency_mean_s",
            "sglang_videoamme_c8_rtf_mean",
            "sglang_videoamme_c8_throughput_qps",
            "sglang_videoamme_c16_latency_mean_s",
            "sglang_videoamme_c16_rtf_mean",
            "sglang_videoamme_c16_throughput_qps",
            "sglang_videoamme_c16_vs_c8_qps_delta_pct",
            "acceptance_row_02_sglang_videoamme_stress",
            "acceptance_row_03_sglang_videoamme_stress",
            "acceptance_row_04_sglang_videoamme_stress",
            "acceptance_row_05_sglang_videoamme_stress",
            "acceptance_row_06_sglang_videoamme_stress",
            "stage_drilldown_budget-sglang_videoamme_budget-01",
            "stage_drilldown_budget-sglang_videoamme_budget-02",
            "stage_drilldown_budget-sglang_videoamme_budget-03",
            "stage_drilldown_budget-sglang_videoamme_budget-04",
            "stage_drilldown_budget-sglang_videoamme_budget-05",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/tables_summary.json",
            "results/qwen35_report_audit_20260619/acceptance_matrix.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "sglang_videoamme_stress",
            "build_report_tables",
            "build_acceptance_matrix",
            "build_stage_latency_budget",
        ],
        "status": "PASS_WITH_CAVEAT",
        "required_for_share": True,
        "reviewer_hook": "Show c=1 health, c=4 strict comparison, c=8 peak throughput, and c=16 as the saturation boundary.",
        "caveat": "c=16 is pressure-boundary evidence, not the recommended serving point.",
    },
    {
        "requirement_id": "short_and_long_text_speech",
        "category": "text_length",
        "original_user_intent": "Cover both short and long text/speech cases.",
        "normalized_requirement": "Synthetic short/long text-to-speech guardrails must cover c=1/4/8, with long c=8 remaining faster than real time.",
        "source_objective_row_ids": ["short_and_long_text"],
        "claim_ids": ["short_and_long_tts_are_covered"],
        "metric_row_ids": [
            "synthetic_long_c8_target_chars",
            "synthetic_long_c8_target_words",
            "synthetic_long_c8_audio_duration_mean_s",
            "synthetic_long_c8_latency_mean_s",
            "synthetic_long_c8_rtf_mean",
            "synthetic_long_c8_rtf_p95",
            "synthetic_long_c8_audio_throughput_s_per_s",
            "acceptance_row_07_sglang_synthetic_speech",
            "acceptance_row_08_sglang_synthetic_speech",
            "acceptance_row_09_sglang_synthetic_speech",
            "acceptance_row_10_sglang_synthetic_speech",
            "acceptance_row_11_sglang_synthetic_speech",
            "acceptance_row_12_sglang_synthetic_speech",
            "stage_drilldown_budget-synthetic_speech_budget-01",
            "stage_drilldown_budget-synthetic_speech_budget-02",
            "stage_drilldown_budget-synthetic_speech_budget-03",
            "stage_drilldown_budget-synthetic_speech_budget-04",
            "stage_drilldown_budget-synthetic_speech_budget-05",
            "stage_drilldown_budget-synthetic_speech_budget-06",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/tables_summary.json",
            "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json",
            "results/qwen35_report_audit_20260619/acceptance_matrix.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "sglang_synthetic_text_to_speech",
            "build_seedtts_meta",
            "build_report_tables",
            "build_acceptance_matrix",
        ],
        "status": "PASS",
        "required_for_share": True,
        "reviewer_hook": "Use acceptance rows for all short/long c=1/4/8 cells and long-c8 numeric rows for the high-concurrency guardrail.",
        "caveat": "",
    },
    {
        "requirement_id": "specific_breakdown",
        "category": "stage_breakdown",
        "original_user_intent": "Provide a very specific breakdown.",
        "normalized_requirement": "The package must expose concrete stage budgets, stage boundaries, and per-boundary bottleneck decisions.",
        "source_objective_row_ids": [
            "stage_breakdown_each_regime",
            "stage_connections_and_bottlenecks",
        ],
        "claim_ids": [
            "stage_handoff_is_not_stalled",
            "code2wav_decode_not_current_compute_bottleneck",
        ],
        "metric_row_ids": [
            "stage_drilldown_summary_rows_total",
            "stage_drilldown_summary_boundary_rows_total",
            "stage_drilldown_summary_budget_rows_total",
            "stage_drilldown_summary_stage_routes_total",
            "stage_drilldown_summary_required_failures",
            "stage_drilldown_stage-boundary-02",
            "stage_drilldown_stage-boundary-03",
            "stage_drilldown_stage-boundary-08",
            "stage_drilldown_stage-boundary-09",
            "stage_drilldown_stage-boundary-10",
            "stage_drilldown_stage-boundary-11",
            "stage_drilldown_stage-boundary-12",
            "acceptance_row_17_stage_connection_health",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            "results/qwen35_report_audit_20260619/stage_causal_graph.json",
            "results/qwen35_report_audit_20260619/stage_drilldown_index.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "build_stage_interactions",
            "build_stage_latency_budget",
            "build_stage_boundary_bottleneck_ledger",
            "build_stage_drilldown_index",
        ],
        "status": "PASS",
        "required_for_share": True,
        "reviewer_hook": "Start from stage_latency_budget for percentages, then use stage_boundary_bottleneck_ledger for boundary-level bottleneck calls.",
        "caveat": "",
    },
    {
        "requirement_id": "every_stage_analyzed",
        "category": "stage_coverage",
        "original_user_intent": "Analyze performance for every stage.",
        "normalized_requirement": "The stage index must cover SGLang, synthetic speech, and vLLM diagnostic stage routes with no required failures.",
        "source_objective_row_ids": [
            "stage_breakdown_each_regime",
            "stage_connections_and_bottlenecks",
        ],
        "claim_ids": [
            "stage_handoff_is_not_stalled",
            "code2wav_decode_not_current_compute_bottleneck",
            "vllm_c8_prebuild_w4_is_offline_diagnostic",
        ],
        "metric_row_ids": [
            "stage_drilldown_summary_rows_total",
            "stage_drilldown_summary_boundary_rows_total",
            "stage_drilldown_summary_budget_rows_total",
            "stage_drilldown_summary_stage_routes_total",
            "stage_drilldown_summary_checks_passed",
            "stage_drilldown_summary_required_failures",
            "stage_drilldown_stage-boundary-01",
            "stage_drilldown_stage-boundary-10",
            "stage_drilldown_stage-boundary-13",
            "stage_drilldown_stage-boundary-16",
            "stage_drilldown_stage-boundary-22",
            "stage_drilldown_stage-boundary-31",
            "stage_drilldown_stage-boundary-37",
            "stage_drilldown_budget-sglang_videoamme_budget-04",
            "stage_drilldown_budget-synthetic_speech_budget-06",
            "stage_drilldown_budget-vllm_offline_budget-04",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/stage_drilldown_index.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/stage_causal_graph.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "build_stage_drilldown_index",
            "build_stage_boundary_bottleneck_ledger",
            "build_stage_latency_budget",
            "build_stage_causal_graph",
        ],
        "status": "PASS",
        "required_for_share": True,
        "reviewer_hook": "Use the stage drilldown summary as the coverage gate, then route to the relevant budget or boundary rows.",
        "caveat": "",
    },
    {
        "requirement_id": "stage_connections_and_interactions",
        "category": "stage_connections",
        "original_user_intent": "Analyze whether stage connections create bottlenecks or influence each other.",
        "normalized_requirement": "Stage-to-stage connections must distinguish healthy handoff, queue/admission bottlenecks, negative preprocessing interaction, and vLLM prompt-feed limitations.",
        "source_objective_row_ids": [
            "stage_connections_and_bottlenecks",
            "optimization_recipe_and_antirecipe",
        ],
        "claim_ids": [
            "stage_handoff_is_not_stalled",
            "code2wav_decode_not_current_compute_bottleneck",
            "raising_preprocessing_concurrency_is_negative_recipe",
            "vllm_c8_prebuild_w4_is_offline_diagnostic",
        ],
        "metric_row_ids": [
            "stage_drilldown_stage-boundary-02",
            "stage_drilldown_stage-boundary-03",
            "stage_drilldown_stage-boundary-08",
            "stage_drilldown_stage-boundary-09",
            "stage_drilldown_stage-boundary-10",
            "stage_drilldown_stage-boundary-11",
            "stage_drilldown_stage-boundary-12",
            "stage_drilldown_stage-boundary-13",
            "stage_drilldown_stage-boundary-16",
            "stage_drilldown_stage-boundary-31",
            "stage_drilldown_stage-boundary-37",
            "acceptance_row_15_negative_optimization",
            "acceptance_row_16_negative_optimization",
            "acceptance_row_17_stage_connection_health",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
            "results/qwen35_report_audit_20260619/rerun_delta_triage.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            "results/qwen35_report_audit_20260619/stage_causal_graph.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "build_stage_interactions",
            "build_stage_boundary_bottleneck_ledger",
            "build_stage_causal_graph",
            "build_rerun_delta_triage",
        ],
        "status": "PASS",
        "required_for_share": True,
        "reviewer_hook": "Use the causal graph for interaction narrative and the ledger rows for exact boundary decisions.",
        "caveat": "",
    },
    {
        "requirement_id": "current_best_recipe_and_antirecipes",
        "category": "optimization",
        "original_user_intent": "Open all useful optimizations, keep the base strong, and reach the current optimum.",
        "normalized_requirement": "The current SGLang recipe, vLLM optimized baseline, c=8 peak, and negative preprocessing recipes must be locked with rollback rules.",
        "source_objective_row_ids": [
            "optimization_recipe_and_antirecipe",
            "sglang_reproduction_path",
            "vllm_reproduction_path",
        ],
        "claim_ids": [
            "sglang_c8_current_high_concurrency_peak",
            "raising_preprocessing_concurrency_is_negative_recipe",
            "vllm_baseline_is_optimized",
        ],
        "metric_row_ids": [
            "sglang_videoamme_c8_throughput_qps",
            "sglang_videoamme_c16_vs_c8_qps_delta_pct",
            "stage_drilldown_stage-boundary-16",
            "acceptance_row_15_negative_optimization",
            "acceptance_row_16_negative_optimization",
            "vllm_c8_prebuild_w4_runner_qps",
            "vllm_c8_prebuild_w4_engine_qps",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
            "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
            "results/qwen35_report_audit_20260619/acceptance_matrix.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "launch_sglang_optimized",
            "sglang_videoamme_stress",
            "build_sglang_optimization_lock",
            "build_vllm_optimization_lock",
            "build_acceptance_matrix",
        ],
        "status": "PASS_WITH_CAVEAT",
        "required_for_share": True,
        "reviewer_hook": "Call this the current measured best recipe, backed by locks and anti-recipes, not a proof of a global optimum.",
        "caveat": "The package proves the best measured current recipe under the audited environment, not a global optimum over all future kernels or placements.",
    },
    {
        "requirement_id": "reproduce_sglang_and_vllm",
        "category": "reproducibility",
        "original_user_intent": "Make it possible to reproduce SGLang-Omni and vLLM performance from the document.",
        "normalized_requirement": "Reproduction command manifest, preflight, environment snapshot, external handoff, and share package validation must cover SGLang and vLLM reruns.",
        "source_objective_row_ids": [
            "sglang_reproduction_path",
            "vllm_reproduction_path",
            "full_audit_and_preflight",
            "share_package_and_hashes",
        ],
        "claim_ids": ["package_share_ready_with_boundaries"],
        "metric_row_ids": [
            "acceptance_row_01_strict_runtime_comparison",
            "acceptance_row_05_sglang_videoamme_stress",
            "acceptance_row_07_sglang_synthetic_speech",
            "acceptance_row_13_vllm_offline_diagnostic",
            "acceptance_row_14_vllm_offline_diagnostic",
            "stage_drilldown_summary_checks_passed",
            "stage_drilldown_summary_required_failures",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/preflight_repro.json",
            "results/qwen35_report_audit_20260619/environment_snapshot.json",
            "results/qwen35_report_audit_20260619/share_package_validation.json",
            "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
            "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
            "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "run_full_audit",
            "launch_sglang_optimized",
            "sglang_videoamme_stress",
            "sglang_synthetic_text_to_speech",
            "sglang_recompute_wer",
            "vllm_c1_original",
            "vllm_c8_original",
            "vllm_c8_prebuild_w4",
            "run_preflight",
            "validate_share_bundle_package",
            "validate_share_bundle_receiver_smoke",
            "validate_extracted_share_bundle",
            "validate_external_standalone_share_bundle",
        ],
        "status": "PASS",
        "required_for_share": True,
        "reviewer_hook": "Start with run_full_audit and preflight, then run SGLang/vLLM benchmark commands only after environment gates pass.",
        "caveat": "",
    },
    {
        "requirement_id": "professional_confident_share_package",
        "category": "share_confidence",
        "original_user_intent": "Produce a professional, confident report that can be shared with a collaborating university.",
        "normalized_requirement": "The final package must contain human-readable reports, machine gates, defense Q&A, caveat boundaries, hashable bundle, and receiver validation.",
        "source_objective_row_ids": [
            "professional_share_report",
            "confidence_and_wording",
            "share_package_and_hashes",
        ],
        "claim_ids": ["package_share_ready_with_boundaries"],
        "metric_row_ids": [
            "stage_drilldown_summary_rows_total",
            "stage_drilldown_summary_checks_passed",
            "stage_drilldown_summary_required_failures",
            "acceptance_row_01_strict_runtime_comparison",
            "acceptance_row_14_vllm_offline_diagnostic",
            "acceptance_row_17_stage_connection_health",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/final_readiness_audit.json",
            "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
            "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
            "results/qwen35_report_audit_20260619/share_package_validation.json",
            "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
            "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
            "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
            "results/qwen35_report_audit_20260619/confidence_ledger.json",
            "results/qwen35_report_audit_20260619/defense_claim_matrix.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "build_final_readiness_audit",
            "build_share_bundle_manifest",
            "build_share_bundle_package",
            "validate_share_bundle_package",
            "validate_share_bundle_receiver_smoke",
            "validate_extracted_share_bundle",
            "validate_external_standalone_share_bundle",
            "build_defense_claim_matrix",
            "build_claim_metric_crosswalk",
        ],
        "status": "PASS_WITH_CAVEAT",
        "required_for_share": True,
        "reviewer_hook": "Share-ready means the package is defensible with documented caveats; it is not a universal parity claim.",
            "caveat": "The package is share-ready with documented caveats; completion now depends on the final evidence audit rather than an evening time gate.",
    },
    {
        "requirement_id": "final_evidence_completion",
        "category": "checkpoint",
        "original_user_intent": "Do not focus on 2026-06-21 evening; provide the comprehensive shareable report once the evidence is complete.",
        "normalized_requirement": "The current evidence package can complete when final completion audit, release seal, package validation, and reproduction gates all pass.",
        "source_objective_row_ids": ["final_evidence_completion"],
        "claim_ids": [],
        "metric_row_ids": [
            "stage_drilldown_summary_rows_total",
            "stage_drilldown_summary_required_failures",
        ],
        "machine_evidence": [
            "results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json",
            "results/qwen35_report_audit_20260619/objective_completion_audit.json",
        ],
        "public_docs": [
            "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
        ],
        "rerun_command_ids": [
            "build_final_checkpoint_watchlist",
            "build_objective_completion_audit",
        ],
        "status": "PASS_WITH_CAVEAT",
        "required_for_share": False,
        "reviewer_hook": "Mark the thread goal complete only after final_completion_audit.completion_allowed_now=true and the updated objective is checked requirement by requirement.",
        "caveat": "This is a completion-control row, not a performance claim.",
    },
]


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


def _audit_green_or_in_progress(payload: dict[str, Any]) -> bool:
    if bool(payload.get("ok")):
        return True
    if not bool(payload.get("in_progress")):
        return False
    steps = payload.get("steps", [])
    return isinstance(steps, list) and all(
        isinstance(step, dict) and bool(step.get("ok")) for step in steps
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _rel_path(root: Path, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path)
    return text


def _dedupe(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_paths(root: Path, values: list[Any]) -> list[str]:
    return _dedupe([_rel_path(root, value) for value in values])


def _path_exists(root: Path, rel_path: str) -> bool:
    path = Path(rel_path)
    if path.is_absolute():
        return path.exists()
    return (root / rel_path).exists()


def _command_ids(repro_manifest: dict[str, Any]) -> set[str]:
    return {
        str(command.get("id"))
        for command in _as_list(repro_manifest.get("commands"))
        if isinstance(command, dict) and command.get("id")
    }


def _metric_rows(metric_provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id")): row
        for row in _as_list(metric_provenance.get("rows"))
        if isinstance(row, dict) and row.get("id")
    }


def _claim_rows(claim_metric_crosswalk: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in _as_list(claim_metric_crosswalk.get("rows"))
        if isinstance(row, dict) and row.get("claim_id")
    }


def _objective_rows(objective_completion: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("requirement_id")): row
        for row in _as_list(objective_completion.get("objective_rows"))
        if isinstance(row, dict) and row.get("requirement_id")
    }


def _is_raw_artifact(path: str) -> bool:
    return path.startswith("results/qwen35_") and not path.startswith(
        AUDIT_DIR.as_posix()
    )


def _compact_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "runtime": row.get("runtime"),
        "scenario": row.get("scenario"),
        "concurrency": row.get("concurrency"),
        "stage_route": row.get("stage_route", []),
        "metric_name": row.get("metric_name"),
        "metric_value": row.get("metric_value"),
        "metric_unit": row.get("metric_unit"),
        "metric_text": row.get("metric_text"),
        "interpretation": row.get("interpretation"),
        "source_json": row.get("source_json"),
        "source_pointer": row.get("source_pointer"),
    }


def _compact_objective_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "requirement_id": row.get("requirement_id"),
        "status": row.get("status"),
        "required_for_share": row.get("required_for_share"),
        "requirement": row.get("requirement"),
        "evidence": row.get("evidence"),
        "caveat": row.get("caveat"),
    }


def _extended_metric_row_ids(
    spec: dict[str, Any],
    claim_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    ids = list(_as_list(spec.get("metric_row_ids")))
    for claim_id in _as_list(spec.get("claim_ids")):
        claim_row = claim_by_id.get(str(claim_id))
        if claim_row:
            ids.extend(_as_list(claim_row.get("metric_row_ids")))
    return _dedupe(ids)


def _metric_value(
    metric_by_id: dict[str, dict[str, Any]],
    row_id: str,
    default: Any = None,
) -> Any:
    row = metric_by_id.get(row_id, {})
    return row.get("metric_value", default)


def _candidate_metric_rows(
    metric_by_id: dict[str, dict[str, Any]],
    row_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        _compact_metric_row(metric_by_id[row_id])
        for row_id in row_ids
        if row_id in metric_by_id
    ]


def _candidate_row(
    root: Path,
    metric_by_id: dict[str, dict[str, Any]],
    command_ids: set[str],
    spec: dict[str, Any],
) -> dict[str, Any]:
    metric_row_ids = [str(row_id) for row_id in _as_list(spec.get("metric_row_ids"))]
    metric_rows = _candidate_metric_rows(metric_by_id, metric_row_ids)
    metric_evidence = [
        evidence
        for row in metric_rows
        for evidence in _as_list(row.get("evidence_files"))
    ]
    evidence_files = _normalize_paths(
        root,
        [
            *list(_as_list(spec.get("machine_evidence"))),
            *metric_evidence,
        ],
    )
    rerun_command_ids = _dedupe(
        [
            *list(_as_list(spec.get("rerun_command_ids"))),
            *[
                command_id
                for row in metric_rows
                for command_id in _as_list(row.get("rerun_command_ids"))
            ],
        ]
    )
    missing_metric_row_ids = [
        row_id for row_id in metric_row_ids if row_id not in metric_by_id
    ]
    missing_evidence_files = [
        evidence for evidence in evidence_files if not _path_exists(root, evidence)
    ]
    missing_command_ids = [
        command_id for command_id in rerun_command_ids if command_id not in command_ids
    ]

    return {
        "candidate_id": spec["candidate_id"],
        "runtime": spec["runtime"],
        "scope": spec["scope"],
        "decision_class": spec["decision_class"],
        "decision": spec["decision"],
        "rationale": spec["rationale"],
        "replacement_rule": spec["replacement_rule"],
        "metric_row_ids": metric_row_ids,
        "metric_rows": metric_rows,
        "machine_evidence": evidence_files,
        "raw_artifacts": sorted(
            evidence for evidence in evidence_files if _is_raw_artifact(evidence)
        ),
        "packaged_evidence": sorted(
            evidence for evidence in evidence_files if not _is_raw_artifact(evidence)
        ),
        "rerun_command_ids": rerun_command_ids,
        "missing_metric_row_ids": missing_metric_row_ids,
        "missing_evidence_files": missing_evidence_files,
        "missing_command_ids": missing_command_ids,
        "ready": not (
            missing_metric_row_ids or missing_evidence_files or missing_command_ids
        ),
    }


def _build_optimization_candidate_ledger(
    root: Path,
    metric_by_id: dict[str, dict[str, Any]],
    command_ids: set[str],
) -> dict[str, Any]:
    candidate_specs = [
        {
            "candidate_id": "sglang_current_best_measured_recipe",
            "runtime": "sglang",
            "scope": "strict c=4 parity plus SGLang c=1/2/4/8/16 pressure envelope",
            "decision_class": "current_best_measured",
            "decision": "accept_current_recipe",
            "rationale": (
                "Use the locked SGLang recipe as the current best measured serving "
                "recipe: strict c=4 beats the optimized vLLM baseline on latency/RTF, "
                "c=8 is the audited serving peak, and c=16 regresses throughput."
            ),
            "replacement_rule": (
                "A replacement must rerun strict c=4 SGLang/vLLM, SGLang c=1/2/4/8/16, "
                "WER/accuracy, stage causal graph, acceptance matrix, and final readiness."
            ),
            "metric_row_ids": [
                "strict_c4_sglang_lower_latency_mean_pct",
                "strict_c4_sglang_lower_latency_p95_pct",
                "strict_c4_sglang_lower_rtf_mean_pct",
                "strict_c4_sglang_lower_rtf_p95_pct",
                "sglang_videoamme_c8_throughput_qps",
                "sglang_videoamme_c16_vs_c8_qps_delta_pct",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/headline_scorecard.json",
                "results/qwen35_report_audit_20260619/acceptance_matrix.json",
                "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
                "results/qwen35_report_audit_20260619/stage_latency_budget.json",
                "results/qwen35_report_audit_20260619/stage_causal_graph.json",
            ],
            "rerun_command_ids": [
                "build_headline_scorecard",
                "build_acceptance_matrix",
                "build_sglang_optimization_lock",
                "build_stage_latency_budget",
                "build_stage_causal_graph",
                "sglang_videoamme_stress",
            ],
        },
        {
            "candidate_id": "sglang_c16_saturation_boundary",
            "runtime": "sglang",
            "scope": "Video-AMME c=16 high-concurrency pressure boundary",
            "decision_class": "not_recommended_default",
            "decision": "keep_as_saturation_boundary",
            "rationale": (
                "c=16 is useful pressure evidence, but its throughput is below c=8, "
                "so it should not become the default serving recipe."
            ),
            "replacement_rule": (
                "Promote c=16 only if a rerun improves c=16 throughput over c=8 "
                "without WER/accuracy or stage-handoff regression."
            ),
            "metric_row_ids": [
                "sglang_videoamme_c8_throughput_qps",
                "sglang_videoamme_c16_throughput_qps",
                "sglang_videoamme_c16_vs_c8_qps_delta_pct",
                "stage_drilldown_stage-boundary-13",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/acceptance_matrix.json",
                "results/qwen35_report_audit_20260619/stage_latency_budget.json",
                "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            ],
            "rerun_command_ids": [
                "sglang_videoamme_stress",
                "build_acceptance_matrix",
                "build_stage_latency_budget",
                "build_stage_boundary_bottleneck_ledger",
            ],
        },
        {
            "candidate_id": "sglang_preproc2_widening",
            "runtime": "sglang",
            "scope": "PREPROCESSING_MAX_CONCURRENCY=2 at c=8",
            "decision_class": "rejected_anti_recipe",
            "decision": "reject_current_recipe_change",
            "rationale": (
                "preproc=2 reduces c=8 throughput and increases latency relative to "
                "the locked baseline, so it is an anti-recipe in this environment."
            ),
            "replacement_rule": (
                "Reconsider only with a placement/admission redesign and a new c=8 "
                "run that beats the baseline plus stage-boundary checks."
            ),
            "metric_row_ids": [
                "acceptance_row_15_negative_optimization",
                "stage_drilldown_stage-boundary-16",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/acceptance_matrix.json",
                "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
                "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            ],
            "rerun_command_ids": [
                "build_acceptance_matrix",
                "build_sglang_optimization_lock",
                "build_stage_boundary_bottleneck_ledger",
            ],
        },
        {
            "candidate_id": "sglang_preproc4_widening",
            "runtime": "sglang",
            "scope": "PREPROCESSING_MAX_CONCURRENCY=4 at c=8",
            "decision_class": "rejected_anti_recipe",
            "decision": "reject_current_recipe_change",
            "rationale": (
                "preproc=4 causes failed samples and low accuracy, making it a "
                "quality-breaking anti-recipe rather than a throughput optimization."
            ),
            "replacement_rule": (
                "Reconsider only if failures disappear and the full acceptance matrix "
                "passes with comparable or better latency/throughput."
            ),
            "metric_row_ids": ["acceptance_row_16_negative_optimization"],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/acceptance_matrix.json",
                "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
            ],
            "rerun_command_ids": [
                "build_acceptance_matrix",
                "build_sglang_optimization_lock",
            ],
        },
        {
            "candidate_id": "vllm_optimized_c4_baseline",
            "runtime": "vllm",
            "scope": "strict warmed c=4 vLLM comparison baseline",
            "decision_class": "accepted_strict_baseline",
            "decision": "accept_as_strong_baseline",
            "rationale": (
                "The vLLM baseline is image/flag locked and used for strict c=4 "
                "headline parity; SGLang must beat this baseline rather than a weak run."
            ),
            "replacement_rule": (
                "Any stronger vLLM c=4 recipe must update the runtime image contract, "
                "runtime comparison contract, headline scorecard, and final readiness."
            ),
            "metric_row_ids": [
                "strict_c4_vllm_latency_mean_s",
                "strict_c4_vllm_latency_p95_s",
                "strict_c4_vllm_rtf_mean",
                "strict_c4_vllm_rtf_p95",
                "strict_c4_sglang_lower_latency_mean_pct",
                "strict_c4_sglang_lower_rtf_mean_pct",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/runtime_image_contract.json",
                "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
                "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
                "results/qwen35_report_audit_20260619/headline_scorecard.json",
            ],
            "rerun_command_ids": [
                "vllm_c4_original",
                "build_runtime_image_contract",
                "build_vllm_optimization_lock",
                "build_runtime_comparison_contract",
                "build_headline_scorecard",
            ],
        },
        {
            "candidate_id": "vllm_original_c8_offline",
            "runtime": "vllm",
            "scope": "original vLLM c=8 offline runner",
            "decision_class": "diagnostic_only",
            "decision": "keep_as_prompt_feed_diagnostic",
            "rationale": (
                "The original vLLM c=8 run is dominated by offline runner prompt "
                "build/feed admission, so it diagnoses host feeding rather than online parity."
            ),
            "replacement_rule": (
                "Promote to online parity only after adding an online ingress run with "
                "the same image/model/data and admission artifacts."
            ),
            "metric_row_ids": [
                "vllm_c8_original_runner_overhead_pct_wall",
                "vllm_c8_original_engine_qps",
                "vllm_c8_original_admission_batch_admission_span_avg_ms",
                "vllm_c8_original_admission_prompt_feed_limited",
                "stage_drilldown_stage-boundary-31",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
                "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
                "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
            ],
            "rerun_command_ids": [
                "vllm_c8_original",
                "diagnose_vllm_admission",
                "summarize_vllm_log_stages",
                "build_runtime_comparison_contract",
            ],
        },
        {
            "candidate_id": "vllm_c8_prebuild_w4_offline",
            "runtime": "vllm",
            "scope": "vLLM c=8 prebuilt prompts with 4 prompt-build workers",
            "decision_class": "optimized_offline_diagnostic",
            "decision": "accept_as_strongest_current_offline_diagnostic",
            "rationale": (
                "prebuild w4 lowers prompt-build wall time and improves runner QPS "
                "versus w1, but remains scoped as offline diagnostic until online "
                "ingress parity artifacts exist."
            ),
            "replacement_rule": (
                "Use as vLLM c=8 diagnostic evidence only; do not replace the strict "
                "c=4 headline baseline or claim online parity without the protocol run."
            ),
            "metric_row_ids": [
                "vllm_c8_prebuild_w4_runner_qps",
                "vllm_c8_prebuild_w4_engine_qps",
                "vllm_c8_prebuild_w4_admission_batch_admission_span_avg_ms",
                "vllm_c8_w4_vs_w1_prompt_build_wall_delta_pct",
                "vllm_c8_w4_vs_w1_runner_qps_delta_pct",
                "stage_drilldown_stage-boundary-37",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
                "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
                "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
            ],
            "rerun_command_ids": [
                "vllm_c8_prebuild_w4",
                "diagnose_vllm_admission",
                "build_vllm_optimization_lock",
                "build_vllm_online_parity_protocol",
            ],
        },
        {
            "candidate_id": "code2wav_first_tuning",
            "runtime": "sglang",
            "scope": "talker_ar to code2wav handoff and vocoder decode",
            "decision_class": "deprioritized_not_current_bottleneck",
            "decision": "do_not_optimize_first",
            "rationale": (
                "The stage graph marks SGLang handoff healthy and code2wav decode not "
                "the current compute bottleneck; tune admission/Talker cadence before "
                "spending effort on code2wav-first work."
            ),
            "replacement_rule": (
                "Promote code2wav-first tuning only if new stage artifacts show decode "
                "or hop p95 becoming the bottleneck across the serving window."
            ),
            "metric_row_ids": [
                "stage_drilldown_stage-boundary-03",
                "stage_drilldown_stage-boundary-09",
                "stage_drilldown_stage-boundary-12",
                "acceptance_row_17_stage_connection_health",
            ],
            "machine_evidence": [
                "results/qwen35_report_audit_20260619/stage_causal_graph.json",
                "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
                "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            ],
            "rerun_command_ids": [
                "build_stage_causal_graph",
                "build_stage_interactions",
                "build_stage_boundary_bottleneck_ledger",
            ],
        },
    ]

    rows = [
        _candidate_row(root, metric_by_id, command_ids, spec)
        for spec in candidate_specs
    ]
    missing_metric_row_ids = sorted(
        {
            row_id
            for row in rows
            for row_id in _as_list(row.get("missing_metric_row_ids"))
        }
    )
    missing_evidence_files = sorted(
        {
            evidence
            for row in rows
            for evidence in _as_list(row.get("missing_evidence_files"))
        }
    )
    missing_command_ids = sorted(
        {
            command_id
            for row in rows
            for command_id in _as_list(row.get("missing_command_ids"))
        }
    )
    decision_classes = sorted(
        {str(row.get("decision_class")) for row in rows if row.get("decision_class")}
    )
    current_best_rows = [
        row
        for row in rows
        if row.get("decision_class") == "current_best_measured"
        and row.get("decision") == "accept_current_recipe"
    ]
    rejected_anti_recipe_rows = [
        row for row in rows if row.get("decision_class") == "rejected_anti_recipe"
    ]
    vllm_rows = [row for row in rows if row.get("runtime") == "vllm"]
    diagnostic_rows = [
        row
        for row in rows
        if row.get("decision_class")
        in {"diagnostic_only", "optimized_offline_diagnostic"}
    ]

    checks = {
        "candidate_rows_total": len(rows) >= 8,
        "current_best_candidate_declared": len(current_best_rows) == 1
        and bool(current_best_rows[0].get("ready")),
        "anti_recipes_locked": len(rejected_anti_recipe_rows) >= 2
        and all(bool(row.get("ready")) for row in rejected_anti_recipe_rows),
        "vllm_baseline_and_diagnostic_locked": len(vllm_rows) >= 3
        and any(row.get("decision_class") == "accepted_strict_baseline" for row in vllm_rows)
        and any(
            row.get("decision_class") == "optimized_offline_diagnostic"
            for row in vllm_rows
        ),
        "candidate_metric_rows_present": not missing_metric_row_ids,
        "candidate_evidence_present": not missing_evidence_files,
        "candidate_commands_present": not missing_command_ids,
        "not_global_optimum_boundary_documented": any(
            "global optimum" in str(row.get("replacement_rule") or "")
            or row.get("decision_class") == "not_recommended_default"
            for row in rows
        ),
        "stage_first_tuning_priority_present": any(
            row.get("candidate_id") == "code2wav_first_tuning"
            and row.get("decision_class") == "deprioritized_not_current_bottleneck"
            for row in rows
        ),
    }
    not_global_boundary_documented = checks[
        "not_global_optimum_boundary_documented"
    ]
    required_failures = [name for name, ok in checks.items() if not ok]
    sglang_c16_delta = _metric_value(
        metric_by_id, "sglang_videoamme_c16_vs_c8_qps_delta_pct"
    )
    vllm_prompt_feed_limited = _metric_value(
        metric_by_id, "vllm_c8_original_admission_prompt_feed_limited"
    )
    return {
        "summary": {
            "ready": not required_failures,
            "candidate_rows_total": len(rows),
            "decision_classes_total": len(decision_classes),
            "current_best_candidate_id": (
                current_best_rows[0]["candidate_id"] if current_best_rows else ""
            ),
            "current_best_candidate_ready": bool(current_best_rows)
            and bool(current_best_rows[0].get("ready")),
            "accepted_current_best_rows_total": len(current_best_rows),
            "rejected_anti_recipe_rows_total": len(rejected_anti_recipe_rows),
            "vllm_rows_total": len(vllm_rows),
            "vllm_diagnostic_rows_total": len(diagnostic_rows),
            "not_global_optimum_boundary": not_global_boundary_documented,
            "sglang_c16_vs_c8_qps_delta_pct": sglang_c16_delta,
            "vllm_original_c8_prompt_feed_limited": bool(vllm_prompt_feed_limited),
            "missing_metric_row_ids": len(missing_metric_row_ids),
            "missing_evidence_files": len(missing_evidence_files),
            "missing_command_ids": len(missing_command_ids),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "Machine-readable verdict ledger for current best measured recipe, "
                "anti-recipes, strong vLLM baseline, vLLM c=8 diagnostic scope, and "
                "stage-first tuning priority."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "decision_classes": decision_classes,
            "missing_metric_row_ids": missing_metric_row_ids,
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "required_failures": required_failures,
        },
        "rows": rows,
    }


def build_objective_requirement_crosswalk(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_summary = _load_json_optional(root / AUDIT_SUMMARY)
    objective_completion = _load_json_optional(root / OBJECTIVE_COMPLETION)
    checkpoint_watchlist = _load_json_optional(root / FINAL_CHECKPOINT)
    claim_metric_crosswalk = _load_json_optional(root / CLAIM_METRIC_CROSSWALK)
    metric_provenance = _load_json_optional(root / METRIC_PROVENANCE)
    repro_manifest = _load_json_optional(root / REPRO_COMMANDS)

    objective_by_id = _objective_rows(objective_completion)
    claim_by_id = _claim_rows(claim_metric_crosswalk)
    metric_by_id = _metric_rows(metric_provenance)
    command_ids = _command_ids(repro_manifest)

    rows: list[dict[str, Any]] = []
    missing_objective_row_ids: list[str] = []
    missing_claim_ids: list[str] = []
    missing_metric_row_ids: list[str] = []

    for spec in OBJECTIVE_REQUIREMENTS:
        source_objective_ids = [
            str(row_id) for row_id in _as_list(spec.get("source_objective_row_ids"))
        ]
        source_objective_rows = [
            objective_by_id[row_id]
            for row_id in source_objective_ids
            if row_id in objective_by_id
        ]
        missing_objective_row_ids.extend(
            row_id for row_id in source_objective_ids if row_id not in objective_by_id
        )

        claim_ids = [str(claim_id) for claim_id in _as_list(spec.get("claim_ids"))]
        claim_rows = [
            claim_by_id[claim_id] for claim_id in claim_ids if claim_id in claim_by_id
        ]
        missing_claim_ids.extend(
            claim_id for claim_id in claim_ids if claim_id not in claim_by_id
        )

        metric_row_ids = _extended_metric_row_ids(spec, claim_by_id)
        missing_for_requirement = [
            row_id for row_id in metric_row_ids if row_id not in metric_by_id
        ]
        missing_metric_row_ids.extend(missing_for_requirement)
        metric_rows = [
            metric_by_id[row_id]
            for row_id in metric_row_ids
            if row_id in metric_by_id
        ]

        evidence_files = _normalize_paths(
            root,
            [
                *COMMON_MACHINE_EVIDENCE,
                *list(_as_list(spec.get("machine_evidence"))),
                *list(_as_list(spec.get("public_docs"))),
                *[
                    evidence
                    for claim_row in claim_rows
                    for evidence in _as_list(claim_row.get("machine_evidence"))
                ],
                *[
                    evidence
                    for metric_row in metric_rows
                    for evidence in _as_list(metric_row.get("evidence_files"))
                ],
            ],
        )
        public_docs = _normalize_paths(root, list(_as_list(spec.get("public_docs"))))
        rerun_command_ids = _dedupe(
            [
                *list(_as_list(spec.get("rerun_command_ids"))),
                *[
                    command_id
                    for claim_row in claim_rows
                    for command_id in _as_list(claim_row.get("rerun_command_ids"))
                ],
                *[
                    command_id
                    for metric_row in metric_rows
                    for command_id in _as_list(metric_row.get("rerun_command_ids"))
                ],
                "build_objective_requirement_crosswalk",
            ]
        )
        raw_artifacts = sorted(path for path in evidence_files if _is_raw_artifact(path))
        packaged_evidence = sorted(
            path for path in evidence_files if not _is_raw_artifact(path)
        )
        rows.append(
            {
                "requirement_id": spec["requirement_id"],
                "category": spec["category"],
                "original_user_intent": spec["original_user_intent"],
                "normalized_requirement": spec["normalized_requirement"],
                "status": spec["status"],
                "required_for_share": bool(spec.get("required_for_share")),
                "caveat": spec.get("caveat", ""),
                "reviewer_answer_hook": spec.get("reviewer_hook", ""),
                "source_objective_row_ids": source_objective_ids,
                "source_objective_rows": [
                    _compact_objective_row(row) for row in source_objective_rows
                ],
                "claim_ids": claim_ids,
                "metric_row_ids": metric_row_ids,
                "metric_rows": [_compact_metric_row(row) for row in metric_rows],
                "machine_evidence": evidence_files,
                "public_docs": public_docs,
                "raw_artifacts": raw_artifacts,
                "packaged_evidence": packaged_evidence,
                "rerun_command_ids": rerun_command_ids,
                "missing_metric_row_ids": missing_for_requirement,
                "support_level": (
                    "metric_rows_and_machine_gates"
                    if metric_rows
                    else "machine_gates_only"
                ),
            }
        )

    evidence_files_all = sorted(
        {
            evidence
            for row in rows
            for evidence in _as_list(row.get("machine_evidence"))
            if evidence
        }
    )
    command_refs = sorted(
        {
            command_id
            for row in rows
            for command_id in _as_list(row.get("rerun_command_ids"))
            if command_id
        }
    )
    categories = sorted({str(row.get("category")) for row in rows if row.get("category")})
    missing_evidence_files = sorted(
        evidence for evidence in evidence_files_all if not _path_exists(root, evidence)
    )
    missing_command_ids = sorted(
        command_id for command_id in command_refs if command_id not in command_ids
    )
    required_categories = {
        "performance",
        "vllm_baseline",
        "concurrency",
        "text_length",
        "stage_breakdown",
        "stage_coverage",
        "stage_connections",
        "optimization",
        "reproducibility",
        "share_confidence",
        "checkpoint",
    }
    objective_summary = _summary(objective_completion)
    checkpoint_summary = _summary(checkpoint_watchlist)
    claim_metric_summary = _summary(claim_metric_crosswalk)
    metric_provenance_summary = _summary(metric_provenance)
    repro_summary = _summary(repro_manifest)

    row_counts = {
        "requirement_rows_total": len(rows),
        "required_rows_total": sum(1 for row in rows if row.get("required_for_share")),
        "pass_rows_total": sum(1 for row in rows if row.get("status") == "PASS"),
        "pass_with_caveat_rows_total": sum(
            1 for row in rows if row.get("status") == "PASS_WITH_CAVEAT"
        ),
        "claim_refs_total": len(
            {
                claim_id
                for row in rows
                for claim_id in _as_list(row.get("claim_ids"))
            }
        ),
        "metric_row_refs_total": sum(
            len(_as_list(row.get("metric_row_ids"))) for row in rows
        ),
        "unique_metric_rows_total": len(
            {
                metric_row_id
                for row in rows
                for metric_row_id in _as_list(row.get("metric_row_ids"))
            }
        ),
        "raw_artifacts_total": len(
            {
                artifact
                for row in rows
                for artifact in _as_list(row.get("raw_artifacts"))
            }
        ),
        "packaged_evidence_files_total": len(
            {
                evidence
                for row in rows
                for evidence in _as_list(row.get("packaged_evidence"))
            }
        ),
        "command_refs_total": len(command_refs),
        "categories_total": len(categories),
    }
    optimization_candidate_ledger = _build_optimization_candidate_ledger(
        root, metric_by_id, command_ids
    )
    optimization_candidate_summary = optimization_candidate_ledger["summary"]
    checkpoint_ready = (
        int(checkpoint_summary.get("checks_total") or 0) >= 21
        and str(checkpoint_summary.get("checkpoint_phase") or "")
        == "completion_audit_ready"
        and int(checkpoint_summary.get("seconds_until_checkpoint") or 0) == 0
        and not any(
            "waiting_for_2026-06-21" in str(blocker)
            for blocker in checkpoint_summary.get("completion_blockers", []) or []
        )
    )
    checkpoint_pending_in_full_audit = (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and int(checkpoint_summary.get("checks_total") or 0) >= 21
        and str(checkpoint_summary.get("checkpoint_phase") or "")
        == "completion_audit_ready"
    )

    required_row_failures = [
        row["requirement_id"]
        for row in rows
        if row.get("required_for_share") and row.get("status") not in {"PASS", "PASS_WITH_CAVEAT"}
    ]
    checks = {
        "objective_completion_ready": bool(
            objective_summary.get("share_ready_with_documented_caveats")
        )
        and int(objective_summary.get("rows_total") or 0) >= 17
        and int(objective_summary.get("required_failures") or 0) == 0
        and not bool(objective_summary.get("goal_complete")),
        "final_checkpoint_watchlist_ready": checkpoint_ready
        or checkpoint_pending_in_full_audit,
        "claim_metric_crosswalk_ready": bool(claim_metric_summary.get("ready"))
        and int(claim_metric_summary.get("claims_total") or 0) >= 10
        and int(claim_metric_summary.get("unique_metric_rows_total") or 0) >= 60
        and int(claim_metric_summary.get("required_failures") or 0) == 0,
        "metric_provenance_ready": bool(metric_provenance_summary.get("ready"))
        and int(metric_provenance_summary.get("rows_total") or 0) >= 150
        and int(metric_provenance_summary.get("required_failures") or 0) == 0,
        "repro_command_registry_ready": bool(
            repro_summary.get("required_command_ids_present")
        )
        and int(repro_summary.get("commands_total") or 0) >= 52,
        "all_required_categories_present": required_categories.issubset(set(categories)),
        "coverage_shape": row_counts["requirement_rows_total"] >= 11
        and row_counts["unique_metric_rows_total"] >= 85
        and row_counts["raw_artifacts_total"] >= 25
        and row_counts["command_refs_total"] >= 25,
        "source_objective_rows_present": not missing_objective_row_ids,
        "claim_ids_present": not missing_claim_ids,
        "metric_row_ids_present": not missing_metric_row_ids,
        "evidence_files_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
        "required_rows_shareable": not required_row_failures,
        "optimization_candidate_ledger_ready": bool(
            optimization_candidate_summary.get("ready")
        )
        and int(optimization_candidate_summary.get("required_failures") or 0) == 0,
        "optimization_current_best_declared": bool(
            optimization_candidate_summary.get("current_best_candidate_ready")
        )
        and optimization_candidate_summary.get("current_best_candidate_id")
        == "sglang_current_best_measured_recipe",
        "optimization_anti_recipes_locked": int(
            optimization_candidate_summary.get("rejected_anti_recipe_rows_total") or 0
        )
        >= 2,
        "optimization_vllm_baseline_and_diagnostic_locked": int(
            optimization_candidate_summary.get("vllm_rows_total") or 0
        )
        >= 3
        and int(
            optimization_candidate_summary.get("vllm_diagnostic_rows_total") or 0
        )
        >= 2
        and bool(
            optimization_candidate_summary.get(
                "vllm_original_c8_prompt_feed_limited"
            )
        ),
        "optimization_candidate_evidence_present": int(
            optimization_candidate_summary.get("missing_evidence_files") or 0
        )
        == 0,
        "optimization_candidate_commands_present": int(
            optimization_candidate_summary.get("missing_command_ids") or 0
        )
        == 0,
        "optimization_not_global_optimum_boundary_present": bool(
            optimization_candidate_summary.get("not_global_optimum_boundary")
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            **row_counts,
            "optimization_candidate_ledger_ready": bool(
                optimization_candidate_summary.get("ready")
            ),
            "optimization_candidate_rows_total": int(
                optimization_candidate_summary.get("candidate_rows_total") or 0
            ),
            "optimization_rejected_anti_recipes_total": int(
                optimization_candidate_summary.get("rejected_anti_recipe_rows_total")
                or 0
            ),
            "optimization_vllm_diagnostic_rows_total": int(
                optimization_candidate_summary.get("vllm_diagnostic_rows_total") or 0
            ),
            "optimization_current_best_candidate_id": optimization_candidate_summary.get(
                "current_best_candidate_id", ""
            ),
            "optimization_not_global_optimum_boundary": bool(
                optimization_candidate_summary.get("not_global_optimum_boundary")
            ),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "goal_complete": bool(objective_summary.get("goal_complete")),
            "completion_allowed_now": bool(
                checkpoint_summary.get("completion_allowed_now")
            ),
            "share_scope": (
                "Maps each original user objective to objective-completion rows, "
                "defense claims, metric provenance row IDs, evidence files, and "
                "reproduction command IDs, plus an optimization candidate ledger."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "categories": categories,
            "missing_required_categories": sorted(required_categories - set(categories)),
            "missing_objective_row_ids": sorted(set(missing_objective_row_ids)),
            "missing_claim_ids": sorted(set(missing_claim_ids)),
            "missing_metric_row_ids": sorted(set(missing_metric_row_ids)),
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "required_row_failures": required_row_failures,
            "required_failures": required_failures,
            "checkpoint_ready": checkpoint_ready,
            "checkpoint_pending_in_full_audit": checkpoint_pending_in_full_audit,
            "optimization_candidate_ledger_required_failures": (
                optimization_candidate_ledger["diagnostics"]["required_failures"]
            ),
        },
        "source_summaries": {
            "objective_completion_audit": objective_summary,
            "final_checkpoint_watchlist": checkpoint_summary,
            "claim_metric_crosswalk": claim_metric_summary,
            "metric_provenance_index": metric_provenance_summary,
            "repro_command_manifest": repro_summary,
        },
        "optimization_candidate_ledger": optimization_candidate_ledger,
        "rows": rows,
    }


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Objective Requirement Crosswalk")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- requirement rows: `{summary['requirement_rows_total']}`")
    print(f"- unique metric rows: `{summary['unique_metric_rows_total']}`")
    print(
        "- optimization candidates: "
        f"`{summary['optimization_candidate_rows_total']}` "
        f"(`{summary['optimization_current_best_candidate_id']}`)"
    )
    print(
        "- optimization anti-recipes / vLLM diagnostics: "
        f"`{summary['optimization_rejected_anti_recipes_total']}` / "
        f"`{summary['optimization_vllm_diagnostic_rows_total']}`"
    )
    print(f"- raw artifacts: `{summary['raw_artifacts_total']}`")
    print(f"- command refs: `{summary['command_refs_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni original-objective evidence crosswalk JSON."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output
    if not output.is_absolute():
        output = root / output

    payload = build_objective_requirement_crosswalk(root)
    _save_json(payload, output)
    _print_markdown(payload)

    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
