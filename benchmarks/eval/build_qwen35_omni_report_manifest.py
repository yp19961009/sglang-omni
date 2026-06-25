# SPDX-License-Identifier: Apache-2.0
"""Build a manifest for the Qwen3.5-Omni performance-report evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval.summarize_qwen35_omni_report_artifacts import (
    STRESS_CASES,
    SYNTHETIC_CASES,
    _preproc_paths,
    _stress_paths,
    _synthetic_paths,
    _vllm_paths,
)


REPORT_FILES = [
    "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md",
    "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_videoamme_perf_report_20260619.md",
    "benchmarks/README.md",
]

SCRIPT_FILES = [
    "benchmarks/eval/benchmark_omni_videomme.py",
    "benchmarks/eval/benchmark_qwen35_speech_synthetic.py",
    "benchmarks/eval/build_videoamme_seedtts_meta.py",
    "benchmarks/eval/build_qwen35_omni_environment_snapshot.py",
    "benchmarks/eval/compute_audio_consistency_from_results.py",
    "benchmarks/eval/summarize_omni_tail_profiles.py",
    "benchmarks/eval/summarize_vllm_offline_runner_overhead.py",
    "benchmarks/eval/summarize_vllm_omni_log_stages.py",
    "benchmarks/eval/diagnose_vllm_offline_admission.py",
    "benchmarks/eval/summarize_qwen35_stage_interactions.py",
    "benchmarks/eval/build_qwen35_omni_headline_scorecard.py",
    "benchmarks/eval/build_qwen35_omni_share_charts.py",
    "benchmarks/eval/build_qwen35_omni_chart_source_consistency.py",
    "benchmarks/eval/build_qwen35_omni_acceptance_matrix.py",
    "benchmarks/eval/build_qwen35_omni_confidence_ledger.py",
    "benchmarks/eval/build_qwen35_omni_final_status_summary.py",
    "benchmarks/eval/build_qwen35_omni_regime_decision_matrix.py",
    "benchmarks/eval/build_qwen35_omni_university_review_packet.py",
    "benchmarks/eval/build_qwen35_omni_university_technical_report.py",
    "benchmarks/eval/build_qwen35_omni_runtime_comparison_contract.py",
    "benchmarks/eval/build_qwen35_omni_runtime_image_contract.py",
    "benchmarks/eval/build_qwen35_omni_rerun_acceptance_contract.py",
    "benchmarks/eval/build_qwen35_omni_rerun_time_budget.py",
    "benchmarks/eval/build_qwen35_omni_sglang_optimization_lock.py",
    "benchmarks/eval/build_qwen35_omni_vllm_optimization_lock.py",
    "benchmarks/eval/build_qwen35_omni_vllm_online_parity_protocol.py",
    "benchmarks/eval/build_qwen35_omni_final_checkpoint_watchlist.py",
    "benchmarks/eval/build_qwen35_omni_tail_confidence_appendix.py",
    "benchmarks/eval/build_qwen35_omni_stage_latency_budget.py",
    "benchmarks/eval/build_qwen35_omni_stage_boundary_bottleneck_ledger.py",
    "benchmarks/eval/build_qwen35_omni_stage_causal_graph.py",
    "benchmarks/eval/build_qwen35_omni_caveat_adjudication_matrix.py",
    "benchmarks/eval/build_qwen35_omni_objective_completion_audit.py",
    "benchmarks/eval/build_qwen35_omni_repro_command_manifest.py",
    "benchmarks/eval/build_qwen35_omni_optimization_candidate_ledger.py",
    "benchmarks/eval/build_qwen35_omni_command_reference_hygiene.py",
    "benchmarks/eval/build_qwen35_omni_final_readiness.py",
    "benchmarks/eval/build_qwen35_omni_share_bundle_manifest.py",
    "benchmarks/eval/build_qwen35_omni_share_bundle_package.py",
    "benchmarks/eval/build_qwen35_omni_share_release_seal.py",
    "benchmarks/eval/build_qwen35_omni_external_standalone_bundle_validation.py",
    "benchmarks/eval/build_qwen35_omni_receiver_quickcheck_contract.py",
    "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
    "benchmarks/eval/qwen35_omni_collaborator_return_check.py",
    "benchmarks/eval/validate_qwen35_omni_share_package.py",
    "benchmarks/eval/preflight_qwen35_omni_repro.py",
    "benchmarks/eval/summarize_qwen35_report_coverage.py",
    "benchmarks/eval/summarize_qwen35_omni_report_artifacts.py",
    "benchmarks/eval/verify_qwen35_omni_report_claims.py",
    "benchmarks/eval/build_qwen35_omni_report_manifest.py",
    "benchmarks/eval/run_qwen35_omni_report_audit.py",
]

AUDIT_FILES = [
    "results/qwen35_report_audit_20260619/tables_summary.json",
    "results/qwen35_report_audit_20260619/claims_verification.json",
    "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
    "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
    "results/qwen35_report_audit_20260619/runtime_image_contract.json",
    "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
    "results/qwen35_report_audit_20260619/rerun_time_budget.json",
    "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
    "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
    "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
    "results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json",
    "results/qwen35_report_audit_20260619/tail_confidence_appendix.json",
    "results/qwen35_report_audit_20260619/stage_latency_budget.json",
    "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
    "results/qwen35_report_audit_20260619/stage_causal_graph.json",
    "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
    "results/qwen35_report_audit_20260619/headline_scorecard.json",
    "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json",
    "results/qwen35_report_audit_20260619/share_charts",
    "results/qwen35_report_audit_20260619/chart_source_consistency.json",
    "results/qwen35_report_audit_20260619/acceptance_matrix.json",
    "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
    "results/qwen35_report_audit_20260619/university_review_packet.json",
    "results/qwen35_report_audit_20260619/university_technical_report.json",
    "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
    "results/qwen35_report_audit_20260619/confidence_ledger.json",
    "results/qwen35_report_audit_20260619/objective_completion_audit.json",
    "results/qwen35_report_audit_20260619/repro_command_manifest.json",
    "results/qwen35_report_audit_20260619/optimization_candidate_ledger.json",
    "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json",
    "results/qwen35_report_audit_20260619/command_reference_hygiene.json",
    "results/qwen35_report_audit_20260619/final_readiness_audit.json",
    "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
    "results/qwen35_report_audit_20260619/share_release_seal.json",
    "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
    "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
    "results/qwen35_report_audit_20260619/collaborator_return_check.json",
    "results/qwen35_report_audit_20260619/preflight_repro.json",
    "results/qwen35_report_audit_20260619/coverage_matrix.json",
    "results/qwen35_report_audit_20260619/environment_snapshot.json",
    "results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst",
    "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(root: Path, path: Path | str, *, kind: str) -> dict[str, Any]:
    full_path = path if isinstance(path, Path) else root / path
    full_path = full_path.resolve()
    record: dict[str, Any] = {
        "kind": kind,
        "path": str(full_path),
        "relative_path": str(full_path.relative_to(root)) if full_path.exists() else str(path),
        "exists": full_path.exists(),
    }
    if full_path.is_file():
        record.update(
            {
                "type": "file",
                "size_bytes": full_path.stat().st_size,
                "sha256": _sha256(full_path),
            }
        )
    elif full_path.is_dir():
        files = [child for child in full_path.rglob("*") if child.is_file()]
        record.update(
            {
                "type": "directory",
                "file_count": len(files),
                "size_bytes": sum(child.stat().st_size for child in files),
            }
        )
    else:
        record["type"] = "missing"
    return record


def _git_value(root: Path, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _artifact_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(_stress_paths(root))
    paths.extend(_synthetic_paths(root))
    paths.extend(_vllm_paths(root))
    paths.extend(_preproc_paths(root))

    stress_root = root / "results/qwen35_sglang_mr8_stress_20260619"
    for case in STRESS_CASES:
        if case.concurrency == 1:
            profile = stress_root / "request_profile_c1_warm_profile_skipwer.txt"
        elif case.concurrency == 2:
            profile = stress_root / "request_profile_c2_warm_profile_skipwer.txt"
        else:
            profile = stress_root / f"request_profile_c{case.concurrency}_profile_skipwer.txt"
        paths.append(profile)

    synthetic_root = root / "results/qwen35_synthetic_speech_20260619"
    for case in SYNTHETIC_CASES:
        paths.append(
            synthetic_root
            / f"request_profile_{case.scenario}_c{case.concurrency}_profile.txt"
        )

    paths.extend(
        [
            root
            / "results/qwen35_sglang_preproc2_mr8_c8_20260619/request_profile_c8_preproc2_profile_skipwer.txt",
            root
            / "results/qwen35_sglang_mr8_preproc4_stress_20260619/warmup_audio_50_c8_skipwer/videoamme_results.json",
            root
            / "results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/videoamme_results.json",
            root
            / "results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/whisper_large_v3_wer.json",
            root
            / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh",
            root
            / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py",
            root
            / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
            root
            / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/vllm_videoamme_report.md",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/vllm_videoamme_report.md",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/vllm_videoamme_report.md",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    records: list[dict[str, Any]] = []
    for rel_path in REPORT_FILES:
        records.append(_file_record(root, rel_path, kind="report"))
    for rel_path in SCRIPT_FILES:
        records.append(_file_record(root, rel_path, kind="script"))
    for rel_path in AUDIT_FILES:
        records.append(_file_record(root, rel_path, kind="audit"))
    for path in _artifact_files(root):
        records.append(_file_record(root, path, kind="artifact"))

    missing = [record for record in records if not record["exists"]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "git": {
            "head": _git_value(root, ["rev-parse", "HEAD"]),
            "status_short": _git_value(root, ["status", "--short"]),
        },
        "summary": {
            "total_records": len(records),
            "missing_records": len(missing),
            "file_records": sum(1 for record in records if record.get("type") == "file"),
            "directory_records": sum(
                1 for record in records if record.get("type") == "directory"
            ),
        },
        "commands": {
            "regenerate_tables": (
                "python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/tables_summary.json"
            ),
            "verify_claims": (
                "python3 -m benchmarks.eval.verify_qwen35_omni_report_claims "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/claims_verification.json"
            ),
            "summarize_vllm_log_stages": (
                "python3 -m benchmarks.eval.summarize_vllm_omni_log_stages "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log "
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log "
                "--labels vLLM-c1 vLLM-c4 vLLM-c8 vLLM-c8-prebuild-w1 vLLM-c8-prebuild-w4 "
                "--skip-first-requests 4 4 8 8 8 "
                "--json-output results/qwen35_report_audit_20260619/vllm_log_stage_summary.json"
            ),
            "diagnose_vllm_admission": (
                "python3 -m benchmarks.eval.diagnose_vllm_offline_admission "
                "--case vLLM-c4 "
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json "
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log 4 "
                "--case vLLM-c8 "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log 8 "
                "--case vLLM-c8-prebuild-w1 "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log 8 "
                "--case vLLM-c8-prebuild-w4 "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json "
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log 8 "
                "--json-output results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"
            ),
            "rerun_vllm_c8_prebuilt_prompts": (
                "RUN_ROOT=\"/home/gangouyu/sglang-omni/results/"
                "qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)\" "
                "MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 "
                "RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 "
                "EXTRA_ARGS=\"--prebuild-prompts --prebuild-workers 4\" "
                "bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh"
            ),
            "preflight_reproduction": (
                "python3 -m benchmarks.eval.preflight_qwen35_omni_repro "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/preflight_repro.json"
            ),
            "environment_snapshot": (
                "python3 -m benchmarks.eval.build_qwen35_omni_environment_snapshot "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/environment_snapshot.json"
            ),
            "build_videoamme_seedtts_meta": (
                "python3 -m benchmarks.eval.build_videoamme_seedtts_meta "
                "--output results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst "
                "--summary-output results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json "
                "--max-samples 50 --target-mode audio_text"
            ),
            "coverage_matrix": (
                "python3 -m benchmarks.eval.summarize_qwen35_report_coverage "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/coverage_matrix.json"
            ),
            "stage_interaction_summary": (
                "python3 -m benchmarks.eval.summarize_qwen35_stage_interactions "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json"
            ),
            "headline_scorecard": (
                "python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/headline_scorecard.json"
            ),
            "share_chart_pack": (
                "python3 -m benchmarks.eval.build_qwen35_omni_share_charts "
                "--root /home/gangouyu/sglang-omni "
                "--output-dir results/qwen35_report_audit_20260619/share_charts "
                "--manifest-output results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json"
            ),
            "chart_source_consistency": (
                "python3 -m benchmarks.eval.build_qwen35_omni_chart_source_consistency "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/chart_source_consistency.json"
            ),
            "acceptance_matrix": (
                "python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/acceptance_matrix.json"
            ),
            "confidence_ledger": (
                "python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger "
                "--root /home/gangouyu/sglang-omni "
                "--json-output results/qwen35_report_audit_20260619/confidence_ledger.json"
            ),
            "final_status_summary": (
                "python3 -m benchmarks.eval.build_qwen35_omni_final_status_summary "
                "--root /home/gangouyu/sglang-omni "
                "--output benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md"
            ),
            "regime_decision_matrix": (
                "python3 -m benchmarks.eval.build_qwen35_omni_regime_decision_matrix "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/regime_decision_matrix.json"
            ),
            "university_technical_report": (
                "python3 -m benchmarks.eval.build_qwen35_omni_university_technical_report "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/university_technical_report.json"
            ),
            "runtime_comparison_contract": (
                "python3 -m benchmarks.eval.build_qwen35_omni_runtime_comparison_contract "
                "--root /home/gangouyu/sglang-omni "
                "--output benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md"
            ),
            "rerun_acceptance_contract": (
                "python3 -m benchmarks.eval.build_qwen35_omni_rerun_acceptance_contract "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/rerun_acceptance_contract.json"
            ),
            "sglang_optimization_lock": (
                "python3 -m benchmarks.eval.build_qwen35_omni_sglang_optimization_lock "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/sglang_optimization_lock.json"
            ),
            "vllm_optimization_lock": (
                "python3 -m benchmarks.eval.build_qwen35_omni_vllm_optimization_lock "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/vllm_optimization_lock.json"
            ),
            "vllm_online_parity_protocol": (
                "python3 -m benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json"
            ),
            "stage_causal_graph": (
                "python3 -m benchmarks.eval.build_qwen35_omni_stage_causal_graph "
                "--root /home/gangouyu/sglang-omni "
                "--output benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/stage_causal_graph.json "
                "--strict"
            ),
            "tail_confidence_appendix": (
                "python3 -m benchmarks.eval.build_qwen35_omni_tail_confidence_appendix "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/tail_confidence_appendix.json"
            ),
            "stage_boundary_bottleneck_ledger": (
                "python3 -m benchmarks.eval.build_qwen35_omni_stage_boundary_bottleneck_ledger "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json"
            ),
            "caveat_adjudication_matrix": (
                "python3 -m benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix "
                "--root /home/gangouyu/sglang-omni "
                "--output benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json"
            ),
            "objective_completion_audit": (
                "python3 -m benchmarks.eval.build_qwen35_omni_objective_completion_audit "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--json-output results/qwen35_report_audit_20260619/objective_completion_audit.json"
            ),
            "repro_command_manifest": (
                "python3 -m benchmarks.eval.build_qwen35_omni_repro_command_manifest "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--json-output results/qwen35_report_audit_20260619/repro_command_manifest.json"
            ),
            "final_readiness_audit": (
                "python3 -m benchmarks.eval.build_qwen35_omni_final_readiness "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--json-output results/qwen35_report_audit_20260619/final_readiness_audit.json"
            ),
            "share_bundle_manifest": (
                "python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_manifest "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--json-output results/qwen35_report_audit_20260619/share_bundle_manifest.json"
            ),
            "share_bundle_package": (
                "python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_package "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--source-manifest results/qwen35_report_audit_20260619/share_bundle_manifest.json "
                "--output results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz "
                "--json-output results/qwen35_report_audit_20260619/share_bundle_package_manifest.json"
            ),
            "share_release_seal": (
                "python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal "
                "--root /home/gangouyu/sglang-omni "
                "--strict "
                "--output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md "
                "--json-output results/qwen35_report_audit_20260619/share_release_seal.json"
            ),
            "run_full_audit": (
                "python3 -m benchmarks.eval.run_qwen35_omni_report_audit "
                "--root /home/gangouyu/sglang-omni "
                "--summary-output results/qwen35_report_audit_20260619/audit_run_summary.json"
            ),
            "build_manifest": (
                "python3 -m benchmarks.eval.build_qwen35_omni_report_manifest "
                "--root /home/gangouyu/sglang-omni "
                "--output results/qwen35_report_audit_20260619/manifest.json"
            ),
        },
        "records": records,
    }


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni report evidence manifest with file hashes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing reports/, benchmarks/, and results/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen35_report_audit_20260619/manifest.json"),
        help="Manifest JSON output path.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = build_manifest(root)
    output = args.output
    if not output.is_absolute():
        output = root / output
    _save_json(manifest, output)

    summary = manifest["summary"]
    print(
        "Manifest written: "
        f"{output} records={summary['total_records']} "
        f"missing={summary['missing_records']} "
        f"files={summary['file_records']} dirs={summary['directory_records']}"
    )
    if summary["missing_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
