# SPDX-License-Identifier: Apache-2.0
"""Run the complete Qwen3.5-Omni performance-report audit pipeline."""

from __future__ import annotations

import argparse
import shlex
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")


def _run_step(label: str, cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    print(f"\n## {label}\n")
    print(" ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    return {
        "label": label,
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _resolve_summary_output(root: Path, output: Path | None) -> Path | None:
    if output is None:
        return None
    if output.is_absolute():
        return output
    return root / output


def _save_in_progress_summary(root: Path, output: Path | None, steps: list[dict[str, Any]]) -> None:
    resolved = _resolve_summary_output(root, output)
    if resolved is None:
        return
    _save_json(
        {
            "steps": steps,
            "ok": all(step["ok"] for step in steps),
            "in_progress": True,
        },
        resolved,
    )


def _stable_share_bundle_package_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    return {
        "ready": summary.get("ready"),
        "file_count": summary.get("file_count"),
        "source_manifest_records": summary.get("source_manifest_records"),
        "checks": summary.get("checks", {}),
    }


def _check_evidence(checks: Any, name: str) -> str | None:
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            evidence = check.get("evidence")
            return evidence if isinstance(evidence, str) else None
    return None


def _stable_share_package_validation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("summary", {}))
    summary.pop("tarball_sha256", None)
    tarball_asset_quality = _check_evidence(
        payload.get("checks"), "tarball contains quick-read and stage-budget assets"
    )
    if tarball_asset_quality:
        summary["tarball_asset_quality_evidence"] = tarball_asset_quality
    extracted_asset_quality = _check_evidence(
        payload.get("checks"), "extracted bundle contains quick-read and stage-budget assets"
    )
    if extracted_asset_quality:
        summary["extracted_asset_quality_evidence"] = extracted_asset_quality
    receiver_smoke = payload.get("receiver_smoke", {})
    if isinstance(receiver_smoke, dict):
        extracted_asset_quality = _check_evidence(
            receiver_smoke.get("extracted_validation_checks"),
            "extracted bundle contains quick-read and stage-budget assets",
        )
        if extracted_asset_quality:
            summary["receiver_smoke_extracted_asset_quality_evidence"] = (
                extracted_asset_quality
            )
    return summary


def _stable_share_release_seal_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("summary", {}))


def _stable_evidence_query_smoke_summary(payload: dict[str, Any]) -> dict[str, Any]:
    host = payload.get("host", {})
    portable = payload.get("portable", {})
    return {
        "ready": payload.get("ready"),
        "host_pass_lines": host.get("pass_lines") if isinstance(host, dict) else None,
        "host_query_output_lines": (
            host.get("query_output_lines") if isinstance(host, dict) else None
        ),
        "portable_pass_lines": (
            portable.get("pass_lines") if isinstance(portable, dict) else None
        ),
        "portable_query_output_lines": (
            portable.get("query_output_lines")
            if isinstance(portable, dict)
            else None
        ),
        "portable_note": portable.get("note") if isinstance(portable, dict) else None,
    }


def _paths(root: Path) -> dict[str, Path]:
    audit_dir = root / AUDIT_DIR
    return {
        "tables": audit_dir / "tables_summary.json",
        "claims": audit_dir / "claims_verification.json",
        "vllm_stages": audit_dir / "vllm_log_stage_summary.json",
        "vllm_admission": audit_dir / "vllm_admission_diagnosis.json",
        "stage_interactions": audit_dir / "stage_interaction_summary.json",
        "headline_scorecard": audit_dir / "headline_scorecard.json",
        "chart_pack_manifest": audit_dir / "share_charts/chart_pack_manifest.json",
        "chart_source_consistency_report": root
        / "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md",
        "chart_source_consistency": audit_dir / "chart_source_consistency.json",
        "slide_asset_map_report": root
        / "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md",
        "slide_asset_map": audit_dir / "slide_asset_map.json",
        "acceptance_matrix": audit_dir / "acceptance_matrix.json",
        "confidence_ledger": audit_dir / "confidence_ledger.json",
        "objective_completion": audit_dir / "objective_completion_audit.json",
        "final_status_summary": root
        / "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
        "regime_decision_matrix": root
        / "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
        "regime_decision_matrix_json": audit_dir / "regime_decision_matrix.json",
        "university_review_packet": root
        / "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md",
        "university_review_packet_json": audit_dir
        / "university_review_packet.json",
        "university_technical_report": root
        / "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
        "university_technical_report_json": audit_dir
        / "university_technical_report.json",
        "runtime_comparison_contract": root
        / "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
        "runtime_comparison_contract_json": audit_dir
        / "runtime_comparison_contract.json",
        "runtime_image_contract_report": root
        / "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
        "runtime_image_contract": audit_dir / "runtime_image_contract.json",
        "rerun_acceptance_contract_report": root
        / "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
        "rerun_acceptance_contract": audit_dir / "rerun_acceptance_contract.json",
        "rerun_delta_triage_report": root
        / "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
        "rerun_delta_triage": audit_dir / "rerun_delta_triage.json",
        "sglang_optimization_lock_report": root
        / "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
        "sglang_optimization_lock": audit_dir / "sglang_optimization_lock.json",
        "vllm_optimization_lock_report": root
        / "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
        "vllm_optimization_lock": audit_dir / "vllm_optimization_lock.json",
        "vllm_online_parity_protocol_report": root
        / "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
        "vllm_online_parity_protocol": audit_dir / "vllm_online_parity_protocol.json",
        "final_checkpoint_watchlist_report": root
        / "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
        "final_checkpoint_watchlist": audit_dir / "final_checkpoint_watchlist.json",
        "final_completion_audit_report": root
        / "benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md",
        "final_completion_audit": audit_dir / "final_completion_audit.json",
        "tail_confidence_appendix_report": root
        / "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
        "tail_confidence_appendix": audit_dir / "tail_confidence_appendix.json",
        "stage_latency_budget_report": root
        / "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
        "stage_latency_budget": audit_dir / "stage_latency_budget.json",
        "stage_boundary_ledger_report": root
        / "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
        "stage_boundary_ledger": audit_dir / "stage_boundary_bottleneck_ledger.json",
        "serving_capacity_matrix_report": root
        / "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md",
        "serving_capacity_matrix": audit_dir / "serving_capacity_matrix.json",
        "stage_drilldown_index": audit_dir / "stage_drilldown_index.json",
        "metric_provenance_index": audit_dir / "metric_provenance_index.json",
        "stage_reproduction_drilldown_report": root
        / "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
        "stage_reproduction_drilldown": audit_dir / "stage_reproduction_drilldown.json",
        "stage_route_decision_matrix_report": root
        / "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
        "stage_route_decision_matrix": audit_dir / "stage_route_decision_matrix.json",
        "pressure_stage_heatmap_report": root
        / "benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md",
        "pressure_stage_heatmap": audit_dir / "pressure_stage_heatmap.json",
        "length_regime_coverage_report": root
        / "benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md",
        "length_regime_coverage": audit_dir / "length_regime_coverage.json",
        "rerun_time_budget_report": root
        / "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md",
        "rerun_time_budget": audit_dir / "rerun_time_budget.json",
        "stage_causal_graph": root
        / "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
        "stage_causal_graph_json": audit_dir / "stage_causal_graph.json",
        "caveat_adjudication_matrix": root
        / "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
        "caveat_adjudication_matrix_json": audit_dir
        / "caveat_adjudication_matrix.json",
        "repro_command_manifest": audit_dir / "repro_command_manifest.json",
        "defense_claim_matrix": audit_dir / "defense_claim_matrix.json",
        "claim_metric_crosswalk": audit_dir / "claim_metric_crosswalk.json",
        "objective_requirement_crosswalk": audit_dir
        / "objective_requirement_crosswalk.json",
        "optimization_candidate_ledger_report": root
        / "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
        "optimization_candidate_ledger": audit_dir
        / "optimization_candidate_ledger.json",
        "final_readiness": audit_dir / "final_readiness_audit.json",
        "share_path_hygiene": audit_dir / "share_path_hygiene.json",
        "command_reference_hygiene": audit_dir / "command_reference_hygiene.json",
        "share_consistency_guard_report": root
        / "benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md",
        "share_consistency_guard": audit_dir / "share_consistency_guard.json",
        "share_bundle_manifest": audit_dir / "share_bundle_manifest.json",
        "share_bundle_package": audit_dir / "share_bundle_package_manifest.json",
        "share_package_validation": audit_dir / "share_package_validation.json",
        "share_package_receiver_smoke_validation": audit_dir
        / "share_package_receiver_smoke_validation.json",
        "share_package_validation_extracted": audit_dir
        / "share_package_validation_extracted.json",
        "share_package_external_standalone_validation": audit_dir
        / "share_package_external_standalone_validation.json",
        "evidence_query_smoke_summary": audit_dir
        / "evidence_query_cards_smoke_summary.json",
        "evidence_query_smoke_host_summary": audit_dir
        / "evidence_query_cards_smoke_host.summary.out",
        "evidence_query_smoke_host_query": audit_dir
        / "evidence_query_cards_smoke_host.query.out",
        "evidence_query_smoke_portable_summary": audit_dir
        / "evidence_query_cards_smoke_portable.summary.out",
        "evidence_query_smoke_portable_query": audit_dir
        / "evidence_query_cards_smoke_portable.query.out",
        "receiver_quickcheck_contract_report": root
        / "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
        "receiver_quickcheck_contract": audit_dir
        / "receiver_quickcheck_contract.json",
        "collaborator_return_check": audit_dir / "collaborator_return_check.json",
        "share_release_seal_report": root
        / "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
        "share_release_seal": audit_dir / "share_release_seal.json",
        "preflight": audit_dir / "preflight_repro.json",
        "preflight_alias": audit_dir / "preflight.json",
        "coverage": audit_dir / "coverage_matrix.json",
        "manifest": audit_dir / "manifest.json",
        "environment_snapshot": audit_dir / "environment_snapshot.json",
        "videoamme_seedtts_meta": audit_dir / "videoamme_seedtts_meta.lst",
        "videoamme_seedtts_summary": audit_dir / "videoamme_seedtts_meta_summary.json",
    }


def _evidence_query_smoke_refresh_cmd(
    root: Path,
    p: dict[str, Path],
    python: str,
    *,
    work_dir: Path,
) -> list[str]:
    tarball = root / AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"
    bundle_root = work_dir / "qwen35_omni_share_bundle_20260621"
    script = root / "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh"
    summary_output = p["evidence_query_smoke_summary"]
    host_summary = p["evidence_query_smoke_host_summary"]
    host_query = p["evidence_query_smoke_host_query"]
    portable_summary = p["evidence_query_smoke_portable_summary"]
    portable_query = p["evidence_query_smoke_portable_query"]
    lines = [
        "set -euo pipefail",
        f"rm -rf {shlex.quote(str(work_dir))}",
        f"mkdir -p {shlex.quote(str(work_dir))}",
        (
            f"bash {shlex.quote(str(script))} "
            f"--root {shlex.quote(str(root))} --mode host "
            f"--output {shlex.quote(str(host_query))} "
            f"> {shlex.quote(str(host_summary))}"
        ),
        (
            f"tar -xzf {shlex.quote(str(tarball))} "
            f"-C {shlex.quote(str(work_dir))}"
        ),
        (
            f"bash {shlex.quote(str(bundle_root / 'benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh'))} "
            f"--root {shlex.quote(str(bundle_root))} --mode portable "
            f"--output {shlex.quote(str(portable_query))} "
            f"> {shlex.quote(str(portable_summary))}"
        ),
        (
            f"{shlex.quote(python)} - "
            f"{shlex.quote(str(tarball))} "
            f"{shlex.quote(str(summary_output))} "
            f"{shlex.quote(str(host_summary))} "
            f"{shlex.quote(str(host_query))} "
            f"{shlex.quote(str(portable_summary))} "
            f"{shlex.quote(str(portable_query))} <<'PY'"
        ),
        "import hashlib, json, sys",
        "from datetime import datetime, timezone",
        "from pathlib import Path",
        "tarball, summary_output, host_summary, host_query, portable_summary, portable_query = map(Path, sys.argv[1:])",
        "def sha256(path):",
        "    digest = hashlib.sha256()",
        "    with path.open('rb') as fp:",
        "        for chunk in iter(lambda: fp.read(1024 * 1024), b''):",
        "            digest.update(chunk)",
        "    return digest.hexdigest()",
        "def line_count(path):",
        "    return sum(1 for _ in path.open(encoding='utf-8'))",
        "def pass_lines(path):",
        "    return sum(1 for line in path.open(encoding='utf-8') if line.startswith('PASS '))",
        "payload = {",
        "    'generated_at_utc': datetime.now(timezone.utc).isoformat(),",
        "    'ready': True,",
        "    'tarball_sha256': sha256(tarball),",
        "    'host': {",
        "        'mode': 'host',",
        "        'summary_output': str(host_summary),",
        "        'query_output': str(host_query),",
        "        'pass_lines': pass_lines(host_summary),",
        "        'query_output_lines': line_count(host_query),",
        "    },",
        "    'portable': {",
        "        'mode': 'portable',",
        "        'summary_output': str(portable_summary),",
        "        'query_output': str(portable_query),",
        "        'pass_lines': pass_lines(portable_summary),",
        "        'query_output_lines': line_count(portable_query),",
        "        'note': 'portable mode skips package-validation card 10 because adjacent validation JSONs are not bundled',",
        "    },",
        "}",
        "payload['ready'] = (payload['host']['pass_lines'] >= 20 and payload['portable']['pass_lines'] >= 18 and payload['host']['query_output_lines'] > 0 and payload['portable']['query_output_lines'] > 0)",
        "summary_output.parent.mkdir(parents=True, exist_ok=True)",
        "summary_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8')",
        "print(f\"evidence query smoke summary written: {summary_output} ready={payload['ready']} host_pass={payload['host']['pass_lines']} portable_pass={payload['portable']['pass_lines']}\")",
        "raise SystemExit(0 if payload['ready'] else 1)",
        "PY",
    ]
    return ["bash", "-lc", "\n".join(lines)]


def build_commands(root: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    p = _paths(root)
    extracted_tmp = Path("/tmp/qwen35_omni_share_bundle_audit")
    extracted_root = extracted_tmp / "qwen35_omni_share_bundle_20260621"
    return [
        (
            "Video-AMME SeedTTS-compatible meta",
            [
                python,
                "-m",
                "benchmarks.eval.build_videoamme_seedtts_meta",
                "--output",
                str(p["videoamme_seedtts_meta"].relative_to(root)),
                "--summary-output",
                str(p["videoamme_seedtts_summary"].relative_to(root)),
                "--max-samples",
                "50",
                "--target-mode",
                "audio_text",
            ],
        ),
        (
            "environment snapshot",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_environment_snapshot",
                "--root",
                str(root),
                "--json-output",
                str(p["environment_snapshot"].relative_to(root)),
            ],
        ),
        (
            "vLLM log-stage summary",
            [
                python,
                "-m",
                "benchmarks.eval.summarize_vllm_omni_log_stages",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log",
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
                "--labels",
                "vLLM-c1",
                "vLLM-c4",
                "vLLM-c8",
                "vLLM-c8-prebuild-w1",
                "vLLM-c8-prebuild-w4",
                "--skip-first-requests",
                "4",
                "4",
                "8",
                "8",
                "8",
                "--json-output",
                str(p["vllm_stages"].relative_to(root)),
            ],
        ),
        (
            "vLLM admission diagnosis",
            [
                python,
                "-m",
                "benchmarks.eval.diagnose_vllm_offline_admission",
                "--case",
                "vLLM-c4",
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
                "4",
                "--case",
                "vLLM-c8",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
                "8",
                "--case",
                "vLLM-c8-prebuild-w1",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
                "8",
                "--case",
                "vLLM-c8-prebuild-w4",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
                "8",
                "--json-output",
                str(p["vllm_admission"].relative_to(root)),
            ],
        ),
        (
            "artifact/table summary",
            [
                python,
                "-m",
                "benchmarks.eval.summarize_qwen35_omni_report_artifacts",
                "--root",
                str(root),
                "--check-only",
                "--json-output",
                str(p["tables"].relative_to(root)),
            ],
        ),
        (
            "claim verifier",
            [
                python,
                "-m",
                "benchmarks.eval.verify_qwen35_omni_report_claims",
                "--root",
                str(root),
                "--json-output",
                str(p["claims"].relative_to(root)),
            ],
        ),
        (
            "stage interaction summary",
            [
                python,
                "-m",
                "benchmarks.eval.summarize_qwen35_stage_interactions",
                "--root",
                str(root),
                "--json-output",
                str(p["stage_interactions"].relative_to(root)),
            ],
        ),
        (
            "stage latency budget bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_latency_budget",
                "--root",
                str(root),
                "--output",
                str(p["stage_latency_budget_report"].relative_to(root)),
                "--json-output",
                str(p["stage_latency_budget"].relative_to(root)),
            ],
        ),
        (
            "tail confidence appendix bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_tail_confidence_appendix",
                "--root",
                str(root),
                "--output",
                str(p["tail_confidence_appendix_report"].relative_to(root)),
                "--json-output",
                str(p["tail_confidence_appendix"].relative_to(root)),
            ],
        ),
        (
            "stage boundary bottleneck ledger bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_boundary_bottleneck_ledger",
                "--root",
                str(root),
                "--output",
                str(p["stage_boundary_ledger_report"].relative_to(root)),
                "--json-output",
                str(p["stage_boundary_ledger"].relative_to(root)),
            ],
        ),
        (
            "serving capacity matrix bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_serving_capacity_matrix",
                "--root",
                str(root),
                "--output",
                str(p["serving_capacity_matrix_report"].relative_to(root)),
                "--json-output",
                str(p["serving_capacity_matrix"].relative_to(root)),
            ],
        ),
        (
            "headline scorecard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_headline_scorecard",
                "--root",
                str(root),
                "--json-output",
                str(p["headline_scorecard"].relative_to(root)),
            ],
        ),
        (
            "share chart pack",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_charts",
                "--root",
                str(root),
                "--output-dir",
                "results/qwen35_report_audit_20260619/share_charts",
                "--manifest-output",
                str(p["chart_pack_manifest"].relative_to(root)),
            ],
        ),
        (
            "chart source consistency",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_chart_source_consistency",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["chart_source_consistency_report"].relative_to(root)),
                "--json-output",
                str(p["chart_source_consistency"].relative_to(root)),
            ],
        ),
        (
            "length regime coverage bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_length_regime_coverage",
                "--root",
                str(root),
                "--output",
                str(p["length_regime_coverage_report"].relative_to(root)),
                "--json-output",
                str(p["length_regime_coverage"].relative_to(root)),
            ],
        ),
        (
            "slide asset map bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_slide_asset_map",
                "--root",
                str(root),
                "--output",
                str(p["slide_asset_map_report"].relative_to(root)),
                "--json-output",
                str(p["slide_asset_map"].relative_to(root)),
            ],
        ),
        (
            "acceptance matrix",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_acceptance_matrix",
                "--root",
                str(root),
                "--json-output",
                str(p["acceptance_matrix"].relative_to(root)),
            ],
        ),
        (
            "confidence ledger",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_confidence_ledger",
                "--root",
                str(root),
                "--json-output",
                str(p["confidence_ledger"].relative_to(root)),
            ],
        ),
        (
            "SGLang optimization lock early bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_sglang_optimization_lock",
                "--root",
                str(root),
                "--output",
                str(p["sglang_optimization_lock_report"].relative_to(root)),
                "--json-output",
                str(p["sglang_optimization_lock"].relative_to(root)),
            ],
        ),
        (
            "vLLM optimization lock early bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_vllm_optimization_lock",
                "--root",
                str(root),
                "--output",
                str(p["vllm_optimization_lock_report"].relative_to(root)),
                "--json-output",
                str(p["vllm_optimization_lock"].relative_to(root)),
            ],
        ),
        (
            "vLLM online parity protocol early bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol",
                "--root",
                str(root),
                "--output",
                str(p["vllm_online_parity_protocol_report"].relative_to(root)),
                "--json-output",
                str(p["vllm_online_parity_protocol"].relative_to(root)),
            ],
        ),
        (
            "runtime image contract early bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_image_contract",
                "--root",
                str(root),
                "--output",
                str(p["runtime_image_contract_report"].relative_to(root)),
                "--json-output",
                str(p["runtime_image_contract"].relative_to(root)),
            ],
        ),
        (
            "share path hygiene early bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_path_hygiene",
                "--root",
                str(root),
                "--json-output",
                str(p["share_path_hygiene"].relative_to(root)),
            ],
        ),
        (
            "share bundle manifest early bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "reproduction command manifest before preflight",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "runtime image contract after command bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_image_contract",
                "--root",
                str(root),
                "--output",
                str(p["runtime_image_contract_report"].relative_to(root)),
                "--json-output",
                str(p["runtime_image_contract"].relative_to(root)),
            ],
        ),
        (
            "objective completion audit bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_objective_completion_audit",
                "--root",
                str(root),
                "--json-output",
                str(p["objective_completion"].relative_to(root)),
            ],
        ),
        (
            "reproduction command manifest after objective bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "stage drilldown index bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_drilldown_index",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["stage_drilldown_index"].relative_to(root)),
            ],
        ),
        (
            "metric provenance index bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_metric_provenance_index",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["metric_provenance_index"].relative_to(root)),
            ],
        ),
        (
            "stage reproduction drilldown bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_reproduction_drilldown",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_reproduction_drilldown_report"].relative_to(root)),
                "--json-output",
                str(p["stage_reproduction_drilldown"].relative_to(root)),
            ],
        ),
        (
            "stage route decision matrix bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_route_decision_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_route_decision_matrix_report"].relative_to(root)),
                "--json-output",
                str(p["stage_route_decision_matrix"].relative_to(root)),
                "--heatmap-output",
                str(p["pressure_stage_heatmap_report"].relative_to(root)),
                "--heatmap-json-output",
                str(p["pressure_stage_heatmap"].relative_to(root)),
            ],
        ),
        (
            "defense claim matrix bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_defense_claim_matrix",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["defense_claim_matrix"].relative_to(root)),
            ],
        ),
        (
            "claim metric crosswalk bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_claim_metric_crosswalk",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["claim_metric_crosswalk"].relative_to(root)),
            ],
        ),
        (
            "rerun delta triage bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_delta_triage",
                "--root",
                str(root),
                "--output",
                str(p["rerun_delta_triage_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_delta_triage"].relative_to(root)),
            ],
        ),
        (
            "rerun acceptance contract bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_acceptance_contract",
                "--root",
                str(root),
                "--output",
                str(p["rerun_acceptance_contract_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_acceptance_contract"].relative_to(root)),
            ],
        ),
        (
            "final status summary bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_status_summary",
                "--root",
                str(root),
                "--output",
                str(p["final_status_summary"].relative_to(root)),
            ],
        ),
        (
            "regime decision matrix bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_regime_decision_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["regime_decision_matrix"].relative_to(root)),
                "--json-output",
                str(p["regime_decision_matrix_json"].relative_to(root)),
            ],
        ),
        (
            "university technical report bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_technical_report",
                "--root",
                str(root),
                "--output",
                str(p["university_technical_report"].relative_to(root)),
                "--json-output",
                str(p["university_technical_report_json"].relative_to(root)),
            ],
        ),
        (
            "university review packet bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_review_packet",
                "--root",
                str(root),
                "--output",
                str(p["university_review_packet"].relative_to(root)),
                "--json-output",
                str(p["university_review_packet_json"].relative_to(root)),
            ],
        ),
        (
            "runtime comparison contract bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_comparison_contract",
                "--root",
                str(root),
                "--output",
                str(p["runtime_comparison_contract"].relative_to(root)),
                "--json-output",
                str(p["runtime_comparison_contract_json"].relative_to(root)),
            ],
        ),
        (
            "SGLang optimization lock bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_sglang_optimization_lock",
                "--root",
                str(root),
                "--output",
                str(p["sglang_optimization_lock_report"].relative_to(root)),
                "--json-output",
                str(p["sglang_optimization_lock"].relative_to(root)),
            ],
        ),
        (
            "vLLM optimization lock bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_vllm_optimization_lock",
                "--root",
                str(root),
                "--output",
                str(p["vllm_optimization_lock_report"].relative_to(root)),
                "--json-output",
                str(p["vllm_optimization_lock"].relative_to(root)),
            ],
        ),
        (
            "vLLM online parity protocol bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol",
                "--root",
                str(root),
                "--output",
                str(p["vllm_online_parity_protocol_report"].relative_to(root)),
                "--json-output",
                str(p["vllm_online_parity_protocol"].relative_to(root)),
            ],
        ),
        (
            "final checkpoint watchlist bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist",
                "--root",
                str(root),
                "--output",
                str(p["final_checkpoint_watchlist_report"].relative_to(root)),
                "--json-output",
                str(p["final_checkpoint_watchlist"].relative_to(root)),
            ],
        ),
        (
            "objective requirement crosswalk bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk",
                "--root",
                str(root),
                "--json-output",
                str(p["objective_requirement_crosswalk"].relative_to(root)),
            ],
        ),
        (
            "optimization candidate ledger bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger",
                "--root",
                str(root),
                "--output",
                str(p["optimization_candidate_ledger_report"].relative_to(root)),
                "--json-output",
                str(p["optimization_candidate_ledger"].relative_to(root)),
            ],
        ),
        (
            "command reference hygiene bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_command_reference_hygiene",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["command_reference_hygiene"].relative_to(root)),
            ],
        ),
        (
            "reproduction command manifest after command hygiene bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "rerun time budget bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_time_budget",
                "--root",
                str(root),
                "--output",
                str(p["rerun_time_budget_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_time_budget"].relative_to(root)),
            ],
        ),
        (
            "final readiness bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "final status summary after readiness bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_status_summary",
                "--root",
                str(root),
                "--output",
                str(p["final_status_summary"].relative_to(root)),
            ],
        ),
        (
            "share bundle manifest bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "stage causal graph bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_causal_graph",
                "--root",
                str(root),
                "--output",
                str(p["stage_causal_graph"].relative_to(root)),
                "--json-output",
                str(p["stage_causal_graph_json"].relative_to(root)),
                "--strict",
            ],
        ),
        (
            "caveat adjudication matrix bootstrap",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix",
                "--root",
                str(root),
                "--output",
                str(p["caveat_adjudication_matrix"].relative_to(root)),
                "--json-output",
                str(p["caveat_adjudication_matrix_json"].relative_to(root)),
            ],
        ),
        (
            "manifest before preflight",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final checkpoint watchlist before preflight",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist",
                "--root",
                str(root),
                "--output",
                str(p["final_checkpoint_watchlist_report"].relative_to(root)),
                "--json-output",
                str(p["final_checkpoint_watchlist"].relative_to(root)),
            ],
        ),
        (
            "runtime image contract before preflight",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_image_contract",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["runtime_image_contract_report"].relative_to(root)),
                "--json-output",
                str(p["runtime_image_contract"].relative_to(root)),
            ],
        ),
        (
            "rerun acceptance contract before preflight",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_acceptance_contract",
                "--root",
                str(root),
                "--output",
                str(p["rerun_acceptance_contract_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_acceptance_contract"].relative_to(root)),
            ],
        ),
        (
            "share path hygiene before preflight strict",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_path_hygiene",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_path_hygiene"].relative_to(root)),
            ],
        ),
        (
            "command reference hygiene before preflight strict",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_command_reference_hygiene",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["command_reference_hygiene"].relative_to(root)),
            ],
        ),
        (
            "share consistency guard pre-refresh before repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_consistency_guard",
                "--root",
                str(root),
                "--output",
                str(p["share_consistency_guard_report"].relative_to(root)),
                "--json-output",
                str(p["share_consistency_guard"].relative_to(root)),
            ],
        ),
        (
            "reproduction command manifest before preflight guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "university technical report before preflight guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_technical_report",
                "--root",
                str(root),
                "--output",
                str(p["university_technical_report"].relative_to(root)),
                "--json-output",
                str(p["university_technical_report_json"].relative_to(root)),
            ],
        ),
        (
            "runtime comparison contract before preflight guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_comparison_contract",
                "--root",
                str(root),
                "--output",
                str(p["runtime_comparison_contract"].relative_to(root)),
                "--json-output",
                str(p["runtime_comparison_contract_json"].relative_to(root)),
            ],
        ),
        (
            "final readiness before preflight guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "university review packet before preflight guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_review_packet",
                "--root",
                str(root),
                "--output",
                str(p["university_review_packet"].relative_to(root)),
                "--json-output",
                str(p["university_review_packet_json"].relative_to(root)),
            ],
        ),
        (
            "share consistency guard before preflight strict",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_consistency_guard",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["share_consistency_guard_report"].relative_to(root)),
                "--json-output",
                str(p["share_consistency_guard"].relative_to(root)),
            ],
        ),
        (
            "reproduction command manifest after preflight guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "reproduction preflight",
            [
                python,
                "-m",
                "benchmarks.eval.preflight_qwen35_omni_repro",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["preflight"].relative_to(root)),
            ],
        ),
        (
            "sync preflight alias",
            [
                python,
                "-c",
                "from pathlib import Path; Path(__import__('sys').argv[2]).write_bytes(Path(__import__('sys').argv[1]).read_bytes())",
                str(p["preflight"].relative_to(root)),
                str(p["preflight_alias"].relative_to(root)),
            ],
        ),
        (
            "requirement coverage matrix",
            [
                python,
                "-m",
                "benchmarks.eval.summarize_qwen35_report_coverage",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["coverage"].relative_to(root)),
            ],
        ),
        (
            "final environment snapshot",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_environment_snapshot",
                "--root",
                str(root),
                "--json-output",
                str(p["environment_snapshot"].relative_to(root)),
            ],
        ),
        (
            "manifest after final environment",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final objective completion audit",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_objective_completion_audit",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["objective_completion"].relative_to(root)),
            ],
        ),
        (
            "manifest after final objective audit",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final stage drilldown index",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_drilldown_index",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["stage_drilldown_index"].relative_to(root)),
            ],
        ),
        (
            "final metric provenance index",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_metric_provenance_index",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["metric_provenance_index"].relative_to(root)),
            ],
        ),
        (
            "final defense claim matrix",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_defense_claim_matrix",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["defense_claim_matrix"].relative_to(root)),
            ],
        ),
        (
            "final claim metric crosswalk",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_claim_metric_crosswalk",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["claim_metric_crosswalk"].relative_to(root)),
            ],
        ),
        (
            "final objective requirement crosswalk",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["objective_requirement_crosswalk"].relative_to(root)),
            ],
        ),
        (
            "final optimization candidate ledger",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["optimization_candidate_ledger_report"].relative_to(root)),
                "--json-output",
                str(p["optimization_candidate_ledger"].relative_to(root)),
            ],
        ),
        (
            "final reproduction command manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "final metric provenance index after repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_metric_provenance_index",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["metric_provenance_index"].relative_to(root)),
            ],
        ),
        (
            "final stage reproduction drilldown after repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_reproduction_drilldown",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_reproduction_drilldown_report"].relative_to(root)),
                "--json-output",
                str(p["stage_reproduction_drilldown"].relative_to(root)),
            ],
        ),
        (
            "final stage route decision matrix after repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_route_decision_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_route_decision_matrix_report"].relative_to(root)),
                "--json-output",
                str(p["stage_route_decision_matrix"].relative_to(root)),
                "--heatmap-output",
                str(p["pressure_stage_heatmap_report"].relative_to(root)),
                "--heatmap-json-output",
                str(p["pressure_stage_heatmap"].relative_to(root)),
            ],
        ),
        (
            "final claim metric crosswalk after repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_claim_metric_crosswalk",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["claim_metric_crosswalk"].relative_to(root)),
            ],
        ),
        (
            "final objective requirement crosswalk after repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["objective_requirement_crosswalk"].relative_to(root)),
            ],
        ),
        (
            "final optimization candidate ledger after repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["optimization_candidate_ledger_report"].relative_to(root)),
                "--json-output",
                str(p["optimization_candidate_ledger"].relative_to(root)),
            ],
        ),
        (
            "final reproduction command manifest after crosswalk refresh",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "final command reference hygiene after crosswalk refresh",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_command_reference_hygiene",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["command_reference_hygiene"].relative_to(root)),
            ],
        ),
        (
            "final share consistency guard after crosswalk refresh",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_consistency_guard",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["share_consistency_guard_report"].relative_to(root)),
                "--json-output",
                str(p["share_consistency_guard"].relative_to(root)),
            ],
        ),
        (
            "final reproduction command manifest after command hygiene",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "receiver quickcheck contract before final readiness",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_receiver_quickcheck_contract",
                "--root",
                str(root),
                "--output",
                str(p["receiver_quickcheck_contract_report"].relative_to(root)),
                "--json-output",
                str(p["receiver_quickcheck_contract"].relative_to(root)),
            ],
        ),
        (
            "share bundle manifest before final readiness",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "final readiness audit",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "final share bundle manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "final manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final share bundle package",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_package",
                "--root",
                str(root),
                "--strict",
                "--source-manifest",
                str(p["share_bundle_manifest"].relative_to(root)),
                "--output",
                "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
                "--json-output",
                str(p["share_bundle_package"].relative_to(root)),
            ],
        ),
        (
            "standalone share package validation before final status",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation",
                "--root",
                str(root),
                "--strict",
                "--work-dir",
                "/tmp/qwen35_omni_external_standalone_bundle_validation_bootstrap",
                "--json-output",
                str(p["share_package_external_standalone_validation"].relative_to(root)),
            ],
        ),
        (
            "final status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_status_summary",
                "--root",
                str(root),
                "--output",
                str(p["final_status_summary"].relative_to(root)),
            ],
        ),
        (
            "final regime decision matrix",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_regime_decision_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["regime_decision_matrix"].relative_to(root)),
                "--json-output",
                str(p["regime_decision_matrix_json"].relative_to(root)),
            ],
        ),
        (
            "final university technical report",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_technical_report",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["university_technical_report"].relative_to(root)),
                "--json-output",
                str(p["university_technical_report_json"].relative_to(root)),
            ],
        ),
        (
            "final runtime comparison contract",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_comparison_contract",
                "--root",
                str(root),
                "--output",
                str(p["runtime_comparison_contract"].relative_to(root)),
                "--json-output",
                str(p["runtime_comparison_contract_json"].relative_to(root)),
                "--strict",
            ],
        ),
        (
            "final SGLang optimization lock",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_sglang_optimization_lock",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["sglang_optimization_lock_report"].relative_to(root)),
                "--json-output",
                str(p["sglang_optimization_lock"].relative_to(root)),
            ],
        ),
        (
            "final vLLM optimization lock",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_vllm_optimization_lock",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["vllm_optimization_lock_report"].relative_to(root)),
                "--json-output",
                str(p["vllm_optimization_lock"].relative_to(root)),
            ],
        ),
        (
            "final vLLM online parity protocol",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["vllm_online_parity_protocol_report"].relative_to(root)),
                "--json-output",
                str(p["vllm_online_parity_protocol"].relative_to(root)),
            ],
        ),
        (
            "final runtime image contract",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_runtime_image_contract",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["runtime_image_contract_report"].relative_to(root)),
                "--json-output",
                str(p["runtime_image_contract"].relative_to(root)),
            ],
        ),
        (
            "final checkpoint watchlist",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["final_checkpoint_watchlist_report"].relative_to(root)),
                "--json-output",
                str(p["final_checkpoint_watchlist"].relative_to(root)),
            ],
        ),
        (
            "final stage latency budget",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_latency_budget",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_latency_budget_report"].relative_to(root)),
                "--json-output",
                str(p["stage_latency_budget"].relative_to(root)),
            ],
        ),
        (
            "final stage boundary bottleneck ledger",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_boundary_bottleneck_ledger",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_boundary_ledger_report"].relative_to(root)),
                "--json-output",
                str(p["stage_boundary_ledger"].relative_to(root)),
            ],
        ),
        (
            "final serving capacity matrix",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_serving_capacity_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["serving_capacity_matrix_report"].relative_to(root)),
                "--json-output",
                str(p["serving_capacity_matrix"].relative_to(root)),
            ],
        ),
        (
            "final stage causal graph",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_causal_graph",
                "--root",
                str(root),
                "--output",
                str(p["stage_causal_graph"].relative_to(root)),
                "--json-output",
                str(p["stage_causal_graph_json"].relative_to(root)),
                "--strict",
            ],
        ),
        (
            "final caveat adjudication matrix",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["caveat_adjudication_matrix"].relative_to(root)),
                "--json-output",
                str(p["caveat_adjudication_matrix_json"].relative_to(root)),
            ],
        ),
        (
            "final checkpoint watchlist after caveat",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["final_checkpoint_watchlist_report"].relative_to(root)),
                "--json-output",
                str(p["final_checkpoint_watchlist"].relative_to(root)),
            ],
        ),
        (
            "final stage latency budget after caveat",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_latency_budget",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_latency_budget_report"].relative_to(root)),
                "--json-output",
                str(p["stage_latency_budget"].relative_to(root)),
            ],
        ),
        (
            "final tail confidence appendix after caveat",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_tail_confidence_appendix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["tail_confidence_appendix_report"].relative_to(root)),
                "--json-output",
                str(p["tail_confidence_appendix"].relative_to(root)),
            ],
        ),
        (
            "final length regime coverage after caveat",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_length_regime_coverage",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["length_regime_coverage_report"].relative_to(root)),
                "--json-output",
                str(p["length_regime_coverage"].relative_to(root)),
            ],
        ),
        (
            "final stage boundary bottleneck ledger after caveat",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_stage_boundary_bottleneck_ledger",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["stage_boundary_ledger_report"].relative_to(root)),
                "--json-output",
                str(p["stage_boundary_ledger"].relative_to(root)),
            ],
        ),
        (
            "final serving capacity matrix after caveat",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_serving_capacity_matrix",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["serving_capacity_matrix_report"].relative_to(root)),
                "--json-output",
                str(p["serving_capacity_matrix"].relative_to(root)),
            ],
        ),
        (
            "final rerun acceptance contract",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_acceptance_contract",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["rerun_acceptance_contract_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_acceptance_contract"].relative_to(root)),
            ],
        ),
        (
            "final rerun delta triage",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_delta_triage",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["rerun_delta_triage_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_delta_triage"].relative_to(root)),
            ],
        ),
        (
            "final slide asset map",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_slide_asset_map",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["slide_asset_map_report"].relative_to(root)),
                "--json-output",
                str(p["slide_asset_map"].relative_to(root)),
            ],
        ),
        (
            "final share path hygiene after status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_path_hygiene",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_path_hygiene"].relative_to(root)),
            ],
        ),
        (
            "final command reference hygiene after status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_command_reference_hygiene",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["command_reference_hygiene"].relative_to(root)),
            ],
        ),
        (
            "final reproduction command manifest before consistency guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "final rerun time budget before consistency guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_time_budget",
                "--root",
                str(root),
                "--output",
                str(p["rerun_time_budget_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_time_budget"].relative_to(root)),
            ],
        ),
        (
            "share bundle manifest before final readiness after status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "final readiness audit after status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "final share consistency guard after final readiness",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_consistency_guard",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["share_consistency_guard_report"].relative_to(root)),
                "--json-output",
                str(p["share_consistency_guard"].relative_to(root)),
            ],
        ),
        (
            "final reproduction command manifest after consistency guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "final rerun time budget after consistency guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_rerun_time_budget",
                "--root",
                str(root),
                "--output",
                str(p["rerun_time_budget_report"].relative_to(root)),
                "--json-output",
                str(p["rerun_time_budget"].relative_to(root)),
            ],
        ),
        (
            "final university technical report after consistency guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_technical_report",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["university_technical_report"].relative_to(root)),
                "--json-output",
                str(p["university_technical_report_json"].relative_to(root)),
            ],
        ),
        (
            "final share consistency guard after university report",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_consistency_guard",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["share_consistency_guard_report"].relative_to(root)),
                "--json-output",
                str(p["share_consistency_guard"].relative_to(root)),
            ],
        ),
        (
            "final reproduction command manifest after university guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_repro_command_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["repro_command_manifest"].relative_to(root)),
            ],
        ),
        (
            "final university technical report after stable repro manifest",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_technical_report",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["university_technical_report"].relative_to(root)),
                "--json-output",
                str(p["university_technical_report_json"].relative_to(root)),
            ],
        ),
        (
            "final university review packet after stable university report",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_university_review_packet",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["university_review_packet"].relative_to(root)),
                "--json-output",
                str(p["university_review_packet_json"].relative_to(root)),
            ],
        ),
        (
            "final readiness audit after stable university report",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "final share bundle manifest after consistency guard",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "final manifest after status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final share bundle package after status summary",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_package",
                "--root",
                str(root),
                "--strict",
                "--source-manifest",
                str(p["share_bundle_manifest"].relative_to(root)),
                "--output",
                "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
                "--json-output",
                str(p["share_bundle_package"].relative_to(root)),
            ],
        ),
        (
            "final share package validation",
            [
                python,
                "-m",
                "benchmarks.eval.validate_qwen35_omni_share_package",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_package_validation"].relative_to(root)),
            ],
        ),
        (
            "final share package receiver smoke validation",
            [
                python,
                "-m",
                "benchmarks.eval.validate_qwen35_omni_share_package",
                "--root",
                str(root),
                "--strict",
                "--receiver-smoke-dir",
                "/tmp/qwen35_omni_receiver_smoke_audit",
                "--json-output",
                str(p["share_package_receiver_smoke_validation"].relative_to(root)),
            ],
        ),
        (
            "final extracted-only share package validation",
            [
                "bash",
                "-lc",
                "\n".join(
                    [
                        "set -euo pipefail",
                        f"rm -rf {extracted_tmp}",
                        f"mkdir -p {extracted_tmp}",
                        "tar -xzf "
                        f"{root}/results/qwen35_report_audit_20260619/"
                        "qwen35_omni_share_bundle_20260621.tar.gz "
                        f"-C {extracted_tmp}",
                        f"{python} {extracted_root}/benchmarks/eval/"
                        "validate_qwen35_omni_share_package.py "
                        f"--root {extracted_root} "
                        "--extracted-only --strict "
                        f"--json-output {p['share_package_validation_extracted']}",
                    ]
                ),
            ],
        ),
        (
            "final external standalone share package validation",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation",
                "--root",
                str(root),
                "--strict",
                "--work-dir",
                "/tmp/qwen35_omni_external_standalone_bundle_validation_audit",
                "--json-output",
                str(p["share_package_external_standalone_validation"].relative_to(root)),
            ],
        ),
        (
            "final evidence-query smoke refresh",
            _evidence_query_smoke_refresh_cmd(
                root,
                p,
                python,
                work_dir=Path("/tmp/qwen35_omni_evidence_query_smoke_audit"),
            ),
        ),
        (
            "final receiver quickcheck contract",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_receiver_quickcheck_contract",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["receiver_quickcheck_contract_report"].relative_to(root)),
                "--json-output",
                str(p["receiver_quickcheck_contract"].relative_to(root)),
            ],
        ),
        (
            "final share release seal",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_release_seal",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["share_release_seal_report"].relative_to(root)),
                "--json-output",
                str(p["share_release_seal"].relative_to(root)),
            ],
        ),
        (
            "final manifest after release seal",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final readiness audit after release seal",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "final manifest before package fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final share bundle manifest after readiness fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_manifest",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_bundle_manifest"].relative_to(root)),
            ],
        ),
        (
            "final share bundle package after readiness fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_bundle_package",
                "--root",
                str(root),
                "--strict",
                "--source-manifest",
                str(p["share_bundle_manifest"].relative_to(root)),
                "--output",
                "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
                "--json-output",
                str(p["share_bundle_package"].relative_to(root)),
            ],
        ),
        (
            "final share package validation after readiness fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.validate_qwen35_omni_share_package",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["share_package_validation"].relative_to(root)),
            ],
        ),
        (
            "final share package receiver smoke after readiness fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.validate_qwen35_omni_share_package",
                "--root",
                str(root),
                "--strict",
                "--receiver-smoke-dir",
                "/tmp/qwen35_omni_receiver_smoke_audit_fixed_point",
                "--json-output",
                str(p["share_package_receiver_smoke_validation"].relative_to(root)),
            ],
        ),
        (
            "final extracted-only package validation after readiness fixed point",
            [
                "bash",
                "-lc",
                "\n".join(
                    [
                        "set -euo pipefail",
                        f"rm -rf {extracted_tmp}",
                        f"mkdir -p {extracted_tmp}",
                        "tar -xzf "
                        f"{root}/results/qwen35_report_audit_20260619/"
                        "qwen35_omni_share_bundle_20260621.tar.gz "
                        f"-C {extracted_tmp}",
                        f"{python} {extracted_root}/benchmarks/eval/"
                        "validate_qwen35_omni_share_package.py "
                        f"--root {extracted_root} "
                        "--extracted-only --strict "
                        f"--json-output {p['share_package_validation_extracted']}",
                    ]
                ),
            ],
        ),
        (
            "final external standalone validation after readiness fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation",
                "--root",
                str(root),
                "--strict",
                "--work-dir",
                "/tmp/qwen35_omni_external_standalone_bundle_validation_fixed_point",
                "--json-output",
                str(p["share_package_external_standalone_validation"].relative_to(root)),
            ],
        ),
        (
            "final evidence-query smoke refresh after package fixed point",
            _evidence_query_smoke_refresh_cmd(
                root,
                p,
                python,
                work_dir=Path(
                    "/tmp/qwen35_omni_evidence_query_smoke_audit_fixed_point"
                ),
            ),
        ),
        (
            "final receiver quickcheck contract after package fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_receiver_quickcheck_contract",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["receiver_quickcheck_contract_report"].relative_to(root)),
                "--json-output",
                str(p["receiver_quickcheck_contract"].relative_to(root)),
            ],
        ),
        (
            "final collaborator return check after package fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.qwen35_omni_collaborator_return_check",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["collaborator_return_check"].relative_to(root)),
            ],
        ),
        (
            "final readiness audit after package fixed point",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_readiness",
                "--root",
                str(root),
                "--strict",
                "--json-output",
                str(p["final_readiness"].relative_to(root)),
            ],
        ),
        (
            "final checkpoint watchlist after release seal",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["final_checkpoint_watchlist_report"].relative_to(root)),
                "--json-output",
                str(p["final_checkpoint_watchlist"].relative_to(root)),
            ],
        ),
        (
            "final manifest after release readiness",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_report_manifest",
                "--root",
                str(root),
                "--output",
                str(p["manifest"].relative_to(root)),
            ],
        ),
        (
            "final completion audit after checkpoint watchlist",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_final_completion_audit",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["final_completion_audit_report"].relative_to(root)),
                "--json-output",
                str(p["final_completion_audit"].relative_to(root)),
            ],
        ),
        (
            "final share release seal after completion audit",
            [
                python,
                "-m",
                "benchmarks.eval.build_qwen35_omni_share_release_seal",
                "--root",
                str(root),
                "--strict",
                "--output",
                str(p["share_release_seal_report"].relative_to(root)),
                "--json-output",
                str(p["share_release_seal"].relative_to(root)),
            ],
        ),
    ]


def _summarize_outputs(root: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    p = _paths(root)
    tables = _load_json(p["tables"])
    claims = _load_json(p["claims"])
    preflight = _load_json(p["preflight"])
    coverage = _load_json(p["coverage"])
    manifest = _load_json(p["manifest"])
    environment = _load_json(p["environment_snapshot"])
    admission = _load_json(p["vllm_admission"])
    runtime_image_contract = _load_json(p["runtime_image_contract"])
    rerun_acceptance_contract = _load_json(p["rerun_acceptance_contract"])
    rerun_delta_triage = _load_json(p["rerun_delta_triage"])
    sglang_lock = _load_json(p["sglang_optimization_lock"])
    vllm_lock = _load_json(p["vllm_optimization_lock"])
    vllm_online_protocol = _load_json(p["vllm_online_parity_protocol"])
    final_checkpoint_watchlist = _load_json(p["final_checkpoint_watchlist"])
    final_completion_audit = _load_json(p["final_completion_audit"])
    tail_confidence_appendix = _load_json(p["tail_confidence_appendix"])
    stage_latency_budget = _load_json(p["stage_latency_budget"])
    stage_boundary_ledger = _load_json(p["stage_boundary_ledger"])
    serving_capacity_matrix = _load_json(p["serving_capacity_matrix"])
    share_consistency_guard = _load_json(p["share_consistency_guard"])
    stage_drilldown_index = _load_json(p["stage_drilldown_index"])
    metric_provenance_index = _load_json(p["metric_provenance_index"])
    stage_reproduction_drilldown = _load_json(p["stage_reproduction_drilldown"])
    stage_route_decision_matrix = _load_json(p["stage_route_decision_matrix"])
    pressure_stage_heatmap = _load_json(p["pressure_stage_heatmap"])
    length_regime_coverage = _load_json(p["length_regime_coverage"])
    rerun_time_budget = _load_json(p["rerun_time_budget"])
    stage_causal_graph = _load_json(p["stage_causal_graph_json"])
    caveat_adjudication_matrix = _load_json(p["caveat_adjudication_matrix_json"])
    regime_decision_matrix = _load_json(p["regime_decision_matrix_json"])
    university_review_packet = _load_json(p["university_review_packet_json"])
    university_technical_report = _load_json(p["university_technical_report_json"])
    runtime_comparison_contract = _load_json(p["runtime_comparison_contract_json"])
    interactions = _load_json(p["stage_interactions"])
    scorecard = _load_json(p["headline_scorecard"])
    chart_pack = _load_json(p["chart_pack_manifest"])
    chart_source_consistency = _load_json(p["chart_source_consistency"])
    acceptance = _load_json(p["acceptance_matrix"])
    confidence = _load_json(p["confidence_ledger"])
    objective = _load_json(p["objective_completion"])
    repro_commands = _load_json(p["repro_command_manifest"])
    defense_claim_matrix = _load_json(p["defense_claim_matrix"])
    claim_metric_crosswalk = _load_json(p["claim_metric_crosswalk"])
    objective_requirement_crosswalk = _load_json(
        p["objective_requirement_crosswalk"]
    )
    optimization_candidate_ledger = _load_json(
        p["optimization_candidate_ledger"]
    )
    final_readiness = _load_json(p["final_readiness"])
    share_path_hygiene = _load_json(p["share_path_hygiene"])
    command_reference_hygiene = _load_json(p["command_reference_hygiene"])
    share_bundle = _load_json(p["share_bundle_manifest"])
    share_package = _load_json(p["share_bundle_package"])
    share_package_validation = _load_json(p["share_package_validation"])
    share_package_receiver_smoke = _load_json(
        p["share_package_receiver_smoke_validation"]
    )
    share_package_validation_extracted = _load_json(
        p["share_package_validation_extracted"]
    )
    share_package_external_standalone = _load_json(
        p["share_package_external_standalone_validation"]
    )
    evidence_query_smoke = _load_json(p["evidence_query_smoke_summary"])
    receiver_quickcheck_contract = _load_json(p["receiver_quickcheck_contract"])
    collaborator_return_check = _load_json(p["collaborator_return_check"])
    share_release_seal = _load_json(p["share_release_seal"])
    videoamme_seedtts = _load_json(p["videoamme_seedtts_summary"])
    return {
        "steps": steps,
        "ok": all(step["ok"] for step in steps),
        "artifact_status": tables.get("artifact_status", []),
        "claims": {
            "passed": claims.get("passed"),
            "total_checks": claims.get("total_checks"),
            "failed_checks": claims.get("failed_checks"),
        },
        "preflight": preflight.get("summary", {}),
        "coverage": coverage.get("summary", {}),
        "manifest": manifest.get("summary", {}),
        "environment_snapshot": {
            "gpu": {
                "ok": environment.get("gpu", {}).get("ok"),
                "count": environment.get("gpu", {}).get("count"),
                "required_gpus": environment.get("gpu", {}).get("required_gpus"),
            },
            "docker_images": {
                label: {
                    "ok": image.get("ok"),
                    "id": image.get("id"),
                    "created": image.get("created"),
                }
                for label, image in environment.get("docker_images", {}).items()
            },
        },
        "videoamme_seedtts_meta": {
            "rows": videoamme_seedtts.get("rows"),
            "target_mode": videoamme_seedtts.get("target_mode"),
            "output": videoamme_seedtts.get("output"),
        },
        "stage_interactions": interactions.get("summary", {}),
        "stage_drilldown_index": stage_drilldown_index.get("summary", {}),
        "metric_provenance_index": metric_provenance_index.get("summary", {}),
        "stage_reproduction_drilldown": stage_reproduction_drilldown.get(
            "summary", {}
        ),
        "stage_route_decision_matrix": stage_route_decision_matrix.get(
            "summary", {}
        ),
        "pressure_stage_heatmap": pressure_stage_heatmap.get("summary", {}),
        "length_regime_coverage": length_regime_coverage.get("summary", {}),
        "rerun_time_budget": rerun_time_budget.get("summary", {}),
        "stage_causal_graph": stage_causal_graph.get("summary", {}),
        "caveat_adjudication_matrix": caveat_adjudication_matrix.get(
            "summary", {}
        ),
        "regime_decision_matrix": regime_decision_matrix.get("summary", {}),
        "university_review_packet": university_review_packet.get("summary", {}),
        "university_technical_report": university_technical_report.get(
            "summary", {}
        ),
        "runtime_comparison_contract": runtime_comparison_contract.get(
            "summary", {}
        ),
        "headline_scorecard": scorecard.get("summary", {}),
        "chart_pack": chart_pack.get("summary", {}),
        "chart_source_consistency": chart_source_consistency.get("summary", {}),
        "acceptance_matrix": acceptance.get("summary", {}),
        "confidence_ledger": confidence.get("summary", {}),
        "objective_completion": objective.get("summary", {}),
        "repro_command_manifest": repro_commands.get("summary", {}),
        "defense_claim_matrix": defense_claim_matrix.get("summary", {}),
        "claim_metric_crosswalk": claim_metric_crosswalk.get("summary", {}),
        "objective_requirement_crosswalk": objective_requirement_crosswalk.get(
            "summary", {}
        ),
        "optimization_candidate_ledger": optimization_candidate_ledger.get(
            "summary", {}
        ),
        "final_readiness": final_readiness.get("summary", {}),
        "share_path_hygiene": share_path_hygiene.get("summary", {}),
        "command_reference_hygiene": command_reference_hygiene.get("summary", {}),
        "share_bundle_manifest": share_bundle.get("summary", {}),
        "share_bundle_package": _stable_share_bundle_package_summary(share_package),
        "share_package_validation": _stable_share_package_validation_summary(
            share_package_validation
        ),
        "share_package_receiver_smoke_validation": (
            _stable_share_package_validation_summary(share_package_receiver_smoke)
        ),
        "share_package_validation_extracted": (
            _stable_share_package_validation_summary(share_package_validation_extracted)
        ),
        "share_package_external_standalone_validation": (
            share_package_external_standalone.get("summary", {})
        ),
        "evidence_query_smoke": _stable_evidence_query_smoke_summary(
            evidence_query_smoke
        ),
        "receiver_quickcheck_contract": (
            receiver_quickcheck_contract.get("summary", {})
        ),
        "collaborator_return_check": collaborator_return_check.get("summary", {}),
        "share_release_seal": _stable_share_release_seal_summary(share_release_seal),
        "package_stability_note": (
            "Tarball identity fields are intentionally omitted from this run "
            "summary to avoid self-referential package hashes. Use "
            "share_bundle_package_manifest.json and the .sha256 file for the "
            "tarball hash. Receiver-smoke validation is adjacent evidence and "
            "is intentionally not a tarball member. The full audit also "
            "regenerates clean-directory extracted-only and external standalone "
            "validation summaries plus the adjacent release seal."
        ),
        "runtime_image_contract": runtime_image_contract.get("summary", {}),
        "rerun_acceptance_contract": rerun_acceptance_contract.get("summary", {}),
        "rerun_delta_triage": rerun_delta_triage.get("summary", {}),
        "sglang_optimization_lock": sglang_lock.get("summary", {}),
        "vllm_optimization_lock": vllm_lock.get("summary", {}),
        "vllm_online_parity_protocol": vllm_online_protocol.get("summary", {}),
        "final_checkpoint_watchlist": final_checkpoint_watchlist.get("summary", {}),
        "final_completion_audit": final_completion_audit.get("summary", {}),
        "tail_confidence_appendix": tail_confidence_appendix.get("summary", {}),
        "stage_latency_budget": stage_latency_budget.get("summary", {}),
        "stage_boundary_bottleneck_ledger": stage_boundary_ledger.get("summary", {}),
        "serving_capacity_matrix": serving_capacity_matrix.get("summary", {}),
        "share_consistency_guard": share_consistency_guard.get("summary", {}),
        "vllm_admission_diagnosis": [
            {
                "label": row.get("label"),
                "diagnosis": row.get("diagnosis"),
                "runner_overhead_pct_wall": row.get("runner_overhead_pct_wall"),
                "batch_admission_span_avg_ms": row.get(
                    "batch_admission_span_avg_ms"
                ),
            }
            for row in admission.get("rows", [])
        ],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n## Audit Summary\n")
    print(f"steps_ok={summary['ok']}")
    print(f"claims={summary['claims']}")
    print(f"preflight={summary['preflight']}")
    print(f"coverage={summary['coverage']}")
    print(f"manifest={summary['manifest']}")
    print(f"environment_snapshot={summary['environment_snapshot']}")
    print(f"videoamme_seedtts_meta={summary['videoamme_seedtts_meta']}")
    print(f"stage_interactions={summary['stage_interactions']}")
    print(f"stage_drilldown_index={summary['stage_drilldown_index']}")
    print(f"metric_provenance_index={summary['metric_provenance_index']}")
    print(f"stage_reproduction_drilldown={summary['stage_reproduction_drilldown']}")
    print(f"stage_route_decision_matrix={summary['stage_route_decision_matrix']}")
    print(f"pressure_stage_heatmap={summary['pressure_stage_heatmap']}")
    print(f"length_regime_coverage={summary['length_regime_coverage']}")
    print(f"rerun_time_budget={summary['rerun_time_budget']}")
    print(f"stage_causal_graph={summary['stage_causal_graph']}")
    print(f"caveat_adjudication_matrix={summary['caveat_adjudication_matrix']}")
    print(f"regime_decision_matrix={summary['regime_decision_matrix']}")
    print(f"university_review_packet={summary['university_review_packet']}")
    print(f"university_technical_report={summary['university_technical_report']}")
    print(f"runtime_comparison_contract={summary['runtime_comparison_contract']}")
    print(f"headline_scorecard={summary['headline_scorecard']}")
    print(f"chart_pack={summary['chart_pack']}")
    print(f"chart_source_consistency={summary['chart_source_consistency']}")
    print(f"acceptance_matrix={summary['acceptance_matrix']}")
    print(f"confidence_ledger={summary['confidence_ledger']}")
    print(f"objective_completion={summary['objective_completion']}")
    print(f"repro_command_manifest={summary['repro_command_manifest']}")
    print(f"defense_claim_matrix={summary['defense_claim_matrix']}")
    print(f"claim_metric_crosswalk={summary['claim_metric_crosswalk']}")
    print(
        "objective_requirement_crosswalk="
        f"{summary['objective_requirement_crosswalk']}"
    )
    print(f"optimization_candidate_ledger={summary['optimization_candidate_ledger']}")
    print(f"final_readiness={summary['final_readiness']}")
    print(f"share_path_hygiene={summary['share_path_hygiene']}")
    print(f"share_bundle_manifest={summary['share_bundle_manifest']}")
    print(f"share_bundle_package={summary['share_bundle_package']}")
    print(f"share_package_validation={summary['share_package_validation']}")
    print(
        "share_package_receiver_smoke_validation="
        f"{summary['share_package_receiver_smoke_validation']}"
    )
    print(f"share_package_validation_extracted={summary['share_package_validation_extracted']}")
    print(
        "share_package_external_standalone_validation="
        f"{summary['share_package_external_standalone_validation']}"
    )
    print(f"evidence_query_smoke={summary['evidence_query_smoke']}")
    print(f"receiver_quickcheck_contract={summary['receiver_quickcheck_contract']}")
    print(f"collaborator_return_check={summary['collaborator_return_check']}")
    print(f"share_release_seal={summary['share_release_seal']}")
    print(f"runtime_image_contract={summary['runtime_image_contract']}")
    print(f"rerun_acceptance_contract={summary['rerun_acceptance_contract']}")
    print(f"rerun_delta_triage={summary['rerun_delta_triage']}")
    print(f"sglang_optimization_lock={summary['sglang_optimization_lock']}")
    print(f"vllm_optimization_lock={summary['vllm_optimization_lock']}")
    print(f"vllm_online_parity_protocol={summary['vllm_online_parity_protocol']}")
    print(f"final_checkpoint_watchlist={summary['final_checkpoint_watchlist']}")
    print(f"final_completion_audit={summary['final_completion_audit']}")
    print(f"tail_confidence_appendix={summary['tail_confidence_appendix']}")
    print(f"stage_latency_budget={summary['stage_latency_budget']}")
    print(
        "stage_boundary_bottleneck_ledger="
        f"{summary['stage_boundary_bottleneck_ledger']}"
    )
    print(f"serving_capacity_matrix={summary['serving_capacity_matrix']}")
    print(f"share_consistency_guard={summary['share_consistency_guard']}")
    print("artifact_status=" + json.dumps(summary["artifact_status"], ensure_ascii=False))
    print(
        "vllm_admission_diagnosis="
        + json.dumps(summary["vllm_admission_diagnosis"], ensure_ascii=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Qwen3.5-Omni report audit/regeneration steps."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON summary output. This summary is not part of the manifest.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    steps: list[dict[str, Any]] = []
    for label, cmd in build_commands(root):
        step = _run_step(label, cmd, cwd=root)
        steps.append(step)
        if not step["ok"]:
            output = _resolve_summary_output(root, args.summary_output)
            if output is not None:
                _save_json({"steps": steps, "ok": False}, output)
            raise SystemExit(step["returncode"] or 1)
        _save_in_progress_summary(root, args.summary_output, steps)

    summary = _summarize_outputs(root, steps)
    print_summary(summary)
    output = _resolve_summary_output(root, args.summary_output)
    if output is not None:
        _save_json(summary, output)


if __name__ == "__main__":
    main()
