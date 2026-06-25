# SPDX-License-Identifier: Apache-2.0
"""Build collaborator rerun acceptance thresholds for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "rerun_acceptance_contract.json"
COLLABORATOR_RERUN_SHEET = Path(
    "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md"
)

CRITICAL_RETURN_EVIDENCE = [
    "results/qwen35_report_audit_20260619/audit_run_summary.json",
    "results/qwen35_report_audit_20260619/environment_snapshot.json",
    "results/qwen35_report_audit_20260619/manifest.json",
    "results/qwen35_report_audit_20260619/coverage_matrix.json",
    "results/qwen35_report_audit_20260619/claims_verification.json",
    "results/qwen35_report_audit_20260619/headline_scorecard.json",
    "results/qwen35_report_audit_20260619/acceptance_matrix.json",
    "results/qwen35_report_audit_20260619/confidence_ledger.json",
    "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
    "results/qwen35_report_audit_20260619/runtime_image_contract.json",
    "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
    "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
    "results/qwen35_report_audit_20260619/repro_command_manifest.json",
    "results/qwen35_report_audit_20260619/final_readiness_audit.json",
    "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
    "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
    "results/qwen35_report_audit_20260619/metric_provenance_index.json",
    "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
    "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
    "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
    "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
    "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
    "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
    "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
    "results/qwen35_report_audit_20260619/stage_latency_budget.json",
    "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
    "results/qwen35_report_audit_20260619/stage_causal_graph.json",
    "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
    "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
    "results/qwen35_report_audit_20260619/rerun_delta_triage.json",
    "results/qwen35_report_audit_20260619/share_package_validation.json",
    "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
    "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
    "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
]

COMMAND_RETURN_EVIDENCE_REQUIREMENTS = {
    "run_full_audit": [
        "results/qwen35_report_audit_20260619/audit_run_summary.json",
        "results/qwen35_report_audit_20260619/manifest.json",
        "results/qwen35_report_audit_20260619/coverage_matrix.json",
        "results/qwen35_report_audit_20260619/claims_verification.json",
        "results/qwen35_report_audit_20260619/final_readiness_audit.json",
        "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
    ],
    "sglang_videoamme_stress": [
        "results/qwen35_report_audit_20260619/headline_scorecard.json",
        "results/qwen35_report_audit_20260619/acceptance_matrix.json",
        "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
        "results/qwen35_report_audit_20260619/stage_latency_budget.json",
        "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
        "results/qwen35_report_audit_20260619/stage_causal_graph.json",
        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
        "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
    ],
    "sglang_synthetic_text_to_speech": [
        "results/qwen35_report_audit_20260619/acceptance_matrix.json",
        "results/qwen35_report_audit_20260619/stage_latency_budget.json",
        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
        "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
    ],
    "sglang_recompute_wer": [
        "results/qwen35_report_audit_20260619/claims_verification.json",
        "results/qwen35_report_audit_20260619/headline_scorecard.json",
        "results/qwen35_report_audit_20260619/acceptance_matrix.json",
    ],
    "vllm_c4_original": [
        "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
        "results/qwen35_report_audit_20260619/headline_scorecard.json",
        "results/qwen35_report_audit_20260619/claims_verification.json",
        "results/qwen35_report_audit_20260619/runtime_image_contract.json",
        "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
        "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
        "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
    ],
    "vllm_c8_original": [
        "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
        "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
        "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
        "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
        "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
    ],
    "vllm_c8_prebuild_w4": [
        "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
        "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
        "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
        "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
    ],
    "build_headline_scorecard": [
        "results/qwen35_report_audit_20260619/headline_scorecard.json",
    ],
    "build_acceptance_matrix": [
        "results/qwen35_report_audit_20260619/acceptance_matrix.json",
    ],
    "build_stage_latency_budget": [
        "results/qwen35_report_audit_20260619/stage_latency_budget.json",
    ],
    "build_stage_boundary_bottleneck_ledger": [
        "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
    ],
    "build_stage_causal_graph": [
        "results/qwen35_report_audit_20260619/stage_causal_graph.json",
    ],
    "build_stage_reproduction_drilldown": [
        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
    ],
    "build_stage_route_decision_matrix": [
        "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
    ],
    "build_runtime_comparison_contract": [
        "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
    ],
    "build_runtime_image_contract": [
        "results/qwen35_report_audit_20260619/runtime_image_contract.json",
    ],
    "build_rerun_acceptance_contract": [
        "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
    ],
    "build_rerun_delta_triage": [
        "results/qwen35_report_audit_20260619/rerun_delta_triage.json",
    ],
    "build_metric_provenance_index": [
        "results/qwen35_report_audit_20260619/metric_provenance_index.json",
    ],
    "build_objective_requirement_crosswalk": [
        "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
    ],
    "build_final_readiness_audit": [
        "results/qwen35_report_audit_20260619/final_readiness_audit.json",
    ],
    "build_share_bundle_manifest": [
        "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
    ],
    "build_share_bundle_package": [
        "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
    ],
    "validate_share_bundle_package": [
        "results/qwen35_report_audit_20260619/share_package_validation.json",
    ],
    "validate_share_bundle_receiver_smoke": [
        "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
    ],
    "validate_extracted_share_bundle": [
        "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
    ],
    "validate_external_standalone_share_bundle": [
        "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
    ],
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    evidence: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "evidence": self.evidence,
        }


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _row_by_key(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def _table(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    rows = payload.get("tables", {}).get(name, [])
    return rows if isinstance(rows, list) else []


def _claim_passed(claims: dict[str, Any], name: str) -> bool:
    for check in claims.get("checks", []):
        if check.get("name") == name:
            return bool(check.get("passed"))
    return False


def _metric_rule(
    *,
    rule_id: str,
    category: str,
    scope: str,
    baseline: dict[str, Any],
    acceptance: dict[str, Any],
    replacement: str,
    evidence: list[str],
    caveat: str = "",
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "category": category,
        "scope": scope,
        "baseline": baseline,
        "acceptance": acceptance,
        "headline_replacement_rule": replacement,
        "evidence": evidence,
        "caveat": caveat,
    }


def _return_evidence_records(root: Path, sheet_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in CRITICAL_RETURN_EVIDENCE:
        name = Path(rel_path).name
        records.append(
            {
                "relative_path": rel_path,
                "exists_in_current_package": (root / rel_path).is_file(),
                "listed_in_collaborator_sheet": rel_path in sheet_text or name in sheet_text,
                "purpose": _return_evidence_purpose(name),
            }
        )
    return records


def _return_evidence_purpose(name: str) -> str:
    purposes = {
        "audit_run_summary.json": "full audit pass/fail and step order",
        "environment_snapshot.json": "hardware, image digest, model/data path",
        "manifest.json": "raw and packaged evidence inventory",
        "coverage_matrix.json": "original requirement coverage",
        "claims_verification.json": "numeric claim gate",
        "headline_scorecard.json": "strict headline scorecard",
        "acceptance_matrix.json": "pressure/regime acceptance rows",
        "confidence_ledger.json": "safe wording and unsupported-claim guard",
        "runtime_comparison_contract.json": "warmed c4-only cross-runtime headline and c8 diagnostic boundary",
        "runtime_image_contract.json": "SGLang/vLLM image and optimization-scope lock",
        "rerun_acceptance_contract.json": "replacement thresholds",
        "receiver_quickcheck_contract.json": "receiver quickcheck contract and public-doc route evidence",
        "repro_command_manifest.json": "exact rerun commands and expected evidence",
        "final_readiness_audit.json": "send/no-send gate after rerun",
        "share_bundle_manifest.json": "share bundle evidence inventory",
        "share_bundle_package_manifest.json": "tarball packaging and checksum creation",
        "metric_provenance_index.json": "metric-to-raw-artifact provenance",
        "objective_requirement_crosswalk.json": "original requirement and optimization verdict crosswalk",
        "sglang_optimization_lock.json": "SGLang recipe lock and anti-recipes",
        "vllm_optimization_lock.json": "optimized vLLM baseline lock",
        "vllm_online_parity_protocol.json": "online parity upgrade gate",
        "vllm_admission_diagnosis.json": "vLLM offline admission and prompt-feed diagnosis",
        "vllm_log_stage_summary.json": "vLLM log-derived stage timing summary",
        "stage_interaction_summary.json": "stage connection flags",
        "stage_latency_budget.json": "stage latency proportions",
        "stage_boundary_bottleneck_ledger.json": "boundary bottleneck verdicts",
        "stage_causal_graph.json": "stage causal edges and raw drilldown",
        "stage_reproduction_drilldown.json": "stage row to jq/artifact/command map",
        "stage_route_decision_matrix.json": "route-level bottleneck decisions",
        "rerun_delta_triage.json": "symptom-to-stage rerun delta triage",
        "share_package_validation.json": "tarball validation evidence",
        "share_package_receiver_smoke_validation.json": "receiver smoke evidence",
        "share_package_validation_extracted.json": "extracted-only validation evidence",
        "share_package_external_standalone_validation.json": "repo-independent standalone validation evidence",
    }
    return purposes.get(name, "required collaborator return evidence")


def _command_return_matrix(
    repro_manifest: dict[str, Any],
    return_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    command_by_id = {
        str(command.get("id")): command
        for command in repro_manifest.get("commands", [])
        if isinstance(command, dict) and command.get("id")
    }
    critical_paths = {
        str(record.get("relative_path"))
        for record in return_evidence
        if record.get("relative_path")
    }

    rows: list[dict[str, Any]] = []
    missing_command_ids: list[str] = []
    missing_required_return_files: list[str] = []
    commands_without_raw_or_gate_evidence: list[str] = []

    for command_id, required_files in COMMAND_RETURN_EVIDENCE_REQUIREMENTS.items():
        command = command_by_id.get(command_id, {})
        if not command:
            missing_command_ids.append(command_id)
        evidence_after_run = [
            str(item)
            for item in command.get("evidence_after_run", [])
            if str(item or "").strip()
        ]
        required_files_missing_from_contract = [
            rel_path for rel_path in required_files if rel_path not in critical_paths
        ]
        missing_required_return_files.extend(required_files_missing_from_contract)
        declared_gate_artifacts = [
            rel_path for rel_path in evidence_after_run if rel_path in critical_paths
        ]
        raw_or_runtime_artifacts = [
            rel_path
            for rel_path in evidence_after_run
            if rel_path not in critical_paths
        ]
        if not declared_gate_artifacts and not raw_or_runtime_artifacts:
            commands_without_raw_or_gate_evidence.append(command_id)
        rows.append(
            {
                "command_id": command_id,
                "phase": command.get("phase", ""),
                "purpose": command.get("purpose", ""),
                "evidence_after_run": evidence_after_run,
                "raw_or_runtime_artifacts": raw_or_runtime_artifacts,
                "declared_gate_artifacts": declared_gate_artifacts,
                "required_return_files": required_files,
                "required_files_missing_from_contract": required_files_missing_from_contract,
                "ready": bool(command)
                and not required_files_missing_from_contract
                and bool(evidence_after_run),
            }
        )

    diagnostics = {
        "missing_command_ids": sorted(set(missing_command_ids)),
        "missing_required_return_files": sorted(set(missing_required_return_files)),
        "commands_without_raw_or_gate_evidence": sorted(
            set(commands_without_raw_or_gate_evidence)
        ),
    }
    return rows, diagnostics


def _sglang_stress_rules(
    stress_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in stress_rows:
        c = _int(row.get("concurrency"))
        stage = _row_by_key(stage_rows, "concurrency", c)
        role = "serving_window"
        if c == 8:
            role = "peak_throughput"
        elif c == 16:
            role = "saturation_boundary"
        qps = _float(row.get("throughput_qps"))
        latency_p95 = _float(row.get("latency_p95_s"))
        wer = _float(row.get("wer_corpus"))
        result.append(
            _metric_rule(
                rule_id=f"sglang_videoamme_c{c}",
                category="sglang_stress",
                scope=f"Video-AMME ci-50 c={c} / {role}",
                baseline={
                    "completed": row.get("n"),
                    "accuracy": row.get("accuracy"),
                    "wer_corpus": row.get("wer_corpus"),
                    "latency_mean_s": row.get("latency_mean_s"),
                    "latency_p95_s": row.get("latency_p95_s"),
                    "rtf_mean": row.get("rtf_mean"),
                    "rtf_p95": row.get("rtf_p95"),
                    "throughput_qps": row.get("throughput_qps"),
                    "hop_p95_ms": stage.get("talker_to_code2wav_hop_p95_ms"),
                    "decode_p95_ms": stage.get("code2wav_decode_p95_ms"),
                    "top_stage": stage.get("top_stage"),
                },
                acceptance={
                    "completed_min": row.get("n"),
                    "accuracy_min": round(max(0.0, _float(row.get("accuracy")) - 0.02), 4),
                    "wer_corpus_max": round(min(0.08, wer + 0.02), 4),
                    "throughput_qps_min_for_same_hardware": round(qps * 0.85, 4),
                    "latency_p95_s_max_for_same_hardware": round(latency_p95 * 1.25, 4),
                    "hop_p95_ms_max": 30.0,
                    "decode_p95_ms_max": 35.0,
                },
                replacement=(
                    "Can replace checkpoint numbers only when environment/image/model/data "
                    "match and all full-audit gates pass."
                ),
                evidence=[
                    str(row.get("result_json") or ""),
                    str(row.get("wer_json") or ""),
                    str(stage.get("profile_json") or ""),
                ],
                caveat=(
                    "c16 is accepted only as saturation evidence, not as a recommended serving point."
                    if c == 16
                    else ""
                ),
            )
        )
    return result


def _synthetic_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        scenario = str(row.get("scenario"))
        c = _int(row.get("concurrency"))
        rtf = _float(row.get("rtf_mean"))
        result.append(
            _metric_rule(
                rule_id=f"synthetic_{scenario}_c{c}",
                category="synthetic_speech",
                scope=f"{scenario} text input + speech output c={c}",
                baseline={
                    "target_chars": row.get("target_chars"),
                    "target_words": row.get("target_words"),
                    "audio_duration_mean_s": row.get("audio_duration_mean_s"),
                    "latency_mean_s": row.get("latency_mean_s"),
                    "latency_p95_s": row.get("latency_p95_s"),
                    "rtf_mean": row.get("rtf_mean"),
                    "rtf_p95": row.get("rtf_p95"),
                    "throughput_qps": row.get("throughput_qps"),
                },
                acceptance={
                    "target_chars_exact": row.get("target_chars"),
                    "target_words_exact": row.get("target_words"),
                    "completed_min": row.get("n"),
                    "rtf_mean_max": 1.0 if scenario == "long" and c == 8 else round(max(1.0, rtf * 1.35), 4),
                    "latency_p95_s_max_for_same_hardware": round(_float(row.get("latency_p95_s")) * 1.25, 4),
                },
                replacement=(
                    "Long c8 must stay faster than real time before the long-text "
                    "speech guardrail can remain a high-confidence claim."
                    if scenario == "long" and c == 8
                    else "Use as shape confirmation unless the full audit and environment match."
                ),
                evidence=[str(row.get("result_json") or "")],
            )
        )
    return result


def _vllm_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    original_c8 = _row_by_key(rows, "label", "vLLM-c8")
    for row in rows:
        label = str(row.get("label"))
        admission_p95 = _float(row.get("batch_admission_span_p95_ms"))
        runner_qps = _float(row.get("runner_qps"))
        category = "vllm_strict_baseline" if label == "vLLM-c4" else "vllm_c8_diagnostic"
        replacement = (
            "c4 can participate in strict warmed headline comparison when claims pass."
            if label == "vLLM-c4"
            else "Keep as offline diagnostic until online ingress and same-scope WER/ASR exist."
        )
        acceptance: dict[str, Any] = {
            "completed_min": 50,
            "diagnosis_expected": row.get("diagnosis"),
            "admission_p95_ms_reference": row.get("batch_admission_span_p95_ms"),
            "runner_qps_min_for_same_hardware": round(runner_qps * 0.80, 4),
        }
        if label == "vLLM-c8":
            acceptance.update(
                {
                    "prompt_feed_limited_must_remain_true_for_original_path": True,
                    "admission_p95_ms_min_for_prompt_feed_diagnosis": 20000.0,
                }
            )
        if label == "vLLM-c8-prebuild-w4":
            acceptance.update(
                {
                    "runner_qps_must_exceed_original_c8": round(
                        _float(original_c8.get("runner_qps")) * 1.05, 4
                    ),
                    "admission_p95_ms_must_be_below_original_c8": round(
                        _float(original_c8.get("batch_admission_span_p95_ms")) * 0.25,
                        1,
                    ),
                }
            )
        result.append(
            _metric_rule(
                rule_id=label.replace("=", "").replace(" ", "_").lower(),
                category=category,
                scope=label,
                baseline={
                    "runner_qps": row.get("runner_qps"),
                    "engine_qps": row.get("engine_qps"),
                    "runner_overhead_pct_wall": row.get("runner_overhead_pct_wall"),
                    "batch_admission_span_avg_ms": row.get("batch_admission_span_avg_ms"),
                    "batch_admission_span_p95_ms": row.get("batch_admission_span_p95_ms"),
                    "diagnosis": row.get("diagnosis"),
                    "prompt_feed_limited": row.get("prompt_feed_limited"),
                },
                acceptance=acceptance,
                replacement=replacement,
                evidence=[
                    str(row.get("result_json") or ""),
                    str(row.get("run_log") or ""),
                ],
                caveat=(
                    "Do not use c8 offline numbers as online serving parity."
                    if "c8" in label.lower()
                    else ""
                ),
            )
        )
    return result


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    tables = _load_json_optional(audit_dir / "tables_summary.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    stage_interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    runtime_contract = _load_json_optional(audit_dir / "runtime_image_contract.json")
    runtime_comparison = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    stage_ledger = _load_json_optional(audit_dir / "stage_boundary_bottleneck_ledger.json")
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")
    rerun_delta_triage = _load_json_optional(audit_dir / "rerun_delta_triage.json")
    collaborator_sheet = _read_text_optional(root / COLLABORATOR_RERUN_SHEET)

    stress_rows = _table(tables, "sglang_stress")
    stage_rows = _table(tables, "sglang_stage_breakdown")
    synthetic_rows = _table(tables, "synthetic_speech")
    preproc_rows = _table(tables, "preprocessing_concurrency")
    vllm_rows = _table(tables, "vllm_admission_diagnosis")

    c8 = _row_by_key(stress_rows, "concurrency", 8)
    c16 = _row_by_key(stress_rows, "concurrency", 16)
    preproc1 = _row_by_key(preproc_rows, "setting", "preproc=1 baseline")
    preproc2 = _row_by_key(preproc_rows, "setting", "preproc=2")
    long_c8 = next(
        (
            row
            for row in synthetic_rows
            if row.get("scenario") == "long" and row.get("concurrency") == 8
        ),
        {},
    )
    vllm_c8 = _row_by_key(vllm_rows, "label", "vLLM-c8")
    vllm_w4 = _row_by_key(vllm_rows, "label", "vLLM-c8-prebuild-w4")
    stage_summary = stage_interactions.get("summary", {})
    runtime_summary = runtime_contract.get("summary", {})
    runtime_comparison_summary = runtime_comparison.get("summary", {})
    final_summary = final_readiness.get("summary", {})
    stage_ledger_summary = stage_ledger.get("summary", {})
    acceptance_summary = acceptance.get("summary", {})
    rerun_delta_summary = rerun_delta_triage.get("summary", {})
    rerun_delta_rows = [
        row for row in rerun_delta_triage.get("rows", []) if isinstance(row, dict)
    ]
    silent_replacement_rows = [
        row
        for row in rerun_delta_rows
        if "silent replacement" in str(row.get("likely_stage", ""))
        or "公开 Markdown" in str(row.get("symptom", ""))
    ]
    measurement_protocol_rows = [
        row
        for row in rerun_delta_rows
        if "skip-first" in str(row.get("symptom", ""))
        or "warmup" in str(row.get("likely_stage", ""))
    ]
    regeneration_rows = [
        row
        for row in rerun_delta_rows
        if "regeneration/full audit" in str(row.get("symptom", ""))
        or "evidence regeneration" in str(row.get("likely_stage", ""))
    ]
    return_evidence = _return_evidence_records(root, collaborator_sheet)
    missing_return_current = [
        record["relative_path"]
        for record in return_evidence
        if not record["exists_in_current_package"]
    ]
    missing_return_sheet = [
        record["relative_path"]
        for record in return_evidence
        if not record["listed_in_collaborator_sheet"]
    ]
    command_return_matrix, command_return_diagnostics = _command_return_matrix(
        repro_manifest, return_evidence
    )

    metric_rules: list[dict[str, Any]] = []
    metric_rules.extend(_sglang_stress_rules(stress_rows, stage_rows))
    metric_rules.extend(_synthetic_rules(synthetic_rows))
    metric_rules.extend(_vllm_rules(vllm_rows))
    metric_rules.extend(
        [
            _metric_rule(
                rule_id="sglang_c8_peak_shape",
                category="cross_condition_shape",
                scope="SGLang c8 remains the best current high-concurrency point",
                baseline={
                    "c8_qps": c8.get("throughput_qps"),
                    "c16_qps": c16.get("throughput_qps"),
                    "c8_latency_mean_s": c8.get("latency_mean_s"),
                    "c16_latency_mean_s": c16.get("latency_mean_s"),
                },
                acceptance={
                    "c8_qps_should_be_highest_or_within_noise": True,
                    "c16_latency_mean_should_exceed_c8": True,
                    "c16_is_not_default_serving_point": True,
                },
                replacement="If c16 beats c8 materially, rerun stage ledger and update serving-window claims before replacing numbers.",
                evidence=[str(c8.get("result_json") or ""), str(c16.get("result_json") or "")],
            ),
            _metric_rule(
                rule_id="preprocessing_parallelism_antirecipe",
                category="negative_optimization_guardrail",
                scope="PREPROCESSING_MAX_CONCURRENCY=2 remains a measured anti-recipe",
                baseline={
                    "preproc1_qps": preproc1.get("throughput_qps"),
                    "preproc2_qps": preproc2.get("throughput_qps"),
                    "qps_delta_pct": (
                        (_float(preproc2.get("throughput_qps")) / _float(preproc1.get("throughput_qps")) - 1.0)
                        * 100.0
                        if _float(preproc1.get("throughput_qps")) > 0
                        else None
                    ),
                },
                acceptance={
                    "preproc2_should_not_be_promoted_unless_qps_exceeds_baseline": True,
                    "preproc2_current_qps_max_for_antirecipe": round(
                        _float(preproc1.get("throughput_qps")) * 0.90, 4
                    ),
                },
                replacement="Do not change the recommended recipe to preproc=2 unless a new placement/admission design beats baseline with quality intact.",
                evidence=[
                    str(audit_dir / "tables_summary.json"),
                    str(audit_dir / "stage_interaction_summary.json"),
                ],
            ),
            _metric_rule(
                rule_id="stage_connection_flags",
                category="stage_connection_guardrail",
                scope="Stage connections and bottleneck attribution",
                baseline={
                    "total_interactions": stage_summary.get("total_interactions"),
                    "status_counts": stage_summary.get("status_counts"),
                },
                acceptance={
                    "sglang_talker_to_code2wav_healthy": True,
                    "sglang_code2wav_decode_not_bottleneck": True,
                    "vllm_original_c8_prompt_feed_limited": True,
                    "preprocessing_parallelism_regresses": True,
                },
                replacement="Any changed flag requires rebuilding stage latency budget, boundary ledger, confidence ledger, and final readiness.",
                evidence=[str(audit_dir / "stage_interaction_summary.json")],
            ),
        ]
    )

    checks: list[Check] = [
        Check(
            "final readiness summary is available",
            _status(
                _int(final_summary.get("checks_total")) >= 28
                and "hard_gates" in final_summary
            ),
            (
                "final_readiness="
                f"ready={final_summary.get('ready')}, "
                f"checks={final_summary.get('checks_passed')}/"
                f"{final_summary.get('checks_total')}, "
                f"required_failures={final_summary.get('required_failures')}"
            ),
        ),
        Check(
            "runtime image contract is ready",
            _status(
                bool(runtime_summary.get("ready"))
                and _int(runtime_summary.get("checks_total")) >= 12
                and _int(runtime_summary.get("required_failures"), 1) == 0
            ),
            f"runtime_image_contract={runtime_summary}",
        ),
        Check(
            "acceptance matrix remains green",
            _status(
                bool(acceptance_summary.get("ready"))
                and _int(acceptance_summary.get("rows_passed"))
                == _int(acceptance_summary.get("rows_total"), -1)
                and _int(acceptance_summary.get("rows_total")) >= 17
            ),
            f"acceptance={acceptance_summary}",
        ),
        Check(
            "stress rows cover c1/c2/c4/c8/c16",
            _status({1, 2, 4, 8, 16}.issubset({int(row.get("concurrency")) for row in stress_rows})),
            f"stress_concurrency={[row.get('concurrency') for row in stress_rows]}",
        ),
        Check(
            "synthetic rows cover short/long c1/c4/c8",
            _status(
                {
                    ("short", 1),
                    ("short", 4),
                    ("short", 8),
                    ("long", 1),
                    ("long", 4),
                    ("long", 8),
                }.issubset(
                    {
                        (str(row.get("scenario")), int(row.get("concurrency")))
                        for row in synthetic_rows
                    }
                )
            ),
            f"synthetic_rows={[(row.get('scenario'), row.get('concurrency')) for row in synthetic_rows]}",
        ),
        Check(
            "c8 is current SGLang throughput peak",
            _status(
                bool(c8)
                and _float(c8.get("throughput_qps"))
                >= max(_float(row.get("throughput_qps")) for row in stress_rows)
            ),
            f"c8_qps={c8.get('throughput_qps')}, all_qps={[row.get('throughput_qps') for row in stress_rows]}",
        ),
        Check(
            "c16 remains saturation boundary",
            _status(
                bool(c8)
                and bool(c16)
                and _float(c16.get("throughput_qps")) < _float(c8.get("throughput_qps"))
                and _float(c16.get("latency_mean_s")) > _float(c8.get("latency_mean_s"))
            ),
            f"c8={c8}, c16={c16}",
        ),
        Check(
            "long c8 remains faster than real time",
            _status(bool(long_c8) and _float(long_c8.get("rtf_mean")) < 1.0),
            f"long_c8={long_c8}",
        ),
        Check(
            "vLLM original c8 remains prompt-feed diagnostic",
            _status(
                bool(vllm_c8.get("prompt_feed_limited"))
                and str(vllm_c8.get("diagnosis")) == "prompt_feed_limited"
            ),
            f"vllm_c8={vllm_c8}",
        ),
        Check(
            "vLLM prebuild w4 improves offline runner QPS",
            _status(
                _float(vllm_w4.get("runner_qps")) > _float(vllm_c8.get("runner_qps"))
                and _float(vllm_w4.get("batch_admission_span_p95_ms"))
                < _float(vllm_c8.get("batch_admission_span_p95_ms"))
            ),
            f"vllm_c8={vllm_c8}, vllm_w4={vllm_w4}",
        ),
        Check(
            "stage interaction guardrails are true",
            _status(
                bool(stage_summary.get("sglang_talker_to_code2wav_healthy"))
                and bool(stage_summary.get("sglang_code2wav_decode_not_bottleneck"))
                and bool(stage_summary.get("vllm_original_c8_prompt_feed_limited"))
                and bool(stage_summary.get("preprocessing_parallelism_regresses"))
            ),
            f"stage_interactions={stage_summary}",
        ),
        Check(
            "stage boundary ledger is ready",
            _status(
                bool(stage_ledger_summary.get("ready"))
                and _int(stage_ledger_summary.get("ledger_rows")) >= 37
                and _int(stage_ledger_summary.get("required_failures"), 1) == 0
            ),
            f"stage_boundary_ledger={stage_ledger_summary}",
        ),
        Check(
            "strict c4 headline claims are machine-verified",
            _status(
                _claim_passed(claims, "SGLang warmed c4 beats vLLM warmed c4 latency/RTF")
                and _claim_passed(claims, "SGLang warmed c4 preserves accuracy/WER vs vLLM")
            ),
            "claims_verification strict c4 latency/RTF and quality checks",
        ),
        Check(
            "rerun rule inventory is complete",
            _status(len(metric_rules) >= 18),
            f"rules={len(metric_rules)}",
        ),
        Check(
            "collaborator return evidence contract is complete",
            _status(
                len(return_evidence) >= len(CRITICAL_RETURN_EVIDENCE)
                and not missing_return_current
                and not missing_return_sheet
                and bool(runtime_comparison_summary.get("ready"))
                and runtime_comparison_summary.get("allowed_cross_runtime_headline")
                == "warmed c=4 only"
                and runtime_comparison_summary.get("vllm_c8_contract")
                == "offline_diagnostic_not_online_parity"
            ),
            (
                f"return_files={len(return_evidence)}, "
                f"missing_current={missing_return_current}, "
                f"missing_sheet={missing_return_sheet}, "
                f"runtime_comparison={runtime_comparison_summary}"
            ),
        ),
        Check(
            "command return evidence matrix is complete",
            _status(
                len(command_return_matrix)
                >= len(COMMAND_RETURN_EVIDENCE_REQUIREMENTS)
                and not command_return_diagnostics["missing_command_ids"]
                and not command_return_diagnostics["missing_required_return_files"]
                and not command_return_diagnostics[
                    "commands_without_raw_or_gate_evidence"
                ]
            ),
            (
                f"command_rows={len(command_return_matrix)}, "
                f"missing_commands={command_return_diagnostics['missing_command_ids']}, "
                f"missing_required_files={command_return_diagnostics['missing_required_return_files']}, "
                f"without_evidence={command_return_diagnostics['commands_without_raw_or_gate_evidence']}"
            ),
        ),
        Check(
            "silent replacement and protocol drift guards are documented",
            _status(
                bool(rerun_delta_summary.get("ready"))
                and _int(rerun_delta_summary.get("required_failures"), 1) == 0
                and _int(rerun_delta_summary.get("rows_total")) >= 19
                and _int(rerun_delta_summary.get("checks_total")) >= 8
                and bool(silent_replacement_rows)
                and bool(measurement_protocol_rows)
                and bool(regeneration_rows)
            ),
            (
                f"rerun_delta_triage={rerun_delta_summary}, "
                f"silent_rows={len(silent_replacement_rows)}, "
                f"protocol_rows={len(measurement_protocol_rows)}, "
                f"regeneration_rows={len(regeneration_rows)}"
            ),
        ),
    ]

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "rules_total": len(metric_rules),
            "sglang_stress_rules": len([r for r in metric_rules if r["category"] == "sglang_stress"]),
            "synthetic_rules": len([r for r in metric_rules if r["category"] == "synthetic_speech"]),
            "vllm_rules": len([r for r in metric_rules if r["category"].startswith("vllm")]),
            "return_evidence_files": len(return_evidence),
            "return_evidence_missing_current": len(missing_return_current),
            "return_evidence_missing_sheet": len(missing_return_sheet),
            "return_evidence_command_rows": len(command_return_matrix),
            "return_evidence_command_missing": len(
                command_return_diagnostics["missing_command_ids"]
            ),
            "return_evidence_command_file_gaps": len(
                command_return_diagnostics["missing_required_return_files"]
            ),
            "replacement_scope": "same hardware/image/model/data plus all gates green",
            "default_decision": "confirm_shape_unless_environment_and_all_gates_match",
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "rules": metric_rules,
        "return_evidence_contract": {
            "required_files_total": len(return_evidence),
            "missing_current": missing_return_current,
            "missing_from_collaborator_sheet": missing_return_sheet,
            "records": return_evidence,
            "command_matrix_rows_total": len(command_return_matrix),
            "command_matrix_diagnostics": command_return_diagnostics,
            "command_matrix": command_return_matrix,
            "replacement_boundary": (
                "A collaborator rerun cannot replace headline numbers unless these "
                "machine evidence files are returned together with raw SGLang/vLLM "
                "result artifacts and all gates remain green."
            ),
        },
        "replacement_decision_matrix": [
            {
                "condition": "same 8x H20, same image digests, same model/data, full audit green",
                "decision": "may replace report numbers after reviewer signoff",
            },
            {
                "condition": "full audit green but hardware or image differs",
                "decision": "external validation appendix only; do not replace headline",
            },
            {
                "condition": "claims, acceptance, confidence, stage ledger, or runtime image contract fails",
                "decision": "do not replace; diagnose the failed gate first",
            },
            {
                "condition": "raw benchmark changed but regeneration/full audit or share seal was not rerun",
                "decision": "do not replace; this is a silent-replacement risk",
            },
            {
                "condition": "vLLM c8 prebuild improves but online ingress/WER is absent",
                "decision": "update offline diagnostic only; do not claim online parity",
            },
        ],
        "source_files": {
            "tables_summary": str(audit_dir / "tables_summary.json"),
            "acceptance_matrix": str(audit_dir / "acceptance_matrix.json"),
            "claims_verification": str(audit_dir / "claims_verification.json"),
            "stage_interaction_summary": str(audit_dir / "stage_interaction_summary.json"),
            "stage_boundary_bottleneck_ledger": str(audit_dir / "stage_boundary_bottleneck_ledger.json"),
            "runtime_comparison_contract": str(audit_dir / "runtime_comparison_contract.json"),
            "runtime_image_contract": str(audit_dir / "runtime_image_contract.json"),
            "repro_command_manifest": str(audit_dir / "repro_command_manifest.json"),
            "rerun_delta_triage": str(audit_dir / "rerun_delta_triage.json"),
            "final_readiness": str(audit_dir / "final_readiness_audit.json"),
            "collaborator_rerun_validation_sheet": str(root / COLLABORATOR_RERUN_SHEET),
        },
    }


def _gate_table(checks: list[dict[str, Any]]) -> list[str]:
    lines = ["| Status | Required | Gate | Evidence |", "| --- | --- | --- | --- |"]
    for check in checks:
        evidence = str(check["evidence"]).replace("|", "\\|")
        required = "yes" if check["required"] else "no"
        lines.append(
            f"| {check['status']} | {required} | {check['name']} | {evidence} |"
        )
    return lines


def _rule_summary_table(rules: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rule | Category | Scope | Baseline | Acceptance / Replacement |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rule in rules:
        baseline = rule.get("baseline", {})
        baseline_bits = []
        for key in [
            "throughput_qps",
            "latency_p95_s",
            "rtf_mean",
            "wer_corpus",
            "diagnosis",
        ]:
            if key in baseline:
                baseline_bits.append(f"{key}={baseline[key]}")
        if not baseline_bits:
            for key, value in list(baseline.items())[:3]:
                baseline_bits.append(f"{key}={value}")
        acceptance = rule.get("acceptance", {})
        accept_bits = []
        for key, value in list(acceptance.items())[:4]:
            accept_bits.append(f"{key}={value}")
        lines.append(
            "| "
            f"`{rule['id']}` | "
            f"{rule['category']} | "
            f"{rule['scope']} | "
            f"{'; '.join(baseline_bits)} | "
            f"{'; '.join(accept_bits)}; {rule['headline_replacement_rule']} |"
        )
    return lines


def _return_evidence_table(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Required evidence | Current file | Listed in sheet | Purpose |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        purpose = str(record["purpose"]).replace("|", "\\|")
        lines.append(
            "| "
            f"`{record['relative_path']}` | "
            f"{record['exists_in_current_package']} | "
            f"{record['listed_in_collaborator_sheet']} | "
            f"{purpose} |"
        )
    return lines


def _short_list(values: list[Any], *, limit: int = 3) -> str:
    items = [str(value) for value in values if str(value or "").strip()]
    shown = items[:limit]
    suffix = f", ... (+{len(items) - limit})" if len(items) > limit else ""
    return ", ".join(shown) + suffix


def _command_return_matrix_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Command ID | Phase | Raw/runtime artifacts | Required return files | Ready |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        raw_artifacts = _short_list(row.get("raw_or_runtime_artifacts", []), limit=2)
        required_files = _short_list(row.get("required_return_files", []), limit=3)
        lines.append(
            "| "
            f"`{row['command_id']}` | "
            f"{row['phase']} | "
            f"{raw_artifacts} | "
            f"{required_files} | "
            f"{row['ready']} |"
        )
    return lines


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = [
        "# Qwen3.5-Omni 复跑验收阈值合同",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份 contract 把当前 checkpoint 的核心数字转换成合作方复跑验收阈值。",
        "它不替代完整报告；用途是判断复跑结果是“确认当前形态”、",
        "“只能作为外部附录”，还是“具备替换 headline 数字的资格”。",
        "",
        "## 1. 当前状态",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| Rules | {summary['rules_total']} |",
        f"| SGLang stress rules | {summary['sglang_stress_rules']} |",
        f"| Synthetic rules | {summary['synthetic_rules']} |",
        f"| vLLM rules | {summary['vllm_rules']} |",
        f"| Return evidence files | {summary['return_evidence_files']} |",
        f"| Command return evidence rows | {summary['return_evidence_command_rows']} |",
        f"| Replacement scope | `{summary['replacement_scope']}` |",
        f"| Default decision | `{summary['default_decision']}` |",
        "",
        "## 2. Gate 明细",
        "",
    ]
    lines.extend(_gate_table(payload["checks"]))
    lines.extend(
        [
            "",
            "## 3. 合作方回传证据合同",
            "",
            "如果复跑方希望进入 headline 数字替换评审，除 raw SGLang/vLLM artifact 外，",
            "必须把下列机器证据一起回传；缺任一项时只能作为附录趋势或需先补证据。",
            "",
        ]
    )
    lines.extend(_return_evidence_table(payload["return_evidence_contract"]["records"]))
    lines.extend(
        [
            "",
            "## 4. 命令到回传证据矩阵",
            "",
            "这张表把关键复跑命令和必须随同回传的 gate 文件绑定起来；",
            "raw/runtime artifact 用来复查原始输出，required return files 用来决定能否替换报告数字。",
            "",
        ]
    )
    lines.extend(
        _command_return_matrix_table(
            payload["return_evidence_contract"]["command_matrix"]
        )
    )
    lines.extend(
        [
            "",
            "## 5. 复跑阈值表",
            "",
        ]
    )
    lines.extend(_rule_summary_table(payload["rules"]))
    lines.extend(
        [
            "",
            "## 6. 是否可以替换报告数字",
            "",
            "| Condition | Decision |",
            "| --- | --- |",
        ]
    )
    for row in payload["replacement_decision_matrix"]:
        lines.append(f"| {row['condition']} | {row['decision']} |")
    lines.extend(
        [
            "",
            "## 7. 使用方式",
            "",
            "1. 合作方先按 handoff runbook 和 reproduction checklist 复跑。",
            "2. 复跑后重跑 full audit、preflight、manifest、final readiness 和本 contract，并按第 3 节回传机器证据。",
            "3. 若硬件、image digest、模型、数据路径任一不同，结果只能作为外部复核附录。",
            "4. 若要替换 headline，必须同时满足本 contract、claims、acceptance、confidence、stage ledger 和 runtime image contract。",
            "5. vLLM c=8 prebuild w4 即使变好，也只能更新 offline diagnostic，不能自动升级为 online parity。",
            "6. raw benchmark、公开 Markdown 或图表任一被替换时，必须同步重跑 regeneration/full audit、share bundle、tarball seal 和 receiver quickcheck，否则按 silent-replacement 风险处理。",
            "",
            "## 8. 机器证据",
            "",
        ]
    )
    for name, path in payload["source_files"].items():
        lines.append(f"- {name}：`{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni collaborator rerun acceptance contract."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_payload(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(payload), encoding="utf-8")
    _save_json(payload, json_output)
    print(
        "Rerun acceptance contract written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']} "
        f"rules={payload['summary']['rules_total']}"
    )
    print(f"Rerun acceptance contract JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
