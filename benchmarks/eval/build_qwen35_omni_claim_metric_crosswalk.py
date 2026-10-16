# SPDX-License-Identifier: Apache-2.0
"""Build claim-to-metric crosswalk for Qwen3.5-Omni defense evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "claim_metric_crosswalk.json"

DEFENSE_CLAIMS = AUDIT_DIR / "defense_claim_matrix.json"
METRIC_PROVENANCE = AUDIT_DIR / "metric_provenance_index.json"
REPRO_COMMANDS = AUDIT_DIR / "repro_command_manifest.json"

CLAIM_METRIC_ROW_IDS = {
    "sglang_warmed_c4_beats_optimized_vllm": [
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
    "vllm_baseline_is_optimized": [
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
    "sglang_c8_current_high_concurrency_peak": [
        "sglang_videoamme_c4_throughput_qps",
        "sglang_videoamme_c8_throughput_qps",
        "sglang_videoamme_c16_throughput_qps",
        "sglang_videoamme_c16_vs_c8_qps_delta_pct",
        "stage_drilldown_budget-sglang_videoamme_budget-04",
        "stage_drilldown_budget-sglang_videoamme_budget-05",
        "acceptance_row_05_sglang_videoamme_stress",
        "acceptance_row_06_sglang_videoamme_stress",
    ],
    "short_and_long_tts_are_covered": [
        "synthetic_long_c8_target_words",
        "synthetic_long_c8_latency_mean_s",
        "synthetic_long_c8_rtf_mean",
        "synthetic_long_c8_rtf_p95",
        "synthetic_long_c8_audio_throughput_s_per_s",
        "stage_drilldown_budget-synthetic_speech_budget-03",
        "stage_drilldown_budget-synthetic_speech_budget-06",
        "acceptance_row_07_sglang_synthetic_speech",
        "acceptance_row_08_sglang_synthetic_speech",
        "acceptance_row_09_sglang_synthetic_speech",
        "acceptance_row_10_sglang_synthetic_speech",
        "acceptance_row_11_sglang_synthetic_speech",
        "acceptance_row_12_sglang_synthetic_speech",
    ],
    "stage_handoff_is_not_stalled": [
        "stage_drilldown_summary_rows_total",
        "stage_drilldown_summary_stage_routes_total",
        "stage_drilldown_stage-boundary-02",
        "stage_drilldown_stage-boundary-05",
        "stage_drilldown_stage-boundary-08",
        "stage_drilldown_stage-boundary-11",
        "stage_drilldown_stage-boundary-14",
        "stage_drilldown_stage-boundary-17",
        "stage_drilldown_stage-boundary-22",
        "acceptance_row_17_stage_connection_health",
    ],
    "code2wav_decode_not_current_compute_bottleneck": [
        "stage_drilldown_stage-boundary-03",
        "stage_drilldown_stage-boundary-06",
        "stage_drilldown_stage-boundary-09",
        "stage_drilldown_stage-boundary-12",
        "stage_drilldown_stage-boundary-15",
        "stage_drilldown_stage-boundary-17",
        "stage_drilldown_stage-boundary-18",
        "stage_drilldown_stage-boundary-19",
        "stage_drilldown_stage-boundary-20",
        "stage_drilldown_stage-boundary-21",
        "stage_drilldown_stage-boundary-22",
        "acceptance_row_17_stage_connection_health",
    ],
    "raising_preprocessing_concurrency_is_negative_recipe": [
        "stage_drilldown_stage-boundary-16",
        "acceptance_row_15_negative_optimization",
        "acceptance_row_16_negative_optimization",
        "stage_drilldown_budget-sglang_videoamme_budget-04",
        "sglang_videoamme_c8_throughput_qps",
    ],
    "vllm_c8_prebuild_w4_is_offline_diagnostic": [
        "vllm_c8_original_admission_batch_admission_span_avg_ms",
        "vllm_c8_original_admission_batch_admission_span_p95_ms",
        "vllm_c8_original_admission_prompt_feed_limited",
        "vllm_c8_prebuild_w4_admission_batch_admission_span_avg_ms",
        "vllm_c8_prebuild_w4_admission_batch_admission_span_p95_ms",
        "vllm_c8_prebuild_w4_runner_qps",
        "vllm_c8_prebuild_w4_engine_qps",
        "stage_drilldown_stage-boundary-31",
        "stage_drilldown_stage-boundary-37",
        "stage_drilldown_budget-vllm_offline_budget-02",
        "stage_drilldown_budget-vllm_offline_budget-04",
        "acceptance_row_13_vllm_offline_diagnostic",
        "acceptance_row_14_vllm_offline_diagnostic",
    ],
    "wer_quality_not_traded_for_speed": [
        "strict_c4_sglang_accuracy",
        "strict_c4_sglang_wer_corpus",
        "strict_c4_vllm_accuracy",
        "strict_c4_vllm_wer_corpus",
        "sglang_videoamme_c1_accuracy",
        "sglang_videoamme_c1_wer_corpus",
        "sglang_videoamme_c4_accuracy",
        "sglang_videoamme_c4_wer_corpus",
        "sglang_videoamme_c8_accuracy",
        "sglang_videoamme_c8_wer_corpus",
        "sglang_videoamme_c16_accuracy",
        "sglang_videoamme_c16_wer_corpus",
        "acceptance_row_01_strict_runtime_comparison",
        "acceptance_row_05_sglang_videoamme_stress",
    ],
    "package_share_ready_with_boundaries": [
        "stage_drilldown_summary_rows_total",
        "stage_drilldown_summary_checks_passed",
        "stage_drilldown_summary_required_failures",
        "acceptance_row_17_stage_connection_health",
        "acceptance_row_01_strict_runtime_comparison",
        "acceptance_row_14_vllm_offline_diagnostic",
    ],
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


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


def _defense_rows(defense_claims: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("claim_id")): row
        for row in _as_list(defense_claims.get("rows"))
        if isinstance(row, dict) and row.get("claim_id")
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


def _reviewer_hook(claim_id: str) -> str:
    hooks = {
        "sglang_warmed_c4_beats_optimized_vllm": (
            "Use the strict_c4 rows to answer latency/RTF/quality questions with "
            "both SGLang and vLLM numbers side by side."
        ),
        "vllm_baseline_is_optimized": (
            "Use the vLLM diagnostic rows with runtime/optimization locks; do not "
            "promote c=8 online parity from offline rows."
        ),
        "sglang_c8_current_high_concurrency_peak": (
            "Show c=4/c=8/c=16 QPS plus the c16-vs-c8 delta before discussing "
            "the c=8 serving window."
        ),
        "short_and_long_tts_are_covered": (
            "Use acceptance rows for all short/long regimes and long-c8 numeric "
            "rows for the high-concurrency guardrail."
        ),
        "stage_handoff_is_not_stalled": (
            "Use stage boundary rows to separate handoff health from admission "
            "queue and stage-local compute pressure."
        ),
        "code2wav_decode_not_current_compute_bottleneck": (
            "Use code2wav boundary rows; the safe wording is not that code2wav is "
            "free, only that it is not the current compute bottleneck."
        ),
        "raising_preprocessing_concurrency_is_negative_recipe": (
            "Use the negative-optimization acceptance rows and boundary-16 to "
            "explain why naive widening is not the current optimization path."
        ),
        "vllm_c8_prebuild_w4_is_offline_diagnostic": (
            "Use original-vs-prebuild rows to show the diagnostic improvement and "
            "the remaining online parity gap."
        ),
        "wer_quality_not_traded_for_speed": (
            "Use WER/accuracy rows across strict c=4 and SGLang stress to defend "
            "that throughput was not bought by quality degradation."
        ),
        "package_share_ready_with_boundaries": (
            "Use this row as the send/no-send boundary: shareable with caveats, "
            "not a claim of universal parity or final goal completion."
        ),
    }
    return hooks.get(claim_id, "")


def build_claim_metric_crosswalk(root: Path) -> dict[str, Any]:
    root = root.resolve()
    defense_claims = _load_json_optional(root / DEFENSE_CLAIMS)
    metric_provenance = _load_json_optional(root / METRIC_PROVENANCE)
    repro_manifest = _load_json_optional(root / REPRO_COMMANDS)

    defense_by_id = _defense_rows(defense_claims)
    metric_by_id = _metric_rows(metric_provenance)
    command_ids = _command_ids(repro_manifest)

    rows: list[dict[str, Any]] = []
    missing_claim_ids: list[str] = []
    missing_metric_row_ids: list[str] = []
    for claim_id, metric_row_ids in CLAIM_METRIC_ROW_IDS.items():
        defense_row = defense_by_id.get(claim_id)
        if defense_row is None:
            missing_claim_ids.append(claim_id)
            continue
        missing_for_claim = [
            row_id for row_id in metric_row_ids if row_id not in metric_by_id
        ]
        missing_metric_row_ids.extend(missing_for_claim)
        metric_rows = [
            metric_by_id[row_id]
            for row_id in metric_row_ids
            if row_id in metric_by_id
        ]
        evidence_files = _normalize_paths(
            root,
            [
                DEFENSE_CLAIMS,
                METRIC_PROVENANCE,
                *list(_as_list(defense_row.get("machine_evidence"))),
                *[
                    evidence
                    for metric_row in metric_rows
                    for evidence in _as_list(metric_row.get("evidence_files"))
                ],
            ],
        )
        rerun_command_ids = _dedupe(
            [
                *list(_as_list(defense_row.get("rerun_command_ids"))),
                *[
                    command_id
                    for metric_row in metric_rows
                    for command_id in _as_list(metric_row.get("rerun_command_ids"))
                ],
                "build_claim_metric_crosswalk",
            ]
        )
        raw_artifacts = sorted(path for path in evidence_files if _is_raw_artifact(path))
        packaged_evidence = sorted(
            path for path in evidence_files if not _is_raw_artifact(path)
        )
        rows.append(
            {
                "claim_id": claim_id,
                "claim": defense_row.get("claim"),
                "allowed_wording": defense_row.get("allowed_wording"),
                "failure_decision": defense_row.get("failure_decision"),
                "reviewer_answer_hook": _reviewer_hook(claim_id),
                "metric_row_ids": metric_row_ids,
                "metric_rows": [_compact_metric_row(row) for row in metric_rows],
                "machine_evidence": evidence_files,
                "raw_artifacts": raw_artifacts,
                "packaged_evidence": packaged_evidence,
                "rerun_command_ids": rerun_command_ids,
                "missing_metric_row_ids": missing_for_claim,
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
    missing_evidence_files = sorted(
        evidence for evidence in evidence_files_all if not _path_exists(root, evidence)
    )
    missing_command_ids = sorted(
        command_id for command_id in command_refs if command_id not in command_ids
    )
    row_counts = {
        "claims_total": len(rows),
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
    }
    required_claims = set(CLAIM_METRIC_ROW_IDS)
    mapped_claims = {str(row.get("claim_id")) for row in rows}
    checks = {
        "defense_claim_matrix_ready": bool(_summary(defense_claims).get("ready"))
        and int(_summary(defense_claims).get("rows_total") or 0) >= 10,
        "metric_provenance_ready": bool(_summary(metric_provenance).get("ready"))
        and int(_summary(metric_provenance).get("rows_total") or 0) >= 150,
        "repro_command_registry_ready": bool(
            _summary(repro_manifest).get("required_command_ids_present")
        )
        and int(_summary(repro_manifest).get("commands_total") or 0) >= 52,
        "all_claims_mapped": required_claims.issubset(mapped_claims)
        and not missing_claim_ids,
        "claims_have_metric_rows": all(row.get("metric_rows") for row in rows),
        "metric_row_ids_present": not missing_metric_row_ids,
        "evidence_files_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
        "coverage_shape": row_counts["claims_total"] >= 10
        and row_counts["unique_metric_rows_total"] >= 60
        and row_counts["raw_artifacts_total"] >= 20
        and row_counts["command_refs_total"] >= 15,
    }
    required_failures = [name for name, ok in checks.items() if not ok]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            **row_counts,
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "Maps each external defense claim to exact metric provenance row IDs, "
                "machine evidence files, raw artifacts, and rerun command IDs."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "missing_claim_ids": missing_claim_ids,
            "missing_metric_row_ids": sorted(set(missing_metric_row_ids)),
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "required_failures": required_failures,
        },
        "source_summaries": {
            "defense_claim_matrix": _summary(defense_claims),
            "metric_provenance_index": _summary(metric_provenance),
            "repro_command_manifest": _summary(repro_manifest),
        },
        "rows": rows,
    }


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Claim Metric Crosswalk")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- claims: `{summary['claims_total']}`")
    print(f"- unique metric rows: `{summary['unique_metric_rows_total']}`")
    print(f"- raw artifacts: `{summary['raw_artifacts_total']}`")
    print(f"- command refs: `{summary['command_refs_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni claim-to-metric crosswalk JSON."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output
    if not output.is_absolute():
        output = root / output

    payload = build_claim_metric_crosswalk(root)
    _save_json(payload, output)
    _print_markdown(payload)

    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
