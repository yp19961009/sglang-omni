# SPDX-License-Identifier: Apache-2.0
"""Build metric-to-evidence provenance index for Qwen3.5-Omni reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "metric_provenance_index.json"

HEADLINE = AUDIT_DIR / "headline_scorecard.json"
ACCEPTANCE = AUDIT_DIR / "acceptance_matrix.json"
CLAIMS = AUDIT_DIR / "claims_verification.json"
STAGE_DRILLDOWN = AUDIT_DIR / "stage_drilldown_index.json"
REPRO_COMMANDS = AUDIT_DIR / "repro_command_manifest.json"
VLLM_ADMISSION = AUDIT_DIR / "vllm_admission_diagnosis.json"
TABLES = AUDIT_DIR / "tables_summary.json"
TAIL_CONFIDENCE = AUDIT_DIR / "tail_confidence_appendix.json"


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _dedupe_text(values: list[Any]) -> list[str]:
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
    return _dedupe_text([_rel_path(root, value) for value in values])


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


def _unit_for(metric_name: str) -> str:
    if metric_name.endswith("_s") or metric_name.endswith("_wall_s"):
        return "seconds"
    if metric_name.endswith("_ms"):
        return "milliseconds"
    if metric_name.endswith("_pct") or metric_name.endswith("_pct_wall"):
        return "percent"
    if metric_name in {"accuracy", "wer_corpus"}:
        return "ratio"
    if metric_name.endswith("_qps") or metric_name == "throughput_qps":
        return "requests_per_second"
    if "audio_throughput" in metric_name:
        return "audio_seconds_per_wall_second"
    if metric_name in {"n", "success", "requests", "successful", "concurrency"}:
        return "count"
    if metric_name in {"target_chars", "target_words"}:
        return "count"
    if metric_name.startswith("rtf"):
        return "ratio"
    return "value"


def _add_metric(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    row_id: str,
    category: str,
    claim_or_table: str,
    runtime: str,
    scenario: str,
    metric_name: str,
    metric_value: Any = None,
    metric_text: str | None = None,
    source_json: Path,
    source_pointer: str,
    evidence_files: list[Any],
    rerun_command_ids: list[str],
    report_routes: list[str] | None = None,
    concurrency: Any = None,
    stage_route: list[str] | None = None,
    metric_payload: dict[str, Any] | None = None,
    interpretation: str | None = None,
) -> None:
    rows.append(
        {
            "id": row_id,
            "category": category,
            "claim_or_table": claim_or_table,
            "runtime": runtime,
            "scenario": scenario,
            "concurrency": concurrency,
            "stage_route": stage_route or [],
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_unit": _unit_for(metric_name),
            "metric_text": metric_text,
            "metric_payload": metric_payload or {},
            "interpretation": interpretation,
            "source_json": source_json.as_posix(),
            "source_pointer": source_pointer,
            "evidence_files": _normalize_paths(
                root, [source_json, *evidence_files],
            ),
            "rerun_command_ids": _dedupe_text(rerun_command_ids),
            "report_routes": report_routes or [],
        }
    )


def _metric_row(
    root: Path,
    rows: list[dict[str, Any]],
    **kwargs: Any,
) -> None:
    _add_metric(root, rows, **kwargs)


def _acceptance_commands(row: dict[str, Any]) -> list[str]:
    regime = str(row.get("regime") or "")
    pressure = str(row.get("pressure") or "")
    if regime == "strict_runtime_comparison":
        return ["build_acceptance_matrix", "build_headline_scorecard", "verify_report_claims"]
    if regime == "sglang_videoamme_stress":
        return ["sglang_videoamme_stress", "sglang_recompute_wer", "build_acceptance_matrix"]
    if regime == "sglang_synthetic_speech":
        return ["sglang_synthetic_text_to_speech", "build_acceptance_matrix"]
    if regime == "vllm_offline_diagnostic":
        first = "vllm_c8_prebuild_w4" if "prebuild" in pressure else "vllm_c8_original"
        return [first, "diagnose_vllm_admission", "build_acceptance_matrix"]
    if regime == "stage_connection_health":
        return [
            "build_stage_interactions",
            "build_stage_boundary_bottleneck_ledger",
            "build_stage_drilldown_index",
            "build_acceptance_matrix",
        ]
    return ["build_acceptance_matrix"]


def _build_strict_c4_rows(
    root: Path, rows: list[dict[str, Any]], headline: dict[str, Any]
) -> None:
    strict = _as_dict(headline.get("strict_c4_comparison"))
    metric_names = [
        "n",
        "success",
        "accuracy",
        "latency_mean_s",
        "latency_p95_s",
        "rtf_mean",
        "rtf_p95",
        "wer_corpus",
    ]
    for runtime in ["sglang", "vllm"]:
        runtime_payload = _as_dict(strict.get(runtime))
        evidence = [runtime_payload.get("artifact"), CLAIMS]
        for metric_name in metric_names:
            _metric_row(
                root,
                rows,
                row_id=f"strict_c4_{runtime}_{metric_name}",
                category="headline_strict_c4",
                claim_or_table="strict warmed c=4 SGLang vs optimized vLLM headline",
                runtime=runtime,
                scenario=str(strict.get("scope") or "Video-AMME ci-50 warmed c=4"),
                concurrency=4,
                metric_name=metric_name,
                metric_value=runtime_payload.get(metric_name),
                source_json=HEADLINE,
                source_pointer=f"/strict_c4_comparison/{runtime}/{metric_name}",
                evidence_files=evidence,
                rerun_command_ids=[
                    "run_full_audit",
                    "build_headline_scorecard",
                    "verify_report_claims",
                ],
                report_routes=[
                    "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
                    "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
                ],
                interpretation=(
                    "Strict apples-to-apples warmed c=4 headline metric; raw rerun "
                    "artifact path is preserved even when the derived gate is audit-only."
                ),
            )
    for metric_name, value in _as_dict(strict.get("relative_sglang_lower_pct")).items():
        _metric_row(
            root,
            rows,
            row_id=f"strict_c4_sglang_lower_{metric_name}_pct",
            category="headline_strict_c4",
            claim_or_table="relative SGLang advantage over optimized vLLM",
            runtime="cross_runtime",
            scenario=str(strict.get("scope") or "Video-AMME ci-50 warmed c=4"),
            concurrency=4,
            metric_name=f"relative_sglang_lower_{metric_name}_pct",
            metric_value=value,
            source_json=HEADLINE,
            source_pointer=f"/strict_c4_comparison/relative_sglang_lower_pct/{metric_name}",
            evidence_files=[
                _as_dict(strict.get("sglang")).get("artifact"),
                _as_dict(strict.get("vllm")).get("artifact"),
                CLAIMS,
            ],
            rerun_command_ids=[
                "run_full_audit",
                "build_headline_scorecard",
                "verify_report_claims",
            ],
            report_routes=[
                "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md",
                "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
            ],
            interpretation="Positive value means SGLang is lower/better for the metric.",
        )


def _build_sglang_stress_rows(
    root: Path, rows: list[dict[str, Any]], headline: dict[str, Any]
) -> None:
    stress = _as_dict(headline.get("sglang_stress"))
    metric_names = [
        "n",
        "accuracy",
        "latency_mean_s",
        "latency_p95_s",
        "rtf_mean",
        "rtf_p95",
        "throughput_qps",
        "audio_throughput_s_per_s",
        "wer_corpus",
    ]
    for stress_row in _as_list(stress.get("rows")):
        if not isinstance(stress_row, dict):
            continue
        concurrency = stress_row.get("concurrency")
        evidence = [stress_row.get("result_json"), stress_row.get("wer_json"), ACCEPTANCE]
        for metric_name in metric_names:
            _metric_row(
                root,
                rows,
                row_id=f"sglang_videoamme_c{concurrency}_{metric_name}",
                category="sglang_videoamme_stress",
                claim_or_table="SGLang Video-AMME single/high concurrency sweep",
                runtime="sglang",
                scenario=f"Video-AMME ci-50 c={concurrency}",
                concurrency=concurrency,
                metric_name=metric_name,
                metric_value=stress_row.get(metric_name),
                source_json=HEADLINE,
                source_pointer=f"/sglang_stress/rows/concurrency={concurrency}/{metric_name}",
                evidence_files=evidence,
                rerun_command_ids=[
                    "sglang_videoamme_stress",
                    "sglang_recompute_wer",
                    "build_headline_scorecard",
                    "build_acceptance_matrix",
                ],
                report_routes=[
                    "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
                    "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
                ],
                interpretation="SGLang pressure-sweep metric under the locked optimized recipe.",
            )
    c16_delta = stress.get("c16_vs_c8_qps_delta_pct")
    if c16_delta is not None:
        peak = _as_dict(stress.get("throughput_peak"))
        _metric_row(
            root,
            rows,
            row_id="sglang_videoamme_c16_vs_c8_qps_delta_pct",
            category="sglang_videoamme_stress",
            claim_or_table="c=16 saturation boundary vs c=8 throughput peak",
            runtime="sglang",
            scenario="Video-AMME ci-50 c=16 compared with c=8",
            concurrency=16,
            metric_name="c16_vs_c8_qps_delta_pct",
            metric_value=c16_delta,
            source_json=HEADLINE,
            source_pointer="/sglang_stress/c16_vs_c8_qps_delta_pct",
            evidence_files=[peak.get("result_json"), ACCEPTANCE],
            rerun_command_ids=[
                "sglang_videoamme_stress",
                "build_headline_scorecard",
                "build_acceptance_matrix",
            ],
            report_routes=[
                "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
            ],
            interpretation="Negative value supports treating c=16 as saturation boundary, not default serving point.",
        )


def _build_synthetic_rows(
    root: Path, rows: list[dict[str, Any]], headline: dict[str, Any]
) -> None:
    synthetic = _as_dict(headline.get("synthetic_long_c8"))
    metric_names = [
        "n",
        "target_chars",
        "target_words",
        "audio_duration_mean_s",
        "latency_mean_s",
        "latency_p95_s",
        "rtf_mean",
        "rtf_p95",
        "throughput_qps",
        "audio_throughput_s_per_s",
    ]
    for metric_name in metric_names:
        _metric_row(
            root,
            rows,
            row_id=f"synthetic_long_c8_{metric_name}",
            category="synthetic_speech",
            claim_or_table="long text-to-speech guardrail at high concurrency",
            runtime="sglang",
            scenario="synthetic long text -> speech c=8",
            concurrency=8,
            metric_name=metric_name,
            metric_value=synthetic.get(metric_name),
            source_json=HEADLINE,
            source_pointer=f"/synthetic_long_c8/{metric_name}",
            evidence_files=[synthetic.get("result_json"), ACCEPTANCE, TABLES],
            rerun_command_ids=[
                "sglang_synthetic_text_to_speech",
                "build_headline_scorecard",
                "build_acceptance_matrix",
                "build_report_tables",
            ],
            report_routes=[
                "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
            ],
            interpretation="Long speech guardrail metric; rtf_mean below 1.0 means faster than realtime.",
        )


def _build_vllm_rows(
    root: Path, rows: list[dict[str, Any]], headline: dict[str, Any]
) -> None:
    diagnostics = _as_dict(headline.get("vllm_c8_diagnostics"))
    metric_names = [
        "requests",
        "successful",
        "wall_time_s",
        "runner_wall_time_s",
        "engine_wall_time_s",
        "prompt_build_wall_s",
        "runner_overhead_pct_wall",
        "engine_overhead_pct_wall",
        "wall_qps",
        "runner_qps",
        "engine_qps",
        "batch_max_qps",
        "wall_audio_throughput_s_per_s",
        "engine_audio_throughput_s_per_s",
    ]
    command_map = {
        "original": ["vllm_c8_original", "diagnose_vllm_admission", "build_headline_scorecard"],
        "prebuild_w1": ["diagnose_vllm_admission", "build_headline_scorecard"],
        "prebuild_w4": ["vllm_c8_prebuild_w4", "diagnose_vllm_admission", "build_headline_scorecard"],
    }
    for case in ["original", "prebuild_w1", "prebuild_w4"]:
        payload = _as_dict(diagnostics.get(case))
        if not payload:
            continue
        admission = _as_dict(payload.get("admission_diagnosis"))
        evidence = [
            payload.get("path"),
            admission.get("result_json"),
            admission.get("run_log"),
            VLLM_ADMISSION,
        ]
        for metric_name in metric_names:
            _metric_row(
                root,
                rows,
                row_id=f"vllm_c8_{case}_{metric_name}",
                category="vllm_offline_diagnostic",
                claim_or_table="vLLM c=8 offline diagnostic and prebuild comparison",
                runtime="vllm",
                scenario=str(payload.get("label") or f"vLLM-c8-{case}"),
                concurrency=payload.get("concurrency"),
                metric_name=metric_name,
                metric_value=payload.get(metric_name),
                source_json=HEADLINE,
                source_pointer=f"/vllm_c8_diagnostics/{case}/{metric_name}",
                evidence_files=evidence,
                rerun_command_ids=[
                    *command_map.get(case, []),
                    "summarize_vllm_log_stages",
                ],
                report_routes=[
                    "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
                    "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
                ],
                interpretation=(
                    "vLLM c=8 is retained as an optimized offline diagnostic; it is not "
                    "used as online serving parity in the current package."
                ),
            )
        for metric_name in [
            "batch_admission_span_avg_ms",
            "batch_admission_span_p95_ms",
            "prompt_feed_limited",
            "engine_boundaries_clean",
        ]:
            if metric_name not in admission:
                continue
            _metric_row(
                root,
                rows,
                row_id=f"vllm_c8_{case}_admission_{metric_name}",
                category="vllm_offline_diagnostic",
                claim_or_table="vLLM c=8 admission diagnosis",
                runtime="vllm",
                scenario=str(payload.get("label") or f"vLLM-c8-{case}"),
                concurrency=payload.get("concurrency"),
                metric_name=metric_name,
                metric_value=admission.get(metric_name),
                source_json=HEADLINE,
                source_pointer=f"/vllm_c8_diagnostics/{case}/admission_diagnosis/{metric_name}",
                evidence_files=evidence,
                rerun_command_ids=[
                    *command_map.get(case, []),
                    "summarize_vllm_log_stages",
                ],
                report_routes=[
                    "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
                    "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
                ],
                interpretation=str(admission.get("diagnosis") or ""),
            )
    for metric_name, value in _as_dict(diagnostics.get("w4_vs_w1")).items():
        _metric_row(
            root,
            rows,
            row_id=f"vllm_c8_w4_vs_w1_{metric_name}",
            category="vllm_offline_diagnostic",
            claim_or_table="vLLM c=8 prebuild w4 improvement over prebuild w1",
            runtime="vllm",
            scenario="vLLM-c8-prebuild-w4 vs vLLM-c8-prebuild-w1",
            concurrency=8,
            metric_name=metric_name,
            metric_value=value,
            source_json=HEADLINE,
            source_pointer=f"/vllm_c8_diagnostics/w4_vs_w1/{metric_name}",
            evidence_files=[
                _as_dict(diagnostics.get("prebuild_w1")).get("path"),
                _as_dict(diagnostics.get("prebuild_w4")).get("path"),
                VLLM_ADMISSION,
            ],
            rerun_command_ids=[
                "vllm_c8_prebuild_w4",
                "diagnose_vllm_admission",
                "build_headline_scorecard",
            ],
            report_routes=[
                "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
            ],
            interpretation="Shows prebuild w4 is the strongest current vLLM offline diagnostic recipe.",
        )


def _tail_runtime(case_id: str) -> str:
    if case_id.startswith(("strict_sglang", "sglang_", "synthetic_")):
        return "sglang"
    if case_id.startswith(("strict_vllm", "vllm_")):
        return "vllm"
    return "mixed"


def _build_tail_confidence_rows(
    root: Path, rows: list[dict[str, Any]], tail_confidence: dict[str, Any]
) -> None:
    summary = _summary(tail_confidence)
    summary_metrics = [
        "strict_c4_sglang_latency_p95_s",
        "strict_c4_vllm_latency_p95_s",
        "strict_c4_sglang_rtf_p95",
        "strict_c4_vllm_rtf_p95",
        "sglang_c8_qps",
        "sglang_c16_qps",
        "long_c8_rtf_p95",
        "vllm_w4_latency_p95_s",
    ]
    for metric_name in summary_metrics:
        _metric_row(
            root,
            rows,
            row_id=f"tail_confidence_summary_{metric_name}",
            category="tail_confidence",
            claim_or_table="per-sample tail-confidence appendix summary gate",
            runtime="mixed",
            scenario="strict c4, SGLang pressure, synthetic speech, and vLLM diagnostics",
            metric_name=metric_name,
            metric_value=summary.get(metric_name),
            source_json=TAIL_CONFIDENCE,
            source_pointer=f"/summary/{metric_name}",
            evidence_files=[TAIL_CONFIDENCE],
            rerun_command_ids=["build_tail_confidence_appendix"],
            report_routes=[
                "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
            ],
            interpretation="Tail-confidence gate metric derived from per-sample raw JSON.",
        )

    for index, case_row in enumerate(_as_list(tail_confidence.get("rows"))):
        if not isinstance(case_row, dict):
            continue
        case_id = str(case_row.get("case_id") or f"row_{index:02d}")
        for source_key, metric_name in [
            ("latency_s", "latency_p95_s"),
            ("rtf", "rtf_p95"),
        ]:
            source_payload = _as_dict(case_row.get(source_key))
            if source_payload.get("p95") is None:
                continue
            _metric_row(
                root,
                rows,
                row_id=f"tail_confidence_{case_id}_{metric_name}",
                category="tail_confidence",
                claim_or_table="per-case latency and RTF tail distribution",
                runtime=_tail_runtime(case_id),
                scenario=str(case_row.get("label") or case_id),
                metric_name=metric_name,
                metric_value=source_payload.get("p95"),
                source_json=TAIL_CONFIDENCE,
                source_pointer=f"/rows/{index}/{source_key}/p95",
                evidence_files=[
                    case_row.get("result_path"),
                    case_row.get("wer_path"),
                    TAIL_CONFIDENCE,
                ],
                rerun_command_ids=["build_tail_confidence_appendix"],
                report_routes=[
                    "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
                    "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
                ],
                interpretation=str(case_row.get("group") or "tail distribution row"),
            )


def _build_acceptance_rows(
    root: Path, rows: list[dict[str, Any]], acceptance: dict[str, Any]
) -> None:
    for index, row in enumerate(_as_list(acceptance.get("rows")), start=1):
        if not isinstance(row, dict):
            continue
        _metric_row(
            root,
            rows,
            row_id=f"acceptance_row_{index:02d}_{row.get('regime')}",
            category="acceptance_matrix",
            claim_or_table=str(row.get("regime") or ""),
            runtime="mixed" if row.get("regime") == "strict_runtime_comparison" else str(row.get("regime") or ""),
            scenario=str(row.get("pressure") or ""),
            concurrency=None,
            metric_name="accepted_key_metrics_text",
            metric_text=str(row.get("key_metrics") or ""),
            metric_value=bool(row.get("accepted")),
            source_json=ACCEPTANCE,
            source_pointer=f"/rows/{index - 1}/key_metrics",
            evidence_files=[*list(_as_list(row.get("evidence"))), CLAIMS],
            rerun_command_ids=_acceptance_commands(row),
            report_routes=[
                "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
            ],
            interpretation=str(row.get("action") or ""),
        )


def _build_stage_rows(
    root: Path, rows: list[dict[str, Any]], stage_drilldown: dict[str, Any]
) -> None:
    stage_summary = _summary(stage_drilldown)
    for metric_name in [
        "rows_total",
        "boundary_rows_total",
        "budget_rows_total",
        "stage_routes_total",
        "checks_total",
        "checks_passed",
        "required_failures",
    ]:
        _metric_row(
            root,
            rows,
            row_id=f"stage_drilldown_summary_{metric_name}",
            category="stage_drilldown",
            claim_or_table="stage budget and boundary drilldown coverage",
            runtime="mixed",
            scenario="all audited stage routes",
            metric_name=metric_name,
            metric_value=stage_summary.get(metric_name),
            source_json=STAGE_DRILLDOWN,
            source_pointer=f"/summary/{metric_name}",
            evidence_files=[STAGE_DRILLDOWN],
            rerun_command_ids=[
                "build_stage_latency_budget",
                "build_stage_boundary_bottleneck_ledger",
                "build_stage_drilldown_index",
            ],
            report_routes=[
                "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            ],
            interpretation="Coverage and health of the stage-by-stage evidence index.",
        )
    for row in _as_list(stage_drilldown.get("rows")):
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or "unknown")
        stage_route = list(_as_list(row.get("stage_route") or row.get("stage_focus")))
        _metric_row(
            root,
            rows,
            row_id=f"stage_drilldown_{row_id}",
            category="stage_drilldown",
            claim_or_table=str(row.get("entry_type") or ""),
            runtime=str(row.get("runtime") or ""),
            scenario=str(row.get("case") or ""),
            metric_name="stage_bottleneck_verdict",
            metric_text=str(row.get("safe_conclusion") or ""),
            metric_value=str(row.get("bottleneck_verdict") or row.get("status") or ""),
            metric_payload=_as_dict(row.get("metrics")),
            source_json=STAGE_DRILLDOWN,
            source_pointer=f"/rows/id={row_id}",
            evidence_files=[*list(_as_list(row.get("evidence_files"))), STAGE_DRILLDOWN],
            rerun_command_ids=[
                *[str(command_id) for command_id in _as_list(row.get("rerun_command_ids"))],
                "build_stage_drilldown_index",
            ],
            report_routes=[
                "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            ],
            stage_route=[str(part) for part in stage_route],
            interpretation=str(row.get("diagnosis") or ""),
        )


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


def _is_raw_artifact(path: str) -> bool:
    return path.startswith("results/qwen35_") and not path.startswith(
        AUDIT_DIR.as_posix()
    )


def build_metric_provenance(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    headline = _load_json_optional(root / HEADLINE)
    acceptance = _load_json_optional(root / ACCEPTANCE)
    stage_drilldown = _load_json_optional(root / STAGE_DRILLDOWN)
    repro_manifest = _load_json_optional(root / REPRO_COMMANDS)
    tail_confidence = _load_json_optional(root / TAIL_CONFIDENCE)

    rows: list[dict[str, Any]] = []
    _build_strict_c4_rows(root, rows, headline)
    _build_sglang_stress_rows(root, rows, headline)
    _build_synthetic_rows(root, rows, headline)
    _build_vllm_rows(root, rows, headline)
    _build_tail_confidence_rows(root, rows, tail_confidence)
    _build_acceptance_rows(root, rows, acceptance)
    _build_stage_rows(root, rows, stage_drilldown)

    command_ids = _command_ids(repro_manifest)
    row_command_ids = sorted(
        {
            command_id
            for row in rows
            for command_id in _as_list(row.get("rerun_command_ids"))
            if command_id
        }
    )
    evidence_files = sorted(
        {
            evidence
            for row in rows
            for evidence in _as_list(row.get("evidence_files"))
            if evidence
        }
    )
    missing_command_ids = sorted(
        command_id for command_id in row_command_ids if command_id not in command_ids
    )
    missing_evidence_files = sorted(
        evidence for evidence in evidence_files if not _path_exists(root, evidence)
    )
    categories = _category_counts(rows)
    source_summaries = {
        "headline_scorecard": _summary(headline),
        "acceptance_matrix": _summary(acceptance),
        "stage_drilldown_index": _summary(stage_drilldown),
        "repro_command_manifest": _summary(repro_manifest),
        "tail_confidence_appendix": _summary(tail_confidence),
    }
    critical_row_ids = {
        "strict_c4_sglang_latency_mean_s",
        "strict_c4_vllm_latency_mean_s",
        "strict_c4_sglang_lower_latency_mean_pct",
        "sglang_videoamme_c8_throughput_qps",
        "sglang_videoamme_c16_vs_c8_qps_delta_pct",
        "synthetic_long_c8_rtf_mean",
        "vllm_c8_original_runner_overhead_pct_wall",
        "vllm_c8_prebuild_w4_runner_qps",
        "tail_confidence_summary_strict_c4_sglang_latency_p95_s",
        "tail_confidence_summary_strict_c4_vllm_latency_p95_s",
        "stage_drilldown_summary_rows_total",
    }
    present_row_ids = {str(row.get("id")) for row in rows}
    checks = {
        "headline_scorecard_ready": bool(source_summaries["headline_scorecard"].get("ready"))
        and int(source_summaries["headline_scorecard"].get("checks_total") or 0) >= 9,
        "acceptance_matrix_ready": bool(source_summaries["acceptance_matrix"].get("ready"))
        and int(source_summaries["acceptance_matrix"].get("rows_total") or 0) >= 17,
        "stage_drilldown_ready": bool(source_summaries["stage_drilldown_index"].get("ready"))
        and int(source_summaries["stage_drilldown_index"].get("rows_total") or 0) >= 52,
        "tail_confidence_ready": bool(
            source_summaries["tail_confidence_appendix"].get("ready")
        )
        and int(source_summaries["tail_confidence_appendix"].get("rows_total") or 0)
        >= 18,
        "repro_command_registry_ready": bool(
            source_summaries["repro_command_manifest"].get(
                "required_command_ids_present"
            )
        )
        and int(source_summaries["repro_command_manifest"].get("commands_total") or 0) >= 60,
        "builder_command_registered": "build_metric_provenance_index" in command_ids,
        "rows_total": len(rows) >= 150,
        "raw_artifacts_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
        "categories_present": {
            "headline_strict_c4",
            "sglang_videoamme_stress",
            "synthetic_speech",
            "vllm_offline_diagnostic",
            "tail_confidence",
            "acceptance_matrix",
            "stage_drilldown",
        }.issubset(categories),
        "critical_metric_rows_present": critical_row_ids.issubset(present_row_ids),
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    raw_artifacts = sorted(path for path in evidence_files if _is_raw_artifact(path))
    packaged_evidence = sorted(path for path in evidence_files if not _is_raw_artifact(path))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "categories": categories,
            "source_jsons_total": len({row.get("source_json") for row in rows}),
            "evidence_files_total": len(evidence_files),
            "raw_artifacts_total": len(raw_artifacts),
            "packaged_evidence_files_total": len(packaged_evidence),
        "command_refs_total": len(row_command_ids),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "This index maps report metrics to local raw artifacts and packaged "
                "machine evidence. Large raw benchmark outputs are referenced by path "
                "and regenerated through repro_command_manifest.json rather than "
                "embedded in the share tarball."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "required_failures": required_failures,
            "raw_artifacts": raw_artifacts,
            "packaged_evidence_files": packaged_evidence,
        },
        "source_summaries": source_summaries,
        "rows": rows,
    }


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Metric Provenance Index")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- rows: `{summary['rows_total']}`")
    print(f"- raw artifacts: `{summary['raw_artifacts_total']}`")
    print(f"- command refs: `{summary['command_refs_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni metric provenance index JSON."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output
    if not output.is_absolute():
        output = root / output

    payload = build_metric_provenance(root)
    _save_json(payload, output)
    _print_markdown(payload)

    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
