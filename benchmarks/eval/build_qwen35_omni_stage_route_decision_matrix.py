# SPDX-License-Identifier: Apache-2.0
"""Build a route-level stage decision matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "stage_route_decision_matrix.json"
DEFAULT_HEATMAP_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md"
)
DEFAULT_HEATMAP_JSON_OUTPUT = AUDIT_DIR / "pressure_stage_heatmap.json"

STAGE_REPRODUCTION = AUDIT_DIR / "stage_reproduction_drilldown.json"
REPRO_COMMANDS = AUDIT_DIR / "repro_command_manifest.json"
STAGE_LATENCY_BUDGET = AUDIT_DIR / "stage_latency_budget.json"
STAGE_BOUNDARY_LEDGER = AUDIT_DIR / "stage_boundary_bottleneck_ledger.json"
TAIL_CONFIDENCE = AUDIT_DIR / "tail_confidence_appendix.json"


CRITICAL_ROUTES = {
    "admission -> preprocessing",
    "admission -> preprocessing -> talker -> code2wav",
    "admission -> preprocessing_queue -> preprocessing -> talker -> code2wav",
    "preprocessing -> encoder_thinker",
    "talker -> code2wav_stream",
    "code2wav_collect -> code2wav_decode",
    "talker -> code2wav -> text_to_speech",
    "thinker -> talker",
    "talker -> code2wav",
    "offline_runner -> engine_admission",
    "offline_runner -> engine_admission -> encoder -> thinker -> talker -> code2wav",
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


def _save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _top_key(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _num(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "n/a"


def _ms(value: Any) -> str:
    return _num(value, 1, "ms")


def _pct_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    return _num(value, 1, "%")


def _tail_rows_by_case(tail_confidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in _as_list(tail_confidence.get("rows"))
        if isinstance(row, dict) and row.get("case_id")
    }


def _tail_metric(row: dict[str, Any], metric: str, stat: str) -> Any:
    value = row.get(metric, {})
    return value.get(stat) if isinstance(value, dict) else None


def _budget_by_concurrency(rows: list[Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            result[int(row.get("concurrency"))] = row
        except Exception:
            continue
    return result


def _synthetic_budget_by_key(rows: list[Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            result[(str(row.get("scenario")), int(row.get("concurrency")))] = row
        except Exception:
            continue
    return result


def _vllm_budget_by_workload(rows: list[Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("workload")): row
        for row in rows
        if isinstance(row, dict) and row.get("workload")
    }


def _tail_summary(row: dict[str, Any]) -> str:
    return (
        f"QPS={_num(row.get('throughput_qps'), 3)}; "
        f"lat_p95={_num(_tail_metric(row, 'latency_s', 'p95'), 3, 's')}; "
        f"RTF_p95={_num(_tail_metric(row, 'rtf', 'p95'), 4)}"
    )


def _sglang_heatmap_decision(concurrency: int, diagnosis: str) -> tuple[str, str, str]:
    if concurrency >= 16:
        return (
            "saturation boundary",
            "Keep as pressure evidence; c8 remains the serving edge for this recipe.",
            "Do not present c16 as the high-concurrency optimum.",
        )
    if concurrency >= 8:
        return (
            "throughput edge",
            "Use c8 as the current high-concurrency sweet spot while watching admission queue.",
            "Do not widen preprocessing concurrency without redesigning admission and placement.",
        )
    if concurrency >= 4:
        return (
            "balanced serving",
            "Use c4 as the balanced serving point and strict cross-runtime reference.",
            "Do not mix stress c4 and strict warmed c4 metrics.",
        )
    return (
        "latency-first guard",
        "Use as the low-concurrency latency and Talker-tail guard.",
        "Do not describe low-concurrency Talker tail as high-concurrency queue saturation.",
    )


def _sglang_heatmap_row(
    *,
    concurrency: int,
    budget: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    decision, action, caveat = _sglang_heatmap_decision(
        concurrency, str(budget.get("diagnosis") or "")
    )
    return {
        "id": f"heatmap-sglang-videoamme-c{concurrency}",
        "runtime": "sglang",
        "pressure": f"Video-AMME c={concurrency}",
        "pressure_axis": "concurrency",
        "headline_metrics": _tail_summary(tail),
        "admission_or_preprocess": (
            f"preproc_lifecycle={_ms(budget.get('preproc_lifecycle_avg_ms'))} / "
            f"{_pct_cell(budget.get('preproc_lifecycle_pct_of_latency'))}; "
            f"queue={_ms(budget.get('preproc_queue_estimate_ms'))} / "
            f"{_pct_cell(budget.get('preproc_queue_pct_of_latency'))}"
        ),
        "talker": (
            f"talker={_ms(budget.get('talker_avg_ms'))} / "
            f"{_pct_cell(budget.get('talker_pct_of_latency'))}; "
            f"top_stage={budget.get('top_stage') or 'n/a'}"
        ),
        "stage_connection": (
            f"talker->code2wav hop_p95={_ms(budget.get('talker_to_code2wav_hop_p95_ms'))}; "
            "handoff is not the current bottleneck"
        ),
        "code2wav": (
            f"decode={_ms(budget.get('code2wav_decode_avg_ms'))} / "
            f"{_pct_cell(budget.get('code2wav_decode_pct_of_latency'))}; "
            f"collect={_ms(budget.get('code2wav_window_collect_avg_ms'))}"
        ),
        "runner_or_engine": "n/a",
        "bottleneck_class": budget.get("diagnosis"),
        "decision": decision,
        "action": action,
        "do_not_say": caveat,
        "evidence_files": [
            str(tail.get("result_path") or ""),
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
        ],
        "rerun_command_ids": [
            "sglang_videoamme_stress",
            "build_stage_latency_budget",
            "build_stage_boundary_bottleneck_ledger",
        ],
    }


def _synthetic_heatmap_row(
    *,
    scenario: str,
    concurrency: int,
    budget: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    text_label = "short text" if scenario == "short" else "long text"
    realtime = "RTF below 1.0" if float(budget.get("rtf_mean") or 9.0) < 1.0 else "RTF watch"
    return {
        "id": f"heatmap-sglang-synthetic-{scenario}-c{concurrency}",
        "runtime": "sglang",
        "pressure": f"Synthetic {text_label} c={concurrency}",
        "pressure_axis": f"{scenario}_text_to_speech",
        "headline_metrics": (
            f"{_tail_summary(tail)}; words={_num(budget.get('target_words'), 0)}; "
            f"audio_mean={_num(budget.get('audio_duration_mean_s'), 1, 's')}"
        ),
        "admission_or_preprocess": "synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim",
        "talker": (
            f"talker={_ms(budget.get('talker_avg_ms'))} / "
            f"{_pct_cell(budget.get('talker_pct_of_latency'))}; {realtime}"
        ),
        "stage_connection": (
            f"talker->code2wav hop_p95={_ms(budget.get('talker_to_code2wav_hop_p95_ms'))}; "
            "handoff delta stays small versus Talker cadence"
        ),
        "code2wav": (
            f"decode={_ms(budget.get('code2wav_decode_avg_ms'))} / "
            f"{_pct_cell(budget.get('code2wav_decode_pct_of_latency'))}; "
            f"collect={_ms(budget.get('code2wav_window_collect_avg_ms'))}"
        ),
        "runner_or_engine": "n/a",
        "bottleneck_class": budget.get("diagnosis"),
        "decision": f"{scenario}-text speech guardrail",
        "action": "Use as length/RTF guardrail; optimize Talker cadence before code2wav compute.",
        "do_not_say": "Do not replace full-set or Video-AMME headline with synthetic-only evidence.",
        "evidence_files": [
            str(tail.get("result_path") or ""),
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
        ],
        "rerun_command_ids": [
            "sglang_synthetic_text_to_speech",
            "build_stage_latency_budget",
            "build_tail_confidence_appendix",
        ],
    }


def _vllm_heatmap_row(
    *,
    workload: str,
    tail_case: str,
    budget: dict[str, Any],
    tail: dict[str, Any],
) -> dict[str, Any]:
    original_prompt_feed = str(budget.get("diagnosis")) == "prompt_feed_limited"
    if workload == "vLLM-c8-prebuild-w4":
        direct_command = "vllm_c8_prebuild_w4"
    elif workload == "vLLM-c8":
        direct_command = "vllm_c8_original"
    elif workload == "vLLM-c4":
        direct_command = "vllm_c4_original"
    else:
        direct_command = ""
    decision = (
        "optimized offline diagnostic"
        if "prebuild" in workload
        else "offline prompt-feed diagnostic"
    )
    return {
        "id": f"heatmap-{workload.lower().replace('=', '').replace('-', '_')}",
        "runtime": "vllm",
        "pressure": workload,
        "pressure_axis": "offline_diagnostic",
        "headline_metrics": _tail_summary(tail),
        "admission_or_preprocess": (
            f"runner_overhead={_pct_cell(budget.get('runner_overhead_pct_wall'))}; "
            f"admission_p95={_ms(budget.get('batch_admission_span_p95_ms'))}"
        ),
        "talker": (
            f"thinker->talker p95={_ms(budget.get('thinker_to_talker_feed_p95_ms'))}; "
            f"talker->code2wav drain_p95={_ms(budget.get('talker_to_code2wav_drain_p95_ms'))}"
        ),
        "stage_connection": (
            "prompt-feed dominates before engine boundaries"
            if original_prompt_feed
            else "prebuild removes most admission span and exposes later engine/talker tail"
        ),
        "code2wav": f"talker/code2wav drain_p95={_ms(budget.get('talker_to_code2wav_drain_p95_ms'))}",
        "runner_or_engine": (
            f"runner_QPS={_num(budget.get('runner_qps'), 4)}; "
            f"engine_QPS={_num(budget.get('engine_qps'), 4)}"
        ),
        "bottleneck_class": budget.get("diagnosis"),
        "decision": decision,
        "action": "Use prebuild/online ingress before any c8 parity claim.",
        "do_not_say": "Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR.",
        "evidence_files": [
            str(tail.get("result_path") or ""),
            "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
        ],
        "rerun_command_ids": [
            command_id
            for command_id in [
                direct_command,
                "summarize_vllm_log_stages",
                "diagnose_vllm_admission",
                "build_stage_latency_budget",
            ]
            if command_id
        ],
        "tail_case_id": tail_case,
    }


def _build_pressure_stage_heatmap(root: Path) -> dict[str, Any]:
    root = root.resolve()
    stage_budget = _load_json_optional(root / STAGE_LATENCY_BUDGET)
    stage_ledger = _load_json_optional(root / STAGE_BOUNDARY_LEDGER)
    stage_reproduction = _load_json_optional(root / STAGE_REPRODUCTION)
    tail_confidence = _load_json_optional(root / TAIL_CONFIDENCE)
    repro_manifest = _load_json_optional(root / REPRO_COMMANDS)

    tails = _tail_rows_by_case(tail_confidence)
    sglang_budget = _budget_by_concurrency(
        _as_list(stage_budget.get("sglang_videoamme_budget"))
    )
    synthetic_budget = _synthetic_budget_by_key(
        _as_list(stage_budget.get("synthetic_speech_budget"))
    )
    vllm_budget = _vllm_budget_by_workload(
        _as_list(stage_budget.get("vllm_offline_budget"))
    )

    rows: list[dict[str, Any]] = []
    for concurrency in [1, 2, 4, 8, 16]:
        rows.append(
            _sglang_heatmap_row(
                concurrency=concurrency,
                budget=sglang_budget.get(concurrency, {}),
                tail=tails.get(f"sglang_stress_c{concurrency}", {}),
            )
        )
    for scenario in ["short", "long"]:
        for concurrency in [1, 4, 8]:
            rows.append(
                _synthetic_heatmap_row(
                    scenario=scenario,
                    concurrency=concurrency,
                    budget=synthetic_budget.get((scenario, concurrency), {}),
                    tail=tails.get(f"synthetic_{scenario}_c{concurrency}", {}),
                )
            )
    for workload, tail_case in [
        ("vLLM-c4", "vllm_c4"),
        ("vLLM-c8", "vllm_c8"),
        ("vLLM-c8-prebuild-w1", "vllm_c8_prebuild_w1"),
        ("vLLM-c8-prebuild-w4", "vllm_c8_prebuild_w4"),
    ]:
        rows.append(
            _vllm_heatmap_row(
                workload=workload,
                tail_case=tail_case,
                budget=vllm_budget.get(workload, {}),
                tail=tails.get(tail_case, {}),
            )
        )

    command_ids = _command_ids(repro_manifest)
    missing_command_ids = sorted(
        {
            command_id
            for row in rows
            for command_id in _as_list(row.get("rerun_command_ids"))
            if command_id not in command_ids
        }
    )
    missing_evidence_files = sorted(
        {
            evidence
            for row in rows
            for evidence in _as_list(row.get("evidence_files"))
            if evidence and not _path_exists(root, evidence)
        }
    )
    row_ids = {str(row.get("id")) for row in rows}
    required_ids = {
        "heatmap-sglang-videoamme-c1",
        "heatmap-sglang-videoamme-c8",
        "heatmap-sglang-videoamme-c16",
        "heatmap-sglang-synthetic-short-c8",
        "heatmap-sglang-synthetic-long-c8",
        "heatmap-vllm_c8_prebuild_w4",
    }
    long_c8 = next(
        (
            row
            for row in rows
            if row.get("id") == "heatmap-sglang-synthetic-long-c8"
        ),
        {},
    )
    vllm_w4 = next(
        (row for row in rows if row.get("id") == "heatmap-vllm_c8_prebuild_w4"),
        {},
    )
    checks = {
        "stage_budget_ready": bool(stage_budget.get("summary", {}).get("ready"))
        and len(stage_budget.get("sglang_videoamme_budget", [])) >= 5
        and len(stage_budget.get("synthetic_speech_budget", [])) >= 6
        and len(stage_budget.get("vllm_offline_budget", [])) >= 4,
        "stage_ledger_ready": bool(stage_ledger.get("summary", {}).get("ready"))
        and int(stage_ledger.get("summary", {}).get("pressure_transition_rows") or 0)
        >= 11,
        "stage_reproduction_ready": bool(
            stage_reproduction.get("summary", {}).get("ready")
        )
        and int(stage_reproduction.get("summary", {}).get("stage_rows_total") or 0)
        >= 52,
        "tail_confidence_ready": bool(tail_confidence.get("summary", {}).get("ready"))
        and int(tail_confidence.get("summary", {}).get("rows_total") or 0) >= 18,
        "required_heatmap_rows_present": required_ids.issubset(row_ids),
        "all_rows_have_stage_cells": all(
            row.get("headline_metrics")
            and row.get("admission_or_preprocess")
            and row.get("talker")
            and row.get("stage_connection")
            and row.get("code2wav")
            and row.get("decision")
            and row.get("do_not_say")
            for row in rows
        ),
        "single_high_short_long_vllm_covered": len(rows) >= 15
        and any(row.get("pressure") == "Video-AMME c=1" for row in rows)
        and any(row.get("pressure") == "Video-AMME c=16" for row in rows)
        and any("Synthetic short text" in str(row.get("pressure")) for row in rows)
        and any("Synthetic long text" in str(row.get("pressure")) for row in rows)
        and any(row.get("runtime") == "vllm" for row in rows),
        "long_c8_realtime_guard": "RTF_p95=0.5001" in str(
            long_c8.get("headline_metrics")
        )
        or "RTF below 1.0" in str(long_c8.get("talker")),
        "vllm_w4_keeps_offline_caveat": vllm_w4.get("decision")
        == "optimized offline diagnostic"
        and "online serving parity" in str(vllm_w4.get("do_not_say")),
        "evidence_files_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "sglang_videoamme_rows": sum(
                1 for row in rows if str(row.get("id", "")).startswith("heatmap-sglang-videoamme")
            ),
            "synthetic_rows": sum(
                1 for row in rows if "synthetic" in str(row.get("id", ""))
            ),
            "vllm_rows": sum(1 for row in rows if row.get("runtime") == "vllm"),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "One-page pressure-by-stage heatmap covering SGLang single/high "
                "concurrency, short/long text-to-speech, and vLLM original/prebuild "
                "diagnostics with stage connection and caveat cells."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "required_failures": required_failures,
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
        },
        "source_summaries": {
            "stage_latency_budget": stage_budget.get("summary", {}),
            "stage_boundary_bottleneck_ledger": stage_ledger.get("summary", {}),
            "stage_reproduction_drilldown": stage_reproduction.get("summary", {}),
            "tail_confidence_appendix": tail_confidence.get("summary", {}),
            "repro_command_manifest": repro_manifest.get("summary", {}),
        },
        "rows": rows,
    }


def _decision(route_key: str, verdicts: Counter[str], statuses: Counter[str]) -> tuple[str, str, str]:
    verdict_keys = set(verdicts)
    status_keys = set(statuses)
    if "saturation_boundary" in verdict_keys or "admission_queue_bottleneck" in verdict_keys:
        return (
            "recommended_window_with_saturation_guard",
            "Keep c4-c8 as the serving window; treat c16 as queue/admission saturation, not a better high-concurrency point.",
            "可以说当前高并发瓶颈是 admission/queue 与 talker tail 叠加；不能把 c16 包装成推荐配置。",
        )
    if "negative_optimization" in verdict_keys or "contention_regression" in status_keys:
        return (
            "anti_recipe",
            "Do not widen preprocessing concurrency in the current recipe; it regresses throughput or fails.",
            "可以说 naive preprocessing parallelism 是负优化证据；优化前先保留 serial preprocessing。",
        )
    if "offline_runner_prompt_feed_bottleneck" in verdict_keys:
        return (
            "vllm_offline_prompt_feed_limited",
            "Use prebuilt prompts or online ingress before making c8 cross-runtime parity claims.",
            "可以说 vLLM original c8 被 offline prompt build/feed admission 限制；不能说这是 online serving parity。",
        )
    if "offline_diagnostic_not_parity" in verdict_keys or "engine_or_workload_limited" in verdict_keys:
        return (
            "optimized_offline_diagnostic",
            "Use prebuild w4 as the strongest offline diagnostic; require online ingress plus WER/ASR before parity replacement.",
            "可以说 prebuild w4 解除主要 admission 问题并暴露 engine/talker tail；不能当作最终线上 parity。",
        )
    if "observed_bottleneck" in verdict_keys or "tail_watch" in verdict_keys:
        return (
            "watch_after_prompt_feed_removed",
            "Track the later vLLM engine/talker/code2wav tail after prompt-feed is removed.",
            "可以说这是 prompt-feed 移走后的后续观察点；不能把它外推为 SGLang 当前瓶颈。",
        )
    if "talker_ar_tail" in verdict_keys:
        return (
            "talker_ar_tail",
            "Optimize Talker AR cadence/batching before code2wav compute for these regimes.",
            "可以说低/中并发主要是 talker AR tail；code2wav 不是优先优化对象。",
        )
    if "long_speech_talker_ar_dominant_but_faster_than_realtime" in verdict_keys:
        return (
            "long_speech_guardrail_pass",
            "Keep long-text/long-speech as a guardrail; it remains faster than real time while Talker AR dominates.",
            "可以说长语音 c8 仍快于实时；不能把 synthetic guardrail 替代官方 full-set。",
        )
    if "faster_than_realtime" in verdict_keys:
        return (
            "speech_generation_guardrail_pass",
            "Use this route as a short/long text-to-speech guardrail and monitor Talker AR under concurrency.",
            "可以说短/长文本语音路径覆盖且通过；仍需保留 full-set caveat。",
        )
    if route_key == "talker -> code2wav_stream":
        return (
            "handoff_healthy",
            "Do not optimize the stream handoff first; p95 hop stays small across measured SGLang pressure.",
            "可以说 SGLang talker->code2wav 连接健康；当前瓶颈不在连接本身。",
        )
    if route_key == "code2wav_collect -> code2wav_decode":
        return (
            "decode_not_bottleneck",
            "Do not optimize vocoder decode first; collect wait reflects Talker chunk cadence.",
            "可以说 code2wav decode 稳定且小；collect wait 是等待 chunk，不是 vocoder 算力瓶颈。",
        )
    if route_key == "thinker -> talker":
        return (
            "handoff_healthy",
            "Keep this as a health check; vLLM thinker-to-talker feed is not the original c8 limiter.",
            "可以说 thinker->talker 连接不是当前主瓶颈；vLLM c8 主要先看 runner admission。",
        )
    return (
        "healthy_or_secondary",
        "Keep as evidence, not the first optimization target.",
        "可以说该 route 不是当前优先瓶颈；需要按对应 stage row 复核。",
    )


def _build_rows(stage_reproduction: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source_rows = [
        row for row in _as_list(stage_reproduction.get("rows")) if isinstance(row, dict)
    ]
    for route in _as_list(stage_reproduction.get("route_summary")):
        if not isinstance(route, dict):
            continue
        route_key = str(route.get("route_key") or "")
        matching = [
            row for row in source_rows if str(row.get("route_key") or "") == route_key
        ]
        statuses = Counter(str(row.get("status") or "") for row in matching)
        verdicts = Counter(str(row.get("bottleneck_verdict") or "") for row in matching)
        decision, focus, talking_point = _decision(route_key, verdicts, statuses)
        stage_row_ids = _dedupe([row.get("stage_row_id") for row in matching])
        metric_row_ids = _dedupe([row.get("metric_row_id") for row in matching])
        evidence_files = _dedupe(
            [
                evidence
                for row in matching
                for evidence in _as_list(row.get("evidence_files"))
            ]
        )
        raw_artifacts = _dedupe(
            [
                artifact
                for row in matching
                for artifact in _as_list(row.get("raw_artifacts"))
            ]
        )
        command_ids = _dedupe(
            [
                command_id
                for row in matching
                for command_id in _as_list(row.get("rerun_command_ids"))
            ]
            + ["build_stage_route_decision_matrix"]
        )
        cases = _dedupe([row.get("case") for row in matching])
        runtimes = _dedupe([row.get("runtime") for row in matching])
        rows.append(
            {
                "route_key": route_key,
                "rows_total": len(matching),
                "runtimes": runtimes,
                "cases": cases,
                "stage_row_ids": stage_row_ids,
                "metric_row_ids": metric_row_ids,
                "status_counts": dict(sorted(statuses.items())),
                "verdict_counts": dict(sorted(verdicts.items())),
                "dominant_status": _top_key(statuses),
                "dominant_verdict": _top_key(verdicts),
                "route_decision": decision,
                "optimization_focus": focus,
                "safe_talking_point": talking_point,
                "evidence_files": evidence_files,
                "raw_artifacts": raw_artifacts,
                "rerun_command_ids": command_ids,
                "jq_queries": {
                    "route_rows": (
                        "jq '.rows[] | select(.route_key == "
                        f"{json.dumps(route_key)})' "
                        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json"
                    ),
                    "first_stage_row": (
                        "jq '.rows[] | select(.stage_row_id == "
                        f"{json.dumps(stage_row_ids[0] if stage_row_ids else '')})' "
                        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json"
                    ),
                    "rerun_commands": (
                        "jq '.commands[] | select(.id as $id | "
                        f"{json.dumps(command_ids)} | index($id))' "
                        "results/qwen35_report_audit_20260619/repro_command_manifest.json"
                    ),
                },
            }
        )
    return rows


def build_stage_route_decision_matrix(root: Path) -> dict[str, Any]:
    root = root.resolve()
    stage_reproduction = _load_json_optional(root / STAGE_REPRODUCTION)
    repro_manifest = _load_json_optional(root / REPRO_COMMANDS)
    rows = _build_rows(stage_reproduction)
    repro_summary = _as_dict(repro_manifest.get("summary"))
    stage_summary = _as_dict(stage_reproduction.get("summary"))
    command_ids = _command_ids(repro_manifest)
    evidence_files = sorted(
        {
            evidence
            for row in rows
            for evidence in _as_list(row.get("evidence_files"))
        }
    )
    raw_artifacts = sorted(
        {
            artifact
            for row in rows
            for artifact in _as_list(row.get("raw_artifacts"))
        }
    )
    command_refs = sorted(
        {
            command_id
            for row in rows
            for command_id in _as_list(row.get("rerun_command_ids"))
        }
    )
    missing_evidence_files = sorted(
        evidence for evidence in evidence_files if not _path_exists(root, evidence)
    )
    missing_command_ids = sorted(
        command_id for command_id in command_refs if command_id not in command_ids
    )
    route_keys = {str(row.get("route_key") or "") for row in rows}
    missing_critical_routes = sorted(CRITICAL_ROUTES - route_keys)
    checks = {
        "stage_reproduction_ready": bool(stage_summary.get("ready"))
        and int(stage_summary.get("stage_rows_total") or 0) >= 52
        and int(stage_summary.get("route_rows_total") or 0) >= 11
        and int(stage_summary.get("required_failures") or 0) == 0,
        "repro_command_registry_ready": bool(
            repro_summary.get("required_command_ids_present")
        )
        and int(repro_summary.get("commands_total") or 0) >= 60,
        "route_rows_present": len(rows) >= 11,
        "critical_routes_covered": not missing_critical_routes,
        "decisions_present": all(
            row.get("route_decision")
            and row.get("optimization_focus")
            and row.get("safe_talking_point")
            for row in rows
        ),
        "evidence_files_present": not missing_evidence_files,
        "raw_artifacts_present": len(raw_artifacts) >= 28,
        "rerun_command_ids_present": not missing_command_ids
        and len(command_refs) >= 15,
        "jq_queries_present": all(
            row.get("jq_queries", {}).get("route_rows")
            and row.get("jq_queries", {}).get("first_stage_row")
            and row.get("jq_queries", {}).get("rerun_commands")
            for row in rows
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "route_rows_total": len(rows),
            "stage_rows_total": sum(int(row.get("rows_total") or 0) for row in rows),
            "critical_routes_total": len(CRITICAL_ROUTES),
            "raw_artifacts_total": len(raw_artifacts),
            "evidence_files_total": len(evidence_files),
            "command_refs_total": len(command_refs),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "Route-level stage decision matrix for explaining bottlenecks, "
                "handoff health, optimization actions, safe external wording, "
                "and drilldown evidence for each audited stage route."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "missing_critical_routes": missing_critical_routes,
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "required_failures": required_failures,
        },
        "source_summaries": {
            "stage_reproduction_drilldown": stage_summary,
            "repro_command_manifest": repro_summary,
        },
        "rows": rows,
    }


def _md(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_counts(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni Stage Route 裁决矩阵",
        "",
        "状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。",
        "工作目录：`/home/gangouyu/sglang-omni`。",
        "用途：把 52 条 stage drilldown 聚合成 11 条 route-level 裁决，方便对外分享时解释瓶颈、连接健康度、优化动作和复核入口。",
        "",
        "## 1. 当前 Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| ready | `{summary['ready']}` |",
        f"| route rows | `{summary['route_rows_total']}` |",
        f"| covered stage rows | `{summary['stage_rows_total']}` |",
        f"| raw artifacts | `{summary['raw_artifacts_total']}` |",
        f"| command refs | `{summary['command_refs_total']}` |",
        f"| checks | `{summary['checks_passed']}/{summary['checks_total']}` |",
        f"| required failures | `{summary['required_failures']}` |",
        "",
        "## 2. Route 裁决总表",
        "",
        "| Route | Runtime | Rows | Dominant Verdict | Decision | Optimization Focus | Safe Talking Point | First jq Query |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        query = row.get("jq_queries", {}).get("route_rows", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("route_key")),
                    _md(", ".join(_as_list(row.get("runtimes")))),
                    str(row.get("rows_total")),
                    _md(row.get("dominant_verdict")),
                    _md(row.get("route_decision")),
                    _md(row.get("optimization_focus")),
                    _md(row.get("safe_talking_point")),
                    "`" + _md(query) + "`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. Reviewer 追问入口",
            "",
            "| Route | Stage Rows | Status Counts | Verdict Counts | Evidence | Commands |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["rows"]:
        evidence = ", ".join(_as_list(row.get("evidence_files"))[:4])
        commands = ", ".join(_as_list(row.get("rerun_command_ids"))[:6])
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("route_key")),
                    _md(", ".join(_as_list(row.get("stage_row_ids")))),
                    _md(_fmt_counts(row.get("status_counts"))),
                    _md(_fmt_counts(row.get("verdict_counts"))),
                    _md(evidence),
                    _md(commands),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 4. 机器证据",
            "",
            "- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`",
            "- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`",
            "- `results/qwen35_report_audit_20260619/stage_drilldown_index.json`",
            "- `results/qwen35_report_audit_20260619/repro_command_manifest.json`",
            "- `benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md`",
            "",
        ]
    )
    return "\n".join(lines)


def render_heatmap_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni Pressure × Stage Heatmap",
        "",
        "状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。",
        "工作目录：`/home/gangouyu/sglang-omni`。",
        "",
        "用途：把单并发/高并发、短/长文本语音、vLLM original/prebuild 诊断压到同一张 stage 热力表。",
        "这份报告只聚合已经审计通过的 JSON，不引入新的 benchmark 数字。",
        "",
        "## 1. Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| ready | `{summary['ready']}` |",
        f"| rows | `{summary['rows_total']}` |",
        f"| SGLang Video-AMME rows | `{summary['sglang_videoamme_rows']}` |",
        f"| synthetic rows | `{summary['synthetic_rows']}` |",
        f"| vLLM rows | `{summary['vllm_rows']}` |",
        f"| checks | `{summary['checks_passed']}/{summary['checks_total']}` |",
        f"| required failures | `{summary['required_failures']}` |",
        "",
        "## 2. Heatmap 总览",
        "",
        "| Pressure | Runtime | Key Metrics | Admission / Preprocess | Talker | Handoff | Code2wav | Runner / Engine | Bottleneck | Decision | Caveat |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("pressure")),
                    _md(row.get("runtime")),
                    _md(row.get("headline_metrics")),
                    _md(row.get("admission_or_preprocess")),
                    _md(row.get("talker")),
                    _md(row.get("stage_connection")),
                    _md(row.get("code2wav")),
                    _md(row.get("runner_or_engine")),
                    _md(row.get("bottleneck_class")),
                    _md(row.get("decision")),
                    _md(row.get("do_not_say")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. 复核入口",
            "",
            "| Pressure | Evidence | Rerun Command IDs |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["rows"]:
        evidence = ", ".join(_as_list(row.get("evidence_files"))[:4])
        commands = ", ".join(f"`{command}`" for command in _as_list(row.get("rerun_command_ids")))
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("pressure")),
                    _md(evidence),
                    _md(commands),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 4. 读法",
            "",
            "- SGLang c=1/c=2：低并发主要看 Talker AR tail，不能说成 admission queue 饱和。",
            "- SGLang c=8：吞吐达到当前峰值，但 queue 已显性化；这是当前 high-concurrency serving edge。",
            "- SGLang c=16：吞吐低于 c=8 且 queue/RTF tail 上升，是 saturation boundary。",
            "- synthetic short/long：用来隔离 thinker/talker/code2wav；长文本 c=8 仍快于实时，优先瓶颈不是 vocoder decode。",
            "- vLLM original c=8：offline runner prompt build/feed 限制 admission；prebuild w4 是 optimized offline diagnostic，不是 online serving parity。",
            "",
            "## 5. 机器证据",
            "",
            "- `results/qwen35_report_audit_20260619/pressure_stage_heatmap.json`",
            "- `results/qwen35_report_audit_20260619/stage_latency_budget.json`",
            "- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`",
            "- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`",
            "- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`",
            "- Rebuild command ID: `build_stage_route_decision_matrix`",
            "",
        ]
    )
    return "\n".join(lines)


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Stage Route Decision Matrix")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- route rows: `{summary['route_rows_total']}`")
    print(f"- covered stage rows: `{summary['stage_rows_total']}`")
    print(f"- raw artifacts: `{summary['raw_artifacts_total']}`")
    print(f"- command refs: `{summary['command_refs_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni stage route decision matrix."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--heatmap-output", type=Path, default=DEFAULT_HEATMAP_OUTPUT)
    parser.add_argument(
        "--heatmap-json-output", type=Path, default=DEFAULT_HEATMAP_JSON_OUTPUT
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    heatmap_output = (
        args.heatmap_output
        if args.heatmap_output.is_absolute()
        else root / args.heatmap_output
    )
    heatmap_json_output = (
        args.heatmap_json_output
        if args.heatmap_json_output.is_absolute()
        else root / args.heatmap_json_output
    )
    payload = build_stage_route_decision_matrix(root)
    heatmap_payload = _build_pressure_stage_heatmap(root)
    _save_json(payload, json_output)
    _save_text(render_markdown(payload), output)
    _save_json(heatmap_payload, heatmap_json_output)
    _save_text(render_heatmap_markdown(heatmap_payload), heatmap_output)
    _print_markdown(payload)
    print(f"Stage route decision matrix written: {output}")
    print(f"Stage route decision matrix JSON written: {json_output}")
    print(f"Pressure-stage heatmap written: {heatmap_output}")
    print(f"Pressure-stage heatmap JSON written: {heatmap_json_output}")
    if args.strict and (
        not payload["summary"]["ready"] or not heatmap_payload["summary"]["ready"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
