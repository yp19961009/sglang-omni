# SPDX-License-Identifier: Apache-2.0
"""Build a stage-boundary bottleneck ledger for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "stage_boundary_bottleneck_ledger.json"


@dataclass(frozen=True)
class LedgerCheck:
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _check(
    checks: list[LedgerCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(LedgerCheck(name, _status(condition), evidence, required))


def _interaction_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("interactions", [])
    return rows if isinstance(rows, list) else []


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _case_name(row: dict[str, Any]) -> str:
    if row.get("label"):
        return str(row["label"])
    workload = str(row.get("workload") or "unknown")
    concurrency = row.get("concurrency")
    return f"{workload} c={concurrency}" if concurrency is not None else workload


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}{suffix}"


def _fmt_unit(value: Any, unit: str, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return _fmt(value, digits, unit)


def _fmt_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _fmt_delta(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.{digits}f}{suffix}"


def _pct_delta(after: Any, before: Any) -> float | None:
    before_value = _num(before, default=0.0)
    if before_value == 0.0:
        return None
    return (_num(after) - before_value) / before_value * 100.0


def _delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return _num(after) - _num(before)


def _table_rows(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    tables = payload.get("tables", {})
    if not isinstance(tables, dict):
        return []
    rows = tables.get(name, [])
    return rows if isinstance(rows, list) else []


def _by_concurrency(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        concurrency = row.get("concurrency")
        if concurrency is None:
            continue
        try:
            result[int(concurrency)] = row
        except (TypeError, ValueError):
            continue
    return result


def _interaction_metric(
    interactions: list[dict[str, Any]],
    *,
    runtime: str,
    workload: str,
    boundary: str,
    concurrency: int | None = None,
    label: str | None = None,
    metric: str,
) -> Any:
    for row in interactions:
        if row.get("runtime") != runtime:
            continue
        if row.get("workload") != workload:
            continue
        if row.get("boundary") != boundary:
            continue
        if concurrency is not None and row.get("concurrency") != concurrency:
            continue
        if label is not None and row.get("label") != label:
            continue
        metrics = _metrics(row)
        return metrics.get(metric)
    return None


def _vllm_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row.get("label") == label:
            return row
    return {}


def _transition_evidence(parts: list[str]) -> str:
    return "; ".join(part for part in parts if part)


def _sglang_pressure_rows(
    tables_payload: dict[str, Any],
    interactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _by_concurrency(_table_rows(tables_payload, "sglang_stress"))
    result: list[dict[str, Any]] = []
    for before_c, after_c in [(1, 2), (2, 4), (4, 8), (8, 16)]:
        before = rows.get(before_c, {})
        after = rows.get(after_c, {})
        if not before or not after:
            continue
        queue_before = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="request_admission_to_preprocessing",
            concurrency=before_c,
            metric="estimated_queue_avg_ms",
        )
        queue_after = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="request_admission_to_preprocessing",
            concurrency=after_c,
            metric="estimated_queue_avg_ms",
        )
        preproc_after = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="request_admission_to_preprocessing",
            concurrency=after_c,
            metric="preproc_lifecycle_avg_ms",
        )
        hop_before = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="talker_to_code2wav_stream",
            concurrency=before_c,
            metric="talker_to_code2wav_hop_p95_ms",
        )
        hop_after = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="talker_to_code2wav_stream",
            concurrency=after_c,
            metric="talker_to_code2wav_hop_p95_ms",
        )
        decode_before = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="code2wav_collect_to_decode",
            concurrency=before_c,
            metric="code2wav_decode_p95_ms",
        )
        decode_after = _interaction_metric(
            interactions,
            runtime="sglang",
            workload="Video-AMME ci-50",
            boundary="code2wav_collect_to_decode",
            concurrency=after_c,
            metric="code2wav_decode_p95_ms",
        )
        qps_delta = _pct_delta(after.get("throughput_qps"), before.get("throughput_qps"))
        latency_delta = _pct_delta(after.get("latency_p95_s"), before.get("latency_p95_s"))
        rtf_delta = _pct_delta(after.get("rtf_p95"), before.get("rtf_p95"))
        queue_delta = _delta(queue_after, queue_before)
        queue_share = (
            _num(queue_after) / _num(preproc_after) * 100.0
            if queue_after is not None and preproc_after
            else None
        )
        if after_c == 16:
            verdict = "saturation_boundary"
            decision = (
                "Do not use c16 as the serving optimum: throughput falls while "
                "admission queue and p95 RTF rise."
            )
        elif after_c == 8:
            verdict = "usable_high_concurrency_window"
            decision = (
                "Keep c8 as the throughput-oriented serving edge; queue pressure is "
                "visible but throughput still improves."
            )
        else:
            verdict = "scales_without_boundary_bottleneck"
            decision = "Throughput improves without moving the main bottleneck to stage handoff or decode."
        result.append(
            {
                "id": f"pressure-sglang-c{before_c}-c{after_c}",
                "runtime": "sglang",
                "workload": "Video-AMME ci-50",
                "transition": f"c{before_c} -> c{after_c}",
                "pressure_axis": "concurrency",
                "qps_delta_pct": qps_delta,
                "latency_p95_delta_pct": latency_delta,
                "rtf_p95_delta_pct": rtf_delta,
                "queue_delta_ms": queue_delta,
                "queue_share_after_pct": queue_share,
                "hop_p95_delta_ms": _delta(hop_after, hop_before),
                "decode_p95_delta_ms": _delta(decode_after, decode_before),
                "verdict": verdict,
                "decision": decision,
                "evidence": _transition_evidence(
                    [
                        f"QPS {_fmt_delta(qps_delta, suffix='%')}",
                        f"latency_p95 {_fmt_delta(latency_delta, suffix='%')}",
                        f"RTF_p95 {_fmt_delta(rtf_delta, suffix='%')}",
                        f"queue_delta {_fmt_delta(queue_delta, suffix='ms')}",
                        f"queue_share_after {_fmt(queue_share, suffix='%')}",
                        f"hop_p95_delta {_fmt_delta(_delta(hop_after, hop_before), suffix='ms')}",
                        f"decode_p95_delta {_fmt_delta(_delta(decode_after, decode_before), suffix='ms')}",
                    ]
                ),
            }
        )
    return result


def _synthetic_pressure_rows(
    tables_payload: dict[str, Any],
    interactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    rows_by_scenario: dict[str, dict[int, dict[str, Any]]] = {}
    for row in _table_rows(tables_payload, "synthetic_speech"):
        scenario = str(row.get("scenario") or "")
        rows_by_scenario.setdefault(scenario, {})
        try:
            rows_by_scenario[scenario][int(row.get("concurrency"))] = row
        except (TypeError, ValueError):
            continue
    for scenario, pairs in [("short", [(1, 4), (4, 8)]), ("long", [(1, 4), (4, 8)])]:
        rows = rows_by_scenario.get(scenario, {})
        workload = f"synthetic_{scenario}"
        for before_c, after_c in pairs:
            before = rows.get(before_c, {})
            after = rows.get(after_c, {})
            if not before or not after:
                continue
            talker_before = _interaction_metric(
                interactions,
                runtime="sglang",
                workload=workload,
                boundary="talker_to_code2wav_stream",
                concurrency=before_c,
                metric="talker_p95_ms",
            )
            talker_after = _interaction_metric(
                interactions,
                runtime="sglang",
                workload=workload,
                boundary="talker_to_code2wav_stream",
                concurrency=after_c,
                metric="talker_p95_ms",
            )
            hop_before = _interaction_metric(
                interactions,
                runtime="sglang",
                workload=workload,
                boundary="talker_to_code2wav_stream",
                concurrency=before_c,
                metric="talker_to_code2wav_hop_p95_ms",
            )
            hop_after = _interaction_metric(
                interactions,
                runtime="sglang",
                workload=workload,
                boundary="talker_to_code2wav_stream",
                concurrency=after_c,
                metric="talker_to_code2wav_hop_p95_ms",
            )
            decode_before = _interaction_metric(
                interactions,
                runtime="sglang",
                workload=workload,
                boundary="talker_to_code2wav_stream",
                concurrency=before_c,
                metric="code2wav_decode_avg_ms",
            )
            decode_after = _interaction_metric(
                interactions,
                runtime="sglang",
                workload=workload,
                boundary="talker_to_code2wav_stream",
                concurrency=after_c,
                metric="code2wav_decode_avg_ms",
            )
            qps_delta = _pct_delta(after.get("throughput_qps"), before.get("throughput_qps"))
            latency_delta = _pct_delta(after.get("latency_p95_s"), before.get("latency_p95_s"))
            rtf_after = _num(after.get("rtf_p95"))
            below_realtime = rtf_after < 1.0
            verdict = (
                "long_text_realtime_guard_holds"
                if scenario == "long" and below_realtime
                else "short_text_scales_below_realtime"
                if scenario == "short" and below_realtime
                else "synthetic_tail_watch"
            )
            result.append(
                {
                    "id": f"pressure-synthetic-{scenario}-c{before_c}-c{after_c}",
                    "runtime": "sglang",
                    "workload": f"synthetic_{scenario}",
                    "transition": f"c{before_c} -> c{after_c}",
                    "pressure_axis": f"{scenario}_text_concurrency",
                    "qps_delta_pct": qps_delta,
                    "latency_p95_delta_pct": latency_delta,
                    "rtf_p95_after": rtf_after,
                    "talker_p95_delta_ms": _delta(talker_after, talker_before),
                    "hop_p95_delta_ms": _delta(hop_after, hop_before),
                    "decode_avg_delta_ms": _delta(decode_after, decode_before),
                    "verdict": verdict,
                    "decision": (
                        "Synthetic speech remains below real-time; the handoff delta stays small, "
                        "so length pressure maps to Talker cadence rather than code2wav decode."
                    ),
                    "evidence": _transition_evidence(
                        [
                            f"QPS {_fmt_delta(qps_delta, suffix='%')}",
                            f"latency_p95 {_fmt_delta(latency_delta, suffix='%')}",
                            f"RTF_p95_after {_fmt(rtf_after)}",
                            f"talker_p95_delta {_fmt_delta(_delta(talker_after, talker_before), suffix='ms')}",
                            f"hop_p95_delta {_fmt_delta(_delta(hop_after, hop_before), suffix='ms')}",
                            f"decode_avg_delta {_fmt_delta(_delta(decode_after, decode_before), suffix='ms')}",
                        ]
                    ),
                }
            )
    return result


def _vllm_pressure_rows(vllm_admission: dict[str, Any]) -> list[dict[str, Any]]:
    rows = vllm_admission.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    transitions = [
        ("vLLM-c4", "vLLM-c8", "original_concurrency"),
        ("vLLM-c8", "vLLM-c8-prebuild-w4", "prebuild_prompt_feed"),
        ("vLLM-c8-prebuild-w1", "vLLM-c8-prebuild-w4", "prebuild_worker_parallelism"),
    ]
    result: list[dict[str, Any]] = []
    for before_label, after_label, axis in transitions:
        before = _vllm_row(rows, before_label)
        after = _vllm_row(rows, after_label)
        if not before or not after:
            continue
        admission_delta = _pct_delta(
            after.get("batch_admission_span_p95_ms"),
            before.get("batch_admission_span_p95_ms"),
        )
        engine_delta = _pct_delta(after.get("engine_qps"), before.get("engine_qps"))
        wall_delta = _pct_delta(after.get("wall_qps"), before.get("wall_qps"))
        runner_pp_delta = _delta(
            after.get("runner_overhead_pct_wall"),
            before.get("runner_overhead_pct_wall"),
        )
        talker_drain_delta = _delta(
            after.get("talker_to_code2wav_drain_p95_ms"),
            before.get("talker_to_code2wav_drain_p95_ms"),
        )
        if axis == "original_concurrency":
            verdict = "offline_prompt_feed_limited"
            decision = "Do not use original c8 wall QPS as online parity; admission span grows sharply."
        elif axis == "prebuild_prompt_feed":
            verdict = "diagnostic_bottleneck_shift"
            decision = "Prebuild removes most admission span and exposes later engine/talker tail; keep offline caveat."
        else:
            verdict = "runner_parallelism_helps_wall_not_engine"
            decision = "w4 improves runner wall time while engine QPS stays flat; not a serving-parity headline."
        result.append(
            {
                "id": f"pressure-vllm-{before_label}-to-{after_label}",
                "runtime": "vllm",
                "workload": "Video-AMME ci-50 offline",
                "transition": f"{before_label} -> {after_label}",
                "pressure_axis": axis,
                "wall_qps_delta_pct": wall_delta,
                "engine_qps_delta_pct": engine_delta,
                "admission_p95_delta_pct": admission_delta,
                "runner_overhead_delta_pp": runner_pp_delta,
                "talker_drain_p95_delta_ms": talker_drain_delta,
                "verdict": verdict,
                "decision": decision,
                "evidence": _transition_evidence(
                    [
                        f"wall_qps {_fmt_delta(wall_delta, suffix='%')}",
                        f"engine_qps {_fmt_delta(engine_delta, suffix='%')}",
                        f"admission_p95 {_fmt_delta(admission_delta, suffix='%')}",
                        f"runner_overhead {_fmt_delta(runner_pp_delta, suffix='pp')}",
                        f"talker_drain_p95_delta {_fmt_delta(talker_drain_delta, suffix='ms')}",
                        f"after_diagnosis={after.get('diagnosis')}",
                    ]
                ),
            }
        )
    return result


def _pressure_rows(
    tables_payload: dict[str, Any],
    interactions: list[dict[str, Any]],
    vllm_admission: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        *_sglang_pressure_rows(tables_payload, interactions),
        *_synthetic_pressure_rows(tables_payload, interactions),
        *_vllm_pressure_rows(vllm_admission),
    ]


def _evidence(row: dict[str, Any]) -> str:
    metrics = _metrics(row)
    boundary = row.get("boundary")
    if boundary == "request_admission_to_preprocessing":
        return (
            "preproc_lifecycle="
            f"{_fmt_unit(metrics.get('preproc_lifecycle_avg_ms'), 'ms')}; "
            f"actual_preproc={_fmt_unit(metrics.get('actual_preprocess_avg_ms'), 'ms')}; "
            f"queue_est={_fmt_unit(metrics.get('estimated_queue_avg_ms'), 'ms')}; "
            f"top_stage={metrics.get('top_stage')}"
        )
    if boundary == "preprocessing_to_encoder_thinker":
        return (
            f"baseline_qps={_fmt(metrics.get('baseline_qps'), 3)}; "
            f"candidate_qps={_fmt(metrics.get('candidate_qps'), 3)}; "
            f"delta={_fmt(metrics.get('qps_delta_pct'), 1)}%"
        )
    if boundary == "talker_to_code2wav_stream":
        return (
            f"hop_p95={_fmt_unit(metrics.get('talker_to_code2wav_hop_p95_ms'), 'ms')}; "
            f"talker_p95={_fmt_unit(metrics.get('talker_p95_ms'), 'ms')}; "
            f"decode_avg={_fmt_unit(metrics.get('code2wav_decode_avg_ms'), 'ms')}"
        )
    if boundary == "code2wav_collect_to_decode":
        return (
            f"decode_avg/p95={_fmt(metrics.get('code2wav_decode_avg_ms'))}/"
            f"{_fmt_unit(metrics.get('code2wav_decode_p95_ms'), 'ms')}; "
            f"collect_minus_decode={_fmt_unit(metrics.get('window_minus_decode_avg_ms'), 'ms')}"
        )
    if boundary == "runner_to_engine_admission":
        return (
            f"admission_avg/p95={_fmt(metrics.get('batch_admission_span_avg_ms'))}/"
            f"{_fmt_unit(metrics.get('batch_admission_span_p95_ms'), 'ms')}; "
            f"runner_overhead={_fmt_unit(metrics.get('runner_overhead_pct_wall'), '%')}; "
            f"engine_qps={_fmt(metrics.get('engine_qps'), 4)}"
        )
    if boundary == "thinker_to_talker":
        return (
            f"thinker_to_talker_p95="
            f"{_fmt_unit(metrics.get('thinker_to_talker_feed_p95_ms'), 'ms')}; "
            f"diagnosis={_fmt_text(metrics.get('diagnosis'))}"
        )
    if boundary == "talker_to_code2wav":
        return (
            f"drain_p95={_fmt_unit(metrics.get('talker_to_code2wav_drain_p95_ms'), 'ms')}; "
            f"feed_to_first_codec_p95="
            f"{_fmt_unit(metrics.get('feed_to_first_codec_p95_ms'), 'ms')}; "
            f"diagnosis={_fmt_text(metrics.get('diagnosis'))}"
        )
    return ", ".join(f"{key}={value}" for key, value in metrics.items())


def _verdict(row: dict[str, Any]) -> str:
    status = str(row.get("status"))
    if status == "healthy":
        return "not_current_bottleneck"
    if status == "queue_limited":
        return "admission_queue_bottleneck"
    if status == "contention_regression":
        return "negative_optimization"
    if status == "prompt_feed_limited":
        return "offline_runner_prompt_feed_bottleneck"
    if status == "diagnostic_only":
        return "offline_diagnostic_not_parity"
    if status == "watch":
        return "tail_watch"
    return "observed_bottleneck"


def _action(row: dict[str, Any]) -> str:
    status = str(row.get("status"))
    runtime = str(row.get("runtime"))
    boundary = str(row.get("boundary"))
    label = str(row.get("label") or "")
    if runtime == "sglang" and status == "queue_limited":
        return "Use c4-c8 as serving window; treat c16 as saturation boundary and avoid claiming more concurrency as better."
    if status == "contention_regression":
        return "Keep preprocessing concurrency at 1 unless placement/admission is redesigned; preproc=2 is a negative recipe."
    if runtime == "sglang" and boundary == "talker_to_code2wav_stream":
        return "No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence."
    if runtime == "sglang" and boundary == "code2wav_collect_to_decode":
        return "Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window."
    if runtime == "vllm" and status == "prompt_feed_limited":
        return "Use as vLLM offline runner diagnosis; do not compare c8 wall time as online serving parity."
    if runtime == "vllm" and status == "diagnostic_only":
        return "Keep as offline diagnostic; do not promote c8 parity without online ingress artifacts."
    if runtime == "vllm" and label.endswith("prebuild-w4"):
        return "Monitor engine/talker tail after prompt feed is removed; keep caveat attached."
    if status == "bottleneck":
        return "Treat as a post-admission tail exposed by diagnostic mode; isolate before making a headline claim."
    return "Keep current recipe; this boundary is not the limiting stage under this condition."


def _scope(row: dict[str, Any]) -> str:
    runtime = str(row.get("runtime"))
    status = str(row.get("status"))
    label = str(row.get("label") or "")
    if runtime == "sglang":
        if status == "queue_limited":
            return "SGLang high-concurrency limit / serving-window decision"
        if status == "contention_regression":
            return "SGLang anti-recipe guardrail"
        return "SGLang stage connection health"
    if label == "vLLM-c4":
        return "strict c4 baseline comparison"
    if "prebuild" in label:
        return "offline diagnostic only"
    if status == "prompt_feed_limited":
        return "offline admission diagnosis"
    return "vLLM stage connection health"


def _ledger_rows(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(interactions, start=1):
        rows.append(
            {
                "id": f"boundary-{index:02d}",
                "runtime": row.get("runtime"),
                "case": _case_name(row),
                "boundary": row.get("boundary"),
                "status": row.get("status"),
                "bottleneck_verdict": _verdict(row),
                "evidence": _evidence(row),
                "decision": _action(row),
                "claim_scope": _scope(row),
                "source_interpretation": row.get("interpretation"),
                "source_metrics": _metrics(row),
            }
        )
    return rows


def _find(
    rows: list[dict[str, Any]],
    *,
    runtime: str,
    boundary: str,
    status: str | None = None,
    label: str | None = None,
    case_contains: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("runtime") != runtime:
            continue
        if row.get("boundary") != boundary:
            continue
        if status is not None and row.get("status") != status:
            continue
        if label is not None and row.get("case") != label:
            continue
        if case_contains is not None and case_contains not in str(row.get("case")):
            continue
        result.append(row)
    return result


def build_ledger(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    interactions_payload = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    tables_payload = _load_json_optional(audit_dir / "tables_summary.json")
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    vllm_admission = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    interactions = _interaction_rows(interactions_payload)
    ledger_rows = _ledger_rows(interactions)
    pressure_rows = _pressure_rows(tables_payload, interactions, vllm_admission)
    status_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for row in ledger_rows:
        status = str(row.get("status"))
        verdict = str(row.get("bottleneck_verdict"))
        status_counts[status] = status_counts.get(status, 0) + 1
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    malformed_evidence_tokens = [
        token
        for row in ledger_rows
        for token in ("n/ams", "n/a%", "nanms", "None")
        if token in str(row.get("evidence"))
    ]

    stage_summary = stage_budget.get("summary", {})
    c8_queue_pct = _num(stage_summary.get("c8_queue_pct_of_latency"))
    c16_queue_pct = _num(stage_summary.get("c16_queue_pct_of_latency"))
    long_c8_rtf = _num(stage_summary.get("long_c8_rtf_mean"), 999.0)
    checks: list[LedgerCheck] = []
    _check(
        checks,
        "interaction rows are represented",
        len(ledger_rows) >= 37,
        f"ledger_rows={len(ledger_rows)}, source_rows={len(interactions)}",
    )
    _check(
        checks,
        "SGLang c8/c16 queue boundary is explicit",
        bool(
            _find(
                ledger_rows,
                runtime="sglang",
                boundary="request_admission_to_preprocessing",
                status="queue_limited",
                case_contains="c=8",
            )
        )
        and bool(
            _find(
                ledger_rows,
                runtime="sglang",
                boundary="request_admission_to_preprocessing",
                status="queue_limited",
                case_contains="c=16",
            )
        )
        and c8_queue_pct >= 25.0
        and c16_queue_pct >= 60.0,
        f"c8_queue_pct={c8_queue_pct}, c16_queue_pct={c16_queue_pct}",
    )
    _check(
        checks,
        "SGLang stream hop stays non-bottleneck",
        all(
            row.get("status") == "healthy"
            for row in _find(
                ledger_rows,
                runtime="sglang",
                boundary="talker_to_code2wav_stream",
            )
        ),
        "all SGLang talker_to_code2wav_stream rows are healthy",
    )
    _check(
        checks,
        "SGLang code2wav decode stays non-bottleneck",
        all(
            row.get("status") == "healthy"
            for row in _find(
                ledger_rows,
                runtime="sglang",
                boundary="code2wav_collect_to_decode",
            )
        ),
        "all SGLang code2wav_collect_to_decode rows are healthy",
    )
    _check(
        checks,
        "preprocessing concurrency anti-recipe is preserved",
        bool(
            _find(
                ledger_rows,
                runtime="sglang",
                boundary="preprocessing_to_encoder_thinker",
                status="contention_regression",
            )
        ),
        "preproc=2 row is marked contention_regression",
    )
    _check(
        checks,
        "long synthetic c8 remains faster than real time",
        long_c8_rtf < 1.0
        and bool(
            _find(
                ledger_rows,
                runtime="sglang",
                boundary="talker_to_code2wav_stream",
                case_contains="synthetic_long c=8",
            )
        ),
        f"long_c8_rtf={long_c8_rtf}",
    )
    _check(
        checks,
        "vLLM c8 prompt-feed limit is explicit",
        bool(
            _find(
                ledger_rows,
                runtime="vllm",
                boundary="runner_to_engine_admission",
                status="prompt_feed_limited",
                label="vLLM-c8",
            )
        ),
        "vLLM-c8 runner_to_engine_admission is prompt_feed_limited",
    )
    _check(
        checks,
        "vLLM prebuild w4 stays diagnostic-only",
        bool(
            _find(
                ledger_rows,
                runtime="vllm",
                boundary="runner_to_engine_admission",
                status="diagnostic_only",
                label="vLLM-c8-prebuild-w4",
            )
        )
        and stage_summary.get("vllm_w4_scope") == "offline_diagnostic_only",
        f"vllm_w4_scope={stage_summary.get('vllm_w4_scope')}",
    )
    _check(
        checks,
        "every ledger row has decision evidence and scope",
        all(
            row.get("decision") and row.get("evidence") and row.get("claim_scope")
            for row in ledger_rows
        )
        and not malformed_evidence_tokens,
        (
            "decision/evidence/claim_scope populated for all rows; "
            f"malformed_evidence_tokens={malformed_evidence_tokens}"
        ),
    )
    _check(
        checks,
        "claim verifier remains green",
        bool(claims.get("passed")) and int(claims.get("failed_checks") or 0) == 0,
        f"claims_passed={claims.get('passed')}, failed={claims.get('failed_checks')}",
    )
    pressure_ids = {str(row.get("id")) for row in pressure_rows}
    c8_c16_pressure = next(
        (
            row
            for row in pressure_rows
            if row.get("id") == "pressure-sglang-c8-c16"
        ),
        {},
    )
    long_c4_c8_pressure = next(
        (
            row
            for row in pressure_rows
            if row.get("id") == "pressure-synthetic-long-c4-c8"
        ),
        {},
    )
    vllm_prebuild_pressure = next(
        (
            row
            for row in pressure_rows
            if row.get("id") == "pressure-vllm-vLLM-c8-to-vLLM-c8-prebuild-w4"
        ),
        {},
    )
    _check(
        checks,
        "pressure propagation matrix covers concurrency, length, and vLLM diagnostics",
        len(pressure_rows) >= 11
        and "pressure-sglang-c8-c16" in pressure_ids
        and "pressure-synthetic-long-c4-c8" in pressure_ids
        and "pressure-vllm-vLLM-c8-to-vLLM-c8-prebuild-w4" in pressure_ids,
        f"pressure_rows={len(pressure_rows)}, required_ids_present={sorted(pressure_ids)[:4]}...",
    )
    _check(
        checks,
        "pressure propagation conclusions preserve serving-window and diagnostic boundaries",
        _num(c8_c16_pressure.get("qps_delta_pct")) < 0
        and _num(c8_c16_pressure.get("queue_delta_ms")) >= 3000
        and _num(long_c4_c8_pressure.get("rtf_p95_after"), default=999.0) < 1.0
        and vllm_prebuild_pressure.get("verdict") == "diagnostic_bottleneck_shift",
        (
            f"c8_to_c16_qps_delta={c8_c16_pressure.get('qps_delta_pct')}; "
            f"c8_to_c16_queue_delta={c8_c16_pressure.get('queue_delta_ms')}; "
            f"long_c8_rtf_p95={long_c4_c8_pressure.get('rtf_p95_after')}; "
            f"vllm_prebuild_verdict={vllm_prebuild_pressure.get('verdict')}"
        ),
    )

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
            "ledger_rows": len(ledger_rows),
            "pressure_transition_rows": len(pressure_rows),
            "status_counts": status_counts,
            "verdict_counts": verdict_counts,
            "recommended_sglang_window": "c4-c8",
            "saturation_boundary": "c16",
            "vllm_c8_scope": "offline_diagnostic_until_online_ingress_artifacts",
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "method_note": (
            "This ledger restates stage_interaction_summary rows as reviewable "
            "bottleneck decisions. It does not introduce new benchmark numbers; "
            "it binds each stage boundary to evidence, decision, and claim scope. "
            "The pressure propagation matrix derives adjacent-regime deltas from "
            "the same audited summaries to explain how pressure moves across stage "
            "boundaries."
        ),
        "ledger_rows": ledger_rows,
        "pressure_transition_rows": pressure_rows,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = [
        "# Qwen3.5-Omni Stage Boundary Bottleneck Ledger",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份台账把 stage interaction 的每一条边界转换为可复核的瓶颈判定：",
        "当前是否是瓶颈、证据数字是什么、对优化和 headline 的约束是什么。",
        "",
        "## 1. Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| Ledger rows | {summary['ledger_rows']} |",
        f"| Pressure transition rows | {summary['pressure_transition_rows']} |",
        f"| Recommended SGLang window | `{summary['recommended_sglang_window']}` |",
        f"| Saturation boundary | `{summary['saturation_boundary']}` |",
        f"| vLLM c8 scope | `{summary['vllm_c8_scope']}` |",
        "",
        "## 2. Reviewer 读法",
        "",
        "- `healthy` 表示该 stage 边界不是当前优化优先级。",
        "- `queue_limited` 表示并发压力主要进入 admission/queue，不等于实际 preprocessing compute 变慢。",
        "- `contention_regression` 表示反例 recipe，不能当作优化结论。",
        "- `prompt_feed_limited` 表示 vLLM offline runner 的 prompt/feed 限制，不能外推为 online serving parity。",
        "- `diagnostic_only` 表示只用于定位瓶颈转移，不能直接提升 headline。",
        "",
        "## 3. Boundary Ledger",
        "",
        "| ID | Runtime | Case | Boundary | Status | Verdict | Evidence | Decision | Scope |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["ledger_rows"]:
        evidence = str(row["evidence"]).replace("|", "\\|")
        decision = str(row["decision"]).replace("|", "\\|")
        scope = str(row["claim_scope"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['runtime']} | {row['case']} | "
            f"{row['boundary']} | {row['status']} | {row['bottleneck_verdict']} | "
            f"{evidence} | {decision} | {scope} |"
        )

    lines.extend(
        [
            "",
            "## 4. Pressure Propagation Matrix",
            "",
            "这张表回答的是：并发、文本长度或 vLLM offline runner 形态变化时，压力沿哪个 stage 边界传导。",
            "它只使用已审计 summary/interaction 数据派生相邻档位 delta，不引入新的 benchmark 数字。",
            "",
            "| ID | Runtime | Workload | Transition | Axis | Verdict | Evidence | Decision |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["pressure_transition_rows"]:
        evidence = str(row["evidence"]).replace("|", "\\|")
        decision = str(row["decision"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['runtime']} | {row['workload']} | "
            f"{row['transition']} | {row['pressure_axis']} | {row['verdict']} | "
            f"{evidence} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## 5. Machine Checks",
            "",
            "| Status | Required | Check | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        required = "yes" if check["required"] else "no"
        evidence = str(check["evidence"]).replace("|", "\\|")
        lines.append(
            f"| {check['status']} | {required} | {check['name']} | {evidence} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Stage Boundary Bottleneck Ledger\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Ledger rows | {summary['ledger_rows']} |")
    print(f"| Pressure transition rows | {summary['pressure_transition_rows']} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni stage-boundary bottleneck ledger."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_ledger(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    _save_text(build_markdown(payload), output)
    _save_json(payload, json_output)
    print_markdown(payload)
    print(
        "Stage boundary bottleneck ledger written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
