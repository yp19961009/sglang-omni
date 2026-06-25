# SPDX-License-Identifier: Apache-2.0
"""Build a serving/capacity decision matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "serving_capacity_matrix.json"

REQUIRED_TAIL_CASES = {
    "sglang_stress_c1",
    "sglang_stress_c2",
    "sglang_stress_c4",
    "sglang_stress_c8",
    "sglang_stress_c16",
    "synthetic_short_c8",
    "synthetic_long_c8",
    "vllm_c8_prebuild_w4",
}


@dataclass(frozen=True)
class MatrixCheck:
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


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _num(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "n/a"


def _ms(value: Any) -> str:
    return _num(value, 1, "ms")


def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("\n", " ").replace("|", "\\|")


def _tail_rows_by_case(tail_confidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in tail_confidence.get("rows", [])
        if isinstance(row, dict) and row.get("case_id")
    }


def _tail_metric(row: dict[str, Any], metric: str, stat: str) -> Any:
    value = row.get(metric, {})
    if isinstance(value, dict):
        return value.get(stat)
    return None


def _budget_by_concurrency(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            result[int(row.get("concurrency"))] = row
        except Exception:
            continue
    return result


def _synthetic_budget_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            result[(str(row.get("scenario")), int(row.get("concurrency")))] = row
        except Exception:
            continue
    return result


def _pressure_by_id(stage_ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id")): row
        for row in stage_ledger.get("pressure_transition_rows", [])
        if isinstance(row, dict) and row.get("id")
    }


def _qps_latency_rtf(row: dict[str, Any]) -> str:
    return (
        f"QPS={_num(row.get('throughput_qps'), 3)}; "
        f"lat_p95={_num(_tail_metric(row, 'latency_s', 'p95'), 3, 's')}; "
        f"RTF_p95={_num(_tail_metric(row, 'rtf', 'p95'), 4)}"
    )


def _queue_guard(row: dict[str, Any]) -> str:
    queue_ms = row.get("preproc_queue_estimate_ms")
    queue_pct = row.get("preproc_queue_pct_of_latency")
    if queue_ms is None or queue_pct is None:
        return "queue=n/a"
    return f"queue={_ms(queue_ms)} / {_num(queue_pct, 1, '%')}"


def _case_file(row: dict[str, Any], key: str = "result_path") -> str:
    return str(row.get(key) or "")


def _make_rows(
    stage_budget: dict[str, Any],
    tail_confidence: dict[str, Any],
    stage_ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    tails = _tail_rows_by_case(tail_confidence)
    sglang_budget = _budget_by_concurrency(stage_budget.get("sglang_videoamme_budget", []))
    synthetic_budget = _synthetic_budget_by_key(
        stage_budget.get("synthetic_speech_budget", [])
    )
    vllm_budget = {
        str(row.get("workload")): row
        for row in stage_budget.get("vllm_offline_budget", [])
        if isinstance(row, dict)
    }
    pressure = _pressure_by_id(stage_ledger)

    c1 = tails.get("sglang_stress_c1", {})
    c2 = tails.get("sglang_stress_c2", {})
    c4 = tails.get("sglang_stress_c4", {})
    c8 = tails.get("sglang_stress_c8", {})
    c16 = tails.get("sglang_stress_c16", {})
    short8 = tails.get("synthetic_short_c8", {})
    long8 = tails.get("synthetic_long_c8", {})
    short_budget = synthetic_budget.get(("short", 8), {})
    long_budget = synthetic_budget.get(("long", 8), {})
    vllm_w4 = vllm_budget.get("vLLM-c8-prebuild-w4", {})

    return [
        {
            "id": "capacity-videoamme-c1-c2",
            "runtime": "sglang",
            "pressure_or_scenario": "Video-AMME c=1-c2",
            "run_choice": "latency-first / single-to-low concurrency",
            "decision": "Use as the latency-first serving guard for low concurrency.",
            "primary_metrics": (
                f"QPS={_num(c1.get('throughput_qps'), 3)}-{_num(c2.get('throughput_qps'), 3)}; "
                f"lat_p95={_num(_tail_metric(c1, 'latency_s', 'p95'), 3, 's')}-"
                f"{_num(_tail_metric(c2, 'latency_s', 'p95'), 3, 's')}; "
                f"RTF_p95={_num(_tail_metric(c1, 'rtf', 'p95'), 4)}-"
                f"{_num(_tail_metric(c2, 'rtf', 'p95'), 4)}"
            ),
            "stage_guard": "Talker AR tail dominates; handoff/decode p95 remains small.",
            "do_not_say": "Do not describe low-concurrency talker tail as high-concurrency admission saturation.",
            "rerun_command_ids": ["sglang_videoamme_stress", "build_tail_confidence_appendix"],
            "evidence_files": [_case_file(c1), _case_file(c2)],
        },
        {
            "id": "capacity-videoamme-c4",
            "runtime": "sglang",
            "pressure_or_scenario": "Video-AMME c=4",
            "run_choice": "balanced serving / strict-headline reference",
            "decision": "Use c4 as the balanced serving point and keep strict warmed c4 as the cross-runtime headline.",
            "primary_metrics": f"{_qps_latency_rtf(c4)}; {_queue_guard(sglang_budget.get(4, {}))}",
            "stage_guard": "Talker AR remains primary; queue is still bounded.",
            "do_not_say": "Do not mix stress c4 and strict warmed c4 as one metric slice.",
            "rerun_command_ids": [
                "sglang_videoamme_stress",
                "vllm_c4_original",
                "build_tail_confidence_appendix",
            ],
            "evidence_files": [_case_file(c4)],
        },
        {
            "id": "capacity-videoamme-c8",
            "runtime": "sglang",
            "pressure_or_scenario": "Video-AMME c=8",
            "run_choice": "throughput edge / current high-concurrency sweet spot",
            "decision": "Use c8 as the current throughput-oriented serving edge.",
            "primary_metrics": f"{_qps_latency_rtf(c8)}; {_queue_guard(sglang_budget.get(8, {}))}",
            "stage_guard": "Admission queue becomes visible, but QPS is still the current peak and handoff is not the bottleneck.",
            "do_not_say": "Do not widen preprocessing concurrency; preproc=2/4 are measured anti-recipes.",
            "rerun_command_ids": ["sglang_videoamme_stress", "build_stage_latency_budget"],
            "evidence_files": [_case_file(c8), _case_file(sglang_budget.get(8, {}), "result_path")],
            "pressure_transition": pressure.get("pressure-sglang-c4-c8", {}).get("evidence"),
        },
        {
            "id": "capacity-videoamme-c16",
            "runtime": "sglang",
            "pressure_or_scenario": "Video-AMME c=16",
            "run_choice": "saturation evidence / pressure boundary",
            "decision": "Keep c16 as saturation evidence, not the recommended serving point.",
            "primary_metrics": f"{_qps_latency_rtf(c16)}; {_queue_guard(sglang_budget.get(16, {}))}",
            "stage_guard": "Throughput is below c8 while queue share and RTF tail rise sharply.",
            "do_not_say": "Do not present c16 as better high-concurrency serving.",
            "rerun_command_ids": ["sglang_videoamme_stress", "build_stage_boundary_bottleneck_ledger"],
            "evidence_files": [_case_file(c16)],
            "pressure_transition": pressure.get("pressure-sglang-c8-c16", {}).get("evidence"),
        },
        {
            "id": "capacity-synthetic-short-c8",
            "runtime": "sglang",
            "pressure_or_scenario": "Synthetic short text c=8",
            "run_choice": "short-text speech high-concurrency guard",
            "decision": "Use as the short-text speech regression guard.",
            "primary_metrics": (
                f"{_qps_latency_rtf(short8)}; "
                f"audio={_num(short_budget.get('audio_duration_mean_s'), 1, 's')}; "
                f"hop_p95={_ms(short_budget.get('talker_to_code2wav_hop_p95_ms'))}"
            ),
            "stage_guard": "Short text remains faster than real time; code2wav decode share is small.",
            "do_not_say": "Do not extrapolate short-text throughput to long text.",
            "rerun_command_ids": [
                "sglang_synthetic_text_to_speech",
                "build_tail_confidence_appendix",
            ],
            "evidence_files": [_case_file(short8)],
            "pressure_transition": pressure.get("pressure-synthetic-short-c4-c8", {}).get("evidence"),
        },
        {
            "id": "capacity-synthetic-long-c8",
            "runtime": "sglang",
            "pressure_or_scenario": "Synthetic long text c=8",
            "run_choice": "long-text/long-speech realtime guard",
            "decision": "Use as the long-text realtime guard.",
            "primary_metrics": (
                f"{_qps_latency_rtf(long8)}; "
                f"audio={_num(long_budget.get('audio_duration_mean_s'), 1, 's')}; "
                f"hop_p95={_ms(long_budget.get('talker_to_code2wav_hop_p95_ms'))}"
            ),
            "stage_guard": "Long c8 RTF p95 remains below 1; pressure maps to Talker cadence.",
            "do_not_say": "Do not call vocoder decode the primary bottleneck.",
            "rerun_command_ids": [
                "sglang_synthetic_text_to_speech",
                "build_stage_latency_budget",
            ],
            "evidence_files": [_case_file(long8)],
            "pressure_transition": pressure.get("pressure-synthetic-long-c4-c8", {}).get("evidence"),
        },
        {
            "id": "capacity-vllm-c8-prebuild-w4",
            "runtime": "vllm",
            "pressure_or_scenario": "vLLM c=8 prebuild w4",
            "run_choice": "optimized offline diagnostic",
            "decision": "Use as the strongest current vLLM c8 offline diagnostic.",
            "primary_metrics": (
                f"runner_QPS={_num(vllm_w4.get('runner_qps'), 4)}; "
                f"engine_QPS={_num(vllm_w4.get('engine_qps'), 4)}; "
                f"admission_p95={_ms(vllm_w4.get('batch_admission_span_p95_ms'))}"
            ),
            "stage_guard": "Prebuild removes most prompt-feed admission and exposes engine/talker tail.",
            "do_not_say": (
                "Do not promote this to online serving parity without online ingress "
                "plus WER/ASR; 不要升级为 online serving parity."
            ),
            "rerun_command_ids": [
                "vllm_c8_prebuild_w4",
                "diagnose_vllm_admission",
            ],
            "evidence_files": [],
            "pressure_transition": pressure.get("pressure-vllm-vLLM-c8-to-vLLM-c8-prebuild-w4", {}).get("evidence"),
        },
    ]


def _check_rows(rows: list[dict[str, Any]], checks: list[MatrixCheck]) -> None:
    checks.append(
        MatrixCheck(
            "all capacity rows have decisions and guardrails",
            _status(
                len(rows) >= 7
                and all(row.get("decision") and row.get("stage_guard") for row in rows)
                and all(row.get("do_not_say") for row in rows)
            ),
            f"rows={len(rows)}",
        )
    )
    checks.append(
        MatrixCheck(
            "capacity rows are reproducible",
            _status(all(row.get("rerun_command_ids") for row in rows)),
            "missing_command_rows="
            + ",".join(row["id"] for row in rows if not row.get("rerun_command_ids")),
        )
    )


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    tail_confidence = _load_json_optional(audit_dir / "tail_confidence_appendix.json")
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_ledger = _load_json_optional(audit_dir / "stage_boundary_bottleneck_ledger.json")

    tail_summary = tail_confidence.get("summary", {})
    stage_budget_summary = stage_budget.get("summary", {})
    stage_ledger_summary = stage_ledger.get("summary", {})
    tails = _tail_rows_by_case(tail_confidence)
    sglang_budget = _budget_by_concurrency(stage_budget.get("sglang_videoamme_budget", []))
    synthetic_budget = _synthetic_budget_by_key(
        stage_budget.get("synthetic_speech_budget", [])
    )
    vllm_budget = {
        str(row.get("workload")): row
        for row in stage_budget.get("vllm_offline_budget", [])
        if isinstance(row, dict)
    }
    pressure = _pressure_by_id(stage_ledger)
    rows = _make_rows(stage_budget, tail_confidence, stage_ledger)

    c4 = tails.get("sglang_stress_c4", {})
    c8 = tails.get("sglang_stress_c8", {})
    c16 = tails.get("sglang_stress_c16", {})
    short8 = tails.get("synthetic_short_c8", {})
    long8 = tails.get("synthetic_long_c8", {})
    vllm_w4 = vllm_budget.get("vLLM-c8-prebuild-w4", {})

    missing_tail_cases = sorted(REQUIRED_TAIL_CASES - set(tails))
    checks: list[MatrixCheck] = [
        MatrixCheck(
            "source summaries are ready",
            _status(
                bool(tail_summary.get("ready"))
                and bool(stage_budget_summary.get("ready"))
                and bool(stage_ledger_summary.get("ready"))
            ),
            (
                f"tail={tail_summary}; stage_budget={stage_budget_summary}; "
                f"stage_ledger={stage_ledger_summary}"
            ),
        ),
        MatrixCheck(
            "required tail cases are present",
            _status(not missing_tail_cases),
            "missing=" + ",".join(missing_tail_cases),
        ),
        MatrixCheck(
            "stage budgets cover serving regimes",
            _status(
                len(sglang_budget) >= 5
                and len(synthetic_budget) >= 6
                and len(vllm_budget) >= 4
            ),
            (
                f"sglang={len(sglang_budget)}, synthetic={len(synthetic_budget)}, "
                f"vllm={len(vllm_budget)}"
            ),
        ),
        MatrixCheck(
            "SGLang c8 is the measured throughput edge",
            _status(
                float(c8.get("throughput_qps") or 0)
                > float(c4.get("throughput_qps") or 0)
                and float(c8.get("throughput_qps") or 0)
                > float(c16.get("throughput_qps") or 0)
            ),
            (
                f"c4={c4.get('throughput_qps')}, c8={c8.get('throughput_qps')}, "
                f"c16={c16.get('throughput_qps')}"
            ),
        ),
        MatrixCheck(
            "SGLang c16 is a saturation boundary",
            _status(
                float(c16.get("throughput_qps") or 0)
                < float(c8.get("throughput_qps") or 0)
                and float(sglang_budget.get(16, {}).get("preproc_queue_pct_of_latency") or 0)
                > float(sglang_budget.get(8, {}).get("preproc_queue_pct_of_latency") or 0)
                and pressure.get("pressure-sglang-c8-c16", {}).get("verdict")
                == "saturation_boundary"
            ),
            (
                f"c8_qps={c8.get('throughput_qps')}, c16_qps={c16.get('throughput_qps')}, "
                f"c8_queue={sglang_budget.get(8, {}).get('preproc_queue_pct_of_latency')}, "
                f"c16_queue={sglang_budget.get(16, {}).get('preproc_queue_pct_of_latency')}"
            ),
        ),
        MatrixCheck(
            "synthetic c8 speech guards remain faster than real time",
            _status(
                float(_tail_metric(short8, "rtf", "p95") or 9) < 1.0
                and float(_tail_metric(long8, "rtf", "p95") or 9) < 1.0
            ),
            (
                f"short_rtf_p95={_tail_metric(short8, 'rtf', 'p95')}, "
                f"long_rtf_p95={_tail_metric(long8, 'rtf', 'p95')}"
            ),
        ),
        MatrixCheck(
            "vLLM c8 prebuild w4 remains offline diagnostic",
            _status(
                vllm_w4.get("valid_comparison_scope") == "offline_diagnostic_only"
                and float(vllm_w4.get("engine_qps") or 0)
                > float(vllm_w4.get("runner_qps") or 0)
            ),
            (
                f"scope={vllm_w4.get('valid_comparison_scope')}, "
                f"runner={vllm_w4.get('runner_qps')}, engine={vllm_w4.get('engine_qps')}"
            ),
        ),
        MatrixCheck(
            "pressure transitions are attached",
            _status(
                int(stage_ledger_summary.get("pressure_transition_rows") or 0) >= 11
                and "pressure-sglang-c8-c16" in pressure
                and "pressure-vllm-vLLM-c8-to-vLLM-c8-prebuild-w4" in pressure
            ),
            f"pressure_transition_rows={stage_ledger_summary.get('pressure_transition_rows')}",
        ),
    ]
    _check_rows(rows, checks)

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
            "rows_total": len(rows),
            "sglang_c8_qps": c8.get("throughput_qps"),
            "sglang_c16_qps": c16.get("throughput_qps"),
            "long_c8_rtf_p95": _tail_metric(long8, "rtf", "p95"),
            "vllm_w4_scope": vllm_w4.get("valid_comparison_scope"),
            "share_scope": (
                "Standalone serving/capacity decision matrix tying pressure regimes "
                "to run choices, stage guardrails, forbidden claims, evidence files, "
                "and rerun command IDs."
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "rows": rows,
        "sources": {
            "tail_confidence_appendix": tail_summary,
            "stage_latency_budget": stage_budget_summary,
            "stage_boundary_bottleneck_ledger": stage_ledger_summary,
        },
    }


def build_markdown(root: Path, payload: dict[str, Any] | None = None) -> str:
    payload = payload or build_payload(root)
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni Serving/Capacity 决策矩阵",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份矩阵把已审计的压力测试结果翻译成运行选择：哪些是服务窗口，哪些只是压力边界或诊断证据。",
        "所有数字来自 `tail_confidence_appendix.json`、`stage_latency_budget.json` 和 `stage_boundary_bottleneck_ledger.json`。",
        "",
        "## 1. Gate",
        "",
        f"- ready：`{summary.get('ready')}`，checks：`{summary.get('checks_passed')}/{summary.get('checks_total')}`，rows：`{summary.get('rows_total')}`。",
        f"- SGLang c8 QPS：`{summary.get('sglang_c8_qps')}`；SGLang c16 QPS：`{summary.get('sglang_c16_qps')}`。",
        f"- long c8 RTF p95：`{summary.get('long_c8_rtf_p95')}`；vLLM w4 scope：`{summary.get('vllm_w4_scope')}`。",
        "",
        "## 2. 决策矩阵",
        "",
        "| 压力/场景 | Runtime | 运行选择 | 可承诺指标 | Stage guard | 不要做 | Rerun IDs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("pressure_or_scenario")),
                    _cell(row.get("runtime")),
                    _cell(row.get("run_choice")),
                    _cell(row.get("primary_metrics")),
                    _cell(row.get("stage_guard")),
                    _cell(row.get("do_not_say")),
                    _cell(", ".join(f"`{item}`" for item in row.get("rerun_command_ids", []))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. Pressure transition 证据",
            "",
            "| ID | Pressure transition evidence |",
            "| --- | --- |",
        ]
    )
    for row in payload["rows"]:
        transition = row.get("pressure_transition")
        if transition:
            lines.append(f"| {_cell(row.get('id'))} | {_cell(transition)} |")
    lines.extend(
        [
            "",
            "## 4. Check 明细",
            "",
            "| Status | Required | Check | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(check.get("status")),
                    "yes" if check.get("required") else "no",
                    _cell(check.get("name")),
                    _cell(check.get("evidence")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 5. 机器证据",
            "",
            "- `results/qwen35_report_audit_20260619/serving_capacity_matrix.json`",
            "- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`",
            "- `results/qwen35_report_audit_20260619/stage_latency_budget.json`",
            "- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`",
            "- `results/qwen35_report_audit_20260619/repro_command_manifest.json`",
            "- Rebuild command ID: `build_serving_capacity_matrix`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni serving/capacity decision matrix."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    payload = build_payload(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(root, payload), encoding="utf-8")
    _save_json(payload, json_output)
    summary = payload["summary"]
    print(
        "Serving/capacity matrix written: "
        f"{output} json={json_output} ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"rows={summary['rows_total']}"
    )
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
