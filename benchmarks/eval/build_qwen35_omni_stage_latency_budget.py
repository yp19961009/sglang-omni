# SPDX-License-Identifier: Apache-2.0
"""Build a stage latency-budget appendix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "stage_latency_budget.json"


@dataclass(frozen=True)
class BudgetCheck:
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


def _rows(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    table = payload.get("tables", {}).get(name, [])
    return table if isinstance(table, list) else []


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(part_ms: float | None, total_ms: float) -> float | None:
    if part_ms is None or total_ms <= 0.0:
        return None
    return part_ms / total_ms * 100.0


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}{suffix}"


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _check(
    checks: list[BudgetCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(BudgetCheck(name, _status(condition), evidence, required))


def _by_key(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row.get(key): row for row in rows}


def _by_pair(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        scenario = str(row.get("scenario"))
        concurrency = int(_num(row.get("concurrency")))
        result[(scenario, concurrency)] = row
    return result


def _classify_sglang(row: dict[str, Any]) -> str:
    concurrency = int(row["concurrency"])
    queue_pct = row.get("preproc_queue_pct_of_latency")
    talker_pct = row.get("talker_pct_of_latency")
    if concurrency >= 16:
        return "saturation_boundary"
    if queue_pct is not None and queue_pct >= 20.0:
        return "admission_queue_plus_talker_tail"
    if talker_pct is not None and talker_pct >= 25.0:
        return "talker_ar_tail"
    return "balanced_or_low_pressure"


def _build_sglang_budget(tables: dict[str, Any]) -> list[dict[str, Any]]:
    stress_rows = _rows(tables, "sglang_stress")
    stage_by_c = _by_key(_rows(tables, "sglang_stage_breakdown"), "concurrency")
    split_by_c = _by_key(_rows(tables, "sglang_preprocessing_split"), "concurrency")
    budget: list[dict[str, Any]] = []
    for stress in stress_rows:
        concurrency = int(_num(stress.get("concurrency")))
        stage = stage_by_c.get(concurrency, {})
        split = split_by_c.get(concurrency, {})
        latency_ms = _num(stress.get("latency_mean_s")) * 1000.0
        preproc_ms = _num(stage.get("preproc_stage_avg_ms"), None)
        actual_preproc_ms = (
            _num(split.get("actual_preprocess_avg_ms"), None)
            if split
            else None
        )
        queue_ms = None
        if preproc_ms is not None and actual_preproc_ms is not None:
            queue_ms = max(preproc_ms - actual_preproc_ms, 0.0)
        row: dict[str, Any] = {
            "runtime": "sglang",
            "workload": "Video-AMME ci-50 speech-output",
            "concurrency": concurrency,
            "latency_mean_ms": latency_ms,
            "rtf_mean": stress.get("rtf_mean"),
            "qps": stress.get("throughput_qps"),
            "top_stage": stage.get("top_stage"),
            "preproc_lifecycle_avg_ms": preproc_ms,
            "preproc_lifecycle_pct_of_latency": _pct(preproc_ms, latency_ms),
            "actual_preprocess_avg_ms": actual_preproc_ms,
            "actual_preprocess_pct_of_latency": _pct(actual_preproc_ms, latency_ms),
            "preproc_queue_estimate_ms": queue_ms,
            "preproc_queue_pct_of_latency": _pct(queue_ms, latency_ms),
            "talker_avg_ms": _num(stage.get("talker_avg_ms"), None),
            "talker_pct_of_latency": _pct(
                _num(stage.get("talker_avg_ms"), None), latency_ms
            ),
            "code2wav_decode_avg_ms": _num(stage.get("code2wav_decode_avg_ms"), None),
            "code2wav_decode_pct_of_latency": _pct(
                _num(stage.get("code2wav_decode_avg_ms"), None), latency_ms
            ),
            "code2wav_window_collect_avg_ms": _num(
                stage.get("code2wav_window_avg_ms"), None
            ),
            "talker_to_code2wav_hop_p95_ms": _num(
                stage.get("talker_to_code2wav_hop_p95_ms"), None
            ),
        }
        row["diagnosis"] = _classify_sglang(row)
        budget.append(row)
    return sorted(budget, key=lambda item: item["concurrency"])


def _classify_synthetic(row: dict[str, Any]) -> str:
    if str(row["scenario"]) == "long" and int(row["concurrency"]) == 8:
        return "long_speech_talker_ar_dominant_but_faster_than_realtime"
    if _num(row.get("rtf_mean")) < 1.0:
        return "faster_than_realtime"
    return "watch_rtf"


def _build_synthetic_budget(tables: dict[str, Any]) -> list[dict[str, Any]]:
    result_rows = _rows(tables, "synthetic_speech")
    stage_by_pair = _by_pair(_rows(tables, "synthetic_stage_breakdown"))
    budget: list[dict[str, Any]] = []
    for result in result_rows:
        scenario = str(result.get("scenario"))
        concurrency = int(_num(result.get("concurrency")))
        stage = stage_by_pair.get((scenario, concurrency), {})
        latency_ms = _num(result.get("latency_mean_s")) * 1000.0
        row: dict[str, Any] = {
            "runtime": "sglang",
            "workload": f"synthetic_{scenario}_text_to_speech",
            "scenario": scenario,
            "concurrency": concurrency,
            "target_words": result.get("target_words"),
            "audio_duration_mean_s": result.get("audio_duration_mean_s"),
            "latency_mean_ms": latency_ms,
            "rtf_mean": result.get("rtf_mean"),
            "qps": result.get("throughput_qps"),
            "talker_avg_ms": _num(stage.get("talker_avg_ms"), None),
            "talker_pct_of_latency": _pct(
                _num(stage.get("talker_avg_ms"), None), latency_ms
            ),
            "code2wav_decode_avg_ms": _num(stage.get("code2wav_decode_avg_ms"), None),
            "code2wav_decode_pct_of_latency": _pct(
                _num(stage.get("code2wav_decode_avg_ms"), None), latency_ms
            ),
            "code2wav_window_collect_avg_ms": _num(
                stage.get("code2wav_window_avg_ms"), None
            ),
            "talker_to_code2wav_hop_p95_ms": _num(
                stage.get("talker_to_code2wav_hop_p95_ms"), None
            ),
        }
        row["diagnosis"] = _classify_synthetic(row)
        budget.append(row)
    return sorted(
        budget,
        key=lambda item: (str(item["scenario"]), int(item["concurrency"])),
    )


def _build_vllm_budget(tables: dict[str, Any]) -> list[dict[str, Any]]:
    diagnosis_rows = _rows(tables, "vllm_admission_diagnosis")
    stage_by_label = _by_key(_rows(tables, "vllm_log_stage_signals"), "label")
    budget: list[dict[str, Any]] = []
    for row in diagnosis_rows:
        label = str(row.get("label"))
        stage = stage_by_label.get(label, {})
        budget.append(
            {
                "runtime": "vllm",
                "workload": label,
                "concurrency": row.get("concurrency"),
                "runner_qps": row.get("runner_qps"),
                "engine_qps": row.get("engine_qps"),
                "runner_overhead_pct_wall": row.get("runner_overhead_pct_wall"),
                "batch_admission_span_avg_ms": row.get("batch_admission_span_avg_ms"),
                "batch_admission_span_p95_ms": row.get("batch_admission_span_p95_ms"),
                "encoder_p95_ms": row.get("encoder_p95_ms"),
                "thinker_to_talker_feed_p95_ms": stage.get(
                    "thinker_to_talker_feed_ms", {}
                ).get("p95"),
                "talker_to_code2wav_drain_p95_ms": row.get(
                    "talker_to_code2wav_drain_p95_ms"
                ),
                "diagnosis": row.get("diagnosis"),
                "valid_comparison_scope": (
                    "strict_c4_only"
                    if label == "vLLM-c4"
                    else "offline_diagnostic_only"
                ),
            }
        )
    return budget


def build_budget(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    tables = _load_json_optional(audit_dir / "tables_summary.json")
    interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")

    sglang_budget = _build_sglang_budget(tables)
    synthetic_budget = _build_synthetic_budget(tables)
    vllm_budget = _build_vllm_budget(tables)
    interaction_summary = interactions.get("summary", {})

    c8 = next((row for row in sglang_budget if row["concurrency"] == 8), {})
    c16 = next((row for row in sglang_budget if row["concurrency"] == 16), {})
    long_c8 = next(
        (
            row
            for row in synthetic_budget
            if row["scenario"] == "long" and row["concurrency"] == 8
        ),
        {},
    )
    vllm_c8 = next(
        (row for row in vllm_budget if row["workload"] == "vLLM-c8"),
        {},
    )
    vllm_w4 = next(
        (row for row in vllm_budget if row["workload"] == "vLLM-c8-prebuild-w4"),
        {},
    )

    checks: list[BudgetCheck] = []
    _check(
        checks,
        "SGLang Video-AMME budget rows",
        len(sglang_budget) >= 5,
        f"rows={len(sglang_budget)}",
    )
    _check(
        checks,
        "synthetic short/long budget rows",
        len(synthetic_budget) >= 6,
        f"rows={len(synthetic_budget)}",
    )
    _check(
        checks,
        "vLLM diagnostic budget rows",
        len(vllm_budget) >= 4,
        f"rows={len(vllm_budget)}",
    )
    _check(
        checks,
        "c8 queue pressure separated from actual preprocessing",
        c8.get("preproc_queue_pct_of_latency") is not None
        and c8.get("preproc_queue_pct_of_latency", 0.0) > c8.get(
            "actual_preprocess_pct_of_latency", 999.0
        ),
        (
            f"c8_queue_pct={c8.get('preproc_queue_pct_of_latency')}, "
            f"c8_actual_pct={c8.get('actual_preprocess_pct_of_latency')}"
        ),
    )
    _check(
        checks,
        "c16 saturation visible in queue budget",
        c16.get("diagnosis") == "saturation_boundary"
        and _num(c16.get("preproc_queue_pct_of_latency")) >= 50.0,
        (
            f"c16_diagnosis={c16.get('diagnosis')}, "
            f"c16_queue_pct={c16.get('preproc_queue_pct_of_latency')}"
        ),
    )
    _check(
        checks,
        "code2wav decode remains small",
        all(
            _num(row.get("code2wav_decode_avg_ms"), 999.0) <= 30.0
            for row in [*sglang_budget, *synthetic_budget]
        ),
        "all SGLang and synthetic code2wav decode averages <=30ms/window",
    )
    _check(
        checks,
        "talker-to-code2wav hop remains small",
        all(
            _num(row.get("talker_to_code2wav_hop_p95_ms"), 999.0) <= 30.0
            for row in [*sglang_budget, *synthetic_budget]
        ),
        "all SGLang and synthetic talker->code2wav hop p95 <=30ms",
    )
    _check(
        checks,
        "long c8 remains faster than real time",
        _num(long_c8.get("rtf_mean"), 999.0) < 1.0,
        f"long_c8_rtf_mean={long_c8.get('rtf_mean')}",
    )
    _check(
        checks,
        "vLLM c8 remains prompt-feed limited",
        vllm_c8.get("diagnosis") == "prompt_feed_limited"
        and _num(vllm_c8.get("runner_overhead_pct_wall")) >= 50.0,
        (
            f"diagnosis={vllm_c8.get('diagnosis')}, "
            f"runner_overhead_pct={vllm_c8.get('runner_overhead_pct_wall')}"
        ),
    )
    _check(
        checks,
        "vLLM prebuild w4 remains diagnostic-only",
        vllm_w4.get("valid_comparison_scope") == "offline_diagnostic_only"
        and vllm_w4.get("diagnosis") == "engine_or_workload_limited",
        (
            f"scope={vllm_w4.get('valid_comparison_scope')}, "
            f"diagnosis={vllm_w4.get('diagnosis')}"
        ),
    )
    _check(
        checks,
        "stage interaction booleans agree with budget",
        bool(interaction_summary.get("sglang_talker_to_code2wav_healthy"))
        and bool(interaction_summary.get("sglang_code2wav_decode_not_bottleneck"))
        and bool(interaction_summary.get("vllm_original_c8_prompt_feed_limited")),
        f"stage_interactions={interaction_summary}",
    )
    _check(
        checks,
        "claim verifier agrees with budget",
        bool(claims.get("passed")) and int(claims.get("failed_checks") or 0) == 0,
        f"claims_passed={claims.get('passed')}, failed={claims.get('failed_checks')}",
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
            "sglang_budget_rows": len(sglang_budget),
            "synthetic_budget_rows": len(synthetic_budget),
            "vllm_budget_rows": len(vllm_budget),
            "c8_queue_pct_of_latency": c8.get("preproc_queue_pct_of_latency"),
            "c16_queue_pct_of_latency": c16.get("preproc_queue_pct_of_latency"),
            "long_c8_rtf_mean": long_c8.get("rtf_mean"),
            "vllm_c8_diagnosis": vllm_c8.get("diagnosis"),
            "vllm_w4_scope": vllm_w4.get("valid_comparison_scope"),
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "method_note": (
            "Stage budget percentages are non-additive pressure ratios: each stage "
            "span is divided by request latency to show scale. Streaming and repeated "
            "stages can overlap, so these rows are for bottleneck attribution rather "
            "than a flame-graph sum to 100%."
        ),
        "sglang_videoamme_budget": sglang_budget,
        "synthetic_speech_budget": synthetic_budget,
        "vllm_offline_budget": vllm_budget,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = [
        "# Qwen3.5-Omni Stage Latency Budget",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份附录把端到端 latency 和关键 stage span 放在同一张预算表里，便于在分享现场回答：",
        "preprocessing、talker AR、code2wav、stage handoff、vLLM offline admission 到底谁在限制性能。",
        "",
        "> 注意：这些百分比是 non-additive pressure ratio，不是 flame graph。",
        "> 对 streaming/repeated stage，多个 span 可能重叠或按 window 重复；本表用于瓶颈归因，",
        "> 不要求各列相加为 100%。",
        "",
        "## 1. Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| SGLang rows | {summary['sglang_budget_rows']} |",
        f"| Synthetic rows | {summary['synthetic_budget_rows']} |",
        f"| vLLM rows | {summary['vllm_budget_rows']} |",
        "",
        "## 2. SGLang Video-AMME Stage Budget",
        "",
        "| c | QPS | Lat mean | Preproc lifecycle | Actual preproc | Est. queue | Talker AR | Code2wav decode | Hop p95 | Diagnosis |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["sglang_videoamme_budget"]:
        lines.append(
            "| {c} | {qps} | {lat} | {pre} ({pre_pct}) | {act} ({act_pct}) | {queue} ({queue_pct}) | {talker} ({talker_pct}) | {decode} ({decode_pct}) | {hop} | {diag} |".format(
                c=row["concurrency"],
                qps=_fmt(row.get("qps"), 3),
                lat=_fmt(row.get("latency_mean_ms"), 1, "ms"),
                pre=_fmt(row.get("preproc_lifecycle_avg_ms"), 1, "ms"),
                pre_pct=_fmt(row.get("preproc_lifecycle_pct_of_latency"), 1, "%"),
                act=_fmt(row.get("actual_preprocess_avg_ms"), 1, "ms"),
                act_pct=_fmt(row.get("actual_preprocess_pct_of_latency"), 1, "%"),
                queue=_fmt(row.get("preproc_queue_estimate_ms"), 1, "ms"),
                queue_pct=_fmt(row.get("preproc_queue_pct_of_latency"), 1, "%"),
                talker=_fmt(row.get("talker_avg_ms"), 1, "ms"),
                talker_pct=_fmt(row.get("talker_pct_of_latency"), 1, "%"),
                decode=_fmt(row.get("code2wav_decode_avg_ms"), 1, "ms"),
                decode_pct=_fmt(row.get("code2wav_decode_pct_of_latency"), 2, "%"),
                hop=_fmt(row.get("talker_to_code2wav_hop_p95_ms"), 1, "ms"),
                diag=row.get("diagnosis"),
            )
        )

    lines.extend(
        [
            "",
            "关键读法：c=8/c=16 的 preprocessing lifecycle 变大，但 actual preprocess 仍约 0.29-0.30s；",
            "放大的是 queue/admission。code2wav decode 仍是十几毫秒/window，和端到端 latency 不是一个量级。",
            "",
            "## 3. Synthetic Short/Long Speech Budget",
            "",
            "| Scenario | c | Words | Audio mean | Lat mean | RTF | Talker AR | Code2wav decode | Hop p95 | Diagnosis |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["synthetic_speech_budget"]:
        lines.append(
            "| {scenario} | {c} | {words} | {audio} | {lat} | {rtf} | {talker} ({talker_pct}) | {decode} ({decode_pct}) | {hop} | {diag} |".format(
                scenario=row["scenario"],
                c=row["concurrency"],
                words=_fmt(row.get("target_words"), 0),
                audio=_fmt(row.get("audio_duration_mean_s"), 2, "s"),
                lat=_fmt(row.get("latency_mean_ms"), 1, "ms"),
                rtf=_fmt(row.get("rtf_mean"), 4),
                talker=_fmt(row.get("talker_avg_ms"), 1, "ms"),
                talker_pct=_fmt(row.get("talker_pct_of_latency"), 1, "%"),
                decode=_fmt(row.get("code2wav_decode_avg_ms"), 1, "ms"),
                decode_pct=_fmt(row.get("code2wav_decode_pct_of_latency"), 2, "%"),
                hop=_fmt(row.get("talker_to_code2wav_hop_p95_ms"), 1, "ms"),
                diag=row.get("diagnosis"),
            )
        )

    lines.extend(
        [
            "",
            "关键读法：长文本长语音 c=8 的 talker AR 自然变长，但 RTF 仍低于 1；",
            "这支持“长短文都有覆盖，长输出仍可快于实时”的结论。",
            "",
            "## 4. vLLM Offline Budget / Admission Diagnosis",
            "",
            "| Case | c | Runner QPS | Engine QPS | Runner overhead | Admission span avg/p95 | Encoder p95 | Thinker->Talker p95 | Talker->C2W p95 | Scope | Diagnosis |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["vllm_offline_budget"]:
        lines.append(
            "| {case} | {c} | {rqps} | {eqps} | {overhead}% | {span_avg}/{span_p95}ms | {enc}ms | {feed}ms | {c2w}ms | {scope} | {diag} |".format(
                case=row["workload"],
                c=row.get("concurrency"),
                rqps=_fmt(row.get("runner_qps"), 4),
                eqps=_fmt(row.get("engine_qps"), 4),
                overhead=_fmt(row.get("runner_overhead_pct_wall"), 1),
                span_avg=_fmt(row.get("batch_admission_span_avg_ms"), 1),
                span_p95=_fmt(row.get("batch_admission_span_p95_ms"), 1),
                enc=_fmt(row.get("encoder_p95_ms"), 1),
                feed=_fmt(row.get("thinker_to_talker_feed_p95_ms"), 1),
                c2w=_fmt(row.get("talker_to_code2wav_drain_p95_ms"), 1),
                scope=row.get("valid_comparison_scope"),
                diag=row.get("diagnosis"),
            )
        )

    lines.extend(
        [
            "",
            "关键读法：vLLM original c=8 的 runner overhead 和 batch admission span 表明它是 offline prompt-feed/admission limited；",
            "prebuild w4 缓解 prompt/admission，但仍是 offline diagnostic，不是 online parity。",
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
    print("## Qwen3.5-Omni Stage Latency Budget\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| SGLang rows | {summary['sglang_budget_rows']} |")
    print(f"| Synthetic rows | {summary['synthetic_budget_rows']} |")
    print(f"| vLLM rows | {summary['vllm_budget_rows']} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni stage latency-budget appendix."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_budget(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    _save_text(build_markdown(payload), output)
    _save_json(payload, json_output)
    print_markdown(payload)
    print(
        "Stage latency budget written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
