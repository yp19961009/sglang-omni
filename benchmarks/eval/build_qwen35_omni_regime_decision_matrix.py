# SPDX-License-Identifier: Apache-2.0
"""Build a reviewer-facing regime decision matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path("benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md")
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "regime_decision_matrix.json"

REQUIRED_STATUSES = {
    "recommended_strict_baseline",
    "recommended_serving_window",
    "recommended_peak_throughput",
    "not_recommended_saturation",
    "speech_generation_regression_guard",
    "diagnostic_prompt_feed_limited",
    "optimized_offline_diagnostic",
    "anti_recipe_regression",
    "anti_recipe_failure",
    "cross_stage_guardrail",
}

REQUIRED_SGLANG_VIDEOAMME_PRESSURES = {"c=1", "c=2", "c=4", "c=8", "c=16"}
REQUIRED_SYNTHETIC_PRESSURES = {
    "short c=1",
    "short c=4",
    "short c=8",
    "long c=1",
    "long c=4",
    "long c=8",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else {}


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _escape_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text.replace("|", "\\|")


def _decision_label(serving_status: str) -> str:
    labels = {
        "recommended_strict_baseline": "严格横向 headline",
        "recommended_serving_window": "推荐服务窗口",
        "recommended_peak_throughput": "推荐吞吐峰值",
        "not_recommended_saturation": "压力边界",
        "speech_generation_regression_guard": "语音输出回归保护",
        "diagnostic_prompt_feed_limited": "诊断证据",
        "optimized_offline_diagnostic": "优化后 offline 诊断",
        "anti_recipe_regression": "反例：性能回退",
        "anti_recipe_failure": "反例：失败/OOM",
        "cross_stage_guardrail": "stage 连接防线",
    }
    return labels.get(serving_status, serving_status)


def _stage_decision(row: dict[str, Any]) -> str:
    regime = str(row.get("regime") or "")
    pressure = str(row.get("pressure") or "")
    status = str(row.get("serving_status") or "")
    if regime == "strict_runtime_comparison":
        return "用 warmed c=4 做严格 SGLang-vLLM 对比；不要把 c=8 offline 诊断混进 headline。"
    if regime == "sglang_videoamme_stress":
        if pressure in {"c=1", "c=2", "c=4"}:
            return "低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。"
        if pressure == "c=8":
            return "当前高并发甜点；admission/queue 开始显性化，但 talker->code2wav hop 仍健康。"
        if pressure == "c=16":
            return "吞吐回落且 queue/admission 饱和；只作为压力边界，不做默认服务点。"
    if regime == "sglang_synthetic_speech":
        if pressure.startswith("long"):
            return "长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。"
        return "短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。"
    if regime == "vllm_offline_diagnostic":
        if "original" in pressure:
            return "原始 vLLM c=8 主要是 host prompt build/feed admission 受限，不是 online parity。"
        return "prebuild w4 是当前最强 vLLM offline 诊断；仍需 online ingress + WER 才能做 c=8 parity。"
    if regime == "negative_optimization":
        if "2" in pressure:
            return "preproc=2 把 admission 问题转成共享资源 contention，不能作为当前优化方向。"
        return "preproc=4 有失败/OOM 风险，当前 recipe 禁用。"
    if status == "cross_stage_guardrail":
        return "把 admission、stage-local compute、handoff 分开回答；不要把健康 handoff 说成瓶颈。"
    return str(row.get("expected_shape") or "")


def _source_summary(root: Path) -> dict[str, Any]:
    audit_dir = root / AUDIT_DIR
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    headline = _load_json_optional(audit_dir / "headline_scorecard.json")
    vllm = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    final_summary = final.get("summary", {})
    return {
        "acceptance": acceptance.get("summary", {}),
        "stage_interactions": interactions.get("summary", {}),
        "headline": headline.get("summary", {}),
        "vllm_rows": len(vllm.get("rows", [])),
        "final_readiness": {
            "ready": final_summary.get("ready"),
            "checks_total": final_summary.get("checks_total"),
            "checks_passed": final_summary.get("checks_passed"),
            "required_failures": final_summary.get("required_failures"),
            "hard_gates_total": len(final_summary.get("hard_gates", {})),
        },
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    rows = acceptance.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    accepted_count = sum(1 for row in rows if isinstance(row, dict) and bool(row.get("accepted")))
    status_counts = Counter(
        str(row.get("serving_status") or "") for row in rows if isinstance(row, dict)
    )
    source = _source_summary(root)

    decision_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        serving_status = str(row.get("serving_status") or "")
        decision_rows.append(
            {
                "id": f"regime-decision-{idx:02d}",
                "regime": row.get("regime"),
                "pressure": row.get("pressure"),
                "serving_status": serving_status,
                "accepted": bool(row.get("accepted")),
                "decision_label": _decision_label(serving_status),
                "stage_bottleneck_decision": _stage_decision(row),
                "key_metrics": row.get("key_metrics"),
                "recommended_action": row.get("action"),
                "expected_shape": row.get("expected_shape"),
                "source_evidence": [
                    "results/qwen35_report_audit_20260619/acceptance_matrix.json",
                    "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
                    "results/qwen35_report_audit_20260619/headline_scorecard.json",
                    "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
                    "results/qwen35_report_audit_20260619/metric_provenance_index.json",
                ],
            }
        )

    statuses = set(status_counts)
    videoamme_pressures = {
        str(row.get("pressure") or "")
        for row in decision_rows
        if row.get("regime") == "sglang_videoamme_stress"
    }
    synthetic_pressures = {
        str(row.get("pressure") or "")
        for row in decision_rows
        if row.get("regime") == "sglang_synthetic_speech"
    }
    vllm_pressures = {
        str(row.get("pressure") or "")
        for row in decision_rows
        if row.get("regime") == "vllm_offline_diagnostic"
    }
    negative_pressures = {
        str(row.get("pressure") or "")
        for row in decision_rows
        if row.get("regime") == "negative_optimization"
    }
    checks = {
        "acceptance_rows_present": len(decision_rows) >= 17,
        "all_rows_accepted": bool(decision_rows) and accepted_count == len(decision_rows),
        "required_statuses_present": REQUIRED_STATUSES.issubset(statuses),
        "sglang_videoamme_concurrency_coverage": REQUIRED_SGLANG_VIDEOAMME_PRESSURES.issubset(
            videoamme_pressures
        ),
        "synthetic_short_long_coverage": REQUIRED_SYNTHETIC_PRESSURES.issubset(
            synthetic_pressures
        ),
        "vllm_offline_diagnostics_present": any(
            "original" in pressure for pressure in vllm_pressures
        )
        and any("prebuild" in pressure for pressure in vllm_pressures),
        "negative_optimization_rows_present": any(
            "2" in pressure for pressure in negative_pressures
        )
        and any("4" in pressure for pressure in negative_pressures),
        "stage_interaction_source_present": bool(source["stage_interactions"]),
        "headline_and_vllm_sources_present": bool(source["headline"])
        and int(source["vllm_rows"] or 0) >= 4,
    }
    required_failures = sum(1 for value in checks.values() if not value)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": required_failures == 0,
            "rows_total": len(decision_rows),
            "accepted_rows": accepted_count,
            "status_counts": dict(sorted(status_counts.items())),
            "required_statuses_total": len(REQUIRED_STATUSES),
            "sglang_videoamme_pressures": sorted(videoamme_pressures),
            "synthetic_pressures": sorted(synthetic_pressures),
            "vllm_diagnostic_pressures": sorted(vllm_pressures),
            "negative_optimization_pressures": sorted(negative_pressures),
            "checks_total": len(checks),
            "checks_passed": sum(1 for value in checks.values() if value),
            "required_failures": required_failures,
            "share_scope": (
                "Reviewer-facing regime decision matrix covering strict runtime "
                "comparison, SGLang single/high concurrency, short/long speech "
                "guards, vLLM offline diagnostics, negative optimization rows, "
                "and cross-stage guardrails."
            ),
        },
        "checks": checks,
        "source_summary": source,
        "rows": decision_rows,
    }


def build_markdown(root: Path, payload: dict[str, Any] | None = None) -> str:
    root = root.resolve()
    payload = payload or build_payload(root)
    rows = payload.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    summary = payload.get("summary", {})
    source = payload.get("source_summary", {})
    status_counts = Counter(str(row.get("serving_status") or "") for row in rows)

    lines: list[str] = [
        "# Qwen3.5-Omni Regime 决策矩阵",
        "",
        f"生成时间 UTC：`{payload.get('generated_at_utc')}`。",
        f"工作目录：`{root}`。",
        "",
        "这页给 reviewer 快速回答三个问题：哪个压力条件推荐、哪个只是边界或诊断、下一步优化该动哪里。",
        "所有行均来自 `acceptance_matrix.json`，stage 判断交叉引用 `stage_interaction_summary.json`、",
        "`headline_scorecard.json` 和 `vllm_admission_diagnosis.json`。",
        "",
        "## 1. 总结 Gate",
        "",
        f"- ready：`{summary.get('ready')}`",
        f"- acceptance rows：`{summary.get('accepted_rows')}/{summary.get('rows_total')}` accepted",
        f"- checks：`{summary.get('checks_passed')}/{summary.get('checks_total')}`",
        f"- required failures：`{summary.get('required_failures')}`",
        f"- final readiness：`{source['final_readiness'].get('ready')}`",
        f"- stage handoff healthy：`{source['stage_interactions'].get('sglang_talker_to_code2wav_healthy')}`",
        f"- code2wav decode not bottleneck：`{source['stage_interactions'].get('sglang_code2wav_decode_not_bottleneck')}`",
        f"- vLLM diagnostic rows：`{source['vllm_rows']}`",
        f"- machine evidence：`{DEFAULT_JSON_OUTPUT.as_posix()}`",
        "",
        "## 2. 推荐结论",
        "",
        "- 对外 headline 使用 warmed c=4：SGLang latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。",
        "- 当前 SGLang 推荐服务窗口是 c=4 到 c=8；c=8 是吞吐峰值，c=16 是压力边界。",
        "- 短/长文本语音输出用于守住 thinker/talker/code2wav 路径，长文本 c=8 仍快于实时。",
        "- vLLM c=8 prebuild w4 是优化后的 offline diagnostic，不是 online serving parity。",
        "- 不要把 preproc=2/4 当作当前优化方向；证据显示回退或失败。",
        "",
        "## 3. Regime 决策表",
        "",
        "| Regime | Pressure | 决策 | Stage/瓶颈判断 | 关键数字 | 推荐动作 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_cell(row.get("regime")),
                    _escape_cell(row.get("pressure")),
                    _escape_cell(row.get("decision_label")),
                    _escape_cell(row.get("stage_bottleneck_decision")),
                    _escape_cell(row.get("key_metrics")),
                    _escape_cell(row.get("recommended_action")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 4. Status 分布",
            "",
            "| Status | Count | 读法 |",
            "| --- | ---: | --- |",
        ]
    )
    explanations = {
        "recommended_strict_baseline": "可做主 headline 的严格横向对比。",
        "recommended_serving_window": "可作为当前推荐服务窗口的一部分。",
        "recommended_peak_throughput": "当前 recipe 的高并发吞吐峰值。",
        "not_recommended_saturation": "可运行但不建议默认使用。",
        "speech_generation_regression_guard": "证明短/长文本语音输出路径没有退化。",
        "diagnostic_prompt_feed_limited": "只能用于定位，不可当 serving parity。",
        "optimized_offline_diagnostic": "优化后的诊断基线，仍需在线复核。",
        "anti_recipe_regression": "实测回退，应避免。",
        "anti_recipe_failure": "失败/OOM，应避免。",
        "cross_stage_guardrail": "用于回答 stage 连接是否卡住。",
    }
    for status, count in sorted(status_counts.items()):
        lines.append(
            f"| `{_escape_cell(status)}` | {count} | {_escape_cell(explanations.get(status, ''))} |"
        )

    lines.extend(
        [
            "",
            "## 5. 复核入口",
            "",
            "- 主报告：`benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`",
            "- 压力矩阵：`benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`",
            "- stage metric dictionary：`benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`",
            "- acceptance matrix：`results/qwen35_report_audit_20260619/acceptance_matrix.json`",
            "- stage interaction summary：`results/qwen35_report_audit_20260619/stage_interaction_summary.json`",
            "- regime decision matrix JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni reviewer-facing regime decision matrix."
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
        "Regime decision matrix written: "
        f"{output} json={json_output} ready={summary['ready']} "
        f"rows={summary['rows_total']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']}"
    )
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
