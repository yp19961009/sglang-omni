# SPDX-License-Identifier: Apache-2.0
"""Build the standalone optimization-candidate ledger for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_INPUT = AUDIT_DIR / "objective_requirement_crosswalk.json"
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "optimization_candidate_ledger.json"

CURRENT_BEST_ID = "sglang_current_best_measured_recipe"


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


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _md(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _metric_briefs(row: dict[str, Any], *, limit: int = 4) -> str:
    parts: list[str] = []
    for metric in _as_rows(row.get("metric_rows"))[:limit]:
        metric_id = str(metric.get("id") or "")
        metric_value = metric.get("metric_value")
        metric_text = metric.get("metric_text")
        unit = str(metric.get("metric_unit") or "")
        if metric_text not in (None, ""):
            value_text = str(metric_text)
        elif metric_value not in (None, ""):
            value_text = f"{metric_value} {unit}".strip()
        else:
            value_text = "present"
        parts.append(f"{metric_id}={value_text}")
    extra = len(_as_rows(row.get("metric_rows"))) - len(parts)
    if extra > 0:
        parts.append(f"+{extra} metrics")
    return "; ".join(parts)


def _evidence_brief(row: dict[str, Any], *, limit: int = 3) -> str:
    evidence = [str(item) for item in row.get("machine_evidence", []) if item]
    parts = evidence[:limit]
    if len(evidence) > limit:
        parts.append(f"+{len(evidence) - limit} files")
    return "; ".join(parts)


def _commands_brief(row: dict[str, Any], *, limit: int = 4) -> str:
    commands = [str(item) for item in row.get("rerun_command_ids", []) if item]
    parts = commands[:limit]
    if len(commands) > limit:
        parts.append(f"+{len(commands) - limit} commands")
    return "; ".join(parts)


def _collect_missing(rows: list[dict[str, Any]], key: str) -> list[str]:
    missing: set[str] = set()
    for row in rows:
        for item in row.get(key, []) or []:
            if item:
                missing.add(str(item))
    return sorted(missing)


def build_payload(root: Path, input_path: Path) -> dict[str, Any]:
    root = root.resolve()
    crosswalk = _load_json_optional(input_path)
    ledger = crosswalk.get("optimization_candidate_ledger", {})
    if not isinstance(ledger, dict):
        ledger = {}

    rows = _as_rows(ledger.get("rows"))
    summary = ledger.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    crosswalk_summary = crosswalk.get("summary", {})
    crosswalk_summary = crosswalk_summary if isinstance(crosswalk_summary, dict) else {}

    current_best_rows = [
        row
        for row in rows
        if row.get("candidate_id") == CURRENT_BEST_ID
        and row.get("decision") == "accept_current_recipe"
    ]
    anti_recipe_rows = [
        row for row in rows if row.get("decision_class") == "rejected_anti_recipe"
    ]
    vllm_rows = [row for row in rows if row.get("runtime") == "vllm"]
    vllm_diagnostic_rows = [
        row
        for row in rows
        if row.get("decision_class") in {"diagnostic_only", "optimized_offline_diagnostic"}
    ]

    missing_metric_row_ids = _collect_missing(rows, "missing_metric_row_ids")
    missing_evidence_files = _collect_missing(rows, "missing_evidence_files")
    missing_command_ids = _collect_missing(rows, "missing_command_ids")
    evidence_paths = sorted(
        {
            str(item)
            for row in rows
            for item in row.get("machine_evidence", []) or []
            if item
        }
    )
    missing_existing_evidence = [
        evidence for evidence in evidence_paths if not (root / evidence).exists()
    ]

    checks = {
        "source_crosswalk_contract_available": _int_value(
            crosswalk_summary.get("requirement_rows_total")
        )
        >= 11
        and _int_value(crosswalk_summary.get("optimization_candidate_rows_total")) >= 8
        and _int_value(crosswalk_summary.get("required_failures"), default=99) <= 1,
        "candidate_rows_total": len(rows) >= 8
        and _int_value(summary.get("candidate_rows_total")) >= 8,
        "current_best_recipe_locked": len(current_best_rows) == 1
        and bool(current_best_rows[0].get("ready")),
        "anti_recipes_locked": len(anti_recipe_rows) >= 2
        and all(bool(row.get("ready")) for row in anti_recipe_rows),
        "vllm_baseline_and_diagnostic_locked": len(vllm_rows) >= 3
        and len(vllm_diagnostic_rows) >= 2
        and any(row.get("decision_class") == "accepted_strict_baseline" for row in vllm_rows),
        "metric_rows_present": not missing_metric_row_ids,
        "evidence_files_present": not missing_evidence_files
        and not missing_existing_evidence,
        "rerun_commands_present": not missing_command_ids,
        "not_global_optimum_boundary_documented": bool(
            summary.get("not_global_optimum_boundary")
        ),
        "final_completion_gate_contract_available": not bool(
            crosswalk_summary.get("goal_complete")
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]

    review_rows = [
        {
            "candidate_id": row.get("candidate_id"),
            "runtime": row.get("runtime"),
            "scope": row.get("scope"),
            "decision_class": row.get("decision_class"),
            "decision": row.get("decision"),
            "ready": bool(row.get("ready")),
            "metric_rows_total": len(_as_rows(row.get("metric_rows"))),
            "machine_evidence_total": len(row.get("machine_evidence", []) or []),
            "rerun_command_ids_total": len(row.get("rerun_command_ids", []) or []),
            "metric_brief": _metric_briefs(row),
            "evidence_brief": _evidence_brief(row),
            "commands_brief": _commands_brief(row),
            "rationale": row.get("rationale"),
            "replacement_rule": row.get("replacement_rule"),
        }
        for row in rows
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "source": str(input_path),
        "summary": {
            "ready": not required_failures,
            "candidate_rows_total": len(rows),
            "current_best_candidate_id": (
                current_best_rows[0].get("candidate_id") if current_best_rows else ""
            ),
            "accepted_current_best_rows_total": len(current_best_rows),
            "rejected_anti_recipe_rows_total": len(anti_recipe_rows),
            "vllm_rows_total": len(vllm_rows),
            "vllm_diagnostic_rows_total": len(vllm_diagnostic_rows),
            "not_global_optimum_boundary": bool(
                summary.get("not_global_optimum_boundary")
            ),
            "source_crosswalk_ready": bool(crosswalk_summary.get("ready")),
            "missing_metric_row_ids": len(missing_metric_row_ids),
            "missing_evidence_files": len(missing_evidence_files)
            + len(missing_existing_evidence),
            "missing_command_ids": len(missing_command_ids),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "Standalone reviewer-facing optimization verdict ledger for the "
                "current measured-best SGLang recipe, anti-recipes, strong vLLM "
                "baseline, vLLM c=8 diagnostic scope, and stage-first tuning priority."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "required_failures": required_failures,
            "missing_metric_row_ids": missing_metric_row_ids,
            "missing_evidence_files": sorted(
                set(missing_evidence_files + missing_existing_evidence)
            ),
            "missing_command_ids": missing_command_ids,
            "source_crosswalk_summary": crosswalk_summary,
            "source_ledger_summary": summary,
        },
        "review_rows": review_rows,
        "rows": rows,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    diagnostics = payload["diagnostics"]
    lines = [
        "# Qwen3.5-Omni 优化候选 Ledger",
        "",
        "状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。",
        "",
        "## 1. 裁决摘要",
        "",
        f"- ready: `{summary['ready']}`",
        f"- candidates: `{summary['candidate_rows_total']}`",
        f"- current measured best: `{summary['current_best_candidate_id']}`",
        (
            "- anti-recipes / vLLM diagnostics: "
            f"`{summary['rejected_anti_recipe_rows_total']}` / "
            f"`{summary['vllm_diagnostic_rows_total']}`"
        ),
        f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`",
        f"- required_failures: `{summary['required_failures']}`",
        "",
        "当前结论是 measured-best，不是对未来 kernel、placement、admission policy "
        "或 online vLLM ingress 的全局数学最优声明。替换当前 best recipe 前，必须同时重跑 "
        "strict c=4 SGLang/vLLM、SGLang c=1/2/4/8/16、quality/WER、stage causal graph、"
        "acceptance matrix、rerun acceptance contract 和 final readiness。",
        "",
        "## 2. Machine Gate",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for name, ok in payload["checks"].items():
        lines.append(f"| `{_md(name)}` | `{'PASS' if ok else 'FAIL'}` |")

    lines.extend(
        [
            "",
            "## 3. 候选裁决总表",
            "",
            "| Candidate | Runtime | Decision class | Decision | Ready | Key metrics | Evidence | Commands |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["review_rows"]:
        lines.append(
            "| "
            f"`{_md(row['candidate_id'])}` | "
            f"{_md(row['runtime'])} | "
            f"`{_md(row['decision_class'])}` | "
            f"`{_md(row['decision'])}` | "
            f"`{row['ready']}` | "
            f"{_md(row['metric_brief'])} | "
            f"{_md(row['evidence_brief'])} | "
            f"{_md(row['commands_brief'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. 解释口径",
            "",
            "- `sglang_current_best_measured_recipe` 是当前可分享的 SGLang measured-best：strict warmed c=4 优于优化版 vLLM baseline，c=8 是当前 SGLang serving peak，c=16 保留为 saturation boundary。",
            "- `sglang_preproc2_widening` 和 `sglang_preproc4_widening` 是反例，不应作为当前 recipe 的优化方向；前者吞吐/延迟退化，后者触发失败样本和质量风险。",
            "- `vllm_optimized_c4_baseline` 是严格 headline 对比中的强基线；vLLM c=8 original/prebuild w4 只用于 offline diagnostic，不能写成 online serving parity。",
            "- `code2wav_first_tuning` 当前不优先，因为 stage graph 显示 handoff 健康、code2wav decode 不是当前主计算瓶颈；应先看 admission/Talker cadence。",
            "",
            "## 5. 复核和刷新入口",
            "",
            "```bash",
            "cd /home/gangouyu/sglang-omni",
            "python3 -m benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk \\",
            "  --root /home/gangouyu/sglang-omni \\",
            "  --strict \\",
            "  --json-output results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
            "python3 -m benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger \\",
            "  --root /home/gangouyu/sglang-omni \\",
            "  --strict \\",
            "  --output benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md \\",
            "  --json-output results/qwen35_report_audit_20260619/optimization_candidate_ledger.json",
            "```",
            "",
            "机器证据：",
            "",
            "- `results/qwen35_report_audit_20260619/optimization_candidate_ledger.json`",
            "- `results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`",
            "- `results/qwen35_report_audit_20260619/repro_command_manifest.json`",
            "- `results/qwen35_report_audit_20260619/sglang_optimization_lock.json`",
            "- `results/qwen35_report_audit_20260619/vllm_optimization_lock.json`",
            "- `results/qwen35_report_audit_20260619/runtime_comparison_contract.json`",
            "",
        ]
    )
    if diagnostics["required_failures"]:
        lines.extend(
            [
                "## 6. Required Failures",
                "",
                f"- `{diagnostics['required_failures']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni optimization-candidate ledger."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )

    payload = build_payload(root, input_path)
    _save_text(build_markdown(payload), output)
    _save_json(payload, json_output)
    print(
        f"Optimization candidate ledger written: {output} "
        f"ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    print(f"Optimization candidate ledger JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
