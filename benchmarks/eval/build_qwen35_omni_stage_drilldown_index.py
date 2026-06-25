# SPDX-License-Identifier: Apache-2.0
"""Build a stage-first drilldown index for Qwen3.5-Omni performance evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "stage_drilldown_index.json"


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


def _list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


def _command_ids(repro_manifest: dict[str, Any]) -> set[str]:
    commands = repro_manifest.get("commands", [])
    return {
        str(command.get("id"))
        for command in commands
        if isinstance(command, dict) and command.get("id")
    }


def _boundary_stage_route(boundary: str) -> list[str]:
    routes = {
        "request_admission_to_preprocessing": ["admission", "preprocessing"],
        "preprocessing_to_encoder_thinker": ["preprocessing", "encoder_thinker"],
        "thinker_to_talker": ["thinker", "talker"],
        "talker_to_code2wav_stream": ["talker", "code2wav_stream"],
        "talker_to_code2wav": ["talker", "code2wav"],
        "code2wav_collect_to_decode": ["code2wav_collect", "code2wav_decode"],
        "runner_to_engine_admission": ["offline_runner", "engine_admission"],
    }
    return routes.get(boundary, [part for part in boundary.split("_to_") if part])


def _case_commands(row: dict[str, Any]) -> list[str]:
    runtime = str(row.get("runtime") or "")
    case = str(row.get("case") or row.get("workload") or "")
    workload = str(row.get("workload") or "")
    if runtime == "vllm":
        commands = [
            "summarize_vllm_log_stages",
            "diagnose_vllm_admission",
            "build_stage_interactions",
            "build_stage_boundary_bottleneck_ledger",
        ]
        if "prebuild-w4" in case:
            commands.insert(0, "vllm_c8_prebuild_w4")
        elif "c8" in case:
            commands.insert(0, "vllm_c8_original")
        else:
            commands.insert(0, "vllm_c1_original")
        return commands
    if "synthetic" in case or "synthetic" in workload:
        return [
            "sglang_synthetic_text_to_speech",
            "build_stage_interactions",
            "build_stage_boundary_bottleneck_ledger",
        ]
    return [
        "sglang_videoamme_stress",
        "build_stage_interactions",
        "build_stage_boundary_bottleneck_ledger",
    ]


def _budget_commands(row: dict[str, Any], budget_type: str) -> list[str]:
    if budget_type == "vllm_offline_budget":
        workload = str(row.get("workload") or "")
        first = "vllm_c8_prebuild_w4" if "prebuild-w4" in workload else (
            "vllm_c8_original" if "c8" in workload else "vllm_c1_original"
        )
        return [
            first,
            "summarize_vllm_log_stages",
            "diagnose_vllm_admission",
            "build_stage_latency_budget",
        ]
    if budget_type == "synthetic_speech_budget":
        return [
            "sglang_synthetic_text_to_speech",
            "build_report_tables",
            "build_stage_latency_budget",
        ]
    return [
        "sglang_videoamme_stress",
        "build_report_tables",
        "build_stage_latency_budget",
    ]


def _boundary_evidence_files(row: dict[str, Any]) -> list[str]:
    runtime = str(row.get("runtime") or "")
    evidence = [
        "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
        "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
        "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
    ]
    if runtime == "vllm":
        evidence.extend(
            [
                "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
                "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
            ]
        )
    return evidence


def _budget_evidence_files(budget_type: str) -> list[str]:
    evidence = [
        "results/qwen35_report_audit_20260619/stage_latency_budget.json",
        "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
    ]
    if budget_type == "vllm_offline_budget":
        evidence.extend(
            [
                "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
                "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
            ]
        )
    else:
        evidence.append("results/qwen35_report_audit_20260619/tables_summary.json")
    return evidence


def _budget_case(row: dict[str, Any]) -> str:
    workload = str(row.get("workload") or "unknown")
    concurrency = row.get("concurrency")
    if row.get("scenario"):
        workload = f"{workload}:{row.get('scenario')}"
    return f"{workload} c={concurrency}" if concurrency is not None else workload


def _stage_focus(row: dict[str, Any], budget_type: str) -> list[str]:
    if budget_type == "vllm_offline_budget":
        return ["offline_runner", "engine_admission", "encoder", "thinker", "talker", "code2wav"]
    if budget_type == "synthetic_speech_budget":
        return ["talker", "code2wav", "text_to_speech"]
    focus = ["admission", "preprocessing", "talker", "code2wav"]
    if row.get("preproc_queue_pct_of_latency") is not None:
        focus.insert(1, "preprocessing_queue")
    return focus


def _build_boundary_rows(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _list(ledger, "ledger_rows"):
        boundary = str(row.get("boundary") or "")
        rows.append(
            {
                "id": f"stage-{row.get('id')}",
                "entry_type": "stage_boundary",
                "runtime": row.get("runtime"),
                "case": row.get("case"),
                "stage_route": _boundary_stage_route(boundary),
                "status": row.get("status"),
                "bottleneck_verdict": row.get("bottleneck_verdict"),
                "diagnosis": row.get("source_interpretation"),
                "safe_conclusion": row.get("decision"),
                "metrics": row.get("source_metrics", {}),
                "evidence_files": _boundary_evidence_files(row),
                "rerun_command_ids": _case_commands(row),
            }
        )
    return rows


def _build_budget_rows(stage_budget: dict[str, Any]) -> list[dict[str, Any]]:
    budget_keys = [
        "sglang_videoamme_budget",
        "synthetic_speech_budget",
        "vllm_offline_budget",
    ]
    rows: list[dict[str, Any]] = []
    for budget_key in budget_keys:
        for index, row in enumerate(_list(stage_budget, budget_key), start=1):
            rows.append(
                {
                    "id": f"budget-{budget_key}-{index:02d}",
                    "entry_type": "latency_budget",
                    "runtime": row.get("runtime"),
                    "case": _budget_case(row),
                    "stage_focus": _stage_focus(row, budget_key),
                    "status": row.get("diagnosis"),
                    "bottleneck_verdict": row.get("diagnosis"),
                    "diagnosis": row.get("diagnosis"),
                    "safe_conclusion": _budget_safe_conclusion(row, budget_key),
                    "metrics": row,
                    "evidence_files": _budget_evidence_files(budget_key),
                    "rerun_command_ids": _budget_commands(row, budget_key),
                }
            )
    return rows


def _budget_safe_conclusion(row: dict[str, Any], budget_type: str) -> str:
    diagnosis = str(row.get("diagnosis") or "")
    if budget_type == "vllm_offline_budget":
        scope = str(row.get("valid_comparison_scope") or "")
        return f"Use as {scope}; diagnosis={diagnosis}."
    if budget_type == "synthetic_speech_budget":
        return f"Short/long TTS budget supports the stated text-speech regime; diagnosis={diagnosis}."
    return f"SGLang c={row.get('concurrency')} budget supports current serving envelope; diagnosis={diagnosis}."


def build_stage_drilldown(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_ledger = _load_json_optional(audit_dir / "stage_boundary_bottleneck_ledger.json")
    stage_interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")

    rows = _build_boundary_rows(stage_ledger) + _build_budget_rows(stage_budget)
    command_ids = _command_ids(repro_manifest)
    missing_command_ids = sorted(
        {
            command_id
            for row in rows
            for command_id in row.get("rerun_command_ids", [])
            if command_id not in command_ids
        }
    )
    missing_evidence_files = sorted(
        {
            rel_path
            for row in rows
            for rel_path in row.get("evidence_files", [])
            if not (root / rel_path).exists()
        }
    )
    stage_routes = {
        " -> ".join(row.get("stage_route", row.get("stage_focus", [])))
        for row in rows
    }
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    budget_summary = _summary(stage_budget)
    ledger_summary = _summary(stage_ledger)
    interaction_summary = _summary(stage_interactions)
    checks = {
        "stage_budget_ready": bool(budget_summary.get("ready"))
        and int(budget_summary.get("required_failures") or 0) == 0,
        "stage_boundary_ledger_ready": bool(ledger_summary.get("ready"))
        and int(ledger_summary.get("required_failures") or 0) == 0,
        "stage_interactions_ready": int(
            interaction_summary.get("total_interactions") or 0
        )
        >= 37,
        "rows_total": len(rows) >= 52,
        "boundary_rows_total": len([row for row in rows if row["entry_type"] == "stage_boundary"]) >= 37,
        "budget_rows_total": len([row for row in rows if row["entry_type"] == "latency_budget"]) >= 15,
        "stage_routes_total": len(stage_routes) >= 7,
        "evidence_files_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
        "critical_findings_preserved": (
            budget_summary.get("vllm_c8_diagnosis") == "prompt_feed_limited"
            and float(budget_summary.get("long_c8_rtf_mean") or 99.0) < 1.0
            and float(budget_summary.get("c16_queue_pct_of_latency") or 0.0) > 60.0
            and bool(interaction_summary.get("sglang_talker_to_code2wav_healthy"))
            and bool(interaction_summary.get("sglang_code2wav_decode_not_bottleneck"))
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "boundary_rows_total": len([row for row in rows if row["entry_type"] == "stage_boundary"]),
            "budget_rows_total": len([row for row in rows if row["entry_type"] == "latency_budget"]),
            "stage_routes_total": len(stage_routes),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "status_counts": status_counts,
            "recommended_sglang_window": "c4-c8",
            "saturation_boundary": "c16",
            "vllm_c8_scope": "offline_diagnostic_until_online_ingress_artifacts",
        },
        "checks": checks,
        "diagnostics": {
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "stage_routes": sorted(stage_routes),
        },
        "source_summaries": {
            "stage_latency_budget": budget_summary,
            "stage_boundary_bottleneck_ledger": ledger_summary,
            "stage_interactions": interaction_summary,
        },
        "rows": rows,
    }


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Stage Drilldown Index")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- rows: `{summary['rows_total']}`")
    print(f"- stage routes: `{summary['stage_routes_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni stage-first drilldown index JSON."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output
    if not output.is_absolute():
        output = root / output

    payload = build_stage_drilldown(root)
    _save_json(payload, output)
    _print_markdown(payload)

    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
