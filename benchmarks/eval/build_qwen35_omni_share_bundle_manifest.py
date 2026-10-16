# SPDX-License-Identifier: Apache-2.0
"""Build the share-bundle manifest for the Qwen3.5-Omni report package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval.build_qwen35_omni_final_readiness import (
    MACHINE_EVIDENCE_FILES,
    REPORT_FILES,
)


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "share_bundle_manifest.json"

EXTRA_MACHINE_EVIDENCE = [
    "results/qwen35_report_audit_20260619/final_readiness_audit.json",
]

SHARE_TOOL_FILES = [
    "benchmarks/eval/validate_qwen35_omni_share_package.py",
    "benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
    "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
    "benchmarks/eval/qwen35_omni_collaborator_return_check.py",
]

PRIMARY_READING_ORDER = [
    "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_unit_test_smoke_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _record(root: Path, rel_path: str, *, category: str, required: bool = True) -> dict[str, Any]:
    path = (root / rel_path).resolve()
    record: dict[str, Any] = {
        "category": category,
        "relative_path": rel_path,
        "path": str(path),
        "required": required,
        "exists": path.exists(),
    }
    if path.is_file():
        record.update(
            {
                "type": "file",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    elif path.is_dir():
        files = sorted(child for child in path.rglob("*") if child.is_file())
        record.update(
            {
                "type": "directory",
                "file_count": len(files),
                "size_bytes": sum(child.stat().st_size for child in files),
                "files": [
                    {
                        "relative_path": str(child.relative_to(root)),
                        "size_bytes": child.stat().st_size,
                        "sha256": _sha256(child),
                    }
                    for child in files
                ],
            }
        )
    else:
        record["type"] = "missing"
    return record


def _dedupe(paths: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def build_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    chart_manifest = _load_json_optional(audit_dir / "share_charts/chart_pack_manifest.json")
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    evidence_manifest = _load_json_optional(audit_dir / "manifest.json")

    chart_records: list[dict[str, Any]] = []
    for item in chart_manifest.get("generated_files", []):
        path_text = str(item.get("path") or "")
        if not path_text:
            continue
        path = Path(path_text)
        try:
            rel_path = str(path.resolve().relative_to(root))
        except ValueError:
            rel_path = path_text
        chart_records.append(_record(root, rel_path, category="share_chart"))

    records: list[dict[str, Any]] = []
    records.extend(
        _record(root, rel_path, category="share_report")
        for rel_path in REPORT_FILES
    )
    machine_evidence = [
        rel_path
        for rel_path in _dedupe([*MACHINE_EVIDENCE_FILES, *EXTRA_MACHINE_EVIDENCE])
        if rel_path != str(DEFAULT_OUTPUT)
    ]
    records.extend(
        _record(root, rel_path, category="machine_evidence")
        for rel_path in machine_evidence
    )
    records.extend(
        _record(root, rel_path, category="share_tool")
        for rel_path in SHARE_TOOL_FILES
    )
    records.append(
        _record(
            root,
            "results/qwen35_report_audit_20260619/share_charts",
            category="share_chart_directory",
        )
    )
    records.extend(chart_records)

    missing_required = [
        record
        for record in records
        if record["required"] and not record["exists"]
    ]
    chart_summary = chart_manifest.get("summary", {})
    final_summary = final_readiness.get("summary", {})
    manifest_summary = evidence_manifest.get("summary", {})
    audit_summary_ok = bool(audit_summary.get("ok"))
    audit_recovery_ok = (
        bool(final_summary.get("ready"))
        and int(final_summary.get("required_failures") or 0) == 0
        and int(manifest_summary.get("missing_records") or 0) == 0
        and bool(chart_summary.get("ready"))
    )
    checks = {
        "all_required_files_present": not missing_required,
        "full_audit_ok": audit_summary_ok or audit_recovery_ok,
        "evidence_manifest_ready": int(manifest_summary.get("missing_records") or 0) == 0,
        "chart_pack_ready": bool(chart_summary.get("ready"))
        and int(chart_summary.get("csv_files") or 0) >= 7
        and int(chart_summary.get("svg_files") or 0) >= 7
        and int(chart_summary.get("generated_files") or 0) >= 14,
        "share_tool_ready": all((root / rel_path).is_file() for rel_path in SHARE_TOOL_FILES),
        "primary_reading_order_present": all(
            (root / rel_path).is_file() for rel_path in PRIMARY_READING_ORDER
        ),
    }

    category_counts: dict[str, int] = {}
    for record in records:
        category = str(record["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": all(checks.values()),
            "records_total": len(records),
            "missing_required": len(missing_required),
            "file_records": sum(1 for record in records if record.get("type") == "file"),
            "directory_records": sum(
                1 for record in records if record.get("type") == "directory"
            ),
            "category_counts": category_counts,
            "checks": checks,
        },
        "primary_reading_order": PRIMARY_READING_ORDER,
        "source_gates": {
            "audit_summary": {
                "ok": audit_summary.get("ok"),
                "recovered_from_current_gates": (not audit_summary_ok)
                and audit_recovery_ok,
                "claims": audit_summary.get("claims"),
                "coverage": audit_summary.get("coverage"),
                "preflight": audit_summary.get("preflight"),
                "manifest": manifest_summary,
                "final_readiness": final_summary,
            },
            "final_readiness": final_summary,
            "chart_pack": chart_summary,
            "evidence_manifest": manifest_summary,
        },
        "records": records,
        "missing_required": missing_required,
        "send_note": (
            "This manifest is the recommended external share bundle. "
            "Raw benchmark artifacts remain tracked by manifest.json."
        ),
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Share Bundle Manifest\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Records | {summary['records_total']} |")
    print(f"| Missing required | {summary['missing_required']} |")
    print(f"| Files | {summary['file_records']} |")
    print(f"| Directories | {summary['directory_records']} |")
    print("\n| Check | Value |")
    print("| --- | ---: |")
    for name, value in summary["checks"].items():
        print(f"| {name} | {value} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the external Qwen3.5-Omni share-bundle manifest."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_bundle(root)
    output = args.json_output
    if not output.is_absolute():
        output = root / output
    _save_json(payload, output)
    print_markdown(payload)
    print(
        "Share bundle manifest written: "
        f"{output} ready={payload['summary']['ready']} "
        f"records={payload['summary']['records_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
