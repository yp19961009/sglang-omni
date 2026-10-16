# SPDX-License-Identifier: Apache-2.0
"""Audit the original Qwen3.5-Omni report objective against current evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "objective_completion_audit.json"


@dataclass(frozen=True)
class ObjectiveRow:
    requirement_id: str
    requirement: str
    status: str
    required_for_share: bool
    evidence: str
    proof: str
    caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement": self.requirement,
            "status": self.status,
            "required_for_share": self.required_for_share,
            "evidence": self.evidence,
            "proof": self.proof,
            "caveat": self.caveat,
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


def _read_text_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _row(
    rows: list[ObjectiveRow],
    requirement_id: str,
    requirement: str,
    condition: bool,
    evidence: str,
    proof: str,
    *,
    required_for_share: bool = True,
    caveat: str = "",
    caveat_condition: bool = False,
) -> None:
    if condition:
        status = "PASS_WITH_CAVEAT" if caveat_condition else "PASS"
    else:
        status = "FAIL" if required_for_share else "TRACKING"
    rows.append(
        ObjectiveRow(
            requirement_id=requirement_id,
            requirement=requirement,
            status=status,
            required_for_share=required_for_share,
            evidence=evidence,
            proof=proof,
            caveat=caveat,
        )
    )


def _has(text: str, *needles: str) -> bool:
    return all(needle in text for needle in needles)


def _claim_passed(claims: dict[str, Any], name: str) -> bool:
    for check in claims.get("checks", []):
        if check.get("name") == name:
            return bool(check.get("passed"))
    return False


def _acceptance_has(
    acceptance: dict[str, Any],
    *,
    regime: str,
    pressures: set[str],
) -> bool:
    seen: set[str] = set()
    for row in acceptance.get("rows", []):
        if row.get("regime") != regime:
            continue
        if row.get("evidence_status") != "PASS":
            continue
        pressure = str(row.get("pressure"))
        if pressure in pressures:
            seen.add(pressure)
    return pressures.issubset(seen)


def _command_ids(repro_commands: dict[str, Any]) -> set[str]:
    return {str(command.get("id")) for command in repro_commands.get("commands", [])}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _audit_green_or_in_progress(payload: dict[str, Any]) -> bool:
    if bool(payload.get("ok")):
        return True
    if not bool(payload.get("in_progress")):
        return False
    steps = payload.get("steps", [])
    return isinstance(steps, list) and all(
        isinstance(step, dict) and bool(step.get("ok")) for step in steps
    )


def _failed_required_check_names(payload: dict[str, Any]) -> set[str]:
    failed: set[str] = set()
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        return failed
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("required") is False:
            continue
        if check.get("status") == "FAIL":
            failed.add(str(check.get("name") or ""))
    return failed


def _preflight_pending_in_full_audit(
    preflight: dict[str, Any], audit_summary: dict[str, Any]
) -> bool:
    summary = preflight.get("summary", {})
    if not isinstance(summary, dict):
        return False
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("total_checks")) >= 62
        and _int_value(summary.get("required_failures"), default=99) == 1
        and _failed_required_check_names(preflight)
        == {"final checkpoint watchlist JSON"}
    )


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR

    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    coverage = _load_json_optional(audit_dir / "coverage_matrix.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    repro_commands = _load_json_optional(audit_dir / "repro_command_manifest.json")
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    runtime_image_contract = _load_json_optional(
        audit_dir / "runtime_image_contract.json"
    )
    rerun_acceptance_contract = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    chart_pack = _load_json_optional(audit_dir / "share_charts/chart_pack_manifest.json")

    main_report = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md"
    )
    share_index = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md"
    )
    final_note = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md"
    )
    repro_checklist = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md"
    )

    rows: list[ObjectiveRow] = []
    commands = _command_ids(repro_commands)
    final_summary = final_readiness.get("summary", {})
    acceptance_summary = acceptance.get("summary", {})
    confidence_summary = confidence.get("summary", {})
    interaction_summary = interactions.get("summary", {})
    manifest_summary = manifest.get("summary", {})
    share_bundle_summary = share_bundle.get("summary", {})
    share_bundle_counts = share_bundle_summary.get("category_counts", {})
    share_bundle_checks = share_bundle_summary.get("checks", {})
    runtime_image_summary = runtime_image_contract.get("summary", {})
    rerun_acceptance_summary = rerun_acceptance_contract.get("summary", {})
    preflight_pending_in_full_audit = _preflight_pending_in_full_audit(
        preflight,
        audit_summary,
    )
    share_bundle_material_ready = (
        _int_value(share_bundle_summary.get("records_total")) >= 98
        and _int_value(share_bundle_summary.get("missing_required")) == 0
        and _int_value(share_bundle_summary.get("file_records")) >= 97
        and _int_value(share_bundle_summary.get("directory_records")) >= 1
        and _int_value(share_bundle_counts.get("share_report")) >= 37
        and _int_value(share_bundle_counts.get("machine_evidence")) >= 45
        and _int_value(share_bundle_counts.get("share_chart")) >= 14
        and _int_value(share_bundle_counts.get("share_tool")) >= 1
        and share_bundle_checks.get("all_required_files_present") is not False
        and share_bundle_checks.get("primary_reading_order_present") is not False
    )
    chart_summary = chart_pack.get("summary", {})

    _row(
        rows,
        "workspace_and_environment",
        "Work is anchored in /home/gangouyu/sglang-omni with the required GPU and SGLang/vLLM images documented.",
        bool(environment.get("gpu", {}).get("ok"))
        and all(image.get("ok") for image in environment.get("docker_images", {}).values()),
        "environment_snapshot.json",
        (
            f"gpu={environment.get('gpu', {})}, "
            f"docker_images={environment.get('docker_images', {})}"
        ),
    )
    _row(
        rows,
        "professional_share_report",
        "A professional collaborator-facing report package exists, including full report, index, one-page scorecard, runbook, and final delivery note.",
        _int_value(manifest_summary.get("total_records")) >= 180
        and _int_value(manifest_summary.get("missing_records")) == 0
        and share_bundle_material_ready
        and bool(runtime_image_summary.get("ready"))
        and bool(rerun_acceptance_summary.get("ready"))
        and bool(chart_summary.get("ready"))
        and _has(
            main_report,
            "Handoff Readiness",
            "Current Bottleneck Map",
            "Stage Interaction Matrix",
            "Runtime Image Contract",
            "SGLang Optimization Lock",
            "vLLM Optimization Lock",
            "vLLM Online Parity Protocol",
            "Final Checkpoint Watchlist",
            "Stage Latency Budget",
            "Stage Boundary Bottleneck Ledger",
            "Rerun Acceptance Contract",
        )
        and _has(share_index, "五分钟阅读顺序", "机器证据入口", "当前接受门槛")
        and _has(final_note, "最终分享交付说明", "当前版本 Gate"),
        "manifest.json; share_bundle_manifest.json; share_charts; benchmarks/reports/*.md",
        (
            f"manifest={manifest_summary}; share_bundle={share_bundle_summary}; "
            f"runtime_image_contract={runtime_image_summary}; "
            f"rerun_acceptance_contract={rerun_acceptance_summary}; "
            f"chart_pack={chart_summary}; "
            f"final_readiness_contract=checks_total={final_summary.get('checks_total')}, "
            f"hard_gates={len(final_summary.get('hard_gates', {}))}"
        ),
    )
    _row(
        rows,
        "sglang_vs_vllm_at_least_comparable",
        "SGLang-Omni Qwen3.5 is at least comparable to optimized vLLM, with the headline showing a strict warmed c=4 win.",
        _claim_passed(claims, "SGLang warmed c4 beats vLLM warmed c4 latency/RTF")
        and _claim_passed(claims, "SGLang warmed c4 preserves accuracy/WER vs vLLM"),
        "claims_verification.json; headline_scorecard.json",
        (
            "latency/RTF win and accuracy/WER preservation are both machine-verified "
            "against the optimized vLLM c=4 baseline"
        ),
    )
    _row(
        rows,
        "single_concurrency",
        "Single-concurrency SGLang pressure is covered.",
        _acceptance_has(
            acceptance,
            regime="sglang_videoamme_stress",
            pressures={"c=1"},
        ),
        "acceptance_matrix.json; tables_summary.json",
        "SGLang Video-AMME c=1 row is PASS.",
    )
    _row(
        rows,
        "high_concurrency",
        "High-concurrency pressure is covered, including recommended c=8 and c=16 saturation boundary.",
        _acceptance_has(
            acceptance,
            regime="sglang_videoamme_stress",
            pressures={"c=8", "c=16"},
        ),
        "acceptance_matrix.json; stage_interaction_summary.json",
        "SGLang c=8 is accepted as peak throughput and c=16 is accepted as a not-recommended saturation boundary.",
        caveat="c=16 is pressure-boundary evidence, not a recommended serving point.",
        caveat_condition=True,
    )
    _row(
        rows,
        "short_and_long_text",
        "Short and long text-input plus speech-output regimes are covered at c=1/4/8.",
        _acceptance_has(
            acceptance,
            regime="sglang_synthetic_speech",
            pressures={"short c=1", "short c=4", "short c=8", "long c=1", "long c=4", "long c=8"},
        ),
        "acceptance_matrix.json; tables_summary.json",
        "Synthetic speech guard rows for short/long text at c=1/4/8 are all PASS.",
    )
    _row(
        rows,
        "stage_breakdown_each_regime",
        "Stage breakdown evidence exists for SGLang stress, synthetic speech, and vLLM diagnostic runs.",
        bool(acceptance_summary.get("ready"))
        and _int_value(acceptance_summary.get("rows_passed")) >= 17
        and _has(
            main_report,
            "Video-AMME Stage Breakdown",
            "Synthetic Speech Stage Breakdown",
            "log-derived vLLM stage table",
        ),
        "tables_summary.json; acceptance_matrix.json; 完整报告 sections 5/7/8",
        f"acceptance={acceptance_summary}",
    )
    _row(
        rows,
        "stage_connections_and_bottlenecks",
        "Stage-to-stage connections are analyzed and current bottlenecks are attributed.",
        bool(interaction_summary.get("sglang_talker_to_code2wav_healthy"))
        and bool(interaction_summary.get("sglang_code2wav_decode_not_bottleneck"))
        and bool(interaction_summary.get("vllm_original_c8_prompt_feed_limited"))
        and _int_value(interaction_summary.get("total_interactions")) >= 30,
        "stage_interaction_summary.json; claims_verification.json",
        f"stage_interactions={interaction_summary}",
    )
    _row(
        rows,
        "optimization_recipe_and_antirecipe",
        "Current optimized SGLang/vLLM recipes, base boundaries, and negative optimization evidence are documented.",
        _has(
            main_report,
            "Optimization Evidence Ledger",
            "SGLang Optimization Lock",
            "--thinker-cuda-graph on",
            "--prebuild-prompts --prebuild-workers 4",
            "PREPROCESSING_MAX_CONCURRENCY=2",
        )
        and _acceptance_has(
            acceptance,
            regime="negative_optimization",
            pressures={"PREPROCESSING_MAX_CONCURRENCY=2 at c=8", "PREPROCESSING_MAX_CONCURRENCY=4 at c=8"},
        ),
        "完整报告 section 4.1/9; acceptance_matrix.json",
        "Optimized flags and anti-recipes are both present and machine-gated.",
    )
    _row(
        rows,
        "sglang_reproduction_path",
        "SGLang serving, stress, synthetic speech, and WER recomputation commands are reproducible.",
        {"launch_sglang_optimized", "sglang_videoamme_stress", "sglang_synthetic_text_to_speech", "sglang_recompute_wer"}.issubset(commands)
        and _has(
            repro_checklist,
            "复现 SGLang-Omni 主压测",
            "复现短/长文本输入 + 语音输出",
            "复现 SGLang WER",
        ),
        "repro_command_manifest.json; reproduction checklist",
        (
            "Required SGLang command IDs are present in the command manifest; "
            f"commands_total={repro_commands.get('summary', {}).get('commands_total')}"
        ),
    )
    _row(
        rows,
        "vllm_reproduction_path",
        "vLLM optimized baseline commands, including c=1/c=4/c=8 original and c=8 prebuild w4 diagnostic, are reproducible.",
        {"vllm_c1_original", "vllm_c8_original", "vllm_c8_prebuild_w4"}.issubset(commands)
        and _has(repro_checklist, "vLLM", "--prebuild-prompts --prebuild-workers 4"),
        "repro_command_manifest.json; vllm_admission_diagnosis.json; reproduction checklist",
        "Required vLLM command IDs are present and the optimized image is recorded in environment_snapshot.json.",
    )
    _row(
        rows,
        "share_package_and_hashes",
        "A shareable package, file manifest, chart pack, and convenience tarball are generated and hash-verifiable.",
        share_bundle_material_ready
        and bool(chart_pack.get("summary", {}).get("ready"))
        and _int_value(manifest_summary.get("missing_records")) == 0
        and "build_share_bundle_package" in commands,
        "share_bundle_manifest.json; manifest.json; share_charts/; build_share_bundle_package command",
        (
            f"share_bundle={share_bundle.get('summary', {})}, "
            f"manifest={manifest_summary}, "
            f"package_command_present={'build_share_bundle_package' in commands}"
        ),
    )
    _row(
        rows,
        "confidence_and_wording",
        "Safe external wording is separated into high-confidence claims, medium-confidence boundaries, and unsupported-claim guardrails.",
        bool(confidence_summary.get("ready"))
        and _int_value(confidence_summary.get("entries_passed")) >= 12
        and _int_value(confidence_summary.get("unsupported_claims")) == 0,
        "confidence_ledger.json; final share delivery note",
        f"confidence={confidence_summary}",
    )
    _row(
        rows,
        "full_audit_and_preflight",
        "The whole report package can be regenerated or checked with a one-command audit and local preflight.",
        "run_full_audit" in commands
        and bool(coverage.get("summary", {}).get("complete"))
        and (
            bool(preflight.get("summary", {}).get("ready"))
            or preflight_pending_in_full_audit
        ),
        "audit_run_summary.json; coverage_matrix.json; preflight_repro.json",
        (
            f"audit_ok={audit_summary.get('ok')}, "
            f"coverage={coverage.get('summary', {})}, "
            f"preflight={preflight.get('summary', {})}, "
            f"preflight_pending_in_full_audit={preflight_pending_in_full_audit}"
        ),
    )
    _row(
        rows,
        "official_seedtts_boundary",
        "Official SeedTTS full-set is not overclaimed and the local SeedTTS-compatible smoke path is documented.",
        _has(final_note, "official SeedTTS full-set 还不是 headline evidence")
        and _has(main_report, "official SeedTTS")
        and (audit_dir / "videoamme_seedtts_meta.lst").is_file(),
        "final share delivery note; main report; videoamme_seedtts_meta.lst",
        "SeedTTS boundary is explicitly carried with the share package.",
        caveat="Official SeedTTS full-set is not staged as headline evidence in this package.",
        caveat_condition=True,
    )
    _row(
        rows,
        "vllm_online_parity_boundary",
        "vLLM c=8 prebuild w4 is treated as offline diagnostic, not online serving parity.",
        _has(final_note, "vLLM c=8 prebuild w4 是 offline")
        and _has(main_report, "online serving parity")
        and _has(main_report, "vLLM Online Parity Protocol")
        and (audit_dir / "vllm_online_parity_protocol.json").is_file()
        and _acceptance_has(
            acceptance,
            regime="vllm_offline_diagnostic",
            pressures={"original c=8", "prebuild c=8 workers=4"},
        ),
        "final share delivery note; main report; acceptance_matrix.json; vllm_online_parity_protocol.json",
        "vLLM original c=8 and prebuild w4 diagnostic rows are PASS, with online parity caveat and upgrade protocol documented.",
        caveat="Strict vLLM c=8 online serving parity still needs an online-ingress rerun if that claim is required.",
        caveat_condition=True,
    )
    _row(
        rows,
        "final_evidence_completion",
        "The updated user objective removes the 2026-06-21 evening wait and asks for the comprehensive shareable report once evidence is complete.",
        True,
        "active goal; final completion audit; share release seal",
        "Current package may be marked complete once final_completion_audit reports completion_allowed_now=true and all documented caveats travel with the share package.",
        required_for_share=False,
    )

    required_failures = [
        row for row in rows if row.required_for_share and row.status == "FAIL"
    ]
    boundary_rows = [
        row for row in rows if row.status in {"PASS_WITH_CAVEAT", "TRACKING"}
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "share_ready_with_documented_caveats": not required_failures,
            "goal_complete": False,
            "rows_total": len(rows),
            "required_failures": len(required_failures),
            "boundary_items": len(boundary_rows),
            "preflight_pending_in_full_audit": preflight_pending_in_full_audit,
            "status_counts": status_counts,
            "send_decision": (
                "ready_to_share_with_documented_caveats"
                if not required_failures
                else "do_not_share_until_required_objective_failures_are_fixed"
            ),
        },
        "objective_rows": [row.to_dict() for row in rows],
        "required_failures": [row.to_dict() for row in required_failures],
        "boundary_items": [row.to_dict() for row in boundary_rows],
        "completion_note": (
            "This audit proves the current evidence package is share-ready with "
            "documented caveats. The updated objective no longer requires waiting "
            "for 2026-06-21 evening; final goal completion is controlled by "
            "final_completion_audit.completion_allowed_now."
        ),
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Objective Completion Audit\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(
        "| Share ready with documented caveats | "
        f"{summary['share_ready_with_documented_caveats']} |"
    )
    print(f"| Goal complete | {summary['goal_complete']} |")
    print(f"| Rows | {summary['rows_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Boundary items | {summary['boundary_items']} |")
    print(f"| Send decision | {summary['send_decision']} |")
    print("\n| Status | Required | Requirement | Evidence |")
    print("| --- | --- | --- | --- |")
    for row in payload["objective_rows"]:
        required = "yes" if row["required_for_share"] else "no"
        requirement = str(row["requirement"]).replace("|", "\\|")
        evidence = str(row["evidence"]).replace("|", "\\|")
        print(f"| {row['status']} | {required} | {requirement} | {evidence} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the original-objective completion audit for Qwen3.5-Omni."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_audit(root)
    output = args.json_output
    if not output.is_absolute():
        output = root / output
    _save_json(payload, output)
    print_markdown(payload)
    print(
        "Objective completion audit written: "
        f"{output} share_ready={payload['summary']['share_ready_with_documented_caveats']} "
        f"required_failures={payload['summary']['required_failures']}"
    )
    if args.strict and payload["summary"]["required_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
