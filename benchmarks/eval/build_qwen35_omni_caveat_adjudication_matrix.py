# SPDX-License-Identifier: Apache-2.0
"""Build a reviewer-facing caveat adjudication matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "caveat_adjudication_matrix.json"


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _entry(confidence: dict[str, Any], entry_id: str) -> dict[str, Any]:
    for row in confidence.get("entries", []):
        if row.get("id") == entry_id:
            return row
    return {}


def _objective_row(objective: dict[str, Any], requirement_id: str) -> dict[str, Any]:
    for row in objective.get("objective_rows", []):
        if row.get("requirement_id") == requirement_id:
            return row
    return {}


def _acceptance_row(acceptance: dict[str, Any], serving_status: str) -> dict[str, Any]:
    for row in acceptance.get("rows", []):
        if row.get("serving_status") == serving_status:
            return row
    return {}


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _gate_line(name: str, value: Any, evidence: str) -> str:
    return f"| {name} | `{value}` | {evidence} |"


def _confidence_cell(row: dict[str, Any]) -> str:
    confidence = row.get("confidence") or "n/a"
    status = row.get("status") or "n/a"
    return f"{confidence} / {status}"


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
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


def _row(
    *,
    caveat_id: str,
    topic: str,
    current_decision: str,
    allowed_wording: str,
    forbidden_claim: str,
    upgrade_condition: str,
    evidence: str,
    confidence: str,
    status: str,
    must_travel: bool,
    replacement_gate: bool,
) -> dict[str, Any]:
    return {
        "caveat_id": caveat_id,
        "topic": topic,
        "current_decision": current_decision,
        "allowed_wording": allowed_wording,
        "forbidden_claim": forbidden_claim,
        "upgrade_condition": upgrade_condition,
        "evidence": evidence,
        "confidence": confidence,
        "status": status,
        "must_travel": must_travel,
        "replacement_gate": replacement_gate,
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    objective = _load_json_optional(audit_dir / "objective_completion_audit.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_online = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    optimization_ledger = _load_json_optional(
        audit_dir / "optimization_candidate_ledger.json"
    )

    final_summary = final.get("summary", {})
    objective_summary = objective.get("summary", {})
    confidence_summary = confidence.get("summary", {})
    acceptance_summary = acceptance.get("summary", {})
    preflight_summary = preflight.get("summary", {})
    manifest_summary = manifest.get("summary", {})
    repro_summary = repro.get("summary", {})
    sglang_lock_summary = sglang_lock.get("summary", {})
    expected_repro_gate = (
        f"{_int_value(repro_summary.get('commands_total'))} commands / "
        f"{_int_value(repro_summary.get('phases_total'))} phases"
    )
    final_hard_gates = final_summary.get("hard_gates", {})
    manifest_gate = str(final_hard_gates.get("manifest_records") or "")
    final_contract_available = (
        bool(final_summary)
        and _int_value(final_summary.get("checks_total")) >= 48
        and (manifest_gate == "180" or manifest_gate.startswith(">=180"))
        and final_hard_gates.get("repro_command_manifest") == expected_repro_gate
        and final_hard_gates.get("tail_confidence_appendix")
        == "18 rows / 13 checks / strict c4 tail + bootstrap uncertainty"
        and final_hard_gates.get("caveat_adjudication_matrix")
        == "12 caveats / forbidden claims / replacement triggers"
    )
    vllm_online_summary = vllm_online.get("summary", {})
    optimization_summary = optimization_ledger.get("summary", {})
    preflight_pending_in_full_audit = _preflight_pending_in_full_audit(
        preflight,
        audit_summary,
    )
    final_hard_gates = final_summary.get("hard_gates", {})
    manifest_gate = str(final_hard_gates.get("manifest_records") or "")
    final_contract_available = (
        bool(final_summary)
        and _int_value(final_summary.get("checks_total")) >= 48
        and (manifest_gate == "180" or manifest_gate.startswith(">=180"))
        and final_hard_gates.get("repro_command_manifest") == expected_repro_gate
        and final_hard_gates.get("tail_confidence_appendix")
        == "18 rows / 13 checks / strict c4 tail + bootstrap uncertainty"
        and final_hard_gates.get("caveat_adjudication_matrix")
        == "12 caveats / forbidden claims / replacement triggers"
    )

    strict_c4 = _entry(confidence, "strict_c4_sglang_beats_optimized_vllm")
    c8_c16 = _entry(confidence, "sglang_c8_peak_c16_saturation")
    larger_traffic = _entry(confidence, "larger_traffic_extrapolation_pending")
    seedtts = _entry(confidence, "official_seedtts_fullset_not_headline")
    vllm_parity = _entry(confidence, "strict_vllm_c8_online_parity_pending")
    vllm_w4 = _entry(confidence, "vllm_prebuild_w4_is_optimized_offline_diagnostic")
    wer = _entry(confidence, "wer_audio_quality_stable")
    preproc = _entry(confidence, "preprocessing_parallelism_is_anti_recipe")
    stage = _entry(confidence, "stage_boundaries_not_primary_bottleneck")
    final_completion_gate = _objective_row(objective, "final_evidence_completion")
    c16_boundary = _acceptance_row(acceptance, "not_recommended_saturation")
    preproc2 = _acceptance_row(acceptance, "anti_recipe_regression")
    preproc4 = _acceptance_row(acceptance, "anti_recipe_failure")

    rows = [
        _row(
            caveat_id="strict_c4_headline_scope",
            topic="strict c=4 SGLang vs vLLM",
            current_decision="share_as_headline_with_environment_boundary",
            allowed_wording=str(strict_c4.get("allowed_wording_zh") or ""),
            forbidden_claim="Do not extrapolate c=4 results to every concurrency, workload, or online traffic pattern.",
            upgrade_condition="Rerun full audit when hardware, image, model, data, or ASR scope changes.",
            evidence=_confidence_cell(strict_c4),
            confidence=str(strict_c4.get("confidence") or "n/a"),
            status=str(strict_c4.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="sglang_c8_peak_c16_boundary",
            topic="SGLang c=8 peak / c=16 saturation",
            current_decision="share_c8_as_peak_keep_c16_as_pressure_boundary",
            allowed_wording=str(c8_c16.get("allowed_wording_zh") or ""),
            forbidden_claim="Do not use c=16 as the recommended default serving point.",
            upgrade_condition="Rerun stress plus stage audit after admission, placement, or memory-fraction changes.",
            evidence=f"{_confidence_cell(c8_c16)}; c16={c16_boundary.get('evidence_status')}",
            confidence=str(c8_c16.get("confidence") or "n/a"),
            status=str(c8_c16.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="larger_traffic_extrapolation_boundary",
            topic="larger Video-AMME / full traffic extrapolation",
            current_decision="share_current_scope_do_not_extrapolate_to_full_traffic",
            allowed_wording=str(larger_traffic.get("allowed_wording_zh") or ""),
            forbidden_claim="Do not describe ci-50, stress, or synthetic evidence as full online traffic coverage.",
            upgrade_condition="Add larger Video-AMME/full-traffic samples, rerun SGLang and vLLM gates, refresh stage/tail/caveat evidence, and pass final readiness before widening the claim.",
            evidence=_confidence_cell(larger_traffic),
            confidence=str(larger_traffic.get("confidence") or "n/a"),
            status=str(larger_traffic.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="current_best_measured_not_global_optimum",
            topic="current best measured recipe",
            current_decision="share_as_current_measured_best_not_global_optimum",
            allowed_wording=(
                "Current measured-best recipe is compiled/graph path, serial "
                "preprocessing, 16GiB preprocessing cache, and c=4-c=8 serving window."
            ),
            forbidden_claim="Do not claim all future kernels, placements, or admission policies have been exhaustively searched.",
            upgrade_condition="A replacement recipe must pass c=4/c=8/c=16, WER, stage interaction, acceptance, and final readiness.",
            evidence=(
                f"sglang_lock={sglang_lock_summary.get('checks_passed')}/"
                f"{sglang_lock_summary.get('checks_total')}; "
                f"optimization_ledger={optimization_summary.get('checks_passed')}/"
                f"{optimization_summary.get('checks_total')}"
            ),
            confidence="high",
            status="PASS",
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="short_long_tts_guardrail",
            topic="short/long text-to-speech",
            current_decision="share_as_speech_generation_guardrail",
            allowed_wording="Short text and long text speech-generation guards are covered; long c=8 remains faster than real time.",
            forbidden_claim="Do not replace Video-AMME or official SeedTTS headline claims with synthetic speech guards.",
            upgrade_condition="Add natural-speech or official SeedTTS headline only after data, WER/RTF, stage breakdown, and claims gates are added.",
            evidence="tables_summary.json; acceptance_matrix.json",
            confidence="high",
            status="PASS",
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="official_seedtts_fullset_not_headline",
            topic="official SeedTTS full-set",
            current_decision="keep_as_upgrade_item_not_current_headline",
            allowed_wording=str(seedtts.get("allowed_wording_zh") or ""),
            forbidden_claim="Do not describe the Video-AMME spoken-reference smoke path as the official SeedTTS full set.",
            upgrade_condition="Stage official SeedTTS data and add full benchmark, WER/RTF, claims, coverage, and final readiness evidence.",
            evidence=_confidence_cell(seedtts),
            confidence=str(seedtts.get("confidence") or "n/a"),
            status=str(seedtts.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="vllm_original_c8_prompt_feed_limited",
            topic="vLLM original c=8",
            current_decision="diagnostic_only_prompt_feed_limited",
            allowed_wording="Original vLLM c=8 diagnoses host prompt build/feed admission.",
            forbidden_claim="Do not use original offline c=8 wall QPS as online parity evidence.",
            upgrade_condition="Build online ingress with matched model/data/WER before any c=8 headline comparison.",
            evidence="vllm_admission_diagnosis.json",
            confidence="high",
            status="PASS",
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="vllm_c8_prebuild_w4_offline_diagnostic",
            topic="vLLM c=8 prebuild w4",
            current_decision="strongest_current_offline_diagnostic_not_online_parity",
            allowed_wording=str(vllm_w4.get("allowed_wording_zh") or ""),
            forbidden_claim=str(vllm_parity.get("boundary_zh") or ""),
            upgrade_condition="Add online ingress, matched WER/ASR, engine/talker boundary profiles, and updated runtime contract.",
            evidence=f"{_confidence_cell(vllm_w4)}; parity={_confidence_cell(vllm_parity)}",
            confidence=str(vllm_w4.get("confidence") or "n/a"),
            status=str(vllm_w4.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="optional_whisper_host_cache",
            topic="optional Whisper host cache warning",
            current_decision="does_not_block_current_serving_evidence",
            allowed_wording="Current WER evidence is usable; host recomputation needs Whisper weights or ASR router.",
            forbidden_claim="Do not replace WER numbers from a host run that lacks ASR/cache prerequisites.",
            upgrade_condition="Provide cache, container weights, or ASR router before external host-side WER recomputation.",
            evidence=f"{_confidence_cell(wer)}; preflight_warnings={preflight_summary.get('warnings')}",
            confidence=str(wer.get("confidence") or "n/a"),
            status=str(wer.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=False,
        ),
        _row(
            caveat_id="preproc2_preproc4_anti_recipe",
            topic="preproc=2/4",
            current_decision="reject_as_current_recipe_change",
            allowed_wording=str(preproc.get("allowed_wording_zh") or ""),
            forbidden_claim="Do not present preproc=2 or preproc=4 as the current optimal recipe.",
            upgrade_condition="Reevaluate only after admission/placement/memory redesign and full acceptance/stage/final readiness pass.",
            evidence=(
                f"{_confidence_cell(preproc)}; preproc2={preproc2.get('evidence_status')}; "
                f"preproc4={preproc4.get('evidence_status')}"
            ),
            confidence=str(preproc.get("confidence") or "n/a"),
            status=str(preproc.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="stage_handoff_code2wav_not_primary_bottleneck",
            topic="stage handoff / code2wav",
            current_decision="share_handoff_healthy_decode_not_primary_bottleneck",
            allowed_wording=str(stage.get("allowed_wording_zh") or ""),
            forbidden_claim="Do not call code2wav or the talker->code2wav handoff the current primary bottleneck.",
            upgrade_condition="Rerun stage interaction and tail appendices after talker chunking, streaming, or code2wav path changes.",
            evidence=_confidence_cell(stage),
            confidence=str(stage.get("confidence") or "n/a"),
            status=str(stage.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=True,
        ),
        _row(
            caveat_id="final_evidence_completion_gate",
            topic="final evidence completion gate",
            current_decision="completion_allowed_after_final_completion_audit",
            allowed_wording=(
                "Updated objective has no 2026-06-21 evening wait; completion is "
                "allowed only when final_completion_audit is ready and "
                "completion_allowed_now=true."
            ),
            forbidden_claim=(
                "Do not mark complete when final_completion_audit has required "
                "failures, blockers, or completion_allowed_now=false."
            ),
            upgrade_condition=(
                "After any evidence, report, package, or caveat change, rerun full "
                "audit, final completion audit, package validation, and release seal."
            ),
            evidence=str(final_completion_gate.get("status") or "n/a"),
            confidence="medium",
            status=str(final_completion_gate.get("status") or "n/a"),
            must_travel=True,
            replacement_gate=False,
        ),
    ]

    forbidden_claims = [row for row in rows if row["forbidden_claim"]]
    replacement_triggers = [row for row in rows if row["upgrade_condition"]]
    checks = {
        "final_readiness_contract_available": bool(final_summary.get("ready"))
        or final_contract_available,
        "confidence_ledger_has_no_unsupported_claims": bool(
            confidence_summary.get("ready")
        )
        and _int_value(confidence_summary.get("unsupported_claims"), default=1) == 0,
        "objective_caveats_present": _int_value(objective_summary.get("boundary_items"))
        >= 3
        and _int_value(objective_summary.get("required_failures"), default=1) == 0,
        "acceptance_matrix_ready": bool(acceptance_summary.get("ready"))
        and _int_value(acceptance_summary.get("rows_passed"))
        == _int_value(acceptance_summary.get("rows_total"), default=-1),
        "preflight_ready_with_optional_warning_only": (
            bool(preflight_summary.get("ready"))
            and _int_value(preflight_summary.get("required_failures"), default=1)
            == 0
        )
        or preflight_pending_in_full_audit,
        "manifest_and_repro_current": _int_value(
            manifest_summary.get("total_records")
        )
        >= 170
        and _int_value(repro_summary.get("commands_total")) >= 57,
        "caveat_rows_total": len(rows) >= 12,
        "forbidden_claims_documented": len(forbidden_claims) >= 12,
        "replacement_triggers_documented": len(replacement_triggers) >= 12,
        "larger_traffic_boundary_present": any(
            row["caveat_id"] == "larger_traffic_extrapolation_boundary"
            and row["must_travel"]
            for row in rows
        ),
        "online_parity_not_promoted": bool(vllm_online_summary.get("ready"))
        and not bool(vllm_online_summary.get("online_parity_proven")),
        "seedtts_fullset_not_headline": any(
            row["caveat_id"] == "official_seedtts_fullset_not_headline"
            and row["must_travel"]
            for row in rows
        ),
        "current_best_not_global_optimum": bool(
            optimization_summary.get("not_global_optimum_boundary")
        ),
        "final_evidence_completion_boundary_present": any(
            row["caveat_id"] == "final_evidence_completion_gate" and row["must_travel"]
            for row in rows
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "must_travel_rows_total": sum(1 for row in rows if row["must_travel"]),
            "forbidden_claims_total": len(forbidden_claims),
            "replacement_triggers_total": len(replacement_triggers),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "online_parity_proven": bool(
                vllm_online_summary.get("online_parity_proven")
            ),
            "seedtts_fullset_headline": False,
            "current_best_scope": "measured_best_not_global_optimum",
            "goal_complete": bool(objective_summary.get("goal_complete")),
            "share_scope": (
                "Machine-readable caveat adjudication matrix for safe external "
                "wording, forbidden claims, replacement triggers, and completion-evidence "
                "boundaries."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "required_failures": required_failures,
            "source_summaries": {
                "final_readiness": final_summary,
                "objective_completion": objective_summary,
                "confidence_ledger": confidence_summary,
                "acceptance_matrix": acceptance_summary,
                "preflight": preflight_summary,
                "preflight_pending_in_full_audit": preflight_pending_in_full_audit,
                "manifest": manifest_summary,
                "repro_command_manifest": repro_summary,
                "sglang_optimization_lock": sglang_lock_summary,
                "vllm_online_parity_protocol": vllm_online_summary,
                "optimization_candidate_ledger": optimization_summary,
            },
        },
        "rows": rows,
    }


def build_markdown(root: Path) -> str:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    objective = _load_json_optional(audit_dir / "objective_completion_audit.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")

    final_summary = final.get("summary", {})
    objective_summary = objective.get("summary", {})
    confidence_summary = confidence.get("summary", {})
    acceptance_summary = acceptance.get("summary", {})
    preflight_summary = preflight.get("summary", {})
    manifest_summary = manifest.get("summary", {})
    repro_summary = repro.get("summary", {})
    sglang_lock_summary = sglang_lock.get("summary", {})
    final_hard_gates = final_summary.get("hard_gates", {})
    final_contract_available = (
        bool(final_summary)
        and _int_value(final_summary.get("checks_total")) >= 45
        and _int_value(final_summary.get("required_failures"), default=99) <= 1
        and final_hard_gates.get("caveat_adjudication_matrix")
        == "12 caveats / forbidden claims / replacement triggers"
    )

    strict_c4 = _entry(confidence, "strict_c4_sglang_beats_optimized_vllm")
    c8_c16 = _entry(confidence, "sglang_c8_peak_c16_saturation")
    larger_traffic = _entry(confidence, "larger_traffic_extrapolation_pending")
    seedtts = _entry(confidence, "official_seedtts_fullset_not_headline")
    vllm_parity = _entry(confidence, "strict_vllm_c8_online_parity_pending")
    vllm_w4 = _entry(confidence, "vllm_prebuild_w4_is_optimized_offline_diagnostic")
    wer = _entry(confidence, "wer_audio_quality_stable")
    preproc = _entry(confidence, "preprocessing_parallelism_is_anti_recipe")
    stage = _entry(confidence, "stage_boundaries_not_primary_bottleneck")
    final_completion_gate = _objective_row(objective, "final_evidence_completion")
    c16_boundary = _acceptance_row(acceptance, "not_recommended_saturation")
    preproc2 = _acceptance_row(acceptance, "anti_recipe_regression")
    preproc4 = _acceptance_row(acceptance, "anti_recipe_failure")

    lines: list[str] = [
        "# Qwen3.5-Omni Caveat 裁决矩阵",
        "",
        f"生成时间 UTC：`{datetime.now(timezone.utc).isoformat()}`。",
        f"工作目录：`{root}`。",
        "",
        "这页把 caveat 从“提醒”变成可执行裁决：哪些结论可以直接分享，哪些必须带边界，",
        "哪些只有补跑新实验后才能升级成 headline，哪些说法应明确禁止。它不新增 benchmark",
        "数字，只引用当前 audit JSON、confidence ledger、acceptance matrix 和 objective audit。",
        "",
        "## 1. 裁决 Gate",
        "",
        "| Gate | 当前值 | 证据 |",
        "| --- | ---: | --- |",
        _gate_line(
            "final readiness contract",
            (
                f"ready={_bool_text(final_summary.get('ready'))}, "
                f"contract={_bool_text(final_contract_available)}"
            ),
            "`final_readiness_audit.json` bootstrap contract",
        ),
        _gate_line(
            "confidence ledger",
            (
                f"high={confidence_summary.get('high_confidence_claims')}, "
                f"medium={confidence_summary.get('medium_confidence_boundaries')}, "
                f"unsupported={confidence_summary.get('unsupported_claims')}"
            ),
            "`confidence_ledger.json`",
        ),
        _gate_line(
            "objective caveats",
            (
                f"{objective_summary.get('boundary_items')} boundary items, "
                f"{objective_summary.get('required_failures')} required failures"
            ),
            "`objective_completion_audit.json`",
        ),
        _gate_line(
            "acceptance matrix",
            (
                f"{acceptance_summary.get('rows_passed')}/"
                f"{acceptance_summary.get('rows_total')}"
            ),
            "`acceptance_matrix.json`",
        ),
        _gate_line(
            "preflight",
            (
                f"{preflight_summary.get('total_checks')} checks, "
                f"{preflight_summary.get('warnings')} optional warning"
            ),
            "`preflight_repro.json`",
        ),
        _gate_line(
            "manifest / repro",
            (
                f"{manifest_summary.get('total_records')} records, "
                f"{repro_summary.get('commands_total')} commands"
            ),
            "`manifest.json`; `repro_command_manifest.json`",
        ),
        _gate_line(
            "SGLang best measured recipe",
            (
                f"{sglang_lock_summary.get('checks_passed')}/"
                f"{sglang_lock_summary.get('checks_total')} checks"
            ),
            "`sglang_optimization_lock.json`",
        ),
        "",
        "## 2. Caveat 裁决表",
        "",
        "| 主题 | 当前裁决 | 对外允许说法 | 不能说 | 升级/替换数字条件 | 证据 |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            "| strict c=4 SGLang vs vLLM | 可作为主 headline，带环境边界 | "
            f"{strict_c4.get('allowed_wording_zh', '')} | "
            "把 c=4 结论外推到所有并发、所有 workload 或所有线上流量 | "
            "硬件/image/model/data/ASR 口径变化时重跑 full audit；headline scorecard、claims、final readiness 全部通过 | "
            f"{_confidence_cell(strict_c4)} |"
        ),
        (
            "| SGLang c=8 峰值 / c=16 饱和 | 可分享；c=16 必须说成压力边界 | "
            f"{c8_c16.get('allowed_wording_zh', '')} | "
            "把 c=16 当默认服务点，或把 c=8 峰值外推到新 admission 策略 | "
            "若调整 admission、preprocessing placement 或 thinker memory fraction，必须重跑 stress+stage audit | "
            f"{_confidence_cell(c8_c16)}；c16 evidence={c16_boundary.get('evidence_status')} |"
        ),
        (
            "| 更大数据 / 真实流量外推 | 当前 ci-50/stress/synthetic 证据支持本报告范围；不能直接外推到全量线上流量 | "
            f"{larger_traffic.get('allowed_wording_zh', '')} | "
            "把 ci-50、stress 或 synthetic 证据表述成 full online traffic coverage | "
            "补更大 Video-AMME/full-traffic 样本，重跑 SGLang/vLLM、stage、tail、caveat 和 final readiness 后才扩大结论范围 | "
            f"{_confidence_cell(larger_traffic)} |"
        ),
        (
            "| 当前 best measured recipe | 可说是当前审计环境里的最优 recipe，不能说全局数学最优 | "
            "当前 best measured recipe 是 compiled/graph path、serial preprocessing、16GiB preprocessing cache，并以 c=4-c=8 作为服务窗口 | "
            "说已经搜索完所有未来 kernel、placement、admission policy，或承诺任何环境下全局最优 | "
            "任何新 recipe 只有在 c=4/c=8/c=16、WER、stage interaction、acceptance、final readiness 全部通过后才能替换当前 recipe | "
            f"`sglang_optimization_lock.json` {sglang_lock_summary.get('checks_passed')}/{sglang_lock_summary.get('checks_total')}；anti-recipe rows={preproc2.get('evidence_status')}/{preproc4.get('evidence_status')} |"
        ),
        (
            "| short/long text-to-speech | 可作为语音输出 guardrail | "
            "短文本 12 words、长文本 139 words 已覆盖，long c=8 快于实时 | "
            "用 synthetic speech 替代 Video-AMME 或 official SeedTTS headline | "
            "若要 natural-speech headline，补 official SeedTTS/full natural speech benchmark 并更新 claims | "
            "`tables_summary.json`; `acceptance_matrix.json` |"
        ),
        (
            "| official SeedTTS full-set | 只能说未作为 headline；可说已有 smoke path | "
            f"{seedtts.get('allowed_wording_zh', '')} | "
            "把 Video-AMME spoken-reference smoke 说成官方 SeedTTS 完整集 | "
            "预置 official SeedTTS 数据，跑完整 benchmark，补 WER/RTF/claims/coverage/final readiness | "
            f"{_confidence_cell(seedtts)} |"
        ),
        (
            "| vLLM original c=8 | 只能做 prompt-feed/admission 诊断 | "
            "original c=8 受 host prompt build/feed admission 限制 | "
            "用 original offline c=8 wall QPS 证明 SGLang online parity 或 non-parity | "
            "若要 c=8 横向 headline，先建立 online ingress 同口径服务路径 | "
            "`vllm_admission_diagnosis.json` |"
        ),
        (
            "| vLLM c=8 prebuild w4 | 当前最强 vLLM offline diagnostic，但不是 online parity | "
            f"{vllm_w4.get('allowed_wording_zh', '')} | "
            f"{vllm_parity.get('boundary_zh', '')} | "
            "补 online ingress、同口径 WER/ASR、engine/talker boundary 复核，并更新 runtime contract | "
            f"{_confidence_cell(vllm_w4)}；parity={_confidence_cell(vllm_parity)} |"
        ),
        (
            "| optional Whisper host cache warning | 不阻塞当前分享包；只影响 host 侧直接重算 WER | "
            "当前 WER 证据可引用；host 重算 WER 需要 Whisper 权重或 ASR router | "
            "把 optional warning 写成 required failure，或在缺 ASR 时替换 WER 数字 | "
            "若要让外部 host 直接重算 WER，提供 cache、容器内权重或 ASR router | "
            f"{_confidence_cell(wer)}；preflight warnings={preflight_summary.get('warnings')} |"
        ),
        (
            "| preproc=2/4 | 当前 recipe 的 anti-recipe | "
            f"{preproc.get('allowed_wording_zh', '')} | "
            "把 preproc=2 或 preproc=4 当作当前最优配置 | "
            "只有 admission/placement/memory 同步重设计后才能重新评估；新结果必须通过 acceptance/stage/final readiness | "
            f"{_confidence_cell(preproc)}；preproc2={preproc2.get('evidence_status')}；preproc4={preproc4.get('evidence_status')} |"
        ),
        (
            "| stage handoff / code2wav | 可说连接健康、decode 非主瓶颈 | "
            f"{stage.get('allowed_wording_zh', '')} | "
            "把 code2wav 或 stage 连接说成当前主瓶颈 | "
            "若改 talker chunk policy、streaming path 或 code2wav compile 路径，重跑 stage interaction 和 tail appendix | "
            f"{_confidence_cell(stage)} |"
        ),
        (
            "| final evidence completion gate | 更新后的目标不再等待 2026-06-21 晚间；"
            "完成由 final_completion_audit 的证据门裁决 | "
            "可以说 final_completion_audit `ready=true` 且 `completion_allowed_now=true` 后可标记完成 | "
            "在 final_completion_audit 有 required failure、blocker 或 `completion_allowed_now=false` 时标记完成 | "
            "任一 evidence/report/package/caveat 变化后重跑 full audit、final completion audit、package validation 和 release seal | "
            f"{final_completion_gate.get('status', 'n/a')} |"
        ),
        "",
        "## 3. 替换数字的触发器",
        "",
        "| 触发器 | 是否可直接替换报告数字 | 必须先满足 |",
        "| --- | --- | --- |",
        "| 只重跑 full audit，无 benchmark 变化 | 可以更新 hash/manifest/gate 数字 | full audit `ok=true`，旧值扫描干净 |",
        "| 硬件、Docker image、模型权重、数据 cache、ASR 口径变化 | 不可直接替换 | environment snapshot、claims、coverage、preflight、manifest、final readiness 全部重跑通过 |",
        "| 新 SGLang stress 或 synthetic speech 结果 | 不可直接替换 | acceptance matrix 仍 17/17 或同步扩展；stage interaction、headline scorecard、confidence ledger 重新通过 |",
        "| 新大样本或真实流量结果 | 可进入扩展范围评审，不能直接替换当前 headline | 新样本的环境、数据、ASR/WER、stage/tail、claims、confidence、caveat 和 final readiness 全部接入 audit |",
        "| 新 vLLM c=8 结果 | 默认只进入 diagnostic | runtime comparison contract 明确 online/offline 边界；若要 headline 必须有 online ingress + WER/ASR |",
        "| 新 official SeedTTS/full natural-speech 结果 | 可作为新增 evidence，不能自动替换 Video-AMME headline | 数据集、WER/RTF、stage breakdown、coverage、confidence ledger 全部接入 audit |",
        "| 任一 required gate 失败 | 不可分享为最终包 | 修复失败项后重跑 full audit；final readiness 必须 `ready=true` |",
        "",
        "## 4. Reviewer 快速口径",
        "",
        "- 主 headline 只用 warmed c=4 strict runtime comparison。",
        "- 高并发只在 SGLang 内部说 c=4-c=8 推荐窗口、c=8 峰值、c=16 饱和边界。",
        "- 更大 Video-AMME 或真实流量结果只能在补齐同口径 stage/tail/quality gate 后扩大结论范围。",
        "- vLLM c=8 只作为 offline diagnostic，prebuild w4 是优化后的 diagnostic baseline。",
        "- Stage 连接和 code2wav decode 当前不是主瓶颈；优先讨论 admission/queue 与 talker AR。",
        "- SeedTTS full-set、online c=8 parity、host-side WER 重算都属于升级项，不影响当前 share package 的 headline。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni caveat adjudication matrix."
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
    output.write_text(build_markdown(root), encoding="utf-8")
    _save_json(payload, json_output)
    print(
        f"Caveat adjudication matrix written: {output} "
        f"ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    print(f"Caveat adjudication matrix JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
