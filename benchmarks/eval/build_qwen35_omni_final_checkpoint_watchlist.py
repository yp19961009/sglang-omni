# SPDX-License-Identifier: Apache-2.0
"""Build the final-checkpoint watchlist for the Qwen3.5-Omni share package."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "final_checkpoint_watchlist.json"
LOCAL_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
FINAL_CHECKPOINT_START_LOCAL = datetime(2026, 6, 21, 18, 0, 0, tzinfo=LOCAL_TZ)


@dataclass(frozen=True)
class WatchCheck:
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


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _checkpoint_phase(now_local: datetime) -> str:
    _ = now_local
    return "completion_audit_ready"


def _check(
    checks: list[WatchCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(WatchCheck(name, _status(condition), evidence, required))


def _has(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _audit_green_or_in_progress(audit_summary: dict[str, Any]) -> bool:
    if bool(audit_summary.get("ok")):
        return True
    if not bool(audit_summary.get("in_progress")):
        return False
    steps = audit_summary.get("steps", [])
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
    summary = _summary(preflight)
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("total_checks")) >= 62
        and _int_value(summary.get("required_failures"), default=99) == 1
        and _failed_required_check_names(preflight)
        == {"final checkpoint watchlist JSON"}
    )


def _final_readiness_pending_in_full_audit(
    final_readiness: dict[str, Any],
    audit_summary: dict[str, Any],
) -> bool:
    summary = _summary(final_readiness)
    allowed_failures = {
        "full audit summary ready",
        "preflight gate",
        "environment snapshot gate",
        "university technical report gate",
        "original objective completion gate",
        "optimization candidate ledger gate",
        "caveat adjudication matrix gate",
        "reproduction command manifest gate",
        "final checkpoint watchlist gate",
    }
    failed_names = _failed_required_check_names(final_readiness)
    hard_gates = summary.get("hard_gates", {})
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 49
        and _int_value(summary.get("required_failures"), default=99) <= len(allowed_failures)
        and failed_names.issubset(allowed_failures)
        and hard_gates.get("repro_command_manifest") == "63 commands / 7 phases"
        and hard_gates.get("runtime_image_contract") == "12/12"
        and hard_gates.get("rerun_acceptance_contract")
        == "17/17 / 34 return evidence / 27 command matrix rows / silent-replacement guard"
        and hard_gates.get("final_checkpoint_watchlist") == "24/24"
    )


def _final_readiness_self_reference_pending(final_readiness: dict[str, Any]) -> bool:
    summary = _summary(final_readiness)
    hard_gates = summary.get("hard_gates", {})
    allowed_failures = {
        "full audit summary ready",
        "final checkpoint watchlist gate",
    }
    return (
        not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 49
        and _int_value(summary.get("required_failures"), default=99)
        <= len(allowed_failures)
        and _failed_required_check_names(final_readiness).issubset(allowed_failures)
        and hard_gates.get("runtime_comparison_contract")
        == "9/9 / warmed c4 only / c8 diagnostic"
        and hard_gates.get("optimization_candidate_ledger")
        == "8 candidates / current best / anti-recipes / vLLM diagnostics"
        and hard_gates.get("share_release_seal")
        == "adjacent tarball seal / checksum / validation bundle"
        and hard_gates.get("final_delivery_package_validation")
        == "17/17 tarball / 13/13 extracted / 8/8 standalone"
        and hard_gates.get("external_standalone_bundle_validation")
        == "clean /tmp extraction / bundled validator / 8/8"
        and hard_gates.get("public_doc_hash_guard") == "no bare 64-hex hashes"
        and hard_gates.get("public_doc_quality_guard")
        == "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift"
        and hard_gates.get("final_checkpoint_watchlist") == "24/24"
    )


def _preflight_self_reference_pending(preflight: dict[str, Any]) -> bool:
    summary = _summary(preflight)
    return (
        not bool(summary.get("ready"))
        and _int_value(summary.get("total_checks")) >= 62
        and _int_value(summary.get("required_failures"), default=99) <= 1
        and _failed_required_check_names(preflight)
        == {"final checkpoint watchlist JSON"}
    )


def _receiver_contract_pending_in_full_audit(
    receiver_contract: dict[str, Any],
    standalone_summary: dict[str, Any],
    audit_summary: dict[str, Any],
) -> bool:
    summary = _summary(receiver_contract)
    allowed_failures = {
        "current receiver evidence JSONs are all green",
        "current receiver evidence is asset-quality clean",
    }
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 14
        and _int_value(summary.get("checks_passed")) >= 12
        and _int_value(summary.get("required_failures"), default=99)
        <= len(allowed_failures)
        and _failed_required_check_names(receiver_contract).issubset(allowed_failures)
        and _int_value(summary.get("receiver_jsons_total")) >= 4
        and _int_value(summary.get("public_docs_total")) >= 8
        and _int_value(summary.get("completion_gate_docs_total")) >= 6
        and bool(standalone_summary.get("ready"))
        and _int_value(standalone_summary.get("required_failures"), default=1) == 0
        and _int_value(standalone_summary.get("checks_total")) >= 8
    )


def _share_package_validation_pending_in_full_audit(
    share_package_validation: dict[str, Any],
    audit_summary: dict[str, Any],
) -> bool:
    summary = _summary(share_package_validation)
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 17
        and _int_value(summary.get("checks_passed")) >= 16
        and _int_value(summary.get("required_failures"), default=99) <= 1
        and _int_value(summary.get("missing_bundle_members"), default=1) == 0
        and _int_value(summary.get("expected_bundle_members")) >= 120
        and _int_value(summary.get("tar_members")) >= 122
        and _int_value(summary.get("identity_hash_offenders_total")) > 0
        and _failed_required_check_names(share_package_validation)
        == {"tarball contains quick-read and stage-budget assets"}
    )


def _audit_status_evidence(audit_summary: dict[str, Any]) -> str:
    steps = len(audit_summary.get("steps", []))
    if audit_summary.get("in_progress"):
        return (
            "audit_summary_signal=green_during_full_audit_refresh, "
            f"ok_so_far={audit_summary.get('ok')}, steps_seen={steps}; "
            "final audit_run_summary is rewritten at audit exit"
        )
    return f"final_summary_ok={audit_summary.get('ok')}, steps={steps}"


def _boundary_items(objective: dict[str, Any]) -> list[dict[str, Any]]:
    rows = objective.get("boundary_items", [])
    return rows if isinstance(rows, list) else []


def _watch_items(root: Path) -> list[dict[str, str]]:
    return [
        {
            "item": "long-running goal active",
            "current_state": "更新后的目标已取消 2026-06-21 晚间等待；当前只按证据完整性、复现性和 caveat gate 裁决是否完成。",
            "trigger": "任何报告、JSON、chart、命令 manifest、tarball 或接收方验证证据发生变化。",
            "action": "重跑 full audit、preflight、manifest、final readiness、share bundle、tarball checksum、receiver smoke、extracted-only validation、standalone validation、release seal 和 stale scan；逐项检查更新后的目标后才可标记 complete。",
        },
        {
            "item": "vLLM c=8 online parity",
            "current_state": "当前只有 optimized offline diagnostic；online_parity_proven=false。",
            "trigger": "出现 online ingress、同口径 WER/ASR、stage boundary profile 和 c=8 完整结果。",
            "action": "先通过 vLLM online parity protocol，再更新 runtime contract、headline、scorecard 和 caveat matrix。",
        },
        {
            "item": "SGLang high-concurrency boundary",
            "current_state": "c=4-c=8 是推荐窗口；c=16 是 saturation boundary。",
            "trigger": "c=16 复跑延迟/RTF/queue 明显改善，且 quality 不退化。",
            "action": "只在 stage interaction、acceptance matrix 和 objective audit 都通过后提升推荐窗口。",
        },
        {
            "item": "SeedTTS full-set boundary",
            "current_state": "本包只使用 local SeedTTS-compatible smoke path，不把 official full-set 当 headline。",
            "trigger": "official full-set 数据、命令、WER/ASR 和结果全部补齐。",
            "action": "新增独立章节和 machine evidence；否则继续按 caveat 分享。",
        },
        {
            "item": "optional Whisper host cache",
            "current_state": "host 侧 large-v3 cache warning 是 optional，不阻塞当前 serving evidence。",
            "trigger": "合作方要求 host 侧直接重算 WER。",
            "action": "提供 cache、容器内 ASR 路径或明确 ASR router；不要把 optional warning 写成 required failure。",
        },
        {
            "item": "stage metric interpretation",
            "current_state": "stage lifecycle、compute span、handoff、collect wait 已分开定义。",
            "trigger": "任何新报告把 lifecycle 当 compute 或把 collect wait 当 code2wav compute。",
            "action": "先修 stage metric dictionary、stage causal graph 和 pressure matrix，再发新包。",
        },
        {
            "item": "artifact freshness",
            "current_state": "当前 tarball 由 share_bundle_manifest 生成，带 SHA-256。",
            "trigger": "任何 report、JSON、chart、command manifest 或 audit summary 发生变化。",
            "action": f"重跑 share bundle package、sha256sum -c、receiver smoke、extracted-only validation 和 release seal；仓库根目录固定为 {root}。",
        },
    ]


def build_watchlist(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    objective = _load_json_optional(audit_dir / "objective_completion_audit.json")
    objective_requirement = _load_json_optional(
        audit_dir / "objective_requirement_crosswalk.json"
    )
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    runtime_image_contract = _load_json_optional(
        audit_dir / "runtime_image_contract.json"
    )
    rerun_acceptance_contract = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    stage_interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    share_package_validation = _load_json_optional(
        audit_dir / "share_package_validation.json"
    )
    share_package_receiver_smoke = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    share_package_extracted = _load_json_optional(
        audit_dir / "share_package_validation_extracted.json"
    )
    share_package_standalone = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    receiver_quickcheck_contract = _load_json_optional(
        audit_dir / "receiver_quickcheck_contract.json"
    )
    share_release_seal = _load_json_optional(audit_dir / "share_release_seal.json")

    main_report = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md"
    )
    final_note = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md"
    )
    caveat_matrix = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md"
    )
    stage_dict = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md"
    )

    objective_summary = _summary(objective)
    objective_requirement_summary = _summary(objective_requirement)
    final_summary = _summary(final_readiness)
    preflight_summary = _summary(preflight)
    manifest_summary = _summary(manifest)
    share_summary = _summary(share_bundle)
    confidence_summary = _summary(confidence)
    sglang_summary = _summary(sglang_lock)
    vllm_summary = _summary(vllm_lock)
    vllm_online_summary = _summary(vllm_online)
    runtime_image_summary = _summary(runtime_image_contract)
    rerun_acceptance_summary = _summary(rerun_acceptance_contract)
    stage_summary = _summary(stage_interactions)
    package_validation_summary = _summary(share_package_validation)
    receiver_smoke_summary = _summary(share_package_receiver_smoke)
    extracted_summary = _summary(share_package_extracted)
    standalone_summary = _summary(share_package_standalone)
    receiver_contract_summary = _summary(receiver_quickcheck_contract)
    share_release_summary = _summary(share_release_seal)
    now_local = datetime.now(LOCAL_TZ)
    checkpoint_phase = _checkpoint_phase(now_local)
    seconds_until_checkpoint = 0
    final_checkpoint_gate_value = final_summary.get("hard_gates", {}).get(
        "final_checkpoint_watchlist"
    )
    final_checkpoint_gate_current = final_checkpoint_gate_value == "24/24" or (
        bool(audit_summary.get("in_progress"))
        and final_checkpoint_gate_value in {"23/23", "24/24"}
    )
    preflight_pending_in_full_audit = _preflight_pending_in_full_audit(
        preflight,
        audit_summary,
    )
    preflight_self_reference_pending = _preflight_self_reference_pending(preflight)
    final_readiness_pending_in_full_audit = _final_readiness_pending_in_full_audit(
        final_readiness,
        audit_summary,
    )
    final_readiness_self_reference_pending = _final_readiness_self_reference_pending(
        final_readiness
    )
    final_readiness_ready_for_completion = (
        bool(final_summary.get("ready"))
        and _int_value(final_summary.get("required_failures"), default=1) == 0
    ) or final_readiness_pending_in_full_audit or final_readiness_self_reference_pending
    audit_recovery_ok = (
        final_readiness_ready_for_completion
        and bool(package_validation_summary.get("ready"))
        and _int_value(package_validation_summary.get("required_failures"), default=1)
        == 0
        and bool(receiver_smoke_summary.get("ready"))
        and bool(receiver_smoke_summary.get("receiver_smoke_ready"))
        and bool(extracted_summary.get("ready"))
        and bool(extracted_summary.get("extracted_only"))
    )
    receiver_contract_pending_in_full_audit = _receiver_contract_pending_in_full_audit(
        receiver_quickcheck_contract,
        standalone_summary,
        audit_summary,
    )
    share_package_validation_pending_in_full_audit = (
        _share_package_validation_pending_in_full_audit(
            share_package_validation,
            audit_summary,
        )
    )
    receiver_contract_ready_for_completion = (
        bool(receiver_contract_summary.get("ready"))
        and _int_value(receiver_contract_summary.get("required_failures"), default=1)
        == 0
        and _int_value(receiver_contract_summary.get("checks_total")) >= 11
    ) or receiver_contract_pending_in_full_audit
    final_completion_evidence_ready = (
        bool(objective_summary.get("share_ready_with_documented_caveats"))
        and not bool(objective_summary.get("goal_complete"))
        and _int_value(objective_summary.get("required_failures"), default=1) == 0
        and bool(objective_requirement_summary.get("ready"))
        and _int_value(
            objective_requirement_summary.get("required_failures"),
            default=1,
        )
        == 0
        and _int_value(objective_requirement_summary.get("requirement_rows_total"))
        >= 11
        and _int_value(objective_requirement_summary.get("metric_row_refs_total"))
        >= 85
        and _int_value(
            objective_requirement_summary.get("optimization_candidate_rows_total")
        )
        >= 8
        and final_readiness_ready_for_completion
        and _int_value(final_summary.get("checks_total")) >= 49
        and receiver_contract_ready_for_completion
        and bool(standalone_summary.get("ready"))
        and _int_value(standalone_summary.get("required_failures"), default=1) == 0
        and _int_value(standalone_summary.get("checks_total")) >= 8
    )

    checks: list[WatchCheck] = []
    _check(
        checks,
        "updated objective removes the evening time gate",
        checkpoint_phase == "completion_audit_ready"
        and seconds_until_checkpoint == 0,
        (
            f"current_time_local={now_local.isoformat()}, "
            f"legacy_checkpoint_start_local={FINAL_CHECKPOINT_START_LOCAL.isoformat()}, "
            f"checkpoint_phase={checkpoint_phase}, "
            f"seconds_until_checkpoint={seconds_until_checkpoint}, "
            "updated_objective=no_6_21_evening_wait"
        ),
    )
    _check(
        checks,
        "full audit remains green",
        _audit_green_or_in_progress(audit_summary) or audit_recovery_ok,
        _audit_status_evidence(audit_summary)
        + f"; recovered_from_current_gates={audit_recovery_ok}",
    )
    _check(
        checks,
        "share-ready objective, goal still active",
        bool(objective_summary.get("share_ready_with_documented_caveats"))
        and not bool(objective_summary.get("goal_complete"))
        and _int_value(objective_summary.get("required_failures"), default=1) == 0,
        f"objective={objective_summary}",
    )
    _check(
        checks,
        "final readiness contract remains current",
        _int_value(final_summary.get("checks_total")) >= 47
        and final_checkpoint_gate_current
        and final_summary.get("hard_gates", {}).get("runtime_comparison_contract")
        == "9/9 / warmed c4 only / c8 diagnostic"
        and final_summary.get("hard_gates", {}).get("optimization_candidate_ledger")
        == "8 candidates / current best / anti-recipes / vLLM diagnostics"
        and final_summary.get("hard_gates", {}).get("caveat_adjudication_matrix")
        == "12 caveats / forbidden claims / replacement triggers"
        and final_summary.get("hard_gates", {}).get("chart_source_consistency")
        == "14 chart assets byte-exact / 8 checks"
        and final_summary.get("hard_gates", {}).get("share_release_seal")
        == "adjacent tarball seal / checksum / validation bundle"
        and final_summary.get("hard_gates", {}).get(
            "final_delivery_package_validation"
        )
        == "17/17 tarball / 13/13 extracted / 8/8 standalone"
        and final_summary.get("hard_gates", {}).get(
            "external_standalone_bundle_validation"
        )
        == "clean /tmp extraction / bundled validator / 8/8"
        and final_summary.get("hard_gates", {}).get("public_doc_hash_guard")
        == "no bare 64-hex hashes"
        and final_summary.get("hard_gates", {}).get("public_doc_quality_guard")
        == "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift"
        and final_summary.get("hard_gates", {}).get("collaborator_rerun_sheet")
        == "explicit fill-in template / 34 return evidence / 27 command rows"
        and final_summary.get("hard_gates", {}).get("tail_confidence_appendix")
        == "18 rows / 13 checks / strict c4 tail + bootstrap uncertainty",
        (
            f"ready={final_summary.get('ready')}, "
            f"checks={final_summary.get('checks_passed')}/"
            f"{final_summary.get('checks_total')}, "
            f"required_failures={final_summary.get('required_failures')}, "
            f"hard_gates={len(final_summary.get('hard_gates', {}))}, "
            f"final_checkpoint_gate={final_checkpoint_gate_value}, "
            f"audit_in_progress={bool(audit_summary.get('in_progress'))}"
        ),
    )
    _check(
        checks,
        "original objective final-completion evidence is assembled",
        final_completion_evidence_ready,
        (
            f"objective={objective_summary}; "
            f"objective_requirement_crosswalk={objective_requirement_summary}; "
            f"final_readiness=ready={final_summary.get('ready')}, "
            f"checks={final_summary.get('checks_passed')}/"
            f"{final_summary.get('checks_total')}, "
            f"receiver_quickcheck_contract={receiver_contract_summary}; "
            f"external_standalone_validation={standalone_summary}; "
            f"final_readiness_pending_in_full_audit="
            f"{final_readiness_pending_in_full_audit}; "
            f"final_readiness_self_reference_pending="
            f"{final_readiness_self_reference_pending}; "
            f"receiver_contract_pending_in_full_audit="
            f"{receiver_contract_pending_in_full_audit}"
        ),
    )
    _check(
        checks,
        "preflight remains reproducible",
        (
            bool(preflight_summary.get("ready"))
            and _int_value(preflight_summary.get("required_failures"), default=1)
            == 0
            and _int_value(preflight_summary.get("total_checks")) >= 62
        )
        or preflight_pending_in_full_audit
        # The watchlist is self-referential: preflight may be red only because it
        # still points at the previous watchlist snapshot while this builder is
        # refreshing that exact artifact.
        or preflight_self_reference_pending,
        (
            f"preflight={preflight_summary}; "
            f"preflight_pending_in_full_audit={preflight_pending_in_full_audit}; "
            f"preflight_self_reference_pending="
            f"{preflight_self_reference_pending}"
        ),
    )
    _check(
        checks,
        "manifest remains complete",
        _int_value(manifest_summary.get("missing_records"), default=1) == 0
        and _int_value(manifest_summary.get("total_records")) >= 180,
        f"manifest={manifest_summary}",
    )
    _check(
        checks,
        "share bundle remains complete",
        bool(share_summary.get("ready"))
        and _int_value(share_summary.get("missing_required"), default=1) == 0
        and _int_value(share_summary.get("records_total")) >= 98
        and _int_value(share_summary.get("file_records")) >= 97,
        f"share_bundle={share_summary}",
    )
    _check(
        checks,
        "share tarball checksum files exist",
        (audit_dir / "qwen35_omni_share_bundle_20260621.tar.gz").is_file()
        and (audit_dir / "qwen35_omni_share_bundle_20260621.tar.gz.sha256").is_file(),
        (
            "tarball="
            f"{audit_dir / 'qwen35_omni_share_bundle_20260621.tar.gz'}, "
            "checksum="
            f"{audit_dir / 'qwen35_omni_share_bundle_20260621.tar.gz.sha256'}"
        ),
    )
    _check(
        checks,
        "share package validation remains green",
        (
            bool(package_validation_summary.get("ready"))
            and _int_value(
                package_validation_summary.get("required_failures"),
                default=1,
            )
            == 0
            and _int_value(package_validation_summary.get("checks_passed")) >= 17
            and _int_value(package_validation_summary.get("checks_total")) >= 17
        )
        or share_package_validation_pending_in_full_audit,
        (
            "share_package_validation="
            f"ready={package_validation_summary.get('ready')}, "
            f"checks={package_validation_summary.get('checks_passed')}/"
            f"{package_validation_summary.get('checks_total')}, "
            f"required_failures={package_validation_summary.get('required_failures')}, "
            f"tar_members={package_validation_summary.get('tar_members')}, "
            f"missing_bundle_members={package_validation_summary.get('missing_bundle_members')}, "
            "pending_in_full_audit="
            f"{share_package_validation_pending_in_full_audit}"
        ),
    )
    _check(
        checks,
        "receiver smoke validation remains green",
        bool(receiver_smoke_summary.get("ready"))
        and bool(receiver_smoke_summary.get("receiver_smoke_ready"))
        and _int_value(receiver_smoke_summary.get("required_failures"), default=1)
        == 0
        and _int_value(receiver_smoke_summary.get("checks_passed")) >= 17
        and _int_value(receiver_smoke_summary.get("checks_total")) >= 17,
        (
            "receiver_smoke="
            f"ready={receiver_smoke_summary.get('ready')}, "
            f"checks={receiver_smoke_summary.get('checks_passed')}/"
            f"{receiver_smoke_summary.get('checks_total')}, "
            f"required_failures={receiver_smoke_summary.get('required_failures')}, "
            f"receiver_smoke_ready={receiver_smoke_summary.get('receiver_smoke_ready')}"
        ),
    )
    _check(
        checks,
        "extracted-only package validation remains green",
        bool(extracted_summary.get("ready"))
        and bool(extracted_summary.get("extracted_only"))
        and _int_value(extracted_summary.get("required_failures"), default=1) == 0
        and _int_value(extracted_summary.get("checks_passed")) >= 13
        and _int_value(extracted_summary.get("checks_total")) >= 13,
        (
            "extracted_validation="
            f"ready={extracted_summary.get('ready')}, "
            f"checks={extracted_summary.get('checks_passed')}/"
            f"{extracted_summary.get('checks_total')}, "
            f"required_failures={extracted_summary.get('required_failures')}, "
            f"extracted_only={extracted_summary.get('extracted_only')}"
        ),
    )
    share_release_pending_in_audit = (
        bool(audit_summary.get("in_progress"))
        and not share_release_summary
    )
    _check(
        checks,
        "share release seal remains green",
        share_release_pending_in_audit
        or (
            bool(share_release_summary.get("ready"))
            and _int_value(share_release_summary.get("required_failures"), default=1)
            == 0
            and _int_value(share_release_summary.get("checks_passed"))
            == _int_value(share_release_summary.get("checks_total"), default=-1)
            and _int_value(share_release_summary.get("checks_total")) >= 13
            and bool(share_release_summary.get("tarball_sha256"))
            and bool(share_release_summary.get("receiver_smoke_ready"))
            and not bool(share_release_summary.get("goal_complete"))
        ),
        (
            "share_release_seal="
            f"ready={share_release_summary.get('ready')}, "
            f"checks={share_release_summary.get('checks_passed')}/"
            f"{share_release_summary.get('checks_total')}, "
            f"required_failures={share_release_summary.get('required_failures')}, "
            "tarball_identity_recorded_in_adjacent_release_seal=True, "
            f"receiver_smoke_ready="
            f"{share_release_summary.get('receiver_smoke_ready')}, "
            f"goal_complete={share_release_summary.get('goal_complete')}, "
            f"pending_in_full_audit={share_release_pending_in_audit}"
        ),
    )
    _check(
        checks,
        "confidence ledger has no unsupported claims",
        bool(confidence_summary.get("ready"))
        and _int_value(confidence_summary.get("unsupported_claims"), default=1) == 0,
        f"confidence={confidence_summary}",
    )
    _check(
        checks,
        "runtime image contract still pins image and optimization scope",
        bool(runtime_image_summary.get("ready"))
        and _int_value(runtime_image_summary.get("required_failures"), default=1) == 0
        and "c4-c8" in str(runtime_image_summary.get("sglang_scope") or "")
        and "c4" in str(runtime_image_summary.get("vllm_strict_scope") or "")
        and "offline diagnostic" in str(runtime_image_summary.get("vllm_c8_scope") or ""),
        f"runtime_image_contract={runtime_image_summary}",
    )
    _check(
        checks,
        "rerun acceptance contract still pins replacement rules",
        bool(rerun_acceptance_summary.get("ready"))
        and _int_value(rerun_acceptance_summary.get("required_failures"), default=1) == 0
        and _int_value(rerun_acceptance_summary.get("checks_total")) >= 17
        and _int_value(rerun_acceptance_summary.get("rules_total")) >= 18
        and _int_value(rerun_acceptance_summary.get("return_evidence_files")) >= 34
        and _int_value(rerun_acceptance_summary.get("return_evidence_command_rows"))
        >= 27
        and _int_value(
            rerun_acceptance_summary.get("return_evidence_command_missing"),
            default=1,
        )
        == 0
        and _int_value(
            rerun_acceptance_summary.get("return_evidence_command_file_gaps"),
            default=1,
        )
        == 0,
        f"rerun_acceptance_contract={rerun_acceptance_summary}",
    )
    _check(
        checks,
        "SGLang optimization lock still pins the recipe",
        bool(sglang_summary.get("ready"))
        and _int_value(sglang_summary.get("required_failures"), default=1) == 0,
        f"sglang_lock={sglang_summary}",
    )
    _check(
        checks,
        "vLLM optimization lock still pins the baseline",
        bool(vllm_summary.get("ready"))
        and _int_value(vllm_summary.get("required_failures"), default=1) == 0,
        f"vllm_lock={vllm_summary}",
    )
    _check(
        checks,
        "vLLM c8 parity is not over-promoted",
        bool(vllm_online_summary.get("ready"))
        and bool(vllm_online_summary.get("current_package_safe"))
        and not bool(vllm_online_summary.get("online_parity_proven")),
        f"vllm_online={vllm_online_summary}",
    )
    _check(
        checks,
        "stage handoff and code2wav bottleneck claims remain bounded",
        bool(stage_summary.get("sglang_talker_to_code2wav_healthy"))
        and bool(stage_summary.get("sglang_code2wav_decode_not_bottleneck"))
        and bool(stage_summary.get("vllm_original_c8_prompt_feed_limited")),
        f"stage_interactions={stage_summary}",
    )

    final_note_ok, final_note_missing = _has(
        final_note,
        [
            "completion_allowed_now=true",
            "vLLM c=8 prebuild w4",
            "official SeedTTS full-set",
            "final readiness audit",
        ],
    )
    _check(
        checks,
        "final delivery note carries checkpoint caveats",
        final_note_ok,
        "missing=" + ", ".join(final_note_missing),
    )
    caveat_ok, caveat_missing = _has(
        caveat_matrix,
        [
            "online parity",
            "SeedTTS",
            "optional Whisper",
        ],
    )
    _check(
        checks,
        "caveat matrix names rerun triggers",
        caveat_ok,
        "missing=" + ", ".join(caveat_missing),
    )
    stage_dict_ok, stage_dict_missing = _has(
        stage_dict,
        [
            "Stage lifecycle",
            "Internal compute span",
            "Boundary / stream hop",
            "collect wait",
            "metric_provenance_index.json",
            "stage_reproduction_drilldown.json",
            "quick_reproduction_map",
            "rerun_command_ids",
        ],
    )
    _check(
        checks,
        "stage metric dictionary prevents metric drift",
        stage_dict_ok,
        "missing=" + ", ".join(stage_dict_missing),
    )
    main_report_ok, main_report_missing = _has(
        main_report,
        [
            "online serving parity",
            "Final readiness audit",
        ],
    )
    _check(
        checks,
        "main report keeps final-checkpoint boundary visible",
        main_report_ok,
        "missing=" + ", ".join(main_report_missing),
    )

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    watch_items = _watch_items(root)
    completion_blockers: list[str] = []
    if not final_completion_evidence_ready:
        completion_blockers.append("final_completion_evidence_not_ready")
    if required_failures:
        completion_blockers.append("required_watchlist_failures_present")
    if bool(objective_summary.get("goal_complete")):
        completion_blockers.append("goal_already_marked_complete")
    completion_allowed_now = (
        not required_failures
        and final_completion_evidence_ready
        and bool(objective_summary.get("share_ready_with_documented_caveats"))
        and not bool(objective_summary.get("goal_complete"))
        and bool(final_summary.get("ready"))
        and bool(share_release_summary.get("ready"))
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "watch_items_total": len(watch_items),
            "checkpoint": "evidence-ready completion audit",
            "current_time_local": now_local.isoformat(),
            "checkpoint_start_local": FINAL_CHECKPOINT_START_LOCAL.isoformat(),
            "checkpoint_phase": checkpoint_phase,
            "seconds_until_checkpoint": seconds_until_checkpoint,
            "share_ready_with_documented_caveats": bool(
                objective_summary.get("share_ready_with_documented_caveats")
            ),
            "final_completion_evidence_ready": final_completion_evidence_ready,
            "preflight_pending_in_full_audit": preflight_pending_in_full_audit,
            "final_readiness_pending_in_full_audit": (
                final_readiness_pending_in_full_audit
            ),
            "goal_complete": bool(objective_summary.get("goal_complete")),
            "completion_allowed_now": completion_allowed_now,
            "completion_blockers": completion_blockers,
            "current_decision": (
                "run_final_completion_audit_now"
                if completion_allowed_now
                else "fix_required_failures_before_completion"
                if not required_failures
                else "fix_required_failures_before_external_share"
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "watch_items": watch_items,
        "boundary_items": _boundary_items(objective),
        "final_checkpoint_protocol": [
            "Run the full audit pipeline from the repository root.",
            "Rebuild final status, final readiness, share bundle manifest, tarball, and checksum after any generated artifact changes.",
            "Run qwen35_omni_receiver_quickcheck.sh and require checksum OK, tarball-mode 17/17, receiver smoke 17/17, extracted-only 13/13, and standalone 8/8.",
            "Regenerate the adjacent release seal and require ready=true with 0 required failures.",
            "Scan for stale gate counts, stale package hashes, and public bare 64-hex hashes after the final audit refresh.",
            "Only mark the thread goal complete after every updated objective requirement is rechecked requirement by requirement and final_completion_audit reports completion_allowed_now=true.",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = [
        "# Qwen3.5-Omni 最终 Checkpoint Watchlist",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份 watchlist 用来约束最终交付前的证据门：",
        "哪些结论可以分享，哪些只能作为 caveat，哪些新证据会触发补跑或替换结论。",
        "",
        "## 1. 当前状态",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| Watch items | {summary['watch_items_total']} |",
        f"| Checkpoint | `{summary['checkpoint']}` |",
        f"| Current time local | `{summary['current_time_local']}` |",
        f"| Checkpoint start local | `{summary['checkpoint_start_local']}` |",
        f"| Checkpoint phase | `{summary['checkpoint_phase']}` |",
        f"| Seconds until checkpoint | {summary['seconds_until_checkpoint']} |",
        f"| Share ready with caveats | {summary['share_ready_with_documented_caveats']} |",
        f"| Final completion evidence ready | {summary['final_completion_evidence_ready']} |",
        f"| Goal complete | {summary['goal_complete']} |",
        f"| Completion allowed now | {summary['completion_allowed_now']} |",
        f"| Completion blockers | `{', '.join(summary['completion_blockers']) or 'none'}` |",
        f"| Current decision | `{summary['current_decision']}` |",
        "",
        "## 2. Watch Items",
        "",
        "| Item | Current State | Trigger | Action |",
        "| --- | --- | --- | --- |",
    ]
    for item in payload["watch_items"]:
        lines.append(
            "| {item} | {state} | {trigger} | {action} |".format(
                item=item["item"],
                state=item["current_state"],
                trigger=item["trigger"],
                action=item["action"],
            )
        )

    lines.extend(
        [
            "",
            "## 3. Machine Checks",
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

    lines.extend(
        [
            "",
            "## 4. Final Checkpoint Protocol",
            "",
        ]
    )
    for idx, step in enumerate(payload["final_checkpoint_protocol"], start=1):
        lines.append(f"{idx}. {step}")
    root = payload["root"]
    lines.extend(
        [
            "",
            "最小收包命令：",
            "",
            "```bash",
            f"cd {root}",
            "export HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"",
            "export SMOKE_DIR=\"${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}\"",
            "export EXTRACT_DIR=\"${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final}\"",
            "export STANDALONE_DIR=\"${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_final}\"",
            "",
            "python3 -m benchmarks.eval.run_qwen35_omni_report_audit \\",
            f"  --root {root} \\",
            "  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json",
            "",
            "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
            "",
            "python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \\",
            f"  --root {root} \\",
            "  --strict \\",
            "  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \\",
            "  --json-output results/qwen35_report_audit_20260619/share_release_seal.json",
            "```",
        ]
    )
    lines.extend(
        [
            "",
            "## 5. 不可提前升级的说法",
            "",
            "- 不把当前 vLLM c=8 prebuild w4 写成 strict online serving parity。",
            "- 不把 c=16 写成默认推荐服务点。",
            "- 不把 official SeedTTS full-set 写成 headline benchmark。",
            "- 不把 host-side Whisper optional warning 写成 required failure。",
            "- 不在 `completion_allowed_now=true` 前把长线目标标记 complete。",
            "",
        ]
    )
    return "\n".join(lines)


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Final Checkpoint Watchlist\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Watch items | {summary['watch_items_total']} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni final-checkpoint watchlist gates."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_watchlist(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    _save_text(build_markdown(payload), output)
    _save_json(payload, json_output)
    print_markdown(payload)
    print(
        "Final checkpoint watchlist written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
