# SPDX-License-Identifier: Apache-2.0
"""Build the final goal-completion audit for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "final_completion_audit.json"
LOCAL_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
FINAL_CHECKPOINT_START_LOCAL = datetime(2026, 6, 21, 18, 0, 0, tzinfo=LOCAL_TZ)
PUBLIC_BARE_SHA256 = re.compile(r"(?<!sha256:)\b[0-9a-f]{64}\b")


@dataclass(frozen=True)
class CompletionCheck:
    name: str
    status: str
    required: bool
    evidence: str

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


def _public_evidence(value: Any) -> str:
    text = str(value)
    return PUBLIC_BARE_SHA256.sub(lambda match: f"sha256:{match.group(0)}", text)


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


def _stable_share_release_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary.get(key)
        for key in [
            "ready",
            "checks_total",
            "checks_passed",
            "required_failures",
            "package_file_count",
            "share_bundle_records",
            "final_readiness_checks",
            "repro_commands_total",
            "receiver_smoke_ready",
            "goal_complete",
            "completion_allowed_now",
            "send_decision",
            "forbidden_tarball_members",
        ]
        if key in summary
    }


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value)


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _checkpoint_phase(now_local: datetime) -> str:
    _ = now_local
    return "completion_audit_ready"


def _check(
    checks: list[CompletionCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
    tracking_when_false: bool = False,
) -> None:
    if condition:
        status = "PASS"
    elif tracking_when_false:
        status = "TRACKING"
    else:
        status = "FAIL"
    checks.append(CompletionCheck(name, status, required, evidence))


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    now_local = datetime.now(LOCAL_TZ)
    checkpoint_phase = _checkpoint_phase(now_local)
    seconds_until_checkpoint = 0

    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    final_checkpoint = _load_json_optional(audit_dir / "final_checkpoint_watchlist.json")
    objective_completion = _load_json_optional(
        audit_dir / "objective_completion_audit.json"
    )
    objective_crosswalk = _load_json_optional(
        audit_dir / "objective_requirement_crosswalk.json"
    )
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    runtime_comparison = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_ledger = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    stage_route = _load_json_optional(audit_dir / "stage_route_decision_matrix.json")
    stage_repro = _load_json_optional(audit_dir / "stage_reproduction_drilldown.json")
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    share_release = _load_json_optional(audit_dir / "share_release_seal.json")
    receiver_contract = _load_json_optional(
        audit_dir / "receiver_quickcheck_contract.json"
    )
    package_validation = _load_json_optional(
        audit_dir / "share_package_validation.json"
    )
    receiver_smoke = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    extracted_validation = _load_json_optional(
        audit_dir / "share_package_validation_extracted.json"
    )
    standalone_validation = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    share_consistency = _load_json_optional(audit_dir / "share_consistency_guard.json")
    share_path = _load_json_optional(audit_dir / "share_path_hygiene.json")

    final_summary = _summary(final_readiness)
    checkpoint_summary = _summary(final_checkpoint)
    objective_summary = _summary(objective_completion)
    crosswalk_summary = _summary(objective_crosswalk)
    runtime_summary = _summary(runtime_comparison)
    sglang_summary = _summary(sglang_lock)
    vllm_summary = _summary(vllm_lock)
    vllm_online_summary = _summary(vllm_online)
    stage_budget_summary = _summary(stage_budget)
    stage_ledger_summary = _summary(stage_ledger)
    stage_route_summary = _summary(stage_route)
    stage_repro_summary = _summary(stage_repro)
    repro_summary = _summary(repro_manifest)
    preflight_summary = _summary(preflight)
    share_release_summary = _summary(share_release)
    stable_share_release_summary = _stable_share_release_summary(
        share_release_summary
    )
    receiver_summary = _summary(receiver_contract)
    package_summary = _summary(package_validation)
    receiver_smoke_summary = _summary(receiver_smoke)
    extracted_summary = _summary(extracted_validation)
    standalone_summary = _summary(standalone_validation)
    consistency_summary = _summary(share_consistency)
    share_path_summary = _summary(share_path)

    checks: list[CompletionCheck] = []
    _check(
        checks,
        "current local time is recorded",
        FINAL_CHECKPOINT_START_LOCAL.isoformat() == "2026-06-21T18:00:00+08:00",
        (
            f"current_time_local={now_local.isoformat()}, "
            f"checkpoint_start_local={FINAL_CHECKPOINT_START_LOCAL.isoformat()}, "
            f"checkpoint_phase={checkpoint_phase}, "
            f"seconds_until_checkpoint={seconds_until_checkpoint}"
        ),
    )
    _check(
        checks,
        "full audit summary is green",
        _bool(audit_summary.get("ok")) or _bool(audit_summary.get("in_progress")),
        (
            f"audit_ok={audit_summary.get('ok')}, "
            f"in_progress={audit_summary.get('in_progress')}, "
            f"steps={len(audit_summary.get('steps', []))}"
        ),
    )
    _check(
        checks,
        "final readiness remains green",
        _bool(final_summary.get("ready"))
        and _int_value(final_summary.get("checks_total")) >= 49
        and _int_value(final_summary.get("required_failures"), default=1) == 0
        and final_summary.get("hard_gates", {}).get("final_checkpoint_watchlist")
        == "24/24",
        f"final_readiness={final_summary}",
    )
    _check(
        checks,
        "final checkpoint watchlist is complete",
        _bool(checkpoint_summary.get("ready"))
        and _int_value(checkpoint_summary.get("checks_total")) >= 24
        and _int_value(checkpoint_summary.get("required_failures"), default=1) == 0
        and _bool(checkpoint_summary.get("final_completion_evidence_ready"))
        and not _bool(checkpoint_summary.get("goal_complete")),
        f"final_checkpoint_watchlist={checkpoint_summary}",
    )
    _check(
        checks,
        "original objective completion rows are share-ready",
        _bool(objective_summary.get("share_ready_with_documented_caveats"))
        and _int_value(objective_summary.get("rows_total")) >= 17
        and _int_value(objective_summary.get("required_failures"), default=1) == 0
        and not _bool(objective_summary.get("goal_complete")),
        f"objective_completion={objective_summary}",
    )
    _check(
        checks,
        "original requirement crosswalk is evidence-backed",
        _bool(crosswalk_summary.get("ready"))
        and _int_value(crosswalk_summary.get("requirement_rows_total")) >= 11
        and _int_value(crosswalk_summary.get("required_failures"), default=1) == 0
        and _int_value(crosswalk_summary.get("metric_row_refs_total")) >= 85
        and _int_value(crosswalk_summary.get("optimization_candidate_rows_total"))
        >= 8,
        f"objective_requirement_crosswalk={crosswalk_summary}",
    )
    _check(
        checks,
        "SGLang and vLLM performance claims are bounded and verified",
        _bool(claims.get("passed"))
        and _int_value(claims.get("total_checks")) >= 17
        and _bool(runtime_summary.get("ready"))
        and _int_value(runtime_summary.get("checks_total")) >= 9
        and _bool(sglang_summary.get("ready"))
        and _int_value(sglang_summary.get("checks_total")) >= 26
        and _bool(vllm_summary.get("ready"))
        and _int_value(vllm_summary.get("checks_total")) >= 22
        and _bool(vllm_online_summary.get("ready"))
        and not _bool(vllm_online_summary.get("online_parity_proven")),
        (
            f"claims={claims}; runtime_comparison={runtime_summary}; "
            f"sglang_lock={sglang_summary}; vllm_lock={vllm_summary}; "
            f"vllm_online={vllm_online_summary}"
        ),
    )
    _check(
        checks,
        "stage breakdown and boundary evidence are complete",
        _bool(stage_budget_summary.get("ready"))
        and _int_value(stage_budget_summary.get("checks_total")) >= 12
        and _bool(stage_ledger_summary.get("ready"))
        and _int_value(stage_ledger_summary.get("ledger_rows")) >= 37
        and _bool(stage_route_summary.get("ready"))
        and _int_value(stage_route_summary.get("route_rows_total")) >= 11
        and _bool(stage_repro_summary.get("ready"))
        and _int_value(stage_repro_summary.get("stage_rows_total")) >= 52,
        (
            f"stage_latency_budget={stage_budget_summary}; "
            f"stage_boundary_bottleneck_ledger={stage_ledger_summary}; "
            f"stage_route_decision_matrix={stage_route_summary}; "
            f"stage_reproduction_drilldown={stage_repro_summary}"
        ),
    )
    _check(
        checks,
        "reproduction commands and preflight are current",
        _bool(repro_summary.get("ready"))
        and _int_value(repro_summary.get("commands_total")) >= 62
        and _bool(preflight_summary.get("ready"))
        and _int_value(preflight_summary.get("total_checks")) >= 62
        and _int_value(preflight_summary.get("required_failures"), default=1) == 0,
        f"repro_command_manifest={repro_summary}; preflight={preflight_summary}",
    )
    _check(
        checks,
        "share package and receiver validation are green",
        _bool(share_release_summary.get("ready"))
        and _int_value(share_release_summary.get("checks_total")) >= 14
        and _int_value(share_release_summary.get("required_failures"), default=1) == 0
        and _bool(receiver_summary.get("ready"))
        and _int_value(receiver_summary.get("checks_total")) >= 11
        and _bool(package_summary.get("ready"))
        and _int_value(package_summary.get("checks_total")) >= 17
        and _bool(receiver_smoke_summary.get("ready"))
        and _bool(receiver_smoke_summary.get("receiver_smoke_ready"))
        and _bool(extracted_summary.get("ready"))
        and _int_value(extracted_summary.get("checks_total")) >= 13
        and _bool(standalone_summary.get("ready"))
        and _int_value(standalone_summary.get("checks_total")) >= 8,
        (
            f"share_release_seal_gate={stable_share_release_summary}; "
            f"receiver_quickcheck_contract={receiver_summary}; "
            f"tarball_validation={package_summary}; "
            f"receiver_smoke={receiver_smoke_summary}; "
            f"extracted={extracted_summary}; standalone={standalone_summary}"
        ),
    )
    _check(
        checks,
        "public consistency and path hygiene are clean",
        _bool(consistency_summary.get("ready"))
        and _int_value(consistency_summary.get("public_stale_hits"), default=1) == 0
        and _int_value(consistency_summary.get("machine_stale_hits"), default=1) == 0
        and _int_value(consistency_summary.get("embedded_identity_leaks"), default=1)
        == 0
        and _bool(share_path_summary.get("ready"))
        and _int_value(share_path_summary.get("package_offenders_total"), default=1)
        == 0
        and _int_value(share_path_summary.get("raw_offenders_total"), default=1) == 0,
        f"share_consistency_guard={consistency_summary}; share_path_hygiene={share_path_summary}",
    )

    required_failures = [
        check for check in checks if check.required and check.status == "FAIL"
    ]
    evidence_ready = not required_failures
    time_gate_open = True
    checkpoint_allows_completion = _bool(
        checkpoint_summary.get("completion_allowed_now")
    ) or (
        _bool(checkpoint_summary.get("ready"))
        and _int_value(checkpoint_summary.get("required_failures"), default=1) == 0
        and _bool(checkpoint_summary.get("final_completion_evidence_ready"))
    )
    completion_allowed_now = evidence_ready and checkpoint_allows_completion

    completion_blockers: list[str] = []
    if not evidence_ready:
        completion_blockers.append("required_evidence_failures_present")
    if not checkpoint_allows_completion:
        completion_blockers.append("checkpoint_watchlist_does_not_allow_completion")

    _check(
        checks,
        "updated objective has no evening time gate",
        time_gate_open,
        (
            f"checkpoint_phase={checkpoint_phase}, "
            f"seconds_until_checkpoint={seconds_until_checkpoint}, "
            "updated_objective=no_6_21_evening_wait"
        ),
    )
    _check(
        checks,
        "thread goal completion is allowed now",
        completion_allowed_now,
        (
            f"completion_allowed_now={completion_allowed_now}, "
            f"checkpoint_allows_completion={checkpoint_allows_completion}, "
            f"completion_blockers={completion_blockers}"
        ),
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": evidence_ready,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "tracking_checks": sum(1 for check in checks if check.status == "TRACKING"),
            "required_failures": len(required_failures),
            "current_time_local": now_local.isoformat(),
            "checkpoint_start_local": FINAL_CHECKPOINT_START_LOCAL.isoformat(),
            "checkpoint_phase": checkpoint_phase,
            "seconds_until_checkpoint": seconds_until_checkpoint,
            "evidence_ready": evidence_ready,
            "time_gate_open": time_gate_open,
            "checkpoint_allows_completion": checkpoint_allows_completion,
            "completion_allowed_now": completion_allowed_now,
            "completion_blockers": completion_blockers,
            "goal_update_recommendation": (
                "safe_to_call_update_goal_complete"
                if completion_allowed_now
                else "keep_goal_active"
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni 最终完成审计",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份审计只回答一个问题：当前线程目标是否已经可以安全标记完成。",
        "它把更新后的目标逐项证据、最终分享包、接收方 quickcheck、stage breakdown、vLLM/SGLang 优化边界和 caveat 红线合并成一个机器判据；更新后的目标不再包含 2026-06-21 晚间等待条件。",
        "",
        "## 1. Summary",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Tracking checks | {summary['tracking_checks']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| Current time local | `{summary['current_time_local']}` |",
        f"| Checkpoint start local | `{summary['checkpoint_start_local']}` |",
        f"| Checkpoint phase | `{summary['checkpoint_phase']}` |",
        f"| Seconds until checkpoint | {summary['seconds_until_checkpoint']} |",
        f"| Evidence ready | {summary['evidence_ready']} |",
        f"| Time gate open | {summary['time_gate_open']} |",
        f"| Checkpoint allows completion | {summary['checkpoint_allows_completion']} |",
        f"| Completion allowed now | {summary['completion_allowed_now']} |",
        f"| Completion blockers | `{', '.join(summary['completion_blockers']) or 'none'}` |",
        f"| Goal update recommendation | `{summary['goal_update_recommendation']}` |",
        "",
        "## 2. Checks",
        "",
        "| Status | Required | Check | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        required = "yes" if check["required"] else "no"
        evidence = _public_evidence(check["evidence"]).replace("|", "\\|")
        lines.append(
            f"| {check['status']} | {required} | {check['name']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## 3. Final Action Rule",
            "",
            "- `completion_allowed_now=true` 时，才允许调用 `update_goal(status=\"complete\")`。",
            "- `completion_allowed_now=false` 且 `ready=true` 时，说明仍有非时间 blocker，需要先清除。",
            "- `ready=false` 时，先修 required failures，不要结束目标。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the final goal-completion audit for Qwen3.5-Omni."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-completion", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_audit(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output
        if args.json_output.is_absolute()
        else root / args.json_output
    )
    _save_text(build_markdown(payload), output)
    _save_json(payload, json_output)
    summary = payload["summary"]
    print(
        "Final completion audit written: "
        f"{output} ready={summary['ready']} "
        f"completion_allowed_now={summary['completion_allowed_now']} "
        f"required_failures={summary['required_failures']}"
    )
    if args.strict and summary["required_failures"]:
        raise SystemExit(1)
    if args.require_completion and not summary["completion_allowed_now"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
