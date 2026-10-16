# SPDX-License-Identifier: Apache-2.0
"""Check a collaborator-returned Qwen3.5-Omni rerun evidence bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_CONTRACT = AUDIT_DIR / "rerun_acceptance_contract.json"
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "collaborator_return_check.json"


CORE_JSON_PATHS = {
    "audit_run_summary": AUDIT_DIR / "audit_run_summary.json",
    "environment_snapshot": AUDIT_DIR / "environment_snapshot.json",
    "final_readiness": AUDIT_DIR / "final_readiness_audit.json",
    "claims": AUDIT_DIR / "claims_verification.json",
    "coverage": AUDIT_DIR / "coverage_matrix.json",
    "acceptance": AUDIT_DIR / "acceptance_matrix.json",
    "confidence": AUDIT_DIR / "confidence_ledger.json",
    "runtime_comparison": AUDIT_DIR / "runtime_comparison_contract.json",
    "runtime_image": AUDIT_DIR / "runtime_image_contract.json",
    "receiver_quickcheck": AUDIT_DIR / "receiver_quickcheck_contract.json",
    "rerun_acceptance": AUDIT_DIR / "rerun_acceptance_contract.json",
    "rerun_delta_triage": AUDIT_DIR / "rerun_delta_triage.json",
    "sglang_lock": AUDIT_DIR / "sglang_optimization_lock.json",
    "vllm_lock": AUDIT_DIR / "vllm_optimization_lock.json",
    "vllm_online_protocol": AUDIT_DIR / "vllm_online_parity_protocol.json",
    "stage_interactions": AUDIT_DIR / "stage_interaction_summary.json",
    "stage_latency_budget": AUDIT_DIR / "stage_latency_budget.json",
    "stage_boundary_ledger": AUDIT_DIR
    / "stage_boundary_bottleneck_ledger.json",
    "stage_causal_graph": AUDIT_DIR / "stage_causal_graph.json",
    "stage_reproduction_drilldown": AUDIT_DIR
    / "stage_reproduction_drilldown.json",
    "stage_route_decision_matrix": AUDIT_DIR
    / "stage_route_decision_matrix.json",
    "share_bundle_manifest": AUDIT_DIR / "share_bundle_manifest.json",
    "share_bundle_package_manifest": AUDIT_DIR
    / "share_bundle_package_manifest.json",
    "share_package_validation": AUDIT_DIR / "share_package_validation.json",
    "share_package_receiver_smoke": AUDIT_DIR
    / "share_package_receiver_smoke_validation.json",
    "share_package_validation_extracted": AUDIT_DIR
    / "share_package_validation_extracted.json",
    "share_package_external_standalone": AUDIT_DIR
    / "share_package_external_standalone_validation.json",
}


@dataclass(frozen=True)
class Check:
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, "missing"
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return {}, "not a JSON object"
    return payload, None


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _check(
    checks: list[Check],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(Check(name, _status(condition), evidence, required))


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def _summary_ready(
    payload: dict[str, Any],
    *,
    min_total_key: str = "checks_total",
    min_total: int | None = None,
) -> bool:
    summary = _summary(payload)
    ready = bool(summary.get("ready"))
    required_failures = _int_value(summary.get("required_failures"))
    if min_total is None:
        return ready and required_failures == 0
    return (
        ready
        and required_failures == 0
        and _int_value(summary.get(min_total_key)) >= min_total
    )


def _required_return_records(contract: dict[str, Any]) -> list[dict[str, Any]]:
    section = contract.get("return_evidence_contract", {})
    records = section.get("records", []) if isinstance(section, dict) else []
    return [record for record in records if isinstance(record, dict)]


def _command_matrix(contract: dict[str, Any]) -> list[dict[str, Any]]:
    section = contract.get("return_evidence_contract", {})
    rows = section.get("command_matrix", []) if isinstance(section, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _json_parse_failures(root: Path, relative_paths: list[str]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for rel_path in relative_paths:
        if not rel_path.endswith(".json"):
            continue
        _, error = _load_json(root / rel_path)
        if error is not None:
            failures.append({"relative_path": rel_path, "error": error})
    return failures


def _load_core_jsons(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    payloads: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for key, rel_path in CORE_JSON_PATHS.items():
        payload, error = _load_json(root / rel_path)
        payloads[key] = payload
        if error is not None:
            errors[key] = error
    return payloads, errors


def _claim_gate_ready(payload: dict[str, Any]) -> bool:
    return (
        bool(payload.get("passed"))
        and _int_value(payload.get("total_checks")) >= 17
        and not payload.get("failed_checks")
    )


def _coverage_gate_ready(payload: dict[str, Any]) -> bool:
    summary = _summary(payload)
    return (
        bool(summary.get("complete"))
        and _int_value(summary.get("missing")) == 0
        and _int_value(summary.get("passed")) >= 34
    )


def _acceptance_gate_ready(payload: dict[str, Any]) -> bool:
    summary = _summary(payload)
    return (
        bool(summary.get("ready"))
        and _int_value(summary.get("rows_total")) >= 17
        and _int_value(summary.get("rows_failed")) == 0
    )


def _confidence_gate_ready(payload: dict[str, Any]) -> bool:
    summary = _summary(payload)
    return (
        bool(summary.get("ready"))
        and _int_value(summary.get("entries_total")) >= 12
        and _int_value(summary.get("entries_failed")) == 0
        and _int_value(summary.get("unsupported_claims")) == 0
    )


def _audit_gate_ready(payload: dict[str, Any]) -> bool:
    if bool(payload.get("ok")) and not bool(payload.get("in_progress")):
        return True
    if not (bool(payload.get("ok")) and bool(payload.get("in_progress"))):
        return False
    steps = payload.get("steps", [])
    return isinstance(steps, list) and all(
        isinstance(step, dict) and bool(step.get("ok")) for step in steps
    )


def _final_readiness_ready(payload: dict[str, Any]) -> bool:
    summary = _summary(payload)
    return (
        bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 49
        and _int_value(summary.get("required_failures")) == 0
    )


def _stage_interactions_ready(payload: dict[str, Any]) -> bool:
    summary = _summary(payload)
    return (
        bool(summary.get("sglang_talker_to_code2wav_healthy"))
        and bool(summary.get("sglang_code2wav_decode_not_bottleneck"))
        and bool(summary.get("vllm_original_c8_prompt_feed_limited"))
        and bool(summary.get("preprocessing_parallelism_regresses"))
    )


def _stage_gates_ready(payloads: dict[str, dict[str, Any]]) -> bool:
    return all(
        [
            _stage_interactions_ready(payloads["stage_interactions"]),
            _summary_ready(payloads["stage_latency_budget"], min_total=12),
            _summary_ready(payloads["stage_boundary_ledger"], min_total=12),
            _summary_ready(payloads["stage_causal_graph"], min_total=7),
            _summary_ready(
                payloads["stage_reproduction_drilldown"], min_total=16
            ),
            _summary_ready(payloads["stage_route_decision_matrix"], min_total=9),
        ]
    )


def _runtime_comparison_ready(payload: dict[str, Any]) -> tuple[bool, str]:
    summary = _summary(payload)
    strict_scope = str(summary.get("strict_scope") or "")
    allowed_headline = str(summary.get("allowed_cross_runtime_headline") or "")
    c8_contract = str(summary.get("vllm_c8_contract") or "")
    baseline_strength = str(summary.get("baseline_strength") or "")
    condition = (
        bool(summary.get("ready"))
        and _int_value(summary.get("required_failures")) == 0
        and "c=4" in strict_scope
        and "c=4" in allowed_headline
        and "offline" in c8_contract
        and "optimized" in baseline_strength
    )
    evidence = (
        f"ready={summary.get('ready')}, strict_scope={strict_scope}, "
        f"allowed_headline={allowed_headline}, c8_contract={c8_contract}, "
        f"baseline_strength={baseline_strength}"
    )
    return condition, evidence


def _vllm_online_caveat_ready(payload: dict[str, Any]) -> tuple[bool, str]:
    summary = _summary(payload)
    condition = (
        bool(summary.get("ready"))
        and _int_value(summary.get("required_failures")) == 0
        and bool(summary.get("current_package_safe"))
        and not bool(summary.get("online_parity_proven"))
    )
    evidence = (
        f"ready={summary.get('ready')}, "
        f"current_package_safe={summary.get('current_package_safe')}, "
        f"online_parity_proven={summary.get('online_parity_proven')}, "
        f"upgrade_decision={summary.get('upgrade_decision')}"
    )
    return condition, evidence


def _environment_match(
    environment: dict[str, Any],
    runtime_image: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    runtime_summary = _summary(runtime_image)
    gpu = environment.get("gpu", {})
    docker_images = environment.get("docker_images", {})
    sglang_image = docker_images.get("sglang", {})
    vllm_image = docker_images.get("vllm", {})
    gpu_names = [
        str(item.get("name") or "")
        for item in gpu.get("gpus", [])
        if isinstance(item, dict)
    ]
    gpu_count = _int_value(gpu.get("count"))
    cuda_version = str(gpu.get("cuda_version") or "")
    gpu_ok = (
        bool(gpu.get("ok"))
        and gpu_count >= 8
        and len(gpu_names) >= 8
        and all("H20" in name for name in gpu_names[:8])
        and cuda_version == "12.8"
    )
    sglang_env_id = str(sglang_image.get("id") or "")
    vllm_env_id = str(vllm_image.get("id") or "")
    sglang_contract_id = str(runtime_summary.get("sglang_image_id") or "")
    vllm_contract_id = str(runtime_summary.get("vllm_image_id") or "")
    sglang_match = bool(sglang_env_id) and sglang_env_id == sglang_contract_id
    vllm_match = bool(vllm_env_id) and vllm_env_id == vllm_contract_id
    comparison = {
        "gpu_ok": gpu_ok,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "cuda_version": cuda_version,
        "gpu_contract": runtime_summary.get("gpu_contract"),
        "sglang_image_id_environment": sglang_env_id,
        "sglang_image_id_contract": sglang_contract_id,
        "sglang_image_id_match": sglang_match,
        "vllm_image_id_environment": vllm_env_id,
        "vllm_image_id_contract": vllm_contract_id,
        "vllm_image_id_match": vllm_match,
    }
    return gpu_ok and sglang_match and vllm_match, comparison


def _share_package_chain_ready(payloads: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    expected = [
        ("share_package_validation", 17),
        ("share_package_receiver_smoke", 17),
        ("share_package_validation_extracted", 13),
        ("share_package_external_standalone", 8),
    ]
    statuses: list[str] = []
    ok = True
    for key, min_checks in expected:
        summary = _summary(payloads[key])
        item_ok = (
            bool(summary.get("ready"))
            and _int_value(summary.get("checks_total")) >= min_checks
            and _int_value(summary.get("required_failures")) == 0
        )
        ok = ok and item_ok
        statuses.append(
            f"{key}={summary.get('ready')}/"
            f"{summary.get('checks_passed')}/{summary.get('checks_total')}/"
            f"required_failures={summary.get('required_failures')}"
        )
    return ok, "; ".join(statuses)


def _decision(
    *,
    missing_required_files: list[str],
    json_failures: list[dict[str, str]],
    command_file_gap_rows: list[dict[str, Any]],
    environment_match: bool,
    required_failures: int,
) -> str:
    if missing_required_files:
        return "missing_required_return_evidence"
    if json_failures:
        return "invalid_return_evidence_json"
    if command_file_gap_rows:
        return "command_matrix_return_files_missing"
    if not environment_match:
        return "appendix_only_environment_or_image_mismatch"
    if required_failures:
        return "do_not_replace_until_required_gates_green"
    return "eligible_for_current_scope_headline_replacement_review"


def build_check(root: Path, *, contract_path: Path) -> dict[str, Any]:
    root = root.resolve()
    contract_full_path = (
        contract_path if contract_path.is_absolute() else root / contract_path
    ).resolve()
    contract, contract_error = _load_json(contract_full_path)
    contract_rel = str(contract_full_path.relative_to(root))
    records = _required_return_records(contract)
    command_rows = _command_matrix(contract)
    required_paths = [
        str(record.get("relative_path") or "")
        for record in records
        if record.get("relative_path")
    ]
    existing_required_paths = [
        rel_path for rel_path in required_paths if (root / rel_path).exists()
    ]
    missing_required_files = [
        rel_path for rel_path in required_paths if not (root / rel_path).exists()
    ]
    json_failures = _json_parse_failures(root, required_paths)
    command_missing_rows = [
        {
            "command_id": row.get("command_id"),
            "phase": row.get("phase"),
            "required_files_missing": [
                rel_path
                for rel_path in row.get("required_return_files", [])
                if rel_path not in existing_required_paths
            ],
            "declared_ready": bool(row.get("ready")),
        }
        for row in command_rows
        if not row.get("ready")
        or any(
            rel_path not in existing_required_paths
            for rel_path in row.get("required_return_files", [])
        )
    ]
    command_rows_ready = [
        row
        for row in command_rows
        if row.get("ready")
        and not any(
            rel_path not in existing_required_paths
            for rel_path in row.get("required_return_files", [])
        )
    ]
    payloads, core_errors = _load_core_jsons(root)

    checks: list[Check] = []
    contract_summary = _summary(contract)
    _check(
        checks,
        "rerun acceptance contract is parseable and current",
        contract_error is None
        and bool(contract_summary.get("ready"))
        and _int_value(contract_summary.get("required_failures")) == 0
        and _int_value(contract_summary.get("return_evidence_files")) == 34
        and _int_value(contract_summary.get("return_evidence_command_rows")) == 27,
        (
            f"error={contract_error}, summary_ready={contract_summary.get('ready')}, "
            f"return_files={contract_summary.get('return_evidence_files')}, "
            f"command_rows={contract_summary.get('return_evidence_command_rows')}, "
            f"required_failures={contract_summary.get('required_failures')}"
        ),
    )
    _check(
        checks,
        "all required return evidence files are present",
        len(required_paths) == 34 and not missing_required_files,
        (
            f"present={len(existing_required_paths)}/{len(required_paths)}, "
            f"missing={missing_required_files}"
        ),
    )
    _check(
        checks,
        "required return JSON files parse",
        not json_failures,
        f"json_parse_failures={json_failures}",
    )
    _check(
        checks,
        "command matrix rows are ready and backed by returned files",
        len(command_rows) == 27
        and len(command_rows_ready) == 27
        and not command_missing_rows,
        (
            f"rows_ready={len(command_rows_ready)}/{len(command_rows)}, "
            f"gaps={command_missing_rows}"
        ),
    )
    _check(
        checks,
        "full audit and final readiness are green",
        _audit_gate_ready(payloads["audit_run_summary"])
        and _final_readiness_ready(payloads["final_readiness"]),
        (
            f"audit_ok={payloads['audit_run_summary'].get('ok')}, "
            f"audit_in_progress={payloads['audit_run_summary'].get('in_progress')}, "
            f"final_ready={_summary(payloads['final_readiness']).get('ready')}, "
            f"final_checks={_summary(payloads['final_readiness']).get('checks_passed')}/"
            f"{_summary(payloads['final_readiness']).get('checks_total')}, "
            f"required_failures="
            f"{_summary(payloads['final_readiness']).get('required_failures')}"
        ),
    )
    _check(
        checks,
        "claim coverage acceptance confidence gates are green",
        _claim_gate_ready(payloads["claims"])
        and _coverage_gate_ready(payloads["coverage"])
        and _acceptance_gate_ready(payloads["acceptance"])
        and _confidence_gate_ready(payloads["confidence"]),
        (
            f"claims_passed={payloads['claims'].get('passed')}/"
            f"{payloads['claims'].get('total_checks')}, "
            f"claims_failed={payloads['claims'].get('failed_checks')}, "
            f"coverage={_summary(payloads['coverage']).get('passed')}/"
            f"{_summary(payloads['coverage']).get('total_requirements')}, "
            f"acceptance={_summary(payloads['acceptance']).get('rows_passed')}/"
            f"{_summary(payloads['acceptance']).get('rows_total')}, "
            f"confidence={_summary(payloads['confidence']).get('entries_passed')}/"
            f"{_summary(payloads['confidence']).get('entries_total')}, "
            f"unsupported_claims="
            f"{_summary(payloads['confidence']).get('unsupported_claims')}"
        ),
    )
    env_match, env_comparison = _environment_match(
        payloads["environment_snapshot"],
        payloads["runtime_image"],
    )
    _check(
        checks,
        "runtime image contract is green",
        _summary_ready(payloads["runtime_image"], min_total=12),
        f"runtime_image={_summary(payloads['runtime_image'])}",
    )
    _check(
        checks,
        "environment matches headline replacement hardware and image contract",
        env_match,
        json.dumps(env_comparison, ensure_ascii=False, sort_keys=True),
    )
    runtime_ok, runtime_evidence = _runtime_comparison_ready(
        payloads["runtime_comparison"]
    )
    _check(
        checks,
        "runtime comparison keeps warmed c4 headline and c8 diagnostic scope",
        runtime_ok,
        runtime_evidence,
    )
    _check(
        checks,
        "stage interaction and boundary gates are green",
        _stage_gates_ready(payloads),
        (
            f"stage_interactions={_summary(payloads['stage_interactions']).get('status_counts')}, "
            f"latency={_summary(payloads['stage_latency_budget']).get('checks_passed')}/"
            f"{_summary(payloads['stage_latency_budget']).get('checks_total')}, "
            f"boundary={_summary(payloads['stage_boundary_ledger']).get('checks_passed')}/"
            f"{_summary(payloads['stage_boundary_ledger']).get('checks_total')}, "
            f"causal={_summary(payloads['stage_causal_graph']).get('checks_passed')}/"
            f"{_summary(payloads['stage_causal_graph']).get('checks_total')}, "
            f"drilldown={_summary(payloads['stage_reproduction_drilldown']).get('checks_passed')}/"
            f"{_summary(payloads['stage_reproduction_drilldown']).get('checks_total')}, "
            f"route={_summary(payloads['stage_route_decision_matrix']).get('checks_passed')}/"
            f"{_summary(payloads['stage_route_decision_matrix']).get('checks_total')}"
        ),
    )
    vllm_protocol_ok, vllm_protocol_evidence = _vllm_online_caveat_ready(
        payloads["vllm_online_protocol"]
    )
    _check(
        checks,
        "vLLM online parity caveat is explicit and safe",
        vllm_protocol_ok,
        vllm_protocol_evidence,
    )
    _check(
        checks,
        "SGLang and vLLM optimization locks are green",
        _summary_ready(payloads["sglang_lock"], min_total=26)
        and _summary_ready(payloads["vllm_lock"], min_total=22),
        (
            f"sglang={_summary(payloads['sglang_lock'])}, "
            f"vllm={_summary(payloads['vllm_lock'])}"
        ),
    )
    _check(
        checks,
        "rerun acceptance and delta triage gates are green",
        _summary_ready(payloads["rerun_acceptance"], min_total=17)
        and _summary_ready(payloads["rerun_delta_triage"], min_total=8),
        (
            f"rerun_acceptance={_summary(payloads['rerun_acceptance'])}, "
            f"rerun_delta_triage={_summary(payloads['rerun_delta_triage'])}"
        ),
    )
    _check(
        checks,
        "share bundle and package manifests are green",
        _summary_ready(payloads["share_bundle_manifest"])
        and _summary_ready(payloads["share_bundle_package_manifest"]),
        (
            f"share_bundle={_summary(payloads['share_bundle_manifest'])}, "
            f"share_package={_summary(payloads['share_bundle_package_manifest'])}"
        ),
    )
    share_chain_ok, share_chain_evidence = _share_package_chain_ready(payloads)
    _check(
        checks,
        "share package validation chain is green",
        share_chain_ok,
        share_chain_evidence,
    )
    _check(
        checks,
        "receiver quickcheck contract is green",
        _summary_ready(payloads["receiver_quickcheck"], min_total=14),
        f"receiver_quickcheck={_summary(payloads['receiver_quickcheck'])}",
    )

    checks_total = len(checks)
    checks_passed = sum(1 for check in checks if check.status == "PASS")
    required_failures = sum(
        1 for check in checks if check.required and check.status != "PASS"
    )
    ready = required_failures == 0
    decision = _decision(
        missing_required_files=missing_required_files,
        json_failures=json_failures,
        command_file_gap_rows=command_missing_rows,
        environment_match=env_match,
        required_failures=required_failures,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "contract": str(contract_full_path),
        "contract_relative_path": contract_rel,
        "summary": {
            "ready": ready,
            "checks_total": checks_total,
            "checks_passed": checks_passed,
            "required_failures": required_failures,
            "required_return_files_total": len(required_paths),
            "required_return_files_present": len(existing_required_paths),
            "required_return_files_missing": len(missing_required_files),
            "json_parse_failures": len(json_failures),
            "command_matrix_rows_total": len(command_rows),
            "command_matrix_rows_ready": len(command_rows_ready),
            "command_matrix_rows_with_gaps": len(command_missing_rows),
            "eligible_for_headline_replacement_review": ready,
            "replacement_review_decision": decision,
            "vllm_online_parity_proven": _summary(
                payloads["vllm_online_protocol"]
            ).get("online_parity_proven"),
            "vllm_c8_scope": _summary(
                payloads["runtime_comparison"]
            ).get("vllm_c8_contract"),
        },
        "environment_comparison": env_comparison,
        "missing_required_return_files": missing_required_files,
        "json_parse_failures": json_failures,
        "command_matrix_rows_with_gaps": command_missing_rows,
        "core_json_load_errors": core_errors,
        "checks": [check.to_dict() for check in checks],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Collaborator Return Check")
    print()
    print(f"ready: `{summary['ready']}`")
    print(
        "headline_replacement_review: "
        f"`{summary['eligible_for_headline_replacement_review']}`"
    )
    print(f"decision: `{summary['replacement_review_decision']}`")
    print(
        "required_return_files: "
        f"`{summary['required_return_files_present']}/"
        f"{summary['required_return_files_total']}`"
    )
    print(
        "command_matrix_rows: "
        f"`{summary['command_matrix_rows_ready']}/"
        f"{summary['command_matrix_rows_total']}`"
    )
    print(
        "checks: "
        f"`{summary['checks_passed']}/{summary['checks_total']}`; "
        f"required_failures=`{summary['required_failures']}`"
    )
    print()
    print("| Check | Status | Evidence |")
    print("| --- | --- | --- |")
    for check in payload["checks"]:
        evidence = str(check["evidence"]).replace("|", "\\|")
        print(f"| {check['name']} | {check['status']} | {evidence} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check returned Qwen3.5-Omni rerun evidence against the "
            "rerun acceptance contract."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Path to rerun_acceptance_contract.json, relative to --root by default.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path. Relative paths are resolved under --root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when the returned bundle is not replacement-review ready.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_check(root, contract_path=args.contract)
    print_markdown(payload)
    if args.json_output:
        output = args.json_output if args.json_output.is_absolute() else root / args.json_output
        _save_json(payload, output)
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
