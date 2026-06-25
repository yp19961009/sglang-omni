# SPDX-License-Identifier: Apache-2.0
"""Build an adjacent release seal for the Qwen3.5-Omni share tarball."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path("benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md")
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "share_release_seal.json"
DEFAULT_TARBALL = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"

ADJACENT_ARTIFACTS = [
    "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
    "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256",
    "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
    "results/qwen35_report_audit_20260619/share_package_validation.json",
    "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
    "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
    "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.summary.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.query.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.summary.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.query.out",
    "results/qwen35_report_audit_20260619/final_completion_audit.json",
    "results/qwen35_report_audit_20260619/share_release_seal.json",
    "benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
]

TARBALL_FORBIDDEN_MEMBERS = {
    "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
    "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256",
    "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
    "results/qwen35_report_audit_20260619/share_package_validation.json",
    "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
    "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
    "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.summary.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.query.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.summary.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.query.out",
    "results/qwen35_report_audit_20260619/final_completion_audit.json",
    "results/qwen35_report_audit_20260619/share_release_seal.json",
    "benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
}


@dataclass(frozen=True)
class SealCheck:
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


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _check(
    checks: list[SealCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    if condition:
        status = "PASS"
    elif required:
        status = "FAIL"
    else:
        status = "WARN"
    checks.append(SealCheck(name, status, evidence, required))


def _checksum_value(path: Path) -> tuple[str | None, str]:
    if not path.is_file():
        return None, "missing checksum file"
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if not parts:
        return None, "empty checksum file"
    return parts[0], text


def _tar_members(path: Path) -> tuple[set[str], str | None]:
    if not path.is_file():
        return set(), "missing tarball"
    try:
        with tarfile.open(path, "r:gz") as tf:
            return set(tf.getnames()), None
    except Exception as exc:
        return set(), str(exc)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _adjacent_artifact_records(
    root: Path,
    rel_paths: list[str],
    *,
    self_rel_paths: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        path = root / rel_path
        self_artifact = rel_path in self_rel_paths
        record: dict[str, Any] = {
            "relative_path": rel_path,
            "exists": path.is_file(),
            "self_artifact": self_artifact,
        }
        if path.is_file() and not self_artifact:
            digest = _sha256(path)
            record.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "sha256_uri": f"sha256:{digest}",
                }
            )
        elif self_artifact:
            record["hash_policy"] = "omitted_to_avoid_self_reference"
            record["size_policy"] = "omitted_to_avoid_self_reference"
        records.append(record)
    return records


def _ready_summary(
    payload: dict[str, Any],
    *,
    min_checks: int = 0,
    require_extracted_only: bool | None = None,
) -> tuple[bool, str]:
    summary = _summary(payload)
    ready = bool(summary.get("ready"))
    checks_passed = _int_value(summary.get("checks_passed"))
    checks_total = _int_value(summary.get("checks_total"))
    required_failures = _int_value(summary.get("required_failures"), default=1)
    extracted_ok = True
    if require_extracted_only is not None:
        extracted_ok = bool(summary.get("extracted_only")) is require_extracted_only
    ok = (
        ready
        and required_failures == 0
        and checks_total >= min_checks
        and checks_passed == checks_total
        and extracted_ok
    )
    tarball_sha = summary.get("tarball_sha256")
    tarball_sha_uri = f"sha256:{tarball_sha}" if tarball_sha else None
    evidence_parts = [
        f"ready={ready}",
        f"checks={checks_passed}/{checks_total}",
        f"required_failures={required_failures}",
        f"warnings={summary.get('warnings')}",
        f"tarball_sha256={tarball_sha_uri}",
        f"tar_members={summary.get('tar_members')}",
        f"expected_bundle_members={summary.get('expected_bundle_members')}",
        f"missing_bundle_members={summary.get('missing_bundle_members')}",
        f"extracted_only={summary.get('extracted_only')}",
    ]
    if "receiver_smoke_ready" in summary:
        evidence_parts.append(f"receiver_smoke_ready={summary.get('receiver_smoke_ready')}")
    evidence = ", ".join(evidence_parts)
    return ok, evidence


def _evidence_query_smoke_state(
    root: Path,
    payload: dict[str, Any],
    *,
    expected_tarball_sha256: str | None,
) -> tuple[bool, dict[str, Any], str]:
    host = payload.get("host", {})
    if not isinstance(host, dict):
        host = {}
    portable = payload.get("portable", {})
    if not isinstance(portable, dict):
        portable = {}

    expected_outputs = [
        str(host.get("summary_output") or ""),
        str(host.get("query_output") or ""),
        str(portable.get("summary_output") or ""),
        str(portable.get("query_output") or ""),
    ]
    missing_outputs = [
        rel_path
        for rel_path in expected_outputs
        if not rel_path or not (root / rel_path).is_file()
    ]
    tarball_sha256 = payload.get("tarball_sha256")
    host_pass_lines = _int_value(host.get("pass_lines"))
    portable_pass_lines = _int_value(portable.get("pass_lines"))
    host_query_output_lines = _int_value(host.get("query_output_lines"))
    portable_query_output_lines = _int_value(portable.get("query_output_lines"))
    ok = (
        bool(payload.get("ready"))
        and bool(expected_tarball_sha256)
        and tarball_sha256 == expected_tarball_sha256
        and host_pass_lines >= 20
        and portable_pass_lines >= 18
        and host_query_output_lines > 0
        and portable_query_output_lines > 0
        and not missing_outputs
    )
    summary = {
        "ready": bool(payload.get("ready")),
        "tarball_sha256_matches": tarball_sha256 == expected_tarball_sha256,
        "host_pass_lines": host_pass_lines,
        "portable_pass_lines": portable_pass_lines,
        "host_query_output_lines": host_query_output_lines,
        "portable_query_output_lines": portable_query_output_lines,
        "outputs_total": len(expected_outputs),
        "missing_outputs": missing_outputs,
        "portable_note": portable.get("note"),
    }
    evidence = (
        f"ready={payload.get('ready')}, "
        f"tarball_sha256=sha256:{tarball_sha256}, "
        f"expected_tarball_sha256=sha256:{expected_tarball_sha256}, "
        f"host_pass_lines={host_pass_lines}, "
        f"portable_pass_lines={portable_pass_lines}, "
        f"host_query_output_lines={host_query_output_lines}, "
        f"portable_query_output_lines={portable_query_output_lines}, "
        f"missing_outputs={missing_outputs}, "
        f"portable_note={portable.get('note')}"
    )
    return ok, summary, evidence


def _boundary_items(objective: dict[str, Any]) -> list[str]:
    rows = objective.get("boundary_items", [])
    if not isinstance(rows, list):
        return []
    result: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("caveat") or row.get("requirement") or "").strip()
        if text:
            result.append(text)
    return result


def build_release_seal(
    root: Path,
    *,
    output: Path,
    json_output: Path,
    tarball: Path,
) -> dict[str, Any]:
    root = root.resolve()
    output = output if output.is_absolute() else root / output
    json_output = json_output if json_output.is_absolute() else root / json_output
    tarball = tarball if tarball.is_absolute() else root / tarball
    checksum_path = tarball.with_suffix(tarball.suffix + ".sha256")
    audit_dir = root / AUDIT_DIR

    package = _load_json_optional(audit_dir / "share_bundle_package_manifest.json")
    validation = _load_json_optional(audit_dir / "share_package_validation.json")
    receiver_smoke = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    extracted = _load_json_optional(audit_dir / "share_package_validation_extracted.json")
    external = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    evidence_query_smoke = _load_json_optional(
        audit_dir / "evidence_query_cards_smoke_summary.json"
    )
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    final_completion = _load_json_optional(audit_dir / "final_completion_audit.json")
    checkpoint = _load_json_optional(audit_dir / "final_checkpoint_watchlist.json")
    objective = _load_json_optional(audit_dir / "objective_completion_audit.json")
    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    coverage = _load_json_optional(audit_dir / "coverage_matrix.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    chart_source = _load_json_optional(audit_dir / "chart_source_consistency.json")
    runtime_image = _load_json_optional(audit_dir / "runtime_image_contract.json")
    runtime_comparison = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")

    actual_sha = _sha256(tarball) if tarball.is_file() else None
    checksum_sha, checksum_text = _checksum_value(checksum_path)
    members, tar_error = _tar_members(tarball)
    self_rel_paths = {_relative(root, output), _relative(root, json_output)}
    adjacent_artifact_records = _adjacent_artifact_records(
        root,
        ADJACENT_ARTIFACTS,
        self_rel_paths=self_rel_paths,
    )
    missing_adjacent_artifacts = [
        record["relative_path"]
        for record in adjacent_artifact_records
        if not record["exists"] and not record["self_artifact"]
    ]
    hashed_adjacent_artifacts = [
        record
        for record in adjacent_artifact_records
        if record.get("sha256_uri")
    ]
    self_adjacent_artifacts = [
        record
        for record in adjacent_artifact_records
        if record.get("self_artifact")
    ]
    evidence_query_smoke_ok, evidence_query_smoke_summary, evidence_query_smoke_evidence = (
        _evidence_query_smoke_state(
            root,
            evidence_query_smoke,
            expected_tarball_sha256=actual_sha,
        )
    )
    forbidden_members = sorted(
        member
        for member in members
        for forbidden in TARBALL_FORBIDDEN_MEMBERS
        if member.endswith("/" + forbidden) or member == forbidden
    )

    package_summary = _summary(package)
    validation_summary = _summary(validation)
    receiver_summary = _summary(receiver_smoke)
    extracted_summary = _summary(extracted)
    external_summary = _summary(external)
    final_summary = _summary(final_readiness)
    final_completion_summary = _summary(final_completion)
    checkpoint_summary = _summary(checkpoint)
    objective_summary = _summary(objective)
    manifest_summary = _summary(manifest)
    preflight_summary = _summary(preflight)
    coverage_summary = _summary(coverage)
    repro_summary = _summary(repro)
    chart_summary = _summary(chart_source)
    runtime_image_summary = _summary(runtime_image)
    runtime_comparison_summary = _summary(runtime_comparison)
    sglang_summary = _summary(sglang_lock)
    vllm_summary = _summary(vllm_lock)
    vllm_online_summary = _summary(vllm_online)
    share_bundle_summary = _summary(share_bundle)

    checks: list[SealCheck] = []
    _check(
        checks,
        "tarball and checksum agree",
        tarball.is_file()
        and checksum_path.is_file()
        and actual_sha is not None
        and checksum_sha == actual_sha
        and package.get("tarball_sha256") == actual_sha
        and validation_summary.get("tarball_sha256") == actual_sha
        and receiver_summary.get("tarball_sha256") == actual_sha,
        (
            f"tarball={_relative(root, tarball)}, checksum={_relative(root, checksum_path)}, "
            f"actual_sha256=sha256:{actual_sha}, checksum_sha256=sha256:{checksum_sha}, "
            f"package_sha256=sha256:{package.get('tarball_sha256')}, "
            f"validation_sha256=sha256:{validation_summary.get('tarball_sha256')}, "
            f"receiver_smoke_sha256=sha256:{receiver_summary.get('tarball_sha256')}, "
            f"checksum_text_sha256=sha256:{checksum_sha}, "
            f"checksum_text_path={_relative(root, tarball)}"
        ),
    )
    _check(
        checks,
        "share bundle package manifest ready",
        bool(package_summary.get("ready"))
        and _int_value(package_summary.get("file_count")) >= 98
        and _int_value(package_summary.get("source_manifest_records")) >= 98,
        f"package={package_summary}",
    )
    validation_ok, validation_evidence = _ready_summary(validation, min_checks=17)
    _check(checks, "tarball-mode validation ready", validation_ok, validation_evidence)
    receiver_ok, receiver_evidence = _ready_summary(receiver_smoke, min_checks=17)
    _check(
        checks,
        "receiver smoke validation ready",
        receiver_ok and bool(receiver_summary.get("receiver_smoke_ready")),
        receiver_evidence,
    )
    extracted_ok, extracted_evidence = _ready_summary(
        extracted, min_checks=13, require_extracted_only=True
    )
    _check(
        checks,
        "extracted-only validation ready",
        extracted_ok,
        extracted_evidence,
    )
    external_ok, external_evidence = _ready_summary(external, min_checks=8)
    _check(
        checks,
        "external standalone validation ready",
        external_ok,
        external_evidence,
    )
    _check(
        checks,
        "release seal remains outside tarball",
        tar_error is None and not forbidden_members and not missing_adjacent_artifacts,
        (
            f"tar_error={tar_error}, forbidden_members={forbidden_members}, "
            f"adjacent_artifacts={len(adjacent_artifact_records)}, "
            f"hashed_adjacent_artifacts={len(hashed_adjacent_artifacts)}, "
            f"missing_adjacent_artifacts={missing_adjacent_artifacts}"
        ),
    )
    _check(
        checks,
        "full audit and final readiness are green",
        (bool(audit_summary.get("ok")) or bool(final_summary.get("ready")))
        and bool(final_summary.get("ready"))
        and _int_value(final_summary.get("checks_total")) >= 47
        and _int_value(final_summary.get("required_failures"), default=1) == 0,
        (
            f"audit_ok={audit_summary.get('ok')}, final_readiness="
            f"{final_summary}"
        ),
    )
    _check(
        checks,
        "objective is share-ready with documented caveats",
        bool(objective_summary.get("share_ready_with_documented_caveats"))
        and _int_value(objective_summary.get("required_failures"), default=1) == 0,
        f"objective={objective_summary}",
    )
    _check(
        checks,
        "completion watchlist allows evidence-based completion",
        bool(checkpoint_summary.get("ready"))
        and checkpoint_summary.get("checkpoint_phase") == "completion_audit_ready"
        and bool(checkpoint_summary.get("completion_allowed_now"))
        and _int_value(checkpoint_summary.get("required_failures"), default=1) == 0,
        f"checkpoint={checkpoint_summary}",
    )
    _check(
        checks,
        "core audit inventory gates are current",
        bool(claims.get("passed"))
        and _int_value(claims.get("failed_checks"), default=1) == 0
        and bool(coverage_summary.get("complete"))
        and _int_value(coverage_summary.get("missing"), default=1) == 0
        and bool(preflight_summary.get("ready"))
        and _int_value(preflight_summary.get("required_failures"), default=1) == 0
        and _int_value(preflight_summary.get("total_checks")) >= 62
        and _int_value(manifest_summary.get("missing_records"), default=1) == 0
        and _int_value(manifest_summary.get("total_records")) >= 180
        and bool(repro_summary.get("ready"))
        and _int_value(repro_summary.get("commands_total")) >= 60
        and evidence_query_smoke_ok,
        (
            f"claims={claims.get('total_checks')}/{claims.get('failed_checks')}; "
            f"coverage={coverage_summary}; preflight={preflight_summary}; "
            f"manifest={manifest_summary}; repro={repro_summary}; "
            f"evidence_query_smoke={evidence_query_smoke_evidence}"
        ),
    )
    _check(
        checks,
        "runtime and optimization contracts are green",
        bool(runtime_image_summary.get("ready"))
        and bool(runtime_comparison_summary.get("ready"))
        and bool(sglang_summary.get("ready"))
        and bool(vllm_summary.get("ready"))
        and _int_value(runtime_image_summary.get("required_failures"), default=1) == 0
        and _int_value(runtime_comparison_summary.get("required_failures"), default=1)
        == 0
        and _int_value(sglang_summary.get("required_failures"), default=1) == 0
        and _int_value(vllm_summary.get("required_failures"), default=1) == 0,
        (
            f"runtime_image={runtime_image_summary}; "
            f"runtime_comparison={runtime_comparison_summary}; "
            f"sglang_lock={sglang_summary}; vllm_lock={vllm_summary}"
        ),
    )
    _check(
        checks,
        "vLLM c8 caveat is preserved",
        bool(vllm_online_summary.get("ready"))
        and bool(vllm_online_summary.get("current_package_safe"))
        and not bool(vllm_online_summary.get("online_parity_proven")),
        f"vllm_online={vllm_online_summary}",
    )
    _check(
        checks,
        "chart source and share bundle gates are green",
        bool(chart_summary.get("ready"))
        and _int_value(chart_summary.get("byte_exact_files")) >= 14
        and bool(share_bundle_summary.get("ready"))
        and _int_value(share_bundle_summary.get("missing_required"), default=1) == 0
        and _int_value(share_bundle_summary.get("records_total")) >= 98,
        f"chart_source={chart_summary}; share_bundle={share_bundle_summary}",
    )

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "tarball": str(tarball),
        "tarball_relative_path": _relative(root, tarball),
        "checksum_file": str(checksum_path),
        "checksum_relative_path": _relative(root, checksum_path),
        "tarball_sha256": actual_sha,
        "tarball_sha256_uri": f"sha256:{actual_sha}" if actual_sha else None,
        "seal_report": _relative(root, output),
        "seal_json": _relative(root, json_output),
        "adjacent_artifacts": ADJACENT_ARTIFACTS,
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "tarball_sha256": actual_sha,
            "tarball_sha256_uri": f"sha256:{actual_sha}" if actual_sha else None,
            "tarball_size_bytes": tarball.stat().st_size if tarball.is_file() else 0,
            "package_file_count": package_summary.get("file_count"),
            "share_bundle_records": share_bundle_summary.get("records_total"),
            "manifest_records": manifest_summary.get("total_records"),
            "final_readiness_checks": final_summary.get("checks_total"),
            "repro_commands_total": repro_summary.get("commands_total"),
            "receiver_smoke_ready": receiver_summary.get("receiver_smoke_ready"),
            "evidence_query_smoke_ready": evidence_query_smoke_summary.get("ready"),
            "evidence_query_smoke_host_pass_lines": evidence_query_smoke_summary.get(
                "host_pass_lines"
            ),
            "evidence_query_smoke_portable_pass_lines": evidence_query_smoke_summary.get(
                "portable_pass_lines"
            ),
            "goal_complete": objective_summary.get("goal_complete"),
            "completion_allowed_now": bool(
                final_completion_summary.get("completion_allowed_now")
            ),
            "completion_blockers": final_completion_summary.get(
                "completion_blockers",
                [],
            ),
            "send_decision": (
                "send_tarball_with_adjacent_seal_and_caveats"
                if not required_failures
                else "do_not_send_until_seal_failures_are_fixed"
            ),
            "forbidden_tarball_members": forbidden_members,
            "adjacent_artifacts_total": len(adjacent_artifact_records),
            "adjacent_artifacts_hashed": len(hashed_adjacent_artifacts),
            "adjacent_artifacts_self_omitted": len(self_adjacent_artifacts),
            "adjacent_artifacts_missing": len(missing_adjacent_artifacts),
        },
        "source_summaries": {
            "package": package_summary,
            "share_package_validation": validation_summary,
            "receiver_smoke_validation": receiver_summary,
            "extracted_validation": extracted_summary,
            "external_standalone_validation": external_summary,
            "evidence_query_smoke": evidence_query_smoke_summary,
            "final_readiness": final_summary,
            "final_completion_audit": final_completion_summary,
            "checkpoint_watchlist": checkpoint_summary,
            "objective_completion": objective_summary,
            "manifest": manifest_summary,
            "preflight": preflight_summary,
            "repro_command_manifest": repro_summary,
            "chart_source_consistency": chart_summary,
            "runtime_image_contract": runtime_image_summary,
            "runtime_comparison_contract": runtime_comparison_summary,
            "sglang_optimization_lock": sglang_summary,
            "vllm_optimization_lock": vllm_summary,
            "vllm_online_parity_protocol": vllm_online_summary,
            "share_bundle_manifest": share_bundle_summary,
        },
        "adjacent_artifact_records": adjacent_artifact_records,
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "caveats_that_must_travel": _boundary_items(objective)
        or [
            "Official SeedTTS full-set is not a headline benchmark in this package.",
            "vLLM c=8 prebuild w4 is an optimized offline diagnostic, not online serving parity.",
            "Host-side Whisper large-v3 cache warning is optional unless WER is recomputed on the host.",
            "c=16 is a saturation boundary, not the recommended serving point.",
        ],
    }
    _save_json(payload, json_output)
    _save_text(build_markdown(payload), output)
    return payload


def _checks_marker(summary: dict[str, Any]) -> str:
    return f"{summary.get('checks_passed')}/{summary.get('checks_total')}"


def _table_text(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _compact_machine_check_evidence(check: dict[str, Any], payload: dict[str, Any]) -> str:
    name = str(check.get("name") or "")
    source = payload["source_summaries"]
    summary = payload["summary"]
    package = source["package"]
    validation = source["share_package_validation"]
    receiver = source["receiver_smoke_validation"]
    extracted = source["extracted_validation"]
    external = source["external_standalone_validation"]
    evidence_query = source["evidence_query_smoke"]
    final = source["final_readiness"]
    final_completion = source["final_completion_audit"]
    checkpoint = source["checkpoint_watchlist"]
    objective = source["objective_completion"]
    manifest = source["manifest"]
    preflight = source["preflight"]
    repro = source["repro_command_manifest"]
    chart = source["chart_source_consistency"]
    runtime_image = source["runtime_image_contract"]
    runtime = source["runtime_comparison_contract"]
    sglang = source["sglang_optimization_lock"]
    vllm = source["vllm_optimization_lock"]
    vllm_online = source["vllm_online_parity_protocol"]
    share_bundle = source["share_bundle_manifest"]
    sha = f"sha256:{summary.get('tarball_sha256')}"

    compact_by_name = {
        "tarball and checksum agree": (
            f"tarball, checksum, package manifest, tarball validation and "
            f"receiver smoke all agree on `{sha}`; bytes `{summary.get('tarball_size_bytes')}`."
        ),
        "share bundle package manifest ready": (
            f"package ready `{package.get('ready')}`, files `{package.get('file_count')}`, "
            f"source records `{package.get('source_manifest_records')}`."
        ),
        "tarball-mode validation ready": (
            f"tarball validation `{_checks_marker(validation)}`, required failures "
            f"`{validation.get('required_failures')}`, tar members `{validation.get('tar_members')}`."
        ),
        "receiver smoke validation ready": (
            f"receiver smoke `{_checks_marker(receiver)}`, receiver_smoke_ready "
            f"`{receiver.get('receiver_smoke_ready')}`."
        ),
        "extracted-only validation ready": (
            f"extracted-only `{_checks_marker(extracted)}`, required failures "
            f"`{extracted.get('required_failures')}`."
        ),
        "external standalone validation ready": (
            f"standalone `{_checks_marker(external)}`, required failures "
            f"`{external.get('required_failures')}`."
        ),
        "release seal remains outside tarball": (
            f"forbidden tarball members `{summary.get('forbidden_tarball_members')}`; "
            f"adjacent artifacts `{summary.get('adjacent_artifacts_total')}`."
        ),
        "full audit and final readiness are green": (
            f"audit ok; final readiness `{_checks_marker(final)}`, required failures "
            f"`{final.get('required_failures')}`."
        ),
        "objective is share-ready but not complete": (
            f"share_ready_with_documented_caveats "
            f"`{objective.get('share_ready_with_documented_caveats')}`; "
            f"goal_complete `{objective.get('goal_complete')}`."
        ),
        "checkpoint watchlist remains active": (
            f"checkpoint `{checkpoint.get('checkpoint_phase')}`; completion_allowed_now "
            f"`{checkpoint.get('completion_allowed_now')}`; blockers "
            f"`{checkpoint.get('completion_blockers')}`."
        ),
        "core audit inventory gates are current": (
            f"preflight checks `{preflight.get('total_checks')}`, manifest records "
            f"`{manifest.get('total_records')}`, repro commands "
            f"`{repro.get('commands_total')}`, evidence-query host/portable "
            f"`{evidence_query.get('host_pass_lines')}/{evidence_query.get('portable_pass_lines')}`."
        ),
        "runtime and optimization contracts are green": (
            f"runtime image `{_checks_marker(runtime_image)}`, comparison "
            f"`{_checks_marker(runtime)}`, SGLang lock `{_checks_marker(sglang)}`, "
            f"vLLM lock `{_checks_marker(vllm)}`; headline "
            f"`{runtime.get('allowed_cross_runtime_headline')}`."
        ),
        "vLLM c8 caveat is preserved": (
            f"vLLM online protocol `{_checks_marker(vllm_online)}`; "
            f"online_parity_proven `{vllm_online.get('online_parity_proven')}`."
        ),
        "chart source and share bundle gates are green": (
            f"chart source `{_checks_marker(chart)}`, byte-exact files "
            f"`{chart.get('byte_exact_files')}`; share bundle records "
            f"`{share_bundle.get('records_total')}`."
        ),
    }
    return compact_by_name.get(name, check.get("evidence", ""))


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    source = payload["source_summaries"]
    lines: list[str] = [
        "# Qwen3.5-Omni Share Release Seal",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份 release seal 是便捷 tarball 外侧的伴随封口证据，用来在发送前一次性确认",
        "tarball、checksum、package manifest、tarball-mode validation、receiver smoke、",
        "extracted-only validation、standalone validation、evidence-query host/portable smoke、",
        "final readiness、final completion audit 和必须携带 caveat。",
        "它描述 tarball 自身，因此不是 tarball 内成员。",
        "",
        "## 1. Seal 状态",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| Send decision | `{summary['send_decision']}` |",
        f"| Tarball SHA-256 | `sha256:{summary['tarball_sha256']}` |",
        f"| Tarball bytes | {summary['tarball_size_bytes']} |",
        f"| Package files | {summary['package_file_count']} |",
        f"| Manifest records | {summary['manifest_records']} |",
        f"| Final readiness checks | {summary['final_readiness_checks']} |",
        f"| Repro commands | {summary['repro_commands_total']} |",
        (
            f"| Adjacent artifacts | {summary['adjacent_artifacts_hashed']}/"
            f"{summary['adjacent_artifacts_total']} hashed; "
            f"{summary['adjacent_artifacts_self_omitted']} self-reference omitted |"
        ),
        f"| Goal complete | {summary['goal_complete']} |",
        f"| Completion allowed now | {summary['completion_allowed_now']} |",
        f"| Completion blockers | `{', '.join(summary['completion_blockers'])}` |",
        "",
        "## 2. 发送文件",
        "",
        "| 文件 | 用途 |",
        "| --- | --- |",
    ]
    send_rows = [
        (payload["tarball_relative_path"], "便捷发送包"),
        (payload["checksum_relative_path"], "`sha256sum -c` 校验输入"),
        (
            "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
            "tarball 文件数、source manifest、tarball SHA-256",
        ),
        (
            "results/qwen35_report_audit_20260619/share_package_validation.json",
            "tarball-mode validation 机器证据",
        ),
        (
            "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
            "receiver smoke 机器证据",
        ),
        (
            "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
            "手动 extracted-only validation 机器证据",
        ),
        (
            "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
            "干净目录 standalone validation 机器证据",
        ),
        (
            "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json",
            "evidence-query host/portable 只读烟测摘要；绑定当前 tarball SHA-256",
        ),
        (
            "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.summary.out",
            "host 仓库态 evidence-query smoke 控制台摘要",
        ),
        (
            "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.query.out",
            "host 仓库态 evidence-query 原始查询输出",
        ),
        (
            "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.summary.out",
            "portable 解包态 evidence-query smoke 控制台摘要",
        ),
        (
            "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.query.out",
            "portable 解包态 evidence-query 原始查询输出",
        ),
        (
            "results/qwen35_report_audit_20260619/final_completion_audit.json",
            "最终 completion audit 机器证据；更新后的目标取消 6.21 晚间等待，completion_allowed_now=true 才能标记完成",
        ),
        (payload["seal_json"], "本 seal 的机器可读版本"),
        (
            "benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md",
            "最终 completion audit 中文可读版本",
        ),
        (payload["seal_report"], "本 seal 的中文可读版本"),
    ]
    for path, purpose in send_rows:
        lines.append(f"| `{path}` | {purpose} |")

    lines.extend(
        [
            "",
            "## 3. 关键 Gate 摘要",
            "",
            "| Gate | Evidence |",
            "| --- | --- |",
            (
                "| final readiness | "
                f"`ready={source['final_readiness'].get('ready')}`, "
                f"`checks={source['final_readiness'].get('checks_passed')}/"
                f"{source['final_readiness'].get('checks_total')}`, "
                f"`required_failures={source['final_readiness'].get('required_failures')}` |"
            ),
            (
                "| final completion audit | "
                f"`ready={source['final_completion_audit'].get('ready')}`, "
                f"`completion_allowed_now="
                f"{source['final_completion_audit'].get('completion_allowed_now')}`, "
                f"`recommendation="
                f"{source['final_completion_audit'].get('goal_update_recommendation')}` |"
            ),
            (
                "| share package validation | "
                f"`ready={source['share_package_validation'].get('ready')}`, "
                f"`checks={source['share_package_validation'].get('checks_passed')}/"
                f"{source['share_package_validation'].get('checks_total')}`, "
                f"`required_failures={source['share_package_validation'].get('required_failures')}` |"
            ),
            (
                "| receiver smoke | "
                f"`ready={source['receiver_smoke_validation'].get('ready')}`, "
                f"`receiver_smoke_ready={source['receiver_smoke_validation'].get('receiver_smoke_ready')}` |"
            ),
            (
                "| extracted-only validation | "
                f"`ready={source['extracted_validation'].get('ready')}`, "
                f"`checks={source['extracted_validation'].get('checks_passed')}/"
                f"{source['extracted_validation'].get('checks_total')}` |"
            ),
            (
                "| external standalone validation | "
                f"`ready={source['external_standalone_validation'].get('ready')}`, "
                f"`checks={source['external_standalone_validation'].get('checks_passed')}/"
                f"{source['external_standalone_validation'].get('checks_total')}` |"
            ),
            (
                "| evidence-query smoke | "
                f"`ready={source['evidence_query_smoke'].get('ready')}`, "
                f"`host_pass={source['evidence_query_smoke'].get('host_pass_lines')}`, "
                f"`portable_pass={source['evidence_query_smoke'].get('portable_pass_lines')}` |"
            ),
            (
                "| runtime fairness | "
                f"`{source['runtime_comparison_contract'].get('allowed_cross_runtime_headline')}`, "
                f"`{source['runtime_comparison_contract'].get('vllm_c8_contract')}` |"
            ),
            (
                "| vLLM c=8 caveat | "
                f"`online_parity_proven={source['vllm_online_parity_protocol'].get('online_parity_proven')}` |"
            ),
            (
                "| chart source consistency | "
                f"`ready={source['chart_source_consistency'].get('ready')}`, "
                f"`byte_exact_files={source['chart_source_consistency'].get('byte_exact_files')}` |"
            ),
            "",
            "## 4. Machine Checks",
            "",
            "| Status | Required | Check | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        required = "yes" if check["required"] else "no"
        evidence = _table_text(_compact_machine_check_evidence(check, payload))
        lines.append(
            f"| {check['status']} | {required} | {check['name']} | {evidence} |"
        )

    lines.extend(
        [
            "",
            "## 5. 接收方第一组命令",
            "",
            "```bash",
            "HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"",
            "SMOKE_DIR=\"${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_release_seal}\"",
            "EXTRACT_DIR=\"${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_release_seal}\"",
            "STANDALONE_DIR=\"${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_release_seal}\"",
            "cd \"$HOST_REPO\"",
            "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
            "python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \\",
            "  --root \"$HOST_REPO\" \\",
            "  --strict \\",
            "  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \\",
            "  --json-output results/qwen35_report_audit_20260619/share_release_seal.json",
            "```",
            "",
            "## 6. 必须携带的 Caveat",
            "",
        ]
    )
    for caveat in payload["caveats_that_must_travel"]:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## 7. 不可提前升级",
            "",
            "- 不把当前 vLLM c=8 prebuild w4 写成 online serving parity。",
            "- 不把 c=16 写成默认推荐服务点。",
            "- 不把 official SeedTTS full-set 写成 headline benchmark。",
            "- 不在 final completion audit 显示 `completion_allowed_now=true` 前把长线目标标记 complete。",
            "",
            "这份 seal 证明当前 tarball 和伴随验证证据可以带 caveat 分享；是否完成由 final completion audit 的证据门决定。",
            "",
        ]
    )
    return "\n".join(lines)


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Share Release Seal\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Tarball SHA-256 | sha256:{summary['tarball_sha256']} |")
    print(f"| Send decision | {summary['send_decision']} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the adjacent Qwen3.5-Omni share release seal."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = build_release_seal(
        args.root,
        output=args.output,
        json_output=args.json_output,
        tarball=args.tarball,
    )
    print_markdown(payload)
    print(
        "Share release seal written: "
        f"{payload['seal_report']} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
