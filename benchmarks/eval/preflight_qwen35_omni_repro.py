# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for reproducing the Qwen3.5-Omni performance report."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SGLANG_IMAGE = "frankleeeee/sglang-omni:dev"
VLLM_IMAGE = (
    "tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/"
    "dashllm:cuda129_cp312_test_vl_13589"
)
SGLANG_IMAGE_ID = "sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84"


@dataclass
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


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.strip()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _container_to_host(path: str, host_mount: Path) -> Path:
    if path.startswith("/myapp/"):
        return host_mount / path.removeprefix("/myapp/")
    return Path(path)


def _exists_check(name: str, path: Path, *, required: bool = True) -> Check:
    if path.exists():
        kind = "directory" if path.is_dir() else "file"
        return Check(name, "PASS", f"{kind} exists: {path}", required)
    return Check(name, "FAIL" if required else "WARN", f"missing: {path}", required)


def _executable_check(name: str, path: Path, *, required: bool = True) -> Check:
    if path.exists() and path.is_file() and path.stat().st_mode & 0o111:
        return Check(name, "PASS", f"executable: {path}", required)
    if path.exists():
        return Check(
            name,
            "FAIL" if required else "WARN",
            f"exists but is not executable: {path}",
            required,
        )
    return Check(name, "FAIL" if required else "WARN", f"missing: {path}", required)


def _file_contains_check(
    name: str,
    path: Path,
    needle: str,
    *,
    required: bool = True,
) -> Check:
    if not path.is_file():
        return Check(name, "FAIL" if required else "WARN", f"missing: {path}", required)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return Check(
            name,
            "FAIL" if required else "WARN",
            f"could not read {path}: {exc}",
            required,
        )
    if needle in text:
        return Check(name, "PASS", f"{path} contains {needle!r}", required)
    return Check(
        name,
        "FAIL" if required else "WARN",
        f"{path} does not contain {needle!r}",
        required,
    )


def _docker_image_check(image: str, *, expected_id: str | None = None) -> Check:
    if shutil.which("docker") is None:
        return Check("docker image: " + image, "WARN", "docker CLI not found", False)
    code, output = _run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}} {{.Created}}"],
        timeout=15,
    )
    if code != 0:
        return Check("docker image: " + image, "WARN", output or "image not found", False)
    image_id = output.split()[0] if output else ""
    if expected_id is not None and image_id != expected_id:
        return Check(
            "docker image: " + image,
            "WARN",
            f"present but id differs: {output}",
            False,
        )
    return Check("docker image: " + image, "PASS", output, False)


def _gpu_check(min_gpus: int) -> Check:
    if shutil.which("nvidia-smi") is None:
        return Check("GPU inventory", "WARN", "nvidia-smi not found", False)
    code, output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        timeout=15,
    )
    if code != 0:
        return Check("GPU inventory", "WARN", output or "nvidia-smi failed", False)
    rows = [line for line in output.splitlines() if line.strip()]
    status = "PASS" if len(rows) >= min_gpus else "WARN"
    first = rows[0] if rows else "no GPUs reported"
    return Check(
        "GPU inventory",
        status,
        f"{len(rows)} GPU(s), first={first}, required_for_report={min_gpus}",
        False,
    )


def _claims_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("claim verifier JSON", "FAIL", f"missing or invalid: {path}")
    passed = bool(payload.get("passed"))
    total = payload.get("total_checks")
    failed = payload.get("failed_checks")
    status = "PASS" if passed and failed == 0 else "FAIL"
    return Check("claim verifier JSON", status, f"checks={total}, failed={failed}: {path}")


def _tables_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("table/artifact audit JSON", "FAIL", f"missing or invalid: {path}")
    missing = 0
    groups: list[str] = []
    for row in payload.get("artifact_status", []):
        row_missing = int(row.get("missing") or 0)
        missing += row_missing
        groups.append(
            f"{row.get('group')}={row.get('present')}/{row.get('total')}"
        )
    status = "PASS" if missing == 0 else "FAIL"
    return Check("table/artifact audit JSON", status, "; ".join(groups) + f": {path}")


def _manifest_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("evidence manifest JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    missing = int(summary.get("missing_records") or 0)
    total = int(summary.get("total_records") or 0)
    files = int(summary.get("file_records") or 0)
    status = "PASS" if missing == 0 and total >= 180 and files >= 178 else "FAIL"
    return Check(
        "evidence manifest JSON",
        status,
        f"records={total}, files={files}, missing={missing}: {path}",
    )


def _vllm_diagnosis_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("vLLM admission diagnosis JSON", "FAIL", f"missing or invalid: {path}")
    rows = payload.get("rows", [])
    labels = [f"{row.get('label')}={row.get('diagnosis')}" for row in rows]
    if not rows:
        return Check("vLLM admission diagnosis JSON", "FAIL", f"no rows: {path}")
    return Check("vLLM admission diagnosis JSON", "PASS", "; ".join(labels) + f": {path}")


def _vllm_optimization_lock_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("vLLM optimization lock JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    status = (
        "PASS"
        if ready and total >= 22 and passed == total and required_failures == 0
        else "FAIL"
    )
    return Check(
        "vLLM optimization lock JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}: {path}"
        ),
    )


def _vllm_online_parity_protocol_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("vLLM online parity protocol JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    current_package_safe = bool(summary.get("current_package_safe"))
    online_parity_proven = bool(summary.get("online_parity_proven"))
    status = (
        "PASS"
        if ready
        and total >= 17
        and passed == total
        and required_failures == 0
        and current_package_safe
        and not online_parity_proven
        else "FAIL"
    )
    return Check(
        "vLLM online parity protocol JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, "
            f"current_package_safe={current_package_safe}, "
            f"online_parity_proven={online_parity_proven}: {path}"
        ),
    )


def _runtime_image_contract_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("runtime image contract JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    sglang_scope = str(summary.get("sglang_scope") or "")
    vllm_strict_scope = str(summary.get("vllm_strict_scope") or "")
    vllm_c8_scope = str(summary.get("vllm_c8_scope") or "")
    status = (
        "PASS"
        if ready
        and total >= 12
        and passed == total
        and required_failures == 0
        and "c4-c8" in sglang_scope
        and "c4" in vllm_strict_scope
        and "offline diagnostic" in vllm_c8_scope
        else "FAIL"
    )
    return Check(
        "runtime image contract JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, "
            f"sglang_scope={sglang_scope}, "
            f"vllm_strict_scope={vllm_strict_scope}, "
            f"vllm_c8_scope={vllm_c8_scope}: {path}"
        ),
    )


def _rerun_acceptance_contract_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("rerun acceptance contract JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    rules_total = int(summary.get("rules_total") or 0)
    return_evidence_files = int(summary.get("return_evidence_files") or 0)
    command_rows = int(summary.get("return_evidence_command_rows") or 0)
    command_missing = int(summary.get("return_evidence_command_missing") or 0)
    command_file_gaps = int(summary.get("return_evidence_command_file_gaps") or 0)
    replacement_scope = str(summary.get("replacement_scope") or "")
    status = (
        "PASS"
        if ready
        and total >= 16
        and passed == total
        and required_failures == 0
        and rules_total >= 18
        and return_evidence_files >= 34
        and command_rows >= 27
        and command_missing == 0
        and command_file_gaps == 0
        and "same hardware/image/model/data" in replacement_scope
        else "FAIL"
    )
    return Check(
        "rerun acceptance contract JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, rules={rules_total}, "
            f"return_evidence={return_evidence_files}, "
            f"command_rows={command_rows}, command_missing={command_missing}, "
            f"command_file_gaps={command_file_gaps}, "
            f"replacement_scope={replacement_scope}: {path}"
        ),
    )


def _audit_green_or_in_progress(payload: dict[str, Any]) -> bool:
    if bool(payload.get("ok")):
        return True
    if not bool(payload.get("in_progress")):
        return False
    steps = payload.get("steps", [])
    return isinstance(steps, list) and all(
        isinstance(step, dict) and bool(step.get("ok")) for step in steps
    )


def _final_checkpoint_watchlist_check(
    path: Path, audit_summary_path: Path | None = None
) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("final checkpoint watchlist JSON", "FAIL", f"missing or invalid: {path}")
    audit_summary = _load_json(audit_summary_path) if audit_summary_path else None
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    watch_items = int(summary.get("watch_items_total") or 0)
    share_ready = bool(summary.get("share_ready_with_documented_caveats"))
    completion_evidence_ready = bool(summary.get("final_completion_evidence_ready"))
    goal_complete = bool(summary.get("goal_complete"))
    completion_allowed = bool(summary.get("completion_allowed_now"))
    checkpoint_phase = str(summary.get("checkpoint_phase") or "")
    completion_gate_ok = (
        checkpoint_phase == "completion_audit_ready"
        and completion_allowed
    )
    ready_status_ok = (
        ready
        and total >= 24
        and passed == total
        and required_failures == 0
        and completion_evidence_ready
    )
    pending_in_full_audit = (
        isinstance(audit_summary, dict)
        and _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and total >= 23
        and required_failures <= 1
        and share_ready
        and not completion_evidence_ready
    )
    status = (
        "PASS"
        if (ready_status_ok or pending_in_full_audit)
        and watch_items >= 7
        and share_ready
        and not goal_complete
        and completion_gate_ok
        else "FAIL"
    )
    return Check(
        "final checkpoint watchlist JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, watch_items={watch_items}, "
            f"share_ready={share_ready}, completion_evidence_ready="
            f"{completion_evidence_ready}, goal_complete={goal_complete}, "
            f"checkpoint_phase={checkpoint_phase}, completion_allowed="
            f"{completion_allowed}, pending_in_full_audit="
            f"{pending_in_full_audit}: {path}"
        ),
    )


def _stage_latency_budget_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("stage latency budget JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    sglang_rows = int(summary.get("sglang_budget_rows") or 0)
    synthetic_rows = int(summary.get("synthetic_budget_rows") or 0)
    vllm_rows = int(summary.get("vllm_budget_rows") or 0)
    status = (
        "PASS"
        if ready
        and total >= 12
        and passed == total
        and required_failures == 0
        and sglang_rows >= 5
        and synthetic_rows >= 6
        and vllm_rows >= 4
        else "FAIL"
    )
    return Check(
        "stage latency budget JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, sglang_rows={sglang_rows}, "
            f"synthetic_rows={synthetic_rows}, vllm_rows={vllm_rows}: {path}"
        ),
    )


def _stage_boundary_ledger_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check(
            "stage boundary bottleneck ledger JSON",
            "FAIL",
            f"missing or invalid: {path}",
        )
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    ledger_rows = int(summary.get("ledger_rows") or 0)
    status = (
        "PASS"
        if ready
        and total >= 10
        and passed == total
        and required_failures == 0
        and ledger_rows >= 37
        else "FAIL"
    )
    return Check(
        "stage boundary bottleneck ledger JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, ledger_rows={ledger_rows}: {path}"
        ),
    )


def _sglang_optimization_lock_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("SGLang optimization lock JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    status = (
        "PASS"
        if ready and total >= 26 and passed == total and required_failures == 0
        else "FAIL"
    )
    return Check(
        "SGLang optimization lock JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}: {path}"
        ),
    )


def _vllm_prebuild_result_check(
    path: Path,
    *,
    label: str,
    expected_workers: int | None = None,
) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check(
            f"vLLM {label} result JSON",
            "FAIL",
            f"missing or invalid: {path}",
        )
    config = payload.get("config", {})
    speed = payload.get("speed", {})
    summary = payload.get("summary", {})
    completed = int(speed.get("completed_requests") or 0)
    failed = int(summary.get("failed") or speed.get("failed_requests") or 0)
    prebuilt = bool(config.get("prebuild_prompts"))
    workers = config.get("prebuild_workers")
    workers_ok = expected_workers is None or workers == expected_workers
    status = "PASS" if prebuilt and workers_ok and completed >= 50 and failed == 0 else "FAIL"
    return Check(
        f"vLLM {label} result JSON",
        status,
        (
            f"prebuild_prompts={prebuilt}, prebuild_workers={workers}, "
            f"completed={completed}, "
            f"failed={failed}: {path}"
        ),
    )


def _stage_interaction_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("stage interaction summary JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    required_flags = [
        "sglang_talker_to_code2wav_healthy",
        "sglang_code2wav_decode_not_bottleneck",
        "vllm_original_c8_prompt_feed_limited",
        "preprocessing_parallelism_regresses",
    ]
    missing_flags = [flag for flag in required_flags if not summary.get(flag)]
    total = int(summary.get("total_interactions") or 0)
    status = "PASS" if not missing_flags and total >= 30 else "FAIL"
    evidence = (
        f"interactions={total}, "
        + ", ".join(f"{flag}={summary.get(flag)}" for flag in required_flags)
        + f": {path}"
    )
    if missing_flags:
        evidence += "; missing_true=" + ",".join(missing_flags)
    return Check("stage interaction summary JSON", status, evidence)


def _headline_scorecard_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("headline scorecard JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    passed = int(summary.get("checks_passed") or 0)
    total = int(summary.get("checks_total") or 0)
    status = "PASS" if ready and total > 0 and passed == total else "FAIL"
    return Check(
        "headline scorecard JSON",
        status,
        f"ready={ready}, checks={passed}/{total}: {path}",
    )


def _chart_pack_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("share chart pack JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    csv_files = int(summary.get("csv_files") or 0)
    svg_files = int(summary.get("svg_files") or 0)
    generated_files = int(summary.get("generated_files") or 0)
    status = (
        "PASS"
        if ready and csv_files >= 7 and svg_files >= 7 and generated_files >= 14
        else "FAIL"
    )
    return Check(
        "share chart pack JSON",
        status,
        (
            f"ready={ready}, csv={csv_files}, svg={svg_files}, "
            f"generated={generated_files}: {path}"
        ),
    )


def _chart_source_consistency_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("chart source consistency JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    total = int(summary.get("checks_total") or 0)
    passed = int(summary.get("checks_passed") or 0)
    required_failures = int(summary.get("required_failures") or 0)
    csv_files = int(summary.get("csv_files_checked") or 0)
    svg_files = int(summary.get("svg_files_checked") or 0)
    byte_exact = int(summary.get("byte_exact_files") or 0)
    status = (
        "PASS"
        if ready
        and total >= 8
        and passed == total
        and required_failures == 0
        and csv_files >= 7
        and svg_files >= 7
        and byte_exact >= 14
        else "FAIL"
    )
    return Check(
        "chart source consistency JSON",
        status,
        (
            f"ready={ready}, checks={passed}/{total}, "
            f"required_failures={required_failures}, byte_exact={byte_exact}: {path}"
        ),
    )


def _acceptance_matrix_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("acceptance matrix JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    passed = int(summary.get("rows_passed") or 0)
    total = int(summary.get("rows_total") or 0)
    failed = int(summary.get("rows_failed") or 0)
    status = "PASS" if ready and failed == 0 and passed == total and total >= 17 else "FAIL"
    return Check(
        "acceptance matrix JSON",
        status,
        f"ready={ready}, rows={passed}/{total}, failed={failed}: {path}",
    )


def _confidence_ledger_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check("confidence ledger JSON", "FAIL", f"missing or invalid: {path}")
    summary = payload.get("summary", {})
    ready = bool(summary.get("ready"))
    passed = int(summary.get("entries_passed") or 0)
    total = int(summary.get("entries_total") or 0)
    failed = int(summary.get("entries_failed") or 0)
    high = int(summary.get("high_confidence_claims") or 0)
    medium = int(summary.get("medium_confidence_boundaries") or 0)
    unsupported = int(summary.get("unsupported_claims") or 0)
    status = (
        "PASS"
        if ready
        and failed == 0
        and passed == total
        and total >= 12
        and high >= 9
        and medium >= 3
        and unsupported == 0
        else "FAIL"
    )
    return Check(
        "confidence ledger JSON",
        status,
        (
            f"ready={ready}, entries={passed}/{total}, failed={failed}, "
            f"high={high}, medium={medium}, unsupported={unsupported}: {path}"
        ),
    )


def _repro_command_manifest_check(path: Path) -> Check:
    payload = _load_json(path)
    if payload is None:
        return Check(
            "reproduction command manifest JSON",
            "FAIL",
            f"missing or invalid: {path}",
        )
    summary = payload.get("summary", {})
    checks = summary.get("checks", {})
    gates = summary.get("expected_gates", {})
    commands_total = int(summary.get("commands_total") or 0)
    ready = bool(summary.get("ready"))
    required_ids = bool(summary.get("required_command_ids_present"))
    coverage_total = int(gates.get("coverage", {}).get("total") or 0)
    preflight_checks = int(gates.get("preflight", {}).get("total_checks") or 0)
    manifest_gate = gates.get("manifest", {})
    manifest_records = int(
        manifest_gate.get("min_total_records")
        or manifest_gate.get("total_records")
        or 0
    )
    status = (
        "PASS"
        if ready
        and required_ids
        and commands_total >= 60
        and coverage_total >= 34
        and preflight_checks >= 62
        and manifest_records >= 180
        and bool(checks.get("synthetic_short_text_shape"))
        and bool(checks.get("synthetic_long_text_shape"))
        and bool(checks.get("rerun_acceptance_contract_ready"))
        and bool(checks.get("command_reference_hygiene_ready"))
        and bool(checks.get("tail_confidence_appendix_ready"))
        else "FAIL"
    )
    return Check(
        "reproduction command manifest JSON",
        status,
        (
            f"ready={ready}, commands={commands_total}, "
            f"required_ids={required_ids}, coverage_total={coverage_total}, "
            f"preflight_checks={preflight_checks}, "
            f"manifest_min_records={manifest_records}: {path}"
        ),
    )


def build_checks(args: argparse.Namespace) -> list[Check]:
    root = args.root.resolve()
    host_mount = args.host_mount.resolve()
    model_host = _container_to_host(args.model_container_path, host_mount)
    videoamme_host = _container_to_host(args.videoamme_container_path, host_mount)

    checks: list[Check] = [
        _exists_check("workspace root", root),
        _exists_check("Qwen3.5-Omni model path", model_host),
        _exists_check("Video-AMME cache", videoamme_host),
        _executable_check(
            "SGLang launch script",
            root / "examples/launch_qwen35_omni_speech_server_container.sh",
        ),
        _executable_check(
            "vLLM offline shell wrapper",
            root
            / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh",
        ),
        _exists_check(
            "vLLM offline Python runner",
            root / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py",
        ),
        _file_contains_check(
            "vLLM prebuilt-prompt runner flag",
            root
            / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py",
            "--prebuild-prompts",
        ),
        _file_contains_check(
            "vLLM wrapper EXTRA_ARGS passthrough",
            root
            / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh",
            "EXTRA_ARGS",
        ),
        _vllm_prebuild_result_check(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            label="c8 prebuilt-prompt w1",
            expected_workers=1,
        ),
        _exists_check(
            "vLLM c8 prebuilt-prompt w1 run log",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
        ),
        _vllm_prebuild_result_check(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            label="c8 prebuilt-prompt w4",
            expected_workers=4,
        ),
        _exists_check(
            "vLLM c8 prebuilt-prompt w4 run log",
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
        ),
        _tables_check(root / "results/qwen35_report_audit_20260619/tables_summary.json"),
        _claims_check(root / "results/qwen35_report_audit_20260619/claims_verification.json"),
        _vllm_diagnosis_check(
            root / "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"
        ),
        _sglang_optimization_lock_check(
            root / "results/qwen35_report_audit_20260619/sglang_optimization_lock.json"
        ),
        _vllm_optimization_lock_check(
            root / "results/qwen35_report_audit_20260619/vllm_optimization_lock.json"
        ),
        _vllm_online_parity_protocol_check(
            root
            / "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json"
        ),
        _runtime_image_contract_check(
            root / "results/qwen35_report_audit_20260619/runtime_image_contract.json"
        ),
        _rerun_acceptance_contract_check(
            root
            / "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json"
        ),
        _final_checkpoint_watchlist_check(
            root
            / "results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json",
            root / "results/qwen35_report_audit_20260619/audit_run_summary.json",
        ),
        _stage_latency_budget_check(
            root / "results/qwen35_report_audit_20260619/stage_latency_budget.json"
        ),
        _stage_boundary_ledger_check(
            root
            / "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json"
        ),
        _stage_interaction_check(
            root / "results/qwen35_report_audit_20260619/stage_interaction_summary.json"
        ),
        _headline_scorecard_check(
            root / "results/qwen35_report_audit_20260619/headline_scorecard.json"
        ),
        _chart_pack_check(
            root
            / "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json"
        ),
        _chart_source_consistency_check(
            root / "results/qwen35_report_audit_20260619/chart_source_consistency.json"
        ),
        _acceptance_matrix_check(
            root / "results/qwen35_report_audit_20260619/acceptance_matrix.json"
        ),
        _confidence_ledger_check(
            root / "results/qwen35_report_audit_20260619/confidence_ledger.json"
        ),
        _repro_command_manifest_check(
            root / "results/qwen35_report_audit_20260619/repro_command_manifest.json"
        ),
        _exists_check(
            "Video-AMME SeedTTS-compatible meta",
            root / "results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst",
        ),
        _exists_check(
            "Video-AMME SeedTTS-compatible meta summary",
            root
            / "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json",
        ),
        _manifest_check(root / "results/qwen35_report_audit_20260619/manifest.json"),
        _exists_check(
            "environment snapshot JSON",
            root / "results/qwen35_report_audit_20260619/environment_snapshot.json",
        ),
        _file_contains_check(
            "Chinese share package index",
            root
            / "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
            "机器证据入口",
        ),
        _file_contains_check(
            "Chinese share deck outline",
            root
            / "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md",
            "Qwen3.5-Omni 性能分析分享 Deck 提纲",
        ),
        _file_contains_check(
            "Chinese requirement evidence map",
            root
            / "benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md",
            "Qwen3.5-Omni 原始需求-证据映射表",
        ),
        _file_contains_check(
            "Chinese pressure matrix",
            root
            / "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
            "Qwen3.5-Omni 压力条件总表",
        ),
        _file_contains_check(
            "Chinese metric source map",
            root
            / "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
            "Qwen3.5-Omni 数字来源索引",
        ),
        _file_contains_check(
            "Chinese stage metric dictionary",
            root
            / "benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md",
            "Qwen3.5-Omni Stage 指标字典",
        ),
        _file_contains_check(
            "Chinese defense Q&A",
            root
            / "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md",
            "Qwen3.5-Omni 性能报告答辩 Q&A",
        ),
        _file_contains_check(
            "Chinese optimization playbook",
            root
            / "benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md",
            "Qwen3.5-Omni 性能优化 Playbook",
        ),
        _exists_check(
            "Chinese reproduction checklist",
            root
            / "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
        ),
        _file_contains_check(
            "Chinese external handoff runbook",
            root
            / "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
            "Qwen3.5-Omni 外部复现 Handoff Runbook",
        ),
        _file_contains_check(
            "Chinese collaborator rerun validation sheet",
            root
            / "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
            "Qwen3.5-Omni 合作方复跑验收表",
        ),
        _file_contains_check(
            "Chinese final share delivery note",
            root
            / "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
            "Qwen3.5-Omni 最终分享交付说明",
        ),
        _file_contains_check(
            "Chinese one-page scorecard",
            root
            / "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
            "Qwen3.5-Omni 一页式核心数字 Scorecard",
        ),
        _file_contains_check(
            "Chinese runtime comparison contract",
            root
            / "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
            "Qwen3.5-Omni Runtime 公平对比合同",
        ),
        _file_contains_check(
            "Chinese runtime image contract",
            root
            / "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
            "Qwen3.5-Omni Runtime Image Contract",
        ),
        _file_contains_check(
            "Chinese rerun acceptance contract",
            root
            / "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
            "Qwen3.5-Omni 复跑验收阈值合同",
        ),
        _file_contains_check(
            "Chinese SGLang optimization lock",
            root
            / "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
            "Qwen3.5-Omni SGLang 优化锁定矩阵",
        ),
        _file_contains_check(
            "Chinese vLLM optimization lock",
            root
            / "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
            "Qwen3.5-Omni vLLM 优化锁定矩阵",
        ),
        _file_contains_check(
            "Chinese vLLM online parity protocol",
            root
            / "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
            "Qwen3.5-Omni vLLM c=8 Online Parity 升级协议",
        ),
        _file_contains_check(
            "Chinese final checkpoint watchlist",
            root
            / "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
            "Qwen3.5-Omni 最终 Checkpoint Watchlist",
        ),
        _file_contains_check(
            "Chinese stage latency budget",
            root
            / "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
            "Qwen3.5-Omni Stage Latency Budget",
        ),
        _file_contains_check(
            "Chinese stage boundary bottleneck ledger",
            root
            / "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            "Qwen3.5-Omni Stage Boundary Bottleneck Ledger",
        ),
        _file_contains_check(
            "Chinese stage causal graph",
            root
            / "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
            "Qwen3.5-Omni Stage 因果图",
        ),
        _file_contains_check(
            "Chinese caveat adjudication matrix",
            root
            / "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
            "Qwen3.5-Omni Caveat 裁决矩阵",
        ),
    ]
    checks.extend(
        [
            _docker_image_check(SGLANG_IMAGE, expected_id=SGLANG_IMAGE_ID),
            _docker_image_check(VLLM_IMAGE),
            _gpu_check(args.min_gpus),
            _exists_check(
                "local Whisper large-v3 cache",
                Path(args.whisper_cache),
                required=False,
            ),
        ]
    )
    return checks


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def print_markdown(checks: list[Check]) -> None:
    print("| Status | Required | Check | Evidence |")
    print("| --- | --- | --- | --- |")
    for check in checks:
        required = "yes" if check.required else "no"
        evidence = check.evidence.replace("|", "\\|")
        print(f"| {check.status} | {required} | {check.name} | {evidence} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight local prerequisites for Qwen3.5-Omni report reproduction."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--host-mount", type=Path, default=Path("/home/gangouyu"))
    parser.add_argument(
        "--model-container-path",
        default="/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315",
    )
    parser.add_argument("--videoamme-container-path", default="/myapp/data/videoamme")
    parser.add_argument("--whisper-cache", default="/root/.cache/whisper/large-v3.pt")
    parser.add_argument("--min-gpus", type=int, default=8)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any required check fails.",
    )
    args = parser.parse_args()

    checks = build_checks(args)
    required_failures = [check for check in checks if check.required and check.status == "FAIL"]
    warnings = [check for check in checks if check.status == "WARN"]
    payload = {
        "root": str(args.root.resolve()),
        "summary": {
            "total_checks": len(checks),
            "required_failures": len(required_failures),
            "warnings": len(warnings),
            "ready": len(required_failures) == 0,
        },
        "checks": [check.to_dict() for check in checks],
    }

    print_markdown(checks)
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = args.root.resolve() / output
        _save_json(payload, output)
    if args.strict and required_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
