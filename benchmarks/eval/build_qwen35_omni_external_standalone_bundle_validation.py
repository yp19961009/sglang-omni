# SPDX-License-Identifier: Apache-2.0
"""Validate the share tarball through an extracted, repository-independent path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_TARBALL = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"
DEFAULT_OUTPUT = AUDIT_DIR / "share_package_external_standalone_validation.json"
DEFAULT_WORK_DIR = Path("/tmp/qwen35_omni_external_standalone_bundle_validation")
DEFAULT_ARC_PREFIX = "qwen35_omni_share_bundle_20260621"


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


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _check(
    checks: list[Check],
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
    checks.append(Check(name, status, evidence, required))


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_reset_work_dir(work_dir: Path) -> None:
    resolved = work_dir.resolve()
    if str(resolved) in {"/", "/tmp", "/var/tmp"}:
        raise ValueError(f"Refusing to clear broad work directory: {resolved}")
    if not str(resolved).startswith(("/tmp/", "/var/tmp/")):
        raise ValueError(f"Standalone validation work dir must be under /tmp: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)
    resolved.mkdir(parents=True, exist_ok=True)


def _safe_extract_tarball(tarball: Path, output_dir: Path) -> tuple[list[str], str | None]:
    unsafe: list[str] = []
    try:
        output_root = output_dir.resolve()
        with tarfile.open(tarball, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                name = member.name
                if name.startswith("/") or ".." in Path(name).parts:
                    unsafe.append(name)
                    continue
                try:
                    (output_root / name).resolve().relative_to(output_root)
                except ValueError:
                    unsafe.append(name)
                    continue
                if member.islnk() or member.issym() or not (
                    member.isfile() or member.isdir()
                ):
                    unsafe.append(name)
            if unsafe:
                return sorted(set(unsafe)), None
            tf.extractall(output_root, members=members)
    except Exception as exc:
        return unsafe, str(exc)
    return [], None


def _check_evidence(payload: dict[str, Any], name: str) -> str:
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        return ""
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            evidence = check.get("evidence")
            return evidence if isinstance(evidence, str) else ""
    return ""


def build_validation(
    root: Path,
    *,
    tarball: Path,
    work_dir: Path,
    arc_prefix: str,
) -> dict[str, Any]:
    root = root.resolve()
    tarball = (tarball if tarball.is_absolute() else root / tarball).resolve()
    work_dir = work_dir.resolve()
    output_validation = work_dir / "external_extracted_validation.json"
    extracted_root = work_dir / arc_prefix
    checks: list[Check] = []

    try:
        _safe_reset_work_dir(work_dir)
        reset_error = None
    except Exception as exc:
        reset_error = str(exc)

    _check(
        checks,
        "standalone work dir prepared",
        reset_error is None and work_dir.is_dir(),
        f"work_dir={work_dir}, reset_error={reset_error}",
    )
    _check(
        checks,
        "source tarball exists",
        tarball.is_file() and tarball.stat().st_size > 0,
        f"tarball={tarball}, size={tarball.stat().st_size if tarball.is_file() else 'missing'}",
    )

    unsafe_members: list[str] = []
    extract_error = "work dir preparation failed" if reset_error else None
    if reset_error is None and tarball.is_file():
        unsafe_members, extract_error = _safe_extract_tarball(tarball, work_dir)
    _check(
        checks,
        "tarball extracts safely outside repository",
        not unsafe_members and extract_error is None and extracted_root.is_dir(),
        (
            f"unsafe_members={unsafe_members[:10]}, extract_error={extract_error}, "
            f"extracted_root={extracted_root}"
        ),
    )

    validator = extracted_root / "benchmarks/eval/validate_qwen35_omni_share_package.py"
    readme = extracted_root / "PACKAGE_README.txt"
    sha_list = extracted_root / "PACKAGE_FILE_SHA256SUMS.txt"
    _check(
        checks,
        "extracted bundle has standalone entry files",
        validator.is_file() and readme.is_file() and sha_list.is_file(),
        (
            f"validator={validator.is_file()}, readme={readme.is_file()}, "
            f"sha_list={sha_list.is_file()}"
        ),
    )

    command = [
        sys.executable,
        str(validator),
        "--root",
        str(extracted_root),
        "--extracted-only",
        "--strict",
        "--json-output",
        str(output_validation),
    ]
    proc_returncode: int | None = None
    proc_output = ""
    if validator.is_file():
        proc = subprocess.run(
            command,
            cwd=extracted_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        proc_returncode = proc.returncode
        proc_output = proc.stdout or ""
    _check(
        checks,
        "package validator runs from extracted root",
        proc_returncode == 0 and output_validation.is_file(),
        (
            f"returncode={proc_returncode}, cwd={extracted_root}, "
            f"command={command}, output_tail={proc_output[-1000:]}"
        ),
    )

    extracted_payload = _load_json_optional(output_validation)
    extracted_summary = extracted_payload.get("summary", {})
    quality_evidence = _check_evidence(
        extracted_payload, "extracted bundle contains quick-read and stage-budget assets"
    )
    _check(
        checks,
        "extracted-only validation gate is green",
        bool(extracted_summary.get("ready"))
        and _int_value(extracted_summary.get("checks_passed"))
        == _int_value(extracted_summary.get("checks_total"), default=-1)
        and _int_value(extracted_summary.get("checks_total")) >= 13
        and _int_value(extracted_summary.get("required_failures"), default=1) == 0
        and bool(extracted_summary.get("extracted_only")),
        f"summary={extracted_summary}",
    )
    _check(
        checks,
        "packaged reports and charts are quality-clean",
        "report_quality_offenders=[]" in quality_evidence
        and "chart_quality_offenders=[]" in quality_evidence,
        quality_evidence or "missing quality evidence",
    )
    _check(
        checks,
        "invocation does not depend on repository module import",
        command[1] == str(validator)
        and str(validator).startswith(str(extracted_root))
        and command[3] == str(extracted_root),
        f"command={command}, cwd={extracted_root}",
    )

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    warnings = [
        check for check in checks if not check.required and check.status != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "tarball": str(tarball),
        "work_dir": str(work_dir),
        "extracted_root": str(extracted_root),
        "validator": str(validator),
        "validation_json": str(output_validation),
        "command": command,
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "warnings": len(warnings),
            "extracted_validation_ready": bool(extracted_summary.get("ready")),
            "extracted_validation_checks": extracted_summary.get("checks_total"),
            "extracted_validation_required_failures": extracted_summary.get(
                "required_failures"
            ),
            "repo_independent_invocation": True,
        },
        "extracted_validation_summary": extracted_summary,
        "extracted_quality_evidence": quality_evidence,
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "warnings": [check.to_dict() for check in warnings],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni External Standalone Bundle Validation\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Extracted validation ready | {summary['extracted_validation_ready']} |")
    print(f"| Extracted root | `{payload['extracted_root']}` |")
    print("\n| Status | Required | Check | Evidence |")
    print("| --- | --- | --- | --- |")
    for check in payload["checks"]:
        evidence = str(check["evidence"]).replace("|", "\\|")
        required = "yes" if check["required"] else "no"
        print(f"| {check['status']} | {required} | {check['name']} | {evidence} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the Qwen3.5-Omni share tarball into a clean /tmp root and "
            "run the bundled validator directly from the extracted package."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--arc-prefix", default=DEFAULT_ARC_PREFIX)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output if args.json_output.is_absolute() else root / args.json_output
    payload = build_validation(
        root,
        tarball=args.tarball,
        work_dir=args.work_dir,
        arc_prefix=args.arc_prefix,
    )
    _save_json(payload, output)
    print_markdown(payload)
    print(
        "External standalone bundle validation written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
