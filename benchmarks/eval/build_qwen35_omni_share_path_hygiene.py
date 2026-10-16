# SPDX-License-Identifier: Apache-2.0
"""Audit path references in Qwen3.5-Omni share-report Markdown files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval.build_qwen35_omni_share_bundle_manifest import build_bundle


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "share_path_hygiene.json"
PATH_PATTERN = re.compile(r"(?:benchmarks|results)/(?:[A-Za-z0-9_./*+\-]+)")
PACKAGE_PREFIXES = (
    "benchmarks/reports/",
    "benchmarks/eval/",
    "results/qwen35_report_audit_20260619/",
)
RAW_PREFIX = "results/qwen35_"
AUDIT_PREFIX = "results/qwen35_report_audit_20260619/"
ADJACENT_ARTIFACTS = {
    "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
    "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256",
    "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
    "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
    "results/qwen35_report_audit_20260619/share_package_validation.json",
    "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
    "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
    "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.summary.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.query.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.summary.out",
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.query.out",
    "results/qwen35_report_audit_20260619/share_release_seal.json",
    "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
    "results/qwen35_report_audit_20260619/preflight_repro.json",
}
GENERATED_OUTPUT_DIRS = {
    "results/qwen35_videoamme_seedtts_smoke_c8",
}
LEGACY_TOKENS = [
    "manifest `159` records",
    "manifest: `159` records",
    "manifest: 159 records",
    "manifest 159 records",
    "manifest: `180` records",
    "manifest: 180 records",
    "manifest: 177 records",
    "manifest 177 records",
    "177 records, 0 missing",
    "159 records, 157 files",
    "benchmark_audio_50_c2_profile_skipwer",
    "request_profile_c2_profile_skipwer",
]


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


def _normalize_ref(ref: str) -> str:
    ref = ref.rstrip(".,;)`|").rstrip("/")
    if ref.endswith("/."):
        ref = ref[:-2]
    return ref


def _bundle_paths(bundle: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for record in bundle.get("records", []):
        if not isinstance(record, dict):
            continue
        rel_path = str(record.get("relative_path") or "").rstrip("/")
        if rel_path:
            paths.add(rel_path)
        for item in record.get("files", []):
            if not isinstance(item, dict):
                continue
            item_rel_path = str(item.get("relative_path") or "").rstrip("/")
            if item_rel_path:
                paths.add(item_rel_path)
    return paths


def _share_report_paths(bundle: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for record in bundle.get("records", []):
        if (
            isinstance(record, dict)
            and record.get("category") == "share_report"
            and record.get("type") == "file"
        ):
            rel_path = str(record.get("relative_path") or "")
            if rel_path.endswith(".md"):
                result.append(rel_path)
    return sorted(set(result))


def _raw_manifest_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        rel_path = str(record.get("relative_path") or "").rstrip("/")
        if rel_path and record.get("exists") is not False:
            paths.add(rel_path)
        for item in record.get("files", []):
            if not isinstance(item, dict):
                continue
            item_rel_path = str(item.get("relative_path") or "").rstrip("/")
            if item_rel_path and item.get("exists") is not False:
                paths.add(item_rel_path)
    return paths


def _extract_refs(root: Path, report_paths: list[str]) -> tuple[dict[str, list[str]], set[str]]:
    refs_by_report: dict[str, list[str]] = {}
    unique_refs: set[str] = set()
    for rel_path in report_paths:
        text = _read_text_optional(root / rel_path)
        refs: list[str] = []
        for match in PATH_PATTERN.finditer(text):
            ref = _normalize_ref(match.group(0))
            if not ref:
                continue
            refs.append(ref)
            unique_refs.add(ref)
        refs_by_report[rel_path] = sorted(set(refs))
    return refs_by_report, unique_refs


def _is_template_fragment(ref: str) -> bool:
    return (
        ref.endswith("_")
        or ref.endswith("_20260620_")
        or ref.endswith("_c")
        or "$(date" in ref
    )


def _legacy_token_hits(root: Path, report_paths: list[str]) -> list[str]:
    hits: list[str] = []
    for rel_path in report_paths:
        text = _read_text_optional(root / rel_path)
        for token in LEGACY_TOKENS:
            if token in text:
                hits.append(f"{rel_path}: {token}")
    return hits


def build_hygiene(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    bundle = build_bundle(root)
    manifest = _load_json_optional(audit_dir / "manifest.json")
    report_paths = _share_report_paths(bundle)
    refs_by_report, unique_refs = _extract_refs(root, report_paths)
    package_paths = _bundle_paths(bundle)
    raw_paths = _raw_manifest_paths(manifest)

    package_refs: list[str] = []
    package_offenders: list[str] = []
    host_repo_only_refs: list[str] = []
    adjacent_refs: list[str] = []
    for ref in sorted(unique_refs):
        if "*" in ref:
            continue
        if not ref.startswith(PACKAGE_PREFIXES):
            continue
        package_refs.append(ref)
        if ref in package_paths:
            continue
        if ref in ADJACENT_ARTIFACTS:
            adjacent_refs.append(ref)
            continue
        if (root / ref).exists():
            host_repo_only_refs.append(ref)
            continue
        package_offenders.append(ref)

    raw_refs = [
        ref
        for ref in sorted(unique_refs)
        if ref.startswith(RAW_PREFIX) and not ref.startswith(AUDIT_PREFIX)
    ]
    raw_artifact_refs: list[str] = []
    raw_directory_refs: list[str] = []
    raw_wildcard_refs: list[str] = []
    raw_generated_outputs: list[str] = []
    raw_template_fragments: list[str] = []
    raw_offenders: list[str] = []
    for ref in raw_refs:
        if "*" in ref:
            raw_wildcard_refs.append(ref)
        elif ref in GENERATED_OUTPUT_DIRS:
            raw_generated_outputs.append(ref)
        elif ref in raw_paths:
            raw_artifact_refs.append(ref)
        elif (root / ref).is_dir():
            raw_directory_refs.append(ref)
        elif _is_template_fragment(ref):
            raw_template_fragments.append(ref)
        else:
            raw_offenders.append(ref)

    receiver_path_map = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md"
    )
    generated_outputs_undocumented = [
        ref for ref in raw_generated_outputs if ref not in receiver_path_map
    ]
    legacy_hits = _legacy_token_hits(root, report_paths)

    checks: list[Check] = []
    _check(
        checks,
        "share reports discovered",
        len(report_paths) >= 34,
        f"share_reports={len(report_paths)}",
    )
    _check(
        checks,
        "package path references classified",
        not package_offenders,
        (
            f"package_refs={len(package_refs)}, package_paths={len(package_paths)}, "
            f"host_repo_only={len(host_repo_only_refs)}, adjacent={len(adjacent_refs)}, "
            f"offenders={package_offenders[:10]}"
        ),
    )
    _check(
        checks,
        "raw artifact references classified",
        not raw_offenders,
        (
            f"raw_refs={len(raw_refs)}, artifacts={len(raw_artifact_refs)}, "
            f"directories={len(raw_directory_refs)}, wildcards={len(raw_wildcard_refs)}, "
            f"generated_outputs={len(raw_generated_outputs)}, "
            f"template_fragments={len(raw_template_fragments)}, "
            f"offenders={raw_offenders[:10]}"
        ),
    )
    _check(
        checks,
        "generated output dirs documented",
        not generated_outputs_undocumented,
        f"generated_outputs={raw_generated_outputs}, undocumented={generated_outputs_undocumented}",
    )
    _check(
        checks,
        "legacy path and gate tokens absent",
        not legacy_hits,
        f"legacy_hits={legacy_hits[:20]}",
    )

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "share_reports_total": len(report_paths),
            "unique_refs_total": len(unique_refs),
            "package_refs_total": len(package_refs),
            "package_offenders_total": len(package_offenders),
            "host_repo_only_refs_total": len(host_repo_only_refs),
            "adjacent_refs_total": len(adjacent_refs),
            "raw_refs_total": len(raw_refs),
            "raw_artifact_refs_total": len(raw_artifact_refs),
            "raw_directory_refs_total": len(raw_directory_refs),
            "raw_wildcard_refs_total": len(raw_wildcard_refs),
            "raw_generated_outputs_total": len(raw_generated_outputs),
            "raw_template_fragments_total": len(raw_template_fragments),
            "raw_offenders_total": len(raw_offenders),
            "legacy_hits_total": len(legacy_hits),
        },
        "checks": [check.to_dict() for check in checks],
        "share_reports": report_paths,
        "refs_by_report": refs_by_report,
        "classifications": {
            "package_offenders": package_offenders,
            "host_repo_only_refs": host_repo_only_refs,
            "adjacent_refs": adjacent_refs,
            "raw_artifact_refs": raw_artifact_refs,
            "raw_directory_refs": raw_directory_refs,
            "raw_wildcard_refs": raw_wildcard_refs,
            "raw_generated_outputs": raw_generated_outputs,
            "raw_template_fragments": raw_template_fragments,
            "raw_offenders": raw_offenders,
            "legacy_hits": legacy_hits,
        },
        "policy": {
            "package_paths": (
                "benchmarks/reports, packaged benchmarks/eval tools, and "
                "results/qwen35_report_audit_20260619 references must be packaged, "
                "documented as adjacent artifacts, or explicitly host-repo-only."
            ),
            "raw_paths": (
                "non-audit results/qwen35_* references must be raw manifest artifacts, "
                "existing directory-level roots, documented wildcards, generated output "
                "directories, or command-template fragments."
            ),
        },
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Share Path Hygiene\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Share reports | {summary['share_reports_total']} |")
    print(f"| Unique refs | {summary['unique_refs_total']} |")
    print(f"| Package offenders | {summary['package_offenders_total']} |")
    print(f"| Raw offenders | {summary['raw_offenders_total']} |")
    print(f"| Generated output refs | {summary['raw_generated_outputs_total']} |")
    print(f"| Legacy token hits | {summary['legacy_hits_total']} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit share-report path references for package/raw-evidence hygiene."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output = args.json_output if args.json_output.is_absolute() else root / args.json_output
    payload = build_hygiene(root)
    _save_json(payload, output)
    print_markdown(payload)
    print(f"Share path hygiene written: {output} ready={payload['summary']['ready']}")
    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
