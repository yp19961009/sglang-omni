# SPDX-License-Identifier: Apache-2.0
"""Validate the Qwen3.5-Omni share tarball or an extracted share bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_TARBALL = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "share_package_validation.json"
PACKAGE_SHA_LIST = Path("PACKAGE_FILE_SHA256SUMS.txt")
PACKAGE_README = Path("PACKAGE_README.txt")
SELF_REFERENTIAL_BUNDLE_HASH_PATHS = {
    "results/qwen35_report_audit_20260619/audit_run_summary.json",
    "results/qwen35_report_audit_20260619/manifest.json",
}
TEXT_PACKAGE_SUFFIXES = (".md", ".json", ".txt")
HEX64_PATTERN = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)
TARBALL_IDENTITY_CONTEXT = re.compile(
    r"tarball|checksum|qwen35_omni_share_bundle_20260621\.tar\.gz|"
    r"share_bundle_package_manifest|share_package_validation|share_release_seal",
    re.IGNORECASE,
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(fp: Any) -> str:
    digest = hashlib.sha256()
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


def _float_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _checksum_path(tarball: Path) -> Path:
    return tarball.with_suffix(tarball.suffix + ".sha256")


def _checksum_value(path: Path) -> tuple[str | None, str]:
    if not path.is_file():
        return None, "missing checksum file"
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if not parts:
        return None, "empty checksum file"
    return parts[0], text


def _tar_members(path: Path) -> tuple[set[str], str | None]:
    try:
        with tarfile.open(path, "r:gz") as tf:
            return set(tf.getnames()), None
    except Exception as exc:
        return set(), str(exc)


def _tar_member_text(path: Path, member_name: str) -> tuple[str, str | None]:
    try:
        with tarfile.open(path, "r:gz") as tf:
            member = tf.extractfile(member_name)
            if member is None:
                return "", f"missing member: {member_name}"
            return member.read().decode("utf-8"), None
    except Exception as exc:
        return "", str(exc)


def _package_readme_text(
    root: Path,
    tarball: Path,
    *,
    arc_prefix: str,
    extracted_only: bool,
) -> tuple[str, str | None]:
    if extracted_only:
        path = root / PACKAGE_README
        if not path.is_file():
            return "", f"missing {PACKAGE_README}"
        try:
            return path.read_text(encoding="utf-8"), None
        except Exception as exc:
            return "", str(exc)
    if not tarball.is_file():
        return "", "missing tarball"
    return _tar_member_text(tarball, f"{arc_prefix}/{PACKAGE_README.as_posix()}")


def _package_readme_boundary_ready(text: str) -> tuple[bool, list[str]]:
    needles = [
        "share_bundle_package_manifest.json",
        "share_package_validation.json",
        "share_package_validation_extracted.json",
        "share_package_receiver_smoke_validation.json",
        "share_package_external_standalone_validation.json",
        "share_release_seal.json",
        "qwen35_omni_share_release_seal_zh_20260621.md",
        "adjacent to the tarball",
        "intentionally not tarball members",
        "self-referential hashes",
        "Start with benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
        "30-second gate summary",
        "three receiver verification commands",
        "first-read files",
        "caveat red lines",
        "For a copy/paste university email or chat message",
        "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md",
        "Then open benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
        "full reading order, evidence map, and defense-question routes",
        "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
        "HOST_REPO, BUNDLE_ROOT, and container paths",
        "For a one-page copy/paste receiver command card",
        "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
        "it chains checksum, receiver quickcheck, standalone validation, full audit",
        "performance rerun entry points, and headline replacement boundaries",
        "For the machine-checked receiver quickcheck contract",
        "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
        "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
        "For the single Chinese technical-report entry point",
        "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
        "results/qwen35_report_audit_20260619/university_technical_report.json",
        "For pressure-by-pressure rerun command IDs",
        "headline replacement boundaries",
        "benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md",
        "For copy/paste jq proof cards",
        "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
        "qwen35_omni_evidence_query_cards_smoke.sh",
        "one-command evidence-card smoke",
        "section 2 first",
        "quick_reproduction_map",
        "5 defense quick routes",
        "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
        "results/qwen35_report_audit_20260619/optimization_candidate_ledger.json",
        "The vLLM baseline uses a Qwen3.5-capable image",
        "compile/CUDA graph",
        "prefix/chunked prefill",
        "shared-memory transfer",
        "encoder compile/batch",
        "and prebuild w4 evidence locked",
        "it is not a weak baseline",
        "vLLM c=8 prebuild w4 remains an optimized offline diagnostic",
        "not online serving parity",
        "For stage breakdown and stage-to-stage bottlenecks",
        "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
        "results/qwen35_report_audit_20260619/stage_causal_graph.json",
        "results/qwen35_report_audit_20260619/stage_drilldown_index.json",
        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
        "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
        "For single/high concurrency and short/long text regimes",
        "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
        "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
        "For one-page headline numbers, presentation flow, and PPT-ready figures",
        "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md",
        "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json",
        "results/qwen35_report_audit_20260619/chart_source_consistency.json",
        "results/qwen35_report_audit_20260619/metric_provenance_index.json",
        "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
        "For safe external wording, caveats that must travel, and defense questions",
        "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
        "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json",
        "results/qwen35_report_audit_20260619/defense_claim_matrix.json",
        "results/qwen35_report_audit_20260619/claim_metric_crosswalk.json",
        "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
        "For external reproduction commands and handoff steps",
        "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
        "results/qwen35_report_audit_20260619/repro_command_manifest.json",
        "For package/raw path reference hygiene",
        "results/qwen35_report_audit_20260619/share_path_hygiene.json",
        "package_offenders_total=0",
        "raw_offenders_total=0",
        "legacy_hits_total=0",
        "For command-reference hygiene",
        "results/qwen35_report_audit_20260619/command_reference_hygiene.json",
        "structured rerun command IDs",
        "critical SGLang/vLLM commands",
        "For reviewer reruns and delta triage",
        "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
        "qwen35_omni_collaborator_return_check.py",
        "34 required return-evidence files",
        "27 command-matrix rows",
        "headline replacement review",
        "Headline replacement boundary",
        "Green / replacement review",
        "same 8x H20 hardware",
        "same SGLang and vLLM image digests",
        "same model",
        "same Video-AMME cache",
        "same ASR/WER path",
        "full audit plus rerun acceptance contract are green",
        "Yellow / appendix only",
        "hardware, image digest, model path, data cache, or ASR/WER path differs",
        "Red / do not replace",
        "any required gate fails",
        "share package validation fails",
        "vLLM c=8 evidence is still offline diagnostic only",
        "Before replacing any report number",
        "update the machine evidence JSON, charts, main report",
        "stage causal graph, Q&A, defense claim matrix, claim metric crosswalk, objective requirement crosswalk",
        "stage reproduction drilldown, stage route decision matrix",
        "share package index, final delivery note",
        "share bundle, checksum, receiver smoke validation, extracted-only validation, external standalone validation, and release seal",
        "Do not claim SGLang/vLLM headline replacement or vLLM c=8 online parity",
        "yellow/red rerun",
        "For a one-command receiver quickcheck, run",
        "benchmarks/eval/qwen35_omni_receiver_quickcheck.sh from the repository root",
        "it chains checksum, tarball-mode validation, receiver-smoke validation",
        "extracted-only validation, and external standalone validation",
        "For a manual receiver-side sanity check, run",
        "benchmarks/eval/validate_qwen35_omni_share_package.py from the repository root",
        "in_progress=true audit_run_summary.json",
        "expected only for extracted-only validation",
        "direct machine gates in the bundle are green",
        "rerun_delta_triage.json",
        "rows_total>=19",
        "checks_passed>=8",
        "completion-state audit_run_summary.json with top-level rerun_delta_triage",
        "If the repository is mounted elsewhere, set HOST_REPO before copying commands.",
        "Quick receiver checks from the repository root when this package is current",
        'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
        'SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_readme}"',
        'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_readme}"',
        'STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_readme}"',
        'cd "$HOST_REPO"',
        "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
        "bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
        "python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check",
        "collaborator_return_check.json",
        '--root "$HOST_REPO" --mode host',
        '--root "$BUNDLE_ROOT" --mode portable',
        "# Manual split commands:",
        "sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256",
        "python3 -m benchmarks.eval.validate_qwen35_omni_share_package",
        '--root "$HOST_REPO"',
        '--receiver-smoke-dir "$SMOKE_DIR"',
        "Manual extracted-root check after unpacking",
        'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_readme}"',
        'rm -rf "$EXTRACT_DIR"',
        "tar -xzf results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
        'cd "$EXTRACT_DIR/qwen35_omni_share_bundle_20260621"',
        'BUNDLE_ROOT="$PWD"',
        "python3 benchmarks/eval/validate_qwen35_omni_share_package.py",
        "--extracted-only",
        '$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json',
        "build_qwen35_omni_external_standalone_bundle_validation",
        "qwen35_omni_external_standalone_bundle_validation_readme",
        "share_package_external_standalone_validation.json",
        "share_release_seal.json",
        "--receiver-smoke-dir",
        "share_package_receiver_smoke_validation.json",
        "Expected gates when this package is current",
        "final readiness 49/49",
        "tarball validation 17/17",
        "extracted-only validation 13/13",
        "external standalone validation 8/8",
        "release seal ready=true",
        "public_doc_quality_guard",
        "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift",
        "hash/table/token/duplicate-heading/semantic-count offenders all empty",
        "tarball validation, safe extraction",
        "external standalone validation in one host-side step",
        "directly scans packaged share_report Markdown",
        "bare hashes, malformed tables, duplicate headings, semantic count drift, and malformed display tokens",
        "validates packaged share_charts CSV/SVG assets are parseable",
        "and render-structured",
        "report_quality_offenders=[]",
        "chart_quality_offenders=[]",
        "Wrong-root hints",
        "Tarball mode expects the repository root",
        "results/qwen35_report_audit_20260619/",
        "Extracted-only mode expects the extracted bundle root",
        "PACKAGE_FILE_SHA256SUMS.txt",
        "Internal file hash list",
        "repository-root-relative paths",
        "used by tarball validation and --extracted-only validation",
        "before or after unpacking",
    ]
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _expected_bundle_members(
    bundle: dict[str, Any],
    *,
    arc_prefix: str,
) -> list[str]:
    rel_paths: list[str] = []
    for record in bundle.get("records", []):
        if record.get("type") == "file":
            rel_paths.append(str(record.get("relative_path") or ""))
        elif record.get("type") == "directory":
            for item in record.get("files", []):
                rel_paths.append(str(item.get("relative_path") or ""))
    rel_paths.append("results/qwen35_report_audit_20260619/share_bundle_manifest.json")
    result: list[str] = []
    seen: set[str] = set()
    for rel_path in rel_paths:
        if not rel_path or rel_path in seen:
            continue
        seen.add(rel_path)
        result.append(f"{arc_prefix}/{rel_path}")
    return result


def _bundle_file_records(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in bundle.get("records", []):
        if record.get("type") == "file":
            rel_path = str(record.get("relative_path") or "")
            if rel_path:
                records.append(
                    {
                        "relative_path": rel_path,
                        "sha256": str(record.get("sha256") or ""),
                    }
                )
        elif record.get("type") == "directory":
            for item in record.get("files", []):
                rel_path = str(item.get("relative_path") or "")
                if rel_path:
                    records.append(
                        {
                            "relative_path": rel_path,
                            "sha256": str(item.get("sha256") or ""),
                        }
                    )
    return records


def _dedupe_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    conflicts: list[str] = []
    for record in records:
        rel_path = str(record.get("relative_path") or "")
        expected = str(record.get("sha256") or "")
        if not rel_path:
            continue
        if rel_path in seen:
            duplicates.append(rel_path)
            if seen[rel_path] and expected and seen[rel_path] != expected:
                conflicts.append(rel_path)
            continue
        seen[rel_path] = expected
        result.append(record)
    return result, duplicates, conflicts


def _missing_or_mismatched_records(
    root: Path, records: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    for record in records:
        rel_path = str(record.get("relative_path") or "")
        expected = str(record.get("sha256") or "")
        path = root / rel_path
        if not path.is_file():
            missing.append(rel_path)
            continue
        if expected and _sha256(path) != expected:
            mismatched.append(rel_path)
    return missing, mismatched


def _parse_sha_list(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.is_file():
        return [], "missing checksum list"
    try:
        return _parse_sha_list_text(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return [], str(exc)


def _parse_sha_list_text(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        records.append({"sha256": parts[0], "relative_path": parts[1].strip()})
    return records


def _invalid_relative_paths(records: list[dict[str, Any]]) -> list[str]:
    return sorted(
        str(record.get("relative_path") or "")
        for record in records
        if Path(str(record.get("relative_path") or "")).is_absolute()
        or str(record.get("relative_path") or "").startswith("./")
        or str(record.get("relative_path") or "") == "."
        or ".." in Path(str(record.get("relative_path") or "")).parts
    )


def _tar_records_hash_status(
    path: Path,
    records: list[dict[str, Any]],
    *,
    arc_prefix: str,
) -> tuple[list[str], list[str], str | None]:
    missing: list[str] = []
    mismatched: list[str] = []
    try:
        with tarfile.open(path, "r:gz") as tf:
            names = set(tf.getnames())
            for record in records:
                rel_path = str(record.get("relative_path") or "")
                expected = str(record.get("sha256") or "")
                member_name = f"{arc_prefix}/{rel_path}"
                if member_name not in names:
                    missing.append(rel_path)
                    continue
                member = tf.extractfile(member_name)
                if member is None:
                    missing.append(rel_path)
                    continue
                if expected and _sha256_stream(member) != expected:
                    mismatched.append(rel_path)
    except Exception as exc:
        return missing, mismatched, str(exc)
    return missing, mismatched, None


def _safe_extract_tarball(path: Path, output_dir: Path) -> tuple[list[str], str | None]:
    unsafe: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_root = output_dir.resolve()
        with tarfile.open(path, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                member_name = member.name
                if member_name.startswith("/") or ".." in Path(member_name).parts:
                    unsafe.append(member_name)
                    continue
                try:
                    (output_root / member_name).resolve().relative_to(output_root)
                except ValueError:
                    unsafe.append(member_name)
                    continue
                if member.islnk() or member.issym() or not (
                    member.isfile() or member.isdir()
                ):
                    unsafe.append(member_name)
            if unsafe:
                return sorted(set(unsafe)), None
            tf.extractall(output_root, members=members)
    except Exception as exc:
        return unsafe, str(exc)
    return [], None


def _sha_record_map(records: list[dict[str, str]]) -> dict[str, str]:
    return {
        str(record.get("relative_path") or ""): str(record.get("sha256") or "")
        for record in records
        if record.get("relative_path")
    }


def _required_suffixes() -> list[str]:
    return [
        "PACKAGE_README.txt",
        "PACKAGE_FILE_SHA256SUMS.txt",
        "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
        "benchmarks/eval/validate_qwen35_omni_share_package.py",
        "benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
        "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
        "benchmarks/eval/qwen35_omni_collaborator_return_check.py",
        "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
        "results/qwen35_report_audit_20260619/runtime_image_contract.json",
        "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
        "results/qwen35_report_audit_20260619/slide_asset_map.json",
        "results/qwen35_report_audit_20260619/rerun_delta_triage.json",
        "results/qwen35_report_audit_20260619/university_review_packet.json",
        "results/qwen35_report_audit_20260619/university_technical_report.json",
        "results/qwen35_report_audit_20260619/metric_provenance_index.json",
        "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
        "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
        "results/qwen35_report_audit_20260619/claim_metric_crosswalk.json",
        "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
        "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
        "results/qwen35_report_audit_20260619/share_path_hygiene.json",
        "results/qwen35_report_audit_20260619/command_reference_hygiene.json",
        "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
        "results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget.csv",
        "results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget_pct.svg",
    ]


def _share_report_paths(bundle: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for record in bundle.get("records", []):
        rel_path = str(record.get("relative_path") or "")
        if (
            record.get("category") == "share_report"
            and record.get("type") == "file"
            and rel_path.endswith(".md")
        ):
            paths.append(rel_path)
    return paths


def _share_chart_paths(bundle: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for record in bundle.get("records", []):
        rel_path = str(record.get("relative_path") or "")
        if (
            record.get("category") == "share_chart"
            and record.get("type") == "file"
            and (rel_path.endswith(".csv") or rel_path.endswith(".svg"))
        ):
            paths.append(rel_path)
    return paths


def _unescaped_pipe_count(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def _chinese_count(value: str) -> int | None:
    mapping = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return mapping.get(value)


def _semantic_consistency_offenders(rel_path: str, text: str) -> list[str]:
    offenders: list[str] = []
    redline_heading = re.compile(
        r"^##\s+\d+\.\s+([一二三四五六七八九十])条红线\s*$",
        flags=re.MULTILINE,
    )
    next_section = re.compile(r"^##\s+\d+\.\s+", flags=re.MULTILINE)
    for match in redline_heading.finditer(text):
        expected = _chinese_count(match.group(1))
        following = text[match.end() :]
        next_match = next_section.search(following)
        section = following[: next_match.start()] if next_match else following
        bullet_count = sum(1 for line in section.splitlines() if line.startswith("- "))
        if expected is not None and bullet_count != expected:
            line_number = text.count("\n", 0, match.start()) + 1
            offenders.append(
                f"{rel_path}:{line_number}:redline_count:"
                f"heading={expected}, bullets={bullet_count}"
            )
    return offenders


def _report_quality_offenders(rel_path: str, text: str) -> list[str]:
    offenders: list[str] = []
    hash_pattern = re.compile(r"(?<!sha256:)\b[0-9a-f]{64}\b")
    heading_pattern = re.compile(r"^#{1,6} .+")
    malformed_tokens = (
        "n/ams",
        "n/a%",
        "nanms",
        "diagnosis=None",
        "主报告 section",
        "对外解释时保持三条边界",
    )
    expected_pipes = 0
    table_start_line = 0
    seen_headings: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in hash_pattern.finditer(line):
            offenders.append(f"{rel_path}:{line_number}:bare_hash:{match.group(0)[:12]}...")
        if heading_pattern.match(line):
            first_seen = seen_headings.get(line)
            if first_seen is not None:
                offenders.append(
                    f"{rel_path}:{line_number}:duplicate_heading:"
                    f"{line} first_seen_line={first_seen}"
                )
            else:
                seen_headings[line] = line_number
        for token in malformed_tokens:
            if token in line:
                offenders.append(f"{rel_path}:{line_number}:malformed_token:{token}")
        if line.startswith("|"):
            pipe_count = _unescaped_pipe_count(line)
            if not table_start_line:
                table_start_line = line_number
                expected_pipes = pipe_count
            elif pipe_count != expected_pipes:
                offenders.append(
                    f"{rel_path}:{line_number}:table_pipes:{pipe_count} "
                    f"expected {expected_pipes} from line {table_start_line}"
                )
        else:
            expected_pipes = 0
            table_start_line = 0
    offenders.extend(_semantic_consistency_offenders(rel_path, text))
    return offenders


def _tarball_identity_hash_offenders(rel_path: str, text: str) -> list[str]:
    offenders: list[str] = []
    if not rel_path.endswith(TEXT_PACKAGE_SUFFIXES):
        return offenders
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not TARBALL_IDENTITY_CONTEXT.search(line):
            continue
        for match in HEX64_PATTERN.finditer(line):
            offenders.append(
                f"{rel_path}:{line_number}:tarball_identity_hash:{match.group(0)[:12]}..."
            )
    return offenders


def _chart_quality_offenders(rel_path: str, text: str) -> list[str]:
    offenders: list[str] = []
    if rel_path.endswith(".csv"):
        try:
            rows = [
                row
                for row in csv.reader(io.StringIO(text))
                if any(cell.strip() for cell in row)
            ]
        except csv.Error as exc:
            return [f"{rel_path}:csv_parse_error:{exc}"]
        if len(rows) < 2:
            offenders.append(f"{rel_path}:csv_too_few_rows:{len(rows)}")
            return offenders
        header_len = len(rows[0])
        if header_len < 2:
            offenders.append(f"{rel_path}:csv_header_too_narrow:{header_len}")
        mismatched_rows = [
            index
            for index, row in enumerate(rows[1:], start=2)
            if len(row) != header_len
        ]
        if mismatched_rows:
            offenders.append(
                f"{rel_path}:csv_column_mismatch_rows:{mismatched_rows[:5]}"
            )
        return offenders
    if rel_path.endswith(".svg"):
        stripped = text.lstrip()
        if len(text) < 200:
            offenders.append(f"{rel_path}:svg_too_small:{len(text)}")
        if not stripped.startswith("<svg"):
            offenders.append(f"{rel_path}:svg_missing_open_tag")
        if "</svg>" not in text:
            offenders.append(f"{rel_path}:svg_missing_close_tag")
        if "viewBox=" not in text:
            offenders.append(f"{rel_path}:svg_missing_viewbox")
        if "width=" not in text or "height=" not in text:
            offenders.append(f"{rel_path}:svg_missing_dimensions")
    return offenders


def _extracted_report_quality_offenders(
    root: Path, bundle: dict[str, Any]
) -> list[str]:
    offenders: list[str] = []
    for rel_path in _share_report_paths(bundle):
        path = root / rel_path
        if not path.is_file():
            offenders.append(f"{rel_path}:missing")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            offenders.append(f"{rel_path}:read_error:{exc}")
            continue
        offenders.extend(_report_quality_offenders(rel_path, text))
    return offenders


def _extracted_chart_quality_offenders(root: Path, bundle: dict[str, Any]) -> list[str]:
    offenders: list[str] = []
    for rel_path in _share_chart_paths(bundle):
        path = root / rel_path
        if not path.is_file():
            offenders.append(f"{rel_path}:missing")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            offenders.append(f"{rel_path}:read_error:{exc}")
            continue
        offenders.extend(_chart_quality_offenders(rel_path, text))
    return offenders


def _extracted_identity_hash_offenders(
    root: Path, records: list[dict[str, str]]
) -> list[str]:
    offenders: list[str] = []
    seen: set[str] = set()
    for record in records:
        rel_path = str(record.get("relative_path") or "")
        if not rel_path or rel_path in seen or not rel_path.endswith(TEXT_PACKAGE_SUFFIXES):
            continue
        seen.add(rel_path)
        path = root / rel_path
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            offenders.append(f"{rel_path}:identity_read_error:{exc}")
            continue
        offenders.extend(_tarball_identity_hash_offenders(rel_path, text))
    return offenders


def _tar_report_quality_offenders(
    tarball: Path,
    bundle: dict[str, Any],
    *,
    arc_prefix: str,
) -> list[str]:
    offenders: list[str] = []
    if not tarball.is_file():
        return ["tarball:missing"]
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            names = set(tf.getnames())
            for rel_path in _share_report_paths(bundle):
                member_name = f"{arc_prefix}/{rel_path}"
                if member_name not in names:
                    offenders.append(f"{rel_path}:missing")
                    continue
                member = tf.extractfile(member_name)
                if member is None:
                    offenders.append(f"{rel_path}:missing_member")
                    continue
                try:
                    text = member.read().decode("utf-8")
                except Exception as exc:
                    offenders.append(f"{rel_path}:read_error:{exc}")
                    continue
                offenders.extend(_report_quality_offenders(rel_path, text))
    except Exception as exc:
        offenders.append(f"tarball:read_error:{exc}")
    return offenders


def _tar_identity_hash_offenders(
    tarball: Path,
    *,
    arc_prefix: str,
) -> list[str]:
    offenders: list[str] = []
    if not tarball.is_file():
        return ["tarball:missing"]
    arc_member_prefix = f"{arc_prefix}/"
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.startswith(arc_member_prefix):
                    continue
                rel_path = member.name[len(arc_member_prefix) :]
                if not rel_path.endswith(TEXT_PACKAGE_SUFFIXES):
                    continue
                member_fp = tf.extractfile(member)
                if member_fp is None:
                    continue
                try:
                    text = member_fp.read().decode("utf-8")
                except Exception as exc:
                    offenders.append(f"{rel_path}:identity_read_error:{exc}")
                    continue
                offenders.extend(_tarball_identity_hash_offenders(rel_path, text))
    except Exception as exc:
        offenders.append(f"tarball:identity_scan_error:{exc}")
    return offenders


def _tar_chart_quality_offenders(
    tarball: Path,
    bundle: dict[str, Any],
    *,
    arc_prefix: str,
) -> list[str]:
    offenders: list[str] = []
    if not tarball.is_file():
        return ["tarball:missing"]
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            names = set(tf.getnames())
            for rel_path in _share_chart_paths(bundle):
                member_name = f"{arc_prefix}/{rel_path}"
                if member_name not in names:
                    offenders.append(f"{rel_path}:missing")
                    continue
                member = tf.extractfile(member_name)
                if member is None:
                    offenders.append(f"{rel_path}:missing_member")
                    continue
                try:
                    text = member.read().decode("utf-8")
                except Exception as exc:
                    offenders.append(f"{rel_path}:read_error:{exc}")
                    continue
                offenders.extend(_chart_quality_offenders(rel_path, text))
    except Exception as exc:
        offenders.append(f"tarball:read_error:{exc}")
    return offenders


def _rerun_delta_summary_ready(summary: dict[str, Any]) -> bool:
    return (
        bool(summary.get("ready"))
        and _int_value(summary.get("rows_total")) >= 19
        and _int_value(summary.get("checks_total")) >= 8
        and _int_value(summary.get("checks_passed")) >= 8
        and _int_value(summary.get("required_failures"), default=1) == 0
    )


def _required_failure_names(payload: dict[str, Any]) -> set[str]:
    failures = payload.get("required_failures", [])
    if not isinstance(failures, list):
        return set()
    names: set[str] = set()
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        name = failure.get("name")
        if name:
            names.add(str(name))
    return names


def _final_readiness_ready_or_in_progress(
    final: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[bool, bool]:
    summary = final.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    strict_ready = (
        bool(summary.get("ready"))
        and _int_value(summary.get("required_failures"), default=1) == 0
        and _int_value(summary.get("checks_total")) >= 37
    )
    recovery_ready = (
        bool(audit.get("in_progress"))
        and bool(audit.get("ok"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 49
        and _int_value(summary.get("required_failures"), default=99) <= 1
        and _required_failure_names(final) == {"final checkpoint watchlist gate"}
        and summary.get("hard_gates", {}).get("final_checkpoint_watchlist")
        == "24/24"
    )
    return strict_ready or recovery_ready, recovery_ready


def _build_common_checks(
    checks: list[Check],
    *,
    audit: dict[str, Any],
    final: dict[str, Any],
    bundle_summary: dict[str, Any],
    final_summary: dict[str, Any],
    chart_summary: dict[str, Any],
    stage_summary: dict[str, Any],
    stage_ledger_summary: dict[str, Any],
    runtime_summary: dict[str, Any],
    rerun_summary: dict[str, Any],
    rerun_delta_summary: dict[str, Any],
    protocol_summary: dict[str, Any],
) -> None:
    audit_delta_summary = audit.get("rerun_delta_triage", {})
    audit_delta_ready = _rerun_delta_summary_ready(audit_delta_summary)
    final_readiness_ok, final_readiness_recovered = (
        _final_readiness_ready_or_in_progress(final, audit)
    )
    in_progress_recovery_ready = (
        bool(audit.get("in_progress"))
        and bool(audit.get("ok"))
        and _rerun_delta_summary_ready(rerun_delta_summary)
        and final_readiness_ok
        and bool(bundle_summary.get("ready"))
        and bool(chart_summary.get("ready"))
    )
    _check(
        checks,
        "share bundle manifest completeness",
        bool(bundle_summary.get("ready"))
        and _int_value(bundle_summary.get("records_total")) >= 70
        and _int_value(bundle_summary.get("missing_required"), default=1) == 0
        and _int_value(bundle_summary.get("category_counts", {}).get("share_chart")) >= 14
        and _int_value(bundle_summary.get("category_counts", {}).get("share_tool")) >= 1,
        f"share_bundle={bundle_summary}",
    )
    _check(
        checks,
        "final readiness gate",
        final_readiness_ok,
        (
            f"final_readiness={final_summary}, "
            f"recovered_from_in_progress_final_checkpoint={final_readiness_recovered}"
        ),
    )
    _check(
        checks,
        "full audit summary gate",
        (bool(audit.get("ok")) and audit_delta_ready)
        or in_progress_recovery_ready,
        (
            f"ok={audit.get('ok')}, "
            f"in_progress={audit.get('in_progress')}, "
            f"rerun_delta_triage={audit_delta_summary}, "
            f"direct_rerun_delta_triage={rerun_delta_summary}, "
            f"recovered_from_in_progress_gates={in_progress_recovery_ready}"
        ),
    )
    _check(
        checks,
        "chart pack includes stage budget assets",
        bool(chart_summary.get("ready"))
        and _int_value(chart_summary.get("csv_files")) >= 7
        and _int_value(chart_summary.get("svg_files")) >= 7
        and bool(chart_summary.get("checks", {}).get("stage_budget_csv"))
        and bool(chart_summary.get("checks", {}).get("stage_budget_svg")),
        f"chart_pack={chart_summary}",
    )
    _check(
        checks,
        "stage latency budget conclusion is preserved",
        bool(stage_summary.get("ready"))
        and _int_value(stage_summary.get("checks_total")) >= 12
        and _float_value(stage_summary.get("c8_queue_pct_of_latency")) >= 25.0
        and _float_value(stage_summary.get("c16_queue_pct_of_latency")) >= 60.0,
        f"stage_latency_budget={stage_summary}",
    )
    _check(
        checks,
        "stage boundary bottleneck ledger is preserved",
        bool(stage_ledger_summary.get("ready"))
        and _int_value(stage_ledger_summary.get("checks_total")) >= 10
        and _int_value(stage_ledger_summary.get("ledger_rows")) >= 37
        and _int_value(stage_ledger_summary.get("required_failures"), default=1) == 0,
        f"stage_boundary_bottleneck_ledger={stage_ledger_summary}",
    )
    _check(
        checks,
        "runtime image contract is preserved",
        bool(runtime_summary.get("ready"))
        and _int_value(runtime_summary.get("checks_total")) >= 12
        and _int_value(runtime_summary.get("required_failures"), default=1) == 0
        and "c4-c8" in str(runtime_summary.get("sglang_scope") or "")
        and "c4" in str(runtime_summary.get("vllm_strict_scope") or "")
        and "offline diagnostic" in str(runtime_summary.get("vllm_c8_scope") or ""),
        f"runtime_image_contract={runtime_summary}",
    )
    _check(
        checks,
        "rerun acceptance contract is preserved",
        bool(rerun_summary.get("ready"))
        and _int_value(rerun_summary.get("checks_total")) >= 15
        and _int_value(rerun_summary.get("required_failures"), default=1) == 0
        and _int_value(rerun_summary.get("rules_total")) >= 18,
        f"rerun_acceptance_contract={rerun_summary}",
    )
    _check(
        checks,
        "vLLM c8 caveat is preserved",
        bool(protocol_summary.get("ready"))
        and bool(protocol_summary.get("current_package_safe"))
        and not bool(protocol_summary.get("online_parity_proven")),
        f"vllm_online_parity_protocol={protocol_summary}",
    )


def build_validation(
    root: Path,
    tarball: Path,
    checksum: Path,
    *,
    extracted_only: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    tarball = tarball if tarball.is_absolute() else root / tarball
    checksum = checksum if checksum.is_absolute() else root / checksum
    audit_dir = root / AUDIT_DIR

    package = _load_json_optional(audit_dir / "share_bundle_package_manifest.json")
    bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    audit = _load_json_optional(audit_dir / "audit_run_summary.json")
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    chart_pack = _load_json_optional(audit_dir / "share_charts/chart_pack_manifest.json")
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_boundary_ledger = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    runtime_image_contract = _load_json_optional(
        audit_dir / "runtime_image_contract.json"
    )
    rerun_acceptance_contract = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    rerun_delta_triage = _load_json_optional(audit_dir / "rerun_delta_triage.json")
    vllm_protocol = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")

    package_summary = package.get("summary", {})
    bundle_summary = bundle.get("summary", {})
    final_summary = final.get("summary", {})
    chart_summary = chart_pack.get("summary", {})
    stage_summary = stage_budget.get("summary", {})
    stage_ledger_summary = stage_boundary_ledger.get("summary", {})
    runtime_summary = runtime_image_contract.get("summary", {})
    rerun_summary = rerun_acceptance_contract.get("summary", {})
    rerun_delta_summary = rerun_delta_triage.get("summary", {})
    protocol_summary = vllm_protocol.get("summary", {})

    arc_prefix = str(package.get("arc_prefix") or "qwen35_omni_share_bundle_20260621")
    checks: list[Check] = []
    package_readme_text, package_readme_error = _package_readme_text(
        root, tarball, arc_prefix=arc_prefix, extracted_only=extracted_only
    )
    package_readme_ready, package_readme_missing = _package_readme_boundary_ready(
        package_readme_text
    )

    actual_sha = None
    tar_names: set[str] = set()
    expected_members: list[str] = []
    missing_members: list[str] = []
    identity_hash_offenders: list[str] = []
    if extracted_only:
        sha_records, sha_list_error = _parse_sha_list(root / PACKAGE_SHA_LIST)
        sha_record_map = _sha_record_map(sha_records)
        invalid_sha_paths = _invalid_relative_paths(sha_records)
        sha_missing, sha_mismatched = _missing_or_mismatched_records(root, sha_records)
        raw_bundle_records = _bundle_file_records(bundle)
        bundle_records, bundle_duplicates, bundle_hash_conflicts = _dedupe_records(
            raw_bundle_records
        )
        bundle_missing, bundle_mismatched = _missing_or_mismatched_records(
            root, bundle_records
        )
        ignored_self_ref_mismatches = sorted(
            rel_path
            for rel_path in bundle_mismatched
            if rel_path in SELF_REFERENTIAL_BUNDLE_HASH_PATHS
        )
        unexpected_bundle_mismatches = sorted(
            rel_path
            for rel_path in bundle_mismatched
            if rel_path not in SELF_REFERENTIAL_BUNDLE_HASH_PATHS
        )
        missing_from_package_checksums = sorted(
            str(record.get("relative_path") or "")
            for record in bundle_records
            if str(record.get("relative_path") or "") not in sha_record_map
        )
        missing_required_paths = [
            suffix
            for suffix in _required_suffixes()
            if not (root / suffix).exists()
        ]
        report_quality_offenders = _extracted_report_quality_offenders(root, bundle)
        chart_quality_offenders = _extracted_chart_quality_offenders(root, bundle)
        identity_hash_offenders = _extracted_identity_hash_offenders(root, sha_records)
        _check(
            checks,
            "package checksum list verifies",
            sha_list_error is None
            and len(sha_records) >= 60
            and not invalid_sha_paths
            and not sha_missing
            and not sha_mismatched,
            (
                f"records={len(sha_records)}, error={sha_list_error}, "
                f"invalid_paths={invalid_sha_paths[:10]}, "
                f"missing={sha_missing[:10]}, mismatched={sha_mismatched[:10]}"
            ),
        )
        _check(
            checks,
            "share-bundle listed files verify",
            len(bundle_records) >= 60
            and not bundle_missing
            and not unexpected_bundle_mismatches
            and not bundle_hash_conflicts
            and not missing_from_package_checksums,
            (
                f"records={len(bundle_records)}, raw_records={len(raw_bundle_records)}, "
                f"duplicates={len(bundle_duplicates)}, missing={bundle_missing[:10]}, "
                f"mismatched={unexpected_bundle_mismatches[:10]}, "
                f"self_ref_hash_mismatches={ignored_self_ref_mismatches}, "
                f"hash_conflicts={bundle_hash_conflicts[:10]}, "
                f"missing_from_package_checksums={missing_from_package_checksums[:10]}"
            ),
        )
        _check(
            checks,
            "extracted bundle contains quick-read and stage-budget assets",
            not missing_required_paths
            and not report_quality_offenders
            and not chart_quality_offenders
            and not identity_hash_offenders,
            (
                f"missing={missing_required_paths}, "
                f"report_quality_offenders={report_quality_offenders[:10]}, "
                f"chart_quality_offenders={chart_quality_offenders[:10]}, "
                f"identity_hash_offenders={identity_hash_offenders[:10]}"
            ),
        )
        _check(
            checks,
            "package README explains adjacent validation artifacts",
            package_readme_error is None and package_readme_ready,
            f"error={package_readme_error}, missing={package_readme_missing}",
        )
    else:
        actual_sha = _sha256(tarball) if tarball.is_file() else None
        expected_sha, checksum_text = _checksum_value(checksum)
        tar_names, tar_error = (
            _tar_members(tarball) if tarball.is_file() else (set(), "missing tarball")
        )
        expected_members = _expected_bundle_members(bundle, arc_prefix=arc_prefix)
        missing_members = sorted(
            member for member in expected_members if member not in tar_names
        )
        arc_member_prefix = f"{arc_prefix}/"
        expected_member_rel_paths = sorted(
            member[len(arc_member_prefix) :]
            for member in expected_members
            if member.startswith(arc_member_prefix)
        )
        missing_suffixes = [
            suffix
            for suffix in _required_suffixes()
            if not any(name.endswith(suffix) for name in tar_names)
        ]
        report_quality_offenders = _tar_report_quality_offenders(
            tarball, bundle, arc_prefix=arc_prefix
        )
        chart_quality_offenders = _tar_chart_quality_offenders(
            tarball, bundle, arc_prefix=arc_prefix
        )
        identity_hash_offenders = _tar_identity_hash_offenders(
            tarball, arc_prefix=arc_prefix
        )
        sha_list_member = f"{arc_prefix}/{PACKAGE_SHA_LIST.as_posix()}"
        tar_sha_text, tar_sha_error = (
            _tar_member_text(tarball, sha_list_member)
            if tarball.is_file()
            else ("", "missing tarball")
        )
        tar_sha_raw_records = (
            _parse_sha_list_text(tar_sha_text) if tar_sha_error is None else []
        )
        (
            tar_sha_records,
            tar_sha_duplicates,
            tar_sha_hash_conflicts,
        ) = _dedupe_records(tar_sha_raw_records)
        tar_sha_record_map = _sha_record_map(tar_sha_records)
        tar_invalid_sha_paths = _invalid_relative_paths(tar_sha_raw_records)
        missing_expected_from_tar_sha = sorted(
            rel_path
            for rel_path in expected_member_rel_paths
            if rel_path not in tar_sha_record_map
        )
        unexpected_tar_sha_paths = sorted(
            rel_path
            for rel_path in tar_sha_record_map
            if rel_path not in set(expected_member_rel_paths)
        )
        (
            tar_sha_missing_members,
            tar_sha_mismatched_members,
            tar_sha_read_error,
        ) = _tar_records_hash_status(
            tarball, tar_sha_records, arc_prefix=arc_prefix
        )
        _check(
            checks,
            "tarball exists",
            tarball.is_file() and tarball.stat().st_size > 0,
            (
                f"{tarball}, "
                f"size={tarball.stat().st_size if tarball.is_file() else 'missing'}"
            ),
        )
        _check(
            checks,
            "tarball checksum matches",
            bool(actual_sha and expected_sha and actual_sha == expected_sha),
            (
                f"actual={actual_sha}, checksum={expected_sha}, "
                f"file={checksum}, text={checksum_text}"
            ),
        )
        _check(
            checks,
            "package manifest consistency",
            bool(package_summary.get("ready"))
            and _int_value(package_summary.get("file_count")) >= 70
            and package.get("tarball_sha256") == actual_sha,
            (
                f"ready={package_summary.get('ready')}, "
                f"file_count={package_summary.get('file_count')}, "
                f"tarball_sha256={package.get('tarball_sha256')}"
            ),
            required=bool(package),
        )
        _check(
            checks,
            "tarball is readable",
            tar_error is None and len(tar_names) >= 60,
            f"members={len(tar_names)}, error={tar_error}",
        )
        _check(
            checks,
            "tarball contains all share-bundle files",
            not missing_members,
            f"expected={len(expected_members)}, missing={missing_members[:10]}",
        )
        _check(
            checks,
            "tarball internal checksum list verifies",
            tar_sha_error is None
            and len(tar_sha_records) == len(expected_member_rel_paths)
            and len(tar_sha_records) >= 70
            and not tar_invalid_sha_paths
            and not tar_sha_duplicates
            and not tar_sha_hash_conflicts
            and not missing_expected_from_tar_sha
            and not unexpected_tar_sha_paths
            and not tar_sha_missing_members
            and not tar_sha_mismatched_members
            and tar_sha_read_error is None,
            (
                f"records={len(tar_sha_records)}, expected={len(expected_member_rel_paths)}, "
                f"error={tar_sha_error}, read_error={tar_sha_read_error}, "
                f"invalid_paths={tar_invalid_sha_paths[:10]}, "
                f"duplicates={tar_sha_duplicates[:10]}, "
                f"hash_conflicts={tar_sha_hash_conflicts[:10]}, "
                f"missing_expected={missing_expected_from_tar_sha[:10]}, "
                f"unexpected={unexpected_tar_sha_paths[:10]}, "
                f"missing_members={tar_sha_missing_members[:10]}, "
                f"mismatched={tar_sha_mismatched_members[:10]}"
            ),
        )
        _check(
            checks,
            "tarball contains quick-read and stage-budget assets",
            not missing_suffixes
            and not report_quality_offenders
            and not chart_quality_offenders
            and not identity_hash_offenders,
            (
                f"missing_suffixes={missing_suffixes}, "
                f"report_quality_offenders={report_quality_offenders[:10]}, "
                f"chart_quality_offenders={chart_quality_offenders[:10]}, "
                f"identity_hash_offenders={identity_hash_offenders[:10]}"
            ),
        )
        _check(
            checks,
            "package README explains adjacent validation artifacts",
            package_readme_error is None and package_readme_ready,
            f"error={package_readme_error}, missing={package_readme_missing}",
        )

    _build_common_checks(
        checks,
        audit=audit,
        final=final,
        bundle_summary=bundle_summary,
        final_summary=final_summary,
        chart_summary=chart_summary,
        stage_summary=stage_summary,
        stage_ledger_summary=stage_ledger_summary,
        runtime_summary=runtime_summary,
        rerun_summary=rerun_summary,
        rerun_delta_summary=rerun_delta_summary,
        protocol_summary=protocol_summary,
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
        "mode": "extracted_only" if extracted_only else "tarball",
        "tarball": str(tarball),
        "checksum": str(checksum),
        "arc_prefix": arc_prefix,
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "warnings": len(warnings),
            "tarball_sha256": actual_sha,
            "tar_members": len(tar_names),
            "expected_bundle_members": len(expected_members),
            "missing_bundle_members": len(missing_members),
            "extracted_only": extracted_only,
            "identity_hash_offenders_total": len(identity_hash_offenders),
            "identity_hash_clean": not identity_hash_offenders,
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "warnings": [check.to_dict() for check in warnings],
    }


def attach_receiver_smoke_check(
    payload: dict[str, Any],
    *,
    extract_dir: Path,
    tarball: Path,
    checksum: Path,
) -> None:
    summary = payload["summary"]
    receiver_smoke: dict[str, Any] = {
        "requested": True,
        "extract_dir": str(extract_dir),
        "ready": False,
        "skipped_reason": None,
        "unsafe_members": [],
        "extract_error": None,
        "extracted_root": None,
        "extracted_validation_summary": {},
    }
    if summary.get("extracted_only"):
        receiver_smoke["skipped_reason"] = "receiver smoke is only valid in tarball mode"
    elif not summary.get("ready"):
        receiver_smoke["skipped_reason"] = "tarball validation did not pass"
    else:
        unsafe_members, extract_error = _safe_extract_tarball(tarball, extract_dir)
        extracted_root = extract_dir / str(
            payload.get("arc_prefix") or "qwen35_omni_share_bundle_20260621"
        )
        receiver_smoke["unsafe_members"] = unsafe_members
        receiver_smoke["extract_error"] = extract_error
        receiver_smoke["extracted_root"] = str(extracted_root)
        if not unsafe_members and extract_error is None:
            extracted_payload = build_validation(
                extracted_root,
                tarball,
                checksum,
                extracted_only=True,
            )
            receiver_smoke["extracted_validation_summary"] = extracted_payload[
                "summary"
            ]
            receiver_smoke["extracted_validation_checks"] = extracted_payload[
                "checks"
            ]
            receiver_smoke["ready"] = bool(extracted_payload["summary"].get("ready"))
    payload["receiver_smoke"] = receiver_smoke
    summary["receiver_smoke_requested"] = True
    summary["receiver_smoke_ready"] = bool(receiver_smoke["ready"])
    summary["ready"] = bool(summary.get("ready")) and bool(receiver_smoke["ready"])


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Share Package Validation\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Warnings | {summary['warnings']} |")
    print(f"| Mode | {payload['mode']} |")
    print(f"| Identity hash clean | {summary.get('identity_hash_clean')} |")
    print(
        "| Identity hash offenders | "
        f"{summary.get('identity_hash_offenders_total')} |"
    )
    if not summary.get("extracted_only"):
        print(f"| Tar members | {summary['tar_members']} |")
        print(f"| SHA-256 | `{summary['tarball_sha256']}` |")
    print("\n| Status | Required | Check | Evidence |")
    print("| --- | --- | --- | --- |")
    for check in payload["checks"]:
        evidence = str(check["evidence"]).replace("|", "\\|")
        required = "yes" if check["required"] else "no"
        print(f"| {check['status']} | {required} | {check['name']} | {evidence} |")
    receiver_smoke = payload.get("receiver_smoke")
    if receiver_smoke:
        print("\n| Receiver Smoke | Value |")
        print("| --- | ---: |")
        print(f"| Ready | {receiver_smoke.get('ready')} |")
        print(f"| Extracted root | `{receiver_smoke.get('extracted_root')}` |")
        print(f"| Skipped reason | {receiver_smoke.get('skipped_reason')} |")
        print(f"| Extract error | {receiver_smoke.get('extract_error')} |")
        print(f"| Unsafe members | {len(receiver_smoke.get('unsafe_members') or [])} |")
        extracted_summary = receiver_smoke.get("extracted_validation_summary") or {}
        if extracted_summary:
            print(
                "| Extracted checks | "
                f"{extracted_summary.get('checks_passed')}/"
                f"{extracted_summary.get('checks_total')} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the Qwen3.5-Omni share package tarball and gates."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--checksum", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--extracted-only",
        action="store_true",
        help="Validate files in an extracted share bundle without requiring the tarball.",
    )
    parser.add_argument(
        "--receiver-smoke-dir",
        type=Path,
        default=None,
        help=(
            "In tarball mode, safely extract the tarball into this directory and "
            "run extracted-only validation against the extracted bundle root."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    tarball = args.tarball if args.tarball.is_absolute() else root / args.tarball
    checksum = args.checksum
    if checksum is None:
        checksum = _checksum_path(tarball)
    elif not checksum.is_absolute():
        checksum = root / checksum
    payload = build_validation(
        root,
        tarball,
        checksum,
        extracted_only=args.extracted_only,
    )
    if args.receiver_smoke_dir is not None:
        receiver_smoke_dir = args.receiver_smoke_dir
        if not receiver_smoke_dir.is_absolute():
            receiver_smoke_dir = root / receiver_smoke_dir
        attach_receiver_smoke_check(
            payload,
            extract_dir=receiver_smoke_dir,
            tarball=tarball,
            checksum=checksum,
        )

    output = args.json_output
    if not output.is_absolute():
        output = root / output
    _save_json(payload, output)
    print_markdown(payload)
    print(
        "Share package validation written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
