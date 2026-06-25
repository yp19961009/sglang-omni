# SPDX-License-Identifier: Apache-2.0
"""Audit command-reference hygiene for the Qwen3.5-Omni share package."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "command_reference_hygiene.json"

STRUCTURED_ARTIFACTS = {
    "metric_provenance_index": AUDIT_DIR / "metric_provenance_index.json",
    "stage_drilldown_index": AUDIT_DIR / "stage_drilldown_index.json",
    "stage_reproduction_drilldown": AUDIT_DIR / "stage_reproduction_drilldown.json",
    "stage_route_decision_matrix": AUDIT_DIR / "stage_route_decision_matrix.json",
    "defense_claim_matrix": AUDIT_DIR / "defense_claim_matrix.json",
    "claim_metric_crosswalk": AUDIT_DIR / "claim_metric_crosswalk.json",
    "objective_requirement_crosswalk": AUDIT_DIR
    / "objective_requirement_crosswalk.json",
    "university_technical_report": AUDIT_DIR / "university_technical_report.json",
    "runtime_image_contract": AUDIT_DIR / "runtime_image_contract.json",
}

COMMAND_REF_KEYS = {
    "rerun_command_ids",
    "command_ids",
    "reproduction_command_ids",
    "required_reproduction_command_ids",
}

CRITICAL_DOC_COMMAND_IDS = {
    "run_full_audit",
    "launch_sglang_optimized",
    "sglang_videoamme_stress",
    "sglang_synthetic_text_to_speech",
    "sglang_recompute_wer",
    "vllm_c1_original",
    "vllm_c4_original",
    "vllm_c8_original",
    "vllm_c8_prebuild_w4",
    "build_repro_command_manifest",
    "build_share_path_hygiene",
    "validate_external_standalone_share_bundle",
}

CRITICAL_MANIFEST_COMMAND_IDS = CRITICAL_DOC_COMMAND_IDS | {
    "build_command_reference_hygiene",
    "build_final_readiness_audit",
    "build_share_bundle_manifest",
    "build_share_bundle_package",
    "validate_share_bundle_package",
    "validate_share_bundle_receiver_smoke",
    "validate_extracted_share_bundle",
}

PUBLIC_DOC_GLOBS = [
    "benchmarks/reports/qwen35_omni_*_20260621.md",
    "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md",
]


@dataclass(frozen=True)
class HygieneCheck:
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


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _check(
    checks: list[HygieneCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(HygieneCheck(name, _status(condition), evidence, required))


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _collect_command_refs(
    payload: Any,
    *,
    artifact: str,
    json_path: str = "$",
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{json_path}.{key}"
            if key in COMMAND_REF_KEYS:
                refs.extend(
                    {
                        "artifact": artifact,
                        "json_path": child_path,
                        "field": key,
                        "command_id": command_id,
                    }
                    for command_id in _as_string_list(value)
                )
            refs.extend(
                _collect_command_refs(
                    value,
                    artifact=artifact,
                    json_path=child_path,
                )
            )
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            refs.extend(
                _collect_command_refs(
                    item,
                    artifact=artifact,
                    json_path=f"{json_path}[{index}]",
                )
            )
    return refs


def _manifest_command_ids(repro_manifest: dict[str, Any]) -> set[str]:
    return {
        str(command.get("id"))
        for command in repro_manifest.get("commands", [])
        if isinstance(command, dict) and command.get("id")
    }


def _manifest_phase_refs(repro_manifest: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for phase in repro_manifest.get("phases", []):
        if not isinstance(phase, dict):
            continue
        phase_id = str(phase.get("id") or "")
        for command_id in _as_string_list(phase.get("command_ids")):
            refs.append(
                {
                    "phase": phase_id,
                    "command_id": command_id,
                }
            )
    return refs


def _public_docs(root: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    seen: set[Path] = set()
    for pattern in PUBLIC_DOC_GLOBS:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rel_path = str(path.relative_to(root))
            text = _read_text_optional(path)
            if text:
                docs[rel_path] = text
    return docs


def build_hygiene(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")
    repro_summary = repro_manifest.get("summary", {})
    manifest_ids = _manifest_command_ids(repro_manifest)
    phase_refs = _manifest_phase_refs(repro_manifest)
    missing_manifest_critical_ids = sorted(CRITICAL_MANIFEST_COMMAND_IDS - manifest_ids)
    unresolved_phase_refs = sorted(
        {
            ref["command_id"]
            for ref in phase_refs
            if ref["command_id"] not in manifest_ids
        }
    )

    missing_structured_artifacts: list[str] = []
    structured_refs: list[dict[str, str]] = []
    artifact_ref_counts: dict[str, int] = {}
    for artifact, rel_path in STRUCTURED_ARTIFACTS.items():
        path = root / rel_path
        payload = _load_json_optional(path)
        if not payload:
            missing_structured_artifacts.append(str(rel_path))
            continue
        refs = _collect_command_refs(payload, artifact=artifact)
        artifact_ref_counts[artifact] = len(refs)
        structured_refs.extend(refs)

    structured_unique_refs = sorted({ref["command_id"] for ref in structured_refs})
    unresolved_structured_refs = [
        ref for ref in structured_refs if ref["command_id"] not in manifest_ids
    ]
    unresolved_structured_ids = sorted(
        {ref["command_id"] for ref in unresolved_structured_refs}
    )
    artifacts_with_refs = sorted(
        artifact for artifact, count in artifact_ref_counts.items() if count > 0
    )

    docs = _public_docs(root)
    doc_command_refs: dict[str, list[str]] = {}
    for command_id in sorted(CRITICAL_DOC_COMMAND_IDS):
        doc_command_refs[command_id] = sorted(
            rel_path for rel_path, text in docs.items() if command_id in text
        )
    missing_doc_command_ids = sorted(
        command_id
        for command_id, paths in doc_command_refs.items()
        if not paths
    )
    manifest_doc_refs = sorted(
        rel_path
        for rel_path, text in docs.items()
        if "repro_command_manifest.json" in text
    )
    hygiene_doc_refs = sorted(
        rel_path
        for rel_path, text in docs.items()
        if "command_reference_hygiene.json" in text
    )

    checks: list[HygieneCheck] = []
    _check(
        checks,
        "manifest command inventory gate",
        bool(repro_summary.get("required_command_ids_present"))
        and int(repro_summary.get("commands_total") or 0) >= 60
        and int(repro_summary.get("phases_total") or 0) >= 7
        and not missing_manifest_critical_ids,
        (
            f"commands={repro_summary.get('commands_total')}, "
            f"phases={repro_summary.get('phases_total')}, "
            f"required_ids={repro_summary.get('required_command_ids_present')}, "
            f"manifest_ready={repro_summary.get('ready')}, "
            f"missing_critical={missing_manifest_critical_ids}"
        ),
    )
    _check(
        checks,
        "manifest phase command refs resolve",
        len(phase_refs) >= 60 and not unresolved_phase_refs,
        (
            f"phase_refs={len(phase_refs)}, "
            f"unresolved_phase_refs={unresolved_phase_refs}"
        ),
    )
    _check(
        checks,
        "structured command ref artifacts present",
        not missing_structured_artifacts,
        f"missing={missing_structured_artifacts}",
    )
    _check(
        checks,
        "structured command refs resolve",
        len(artifacts_with_refs) >= 7
        and len(structured_unique_refs) >= 60
        and not unresolved_structured_refs,
        (
            f"artifacts_with_refs={len(artifacts_with_refs)}, "
            f"unique_refs={len(structured_unique_refs)}, "
            f"total_refs={len(structured_refs)}, "
            f"unresolved_ids={unresolved_structured_ids}"
        ),
    )
    _check(
        checks,
        "critical public commands documented",
        not missing_doc_command_ids,
        (
            f"commands={len(CRITICAL_DOC_COMMAND_IDS)}, "
            f"missing={missing_doc_command_ids}"
        ),
    )
    _check(
        checks,
        "public docs point to command manifests",
        len(manifest_doc_refs) >= 6 and len(hygiene_doc_refs) >= 2,
        (
            f"repro_manifest_doc_refs={len(manifest_doc_refs)}, "
            f"command_hygiene_doc_refs={len(hygiene_doc_refs)}"
        ),
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
            "manifest_commands_total": len(manifest_ids),
            "manifest_phase_command_refs_total": len(phase_refs),
            "structured_artifacts_total": len(STRUCTURED_ARTIFACTS),
            "structured_artifacts_with_refs_total": len(artifacts_with_refs),
            "structured_command_refs_total": len(structured_refs),
            "structured_unique_command_refs_total": len(structured_unique_refs),
            "unresolved_command_refs_total": len(unresolved_structured_refs),
            "critical_command_doc_refs_total": sum(
                len(paths) for paths in doc_command_refs.values()
            ),
            "missing_critical_doc_command_ids": missing_doc_command_ids,
            "public_repro_manifest_doc_refs_total": len(manifest_doc_refs),
            "public_command_hygiene_doc_refs_total": len(hygiene_doc_refs),
        },
        "checks": [check.to_dict() for check in checks],
        "diagnostics": {
            "repro_command_manifest_summary": repro_summary,
            "critical_manifest_command_ids": sorted(CRITICAL_MANIFEST_COMMAND_IDS),
            "missing_manifest_critical_ids": missing_manifest_critical_ids,
            "unresolved_phase_refs": unresolved_phase_refs,
            "structured_artifact_ref_counts": artifact_ref_counts,
            "missing_structured_artifacts": missing_structured_artifacts,
            "structured_unique_command_refs": structured_unique_refs,
            "unresolved_structured_command_ids": unresolved_structured_ids,
            "unresolved_structured_refs": unresolved_structured_refs[:100],
            "public_doc_command_refs": doc_command_refs,
            "public_repro_manifest_doc_refs": manifest_doc_refs,
            "public_command_hygiene_doc_refs": hygiene_doc_refs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni command-reference hygiene evidence."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output
    if not output.is_absolute():
        output = root / output
    payload = build_hygiene(root)
    _save_json(payload, output)
    summary = payload["summary"]
    print(
        "[command-reference] "
        f"ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"structured_refs={summary['structured_command_refs_total']} "
        f"unique_refs={summary['structured_unique_command_refs_total']} "
        f"unresolved={summary['unresolved_command_refs_total']} "
        f"critical_doc_missing={summary['missing_critical_doc_command_ids']}"
    )
    if args.strict and not summary["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
