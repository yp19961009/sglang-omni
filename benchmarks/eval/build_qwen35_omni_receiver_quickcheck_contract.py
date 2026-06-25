# SPDX-License-Identifier: Apache-2.0
"""Build the receiver-quickcheck contract for the Qwen3.5-Omni share package."""

from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "receiver_quickcheck_contract.json"
DEFAULT_TARBALL = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"
QUICKCHECK_SCRIPT = Path("benchmarks/eval/qwen35_omni_receiver_quickcheck.sh")
EVIDENCE_CARD_SMOKE_SCRIPT = Path(
    "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh"
)
STAGE_METRIC_DICTIONARY_DOC = Path(
    "benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md"
)

EXPECTED_SUMMARIES = {
    "tarball": {
        "path": AUDIT_DIR / "share_package_validation.json",
        "checks": 17,
        "extra": {},
    },
    "receiver_smoke": {
        "path": AUDIT_DIR / "share_package_receiver_smoke_validation.json",
        "checks": 17,
        "extra": {"receiver_smoke_ready": True},
    },
    "extracted": {
        "path": AUDIT_DIR / "share_package_validation_extracted.json",
        "checks": 13,
        "extra": {"extracted_only": True},
    },
    "standalone": {
        "path": AUDIT_DIR / "share_package_external_standalone_validation.json",
        "checks": 8,
        "extra": {"repo_independent_invocation": True},
    },
}

PUBLIC_DOCS = [
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
]

TRIAGE_DOCS = [
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
]

COMPLETION_GATE_DOCS = [
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
]

EVIDENCE_SMOKE_EXPLICIT_DOCS = [
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
]

WER_ASR_DOCS = {
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md": [
        "WER/ASR 快查",
        "claims_verification.json",
        "headline_scorecard.json",
        "check_wer_asr_path",
        "asr_router_or_container_cache_required",
        "optional warning",
        "/root/.cache/whisper/large-v3.pt",
        "同一 ASR/WER 路径",
        "替换 headline",
    ],
    "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md": [
        "复现 SGLang WER",
        "ASR/WER 路径",
        "asr_router_or_container_cache_required",
        "optional warning",
        "/root/.cache/whisper/large-v3.pt",
        "compute_audio_consistency_from_results",
        "whisper_large_v3_local_wer.json",
        "throughput 峰值不是通过牺牲语音一致性得到的",
    ],
    "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md": [
        "同 ASR/WER 口径",
        "/root/.cache/whisper/large-v3.pt",
        "optional",
        "ASR router",
        "WER/ASR 结果不可复现",
        "不替换吞吐结论",
    ],
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


def _ordered(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    positions: list[int] = []
    missing: list[str] = []
    for needle in needles:
        pos = text.find(needle)
        if pos < 0:
            missing.append(needle)
        positions.append(pos)
    return not missing and positions == sorted(positions), missing


def _summary_ready(summary: dict[str, Any], checks_total: int, extra: dict[str, Any]) -> bool:
    if not bool(summary.get("ready")):
        return False
    if int(summary.get("checks_passed") or -1) != checks_total:
        return False
    if int(summary.get("checks_total") or -1) != checks_total:
        return False
    if int(summary.get("required_failures") or 0) != 0:
        return False
    for key, expected in extra.items():
        if summary.get(key) != expected:
            return False
    return True


def _quality_clean(payload: dict[str, Any]) -> bool:
    evidence_parts = []
    for check in payload.get("checks", []):
        if isinstance(check, dict):
            evidence_parts.append(str(check.get("evidence") or ""))
    evidence = "\n".join(evidence_parts)
    return (
        "report_quality_offenders=[]" in evidence
        and "chart_quality_offenders=[]" in evidence
        and "identity_hash_offenders=[]" in evidence
    )


def _receiver_summary_for_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep receiver status fields but omit tarball identity hashes."""

    keep_keys = [
        "ready",
        "checks_total",
        "checks_passed",
        "required_failures",
        "warnings",
        "tar_members",
        "expected_bundle_members",
        "missing_bundle_members",
        "extracted_only",
        "receiver_smoke_ready",
        "extracted_validation_ready",
        "extracted_validation_checks",
        "extracted_validation_required_failures",
        "repo_independent_invocation",
        "identity_hash_offenders_total",
        "identity_hash_clean",
    ]
    return {key: summary.get(key) for key in keep_keys if key in summary}


def _tar_has_members(tarball: Path, arc_prefix: str) -> tuple[bool, list[str], str | None]:
    required = [
        f"{arc_prefix}/benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
        f"{arc_prefix}/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
        f"{arc_prefix}/benchmarks/eval/validate_qwen35_omni_share_package.py",
        f"{arc_prefix}/benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
        f"{arc_prefix}/benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
        f"{arc_prefix}/results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
    ]
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            names = set(tf.getnames())
    except Exception as exc:
        return False, required, str(exc)
    missing = [name for name in required if name not in names]
    return not missing, missing, None


def build_contract(root: Path, *, tarball: Path, arc_prefix: str) -> dict[str, Any]:
    root = root.resolve()
    tarball = (tarball if tarball.is_absolute() else root / tarball).resolve()
    script_text = _read_text_optional(root / QUICKCHECK_SCRIPT)
    evidence_smoke_text = _read_text_optional(root / EVIDENCE_CARD_SMOKE_SCRIPT)
    checks: list[Check] = []

    step_needles = [
        "[1/6] checksum",
        "[2/6] tarball-mode validation",
        "[3/6] receiver-smoke validation",
        "[4/6] extracted-only validation",
        "[5/6] external standalone validation",
        "[6/6] summary",
    ]
    ordered_steps, missing_steps = _ordered(script_text, step_needles)
    _check(
        checks,
        "quickcheck script has ordered six-step receiver flow",
        ordered_steps,
        f"missing_steps={missing_steps}",
    )

    script_needles = [
        "sha256sum -c \"$CHECKSUM\"",
        "--json-output \"${AUDIT_DIR_ABS}/share_package_validation.json\"",
        "--receiver-smoke-dir \"$SMOKE_DIR\"",
        "--json-output \"${AUDIT_DIR_ABS}/share_package_receiver_smoke_validation.json\"",
        "--extracted-only",
        "--json-output \"${AUDIT_DIR_ABS}/share_package_validation_extracted.json\"",
        "build_qwen35_omni_external_standalone_bundle_validation",
        "--work-dir \"$STANDALONE_DIR\"",
        "--json-output \"${AUDIT_DIR_ABS}/share_package_external_standalone_validation.json\"",
    ]
    missing_script_needles = [
        needle for needle in script_needles if needle not in script_text
    ]
    _check(
        checks,
        "quickcheck script writes all receiver evidence JSONs",
        not missing_script_needles,
        f"missing={missing_script_needles}",
    )

    safety_needles = [
        "set -euo pipefail",
        "prepare_tmp_dir",
        "ALLOW_NON_TMP_EXTRACT",
        '[[ -n "$target" && "$target" != "/" ]]',
        '"$target" != /tmp/*',
        "rm -rf \"$target\"",
        "STANDALONE_DIR",
    ]
    missing_safety = [needle for needle in safety_needles if needle not in script_text]
    _check(
        checks,
        "quickcheck script keeps destructive temp cleanup bounded",
        not missing_safety,
        f"missing={missing_safety}",
    )

    evidence_summaries: dict[str, dict[str, Any]] = {}
    evidence_ready = True
    evidence_failures: list[str] = []
    quality_failures: list[str] = []
    for label, spec in EXPECTED_SUMMARIES.items():
        payload = _load_json_optional(root / spec["path"])
        summary = payload.get("summary", {})
        contract_summary = _receiver_summary_for_contract(summary)
        evidence_summaries[label] = contract_summary
        if not _summary_ready(summary, int(spec["checks"]), dict(spec["extra"])):
            evidence_ready = False
            evidence_failures.append(f"{label}:{contract_summary}")
        if label in {"tarball", "receiver_smoke", "extracted", "standalone"} and not _quality_clean(payload):
            quality_failures.append(label)
    _check(
        checks,
        "current receiver evidence JSONs are all green",
        evidence_ready,
        "; ".join(evidence_failures),
    )
    _check(
        checks,
        "current receiver evidence is asset-quality clean",
        not quality_failures,
        f"quality_failures={quality_failures}",
    )

    docs_missing: dict[str, list[str]] = {}
    doc_needles = [
        "qwen35_omni_receiver_quickcheck.sh",
        "tarball",
        "receiver",
        "extracted-only",
        "standalone",
    ]
    for rel_path in PUBLIC_DOCS:
        text = _read_text_optional(root / rel_path)
        missing = [needle for needle in doc_needles if needle not in text]
        if missing:
            docs_missing[rel_path] = missing
    _check(
        checks,
        "public receiver docs route to the quickcheck contract",
        not docs_missing,
        f"missing={docs_missing}",
    )

    command_card_text = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md"
    )
    stage_dictionary_text = _read_text_optional(root / STAGE_METRIC_DICTIONARY_DOC)
    evidence_smoke_cli_needles = [
        "--root|--bundle-root",
        "--mode",
        "--output",
        "PASS/skip summary stays on stdout",
        "summary_output=stdout",
        "MODE must be host, portable, or auto",
        "portable mode skips card 10",
        'if [[ "$MODE" == "host" ]]',
        "HOST_REPO:-${BUNDLE_ROOT:-$(pwd)}",
    ]
    evidence_smoke_cli_missing = [
        needle for needle in evidence_smoke_cli_needles if needle not in evidence_smoke_text
    ]
    evidence_card_route_missing = [
        needle
        for needle in [
            str(EVIDENCE_CARD_SMOKE_SCRIPT),
            "qwen35_omni_evidence_query_cards_zh_20260621.md",
            "host 仓库态",
            "portable 子集",
            "只读 smoke",
            '--root "$HOST_REPO" --mode host',
            '--root "$BUNDLE_ROOT" --mode portable',
            "qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
            "qwen35_omni_stage_metric_dictionary_zh_20260621.md",
            "quick_reproduction_map",
            "5 条答辩 quick route",
            "lifecycle/compute/handoff/collect wait",
            "按 command id",
            "repro_command_manifest.json",
            "jq -r --arg id sglang_videoamme_stress",
            "vllm_c8_prebuild_w4",
            ".commands[] | select(.id == $id) | .command",
            "`PASS`/skip 摘要打印在 stdout",
            "`--output` 只保存查询卡 bash block",
            "*.summary.out",
            "*.query.out",
        ]
        if needle not in command_card_text
    ]
    explicit_smoke_doc_missing: dict[str, list[str]] = {}
    for rel_path in EVIDENCE_SMOKE_EXPLICIT_DOCS:
        text = _read_text_optional(root / rel_path)
        missing = [
            needle
            for needle in [
                str(EVIDENCE_CARD_SMOKE_SCRIPT),
                "--mode host",
                "--mode portable",
            ]
            if needle not in text
        ]
        if missing:
            explicit_smoke_doc_missing[rel_path] = missing
    _check(
        checks,
        "receiver command card routes to evidence-query smoke, stage dictionary, and command-id lookup",
        not evidence_card_route_missing
        and not evidence_smoke_cli_missing
        and not explicit_smoke_doc_missing,
        (
            f"route_missing={evidence_card_route_missing}; "
            f"cli_missing={evidence_smoke_cli_missing}; "
            f"explicit_doc_missing={explicit_smoke_doc_missing}"
        ),
    )
    stage_dictionary_crosswalk_needles = [
        "指标到证据/复跑入口",
        "metric_provenance_index.json",
        "stage_reproduction_drilldown.json",
        "quick_reproduction_map",
        "rerun_command_ids",
        "vllm_admission_diagnosis.json",
        "sglang_videoamme_stress",
        "vllm_c8_prebuild_w4",
    ]
    stage_dictionary_crosswalk_missing = [
        needle
        for needle in stage_dictionary_crosswalk_needles
        if needle not in stage_dictionary_text
    ]
    _check(
        checks,
        "stage metric dictionary carries evidence and rerun crosswalk",
        not stage_dictionary_crosswalk_missing,
        f"missing={stage_dictionary_crosswalk_missing}",
    )

    completion_gate_missing: dict[str, list[str]] = {}
    completion_gate_needles = [
        "final_completion_audit.json",
        "qwen35_omni_final_completion_audit_zh_20260621.md",
        "completion_allowed_now",
    ]
    for rel_path in COMPLETION_GATE_DOCS:
        text = _read_text_optional(root / rel_path)
        missing = [needle for needle in completion_gate_needles if needle not in text]
        if missing:
            completion_gate_missing[rel_path] = missing
    _check(
        checks,
        "public receiver docs route to the final completion gate",
        not completion_gate_missing,
        f"missing={completion_gate_missing}",
    )

    triage_missing: dict[str, list[str]] = {}
    triage_needles = [
        "快检失败分流",
        "share_package_validation.json",
        "share_package_receiver_smoke_validation.json",
        "share_package_validation_extracted.json",
        "share_package_external_standalone_validation.json",
        "report_quality_offenders",
        "chart_quality_offenders",
    ]
    for rel_path in TRIAGE_DOCS:
        text = _read_text_optional(root / rel_path)
        missing = [needle for needle in triage_needles if needle not in text]
        if missing:
            triage_missing[rel_path] = missing
    _check(
        checks,
        "public receiver docs include quickcheck failure triage",
        not triage_missing,
        f"missing={triage_missing}",
    )

    wer_asr_missing: dict[str, list[str]] = {}
    for rel_path, needles in WER_ASR_DOCS.items():
        text = _read_text_optional(root / rel_path)
        missing = [needle for needle in needles if needle not in text]
        if missing:
            wer_asr_missing[rel_path] = missing
    wer_asr_guard_needles = sorted(
        {needle for needles in WER_ASR_DOCS.values() for needle in needles}
    )
    _check(
        checks,
        "public receiver docs preserve WER/ASR rerun path",
        not wer_asr_missing,
        (
            f"checked_docs={list(WER_ASR_DOCS)}; "
            f"guard_needles={wer_asr_guard_needles}; "
            f"missing={wer_asr_missing}"
        ),
    )

    self_ref_docs = {
        "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md": [
            "in_progress=true",
            "recovered_from_in_progress_gates=True",
            "direct_rerun_delta_triage",
            "rows_total>=19",
            "checks_passed>=8",
            "required_failures=0",
            "顶层暴露",
            "rerun_delta_triage",
        ],
        "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md": [
            "audit summary 自引用状态",
            "Tarball 相邻伴随终检证据",
            "不是 `BUNDLE_ROOT` 内成员",
            "final_completion_audit.json",
            "share_release_seal.json",
            "in_progress=true",
            "tarball 自引用哈希",
            "direct_rerun_delta_triage",
            "rows_total>=19",
            "checks_passed>=8",
            "required_failures=0",
            "顶层直接暴露",
            "rerun_delta_triage",
        ],
        "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md": [
            "in_progress=true",
            "direct_rerun_delta_triage",
            "rows_total>=19",
            "checks_passed>=8",
            "required_failures=0",
            "自举状态",
            "rerun_delta_triage",
        ],
    }
    self_ref_missing: dict[str, list[str]] = {}
    for rel_path, needles in self_ref_docs.items():
        text = _read_text_optional(root / rel_path)
        missing = [needle for needle in needles if needle not in text]
        if missing:
            self_ref_missing[rel_path] = missing
    _check(
        checks,
        "public receiver docs explain audit-summary self-reference recovery",
        not self_ref_missing,
        f"missing={self_ref_missing}",
    )

    repro = _load_json_optional(root / AUDIT_DIR / "repro_command_manifest.json")
    commands = {
        str(command.get("id") or command.get("command_id")): command
        for command in repro.get("commands", [])
        if isinstance(command, dict)
    }
    receiver_cmd = commands.get("validate_share_bundle_receiver_smoke", {})
    receiver_command_text = "\n".join(
        str(receiver_cmd.get(key) or "") for key in ("purpose", "command", "expected")
    )
    command_evidence = receiver_cmd.get("evidence_after_run") or []
    command_ok = (
        "receiver quickcheck wrapper" in receiver_command_text
        and "qwen35_omni_receiver_quickcheck.sh" in receiver_command_text
        and all(
            str(spec["path"]) in command_evidence
            for spec in EXPECTED_SUMMARIES.values()
        )
    )
    _check(
        checks,
        "repro command manifest binds quickcheck to all receiver evidence files",
        command_ok,
        f"command_id=validate_share_bundle_receiver_smoke, evidence_after_run={command_evidence}",
    )

    share_bundle = _load_json_optional(root / AUDIT_DIR / "share_bundle_manifest.json")
    bundle_paths = {
        str(record.get("relative_path") or "")
        for record in share_bundle.get("records", [])
        if isinstance(record, dict)
    }
    bundle_ok = (
        str(QUICKCHECK_SCRIPT) in bundle_paths
        and str(EVIDENCE_CARD_SMOKE_SCRIPT) in bundle_paths
        and str(DEFAULT_OUTPUT) in bundle_paths
        and str(DEFAULT_JSON_OUTPUT) in bundle_paths
    )
    _check(
        checks,
        "share bundle manifest includes quickcheck tool and contract evidence",
        bundle_ok,
        f"quickcheck_present={str(QUICKCHECK_SCRIPT) in bundle_paths}, "
        f"evidence_smoke_present={str(EVIDENCE_CARD_SMOKE_SCRIPT) in bundle_paths}, "
        f"contract_report_present={str(DEFAULT_OUTPUT) in bundle_paths}, "
        f"contract_json_present={str(DEFAULT_JSON_OUTPUT) in bundle_paths}",
        required=False,
    )

    tar_ok, tar_missing, tar_error = _tar_has_members(tarball, arc_prefix)
    _check(
        checks,
        "tarball contains quickcheck tool and contract evidence",
        tar_ok,
        f"missing={tar_missing}, error={tar_error}",
        required=False,
    )

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    warnings = [check for check in checks if not check.required and check.status != "PASS"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "quickcheck_script": str(QUICKCHECK_SCRIPT),
        "evidence_files": {
            label: str(spec["path"]) for label, spec in EXPECTED_SUMMARIES.items()
        },
        "public_docs": PUBLIC_DOCS,
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "warnings": len(warnings),
            "tarball_contract_checked": tarball.is_file(),
            "tarball_contains_contract": tar_ok,
            "quickcheck_steps": len(step_needles),
            "receiver_jsons_total": len(EXPECTED_SUMMARIES),
            "public_docs_total": len(PUBLIC_DOCS),
            "completion_gate_docs_total": len(COMPLETION_GATE_DOCS),
            "wer_asr_docs_total": len(WER_ASR_DOCS),
            "evidence_smoke_cli_options": len(evidence_smoke_cli_needles),
            "evidence_smoke_explicit_docs_total": len(EVIDENCE_SMOKE_EXPLICIT_DOCS),
            "stage_dictionary_crosswalk_needles_total": len(
                stage_dictionary_crosswalk_needles
            ),
        },
        "evidence_summaries": evidence_summaries,
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni 接收方 Quickcheck Contract",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份 contract 用来证明接收方第一条命令没有漂移：",
        "`qwen35_omni_receiver_quickcheck.sh` 必须同时覆盖 checksum、tarball-mode validation、",
        "receiver-smoke validation、extracted-only validation 和 external standalone validation。",
        "命令卡还必须提供 `qwen35_omni_evidence_query_cards_smoke.sh`，用于只读验证",
        "`qwen35_omni_evidence_query_cards_zh_20260621.md` 里的查询入口仍能命中关键证据；",
        "也必须把 `qwen35_omni_stage_metric_dictionary_zh_20260621.md` 放进 stage",
        "读法入口，避免 reviewer 混淆 lifecycle、compute、handoff 和 collect wait。",
        "stage metric dictionary 还必须保留 evidence/rerun crosswalk，把指标口径连到",
        "`metric_provenance_index.json`、`stage_reproduction_drilldown.json`、",
        "`quick_reproduction_map` 和 `rerun_command_ids`。",
        "命令卡还必须能从 `repro_command_manifest.json` 按 command id 直接抽出",
        "`sglang_videoamme_stress` 和 `vllm_c8_prebuild_w4` 等复跑命令。",
        "",
        "## 1. Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary['ready']} |",
        f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |",
        f"| Required failures | {summary['required_failures']} |",
        f"| Warnings | {summary['warnings']} |",
        f"| Receiver JSONs | {summary['receiver_jsons_total']} |",
        f"| Public docs | {summary['public_docs_total']} |",
        f"| Completion-gate docs | {summary['completion_gate_docs_total']} |",
        f"| WER/ASR docs | {summary['wer_asr_docs_total']} |",
        f"| Evidence smoke CLI options | {summary['evidence_smoke_cli_options']} |",
        f"| Evidence smoke explicit docs | {summary['evidence_smoke_explicit_docs_total']} |",
        "",
        "## 2. Receiver Evidence",
        "",
        "| Label | File | Ready | Checks | Required failures |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, rel_path in payload["evidence_files"].items():
        evidence_summary = payload["evidence_summaries"].get(label, {})
        lines.append(
            "| {label} | `{path}` | {ready} | {passed}/{total} | {failures} |".format(
                label=label,
                path=rel_path,
                ready=evidence_summary.get("ready"),
                passed=evidence_summary.get("checks_passed"),
                total=evidence_summary.get("checks_total"),
                failures=evidence_summary.get("required_failures"),
            )
        )
    lines.extend(
        [
            "",
            "## 3. Contract Checks",
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
            "## 4. 复现入口",
            "",
            "```bash",
            'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'cd "$HOST_REPO"',
            "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
            'bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host',
            'BUNDLE_ROOT="${BUNDLE_ROOT:-/tmp/qwen35_omni_share_bundle_final/qwen35_omni_share_bundle_20260621}"',
            'bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable',
            "jq '.summary' results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
            "```",
            "",
            "通过标准：quickcheck summary 显示 tarball `17/17`、receiver smoke `17/17`、",
            "extracted-only `13/13`、standalone `8/8`；contract JSON 显示 `ready=true`、",
            "`required_failures=0`；evidence-query smoke 显示 host 或 portable 查询卡全部通过。",
            "最终 goal completion 另看相邻 `final_completion_audit.json` 和",
            "`qwen35_omni_final_completion_audit_zh_20260621.md`；更新后的目标不再等待 6.21 晚上，",
            "`completion_allowed_now=true` 表示可以进入最终完成裁决。",
            "",
            "解包态如果因为 tarball 自引用哈希看到 `audit_run_summary.json` 为 `in_progress=true`，",
            "只能在公开接收方文档解释 `recovered_from_in_progress_gates=True` 和",
            "`direct_rerun_delta_triage` 约束时接受：`rows_total>=19`、`checks_passed>=8`、",
            "`required_failures=0`。仓库根最终 summary 必须是完成态 `ok=true`，并在顶层暴露",
            "`rerun_delta_triage`。",
            "",
            "## 5. 快检失败分流",
            "",
            "| 失败点 | 先看证据 | 裁决 |",
            "| --- | --- | --- |",
            "| checksum FAIL | `.tar.gz.sha256` 和 `sha256sum -c` 输出 | 不进入报告阅读；先重传 tarball/checksum。 |",
            "| tarball-mode validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation.json` | 不发送为最终包；修复缺失/mismatch 后重建 tarball。 |",
            "| receiver smoke FAIL | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | 不替换任何 benchmark 数字；先确认安全解包和 nested extracted-only gate。 |",
            "| extracted-only validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 说明解包目录或随包文件不自洽；回到 `BUNDLE_ROOT`/`HOST_REPO` 路径手册排查。 |",
            "| standalone validation FAIL | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 说明随包 validator 不能在干净 `/tmp` 独立运行；先修包，不进入外部复现。 |",
            "| quality offenders 非空 | 对应 validation JSON 里的 `report_quality_offenders` / `chart_quality_offenders` | 先修 Markdown/CSV/SVG 质量，再重跑 full audit 和 release seal。 |",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni receiver quickcheck contract."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--tarball", type=Path, default=DEFAULT_TARBALL)
    parser.add_argument("--arc-prefix", default="qwen35_omni_share_bundle_20260621")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_contract(root, tarball=args.tarball, arc_prefix=args.arc_prefix)

    json_output = args.json_output
    if not json_output.is_absolute():
        json_output = root / json_output
    _save_json(payload, json_output)

    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(payload), encoding="utf-8")

    print("## Qwen3.5-Omni Receiver Quickcheck Contract\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {payload['summary']['ready']} |")
    print(
        f"| Checks | {payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']} |"
    )
    print(f"| Required failures | {payload['summary']['required_failures']} |")
    print(
        "Receiver quickcheck contract written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
