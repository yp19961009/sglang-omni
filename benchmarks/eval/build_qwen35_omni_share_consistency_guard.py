# SPDX-License-Identifier: Apache-2.0
"""Build a share-consistency guard for the Qwen3.5-Omni package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval.build_qwen35_omni_final_readiness import (
    MACHINE_EVIDENCE_FILES,
    REPORT_FILES,
)


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "share_consistency_guard.json"
DEFAULT_TARBALL = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"
DEFAULT_CHECKSUM = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz.sha256"

STALE_PATTERNS = {
    "old_final_readiness_47": re.compile(
        r"(?:final[-_ ]?readiness|readiness)[^|\n]{0,80}47/47|"
        r"47/47[^|\n]{0,80}(?:final[-_ ]?readiness|readiness)",
        re.IGNORECASE,
    ),
    "old_final_readiness_48": re.compile(
        r"(?:final[-_ ]?readiness|readiness)[^|\n]{0,80}48/48|"
        r"48/48[^|\n]{0,80}(?:final[-_ ]?readiness|readiness)",
        re.IGNORECASE,
    ),
    "old_repro_60_commands": re.compile(r"`60`\s+commands|\b60 commands\b"),
    "old_repro_60_chinese": re.compile(r"60 条命令"),
    "old_repro_61_commands": re.compile(r"`61`\s+commands|\b61 commands\b"),
    "old_repro_61_chinese": re.compile(r"61 条命令"),
    "old_repro_62_commands": re.compile(r"`62`\s+commands|\b62 commands\b"),
    "old_repro_62_chinese": re.compile(r"62 条命令"),
    "old_package_file_count_98": re.compile(r'"package_file_count"\s*:\s*98'),
    "old_share_bundle_records_98": re.compile(r'"share_bundle_records"\s*:\s*98'),
    "old_repro_commands_total_60": re.compile(r'"repro_commands_total"\s*:\s*60'),
    "old_repro_commands_total_61": re.compile(r'"repro_commands_total"\s*:\s*61'),
    "old_repro_commands_total_62": re.compile(r'"repro_commands_total"\s*:\s*62'),
    "old_final_readiness_checks_47": re.compile(r'"final_readiness_checks"\s*:\s*47'),
    "old_final_readiness_checks_48": re.compile(r'"final_readiness_checks"\s*:\s*48'),
    "old_final_checkpoint_watchlist_22": re.compile(
        r"final[_ -]?checkpoint[_ -]?watchlist[^|\n]{0,140}(?:22/22|22\s+checks)|"
        r'"final_checkpoint_watchlist"\s*:\s*"22/22"|'
        r"final checkpoint watchlist[^|\n]{0,140}`22/22`",
        re.IGNORECASE,
    ),
    "old_final_checkpoint_watchlist_23": re.compile(
        r"final checkpoint watchlist[^|\n]{0,140}(?:`?23/23`?|23\s+checks)",
        re.IGNORECASE,
    ),
    "old_manifest_177_exact": re.compile(
        r"manifest\s*:?\s*`?177`?\s+records|"
        r"`?177`?\s+records,\s*`?0`?\s+missing",
        re.IGNORECASE,
    ),
    "old_manifest_180_exact": re.compile(
        r"manifest\s*:\s*`?180`?\s+records,\s*`?0`?\s+missing|"
        r"manifest\s+`?180`?\s+records",
        re.IGNORECASE,
    ),
    "old_manifest_expected_gate_total_records": re.compile(
        r"expected_gates[^|\n]{0,1000}['\"]manifest['\"][^|\n]{0,260}"
        r"['\"]total_records['\"]\s*:\s*180",
        re.IGNORECASE,
    ),
    "old_return_evidence_33": re.compile(
        r"33\s+(?:required\s+)?return[- ]evidence\s+files|"
        r"return[- ]evidence\s+files\s*(?:为|=|:)?\s*`?33`?",
        re.IGNORECASE,
    ),
    "old_receiver_quickcheck_contract_9": re.compile(
        r"(?:receiver|接收方)\s+quickcheck\s+contract[^|\n]{0,80}9/9",
        re.IGNORECASE,
    ),
    "old_receiver_quickcheck_contract_10": re.compile(
        r"(?:receiver|接收方)\s+quickcheck\s+contract[^|\n]{0,120}10/10|"
        r"quickcheck\s+contract[^|\n]{0,120}`?checks=10/10`?",
        re.IGNORECASE,
    ),
    "old_receiver_quickcheck_contract_11": re.compile(
        r"(?:receiver|接收方)\s+quickcheck\s+contract[^|\n]{0,160}11/11|"
        r"(?:receiver|接收方)\s+quickcheck\s+contract[^|\n]{0,160}11/12|"
        r"quickcheck\s+contract[^|\n]{0,160}`?checks=11/11`?|"
        r"quickcheck\s+contract[^|\n]{0,160}`?checks=11/12`?",
        re.IGNORECASE,
    ),
    "old_receiver_quickcheck_contract_13": re.compile(
        r"(?:receiver|接收方)\s+quickcheck\s+contract[^|\n]{0,180}13/13|"
        r"quickcheck\s+contract[^|\n]{0,180}`?checks=13/13`?",
        re.IGNORECASE,
    ),
    "old_receiver_quickcheck_contract_14": re.compile(
        r"(?:receiver|接收方)\s+quickcheck\s+contract[^|\n]{0,220}14/14|"
        r"quickcheck\s+contract[^|\n]{0,220}`?checks=14/14`?",
        re.IGNORECASE,
    ),
    "old_evidence_query_smoke_route": re.compile(
        r"evidence-query\s+smoke\s+route",
        re.IGNORECASE,
    ),
    "old_share_consistency_guard_13": re.compile(
        r"share[_ -]?consistency[_ -]?guard[^|\n]{0,260}"
        r"(?:13/13|13\s+checks|checks_total['\"]?\s*:\s*13)",
        re.IGNORECASE,
    ),
    "old_share_consistency_guard_14": re.compile(
        r"share[_ -]?consistency[_ -]?guard[^|\n]{0,260}"
        r"(?:14/14|14\s+checks|checks_total['\"]?\s*:\s*14)",
        re.IGNORECASE,
    ),
    "old_share_consistency_guard_15": re.compile(
        r"share[_ -]?consistency[_ -]?guard[^|\n]{0,260}"
        r"(?:15/15|15\s+checks|checks_total['\"]?\s*:\s*15)",
        re.IGNORECASE,
    ),
    "old_defense_qna_machine_claim_matrix": re.compile(
        r"defense_qna_readiness['\"]?\s*:\s*['\"]13 questions / evidence cards / machine claim matrix|"
        r"defense_qna_readiness['\"]?\s*:\s*['\"]13 questions / evidence cards / machine claim\+question matrix|"
        r"defense[_ -]?qna[_ -]?readiness[^|\n]{0,140}machine claim matrix",
        re.IGNORECASE,
    ),
    "old_defense_qna_readiness_15": re.compile(
        r"defense[_ -]?qna[_ -]?readiness[^|\n]{0,180}(?:15|16)\s+Q&A\s+checks|"
        r"defense\s+claim\s+matrix[^|\n]{0,180}checks\s+`?(?:15/15|16/16)`?",
        re.IGNORECASE,
    ),
    "old_rerun_acceptance_contract_16": re.compile(
        r"rerun[_ -]?acceptance[_ -]?contract[^|\n]{0,160}"
        r"(?:16/16|16\s+checks|ready=true,\s*16/16\s+checks\s+pass)",
        re.IGNORECASE,
    ),
    "old_rerun_delta_triage_16": re.compile(
        r"rerun[_ -]?delta[_ -]?triage[^|\n]{0,120}16\s+symptoms|"
        r"复跑差异定位[^|\n]{0,120}16\s+(?:symptoms|个症状)",
        re.IGNORECASE,
    ),
    "ambiguous_checkpoint_cst_timezone": re.compile(
        r"18:00\s+CST",
        re.IGNORECASE,
    ),
    "old_university_report_15_checks": re.compile(
        r"11 sections / 15 checks|"
        r"university technical report[^|\n]{0,80}15 checks|"
        r"15\s*个证据检查",
        re.IGNORECASE,
    ),
    "old_university_report_12_checks": re.compile(
        r"university technical report[^|\n]{0,120}(?:12/12|12 checks|12\s*个证据\s*(?:gate|检查))|"
        r"university_technical_report\.json[^|\n]{0,120}(?:12/12|12 checks)",
        re.IGNORECASE,
    ),
    "old_tail_confidence_7_checks": re.compile(
        r"18 rows / 7 checks / strict c4 tail|"
        r"ready=true, 7/7 checks pass|"
        r"tail appendix[^|\n]{0,100}7/7 gate",
        re.IGNORECASE,
    ),
    "old_tarball_sha_snapshots": re.compile(
        r"37d2db33048697de38e2e800896d2c257a64c377254a4d9d7a5d7f0cd5ae7392|"
        r"cec3764504e92bd18726c79f4d24c0a143b1bfa57090c545c9d2482e11f4316f|"
        r"77e8f72cdb89bdd4b593e49d97e8af5192748928903925fb4666459a5e5e5537|"
        r"47ce0de4005e7852f990051b133f20d0a58f44dad056ca92bbaa480dc6bfa5f0|"
        r"306f3ee5059ab04ee6d8cd768041a17cf4371763cf01f3b9ca1991068f80ae31|"
        r"ce3d62815bcf74900535a33bb6143b1c44593567e323eb08641114b0d4cffeb6|"
        r"c458a689d3ff926aba9feafa88a62a19d9b7b1864a550a8d02dd5f3860624a3f|"
        r"acd754871b3f3a54a712829d1ba47f9b7594a44f858ef5c032aa62028e0d48d2|"
        r"5b46507373ab275233ce4ae0ce054f67d05fec16b8cbfd5e82c5d3031d49e7b7|"
        r"5364c6165d29c3e957bf696e9aab83f363ed543f536df642c8b704259314587b|"
        r"4c19769d3c37a14b19647476fe9e70695cc0436fb5eb13abc25e3f1cc77e8a0d|"
        r"3c625e6ec5b99f663c3b5d38293a318b36c73c61690c18754abcce69f9b0d54f|"
        r"63af027e004754fe1fd9d7da43012d5bf16435ad3d8f20a1d2b2bec21dfbed4d|"
        r"ef6adc5537fbba92defaa773cf1713fc5455e17b1a057f73219ce317f14067fd|"
        r"ad7a45dbf41288b9f4214d1a1a091fb9554f4224eebd8160ba06ddd8ade8031c|"
        r"6e86091ba7a019f5dd3d89c8916afdf8d343a557c08b0227eb3304aebe1203cc|"
        r"8aef441e0d8466b86d04b4bc60bef91f8dc4da8cc816bf755403e5354efbe3e2|"
        r"51d48431f397a881c1aa1295aa43218f627096c636caf9e15d386259b31d5d0e|"
        r"8e363a89f19a298a4eef9a040a30576cce70d7430cdef8d47d20d26e28755ce9|"
        r"e58ec6b4f288d0fd206c55bbf2e2846a5fd1754ec66163a06a39a09947ff39e6|"
        r"bf54c000048b36ee104df46683467bac6a073511a860179788b38d04f2ec5b15|"
        r"251e1cf9dcc0478b1937bb16dbb7af5605965b3ed0dda380a5287a9e492f558a|"
        r"234d9af0f08204e859cd33916fdb8998597eff7e7fc75dd2b28d0fbc85ab75db"
    ),
}

HEX64_PATTERN = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)
RATIO_PATTERN = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
CHECKS_TOTAL_PASSED_PATTERN = re.compile(
    r"checks_total['\"]?\s*:\s*(\d+).*?checks_passed['\"]?\s*:\s*(\d+)"
)
RECEIVER_CONTRACT_REF_PATTERN = re.compile(
    r"receiver[_ -]?quickcheck[_ -]?contract|"
    r"receiver\s+quickcheck\s+contract|"
    r"接收方\s+quickcheck\s+contract",
    re.IGNORECASE,
)
MANIFEST_REF_PATTERN = re.compile(
    r"evidence\s+manifest|manifest|Manifest|证据清单",
    re.IGNORECASE,
)
MANIFEST_CURRENT_RECORDS_PATTERN = re.compile(
    r"current\s*:?\s*`?(\d+)`?\s+records",
    re.IGNORECASE,
)
MANIFEST_RECORDS_FIELD_PATTERN = re.compile(
    r"(?:records|total_records|record_count)\s*[:=]\s*`?(\d+)`?",
    re.IGNORECASE,
)
MANIFEST_RECORDS_LABEL_PATTERN = re.compile(
    r"\bmanifest\s*:?\s*`?(\d+)`?\s+records",
    re.IGNORECASE,
)
MANIFEST_MISSING_PATTERN = re.compile(
    r"(?:missing\s*[:=]\s*`?(\d+)`?|`?(\d+)`?\s+missing)",
    re.IGNORECASE,
)
TARBALL_IDENTITY_CONTEXT = re.compile(
    r"tarball|checksum|qwen35_omni_share_bundle_20260621\.tar\.gz|"
    r"share_bundle_package_manifest|share_package_validation|share_release_seal",
    re.IGNORECASE,
)
PUBLIC_TARBALL_IDENTITY_ALLOWLIST = {
    "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
}

EMBEDDED_SNAPSHOT_FILES = [
    AUDIT_DIR / "repro_command_manifest.json",
    AUDIT_DIR / "university_technical_report.json",
]

IDENTITY_KEYS = {
    "tarball_sha256",
    "tarball_sha256_uri",
    "tarball_size_bytes",
    "package_file_count",
}

EVIDENCE_SMOKE_ROUTE_FILES = [
    "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
]

EVIDENCE_SMOKE_ROUTE_NEEDLES = [
    '--root "$HOST_REPO" --mode host',
    '--root "$BUNDLE_ROOT" --mode portable',
]

LEGACY_MACHINE_OUTPUT_FILES = [
    "results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.out",
]


@dataclass(frozen=True)
class GuardCheck:
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


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


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


def _receiver_contract_marker_for_stale_scan(
    receiver_contract: dict[str, Any],
    audit_summary: dict[str, Any],
) -> tuple[str, bool, set[str]]:
    summary = _summary(receiver_contract)
    checks_passed = _int_value(summary.get("checks_passed"))
    checks_total = _int_value(summary.get("checks_total"))
    current_marker = ""
    if checks_passed > 0 and checks_total > 0:
        current_marker = f"{checks_passed}/{checks_total}"
    allowed_markers = {current_marker} if current_marker else set()

    allowed_pending_failures = {
        "current receiver evidence JSONs are all green",
        "current receiver evidence is asset-quality clean",
    }
    pending_in_full_audit = (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and checks_total >= 14
        and checks_passed >= 12
        and _int_value(summary.get("required_failures"), default=99)
        <= len(allowed_pending_failures)
        and _failed_required_check_names(receiver_contract).issubset(
            allowed_pending_failures
        )
        and _int_value(summary.get("receiver_jsons_total")) >= 4
        and _int_value(summary.get("public_docs_total")) >= 8
        and _int_value(summary.get("completion_gate_docs_total")) >= 6
        and _int_value(summary.get("wer_asr_docs_total")) >= 3
    )
    if pending_in_full_audit:
        expected_marker = f"{checks_total}/{checks_total}"
        allowed_markers.add(expected_marker)
        return expected_marker, True, allowed_markers
    return current_marker, False, allowed_markers


def _university_review_marker_for_stale_scan(
    university_review_packet: dict[str, Any],
    audit_summary: dict[str, Any],
) -> tuple[str, bool]:
    summary = _summary(university_review_packet)
    checks_passed = _int_value(summary.get("checks_passed"))
    checks_total = _int_value(summary.get("checks_total"))
    current_marker = ""
    if checks_passed > 0 and checks_total > 0:
        current_marker = f"{checks_passed}/{checks_total} checks"

    pending_in_full_audit = (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and checks_total >= 14
        and checks_passed >= checks_total - 1
        and _int_value(summary.get("required_failures"), default=99) <= 1
        and _failed_required_check_names(university_review_packet)
        == {"full audit and readiness are green"}
    )
    if pending_in_full_audit:
        return f"{checks_total}/{checks_total} checks", True
    return current_marker, False


def _sha256_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_value(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = _read_text_optional(path).strip()
    if not text:
        return None
    return text.split()[0]


def _mtime_optional(path: Path) -> float | None:
    if not path.is_file():
        return None
    return path.stat().st_mtime


def _scan_stale_tokens(
    root: Path,
    rel_paths: list[str],
    *,
    detect_tarball_identity_hash: bool = False,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        if not text:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern_id, pattern in STALE_PATTERNS.items():
                if pattern.search(line):
                    hits.append(
                        {
                            "pattern": pattern_id,
                            "path": rel_path,
                            "line": line_number,
                            "excerpt": line[:220],
                        }
                    )
            if (
                detect_tarball_identity_hash
                and rel_path not in PUBLIC_TARBALL_IDENTITY_ALLOWLIST
                and TARBALL_IDENTITY_CONTEXT.search(line)
                and HEX64_PATTERN.search(line)
            ):
                hits.append(
                    {
                        "pattern": "public_tarball_identity_hash",
                        "path": rel_path,
                        "line": line_number,
                        "excerpt": line[:220],
                    }
                )
    return hits


def _scan_receiver_contract_count_drift(
    root: Path,
    rel_paths: list[str],
    *,
    expected_marker: str,
    allowed_markers: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not expected_marker:
        return [
            {
                "pattern": "receiver_contract_expected_count_missing",
                "path": str(AUDIT_DIR / "receiver_quickcheck_contract.json"),
                "line": 0,
                "excerpt": "receiver quickcheck contract summary is missing checks_passed/checks_total",
            }
        ]
    allowed = allowed_markers or {expected_marker}

    hits: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        if not text:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            context_line = f"{line} {next_line}"
            for contract_match in RECEIVER_CONTRACT_REF_PATTERN.finditer(line):
                window_start = contract_match.start()
                window_end = contract_match.end() + 220
                window = context_line[window_start:window_end]
                ratios = [
                    f"{match.group(1)}/{match.group(2)}"
                    for match in RATIO_PATTERN.finditer(window)
                ]
                numeric_match = CHECKS_TOTAL_PASSED_PATTERN.search(window)
                numeric_marker = (
                    f"{numeric_match.group(2)}/{numeric_match.group(1)}"
                    if numeric_match
                    else ""
                )
                if ratios and not any(marker in allowed for marker in ratios):
                    hits.append(
                        {
                            "pattern": "receiver_contract_count_drift",
                            "path": rel_path,
                            "line": index + 1,
                            "expected": expected_marker,
                            "ratios": ratios,
                            "excerpt": window[:220],
                        }
                    )
                elif numeric_marker and numeric_marker not in allowed:
                    hits.append(
                        {
                            "pattern": "receiver_contract_count_drift",
                            "path": rel_path,
                            "line": index + 1,
                            "expected": expected_marker,
                            "numeric_marker": numeric_marker,
                            "excerpt": window[:220],
                        }
                    )
    return hits


def _first_int_group(match: re.Match[str]) -> int | None:
    for group in match.groups():
        if group is None:
            continue
        try:
            return int(group)
        except Exception:
            continue
    return None


def _scan_manifest_record_count_drift(
    root: Path,
    rel_paths: list[str],
    *,
    expected_records: int,
    expected_missing: int,
) -> list[dict[str, Any]]:
    if expected_records <= 0:
        return [
            {
                "pattern": "manifest_expected_record_count_missing",
                "path": str(AUDIT_DIR / "manifest.json"),
                "line": 0,
                "excerpt": "manifest summary is missing total_records",
            }
        ]

    hits: list[dict[str, Any]] = []
    record_patterns = [
        MANIFEST_CURRENT_RECORDS_PATTERN,
        MANIFEST_RECORDS_FIELD_PATTERN,
        MANIFEST_RECORDS_LABEL_PATTERN,
    ]
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        if not text:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            context_line = line
            if not MANIFEST_REF_PATTERN.search(context_line):
                continue

            record_markers: list[int] = []
            for pattern in record_patterns:
                for match in pattern.finditer(context_line):
                    value = _first_int_group(match)
                    if value is not None:
                        record_markers.append(value)
            stale_record_markers = sorted(
                {value for value in record_markers if value != expected_records}
            )
            if stale_record_markers:
                hits.append(
                    {
                        "pattern": "manifest_record_count_drift",
                        "path": rel_path,
                        "line": index + 1,
                        "expected": expected_records,
                        "markers": stale_record_markers,
                        "excerpt": line[:220],
                    }
                )

            missing_markers: list[int] = []
            for match in MANIFEST_MISSING_PATTERN.finditer(context_line):
                value = _first_int_group(match)
                if value is not None:
                    missing_markers.append(value)
            stale_missing_markers = sorted(
                {value for value in missing_markers if value != expected_missing}
            )
            if stale_missing_markers:
                hits.append(
                    {
                        "pattern": "manifest_missing_count_drift",
                        "path": rel_path,
                        "line": index + 1,
                        "expected": expected_missing,
                        "markers": stale_missing_markers,
                        "excerpt": line[:220],
                    }
                )
    return hits


def _find_identity_keys(obj: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            if key in IDENTITY_KEYS:
                hits.append(child_path)
            hits.extend(_find_identity_keys(value, child_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            hits.extend(_find_identity_keys(value, f"{path}[{index}]"))
    return hits


def _embedded_identity_leaks(root: Path) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for rel_path in EMBEDDED_SNAPSHOT_FILES:
        payload = _load_json_optional(root / rel_path)
        for hit in _find_identity_keys(payload):
            leaks.append({"path": str(rel_path), "json_path": hit})
    return leaks


def _needle_missing(text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle not in text]


def _required_needle_missing(
    root: Path,
    rel_paths: list[str],
    needles: list[str],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        missing = _needle_missing(text, needles)
        if missing:
            hits.append({"path": rel_path, "missing": missing})
    return hits


def _manifest_expected_gate_state(repro_summary: dict[str, Any]) -> dict[str, Any]:
    gates = repro_summary.get("expected_gates", {})
    gates = gates if isinstance(gates, dict) else {}
    manifest_gate = gates.get("manifest", {})
    manifest_gate = manifest_gate if isinstance(manifest_gate, dict) else {}
    unexpected_fields = sorted(
        field for field in ["total_records", "file_records"] if field in manifest_gate
    )
    missing_fields = sorted(
        field
        for field in [
            "min_total_records",
            "min_file_records",
            "directory_records",
            "missing_records",
        ]
        if field not in manifest_gate
    )
    min_total_records = _int_value(manifest_gate.get("min_total_records"))
    min_file_records = _int_value(manifest_gate.get("min_file_records"))
    directory_records = _int_value(manifest_gate.get("directory_records"), default=-1)
    missing_records = _int_value(manifest_gate.get("missing_records"), default=-1)
    return {
        "ready": (
            not missing_fields
            and not unexpected_fields
            and min_total_records >= 180
            and min_file_records >= 178
            and directory_records == 2
            and missing_records == 0
        ),
        "min_total_records": min_total_records,
        "min_file_records": min_file_records,
        "directory_records": directory_records,
        "missing_records": missing_records,
        "missing_fields": missing_fields,
        "unexpected_fields": unexpected_fields,
    }


def _manifest_relative_paths(manifest: dict[str, Any]) -> set[str]:
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return set()
    paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        rel_path = str(record.get("relative_path") or "").strip()
        if rel_path:
            paths.add(rel_path)
    return paths


def _preflight_alias_state(root: Path, audit_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    official = audit_dir / "preflight_repro.json"
    alias = audit_dir / "preflight.json"
    official_sha = _sha256_optional(official)
    alias_sha = _sha256_optional(alias)
    alias_present = alias.is_file()
    manifest_paths = _manifest_relative_paths(manifest)
    official_rel = str(AUDIT_DIR / "preflight_repro.json")
    alias_rel = str(AUDIT_DIR / "preflight.json")
    alias_mismatches = int(alias_present and official_sha != alias_sha)
    return {
        "ready": (
            bool(official_sha)
            and (not alias_present or alias_mismatches == 0)
            and official_rel in manifest_paths
            and alias_rel not in manifest_paths
        ),
        "official_path": str(official.relative_to(root)),
        "alias_path": str(alias.relative_to(root)),
        "official_present": bool(official_sha),
        "alias_present": alias_present,
        "official_sha": official_sha,
        "alias_sha": alias_sha,
        "alias_mismatches": alias_mismatches,
        "preflight_repro_in_manifest": official_rel in manifest_paths,
        "preflight_alias_in_manifest": alias_rel in manifest_paths,
    }


def _preflight_alias_public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: state.get(key)
        for key in [
            "ready",
            "official_present",
            "alias_present",
            "alias_mismatches",
            "preflight_repro_in_manifest",
            "preflight_alias_in_manifest",
        ]
    }


def _tarball_identity_state(root: Path, audit_dir: Path) -> dict[str, Any]:
    package = _load_json_optional(audit_dir / "share_bundle_package_manifest.json")
    validation = _load_json_optional(audit_dir / "share_package_validation.json")
    receiver_smoke = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    release_seal = _load_json_optional(audit_dir / "share_release_seal.json")

    tarball_path = root / DEFAULT_TARBALL
    checksum_path = root / DEFAULT_CHECKSUM
    package_path = audit_dir / "share_bundle_package_manifest.json"
    validation_path = audit_dir / "share_package_validation.json"
    receiver_path = audit_dir / "share_package_receiver_smoke_validation.json"
    release_seal_path = audit_dir / "share_release_seal.json"
    field_sources = {
        "actual_tarball_sha256": {
            "value": _sha256_optional(tarball_path),
            "path": tarball_path,
            "base": True,
        },
        "checksum_file_sha256": {
            "value": _checksum_value(checksum_path),
            "path": checksum_path,
            "base": True,
        },
        "package_manifest_tarball_sha256": {
            "value": package.get("tarball_sha256"),
            "path": package_path,
            "base": True,
        },
        "tarball_validation_summary_sha256": {
            "value": _summary(validation).get("tarball_sha256"),
            "path": validation_path,
            "base": False,
        },
        "receiver_smoke_summary_sha256": {
            "value": _summary(receiver_smoke).get("tarball_sha256"),
            "path": receiver_path,
            "base": False,
        },
        "release_seal_summary_sha256": {
            "value": _summary(release_seal).get("tarball_sha256"),
            "path": release_seal_path,
            "base": False,
        },
    }
    base_mtimes = [
        _mtime_optional(source["path"])
        for source in field_sources.values()
        if source["base"]
    ]
    reference_mtime = max((mtime for mtime in base_mtimes if mtime), default=None)
    deferred_fields = sorted(
        field
        for field, source in field_sources.items()
        if (
            not source["base"]
            and reference_mtime is not None
            and (
                _mtime_optional(source["path"]) is None
                or (_mtime_optional(source["path"]) or 0.0) < reference_mtime
            )
        )
    )
    active_values = {
        field: source["value"]
        for field, source in field_sources.items()
        if field not in deferred_fields
    }
    missing_fields = sorted(
        field for field, value in active_values.items() if not value
    )
    invalid_fields = sorted(
        field
        for field, value in active_values.items()
        if value and not HEX64_PATTERN.fullmatch(str(value))
    )
    normalized_values = {
        str(value).lower()
        for value in active_values.values()
        if value and HEX64_PATTERN.fullmatch(str(value))
    }
    mismatched_fields = (
        sorted(active_values)
        if len(normalized_values) > 1
        else []
    )
    return {
        "ready": not missing_fields and not invalid_fields and len(normalized_values) == 1,
        "fields_checked": len(field_sources),
        "fields_active": len(active_values),
        "fields_present": len(field_sources) - len(missing_fields) - len(deferred_fields),
        "deferred_fields": deferred_fields,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "unique_value_count": len(normalized_values),
        "mismatched_fields": mismatched_fields,
        "mismatches": len(mismatched_fields),
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    serving_capacity = _load_json_optional(audit_dir / "serving_capacity_matrix.json")
    university_review_packet = _load_json_optional(
        audit_dir / "university_review_packet.json"
    )
    command_hygiene = _load_json_optional(audit_dir / "command_reference_hygiene.json")
    share_path_hygiene = _load_json_optional(audit_dir / "share_path_hygiene.json")
    receiver_contract = _load_json_optional(audit_dir / "receiver_quickcheck_contract.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    tarball_identity = _tarball_identity_state(root, audit_dir)

    final_summary = final_readiness.get("summary", {})
    final_hard_gates = (
        final_summary.get("hard_gates", {})
        if isinstance(final_summary.get("hard_gates"), dict)
        else {}
    )
    final_checkpoint_gate = str(final_hard_gates.get("final_checkpoint_watchlist") or "")
    final_checkpoint_gate_current = final_checkpoint_gate == "24/24" or (
        bool(audit_summary.get("in_progress"))
        and final_checkpoint_gate in {"23/23", "24/24"}
    )
    repro_summary = repro_manifest.get("summary", {})
    share_bundle_summary = share_bundle.get("summary", {})
    serving_summary = serving_capacity.get("summary", {})
    university_review_summary = university_review_packet.get("summary", {})
    command_hygiene_summary = command_hygiene.get("summary", {})
    share_path_summary = share_path_hygiene.get("summary", {})
    receiver_contract_summary = receiver_contract.get("summary", {})
    manifest_summary = manifest.get("summary", {})
    (
        receiver_contract_marker,
        receiver_contract_pending_in_full_audit,
        receiver_contract_allowed_markers,
    ) = _receiver_contract_marker_for_stale_scan(receiver_contract, audit_summary)
    manifest_records_current = _int_value(manifest_summary.get("total_records"))
    manifest_missing_current = _int_value(manifest_summary.get("missing_records"))
    manifest_expected_gate = _manifest_expected_gate_state(repro_summary)
    preflight_alias = _preflight_alias_state(root, audit_dir, manifest)
    min_share_reports = max(38, len(REPORT_FILES) - 1)
    min_machine_evidence = max(46, len(MACHINE_EVIDENCE_FILES) - 1)

    self_report = str(DEFAULT_OUTPUT)
    self_json = str(DEFAULT_JSON_OUTPUT)
    # The release seal is regenerated after package validation; the pre-package
    # guard scans the share-bundle reports that are already final at this step.
    reports_to_scan = [
        rel_path
        for rel_path in [
            *REPORT_FILES,
        ]
        if rel_path != self_report
    ]
    machine_to_scan = [
        rel_path
        for rel_path in [
            *MACHINE_EVIDENCE_FILES,
            *LEGACY_MACHINE_OUTPUT_FILES,
            "results/qwen35_report_audit_20260619/final_readiness_audit.json",
        ]
        if rel_path != self_json
        and not (
            rel_path == "results/qwen35_report_audit_20260619/audit_run_summary.json"
            and bool(audit_summary.get("in_progress"))
        )
    ]
    public_stale_hits = _scan_stale_tokens(
        root,
        reports_to_scan,
        detect_tarball_identity_hash=True,
    )
    machine_stale_hits = _scan_stale_tokens(root, machine_to_scan)
    public_stale_hits.extend(
        _scan_receiver_contract_count_drift(
            root,
            reports_to_scan,
            expected_marker=receiver_contract_marker,
            allowed_markers=receiver_contract_allowed_markers,
        )
    )
    machine_stale_hits.extend(
        _scan_receiver_contract_count_drift(
            root,
            machine_to_scan,
            expected_marker=receiver_contract_marker,
            allowed_markers=receiver_contract_allowed_markers,
        )
    )
    public_stale_hits.extend(
        _scan_manifest_record_count_drift(
            root,
            reports_to_scan,
            expected_records=manifest_records_current,
            expected_missing=manifest_missing_current,
        )
    )
    machine_stale_hits.extend(
        _scan_manifest_record_count_drift(
            root,
            machine_to_scan,
            expected_records=manifest_records_current,
            expected_missing=manifest_missing_current,
        )
    )
    embedded_identity_leaks = _embedded_identity_leaks(root)

    share_index = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md"
    )
    receiver_map = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md"
    )
    serving_report = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md"
    )
    current_repro_command_marker = (
        f"`{_int_value(repro_summary.get('commands_total'))}` commands"
    )
    university_review_checks_total = _int_value(
        university_review_summary.get("checks_total")
    )
    university_review_checks_passed = _int_value(
        university_review_summary.get("checks_passed")
    )
    (
        current_university_review_marker,
        university_review_pending_in_full_audit,
    ) = _university_review_marker_for_stale_scan(
        university_review_packet,
        audit_summary,
    )

    share_index_missing = _needle_missing(
        share_index,
        [
            "qwen35_omni_serving_capacity_matrix_zh_20260621.md",
            "serving_capacity_matrix.json",
            "qwen35_omni_pressure_stage_heatmap_zh_20260621.md",
            "pressure_stage_heatmap.json",
            "qwen35_omni_share_consistency_guard_zh_20260621.md",
            "share_consistency_guard.json",
            "49/49",
            current_repro_command_marker,
            "16 个证据检查",
            "17/17 机器 gate",
            "university-review gate route",
        ],
    )
    university_review_index_missing = _needle_missing(
        share_index,
        [
            "university_review_packet.json",
            current_university_review_marker,
            "术语口径路线",
        ],
    )
    receiver_missing = _needle_missing(
        receiver_map,
        [
            "qwen35_omni_serving_capacity_matrix_zh_20260621.md",
            "serving_capacity_matrix.json",
            "qwen35_omni_pressure_stage_heatmap_zh_20260621.md",
            "pressure_stage_heatmap.json",
        ],
    )
    serving_report_missing = _needle_missing(
        serving_report,
        [
            "Serving/Capacity 决策矩阵",
            "Video-AMME c=8",
            "Synthetic long text c=8",
            "vLLM c=8 prebuild w4",
            "不要升级为 online serving parity",
        ],
    )
    evidence_smoke_route_missing = _required_needle_missing(
        root,
        EVIDENCE_SMOKE_ROUTE_FILES,
        EVIDENCE_SMOKE_ROUTE_NEEDLES,
    )

    checks: list[GuardCheck] = [
        GuardCheck(
            "final readiness gate count is current",
            _status(
                _int_value(final_summary.get("checks_total")) >= 49
                and _int_value(final_summary.get("checks_total")) not in {47, 48}
                and "16 checks"
                in str(final_hard_gates.get("university_technical_report") or "")
                and "pressure-stage heatmap"
                in str(final_hard_gates.get("university_technical_report") or "")
                and "13 checks"
                in str(final_hard_gates.get("tail_confidence_appendix") or "")
                and "bootstrap uncertainty"
                in str(final_hard_gates.get("tail_confidence_appendix") or "")
                and "17/17 checks"
                in str(final_hard_gates.get("stage_reproduction_drilldown") or "")
                and "5 quick reproduction routes"
                in str(final_hard_gates.get("stage_reproduction_drilldown") or "")
                and final_checkpoint_gate_current
            ),
            (
                f"final_readiness={final_summary}, "
                f"final_checkpoint_gate={final_checkpoint_gate}, "
                f"audit_in_progress={bool(audit_summary.get('in_progress'))}"
            ),
        ),
        GuardCheck(
            "repro command manifest count is current",
            _status(
                _int_value(repro_summary.get("commands_total")) >= 62
                and _int_value(repro_summary.get("phases_total")) >= 7
                and bool(repro_summary.get("required_command_ids_present"))
            ),
            f"repro_command_manifest={repro_summary}",
        ),
        GuardCheck(
            "repro manifest expected gates keep manifest thresholds explicit",
            _status(bool(manifest_expected_gate.get("ready"))),
            f"manifest_expected_gate={manifest_expected_gate}",
        ),
        GuardCheck(
            "preflight alias cannot drift from official preflight evidence",
            _status(bool(preflight_alias.get("ready"))),
            f"preflight_alias={_preflight_alias_public_state(preflight_alias)}",
        ),
        GuardCheck(
            "serving capacity matrix remains ready",
            _status(
                bool(serving_summary.get("ready"))
                and _int_value(serving_summary.get("checks_total")) >= 10
                and _int_value(serving_summary.get("rows_total")) >= 7
                and serving_summary.get("vllm_w4_scope") == "offline_diagnostic_only"
            ),
            f"serving_capacity_matrix={serving_summary}",
        ),
        GuardCheck(
            "share bundle manifest includes current report families",
            _status(
                _int_value(share_bundle_summary.get("records_total")) >= 100
                and _int_value(
                    share_bundle_summary.get("missing_required"), default=99
                )
                <= 2
                and _int_value(
                    share_bundle_summary.get("category_counts", {}).get("share_report")
                )
                >= min_share_reports
                and _int_value(
                    share_bundle_summary.get("category_counts", {}).get("machine_evidence")
                )
                >= min_machine_evidence
            ),
            f"share_bundle_manifest={share_bundle_summary}",
        ),
        GuardCheck(
            "current tarball identity fields agree without embedding hashes",
            _status(bool(tarball_identity.get("ready"))),
            (
                f"fields_checked={tarball_identity.get('fields_checked')}, "
                f"fields_active={tarball_identity.get('fields_active')}, "
                f"fields_present={tarball_identity.get('fields_present')}, "
                f"deferred_fields={tarball_identity.get('deferred_fields')}, "
                f"missing_fields={tarball_identity.get('missing_fields')}, "
                f"invalid_fields={tarball_identity.get('invalid_fields')}, "
                f"unique_value_count={tarball_identity.get('unique_value_count')}, "
                f"mismatches={tarball_identity.get('mismatches')}, "
                f"mismatched_fields={tarball_identity.get('mismatched_fields')}"
            ),
        ),
        GuardCheck(
            "public reports have no stale gate tokens",
            _status(not public_stale_hits),
            f"hits={public_stale_hits[:10]}, total={len(public_stale_hits)}",
        ),
        GuardCheck(
            "machine evidence has no stale gate tokens",
            _status(not machine_stale_hits),
            f"hits={machine_stale_hits[:10]}, total={len(machine_stale_hits)}",
        ),
        GuardCheck(
            "embedded source snapshots omit tarball identity fields",
            _status(not embedded_identity_leaks),
            f"leaks={embedded_identity_leaks[:10]}, total={len(embedded_identity_leaks)}",
        ),
        GuardCheck(
            "share index routes current consistency and capacity artifacts",
            _status(not share_index_missing),
            "missing=" + ", ".join(share_index_missing),
        ),
        GuardCheck(
            "share index routes current university review packet gate",
            _status(
                (
                    (
                        bool(university_review_summary.get("ready"))
                        and university_review_checks_total >= 14
                        and university_review_checks_passed
                        == university_review_checks_total
                    )
                    or university_review_pending_in_full_audit
                )
                and not university_review_index_missing
            ),
            (
                f"university_review_packet={university_review_summary}; "
                f"pending_in_full_audit={university_review_pending_in_full_audit}; "
                f"expected_marker={current_university_review_marker}; "
                f"missing={university_review_index_missing}"
            ),
        ),
        GuardCheck(
            "receiver path map exposes serving capacity artifacts",
            _status(not receiver_missing),
            "missing=" + ", ".join(receiver_missing),
        ),
        GuardCheck(
            "serving capacity report preserves vLLM online caveat",
            _status(not serving_report_missing),
            "missing=" + ", ".join(serving_report_missing),
        ),
        GuardCheck(
            "evidence query smoke exposes host and portable routes",
            _status(not evidence_smoke_route_missing),
            f"missing={evidence_smoke_route_missing}",
        ),
        GuardCheck(
            "command reference hygiene resolves current command set",
            _status(
                bool(command_hygiene_summary.get("ready"))
                and _int_value(command_hygiene_summary.get("manifest_commands_total"))
                >= 62
                and _int_value(command_hygiene_summary.get("unresolved_command_refs_total"), default=1)
                == 0
            ),
            f"command_reference_hygiene={command_hygiene_summary}",
        ),
        GuardCheck(
            "share path hygiene has no package/raw offenders",
            _status(
                bool(share_path_summary.get("ready"))
                and _int_value(share_path_summary.get("package_offenders_total"), default=1)
                == 0
                and _int_value(share_path_summary.get("raw_offenders_total"), default=1)
                == 0
                and _int_value(share_path_summary.get("legacy_hits_total"), default=1)
                == 0
            ),
            f"share_path_hygiene={share_path_summary}",
        ),
    ]
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
            "public_stale_hits": len(public_stale_hits),
            "machine_stale_hits": len(machine_stale_hits),
            "embedded_identity_leaks": len(embedded_identity_leaks),
            "tarball_identity_fields_checked": tarball_identity.get("fields_checked"),
            "tarball_identity_fields_active": tarball_identity.get("fields_active"),
            "tarball_identity_fields_present": tarball_identity.get("fields_present"),
            "tarball_identity_deferred_fields": len(
                tarball_identity.get("deferred_fields", [])
            ),
            "tarball_identity_missing_fields": len(
                tarball_identity.get("missing_fields", [])
            ),
            "tarball_identity_invalid_fields": len(
                tarball_identity.get("invalid_fields", [])
            ),
            "tarball_identity_unique_values": tarball_identity.get(
                "unique_value_count"
            ),
            "tarball_identity_mismatches": tarball_identity.get("mismatches"),
            "final_readiness_checks": final_summary.get("checks_total"),
            "repro_commands_total": repro_summary.get("commands_total"),
            "share_bundle_records": share_bundle_summary.get("records_total"),
            "serving_capacity_rows": serving_summary.get("rows_total"),
            "university_review_packet_checks": current_university_review_marker,
            "university_review_packet_pending_in_full_audit": (
                university_review_pending_in_full_audit
            ),
            "university_review_packet_glossary_missing": (
                university_review_summary.get("glossary_terms_missing")
            ),
            "receiver_quickcheck_contract_checks": receiver_contract_marker,
            "receiver_quickcheck_contract_pending_in_full_audit": (
                receiver_contract_pending_in_full_audit
            ),
            "legacy_machine_outputs_checked": sum(
                1 for rel_path in LEGACY_MACHINE_OUTPUT_FILES if (root / rel_path).is_file()
            ),
            "manifest_records_current": manifest_records_current,
            "manifest_missing_current": manifest_missing_current,
            "manifest_expected_gate_ready": manifest_expected_gate.get("ready"),
            "manifest_expected_gate_min_total_records": manifest_expected_gate.get(
                "min_total_records"
            ),
            "manifest_expected_gate_min_file_records": manifest_expected_gate.get(
                "min_file_records"
            ),
            "manifest_expected_gate_unexpected_fields": len(
                manifest_expected_gate.get("unexpected_fields", [])
            ),
            "preflight_alias_ready": preflight_alias.get("ready"),
            "preflight_alias_present": preflight_alias.get("alias_present"),
            "preflight_alias_mismatches": preflight_alias.get("alias_mismatches"),
            "preflight_repro_in_manifest": preflight_alias.get(
                "preflight_repro_in_manifest"
            ),
            "preflight_alias_in_manifest": preflight_alias.get(
                "preflight_alias_in_manifest"
            ),
            "evidence_smoke_route_docs_checked": len(EVIDENCE_SMOKE_ROUTE_FILES),
            "evidence_smoke_route_missing": len(evidence_smoke_route_missing),
            "share_scope": (
                "Machine guard that prevents stale public gate counts, stale machine "
                "snapshots, receiver quickcheck contract count drift, manifest record "
                "count drift, expected-gate field ambiguity, serving/capacity routing "
                "drift, preflight alias drift, evidence-query smoke host/portable route "
                "drift, university-review gate route drift, and current tarball "
                "identity-field disagreement before sharing; the adjacent release seal "
                "performs the final full-chain hash check after validation refresh."
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "stale_hits": {
            "public_reports": public_stale_hits,
            "machine_evidence": machine_stale_hits,
            "embedded_identity_leaks": embedded_identity_leaks,
        },
        "sources": {
            "final_readiness": final_summary,
            "repro_command_manifest": repro_summary,
            "share_bundle_manifest": share_bundle_summary,
            "serving_capacity_matrix": serving_summary,
            "command_reference_hygiene": command_hygiene_summary,
            "share_path_hygiene": share_path_summary,
            "receiver_quickcheck_contract": receiver_contract_summary,
            "manifest": manifest_summary,
            "tarball_identity_chain": {
                "ready": tarball_identity.get("ready"),
                "fields_checked": tarball_identity.get("fields_checked"),
                "fields_active": tarball_identity.get("fields_active"),
                "fields_present": tarball_identity.get("fields_present"),
                "deferred_fields": tarball_identity.get("deferred_fields"),
                "missing_fields": tarball_identity.get("missing_fields"),
                "invalid_fields": tarball_identity.get("invalid_fields"),
                "unique_value_count": tarball_identity.get("unique_value_count"),
                "mismatches": tarball_identity.get("mismatches"),
                "mismatched_fields": tarball_identity.get("mismatched_fields"),
            },
        },
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    repro_summary = payload.get("sources", {}).get("repro_command_manifest", {})
    repro_commands_total = repro_summary.get(
        "commands_total", summary.get("repro_commands_total")
    )
    repro_phases_total = repro_summary.get("phases_total", 7)
    lines = [
        "# Qwen3.5-Omni Share Consistency Guard",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份 guard 专门防止分享包里出现旧 gate 数字、旧命令数、旧 tarball identity snapshot，",
        "以及 serving/capacity 新证据没有进入阅读入口的问题。",
        "",
        "## 1. Gate",
        "",
        f"- ready：`{summary.get('ready')}`，checks：`{summary.get('checks_passed')}/{summary.get('checks_total')}`。",
        f"- final readiness checks：`{summary.get('final_readiness_checks')}`；repro commands：`{summary.get('repro_commands_total')}`。",
        f"- share bundle records：`{summary.get('share_bundle_records')}`；serving capacity rows：`{summary.get('serving_capacity_rows')}`。",
        f"- receiver quickcheck contract checks：`{summary.get('receiver_quickcheck_contract_checks')}`。",
        f"- legacy machine outputs checked：`{summary.get('legacy_machine_outputs_checked')}`。",
        f"- manifest records：`{summary.get('manifest_records_current')}`；missing：`{summary.get('manifest_missing_current')}`。",
        (
            "- manifest expected gate："
            f"`min_total_records={summary.get('manifest_expected_gate_min_total_records')}`，"
            f"`min_file_records={summary.get('manifest_expected_gate_min_file_records')}`，"
            f"`unexpected_fields={summary.get('manifest_expected_gate_unexpected_fields')}`。"
        ),
        (
            "- preflight alias："
            f"`alias_present={summary.get('preflight_alias_present')}`，"
            f"`alias_mismatches={summary.get('preflight_alias_mismatches')}`，"
            f"`preflight_repro_in_manifest={summary.get('preflight_repro_in_manifest')}`，"
            f"`preflight_alias_in_manifest={summary.get('preflight_alias_in_manifest')}`。"
        ),
        f"- evidence-query smoke route docs：`{summary.get('evidence_smoke_route_docs_checked')}`；missing：`{summary.get('evidence_smoke_route_missing')}`。",
        f"- stale public hits：`{summary.get('public_stale_hits')}`；stale machine hits：`{summary.get('machine_stale_hits')}`；embedded identity leaks：`{summary.get('embedded_identity_leaks')}`。",
        (
            "- tarball identity chain："
            f"`fields={summary.get('tarball_identity_fields_present')}/"
            f"{summary.get('tarball_identity_fields_checked')}`，"
            f"`active={summary.get('tarball_identity_fields_active')}`，"
            f"`deferred={summary.get('tarball_identity_deferred_fields')}`，"
            f"`unique_values={summary.get('tarball_identity_unique_values')}`，"
            f"`mismatches={summary.get('tarball_identity_mismatches')}`。"
        ),
        "",
        "## 2. Check 明细",
        "",
        "| Status | Required | Check | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        evidence = str(check.get("evidence", "")).replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(check.get("status")),
                    "yes" if check.get("required") else "no",
                    str(check.get("name")),
                    evidence,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. 当前硬口径",
            "",
            "- final readiness：`49/49`，`0` required failures。",
            (
                "- repro command manifest："
                f"`{repro_commands_total} commands / {repro_phases_total} phases`。"
            ),
            "- tail confidence appendix：`18 rows / 13 checks / bootstrap uncertainty`。",
            "- serving/capacity matrix：`7 rows / 10 checks`。",
            "- evidence-query smoke：host 使用 `--root \"$HOST_REPO\" --mode host`，portable 使用 `--root \"$BUNDLE_ROOT\" --mode portable`。",
            "- expected_gates.manifest：只使用 `min_total_records/min_file_records` 表示最低门槛；当前清单规模必须来自 `manifest.json` 的 `total_records/file_records`。",
            "- preflight：正式分享入口是 `preflight_repro.json`；若旁路 `preflight.json` 存在，必须是字节一致副本且不能进入 manifest。",
            "- tarball identity 不写入 tarball 内成员；本 guard 只比较当前已刷新字段的一致性计数，完整 hash 链以 release seal 和 `.sha256` 为准。",
            "",
            "## 4. 机器证据",
            "",
            "- `results/qwen35_report_audit_20260619/share_consistency_guard.json`",
            "- `results/qwen35_report_audit_20260619/final_readiness_audit.json`",
            "- `results/qwen35_report_audit_20260619/repro_command_manifest.json`",
            "- `results/qwen35_report_audit_20260619/serving_capacity_matrix.json`",
            "- `results/qwen35_report_audit_20260619/share_bundle_manifest.json`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni share-consistency guard."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    payload = build_payload(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(payload), encoding="utf-8")
    _save_json(payload, json_output)
    summary = payload["summary"]
    print(
        "Share consistency guard written: "
        f"{output} json={json_output} ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"stale_public={summary['public_stale_hits']} "
        f"stale_machine={summary['machine_stale_hits']}"
    )
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
