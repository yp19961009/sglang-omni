# SPDX-License-Identifier: Apache-2.0
"""Build the final share-readiness audit for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "final_readiness_audit.json"


REPORT_FILES = [
    "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md",
    "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_unit_test_smoke_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
    "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md",
]

MACHINE_EVIDENCE_FILES = [
    "results/qwen35_report_audit_20260619/audit_run_summary.json",
    "results/qwen35_report_audit_20260619/environment_snapshot.json",
    "results/qwen35_report_audit_20260619/manifest.json",
    "results/qwen35_report_audit_20260619/claims_verification.json",
    "results/qwen35_report_audit_20260619/coverage_matrix.json",
    "results/qwen35_report_audit_20260619/headline_scorecard.json",
    "results/qwen35_report_audit_20260619/tail_confidence_appendix.json",
    "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json",
    "results/qwen35_report_audit_20260619/chart_source_consistency.json",
    "results/qwen35_report_audit_20260619/slide_asset_map.json",
    "results/qwen35_report_audit_20260619/acceptance_matrix.json",
    "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
    "results/qwen35_report_audit_20260619/university_review_packet.json",
    "results/qwen35_report_audit_20260619/university_technical_report.json",
    "results/qwen35_report_audit_20260619/serving_capacity_matrix.json",
    "results/qwen35_report_audit_20260619/share_consistency_guard.json",
    "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
    "results/qwen35_report_audit_20260619/confidence_ledger.json",
    "results/qwen35_report_audit_20260619/objective_completion_audit.json",
    "results/qwen35_report_audit_20260619/repro_command_manifest.json",
    "results/qwen35_report_audit_20260619/rerun_time_budget.json",
    "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
    "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
    "results/qwen35_report_audit_20260619/runtime_image_contract.json",
    "results/qwen35_report_audit_20260619/unit_test_smoke.json",
    "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
    "results/qwen35_report_audit_20260619/rerun_delta_triage.json",
    "results/qwen35_report_audit_20260619/defense_claim_matrix.json",
    "results/qwen35_report_audit_20260619/claim_metric_crosswalk.json",
    "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
    "results/qwen35_report_audit_20260619/optimization_candidate_ledger.json",
    "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
    "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
    "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
    "results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json",
    "results/qwen35_report_audit_20260619/pressure_stage_heatmap.json",
    "results/qwen35_report_audit_20260619/length_regime_coverage.json",
    "results/qwen35_report_audit_20260619/stage_latency_budget.json",
    "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
    "results/qwen35_report_audit_20260619/stage_causal_graph.json",
    "results/qwen35_report_audit_20260619/stage_drilldown_index.json",
    "results/qwen35_report_audit_20260619/metric_provenance_index.json",
    "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
    "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
    "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json",
    "results/qwen35_report_audit_20260619/share_path_hygiene.json",
    "results/qwen35_report_audit_20260619/command_reference_hygiene.json",
    "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
    "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
    "results/qwen35_report_audit_20260619/tables_summary.json",
    "results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst",
    "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json",
    "results/qwen35_report_audit_20260619/share_bundle_manifest.json",
]

SHARE_TOOL_FILES = [
    "benchmarks/eval/validate_qwen35_omni_share_package.py",
    "benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
    "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
    "benchmarks/eval/qwen35_omni_collaborator_return_check.py",
]


@dataclass(frozen=True)
class ReadinessCheck:
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
    checks: list[ReadinessCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(ReadinessCheck(name, _status(condition), evidence, required))


def _all_exist(root: Path, rel_paths: list[str]) -> tuple[bool, list[str]]:
    missing = [rel for rel in rel_paths if not (root / rel).exists()]
    return not missing, missing


def _all_have(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _section_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0]


def _named_sources_have(
    sources: dict[str, str], needles: list[str]
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for source_name, text in sources.items():
        for needle in needles:
            if needle not in text:
                missing.append(f"{source_name}:{needle}")
    return not missing, missing


def _manifest_relative_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        rel_path = str(record.get("relative_path") or "")
        if rel_path and record.get("exists") is not False:
            paths.add(rel_path)
        for item in record.get("files", []):
            if not isinstance(item, dict):
                continue
            item_rel_path = str(item.get("relative_path") or "")
            if item_rel_path and item.get("exists") is not False:
                paths.add(item_rel_path)
    return paths


def _stage_causal_manifest_backing(
    stage_causal_graph: str, manifest: dict[str, Any]
) -> tuple[bool, str]:
    raw_paths = sorted(
        set(
            re.findall(
                r"`(results/qwen35_(?!report_audit)[^`]+)`",
                stage_causal_graph,
            )
        )
    )
    manifest_paths = _manifest_relative_paths(manifest)
    missing = [path for path in raw_paths if path not in manifest_paths]
    ok = len(raw_paths) >= 10 and not missing
    return (
        ok,
        f"raw_paths={len(raw_paths)}, missing_from_manifest={missing[:10]}",
    )


def _share_index_structure(text: str) -> tuple[bool, str]:
    start = "## 1. 五分钟阅读顺序"
    end = "## 2. 当前可对外讲的结论"
    section_2_headings = text.count(end)
    numbers: list[int] = []
    if start in text and end in text:
        section = text.split(start, 1)[1].split(end, 1)[0]
        for line in section.splitlines():
            match = re.match(r"^(\d+)\. ", line)
            if match:
                numbers.append(int(match.group(1)))
    expected = list(range(1, max(numbers) + 1)) if numbers else []
    condition = (
        bool(numbers)
        and numbers == expected
        and numbers[-1] >= 29
        and section_2_headings == 1
    )
    evidence = (
        f"numbers={numbers}, expected_last={expected[-1] if expected else None}, "
        f"section_2_headings={section_2_headings}"
    )
    return condition, evidence


def _share_index_navigation_coverage(text: str) -> tuple[bool, str]:
    missing_reports: list[str] = []
    for rel_path in REPORT_FILES:
        if rel_path == "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md":
            continue
        name = Path(rel_path).name
        if name not in text and rel_path not in text:
            missing_reports.append(rel_path)

    missing_machine_evidence: list[str] = []
    for rel_path in MACHINE_EVIDENCE_FILES:
        name = Path(rel_path).name
        if name not in text and rel_path not in text:
            missing_machine_evidence.append(rel_path)

    condition = not missing_reports and not missing_machine_evidence
    evidence = (
        f"report_refs={len(REPORT_FILES) - 1 - len(missing_reports)}/"
        f"{len(REPORT_FILES) - 1}, "
        f"machine_refs={len(MACHINE_EVIDENCE_FILES) - len(missing_machine_evidence)}/"
        f"{len(MACHINE_EVIDENCE_FILES)}, "
        f"missing_reports={missing_reports}, "
        f"missing_machine_evidence={missing_machine_evidence}"
    )
    return condition, evidence


def _public_doc_hardcoded_hashes(root: Path, rel_paths: list[str]) -> list[str]:
    offenders: list[str] = []
    hash_pattern = re.compile(r"(?<!sha256:)\b[0-9a-f]{64}\b")
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in hash_pattern.finditer(line):
                short_hash = match.group(0)[:12]
                offenders.append(f"{rel_path}:{line_number}:{short_hash}...")
    return offenders


def _unescaped_pipe_count(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def _public_doc_markdown_table_offenders(
    root: Path, rel_paths: list[str]
) -> list[str]:
    offenders: list[str] = []
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        expected_pipes = 0
        table_start_line = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.startswith("|"):
                pipe_count = _unescaped_pipe_count(line)
                if not table_start_line:
                    table_start_line = line_number
                    expected_pipes = pipe_count
                elif pipe_count != expected_pipes:
                    offenders.append(
                        f"{rel_path}:{line_number}:expected {expected_pipes} "
                        f"pipes from table line {table_start_line}, got {pipe_count}"
                    )
            else:
                expected_pipes = 0
                table_start_line = 0
    return offenders


def _public_doc_malformed_text_tokens(root: Path, rel_paths: list[str]) -> list[str]:
    offenders: list[str] = []
    malformed_tokens = (
        "n/ams",
        "n/a%",
        "nanms",
        "diagnosis=None",
        "主报告 section",
        "对外解释时保持三条边界",
    )
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for token in malformed_tokens:
                if token in line:
                    offenders.append(f"{rel_path}:{line_number}:{token}")
    return offenders


def _public_doc_duplicate_heading_offenders(
    root: Path, rel_paths: list[str]
) -> list[str]:
    offenders: list[str] = []
    heading_pattern = re.compile(r"^#{1,6} .+")
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        seen: dict[str, int] = {}
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not heading_pattern.match(line):
                continue
            first_seen = seen.get(line)
            if first_seen is not None:
                offenders.append(
                    f"{rel_path}:{line_number}:duplicate heading from line {first_seen}: {line}"
                )
            else:
                seen[line] = line_number
    return offenders


def _pressure_matrix_na_legend_missing(root: Path) -> list[str]:
    rel_path = "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md"
    text = _read_text_optional(root / rel_path)
    required_terms = (
        "| n/a | 不适用或未计入该行结论",
        "低并发 queue estimate",
        "失败反例无稳定性能数字",
        "diagnostic 行未计算 WER/ASR",
        "不能把 n/a 当作 0",
    )
    missing = [term for term in required_terms if term not in text]
    return [f"{rel_path}:{term}" for term in missing]


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


def _public_doc_semantic_consistency_offenders(
    root: Path, rel_paths: list[str]
) -> list[str]:
    offenders: list[str] = []
    redline_heading = re.compile(
        r"^##\s+\d+\.\s+([一二三四五六七八九十])条红线\s*$",
        flags=re.MULTILINE,
    )
    next_section = re.compile(r"^##\s+\d+\.\s+", flags=re.MULTILINE)
    for rel_path in rel_paths:
        text = _read_text_optional(root / rel_path)
        for match in redline_heading.finditer(text):
            expected = _chinese_count(match.group(1))
            following = text[match.end() :]
            next_match = next_section.search(following)
            section = following[: next_match.start()] if next_match else following
            bullet_count = sum(
                1 for line in section.splitlines() if line.startswith("- ")
            )
            if expected is not None and bullet_count != expected:
                line_number = text.count("\n", 0, match.start()) + 1
                offenders.append(
                    f"{rel_path}:{line_number}:redline_count:"
                    f"heading={expected}, bullets={bullet_count}"
                )
    return offenders


def _defense_qna_readiness(
    qna_text: str,
    share_index_text: str,
    defense_claim_matrix: dict[str, Any],
) -> tuple[bool, str]:
    expected_questions = [
        "SGLang-Omni 是否至少和 vLLM 相当？",
        "vLLM baseline 是不是故意设弱了？",
        "为什么 headline 选 warmed c=4？",
        "单并发结果说明什么？",
        "为什么 c=8 是推荐高并发点？",
        "长短文本/语音输出覆盖了吗？",
        "每个 stage 的瓶颈在哪里？",
        "Stage 之间有没有卡住？",
        "为什么不直接提高 preprocessing 并发？",
        "vLLM c=8 为什么不能直接作为 online parity？",
        "WER 和语音一致性有没有退化？",
        "现在的结论有多大外推范围？",
        "如何现场复现或验收？",
    ]
    evidence_card_section = ""
    if "## 现场证据卡" in qna_text and "## 1. " in qna_text:
        evidence_card_section = qna_text.split("## 现场证据卡", 1)[1].split(
            "## 1. ", 1
        )[0]
    evidence_card_table_lines = [
        line
        for line in evidence_card_section.splitlines()
        if line.startswith("| ")
    ]
    malformed_evidence_card_rows = [
        line
        for line in evidence_card_table_lines
        if len(re.findall(r"(?<!\\)\|", line)) != 4
    ]
    claim_matrix_section = ""
    if "## 14. 主张-证据-复跑-裁决矩阵" in qna_text:
        claim_matrix_section = qna_text.split(
            "## 14. 主张-证据-复跑-裁决矩阵", 1
        )[1]
    claim_matrix_table_lines = [
        line for line in claim_matrix_section.splitlines() if line.startswith("| ")
    ]
    malformed_claim_matrix_rows = [
        line
        for line in claim_matrix_table_lines
        if len(re.findall(r"(?<!\\)\|", line)) != 6
    ]
    required_terms = [
        "短答：",
        "可说：",
        "不能说：",
        "证据入口：",
        "warmed c=4",
        "c=8",
        "long c=8",
        "stage",
        "code2wav",
        "preprocessing",
        "online parity",
        "run_qwen35_omni_report_audit",
        "qwen35_omni_vllm_optimization_lock_zh_20260621.md",
        "vllm_optimization_lock.json",
        "compile/CUDA graph",
        "prefix/chunked prefill",
        "shared-memory transfer",
        "encoder compile/batch",
        "prebuild w4",
        "现场证据卡",
        "JSON key",
        "只读 `jq`",
        "jq '.strict_c4_comparison'",
        "jq '.sglang_stress.throughput_peak'",
        "jq '.synthetic_long_c8'",
        "jq '.summary' results/qwen35_report_audit_20260619/stage_interaction_summary.json",
        "stage_reproduction_drilldown.json",
        "quick_reproduction_map",
        "5 条答辩",
        "quick route",
        "jq '.quick_reproduction_map[]",
        "jq '.rows[] \\| select(.label == \"vLLM-c8\")' results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
        "不依赖 rows 顺序",
        "jq '.vllm_c8_diagnostics.prebuild_w4'",
        "jq '.summary' results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
        "share_package_receiver_smoke_validation",
        "qwen35_omni_stage_causal_graph_zh_20260621.md",
        "manifest-backed 原始证据 Drilldown",
        "原始 artifact",
        "rg -n '原始证据 Drilldown' benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
        "主张-证据-复跑-裁决矩阵",
        "SGLang warmed c=4 优于优化版 vLLM",
        "复跑命令/入口",
        "失败时裁决",
        "rerun delta triage",
        "vLLM c=8 prebuild w4 只是 offline diagnostic",
        "当前包可分享但仍有边界",
        "/root/.cache/whisper/large-v3.pt",
        "optional warning",
        "ASR router",
        "serving benchmark 失败",
        "receiver_quickcheck_contract.json",
        "check_wer_asr_path",
        "public receiver docs preserve WER/ASR rerun path",
        "更大 Video-AMME、真实线上流量、官方 SeedTTS full-set",
        "更大数据和真实流量是下一阶段外推验证",
        "ci-50 等价于所有线上流量",
        "更大数据/真实流量外推",
    ]
    question_headings = re.findall(r"^## \d+\. ", qna_text, flags=re.MULTILINE)
    missing_questions = [
        question for question in expected_questions if question not in qna_text
    ]
    missing_terms = [term for term in required_terms if term not in qna_text]
    route_present = "qwen35_omni_defense_qna_zh_20260621.md" in share_index_text
    defense_claim_summary = defense_claim_matrix.get("summary", {})
    defense_claim_ready = (
        bool(defense_claim_summary.get("ready"))
        and _int_value(defense_claim_summary.get("rows_total")) >= 10
        and _int_value(defense_claim_summary.get("question_rows_total")) >= 13
        and _int_value(defense_claim_summary.get("checks_total")) >= 17
        and _int_value(defense_claim_summary.get("required_failures"), default=1)
        == 0
        and bool(defense_claim_summary.get("qna_questions_covered"))
        and bool(defense_claim_summary.get("qna_claims_covered"))
        and bool(defense_claim_summary.get("qna_wer_asr_path_guard"))
        and bool(defense_claim_summary.get("qna_full_traffic_scope_guard"))
        and bool(defense_claim_summary.get("qna_current_matrix_gate_marker"))
        and _int_value(defense_claim_summary.get("failure_decisions_total")) >= 10
    )
    condition = (
        len(question_headings) >= len(expected_questions)
        and not missing_questions
        and not missing_terms
        and qna_text.count("短答：") >= 13
        and qna_text.count("可说：") >= 12
        and qna_text.count("不能说：") >= 8
        and qna_text.count("证据入口：") >= 13
        and len(evidence_card_table_lines) >= 10
        and not malformed_evidence_card_rows
        and len(claim_matrix_table_lines) >= 12
        and not malformed_claim_matrix_rows
        and defense_claim_ready
        and route_present
    )
    evidence = (
        f"questions={len(question_headings)}, short_answers={qna_text.count('短答：')}, "
        f"safe_wording={qna_text.count('可说：')}, unsafe_boundaries={qna_text.count('不能说：')}, "
        f"evidence_entries={qna_text.count('证据入口：')}, route_present={route_present}, "
        f"evidence_card_rows={len(evidence_card_table_lines)}, "
        f"malformed_evidence_card_rows={malformed_evidence_card_rows}, "
        f"claim_matrix_rows={len(claim_matrix_table_lines)}, "
        f"malformed_claim_matrix_rows={malformed_claim_matrix_rows}, "
        f"defense_claim_matrix={defense_claim_summary}, "
        f"missing_questions={missing_questions}, missing_terms={missing_terms}"
    )
    return condition, evidence


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


def _audit_green_or_in_progress(audit_summary: dict[str, Any]) -> bool:
    return bool(audit_summary.get("ok")) or (
        bool(audit_summary.get("in_progress"))
        and bool(audit_summary.get("steps"))
    )


def _audit_status_evidence(audit_summary: dict[str, Any], summary_path: Path) -> str:
    steps = len(audit_summary.get("steps", []))
    if audit_summary.get("in_progress"):
        return (
            "audit_summary_signal=green_during_full_audit_refresh, "
            f"ok_so_far={audit_summary.get('ok')}, steps_seen={steps}; "
            "final audit_run_summary is rewritten at audit exit: "
            f"{summary_path}"
        )
    return (
        f"final_summary_ok={audit_summary.get('ok')}, steps={steps}: "
        f"{summary_path}"
    )


def build_readiness(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    audit_summary = _load_json_optional(audit_dir / "audit_run_summary.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    coverage = _load_json_optional(audit_dir / "coverage_matrix.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    scorecard = _load_json_optional(audit_dir / "headline_scorecard.json")
    chart_pack = _load_json_optional(audit_dir / "share_charts/chart_pack_manifest.json")
    chart_source_consistency = _load_json_optional(
        audit_dir / "chart_source_consistency.json"
    )
    slide_asset_map = _load_json_optional(audit_dir / "slide_asset_map.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    regime_decision_matrix = _load_json_optional(
        audit_dir / "regime_decision_matrix.json"
    )
    university_technical_report = _load_json_optional(
        audit_dir / "university_technical_report.json"
    )
    runtime_comparison_contract = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    objective = _load_json_optional(audit_dir / "objective_completion_audit.json")
    repro_commands = _load_json_optional(audit_dir / "repro_command_manifest.json")
    rerun_time_budget = _load_json_optional(audit_dir / "rerun_time_budget.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    stage_interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    runtime_image_contract = _load_json_optional(
        audit_dir / "runtime_image_contract.json"
    )
    rerun_acceptance_contract = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    rerun_delta_triage = _load_json_optional(audit_dir / "rerun_delta_triage.json")
    defense_claim_matrix = _load_json_optional(
        audit_dir / "defense_claim_matrix.json"
    )
    claim_metric_crosswalk = _load_json_optional(
        audit_dir / "claim_metric_crosswalk.json"
    )
    objective_requirement_crosswalk = _load_json_optional(
        audit_dir / "objective_requirement_crosswalk.json"
    )
    optimization_candidate_ledger = _load_json_optional(
        audit_dir / "optimization_candidate_ledger.json"
    )
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online_protocol = _load_json_optional(
        audit_dir / "vllm_online_parity_protocol.json"
    )
    checkpoint_watchlist = _load_json_optional(
        audit_dir / "final_checkpoint_watchlist.json"
    )
    pressure_stage_heatmap = _load_json_optional(
        audit_dir / "pressure_stage_heatmap.json"
    )
    length_regime_coverage = _load_json_optional(
        audit_dir / "length_regime_coverage.json"
    )
    stage_latency_budget = _load_json_optional(
        audit_dir / "stage_latency_budget.json"
    )
    stage_boundary_ledger = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    serving_capacity_matrix = _load_json_optional(
        audit_dir / "serving_capacity_matrix.json"
    )
    stage_causal_graph_json = _load_json_optional(
        audit_dir / "stage_causal_graph.json"
    )
    stage_drilldown_index = _load_json_optional(
        audit_dir / "stage_drilldown_index.json"
    )
    metric_provenance_index = _load_json_optional(
        audit_dir / "metric_provenance_index.json"
    )
    stage_reproduction_drilldown = _load_json_optional(
        audit_dir / "stage_reproduction_drilldown.json"
    )
    stage_route_decision_matrix = _load_json_optional(
        audit_dir / "stage_route_decision_matrix.json"
    )
    caveat_adjudication_matrix = _load_json_optional(
        audit_dir / "caveat_adjudication_matrix.json"
    )
    share_path_hygiene = _load_json_optional(audit_dir / "share_path_hygiene.json")
    command_reference_hygiene = _load_json_optional(
        audit_dir / "command_reference_hygiene.json"
    )
    external_standalone_validation = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    receiver_quickcheck_contract = _load_json_optional(
        audit_dir / "receiver_quickcheck_contract.json"
    )
    share_release_seal = _load_json_optional(audit_dir / "share_release_seal.json")
    tail_confidence_appendix = _load_json_optional(
        audit_dir / "tail_confidence_appendix.json"
    )

    main_report = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md"
    )
    final_note = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md"
    )
    cover_note = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md"
    )
    university_report = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md"
    )
    serving_capacity_report = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md"
    )
    final_status = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md"
    )
    share_index = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md"
    )
    collaboration_brief = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md"
    )
    receiver_path_map = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md"
    )
    receiver_quickcheck_contract_text = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md"
    )
    defense_qna = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md"
    )
    scorecard_zh = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md"
    )
    share_deck = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md"
    )
    slide_asset_map_text = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md"
    )
    handoff_runbook = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md"
    )
    reproduction_checklist = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md"
    )
    collaborator_sheet = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md"
    )
    rerun_delta_triage_text = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md"
    )
    stage_causal_graph = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md"
    )
    length_regime_report = _read_text_optional(
        root
        / "benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md"
    )
    rerun_time_budget_report = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md"
    )
    package_builder = _read_text_optional(
        root / "benchmarks/eval/build_qwen35_omni_share_bundle_package.py"
    )

    checks: list[ReadinessCheck] = []

    _check(
        checks,
        "full audit summary ready",
        _audit_green_or_in_progress(audit_summary),
        _audit_status_evidence(
            audit_summary,
            audit_dir / "audit_run_summary.json",
        ),
    )
    _check(
        checks,
        "claim verifier gate",
        bool(claims.get("passed"))
        and _int_value(claims.get("total_checks")) >= 17
        and _int_value(claims.get("failed_checks"), default=1) == 0,
        (
            f"passed={claims.get('passed')}, total={claims.get('total_checks')}, "
            f"failed={claims.get('failed_checks')}"
        ),
    )
    coverage_summary = coverage.get("summary", {})
    _check(
        checks,
        "requirement coverage gate",
        bool(coverage_summary.get("complete"))
        and _int_value(coverage_summary.get("passed")) >= 34
        and _int_value(coverage_summary.get("missing"), default=1) == 0,
        f"coverage={coverage_summary}",
    )
    preflight_summary = preflight.get("summary", {})
    _check(
        checks,
        "preflight gate",
        bool(preflight_summary.get("ready"))
        and _int_value(preflight_summary.get("total_checks")) >= 62
        and _int_value(preflight_summary.get("required_failures"), default=1) == 0,
        f"preflight={preflight_summary}",
    )
    manifest_summary = manifest.get("summary", {})
    manifest_records_current = _int_value(manifest_summary.get("total_records"))
    _check(
        checks,
        "evidence manifest gate",
        manifest_records_current >= 180
        and _int_value(manifest_summary.get("missing_records"), default=1) == 0
        and _int_value(manifest_summary.get("file_records")) >= 178,
        f"manifest={manifest_summary}",
    )
    environment_summary = environment.get("audit", {}).get("summary", {})
    _check(
        checks,
        "environment snapshot gate",
        bool(environment.get("gpu", {}).get("ok"))
        and all(
            image.get("ok")
            for image in environment.get("docker_images", {}).values()
        )
        and bool(environment_summary.get("ready")),
        (
            f"gpu_ok={environment.get('gpu', {}).get('ok')}, "
            f"docker={environment.get('docker_images', {})}, "
            f"audit_ready={environment_summary.get('ready')}"
        ),
    )
    scorecard_summary = scorecard.get("summary", {})
    metric_provenance_summary = metric_provenance_index.get("summary", {})
    _check(
        checks,
        "headline scorecard gate",
        bool(scorecard_summary.get("ready"))
        and _int_value(scorecard_summary.get("checks_passed"))
        == _int_value(scorecard_summary.get("checks_total"), default=-1)
        and _int_value(scorecard_summary.get("checks_total")) >= 9
        and bool(metric_provenance_summary.get("ready"))
        and _int_value(metric_provenance_summary.get("rows_total")) >= 150
        and _int_value(metric_provenance_summary.get("raw_artifacts_total")) >= 15
        and _int_value(metric_provenance_summary.get("command_refs_total")) >= 15
        and _int_value(metric_provenance_summary.get("required_failures"), default=1)
        == 0,
        f"headline={scorecard_summary}; metric_provenance={metric_provenance_summary}",
    )
    chart_summary = chart_pack.get("summary", {})
    _check(
        checks,
        "share chart pack gate",
        bool(chart_summary.get("ready"))
        and _int_value(chart_summary.get("csv_files")) >= 7
        and _int_value(chart_summary.get("svg_files")) >= 7
        and _int_value(chart_summary.get("generated_files")) >= 14,
        f"chart_pack={chart_summary}",
    )
    chart_source_summary = chart_source_consistency.get("summary", {})
    _check(
        checks,
        "chart source consistency gate",
        bool(chart_source_summary.get("ready"))
        and _int_value(chart_source_summary.get("checks_total")) >= 8
        and _int_value(chart_source_summary.get("checks_passed"))
        == _int_value(chart_source_summary.get("checks_total"), default=-1)
        and _int_value(chart_source_summary.get("required_failures"), default=1) == 0
        and _int_value(chart_source_summary.get("csv_files_checked")) >= 7
        and _int_value(chart_source_summary.get("svg_files_checked")) >= 7
        and _int_value(chart_source_summary.get("byte_exact_files")) >= 14,
        f"chart_source_consistency={chart_source_summary}",
    )
    slide_asset_summary = slide_asset_map.get("summary", {})
    acceptance_summary = acceptance.get("summary", {})
    _check(
        checks,
        "acceptance matrix gate",
        bool(acceptance_summary.get("ready"))
        and _int_value(acceptance_summary.get("rows_passed"))
        == _int_value(acceptance_summary.get("rows_total"), default=-1)
        and _int_value(acceptance_summary.get("rows_total")) >= 17
        and _int_value(acceptance_summary.get("rows_failed"), default=1) == 0,
        f"acceptance={acceptance_summary}",
    )
    regime_summary = regime_decision_matrix.get("summary", {})
    _check(
        checks,
        "regime decision matrix gate",
        bool(regime_summary.get("ready"))
        and _int_value(regime_summary.get("rows_total")) >= 17
        and _int_value(regime_summary.get("accepted_rows")) >= 17
        and _int_value(regime_summary.get("checks_total")) >= 9
        and _int_value(regime_summary.get("checks_passed"))
        == _int_value(regime_summary.get("checks_total"), default=-1)
        and _int_value(regime_summary.get("required_failures"), default=1) == 0,
        f"regime_decision_matrix={regime_summary}",
    )
    university_summary = university_technical_report.get("summary", {})
    university_report_ok, university_report_missing = _all_have(
        university_report,
        [
            "Qwen3.5-Omni SGLang-Omni 中文技术报告",
            "Executive Summary",
            "严格 SGLang-vLLM 对比",
            "c=4 指标口径说明",
            "strict warmed c=4 headline",
            "SGLang stress c=4",
            "不要拿来替代 strict vLLM headline",
            "Runtime fairness / 镜像与优化锁定",
            "baseline 是否公平",
            "VLLM_ENABLE_TORCH_COMPILE=True",
            "online_parity_proven",
            "SGLang 单并发和高并发压力结论",
            "短/长文本语音输出结论",
            "Serving/capacity 决策矩阵",
            "latency-first / 单并发到低并发",
            "throughput edge / 当前高并发甜点",
            "saturation evidence / 压力边界",
            "optimized offline diagnostic",
            "Stage Breakdown 和连接瓶颈",
            "Pressure × Stage Heatmap",
            "pressure_stage_heatmap.json",
            "SGLang Video-AMME stage latency budget",
            "短/长文本语音 stage latency budget",
            "vLLM offline stage latency budget",
            "Route 复现索引",
            "vLLM c=8 诊断边界",
            "优化锁和反例",
            "当前 best measured recipe 裁决",
            "best measured recipe",
            "不能说已经搜索完所有 future kernel",
            "新 recipe 必须补齐 c=4/c=8/c=16",
            "复现入口",
            "jq -r --arg id vllm_c4_original",
            "launch_sglang_optimized",
            "sglang_videoamme_stress",
            "sglang_synthetic_text_to_speech",
            "vllm_c4_original",
            "vllm_c8_prebuild_w4",
            "可分享边界",
            "现场答辩速查",
            "defense_claim_matrix.json",
            "失败时裁决",
            "tail_confidence_appendix.json",
            "stage_latency_budget.json",
            "stage_boundary_bottleneck_ledger.json",
            "regime_decision_matrix.json",
            "stage_route_decision_matrix.json",
            "stage_reproduction_drilldown.json",
            "repro_command_manifest.json",
        ],
    )
    _check(
        checks,
        "university technical report gate",
        bool(university_summary.get("ready"))
        and _int_value(university_summary.get("checks_total")) >= 16
        and _int_value(university_summary.get("checks_passed"))
        == _int_value(university_summary.get("checks_total"), default=-1)
        and _int_value(university_summary.get("required_failures"), default=1) == 0
        and university_report_ok,
        (
            f"university_technical_report={university_summary}; "
            f"missing={university_report_missing}"
        ),
    )
    runtime_comparison_summary = runtime_comparison_contract.get("summary", {})
    _check(
        checks,
        "runtime comparison contract gate",
        bool(runtime_comparison_summary.get("ready"))
        and _int_value(runtime_comparison_summary.get("checks_total")) >= 9
        and _int_value(runtime_comparison_summary.get("required_failures"), default=1)
        == 0
        and runtime_comparison_summary.get("allowed_cross_runtime_headline")
        == "warmed c=4 only"
        and runtime_comparison_summary.get("vllm_c8_contract")
        == "offline_diagnostic_not_online_parity"
        and "optimized image" in str(
            runtime_comparison_summary.get("baseline_strength") or ""
        ),
        f"runtime_comparison_contract={runtime_comparison_summary}",
    )
    confidence_summary = confidence.get("summary", {})
    _check(
        checks,
        "confidence ledger gate",
        bool(confidence_summary.get("ready"))
        and _int_value(confidence_summary.get("entries_passed"))
        == _int_value(confidence_summary.get("entries_total"), default=-1)
        and _int_value(confidence_summary.get("entries_total")) >= 12
        and _int_value(confidence_summary.get("unsupported_claims"), default=1) == 0,
        f"confidence={confidence_summary}",
    )
    caveat_summary = caveat_adjudication_matrix.get("summary", {})
    _check(
        checks,
        "caveat adjudication matrix gate",
        bool(caveat_summary.get("ready"))
        and _int_value(caveat_summary.get("rows_total")) >= 12
        and _int_value(caveat_summary.get("must_travel_rows_total")) >= 12
        and _int_value(caveat_summary.get("forbidden_claims_total")) >= 12
        and _int_value(caveat_summary.get("replacement_triggers_total")) >= 12
        and _int_value(caveat_summary.get("checks_total")) >= 14
        and _int_value(caveat_summary.get("required_failures"), default=1) == 0
        and not bool(caveat_summary.get("online_parity_proven"))
        and not bool(caveat_summary.get("seedtts_fullset_headline"))
        and caveat_summary.get("current_best_scope")
        == "measured_best_not_global_optimum"
        and not bool(caveat_summary.get("goal_complete")),
        f"caveat_adjudication_matrix={caveat_summary}",
    )
    objective_summary = objective.get("summary", {})
    objective_requirement_summary = objective_requirement_crosswalk.get("summary", {})
    optimization_candidate_summary = optimization_candidate_ledger.get("summary", {})
    _check(
        checks,
        "original objective completion gate",
        bool(objective_summary.get("share_ready_with_documented_caveats"))
        and _int_value(objective_summary.get("rows_total")) >= 17
        and _int_value(objective_summary.get("required_failures"), default=1) == 0
        and bool(objective_requirement_summary.get("ready"))
        and _int_value(objective_requirement_summary.get("requirement_rows_total")) >= 11
        and _int_value(objective_requirement_summary.get("unique_metric_rows_total")) >= 85
        and bool(
            objective_requirement_summary.get("optimization_candidate_ledger_ready")
        )
        and _int_value(
            objective_requirement_summary.get("optimization_candidate_rows_total")
        )
        >= 8
        and _int_value(
            objective_requirement_summary.get("optimization_rejected_anti_recipes_total")
        )
        >= 2
        and _int_value(
            objective_requirement_summary.get("optimization_vllm_diagnostic_rows_total")
        )
        >= 2
        and objective_requirement_summary.get(
            "optimization_current_best_candidate_id"
        )
        == "sglang_current_best_measured_recipe"
        and _int_value(objective_requirement_summary.get("required_failures"), default=1)
        == 0
        and not bool(objective_requirement_summary.get("goal_complete")),
        (
            f"objective_completion={objective_summary}; "
            f"objective_requirement_crosswalk={objective_requirement_summary}"
        ),
    )
    _check(
        checks,
        "optimization candidate ledger gate",
        bool(optimization_candidate_summary.get("ready"))
        and _int_value(optimization_candidate_summary.get("candidate_rows_total")) >= 8
        and optimization_candidate_summary.get("current_best_candidate_id")
        == "sglang_current_best_measured_recipe"
        and _int_value(
            optimization_candidate_summary.get("accepted_current_best_rows_total")
        )
        >= 1
        and _int_value(
            optimization_candidate_summary.get("rejected_anti_recipe_rows_total")
        )
        >= 2
        and _int_value(optimization_candidate_summary.get("vllm_diagnostic_rows_total"))
        >= 2
        and bool(optimization_candidate_summary.get("not_global_optimum_boundary"))
        and _int_value(
            optimization_candidate_summary.get("missing_metric_row_ids"), default=1
        )
        == 0
        and _int_value(
            optimization_candidate_summary.get("missing_evidence_files"), default=1
        )
        == 0
        and _int_value(
            optimization_candidate_summary.get("missing_command_ids"), default=1
        )
        == 0
        and _int_value(
            optimization_candidate_summary.get("required_failures"), default=1
        )
        == 0,
        f"optimization_candidate_ledger={optimization_candidate_summary}",
    )
    repro_summary = repro_commands.get("summary", {})
    repro_commands_total = _int_value(repro_summary.get("commands_total"))
    repro_phases_total = _int_value(repro_summary.get("phases_total"))
    repro_expected_manifest_gate = (
        repro_summary.get("expected_gates", {}).get("manifest", {})
        if isinstance(repro_summary.get("expected_gates", {}), dict)
        else {}
    )
    repro_expected_manifest_gate = (
        repro_expected_manifest_gate
        if isinstance(repro_expected_manifest_gate, dict)
        else {}
    )
    repro_manifest_gate_zh = (
        f"repro command manifest `ready=true`，"
        f"{repro_commands_total} 条命令、{repro_phases_total} 个阶段"
    )
    repro_manifest_gate_en = (
        f"repro command manifest: `ready=true`, "
        f"`{repro_commands_total}` commands / `{repro_phases_total}` phases"
    )
    command_reference_summary = command_reference_hygiene.get("summary", {})
    rerun_time_budget_summary = rerun_time_budget.get("summary", {})
    rerun_time_budget_text_ok, rerun_time_budget_text_missing = _all_have(
        rerun_time_budget_report,
        [
            "复跑耗时/算力预算",
            "rerun_time_budget.json",
            "sglang_videoamme_stress",
            "sglang_synthetic_text_to_speech",
            "vllm_c4_original",
            "vllm_c8_prebuild_w4",
            "不包含 server launch/warmup/WER/ASR",
        ],
    )
    _check(
        checks,
        "reproduction command manifest gate",
        bool(repro_summary.get("ready"))
        and bool(repro_summary.get("required_command_ids_present"))
        and _int_value(repro_summary.get("commands_total")) >= 60
        and bool(command_reference_summary.get("ready"))
        and _int_value(command_reference_summary.get("required_failures"), default=1)
        == 0
        and _int_value(command_reference_summary.get("checks_total")) >= 6
        and _int_value(
            command_reference_summary.get("unresolved_command_refs_total"), default=1
        )
        == 0
        and _int_value(
            repro_expected_manifest_gate.get("min_total_records")
        )
        >= 180
        and _int_value(repro_expected_manifest_gate.get("min_file_records")) >= 178
        and "total_records" not in repro_expected_manifest_gate
        and "file_records" not in repro_expected_manifest_gate
        and bool(rerun_time_budget_summary.get("ready"))
        and _int_value(rerun_time_budget_summary.get("required_failures"), default=1)
        == 0
        and _int_value(rerun_time_budget_summary.get("rows_total")) >= 8
        and _int_value(rerun_time_budget_summary.get("timed_rows")) >= 6
        and _int_value(rerun_time_budget_summary.get("command_refs_total")) >= 8
        and bool(rerun_time_budget_summary.get("required_command_ids_present"))
        and _float_value(
            rerun_time_budget_summary.get("total_timed_benchmark_wall_s")
        )
        > 0
        and _float_value(
            rerun_time_budget_summary.get(
                "equivalent_8gpu_timed_lower_bound_gpu_hours"
            )
        )
        > 0
        and rerun_time_budget_text_ok,
        (
            f"repro_command_manifest={repro_summary}; "
            f"command_reference_hygiene={command_reference_summary}; "
            f"rerun_time_budget={rerun_time_budget_summary}; "
            f"rerun_time_budget_missing={rerun_time_budget_text_missing}"
        ),
    )
    runtime_image_summary = runtime_image_contract.get("summary", {})
    _check(
        checks,
        "runtime image contract gate",
        bool(runtime_image_summary.get("ready"))
        and _int_value(runtime_image_summary.get("required_failures"), default=1) == 0
        and _int_value(runtime_image_summary.get("checks_total")) >= 12
        and "c4-c8" in str(runtime_image_summary.get("sglang_scope") or "")
        and "c4" in str(runtime_image_summary.get("vllm_strict_scope") or "")
        and "offline diagnostic" in str(runtime_image_summary.get("vllm_c8_scope") or ""),
        f"runtime_image_contract={runtime_image_summary}",
    )
    rerun_acceptance_summary = rerun_acceptance_contract.get("summary", {})
    _check(
        checks,
        "rerun acceptance contract gate",
        bool(rerun_acceptance_summary.get("ready"))
        and _int_value(rerun_acceptance_summary.get("required_failures"), default=1)
        == 0
        and _int_value(rerun_acceptance_summary.get("checks_total")) >= 17
        and _int_value(rerun_acceptance_summary.get("rules_total")) >= 18
        and _int_value(rerun_acceptance_summary.get("return_evidence_files")) >= 34
        and _int_value(rerun_acceptance_summary.get("return_evidence_command_rows"))
        >= 27
        and _int_value(
            rerun_acceptance_summary.get("return_evidence_command_missing"),
            default=1,
        )
        == 0
        and _int_value(
            rerun_acceptance_summary.get("return_evidence_command_file_gaps"),
            default=1,
        )
        == 0
        and "same hardware/image/model/data"
        in str(rerun_acceptance_summary.get("replacement_scope") or ""),
        f"rerun_acceptance_contract={rerun_acceptance_summary}",
    )
    rerun_delta_triage_summary = rerun_delta_triage.get("summary", {})
    sglang_lock_summary = sglang_lock.get("summary", {})
    _check(
        checks,
        "SGLang optimization lock gate",
        bool(sglang_lock_summary.get("ready"))
        and _int_value(sglang_lock_summary.get("required_failures"), default=1) == 0
        and _int_value(sglang_lock_summary.get("checks_total")) >= 26,
        f"sglang_optimization_lock={sglang_lock_summary}",
    )
    vllm_lock_summary = vllm_lock.get("summary", {})
    _check(
        checks,
        "vLLM optimization lock gate",
        bool(vllm_lock_summary.get("ready"))
        and _int_value(vllm_lock_summary.get("required_failures"), default=1) == 0
        and _int_value(vllm_lock_summary.get("checks_total")) >= 22,
        f"vllm_optimization_lock={vllm_lock_summary}",
    )
    vllm_online_summary = vllm_online_protocol.get("summary", {})
    _check(
        checks,
        "vLLM online parity protocol gate",
        bool(vllm_online_summary.get("ready"))
        and _int_value(vllm_online_summary.get("required_failures"), default=1) == 0
        and _int_value(vllm_online_summary.get("checks_total")) >= 18
        and bool(vllm_online_summary.get("current_package_safe"))
        and not bool(vllm_online_summary.get("online_parity_proven")),
        f"vllm_online_parity_protocol={vllm_online_summary}",
    )
    checkpoint_summary = checkpoint_watchlist.get("summary", {})
    _check(
        checks,
        "final checkpoint watchlist gate",
        bool(checkpoint_summary.get("ready"))
        and _int_value(checkpoint_summary.get("required_failures"), default=1) == 0
        and _int_value(checkpoint_summary.get("checks_total")) >= 24
        and _int_value(checkpoint_summary.get("watch_items_total")) >= 7
        and checkpoint_summary.get("checkpoint_phase") == "completion_audit_ready"
        and bool(checkpoint_summary.get("share_ready_with_documented_caveats"))
        and bool(checkpoint_summary.get("final_completion_evidence_ready"))
        and not bool(checkpoint_summary.get("goal_complete"))
        and not any(
            "waiting_for_2026-06-21" in str(blocker)
            for blocker in checkpoint_summary.get("completion_blockers", []) or []
        ),
        f"final_checkpoint_watchlist={checkpoint_summary}",
    )
    tail_confidence_summary = tail_confidence_appendix.get("summary", {})
    _check(
        checks,
        "tail confidence appendix gate",
        bool(tail_confidence_summary.get("ready"))
        and _int_value(tail_confidence_summary.get("required_failures"), default=1)
        == 0
        and _int_value(tail_confidence_summary.get("checks_total")) >= 13
        and _int_value(tail_confidence_summary.get("rows_total")) >= 18
        and _int_value(tail_confidence_summary.get("bootstrap_rows_total")) >= 9
        and _float_value(
            tail_confidence_summary.get(
                "strict_c4_latency_mean_advantage_ci95_low_s"
            )
        )
        > 0.0
        and _float_value(tail_confidence_summary.get("long_c8_rtf_p95_ci95_high"))
        < 1.0
        and _float_value(tail_confidence_summary.get("strict_c4_sglang_latency_p95_s"))
        < _float_value(tail_confidence_summary.get("strict_c4_vllm_latency_p95_s"))
        and _float_value(tail_confidence_summary.get("strict_c4_sglang_rtf_p95"))
        < _float_value(tail_confidence_summary.get("strict_c4_vllm_rtf_p95"))
        and _float_value(tail_confidence_summary.get("sglang_c8_qps"))
        > _float_value(tail_confidence_summary.get("sglang_c16_qps")),
        f"tail_confidence_appendix={tail_confidence_summary}",
    )
    stage_budget_summary = stage_latency_budget.get("summary", {})
    _check(
        checks,
        "stage latency budget gate",
        bool(stage_budget_summary.get("ready"))
        and _int_value(stage_budget_summary.get("required_failures"), default=1) == 0
        and _int_value(stage_budget_summary.get("checks_total")) >= 12
        and _int_value(stage_budget_summary.get("sglang_budget_rows")) >= 5
        and _int_value(stage_budget_summary.get("synthetic_budget_rows")) >= 6
        and _int_value(stage_budget_summary.get("vllm_budget_rows")) >= 4,
        f"stage_latency_budget={stage_budget_summary}",
    )
    stage_ledger_summary = stage_boundary_ledger.get("summary", {})
    _check(
        checks,
        "stage boundary bottleneck ledger gate",
        bool(stage_ledger_summary.get("ready"))
        and _int_value(stage_ledger_summary.get("required_failures"), default=1) == 0
        and _int_value(stage_ledger_summary.get("checks_total")) >= 12
        and _int_value(stage_ledger_summary.get("ledger_rows")) >= 37
        and _int_value(stage_ledger_summary.get("pressure_transition_rows")) >= 11,
        f"stage_boundary_bottleneck_ledger={stage_ledger_summary}",
    )
    serving_capacity_summary = serving_capacity_matrix.get("summary", {})
    serving_capacity_text_ok, serving_capacity_text_missing = _all_have(
        serving_capacity_report,
        [
            "Serving/Capacity 决策矩阵",
            "Video-AMME c=8",
            "Synthetic long text c=8",
            "vLLM c=8 prebuild w4",
            "optimized offline diagnostic",
            "不要升级为 online serving parity",
            "build_serving_capacity_matrix",
        ],
    )
    _check(
        checks,
        "serving capacity matrix gate",
        bool(serving_capacity_summary.get("ready"))
        and _int_value(serving_capacity_summary.get("required_failures"), default=1)
        == 0
        and _int_value(serving_capacity_summary.get("checks_total")) >= 10
        and _int_value(serving_capacity_summary.get("checks_passed"))
        == _int_value(serving_capacity_summary.get("checks_total"), default=-1)
        and _int_value(serving_capacity_summary.get("rows_total")) >= 7
        and _float_value(serving_capacity_summary.get("sglang_c8_qps"))
        > _float_value(serving_capacity_summary.get("sglang_c16_qps"))
        and _float_value(serving_capacity_summary.get("long_c8_rtf_p95")) < 1.0
        and serving_capacity_summary.get("vllm_w4_scope")
        == "offline_diagnostic_only"
        and serving_capacity_text_ok,
        (
            f"serving_capacity_matrix={serving_capacity_summary}; "
            "missing=" + ", ".join(serving_capacity_text_missing)
        ),
    )
    share_bundle_summary = share_bundle.get("summary", {})
    _check(
        checks,
        "share bundle manifest gate",
        bool(share_bundle_summary.get("ready"))
        and _int_value(share_bundle_summary.get("missing_required"), default=1) == 0
        and _int_value(share_bundle_summary.get("records_total")) >= 70
        and _int_value(share_bundle_summary.get("file_records")) >= 69
        and _int_value(
            share_bundle_summary.get("category_counts", {}).get("share_report")
        )
        >= len(REPORT_FILES),
        f"share_bundle_manifest={share_bundle_summary}",
    )
    stage_summary = stage_interactions.get("summary", {})
    pressure_stage_heatmap_summary = pressure_stage_heatmap.get("summary", {})
    length_regime_summary = length_regime_coverage.get("summary", {})
    stage_drilldown_summary = stage_drilldown_index.get("summary", {})
    stage_reproduction_summary = stage_reproduction_drilldown.get("summary", {})
    stage_route_summary = stage_route_decision_matrix.get("summary", {})
    stage_causal_graph_summary = stage_causal_graph_json.get("summary", {})
    stage_causal_ok, stage_causal_missing = _all_have(
        stage_causal_graph,
        [
            "原始证据 Drilldown",
            "Raw artifact drilldown",
            "tables summary raw path index",
            "vLLM admission diagnosis",
            "SGLang c=8/c=16 高并发为什么是 admission/queue",
            "SGLang talker->code2wav handoff 是否卡住",
            "短/长文本语音是否覆盖",
            "vLLM original c=8 为什么不能当 online parity",
            "vLLM c=8 prebuild w4 改善了什么",
            "manifest 证据清单",
            "results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json",
            "results/qwen35_sglang_mr8_stress_20260619/request_profile_c8_profile_skipwer.json",
            "results/qwen35_synthetic_speech_20260619/long_c8/synthetic_speech_results.json",
            "results/qwen35_synthetic_speech_20260619/request_profile_long_c8_profile.json",
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
        ],
    )
    stage_causal_manifest_ok, stage_causal_manifest_evidence = (
        _stage_causal_manifest_backing(stage_causal_graph, manifest)
    )
    length_regime_text_ok, length_regime_text_missing = _all_have(
        length_regime_report,
        [
            "长短输入/输出 Length-Regime 覆盖矩阵",
            "Video-AMME ci-50",
            "74 chars / 12 words",
            "944 chars / 139 words",
            "long c=8 仍快于实时",
            "talker->code2wav hop p95",
            "code2wav decode p95",
            "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
            "length_regime_coverage.json",
            "synthetic_short_long_speech.csv",
        ],
    )
    _check(
        checks,
        "stage interaction gate",
        _int_value(stage_summary.get("total_interactions")) >= 30
        and bool(stage_summary.get("sglang_talker_to_code2wav_healthy"))
        and bool(stage_summary.get("sglang_code2wav_decode_not_bottleneck"))
        and bool(stage_summary.get("vllm_original_c8_prompt_feed_limited"))
        and bool(stage_summary.get("preprocessing_parallelism_regresses"))
        and bool(stage_drilldown_summary.get("ready"))
        and _int_value(stage_drilldown_summary.get("rows_total")) >= 52
        and _int_value(stage_drilldown_summary.get("required_failures"), default=1)
        == 0
        and _int_value(stage_drilldown_summary.get("stage_routes_total")) >= 7
        and bool(stage_reproduction_summary.get("ready"))
        and _int_value(stage_reproduction_summary.get("stage_rows_total")) >= 52
        and _int_value(stage_reproduction_summary.get("route_rows_total")) >= 11
        and _int_value(stage_reproduction_summary.get("command_refs_total")) >= 15
        and _int_value(stage_reproduction_summary.get("quick_reproduction_routes_total"))
        >= 5
        and _int_value(stage_reproduction_summary.get("checks_total")) >= 17
        and _int_value(stage_reproduction_summary.get("checks_passed")) >= 17
        and _int_value(stage_reproduction_summary.get("required_failures"), default=1)
        == 0
        and bool(pressure_stage_heatmap_summary.get("ready"))
        and _int_value(pressure_stage_heatmap_summary.get("rows_total")) >= 15
        and _int_value(pressure_stage_heatmap_summary.get("sglang_videoamme_rows")) >= 5
        and _int_value(pressure_stage_heatmap_summary.get("synthetic_rows")) >= 6
        and _int_value(pressure_stage_heatmap_summary.get("vllm_rows")) >= 4
        and _int_value(pressure_stage_heatmap_summary.get("checks_total")) >= 11
        and _int_value(
            pressure_stage_heatmap_summary.get("required_failures"), default=1
        )
        == 0
        and bool(length_regime_summary.get("ready"))
        and _int_value(length_regime_summary.get("required_failures"), default=1)
        == 0
        and _int_value(length_regime_summary.get("rows_total")) >= 7
        and _int_value(length_regime_summary.get("synthetic_rows")) >= 6
        and _int_value(length_regime_summary.get("short_rows")) >= 3
        and _int_value(length_regime_summary.get("long_rows")) >= 3
        and _float_value(length_regime_summary.get("short_target_chars")) == 74.0
        and _float_value(length_regime_summary.get("long_target_chars")) == 944.0
        and _float_value(length_regime_summary.get("long_c8_rtf_p95")) < 1.0
        and _float_value(
            length_regime_summary.get("max_talker_to_code2wav_hop_p95_ms"),
            default=999.0,
        )
        <= 30.0
        and _float_value(
            length_regime_summary.get("max_code2wav_decode_p95_ms"),
            default=999.0,
        )
        <= 30.0
        and length_regime_text_ok
        and bool(stage_route_summary.get("ready"))
        and _int_value(stage_route_summary.get("route_rows_total")) >= 11
        and _int_value(stage_route_summary.get("stage_rows_total")) >= 52
        and _int_value(stage_route_summary.get("command_refs_total")) >= 15
        and _int_value(stage_route_summary.get("required_failures"), default=1) == 0
        and bool(stage_causal_graph_summary.get("ready"))
        and _int_value(stage_causal_graph_summary.get("checks_total")) >= 7
        and _int_value(stage_causal_graph_summary.get("required_failures"), default=1)
        == 0
        and _int_value(stage_causal_graph_summary.get("causal_edges_total")) >= 7
        and _int_value(stage_causal_graph_summary.get("raw_drilldown_rows")) >= 5
        and _int_value(stage_causal_graph_summary.get("raw_artifacts_total")) >= 14
        and _int_value(
            stage_causal_graph_summary.get("manifest_missing_raw_artifacts"),
            default=1,
        )
        == 0
        and bool(stage_causal_graph_summary.get("sglang_handoff_healthy"))
        and bool(stage_causal_graph_summary.get("sglang_decode_not_bottleneck"))
        and stage_causal_ok
        and stage_causal_manifest_ok,
        (
            f"stage_interactions={stage_summary}; "
            f"stage_drilldown_index={stage_drilldown_summary}; "
            f"stage_reproduction_drilldown={stage_reproduction_summary}; "
            f"pressure_stage_heatmap={pressure_stage_heatmap_summary}; "
            f"length_regime_coverage={length_regime_summary}; "
            f"length_regime_missing={length_regime_text_missing}; "
            f"stage_route_decision_matrix={stage_route_summary}; "
            f"stage_causal_graph={stage_causal_graph_summary}; "
            f"stage_causal_missing={stage_causal_missing}; "
            f"stage_causal_manifest={stage_causal_manifest_evidence}"
        ),
    )

    reports_exist, missing_reports = _all_exist(root, REPORT_FILES)
    _check(
        checks,
        "share report files present",
        reports_exist,
        f"reports={len(REPORT_FILES)}, missing={missing_reports}",
    )
    public_doc_hash_offenders = _public_doc_hardcoded_hashes(root, REPORT_FILES)
    public_doc_table_offenders = _public_doc_markdown_table_offenders(
        root, REPORT_FILES
    )
    public_doc_malformed_tokens = _public_doc_malformed_text_tokens(
        root, REPORT_FILES
    )
    public_doc_duplicate_heading_offenders = _public_doc_duplicate_heading_offenders(
        root, REPORT_FILES
    )
    public_doc_semantic_offenders = _public_doc_semantic_consistency_offenders(
        root, REPORT_FILES
    )
    pressure_matrix_na_legend_missing = _pressure_matrix_na_legend_missing(root)
    _check(
        checks,
        "public share docs quality guard",
        not public_doc_hash_offenders
        and not public_doc_table_offenders
        and not public_doc_malformed_tokens
        and not public_doc_duplicate_heading_offenders
        and not public_doc_semantic_offenders
        and not pressure_matrix_na_legend_missing,
        (
            f"hash_offenders={public_doc_hash_offenders[:10]}, "
            f"table_offenders={public_doc_table_offenders[:10]}, "
            f"malformed_tokens={public_doc_malformed_tokens[:10]}, "
            f"duplicate_heading_offenders={public_doc_duplicate_heading_offenders[:10]}, "
            f"semantic_offenders={public_doc_semantic_offenders[:10]}, "
            f"pressure_matrix_na_legend_missing={pressure_matrix_na_legend_missing[:10]}"
        ),
    )
    defense_qna_ok, defense_qna_evidence = _defense_qna_readiness(
        defense_qna, share_index, defense_claim_matrix
    )
    claim_metric_summary = claim_metric_crosswalk.get("summary", {})
    claim_metric_ready = (
        bool(claim_metric_summary.get("ready"))
        and _int_value(claim_metric_summary.get("claims_total")) >= 10
        and _int_value(claim_metric_summary.get("unique_metric_rows_total")) >= 60
        and _int_value(claim_metric_summary.get("raw_artifacts_total")) >= 20
        and _int_value(claim_metric_summary.get("command_refs_total")) >= 15
        and _int_value(claim_metric_summary.get("required_failures"), default=1)
        == 0
    )
    _check(
        checks,
        "defense Q&A readiness gate",
        defense_qna_ok and claim_metric_ready,
        f"{defense_qna_evidence}; claim_metric_crosswalk={claim_metric_summary}",
    )
    machine_files_exist, missing_machine_files = _all_exist(root, MACHINE_EVIDENCE_FILES)
    _check(
        checks,
        "machine evidence files present",
        machine_files_exist,
        f"machine_files={len(MACHINE_EVIDENCE_FILES)}, missing={missing_machine_files}",
    )
    share_tools_exist, missing_share_tools = _all_exist(root, SHARE_TOOL_FILES)
    _check(
        checks,
        "share validation tools present",
        share_tools_exist,
        f"share_tools={len(SHARE_TOOL_FILES)}, missing={missing_share_tools}",
    )

    final_note_ok, final_note_missing = _all_have(
        final_note,
        [
            "最终分享交付说明",
            "中文技术报告",
            "qwen35_omni_university_technical_report_zh_20260621.md",
            "qwen35_omni_receiver_command_card_zh_20260621.md",
            "qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
            "一页式收包快检",
            "full audit `ok=true`",
            "coverage `34/34`",
            "preflight `62`",
            "manifest current",
            "minimum `180`",
            "checkpoint watchlist",
            "stage latency budget",
            "stage boundary bottleneck ledger",
            "Deck 图表资产映射",
            "slide asset map",
            "chart source consistency",
            "results/qwen35_report_audit_20260619/chart_source_consistency.json",
            "runtime image contract",
            "rerun acceptance contract",
            "34 return evidence files",
            "share bundle manifest",
            "final readiness audit `ready=true`",
            "public doc quality guard",
            "public_doc_quality_guard",
            "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift",
            "hash/table/token/duplicate-heading/semantic-count offenders",
            "伴随验证证据",
            "接收方拿到 tarball 后，先做收包快检",
            "receiver_smoke_ready=true",
            "tarball validation `17/17`",
            "nested extracted-only validation `13/13`",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
            "public_doc_quality_guard",
            "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift",
            "vLLM c=8 prebuild w4 是 offline",
            "vLLM online parity protocol",
            "final checkpoint watchlist",
            "stage latency budget",
            "official SeedTTS full-set 还不是 headline evidence",
            "runtime comparison contract",
            "runtime image contract",
            "rerun acceptance contract",
            "SGLang optimization lock",
            "vLLM optimization lock",
            "stage causal graph",
            "manifest-backed 原始证据 Drilldown",
            "stage boundary bottleneck ledger",
            "5 条答辩 quick route",
            "caveat adjudication matrix",
            "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json",
            "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
        ],
    )
    entry_scope_redline = "ci-50/stress/synthetic 证据不能直接外推到完整线上流量"
    entry_scope_missing = [
        label
        for label, text in [
            ("cover_note", cover_note),
            ("final_note", final_note),
            ("share_index", share_index),
            ("collaboration_brief", collaboration_brief),
            ("final_status", final_status),
        ]
        if entry_scope_redline not in text
    ]
    final_note_ok = final_note_ok and not entry_scope_missing
    final_note_missing = list(final_note_missing) + [
        f"{label}:{entry_scope_redline}" for label in entry_scope_missing
    ]
    _check(
        checks,
        "final delivery note wording gate",
        final_note_ok,
        "missing=" + ", ".join(final_note_missing),
    )
    share_index_ok, share_index_missing = _all_have(
        share_index,
        [
            "机器证据入口",
            "qwen35_omni_receiver_command_card_zh_20260621.md",
            "qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
            "一页式复制命令",
            "qwen35_omni_university_technical_report_zh_20260621.md",
            "university_technical_report.json",
            "qwen35_omni_serving_capacity_matrix_zh_20260621.md",
            "serving_capacity_matrix.json",
            "qwen35_omni_regime_decision_matrix_zh_20260621.md",
            "regime_decision_matrix.json",
            "qwen35_omni_runtime_comparison_contract_zh_20260621.md",
            "qwen35_omni_runtime_image_contract_zh_20260621.md",
            "qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
            "qwen35_omni_rerun_delta_triage_zh_20260621.md",
            "qwen35_omni_sglang_optimization_lock_zh_20260621.md",
            "qwen35_omni_vllm_optimization_lock_zh_20260621.md",
            "qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
            "optimization_candidate_ledger.json",
            "qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
            "qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
            "qwen35_omni_tail_confidence_appendix_zh_20260621.md",
            "qwen35_omni_stage_latency_budget_zh_20260621.md",
            "qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            "qwen35_omni_stage_causal_graph_zh_20260621.md",
            "qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
            "caveat_adjudication_matrix.json",
            "manifest-backed 原始证据 Drilldown",
            "manifest 证据清单",
            "final readiness audit",
            "repro command manifest",
            "coverage `34/34`",
            "manifest current",
            "minimum `180`",
            "checkpoint watchlist",
            "stage latency budget",
            "stage boundary bottleneck ledger",
            "stage reproduction drilldown",
            "5 条答辩 quick route",
            "quick_reproduction_map",
            "runtime image contract",
            "rerun acceptance contract",
            "return evidence files `34`",
            "rerun delta triage",
            "收包快检",
            "release seal",
            "sha256sum -c",
            "tarball validation `17/17`",
            "validation `13/13`、standalone validation `8/8` 后进入报告阅读",
            "public_doc_quality_guard",
            "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift",
            "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
        ],
    )
    _check(
        checks,
        "share package index route gate",
        share_index_ok,
        "missing=" + ", ".join(share_index_missing),
    )
    share_index_structure_ok, share_index_structure_evidence = _share_index_structure(
        share_index
    )
    navigation_coverage_ok, navigation_coverage_evidence = (
        _share_index_navigation_coverage(share_index)
    )
    _check(
        checks,
        "share package index structure gate",
        share_index_structure_ok and navigation_coverage_ok,
        share_index_structure_evidence + "; " + navigation_coverage_evidence,
    )
    scorecard_ok, scorecard_missing = _all_have(
        scorecard_zh,
        [
            "当前机器 Gate",
            "coverage `34/34`",
            "preflight `62`",
            "manifest current",
            "minimum `180`",
            "checkpoint watchlist",
            "stage latency budget",
            "stage boundary bottleneck ledger",
            "runtime image contract",
            "rerun acceptance contract",
            "34 return evidence files",
            "final readiness audit `ready=true`",
        ],
    )
    _check(
        checks,
        "one-page scorecard gate wording",
        scorecard_ok,
        "missing=" + ", ".join(scorecard_missing),
    )
    deck_ok, deck_missing = _all_have(
        share_deck,
        [
            "15 分钟讲稿节奏",
            "被追问时的证据跳转",
            "qwen35_omni_slide_asset_map_zh_20260621.md",
            "strict_c4_latency_rtf.svg",
            "sglang_stage_latency_budget_pct.svg",
            "vllm_c8_diagnostic_qps.svg",
            "0-2 min",
            "13-15 min",
            "vLLM baseline 会不会太弱？",
            "stage 之间是不是卡住？",
            "code2wav 是不是瓶颈？",
            "能不能复现？",
            "receiver smoke",
            "extracted-only validation",
            "vLLM c=8 prebuild w4 是 offline diagnostic",
            "qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
            "vllm_online_parity_protocol.json",
            "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
        ],
    )
    share_index_deck_ok, share_index_deck_missing = _all_have(
        share_index,
        [
            "15 分钟讲稿节奏",
            "现场追问证据跳转",
            "qwen35_omni_share_deck_outline_zh_20260621.md",
            "qwen35_omni_slide_asset_map_zh_20260621.md",
        ],
    )
    slide_asset_map_ok, slide_asset_map_missing = _all_have(
        slide_asset_map_text,
        [
            "分享 Deck 图表资产映射",
            "strict_c4_latency_rtf.svg",
            "sglang_pressure_qps.svg",
            "sglang_pressure_latency.svg",
            "sglang_stage_latency_budget_pct.svg",
            "sglang_handoff_decode_ms.svg",
            "synthetic_short_long_rtf.svg",
            "vllm_c8_diagnostic_qps.svg",
            "rerun_delta_triage.json",
            "不要手工改图里的数字",
            "build_qwen35_omni_slide_asset_map",
        ],
    )
    slide_asset_json_ok = (
        bool(slide_asset_summary.get("ready"))
        and _int_value(slide_asset_summary.get("required_failures"), default=1) == 0
        and _int_value(slide_asset_summary.get("rows_total")) >= 10
        and _int_value(slide_asset_summary.get("chart_assets_total")) >= 14
    )
    _check(
        checks,
        "share deck presenter readiness gate",
        deck_ok and share_index_deck_ok and slide_asset_map_ok and slide_asset_json_ok,
        (
            "deck_missing="
            + ", ".join(deck_missing)
            + "; index_missing="
            + ", ".join(share_index_deck_missing)
            + "; slide_asset_map_missing="
            + ", ".join(slide_asset_map_missing)
            + f"; slide_asset_map={slide_asset_summary}"
        ),
    )
    final_delivery_package_ok, final_delivery_package_missing = _all_have(
        final_note,
        [
            "当前 tarball SHA-256 以同目录的",
            "qwen35_omni_share_bundle_20260621.tar.gz.sha256",
            "发送前用 `sha256sum -c` 验证",
            "final readiness audit `ready=true`，49/49 checks，0 required failures",
            "share package validation `ready=true`，17/17 checks，0 required failures",
            "extracted-only package validation `ready=true`，13/13 checks，0 required failures",
            "external standalone package validation `ready=true`，8/8 checks，0 required failures",
            "share release seal `ready=true`",
            "share_release_seal.json",
            "qwen35_omni_share_release_seal_zh_20260621.md",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
            "qwen35_omni_receiver_quickcheck.sh",
            "路径说明",
            'export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_delivery}"',
            'export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_delivery}"',
            'export STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_delivery}"',
            'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_delivery}"',
            'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_delivery}"',
            'STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_delivery}"',
            'cd "$HOST_REPO"',
            '--root "$HOST_REPO"',
            "/tmp/qwen35_omni_receiver_smoke_delivery",
            "/tmp/qwen35_omni_share_bundle_delivery",
            "/tmp/qwen35_omni_external_standalone_bundle_validation_delivery",
            "share_package_receiver_smoke_validation.json",
            "share_package_external_standalone_validation.json",
            "`share_package_validation.json` 为 `ready=true`，且 checks 为 `17/17`。",
            "解包后用随包 validator 运行 extracted-only 校验应为 `13/13`。",
            "standalone 校验应为 `8/8`。",
            "vLLM baseline 不是弱 baseline",
            "Qwen3.5-capable 镜像",
            "compile/CUDA graph",
            "prefix/chunked prefill",
            "shared-memory transfer",
            "encoder compile/batch",
            "prebuild w4",
        ],
    )
    _check(
        checks,
        "final delivery package validation wording",
        final_delivery_package_ok,
        "missing=" + ", ".join(final_delivery_package_missing),
    )
    package_boundary_ok, package_boundary_missing = _named_sources_have(
        {
            "final_note": final_note,
            "share_index": share_index,
            "final_status": final_status,
        },
        [
            "share_bundle_package_manifest.json",
            "share_package_validation.json",
            "share_package_validation_extracted.json",
            "share_package_receiver_smoke_validation.json",
            "share_package_external_standalone_validation.json",
            "share_release_seal.json",
            "不是 tarball 内成员",
            "自引用 hash",
            "PACKAGE_FILE_SHA256SUMS.txt",
            "逐文件 hash",
            "相对仓库根",
            "重复 heading",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
        ],
    )
    final_status_package_ok, final_status_package_missing = _all_have(
        final_status,
        [
            "share package validation",
            "`17/17` checks",
            "receiver smoke validation",
            "receiver_smoke_ready=true",
            "nested extracted-only `13/13`",
            "extracted-only validation",
            "external standalone validation",
            "share release seal",
            "extracted_only=true",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
            "如果接收方仓库路径不是 `/home/gangouyu/sglang-omni`，先改 `HOST_REPO`",
            'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final_status}"',
            'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final_status}"',
            'STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_final_status}"',
            'cd "$HOST_REPO"',
            "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
            '--root "$HOST_REPO"',
            "share_package_external_standalone_validation.json",
            "/tmp/qwen35_omni_external_standalone_bundle_validation_final_status",
            "/tmp/qwen35_omni_receiver_smoke_final_status",
            "/tmp/qwen35_omni_share_bundle_final_status",
            "当前 tarball digest 的权威来源",
            "tarball_sha256",
            "share_release_seal.json",
            "不内嵌 tarball digest 数值",
            "自引用漂移",
            "qwen35_omni_one_page_scorecard_zh_20260621.md",
            "qwen35_omni_receiver_package_path_map_zh_20260621.md",
            "qwen35_omni_receiver_command_card_zh_20260621.md",
            "qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
            "qwen35_omni_requirement_evidence_map_zh_20260621.md",
            "qwen35_omni_pressure_matrix_zh_20260621.md",
            "qwen35_omni_metric_source_map_zh_20260621.md",
            "qwen35_omni_stage_metric_dictionary_zh_20260621.md",
            "qwen35_omni_external_handoff_runbook_zh_20260621.md",
            "qwen35_omni_reproduction_checklist_zh_20260621.md",
            "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
            "qwen35_omni_rerun_delta_triage_zh_20260621.md",
            "qwen35_omni_defense_qna_zh_20260621.md",
            "qwen35_omni_share_deck_outline_zh_20260621.md",
            "qwen35_omni_slide_asset_map_zh_20260621.md",
            "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
        ],
    )
    share_index_package_commands_ok, share_index_package_commands_missing = _all_have(
        share_index,
        [
            "路径说明",
            "qwen35_omni_receiver_package_path_map_zh_20260621.md",
            'export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_index}"',
            'export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_index}"',
            'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_index}"',
            'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_index}"',
            'cd "$HOST_REPO"',
            '--root "$HOST_REPO"',
            '--receiver-smoke-dir "$SMOKE_DIR"',
            'cd "$EXTRACT_DIR/qwen35_omni_share_bundle_20260621"',
            '$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json',
            "build_qwen35_omni_external_standalone_bundle_validation",
            "share_package_external_standalone_validation.json",
            "/tmp/qwen35_omni_external_standalone_bundle_validation_index",
            "/tmp/qwen35_omni_receiver_smoke_index",
            "/tmp/qwen35_omni_share_bundle_index",
        ],
    )
    receiver_path_map_ok, receiver_path_map_missing = _all_have(
        receiver_path_map,
        [
            "Qwen3.5-Omni 接收方路径手册",
            "三个根目录",
            "HOST_REPO",
            "BUNDLE_ROOT",
            "只读验收路径",
            "解包后优先打开的相对路径",
            "Tarball 相邻伴随终检证据",
            "不是 `BUNDLE_ROOT` 内成员",
            "final_completion_audit.json",
            "share_release_seal.json",
            "不能把它当作 final completion audit 或 release seal 的替代",
            "什么不能从解包目录直接做",
            "复跑时的路径口径",
            "判断路径是否用错",
            "PACKAGE_FILE_SHA256SUMS.txt",
            "validate_qwen35_omni_share_package.py",
            "--extracted-only",
            "run_qwen35_omni_report_audit",
            "vLLM c=8 prebuild w4",
            "online serving parity",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
        ],
    )
    receiver_bundle_priority_section = _section_between(
        receiver_path_map,
        "## 3. 解包后优先打开的相对路径",
        "## 4. Tarball 相邻伴随终检证据",
    )
    receiver_bundle_priority_adjacent_only_hits = [
        term
        for term in [
            "final_completion_audit.json",
            "qwen35_omni_final_completion_audit_zh_20260621.md",
            "share_release_seal.json",
            "qwen35_omni_share_release_seal_zh_20260621.md",
        ]
        if term in receiver_bundle_priority_section
    ]
    receiver_bundle_priority_boundary_ok = (
        bool(receiver_bundle_priority_section)
        and not receiver_bundle_priority_adjacent_only_hits
    )
    package_builder_boundary_ok = all(
        needle in package_builder
        for needle in [
            "adjacent to the tarball",
            "intentionally not tarball",
            "members to avoid self-referential hashes",
            "If the repository is mounted elsewhere, set HOST_REPO before copying commands",
            "Quick receiver checks from the repository root when this package is current",
            "HOST_REPO",
            "SMOKE_DIR",
            "EXTRACT_DIR",
            "STANDALONE_DIR",
            "qwen35_omni_receiver_smoke_readme",
            "qwen35_omni_share_bundle_readme",
            "qwen35_omni_external_standalone_bundle_validation_readme",
            "qwen35_omni_receiver_package_path_map_zh_20260621.md",
            "qwen35_omni_receiver_command_card_zh_20260621.md",
            "qwen35_omni_receiver_quickcheck_contract_zh_20260621.md",
            "qwen35_omni_university_technical_report_zh_20260621.md",
            "results/qwen35_report_audit_20260619/university_technical_report.json",
            "qwen35_omni_runtime_image_contract_zh_20260621.md",
            "qwen35_omni_vllm_optimization_lock_zh_20260621.md",
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
            "qwen35_omni_tail_confidence_appendix_zh_20260621.md",
            "qwen35_omni_stage_latency_budget_zh_20260621.md",
            "qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            "qwen35_omni_stage_causal_graph_zh_20260621.md",
            "For single/high concurrency and short/long text regimes",
            "qwen35_omni_pressure_matrix_zh_20260621.md",
            "qwen35_omni_regime_decision_matrix_zh_20260621.md",
            "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
            "For one-page headline numbers, presentation flow, and PPT-ready figures",
            "qwen35_omni_one_page_scorecard_zh_20260621.md",
            "qwen35_omni_share_deck_outline_zh_20260621.md",
            "qwen35_omni_slide_asset_map_zh_20260621.md",
            "qwen35_omni_chart_source_consistency_zh_20260621.md",
            "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json",
            "results/qwen35_report_audit_20260619/chart_source_consistency.json",
            "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json",
            "qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
            "results/qwen35_report_audit_20260619/optimization_candidate_ledger.json",
            "For safe external wording, caveats that must travel, and defense questions",
            "qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
            "qwen35_omni_defense_qna_zh_20260621.md",
            "qwen35_omni_final_share_delivery_note_zh_20260621.md",
            "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json",
            "objective requirement crosswalk",
            "For external reproduction commands and handoff steps",
            "qwen35_omni_external_handoff_runbook_zh_20260621.md",
            "qwen35_omni_reproduction_checklist_zh_20260621.md",
            "For reviewer reruns and delta triage",
            "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
            "qwen35_omni_rerun_delta_triage_zh_20260621.md",
            "qwen35_omni_collaborator_return_check.py",
            "34 required return-evidence files",
            "27 command-matrix rows",
            "headline replacement review",
            "For command-reference hygiene",
            "results/qwen35_report_audit_20260619/command_reference_hygiene.json",
            "structured rerun command IDs",
            "cd \\\"$HOST_REPO\\\"",
            "python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check",
            "collaborator_return_check.json",
            "sha256sum -c results/qwen35_report_audit_20260619/",
            "python3 -m benchmarks.eval.validate_qwen35_omni_share_package",
            "--root \\\"$HOST_REPO\\\"",
            "--receiver-smoke-dir \\\"$SMOKE_DIR\\\"",
            "Manual extracted-root check after unpacking",
            "rm -rf \\\"$EXTRACT_DIR\\\"",
            "tar -xzf results/qwen35_report_audit_20260619/",
            "cd \\\"$EXTRACT_DIR/qwen35_omni_share_bundle_20260621\\\"",
            "python3 benchmarks/eval/validate_qwen35_omni_share_package.py",
            "--extracted-only",
            "$HOST_REPO/results/qwen35_report_audit_20260619/",
            "share_package_validation_extracted.json",
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
            "release seal ready",
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
            "/tmp/qwen35_omni_receiver_smoke_readme",
            "share_package_validation_extracted.json",
        ]
    )
    share_path_hygiene_summary = share_path_hygiene.get("summary", {})
    share_path_hygiene_ready = (
        bool(share_path_hygiene_summary.get("ready"))
        and _int_value(share_path_hygiene_summary.get("required_failures"), default=1)
        == 0
        and _int_value(share_path_hygiene_summary.get("checks_total")) >= 5
        and _int_value(share_path_hygiene_summary.get("package_offenders_total"), default=1)
        == 0
        and _int_value(share_path_hygiene_summary.get("raw_offenders_total"), default=1)
        == 0
        and _int_value(share_path_hygiene_summary.get("legacy_hits_total"), default=1)
        == 0
    )
    _check(
        checks,
        "package artifact boundary wording gate",
        package_boundary_ok
        and package_builder_boundary_ok
        and final_status_package_ok
        and share_index_package_commands_ok
        and receiver_path_map_ok
        and receiver_bundle_priority_boundary_ok
        and share_path_hygiene_ready,
        (
            "missing="
            + ", ".join(package_boundary_missing)
            + "; final_status_missing="
            + ", ".join(final_status_package_missing)
            + "; share_index_package_commands_missing="
            + ", ".join(share_index_package_commands_missing)
            + "; receiver_path_map_missing="
            + ", ".join(receiver_path_map_missing)
            + "; receiver_bundle_priority_adjacent_only_hits="
            + ", ".join(receiver_bundle_priority_adjacent_only_hits)
            + f", package_builder_boundary_ok={package_builder_boundary_ok}"
            + f", share_path_hygiene={share_path_hygiene_summary}"
        ),
    )
    share_release_summary = share_release_seal.get("summary", {})
    share_release_pending_in_audit = (
        bool(audit_summary.get("in_progress"))
        and not share_release_summary
    )
    _check(
        checks,
        "share release seal gate",
        share_release_pending_in_audit
        or (
            bool(share_release_summary.get("ready"))
            and _int_value(share_release_summary.get("required_failures"), default=1)
            == 0
            and _int_value(share_release_summary.get("checks_total")) >= 13
            and _int_value(share_release_summary.get("checks_passed"))
            == _int_value(share_release_summary.get("checks_total"), default=-1)
            and bool(share_release_summary.get("tarball_sha256"))
            and bool(share_release_summary.get("receiver_smoke_ready"))
            and not bool(share_release_summary.get("goal_complete"))
        ),
        (
            "share_release_seal="
            f"ready={share_release_summary.get('ready')}, "
            f"checks={share_release_summary.get('checks_passed')}/"
            f"{share_release_summary.get('checks_total')}, "
            f"required_failures={share_release_summary.get('required_failures')}, "
            "tarball_identity_recorded_in_adjacent_release_seal="
            f"{bool(share_release_summary.get('tarball_sha256'))}, "
            f"receiver_smoke_ready={share_release_summary.get('receiver_smoke_ready')}, "
            f"goal_complete={share_release_summary.get('goal_complete')}, "
            "completion_allowed_now="
            f"{share_release_summary.get('completion_allowed_now')}, "
            f"pending_in_full_audit={share_release_pending_in_audit}"
        ),
    )
    external_standalone_summary = external_standalone_validation.get("summary", {})
    external_quality_evidence = str(
        external_standalone_validation.get("extracted_quality_evidence") or ""
    )
    external_standalone_ready = (
        bool(external_standalone_summary.get("ready"))
        and _int_value(external_standalone_summary.get("checks_total")) >= 8
        and _int_value(external_standalone_summary.get("checks_passed"))
        == _int_value(external_standalone_summary.get("checks_total"), default=-1)
        and _int_value(
            external_standalone_summary.get("required_failures"), default=1
        )
        == 0
        and bool(external_standalone_summary.get("extracted_validation_ready"))
        and _int_value(
            external_standalone_summary.get("extracted_validation_checks")
        )
        >= 13
        and _int_value(
            external_standalone_summary.get(
                "extracted_validation_required_failures"
            ),
            default=1,
        )
        == 0
        and bool(external_standalone_summary.get("repo_independent_invocation"))
        and "report_quality_offenders=[]" in external_quality_evidence
        and "chart_quality_offenders=[]" in external_quality_evidence
    )
    external_standalone_pending_in_audit = (
        bool(audit_summary.get("in_progress"))
        and _audit_green_or_in_progress(audit_summary)
        and not external_standalone_ready
    )
    _check(
        checks,
        "external standalone bundle validation gate",
        external_standalone_pending_in_audit or external_standalone_ready,
        (
            f"external_standalone_validation={external_standalone_summary}; "
            f"quality={external_quality_evidence}; "
            f"pending_in_full_audit={external_standalone_pending_in_audit}"
        ),
    )
    quickcheck_contract_summary = receiver_quickcheck_contract.get("summary", {})
    quickcheck_contract_ok, quickcheck_contract_missing = _all_have(
        receiver_quickcheck_contract_text,
        [
            "接收方 Quickcheck Contract",
            "qwen35_omni_receiver_quickcheck.sh",
            "tarball-mode validation",
            "receiver-smoke validation",
            "extracted-only validation",
            "external standalone validation",
            "qwen35_omni_evidence_query_cards_smoke.sh",
            "qwen35_omni_evidence_query_cards_zh_20260621.md",
            "receiver command card routes to evidence-query smoke, stage dictionary, and command-id lookup",
            "stage metric dictionary carries evidence and rerun crosswalk",
            "只读验证",
            "metric_provenance_index.json",
            "stage_reproduction_drilldown.json",
            "quick_reproduction_map",
            "rerun_command_ids",
            "receiver_quickcheck_contract.json",
            "快检失败分流",
            "required_failures=0",
            "audit_run_summary.json",
            "in_progress=true",
            "recovered_from_in_progress_gates=True",
            "direct_rerun_delta_triage",
            "rows_total>=19",
            "checks_passed>=8",
            "public receiver docs preserve WER/ASR rerun path",
            "check_wer_asr_path",
            "asr_router_or_container_cache_required",
            "whisper_large_v3_local_wer.json",
            "optional warning",
            "final_completion_audit.json",
            "qwen35_omni_final_completion_audit_zh_20260621.md",
            "completion_allowed_now",
            "顶层暴露",
            "rerun_delta_triage",
            "Evidence smoke CLI options",
            "Evidence smoke explicit docs",
            '--root "$HOST_REPO" --mode host',
            '--root "$BUNDLE_ROOT" --mode portable',
        ],
    )
    quickcheck_contract_ready = (
        bool(quickcheck_contract_summary.get("ready"))
        and _int_value(quickcheck_contract_summary.get("checks_total")) >= 15
        and _int_value(quickcheck_contract_summary.get("required_failures"), default=1)
        == 0
        and _int_value(quickcheck_contract_summary.get("receiver_jsons_total")) == 4
        and _int_value(quickcheck_contract_summary.get("quickcheck_steps")) == 6
        and _int_value(quickcheck_contract_summary.get("public_docs_total")) >= 8
        and _int_value(quickcheck_contract_summary.get("wer_asr_docs_total")) >= 3
        and _int_value(quickcheck_contract_summary.get("evidence_smoke_cli_options")) >= 7
        and _int_value(
            quickcheck_contract_summary.get("evidence_smoke_explicit_docs_total")
        )
        >= 4
        and _int_value(
            quickcheck_contract_summary.get(
                "stage_dictionary_crosswalk_needles_total"
            )
        )
        >= 8
        and quickcheck_contract_ok
    )
    quickcheck_contract_pending_in_audit = (
        bool(audit_summary.get("in_progress"))
        and _audit_green_or_in_progress(audit_summary)
        and not quickcheck_contract_ready
    )
    _check(
        checks,
        "receiver quickcheck contract gate",
        quickcheck_contract_pending_in_audit or quickcheck_contract_ready,
        (
            f"receiver_quickcheck_contract={quickcheck_contract_summary}; "
            f"missing={quickcheck_contract_missing}; "
            f"pending_in_full_audit={quickcheck_contract_pending_in_audit}"
        ),
    )
    handoff_ok, handoff_missing = _all_have(
        handoff_runbook,
        [
            "preflight `62` checks, `0` required failures",
            "manifest current",
            "minimum `180`",
            "`0` missing",
            repro_manifest_gate_zh,
            "final readiness audit `ready=true`，49/49 checks，0 required failures",
            "share bundle manifest `ready=true`",
            "share package validation `ready=true`，17/17 checks，0 required failures",
            "qwen35_omni_rerun_delta_triage_zh_20260621.md",
            "rerun_delta_triage.json",
            "路径映射约定",
            "qwen35_omni_receiver_package_path_map_zh_20260621.md",
            'export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}"',
            'export EXTRACTED_BUNDLE="${EXTRACTED_BUNDLE:-${SMOKE_DIR}/qwen35_omni_share_bundle_20260621}"',
            'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}"',
            'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final}"',
            'STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_handoff}"',
            'EXTRACTED_BUNDLE="${EXTRACTED_BUNDLE:-${SMOKE_DIR}/qwen35_omni_share_bundle_20260621}"',
            'cd "$HOST_REPO"',
            "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
            '--root "$HOST_REPO"',
            'cd "$EXTRACTED_BUNDLE"',
            '$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json',
            "收包快检",
            "sha256sum",
            "receiver_smoke_ready=true",
            "tarball-mode validation 为 `17/17`",
            "nested extracted-only validation 为 `13/13`",
            "share_package_external_standalone_validation.json",
            "standalone checks 为 `8/8`",
            "qwen35_omni_receiver_quickcheck.sh",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
            "解包目录只用于阅读和接收方校验",
            "receiver quickcheck 会同时覆盖 checksum、tarball-mode validation、receiver-smoke validation、extracted-only validation 和 standalone validation",
            "vLLM c=8 `--prebuild-prompts --prebuild-workers 4`",
            "qwen35_omni_stage_latency_budget_zh_20260621.md",
            "qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
            "qwen35_omni_stage_causal_graph_zh_20260621.md",
            "qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
            "qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
            "stage_reproduction_drilldown.json",
            "stage_route_decision_matrix.json",
            "manifest-backed 原始证据",
            "Drilldown",
            "manifest.json",
            "raw path",
        ],
    )
    _check(
        checks,
        "external handoff runbook gate wording",
        handoff_ok,
        "missing=" + ", ".join(handoff_missing),
    )
    reproduction_ok, reproduction_missing = _all_have(
        reproduction_checklist,
        [
            "preflight: `62` checks, `0` required failures",
            "manifest current",
            "minimum `180`",
            "`0` missing",
            repro_manifest_gate_en,
            "final readiness: `ready=true`, `49/49` checks, `0` required failures",
            "路径映射约定",
            'export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_repro}"',
            'HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"',
            'EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_repro}"',
            'cd "$HOST_REPO"',
            '--root "$HOST_REPO"',
            'RUN_ROOT="${HOST_REPO}/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)"',
            'container 内的 `/myapp/sglang-omni`',
            '"${HOST_REPO}/results/..."',
            "share_package_validation.json",
            "share_package_receiver_smoke_validation.json",
            "share_package_validation_extracted.json",
            "share_package_external_standalone_validation.json",
            "--receiver-smoke-dir",
            "tarball-mode validation 为 `17/17`",
            "receiver_smoke_ready=true",
            "nested extracted-only validation 为 `13/13`",
            "extracted-only validation 为 `13/13`",
            "standalone validation 为 `8/8`",
            "report_quality_offenders=[]",
            "chart_quality_offenders=[]",
            "重复 heading",
            "/tmp/qwen35_omni_receiver_smoke_repro",
            "/tmp/qwen35_omni_share_bundle_repro",
            '$EXTRACT_DIR/qwen35_omni_share_bundle_20260621',
            '$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json',
            "build_qwen35_omni_external_standalone_bundle_validation",
            "/tmp/qwen35_omni_external_standalone_bundle_validation_repro",
            "qwen35_omni_external_handoff_runbook_zh_20260621.md",
            "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
            "qwen35_omni_rerun_delta_triage_zh_20260621.md",
            "qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
            "qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
            "build_qwen35_omni_rerun_delta_triage",
            "build_qwen35_omni_stage_reproduction_drilldown",
            "build_qwen35_omni_stage_route_decision_matrix",
            "rerun_delta_triage.json",
            "stage_reproduction_drilldown.json",
            "stage_route_decision_matrix.json",
        ],
    )
    _check(
        checks,
        "reproduction checklist gate wording",
        reproduction_ok,
        "missing=" + ", ".join(reproduction_missing),
    )
    collaborator_sheet_ok, collaborator_sheet_missing = _all_have(
        collaborator_sheet,
        [
            "填写说明",
            "表格中的空白单元格是给合作方复跑后填写的工作区",
            "不代表当前报告缺失证据",
            "当前 checkpoint / 当前接受门槛 / 当前 checkpoint 形态",
            "`34` return evidence files",
            "`27` command matrix rows",
            "command rows `27`",
            "复跑结果（合作者填写）",
            "结果（合作者填写）",
            "判定（PASS/FAIL/附录）",
            "三色替换判定",
            "绿：可进入替换评审",
            "黄：只确认趋势",
            "红：不得替换",
            "数字替换审批栈",
            "重新校验 tarball checksum、receiver smoke、receiver quickcheck contract、extracted-only validation 和 standalone validation",
            "vLLM c=8 online parity 已证明",
            "只能作为外部附录",
            "差异定位矩阵",
            "复跑出现差异时先填这一节",
            "不要先改主报告文案",
            "silent-replacement 风险",
            "regeneration/full audit",
            "skip-first",
            "是否可替换 headline",
            "sglang_talker_to_code2wav_healthy=false",
            "vLLM c=8 prebuild w4 显著改善",
            "share_package_receiver_smoke_validation.json",
            "receiver_quickcheck_contract.json",
            "receiver quickcheck contract",
            "runtime_comparison_contract.json",
            "runtime_image_contract.json",
            "sglang_optimization_lock.json",
            "vllm_optimization_lock.json",
            "vllm_online_parity_protocol.json",
            "tail_confidence_appendix.json",
            "stage_latency_budget.json",
            "stage_boundary_bottleneck_ledger.json",
            "stage_reproduction_drilldown.json",
            "stage_route_decision_matrix.json",
            "share_package_validation.json",
            "share_package_validation_extracted.json",
            "share_package_external_standalone_validation.json",
        ],
    )
    rerun_delta_triage_ok, rerun_delta_triage_missing = _all_have(
        rerun_delta_triage_text,
        [
            "外部复跑差异定位矩阵",
            "复跑症状",
            "优先定位 stage / boundary",
            "第一证据",
            "裁决边界",
            "数字替换范围",
            "online_parity_proven=false",
            "package 或 audit 不绿时，不讨论性能结论",
            "code2wav decode compute",
            "vLLM parity scope",
            "stream handoff boundary",
            "collaborator rerun validation sheet",
        ],
    )
    rerun_delta_json_ok = (
        bool(rerun_delta_triage_summary.get("ready"))
        and _int_value(rerun_delta_triage_summary.get("required_failures"), default=1)
        == 0
        and _int_value(rerun_delta_triage_summary.get("rows_total")) >= 19
        and _int_value(rerun_delta_triage_summary.get("checks_total")) >= 8
    )
    _check(
        checks,
        "collaborator rerun worksheet and delta triage gate",
        collaborator_sheet_ok and rerun_delta_triage_ok and rerun_delta_json_ok,
        (
            "sheet_missing="
            + ", ".join(collaborator_sheet_missing)
            + "; triage_missing="
            + ", ".join(rerun_delta_triage_missing)
            + f"; rerun_delta_triage={rerun_delta_triage_summary}"
        ),
    )
    main_report_ok, main_report_missing = _all_have(
        main_report,
        [
            "Handoff Readiness",
            "Chinese university technical report",
            "qwen35_omni_university_technical_report_zh_20260621.md",
            "Final readiness audit",
            "Current Bottleneck Map",
            "Stage Interaction Matrix",
            "Remaining Work Before Final 2026-06-21 Version",
            "official SeedTTS",
            "online serving parity",
            "SGLang Optimization Lock",
            "vLLM Optimization Lock",
            "vLLM Online Parity Protocol",
            "Final Checkpoint Watchlist",
            "Stage Latency Budget",
            "Stage Boundary Bottleneck Ledger",
            "Stage Reproduction Drilldown",
            "Stage Route Decision Matrix",
            "Runtime Image Contract",
            "Rerun Acceptance Contract",
        ],
    )
    _check(
        checks,
        "main report final-share wording gate",
        main_report_ok,
        "missing=" + ", ".join(main_report_missing),
    )

    required_failures = [
        check
        for check in checks
        if check.required and check.status != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "entry_scope_redline": entry_scope_redline,
            "entry_scope_redline_docs": [
                "cover_note",
                "final_note",
                "share_index",
                "collaboration_brief",
                "final_status",
            ],
            "entry_scope_redline_missing": entry_scope_missing,
            "hard_gates": {
                "claims": "17/17",
                "coverage": "34/34",
                "preflight_checks": "62",
                "manifest_records": f">=180 / current {manifest_records_current}",
                "stage_causal_graph": "7/7 causal edges / manifest-backed raw drilldown",
                "headline_scorecard": "9/9",
                "regime_decision_matrix": "17 rows / 9 checks / scenario decisions",
                "university_technical_report": (
                    "11 sections / 16 checks / Chinese report with runtime fairness, capacity matrices, and pressure-stage heatmap"
                ),
                "serving_capacity_matrix": (
                    "7 rows / 10 checks / c8 serving edge and c16 saturation guard"
                ),
                "share_consistency_guard": (
                    "17 checks / stale gate tokens / legacy evidence-query output scan / university-review gate route / explicit manifest minimum gates / preflight alias guard / evidence-query host-portable routes / current identity-field agreement / embedded identity guarded"
                ),
                "runtime_comparison_contract": "9/9 / warmed c4 only / c8 diagnostic",
                "metric_provenance_index": ">=150 rows / raw artifacts / command refs",
                "stage_reproduction_drilldown": (
                    "52 rows / 17/17 checks / 5 quick reproduction routes / jq queries / command refs"
                ),
                "stage_route_decision_matrix": "11 routes / 52 stage rows / route decisions",
                "claim_metric_crosswalk": "10 claims / >=60 metric rows",
                "objective_requirement_crosswalk": (
                    ">=11 original requirement rows / >=85 metric rows / "
                    ">=8 optimization candidates"
                ),
                "optimization_candidate_ledger": (
                    "8 candidates / current best / anti-recipes / vLLM diagnostics"
                ),
                "chart_pack": "7 csv / 7 svg",
                "chart_source_consistency": "14 chart assets byte-exact / 8 checks",
                "acceptance_matrix": "17/17",
                "confidence_ledger": "12/12",
                "caveat_adjudication_matrix": (
                    "12 caveats / forbidden claims / replacement triggers"
                ),
                "objective_completion_audit": "17 rows / 0 required failures",
                "repro_command_manifest": (
                    f"{_int_value(repro_summary.get('commands_total'))} commands / "
                    f"{_int_value(repro_summary.get('phases_total'))} phases"
                ),
                "rerun_time_budget": (
                    "9 budget rows / 6 timed rows / command-backed wall-time lower bound"
                ),
                "command_reference_hygiene": "structured command refs resolve / critical commands documented",
                "runtime_image_contract": "12/12",
                "rerun_acceptance_contract": "17/17 / 34 return evidence / 27 command matrix rows / silent-replacement guard",
                "sglang_optimization_lock": "26/26",
                "vllm_optimization_lock": "22/22",
                "vllm_online_parity_protocol": "18/18",
                "final_checkpoint_watchlist": "24/24",
                "tail_confidence_appendix": (
                    "18 rows / 13 checks / strict c4 tail + bootstrap uncertainty"
                ),
                "pressure_stage_heatmap": "15 rows / SGLang + synthetic + vLLM stage heatmap",
                "length_regime_coverage": (
                    "7 rows / Video-AMME target length + synthetic short/long shape guard"
                ),
                "stage_latency_budget": "12/12",
                "stage_boundary_bottleneck_ledger": "12/12 / 11 pressure transitions",
                "share_bundle_manifest": "ready",
                "share_package_index_structure": "1..33 / single section-2 heading",
                "unit_test_smoke": (
                    "235 focused container tests passed / broader suite needs optional deps"
                ),
                "share_deck_outline": "15-min script / challenge evidence jumps / slide asset map",
                "final_delivery_package_validation": (
                    "17/17 tarball / 13/13 extracted / 8/8 standalone"
                ),
                "external_standalone_bundle_validation": (
                    "clean /tmp extraction / bundled validator / 8/8"
                ),
                "share_release_seal": (
                    "adjacent tarball seal / checksum / validation bundle"
                ),
                "receiver_quickcheck_contract": (
                    f"{_int_value(quickcheck_contract_summary.get('checks_passed'))}/"
                    f"{_int_value(quickcheck_contract_summary.get('checks_total'))} / "
                    "six-step wrapper / 4 receiver JSONs / "
                    "evidence-query CLI/docs / self-reference recovery docs / "
                    "WER/ASR path guard / stage dictionary crosswalk / "
                    "final completion route"
                ),
                "public_doc_hash_guard": "no bare 64-hex hashes",
                "public_doc_quality_guard": (
                    "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift"
                ),
                "defense_qna_readiness": (
                    "13 prepared questions / 17 Q&A checks / evidence cards / "
                    "stage quick-route card / machine claim+question matrix / "
                    "WER-ASR path guard / full-traffic scope guard"
                ),
                "package_artifact_boundary": "adjacent validation artifacts not tar members",
                "share_path_hygiene": "0 package/raw offenders / generated outputs documented",
                "external_handoff_runbook": "current gate wording",
                "reproduction_checklist": "current gate wording",
                "collaborator_rerun_sheet": (
                    "explicit fill-in template / 34 return evidence / "
                    "27 command rows"
                ),
                "rerun_delta_triage": "19 symptoms / stage routes / protocol drift and silent-replacement boundary",
                "slide_asset_map": "10 slide rows / chart assets / no hand-edited numbers",
            },
        },
        "checks": [check.to_dict() for check in checks],
        "send_decision": (
            "ready_to_share_with_documented_caveats"
            if not required_failures
            else "do_not_share_until_required_failures_are_fixed"
        ),
        "required_failures": [check.to_dict() for check in required_failures],
        "caveats_that_must_travel": [
            "Official SeedTTS full-set is not a headline benchmark in this package.",
            "vLLM c=8 prebuild w4 is an optimized offline diagnostic, not online serving parity.",
            "ci-50/stress/synthetic evidence must not be extrapolated to full online traffic without same-scope reruns.",
            "Host-side Whisper large-v3 cache warning is optional unless WER is recomputed on the host.",
            "c=16 is a saturation boundary, not the recommended serving point.",
        ],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Final Readiness Audit\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")
    print(f"| Send decision | {payload['send_decision']} |")
    print("\n| Status | Required | Check | Evidence |")
    print("| --- | --- | --- | --- |")
    for check in payload["checks"]:
        evidence = str(check["evidence"]).replace("|", "\\|")
        required = "yes" if check["required"] else "no"
        print(
            f"| {check['status']} | {required} | {check['name']} | {evidence} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final share-readiness gates for the Qwen3.5-Omni report."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_readiness(root)
    output = args.json_output
    if not output.is_absolute():
        output = root / output
    _save_json(payload, output)
    print_markdown(payload)
    print(
        "Final readiness audit written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
