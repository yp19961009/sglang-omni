# SPDX-License-Identifier: Apache-2.0
"""Build a requirement-coverage matrix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_REPORT = Path("benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md")
DEFAULT_SHARE_ZH = Path("benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md")
DEFAULT_BRIEF_ZH = Path("benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md")
DEFAULT_DECK_ZH = Path("benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md")
DEFAULT_EVIDENCE_ZH = Path("benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md")
DEFAULT_PRESSURE_ZH = Path("benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md")
DEFAULT_SOURCE_MAP_ZH = Path("benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md")
DEFAULT_STAGE_DICT_ZH = Path("benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md")
DEFAULT_QNA_ZH = Path("benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md")
DEFAULT_PLAYBOOK_ZH = Path("benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md")
DEFAULT_REPRO_ZH = Path("benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md")
DEFAULT_HANDOFF_ZH = Path("benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md")
DEFAULT_RERUN_SHEET_ZH = Path("benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md")
DEFAULT_DELIVERY_NOTE_ZH = Path("benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md")
DEFAULT_SCORECARD_ZH = Path("benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md")
DEFAULT_SGLANG_LOCK_ZH = Path("benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md")
DEFAULT_VLLM_LOCK_ZH = Path("benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md")
DEFAULT_VLLM_ONLINE_PROTOCOL_ZH = Path(
    "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md"
)


@dataclass(frozen=True)
class CoverageRow:
    requirement: str
    status: str
    evidence: str
    reproducibility: str

    def to_dict(self) -> dict[str, str]:
        return {
            "requirement": self.requirement,
            "status": self.status,
            "evidence": self.evidence,
            "reproducibility": self.reproducibility,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json(path)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _claim_passed(claims: dict[str, Any], name: str) -> bool:
    for check in claims.get("checks", []):
        if check.get("name") == name:
            return bool(check.get("passed"))
    return False


def _table_rows(tables: dict[str, Any], table_name: str) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get(table_name, [])
    return rows if isinstance(rows, list) else []


def _report_has(report_text: str, *needles: str) -> bool:
    return all(needle in report_text for needle in needles)


def _status(condition: bool) -> str:
    return "PASS" if condition else "MISSING"


def build_rows(root: Path) -> list[CoverageRow]:
    audit_dir = root / DEFAULT_AUDIT_DIR
    report_path = root / DEFAULT_REPORT
    share_zh_path = root / DEFAULT_SHARE_ZH
    brief_zh_path = root / DEFAULT_BRIEF_ZH
    deck_zh_path = root / DEFAULT_DECK_ZH
    evidence_zh_path = root / DEFAULT_EVIDENCE_ZH
    pressure_zh_path = root / DEFAULT_PRESSURE_ZH
    source_map_zh_path = root / DEFAULT_SOURCE_MAP_ZH
    stage_dict_zh_path = root / DEFAULT_STAGE_DICT_ZH
    qna_zh_path = root / DEFAULT_QNA_ZH
    playbook_zh_path = root / DEFAULT_PLAYBOOK_ZH
    repro_zh_path = root / DEFAULT_REPRO_ZH
    handoff_zh_path = root / DEFAULT_HANDOFF_ZH
    rerun_sheet_zh_path = root / DEFAULT_RERUN_SHEET_ZH
    delivery_note_zh_path = root / DEFAULT_DELIVERY_NOTE_ZH
    scorecard_zh_path = root / DEFAULT_SCORECARD_ZH
    sglang_lock_zh_path = root / DEFAULT_SGLANG_LOCK_ZH
    vllm_lock_zh_path = root / DEFAULT_VLLM_LOCK_ZH
    vllm_online_protocol_zh_path = root / DEFAULT_VLLM_ONLINE_PROTOCOL_ZH
    tables = _load_json(audit_dir / "tables_summary.json")
    claims = _load_json(audit_dir / "claims_verification.json")
    preflight = _load_json(audit_dir / "preflight_repro.json")
    manifest = _load_json(audit_dir / "manifest.json")
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    stage_interactions = _load_json_optional(
        audit_dir / "stage_interaction_summary.json"
    )
    scorecard = _load_json_optional(audit_dir / "headline_scorecard.json")
    chart_pack = _load_json_optional(
        audit_dir / "share_charts/chart_pack_manifest.json"
    )
    chart_source_consistency = _load_json_optional(
        audit_dir / "chart_source_consistency.json"
    )
    acceptance_matrix = _load_json_optional(audit_dir / "acceptance_matrix.json")
    confidence_ledger = _load_json_optional(audit_dir / "confidence_ledger.json")
    repro_commands = _load_json_optional(audit_dir / "repro_command_manifest.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online_protocol = _load_json_optional(
        audit_dir / "vllm_online_parity_protocol.json"
    )
    seedtts_meta = audit_dir / "videoamme_seedtts_meta.lst"
    seedtts_summary = _load_json_optional(
        audit_dir / "videoamme_seedtts_meta_summary.json"
    )
    report_text = report_path.read_text(encoding="utf-8")
    share_zh_text = share_zh_path.read_text(encoding="utf-8") if share_zh_path.is_file() else ""
    brief_zh_text = brief_zh_path.read_text(encoding="utf-8") if brief_zh_path.is_file() else ""
    deck_zh_text = deck_zh_path.read_text(encoding="utf-8") if deck_zh_path.is_file() else ""
    evidence_zh_text = evidence_zh_path.read_text(encoding="utf-8") if evidence_zh_path.is_file() else ""
    pressure_zh_text = pressure_zh_path.read_text(encoding="utf-8") if pressure_zh_path.is_file() else ""
    source_map_zh_text = source_map_zh_path.read_text(encoding="utf-8") if source_map_zh_path.is_file() else ""
    stage_dict_zh_text = stage_dict_zh_path.read_text(encoding="utf-8") if stage_dict_zh_path.is_file() else ""
    qna_zh_text = qna_zh_path.read_text(encoding="utf-8") if qna_zh_path.is_file() else ""
    playbook_zh_text = playbook_zh_path.read_text(encoding="utf-8") if playbook_zh_path.is_file() else ""
    repro_zh_text = repro_zh_path.read_text(encoding="utf-8") if repro_zh_path.is_file() else ""
    handoff_zh_text = handoff_zh_path.read_text(encoding="utf-8") if handoff_zh_path.is_file() else ""
    rerun_sheet_zh_text = rerun_sheet_zh_path.read_text(encoding="utf-8") if rerun_sheet_zh_path.is_file() else ""
    delivery_note_zh_text = delivery_note_zh_path.read_text(encoding="utf-8") if delivery_note_zh_path.is_file() else ""
    scorecard_zh_text = scorecard_zh_path.read_text(encoding="utf-8") if scorecard_zh_path.is_file() else ""
    sglang_lock_zh_text = sglang_lock_zh_path.read_text(encoding="utf-8") if sglang_lock_zh_path.is_file() else ""
    vllm_lock_zh_text = vllm_lock_zh_path.read_text(encoding="utf-8") if vllm_lock_zh_path.is_file() else ""
    vllm_online_protocol_zh_text = (
        vllm_online_protocol_zh_path.read_text(encoding="utf-8")
        if vllm_online_protocol_zh_path.is_file()
        else ""
    )

    stress_rows = _table_rows(tables, "sglang_stress")
    stress_concurrency = {int(row["concurrency"]) for row in stress_rows}
    synthetic_rows = _table_rows(tables, "synthetic_speech")
    synthetic_pairs = {
        (str(row.get("scenario")), int(row.get("concurrency")))
        for row in synthetic_rows
    }
    short_text_words = [
        float(row.get("target_words") or 0)
        for row in synthetic_rows
        if row.get("scenario") == "short"
    ]
    long_text_words = [
        float(row.get("target_words") or 0)
        for row in synthetic_rows
        if row.get("scenario") == "long"
    ]
    stage_rows = _table_rows(tables, "sglang_stage_breakdown")
    synthetic_stage_rows = _table_rows(tables, "synthetic_stage_breakdown")
    vllm_stage_rows = _table_rows(tables, "vllm_log_stage_signals")
    vllm_admission_rows = _table_rows(tables, "vllm_admission_diagnosis")
    stage_interaction_summary = stage_interactions.get("summary", {})
    scorecard_summary = scorecard.get("summary", {})
    chart_summary = chart_pack.get("summary", {})
    chart_source_summary = chart_source_consistency.get("summary", {})
    acceptance_summary = acceptance_matrix.get("summary", {})
    confidence_summary = confidence_ledger.get("summary", {})
    repro_commands_summary = repro_commands.get("summary", {})
    sglang_lock_summary = sglang_lock.get("summary", {})
    vllm_lock_summary = vllm_lock.get("summary", {})
    vllm_online_protocol_summary = vllm_online_protocol.get("summary", {})

    rows = [
        CoverageRow(
            "SGLang vs vLLM optimized baseline, at least comparable",
            _status(
                _claim_passed(
                    claims,
                    "SGLang warmed c4 beats vLLM warmed c4 latency/RTF",
                )
                and _claim_passed(
                    claims,
                    "SGLang warmed c4 preserves accuracy/WER vs vLLM",
                )
            ),
            "Warmed c=4 SGLang beats vLLM on latency mean/p95 and RTF mean/p95 while preserving accuracy/WER.",
            "claims_verification.json; 完整报告 section 1/5",
        ),
        CoverageRow(
            "Optimization switches and base/optimized boundary",
            _status(
                _report_has(
                    report_text,
                    "### 4.1 Optimization Evidence Ledger",
                    "--thinker-cuda-graph on",
                    "PREPROCESSING_MAX_CONCURRENCY=1",
                    "FULL_AND_PIECEWISE",
                    "--prebuild-prompts --prebuild-workers 4",
                    "not an online serving parity claim",
                )
                and _report_has(brief_zh_text, "优化开关证据 Ledger", "不是弱 baseline")
                and _report_has(repro_zh_text, "优化开关", "base/optimized 边界")
                and bool(sglang_lock_summary.get("ready"))
                and bool(vllm_lock_summary.get("ready"))
                and _report_has(
                    sglang_lock_zh_text,
                    "SGLang 优化锁定矩阵",
                    "NO_CODE2WAV_TORCH_COMPILE=0",
                    "--thinker-max-running-requests 8",
                    "PREPROCESSING_MAX_CONCURRENCY=1",
                )
                and _report_has(
                    vllm_lock_zh_text,
                    "vLLM 优化锁定矩阵",
                    "VLLM_ENABLE_TORCH_COMPILE=True",
                    "FULL_AND_PIECEWISE",
                    "--prebuild-prompts --prebuild-workers 4",
                )
            ),
            "SGLang/vLLM optimization switches, rejected anti-recipes, comparison boundaries, and both SGLang/vLLM optimization locks are explicitly documented.",
            "完整报告 section 4.1; Chinese brief section 4.2; reproduction checklist section 7; sglang_optimization_lock.json; vllm_optimization_lock.json",
        ),
        CoverageRow(
            "Collaborator-facing Chinese brief",
            _status(
                brief_zh_path.is_file()
                and _report_has(
                    brief_zh_text,
                    "结论先行",
                    "复现入口",
                    "当前不能过度声明的部分",
                    "manifest current",
                    "minimum 180",
                    "34 required",
                )
                and _report_has(
                    report_text,
                    "Chinese collaborator brief",
                    "qwen35_omni_collaboration_brief_zh_20260621.md",
                )
            ),
            "Chinese handoff brief exists and is linked from the full report handoff table.",
            "qwen35_omni_collaboration_brief_zh_20260621.md; section 1.1",
        ),
        CoverageRow(
            "Share-package index and defense route",
            _status(
                share_zh_path.is_file()
                and _report_has(
                    share_zh_text,
                    "五分钟阅读顺序",
                    "当前可对外讲的结论",
                    "不应过度声明的边界",
                    "机器证据入口",
                    "答辩问题索引",
                    "当前接受门槛",
                    "final_completion_audit.json",
                    "qwen35_omni_final_completion_audit_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_share_package_index_zh_20260621.md",
                )
                and _report_has(brief_zh_text, "分享包索引")
                and _report_has(repro_zh_text, "分享包索引")
            ),
            "Chinese package index tells collaborators what to read first, which evidence to inspect, and which claims are safe.",
            "qwen35_omni_share_package_index_zh_20260621.md; 完整报告 section 1.1",
        ),
        CoverageRow(
            "Collaborator-facing share deck outline",
            _status(
                deck_zh_path.is_file()
                and _report_has(
                    deck_zh_text,
                    "Qwen3.5-Omni 性能分析分享 Deck 提纲",
                    "Headline",
                    "Stage breakdown",
                    "复现路径",
                    "后续优化路线",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_share_deck_outline_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_share_deck_outline_zh_20260621.md",
                )
            ),
            "Chinese deck outline maps the headline, stage evidence, caveats, and reproduction gates into a 15-25 minute sharing flow.",
            "qwen35_omni_share_deck_outline_zh_20260621.md; share index section 1",
        ),
        CoverageRow(
            "Original-objective evidence map",
            _status(
                evidence_zh_path.is_file()
                and _report_has(
                    evidence_zh_text,
                    "Qwen3.5-Omni 原始需求-证据映射表",
                    "单并发和高并发",
                    "短/长文本输入",
                    "Stage breakdown",
                    "stage 连接",
                    "vLLM 边界",
                    "复现路径",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_requirement_evidence_map_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_requirement_evidence_map_zh_20260621.md",
                )
            ),
            "Chinese evidence map ties each original user requirement to the strongest local evidence, confidence boundary, and reproduction entry point.",
            "qwen35_omni_requirement_evidence_map_zh_20260621.md; share index section 1",
        ),
        CoverageRow(
            "Human-readable pressure-condition matrix",
            _status(
                pressure_zh_path.is_file()
                and _report_has(
                    pressure_zh_text,
                    "Qwen3.5-Omni 压力条件总表",
                    "SGLang Video-AMME c=1/2/4/8/16",
                    "短/长文本输入",
                    "vLLM 对比和 c=8 诊断",
                    "preproc=2",
                    "coverage `34/34`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_pressure_matrix_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_pressure_matrix_zh_20260621.md",
                )
            ),
            "Chinese pressure matrix summarizes every measured pressure condition, recommendation status, bottleneck, and evidence path in one human-readable table.",
            "qwen35_omni_pressure_matrix_zh_20260621.md; acceptance_matrix.json",
        ),
        CoverageRow(
            "Human-readable metric source map",
            _status(
                source_map_zh_path.is_file()
                and _report_has(
                    source_map_zh_text,
                    "Qwen3.5-Omni 数字来源索引",
                    "Headline 数字",
                    "SGLang Pressure 数字",
                    "Stage 连接数字",
                    "vLLM c=8 诊断数字",
                    "preproc=2",
                    "coverage `34/34`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_metric_source_map_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_metric_source_map_zh_20260621.md",
                )
            ),
            "Chinese metric source map ties headline, pressure, stage, vLLM diagnostic, and anti-recipe numbers to machine evidence and regeneration commands.",
            "qwen35_omni_metric_source_map_zh_20260621.md; headline_scorecard.json; tables_summary.json",
        ),
        CoverageRow(
            "Human-readable stage metric dictionary",
            _status(
                stage_dict_zh_path.is_file()
                and _report_has(
                    stage_dict_zh_text,
                    "Qwen3.5-Omni Stage 指标字典",
                    "stage_input_received->stage_complete",
                    "preprocess_start->preprocess_end",
                    "code2wav_window_collect",
                    "talker_to_code2wav_hop",
                    "batch_admission_span_ms",
                    "coverage `34/34`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_stage_metric_dictionary_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_stage_metric_dictionary_zh_20260621.md",
                )
            ),
            "Chinese stage metric dictionary defines lifecycle, compute, handoff, collect-wait, and vLLM admission metrics so stage breakdowns are interpreted correctly.",
            "qwen35_omni_stage_metric_dictionary_zh_20260621.md; stage_interaction_summary.json; tables_summary.json",
        ),
        CoverageRow(
            "Collaborator-facing defense Q&A",
            _status(
                qna_zh_path.is_file()
                and _report_has(
                    qna_zh_text,
                    "Qwen3.5-Omni 性能报告答辩 Q&A",
                    "SGLang-Omni 是否至少和 vLLM 相当",
                    "vLLM baseline 是不是故意设弱了",
                    "Stage 之间有没有卡住",
                    "如何现场复现或验收",
                    "coverage `34/34`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_defense_qna_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_defense_qna_zh_20260621.md",
                )
            ),
            "Chinese defense Q&A gives ready-to-say answers, unsafe-wording boundaries, and evidence links for common collaborator questions.",
            "qwen35_omni_defense_qna_zh_20260621.md; share index section 1/6",
        ),
        CoverageRow(
            "Optimization playbook and safe tuning protocol",
            _status(
                playbook_zh_path.is_file()
                and _report_has(
                    playbook_zh_text,
                    "Qwen3.5-Omni 性能优化 Playbook",
                    "当前推荐 recipe",
                    "不要先优化 code2wav",
                    "preproc=2",
                    "c=8",
                    "回滚规则",
                    "coverage `34/34`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_optimization_playbook_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_optimization_playbook_zh_20260621.md",
                )
            ),
            "Chinese optimization playbook maps measured bottlenecks to safe knobs, experiment order, acceptance gates, and rollback rules.",
            "qwen35_omni_optimization_playbook_zh_20260621.md; 完整报告 section 12/14",
        ),
        CoverageRow(
            "Reproducibility environment snapshot",
            _status(
                bool(environment)
                and bool(environment.get("gpu", {}).get("ok"))
                and all(
                    image.get("ok")
                    for image in environment.get("docker_images", {}).values()
                )
                and _report_has(report_text, "environment_snapshot.json")
            ),
            "Environment snapshot records GPU inventory, Docker image IDs, git state, paths, and audit summaries.",
            "environment_snapshot.json; section 1.1/3/11.8",
        ),
        CoverageRow(
            "Step-by-step Chinese reproduction checklist",
            _status(
                repro_zh_path.is_file()
                and _report_has(
                    repro_zh_text,
                    "复现 SGLang-Omni 主压测",
                    "复现 vLLM baseline",
                    "重新生成报告表格",
                    "最终接受标准",
                    "manifest current",
                    "minimum `180`",
                )
                and _report_has(
                    report_text,
                    "Chinese reproduction checklist",
                    "qwen35_omni_reproduction_checklist_zh_20260621.md",
                )
                and _report_has(brief_zh_text, "复现清单")
            ),
            "Chinese checklist gives ordered SGLang/vLLM rerun commands, expected shapes, and final acceptance gates.",
            "qwen35_omni_reproduction_checklist_zh_20260621.md; section 1.1/6",
        ),
        CoverageRow(
            "External handoff runbook for reviewers",
            _status(
                handoff_zh_path.is_file()
                and _report_has(
                    handoff_zh_text,
                    "Qwen3.5-Omni 外部复现 Handoff Runbook",
                    "Host 侧第一条命令",
                    "SGLang 复现主线",
                    "vLLM 复现主线",
                    "Stage 读表规则",
                    "替换报告数字的红线",
                    "coverage `34/34`",
                    "preflight `62` checks",
                    "manifest current",
                    "minimum `180`",
                    "final_completion_audit.json",
                    "qwen35_omni_final_completion_audit_zh_20260621.md",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_external_handoff_runbook_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_external_handoff_runbook_zh_20260621.md",
                )
                and _report_has(
                    repro_zh_text,
                    "qwen35_omni_external_handoff_runbook_zh_20260621.md",
                )
            ),
            "Chinese external handoff runbook gives reviewers the shortest audited reproduction path, rerun order, stage-reading rules, and replacement gates.",
            "qwen35_omni_external_handoff_runbook_zh_20260621.md; run_qwen35_omni_report_audit.py",
        ),
        CoverageRow(
            "Collaborator rerun validation sheet",
            _status(
                rerun_sheet_zh_path.is_file()
                and _report_has(
                    rerun_sheet_zh_text,
                    "Qwen3.5-Omni 合作方复跑验收表",
                    "复跑结果登记",
                    "环境差异",
                    "SGLang 复跑验收",
                    "vLLM 复跑验收",
                    "Stage 连接复核",
                    "是否可以替换报告数字",
                    "复跑后应回传的文件",
                    "coverage `34/34`",
                    "preflight `62` checks",
                    "manifest current",
                    "minimum `180`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
                )
                and _report_has(
                    handoff_zh_text,
                    "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
                )
            ),
            "Chinese collaborator rerun validation sheet gives reviewers a structured pass/fail worksheet for environment deltas, SGLang/vLLM reruns, stage flags, replacement rules, and return artifacts.",
            "qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md; audit_run_summary.json",
        ),
        CoverageRow(
            "Final share delivery note",
            _status(
                delivery_note_zh_path.is_file()
                and _report_has(
                    delivery_note_zh_text,
                    "Qwen3.5-Omni 最终分享交付说明",
                    "建议发送文件",
                    "当前版本 Gate",
                    "对外邮件/消息模板",
                    "对外可引用结论",
                    "接收方第一步",
                    "不建议发送的说法",
                    "最终交付自检",
                    "coverage `34/34`",
                    "preflight `62` checks",
                    "manifest current",
                    "minimum `180`",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_final_share_delivery_note_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_final_share_delivery_note_zh_20260621.md",
                )
                and _report_has(
                    brief_zh_text,
                    "qwen35_omni_final_share_delivery_note_zh_20260621.md",
                )
            ),
            "Chinese final share delivery note tells the sender which files and machine evidence to include, how to introduce the package, which claims are safe, and which caveats must travel with the share.",
            "qwen35_omni_final_share_delivery_note_zh_20260621.md; share package index",
        ),
        CoverageRow(
            "One-page core-number scorecard",
            _status(
                scorecard_zh_path.is_file()
                and _report_has(
                    scorecard_zh_text,
                    "Qwen3.5-Omni 一页式核心数字 Scorecard",
                    "Headline",
                    "SGLang 压力条件",
                    "短/长文本输入",
                    "Stage 结论",
                    "vLLM c=8 诊断",
                    "当前机器 Gate",
                    "coverage `34/34`",
                    "preflight `62` checks",
                    "manifest current",
                    "minimum `180`",
                    "online serving parity",
                )
                and _report_has(
                    share_zh_text,
                    "qwen35_omni_one_page_scorecard_zh_20260621.md",
                )
                and _report_has(
                    report_text,
                    "qwen35_omni_one_page_scorecard_zh_20260621.md",
                )
                and _report_has(
                    delivery_note_zh_text,
                    "qwen35_omni_one_page_scorecard_zh_20260621.md",
                )
            ),
            "Chinese one-page scorecard condenses the cross-runtime headline, SGLang pressure sweep, short/long text-to-speech guardrails, stage health, vLLM c8 diagnostic, and current gates for quick sharing.",
            "qwen35_omni_one_page_scorecard_zh_20260621.md; headline_scorecard.json; tables_summary.json",
        ),
        CoverageRow(
            "Slide-ready chart and CSV pack",
            _status(
                bool(chart_summary.get("ready"))
                and int(chart_summary.get("csv_files") or 0) >= 7
                and int(chart_summary.get("svg_files") or 0) >= 7
                and int(chart_summary.get("generated_files") or 0) >= 14
                and bool(chart_source_summary.get("ready"))
                and int(chart_source_summary.get("byte_exact_files") or 0) >= 14
                and int(chart_source_summary.get("required_failures") or 0) == 0
                and _report_has(
                    share_zh_text,
                    "share_charts/chart_pack_manifest.json",
                    "chart_source_consistency.json",
                    "share_charts/",
                    "build_qwen35_omni_share_charts",
                )
                and _report_has(
                    delivery_note_zh_text,
                    "share_charts/chart_pack_manifest.json",
                    "chart_source_consistency.json",
                    "SVG/CSV",
                )
                and _report_has(
                    report_text,
                    "share_charts/chart_pack_manifest.json",
                    "chart_source_consistency.json",
                    "build_qwen35_omni_share_charts",
                )
            ),
            "Audited SVG/CSV assets provide slide-ready charts and spreadsheet rows derived from the report JSONs, with byte-exact source consistency checks to prevent hand-edited numbers.",
            "share_charts/chart_pack_manifest.json; chart_source_consistency.json; build_qwen35_omni_share_charts.py",
        ),
        CoverageRow(
            "Machine-readable reproduction command manifest",
            _status(
                bool(repro_commands_summary.get("ready"))
                and bool(repro_commands_summary.get("required_command_ids_present"))
                and int(repro_commands_summary.get("commands_total") or 0) >= 60
                and int(
                    repro_commands_summary.get("expected_gates", {})
                    .get("coverage", {})
                    .get("total")
                    or 0
                )
                >= 34
                and int(
                    repro_commands_summary.get("expected_gates", {})
                    .get("preflight", {})
                    .get("total_checks")
                    or 0
                )
                >= 62
                and int(
                    repro_commands_summary.get("expected_gates", {})
                    .get("manifest", {})
                    .get("min_total_records")
                    or repro_commands_summary.get("expected_gates", {})
                    .get("manifest", {})
                    .get("total_records")
                    or 0
                )
                >= 180
                and _report_has(report_text, "repro_command_manifest.json")
                and _report_has(share_zh_text, "repro_command_manifest.json")
                and _report_has(repro_zh_text, "repro_command_manifest.json")
            ),
            "Audited command manifest records the exact full-audit, SGLang, vLLM, table, chart, preflight, coverage, and manifest commands needed for collaborator reruns.",
            "repro_command_manifest.json; build_qwen35_omni_repro_command_manifest.py",
        ),
        CoverageRow(
            "Single-concurrency and high-concurrency SGLang stress",
            _status({1, 2, 4, 8, 16}.issubset(stress_concurrency)),
            "SGLang Video-AMME stress rows cover c=1/2/4/8/16.",
            "tables_summary.json table sglang_stress; section 7",
        ),
        CoverageRow(
            "Short/long text-input and speech-output workloads",
            _status(
                {
                    ("short", 1),
                    ("short", 4),
                    ("short", 8),
                    ("long", 1),
                    ("long", 4),
                    ("long", 8),
                }.issubset(synthetic_pairs)
                and min(short_text_words or [999.0]) <= 20.0
                and max(long_text_words or [0.0]) >= 100.0
                and _report_has(
                    report_text,
                    "74-character / 12-word",
                    "944-character / 139-word",
                    "audited short/long text",
                )
                and _claim_passed(claims, "long synthetic speech remains faster than real time")
            ),
            "Synthetic text-to-speech rows cover short text (12 words) and long text (139 words) at c=1/4/8; long output remains faster than real time.",
            "tables_summary.json table synthetic_speech; section 8",
        ),
        CoverageRow(
            "Spoken-reference/SeedTTS-style smoke path without external cache",
            _status(
                seedtts_meta.is_file()
                and int(seedtts_summary.get("rows") or 0) >= 50
                and _report_has(
                    report_text,
                    "build_videoamme_seedtts_meta",
                    "SeedTTS-compatible",
                )
            ),
            "Local Video-AMME spoken-question audio is exported as a SeedTTS-compatible meta.lst for voice-clone/TTS smoke runs.",
            "videoamme_seedtts_meta.lst; videoamme_seedtts_meta_summary.json; section 8.3/11.6",
        ),
        CoverageRow(
            "Stage-level breakdown for each measured regime",
            _status(
                len(stage_rows) >= 5
                and len(synthetic_stage_rows) >= 6
                and len(vllm_stage_rows) >= 5
            ),
            "SGLang Video-AMME, synthetic speech, and vLLM log-derived stage tables, including vLLM c8 prebuild w1/w4, are present.",
            "tables_summary.json stage tables; sections 5/7/8",
        ),
        CoverageRow(
            "Machine-readable stage interaction summary",
            _status(
                int(stage_interaction_summary.get("total_interactions") or 0) >= 30
                and bool(
                    stage_interaction_summary.get(
                        "sglang_talker_to_code2wav_healthy"
                    )
                )
                and bool(
                    stage_interaction_summary.get(
                        "sglang_code2wav_decode_not_bottleneck"
                    )
                )
                and bool(
                    stage_interaction_summary.get(
                        "vllm_original_c8_prompt_feed_limited"
                    )
                )
                and bool(
                    stage_interaction_summary.get(
                        "preprocessing_parallelism_regresses"
                    )
                )
                and _report_has(report_text, "stage_interaction_summary.json")
            ),
            "Stage boundary health, queue/admission effects, vLLM prompt-feed limits, and preprocessing contention are summarized in JSON.",
            "stage_interaction_summary.json; 完整报告 section 12.1",
        ),
        CoverageRow(
            "Machine-readable headline scorecard",
            _status(
                bool(scorecard_summary.get("ready"))
                and int(scorecard_summary.get("checks_passed") or 0)
                == int(scorecard_summary.get("checks_total") or -1)
                and int(scorecard_summary.get("checks_total") or 0) >= 9
                and _report_has(report_text, "headline_scorecard.json")
                and _report_has(share_zh_text, "headline_scorecard.json")
            ),
            "Headline c=4 comparison, c=8 peak, long speech, vLLM c8 diagnostic, and stage flags are summarized in JSON.",
            "headline_scorecard.json; 完整报告 section 1.1; share index section 4",
        ),
        CoverageRow(
            "Machine-readable per-regime acceptance matrix",
            _status(
                bool(acceptance_summary.get("ready"))
                and int(acceptance_summary.get("rows_passed") or 0)
                == int(acceptance_summary.get("rows_total") or -1)
                and int(acceptance_summary.get("rows_total") or 0) >= 17
                and int(acceptance_summary.get("rows_failed", 1)) == 0
                and _report_has(report_text, "acceptance_matrix.json")
                and _report_has(share_zh_text, "acceptance_matrix.json")
            ),
            "Single/high concurrency, short/long text-to-speech, vLLM diagnostics, and anti-recipes have per-regime pass/fail evidence.",
            "acceptance_matrix.json; 完整报告 section 1.1; share index section 4",
        ),
        CoverageRow(
            "Machine-readable confidence ledger",
            _status(
                bool(confidence_summary.get("ready"))
                and int(confidence_summary.get("entries_passed") or 0)
                == int(confidence_summary.get("entries_total") or -1)
                and int(confidence_summary.get("entries_total") or 0) >= 12
                and int(confidence_summary.get("entries_failed", 1)) == 0
                and int(confidence_summary.get("high_confidence_claims") or 0) >= 9
                and int(confidence_summary.get("medium_confidence_boundaries") or 0) >= 3
                and int(confidence_summary.get("unsupported_claims", 1)) == 0
                and _report_has(report_text, "confidence_ledger.json")
                and _report_has(share_zh_text, "confidence_ledger.json")
            ),
            "Safe high-confidence claims, medium-confidence boundaries, and unsupported-claim guardrails are summarized in JSON.",
            "confidence_ledger.json; 完整报告 section 1.1/14; share index section 4",
        ),
        CoverageRow(
            "Stage-to-stage connection and bottleneck attribution",
            _status(
                _claim_passed(
                    claims,
                    "code2wav decode and talker->code2wav hop are not bottlenecks",
                )
                and _claim_passed(
                    claims,
                    "vLLM log-derived stage boundaries are not hidden bottlenecks",
                )
            ),
            "SGLang talker->code2wav hop and vLLM thinker/talker/code2wav boundary checks pass.",
            "claims_verification.json; sections 7.1/12",
        ),
        CoverageRow(
            "High-concurrency bottleneck and optimization direction",
            _status(
                _claim_passed(claims, "SGLang stress c8 is throughput peak")
                and _claim_passed(
                    claims,
                    "preprocessing lifecycle growth is queue/admission dominated",
                )
            ),
            "c=8 is SGLang throughput peak; c=16 regresses through preprocessing admission/queueing.",
            "claims_verification.json; 完整报告 section 12",
        ),
        CoverageRow(
            "Negative optimization evidence for naive preprocessing parallelism",
            _status(
                _claim_passed(claims, "naive preprocessing parallelism regresses or fails")
                and len(_table_rows(tables, "preprocessing_concurrency")) >= 2
            ),
            "preproc=2 regresses throughput; preproc=4 fails/OOMs.",
            "tables_summary.json preprocessing_concurrency; section 9",
        ),
        CoverageRow(
            "vLLM optimized image and c8 caveat are explicit",
            _status(
                len(vllm_admission_rows) >= 4
                and _claim_passed(
                    claims,
                    "vLLM offline admission diagnosis classifies c4/c8 as prompt-feed limited",
                )
                and _claim_passed(
                    claims,
                    "vLLM c8 prebuilt prompts with 4 workers improve runner-side wall clock",
                )
                and _report_has(
                    report_text,
                    "Qwen3.5-capable vLLM image",
                    "prompt_feed_limited",
                    "--prebuild-prompts",
                    "vLLM-c8-prebuild-w4",
                    "vLLM Online Parity Protocol",
                )
                and bool(vllm_online_protocol_summary.get("ready"))
                and not bool(vllm_online_protocol_summary.get("online_parity_proven"))
                and _report_has(
                    vllm_online_protocol_zh_text,
                    "vLLM c=8 Online Parity 升级协议",
                    "online_parity_proven",
                    "do_not_promote_c8_parity_without_online_ingress_artifacts",
                )
            ),
            "vLLM optimized Docker image is documented; offline c8 is diagnosed as prompt-feed limited; prebuilt-prompt c8 w4 evidence and the online-parity upgrade protocol are documented.",
            "vllm_admission_diagnosis.json; vllm_online_parity_protocol.json; sections 3/5",
        ),
        CoverageRow(
            "Audio/text consistency and WER stability",
            _status(
                _claim_passed(claims, "SGLang stress WER remains stable")
                and _claim_passed(
                    claims,
                    "SGLang warmed c4 preserves accuracy/WER vs vLLM",
                )
            ),
            "Offline Whisper large-v3 WER remains stable across SGLang stress rows and better than vLLM c4.",
            "claims_verification.json; sections 1/7",
        ),
        CoverageRow(
            "Cold-start/warmup caveat is documented",
            _status(
                _report_has(
                    report_text,
                    "## 10. Cold Start And Warmup",
                    "compile/capture requests",
                )
            ),
            "Report separates cold/full-run and warmed steady-state numbers.",
            "section 10",
        ),
        CoverageRow(
            "Reproduction commands and audit/preflight path",
            _status(
                bool(preflight.get("summary", {}).get("ready"))
                and int(manifest.get("summary", {}).get("missing_records", 1)) == 0
                and _report_has(report_text, "run_qwen35_omni_report_audit")
            ),
            "Preflight is ready, manifest has no missing records, and one-command audit is documented.",
            "preflight_repro.json; manifest.json; section 11.8",
        ),
    ]
    return rows


def print_markdown(rows: list[CoverageRow]) -> None:
    print("| Status | Requirement | Evidence | Reproducibility |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row.status} | {row.requirement} | "
            f"{row.evidence} | {row.reproducibility} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize user-objective coverage for the Qwen3.5-Omni report."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args.root.resolve())
    missing = [row for row in rows if row.status != "PASS"]
    payload = {
        "root": str(args.root.resolve()),
        "summary": {
            "total_requirements": len(rows),
            "passed": len(rows) - len(missing),
            "missing": len(missing),
            "complete": not missing,
        },
        "rows": [row.to_dict() for row in rows],
    }

    print_markdown(rows)
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = args.root.resolve() / output
        _save_json(payload, output)
    if args.strict and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
