# SPDX-License-Identifier: Apache-2.0
"""Build a machine-readable defense claim matrix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "defense_claim_matrix.json"
QNA_REPORT = Path("benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md")


@dataclass(frozen=True)
class ClaimDefenseRow:
    claim_id: str
    claim: str
    allowed_wording: str
    machine_evidence: list[str]
    rerun_command_ids: list[str]
    rerun_notes: list[str]
    failure_decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "allowed_wording": self.allowed_wording,
            "machine_evidence": self.machine_evidence,
            "rerun_command_ids": self.rerun_command_ids,
            "rerun_notes": self.rerun_notes,
            "failure_decision": self.failure_decision,
        }


@dataclass(frozen=True)
class QnaQuestionRow:
    question_id: str
    question: str
    linked_claim_ids: list[str]
    qna_section: str
    evidence_jump: str
    answer_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "linked_claim_ids": self.linked_claim_ids,
            "qna_section": self.qna_section,
            "evidence_jump": self.evidence_jump,
            "answer_boundary": self.answer_boundary,
        }


MATRIX_ROWS = [
    ClaimDefenseRow(
        claim_id="sglang_warmed_c4_beats_optimized_vllm",
        claim="SGLang warmed c=4 优于优化版 vLLM",
        allowed_wording=(
            "当前 8x H20、Video-AMME ci-50、warmed c=4 严格对比中，"
            "SGLang latency/RTF 更好且质量不退化。"
        ),
        machine_evidence=[
            "results/qwen35_report_audit_20260619/claims_verification.json",
            "results/qwen35_report_audit_20260619/headline_scorecard.json",
            "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
        ],
        rerun_command_ids=[
            "run_full_audit",
            "vllm_c1_original",
            "vllm_c4_original",
            "vllm_c8_original",
            "build_runtime_comparison_contract",
        ],
        rerun_notes=["strict c=4 artifacts are refreshed by full audit"],
        failure_decision=(
            "claims 或 headline 失败时不得沿用 headline，进入 rerun acceptance 替换评审。"
        ),
    ),
    ClaimDefenseRow(
        claim_id="vllm_baseline_is_optimized",
        claim="vLLM baseline 不是弱 baseline",
        allowed_wording=(
            "vLLM 使用 Qwen3.5-capable 镜像和 compile/CUDA graph、"
            "prefix/chunked prefill、shared-memory transfer、encoder compile/batch 等优化证据。"
        ),
        machine_evidence=[
            "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
            "results/qwen35_report_audit_20260619/runtime_image_contract.json",
            "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json",
        ],
        rerun_command_ids=[
            "build_vllm_optimization_lock",
            "build_runtime_image_contract",
            "summarize_vllm_log_stages",
            "vllm_c1_original",
            "vllm_c8_original",
        ],
        rerun_notes=[],
        failure_decision=(
            "image/optimization lock 失败时只能说现有 vLLM 证据不可复核，不能说 baseline 公平。"
        ),
    ),
    ClaimDefenseRow(
        claim_id="sglang_c8_current_high_concurrency_peak",
        claim="SGLang c=8 是当前高并发峰值",
        allowed_wording="c=8 是当前 recipe 的吞吐峰值，c=16 是压力边界，不是推荐默认点。",
        machine_evidence=[
            "results/qwen35_report_audit_20260619/acceptance_matrix.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
        ],
        rerun_command_ids=[
            "sglang_videoamme_stress",
            "build_acceptance_matrix",
            "build_stage_latency_budget",
        ],
        rerun_notes=[],
        failure_decision=(
            "c=8 不再为峰值时先按 rerun delta triage 定位 admission/queue，不直接改主报告数字。"
        ),
    ),
    ClaimDefenseRow(
        claim_id="short_and_long_tts_are_covered",
        claim="short/long text-to-speech 已覆盖",
        allowed_wording=(
            "short 74 chars / 12 words，long 944 chars / 139 words，"
            "c=1/4/8 均覆盖，long c=8 仍快于实时。"
        ),
        machine_evidence=[
            "results/qwen35_report_audit_20260619/length_regime_coverage.json",
            "results/qwen35_report_audit_20260619/tables_summary.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/headline_scorecard.json",
        ],
        rerun_command_ids=[
            "sglang_synthetic_text_to_speech",
            "build_report_tables",
            "build_stage_latency_budget",
        ],
        rerun_notes=[],
        failure_decision="输入形状或 long c=8 RTF 失败时，相关长短文结论不得替换或外推。",
    ),
    ClaimDefenseRow(
        claim_id="stage_handoff_is_not_stalled",
        claim="stage handoff 没有卡住",
        allowed_wording="talker 到 code2wav 的 stream hop p95 约 15-24ms，当前不是主瓶颈。",
        machine_evidence=[
            "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
        ],
        rerun_command_ids=[
            "build_stage_interactions",
            "build_stage_boundary_bottleneck_ledger",
            "build_stage_causal_graph",
        ],
        rerun_notes=[],
        failure_decision="handoff health 失败时，不能继续说 stage 连接健康，先补 profile drilldown。",
    ),
    ClaimDefenseRow(
        claim_id="code2wav_decode_not_current_compute_bottleneck",
        claim="code2wav decode 不是当前 compute bottleneck",
        allowed_wording="decode 平均约 14-17ms/window，collect wait 更多是在等 talker chunk cadence。",
        machine_evidence=[
            "results/qwen35_report_audit_20260619/claims_verification.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
        ],
        rerun_command_ids=[
            "sglang_videoamme_stress",
            "sglang_synthetic_text_to_speech",
            "build_stage_boundary_bottleneck_ledger",
        ],
        rerun_notes=[],
        failure_decision="decode 成为主项时，当前 code2wav-not-bottleneck 结论必须撤回或重写。",
    ),
    ClaimDefenseRow(
        claim_id="raising_preprocessing_concurrency_is_negative_recipe",
        claim="朴素提高 preprocessing 并发是负优化",
        allowed_wording=(
            "preproc=2 回退，preproc=4 失败；当前应先管 admission、placement "
            "和 shared-resource contention。"
        ),
        machine_evidence=[
            "results/qwen35_report_audit_20260619/acceptance_matrix.json",
            "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
            "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
        ],
        rerun_command_ids=[
            "build_sglang_optimization_lock",
            "build_acceptance_matrix",
            "build_stage_boundary_bottleneck_ledger",
        ],
        rerun_notes=["anti-recipe artifacts are linked from optimization lock"],
        failure_decision=(
            "新候选 recipe 必须补齐 c=4/c=8/c=16、WER、profile 和稳定性证据后再评审。"
        ),
    ),
    ClaimDefenseRow(
        claim_id="vllm_c8_prebuild_w4_is_offline_diagnostic",
        claim="vLLM c=8 prebuild w4 只是 offline diagnostic",
        allowed_wording="prebuild w4 改善 runner prompt build/feed，但没有证明 online serving parity。",
        machine_evidence=[
            "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
            "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
            "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
        ],
        rerun_command_ids=[
            "vllm_c8_prebuild_w4",
            "build_vllm_online_parity_protocol",
            "build_runtime_comparison_contract",
        ],
        rerun_notes=[],
        failure_decision="online_parity_proven=false 时不得把 c=8 prebuild 写成 online parity。",
    ),
    ClaimDefenseRow(
        claim_id="wer_quality_not_traded_for_speed",
        claim="WER/quality 没有为性能让步",
        allowed_wording="SGLang stress WER 稳定，strict c=4 WER/accuracy 不劣于 vLLM。",
        machine_evidence=[
            "results/qwen35_report_audit_20260619/claims_verification.json",
            "results/qwen35_report_audit_20260619/headline_scorecard.json",
            "results/qwen35_report_audit_20260619/acceptance_matrix.json",
        ],
        rerun_command_ids=[
            "sglang_recompute_wer",
            "verify_report_claims",
            "build_headline_scorecard",
        ],
        rerun_notes=[],
        failure_decision="WER 或 ASR 路径不一致时，不得只替换 latency/RTF headline。",
    ),
    ClaimDefenseRow(
        claim_id="package_share_ready_with_boundaries",
        claim="当前包可分享但仍有边界",
        allowed_wording=(
            "share-ready 是带 caveat 的阶段稿；更大数据/真实流量外推、"
            "SeedTTS full-set 和 vLLM c=8 online parity 不能越界。"
        ),
        machine_evidence=[
            "results/qwen35_report_audit_20260619/final_readiness_audit.json",
            "results/qwen35_report_audit_20260619/confidence_ledger.json",
            "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
        ],
        rerun_command_ids=[
            "run_full_audit",
            "validate_share_bundle_package",
            "validate_share_bundle_receiver_smoke",
        ],
        rerun_notes=[],
        failure_decision="任一 package/readiness gate 失败时先修包，不讨论性能结论。",
    ),
]


QNA_QUESTION_ROWS = [
    QnaQuestionRow(
        question_id="q01_sglang_vs_vllm",
        question="SGLang-Omni 是否至少和 vLLM 相当？",
        linked_claim_ids=[
            "sglang_warmed_c4_beats_optimized_vllm",
            "wer_quality_not_traded_for_speed",
        ],
        qna_section="## 1. SGLang-Omni 是否至少和 vLLM 相当？",
        evidence_jump="jq '.strict_c4_comparison' results/qwen35_report_audit_20260619/headline_scorecard.json",
        answer_boundary="只有 warmed c=4 是严格跨 runtime headline 口径。",
    ),
    QnaQuestionRow(
        question_id="q02_vllm_baseline_strength",
        question="vLLM baseline 是不是故意设弱了？",
        linked_claim_ids=[
            "vllm_baseline_is_optimized",
            "vllm_c8_prebuild_w4_is_offline_diagnostic",
        ],
        qna_section="## 2. vLLM baseline 是不是故意设弱了？",
        evidence_jump="jq '.summary' results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
        answer_boundary="不能声明已经穷尽所有可能的 vLLM 优化。",
    ),
    QnaQuestionRow(
        question_id="q03_headline_scope",
        question="为什么 headline 选 warmed c=4？",
        linked_claim_ids=["sglang_warmed_c4_beats_optimized_vllm"],
        qna_section="## 3. 为什么 headline 选 warmed c=4？",
        evidence_jump="jq '.strict_c4_comparison' results/qwen35_report_audit_20260619/headline_scorecard.json",
        answer_boundary="不要把 strict warmed c4 和 SGLang stress sweep c4 混用。",
    ),
    QnaQuestionRow(
        question_id="q04_single_concurrency",
        question="单并发结果说明什么？",
        linked_claim_ids=[
            "sglang_c8_current_high_concurrency_peak",
            "stage_handoff_is_not_stalled",
        ],
        qna_section="## 4. 单并发结果说明什么？",
        evidence_jump="jq '.rows[] | select(.pressure == \"c=1\")' results/qwen35_report_audit_20260619/acceptance_matrix.json",
        answer_boundary="c=1 用于隔离基础路径和 tail 形态，不是吞吐 headline。",
    ),
    QnaQuestionRow(
        question_id="q05_high_concurrency_peak",
        question="为什么 c=8 是推荐高并发点？",
        linked_claim_ids=[
            "sglang_c8_current_high_concurrency_peak",
            "stage_handoff_is_not_stalled",
        ],
        qna_section="## 5. 为什么 c=8 是推荐高并发点？",
        evidence_jump="jq '.sglang_stress.throughput_peak' results/qwen35_report_audit_20260619/headline_scorecard.json",
        answer_boundary="c16 是饱和边界证据，不是推荐服务点。",
    ),
    QnaQuestionRow(
        question_id="q06_short_long_tts",
        question="长短文本/语音输出覆盖了吗？",
        linked_claim_ids=["short_and_long_tts_are_covered"],
        qna_section="## 6. 长短文本/语音输出覆盖了吗？",
        evidence_jump="jq '.summary' results/qwen35_report_audit_20260619/length_regime_coverage.json",
        answer_boundary="Synthetic long speech 是 guardrail，不是官方 SeedTTS full-set headline 证据。",
    ),
    QnaQuestionRow(
        question_id="q07_stage_bottlenecks",
        question="每个 stage 的瓶颈在哪里？",
        linked_claim_ids=[
            "stage_handoff_is_not_stalled",
            "code2wav_decode_not_current_compute_bottleneck",
            "sglang_c8_current_high_concurrency_peak",
        ],
        qna_section="## 7. 每个 stage 的瓶颈在哪里？",
        evidence_jump="jq '.summary' results/qwen35_report_audit_20260619/stage_interaction_summary.json",
        answer_boundary="必须区分 admission/queue、Talker cadence、handoff、collect wait 和 decode compute。",
    ),
    QnaQuestionRow(
        question_id="q08_stage_handoff",
        question="Stage 之间有没有卡住？",
        linked_claim_ids=[
            "stage_handoff_is_not_stalled",
            "code2wav_decode_not_current_compute_bottleneck",
        ],
        qna_section="## 8. Stage 之间有没有卡住？",
        evidence_jump="rg -n '原始证据 Drilldown' benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
        answer_boundary="不能声明 stage boundary 永远不会成为瓶颈。",
    ),
    QnaQuestionRow(
        question_id="q09_preprocessing_concurrency",
        question="为什么不直接提高 preprocessing 并发？",
        linked_claim_ids=["raising_preprocessing_concurrency_is_negative_recipe"],
        qna_section="## 9. 为什么不直接提高 preprocessing 并发？",
        evidence_jump="jq '.rows[] | select(.serving_status | contains(\"anti_recipe\"))' results/qwen35_report_audit_20260619/acceptance_matrix.json",
        answer_boundary="新 recipe 必须补跑 c4/c8/c16、WER、profile 和 readiness gates。",
    ),
    QnaQuestionRow(
        question_id="q10_vllm_c8_online_parity",
        question="vLLM c=8 为什么不能直接作为 online parity？",
        linked_claim_ids=[
            "vllm_c8_prebuild_w4_is_offline_diagnostic",
            "vllm_baseline_is_optimized",
        ],
        qna_section="## 10. vLLM c=8 为什么不能直接作为 online parity？",
        evidence_jump="jq '.rows[] | select(.label == \"vLLM-c8\")' results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
        answer_boundary="online_parity_proven=false 时禁止 c8 online parity 口径。",
    ),
    QnaQuestionRow(
        question_id="q11_quality_wer",
        question="WER 和语音一致性有没有退化？",
        linked_claim_ids=["wer_quality_not_traded_for_speed"],
        qna_section="## 11. WER 和语音一致性有没有退化？",
        evidence_jump="jq '.checks' results/qwen35_report_audit_20260619/claims_verification.json",
        answer_boundary="没有同口径 WER 或 ASR 证据时，不得只替换 latency/RTF。",
    ),
    QnaQuestionRow(
        question_id="q12_extrapolation_scope",
        question="现在的结论有多大外推范围？",
        linked_claim_ids=["package_share_ready_with_boundaries"],
        qna_section="## 12. 现在的结论有多大外推范围？",
        evidence_jump="jq '.summary' results/qwen35_report_audit_20260619/confidence_ledger.json",
        answer_boundary="不能声明覆盖所有 workload、全量线上流量或官方 full-set。",
    ),
    QnaQuestionRow(
        question_id="q13_reproduction_acceptance",
        question="如何现场复现或验收？",
        linked_claim_ids=[
            "package_share_ready_with_boundaries",
            "sglang_warmed_c4_beats_optimized_vllm",
        ],
        qna_section="## 13. 如何现场复现或验收？",
        evidence_jump="python3 -m benchmarks.eval.run_qwen35_omni_report_audit --root /home/gangouyu/sglang-omni --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json",
        answer_boundary="任何 headline 替换都必须通过 rerun acceptance 和 rerun delta triage。",
    ),
]


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


def _qna_claim_matrix_rows(qna_text: str) -> tuple[int, list[str]]:
    if "## 14. 主张-证据-复跑-裁决矩阵" not in qna_text:
        return 0, ["missing section 14"]
    section = qna_text.split("## 14. 主张-证据-复跑-裁决矩阵", 1)[1]
    lines = [line for line in section.splitlines() if line.startswith("| ")]
    malformed = [
        line for line in lines if len(re.findall(r"(?<!\\)\|", line)) != 6
    ]
    data_rows = [
        line
        for line in lines
        if not line.startswith("| 主张 |") and not line.startswith("| --- |")
    ]
    return len(data_rows), malformed


def _command_ids(repro_manifest: dict[str, Any]) -> set[str]:
    commands = repro_manifest.get("commands", [])
    return {
        str(command.get("id"))
        for command in commands
        if isinstance(command, dict) and command.get("id")
    }


def build_defense_claim_matrix(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    qna_text = _read_text_optional(root / QNA_REPORT)
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")
    repro_command_ids = _command_ids(repro_manifest)

    rows = [row.to_dict() for row in MATRIX_ROWS]
    row_ids = [row["claim_id"] for row in rows]
    question_rows = [row.to_dict() for row in QNA_QUESTION_ROWS]
    question_ids = [row["question_id"] for row in question_rows]
    claim_ids = set(row_ids)
    missing_question_sections = [
        row.qna_section for row in QNA_QUESTION_ROWS if row.qna_section not in qna_text
    ]
    missing_question_text = [
        row.question for row in QNA_QUESTION_ROWS if row.question not in qna_text
    ]
    question_claim_id_gaps = sorted(
        {
            claim_id
            for row in QNA_QUESTION_ROWS
            for claim_id in row.linked_claim_ids
            if claim_id not in claim_ids
        }
    )
    questions_without_claims = [
        row.question_id for row in QNA_QUESTION_ROWS if not row.linked_claim_ids
    ]
    questions_without_evidence_jump = [
        row.question_id for row in QNA_QUESTION_ROWS if not row.evidence_jump.strip()
    ]
    questions_without_boundary = [
        row.question_id for row in QNA_QUESTION_ROWS if not row.answer_boundary.strip()
    ]
    qna_rows_total, malformed_qna_rows = _qna_claim_matrix_rows(qna_text)

    missing_claims_in_qna = [
        row.claim for row in MATRIX_ROWS if row.claim not in qna_text
    ]
    missing_evidence_files = sorted(
        {
            rel_path
            for row in MATRIX_ROWS
            for rel_path in row.machine_evidence
            if not (root / rel_path).exists()
        }
    )
    missing_command_ids = sorted(
        {
            command_id
            for row in MATRIX_ROWS
            for command_id in row.rerun_command_ids
            if command_id not in repro_command_ids
        }
    )
    failure_decisions_missing = [
        row.claim_id for row in MATRIX_ROWS if not row.failure_decision.strip()
    ]
    allowed_wording_missing = [
        row.claim_id for row in MATRIX_ROWS if not row.allowed_wording.strip()
    ]
    qna_wer_asr_path_needles = [
        "/root/.cache/whisper/large-v3.pt",
        "optional warning",
        "ASR router",
        "serving benchmark 失败",
        "receiver_quickcheck_contract.json",
        "check_wer_asr_path",
        "public receiver docs preserve WER/ASR rerun path",
    ]
    qna_wer_asr_path_missing = [
        needle for needle in qna_wer_asr_path_needles if needle not in qna_text
    ]
    qna_full_traffic_scope_needles = [
        "更大 Video-AMME、真实线上流量、官方 SeedTTS full-set",
        "更大数据和真实流量是下一阶段外推验证",
        "ci-50 等价于所有线上流量",
        "更大数据/真实流量外推",
    ]
    qna_full_traffic_scope_missing = [
        needle for needle in qna_full_traffic_scope_needles if needle not in qna_text
    ]

    checks = {
        "rows_total": len(rows) >= 10,
        "row_ids_unique": len(row_ids) == len(set(row_ids)),
        "question_rows_total": len(question_rows) >= 13,
        "question_row_ids_unique": len(question_ids) == len(set(question_ids)),
        "qna_questions_covered": not missing_question_sections
        and not missing_question_text,
        "question_claim_ids_resolve": not question_claim_id_gaps
        and not questions_without_claims,
        "question_evidence_jumps_present": not questions_without_evidence_jump,
        "question_boundaries_present": not questions_without_boundary,
        "qna_section14_rows_match": qna_rows_total >= len(rows)
        and not malformed_qna_rows,
        "qna_claims_covered": not missing_claims_in_qna,
        "evidence_files_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
        "allowed_wording_present": not allowed_wording_missing,
        "failure_decisions_present": not failure_decisions_missing,
        "qna_wer_asr_path_guard": not qna_wer_asr_path_missing,
        "qna_full_traffic_scope_guard": not qna_full_traffic_scope_missing,
    }
    expected_matrix_gate = f"checks `{len(checks) + 1}/{len(checks) + 1}`"
    checks["qna_current_matrix_gate_marker"] = expected_matrix_gate in qna_text
    required_failures = [name for name, ok in checks.items() if not ok]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "question_rows_total": len(question_rows),
            "question_claim_refs_total": sum(
                len(row["linked_claim_ids"]) for row in question_rows
            ),
            "qna_section14_rows_total": qna_rows_total,
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "qna_questions_covered": checks["qna_questions_covered"],
            "qna_claims_covered": checks["qna_claims_covered"],
            "failure_decisions_total": len(rows) - len(failure_decisions_missing),
            "qna_wer_asr_path_guard": checks["qna_wer_asr_path_guard"],
            "qna_full_traffic_scope_guard": checks[
                "qna_full_traffic_scope_guard"
            ],
            "qna_current_matrix_gate_marker": checks[
                "qna_current_matrix_gate_marker"
            ],
            "qna_expected_matrix_gate": expected_matrix_gate,
        },
        "checks": checks,
        "diagnostics": {
            "missing_claims_in_qna": missing_claims_in_qna,
            "missing_question_sections": missing_question_sections,
            "missing_question_text": missing_question_text,
            "question_claim_id_gaps": question_claim_id_gaps,
            "questions_without_claims": questions_without_claims,
            "questions_without_evidence_jump": questions_without_evidence_jump,
            "questions_without_boundary": questions_without_boundary,
            "malformed_qna_rows": malformed_qna_rows,
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "allowed_wording_missing": allowed_wording_missing,
            "failure_decisions_missing": failure_decisions_missing,
            "qna_wer_asr_path_missing": qna_wer_asr_path_missing,
            "qna_full_traffic_scope_missing": qna_full_traffic_scope_missing,
            "qna_expected_matrix_gate": expected_matrix_gate,
        },
        "qna_question_rows": question_rows,
        "rows": rows,
        "sources": {
            "qna_report": str(root / QNA_REPORT),
            "repro_command_manifest": str(audit_dir / "repro_command_manifest.json"),
        },
    }


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Defense Claim Matrix")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- rows: `{summary['rows_total']}`")
    print(f"- question rows: `{summary['question_rows_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni defense-claim matrix JSON."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.json_output
    if not output.is_absolute():
        output = root / output

    payload = build_defense_claim_matrix(root)
    _save_json(payload, output)
    _print_markdown(payload)

    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
