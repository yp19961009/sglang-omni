# SPDX-License-Identifier: Apache-2.0
"""Build a concise final-status Markdown summary for the Qwen3.5-Omni share package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path("benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else {}


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _yes(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _int_or(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary_table(rows: list[tuple[str, Any, str]]) -> list[str]:
    lines = ["| Gate | Status | Evidence |", "| --- | --- | --- |"]
    for gate, status, evidence in rows:
        lines.append(f"| {gate} | `{status}` | {evidence} |")
    return lines


def _check_evidence(payload: dict[str, Any], name: str) -> str:
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        return ""
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            evidence = check.get("evidence")
            return evidence if isinstance(evidence, str) else ""
    return ""


def _quality_evidence_summary(evidence: str, source_json: str) -> str:
    if not evidence:
        return f"quality evidence missing; see `{source_json}`"
    report_clean = "report_quality_offenders=[]" in evidence
    chart_clean = "chart_quality_offenders=[]" in evidence
    if report_clean and chart_clean:
        return "`report_quality_offenders=[]`, `chart_quality_offenders=[]`"
    return (
        "quality offenders are non-empty; inspect "
        f"`{source_json}` before sharing"
    )


def _boundary_rows(objective: dict[str, Any]) -> list[dict[str, Any]]:
    rows = objective.get("boundary_items", [])
    return rows if isinstance(rows, list) else []


def build_markdown(root: Path) -> str:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    audit = _load_json_optional(audit_dir / "audit_run_summary.json")
    objective = _load_json_optional(audit_dir / "objective_completion_audit.json")
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    final_completion = _load_json_optional(audit_dir / "final_completion_audit.json")
    package = _load_json_optional(audit_dir / "share_bundle_package_manifest.json")
    bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    share_consistency_guard = _load_json_optional(
        audit_dir / "share_consistency_guard.json"
    )
    share_release_seal = _load_json_optional(audit_dir / "share_release_seal.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    slide_asset_map = _load_json_optional(audit_dir / "slide_asset_map.json")
    vllm_online_protocol = _load_json_optional(
        audit_dir / "vllm_online_parity_protocol.json"
    )
    runtime_image_contract = _load_json_optional(
        audit_dir / "runtime_image_contract.json"
    )
    runtime_comparison_contract = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    rerun_acceptance_contract = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    rerun_time_budget = _load_json_optional(audit_dir / "rerun_time_budget.json")
    rerun_delta_triage = _load_json_optional(audit_dir / "rerun_delta_triage.json")
    checkpoint_watchlist = _load_json_optional(
        audit_dir / "final_checkpoint_watchlist.json"
    )
    stage_latency_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    length_regime_coverage = _load_json_optional(
        audit_dir / "length_regime_coverage.json"
    )
    stage_boundary_ledger = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    stage_causal_graph = _load_json_optional(audit_dir / "stage_causal_graph.json")
    share_package_validation = _load_json_optional(
        audit_dir / "share_package_validation.json"
    )
    share_package_receiver_smoke_validation = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    share_package_validation_extracted = _load_json_optional(
        audit_dir / "share_package_validation_extracted.json"
    )
    external_standalone_validation = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    receiver_quickcheck_contract = _load_json_optional(
        audit_dir / "receiver_quickcheck_contract.json"
    )
    collaborator_return_check = _load_json_optional(
        audit_dir / "collaborator_return_check.json"
    )
    university_review_packet = _load_json_optional(
        audit_dir / "university_review_packet.json"
    )

    audit_ok = audit.get("ok")
    objective_summary = objective.get("summary", {})
    final_summary = final.get("summary", {})
    final_completion_summary = final_completion.get("summary", {})
    package_summary = package.get("summary", {})
    bundle_summary = bundle.get("summary", {})
    share_consistency_summary = share_consistency_guard.get("summary", {})
    share_release_seal_summary = share_release_seal.get("summary", {})
    manifest_summary = manifest.get("summary", {})
    repro_summary = repro.get("summary", {})
    slide_asset_summary = slide_asset_map.get("summary", {})
    protocol_summary = vllm_online_protocol.get("summary", {})
    runtime_comparison_summary = runtime_comparison_contract.get("summary", {})
    runtime_image_summary = runtime_image_contract.get("summary", {})
    rerun_acceptance_summary = rerun_acceptance_contract.get("summary", {})
    rerun_time_budget_summary = rerun_time_budget.get("summary", {})
    rerun_delta_summary = rerun_delta_triage.get("summary", {})
    checkpoint_summary = checkpoint_watchlist.get("summary", {})
    stage_budget_summary = stage_latency_budget.get("summary", {})
    length_regime_summary = length_regime_coverage.get("summary", {})
    stage_ledger_summary = stage_boundary_ledger.get("summary", {})
    stage_causal_summary = stage_causal_graph.get("summary", {})
    share_package_summary = share_package_validation.get("summary", {})
    receiver_smoke_summary = share_package_receiver_smoke_validation.get("summary", {})
    receiver_smoke = share_package_receiver_smoke_validation.get("receiver_smoke", {})
    nested_extracted_summary = receiver_smoke.get("extracted_validation_summary", {})
    extracted_summary = share_package_validation_extracted.get("summary", {})
    external_standalone_summary = external_standalone_validation.get("summary", {})
    receiver_quickcheck_contract_summary = receiver_quickcheck_contract.get(
        "summary", {}
    )
    collaborator_return_summary = collaborator_return_check.get("summary", {})
    university_review_summary = university_review_packet.get("summary", {})
    tarball_asset_evidence = _check_evidence(
        share_package_validation, "tarball contains quick-read and stage-budget assets"
    )
    tarball_asset_evidence = _quality_evidence_summary(
        tarball_asset_evidence,
        "results/qwen35_report_audit_20260619/share_package_validation.json",
    )
    extracted_asset_evidence = _check_evidence(
        share_package_validation_extracted,
        "extracted bundle contains quick-read and stage-budget assets",
    )
    extracted_asset_evidence = _quality_evidence_summary(
        extracted_asset_evidence,
        "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
    )

    lines: list[str] = [
        "# Qwen3.5-Omni 最终状态摘要",
        "",
        f"生成时间 UTC：`{datetime.now(timezone.utc).isoformat()}`。",
        f"工作目录：`{root}`。",
        "",
        "这是一页给合作方或内部 reviewer 的状态摘要。数字来自当前审计 JSON；若和其他",
        "文档不一致，以 `audit_run_summary.json`、`objective_completion_audit.json` 和",
        "`final_readiness_audit.json`、`final_completion_audit.json` 为准。",
        "",
        "## 1. 当前结论",
        "",
        "- 当前 evidence package 可以作为分享版本发送，但必须携带 caveat。",
        "- 原始目标逐项审计为 `share_ready_with_documented_caveats=true`，不是把长线 goal 标记完成。",
        "- 严格 headline 是 SGLang-Omni Qwen3.5 在 warmed c=4 上优于优化版 vLLM c=4，且 accuracy/WER 不退化。",
        "- 高并发推荐窗口是 c=4 到 c=8；c=16 是压力边界，不是默认服务点。",
        "- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑和 gate 全绿。",
        "",
        "## 2. 机器 Gate",
        "",
    ]
    lines.extend(
        _summary_table(
            [
                ("full audit", _yes(audit_ok), "`audit_run_summary.json`"),
                (
                    "objective completion",
                    _yes(objective_summary.get("share_ready_with_documented_caveats")),
                    (
                        f"`{objective_summary.get('rows_total')}` rows, "
                        f"`{objective_summary.get('required_failures')}` required failures"
                    ),
                ),
                (
                    "final readiness",
                    _yes(final_summary.get("ready")),
                    (
                        f"`{final_summary.get('checks_passed')}/"
                        f"{final_summary.get('checks_total')}` checks"
                    ),
                ),
                (
                    "final completion audit",
                    _yes(final_completion_summary.get("ready")),
                    (
                        f"`{final_completion_summary.get('checks_total')}` checks, "
                        f"`{final_completion_summary.get('tracking_checks')}` tracking, "
                        f"`{final_completion_summary.get('required_failures')}` required failures, "
                        "completion_allowed_now="
                        f"`{final_completion_summary.get('completion_allowed_now')}`, "
                        "blockers="
                        f"`{final_completion_summary.get('completion_blockers')}`"
                    ),
                ),
                (
                    "runtime fairness contract",
                    _yes(runtime_comparison_summary.get("ready")),
                    (
                        f"`{runtime_comparison_summary.get('checks_passed')}/"
                        f"{runtime_comparison_summary.get('checks_total')}` checks, "
                        f"{runtime_comparison_summary.get('allowed_cross_runtime_headline')}, "
                        f"{runtime_comparison_summary.get('vllm_c8_contract')}"
                    ),
                ),
                (
                    "public doc quality guard",
                    _yes(
                        final_summary.get("hard_gates", {}).get(
                            "public_doc_quality_guard"
                        )
                    ),
                    (
                        "`public_doc_quality_guard`="
                        f"`{final_summary.get('hard_gates', {}).get('public_doc_quality_guard')}`; "
                        "hash/table/token/duplicate-heading/semantic-count offenders all empty"
                    ),
                ),
                (
                    "manifest",
                    _yes(_int_or(manifest_summary.get("missing_records"), 1) == 0),
                    (
                        f"`{manifest_summary.get('total_records')}` records, "
                        f"`{manifest_summary.get('missing_records')}` missing"
                    ),
                ),
                (
                    "repro command manifest",
                    _yes(repro_summary.get("ready")),
                    (
                        f"`{repro_summary.get('commands_total')}` commands, "
                        f"`{repro_summary.get('phases_total')}` phases"
                    ),
                ),
                (
                    "slide asset map",
                    _yes(slide_asset_summary.get("ready")),
                    (
                        f"`{slide_asset_summary.get('rows_total')}` rows, "
                        f"`{slide_asset_summary.get('checks_passed')}/"
                        f"{slide_asset_summary.get('checks_total')}` checks"
                    ),
                ),
                (
                    "share bundle manifest",
                    _yes(bundle_summary.get("ready")),
                    (
                        f"`{bundle_summary.get('records_total')}` records, "
                        f"`{bundle_summary.get('missing_required')}` missing required"
                    ),
                ),
                (
                    "share consistency guard",
                    _yes(share_consistency_summary.get("ready")),
                    (
                        f"`{share_consistency_summary.get('checks_passed')}/"
                        f"{share_consistency_summary.get('checks_total')}` checks, "
                        "stale public/machine hits="
                        f"`{share_consistency_summary.get('public_stale_hits')}/"
                        f"{share_consistency_summary.get('machine_stale_hits')}`, "
                        "embedded leaks="
                        f"`{share_consistency_summary.get('embedded_identity_leaks')}`, "
                        "current identity mismatches="
                        f"`{share_consistency_summary.get('tarball_identity_mismatches')}`; "
                        "full post-validation hash chain is sealed by adjacent release seal"
                    ),
                ),
                (
                    "runtime image contract",
                    _yes(runtime_image_summary.get("ready")),
                    (
                        f"`{runtime_image_summary.get('checks_passed')}/"
                        f"{runtime_image_summary.get('checks_total')}` checks, "
                        f"SGLang `{runtime_image_summary.get('sglang_scope')}`, "
                        f"vLLM `{runtime_image_summary.get('vllm_c8_scope')}`"
                    ),
                ),
                (
                    "rerun acceptance contract",
                    _yes(rerun_acceptance_summary.get("ready")),
                    (
                        f"`{rerun_acceptance_summary.get('checks_passed')}/"
                        f"{rerun_acceptance_summary.get('checks_total')}` checks, "
                        f"`{rerun_acceptance_summary.get('rules_total')}` rules, "
                        f"`{rerun_acceptance_summary.get('return_evidence_files')}` "
                        "return evidence files, "
                        f"`{rerun_acceptance_summary.get('return_evidence_command_rows')}` "
                        "command matrix rows"
                    ),
                ),
                (
                    "rerun time budget",
                    _yes(rerun_time_budget_summary.get("ready")),
                    (
                        f"`{rerun_time_budget_summary.get('rows_total')}` rows, "
                        f"`{rerun_time_budget_summary.get('timed_rows')}` timed rows, "
                        "timed lower bound="
                        f"`{rerun_time_budget_summary.get('total_timed_benchmark_wall_s')}s`, "
                        "8-GPU lower bound="
                        f"`{rerun_time_budget_summary.get('equivalent_8gpu_timed_lower_bound_gpu_hours')} GPUh`"
                    ),
                ),
                (
                    "rerun delta triage",
                    _yes(rerun_delta_summary.get("ready")),
                    (
                        f"`{rerun_delta_summary.get('rows_total')}` symptoms, "
                        f"`{rerun_delta_summary.get('checks_passed')}/"
                        f"{rerun_delta_summary.get('checks_total')}` checks"
                    ),
                ),
                (
                    "final checkpoint watchlist",
                    _yes(checkpoint_summary.get("ready")),
                    (
                        f"`{checkpoint_summary.get('checks_passed')}/"
                        f"{checkpoint_summary.get('checks_total')}` checks, "
                        f"`{checkpoint_summary.get('watch_items_total')}` watch items"
                    ),
                ),
                (
                    "stage latency budget",
                    _yes(stage_budget_summary.get("ready")),
                    (
                        f"`{stage_budget_summary.get('checks_passed')}/"
                        f"{stage_budget_summary.get('checks_total')}` checks, "
                        f"SGLang `{stage_budget_summary.get('sglang_budget_rows')}` / "
                        f"synthetic `{stage_budget_summary.get('synthetic_budget_rows')}` / "
                        f"vLLM `{stage_budget_summary.get('vllm_budget_rows')}` rows"
                    ),
                ),
                (
                    "length regime coverage",
                    _yes(length_regime_summary.get("ready")),
                    (
                        f"`{length_regime_summary.get('checks_passed')}/"
                        f"{length_regime_summary.get('checks_total')}` checks, "
                        f"`{length_regime_summary.get('rows_total')}` rows, "
                        f"long c=8 RTF p95="
                        f"`{length_regime_summary.get('long_c8_rtf_p95')}`, "
                        f"max hop/decode p95="
                        f"`{length_regime_summary.get('max_talker_to_code2wav_hop_p95_ms')}/"
                        f"{length_regime_summary.get('max_code2wav_decode_p95_ms')}ms`"
                    ),
                ),
                (
                    "stage boundary bottleneck ledger",
                    _yes(stage_ledger_summary.get("ready")),
                    (
                        f"`{stage_ledger_summary.get('checks_passed')}/"
                        f"{stage_ledger_summary.get('checks_total')}` checks, "
                        f"`{stage_ledger_summary.get('ledger_rows')}` boundary rows"
                    ),
                ),
                (
                    "stage causal graph JSON",
                    _yes(stage_causal_summary.get("ready")),
                    (
                        f"`{stage_causal_summary.get('checks_passed')}/"
                        f"{stage_causal_summary.get('checks_total')}` checks, "
                        f"`{stage_causal_summary.get('causal_edges_total')}` edges, "
                        f"`{stage_causal_summary.get('raw_drilldown_rows')}` drilldown rows"
                    ),
                ),
                (
                    "share tarball",
                    _yes(package_summary.get("ready")),
                    f"checksum `{package.get('checksum_relative_path')}`",
                ),
                (
                    "share package validation",
                    _yes(share_package_summary.get("ready")),
                    (
                        f"`{share_package_summary.get('checks_passed')}/"
                        f"{share_package_summary.get('checks_total')}` checks, "
                        f"`{share_package_summary.get('required_failures')}` required failures, "
                        f"{tarball_asset_evidence}"
                    ),
                ),
                (
                    "receiver smoke validation",
                    _yes(
                        receiver_smoke_summary.get("ready")
                        and receiver_smoke_summary.get("receiver_smoke_ready")
                    ),
                    (
                        f"`{receiver_smoke_summary.get('checks_passed')}/"
                        f"{receiver_smoke_summary.get('checks_total')}` checks, "
                        "receiver_smoke_ready="
                        f"{str(receiver_smoke_summary.get('receiver_smoke_ready')).lower()}, "
                        "nested extracted-only "
                        f"`{nested_extracted_summary.get('checks_passed')}/"
                        f"{nested_extracted_summary.get('checks_total')}`"
                    ),
                ),
                (
                    "extracted-only validation",
                    _yes(
                        extracted_summary.get("ready")
                        and extracted_summary.get("extracted_only")
                    ),
                    (
                        f"`{extracted_summary.get('checks_passed')}/"
                        f"{extracted_summary.get('checks_total')}` checks, "
                        "extracted_only="
                        f"{str(extracted_summary.get('extracted_only')).lower()}, "
                        f"{extracted_asset_evidence}"
                    ),
                ),
                (
                    "external standalone validation",
                    _yes(external_standalone_summary.get("ready")),
                    (
                        f"`{external_standalone_summary.get('checks_passed')}/"
                        f"{external_standalone_summary.get('checks_total')}` checks, "
                        "bundled validator, nested extracted-only "
                        f"`{external_standalone_summary.get('extracted_validation_checks')}` checks"
                    ),
                ),
                (
                    "receiver quickcheck contract",
                    _yes(receiver_quickcheck_contract_summary.get("ready")),
                    (
                        f"`{receiver_quickcheck_contract_summary.get('checks_passed')}/"
                        f"{receiver_quickcheck_contract_summary.get('checks_total')}` checks, "
                        f"`{receiver_quickcheck_contract_summary.get('receiver_jsons_total')}` "
                        "receiver JSONs, "
                        f"`{receiver_quickcheck_contract_summary.get('public_docs_total')}` "
                        "public docs, "
                        f"`{receiver_quickcheck_contract_summary.get('completion_gate_docs_total')}` "
                        "completion-gate docs, "
                        f"`{receiver_quickcheck_contract_summary.get('wer_asr_docs_total')}` "
                        "WER/ASR docs, "
                        f"`{receiver_quickcheck_contract_summary.get('evidence_smoke_explicit_docs_total')}` "
                        "evidence-smoke docs"
                    ),
                ),
                (
                    "collaborator return check",
                    _yes(collaborator_return_summary.get("ready")),
                    (
                        f"`{collaborator_return_summary.get('checks_passed')}/"
                        f"{collaborator_return_summary.get('checks_total')}` checks, "
                        f"`{collaborator_return_summary.get('required_return_files_present')}/"
                        f"{collaborator_return_summary.get('required_return_files_total')}` "
                        "return files, "
                        f"`{collaborator_return_summary.get('command_matrix_rows_ready')}/"
                        f"{collaborator_return_summary.get('command_matrix_rows_total')}` "
                        "command rows, decision="
                        f"`{collaborator_return_summary.get('replacement_review_decision')}`"
                    ),
                ),
                (
                    "share release seal",
                    _yes(share_release_seal_summary.get("ready")),
                    (
                        f"`{share_release_seal_summary.get('checks_passed')}/"
                        f"{share_release_seal_summary.get('checks_total')}` checks, "
                        "receiver_smoke_ready="
                        f"{str(share_release_seal_summary.get('receiver_smoke_ready')).lower()}, "
                        "forbidden_tarball_members="
                        f"`{share_release_seal_summary.get('forbidden_tarball_members')}`, "
                        f"send_decision=`{share_release_seal_summary.get('send_decision')}`"
                    ),
                ),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## 3. 发送文件",
            "",
            f"- 便捷 tarball：`{package.get('tarball_relative_path')}`",
            f"- tarball checksum：`{package.get('checksum_relative_path')}`",
            "- 随包逐文件 hash 清单：tarball 内 `PACKAGE_FILE_SHA256SUMS.txt`",
            "- 伴随 package manifest：`results/qwen35_report_audit_20260619/share_bundle_package_manifest.json`",
            "- 伴随 package validation：`results/qwen35_report_audit_20260619/share_package_validation.json`",
            "- 伴随 extracted validation：`results/qwen35_report_audit_20260619/share_package_validation_extracted.json`",
            "- 伴随 receiver smoke validation：`results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`",
            "- 伴随 external standalone validation：`results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`",
            "- 伴随 receiver quickcheck contract JSON：`results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`",
            "- 伴随 collaborator return check JSON：`results/qwen35_report_audit_20260619/collaborator_return_check.json`",
            "- 伴随 final completion audit JSON：`results/qwen35_report_audit_20260619/final_completion_audit.json`",
            "- 伴随 final completion audit 报告：`benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`",
            "- 伴随 share release seal JSON：`results/qwen35_report_audit_20260619/share_release_seal.json`",
            "- 伴随 share release seal 报告：`benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md`",
            "- 主报告：`benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`",
            "- 分享索引：`benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`",
            "- 高校合作方审阅会议包：`benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md`",
            "- 高校合作方审阅会议包 JSON：`results/qwen35_report_audit_20260619/university_review_packet.json`",
            "- 分享一致性 guard：`benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md`",
            "- 分享一致性 guard JSON：`results/qwen35_report_audit_20260619/share_consistency_guard.json`",
            "- 接收方路径手册：`benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md`",
            "- 接收方命令卡：`benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`",
            "- 接收方 quickcheck contract：`benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md`",
            "- 一页式 scorecard：`benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`",
            "- Deck 图表资产映射：`benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md`",
            "- 图表来源一致性报告：`benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md`",
            "- 图表来源一致性 JSON：`results/qwen35_report_audit_20260619/chart_source_consistency.json`",
            "- 决策矩阵：`benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md`",
            "- 决策矩阵 JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`",
            "- 原始需求证据映射：`benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`",
            "- 压力条件总表：`benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`",
            "- 指标来源索引：`benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`",
            "- 公平对比合同：`benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md`",
            "- 公平对比合同 JSON：`results/qwen35_report_audit_20260619/runtime_comparison_contract.json`",
            "- Runtime image contract：`benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md`",
            "- 复跑验收阈值合同：`benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`",
            "- 复跑耗时/算力预算：`benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`",
            "- 复跑耗时/算力预算 JSON：`results/qwen35_report_audit_20260619/rerun_time_budget.json`",
            "- 复跑差异定位矩阵：`benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`",
            "- SGLang 优化锁定矩阵：`benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md`",
            "- vLLM 优化锁定矩阵：`benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`",
            "- 优化候选裁决 ledger：`benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md`",
            "- 优化候选裁决 JSON：`results/qwen35_report_audit_20260619/optimization_candidate_ledger.json`",
            "- Caveat 裁决矩阵 JSON：`results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json`",
            "- vLLM c=8 online parity 升级协议：`benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md`",
            "- 最终 checkpoint watchlist：`benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md`",
            "- Stage latency budget：`benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`",
            "- 长短输入/输出 coverage：`benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md`",
            "- 长短输入/输出 coverage JSON：`results/qwen35_report_audit_20260619/length_regime_coverage.json`",
            "- Stage boundary bottleneck ledger：`benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`",
            "- Stage 因果图：`benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`",
            "- Stage 因果图 JSON：`results/qwen35_report_audit_20260619/stage_causal_graph.json`",
            "- Stage 指标字典：`benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`",
            "- Caveat 裁决矩阵：`benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md`",
            "- 最终交付说明：`benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md`",
            "- 外部复现 handoff runbook：`benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`",
            "- 复现清单：`benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`",
            "- 合作方复跑验收表：`benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`",
            "- 复跑差异定位 JSON：`results/qwen35_report_audit_20260619/rerun_delta_triage.json`",
            "- 答辩 Q&A：`benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`",
            "- 分享 deck 提纲：`benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md`",
            "- 快速收包校验脚本：`benchmarks/eval/validate_qwen35_omni_share_package.py`",
            "- 原始目标审计：`results/qwen35_report_audit_20260619/objective_completion_audit.json`",
            "- Deck 图表资产映射 JSON：`results/qwen35_report_audit_20260619/slide_asset_map.json`",
            "",
            "当前 tarball digest 的权威来源是同目录 `.sha256` 文件和",
            "`share_bundle_package_manifest.json` 的 `tarball_sha256` 字段；随包 Markdown",
            "不内嵌 tarball digest 数值，避免报告文本变更导致 tarball hash 自引用漂移。",
            (
                "随包 `share_consistency_guard` 是 "
                f"{share_consistency_summary.get('checks_passed')}/"
                f"{share_consistency_summary.get('checks_total')} 的包内一致性 guard；"
                "包内副本可能显示"
            ),
            "`tarball_identity_fields_active=3`、`tarball_identity_deferred_fields=3`，这是预期状态，",
            "因为 tarball-mode validation、receiver smoke validation 和 release seal 必须在打包后生成。",
            "完整 post-validation hash 链由相邻 `.sha256`、package manifest、validation、receiver smoke 和",
            "share release seal 共同证明。",
            "",
            "`share_bundle_package_manifest.json`、`share_package_validation.json`、",
            "`share_package_validation_extracted.json`、`share_package_receiver_smoke_validation.json` 和",
            "`share_package_external_standalone_validation.json`，以及",
            "`receiver_quickcheck_contract.json`、`final_completion_audit.json`、`share_release_seal.json` 是",
            "和 tarball 同目录保存或运行生成的伴随验证证据，",
            "不是 tarball 内成员；它们描述 tarball 自身，放入 tarball 会造成自引用 hash。",
            "`qwen35_omni_final_completion_audit_zh_20260621.md` 是给人读的最终 completion gate，",
            "也应和 tarball 相邻保存。",
            "`qwen35_omni_share_release_seal_zh_20260621.md` 是给人读的同一封口证据，",
            "也应和 tarball 相邻保存。",
            "`PACKAGE_FILE_SHA256SUMS.txt` 是 tarball 内成员，记录每个随包源文件的相对仓库根路径和",
            "逐文件 hash；tarball-mode validator 会先用它直接校验 tar member 内容，接收方解包后",
            "运行 extracted-only validator 时会再次复核报告、证据 JSON、工具脚本和图表资产。",
            "同一个 validator 还会直接扫描随包 `share_report` Markdown 中的裸 hash、坏表格、重复 heading 和坏展示 token，",
            "并检查随包 `share_charts` CSV/SVG 可解析、非空且结构可渲染；通过时 evidence 应显示",
            "`report_quality_offenders=[]` 和 `chart_quality_offenders=[]`。",
            "",
            "如果接收方仓库路径不是 `/home/gangouyu/sglang-omni`，先改 `HOST_REPO`；",
            "`SMOKE_DIR` 和 `EXTRACT_DIR` 只用于收包校验和手动解包校验。",
            "",
            "接收方可先执行：",
            "",
            "```bash",
            "HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"",
            "SMOKE_DIR=\"${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final_status}\"",
            "EXTRACT_DIR=\"${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final_status}\"",
            "STANDALONE_DIR=\"${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_final_status}\"",
            "cd \"$HOST_REPO\"",
            "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
            "jq '.summary' results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
            "python3 -m benchmarks.eval.build_qwen35_omni_final_completion_audit \\",
            "  --root \"$HOST_REPO\" \\",
            "  --output benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md \\",
            "  --json-output results/qwen35_report_audit_20260619/final_completion_audit.json",
            "python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \\",
            "  --root \"$HOST_REPO\" \\",
            "  --strict \\",
            "  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \\",
            "  --json-output results/qwen35_report_audit_20260619/share_release_seal.json",
            "python3 -m benchmarks.eval.run_qwen35_omni_report_audit \\",
            "  --root \"$HOST_REPO\" \\",
            "  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json",
            "```",
            "",
            "## 4. 必须携带的 Caveat",
            "",
        ]
    )
    if protocol_summary:
        lines.append(
            "- vLLM c=8 online parity protocol 当前为 "
            f"`ready={protocol_summary.get('ready')}`，"
            f"`online_parity_proven={protocol_summary.get('online_parity_proven')}`；"
            "升级 headline 前必须先补 online ingress artifacts。"
        )
    lines.append(
        "- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；"
        "更大 Video-AMME 或真实线上流量需要同口径 stage/tail/quality gate。"
    )
    boundary_rows = _boundary_rows(objective)
    if boundary_rows:
        for row in boundary_rows:
            caveat = str(row.get("caveat") or row.get("requirement") or "").strip()
            if caveat:
                lines.append(f"- {caveat}")
    else:
        lines.extend(
            [
                "- Official SeedTTS full-set is not a headline benchmark in this package.",
                "- vLLM c=8 prebuild w4 is an optimized offline diagnostic, not online serving parity.",
                "- ci-50/stress/synthetic evidence must not be extrapolated to full online traffic without same-scope reruns.",
                "- c=16 is a saturation boundary, not the recommended serving point.",
            ]
        )
    lines.extend(
        [
            "",
            "## 5. 当前发送判断",
            "",
            f"- send decision：`{objective_summary.get('send_decision')}`",
            (
                "- university review packet："
                f"`ready={university_review_summary.get('ready')}`，"
                f"`checks={university_review_summary.get('checks_passed')}/"
                f"{university_review_summary.get('checks_total')}`"
            ),
            f"- long-running goal complete：`{objective_summary.get('goal_complete')}`",
            f"- final checkpoint watchlist：`ready={checkpoint_summary.get('ready')}`，"
            f"`completion_allowed_now={checkpoint_summary.get('completion_allowed_now')}`",
            f"- stage latency budget：`ready={stage_budget_summary.get('ready')}`，"
            f"`checks={stage_budget_summary.get('checks_passed')}/"
            f"{stage_budget_summary.get('checks_total')}`",
            f"- stage boundary bottleneck ledger：`ready={stage_ledger_summary.get('ready')}`，"
            f"`checks={stage_ledger_summary.get('checks_passed')}/"
            f"{stage_ledger_summary.get('checks_total')}`，"
            f"`rows={stage_ledger_summary.get('ledger_rows')}`",
            f"- stage causal graph JSON：`ready={stage_causal_summary.get('ready')}`，"
            f"`checks={stage_causal_summary.get('checks_passed')}/"
            f"{stage_causal_summary.get('checks_total')}`，"
            f"`edges={stage_causal_summary.get('causal_edges_total')}`，"
            f"`raw_drilldown={stage_causal_summary.get('raw_drilldown_rows')}`",
            f"- slide asset map：`ready={slide_asset_summary.get('ready')}`，"
            f"`rows={slide_asset_summary.get('rows_total')}`，"
            f"`checks={slide_asset_summary.get('checks_passed')}/"
            f"{slide_asset_summary.get('checks_total')}`",
            f"- rerun delta triage：`ready={rerun_delta_summary.get('ready')}`，"
            f"`rows={rerun_delta_summary.get('rows_total')}`，"
            f"`checks={rerun_delta_summary.get('checks_passed')}/"
            f"{rerun_delta_summary.get('checks_total')}`",
            f"- rerun time budget：`ready={rerun_time_budget_summary.get('ready')}`，"
            f"`rows={rerun_time_budget_summary.get('rows_total')}`，"
            f"`timed_rows={rerun_time_budget_summary.get('timed_rows')}`，"
            f"`8gpu_lower_bound_gpuh="
            f"{rerun_time_budget_summary.get('equivalent_8gpu_timed_lower_bound_gpu_hours')}`",
            f"- runtime fairness contract：`ready={runtime_comparison_summary.get('ready')}`，"
            f"`checks={runtime_comparison_summary.get('checks_passed')}/"
            f"{runtime_comparison_summary.get('checks_total')}`，"
            f"`headline={runtime_comparison_summary.get('allowed_cross_runtime_headline')}`",
            f"- external standalone validation：`ready={external_standalone_summary.get('ready')}`，"
            f"`checks={external_standalone_summary.get('checks_passed')}/"
            f"{external_standalone_summary.get('checks_total')}`",
            f"- receiver quickcheck contract：`ready={receiver_quickcheck_contract_summary.get('ready')}`，"
            f"`checks={receiver_quickcheck_contract_summary.get('checks_passed')}/"
            f"{receiver_quickcheck_contract_summary.get('checks_total')}`，"
            f"`wer_asr_docs={receiver_quickcheck_contract_summary.get('wer_asr_docs_total')}`，"
            f"`evidence_smoke_docs="
            f"{receiver_quickcheck_contract_summary.get('evidence_smoke_explicit_docs_total')}`，"
            f"`required_failures={receiver_quickcheck_contract_summary.get('required_failures')}`",
            f"- collaborator return check：`ready={collaborator_return_summary.get('ready')}`，"
            f"`checks={collaborator_return_summary.get('checks_passed')}/"
            f"{collaborator_return_summary.get('checks_total')}`，"
            f"`return_files={collaborator_return_summary.get('required_return_files_present')}/"
            f"{collaborator_return_summary.get('required_return_files_total')}`，"
            f"`command_rows={collaborator_return_summary.get('command_matrix_rows_ready')}/"
            f"{collaborator_return_summary.get('command_matrix_rows_total')}`，"
            f"`decision={collaborator_return_summary.get('replacement_review_decision')}`",
            f"- share consistency guard：`ready={share_consistency_summary.get('ready')}`，"
            f"`checks={share_consistency_summary.get('checks_passed')}/"
            f"{share_consistency_summary.get('checks_total')}`，"
            f"`identity_mismatches={share_consistency_summary.get('tarball_identity_mismatches')}`",
            f"- share release seal：`ready={share_release_seal_summary.get('ready')}`，"
            f"`checks={share_release_seal_summary.get('checks_passed')}/"
            f"{share_release_seal_summary.get('checks_total')}`，"
            f"`send_decision={share_release_seal_summary.get('send_decision')}`",
            "",
            "这页摘要证明当前 share package 已可带 caveat 分享；更新后的目标不再等待 6.21 晚上，是否完成由 final completion audit 的证据门裁决。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni final status Markdown summary."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(root), encoding="utf-8")
    print(f"Final status summary written: {output}")


if __name__ == "__main__":
    main()
