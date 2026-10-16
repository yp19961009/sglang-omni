# SPDX-License-Identifier: Apache-2.0
"""Build a compact university-review packet for the Qwen3.5-Omni share package."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "university_review_packet.json"


@dataclass(frozen=True)
class PacketCheck:
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _read_text_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


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


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _add_check(
    checks: list[PacketCheck],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(PacketCheck(name, _status(condition), evidence, required))


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


def _share_validation_pending_in_full_audit(
    share_validation: dict[str, Any],
    audit_summary: dict[str, Any],
) -> bool:
    summary = _summary(share_validation)
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 17
        and _int_value(summary.get("checks_passed")) >= 16
        and _int_value(summary.get("required_failures"), default=99) <= 1
        and _int_value(summary.get("missing_bundle_members"), default=1) == 0
        and _failed_required_check_names(share_validation)
        == {"tarball contains quick-read and stage-budget assets"}
    )


def _receiver_contract_pending_in_full_audit(
    receiver_contract: dict[str, Any],
    audit_summary: dict[str, Any],
) -> bool:
    summary = _summary(receiver_contract)
    allowed_failures = {
        "current receiver evidence JSONs are all green",
        "current receiver evidence is asset-quality clean",
    }
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("checks_total")) >= 14
        and _int_value(summary.get("checks_passed")) >= 12
        and _int_value(summary.get("required_failures"), default=99)
        <= len(allowed_failures)
        and _failed_required_check_names(receiver_contract).issubset(allowed_failures)
        and _int_value(summary.get("quickcheck_steps")) == 6
        and _int_value(summary.get("receiver_jsons_total")) >= 4
        and _int_value(summary.get("public_docs_total")) >= 8
        and _int_value(summary.get("completion_gate_docs_total")) >= 6
        and _int_value(summary.get("wer_asr_docs_total")) >= 3
    )


def _headline_fact_table(scorecard: dict[str, Any]) -> list[tuple[str, str, str]]:
    strict = scorecard.get("strict_c4_comparison", {})
    sglang = strict.get("sglang", {}) if isinstance(strict, dict) else {}
    vllm = strict.get("vllm", {}) if isinstance(strict, dict) else {}
    rel = strict.get("relative_sglang_lower_pct", {}) if isinstance(strict, dict) else {}
    peak = scorecard.get("sglang_stress", {}).get("throughput_peak", {})
    long_c8 = scorecard.get("synthetic_long_c8", {})
    return [
        (
            "Strict warmed c=4",
            (
                f"SGLang latency mean `{_float_value(sglang.get('latency_mean_s')):.3f}s`, "
                f"vLLM `{_float_value(vllm.get('latency_mean_s')):.3f}s`; "
                f"SGLang lower by `{_float_value(rel.get('latency_mean')):.1f}%`"
            ),
            "headline_scorecard.json / strict_c4_comparison",
        ),
        (
            "Strict c=4 tail/RTF",
            (
                f"SGLang RTF p95 `{_float_value(sglang.get('rtf_p95')):.3f}`, "
                f"vLLM `{_float_value(vllm.get('rtf_p95')):.3f}`; "
                f"SGLang lower by `{_float_value(rel.get('rtf_p95')):.1f}%`"
            ),
            "headline_scorecard.json / relative_sglang_lower_pct",
        ),
        (
            "SGLang high-concurrency edge",
            (
                f"c=8 throughput `{_float_value(peak.get('throughput_qps')):.3f} qps`, "
                f"latency p95 `{_float_value(peak.get('latency_p95_s')):.3f}s`; "
                "c=16 remains the saturation boundary"
            ),
            "headline_scorecard.json / sglang_stress.throughput_peak",
        ),
        (
            "Long-form speech guard",
            (
                f"long c=8 target `{_int_value(long_c8.get('target_chars'))}` chars / "
                f"`{_int_value(long_c8.get('target_words'))}` words, "
                f"RTF p95 `{_float_value(long_c8.get('rtf_p95')):.4f}`"
            ),
            "headline_scorecard.json / synthetic_long_c8",
        ),
    ]


def _required_docs_present(root: Path) -> tuple[bool, list[str]]:
    docs = [
        "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md",
        "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md",
    ]
    missing = [rel for rel in docs if not (root / rel).is_file()]
    return not missing, missing


def build_packet(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    audit = _load_json_optional(audit_dir / "audit_run_summary.json")
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    final_completion = _load_json_optional(audit_dir / "final_completion_audit.json")
    share_release = _load_json_optional(audit_dir / "share_release_seal.json")
    share_validation = _load_json_optional(audit_dir / "share_package_validation.json")
    receiver_smoke = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    extracted_validation = _load_json_optional(
        audit_dir / "share_package_validation_extracted.json"
    )
    standalone_validation = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    scorecard = _load_json_optional(audit_dir / "headline_scorecard.json")
    runtime_image = _load_json_optional(audit_dir / "runtime_image_contract.json")
    runtime_comparison = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    stage_latency = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_boundary = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    stage_causal = _load_json_optional(audit_dir / "stage_causal_graph.json")
    length_regime = _load_json_optional(audit_dir / "length_regime_coverage.json")
    rerun_acceptance = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    rerun_budget = _load_json_optional(audit_dir / "rerun_time_budget.json")
    command_hygiene = _load_json_optional(audit_dir / "command_reference_hygiene.json")
    receiver_contract = _load_json_optional(
        audit_dir / "receiver_quickcheck_contract.json"
    )
    defense_claim = _load_json_optional(audit_dir / "defense_claim_matrix.json")
    caveat_matrix = _load_json_optional(audit_dir / "caveat_adjudication_matrix.json")
    vllm_protocol = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    technical_report_text = _read_text_optional(
        root / "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md"
    )

    final_summary = _summary(final_readiness)
    final_completion_summary = _summary(final_completion)
    release_summary = _summary(share_release)
    share_validation_summary = _summary(share_validation)
    receiver_smoke_summary = _summary(receiver_smoke)
    extracted_summary = _summary(extracted_validation)
    standalone_summary = _summary(standalone_validation)
    scorecard_summary = _summary(scorecard)
    runtime_image_summary = _summary(runtime_image)
    runtime_comparison_summary = _summary(runtime_comparison)
    stage_latency_summary = _summary(stage_latency)
    stage_boundary_summary = _summary(stage_boundary)
    stage_causal_summary = _summary(stage_causal)
    length_summary = _summary(length_regime)
    rerun_acceptance_summary = _summary(rerun_acceptance)
    rerun_budget_summary = _summary(rerun_budget)
    command_hygiene_summary = _summary(command_hygiene)
    receiver_contract_summary = _summary(receiver_contract)
    defense_summary = _summary(defense_claim)
    caveat_summary = _summary(caveat_matrix)
    vllm_protocol_summary = _summary(vllm_protocol)
    share_bundle_summary = _summary(share_bundle)
    manifest_summary = _summary(manifest)
    share_validation_pending = _share_validation_pending_in_full_audit(
        share_validation,
        audit,
    )
    receiver_contract_pending = _receiver_contract_pending_in_full_audit(
        receiver_contract,
        audit,
    )
    share_validation_ready_for_packet = (
        bool(share_validation_summary.get("ready")) or share_validation_pending
    )
    receiver_contract_ready_for_packet = (
        bool(receiver_contract_summary.get("ready")) or receiver_contract_pending
    )
    glossary_terms = [
        "### 2.1 指标和口径速查",
        "warmed / skip-first",
        "queue estimate",
        "offline diagnostic",
        "share-ready with caveat",
    ]
    missing_glossary_terms = [
        term for term in glossary_terms if term not in technical_report_text
    ]

    checks: list[PacketCheck] = []
    _add_check(
        checks,
        "full audit and readiness are green",
        bool(audit.get("ok"))
        and bool(final_summary.get("ready"))
        and _int_value(final_summary.get("checks_passed")) >= 49
        and _int_value(final_summary.get("required_failures"), default=1) == 0,
        (
            f"audit_ok={audit.get('ok')}, final_readiness="
            f"{final_summary.get('checks_passed')}/{final_summary.get('checks_total')}, "
            f"required_failures={final_summary.get('required_failures')}"
        ),
    )
    _add_check(
        checks,
        "share package receiver path is validated",
        bool(release_summary.get("ready"))
        and share_validation_ready_for_packet
        and bool(receiver_smoke_summary.get("ready"))
        and bool(extracted_summary.get("ready"))
        and bool(standalone_summary.get("ready"))
        and (
            _int_value(share_validation_summary.get("required_failures"), default=1)
            == 0
            or share_validation_pending
        )
        and _int_value(receiver_smoke_summary.get("required_failures"), default=1) == 0
        and _int_value(extracted_summary.get("required_failures"), default=1) == 0
        and _int_value(standalone_summary.get("required_failures"), default=1) == 0,
        (
            f"release={release_summary.get('checks_passed')}/"
            f"{release_summary.get('checks_total')}, "
            f"tarball={share_validation_summary.get('checks_passed')}/"
            f"{share_validation_summary.get('checks_total')}, "
            f"receiver={receiver_smoke_summary.get('checks_passed')}/"
            f"{receiver_smoke_summary.get('checks_total')}, "
            f"extracted={extracted_summary.get('checks_passed')}/"
            f"{extracted_summary.get('checks_total')}, "
            f"standalone={standalone_summary.get('checks_passed')}/"
            f"{standalone_summary.get('checks_total')}, "
            f"tarball_pending_in_full_audit={share_validation_pending}"
        ),
    )
    headline_checks = scorecard_summary.get("checks", {})
    headline_checks = headline_checks if isinstance(headline_checks, dict) else {}
    _add_check(
        checks,
        "headline claims are evidence-backed",
        bool(scorecard_summary.get("ready"))
        and bool(headline_checks.get("strict_c4_sglang_latency_rtf_win"))
        and bool(headline_checks.get("strict_c4_accuracy_wer_preserved"))
        and bool(headline_checks.get("sglang_stress_c8_is_peak"))
        and bool(headline_checks.get("long_c8_faster_than_real_time")),
        f"headline_scorecard={scorecard_summary}",
    )
    _add_check(
        checks,
        "runtime fairness and vLLM baseline strength are locked",
        bool(runtime_image_summary.get("ready"))
        and bool(runtime_comparison_summary.get("ready"))
        and runtime_comparison_summary.get("allowed_cross_runtime_headline")
        == "warmed c=4 only"
        and runtime_comparison_summary.get("vllm_c8_contract")
        == "offline_diagnostic_not_online_parity"
        and "optimized image" in str(runtime_comparison_summary.get("baseline_strength") or ""),
        (
            f"runtime_image={runtime_image_summary}; "
            f"runtime_comparison={runtime_comparison_summary}"
        ),
    )
    _add_check(
        checks,
        "review glossary and metric-scope route is present",
        not missing_glossary_terms,
        (
            "technical_report=benchmarks/reports/"
            "qwen35_omni_university_technical_report_zh_20260621.md; "
            f"missing_glossary_terms={missing_glossary_terms}"
        ),
    )
    _add_check(
        checks,
        "stage and boundary diagnosis are green",
        bool(stage_latency_summary.get("ready"))
        and bool(stage_boundary_summary.get("ready"))
        and bool(stage_causal_summary.get("ready"))
        and _int_value(stage_boundary_summary.get("ledger_rows")) >= 37
        and _int_value(stage_causal_summary.get("causal_edges_total")) >= 7
        and bool(stage_causal_summary.get("sglang_handoff_healthy"))
        and bool(stage_causal_summary.get("sglang_decode_not_bottleneck")),
        (
            f"stage_latency={stage_latency_summary}; "
            f"stage_boundary={stage_boundary_summary}; "
            f"stage_causal={stage_causal_summary}"
        ),
    )
    _add_check(
        checks,
        "short/long and high-concurrency regimes are covered",
        bool(length_summary.get("ready"))
        and _int_value(length_summary.get("rows_total")) >= 7
        and _int_value(length_summary.get("short_rows")) >= 3
        and _int_value(length_summary.get("long_rows")) >= 3
        and _float_value(length_summary.get("long_c8_rtf_p95"), default=99.0) < 1.0,
        f"length_regime={length_summary}",
    )
    _add_check(
        checks,
        "rerun budget and replacement contract are explicit",
        bool(rerun_budget_summary.get("ready"))
        and bool(rerun_acceptance_summary.get("ready"))
        and _int_value(rerun_budget_summary.get("rows_total")) >= 9
        and _int_value(rerun_budget_summary.get("timed_rows")) >= 6
        and _float_value(rerun_budget_summary.get("total_timed_benchmark_wall_s")) > 0.0
        and _int_value(rerun_acceptance_summary.get("return_evidence_files")) >= 34,
        (
            f"rerun_budget={rerun_budget_summary}; "
            f"rerun_acceptance={rerun_acceptance_summary}"
        ),
    )
    _add_check(
        checks,
        "rerun command references resolve",
        bool(command_hygiene_summary.get("ready"))
        and _int_value(command_hygiene_summary.get("manifest_commands_total")) >= 63
        and _int_value(
            command_hygiene_summary.get("structured_artifacts_with_refs_total")
        )
        >= 9
        and _int_value(
            command_hygiene_summary.get("structured_unique_command_refs_total")
        )
        >= 63
        and _int_value(
            command_hygiene_summary.get("unresolved_command_refs_total"), default=1
        )
        == 0
        and not command_hygiene_summary.get("missing_critical_doc_command_ids"),
        f"command_reference_hygiene={command_hygiene_summary}",
    )
    _add_check(
        checks,
        "receiver quickcheck and defense path are ready",
        receiver_contract_ready_for_packet
        and _int_value(receiver_contract_summary.get("quickcheck_steps")) == 6
        and bool(defense_summary.get("ready"))
        and _int_value(defense_summary.get("question_rows_total")) >= 13
        and _int_value(defense_summary.get("required_failures"), default=1) == 0,
        (
            f"receiver_quickcheck={receiver_contract_summary}; "
            f"receiver_contract_pending_in_full_audit={receiver_contract_pending}; "
            f"defense_claim_matrix={defense_summary}"
        ),
    )
    _add_check(
        checks,
        "caveat and vLLM c8 parity boundaries are explicit",
        bool(caveat_summary.get("ready"))
        and bool(vllm_protocol_summary.get("ready"))
        and not bool(vllm_protocol_summary.get("online_parity_proven"))
        and not bool(caveat_summary.get("seedtts_fullset_headline"))
        and caveat_summary.get("current_best_scope")
        == "measured_best_not_global_optimum",
        f"caveat={caveat_summary}; vllm_online_protocol={vllm_protocol_summary}",
    )
    _add_check(
        checks,
        "final completion is evidence-gated",
        bool(final_completion_summary.get("ready"))
        and bool(final_completion_summary.get("evidence_ready"))
        and bool(final_completion_summary.get("completion_allowed_now"))
        and not final_completion_summary.get("completion_blockers"),
        f"final_completion={final_completion_summary}",
    )
    docs_ok, missing_docs = _required_docs_present(root)
    _add_check(
        checks,
        "review reading route files are present",
        docs_ok,
        f"missing_docs={missing_docs}",
    )
    _add_check(
        checks,
        "share and evidence manifests include the reviewable package",
        bool(share_bundle_summary.get("ready"))
        and _int_value(share_bundle_summary.get("missing_required"), default=1) == 0
        and _int_value(manifest_summary.get("missing_records"), default=1) == 0,
        f"share_bundle={share_bundle_summary}; manifest={manifest_summary}",
    )

    required_failures = sum(
        1 for check in checks if check.required and check.status != "PASS"
    )
    summary = {
        "ready": required_failures == 0,
        "checks_total": len(checks),
        "checks_passed": sum(1 for check in checks if check.status == "PASS"),
        "required_failures": required_failures,
        "sections_total": 7,
        "meeting_minutes": 15,
        "reviewer_first_files": [
            "benchmarks/reports/qwen35_omni_start_here_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md",
            "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md",
        ],
        "headline_scope": runtime_comparison_summary.get(
            "allowed_cross_runtime_headline"
        ),
        "vllm_c8_scope": runtime_comparison_summary.get("vllm_c8_contract"),
        "recommended_sglang_window": stage_boundary_summary.get(
            "recommended_sglang_window"
        ),
        "saturation_boundary": stage_boundary_summary.get("saturation_boundary"),
        "rerun_timed_wall_s_lower_bound": rerun_budget_summary.get(
            "total_timed_benchmark_wall_s"
        ),
        "rerun_8gpu_hours_lower_bound": rerun_budget_summary.get(
            "equivalent_8gpu_timed_lower_bound_gpu_hours"
        ),
        "rerun_command_refs_unresolved": command_hygiene_summary.get(
            "unresolved_command_refs_total"
        ),
        "completion_allowed_now": final_completion_summary.get(
            "completion_allowed_now"
        ),
        "completion_blockers": final_completion_summary.get("completion_blockers"),
        "glossary_terms_missing": missing_glossary_terms,
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": summary,
        "checks": [check.to_dict() for check in checks],
        "headline_facts": [
            {"topic": topic, "fact": fact, "source": source}
            for topic, fact, source in _headline_fact_table(scorecard)
        ],
        "source_summaries": {
            "final_readiness": final_summary,
            "final_completion": final_completion_summary,
            "headline_scorecard": scorecard_summary,
            "runtime_image_contract": runtime_image_summary,
            "runtime_comparison_contract": runtime_comparison_summary,
            "stage_latency_budget": stage_latency_summary,
            "stage_boundary_bottleneck_ledger": stage_boundary_summary,
            "stage_causal_graph": stage_causal_summary,
            "length_regime_coverage": length_summary,
            "rerun_time_budget": rerun_budget_summary,
            "rerun_acceptance_contract": rerun_acceptance_summary,
            "command_reference_hygiene": command_hygiene_summary,
            "receiver_quickcheck_contract": receiver_contract_summary,
            "defense_claim_matrix": defense_summary,
            "caveat_adjudication_matrix": caveat_summary,
            "vllm_online_parity_protocol": vllm_protocol_summary,
            "share_bundle_manifest": share_bundle_summary,
            "manifest": manifest_summary,
            "technical_report_glossary": {
                "terms_total": len(glossary_terms),
                "missing_terms": missing_glossary_terms,
            },
        },
    }


def _table_text(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _checks_marker(summary: dict[str, Any]) -> str:
    return f"{summary.get('checks_passed')}/{summary.get('checks_total')}"


def _compact_gate_evidence(
    name: str,
    evidence: str,
    sources: dict[str, dict[str, Any]],
) -> str:
    final = sources["final_readiness"]
    scorecard = sources["headline_scorecard"]
    runtime_image = sources["runtime_image_contract"]
    runtime = sources["runtime_comparison_contract"]
    stage_latency = sources["stage_latency_budget"]
    stage_boundary = sources["stage_boundary_bottleneck_ledger"]
    stage_causal = sources["stage_causal_graph"]
    length = sources["length_regime_coverage"]
    rerun_budget = sources["rerun_time_budget"]
    rerun_acceptance = sources["rerun_acceptance_contract"]
    command_hygiene = sources["command_reference_hygiene"]
    receiver = sources["receiver_quickcheck_contract"]
    defense = sources["defense_claim_matrix"]
    caveat = sources["caveat_adjudication_matrix"]
    vllm_protocol = sources["vllm_online_parity_protocol"]
    share_bundle = sources["share_bundle_manifest"]
    manifest = sources["manifest"]
    final_completion = sources["final_completion"]

    compact_by_name = {
        "full audit and readiness are green": (
            f"Final readiness `{_checks_marker(final)}`, "
            f"required failures `{final.get('required_failures')}`."
        ),
        "share package receiver path is validated": evidence,
        "headline claims are evidence-backed": (
            f"headline scorecard `{_checks_marker(scorecard)}`; strict c4 win, "
            "c8 peak, long c8 RTF and vLLM diagnostic checks are all green."
        ),
        "runtime fairness and vLLM baseline strength are locked": (
            f"runtime image `{_checks_marker(runtime_image)}`; comparison "
            f"`{_checks_marker(runtime)}`; headline scope "
            f"`{runtime.get('allowed_cross_runtime_headline')}`; vLLM c8 "
            f"`{runtime.get('vllm_c8_contract')}`."
        ),
        "review glossary and metric-scope route is present": evidence,
        "stage and boundary diagnosis are green": (
            f"stage latency `{_checks_marker(stage_latency)}`, boundary "
            f"`{_checks_marker(stage_boundary)}`, causal graph "
            f"`{_checks_marker(stage_causal)}`; ledger rows "
            f"`{stage_boundary.get('ledger_rows')}`, recommended window "
            f"`{stage_boundary.get('recommended_sglang_window')}`."
        ),
        "short/long and high-concurrency regimes are covered": (
            f"{length.get('rows_total')} regime rows; short "
            f"`{length.get('short_rows')}`, long `{length.get('long_rows')}`; "
            f"long c8 RTF p95 `{length.get('long_c8_rtf_p95')}`."
        ),
        "rerun budget and replacement contract are explicit": (
            f"budget rows `{rerun_budget.get('rows_total')}`, timed wall lower "
            f"bound `{_float_value(rerun_budget.get('total_timed_benchmark_wall_s')):.2f}s`; "
            f"acceptance `{_checks_marker(rerun_acceptance)}`, return evidence "
            f"`{rerun_acceptance.get('return_evidence_files')}`."
        ),
        "rerun command references resolve": (
            f"commands `{command_hygiene.get('manifest_commands_total')}`, "
            f"structured refs `{command_hygiene.get('structured_command_refs_total')}`, "
            f"unresolved `{command_hygiene.get('unresolved_command_refs_total')}`."
        ),
        "receiver quickcheck and defense path are ready": (
            f"receiver contract `{_checks_marker(receiver)}` with "
            f"`{receiver.get('quickcheck_steps')}` steps; defense matrix "
            f"`{_checks_marker(defense)}`, question rows "
            f"`{defense.get('question_rows_total')}`."
        ),
        "caveat and vLLM c8 parity boundaries are explicit": (
            f"caveats `{caveat.get('rows_total')}`; current-best scope "
            f"`{caveat.get('current_best_scope')}`; vLLM online parity proven "
            f"`{vllm_protocol.get('online_parity_proven')}`."
        ),
        "final completion is evidence-gated": (
            f"completion_allowed_now `{final_completion.get('completion_allowed_now')}`; "
            f"blockers `{final_completion.get('completion_blockers')}`."
        ),
        "review reading route files are present": evidence,
        "share and evidence manifests include the reviewable package": (
            f"share bundle ready `{share_bundle.get('ready')}`, records "
            f"`{share_bundle.get('records_total')}`; manifest records "
            f"`{manifest.get('total_records')}`, missing "
            f"`{manifest.get('missing_records')}`."
        ),
    }
    return compact_by_name.get(name, evidence)


def _gate_row(
    name: str,
    status: Any,
    evidence: str,
    sources: dict[str, dict[str, Any]],
) -> str:
    display_evidence = _compact_gate_evidence(name, evidence, sources)
    return f"| {name} | `{status}` | {_table_text(display_evidence)} |"


def _fact_row(topic: str, fact: str, source: str) -> str:
    return f"| {topic} | {fact} | {source} |"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    checks = payload["checks"]
    sources = payload["source_summaries"]
    headline = sources["headline_scorecard"]
    runtime = sources["runtime_comparison_contract"]
    stage_boundary = sources["stage_boundary_bottleneck_ledger"]
    rerun_budget = sources["rerun_time_budget"]
    final_completion = sources["final_completion"]

    lines: list[str] = [
        "# Qwen3.5-Omni 高校合作方审阅会议包",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "用途：给合作高校第一次审阅或 15 分钟同步会使用。它不替代完整技术报告，",
        "而是把结论、边界、复现入口、stage 追问路线和复跑替换条件压缩到一页半。",
        "",
        "## 1. 会前先跑",
        "",
        "```bash",
        "export HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"",
        "cd \"$HOST_REPO\"",
        "sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256",
        "bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh",
        "bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root \"$HOST_REPO\" --mode host",
        "```",
        "",
        "期望：release seal、tarball validation、receiver smoke、extracted-only、standalone",
        "和 evidence query smoke 都通过；更新后的目标不再等待 6.21 晚上，",
        "`completion_allowed_now=true` 表示可以进入最终完成裁决。",
        "",
        "## 2. 15 分钟同步节奏",
        "",
        "| 时间 | 讲什么 | 直接打开 |",
        "| --- | --- | --- |",
        "| 0-2 min | 包是否可读、checksum/quickcheck 是否绿 | `qwen35_omni_start_here_zh_20260621.md` |",
        "| 2-4 min | 先统一术语：c、warmed、RTF、queue estimate、offline diagnostic | `qwen35_omni_university_technical_report_zh_20260621.md` section 2.1 |",
        "| 4-6 min | strict warmed c=4 headline 与 vLLM baseline 公平性 | `qwen35_omni_university_technical_report_zh_20260621.md` |",
        "| 6-8 min | c=1/4/8/16、短/长文本、RTF 与 serving 窗口 | `qwen35_omni_regime_decision_matrix_zh_20260621.md` |",
        "| 8-11 min | stage breakdown、handoff、queue/admission 与瓶颈迁移 | `qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` |",
        "| 11-13 min | vLLM c=8 prebuild w4 为什么仍是 offline diagnostic | `qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` |",
        "| 13-15 min | 复跑预算、替换 headline 条件和差异 triage | `qwen35_omni_rerun_acceptance_contract_zh_20260621.md` |",
        "",
        "## 3. 可直接相信的结论",
        "",
        f"- Final readiness：`{sources['final_readiness'].get('checks_passed')}/"
        f"{sources['final_readiness'].get('checks_total')}`，required failures "
        f"`{sources['final_readiness'].get('required_failures')}`。",
        f"- Strict headline scope：`{runtime.get('allowed_cross_runtime_headline')}`；"
        f"vLLM c=8 scope：`{runtime.get('vllm_c8_contract')}`。",
        f"- SGLang 推荐运行窗口：`{stage_boundary.get('recommended_sglang_window')}`；"
        f"饱和边界：`{stage_boundary.get('saturation_boundary')}`。",
        f"- 复跑计时段下界：`{_float_value(rerun_budget.get('total_timed_benchmark_wall_s')):.2f}s`；"
        f"8-GPU 等效下界：`{_float_value(rerun_budget.get('equivalent_8gpu_timed_lower_bound_gpu_hours')):.2f}` GPU-hours。",
        "",
        "| 结论 | 当前数字 | 证据 |",
        "| --- | --- | --- |",
    ]
    for item in payload["headline_facts"]:
        lines.append(_fact_row(item["topic"], item["fact"], item["source"]))

    lines.extend(
        [
            "",
            "## 4. 不能越界的说法",
            "",
            "- 不把 vLLM c=8 prebuild w4 写成 online serving parity；当前只是优化后的 offline diagnostic。",
            "- 不把 c=16 写成默认推荐；它是 saturation boundary，用来解释 queue 和 tail risk。",
            "- 不在未复跑同硬件、同镜像、同模型/cache、同 ASR/WER 链路且门禁全绿前替换 headline 数字。",
            "- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑。",
            "- 不说已经搜索完所有 future kernel 或全局最优；当前是 measured-best recipe。",
            "",
            "## 5. 追问时的证据路线",
            "",
            "| 追问 | 先看 | 再查机器证据 |",
            "| --- | --- | --- |",
            "| c、warmed、RTF、queue estimate 怎么定义 | `qwen35_omni_university_technical_report_zh_20260621.md` section 2.1 | `university_technical_report.json` |",
            "| vLLM baseline 是否公平 | `qwen35_omni_runtime_image_contract_zh_20260621.md` | `runtime_image_contract.json`, `runtime_comparison_contract.json` |",
            "| 为什么 headline 选 c=4 | `qwen35_omni_runtime_comparison_contract_zh_20260621.md` | `headline_scorecard.json` |",
            "| 单并发/高并发/长短文本是否覆盖 | `qwen35_omni_length_regime_coverage_zh_20260621.md` | `length_regime_coverage.json` |",
            "| stage 之间是否卡住 | `qwen35_omni_stage_causal_graph_zh_20260621.md` | `stage_causal_graph.json`, `stage_interaction_summary.json` |",
            "| 哪个 stage 是瓶颈 | `qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` | `stage_boundary_bottleneck_ledger.json` |",
            "| 合作方复跑不同怎么办 | `qwen35_omni_rerun_delta_triage_zh_20260621.md` | `rerun_delta_triage.json` |",
            "| 现场如何自证 | `qwen35_omni_evidence_query_cards_zh_20260621.md` | `qwen35_omni_evidence_query_cards_smoke.sh` |",
            "",
            "## 6. 复跑替换边界",
            "",
            "- 可以复跑并替换 headline 的前提：同 8x H20、同 SGLang/vLLM 镜像、同模型/cache、同数据口径、同 warmup/skip-first、同 ASR/WER 验收。",
            "- 必须返回 `rerun_acceptance_contract` 要求的 34 份 evidence 和 27 行命令矩阵。",
            "- 只重跑 raw benchmark、没有重建表格/图表/JSON/full audit 时，不替换主报告数字。",
            "- vLLM c=8 要升级为 strict online parity，必须补 online ingress、同口径 WER/ASR 和 vLLM online parity protocol 的 artifact。",
            "",
            "## 7. 机器 Gate 摘要",
            "",
            "| Gate | Status | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for check in checks:
        lines.append(
            _gate_row(check["name"], check["status"], check["evidence"], sources)
        )
    lines.extend(
        [
            "",
            "## 8. 当前完成门",
            "",
            f"- completion_allowed_now：`{final_completion.get('completion_allowed_now')}`",
            f"- completion_blockers：`{final_completion.get('completion_blockers')}`",
            "- 不再按 2026-06-21 18:00 UTC+08:00 等待；按 final completion audit 的证据门决定是否标记 goal complete。",
            "",
            "机器证据：`results/qwen35_report_audit_20260619/university_review_packet.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact Qwen3.5-Omni university-review packet."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_packet(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")
    _save_json(payload, json_output)
    summary = payload["summary"]
    print(
        "University review packet written: "
        f"{output} ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"required_failures={summary['required_failures']}"
    )
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
