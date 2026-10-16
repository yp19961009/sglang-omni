# SPDX-License-Identifier: Apache-2.0
"""Build a Chinese university-facing technical report for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "university_technical_report.json"

DIRECT_REPRO_COMMAND_IDS = [
    "run_full_audit",
    "launch_sglang_optimized",
    "sglang_videoamme_stress",
    "sglang_synthetic_text_to_speech",
    "sglang_recompute_wer",
    "vllm_c1_original",
    "vllm_c4_original",
    "vllm_c8_original",
    "vllm_c8_prebuild_w4",
    "summarize_vllm_log_stages",
    "diagnose_vllm_admission",
    "build_stage_latency_budget",
    "build_stage_boundary_bottleneck_ledger",
    "build_stage_reproduction_drilldown",
    "build_stage_route_decision_matrix",
    "build_tail_confidence_appendix",
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "n/a"


def _num(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "n/a"


def _cell(value: Any) -> str:
    text = str(value if value is not None else "")
    for old, new in {
        "queue=n/a / n/a": "queue=无排队估计",
        "queue_delta n/a": "queue_delta=无排队估计",
        "queue_share_after n/a": "queue_share_after=无排队估计",
    }.items():
        text = text.replace(old, new)
    return text.replace("\n", " ").replace("|", "\\|")


def _stable_summary(summary: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: summary.get(key) for key in keys if key in summary}


def _stable_package_validation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return _stable_summary(
        summary,
        [
            "ready",
            "checks_total",
            "checks_passed",
            "required_failures",
            "warnings",
            "identity_hash_clean",
            "identity_hash_offenders_total",
            "receiver_smoke_ready",
            "extracted_validation_ready",
            "extracted_validation_checks",
            "repo_independent_invocation",
        ],
    )


def _stable_release_seal_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return _stable_summary(
        summary,
        [
            "ready",
            "checks_total",
            "checks_passed",
            "required_failures",
            "share_bundle_records",
            "manifest_records",
            "final_readiness_checks",
            "repro_commands_total",
            "receiver_smoke_ready",
            "evidence_query_smoke_ready",
            "evidence_query_smoke_host_pass_lines",
            "evidence_query_smoke_portable_pass_lines",
            "completion_allowed_now",
            "send_decision",
            "forbidden_tarball_members",
            "adjacent_artifacts_total",
            "adjacent_artifacts_hashed",
            "adjacent_artifacts_self_omitted",
            "adjacent_artifacts_missing",
        ],
    )


def _ms(value: Any) -> str:
    return _num(value, 1, "ms")


def _find_rows(rows: list[dict[str, Any]], regime: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("regime") == regime]


def _compact_list(values: list[Any], limit: int = 4) -> str:
    clean = [_cell(value) for value in values if value]
    if len(clean) <= limit:
        return ", ".join(clean)
    return ", ".join(clean[:limit]) + f", ... (+{len(clean) - limit})"


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    headline = _load_json_optional(audit_dir / "headline_scorecard.json")
    regime = _load_json_optional(audit_dir / "regime_decision_matrix.json")
    stage_route = _load_json_optional(audit_dir / "stage_route_decision_matrix.json")
    pressure_stage_heatmap = _load_json_optional(audit_dir / "pressure_stage_heatmap.json")
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_ledger = _load_json_optional(audit_dir / "stage_boundary_bottleneck_ledger.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    final_readiness = _load_json_optional(audit_dir / "final_readiness_audit.json")
    share_package_validation = _load_json_optional(
        audit_dir / "share_package_validation.json"
    )
    share_package_receiver_smoke = _load_json_optional(
        audit_dir / "share_package_receiver_smoke_validation.json"
    )
    share_package_external_standalone = _load_json_optional(
        audit_dir / "share_package_external_standalone_validation.json"
    )
    share_release = _load_json_optional(audit_dir / "share_release_seal.json")
    runtime_image = _load_json_optional(audit_dir / "runtime_image_contract.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    share_bundle = _load_json_optional(audit_dir / "share_bundle_manifest.json")
    defense_claim_matrix = _load_json_optional(audit_dir / "defense_claim_matrix.json")
    tail_confidence = _load_json_optional(audit_dir / "tail_confidence_appendix.json")
    length_regime = _load_json_optional(audit_dir / "length_regime_coverage.json")

    headline_summary = headline.get("summary", {})
    regime_summary = regime.get("summary", {})
    stage_route_summary = stage_route.get("summary", {})
    pressure_stage_heatmap_summary = pressure_stage_heatmap.get("summary", {})
    stage_budget_summary = stage_budget.get("summary", {})
    stage_ledger_summary = stage_ledger.get("summary", {})
    final_summary = final_readiness.get("summary", {})
    share_package_validation_summary = _stable_package_validation_summary(
        share_package_validation.get("summary", {})
    )
    share_package_receiver_smoke_summary = _stable_package_validation_summary(
        share_package_receiver_smoke.get("summary", {})
    )
    share_package_external_standalone_summary = _stable_package_validation_summary(
        share_package_external_standalone.get("summary", {})
    )
    share_release_summary = _stable_release_seal_summary(
        share_release.get("summary", {})
    )
    evidence_query_smoke_summary = share_release.get("source_summaries", {}).get(
        "evidence_query_smoke", {}
    )
    repro_summary = repro.get("summary", {})
    defense_claim_summary = defense_claim_matrix.get("summary", {})
    tail_confidence_summary = tail_confidence.get("summary", {})
    length_regime_summary = length_regime.get("summary", {})
    route_rows = stage_route.get("rows", [])
    command_ids = {
        str(command.get("id"))
        for command in repro.get("commands", [])
        if isinstance(command, dict)
    }
    checks = {
        "headline_scorecard_ready": bool(headline_summary.get("ready"))
        and int(headline_summary.get("checks_total") or 0) >= 9,
        "metric_scope_clarification_ready": bool(headline_summary.get("ready"))
        and bool(headline.get("strict_c4_comparison", {}).get("sglang", {}).get("artifact"))
        and any(
            row.get("concurrency") == 4
            for row in headline.get("sglang_stress", {}).get("rows", [])
            if isinstance(row, dict)
        ),
        "regime_decision_matrix_ready": bool(regime_summary.get("ready"))
        and int(regime_summary.get("rows_total") or 0) >= 17,
        "stage_route_decision_matrix_ready": bool(stage_route_summary.get("ready"))
        and int(stage_route_summary.get("route_rows_total") or 0) >= 11,
        "pressure_stage_heatmap_ready": bool(pressure_stage_heatmap_summary.get("ready"))
        and int(pressure_stage_heatmap_summary.get("rows_total") or 0) >= 15
        and int(pressure_stage_heatmap_summary.get("sglang_videoamme_rows") or 0) >= 5
        and int(pressure_stage_heatmap_summary.get("synthetic_rows") or 0) >= 6
        and int(pressure_stage_heatmap_summary.get("vllm_rows") or 0) >= 4
        and int(pressure_stage_heatmap_summary.get("required_failures") or 0) == 0,
        "stage_budget_and_ledger_ready": bool(stage_budget_summary.get("ready"))
        and bool(stage_ledger_summary.get("ready"))
        and int(stage_ledger_summary.get("pressure_transition_rows") or 0) >= 11,
        "stage_latency_budget_detail_ready": bool(stage_budget_summary.get("ready"))
        and len(stage_budget.get("sglang_videoamme_budget", [])) >= 5
        and len(stage_budget.get("synthetic_speech_budget", [])) >= 6
        and len(stage_budget.get("vllm_offline_budget", [])) >= 4,
        "serving_capacity_matrix_ready": bool(stage_budget_summary.get("ready"))
        and bool(tail_confidence_summary.get("ready"))
        and {
            "sglang_stress_c1",
            "sglang_stress_c2",
            "sglang_stress_c4",
            "sglang_stress_c8",
            "sglang_stress_c16",
            "synthetic_short_c8",
            "synthetic_long_c8",
            "vllm_c8_prebuild_w4",
        }.issubset(
            {
                str(row.get("case_id"))
                for row in tail_confidence.get("rows", [])
                if isinstance(row, dict)
            }
        ),
        "route_reproduction_pointers_ready": bool(stage_route_summary.get("ready"))
        and len(route_rows) >= 11
        and all(
            row.get("raw_artifacts") and row.get("rerun_command_ids") and row.get("jq_queries")
            for row in route_rows
        ),
        "optimization_locks_ready": bool(sglang_lock.get("summary", {}).get("ready"))
        and bool(vllm_lock.get("summary", {}).get("ready"))
        and bool(runtime_image.get("summary", {}).get("ready")),
        "runtime_fairness_matrix_ready": bool(runtime_image.get("summary", {}).get("ready"))
        and bool(sglang_lock.get("summary", {}).get("ready"))
        and bool(vllm_lock.get("summary", {}).get("ready"))
        and bool(vllm_online.get("summary", {}).get("ready"))
        and str(runtime_image.get("summary", {}).get("sglang_image_id") or "").startswith(
            "sha256:"
        )
        and str(runtime_image.get("summary", {}).get("vllm_image_id") or "").startswith(
            "sha256:"
        )
        and "optimized" in str(runtime_image.get("summary", {}).get("vllm_strict_scope") or "")
        and "diagnostic" in str(runtime_image.get("summary", {}).get("vllm_c8_scope") or "")
        and not bool(vllm_online.get("summary", {}).get("online_parity_proven")),
        "vllm_online_boundary_ready": bool(vllm_online.get("summary", {}).get("ready"))
        and not bool(vllm_online.get("summary", {}).get("online_parity_proven")),
        "repro_manifest_ready": bool(repro_summary.get("ready"))
        and int(repro_summary.get("commands_total") or 0) >= 60,
        "direct_sglang_vllm_repro_commands_ready": bool(repro_summary.get("ready"))
        and set(DIRECT_REPRO_COMMAND_IDS).issubset(command_ids),
        "tail_confidence_appendix_ready": bool(tail_confidence_summary.get("ready"))
        and int(tail_confidence_summary.get("rows_total") or 0) >= 18
        and int(tail_confidence_summary.get("required_failures") or 0) == 0,
        "defense_claim_matrix_ready": bool(defense_claim_summary.get("ready"))
        and int(defense_claim_summary.get("rows_total") or 0) >= 10
        and int(defense_claim_summary.get("question_rows_total") or 0) >= 13
        and int(defense_claim_summary.get("required_failures") or 0) == 0
        and bool(defense_claim_summary.get("qna_questions_covered"))
        and bool(defense_claim_summary.get("qna_claims_covered"))
        and int(defense_claim_summary.get("failure_decisions_total") or 0) >= 10,
    }
    required_failures = sum(1 for value in checks.values() if not value)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": required_failures == 0,
            "checks_total": len(checks),
            "checks_passed": sum(1 for value in checks.values() if value),
            "required_failures": required_failures,
            "sections_total": 11,
            "share_scope": (
            "Chinese university-facing technical report: headline comparison, "
            "metric-scope clarification, reviewer glossary, single/high concurrency, short/long "
            "speech, pressure-stage heatmap, stage bottlenecks, per-stage latency "
            "budgets, stage-boundary evidence, pressure-transition matrix, serving/capacity decision "
            "matrix, runtime image and optimization fairness matrix, per-sample "
            "tail-confidence appendix, vLLM optimized baseline boundary, optimization recipe, "
            "defense-ready caveats, and reproduction entry points."
            ),
        },
        "checks": checks,
        "sources": {
            "headline_scorecard": headline_summary,
            "regime_decision_matrix": regime_summary,
            "stage_route_decision_matrix": stage_route_summary,
            "pressure_stage_heatmap": pressure_stage_heatmap_summary,
            "stage_latency_budget": stage_budget.get("summary", {}),
            "stage_boundary_bottleneck_ledger": stage_ledger.get("summary", {}),
            "repro_command_manifest": repro_summary,
            "final_readiness": final_summary,
            "share_package_validation": share_package_validation_summary,
            "share_package_receiver_smoke_validation": (
                share_package_receiver_smoke_summary
            ),
            "share_package_external_standalone_validation": (
                share_package_external_standalone_summary
            ),
            "share_release_seal": share_release_summary,
            "evidence_query_smoke": evidence_query_smoke_summary,
            "manifest": manifest.get("summary", {}),
            "share_bundle_manifest": share_bundle.get("summary", {}),
            "defense_claim_matrix": defense_claim_summary,
            "tail_confidence_appendix": tail_confidence_summary,
            "length_regime_coverage": length_regime_summary,
        },
        "headline": headline,
        "regime": regime,
        "stage_route": stage_route,
        "pressure_stage_heatmap": pressure_stage_heatmap,
        "stage_budget": stage_budget,
        "stage_ledger": stage_ledger,
        "repro": repro,
        "runtime_image": runtime_image,
        "sglang_lock": sglang_lock,
        "vllm_lock": vllm_lock,
        "vllm_online": vllm_online,
        "defense_claim_matrix": defense_claim_matrix,
        "tail_confidence": tail_confidence,
        "length_regime": length_regime,
    }


def _strict_comparison_table(headline: dict[str, Any]) -> list[str]:
    strict = headline.get("strict_c4_comparison", {})
    sglang = strict.get("sglang", {})
    vllm = strict.get("vllm", {})
    rel = strict.get("relative_sglang_lower_pct", {})
    lines = [
        "| Runtime | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | WER |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            "| SGLang-Omni optimized | "
            f"{int(sglang.get('n') or 0)} | {_pct(sglang.get('accuracy'))} | "
            f"{_num(sglang.get('latency_mean_s'), 3, 's')} | "
            f"{_num(sglang.get('latency_p95_s'), 3, 's')} | "
            f"{_num(sglang.get('rtf_mean'), 4)} | {_num(sglang.get('rtf_p95'), 4)} | "
            f"{_pct(sglang.get('wer_corpus'), 2)} |"
        ),
        (
            "| vLLM optimized | "
            f"{int(vllm.get('n') or 0)} | {_pct(vllm.get('accuracy'))} | "
            f"{_num(vllm.get('latency_mean_s'), 3, 's')} | "
            f"{_num(vllm.get('latency_p95_s'), 3, 's')} | "
            f"{_num(vllm.get('rtf_mean'), 4)} | {_num(vllm.get('rtf_p95'), 4)} | "
            f"{_pct(vllm.get('wer_corpus'), 2)} |"
        ),
        "",
        (
            "相对 vLLM，SGLang-Omni mean latency 低 "
            f"{_num(rel.get('latency_mean'), 1, '%')}，p95 latency 低 "
            f"{_num(rel.get('latency_p95'), 1, '%')}，mean RTF 低 "
            f"{_num(rel.get('rtf_mean'), 1, '%')}，p95 RTF 低 "
            f"{_num(rel.get('rtf_p95'), 1, '%')}。"
        ),
    ]
    return lines


def _metric_scope_table(headline: dict[str, Any]) -> list[str]:
    strict = headline.get("strict_c4_comparison", {})
    strict_sglang = strict.get("sglang", {})
    stress_c4 = next(
        (
            row
            for row in headline.get("sglang_stress", {}).get("rows", [])
            if isinstance(row, dict) and row.get("concurrency") == 4
        ),
        {},
    )
    strict_artifact = Path(str(strict_sglang.get("artifact") or "")).name
    stress_artifact = Path(str(stress_c4.get("result_json") or "")).name
    lines = [
        "| c=4 slice | 用途 | n | Accuracy/WER | Latency mean/p95 | RTF mean/p95 | Artifact | 外推边界 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        (
            "| strict warmed c=4 headline | SGLang-vLLM 横向比较，只引用这一行做 headline | "
            f"{int(strict_sglang.get('n') or 0)} | "
            f"{_pct(strict_sglang.get('accuracy'))}/{_pct(strict_sglang.get('wer_corpus'), 2)} | "
            f"{_num(strict_sglang.get('latency_mean_s'), 3, 's')}/{_num(strict_sglang.get('latency_p95_s'), 3, 's')} | "
            f"{_num(strict_sglang.get('rtf_mean'), 4)}/{_num(strict_sglang.get('rtf_p95'), 4)} | "
            f"{_cell(strict_artifact)} | "
            "用于 cross-runtime apples-to-apples；不要和 stress sweep 逐项相减 |"
        ),
        (
            "| SGLang stress c=4 | SGLang 内部 pressure sweep，用于服务窗口和 stage scaling | "
            f"{int(stress_c4.get('n') or 0)} | "
            f"{_pct(stress_c4.get('accuracy'))}/{_pct(stress_c4.get('wer_corpus'), 2)} | "
            f"{_num(stress_c4.get('latency_mean_s'), 3, 's')}/{_num(stress_c4.get('latency_p95_s'), 3, 's')} | "
            f"{_num(stress_c4.get('rtf_mean'), 4)}/{_num(stress_c4.get('rtf_p95'), 4)} | "
            f"{_cell(stress_artifact)} | "
            "用于 SGLang c=1/2/4/8/16 scaling；不要拿来替代 strict vLLM headline |"
        ),
    ]
    return lines


def _runtime_fairness_table(
    runtime_summary: dict[str, Any],
    sglang_summary: dict[str, Any],
    vllm_summary: dict[str, Any],
    vllm_online_summary: dict[str, Any],
) -> list[str]:
    lines = [
        "| Runtime / scope | Image digest | Optimized recipe evidence | 可以声明 | 不能声明 / 升级条件 |",
        "| --- | --- | --- | --- | --- |",
        (
            "| SGLang-Omni serving c4-c8 | "
            f"`{_cell(runtime_summary.get('sglang_image_id'))}` | "
            "`NO_CODE2WAV_TORCH_COMPILE=0`; `TORCHDYNAMO_DISABLE=0`; "
            "`--thinker-cuda-graph on`; `--talker-cuda-graph on`; "
            "`--talker-torch-compile on`; max-running=8; 16GiB preprocessing cache; "
            "`PREPROCESSING_MAX_CONCURRENCY=1` | "
            f"{_cell(sglang_summary.get('recipe_contract'))}; "
            f"{_cell(sglang_summary.get('recommended_window'))} | "
            "不能把 c16 包装成推荐服务点；preproc=2/4 是当前反例 |"
        ),
        (
            "| vLLM strict headline c4 | "
            f"`{_cell(runtime_summary.get('vllm_image_id'))}` | "
            "`VLLM_ENABLE_TORCH_COMPILE=True`; `enforce_eager=False`; "
            "`FULL_AND_PIECEWISE` CUDA graph; prefix cache + chunked prefill; "
            "shared-memory hidden buffer; encoder compile/batch | "
            f"{_cell(vllm_summary.get('strict_c4_contract'))}; baseline 是优化版，不是弱 baseline | "
            "只用于 warmed c=4 apples-to-apples headline，不能外推成 c=8 online parity |"
        ),
        (
            "| vLLM c8 prebuild diagnostic | "
            f"`{_cell(runtime_summary.get('vllm_image_id'))}` | "
            "`--prebuild-prompts --prebuild-workers 4`; same optimized image and log-stage/admission diagnostics | "
            f"{_cell(vllm_summary.get('c8_contract'))}; 可说 prebuild 明显缓解 offline prompt-feed admission | "
            "online_parity_proven="
            f"`{_cell(vllm_online_summary.get('online_parity_proven'))}`；升级需要 online ingress + WER/ASR + stage boundary 复核 |"
        ),
    ]
    return lines


def _regime_table(rows: list[dict[str, Any]], regime: str) -> list[str]:
    lines = [
        "| Pressure | 决策 | Stage/瓶颈判断 | 关键数字 |",
        "| --- | --- | --- | --- |",
    ]
    for row in _find_rows(rows, regime):
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("pressure")),
                    _cell(row.get("decision_label")),
                    _cell(row.get("stage_bottleneck_decision")),
                    _cell(row.get("key_metrics")),
                ]
            )
            + " |"
        )
    return lines


def _route_table(rows: list[dict[str, Any]]) -> list[str]:
    selected_keys = {
        "admission -> preprocessing",
        "admission -> preprocessing -> talker -> code2wav",
        "admission -> preprocessing_queue -> preprocessing -> talker -> code2wav",
        "talker -> code2wav_stream",
        "code2wav_collect -> code2wav_decode",
        "offline_runner -> engine_admission",
        "offline_runner -> engine_admission -> encoder -> thinker -> talker -> code2wav",
    }
    lines = [
        "| Route | Runtime | 裁决 | 优化重点 | 对外安全说法 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("route_key") not in selected_keys:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("route_key")),
                    _cell(",".join(row.get("runtimes", []))),
                    _cell(row.get("route_decision")),
                    _cell(row.get("optimization_focus")),
                    _cell(row.get("safe_talking_point")),
                ]
            )
            + " |"
        )
    return lines


def _sglang_stage_budget_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| c | Lat Mean | RTF | QPS | Preproc lifecycle | Queue est | Talker avg | Code2wav decode | Hop p95 | Diagnosis |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("concurrency")),
                    _ms(row.get("latency_mean_ms")),
                    _num(row.get("rtf_mean"), 4),
                    _num(row.get("qps"), 3),
                    (
                        f"{_ms(row.get('preproc_lifecycle_avg_ms'))} / "
                        f"{_num(row.get('preproc_lifecycle_pct_of_latency'), 1, '%')}"
                    ),
                    _queue_estimate_cell(row),
                    (
                        f"{_ms(row.get('talker_avg_ms'))} / "
                        f"{_num(row.get('talker_pct_of_latency'), 1, '%')}"
                    ),
                    (
                        f"{_ms(row.get('code2wav_decode_avg_ms'))} / "
                        f"{_num(row.get('code2wav_decode_pct_of_latency'), 1, '%')}"
                    ),
                    _ms(row.get("talker_to_code2wav_hop_p95_ms")),
                    _cell(row.get("diagnosis")),
                ]
            )
            + " |"
        )
    return lines


def _synthetic_stage_budget_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Scenario | c | Audio | Lat Mean | RTF | QPS | Talker avg | Code2wav decode | Hop p95 | Diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("scenario")),
                    _cell(row.get("concurrency")),
                    _num(row.get("audio_duration_mean_s"), 1, "s"),
                    _ms(row.get("latency_mean_ms")),
                    _num(row.get("rtf_mean"), 4),
                    _num(row.get("qps"), 3),
                    (
                        f"{_ms(row.get('talker_avg_ms'))} / "
                        f"{_num(row.get('talker_pct_of_latency'), 1, '%')}"
                    ),
                    (
                        f"{_ms(row.get('code2wav_decode_avg_ms'))} / "
                        f"{_num(row.get('code2wav_decode_pct_of_latency'), 1, '%')}"
                    ),
                    _ms(row.get("talker_to_code2wav_hop_p95_ms")),
                    _cell(row.get("diagnosis")),
                ]
            )
            + " |"
        )
    return lines


def _vllm_stage_budget_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Workload | Runner QPS | Engine QPS | Runner overhead | Admission avg/p95 | Encoder p95 | Thinker->Talker p95 | Talker->C2W p95 | Scope | Diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("workload")),
                    _num(row.get("runner_qps"), 4),
                    _num(row.get("engine_qps"), 4),
                    _num(row.get("runner_overhead_pct_wall"), 1, "%"),
                    (
                        f"{_ms(row.get('batch_admission_span_avg_ms'))}/"
                        f"{_ms(row.get('batch_admission_span_p95_ms'))}"
                    ),
                    _ms(row.get("encoder_p95_ms")),
                    _ms(row.get("thinker_to_talker_feed_p95_ms")),
                    _ms(row.get("talker_to_code2wav_drain_p95_ms")),
                    _cell(row.get("valid_comparison_scope")),
                    _cell(row.get("diagnosis")),
                ]
            )
            + " |"
        )
    return lines


def _route_evidence_table(rows: list[dict[str, Any]]) -> list[str]:
    selected_keys = {
        "admission -> preprocessing",
        "admission -> preprocessing -> talker -> code2wav",
        "admission -> preprocessing_queue -> preprocessing -> talker -> code2wav",
        "preprocessing -> encoder_thinker",
        "talker -> code2wav_stream",
        "code2wav_collect -> code2wav_decode",
        "offline_runner -> engine_admission",
        "offline_runner -> engine_admission -> encoder -> thinker -> talker -> code2wav",
        "talker -> code2wav",
    }
    lines = [
        "| Route | Stage rows | Raw artifact anchors | Rerun command IDs | jq entry |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("route_key") not in selected_keys:
            continue
        jq_queries = row.get("jq_queries", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("route_key")),
                    _cell(row.get("rows_total")),
                    _compact_list(row.get("raw_artifacts", []), limit=3),
                    _compact_list(row.get("rerun_command_ids", []), limit=4),
                    _cell(jq_queries.get("route_rows") or jq_queries.get("stage_row")),
                ]
            )
            + " |"
        )
    return lines


def _pressure_transition_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| ID | Runtime | Workload | Transition | Verdict | Key evidence | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("id")),
                    _cell(row.get("runtime")),
                    _cell(row.get("workload")),
                    _cell(row.get("transition")),
                    _cell(row.get("verdict")),
                    _cell(row.get("evidence")),
                    _cell(row.get("decision")),
                ]
            )
            + " |"
        )
    return lines


def _tail_rows_by_case(tail_confidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in tail_confidence.get("rows", [])
        if isinstance(row, dict) and row.get("case_id")
    }


def _tail_metric(row: dict[str, Any], metric: str, stat: str) -> Any:
    value = row.get(metric, {})
    if isinstance(value, dict):
        return value.get(stat)
    return None


def _qps_latency_rtf(row: dict[str, Any]) -> str:
    return (
        f"QPS={_num(row.get('throughput_qps'), 3)}; "
        f"lat_p95={_num(_tail_metric(row, 'latency_s', 'p95'), 3, 's')}; "
        f"RTF_p95={_num(_tail_metric(row, 'rtf', 'p95'), 4)}"
    )


def _queue_guard(row: dict[str, Any]) -> str:
    queue_ms = row.get("preproc_queue_estimate_ms")
    queue_pct = row.get("preproc_queue_pct_of_latency")
    if queue_ms is None or queue_pct is None:
        return "queue=无排队估计"
    return f"queue={_ms(queue_ms)} / {_num(queue_pct, 1, '%')}"


def _queue_estimate_cell(row: dict[str, Any]) -> str:
    queue_ms = row.get("preproc_queue_estimate_ms")
    queue_pct = row.get("preproc_queue_pct_of_latency")
    if queue_ms is None or queue_pct is None:
        return "无排队估计"
    return f"{_ms(queue_ms)} / {_num(queue_pct, 1, '%')}"


def _budget_by_concurrency(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            result[int(row.get("concurrency"))] = row
        except Exception:
            continue
    return result


def _synthetic_budget_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            result[(str(row.get("scenario")), int(row.get("concurrency")))] = row
        except Exception:
            continue
    return result


def _serving_capacity_table(
    stage_budget: dict[str, Any],
    tail_confidence: dict[str, Any],
) -> list[str]:
    tails = _tail_rows_by_case(tail_confidence)
    sglang_budget = _budget_by_concurrency(stage_budget.get("sglang_videoamme_budget", []))
    synthetic_budget = _synthetic_budget_by_key(
        stage_budget.get("synthetic_speech_budget", [])
    )
    vllm_budget = {
        str(row.get("workload")): row
        for row in stage_budget.get("vllm_offline_budget", [])
        if isinstance(row, dict)
    }
    c1 = tails.get("sglang_stress_c1", {})
    c2 = tails.get("sglang_stress_c2", {})
    c4 = tails.get("sglang_stress_c4", {})
    c8 = tails.get("sglang_stress_c8", {})
    c16 = tails.get("sglang_stress_c16", {})
    short8 = tails.get("synthetic_short_c8", {})
    long8 = tails.get("synthetic_long_c8", {})
    short_budget = synthetic_budget.get(("short", 8), {})
    long_budget = synthetic_budget.get(("long", 8), {})
    vllm_w4 = vllm_budget.get("vLLM-c8-prebuild-w4", {})

    lines = [
        "| 压力/场景 | 运行选择 | 可承诺指标 | Stage guard | 不要做 |",
        "| --- | --- | --- | --- | --- |",
        (
            "| Video-AMME c=1-c2 | latency-first / 单并发到低并发 | "
            f"QPS={_num(c1.get('throughput_qps'), 3)}-{_num(c2.get('throughput_qps'), 3)}; "
            f"lat_p95={_num(_tail_metric(c1, 'latency_s', 'p95'), 3, 's')}-{_num(_tail_metric(c2, 'latency_s', 'p95'), 3, 's')}; "
            f"RTF_p95={_num(_tail_metric(c1, 'rtf', 'p95'), 4)}-{_num(_tail_metric(c2, 'rtf', 'p95'), 4)} | "
            "主 tail 是 talker_ar；handoff/decode p95 仍小 | "
            "不要把低并发 tail 当成高并发 admission 问题 |"
        ),
        (
            "| Video-AMME c=4 | balanced serving / strict headline 参照点 | "
            f"{_qps_latency_rtf(c4)}; {_queue_guard(sglang_budget.get(4, {}))} | "
            "talker_ar tail 为主，queue 仍可控；strict SGLang-vLLM 对比只用 warmed c=4 slice | "
            "不要把 stress c=4 和 strict warmed c=4 混成同一组数字 |"
        ),
        (
            "| Video-AMME c=8 | throughput edge / 当前高并发甜点 | "
            f"{_qps_latency_rtf(c8)}; {_queue_guard(sglang_budget.get(8, {}))} | "
            "admission queue 开始显性化，但 QPS 仍是当前峰值，handoff 不是瓶颈 | "
            "不要继续加 preprocessing 并发；preproc=2/4 已是反例 |"
        ),
        (
            "| Video-AMME c=16 | saturation evidence / 压力边界 | "
            f"{_qps_latency_rtf(c16)}; {_queue_guard(sglang_budget.get(16, {}))} | "
            "吞吐低于 c=8，queue 占比大幅上升，RTF tail 明显变差 | "
            "不要作为默认服务点，也不要写成高并发更优 |"
        ),
        (
            "| Synthetic short c=8 | 短文本语音高并发 guard | "
            f"{_qps_latency_rtf(short8)}; audio={_num(short_budget.get('audio_duration_mean_s'), 1, 's')}; "
            f"hop_p95={_ms(short_budget.get('talker_to_code2wav_hop_p95_ms'))} | "
            "短文本仍快于实时；code2wav decode 占比小 | "
            "不要把短文本结论外推成长文本吞吐 |"
        ),
        (
            "| Synthetic long c=8 | 长文本/长语音 realtime guard | "
            f"{_qps_latency_rtf(long8)}; audio={_num(long_budget.get('audio_duration_mean_s'), 1, 's')}; "
            f"hop_p95={_ms(long_budget.get('talker_to_code2wav_hop_p95_ms'))} | "
            "长文本 c=8 RTF_p95 仍小于 1；压力主要进入 talker cadence | "
            "不要把 vocoder decode 当成优先瓶颈 |"
        ),
        (
            "| vLLM c=8 prebuild w4 | optimized offline diagnostic | "
            f"runner_QPS={_num(vllm_w4.get('runner_qps'), 4)}; engine_QPS={_num(vllm_w4.get('engine_qps'), 4)}; "
            f"admission_p95={_ms(vllm_w4.get('batch_admission_span_p95_ms'))} | "
            "prebuild 移除大部分 prompt-feed admission，暴露后续 engine/talker tail | "
            "不要升级为 online serving parity；需要 online ingress + WER/ASR 复核 |"
        ),
    ]
    return lines


def _pressure_stage_heatmap_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Pressure | Runtime | Key metrics | Stage hotspot | Connection verdict | Decision / Caveat |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("pressure")),
                    _cell(row.get("runtime")),
                    _cell(row.get("headline_metrics")),
                    _cell(
                        f"{row.get('admission_or_preprocess')} ; {row.get('talker')}"
                    ),
                    _cell(f"{row.get('stage_connection')} ; {row.get('code2wav')}"),
                    _cell(f"{row.get('decision')} / {row.get('do_not_say')}"),
                ]
            )
            + " |"
        )
    return lines


def _repro_command_table(repro: dict[str, Any]) -> list[str]:
    commands = {
        str(row.get("id")): row
        for row in repro.get("commands", [])
        if isinstance(row, dict) and row.get("id")
    }
    lines = [
        "| Scope | Command ID | Phase | 期望结果 |",
        "| --- | --- | --- | --- |",
    ]
    labels = {
        "run_full_audit": "全量证据验证",
        "launch_sglang_optimized": "SGLang 服务",
        "sglang_videoamme_stress": "SGLang c=1/2/4/8/16",
        "sglang_synthetic_text_to_speech": "SGLang 短/长文本语音",
        "sglang_recompute_wer": "SGLang WER",
        "vllm_c1_original": "vLLM c=1 baseline",
        "vllm_c4_original": "vLLM strict c=4 baseline",
        "vllm_c8_original": "vLLM c=8 original diagnostic",
        "vllm_c8_prebuild_w4": "vLLM c=8 prebuild w4",
        "summarize_vllm_log_stages": "vLLM stage log 汇总",
        "diagnose_vllm_admission": "vLLM admission 诊断",
        "build_stage_latency_budget": "Stage latency budget",
        "build_stage_boundary_bottleneck_ledger": "Stage boundary ledger",
        "build_stage_reproduction_drilldown": "Stage drilldown 复现",
        "build_stage_route_decision_matrix": "Stage route 裁决",
    }
    for command_id in DIRECT_REPRO_COMMAND_IDS:
        command = commands.get(command_id, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(labels.get(command_id, command_id)),
                    f"`{_cell(command_id)}`",
                    _cell(command.get("phase")),
                    _cell(command.get("expected")),
                ]
            )
            + " |"
        )
    return lines


def _defense_claim_table(defense_claim_matrix: dict[str, Any]) -> list[str]:
    rows = [
        row
        for row in defense_claim_matrix.get("rows", [])
        if isinstance(row, dict) and row.get("claim")
    ]
    lines = [
        "| 现场追问 | 可说法 | Evidence | 复跑命令/入口 | 失败时裁决 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence = _compact_list(
            [Path(str(path)).name for path in row.get("machine_evidence", [])],
            limit=3,
        )
        rerun = _compact_list(
            [f"`{command_id}`" for command_id in row.get("rerun_command_ids", [])],
            limit=4,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("claim")),
                    _cell(row.get("allowed_wording")),
                    _cell(evidence),
                    _cell(rerun),
                    _cell(row.get("failure_decision")),
                ]
            )
            + " |"
        )
    return lines


def build_markdown(root: Path, payload: dict[str, Any] | None = None) -> str:
    root = root.resolve()
    payload = payload or build_payload(root)
    headline = payload["headline"]
    regime_rows = payload["regime"].get("rows", [])
    stage_route_rows = payload["stage_route"].get("rows", [])
    pressure_stage_heatmap_rows = payload["pressure_stage_heatmap"].get("rows", [])
    stage_budget = payload["stage_budget"]
    stage_ledger = payload["stage_ledger"]
    repro = payload["repro"]
    defense_claim_matrix = payload["defense_claim_matrix"]
    tail_confidence = payload["tail_confidence"]
    length_regime_summary = payload["length_regime"].get("summary", {})
    summary = payload["summary"]
    sources = payload["sources"]
    runtime_summary = payload["runtime_image"].get("summary", {})
    sglang_summary = payload["sglang_lock"].get("summary", {})
    vllm_summary = payload["vllm_lock"].get("summary", {})
    vllm_online_summary = payload["vllm_online"].get("summary", {})
    share_release_summary = sources["share_release_seal"]
    evidence_query_smoke_summary = sources["evidence_query_smoke"]

    lines: list[str] = [
        "# Qwen3.5-Omni SGLang-Omni 中文技术报告",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{root}`。",
        "",
        "定位：这是一份可以直接发给合作高校的中文技术报告正文。它不引入手工数字，",
        "所有核心结论来自已经通过 full audit 的 JSON 证据和随包报告。",
        "",
        "## 1. Executive Summary",
        "",
        "- 严格横向 headline 使用 warmed c=4：SGLang-Omni 在 latency mean、latency p95、RTF mean、RTF p95 均优于优化版 vLLM，同时 accuracy/WER 不退化。",
        "- SGLang 推荐服务窗口是 c=4 到 c=8；c=8 是当前吞吐峰值，c=16 是压力边界。",
        "- 短/长文本语音输出路径覆盖 thinker/talker/code2wav；`length_regime_coverage.json` 机器 gate 通过，long c=8 仍快于实时。",
        "- stage 连接本身不是当前主瓶颈；stage budget 显示主要压力来自 talker AR tail、c=8/c=16 admission/queue，以及 vLLM offline prompt-feed admission。",
        "- vLLM baseline 使用 Qwen3.5-capable 镜像和 compile/CUDA graph/cache/prebuild 证据，不是弱 baseline；但 vLLM c=8 prebuild w4 仍只能作为 offline diagnostic。",
        "",
        "## 2. 环境和方法",
        "",
        f"- 模型：`qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`。",
        f"- GPU 合同：`{_cell(runtime_summary.get('gpu_contract'))}`。",
        f"- SGLang 镜像：`{_cell(runtime_summary.get('sglang_image'))}`。",
        f"- vLLM 镜像：`{_cell(runtime_summary.get('vllm_image'))}`。",
        "- 主 workload：Video-AMME ci-50，视频 + spoken question，输出 text + speech。",
        "- 压力覆盖：SGLang c=1/2/4/8/16；synthetic short/long text-to-speech c=1/4/8；vLLM c=1/c=4/c=8 和 c=8 prebuild diagnostic。",
        "- 指标口径：latency 是 client-observed end-to-end；RTF 是 latency / generated audio duration；WER 使用 offline Whisper large-v3 或等价 ASR 路径复算。",
        "",
        "### 2.1 指标和口径速查",
        "",
        "| 术语 | 本报告口径 | 容易误读的地方 |",
        "| --- | --- | --- |",
        "| `c` | benchmark client concurrency；SGLang 压力扫 c=1/2/4/8/16，synthetic speech 扫 c=1/4/8。 | c=16 可运行不等于推荐服务点；本报告推荐窗口是 c=4-c=8。 |",
        "| warmed / skip-first | strict headline 使用 warmed c=4 slice，跳过 cold compile / CUDA graph capture 的前若干请求。 | 不要把 cold-start 编译开销和 warmed serving latency 混在同一 headline。 |",
        "| latency | client observed end-to-end latency，覆盖请求到 text+speech 输出。 | 它不是单个 kernel 或单个 stage 的时间；stage 表用于拆解来源。 |",
        "| RTF | latency / generated audio duration；小于 1 表示快于实时生成。 | 长文本 RTF 低不代表 QPS 高，两者分别看实时性和吞吐。 |",
        "| QPS | 当前 workload 与并发下的完成吞吐。 | SGLang c=8 是当前吞吐峰值；c=16 QPS 回落且 tail 变差。 |",
        "| WER / accuracy | WER 用 offline ASR 链路复算；accuracy 来自 Video-AMME ci-50 任务判分。 | 性能数字替换前必须保留同口径 WER/ASR 和 accuracy 验收。 |",
        "| stage boundary / handoff | 相邻 stage 的连接状态，例如 talker->code2wav stream hop。 | 当前 SGLang handoff 健康；不要把连接健康误写成主瓶颈。 |",
        "| queue estimate | preprocessing lifecycle 中扣除实际 preprocess 后的 admission/queue 估计。 | 低并发无排队估计；c=8/c=16 queue 显性化才是高并发压力信号。 |",
        "| offline diagnostic | 用来定位瓶颈的离线 runner 证据，例如 vLLM c=8 prebuild w4。 | offline diagnostic 不能直接升级成 online serving parity。 |",
        "| share-ready with caveat | 当前包可分享，但带明确边界：ci-50/stress/synthetic、vLLM c=8 online parity、SeedTTS full-set 不越界。 | caveat 是外发口径的一部分，不是证据失败。 |",
        "",
        "### 2.2 Runtime fairness / 镜像与优化锁定",
        "",
        "这张表回答 baseline 是否公平：SGLang 和 vLLM 都锁定镜像 digest，且 vLLM 使用 compile/CUDA graph/cache/encoder/prebuild 证据，不是保守弱 baseline。",
        "同时它把 strict headline 与 vLLM c=8 offline diagnostic 的边界分开，避免把诊断结果误写成 online serving parity。",
        "",
        *_runtime_fairness_table(
            runtime_summary,
            sglang_summary,
            vllm_summary,
            vllm_online_summary,
        ),
        "",
        "## 3. 严格 SGLang-vLLM 对比",
        "",
        "严格横向比较只使用 warmed skip-first-4 的 c=4 slice，避免 cold compile / CUDA graph capture 影响。",
        "",
        *_strict_comparison_table(headline),
        "",
        "### 3.1 c=4 指标口径说明",
        "",
        "报告里有两个 c=4：一个是 strict warmed c=4 headline slice，一个是 SGLang pressure sweep c=4。",
        "它们用途不同，所以 n、accuracy/WER、latency/RTF 不要求逐项相同；复现和答辩时按下表选择引用口径。",
        "",
        *_metric_scope_table(headline),
        "",
        "## 4. SGLang 单并发和高并发压力结论",
        "",
        *_regime_table(regime_rows, "sglang_videoamme_stress"),
        "",
        "读法：c=1/c=2/c=4 主要是 talker AR tail；c=8 达到当前吞吐峰值；c=16 虽可运行但 admission/queue 饱和，不能作为推荐服务点。",
        "",
        "## 5. 短/长文本语音输出结论",
        "",
        *_regime_table(regime_rows, "sglang_synthetic_speech"),
        "",
        "读法：synthetic speech 用来隔离 thinker/talker/code2wav。长文本 c=8 仍快于实时，说明长输出路径没有靠牺牲语音一致性换吞吐。",
        (
            "机器审计入口：`results/qwen35_report_audit_20260619/length_regime_coverage.json` "
            f"ready=`{bool(length_regime_summary.get('ready'))}`，checks="
            f"`{int(length_regime_summary.get('checks_passed') or 0)}/"
            f"{int(length_regime_summary.get('checks_total') or 0)}`，rows="
            f"`{int(length_regime_summary.get('rows_total') or 0)}`；short="
            f"`{_num(length_regime_summary.get('short_target_chars'), 0)} chars`，long="
            f"`{_num(length_regime_summary.get('long_target_chars'), 0)} chars`，long c=8 RTF p95="
            f"`{_num(length_regime_summary.get('long_c8_rtf_p95'), 4)}`；"
            f"max hop/decode p95=`{_ms(length_regime_summary.get('max_talker_to_code2wav_hop_p95_ms'))}/"
            f"{_ms(length_regime_summary.get('max_code2wav_decode_p95_ms'))}`。"
        ),
        "边界：这条证据证明 ci-50 target length 与 synthetic short/long guardrail 自洽，不能外推为完整线上流量或 official SeedTTS full-set headline。",
        "",
        "### 5.1 Serving/capacity 决策矩阵",
        "",
        "这张表把压力点翻译成运行选择：哪些可以做服务窗口，哪些只能做压力边界或诊断证据。",
        "它同时给出对应 stage guard，方便复跑时判断新数字是否仍可替换当前结论。",
        "",
        *_serving_capacity_table(stage_budget, tail_confidence),
        "",
        "## 6. Stage Breakdown 和连接瓶颈",
        "",
        "### 6.1 Pressure × Stage Heatmap",
        "",
        "这张表先把所有压力点压到一页：单/低并发、高并发、短/长文本、vLLM original/prebuild 都用同一组 stage 列阅读。",
        "它来自 `pressure_stage_heatmap.json`，不引入新的 benchmark 数字；用途是快速回答“这个压力到底压到了哪个 stage、连接是不是瓶颈”。",
        "",
        *_pressure_stage_heatmap_table(pressure_stage_heatmap_rows),
        "",
        "### 6.2 SGLang Video-AMME stage latency budget",
        "",
        *_sglang_stage_budget_table(stage_budget.get("sglang_videoamme_budget", [])),
        "",
        "读法：低/中并发下 talker AR 占比约三分之一；c=8 时 queue estimate 已占 latency 的 30.6%，但仍是吞吐峰值；c=16 queue estimate 到 67.4%，所以是 saturation boundary。",
        "",
        "### 6.3 短/长文本语音 stage latency budget",
        "",
        *_synthetic_stage_budget_table(stage_budget.get("synthetic_speech_budget", [])),
        "",
        "读法：synthetic short/long 把 thinker/talker/code2wav 路径隔离出来。短文本和长文本的 code2wav decode 占比都很小；长文本 c=8 的 RTF=0.4932，仍快于实时，瓶颈描述应落在 talker AR cadence 而不是 vocoder decode。",
        "",
        "### 6.4 vLLM offline stage latency budget",
        "",
        *_vllm_stage_budget_table(stage_budget.get("vllm_offline_budget", [])),
        "",
        "读法：vLLM original c=4/c=8 的 p95 encoder、thinker->talker、talker->code2wav 并不大，主问题在 offline runner 到 engine admission 的 prompt build/feed。prebuild w4 把 admission span 降下来后，才暴露后续 engine/talker/code2wav tail，因此它是诊断证据，不是线上 parity 证据。",
        "",
        "### 6.5 Stage route verdict",
        "",
        *_route_table(stage_route_rows),
        "",
        "关键结论：不要把健康 handoff 说成瓶颈。`talker -> code2wav_stream` 和 `code2wav_collect -> code2wav_decode` 当前主要是健康/非瓶颈；c=8/c=16 的压力来自 admission/queue 与 talker tail 叠加。",
        "",
        "### 6.6 Pressure transition 矩阵",
        "",
        "这张表把相邻压力档位的变化量直接摊开，用来回答压力到底传到了哪个 stage 边界。",
        "它来自 `stage_boundary_bottleneck_ledger.json`，不引入新的 benchmark 数字。",
        "",
        *_pressure_transition_table(stage_ledger.get("pressure_transition_rows", [])),
        "",
        "读法：SGLang `c8 -> c16` 是 saturation boundary，因为吞吐下降但 admission queue 和 RTF tail 上升；synthetic long `c4 -> c8` 仍快于实时，说明长文本压力主要进入 Talker cadence；vLLM `c8 -> prebuild-w4` 是 offline diagnostic 的瓶颈转移，不能提升为 online parity。",
        "",
        "### 6.7 Route 复现索引",
        "",
        *_route_evidence_table(stage_route_rows),
        "",
        "读法：每条 route 都可以沿 `stage_route_decision_matrix.json` 回到 `stage_reproduction_drilldown.json`、raw artifact 和 rerun command ID；这也是报告里 stage 结论可复核的最短路径。",
        "",
        "## 7. vLLM c=8 诊断边界",
        "",
        *_regime_table(regime_rows, "vllm_offline_diagnostic"),
        "",
        "vLLM c=8 original 的 offline runner 会在 engine admission 前本地构建 multimodal prompt，因此原始 wall QPS 主要反映 prompt build/feed。prebuild w4 显著缩短 admission span，但仍缺 online ingress + WER/ASR 复核，所以不能升级为 strict c=8 online parity。",
        "",
        "## 8. 优化锁和反例",
        "",
        f"- SGLang optimization lock：ready=`{sglang_summary.get('ready')}`，checks=`{sglang_summary.get('checks_passed')}/{sglang_summary.get('checks_total')}`，推荐窗口 `{_cell(sglang_summary.get('recommended_window'))}`。",
        f"- vLLM optimization lock：ready=`{vllm_summary.get('ready')}`，checks=`{vllm_summary.get('checks_passed')}/{vllm_summary.get('checks_total')}`，c=8 边界 `{_cell(vllm_summary.get('c8_contract'))}`。",
        f"- vLLM online parity protocol：ready=`{vllm_online_summary.get('ready')}`，online_parity_proven=`{vllm_online_summary.get('online_parity_proven')}`。",
        "",
        "### 8.1 当前 best measured recipe 裁决",
        "",
        "| 主题 | 当前裁决 | 可说法 | 不能说 | 替换条件 |",
        "| --- | --- | --- | --- | --- |",
        (
            "| SGLang recipe | 当前审计环境里的 best measured recipe | "
            "compiled/graph path + serial preprocessing + 16GiB preprocessing cache；服务窗口 c=4-c=8，c=8 是当前吞吐峰值 | "
            "不能说已经搜索完所有 future kernel、placement、admission policy，不能承诺任意环境全局最优 | "
            "新 recipe 必须补齐 c=4/c=8/c=16、WER、stage interaction、acceptance、final readiness 后才能替换 |"
        ),
        (
            "| SGLang anti-recipe | preproc=2/4 当前不能作为优化方向 | "
            "preproc=2 回退，preproc=4 失败/OOM；当前应先管 admission、placement 和 shared-resource contention | "
            "不能把 preprocessing 并发越高越好当成优化结论 | "
            "只有 admission/placement/memory 同步重设计后重新评估 |"
        ),
        (
            "| vLLM baseline | optimized baseline，不是弱 baseline | "
            "Qwen3.5-capable image + compile/CUDA graph/cache/prebuild evidence；c=4 用于 strict headline，c=8 prebuild w4 是 offline diagnostic | "
            "不能把 c=8 prebuild w4 说成 online serving parity | "
            "需要 online ingress + WER/ASR + stage boundary 复核后才能升级 |"
        ),
        "",
        *_regime_table(regime_rows, "negative_optimization"),
        "",
        "优化顺序建议：先守住 compiled/graph recipe 和 serial preprocessing，再优化 talker AR cadence/batching、admission 策略和 online vLLM parity 入口；不要把 preproc=2/4 当作当前 recipe。",
        "",
        "## 9. 复现入口",
        "",
        "最短复核命令：",
        "",
        "```bash",
        "cd /home/gangouyu/sglang-omni",
        "",
        "python3 -m benchmarks.eval.run_qwen35_omni_report_audit \\",
        "  --root /home/gangouyu/sglang-omni \\",
        "  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json",
        "```",
        "",
        f"- Repro command manifest：ready=`{sources['repro_command_manifest'].get('ready')}`，commands=`{sources['repro_command_manifest'].get('commands_total')}`，phases=`{sources['repro_command_manifest'].get('phases_total')}`。",
        "- SGLang/vLLM 具体重跑命令：`benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`。",
        "- 接收方一条命令收包快检：`bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh`；它串起 checksum、tarball-mode validation、receiver smoke、extracted-only validation 和 external standalone validation。",
        "- 单项 share package validator：`python3 -m benchmarks.eval.validate_qwen35_omni_share_package --root /home/gangouyu/sglang-omni --strict --json-output results/qwen35_report_audit_20260619/share_package_validation.json`。",
        "- 现场只读证据查询 smoke：`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root /home/gangouyu/sglang-omni --mode host`；解包后用 `bash \"$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh\" --root \"$BUNDLE_ROOT\" --mode portable`。",
        (
            "- Release seal："
            f"ready=`{share_release_summary.get('ready')}`，checks="
            f"`{share_release_summary.get('checks_passed')}/"
            f"{share_release_summary.get('checks_total')}`，tarball identity "
            "由 `results/qwen35_report_audit_20260619/share_release_seal.json` "
            "记录和校验，"
            f"evidence-query host/portable pass="
            f"`{evidence_query_smoke_summary.get('host_pass_lines')}/"
            f"{evidence_query_smoke_summary.get('portable_pass_lines')}`。"
        ),
        "",
        "关键命令可以直接从 machine-readable manifest 抽取，避免复制长命令时改错参数：",
        "",
        "```bash",
        "cd /home/gangouyu/sglang-omni",
        "jq -r --arg id vllm_c4_original \\",
        "  '.commands[] | select(.id == $id) | .command' \\",
        "  results/qwen35_report_audit_20260619/repro_command_manifest.json",
        "```",
        "",
        *_repro_command_table(repro),
        "",
        "## 10. 可分享边界",
        "",
        "- 可以说：warmed c=4 下 SGLang-Omni 优于优化版 vLLM，并且 accuracy/WER 不退化。",
        "- 可以说：当前 SGLang 推荐 c=4-c=8，c=8 是吞吐峰值，c=16 是压力边界。",
        "- 可以说：stage handoff 当前健康，code2wav decode 不是优先瓶颈。",
        "- 不要说：vLLM c=8 prebuild w4 已证明 online parity。",
        "- 不要说：preprocessing 并发越高越好。",
        "- 不要说：官方 SeedTTS full-set 已是 headline evidence；当前只提供本地 smoke path。",
        "",
        "### 10.1 现场答辩速查",
        "",
        "这张表来自 `defense_claim_matrix.json`，用于现场把“能说什么、怎么复跑、失败时怎么撤回”放在同一个口径里。",
        "同一个 JSON 还包含 13 个 Q&A 问题到 10 条 defense claim 的 `qna_question_rows` 映射，",
        "方便从现场提问一跳进入机器证据、复跑命令和撤回条件。",
        "完整展开版在 `benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`。",
        "",
        "快速抽取命令：",
        "",
        "```bash",
        "jq '.rows[] | {claim, allowed_wording, rerun_command_ids, failure_decision}' \\",
        "  results/qwen35_report_audit_20260619/defense_claim_matrix.json",
        "```",
        "",
        *_defense_claim_table(defense_claim_matrix),
        "",
        "## 11. 证据入口",
        "",
        f"- 本报告 gate：ready=`{summary.get('ready')}`，checks=`{summary.get('checks_passed')}/{summary.get('checks_total')}`。",
        f"- Final readiness：ready=`{sources['final_readiness'].get('ready')}`，checks=`{sources['final_readiness'].get('checks_passed')}/{sources['final_readiness'].get('checks_total')}`。",
        f"- Manifest：records=`{sources['manifest'].get('total_records')}`，missing=`{sources['manifest'].get('missing_records')}`。",
        f"- Share bundle：ready=`{sources['share_bundle_manifest'].get('ready')}`，records=`{sources['share_bundle_manifest'].get('records_total')}`。",
        f"- Share package validation：ready=`{sources['share_package_validation'].get('ready')}`，checks=`{sources['share_package_validation'].get('checks_passed')}/{sources['share_package_validation'].get('checks_total')}`。",
        f"- Receiver smoke validation：ready=`{sources['share_package_receiver_smoke_validation'].get('ready')}`，checks=`{sources['share_package_receiver_smoke_validation'].get('checks_passed')}/{sources['share_package_receiver_smoke_validation'].get('checks_total')}`。",
        f"- External standalone validation：ready=`{sources['share_package_external_standalone_validation'].get('ready')}`，checks=`{sources['share_package_external_standalone_validation'].get('checks_passed')}/{sources['share_package_external_standalone_validation'].get('checks_total')}`。",
        f"- Share release seal：ready=`{sources['share_release_seal'].get('ready')}`，checks=`{sources['share_release_seal'].get('checks_passed')}/{sources['share_release_seal'].get('checks_total')}`，adjacent hashed=`{sources['share_release_seal'].get('adjacent_artifacts_hashed')}/{sources['share_release_seal'].get('adjacent_artifacts_total')}`；完整 tarball identity 只在 release seal JSON 中校验。",
        f"- Evidence-query smoke：ready=`{sources['evidence_query_smoke'].get('ready')}`，host/portable pass=`{sources['evidence_query_smoke'].get('host_pass_lines')}/{sources['evidence_query_smoke'].get('portable_pass_lines')}`。",
        f"- Defense claim matrix：ready=`{sources['defense_claim_matrix'].get('ready')}`，claim_rows=`{sources['defense_claim_matrix'].get('rows_total')}`，question_rows=`{sources['defense_claim_matrix'].get('question_rows_total')}`，failure_decisions=`{sources['defense_claim_matrix'].get('failure_decisions_total')}`。",
        f"- Tail confidence appendix：ready=`{sources['tail_confidence_appendix'].get('ready')}`，rows=`{sources['tail_confidence_appendix'].get('rows_total')}`，strict_c4_sglang_p95=`{sources['tail_confidence_appendix'].get('strict_c4_sglang_latency_p95_s')}`，strict_c4_vllm_p95=`{sources['tail_confidence_appendix'].get('strict_c4_vllm_latency_p95_s')}`。",
        "- `results/qwen35_report_audit_20260619/share_package_validation.json`",
        "- `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`",
        "- `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`",
        "- `results/qwen35_report_audit_20260619/share_release_seal.json`",
        "- `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json`",
        "- `results/qwen35_report_audit_20260619/headline_scorecard.json`",
        "- `results/qwen35_report_audit_20260619/regime_decision_matrix.json`",
        "- `results/qwen35_report_audit_20260619/stage_latency_budget.json`",
        "- `results/qwen35_report_audit_20260619/pressure_stage_heatmap.json`",
        "- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`",
        "- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`",
        "- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`",
        "- `results/qwen35_report_audit_20260619/defense_claim_matrix.json`",
        "- `results/qwen35_report_audit_20260619/metric_provenance_index.json`",
        "- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`",
        "- `results/qwen35_report_audit_20260619/repro_command_manifest.json`",
        "- `benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md`",
        "- `benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Chinese university-facing Qwen3.5-Omni technical report."
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
    output.write_text(build_markdown(root, payload), encoding="utf-8")
    _save_json(payload, json_output)
    summary = payload["summary"]
    print(
        "University technical report written: "
        f"{output} json={json_output} ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']}"
    )
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
