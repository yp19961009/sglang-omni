# SPDX-License-Identifier: Apache-2.0
"""Build a confidence ledger for the Qwen3.5-Omni handoff report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _claim_passed(claims: dict[str, Any], name: str) -> bool:
    for row in claims.get("checks", []):
        if row.get("name") == name:
            return bool(row.get("passed"))
    return False


def _acceptance_row(
    acceptance: dict[str, Any], *, regime: str, pressure: str
) -> dict[str, Any]:
    for row in acceptance.get("rows", []):
        if row.get("regime") == regime and row.get("pressure") == pressure:
            return row
    return {}


def _score_check(scorecard: dict[str, Any], name: str) -> bool:
    return bool(scorecard.get("summary", {}).get("checks", {}).get(name))


def _path(root: Path, rel: str) -> str:
    return str((root / rel).resolve())


def _entry(
    *,
    claim_id: str,
    category: str,
    confidence: str,
    claim_zh: str,
    allowed_wording_zh: str,
    boundary_zh: str,
    evidence: list[str],
    checks: dict[str, bool],
) -> dict[str, Any]:
    passed = all(checks.values())
    return {
        "id": claim_id,
        "category": category,
        "confidence": confidence,
        "status": "PASS" if passed else "FAIL",
        "claim_zh": claim_zh,
        "allowed_wording_zh": allowed_wording_zh,
        "boundary_zh": boundary_zh,
        "evidence": evidence,
        "checks": checks,
    }


def build_ledger(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    claims = _load_json(audit_dir / "claims_verification.json")
    scorecard = _load_json(audit_dir / "headline_scorecard.json")
    acceptance = _load_json(audit_dir / "acceptance_matrix.json")
    stage_interactions = _load_json(audit_dir / "stage_interaction_summary.json")
    coverage = _load_json(audit_dir / "coverage_matrix.json")
    seedtts = _load_json(audit_dir / "videoamme_seedtts_meta_summary.json")

    entries = [
        _entry(
            claim_id="strict_c4_sglang_beats_optimized_vllm",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="warmed c=4 严格横向对比中，SGLang-Omni latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。",
            allowed_wording_zh="可以作为主 headline：在 Video-AMME ci-50 warmed c=4 上，SGLang-Omni 至少与优化版 vLLM 相当，并在 latency/RTF 上更优。",
            boundary_zh="仅覆盖当前 8x H20、Video-AMME ci-50、warmed steady-state、语音输出 workload。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/headline_scorecard.json"),
                _path(root, "results/qwen35_report_audit_20260619/claims_verification.json"),
            ],
            checks={
                "scorecard_latency_rtf_win": _score_check(
                    scorecard, "strict_c4_sglang_latency_rtf_win"
                ),
                "scorecard_quality_preserved": _score_check(
                    scorecard, "strict_c4_accuracy_wer_preserved"
                ),
                "claim_latency_rtf_win": _claim_passed(
                    claims, "SGLang warmed c4 beats vLLM warmed c4 latency/RTF"
                ),
                "claim_quality_preserved": _claim_passed(
                    claims, "SGLang warmed c4 preserves accuracy/WER vs vLLM"
                ),
            },
        ),
        _entry(
            claim_id="sglang_c8_peak_c16_saturation",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="当前 SGLang recipe 的 Video-AMME 压测吞吐峰值是 c=8；c=16 是饱和边界，不是推荐运行点。",
            allowed_wording_zh="可以说 c=8 是当前高并发 sweet spot，c=16 用来证明 admission/queueing 饱和。",
            boundary_zh="不外推到不同 admission 策略、不同 GPU 拓扑或更大数据分布。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/acceptance_matrix.json"),
                _path(root, "results/qwen35_report_audit_20260619/claims_verification.json"),
            ],
            checks={
                "scorecard_c8_peak": _score_check(scorecard, "sglang_stress_c8_is_peak"),
                "scorecard_c16_regresses": _score_check(
                    scorecard, "sglang_c16_regresses_vs_c8"
                ),
                "claim_c8_peak": _claim_passed(claims, "SGLang stress c8 is throughput peak"),
                "acceptance_c8": bool(
                    _acceptance_row(
                        acceptance,
                        regime="sglang_videoamme_stress",
                        pressure="c=8",
                    ).get("accepted")
                ),
                "acceptance_c16_boundary": bool(
                    _acceptance_row(
                        acceptance,
                        regime="sglang_videoamme_stress",
                        pressure="c=16",
                    ).get("accepted")
                ),
            },
        ),
        _entry(
            claim_id="short_long_speech_faster_than_realtime",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="短/长文本输入 + 语音输出在 c=1/4/8 下都有覆盖；long c=8 仍快于实时。",
            allowed_wording_zh="可以说短文本 12 words、长文本 139 words 的 text-to-speech guardrail 已覆盖，long c=8 平均 RTF 0.4932。",
            boundary_zh="这是 synthetic text-to-speech guardrail，不替代官方 SeedTTS full-set headline。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/acceptance_matrix.json"),
                _path(root, "results/qwen35_report_audit_20260619/tables_summary.json"),
            ],
            checks={
                "scorecard_long_rt": _score_check(scorecard, "long_c8_faster_than_real_time"),
                "claim_long_rt": _claim_passed(
                    claims, "long synthetic speech remains faster than real time"
                ),
                "acceptance_short_c8": bool(
                    _acceptance_row(
                        acceptance,
                        regime="sglang_synthetic_speech",
                        pressure="short c=8",
                    ).get("accepted")
                ),
                "acceptance_long_c8": bool(
                    _acceptance_row(
                        acceptance,
                        regime="sglang_synthetic_speech",
                        pressure="long c=8",
                    ).get("accepted")
                ),
            },
        ),
        _entry(
            claim_id="stage_boundaries_not_primary_bottleneck",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="当前主要瓶颈不是 stage 间 handoff，也不是 code2wav decode compute。",
            allowed_wording_zh="可以说 talker->code2wav hop 和 code2wav decode 都有明确非瓶颈证据。",
            boundary_zh="vLLM prebuild 后暴露的 engine/talker-side tail 仍需后续在线路径复核。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/stage_interaction_summary.json"),
                _path(root, "results/qwen35_report_audit_20260619/claims_verification.json"),
            ],
            checks={
                "scorecard_stage_connections": _score_check(
                    scorecard, "stage_connections_healthy"
                ),
                "summary_talker_to_code2wav": bool(
                    stage_interactions.get("summary", {}).get(
                        "sglang_talker_to_code2wav_healthy"
                    )
                ),
                "summary_code2wav_not_bottleneck": bool(
                    stage_interactions.get("summary", {}).get(
                        "sglang_code2wav_decode_not_bottleneck"
                    )
                ),
                "claim_code2wav_not_bottleneck": _claim_passed(
                    claims, "code2wav decode and talker->code2wav hop are not bottlenecks"
                ),
            },
        ),
        _entry(
            claim_id="high_concurrency_admission_and_talker_tail",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="高并发变慢主要来自 preprocessing admission/queueing 和 talker AR tail 的组合。",
            allowed_wording_zh="可以说 c=8 开始 queueing 变明显，c=16 queue/admission 主导短输出 tail。",
            boundary_zh="若改变 preprocessing placement 或 thinker admission，需要重新跑 tail appendix。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/stage_interaction_summary.json"),
                _path(root, "results/qwen35_report_audit_20260619/claims_verification.json"),
            ],
            checks={
                "claim_queue_growth": _claim_passed(
                    claims, "preprocessing lifecycle growth is queue/admission dominated"
                ),
                "summary_has_queue_limited": int(
                    stage_interactions.get("summary", {})
                    .get("status_counts", {})
                    .get("queue_limited", 0)
                )
                >= 2,
                "acceptance_c16_boundary": bool(
                    _acceptance_row(
                        acceptance,
                        regime="sglang_videoamme_stress",
                        pressure="c=16",
                    ).get("accepted")
                ),
            },
        ),
        _entry(
            claim_id="preprocessing_parallelism_is_anti_recipe",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="朴素扩大 preprocessing 并发是负优化：preproc=2 变慢，preproc=4 出现失败/OOM 风险。",
            allowed_wording_zh="可以明确把 preproc=2/4 列为当前 recipe 的 anti-recipe。",
            boundary_zh="只有在同时改变 admission、memory fraction 或 placement 后才值得重新评估。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/acceptance_matrix.json"),
                _path(root, "results/qwen35_report_audit_20260619/claims_verification.json"),
            ],
            checks={
                "scorecard_preproc_regresses": _score_check(
                    scorecard, "preprocessing_parallelism_regresses"
                ),
                "claim_preproc_regresses": _claim_passed(
                    claims, "naive preprocessing parallelism regresses or fails"
                ),
                "acceptance_preproc2": bool(
                    _acceptance_row(
                        acceptance,
                        regime="negative_optimization",
                        pressure="PREPROCESSING_MAX_CONCURRENCY=2 at c=8",
                    ).get("accepted")
                ),
                "acceptance_preproc4": bool(
                    _acceptance_row(
                        acceptance,
                        regime="negative_optimization",
                        pressure="PREPROCESSING_MAX_CONCURRENCY=4 at c=8",
                    ).get("accepted")
                ),
            },
        ),
        _entry(
            claim_id="vllm_original_c8_prompt_feed_limited",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="vLLM 原始 offline c=8 主要受 host prompt build/feed admission 限制。",
            allowed_wording_zh="可以说原始 vLLM c=8 不能直接作为 serving parity 反例，因为 runner/admission 限制占主导。",
            boundary_zh="这是 offline runner 诊断，不是在线服务路径结论。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"),
                _path(root, "results/qwen35_report_audit_20260619/vllm_log_stage_summary.json"),
            ],
            checks={
                "scorecard_vllm_prompt_feed": _score_check(
                    scorecard, "vllm_original_c8_prompt_feed_limited"
                ),
                "claim_vllm_c8_prompt_feed": _claim_passed(
                    claims, "vLLM c8 offline runner is prompt-feed/admission limited"
                ),
                "claim_vllm_diagnosis": _claim_passed(
                    claims,
                    "vLLM offline admission diagnosis classifies c4/c8 as prompt-feed limited",
                ),
                "acceptance_original_c8": bool(
                    _acceptance_row(
                        acceptance,
                        regime="vllm_offline_diagnostic",
                        pressure="original c=8",
                    ).get("accepted")
                ),
            },
        ),
        _entry(
            claim_id="vllm_prebuild_w4_is_optimized_offline_diagnostic",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="vLLM c=8 prebuild w4 改善 runner wall，并把瓶颈暴露到 engine/workload/talker-side tail。",
            allowed_wording_zh="可以把 prebuild w4 作为优化过的 offline diagnostic baseline。",
            boundary_zh="不能把它说成严格 online serving-throughput parity。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/acceptance_matrix.json"),
                _path(root, "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"),
            ],
            checks={
                "scorecard_w4_improves": _score_check(
                    scorecard, "vllm_w4_prebuild_improves_runner_wall"
                ),
                "claim_w4_improves": _claim_passed(
                    claims,
                    "vLLM c8 prebuilt prompts with 4 workers improve runner-side wall clock",
                ),
                "acceptance_w4": bool(
                    _acceptance_row(
                        acceptance,
                        regime="vllm_offline_diagnostic",
                        pressure="prebuild c=8 workers=4",
                    ).get("accepted")
                ),
            },
        ),
        _entry(
            claim_id="wer_audio_quality_stable",
            category="safe_high_confidence_claim",
            confidence="high",
            claim_zh="SGLang stress sweep 的 WER/audio consistency 稳定，且 warmed c=4 WER 优于 vLLM c=4。",
            allowed_wording_zh="可以说当前性能优化没有牺牲语音一致性指标。",
            boundary_zh="WER 来自本地 offline Whisper large-v3，对外复现需要可用 ASR 权重或路由。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/claims_verification.json"),
                _path(root, "results/qwen35_report_audit_20260619/preflight_repro.json"),
            ],
            checks={
                "claim_wer_stable": _claim_passed(claims, "SGLang stress WER remains stable"),
                "claim_c4_quality": _claim_passed(
                    claims, "SGLang warmed c4 preserves accuracy/WER vs vLLM"
                ),
            },
        ),
        _entry(
            claim_id="official_seedtts_fullset_not_headline",
            category="bounded_medium_confidence_statement",
            confidence="medium",
            claim_zh="官方 SeedTTS full-set 当前不能作为 headline；本地只提供 Video-AMME spoken-reference smoke path。",
            allowed_wording_zh="可以说已提供 SeedTTS-compatible smoke meta，官方 full-set 需要预置数据后再跑。",
            boundary_zh="不要把 smoke path 描述成官方 SeedTTS 完整集性能。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json"),
                _path(root, "results/qwen35_report_audit_20260619/coverage_matrix.json"),
            ],
            checks={
                "seedtts_smoke_rows": int(seedtts.get("rows") or 0) >= 50,
                "coverage_complete": bool(coverage.get("summary", {}).get("complete")),
            },
        ),
        _entry(
            claim_id="strict_vllm_c8_online_parity_pending",
            category="bounded_medium_confidence_statement",
            confidence="medium",
            claim_zh="严格 vLLM c=8 online serving-throughput parity 仍需在线 ingress + WER/ASR 复核。",
            allowed_wording_zh="可以说当前有 offline diagnostic，不做 online parity 过度声明。",
            boundary_zh="不要宣称已经完成严格 c=8 在线 serving parity。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/acceptance_matrix.json"),
                _path(root, "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"),
            ],
            checks={
                "acceptance_w4_diagnostic": bool(
                    _acceptance_row(
                        acceptance,
                        regime="vllm_offline_diagnostic",
                        pressure="prebuild c=8 workers=4",
                    ).get("accepted")
                ),
                "w4_status_is_diagnostic": _acceptance_row(
                    acceptance,
                    regime="vllm_offline_diagnostic",
                    pressure="prebuild c=8 workers=4",
                ).get("serving_status")
                == "optimized_offline_diagnostic",
            },
        ),
        _entry(
            claim_id="larger_traffic_extrapolation_pending",
            category="bounded_medium_confidence_statement",
            confidence="medium",
            claim_zh="对更大 Video-AMME/full traffic 的外推仍是中置信，需要更大数据和真实流量复核。",
            allowed_wording_zh="可以说 ci-50/stress/synthetic 证据充分支持当前报告范围。",
            boundary_zh="不要把 ci-50 直接表述成全量线上流量结论。",
            evidence=[
                _path(root, "results/qwen35_report_audit_20260619/coverage_matrix.json"),
                _path(root, "results/qwen35_report_audit_20260619/acceptance_matrix.json"),
            ],
            checks={
                "coverage_complete": bool(coverage.get("summary", {}).get("complete")),
                "acceptance_ready": bool(acceptance.get("summary", {}).get("ready")),
            },
        ),
    ]

    confidence_counts = Counter(row["confidence"] for row in entries)
    category_counts = Counter(row["category"] for row in entries)
    failed = [row for row in entries if row["status"] != "PASS"]
    high_entries = [row for row in entries if row["confidence"] == "high"]
    medium_entries = [row for row in entries if row["confidence"] == "medium"]
    unsupported = [row for row in entries if row["confidence"] == "unsupported"]

    return {
        "root": str(root),
        "summary": {
            "ready": not failed and len(high_entries) >= 9 and len(medium_entries) >= 3,
            "entries_total": len(entries),
            "entries_passed": len(entries) - len(failed),
            "entries_failed": len(failed),
            "high_confidence_claims": len(high_entries),
            "medium_confidence_boundaries": len(medium_entries),
            "unsupported_claims": len(unsupported),
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "entries": entries,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Confidence Ledger\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Entries | {summary['entries_passed']}/{summary['entries_total']} |")
    print(f"| High-confidence claims | {summary['high_confidence_claims']} |")
    print(f"| Medium-confidence boundaries | {summary['medium_confidence_boundaries']} |")
    print(f"| Unsupported claims | {summary['unsupported_claims']} |")
    print()
    print("| Confidence | Category | Claim | Boundary | Status |")
    print("| --- | --- | --- | --- | --- |")
    for row in payload["entries"]:
        print(
            f"| {row['confidence']} | {row['category']} | {row['claim_zh']} | "
            f"{row['boundary_zh']} | {row['status']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the confidence ledger for Qwen3.5-Omni report claims."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_ledger(root)
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = root / output
        _save_json(payload, output)
    print_markdown(payload)
    if not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
