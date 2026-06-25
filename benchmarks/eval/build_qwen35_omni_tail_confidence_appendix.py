# SPDX-License-Identifier: Apache-2.0
"""Build a tail-confidence appendix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "tail_confidence_appendix.json"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260621


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    group: str
    label: str
    result_path: str
    skip_first: int = 0
    wer_path: str | None = None
    required: bool = True


STRICT_CASES = [
    CaseSpec(
        "strict_sglang_c4_warm",
        "strict_c4",
        "SGLang warmed c=4",
        "results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/"
        "benchmark_audio_50_c4_warm_profile_no_wer/videoamme_results.json",
        skip_first=4,
        wer_path=(
            "results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/"
            "benchmark_audio_50_c4_warm_profile_no_wer/whisper_large_v3_wer.json"
        ),
    ),
    CaseSpec(
        "strict_vllm_c4_warm",
        "strict_c4",
        "vLLM warmed c=4",
        "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/"
        "benchmark_audio_50_c4_offline_compile/videoamme_results.json",
        skip_first=4,
        wer_path=(
            "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/"
            "benchmark_audio_50_c4_offline_compile/whisper_large_v3_wer.json"
        ),
    ),
]

SGLANG_STRESS_CASES = [
    CaseSpec(
        "sglang_stress_c1",
        "sglang_stress",
        "SGLang Video-AMME c=1",
        "results/qwen35_sglang_mr8_stress_20260619/"
        "benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json",
        wer_path=(
            "results/qwen35_sglang_mr8_stress_20260619/"
            "benchmark_audio_50_c1_warm_profile_skipwer/whisper_large_v3_local_wer.json"
        ),
    ),
    CaseSpec(
        "sglang_stress_c2",
        "sglang_stress",
        "SGLang Video-AMME c=2",
        "results/qwen35_sglang_mr8_stress_20260619/"
        "benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json",
        wer_path=(
            "results/qwen35_sglang_mr8_stress_20260619/"
            "benchmark_audio_50_c2_warm_profile_skipwer/whisper_large_v3_local_wer.json"
        ),
    ),
    CaseSpec(
        "sglang_stress_c4",
        "sglang_stress",
        "SGLang Video-AMME c=4",
        "results/qwen35_sglang_mr8_stress_20260619/"
        "benchmark_audio_50_c4_profile_skipwer/videoamme_results.json",
        wer_path=(
            "results/qwen35_sglang_mr8_stress_20260619/"
            "benchmark_audio_50_c4_profile_skipwer/whisper_large_v3_local_wer.json"
        ),
    ),
    CaseSpec(
        "sglang_stress_c8",
        "sglang_stress",
        "SGLang Video-AMME c=8",
        "results/qwen35_sglang_mr8_stress_20260619/"
        "benchmark_audio_50_c8_profile_skipwer/videoamme_results.json",
        wer_path=(
            "results/qwen35_sglang_mr8_stress_20260619/"
            "benchmark_audio_50_c8_profile_skipwer/whisper_large_v3_local_wer.json"
        ),
    ),
    CaseSpec(
        "sglang_stress_c16",
        "sglang_stress",
        "SGLang Video-AMME c=16",
        "results/qwen35_sglang_mr8_stress_20260619/"
        "benchmark_audio_50_c16_profile_skipwer/videoamme_results.json",
        wer_path=(
            "results/qwen35_sglang_mr8_stress_20260619/"
            "benchmark_audio_50_c16_profile_skipwer/whisper_large_v3_local_wer.json"
        ),
    ),
]

SYNTHETIC_CASES = [
    CaseSpec(
        f"synthetic_{scenario}_c{concurrency}",
        "synthetic_speech",
        f"Synthetic {scenario} c={concurrency}",
        f"results/qwen35_synthetic_speech_20260619/{scenario}_c{concurrency}/"
        "synthetic_speech_results.json",
    )
    for scenario in ("short", "long")
    for concurrency in (1, 4, 8)
]

VLLM_CASES = [
    CaseSpec(
        "vllm_c1",
        "vllm_diagnostic",
        "vLLM original c=1",
        "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/"
        "benchmark_audio_50_c1_offline_compile/videoamme_results.json",
        skip_first=4,
    ),
    CaseSpec(
        "vllm_c4",
        "vllm_diagnostic",
        "vLLM warmed c=4",
        "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/"
        "benchmark_audio_50_c4_offline_compile/videoamme_results.json",
        skip_first=4,
        wer_path=(
            "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/"
            "benchmark_audio_50_c4_offline_compile/whisper_large_v3_wer.json"
        ),
    ),
    CaseSpec(
        "vllm_c8",
        "vllm_diagnostic",
        "vLLM original c=8",
        "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/"
        "benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        skip_first=8,
    ),
    CaseSpec(
        "vllm_c8_prebuild_w1",
        "vllm_diagnostic",
        "vLLM prebuild c=8 w1",
        "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/"
        "benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        skip_first=8,
    ),
    CaseSpec(
        "vllm_c8_prebuild_w4",
        "vllm_diagnostic",
        "vLLM prebuild c=8 w4",
        "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/"
        "benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        skip_first=8,
    ),
]

ALL_CASES = [*STRICT_CASES, *SGLANG_STRESS_CASES, *SYNTHETIC_CASES, *VLLM_CASES]


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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _dist(values: list[float]) -> dict[str, float | None]:
    q25 = _quantile(values, 0.25)
    q75 = _quantile(values, 0.75)
    return {
        "mean": _mean(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "iqr": (q75 - q25) if q25 is not None and q75 is not None else None,
    }


def _case_metric_values(root: Path, spec: CaseSpec, metric: str) -> list[float]:
    payload = _load_json(root / spec.result_path)
    per_sample = payload.get("per_sample", [])
    rows = per_sample if isinstance(per_sample, list) else []
    effective_rows = [
        row for row in rows[spec.skip_first :] if isinstance(row, dict)
    ]
    return [
        value
        for value in (_float(row.get(metric)) for row in effective_rows)
        if value is not None
    ]


def _stat_mean(values: list[float]) -> float:
    return float(_mean(values) or 0.0)


def _stat_p95(values: list[float]) -> float:
    return float(_quantile(values, 0.95) or 0.0)


def _bootstrap_stat(
    values: list[float], stat_fn: Any, *, seed: int
) -> dict[str, Any]:
    if not values:
        return {
            "point_estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "draws": BOOTSTRAP_DRAWS,
            "seed": seed,
        }
    rng = random.Random(seed)
    draws: list[float] = []
    n = len(values)
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [values[rng.randrange(n)] for __ in range(n)]
        draws.append(float(stat_fn(sample)))
    return {
        "point_estimate": float(stat_fn(values)),
        "ci95_low": _quantile(draws, 0.025),
        "ci95_high": _quantile(draws, 0.975),
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
    }


def _bootstrap_delta(
    lhs_values: list[float], rhs_values: list[float], stat_fn: Any, *, seed: int
) -> dict[str, Any]:
    if not lhs_values or not rhs_values:
        return {
            "point_estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "draws": BOOTSTRAP_DRAWS,
            "seed": seed,
        }
    rng = random.Random(seed)
    lhs_n = len(lhs_values)
    rhs_n = len(rhs_values)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        lhs_sample = [lhs_values[rng.randrange(lhs_n)] for __ in range(lhs_n)]
        rhs_sample = [rhs_values[rng.randrange(rhs_n)] for __ in range(rhs_n)]
        draws.append(float(stat_fn(rhs_sample)) - float(stat_fn(lhs_sample)))
    return {
        "point_estimate": float(stat_fn(rhs_values)) - float(stat_fn(lhs_values)),
        "ci95_low": _quantile(draws, 0.025),
        "ci95_high": _quantile(draws, 0.975),
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
    }


def _comparison_row(
    *,
    comparison_id: str,
    label: str,
    source_cases: list[str],
    metric: str,
    statistic: str,
    unit: str,
    direction: str,
    estimate: dict[str, Any],
    decision: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "label": label,
        "source_cases": source_cases,
        "metric": metric,
        "statistic": statistic,
        "unit": unit,
        "direction": direction,
        "point_estimate": estimate.get("point_estimate"),
        "ci95_low": estimate.get("ci95_low"),
        "ci95_high": estimate.get("ci95_high"),
        "draws": estimate.get("draws"),
        "seed": estimate.get("seed"),
        "decision": decision,
        "interpretation": interpretation,
    }


def _by_comparison(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["comparison_id"]): row for row in rows}


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    numeric = _float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}{suffix}"


def _pct(value: Any, digits: int = 1) -> str:
    numeric = _float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric * 100.0:.{digits}f}%"


def _success_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if bool(row.get("is_success", True)))


def _accuracy(rows: list[dict[str, Any]], summary: dict[str, Any]) -> float | None:
    if "accuracy" in summary:
        return _float(summary.get("accuracy"))
    if not any("is_correct" in row for row in rows):
        return None
    correct = sum(1 for row in rows if row.get("is_correct") is True)
    return correct / len(rows) if rows else None


def _wer_summary(root: Path, spec: CaseSpec) -> dict[str, Any]:
    if not spec.wer_path:
        return {}
    payload = _load_json_optional(root / spec.wer_path)
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _case_row(root: Path, spec: CaseSpec) -> dict[str, Any]:
    path = root / spec.result_path
    payload = _load_json(path)
    summary = payload.get("summary", {})
    speed = payload.get("speed", {})
    per_sample = payload.get("per_sample", [])
    rows = per_sample if isinstance(per_sample, list) else []
    effective_rows = [
        row for row in rows[spec.skip_first :] if isinstance(row, dict)
    ]
    latency_values = [
        value
        for value in (_float(row.get("latency_s")) for row in effective_rows)
        if value is not None
    ]
    rtf_values = [
        value
        for value in (_float(row.get("rtf")) for row in effective_rows)
        if value is not None
    ]
    audio_values = [
        value
        for value in (_float(row.get("audio_duration_s")) for row in effective_rows)
        if value is not None
    ]
    wer = _wer_summary(root, spec)
    return {
        "case_id": spec.case_id,
        "group": spec.group,
        "label": spec.label,
        "result_path": spec.result_path,
        "wer_path": spec.wer_path,
        "skip_first": spec.skip_first,
        "n_raw": len(rows),
        "n_effective": len(effective_rows),
        "success": _success_count(effective_rows),
        "failures": len(effective_rows) - _success_count(effective_rows),
        "accuracy": _accuracy(effective_rows, summary),
        "wer_corpus": _float(wer.get("wer_corpus")),
        "wer_p95": _float(wer.get("wer_per_sample_p95")),
        "throughput_qps": _float(speed.get("throughput_qps")),
        "audio_throughput_s_per_s": _float(speed.get("audio_throughput_s_per_s")),
        "latency_s": _dist(latency_values),
        "rtf": _dist(rtf_values),
        "audio_duration_s": _dist(audio_values),
        "source_summary": {
            "speed_latency_mean_s": speed.get("latency_mean_s"),
            "speed_latency_p95_s": speed.get("latency_p95_s"),
            "speed_rtf_mean": speed.get("rtf_mean"),
            "speed_rtf_p95": speed.get("rtf_p95"),
            "summary_total_samples": summary.get("total_samples"),
            "summary_failed": summary.get("failed"),
        },
    }


def _by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in rows}


def _check(name: str, condition: bool, evidence: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "required": True,
        "evidence": evidence,
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    missing = [
        spec.result_path
        for spec in ALL_CASES
        if spec.required and not (root / spec.result_path).is_file()
    ]
    rows = [_case_row(root, spec) for spec in ALL_CASES if (root / spec.result_path).is_file()]
    by_id = _by_case(rows)
    spec_by_id = {spec.case_id: spec for spec in ALL_CASES}
    values_by_id = {
        case_id: {
            "latency_s": _case_metric_values(root, spec, "latency_s"),
            "rtf": _case_metric_values(root, spec, "rtf"),
        }
        for case_id, spec in spec_by_id.items()
        if (root / spec.result_path).is_file()
    }

    strict_sglang = by_id["strict_sglang_c4_warm"]
    strict_vllm = by_id["strict_vllm_c4_warm"]
    sglang_c8 = by_id["sglang_stress_c8"]
    sglang_c16 = by_id["sglang_stress_c16"]
    long_c8 = by_id["synthetic_long_c8"]
    vllm_c8 = by_id["vllm_c8"]
    vllm_w4 = by_id["vllm_c8_prebuild_w4"]
    comparison_rows = [
        _comparison_row(
            comparison_id="strict_c4_latency_mean_advantage",
            label="Strict c=4 mean latency advantage",
            source_cases=["strict_sglang_c4_warm", "strict_vllm_c4_warm"],
            metric="latency_s",
            statistic="mean_delta",
            unit="s",
            direction="vLLM - SGLang; positive means SGLang is lower",
            estimate=_bootstrap_delta(
                values_by_id["strict_sglang_c4_warm"]["latency_s"],
                values_by_id["strict_vllm_c4_warm"]["latency_s"],
                _stat_mean,
                seed=BOOTSTRAP_SEED + 1,
            ),
            decision="ci95_low_gt_0",
            interpretation=(
                "Bootstrap 95% CI excludes zero, so the strict c=4 latency "
                "mean advantage is a strong headline support."
            ),
        ),
        _comparison_row(
            comparison_id="strict_c4_latency_p95_advantage",
            label="Strict c=4 p95 latency advantage",
            source_cases=["strict_sglang_c4_warm", "strict_vllm_c4_warm"],
            metric="latency_s",
            statistic="p95_delta",
            unit="s",
            direction="vLLM - SGLang; positive means SGLang is lower",
            estimate=_bootstrap_delta(
                values_by_id["strict_sglang_c4_warm"]["latency_s"],
                values_by_id["strict_vllm_c4_warm"]["latency_s"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 2,
            ),
            decision="point_estimate_gt_0_with_ci_overlap",
            interpretation=(
                "Point estimate favors SGLang; the p95 CI overlaps zero, so "
                "this is supporting tail evidence, not a standalone significance claim."
            ),
        ),
        _comparison_row(
            comparison_id="strict_c4_rtf_mean_advantage",
            label="Strict c=4 mean RTF advantage",
            source_cases=["strict_sglang_c4_warm", "strict_vllm_c4_warm"],
            metric="rtf",
            statistic="mean_delta",
            unit="",
            direction="vLLM - SGLang; positive means SGLang is lower",
            estimate=_bootstrap_delta(
                values_by_id["strict_sglang_c4_warm"]["rtf"],
                values_by_id["strict_vllm_c4_warm"]["rtf"],
                _stat_mean,
                seed=BOOTSTRAP_SEED + 3,
            ),
            decision="point_estimate_gt_0_with_ci_overlap",
            interpretation=(
                "Point estimate favors SGLang, but the bootstrap CI overlaps "
                "zero; report it as directional support with a caveat."
            ),
        ),
        _comparison_row(
            comparison_id="strict_c4_rtf_p95_advantage",
            label="Strict c=4 p95 RTF advantage",
            source_cases=["strict_sglang_c4_warm", "strict_vllm_c4_warm"],
            metric="rtf",
            statistic="p95_delta",
            unit="",
            direction="vLLM - SGLang; positive means SGLang is lower",
            estimate=_bootstrap_delta(
                values_by_id["strict_sglang_c4_warm"]["rtf"],
                values_by_id["strict_vllm_c4_warm"]["rtf"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 4,
            ),
            decision="point_estimate_gt_0_with_ci_overlap",
            interpretation=(
                "Point estimate favors SGLang; CI overlap keeps this as tail "
                "support rather than an inferential RTF-only claim."
            ),
        ),
        _comparison_row(
            comparison_id="sglang_c16_vs_c8_latency_p95_penalty",
            label="SGLang c16 vs c8 p95 latency penalty",
            source_cases=["sglang_stress_c8", "sglang_stress_c16"],
            metric="latency_s",
            statistic="p95_delta",
            unit="s",
            direction="c16 - c8; positive means c16 tail is worse",
            estimate=_bootstrap_delta(
                values_by_id["sglang_stress_c8"]["latency_s"],
                values_by_id["sglang_stress_c16"]["latency_s"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 5,
            ),
            decision="ci95_low_gt_0",
            interpretation=(
                "Bootstrap CI excludes zero; this supports c16 as a saturation "
                "boundary, not the recommended serving point."
            ),
        ),
        _comparison_row(
            comparison_id="sglang_c16_vs_c8_rtf_p95_penalty",
            label="SGLang c16 vs c8 p95 RTF penalty",
            source_cases=["sglang_stress_c8", "sglang_stress_c16"],
            metric="rtf",
            statistic="p95_delta",
            unit="",
            direction="c16 - c8; positive means c16 tail is worse",
            estimate=_bootstrap_delta(
                values_by_id["sglang_stress_c8"]["rtf"],
                values_by_id["sglang_stress_c16"]["rtf"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 6,
            ),
            decision="ci95_low_gt_0",
            interpretation=(
                "Bootstrap CI excludes zero; c16 increases tail RTF even though "
                "quality remains stable."
            ),
        ),
        _comparison_row(
            comparison_id="synthetic_long_c8_rtf_p95_realtime_margin",
            label="Synthetic long c8 p95 RTF real-time margin",
            source_cases=["synthetic_long_c8"],
            metric="rtf",
            statistic="p95",
            unit="",
            direction="upper CI below 1 means faster than real time",
            estimate=_bootstrap_stat(
                values_by_id["synthetic_long_c8"]["rtf"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 7,
            ),
            decision="ci95_high_lt_1",
            interpretation=(
                "Bootstrap upper bound remains below 1.0, so the long-text "
                "synthetic guardrail stays faster than real time."
            ),
        ),
        _comparison_row(
            comparison_id="vllm_w4_vs_original_latency_p95_penalty",
            label="vLLM prebuild w4 vs original c8 p95 latency penalty",
            source_cases=["vllm_c8", "vllm_c8_prebuild_w4"],
            metric="latency_s",
            statistic="p95_delta",
            unit="s",
            direction="prebuild w4 - original; positive means larger tail",
            estimate=_bootstrap_delta(
                values_by_id["vllm_c8"]["latency_s"],
                values_by_id["vllm_c8_prebuild_w4"]["latency_s"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 8,
            ),
            decision="ci95_low_gt_0",
            interpretation=(
                "Prebuild w4 improves throughput diagnostics but exposes a "
                "larger per-request latency tail."
            ),
        ),
        _comparison_row(
            comparison_id="vllm_w4_vs_original_rtf_p95_penalty",
            label="vLLM prebuild w4 vs original c8 p95 RTF penalty",
            source_cases=["vllm_c8", "vllm_c8_prebuild_w4"],
            metric="rtf",
            statistic="p95_delta",
            unit="",
            direction="prebuild w4 - original; positive means larger tail",
            estimate=_bootstrap_delta(
                values_by_id["vllm_c8"]["rtf"],
                values_by_id["vllm_c8_prebuild_w4"]["rtf"],
                _stat_p95,
                seed=BOOTSTRAP_SEED + 9,
            ),
            decision="ci95_low_gt_0",
            interpretation=(
                "The RTF tail penalty also excludes zero; keep prebuild w4 as "
                "offline diagnostic evidence, not online parity."
            ),
        ),
    ]
    by_comparison = _by_comparison(comparison_rows)

    checks = [
        _check(
            "all required per-sample artifacts are present",
            not missing,
            f"missing={missing}",
        ),
        _check(
            "strict warmed c4 SGLang tail beats vLLM",
            strict_sglang["latency_s"]["p95"] < strict_vllm["latency_s"]["p95"]
            and strict_sglang["rtf"]["p95"] < strict_vllm["rtf"]["p95"]
            and strict_sglang["latency_s"]["mean"] < strict_vllm["latency_s"]["mean"]
            and strict_sglang["rtf"]["mean"] < strict_vllm["rtf"]["mean"],
            (
                "lat_mean/p95 "
                f"SGLang={_fmt(strict_sglang['latency_s']['mean'])}/"
                f"{_fmt(strict_sglang['latency_s']['p95'])}s, "
                f"vLLM={_fmt(strict_vllm['latency_s']['mean'])}/"
                f"{_fmt(strict_vllm['latency_s']['p95'])}s; "
                "rtf_mean/p95 "
                f"SGLang={_fmt(strict_sglang['rtf']['mean'])}/"
                f"{_fmt(strict_sglang['rtf']['p95'])}, "
                f"vLLM={_fmt(strict_vllm['rtf']['mean'])}/"
                f"{_fmt(strict_vllm['rtf']['p95'])}"
            ),
        ),
        _check(
            "strict warmed c4 quality does not regress",
            strict_sglang["accuracy"] >= strict_vllm["accuracy"]
            and strict_sglang["wer_corpus"] <= strict_vllm["wer_corpus"],
            (
                f"accuracy SGLang={_pct(strict_sglang['accuracy'])}, "
                f"vLLM={_pct(strict_vllm['accuracy'])}; "
                f"WER SGLang={_pct(strict_sglang['wer_corpus'])}, "
                f"vLLM={_pct(strict_vllm['wer_corpus'])}"
            ),
        ),
        _check(
            "SGLang stress keeps zero per-sample failures",
            all(row["failures"] == 0 for row in rows if row["group"] == "sglang_stress"),
            "failures="
            + ", ".join(
                f"{row['case_id']}={row['failures']}"
                for row in rows
                if row["group"] == "sglang_stress"
            ),
        ),
        _check(
            "SGLang c8 remains the throughput peak and c16 remains saturation",
            sglang_c8["throughput_qps"] > sglang_c16["throughput_qps"]
            and sglang_c16["latency_s"]["p95"] > sglang_c8["latency_s"]["p95"],
            (
                f"c8_qps={_fmt(sglang_c8['throughput_qps'])}, "
                f"c16_qps={_fmt(sglang_c16['throughput_qps'])}; "
                f"c8_p95={_fmt(sglang_c8['latency_s']['p95'])}s, "
                f"c16_p95={_fmt(sglang_c16['latency_s']['p95'])}s"
            ),
        ),
        _check(
            "long synthetic c8 tail remains faster than real time",
            long_c8["rtf"]["p95"] < 1.0 and long_c8["failures"] == 0,
            (
                f"long_c8_rtf_p95={_fmt(long_c8['rtf']['p95'])}, "
                f"failures={long_c8['failures']}"
            ),
        ),
        _check(
            "vLLM c8 prebuild w4 improves throughput but exposes tail",
            vllm_w4["throughput_qps"] > vllm_c8["throughput_qps"]
            and vllm_w4["latency_s"]["p95"] > vllm_c8["latency_s"]["p95"],
            (
                f"original_c8_p95={_fmt(vllm_c8['latency_s']['p95'])}s, "
                f"w4_p95={_fmt(vllm_w4['latency_s']['p95'])}s; "
                f"original_qps={_fmt(vllm_c8['throughput_qps'])}, "
                f"w4_qps={_fmt(vllm_w4['throughput_qps'])}"
            ),
        ),
        _check(
            "strict c4 latency mean bootstrap advantage excludes zero",
            _float(
                by_comparison["strict_c4_latency_mean_advantage"].get("ci95_low")
            )
            > 0.0,
            (
                "advantage_ci95="
                f"{_fmt(by_comparison['strict_c4_latency_mean_advantage']['ci95_low'])}/"
                f"{_fmt(by_comparison['strict_c4_latency_mean_advantage']['ci95_high'])}s"
            ),
        ),
        _check(
            "strict c4 RTF bootstrap overlap is explicitly caveated",
            _float(by_comparison["strict_c4_rtf_mean_advantage"].get("point_estimate"))
            > 0.0
            and _float(by_comparison["strict_c4_rtf_p95_advantage"].get("point_estimate"))
            > 0.0
            and (
                _float(by_comparison["strict_c4_rtf_mean_advantage"].get("ci95_low"))
                <= 0.0
                or _float(by_comparison["strict_c4_rtf_p95_advantage"].get("ci95_low"))
                <= 0.0
            ),
            (
                "rtf_mean_ci95="
                f"{_fmt(by_comparison['strict_c4_rtf_mean_advantage']['ci95_low'])}/"
                f"{_fmt(by_comparison['strict_c4_rtf_mean_advantage']['ci95_high'])}; "
                "rtf_p95_ci95="
                f"{_fmt(by_comparison['strict_c4_rtf_p95_advantage']['ci95_low'])}/"
                f"{_fmt(by_comparison['strict_c4_rtf_p95_advantage']['ci95_high'])}"
            ),
        ),
        _check(
            "strict c4 latency p95 point advantage remains positive",
            _float(
                by_comparison["strict_c4_latency_p95_advantage"].get(
                    "point_estimate"
                )
            )
            > 0.0,
            (
                "latency_p95_advantage="
                f"{_fmt(by_comparison['strict_c4_latency_p95_advantage']['point_estimate'])}s; "
                "ci95="
                f"{_fmt(by_comparison['strict_c4_latency_p95_advantage']['ci95_low'])}/"
                f"{_fmt(by_comparison['strict_c4_latency_p95_advantage']['ci95_high'])}s"
            ),
        ),
        _check(
            "SGLang c16 tail penalty bootstrap excludes zero",
            _float(
                by_comparison["sglang_c16_vs_c8_latency_p95_penalty"].get(
                    "ci95_low"
                )
            )
            > 0.0
            and _float(
                by_comparison["sglang_c16_vs_c8_rtf_p95_penalty"].get("ci95_low")
            )
            > 0.0,
            (
                "latency_ci95="
                f"{_fmt(by_comparison['sglang_c16_vs_c8_latency_p95_penalty']['ci95_low'])}/"
                f"{_fmt(by_comparison['sglang_c16_vs_c8_latency_p95_penalty']['ci95_high'])}s; "
                "rtf_ci95="
                f"{_fmt(by_comparison['sglang_c16_vs_c8_rtf_p95_penalty']['ci95_low'])}/"
                f"{_fmt(by_comparison['sglang_c16_vs_c8_rtf_p95_penalty']['ci95_high'])}"
            ),
        ),
        _check(
            "long synthetic c8 bootstrap p95 RTF upper bound remains real-time",
            _float(
                by_comparison["synthetic_long_c8_rtf_p95_realtime_margin"].get(
                    "ci95_high"
                )
            )
            < 1.0,
            (
                "long_c8_rtf_p95_ci95="
                f"{_fmt(by_comparison['synthetic_long_c8_rtf_p95_realtime_margin']['ci95_low'])}/"
                f"{_fmt(by_comparison['synthetic_long_c8_rtf_p95_realtime_margin']['ci95_high'])}"
            ),
        ),
        _check(
            "vLLM prebuild w4 tail penalty bootstrap excludes zero",
            _float(
                by_comparison["vllm_w4_vs_original_latency_p95_penalty"].get(
                    "ci95_low"
                )
            )
            > 0.0
            and _float(
                by_comparison["vllm_w4_vs_original_rtf_p95_penalty"].get("ci95_low")
            )
            > 0.0,
            (
                "latency_ci95="
                f"{_fmt(by_comparison['vllm_w4_vs_original_latency_p95_penalty']['ci95_low'])}/"
                f"{_fmt(by_comparison['vllm_w4_vs_original_latency_p95_penalty']['ci95_high'])}s; "
                "rtf_ci95="
                f"{_fmt(by_comparison['vllm_w4_vs_original_rtf_p95_penalty']['ci95_low'])}/"
                f"{_fmt(by_comparison['vllm_w4_vs_original_rtf_p95_penalty']['ci95_high'])}"
            ),
        ),
    ]
    required_failures = [
        check for check in checks if check["required"] and check["status"] != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "bootstrap_rows_total": len(comparison_rows),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check["status"] == "PASS"),
            "required_failures": len(required_failures),
            "strict_c4_sglang_latency_p95_s": strict_sglang["latency_s"]["p95"],
            "strict_c4_vllm_latency_p95_s": strict_vllm["latency_s"]["p95"],
            "strict_c4_sglang_rtf_p95": strict_sglang["rtf"]["p95"],
            "strict_c4_vllm_rtf_p95": strict_vllm["rtf"]["p95"],
            "sglang_c8_qps": sglang_c8["throughput_qps"],
            "sglang_c16_qps": sglang_c16["throughput_qps"],
            "long_c8_rtf_p95": long_c8["rtf"]["p95"],
            "vllm_w4_latency_p95_s": vllm_w4["latency_s"]["p95"],
            "strict_c4_latency_mean_advantage_ci95_low_s": by_comparison[
                "strict_c4_latency_mean_advantage"
            ]["ci95_low"],
            "strict_c4_latency_mean_advantage_ci95_high_s": by_comparison[
                "strict_c4_latency_mean_advantage"
            ]["ci95_high"],
            "strict_c4_rtf_mean_advantage_ci95_low": by_comparison[
                "strict_c4_rtf_mean_advantage"
            ]["ci95_low"],
            "long_c8_rtf_p95_ci95_high": by_comparison[
                "synthetic_long_c8_rtf_p95_realtime_margin"
            ]["ci95_high"],
            "share_scope": (
                "Per-sample tail and distribution appendix for strict c4, "
                "SGLang stress, synthetic short/long speech, vLLM diagnostics, "
                "and deterministic bootstrap uncertainty checks."
            ),
        },
        "checks": checks,
        "rows": rows,
        "bootstrap_comparisons": comparison_rows,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni Tail Confidence Appendix",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这页只使用已审计 raw `per_sample` 结果，不新增 benchmark。它补充 mean/p95 以外的",
        "p50/p90/IQR/max 视角，帮助 reviewer 判断 headline、压力窗口和长短文结论是否只是偶然尾部。",
        "",
        "## 1. Machine Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | `{summary['ready']}` |",
        f"| Rows | `{summary['rows_total']}` |",
        f"| Bootstrap comparisons | `{summary['bootstrap_rows_total']}` |",
        f"| Bootstrap draws | `{summary['bootstrap_draws']}` |",
        f"| Checks | `{summary['checks_passed']}/{summary['checks_total']}` |",
        f"| Required failures | `{summary['required_failures']}` |",
        "",
        "## 2. Distribution Rows",
        "",
        "| Group | Case | n | Success | Accuracy | WER | QPS | Lat p50/p90/p95/max | RTF p50/p90/p95/max | IQR Lat / RTF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['group']} | {row['label']} | {row['n_effective']} | "
            f"{row['success']}/{row['n_effective']} | {_pct(row['accuracy'])} | "
            f"{_pct(row['wer_corpus'])} | {_fmt(row['throughput_qps'])} | "
            f"{_fmt(row['latency_s']['p50'])}/"
            f"{_fmt(row['latency_s']['p90'])}/"
            f"{_fmt(row['latency_s']['p95'])}/"
            f"{_fmt(row['latency_s']['max'])}s | "
            f"{_fmt(row['rtf']['p50'])}/"
            f"{_fmt(row['rtf']['p90'])}/"
            f"{_fmt(row['rtf']['p95'])}/"
            f"{_fmt(row['rtf']['max'])} | "
            f"{_fmt(row['latency_s']['iqr'])}s / {_fmt(row['rtf']['iqr'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Bootstrap Comparisons",
            "",
            "| Comparison | Metric | Stat | Point | 95% CI | Decision | Interpretation |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["bootstrap_comparisons"]:
        suffix = row.get("unit") or ""
        point = _fmt(row.get("point_estimate"), suffix=suffix)
        ci = (
            f"{_fmt(row.get('ci95_low'), suffix=suffix)}/"
            f"{_fmt(row.get('ci95_high'), suffix=suffix)}"
        )
        lines.append(
            "| "
            f"{row['label']} | {row['metric']} | {row['statistic']} | "
            f"{point} | {ci} | {row['decision']} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Interpretation",
            "",
            "- Strict warmed c=4 的 SGLang latency/RTF mean 和 p95 同时优于优化版 vLLM，quality/WER 不退化。",
            "- Bootstrap 显示 strict c=4 mean latency 优势的 95% CI 排除 0；RTF 的点估计也优，但 CI 有重叠，所以不要单独用 RTF 做显著性表述。",
            "- SGLang c=8 是当前 throughput peak；c=16 的 tail 更高且 QPS 回落，因此只作为 saturation boundary。",
            "- Synthetic long c=8 的 RTF p95 仍低于 1，说明长文本/长语音 guardrail 快于实时。",
            "- vLLM c=8 prebuild w4 提高 runner/engine QPS，但 per-request tail 更高；这正是 offline diagnostic，不升级成 online parity。",
            "",
            "## 5. Checks",
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
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni per-sample tail confidence appendix."
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
    _save_json(payload, json_output)
    _save_text(build_markdown(payload), output)
    print(
        "Tail confidence appendix written: "
        f"{_relative(root, output)} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
