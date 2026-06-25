# SPDX-License-Identifier: Apache-2.0
"""Build the headline metric scorecard for the Qwen3.5-Omni handoff report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.eval.summarize_qwen35_omni_report_artifacts import (
    sglang_stress_rows,
    synthetic_rows,
    vllm_admission_diagnosis_rows,
    vllm_overhead_rows,
)
from benchmarks.eval.verify_qwen35_omni_report_claims import (
    _load_json,
    _paths,
    _request_slice_metrics,
)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _pct_lower(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (old - new) / old * 100.0


def _pct_delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new / old - 1.0) * 100.0


def _row_by_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row.get("label") == label:
            return row
    return {}


def _row_by_concurrency(rows: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("concurrency") or 0) == concurrency:
            return row
    return {}


def _synthetic_row(
    rows: list[dict[str, Any]], *, scenario: str, concurrency: int
) -> dict[str, Any]:
    for row in rows:
        if row.get("scenario") == scenario and int(row.get("concurrency") or 0) == concurrency:
            return row
    return {}


def _stage_summary(root: Path) -> dict[str, Any]:
    path = root / "results/qwen35_report_audit_20260619/stage_interaction_summary.json"
    if path.is_file():
        return _load_json(path).get("summary", {})
    return {}


def build_scorecard(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = _paths(root)

    vllm_c4 = _request_slice_metrics(paths["vllm_c4"], skip_first=4)
    sglang_c4 = _request_slice_metrics(paths["sglang_c4"], skip_first=4)
    vllm_wer = _load_json(paths["vllm_c4_wer"])["summary"]["wer_corpus"]
    sglang_wer = _load_json(paths["sglang_c4_wer"])["summary"]["wer_corpus"]

    stress_rows = sglang_stress_rows(root)
    stress_peak = max(stress_rows, key=lambda row: float(row["throughput_qps"]))
    stress_c8 = _row_by_concurrency(stress_rows, 8)
    stress_c16 = _row_by_concurrency(stress_rows, 16)

    synthetic = synthetic_rows(root)
    long_c8 = _synthetic_row(synthetic, scenario="long", concurrency=8)

    overhead = vllm_overhead_rows(root)
    vllm_c8 = _row_by_label(overhead, "vLLM-c8")
    vllm_prebuild_w1 = _row_by_label(overhead, "vLLM-c8-prebuild-w1")
    vllm_prebuild_w4 = _row_by_label(overhead, "vLLM-c8-prebuild-w4")
    admission = vllm_admission_diagnosis_rows(root)
    vllm_c8_admission = _row_by_label(admission, "vLLM-c8")
    vllm_w4_admission = _row_by_label(admission, "vLLM-c8-prebuild-w4")

    stage_summary = _stage_summary(root)

    checks = {
        "strict_c4_sglang_latency_rtf_win": (
            sglang_c4["latency_mean_s"] < vllm_c4["latency_mean_s"]
            and sglang_c4["latency_p95_s"] < vllm_c4["latency_p95_s"]
            and sglang_c4["rtf_mean"] < vllm_c4["rtf_mean"]
            and sglang_c4["rtf_p95"] < vllm_c4["rtf_p95"]
        ),
        "strict_c4_accuracy_wer_preserved": (
            sglang_c4["accuracy"] >= vllm_c4["accuracy"] and sglang_wer <= vllm_wer
        ),
        "sglang_stress_c8_is_peak": int(stress_peak.get("concurrency") or 0) == 8,
        "sglang_c16_regresses_vs_c8": (
            float(stress_c16.get("throughput_qps") or 0)
            < float(stress_c8.get("throughput_qps") or 0)
        ),
        "long_c8_faster_than_real_time": float(long_c8.get("rtf_mean") or 9) < 1.0,
        "stage_connections_healthy": bool(
            stage_summary.get("sglang_talker_to_code2wav_healthy")
        )
        and bool(stage_summary.get("sglang_code2wav_decode_not_bottleneck")),
        "vllm_original_c8_prompt_feed_limited": bool(
            stage_summary.get("vllm_original_c8_prompt_feed_limited")
        )
        and vllm_c8_admission.get("diagnosis") == "prompt_feed_limited",
        "preprocessing_parallelism_regresses": bool(
            stage_summary.get("preprocessing_parallelism_regresses")
        ),
        "vllm_w4_prebuild_improves_runner_wall": (
            float(vllm_prebuild_w4.get("prompt_build_wall_s") or 0)
            < float(vllm_prebuild_w1.get("prompt_build_wall_s") or 0)
            and float(vllm_prebuild_w4.get("runner_qps") or 0)
            > float(vllm_prebuild_w1.get("runner_qps") or 0)
        ),
    }

    return {
        "root": str(root),
        "summary": {
            "ready": all(checks.values()),
            "checks_passed": sum(1 for value in checks.values() if value),
            "checks_total": len(checks),
            "checks": checks,
        },
        "strict_c4_comparison": {
            "scope": "Video-AMME ci-50, warmed skip-first-4, c=4",
            "sglang": {
                **sglang_c4,
                "wer_corpus": sglang_wer,
                "artifact": str(paths["sglang_c4"]),
            },
            "vllm": {
                **vllm_c4,
                "wer_corpus": vllm_wer,
                "artifact": str(paths["vllm_c4"]),
            },
            "relative_sglang_lower_pct": {
                "latency_mean": _pct_lower(
                    sglang_c4["latency_mean_s"], vllm_c4["latency_mean_s"]
                ),
                "latency_p95": _pct_lower(
                    sglang_c4["latency_p95_s"], vllm_c4["latency_p95_s"]
                ),
                "rtf_mean": _pct_lower(sglang_c4["rtf_mean"], vllm_c4["rtf_mean"]),
                "rtf_p95": _pct_lower(sglang_c4["rtf_p95"], vllm_c4["rtf_p95"]),
            },
        },
        "sglang_stress": {
            "rows": stress_rows,
            "throughput_peak": stress_peak,
            "c16_vs_c8_qps_delta_pct": _pct_delta(
                float(stress_c16.get("throughput_qps") or 0),
                float(stress_c8.get("throughput_qps") or 0),
            ),
        },
        "synthetic_long_c8": long_c8,
        "vllm_c8_diagnostics": {
            "original": {
                **vllm_c8,
                "admission_diagnosis": vllm_c8_admission,
            },
            "prebuild_w1": vllm_prebuild_w1,
            "prebuild_w4": {
                **vllm_prebuild_w4,
                "admission_diagnosis": vllm_w4_admission,
            },
            "w4_vs_w1": {
                "prompt_build_wall_delta_pct": _pct_delta(
                    float(vllm_prebuild_w4.get("prompt_build_wall_s") or 0),
                    float(vllm_prebuild_w1.get("prompt_build_wall_s") or 0),
                ),
                "runner_qps_delta_pct": _pct_delta(
                    float(vllm_prebuild_w4.get("runner_qps") or 0),
                    float(vllm_prebuild_w1.get("runner_qps") or 0),
                ),
            },
        },
        "stage_interactions": stage_summary,
        "safe_claims": [
            "SGLang-Omni warmed c=4 is faster than optimized vLLM on latency and RTF while preserving accuracy/WER.",
            "SGLang c=8 is the measured throughput peak for the current Video-AMME ci-50 recipe.",
            "Long synthetic speech remains faster than real time at c=8.",
            "Code2wav decode and SGLang talker->code2wav handoff are not current bottlenecks.",
            "vLLM original c=8 offline wall throughput is prompt-feed/admission limited; prebuild w4 is a diagnostic, not online parity.",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def print_markdown(payload: dict[str, Any]) -> None:
    print("## Qwen3.5-Omni Headline Scorecard\n")
    summary = payload["summary"]
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print()

    cmp = payload["strict_c4_comparison"]
    sglang = cmp["sglang"]
    vllm = cmp["vllm"]
    print("| Runtime | Lat Mean | Lat P95 | RTF Mean | RTF P95 | Accuracy | WER |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, row in [("SGLang", sglang), ("vLLM", vllm)]:
        print(
            f"| {label} | {_fmt(row['latency_mean_s'])}s | "
            f"{_fmt(row['latency_p95_s'])}s | {_fmt(row['rtf_mean'], 4)} | "
            f"{_fmt(row['rtf_p95'], 4)} | {_fmt(row['accuracy'] * 100, 1)}% | "
            f"{_fmt(row['wer_corpus'] * 100, 2)}% |"
        )
    print()

    peak = payload["sglang_stress"]["throughput_peak"]
    long_c8 = payload["synthetic_long_c8"]
    vllm_w4 = payload["vllm_c8_diagnostics"]["prebuild_w4"]
    print("| Headline | Value |")
    print("| --- | --- |")
    print(f"| SGLang stress throughput peak | c={peak['concurrency']}, {_fmt(peak['throughput_qps'])} QPS |")
    print(
        "| Synthetic long c=8 | "
        f"audio {_fmt(long_c8['audio_duration_mean_s'], 1)}s, "
        f"latency {_fmt(long_c8['latency_mean_s'], 1)}s, RTF {_fmt(long_c8['rtf_mean'], 4)} |"
    )
    print(
        "| vLLM c8 prebuild w4 | "
        f"runner QPS {_fmt(vllm_w4['runner_qps'], 4)}, "
        f"engine QPS {_fmt(vllm_w4['engine_qps'], 4)}, "
        f"prompt wall {_fmt(vllm_w4['prompt_build_wall_s'], 1)}s |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni headline scorecard from report artifacts."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = build_scorecard(args.root.resolve())
    print_markdown(payload)
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = args.root.resolve() / output
        _save_json(payload, output)


if __name__ == "__main__":
    main()
