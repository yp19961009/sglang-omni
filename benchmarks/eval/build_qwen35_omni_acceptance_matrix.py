# SPDX-License-Identifier: Apache-2.0
"""Build a per-regime acceptance matrix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.eval.build_qwen35_omni_headline_scorecard import build_scorecard
from benchmarks.eval.summarize_qwen35_omni_report_artifacts import (
    preproc_rows,
    sglang_stage_rows,
    sglang_stress_rows,
    synthetic_rows,
    synthetic_stage_rows,
    vllm_admission_diagnosis_rows,
    vllm_overhead_rows,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return ""


def _row_by_key(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
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


def _label_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row.get("label") == label:
            return row
    return {}


def _metric_line(parts: list[tuple[str, Any, int, str]]) -> str:
    formatted: list[str] = []
    for label, value, digits, suffix in parts:
        if suffix == "%":
            formatted.append(f"{label}={_pct(value, digits)}")
        else:
            formatted.append(f"{label}={_fmt(value, digits)}{suffix}")
    return ", ".join(formatted)


def _stage_interaction_summary(root: Path) -> dict[str, Any]:
    path = root / "results/qwen35_report_audit_20260619/stage_interaction_summary.json"
    if not path.is_file():
        return {}
    return _load_json(path).get("summary", {})


def _add_row(
    rows: list[dict[str, Any]],
    *,
    regime: str,
    pressure: str,
    serving_status: str,
    accepted: bool,
    expected_shape: str,
    key_metrics: str,
    evidence: list[str],
    action: str,
) -> None:
    rows.append(
        {
            "regime": regime,
            "pressure": pressure,
            "serving_status": serving_status,
            "evidence_status": "PASS" if accepted else "FAIL",
            "accepted": accepted,
            "expected_shape": expected_shape,
            "key_metrics": key_metrics,
            "evidence": evidence,
            "action": action,
        }
    )


def build_matrix(root: Path) -> dict[str, Any]:
    root = root.resolve()
    scorecard = build_scorecard(root)
    score_checks = scorecard["summary"]["checks"]
    stress = sglang_stress_rows(root)
    stress_stages = sglang_stage_rows(root)
    synthetic = synthetic_rows(root)
    synthetic_stages = synthetic_stage_rows(root)
    preproc = preproc_rows(root)
    vllm_overhead = vllm_overhead_rows(root)
    vllm_admission = vllm_admission_diagnosis_rows(root)
    stage_summary = _stage_interaction_summary(root)

    rows: list[dict[str, Any]] = []

    c4_cmp = scorecard["strict_c4_comparison"]
    sglang_c4 = c4_cmp["sglang"]
    vllm_c4 = c4_cmp["vllm"]
    _add_row(
        rows,
        regime="strict_runtime_comparison",
        pressure="Video-AMME ci-50 warmed c=4",
        serving_status="recommended_strict_baseline",
        accepted=bool(score_checks["strict_c4_sglang_latency_rtf_win"])
        and bool(score_checks["strict_c4_accuracy_wer_preserved"]),
        expected_shape="SGLang beats optimized vLLM on warmed latency/RTF and preserves accuracy/WER.",
        key_metrics=(
            f"SGLang lat={_fmt(sglang_c4['latency_mean_s'])}/{_fmt(sglang_c4['latency_p95_s'])}s, "
            f"RTF={_fmt(sglang_c4['rtf_mean'], 4)}/{_fmt(sglang_c4['rtf_p95'], 4)}, "
            f"acc={_pct(sglang_c4['accuracy'])}, WER={_pct(sglang_c4['wer_corpus'], 2)}; "
            f"vLLM lat={_fmt(vllm_c4['latency_mean_s'])}/{_fmt(vllm_c4['latency_p95_s'])}s, "
            f"RTF={_fmt(vllm_c4['rtf_mean'], 4)}/{_fmt(vllm_c4['rtf_p95'], 4)}, "
            f"acc={_pct(vllm_c4['accuracy'])}, WER={_pct(vllm_c4['wer_corpus'], 2)}"
        ),
        evidence=[sglang_c4["artifact"], vllm_c4["artifact"]],
        action="Use this as the main cross-runtime headline.",
    )

    peak = scorecard["sglang_stress"]["throughput_peak"]
    c8 = _row_by_concurrency(stress, 8)
    c16 = _row_by_concurrency(stress, 16)
    for concurrency in [1, 2, 4, 8, 16]:
        row = _row_by_concurrency(stress, concurrency)
        stage = _row_by_concurrency(stress_stages, concurrency)
        hop_p95 = float(stage.get("talker_to_code2wav_hop_p95_ms") or 999.0)
        decode_p95 = float(stage.get("code2wav_decode_p95_ms") or 999.0)
        base_ok = (
            int(row.get("n") or 0) >= 50
            and float(row.get("accuracy") or 0.0) >= 0.65
            and float(row.get("wer_corpus") or 1.0) <= 0.05
            and hop_p95 <= 30.0
            and decode_p95 <= 35.0
        )
        if concurrency in {1, 2, 4}:
            accepted = base_ok and "talker_ar" in str(stage.get("top_stage", ""))
            expected = "Stable single/low-concurrency run; talker AR is the main tail, not code2wav."
            serving_status = "recommended_serving_window"
            action = "Use for steady-state quality/latency shape."
        elif concurrency == 8:
            accepted = base_ok and int(peak.get("concurrency") or 0) == 8
            expected = "Throughput peak; admission queue starts to matter but stage handoff stays healthy."
            serving_status = "recommended_peak_throughput"
            action = "Use as the current high-concurrency sweet spot."
        else:
            accepted = (
                base_ok
                and float(row.get("throughput_qps") or 0.0)
                < float(c8.get("throughput_qps") or 0.0)
                and float(stage.get("preproc_stage_avg_ms") or 0.0) > 1000.0
            )
            expected = "Saturation evidence: lower QPS than c=8 and large preprocessing admission queue."
            serving_status = "not_recommended_saturation"
            action = "Do not present c=16 as a recommended serving point."
        _add_row(
            rows,
            regime="sglang_videoamme_stress",
            pressure=f"c={concurrency}",
            serving_status=serving_status,
            accepted=accepted,
            expected_shape=expected,
            key_metrics=_metric_line(
                [
                    ("acc", row.get("accuracy"), 1, "%"),
                    ("WER", row.get("wer_corpus"), 2, "%"),
                    ("lat_mean", row.get("latency_mean_s"), 3, "s"),
                    ("lat_p95", row.get("latency_p95_s"), 3, "s"),
                    ("QPS", row.get("throughput_qps"), 3, ""),
                    ("hop_p95", hop_p95, 1, "ms"),
                    ("decode_p95", decode_p95, 1, "ms"),
                ]
            ),
            evidence=[row.get("result_json", ""), row.get("wer_json", ""), stage.get("profile_json", "")],
            action=action,
        )

    for scenario in ["short", "long"]:
        for concurrency in [1, 4, 8]:
            row = _synthetic_row(synthetic, scenario=scenario, concurrency=concurrency)
            stage = next(
                (
                    item
                    for item in synthetic_stages
                    if item.get("scenario") == scenario
                    and int(item.get("concurrency") or 0) == concurrency
                ),
                {},
            )
            hop_p95 = float(stage.get("talker_to_code2wav_hop_p95_ms") or 999.0)
            decode_p95 = float(stage.get("code2wav_decode_p95_ms") or 999.0)
            accepted = (
                int(row.get("n") or 0) > 0
                and float(row.get("rtf_mean") or 9.0) < 1.0
                and hop_p95 <= 30.0
                and decode_p95 <= 35.0
            )
            _add_row(
                rows,
                regime="sglang_synthetic_speech",
                pressure=f"{scenario} c={concurrency}",
                serving_status="speech_generation_regression_guard",
                accepted=accepted,
                expected_shape="Short/long text-to-speech output stays faster than real time; code2wav boundary remains small.",
                key_metrics=_metric_line(
                    [
                        ("text_words", row.get("target_words"), 0, ""),
                        ("audio", row.get("audio_duration_mean_s"), 1, "s"),
                        ("lat_mean", row.get("latency_mean_s"), 3, "s"),
                        ("RTF", row.get("rtf_mean"), 4, ""),
                        ("QPS", row.get("throughput_qps"), 3, ""),
                        ("hop_p95", hop_p95, 1, "ms"),
                        ("decode_p95", decode_p95, 1, "ms"),
                    ]
                ),
                evidence=[row.get("result_json", ""), stage.get("profile_json", "")],
                action="Use as the short/long text-input and speech-output guardrail.",
            )

    original_c8 = _label_row(vllm_admission, "vLLM-c8")
    _add_row(
        rows,
        regime="vllm_offline_diagnostic",
        pressure="original c=8",
        serving_status="diagnostic_prompt_feed_limited",
        accepted=original_c8.get("diagnosis") == "prompt_feed_limited"
        and float(original_c8.get("runner_overhead_pct_wall") or 0.0) > 70.0
        and float(original_c8.get("batch_admission_span_avg_ms") or 0.0) > 30000.0,
        expected_shape="Original vLLM offline c=8 is limited by host prompt build/feed admission, not engine stage boundaries.",
        key_metrics=(
            f"runner_overhead={_fmt(original_c8.get('runner_overhead_pct_wall'), 1)}%, "
            f"admission_avg={_fmt(original_c8.get('batch_admission_span_avg_ms'), 1)}ms, "
            f"engine_QPS={_fmt(original_c8.get('engine_qps'), 4)}"
        ),
        evidence=[original_c8.get("result_json", ""), original_c8.get("run_log", "")],
        action="Use only as diagnosis; do not use as strict online parity evidence.",
    )

    prebuild_w1 = _label_row(vllm_overhead, "vLLM-c8-prebuild-w1")
    prebuild_w4 = _label_row(vllm_overhead, "vLLM-c8-prebuild-w4")
    prebuild_w4_diag = _label_row(vllm_admission, "vLLM-c8-prebuild-w4")
    _add_row(
        rows,
        regime="vllm_offline_diagnostic",
        pressure="prebuild c=8 workers=4",
        serving_status="optimized_offline_diagnostic",
        accepted=bool(prebuild_w4.get("prebuild_prompts"))
        and int(prebuild_w4.get("prebuild_workers") or 0) == 4
        and float(prebuild_w4.get("prompt_build_wall_s") or 999.0)
        < float(prebuild_w1.get("prompt_build_wall_s") or 0.0)
        and float(prebuild_w4.get("runner_qps") or 0.0)
        > float(prebuild_w1.get("runner_qps") or 999.0)
        and prebuild_w4_diag.get("diagnosis") == "engine_or_workload_limited",
        expected_shape="Four prompt-build workers reduce runner wall time and expose later engine/workload tail.",
        key_metrics=_metric_line(
            [
                ("prompt_wall", prebuild_w4.get("prompt_build_wall_s"), 1, "s"),
                ("runner_QPS", prebuild_w4.get("runner_qps"), 4, ""),
                ("engine_QPS", prebuild_w4.get("engine_qps"), 4, ""),
                ("admission_avg", prebuild_w4_diag.get("batch_admission_span_avg_ms"), 1, "ms"),
            ]
        ),
        evidence=[prebuild_w4.get("path", ""), prebuild_w4_diag.get("run_log", "")],
        action="Use as optimized offline diagnostic; require online ingress plus WER before c=8 parity claims.",
    )

    preproc1 = _row_by_key(preproc, "setting", "preproc=1 baseline")
    preproc2 = _row_by_key(preproc, "setting", "preproc=2")
    _add_row(
        rows,
        regime="negative_optimization",
        pressure="PREPROCESSING_MAX_CONCURRENCY=2 at c=8",
        serving_status="anti_recipe_regression",
        accepted=float(preproc2.get("throughput_qps") or 0.0)
        < float(preproc1.get("throughput_qps") or 0.0)
        and float(preproc2.get("latency_mean_s") or 0.0)
        > float(preproc1.get("latency_mean_s") or 0.0),
        expected_shape="Naive preprocessing widening reduces throughput and increases latency.",
        key_metrics=(
            f"baseline_QPS={_fmt(preproc1.get('throughput_qps'))}, "
            f"preproc2_QPS={_fmt(preproc2.get('throughput_qps'))}, "
            f"baseline_lat={_fmt(preproc1.get('latency_mean_s'))}s, "
            f"preproc2_lat={_fmt(preproc2.get('latency_mean_s'))}s"
        ),
        evidence=[
            str(
                root
                / "results/qwen35_sglang_preproc2_mr8_c8_20260619/benchmark_audio_50_c8_preproc2_profile_skipwer/videoamme_results.json"
            )
        ],
        action="Keep preprocessing concurrency at 1 unless placement/admission is redesigned.",
    )

    preproc4_path = (
        root
        / "results/qwen35_sglang_mr8_preproc4_stress_20260619/warmup_audio_50_c8_skipwer/videoamme_results.json"
    )
    preproc4 = _load_json(preproc4_path)
    _add_row(
        rows,
        regime="negative_optimization",
        pressure="PREPROCESSING_MAX_CONCURRENCY=4 at c=8",
        serving_status="anti_recipe_failure",
        accepted=int(preproc4.get("summary", {}).get("failed") or 0) > 0,
        expected_shape="More aggressive preprocessing widening causes request failures/OOM risk.",
        key_metrics=(
            f"failed={preproc4.get('summary', {}).get('failed')}, "
            f"accuracy={_pct(preproc4.get('summary', {}).get('accuracy'))}"
        ),
        evidence=[str(preproc4_path)],
        action="Do not use preproc=4 in the current recipe.",
    )

    required_flags = [
        "sglang_talker_to_code2wav_healthy",
        "sglang_code2wav_decode_not_bottleneck",
        "vllm_original_c8_prompt_feed_limited",
        "preprocessing_parallelism_regresses",
    ]
    _add_row(
        rows,
        regime="stage_connection_health",
        pressure="all audited boundaries",
        serving_status="cross_stage_guardrail",
        accepted=all(stage_summary.get(flag) for flag in required_flags),
        expected_shape="Stage-to-stage handoff is separated from admission/queue and stage-local compute bottlenecks.",
        key_metrics=", ".join(f"{flag}={stage_summary.get(flag)}" for flag in required_flags),
        evidence=[
            str(root / "results/qwen35_report_audit_20260619/stage_interaction_summary.json")
        ],
        action="Use this row to answer whether stage boundaries themselves are the bottleneck.",
    )

    status_counts = Counter(row["evidence_status"] for row in rows)
    serving_counts = Counter(row["serving_status"] for row in rows)
    failed_rows = [row for row in rows if row["evidence_status"] != "PASS"]
    return {
        "root": str(root),
        "summary": {
            "ready": not failed_rows,
            "rows_total": len(rows),
            "rows_passed": status_counts.get("PASS", 0),
            "rows_failed": len(failed_rows),
            "evidence_status_counts": dict(sorted(status_counts.items())),
            "serving_status_counts": dict(sorted(serving_counts.items())),
        },
        "rows": rows,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Acceptance Matrix\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Rows | {summary['rows_passed']}/{summary['rows_total']} |")
    print()
    print("| Regime | Pressure | Serving Status | Evidence | Key Metrics | Action |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in payload["rows"]:
        print(
            f"| {row['regime']} | {row['pressure']} | {row['serving_status']} | "
            f"{row['evidence_status']} | {row['key_metrics']} | {row['action']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the per-regime acceptance matrix for Qwen3.5-Omni."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_matrix(root)
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
