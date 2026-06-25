# SPDX-License-Identifier: Apache-2.0
"""Diagnose vLLM offline-runner prompt-feed/admission bottlenecks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.eval.summarize_vllm_offline_runner_overhead import (
    summarize as summarize_runner_overhead,
)
from benchmarks.eval.summarize_vllm_omni_log_stages import (
    summarize as summarize_log_stages,
)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def diagnose_case(
    *,
    label: str,
    result_json: Path,
    run_log: Path,
    skip_first_requests: int,
) -> dict[str, Any]:
    overhead = summarize_runner_overhead(result_json, label)
    stages = summarize_log_stages(
        run_log,
        label=label,
        skip_first_requests=skip_first_requests,
    )

    admission_avg_ms = float(stages["batch_admission_span_ms"]["avg"] or 0)
    admission_p95_ms = float(stages["batch_admission_span_ms"]["p95"] or 0)
    last_lag_avg_ms = float(stages["batch_last_engine_lag_ms"]["avg"] or 0)
    last_lag_p95_ms = float(stages["batch_last_engine_lag_ms"]["p95"] or 0)
    encoder_p95_ms = float(stages["encoder_mm_ms"]["p95"] or 0)
    c2w_drain_p95_ms = float(stages["talker_to_code2wav_drain_ms"]["p95"] or 0)
    overhead_pct = float(overhead["runner_overhead_pct_wall"])

    prompt_feed_limited = overhead_pct >= 50 and admission_avg_ms >= 10000
    engine_boundaries_clean = encoder_p95_ms < 60 and c2w_drain_p95_ms < 25

    if prompt_feed_limited and engine_boundaries_clean:
        diagnosis = "prompt_feed_limited"
        recommendation = (
            "prebuild or parallelize multimodal prompt construction before "
            "timed engine admission; then rerun wall-QPS comparison"
        )
    elif prompt_feed_limited:
        diagnosis = "mixed_prompt_feed_and_engine_boundary"
        recommendation = (
            "separate prompt construction from timed serving and inspect encoder "
            "or code2wav boundary tails"
        )
    else:
        diagnosis = "engine_or_workload_limited"
        recommendation = (
            "runner admission is not the dominant signal; inspect engine stage "
            "latency and generated-audio length distribution"
        )

    return {
        "label": label,
        "result_json": str(result_json),
        "run_log": str(run_log),
        "skip_first_requests": skip_first_requests,
        "concurrency": overhead["concurrency"],
        "requests": overhead["requests"],
        "wall_time_s": overhead["wall_time_s"],
        "runner_wall_time_s": overhead["runner_wall_time_s"],
        "engine_wall_time_s": overhead["engine_wall_time_s"],
        "prompt_build_wall_s": overhead["prompt_build_wall_s"],
        "runner_overhead_s": overhead["runner_overhead_s"],
        "runner_overhead_pct_wall": overhead_pct,
        "engine_overhead_s": overhead["engine_overhead_s"],
        "engine_overhead_pct_wall": overhead["engine_overhead_pct_wall"],
        "wall_qps": overhead["wall_qps"],
        "runner_qps": overhead["runner_qps"],
        "engine_qps": overhead["engine_qps"],
        "batch_max_qps": overhead["batch_max_qps"],
        "batch_admission_span_avg_ms": admission_avg_ms,
        "batch_admission_span_p95_ms": admission_p95_ms,
        "batch_last_engine_lag_avg_ms": last_lag_avg_ms,
        "batch_last_engine_lag_p95_ms": last_lag_p95_ms,
        "encoder_p95_ms": encoder_p95_ms,
        "talker_to_code2wav_drain_p95_ms": c2w_drain_p95_ms,
        "prompt_feed_limited": prompt_feed_limited,
        "engine_boundaries_clean": engine_boundaries_clean,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print(
        "| Label | c | Runner QPS | Engine QPS | Batch-max QPS | "
        "Runner Overhead | Engine Overhead | "
        "Admission Span Avg/P95 | Last Engine Lag Avg/P95 | "
        "Boundary P95 Encoder/C2W | Diagnosis |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for row in rows:
        print(
            "| {label} | {c} | {runner_qps} | {engine_qps} | {batch_qps} | "
            "{overhead}s ({overhead_pct}%) | "
            "{engine_overhead}s ({engine_overhead_pct}%) | "
            "{span_avg}/{span_p95}ms | "
            "{last_avg}/{last_p95}ms | "
            "{encoder_p95}/{c2w_p95}ms | {diagnosis} |".format(
                label=row["label"],
                c=row["concurrency"],
                runner_qps=_fmt(row["runner_qps"], 4),
                engine_qps=_fmt(row["engine_qps"], 4),
                batch_qps=_fmt(row["batch_max_qps"], 4),
                overhead=_fmt(row["runner_overhead_s"], 1),
                overhead_pct=_fmt(row["runner_overhead_pct_wall"], 1),
                engine_overhead=_fmt(row["engine_overhead_s"], 1),
                engine_overhead_pct=_fmt(row["engine_overhead_pct_wall"], 1),
                span_avg=_fmt(row["batch_admission_span_avg_ms"], 1),
                span_p95=_fmt(row["batch_admission_span_p95_ms"], 1),
                last_avg=_fmt(row["batch_last_engine_lag_avg_ms"], 1),
                last_p95=_fmt(row["batch_last_engine_lag_p95_ms"], 1),
                encoder_p95=_fmt(row["encoder_p95_ms"], 1),
                c2w_p95=_fmt(row["talker_to_code2wav_drain_p95_ms"], 1),
                diagnosis=row["diagnosis"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether a vLLM Qwen3.5 offline Video-AMME run is limited "
            "by prompt construction / engine admission rather than engine-side "
            "stage boundaries."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("LABEL", "RESULT_JSON", "RUN_LOG", "SKIP_FIRST_REQUESTS"),
        required=True,
        help=(
            "One case to diagnose. Repeat for multiple cases. "
            "SKIP_FIRST_REQUESTS should match the warmed slice used in the report."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        diagnose_case(
            label=label,
            result_json=Path(result_json),
            run_log=Path(run_log),
            skip_first_requests=int(skip_first_requests),
        )
        for label, result_json, run_log, skip_first_requests in args.case
    ]
    print_markdown(rows)
    if args.json_output:
        _save_json({"rows": rows}, args.json_output)


if __name__ == "__main__":
    main()
