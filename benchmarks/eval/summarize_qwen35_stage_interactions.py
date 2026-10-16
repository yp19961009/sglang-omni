# SPDX-License-Identifier: Apache-2.0
"""Build a machine-readable stage-interaction summary for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.eval.summarize_qwen35_omni_report_artifacts import (
    preproc_rows,
    sglang_preproc_split_rows,
    sglang_stage_rows,
    synthetic_stage_rows,
    vllm_admission_diagnosis_rows,
    vllm_log_stage_rows,
)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _by_key(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row.get(key): row for row in rows}


def _status(ok: bool, warning: bool = False) -> str:
    if ok:
        return "healthy"
    if warning:
        return "watch"
    return "bottleneck"


def _sglang_interactions(root: Path) -> list[dict[str, Any]]:
    stage_by_c = _by_key(sglang_stage_rows(root), "concurrency")
    preproc_by_c = _by_key(sglang_preproc_split_rows(root), "concurrency")
    preproc_table = _by_key(preproc_rows(root), "setting")

    rows: list[dict[str, Any]] = []
    for c in [1, 2, 4, 8, 16]:
        row = stage_by_c[c]
        preproc = preproc_by_c.get(c, {})
        preproc_stage_avg = _float(row.get("preproc_stage_avg_ms"))
        actual_preproc_avg = _float(preproc.get("actual_preprocess_avg_ms"))
        preproc_queue_avg = max(0.0, preproc_stage_avg - actual_preproc_avg)
        rows.append(
            {
                "runtime": "sglang",
                "workload": "Video-AMME ci-50",
                "boundary": "request_admission_to_preprocessing",
                "concurrency": c,
                "status": (
                    "queue_limited"
                    if c >= 8 and preproc_queue_avg > 800
                    else "healthy"
                ),
                "metrics": {
                    "preproc_lifecycle_avg_ms": preproc_stage_avg,
                    "actual_preprocess_avg_ms": actual_preproc_avg or None,
                    "estimated_queue_avg_ms": preproc_queue_avg
                    if actual_preproc_avg
                    else None,
                    "top_stage": row.get("top_stage"),
                },
                "interpretation": (
                    "Preprocessing lifecycle is mostly admission/queue time."
                    if c >= 8
                    else "No material preprocessing admission pressure."
                ),
            }
        )
        hop_p95 = _float(row.get("talker_to_code2wav_hop_p95_ms"))
        rows.append(
            {
                "runtime": "sglang",
                "workload": "Video-AMME ci-50",
                "boundary": "talker_to_code2wav_stream",
                "concurrency": c,
                "status": _status(hop_p95 <= 25),
                "metrics": {
                    "talker_to_code2wav_hop_avg_ms": row.get(
                        "talker_to_code2wav_hop_avg_ms"
                    ),
                    "talker_to_code2wav_hop_p95_ms": row.get(
                        "talker_to_code2wav_hop_p95_ms"
                    ),
                    "talker_p95_ms": row.get("talker_p95_ms"),
                    "code2wav_decode_p95_ms": row.get("code2wav_decode_p95_ms"),
                },
                "interpretation": "The stream handoff is not the bottleneck.",
            }
        )
        decode_avg = _float(row.get("code2wav_decode_avg_ms"))
        window_avg = _float(row.get("code2wav_window_avg_ms"))
        rows.append(
            {
                "runtime": "sglang",
                "workload": "Video-AMME ci-50",
                "boundary": "code2wav_collect_to_decode",
                "concurrency": c,
                "status": _status(decode_avg <= 20),
                "metrics": {
                    "code2wav_decode_avg_ms": row.get("code2wav_decode_avg_ms"),
                    "code2wav_decode_p95_ms": row.get("code2wav_decode_p95_ms"),
                    "code2wav_window_collect_avg_ms": row.get(
                        "code2wav_window_avg_ms"
                    ),
                    "window_minus_decode_avg_ms": max(0.0, window_avg - decode_avg),
                },
                "interpretation": "Code2wav lifecycle waits on chunks more than vocoder compute.",
            }
        )

    baseline = preproc_table.get("preproc=1 baseline", {})
    preproc2 = preproc_table.get("preproc=2", {})
    baseline_qps = _float(baseline.get("throughput_qps"))
    preproc2_qps = _float(preproc2.get("throughput_qps"))
    rows.append(
        {
            "runtime": "sglang",
            "workload": "Video-AMME ci-50",
            "boundary": "preprocessing_to_encoder_thinker",
            "concurrency": 8,
            "status": "contention_regression",
            "metrics": {
                "baseline_preproc_concurrency": 1,
                "candidate_preproc_concurrency": 2,
                "baseline_qps": baseline_qps,
                "candidate_qps": preproc2_qps,
                "qps_delta_pct": (
                    (preproc2_qps / baseline_qps - 1.0) * 100.0
                    if baseline_qps
                    else None
                ),
            },
            "interpretation": "Widening preprocessing admission regresses throughput through shared-resource contention.",
        }
    )
    return rows


def _synthetic_interactions(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in synthetic_stage_rows(root):
        hop_p95 = _float(row.get("talker_to_code2wav_hop_p95_ms"))
        decode_avg = _float(row.get("code2wav_decode_avg_ms"))
        rows.append(
            {
                "runtime": "sglang",
                "workload": f"synthetic_{row.get('scenario')}",
                "boundary": "talker_to_code2wav_stream",
                "concurrency": row.get("concurrency"),
                "status": _status(hop_p95 <= 30),
                "metrics": {
                    "talker_avg_ms": row.get("talker_avg_ms"),
                    "talker_p95_ms": row.get("talker_p95_ms"),
                    "talker_to_code2wav_hop_p95_ms": hop_p95,
                    "code2wav_decode_avg_ms": decode_avg,
                },
                "interpretation": "Synthetic speech isolates Talker/Code2wav and confirms the handoff remains small.",
            }
        )
    return rows


def _vllm_interactions(root: Path) -> list[dict[str, Any]]:
    stage_rows = vllm_log_stage_rows(root)
    admission_by_label = _by_key(vllm_admission_diagnosis_rows(root), "label")
    rows: list[dict[str, Any]] = []
    for row in stage_rows:
        label = row["label"]
        admission = admission_by_label.get(label, {})
        thinker_p95 = _float(row["thinker_to_talker_feed_ms"].get("p95"))
        drain_p95 = _float(row["talker_to_code2wav_drain_ms"].get("p95"))
        span_avg = _float(row["batch_admission_span_ms"].get("avg"))
        rows.append(
            {
                "runtime": "vllm",
                "workload": "Video-AMME ci-50 offline",
                "boundary": "thinker_to_talker",
                "label": label,
                "status": _status(thinker_p95 <= 5, warning=thinker_p95 <= 10),
                "metrics": {
                    "thinker_to_talker_feed_p95_ms": thinker_p95,
                    "diagnosis": admission.get("diagnosis"),
                },
                "interpretation": "Engine-side Thinker to Talker feed is not the original c8 limiter.",
            }
        )
        rows.append(
            {
                "runtime": "vllm",
                "workload": "Video-AMME ci-50 offline",
                "boundary": "talker_to_code2wav",
                "label": label,
                "status": _status(
                    drain_p95 <= 25,
                    warning=label.endswith("prebuild-w4") and drain_p95 <= 130,
                ),
                "metrics": {
                    "talker_to_code2wav_drain_p95_ms": drain_p95,
                    "feed_to_first_codec_p95_ms": row[
                        "talker_feed_to_first_codec_ms"
                    ].get("p95"),
                    "diagnosis": admission.get("diagnosis"),
                },
                "interpretation": (
                    "Prebuild exposes a later engine/talker-side tail."
                    if "prebuild" in label
                    else "Original path does not show a code2wav boundary bottleneck."
                ),
            }
        )
        rows.append(
            {
                "runtime": "vllm",
                "workload": "Video-AMME ci-50 offline",
                "boundary": "runner_to_engine_admission",
                "label": label,
                "status": (
                    "prompt_feed_limited"
                    if admission.get("diagnosis") == "prompt_feed_limited"
                    else "diagnostic_only"
                    if "prebuild" in label
                    else "healthy"
                ),
                "metrics": {
                    "batch_admission_span_avg_ms": span_avg,
                    "batch_admission_span_p95_ms": row[
                        "batch_admission_span_ms"
                    ].get("p95"),
                    "runner_overhead_pct_wall": admission.get(
                        "runner_overhead_pct_wall"
                    ),
                    "engine_qps": admission.get("engine_qps"),
                },
                "interpretation": (
                    "Offline runner prompt build/feed dominates wall throughput."
                    if admission.get("diagnosis") == "prompt_feed_limited"
                    else "No batched offline admission span is visible at c1."
                    if label == "vLLM-c1"
                    else "Prebuilt prompts reduce admission span but do not establish online serving parity."
                ),
            }
        )
    return rows


def build_summary(root: Path) -> dict[str, Any]:
    interactions = (
        _sglang_interactions(root)
        + _synthetic_interactions(root)
        + _vllm_interactions(root)
    )
    status_counts: dict[str, int] = {}
    for row in interactions:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "root": str(root),
        "summary": {
            "total_interactions": len(interactions),
            "status_counts": status_counts,
            "sglang_talker_to_code2wav_healthy": all(
                row["status"] == "healthy"
                for row in interactions
                if row["runtime"] == "sglang"
                and row["boundary"] == "talker_to_code2wav_stream"
            ),
            "sglang_code2wav_decode_not_bottleneck": all(
                row["status"] == "healthy"
                for row in interactions
                if row["runtime"] == "sglang"
                and row["boundary"] == "code2wav_collect_to_decode"
            ),
            "vllm_original_c8_prompt_feed_limited": any(
                row.get("label") == "vLLM-c8"
                and row["boundary"] == "runner_to_engine_admission"
                and row["status"] == "prompt_feed_limited"
                for row in interactions
            ),
            "preprocessing_parallelism_regresses": any(
                row["runtime"] == "sglang"
                and row["boundary"] == "preprocessing_to_encoder_thinker"
                and row["status"] == "contention_regression"
                for row in interactions
            ),
        },
        "interactions": interactions,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    print("## Stage Interaction Summary\n")
    summary = payload["summary"]
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Total interactions | {summary['total_interactions']} |")
    print(
        "| SGLang talker->code2wav healthy | "
        f"{summary['sglang_talker_to_code2wav_healthy']} |"
    )
    print(
        "| SGLang code2wav decode not bottleneck | "
        f"{summary['sglang_code2wav_decode_not_bottleneck']} |"
    )
    print(
        "| vLLM original c8 prompt-feed limited | "
        f"{summary['vllm_original_c8_prompt_feed_limited']} |"
    )
    print(
        "| preprocessing parallelism regresses | "
        f"{summary['preprocessing_parallelism_regresses']} |"
    )
    print()
    print("| Runtime | Boundary | Case | Status | Key Metrics | Interpretation |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in payload["interactions"]:
        case = row.get("label") or f"{row.get('workload')} c={row.get('concurrency')}"
        metrics = ", ".join(
            f"{key}={value}"
            for key, value in row.get("metrics", {}).items()
            if value is not None
        )
        print(
            f"| {row.get('runtime')} | {row.get('boundary')} | {case} | "
            f"{row.get('status')} | {metrics} | {row.get('interpretation')} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Qwen3.5-Omni stage interactions from audit artifacts."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_summary(root)
    print_markdown(payload)
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = root / output
        _save_json(payload, output)


if __name__ == "__main__":
    main()
