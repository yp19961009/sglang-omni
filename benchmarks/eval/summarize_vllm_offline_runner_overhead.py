# SPDX-License-Identifier: Apache-2.0
"""Summarize vLLM offline runner wall-time overhead from saved result JSON.

The Qwen3.5-Omni vLLM offline runner used in this workspace can either build
multimodal prompts synchronously inside the timed loop or prebuild prompts before
engine admission. Per-request latency starts after the prompt object is ready.
This helper keeps both clocks explicit:

* runner wall covers local prompt construction plus engine execution;
* engine wall covers the timed request loop used by the runner's speed table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def summarize(path: Path, label: str | None = None) -> dict[str, Any]:
    payload = _load_json(path)
    config = payload.get("config", {})
    records = payload.get("per_sample", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} does not contain non-empty per_sample records")

    concurrency = int(config.get("max_concurrency") or 1)
    timed_wall_s = float(config.get("wall_clock_s") or 0)
    runner_wall_s = float(config.get("runner_wall_clock_s") or timed_wall_s)
    engine_wall_s = float(config.get("engine_wall_clock_s") or timed_wall_s)
    prompt_build_wall_s = float(config.get("prompt_build_wall_s") or 0)
    if runner_wall_s <= 0:
        raise ValueError(
            f"{path} config.runner_wall_clock_s/wall_clock_s is missing or non-positive"
        )
    if engine_wall_s <= 0:
        raise ValueError(
            f"{path} config.engine_wall_clock_s/wall_clock_s is missing or non-positive"
        )

    batches = [records[i : i + concurrency] for i in range(0, len(records), concurrency)]
    batch_max_latency_s = [
        max(float(record.get("latency_s") or 0) for record in batch)
        for batch in batches
    ]
    batch_max_sum_s = sum(batch_max_latency_s)
    runner_overhead_s = runner_wall_s - batch_max_sum_s
    engine_overhead_s = engine_wall_s - batch_max_sum_s
    successful = [record for record in records if record.get("is_success")]
    total_audio_s = sum(float(record.get("audio_duration_s") or 0) for record in successful)

    return {
        "label": label or path.parent.name,
        "path": str(path),
        "concurrency": concurrency,
        "requests": len(records),
        "successful": len(successful),
        # Backward-compatible aliases use the true end-to-end runner wall.
        "wall_time_s": runner_wall_s,
        "runner_wall_time_s": runner_wall_s,
        "engine_wall_time_s": engine_wall_s,
        "timed_wall_time_s": timed_wall_s,
        "prompt_build_wall_s": prompt_build_wall_s,
        "prebuild_prompts": bool(config.get("prebuild_prompts")),
        "prebuild_workers": config.get("prebuild_workers"),
        "batch_max_sum_s": batch_max_sum_s,
        "runner_overhead_s": runner_overhead_s,
        "runner_overhead_pct_wall": runner_overhead_s / runner_wall_s * 100,
        "engine_overhead_s": engine_overhead_s,
        "engine_overhead_pct_wall": engine_overhead_s / engine_wall_s * 100,
        "wall_qps": len(records) / runner_wall_s,
        "runner_qps": len(records) / runner_wall_s,
        "engine_qps": len(records) / engine_wall_s,
        "batch_max_qps": len(records) / batch_max_sum_s if batch_max_sum_s > 0 else 0,
        "wall_audio_throughput_s_per_s": total_audio_s / runner_wall_s,
        "runner_audio_throughput_s_per_s": total_audio_s / runner_wall_s,
        "engine_audio_throughput_s_per_s": total_audio_s / engine_wall_s,
        "batch_max_audio_throughput_s_per_s": (
            total_audio_s / batch_max_sum_s if batch_max_sum_s > 0 else 0
        ),
        "batch_max_latency_s": batch_max_latency_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate vLLM offline-runner overhead from result JSON files."
    )
    parser.add_argument("results", nargs="+", help="vLLM videoamme_results.json files")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels, one per result JSON.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in {0, len(args.results)}:
        parser.error("--labels must be omitted or have the same length as results")

    labels = args.labels or [None] * len(args.results)
    rows = [summarize(Path(path), label) for path, label in zip(args.results, labels)]

    print(
        "| Label | c | Requests | Runner Wall s | Engine Wall s | Prompt Build s | "
        "Batch-max Sum s | Runner Overhead | Engine Overhead | Runner QPS | "
        "Engine QPS | Batch-max QPS | Runner Audio Thr | Engine Audio Thr |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        print(
            "| {label} | {concurrency} | {requests} | {runner_wall} | "
            "{engine_wall} | {prompt_wall} | {batch_sum} | "
            "{runner_overhead}s ({runner_overhead_pct}%) | "
            "{engine_overhead}s ({engine_overhead_pct}%) | "
            "{runner_qps} | {engine_qps} | {batch_qps} | "
            "{runner_audio} | {engine_audio} |".format(
                label=row["label"],
                concurrency=row["concurrency"],
                requests=row["requests"],
                runner_wall=_fmt(row["runner_wall_time_s"], 1),
                engine_wall=_fmt(row["engine_wall_time_s"], 1),
                prompt_wall=_fmt(row["prompt_build_wall_s"], 1),
                batch_sum=_fmt(row["batch_max_sum_s"], 1),
                runner_overhead=_fmt(row["runner_overhead_s"], 1),
                runner_overhead_pct=_fmt(row["runner_overhead_pct_wall"], 1),
                engine_overhead=_fmt(row["engine_overhead_s"], 1),
                engine_overhead_pct=_fmt(row["engine_overhead_pct_wall"], 1),
                runner_qps=_fmt(row["runner_qps"], 4),
                engine_qps=_fmt(row["engine_qps"], 4),
                batch_qps=_fmt(row["batch_max_qps"], 4),
                runner_audio=_fmt(row["runner_audio_throughput_s_per_s"], 4),
                engine_audio=_fmt(row["engine_audio_throughput_s_per_s"], 4),
            )
        )


if __name__ == "__main__":
    main()
