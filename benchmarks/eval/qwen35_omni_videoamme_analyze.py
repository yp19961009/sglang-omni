# SPDX-License-Identifier: Apache-2.0
"""Analyze Qwen3.5-Omni Video-AMME stress outputs.

This script consumes the artifacts produced by
``qwen35_omni_videoamme_stress.sh`` and writes a compact report that answers:

* whether the run is valid,
* which concurrency has the best throughput,
* where saturation begins,
* which stage or stage boundary is likely responsible for latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"(?:^|[_/\-])c(\d+)(?:[_/\-]|$)")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _fmt(value: Any, digits: int = 3) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return ""
    return f"{parsed * 100:.{digits}f}%"


def _parse_c(path: Path | str) -> int | None:
    text = str(path)
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


@dataclass
class RunFiles:
    result_jsons: dict[int, Path]
    profile_jsons: dict[int, Path]


def _discover_files(run_root: Path) -> RunFiles:
    result_jsons: dict[int, Path] = {}
    for path in sorted(run_root.glob("benchmark_audio_*_c*/videoamme_results.json")):
        c = _parse_c(path)
        if c is not None:
            result_jsons[c] = path

    profile_jsons: dict[int, Path] = {}
    for path in sorted(run_root.glob("request_profile_c*_profile*.json")):
        c = _parse_c(path)
        if c is not None:
            profile_jsons[c] = path

    return RunFiles(result_jsons=result_jsons, profile_jsons=profile_jsons)


def _load_performance(run_root: Path, files: RunFiles) -> dict[int, dict[str, Any]]:
    rows = _read_tsv(run_root / "performance_table.tsv")
    perf: dict[int, dict[str, Any]] = {}
    for row in rows:
        c = _to_int(row.get("C"))
        if c is not None:
            perf[c] = dict(row)

    for c, path in files.result_jsons.items():
        if c in perf:
            perf[c].setdefault("result_json", str(path))
            continue
        payload = _read_json(path)
        speed = payload.get("speed") or {}
        summary = payload.get("summary") or {}
        perf[c] = {
            "C": c,
            "completed": speed.get("completed_requests"),
            "failed": speed.get("failed_requests"),
            "accuracy": summary.get("accuracy"),
            "qps": speed.get("throughput_qps"),
            "lat_mean_s": speed.get("latency_mean_s"),
            "lat_median_s": speed.get("latency_median_s"),
            "lat_p95_s": speed.get("latency_p95_s"),
            "lat_p99_s": speed.get("latency_p99_s"),
            "rtf_mean": speed.get("rtf_mean"),
            "output_tok_per_req_s": speed.get("output_tok_per_req_s"),
            "output_throughput": speed.get("output_throughput"),
            "prompt_tokens_total": speed.get("prompt_tokens_total"),
            "output_tokens_total": speed.get("output_tokens_total"),
            "result_json": str(path),
        }
    return perf


def _load_stage_rows(run_root: Path, files: RunFiles) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(row) for row in _read_tsv(run_root / "stage_breakdown.tsv")]
    loaded_cs = {_to_int(row.get("C")) for row in rows}

    for c, path in files.profile_jsons.items():
        if c in loaded_cs:
            continue
        payload = _read_json(path)
        for row in payload.get("stage_breakdown") or []:
            item = dict(row)
            item["C"] = c
            item["profile_json"] = str(path)
            rows.append(item)
    return rows


def _load_hop_rows(run_root: Path, files: RunFiles) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(row) for row in _read_tsv(run_root / "hop_breakdown.tsv")]
    loaded_cs = {_to_int(row.get("C")) for row in rows}

    for c, path in files.profile_jsons.items():
        if c in loaded_cs:
            continue
        payload = _read_json(path)
        for row in payload.get("hop_breakdown") or []:
            item = dict(row)
            item["C"] = c
            item["profile_json"] = str(path)
            rows.append(item)
    return rows


def _stage_metric(
    rows: list[dict[str, Any]],
    c: int,
    stage: str,
    interval: str,
    key: str,
) -> float | None:
    for row in rows:
        if (
            _to_int(row.get("C")) == c
            and row.get("stage") == stage
            and row.get("interval") == interval
        ):
            return _to_float(row.get(key))
    return None


def _top_stage(
    rows: list[dict[str, Any]],
    c: int,
    key: str,
    *,
    interval: str | None = "stage_input_received->stage_complete",
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if _to_int(row.get("C")) != c:
            continue
        if interval is not None and row.get("interval") != interval:
            continue
        if _to_float(row.get(key)) is None:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row.get(key)) or -1.0)


def _top_hop(rows: list[dict[str, Any]], c: int, key: str = "p95_ms") -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if _to_int(row.get("C")) == c and _to_float(row.get(key)) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _to_float(row.get(key)) or -1.0)


def _slow_samples(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    samples = payload.get("per_sample") or []
    if not isinstance(samples, list):
        return []

    def latency(sample: dict[str, Any]) -> float:
        return _to_float(sample.get("latency_s")) or -1.0

    result = []
    for sample in sorted(samples, key=latency, reverse=True)[:limit]:
        if not isinstance(sample, dict):
            continue
        result.append(
            {
                "sample_id": sample.get("sample_id")
                or sample.get("id")
                or sample.get("question_id")
                or sample.get("index")
                or "",
                "latency_s": sample.get("latency_s"),
                "success": sample.get("success"),
                "error": sample.get("error") or "",
            }
        )
    return result


def _slow_timelines(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    timelines = payload.get("timelines") or {}
    if not isinstance(timelines, dict):
        return []
    rows = []
    for request_id, events in timelines.items():
        if not isinstance(events, list) or not events:
            continue
        max_t = max((_to_float(event.get("t_rel_ms")) or 0.0) for event in events if isinstance(event, dict))
        stages = []
        for event in events:
            if not isinstance(event, dict):
                continue
            stage = str(event.get("stage") or "")
            if stage and stage not in stages:
                stages.append(stage)
        rows.append({"request_id": request_id, "total_ms": max_t, "stages": " -> ".join(stages)})
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:limit]


def _build_analysis(
    run_root: Path,
    perf: dict[int, dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    hop_rows: list[dict[str, Any]],
    files: RunFiles,
) -> dict[str, Any]:
    concurrencies = sorted(set(perf) | {_to_int(row.get("C")) for row in stage_rows if _to_int(row.get("C")) is not None})
    concurrencies = [c for c in concurrencies if c is not None]

    warnings: list[str] = []
    if not perf:
        warnings.append("No performance rows were found.")
    missing_profiles = [c for c in sorted(perf) if c not in files.profile_jsons]
    if missing_profiles:
        warnings.append(f"Missing request profile JSON for C={missing_profiles}.")

    completed_values = [_to_int(row.get("completed")) for row in perf.values()]
    min_completed = min((value for value in completed_values if value is not None), default=None)
    if min_completed is not None and min_completed < 20:
        warnings.append(
            f"Small sample run detected: min completed={min_completed}. Treat as smoke, not final performance evidence."
        )

    failed_total = sum(_to_int(row.get("failed")) or 0 for row in perf.values())
    if failed_total:
        warnings.append(f"Failures detected across the sweep: failed_total={failed_total}.")

    qps_rows = [
        (c, _to_float(row.get("qps")), _to_float(row.get("lat_p95_s")))
        for c, row in perf.items()
        if _to_float(row.get("qps")) is not None
    ]
    best_c = None
    best_qps = None
    best_lat_p95 = None
    if qps_rows:
        best_c, best_qps, best_lat_p95 = max(qps_rows, key=lambda item: item[1] or -1.0)

    saturation = None
    if best_c is not None and best_qps is not None:
        for c, qps, lat_p95 in sorted(qps_rows):
            if c <= best_c or qps is None:
                continue
            lat_regressed = (
                best_lat_p95 is not None
                and lat_p95 is not None
                and lat_p95 >= best_lat_p95 * 1.10
            )
            qps_regressed = qps <= best_qps * 0.98
            if qps_regressed and lat_regressed:
                saturation = {"C": c, "qps": qps, "lat_p95_s": lat_p95}
                break

    derived_rows = []
    for c in concurrencies:
        perf_row = perf.get(c, {})
        pre_life = _stage_metric(
            stage_rows, c, "preprocessing", "stage_input_received->stage_complete", "avg_ms"
        )
        pre_life_p95 = _stage_metric(
            stage_rows, c, "preprocessing", "stage_input_received->stage_complete", "p95_ms"
        )
        pre_compute = _stage_metric(stage_rows, c, "preprocessing", "preprocess_start->preprocess_end", "avg_ms")
        pre_hf = _stage_metric(
            stage_rows,
            c,
            "preprocessing",
            "preprocess_hf_processor_start->preprocess_hf_processor_end",
            "avg_ms",
        )
        pre_queue = None
        pre_queue_pct = None
        if pre_life is not None and pre_compute is not None:
            pre_queue = max(0.0, pre_life - pre_compute)
            pre_queue_pct = pre_queue / pre_life if pre_life > 0 else None

        talker_life_p95 = _stage_metric(stage_rows, c, "talker_ar", "stage_input_received->stage_complete", "p95_ms")
        thinker_first = _stage_metric(
            stage_rows, c, "thinker", "scheduler_prefill_start->stage_first_stream_chunk_sent", "avg_ms"
        )
        talker_first = _stage_metric(
            stage_rows, c, "talker_ar", "scheduler_prefill_start->stage_first_stream_chunk_sent", "avg_ms"
        )
        c2w_collect = _stage_metric(
            stage_rows,
            c,
            "code2wav",
            "code2wav_window_collect_start->code2wav_window_collect_end",
            "avg_ms",
        )
        c2w_decode = _stage_metric(
            stage_rows,
            c,
            "code2wav",
            "code2wav_decode_start->code2wav_decode_end",
            "avg_ms",
        )
        c2w_decode_pct = None
        if c2w_collect is not None and c2w_collect > 0 and c2w_decode is not None:
            c2w_decode_pct = c2w_decode / c2w_collect

        top_life = _top_stage(stage_rows, c, "avg_ms")
        top_p95 = _top_stage(stage_rows, c, "p95_ms")
        top_hop = _top_hop(hop_rows, c, "p95_ms")
        top_hop_p95 = _to_float(top_hop.get("p95_ms")) if top_hop else None

        diagnosis = []
        failed = _to_int(perf_row.get("failed")) or 0
        if failed:
            diagnosis.append("request failures")
        if pre_queue is not None and pre_queue_pct is not None:
            if pre_queue >= 500 and pre_queue_pct >= 0.30:
                diagnosis.append("admission/preprocessing queue pressure")
            elif pre_queue >= 200:
                diagnosis.append("moderate preprocessing queue")
        if talker_life_p95 is not None and talker_life_p95 >= 1000:
            diagnosis.append("talker AR tail")
        if c2w_collect is not None and c2w_decode is not None:
            if c2w_decode < 50 and c2w_collect >= c2w_decode * 2:
                diagnosis.append("code2wav waits for talker cadence")
            elif c2w_decode >= 50 or (c2w_decode_pct is not None and c2w_decode_pct >= 0.70):
                diagnosis.append("possible code2wav compute pressure")
        if top_hop_p95 is not None and top_hop_p95 >= 100:
            diagnosis.append("stage-hop/backpressure tail")
        if not diagnosis:
            diagnosis.append("no dominant bottleneck in aggregate")

        derived_rows.append(
            {
                "C": c,
                "completed": _to_int(perf_row.get("completed")),
                "failed": failed,
                "qps": _to_float(perf_row.get("qps")),
                "lat_p95_s": _to_float(perf_row.get("lat_p95_s")),
                "rtf_mean": _to_float(perf_row.get("rtf_mean")),
                "pre_life_avg_ms": pre_life,
                "pre_life_p95_ms": pre_life_p95,
                "pre_compute_avg_ms": pre_compute,
                "pre_hf_processor_avg_ms": pre_hf,
                "pre_queue_est_avg_ms": pre_queue,
                "pre_queue_est_pct": pre_queue_pct,
                "thinker_first_avg_ms": thinker_first,
                "talker_first_avg_ms": talker_first,
                "talker_life_p95_ms": talker_life_p95,
                "code2wav_collect_avg_ms": c2w_collect,
                "code2wav_decode_avg_ms": c2w_decode,
                "code2wav_decode_pct_of_collect": c2w_decode_pct,
                "top_lifecycle_stage": None if not top_life else top_life.get("stage"),
                "top_lifecycle_avg_ms": None if not top_life else _to_float(top_life.get("avg_ms")),
                "top_p95_stage": None if not top_p95 else top_p95.get("stage"),
                "top_p95_ms": None if not top_p95 else _to_float(top_p95.get("p95_ms")),
                "top_hop": None if not top_hop else f"{top_hop.get('src')}->{top_hop.get('dst')} ({top_hop.get('kind')})",
                "top_hop_p95_ms": top_hop_p95,
                "diagnosis": "; ".join(diagnosis),
            }
        )

    slow_samples = {
        str(c): _slow_samples(path)
        for c, path in sorted(files.result_jsons.items())
    }
    slow_timelines = {
        str(c): _slow_timelines(path)
        for c, path in sorted(files.profile_jsons.items())
    }

    return {
        "run_root": str(run_root),
        "warnings": warnings,
        "best_concurrency": {
            "C": best_c,
            "qps": best_qps,
            "lat_p95_s": best_lat_p95,
        },
        "saturation_after_peak": saturation,
        "derived_rows": derived_rows,
        "slow_samples": slow_samples,
        "slow_timelines": slow_timelines,
        "files": {
            "result_jsons": {str(c): str(path) for c, path in sorted(files.result_jsons.items())},
            "profile_jsons": {str(c): str(path) for c, path in sorted(files.profile_jsons.items())},
        },
    }


def _render_markdown(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Qwen3.5-Omni Video-AMME Stress Analysis\n")
    lines.append(f"Run root: `{analysis['run_root']}`\n")

    warnings = analysis.get("warnings") or []
    if warnings:
        lines.append("## Warnings\n")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    best = analysis.get("best_concurrency") or {}
    saturation = analysis.get("saturation_after_peak")
    lines.append("## Executive Summary\n")
    if best.get("C") is not None:
        lines.append(
            f"- Best observed throughput: C={best.get('C')} with qps={_fmt(best.get('qps'))}, "
            f"lat_p95_s={_fmt(best.get('lat_p95_s'))}."
        )
    else:
        lines.append("- Best observed throughput: unavailable.")
    if saturation:
        lines.append(
            f"- Saturation evidence after peak: C={saturation.get('C')} "
            f"qps={_fmt(saturation.get('qps'))}, lat_p95_s={_fmt(saturation.get('lat_p95_s'))}."
        )
    else:
        lines.append("- Saturation evidence after peak: not detected in this sweep.")
    lines.append("")

    derived = analysis.get("derived_rows") or []
    lines.append("## Performance And Derived Stage Budget\n")
    lines.append(
        _markdown_table(
            [
                "C",
                "completed",
                "failed",
                "qps",
                "lat_p95_s",
                "pre_queue_est_ms",
                "pre_queue_pct",
                "talker_p95_ms",
                "c2w_collect_ms",
                "c2w_decode_ms",
                "decode/collect",
                "top_hop_p95_ms",
                "diagnosis",
            ],
            [
                [
                    row.get("C"),
                    row.get("completed"),
                    row.get("failed"),
                    _fmt(row.get("qps")),
                    _fmt(row.get("lat_p95_s")),
                    _fmt(row.get("pre_queue_est_avg_ms")),
                    _fmt_pct(row.get("pre_queue_est_pct")),
                    _fmt(row.get("talker_life_p95_ms")),
                    _fmt(row.get("code2wav_collect_avg_ms")),
                    _fmt(row.get("code2wav_decode_avg_ms")),
                    _fmt_pct(row.get("code2wav_decode_pct_of_collect")),
                    _fmt(row.get("top_hop_p95_ms")),
                    row.get("diagnosis"),
                ]
                for row in derived
            ],
        )
    )
    lines.append("")

    lines.append("## Top Lifecycle Stages\n")
    lines.append(
        _markdown_table(
            ["C", "top_avg_stage", "top_avg_ms", "top_p95_stage", "top_p95_ms"],
            [
                [
                    row.get("C"),
                    row.get("top_lifecycle_stage"),
                    _fmt(row.get("top_lifecycle_avg_ms")),
                    row.get("top_p95_stage"),
                    _fmt(row.get("top_p95_ms")),
                ]
                for row in derived
            ],
        )
    )
    lines.append("")

    lines.append("## Slowest Samples\n")
    sample_rows = []
    for c, samples in (analysis.get("slow_samples") or {}).items():
        for sample in samples[:3]:
            sample_rows.append(
                [
                    c,
                    sample.get("sample_id"),
                    _fmt(sample.get("latency_s")),
                    sample.get("success"),
                    sample.get("error"),
                ]
            )
    lines.append(_markdown_table(["C", "sample_id", "latency_s", "success", "error"], sample_rows))
    lines.append("")

    lines.append("## Slowest Request Timelines\n")
    timeline_rows = []
    for c, timelines in (analysis.get("slow_timelines") or {}).items():
        for timeline in timelines[:3]:
            timeline_rows.append(
                [
                    c,
                    timeline.get("request_id"),
                    _fmt(timeline.get("total_ms")),
                    timeline.get("stages"),
                ]
            )
    lines.append(_markdown_table(["C", "request_id", "total_ms", "stages"], timeline_rows))
    lines.append("")

    lines.append("## Reading Guide\n")
    lines.append("- Use qps and lat_p95_s to choose the serving point.")
    lines.append("- pre_queue_est_ms = preprocessing lifecycle avg minus preprocessing compute avg.")
    lines.append("- If code2wav collect is much larger than decode, code2wav is mostly waiting for talker chunks.")
    lines.append("- If top_hop_p95_ms is high, check stage-to-stage backpressure or IPC.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    parser.add_argument("--print", action="store_true", help="Print the markdown report to stdout.")
    args = parser.parse_args()

    run_root = args.run_root
    files = _discover_files(run_root)
    perf = _load_performance(run_root, files)
    stage_rows = _load_stage_rows(run_root, files)
    hop_rows = _load_hop_rows(run_root, files)
    analysis = _build_analysis(run_root, perf, stage_rows, hop_rows, files)

    json_output = args.json_output or run_root / "analysis_summary.json"
    md_output = args.md_output or run_root / "analysis_report.md"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = _render_markdown(analysis)
    md_output.write_text(markdown, encoding="utf-8")

    if args.print:
        print(markdown)
    else:
        best = analysis.get("best_concurrency") or {}
        print(f"analysis_json={json_output}")
        print(f"analysis_md={md_output}")
        print(f"best_C={best.get('C')} qps={_fmt(best.get('qps'))} lat_p95_s={_fmt(best.get('lat_p95_s'))}")
        if analysis.get("warnings"):
            print("warnings=" + " | ".join(analysis["warnings"]))


if __name__ == "__main__":
    main()
