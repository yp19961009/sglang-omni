# SPDX-License-Identifier: Apache-2.0
"""Build slide-ready CSV/SVG figures for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def _percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    return _fmt(_as_float(value) * 100.0, digits)


def _csv_record(path: Path, label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "csv",
        "label": label,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "rows": len(rows),
    }


def _file_record(path: Path, *, kind: str, label: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }


def _write_csv(
    path: Path,
    *,
    label: str,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return _csv_record(path, label, rows)


def _nice_ymax(values: list[float]) -> float:
    vmax = max([0.0, *values])
    if vmax <= 0.0:
        return 1.0
    return vmax * 1.15


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #4b5563; }",
        ".axis { stroke: #374151; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".legend { font-size: 13px; fill: #111827; }",
        ".value { font-size: 11px; fill: #111827; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
) -> dict[str, Any]:
    width, height = 1040, 620
    left, right, top, bottom = 92, 44, 82, 92
    inner_w = width - left - right
    inner_h = height - top - bottom
    values = [value for _, data in series for value in data]
    y_max = _nice_ymax(values)
    group_w = inner_w / max(1, len(categories))
    bar_w = min(52.0, group_w / (len(series) + 1.4))
    lines = _svg_header(width, height)
    lines.extend(
        [
            f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>',
            f'<text class="subtitle" x="{left}" y="60">{html.escape(subtitle)}</text>',
            f'<text class="subtitle" x="{left}" y="{height - 18}">{html.escape(y_label)}</text>',
        ]
    )
    for i in range(6):
        y_value = y_max * i / 5.0
        y = top + inner_h - (y_value / y_max) * inner_h
        lines.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{_fmt(y_value, 2)}</text>')
    lines.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + inner_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + inner_h}" y2="{top + inner_h}"/>')

    legend_x = left
    for idx, (name, _) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        x = legend_x + idx * 160
        lines.append(f'<rect x="{x}" y="{height - 54}" width="16" height="16" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{x + 22}" y="{height - 41}">{html.escape(name)}</text>')

    for cat_idx, category in enumerate(categories):
        base_x = left + cat_idx * group_w + group_w / 2.0
        lines.append(
            f'<text class="tick" x="{base_x:.1f}" y="{top + inner_h + 28}" text-anchor="middle">{html.escape(category)}</text>'
        )
        for series_idx, (_, data) in enumerate(series):
            value = data[cat_idx]
            color = PALETTE[series_idx % len(PALETTE)]
            x = base_x - (bar_w * len(series)) / 2.0 + series_idx * bar_w
            h = (value / y_max) * inner_h if y_max else 0.0
            y = top + inner_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{h:.1f}" fill="{color}"/>')
            lines.append(f'<text class="value" x="{x + (bar_w - 4) / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{_fmt(value, 2)}</text>')
    lines.extend(_svg_footer())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _file_record(path, kind="svg", label=title)


def _write_line_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
) -> dict[str, Any]:
    width, height = 1040, 620
    left, right, top, bottom = 92, 44, 82, 94
    inner_w = width - left - right
    inner_h = height - top - bottom
    values = [value for _, data in series for value in data]
    y_max = _nice_ymax(values)
    step_x = inner_w / max(1, len(x_labels) - 1)
    lines = _svg_header(width, height)
    lines.extend(
        [
            f'<text class="title" x="{left}" y="38">{html.escape(title)}</text>',
            f'<text class="subtitle" x="{left}" y="60">{html.escape(subtitle)}</text>',
            f'<text class="subtitle" x="{left}" y="{height - 18}">{html.escape(y_label)}</text>',
        ]
    )
    for i in range(6):
        y_value = y_max * i / 5.0
        y = top + inner_h - (y_value / y_max) * inner_h
        lines.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{_fmt(y_value, 2)}</text>')
    lines.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + inner_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + inner_h}" y2="{top + inner_h}"/>')
    for idx, label in enumerate(x_labels):
        x = left + idx * step_x
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + inner_h + 28}" text-anchor="middle">{html.escape(label)}</text>')

    for series_idx, (name, data) in enumerate(series):
        color = PALETTE[series_idx % len(PALETTE)]
        points: list[tuple[float, float]] = []
        for idx, value in enumerate(data):
            x = left + idx * step_x
            y = top + inner_h - (value / y_max) * inner_h if y_max else top + inner_h
            points.append((x, y))
        if points:
            path_data = " ".join(
                ("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}"
                for idx, (x, y) in enumerate(points)
            )
            lines.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="3"/>')
        for value, (x, y) in zip(data, points):
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            lines.append(f'<text class="value" x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle">{_fmt(value, 2)}</text>')
        legend_x = left + series_idx * 210
        lines.append(f'<rect x="{legend_x}" y="{height - 58}" width="16" height="16" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{legend_x + 22}" y="{height - 45}">{html.escape(name)}</text>')
    lines.extend(_svg_footer())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _file_record(path, kind="svg", label=title)


def _table_rows(tables: dict[str, Any], name: str) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get(name, [])
    return rows if isinstance(rows, list) else []


def build_chart_pack(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = root / AUDIT_DIR
    tables = _load_json(audit_dir / "tables_summary.json")
    scorecard = _load_json(audit_dir / "headline_scorecard.json")
    interactions = _load_json(audit_dir / "stage_interaction_summary.json")
    admission = _load_json(audit_dir / "vllm_admission_diagnosis.json")
    stage_budget = _load_json(audit_dir / "stage_latency_budget.json")

    stress_rows = _table_rows(tables, "sglang_stress")
    stage_rows = _table_rows(tables, "sglang_stage_breakdown")
    synthetic_rows = _table_rows(tables, "synthetic_speech")
    synthetic_stage_rows = _table_rows(tables, "synthetic_stage_breakdown")
    preproc_rows = _table_rows(tables, "preprocessing_concurrency")
    admission_rows = admission.get("rows", [])
    sglang_budget_rows = stage_budget.get("sglang_videoamme_budget", [])

    stage_by_c = {int(row["concurrency"]): row for row in stage_rows}
    synthetic_stage_by_key = {
        (str(row.get("scenario")), int(row.get("concurrency"))): row
        for row in synthetic_stage_rows
    }

    generated: list[dict[str, Any]] = []
    strict = scorecard["strict_c4_comparison"]
    strict_rows = []
    for runtime in ["vllm", "sglang"]:
        row = strict[runtime]
        strict_rows.append(
            {
                "runtime": runtime,
                "scope": strict["scope"],
                "n": _fmt(row.get("n"), 0),
                "accuracy_pct": _percent(row.get("accuracy"), 2),
                "latency_mean_s": _fmt(row.get("latency_mean_s")),
                "latency_p95_s": _fmt(row.get("latency_p95_s")),
                "rtf_mean": _fmt(row.get("rtf_mean")),
                "rtf_p95": _fmt(row.get("rtf_p95")),
                "wer_pct": _percent(row.get("wer_corpus"), 2),
                "artifact": row.get("artifact", ""),
            }
        )
    generated.append(
        _write_csv(
            output_dir / "strict_c4_runtime_comparison.csv",
            label="Strict c=4 runtime comparison",
            columns=[
                "runtime",
                "scope",
                "n",
                "accuracy_pct",
                "latency_mean_s",
                "latency_p95_s",
                "rtf_mean",
                "rtf_p95",
                "wer_pct",
                "artifact",
            ],
            rows=strict_rows,
        )
    )

    pressure_rows: list[dict[str, Any]] = []
    for row in stress_rows:
        concurrency = int(row["concurrency"])
        stage = stage_by_c.get(concurrency, {})
        pressure_rows.append(
            {
                "concurrency": concurrency,
                "n": row.get("n"),
                "accuracy_pct": _percent(row.get("accuracy"), 2),
                "wer_pct": _percent(row.get("wer_corpus"), 2),
                "throughput_qps": _fmt(row.get("throughput_qps")),
                "audio_throughput_s_per_s": _fmt(row.get("audio_throughput_s_per_s")),
                "latency_mean_s": _fmt(row.get("latency_mean_s")),
                "latency_p95_s": _fmt(row.get("latency_p95_s")),
                "rtf_mean": _fmt(row.get("rtf_mean")),
                "rtf_p95": _fmt(row.get("rtf_p95")),
                "preproc_stage_avg_ms": _fmt(stage.get("preproc_stage_avg_ms"), 3),
                "talker_p95_ms": _fmt(stage.get("talker_p95_ms"), 3),
                "talker_to_code2wav_hop_p95_ms": _fmt(
                    stage.get("talker_to_code2wav_hop_p95_ms"), 3
                ),
                "code2wav_decode_p95_ms": _fmt(stage.get("code2wav_decode_p95_ms"), 3),
                "top_stage": stage.get("top_stage", ""),
                "result_json": row.get("result_json", ""),
                "profile_json": stage.get("profile_json", ""),
            }
        )
    generated.append(
        _write_csv(
            output_dir / "sglang_pressure_sweep.csv",
            label="SGLang pressure sweep",
            columns=[
                "concurrency",
                "n",
                "accuracy_pct",
                "wer_pct",
                "throughput_qps",
                "audio_throughput_s_per_s",
                "latency_mean_s",
                "latency_p95_s",
                "rtf_mean",
                "rtf_p95",
                "preproc_stage_avg_ms",
                "talker_p95_ms",
                "talker_to_code2wav_hop_p95_ms",
                "code2wav_decode_p95_ms",
                "top_stage",
                "result_json",
                "profile_json",
            ],
            rows=pressure_rows,
        )
    )

    synthetic_csv_rows: list[dict[str, Any]] = []
    for row in synthetic_rows:
        key = (str(row.get("scenario")), int(row.get("concurrency")))
        stage = synthetic_stage_by_key.get(key, {})
        synthetic_csv_rows.append(
            {
                "scenario": row.get("scenario"),
                "concurrency": row.get("concurrency"),
                "n": row.get("n"),
                "target_chars": _fmt(row.get("target_chars"), 0),
                "target_words": _fmt(row.get("target_words"), 0),
                "audio_duration_mean_s": _fmt(row.get("audio_duration_mean_s")),
                "throughput_qps": _fmt(row.get("throughput_qps")),
                "audio_throughput_s_per_s": _fmt(row.get("audio_throughput_s_per_s")),
                "latency_mean_s": _fmt(row.get("latency_mean_s")),
                "latency_p95_s": _fmt(row.get("latency_p95_s")),
                "rtf_mean": _fmt(row.get("rtf_mean")),
                "rtf_p95": _fmt(row.get("rtf_p95")),
                "talker_p95_ms": _fmt(stage.get("talker_p95_ms"), 3),
                "talker_to_code2wav_hop_p95_ms": _fmt(
                    stage.get("talker_to_code2wav_hop_p95_ms"), 3
                ),
                "code2wav_decode_p95_ms": _fmt(stage.get("code2wav_decode_p95_ms"), 3),
                "result_json": row.get("result_json", ""),
                "profile_json": stage.get("profile_json", ""),
            }
        )
    generated.append(
        _write_csv(
            output_dir / "synthetic_short_long_speech.csv",
            label="Synthetic short/long text-to-speech",
            columns=[
                "scenario",
                "concurrency",
                "n",
                "target_chars",
                "target_words",
                "audio_duration_mean_s",
                "throughput_qps",
                "audio_throughput_s_per_s",
                "latency_mean_s",
                "latency_p95_s",
                "rtf_mean",
                "rtf_p95",
                "talker_p95_ms",
                "talker_to_code2wav_hop_p95_ms",
                "code2wav_decode_p95_ms",
                "result_json",
                "profile_json",
            ],
            rows=synthetic_csv_rows,
        )
    )

    interaction_rows: list[dict[str, Any]] = []
    for row in interactions.get("interactions", []):
        metrics = row.get("metrics", {})
        interaction_rows.append(
            {
                "runtime": row.get("runtime", ""),
                "workload": row.get("workload", ""),
                "concurrency": row.get("concurrency", ""),
                "boundary": row.get("boundary", ""),
                "status": row.get("status", ""),
                "preproc_lifecycle_avg_ms": _fmt(metrics.get("preproc_lifecycle_avg_ms"), 3),
                "estimated_queue_avg_ms": _fmt(metrics.get("estimated_queue_avg_ms"), 3),
                "talker_to_code2wav_hop_p95_ms": _fmt(
                    metrics.get("talker_to_code2wav_hop_p95_ms"), 3
                ),
                "code2wav_decode_p95_ms": _fmt(metrics.get("code2wav_decode_p95_ms"), 3),
                "batch_admission_span_p95_ms": _fmt(
                    metrics.get("batch_admission_span_p95_ms"), 3
                ),
                "interpretation": row.get("interpretation", ""),
            }
        )
    generated.append(
        _write_csv(
            output_dir / "stage_connection_health.csv",
            label="Stage connection health",
            columns=[
                "runtime",
                "workload",
                "concurrency",
                "boundary",
                "status",
                "preproc_lifecycle_avg_ms",
                "estimated_queue_avg_ms",
                "talker_to_code2wav_hop_p95_ms",
                "code2wav_decode_p95_ms",
                "batch_admission_span_p95_ms",
                "interpretation",
            ],
            rows=interaction_rows,
        )
    )

    vllm_csv_rows = []
    for row in admission_rows:
        vllm_csv_rows.append(
            {
                "label": row.get("label"),
                "concurrency": row.get("concurrency"),
                "diagnosis": row.get("diagnosis"),
                "wall_qps": _fmt(row.get("wall_qps")),
                "runner_qps": _fmt(row.get("runner_qps")),
                "engine_qps": _fmt(row.get("engine_qps")),
                "runner_wall_time_s": _fmt(row.get("runner_wall_time_s")),
                "engine_wall_time_s": _fmt(row.get("engine_wall_time_s")),
                "prompt_build_wall_s": _fmt(row.get("prompt_build_wall_s")),
                "runner_overhead_pct_wall": _fmt(row.get("runner_overhead_pct_wall"), 2),
                "batch_admission_span_avg_ms": _fmt(
                    row.get("batch_admission_span_avg_ms"), 3
                ),
                "batch_admission_span_p95_ms": _fmt(
                    row.get("batch_admission_span_p95_ms"), 3
                ),
                "talker_to_code2wav_drain_p95_ms": _fmt(
                    row.get("talker_to_code2wav_drain_p95_ms"), 3
                ),
                "result_json": row.get("result_json", ""),
                "run_log": row.get("run_log", ""),
            }
        )
    generated.append(
        _write_csv(
            output_dir / "vllm_admission_diagnosis.csv",
            label="vLLM admission diagnosis",
            columns=[
                "label",
                "concurrency",
                "diagnosis",
                "wall_qps",
                "runner_qps",
                "engine_qps",
                "runner_wall_time_s",
                "engine_wall_time_s",
                "prompt_build_wall_s",
                "runner_overhead_pct_wall",
                "batch_admission_span_avg_ms",
                "batch_admission_span_p95_ms",
                "talker_to_code2wav_drain_p95_ms",
                "result_json",
                "run_log",
            ],
            rows=vllm_csv_rows,
        )
    )

    preproc_csv_rows = [
        {
            "setting": row.get("setting"),
            "completed": row.get("completed"),
            "failed": row.get("failed"),
            "throughput_qps": _fmt(row.get("throughput_qps")),
            "latency_mean_s": _fmt(row.get("latency_mean_s")),
            "latency_p95_s": _fmt(row.get("latency_p95_s")),
            "rtf_mean": _fmt(row.get("rtf_mean")),
            "rtf_p95": _fmt(row.get("rtf_p95")),
            "accuracy_pct": _percent(row.get("accuracy"), 2),
        }
        for row in preproc_rows
    ]
    generated.append(
        _write_csv(
            output_dir / "preprocessing_antirecipe.csv",
            label="Preprocessing anti-recipe",
            columns=[
                "setting",
                "completed",
                "failed",
                "throughput_qps",
                "latency_mean_s",
                "latency_p95_s",
                "rtf_mean",
                "rtf_p95",
                "accuracy_pct",
            ],
            rows=preproc_csv_rows,
        )
    )

    budget_csv_rows: list[dict[str, Any]] = []
    for row in sglang_budget_rows:
        budget_csv_rows.append(
            {
                "runtime": row.get("runtime"),
                "workload": row.get("workload"),
                "concurrency": row.get("concurrency"),
                "latency_mean_ms": _fmt(row.get("latency_mean_ms"), 3),
                "rtf_mean": _fmt(row.get("rtf_mean"), 4),
                "qps": _fmt(row.get("qps"), 4),
                "preproc_lifecycle_pct_of_latency": _fmt(
                    row.get("preproc_lifecycle_pct_of_latency"), 3
                ),
                "actual_preprocess_pct_of_latency": _fmt(
                    row.get("actual_preprocess_pct_of_latency"), 3
                ),
                "preproc_queue_pct_of_latency": _fmt(
                    row.get("preproc_queue_pct_of_latency"), 3
                ),
                "talker_pct_of_latency": _fmt(row.get("talker_pct_of_latency"), 3),
                "code2wav_decode_pct_of_latency": _fmt(
                    row.get("code2wav_decode_pct_of_latency"), 3
                ),
                "preproc_lifecycle_avg_ms": _fmt(
                    row.get("preproc_lifecycle_avg_ms"), 3
                ),
                "actual_preprocess_avg_ms": _fmt(
                    row.get("actual_preprocess_avg_ms"), 3
                ),
                "preproc_queue_estimate_ms": _fmt(
                    row.get("preproc_queue_estimate_ms"), 3
                ),
                "talker_avg_ms": _fmt(row.get("talker_avg_ms"), 3),
                "code2wav_decode_avg_ms": _fmt(
                    row.get("code2wav_decode_avg_ms"), 3
                ),
                "diagnosis": row.get("diagnosis", ""),
            }
        )
    generated.append(
        _write_csv(
            output_dir / "sglang_stage_latency_budget.csv",
            label="SGLang stage latency budget",
            columns=[
                "runtime",
                "workload",
                "concurrency",
                "latency_mean_ms",
                "rtf_mean",
                "qps",
                "preproc_lifecycle_pct_of_latency",
                "actual_preprocess_pct_of_latency",
                "preproc_queue_pct_of_latency",
                "talker_pct_of_latency",
                "code2wav_decode_pct_of_latency",
                "preproc_lifecycle_avg_ms",
                "actual_preprocess_avg_ms",
                "preproc_queue_estimate_ms",
                "talker_avg_ms",
                "code2wav_decode_avg_ms",
                "diagnosis",
            ],
            rows=budget_csv_rows,
        )
    )

    generated.append(
        _write_grouped_bar_svg(
            output_dir / "strict_c4_latency_rtf.svg",
            title="Strict c=4 Runtime Comparison",
            subtitle="Video-AMME ci-50 warmed slice; lower is better for all bars",
            categories=["lat mean", "lat p95", "RTF mean", "RTF p95"],
            series=[
                (
                    "vLLM",
                    [
                        _as_float(strict["vllm"]["latency_mean_s"]),
                        _as_float(strict["vllm"]["latency_p95_s"]),
                        _as_float(strict["vllm"]["rtf_mean"]),
                        _as_float(strict["vllm"]["rtf_p95"]),
                    ],
                ),
                (
                    "SGLang",
                    [
                        _as_float(strict["sglang"]["latency_mean_s"]),
                        _as_float(strict["sglang"]["latency_p95_s"]),
                        _as_float(strict["sglang"]["rtf_mean"]),
                        _as_float(strict["sglang"]["rtf_p95"]),
                    ],
                ),
            ],
            y_label="Latency in seconds or RTF units",
        )
    )

    stress_sorted = sorted(stress_rows, key=lambda row: int(row["concurrency"]))
    x_labels = [f"c{int(row['concurrency'])}" for row in stress_sorted]
    generated.append(
        _write_line_svg(
            output_dir / "sglang_pressure_qps.svg",
            title="SGLang Pressure Sweep: QPS",
            subtitle="c=8 is the measured throughput peak; c=16 regresses",
            x_labels=x_labels,
            series=[("QPS", [_as_float(row.get("throughput_qps")) for row in stress_sorted])],
            y_label="requests / second",
        )
    )
    generated.append(
        _write_line_svg(
            output_dir / "sglang_pressure_latency.svg",
            title="SGLang Pressure Sweep: Latency",
            subtitle="High-concurrency latency rises as admission and talker tail interact",
            x_labels=x_labels,
            series=[
                ("mean latency", [_as_float(row.get("latency_mean_s")) for row in stress_sorted]),
                ("p95 latency", [_as_float(row.get("latency_p95_s")) for row in stress_sorted]),
            ],
            y_label="seconds",
        )
    )
    generated.append(
        _write_line_svg(
            output_dir / "sglang_handoff_decode_ms.svg",
            title="SGLang Stage Connection Health",
            subtitle="talker->code2wav hop and code2wav decode stay small versus request tails",
            x_labels=x_labels,
            series=[
                (
                    "hop p95",
                    [
                        _as_float(stage_by_c[int(row["concurrency"])].get("talker_to_code2wav_hop_p95_ms"))
                        for row in stress_sorted
                    ],
                ),
                (
                    "decode p95",
                    [
                        _as_float(stage_by_c[int(row["concurrency"])].get("code2wav_decode_p95_ms"))
                        for row in stress_sorted
                    ],
                ),
            ],
            y_label="milliseconds",
        )
    )
    budget_sorted = sorted(
        sglang_budget_rows, key=lambda row: int(row.get("concurrency") or 0)
    )
    budget_x = [f"c{int(row.get('concurrency') or 0)}" for row in budget_sorted]
    generated.append(
        _write_line_svg(
            output_dir / "sglang_stage_latency_budget_pct.svg",
            title="SGLang Stage Latency Budget",
            subtitle=(
                "Non-additive stage pressure ratios; c=8/c=16 expose admission queueing"
            ),
            x_labels=budget_x,
            series=[
                (
                    "preproc lifecycle",
                    [
                        _as_float(row.get("preproc_lifecycle_pct_of_latency"))
                        for row in budget_sorted
                    ],
                ),
                (
                    "queue estimate",
                    [
                        _as_float(row.get("preproc_queue_pct_of_latency"))
                        for row in budget_sorted
                    ],
                ),
                (
                    "talker",
                    [
                        _as_float(row.get("talker_pct_of_latency"))
                        for row in budget_sorted
                    ],
                ),
                (
                    "code2wav decode",
                    [
                        _as_float(row.get("code2wav_decode_pct_of_latency"))
                        for row in budget_sorted
                    ],
                ),
            ],
            y_label="percent of request latency",
        )
    )

    synthetic_sorted = sorted(
        synthetic_rows, key=lambda row: (str(row.get("scenario")), int(row.get("concurrency")))
    )
    synth_x = [f"{row.get('scenario')} c{int(row.get('concurrency'))}" for row in synthetic_sorted]
    generated.append(
        _write_line_svg(
            output_dir / "synthetic_short_long_rtf.svg",
            title="Synthetic Speech: Short/Long RTF",
            subtitle="long c=8 remains faster than real time",
            x_labels=synth_x,
            series=[("RTF mean", [_as_float(row.get("rtf_mean")) for row in synthetic_sorted])],
            y_label="RTF mean",
        )
    )

    c8_rows = [row for row in admission_rows if int(row.get("concurrency") or 0) == 8]
    generated.append(
        _write_grouped_bar_svg(
            output_dir / "vllm_c8_diagnostic_qps.svg",
            title="vLLM c=8 Offline Diagnostic",
            subtitle="prebuild w4 improves runner wall clock but remains diagnostic, not online parity",
            categories=[str(row.get("label", "")).replace("vLLM-", "") for row in c8_rows],
            series=[
                ("runner QPS", [_as_float(row.get("runner_qps")) for row in c8_rows]),
                ("engine QPS", [_as_float(row.get("engine_qps")) for row in c8_rows]),
            ],
            y_label="requests / second",
        )
    )

    csv_count = sum(1 for item in generated if item["kind"] == "csv")
    svg_count = sum(1 for item in generated if item["kind"] == "svg")
    sources = [
        audit_dir / "tables_summary.json",
        audit_dir / "headline_scorecard.json",
        audit_dir / "stage_interaction_summary.json",
        audit_dir / "vllm_admission_diagnosis.json",
        audit_dir / "stage_latency_budget.json",
    ]
    checks = {
        "strict_c4_csv": any(item["path"].endswith("strict_c4_runtime_comparison.csv") for item in generated),
        "pressure_csv": any(item["path"].endswith("sglang_pressure_sweep.csv") for item in generated),
        "stage_csv": any(item["path"].endswith("stage_connection_health.csv") for item in generated),
        "stage_budget_csv": any(item["path"].endswith("sglang_stage_latency_budget.csv") for item in generated),
        "stage_budget_svg": any(item["path"].endswith("sglang_stage_latency_budget_pct.svg") for item in generated),
        "vllm_csv": any(item["path"].endswith("vllm_admission_diagnosis.csv") for item in generated),
        "enough_svg": svg_count >= 7,
        "enough_csv": csv_count >= 7,
    }
    manifest = {
        "root": str(root),
        "output_dir": str(output_dir),
        "summary": {
            "ready": all(checks.values()),
            "csv_files": csv_count,
            "svg_files": svg_count,
            "generated_files": len(generated),
            "checks": checks,
        },
        "sources": [
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for path in sources
        ],
        "generated_files": generated,
        "usage": {
            "primary_slides": [
                "strict_c4_latency_rtf.svg",
                "sglang_pressure_qps.svg",
                "sglang_stage_latency_budget_pct.svg",
                "sglang_handoff_decode_ms.svg",
                "vllm_c8_diagnostic_qps.svg",
            ],
            "spreadsheet_entry": "sglang_pressure_sweep.csv",
            "note": "SVG/CSV files are derived from audited JSONs; do not hand-edit numbers in slides.",
        },
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build slide-ready CSV/SVG chart assets from Qwen3.5-Omni audit JSONs."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_DIR / "share_charts",
    )
    parser.add_argument("--manifest-output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = build_chart_pack(root, output_dir)

    manifest_output = args.manifest_output
    if manifest_output is None:
        manifest_output = output_dir / "chart_pack_manifest.json"
    elif not manifest_output.is_absolute():
        manifest_output = root / manifest_output
    _save_json(manifest, manifest_output)
    summary = manifest["summary"]
    print(
        "Chart pack written: "
        f"{output_dir} csv={summary['csv_files']} "
        f"svg={summary['svg_files']} ready={summary['ready']}"
    )
    if not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
