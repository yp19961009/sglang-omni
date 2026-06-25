# SPDX-License-Identifier: Apache-2.0
"""Build a reviewer-facing stage reproduction drilldown for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "stage_reproduction_drilldown.json"

STAGE_DRILLDOWN = AUDIT_DIR / "stage_drilldown_index.json"
METRIC_PROVENANCE = AUDIT_DIR / "metric_provenance_index.json"
REPRO_COMMANDS = AUDIT_DIR / "repro_command_manifest.json"

QUICK_REPRO_GUIDE = [
    {
        "stage_row_id": "stage-boundary-10",
        "question": "SGLang c=8 为什么进入 admission queue?",
        "read_first": "高并发拐点；先看 estimated queue，再看 request timeline。",
    },
    {
        "stage_row_id": "stage-boundary-13",
        "question": "SGLang c=16 为什么不是推荐 serving 点?",
        "read_first": "饱和边界；queue 增长已经压过 stage handoff。",
    },
    {
        "stage_row_id": "budget-synthetic_speech_budget-03",
        "question": "长文本/长语音 c=8 是否仍快于实时?",
        "read_first": "长文本 guardrail；看 RTF 与 Talker cadence。",
    },
    {
        "stage_row_id": "stage-boundary-31",
        "question": "vLLM original c=8 为什么不能直接做 online parity?",
        "read_first": "offline runner prompt-feed/admission 限制。",
    },
    {
        "stage_row_id": "stage-boundary-37",
        "question": "vLLM prebuild w4 证明了什么、没证明什么?",
        "read_first": "诊断-only；解除 admission 后暴露 engine/talker tail。",
    },
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


def _save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _rel_path(root: Path, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path)
    return text


def _normalize_paths(root: Path, values: list[Any]) -> list[str]:
    return _dedupe([_rel_path(root, value) for value in values])


def _path_exists(root: Path, rel_path: str) -> bool:
    path = Path(rel_path)
    if path.is_absolute():
        return path.exists()
    return (root / rel_path).exists()


def _metric_rows(metric_provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id")): row
        for row in _as_list(metric_provenance.get("rows"))
        if isinstance(row, dict) and row.get("id")
    }


def _command_ids(repro_manifest: dict[str, Any]) -> set[str]:
    return {
        str(command.get("id"))
        for command in _as_list(repro_manifest.get("commands"))
        if isinstance(command, dict) and command.get("id")
    }


def _metric_row_id(stage_row_id: str) -> str:
    return f"stage_drilldown_{stage_row_id}"


def _route(row: dict[str, Any]) -> list[str]:
    route = _as_list(row.get("stage_route"))
    if route:
        return [str(item) for item in route]
    return [str(item) for item in _as_list(row.get("stage_focus"))]


def _route_key(row: dict[str, Any]) -> str:
    route = _route(row)
    return " -> ".join(route) if route else "summary"


def _first_metric_text(metrics: dict[str, Any]) -> str:
    if not isinstance(metrics, dict):
        return ""
    priority = [
        "preproc_lifecycle_avg_ms",
        "estimated_queue_avg_ms",
        "talker_to_code2wav_hop_p95_ms",
        "code2wav_decode_p95_ms",
        "code2wav_window_collect_avg_ms",
        "rtf_mean",
        "engine_qps",
        "runner_qps",
        "batch_admission_span_avg_ms",
        "talker_ar_avg_ms",
        "latency_mean_s",
    ]
    parts: list[str] = []
    for key in priority:
        if key in metrics and metrics.get(key) is not None:
            parts.append(f"{key}={metrics.get(key)}")
        if len(parts) >= 3:
            break
    if not parts:
        for key, value in metrics.items():
            if value is not None:
                parts.append(f"{key}={value}")
            if len(parts) >= 3:
                break
    return "; ".join(parts)


def _is_raw_artifact(path: str) -> bool:
    return path.startswith("results/qwen35_") and not path.startswith(
        AUDIT_DIR.as_posix()
    )


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _load_json_array_or_object(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _event_time(event: dict[str, Any]) -> float | None:
    return _num(event.get("t_rel_ms"))


def _events_for_stage(events: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("stage") == stage]


def _first_event_time(
    events: list[dict[str, Any]], stage: str, event_name: str
) -> float | None:
    values = [
        time
        for event in _events_for_stage(events, stage)
        if event.get("event_name") == event_name
        for time in [_event_time(event)]
        if time is not None
    ]
    return min(values) if values else None


def _last_event_time(
    events: list[dict[str, Any]], stage: str, event_name: str
) -> float | None:
    values = [
        time
        for event in _events_for_stage(events, stage)
        if event.get("event_name") == event_name
        for time in [_event_time(event)]
        if time is not None
    ]
    return max(values) if values else None


def _duration_between(
    events: list[dict[str, Any]],
    stage: str,
    start_event: str,
    end_event: str,
) -> float | None:
    start = _first_event_time(events, stage, start_event)
    end = _last_event_time(events, stage, end_event)
    if start is None or end is None or end < start:
        return None
    return end - start


def _sum_paired_intervals(
    events: list[dict[str, Any]],
    stage: str,
    start_event: str,
    end_event: str,
) -> tuple[float | None, int, float | None]:
    starts: list[float] = []
    durations: list[float] = []
    for event in sorted(events, key=lambda item: _event_time(item) or 0.0):
        if event.get("stage") != stage:
            continue
        time = _event_time(event)
        if time is None:
            continue
        if event.get("event_name") == start_event:
            starts.append(time)
        elif event.get("event_name") == end_event and starts:
            start = starts.pop(0)
            if time >= start:
                durations.append(time - start)
    if not durations:
        return None, 0, None
    return sum(durations), len(durations), max(durations)


def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _stream_hop_stats(
    events: list[dict[str, Any]], src: str, dst: str
) -> dict[str, Any]:
    sent: dict[Any, float] = {}
    deltas: list[float] = []
    for event in sorted(events, key=lambda item: _event_time(item) or 0.0):
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        time = _event_time(event)
        if time is None:
            continue
        if (
            event.get("stage") == src
            and event.get("event_name") == "stage_stream_chunk_sent"
            and metadata.get("to_stage") == dst
        ):
            sent[metadata.get("chunk_id")] = time
        elif (
            event.get("stage") == dst
            and event.get("event_name") == "stage_stream_chunk_received"
            and metadata.get("from_stage") == src
        ):
            chunk_id = metadata.get("chunk_id")
            start = sent.get(chunk_id)
            if start is not None and time >= start:
                deltas.append(time - start)
    return {
        "count": len(deltas),
        "avg_ms": _round_ms(sum(deltas) / len(deltas)) if deltas else None,
        "p95_ms": _round_ms(_percentile(deltas, 95.0)),
        "max_ms": _round_ms(max(deltas)) if deltas else None,
    }


def _trace_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    terminal_ms = _last_event_time(events, "coordinator", "terminal_response")
    if terminal_ms is None:
        terminal_ms = max(
            (_event_time(event) or 0.0 for event in events),
            default=0.0,
        )
    stage_lifecycle_ms = {
        stage: _round_ms(
            _duration_between(
                events,
                stage,
                "stage_input_received",
                "stage_complete",
            )
        )
        for stage in [
            "preprocessing",
            "image_encoder",
            "audio_encoder",
            "mm_aggregate",
            "thinker",
            "talker_ar",
            "code2wav",
            "decode",
        ]
    }
    code2wav_decode_total, code2wav_decode_count, code2wav_decode_max = (
        _sum_paired_intervals(
            events,
            "code2wav",
            "code2wav_decode_start",
            "code2wav_decode_end",
        )
    )
    code2wav_collect_total, code2wav_collect_count, code2wav_collect_max = (
        _sum_paired_intervals(
            events,
            "code2wav",
            "code2wav_window_collect_start",
            "code2wav_window_collect_end",
        )
    )
    return {
        "request_id": str(events[0].get("request_id") or "") if events else "",
        "terminal_ms": _round_ms(terminal_ms),
        "stage_lifecycle_ms": {
            key: value for key, value in stage_lifecycle_ms.items() if value is not None
        },
        "actual_preprocess_ms": _round_ms(
            _duration_between(events, "preprocessing", "preprocess_start", "preprocess_end")
        ),
        "code2wav_decode": {
            "total_ms": _round_ms(code2wav_decode_total),
            "count": code2wav_decode_count,
            "max_ms": _round_ms(code2wav_decode_max),
        },
        "code2wav_collect": {
            "total_ms": _round_ms(code2wav_collect_total),
            "count": code2wav_collect_count,
            "max_ms": _round_ms(code2wav_collect_max),
        },
        "stream_hops": {
            "thinker_to_talker_ar": _stream_hop_stats(events, "thinker", "talker_ar"),
            "talker_ar_to_code2wav": _stream_hop_stats(events, "talker_ar", "code2wav"),
        },
    }


def _select_request_trace(
    root: Path,
    *,
    trace_id: str,
    label: str,
    runtime: str,
    case: str,
    profile_path: str,
    selector: str,
    interpretation: str,
    raw_artifacts: list[str],
) -> dict[str, Any]:
    path = root / profile_path
    payload = _load_json_array_or_object(path)
    timelines = payload.get("timelines", {}) if isinstance(payload, dict) else {}
    request_summaries = [
        _trace_summary(events)
        for events in timelines.values()
        if isinstance(events, list) and events
    ]
    request_summaries = [
        row for row in request_summaries if _num(row.get("terminal_ms")) is not None
    ]
    selected: dict[str, Any] = {}
    if request_summaries:
        ordered = sorted(request_summaries, key=lambda row: _num(row.get("terminal_ms"), 0.0) or 0.0)
        if selector == "tail":
            selected = ordered[-1]
        elif selector == "median":
            selected = ordered[len(ordered) // 2]
        else:
            selected = ordered[0]
    selected_terminal = _num(selected.get("terminal_ms"))
    terminal_values = [
        _num(row.get("terminal_ms"), 0.0) or 0.0 for row in request_summaries
    ]
    return {
        "trace_id": trace_id,
        "trace_type": "request_timeline",
        "label": label,
        "runtime": runtime,
        "case": case,
        "selector": selector,
        "request_count": len(request_summaries),
        "request_id": selected.get("request_id"),
        "terminal_ms": selected.get("terminal_ms"),
        "terminal_rank": (
            1
            + sum(1 for value in terminal_values if selected_terminal is not None and value < selected_terminal)
            if selected_terminal is not None
            else None
        ),
        "stage_lifecycle_ms": selected.get("stage_lifecycle_ms", {}),
        "actual_preprocess_ms": selected.get("actual_preprocess_ms"),
        "code2wav_decode": selected.get("code2wav_decode", {}),
        "code2wav_collect": selected.get("code2wav_collect", {}),
        "stream_hops": selected.get("stream_hops", {}),
        "interpretation": interpretation,
        "raw_artifacts": _dedupe([profile_path, *raw_artifacts]),
        "jq_queries": {
            "request_timeline": (
                "jq --arg id "
                f"{json.dumps(str(selected.get('request_id') or ''))} "
                "'.timelines[$id]' "
                f"{profile_path}"
            ),
            "profile_summary": f"jq '.stage_breakdown, .hop_breakdown' {profile_path}",
        },
    }


def _vllm_trace_rows(root: Path) -> list[dict[str, Any]]:
    log_summary = _load_json_optional(root / AUDIT_DIR / "vllm_log_stage_summary.json")
    admission = _load_json_optional(root / AUDIT_DIR / "vllm_admission_diagnosis.json")
    by_label = {
        str(row.get("label")): row
        for row in _as_list(log_summary.get("rows"))
        if isinstance(row, dict)
    }
    admission_by_label = {
        str(row.get("label")): row
        for row in _as_list(admission.get("rows"))
        if isinstance(row, dict)
    }
    labels = ["vLLM-c8", "vLLM-c8-prebuild-w4"]
    rows: list[dict[str, Any]] = []
    for label in labels:
        row = by_label.get(label, {})
        adm = admission_by_label.get(label, {})
        rows.append(
            {
                "trace_id": f"vllm_{label.lower().replace('-', '_')}_aggregate",
                "trace_type": "batch_aggregate",
                "label": label,
                "runtime": "vllm",
                "case": label,
                "selector": "p95_batch_signal",
                "request_count": row.get("included_request_ids"),
                "batch_count": row.get("included_batch_count"),
                "batch_admission_span_ms": row.get("batch_admission_span_ms", {}),
                "batch_last_engine_lag_ms": row.get("batch_last_engine_lag_ms", {}),
                "engine_boundary_p95_ms": {
                    "encoder": row.get("encoder_mm_ms", {}).get("p95"),
                    "thinker_to_talker": row.get("thinker_to_talker_feed_ms", {}).get("p95"),
                    "talker_to_code2wav_drain": row.get("talker_to_code2wav_drain_ms", {}).get("p95"),
                    "talker_feed_to_first_codec": row.get("talker_feed_to_first_codec_ms", {}).get("p95"),
                },
                "diagnosis": adm.get("diagnosis"),
                "interpretation": (
                    "Original c8 is runner prompt-feed/admission limited; engine-side stage boundaries are not the primary limiter."
                    if label == "vLLM-c8"
                    else "Prebuild w4 removes most offline admission span and exposes later engine/talker tails; keep it diagnostic-only."
                ),
                "raw_artifacts": _case_raw_artifacts("vllm", label),
                "jq_queries": {
                    "log_stage_row": (
                        "jq '.rows[] | select(.label == "
                        f"\"{label}\")' results/qwen35_report_audit_20260619/vllm_log_stage_summary.json"
                    ),
                    "admission_row": (
                        "jq '.rows[] | select(.label == "
                        f"\"{label}\")' results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"
                    ),
                },
            }
        )
    return rows


def _representative_traces(root: Path) -> list[dict[str, Any]]:
    sglang_traces = [
        {
            "trace_id": "sglang_videoamme_c1_median",
            "label": "SGLang Video-AMME c=1 median request",
            "runtime": "sglang",
            "case": "Video-AMME ci-50 c=1",
            "profile_path": "results/qwen35_sglang_mr8_stress_20260619/request_profile_c1_warm_profile_skipwer.json",
            "selector": "median",
            "interpretation": "Single-concurrency median path: talker dominates while preprocessing and code2wav remain bounded.",
            "raw_artifacts": _case_raw_artifacts("sglang", "Video-AMME ci-50 c=1"),
        },
        {
            "trace_id": "sglang_videoamme_c4_median",
            "label": "SGLang Video-AMME c=4 median request",
            "runtime": "sglang",
            "case": "Video-AMME ci-50 c=4",
            "profile_path": "results/qwen35_sglang_mr8_stress_20260619/request_profile_c4_profile_skipwer.json",
            "selector": "median",
            "interpretation": "Balanced serving point: route timing remains bounded before the c8 queue transition.",
            "raw_artifacts": _case_raw_artifacts("sglang", "Video-AMME ci-50 c=4"),
        },
        {
            "trace_id": "sglang_videoamme_c8_tail",
            "label": "SGLang Video-AMME c=8 tail request",
            "runtime": "sglang",
            "case": "Video-AMME ci-50 c=8",
            "profile_path": "results/qwen35_sglang_mr8_stress_20260619/request_profile_c8_profile_skipwer.json",
            "selector": "tail",
            "interpretation": "Throughput edge: tail request shows admission/queue plus talker pressure, while talker-to-code2wav hops stay small.",
            "raw_artifacts": _case_raw_artifacts("sglang", "Video-AMME ci-50 c=8"),
        },
        {
            "trace_id": "sglang_videoamme_c16_tail",
            "label": "SGLang Video-AMME c=16 tail request",
            "runtime": "sglang",
            "case": "Video-AMME ci-50 c=16",
            "profile_path": "results/qwen35_sglang_mr8_stress_20260619/request_profile_c16_profile_skipwer.json",
            "selector": "tail",
            "interpretation": "Saturation boundary: tail request should be read as pressure evidence, not the recommended serving point.",
            "raw_artifacts": _case_raw_artifacts("sglang", "Video-AMME ci-50 c=16"),
        },
        {
            "trace_id": "sglang_synthetic_short_c8_tail",
            "label": "SGLang synthetic short c=8 tail request",
            "runtime": "sglang",
            "case": "synthetic_short c=8",
            "profile_path": "results/qwen35_synthetic_speech_20260619/request_profile_short_c8_profile.json",
            "selector": "tail",
            "interpretation": "Short text-to-speech isolates Talker/Code2wav without video/audio encoder pressure.",
            "raw_artifacts": _case_raw_artifacts("sglang", "synthetic_short c=8"),
        },
        {
            "trace_id": "sglang_synthetic_long_c8_tail",
            "label": "SGLang synthetic long c=8 tail request",
            "runtime": "sglang",
            "case": "synthetic_long c=8",
            "profile_path": "results/qwen35_synthetic_speech_20260619/request_profile_long_c8_profile.json",
            "selector": "tail",
            "interpretation": "Long text-to-speech remains faster than real time; the trace shows Talker cadence dominates, not vocoder decode.",
            "raw_artifacts": _case_raw_artifacts("sglang", "synthetic_long c=8"),
        },
    ]
    rows = [
        _select_request_trace(root, **trace)
        for trace in sglang_traces
    ]
    rows.extend(_vllm_trace_rows(root))
    return rows


def _case_raw_artifacts(runtime: Any, case: Any) -> list[str]:
    runtime_text = str(runtime or "")
    case_text = str(case or "")
    if runtime_text == "sglang" and "Video-AMME ci-50" in case_text:
        match = re.search(r"c=(\d+)", case_text)
        if not match:
            return []
        concurrency = int(match.group(1))
        if concurrency in {1, 2}:
            suffix = f"c{concurrency}_warm"
        else:
            suffix = f"c{concurrency}"
        return [
            (
                "results/qwen35_sglang_mr8_stress_20260619/"
                f"benchmark_audio_50_{suffix}_profile_skipwer/videoamme_results.json"
            ),
            (
                "results/qwen35_sglang_mr8_stress_20260619/"
                f"request_profile_{suffix}_profile_skipwer.json"
            ),
        ]
    if runtime_text == "sglang" and case_text.startswith("synthetic_"):
        match = re.search(r"c=(\d+)", case_text)
        if not match:
            return []
        scenario = "long" if "long" in case_text else "short"
        concurrency = int(match.group(1))
        return [
            (
                "results/qwen35_synthetic_speech_20260619/"
                f"{scenario}_c{concurrency}/synthetic_speech_results.json"
            ),
            (
                "results/qwen35_synthetic_speech_20260619/"
                f"request_profile_{scenario}_c{concurrency}_profile.json"
            ),
        ]
    vllm_case_map = {
        "vLLM-c1": [
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json",
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log",
        ],
        "vLLM-c4": [
            "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
            "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
        ],
        "vLLM-c8": [
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
        ],
        "vLLM-c8-prebuild-w1": [
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
        ],
        "vLLM-c8-prebuild-w4": [
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
        ],
    }
    if case_text in vllm_case_map:
        return vllm_case_map[case_text]
    for prefix, artifacts in sorted(
        vllm_case_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if case_text.startswith(prefix):
            return artifacts
    return []


def _build_rows(
    root: Path,
    *,
    stage_drilldown: dict[str, Any],
    metric_provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    metric_by_id = _metric_rows(metric_provenance)
    rows: list[dict[str, Any]] = []
    missing_metric_row_ids: list[str] = []
    for source_row in _as_list(stage_drilldown.get("rows")):
        if not isinstance(source_row, dict):
            continue
        stage_row_id = str(source_row.get("id") or "")
        metric_row_id = _metric_row_id(stage_row_id)
        metric_row = metric_by_id.get(metric_row_id)
        if metric_row is None:
            missing_metric_row_ids.append(metric_row_id)
        metrics = source_row.get("metrics", {})
        evidence_files = _normalize_paths(
            root,
            [
                STAGE_DRILLDOWN.as_posix(),
                METRIC_PROVENANCE.as_posix(),
                *list(_as_list(source_row.get("evidence_files"))),
                *list(_as_list(metric_row.get("evidence_files") if metric_row else [])),
                *_case_raw_artifacts(
                    source_row.get("runtime"),
                    source_row.get("case"),
                ),
            ],
        )
        command_ids = _dedupe(
            [
                *list(_as_list(source_row.get("rerun_command_ids"))),
                *list(_as_list(metric_row.get("rerun_command_ids") if metric_row else [])),
                "build_metric_provenance_index",
                "build_repro_command_manifest",
                "build_stage_reproduction_drilldown",
            ]
        )
        rows.append(
            {
                "stage_row_id": stage_row_id,
                "metric_row_id": metric_row_id,
                "entry_type": source_row.get("entry_type"),
                "runtime": source_row.get("runtime"),
                "case": source_row.get("case"),
                "route": _route(source_row),
                "route_key": _route_key(source_row),
                "status": source_row.get("status"),
                "bottleneck_verdict": source_row.get("bottleneck_verdict"),
                "diagnosis": source_row.get("diagnosis"),
                "safe_conclusion": source_row.get("safe_conclusion"),
                "key_metric_text": _first_metric_text(metrics),
                "metric_source_json": metric_row.get("source_json") if metric_row else None,
                "metric_source_pointer": metric_row.get("source_pointer") if metric_row else None,
                "evidence_files": evidence_files,
                "raw_artifacts": sorted(
                    path for path in evidence_files if _is_raw_artifact(path)
                ),
                "rerun_command_ids": command_ids,
                "jq_queries": {
                    "stage_row": (
                        "jq '.rows[] | select(.id == "
                        f"\"{stage_row_id}\")' "
                        "results/qwen35_report_audit_20260619/stage_drilldown_index.json"
                    ),
                    "metric_row": (
                        "jq '.rows[] | select(.id == "
                        f"\"{metric_row_id}\")' "
                        "results/qwen35_report_audit_20260619/metric_provenance_index.json"
                    ),
                    "rerun_commands": (
                        "jq '.commands[] | select(.id as $id | "
                        f"{json.dumps(command_ids)} | index($id))' "
                        "results/qwen35_report_audit_20260619/repro_command_manifest.json"
                    ),
                },
            }
        )
    return rows, missing_metric_row_ids


def _route_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("route_key") or "summary")].append(row)
    summaries: list[dict[str, Any]] = []
    for route_key, route_rows in sorted(grouped.items()):
        statuses = Counter(str(row.get("status") or "") for row in route_rows)
        verdicts = Counter(str(row.get("bottleneck_verdict") or "") for row in route_rows)
        runtimes = sorted({str(row.get("runtime") or "") for row in route_rows})
        summaries.append(
            {
                "route_key": route_key,
                "rows_total": len(route_rows),
                "runtimes": runtimes,
                "status_counts": dict(sorted(statuses.items())),
                "verdict_counts": dict(sorted(verdicts.items())),
                "first_stage_row_id": route_rows[0].get("stage_row_id"),
            }
        )
    return summaries


def build_stage_reproduction_drilldown(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    stage_drilldown = _load_json_optional(root / STAGE_DRILLDOWN)
    metric_provenance = _load_json_optional(root / METRIC_PROVENANCE)
    repro_manifest = _load_json_optional(root / REPRO_COMMANDS)

    rows, missing_metric_row_ids = _build_rows(
        root,
        stage_drilldown=stage_drilldown,
        metric_provenance=metric_provenance,
    )
    route_rows = _route_summary(rows)
    stage_summary = _summary(stage_drilldown)
    metric_summary = _summary(metric_provenance)
    repro_summary = _summary(repro_manifest)
    command_ids = _command_ids(repro_manifest)
    evidence_files = sorted(
        {
            evidence
            for row in rows
            for evidence in _as_list(row.get("evidence_files"))
        }
    )
    missing_evidence_files = sorted(
        evidence for evidence in evidence_files if not _path_exists(root, evidence)
    )
    command_refs = sorted(
        {
            command_id
            for row in rows
            for command_id in _as_list(row.get("rerun_command_ids"))
        }
    )
    missing_command_ids = sorted(
        command_id for command_id in command_refs if command_id not in command_ids
    )
    row_ids = [str(row.get("stage_row_id") or "") for row in rows]
    duplicate_stage_row_ids = sorted(
        row_id for row_id, count in Counter(row_ids).items() if row_id and count > 1
    )
    raw_artifacts = sorted(
        {
            artifact
            for row in rows
            for artifact in _as_list(row.get("raw_artifacts"))
        }
    )
    quick_reproduction_map = _quick_reproduction_map(rows)
    representative_traces = _representative_traces(root)
    representative_raw_artifacts = sorted(
        {
            artifact
            for row in representative_traces
            for artifact in _as_list(row.get("raw_artifacts"))
        }
    )
    missing_representative_raw_artifacts = sorted(
        artifact
        for artifact in representative_raw_artifacts
        if not _path_exists(root, artifact)
    )
    representative_sglang_cases = {
        str(row.get("case") or "")
        for row in representative_traces
        if row.get("runtime") == "sglang"
    }
    representative_vllm_cases = {
        str(row.get("case") or "")
        for row in representative_traces
        if row.get("runtime") == "vllm"
    }
    required_representative_cases = {
        "Video-AMME ci-50 c=1",
        "Video-AMME ci-50 c=4",
        "Video-AMME ci-50 c=8",
        "Video-AMME ci-50 c=16",
        "synthetic_short c=8",
        "synthetic_long c=8",
    }
    checks = {
        "stage_drilldown_ready": bool(stage_summary.get("ready"))
        and int(stage_summary.get("rows_total") or 0) >= 52
        and int(stage_summary.get("boundary_rows_total") or 0) >= 37
        and int(stage_summary.get("budget_rows_total") or 0) >= 15
        and int(stage_summary.get("stage_routes_total") or 0) >= 11
        and int(stage_summary.get("required_failures") or 0) == 0,
        "metric_provenance_ready": bool(metric_summary.get("ready"))
        and int(metric_summary.get("rows_total") or 0) >= 150
        and int(metric_summary.get("required_failures") or 0) == 0,
        "repro_command_registry_ready": bool(
            repro_summary.get("required_command_ids_present")
        )
        and int(repro_summary.get("commands_total") or 0) >= 52,
        "row_shape": len(rows) >= 52
        and sum(1 for row in rows if row.get("entry_type") == "stage_boundary") >= 37
        and sum(1 for row in rows if row.get("entry_type") == "latency_budget") >= 15,
        "route_coverage": len(route_rows) >= 11,
        "metric_rows_present": not missing_metric_row_ids,
        "evidence_files_present": not missing_evidence_files,
        "rerun_command_ids_present": not missing_command_ids,
        "raw_artifact_links_present": len(raw_artifacts) >= 10,
        "jq_queries_present": all(
            row.get("jq_queries", {}).get("stage_row")
            and row.get("jq_queries", {}).get("metric_row")
            and row.get("jq_queries", {}).get("rerun_commands")
            for row in rows
        ),
        "stage_row_ids_unique": not duplicate_stage_row_ids,
        "representative_sglang_traces_present": required_representative_cases.issubset(
            representative_sglang_cases
        )
        and sum(1 for row in representative_traces if row.get("runtime") == "sglang")
        >= 6,
        "representative_vllm_traces_present": {
            "vLLM-c8",
            "vLLM-c8-prebuild-w4",
        }.issubset(representative_vllm_cases),
        "representative_trace_raw_artifacts_present": not missing_representative_raw_artifacts,
        "representative_trace_jq_queries_present": all(
            row.get("jq_queries") for row in representative_traces
        ),
        "representative_trace_stage_data_present": all(
            row.get("trace_type") != "request_timeline"
            or (
                row.get("request_id")
                and row.get("terminal_ms") is not None
                and row.get("stage_lifecycle_ms")
            )
            for row in representative_traces
        ),
        "quick_reproduction_map_present": len(quick_reproduction_map)
        == len(QUICK_REPRO_GUIDE)
        and all(
            row.get("stage_query")
            and row.get("metric_query")
            and row.get("first_rerun_command_id")
            for row in quick_reproduction_map
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "stage_rows_total": len(rows),
            "boundary_rows_total": sum(
                1 for row in rows if row.get("entry_type") == "stage_boundary"
            ),
            "budget_rows_total": sum(
                1 for row in rows if row.get("entry_type") == "latency_budget"
            ),
            "route_rows_total": len(route_rows),
            "representative_traces_total": len(representative_traces),
            "representative_sglang_request_traces_total": sum(
                1 for row in representative_traces if row.get("runtime") == "sglang"
            ),
            "representative_vllm_diagnostic_traces_total": sum(
                1 for row in representative_traces if row.get("runtime") == "vllm"
            ),
            "raw_artifacts_total": len(raw_artifacts),
            "representative_raw_artifacts_total": len(representative_raw_artifacts),
            "evidence_files_total": len(evidence_files),
            "command_refs_total": len(command_refs),
            "quick_reproduction_routes_total": len(quick_reproduction_map),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "share_scope": (
                "Reviewer-facing drilldown for reproducing every audited stage "
                "boundary and latency-budget row through jq queries, metric "
                "provenance rows, evidence files, raw artifacts, and rerun commands."
            ),
        },
        "checks": checks,
        "diagnostics": {
            "missing_metric_row_ids": sorted(set(missing_metric_row_ids)),
            "missing_evidence_files": missing_evidence_files,
            "missing_command_ids": missing_command_ids,
            "duplicate_stage_row_ids": duplicate_stage_row_ids,
            "missing_representative_raw_artifacts": missing_representative_raw_artifacts,
            "required_failures": required_failures,
        },
        "source_summaries": {
            "stage_drilldown_index": stage_summary,
            "metric_provenance_index": metric_summary,
            "repro_command_manifest": repro_summary,
        },
        "route_summary": route_rows,
        "quick_reproduction_map": quick_reproduction_map,
        "representative_traces": representative_traces,
        "rows": rows,
    }


def _md(text: Any) -> str:
    value = str(text if text is not None else "")
    return value.replace("|", "\\|").replace("\n", " ")


def _fmt_counter(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def _critical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    critical_statuses = {
        "queue_limited",
        "contention_regression",
        "prompt_feed_limited",
        "bottleneck",
        "diagnostic_only",
        "watch",
        "admission_queue_plus_talker_tail",
        "saturation_boundary",
        "engine_or_workload_limited",
        "long_speech_talker_ar_dominant_but_faster_than_realtime",
    }
    selected = [
        row
        for row in rows
        if str(row.get("status") or "") in critical_statuses
        or str(row.get("bottleneck_verdict") or "") not in {"", "not_current_bottleneck"}
    ]
    return selected[:24]


def _quick_reproduction_map(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_id = {str(row.get("stage_row_id") or ""): row for row in rows}
    quick_rows: list[dict[str, Any]] = []
    for item in QUICK_REPRO_GUIDE:
        stage_row_id = item["stage_row_id"]
        row = rows_by_id.get(stage_row_id)
        if not row:
            continue
        jq_queries = row.get("jq_queries", {})
        if not isinstance(jq_queries, dict):
            jq_queries = {}
        rerun_commands = _as_list(row.get("rerun_command_ids"))
        quick_rows.append(
            {
                "question": item["question"],
                "stage_row_id": stage_row_id,
                "metric_row_id": row.get("metric_row_id"),
                "runtime": row.get("runtime"),
                "case": row.get("case"),
                "route_key": row.get("route_key"),
                "status": row.get("status"),
                "bottleneck_verdict": row.get("bottleneck_verdict"),
                "key_metric_text": row.get("key_metric_text"),
                "read_first": item["read_first"],
                "first_rerun_command_id": rerun_commands[0] if rerun_commands else "",
                "stage_query": jq_queries.get("stage_row", ""),
                "metric_query": jq_queries.get("metric_row", ""),
            }
        )
    return quick_rows


def _format_trace_stages(row: dict[str, Any]) -> str:
    stages = row.get("stage_lifecycle_ms", {})
    if not isinstance(stages, dict):
        return ""
    priority = [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "talker_ar",
        "code2wav",
        "decode",
    ]
    return "; ".join(
        f"{stage}={stages[stage]}ms"
        for stage in priority
        if stages.get(stage) is not None
    )


def _format_trace_hops(row: dict[str, Any]) -> str:
    hops = row.get("stream_hops", {})
    if not isinstance(hops, dict):
        return ""
    parts: list[str] = []
    for key in ["thinker_to_talker_ar", "talker_ar_to_code2wav"]:
        value = hops.get(key, {})
        if not isinstance(value, dict) or not value.get("count"):
            continue
        parts.append(
            f"{key}: n={value.get('count')}, p95={value.get('p95_ms')}ms, max={value.get('max_ms')}ms"
        )
    return "; ".join(parts)


def _format_vllm_trace(row: dict[str, Any]) -> str:
    if row.get("trace_type") != "batch_aggregate":
        return ""
    admission = row.get("batch_admission_span_ms", {})
    lag = row.get("batch_last_engine_lag_ms", {})
    boundary = row.get("engine_boundary_p95_ms", {})
    if not isinstance(admission, dict):
        admission = {}
    if not isinstance(lag, dict):
        lag = {}
    if not isinstance(boundary, dict):
        boundary = {}
    return (
        f"admission_p95={admission.get('p95')}ms; "
        f"last_engine_lag_p95={lag.get('p95')}ms; "
        f"encoder_p95={boundary.get('encoder')}ms; "
        f"thinker_to_talker_p95={boundary.get('thinker_to_talker')}ms; "
        f"talker_to_c2w_p95={boundary.get('talker_to_code2wav_drain')}ms"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = [
        "# Qwen3.5-Omni Stage 复现实操 Drilldown",
        "",
        "状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。",
        "工作目录：`/home/gangouyu/sglang-omni`。",
        "用途：给外部 reviewer 按 stage boundary 或 latency budget 复核结论、定位原始证据、读取 metric provenance row，并找到对应复跑命令。",
        "",
        "## 1. 当前 Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| ready | `{summary['ready']}` |",
        f"| stage rows | `{summary['stage_rows_total']}` |",
        f"| boundary rows | `{summary['boundary_rows_total']}` |",
        f"| budget rows | `{summary['budget_rows_total']}` |",
        f"| route rows | `{summary['route_rows_total']}` |",
        f"| representative traces | `{summary['representative_traces_total']}` |",
        f"| representative SGLang traces | `{summary['representative_sglang_request_traces_total']}` |",
        f"| representative vLLM traces | `{summary['representative_vllm_diagnostic_traces_total']}` |",
        f"| raw artifacts | `{summary['raw_artifacts_total']}` |",
        f"| representative raw artifacts | `{summary['representative_raw_artifacts_total']}` |",
        f"| command refs | `{summary['command_refs_total']}` |",
        f"| quick reproduction routes | `{summary['quick_reproduction_routes_total']}` |",
        f"| checks | `{summary['checks_passed']}/{summary['checks_total']}` |",
        f"| required failures | `{summary['required_failures']}` |",
        "",
        "## 2. 答辩先查路线图",
        "",
        "先不要从完整大表开始读。外部 reviewer 通常会追问下面五类问题；每一行都能直接回到 stage row、metric provenance row、原始 artifact 和复跑 command。",
        "",
        "| 问题 | Stage Row | Runtime / Case | Key Signal | First Command |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("quick_reproduction_map", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("question")),
                    _md(row.get("stage_row_id")),
                    _md(f"{row.get('runtime')} / {row.get('case')}"),
                    _md(row.get("key_metric_text")),
                    _md(row.get("first_rerun_command_id")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "三步复现法：",
            "",
            "```bash",
            "cd /home/gangouyu/sglang-omni",
            "jq '.quick_reproduction_map[] | {question, stage_row_id, metric_row_id, first_rerun_command_id}' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
            "jq '.rows[] | select(.stage_row_id == \"stage-boundary-10\")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
            "jq '.commands[] | select(.id == \"sglang_videoamme_stress\")' results/qwen35_report_audit_20260619/repro_command_manifest.json",
            "```",
            "",
            "第一条命令给路线图，第二条换成要追的 `stage_row_id`，第三条换成该行的 `first_rerun_command_id`。如果要看最原始的 stage source，可再用下一节的 `stage_drilldown_index.json` 查询。",
            "",
            "## 3. 单个 Stage 怎么查",
            "",
            "以 SGLang c=8 admission 到 preprocessing 的 queue 边界为例：",
            "",
            "```bash",
            "cd /home/gangouyu/sglang-omni",
            "jq '.rows[] | select(.id == \"stage-boundary-10\")' results/qwen35_report_audit_20260619/stage_drilldown_index.json",
            "jq '.rows[] | select(.id == \"stage_drilldown_stage-boundary-10\")' results/qwen35_report_audit_20260619/metric_provenance_index.json",
            "jq '.commands[] | select(.id == \"sglang_videoamme_stress\")' results/qwen35_report_audit_20260619/repro_command_manifest.json",
            "```",
            "",
            "如果 reviewer 问任意 stage，只要把 `stage-boundary-10` 换成本文件表中的 `stage_row_id`，并把 metric row 换成对应的 `metric_row_id`。",
            "",
            "## 4. Route 覆盖总览",
            "",
            "| Route | Rows | Runtimes | Status Counts | Verdict Counts | First Row |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in payload["route_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("route_key")),
                    str(row.get("rows_total")),
                    _md(", ".join(_as_list(row.get("runtimes")))),
                    _md(_fmt_counter(row.get("status_counts"))),
                    _md(_fmt_counter(row.get("verdict_counts"))),
                    _md(row.get("first_stage_row_id")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 5. 优先追问行",
            "",
            "| Stage Row | Metric Row | Runtime | Case | Route | Status | Verdict | Key Metrics | Commands |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in _critical_rows(payload["rows"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("stage_row_id")),
                    _md(row.get("metric_row_id")),
                    _md(row.get("runtime")),
                    _md(row.get("case")),
                    _md(row.get("route_key")),
                    _md(row.get("status")),
                    _md(row.get("bottleneck_verdict")),
                    _md(row.get("key_metric_text")),
                    _md(", ".join(_as_list(row.get("rerun_command_ids"))[:5])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 6. Representative Request / Batch Trace",
            "",
            "这张表给答辩时快速下钻用：SGLang 行是从原始 `request_profile_*.json` 中抽出的 median/tail request timeline，vLLM 行是 offline batch aggregate。它不替代总体统计，只回答“这条 stage 结论在原始 timeline 里长什么样”。",
            "",
            "| Trace | Type | Runtime | Case | Selector | Request/Batch | Terminal / Key Time | Stage Lifecycle | Hops / vLLM Boundary | Raw Artifacts | jq |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("representative_traces", []):
        if not isinstance(row, dict):
            continue
        if row.get("trace_type") == "batch_aggregate":
            request_or_batch = (
                f"requests={row.get('request_count')}, batches={row.get('batch_count')}"
            )
            key_time = _format_vllm_trace(row)
            stages = ""
            hops = _md(row.get("interpretation"))
        else:
            request_or_batch = str(row.get("request_id") or "")
            key_time = f"{row.get('terminal_ms')}ms"
            stages = _format_trace_stages(row)
            hops = _format_trace_hops(row)
        jq_entry = next(iter(row.get("jq_queries", {}).values()), "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("trace_id")),
                    _md(row.get("trace_type")),
                    _md(row.get("runtime")),
                    _md(row.get("case")),
                    _md(row.get("selector")),
                    _md(request_or_batch),
                    _md(key_time),
                    _md(stages),
                    _md(hops),
                    _md(", ".join(_as_list(row.get("raw_artifacts"))[:3])),
                    "`" + _md(jq_entry) + "`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "读法：如果 c=8/c=16 的 `preprocessing` lifecycle 明显高于 actual preprocess，说明压力是 admission/queue；如果 `talker_ar_to_code2wav` p95 仍是几十毫秒级，说明 stage connection 不是主要瓶颈。synthetic long c=8 用来确认长文本压力落在 Talker cadence，而不是 code2wav decode。vLLM c=8 original/prebuild 的对照只用于解释 offline runner admission 和后续 engine/talker tail，不能升级成 online parity。",
            "",
            "## 7. 完整 Stage 复现索引",
            "",
            "| Stage Row | Metric Row | Runtime | Case | Route | Status | Evidence | jq Stage Query |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["rows"]:
        evidence = ", ".join(_as_list(row.get("evidence_files"))[:3])
        stage_query = row.get("jq_queries", {}).get("stage_row", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("stage_row_id")),
                    _md(row.get("metric_row_id")),
                    _md(row.get("runtime")),
                    _md(row.get("case")),
                    _md(row.get("route_key")),
                    _md(row.get("status")),
                    _md(evidence),
                    "`" + _md(stage_query) + "`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 8. 机器证据",
            "",
            "- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`",
            "- `results/qwen35_report_audit_20260619/stage_drilldown_index.json`",
            "- `results/qwen35_report_audit_20260619/metric_provenance_index.json`",
            "- `results/qwen35_report_audit_20260619/repro_command_manifest.json`",
            "- `benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`",
            "- `benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`",
            "- `benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`",
            "",
        ]
    )
    return "\n".join(lines)


def _print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# Qwen3.5-Omni Stage Reproduction Drilldown")
    print()
    print(f"- ready: `{summary['ready']}`")
    print(f"- stage rows: `{summary['stage_rows_total']}`")
    print(f"- route rows: `{summary['route_rows_total']}`")
    print(f"- raw artifacts: `{summary['raw_artifacts_total']}`")
    print(f"- command refs: `{summary['command_refs_total']}`")
    print(f"- checks: `{summary['checks_passed']}/{summary['checks_total']}`")
    print(f"- required_failures: `{summary['required_failures']}`")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni stage reproduction drilldown."
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
    payload = build_stage_reproduction_drilldown(root)
    _save_json(payload, json_output)
    _save_text(render_markdown(payload), output)
    _print_markdown(payload)
    print(f"Stage reproduction drilldown written: {output}")
    print(f"Stage reproduction drilldown JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
