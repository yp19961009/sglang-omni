#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze one Qwen3.5-Omni Video-AMME single-example breakdown rerun."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SAMPLE_ID = "011-1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _find_result_with_sample(root: Path, sample_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = _find_files(root, "videoamme_results.json")
    for path in candidates:
        data = _read_json(path)
        if any(row.get("sample_id") == sample_id for row in data.get("per_sample", [])):
            return path, data
    raise FileNotFoundError(f"no videoamme_results.json with sample_id={sample_id} under {root}")


def _sample(data: dict[str, Any], sample_id: str) -> dict[str, Any]:
    for row in data.get("per_sample", []):
        if row.get("sample_id") == sample_id:
            return row
    raise KeyError(f"sample {sample_id} not found")


def _profile_path(root: Path) -> Path:
    candidates = sorted(root.glob("request_profile*.json"))
    if not candidates:
        candidates = _find_files(root, "request_profile*.json")
    if not candidates:
        raise FileNotFoundError(f"no request_profile*.json under {root}")
    return candidates[0]


def _first_event(
    timeline: list[dict[str, Any]],
    stage: str,
    event_name: str,
    *,
    metadata_key: str | None = None,
    metadata_value: Any | None = None,
) -> dict[str, Any]:
    for event in timeline:
        if event.get("stage") != stage or event.get("event_name") != event_name:
            continue
        if metadata_key is not None:
            if event.get("metadata", {}).get(metadata_key) != metadata_value:
                continue
        return event
    raise KeyError(f"event not found: {stage} {event_name} {metadata_key}={metadata_value}")


def _stage_interval(
    profile: dict[str, Any],
    stage: str,
    interval: str,
) -> dict[str, Any] | None:
    for row in profile.get("stage_breakdown", []):
        if row.get("stage") == stage and row.get("interval") == interval:
            return row
    return None


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _pct(part_ms: float, total_ms: float) -> float:
    if total_ms <= 0:
        return 0.0
    return part_ms / total_ms * 100.0


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "total_ms": 0.0, "avg_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "count": len(values),
        "total_ms": round(sum(values), 3),
        "avg_ms": round(statistics.mean(values), 3),
        "p95_ms": round(p95, 3),
        "max_ms": round(max(values), 3),
    }


def _parse_log_ts(line: str) -> datetime | None:
    match = re.search(r"06-22 (?P<t>\d\d:\d\d:\d\d\.\d{3})", line)
    if not match:
        return None
    return datetime.strptime("2026-06-22 " + match.group("t"), "%Y-%m-%d %H:%M:%S.%f")


@dataclass
class VllmLogEvent:
    line_no: int
    line: str
    ts: datetime | None


def _read_log_events(path: Path) -> list[VllmLogEvent]:
    return [
        VllmLogEvent(line_no=i, line=line.rstrip("\n"), ts=_parse_log_ts(line))
        for i, line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1)
    ]


def _find_vllm_log(root: Path) -> Path:
    path = root / "run.log"
    if path.exists():
        return path
    candidates = _find_files(root, "run.log")
    if not candidates:
        raise FileNotFoundError(f"no run.log under {root}")
    return candidates[0]


def analyze_sglang(root: Path) -> dict[str, Any]:
    result_path, result_json = _find_result_with_sample(root, SAMPLE_ID)
    sample = _sample(result_json, SAMPLE_ID)
    profile_file = _profile_path(root)
    profile = _read_json(profile_file)
    timelines = profile.get("timelines", {})
    if len(timelines) != 1:
        raise ValueError(f"expected exactly one SGLang profiled request, got {len(timelines)}")
    request_id, timeline = next(iter(timelines.items()))
    timeline = sorted(timeline, key=lambda item: item["t_rel_ms"])

    admission = _first_event(timeline, "coordinator", "request_admission")["t_rel_ms"]
    preprocess_start = _first_event(timeline, "preprocessing", "preprocess_start")["t_rel_ms"]
    preprocess_end = _first_event(timeline, "preprocessing", "preprocess_end")["t_rel_ms"]
    image_start = _first_event(timeline, "image_encoder", "encoder_start")["t_rel_ms"]
    image_end = _first_event(timeline, "image_encoder", "encoder_end")["t_rel_ms"]
    image_to_mm = _first_event(
        timeline,
        "mm_aggregate",
        "stage_input_received",
        metadata_key="from_stage",
        metadata_value="image_encoder",
    )["t_rel_ms"]
    thinker_input = _first_event(timeline, "thinker", "stage_input_received")["t_rel_ms"]
    thinker_prefill = _first_event(timeline, "thinker", "scheduler_prefill_start")["t_rel_ms"]
    thinker_first_emit = _first_event(timeline, "thinker", "scheduler_first_emit")["t_rel_ms"]
    audio_end = _first_event(timeline, "audio_encoder", "encoder_end")["t_rel_ms"]
    mm_done = _first_event(timeline, "mm_aggregate", "stage_complete")["t_rel_ms"]
    first_audio = _first_event(timeline, "code2wav", "stage_first_stream_chunk_sent")["t_rel_ms"]
    code2wav_done = _first_event(timeline, "code2wav", "stage_complete")["t_rel_ms"]

    ttft_ms = float(sample["text_ttft_s"]) * 1000.0
    components = [
        ("coordinator ingress to preprocess start", preprocess_start - admission),
        ("preprocessing core", preprocess_end - preprocess_start),
        ("preprocessing output / IPC / image admission", image_start - preprocess_end),
        ("image/video encoder compute", image_end - image_start),
        ("image output + mm fan-in", image_to_mm - image_end),
        ("mm aggregate + hop to thinker", thinker_input - image_to_mm),
        ("thinker request build / queue", thinker_prefill - thinker_input),
        ("thinker prefill to first emit", thinker_first_emit - thinker_prefill),
        ("emit/decode/client transfer", ttft_ms - thinker_first_emit),
    ]

    preprocessing_intervals = {
        "media_load_ms": _stage_interval(
            profile, "preprocessing", "preprocess_media_load_start->preprocess_media_load_end"
        ),
        "video_load_ms": _stage_interval(
            profile, "preprocessing", "preprocess_video_load_start->preprocess_video_load_end"
        ),
        "audio_load_ms": _stage_interval(
            profile, "preprocessing", "preprocess_audio_load_start->preprocess_audio_load_end"
        ),
        "hf_processor_ms": _stage_interval(
            profile,
            "preprocessing",
            "preprocess_hf_processor_start->preprocess_hf_processor_end",
        ),
        "prompt_build_ms": _stage_interval(
            profile, "preprocessing", "preprocess_prompt_start->preprocess_prompt_end"
        ),
    }

    return {
        "result_path": str(result_path),
        "profile_path": str(profile_file),
        "request_id": request_id,
        "sample": sample,
        "ttft_components": [
            {
                "component": name,
                "ms": round(ms, 3),
                "pct_of_ttft": round(_pct(ms, ttft_ms), 3),
            }
            for name, ms in components
        ],
        "ttft_component_sum_ms": round(sum(ms for _, ms in components), 3),
        "ttft_json_ms": round(ttft_ms, 3),
        "timeline_anchors_ms": {
            "admission": round(admission, 3),
            "preprocess_start": round(preprocess_start, 3),
            "preprocess_end": round(preprocess_end, 3),
            "image_encoder_start": round(image_start, 3),
            "image_encoder_end": round(image_end, 3),
            "image_payload_to_mm": round(image_to_mm, 3),
            "mm_aggregate_done": round(mm_done, 3),
            "thinker_input": round(thinker_input, 3),
            "thinker_prefill_start": round(thinker_prefill, 3),
            "thinker_first_emit": round(thinker_first_emit, 3),
            "audio_encoder_end": round(audio_end, 3),
            "first_audio_payload": round(first_audio, 3),
            "code2wav_done": round(code2wav_done, 3),
        },
        "preprocessing_detail_ms": {
            key: (round(row["total_ms"], 3) if row else None)
            for key, row in preprocessing_intervals.items()
        },
        "top_stages": sorted(
            profile.get("stage_breakdown", []),
            key=lambda row: row.get("total_ms", 0.0),
            reverse=True,
        )[:16],
        "top_hops": sorted(
            profile.get("hop_breakdown", []),
            key=lambda row: row.get("total_ms", 0.0),
            reverse=True,
        )[:12],
    }


def _line_after(events: list[VllmLogEvent], start_index: int, pattern: str) -> VllmLogEvent | None:
    regex = re.compile(pattern)
    for event in events[start_index:]:
        if regex.search(event.line):
            return event
    return None


def analyze_vllm(root: Path) -> dict[str, Any]:
    result_path, result_json = _find_result_with_sample(root, SAMPLE_ID)
    sample = _sample(result_json, SAMPLE_ID)
    log_path = _find_vllm_log(root)
    events = _read_log_events(log_path)

    rid = None
    rid_event_index = None
    for idx, event in enumerate(events):
        match = re.search(r"Request added\. reqid=(?P<rid>\S*011-1\S*) prompt=", event.line)
        if match:
            rid = match.group("rid")
            rid_event_index = idx
            break
    if rid is None or rid_event_index is None:
        raise ValueError("could not identify vLLM request id for sample 011-1")

    input_start_index = next(
        idx for idx, event in enumerate(events) if rid in event.line and "input_preprocessor start" in event.line
    )
    input_finish_event = _line_after(
        events,
        input_start_index,
        rf"input_preprocessor finished, rid: {re.escape(rid)}, cost:(?P<cost>[0-9.]+)",
    )
    if input_finish_event is None:
        raise ValueError("missing input_preprocessor finish event")
    input_cost_match = re.search(r"cost:(?P<cost>[0-9.]+)", input_finish_event.line)
    input_preprocessor_ms = float(input_cost_match.group("cost"))

    qwen_stats = _line_after(
        events,
        input_start_index,
        r"Qwen3OmniNextProcessor preprocessing stats: .*total_preprocess=(?P<total>[0-9.]+)s",
    )
    qwen_detail: dict[str, float] = {}
    if qwen_stats:
        match = re.search(
            r"audio_items=\d+ \((?P<audio>[0-9.]+)s\), image_items=\d+ \((?P<image>[0-9.]+)s\), "
            r"video_items=\d+ \((?P<video>[0-9.]+)s\), total_preprocess=(?P<total>[0-9.]+)s",
            qwen_stats.line,
        )
        if match:
            qwen_detail = {
                "audio_ms": round(float(match.group("audio")) * 1000, 3),
                "image_ms": round(float(match.group("image")) * 1000, 3),
                "video_ms": round(float(match.group("video")) * 1000, 3),
                "total_ms": round(float(match.group("total")) * 1000, 3),
            }

    measured_video_log = None
    for event in events:
        if "decord:" in event.line and "OE5S-NbNsro.mp4" in event.line:
            measured_video_log = event
            break
    video_read_ms = None
    if measured_video_log:
        match = re.search(r"time=(?P<seconds>[0-9.]+)s", measured_video_log.line)
        if match:
            video_read_ms = round(float(match.group("seconds")) * 1000, 3)

    def find_rid_line(text: str) -> VllmLogEvent | None:
        for event in events:
            if rid in event.line and text in event.line:
                return event
        return None

    first_text_event = find_rid_line("Thinker generator got first output")
    thinker_done_event = find_rid_line("Thinker request finished")
    talker_first_event = find_rid_line("TALKER_FEED_FIRST")
    talker_done_event = find_rid_line("Talker request finished, reason")
    code2wav_done_event = find_rid_line("Code2Wav request finished")

    mm_encode_event = find_rid_line("encode all mm inputs done")
    if mm_encode_event is None:
        for event in events[input_start_index:]:
            if "encode all mm inputs done" in event.line and rid in event.line:
                mm_encode_event = event
                break
    mm_encode_ms = None
    if mm_encode_event:
        match = re.search(r"cost:\s*(?P<cost>[0-9.]+) ms", mm_encode_event.line)
        if match:
            mm_encode_ms = round(float(match.group("cost")), 3)

    hfp_cached = _line_after(events, input_start_index, r"HFPREP_PROFILE cached_apply:")
    hfp_apply = _line_after(events, input_start_index, r"HFPREP_PROFILE apply:")
    shm_put = _line_after(events, input_start_index, r"SHM_PROFILE put:")
    hfp_detail: dict[str, float] = {}
    if hfp_cached:
        match = re.search(
            r"total=(?P<total>[0-9.]+)ms hash=(?P<hash>[0-9.]+)ms "
            r"cache_lookup=(?P<lookup>[0-9.]+)ms hf_proc_miss=(?P<hf>[0-9.]+)ms "
            r"merge=(?P<merge>[0-9.]+)ms",
            hfp_cached.line,
        )
        if match:
            hfp_detail.update(
                {
                    "cached_apply_total_ms": float(match.group("total")),
                    "hash_ms": float(match.group("hash")),
                    "cache_lookup_ms": float(match.group("lookup")),
                    "hf_proc_miss_ms": float(match.group("hf")),
                    "merge_ms": float(match.group("merge")),
                }
            )
    if hfp_apply:
        match = re.search(
            r"total=(?P<total>[0-9.]+)ms cached_hf=(?P<cached>[0-9.]+)ms "
            r"prompt_updates=(?P<updates>[0-9.]+)ms",
            hfp_apply.line,
        )
        if match:
            hfp_detail.update(
                {
                    "apply_total_ms": float(match.group("total")),
                    "apply_cached_hf_ms": float(match.group("cached")),
                    "prompt_updates_ms": float(match.group("updates")),
                }
            )
    shm_detail: dict[str, float] = {}
    if shm_put:
        match = re.search(r"total=(?P<total>[0-9.]+)ms .*shm_put=(?P<put>[0-9.]+)ms", shm_put.line)
        if match:
            shm_detail = {
                "total_ms": float(match.group("total")),
                "shm_put_ms": float(match.group("put")),
            }

    c2w_costs: list[float] = []
    talker_to_c2w_ts: list[datetime] = []
    for event in events:
        if rid in event.line and "C2W_BATCH" in event.line:
            match = re.search(r"cost=(?P<cost>\d+)ms", event.line)
            if match:
                c2w_costs.append(float(match.group("cost")))
        if rid in event.line and "talker to code2wav:" in event.line and event.ts is not None:
            talker_to_c2w_ts.append(event.ts)
    intervals = [
        (talker_to_c2w_ts[i] - talker_to_c2w_ts[i - 1]).total_seconds() * 1000.0
        for i in range(1, len(talker_to_c2w_ts))
    ]

    prompt_build_ms = float(sample.get("prompt_build_s") or 0.0) * 1000.0
    engine_ttft_ms = float(sample.get("engine_text_ttft_s") or 0.0) * 1000.0
    ttft_ms = float(sample["text_ttft_s"]) * 1000.0
    post_input_engine_ms = engine_ttft_ms - input_preprocessor_ms

    return {
        "result_path": str(result_path),
        "log_path": str(log_path),
        "request_id": rid,
        "sample": sample,
        "fast_video_reader": {
            "forced_decord": any("FORCE_QWENVL_VIDEO_READER decord" in e.line for e in events),
            "using_decord": any("qwen-vl-utils using decord" in e.line for e in events),
            "measured_video_read_ms": video_read_ms,
            "measured_video_log_line": measured_video_log.line_no if measured_video_log else None,
        },
        "ttft_components": [
            {
                "component": "external prompt/media build before engine submit",
                "ms": round(prompt_build_ms, 3),
                "pct_of_ttft": round(_pct(prompt_build_ms, ttft_ms), 3),
            },
            {
                "component": "engine input preprocessor",
                "ms": round(input_preprocessor_ms, 3),
                "pct_of_ttft": round(_pct(input_preprocessor_ms, ttft_ms), 3),
            },
            {
                "component": "engine MM encode + prefill + first decode",
                "ms": round(post_input_engine_ms, 3),
                "pct_of_ttft": round(_pct(post_input_engine_ms, ttft_ms), 3),
            },
        ],
        "ttft_component_sum_ms": round(prompt_build_ms + input_preprocessor_ms + post_input_engine_ms, 3),
        "ttft_json_ms": round(ttft_ms, 3),
        "engine_text_ttft_ms": round(engine_ttft_ms, 3),
        "qwen_processor_detail_ms": qwen_detail,
        "hfp_detail_ms": {key: round(value, 3) for key, value in hfp_detail.items()},
        "shm_detail_ms": {key: round(value, 3) for key, value in shm_detail.items()},
        "mm_encode_ms": mm_encode_ms,
        "timeline_events": {
            "input_preprocessor_start_line": events[input_start_index].line_no,
            "input_preprocessor_finish_line": input_finish_event.line_no,
            "first_text_line": first_text_event.line_no if first_text_event else None,
            "talker_first_feed_line": talker_first_event.line_no if talker_first_event else None,
            "thinker_done_line": thinker_done_event.line_no if thinker_done_event else None,
            "talker_done_line": talker_done_event.line_no if talker_done_event else None,
            "code2wav_done_line": code2wav_done_event.line_no if code2wav_done_event else None,
        },
        "c2w_batch_compute": _stats(c2w_costs),
        "talker_to_c2w_interval": _stats(intervals),
    }


def _metric_table(sglang: dict[str, Any], vllm: dict[str, Any]) -> list[dict[str, Any]]:
    s = sglang["sample"]
    v = vllm["sample"]
    keys = [
        "latency_s",
        "text_ttft_s",
        "audio_ttfp_s",
        "audio_duration_s",
        "prompt_tokens",
        "completion_tokens",
        "audio_chunks",
        "inter_chunk_mean_s",
        "rtf",
    ]
    rows = []
    for key in keys:
        sv = s.get(key)
        vv = v.get(key)
        delta = vv - sv if isinstance(sv, (int, float)) and isinstance(vv, (int, float)) else None
        rows.append({"metric": key, "sglang": sv, "vllm_decord": vv, "delta_vllm_minus_sglang": _round(delta, 4) if delta is not None else None})
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    s = summary["sglang"]["sample"]
    v = summary["vllm"]["sample"]
    lines: list[str] = []
    lines.append("# Qwen3.5-Omni Single Example Breakdown Analysis")
    lines.append("")
    lines.append(f"Sample: `{SAMPLE_ID}`. Expected answer `{s.get('expected')}`.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Framework | correct | latency_s | text_ttft_s | audio_ttfp_s | prompt_tokens | output_tokens | audio_duration_s |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(f"| SGLang | {s.get('is_correct')} | {s.get('latency_s')} | {s.get('text_ttft_s')} | {s.get('audio_ttfp_s')} | {s.get('prompt_tokens')} | {s.get('completion_tokens')} | {s.get('audio_duration_s')} |")
    lines.append(f"| vLLM decord | {v.get('is_correct')} | {v.get('latency_s')} | {v.get('text_ttft_s')} | {v.get('audio_ttfp_s')} | {v.get('prompt_tokens')} | {v.get('completion_tokens')} | {v.get('audio_duration_s')} |")
    lines.append("")
    lines.append("## SGLang TTFT Critical Path")
    lines.append("")
    lines.append("| Component | ms | pct_of_ttft |")
    lines.append("| --- | ---: | ---: |")
    for row in summary["sglang"]["ttft_components"]:
        lines.append(f"| {row['component']} | {row['ms']} | {row['pct_of_ttft']}% |")
    lines.append("")
    lines.append(f"Additive check: components `{summary['sglang']['ttft_component_sum_ms']} ms`, JSON TTFT `{summary['sglang']['ttft_json_ms']} ms`.")
    lines.append("")
    lines.append("## vLLM Decord TTFT Critical Path")
    lines.append("")
    lines.append("| Component | ms | pct_of_ttft |")
    lines.append("| --- | ---: | ---: |")
    for row in summary["vllm"]["ttft_components"]:
        lines.append(f"| {row['component']} | {row['ms']} | {row['pct_of_ttft']}% |")
    lines.append("")
    lines.append(f"Additive check: components `{summary['vllm']['ttft_component_sum_ms']} ms`, JSON TTFT `{summary['vllm']['ttft_json_ms']} ms`.")
    lines.append("")
    lines.append("## vLLM Fast Reader Proof")
    lines.append("")
    fast = summary["vllm"]["fast_video_reader"]
    lines.append(f"- Forced decord: `{fast['forced_decord']}`")
    lines.append(f"- Runtime using decord: `{fast['using_decord']}`")
    lines.append(f"- Measured video read: `{fast['measured_video_read_ms']} ms`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- SGLang result: `{summary['sglang']['result_path']}`")
    lines.append(f"- SGLang profile: `{summary['sglang']['profile_path']}`")
    lines.append(f"- vLLM result: `{summary['vllm']['result_path']}`")
    lines.append(f"- vLLM log: `{summary['vllm']['log_path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-run-root", type=Path, required=True)
    parser.add_argument("--vllm-run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    summary = {
        "sample_id": SAMPLE_ID,
        "sglang": analyze_sglang(args.sglang_run_root),
        "vllm": analyze_vllm(args.vllm_run_root),
    }
    summary["metric_table"] = _metric_table(summary["sglang"], summary["vllm"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = render_markdown(summary)
    (args.out_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    if args.print:
        print(report)


if __name__ == "__main__":
    main()
