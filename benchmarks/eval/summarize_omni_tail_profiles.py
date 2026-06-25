# SPDX-License-Identifier: Apache-2.0
"""Summarize Omni benchmark tail requests with request-profiler stage timings.

The benchmark result JSON keeps user-facing sample ids, while the profiler JSON
keeps service-side request ids. When direct metadata is unavailable, this tool
rank-matches the two sorted latency distributions and reports the match error.
Use the output as a tail-bottleneck attribution aid, not as a request-id map.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sglang_omni.profiler.views import RequestTimeline, compute_stage_intervals


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * q
    floor = int(k)
    ceil = min(floor + 1, len(values) - 1)
    if floor == ceil:
        return values[floor]
    return values[floor] + (values[ceil] - values[floor]) * (k - floor)


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}{suffix}"


def _timeline_total_ms(timeline: list[dict[str, Any]]) -> float:
    admissions = [
        event["timestamp_ns"]
        for event in timeline
        if event.get("stage") == "coordinator"
        and event.get("event_name") == "request_admission"
    ]
    terminals = [
        event["timestamp_ns"]
        for event in timeline
        if event.get("stage") == "coordinator"
        and event.get("event_name") == "terminal_response"
    ]
    if not admissions or not terminals:
        timestamps = [event["timestamp_ns"] for event in timeline]
        return (max(timestamps) - min(timestamps)) / 1e6 if timestamps else 0.0
    return (max(terminals) - min(admissions)) / 1e6


def _interval_map(
    profile_json: dict[str, Any],
) -> dict[tuple[str, str, str], list[float]]:
    timelines = {
        rid: RequestTimeline(request_id=rid, events=events)
        for rid, events in profile_json["timelines"].items()
    }
    out: dict[tuple[str, str, str], list[float]] = {}
    for interval in compute_stage_intervals(timelines):
        name = f"{interval.open_event}->{interval.close_event}"
        key = (interval.request_id, interval.stage, name)
        out.setdefault(key, []).append(interval.duration_ms)
    return out


def _one(
    intervals: dict[tuple[str, str, str], list[float]],
    request_id: str,
    stage: str,
    interval: str,
) -> float:
    values = intervals.get((request_id, stage, interval), [])
    return values[0] if values else 0.0


def _sum(
    intervals: dict[tuple[str, str, str], list[float]],
    request_id: str,
    stage: str,
    interval: str,
) -> float:
    return sum(intervals.get((request_id, stage, interval), []))


def _hop_stats(
    timeline: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    pending: dict[tuple[str, str, str, str, int | None], list[int]] = {}
    values: dict[tuple[str, str, str], list[float]] = {}

    for event in timeline:
        name = event.get("event_name")
        metadata = event.get("metadata") or {}
        timestamp_ns = int(event.get("timestamp_ns", 0))
        request_id = event.get("request_id", "")
        stage = event.get("stage", "unknown")

        if name == "stage_hop_sent":
            dst = metadata.get("to_stage", "?")
            key = (request_id, stage, dst, "payload", None)
            pending.setdefault(key, []).append(timestamp_ns)
        elif name == "stage_stream_chunk_sent":
            dst = metadata.get("to_stage", "?")
            key = (
                request_id,
                stage,
                dst,
                "stream_chunk",
                metadata.get("chunk_id"),
            )
            pending.setdefault(key, []).append(timestamp_ns)
        elif name == "stage_input_received":
            src = metadata.get("from_stage")
            if not src or src == "coordinator":
                continue
            key = (request_id, src, stage, "payload", None)
            if pending.get(key):
                sent_ns = pending[key].pop(0)
                values.setdefault((src, stage, "payload"), []).append(
                    (timestamp_ns - sent_ns) / 1e6
                )
        elif name == "stage_stream_chunk_received":
            src = metadata.get("from_stage")
            if not src:
                continue
            key = (
                request_id,
                src,
                stage,
                "stream_chunk",
                metadata.get("chunk_id"),
            )
            if pending.get(key):
                sent_ns = pending[key].pop(0)
                values.setdefault((src, stage, "stream_chunk"), []).append(
                    (timestamp_ns - sent_ns) / 1e6
                )

    return {
        key: {
            "avg_ms": statistics.fmean(durations),
            "p95_ms": _percentile(durations, 0.95),
            "max_ms": max(durations),
            "count": float(len(durations)),
        }
        for key, durations in values.items()
    }


def _profile_entries(profile_json: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": request_id,
            "timeline": timeline,
            "total_ms": _timeline_total_ms(timeline),
        }
        for request_id, timeline in profile_json["timelines"].items()
    ]


def _dominant_component(
    *,
    preproc_queue_ms: float,
    preproc_compute_ms: float,
    talker_ms: float,
    thinker_ms: float,
    code2wav_window_ms: float,
    code2wav_decode_ms: float,
) -> str:
    components = {
        "preproc_queue": preproc_queue_ms,
        "preproc_compute": preproc_compute_ms,
        "talker": talker_ms,
        "thinker": thinker_ms,
        "code2wav_window": code2wav_window_ms,
        "code2wav_decode": code2wav_decode_ms,
    }
    return max(components.items(), key=lambda item: item[1])[0]


def summarize(
    result_json: dict[str, Any],
    profile_json: dict[str, Any],
    *,
    label: str,
    top_k: int,
) -> str:
    samples = sorted(
        result_json["per_sample"],
        key=lambda sample: float(sample.get("latency_s") or 0.0),
        reverse=True,
    )
    profiles = sorted(
        _profile_entries(profile_json),
        key=lambda entry: float(entry["total_ms"]),
        reverse=True,
    )
    if len(samples) != len(profiles):
        raise ValueError(
            f"count mismatch: {len(samples)} per-sample rows vs "
            f"{len(profiles)} profiler timelines"
        )

    intervals = _interval_map(profile_json)
    deltas_ms = [
        abs(float(sample.get("latency_s") or 0.0) * 1000.0 - profile["total_ms"])
        for sample, profile in zip(samples, profiles)
    ]

    lines = [
        f"### {label} rank-match quality",
        "",
        "| Max Delta | P95 Delta | Mean Delta |",
        "| ---: | ---: | ---: |",
        "| {} | {} | {} |".format(
            _fmt(max(deltas_ms), 2, "ms"),
            _fmt(_percentile(deltas_ms, 0.95), 2, "ms"),
            _fmt(statistics.fmean(deltas_ms), 2, "ms"),
        ),
        "",
        f"### {label} top {top_k} tail requests",
        "",
        (
            "| Rank | Sample | Client Latency | Profiler Total | Audio | RTF | "
            "Preproc Life | Preproc Compute | Preproc Queue | Talker | Thinker | "
            "C2W Window Sum | C2W Decode Sum | T->C Stream P95 | Dominant |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | --- |"
        ),
    ]

    for rank, (sample, profile) in enumerate(zip(samples, profiles), 1):
        if rank > top_k:
            break
        request_id = profile["request_id"]
        preproc_life_ms = _one(
            intervals,
            request_id,
            "preprocessing",
            "stage_input_received->stage_complete",
        )
        preproc_compute_ms = _one(
            intervals,
            request_id,
            "preprocessing",
            "preprocess_start->preprocess_end",
        )
        preproc_queue_ms = max(0.0, preproc_life_ms - preproc_compute_ms)
        talker_ms = _one(
            intervals,
            request_id,
            "talker_ar",
            "stage_input_received->stage_complete",
        )
        thinker_ms = _one(
            intervals,
            request_id,
            "thinker",
            "stage_input_received->stage_complete",
        )
        code2wav_window_ms = _sum(
            intervals,
            request_id,
            "code2wav",
            "code2wav_window_collect_start->code2wav_window_collect_end",
        )
        code2wav_decode_ms = _sum(
            intervals,
            request_id,
            "code2wav",
            "code2wav_decode_start->code2wav_decode_end",
        )
        talker_to_code2wav = _hop_stats(profile["timeline"]).get(
            ("talker_ar", "code2wav", "stream_chunk"), {}
        )
        dominant = _dominant_component(
            preproc_queue_ms=preproc_queue_ms,
            preproc_compute_ms=preproc_compute_ms,
            talker_ms=talker_ms,
            thinker_ms=thinker_ms,
            code2wav_window_ms=code2wav_window_ms,
            code2wav_decode_ms=code2wav_decode_ms,
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                rank,
                sample.get("sample_id", ""),
                _fmt(float(sample.get("latency_s") or 0.0), 4, "s"),
                _fmt(float(profile["total_ms"]) / 1000.0, 4, "s"),
                _fmt(float(sample.get("audio_duration_s") or 0.0), 2, "s"),
                _fmt(float(sample.get("rtf") or 0.0), 4),
                _fmt(preproc_life_ms, 1, "ms"),
                _fmt(preproc_compute_ms, 1, "ms"),
                _fmt(preproc_queue_ms, 1, "ms"),
                _fmt(talker_ms, 1, "ms"),
                _fmt(thinker_ms, 1, "ms"),
                _fmt(code2wav_window_ms, 1, "ms"),
                _fmt(code2wav_decode_ms, 1, "ms"),
                _fmt(float(talker_to_code2wav.get("p95_ms", 0.0)), 1, "ms"),
                dominant,
            )
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Omni benchmark tail requests with profiler stages."
    )
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(
        summarize(
            _load_json(args.result_json),
            _load_json(args.profile_json),
            label=args.label,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
