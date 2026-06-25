# SPDX-License-Identifier: Apache-2.0
"""Summarize vLLM-Omni stage signals from saved Qwen3.5-Omni run logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


TS_RE = re.compile(
    r"INFO (?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\.(?P<msec>\d{3})"
)
PY_LOG_TS_RE = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}),(?P<msec>\d{3})"
)
BATCH_RE = re.compile(r"Running batch (?P<batch_index>\d+):")
PROCESSOR_RE = re.compile(
    r"Qwen3OmniNextProcessor preprocessing stats: "
    r"audio_items=(?P<audio_items>\d+) \((?P<audio_s>[0-9.]+)s\), "
    r"image_items=(?P<image_items>\d+) \((?P<image_s>[0-9.]+)s\), "
    r"video_items=(?P<video_items>\d+) \((?P<video_s>[0-9.]+)s\), "
    r"total_preprocess=(?P<total_s>[0-9.]+)s, "
    r"replace_multimodal_special_tokens=(?P<replace>skipped|[0-9.]+s)"
)
INPUT_PREPROCESSOR_RE = re.compile(
    r"input_preprocessor finished, rid: (?P<rid>[^,\s]+), cost:(?P<cost_ms>[0-9.]+)"
)
ENCODER_RE = re.compile(
    r"encode all mm inputs done, cost: (?P<cost_ms>[0-9.]+) ms, "
    r"\(request_id, item_cnt\): (?P<items>\{.*\})"
)
RID_BRACKET_RE = re.compile(r"\[(?P<rid>\d+-[^]]+)\]")
TALKER_FEED_FIRST_RE = re.compile(r"TALKER_FEED_FIRST")
TALKER_CODEC_RE = re.compile(r"talker codec #(?P<codec_index>\d+)")
THINKER_DONE_RE = re.compile(r"reason=thinker_generator_finished")
TALKER_EXIT_RE = re.compile(r"reason=talker_generator_exit")
CODE2WAV_DONE_RE = re.compile(r"reason=code2wav_finished")


@dataclass
class RequestEvents:
    first_seen_s: float | None = None
    input_preprocessor_cost_ms: list[float] = field(default_factory=list)
    encoder_cost_ms: list[float] = field(default_factory=list)
    thinker_done_s: float | None = None
    talker_feed_first_s: float | None = None
    codec_times_s: list[float] = field(default_factory=list)
    talker_exit_s: float | None = None
    code2wav_done_s: float | None = None

    def touch(self, timestamp_s: float) -> None:
        if self.first_seen_s is None or timestamp_s < self.first_seen_s:
            self.first_seen_s = timestamp_s


def _timestamp_seconds(line: str) -> float | None:
    match = TS_RE.search(line)
    if match:
        dt = datetime(
            2026,
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            int(match.group("msec")) * 1000,
        )
        return dt.timestamp()
    match = PY_LOG_TS_RE.search(line)
    if match:
        dt = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            int(match.group("msec")) * 1000,
        )
        return dt.timestamp()
    return None


def _rid_from_brackets(line: str) -> str | None:
    match = RID_BRACKET_RE.search(line)
    if not match:
        return None
    return match.group("rid")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "avg": mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


def _fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _parse_item_counts(raw: str) -> list[tuple[str, int]]:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    items: list[tuple[str, int]] = []
    for key, value in parsed.items():
        if isinstance(key, str):
            try:
                items.append((key, int(value)))
            except (TypeError, ValueError):
                continue
    return items


def _nonnegative_ms(end_s: float, start_s: float) -> float:
    return max(0.0, (end_s - start_s) * 1000.0)


def _batch_index_from_rid(rid: str) -> int | None:
    try:
        return int(rid.split("-", 1)[0])
    except (TypeError, ValueError):
        return None


def parse_log(
    path: Path,
) -> tuple[dict[str, RequestEvents], list[dict[str, Any]], dict[int, float]]:
    requests: dict[str, RequestEvents] = {}
    processor_stats: list[dict[str, Any]] = []
    batch_starts_s: dict[int, float] = {}

    with path.open(encoding="utf-8", errors="replace") as fp:
        for line in fp:
            timestamp_s = _timestamp_seconds(line)

            batch_match = BATCH_RE.search(line)
            if batch_match and timestamp_s is not None:
                batch_starts_s[int(batch_match.group("batch_index"))] = timestamp_s

            processor_match = PROCESSOR_RE.search(line)
            if processor_match:
                processor_stats.append(
                    {
                        "timestamp_s": timestamp_s,
                        "audio_items": int(processor_match.group("audio_items")),
                        "audio_ms": float(processor_match.group("audio_s")) * 1000.0,
                        "image_items": int(processor_match.group("image_items")),
                        "image_ms": float(processor_match.group("image_s")) * 1000.0,
                        "video_items": int(processor_match.group("video_items")),
                        "video_ms": float(processor_match.group("video_s")) * 1000.0,
                        "total_preprocess_ms": float(processor_match.group("total_s"))
                        * 1000.0,
                    }
                )

            input_match = INPUT_PREPROCESSOR_RE.search(line)
            if input_match and timestamp_s is not None:
                rid = input_match.group("rid")
                request = requests.setdefault(rid, RequestEvents())
                request.touch(timestamp_s)
                request.input_preprocessor_cost_ms.append(
                    float(input_match.group("cost_ms"))
                )

            encoder_match = ENCODER_RE.search(line)
            if encoder_match and timestamp_s is not None:
                for rid, _item_count in _parse_item_counts(encoder_match.group("items")):
                    request = requests.setdefault(rid, RequestEvents())
                    request.touch(timestamp_s)
                    request.encoder_cost_ms.append(float(encoder_match.group("cost_ms")))

            rid = _rid_from_brackets(line)
            if rid and timestamp_s is not None:
                request = requests.setdefault(rid, RequestEvents())
                request.touch(timestamp_s)
                if THINKER_DONE_RE.search(line):
                    request.thinker_done_s = timestamp_s
                if TALKER_FEED_FIRST_RE.search(line):
                    request.talker_feed_first_s = timestamp_s
                if TALKER_CODEC_RE.search(line):
                    request.codec_times_s.append(timestamp_s)
                if TALKER_EXIT_RE.search(line):
                    request.talker_exit_s = timestamp_s
                if CODE2WAV_DONE_RE.search(line):
                    request.code2wav_done_s = timestamp_s

    return requests, processor_stats, batch_starts_s


def _included_requests(
    requests: dict[str, RequestEvents], skip_first_requests: int
) -> list[tuple[str, RequestEvents]]:
    ordered = sorted(
        (
            (rid, request)
            for rid, request in requests.items()
            if request.first_seen_s is not None
        ),
        key=lambda item: (item[1].first_seen_s or 0, item[0]),
    )
    return ordered[max(skip_first_requests, 0) :]


def summarize(
    path: Path,
    *,
    label: str | None = None,
    skip_first_requests: int = 0,
) -> dict[str, Any]:
    requests, processor_stats, batch_starts_s = parse_log(path)
    included = _included_requests(requests, skip_first_requests)
    included_rids = {rid for rid, _request in included}

    processor_rows = processor_stats[max(skip_first_requests, 0) :]
    input_preprocessor_costs = [
        cost
        for rid, request in included
        for cost in request.input_preprocessor_cost_ms
        if rid in included_rids
    ]
    encoder_costs = [
        cost
        for rid, request in included
        for cost in request.encoder_cost_ms
        if rid in included_rids
    ]

    thinker_to_talker_feed_ms = []
    feed_to_first_codec_ms = []
    codec_avg_gap_ms = []
    codec_active_ms = []
    talker_to_code2wav_drain_ms = []
    code2wav_finish_after_first_codec_ms = []
    codec_counts = []
    batch_first_engine_lag_ms = []
    batch_last_engine_lag_ms = []
    batch_admission_span_ms = []
    batch_request_counts = []

    for _rid, request in included:
        if request.thinker_done_s is not None and request.talker_feed_first_s is not None:
            thinker_to_talker_feed_ms.append(
                _nonnegative_ms(request.talker_feed_first_s, request.thinker_done_s)
            )
        if request.codec_times_s:
            codec_times = sorted(request.codec_times_s)
            codec_counts.append(len(codec_times))
            if request.talker_feed_first_s is not None:
                feed_to_first_codec_ms.append(
                    _nonnegative_ms(codec_times[0], request.talker_feed_first_s)
                )
            if len(codec_times) > 1:
                active_ms = _nonnegative_ms(codec_times[-1], codec_times[0])
                codec_active_ms.append(active_ms)
                codec_avg_gap_ms.append(active_ms / (len(codec_times) - 1))
            if request.code2wav_done_s is not None:
                code2wav_finish_after_first_codec_ms.append(
                    _nonnegative_ms(request.code2wav_done_s, codec_times[0])
                )
        if request.talker_exit_s is not None and request.code2wav_done_s is not None:
            talker_to_code2wav_drain_ms.append(
                _nonnegative_ms(request.code2wav_done_s, request.talker_exit_s)
            )

    requests_by_batch: dict[int, list[float]] = {}
    for rid, request in included:
        if request.first_seen_s is None:
            continue
        batch_index = _batch_index_from_rid(rid)
        if batch_index is None:
            continue
        requests_by_batch.setdefault(batch_index, []).append(request.first_seen_s)

    for batch_index, request_times in requests_by_batch.items():
        batch_start_s = batch_starts_s.get(batch_index)
        if batch_start_s is None:
            continue
        first_seen_s = min(request_times)
        last_seen_s = max(request_times)
        batch_first_engine_lag_ms.append(_nonnegative_ms(first_seen_s, batch_start_s))
        batch_last_engine_lag_ms.append(_nonnegative_ms(last_seen_s, batch_start_s))
        batch_admission_span_ms.append(_nonnegative_ms(last_seen_s, first_seen_s))
        batch_request_counts.append(float(len(request_times)))

    return {
        "label": label or path.parent.name,
        "path": str(path),
        "skip_first_requests": skip_first_requests,
        "total_request_ids": len(requests),
        "included_request_ids": len(included),
        "batch_count": len(batch_starts_s),
        "included_batch_count": len(batch_request_counts),
        "processor_total_ms": _stats(
            [row["total_preprocess_ms"] for row in processor_rows]
        ),
        "processor_video_ms": _stats([row["video_ms"] for row in processor_rows]),
        "processor_audio_ms": _stats([row["audio_ms"] for row in processor_rows]),
        "input_preprocessor_lifecycle_ms": _stats(input_preprocessor_costs),
        "encoder_mm_ms": _stats(encoder_costs),
        "thinker_to_talker_feed_ms": _stats(thinker_to_talker_feed_ms),
        "talker_feed_to_first_codec_ms": _stats(feed_to_first_codec_ms),
        "talker_codec_avg_gap_ms": _stats(codec_avg_gap_ms),
        "talker_codec_active_ms": _stats(codec_active_ms),
        "talker_to_code2wav_drain_ms": _stats(talker_to_code2wav_drain_ms),
        "code2wav_finish_after_first_codec_ms": _stats(
            code2wav_finish_after_first_codec_ms
        ),
        "talker_codec_count": _stats([float(count) for count in codec_counts]),
        "batch_first_engine_lag_ms": _stats(batch_first_engine_lag_ms),
        "batch_last_engine_lag_ms": _stats(batch_last_engine_lag_ms),
        "batch_admission_span_ms": _stats(batch_admission_span_ms),
        "batch_request_count": _stats(batch_request_counts),
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print(
        "| Run | Skip | Request IDs | Processor Total Avg/P95 | "
        "Input Preproc Avg/P95 | Encoder Avg/P95 | Thinker->Talker Avg/P95 | "
        "Feed->Codec Avg/P95 | Codec Gap Avg/P95 | Talker->C2W Drain Avg/P95 |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        processor = row["processor_total_ms"]
        input_preproc = row["input_preprocessor_lifecycle_ms"]
        encoder = row["encoder_mm_ms"]
        thinker_to_talker = row["thinker_to_talker_feed_ms"]
        first_codec = row["talker_feed_to_first_codec_ms"]
        codec_gap = row["talker_codec_avg_gap_ms"]
        drain = row["talker_to_code2wav_drain_ms"]
        print(
            "| {label} | {skip} | {included}/{total} | "
            "{processor_avg}/{processor_p95}ms | "
            "{input_avg}/{input_p95}ms | "
            "{encoder_avg}/{encoder_p95}ms | "
            "{tt_avg}/{tt_p95}ms | "
            "{fc_avg}/{fc_p95}ms | "
            "{gap_avg}/{gap_p95}ms | "
            "{drain_avg}/{drain_p95}ms |".format(
                label=row["label"],
                skip=row["skip_first_requests"],
                included=row["included_request_ids"],
                total=row["total_request_ids"],
                processor_avg=_fmt(processor["avg"]),
                processor_p95=_fmt(processor["p95"]),
                input_avg=_fmt(input_preproc["avg"]),
                input_p95=_fmt(input_preproc["p95"]),
                encoder_avg=_fmt(encoder["avg"]),
                encoder_p95=_fmt(encoder["p95"]),
                tt_avg=_fmt(thinker_to_talker["avg"]),
                tt_p95=_fmt(thinker_to_talker["p95"]),
                fc_avg=_fmt(first_codec["avg"]),
                fc_p95=_fmt(first_codec["p95"]),
                gap_avg=_fmt(codec_gap["avg"]),
                gap_p95=_fmt(codec_gap["p95"]),
                drain_avg=_fmt(drain["avg"]),
                drain_p95=_fmt(drain["p95"]),
            )
        )

    print()
    print("| Run | Included Batches | Req/Batch Avg/P95 | First Engine Lag Avg/P95 | Last Engine Lag Avg/P95 | Batch Admission Span Avg/P95 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        request_count = row["batch_request_count"]
        first_lag = row["batch_first_engine_lag_ms"]
        last_lag = row["batch_last_engine_lag_ms"]
        span = row["batch_admission_span_ms"]
        print(
            "| {label} | {included}/{total} | "
            "{req_avg}/{req_p95} | "
            "{first_avg}/{first_p95}ms | "
            "{last_avg}/{last_p95}ms | "
            "{span_avg}/{span_p95}ms |".format(
                label=row["label"],
                included=row["included_batch_count"],
                total=row["batch_count"],
                req_avg=_fmt(request_count["avg"], 1),
                req_p95=_fmt(request_count["p95"], 1),
                first_avg=_fmt(first_lag["avg"]),
                first_p95=_fmt(first_lag["p95"]),
                last_avg=_fmt(last_lag["avg"]),
                last_p95=_fmt(last_lag["p95"]),
                span_avg=_fmt(span["avg"]),
                span_p95=_fmt(span["p95"]),
            )
        )


def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM-Omni stage timings from Qwen3.5 run logs."
    )
    parser.add_argument("logs", nargs="+", help="vLLM run.log files")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels, one per log.",
    )
    parser.add_argument(
        "--skip-first-requests",
        nargs="*",
        type=int,
        default=None,
        help="Optional warmed-request drop counts, one per log.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in {0, len(args.logs)}:
        parser.error("--labels must be omitted or have the same length as logs")
    if args.skip_first_requests is not None and len(args.skip_first_requests) not in {
        0,
        len(args.logs),
    }:
        parser.error(
            "--skip-first-requests must be omitted or have the same length as logs"
        )

    labels = args.labels or [None] * len(args.logs)
    skip_counts = args.skip_first_requests or [0] * len(args.logs)
    rows = [
        summarize(Path(path), label=label, skip_first_requests=skip_count)
        for path, label, skip_count in zip(args.logs, labels, skip_counts)
    ]
    print_markdown(rows)
    if args.json_output:
        _save_json({"rows": rows}, args.json_output)


if __name__ == "__main__":
    main()
