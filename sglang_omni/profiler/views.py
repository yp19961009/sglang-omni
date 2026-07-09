# SPDX-License-Identifier: Apache-2.0
"""Profiler views: per-request timeline, stage breakdown, hop breakdown.

Reads JSONL events emitted by :mod:`sglang_omni.profiler.event_recorder`.
Input may be a single file, a directory of ``events_*_<pid>.jsonl``, or a
list of either. Framework-free so reports can be regenerated locally.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def iter_events(source: str | Path | Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield every JSON event from a file, directory, or list of either."""
    paths: list[Path] = []
    if isinstance(source, (str, Path)):
        sources: list[str | Path] = [source]
    else:
        sources = list(source)
    for raw in sources:
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("events_*.jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            logger.warning("iter_events: skipping non-existent path %s", p)
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "iter_events: skipping malformed line %d in %s",
                        line_no,
                        path,
                    )


def load_events(source: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Return all events sorted by ``timestamp_ns`` (stable)."""
    events = list(iter_events(source))
    events.sort(key=lambda e: e.get("timestamp_ns", 0))
    return events


# ---------------------------------------------------------------------------
# View 1: per-request timeline
# ---------------------------------------------------------------------------


@dataclass
class RequestTimeline:
    """All events for a single request, sorted by time."""

    request_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def t0_ns(self) -> int | None:
        if not self.events:
            return None
        return self.events[0]["timestamp_ns"]

    @property
    def t_end_ns(self) -> int | None:
        if not self.events:
            return None
        return self.events[-1]["timestamp_ns"]

    @property
    def total_ms(self) -> float:
        if not self.events:
            return 0.0
        t0 = self.t0_ns
        t1 = self.t_end_ns
        assert t0 is not None and t1 is not None
        return (t1 - t0) / 1e6

    def to_relative(self) -> list[dict[str, Any]]:
        """Return events with an added ``t_rel_ms`` field anchored at t0."""
        if not self.events:
            return []
        t0 = self.t0_ns
        assert t0 is not None
        result = []
        for ev in self.events:
            out = dict(ev)
            out["t_rel_ms"] = (ev["timestamp_ns"] - t0) / 1e6
            result.append(out)
        return result


def reconstruct_timelines(
    source: str | Path | Iterable[str | Path],
) -> dict[str, RequestTimeline]:
    """Group every event by ``request_id`` into a per-request timeline."""
    events = load_events(source)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        rid = ev.get("request_id")
        if not rid:
            continue
        grouped[rid].append(ev)
    return {
        rid: RequestTimeline(request_id=rid, events=evts)
        for rid, evts in grouped.items()
    }


# ---------------------------------------------------------------------------
# View 2: stage breakdown
# ---------------------------------------------------------------------------


# (opener, closer) pairs framing a stage-local interval. Opener can
# appear in multiple pairs (e.g. prefill_start closes against both
# first_emit and first_stream_chunk_sent → thinker TTFT / talker TTFCC).
_STAGE_INTERVAL_EVENTS = (
    ("stage_input_received", "stage_complete"),
    ("encoder_start", "encoder_end"),
    ("preprocess_start", "preprocess_end"),
    ("preprocess_cache_lookup_start", "preprocess_cache_lookup_end"),
    ("preprocess_cache_key_start", "preprocess_cache_key_end"),
    ("preprocess_media_load_start", "preprocess_media_load_end"),
    ("preprocess_prompt_start", "preprocess_prompt_end"),
    ("preprocess_hf_processor_start", "preprocess_hf_processor_end"),
    ("preprocess_finalize_start", "preprocess_finalize_end"),
    ("scheduler_request_build_start", "scheduler_request_build_end"),
    ("scheduler_prefill_start", "stage_first_stream_chunk_sent"),
    ("scheduler_prefill_start", "scheduler_first_emit"),
)


@dataclass
class StageInterval:
    """One occurrence of (open_event, close_event) inside a stage."""

    request_id: str
    stage: str
    open_event: str
    close_event: str
    open_ns: int
    close_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.close_ns - self.open_ns) / 1e6


@dataclass
class StageBreakdownRow:
    stage: str
    interval_name: str
    count: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "interval": self.interval_name,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0, 1]. ``values`` is sorted."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * q
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def compute_stage_intervals(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
    interval_events: tuple[tuple[str, str], ...] = _STAGE_INTERVAL_EVENTS,
) -> list[StageInterval]:
    """Pair open/close events into (stage, request_id)-scoped intervals."""
    if timelines is None:
        if source is None:
            raise ValueError("compute_stage_intervals requires timelines or source")
        timelines = reconstruct_timelines(source)

    # Per-pair lookup: one opener event seeds every pair it's in; each
    # closer consumes only its own pair's stack.
    opens_to_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    closes_to_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for opener, closer in interval_events:
        opens_to_pairs[opener].append((opener, closer))
        closes_to_pairs[closer].append((opener, closer))

    out: list[StageInterval] = []
    for rid, tl in timelines.items():
        pending: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for ev in tl.events:
            name = ev.get("event_name")
            stage = ev.get("stage", "unknown")
            ts = int(ev.get("timestamp_ns", 0))
            # An event may be both opener and closer (chained intervals).
            if name in opens_to_pairs:
                for opener, closer in opens_to_pairs[name]:
                    pending[(stage, opener, closer)].append(ts)
            if name in closes_to_pairs:
                for opener, closer in closes_to_pairs[name]:
                    stack = pending.get((stage, opener, closer))
                    if not stack:
                        continue
                    open_ns = stack.pop(0)
                    out.append(
                        StageInterval(
                            request_id=rid,
                            stage=stage,
                            open_event=opener,
                            close_event=closer,
                            open_ns=open_ns,
                            close_ns=ts,
                        )
                    )
    return out


def stage_breakdown(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
) -> list[StageBreakdownRow]:
    """Aggregate interval durations by (stage, open/close pair)."""
    intervals = compute_stage_intervals(timelines, source=source)
    bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for it in intervals:
        key = (it.stage, it.open_event, it.close_event)
        bucket[key].append(it.duration_ms)
    rows: list[StageBreakdownRow] = []
    for (stage, open_ev, close_ev), durations in bucket.items():
        durations.sort()
        rows.append(
            StageBreakdownRow(
                stage=stage,
                interval_name=f"{open_ev}->{close_ev}",
                count=len(durations),
                total_ms=sum(durations),
                avg_ms=sum(durations) / len(durations),
                p50_ms=_percentile(durations, 0.50),
                p95_ms=_percentile(durations, 0.95),
                max_ms=durations[-1],
            )
        )
    rows.sort(key=lambda r: (-r.total_ms, r.stage))
    return rows


# ---------------------------------------------------------------------------
# View 3: hop breakdown (stage_a -> stage_b)
# ---------------------------------------------------------------------------


@dataclass
class HopBreakdownRow:
    src_stage: str
    dst_stage: str
    kind: str  # "payload" or "stream_chunk"
    count: int
    total_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src_stage,
            "dst": self.dst_stage,
            "kind": self.kind,
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def hop_breakdown(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
) -> list[HopBreakdownRow]:
    """Pair `stage_hop_sent` / `stage_input_received` and stream-chunk variants.

    Matching is keyed by ``(request_id, src_stage, dst_stage, chunk_id?)``.
    ``chunk_id`` distinguishes stream-chunk hops; payload hops have none.
    """
    if timelines is None:
        if source is None:
            raise ValueError("hop_breakdown requires timelines or source")
        timelines = reconstruct_timelines(source)

    # Pending sends keyed by (rid, src, dst, kind, chunk_id_or_None) -> list of ts.
    pending: dict[tuple[str, str, str, str, int | None], list[int]] = defaultdict(list)
    durations: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for rid, tl in timelines.items():
        for ev in tl.events:
            name = ev.get("event_name")
            md = ev.get("metadata") or {}
            ts = int(ev.get("timestamp_ns", 0))
            stage = ev.get("stage", "unknown")
            if name == "stage_hop_sent":
                dst = md.get("to_stage", "?")
                pending[(rid, stage, dst, "payload", None)].append(ts)
            elif name == "stage_stream_chunk_sent":
                dst = md.get("to_stage", "?")
                chunk_id = md.get("chunk_id")
                pending[(rid, stage, dst, "stream_chunk", chunk_id)].append(ts)
            elif name == "stage_input_received":
                src = md.get("from_stage")
                if not src or src == "coordinator":
                    continue
                key = (rid, src, stage, "payload", None)
                stack = pending.get(key)
                if stack:
                    open_ns = stack.pop(0)
                    durations[(src, stage, "payload")].append((ts - open_ns) / 1e6)
            elif name == "stage_stream_chunk_received":
                src = md.get("from_stage")
                chunk_id = md.get("chunk_id")
                if not src:
                    continue
                key = (rid, src, stage, "stream_chunk", chunk_id)
                stack = pending.get(key)
                if stack:
                    open_ns = stack.pop(0)
                    durations[(src, stage, "stream_chunk")].append((ts - open_ns) / 1e6)

    rows: list[HopBreakdownRow] = []
    for (src, dst, kind), values in durations.items():
        values.sort()
        rows.append(
            HopBreakdownRow(
                src_stage=src,
                dst_stage=dst,
                kind=kind,
                count=len(values),
                total_ms=sum(values),
                avg_ms=sum(values) / len(values),
                p50_ms=_percentile(values, 0.50),
                p95_ms=_percentile(values, 0.95),
                max_ms=values[-1],
            )
        )
    rows.sort(key=lambda r: (-r.total_ms, r.src_stage, r.dst_stage))
    return rows


# ---------------------------------------------------------------------------
# View 4: first generated token TTFT
# ---------------------------------------------------------------------------


def _first_event(
    events: list[dict[str, Any]],
    *,
    name: str,
    stage: str | None = None,
) -> dict[str, Any] | None:
    for ev in events:
        if ev.get("event_name") != name:
            continue
        if stage is not None and ev.get("stage") != stage:
            continue
        return ev
    return None


def _first_event_before_or_at(
    events: list[dict[str, Any]],
    *,
    name: str,
    stage: str | None = None,
    close_ns: int,
) -> dict[str, Any] | None:
    matched: dict[str, Any] | None = None
    for ev in events:
        if ev.get("event_name") != name:
            continue
        if stage is not None and ev.get("stage") != stage:
            continue
        if int(ev.get("timestamp_ns", 0)) > close_ns:
            break
        matched = ev
    return matched


def first_token_ttft(
    timelines: dict[str, RequestTimeline] | None = None,
    *,
    source: str | Path | Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Return per-request first generated token timings.

    ``scheduler_first_emit`` is emitted by the AR scheduler when the first
    sampled token is observed, before detokenization and before client SSE
    delivery. ``request_to_first_token_ms`` anchors at the first profiler event
    for the request, ``post_media_to_first_token_ms`` starts after media load /
    frame extraction / resize and before the HF processor, while
    ``prefill_to_first_token_ms`` isolates the thinker prefill/first-token
    portion.
    """
    if timelines is None:
        if source is None:
            raise ValueError("first_token_ttft requires timelines or source")
        timelines = reconstruct_timelines(source)

    rows: list[dict[str, Any]] = []
    for rid, tl in timelines.items():
        token_ev = _first_event(tl.events, name="scheduler_first_emit")
        if token_ev is None or tl.t0_ns is None:
            continue

        stage = str(token_ev.get("stage") or "unknown")
        token_ns = int(token_ev.get("timestamp_ns", 0))
        prefill_ev = _first_event_before_or_at(
            tl.events,
            name="scheduler_prefill_start",
            stage=stage,
            close_ns=token_ns,
        )
        prefill_ms = None
        if prefill_ev is not None:
            prefill_ms = (token_ns - int(prefill_ev["timestamp_ns"])) / 1e6

        post_media_ev = _first_event_before_or_at(
            tl.events,
            name="preprocess_hf_processor_start",
            close_ns=token_ns,
        )
        post_media_ms = None
        if post_media_ev is not None:
            post_media_ms = (token_ns - int(post_media_ev["timestamp_ns"])) / 1e6

        rows.append(
            {
                "request_id": rid,
                "stage": stage,
                "request_to_first_token_ms": round((token_ns - tl.t0_ns) / 1e6, 3),
                "post_media_to_first_token_ms": (
                    round(post_media_ms, 3) if post_media_ms is not None else None
                ),
                "prefill_to_first_token_ms": (
                    round(prefill_ms, 3) if prefill_ms is not None else None
                ),
            }
        )
    rows.sort(key=lambda r: r["request_to_first_token_ms"])
    return rows


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_ms": round(sum(ordered) / len(ordered), 3),
        "p50_ms": round(_percentile(ordered, 0.50), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "max_ms": round(ordered[-1], 3),
    }


def first_token_ttft_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    request_values = [
        float(row["request_to_first_token_ms"])
        for row in rows
        if row.get("request_to_first_token_ms") is not None
    ]
    prefill_values = [
        float(row["prefill_to_first_token_ms"])
        for row in rows
        if row.get("prefill_to_first_token_ms") is not None
    ]
    post_media_values = [
        float(row["post_media_to_first_token_ms"])
        for row in rows
        if row.get("post_media_to_first_token_ms") is not None
    ]
    return {
        "request_to_first_token": _metric_summary(request_values),
        "post_media_to_first_token": _metric_summary(post_media_values),
        "prefill_to_first_token": _metric_summary(prefill_values),
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def build_report(source: str | Path | Iterable[str | Path]) -> dict[str, Any]:
    """Return every profiler view as a single dict for JSON serialization."""
    timelines = reconstruct_timelines(source)
    token_ttft_rows = first_token_ttft(timelines)
    return {
        "timelines": {rid: tl.to_relative() for rid, tl in timelines.items()},
        "stage_breakdown": [row.to_dict() for row in stage_breakdown(timelines)],
        "hop_breakdown": [row.to_dict() for row in hop_breakdown(timelines)],
        "first_token_ttft": token_ttft_rows,
        "first_token_ttft_summary": first_token_ttft_summary(token_ttft_rows),
        "request_count": len(timelines),
    }


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Pretty-print a list of dicts as a fixed-width table."""
    if not rows:
        return "(empty)\n"
    widths = {
        c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns
    }
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = "\n".join(
        " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows
    )
    return f"{header}\n{sep}\n{body}\n"
