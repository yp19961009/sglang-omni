#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run concurrent SGLang Qwen3.5-Omni RTC-style sessions.

Each session feeds realtime prefix chunks incrementally, then measures the
final streamed chunk. By default this mirrors the vLLM run_rtc_profile shape:
each worker sends pre-run chunks 1..TRUNK_SIZE with max_tokens=2, then
immediately streams the actual request for the same TRUNK_SIZE. Use
--prefix-max-tokens 0 for pure cache-extension pre-runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import statistics
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.eval.qwen35_omni_sglang_rtc_profile import (
    _apply_video_request_options,
    make_rtc_messages,
    post_chat,
)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct / 100.0))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _last_audio_ms(row: dict[str, Any]) -> float | None:
    ttfa_ms = row.get("ttfa_ms")
    if ttfa_ms is None:
        return None
    return float(ttfa_ms) + sum(float(v) for v in row.get("inter_chunk_ms") or [])


def _audio_tail_ms(row: dict[str, Any]) -> float | None:
    ttfa_ms = row.get("ttfa_ms")
    last_audio_ms = row.get("last_audio_ms")
    if ttfa_ms is None or last_audio_ms is None:
        return None
    return float(last_audio_ms) - float(ttfa_ms)


def _finish_tail_ms(row: dict[str, Any]) -> float | None:
    last_audio_ms = row.get("last_audio_ms")
    e2e_total_ms = row.get("e2e_total_ms")
    if last_audio_ms is None or e2e_total_ms is None:
        return None
    return float(e2e_total_ms) - float(last_audio_ms)


def _chunk_interval_avg_ms(row: dict[str, Any]) -> float | None:
    intervals = [float(v) for v in row.get("inter_chunk_ms") or []]
    return _mean(intervals)


def _audio_tail_rtf(row: dict[str, Any]) -> float | None:
    tail_ms = row.get("audio_tail_ms")
    duration_s = row.get("audio_duration_s")
    if tail_ms is None or duration_s is None or float(duration_s) <= 0:
        return None
    return float(tail_ms) / (float(duration_s) * 1000.0)


def _profile_event_matches(
    event: dict[str, Any],
    *,
    stage: str | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if stage is not None and event.get("stage") != stage:
        return False
    if name is not None and event.get("event_name") != name:
        return False
    if metadata:
        event_metadata = event.get("metadata") or {}
        for key, value in metadata.items():
            if event_metadata.get(key) != value:
                return False
    return True


def _profile_first_ms(
    events: list[dict[str, Any]],
    *,
    stage: str | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> float | None:
    for event in events:
        if _profile_event_matches(event, stage=stage, name=name, metadata=metadata):
            value = event.get("t_rel_ms")
            return float(value) if value is not None else None
    return None


def _profile_last_ms(
    events: list[dict[str, Any]],
    *,
    stage: str | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> float | None:
    for event in reversed(events):
        if _profile_event_matches(event, stage=stage, name=name, metadata=metadata):
            value = event.get("t_rel_ms")
            return float(value) if value is not None else None
    return None


def _profile_interval_ms(
    events: list[dict[str, Any]],
    *,
    stage: str,
    start: str,
    end: str,
    start_metadata: dict[str, Any] | None = None,
    end_metadata: dict[str, Any] | None = None,
) -> float | None:
    start_ms = _profile_first_ms(
        events, stage=stage, name=start, metadata=start_metadata
    )
    end_ms = _profile_first_ms(events, stage=stage, name=end, metadata=end_metadata)
    if start_ms is None or end_ms is None:
        return None
    return max(0.0, end_ms - start_ms)


def _profile_count_events(
    events: list[dict[str, Any]],
    *,
    stage: str | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    return sum(
        1
        for event in events
        if _profile_event_matches(event, stage=stage, name=name, metadata=metadata)
    )


def _profile_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _profile_stat_fields(
    rows: list[dict[str, Any]], key: str, prefix: str
) -> dict[str, float | None]:
    values = _profile_values(rows, key)
    return {
        f"{prefix}_avg_ms": _mean(values),
        f"{prefix}_p50_ms": _percentile(values, 50),
        f"{prefix}_p95_ms": _percentile(values, 95),
        f"{prefix}_p99_ms": _percentile(values, 99),
    }


def _extract_vllm_style_profile_rows(
    profile: dict[str, Any] | None, actual_request_ids: set[str]
) -> list[dict[str, Any]]:
    if not profile:
        return []
    timelines = profile.get("timelines") or {}
    rows: list[dict[str, Any]] = []
    for request_id, raw_events in timelines.items():
        if request_id.startswith(("__prefix__", "__pr__")):
            continue
        events = sorted(raw_events, key=lambda event: float(event.get("t_rel_ms") or 0.0))
        if not events:
            continue

        thinker_first_emit_ms = _profile_first_ms(
            events, stage="thinker", name="scheduler_first_emit"
        )
        thinker_first_stream_sent_ms = _profile_first_ms(
            events,
            stage="thinker",
            name="stage_first_stream_chunk_sent",
            metadata={"to_stage": "decode"},
        )
        first_text_sent_ms = _profile_first_ms(
            events,
            stage="decode",
            name="stage_first_stream_chunk_sent",
            metadata={"modality": "text"},
        )
        first_text_received_ms = _profile_first_ms(
            events,
            stage="coordinator",
            name="stage_stream_chunk_received",
            metadata={"from_stage": "decode", "modality": "text"},
        )
        first_audio_sent_ms = _profile_first_ms(
            events,
            stage="code2wav",
            name="stage_first_stream_chunk_sent",
            metadata={"modality": "audio"},
        )
        first_audio_received_ms = _profile_first_ms(
            events,
            stage="coordinator",
            name="stage_stream_chunk_received",
            metadata={"from_stage": "code2wav", "modality": "audio"},
        )
        first_code2wav_input_ms = _profile_first_ms(
            events,
            stage="code2wav",
            name="stage_stream_chunk_received",
            metadata={"from_stage": "talker_ar"},
        )
        last_audio_received_ms = _profile_last_ms(
            events,
            stage="coordinator",
            name="stage_stream_chunk_received",
            metadata={"from_stage": "code2wav", "modality": "audio"},
        )
        profile_e2e_ms = (
            float(events[-1].get("t_rel_ms")) if events[-1].get("t_rel_ms") is not None else None
        )
        if first_audio_sent_ms is None and first_audio_received_ms is None:
            continue
        code2wav_first_chunk_ms = (
            first_audio_sent_ms - first_code2wav_input_ms
            if first_audio_sent_ms is not None and first_code2wav_input_ms is not None
            else None
        )

        row = {
            "profile_request_id": request_id,
            "profile_start_timestamp_ns": events[0].get("timestamp_ns"),
            "profile_thinker_ttft_ms": thinker_first_emit_ms,
            "profile_thinker_first_stream_sent_ms": thinker_first_stream_sent_ms,
            "profile_e2e_ttfa_ms": first_audio_received_ms or first_audio_sent_ms,
            "profile_first_audio_sent_ms": first_audio_sent_ms,
            "profile_first_audio_received_ms": first_audio_received_ms,
            "profile_first_text_sent_ms": first_text_sent_ms,
            "profile_first_text_received_ms": first_text_received_ms,
            "profile_e2e_total_ms": profile_e2e_ms,
            "profile_last_audio_received_ms": last_audio_received_ms,
            "profile_ttfa_thinker_prefill_ms": thinker_first_emit_ms,
            "profile_hf_preproc_ms": _profile_interval_ms(
                events,
                stage="preprocessing",
                start="preprocess_hf_processor_start",
                end="preprocess_hf_processor_end",
            ),
            "profile_preprocessing_ms": _profile_interval_ms(
                events,
                stage="preprocessing",
                start="stage_input_received",
                end="stage_complete",
            ),
            "profile_image_encoder_ms": _profile_interval_ms(
                events,
                stage="image_encoder",
                start="stage_input_received",
                end="stage_complete",
            ),
            "profile_audio_encoder_ms": _profile_interval_ms(
                events,
                stage="audio_encoder",
                start="stage_input_received",
                end="stage_complete",
            ),
            "profile_mm_aggregate_ms": _profile_interval_ms(
                events,
                stage="mm_aggregate",
                start="stage_input_received",
                end="stage_complete",
            ),
            "profile_thinker_prefill_inner_ms": _profile_interval_ms(
                events,
                stage="thinker",
                start="scheduler_prefill_start",
                end="scheduler_first_emit",
            ),
            "profile_talker_prefill_ms": _profile_interval_ms(
                events,
                stage="talker_ar",
                start="scheduler_prefill_start",
                end="stage_first_stream_chunk_sent",
            ),
            "profile_code2wav_first_chunk_ms": code2wav_first_chunk_ms,
            "profile_code2wav_total_ms": _profile_interval_ms(
                events,
                stage="code2wav",
                start="stage_input_received",
                end="stage_complete",
            ),
            "profile_audio_chunk_count": _profile_count_events(
                events,
                stage="code2wav",
                name="stage_stream_chunk_sent",
                metadata={"modality": "audio"},
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: int(row.get("profile_start_timestamp_ns") or 0))
    return rows


def _vllm_style_profile_metrics(profile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not profile_rows:
        return {"profile_num_requests": 0}
    metrics: dict[str, Any] = {
        "profile_num_requests": len(profile_rows),
        "profile_stats_source": "sglang_request_profiler_vllm_style",
        "profile_actual_filter": "timeline has code2wav audio stream output",
        "profile_ttft_semantics": "request_admission->thinker.scheduler_first_emit",
        "profile_ttfa_semantics": (
            "request_admission->coordinator first audio chunk received from code2wav"
        ),
    }
    metrics.update(
        _profile_stat_fields(
            profile_rows, "profile_thinker_ttft_ms", "profile_ttft"
        )
    )
    metrics.update(
        _profile_stat_fields(profile_rows, "profile_e2e_ttfa_ms", "profile_ttfa")
    )
    metrics.update(
        _profile_stat_fields(
            profile_rows, "profile_e2e_total_ms", "profile_e2e_total"
        )
    )
    metrics.update(
        _profile_stat_fields(
            profile_rows, "profile_ttfa_thinker_prefill_ms", "profile_ttfa_thinker_prefill"
        )
    )
    metrics.update(
        _profile_stat_fields(profile_rows, "profile_hf_preproc_ms", "profile_ttfa_hf_preproc")
    )
    metrics.update(
        _profile_stat_fields(
            profile_rows, "profile_talker_prefill_ms", "profile_ttfa_talker_prefill"
        )
    )
    metrics.update(
        _profile_stat_fields(
            profile_rows, "profile_code2wav_first_chunk_ms", "profile_ttfa_code2wav_first_chunk"
        )
    )
    metrics["profile_audio_chunk_count_avg"] = _mean(
        _profile_values(profile_rows, "profile_audio_chunk_count")
    )
    return metrics


def _compact_error(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


async def _start_request_profile(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    run_id: str,
    event_dir: Path,
) -> None:
    url = base_url.rstrip("/") + "/start_request_profile"
    async with session.post(
        url,
        json={"run_id": run_id, "event_dir": str(event_dir.resolve())},
    ) as response:
        response.raise_for_status()
        print("profile_start", await response.text(), flush=True)


async def _stop_request_profile(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    run_id: str,
) -> None:
    url = base_url.rstrip("/") + "/stop_request_profile"
    async with session.post(url, json={"run_id": run_id}) as response:
        response.raise_for_status()
        print("profile_stop", await response.text(), flush=True)


def _is_pure_bang_text(text: str | None) -> bool:
    stripped = (text or "").strip()
    stripped = stripped.strip('"\'“”')
    stripped = stripped.lstrip("#＃$＄")
    return bool(stripped) and set(stripped) <= {"!", "！"}


def _session_offsets(args: argparse.Namespace, sample_idx: int) -> dict[str, int]:
    return {
        "batch_idx": sample_idx % args.num_batches,
        "sil_start_idx": (args.sil_offset + sample_idx) * args.trunk_size,
        "video_start_idx": (
            (args.sil_offset + sample_idx) * args.trunk_size * args.video_fps
            if not args.audio_only
            else 0
        ),
        "question_idx": args.question_offset + sample_idx,
    }


def _session_context(
    args: argparse.Namespace, out_dir: Path, sample_idx: int
) -> dict[str, Any]:
    request_id = f"sg_c{args.concurrency}_s{sample_idx:03d}_{uuid.uuid4()}"
    sample_dir = out_dir / f"sample_{sample_idx:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sample_idx": sample_idx,
        "request_id": request_id,
        "media_cache_namespace": f"rtc:{request_id}",
        "offsets": _session_offsets(args, sample_idx),
        "sample_dir": sample_dir,
    }


def _apply_talker_request_options(
    payload: dict[str, Any], args: argparse.Namespace
) -> None:
    option_names = (
        "talker_temperature",
        "talker_top_k",
        "talker_top_p",
        "talker_min_p",
        "talker_repetition_penalty",
        "talker_seed",
        "subtalker_temperature",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_min_p",
        "subtalker_repetition_penalty",
        "subtalker_seed",
    )
    for name in option_names:
        value = getattr(args, name, None)
        if value is not None:
            payload[name] = value


async def _run_prefix_extensions(
    *,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_url: str,
    context: dict[str, Any],
) -> list[float]:
    request_id = context["request_id"]
    media_cache_namespace = context["media_cache_namespace"]
    offsets = context["offsets"]
    prefix_times: list[float] = []
    for trunk in range(1, args.trunk_size + 1):
        messages = make_rtc_messages(
            test_dir=Path(args.rtc_test_dir),
            trunk_size=trunk,
            batch_idx=offsets["batch_idx"],
            pre_run=True,
            audio_only=args.audio_only,
            video_fps=args.video_fps,
            max_chunks_per_turn=args.max_chunks_per_turn,
            sil_start_idx=offsets["sil_start_idx"],
            video_start_idx=offsets["video_start_idx"],
            question_idx=offsets["question_idx"],
            visual_mode=args.visual_mode,
        )
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "modalities": ["text"],
            "max_tokens": args.prerun_max_tokens,
            "temperature": args.temperature,
            "stream": False,
            "video_fps": args.video_fps,
            "metadata": {
                "request_id": f"__prefix__{request_id}_t{trunk}",
                "media_cache_namespace": media_cache_namespace,
                "trunk_size": trunk,
                "pre_run": True,
                "realtime_prefix": True,
            },
        }
        _apply_video_request_options(payload, args)
        t0 = time.perf_counter()
        await post_chat(session, api_url=api_url, payload=payload, stream=False)
        prefix_times.append((time.perf_counter() - t0) * 1000.0)
    return prefix_times


async def _run_actual(
    *,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_url: str,
    context: dict[str, Any],
    prefix_times: list[float],
) -> dict[str, Any]:
    request_id = context["request_id"]
    media_cache_namespace = context["media_cache_namespace"]
    offsets = context["offsets"]
    sample_idx = int(context["sample_idx"])
    sample_dir = context["sample_dir"]

    messages = make_rtc_messages(
        test_dir=Path(args.rtc_test_dir),
        trunk_size=args.trunk_size,
        batch_idx=offsets["batch_idx"],
        pre_run=False,
        audio_only=args.audio_only,
        video_fps=args.video_fps,
        max_chunks_per_turn=args.max_chunks_per_turn,
        sil_start_idx=offsets["sil_start_idx"],
        video_start_idx=offsets["video_start_idx"],
        question_idx=offsets["question_idx"],
        visual_mode=args.visual_mode,
    )
    payload = {
        "model": args.model,
        "messages": messages,
        "modalities": ["text"] if args.text_only else ["text", "audio"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "video_fps": args.video_fps,
        "metadata": {
            "request_id": request_id,
            "media_cache_namespace": media_cache_namespace,
            "trunk_size": args.trunk_size,
            "pre_run": False,
        },
    }
    if not args.text_only:
        payload["audio"] = {"format": "wav", "voice": args.voice}
    _apply_video_request_options(payload, args)
    _apply_talker_request_options(payload, args)

    measured = await post_chat(
        session,
        api_url=api_url,
        payload=payload,
        stream=True,
        output_wav=sample_dir / f"{request_id}.wav" if not args.text_only else None,
    )
    row = {
        **asdict(measured),
        "sample_idx": sample_idx,
        "success": True,
        "error": None,
        "prefix_total_ms": sum(prefix_times),
        "prefix_avg_ms": _mean(prefix_times),
        "pre_run_total_ms": sum(prefix_times),
        "pre_run_avg_ms": _mean(prefix_times),
        "batch_idx": offsets["batch_idx"],
        "sil_start_idx": offsets["sil_start_idx"],
        "video_start_idx": offsets["video_start_idx"],
        "question_idx": offsets["question_idx"],
    }
    row["client_first_text_event_ms"] = row.get("first_text_event_ms")
    row["last_audio_ms"] = _last_audio_ms(row)
    row["audio_tail_ms"] = _audio_tail_ms(row)
    row["finish_tail_ms"] = _finish_tail_ms(row)
    row["chunk_interval_avg_ms"] = _chunk_interval_avg_ms(row)
    row["audio_chunk_count"] = (
        len(row.get("inter_chunk_ms") or []) + 1
        if row.get("ttfa_ms") is not None
        else 0
    )
    row["audio_tail_rtf"] = _audio_tail_rtf(row)
    (sample_dir / "result.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return row


async def _run_session(
    *,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_url: str,
    out_dir: Path,
    sample_idx: int,
) -> dict[str, Any]:
    context = _session_context(args, out_dir, sample_idx)
    if args.skip_prerun:
        prefix_times = []
    else:
        prefix_times = await _run_prefix_extensions(
            session=session, args=args, api_url=api_url, context=context
        )
        if args.post_prerun_sleep_ms > 0:
            await asyncio.sleep(args.post_prerun_sleep_ms / 1000.0)
    return await _run_actual(
        session=session,
        args=args,
        api_url=api_url,
        context=context,
        prefix_times=prefix_times,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    api_url = base_url + "/v1/chat/completions"
    profile_actual_run_id = args.profile_actual_run_id
    profile_actual_event_dir = (
        Path(args.profile_actual_event_dir)
        if args.profile_actual_event_dir
        else out_dir / "events"
    )
    profile_actual_json = (
        Path(args.profile_actual_json)
        if args.profile_actual_json
        else (
            out_dir / f"request_profile_{profile_actual_run_id}.json"
            if profile_actual_run_id
            else None
        )
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    profile_json: dict[str, Any] | None = None
    profile_rows: list[dict[str, Any]] = []
    profile_metrics: dict[str, Any] = {"profile_num_requests": 0}
    lock = asyncio.Lock()

    async def actual_worker(
        *,
        session: aiohttp.ClientSession,
        queue: asyncio.Queue[dict[str, Any] | None],
        prefix_by_sample: dict[int, list[float]],
        worker_idx: int,
    ) -> None:
        if worker_idx and args.stagger_ms > 0:
            await asyncio.sleep(worker_idx * args.stagger_ms / 1000.0)
        while True:
            context = await queue.get()
            if context is None:
                return
            sample_idx = int(context["sample_idx"])
            try:
                row = await _run_actual(
                    session=session,
                    args=args,
                    api_url=api_url,
                    context=context,
                    prefix_times=prefix_by_sample.get(sample_idx, []),
                )
                async with lock:
                    rows.append(row)
                    print(
                        f"completed {len(rows)}/{args.total_samples} "
                        f"sample={sample_idx}",
                        flush=True,
                    )
            except Exception as exc:
                err = {
                    "sample_idx": sample_idx,
                    "success": False,
                    "error": _compact_error(exc),
                }
                async with lock:
                    errors.append(err)
                    print(f"failed sample={sample_idx}: {err['error']}", flush=True)

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.serialize_prerun or args.barrier_prerun:
            contexts = [
                _session_context(args, out_dir, sample_idx)
                for sample_idx in range(args.total_samples)
            ]
            prefix_by_sample: dict[int, list[float]] = {}

            prefix_t0 = time.perf_counter()
            if args.serialize_prerun:
                for context in contexts:
                    sample_idx = int(context["sample_idx"])
                    prefix_by_sample[sample_idx] = await _run_prefix_extensions(
                        session=session,
                        args=args,
                        api_url=api_url,
                        context=context,
                    )
                    print(
                        f"prefix completed sample={sample_idx} "
                        f"total_ms={sum(prefix_by_sample[sample_idx]):.3f}",
                        flush=True,
                    )
            else:
                pre_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
                for context in contexts:
                    pre_queue.put_nowait(context)
                for _ in range(args.concurrency):
                    pre_queue.put_nowait(None)

                async def prefix_worker(worker_idx: int) -> None:
                    if worker_idx and args.stagger_ms > 0:
                        await asyncio.sleep(worker_idx * args.stagger_ms / 1000.0)
                    while True:
                        context = await pre_queue.get()
                        if context is None:
                            return
                        sample_idx = int(context["sample_idx"])
                        try:
                            times = await _run_prefix_extensions(
                                session=session,
                                args=args,
                                api_url=api_url,
                                context=context,
                            )
                        except Exception as exc:
                            err = {
                                "sample_idx": sample_idx,
                                "success": False,
                                "error": _compact_error(exc),
                            }
                            async with lock:
                                errors.append(err)
                                print(
                                    f"failed prefix sample={sample_idx}: "
                                    f"{err['error']}",
                                    flush=True,
                                )
                            continue
                        async with lock:
                            prefix_by_sample[sample_idx] = times
                            print(
                                f"prefix completed {len(prefix_by_sample)}/"
                                f"{args.total_samples} sample={sample_idx} "
                                f"total_ms={sum(times):.3f}",
                                flush=True,
                            )

                await asyncio.gather(
                    *(prefix_worker(i) for i in range(args.concurrency))
                )
            prefix_elapsed_s = time.perf_counter() - prefix_t0

            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            for context in contexts:
                if int(context["sample_idx"]) not in prefix_by_sample:
                    continue
                queue.put_nowait(context)
            for _ in range(args.concurrency):
                queue.put_nowait(None)

            profile_started = False
            try:
                if profile_actual_run_id:
                    await _start_request_profile(
                        session,
                        base_url=base_url,
                        run_id=profile_actual_run_id,
                        event_dir=profile_actual_event_dir,
                    )
                    profile_started = True

                t0 = time.perf_counter()
                if args.post_prerun_sleep_ms > 0:
                    await asyncio.sleep(args.post_prerun_sleep_ms / 1000.0)
                await asyncio.gather(
                    *(
                        actual_worker(
                            session=session,
                            queue=queue,
                            prefix_by_sample=prefix_by_sample,
                            worker_idx=i,
                        )
                        for i in range(args.concurrency)
                    )
                )
                actual_elapsed_s = time.perf_counter() - t0
            finally:
                if profile_started:
                    await _stop_request_profile(
                        session,
                        base_url=base_url,
                        run_id=profile_actual_run_id,
                    )
            elapsed_s = prefix_elapsed_s + actual_elapsed_s
        else:
            queue: asyncio.Queue[int | None] = asyncio.Queue()
            for sample_idx in range(args.total_samples):
                queue.put_nowait(sample_idx)
            for _ in range(args.concurrency):
                queue.put_nowait(None)

            async def session_worker(worker_idx: int) -> None:
                if worker_idx and args.stagger_ms > 0:
                    await asyncio.sleep(worker_idx * args.stagger_ms / 1000.0)
                while True:
                    sample_idx = await queue.get()
                    if sample_idx is None:
                        return
                    try:
                        row = await _run_session(
                            session=session,
                            args=args,
                            api_url=api_url,
                            out_dir=out_dir,
                            sample_idx=sample_idx,
                        )
                        async with lock:
                            rows.append(row)
                            print(
                                f"completed {len(rows)}/{args.total_samples} "
                                f"sample={sample_idx}",
                                flush=True,
                            )
                    except Exception as exc:
                        err = {
                            "sample_idx": sample_idx,
                            "success": False,
                            "error": _compact_error(exc),
                        }
                        async with lock:
                            errors.append(err)
                            print(
                                f"failed sample={sample_idx}: {err['error']}",
                                flush=True,
                            )

            profile_started = False
            try:
                if profile_actual_run_id:
                    await _start_request_profile(
                        session,
                        base_url=base_url,
                        run_id=profile_actual_run_id,
                        event_dir=profile_actual_event_dir,
                    )
                    profile_started = True

                t0 = time.perf_counter()
                await asyncio.gather(
                    *(session_worker(i) for i in range(args.concurrency))
                )
                elapsed_s = time.perf_counter() - t0
                actual_elapsed_s = None
            finally:
                if profile_started:
                    await _stop_request_profile(
                        session,
                        base_url=base_url,
                        run_id=profile_actual_run_id,
                    )

    rows.sort(key=lambda item: int(item["sample_idx"]))
    errors.sort(key=lambda item: int(item["sample_idx"]))
    completed = len(rows)
    failed = len(errors)
    ttfa = [float(row["ttfa_ms"]) for row in rows if row.get("ttfa_ms") is not None]
    first_output = [
        float(row["first_output_ms"])
        for row in rows
        if row.get("first_output_ms") is not None
    ]
    client_first_text_event = [
        float(row["first_text_event_ms"])
        for row in rows
        if row.get("first_text_event_ms") is not None
    ]
    first_audio_event = [
        float(row["first_audio_event_ms"])
        for row in rows
        if row.get("first_audio_event_ms") is not None
    ]
    text_audio_event_gap = [
        float(row["text_audio_event_gap_ms"])
        for row in rows
        if row.get("text_audio_event_gap_ms") is not None
    ]
    audio_before_text_sample_indices = [
        int(row["sample_idx"])
        for row in rows
        if (row.get("text_audio_event_gap_ms") or 0) < 0
    ]
    last_audio = [
        float(row["last_audio_ms"])
        for row in rows
        if row.get("last_audio_ms") is not None
    ]
    audio_tail = [
        float(row["audio_tail_ms"])
        for row in rows
        if row.get("audio_tail_ms") is not None
    ]
    finish_tail = [
        float(row["finish_tail_ms"])
        for row in rows
        if row.get("finish_tail_ms") is not None
    ]
    chunk_interval_avg = [
        float(row["chunk_interval_avg_ms"])
        for row in rows
        if row.get("chunk_interval_avg_ms") is not None
    ]
    chunk_intervals = [
        float(value)
        for row in rows
        for value in (row.get("inter_chunk_ms") or [])
    ]
    audio_chunk_count = [
        float(row["audio_chunk_count"])
        for row in rows
        if row.get("audio_chunk_count") is not None
    ]
    audio_tail_rtf = [
        float(row["audio_tail_rtf"])
        for row in rows
        if row.get("audio_tail_rtf") is not None
    ]
    e2e = [float(row["e2e_total_ms"]) for row in rows]
    audio_dur = [
        float(row["audio_duration_s"])
        for row in rows
        if row.get("audio_duration_s") is not None
    ]
    bang_sample_indices = [
        int(row["sample_idx"]) for row in rows if _is_pure_bang_text(row.get("text"))
    ]
    if profile_actual_run_id and profile_actual_json is not None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang_omni.profiler",
                str(profile_actual_event_dir.resolve()),
                "--format",
                "json",
                "--out",
                str(profile_actual_json.resolve()),
            ],
            check=True,
        )
        profile_json = json.loads(profile_actual_json.read_text(encoding="utf-8"))
        profile_rows = _extract_vllm_style_profile_rows(profile_json, set())
        profile_by_request_id = {
            str(row["profile_request_id"]): row for row in profile_rows
        }
        matched_profile_rows = 0
        for row in rows:
            row.setdefault("client_first_text_event_ms", row.get("first_text_event_ms"))
            profile_row = profile_by_request_id.get(str(row.get("request_id")))
            if profile_row:
                row.update(profile_row)
                matched_profile_rows += 1
        if matched_profile_rows == 0 and len(profile_rows) == len(rows):
            for row, profile_row in zip(rows, profile_rows):
                row.setdefault("client_first_text_event_ms", row.get("first_text_event_ms"))
                row.update(profile_row)
        profile_metrics = _vllm_style_profile_metrics(profile_rows)

    profile_ttft = _profile_values(rows, "profile_thinker_ttft_ms")
    if profile_ttft:
        ttft = profile_ttft
        ttft_semantics = "first token: request_admission->thinker.scheduler_first_emit"
        for row in rows:
            if row.get("profile_thinker_ttft_ms") is not None:
                row.setdefault("client_first_text_event_ms", row.get("first_text_event_ms"))
                row["ttft_ms"] = row["profile_thinker_ttft_ms"]
                row["ttft_semantics"] = ttft_semantics
    else:
        ttft = client_first_text_event
        ttft_semantics = "first streamed text event (fallback; request profile unavailable)"
        for row in rows:
            row.setdefault("client_first_text_event_ms", row.get("first_text_event_ms"))
            row["ttft_semantics"] = ttft_semantics

    first_text_event = client_first_text_event

    metrics = {
        "trunk_size": args.trunk_size,
        "concurrency": args.concurrency,
        "total_samples": args.total_samples,
        "completed": completed,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "actual_elapsed_s": actual_elapsed_s,
        "serialize_prefix": bool(args.serialize_prerun),
        "barrier_prefix": bool(args.barrier_prerun),
        "skip_prefix": bool(args.skip_prerun),
        "serialize_prerun": bool(args.serialize_prerun),
        "barrier_prerun": bool(args.barrier_prerun),
        "skip_prerun": bool(args.skip_prerun),
        "realtime_prefix_trunk_size": args.trunk_size,
        "qps": completed / elapsed_s if elapsed_s > 0 else None,
        "concurrency_shape": (
            "serialized_prefix"
            if args.serialize_prerun
            else "barrier_prefix"
            if args.barrier_prerun
            else "vllm_pipeline"
        ),
        "mode": "text" if args.text_only else "text_audio",
        "max_tokens": args.max_tokens,
        "prefix_max_tokens": args.prerun_max_tokens,
        "prerun_max_tokens": args.prerun_max_tokens,
        "bang_count": len(bang_sample_indices),
        "bang_sample_indices": bang_sample_indices,
        "ttft_semantics": ttft_semantics,
        "ttfa_semantics": "first streamed audio event",
        "ttft_avg_ms": _mean(ttft),
        "ttft_p50_ms": _percentile(ttft, 50),
        "ttft_p95_ms": _percentile(ttft, 95),
        "ttft_p99_ms": _percentile(ttft, 99),
        "ttfa_avg_ms": _mean(ttfa),
        "ttfa_p50_ms": _percentile(ttfa, 50),
        "ttfa_p95_ms": _percentile(ttfa, 95),
        "ttfa_p99_ms": _percentile(ttfa, 99),
        "first_output_avg_ms": _mean(first_output),
        "first_output_p50_ms": _percentile(first_output, 50),
        "first_output_p95_ms": _percentile(first_output, 95),
        "first_output_p99_ms": _percentile(first_output, 99),
        "first_text_event_avg_ms": _mean(first_text_event),
        "first_text_event_p50_ms": _percentile(first_text_event, 50),
        "first_text_event_p95_ms": _percentile(first_text_event, 95),
        "first_text_event_p99_ms": _percentile(first_text_event, 99),
        "client_first_text_event_avg_ms": _mean(client_first_text_event),
        "client_first_text_event_p50_ms": _percentile(client_first_text_event, 50),
        "client_first_text_event_p95_ms": _percentile(client_first_text_event, 95),
        "client_first_text_event_p99_ms": _percentile(client_first_text_event, 99),
        "first_audio_event_avg_ms": _mean(first_audio_event),
        "first_audio_event_p50_ms": _percentile(first_audio_event, 50),
        "first_audio_event_p95_ms": _percentile(first_audio_event, 95),
        "first_audio_event_p99_ms": _percentile(first_audio_event, 99),
        "text_audio_event_gap_avg_ms": _mean(text_audio_event_gap),
        "text_audio_event_gap_p50_ms": _percentile(text_audio_event_gap, 50),
        "text_audio_event_gap_p95_ms": _percentile(text_audio_event_gap, 95),
        "text_audio_event_gap_p99_ms": _percentile(text_audio_event_gap, 99),
        "audio_before_text_event_count": len(audio_before_text_sample_indices),
        "audio_before_text_sample_indices": audio_before_text_sample_indices,
        "first_output_type_counts": {
            name: sum(1 for row in rows if row.get("first_output_type") == name)
            for name in ("text", "audio")
        },
        "last_audio_avg_ms": _mean(last_audio),
        "last_audio_p50_ms": _percentile(last_audio, 50),
        "last_audio_p95_ms": _percentile(last_audio, 95),
        "last_audio_p99_ms": _percentile(last_audio, 99),
        "audio_tail_avg_ms": _mean(audio_tail),
        "audio_tail_p50_ms": _percentile(audio_tail, 50),
        "audio_tail_p95_ms": _percentile(audio_tail, 95),
        "audio_tail_p99_ms": _percentile(audio_tail, 99),
        "audio_tail_rtf_avg": _mean(audio_tail_rtf),
        "audio_tail_rtf_p95": _percentile(audio_tail_rtf, 95),
        "finish_tail_avg_ms": _mean(finish_tail),
        "finish_tail_p50_ms": _percentile(finish_tail, 50),
        "finish_tail_p95_ms": _percentile(finish_tail, 95),
        "finish_tail_p99_ms": _percentile(finish_tail, 99),
        "chunk_interval_avg_per_req_ms": _mean(chunk_interval_avg),
        "chunk_interval_all_avg_ms": _mean(chunk_intervals),
        "chunk_interval_all_p50_ms": _percentile(chunk_intervals, 50),
        "chunk_interval_all_p95_ms": _percentile(chunk_intervals, 95),
        "chunk_interval_all_p99_ms": _percentile(chunk_intervals, 99),
        "audio_chunk_count_avg": _mean(audio_chunk_count),
        "e2e_avg_ms": _mean(e2e),
        "e2e_p50_ms": _percentile(e2e, 50),
        "e2e_p95_ms": _percentile(e2e, 95),
        "e2e_p99_ms": _percentile(e2e, 99),
        "audio_duration_avg_s": _mean(audio_dur),
        "prompt_tokens_avg": _mean([float(row["prompt_tokens"]) for row in rows]),
        "completion_tokens_avg": _mean(
            [float(row["completion_tokens"]) for row in rows]
        ),
        "profile_actual_run_id": profile_actual_run_id,
        "profile_actual_event_dir": (
            str(profile_actual_event_dir.resolve()) if profile_actual_run_id else None
        ),
        "profile_actual_json": (
            str(profile_actual_json.resolve())
            if profile_actual_run_id and profile_actual_json is not None
            else None
        ),
        "errors": errors,
    }
    metrics.update(profile_metrics)
    for row in rows:
        try:
            sample_idx = int(row["sample_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        result_path = out_dir / f"sample_{sample_idx:02d}" / "result.json"
        if result_path.exists():
            result_path.write_text(
                json.dumps(row, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "per_request.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        (out_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8161")
    parser.add_argument("--model", default="qwen3_5-omni")
    parser.add_argument("--rtc-test-dir", default="/myapp/data/share-data-6batch")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trunk-size", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--total-samples", type=int, default=12)
    parser.add_argument("--stagger-ms", type=int, default=300)
    parser.add_argument("--num-batches", type=int, default=6)
    parser.add_argument("--sil-offset", type=int, default=0)
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument(
        "--visual-mode",
        choices=("image_frames", "video_frames"),
        default="video_frames",
    )
    parser.add_argument("--video-max-frames", type=int, default=0)
    parser.add_argument("--video-min-pixels", type=int, default=0)
    parser.add_argument("--video-max-pixels", type=int, default=0)
    parser.add_argument("--video-total-pixels", type=int, default=0)
    parser.add_argument(
        "--video-override-max-pixels",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-chunks-per-turn", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prerun-max-tokens",
        "--prefix-max-tokens",
        dest="prerun_max_tokens",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--post-prerun-sleep-ms",
        "--post-prefix-sleep-ms",
        dest="post_prerun_sleep_ms",
        type=int,
        default=0,
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--talker-temperature", type=float, default=None)
    parser.add_argument("--talker-top-k", type=int, default=None)
    parser.add_argument("--talker-top-p", type=float, default=None)
    parser.add_argument("--talker-min-p", type=float, default=None)
    parser.add_argument("--talker-repetition-penalty", type=float, default=None)
    parser.add_argument("--talker-seed", type=int, default=None)
    parser.add_argument("--subtalker-temperature", type=float, default=None)
    parser.add_argument("--subtalker-top-k", type=int, default=None)
    parser.add_argument("--subtalker-top-p", type=float, default=None)
    parser.add_argument("--subtalker-min-p", type=float, default=None)
    parser.add_argument("--subtalker-repetition-penalty", type=float, default=None)
    parser.add_argument("--subtalker-seed", type=int, default=None)
    parser.add_argument("--voice", default="f245")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--skip-prerun",
        "--skip-prefix",
        dest="skip_prerun",
        action="store_true",
        help=(
            "Skip RTC prefix-extension requests and measure one full trunk-size "
            "request per session."
        ),
    )
    parser.add_argument(
        "--serialize-prerun",
        "--serialize-prefix",
        dest="serialize_prerun",
        action="store_true",
        help=(
            "Run all RTC prefix-extension requests serially, then launch the "
            "measured final chunk requests at the requested concurrency."
        ),
    )
    parser.add_argument(
        "--barrier-prerun",
        "--barrier-prefix",
        dest="barrier_prerun",
        action="store_true",
        help=(
            "Run RTC prefix-extension requests concurrently at the requested "
            "concurrency, wait for all of them, then launch measured final chunk "
            "requests."
        ),
    )
    parser.add_argument(
        "--profile-actual-run-id",
        default=None,
        help=(
            "Enable request profiling and write vLLM-style profile metrics. "
            "In pipeline mode this profiles prefix and actual requests, then "
            "filters prefix request IDs during post-processing."
        ),
    )
    parser.add_argument(
        "--profile-actual-event-dir",
        default=None,
        help="Event directory for --profile-actual-run-id; defaults to OUTPUT_DIR/events.",
    )
    parser.add_argument(
        "--profile-actual-json",
        default=None,
        help=(
            "Profiler JSON output path for --profile-actual-run-id; defaults to "
            "OUTPUT_DIR/request_profile_RUN_ID.json."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    if args.serialize_prerun and args.barrier_prerun:
        parser.error("--serialize-prefix and --barrier-prefix are mutually exclusive")
    if args.skip_prerun and (args.serialize_prerun or args.barrier_prerun):
        parser.error("--skip-prefix cannot be combined with prefix modes")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
