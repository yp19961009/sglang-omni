#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run concurrent SGLang Qwen3.5-Omni RTC-style sessions.

Each session mirrors the vLLM RTC client shape: pre_run trunk 1..T to warm
prefix/MM caches, then one measured actual_run at trunk T.
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


async def _run_preruns(
    *,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_url: str,
    context: dict[str, Any],
) -> list[float]:
    request_id = context["request_id"]
    media_cache_namespace = context["media_cache_namespace"]
    offsets = context["offsets"]
    pre_run_times: list[float] = []
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
                "request_id": f"__pr__{request_id}_t{trunk}",
                "media_cache_namespace": media_cache_namespace,
                "trunk_size": trunk,
                "pre_run": True,
            },
        }
        _apply_video_request_options(payload, args)
        t0 = time.perf_counter()
        await post_chat(session, api_url=api_url, payload=payload, stream=False)
        pre_run_times.append((time.perf_counter() - t0) * 1000.0)
    return pre_run_times


async def _run_actual(
    *,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_url: str,
    context: dict[str, Any],
    pre_run_times: list[float],
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
        "pre_run_total_ms": sum(pre_run_times),
        "pre_run_avg_ms": _mean(pre_run_times),
        "batch_idx": offsets["batch_idx"],
        "sil_start_idx": offsets["sil_start_idx"],
        "video_start_idx": offsets["video_start_idx"],
        "question_idx": offsets["question_idx"],
    }
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
        pre_run_times = []
    else:
        pre_run_times = await _run_preruns(
            session=session, args=args, api_url=api_url, context=context
        )
        if args.post_prerun_sleep_ms > 0:
            await asyncio.sleep(args.post_prerun_sleep_ms / 1000.0)
    return await _run_actual(
        session=session,
        args=args,
        api_url=api_url,
        context=context,
        pre_run_times=pre_run_times,
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
    lock = asyncio.Lock()

    async def actual_worker(
        *,
        session: aiohttp.ClientSession,
        queue: asyncio.Queue[dict[str, Any] | None],
        pre_run_by_sample: dict[int, list[float]],
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
                    pre_run_times=pre_run_by_sample.get(sample_idx, []),
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
            pre_run_by_sample: dict[int, list[float]] = {}

            pre_t0 = time.perf_counter()
            if args.serialize_prerun:
                for context in contexts:
                    sample_idx = int(context["sample_idx"])
                    pre_run_by_sample[sample_idx] = await _run_preruns(
                        session=session,
                        args=args,
                        api_url=api_url,
                        context=context,
                    )
                    print(
                        f"prerun completed sample={sample_idx} "
                        f"total_ms={sum(pre_run_by_sample[sample_idx]):.3f}",
                        flush=True,
                    )
            else:
                pre_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
                for context in contexts:
                    pre_queue.put_nowait(context)
                for _ in range(args.concurrency):
                    pre_queue.put_nowait(None)

                async def prerun_worker(worker_idx: int) -> None:
                    if worker_idx and args.stagger_ms > 0:
                        await asyncio.sleep(worker_idx * args.stagger_ms / 1000.0)
                    while True:
                        context = await pre_queue.get()
                        if context is None:
                            return
                        sample_idx = int(context["sample_idx"])
                        try:
                            times = await _run_preruns(
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
                                    f"failed prerun sample={sample_idx}: "
                                    f"{err['error']}",
                                    flush=True,
                                )
                            continue
                        async with lock:
                            pre_run_by_sample[sample_idx] = times
                            print(
                                f"prerun completed {len(pre_run_by_sample)}/"
                                f"{args.total_samples} sample={sample_idx} "
                                f"total_ms={sum(times):.3f}",
                                flush=True,
                            )

                await asyncio.gather(
                    *(prerun_worker(i) for i in range(args.concurrency))
                )
            pre_elapsed_s = time.perf_counter() - pre_t0

            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            for context in contexts:
                if int(context["sample_idx"]) not in pre_run_by_sample:
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
                            pre_run_by_sample=pre_run_by_sample,
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
            elapsed_s = pre_elapsed_s + actual_elapsed_s
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

            t0 = time.perf_counter()
            await asyncio.gather(
                *(session_worker(i) for i in range(args.concurrency))
            )
            elapsed_s = time.perf_counter() - t0
            actual_elapsed_s = None

    rows.sort(key=lambda item: int(item["sample_idx"]))
    errors.sort(key=lambda item: int(item["sample_idx"]))
    completed = len(rows)
    failed = len(errors)
    ttft = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    ttfa = [float(row["ttfa_ms"]) for row in rows if row.get("ttfa_ms") is not None]
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

    metrics = {
        "trunk_size": args.trunk_size,
        "concurrency": args.concurrency,
        "total_samples": args.total_samples,
        "completed": completed,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "actual_elapsed_s": actual_elapsed_s,
        "serialize_prerun": bool(args.serialize_prerun),
        "barrier_prerun": bool(args.barrier_prerun),
        "skip_prerun": bool(args.skip_prerun),
        "qps": completed / elapsed_s if elapsed_s > 0 else None,
        "mode": "text" if args.text_only else "text_audio",
        "max_tokens": args.max_tokens,
        "prerun_max_tokens": args.prerun_max_tokens,
        "bang_count": len(bang_sample_indices),
        "bang_sample_indices": bang_sample_indices,
        "ttft_avg_ms": _mean(ttft),
        "ttft_p50_ms": _percentile(ttft, 50),
        "ttft_p95_ms": _percentile(ttft, 95),
        "ttft_p99_ms": _percentile(ttft, 99),
        "ttfa_avg_ms": _mean(ttfa),
        "ttfa_p50_ms": _percentile(ttfa, 50),
        "ttfa_p95_ms": _percentile(ttfa, 95),
        "ttfa_p99_ms": _percentile(ttfa, 99),
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
    parser.add_argument("--prerun-max-tokens", type=int, default=0)
    parser.add_argument("--post-prerun-sleep-ms", type=int, default=0)
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
        action="store_true",
        help=(
            "Skip RTC cache-population pre-run requests and measure one actual "
            "trunk-size request per session. This matches chunkwise realtime "
            "latency runs where only the final streamed request is measured."
        ),
    )
    parser.add_argument(
        "--serialize-prerun",
        action="store_true",
        help=(
            "Run all RTC pre-run cache population requests serially, then launch "
            "the measured actual requests at the requested concurrency."
        ),
    )
    parser.add_argument(
        "--barrier-prerun",
        action="store_true",
        help=(
            "Run RTC pre-run cache population requests concurrently at the requested "
            "concurrency, wait for all of them, then launch measured actual requests."
        ),
    )
    parser.add_argument(
        "--profile-actual-run-id",
        default=None,
        help=(
            "Start request profiling after barrier/serialized pre-run completes, "
            "then stop it after the measured actual requests."
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
        parser.error("--serialize-prerun and --barrier-prerun are mutually exclusive")
    if args.skip_prerun and (args.serialize_prerun or args.barrier_prerun):
        parser.error("--skip-prerun cannot be combined with pre-run modes")
    if args.profile_actual_run_id and not (
        args.serialize_prerun or args.barrier_prerun
    ):
        parser.error(
            "--profile-actual-run-id requires --serialize-prerun or --barrier-prerun"
        )
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
