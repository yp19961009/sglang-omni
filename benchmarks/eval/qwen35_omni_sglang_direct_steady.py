#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct-full steady-state RTC benchmark for SGLang Qwen3.5-Omni.

This client intentionally does not send incremental pre-runs.  Each request
contains the full trunk in one OpenAI-compatible streaming chat request:
``trunk_size - 1`` history chunks plus the final question chunk.

Workers keep a closed-loop concurrency level: whenever one request finishes
before the active window ends, the worker immediately admits another request.
The media offsets advance monotonically so requests use different audio/video
paths and avoid path-keyed media cache hits within a run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval.qwen35_omni_sglang_rtc_profile import (
    _apply_talker_request_options,
    _apply_video_request_options,
    make_rtc_messages,
    post_chat,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x.strip()]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct / 100.0))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _is_bang_text(text: str | None) -> bool:
    stripped = (text or "").strip().strip('"\'')
    stripped = stripped.lstrip("#$")
    return bool(stripped) and set(stripped) <= {"!"}


def _last_audio_ms(row: dict[str, Any]) -> float | None:
    ttfa = row.get("ttfa_ms")
    if ttfa is None:
        return None
    return float(ttfa) + sum(float(x) for x in row.get("inter_chunk_ms") or [])


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    window_s: float,
    total_wall_s: float,
    completed_in_window: int,
) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    ttft = values("ttft_ms")
    ttfa = values("ttfa_ms")
    e2e = values("e2e_total_ms")
    first_output = values("first_output_ms")
    first_text = values("first_text_event_ms")
    first_audio = values("first_audio_event_ms")
    audio_duration = values("audio_duration_s")
    prompt_tokens = values("prompt_tokens")
    completion_tokens = values("completion_tokens")
    last_audio = [_last_audio_ms(r) for r in rows]
    last_audio = [float(x) for x in last_audio if x is not None]
    audio_chunks = values("audio_chunk_count")
    chunk_intervals = [
        float(x) for r in rows for x in (r.get("inter_chunk_ms") or [])
    ]
    return {
        "requests": len(rows),
        "completed_in_window": completed_in_window,
        "window_s": window_s,
        "total_wall_s": total_wall_s,
        "admission_qps": len(rows) / window_s if window_s > 0 else None,
        "completion_qps": completed_in_window / window_s if window_s > 0 else None,
        "wall_qps": len(rows) / total_wall_s if total_wall_s > 0 else None,
        "ttft_avg_ms": _mean(ttft),
        "ttft_p50_ms": _percentile(ttft, 50),
        "ttft_p95_ms": _percentile(ttft, 95),
        "ttft_p99_ms": _percentile(ttft, 99),
        "ttfa_avg_ms": _mean(ttfa),
        "ttfa_p50_ms": _percentile(ttfa, 50),
        "ttfa_p95_ms": _percentile(ttfa, 95),
        "ttfa_p99_ms": _percentile(ttfa, 99),
        "first_output_avg_ms": _mean(first_output),
        "first_text_avg_ms": _mean(first_text),
        "first_audio_avg_ms": _mean(first_audio),
        "last_audio_avg_ms": _mean(last_audio),
        "last_audio_p99_ms": _percentile(last_audio, 99),
        "e2e_avg_ms": _mean(e2e),
        "e2e_p50_ms": _percentile(e2e, 50),
        "e2e_p95_ms": _percentile(e2e, 95),
        "e2e_p99_ms": _percentile(e2e, 99),
        "audio_duration_avg_s": _mean(audio_duration),
        "audio_chunk_count_avg": _mean(audio_chunks),
        "chunk_interval_avg_ms": _mean(chunk_intervals),
        "chunk_interval_p99_ms": _percentile(chunk_intervals, 99),
        "prompt_tokens_avg": _mean(prompt_tokens),
        "completion_tokens_avg": _mean(completion_tokens),
        "bang_count": sum(1 for r in rows if _is_bang_text(r.get("text"))),
        "bang_sample_indices": [
            int(r["sample_idx"]) for r in rows if _is_bang_text(r.get("text"))
        ],
    }


class SequenceAllocator:
    def __init__(
        self,
        *,
        sequence_offset: int,
        num_batches: int,
        max_sequences_per_batch: int,
    ) -> None:
        self.sequence_offset = sequence_offset
        self.num_batches = num_batches
        self.max_sequences_per_batch = max_sequences_per_batch
        self.next_ordinal = 0
        self.lock = asyncio.Lock()

    async def next(self) -> tuple[int, int, int] | None:
        async with self.lock:
            ordinal = self.next_ordinal
            self.next_ordinal += 1
        global_slot = self.sequence_offset + ordinal
        batch_idx = global_slot % self.num_batches
        seq_in_batch = global_slot // self.num_batches
        if seq_in_batch >= self.max_sequences_per_batch:
            return None
        return ordinal, batch_idx, seq_in_batch


async def _run_one_request(
    *,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_url: str,
    out_dir: Path,
    sample_idx: int,
    batch_idx: int,
    seq_in_batch: int,
    trunk_size: int,
    concurrency: int,
    run_t0: float,
    capture: bool,
) -> dict[str, Any]:
    request_id = f"sg_direct_c{concurrency}_ck{trunk_size}_s{sample_idx:05d}_{uuid.uuid4()}"
    media_cache_namespace = f"direct-steady:{request_id}"
    sil_start_idx = seq_in_batch * trunk_size
    video_start_idx = seq_in_batch * trunk_size * args.video_fps
    question_idx = seq_in_batch
    messages = make_rtc_messages(
        test_dir=Path(args.rtc_test_dir),
        trunk_size=trunk_size,
        batch_idx=batch_idx,
        pre_run=False,
        audio_only=args.audio_only,
        video_fps=args.video_fps,
        max_chunks_per_turn=args.max_chunks_per_turn,
        sil_start_idx=sil_start_idx,
        video_start_idx=0 if args.audio_only else video_start_idx,
        question_idx=question_idx,
        visual_mode=args.visual_mode,
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "modalities": ["text"] if args.text_only else ["text", "audio"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "stream": True,
        "video_fps": args.video_fps,
        "metadata": {
            "request_id": request_id,
            "media_cache_namespace": media_cache_namespace,
            "trunk_size": trunk_size,
            "pre_run": False,
            "direct_full": True,
        },
    }
    if not args.text_only:
        payload["audio"] = {"format": "wav", "voice": args.voice}
    _apply_video_request_options(payload, args)
    _apply_talker_request_options(payload, args)

    sample_dir = out_dir / f"sample_{sample_idx:05d}"
    wav_path = sample_dir / f"{request_id}.wav" if capture and not args.text_only else None
    start_rel_s = time.perf_counter() - run_t0
    measured = await post_chat(
        session,
        api_url=api_url,
        payload=payload,
        stream=True,
        output_wav=wav_path,
    )
    end_rel_s = time.perf_counter() - run_t0
    row = {
        **asdict(measured),
        "sample_idx": sample_idx,
        "success": True,
        "error": None,
        "start_rel_s": start_rel_s,
        "end_rel_s": end_rel_s,
        "batch_idx": batch_idx,
        "seq_in_batch": seq_in_batch,
        "sil_start_idx": sil_start_idx,
        "video_start_idx": 0 if args.audio_only else video_start_idx,
        "question_idx": question_idx,
        "capture": capture,
    }
    if capture:
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return row


async def run_combo(
    *,
    args: argparse.Namespace,
    trunk_size: int,
    concurrency: int,
    combo_index: int,
    root_out: Path,
) -> dict[str, Any]:
    out_dir = root_out / f"chunk_{trunk_size}_c{concurrency}"
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = args.base_url.rstrip("/") + "/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    allocator = SequenceAllocator(
        sequence_offset=args.sequence_offset + combo_index * args.request_offset_stride,
        num_batches=args.num_batches,
        max_sequences_per_batch=args.max_sequences_per_batch,
    )
    run_t0 = time.perf_counter()
    admit_until = run_t0 + args.duration_s
    measure_from_s = min(args.warmup_s, args.duration_s)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    capture_count = 0
    lock = asyncio.Lock()

    async def worker(worker_idx: int) -> None:
        nonlocal capture_count
        if worker_idx and args.stagger_ms > 0:
            await asyncio.sleep(worker_idx * args.stagger_ms / 1000.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while time.perf_counter() < admit_until:
                item = await allocator.next()
                if item is None:
                    async with lock:
                        errors.append({
                            "worker_idx": worker_idx,
                            "error": "media sequence pool exhausted",
                        })
                    return
                sample_idx, batch_idx, seq_in_batch = item
                async with lock:
                    capture = capture_count < args.capture_limit
                    if capture:
                        capture_count += 1
                try:
                    row = await _run_one_request(
                        session=session,
                        args=args,
                        api_url=api_url,
                        out_dir=out_dir,
                        sample_idx=sample_idx,
                        batch_idx=batch_idx,
                        seq_in_batch=seq_in_batch,
                        trunk_size=trunk_size,
                        concurrency=concurrency,
                        run_t0=run_t0,
                        capture=capture,
                    )
                    async with lock:
                        rows.append(row)
                        if len(rows) % max(1, concurrency) == 0:
                            print(
                                f"chunk={trunk_size} c={concurrency} "
                                f"completed={len(rows)} t={time.perf_counter() - run_t0:.1f}s",
                                flush=True,
                            )
                except Exception as exc:  # noqa: BLE001 - benchmark records failures.
                    async with lock:
                        errors.append({
                            "worker_idx": worker_idx,
                            "sample_idx": sample_idx,
                            "batch_idx": batch_idx,
                            "seq_in_batch": seq_in_batch,
                            "error": f"{type(exc).__name__}: {exc}",
                            "start_rel_s": time.perf_counter() - run_t0,
                        })

    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    total_wall_s = time.perf_counter() - run_t0
    rows.sort(key=lambda r: int(r["sample_idx"]))
    errors.sort(key=lambda r: int(r.get("sample_idx", -1)))

    steady_rows = [
        r for r in rows
        if float(r.get("start_rel_s") or 0.0) >= measure_from_s
        and float(r.get("start_rel_s") or 0.0) < args.duration_s
    ]
    if not steady_rows:
        steady_rows = rows
        measure_from_s = 0.0
    measurement_window_s = max(0.001, args.duration_s - measure_from_s)
    completed_in_window = sum(
        1
        for r in rows
        if measure_from_s <= float(r.get("end_rel_s") or 0.0) < args.duration_s
    )
    metrics = {
        "backend": "sglang-omni",
        "shape": "direct_full_no_prerun_closed_loop",
        "trunk_size": trunk_size,
        "concurrency": concurrency,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "measurement_from_s": measure_from_s,
        "completed": len(rows),
        "failed": len(errors),
        "total_wall_s": total_wall_s,
        "audio_only": args.audio_only,
        "text_only": args.text_only,
        "capture_limit": args.capture_limit,
        "ttft_semantics": "client first streamed text event",
        "ttfa_semantics": "client first streamed audio event",
        "all": _summarize_rows(
            rows,
            window_s=max(total_wall_s, 0.001),
            total_wall_s=total_wall_s,
            completed_in_window=len(rows),
        ),
        "steady": _summarize_rows(
            steady_rows,
            window_s=measurement_window_s,
            total_wall_s=total_wall_s,
            completed_in_window=completed_in_window,
        ),
        "errors": errors,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "per_request.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


async def main_async(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_out = Path(args.output_dir) / f"sglang_direct_steady_{timestamp}"
    root_out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["chunk_sizes"] = args.chunk_sizes
    config["concurrency_levels"] = args.concurrency_levels
    (root_out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summaries: list[dict[str, Any]] = []
    strict_failures: list[dict[str, Any]] = []
    combos = list(product(args.chunk_sizes, args.concurrency_levels))
    for combo_index, (trunk_size, concurrency) in enumerate(combos):
        print(
            f"\n=== [{combo_index + 1}/{len(combos)}] "
            f"sglang direct-full chunk={trunk_size} concurrency={concurrency} ===",
            flush=True,
        )
        metrics = await run_combo(
            args=args,
            trunk_size=trunk_size,
            concurrency=concurrency,
            combo_index=combo_index,
            root_out=root_out,
        )
        summaries.append(metrics)
        steady = metrics["steady"]
        combo_failures: list[str] = []
        if args.fail_on_error and int(metrics.get("failed") or 0) > 0:
            combo_failures.append(f"failed={metrics.get('failed')}")
        if args.fail_on_bang and int(steady.get("bang_count") or 0) > 0:
            combo_failures.append(f"bang_count={steady.get('bang_count')}")
        if (
            args.min_steady_requests > 0
            and int(steady.get("requests") or 0) < args.min_steady_requests
        ):
            combo_failures.append(
                f"steady_requests={steady.get('requests')}<min={args.min_steady_requests}"
            )
        if combo_failures:
            strict_failures.append(
                {
                    "trunk_size": trunk_size,
                    "concurrency": concurrency,
                    "failures": combo_failures,
                }
            )
        print(
            "summary "
            f"req={steady['requests']} qps={steady['admission_qps']:.4f} "
            f"ttft_avg={steady['ttft_avg_ms']} ttfa_avg={steady['ttfa_avg_ms']} "
            f"e2e_avg={steady['e2e_avg_ms']} failed={metrics.get('failed')} "
            f"bang={steady['bang_count']}",
            flush=True,
        )
    (root_out / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if strict_failures:
        (root_out / "strict_failures.json").write_text(
            json.dumps(strict_failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"strict_failures={strict_failures}", flush=True)
        raise SystemExit(2)
    print(f"\nresults={root_out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8162")
    parser.add_argument("--model", default="qwen3_5-omni")
    parser.add_argument("--rtc-test-dir", default="/myapp/data/share-data-6batch")
    parser.add_argument("--output-dir", default="results/direct_steady")
    parser.add_argument("--chunk-sizes", type=_parse_int_list, default=[40])
    parser.add_argument("--concurrency-levels", type=_parse_int_list, default=[12])
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--stagger-ms", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=6)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--request-offset-stride", type=int, default=480)
    parser.add_argument("--max-sequences-per-batch", type=int, default=700)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument("--max-chunks-per-turn", type=int, default=6)
    parser.add_argument("--visual-mode", choices=["video_frames", "images"], default="video_frames")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--capture-limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3408)
    parser.add_argument("--voice", default="f245")
    parser.add_argument("--talker-temperature", type=float, default=0.9)
    parser.add_argument("--talker-top-k", type=int, default=50)
    parser.add_argument("--talker-top-p", type=float, default=1.0)
    parser.add_argument("--talker-min-p", type=float, default=None)
    parser.add_argument("--talker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--talker-seed", type=int, default=3408)
    parser.add_argument("--subtalker-temperature", type=float, default=0.1)
    parser.add_argument("--subtalker-top-k", type=int, default=5)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-min-p", type=float, default=None)
    parser.add_argument("--subtalker-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-seed", type=int, default=3408)
    parser.add_argument("--video-max-frames", type=int, default=0)
    parser.add_argument("--video-min-pixels", type=int, default=0)
    parser.add_argument("--video-max-pixels", type=int, default=0)
    parser.add_argument("--video-total-pixels", type=int, default=0)
    parser.add_argument("--video-override-max-pixels", type=int, choices=[0, 1], default=None)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--fail-on-bang", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--min-steady-requests", type=int, default=0)
    args = parser.parse_args()
    if isinstance(args.chunk_sizes, str):
        args.chunk_sizes = _parse_int_list(args.chunk_sizes)
    if isinstance(args.concurrency_levels, str):
        args.concurrency_levels = _parse_int_list(args.concurrency_levels)
    return args


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
