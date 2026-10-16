#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run an SGLang Qwen3.5-Omni RTC-style single-session profile.

This mirrors the vLLM ``rtc_profile_client`` single-session shape:
incremental pre-runs for trunk 1..T warm prefix/MM caches, then one measured
audio-video actual run at trunk T.  The measured request is streamed so TTFT
and TTFA are measured from the OpenAI-compatible HTTP boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import json
import math
import os
import statistics
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.benchmarker.utils import get_wav_duration


RTC_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)
ASSISTANT_STUB = "我是千问多模态大模型"


def _frame_paths(test_dir: Path, batch_idx: int, start: int, fps: int) -> list[str]:
    return [
        str(test_dir / f"video_batches/video_b{batch_idx}/frame_{start + k:04d}.jpg")
        for k in range(fps * 2)
    ]


def _make_chunk_content(
    *,
    test_dir: Path,
    n_chunks: int,
    batch_idx: int,
    fps: int,
    frame_start: int,
    sil_start: int,
    audio_only: bool,
    visual_mode: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    frame_idx = frame_start
    audio_idx = sil_start
    for _ in range(n_chunks):
        if not audio_only:
            frames = _frame_paths(test_dir, batch_idx, frame_idx, fps)
            if visual_mode == "video_frames":
                content.append({"type": "video", "video": frames})
            else:
                content.extend({"type": "image", "image": frame} for frame in frames)
            frame_idx += fps * 2
        content.append(
            {
                "type": "audio",
                "audio": str(
                    test_dir
                    / f"audio_batches/video_input_b{batch_idx}"
                    / f"silence_with_noise/2s_random_silence_{audio_idx}.wav"
                ),
            }
        )
        audio_idx += 1
    return content


def make_rtc_messages(
    *,
    test_dir: Path,
    trunk_size: int,
    batch_idx: int,
    pre_run: bool,
    audio_only: bool,
    video_fps: int,
    max_chunks_per_turn: int,
    sil_start_idx: int,
    video_start_idx: int,
    question_idx: int,
    visual_mode: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RTC_SYSTEM_PROMPT},
    ]
    frame_idx = video_start_idx
    audio_idx = sil_start_idx
    data_chunks = trunk_size - 1

    while data_chunks > max_chunks_per_turn:
        page = _make_chunk_content(
            test_dir=test_dir,
            n_chunks=max_chunks_per_turn,
            batch_idx=batch_idx,
            fps=video_fps,
            frame_start=frame_idx,
            sil_start=audio_idx,
            audio_only=audio_only,
            visual_mode=visual_mode,
        )
        frame_idx += max_chunks_per_turn * video_fps * 2 if not audio_only else 0
        audio_idx += max_chunks_per_turn
        messages.append({"role": "user", "content": page})
        messages.append({"role": "assistant", "content": ASSISTANT_STUB})
        data_chunks -= max_chunks_per_turn

    content = _make_chunk_content(
        test_dir=test_dir,
        n_chunks=data_chunks,
        batch_idx=batch_idx,
        fps=video_fps,
        frame_start=frame_idx,
        sil_start=audio_idx,
        audio_only=audio_only,
        visual_mode=visual_mode,
    )
    if not pre_run:
        q_frame_idx = frame_idx + data_chunks * video_fps * 2 if not audio_only else 0
        if not audio_only:
            frames = _frame_paths(test_dir, batch_idx, q_frame_idx, video_fps)
            if visual_mode == "video_frames":
                content.append({"type": "video", "video": frames})
            else:
                content.extend({"type": "image", "image": frame} for frame in frames)
        content.append(
            {
                "type": "audio",
                "audio": str(
                    test_dir
                    / f"audio_batches/video_input_b{batch_idx}"
                    / f"question_with_noise/2s_question_with_noise_{question_idx}.wav"
                ),
            }
        )

    messages.append({"role": "user", "content": content})
    return messages


def _apply_video_request_options(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.video_max_frames:
        payload["video_max_frames"] = args.video_max_frames
    if args.video_min_pixels:
        payload["video_min_pixels"] = args.video_min_pixels
    if args.video_max_pixels:
        payload["video_max_pixels"] = args.video_max_pixels
    if args.video_total_pixels:
        payload["video_total_pixels"] = args.video_total_pixels
    if args.video_override_max_pixels is not None:
        payload["video_override_max_pixels"] = bool(args.video_override_max_pixels)


def _apply_talker_request_options(payload: dict[str, Any], args: argparse.Namespace) -> None:
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


async def _iter_sse_json(response: aiohttp.ClientResponse):
    while True:
        raw = await response.content.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if not body or body == "[DONE]":
            continue
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


def _event_text_delta(evt: dict[str, Any]) -> str | None:
    for choice in evt.get("choices", []):
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def _event_audio_obj(evt: dict[str, Any]) -> dict[str, Any] | None:
    for choice in evt.get("choices", []):
        delta = choice.get("delta") or {}
        audio = delta.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            return audio
    return None


def _event_usage(evt: dict[str, Any]) -> dict[str, Any] | None:
    usage = evt.get("usage")
    return usage if isinstance(usage, dict) else None


def _wav_chunk_payload(
    wav_bytes: bytes,
) -> tuple[tuple[int, int, int, str, str], bytes] | None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            params = (
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getframerate(),
                wav_file.getcomptype(),
                wav_file.getcompname(),
            )
            frames = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error):
        return None
    return params, frames


def _combine_wav_chunks(
    chunks: list[tuple[tuple[int, int, int, str, str], bytes]],
) -> bytes | None:
    if not chunks:
        return None
    params = chunks[0][0]
    frames: list[bytes] = []
    for chunk_params, chunk_frames in chunks:
        if chunk_params != params:
            return None
        frames.append(chunk_frames)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        channels, sample_width, sample_rate, comp_type, comp_name = params
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.setcomptype(comp_type, comp_name)
        wav_file.writeframes(b"".join(frames))
    return buffer.getvalue()


@dataclass
class RequestMetrics:
    request_id: str
    text: str
    first_output_ms: float | None
    first_output_type: str | None
    first_text_event_ms: float | None
    first_audio_event_ms: float | None
    text_audio_event_gap_ms: float | None
    ttft_ms: float | None
    ttfa_ms: float | None
    e2e_total_ms: float
    audio_chunk_count: int
    audio_duration_s: float | None
    inter_chunk_ms: list[float]
    prompt_tokens: int
    completion_tokens: int
    wav_path: str | None


async def post_chat(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    payload: dict[str, Any],
    stream: bool,
    output_wav: Path | None = None,
) -> RequestMetrics:
    start = time.perf_counter()
    text_parts: list[str] = []
    first_output_at: float | None = None
    first_output_type: str | None = None
    first_text_at: float | None = None
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    inter_chunk_ms: list[float] = []
    audio_duration_s = 0.0
    audio_chunks = 0
    wav_chunks: list[tuple[tuple[int, int, int, str, str], bytes]] = []
    usage: dict[str, Any] = {}

    async with session.post(api_url, json=payload) as response:
        response.raise_for_status()
        if stream:
            async for evt in _iter_sse_json(response):
                now = time.perf_counter()
                text_delta = _event_text_delta(evt)
                if text_delta:
                    text_parts.append(text_delta)
                    if first_text_at is None:
                        first_text_at = now
                    if first_output_at is None:
                        first_output_at = now
                        first_output_type = "text"
                audio_obj = _event_audio_obj(evt)
                if audio_obj is not None:
                    if first_audio_at is None:
                        first_audio_at = now
                    if first_output_at is None:
                        first_output_at = now
                        first_output_type = "audio"
                    if last_audio_at is not None:
                        inter_chunk_ms.append((now - last_audio_at) * 1000.0)
                    last_audio_at = now
                    audio_chunks += 1
                    try:
                        wav_bytes = base64.b64decode(audio_obj.get("data", ""), validate=True)
                        audio_duration_s += get_wav_duration(wav_bytes)
                        wav_chunk = _wav_chunk_payload(wav_bytes)
                        if wav_chunk is not None:
                            wav_chunks.append(wav_chunk)
                    except (binascii.Error, ValueError, wave.Error):
                        pass
                evt_usage = _event_usage(evt)
                if evt_usage:
                    usage = evt_usage
        else:
            body = await response.json()
            message = body.get("choices", [{}])[0].get("message", {})
            text = message.get("content", "") or ""
            if text:
                text_parts.append(text)
                first_text_at = time.perf_counter()
                first_output_at = first_text_at
                first_output_type = "text"
            usage = body.get("usage") or {}

    end = time.perf_counter()
    wav_path: str | None = None
    full_wav = _combine_wav_chunks(wav_chunks)
    if output_wav is not None and full_wav is not None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(full_wav)
        wav_path = str(output_wav)

    first_output_ms = (
        (first_output_at - start) * 1000.0 if first_output_at is not None else None
    )
    first_text_event_ms = (
        (first_text_at - start) * 1000.0 if first_text_at is not None else None
    )
    first_audio_event_ms = (
        (first_audio_at - start) * 1000.0 if first_audio_at is not None else None
    )
    text_audio_event_gap_ms = (
        first_audio_event_ms - first_text_event_ms
        if first_text_event_ms is not None and first_audio_event_ms is not None
        else None
    )

    return RequestMetrics(
        request_id=str(payload.get("metadata", {}).get("request_id") or ""),
        text="".join(text_parts),
        first_output_ms=first_output_ms,
        first_output_type=first_output_type,
        first_text_event_ms=first_text_event_ms,
        first_audio_event_ms=first_audio_event_ms,
        text_audio_event_gap_ms=text_audio_event_gap_ms,
        ttft_ms=first_text_event_ms,
        ttfa_ms=first_audio_event_ms,
        e2e_total_ms=(end - start) * 1000.0,
        audio_chunk_count=audio_chunks,
        audio_duration_s=audio_duration_s if audio_duration_s > 0 else None,
        inter_chunk_ms=inter_chunk_ms,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        wav_path=wav_path,
    )


def _profile_rows(profile: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return profile.get("stage_breakdown", []), profile.get("hop_breakdown", [])


def _stage_total(profile: dict[str, Any], stage: str, interval: str) -> float | None:
    for row in profile.get("stage_breakdown", []):
        if row.get("stage") == stage and row.get("interval") == interval:
            return float(row.get("total_ms", 0.0))
    return None


def _timeline_first(
    timeline: list[dict[str, Any]],
    stage: str,
    event_name: str,
) -> float | None:
    for event in timeline:
        if event.get("stage") == stage and event.get("event_name") == event_name:
            return float(event.get("t_rel_ms", 0.0))
    return None


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    timelines = profile.get("timelines", {})
    request_id = next(iter(timelines), None)
    timeline = sorted(timelines.get(request_id, []), key=lambda item: item.get("t_rel_ms", 0.0)) if request_id else []
    return {
        "request_count": profile.get("request_count"),
        "profile_request_id": request_id,
        "stage_ms": {
            "preprocessing": _stage_total(profile, "preprocessing", "stage_input_received->stage_complete"),
            "preprocess_core": _stage_total(profile, "preprocessing", "preprocess_start->preprocess_end"),
            "preprocess_video_load": _stage_total(profile, "preprocessing", "preprocess_video_load_start->preprocess_video_load_end"),
            "preprocess_audio_load": _stage_total(profile, "preprocessing", "preprocess_audio_load_start->preprocess_audio_load_end"),
            "preprocess_hf_processor": _stage_total(profile, "preprocessing", "preprocess_hf_processor_start->preprocess_hf_processor_end"),
            "preprocess_prompt": _stage_total(profile, "preprocessing", "preprocess_prompt_start->preprocess_prompt_end"),
            "audio_encoder": _stage_total(profile, "audio_encoder", "stage_input_received->stage_complete"),
            "image_encoder": _stage_total(profile, "image_encoder", "stage_input_received->stage_complete"),
            "mm_aggregate": _stage_total(profile, "mm_aggregate", "stage_input_received->stage_complete"),
            "thinker": _stage_total(profile, "thinker", "stage_input_received->stage_complete"),
            "thinker_prefill_to_first_emit": _stage_total(profile, "thinker", "scheduler_prefill_start->scheduler_first_emit"),
            "talker_ar": _stage_total(profile, "talker_ar", "stage_input_received->stage_complete"),
            "talker_prefill_to_first_chunk": _stage_total(profile, "talker_ar", "scheduler_prefill_start->stage_first_stream_chunk_sent"),
            "code2wav": _stage_total(profile, "code2wav", "stage_input_received->stage_complete"),
            "code2wav_first_decode": _stage_total(profile, "code2wav", "code2wav_decode_start->code2wav_decode_end"),
        },
        "timeline_anchors_ms": {
            "admission": _timeline_first(timeline, "coordinator", "request_admission"),
            "preprocess_start": _timeline_first(timeline, "preprocessing", "preprocess_start"),
            "preprocess_end": _timeline_first(timeline, "preprocessing", "preprocess_end"),
            "image_encoder_start": _timeline_first(timeline, "image_encoder", "encoder_start"),
            "image_encoder_end": _timeline_first(timeline, "image_encoder", "encoder_end"),
            "audio_encoder_start": _timeline_first(timeline, "audio_encoder", "encoder_start"),
            "audio_encoder_end": _timeline_first(timeline, "audio_encoder", "encoder_end"),
            "thinker_prefill_start": _timeline_first(timeline, "thinker", "scheduler_prefill_start"),
            "thinker_first_emit": _timeline_first(timeline, "thinker", "scheduler_first_emit"),
            "talker_first_chunk": _timeline_first(timeline, "talker_ar", "stage_first_stream_chunk_sent"),
            "code2wav_first_chunk": _timeline_first(timeline, "code2wav", "stage_first_stream_chunk_sent"),
            "code2wav_done": _timeline_first(timeline, "code2wav", "stage_complete"),
        },
        "top_stage_by_total_ms": sorted(
            profile.get("stage_breakdown", []),
            key=lambda row: row.get("total_ms", 0.0),
            reverse=True,
        )[:16],
        "top_hop_by_total_ms": sorted(
            profile.get("hop_breakdown", []),
            key=lambda row: row.get("total_ms", 0.0),
            reverse=True,
        )[:16],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = args.base_url.rstrip("/") + "/v1/chat/completions"
    profile_start_url = args.base_url.rstrip("/") + "/start_request_profile"
    profile_stop_url = args.base_url.rstrip("/") + "/stop_request_profile"
    request_id = f"sg_ck{args.trunk_size}_{uuid.uuid4()}"
    media_cache_namespace = f"rtc:{request_id}"

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for trunk in range(1, args.trunk_size + 1):
            messages = make_rtc_messages(
                test_dir=Path(args.rtc_test_dir),
                trunk_size=trunk,
                batch_idx=args.batch_idx,
                pre_run=True,
                audio_only=args.audio_only,
                video_fps=args.video_fps,
                max_chunks_per_turn=args.max_chunks_per_turn,
                sil_start_idx=args.sil_start_idx,
                video_start_idx=args.video_start_idx,
                question_idx=args.question_idx,
                visual_mode=args.visual_mode,
            )
            payload = {
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
            _apply_talker_request_options(payload, args)
            t0 = time.perf_counter()
            await post_chat(session, api_url=api_url, payload=payload, stream=False)
            print(f"pre_run trunk={trunk}/{args.trunk_size} done in {(time.perf_counter()-t0):.3f}s", flush=True)

        profile_json: dict[str, Any] | None = None
        if args.profile:
            async with session.post(
                profile_start_url,
                json={"run_id": args.run_id, "event_dir": str(Path(args.event_dir).resolve())},
            ) as response:
                response.raise_for_status()
                print("profile_start", await response.text(), flush=True)

        messages = make_rtc_messages(
            test_dir=Path(args.rtc_test_dir),
            trunk_size=args.trunk_size,
            batch_idx=args.batch_idx,
            pre_run=False,
            audio_only=args.audio_only,
            video_fps=args.video_fps,
            max_chunks_per_turn=args.max_chunks_per_turn,
            sil_start_idx=args.sil_start_idx,
            video_start_idx=args.video_start_idx,
                question_idx=args.question_idx,
                visual_mode=args.visual_mode,
            )
        payload = {
            "model": args.model,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"format": "wav", "voice": args.voice},
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
        _apply_video_request_options(payload, args)
        _apply_talker_request_options(payload, args)

        measured = await post_chat(
            session,
            api_url=api_url,
            payload=payload,
            stream=True,
            output_wav=out_dir / "audio" / f"{request_id}.wav",
        )

        if args.profile:
            async with session.post(profile_stop_url, json={"run_id": args.run_id}) as response:
                response.raise_for_status()
                print("profile_stop", await response.text(), flush=True)

    if args.profile:
        # Convert event JSONL to a compact JSON view with the existing profiler.
        import subprocess

        profile_out = out_dir / f"request_profile_{args.run_id}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang_omni.profiler",
                str(Path(args.event_dir).resolve()),
                "--format",
                "json",
                "--out",
                str(profile_out.resolve()),
            ],
            check=True,
        )
        with profile_out.open("r", encoding="utf-8") as f:
            profile_json = json.load(f)

    ttft_ms = measured.ttft_ms
    ttfa_ms = measured.ttfa_ms
    e2e_ms = measured.e2e_total_ms
    audio_dur = measured.audio_duration_s
    rtf = e2e_ms / 1000.0 / audio_dur if audio_dur and audio_dur > 0 else None
    c2w_rtf = None
    if profile_json:
        c2w_ms = _stage_total(profile_json, "code2wav", "stage_input_received->stage_complete")
        if c2w_ms and audio_dur and audio_dur > 0:
            c2w_rtf = c2w_ms / 1000.0 / audio_dur

    metrics = {
        "trunk_size": args.trunk_size,
        "concurrency": 1,
        "total_samples": 1,
        "num_requests": 1,
        "avg_ttft_ms": ttft_ms,
        "p50_ttft_ms": ttft_ms,
        "p99_ttft_ms": ttft_ms,
        "ttft_semantics": "first streamed text event",
        "ttfa_semantics": "first streamed audio event",
        "first_output_ms": measured.first_output_ms,
        "first_output_type": measured.first_output_type,
        "first_text_event_ms": measured.first_text_event_ms,
        "first_audio_event_ms": measured.first_audio_event_ms,
        "text_audio_event_gap_ms": measured.text_audio_event_gap_ms,
        "audio_before_text_event": (
            measured.text_audio_event_gap_ms is not None
            and measured.text_audio_event_gap_ms < 0
        ),
        "avg_ttfa_ms": ttfa_ms,
        "p50_ttfa_ms": ttfa_ms,
        "p99_ttfa_ms": ttfa_ms,
        "e2e_total_ms": e2e_ms,
        "audio_duration_s": audio_dur,
        "audio_rtf_e2e": rtf,
        "audio_rtf_code2wav_stage": c2w_rtf,
        "audio_chunks": measured.audio_chunk_count,
        "inter_chunk_mean_ms": statistics.fmean(measured.inter_chunk_ms) if measured.inter_chunk_ms else None,
        "prompt_tokens": measured.prompt_tokens,
        "completion_tokens": measured.completion_tokens,
        "visual_mode": args.visual_mode,
        "media_cache_namespace": media_cache_namespace,
    }

    profile_summary = summarize_profile(profile_json) if profile_json else {}
    if profile_summary:
        metrics.update(
            {
                "stage_preprocessing_ms": profile_summary["stage_ms"].get("preprocessing"),
                "stage_image_encoder_ms": profile_summary["stage_ms"].get("image_encoder"),
                "stage_audio_encoder_ms": profile_summary["stage_ms"].get("audio_encoder"),
                "stage_mm_aggregate_ms": profile_summary["stage_ms"].get("mm_aggregate"),
                "stage_thinker_ms": profile_summary["stage_ms"].get("thinker"),
                "stage_talker_ar_ms": profile_summary["stage_ms"].get("talker_ar"),
                "stage_code2wav_ms": profile_summary["stage_ms"].get("code2wav"),
                "thinker_prefill_to_first_emit_ms": profile_summary["stage_ms"].get("thinker_prefill_to_first_emit"),
                "talker_prefill_to_first_chunk_ms": profile_summary["stage_ms"].get("talker_prefill_to_first_chunk"),
            }
        )

    per_request = {
        "request_id": request_id,
        "text": measured.text,
        "ttft_ms": ttft_ms,
        "ttfa_ms": ttfa_ms,
        "first_output_ms": measured.first_output_ms,
        "first_output_type": measured.first_output_type,
        "first_text_event_ms": measured.first_text_event_ms,
        "first_audio_event_ms": measured.first_audio_event_ms,
        "text_audio_event_gap_ms": measured.text_audio_event_gap_ms,
        "e2e_total_ms": e2e_ms,
        "audio_chunk_count": measured.audio_chunk_count,
        "audio_duration_s": audio_dur,
        "inter_chunk_mean_ms": metrics["inter_chunk_mean_ms"],
        "wav_path": measured.wav_path,
        "prompt_tokens": measured.prompt_tokens,
        "completion_tokens": measured.completion_tokens,
        "profile": profile_summary,
    }

    comparison = [
        {
            "trunk_size": args.trunk_size,
            "concurrency": 1,
            "total_samples": 1,
            "audio_only": args.audio_only,
            "completed": 1,
            "elapsed_s": round(e2e_ms / 1000.0, 3),
            "metrics": metrics,
        }
    ]

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "per_request.json").write_text(json.dumps([per_request], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSGLang RTC Profile Result")
    print(f"  output_dir={out_dir}")
    print(f"  request_id={request_id}")
    print(f"  TTFT={ttft_ms:.2f}ms" if ttft_ms is not None else "  TTFT=-")
    print(f"  TTFA={ttfa_ms:.2f}ms" if ttfa_ms is not None else "  TTFA=-")
    print(f"  e2e={e2e_ms:.2f}ms audio_chunks={measured.audio_chunk_count}")
    if audio_dur:
        print(f"  audio_duration={audio_dur:.3f}s e2e_rtf={rtf:.3f}")
    return {"metrics": metrics, "per_request": per_request}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8161")
    parser.add_argument("--model", default="qwen3_5-omni")
    parser.add_argument("--rtc-test-dir", default="/myapp/data/share-data-6batch")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--run-id", default="sglang_rtc_trunk40_c1")
    parser.add_argument("--trunk-size", type=int, default=40)
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--sil-start-idx", type=int, default=0)
    parser.add_argument("--video-start-idx", type=int, default=0)
    parser.add_argument("--question-idx", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=1)
    parser.add_argument(
        "--visual-mode",
        choices=("image_frames", "video_frames"),
        default="video_frames",
        help=(
            "How to send the extracted JPEG frames to SGLang. video_frames "
            "matches the vLLM RTC prompt shape; image_frames is only for "
            "diagnostics and can create many more visual tokens."
        ),
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
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--prerun-max-tokens", type=int, default=0)
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
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-s", type=float, default=1800)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
