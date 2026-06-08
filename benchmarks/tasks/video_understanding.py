# SPDX-License-Identifier: Apache-2.0
"""Video understanding benchmark helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import random
import struct
import statistics
import time
import wave
from typing import Any, TypedDict

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.videomme import VideoAMMESample, VideoMMESample
from benchmarks.tasks.visual_understand import parse_multi_choice_response

logger = logging.getLogger(__name__)

VIDEOAMME_REQUEST_TEXT = (
    "Use the video and the audio question to answer. "
    "Return the final answer as Answer: $LETTER."
)


class VideoMMERecord(TypedDict):
    sample_id: str
    video_path: str
    url: str
    video_id: str
    question_id: str
    duration: str
    domain: str
    sub_category: str
    task_type: str
    expected: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    output_token_rate: float | None
    audio_duration_s: float | None
    rtf: float | None
    audio_ttfp_s: float | None
    text_ttft_s: float | None
    audio_chunks: int
    inter_chunk_mean_s: float | None
    wav_path: str
    predicted: str
    raw_response: str
    is_correct: bool
    is_success: bool
    is_mc_fallback: bool
    error: str


def _chat_completion_audio_obj(
    body: dict[str, Any],
    message: dict[str, Any],
) -> dict[str, Any] | None:
    audio_obj = message.get("audio")
    if isinstance(audio_obj, dict):
        return audio_obj
    top_level_audio = body.get("audio")
    if isinstance(top_level_audio, dict):
        # 中文说明：sglang-omni 使用 OpenAI-style message.audio；
        # vLLM perf_v2 的 Qwen3.5 server 非流式响应把音频放在顶层
        # audio。benchmark 同时兼容，方便同一套脚本横向对比。
        return top_level_audio
    return None


def _apply_chat_completion_response(
    result: RequestResult,
    body: dict[str, Any],
    *,
    audio_output_dir: str | None,
    sample_id: str,
) -> bool:
    message = body.get("choices", [{}])[0].get("message", {})
    result.text = message.get("content", "") or ""
    wav_bytes = b""

    if audio_output_dir:
        audio_obj = _chat_completion_audio_obj(body, message)
        if audio_obj is None:
            result.error = "No audio in response"
            return False
        audio_b64 = audio_obj.get("data", "")
        if not audio_b64:
            result.error = "Empty audio data in response"
            return False
        try:
            wav_bytes = base64.b64decode(audio_b64, validate=True)
            result.audio_duration_s = round(get_wav_duration(wav_bytes), 4)
        except (binascii.Error, ValueError, struct.error) as exc:
            result.error = f"Invalid audio data: {exc}"
            return False

    usage = body.get("usage", {})
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)

    if audio_output_dir and result.audio_duration_s > 0:
        try:
            os.makedirs(audio_output_dir, exist_ok=True)
            wav_path = os.path.join(audio_output_dir, f"{sample_id}.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
        except OSError as exc:
            result.error = f"Failed to save audio: {exc}"
            return False
        result.wav_path = wav_path

    result.is_success = True
    return True


def _event_text_delta(evt: dict[str, Any]) -> str | None:
    for choice in evt.get("choices", []):
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def _event_audio_obj(evt: dict[str, Any]) -> dict[str, Any] | None:
    top_level_audio = evt.get("audio")
    if isinstance(top_level_audio, dict) and top_level_audio.get("data"):
        # 中文说明：vLLM perf_v2 的 Qwen3.5 streaming server 会发送
        # object=chat.completion.audio 的顶层 audio 事件；sglang-omni
        # 则走 choices[].delta.audio。两种都算首音频 chunk。
        return top_level_audio
    for choice in evt.get("choices", []):
        delta = choice.get("delta") or {}
        audio = delta.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            return audio
    return None


def _event_usage(evt: dict[str, Any]) -> dict[str, Any] | None:
    usage = evt.get("usage")
    return usage if isinstance(usage, dict) else None


def _decode_wav_duration(audio_obj: dict[str, Any]) -> tuple[bytes, float] | None:
    audio_b64 = audio_obj.get("data", "")
    if not audio_b64:
        return None
    try:
        wav_bytes = base64.b64decode(audio_b64, validate=True)
        return wav_bytes, round(get_wav_duration(wav_bytes), 4)
    except (binascii.Error, ValueError, struct.error):
        return None


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
            logger.debug("Skipping streamed audio save: WAV chunk params changed")
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
            logger.debug("Skipping non-JSON SSE payload: %s", body[:120])


async def _apply_chat_completion_stream_response(
    result: RequestResult,
    response: aiohttp.ClientResponse,
    *,
    audio_output_dir: str | None,
    sample_id: str,
    start_time: float,
) -> bool:
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    audio_duration_s = 0.0
    audio_chunks = 0
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    inter_chunk_s: list[float] = []
    first_text_at: float | None = None
    full_audio_wav: bytes | None = None
    delta_audio_chunks: list[tuple[tuple[int, int, int, str, str], bytes]] = []

    async for evt in _iter_sse_json(response):
        now = time.perf_counter()
        text_delta = _event_text_delta(evt)
        if text_delta:
            text_parts.append(text_delta)
            if first_text_at is None:
                first_text_at = now

        audio_obj = _event_audio_obj(evt)
        if audio_obj is not None:
            if first_audio_at is None:
                first_audio_at = now
            if last_audio_at is not None:
                inter_chunk_s.append(now - last_audio_at)
            last_audio_at = now
            audio_chunks += 1
            decoded = _decode_wav_duration(audio_obj)
            if decoded is not None:
                wav_bytes, duration_s = decoded
                audio_duration_s += duration_s
                if evt.get("object") == "chat.completion.audio":
                    # 中文说明：vLLM 顶层 audio 事件通常是完整 WAV，可保存
                    # 给 WER；sglang-omni 的 delta.audio 则在下面按 PCM
                    # 拼接，避免直接拼 WAV 文件头。
                    full_audio_wav = wav_bytes
                else:
                    wav_chunk = _wav_chunk_payload(wav_bytes)
                    if wav_chunk is not None:
                        delta_audio_chunks.append(wav_chunk)

        evt_usage = _event_usage(evt)
        if evt_usage:
            usage = evt_usage

    result.text = "".join(text_parts)
    if first_audio_at is not None:
        result.audio_ttfp_s = first_audio_at - start_time
    if first_text_at is not None:
        result.text_ttft_s = first_text_at - start_time
    result.inter_chunk_s = inter_chunk_s
    if audio_duration_s > 0:
        result.audio_duration_s = round(audio_duration_s, 4)
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)

    if audio_output_dir and audio_chunks == 0:
        result.error = "No audio chunks in streaming response"
        return False
    if audio_output_dir and full_audio_wav is None and delta_audio_chunks:
        # 中文说明：sglang-omni 的 streaming chat audio 走 delta.audio，
        # 每个 delta 是一个独立 WAV chunk。这里只在 WAV 参数一致时拼接
        # PCM，保证 streaming benchmark 也能保存音频用于 WER/人工检查。
        full_audio_wav = _combine_wav_chunks(delta_audio_chunks)
    if audio_output_dir and full_audio_wav is not None:
        try:
            os.makedirs(audio_output_dir, exist_ok=True)
            wav_path = os.path.join(audio_output_dir, f"{sample_id}.wav")
            with open(wav_path, "wb") as f:
                f.write(full_audio_wav)
        except OSError as exc:
            result.error = f"Failed to save audio: {exc}"
            return False
        result.wav_path = wav_path

    # 中文说明：streaming 下如果只有 delta audio 而没有完整顶层 WAV，
    # 仍然可以用于性能压测；WER 需要 --skip-wer 或 vLLM 顶层完整音频事件。
    result.is_success = True
    return True


def make_video_send_fn(
    model_name: str,
    api_url: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    video_fps: float | None = None,
    video_max_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    video_total_pixels: int | None = None,
    enable_audio_input: bool = False,
    audio_output_dir: str | None = None,
    audio_format: str = "wav",
    audio_voice: str | None = None,
    audio_language: str | None = None,
    stream_output: bool = False,
    fixed_prompt: str | None = None,
) -> SendFn:
    if audio_output_dir and audio_format != "wav":
        raise ValueError(
            "Video benchmark audio metrics currently require wav output; "
            f"got {audio_format!r}"
        )
    modalities = ["text", "audio"] if audio_output_dir else ["text"]

    async def send_fn(
        session: aiohttp.ClientSession,
        sample: VideoMMESample | VideoAMMESample,
    ) -> RequestResult:
        prompt = fixed_prompt or sample.prompt
        result = RequestResult(
            request_id=sample.sample_id,
            text=prompt[:60],
        )
        content_parts: list[dict[str, Any]] = [
            {"type": "video", "video": sample.video_path},
        ]
        if enable_audio_input:
            assert isinstance(sample, VideoAMMESample)
            content_parts.append({"type": "audio", "audio": sample.audio_path})
        content_parts.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": model_name,
            # 中文说明：用 OpenAI content parts 同时兼容 sglang-omni 和
            # vLLM perf_v2 的 Qwen3.5 OpenAI server；后者不会读取顶层
            # videos/audios 字段。
            "messages": [{"role": "user", "content": content_parts}],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream_output,
            "metadata": {"sample_id": sample.sample_id},
        }
        if audio_output_dir:
            # sglang-omni honors modalities; vLLM perf_v2 server uses
            # enable_audio_output, while offline scripts commonly use do_wave.
            payload["enable_audio_output"] = True
            payload["do_wave"] = True
            audio_config = {"format": audio_format}
            # 中文说明：Qwen3.5-Omni talker 支持通过 OpenAI audio 字段
            # 选择 voice/language。benchmark 只透传显式参数，默认行为仍和
            # Qwen3-Omni 旧压测一致。
            if audio_voice:
                audio_config["voice"] = audio_voice
                payload["voice_type"] = audio_voice
            if audio_language:
                audio_config["language"] = audio_language
            payload["audio"] = audio_config
        if video_fps is not None:
            payload["video_fps"] = video_fps
        if video_max_frames is not None:
            payload["video_max_frames"] = video_max_frames
        if video_min_pixels is not None:
            payload["video_min_pixels"] = video_min_pixels
        if video_max_pixels is not None:
            payload["video_max_pixels"] = video_max_pixels
        if video_total_pixels is not None:
            payload["video_total_pixels"] = video_total_pixels

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                if stream_output:
                    if not await _apply_chat_completion_stream_response(
                        result,
                        response,
                        audio_output_dir=audio_output_dir,
                        sample_id=sample.sample_id,
                        start_time=start_time,
                    ):
                        return result
                else:
                    body = await response.json()
                    if not _apply_chat_completion_response(
                        result,
                        body,
                        audio_output_dir=audio_output_dir,
                        sample_id=sample.sample_id,
                    ):
                        return result

            elapsed = time.perf_counter() - start_time
            result.engine_time_s = elapsed
            if result.audio_duration_s > 0:
                result.rtf = elapsed / result.audio_duration_s
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def build_videomme_result_records(
    samples: list[VideoMMESample],
    results: list[RequestResult],
) -> list[VideoMMERecord]:
    """Parse responses into persisted per-sample records."""
    assert len(samples) == len(
        results
    ), f"Sample/result count mismatch: {len(samples)} samples vs {len(results)} results"
    random.seed(42)

    per_sample: list[VideoMMERecord] = []

    for sample, result in zip(samples, results):
        record: VideoMMERecord = {
            "sample_id": sample.sample_id,
            "video_path": sample.video_path,
            "url": sample.url,
            "video_id": sample.video_id,
            "question_id": sample.question_id,
            "duration": sample.duration,
            "domain": sample.domain,
            "sub_category": sample.sub_category,
            "task_type": sample.task_type,
            "expected": sample.answer,
            "latency_s": round(result.latency_s, 4),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "output_token_rate": (
                round(result.tok_per_s, 1) if result.tok_per_s > 0 else None
            ),
            "audio_duration_s": (
                round(result.audio_duration_s, 4)
                if result.audio_duration_s > 0
                else None
            ),
            "rtf": (round(result.rtf, 4) if result.rtf > 0 else None),
            "audio_ttfp_s": (
                round(result.audio_ttfp_s, 4)
                if getattr(result, "audio_ttfp_s", None) is not None
                else None
            ),
            "text_ttft_s": (
                round(result.text_ttft_s, 4)
                if getattr(result, "text_ttft_s", None) is not None
                else None
            ),
            "audio_chunks": (
                1 + len(getattr(result, "inter_chunk_s", []) or [])
                if getattr(result, "audio_ttfp_s", None) is not None
                else 0
            ),
            "inter_chunk_mean_s": (
                round(statistics.fmean(result.inter_chunk_s), 4)
                if getattr(result, "inter_chunk_s", None)
                else None
            ),
            "wav_path": result.wav_path or "",
            "predicted": "",
            "raw_response": result.error,
            "is_correct": False,
            "is_success": False,
            "is_mc_fallback": False,
            "error": result.error,
        }

        if not result.is_success:
            per_sample.append(record)
            continue

        predicted, is_fallback = parse_multi_choice_response(
            result.text,
            sample.all_choices,
            sample.index2ans,
        )
        is_correct = predicted == sample.answer
        if is_fallback:
            logger.debug("Video-MME parse fallback for sample %s", sample.sample_id)

        record.update(
            predicted=predicted,
            raw_response=result.text,
            is_correct=is_correct,
            is_success=True,
            is_mc_fallback=is_fallback,
            error="",
        )
        per_sample.append(record)

    return per_sample
