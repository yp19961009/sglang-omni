# SPDX-License-Identifier: Apache-2.0
"""Video understanding benchmark helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import random
import re
import struct
import time
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
    wav_path: str
    predicted: str
    raw_response: str
    is_correct: bool
    is_success: bool
    is_mc_fallback: bool
    text_ttft_s: float | None
    error: str


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
        audio_obj = message.get("audio")
        if not isinstance(audio_obj, dict):
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


def _apply_usage(result: RequestResult, usage: dict[str, Any] | None) -> None:
    if not usage:
        return
    result.prompt_tokens = usage.get("prompt_tokens", 0) or result.prompt_tokens
    result.completion_tokens = (
        usage.get("completion_tokens", 0) or result.completion_tokens
    )


async def _apply_streaming_chat_completion_response(
    result: RequestResult,
    response: aiohttp.ClientResponse,
    *,
    start_time: float,
) -> bool:
    text_chunks: list[str] = []
    usage: dict[str, Any] | None = None
    buffer = ""

    async for raw_chunk in response.content.iter_any():
        if not raw_chunk:
            continue
        buffer += raw_chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                continue

            try:
                body = json.loads(data)
            except json.JSONDecodeError as exc:
                result.error = f"Invalid streaming chunk: {exc}"
                return False

            chunk_usage = body.get("usage")
            if isinstance(chunk_usage, dict):
                usage = chunk_usage

            choices = body.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                content = str(content)
            if content == "":
                continue

            if result.text_ttft_s is None:
                result.text_ttft_s = time.perf_counter() - start_time
            text_chunks.append(content)

    result.text = "".join(text_chunks)
    _apply_usage(result, usage)
    if result.completion_tokens <= 0 and text_chunks:
        # SGLang emits one content delta per generated token. Use that as a
        # fallback when the streaming final chunk does not include usage.
        result.completion_tokens = len(text_chunks)
    result.is_success = True
    return True


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _cache_tag(value: float | int | None) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def _preprocessed_video_spec(
    sample: VideoMMESample | VideoAMMESample,
    *,
    preprocessed_video_dir: str | None,
    video_fps: float | None,
    video_max_frames: int | None,
    video_max_pixels: int | None,
    reuse_loaded: bool = False,
) -> dict[str, Any] | None:
    if not preprocessed_video_dir:
        return None

    video_key = sample.video_id or sample.sample_id
    filename = (
        f"{_safe_cache_key(video_key)}_fps{_cache_tag(video_fps)}_"
        f"frames{_cache_tag(video_max_frames)}_px{_cache_tag(video_max_pixels)}.pt"
    )
    path = os.path.join(preprocessed_video_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing preprocessed video cache for sample {sample.sample_id}: {path}"
        )
    spec: dict[str, Any] = {"path": path}
    if reuse_loaded:
        spec["reuse_loaded"] = True
    return spec


def _preprocessed_audio_spec(
    sample: VideoAMMESample,
    *,
    preprocessed_audio_dir: str | None,
    reuse_loaded: bool = False,
) -> dict[str, Any] | None:
    if not preprocessed_audio_dir:
        return None

    filename = f"{_safe_cache_key(sample.sample_id)}_sr16000.pt"
    path = os.path.join(preprocessed_audio_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing preprocessed audio cache for sample {sample.sample_id}: {path}"
        )
    spec: dict[str, Any] = {"path": path}
    if reuse_loaded:
        spec["reuse_loaded"] = True
    return spec


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
    preprocessed_video_dir: str | None = None,
    preprocessed_audio_dir: str | None = None,
    reuse_preprocessed_media: bool = False,
    enable_audio_input: bool = False,
    audio_output_dir: str | None = None,
    talker_max_new_tokens: int | None = None,
    talker_temperature: float | None = None,
    fixed_prompt: str | None = None,
    stream: bool = False,
) -> SendFn:
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

        start_time = time.perf_counter()
        try:
            preprocessed_video = _preprocessed_video_spec(
                sample,
                preprocessed_video_dir=preprocessed_video_dir,
                video_fps=video_fps,
                video_max_frames=video_max_frames,
                video_max_pixels=video_max_pixels,
                reuse_loaded=reuse_preprocessed_media,
            )
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": modalities,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
            if preprocessed_video is None:
                payload["videos"] = [sample.video_path]
            else:
                payload["preprocessed_videos"] = [preprocessed_video]
            if stream:
                payload["stream_options"] = {"include_usage": True}
            if enable_audio_input:
                assert isinstance(sample, VideoAMMESample)
                preprocessed_audio = _preprocessed_audio_spec(
                    sample,
                    preprocessed_audio_dir=preprocessed_audio_dir,
                    reuse_loaded=reuse_preprocessed_media,
                )
                if preprocessed_audio is None:
                    payload["audios"] = [sample.audio_path]
                else:
                    payload["preprocessed_audios"] = [preprocessed_audio]
            if audio_output_dir:
                payload["audio"] = {"format": "wav"}
                if talker_max_new_tokens is not None:
                    payload["talker_max_new_tokens"] = talker_max_new_tokens
                if talker_temperature is not None:
                    payload["talker_temperature"] = talker_temperature
            if preprocessed_video is None:
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

            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                if stream:
                    if audio_output_dir:
                        result.error = (
                            "Streaming Video-MME benchmark supports text output only"
                        )
                        return result
                    if not await _apply_streaming_chat_completion_response(
                        result,
                        response,
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
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
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
            "wav_path": result.wav_path or "",
            "predicted": "",
            "raw_response": result.error,
            "is_correct": False,
            "is_success": False,
            "is_mc_fallback": False,
            "text_ttft_s": (
                round(result.text_ttft_s, 4) if result.text_ttft_s is not None else None
            ),
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
