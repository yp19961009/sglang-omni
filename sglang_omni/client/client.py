# SPDX-License-Identifier: Apache-2.0
"""Client wrapper for coordinator-based pipelines."""

from __future__ import annotations

import base64
import uuid
from dataclasses import replace
from typing import Any, AsyncIterator, Callable

import numpy as np

from sglang_omni.client.audio import (
    FORMAT_MIME_TYPES,
    encode_audio,
    to_numpy,
)
from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    ClientError,
    CompletionAudio,
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
    GenerateRequest,
    SpeechResult,
    UsageInfo,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, RequestState, StreamMessage

_MEDIA_METADATA_KEYS = (
    "image",
    "images",
    "input_image",
    "input_images",
    "image_url",
    "image_urls",
    "audio",
    "audios",
    "input_audio",
    "input_audios",
    "audio_url",
    "audio_urls",
    "video",
    "videos",
    "input_video",
    "input_videos",
    "video_url",
    "video_urls",
)
_MULTIMODAL_METADATA_OPTION_KEYS = (
    "audio_downsample_chunk_size",
    "audio_downsample_times",
    "audio_sampling_rate",
    "audio_target_sr",
    "audio_timestamp_interval",
    "dependent_audio",
    "downsample_chunk_size",
    "downsample_times",
    "fps",
    "image_max_pixels",
    "image_min_pixels",
    "max_frames",
    "max_pixels",
    "min_frames",
    "min_pixels",
    "mm_processor_kwargs",
    "multi_modal_data",
    "multi_modal_uuids",
    "return_metadata",
    "return_video_metadata",
    "sampling_rate",
    "seconds_per_chunk",
    "timestamp_interval",
    "total_pixels",
    "function_call",
    "parallel_tool_calls",
    "tool_choice",
    "tools",
    "use_audio_in_video",
    "video_dependent_audio",
    "video_fps",
    "video_max_frames",
    "video_max_pixels",
    "video_metadata",
    "video_min_frames",
    "video_min_pixels",
    "video_position_id_per_seconds",
    "video_seconds_per_chunk",
    "video_total_pixels",
    "videos_metadata",
)
_OPENAI_AUDIO_OUTPUT_CONFIG_KEYS = frozenset(
    {
        "format",
        "instruction",
        "lang",
        "language",
        "language_id",
        "speaker",
        "style",
        "target_language",
        "voice",
        "voice_clone",
        "voice_clone_info",
        "voice_clone_path",
        "voice_style",
        "voice_type",
        "xvector_info",
        "xvector_info_path",
    }
)
_AUDIO_MEDIA_PAYLOAD_KEYS = frozenset(
    {
        "audio",
        "audio_url",
        "data",
        "input_audio",
        "path",
        "samples",
        "url",
    }
)


def _format_from_mime_type(mime_type: str, fallback: str) -> str:
    for ext, candidate_mime in FORMAT_MIME_TYPES.items():
        if candidate_mime == mime_type:
            return ext
    return fallback


def _encode_audio_base64_and_format(
    audio: Any,
    *,
    output_format: str,
    sample_rate: int | None = None,
) -> tuple[str, str]:
    encode_kwargs: dict[str, Any] = {"response_format": output_format}
    if sample_rate is not None:
        encode_kwargs["sample_rate"] = sample_rate
    audio_bytes, mime_type = encode_audio(audio, **encode_kwargs)
    actual_format = _format_from_mime_type(mime_type, output_format)
    # encode_audio may fall back to WAV when pydub/soundfile is unavailable or
    # the format is unknown. Return the actual format as well, so OpenAI chat
    # audio responses do not claim mp3/opus while carrying WAV data.
    return base64.b64encode(audio_bytes).decode("ascii"), actual_format


def _metadata_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    if hasattr(value, "numel"):
        try:
            return int(value.numel()) > 0
        except Exception:
            return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    # Callers may pass PIL Images, numpy scalars, or other media objects
    # directly. Avoid bool(value), which can fail for multi-element arrays/tensors
    # before they reach the preprocessor.
    return True


def _looks_like_openai_audio_output_config(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    keys = set(value)
    if keys & _AUDIO_MEDIA_PAYLOAD_KEYS:
        return False
    return bool(keys & _OPENAI_AUDIO_OUTPUT_CONFIG_KEYS)


class Client:
    """Internal client used by API adapters."""

    def __init__(
        self,
        coordinator: Coordinator,
        result_builder: Callable[[str, Any], GenerateChunk] | None = None,
        stream_builder: Callable[[str, StreamMessage], GenerateChunk] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._result_builder = result_builder or self._default_result_builder
        self._stream_builder = stream_builder or self._default_stream_builder

    # ------------------------------------------------------------------
    # Low-level generate (backward compatible)
    # ------------------------------------------------------------------

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        req_id = request_id or str(uuid.uuid4())
        omni_request = self._build_omni_request(request)
        if request.stream:
            async for msg in self._coordinator.stream(req_id, omni_request):
                if isinstance(msg, StreamMessage):
                    yield self._stream_builder(req_id, msg)
                else:
                    yield self._result_builder(req_id, msg.result)
            return

        result = await self._coordinator.submit(req_id, omni_request)
        yield self._result_builder(req_id, result)

    # ------------------------------------------------------------------
    # High-level: non-streaming completion
    # ------------------------------------------------------------------

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        """Run a non-streaming completion and return an aggregated result.

        Iterates ``generate()``, accumulates text, concatenates audio chunks,
        and encodes audio to base64.

        Raises:
            ClientError: If the pipeline produces no response at all.
        """
        text_parts: list[str] = []
        audio_chunks: list[Any] = []
        sample_rate: int | None = None
        last_chunk: GenerateChunk | None = None
        finish_reason: str | None = None

        async for chunk in self.generate(request, request_id=request_id):
            last_chunk = chunk
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            if chunk.sample_rate is not None:
                sample_rate = chunk.sample_rate
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        if last_chunk is None:
            raise ClientError("No response from pipeline")

        full_text = "".join(text_parts)

        audio: CompletionAudio | None = None
        if audio_chunks:
            if len(audio_chunks) == 1:
                combined = audio_chunks[0]
            else:
                combined = np.concatenate([to_numpy(c) for c in audio_chunks])
            # The code2wav stage returns the real sample_rate with audio chunks.
            # Chat completion must encode with it as well; otherwise non-24k
            # profiles play at the wrong speed/duration. The speech path already
            # passes the same value through.
            audio_b64, actual_format = _encode_audio_base64_and_format(
                combined,
                output_format=audio_format,
                sample_rate=sample_rate,
            )
            audio = CompletionAudio(
                id=f"audio-{request_id}",
                data=audio_b64,
                format=actual_format,
                sample_rate=sample_rate,
                transcript=full_text if full_text else None,
            )

        return CompletionResult(
            request_id=request_id,
            text=full_text,
            audio=audio,
            finish_reason=finish_reason or "stop",
            usage=last_chunk.usage,
        )

    # ------------------------------------------------------------------
    # High-level: streaming completion
    # ------------------------------------------------------------------

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        """Iterate ``generate()`` and yield high-level stream chunks.

        Audio data is base64-encoded before yielding so that callers never
        need to touch numpy / raw bytes.
        """
        async for chunk in self.generate(request, request_id=request_id):
            audio_b64: str | None = None
            actual_audio_format: str | None = None
            if chunk.modality == "audio" and chunk.audio_data is not None:
                audio_b64, actual_audio_format = _encode_audio_base64_and_format(
                    chunk.audio_data,
                    output_format=audio_format,
                    sample_rate=chunk.sample_rate,
                )

            yield CompletionStreamChunk(
                request_id=request_id,
                text=chunk.text,
                modality=chunk.modality,
                audio_b64=audio_b64,
                audio_format=actual_audio_format,
                sample_rate=chunk.sample_rate,
                finish_reason=chunk.finish_reason,
                usage=chunk.usage,
                stage_name=chunk.stage_name,
            )

    # ------------------------------------------------------------------
    # High-level: text-to-speech
    # ------------------------------------------------------------------

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> SpeechResult:
        """Run a TTS request and return encoded audio bytes.

        Raises:
            ClientError: If the pipeline produces no audio output.
        """
        audio_chunks: list[Any] = []
        sample_rate: int | None = None
        last_chunk: GenerateChunk | None = None
        extra_params = dict(request.extra_params)
        extra_params.pop("stream", None)
        request = replace(request, stream=False, extra_params=extra_params)

        async for chunk in self.generate(request, request_id=request_id):
            if chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            if chunk.sample_rate is not None:
                sample_rate = chunk.sample_rate
            last_chunk = chunk

        if not audio_chunks:
            raise ClientError("No audio output generated from the pipeline.")

        if len(audio_chunks) == 1:
            audio_data = audio_chunks[0]
        else:
            audio_data = np.concatenate([to_numpy(c) for c in audio_chunks])

        encode_kwargs: dict[str, Any] = {
            "response_format": response_format,
            "speed": speed,
        }
        if sample_rate is not None:
            encode_kwargs["sample_rate"] = sample_rate

        audio_bytes, mime_type = encode_audio(audio_data, **encode_kwargs)

        # Derive actual format from MIME type (encode_audio may fall back
        # to WAV if the requested codec is unavailable).
        actual_format = response_format
        for ext, mt in FORMAT_MIME_TYPES.items():
            if mt == mime_type:
                actual_format = ext
                break

        return SpeechResult(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            format=actual_format,
            usage=last_chunk.usage if last_chunk else None,
        )

    # ------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        success = await self._coordinator.abort(request_id)
        return AbortResult(success=success, level_applied=level)

    async def get_status(self, request_id: str) -> RequestState | None:
        info = self._coordinator.get_request_info(request_id)
        if info is None:
            return None
        return info.state

    def health(self) -> dict[str, Any]:
        return self._coordinator.health()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _set_audio_data(chunk: GenerateChunk, data: dict[str, Any]) -> None:
        audio_data = data.get("audio_data") or data.get("audio")
        if audio_data is None and data.get("audio_waveform") is not None:
            raw = data.get("audio_waveform")
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            dtype = np.dtype(data.get("audio_waveform_dtype", "float32"))
            arr = np.frombuffer(raw, dtype=dtype)
            shape = data.get("audio_waveform_shape")
            if shape:
                arr = arr.reshape(shape)
            audio_data = arr.copy()
        if audio_data is not None:
            chunk.audio_data = audio_data
            chunk.modality = "audio"
        sample_rate = data.get("sample_rate")
        if sample_rate is not None:
            chunk.sample_rate = sample_rate

    @staticmethod
    def _build_usage_info(data: dict[str, Any]) -> UsageInfo | None:
        usage = dict(data.get("usage") or {})
        if "prompt_tokens" not in usage and data.get("prompt_tokens") is not None:
            usage["prompt_tokens"] = data.get("prompt_tokens")
        if (
            "completion_tokens" not in usage
            and data.get("completion_tokens") is not None
        ):
            usage["completion_tokens"] = data.get("completion_tokens")
        if "total_tokens" not in usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is not None or completion_tokens is not None:
                usage["total_tokens"] = (prompt_tokens or 0) + (completion_tokens or 0)
        if "engine_time_s" not in usage and data.get("engine_time_s") is not None:
            usage["engine_time_s"] = data.get("engine_time_s")
        return UsageInfo.from_dict(usage)

    @staticmethod
    def _build_omni_request(request: GenerateRequest) -> OmniRequest:
        inputs = _extract_inputs(request)
        params = _build_params(request)
        metadata = dict(request.metadata)
        if "audio_config" not in metadata and _looks_like_openai_audio_output_config(
            metadata.get("audio")
        ):
            # The internal Client can also receive OpenAI-style
            # metadata.audio={voice,format,...}. The request builder reads
            # audio_config, so copy it there to avoid keeping voice/language
            # settings only in tracing metadata and missing talker param
            # normalization.
            metadata["audio_config"] = metadata["audio"]
        if request.model:
            metadata.setdefault("model", request.model)
        if request.output_modalities:
            metadata["output_modalities"] = request.output_modalities
        return OmniRequest(inputs=inputs, params=params, metadata=metadata)

    @staticmethod
    def _default_result_builder(request_id: str, result: Any) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id, finish_reason="stop")
        if isinstance(result, GenerateChunk):
            result.request_id = request_id
            return result
        if isinstance(result, dict):
            # Multi-terminal merged result, e.g. decode + code2wav/talker/
            # talker_stream.
            audio_result = None
            if "decode" in result:
                for audio_stage in ("code2wav", "talker", "talker_stream"):
                    if audio_stage in result:
                        audio_result = result[audio_stage] or {}
                        break
            if audio_result is not None:
                decode_result = result["decode"] or {}
                text = decode_result.get("text")
                if isinstance(text, str):
                    chunk.text = text
                finish_reason = decode_result.get("finish_reason")
                if finish_reason is not None:
                    chunk.finish_reason = finish_reason
                Client._set_audio_data(chunk, audio_result)
                chunk.usage = Client._build_usage_info(
                    decode_result
                ) or Client._build_usage_info(audio_result)
                return chunk
            text = result.get("text")
            if isinstance(text, str):
                chunk.text = text
            token_ids = result.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    token_ids = token_ids.tolist()
                chunk.token_ids = list(token_ids)
            logprobs = result.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            finish_reason = result.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            chunk.stage_id = result.get("stage_id")
            chunk.stage_name = result.get("stage_name")
            modality = result.get("modality")
            if modality is not None:
                chunk.modality = modality
            Client._set_audio_data(chunk, result)
            chunk.usage = Client._build_usage_info(result)
            return chunk
        if isinstance(result, str):
            chunk.text = result
            return chunk
        chunk.text = str(result)
        return chunk

    @staticmethod
    def _default_stream_builder(request_id: str, msg: StreamMessage) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id)
        chunk.stage_name = msg.stage_name or msg.from_stage
        chunk.stage_id = msg.stage_id
        if msg.modality:
            chunk.modality = msg.modality

        data = msg.chunk
        if isinstance(data, GenerateChunk):
            data.request_id = request_id
            if data.stage_name is None:
                data.stage_name = chunk.stage_name
            if data.stage_id is None:
                data.stage_id = chunk.stage_id
            if not data.modality and chunk.modality:
                data.modality = chunk.modality
            return data
        if isinstance(data, dict):
            text = data.get("text")
            if isinstance(text, str):
                chunk.text = text
            token_ids = data.get("token_ids")
            if token_ids is not None:
                if not isinstance(token_ids, (list, tuple)):
                    token_ids = token_ids.tolist()
                chunk.token_ids = list(token_ids)
            logprobs = data.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            finish_reason = data.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            chunk.usage = Client._build_usage_info(data)
            stage_name = data.get("stage_name")
            if stage_name is not None:
                chunk.stage_name = stage_name
            stage_id = data.get("stage_id")
            if stage_id is not None:
                chunk.stage_id = stage_id
            modality = data.get("modality")
            if modality is not None:
                chunk.modality = modality
            Client._set_audio_data(chunk, data)
            return chunk
        if isinstance(data, str):
            chunk.text = data
            return chunk
        if isinstance(data, int):
            chunk.token_ids = [data]
            return chunk
        chunk.text = str(data)
        return chunk


def _extract_inputs(request: GenerateRequest) -> Any:
    choices = [
        request.prompt is not None,
        request.prompt_token_ids is not None,
        request.messages is not None,
    ]
    if sum(choices) != 1:
        raise ValueError(
            "GenerateRequest requires exactly one input: "
            "prompt, prompt_token_ids, or messages."
        )
    media_and_options = _extract_multimodal_metadata(request)
    if request.prompt is not None:
        if media_and_options:
            return {"prompt": request.prompt, **media_and_options}
        return request.prompt
    if request.prompt_token_ids is not None:
        prompt_token_ids = list(request.prompt_token_ids)
        if media_and_options:
            return {"prompt_token_ids": prompt_token_ids, **media_and_options}
        return prompt_token_ids

    # Build messages list
    messages = [msg.to_dict() for msg in request.messages or []]

    # If we have media/options, return a dict with messages and media.
    # Otherwise, return just the messages list (for backward compatibility).
    if media_and_options:
        return {"messages": messages, **media_and_options}
    return messages


def _extract_multimodal_metadata(request: GenerateRequest) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _MEDIA_METADATA_KEYS:
        value = request.metadata.get(key)
        if key == "audio" and _looks_like_openai_audio_output_config(value):
            # OpenAI chat/completions metadata.audio is output-audio config
            # (voice/format/language), not audio encoder input. Real input audio
            # can still be passed through audios/input_audio/audio_url.
            continue
        if _metadata_value_is_present(value):
            result[key] = value
    for key in _MULTIMODAL_METADATA_OPTION_KEYS:
        value = request.metadata.get(key)
        if value is not None:
            # prompt/prompt_token_ids may also carry multimodal input. These
            # decode/processor params must enter inputs with the media; otherwise
            # the preprocessor cannot see them and real Qwen3.5 requests lose
            # fps/audio-track details.
            result[key] = value
    return result


def _build_params(request: GenerateRequest) -> dict[str, Any]:
    params = request.sampling.to_dict()
    max_new_tokens = request.sampling.max_new_tokens
    if request.max_tokens is not None:
        max_new_tokens = request.max_tokens
    if max_new_tokens is None:
        params.pop("max_new_tokens", None)
    else:
        params["max_new_tokens"] = max_new_tokens
    params["stream"] = request.stream
    if request.stage_sampling:
        params["stage_sampling"] = {
            key: value.to_dict() for key, value in request.stage_sampling.items()
        }
    if request.stage_params:
        params["stage_params"] = request.stage_params
    if request.extra_params:
        params.update(request.extra_params)
    return params
