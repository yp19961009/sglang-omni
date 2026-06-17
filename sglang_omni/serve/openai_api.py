# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/chat/completions  — Text (+ audio) chat completions
- POST /v1/audio/speech      — Text-to-speech synthesis
- GET  /v1/models            — List available models
- GET  /v1/fs/list           — Browse filesystem directories
- GET  /v1/fs/file           — Download a file
- GET  /health               — Health check
- WS   /v1/realtime          — OpenAI-compatible Realtime API (when enabled)
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from sglang_omni.client import (
    Client,
    ClientError,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    FORMAT_MIME_TYPES,
    apply_speed,
    encode_audio,
    encode_pcm,
    to_numpy,
)
from sglang_omni.http.favicon import register_favicon
from sglang_omni.models.tts_streaming import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.serve.protocol import (
    ChatCompletionAudio,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    CreateSpeechRequest,
    ModelCard,
    ModelList,
    TranscriptionResponse,
    UsageResponse,
)

logger = logging.getLogger(__name__)
MIME_TO_FORMAT = {mime: fmt for fmt, mime in FORMAT_MIME_TYPES.items()}
STREAM_DONE_SENTINEL = "[DONE]"
RAW_PCM_DEFAULT_INITIAL_CODEC_CHUNK_FRAMES = 1

_BAD_REQUEST_MARKERS = (
    "longer than the model's context length",
    "Requested token count exceeds the model's maximum context length",
)


def _is_bad_request_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS)


def _default_generation_value(
    default_generation_params: dict[str, Any] | None,
    key: str,
) -> Any:
    if not default_generation_params:
        return None
    return default_generation_params.get(key)


def _request_or_default_generation_value(
    req: Any,
    key: str,
    default_generation_params: dict[str, Any] | None = None,
) -> Any:
    value = getattr(req, key, None)
    if value is not None:
        return value
    return _default_generation_value(default_generation_params, key)


def _chat_effective_max_tokens(
    req: ChatCompletionRequest,
    default_generation_params: dict[str, Any] | None = None,
) -> Any:
    if req.max_completion_tokens is not None:
        return req.max_completion_tokens
    if req.max_tokens is not None:
        return req.max_tokens
    default_max_completion_tokens = _default_generation_value(
        default_generation_params,
        "max_completion_tokens",
    )
    if default_max_completion_tokens is not None:
        return default_max_completion_tokens
    return _default_generation_value(default_generation_params, "max_tokens")


def _stage_sampling_overrides(raw: Any) -> dict[str, SamplingParams] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("stage_sampling must be a mapping of stage name to params")

    stage_sampling: dict[str, SamplingParams] = {}
    for stage_name, params_dict in raw.items():
        if isinstance(params_dict, SamplingParams):
            stage_sampling[str(stage_name)] = params_dict
            continue
        if not isinstance(params_dict, dict):
            raise ValueError(
                "stage_sampling values must be SamplingParams or mappings"
            )
        stage_sampling[str(stage_name)] = SamplingParams(**params_dict)
    return stage_sampling or None


def _stage_param_overrides(raw: Any) -> dict[str, dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("stage_params must be a mapping of stage name to params")

    stage_params: dict[str, dict[str, Any]] = {}
    for stage_name, params in raw.items():
        if not isinstance(params, dict):
            raise ValueError("stage_params values must be mappings")
        stage_params[str(stage_name)] = dict(params)
    return stage_params or None


def _chat_audio_config(req: ChatCompletionRequest) -> dict[str, Any] | None:
    if isinstance(req.audio, dict):
        return req.audio
    return None


def _request_metadata(req: ChatCompletionRequest) -> dict[str, Any]:
    metadata = req.metadata if isinstance(req.metadata, dict) else None
    metadata = dict(metadata or {})
    for key in ("request_id", "user"):
        value = getattr(req, key, None)
        if value is not None:
            metadata[key] = value
    return metadata


def _legacy_functions_as_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            tools.append(dict(item))
        else:
            tools.append({"type": "function", "function": dict(item)})
    return tools


def _requested_modalities(req: ChatCompletionRequest) -> list[str]:
    return list(req.modalities) if req.modalities else ["text"]


def _stream_include_usage(req: ChatCompletionRequest) -> bool:
    stream_options = req.stream_options
    if not isinstance(stream_options, dict):
        return False
    return bool(stream_options.get("include_usage"))


def _default_talker_extra_params(
    default_talker_params: dict[str, Any] | None,
) -> dict[str, Any]:
    if not default_talker_params:
        return {}
    return {
        key: value for key, value in default_talker_params.items() if value is not None
    }


def _chat_default_talker_extra_params(
    req: ChatCompletionRequest,
    default_talker_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = _default_talker_extra_params(default_talker_params)
    audio_config = _chat_audio_config(req)
    if not isinstance(audio_config, dict):
        return params

    override_groups = (
        (("voice_type", "voice", "speaker"), ("voice_type", "voice", "speaker")),
        (
            (
                "language",
                "lang",
                "language_id",
                "language_type",
                "target_language",
                "target_lang",
            ),
            (
                "language",
                "lang",
                "language_id",
                "language_type",
                "target_language",
                "target_lang",
            ),
        ),
        (("voice_style", "style"), ("voice_style", "style")),
        (
            (
                "instruction",
                "talker_instruction",
                "assistant_instruct_ids",
                "talker_assistant_prompt_ids",
                "voice_style_ids",
                "instruct_ids",
            ),
            ("instruction", "talker_instruction"),
        ),
        (
            (
                "xvector_info",
                "voice_clone_info",
                "xvector_info_path",
                "voice_clone_path",
                "voice_clone",
            ),
            (
                "xvector_info",
                "voice_clone_info",
                "xvector_info_path",
                "voice_clone_path",
                "voice_clone",
            ),
        ),
    )
    for default_keys, audio_keys in override_groups:
        if any(audio_config.get(key) is not None for key in audio_keys):
            for key in default_keys:
                params.pop(key, None)
    return params


def _speech_effective_max_new_tokens(
    req: CreateSpeechRequest,
    default_generation_params: dict[str, Any] | None = None,
) -> Any:
    if req.max_new_tokens is not None:
        return req.max_new_tokens
    if req.max_completion_tokens is not None:
        return req.max_completion_tokens
    if req.max_tokens is not None:
        return req.max_tokens
    default_max_new_tokens = _default_generation_value(
        default_generation_params,
        "max_new_tokens",
    )
    if default_max_new_tokens is not None:
        return default_max_new_tokens
    default_max_completion_tokens = _default_generation_value(
        default_generation_params,
        "max_completion_tokens",
    )
    if default_max_completion_tokens is not None:
        return default_max_completion_tokens
    return _default_generation_value(default_generation_params, "max_tokens")


def create_app(
    client: Client,
    *,
    model_name: str | None = None,
    enable_realtime: bool = False,
    default_talker_params: dict[str, Any] | None = None,
    default_generation_params: dict[str, Any] | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        client: Client instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.
        enable_realtime: If True, mount the WebSocket ``/v1/realtime``
            endpoint (OpenAI Realtime API).
        default_talker_params: Service-level defaults merged into
            ``GenerateRequest.extra_params`` for audio-output requests.
        default_generation_params: Service-level sampling/max-token defaults.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="sglang-omni", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references in app state for access from route handlers
    app.state.client = client
    app.state.model_name = model_name or "sglang-omni"
    app.state.realtime_enabled = enable_realtime
    app.state.default_talker_params = dict(default_talker_params or {})
    app.state.default_generation_params = dict(default_generation_params or {})

    # Register all routes
    register_favicon(app)
    _register_health(app)
    _register_models(app)
    _register_chat_completions(app)
    _register_speech(app)
    _register_transcriptions(app)
    if enable_realtime:
        _register_realtime(app)

    return app


def _register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint (includes filesystem browse info)."""
        client: Client = app.state.client
        info = client.health()
        is_running = info.get("running", False)
        status_code = 200 if is_running else 503
        return JSONResponse(
            content={
                "status": "healthy" if is_running else "unhealthy",
                **info,
            },
            status_code=status_code,
        )


def _register_models(app: FastAPI) -> None:
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        model_name: str = app.state.model_name
        model_list = ModelList(
            data=[
                ModelCard(
                    id=model_name,
                    root=model_name,
                    created=0,
                )
            ]
        )
        return JSONResponse(content=model_list.model_dump())


def _register_chat_completions(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        default_talker_params: dict[str, Any] = app.state.default_talker_params
        default_generation_params: dict[str, Any] = (
            app.state.default_generation_params
        )

        request_id = req.request_id or str(uuid.uuid4())
        response_id = f"chatcmpl-{request_id}"
        created = int(time.time())
        model = req.model or default_model

        gen_req = _build_chat_generate_request(
            req,
            default_model=default_model,
            default_talker_params=default_talker_params,
            default_generation_params=default_generation_params,
        )

        # Determine audio format from request
        audio_format = "wav"
        audio_config = _chat_audio_config(req)
        if audio_config is not None:
            audio_format = audio_config.get("format", "wav")

        if req.stream:
            return StreamingResponse(
                _chat_stream(
                    client,
                    gen_req,
                    request_id,
                    response_id,
                    created,
                    model,
                    req,
                    audio_format,
                ),
                media_type="text/event-stream",
            )

        return await _chat_non_stream(
            client,
            gen_req,
            request_id,
            response_id,
            created,
            model,
            req,
            audio_format,
        )


async def _chat_non_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
    audio_format: str,
) -> JSONResponse:
    """Handle non-streaming chat completions."""
    try:
        result = await client.completion(
            gen_req,
            request_id=request_id,
            audio_format=audio_format,
        )
    except ClientError as exc:
        if _is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error generating response for request %s", request_id)
        if _is_bad_request_error(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    requested_modalities = _requested_modalities(req)

    # Build message content
    message: dict[str, Any] = {"role": "assistant"}

    if "text" in requested_modalities and result.text:
        message["content"] = result.text

    if "audio" in requested_modalities and result.audio is not None:
        message["audio"] = {
            "id": result.audio.id,
            "data": result.audio.data,
            "format": result.audio.format or audio_format,
            "transcript": result.audio.transcript,
        }
        if result.audio.sample_rate is not None:
            # 中文说明：code2wav 返回真实 sample_rate 后，非流式 chat
            # 也把它透给调用方；旧路径没有该字段时保持响应兼容。
            message["audio"]["sample_rate"] = result.audio.sample_rate

    if "content" not in message and "audio" not in message:
        message["content"] = result.text

    # Build usage
    usage = None
    if result.usage is not None:
        usage = UsageResponse(
            prompt_tokens=result.usage.prompt_tokens or 0,
            completion_tokens=result.usage.completion_tokens or 0,
            total_tokens=result.usage.total_tokens or 0,
        )

    response = ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=result.finish_reason or "stop",
            )
        ],
        usage=usage,
    )

    return JSONResponse(content=response.model_dump())


async def _chat_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
    audio_format: str,
):
    """Streaming chat completion generator (yields SSE events)."""
    role_sent = False
    requested_modalities = _requested_modalities(req)
    include_usage_chunk = _stream_include_usage(req)
    finish_reason: str | None = None
    final_usage: UsageResponse | None = None

    async for chunk in client.completion_stream(
        gen_req,
        request_id=request_id,
        audio_format=audio_format,
    ):
        # Capture finish info for the dedicated finish chunk after the loop.
        # Some pipelines only emit a final aggregate chunk; do not drop its
        # text/audio just because it already carries a finish reason.
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                final_usage = UsageResponse(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )
            has_payload = (
                chunk.modality == "text"
                and bool(chunk.text)
                and "text" in requested_modalities
            ) or (
                chunk.modality == "audio"
                and chunk.audio_b64 is not None
                and "audio" in requested_modalities
            )
            if not has_payload:
                continue

        delta = ChatCompletionStreamDelta()
        emit = False

        # Send role on first chunk
        if not role_sent:
            delta.role = "assistant"
            role_sent = True
            emit = True

        # Text chunk
        if chunk.modality == "text" and chunk.text and "text" in requested_modalities:
            delta.content = chunk.text
            emit = True

        # Audio chunk
        if (
            chunk.modality == "audio"
            and chunk.audio_b64 is not None
            and "audio" in requested_modalities
        ):
            delta.audio = ChatCompletionAudio(
                id=f"audio-{request_id}",
                data=chunk.audio_b64,
                format=chunk.audio_format or audio_format,
                sample_rate=chunk.sample_rate,
            )
            emit = True

        if not emit:
            continue

        stream_resp = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

        data = stream_resp.model_dump(exclude_none=True)
        for choice in data.get("choices", []):
            choice.setdefault("finish_reason", None)
        yield f"data: {json.dumps(data)}\n\n"

    # Finish chunk: empty delta + finish_reason.
    finish_resp = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionStreamDelta(),
                finish_reason=finish_reason or "stop",
            )
        ],
        usage=None if include_usage_chunk else final_usage,
    )
    data = finish_resp.model_dump(exclude_none=True)
    for choice in data.get("choices", []):
        choice.setdefault("finish_reason", None)
    yield f"data: {json.dumps(data)}\n\n"

    if include_usage_chunk and final_usage is not None:
        usage_resp = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[],
            usage=final_usage,
        )
        yield f"data: {json.dumps(usage_resp.model_dump(exclude_none=True))}\n\n"

    yield f"data: {STREAM_DONE_SENTINEL}\n\n"


def _build_chat_generate_request(
    req: ChatCompletionRequest,
    *,
    default_model: str | None = None,
    default_talker_params: dict[str, Any] | None = None,
    default_generation_params: dict[str, Any] | None = None,
) -> GenerateRequest:
    """Convert a ChatCompletionRequest into a client GenerateRequest."""
    # Parse stop sequences
    stop: list[str] = []
    stop_value = _request_or_default_generation_value(
        req,
        "stop",
        default_generation_params,
    )
    if isinstance(stop_value, str):
        stop = [stop_value]
    elif isinstance(stop_value, list):
        stop = list(stop_value)

    temperature = _request_or_default_generation_value(
        req,
        "temperature",
        default_generation_params,
    )
    top_p = _request_or_default_generation_value(
        req,
        "top_p",
        default_generation_params,
    )
    top_k = _request_or_default_generation_value(
        req,
        "top_k",
        default_generation_params,
    )
    min_p = _request_or_default_generation_value(
        req,
        "min_p",
        default_generation_params,
    )
    repetition_penalty = _request_or_default_generation_value(
        req,
        "repetition_penalty",
        default_generation_params,
    )
    frequency_penalty = _request_or_default_generation_value(
        req,
        "frequency_penalty",
        default_generation_params,
    )
    presence_penalty = _request_or_default_generation_value(
        req,
        "presence_penalty",
        default_generation_params,
    )
    seed = _request_or_default_generation_value(
        req,
        "seed",
        default_generation_params,
    )
    max_tokens = _chat_effective_max_tokens(req, default_generation_params)

    # Build sampling params
    sampling = SamplingParams(
        temperature=temperature if temperature is not None else 1.0,
        top_p=top_p if top_p is not None else 1.0,
        top_k=top_k if top_k is not None else -1,
        min_p=min_p if min_p is not None else 0.0,
        repetition_penalty=(
            repetition_penalty if repetition_penalty is not None else 1.0
        ),
        frequency_penalty=(
            frequency_penalty if frequency_penalty is not None else 0.0
        ),
        presence_penalty=(
            presence_penalty if presence_penalty is not None else 0.0
        ),
        stop=stop,
        seed=seed,
        max_new_tokens=max_tokens,
    )

    # Convert messages. Keep OpenAI tool-call metadata for chat templates that
    # render multi-turn function-call or MCP conversations.
    messages = [
        Message(
            role=m.role,
            content=m.content,
            name=m.name,
            tool_calls=m.tool_calls,
            tool_call_id=m.tool_call_id,
            function_call=m.function_call,
        )
        for m in req.messages
    ]

    # Determine output modalities
    output_modalities = _requested_modalities(req)  # e.g. ["text", "audio"]

    stage_sampling = _stage_sampling_overrides(req.stage_sampling)
    stage_params = _stage_param_overrides(req.stage_params)

    # Extract audios, images, and videos from request
    audios = list(req.audios) if req.audios else None
    images = list(req.images) if req.images else None
    videos = list(req.videos) if req.videos else None

    # Merge audio config, audios, images, and videos into metadata.
    metadata: dict[str, Any] = _request_metadata(req)
    audio_config = _chat_audio_config(req)
    if audio_config is not None:
        metadata["audio_config"] = audio_config
    tools = req.tools
    if tools is None:
        tools = _legacy_functions_as_tools(req.functions)
    if tools is not None:
        # 中文说明：tools 只用于 chat template 渲染，不在服务端执行。
        metadata["tools"] = tools
    for key in ("tool_choice", "parallel_tool_calls", "function_call"):
        value = getattr(req, key, None)
        if value is not None:
            metadata[key] = value
    if audios:
        metadata["audios"] = audios
    if images:
        metadata["images"] = images
    if videos:
        metadata["videos"] = videos
    for key in (
        "image_min_pixels",
        "image_max_pixels",
        "video_fps",
        "video_min_frames",
        "video_max_frames",
        "video_min_pixels",
        "video_max_pixels",
        "video_total_pixels",
        "use_audio_in_video",
        "dependent_audio",
        "video_dependent_audio",
        "video_seconds_per_chunk",
        "video_position_id_per_seconds",
        "return_video_metadata",
        "video_metadata",
        "audio_target_sr",
        "audio_timestamp_interval",
        "audio_downsample_times",
        "audio_downsample_chunk_size",
    ):
        value = getattr(req, key, None)
        if value is not None:
            metadata[key] = value

    extra_params = _chat_default_talker_extra_params(req, default_talker_params)
    for field_name in (
        "talker_temperature",
        "talker_top_p",
        "talker_top_k",
        "talker_repetition_penalty",
        "talker_max_new_tokens",
    ):
        value = getattr(req, field_name, None)
        if value is not None:
            extra_params[field_name] = value

    return GenerateRequest(
        model=req.model or default_model,
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=stage_params,
        extra_params=extra_params,
        stream=req.stream,
        max_tokens=max_tokens,
        output_modalities=output_modalities,
        metadata=metadata,
    )


def _register_realtime(app: FastAPI) -> None:
    """Mount the OpenAI-compatible WebSocket Realtime endpoint."""
    from sglang_omni.serve.realtime import RealtimeSessionManager

    client: Client = app.state.client
    model_name: str = app.state.model_name
    manager = RealtimeSessionManager(client=client, model_name=model_name)
    app.state.realtime_manager = manager

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        session = manager.open(websocket)
        try:
            await session.run()
        finally:
            await manager.close(session.session_id)


def _register_speech(app: FastAPI) -> None:
    @app.post("/v1/audio/speech")
    async def create_speech(req: CreateSpeechRequest) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        default_talker_params: dict[str, Any] = app.state.default_talker_params
        default_generation_params: dict[str, Any] = (
            app.state.default_generation_params
        )

        request_id = f"speech-{uuid.uuid4()}"

        gen_req = build_speech_generate_request(
            req,
            default_model,
            default_talker_params=default_talker_params,
            default_generation_params=default_generation_params,
        )
        if req.stream:
            if req.stream_format == "audio":
                try:
                    return await _speech_audio_response(
                        client=client,
                        gen_req=gen_req,
                        request_id=request_id,
                        speed=req.speed,
                    )
                except ClientError as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                except Exception as exc:
                    logger.exception(
                        "Error preparing raw PCM speech stream for request %s",
                        request_id,
                    )
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
            return StreamingResponse(
                _speech_stream(
                    client=client,
                    gen_req=gen_req,
                    request_id=request_id,
                    response_format=req.response_format,
                    speed=req.speed,
                ),
                media_type="text/event-stream",
            )

        try:
            result = await client.speech(
                gen_req,
                request_id=request_id,
                response_format=req.response_format,
                speed=req.speed,
            )
        except ClientError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error generating speech for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        headers = {
            "Content-Disposition": f'attachment; filename="speech.{result.format}"',
        }
        if result.usage is not None:
            if result.usage.prompt_tokens is not None:
                headers["X-Prompt-Tokens"] = str(result.usage.prompt_tokens)
            if result.usage.completion_tokens is not None:
                headers["X-Completion-Tokens"] = str(result.usage.completion_tokens)
            if result.usage.engine_time_s is not None:
                headers["X-Engine-Time"] = str(result.usage.engine_time_s)

        return Response(
            content=result.audio_bytes,
            media_type=result.mime_type,
            headers=headers,
        )


async def _speech_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_format: str,
    speed: float,
):
    """Streaming speech generator (yields SSE events with audio chunks)."""
    chunk_index = 0
    emitted_samples = 0
    finish_reason: str | None = None
    usage: dict | None = None

    async for chunk in client.generate(gen_req, request_id=request_id):
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                usage = chunk.usage.to_dict()

        if chunk.audio_data is None:
            continue

        sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
        audio_data, emitted_samples = _select_speech_audio_delta(
            chunk.audio_data,
            emitted_samples=emitted_samples,
            is_terminal=chunk.finish_reason is not None,
        )
        if audio_data is None:
            continue

        audio_bytes, mime_type = encode_audio(
            audio_data,
            response_format=response_format,
            sample_rate=sample_rate,
            speed=speed,
        )
        actual_format = MIME_TO_FORMAT.get(mime_type, response_format)
        payload = {
            "id": f"speech-{request_id}",
            "object": "audio.speech.chunk",
            "index": chunk_index,
            "audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": actual_format,
                "mime_type": mime_type,
                "sample_rate": sample_rate,
            },
            "finish_reason": None,
        }
        yield f"data: {json.dumps(payload)}\n\n"
        chunk_index += 1

    final_payload = {
        "id": f"speech-{request_id}",
        "object": "audio.speech.chunk",
        "index": chunk_index,
        "audio": None,
        "finish_reason": finish_reason or "stop",
        "usage": usage,
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield f"data: {STREAM_DONE_SENTINEL}\n\n"


def _speech_pcm_chunk_bytes(
    chunk: Any,
    *,
    emitted_samples: int,
    speed: float,
) -> tuple[bytes | None, int, int]:
    sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
    audio_data, emitted_samples = _select_speech_audio_delta(
        chunk.audio_data,
        emitted_samples=emitted_samples,
        is_terminal=chunk.finish_reason is not None,
    )
    if audio_data is None:
        return None, emitted_samples, sample_rate

    if speed != 1.0:
        audio_data, sample_rate = apply_speed(audio_data, speed, sample_rate)
    return encode_pcm(audio_data, sample_rate), emitted_samples, sample_rate


async def _speech_audio_response(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    speed: float,
) -> StreamingResponse:
    """Build a raw PCM stream after deriving headers from the first audio chunk."""
    emitted_samples = 0
    chunk_stream = client.generate(gen_req, request_id=request_id)
    first_audio_bytes: bytes | None = None
    stream_sample_rate: int | None = None

    async for chunk in chunk_stream:
        if chunk.audio_data is None:
            continue

        first_audio_bytes, emitted_samples, stream_sample_rate = (
            _speech_pcm_chunk_bytes(
                chunk,
                emitted_samples=emitted_samples,
                speed=speed,
            )
        )
        if first_audio_bytes is not None:
            break

    if first_audio_bytes is None or stream_sample_rate is None:
        raise RuntimeError("No audio chunks received from raw PCM speech stream")

    async def _body():
        nonlocal emitted_samples
        yield first_audio_bytes

        async for chunk in chunk_stream:
            if chunk.audio_data is None:
                continue

            audio_bytes, emitted_samples, sample_rate = _speech_pcm_chunk_bytes(
                chunk,
                emitted_samples=emitted_samples,
                speed=speed,
            )
            if audio_bytes is None:
                continue
            if sample_rate != stream_sample_rate:
                raise RuntimeError(
                    "Raw PCM speech stream sample rate changed from "
                    f"{stream_sample_rate} to {sample_rate}"
                )
            yield audio_bytes

    return StreamingResponse(
        _body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(stream_sample_rate),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


def _select_speech_audio_delta(
    audio_data: Any,
    *,
    emitted_samples: int,
    is_terminal: bool,
) -> tuple[Any | None, int]:
    audio = to_numpy(audio_data)
    if audio.ndim > 1:
        audio = audio.squeeze()
    if audio.ndim > 1:
        if audio.shape[0] < audio.shape[-1]:
            audio = audio[0]
        else:
            audio = audio[:, 0]

    total_samples = int(audio.shape[-1]) if audio.ndim else 0
    if not is_terminal:
        return audio, emitted_samples + total_samples
    if total_samples <= emitted_samples:
        return None, emitted_samples
    return audio[emitted_samples:], total_samples


def build_speech_generate_request(
    req: CreateSpeechRequest,
    default_model: str,
    *,
    default_talker_params: dict[str, Any] | None = None,
    default_generation_params: dict[str, Any] | None = None,
) -> GenerateRequest:
    """Convert a CreateSpeechRequest into a client GenerateRequest."""

    generation_fields = (
        "max_tokens",
        "max_completion_tokens",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
    )
    explicit_generation_params = sorted(
        field
        for field in generation_fields
        if getattr(req, field, None) is not None
    )
    initial_codec_chunk_frames = req.initial_codec_chunk_frames
    if (
        initial_codec_chunk_frames is None
        and req.stream
        and req.stream_format == "audio"
    ):
        initial_codec_chunk_frames = RAW_PCM_DEFAULT_INITIAL_CODEC_CHUNK_FRAMES

    # Build TTS-specific parameters to pass through the pipeline
    tts_params: dict[str, Any] = {
        "voice": req.voice,
        "response_format": req.response_format,
        "speed": req.speed,
    }
    if explicit_generation_params:
        tts_params["explicit_generation_params"] = explicit_generation_params
    if req.task_type is not None:
        tts_params["task_type"] = req.task_type
    if req.language is not None:
        tts_params["language"] = req.language
    if req.instructions is not None:
        tts_params["instructions"] = req.instructions
    if req.ref_audio is not None:
        tts_params["ref_audio"] = req.ref_audio
    if req.ref_text is not None:
        tts_params["ref_text"] = req.ref_text
    if req.token_count is not None:
        tts_params["token_count"] = req.token_count
    if req.duration_tokens is not None:
        tts_params["duration_tokens"] = req.duration_tokens
    extra_params: dict[str, Any] = {}
    if initial_codec_chunk_frames is not None:
        extra_params[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = initial_codec_chunk_frames
    seed = _request_or_default_generation_value(
        req,
        "seed",
        default_generation_params,
    )
    if seed is not None:
        tts_params["seed"] = seed

    # Sampling params — use S2-Pro-tuned defaults
    sampling = SamplingParams(
        temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1
    )
    max_new_tokens = _speech_effective_max_new_tokens(
        req,
        default_generation_params,
    )
    if max_new_tokens is not None:
        sampling.max_new_tokens = max_new_tokens
    temperature = _request_or_default_generation_value(
        req,
        "temperature",
        default_generation_params,
    )
    if temperature is not None:
        sampling.temperature = temperature
    top_p = _request_or_default_generation_value(
        req,
        "top_p",
        default_generation_params,
    )
    if top_p is not None:
        sampling.top_p = top_p
    top_k = _request_or_default_generation_value(
        req,
        "top_k",
        default_generation_params,
    )
    if top_k is not None:
        sampling.top_k = top_k
    min_p = _request_or_default_generation_value(
        req,
        "min_p",
        default_generation_params,
    )
    if min_p is not None:
        sampling.min_p = min_p
    repetition_penalty = _request_or_default_generation_value(
        req,
        "repetition_penalty",
        default_generation_params,
    )
    if repetition_penalty is not None:
        sampling.repetition_penalty = repetition_penalty
    frequency_penalty = _request_or_default_generation_value(
        req,
        "frequency_penalty",
        default_generation_params,
    )
    if frequency_penalty is not None:
        sampling.frequency_penalty = frequency_penalty
    presence_penalty = _request_or_default_generation_value(
        req,
        "presence_penalty",
        default_generation_params,
    )
    if presence_penalty is not None:
        sampling.presence_penalty = presence_penalty
    stop = _request_or_default_generation_value(
        req,
        "stop",
        default_generation_params,
    )
    if isinstance(stop, str):
        sampling.stop = [stop]
    elif isinstance(stop, list):
        sampling.stop = list(stop)
    if seed is not None:
        sampling.seed = seed

    # Build prompt: plain string if no references, dict otherwise
    prompt: Any = req.input
    references: list[dict[str, Any]] = []
    if req.references:
        references.extend(
            [reference.model_dump(exclude_none=True) for reference in req.references]
        )

    # Backward compatibility with ref_audio/ref_text form.
    if req.ref_audio is not None:
        ref: dict[str, Any] = {"audio_path": req.ref_audio}
        if req.ref_text is not None:
            ref["text"] = req.ref_text
        references.append(ref)

    if references:
        prompt = {"text": req.input, "references": references}

    extra_params.update(_default_talker_extra_params(default_talker_params))

    return GenerateRequest(
        model=req.model or default_model,
        prompt=prompt,
        sampling=sampling,
        stage_sampling=_stage_sampling_overrides(req.stage_sampling),
        stage_params=_stage_param_overrides(req.stage_params),
        extra_params=extra_params,
        stream=req.stream,
        output_modalities=["audio"],
        metadata={
            "task": "tts",
            "tts_params": tts_params,
        },
    )


_build_speech_generate_request = build_speech_generate_request


def _register_transcriptions(app: FastAPI) -> None:
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float | None = Form(default=None),
    ) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name
        request_id = f"transcription-{uuid.uuid4()}"

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        gen_req = build_transcription_generate_request(
            audio_bytes=audio_bytes,
            filename=file.filename,
            content_type=file.content_type,
            model=model or default_model,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

        try:
            result = await client.completion(gen_req, request_id=request_id)
        except ClientError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error transcribing audio for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        text = result.text
        normalized_response_format = response_format.strip().lower()
        if normalized_response_format == "text":
            return PlainTextResponse(text)
        if normalized_response_format not in {"json", "verbose_json"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported response_format for /v1/audio/transcriptions: "
                    f"{response_format!r}"
                ),
            )
        return JSONResponse(content=TranscriptionResponse(text=text).model_dump())


def build_transcription_generate_request(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    model: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
) -> GenerateRequest:
    params: dict[str, Any] = {"task": "transcribe"}
    if language is not None:
        params["language"] = language
    if prompt is not None:
        params["prompt"] = prompt
    if temperature is not None:
        params["temperature"] = temperature

    return GenerateRequest(
        model=model,
        prompt={
            "audio_bytes": audio_bytes,
            "filename": filename,
            "content_type": content_type,
        },
        extra_params=params,
        stream=False,
        output_modalities=["text"],
        metadata={"task": "asr"},
    )
