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
from sglang_omni.models.qwen3_5_omni.config import normalize_qwen35_omni_model_name
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
_CHAT_PARAMETERS_ALIASES: dict[str, tuple[str, ...]] = {
    "audios": ("input_audio", "input_audios", "audio_url", "audio_urls"),
    "images": ("image", "input_image", "input_images", "image_url", "image_urls"),
    "videos": ("video", "input_video", "input_videos", "video_url", "video_urls"),
    "image_min_pixels": ("min_pixels",),
    "image_max_pixels": ("max_pixels",),
    "video_fps": ("fps",),
    "video_min_frames": ("min_frames",),
    "video_max_frames": ("max_frames",),
    "video_min_pixels": ("min_pixels",),
    "video_max_pixels": ("max_pixels",),
    "video_total_pixels": ("total_pixels",),
    "video_seconds_per_chunk": ("seconds_per_chunk",),
    "video_position_id_per_seconds": ("position_id_per_seconds",),
    "return_video_metadata": ("return_metadata",),
    "video_metadata": ("videos_metadata",),
    "audio_target_sr": ("audio_sampling_rate", "sampling_rate"),
    "audio_timestamp_interval": ("timestamp_interval",),
    "audio_downsample_times": ("downsample_times",),
    "audio_downsample_chunk_size": ("downsample_chunk_size",),
    "stage_params": ("stage_parameters",),
    "stage_sampling": ("stage_sampling_params", "stage_sampling_parameters"),
    "enable_tn": (
        "enable_text_normalization",
        "text_normalization",
    ),
}
_TALKER_EXTRA_FIELD_NAMES = (
    "talker_temperature",
    "talker_top_p",
    "talker_top_k",
    "talker_min_p",
    "talker_repetition_penalty",
    "talker_seed",
    "talker_sampling_seed",
    "talker_max_tokens",
    "talker_max_completion_tokens",
    "talker_max_new_tokens",
    "talker_params",
    "subtalker_temperature",
    "subtalker_top_p",
    "subtalker_top_k",
    "subtalker_min_p",
    "subtalker_repetition_penalty",
    "subtalker_seed",
    "subtalker_sampling_seed",
    "subtalker_max_tokens",
    "subtalker_max_completion_tokens",
    "subtalker_max_new_tokens",
    "subtalker_params",
    "voice_type",
    "voice",
    "speaker",
    "speaker_id",
    "language",
    "lang",
    "language_type",
    "target_language",
    "target_lang",
    "source_lang",
    "source_language",
    "language_id",
    "voice_style",
    "style",
    "instruction",
    "talker_instruction",
    "assistant_instruct_ids",
    "talker_assistant_prompt_ids",
    "voice_style_ids",
    "instruct_ids",
    "system_instruct_ids",
    "talker_system_instruct_ids",
    "speaker_system_instruct_ids",
    "talker_system_instruct",
    "system_instruct",
    "voice_clone_system_instruct",
    "prompt_speaker_codes",
    "speaker_codec_codes",
    "voice_clone_codes",
    "tts_generate_mode",
    "qwen_tts_generate_mode",
    "enable_tn",
    "do_wave",
)


def _is_bad_request_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _BAD_REQUEST_MARKERS)


def _normalize_openai_model_name(model_name: str | None) -> str | None:
    # 中文说明：Qwen3.5-Omni 用户命令里常混用 qwen3.5/qwen3_5/qwen3-5。
    # serve 层只归一已知 Qwen3.5 别名，其他模型名保持原样。
    return normalize_qwen35_omni_model_name(model_name)


def _coerce_modalities(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        # 中文说明：OpenAI 正式字段通常是 list，但 curl/压测脚本常把
        # modalities 写成 "text,audio" 或 "text audio"。serve 层先统一
        # 切成 token，避免 Qwen3.5 talker gating 把它当作单个模态。
        values = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            if isinstance(item, str):
                values.extend(item.replace(",", " ").split())
            else:
                values.append(item)
    else:
        return None
    modalities = [str(item).lower() for item in values if str(item).strip()]
    return modalities or None


def _media_payload_value(value: Any, *, media_prefix: str) -> Any:
    if isinstance(value, dict):
        path = value.get("path")
        if path is not None:
            # 中文说明：vLLM Qwen3.5 OpenAI demo 会把本地临时文件作为
            # {"path": ...} 或等价裸字符串传入；serve 层先归一成路径，
            # 后续 preprocessor/resource connector 才能复用同一套加载逻辑。
            return path
        url = value.get("url")
        if url is not None:
            return url
        data = value.get("data")
        if data is not None:
            media_format = str(value.get("format") or "octet-stream")
            return f"data:{media_prefix}/{media_format};base64,{data}"
    return value


def _coerce_media_list(value: Any, *, media_prefix: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value] if value else None
    if isinstance(value, (list, tuple, set)):
        values = [
            _media_payload_value(item, media_prefix=media_prefix)
            for item in value
            if item is not None
        ]
        return values or None
    # 中文说明：兼容直接传 PIL/ndarray/tensor 这类媒体对象；不要在 serve 层
    # 强行 bool(value)，避免多元素数组/张量报错。OpenAI-style {"url": ...}
    # / {"data": ...} 则先转成 resource connector 能加载的路径或 data URL。
    return [_media_payload_value(value, media_prefix=media_prefix)]


def _coerce_audio_output_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "audio"}:
            return True
        if normalized in {"0", "false", "no", "off", "text", "none"}:
            return False
    return None


def _nested_parameters(req: Any) -> dict[str, Any]:
    parameters = getattr(req, "parameters", None)
    return parameters if isinstance(parameters, dict) else {}


def _chat_nested_parameters(req: ChatCompletionRequest) -> dict[str, Any]:
    return _nested_parameters(req)


def _request_or_parameters_value(req: Any, key: str) -> Any:
    fields_set = getattr(req, "model_fields_set", None)
    field_is_explicit = fields_set is None or key in fields_set
    if field_is_explicit:
        value = getattr(req, key, None)
        if value is not None:
            return value
    parameters = _nested_parameters(req)
    value = parameters.get(key)
    if value is not None:
        return value
    for alias in _CHAT_PARAMETERS_ALIASES.get(key, ()):
        value = parameters.get(alias)
        if value is not None:
            # 中文说明：Pydantic 只会处理顶层 OpenAI body 的 alias；
            # DashScope/vLLM-style parameters 是普通 dict，需要这里补齐
            # fps/max_pixels/timestamp_interval 等常用短名。
            return value
    if not field_is_explicit:
        # 中文说明：Pydantic 默认值不是用户意图。尤其 /v1/audio/speech
        # 的 voice 默认是 "default"，不能抢在 parameters.voice_type 前面，
        # 否则 Qwen3.5 request builder 会按 voice 优先级选错音色。
        return None
    return None


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
    value = _request_or_parameters_value(req, key)
    if value is not None:
        return value
    return _default_generation_value(default_generation_params, key)


def _chat_effective_max_tokens(
    req: ChatCompletionRequest,
    default_generation_params: dict[str, Any] | None = None,
) -> Any:
    max_completion_tokens = _request_or_parameters_value(
        req,
        "max_completion_tokens",
    )
    if max_completion_tokens is not None:
        return max_completion_tokens
    max_tokens = _request_or_parameters_value(req, "max_tokens")
    if max_tokens is not None:
        return max_tokens
    default_max_completion_tokens = _default_generation_value(
        default_generation_params,
        "max_completion_tokens",
    )
    if default_max_completion_tokens is not None:
        return default_max_completion_tokens
    return _default_generation_value(default_generation_params, "max_tokens")


def _stage_sampling_overrides(req: Any) -> dict[str, SamplingParams] | None:
    raw = _request_or_parameters_value(req, "stage_sampling")
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
        # 中文说明：兼容 parameters.stage_sampling，保证 Qwen3.5 压测时
        # thinker/talker/subtalker 等 stage 级采样参数能进入调度层。
        stage_sampling[str(stage_name)] = SamplingParams(**params_dict)
    return stage_sampling or None


def _stage_param_overrides(req: Any) -> dict[str, dict[str, Any]] | None:
    raw = _request_or_parameters_value(req, "stage_params")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("stage_params must be a mapping of stage name to params")

    stage_params: dict[str, dict[str, Any]] = {}
    for stage_name, params in raw.items():
        if not isinstance(params, dict):
            raise ValueError("stage_params values must be mappings")
        # 中文说明：复制一层，避免后续 pipeline 侧合并/补默认值时修改
        # FastAPI/Pydantic 保存的原始请求对象。
        stage_params[str(stage_name)] = dict(params)
    return stage_params or None


def _chat_audio_config(req: ChatCompletionRequest) -> dict[str, Any] | None:
    if isinstance(req.audio, dict):
        return req.audio
    audio_config = _chat_nested_parameters(req).get("audio")
    return audio_config if isinstance(audio_config, dict) else None


def _request_metadata(req: ChatCompletionRequest) -> dict[str, Any]:
    metadata = req.metadata if isinstance(req.metadata, dict) else None
    if metadata is None:
        nested_metadata = _chat_nested_parameters(req).get("metadata")
        metadata = nested_metadata if isinstance(nested_metadata, dict) else None
    metadata = dict(metadata or {})
    for key in ("request_id", "user"):
        value = getattr(req, key, None)
        if value is not None:
            # 中文说明：这些字段只用于请求追踪/压测归因；放进 metadata 后
            # coordinator、stage profile 和下游日志都能拿到同一个业务标识。
            metadata[key] = value
    # 中文说明：metadata 是请求追踪/压测标签，不直接执行；复制一份避免
    # 后续内部字段合并时修改用户传入的原始对象。
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
    for key in ("modalities", "output_modalities", "response_modalities"):
        modalities = _coerce_modalities(_request_or_parameters_value(req, key))
        if modalities is not None:
            return modalities

    for key in ("enable_audio_output", "audio_output", "return_audio", "do_wave"):
        enabled = _coerce_audio_output_flag(_request_or_parameters_value(req, key))
        if enabled is not None:
            # 中文说明：兼容 vLLM/Qwen3.5 内部压测常用的布尔音频开关；
            # do_wave=True 在 vLLM 脚本里表示最终需要合成 waveform。
            # 显式 modalities 仍优先，避免布尔开关覆盖用户的精确模态列表。
            return ["text", "audio"] if enabled else ["text"]

    return ["text"]


def _stream_include_usage(req: ChatCompletionRequest) -> bool:
    stream_options = _request_or_parameters_value(req, "stream_options")
    if not isinstance(stream_options, dict):
        return False
    enabled = _coerce_audio_output_flag(stream_options.get("include_usage"))
    # 中文说明：OpenAI/vLLM streaming 客户端常用
    # stream_options.include_usage 请求一个 usage-only SSE chunk。默认仍保持
    # sglang-omni 旧行为，避免影响现有流式消费者。
    return bool(enabled)


def _normalize_voice_clone_value(value: Any) -> Any:
    if isinstance(value, dict) and len(value) == 1:
        for path_key in ("path", "xvector_info_path", "voice_clone_path"):
            if value.get(path_key) is not None:
                return value[path_key]
    return value


def _translation_target_language(options: Any) -> Any:
    if not isinstance(options, dict):
        return None
    for key in (
        "target_language",
        "target_lang",
        "targetLanguage",
        "language",
        "lang",
    ):
        value = options.get(key)
        if value is not None:
            return value
    return None


def _talker_extra_params(req: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field_name in _TALKER_EXTRA_FIELD_NAMES:
        value = _request_or_parameters_value(req, field_name)
        if value is not None:
            params[field_name] = value

    translation_options = _request_or_parameters_value(req, "translation_options")
    if translation_options is not None:
        # 中文说明：vLLM Qwen3.5 perf_v2 的翻译请求把目标语言放在
        # translation_options.target_lang。serve 层先保留原结构，再补一份
        # target_language 给 talker/request builder 走统一语言控制路径。
        params["translation_options"] = translation_options
        target_language = _translation_target_language(translation_options)
        if target_language is not None and not any(
            params.get(key) is not None
            for key in (
                "language_id",
                "language",
                "lang",
                "language_type",
                "target_language",
                "target_lang",
            )
        ):
            params["target_language"] = target_language

    xvector_info = _request_or_parameters_value(req, "xvector_info")
    if xvector_info is None:
        xvector_info = _request_or_parameters_value(req, "xvector_info_path")
    if xvector_info is not None:
        # 中文说明：Qwen3.5 talker 统一从 params 读取 xvector_info。
        # 顶层 path alias 在 serve 层归一，避免 request builder 再分支。
        params["xvector_info"] = _normalize_voice_clone_value(xvector_info)

    voice_clone_info = _request_or_parameters_value(req, "voice_clone_info")
    if voice_clone_info is None:
        voice_clone_info = _request_or_parameters_value(req, "voice_clone_path")
    if voice_clone_info is None:
        voice_clone_info = _request_or_parameters_value(req, "voice_clone")
    if voice_clone_info is not None:
        params["voice_clone_info"] = _normalize_voice_clone_value(voice_clone_info)

    return params


def _merge_default_talker_params(
    default_talker_params: dict[str, Any] | None,
    request_talker_params: dict[str, Any],
) -> dict[str, Any]:
    if not default_talker_params:
        return request_talker_params
    # 中文说明：server 默认参数只负责补缺省值，请求显式传入的
    # voice/voice_type/enable_tn 等控制项永远覆盖默认值。
    merged = {
        key: value
        for key, value in default_talker_params.items()
        if value is not None
    }
    merged.update(request_talker_params)
    return merged


def _chat_talker_extra_params(
    req: ChatCompletionRequest,
    default_talker_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _merge_default_talker_params(
        default_talker_params,
        _talker_extra_params(req),
    )


def _speech_talker_extra_params(
    req: CreateSpeechRequest,
    default_talker_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = _talker_extra_params(req)
    if req.instructions is not None and params.get("instruction") is None:
        # 中文说明：OpenAI speech 的 instructions 对 Qwen3.5 talker 等价于
        # voice/style instruction，放进 params 才能被 request builder 读取。
        params["instruction"] = req.instructions
    return _merge_default_talker_params(default_talker_params, params)


def _speech_effective_max_new_tokens(
    req: CreateSpeechRequest,
    default_generation_params: dict[str, Any] | None = None,
) -> Any:
    max_new_tokens = _request_or_parameters_value(req, "max_new_tokens")
    if max_new_tokens is not None:
        return max_new_tokens
    max_completion_tokens = _request_or_parameters_value(
        req,
        "max_completion_tokens",
    )
    if max_completion_tokens is not None:
        return max_completion_tokens
    max_tokens = _request_or_parameters_value(req, "max_tokens")
    if max_tokens is not None:
        return max_tokens
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
    app.state.model_name = _normalize_openai_model_name(
        model_name or "sglang-omni"
    )
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
        model = _normalize_openai_model_name(req.model or default_model)

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
        # 中文说明：vLLM perf_v2 的 Qwen3.5 Omni server 在非流式响应
        # 顶层额外返回 audio: {data, format}。保留 message.audio 的同时补
        # 这个兼容字段，方便同一套压测脚本横向对比 vLLM 和 sglang-omni。
        top_level_audio = {
            "data": result.audio.data,
            "format": result.audio.format or audio_format,
        }
        if result.audio.sample_rate is not None:
            top_level_audio["sample_rate"] = result.audio.sample_rate
    else:
        top_level_audio = None

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
                # 中文说明：vLLM/OpenAI 非流式响应在正常结束时通常返回
                # finish_reason="stop"。部分内部 fake/旧 stage 不带该字段，
                # serve 层补齐可让压测脚本和 OpenAI SDK 解析更稳定。
                finish_reason=result.finish_reason or "stop",
            )
        ],
        usage=usage,
    )

    payload = response.model_dump()
    if top_level_audio is not None:
        payload["audio"] = top_level_audio
    return JSONResponse(content=payload)


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

    # Build per-stage sampling/runtime overrides. 顶层字段和
    # DashScope/vLLM-style parameters 都走同一套白名单解析。
    stage_sampling = _stage_sampling_overrides(req)
    stage_params = _stage_param_overrides(req)

    # Extract audios, images, and videos from request
    audios: list[Any] | None = None
    request_audios = _request_or_parameters_value(req, "audios")
    if request_audios is not None:
        audios = _coerce_media_list(request_audios, media_prefix="audio")

    images: list[Any] | None = None
    request_images = _request_or_parameters_value(req, "images")
    if request_images is not None:
        images = _coerce_media_list(request_images, media_prefix="image")

    videos: list[Any] | None = None
    request_videos = _request_or_parameters_value(req, "videos")
    if request_videos is not None:
        videos = _coerce_media_list(request_videos, media_prefix="video")

    # Merge audio config, audios, images, and videos into metadata.
    # 中文说明：这些 request-level 预处理参数必须跟媒体一起进入
    # GenerateRequest.metadata，client 会再把它们提升到 OmniRequest.inputs。
    metadata: dict[str, Any] = _request_metadata(req)
    audio_config = _chat_audio_config(req)
    if audio_config is not None:
        metadata["audio_config"] = audio_config
    tools = _request_or_parameters_value(req, "tools")
    if tools is None:
        tools = _legacy_functions_as_tools(_request_or_parameters_value(req, "functions"))
    if tools is not None:
        # 中文说明：tools 只用于 chat template 渲染，不在服务端执行。
        metadata["tools"] = tools
    for key in ("tool_choice", "parallel_tool_calls", "function_call"):
        value = _request_or_parameters_value(req, key)
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
        "multi_modal_data",
        "mm_processor_kwargs",
        "multi_modal_uuids",
        "audio_target_sr",
        "audio_timestamp_interval",
        "audio_downsample_times",
        "audio_downsample_chunk_size",
    ):
        value = _request_or_parameters_value(req, key)
        if value is not None:
            metadata[key] = value

    extra_params = _chat_talker_extra_params(req, default_talker_params)

    return GenerateRequest(
        model=_normalize_openai_model_name(req.model or default_model),
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
        if _request_or_parameters_value(req, field) is not None
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

    # 中文说明：metadata 保留给 HTTP/TTS 响应层；Qwen3.5 talker/code2wav
    # request builder 从 OmniRequest.params 读取 voice、clone、subtalker
    # 采样等控制项，所以这里必须通过 extra_params 进入 client._build_params。
    extra_params = _speech_talker_extra_params(req, default_talker_params)

    return GenerateRequest(
        model=_normalize_openai_model_name(req.model or default_model),
        prompt=prompt,
        sampling=sampling,
        stage_sampling=_stage_sampling_overrides(req),
        stage_params=_stage_param_overrides(req),
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
