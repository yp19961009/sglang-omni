# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible request/response protocol definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared / Common
# ---------------------------------------------------------------------------


class UsageResponse(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Chat Completion
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionAudio(BaseModel):
    """Audio data returned in a chat completion response."""

    id: str
    data: str  # base64-encoded audio
    format: str | None = None
    sample_rate: int | None = None
    expires_at: int | None = None
    transcript: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Streaming
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    # DashScope/vLLM-style nested request parameters. Only known fields are
    # lifted by the OpenAI server; the dict is not blindly forwarded.
    parameters: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    # Tool/function-call prompt material. The server only preserves these for
    # the model chat template; it does not execute tools.
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    functions: list[dict[str, Any]] | None = None
    function_call: Any | None = None

    # Multi-modal output control
    modalities: str | list[str] | None = None  # e.g. ["text", "audio"]
    output_modalities: str | list[str] | None = None
    response_modalities: str | list[str] | None = None
    enable_audio_output: Any | None = None
    audio_output: Any | None = None
    return_audio: Any | None = None
    do_wave: Any | None = None

    # Audio output configuration
    audio: dict[str, Any] | None = None  # {"voice": "...", "format": "wav"}

    # Audio input (sglang-omni extension)
    # Can be an audio path/URL, OpenAI-style input_audio dict, or a list.
    # 中文说明：不要把顶层 audio 当输入别名；OpenAI 里 audio 是输出音频配置。
    audios: str | dict[str, Any] | list[Any] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "audios",
            "input_audio",
            "input_audios",
            "audio_url",
            "audio_urls",
        ),
    )

    # Image input (sglang-omni extension)
    # Can be a path/URL, OpenAI-style image_url dict, or a list.
    images: str | dict[str, Any] | list[Any] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "images",
            "image",
            "input_images",
            "input_image",
            "image_urls",
            "image_url",
        ),
    )
    image_min_pixels: int | None = Field(
        default=None,
        validation_alias=AliasChoices("image_min_pixels", "min_pixels"),
    )
    image_max_pixels: int | None = Field(
        default=None,
        validation_alias=AliasChoices("image_max_pixels", "max_pixels"),
    )

    # Video input (sglang-omni extension)
    # Can be a path/URL, OpenAI-style input_video/video_url dict, or a list.
    videos: str | dict[str, Any] | list[Any] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "videos",
            "video",
            "input_videos",
            "input_video",
            "video_urls",
            "video_url",
        ),
    )
    video_fps: float | None = Field(
        default=None,
        validation_alias=AliasChoices("video_fps", "fps"),
    )
    video_min_frames: int | None = Field(
        default=None,
        validation_alias=AliasChoices("video_min_frames", "min_frames"),
    )
    video_max_frames: int | None = Field(
        default=None,
        validation_alias=AliasChoices("video_max_frames", "max_frames"),
    )
    video_min_pixels: int | None = Field(
        default=None,
        validation_alias=AliasChoices("video_min_pixels", "min_pixels"),
    )
    video_max_pixels: int | None = Field(
        default=None,
        validation_alias=AliasChoices("video_max_pixels", "max_pixels"),
    )
    video_total_pixels: int | None = Field(
        default=None,
        validation_alias=AliasChoices("video_total_pixels", "total_pixels"),
    )
    use_audio_in_video: bool | list[bool] | None = None
    dependent_audio: list[int] | list[bool] | None = None
    video_dependent_audio: list[int] | list[bool] | None = None
    video_seconds_per_chunk: float | None = Field(
        default=None,
        validation_alias=AliasChoices("video_seconds_per_chunk", "seconds_per_chunk"),
    )
    video_position_id_per_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "video_position_id_per_seconds",
            "position_id_per_seconds",
        ),
    )
    return_video_metadata: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("return_video_metadata", "return_metadata"),
    )
    video_metadata: Any | None = Field(
        default=None,
        validation_alias=AliasChoices("video_metadata", "videos_metadata"),
    )

    # vLLM-compatible multimodal prompt fields. These are useful when reusing
    # Qwen3.5-Omni perf_v2 request builders against the OpenAI endpoint.
    multi_modal_data: Any | None = None
    mm_processor_kwargs: Any | None = None
    multi_modal_uuids: Any | None = None

    # Audio processor overrides (sglang-omni extension)
    audio_target_sr: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "audio_target_sr",
            "audio_sampling_rate",
            "sampling_rate",
        ),
    )
    audio_timestamp_interval: int | None = Field(
        default=None,
        validation_alias=AliasChoices("audio_timestamp_interval", "timestamp_interval"),
    )
    audio_downsample_times: int | None = Field(
        default=None,
        validation_alias=AliasChoices("audio_downsample_times", "downsample_times"),
    )
    audio_downsample_chunk_size: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "audio_downsample_chunk_size",
            "downsample_chunk_size",
        ),
    )

    # Per-stage sampling overrides (sglang-omni specific)
    stage_sampling: dict[str, dict[str, Any]] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None

    # Talker-specific overrides for Qwen3-Omni speech output
    talker_temperature: float | None = None
    talker_top_p: float | None = None
    talker_top_k: int | None = None
    talker_min_p: float | None = None
    talker_repetition_penalty: float | None = None
    talker_seed: int | None = None
    talker_sampling_seed: int | None = None
    talker_max_tokens: int | None = None
    talker_max_completion_tokens: int | None = None
    talker_max_new_tokens: int | None = None
    talker_params: dict[str, Any] | None = None
    subtalker_temperature: float | None = None
    subtalker_top_p: float | None = None
    subtalker_top_k: int | None = None
    subtalker_min_p: float | None = None
    subtalker_repetition_penalty: float | None = None
    subtalker_seed: int | None = None
    subtalker_sampling_seed: int | None = None
    subtalker_max_tokens: int | None = None
    subtalker_max_completion_tokens: int | None = None
    subtalker_max_new_tokens: int | None = None
    subtalker_params: dict[str, Any] | None = None

    # Qwen3.5-Omni talker controls. These mirror the params consumed by the
    # Qwen3.5 request builders and are intentionally whitelisted instead of
    # accepting arbitrary extra request fields.
    voice_type: str | None = None
    voice: str | None = None
    speaker: str | None = None
    speaker_id: int | None = None
    language: str | None = None
    lang: str | None = None
    language_type: str | None = None
    target_language: str | None = None
    target_lang: str | None = None
    source_lang: str | None = None
    source_language: str | None = None
    translation_options: dict[str, Any] | None = None
    language_id: int | None = None
    voice_style: str | None = None
    style: str | None = None
    instruction: str | None = None
    talker_instruction: str | None = None
    assistant_instruct_ids: list[int] | None = None
    talker_assistant_prompt_ids: list[int] | None = None
    voice_style_ids: list[int] | None = None
    instruct_ids: list[int] | None = None
    system_instruct_ids: list[int] | None = None
    talker_system_instruct_ids: list[int] | None = None
    speaker_system_instruct_ids: list[int] | None = None
    talker_system_instruct: str | list[int] | None = None
    system_instruct: str | list[int] | None = None
    voice_clone_system_instruct: str | list[int] | None = None
    xvector_info: Any | None = None
    voice_clone_info: Any | None = None
    xvector_info_path: str | None = None
    voice_clone_path: str | None = None
    voice_clone: Any | None = None
    prompt_speaker_codes: Any | None = None
    speaker_codec_codes: Any | None = None
    voice_clone_codes: Any | None = None
    tts_generate_mode: str | None = None
    qwen_tts_generate_mode: str | None = None
    enable_tn: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "enable_tn",
            "enable_text_normalization",
            "text_normalization",
        ),
    )

    # Misc
    request_id: str | None = None
    user: str | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: dict[str, Any]
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageResponse | None = None


class ChatCompletionStreamDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    audio: ChatCompletionAudio | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageResponse | None = None


# ---------------------------------------------------------------------------
# Speech (TTS)
# ---------------------------------------------------------------------------


class SpeechReference(BaseModel):
    """Reference item for voice cloning in /v1/audio/speech."""

    audio_path: str | None = None
    text: str | None = None
    vq_codes: list[list[int]] | list[int] | None = None


class CreateSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request.

    Standard OpenAI fields plus extensions for advanced TTS models
    (e.g. voice cloning, style instructions).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Standard OpenAI fields
    model: str | None = None
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False
    stream_format: Literal["sse", "audio"] = "sse"

    # DashScope/vLLM-style nested request parameters. Speech requests use the
    # same conservative whitelist as chat requests when lifting values.
    parameters: dict[str, Any] | None = None

    # Advanced TTS extensions
    task_type: str | None = None  # e.g. "Base", "CustomVoice", "VoiceDesign"
    language: str | None = None
    instructions: str | None = None  # style/emotion instructions

    # Voice cloning parameters
    ref_audio: str | None = None  # path or URL to reference audio
    ref_text: str | None = None  # transcript of reference audio
    references: list[SpeechReference] | None = None  # S2-Pro-style refs
    token_count: int | None = None  # MOSS-TTS duration token target
    duration_tokens: int | None = None  # alias for token_count
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)

    # Generation parameters
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Per-stage overrides (sglang-omni specific)
    stage_sampling: dict[str, dict[str, Any]] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_stream_format(self) -> "CreateSpeechRequest":
        if self.stream_format == "audio":
            if not self.stream:
                raise ValueError('stream_format="audio" requires stream=true')
            if self.response_format.lower() != "pcm":
                raise ValueError('stream_format="audio" requires response_format="pcm"')
        return self
    # Qwen3.5-Omni talker controls. They mirror ChatCompletionRequest so
    # /v1/audio/speech can drive the same talker/code2wav path.
    talker_temperature: float | None = None
    talker_top_p: float | None = None
    talker_top_k: int | None = None
    talker_min_p: float | None = None
    talker_repetition_penalty: float | None = None
    talker_seed: int | None = None
    talker_sampling_seed: int | None = None
    talker_max_tokens: int | None = None
    talker_max_completion_tokens: int | None = None
    talker_max_new_tokens: int | None = None
    talker_params: dict[str, Any] | None = None
    subtalker_temperature: float | None = None
    subtalker_top_p: float | None = None
    subtalker_top_k: int | None = None
    subtalker_min_p: float | None = None
    subtalker_repetition_penalty: float | None = None
    subtalker_seed: int | None = None
    subtalker_sampling_seed: int | None = None
    subtalker_max_tokens: int | None = None
    subtalker_max_completion_tokens: int | None = None
    subtalker_max_new_tokens: int | None = None
    subtalker_params: dict[str, Any] | None = None

    voice_type: str | None = None
    speaker: str | None = None
    speaker_id: int | None = None
    lang: str | None = None
    language_type: str | None = None
    target_language: str | None = None
    target_lang: str | None = None
    source_lang: str | None = None
    source_language: str | None = None
    translation_options: dict[str, Any] | None = None
    language_id: int | None = None
    voice_style: str | None = None
    style: str | None = None
    instruction: str | None = None
    talker_instruction: str | None = None
    assistant_instruct_ids: list[int] | None = None
    talker_assistant_prompt_ids: list[int] | None = None
    voice_style_ids: list[int] | None = None
    instruct_ids: list[int] | None = None
    system_instruct_ids: list[int] | None = None
    talker_system_instruct_ids: list[int] | None = None
    speaker_system_instruct_ids: list[int] | None = None
    talker_system_instruct: str | list[int] | None = None
    system_instruct: str | list[int] | None = None
    voice_clone_system_instruct: str | list[int] | None = None
    xvector_info: Any | None = None
    voice_clone_info: Any | None = None
    xvector_info_path: str | None = None
    voice_clone_path: str | None = None
    voice_clone: Any | None = None
    prompt_speaker_codes: Any | None = None
    speaker_codec_codes: Any | None = None
    voice_clone_codes: Any | None = None
    tts_generate_mode: str | None = None
    qwen_tts_generate_mode: str | None = None
    enable_tn: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "enable_tn",
            "enable_text_normalization",
            "text_normalization",
        ),
    )


# ---------------------------------------------------------------------------
# Audio transcription (ASR)
# ---------------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


class ModelPermission(BaseModel):
    """Model permission info."""

    id: str = "modelperm-default"
    object: str = "model_permission"
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True


class ModelCard(BaseModel):
    """A single model entry."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "sglang-omni"
    permission: list[ModelPermission] = Field(
        default_factory=lambda: [ModelPermission()]
    )
    root: str | None = None


class ModelList(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
