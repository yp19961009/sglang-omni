# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for Qwen3-Omni."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import build_lightweight_mm_inputs
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing import (
    build_audio_mm_inputs,
    build_image_mm_inputs,
    build_video_mm_inputs,
    compute_audio_cache_key,
    compute_image_cache_key,
    compute_video_cache_key,
    ensure_audio_list_async,
    ensure_chat_template,
    ensure_image_list_async,
    ensure_video_list_async,
    normalize_messages,
)
from sglang_omni.preprocessing.video import derive_video_total_pixels_from_mm_len
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)
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


def _resolve_local_model_dir(model_path: str) -> str:
    """Resolve a local model directory without eagerly hydrating full snapshots."""
    path = Path(model_path)
    if path.exists():
        return str(path)
    try:
        return str(resolve_model_path(model_path, local_files_only=True))
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Local-only model resolution failed for %s; falling back to hub id",
            model_path,
            exc_info=exc,
        )
        return model_path


def _combine_cache_keys(*keys: str | None) -> str | None:
    parts = [key for key in keys if key]
    if not parts:
        return None
    return "|".join(parts)


def _contextualize_cache_key(base_key: str | None, **context: Any) -> str | None:
    if base_key is None:
        return None
    parts = [base_key]
    for key in sorted(context):
        value = _normalize_cache_context_value(context[key])
        if value is not None:
            parts.append(f"{key}={value}")
    return "|".join(parts)


def _normalize_cache_context_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_cache_context_value(item) for item in value)
    return value


def _media_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return len(value) > 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    # 中文说明：预采样音视频常用 torch.Tensor / numpy.ndarray 表达。
    # 这些对象不能放进 bool(value)、A or B 这类分支，否则多元素输入会
    # 在真正进入 processor 前就因为 truthiness 报错。
    return True


def _first_present_media_input(
    inputs: dict[str, Any],
    keys: tuple[str, ...],
) -> Any | None:
    for key in keys:
        value = inputs.get(key)
        if _media_value_is_present(value):
            return value
    return None


def _looks_like_openai_audio_output_config(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    keys = set(value)
    if keys & _AUDIO_MEDIA_PAYLOAD_KEYS:
        return False
    return bool(keys & _OPENAI_AUDIO_OUTPUT_CONFIG_KEYS)


def _first_present_audio_input(inputs: dict[str, Any]) -> Any | None:
    value = inputs.get("audios")
    if _media_value_is_present(value):
        return value
    value = inputs.get("audio")
    if _looks_like_openai_audio_output_config(value):
        # 中文说明：OpenAI chat/completions 的顶层 audio 是输出配置
        # （voice/format/language），不是输入音频。直连 OmniRequest 时也要
        # 避免把它送进 audio encoder。
        return None
    return value if _media_value_is_present(value) else None


def _media_item_count(value: Any) -> int:
    if not _media_value_is_present(value):
        return 0
    return len(value) if isinstance(value, list) else 1


def _request_raw_prompt_text(inputs: Any) -> str | None:
    if not isinstance(inputs, dict):
        return None
    if inputs.get("messages") is not None:
        return None
    prompt = inputs.get("prompt")
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
        first = prompt[0]
        if isinstance(first, str):
            return first
    return str(prompt)


def _request_prompt_token_ids(inputs: Any) -> torch.Tensor | None:
    if not isinstance(inputs, dict):
        return None
    token_ids = inputs.get("prompt_token_ids")
    if token_ids is None:
        return None
    token_ids_tensor = torch.as_tensor(token_ids, dtype=torch.long)
    if token_ids_tensor.dim() == 2 and token_ids_tensor.shape[0] == 1:
        token_ids_tensor = token_ids_tensor[0]
    if token_ids_tensor.dim() != 1:
        raise ValueError(
            "prompt_token_ids must be a 1-D token list for a single Omni request"
        )
    return token_ids_tensor


def _build_audio_mm_inputs_compat(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    audio_inputs = build_audio_mm_inputs(hf_inputs)
    if audio_inputs.get("input_features") is None:
        # 中文说明：Qwen3-Omni 老 processor 输出 input_features；
        # Qwen3.5-Omni Next / vLLM processor 输出 input_audio_features。
        # 这里归一成现有 audio_encoder/batch 逻辑使用的字段名。
        audio_inputs["input_features"] = hf_inputs.get("input_audio_features")
    return audio_inputs


DEFAULT_THINKER_MAX_NEW_TOKENS = 2048
QWEN3_OMNI_CHAT_TEMPLATE_FALLBACK_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _resolve_request_max_new_tokens(params: Any) -> int:
    if not isinstance(params, dict):
        return DEFAULT_THINKER_MAX_NEW_TOKENS
    for key in ("max_new_tokens", "max_tokens"):
        value = params.get(key)
        if value is not None:
            # 中文说明：OpenAI-compatible 请求使用 max_tokens，
            # sglang 内部 SamplingParams 使用 max_new_tokens。预处理阶段的
            # context length 校验也要与后续采样保持同一语义。
            return int(value)
    return DEFAULT_THINKER_MAX_NEW_TOKENS


def validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int = DEFAULT_THINKER_MAX_NEW_TOKENS,
    request_id: str | None = None,
) -> None:
    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )
    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} + "
            f"max_new_tokens {int(max_new_tokens)} = {total_tokens} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


class Qwen3OmniPreprocessor:
    """CPU-side preprocessing and tokenization using the HF processor."""

    processor_cls = Qwen3OmniMoeProcessor
    chat_template_fallback_model_paths = (QWEN3_OMNI_CHAT_TEMPLATE_FALLBACK_MODEL,)

    @classmethod
    def _load_processor_cls(cls):
        return cls.processor_cls

    def __init__(
        self,
        model_path: str,
        max_seq_len: int | None = None,
        *,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        video_min_frames: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_pixels: int | None = None,
        video_override_max_pixels: bool | None = None,
        video_seconds_per_chunk: float | None = None,
        video_position_id_per_seconds: float | None = None,
        audio_target_sr: int | None = None,
    ):
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.default_image_min_pixels = (
            int(image_min_pixels) if image_min_pixels is not None else None
        )
        self.default_image_max_pixels = (
            int(image_max_pixels) if image_max_pixels is not None else None
        )
        self.default_video_fps = float(video_fps) if video_fps is not None else None
        self.default_video_max_frames = (
            int(video_max_frames) if video_max_frames is not None else None
        )
        self.default_video_min_frames = (
            int(video_min_frames) if video_min_frames is not None else None
        )
        self.default_video_min_pixels = (
            int(video_min_pixels) if video_min_pixels is not None else None
        )
        self.default_video_max_pixels = (
            int(video_max_pixels) if video_max_pixels is not None else None
        )
        self.default_video_total_pixels = (
            int(video_total_pixels) if video_total_pixels is not None else None
        )
        self.default_video_override_max_pixels = (
            bool(video_override_max_pixels)
            if video_override_max_pixels is not None
            else False
        )
        self.default_video_seconds_per_chunk = (
            float(video_seconds_per_chunk)
            if video_seconds_per_chunk is not None
            else None
        )
        self.default_video_position_id_per_seconds = (
            float(video_position_id_per_seconds)
            if video_position_id_per_seconds is not None
            else None
        )
        self.default_audio_target_sr = (
            int(audio_target_sr) if audio_target_sr is not None else 16000
        )
        self.model_dir = _resolve_local_model_dir(model_path)
        processor_cls = self._load_processor_cls()
        try:
            self.processor = processor_cls.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
        except (OSError, ValueError, RuntimeError):
            if Path(model_path).exists():
                raise
            self.processor = processor_cls.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            self.model_dir = str(resolve_model_path(model_path, local_files_only=False))
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(
            self.tokenizer,
            model_path=self.model_dir,
            fallback_model_paths=self.chat_template_fallback_model_paths,
        )
        if not getattr(self.processor, "chat_template", None) and getattr(
            self.tokenizer, "chat_template", None
        ):
            self.processor.chat_template = self.tokenizer.chat_template

    def _processor_kwargs_for_request(
        self,
        *,
        request_inputs: Any,
        images_kwargs: dict[str, Any],
        videos_kwargs: dict[str, Any],
        audio_target_sr: int,
    ) -> dict[str, Any]:
        del request_inputs, audio_target_sr
        processor_kwargs: dict[str, Any] = {}
        if images_kwargs:
            processor_kwargs["images_kwargs"] = images_kwargs
        if videos_kwargs:
            processor_kwargs["videos_kwargs"] = videos_kwargs
        return processor_kwargs

    def _resolve_use_audio_in_video(
        self,
        request_inputs: dict[str, Any],
        raw_videos: Any,
    ) -> Any:
        del raw_videos
        return request_inputs.get("use_audio_in_video")

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
        num_videos: int,
    ) -> list[dict[str, Any]]:
        """Convert simple messages to HF's structured multimodal format."""
        if num_images == 0 and num_audios == 0 and num_videos == 0:
            return messages

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Only inject placeholders into the last user message
            if i == len(messages) - 1 and role == "user":
                content_parts: list[dict[str, Any]] = []
                # Placeholders come BEFORE text (Qwen3-Omni format)
                for _ in range(num_images):
                    content_parts.append({"type": "image"})
                for _ in range(num_videos):
                    content_parts.append({"type": "video"})
                for _ in range(num_audios):
                    content_parts.append({"type": "audio"})
                content_parts.append({"type": "text", "text": content})
                result.append({"role": role, "content": content_parts})
            else:
                result.append(msg)

        return result

    def _merge_audio_inputs_for_request(
        self,
        *,
        messages: Any,
        explicit_audios: Any,
        video_audios: list[Any] | None,
        use_audio_in_video: Any,
    ) -> tuple[Any, bool]:
        del messages, use_audio_in_video
        if not video_audios:
            return explicit_audios, False

        # Filter out videos without audio. This preserves the existing Qwen3 path:
        # explicit audio samples first, then extracted video audio samples.
        extracted = [audio for audio in video_audios if audio is not None]
        if not extracted:
            return explicit_audios, False

        if _media_value_is_present(explicit_audios):
            if isinstance(explicit_audios, list):
                return explicit_audios + extracted, True
            return [explicit_audios] + extracted, True
        return extracted, True

    def _audio_is_dependent_for_request(
        self,
        *,
        request_inputs: Any | None = None,
        messages: Any,
        explicit_audios: Any,
        video_audios: list[Any] | None,
        use_audio_in_video: Any,
    ) -> list[bool] | None:
        del messages, explicit_audios, video_audios, use_audio_in_video
        return None

    def _processor_use_audio_in_video_value(self, use_audio_in_video: Any) -> Any:
        return bool(use_audio_in_video)

    def _media_cache_key_for_request(
        self,
        *,
        request_inputs: dict[str, Any],
        modality: str,
        raw_value: Any,
        compute_cache_key: Callable[[Any], str | None],
    ) -> str | None:
        del request_inputs, modality
        return compute_cache_key(raw_value)

    async def __call__(self, payload: StagePayload) -> StagePayload:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="preprocess_start",
        )
        try:
            result = await self._call_impl(payload)
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="preprocess_end",
            )
        return result

    async def _call_impl(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            tools = inputs.get("tools")
            raw_prompt_text = _request_raw_prompt_text(inputs)
            raw_prompt_token_ids = _request_prompt_token_ids(inputs)
            raw_images = _first_present_media_input(inputs, ("images", "image"))
            raw_videos = _first_present_media_input(inputs, ("videos", "video"))
            raw_audios = _first_present_audio_input(inputs)
            audio_target_sr = int(
                inputs.get(
                    "audio_target_sr",
                    getattr(self, "default_audio_target_sr", 16000),
                )
            )
            image_min_pixels = inputs.get(
                "image_min_pixels",
                self.default_image_min_pixels,
            )
            image_max_pixels = inputs.get(
                "image_max_pixels",
                self.default_image_max_pixels,
            )
            video_fps = inputs.get("video_fps", self.default_video_fps)
            video_max_frames = inputs.get(
                "video_max_frames",
                self.default_video_max_frames,
            )
            video_min_frames = inputs.get(
                "video_min_frames",
                self.default_video_min_frames,
            )
            video_min_pixels = inputs.get(
                "video_min_pixels",
                self.default_video_min_pixels,
            )
            video_max_pixels = inputs.get(
                "video_max_pixels",
                self.default_video_max_pixels,
            )
            video_total_pixels = inputs.get(
                "video_total_pixels",
                self.default_video_total_pixels,
            )
            video_override_max_pixels = inputs.get(
                "video_override_max_pixels",
                inputs.get(
                    "override_video_max_pixels",
                    self.default_video_override_max_pixels,
                ),
            )
            use_audio_in_video = self._resolve_use_audio_in_video(inputs, raw_videos)
            video_seconds_per_chunk = inputs.get(
                "video_seconds_per_chunk",
                getattr(self, "default_video_seconds_per_chunk", None),
            )
            video_position_id_per_seconds = inputs.get(
                "video_position_id_per_seconds",
                getattr(self, "default_video_position_id_per_seconds", None),
            )
            audio_from_video = False
            num_explicit_audios = 0
            resolved_image_min_pixels = (
                int(image_min_pixels) if image_min_pixels is not None else None
            )
            resolved_image_max_pixels = (
                int(image_max_pixels) if image_max_pixels is not None else None
            )
            resolved_video_fps = float(video_fps) if video_fps is not None else None
            resolved_video_max_frames = (
                int(video_max_frames) if video_max_frames is not None else None
            )
            resolved_video_min_frames = (
                int(video_min_frames) if video_min_frames is not None else None
            )
            resolved_video_min_pixels = (
                int(video_min_pixels) if video_min_pixels is not None else None
            )
            resolved_video_max_pixels = (
                int(video_max_pixels) if video_max_pixels is not None else None
            )
            resolved_video_total_pixels = (
                int(video_total_pixels) if video_total_pixels is not None else None
            )
            resolved_video_override_max_pixels = bool(video_override_max_pixels)
            if (
                resolved_video_override_max_pixels
                and resolved_video_total_pixels is None
                and self.max_seq_len is not None
            ):
                resolved_video_total_pixels = derive_video_total_pixels_from_mm_len(
                    int(self.max_seq_len)
                )
            resolved_video_seconds_per_chunk = (
                float(video_seconds_per_chunk)
                if video_seconds_per_chunk is not None
                else None
            )
            resolved_video_position_id_per_seconds = (
                float(video_position_id_per_seconds)
                if video_position_id_per_seconds is not None
                else None
            )

            # Compute cache keys BEFORE conversion (paths are cheap to hash)
            image_cache_key = self._media_cache_key_for_request(
                request_inputs=inputs,
                modality="image",
                raw_value=raw_images,
                compute_cache_key=compute_image_cache_key,
            )
            raw_audio_cache_key = self._media_cache_key_for_request(
                request_inputs=inputs,
                modality="audio",
                raw_value=raw_audios,
                compute_cache_key=compute_audio_cache_key,
            )
            video_cache_key = self._media_cache_key_for_request(
                request_inputs=inputs,
                modality="video",
                raw_value=raw_videos,
                compute_cache_key=compute_video_cache_key,
            )

            # Count explicit audio inputs (for placeholder insertion)
            num_explicit_audios = _media_item_count(raw_audios)
            explicit_audios_for_mask: Any = []
            video_audios_for_mask: list[Any] = []

            # Use async versions for concurrent loading
            # If video audio is needed, extract it during video loading to avoid
            # duplicate downloads.
            extract_audio_from_video_flag: Any = False
            if _media_value_is_present(raw_videos):
                extract_audio_from_video_flag = (
                    self._processor_use_audio_in_video_value(use_audio_in_video)
                )

            images, videos_result, audios_result = await asyncio.gather(
                ensure_image_list_async(raw_images),
                ensure_video_list_async(
                    raw_videos,
                    fps=resolved_video_fps,
                    max_frames=resolved_video_max_frames,
                    min_frames=resolved_video_min_frames,
                    min_pixels=resolved_video_min_pixels,
                    max_pixels=resolved_video_max_pixels,
                    total_pixels=resolved_video_total_pixels,
                    override_max_pixels=resolved_video_override_max_pixels,
                    extract_audio=extract_audio_from_video_flag,
                    audio_target_sr=audio_target_sr,
                ),
                ensure_audio_list_async(raw_audios, target_sr=audio_target_sr),
            )
            videos, sampled_video_fps, extracted_audio_from_video = videos_result
            explicit_audios_for_mask = audios_result
            video_audios_for_mask = extracted_audio_from_video

            audios, audio_from_video = self._merge_audio_inputs_for_request(
                messages=messages,
                explicit_audios=audios_result,
                video_audios=extracted_audio_from_video,
                use_audio_in_video=use_audio_in_video,
            )
        else:
            messages = inputs
            tools = None
            raw_prompt_text = None
            raw_prompt_token_ids = None
            images = []
            videos = []
            audios = []
            image_cache_key = None
            raw_audio_cache_key = None
            video_cache_key = None
            audio_target_sr = getattr(self, "default_audio_target_sr", 16000)
            image_min_pixels = self.default_image_min_pixels
            image_max_pixels = self.default_image_max_pixels
            video_fps = self.default_video_fps
            video_max_frames = self.default_video_max_frames
            video_min_frames = self.default_video_min_frames
            video_min_pixels = self.default_video_min_pixels
            video_max_pixels = self.default_video_max_pixels
            video_total_pixels = self.default_video_total_pixels
            sampled_video_fps = None
            use_audio_in_video = None
            video_seconds_per_chunk = getattr(
                self,
                "default_video_seconds_per_chunk",
                None,
            )
            video_position_id_per_seconds = getattr(
                self,
                "default_video_position_id_per_seconds",
                None,
            )
            audio_from_video = False
            num_explicit_audios = 0
            resolved_image_min_pixels = None
            resolved_image_max_pixels = None
            resolved_video_fps = None
            resolved_video_max_frames = None
            resolved_video_min_frames = None
            resolved_video_min_pixels = None
            resolved_video_max_pixels = None
            resolved_video_total_pixels = None
            resolved_video_seconds_per_chunk = None
            resolved_video_override_max_pixels = False
            resolved_video_position_id_per_seconds = None
            explicit_audios_for_mask = []
            video_audios_for_mask = []

        if raw_prompt_text is not None:
            # 中文说明：vLLM 的 TextPrompt 已经把 chat template 和多模态
            # 占位符渲染进 prompt。这里直接复用，避免再次套模板导致占位符重复。
            prompt_text = raw_prompt_text
        elif raw_prompt_token_ids is not None:
            # 中文说明：vLLM 的 TokensPrompt 已经完成 tokenization。HF
            # processor 仍用于多模态特征抽取，最终 input_ids 由调用侧 token
            # 覆盖，避免把 tokenized prompt 反向转文本再编码。
            prompt_text = ""
        else:
            messages_norm = normalize_messages(messages)
            # Insert placeholders:
            # - Explicit audio files get independent audio placeholders
            # - Video audio (when use_audio_in_video=True) is handled by video token,
            #   no separate placeholder
            num_audios_for_placeholder = num_explicit_audios
            messages_mm = self._build_multimodal_messages(
                messages_norm,
                num_images=len(images),
                num_audios=num_audios_for_placeholder,
                num_videos=len(videos),
            )
            chat_template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if tools is not None:
                # 中文说明：Qwen3.5 vLLM 离线 function-call 示例会把 tools
                # 交给 chat template 渲染；这里仅传递 schema，不执行工具。
                chat_template_kwargs["tools"] = tools
            prompt_text = self.processor.apply_chat_template(
                messages_mm,
                **chat_template_kwargs,
            )

        videos_kwargs: dict[str, Any] = {}
        if sampled_video_fps is not None:
            videos_kwargs["fps"] = (
                sampled_video_fps[0]
                if len(sampled_video_fps) == 1
                else sampled_video_fps
            )
        elif resolved_video_fps is not None:
            videos_kwargs["fps"] = resolved_video_fps
        if resolved_video_max_frames is not None:
            videos_kwargs["max_frames"] = resolved_video_max_frames
        if resolved_video_min_frames is not None:
            videos_kwargs["min_frames"] = resolved_video_min_frames
        if resolved_video_min_pixels is not None:
            videos_kwargs["min_pixels"] = resolved_video_min_pixels
        if resolved_video_max_pixels is not None:
            videos_kwargs["max_pixels"] = resolved_video_max_pixels
        if resolved_video_total_pixels is not None:
            videos_kwargs["total_pixels"] = resolved_video_total_pixels
        if use_audio_in_video is not None:
            videos_kwargs["use_audio_in_video"] = (
                self._processor_use_audio_in_video_value(use_audio_in_video)
            )
        if resolved_video_seconds_per_chunk is not None:
            videos_kwargs["seconds_per_chunk"] = resolved_video_seconds_per_chunk
        if resolved_video_position_id_per_seconds is not None:
            videos_kwargs["position_id_per_seconds"] = float(
                resolved_video_position_id_per_seconds
            )
        if videos:
            # torchcodec backend expects a non-None device string
            videos_kwargs.setdefault("device", "cpu")
        images_kwargs: dict[str, Any] = {}
        if resolved_image_min_pixels is not None:
            images_kwargs["min_pixels"] = resolved_image_min_pixels
        if resolved_image_max_pixels is not None:
            images_kwargs["max_pixels"] = resolved_image_max_pixels
        processor_kwargs = self._processor_kwargs_for_request(
            request_inputs=inputs,
            images_kwargs=images_kwargs,
            videos_kwargs=videos_kwargs,
            audio_target_sr=audio_target_sr,
        )

        hf_inputs = self.processor(
            text=prompt_text,
            images=images or None,
            videos=videos or None,
            audio=audios or None,
            add_special_tokens=False,
            return_tensors="pt",
            **processor_kwargs,
        )

        if raw_prompt_token_ids is not None:
            input_ids = raw_prompt_token_ids.to(dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = hf_inputs["input_ids"][0]
            attention_mask = hf_inputs.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask[0]
            else:
                attention_mask = torch.ones_like(input_ids)

        validate_prompt_seq_len(
            input_ids,
            max_seq_len=self.max_seq_len,
            max_new_tokens=_resolve_request_max_new_tokens(payload.request.params),
            request_id=payload.request_id,
        )

        full_mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(hf_inputs),
            "audio": _build_audio_mm_inputs_compat(hf_inputs),
            "video": build_video_mm_inputs(hf_inputs),
        }
        if use_audio_in_video is not None:
            full_mm_inputs["video"]["use_audio_in_video"] = (
                self._processor_use_audio_in_video_value(use_audio_in_video)
            )

        # Build encoder_inputs with cache_key for efficient caching.
        # Include preprocessing parameters that materially change encoder outputs.
        image_encoder_inputs = {
            **full_mm_inputs["image"],
            **full_mm_inputs["video"],
        }
        effective_video_fps: tuple[float, ...] | None = None
        if sampled_video_fps is not None:
            effective_video_fps = tuple(float(fps) for fps in sampled_video_fps)
        elif resolved_video_fps is not None:
            effective_video_fps = (resolved_video_fps,)

        contextual_image_cache_key = _contextualize_cache_key(
            image_cache_key,
            min_pixels=resolved_image_min_pixels,
            max_pixels=resolved_image_max_pixels,
        )
        contextual_video_cache_key = _contextualize_cache_key(
            video_cache_key,
            fps=effective_video_fps,
            max_frames=resolved_video_max_frames,
            min_frames=resolved_video_min_frames,
            min_pixels=resolved_video_min_pixels,
            max_pixels=resolved_video_max_pixels,
            total_pixels=resolved_video_total_pixels,
            override_max_pixels=resolved_video_override_max_pixels,
            seconds_per_chunk=resolved_video_seconds_per_chunk,
        )
        combined_cache_key = _combine_cache_keys(
            contextual_image_cache_key, contextual_video_cache_key
        )
        if combined_cache_key:
            image_encoder_inputs["cache_key"] = combined_cache_key

        audio_encoder_inputs = {**full_mm_inputs["audio"]}
        audio_is_dependent = self._audio_is_dependent_for_request(
            request_inputs=inputs,
            messages=messages,
            explicit_audios=explicit_audios_for_mask,
            video_audios=video_audios_for_mask,
            use_audio_in_video=use_audio_in_video,
        )
        if audio_is_dependent is not None:
            full_mm_inputs["audio"]["audio_is_dependent"] = audio_is_dependent
        contextualized_audio_cache_key = _contextualize_cache_key(
            raw_audio_cache_key,
            audio_is_dependent=audio_is_dependent,
            target_sr=audio_target_sr,
            use_audio_in_video=(
                self._processor_use_audio_in_video_value(use_audio_in_video)
                if use_audio_in_video is not None
                else None
            ),
        )
        if audio_from_video:
            contextualized_audio_cache_key = _combine_cache_keys(
                contextualized_audio_cache_key,
                _contextualize_cache_key(
                    video_cache_key,
                    audio_is_dependent=audio_is_dependent,
                    extracted_audio=True,
                    target_sr=audio_target_sr,
                    use_audio_in_video=(
                        self._processor_use_audio_in_video_value(use_audio_in_video)
                        if use_audio_in_video is not None
                        else None
                    ),
                ),
            )
        if contextualized_audio_cache_key:
            audio_encoder_inputs["cache_key"] = contextualized_audio_cache_key

        encoder_inputs: dict[str, dict[str, Any]] = {}
        image_encoder_inputs = {
            k: v for k, v in image_encoder_inputs.items() if v is not None
        }
        if (
            image_encoder_inputs.get("pixel_values") is not None
            or image_encoder_inputs.get("pixel_values_videos") is not None
        ):
            encoder_inputs["image_encoder"] = image_encoder_inputs
        else:
            encoder_inputs["image_encoder"] = {"_skip": True, "_result": {}}
        if audio_encoder_inputs.get("input_features") is not None:
            encoder_inputs["audio_encoder"] = audio_encoder_inputs
        else:
            encoder_inputs["audio_encoder"] = {"_skip": True, "_result": {}}

        state = Qwen3OmniPipelineState(
            mm_inputs=build_lightweight_mm_inputs(full_mm_inputs),
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            encoder_inputs=encoder_inputs,
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        return payload
