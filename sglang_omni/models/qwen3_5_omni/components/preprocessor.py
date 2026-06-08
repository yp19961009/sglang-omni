# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for Qwen3.5-Omni."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Callable

from sglang_omni.models.qwen3_5_omni.components.audio_encoder import (
    DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
    DEFAULT_DOWNSAMPLE_TIMES,
)
from sglang_omni.models.qwen3_5_omni.components.common import (
    load_qwen35_thinker_config,
)

logger = logging.getLogger(__name__)
DEFAULT_AUDIO_TIMESTAMP_INTERVAL = 60
_OPENAI_MEDIA_PLACEHOLDERS = {
    "image": "<|sglang_omni_qwen35_image|>",
    "video": "<|sglang_omni_qwen35_video|>",
    "audio": "<|sglang_omni_qwen35_audio|>",
}
_OPENAI_MEDIA_PLACEHOLDER_TO_TYPE = {
    placeholder: media_type
    for media_type, placeholder in _OPENAI_MEDIA_PLACEHOLDERS.items()
}
_OPENAI_MEDIA_PLACEHOLDER_PATTERN = re.compile(
    "|".join(re.escape(token) for token in _OPENAI_MEDIA_PLACEHOLDERS.values())
)
_OPENAI_VIDEO_OPTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("video_fps", ("video_fps", "fps")),
    ("video_max_frames", ("video_max_frames", "max_frames")),
    ("video_min_frames", ("video_min_frames", "min_frames")),
    ("video_min_pixels", ("video_min_pixels", "min_pixels")),
    ("video_max_pixels", ("video_max_pixels", "max_pixels")),
    ("video_total_pixels", ("video_total_pixels", "total_pixels")),
    ("use_audio_in_video", ("use_audio_in_video",)),
    ("video_seconds_per_chunk", ("video_seconds_per_chunk", "seconds_per_chunk")),
    (
        "video_position_id_per_seconds",
        ("video_position_id_per_seconds", "position_id_per_seconds"),
    ),
)
_OPENAI_IMAGE_OPTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("image_min_pixels", ("image_min_pixels", "min_pixels")),
    ("image_max_pixels", ("image_max_pixels", "max_pixels")),
)
_OPENAI_USE_AUDIO_IN_VIDEO_VALUES = "_openai_use_audio_in_video_values"
_AUDIO_PROCESSOR_OPTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("audio_target_sr", ("audio_target_sr", "audio_sampling_rate", "sampling_rate")),
    (
        "audio_timestamp_interval",
        ("audio_timestamp_interval", "timestamp_interval"),
    ),
    ("audio_downsample_times", ("audio_downsample_times", "downsample_times")),
    (
        "audio_downsample_chunk_size",
        ("audio_downsample_chunk_size", "downsample_chunk_size"),
    ),
)
_AUDIO_REQUEST_INPUT_ALIASES = (
    "audios",
    "input_audios",
    "input_audio",
    "audio_urls",
    "audio_url",
    "audio",
)
_AUDIO_REQUEST_PARAM_INPUT_ALIASES = _AUDIO_REQUEST_INPUT_ALIASES[:-1]
_IMAGE_REQUEST_INPUT_ALIASES = (
    "images",
    "input_images",
    "input_image",
    "image_urls",
    "image_url",
    "image",
)
_VIDEO_REQUEST_INPUT_ALIASES = (
    "videos",
    "input_videos",
    "input_video",
    "video_urls",
    "video_url",
    "video",
)
_REQUEST_MEDIA_INPUT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("images", _IMAGE_REQUEST_INPUT_ALIASES),
    ("videos", _VIDEO_REQUEST_INPUT_ALIASES),
    ("audios", _AUDIO_REQUEST_PARAM_INPUT_ALIASES),
)
_REQUEST_PARAM_INPUT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    *_OPENAI_VIDEO_OPTION_ALIASES,
    *_OPENAI_IMAGE_OPTION_ALIASES,
    ("dependent_audio", ("dependent_audio", "video_dependent_audio")),
    *_AUDIO_PROCESSOR_OPTION_ALIASES,
)
_VLLM_MM_DATA_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("images", _IMAGE_REQUEST_INPUT_ALIASES),
    ("videos", _VIDEO_REQUEST_INPUT_ALIASES),
    ("audios", _AUDIO_REQUEST_INPUT_ALIASES),
)
_VLLM_VIDEO_EXTRA_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("video_metadata", ("video_metadata", "videos_metadata")),
    ("return_video_metadata", ("return_video_metadata", "return_metadata")),
)
_VLLM_UUID_ALIASES = {
    "image": _IMAGE_REQUEST_INPUT_ALIASES,
    "video": _VIDEO_REQUEST_INPUT_ALIASES,
    "audio": _AUDIO_REQUEST_INPUT_ALIASES,
}
_REQUEST_BOOL_TRUE = {"1", "true", "yes", "on"}
_REQUEST_BOOL_FALSE = {"0", "false", "no", "off"}
_REQUEST_BOOL_WORD_TRUE = {"true", "yes", "on"}
_REQUEST_BOOL_WORD_FALSE = {"false", "no", "off"}


@dataclass(frozen=True)
class _OrderedAudioInputs:
    audio: list[Any]
    used_video_audio: bool
    audio_is_dependent: list[bool]


def _request_bool_string(value: Any) -> bool | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _REQUEST_BOOL_TRUE:
        return True
    if normalized in _REQUEST_BOOL_FALSE:
        return False
    return None


def _request_bool_word_string(value: Any) -> bool | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _REQUEST_BOOL_WORD_TRUE:
        return True
    if normalized in _REQUEST_BOOL_WORD_FALSE:
        return False
    return None


def _request_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    parsed = _request_bool_string(value)
    if parsed is not None:
        return parsed
    return bool(value)


def _request_bool_value_or_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_request_bool_value(item) for item in value]
    return _request_bool_value(value)


class _Qwen35ProcessorShim:
    """Compose a working processor from Qwen2VL components.

    transformers lacks Qwen3OmniNextProcessor and the checkpoint has no
    custom processor code. The Qwen3.5-Omni vision pipeline reuses
    Qwen2VL image/video processing, so Qwen2VLProcessor works.
    """

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        from transformers import (
            AutoTokenizer,
            Qwen2VLImageProcessor,
            Qwen2VLProcessor,
            Qwen2VLVideoProcessor,
        )

        kw = {
            k: v
            for k, v in kwargs.items()
            if k in ("trust_remote_code", "local_files_only")
        }
        tok = AutoTokenizer.from_pretrained(model_dir, **kw)
        img = Qwen2VLImageProcessor.from_pretrained(model_dir, **kw)
        vid = Qwen2VLVideoProcessor()
        return Qwen2VLProcessor(
            tokenizer=tok,
            image_processor=img,
            video_processor=vid,
            chat_template=tok.chat_template,
        )


def _load_qwen3_omni_next_processor():
    try:
        from transformers.models.qwen3_omni_next.processing_qwen3_omni_next import (
            Qwen3OmniNextProcessor,
        )
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith(
            "transformers.models.qwen3_omni_next"
        ):
            raise
        return _Qwen35ProcessorShim
    return Qwen3OmniNextProcessor


from sglang_omni.models.qwen3_omni.components.preprocessor import (  # noqa: E402
    Qwen3OmniPreprocessor,
    _first_present_audio_input,
    _first_present_media_input,
    _media_item_count,
    _media_value_is_present,
)


class Qwen35OmniPreprocessor(Qwen3OmniPreprocessor):
    """CPU-side preprocessing using HF's Qwen3OmniNextProcessor."""

    chat_template_fallback_model_paths: tuple[str, ...] = ()

    def __init__(
        self,
        *args: Any,
        limit_mm_per_prompt: dict[str, int] | None = None,
        audio_timestamp_interval: int | None = None,
        audio_downsample_times: int | None = None,
        audio_downsample_chunk_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.limit_mm_per_prompt = _normalize_limit_mm_per_prompt(
            limit_mm_per_prompt
        )
        super().__init__(*args, **kwargs)
        self._audio_processor_defaults = self._load_audio_processor_defaults()
        _apply_audio_processor_default_overrides(
            self._audio_processor_defaults,
            timestamp_interval=audio_timestamp_interval,
            downsample_times=audio_downsample_times,
            downsample_chunk_size=audio_downsample_chunk_size,
        )

    @classmethod
    def _load_processor_cls(cls):
        return _load_qwen3_omni_next_processor()

    async def _call_impl(self, payload):
        request_inputs = _merge_request_params_into_inputs(
            payload.request.inputs,
            payload.request.params,
        )
        request_inputs = _normalize_vllm_multimodal_inputs(request_inputs)
        normalized_inputs = _normalize_openai_multimodal_inputs(request_inputs)
        normalized_inputs = _normalize_request_level_media_aliases(normalized_inputs)
        self._validate_limit_mm_per_prompt(
            normalized_inputs,
            request_id=payload.request_id,
        )
        if (
            request_inputs is payload.request.inputs
            and normalized_inputs is request_inputs
        ):
            return await super()._call_impl(payload)

        request = payload.request.__class__(
            inputs=normalized_inputs,
            params=payload.request.params,
            metadata=payload.request.metadata,
        )
        normalized_payload = payload.__class__(
            request_id=payload.request_id,
            request=request,
            data=payload.data,
        )
        return await super()._call_impl(normalized_payload)

    def _validate_limit_mm_per_prompt(
        self,
        request_inputs: Any,
        *,
        request_id: str | None = None,
    ) -> None:
        limit_mm_per_prompt = getattr(self, "limit_mm_per_prompt", {})
        if not limit_mm_per_prompt or not isinstance(request_inputs, dict):
            return
        counts = _count_limit_mm_inputs(request_inputs)
        for modality, limit in limit_mm_per_prompt.items():
            count = counts.get(modality, 0)
            if count <= limit:
                continue
            prefix = f"request {request_id}: " if request_id else ""
            raise ValueError(
                f"{prefix}Qwen3.5 limit_mm_per_prompt exceeded for "
                f"{modality}: {count} > {limit}"
            )

    def _load_audio_processor_defaults(self) -> dict[str, int]:
        defaults: dict[str, int] = {
            "downsample_times": DEFAULT_DOWNSAMPLE_TIMES,
            "downsample_chunk_size": DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
            "timestamp_interval": DEFAULT_AUDIO_TIMESTAMP_INTERVAL,
        }
        try:
            thinker_config = load_qwen35_thinker_config(self.model_dir)
            audio_config = thinker_config.audio_config
        except Exception as exc:
            logger.debug(
                "Qwen3.5-Omni audio processor config fallback to defaults: %s",
                exc,
            )
            return defaults

        for key, aliases in {
            "downsample_times": ("downsample_times",),
            "downsample_chunk_size": (
                "downsample_chunk_size",
                "chunk_size",
            ),
            "timestamp_interval": ("timestamp_interval",),
        }.items():
            value = _first_config_value(audio_config, aliases)
            if value is not None:
                # 中文说明：Qwen3OmniNextProcessor 会直接 pop 这三个键；
                # 即使模型 config 暂时缺字段，也保留 vLLM 默认值，
                # 避免真实调用时报错。
                defaults[key] = int(value)
        return defaults

    def _processor_kwargs_for_request(
        self,
        *,
        request_inputs: Any,
        images_kwargs: dict[str, Any],
        videos_kwargs: dict[str, Any],
        audio_target_sr: int,
    ) -> dict[str, Any]:
        processor_kwargs = super()._processor_kwargs_for_request(
            request_inputs=request_inputs,
            images_kwargs=images_kwargs,
            videos_kwargs=videos_kwargs,
            audio_target_sr=audio_target_sr,
        )
        if videos_kwargs or _request_has_video_inputs(request_inputs):
            video_kwargs = dict(processor_kwargs.get("videos_kwargs", {}))
            if isinstance(request_inputs, dict):
                _copy_optional_value(
                    video_kwargs,
                    request_inputs,
                    target="video_metadata",
                    aliases=("video_metadata", "videos_metadata"),
                )
                return_metadata = _first_param_value(
                    request_inputs,
                    ("return_video_metadata", "return_metadata"),
                )
                if return_metadata is not None:
                    video_kwargs["return_metadata"] = _request_bool_value(
                        return_metadata
                    )
            video_audio_requested = (
                _dependent_audio_requests_video_audio(request_inputs)
                or _use_audio_in_video_enabled(video_kwargs)
            )
            if video_audio_requested:
                # 中文说明：vLLM perf_v2 的 dependent_audio 是 engine 层
                # mm_processor_kwargs，不是 HF video_processor 的参数。sglang
                # 当前在 CPU preprocessor 里直接完成 token 展开，因此这里仅把
                # 它当作“视频使用内置音轨”的语义信号，不传给 HF processor。
                video_kwargs.setdefault("use_audio_in_video", True)
            video_kwargs.setdefault("return_metadata", True)
            processor_kwargs["videos_kwargs"] = video_kwargs

        audio_kwargs = {
            "sampling_rate": int(audio_target_sr),
            # 中文说明：对齐 vLLM perf_v2 Qwen3OmniNextProcessorKwargs 默认值。
            # 即使未来 remote processor 的默认合并逻辑有差异，SGLang 也会
            # 稳定产出 attention mask，并避免音频被 feature_extractor 截断。
            "padding": True,
            "return_attention_mask": True,
            "truncation": False,
        }
        audio_kwargs.update(getattr(self, "_audio_processor_defaults", {}))
        if isinstance(request_inputs, dict):
            _copy_audio_override(
                audio_kwargs,
                request_inputs,
                target="timestamp_interval",
                aliases=("audio_timestamp_interval", "timestamp_interval"),
            )
            _copy_audio_override(
                audio_kwargs,
                request_inputs,
                target="downsample_times",
                aliases=("audio_downsample_times", "downsample_times"),
            )
            _copy_audio_override(
                audio_kwargs,
                request_inputs,
                target="downsample_chunk_size",
                aliases=(
                    "audio_downsample_chunk_size",
                    "downsample_chunk_size",
                ),
            )

        # 中文说明：Qwen3OmniNextProcessor 会用这些参数展开音频 token。
        # 保持 processor 和 audio encoder 的 downsample 契约一致，避免
        # placeholder 数量和 encoder 输出长度在真实权重下错位。
        processor_kwargs["audio_kwargs"] = audio_kwargs
        return processor_kwargs

    def _resolve_use_audio_in_video(
        self,
        request_inputs: dict[str, Any],
        raw_videos: Any,
    ) -> Any:
        # 中文说明：Qwen3OmniNextProcessor 默认 use_audio_in_video=False。
        # 保持“请求显式开启才抽取视频音轨”的语义；dependent_audio 也是
        # vLLM Qwen3.5 显式入口，因此可作为开启信号。
        value = super()._resolve_use_audio_in_video(request_inputs, raw_videos)
        if value is not None:
            return _request_bool_value_or_list(value)
        return _dependent_audio_requests_video_audio(request_inputs) or None

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
        num_videos: int,
    ) -> list[dict[str, Any]]:
        if _messages_have_openai_media_placeholders(messages):
            return _restore_openai_media_placeholders(messages)
        return super()._build_multimodal_messages(
            messages,
            num_images=num_images,
            num_audios=num_audios,
            num_videos=num_videos,
        )

    def _merge_audio_inputs_for_request(
        self,
        *,
        messages: Any,
        explicit_audios: Any,
        video_audios: list[Any] | None,
        use_audio_in_video: Any,
    ) -> tuple[Any, bool]:
        explicit_audio_list = _as_media_list(explicit_audios)
        video_audio_list = list(video_audios or [])
        if _messages_have_openai_media_placeholders(messages):
            ordered = _ordered_audio_for_openai_media_placeholders(
                messages=messages,
                explicit_audios=explicit_audio_list,
                video_audios=video_audio_list,
                use_audio_in_video=use_audio_in_video,
            )
        else:
            ordered = _ordered_audio_for_qwen35_default_placeholders(
                explicit_audios=explicit_audio_list,
                video_audios=video_audio_list,
                use_audio_in_video=use_audio_in_video,
            )
        if ordered is None:
            return super()._merge_audio_inputs_for_request(
                messages=messages,
                explicit_audios=explicit_audios,
                video_audios=video_audios,
                use_audio_in_video=use_audio_in_video,
            )
        return ordered.audio, ordered.used_video_audio

    def _audio_is_dependent_for_request(
        self,
        *,
        request_inputs: Any | None = None,
        messages: Any,
        explicit_audios: Any,
        video_audios: list[Any] | None,
        use_audio_in_video: Any,
    ) -> list[bool] | None:
        explicit_audio_list = _as_media_list(explicit_audios)
        video_audio_list = list(video_audios or [])
        if _messages_have_openai_media_placeholders(messages):
            ordered = _ordered_audio_for_openai_media_placeholders(
                messages=messages,
                explicit_audios=explicit_audio_list,
                video_audios=video_audio_list,
                use_audio_in_video=use_audio_in_video,
            )
        else:
            ordered = _ordered_audio_for_qwen35_default_placeholders(
                explicit_audios=explicit_audio_list,
                video_audios=video_audio_list,
                use_audio_in_video=use_audio_in_video,
            )
        if ordered is not None:
            return ordered.audio_is_dependent or None
        dependent_audio = _request_dependent_audio(request_inputs)
        if dependent_audio is not None and explicit_audio_list:
            return _dependent_audio_mask(dependent_audio, len(explicit_audio_list))
        if explicit_audio_list:
            return [False] * len(explicit_audio_list)
        return None

    def _processor_use_audio_in_video_value(self, use_audio_in_video: Any) -> Any:
        if isinstance(use_audio_in_video, (list, tuple)):
            return [_request_bool_value(item) for item in use_audio_in_video]
        return _request_bool_value(use_audio_in_video)

    def _media_cache_key_for_request(
        self,
        *,
        request_inputs: dict[str, Any],
        modality: str,
        raw_value: Any,
        compute_cache_key: Callable[[Any], str | None],
    ) -> str | None:
        media_count = _media_item_count(raw_value)
        uuid_key = _vllm_uuid_cache_key(
            request_inputs,
            modality,
            expected_count=media_count,
        )
        if uuid_key is not None and _media_value_is_present(raw_value):
            return uuid_key
        return super()._media_cache_key_for_request(
            request_inputs=request_inputs,
            modality=modality,
            raw_value=raw_value,
            compute_cache_key=compute_cache_key,
        )


def _first_config_value(config: Any, aliases: tuple[str, ...]) -> Any | None:
    for name in aliases:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _normalize_openai_multimodal_inputs(inputs: Any) -> Any:
    """Extract OpenAI content-part media into the existing top-level fields."""

    if isinstance(inputs, dict):
        messages = inputs.get("messages")
        normalized = _normalize_openai_messages(
            messages,
            preserve_media_placeholders=not _request_has_top_level_media(inputs),
        )
        if normalized is None:
            return inputs
        new_inputs = dict(inputs)
        new_inputs["messages"] = normalized["messages"]
        _merge_extracted_media(new_inputs, normalized)
        _merge_extracted_image_options(new_inputs, normalized)
        _merge_extracted_video_options(new_inputs, normalized)
        _merge_extracted_audio_options(new_inputs, normalized)
        return new_inputs

    normalized = _normalize_openai_messages(inputs)
    if normalized is None:
        return inputs
    has_media = any(normalized[key] for key in ("images", "videos", "audios"))
    has_image_options = bool(normalized.get("image_options"))
    has_video_options = bool(normalized.get("video_options"))
    has_audio_options = bool(normalized.get("audio_options"))
    if (
        not has_media
        and not has_image_options
        and not has_video_options
        and not has_audio_options
    ):
        return normalized["messages"]
    result = {
        "messages": normalized["messages"],
        **{
            key: normalized[key]
            for key in ("images", "videos", "audios")
            if normalized[key]
        },
    }
    _merge_extracted_image_options(result, normalized)
    _merge_extracted_video_options(result, normalized)
    _merge_extracted_audio_options(result, normalized)
    return result


def _apply_audio_processor_default_overrides(
    defaults: dict[str, int],
    *,
    timestamp_interval: int | None,
    downsample_times: int | None,
    downsample_chunk_size: int | None,
) -> None:
    # 中文说明：这些是服务级默认值；请求里的 audio_kwargs/params 仍会在
    # _processor_kwargs_for_request 中覆盖它们。
    if timestamp_interval is not None:
        defaults["timestamp_interval"] = int(timestamp_interval)
    if downsample_times is not None:
        defaults["downsample_times"] = int(downsample_times)
    if downsample_chunk_size is not None:
        defaults["downsample_chunk_size"] = int(downsample_chunk_size)


def _normalize_vllm_multimodal_inputs(inputs: Any) -> Any:
    """Lift vLLM TextPrompt fields into sglang-omni's request shape."""

    if not isinstance(inputs, dict):
        return inputs
    multi_modal_data = inputs.get("multi_modal_data")
    mm_processor_kwargs = inputs.get("mm_processor_kwargs")
    mm_data_items = _vllm_mapping_sequence(multi_modal_data)
    mm_kwargs_items = _vllm_mapping_sequence(mm_processor_kwargs)
    if not mm_data_items and not mm_kwargs_items:
        return inputs

    normalized = dict(inputs)
    changed = False

    if mm_data_items:
        for target_key, aliases in _VLLM_MM_DATA_ALIASES:
            value = _first_present_vllm_media_input(mm_data_items, aliases)
            if not _media_value_is_present(value):
                continue
            if _has_request_media_input_value(normalized, target_key, aliases):
                continue
            # 中文说明：vLLM TextPrompt 使用 multi_modal_data 承载已经加载/
            # 解码过的媒体；提升到 sglang 顶层后，复用现有 tensor-safe
            # 媒体加载和 HF processor 路径。list[dict] 形态来自 vLLM
            # 通用 PromptType，这里按请求内顺序展开，避免外部复用时丢输入。
            normalized[target_key] = value
            changed = True

    for kwargs_item in mm_kwargs_items:
        changed = (
            _merge_mm_processor_kwargs_into_inputs(normalized, kwargs_item)
            or changed
        )

    return normalized if changed else inputs


def _vllm_mapping_sequence(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, dict) for item in value
    ):
        return list(value)
    return []


def _vllm_uuid_cache_key(
    inputs: dict[str, Any],
    modality: str,
    *,
    expected_count: int | None = None,
) -> str | None:
    aliases = _VLLM_UUID_ALIASES.get(modality)
    if aliases is None:
        return None
    uuid_items = _vllm_mapping_sequence(inputs.get("multi_modal_uuids"))
    if not uuid_items:
        return None

    values: list[Any] = []
    for item in uuid_items:
        value = _first_present_media_input(item, aliases)
        if _media_value_is_present(value):
            values.extend(_as_uuid_list(value))
    if not values:
        return None
    if expected_count is not None and len(values) != expected_count:
        # 中文说明：partial uuid 不能安全代表完整媒体集合，否则两个请求只要
        # 共享第一个 uuid，就可能错误复用后续不同视频/音频的 encoder 输出。
        return None

    # 中文说明：vLLM 的 multi_modal_uuids 是外部资源缓存标识。这里把它
    # 转成 encoder cache key 的稳定前缀，后续仍会叠加 fps/max_pixels 等
    # processor 参数，避免同一 uuid 在不同采样配置下误复用。
    encoded = json.dumps(
        [_json_cache_value(value) for value in values],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"vllm_uuid:{modality}:{encoded}"


def _as_uuid_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _json_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_cache_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_json_cache_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_present_vllm_media_input(
    sources: list[dict[str, Any]],
    aliases: tuple[str, ...],
) -> Any | None:
    if len(sources) == 1:
        return _first_present_media_input(sources[0], aliases)

    values: list[Any] = []
    for source in sources:
        value = _first_present_media_input(source, aliases)
        if _media_value_is_present(value):
            values.extend(_as_media_list(value))
    return values or None


def _merge_mm_processor_kwargs_into_inputs(
    inputs: dict[str, Any],
    mm_processor_kwargs: dict[str, Any],
) -> bool:
    changed = False
    # 中文说明：vLLM perf_v2 的 mm_processor_kwargs 多数是平铺字段，
    # 例如 fps/use_audio_in_video/dependent_audio；同时兼容 HF 风格的
    # videos_kwargs/images_kwargs/audio_kwargs 嵌套写法。
    changed = (
        _merge_processor_aliases(
            inputs,
            mm_processor_kwargs,
            _REQUEST_PARAM_INPUT_ALIASES,
        )
        or changed
    )
    changed = (
        _merge_processor_aliases(
            inputs,
            mm_processor_kwargs,
            _VLLM_VIDEO_EXTRA_ALIASES,
        )
        or changed
    )
    for nested_key, aliases in (
        ("videos_kwargs", (*_OPENAI_VIDEO_OPTION_ALIASES, *_VLLM_VIDEO_EXTRA_ALIASES)),
        ("images_kwargs", _OPENAI_IMAGE_OPTION_ALIASES),
        ("audio_kwargs", _AUDIO_PROCESSOR_OPTION_ALIASES),
    ):
        nested = mm_processor_kwargs.get(nested_key)
        if isinstance(nested, dict):
            changed = _merge_processor_aliases(inputs, nested, aliases) or changed
    return changed


def _merge_processor_aliases(
    inputs: dict[str, Any],
    source: dict[str, Any],
    aliases_table: tuple[tuple[str, tuple[str, ...]], ...],
) -> bool:
    changed = False
    for target_key, aliases in aliases_table:
        value = _first_param_value(source, aliases)
        if value is None or _has_request_input_value(inputs, (target_key, *aliases)):
            continue
        inputs[target_key] = _normalize_processor_alias_value(target_key, value)
        changed = True
    return changed


def _normalize_processor_alias_value(target_key: str, value: Any) -> Any:
    if target_key in {"dependent_audio", "use_audio_in_video", "video_metadata"}:
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _merge_request_params_into_inputs(inputs: Any, params: Any) -> Any:
    if not isinstance(params, dict):
        return inputs

    extracted = _request_param_input_values(params)
    if not extracted:
        return inputs
    if isinstance(inputs, dict):
        merged = dict(inputs)
    elif isinstance(inputs, list):
        merged = {"messages": inputs}
    else:
        return inputs

    changed = False
    for target_key, aliases, value in extracted:
        if _has_request_input_value(merged, (target_key, *aliases)):
            continue
        # 中文说明：vLLM Qwen3.5 OpenAI server 把 use_audio_in_video
        # 暴露为请求顶层字段；进入 sglang serve 后可能落在 params。
        # 这里只提升白名单 multimodal 预处理参数，避免污染采样参数。
        merged[target_key] = value
        changed = True

    return merged if changed else inputs


def _request_param_input_values(
    params: dict[str, Any],
) -> list[tuple[str, tuple[str, ...], Any]]:
    values: list[tuple[str, tuple[str, ...], Any]] = []
    for target_key, aliases in (
        _REQUEST_MEDIA_INPUT_ALIASES + _REQUEST_PARAM_INPUT_ALIASES
    ):
        value = _first_param_value(params, aliases)
        if value is not None:
            values.append((target_key, aliases, value))
    return values


def _first_param_value(params: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    for name in aliases:
        value = params.get(name)
        if value is not None:
            return value
    return None


def _has_request_input_value(
    inputs: dict[str, Any],
    aliases: tuple[str, ...],
) -> bool:
    return any(inputs.get(name) is not None for name in aliases)


def _normalize_openai_messages(
    messages: Any,
    *,
    preserve_media_placeholders: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(messages, list):
        return None

    changed = False
    normalized_messages: list[dict[str, Any]] = []
    images: list[Any] = []
    videos: list[Any] = []
    audios: list[Any] = []
    image_options: dict[str, Any] = {}
    video_options: dict[str, Any] = {}
    audio_options: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, dict):
            return None
        content = message.get("content", "")
        if not isinstance(content, list):
            normalized_messages.append(message)
            continue
        changed = True
        content_parts: list[str] = []
        for part in content:
            _collect_openai_content_part(
                part,
                content_parts=content_parts,
                images=images,
                videos=videos,
                audios=audios,
                image_options=image_options,
                video_options=video_options,
                audio_options=audio_options,
                preserve_media_placeholders=preserve_media_placeholders,
            )
        normalized = dict(message)
        normalized["content"] = "".join(content_parts)
        normalized_messages.append(normalized)

    if not changed:
        return None
    return {
        "messages": normalized_messages,
        "images": images,
        "videos": videos,
        "audios": audios,
        "image_options": image_options,
        "video_options": video_options,
        "audio_options": audio_options,
    }


def _collect_openai_content_part(
    part: Any,
    *,
    content_parts: list[str],
    images: list[Any],
    videos: list[Any],
    audios: list[Any],
    image_options: dict[str, Any],
    video_options: dict[str, Any],
    audio_options: dict[str, Any],
    preserve_media_placeholders: bool,
) -> None:
    if isinstance(part, str):
        _append_openai_text_part(content_parts, part)
        return
    if not isinstance(part, dict):
        _append_openai_text_part(content_parts, str(part))
        return

    part_type = str(part.get("type", "")).lower()
    if part_type in {"text", "input_text"} or (
        not part_type and ("text" in part or "input_text" in part)
    ):
        text = _openai_text_part_value(part)
        if text is not None:
            _append_openai_text_part(content_parts, str(text))
        return

    image = _openai_media_value(part, "image", "image_url", media_prefix="image")
    if image is not None:
        images.append(image)
        if preserve_media_placeholders:
            content_parts.append(_OPENAI_MEDIA_PLACEHOLDERS["image"])
        _collect_openai_image_options(part, image_options)
        return
    video = _openai_media_value(
        part,
        "video",
        "video_url",
        "input_video",
        media_prefix="video",
    )
    if video is not None:
        videos.append(video)
        if preserve_media_placeholders:
            content_parts.append(_OPENAI_MEDIA_PLACEHOLDERS["video"])
        _collect_openai_video_options(part, video_options)
        return
    audio = _openai_media_value(
        part,
        "audio",
        "audio_url",
        "input_audio",
        media_prefix="audio",
    )
    if audio is not None:
        audios.append(audio)
        if preserve_media_placeholders:
            content_parts.append(_OPENAI_MEDIA_PLACEHOLDERS["audio"])
        _collect_openai_audio_options(part, audio_options)
        return

    text = _openai_text_part_value(part)
    if text is not None:
        _append_openai_text_part(content_parts, str(text))


def _openai_text_part_value(part: dict[str, Any]) -> Any | None:
    if "text" in part:
        return part.get("text")
    # 中文说明：OpenAI Responses 风格 content part 常用 input_text；
    # Qwen3.5 仍只需要最终文本内容，统一并入普通 text part。
    return part.get("input_text")


def _append_openai_text_part(content_parts: list[str], text: str) -> None:
    if not text:
        return
    if content_parts and content_parts[-1] not in _OPENAI_MEDIA_PLACEHOLDER_TO_TYPE:
        # 中文说明：连续 OpenAI text parts 等价于一个普通文本消息；
        # 保留换行，避免把两个独立文本片段直接粘在一起。
        content_parts[-1] = f"{content_parts[-1]}\n{text}"
    else:
        content_parts.append(text)


def _messages_have_openai_media_placeholders(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) and _OPENAI_MEDIA_PLACEHOLDER_PATTERN.search(content):
            return True
    return False


def _restore_openai_media_placeholders(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    restored_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str) or not _OPENAI_MEDIA_PLACEHOLDER_PATTERN.search(
            content
        ):
            restored_messages.append(message)
            continue

        content_parts: list[dict[str, Any]] = []
        pos = 0
        for match in _OPENAI_MEDIA_PLACEHOLDER_PATTERN.finditer(content):
            text = content[pos : match.start()]
            if text:
                content_parts.append({"type": "text", "text": text})
            content_parts.append(
                {"type": _OPENAI_MEDIA_PLACEHOLDER_TO_TYPE[match.group(0)]}
            )
            pos = match.end()
        trailing_text = content[pos:]
        if trailing_text:
            content_parts.append({"type": "text", "text": trailing_text})
        if not content_parts:
            content_parts.append({"type": "text", "text": ""})

        restored = dict(message)
        # 中文说明：HF Qwen3OmniNextProcessor 的 chat template 识别
        # {"type": "video/audio/image"}，这能保留 OpenAI content part 原始顺序。
        restored["content"] = content_parts
        restored_messages.append(restored)
    return restored_messages


def _as_media_list(value: Any) -> list[Any]:
    if not _media_value_is_present(value):
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _first_present_qwen35_audio_input_key(inputs: dict[str, Any]) -> str | None:
    for key in _AUDIO_REQUEST_INPUT_ALIASES[:-1]:
        if _media_value_is_present(inputs.get(key)):
            return key
    if _first_present_audio_input({"audio": inputs.get("audio")}) is not None:
        return "audio"
    return None


def _first_present_qwen35_audio_input(inputs: dict[str, Any]) -> Any | None:
    key = _first_present_qwen35_audio_input_key(inputs)
    return inputs.get(key) if key is not None else None


def _media_prefix_for_target_key(target_key: str) -> str:
    return {
        "images": "image",
        "videos": "video",
        "audios": "audio",
    }[target_key]


def _normalize_request_media_payload(
    value: Any,
    *,
    media_prefix: str,
) -> tuple[Any, bool]:
    if isinstance(value, list):
        changed = False
        normalized = []
        for item in value:
            item_value, item_changed = _normalize_request_media_payload(
                item,
                media_prefix=media_prefix,
            )
            normalized.append(item_value)
            changed = changed or item_changed
        return (normalized, changed)
    if isinstance(value, tuple):
        normalized, changed = _normalize_request_media_payload(
            list(value),
            media_prefix=media_prefix,
        )
        return (normalized, changed)
    normalized_value = _media_payload_value(value, media_prefix=media_prefix)
    return (normalized_value, normalized_value is not value)


def _first_present_request_media_input(
    inputs: dict[str, Any],
    target_key: str,
    aliases: tuple[str, ...],
) -> Any | None:
    if target_key == "audios":
        value = _first_present_qwen35_audio_input(inputs)
        if value is None:
            return None
        normalized, _changed = _normalize_request_media_payload(
            value,
            media_prefix="audio",
        )
        return normalized
    for key in aliases:
        value = inputs.get(key)
        if _media_value_is_present(value):
            normalized, _changed = _normalize_request_media_payload(
                value,
                media_prefix=_media_prefix_for_target_key(target_key),
            )
            return normalized
    return None


def _has_request_media_input_value(
    inputs: dict[str, Any],
    target_key: str,
    aliases: tuple[str, ...],
) -> bool:
    return _first_present_request_media_input(inputs, target_key, aliases) is not None


def _normalize_request_level_media_aliases(inputs: Any) -> Any:
    if not isinstance(inputs, dict):
        return inputs
    normalized: dict[str, Any] | None = None
    for target_key, aliases in _REQUEST_MEDIA_INPUT_ALIASES:
        if _media_value_is_present(inputs.get(target_key)):
            current_value, changed = _normalize_request_media_payload(
                inputs[target_key],
                media_prefix=_media_prefix_for_target_key(target_key),
            )
            if changed:
                if normalized is None:
                    normalized = dict(inputs)
                normalized[target_key] = current_value
            continue
        value = _first_present_request_media_input(inputs, target_key, aliases)
        if value is None:
            continue
        if normalized is None:
            normalized = dict(inputs)
        # 中文说明：Qwen3 基类只读取 images/videos/audios。这里把
        # Qwen3.5/vLLM 常见的 input_*/*_url 统一成复数字段，避免直连
        # client 或自定义压测脚本漏输入。
        normalized[target_key] = value
    return normalized if normalized is not None else inputs


_LIMIT_MM_MODALITY_ALIASES = {
    "image": _IMAGE_REQUEST_INPUT_ALIASES,
    "video": _VIDEO_REQUEST_INPUT_ALIASES,
    "audio": _AUDIO_REQUEST_INPUT_ALIASES,
}


def _normalize_limit_mm_per_prompt(
    value: dict[str, int] | None,
) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("limit_mm_per_prompt must be a dict")

    normalized: dict[str, int] = {}
    for raw_key, raw_limit in value.items():
        modality = _normalize_limit_mm_modality(raw_key)
        if modality is None:
            raise ValueError(
                f"limit_mm_per_prompt modality must be one of "
                f"{sorted(_LIMIT_MM_MODALITY_ALIASES)}, got {raw_key!r}"
            )
        limit = int(raw_limit)
        if limit < 0:
            raise ValueError(
                f"limit_mm_per_prompt[{modality!r}] must be >= 0, got {limit}"
            )
        normalized[modality] = limit
    return normalized


def _normalize_limit_mm_modality(value: Any) -> str | None:
    name = str(value).strip().lower()
    for modality, aliases in _LIMIT_MM_MODALITY_ALIASES.items():
        if name in aliases:
            return modality
    return None


def _count_limit_mm_inputs(request_inputs: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for modality, aliases in _LIMIT_MM_MODALITY_ALIASES.items():
        if modality == "audio":
            value = _first_present_qwen35_audio_input(request_inputs)
        else:
            value = _first_present_media_input(request_inputs, aliases)
        counts[modality] = _media_item_count(value)
    return counts


def _ordered_audio_for_openai_media_placeholders(
    *,
    messages: Any,
    explicit_audios: list[Any],
    video_audios: list[Any],
    use_audio_in_video: Any,
) -> _OrderedAudioInputs | None:
    if not isinstance(messages, list):
        return None

    explicit_idx = 0
    video_idx = 0
    saw_placeholder = False
    used_video_audio = False
    ordered_audio: list[Any] = []
    audio_is_dependent: list[bool] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        for match in _OPENAI_MEDIA_PLACEHOLDER_PATTERN.finditer(content):
            saw_placeholder = True
            media_type = _OPENAI_MEDIA_PLACEHOLDER_TO_TYPE[match.group(0)]
            if media_type == "audio":
                if explicit_idx < len(explicit_audios):
                    ordered_audio.append(explicit_audios[explicit_idx])
                    audio_is_dependent.append(False)
                explicit_idx += 1
            elif media_type == "video":
                video_audio = (
                    video_audios[video_idx]
                    if video_idx < len(video_audios)
                    else None
                )
                if (
                    _video_uses_audio(use_audio_in_video, video_idx)
                    and video_audio is not None
                ):
                    ordered_audio.append(video_audio)
                    audio_is_dependent.append(True)
                    used_video_audio = True
                video_idx += 1

    if not saw_placeholder:
        return None

    # 中文说明：OpenAI content part 是最精确的顺序来源。若请求同时混入了
    # 无 placeholder 的顶层音频，作为兜底追加到末尾，避免静默丢输入。
    ordered_audio.extend(explicit_audios[explicit_idx:])
    audio_is_dependent.extend(False for _ in explicit_audios[explicit_idx:])
    for idx, video_audio in enumerate(video_audios[video_idx:], start=video_idx):
        if _video_uses_audio(use_audio_in_video, idx) and video_audio is not None:
            ordered_audio.append(video_audio)
            audio_is_dependent.append(True)
            used_video_audio = True

    return _OrderedAudioInputs(
        audio=ordered_audio,
        used_video_audio=used_video_audio,
        audio_is_dependent=audio_is_dependent,
    )


def _ordered_audio_for_qwen35_default_placeholders(
    *,
    explicit_audios: list[Any],
    video_audios: list[Any],
    use_audio_in_video: Any,
) -> _OrderedAudioInputs | None:
    ordered_audio: list[Any] = []
    audio_is_dependent: list[bool] = []
    used_video_audio = False
    for idx, video_audio in enumerate(video_audios):
        if _video_uses_audio(use_audio_in_video, idx) and video_audio is not None:
            ordered_audio.append(video_audio)
            audio_is_dependent.append(True)
            used_video_audio = True

    if not used_video_audio:
        if any(video_audio is not None for video_audio in video_audios):
            return _OrderedAudioInputs(
                audio=explicit_audios,
                used_video_audio=False,
                audio_is_dependent=[False] * len(explicit_audios),
            )
        return None

    # 中文说明：普通顶层 videos/audios 走父类的模板注入顺序：
    # video 在 audio 前。
    # 因此给 HF processor 的 audio 列表也要先放视频音轨，再放独立音频。
    ordered_audio.extend(explicit_audios)
    audio_is_dependent.extend(False for _ in explicit_audios)
    return _OrderedAudioInputs(
        audio=ordered_audio,
        used_video_audio=True,
        audio_is_dependent=audio_is_dependent,
    )


def _video_uses_audio(use_audio_in_video: Any, video_index: int) -> bool:
    if isinstance(use_audio_in_video, (list, tuple)):
        if video_index >= len(use_audio_in_video):
            return False
        return _request_bool_value(use_audio_in_video[video_index])
    return _request_bool_value(use_audio_in_video)


def _openai_media_value(
    part: dict[str, Any],
    *keys: str,
    media_prefix: str,
) -> Any | None:
    part_type = str(part.get("type", "")).lower()
    candidates = [key for key in keys if key in part]
    if part_type in keys and part_type in part:
        candidates.insert(0, part_type)
    for key in candidates:
        value = _media_payload_value(part.get(key), media_prefix=media_prefix)
        if value is not None:
            return value
    if part_type in keys:
        # 中文说明：兼容轻量 demo/压测脚本直接写
        # {"type": "audio", "url": ...} 或 {"type": "input_video", "data": ...}
        # 的形式。只有 type 明确是当前媒体类型时才读取顶层 url/path/data，
        # 避免未知 content part 被误当成媒体。
        return _media_payload_value(part, media_prefix=media_prefix)
    return None


def _media_payload_value(value: Any, *, media_prefix: str) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        path = value.get("path")
        if path is not None:
            return path
        url = value.get("url")
        if url is not None:
            return url
        data = value.get("data")
        if data is not None:
            media_format = str(value.get("format") or "octet-stream")
            return f"data:{media_prefix}/{media_format};base64,{data}"
        return None
    return value


def _collect_openai_video_options(
    part: dict[str, Any],
    video_options: dict[str, Any],
) -> None:
    # 中文说明：Qwen3OmniNextProcessor 支持 use_audio_in_video 为 bool 或
    # per-video bool list。OpenAI content parts 是逐视频配置，先按视频顺序
    # 记录内部列表，merge 时再折叠成单视频 bool 或多视频 list。
    use_audio_value = _first_openai_video_option(part, ("use_audio_in_video",))
    video_options.setdefault(_OPENAI_USE_AUDIO_IN_VIDEO_VALUES, []).append(
        use_audio_value
    )
    for target_key, aliases in _OPENAI_VIDEO_OPTION_ALIASES:
        if target_key == "use_audio_in_video":
            continue
        if target_key in video_options:
            continue
        value = _first_openai_video_option(part, aliases)
        if value is not None:
            # 中文说明：当前底层 video loader 只支持 request 级别的采样参数。
            # OpenAI content part 里的同名参数先提升到顶层；未来若支持逐视频
            # 参数，可以在这里扩展成 per-video 结构。
            video_options[target_key] = value


def _collect_openai_image_options(
    part: dict[str, Any],
    image_options: dict[str, Any],
) -> None:
    for target_key, aliases in _OPENAI_IMAGE_OPTION_ALIASES:
        if target_key in image_options:
            continue
        value = _first_openai_media_option(
            part,
            aliases,
            nested_keys=("image", "image_url"),
        )
        if value is not None:
            # 中文说明：当前 HF processor 接受 request 级 images_kwargs。
            # 先对齐 vLLM 示例里的 image min/max pixels；多图逐项尺寸未来
            # 如果底层支持，再扩展成 per-image 参数。
            image_options[target_key] = value


def _collect_openai_audio_options(
    part: dict[str, Any],
    audio_options: dict[str, Any],
) -> None:
    for target_key, aliases in _AUDIO_PROCESSOR_OPTION_ALIASES:
        if target_key in audio_options:
            continue
        value = _first_openai_media_option(
            part,
            aliases,
            nested_keys=("audio", "audio_url", "input_audio"),
        )
        if value is not None:
            # 中文说明：OpenAI audio content part 里的采样/下采样参数本质上
            # 是 Qwen3OmniNextProcessor 的 request 级 audio_kwargs。
            audio_options[target_key] = value


def _first_openai_video_option(
    part: dict[str, Any],
    aliases: tuple[str, ...],
) -> Any | None:
    return _first_openai_media_option(
        part,
        aliases,
        nested_keys=("video", "video_url", "input_video"),
    )


def _first_openai_media_option(
    part: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    nested_keys: tuple[str, ...],
) -> Any | None:
    for name in aliases:
        value = part.get(name)
        if value is not None:
            return value
    for key in nested_keys:
        value = part.get(key)
        if not isinstance(value, dict):
            continue
        for name in aliases:
            nested = value.get(name)
            if nested is not None:
                return nested
    return None


def _merge_extracted_media(
    inputs: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    for target_key, aliases in (
        ("images", ("images", "image")),
        ("videos", ("videos", "video")),
        ("audios", _AUDIO_REQUEST_INPUT_ALIASES),
    ):
        extracted = normalized[target_key]
        if not extracted:
            continue
        existing_key = (
            _first_present_qwen35_audio_input_key(inputs)
            if target_key == "audios"
            else next(
                (
                    key
                    for key in aliases
                    if _media_value_is_present(inputs.get(key))
                ),
                None,
            )
        )
        existing_key = existing_key or target_key
        existing = inputs.get(existing_key)
        if _media_value_is_present(existing):
            prefix = existing if isinstance(existing, list) else [existing]
            inputs[existing_key] = [*prefix, *extracted]
        else:
            inputs[target_key] = extracted


def _merge_extracted_video_options(
    inputs: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    for key, value in normalized.get("video_options", {}).items():
        if key == _OPENAI_USE_AUDIO_IN_VIDEO_VALUES:
            continue
        inputs.setdefault(key, value)
    _merge_openai_use_audio_in_video_option(inputs, normalized)


def _merge_extracted_image_options(
    inputs: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    for key, value in normalized.get("image_options", {}).items():
        inputs.setdefault(key, value)


def _merge_extracted_audio_options(
    inputs: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    for key, value in normalized.get("audio_options", {}).items():
        inputs.setdefault(key, value)


def _merge_openai_use_audio_in_video_option(
    inputs: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    if "use_audio_in_video" in inputs:
        return
    values = normalized.get("video_options", {}).get(_OPENAI_USE_AUDIO_IN_VIDEO_VALUES)
    if not isinstance(values, list) or not any(value is not None for value in values):
        return
    resolved = [_request_bool_value(value) for value in values]
    inputs["use_audio_in_video"] = resolved[0] if len(resolved) == 1 else resolved


def _request_has_video_inputs(request_inputs: Any) -> bool:
    if not isinstance(request_inputs, dict):
        return False
    for key in ("videos", "video"):
        value = request_inputs.get(key)
        if _media_value_is_present(value):
            return True
    return False


def _request_has_top_level_media(request_inputs: Any) -> bool:
    if not isinstance(request_inputs, dict):
        return False
    if _first_present_qwen35_audio_input(request_inputs) is not None:
        return True
    for key in ("images", "image", "videos", "video"):
        if _media_value_is_present(request_inputs.get(key)):
            return True
    return False


def _request_dependent_audio(request_inputs: Any) -> Any | None:
    if not isinstance(request_inputs, dict):
        return None
    value = request_inputs.get("dependent_audio")
    if value is None:
        value = request_inputs.get("video_dependent_audio")
    return value


def _dependent_audio_requests_video_audio(request_inputs: Any) -> bool:
    dependent_audio = _request_dependent_audio(request_inputs)
    if dependent_audio is None:
        return False
    if hasattr(dependent_audio, "detach"):
        dependent_audio = dependent_audio.detach().cpu()
    if hasattr(dependent_audio, "tolist"):
        dependent_audio = dependent_audio.tolist()
    if isinstance(dependent_audio, bool):
        return bool(dependent_audio)
    if isinstance(dependent_audio, int):
        return int(dependent_audio) >= 0
    if isinstance(dependent_audio, str):
        try:
            return int(dependent_audio) >= 0
        except ValueError:
            parsed = _request_bool_string(dependent_audio)
            return parsed if parsed is not None else bool(dependent_audio)
    if isinstance(dependent_audio, (list, tuple, set)):
        values = list(dependent_audio)
        if not values:
            return False
        if all(isinstance(value, bool) for value in values):
            return any(bool(value) for value in values)
        for value in values:
            try:
                if int(value) >= 0:
                    return True
            except (TypeError, ValueError):
                parsed = _request_bool_string(value)
                if parsed is not None:
                    if parsed:
                        return True
                elif bool(value):
                    return True
        return False
    return _request_bool_value(dependent_audio)


def _use_audio_in_video_enabled(videos_kwargs: dict[str, Any]) -> bool:
    value = videos_kwargs.get("use_audio_in_video")
    if isinstance(value, (list, tuple)):
        return any(_request_bool_value(item) for item in value)
    return _request_bool_value(value)


def _dependent_audio_mask(dependent_audio: Any, audio_count: int) -> list[bool]:
    if audio_count <= 0:
        return []
    if hasattr(dependent_audio, "detach"):
        dependent_audio = dependent_audio.detach().cpu()
    if hasattr(dependent_audio, "tolist"):
        dependent_audio = dependent_audio.tolist()
    if isinstance(dependent_audio, bool):
        return [bool(dependent_audio)] + [False] * (audio_count - 1)
    if isinstance(dependent_audio, int):
        values = [dependent_audio]
    elif isinstance(dependent_audio, (list, tuple)):
        values = list(dependent_audio)
    else:
        try:
            values = list(dependent_audio)
        except TypeError:
            values = [dependent_audio]

    if values and all(
        isinstance(value, bool) or _request_bool_word_string(value) is not None
        for value in values
    ):
        mask = [_request_bool_value(value) for value in values[:audio_count]]
        mask.extend(False for _ in range(audio_count - len(mask)))
        return mask

    mask = [False] * audio_count
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < audio_count:
            mask[index] = True
    return mask


def _copy_audio_override(
    audio_kwargs: dict[str, Any],
    request_inputs: dict[str, Any],
    *,
    target: str,
    aliases: tuple[str, ...],
) -> None:
    for name in aliases:
        value = request_inputs.get(name)
        if value is not None:
            audio_kwargs[target] = int(value)
            return


def _copy_optional_value(
    target_dict: dict[str, Any],
    request_inputs: dict[str, Any],
    *,
    target: str,
    aliases: tuple[str, ...],
) -> None:
    for name in aliases:
        value = request_inputs.get(name)
        if value is not None:
            target_dict[target] = value
            return
