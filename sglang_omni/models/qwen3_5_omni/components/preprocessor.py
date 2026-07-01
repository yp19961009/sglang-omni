# SPDX-License-Identifier: Apache-2.0
"""Model-specific preprocessor for Qwen3.5-Omni."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import logging
import math
import os
import re
from typing import Any

import numpy as np

from sglang_omni.models.qwen3_5_omni.components.audio_encoder import (
    DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
    DEFAULT_DOWNSAMPLE_TIMES,
)
from sglang_omni.models.qwen3_5_omni.components.common import (
    load_qwen35_thinker_config,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event

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
_PROCESSOR_ITEM_CACHE_KEYS = "_sglang_omni_item_cache_keys"
_PROCESSOR_PROFILE_REQUEST_ID = "_sglang_omni_profile_request_id"
_PROCESSOR_TRACE_CACHE_SUMMARY = "_sglang_omni_trace_cache_summary"
_PROCESSOR_ITEM_CACHE_MAX_ENTRIES = 512
_PLAIN_TEXT_TOKEN_CACHE_MAX_ENTRIES = 4096
_OMIT_CACHED_VISUAL_ITEM_PAYLOADS_ENV = (
    "SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS"
)
_VIDEO_PROCESSOR_CACHE_CLONE_ON_HIT_ENV = (
    "SGLANG_OMNI_VIDEO_PROCESSOR_CACHE_CLONE_ON_HIT"
)
_TRACE_PROCESSOR_CACHE_ENV = "SGLANG_OMNI_TRACE_PROCESSOR_CACHE"
_TRACE_PROCESSOR_CACHE_DETAIL_ENV = "SGLANG_OMNI_TRACE_PROCESSOR_CACHE_DETAIL"
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


def _trace_processor_cache_enabled() -> bool:
    raw = os.getenv(_TRACE_PROCESSOR_CACHE_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _trace_processor_cache_detail_enabled() -> bool:
    if not _trace_processor_cache_enabled():
        return False
    raw = os.getenv(_TRACE_PROCESSOR_CACHE_DETAIL_ENV)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _short_processor_cache_key(cache_key: Any) -> str:
    if cache_key is None:
        return "-"
    text = repr(cache_key)
    if len(text) <= 80:
        return text
    return f"{text[:48]}...{text[-16:]}"


def _trace_processor_cache(
    modality: str | None,
    action: str,
    *,
    cache_key: Any,
    index: int | None = None,
    cache_entries: int | None = None,
    detail: str | None = None,
) -> None:
    if not _trace_processor_cache_detail_enabled():
        return
    logger.info(
        "qwen35_processor_cache modality=%s action=%s index=%s key=%s "
        "cache_entries=%s detail=%s",
        modality or "-",
        action,
        index if index is not None else "-",
        _short_processor_cache_key(cache_key),
        cache_entries if cache_entries is not None else "-",
        detail or "-",
    )


def _trace_processor_cache_summary(
    modality: str,
    *,
    item_count: int,
    hit_count: int,
    miss_count: int,
    store_count: int,
    no_key_count: int,
    cache_entries: int,
    detail: str | None = None,
) -> None:
    if not _trace_processor_cache_enabled():
        return
    keyed_count = item_count - no_key_count
    hit_rate = (hit_count / keyed_count) if keyed_count > 0 else 0.0
    logger.info(
        "qwen35_processor_cache_summary modality=%s items=%s keyed=%s "
        "hits=%s misses=%s stores=%s no_key=%s hit_rate=%.4f "
        "cache_entries=%s detail=%s",
        modality,
        item_count,
        keyed_count,
        hit_count,
        miss_count,
        store_count,
        no_key_count,
        hit_rate,
        cache_entries,
        detail or "-",
    )


def _freeze_processor_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, _freeze_processor_cache_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, list):
        return tuple(_freeze_processor_cache_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_processor_cache_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required in serving.
        return False
    return isinstance(value, torch.Tensor)


def _first_dim_len(value: Any) -> int | None:
    if _is_torch_tensor(value) or isinstance(value, np.ndarray):
        return int(value.shape[0]) if getattr(value, "ndim", 0) > 0 else None
    return None


def _slice_first_dim(value: Any, start: int, end: int) -> Any:
    if _is_torch_tensor(value) or isinstance(value, np.ndarray):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    return value


def _concat_first_dim(values: list[Any]) -> Any:
    if not values:
        return None
    if all(_is_torch_tensor(value) for value in values):
        import torch

        return torch.cat(values, dim=0)
    if all(isinstance(value, np.ndarray) for value in values):
        return np.concatenate(values, axis=0)
    if all(isinstance(value, list) for value in values):
        merged: list[Any] = []
        for value in values:
            merged.extend(value)
        return merged
    return values[0]


def _pad_last_dim(value: Any, target_size: int, *, pad_value: int = 0) -> Any:
    current = int(value.shape[-1])
    if current >= target_size:
        return value
    pad_width = target_size - current
    if _is_torch_tensor(value):
        import torch.nn.functional as F

        return F.pad(value, (0, pad_width), value=pad_value)
    if isinstance(value, np.ndarray):
        widths = [(0, 0)] * value.ndim
        widths[-1] = (0, pad_width)
        return np.pad(value, widths, constant_values=pad_value)
    return value


def _combine_entry_dicts(
    entries: list[dict[str, Any]],
    *,
    pad_last_dim: bool = False,
) -> dict[str, Any]:
    if not entries:
        return {}
    combined: dict[str, Any] = {}
    keys: list[str] = []
    for entry in entries:
        for key in entry:
            if key not in keys:
                keys.append(key)
    for key in keys:
        values = [entry[key] for entry in entries if key in entry]
        if pad_last_dim and values and all(
            (_is_torch_tensor(value) or isinstance(value, np.ndarray))
            and getattr(value, "ndim", 0) > 0
            for value in values
        ):
            target_size = max(int(value.shape[-1]) for value in values)
            values = [_pad_last_dim(value, target_size) for value in values]
        combined[key] = _concat_first_dim(values)
    return combined


def _prod_value(value: Any) -> int:
    if _is_torch_tensor(value):
        return int(value.prod().item())
    if isinstance(value, np.ndarray):
        return int(value.prod().item())
    product = 1
    for item in value:
        product *= int(item)
    return product


def _metadata_at(video_metadata: Any, index: int) -> Any:
    if isinstance(video_metadata, list) and index < len(video_metadata):
        return video_metadata[index]
    return None


def _omit_cached_visual_item_payloads_enabled() -> bool:
    value = os.getenv(_OMIT_CACHED_VISUAL_ITEM_PAYLOADS_ENV, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _video_processor_cache_clone_on_hit_enabled() -> bool:
    value = os.getenv(_VIDEO_PROCESSOR_CACHE_CLONE_ON_HIT_ENV)
    if value is None or value == "":
        return True
    return value.lower() not in ("0", "false", "no", "off")


def _cached_video_pixel_fallbacks_enabled() -> bool:
    value = os.getenv("SGLANG_OMNI_CACHED_VIDEO_PIXEL_FALLBACKS")
    if value is None or value == "":
        return False
    return value.lower() not in ("0", "false", "no", "off")


def _video_entry_with_empty_pixels(
    entry: tuple[dict[str, Any], Any],
) -> tuple[dict[str, Any], Any]:
    video_inputs, metadata = entry
    pixels = video_inputs.get("pixel_values_videos")
    if pixels is None:
        return entry
    trimmed = dict(video_inputs)
    trimmed["pixel_values_videos"] = _slice_first_dim(pixels, 0, 0)
    return trimmed, metadata


def _clone_processor_cache_value(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_processor_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_processor_cache_value(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        pass
    return value


def _emit_processor_profile_event(
    request_id: str | None,
    event_name: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if request_id:
        _emit_event(
            request_id=request_id,
            stage=None,
            event_name=f"qwen35_processor_{event_name}",
            metadata=metadata,
        )


class _Qwen35ProcessorShim:
    """Local Qwen3.5-Omni processor fallback.

    Some deployed transformers builds do not ship qwen3_omni_next yet. The
    fallback must still process audio; otherwise the audio placeholder is
    tokenized as ordinary text and the thinker never receives audio embeddings.
    """

    supports_item_cache_keys = True

    def __init__(
        self,
        *,
        tokenizer,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        chat_template=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.feature_extractor = feature_extractor
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)
        self.image_token = tokenizer.image_token
        self.audio_token = tokenizer.audio_token
        self.video_token = tokenizer.video_token
        self.vision_bos_token = tokenizer.vision_bos_token
        self.vision_eos_token = tokenizer.vision_eos_token
        self.audio_bos_token = tokenizer.audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token
        self.video_token_block = (
            self.vision_bos_token + self.video_token + self.vision_eos_token
        )
        self.mm_token_pattern = re.compile(
            "|".join(
                re.escape(tok)
                for tok in (
                    self.video_token_block,
                    self.audio_token,
                    self.image_token,
                    self.video_token,
                )
                if tok
            )
        )
        self._tokenizer_special_ids: dict[str, int] = {}
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            special_tokens = [
                *getattr(tokenizer, "all_special_tokens", ()),
                self.audio_token,
                self.image_token,
                self.video_token,
                self.vision_bos_token,
                self.vision_eos_token,
                self.audio_bos_token,
                self.audio_eos_token,
            ]
            for token in special_tokens:
                if not token or token in self._tokenizer_special_ids:
                    continue
                token_id = convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    self._tokenizer_special_ids[token] = token_id
        self._tokenizer_special_pattern = (
            re.compile(
                "|".join(
                    re.escape(token)
                    for token in sorted(
                        self._tokenizer_special_ids, key=len, reverse=True
                    )
                )
            )
            if self._tokenizer_special_ids
            else None
        )
        self._audio_item_processor_cache: OrderedDict[Any, dict[str, Any]] = (
            OrderedDict()
        )
        self._video_item_processor_cache: OrderedDict[
            Any, tuple[dict[str, Any], Any]
        ] = OrderedDict()
        self._plain_text_token_cache: OrderedDict[str, tuple[int, ...]] = (
            OrderedDict()
        )

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        from transformers import (
            AutoTokenizer,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
            WhisperFeatureExtractor,
        )

        kw = {
            k: v
            for k, v in kwargs.items()
            if k in ("trust_remote_code", "local_files_only")
        }
        tokenizer = AutoTokenizer.from_pretrained(model_dir, **kw)
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_dir, **kw)
        try:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_dir,
                **kw,
            )
        except (OSError, ValueError):
            feature_extractor = WhisperFeatureExtractor()
        video_processor = Qwen2VLVideoProcessor.from_pretrained(model_dir, **kw)
        return cls(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            chat_template=tokenizer.chat_template,
        )

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(
        self,
        *,
        text=None,
        images=None,
        videos=None,
        audio=None,
        return_tensors=None,
        **kwargs,
    ):
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        profile_request_id = kwargs.pop(_PROCESSOR_PROFILE_REQUEST_ID, None)
        profile_metadata = {
            "num_text": len(text) if isinstance(text, list) else int(text is not None),
            "num_images": len(images) if isinstance(images, list) else int(images is not None),
            "num_videos": len(videos) if isinstance(videos, list) else int(videos is not None),
            "num_audios": len(audio) if isinstance(audio, list) else int(audio is not None),
        }
        _emit_processor_profile_event(
            profile_request_id, "call_start", profile_metadata
        )
        output_kwargs = self._merge_processor_kwargs(kwargs, videos=videos)
        try:
            _emit_processor_profile_event(
                profile_request_id, "audio_start", profile_metadata
            )
            audio_inputs, audio_lengths = self._process_audio(
                audio,
                output_kwargs["audio_kwargs"],
                downsample_times=output_kwargs["downsample_times"],
                downsample_chunk_size=output_kwargs["downsample_chunk_size"],
            )
            _emit_processor_profile_event(
                profile_request_id, "audio_end", profile_metadata
            )

            _emit_processor_profile_event(
                profile_request_id, "image_start", profile_metadata
            )
            image_inputs, image_grid_thw = self._process_images(
                images,
                output_kwargs["images_kwargs"],
            )
            _emit_processor_profile_event(
                profile_request_id, "image_end", profile_metadata
            )

            _emit_processor_profile_event(
                profile_request_id, "video_start", profile_metadata
            )
            video_inputs, video_grid_thw, video_metadata = self._process_videos(
                videos,
                output_kwargs["videos_kwargs"],
            )
            _emit_processor_profile_event(
                profile_request_id, "video_end", profile_metadata
            )

            if not isinstance(text, list):
                text = [text]
            _emit_processor_profile_event(
                profile_request_id, "replace_tokens_start", profile_metadata
            )
            text = self.replace_multimodal_special_tokens(
                text,
                iter(audio_lengths),
                iter(image_grid_thw),
                iter(video_grid_thw),
                video_metadata=iter(video_metadata),
                audio_tokens_per_second=math.ceil(
                    self.feature_extractor.sampling_rate
                    / self.feature_extractor.hop_length
                    / 2**output_kwargs["downsample_times"]
                ),
                audio_timestamp_interval=output_kwargs["audio_timestamp_interval"],
                use_audio_in_video=iter(video_inputs.pop("use_audio_in_video", [])),
            )
            video_pixel_fallbacks = video_inputs.pop(
                "video_item_pixel_fallbacks",
                None,
            )
            _emit_processor_profile_event(
                profile_request_id, "replace_tokens_end", profile_metadata
            )

            _emit_processor_profile_event(
                profile_request_id, "tokenize_start", profile_metadata
            )
            text_inputs = self._fast_tokenize_with_special_ids(
                text,
                output_kwargs["text_kwargs"],
            )
            tokenize_metadata = dict(profile_metadata)
            tokenize_metadata["fast_special_token_path"] = text_inputs is not None
            if text_inputs is None:
                text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            _emit_processor_profile_event(
                profile_request_id, "tokenize_end", tokenize_metadata
            )
            from transformers.feature_extraction_utils import BatchFeature

            batch = BatchFeature(
                data={**text_inputs, **image_inputs, **video_inputs, **audio_inputs},
                tensor_type=return_tensors,
            )
            if video_pixel_fallbacks is not None:
                batch["video_item_pixel_fallbacks"] = video_pixel_fallbacks
            return batch
        finally:
            _emit_processor_profile_event(
                profile_request_id, "call_end", profile_metadata
            )

    def _fast_tokenize_with_special_ids(
        self,
        text: list[str],
        text_kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._tokenizer_special_pattern is None:
            return None
        if not isinstance(text, list) or len(text) != 1:
            return None
        if text_kwargs.get("add_special_tokens", False):
            return None
        if text_kwargs.get("padding") not in (None, False):
            return None
        unsupported = set(text_kwargs) - {
            "add_special_tokens",
            "padding",
            "padding_side",
            "return_attention_mask",
            "truncation",
        }
        if unsupported:
            return None
        if text_kwargs.get("truncation") not in (None, False):
            return None

        token_ids: list[int] = []
        sample = text[0]
        cursor = 0
        while True:
            match = self._tokenizer_special_pattern.search(sample, cursor)
            if match is None:
                break
            if match.start() > cursor:
                plain_ids = self._tokenize_plain_text_ids(sample[cursor : match.start()])
                if plain_ids is None:
                    return None
                token_ids.extend(plain_ids)
            special_token = match.group(0)
            token_id = self._tokenizer_special_ids.get(special_token)
            if token_id is None:
                return None
            run_count, run_end = self._special_token_run(
                sample,
                start=match.start(),
                token=special_token,
            )
            token_ids.extend([token_id] * run_count)
            cursor = run_end
        if cursor < len(sample):
            plain_ids = self._tokenize_plain_text_ids(sample[cursor:])
            if plain_ids is None:
                return None
            token_ids.extend(plain_ids)
        return {
            "input_ids": [token_ids],
            "attention_mask": [[1] * len(token_ids)],
        }

    @staticmethod
    def _special_token_run(
        sample: str,
        *,
        start: int,
        token: str,
    ) -> tuple[int, int]:
        token_len = len(token)
        if token_len <= 0:
            return 1, start
        cursor = start
        count = 0
        sample_len = len(sample)
        while cursor + token_len <= sample_len and sample.startswith(token, cursor):
            count += 1
            cursor += token_len
        return max(count, 1), cursor

    def _tokenize_plain_text_ids(self, text: str) -> list[int] | None:
        if not text:
            return []
        cached = self._plain_text_token_cache.get(text)
        if cached is not None:
            self._plain_text_token_cache.move_to_end(text)
            return list(cached)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
        )
        try:
            input_ids = encoded["input_ids"]
        except (KeyError, TypeError):
            return None
        if _is_torch_tensor(input_ids):
            input_ids = input_ids.reshape(-1).tolist()
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                return None
            input_ids = input_ids[0]
        if not isinstance(input_ids, list):
            return None
        try:
            token_ids = tuple(int(token_id) for token_id in input_ids)
        except (TypeError, ValueError):
            return None
        self._plain_text_token_cache[text] = token_ids
        self._plain_text_token_cache.move_to_end(text)
        while len(self._plain_text_token_cache) > _PLAIN_TEXT_TOKEN_CACHE_MAX_ENTRIES:
            self._plain_text_token_cache.popitem(last=False)
        return list(token_ids)

    def _merge_processor_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        videos: Any,
    ) -> dict[str, Any]:
        text_kwargs = {"padding": False, "padding_side": "left"}
        text_kwargs.update(kwargs.get("text_kwargs") or {})
        if "add_special_tokens" in kwargs:
            text_kwargs["add_special_tokens"] = kwargs["add_special_tokens"]

        audio_kwargs = {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
            "truncation": False,
            "timestamp_interval": DEFAULT_AUDIO_TIMESTAMP_INTERVAL,
            "downsample_times": DEFAULT_DOWNSAMPLE_TIMES,
            "downsample_chunk_size": DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
        }
        audio_kwargs.update(kwargs.get("audio_kwargs") or {})
        downsample_times = int(audio_kwargs.pop("downsample_times"))
        downsample_chunk_size = int(audio_kwargs.pop("downsample_chunk_size"))
        audio_timestamp_interval = int(audio_kwargs.pop("timestamp_interval"))

        video_kwargs = {"use_audio_in_video": False, "return_metadata": True}
        video_kwargs.update(kwargs.get("videos_kwargs") or {})
        video_kwargs = self._with_effective_video_resize_size(video_kwargs)
        use_audio_in_video = video_kwargs.get("use_audio_in_video", False)
        if not isinstance(use_audio_in_video, list):
            video_count = (
                len(videos) if isinstance(videos, list) else int(videos is not None)
            )
            use_audio_in_video = [use_audio_in_video] * video_count
        video_kwargs["use_audio_in_video"] = use_audio_in_video

        return {
            "text_kwargs": text_kwargs,
            "audio_kwargs": audio_kwargs,
            "images_kwargs": dict(kwargs.get("images_kwargs") or {}),
            "videos_kwargs": video_kwargs,
            "downsample_times": downsample_times,
            "downsample_chunk_size": downsample_chunk_size,
            "audio_timestamp_interval": audio_timestamp_interval,
        }

    def _with_effective_video_resize_size(
        self,
        video_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if "size" in video_kwargs:
            return video_kwargs
        min_pixels = video_kwargs.get("min_pixels")
        max_pixels = video_kwargs.get("max_pixels")
        if min_pixels is None and max_pixels is None:
            return video_kwargs
        if isinstance(min_pixels, (list, tuple)) or isinstance(max_pixels, (list, tuple)):
            return video_kwargs

        default_size = getattr(self.video_processor, "size", None) or {}
        shortest_edge = (
            int(min_pixels)
            if min_pixels is not None
            else default_size.get("shortest_edge")
        )
        longest_edge = (
            int(max_pixels)
            if max_pixels is not None
            else default_size.get("longest_edge")
        )
        if shortest_edge is None or longest_edge is None:
            return video_kwargs

        updated = dict(video_kwargs)
        # The current transformers Qwen2VLVideoProcessor ignores min_pixels and
        # max_pixels during _preprocess, but does honor size.shortest/longest.
        updated["size"] = {
            "shortest_edge": int(shortest_edge),
            "longest_edge": int(longest_edge),
        }
        return updated

    def _process_audio(
        self,
        audio,
        audio_kwargs: dict[str, Any],
        *,
        downsample_times: int,
        downsample_chunk_size: int,
    ):
        if audio is None:
            return {}, []
        if not isinstance(audio, list):
            audio = [audio]
        item_cache_keys = audio_kwargs.pop(_PROCESSOR_ITEM_CACHE_KEYS, None)
        trace_cache_summary = bool(
            audio_kwargs.pop(_PROCESSOR_TRACE_CACHE_SUMMARY, False)
        )
        sampling_rate = int(audio_kwargs.get("sampling_rate", 16000))
        padded_audio = self._pad_audio_for_feature_extractor(audio, sampling_rate)
        if self._can_use_item_cache(item_cache_keys, len(padded_audio)):
            audio_inputs = self._process_audio_with_item_cache(
                padded_audio,
                audio_kwargs,
                item_cache_keys,
                trace_cache_summary=trace_cache_summary,
                downsample_times=downsample_times,
                downsample_chunk_size=downsample_chunk_size,
            )
        else:
            audio_inputs = self._extract_audio_features(padded_audio, audio_kwargs)

        feature_attention_mask = audio_inputs.get("feature_attention_mask")
        if feature_attention_mask is None:
            return audio_inputs, []
        audio_lengths = _qwen35_feat_extract_output_lengths(
            feature_attention_mask.sum(-1),
            downsample_times=downsample_times,
            chunk_size=downsample_chunk_size,
        )
        return audio_inputs, audio_lengths.tolist()

    def _pad_audio_for_feature_extractor(self, audio, sampling_rate: int):
        padded_audio = []
        for sample in audio:
            remainder = sample.shape[-1] % sampling_rate
            if remainder:
                sample = np.pad(sample, (0, sampling_rate - remainder))
            padded_audio.append(sample)
        return padded_audio

    def _extract_audio_features(self, padded_audio, audio_kwargs: dict[str, Any]):
        audio_inputs = self.feature_extractor(padded_audio, **audio_kwargs)
        if "attention_mask" in audio_inputs:
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )
        if "input_features" in audio_inputs:
            audio_inputs["input_audio_features"] = audio_inputs.pop(
                "input_features"
            )
        return audio_inputs

    def _process_audio_with_item_cache(
        self,
        padded_audio,
        audio_kwargs: dict[str, Any],
        item_cache_keys,
        *,
        trace_cache_summary: bool,
        downsample_times: int,
        downsample_chunk_size: int,
    ) -> dict[str, Any]:
        entries: list[dict[str, Any] | None] = [None] * len(padded_audio)
        miss_indices: list[int] = []
        miss_audio: list[Any] = []
        cache_keys_by_index: list[Any | None] = []
        hit_count = 0
        no_key_count = 0
        trace_detail = (
            trace_cache_summary and _trace_processor_cache_detail_enabled()
        )
        for index, sample in enumerate(padded_audio):
            del sample
            cache_key = self._processor_item_cache_key(
                "audio",
                item_cache_keys[index],
                audio_kwargs,
                extra=(downsample_times, downsample_chunk_size),
            )
            cache_keys_by_index.append(cache_key)
            cached = self._processor_cache_get(
                self._audio_item_processor_cache,
                cache_key,
                modality="audio",
                index=index,
                trace_detail=trace_detail,
            )
            if cached is not None:
                if trace_cache_summary:
                    hit_count += 1
                entries[index] = cached
                continue
            if trace_cache_summary and cache_key is None:
                no_key_count += 1
            miss_indices.append(index)
            miss_audio.append(padded_audio[index])

        store_count = 0
        if miss_audio:
            miss_inputs = self._extract_audio_features(miss_audio, audio_kwargs)
            miss_entries = self._split_batch_inputs(miss_inputs, len(miss_audio))
            for miss_pos, index in enumerate(miss_indices):
                entry = miss_entries[miss_pos]
                entries[index] = entry
                self._processor_cache_set(
                    self._audio_item_processor_cache,
                    cache_keys_by_index[index],
                    entry,
                    modality="audio",
                    index=index,
                    trace_detail=trace_detail,
                )
                if trace_cache_summary and cache_keys_by_index[index] is not None:
                    store_count += 1

        if trace_cache_summary:
            _trace_processor_cache_summary(
                "audio",
                item_count=len(padded_audio),
                hit_count=hit_count,
                miss_count=len(miss_indices) - no_key_count,
                store_count=store_count,
                no_key_count=no_key_count,
                cache_entries=len(self._audio_item_processor_cache),
            )

        return self._combine_audio_entries(
            [entry for entry in entries if entry is not None]
        )

    def _process_images(self, images, images_kwargs: dict[str, Any]):
        if images is None:
            return {}, []
        image_inputs = self.image_processor(images=images, **images_kwargs)
        return image_inputs, image_inputs.get("image_grid_thw", [])

    def _process_videos(self, videos, video_kwargs: dict[str, Any]):
        if videos is None:
            return {}, [], []
        videos = _wrap_video_frame_path_items(videos)
        if not isinstance(videos, list):
            videos = [videos]
        if video_kwargs.get("device") is None:
            video_kwargs["device"] = "cpu"
        item_cache_keys = video_kwargs.pop(_PROCESSOR_ITEM_CACHE_KEYS, None)
        trace_cache_summary = bool(
            video_kwargs.pop(_PROCESSOR_TRACE_CACHE_SUMMARY, False)
        )
        use_audio_in_video = video_kwargs.pop("use_audio_in_video", [])
        if self._can_use_item_cache(item_cache_keys, len(videos)):
            video_inputs, video_metadata = self._process_videos_with_item_cache(
                videos,
                video_kwargs,
                item_cache_keys,
                trace_cache_summary=trace_cache_summary,
            )
        else:
            video_inputs = self.video_processor(videos=videos, **video_kwargs)
            video_metadata = video_inputs.get("video_metadata", [])
            video_inputs.pop("video_metadata", None)
        video_inputs["use_audio_in_video"] = use_audio_in_video
        return video_inputs, video_inputs.get("video_grid_thw", []), video_metadata

    def _process_videos_with_item_cache(
        self,
        videos,
        video_kwargs: dict[str, Any],
        item_cache_keys,
        *,
        trace_cache_summary: bool,
    ) -> tuple[dict[str, Any], list[Any]]:
        entries: list[tuple[dict[str, Any], Any] | None] = [None] * len(videos)
        miss_indices: list[int] = []
        miss_videos: list[Any] = []
        cache_keys_by_index: list[Any | None] = []
        pixel_present: list[bool] = []
        pixel_fallbacks: list[Any | None] = [None] * len(videos)
        omit_cached_pixels = _omit_cached_visual_item_payloads_enabled()
        keep_pixel_fallbacks = omit_cached_pixels and _cached_video_pixel_fallbacks_enabled()
        hit_count = 0
        no_key_count = 0
        trace_detail = (
            trace_cache_summary and _trace_processor_cache_detail_enabled()
        )
        for index, video in enumerate(videos):
            per_item_kwargs = self._select_item_processor_kwargs(
                video_kwargs,
                index,
                len(videos),
            )
            cache_key = self._processor_item_cache_key(
                "video",
                item_cache_keys[index],
                per_item_kwargs,
            )
            cache_keys_by_index.append(cache_key)
            cached = self._processor_cache_get(
                self._video_item_processor_cache,
                cache_key,
                modality="video",
                index=index,
                clone=(
                    _video_processor_cache_clone_on_hit_enabled()
                    and not omit_cached_pixels
                ),
                trace_detail=trace_detail,
            )
            if cached is not None:
                if trace_cache_summary:
                    hit_count += 1
                if omit_cached_pixels:
                    pixel_present.append(False)
                    if keep_pixel_fallbacks:
                        pixel_fallbacks[index] = cached[0].get("pixel_values_videos")
                    entries[index] = _video_entry_with_empty_pixels(cached)
                else:
                    pixel_present.append(True)
                    entries[index] = cached
                continue
            if trace_cache_summary and cache_key is None:
                no_key_count += 1
            pixel_present.append(True)
            miss_indices.append(index)
            miss_videos.append(video)

        store_count = 0
        if miss_videos:
            miss_kwargs = self._select_item_processor_kwargs(
                video_kwargs,
                miss_indices,
                len(videos),
            )
            miss_inputs = self.video_processor(videos=miss_videos, **miss_kwargs)
            miss_metadata = miss_inputs.get("video_metadata", [])
            miss_inputs.pop("video_metadata", None)
            miss_entries = self._split_video_inputs(
                miss_inputs,
                miss_metadata,
                len(miss_videos),
            )
            for miss_pos, index in enumerate(miss_indices):
                entry = miss_entries[miss_pos]
                entries[index] = entry
                self._processor_cache_set(
                    self._video_item_processor_cache,
                    cache_keys_by_index[index],
                    entry,
                    modality="video",
                    index=index,
                    trace_detail=trace_detail,
                )
                if trace_cache_summary and cache_keys_by_index[index] is not None:
                    store_count += 1

        combined, metadata = self._combine_video_entries(
            [entry for entry in entries if entry is not None]
        )
        if pixel_present and not all(pixel_present):
            combined["video_item_pixel_present"] = pixel_present
            if any(item is not None for item in pixel_fallbacks):
                combined["video_item_pixel_fallbacks"] = pixel_fallbacks
        if trace_cache_summary:
            _trace_processor_cache_summary(
                "video",
                item_count=len(videos),
                hit_count=hit_count,
                miss_count=len(miss_indices) - no_key_count,
                store_count=store_count,
                no_key_count=no_key_count,
                cache_entries=len(self._video_item_processor_cache),
                detail=(
                    f"omit_cached_pixels={int(omit_cached_pixels)} "
                    f"pixel_fallbacks={int(keep_pixel_fallbacks)}"
                ),
            )
        return combined, metadata

    def _can_use_item_cache(self, item_cache_keys, item_count: int) -> bool:
        return (
            isinstance(item_cache_keys, list)
            and len(item_cache_keys) == item_count
            and any(item_cache_keys)
        )

    def _processor_cache_get(
        self,
        cache: OrderedDict,
        cache_key: Any,
        *,
        modality: str | None = None,
        index: int | None = None,
        clone: bool = True,
        trace_detail: bool = False,
    ):
        if cache_key is None:
            if trace_detail:
                _trace_processor_cache(
                    modality, "skip_get_no_key", cache_key=cache_key, index=index
                )
            return None
        try:
            value = cache.pop(cache_key)
        except KeyError:
            if trace_detail:
                _trace_processor_cache(
                    modality,
                    "miss",
                    cache_key=cache_key,
                    index=index,
                    cache_entries=len(cache),
                )
            return None
        cache[cache_key] = value
        if trace_detail:
            _trace_processor_cache(
                modality,
                "hit",
                cache_key=cache_key,
                index=index,
                cache_entries=len(cache),
            )
        if clone:
            return _clone_processor_cache_value(value)
        return value

    def _processor_cache_set(
        self,
        cache: OrderedDict,
        cache_key: Any,
        value: Any,
        *,
        modality: str | None = None,
        index: int | None = None,
        trace_detail: bool = False,
    ) -> None:
        if cache_key is None:
            if trace_detail:
                _trace_processor_cache(
                    modality, "skip_store_no_key", cache_key=cache_key, index=index
                )
            return
        cache[cache_key] = _clone_processor_cache_value(value)
        cache.move_to_end(cache_key)
        if trace_detail:
            _trace_processor_cache(
                modality,
                "store",
                cache_key=cache_key,
                index=index,
                cache_entries=len(cache),
            )
        while len(cache) > _PROCESSOR_ITEM_CACHE_MAX_ENTRIES:
            evicted_key, _ = cache.popitem(last=False)
            if trace_detail:
                _trace_processor_cache(
                    modality,
                    "evict_entries",
                    cache_key=evicted_key,
                    index=index,
                    cache_entries=len(cache),
                    detail=f"max_entries={_PROCESSOR_ITEM_CACHE_MAX_ENTRIES}",
                )

    def _processor_item_cache_key(
        self,
        modality: str,
        item_cache_key: Any,
        processor_kwargs: dict[str, Any],
        *,
        extra: tuple[Any, ...] = (),
    ):
        if item_cache_key is None:
            return None
        return (
            modality,
            item_cache_key,
            _freeze_processor_cache_value(processor_kwargs),
            _freeze_processor_cache_value(extra),
        )

    def _select_item_processor_kwargs(
        self,
        kwargs: dict[str, Any],
        indices,
        item_count: int,
    ) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        index_list = indices if isinstance(indices, list) else [indices]
        for key, value in kwargs.items():
            if isinstance(value, list) and len(value) == item_count:
                values = [value[index] for index in index_list]
                selected[key] = values[0] if len(values) == 1 else values
            else:
                selected[key] = value
        return selected

    def _split_batch_inputs(
        self,
        batch_inputs: dict[str, Any],
        item_count: int,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = [dict() for _ in range(item_count)]
        for key, value in batch_inputs.items():
            for index in range(item_count):
                entries[index][key] = _slice_first_dim(value, index, index + 1)
        return entries

    def _split_video_inputs(
        self,
        video_inputs: dict[str, Any],
        video_metadata,
        item_count: int,
    ) -> list[tuple[dict[str, Any], Any]]:
        grid_thw = video_inputs.get("video_grid_thw")
        if grid_thw is None:
            return [
                (entry, _metadata_at(video_metadata, index))
                for index, entry in enumerate(
                    self._split_batch_inputs(video_inputs, item_count)
                )
            ]
        patch_lengths = [int(_prod_value(row)) for row in grid_thw]
        pixel_offset = 0
        entries: list[tuple[dict[str, Any], Any]] = []
        for index, patch_length in enumerate(patch_lengths):
            entry: dict[str, Any] = {}
            for key, value in video_inputs.items():
                if key == "pixel_values_videos":
                    entry[key] = _slice_first_dim(
                        value,
                        pixel_offset,
                        pixel_offset + patch_length,
                    )
                elif key == "video_grid_thw":
                    entry[key] = _slice_first_dim(value, index, index + 1)
                elif _first_dim_len(value) == item_count:
                    entry[key] = _slice_first_dim(value, index, index + 1)
                elif isinstance(value, list) and len(value) == item_count:
                    entry[key] = [value[index]]
                else:
                    entry[key] = value
            entries.append((entry, _metadata_at(video_metadata, index)))
            pixel_offset += patch_length
        return entries

    def _combine_video_entries(
        self,
        entries: list[tuple[dict[str, Any], Any]],
    ) -> tuple[dict[str, Any], list[Any]]:
        return (
            _combine_entry_dicts([entry for entry, _metadata in entries]),
            [metadata for _entry, metadata in entries],
        )

    def _combine_audio_entries(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        return _combine_entry_dicts(entries, pad_last_dim=True)

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        *,
        video_metadata,
        audio_tokens_per_second,
        audio_timestamp_interval,
        use_audio_in_video,
    ):
        merge_length_image = self.image_processor.merge_size**2
        processed_text = []
        for sample in text:
            def _replace_special_token(match: re.Match[str]) -> str:
                special_token = match.group(0)
                if special_token == self.audio_token:
                    return self._get_audio_tokens(
                        next(audio_lengths),
                        audio_tokens_per_second,
                        audio_timestamp_interval,
                    )
                if special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    return self.image_token * int(image_seq_length)
                if special_token in (self.video_token_block, self.video_token):
                    metadata = next(video_metadata)
                    use_audio = next(use_audio_in_video)
                    metadata_fps = getattr(metadata, "fps", None) or 24
                    return self._get_video_tokens(
                        metadata.frames_indices,
                        metadata_fps,
                        next(video_grid_thw),
                        self.image_processor.merge_size,
                        audio_tokens_per_second if use_audio else None,
                        next(audio_lengths) if use_audio else None,
                    )
                return special_token

            sample = self.mm_token_pattern.sub(_replace_special_token, sample)
            if "_placeholder|>" in sample:
                sample = sample.replace("<|audio_placeholder|>", self.audio_token)
                sample = sample.replace("<|image_placeholder|>", self.image_token)
                sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def _get_audio_tokens(
        self,
        audio_length: int,
        tokens_per_second: int,
        timestamp_interval: int,
    ):
        tokens_interval = tokens_per_second * timestamp_interval
        num_full_chunks = math.floor(audio_length / tokens_interval)
        num_residual_tokens = audio_length % tokens_interval
        audio_placeholders = ""
        for i in range(num_full_chunks):
            audio_placeholders += f"<{i * timestamp_interval:.1f} seconds>"
            audio_placeholders += self.audio_token * tokens_interval
        if num_residual_tokens > 0:
            audio_placeholders += (
                f"<{num_full_chunks * timestamp_interval:.1f} seconds>"
            )
            audio_placeholders += self.audio_token * int(num_residual_tokens)
        return audio_placeholders

    def _get_video_tokens(
        self,
        indices,
        video_fps: float,
        video_grid_thw,
        merge_size: int = 2,
        audio_tokens_per_second: int | None = None,
        audio_length: int | None = None,
    ):
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        else:
            indices = list(indices)
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]

        video_placeholders = []
        merge_length = merge_size**2
        frame_seqlen = video_grid_thw[1:].prod() // merge_length
        for frame_idx in range(video_grid_thw[0]):
            video_placeholders.append(
                f"<{timestamps[frame_idx]:.1f} seconds>"
                + self.vision_bos_token
                + self.video_token * int(frame_seqlen)
                + self.vision_eos_token
            )

        if audio_tokens_per_second:
            assert audio_length is not None
            with_audio = [self.audio_bos_token]
            video_second_per_chunk = indices[-1] / video_fps / video_grid_thw[0]
            audio_tokens = 0
            for i, video_chunk in enumerate(video_placeholders):
                if i == len(video_placeholders) - 1:
                    break
                audio_token = math.floor(
                    (i + 1) * video_second_per_chunk * audio_tokens_per_second
                ) - math.floor(i * video_second_per_chunk * audio_tokens_per_second)
                audio_token = min(audio_token, audio_length - audio_tokens)
                with_audio.append(video_chunk + self.audio_token * audio_token)
                audio_tokens += audio_token
            with_audio.append(
                video_placeholders[-1]
                + self.audio_token * (audio_length - audio_tokens)
            )
            with_audio.append(self.audio_eos_token)
            video_placeholders = with_audio
        return "".join(video_placeholders)

    @property
    def model_input_names(self):
        names = []
        for component in (
            self.tokenizer,
            self.feature_extractor,
            self.image_processor,
        ):
            for name in getattr(component, "model_input_names", []):
                if name not in names:
                    names.append(name)
        for name in ("feature_attention_mask", "video_second_per_grid"):
            if name not in names:
                names.append(name)
        return names


def _qwen35_feat_extract_output_lengths(
    input_lengths,
    *,
    downsample_times: int = DEFAULT_DOWNSAMPLE_TIMES,
    chunk_size: int = DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
):
    input_lengths_leave = input_lengths % chunk_size
    for _ in range(downsample_times):
        input_lengths_leave = (input_lengths_leave - 1) // 2 + 1
    return input_lengths_leave + (input_lengths // chunk_size) * math.ceil(
        100 / 2**downsample_times
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
        normalize_metadata: dict[str, Any] = {}
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="preprocess_normalize_start",
            metadata=normalize_metadata,
        )
        try:
            request_inputs = _merge_request_params_into_inputs(
                payload.request.inputs,
                payload.request.params,
            )
            normalized_inputs = _normalize_openai_multimodal_inputs(request_inputs)
            normalized_inputs = _normalize_request_level_media_aliases(normalized_inputs)
            self._validate_limit_mm_per_prompt(
                normalized_inputs,
                request_id=payload.request_id,
            )
            normalize_metadata["changed"] = normalized_inputs is not request_inputs
            normalize_metadata["input_type"] = type(normalized_inputs).__name__
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="preprocess_normalize_end",
                metadata=normalize_metadata,
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
                # Qwen3OmniNextProcessor pops these three keys directly. Keep
                # the Qwen reference defaults even if the model config is
                # missing them, so real requests do not fail later.
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
                # dependent_audio is a request-level semantic flag, not an HF
                # video_processor argument. SGLang expands tokens directly in
                # the CPU preprocessor, so treat it only as a signal that video
                # uses an embedded audio track and do not pass it to the HF
                # processor.
                video_kwargs.setdefault("use_audio_in_video", True)
            video_kwargs.setdefault("return_metadata", True)
            processor_kwargs["videos_kwargs"] = video_kwargs

        audio_kwargs = {
            "sampling_rate": int(audio_target_sr),
            # Pin Qwen3.5 audio processor defaults explicitly so remote
            # processor version differences do not change merge behavior,
            # attention masks, or truncation.
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

        # Qwen3OmniNextProcessor uses these parameters when expanding audio
        # tokens. Keep the processor and audio encoder downsample contract in
        # sync so placeholder counts match encoder output lengths under real
        # weights.
        processor_kwargs["audio_kwargs"] = audio_kwargs
        return processor_kwargs

    def _resolve_use_audio_in_video(
        self,
        request_inputs: dict[str, Any],
        raw_videos: Any,
    ) -> Any:
        # Qwen3OmniNextProcessor defaults use_audio_in_video to False. Preserve
        # the "extract video audio only when explicitly requested" semantics;
        # dependent_audio is also an explicit Qwen3.5 reference entry point and
        # can enable the same behavior.
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

def _first_config_value(config: Any, aliases: tuple[str, ...]) -> Any | None:
    for name in aliases:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _is_frame_path_video_item(item: Any) -> bool:
    return isinstance(item, list) and bool(item) and all(isinstance(x, str) for x in item)


def _wrap_video_frame_path_items(videos: Any) -> Any:
    """Preserve ``[frame.jpg, ...]`` as one video item for HF processing.

    The transformers version in this image recursively flattens a two-level
    ``[[frame1, frame2]]`` input into independent video paths.  A three-level
    ``[[[frame1, frame2]]]`` input keeps the frame list together and lets the
    video processor treat it as a single decoded video.
    """

    if not isinstance(videos, list):
        return videos
    return [[item] if _is_frame_path_video_item(item) else item for item in videos]


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
    # These are service-level defaults; request-level audio_kwargs/params still
    # override them in _processor_kwargs_for_request.
    if timestamp_interval is not None:
        defaults["timestamp_interval"] = int(timestamp_interval)
    if downsample_times is not None:
        defaults["downsample_times"] = int(downsample_times)
    if downsample_chunk_size is not None:
        defaults["downsample_chunk_size"] = int(downsample_chunk_size)


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
        # The Qwen3.5 OpenAI reference server exposes use_audio_in_video as a
        # top-level request field, which may land in params after entering
        # sglang serve. Promote only whitelisted multimodal preprocessing
        # parameters so sampling params are not polluted.
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
    # OpenAI Responses-style content parts often use input_text. Qwen3.5 only
    # needs the final text content, so fold it into a normal text part.
    return part.get("input_text")


def _append_openai_text_part(content_parts: list[str], text: str) -> None:
    if not text:
        return
    if content_parts and content_parts[-1] not in _OPENAI_MEDIA_PLACEHOLDER_TO_TYPE:
        # Consecutive OpenAI text parts are equivalent to one plain text
        # message. Preserve newlines so independent text parts are not glued
        # together.
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
        # HF Qwen3OmniNextProcessor chat templates recognize
        # {"type": "video/audio/image"}, which preserves the original OpenAI
        # content-part order.
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
        # The Qwen3 base class only reads images/videos/audios. Normalize common
        # Qwen3.5 input_* and *_url forms into plural fields so direct clients or
        # custom benchmark scripts do not drop inputs.
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

    # OpenAI content parts are the most precise source of ordering. If a request
    # also includes top-level audio without a placeholder, append it at the end
    # as a fallback instead of silently dropping it.
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

    # Plain top-level videos/audios follow the parent class template injection
    # order: video before audio. Keep the HF processor audio list in the same
    # order, with video tracks before standalone audio.
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
        # Support lightweight demos and benchmark scripts that use forms like
        # {"type": "audio", "url": ...} or {"type": "input_video", "data": ...}.
        # Read top-level url/path/data only when type explicitly matches the
        # current media kind, so unknown content parts are not mistaken for
        # media.
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
    # Qwen3OmniNextProcessor supports use_audio_in_video as either a bool or a
    # per-video bool list. OpenAI content parts configure videos individually,
    # so keep an internal list in video order and collapse it to a single-video
    # bool or multi-video list during merge.
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
            # The current lower-level video loader only supports request-level
            # sampling parameters. Promote matching OpenAI content-part
            # parameters to the top level for now; if per-video parameters are
            # supported later, this can expand into a per-video structure.
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
            # The current HF processor accepts request-level images_kwargs.
            # First match the image min/max pixels used in Qwen reference
            # examples; if lower layers later support per-image sizing, this can
            # expand into per-image parameters.
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
            # Sampling/downsampling parameters in OpenAI audio content parts are
            # request-level audio_kwargs for Qwen3OmniNextProcessor.
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
