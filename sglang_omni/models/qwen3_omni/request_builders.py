# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-Omni stages."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import xxhash

from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.payload_types import (
    Qwen3OmniPipelineState,
    ThinkerOutput,
)
from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.types import ARRequestData

logger = logging.getLogger(__name__)

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
TALKER_STAGE = "talker_ar"
CODE2WAV_STAGE = "code2wav"
MM_AGGREGATE_STAGE = "mm_aggregate"
_INLINE_CPU_STREAM_CHUNK_MAX_BYTES_ENV = (
    "SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES"
)
_DECODE_STREAM_TOKEN_BATCH_SIZE_ENV = "SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE"
_DECODE_STREAM_IMMEDIATE_TOKEN_COUNT_ENV = (
    "SGLANG_OMNI_DECODE_STREAM_IMMEDIATE_TOKEN_COUNT"
)
_DECODE_STREAM_BATCH_STATE_MAX = 10000

# Note(Chenchen Hong): PyTorch sampling_seed must fit a positive int32.
MAX_INT32_POSITIVE = 0x7FFFFFFF


def _inline_cpu_stream_chunks_enabled() -> bool:
    raw = os.getenv(_INLINE_CPU_STREAM_CHUNK_MAX_BYTES_ENV)
    if raw is None or raw == "":
        return False
    try:
        return int(raw) > 0
    except ValueError:
        return False


def _decode_stream_token_batch_size() -> int:
    raw = os.getenv(_DECODE_STREAM_TOKEN_BATCH_SIZE_ENV)
    if raw is None or raw == "":
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _decode_stream_immediate_token_count() -> int:
    raw = os.getenv(_DECODE_STREAM_IMMEDIATE_TOKEN_COUNT_ENV)
    if raw is None or raw == "":
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _make_decode_stream_message(
    request_id: str,
    token_ids: list[int],
) -> OutgoingMessage:
    if len(token_ids) == 1:
        metadata = {"token_id": token_ids[0]}
        if _inline_cpu_stream_chunks_enabled():
            data: int | list[int] | torch.Tensor = token_ids[0]
        else:
            data = torch.tensor(token_ids, dtype=torch.long)
    else:
        metadata = {
            "token_id": token_ids[-1],
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
        }
        if _inline_cpu_stream_chunks_enabled():
            data = list(token_ids)
        else:
            data = torch.tensor(token_ids, dtype=torch.long)

    return OutgoingMessage(
        request_id=request_id,
        type="stream",
        data=data,
        target=DECODE_STAGE,
        metadata=metadata,
    )


class _DecodeStreamTokenBatcher:
    """Batch thinker text tokens after a short immediate-token prefix."""

    def __init__(self) -> None:
        self._buffers: OrderedDict[str, list[int]] = OrderedDict()
        self._sent_counts: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _evict_oldest(items: OrderedDict[str, Any]) -> None:
        while len(items) > _DECODE_STREAM_BATCH_STATE_MAX:
            items.popitem(last=False)

    def build(self, request_id: str, token_id: int) -> OutgoingMessage | None:
        batch_size = _decode_stream_token_batch_size()
        if batch_size <= 1:
            return _make_decode_stream_message(request_id, [token_id])

        sent_count = self._sent_counts.get(request_id, 0)
        if sent_count < _decode_stream_immediate_token_count():
            self._sent_counts[request_id] = sent_count + 1
            self._sent_counts.move_to_end(request_id)
            self._evict_oldest(self._sent_counts)
            return _make_decode_stream_message(request_id, [token_id])

        self._sent_counts.move_to_end(request_id)
        pending = self._buffers.setdefault(request_id, [])
        pending.append(token_id)
        self._buffers.move_to_end(request_id)
        self._evict_oldest(self._buffers)
        if len(pending) < batch_size:
            return None

        token_ids = list(pending)
        self._buffers.pop(request_id, None)
        return _make_decode_stream_message(request_id, token_ids)

_MEDIA_MODEL_INPUT_KEYS = (
    "audio_embeds",
    "image_embeds",
    "video_embeds",
    "deepstack_input_embeds",
    "deepstack_visual_embeds",
    "image_deepstack_visual_embeds",
    "video_deepstack_visual_embeds",
)

_MODALITY_MODEL_INPUT_KEYS = {
    "audio": ("audio_embeds",),
    "image": (
        "image_embeds",
        "deepstack_visual_embeds",
        "image_deepstack_visual_embeds",
    ),
    "video": (
        "video_embeds",
        "deepstack_visual_embeds",
        "video_deepstack_visual_embeds",
    ),
}


def _resolve_seed(params: dict[str, Any]) -> int | None:
    """Resolve random seed from request params (accepts both ``seed`` and ``sampling_seed``)."""
    for key in ("seed", "sampling_seed"):
        value = params.get(key)
        if value is not None:
            return int(value)
    return None


def _resolve_max_new_tokens(params: dict[str, Any], default: int = 2048) -> int:
    for key in ("max_new_tokens", "max_completion_tokens", "max_tokens"):
        value = params.get(key)
        if value is not None:
            # OpenAI chat's newer field is max_completion_tokens, while the
            # legacy-compatible field is max_tokens. SGLang SamplingParams uses
            # max_new_tokens. Keep this consistent with preprocessor context
            # length validation.
            return int(value)
    return int(default)


def _has_non_empty_media_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, dict):
        return any(_has_non_empty_media_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_empty_media_value(item) for item in value)
    return True


def _has_multimodal_model_inputs(model_inputs: dict[str, Any]) -> bool:
    return any(
        _has_non_empty_media_value(model_inputs.get(key))
        for key in _MEDIA_MODEL_INPUT_KEYS
    )


def _has_modality_model_inputs(model_inputs: dict[str, Any], modality: str) -> bool:
    return any(
        _has_non_empty_media_value(model_inputs.get(key))
        for key in _MODALITY_MODEL_INPUT_KEYS.get(modality, ())
    )


def _input_embeds_radix_extra_key(prefix: str, request_id: str, embeds: Any) -> str:
    return f"{prefix}:{request_id}:{id(embeds):x}"


def _media_cache_radix_extra_key(media_cache_keys: dict[str, Any]) -> str | None:
    if not media_cache_keys:
        return None
    parts = [
        f"{modality}={media_cache_keys[modality]}"
        for modality in sorted(media_cache_keys)
        if media_cache_keys.get(modality) is not None
    ]
    if not parts:
        return None
    return "media-cache:" + "|".join(parts)


def _install_prefix_cache_limit_patch(Req: Any) -> None:
    if getattr(Req, "_omni_prefix_cache_limit_patched", False):
        return

    original = Req.init_next_round_input

    def _init_next_round_input_with_omni_limit(self, tree_cache=None):
        limit = getattr(self, "_omni_max_prefix_cache_len", None)
        if limit is None or tree_cache is None:
            result = original(self, tree_cache)
        else:
            old_return_logprob = self.return_logprob
            old_logprob_start_len = self.logprob_start_len
            try:
                self.return_logprob = True
                if old_return_logprob and old_logprob_start_len >= 0:
                    self.logprob_start_len = min(int(old_logprob_start_len), int(limit))
                else:
                    self.logprob_start_len = int(limit)
                result = original(self, tree_cache)
            finally:
                self.return_logprob = old_return_logprob
                self.logprob_start_len = old_logprob_start_len
        return result

    Req._omni_original_init_next_round_input = original
    Req.init_next_round_input = _init_next_round_input_with_omni_limit
    Req._omni_prefix_cache_limit_patched = True


def _install_mamba_branching_hint_patch(Req: Any) -> None:
    if getattr(Req, "_omni_mamba_branching_hint_patched", False):
        return

    original = Req.init_next_round_input

    def _init_next_round_input_with_omni_mamba_hint(self, tree_cache=None):
        result = original(self, tree_cache)
        hint = getattr(self, "_omni_mamba_branching_seqlen", None)
        if hint is None or getattr(self, "mamba_branching_seqlen", None) is not None:
            return result
        try:
            hint = int(hint)
        except (TypeError, ValueError):
            return result
        prefix_indices = getattr(self, "prefix_indices", None)
        prefix_len = len(prefix_indices) if prefix_indices is not None else 0
        if hint > prefix_len:
            self.mamba_branching_seqlen = hint
        return result

    Req._omni_original_init_next_round_input_for_mamba_hint = original
    Req.init_next_round_input = _init_next_round_input_with_omni_mamba_hint
    Req._omni_mamba_branching_hint_patched = True


def _first_multimodal_token_index(
    input_ids: list[int],
    model_inputs: dict[str, Any],
    thinker_config: Any,
    pad_values: dict[str, int],
) -> int | None:
    first_index: int | None = None
    for modality, config_key in [
        ("image", "image_token_id"),
        ("video", "video_token_id"),
        ("audio", "audio_token_id"),
    ]:
        if not _has_modality_model_inputs(model_inputs, modality):
            continue
        token_id = pad_values.get(modality, getattr(thinker_config, config_key, None))
        if token_id is None:
            continue
        try:
            index = input_ids.index(int(token_id))
        except ValueError:
            continue
        first_index = index if first_index is None else min(first_index, index)
    return first_index


def _first_modality_token_index(
    input_ids: list[int],
    *,
    modality: str,
    model_inputs: dict[str, Any],
    thinker_config: Any,
    pad_values: dict[str, int],
) -> int | None:
    if not _has_modality_model_inputs(model_inputs, modality):
        return None
    token_id = pad_values.get(
        modality, getattr(thinker_config, f"{modality}_token_id", None)
    )
    if token_id is None:
        return None
    try:
        return input_ids.index(int(token_id))
    except ValueError:
        return None


def _mamba_branching_chunk_size() -> int | None:
    try:
        from sglang.srt.layers.attention.fla.chunk_delta_h import (
            CHUNK_SIZE as fla_chunk_size,
        )
    except Exception:
        return None
    try:
        chunk_size = int(fla_chunk_size)
    except (TypeError, ValueError):
        return None
    return chunk_size if chunk_size > 0 else None


def _media_mamba_branching_seqlen(
    input_ids: list[int],
    model_inputs: dict[str, Any],
    thinker_config: Any,
    pad_values: dict[str, int],
) -> int | None:
    """Aligned visual-prefix boundary for hybrid-SSM prefix cache reuse."""

    first_audio = _first_modality_token_index(
        input_ids,
        modality="audio",
        model_inputs=model_inputs,
        thinker_config=thinker_config,
        pad_values=pad_values,
    )
    if first_audio is None:
        return None

    visual_indices = [
        idx
        for idx in (
            _first_modality_token_index(
                input_ids,
                modality="image",
                model_inputs=model_inputs,
                thinker_config=thinker_config,
                pad_values=pad_values,
            ),
            _first_modality_token_index(
                input_ids,
                modality="video",
                model_inputs=model_inputs,
                thinker_config=thinker_config,
                pad_values=pad_values,
            ),
        )
        if idx is not None
    ]
    if not visual_indices or min(visual_indices) >= first_audio:
        return None

    chunk_size = _mamba_branching_chunk_size()
    if chunk_size is None:
        return None
    aligned = first_audio // chunk_size * chunk_size
    return aligned if aligned > 0 else None


def output_modalities(request: OmniRequest | None) -> set[str] | None:
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    modalities = metadata.get("output_modalities")
    if modalities is None:
        return None
    if isinstance(modalities, str):
        values = (modalities,)
    elif isinstance(modalities, (list, tuple, set)):
        values = modalities
    else:
        return None
    return {str(modality).lower() for modality in values}


def should_generate_audio_output(
    payload_or_request: StagePayload | OmniRequest | None,
) -> bool:
    request = (
        payload_or_request.request
        if isinstance(payload_or_request, StagePayload)
        else payload_or_request
    )
    modalities = output_modalities(request)
    return modalities is None or "audio" in modalities


def resolve_thinker_next_stages(
    request_id: str, output: StagePayload
) -> str | list[str]:
    del request_id, output
    return DECODE_STAGE


def resolve_mm_aggregate_next_stages(
    request_id: str, output: StagePayload
) -> str | list[str]:
    del request_id
    if should_generate_audio_output(output):
        return [THINKER_STAGE, TALKER_STAGE]
    return THINKER_STAGE


def resolve_thinker_stream_done_targets(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    if should_generate_audio_output(output):
        return [TALKER_STAGE, DECODE_STAGE]
    return [DECODE_STAGE]


def resolve_terminal_stages(request: OmniRequest) -> list[str]:
    if should_generate_audio_output(request):
        return [DECODE_STAGE, CODE2WAV_STAGE]
    return [DECODE_STAGE]


def resolve_preprocessing_next_stages(
    request_id: str, output: StagePayload
) -> list[str]:
    """Route to encoders for present media, then always aggregate."""

    del request_id
    state = Qwen3OmniPipelineState.from_dict(output.data)
    return [
        *_encoder_stages_with_model_inputs(state.encoder_inputs),
        MM_AGGREGATE_STAGE,
    ]


def resolve_mm_aggregate_wait_sources(
    request_id: str,
    from_stage: str,
    payload: StagePayload,
) -> list[str] | None:
    del request_id
    if from_stage != "preprocessing":
        return None
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    return ["preprocessing", *_active_encoder_stages(state.encoder_inputs)]


def project_thinker_to_decode(payload: StagePayload) -> StagePayload:
    """Keep decode payload focused on text detokenization state."""
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    state.prompt = _prompt_token_count_only(state.prompt)
    state.thinker_inputs = {}
    state.stream_state = _copy_mutable_containers(state.stream_state)

    if isinstance(state.thinker_out, dict):
        thinker_out = dict(state.thinker_out)
        if "extra_model_outputs" in thinker_out:
            thinker_out["extra_model_outputs"] = {}
        state.thinker_out = thinker_out

    if state.engine_outputs:
        engine_outputs = dict(state.engine_outputs)
        thinker_engine_out = engine_outputs.get(THINKER_STAGE)
        if isinstance(thinker_engine_out, dict):
            thinker_engine_out = dict(thinker_engine_out)
            if "extra_model_outputs" in thinker_engine_out:
                thinker_engine_out["extra_model_outputs"] = {}
            engine_outputs[THINKER_STAGE] = thinker_engine_out
        state.engine_outputs = engine_outputs

    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _prompt_token_count_only(prompt: Any) -> dict[str, Any] | None:
    if not isinstance(prompt, dict):
        return None
    prompt_tokens = _sequence_length(prompt.get("input_ids"))
    if prompt_tokens is None:
        return None
    return {"prompt_tokens": prompt_tokens}


def _sequence_length(value: Any) -> int | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError):
            return None
    try:
        return len(value)
    except TypeError:
        return None


def project_talker_to_code2wav(payload: StagePayload) -> StagePayload:
    """Keep code2wav payload as a request latch; code tensors arrive by stream."""
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={},
    )


@dataclass(slots=True)
class EncoderRequestData:
    """Typed encoder request data for pre-thinker stages."""

    model_inputs: dict[str, Any]
    cache_key: str | None = None
    skip_result: dict[str, Any] | None = None
    item_cache_keys: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    item_pixel_present: dict[str, tuple[bool, ...]] = field(default_factory=dict)


def build_encoder_request(
    state: Qwen3OmniPipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(model_inputs={}, skip_result={})
    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            model_inputs={},
            skip_result=skip_result if isinstance(skip_result, dict) else {},
        )
    cache_key = inputs.get("cache_key")
    item_cache_keys: dict[str, tuple[str | None, ...]] = {}
    item_pixel_present: dict[str, tuple[bool, ...]] = {}
    for modality, key_name in (
        ("image", "image_item_cache_keys"),
        ("video", "video_item_cache_keys"),
        ("audio", "audio_item_cache_keys"),
    ):
        raw_keys = inputs.get(key_name)
        if isinstance(raw_keys, (list, tuple)):
            item_cache_keys[modality] = tuple(
                str(item) if item is not None else None for item in raw_keys
            )
    for modality, key_name in (
        ("image", "image_item_pixel_present"),
        ("video", "video_item_pixel_present"),
    ):
        raw_mask = inputs.get(key_name)
        if isinstance(raw_mask, (list, tuple)):
            item_pixel_present[modality] = tuple(bool(item) for item in raw_mask)
    model_inputs = {
        k: v
        for k, v in inputs.items()
        if k
        not in (
            "cache_key",
            "_active",
            "image_item_cache_keys",
            "video_item_cache_keys",
            "audio_item_cache_keys",
            "image_item_pixel_present",
            "video_item_pixel_present",
        )
    }
    return EncoderRequestData(
        model_inputs=model_inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
        item_cache_keys=item_cache_keys,
        item_pixel_present=item_pixel_present,
    )


def apply_encoder_result(
    state: Qwen3OmniPipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    if isinstance(result, EncoderRequestData):
        encoder_out = result.skip_result if result.skip_result is not None else {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_lightweight_mm_inputs(mm_inputs: dict[str, Any]) -> dict[str, Any]:
    mm_image = mm_inputs.get("image", {})
    mm_audio = mm_inputs.get("audio", {})
    mm_video = mm_inputs.get("video", {})
    return {
        "image": _select_present_fields(mm_image, ("image_grid_thw",)),
        "audio": _select_present_fields(
            mm_audio,
            (
                "feature_attention_mask",
                "audio_feature_lengths",
                "audio_is_dependent",
            ),
        ),
        "video": _select_present_fields(
            mm_video,
            ("video_grid_thw", "video_second_per_grid", "use_audio_in_video"),
        ),
    }


def project_preprocessing_to_image_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=IMAGE_STAGE)


def project_preprocessing_to_audio_encoder(payload: StagePayload) -> StagePayload:
    return _project_preprocessing_to_encoder(payload, stage_name=AUDIO_STAGE)


def project_preprocessing_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    projected = Qwen3OmniPipelineState(
        prompt=_copy_mutable_containers(state.prompt),
        mm_inputs=build_lightweight_mm_inputs(state.mm_inputs),
        encoder_inputs=_project_encoder_input_metadata(state.encoder_inputs),
        stream_state=_copy_mutable_containers(state.stream_state),
    )
    return _payload_with_state(payload, projected)


def project_encoder_to_mm_aggregate(payload: StagePayload) -> StagePayload:
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    stage_name = _single_encoder_stage_name(state)
    encoder_out = state.encoder_outs.get(stage_name, {})
    projected = Qwen3OmniPipelineState(encoder_outs={stage_name: encoder_out})
    return _payload_with_state(payload, projected)


def project_mm_aggregate_to_thinker(payload: StagePayload) -> StagePayload:
    """Fan-out projection: give thinker its own prompt and model input containers."""
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    projected = Qwen3OmniPipelineState(
        prompt=_copy_mutable_containers(state.prompt),
        thinker_inputs=_copy_mutable_containers(state.thinker_inputs),
        stream_state=_copy_mutable_containers(state.stream_state),
    )
    return _payload_with_state(payload, projected)


def project_mm_aggregate_to_talker_ar(payload: StagePayload) -> StagePayload:
    """Early-submit projection: ship prompt + thinker_inputs to the talker."""
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    projected = Qwen3OmniPipelineState(
        prompt=_copy_mutable_containers(state.prompt),
        thinker_inputs=_copy_mutable_containers(state.thinker_inputs),
    )
    return _payload_with_state(payload, projected)


def _project_preprocessing_to_encoder(
    payload: StagePayload,
    *,
    stage_name: str,
) -> StagePayload:
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    projected = Qwen3OmniPipelineState(
        encoder_inputs=_select_encoder_inputs(
            state.encoder_inputs, stage_name=stage_name
        )
    )
    return _payload_with_state(payload, projected)


def _payload_with_state(
    payload: StagePayload, state: Qwen3OmniPipelineState
) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=_lightweight_downstream_request(payload.request),
        data=state.to_dict(),
    )


def _lightweight_downstream_request(request: Any) -> Any:
    return request.__class__(
        inputs={},
        params=dict(request.params or {}),
        metadata=dict(request.metadata or {}),
    )


def _copy_mutable_containers(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {key: _copy_mutable_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_mutable_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_mutable_containers(item) for item in value)
    if isinstance(value, set):
        return {_copy_mutable_containers(item) for item in value}
    if isinstance(value, bytearray):
        return bytearray(value)
    return value


def _select_encoder_inputs(
    encoder_inputs: dict[str, dict[str, Any]],
    *,
    stage_name: str,
) -> dict[str, dict[str, Any]]:
    stage_inputs = encoder_inputs.get(stage_name)
    if not isinstance(stage_inputs, dict):
        return {}
    return {stage_name: _copy_mutable_containers(stage_inputs)}


def _project_encoder_input_metadata(
    encoder_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    projected: dict[str, dict[str, Any]] = {}
    for stage_name, stage_inputs in encoder_inputs.items():
        if not isinstance(stage_inputs, dict):
            continue
        stage_metadata: dict[str, Any] = {}
        cache_key = stage_inputs.get("cache_key")
        if cache_key is not None:
            stage_metadata["cache_key"] = cache_key
        if stage_inputs.get("_skip"):
            stage_metadata["_skip"] = True
        elif _is_active_encoder_branch(stage_name, stage_inputs):
            stage_metadata["_active"] = True
        if stage_metadata:
            projected[stage_name] = stage_metadata
    return projected


def _encoder_stages_with_model_inputs(
    encoder_inputs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        stage_name
        for stage_name in (IMAGE_STAGE, AUDIO_STAGE)
        if _has_encoder_model_input(stage_name, encoder_inputs.get(stage_name))
    ]


def _active_encoder_stages(
    encoder_inputs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        stage_name
        for stage_name in (IMAGE_STAGE, AUDIO_STAGE)
        if _is_active_encoder_branch(stage_name, encoder_inputs.get(stage_name))
    ]


def _is_active_encoder_branch(stage_name: str, stage_inputs: Any) -> bool:
    if not isinstance(stage_inputs, dict) or stage_inputs.get("_skip"):
        return False
    active_marker = stage_inputs.get("_active")
    if active_marker is not None:
        return active_marker is True
    return _has_encoder_model_input(stage_name, stage_inputs)


def _has_encoder_model_input(stage_name: str, stage_inputs: Any) -> bool:
    if not isinstance(stage_inputs, dict) or stage_inputs.get("_skip"):
        return False
    if stage_inputs.get("_active") is False:
        return False
    if stage_name == IMAGE_STAGE:
        return (
            stage_inputs.get("pixel_values") is not None
            or stage_inputs.get("pixel_values_videos") is not None
        )
    if stage_name == AUDIO_STAGE:
        return stage_inputs.get("input_features") is not None
    return False


def _select_present_fields(
    source: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is not None:
            selected[key] = value
    return selected


def _single_encoder_stage_name(state: Qwen3OmniPipelineState) -> str:
    if len(state.encoder_outs) != 1:
        raise ValueError(
            f"Expected exactly one encoder output in payload, got {sorted(state.encoder_outs)}"
        )
    return next(iter(state.encoder_outs))


def build_thinker_request(
    state: Qwen3OmniPipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    prompt = state.prompt
    input_ids = prompt["input_ids"]
    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }

    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def _compute_mrope_positions(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> torch.Tensor | None:
    """Compute M-RoPE positions for multimodal inputs."""
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    spatial_merge_size = thinker_config.vision_config.spatial_merge_size
    image_token_id = thinker_config.image_token_id
    video_token_id = thinker_config.video_token_id
    vision_start_token_id = thinker_config.vision_start_token_id
    tokens_per_second = thinker_config.vision_config.tokens_per_second
    audio_token_id = thinker_config.audio_token_id
    audio_start_token_id = thinker_config.audio_start_token_id
    position_id_per_seconds = thinker_config.position_id_per_seconds
    use_audio_in_video = model_inputs.get("use_audio_in_video", False)
    audio_feature_lengths = model_inputs.get("audio_feature_lengths")

    ids_2d = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    # Move all tensors to CPU — get_rope_index creates CPU tensors internally
    ids_2d = ids_2d.cpu()
    if isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.cpu()
    if isinstance(video_grid_thw, torch.Tensor):
        video_grid_thw = video_grid_thw.cpu()
    second_per_grid_ts = model_inputs.get("video_second_per_grid")
    if isinstance(second_per_grid_ts, torch.Tensor):
        second_per_grid_ts = second_per_grid_ts.cpu()
    if isinstance(audio_feature_lengths, torch.Tensor):
        audio_feature_lengths = audio_feature_lengths.cpu()

    kwargs: dict[str, Any] = {
        "audio_token_id": audio_token_id,
        "audio_start_token_id": audio_start_token_id,
        "position_id_per_seconds": position_id_per_seconds,
        "use_audio_in_video": use_audio_in_video,
        "audio_seqlens": audio_feature_lengths,
    }

    mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        model_type="qwen3_omni_moe",
        tokens_per_second=tokens_per_second,
        input_ids=ids_2d,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        **kwargs,
    )
    # mrope_positions: [3, 1, seq_len] -> [3, seq_len]
    return mrope_positions.squeeze(1), mrope_position_delta


def build_sglang_thinker_request(
    state: Qwen3OmniPipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    request_id: str | None = None,
    thinker_config: Any = None,
    mrope_position_builder: (
        Callable[[torch.Tensor, dict[str, Any], Any], Any] | None
    ) = None,
    limit_prefix_cache_before_media: bool = False,
    mamba_media_branching_cache: bool = False,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData from pipeline state.

    Constructs a SGLang Req with normalized SamplingParams, then wraps it
    in SGLangARRequestData (which inherits ARRequestData).
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    # SGLangARRequestData already imported at module level

    prompt = state.prompt
    input_ids = prompt["input_ids"]
    original_input_ids = input_ids

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}
    rid = request_id or "req-0"

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v
            for k, v in thinker_inputs.items()
            if k not in ("capture_model_output_keys", "media_cache_keys")
        }
    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    media_cache_keys = thinker_inputs.get("media_cache_keys", {})
    if (model_inputs or media_cache_keys) and "original_input_ids" not in model_inputs:
        prompt_original_ids = prompt.get("original_input_ids")
        if prompt_original_ids is not None:
            model_inputs["original_input_ids"] = prompt_original_ids
        else:
            model_inputs["original_input_ids"] = original_input_ids
    pad_values: dict[str, int] = {}
    if thinker_config is not None and (
        media_cache_keys or _has_multimodal_model_inputs(model_inputs)
    ):
        token_id_map: dict[int, int] = {}
        for modality, orig_token_id in [
            ("image", thinker_config.image_token_id),
            ("video", thinker_config.video_token_id),
            ("audio", thinker_config.audio_token_id),
        ]:
            if orig_token_id is None:
                continue
            cache_key = media_cache_keys.get(modality)
            if cache_key is None:
                if not _has_modality_model_inputs(model_inputs, modality):
                    continue
                pad_key = _input_embeds_radix_extra_key(
                    f"media-pad:{modality}", rid, model_inputs
                )
            else:
                pad_key = str(cache_key)
            h = xxhash.xxh3_64(pad_key.encode()).intdigest()
            pad_val = vocab_size + h % (1 << 62)
            pad_values[modality] = pad_val
            token_id_map[orig_token_id] = pad_val
        if token_id_map:
            input_ids = input_ids.clone()
            for orig_id, pad_val in token_id_map.items():
                input_ids[input_ids == orig_id] = pad_val
        if pad_values:
            model_inputs["pad_values"] = pad_values
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)
    input_ids_list = input_ids.to(dtype=torch.long).tolist()

    max_new_tokens = _resolve_max_new_tokens(params, 2048)
    temperature = params.get("temperature", 0.0)
    top_p = params.get("top_p", 1.0)
    top_k = params.get("top_k", -1)
    min_p = params.get("min_p", 0.0)
    frequency_penalty = params.get("frequency_penalty", 0.0)
    presence_penalty = params.get("presence_penalty", 0.0)
    repetition_penalty = params.get("repetition_penalty", 1.0)
    stop = params.get("stop") or []
    stop_token_ids = params.get("stop_token_ids") or []
    seed = _resolve_seed(params)

    # Build SGLang SamplingParams and normalize
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        stop=stop,
        stop_token_ids=stop_token_ids,
        sampling_seed=seed,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    # Build SGLang Req
    extra_key = _media_cache_radix_extra_key(media_cache_keys)
    if extra_key is None and _has_multimodal_model_inputs(model_inputs) and not pad_values:
        extra_key = _input_embeds_radix_extra_key("media-inputs", rid, model_inputs)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        extra_key=extra_key,
        vocab_size=vocab_size,
    )
    if limit_prefix_cache_before_media and thinker_config is not None:
        prefix_cache_limit = _first_multimodal_token_index(
            input_ids_list, model_inputs, thinker_config, pad_values
        )
        if prefix_cache_limit is not None:
            _install_prefix_cache_limit_patch(Req)
            req._omni_max_prefix_cache_len = int(prefix_cache_limit)
    if mamba_media_branching_cache and thinker_config is not None:
        branching_seqlen = _media_mamba_branching_seqlen(
            input_ids_list, model_inputs, thinker_config, pad_values
        )
        if branching_seqlen is not None:
            _install_mamba_branching_hint_patch(Req)
            req._omni_mamba_branching_seqlen = int(branching_seqlen)
    req.tokenizer = tokenizer

    # Compute M-RoPE positions and attach multimodal_inputs to Req
    if thinker_config is not None and model_inputs:
        mrope_builder = mrope_position_builder or _compute_mrope_positions
        mrope_result = mrope_builder(
            original_input_ids.to(dtype=torch.long), model_inputs, thinker_config
        )
        if mrope_result is not None:
            mrope_positions, mrope_position_delta = mrope_result
            mm_inputs = MultimodalInputs(mm_items=[])
            mm_inputs.mrope_positions = mrope_positions
            mm_inputs.mrope_position_delta = mrope_position_delta
            req.multimodal_inputs = mm_inputs

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None
    req._codec_suppress_tokens = None

    # Build SGLangARRequestData — output_ids points to req.output_ids
    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    return data


def build_sglang_talker_request(
    thinker_hidden_states: torch.Tensor,
    *,
    tokenizer: Any,
    codec_vocab_size: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.05,
    request_id: str | None = None,
    codec_bos_id: int = 2149,
    codec_eos_id: int | None = None,
    suppress_tokens: list[int] | None = None,
    thinker_layer_hidden: torch.Tensor | None = None,
    thinker_token_ids: list[int] | torch.Tensor | None = None,
    audio_token_id: int | None = None,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
    talker_input_embeds: torch.Tensor | None = None,
    talker_input_ids: torch.Tensor | list[int] | None = None,
    input_embeds_are_projected: bool = False,
    pending_text_queue: (
        PendingTextTensorQueue | Iterable[torch.Tensor] | torch.Tensor | None
    ) = None,
    tts_pad_embed: torch.Tensor | None = None,
    thinker_chunks_done: bool = True,
    thinker_config: Any = None,
    talker_model_inputs: dict[str, Any] | None = None,
    seed: int | None = None,
    mrope_position_builder: (
        Callable[[torch.Tensor, dict[str, Any], Any], Any] | None
    ) = None,
) -> "SGLangARRequestData":
    """Build SGLang AR request for the Talker from thinker hidden states.

    Uses dummy input_ids of matching length for position tracking, while the
    request data keeps a device-backed FIFO of future text rows for decode.

    Stores the original tensor on SGLangARRequestData.prefill_input_embeds
    when input_embeds_are_projected, so the model runner can skip the
    list→tensor reconversion during prefill.

    Args:
        thinker_hidden_states: Embed layer hidden states [seq_len, hidden_size].
        thinker_layer_hidden: Optional layer-N hidden states for dual-layer mode.
        thinker_token_ids: Optional thinker output token ids aligned with hidden states.
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    # SGLangARRequestData already imported at module level

    if talker_input_embeds is not None:
        prefill_embeds_tensor = talker_input_embeds
        input_ids_tensor = torch.as_tensor(talker_input_ids, dtype=torch.long)
        input_ids_list = input_ids_tensor.tolist()
        seq_len = len(input_ids_list)
    else:
        # thinker_hidden_states: [seq_len, thinker_hidden_size]
        seq_len = thinker_hidden_states.shape[0]

        # Dummy input_ids — codec BOS token repeated for each position
        input_ids_list = [codec_bos_id] * seq_len
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)

        prefill_embeds_tensor = thinker_hidden_states

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=[int(codec_eos_id)] if codec_eos_id is not None else None,
        logit_bias=None,
        sampling_seed=seed,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(codec_vocab_size)

    rid = request_id or "talker-req-0"
    input_embeds_extra_key = _input_embeds_radix_extra_key(
        "projected-input-embeds" if input_embeds_are_projected else "input-embeds",
        rid,
        prefill_embeds_tensor,
    )
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        # Convert hidden states to list-of-lists for Req.input_embeds
        input_embeds=(
            None if input_embeds_are_projected else prefill_embeds_tensor.cpu().tolist()
        ),
        eos_token_ids={int(codec_eos_id)} if codec_eos_id is not None else None,
        extra_key=input_embeds_extra_key,
        vocab_size=codec_vocab_size,
    )
    req.tokenizer = tokenizer
    req._input_embeds_are_projected = bool(input_embeds_are_projected)
    req.omni_model_inputs = dict(talker_model_inputs or {})
    req._omni_consumed = None
    req._codec_suppress_tokens = (
        tuple(int(token_id) for token_id in suppress_tokens)
        if suppress_tokens
        else None
    )
    if thinker_config is not None and talker_model_inputs:
        mrope_builder = mrope_position_builder or _compute_mrope_positions
        mrope_result = mrope_builder(
            input_ids_tensor.to(dtype=torch.long),
            talker_model_inputs or {},
            thinker_config,
        )
        if mrope_result is not None:
            mrope_positions, mrope_position_delta = mrope_result
            mm_inputs = MultimodalInputs(mm_items=[])
            mm_inputs.mrope_positions = mrope_positions
            mm_inputs.mrope_position_delta = mrope_position_delta
            req.multimodal_inputs = mm_inputs

    multimodal_mask: torch.Tensor | None = None
    if thinker_token_ids is not None:
        token_ids = torch.as_tensor(thinker_token_ids, dtype=torch.long)
        if token_ids.numel() == seq_len:
            mask = torch.zeros(seq_len, dtype=torch.bool)
            for token_id in (audio_token_id, image_token_id, video_token_id):
                if token_id is not None:
                    mask |= token_ids == int(token_id)
            multimodal_mask = mask

    if thinker_layer_hidden is not None:
        req.omni_model_inputs["talker_layer_hidden_states"] = thinker_layer_hidden
        req.omni_model_inputs["talker_multimodal_mask"] = multimodal_mask
    elif req.omni_model_inputs:
        req.omni_model_inputs["talker_layer_hidden_states"] = None
        req.omni_model_inputs["talker_multimodal_mask"] = None
    else:
        req.omni_model_inputs = None

    data = SGLangARRequestData(
        input_ids=input_ids_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
        prefill_input_embeds=(
            prefill_embeds_tensor if input_embeds_are_projected else None
        ),
    )
    data.suppress_tokens = list(req._codec_suppress_tokens or [])
    data.talker_model_inputs = dict(talker_model_inputs or {})
    if thinker_layer_hidden is not None:
        data.extra_model_outputs["thinker_layer_hidden"] = thinker_layer_hidden
    if multimodal_mask is not None:
        data.extra_model_outputs["talker_multimodal_mask"] = multimodal_mask
    data.input_embeds_are_projected = bool(input_embeds_are_projected)
    data.thinker_chunks_done = bool(thinker_chunks_done)
    data.pending_text_queue = coerce_pending_text_queue(pending_text_queue)
    data.tts_pad_embed = tts_pad_embed
    return data


def apply_thinker_result(
    state: Qwen3OmniPipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    output_ids = list(result.output_ids)
    thinker_out: ThinkerOutput = {
        "output_ids": output_ids,
        "step": len(output_ids),
        "is_final": True,
        "extra_model_outputs": dict(result.extra_model_outputs),
    }

    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is not None:
        thinker_out["finish_reason"] = finish_reason

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


def make_thinker_stream_output_builder():
    decode_stream_batcher = _DecodeStreamTokenBatcher()

    def _normalize_chunk_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
        if hidden is None:
            return None
        if hidden.ndim == 1:
            return hidden
        if hidden.ndim == 2:
            return hidden[0]
        return None

    def _split_dual_layer_hidden(
        hidden: dict[str | int, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if isinstance(hidden, torch.Tensor):
            return _normalize_chunk_hidden(hidden), None

        embed = hidden.get("embed")
        if embed is None and 0 in hidden:
            embed = hidden[0]
        if embed is None and "0" in hidden:
            embed = hidden["0"]

        layer_hidden = None
        for key, value in hidden.items():
            if key in ("embed", 0, "0"):
                continue
            if isinstance(value, torch.Tensor):
                layer_hidden = value
                break
        return _normalize_chunk_hidden(embed), _normalize_chunk_hidden(layer_hidden)

    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = getattr(req_data, "req", None)
        if req is not None and int(getattr(req, "is_chunked", 0) or 0) > 0:
            # While chunked prefill is still consuming prompt tokens, suppress
            # hidden-state streaming to the talker.
            # Emitting chunks this early lets prompt-side states masquerade as the
            # first assistant token and can leak the user/ref-text prompt into TTS.
            return []
        if req_output.data is None:
            return []

        token_id = int(req_output.data)
        messages: list[OutgoingMessage] = []

        # Skip per-token decode emit when not streaming; talker_ar below stays
        # unconditional since talker generates audio either way.
        stage_payload = req_data.stage_payload
        is_streaming = bool(
            stage_payload is not None
            and (stage_payload.request.params or {}).get("stream", False)
        )
        if is_streaming:
            # Inline CPU stream chunks keep tiny text-token deltas off the relay
            # tensor path. Preserve the old tensor payload when inline is not
            # enabled because relay_io.write_blob is tensor-only.
            decode_msg = decode_stream_batcher.build(request_id, token_id)
            if decode_msg is not None:
                messages.append(decode_msg)

        if not should_generate_audio_output(stage_payload):
            return messages

        # Speech mode: also stream hidden states to the talker for codec gen.
        extra = req_output.extra
        if isinstance(extra, dict) and "hidden_states" in extra:
            embed, layer_hidden = _split_dual_layer_hidden(extra["hidden_states"])
            if embed is not None:
                metadata = {"token_id": token_id}
                if layer_hidden is not None:
                    metadata["layer_hidden"] = layer_hidden
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=embed,
                        target="talker_ar",
                        metadata=metadata,
                    )
                )
            elif layer_hidden is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=layer_hidden,
                        target="talker_ar",
                        metadata={"token_id": token_id},
                    )
                )

        return messages

    return _build_stream_output


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    thinker_config: Any = None,
    stage_name: str = "thinker",
    mrope_position_builder: (
        Callable[[torch.Tensor, dict[str, Any], Any], Any] | None
    ) = None,
):
    """Build model-specific StagePayload <-> scheduler adapters for thinker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        params = payload.request.params or {}
        req_data = build_sglang_thinker_request(
            state,
            params=params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
            thinker_config=thinker_config,
            mrope_position_builder=mrope_position_builder,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        apply_thinker_result(state, stage_name=stage_name, result=data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


def make_talker_scheduler_adapters(
    *,
    tokenizer: Any,
    codec_vocab_size: int,
    model: Any,
    model_path: str,
    thinker_config: Any,
    required_aux_hidden_key: int,
    codec_bos_id: int = 2149,
    codec_eos_id: int | None = None,
    codec_nothink_id: int = 2155,
    codec_think_bos_id: int = 2156,
    codec_think_eos_id: int = 2157,
    codec_pad_id: int = 2148,
    audio_token_id: int | None = None,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
    tts_bos_token_id: int = 151672,
    tts_eos_token_id: int = 151673,
    tts_pad_token_id: int = 151671,
    im_start_token_id: int = 151644,
    im_end_token_id: int = 151645,
    system_token_id: int = 8948,
    user_token_id: int = 872,
    assistant_token_id: int = 77091,
    speaker_map: dict[str, int] | None = None,
    mrope_position_builder: (
        Callable[[torch.Tensor, dict[str, Any], Any], Any] | None
    ) = None,
):
    """Build model-specific StagePayload <-> scheduler adapters for talker."""
    prefill_builder = TalkerPrefillBuilder(
        model=model,
        model_path=model_path,
        audio_token_id=audio_token_id,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        tts_bos_token_id=tts_bos_token_id,
        tts_eos_token_id=tts_eos_token_id,
        tts_pad_token_id=tts_pad_token_id,
        im_start_token_id=im_start_token_id,
        im_end_token_id=im_end_token_id,
        system_token_id=system_token_id,
        user_token_id=user_token_id,
        assistant_token_id=assistant_token_id,
        codec_bos_id=codec_bos_id,
        codec_nothink_id=codec_nothink_id,
        codec_think_bos_id=codec_think_bos_id,
        codec_think_eos_id=codec_think_eos_id,
        codec_pad_id=codec_pad_id,
        speaker_map=speaker_map,
    )

    def _resolve_talker_sampling_config(params: dict[str, Any]) -> dict[str, Any]:
        codec_eos_id = int(getattr(model.config, "codec_eos_token_id", -1))
        suppress_tokens = [
            token_id
            for token_id in range(max(codec_vocab_size - 1024, 0), codec_vocab_size)
            if token_id != codec_eos_id
        ]
        return {
            "max_new_tokens": int(params.get("talker_max_new_tokens", 4096)),
            "temperature": float(params.get("talker_temperature", 0.9)),
            "top_k": int(params.get("talker_top_k", 50)),
            "top_p": float(params.get("talker_top_p", 1.0)),
            "min_p": float(params.get("talker_min_p", 0.0)),
            "repetition_penalty": float(params.get("talker_repetition_penalty", 1.05)),
            "codec_eos_id": codec_eos_id if codec_eos_id >= 0 else None,
            "suppress_tokens": suppress_tokens,
            "seed": _resolve_seed(params),
        }

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        return _build_talker_request_data(
            payload,
            prefill_builder=prefill_builder,
            tokenizer=tokenizer,
            codec_vocab_size=codec_vocab_size,
            codec_bos_id=codec_bos_id,
            audio_token_id=audio_token_id,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            thinker_config=thinker_config,
            resolve_sampling_config=_resolve_talker_sampling_config,
            mrope_position_builder=mrope_position_builder,
        )

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=payload.data,
        )

    return (
        request_builder,
        result_adapter,
        prefill_builder.append_text_chunk,
        prefill_builder.mark_thinker_done,
    )


def _build_talker_request_data(
    payload: StagePayload,
    *,
    prefill_builder: TalkerPrefillBuilder,
    tokenizer: Any,
    codec_vocab_size: int,
    codec_bos_id: int,
    audio_token_id: int | None,
    image_token_id: int | None,
    video_token_id: int | None,
    thinker_config: Any,
    resolve_sampling_config: Callable[[dict[str, Any]], dict[str, Any]],
    mrope_position_builder: (
        Callable[[torch.Tensor, dict[str, Any], Any], Any] | None
    ) = None,
) -> SGLangARRequestData:
    params = payload.request.params
    sampling_cfg = resolve_sampling_config(params)
    if sampling_cfg.get("seed") is None:
        sampling_cfg["seed"] = (
            xxhash.xxh64_intdigest(str(payload.request_id).encode("utf-8"))
            & MAX_INT32_POSITIVE
        )
    thinker_chunks = list(payload.prefetched_chunks)
    thinker_done = bool(payload.prefetched_stream_done)

    if not thinker_chunks:
        raise RuntimeError(
            "talker request_builder requires prefetched thinker chunks; "
            "check the partial-start readiness policy or upstream wiring"
        )

    prompt_prefill = prefill_builder.build_prompt_prefill(
        payload,
        thinker_chunks,
        thinker_done=thinker_done,
    )
    pending_text_queue = prompt_prefill["pending_text_queue"]
    pending_text_rows = len(pending_text_queue) if pending_text_queue is not None else 0
    logger.debug(
        "talker_request_build request_id=%s thinker_chunks=%d "
        "talker_input_rows=%d future_text_rows=%d thinker_done=%s",
        payload.request_id,
        len(thinker_chunks),
        int(prompt_prefill["input_embeds"].shape[0]),
        pending_text_rows,
        thinker_done,
    )
    req_data = build_sglang_talker_request(
        thinker_hidden_states=torch.empty(0),
        tokenizer=tokenizer,
        codec_vocab_size=codec_vocab_size,
        max_new_tokens=sampling_cfg["max_new_tokens"],
        temperature=sampling_cfg["temperature"],
        top_k=sampling_cfg["top_k"],
        top_p=sampling_cfg["top_p"],
        min_p=sampling_cfg.get("min_p", 0.0),
        repetition_penalty=sampling_cfg["repetition_penalty"],
        request_id=payload.request_id,
        codec_bos_id=codec_bos_id,
        codec_eos_id=sampling_cfg["codec_eos_id"],
        suppress_tokens=sampling_cfg["suppress_tokens"],
        audio_token_id=audio_token_id,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        talker_input_embeds=prompt_prefill["input_embeds"],
        talker_input_ids=prompt_prefill["input_ids"],
        input_embeds_are_projected=True,
        pending_text_queue=pending_text_queue,
        tts_pad_embed=prompt_prefill["tts_pad_embed"],
        thinker_chunks_done=thinker_done,
        thinker_config=thinker_config,
        talker_model_inputs=prompt_prefill["prompt_model_inputs"],
        seed=sampling_cfg.get("seed"),
        mrope_position_builder=mrope_position_builder,
    )
    req_data.tts_eos_embed = prompt_prefill["tts_eos_embed"]
    req_data.stage_payload = payload
    return req_data
