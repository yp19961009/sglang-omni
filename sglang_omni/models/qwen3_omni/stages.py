# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3-Omni pipelines.

Each factory returns either:
- A callable (compute_fn) for simple stages
- An OmniScheduler for AR stages
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_omni.bootstrap import create_thinker_scheduler
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    create_streaming_detokenize_scheduler,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes
from sglang_omni.utils.misc import avail_gpu_mem

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"

logger = logging.getLogger(__name__)

_ENCODER_CACHE_MAX_BYTES_ENV = "SGLANG_OMNI_ENCODER_CACHE_MAX_BYTES"
_ENCODER_CACHE_MAX_ENTRIES_ENV = "SGLANG_OMNI_ENCODER_CACHE_MAX_ENTRIES"
_STORE_ITEM_PLAN_COMBINED_CACHE_ENV = (
    "SGLANG_OMNI_STORE_ITEM_PLAN_COMBINED_ENCODER_CACHE"
)
_IMAGE_ENCODER_BATCH_BUDGET_BYTES_ENV = (
    "SGLANG_OMNI_IMAGE_ENCODER_BATCH_BUDGET_BYTES"
)
_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES_ENV = (
    "SGLANG_OMNI_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES"
)
_VISUAL_ITEM_BATCH_STATS_ENV = "SGLANG_OMNI_VISUAL_ITEM_BATCH_STATS"
_COMPACT_VISUAL_ENCODER_RESULTS_ENV = "SGLANG_OMNI_COMPACT_VISUAL_ENCODER_RESULTS"
_COMPACT_AUDIO_ENCODER_RESULTS_ENV = "SGLANG_OMNI_COMPACT_AUDIO_ENCODER_RESULTS"
_IMAGE_ENCODER_EMPTY_CACHE_ENV = "SGLANG_OMNI_IMAGE_ENCODER_EMPTY_CACHE"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in ("", "0", "false", "no", "off")


# Image-encoder batching budget; the multiplier accounts for transient activations.
QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES = _env_int(
    _IMAGE_ENCODER_BATCH_BUDGET_BYTES_ENV,
    10 * 1024**3,
)
QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER = 5
QWEN3_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES = _env_int(
    _IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES_ENV,
    QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
)


# CPU LRU cap for repeated-media encoder outputs.
QWEN3_ENCODER_CACHE_MAX_BYTES = _env_int(_ENCODER_CACHE_MAX_BYTES_ENV, 4 * 1024**3)
QWEN3_ENCODER_CACHE_MAX_ENTRIES = _env_int(_ENCODER_CACHE_MAX_ENTRIES_ENV, 64)


@dataclass(frozen=True)
class _ArMemoryContract:
    mem_fraction_static_pinned: bool
    effective_total_gpu_memory_fraction: float | None
    applied_encoder_mem_reserve: float


def _apply_qwen_thinker_encoder_reserve(
    server_args: Any,
    *,
    has_explicit_mem_fraction_static: bool,
    encoder_mem_reserve: float,
) -> bool:
    if has_explicit_mem_fraction_static:
        return False
    apply_encoder_mem_reserve(server_args, encoder_mem_reserve)
    return True


def _apply_colocated_ar_memory_contract(
    overrides: dict[str, Any],
    *,
    stage_name: str,
    total_gpu_memory_fraction: float | None,
    encoder_mem_reserve: float = 0.0,
) -> _ArMemoryContract:
    """Derive or validate SGLang AR memory args for a colocated stage."""

    if total_gpu_memory_fraction is None:
        return _ArMemoryContract(
            mem_fraction_static_pinned=overrides.get("mem_fraction_static") is not None,
            effective_total_gpu_memory_fraction=None,
            applied_encoder_mem_reserve=0.0,
        )

    explicit_mem_fraction = overrides.get("mem_fraction_static")
    if explicit_mem_fraction is not None:
        if encoder_mem_reserve:
            raise ValueError(
                f"Stage {stage_name} cannot apply encoder_mem_reserve when "
                "runtime.sglang_server_args.mem_fraction_static is explicitly set."
            )
        if abs(float(explicit_mem_fraction) - total_gpu_memory_fraction) > 1e-3:
            raise ValueError(
                f"Stage {stage_name} sets conflicting colocated memory "
                "contracts: runtime.resources.total_gpu_memory_fraction="
                f"{total_gpu_memory_fraction:.3f} and "
                "runtime.sglang_server_args.mem_fraction_static="
                f"{float(explicit_mem_fraction):.3f}. Use one value or make "
                "the explicit SGLang override match the stage total budget."
            )
        return _ArMemoryContract(
            mem_fraction_static_pinned=True,
            effective_total_gpu_memory_fraction=total_gpu_memory_fraction,
            applied_encoder_mem_reserve=0.0,
        )

    effective_total_gpu_memory_fraction = _apply_colocated_encoder_mem_reserve(
        total_gpu_memory_fraction,
        encoder_mem_reserve,
    )
    overrides["mem_fraction_static"] = effective_total_gpu_memory_fraction
    applied_encoder_mem_reserve = (
        encoder_mem_reserve
        if effective_total_gpu_memory_fraction != total_gpu_memory_fraction
        else 0.0
    )
    return _ArMemoryContract(
        mem_fraction_static_pinned=True,
        effective_total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
        applied_encoder_mem_reserve=applied_encoder_mem_reserve,
    )


def _apply_colocated_encoder_mem_reserve(
    total_gpu_memory_fraction: float,
    encoder_mem_reserve: float,
) -> float:
    if not 0.0 <= encoder_mem_reserve < 1.0:
        raise ValueError("encoder_mem_reserve must be in [0, 1)")
    if encoder_mem_reserve == 0:
        return total_gpu_memory_fraction

    effective_total_gpu_memory_fraction = (
        total_gpu_memory_fraction - encoder_mem_reserve
    )
    if effective_total_gpu_memory_fraction < 0.1:
        raise ValueError(
            f"colocated total_gpu_memory_fraction {total_gpu_memory_fraction:.3f} "
            f"minus encoder_mem_reserve {encoder_mem_reserve:.3f} = "
            f"{effective_total_gpu_memory_fraction:.3f} is below the safe floor "
            "0.1; lower encoder_mem_reserve or increase the thinker stage budget."
        )
    return round(effective_total_gpu_memory_fraction, 3)


def load_state(payload: StagePayload) -> Qwen3OmniPipelineState:
    return Qwen3OmniPipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Qwen3OmniPipelineState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _run_single_encoder_payload(
    payload: StagePayload,
    *,
    stage_name: str,
    model: Any,
    cache: StageOutputCache | None = None,
) -> StagePayload:
    if stage_name == IMAGE_STAGE:
        return _run_single_image_encoder_payload(payload, model=model, cache=cache)
    if stage_name == AUDIO_STAGE:
        return _run_single_audio_encoder_payload(payload, model=model, cache=cache)

    state = load_state(payload)
    request = build_encoder_request(state, stage_name=stage_name)
    if request.skip_result is not None:
        result = request.skip_result
    else:
        result = _lookup_cached_encoder_output(
            request=request,
            request_id=payload.request_id,
            stage_name=stage_name,
            cache=cache,
        )
        if result is None:
            with torch.no_grad():
                result = model(**request.model_inputs)
            _store_cached_encoder_output(
                request=request,
                request_id=payload.request_id,
                stage_name=stage_name,
                cache=cache,
                result=result,
            )
    apply_encoder_result(state, stage_name=stage_name, result=result)
    return store_state(payload, state)


def _run_single_audio_encoder_payload(
    payload: StagePayload,
    *,
    model: Any,
    cache: StageOutputCache | None = None,
) -> StagePayload:
    state = load_state(payload)
    request = build_encoder_request(state, stage_name=AUDIO_STAGE)
    if request.skip_result is not None:
        apply_encoder_result(state, stage_name=AUDIO_STAGE, result=request.skip_result)
        return store_state(payload, state)

    cached = _lookup_cached_encoder_output(
        request=request,
        request_id=payload.request_id,
        stage_name=AUDIO_STAGE,
        cache=cache,
    )
    if cached is not None:
        apply_encoder_result(state, stage_name=AUDIO_STAGE, result=cached)
        return store_state(payload, state)

    plan = _prepare_audio_item_cache_plan(
        idx=0,
        payload=payload,
        state=state,
        request=request,
        cache=cache,
    )
    if plan is not None:
        results: list[StagePayload | None] = [None]
        _execute_audio_item_cache_plans(
            [plan],
            model=model,
            cache=cache,
            results=results,
        )
        if results[0] is not None:
            return results[0]

    with torch.no_grad():
        result = model(**request.model_inputs)
    result = _compact_audio_encoder_result(result)
    _store_cached_encoder_output(
        request=request,
        request_id=payload.request_id,
        stage_name=AUDIO_STAGE,
        cache=cache,
        result=result,
    )
    apply_encoder_result(state, stage_name=AUDIO_STAGE, result=result)
    return store_state(payload, state)


def _run_single_image_encoder_payload(
    payload: StagePayload,
    *,
    model: Any,
    cache: StageOutputCache | None = None,
) -> StagePayload:
    state = load_state(payload)
    request = build_encoder_request(state, stage_name=IMAGE_STAGE)
    if request.skip_result is not None:
        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=request.skip_result)
        return store_state(payload, state)

    cached = _lookup_cached_encoder_output(
        request=request,
        request_id=payload.request_id,
        stage_name=IMAGE_STAGE,
        cache=cache,
    )
    if cached is not None:
        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=cached)
        return store_state(payload, state)

    if _image_request_is_batchable(request):
        plan = _prepare_visual_item_cache_plan(
            idx=0,
            payload=payload,
            state=state,
            request=request,
            model=model,
            cache=cache,
        )
        if plan is not None:
            results: list[StagePayload | None] = [None]
            _execute_visual_item_cache_plans(
                [plan],
                model=model,
                cache=cache,
                results=results,
            )
            if results[0] is not None:
                return results[0]

    with torch.no_grad():
        result = model(**request.model_inputs)
    _store_cached_encoder_output(
        request=request,
        request_id=payload.request_id,
        stage_name=IMAGE_STAGE,
        cache=cache,
        result=result,
    )
    apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
    return store_state(payload, state)


def _image_request_is_batchable(request: Any) -> bool:
    if request.skip_result is not None:
        return False
    input_dict = request.model_inputs
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        value = input_dict.get(key)
        if value is not None and not isinstance(value, torch.Tensor):
            return False
    return True


def _split_visual_features(
    tensor: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]


def _split_visual_multiscale(
    tensors: list[torch.Tensor] | None,
    *,
    start: int,
    end: int,
) -> list[torch.Tensor] | None:
    if tensors is None:
        return None
    return [tensor[start:end] for tensor in tensors]


def _compact_visual_encoder_results_enabled() -> bool:
    return _env_bool(_COMPACT_VISUAL_ENCODER_RESULTS_ENV, default=True)


def _compact_audio_encoder_results_enabled() -> bool:
    return _env_bool(_COMPACT_AUDIO_ENCODER_RESULTS_ENV, default=True)


def _image_encoder_empty_cache_enabled() -> bool:
    return _env_bool(_IMAGE_ENCODER_EMPTY_CACHE_ENV, default=False)


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except RuntimeError:
        return _tensor_bytes(tensor)


def _compact_visual_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if not _compact_visual_encoder_results_enabled():
        return tensor
    if tensor.numel() == 0:
        return tensor.contiguous()

    tensor_bytes = _tensor_bytes(tensor)
    storage_bytes = _tensor_storage_nbytes(tensor)
    if (
        not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or storage_bytes > tensor_bytes
    ):
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _compact_visual_encoder_result(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _compact_visual_tensor(value)
    if isinstance(value, dict):
        return {
            key: _compact_visual_encoder_result(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_compact_visual_encoder_result(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_compact_visual_encoder_result(item) for item in value)
    return value


def _compact_audio_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if not _compact_audio_encoder_results_enabled():
        return tensor
    if tensor.numel() == 0:
        return tensor.contiguous()

    tensor_bytes = _tensor_bytes(tensor)
    storage_bytes = _tensor_storage_nbytes(tensor)
    if (
        not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or storage_bytes > tensor_bytes
    ):
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _compact_audio_encoder_result(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _compact_audio_tensor(value)
    if isinstance(value, dict):
        return {
            key: _compact_audio_encoder_result(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_compact_audio_encoder_result(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_compact_audio_encoder_result(item) for item in value)
    return value


def _empty_image_encoder_cache_if_requested() -> None:
    if not _image_encoder_empty_cache_enabled():
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


@dataclass
class _VisualItemCachePlan:
    idx: int
    payload: StagePayload
    state: Qwen3OmniPipelineState
    request: Any
    image_items: list[dict[str, Any]]
    video_items: list[dict[str, Any]]


@dataclass
class _AudioItemCachePlan:
    idx: int
    payload: StagePayload
    state: Qwen3OmniPipelineState
    request: Any
    items: list[dict[str, Any]]


def _visual_item_cache_key(modality: str, cache_key: str | None) -> str | None:
    if cache_key is None:
        return None
    return f"visual_item:{modality}:{cache_key}"


def _audio_item_cache_key(cache_key: str | None) -> str | None:
    if cache_key is None:
        return None
    return f"audio_item:{cache_key}"


def _visual_item_keys(
    request: Any,
    *,
    modality: str,
    rows: int,
) -> list[str | None] | None:
    if rows <= 0:
        return []
    raw_keys = getattr(request, "item_cache_keys", {}).get(modality)
    if raw_keys is None or len(raw_keys) != rows:
        return None
    return [_visual_item_cache_key(modality, key) for key in raw_keys]


def _visual_item_pixel_present(
    request: Any,
    *,
    modality: str,
    rows: int,
) -> list[bool] | None:
    if rows <= 0:
        return []
    raw_mask = getattr(request, "item_pixel_present", {}).get(modality)
    if raw_mask is None:
        return [True] * rows
    if len(raw_mask) != rows:
        return None
    return [bool(item) for item in raw_mask]


def _split_visual_items(
    *,
    model_inputs: dict[str, Any],
    modality: str,
    item_keys: list[str | None],
    pixel_present: list[bool],
    merge: int,
) -> list[dict[str, Any]] | None:
    if modality == "image":
        pixels_key = "pixel_values"
        grid_key = "image_grid_thw"
    elif modality == "video":
        pixels_key = "pixel_values_videos"
        grid_key = "video_grid_thw"
    else:
        raise ValueError(f"Unsupported visual modality: {modality}")

    pixels = model_inputs.get(pixels_key)
    grid = model_inputs.get(grid_key)
    if grid is None:
        return []
    if not isinstance(grid, torch.Tensor) or not isinstance(pixels, torch.Tensor):
        return None
    if len(item_keys) != int(grid.shape[0]):
        return None
    if len(pixel_present) != int(grid.shape[0]):
        return None

    grid_long = grid.to(dtype=torch.long)
    patch_counts = grid_long.prod(dim=-1).tolist()
    present_patch_count = sum(
        int(count)
        for count, has_pixels in zip(patch_counts, pixel_present)
        if has_pixels
    )
    if present_patch_count != int(pixels.shape[0]):
        return None

    items: list[dict[str, Any]] = []
    cursor = 0
    for row, (patch_count, has_pixels) in enumerate(zip(patch_counts, pixel_present)):
        patch_count = int(patch_count)
        end = cursor + patch_count if has_pixels else cursor
        items.append(
            {
                "modality": modality,
                "cache_key": item_keys[row],
                "pixels": pixels[cursor:end] if has_pixels else None,
                "grid": grid[row : row + 1],
                "token_count": patch_count // merge,
                "result": None,
            }
        )
        cursor = end
    return items


def _lookup_visual_item_result(
    *,
    item: dict[str, Any],
    request_id: str,
    cache: StageOutputCache | None,
) -> bool:
    cache_key = item.get("cache_key")
    if cache is None or cache_key is None:
        return False
    cached = cache.get(cache_key)
    if cached is None:
        _trace_encoder_cache(
            IMAGE_STAGE,
            "item_miss",
            request_id=request_id,
            cache_key=cache_key,
            input_bytes=_nested_tensor_bytes(
                {"pixels": item.get("pixels"), "grid": item.get("grid")}
            ),
        )
        return False
    item["result"] = cached
    _trace_encoder_cache(
        IMAGE_STAGE,
        "item_hit",
        request_id=request_id,
        cache_key=cache_key,
        input_bytes=_nested_tensor_bytes(
            {"pixels": item.get("pixels"), "grid": item.get("grid")}
        ),
        output_bytes=_nested_tensor_bytes(cached),
    )
    return True


def _prepare_visual_item_cache_plan(
    *,
    idx: int,
    payload: StagePayload,
    state: Qwen3OmniPipelineState,
    request: Any,
    model: Any,
    cache: StageOutputCache | None,
) -> _VisualItemCachePlan | None:
    if cache is None:
        return None
    model_inputs = request.model_inputs
    image_grid = model_inputs.get("image_grid_thw")
    video_grid = model_inputs.get("video_grid_thw")
    image_rows = int(image_grid.shape[0]) if isinstance(image_grid, torch.Tensor) else 0
    video_rows = int(video_grid.shape[0]) if isinstance(video_grid, torch.Tensor) else 0
    if image_rows == 0 and video_rows == 0:
        return None

    image_keys = _visual_item_keys(request, modality="image", rows=image_rows)
    video_keys = _visual_item_keys(request, modality="video", rows=video_rows)
    if image_keys is None or video_keys is None:
        return None
    image_pixel_present = _visual_item_pixel_present(
        request, modality="image", rows=image_rows
    )
    video_pixel_present = _visual_item_pixel_present(
        request, modality="video", rows=video_rows
    )
    if image_pixel_present is None or video_pixel_present is None:
        return None
    if not any(key is not None for key in (*image_keys, *video_keys)):
        return None

    merge = int(model.spatial_merge_size) ** 2
    image_items = _split_visual_items(
        model_inputs=model_inputs,
        modality="image",
        item_keys=image_keys,
        pixel_present=image_pixel_present,
        merge=merge,
    )
    video_items = _split_visual_items(
        model_inputs=model_inputs,
        modality="video",
        item_keys=video_keys,
        pixel_present=video_pixel_present,
        merge=merge,
    )
    if image_items is None or video_items is None:
        return None

    for item in (*image_items, *video_items):
        item["request_id"] = payload.request_id
        _lookup_visual_item_result(
            item=item, request_id=payload.request_id, cache=cache
        )
        if item["result"] is None and item.get("pixels") is None:
            raise RuntimeError(
                "Visual item payload was omitted but encoder item cache missed "
                f"for {item['modality']} key={item.get('cache_key')!r}. "
                "Disable SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS or "
                "increase the encoder cache capacity."
            )

    return _VisualItemCachePlan(
        idx=idx,
        payload=payload,
        state=state,
        request=request,
        image_items=image_items,
        video_items=video_items,
    )


def _cat_tensors_preserving_cpu_hits(tensors: list[torch.Tensor]) -> torch.Tensor:
    target_device = (
        torch.device("cpu")
        if any(tensor.device.type == "cpu" for tensor in tensors)
        else tensors[0].device
    )
    return torch.cat([tensor.to(device=target_device) for tensor in tensors], dim=0)


def _combine_item_field(
    item_results: list[dict[str, Any]],
    key: str,
) -> torch.Tensor | None:
    tensors = [result.get(key) for result in item_results]
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        return None
    return _cat_tensors_preserving_cpu_hits(tensors)


def _combine_item_multiscale_field(
    item_results: list[dict[str, Any]],
    key: str,
) -> list[torch.Tensor] | None:
    layers_by_item = [result.get(key) for result in item_results]
    if not layers_by_item or any(
        not isinstance(layers, list) for layers in layers_by_item
    ):
        return None
    layer_count = len(layers_by_item[0])
    if any(len(layers) != layer_count for layers in layers_by_item):
        return None
    combined: list[torch.Tensor] = []
    for layer_idx in range(layer_count):
        tensors = [layers[layer_idx] for layers in layers_by_item]
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            return None
        combined.append(_cat_tensors_preserving_cpu_hits(tensors))
    return combined


def _combine_visual_item_results(plan: _VisualItemCachePlan) -> dict[str, Any]:
    result: dict[str, Any] = {}
    image_results = [item["result"] for item in plan.image_items]
    video_results = [item["result"] for item in plan.video_items]
    if image_results:
        result["image_embeds"] = _combine_item_field(image_results, "image_embeds")
        result["image_grid_thw"] = _combine_item_field(image_results, "image_grid_thw")
        result["image_token_counts"] = _combine_item_field(
            image_results,
            "image_token_counts",
        )
        result["deepstack_visual_embeds_image"] = _combine_item_multiscale_field(
            image_results,
            "deepstack_visual_embeds_image",
        )
    if video_results:
        result["video_embeds"] = _combine_item_field(video_results, "video_embeds")
        result["video_grid_thw"] = _combine_item_field(video_results, "video_grid_thw")
        result["video_token_counts"] = _combine_item_field(
            video_results,
            "video_token_counts",
        )
        result["deepstack_visual_embeds_video"] = _combine_item_multiscale_field(
            video_results,
            "deepstack_visual_embeds_video",
        )
    return {key: value for key, value in result.items() if value is not None}


def _item_result_from_combined(
    *,
    modality: str,
    item: dict[str, Any],
    combined: dict[str, Any],
    row_cursor: int,
    token_cursor: int,
) -> tuple[dict[str, Any], int, int]:
    token_count = int(item["token_count"])
    row_end = row_cursor + 1
    token_end = token_cursor + token_count
    if modality == "image":
        result = {
            "image_embeds": _split_visual_features(
                combined.get("image_embeds"),
                start=token_cursor,
                end=token_end,
            ),
            "image_grid_thw": combined.get("image_grid_thw")[row_cursor:row_end],
            "image_token_counts": combined.get("image_token_counts")[
                row_cursor:row_end
            ],
            "deepstack_visual_embeds_image": _split_visual_multiscale(
                combined.get("deepstack_visual_embeds_image"),
                start=token_cursor,
                end=token_end,
            ),
        }
    else:
        result = {
            "video_embeds": _split_visual_features(
                combined.get("video_embeds"),
                start=token_cursor,
                end=token_end,
            ),
            "video_grid_thw": combined.get("video_grid_thw")[row_cursor:row_end],
            "video_token_counts": combined.get("video_token_counts")[
                row_cursor:row_end
            ],
            "deepstack_visual_embeds_video": _split_visual_multiscale(
                combined.get("deepstack_visual_embeds_video"),
                start=token_cursor,
                end=token_end,
            ),
        }
    return _compact_visual_encoder_result(result), row_end, token_end


def _visual_item_batch_stats_enabled() -> bool:
    return _env_bool(_VISUAL_ITEM_BATCH_STATS_ENV, default=False)


def _visual_item_cost(item: dict[str, Any], *, model: Any) -> int:
    hidden = int(model.out_hidden_size)
    output_layers = 1 + int(model.deepstack_layers)
    dtype_bytes = int(model.visual_dtype_bytes)
    token_count = int(item.get("token_count") or 0)
    output_bytes = token_count * hidden * dtype_bytes * output_layers
    return (
        _nested_tensor_bytes({"pixels": item.get("pixels"), "grid": item.get("grid")})
        + output_bytes
    ) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER


def _chunk_visual_items_by_cost(
    items: list[dict[str, Any]],
    *,
    model: Any,
) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    budget = int(QWEN3_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES)
    if budget <= 0:
        return [items]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_cost = 0
    for item in items:
        item_cost = max(_visual_item_cost(item, model=model), 0)
        if current and current_cost + item_cost > budget:
            chunks.append(current)
            current = []
            current_cost = 0
        current.append(item)
        current_cost += item_cost
    if current:
        chunks.append(current)
    return chunks


def _dedup_visual_items_by_cache_key(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    unique_items: list[dict[str, Any]] = []
    leader_by_key: dict[str, dict[str, Any]] = {}
    leader_index_by_key: dict[str, int] = {}
    duplicate_items: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        cache_key = item.get("cache_key")
        if cache_key is None:
            unique_items.append(item)
            continue

        leader = leader_by_key.get(cache_key)
        if leader is None:
            leader_by_key[cache_key] = item
            leader_index_by_key[cache_key] = len(unique_items)
            unique_items.append(item)
            continue

        if leader.get("pixels") is None and item.get("pixels") is not None:
            duplicate_items.setdefault(cache_key, []).append(leader)
            leader_by_key[cache_key] = item
            unique_items[leader_index_by_key[cache_key]] = item
        else:
            duplicate_items.setdefault(cache_key, []).append(item)

    return unique_items, duplicate_items


def _copy_visual_item_results_to_duplicates(
    leaders: list[dict[str, Any]],
    duplicate_items: dict[str, list[dict[str, Any]]],
) -> None:
    if not duplicate_items:
        return
    leader_by_key = {
        item.get("cache_key"): item
        for item in leaders
        if item.get("cache_key") is not None
    }
    for cache_key, duplicates in duplicate_items.items():
        leader = leader_by_key.get(cache_key)
        if leader is None or leader.get("result") is None:
            continue
        for item in duplicates:
            item["result"] = leader["result"]


def _encode_visual_item_batch(
    items: list[dict[str, Any]],
    *,
    model: Any,
    cache: StageOutputCache | None,
) -> None:
    image_pixels: list[torch.Tensor] = []
    image_grids: list[torch.Tensor] = []
    video_pixels: list[torch.Tensor] = []
    video_grids: list[torch.Tensor] = []
    for item in items:
        if item.get("pixels") is None:
            raise RuntimeError(
                "Visual item cache miss cannot be encoded because its "
                f"payload pixels were omitted: {item.get('cache_key')!r}"
            )
        if item["modality"] == "image":
            image_pixels.append(item["pixels"])
            image_grids.append(item["grid"])
        else:
            video_pixels.append(item["pixels"])
            video_grids.append(item["grid"])

    batched_inputs: dict[str, Any] = {}
    if image_pixels:
        batched_inputs["pixel_values"] = torch.cat(image_pixels, dim=0)
        batched_inputs["image_grid_thw"] = torch.cat(image_grids, dim=0)
    if video_pixels:
        batched_inputs["pixel_values_videos"] = torch.cat(video_pixels, dim=0)
        batched_inputs["video_grid_thw"] = torch.cat(video_grids, dim=0)

    with torch.no_grad():
        combined = model(**batched_inputs)

    image_row_cursor = image_token_cursor = 0
    video_row_cursor = video_token_cursor = 0
    for item in items:
        if item["modality"] == "image":
            item_result, image_row_cursor, image_token_cursor = (
                _item_result_from_combined(
                    modality="image",
                    item=item,
                    combined=combined,
                    row_cursor=image_row_cursor,
                    token_cursor=image_token_cursor,
                )
            )
        else:
            item_result, video_row_cursor, video_token_cursor = (
                _item_result_from_combined(
                    modality="video",
                    item=item,
                    combined=combined,
                    row_cursor=video_row_cursor,
                    token_cursor=video_token_cursor,
                )
            )
        item["result"] = item_result
        cache_key = item.get("cache_key")
        if cache is not None and cache_key is not None:
            cache.put(cache_key, item_result)
            _trace_encoder_cache(
                IMAGE_STAGE,
                "item_store",
                request_id=str(item.get("request_id", "item")),
                cache_key=cache_key,
                input_bytes=_nested_tensor_bytes(
                    {"pixels": item.get("pixels"), "grid": item.get("grid")}
                ),
                output_bytes=_nested_tensor_bytes(item_result),
            )
    del combined, batched_inputs
    _empty_image_encoder_cache_if_requested()


def _execute_visual_item_cache_plans(
    plans: list[_VisualItemCachePlan],
    *,
    model: Any,
    cache: StageOutputCache | None,
    results: list[StagePayload | None],
) -> None:
    missing_items = [
        item
        for plan in plans
        for item in (*plan.image_items, *plan.video_items)
        if item.get("result") is None
    ]
    if missing_items:
        raw_missing_count = len(missing_items)
        missing_items, duplicate_items = _dedup_visual_items_by_cache_key(
            missing_items
        )
        duplicate_count = sum(len(items) for items in duplicate_items.values())
        chunks = _chunk_visual_items_by_cost(missing_items, model=model)
        if _visual_item_batch_stats_enabled():
            costs = [
                sum(_visual_item_cost(item, model=model) for item in chunk)
                for chunk in chunks
            ]
            logger.info(
                "visual_item_batch_stats raw_missing_items=%d unique_items=%d duplicate_items=%d chunks=%d chunk_sizes=%s chunk_costs=%s request_budget=%d item_budget=%d",
                raw_missing_count,
                len(missing_items),
                duplicate_count,
                len(chunks),
                [len(chunk) for chunk in chunks],
                costs,
                QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
                QWEN3_IMAGE_ENCODER_ITEM_BATCH_BUDGET_BYTES,
            )
        for chunk in chunks:
            _encode_visual_item_batch(chunk, model=model, cache=cache)
        _copy_visual_item_results_to_duplicates(missing_items, duplicate_items)

    for plan in plans:
        stage_result = _combine_visual_item_results(plan)
        if _store_item_plan_combined_encoder_cache_enabled():
            _store_cached_encoder_output(
                request=plan.request,
                request_id=plan.payload.request_id,
                stage_name=IMAGE_STAGE,
                cache=cache,
                result=stage_result,
            )
        apply_encoder_result(plan.state, stage_name=IMAGE_STAGE, result=stage_result)
        results[plan.idx] = store_state(plan.payload, plan.state)


def _create_image_encoder_request_cost_fn(
    model: Qwen3OmniImageEncoder,
    cache: StageOutputCache | None = None,
):
    merge = int(model.spatial_merge_size) ** 2
    hidden = int(model.out_hidden_size)
    output_layers = 1 + int(model.deepstack_layers)
    dtype_bytes = int(model.visual_dtype_bytes)

    def _full_request_cost(request: Any) -> int:
        model_inputs = request.model_inputs
        raw_bytes = _tensor_bytes(model_inputs.get("pixel_values"))
        raw_bytes += _tensor_bytes(model_inputs.get("pixel_values_videos"))
        visual_tokens = _grid_visual_tokens(model_inputs.get("image_grid_thw"), merge)
        visual_tokens += _grid_visual_tokens(
            model_inputs.get("video_grid_thw"),
            merge,
        )
        output_bytes = visual_tokens * hidden * dtype_bytes * output_layers
        return (raw_bytes + output_bytes) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER

    def _cache_has(cache_key: str | None) -> bool:
        return cache is not None and cache.contains(cache_key)

    def _item_cache_miss_cost(request: Any) -> int | None:
        if cache is None:
            return None
        model_inputs = request.model_inputs
        image_grid = model_inputs.get("image_grid_thw")
        video_grid = model_inputs.get("video_grid_thw")
        image_rows = (
            int(image_grid.shape[0]) if isinstance(image_grid, torch.Tensor) else 0
        )
        video_rows = (
            int(video_grid.shape[0]) if isinstance(video_grid, torch.Tensor) else 0
        )
        if image_rows == 0 and video_rows == 0:
            return None

        image_keys = _visual_item_keys(request, modality="image", rows=image_rows)
        video_keys = _visual_item_keys(request, modality="video", rows=video_rows)
        if image_keys is None or video_keys is None:
            return None
        if not any(key is not None for key in (*image_keys, *video_keys)):
            return None

        image_pixel_present = _visual_item_pixel_present(
            request, modality="image", rows=image_rows
        )
        video_pixel_present = _visual_item_pixel_present(
            request, modality="video", rows=video_rows
        )
        if image_pixel_present is None or video_pixel_present is None:
            return None

        image_items = _split_visual_items(
            model_inputs=model_inputs,
            modality="image",
            item_keys=image_keys,
            pixel_present=image_pixel_present,
            merge=merge,
        )
        video_items = _split_visual_items(
            model_inputs=model_inputs,
            modality="video",
            item_keys=video_keys,
            pixel_present=video_pixel_present,
            merge=merge,
        )
        if image_items is None or video_items is None:
            return None

        cost = 0
        for item in (*image_items, *video_items):
            cache_key = item.get("cache_key")
            if cache_key is not None and _cache_has(cache_key):
                continue
            if item.get("pixels") is None:
                continue
            cost += _visual_item_cost(item, model=model)
        return cost

    def _cost(payload: StagePayload) -> int:
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)
        if request.skip_result is not None:
            return 0
        miss_cost = _item_cache_miss_cost(request)
        if miss_cost is not None:
            return miss_cost
        return _full_request_cost(request)

    return _cost


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_bytes(value)
    if isinstance(value, dict):
        return sum(_nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensor_bytes(item) for item in value)
    return 0


def _encoder_cache_trace_enabled() -> bool:
    return _env_bool("SGLANG_OMNI_TRACE_ENCODER_CACHE", default=False)


def _store_item_plan_combined_encoder_cache_enabled() -> bool:
    # Item-plan requests already store each image/video/audio item separately.
    # Caching every combined prefix as well is actively harmful for realtime
    # chunk workloads: trunk=1..N creates many large combined entries that evict
    # the item outputs needed by the final request.
    return _env_bool(_STORE_ITEM_PLAN_COMBINED_CACHE_ENV, default=False)


def _short_cache_key(cache_key: str | None) -> str:
    if not cache_key:
        return "-"
    if len(cache_key) <= 32:
        return cache_key
    return f"{cache_key[:16]}...{cache_key[-8:]}"


def _trace_encoder_cache(
    stage_name: str,
    action: str,
    *,
    request_id: str,
    cache_key: str | None,
    input_bytes: int | None = None,
    output_bytes: int | None = None,
    detail: str | None = None,
) -> None:
    if not _encoder_cache_trace_enabled():
        return
    parts = [
        f"stage={stage_name}",
        f"action={action}",
        f"req={request_id}",
        f"key={_short_cache_key(cache_key)}",
    ]
    if input_bytes is not None:
        parts.append(f"input_bytes={input_bytes}")
    if output_bytes is not None:
        parts.append(f"output_bytes={output_bytes}")
    if detail:
        parts.append(detail)
    logger.info("encoder_cache %s", " ".join(parts))


def _lookup_cached_encoder_output(
    *,
    request: Any,
    request_id: str,
    stage_name: str,
    cache: StageOutputCache | None,
) -> Any | None:
    if cache is None or request.cache_key is None:
        return None
    cached = cache.get(request.cache_key)
    if cached is None:
        _trace_encoder_cache(
            stage_name,
            "miss",
            request_id=request_id,
            cache_key=request.cache_key,
            input_bytes=_nested_tensor_bytes(request.model_inputs),
        )
        return None
    _trace_encoder_cache(
        stage_name,
        "hit",
        request_id=request_id,
        cache_key=request.cache_key,
        input_bytes=_nested_tensor_bytes(request.model_inputs),
        output_bytes=_nested_tensor_bytes(cached),
    )
    return cached


def _store_cached_encoder_output(
    *,
    request: Any,
    request_id: str,
    stage_name: str,
    cache: StageOutputCache | None,
    result: Any,
) -> None:
    if cache is None or request.cache_key is None:
        return
    cache.put(request.cache_key, result)
    _trace_encoder_cache(
        stage_name,
        "store",
        request_id=request_id,
        cache_key=request.cache_key,
        input_bytes=_nested_tensor_bytes(request.model_inputs),
        output_bytes=_nested_tensor_bytes(result),
    )


def _grid_visual_tokens(grid: Any, merge: int) -> int:
    if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
        return 0
    return int((grid.to(dtype=torch.long).prod(dim=-1) // merge).sum().item())


def _batch_image_encoder_payloads(
    payloads: list[StagePayload],
    *,
    model: Any,
    cache: StageOutputCache | None = None,
) -> list[StagePayload]:
    results: list[StagePayload | None] = [None] * len(payloads)
    active: list[tuple[int, StagePayload, Any, Any]] = []
    item_plans: list[_VisualItemCachePlan] = []
    duplicate_waiters: dict[str, list[tuple[int, StagePayload, Any]]] = {}
    active_cache_keys: set[str] = set()
    active_cache_leaders: dict[str, str] = {}

    for idx, payload in enumerate(payloads):
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)
        if request.skip_result is not None:
            results[idx] = _run_single_encoder_payload(
                payload,
                stage_name=IMAGE_STAGE,
                model=model,
                cache=cache,
            )
            continue

        cached = _lookup_cached_encoder_output(
            request=request,
            request_id=payload.request_id,
            stage_name=IMAGE_STAGE,
            cache=cache,
        )
        if cached is not None:
            apply_encoder_result(state, stage_name=IMAGE_STAGE, result=cached)
            results[idx] = store_state(payload, state)
            continue

        if not _image_request_is_batchable(request):
            results[idx] = _run_single_encoder_payload(
                payload,
                stage_name=IMAGE_STAGE,
                model=model,
                cache=cache,
            )
            continue

        plan = _prepare_visual_item_cache_plan(
            idx=idx,
            payload=payload,
            state=state,
            request=request,
            model=model,
            cache=cache,
        )
        if plan is not None:
            item_plans.append(plan)
            continue

        cache_key = request.cache_key
        if cache_key is not None and cache_key in active_cache_keys:
            duplicate_waiters.setdefault(cache_key, []).append((idx, payload, state))
            _trace_encoder_cache(
                IMAGE_STAGE,
                "dedup_same_batch",
                request_id=payload.request_id,
                cache_key=cache_key,
                input_bytes=_nested_tensor_bytes(request.model_inputs),
                detail=f"leader={active_cache_leaders[cache_key]}",
            )
            continue

        active.append((idx, payload, state, request))
        if cache_key is not None:
            active_cache_keys.add(cache_key)
            active_cache_leaders[cache_key] = payload.request_id

    if not active and not item_plans:
        return [result for result in results if result is not None]

    if active:
        image_pixels: list[torch.Tensor] = []
        image_grids: list[torch.Tensor] = []
        video_pixels: list[torch.Tensor] = []
        video_grids: list[torch.Tensor] = []
        metas: list[dict[str, Any]] = []
        merge = model.spatial_merge_size**2

        for idx, payload, state, request in active:
            input_dict = request.model_inputs
            image_grid = input_dict.get("image_grid_thw")
            video_grid = input_dict.get("video_grid_thw")
            image_rows = (
                int(image_grid.shape[0]) if isinstance(image_grid, torch.Tensor) else 0
            )
            video_rows = (
                int(video_grid.shape[0]) if isinstance(video_grid, torch.Tensor) else 0
            )
            image_token_counts = (
                (image_grid.prod(-1) // merge).to(dtype=torch.long)
                if isinstance(image_grid, torch.Tensor)
                else None
            )
            video_token_counts = (
                (video_grid.prod(-1) // merge).to(dtype=torch.long)
                if isinstance(video_grid, torch.Tensor)
                else None
            )
            image_token_total = (
                int(image_token_counts.sum().item())
                if isinstance(image_token_counts, torch.Tensor)
                else 0
            )
            video_token_total = (
                int(video_token_counts.sum().item())
                if isinstance(video_token_counts, torch.Tensor)
                else 0
            )
            if isinstance(input_dict.get("pixel_values"), torch.Tensor):
                image_pixels.append(input_dict["pixel_values"])
                image_grids.append(image_grid)
            if isinstance(input_dict.get("pixel_values_videos"), torch.Tensor):
                video_pixels.append(input_dict["pixel_values_videos"])
                video_grids.append(video_grid)
            metas.append(
                {
                    "idx": idx,
                    "payload": payload,
                    "state": state,
                    "request": request,
                    "image_rows": image_rows,
                    "video_rows": video_rows,
                    "image_token_total": image_token_total,
                    "video_token_total": video_token_total,
                }
            )

        batched_inputs: dict[str, Any] = {}
        if image_pixels:
            batched_inputs["pixel_values"] = torch.cat(image_pixels, dim=0)
            batched_inputs["image_grid_thw"] = torch.cat(image_grids, dim=0)
        if video_pixels:
            batched_inputs["pixel_values_videos"] = torch.cat(video_pixels, dim=0)
            batched_inputs["video_grid_thw"] = torch.cat(video_grids, dim=0)

        with torch.no_grad():
            combined = model(**batched_inputs)

        image_grid_all = combined.get("image_grid_thw")
        image_counts_all = combined.get("image_token_counts")
        image_embeds_all = combined.get("image_embeds")
        image_multiscale_all = combined.get("deepstack_visual_embeds_image")
        video_grid_all = combined.get("video_grid_thw")
        video_counts_all = combined.get("video_token_counts")
        video_embeds_all = combined.get("video_embeds")
        video_multiscale_all = combined.get("deepstack_visual_embeds_video")

        image_row_cursor = 0
        image_token_cursor = 0
        video_row_cursor = 0
        video_token_cursor = 0
        computed_by_cache_key: dict[str, dict[str, Any]] = {}
        for meta in metas:
            stage_result: dict[str, Any] = {}
            if meta["image_rows"] > 0:
                row_end = image_row_cursor + meta["image_rows"]
                token_end = image_token_cursor + meta["image_token_total"]
                stage_result["image_embeds"] = _split_visual_features(
                    image_embeds_all, start=image_token_cursor, end=token_end
                )
                stage_result["image_grid_thw"] = image_grid_all[
                    image_row_cursor:row_end
                ]
                stage_result["image_token_counts"] = image_counts_all[
                    image_row_cursor:row_end
                ]
                stage_result["deepstack_visual_embeds_image"] = (
                    _split_visual_multiscale(
                        image_multiscale_all,
                        start=image_token_cursor,
                        end=token_end,
                    )
                )
                image_row_cursor = row_end
                image_token_cursor = token_end
            if meta["video_rows"] > 0:
                row_end = video_row_cursor + meta["video_rows"]
                token_end = video_token_cursor + meta["video_token_total"]
                stage_result["video_embeds"] = _split_visual_features(
                    video_embeds_all, start=video_token_cursor, end=token_end
                )
                stage_result["video_grid_thw"] = video_grid_all[
                    video_row_cursor:row_end
                ]
                stage_result["video_token_counts"] = video_counts_all[
                    video_row_cursor:row_end
                ]
                stage_result["deepstack_visual_embeds_video"] = (
                    _split_visual_multiscale(
                        video_multiscale_all,
                        start=video_token_cursor,
                        end=token_end,
                    )
                )
                video_row_cursor = row_end
                video_token_cursor = token_end
            request = meta["request"]
            stage_result = _compact_visual_encoder_result(stage_result)
            _store_cached_encoder_output(
                request=request,
                request_id=meta["payload"].request_id,
                stage_name=IMAGE_STAGE,
                cache=cache,
                result=stage_result,
            )
            if request.cache_key is not None:
                computed_by_cache_key[request.cache_key] = stage_result
            apply_encoder_result(
                meta["state"],
                stage_name=IMAGE_STAGE,
                result=stage_result,
            )
            results[meta["idx"]] = store_state(meta["payload"], meta["state"])

        for cache_key, waiters in duplicate_waiters.items():
            stage_result = computed_by_cache_key.get(cache_key)
            if stage_result is None:
                continue
            for idx, payload, state in waiters:
                apply_encoder_result(state, stage_name=IMAGE_STAGE, result=stage_result)
                results[idx] = store_state(payload, state)
        del (
            combined,
            batched_inputs,
            image_grid_all,
            image_counts_all,
            image_embeds_all,
            image_multiscale_all,
            video_grid_all,
            video_counts_all,
            video_embeds_all,
            video_multiscale_all,
        )
        _empty_image_encoder_cache_if_requested()

    if item_plans:
        _execute_visual_item_cache_plans(
            item_plans,
            model=model,
            cache=cache,
            results=results,
        )

    return [result for result in results if result is not None]


def _audio_request_is_batchable(request: Any) -> bool:
    if request.skip_result is not None:
        return False
    input_dict = request.model_inputs
    features = input_dict.get("input_features")
    if not isinstance(features, torch.Tensor):
        return False
    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    return (lengths is None or isinstance(lengths, torch.Tensor)) and (
        mask is None or isinstance(mask, torch.Tensor)
    )


def _normalize_audio_request_tensors(
    request: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_dict = request.model_inputs
    features = input_dict["input_features"]
    if features.ndim == 2:
        features = features.unsqueeze(0)

    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(dtype=torch.long).view(-1)
    elif isinstance(mask, torch.Tensor):
        lengths = mask.to(dtype=torch.long).sum(dim=1).view(-1)
    else:
        raise ValueError("audio_feature_lengths or feature_attention_mask is required")

    time_dim = features.shape[-1]
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
    else:
        steps = torch.arange(time_dim, dtype=torch.long).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)

    return features, mask, lengths


def _pad_audio_features(features: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(features.shape[-1])
    if pad <= 0:
        return features
    return F.pad(features, (0, pad))


def _pad_audio_mask(mask: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(mask.shape[-1])
    if pad <= 0:
        return mask
    return F.pad(mask, (0, pad), value=False)


def _audio_item_keys(request: Any, rows: int) -> list[str | None] | None:
    if rows <= 0:
        return []
    raw_keys = getattr(request, "item_cache_keys", {}).get("audio")
    if raw_keys is None or len(raw_keys) != rows:
        return None
    return [_audio_item_cache_key(key) for key in raw_keys]


def _lookup_audio_item_result(
    *,
    item: dict[str, Any],
    request_id: str,
    cache: StageOutputCache | None,
) -> None:
    cache_key = item.get("cache_key")
    if cache is None or cache_key is None:
        return
    cached = cache.get(cache_key)
    if cached is None:
        _trace_encoder_cache(
            AUDIO_STAGE,
            "item_miss",
            request_id=request_id,
            cache_key=cache_key,
            input_bytes=_nested_tensor_bytes(
                {
                    "features": item.get("features"),
                    "mask": item.get("mask"),
                    "length": item.get("length"),
                }
            ),
        )
        return
    item["result"] = cached
    _trace_encoder_cache(
        AUDIO_STAGE,
        "item_hit",
        request_id=request_id,
        cache_key=cache_key,
        input_bytes=_nested_tensor_bytes(
            {
                "features": item.get("features"),
                "mask": item.get("mask"),
                "length": item.get("length"),
            }
        ),
        output_bytes=_nested_tensor_bytes(cached),
    )


def _prepare_audio_item_cache_plan(
    *,
    idx: int,
    payload: StagePayload,
    state: Qwen3OmniPipelineState,
    request: Any,
    cache: StageOutputCache | None,
) -> _AudioItemCachePlan | None:
    if cache is None or not _audio_request_is_batchable(request):
        return None

    features, mask, lengths = _normalize_audio_request_tensors(request)
    rows = int(lengths.shape[0])
    item_keys = _audio_item_keys(request, rows)
    if item_keys is None or not any(key is not None for key in item_keys):
        return None

    items: list[dict[str, Any]] = []
    for row, cache_key in enumerate(item_keys):
        length = lengths[row : row + 1]
        item = {
            "cache_key": cache_key,
            "features": features[row : row + 1],
            "mask": mask[row : row + 1],
            "length": length,
            "result": None,
            "request_id": payload.request_id,
        }
        _lookup_audio_item_result(
            item=item,
            request_id=payload.request_id,
            cache=cache,
        )
        items.append(item)

    return _AudioItemCachePlan(
        idx=idx,
        payload=payload,
        state=state,
        request=request,
        items=items,
    )


def _combine_audio_item_results(plan: _AudioItemCachePlan) -> dict[str, Any]:
    item_results = [item["result"] for item in plan.items]
    embeds = _combine_item_field(item_results, "audio_embeds")
    feature_lengths = _combine_item_field(item_results, "audio_feature_lengths")
    output_lengths = _combine_item_field(item_results, "audio_output_lengths")
    result: dict[str, Any] = {}
    if embeds is not None:
        result["audio_embeds"] = embeds
    if feature_lengths is not None:
        result["audio_feature_lengths"] = feature_lengths
    if output_lengths is not None:
        result["audio_output_lengths"] = output_lengths
    return result


def _execute_audio_item_cache_plans(
    plans: list[_AudioItemCachePlan],
    *,
    model: Any,
    cache: StageOutputCache | None,
    results: list[StagePayload | None],
) -> None:
    missing_items = [
        item for plan in plans for item in plan.items if item.get("result") is None
    ]
    if missing_items:
        max_time = max(int(item["features"].shape[-1]) for item in missing_items)
        batched_features = torch.cat(
            [
                _pad_audio_features(item["features"], max_time)
                for item in missing_items
            ],
            dim=0,
        )
        batched_mask = torch.cat(
            [_pad_audio_mask(item["mask"], max_time) for item in missing_items],
            dim=0,
        )
        batched_lengths = torch.cat([item["length"] for item in missing_items], dim=0)

        with torch.no_grad():
            combined = model(
                input_features=batched_features,
                feature_attention_mask=batched_mask,
                audio_feature_lengths=batched_lengths,
            )

        output_lengths = combined["audio_output_lengths"]
        embeds = combined["audio_embeds"]
        token_cursor = 0
        for row, item in enumerate(missing_items):
            output_length = output_lengths[row : row + 1]
            token_end = token_cursor + int(output_length.sum().item())
            item_result = _compact_audio_encoder_result(
                {
                    "audio_embeds": embeds[token_cursor:token_end],
                    "audio_feature_lengths": combined["audio_feature_lengths"][
                        row : row + 1
                    ],
                    "audio_output_lengths": output_length,
                }
            )
            item["result"] = item_result
            token_cursor = token_end
            cache_key = item.get("cache_key")
            if cache is not None and cache_key is not None:
                cache.put(cache_key, item_result)
                _trace_encoder_cache(
                    AUDIO_STAGE,
                    "item_store",
                    request_id=str(item.get("request_id", "item")),
                    cache_key=cache_key,
                    input_bytes=_nested_tensor_bytes(
                        {
                            "features": item.get("features"),
                            "mask": item.get("mask"),
                            "length": item.get("length"),
                        }
                    ),
                    output_bytes=_nested_tensor_bytes(item_result),
                )

    for plan in plans:
        stage_result = _combine_audio_item_results(plan)
        if _store_item_plan_combined_encoder_cache_enabled():
            _store_cached_encoder_output(
                request=plan.request,
                request_id=plan.payload.request_id,
                stage_name=AUDIO_STAGE,
                cache=cache,
                result=stage_result,
            )
        apply_encoder_result(plan.state, stage_name=AUDIO_STAGE, result=stage_result)
        results[plan.idx] = store_state(plan.payload, plan.state)


def _batch_audio_encoder_payloads(
    payloads: list[StagePayload],
    *,
    model: Any,
    cache: StageOutputCache | None = None,
) -> list[StagePayload]:
    results: list[StagePayload | None] = [None] * len(payloads)
    active: list[tuple[int, StagePayload, Any, Any]] = []
    item_plans: list[_AudioItemCachePlan] = []

    for idx, payload in enumerate(payloads):
        state = load_state(payload)
        request = build_encoder_request(state, stage_name=AUDIO_STAGE)
        if request.skip_result is not None:
            results[idx] = _run_single_encoder_payload(
                payload,
                stage_name=AUDIO_STAGE,
                model=model,
                cache=cache,
            )
            continue

        cached = _lookup_cached_encoder_output(
            request=request,
            request_id=payload.request_id,
            stage_name=AUDIO_STAGE,
            cache=cache,
        )
        if cached is not None:
            apply_encoder_result(state, stage_name=AUDIO_STAGE, result=cached)
            results[idx] = store_state(payload, state)
            continue

        if not _audio_request_is_batchable(request):
            results[idx] = _run_single_encoder_payload(
                payload,
                stage_name=AUDIO_STAGE,
                model=model,
                cache=cache,
            )
            continue

        plan = _prepare_audio_item_cache_plan(
            idx=idx,
            payload=payload,
            state=state,
            request=request,
            cache=cache,
        )
        if plan is not None:
            item_plans.append(plan)
            continue

        active.append((idx, payload, state, request))

    if not active and not item_plans:
        return [result for result in results if result is not None]

    if active:
        normalized = []
        max_time = 0
        for idx, payload, state, request in active:
            features, mask, lengths = _normalize_audio_request_tensors(request)
            max_time = max(max_time, int(features.shape[-1]))
            normalized.append(
                {
                    "idx": idx,
                    "payload": payload,
                    "state": state,
                    "features": features,
                    "mask": mask,
                    "lengths": lengths,
                    "count": int(lengths.shape[0]),
                    "request": request,
                }
            )

        batched_features = torch.cat(
            [_pad_audio_features(item["features"], max_time) for item in normalized],
            dim=0,
        )
        batched_mask = torch.cat(
            [_pad_audio_mask(item["mask"], max_time) for item in normalized], dim=0
        )
        batched_lengths = torch.cat([item["lengths"] for item in normalized], dim=0)

        with torch.no_grad():
            combined = model(
                input_features=batched_features,
                feature_attention_mask=batched_mask,
                audio_feature_lengths=batched_lengths,
            )

        output_lengths = combined["audio_output_lengths"]
        embeds = combined["audio_embeds"]
        row_cursor = 0
        token_cursor = 0
        for item in normalized:
            row_end = row_cursor + item["count"]
            req_output_lengths = output_lengths[row_cursor:row_end]
            token_end = token_cursor + int(req_output_lengths.sum().item())
            stage_result = _compact_audio_encoder_result(
                {
                    "audio_embeds": embeds[token_cursor:token_end],
                    "audio_feature_lengths": combined["audio_feature_lengths"][
                        row_cursor:row_end
                    ],
                    "audio_output_lengths": req_output_lengths,
                }
            )
            _store_cached_encoder_output(
                request=item["request"],
                request_id=item["payload"].request_id,
                stage_name=AUDIO_STAGE,
                cache=cache,
                result=stage_result,
            )
            apply_encoder_result(
                item["state"],
                stage_name=AUDIO_STAGE,
                result=stage_result,
            )
            results[item["idx"]] = store_state(item["payload"], item["state"])
            row_cursor = row_end
            token_cursor = token_end

    if item_plans:
        _execute_audio_item_cache_plans(
            item_plans,
            model=model,
            cache=cache,
            results=results,
        )

    return [result for result in results if result is not None]


# ---------------------------------------------------------------------------
# Simple stages — return SimpleScheduler
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_path: str,
    *,
    thinker_max_seq_len: int | None = None,
    video_fps: float | None = None,
    video_max_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    video_total_pixels: int | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = Qwen3OmniPreprocessor(
        model_path=model_path,
        max_seq_len=thinker_max_seq_len,
        video_fps=video_fps,
        video_max_frames=video_max_frames,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_total_pixels=video_total_pixels,
    )

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(_identity)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen3OmniImageEncoder(model_path=model_path, device=device, dtype=dtype)
    cache = StageOutputCache(
        max_size=QWEN3_ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=QWEN3_ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode(payload: StagePayload) -> StagePayload:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="encoder_start",
            metadata={"modality": "image", "batch_size": 1},
        )
        try:
            return _run_single_encoder_payload(
                payload,
                stage_name=IMAGE_STAGE,
                model=model,
                cache=cache,
            )
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="encoder_end",
                metadata={"modality": "image", "batch_size": 1},
            )

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        for p in payloads:
            _emit_event(
                request_id=p.request_id,
                stage=None,
                event_name="encoder_start",
                metadata={"modality": "image", "batch_size": len(payloads)},
            )
        try:
            return _batch_image_encoder_payloads(
                payloads,
                model=model,
                cache=cache,
            )
        finally:
            for p in payloads:
                _emit_event(
                    request_id=p.request_id,
                    stage=None,
                    event_name="encoder_end",
                    metadata={"modality": "image", "batch_size": len(payloads)},
                )

    # Preserve the calibrated image-encoder batching shape and add a small
    # batch_wait so video benchmarks at concurrency=16 batch together.
    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=32,
        max_batch_wait_ms=50,
        request_cost_fn=_create_image_encoder_request_cost_fn(model, cache),
        max_batch_cost=QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
    )


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)
    cache = StageOutputCache(
        max_size=QWEN3_ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=QWEN3_ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode(payload: StagePayload) -> StagePayload:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="encoder_start",
            metadata={"modality": "audio", "batch_size": 1},
        )
        try:
            return _run_single_encoder_payload(
                payload,
                stage_name=AUDIO_STAGE,
                model=model,
                cache=cache,
            )
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="encoder_end",
                metadata={"modality": "audio", "batch_size": 1},
            )

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        for p in payloads:
            _emit_event(
                request_id=p.request_id,
                stage=None,
                event_name="encoder_start",
                metadata={"modality": "audio", "batch_size": len(payloads)},
            )
        try:
            return _batch_audio_encoder_payloads(
                payloads,
                model=model,
                cache=cache,
            )
        finally:
            for p in payloads:
                _emit_event(
                    request_id=p.request_id,
                    stage=None,
                    event_name="encoder_end",
                    metadata={"modality": "audio", "batch_size": len(payloads)},
                )

    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=32,
        max_batch_wait_ms=50,
    )


def create_decode_executor(model_path: str):
    return create_streaming_detokenize_scheduler(model_path)


# ---------------------------------------------------------------------------
# AR stages — return OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    encoder_mem_reserve: float = 0.05,
    speech_enabled: bool = False,
    total_gpu_memory_fraction: float | None = None,
):
    """Returns OmniScheduler for thinker."""

    overrides: dict[str, Any] = {"disable_cuda_graph": False}
    if server_args_overrides:
        overrides.update(server_args_overrides)
    overrides["tp_size"] = tp_size
    has_explicit_colocated_mem_fraction = (
        total_gpu_memory_fraction is not None
        and overrides.get("mem_fraction_static") is not None
    )
    colocated_encoder_mem_reserve = (
        encoder_mem_reserve
        if total_gpu_memory_fraction is not None
        and not has_explicit_colocated_mem_fraction
        else 0.0
    )
    memory_contract = _apply_colocated_ar_memory_contract(
        overrides,
        stage_name="thinker",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        encoder_mem_reserve=colocated_encoder_mem_reserve,
    )
    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    if total_gpu_memory_fraction is None:
        encoder_reserve_applied = _apply_qwen_thinker_encoder_reserve(
            server_args,
            has_explicit_mem_fraction_static=(
                memory_contract.mem_fraction_static_pinned
            ),
            encoder_mem_reserve=encoder_mem_reserve,
        )
        effective_total_gpu_memory_fraction = total_gpu_memory_fraction
        applied_encoder_reserve = (
            encoder_mem_reserve if encoder_reserve_applied else 0.0
        )
    else:
        effective_total_gpu_memory_fraction = (
            memory_contract.effective_total_gpu_memory_fraction
        )
        applied_encoder_reserve = memory_contract.applied_encoder_mem_reserve

    pre_load_avail_mem = avail_gpu_mem(gpu_id)
    pre_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_startup stage=thinker gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={thinker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"effective_total_gpu_memory_fraction={effective_total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"encoder_mem_reserve={applied_encoder_reserve} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
    )
    scheduler = create_thinker_scheduler(
        server_args,
        gpu_id,
        speech_enabled=speech_enabled,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
    )
    post_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_started stage=thinker gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={thinker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"effective_total_gpu_memory_fraction={effective_total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
        f" post_load_process_mem={format_bytes_gib(post_load_process_mem)}"
    )
    return scheduler


def create_talker_ar_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    talker_max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    weight_prefix: str = "talker.",
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = 5,
):
    """Returns OmniScheduler for talker."""
    from sglang_omni.models.qwen3_omni.bootstrap import create_talker_scheduler

    # Note (Xuesong, Chenyang): cuda_graph defaults to ON for the talker
    # after #384, which routed talker MoE through `self.experts` (FusedMoE)
    # — the `fused_experts (full graph)` backend picked in #344. Caller can
    # override via factory_args or the `--talker-cuda-graph off` CLI flag.
    # Note (Xuesong): pytorch backend works around an sglang upstream gap —
    # Sampler.forward doesn't forward seed to flashinfer, so
    # under cuda graph the captured RNG is boot-dependent and ~5% of prompts
    # trigger degenerate AR loops (see #408). Revert once upstream lands.
    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "sampling_backend": "pytorch",
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    overrides["tp_size"] = tp_size
    _apply_colocated_ar_memory_contract(
        overrides,
        stage_name="talker_ar",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    server_args = build_sglang_server_args(
        model_path,
        context_length=talker_max_seq_len,
        **overrides,
    )
    pre_load_avail_mem = avail_gpu_mem(gpu_id)
    pre_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_startup stage=talker_ar gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={talker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
    )
    scheduler = create_talker_scheduler(
        server_args,
        gpu_id,
        weight_prefix=weight_prefix,
        speech_enabled=speech_enabled,
        feedback_enabled=feedback_enabled,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
    )
    post_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_started stage=talker_ar gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={talker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
        f" post_load_process_mem={format_bytes_gib(post_load_process_mem)}"
    )
    return scheduler
