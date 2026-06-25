# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3.5-Omni pipelines."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

from sglang_omni.models.qwen3_5_omni.bootstrap import create_thinker_scheduler
from sglang_omni.models.qwen3_5_omni.components.preprocessor import (
    Qwen35OmniPreprocessor,
)
from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
    QWEN3_5_OMNI_MAX_PREFILL_TOKENS,
    QWEN3_5_OMNI_TALKER_MAX_SEQ_LEN,
    QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
)
from sglang_omni.models.qwen3_omni.stages import (
    _apply_colocated_ar_memory_contract,
    _apply_qwen_thinker_encoder_reserve,
    _batch_audio_encoder_payloads,
    _batch_image_encoder_payloads,
    _create_image_encoder_request_cost_fn,
    _emit_event,
    _run_single_encoder_payload,
    create_aggregate_executor as _create_aggregate_executor,
    create_decode_executor as _create_decode_executor,
    QWEN3_ENCODER_CACHE_MAX_BYTES,
    QWEN3_ENCODER_CACHE_MAX_ENTRIES,
    QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import build_sglang_server_args
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes
from sglang_omni.utils.misc import avail_gpu_mem

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"
TALKER_STAGE = "talker_ar"
PREPROCESSING_STAGE = "preprocessing"

logger = logging.getLogger(__name__)

_ENCODER_MAX_BATCH_WAIT_MS_ENV = "SGLANG_OMNI_ENCODER_MAX_BATCH_WAIT_MS"


_ENCODER_IMPL_CANDIDATES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    IMAGE_STAGE: (
        (
            "sglang_omni.models.qwen3_5_omni.components.image_encoder",
            ("Qwen35OmniImageEncoder", "Qwen3OmniNextImageEncoder"),
        ),
    ),
    AUDIO_STAGE: (
        (
            "sglang_omni.models.qwen3_5_omni.components.audio_encoder",
            ("Qwen35OmniAudioEncoder", "Qwen3OmniNextAudioEncoder"),
        ),
    ),
}

_STAGE_MODEL_SUBDIRS = {
    PREPROCESSING_STAGE: ("thinker",),
    IMAGE_STAGE: ("thinker",),
    AUDIO_STAGE: ("thinker",),
    THINKER_STAGE: ("thinker",),
    # Qwen3.5 server and RTC scripts use talker_lm by default, while some
    # offline scripts and older checkpoints use talker. Keep that priority when
    # auto-detecting split subdirectories.
    TALKER_STAGE: ("talker_lm", "talker"),
}


def _encoder_max_batch_wait_ms() -> int:
    value = os.getenv(_ENCODER_MAX_BATCH_WAIT_MS_ENV)
    if value is None:
        return 50
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default 50ms",
            _ENCODER_MAX_BATCH_WAIT_MS_ENV,
            value,
        )
        return 50


def _resolve_qwen35_stage_model_path(model_path: str, stage_name: str) -> str:
    subdirs = _STAGE_MODEL_SUBDIRS.get(stage_name)
    if not subdirs:
        return model_path
    for subdir in subdirs:
        candidate = Path(model_path) / subdir
        # Qwen3.5 bring-up scripts can auto-select root/thinker or
        # root/talker_lm from a root checkpoint. Require config.json in the
        # subdirectory so HuggingFace repo ids or plain root checkpoints are not
        # mistaken for local split checkpoints.
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return str(candidate)
    return model_path


def _load_qwen35_encoder(
    stage_name: str,
    *,
    model_path: str,
    device: str,
    dtype: str | None,
) -> Any | None:
    candidates = _ENCODER_IMPL_CANDIDATES.get(stage_name)
    if not candidates:
        return None

    import_errors: list[str] = []
    for module_name, class_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
        for class_name in class_names:
            model_cls = getattr(module, class_name, None)
            if model_cls is None:
                continue
            logger.info(
                "loading Qwen3.5-Omni %s via %s.%s",
                stage_name,
                module_name,
                class_name,
            )
            return model_cls(model_path=model_path, device=device, dtype=dtype)
    detail = "; ".join(import_errors) if import_errors else "no candidate class found"
    raise ImportError(
        f"Qwen3.5-Omni {stage_name} encoder implementation is unavailable: {detail}"
    )


def create_preprocessing_executor(
    model_path: str,
    *,
    thinker_max_seq_len: int | None = None,
    max_concurrency: int = 1,
    limit_mm_per_prompt: dict[str, int] | None = None,
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
    audio_timestamp_interval: int | None = None,
    audio_downsample_times: int | None = None,
    audio_downsample_chunk_size: int | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    resolved_model_path = _resolve_qwen35_stage_model_path(
        model_path,
        PREPROCESSING_STAGE,
    )
    preprocessor = Qwen35OmniPreprocessor(
        model_path=resolved_model_path,
        max_seq_len=thinker_max_seq_len,
        limit_mm_per_prompt=limit_mm_per_prompt,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        video_fps=video_fps,
        video_max_frames=video_max_frames,
        video_min_frames=video_min_frames,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_total_pixels=video_total_pixels,
        video_override_max_pixels=video_override_max_pixels,
        video_seconds_per_chunk=video_seconds_per_chunk,
        video_position_id_per_seconds=video_position_id_per_seconds,
        audio_target_sr=audio_target_sr,
        audio_timestamp_interval=audio_timestamp_interval,
        audio_downsample_times=audio_downsample_times,
        audio_downsample_chunk_size=audio_downsample_chunk_size,
    )

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_aggregate_executor():
    return _create_aggregate_executor()


def _create_encoder_executor(stage_name: str, model: Any):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

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
            metadata={"modality": stage_name.split("_", 1)[0], "batch_size": 1},
        )
        try:
            return _run_single_encoder_payload(
                payload,
                stage_name=stage_name,
                model=model,
                cache=cache,
            )
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="encoder_end",
                metadata={"modality": stage_name.split("_", 1)[0], "batch_size": 1},
            )

    return SimpleScheduler(_encode)


def _create_batched_encoder_executor(stage_name: str, model: Any):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    cache = StageOutputCache(
        max_size=QWEN3_ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=QWEN3_ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )
    modality = stage_name.split("_", 1)[0]
    batch_fn = (
        _batch_image_encoder_payloads
        if stage_name == IMAGE_STAGE
        else _batch_audio_encoder_payloads
    )

    def _encode(payload: StagePayload) -> StagePayload:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="encoder_start",
            metadata={"modality": modality, "batch_size": 1},
        )
        try:
            return _run_single_encoder_payload(
                payload,
                stage_name=stage_name,
                model=model,
                cache=cache,
            )
        finally:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="encoder_end",
                metadata={"modality": modality, "batch_size": 1},
            )

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        for payload in payloads:
            _emit_event(
                request_id=payload.request_id,
                stage=None,
                event_name="encoder_start",
                metadata={"modality": modality, "batch_size": len(payloads)},
            )
        try:
            return batch_fn(payloads, model=model, cache=cache)
        finally:
            for payload in payloads:
                _emit_event(
                    request_id=payload.request_id,
                    stage=None,
                    event_name="encoder_end",
                    metadata={"modality": modality, "batch_size": len(payloads)},
                )

    kwargs: dict[str, Any] = {
        "batch_compute_fn": _encode_batch,
        "max_batch_size": 32,
        "max_batch_wait_ms": _encoder_max_batch_wait_ms(),
    }
    if stage_name == IMAGE_STAGE:
        # Reuse Qwen3-Omni's calibrated visual encoder batch budget so
        # concurrent video benchmarks do not degrade into per-request serial
        # vision encoding.
        kwargs["request_cost_fn"] = _create_image_encoder_request_cost_fn(model)
        kwargs["max_batch_cost"] = QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES
    return SimpleScheduler(_encode, **kwargs)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    resolved_model_path = _resolve_qwen35_stage_model_path(model_path, IMAGE_STAGE)
    model = _load_qwen35_encoder(
        IMAGE_STAGE,
        model_path=resolved_model_path,
        device=device,
        dtype=dtype,
    )
    return _create_batched_encoder_executor(IMAGE_STAGE, model)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    resolved_model_path = _resolve_qwen35_stage_model_path(model_path, AUDIO_STAGE)
    model = _load_qwen35_encoder(
        AUDIO_STAGE,
        model_path=resolved_model_path,
        device=device,
        dtype=dtype,
    )
    return _create_batched_encoder_executor(AUDIO_STAGE, model)


def create_decode_executor(model_path: str):
    return _create_decode_executor(model_path)


_hf_config_registered = False


def _ensure_qwen3_omni_next_autoconfig():
    global _hf_config_registered
    if _hf_config_registered:
        return
    _hf_config_registered = True
    from sglang_omni.models.qwen3_5_omni.hf_config import register_qwen35_hf_config

    register_qwen35_hf_config()


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    thinker_max_seq_len: int = QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
    server_args_overrides: dict[str, Any] | None = None,
    encoder_mem_reserve: float = 0.05,
    speech_enabled: bool = False,
    capture_hidden_layers: list[int] | None = None,
    total_gpu_memory_fraction: float | None = None,
    root_model_path: str | None = None,
):
    """Returns OmniScheduler for the Qwen3.5-Omni thinker."""
    _ensure_qwen3_omni_next_autoconfig()

    resolved_model_path = _resolve_qwen35_stage_model_path(
        model_path,
        THINKER_STAGE,
    )
    metadata_model_path = root_model_path
    if metadata_model_path is None and resolved_model_path != model_path:
        metadata_model_path = model_path
    overrides: dict[str, Any] = {"disable_cuda_graph": True, "mem_fraction_static": 0.88}
    overrides.setdefault("max_prefill_tokens", QWEN3_5_OMNI_MAX_PREFILL_TOKENS)
    overrides.setdefault("chunked_prefill_size", QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE)
    has_user_mem_fraction_static = (
        server_args_overrides is not None
        and server_args_overrides.get("mem_fraction_static") is not None
    )
    if server_args_overrides:
        overrides.update(server_args_overrides)
    mamba_scheduler_strategy = os.getenv(
        "QWEN35_THINKER_MAMBA_SCHEDULER_STRATEGY",
        "extra_buffer",
    )
    if mamba_scheduler_strategy:
        overrides["mamba_scheduler_strategy"] = mamba_scheduler_strategy
    overrides["tp_size"] = tp_size
    has_explicit_colocated_mem_fraction = (
        total_gpu_memory_fraction is not None
        and has_user_mem_fraction_static
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
        resolved_model_path,
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
        capture_hidden_layers=capture_hidden_layers,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
        root_model_path=metadata_model_path,
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
    talker_max_seq_len: int = QWEN3_5_OMNI_TALKER_MAX_SEQ_LEN,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    weight_prefix: str = "talker.",
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = 5,
    root_model_path: str | None = None,
):
    """Returns OmniScheduler for the Qwen3.5-Omni talker."""
    _ensure_qwen3_omni_next_autoconfig()
    from sglang_omni.models.qwen3_5_omni.bootstrap import create_talker_scheduler

    resolved_model_path = _resolve_qwen35_stage_model_path(model_path, TALKER_STAGE)
    if root_model_path is None and resolved_model_path != model_path:
        # Split checkpoints commonly store unprefixed talker weights under
        # root/talker_lm or root/talker, while the root config still provides
        # tokenizer/special tokens.
        root_model_path = model_path
        weight_prefix = ""

    overrides: dict[str, Any] = {
        "disable_cuda_graph": True,
        "max_prefill_tokens": QWEN3_5_OMNI_MAX_PREFILL_TOKENS,
        "chunked_prefill_size": QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
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
        resolved_model_path,
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
        root_model_path=root_model_path,
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
