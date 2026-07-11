# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3.5-Omni text-output pipelines."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang.srt.utils.common import is_hopper_with_cuda_12_3

from sglang_omni.models.qwen3_omni import stages as qwen3_stages
from sglang_omni.models.qwen35_omni.bootstrap import create_thinker_scheduler
from sglang_omni.models.qwen35_omni.components.audio_encoder import (
    Qwen35OmniAudioEncoder,
)
from sglang_omni.models.qwen35_omni.components.image_encoder import (
    Qwen35OmniImageEncoder,
)
from sglang_omni.models.qwen35_omni.components.preprocessor import (
    Qwen35OmniPreprocessor,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.sglang_backend import build_sglang_server_args
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes
from sglang_omni.utils.misc import avail_gpu_mem

logger = logging.getLogger(__name__)

IMAGE_STAGE = qwen3_stages.IMAGE_STAGE
AUDIO_STAGE = qwen3_stages.AUDIO_STAGE
THINKER_STAGE = qwen3_stages.THINKER_STAGE


def _default_thinker_attention_backend() -> str:
    return "fa3" if is_hopper_with_cuda_12_3() else "triton"


load_state = qwen3_stages.load_state
store_state = qwen3_stages.store_state
create_aggregate_executor = qwen3_stages.create_aggregate_executor
create_decode_executor = qwen3_stages.create_decode_executor


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

    preprocessor = Qwen35OmniPreprocessor(
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


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    dedup_same_batch: bool = True,
    backend: str = "hf",
    attention_backend: str = "sdpa_grouped",
    max_batch_wait_ms: float = 50,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen35OmniImageEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
        backend=backend,
        attention_backend=attention_backend,
    )
    logger.info(
        "Qwen3.5 image encoder same-batch dedup enabled=%s max_batch_wait_ms=%s",
        dedup_same_batch,
        max_batch_wait_ms,
    )

    def _encode(payload: StagePayload) -> StagePayload:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="encoder_start",
            metadata={"modality": "image", "batch_size": 1},
        )
        try:
            return qwen3_stages._run_single_encoder_payload(
                payload,
                stage_name=IMAGE_STAGE,
                model=model,
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
            return qwen3_stages._batch_image_encoder_payloads(
                payloads,
                model=model,
                dedup_same_batch=dedup_same_batch,
            )
        finally:
            for p in payloads:
                _emit_event(
                    request_id=p.request_id,
                    stage=None,
                    event_name="encoder_end",
                    metadata={"modality": "image", "batch_size": len(payloads)},
                )

    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=32,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=qwen3_stages._create_image_encoder_request_cost_fn(model),
        max_batch_cost=qwen3_stages.QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
    )


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_batch_wait_ms: float = 50,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen35OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload: StagePayload) -> StagePayload:
        _emit_event(
            request_id=payload.request_id,
            stage=None,
            event_name="encoder_start",
            metadata={"modality": "audio", "batch_size": 1},
        )
        try:
            return qwen3_stages._run_single_encoder_payload(
                payload,
                stage_name=AUDIO_STAGE,
                model=model,
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
            return qwen3_stages._batch_audio_encoder_payloads(
                payloads,
                model=model,
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
        max_batch_wait_ms=max_batch_wait_ms,
    )


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
    total_gpu_memory_fraction: float | None = None,
):
    attention_backend = _default_thinker_attention_backend()
    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "disable_radix_cache": True,
        "enable_mixed_chunk": True,
        "chunked_prefill_size": 8192,
        "max_running_requests": 1,
        "sampling_backend": "pytorch",
        "attention_backend": attention_backend,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    logger.info(
        "Qwen3.5 thinker attention backend=%s",
        overrides["attention_backend"],
    )
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
    memory_contract = qwen3_stages._apply_colocated_ar_memory_contract(
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
    for attr, default in (
        ("enable_hisparse", False),
        ("enable_priority_scheduling", False),
        ("disable_priority_preemption", True),
    ):
        if not hasattr(server_args, attr):
            setattr(server_args, attr, default)
    if total_gpu_memory_fraction is None:
        reserve_applied = qwen3_stages._apply_qwen_thinker_encoder_reserve(
            server_args,
            has_explicit_mem_fraction_static=memory_contract.mem_fraction_static_pinned,
            encoder_mem_reserve=encoder_mem_reserve,
        )
        effective_total_gpu_memory_fraction = total_gpu_memory_fraction
        applied_encoder_reserve = encoder_mem_reserve if reserve_applied else 0.0
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
    talker_max_seq_len: int = 32768,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    weight_prefix: str = "talker.",
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = True,
    partial_start_min_chunks: int = 4,
):
    """Create the Qwen3.5 talker AR stage with 4x4 streaming interleave."""
    del speech_enabled
    from sglang_omni.models.qwen35_omni.bootstrap import create_talker_scheduler

    attention_backend = _default_thinker_attention_backend()
    overrides = build_generation_batch_overrides(
        max_running_requests=32,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=True,
        sampling_backend="pytorch",
        attention_backend=attention_backend,
    )
    overrides["tp_size"] = tp_size
    qwen3_stages._apply_colocated_ar_memory_contract(
        overrides,
        stage_name="talker_ar",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    server_args = build_sglang_server_args(
        model_path,
        context_length=talker_max_seq_len,
        **overrides,
    )
    for attr, default in (
        ("enable_hisparse", False),
        ("enable_priority_scheduling", False),
        ("disable_priority_preemption", True),
    ):
        if not hasattr(server_args, attr):
            setattr(server_args, attr, default)
    validate_generation_batch_policy(
        model_name="Qwen3.5-Omni talker_ar",
        server_args=server_args,
    )
    logger.info(
        "sglang_ar_startup stage=talker_ar gpu_id=%s tp_rank=%s/%s "
        "context_length=%s cuda_graph=%s attention_backend=%s",
        gpu_id,
        tp_rank,
        tp_size,
        talker_max_seq_len,
        not server_args.disable_cuda_graph,
        server_args.attention_backend,
    )
    return create_talker_scheduler(
        server_args,
        gpu_id,
        weight_prefix=weight_prefix,
        feedback_enabled=feedback_enabled,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
    )
