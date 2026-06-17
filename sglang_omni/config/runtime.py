# SPDX-License-Identifier: Apache-2.0
"""Resolve typed runtime config into stage factory arguments."""

from __future__ import annotations

import inspect
from typing import Any

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils.imports import import_string

_MAPPED_STAGE_RUNTIME_FIELDS = (
    "max_seq_len",
    "image_min_pixels",
    "image_max_pixels",
    "video_fps",
    "video_max_frames",
    "video_min_frames",
    "video_min_pixels",
    "video_max_pixels",
    "video_total_pixels",
    "video_override_max_pixels",
    "video_seconds_per_chunk",
    "video_position_id_per_seconds",
    "audio_target_sr",
    "audio_timestamp_interval",
    "timestamp_interval",
    "audio_downsample_times",
    "downsample_times",
    "audio_downsample_chunk_size",
    "downsample_chunk_size",
    "code2wav_stream_chunk_size",
    "send_chunk_size",
    "code2wav_codec_eos_token_id",
    "code2wav_sample_rate",
    "code2wav_left_context_size",
    "code2wav_enable_dynamic_chunk",
    "enable_dynamic_chunk",
    "code2wav_dynamic_batch",
    "dynamic_batch",
    "code2wav_dynamic_chunk_sizes",
    "code2wav_dynamic_chunk_steps",
    "code2wav_enable_torch_compile",
    "code2wav_enable_torch_compile_first_chunk",
    "enable_torch_compile_first_chunk",
    "code2wav_odeint_method",
    "odeint_method",
    "code2wav_odeint_method_relaxed",
    "odeint_method_relaxed",
    "code2wav_batched_chunk",
    "batched_chunk",
    "code2wav_frequency",
    "frequency",
    "code2wav_dit_quant",
    "dit_quant",
)


def resolve_stage_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """Resolve final factory kwargs for a stage.

    Values are built from stage.factory_args, runtime_overrides, and typed
    stage.runtime fields, with typed runtime owning V1 resource contracts.
    Placement budgets are injected only when the factory declares them.
    """

    args = dict(stage_cfg.factory_args)
    runtime_overrides = global_cfg.runtime_overrides.get(stage_cfg.name, {})
    _validate_runtime_sources(stage_cfg, args, runtime_overrides)
    _merge_factory_arg_overrides(args, runtime_overrides)
    _apply_typed_runtime_args(args, stage_cfg) # 把通用 runtime 配置应用到当前 stage 的 factory kwargs 上

    factory = import_string(stage_cfg.factory)
    sig = inspect.signature(factory)

    if "model_path" in sig.parameters and "model_path" not in args:
        args["model_path"] = global_cfg.model_path

    if "gpu_id" in sig.parameters and "gpu_id" not in args:
        args["gpu_id"] = (
            gpu_id
            if gpu_id is not None
            else _resolve_primary_gpu_id(stage_cfg, global_cfg)
        )

    total_gpu_memory_fraction = stage_cfg.runtime.resources.total_gpu_memory_fraction
    if (
        total_gpu_memory_fraction is not None
        and "total_gpu_memory_fraction" in sig.parameters
        and "total_gpu_memory_fraction" not in args
    ):
        args["total_gpu_memory_fraction"] = total_gpu_memory_fraction

    return args


def reject_untyped_total_gpu_memory_fraction(
    stage_name: str,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    if (
        factory_args.get("total_gpu_memory_fraction") is None
        and runtime_overrides.get("total_gpu_memory_fraction") is None
    ):
        return
    raise ValueError(
        f"Stage {stage_name!r} sets total_gpu_memory_fraction through "
        "factory_args/runtime_overrides; set "
        "runtime.resources.total_gpu_memory_fraction instead"
    )


def _validate_runtime_sources(
    stage_cfg: StageConfig,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    """Validate ownership of runtime fields."""

    typed_mem_fraction = stage_cfg.runtime.sglang_server_args.mem_fraction_static
    if typed_mem_fraction is not None and _server_args_mem_fraction_static_is_set(
        factory_args,
        runtime_overrides,
    ):
        raise ValueError(
            f"Stage {stage_cfg.name!r} sets mem_fraction_static through both "
            "server_args_overrides and typed "
            "runtime.sglang_server_args.mem_fraction_static"
        )

    reject_untyped_total_gpu_memory_fraction(
        stage_cfg.name,
        factory_args,
        runtime_overrides,
    )

    for field_name in _MAPPED_STAGE_RUNTIME_FIELDS:
        value = getattr(stage_cfg.runtime, field_name)
        if value is None:
            continue
        target_arg = stage_cfg.runtime_arg_map.get(field_name)
        if target_arg and target_arg in runtime_overrides:
            raise ValueError(
                f"Stage {stage_cfg.name!r} sets {target_arg!r} through both "
                f"runtime_overrides and typed runtime.{field_name}"
            )


def _server_args_mem_fraction_static_is_set(
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> bool:
    for source in (factory_args, runtime_overrides):
        server_args = source.get("server_args_overrides")
        if (
            isinstance(server_args, dict)
            and server_args.get("mem_fraction_static") is not None
        ):
            return True
    return False


def _merge_factory_arg_overrides(
    args: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    for key, value in overrides.items():
        if (
            key == "server_args_overrides"
            and isinstance(value, dict)
            and isinstance(args.get(key), dict)
        ):
            merged = dict(args[key])
            merged.update(value)
            args[key] = merged
            continue
        args[key] = value


def _apply_typed_runtime_args(args: dict[str, Any], stage_cfg: StageConfig) -> None:
    runtime = stage_cfg.runtime

    for field_name in _MAPPED_STAGE_RUNTIME_FIELDS:
        value = getattr(runtime, field_name)
        if value is None:
            continue
        target_arg = stage_cfg.runtime_arg_map.get(field_name)
        if not target_arg:
            raise ValueError(
                f"Stage {stage_cfg.name!r} sets runtime.{field_name} but does not "
                f"define runtime_arg_map[{field_name!r}]"
            )
        args[target_arg] = value

    mem_fraction_static = runtime.sglang_server_args.mem_fraction_static
    if mem_fraction_static is not None:
        overrides = dict(args.get("server_args_overrides") or {})
        overrides["mem_fraction_static"] = mem_fraction_static
        args["server_args_overrides"] = overrides
    max_running_requests = runtime.sglang_server_args.max_running_requests
    if max_running_requests is not None:
        overrides = dict(args.get("server_args_overrides") or {})
        overrides["max_running_requests"] = int(max_running_requests)
        args["server_args_overrides"] = overrides


def _resolve_primary_gpu_id(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> int | None:
    placement = global_cfg.gpu_placement.get(stage_cfg.name)
    if placement is None:
        return None
    if isinstance(placement, list):
        return placement[0]
    return int(placement)
