# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3.5-Omni Next text output."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

# Import registers local AutoConfig classes for qwen3_omni_next.
import sglang_omni.models.qwen35_omni.hf_config  # noqa: F401
from sglang_omni.config import PipelineConfig, PlacementConfig, StageConfig
from sglang_omni.models.qwen3_omni.config import (
    _DEEPGEMM_PRECOMPILE_ENV_DEFAULTS,
)

_PKG = "sglang_omni.models.qwen35_omni"
THINKER_STAGE = "thinker"


def _preprocessing_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="preprocessing",
        process=process,
        factory=f"{_PKG}.stages.create_preprocessing_executor",
        factory_args={"thinker_max_seq_len": 8192},
        runtime_arg_map={
            "max_seq_len": "thinker_max_seq_len",
            "video_fps": "video_fps",
        },
        next=["image_encoder", "audio_encoder", "mm_aggregate"],
        route_fn=f"{_PKG}.request_builders.resolve_preprocessing_next_stages",
        project_payload={
            "image_encoder": f"{_PKG}.request_builders.project_preprocessing_to_image_encoder",
            "audio_encoder": f"{_PKG}.request_builders.project_preprocessing_to_audio_encoder",
            "mm_aggregate": f"{_PKG}.request_builders.project_preprocessing_to_mm_aggregate",
        },
    )


def _image_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory=f"{_PKG}.stages.create_image_encoder_executor",
        factory_args={
            "device": "cuda",
            "dtype": None,
            "backend": "sglang",
            "max_batch_wait_ms": 0,
        },
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _audio_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        process=process,
        factory=f"{_PKG}.stages.create_audio_encoder_executor",
        factory_args={"device": "cuda", "dtype": None, "max_batch_wait_ms": 0},
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _aggregate_stage(*, process: str, speech_enabled: bool = False) -> StageConfig:
    if speech_enabled:
        return StageConfig(
            name="mm_aggregate",
            process=process,
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next=["thinker", "talker_ar"],
            route_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_next_stages",
            project_payload={
                "talker_ar": (
                    f"{_PKG}.request_builders.project_mm_aggregate_to_talker_ar"
                )
            },
        )
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory=f"{_PKG}.stages.create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
    )


def _thinker_stage(
    *, gpu: int, process: str, speech_enabled: bool = False
) -> StageConfig:
    return StageConfig(
        name="thinker",
        process=process,
        factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory_args={"thinker_max_seq_len": 8192},
        gpu=gpu,
        runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
        next="decode",
        stream_to=["talker_ar", "decode"] if speech_enabled else ["decode"],
        route_fn=(
            f"{_PKG}.request_builders.resolve_thinker_next_stages"
            if speech_enabled
            else None
        ),
        stream_done_to_fn=(
            f"{_PKG}.request_builders.resolve_thinker_stream_done_targets"
            if speech_enabled
            else None
        ),
        project_payload={
            "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
        },
    )


def _decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _talker_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="talker_ar",
        process=process,
        factory=f"{_PKG}.stages.create_talker_ar_executor_from_config",
        factory_args={
            "talker_max_seq_len": 32768,
            "speech_enabled": True,
            "feedback_enabled": True,
            "enable_partial_start": True,
            "partial_start_min_chunks": 4,
        },
        gpu=gpu,
        runtime_arg_map={"max_seq_len": "talker_max_seq_len"},
        next="code2wav",
        stream_to=["code2wav"],
        project_payload={
            "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav"
        },
        can_accept_stream_before_payload=True,
    )


def _code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        factory_args={"device": "cuda"},
        gpu=gpu,
        terminal=True,
        can_accept_stream_before_payload=True,
    )


_TEXT_DEFAULT_PROCESSES = {
    "preprocessing": "preprocessing",
    "image_encoder": "pipeline",
    "audio_encoder": "pipeline",
    "mm_aggregate": "pipeline",
    "thinker": "pipeline",
    "decode": "pipeline",
}


def _text_stages(*, process_by_stage: dict[str, str]) -> list[StageConfig]:
    return [
        _preprocessing_stage(process=process_by_stage["preprocessing"]),
        _image_encoder_stage(gpu=0, process=process_by_stage["image_encoder"]),
        _audio_encoder_stage(gpu=0, process=process_by_stage["audio_encoder"]),
        _aggregate_stage(process=process_by_stage["mm_aggregate"]),
        _thinker_stage(gpu=0, process=process_by_stage["thinker"]),
        _decode_stage(process=process_by_stage["decode"]),
    ]


_SPEECH_DEFAULT_PROCESSES = {
    **_TEXT_DEFAULT_PROCESSES,
    "talker_ar": "talker_ar",
    "code2wav": "code2wav",
}


def _speech_stages() -> list[StageConfig]:
    processes = _SPEECH_DEFAULT_PROCESSES
    return [
        _preprocessing_stage(process=processes["preprocessing"]),
        _image_encoder_stage(gpu=0, process=processes["image_encoder"]),
        _audio_encoder_stage(gpu=0, process=processes["audio_encoder"]),
        _aggregate_stage(process=processes["mm_aggregate"], speech_enabled=True),
        _thinker_stage(gpu=0, process=processes["thinker"], speech_enabled=True),
        _decode_stage(process=processes["decode"]),
        _talker_stage(gpu=1, process=processes["talker_ar"]),
        _code2wav_stage(gpu=1, process=processes["code2wav"]),
    ]


class Qwen35OmniPipelineConfig(PipelineConfig):
    """6-stage Qwen3.5-Omni text-output pipeline."""

    architecture: ClassVar[str] = "Qwen3OmniNextForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Qwen35OmniNextForConditionalGeneration",
    )
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = (
        THINKER_STAGE,
    )
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    @classmethod
    def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
        return {THINKER_STAGE}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": THINKER_STAGE}

    model_path: str
    placement_policy: str | None = None
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: _text_stages(process_by_stage=_TEXT_DEFAULT_PROCESSES)
    )


class Qwen35OmniSpeechPipelineConfig(Qwen35OmniPipelineConfig):
    """Qwen3.5-Omni text+speech pipeline using a dedicated talker GPU."""

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE, "talker": "talker_ar"}

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "talker_ar"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "talker_ar"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "talker_ar"}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "code2wav"

    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"
    stages: list[StageConfig] = Field(default_factory=_speech_stages)


EntryClass = Qwen35OmniPipelineConfig
Variants = {
    "text": Qwen35OmniPipelineConfig,
    "speech": Qwen35OmniSpeechPipelineConfig,
}
