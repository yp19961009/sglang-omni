# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3.5-Omni."""

from __future__ import annotations

import os
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    PlacementConfig,
    RelayConfig,
    SGLangServerArgsConfig,
    StageConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.qwen3_5_omni"
_PLACEMENT_POLICY = "sglang_omni.models.qwen3_omni.placement.Qwen3OmniPlacementPolicy"
MIN_PARTIAL_START_CHUNKS = 3
# The online Qwen3.5-Omni speech profile uses max_model_len=192000 by
# default. Longer 262144 contexts mainly show up in thinker-only eval and
# long-context scripts. Keep SGLang's default aligned with the online speech
# profile to reduce first-smoke memory pressure; pass --thinker-max-seq-len
# explicitly when 256k is needed.
QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN = 192000
QWEN3_5_OMNI_TALKER_MAX_SEQ_LEN = 32768
QWEN3_5_OMNI_MAX_PREFILL_TOKENS = 32768
QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE = 32768
QWEN3_5_OMNI_LIMIT_MM_PER_PROMPT = {
    "audio": 960,
    "image": 960,
    "video": 960,
}
# The reference H20 profile uses max_num_seqs=32. The matching SGLang field is
# max_running_requests, which the talker also uses for its internal buffers.
QWEN3_5_OMNI_MAX_RUNNING_REQUESTS = 32
QWEN3_5_OMNI_CODE2WAV_ENABLE_TORCH_COMPILE = True
QWEN3_5_OMNI_MODEL_NAME = "qwen3.5-omni"
QWEN3_5_OMNI_MODEL_NAME_ALIASES = (
    QWEN3_5_OMNI_MODEL_NAME,
    "qwen3-5-omni",
    "qwen3_5_omni",
    "qwen35-omni",
    "qwen35_omni",
    "qwen35omni",
)

# See the Qwen3-Omni config for the DeepGEMM rationale. Keep the default here
# while Qwen3.5-Omni reuses the same AR scheduling envelope.
_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS = {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}
_COLOCATE_MM_AGGREGATE_ENV = "SGLANG_OMNI_COLOCATE_MM_AGGREGATE_WITH_THINKER"
_COLOCATE_MM_AGGREGATE_WITH_IMAGE_ENV = (
    "SGLANG_OMNI_COLOCATE_MM_AGGREGATE_WITH_IMAGE_ENCODER"
)
_COLOCATE_IMAGE_ENCODER_WITH_THINKER_ENV = (
    "SGLANG_OMNI_COLOCATE_IMAGE_ENCODER_WITH_THINKER"
)
_COLOCATE_PREPROCESSING_WITH_THINKER_ENV = (
    "SGLANG_OMNI_COLOCATE_PREPROCESSING_WITH_THINKER"
)
_MM_AGGREGATE_RELAY_GPU_ENV = "SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_THINKER_GPU"
_MM_AGGREGATE_RELAY_TALKER_GPU_ENV = (
    "SGLANG_OMNI_MM_AGGREGATE_RELAY_ON_TALKER_GPU"
)

# Keep architecture names aligned with the reference registry. The root model
# uses the root architecture, while thinker-only and thinker-MTP names are
# registered as aliases. MTP runtime support is still reported as unavailable by
# preflight.
QWEN3_5_OMNI_ARCH = "Qwen3OmniNextForConditionalGeneration"
QWEN3_5_OMNI_ARCH_ALIASES = (
    "Qwen3OmniNextThinkerForConditionalGeneration",
    "Qwen3OmniNextThinkerMTP",
)


def normalize_qwen35_omni_model_name(model_name: str | None) -> str | None:
    """Return the canonical OpenAI model name for common Qwen3.5 spellings."""

    if model_name is None:
        return None
    compact = "".join(ch for ch in model_name.strip().lower() if ch.isalnum())
    alias_keys = {
        "".join(ch for ch in alias.lower() if ch.isalnum())
        for alias in QWEN3_5_OMNI_MODEL_NAME_ALIASES
    }
    if compact in alias_keys:
        # Users and benchmarks commonly mix qwen3.5-omni, qwen3_5_omni, and
        # qwen3-5-omni spellings. Expose one canonical name at the serve entry
        # point so /v1/models, result JSON, and benchmark labels stay stable.
        return QWEN3_5_OMNI_MODEL_NAME
    return model_name


def _qwen35_ar_server_args_overrides() -> dict[str, int]:
    return {
        "max_prefill_tokens": QWEN3_5_OMNI_MAX_PREFILL_TOKENS,
        "chunked_prefill_size": QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
    }


def _qwen35_ar_runtime() -> StageRuntimeConfig:
    return StageRuntimeConfig(
        sglang_server_args=SGLangServerArgsConfig(
            max_running_requests=QWEN3_5_OMNI_MAX_RUNNING_REQUESTS,
        )
    )


def _preprocessing_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="preprocessing",
        process=process,
        factory=f"{_PKG}.stages.create_preprocessing_executor",
        # The Qwen3.5 online speech profile uses a thinker max_model_len of
        # 192000. Long video and audio inputs need a larger default guard than
        # Qwen3's 8192; 256k eval runs can override this via runtime or CLI.
        factory_args={
            "thinker_max_seq_len": QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
            "limit_mm_per_prompt": dict(QWEN3_5_OMNI_LIMIT_MM_PER_PROMPT),
        },
        # Map shared runtime overrides to preprocessing factory kwargs.
        runtime_arg_map={
            "max_seq_len": "thinker_max_seq_len",
            # Common benchmark visual defaults can be supplied through
            # runtime/YAML and forwarded to the preprocessor, instead of being
            # repeated in each request's videos_kwargs.
            "image_min_pixels": "image_min_pixels",
            "image_max_pixels": "image_max_pixels",
            "video_fps": "video_fps",
            "video_max_frames": "video_max_frames",
            "video_min_frames": "video_min_frames",
            "video_min_pixels": "video_min_pixels",
            "video_max_pixels": "video_max_pixels",
            "video_total_pixels": "video_total_pixels",
            "video_override_max_pixels": "video_override_max_pixels",
            "video_seconds_per_chunk": "video_seconds_per_chunk",
            "video_position_id_per_seconds": "video_position_id_per_seconds",
            "audio_target_sr": "audio_target_sr",
            "audio_sampling_rate": "audio_target_sr",
            "sampling_rate": "audio_target_sr",
            # Accept the short names used in some Qwen3.5 configs and map them
            # to the audio_* arguments consumed by the SGLang-Omni
            # preprocessing executor.
            "audio_timestamp_interval": "audio_timestamp_interval",
            "timestamp_interval": "audio_timestamp_interval",
            "audio_downsample_times": "audio_downsample_times",
            "downsample_times": "audio_downsample_times",
            "audio_downsample_chunk_size": "audio_downsample_chunk_size",
            "downsample_chunk_size": "audio_downsample_chunk_size",
        },
        next=["image_encoder", "audio_encoder", "mm_aggregate"],
        route_fn=f"{_PKG}.request_builders.resolve_preprocessing_next_stages",
        project_payload={
            "image_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_image_encoder"
            ),
            "audio_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_audio_encoder"
            ),
            "mm_aggregate": (
                f"{_PKG}.request_builders.project_preprocessing_to_mm_aggregate"
            ),
        },
    )


def _image_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory=f"{_PKG}.stages.create_image_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
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
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        next="mm_aggregate",
        project_payload={
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    )


def _aggregate_stage(
    *,
    process: str,
    speech_enabled: bool = False,
    gpu: int | None = None,
    relay_device: str | None = None,
) -> StageConfig:
    relay = RelayConfig(device=relay_device) if relay_device is not None else None
    if speech_enabled:
        return StageConfig(
            name="mm_aggregate",
            process=process,
            factory=f"{_PKG}.stages.create_aggregate_executor",
            gpu=gpu,
            relay=relay,
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next=["thinker", "talker_ar"],
            route_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_next_stages",
            project_payload={
                "thinker": (
                    f"{_PKG}.request_builders.project_mm_aggregate_to_thinker"
                ),
                "talker_ar": (
                    f"{_PKG}.request_builders.project_mm_aggregate_to_talker_ar"
                ),
            },
        )
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory=f"{_PKG}.stages.create_aggregate_executor",
        gpu=gpu,
        relay=relay,
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
    )


def _thinker_stage(*, gpu: int, speech_enabled: bool, process: str) -> StageConfig:
    factory_args = {
        "thinker_max_seq_len": QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
        "server_args_overrides": _qwen35_ar_server_args_overrides(),
    }
    if speech_enabled:
        factory_args["speech_enabled"] = True
    return StageConfig(
        name="thinker",
        process=process,
        factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory_args=factory_args,
        gpu=gpu,
        runtime=_qwen35_ar_runtime(),
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


def _talker_stage(
    *,
    gpu: int,
    process: str,
    enable_partial_start: bool,
) -> StageConfig:
    return StageConfig(
        name="talker_ar",
        process=process,
        factory=f"{_PKG}.stages.create_talker_ar_executor_from_config",
        factory_args={
            "talker_max_seq_len": QWEN3_5_OMNI_TALKER_MAX_SEQ_LEN,
            "server_args_overrides": _qwen35_ar_server_args_overrides(),
            "speech_enabled": True,
            "feedback_enabled": True,
            "enable_partial_start": enable_partial_start,
            "partial_start_min_chunks": 5,
        },
        gpu=gpu,
        runtime=_qwen35_ar_runtime(),
        runtime_arg_map={"max_seq_len": "talker_max_seq_len"},
        next="code2wav",
        stream_to=["code2wav"],
        project_payload={
            "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav",
        },
        can_accept_stream_before_payload=True,
    )


def _code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        # The Qwen3.5-Omni reference profile enables code2wav torch.compile by
        # default. Keep the same hot-path audio performance default while still
        # allowing CLI/YAML to disable it with --no-code2wav-torch-compile.
        factory_args={
            "device": "cuda",
            "enable_torch_compile": QWEN3_5_OMNI_CODE2WAV_ENABLE_TORCH_COMPILE,
        },
        runtime_arg_map={
            # The reference send_chunk_size maps to the code2wav scheduler's
            # stream_chunk_size, which affects first-audio chunking and
            # streaming throughput.
            "code2wav_stream_chunk_size": "stream_chunk_size",
            "send_chunk_size": "stream_chunk_size",
            "code2wav_codec_eos_token_id": "codec_eos_token_id",
            "code2wav_sample_rate": "sample_rate",
            "code2wav_left_context_size": "left_context_size",
            "code2wav_enable_dynamic_chunk": "enable_dynamic_chunk",
            # The reference code2wav_dynamic_batch flag controls an engine-side
            # dynamic decoding strategy. In SGLang it maps to the existing
            # dynamic chunk scheduler.
            "enable_dynamic_chunk": "enable_dynamic_chunk",
            "code2wav_dynamic_batch": "enable_dynamic_chunk",
            "dynamic_batch": "enable_dynamic_chunk",
            "code2wav_dynamic_chunk_sizes": "dynamic_chunk_sizes",
            "code2wav_dynamic_chunk_steps": "dynamic_chunk_steps",
            "code2wav_enable_torch_compile": "enable_torch_compile",
            "code2wav_enable_torch_compile_first_chunk": (
                "enable_torch_compile_first_chunk"
            ),
            "enable_torch_compile_first_chunk": (
                "enable_torch_compile_first_chunk"
            ),
            "code2wav_odeint_method": "odeint_method",
            "odeint_method": "odeint_method",
            "code2wav_odeint_method_relaxed": "odeint_method_relaxed",
            "odeint_method_relaxed": "odeint_method_relaxed",
            "code2wav_batched_chunk": "batched_chunk",
            "batched_chunk": "batched_chunk",
            "code2wav_frequency": "frequency",
            "frequency": "frequency",
            "code2wav_dit_quant": "dit_quant",
            "dit_quant": "dit_quant",
        },
        gpu=gpu,
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _text_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="pipeline"),
        _image_encoder_stage(gpu=0, process="pipeline"),
        _audio_encoder_stage(gpu=0, process="pipeline"),
        _aggregate_stage(process="pipeline", speech_enabled=False),
        _thinker_stage(gpu=0, speech_enabled=False, process="pipeline"),
        _decode_stage(process="pipeline"),
    ]


def _speech_stages(
    *,
    thinker_gpu: int,
    talker_gpu: int,
    process_by_stage: dict[str, str],
    enable_partial_start: bool,
) -> list[StageConfig]:
    if _env_enabled(_MM_AGGREGATE_RELAY_GPU_ENV):
        mm_aggregate_gpu = thinker_gpu
    else:
        mm_aggregate_gpu = None
    mm_aggregate_relay_device = (
        f"cuda:{talker_gpu}"
        if _env_enabled(_MM_AGGREGATE_RELAY_TALKER_GPU_ENV)
        else None
    )
    return [
        _preprocessing_stage(process=process_by_stage["preprocessing"]),
        _image_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["image_encoder"],
        ),
        _audio_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["audio_encoder"],
        ),
        _aggregate_stage(
            process=process_by_stage["mm_aggregate"],
            speech_enabled=True,
            gpu=mm_aggregate_gpu,
            relay_device=mm_aggregate_relay_device,
        ),
        _thinker_stage(
            gpu=thinker_gpu,
            speech_enabled=True,
            process=process_by_stage["thinker"],
        ),
        _decode_stage(process=process_by_stage["decode"]),
        _talker_stage(
            gpu=talker_gpu,
            process=process_by_stage["talker_ar"],
            enable_partial_start=enable_partial_start,
        ),
        _code2wav_stage(gpu=talker_gpu, process=process_by_stage["code2wav"]),
    ]


_SPEECH_DEFAULT_PROCESSES = {
    "preprocessing": "preprocessing",
    "image_encoder": "image_encoder",
    "audio_encoder": "audio_encoder",
    "mm_aggregate": "mm_aggregate",
    "thinker": "thinker",
    "decode": "decode",
    "talker_ar": "talker_ar",
    "code2wav": "code2wav",
}


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _speech_default_processes() -> dict[str, str]:
    processes = dict(_SPEECH_DEFAULT_PROCESSES)
    if _env_enabled(_COLOCATE_PREPROCESSING_WITH_THINKER_ENV):
        processes["preprocessing"] = processes["thinker"]
    if _env_enabled(_COLOCATE_IMAGE_ENCODER_WITH_THINKER_ENV):
        processes["image_encoder"] = processes["thinker"]
    if _env_enabled(_COLOCATE_MM_AGGREGATE_ENV):
        processes["mm_aggregate"] = processes["thinker"]
    elif _env_enabled(_COLOCATE_MM_AGGREGATE_WITH_IMAGE_ENV):
        processes["mm_aggregate"] = processes["image_encoder"]
    return processes


class Qwen35OmniPipelineConfig(PipelineConfig):
    """6-stage Qwen3.5-Omni thinker pipeline."""

    architecture: ClassVar[str] = QWEN3_5_OMNI_ARCH
    architecture_aliases: ClassVar[tuple[str, ...]] = QWEN3_5_OMNI_ARCH_ALIASES
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    @classmethod
    def max_running_requests_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_text_stages)


class Qwen35OmniSpeechPipelineConfig(PipelineConfig):
    """8-stage Qwen3.5-Omni speech pipeline (text + audio output)."""

    architecture: ClassVar[str] = QWEN3_5_OMNI_ARCH
    architecture_aliases: ClassVar[tuple[str, ...]] = QWEN3_5_OMNI_ARCH_ALIASES
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker", "talker": "talker_ar"}

    @classmethod
    def max_running_requests_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker", "talker": "talker_ar"}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": "thinker"}

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "talker_ar"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "talker_ar"}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "code2wav"

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(
            thinker_gpu=0,
            talker_gpu=1,
            process_by_stage=_speech_default_processes(),
            enable_partial_start=True,
        )
    )


class Qwen35OmniSpeechColocatedPipelineConfig(Qwen35OmniSpeechPipelineConfig):
    """8-stage Qwen3.5-Omni speech pipeline for single-GPU colocation."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(
            thinker_gpu=0,
            talker_gpu=0,
            process_by_stage=_speech_default_processes(),
            enable_partial_start=False,
        )
    )


EntryClass = Qwen35OmniSpeechPipelineConfig

Variants = {
    "text": Qwen35OmniPipelineConfig,
    "speech": Qwen35OmniSpeechPipelineConfig,
    "speech-colocated": Qwen35OmniSpeechColocatedPipelineConfig,
}
