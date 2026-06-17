# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
    QWEN3_5_OMNI_CODE2WAV_ENABLE_TORCH_COMPILE,
    QWEN3_5_OMNI_LIMIT_MM_PER_PROMPT,
    QWEN3_5_OMNI_MAX_RUNNING_REQUESTS,
    QWEN3_5_OMNI_MAX_PREFILL_TOKENS,
    QWEN3_5_OMNI_MODEL_NAME,
    QWEN3_5_OMNI_TALKER_MAX_SEQ_LEN,
    QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
    Qwen35OmniPipelineConfig,
    Qwen35OmniSpeechColocatedPipelineConfig,
    Qwen35OmniSpeechPipelineConfig,
    Variants,
    normalize_qwen35_omni_model_name,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_qwen35_omni_registry_architectures():
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3OmniNextForConditionalGeneration")
        is Qwen35OmniSpeechPipelineConfig
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(
            "Qwen3OmniNextThinkerForConditionalGeneration"
        )
        is Qwen35OmniSpeechPipelineConfig
    )


def test_qwen35_omni_variants():
    assert Variants["text"] is Qwen35OmniPipelineConfig
    assert Variants["speech"] is Qwen35OmniSpeechPipelineConfig
    assert Variants["speech-colocated"] is Qwen35OmniSpeechColocatedPipelineConfig


def test_qwen35_model_name_aliases_normalize_to_canonical_name():
    for alias in (
        "qwen3.5-omni",
        "qwen3-5-omni",
        "qwen3_5_omni",
        "qwen35-omni",
        "qwen35_omni",
        "QWEN35OMNI",
    ):
        assert normalize_qwen35_omni_model_name(alias) == QWEN3_5_OMNI_MODEL_NAME

    assert normalize_qwen35_omni_model_name("custom-model") == "custom-model"
    assert normalize_qwen35_omni_model_name(None) is None


def test_qwen35_omni_stage_factories_are_model_specific():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    assert (
        config.terminal_stages_fn
        == "sglang_omni.models.qwen3_5_omni.request_builders.resolve_terminal_stages"
    )
    assert (
        _stage(config, "preprocessing").factory
        == "sglang_omni.models.qwen3_5_omni.stages.create_preprocessing_executor"
    )
    assert (
        _stage(config, "thinker").factory
        == "sglang_omni.models.qwen3_5_omni.stages.create_sglang_thinker_executor_from_config"
    )
    assert (
        _stage(config, "talker_ar").factory
        == "sglang_omni.models.qwen3_5_omni.stages.create_talker_ar_executor_from_config"
    )
    assert (
        _stage(config, "code2wav").factory
        == "sglang_omni.models.qwen3_5_omni.components.code2wav_scheduler.create_code2wav_scheduler"
    )


def test_qwen35_omni_defaults_match_native_context_envelope():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    assert QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN == 192000
    assert QWEN3_5_OMNI_LIMIT_MM_PER_PROMPT == {
        "audio": 960,
        "image": 960,
        "video": 960,
    }
    assert (
        _stage(config, "preprocessing").factory_args["thinker_max_seq_len"]
        == QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN
    )
    assert (
        _stage(config, "preprocessing").factory_args["limit_mm_per_prompt"]
        == QWEN3_5_OMNI_LIMIT_MM_PER_PROMPT
    )
    assert (
        _stage(config, "thinker").factory_args["thinker_max_seq_len"]
        == QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN
    )
    assert (
        _stage(config, "talker_ar").factory_args["talker_max_seq_len"]
        == QWEN3_5_OMNI_TALKER_MAX_SEQ_LEN
    )
    for stage_name in ("thinker", "talker_ar"):
        overrides = _stage(config, stage_name).factory_args["server_args_overrides"]
        assert overrides["max_prefill_tokens"] == QWEN3_5_OMNI_MAX_PREFILL_TOKENS
        assert (
            overrides["chunked_prefill_size"]
            == QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE
        )
        assert (
            _stage(
                config,
                stage_name,
            ).runtime.sglang_server_args.max_running_requests
            == QWEN3_5_OMNI_MAX_RUNNING_REQUESTS
        )
    assert (
        _stage(config, "code2wav").factory_args["enable_torch_compile"]
        is QWEN3_5_OMNI_CODE2WAV_ENABLE_TORCH_COMPILE
    )


def test_qwen35_omni_preprocessing_runtime_video_args_are_mapped():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    runtime_arg_map = _stage(config, "preprocessing").runtime_arg_map

    assert runtime_arg_map["image_min_pixels"] == "image_min_pixels"
    assert runtime_arg_map["image_max_pixels"] == "image_max_pixels"
    assert runtime_arg_map["video_fps"] == "video_fps"
    assert runtime_arg_map["video_max_frames"] == "video_max_frames"
    assert runtime_arg_map["video_min_frames"] == "video_min_frames"
    assert runtime_arg_map["video_min_pixels"] == "video_min_pixels"
    assert runtime_arg_map["video_max_pixels"] == "video_max_pixels"
    assert runtime_arg_map["video_total_pixels"] == "video_total_pixels"
    assert runtime_arg_map["video_override_max_pixels"] == "video_override_max_pixels"
    assert runtime_arg_map["video_seconds_per_chunk"] == "video_seconds_per_chunk"
    assert (
        runtime_arg_map["video_position_id_per_seconds"]
        == "video_position_id_per_seconds"
    )
    assert runtime_arg_map["audio_target_sr"] == "audio_target_sr"
    assert runtime_arg_map["audio_sampling_rate"] == "audio_target_sr"
    assert runtime_arg_map["sampling_rate"] == "audio_target_sr"
    assert (
        runtime_arg_map["audio_timestamp_interval"]
        == "audio_timestamp_interval"
    )
    assert runtime_arg_map["timestamp_interval"] == "audio_timestamp_interval"
    assert runtime_arg_map["audio_downsample_times"] == "audio_downsample_times"
    assert runtime_arg_map["downsample_times"] == "audio_downsample_times"
    assert (
        runtime_arg_map["audio_downsample_chunk_size"]
        == "audio_downsample_chunk_size"
    )
    assert (
        runtime_arg_map["downsample_chunk_size"]
        == "audio_downsample_chunk_size"
    )


def test_qwen35_omni_code2wav_runtime_args_are_mapped():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    runtime_arg_map = _stage(config, "code2wav").runtime_arg_map

    assert runtime_arg_map["code2wav_stream_chunk_size"] == "stream_chunk_size"
    assert runtime_arg_map["send_chunk_size"] == "stream_chunk_size"
    assert runtime_arg_map["code2wav_codec_eos_token_id"] == "codec_eos_token_id"
    assert runtime_arg_map["code2wav_sample_rate"] == "sample_rate"
    assert runtime_arg_map["code2wav_left_context_size"] == "left_context_size"
    assert runtime_arg_map["code2wav_enable_dynamic_chunk"] == "enable_dynamic_chunk"
    assert runtime_arg_map["code2wav_dynamic_chunk_sizes"] == "dynamic_chunk_sizes"
    assert runtime_arg_map["code2wav_dynamic_chunk_steps"] == "dynamic_chunk_steps"
    assert runtime_arg_map["code2wav_enable_torch_compile"] == (
        "enable_torch_compile"
    )
    assert runtime_arg_map["code2wav_enable_torch_compile_first_chunk"] == (
        "enable_torch_compile_first_chunk"
    )
    assert runtime_arg_map["enable_torch_compile_first_chunk"] == (
        "enable_torch_compile_first_chunk"
    )
    assert runtime_arg_map["code2wav_odeint_method"] == "odeint_method"
    assert runtime_arg_map["odeint_method"] == "odeint_method"
    assert runtime_arg_map["code2wav_odeint_method_relaxed"] == (
        "odeint_method_relaxed"
    )
    assert runtime_arg_map["odeint_method_relaxed"] == "odeint_method_relaxed"
    assert runtime_arg_map["code2wav_batched_chunk"] == "batched_chunk"
    assert runtime_arg_map["batched_chunk"] == "batched_chunk"
    assert runtime_arg_map["code2wav_frequency"] == "frequency"
    assert runtime_arg_map["frequency"] == "frequency"
    assert runtime_arg_map["code2wav_dit_quant"] == "dit_quant"
    assert runtime_arg_map["dit_quant"] == "dit_quant"


def test_qwen35_omni_colocated_topology_places_gpu_stages_together():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")

    assert _stage(config, "image_encoder").gpu == 0
    assert _stage(config, "audio_encoder").gpu == 0
    assert _stage(config, "thinker").gpu == 0
    assert _stage(config, "talker_ar").gpu == 0
    assert _stage(config, "code2wav").gpu == 0
