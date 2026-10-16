# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager, resolve_config_cls_for_model_path
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.schema import StageRuntimeConfig
from sglang_omni.models.qwen3_5_omni.components.code2wav_scheduler import (
    QWEN35_CODE2WAV_STREAM_CHUNK_SIZE,
)
from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_MAX_RUNNING_REQUESTS,
    Qwen35OmniPipelineConfig,
    Qwen35OmniSpeechColocatedPipelineConfig,
    Qwen35OmniSpeechPipelineConfig,
)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _write_minimal_config(
    path,
    architecture: str | None,
    *,
    model_type: str = "qwen3_omni_next",
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {"model_type": model_type}
    if architecture is not None:
        config["architectures"] = [architecture]
    (path / "config.json").write_text(json.dumps(config))


def test_resolve_qwen35_config_from_raw_architecture(tmp_path):
    _write_minimal_config(tmp_path, "Qwen3OmniNextForConditionalGeneration")

    assert resolve_config_cls_for_model_path(str(tmp_path)) is (
        Qwen35OmniSpeechPipelineConfig
    )


def test_resolve_qwen35_config_from_thinker_alias(tmp_path):
    _write_minimal_config(tmp_path, "Qwen3OmniNextThinkerForConditionalGeneration")

    assert resolve_config_cls_for_model_path(str(tmp_path)) is (
        Qwen35OmniSpeechPipelineConfig
    )


def test_resolve_qwen35_config_from_model_type_without_architectures(tmp_path):
    _write_minimal_config(tmp_path, None)

    assert resolve_config_cls_for_model_path(str(tmp_path)) is (
        Qwen35OmniSpeechPipelineConfig
    )


def test_resolve_qwen35_config_from_split_thinker_model_type(tmp_path):
    _write_minimal_config(
        tmp_path,
        None,
        model_type="qwen3_omni_next_thinker",
    )

    assert resolve_config_cls_for_model_path(str(tmp_path)) is (
        Qwen35OmniSpeechPipelineConfig
    )


def test_resolve_qwen35_config_from_thinker_mtp_model_type(tmp_path):
    _write_minimal_config(
        tmp_path,
        None,
        model_type="qwen3_omni_next_thinker_mtp",
    )

    assert resolve_config_cls_for_model_path(str(tmp_path)) is (
        Qwen35OmniSpeechPipelineConfig
    )


def test_resolve_qwen35_config_from_thinker_mtp_architecture(tmp_path):
    _write_minimal_config(tmp_path, "Qwen3OmniNextThinkerMTP")

    assert resolve_config_cls_for_model_path(str(tmp_path)) is (
        Qwen35OmniSpeechPipelineConfig
    )


def test_qwen35_config_manager_variant_selection(tmp_path):
    _write_minimal_config(tmp_path, "Qwen3OmniNextForConditionalGeneration")

    manager = ConfigManager.from_model_path(str(tmp_path), variant="text")
    assert isinstance(manager.config, Qwen35OmniPipelineConfig)

    manager = ConfigManager.from_model_path(str(tmp_path), variant="speech-colocated")
    assert isinstance(manager.config, Qwen35OmniSpeechColocatedPipelineConfig)


def test_qwen35_h20_colocated_yaml_loads_stage_memory_budgets():
    manager = ConfigManager.from_file(
        "examples/configs/qwen3_5_omni_colocated_h20.yaml"
    )
    config = manager.config

    assert isinstance(config, Qwen35OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-5-omni-colocated-h20"
    assert config.model_path == "Qwen/Qwen3.5-Omni"

    by_stage = {stage.name: stage for stage in config.stages}
    assert set(by_stage) >= {
        "image_encoder",
        "audio_encoder",
        "thinker",
        "talker_ar",
        "code2wav",
    }
    assert (
        by_stage["thinker"].runtime.resources.total_gpu_memory_fraction
        == 0.75
    )
    assert (
        by_stage["thinker"].runtime.sglang_server_args.max_running_requests
        == QWEN3_5_OMNI_MAX_RUNNING_REQUESTS
    )
    assert by_stage["preprocessing"].runtime.image_max_pixels == 401408
    assert by_stage["preprocessing"].runtime.video_fps == 2
    assert by_stage["preprocessing"].runtime.video_max_frames == 128
    assert by_stage["preprocessing"].runtime.video_min_frames == 4
    assert by_stage["preprocessing"].runtime.video_max_pixels == 401408
    assert by_stage["preprocessing"].runtime.audio_target_sr == 16000
    assert by_stage["preprocessing"].runtime.audio_timestamp_interval == 60
    assert by_stage["preprocessing"].runtime.audio_downsample_times == 4
    assert by_stage["preprocessing"].runtime.audio_downsample_chunk_size == 100
    assert (
        by_stage["talker_ar"].runtime.resources.total_gpu_memory_fraction
        == 0.12
    )
    assert (
        by_stage["talker_ar"].runtime.sglang_server_args.max_running_requests
        == QWEN3_5_OMNI_MAX_RUNNING_REQUESTS
    )
    code2wav_runtime = by_stage["code2wav"].runtime
    assert code2wav_runtime.code2wav_stream_chunk_size == 8
    assert code2wav_runtime.send_chunk_size == 8
    assert code2wav_runtime.code2wav_enable_torch_compile is True
    assert code2wav_runtime.code2wav_odeint_method == "rk4"
    assert code2wav_runtime.code2wav_odeint_method_relaxed is True
    assert code2wav_runtime.code2wav_frequency == "50hz"
    assert code2wav_runtime.code2wav_dit_quant == "fp8"
    assert QWEN35_CODE2WAV_STREAM_CHUNK_SIZE == 4
    assert (
        Path("examples/configs/qwen3_5_omni_colocated_h20.yaml")
        .read_text()
        .startswith("# SPDX-License-Identifier: Apache-2.0")
    )


def test_qwen35_preprocessing_runtime_video_args_resolve_to_factory_args():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    preprocessing = _stage(config, "preprocessing")
    preprocessing.runtime.image_min_pixels = 100352
    preprocessing.runtime.image_max_pixels = 401408
    preprocessing.runtime.video_fps = 2.0
    preprocessing.runtime.video_max_frames = 128
    preprocessing.runtime.video_min_frames = 4
    preprocessing.runtime.video_min_pixels = 4096
    preprocessing.runtime.video_max_pixels = 401408
    preprocessing.runtime.video_total_pixels = 32768 * 768
    preprocessing.runtime.video_seconds_per_chunk = 2.0
    preprocessing.runtime.video_position_id_per_seconds = 25.0
    preprocessing.runtime.audio_target_sr = 16000
    preprocessing.runtime.audio_timestamp_interval = 30
    preprocessing.runtime.audio_downsample_times = 4
    preprocessing.runtime.audio_downsample_chunk_size = 100

    args = resolve_stage_factory_args(preprocessing, config)

    assert args["image_min_pixels"] == 100352
    assert args["image_max_pixels"] == 401408
    assert args["video_fps"] == 2.0
    assert args["video_max_frames"] == 128
    assert args["video_min_frames"] == 4
    assert args["video_min_pixels"] == 4096
    assert args["video_max_pixels"] == 401408
    assert args["video_total_pixels"] == 32768 * 768
    assert args["video_seconds_per_chunk"] == 2.0
    assert args["video_position_id_per_seconds"] == 25.0
    assert args["audio_target_sr"] == 16000
    assert args["audio_timestamp_interval"] == 30
    assert args["audio_downsample_times"] == 4
    assert args["audio_downsample_chunk_size"] == 100


def test_qwen35_preprocessing_runtime_audio_sampling_alias_resolves():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    preprocessing = _stage(config, "preprocessing")
    preprocessing.runtime = StageRuntimeConfig(audio_sampling_rate=22050)

    args = resolve_stage_factory_args(preprocessing, config)

    assert preprocessing.runtime.audio_target_sr == 22050
    assert args["audio_target_sr"] == 22050


def test_qwen35_preprocessing_runtime_audio_processor_short_aliases_resolve():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    preprocessing = _stage(config, "preprocessing")
    preprocessing.runtime = StageRuntimeConfig(
        timestamp_interval=45,
        downsample_times=3,
        downsample_chunk_size=80,
    )

    args = resolve_stage_factory_args(preprocessing, config)

    assert preprocessing.runtime.audio_timestamp_interval == 45
    assert preprocessing.runtime.audio_downsample_times == 3
    assert preprocessing.runtime.audio_downsample_chunk_size == 80
    assert args["audio_timestamp_interval"] == 45
    assert args["audio_downsample_times"] == 3
    assert args["audio_downsample_chunk_size"] == 80


def test_qwen35_preprocessing_runtime_audio_processor_short_alias_mutation_resolves():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    preprocessing = _stage(config, "preprocessing")
    preprocessing.runtime.timestamp_interval = 45
    preprocessing.runtime.downsample_times = 3
    preprocessing.runtime.downsample_chunk_size = 80

    args = resolve_stage_factory_args(preprocessing, config)

    assert args["audio_timestamp_interval"] == 45
    assert args["audio_downsample_times"] == 3
    assert args["audio_downsample_chunk_size"] == 80


def test_qwen35_code2wav_runtime_send_chunk_alias_resolves():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    code2wav = _stage(config, "code2wav")
    code2wav.runtime = StageRuntimeConfig(send_chunk_size=12)

    args = resolve_stage_factory_args(code2wav, config)

    assert code2wav.runtime.code2wav_stream_chunk_size == 12
    assert args["stream_chunk_size"] == 12


def test_qwen35_code2wav_runtime_send_chunk_mutation_resolves():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    code2wav = _stage(config, "code2wav")
    code2wav.runtime.send_chunk_size = 12

    args = resolve_stage_factory_args(code2wav, config)

    assert args["stream_chunk_size"] == 12


def test_qwen35_code2wav_perf_runtime_args_resolve():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    code2wav = _stage(config, "code2wav")
    code2wav.runtime = StageRuntimeConfig(
        code2wav_codec_eos_token_id=3000,
        code2wav_sample_rate=48000,
        code2wav_left_context_size=0,
        code2wav_enable_dynamic_chunk=True,
        code2wav_dynamic_chunk_sizes="2,4,8",
        code2wav_dynamic_chunk_steps=(8, 4, 1),
        code2wav_enable_torch_compile=False,
        enable_torch_compile_first_chunk=True,
        odeint_method="RK4",
        odeint_method_relaxed=True,
        batched_chunk=2,
        frequency="50HZ",
        dit_quant="FP8",
    )

    args = resolve_stage_factory_args(code2wav, config)

    assert args["enable_torch_compile"] is False
    assert args["enable_torch_compile_first_chunk"] is True
    assert args["codec_eos_token_id"] == 3000
    assert args["sample_rate"] == 48000
    assert args["left_context_size"] == 0
    assert args["enable_dynamic_chunk"] is True
    assert args["dynamic_chunk_sizes"] == (2, 4, 8)
    assert args["dynamic_chunk_steps"] == (8, 4, 1)
    assert args["odeint_method"] == "RK4"
    assert args["odeint_method_relaxed"] is True
    assert args["batched_chunk"] == 2
    assert args["frequency"] == "50HZ"
    assert args["dit_quant"] == "FP8"


def test_qwen35_code2wav_perf_runtime_mutation_resolves():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    code2wav = _stage(config, "code2wav")
    code2wav.runtime.code2wav_odeint_method = "euler"
    code2wav.runtime.code2wav_frequency = "25hz"
    code2wav.runtime.code2wav_batched_chunk = 1
    code2wav.runtime.code2wav_enable_dynamic_chunk = True
    code2wav.runtime.code2wav_dynamic_chunk_sizes = (2, 4)
    code2wav.runtime.code2wav_dynamic_chunk_steps = (1, 1)

    args = resolve_stage_factory_args(code2wav, config)

    assert args["odeint_method"] == "euler"
    assert args["frequency"] == "25hz"
    assert args["batched_chunk"] == 1
    assert args["enable_dynamic_chunk"] is True
    assert args["dynamic_chunk_sizes"] == (2, 4)
    assert args["dynamic_chunk_steps"] == (1, 1)


def test_qwen35_code2wav_dynamic_batch_alias_resolves():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    code2wav = _stage(config, "code2wav")
    code2wav.runtime = StageRuntimeConfig(code2wav_dynamic_batch=True)

    args = resolve_stage_factory_args(code2wav, config)

    assert code2wav.runtime.code2wav_enable_dynamic_chunk is True
    assert args["enable_dynamic_chunk"] is True


def test_stage_runtime_rejects_conflicting_audio_sampling_aliases():
    with pytest.raises(ValueError, match="audio sampling rate aliases disagree"):
        StageRuntimeConfig(audio_target_sr=16000, sampling_rate=24000)


def test_stage_runtime_rejects_conflicting_audio_processor_aliases():
    with pytest.raises(ValueError, match="processor timestamp interval"):
        StageRuntimeConfig(audio_timestamp_interval=30, timestamp_interval=60)

    with pytest.raises(ValueError, match="processor downsample times"):
        StageRuntimeConfig(audio_downsample_times=4, downsample_times=5)

    with pytest.raises(ValueError, match="processor downsample chunk size"):
        StageRuntimeConfig(
            audio_downsample_chunk_size=100,
            downsample_chunk_size=80,
        )


def test_stage_runtime_rejects_conflicting_code2wav_chunk_aliases():
    with pytest.raises(ValueError, match="code2wav stream chunk size"):
        StageRuntimeConfig(
            code2wav_stream_chunk_size=8,
            send_chunk_size=12,
        )

    with pytest.raises(ValueError, match="code2wav dynamic chunk"):
        StageRuntimeConfig(
            code2wav_enable_dynamic_chunk=True,
            code2wav_dynamic_batch=False,
        )


def test_stage_runtime_rejects_conflicting_code2wav_perf_aliases():
    with pytest.raises(ValueError, match="code2wav odeint method"):
        StageRuntimeConfig(code2wav_odeint_method="rk4", odeint_method="euler")

    with pytest.raises(ValueError, match="code2wav odeint method relaxed"):
        StageRuntimeConfig(
            code2wav_odeint_method_relaxed=True,
            odeint_method_relaxed=False,
        )

    with pytest.raises(ValueError, match="code2wav batched chunk"):
        StageRuntimeConfig(code2wav_batched_chunk=1, batched_chunk=2)

    with pytest.raises(ValueError, match="code2wav frequency"):
        StageRuntimeConfig(code2wav_frequency="25hz", frequency="50hz")

    with pytest.raises(ValueError, match="code2wav dit quant"):
        StageRuntimeConfig(code2wav_dit_quant="fp8", dit_quant="nf4")


def test_stage_runtime_rejects_invalid_code2wav_runtime_values():
    with pytest.raises(ValueError, match="code2wav_sample_rate"):
        StageRuntimeConfig(code2wav_sample_rate=0)

    with pytest.raises(ValueError, match="code2wav_codec_eos_token_id"):
        StageRuntimeConfig(code2wav_codec_eos_token_id=-1)

    with pytest.raises(ValueError, match="code2wav_left_context_size"):
        StageRuntimeConfig(code2wav_left_context_size=-1)

    with pytest.raises(ValueError, match="dynamic chunk sizes"):
        StageRuntimeConfig(code2wav_dynamic_chunk_sizes="2,0")

    with pytest.raises(ValueError, match="dynamic chunk steps"):
        StageRuntimeConfig(code2wav_dynamic_chunk_steps=())


def test_qwen35_typed_sglang_runtime_args_resolve_to_factory_args():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")
    thinker = _stage(config, "thinker")
    thinker.runtime.sglang_server_args.mem_fraction_static = 0.42
    thinker.runtime.sglang_server_args.max_running_requests = 24

    args = resolve_stage_factory_args(thinker, config)

    overrides = args["server_args_overrides"]
    assert overrides["mem_fraction_static"] == 0.42
    assert overrides["max_running_requests"] == 24
