# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
import typer

serve_module = importlib.import_module("sglang_omni.cli.serve")

from sglang_omni.cli.serve import (
    _parse_single_gpu_visible_device,
    _run_qwen35_preflight_or_raise,
    _resolve_toggle_alias_flags,
    _validate_colocate_config,
    _validate_qwen35_disable_mtp_request,
    _validate_qwen35_kv_transfer_request,
    _validate_qwen35_mamba_cache_mode_request,
    _validate_qwen35_max_seq_len_to_capture_request,
    _validate_qwen35_speculative_config_request,
    _validate_qwen35_vllm_engine_profile_request,
    apply_code2wav_cli_overrides,
    apply_limit_mm_per_prompt_cli_override,
    apply_max_running_requests_cli_overrides,
    apply_parallelism_cli_overrides,
    apply_partial_start_cli_overrides,
    apply_preprocessing_video_cli_overrides,
    apply_qwen35_max_mm_len_cli_override,
    apply_qwen35_max_model_len_cli_override,
    apply_talker_server_args_cli_overrides,
    apply_talker_model_path_cli_override,
    apply_vllm_ar_server_args_cli_overrides,
    serve,
)
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.qwen3_5_omni.config import (
    Qwen35OmniPipelineConfig,
    Qwen35OmniSpeechColocatedPipelineConfig,
    Qwen35OmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_5_omni.preflight import PreflightReport


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _serve_option_names(param_name: str) -> set[str]:
    annotation = get_type_hints(serve, include_extras=True)[param_name]
    for item in get_args(annotation)[1:]:
        param_decls = getattr(item, "param_decls", None)
        if param_decls is not None:
            names = set(param_decls)
            # Typer stores the first Annotated Option name in ``default`` and
            # the remaining aliases in ``param_decls``.
            default = getattr(item, "default", None)
            if isinstance(default, str) and default.startswith("--"):
                names.add(default)
            return names
    raise AssertionError(f"{param_name} has no Typer option metadata")


class _FakeServeConfigManager:
    def __init__(self, config):
        self.config = config

    def parse_extra_args(self, args):
        assert args == []
        return {}

    def merge_config(self, extra_args):
        assert extra_args == {}
        return self.config


def _patch_serve_config_manager(monkeypatch, config):
    def fake_from_model_path(cls, model_path, *, variant=None):
        assert model_path == "/models/qwen35"
        assert variant == "text"
        return _FakeServeConfigManager(config)

    monkeypatch.setattr(
        serve_module.ConfigManager,
        "from_model_path",
        classmethod(fake_from_model_path),
    )


def _patch_serve_model_path_config_manager(
    monkeypatch,
    config,
    *,
    expected_model_path="/models/qwen35",
    expected_variant=None,
):
    def fake_from_model_path(cls, model_path, *, variant=None):
        assert model_path == expected_model_path
        assert variant == expected_variant
        return _FakeServeConfigManager(config)

    monkeypatch.setattr(
        serve_module.ConfigManager,
        "from_model_path",
        classmethod(fake_from_model_path),
    )


def test_qwen35_cli_colocate_accepts_colocated_config():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")

    _validate_colocate_config(config)


def test_qwen35_cli_disable_mtp_accepts_qwen35_config():
    _validate_qwen35_disable_mtp_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        disable_mtp=True,
    )


def test_qwen35_cli_disable_mtp_rejects_non_qwen35_config():
    class OtherConfig(Qwen35OmniSpeechPipelineConfig):
        pass

    with pytest.raises(typer.BadParameter, match="disable-mtp"):
        _validate_qwen35_disable_mtp_request(
            OtherConfig(model_path="dummy"),
            disable_mtp=True,
        )


def test_qwen35_cli_max_seq_len_to_capture_accepts_qwen35_config():
    _validate_qwen35_max_seq_len_to_capture_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        max_seq_len_to_capture=262144,
    )


def test_qwen35_cli_max_seq_len_to_capture_rejects_bad_value():
    with pytest.raises(typer.BadParameter, match="max-seq-len-to-capture"):
        _validate_qwen35_max_seq_len_to_capture_request(
            Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
            max_seq_len_to_capture=0,
        )


def test_qwen35_cli_max_seq_len_to_capture_rejects_non_qwen35_config():
    class OtherConfig(Qwen35OmniSpeechPipelineConfig):
        pass

    with pytest.raises(typer.BadParameter, match="max-seq-len-to-capture"):
        _validate_qwen35_max_seq_len_to_capture_request(
            OtherConfig(model_path="dummy"),
            max_seq_len_to_capture=262144,
        )


def test_qwen35_cli_mamba_cache_mode_accepts_none():
    _validate_qwen35_mamba_cache_mode_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        mamba_cache_mode="none",
    )


def test_qwen35_cli_mamba_cache_mode_rejects_light():
    with pytest.raises(typer.BadParameter, match="mamba-cache-mode"):
        _validate_qwen35_mamba_cache_mode_request(
            Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
            mamba_cache_mode="light",
        )


def test_qwen35_cli_mamba_cache_mode_rejects_non_qwen35_config():
    class OtherConfig(Qwen35OmniSpeechPipelineConfig):
        pass

    with pytest.raises(typer.BadParameter, match="mamba-cache-mode"):
        _validate_qwen35_mamba_cache_mode_request(
            OtherConfig(model_path="dummy"),
            mamba_cache_mode="none",
        )


def test_qwen35_cli_kv_transfer_accepts_noop_values():
    _validate_qwen35_kv_transfer_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        enable_disaggregated_prefilling="0",
        kv_transfer_config="{}",
    )


def test_qwen35_cli_kv_transfer_rejects_nonempty_config():
    with pytest.raises(typer.BadParameter, match="kv-transfer-config"):
        _validate_qwen35_kv_transfer_request(
            Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
            enable_disaggregated_prefilling=None,
            kv_transfer_config='{"kv_connector":"HybridConnector"}',
        )


def test_qwen35_cli_kv_transfer_rejects_disaggregated_prefill():
    with pytest.raises(typer.BadParameter, match="disaggregated-prefilling"):
        _validate_qwen35_kv_transfer_request(
            Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
            enable_disaggregated_prefilling="1",
            kv_transfer_config=None,
        )


def test_qwen35_cli_vllm_engine_profile_accepts_noops():
    _validate_qwen35_vllm_engine_profile_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        distributed_executor_backend="mp",
        kv_cache_dtype="auto",
        enable_expert_parallel="0",
        mm_processor_cache_gb=0,
        max_mm_len=None,
        use_omni_engine="1",
        use_omni_rpc_engine=None,
        use_zero_shot="true",
        skip_mm_profiling="true",
        override_video_max_pixels="false",
        is_thinker="true",
        thinker_only_marker="false",
        text_only=False,
    )


def test_qwen35_cli_vllm_engine_profile_rejects_kv_cache_dtype():
    with pytest.raises(typer.BadParameter, match="kv-cache-dtype"):
        _validate_qwen35_vllm_engine_profile_request(
            Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
            distributed_executor_backend=None,
            kv_cache_dtype="tq4",
            enable_expert_parallel=None,
            mm_processor_cache_gb=None,
            max_mm_len=None,
            use_omni_engine=None,
            use_omni_rpc_engine=None,
            use_zero_shot=None,
            skip_mm_profiling=None,
            override_video_max_pixels=None,
            is_thinker=None,
            thinker_only_marker=None,
            text_only=False,
        )


def test_qwen35_cli_vllm_engine_profile_accepts_max_mm_len():
    _validate_qwen35_vllm_engine_profile_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        distributed_executor_backend=None,
        kv_cache_dtype=None,
        enable_expert_parallel=None,
        mm_processor_cache_gb=None,
        max_mm_len=256000,
        use_omni_engine=None,
        use_omni_rpc_engine=None,
        use_zero_shot=None,
        skip_mm_profiling=None,
        override_video_max_pixels=None,
        is_thinker=None,
        thinker_only_marker=None,
        text_only=False,
    )


def test_qwen35_cli_empty_speculative_config_is_noop():
    _validate_qwen35_speculative_config_request(
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        speculative_config="{}",
    )


def test_qwen35_cli_speculative_config_rejects_mtp():
    with pytest.raises(typer.BadParameter, match="speculative-config"):
        _validate_qwen35_speculative_config_request(
            Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
            speculative_config=(
                '{"method":"qwen3_omni_next_thinker_mtp",'
                '"num_speculative_tokens":4}'
            ),
        )


def test_qwen35_cli_split_path_aliases_match_example_launcher():
    assert "--model" in _serve_option_names("model_path")
    assert "--thinker-model" in _serve_option_names("thinker_model_path")
    assert "--thinker-model-path" in _serve_option_names("thinker_model_path")
    assert "--vllm-profile" in _serve_option_names("vllm_profile")
    assert "--preflight" in _serve_option_names("preflight")
    assert "--serve-port" in _serve_option_names("port")
    assert "--serve_port" in _serve_option_names("port")
    assert "--talker-model-path" in _serve_option_names("talker_model_path")
    assert "--talker-path" in _serve_option_names("talker_model_path")
    assert "--talker-model" in _serve_option_names("talker_model_path")
    assert "--code2wav-model-path" in _serve_option_names("code2wav_model_path")
    assert "--code2wav-path" in _serve_option_names("code2wav_model_path")
    assert "--code2wav-model-folder" in _serve_option_names(
        "code2wav_model_folder"
    )
    assert "--code2wav-model" in _serve_option_names("code2wav_model_folder")
    assert "--send-chunk-size" in _serve_option_names(
        "code2wav_stream_chunk_size"
    )
    assert "--sample-rate" in _serve_option_names("code2wav_sample_rate")
    assert "--xvector-info-path" in _serve_option_names("xvector_info_path")
    assert "--voice-clone-info-path" in _serve_option_names("xvector_info_path")
    assert "--voice-clone-path" in _serve_option_names("xvector_info_path")
    assert "--disable-mtp" in _serve_option_names("disable_mtp")
    assert "--code2wav-frequency" in _serve_option_names("code2wav_frequency")
    assert "--code2wav-dit-quantization" in _serve_option_names(
        "code2wav_dit_quant"
    )
    assert "--limit-mm-per-prompt" in _serve_option_names("limit_mm_per_prompt")
    assert "--limit-mm-per-prompt-image" in _serve_option_names(
        "limit_mm_per_prompt_image"
    )
    assert "--max-tokens" in _serve_option_names("max_tokens")
    assert "--seed" in _serve_option_names("seed")
    assert "--temperature" in _serve_option_names("temperature")
    assert "--top-p" in _serve_option_names("top_p")
    assert "--top-k" in _serve_option_names("top_k")
    assert "--repetition-penalty" in _serve_option_names("repetition_penalty")
    assert "--frequency-penalty" in _serve_option_names("frequency_penalty")
    assert "--presence-penalty" in _serve_option_names("presence_penalty")
    assert "--voice-type" in _serve_option_names("voice_type")
    assert "--enable-tn" in _serve_option_names("enable_tn")
    assert "--enable-text-normalization" in _serve_option_names("enable_tn")
    assert "--disable-tn" in _serve_option_names("disable_tn")
    assert "--no-enable-tn" in _serve_option_names("disable_tn")
    assert "--gpu-memory-utilization" in _serve_option_names("mem_fraction_static")
    assert "--talker-gpu-memory-utilization" in _serve_option_names(
        "talker_mem_fraction_static"
    )
    assert "--talker-quantization" in _serve_option_names("talker_quantization")
    assert "--thinker-visible-devices" in _serve_option_names(
        "thinker_visible_devices"
    )
    assert "--thinker-devices" in _serve_option_names("thinker_visible_devices")
    assert "--talker-visible-devices" in _serve_option_names(
        "talker_visible_devices"
    )
    assert "--talker-devices" in _serve_option_names("talker_visible_devices")
    assert "--code2wav-visible-devices" in _serve_option_names(
        "code2wav_visible_devices"
    )
    assert "--code2wav-devices" in _serve_option_names("code2wav_visible_devices")


def test_qwen35_cli_vllm_direct_aliases_parse_as_profile_defaults():
    defaults = serve_module._parse_qwen35_vllm_profile_defaults(
        (
            "--model",
            "/models/qwen35",
            "--thinker-model",
            "/models/qwen35/thinker",
            "--talker-model",
            "/models/qwen35/talker_lm",
            "--code2wav-model",
            "code2wav",
            "--thinker-devices",
            "[0,1]",
            "--talker-devices",
            "[2]",
            "--code2wav-devices",
            "[3]",
            "--thinker-tensor-parallel-size",
            "2",
            "--host",
            "127.0.0.1",
            "--serve-port",
            "29000",
            "--video-fps",
            "2",
            "--sample-rate",
            "24000",
            "--max-tokens",
            "512",
            "--seed",
            "0",
            "--voice-type",
            "Cherry",
            "--talker-quantization",
            "fp8",
            "--enable-tn",
        )
    )

    assert defaults == {
        "model_path": "/models/qwen35",
        "thinker_model_path": "/models/qwen35/thinker",
        "talker_model_path": "/models/qwen35/talker_lm",
        "code2wav_model_folder": "code2wav",
        "thinker_visible_devices": "[0,1]",
        "talker_visible_devices": "[2]",
        "code2wav_visible_devices": "[3]",
        "thinker_tp_size": 2,
        "host": "127.0.0.1",
        "port": 29000,
        "video_fps": 2.0,
        "code2wav_sample_rate": 24000,
        "max_tokens": 512,
        "seed": 0,
        "voice_type": "Cherry",
        "talker_quantization": "fp8",
        "enable_tn": True,
    }


def test_qwen35_cli_thinker_model_alias_is_noop_with_root_model(monkeypatch):
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(monkeypatch, config)
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        model_path="/models/qwen35",
        thinker_model_path="/models/qwen35/thinker",
    )

    assert captured["pipeline_config"].model_path == "/models/qwen35"


def test_qwen35_cli_thinker_model_alias_can_supply_model_path(monkeypatch):
    config = Qwen35OmniPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(
        monkeypatch,
        config,
        expected_model_path="/models/qwen35/thinker",
    )
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        thinker_model_path="/models/qwen35/thinker",
    )

    assert captured["pipeline_config"].model_path == "/models/qwen35/thinker"


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model_path": "/models/qwen35",
            "thinker_model_path": "/other/qwen35/thinker",
        },
        {
            "config": "examples/configs/qwen3_5_omni_colocated_h20.yaml",
            "thinker_model_path": "/models/qwen35/thinker",
        },
    ],
)
def test_qwen35_cli_thinker_model_alias_rejects_ambiguous_paths(
    monkeypatch,
    kwargs,
):
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda *args, **kwargs: pytest.fail("launch_server should not run"),
    )

    with pytest.raises(typer.BadParameter, match="thinker-model"):
        serve_module.serve(SimpleNamespace(args=[]), **kwargs)


def test_qwen35_cli_vllm_profile_defaults_apply_to_speech(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "h20.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/qwen35",
                    "host": "127.0.0.1",
                    "dtype": "bfloat16",
                    "max_model_len": 192000,
                    "max_num_batched_tokens": 512,
                    "max_num_seqs": 24,
                    "max_tokens": 768,
                    "seed": 3,
                    "gpu_memory_utilization": 0.6,
                    "talker_gpu_memory_utilization": 0.8,
                    "talker_quantization": "fp8",
                    "talker_visible_devices": [1],
                    "code2wav_visible_devices": [1],
                    "code2wav_model_folder": "code2wav",
                    "send_chunk_size": 8,
                    "omni_video_fps": 2,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                    "limit_mm_per_prompt": {
                        "audio": 960,
                        "image": 960,
                        "video": 960,
                    },
                }
            }
        )
    )
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(monkeypatch, config)
    captured = {}

    def fake_launch_server(pipeline_config, **kwargs):
        captured["pipeline_config"] = pipeline_config
        captured.update(kwargs)

    monkeypatch.setattr(serve_module, "launch_server", fake_launch_server)

    serve_module.serve(
        SimpleNamespace(args=[]),
        vllm_profile=str(profile),
    )

    merged = captured["pipeline_config"]
    assert merged.model_path == "/models/qwen35"
    assert _stage(merged, "talker_ar").gpu == 1
    assert _stage(merged, "code2wav").gpu == 1
    assert _stage(merged, "thinker").runtime.max_seq_len == 192000
    assert (
        _stage(merged, "thinker").runtime.sglang_server_args.max_running_requests
        == 24
    )
    assert (
        _stage(merged, "talker_ar").runtime.sglang_server_args.max_running_requests
        == 24
    )
    thinker_args = _stage(merged, "thinker").factory_args["server_args_overrides"]
    talker_args = _stage(merged, "talker_ar").factory_args["server_args_overrides"]
    assert thinker_args["dtype"] == "bfloat16"
    assert talker_args["dtype"] == "bfloat16"
    assert talker_args["quantization"] == "fp8"
    assert _stage(merged, "thinker").runtime.sglang_server_args.mem_fraction_static == 0.6
    assert _stage(merged, "talker_ar").runtime.sglang_server_args.mem_fraction_static == 0.8
    assert thinker_args["disable_radix_cache"] is False
    assert talker_args["disable_radix_cache"] is False
    assert thinker_args["max_prefill_tokens"] == 512
    assert talker_args["max_prefill_tokens"] == 512
    assert _stage(merged, "code2wav").factory_args["code2wav_model_path"] == (
        "/models/qwen35/code2wav"
    )
    assert resolve_stage_factory_args(_stage(merged, "code2wav"), merged)[
        "stream_chunk_size"
    ] == 8
    assert _stage(merged, "preprocessing").factory_args[
        "limit_mm_per_prompt"
    ] == {
        "audio": 960,
        "image": 960,
        "video": 960,
    }
    assert _stage(merged, "preprocessing").runtime.video_fps == 2.0
    assert captured["host"] == "127.0.0.1"
    assert captured["default_generation_params"]["max_tokens"] == 768
    assert captured["default_generation_params"]["seed"] == 3


def test_qwen35_cli_vllm_profile_disable_mtp_override_allows_mtp_profile(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "mtp.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/qwen35",
                    "speculative_config": {
                        "method": "qwen3_omni_next_thinker_mtp",
                        "num_speculative_tokens": 4,
                    },
                }
            }
        )
    )
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(monkeypatch, config)
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        vllm_profile=str(profile),
        disable_mtp=True,
    )

    assert captured["pipeline_config"].model_path == "/models/qwen35"


def test_qwen35_cli_vllm_profile_allows_cli_overrides(monkeypatch, tmp_path):
    profile = tmp_path / "h20.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/from-profile",
                    "talker_visible_devices": [1],
                    "code2wav_model_folder": "code2wav",
                    "send_chunk_size": 8,
                }
            }
        )
    )
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(
        monkeypatch,
        config,
        expected_model_path="/models/from-cli",
    )
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        vllm_profile=str(profile),
        model_path="/models/from-cli",
        talker_gpu=2,
        code2wav_stream_chunk_size=16,
    )

    merged = captured["pipeline_config"]
    assert merged.model_path == "/models/from-cli"
    assert _stage(merged, "talker_ar").gpu == 2
    assert _stage(merged, "code2wav").factory_args["code2wav_model_path"] == (
        "/models/from-cli/code2wav"
    )
    assert resolve_stage_factory_args(_stage(merged, "code2wav"), merged)[
        "stream_chunk_size"
    ] == 16


def test_qwen35_cli_vllm_profile_can_select_text_only(monkeypatch, tmp_path):
    profile = tmp_path / "thinker.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/qwen35",
                    "thinker_only": True,
                    "dtype": "bfloat16",
                    "max_model_len": 192000,
                    "talker_visible_devices": [1],
                    "code2wav_model_folder": "code2wav",
                }
            }
        )
    )
    config = Qwen35OmniPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(
        monkeypatch,
        config,
        expected_variant="text",
    )
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        vllm_profile=str(profile),
    )

    merged = captured["pipeline_config"]
    assert isinstance(merged, Qwen35OmniPipelineConfig)
    assert merged.model_path == "/models/qwen35"
    assert _stage(merged, "thinker").runtime.max_seq_len == 192000
    assert (
        _stage(merged, "thinker")
        .factory_args["server_args_overrides"]["dtype"]
        == "bfloat16"
    )


def test_qwen35_cli_vllm_profile_rejects_unsupported(monkeypatch, tmp_path):
    profile = tmp_path / "bad.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "mamba_cache_mode": "light",
                }
            }
        )
    )
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda *args, **kwargs: pytest.fail("launch_server should not run"),
    )

    with pytest.raises(typer.BadParameter, match="profile preflight FAIL"):
        serve_module.serve(
            SimpleNamespace(args=[]),
            vllm_profile=str(profile),
            model_path="/models/qwen35",
        )


def test_qwen35_cli_forwards_default_generation_params(monkeypatch):
    config = Qwen35OmniPipelineConfig(model_path="/models/from-config")
    _patch_serve_config_manager(monkeypatch, config)
    captured = {}

    def fake_launch_server(pipeline_config, **kwargs):
        captured["pipeline_config"] = pipeline_config
        captured.update(kwargs)

    monkeypatch.setattr(serve_module, "launch_server", fake_launch_server)

    serve_module.serve(
        SimpleNamespace(args=[]),
        model_path="/models/qwen35",
        text_only=True,
        max_tokens=512,
        seed=7,
    )

    assert captured["pipeline_config"].model_path == "/models/qwen35"
    assert captured["default_generation_params"] == {
        "max_tokens": 512,
        "seed": 7,
    }


def test_qwen35_cli_forwards_default_talker_params_for_speech(monkeypatch):
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(monkeypatch, config)
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(SimpleNamespace(args=[]), model_path="/models/qwen35")

    assert captured["pipeline_config"].model_path == "/models/qwen35"
    assert captured["default_talker_params"] == {"voice_type": "f245"}
    assert captured["default_generation_params"] == {
        "temperature": 0.000001,
        "top_k": 1,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": 2048,
        "seed": 0,
    }


def test_qwen35_cli_default_request_params_allow_cli_overrides(monkeypatch):
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/from-config")
    _patch_serve_model_path_config_manager(monkeypatch, config)
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        model_path="/models/qwen35",
        voice_type="Cherry",
        enable_tn=True,
        max_tokens=512,
        seed=7,
        temperature=0.2,
        top_p=0.7,
        top_k=4,
        repetition_penalty=1.2,
        frequency_penalty=0.1,
        presence_penalty=0.3,
    )

    assert captured["default_talker_params"] == {
        "voice_type": "Cherry",
        "enable_tn": True,
    }
    assert captured["default_generation_params"] == {
        "temperature": 0.2,
        "top_k": 4,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "presence_penalty": 0.3,
        "max_tokens": 512,
        "seed": 7,
        "frequency_penalty": 0.1,
    }


def test_qwen35_cli_text_only_does_not_inject_talker_defaults(monkeypatch):
    config = Qwen35OmniPipelineConfig(model_path="/models/from-config")
    _patch_serve_config_manager(monkeypatch, config)
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda pipeline_config, **kwargs: captured.update(
            {"pipeline_config": pipeline_config, **kwargs}
        ),
    )

    serve_module.serve(
        SimpleNamespace(args=[]),
        model_path="/models/qwen35",
        text_only=True,
    )

    assert captured["default_talker_params"] is None
    assert captured["default_generation_params"] is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_tokens": 0}, "--max-tokens"),
        ({"seed": -1}, "--seed"),
        ({"temperature": -0.1}, "--temperature"),
        ({"top_p": 0}, "--top-p"),
        ({"top_p": 1.1}, "--top-p"),
        ({"top_k": -2}, "--top-k"),
        ({"repetition_penalty": 0}, "--repetition-penalty"),
        ({"voice_type": ""}, "--voice-type"),
        ({"enable_tn": True, "disable_tn": True}, "--enable-tn"),
    ],
)
def test_qwen35_cli_rejects_bad_default_generation_params(
    monkeypatch,
    kwargs,
    match,
):
    _patch_serve_config_manager(
        monkeypatch,
        Qwen35OmniPipelineConfig(model_path="/models/from-config"),
    )
    monkeypatch.setattr(
        serve_module,
        "launch_server",
        lambda *args, **kwargs: pytest.fail("launch_server should not run"),
    )

    with pytest.raises(typer.BadParameter, match=match):
        serve_module.serve(
            SimpleNamespace(args=[]),
            model_path="/models/qwen35",
            text_only=True,
            **kwargs,
        )


def test_qwen35_cli_video_preprocessing_aliases_are_available():
    assert "--max-running-requests" in _serve_option_names("max_running_requests")
    assert "--max_running_requests" in _serve_option_names("max_running_requests")
    assert "--max-num-seqs" in _serve_option_names("max_running_requests")
    assert "--max_num_seqs" in _serve_option_names("max_running_requests")
    assert "--max-model-len" in _serve_option_names("max_model_len")
    assert "--max_model_len" in _serve_option_names("max_model_len")
    assert "--max-seq-len-to-capture" in _serve_option_names(
        "max_seq_len_to_capture"
    )
    assert "--max_seq_len_to_capture" in _serve_option_names(
        "max_seq_len_to_capture"
    )
    assert "--compilation-config" in _serve_option_names("compilation_config")
    assert "--compilation_config" in _serve_option_names("compilation_config")
    assert "--mamba-cache-mode" in _serve_option_names("mamba_cache_mode")
    assert "--mamba_cache_mode" in _serve_option_names("mamba_cache_mode")
    assert "--kv-transfer-config" in _serve_option_names("kv_transfer_config")
    assert "--kv_transfer_config" in _serve_option_names("kv_transfer_config")
    assert "--enable-disaggregated-prefilling" in _serve_option_names(
        "enable_disaggregated_prefilling"
    )
    assert "--enable_disaggregated_prefilling" in _serve_option_names(
        "enable_disaggregated_prefilling"
    )
    assert "--tensor-parallel-size" in _serve_option_names("tensor_parallel_size")
    assert "--tensor_parallel_size" in _serve_option_names("tensor_parallel_size")
    assert "--thinker-tensor-parallel-size" in _serve_option_names(
        "thinker_tp_size"
    )
    assert "--thinker_tensor_parallel_size" in _serve_option_names(
        "thinker_tp_size"
    )
    assert "--distributed-executor-backend" in _serve_option_names(
        "distributed_executor_backend"
    )
    assert "--kv-cache-dtype" in _serve_option_names("kv_cache_dtype")
    assert "--enable-expert-parallel" in _serve_option_names(
        "enable_expert_parallel"
    )
    assert "--max-mm-len" in _serve_option_names("max_mm_len")
    assert "--speculative-config" in _serve_option_names("speculative_config")
    assert "--mm-processor-cache-gb" in _serve_option_names(
        "mm_processor_cache_gb"
    )
    assert "--use-omni-rpc-engine" in _serve_option_names("use_omni_rpc_engine")
    assert "--thinker-only" in _serve_option_names("thinker_only_marker")
    assert "--video-needs-metadata" in _serve_option_names("video_needs_metadata")
    assert "--override-video-max-pixels" in _serve_option_names(
        "override_video_max_pixels"
    )
    assert "--max-num-batched-tokens" in _serve_option_names(
        "max_num_batched_tokens"
    )
    assert "--block-size" in _serve_option_names("page_size")
    assert "--page-size" in _serve_option_names("page_size")
    assert "--thinker-max-seq-len" in _serve_option_names("thinker_max_seq_len")
    assert (
        "--thinker-max-running-requests"
        in _serve_option_names("thinker_max_running_requests")
    )
    assert (
        "--talker-max-running-requests"
        in _serve_option_names("talker_max_running_requests")
    )
    assert "--image-min-pixels" in _serve_option_names("image_min_pixels")
    assert "--image_min_pixels" in _serve_option_names("image_min_pixels")
    assert "--image-max-pixels" in _serve_option_names("image_max_pixels")
    assert "--image_max_pixels" in _serve_option_names("image_max_pixels")
    assert "--video-fps" in _serve_option_names("video_fps")
    assert "--video_fps" in _serve_option_names("video_fps")
    assert "--video-max-frames" in _serve_option_names("video_max_frames")
    assert "--video_max_frames" in _serve_option_names("video_max_frames")
    assert "--video-min-frames" in _serve_option_names("video_min_frames")
    assert "--video_min_frames" in _serve_option_names("video_min_frames")
    assert "--video-max-pixels" in _serve_option_names("video_max_pixels")
    assert "--video_max_pixels" in _serve_option_names("video_max_pixels")
    assert "--video-seconds-per-chunk" in _serve_option_names(
        "video_seconds_per_chunk"
    )
    assert "--seconds-per-chunk" in _serve_option_names("video_seconds_per_chunk")
    assert "--video-position-id-per-seconds" in _serve_option_names(
        "video_position_id_per_seconds"
    )
    assert "--position-id-per-seconds" in _serve_option_names(
        "video_position_id_per_seconds"
    )
    assert "--audio-target-sr" in _serve_option_names("audio_target_sr")
    assert "--audio-sampling-rate" in _serve_option_names("audio_target_sr")
    assert "--sampling-rate" in _serve_option_names("audio_target_sr")
    assert "--audio-timestamp-interval" in _serve_option_names(
        "audio_timestamp_interval"
    )
    assert "--timestamp-interval" in _serve_option_names(
        "audio_timestamp_interval"
    )
    assert "--audio-downsample-times" in _serve_option_names(
        "audio_downsample_times"
    )
    assert "--downsample-times" in _serve_option_names("audio_downsample_times")
    assert "--audio-downsample-chunk-size" in _serve_option_names(
        "audio_downsample_chunk_size"
    )
    assert "--downsample-chunk-size" in _serve_option_names(
        "audio_downsample_chunk_size"
    )


def test_qwen35_cli_vllm_ar_aliases_are_available():
    assert "--thinker-quantization" in _serve_option_names("quantization")
    assert "--dtype" in _serve_option_names("dtype")
    assert "--thinker-dtype" in _serve_option_names("thinker_dtype")
    assert "--talker-dtype" in _serve_option_names("talker_dtype")
    assert "--mamba-cache-dtype" in _serve_option_names("mamba_ssm_dtype")
    assert "--mamba-ssm-dtype" in _serve_option_names("mamba_ssm_dtype")
    assert "--enforce-eager" in _serve_option_names("enforce_eager")
    assert "--thinker-enforce-eager" in _serve_option_names("thinker_enforce_eager")
    assert "--talker-enforce-eager" in _serve_option_names("talker_enforce_eager")
    assert "--enable-prefix-caching" in _serve_option_names("enable_prefix_caching")
    assert "--no-enable-prefix-caching" in _serve_option_names(
        "disable_prefix_caching"
    )
    assert "--enable-chunked-prefill" in _serve_option_names(
        "enable_chunked_prefill"
    )
    assert "--no-enable-chunked-prefill" in _serve_option_names(
        "disable_chunked_prefill"
    )


def test_qwen35_cli_code2wav_boolean_aliases_match_example_launcher():
    assert (
        "--code2wav-enable-torch-compile"
        in _serve_option_names("code2wav_enable_torch_compile")
    )
    assert (
        "--enable-torch-compile"
        in _serve_option_names("code2wav_enable_torch_compile")
    )
    assert (
        "--no-code2wav-enable-torch-compile"
        in _serve_option_names("code2wav_disable_torch_compile")
    )
    assert (
        "--no-enable-torch-compile"
        in _serve_option_names("code2wav_disable_torch_compile")
    )
    assert (
        "--no-code2wav-torch-compile"
        in _serve_option_names("code2wav_disable_torch_compile")
    )
    assert (
        "--code2wav-enable-dynamic-chunk"
        in _serve_option_names("code2wav_enable_dynamic_chunk")
    )
    assert (
        "--no-code2wav-enable-dynamic-chunk"
        in _serve_option_names("code2wav_disable_dynamic_chunk")
    )
    assert (
        "--no-code2wav-dynamic-chunk"
        in _serve_option_names("code2wav_disable_dynamic_chunk")
    )
    assert (
        "--enable-torch-compile-first-chunk"
        in _serve_option_names("code2wav_enable_torch_compile_first_chunk")
    )
    assert (
        "--no-enable-torch-compile-first-chunk"
        in _serve_option_names("code2wav_disable_torch_compile_first_chunk")
    )
    assert (
        "--odeint-method-relaxed"
        in _serve_option_names("code2wav_enable_odeint_method_relaxed")
    )
    assert (
        "--no-odeint-method-relaxed"
        in _serve_option_names("code2wav_disable_odeint_method_relaxed")
    )


def test_qwen35_cli_toggle_aliases_resolve_modes():
    assert (
        _resolve_toggle_alias_flags(
            flag_name="code2wav_torch_compile",
            mode="default",
            enable_alias=True,
            disable_alias=False,
        )
        == "on"
    )
    assert (
        _resolve_toggle_alias_flags(
            flag_name="code2wav_torch_compile",
            mode="default",
            enable_alias=False,
            disable_alias=True,
        )
        == "off"
    )


def test_qwen35_cli_toggle_aliases_reject_conflicts():
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        _resolve_toggle_alias_flags(
            flag_name="code2wav_torch_compile",
            mode="off",
            enable_alias=True,
            disable_alias=False,
        )


def test_qwen35_cli_parses_single_visible_device_alias():
    assert _parse_single_gpu_visible_device(
        "--talker-visible-devices",
        "[2]",
    ) == 2
    assert _parse_single_gpu_visible_device(
        "--talker-visible-devices",
        "2",
    ) == 2
    with pytest.raises(typer.BadParameter, match="exactly 1 GPU"):
        _parse_single_gpu_visible_device(
            "--talker-visible-devices",
            "[1, 2]",
        )


def test_qwen35_cli_preflight_runs_text_checks(monkeypatch):
    from sglang_omni.models.qwen3_5_omni import preflight

    calls = []

    def fake_run(model_path, *, speech, code2wav_model_path, **kwargs):
        calls.append((model_path, speech, code2wav_model_path, kwargs))
        return PreflightReport(model_path=model_path, issues=())

    monkeypatch.setattr(preflight, "run_qwen35_preflight", fake_run)

    _run_qwen35_preflight_or_raise(
        Qwen35OmniPipelineConfig(model_path="/models/qwen35"),
        code2wav_model_path=None,
    )

    assert calls == [("/models/qwen35", False, None, {})]


def test_qwen35_cli_preflight_forwards_xvector_checks(monkeypatch):
    from sglang_omni.models.qwen3_5_omni import preflight

    calls = []

    def fake_run(model_path, *, speech, code2wav_model_path, **kwargs):
        calls.append((model_path, speech, code2wav_model_path, kwargs))
        return PreflightReport(model_path=model_path, issues=())

    monkeypatch.setattr(preflight, "run_qwen35_preflight", fake_run)

    _run_qwen35_preflight_or_raise(
        Qwen35OmniSpeechPipelineConfig(model_path="/models/qwen35"),
        code2wav_model_path="/models/qwen35/code2wav",
        xvector_info_paths=("/voices/ref-a", "/voices/ref-b"),
        validate_xvector_pickle=True,
    )

    assert calls == [
        (
            "/models/qwen35",
            True,
            "/models/qwen35/code2wav",
            {
                "xvector_info_paths": ("/voices/ref-a", "/voices/ref-b"),
                "validate_xvector_pickle": True,
            },
        )
    ]


def test_qwen35_cli_preflight_uses_merged_code2wav_path(monkeypatch):
    from sglang_omni.models.qwen3_5_omni import preflight

    calls = []
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/qwen35")
    _stage(config, "code2wav").factory_args[
        "code2wav_model_path"
    ] = "/models/qwen35/codec"

    def fake_run(model_path, *, speech, code2wav_model_path, **kwargs):
        calls.append((model_path, speech, code2wav_model_path, kwargs))
        return PreflightReport(model_path=model_path, issues=())

    monkeypatch.setattr(preflight, "run_qwen35_preflight", fake_run)

    _run_qwen35_preflight_or_raise(config, code2wav_model_path=None)

    assert calls == [("/models/qwen35", True, "/models/qwen35/codec", {})]


def test_qwen35_cli_preflight_cli_code2wav_path_wins(monkeypatch):
    from sglang_omni.models.qwen3_5_omni import preflight

    calls = []
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/qwen35")
    _stage(config, "code2wav").factory_args[
        "code2wav_model_path"
    ] = "/models/qwen35/from-yaml"

    def fake_run(model_path, *, speech, code2wav_model_path, **kwargs):
        calls.append((model_path, speech, code2wav_model_path, kwargs))
        return PreflightReport(model_path=model_path, issues=())

    monkeypatch.setattr(preflight, "run_qwen35_preflight", fake_run)

    _run_qwen35_preflight_or_raise(
        config,
        code2wav_model_path="/models/qwen35/from-cli",
    )

    assert calls == [("/models/qwen35", True, "/models/qwen35/from-cli", {})]


def test_qwen35_cli_preflight_raises_on_failed_report(monkeypatch):
    from sglang_omni.models.qwen3_5_omni import preflight

    def fake_run(model_path, *, speech, code2wav_model_path, **kwargs):
        return PreflightReport(
            model_path=model_path,
            issues=(preflight.PreflightIssue("error", model_path, "missing files"),),
        )

    monkeypatch.setattr(preflight, "run_qwen35_preflight", fake_run)

    with pytest.raises(typer.BadParameter, match="preflight FAIL"):
        _run_qwen35_preflight_or_raise(
            Qwen35OmniPipelineConfig(model_path="/models/qwen35"),
            code2wav_model_path=None,
        )


def test_qwen35_cli_partial_start_accepts_talker_factory():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_partial_start_cli_overrides(config, talker_partial_start="off")

    assert _stage(config, "talker_ar").factory_args["enable_partial_start"] is False


def test_qwen35_cli_colocated_rejects_moving_talker_gpu():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="talker_ar away"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=1,
            code2wav_gpu=None,
        )


def test_qwen35_cli_colocated_rejects_moving_code2wav_gpu():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="code2wav away"):
        apply_parallelism_cli_overrides(
            config,
            thinker_tp_size=None,
            thinker_gpus=None,
            talker_gpu=None,
            code2wav_gpu=1,
        )


def test_qwen35_cli_applies_global_max_running_requests():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_max_running_requests_cli_overrides(
        config,
        max_running_requests=24,
        thinker_max_running_requests=None,
        talker_max_running_requests=None,
    )

    assert (
        _stage(config, "thinker")
        .runtime.sglang_server_args.max_running_requests
        == 24
    )
    assert (
        _stage(config, "talker_ar")
        .runtime.sglang_server_args.max_running_requests
        == 24
    )


def test_qwen35_cli_per_stage_max_running_requests_override_global():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_max_running_requests_cli_overrides(
        config,
        max_running_requests=24,
        thinker_max_running_requests=16,
        talker_max_running_requests=8,
    )

    assert (
        _stage(config, "thinker")
        .runtime.sglang_server_args.max_running_requests
        == 16
    )
    assert (
        _stage(config, "talker_ar")
        .runtime.sglang_server_args.max_running_requests
        == 8
    )


def test_qwen35_cli_rejects_talker_max_running_requests_for_text_pipeline():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="talker-max-running"):
        apply_max_running_requests_cli_overrides(
            config,
            max_running_requests=None,
            thinker_max_running_requests=None,
            talker_max_running_requests=8,
        )


def test_qwen35_cli_applies_max_model_len_to_thinker_runtime():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_qwen35_max_model_len_cli_override(
        config,
        max_model_len=192000,
        thinker_max_seq_len=None,
    )

    assert _stage(config, "preprocessing").runtime.max_seq_len == 192000
    assert _stage(config, "thinker").runtime.max_seq_len == 192000
    assert _stage(config, "talker_ar").runtime.max_seq_len is None


def test_qwen35_cli_thinker_max_seq_len_wins_over_max_model_len():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_qwen35_max_model_len_cli_override(
        config,
        max_model_len=192000,
        thinker_max_seq_len=131072,
    )

    assert _stage(config, "preprocessing").runtime.max_seq_len == 131072
    assert _stage(config, "thinker").runtime.max_seq_len == 131072


def test_qwen35_cli_applies_max_mm_len_to_preprocessing_runtime():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")
    apply_qwen35_max_model_len_cli_override(
        config,
        max_model_len=262144,
        thinker_max_seq_len=None,
    )

    apply_qwen35_max_mm_len_cli_override(config, max_mm_len=256000)

    assert _stage(config, "preprocessing").runtime.max_seq_len == 256000
    assert _stage(config, "thinker").runtime.max_seq_len == 262144


def test_qwen35_cli_max_mm_len_rejects_context_overflow():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")
    apply_qwen35_max_model_len_cli_override(
        config,
        max_model_len=192000,
        thinker_max_seq_len=None,
    )

    with pytest.raises(typer.BadParameter, match="max-mm-len"):
        apply_qwen35_max_mm_len_cli_override(config, max_mm_len=256000)


def test_qwen35_cli_max_model_len_rejects_non_qwen35_config():
    class OtherConfig(Qwen35OmniSpeechPipelineConfig):
        pass

    with pytest.raises(typer.BadParameter, match="max-model-len"):
        apply_qwen35_max_model_len_cli_override(
            OtherConfig(model_path="dummy"),
            max_model_len=192000,
            thinker_max_seq_len=None,
        )


def test_qwen35_cli_applies_vllm_ar_server_args_to_speech_stages():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_vllm_ar_server_args_cli_overrides(
        config,
        prefix_caching="off",
        chunked_prefill="off",
        enforce_eager=True,
        thinker_enforce_eager=False,
        talker_enforce_eager=False,
        dtype="bfloat16",
        talker_dtype="float16",
        mamba_ssm_dtype="float32",
        max_num_batched_tokens=512,
        page_size=16,
        compilation_config='{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    )

    thinker_args = _stage(config, "thinker").factory_args["server_args_overrides"]
    talker_args = _stage(config, "talker_ar").factory_args["server_args_overrides"]
    assert thinker_args["disable_radix_cache"] is True
    assert talker_args["disable_radix_cache"] is True
    assert thinker_args["chunked_prefill_size"] is None
    assert talker_args["chunked_prefill_size"] is None
    assert thinker_args["max_prefill_tokens"] == 512
    assert talker_args["max_prefill_tokens"] == 512
    assert thinker_args["disable_cuda_graph"] is True
    assert talker_args["disable_cuda_graph"] is True
    assert thinker_args["dtype"] == "bfloat16"
    assert talker_args["dtype"] == "float16"
    assert thinker_args["mamba_ssm_dtype"] == "float32"
    assert talker_args["mamba_ssm_dtype"] == "float32"
    assert thinker_args["page_size"] == 16
    assert talker_args["page_size"] == 16


def test_qwen35_cli_applies_talker_quantization_to_speech_stage():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_talker_server_args_cli_overrides(
        config,
        talker_quantization="fp8",
    )

    talker_args = _stage(config, "talker_ar").factory_args[
        "server_args_overrides"
    ]
    assert talker_args["quantization"] == "fp8"


def test_qwen35_cli_talker_quantization_rejects_text_pipeline():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="talker-quantization"):
        apply_talker_server_args_cli_overrides(
            config,
            talker_quantization="fp8",
        )


def test_qwen35_cli_applies_vllm_ar_server_args_to_text_stage():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    apply_vllm_ar_server_args_cli_overrides(
        config,
        prefix_caching="on",
        chunked_prefill="on",
        enforce_eager=False,
        thinker_enforce_eager=True,
        talker_enforce_eager=False,
        dtype="bfloat16",
        mamba_ssm_dtype="float32",
        max_num_batched_tokens=512,
        page_size=16,
        compilation_config=None,
    )

    thinker_args = _stage(config, "thinker").factory_args["server_args_overrides"]
    assert thinker_args["disable_radix_cache"] is False
    assert thinker_args["chunked_prefill_size"] == 512
    assert thinker_args["max_prefill_tokens"] == 512
    assert thinker_args["disable_cuda_graph"] is True
    assert thinker_args["dtype"] == "bfloat16"
    assert thinker_args["mamba_ssm_dtype"] == "float32"
    assert thinker_args["page_size"] == 16


def test_qwen35_cli_compilation_config_can_disable_cuda_graph():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_vllm_ar_server_args_cli_overrides(
        config,
        prefix_caching="default",
        chunked_prefill="default",
        enforce_eager=False,
        thinker_enforce_eager=False,
        talker_enforce_eager=False,
        compilation_config='{"cudagraph_mode":"none"}',
    )

    thinker_args = _stage(config, "thinker").factory_args["server_args_overrides"]
    talker_args = _stage(config, "talker_ar").factory_args["server_args_overrides"]
    assert thinker_args["disable_cuda_graph"] is True
    assert talker_args["disable_cuda_graph"] is True


def test_qwen35_cli_compilation_config_rejects_unsupported_inductor():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="use_inductor"):
        apply_vllm_ar_server_args_cli_overrides(
            config,
            prefix_caching="default",
            chunked_prefill="default",
            enforce_eager=False,
            thinker_enforce_eager=False,
            talker_enforce_eager=False,
            compilation_config='{"use_inductor":true}',
        )


def test_qwen35_cli_rejects_unsupported_mamba_cache_mode():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="mamba-cache-mode"):
        apply_vllm_ar_server_args_cli_overrides(
            config,
            prefix_caching="default",
            chunked_prefill="default",
            enforce_eager=False,
            thinker_enforce_eager=False,
            talker_enforce_eager=False,
            mamba_cache_mode="all",
        )


def test_qwen35_cli_rejects_unsupported_kv_transfer_config():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="kv-transfer-config"):
        apply_vllm_ar_server_args_cli_overrides(
            config,
            prefix_caching="default",
            chunked_prefill="default",
            enforce_eager=False,
            thinker_enforce_eager=False,
            talker_enforce_eager=False,
            kv_transfer_config='{"kv_connector":"HybridConnector"}',
        )


def test_qwen35_cli_rejects_disaggregated_prefilling():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="disaggregated-prefilling"):
        apply_vllm_ar_server_args_cli_overrides(
            config,
            prefix_caching="default",
            chunked_prefill="default",
            enforce_eager=False,
            thinker_enforce_eager=False,
            talker_enforce_eager=False,
            enable_disaggregated_prefilling="true",
        )


def test_qwen35_cli_vllm_talker_eager_rejects_text_pipeline():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="talker-enforce-eager"):
        apply_vllm_ar_server_args_cli_overrides(
            config,
            prefix_caching="default",
            chunked_prefill="default",
            enforce_eager=False,
            thinker_enforce_eager=False,
            talker_enforce_eager=True,
        )


def test_qwen35_cli_applies_talker_model_path_override():
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/qwen35")

    apply_talker_model_path_cli_override(
        config,
        talker_model_path="/models/qwen35/talker",
    )

    args = _stage(config, "talker_ar").factory_args
    assert args["model_path"] == "/models/qwen35/talker"
    assert args["root_model_path"] == "/models/qwen35"
    assert args["weight_prefix"] == ""


def test_qwen35_cli_applies_talker_model_path_to_colocated_config():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="/models/qwen35")

    apply_talker_model_path_cli_override(
        config,
        talker_model_path="/models/qwen35/talker",
    )

    args = _stage(config, "talker_ar").factory_args
    assert args["model_path"] == "/models/qwen35/talker"
    assert args["root_model_path"] == "/models/qwen35"
    assert args["weight_prefix"] == ""


def test_qwen35_cli_applies_video_preprocessing_runtime_overrides():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_preprocessing_video_cli_overrides(
        config,
        image_min_pixels=100352,
        image_max_pixels=401408,
        video_fps=2.0,
        video_max_frames=128,
        video_min_frames=4,
        video_min_pixels=4096,
        video_max_pixels=401408,
        video_total_pixels=32768 * 768,
        video_override_max_pixels=True,
        video_seconds_per_chunk=2.0,
        video_position_id_per_seconds=25.0,
        audio_target_sr=16000,
        audio_timestamp_interval=30,
        audio_downsample_times=4,
        audio_downsample_chunk_size=100,
    )

    runtime = _stage(config, "preprocessing").runtime
    assert runtime.image_min_pixels == 100352
    assert runtime.image_max_pixels == 401408
    assert runtime.video_fps == 2.0
    assert runtime.video_max_frames == 128
    assert runtime.video_min_frames == 4
    assert runtime.video_min_pixels == 4096
    assert runtime.video_max_pixels == 401408
    assert runtime.video_total_pixels == 32768 * 768
    assert runtime.video_override_max_pixels is True
    assert runtime.video_seconds_per_chunk == 2.0
    assert runtime.video_position_id_per_seconds == 25.0
    assert runtime.audio_target_sr == 16000
    assert runtime.audio_timestamp_interval == 30
    assert runtime.audio_downsample_times == 4
    assert runtime.audio_downsample_chunk_size == 100


def test_qwen35_cli_video_preprocessing_rejects_bad_values():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="--video-max-frames"):
        apply_preprocessing_video_cli_overrides(
            config,
            video_fps=None,
            video_max_frames=0,
            video_min_frames=None,
            video_min_pixels=None,
            video_max_pixels=None,
            video_total_pixels=None,
            video_seconds_per_chunk=None,
            video_position_id_per_seconds=None,
            audio_target_sr=None,
            audio_timestamp_interval=None,
            audio_downsample_times=None,
            audio_downsample_chunk_size=None,
        )

    with pytest.raises(typer.BadParameter, match="--audio-downsample-times"):
        apply_preprocessing_video_cli_overrides(
            config,
            video_fps=None,
            video_max_frames=None,
            video_min_frames=None,
            video_min_pixels=None,
            video_max_pixels=None,
            video_total_pixels=None,
            video_seconds_per_chunk=None,
            video_position_id_per_seconds=None,
            audio_target_sr=None,
            audio_timestamp_interval=None,
            audio_downsample_times=-1,
            audio_downsample_chunk_size=None,
        )


def test_qwen35_cli_applies_limit_mm_per_prompt_override():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_limit_mm_per_prompt_cli_override(
        config,
        limit_mm_per_prompt='{"image": 2, "video": 1}',
        limit_mm_per_prompt_image=None,
        limit_mm_per_prompt_video=None,
        limit_mm_per_prompt_audio=0,
    )

    limit = _stage(config, "preprocessing").factory_args["limit_mm_per_prompt"]
    assert limit["audio"] == 0
    assert limit["image"] == 2
    assert limit["video"] == 1


def test_qwen35_cli_limit_mm_per_prompt_rejects_non_qwen35_config():
    class OtherConfig(Qwen35OmniPipelineConfig):
        pass

    with pytest.raises(typer.BadParameter, match="limit-mm-per-prompt"):
        apply_limit_mm_per_prompt_cli_override(
            OtherConfig(model_path="dummy"),
            limit_mm_per_prompt='{"image": 2}',
            limit_mm_per_prompt_image=None,
            limit_mm_per_prompt_video=None,
            limit_mm_per_prompt_audio=None,
        )


def test_qwen35_cli_talker_model_path_rejects_text_pipeline():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="--talker-model-path"):
        apply_talker_model_path_cli_override(
            config,
            talker_model_path="/models/qwen35/talker",
        )


def test_qwen35_cli_applies_code2wav_factory_overrides():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    apply_code2wav_cli_overrides(
        config,
        code2wav_model_path="/models/qwen35/codec",
        code2wav_torch_compile="on",
        code2wav_torch_compile_first_chunk="on",
        code2wav_sample_rate=48000,
        code2wav_stream_chunk_size=8,
        code2wav_left_context_size=0,
        code2wav_codec_eos_token_id=3000,
        code2wav_dynamic_chunk="on",
        code2wav_dynamic_chunk_sizes="2,4,8",
        code2wav_dynamic_chunk_steps="8 4 1",
        code2wav_odeint_method="RK4",
        code2wav_odeint_method_relaxed="on",
        code2wav_batched_chunk=2,
        code2wav_frequency="50HZ",
        code2wav_dit_quant="FP8",
    )

    code2wav = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav, config)
    assert code2wav.factory_args["code2wav_model_path"] == "/models/qwen35/codec"
    assert args["enable_torch_compile"] is True
    assert args["enable_torch_compile_first_chunk"] is True
    assert args["sample_rate"] == 48000
    assert args["stream_chunk_size"] == 8
    assert args["left_context_size"] == 0
    assert args["codec_eos_token_id"] == 3000
    assert args["enable_dynamic_chunk"] is True
    assert args["dynamic_chunk_sizes"] == (2, 4, 8)
    assert args["dynamic_chunk_steps"] == (8, 4, 1)
    assert args["odeint_method"] == "rk4"
    assert args["odeint_method_relaxed"] is True
    assert args["batched_chunk"] == 2
    assert args["frequency"] == "50hz"
    assert args["dit_quant"] == "fp8"


def test_qwen35_cli_resolves_code2wav_model_folder_under_model_root():
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/qwen35")

    apply_code2wav_cli_overrides(
        config,
        code2wav_model_path=None,
        code2wav_torch_compile="default",
        code2wav_torch_compile_first_chunk="default",
        code2wav_sample_rate=None,
        code2wav_stream_chunk_size=None,
        code2wav_left_context_size=None,
        code2wav_codec_eos_token_id=None,
        code2wav_dynamic_chunk="default",
        code2wav_dynamic_chunk_sizes=None,
        code2wav_dynamic_chunk_steps=None,
        code2wav_odeint_method=None,
        code2wav_odeint_method_relaxed="default",
        code2wav_batched_chunk=None,
        code2wav_frequency=None,
        code2wav_dit_quant=None,
        code2wav_model_folder="qwen3_5_omni_codec_decode_online_0306",
    )

    code2wav = _stage(config, "code2wav")
    assert code2wav.factory_args["code2wav_model_path"] == (
        "/models/qwen35/qwen3_5_omni_codec_decode_online_0306"
    )


def test_qwen35_cli_code2wav_model_path_wins_over_folder():
    config = Qwen35OmniSpeechPipelineConfig(model_path="/models/qwen35")

    apply_code2wav_cli_overrides(
        config,
        code2wav_model_path="/models/custom-codec",
        code2wav_torch_compile="default",
        code2wav_torch_compile_first_chunk="default",
        code2wav_sample_rate=None,
        code2wav_stream_chunk_size=None,
        code2wav_left_context_size=None,
        code2wav_codec_eos_token_id=None,
        code2wav_dynamic_chunk="default",
        code2wav_dynamic_chunk_sizes=None,
        code2wav_dynamic_chunk_steps=None,
        code2wav_odeint_method=None,
        code2wav_odeint_method_relaxed="default",
        code2wav_batched_chunk=None,
        code2wav_frequency=None,
        code2wav_dit_quant=None,
        code2wav_model_folder="qwen3_5_omni_codec_decode_online_0306",
    )

    code2wav = _stage(config, "code2wav")
    assert code2wav.factory_args["code2wav_model_path"] == "/models/custom-codec"


def test_qwen35_cli_applies_code2wav_overrides_to_colocated_config():
    config = Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy")

    apply_code2wav_cli_overrides(
        config,
        code2wav_model_path=None,
        code2wav_torch_compile="off",
        code2wav_torch_compile_first_chunk="off",
        code2wav_sample_rate=None,
        code2wav_stream_chunk_size=None,
        code2wav_left_context_size=None,
        code2wav_codec_eos_token_id=None,
        code2wav_dynamic_chunk="off",
        code2wav_dynamic_chunk_sizes=None,
        code2wav_dynamic_chunk_steps=None,
        code2wav_odeint_method=None,
        code2wav_odeint_method_relaxed="off",
        code2wav_batched_chunk=None,
        code2wav_frequency=None,
        code2wav_dit_quant=None,
    )

    args = resolve_stage_factory_args(_stage(config, "code2wav"), config)
    assert args["enable_torch_compile"] is False
    assert args["enable_torch_compile_first_chunk"] is False
    assert args["enable_dynamic_chunk"] is False
    assert args["odeint_method_relaxed"] is False


def test_qwen35_cli_code2wav_overrides_reject_text_pipeline():
    config = Qwen35OmniPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="--code2wav-\\*"):
        apply_code2wav_cli_overrides(
            config,
            code2wav_model_path="/models/qwen35/codec",
            code2wav_torch_compile="default",
            code2wav_torch_compile_first_chunk="default",
            code2wav_sample_rate=None,
            code2wav_stream_chunk_size=None,
            code2wav_left_context_size=None,
            code2wav_codec_eos_token_id=None,
            code2wav_dynamic_chunk="default",
            code2wav_dynamic_chunk_sizes=None,
            code2wav_dynamic_chunk_steps=None,
            code2wav_odeint_method=None,
            code2wav_odeint_method_relaxed="default",
            code2wav_batched_chunk=None,
            code2wav_frequency=None,
            code2wav_dit_quant=None,
        )


def test_qwen35_cli_code2wav_overrides_reject_bad_dynamic_chunks():
    config = Qwen35OmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="values must be >= 1"):
        apply_code2wav_cli_overrides(
            config,
            code2wav_model_path=None,
            code2wav_torch_compile="default",
            code2wav_torch_compile_first_chunk="default",
            code2wav_sample_rate=None,
            code2wav_stream_chunk_size=None,
            code2wav_left_context_size=None,
            code2wav_codec_eos_token_id=None,
            code2wav_dynamic_chunk="default",
            code2wav_dynamic_chunk_sizes="4,0",
            code2wav_dynamic_chunk_steps=None,
            code2wav_odeint_method=None,
            code2wav_odeint_method_relaxed="default",
            code2wav_batched_chunk=None,
            code2wav_frequency=None,
            code2wav_dit_quant=None,
        )
