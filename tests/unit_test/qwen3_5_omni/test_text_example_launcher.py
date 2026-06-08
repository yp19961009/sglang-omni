# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples import run_qwen3_5_omni_server as launcher
from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
    QWEN3_5_OMNI_MAX_PREFILL_TOKENS,
    QWEN3_5_OMNI_MODEL_NAME,
    QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
    Qwen35OmniPipelineConfig,
)


def _args(**overrides):
    data = dict(
        model_path="/models/qwen35",
        gpu_thinker=1,
        thinker_visible_devices=None,
        gpu_image_encoder=None,
        gpu_audio_encoder=2,
        thinker_max_seq_len=16384,
        mem_fraction_static=0.45,
        max_running_requests=None,
        max_seq_len_to_capture=None,
        compilation_config=None,
        max_num_batched_tokens=None,
        page_size=None,
        enable_prefix_caching=None,
        enable_chunked_prefill=None,
        thinker_enforce_eager=False,
        thinker_quantization=None,
        thinker_dtype=None,
        mamba_ssm_dtype=None,
        mamba_cache_mode=None,
        kv_transfer_config=None,
        enable_disaggregated_prefilling=None,
        tensor_parallel_size=None,
        distributed_executor_backend=None,
        kv_cache_dtype=None,
        enable_expert_parallel=None,
        max_mm_len=None,
        mm_processor_cache_gb=None,
        speculative_config=None,
        use_omni_engine=None,
        use_omni_rpc_engine=None,
        is_thinker=None,
        thinker_only=None,
        use_zero_shot=None,
        skip_mm_profiling=None,
        override_video_max_pixels=None,
        disable_mtp=False,
        image_min_pixels=None,
        image_max_pixels=None,
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
        audio_downsample_times=None,
        audio_downsample_chunk_size=None,
        limit_mm_per_prompt=None,
        limit_mm_per_prompt_image=None,
        limit_mm_per_prompt_video=None,
        limit_mm_per_prompt_audio=None,
        relay_backend="shm",
        host="0.0.0.0",
        port=8009,
        model_name="qwen3.5-omni",
        preflight=False,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_qwen35_text_parse_args_uses_qwen35_context_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py"])

    args = launcher.parse_args()

    assert args.thinker_max_seq_len == QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN
    assert args.preflight is False


def test_qwen35_text_parse_args_normalizes_model_name_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--model-name", "qwen3_5_omni"],
    )

    args = launcher.parse_args()

    assert args.model_name == QWEN3_5_OMNI_MODEL_NAME


def test_qwen35_text_parse_args_accepts_vllm_model_alias(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--model", "/models/qwen35"])

    args = launcher.parse_args()

    assert args.model_path == "/models/qwen35"


def test_qwen35_text_parse_args_accepts_vllm_thinker_model_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--thinker-model", "/models/qwen35/thinker"],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/qwen35/thinker"


def test_qwen35_text_parse_args_accepts_preflight(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--preflight"])

    args = launcher.parse_args()

    assert args.preflight is True


def test_qwen35_text_parse_args_accepts_max_running_requests(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--max-running-requests", "24"],
    )

    args = launcher.parse_args()

    assert args.max_running_requests == 24


def test_qwen35_text_parse_args_accepts_vllm_max_num_seqs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--max-num-seqs", "24"])

    args = launcher.parse_args()

    assert args.max_running_requests == 24


def test_qwen35_text_parse_args_accepts_vllm_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--max-num-batched-tokens", "512"],
    )

    args = launcher.parse_args()

    assert args.max_num_batched_tokens == 512


def test_qwen35_text_parse_args_accepts_vllm_block_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--block-size", "16"])

    args = launcher.parse_args()

    assert args.page_size == 16


def test_qwen35_text_parse_args_accepts_vllm_quantization_alias(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--quantization", "FP8"])

    args = launcher.parse_args()

    assert args.thinker_quantization == "fp8"


def test_qwen35_text_parse_args_accepts_vllm_max_model_len(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--max-model-len", "192000"])

    args = launcher.parse_args()

    assert args.thinker_max_seq_len == 192000


def test_qwen35_text_parse_args_accepts_vllm_max_mm_len(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--max-mm-len", "4096"])

    args = launcher.parse_args()

    assert args.max_mm_len == 4096


def test_qwen35_text_parse_args_accepts_empty_speculative_config(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--speculative-config", "{}"],
    )

    args = launcher.parse_args()

    assert args.speculative_config == {}


def test_qwen35_text_parse_args_accepts_vllm_max_seq_len_to_capture(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--max-seq-len-to-capture", "262144"],
    )

    args = launcher.parse_args()

    assert args.max_seq_len_to_capture == 262144


def test_qwen35_text_parse_args_accepts_vllm_visible_devices(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--thinker-visible-devices", "[3]"],
    )

    args = launcher.parse_args()

    assert args.thinker_visible_devices == (3,)


def test_qwen35_text_parse_args_accepts_vllm_devices_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--thinker-devices",
            "[3]",
            "--thinker-tensor-parallel-size",
            "1",
        ],
    )

    args = launcher.parse_args()

    assert args.thinker_visible_devices == (3,)
    assert args.tensor_parallel_size == 1


def test_qwen35_text_parse_args_accepts_vllm_compilation_config(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--compilation-config",
            '{"cudagraph_mode":"FULL_DECODE_ONLY","use_inductor":false}',
        ],
    )

    args = launcher.parse_args()

    assert args.compilation_config == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "use_inductor": False,
    }


def test_qwen35_text_parse_args_accepts_disable_mtp(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--disable-mtp"])

    args = launcher.parse_args()

    assert args.disable_mtp is True


def test_qwen35_text_parse_args_accepts_video_needs_metadata(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--video-needs-metadata"],
    )

    args = launcher.parse_args()

    assert args.video_needs_metadata is True


def test_qwen35_text_parse_args_accepts_vllm_gpu_memory_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--gpu-memory-utilization", "0.55"],
    )

    args = launcher.parse_args()

    assert args.mem_fraction_static == 0.55


def test_qwen35_text_parse_args_accepts_dtype_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--dtype",
            "BFLOAT16",
            "--mamba-cache-dtype",
            "FLOAT32",
        ],
    )

    args = launcher.parse_args()

    assert args.thinker_dtype == "bfloat16"
    assert args.mamba_ssm_dtype == "float32"


def test_qwen35_text_parse_args_accepts_vllm_mamba_cache_mode_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--mamba-cache-mode", "none"])

    args = launcher.parse_args()

    assert args.mamba_cache_mode == "none"


def test_qwen35_text_parse_args_accepts_vllm_kv_transfer_noop_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--thinker-kv-transfer-config={}",
            "--enable-disaggregated-prefilling",
            "false",
            "--serve-port",
            "29000",
        ],
    )

    args = launcher.parse_args()

    assert args.kv_transfer_config == {}
    assert args.enable_disaggregated_prefilling is False
    assert args.port == 29000


def test_qwen35_text_parse_args_accepts_vllm_engine_profile_noops(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--tensor-parallel-size",
            "1",
            "--distributed-executor-backend=mp",
            "--kv-cache-dtype",
            "auto",
            "--enable-expert-parallel=false",
            "--mm-processor-cache-gb",
            "0",
            "--use-omni-rpc-engine",
            "--is-thinker=true",
            "--thinker-only=true",
            "--skip-mm-profiling",
        ],
    )

    args = launcher.parse_args()

    assert args.tensor_parallel_size == 1
    assert args.distributed_executor_backend == "mp"
    assert args.kv_cache_dtype == "auto"
    assert args.enable_expert_parallel is False
    assert args.mm_processor_cache_gb == 0
    assert args.use_omni_rpc_engine is True
    assert args.is_thinker is True
    assert args.thinker_only is True
    assert args.skip_mm_profiling is True


def test_qwen35_text_parse_args_applies_vllm_profile_defaults(monkeypatch, tmp_path):
    profile = tmp_path / "h20.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/qwen35",
                    "host": "127.0.0.1",
                    "port": 8009,
                    "dtype": "bfloat16",
                    "max_model_len": 192000,
                    "omni_video_fps": 2,
                    "thinker_gpu_memory_utilization": 0.7,
                    "talker_gpu_memory_utilization": 0.2,
                    "talker_visible_devices": [1],
                    "code2wav_visible_devices": [1],
                    "code2wav_model_folder": "code2wav",
                    "send_chunk_size": 8,
                    "thinker_only": True,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                }
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--vllm-profile",
            str(profile),
        ],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/qwen35"
    assert args.host == "127.0.0.1"
    assert args.port == 8009
    assert args.thinker_dtype == "bfloat16"
    assert args.thinker_max_seq_len == 192000
    assert args.video_fps == 2.0
    assert args.mem_fraction_static == 0.7
    assert args.thinker_only is True
    assert args.enable_prefix_caching is True
    assert args.enable_chunked_prefill is True


def test_qwen35_text_parse_args_vllm_profile_allows_cli_overrides(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "h20.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/from-profile",
                    "host": "127.0.0.1",
                    "port": 8009,
                    "dtype": "bfloat16",
                    "max_model_len": 192000,
                    "omni_video_fps": 2,
                    "thinker_gpu_memory_utilization": 0.7,
                }
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--vllm-profile",
            str(profile),
            "--model-path",
            "/models/from-cli",
            "--host",
            "0.0.0.0",
            "--port",
            "9010",
            "--dtype",
            "float16",
            "--video-fps",
            "4",
            "--gpu-memory-utilization",
            "0.5",
        ],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/from-cli"
    assert args.host == "0.0.0.0"
    assert args.port == 9010
    assert args.thinker_dtype == "float16"
    assert args.video_fps == 4.0
    assert args.mem_fraction_static == 0.5


def test_qwen35_text_parse_args_vllm_mtp_profile_allows_cli_disable_mtp(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "h20_mtp.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/from-profile",
                    "dtype": "bfloat16",
                    "max_model_len": 262144,
                    "speculative_config": {
                        "method": "qwen3_omni_next_thinker_mtp",
                        "num_speculative_tokens": 4,
                    },
                }
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--vllm-profile",
            str(profile),
            "--disable-mtp",
        ],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/from-profile"
    assert args.thinker_dtype == "bfloat16"
    assert args.thinker_max_seq_len == 262144
    assert args.disable_mtp is True


def test_qwen35_text_parse_args_vllm_profile_rejects_unsupported(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "unsupported.config"
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
        sys,
        "argv",
        [
            "server.py",
            "--vllm-profile",
            str(profile),
        ],
    )

    with pytest.raises(ValueError, match="profile preflight FAIL"):
        launcher.parse_args()


def test_qwen35_text_parse_args_accepts_limit_mm_per_prompt(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--limit-mm-per-prompt",
            '{"image": 2}',
            "--limit-mm-per-prompt-video",
            "1",
        ],
    )

    args = launcher.parse_args()

    assert args.limit_mm_per_prompt == {"image": 2}
    assert args.limit_mm_per_prompt_video == 1


def test_qwen35_text_parse_args_accepts_preprocessing_audio_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--video-seconds-per-chunk",
            "2.5",
            "--position-id-per-seconds=25",
            "--sampling_rate",
            "16000",
            "--timestamp-interval=30",
            "--audio-downsample-times=4",
            "--downsample-chunk-size=100",
        ],
    )

    args = launcher.parse_args()

    assert args.video_seconds_per_chunk == 2.5
    assert args.video_position_id_per_seconds == 25.0
    assert args.audio_target_sr == 16000
    assert args.audio_timestamp_interval == 30
    assert args.audio_downsample_times == 4
    assert args.audio_downsample_chunk_size == 100


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_builds_text_pipeline(launch_server):
    launcher._launch_server(_args())

    config = launch_server.call_args.args[0]
    assert isinstance(config, Qwen35OmniPipelineConfig)
    assert config.model_path == "/models/qwen35"
    assert _stage(config, "thinker").gpu == 1
    assert _stage(config, "image_encoder").gpu == 1
    assert _stage(config, "audio_encoder").gpu == 2
    assert _stage(config, "thinker").factory_args["thinker_max_seq_len"] == 16384
    assert (
        _stage(config, "preprocessing").factory_args["thinker_max_seq_len"] == 16384
    )
    assert (
        _stage(config, "thinker")
        .factory_args["server_args_overrides"]["mem_fraction_static"]
        == 0.45
    )
    overrides = _stage(config, "thinker").factory_args["server_args_overrides"]
    assert overrides["max_prefill_tokens"] == QWEN3_5_OMNI_MAX_PREFILL_TOKENS
    assert overrides["chunked_prefill_size"] == QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_maps_visible_device_alias(launch_server):
    launcher._launch_server(
        _args(
            thinker_visible_devices=(3,),
            gpu_image_encoder=None,
            gpu_audio_encoder=None,
        )
    )

    config = launch_server.call_args.args[0]
    assert _stage(config, "thinker").gpu == 3
    assert _stage(config, "image_encoder").gpu == 3
    assert _stage(config, "audio_encoder").gpu == 3


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_applies_vllm_ar_server_args(launch_server):
    launcher._launch_server(
        _args(
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            thinker_enforce_eager=True,
            thinker_quantization="nvfp4",
            thinker_dtype="bfloat16",
            mamba_ssm_dtype="float32",
            max_num_batched_tokens=512,
            page_size=16,
        )
    )

    config = launch_server.call_args.args[0]
    overrides = _stage(config, "thinker").factory_args["server_args_overrides"]
    assert overrides["disable_radix_cache"] is True
    assert overrides["chunked_prefill_size"] is None
    assert overrides["max_prefill_tokens"] == 512
    assert overrides["disable_cuda_graph"] is True
    assert overrides["quantization"] == "nvfp4"
    assert overrides["dtype"] == "bfloat16"
    assert overrides["mamba_ssm_dtype"] == "float32"
    assert overrides["page_size"] == 16


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_coerces_vllm_bool_strings(launch_server):
    launcher._launch_server(
        _args(
            enable_prefix_caching="false",
            enable_chunked_prefill="false",
            thinker_enforce_eager="false",
            enable_disaggregated_prefilling="false",
            enable_expert_parallel="false",
            is_thinker="true",
            thinker_only="true",
            override_video_max_pixels="false",
            preflight="false",
            compilation_config={
                "use_inductor": "false",
                "pass_config": {"fuse_mla": "false"},
            },
        )
    )

    config = launch_server.call_args.args[0]
    overrides = _stage(config, "thinker").factory_args["server_args_overrides"]
    runtime = _stage(config, "preprocessing").runtime
    assert overrides["disable_radix_cache"] is True
    assert overrides["chunked_prefill_size"] is None
    assert "disable_cuda_graph" not in overrides
    assert runtime.video_override_max_pixels is False


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_bad_bool_string(launch_server):
    with pytest.raises(ValueError, match="enable-prefix-caching"):
        launcher._launch_server(_args(enable_prefix_caching="maybe"))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_applies_max_num_batched_tokens(launch_server):
    launcher._launch_server(
        _args(max_num_batched_tokens=512, enable_chunked_prefill=True)
    )

    config = launch_server.call_args.args[0]
    overrides = _stage(config, "thinker").factory_args["server_args_overrides"]
    assert overrides["max_prefill_tokens"] == 512
    assert overrides["chunked_prefill_size"] == 512


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_compilation_config_can_disable_cuda_graph(
    launch_server,
):
    launcher._launch_server(_args(compilation_config={"cudagraph_mode": "none"}))

    config = launch_server.call_args.args[0]
    overrides = _stage(config, "thinker").factory_args["server_args_overrides"]
    assert overrides["disable_cuda_graph"] is True


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_unsupported_mamba_cache_mode(launch_server):
    with pytest.raises(ValueError, match="mamba-cache-mode"):
        launcher._launch_server(_args(mamba_cache_mode="light"))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_unsupported_kv_transfer_config(
    launch_server,
):
    with pytest.raises(ValueError, match="kv-transfer-config"):
        launcher._launch_server(
            _args(kv_transfer_config={"kv_connector": "HybridConnector"})
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_disaggregated_prefilling(launch_server):
    with pytest.raises(ValueError, match="disaggregated-prefilling"):
        launcher._launch_server(_args(enable_disaggregated_prefilling=True))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_unsupported_kv_cache_dtype(launch_server):
    with pytest.raises(ValueError, match="kv-cache-dtype"):
        launcher._launch_server(_args(kv_cache_dtype="tq4"))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_expert_parallel(launch_server):
    with pytest.raises(ValueError, match="expert-parallel"):
        launcher._launch_server(_args(enable_expert_parallel=True))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_speculative_mtp_config(launch_server):
    with pytest.raises(ValueError, match="speculative-config"):
        launcher._launch_server(
            _args(
                speculative_config={
                    "method": "qwen3_omni_next_thinker_mtp",
                    "num_speculative_tokens": 4,
                }
            )
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_ignores_mtp_config_when_disabled(launch_server):
    launcher._launch_server(
        _args(
            disable_mtp=True,
            speculative_config={
                "method": "qwen3_omni_next_thinker_mtp",
                "num_speculative_tokens": 4,
            },
        )
    )

    assert launch_server.called


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_applies_max_mm_len_to_preprocessing(
    launch_server,
):
    launcher._launch_server(_args(max_mm_len=4096))

    config = launch_server.call_args.args[0]
    assert _stage(config, "preprocessing").runtime.max_seq_len == 4096
    assert _stage(config, "thinker").runtime.max_seq_len is None


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_too_large_max_mm_len(launch_server):
    with pytest.raises(ValueError, match="max-mm-len"):
        launcher._launch_server(
            _args(max_mm_len=8193, thinker_max_seq_len=8192)
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_rejects_speech_profile_marker(launch_server):
    with pytest.raises(ValueError, match="thinker-only=false"):
        launcher._launch_server(_args(thinker_only=False))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_sets_max_running_requests(launch_server):
    launcher._launch_server(_args(max_running_requests=24))

    config = launch_server.call_args.args[0]
    assert (
        _stage(config, "thinker").runtime.sglang_server_args.max_running_requests
        == 24
    )


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_runs_preflight_before_launch(
    launch_server,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        launcher,
        "_run_preflight_or_raise",
        lambda **kwargs: calls.append(kwargs),
    )

    launcher._launch_server(_args(preflight=True))

    assert calls == [
        {
            "model_path": "/models/qwen35",
            "speech": False,
        }
    ]
    launch_server.assert_called_once()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_preflight_failure_skips_launch(
    launch_server,
    monkeypatch,
):
    def _fail_preflight(**kwargs):
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(launcher, "_run_preflight_or_raise", _fail_preflight)

    with pytest.raises(RuntimeError, match="preflight failed"):
        launcher._launch_server(_args(preflight=True))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_sets_video_preprocessing_runtime_args(launch_server):
    launcher._launch_server(
        _args(
            image_min_pixels=100352,
            image_max_pixels=401408,
            video_fps=2.0,
            video_max_frames=128,
            video_min_frames=4,
            video_min_pixels=4096,
            video_max_pixels=401408,
            video_total_pixels=32768 * 768,
            override_video_max_pixels=True,
            video_seconds_per_chunk=2.0,
            video_position_id_per_seconds=25.0,
            audio_target_sr=16000,
            audio_timestamp_interval=30,
            audio_downsample_times=4,
            audio_downsample_chunk_size=100,
        )
    )

    config = launch_server.call_args.args[0]
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


@patch("sglang_omni.serve.launch_server")
def test_qwen35_text_launcher_sets_limit_mm_per_prompt(launch_server):
    launcher._launch_server(
        _args(
            limit_mm_per_prompt={"image": 2},
            limit_mm_per_prompt_video=1,
        )
    )

    config = launch_server.call_args.args[0]
    limit = _stage(config, "preprocessing").factory_args["limit_mm_per_prompt"]
    assert limit["image"] == 2
    assert limit["video"] == 1


def test_qwen35_text_launcher_rejects_bad_mem_fraction():
    with pytest.raises(ValueError, match="--mem-fraction-static"):
        launcher._validate_fraction("--mem-fraction-static", 1.5)


def test_qwen35_text_launcher_rejects_bad_video_args():
    with pytest.raises(ValueError, match="--video-fps"):
        launcher._launch_server(_args(video_fps=0.0))


def test_qwen35_text_launcher_rejects_bad_max_running_requests():
    with pytest.raises(ValueError, match="--max-running-requests"):
        launcher._launch_server(_args(max_running_requests=0))
