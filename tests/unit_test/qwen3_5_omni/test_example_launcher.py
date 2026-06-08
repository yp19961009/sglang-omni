# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from examples import run_qwen3_5_omni_speech_server as launcher
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
    QWEN3_5_OMNI_MAX_RUNNING_REQUESTS,
    QWEN3_5_OMNI_MAX_PREFILL_TOKENS,
    QWEN3_5_OMNI_MODEL_NAME,
    QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
    Qwen35OmniSpeechColocatedPipelineConfig,
    Qwen35OmniSpeechPipelineConfig,
)


def _args(**overrides):
    data = dict(
        model_path="/models/qwen35",
        talker_model_path=None,
        code2wav_model_path=None,
        code2wav_model_folder=None,
        code2wav_enable_torch_compile=None,
        code2wav_enable_torch_compile_first_chunk=None,
        code2wav_codec_eos_token_id=None,
        code2wav_sample_rate=None,
        code2wav_stream_chunk_size=None,
        code2wav_left_context_size=None,
        code2wav_enable_dynamic_chunk=None,
        code2wav_dynamic_chunk_sizes=None,
        code2wav_dynamic_chunk_steps=None,
        code2wav_odeint_method=None,
        code2wav_odeint_method_relaxed=None,
        code2wav_batched_chunk=None,
        code2wav_frequency=None,
        code2wav_dit_quant=None,
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
        gpu_thinker=0,
        gpu_talker=None,
        gpu_code_predictor=None,
        gpu_code2wav=None,
        gpu_image_encoder=None,
        gpu_audio_encoder=None,
        thinker_tp_size=1,
        gpu_thinker_tp=None,
        relay_backend="shm",
        thinker_max_seq_len=8192,
        thinker_quantization=None,
        talker_quantization=None,
        dtype=None,
        thinker_dtype=None,
        talker_dtype=None,
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
        max_num_batched_tokens=None,
        page_size=None,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
        max_running_requests=None,
        thinker_max_running_requests=None,
        talker_max_running_requests=None,
        voice_type=launcher.QWEN35_DEFAULT_VOICE_TYPE,
        enable_tn=None,
        max_tokens=launcher.QWEN35_DEFAULT_MAX_TOKENS,
        seed=launcher.QWEN35_DEFAULT_SEED,
        enable_partial_start=None,
        partial_start_min_chunks=5,
        colocated=False,
        host="0.0.0.0",
        port=8008,
        model_name="qwen3.5-omni",
        preflight=False,
        xvector_info_paths=(),
        validate_xvector_pickle=False,
        max_seq_len_to_capture=None,
        compilation_config=None,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_qwen35_parse_args_uses_qwen35_context_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py"])

    args = launcher.parse_args()

    assert args.thinker_max_seq_len == QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN
    assert args.model_path == launcher.QWEN35_DEFAULT_MODEL_PATH
    assert args.model_name == launcher.QWEN35_DEFAULT_MODEL_NAME
    assert args.voice_type == launcher.QWEN35_DEFAULT_VOICE_TYPE
    assert args.enable_tn is None
    assert args.max_tokens == launcher.QWEN35_DEFAULT_MAX_TOKENS
    assert args.seed == launcher.QWEN35_DEFAULT_SEED
    assert args.preflight is False


def test_qwen35_parse_args_accepts_vllm_voice_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--voice-type",
            "Cherry",
            "--enable-tn",
        ],
    )

    args = launcher.parse_args()

    assert args.voice_type == "Cherry"
    assert args.enable_tn is True


def test_qwen35_parse_args_accepts_vllm_generation_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--max-tokens",
            "512",
            "--seed",
            "7",
        ],
    )

    args = launcher.parse_args()

    assert args.max_tokens == 512
    assert args.seed == 7


def test_qwen35_parse_args_normalizes_model_name_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--model-name", "qwen3-5-omni"],
    )

    args = launcher.parse_args()

    assert args.model_name == QWEN3_5_OMNI_MODEL_NAME


def test_qwen35_parse_args_accepts_preflight(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--preflight",
            "--voice-clone-info-path",
            "/voices/ref-a",
            "--validate-xvector-pickle",
        ],
    )

    args = launcher.parse_args()

    assert args.preflight is True
    assert args.xvector_info_paths == ("/voices/ref-a",)
    assert args.validate_xvector_pickle is True


def test_qwen35_parse_args_resolves_code2wav_model_folder(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--code2wav-model-folder",
            "qwen3_5_omni_codec_decode_online_0306",
        ],
    )

    args = launcher.parse_args()

    assert args.code2wav_model_path == (
        "/models/qwen35/qwen3_5_omni_codec_decode_online_0306"
    )


def test_qwen35_parse_args_accepts_vllm_model_path_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--model",
            "/models/qwen35",
            "--thinker-model",
            "/models/qwen35/thinker",
            "--talker-model",
            "/models/qwen35/talker_lm",
            "--code2wav-model",
            "/models/qwen35/code2wav",
        ],
    )

    args = launcher.parse_args()

    # 中文说明：speech server 保留 root model_path，split checkpoint
    # 下的 thinker/talker/code2wav 由 stage resolver 或显式路径处理。
    assert args.model_path == "/models/qwen35"
    assert args.talker_model_path == "/models/qwen35/talker_lm"
    assert args.code2wav_model_path == "/models/qwen35/code2wav"


def test_qwen35_parse_args_resolves_vllm_code2wav_model_folder_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--model",
            "/models/qwen35",
            "--code2wav-model",
            "code2wav",
        ],
    )

    args = launcher.parse_args()

    assert args.code2wav_model_folder == "code2wav"
    assert args.code2wav_model_path == "/models/qwen35/code2wav"


def test_qwen35_parse_args_accepts_standalone_thinker_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--thinker-model", "/models/qwen35/thinker"],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/qwen35/thinker"


def test_qwen35_parse_args_keeps_explicit_context_length(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--thinker-max-seq-len", "8192"],
    )

    args = launcher.parse_args()

    assert args.thinker_max_seq_len == 8192


def test_qwen35_parse_args_accepts_vllm_max_model_len(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--max-model-len", "192000"],
    )

    args = launcher.parse_args()

    assert args.thinker_max_seq_len == 192000


def test_qwen35_parse_args_accepts_vllm_max_seq_len_to_capture(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--max-seq-len-to-capture", "262144"],
    )

    args = launcher.parse_args()

    assert args.max_seq_len_to_capture == 262144


def test_qwen35_parse_args_accepts_vllm_visible_devices(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--thinker-visible-devices",
            "[0, 1]",
            "--talker_visible_devices=2",
            "--code2wav-visible-devices",
            "[2]",
        ],
    )

    args = launcher.parse_args()

    assert args.gpu_thinker == 0
    assert args.thinker_tp_size == 2
    assert args.gpu_thinker_tp == "0,1"
    assert args.gpu_talker == 2
    assert args.gpu_code2wav == 2


def test_qwen35_parse_args_accepts_vllm_devices_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--thinker-devices",
            "[0, 1]",
            "--thinker-tensor-parallel-size",
            "2",
            "--talker-devices=2",
            "--code2wav-devices",
            "[3]",
        ],
    )

    args = launcher.parse_args()

    assert args.gpu_thinker == 0
    assert args.thinker_tp_size == 2
    assert args.gpu_thinker_tp == "0,1"
    assert args.gpu_talker == 2
    assert args.gpu_code2wav == 3


def test_qwen35_parse_args_accepts_vllm_compilation_config(monkeypatch):
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


def test_qwen35_parse_args_accepts_vllm_mamba_cache_mode_none(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--mamba-cache-mode", "none"],
    )

    args = launcher.parse_args()

    assert args.mamba_cache_mode == "none"


def test_qwen35_parse_args_accepts_vllm_kv_transfer_noop_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server.py",
            "--thinker-kv-transfer-config",
            "{}",
            "--enable-disaggregated-prefilling=0",
            "--serve-port",
            "29000",
        ],
    )

    args = launcher.parse_args()

    assert args.kv_transfer_config == {}
    assert args.enable_disaggregated_prefilling is False
    assert args.port == 29000


def test_qwen35_parse_args_accepts_vllm_engine_profile_noops(monkeypatch):
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
            "--thinker-only=false",
            "--use-zero-shot",
            "--skip-mm-profiling",
            "--video-needs-metadata",
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
    assert args.thinker_only is False
    assert args.use_zero_shot is True
    assert args.skip_mm_profiling is True
    assert args.video_needs_metadata is True


def test_qwen35_parse_args_applies_vllm_profile_defaults(monkeypatch, tmp_path):
    profile = tmp_path / "h20.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "model": "/models/qwen35",
                    "host": "127.0.0.1",
                    "port": 8008,
                    "dtype": "bfloat16",
                    "max_model_len": 192000,
                    "send_chunk_size": 8,
                    "omni_video_fps": 2,
                    "talker_visible_devices": [1],
                    "code2wav_visible_devices": [1],
                    "code2wav_model_folder": "code2wav",
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                    "distributed_executor_backend": "mp",
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
    assert args.port == 8008
    assert args.dtype == "bfloat16"
    assert args.thinker_max_seq_len == 192000
    assert args.code2wav_stream_chunk_size == 8
    assert args.video_fps == 2.0
    assert args.gpu_talker == 1
    assert args.gpu_code2wav == 1
    assert args.code2wav_model_path == "/models/qwen35/code2wav"
    assert args.enable_prefix_caching is True
    assert args.enable_chunked_prefill is True
    assert args.distributed_executor_backend is None


def test_qwen35_parse_args_accepts_real_h20_vllm_profile_shape(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "h20.config"
    # 中文说明：这个片段对齐 vLLM dev/qwenc_perf_v2 的
    # util/vllmgen/configs/qwen3.5-omni/23b_fp8/h20.config。profile 本身
    # 不写模型路径，真实启动时由用户通过 --model/--model-path 补上。
    profile.write_text(
        json.dumps(
            {
                "envs": {
                    "CHAT_CONFIG": "pre-qwen3.5-omni",
                    "DS_LLM_OMNI_RPC": "1",
                    "VLLM_USE_V1": "1",
                    "VLLM_OMNI_PREPROCESS_ON_GPU": "0",
                    "VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING": "true",
                },
                "engine_args": {
                    "mamba_cache_dtype": "float32",
                    "dtype": "bfloat16",
                    "send_chunk_size": 8,
                    "max_model_len": 262144,
                    "max_seq_len_to_capture": 262144,
                    "max_num_batched_tokens": 32768,
                    "talker_gpu_memory_utilization": 0.8,
                    "gpu_memory_utilization": 0.6,
                    "use_omni_rpc_engine": True,
                    "thinker_only": False,
                    "distributed_executor_backend": "mp",
                    "enable_prefix_caching": True,
                    "max_num_seqs": 32,
                    "enable_chunked_prefill": True,
                    "enforce_eager": False,
                    "code2wav_model_folder": (
                        "qwen3_5_omni_codec_decode_online_0306"
                    ),
                    "code2wav_enable_torch_compile": True,
                    "limit_mm_per_prompt": {
                        "audio": 2048,
                        "video": 512,
                        "image": 2048,
                    },
                    "compilation_config": {
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "use_inductor": False,
                        "pass_config": {
                            "fuse_norm_quant": False,
                            "fuse_act_quant": False,
                            "fuse_attn_quant": False,
                        },
                    },
                },
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
            "--model",
            "/models/qwen35",
        ],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/qwen35"
    assert args.dtype == "bfloat16"
    assert args.mamba_ssm_dtype == "float32"
    assert args.thinker_max_seq_len == 262144
    # 中文说明：SGLang 没有独立的 capture 长度开关，preflight 会把
    # max_seq_len_to_capture 当作 vLLM-compatible no-op；真正影响上下文
    # 的是上面的 max_model_len -> thinker_max_seq_len。
    assert args.max_seq_len_to_capture is None
    assert args.max_num_batched_tokens == 32768
    assert args.code2wav_stream_chunk_size == 8
    assert args.mem_fraction_static == 0.6
    assert args.talker_mem_fraction_static == 0.8
    # 中文说明：这两个是 vLLM 引擎选择 marker；profile preflight 接受它们，
    # 但不会转成 SGLang runtime 参数，避免把 no-op 写进启动状态。
    assert args.use_omni_rpc_engine is None
    assert args.thinker_only is None
    assert args.enable_prefix_caching is True
    assert args.max_running_requests == 32
    assert args.enable_chunked_prefill is True
    assert args.enforce_eager is None
    assert (
        args.code2wav_model_path
        == "/models/qwen35/qwen3_5_omni_codec_decode_online_0306"
    )
    assert args.code2wav_enable_torch_compile is True
    assert args.limit_mm_per_prompt == {
        "audio": 2048,
        "video": 512,
        "image": 2048,
    }
    # FULL_DECODE_ONLY 是当前 SGLang Qwen3.5 默认 graph 策略，不需要
    # 作为 launcher 参数保留下来；preflight 只会映射非默认/禁用场景。
    assert args.compilation_config is None
    assert args.distributed_executor_backend is None


def test_qwen35_parse_args_rejects_real_h20_mtp_profile(monkeypatch, tmp_path):
    profile = tmp_path / "h20_mtp.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "dtype": "bfloat16",
                    "max_model_len": 262144,
                    "code2wav_model_folder": "code2wav",
                    "distributed_executor_backend": "mp",
                    "speculative_config": {
                        "method": "qwen3_omni_next_thinker_mtp",
                        "num_speculative_tokens": 4,
                    },
                    "use_zero_shot": True,
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
            "--model",
            "/models/qwen35",
        ],
    )

    with pytest.raises(ValueError, match="speculative_config"):
        launcher.parse_args()


def test_qwen35_parse_args_vllm_mtp_profile_allows_cli_disable_mtp(
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
                    "code2wav_model_folder": "code2wav",
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
    assert args.dtype == "bfloat16"
    assert args.thinker_max_seq_len == 262144
    assert args.code2wav_model_path == "/models/from-profile/code2wav"
    assert args.disable_mtp is True


def test_qwen35_parse_args_vllm_profile_allows_cli_overrides(
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
                    "port": 8008,
                    "send_chunk_size": 8,
                    "omni_video_fps": 2,
                    "talker_visible_devices": [1],
                    "code2wav_model_folder": "code2wav",
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
            "--model",
            "/models/from-cli",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--send-chunk-size",
            "16",
            "--video-fps",
            "4",
            "--talker-visible-devices",
            "[2]",
        ],
    )

    args = launcher.parse_args()

    assert args.model_path == "/models/from-cli"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.code2wav_stream_chunk_size == 16
    assert args.video_fps == 4.0
    assert args.gpu_talker == 2
    assert args.code2wav_model_path == "/models/from-cli/code2wav"


def test_qwen35_parse_args_vllm_profile_rejects_unsupported(
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


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_uses_speech_config(launch_server):
    launcher._launch_speech_server(_args())

    config = launch_server.call_args.args[0]
    assert launch_server.call_args.kwargs["default_talker_params"] == {
        "voice_type": launcher.QWEN35_DEFAULT_VOICE_TYPE,
    }
    assert launch_server.call_args.kwargs["default_generation_params"] == {
        "temperature": launcher.QWEN35_DEFAULT_TEMPERATURE,
        "top_k": launcher.QWEN35_DEFAULT_TOP_K,
        "top_p": launcher.QWEN35_DEFAULT_TOP_P,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": launcher.QWEN35_DEFAULT_MAX_TOKENS,
        "seed": launcher.QWEN35_DEFAULT_SEED,
    }
    assert isinstance(config, Qwen35OmniSpeechPipelineConfig)
    assert config.model_path == "/models/qwen35"
    assert _stage(config, "thinker").gpu == 0
    assert _stage(config, "talker_ar").gpu == 1
    assert _stage(config, "code2wav").gpu == 1
    for stage_name in ("thinker", "talker_ar"):
        overrides = _stage(config, stage_name).factory_args["server_args_overrides"]
        assert overrides["max_prefill_tokens"] == QWEN3_5_OMNI_MAX_PREFILL_TOKENS
        assert (
            overrides["chunked_prefill_size"]
            == QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE
        )
        assert (
            _stage(config, stage_name)
            .runtime.sglang_server_args.max_running_requests
            == QWEN3_5_OMNI_MAX_RUNNING_REQUESTS
        )


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_passes_default_talker_params(launch_server):
    launcher._launch_speech_server(
        _args(
            voice_type="Cherry",
            enable_tn="true",
        )
    )

    assert launch_server.call_args.kwargs["default_talker_params"] == {
        "voice_type": "Cherry",
        "enable_tn": True,
    }


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_passes_default_generation_params(launch_server):
    launcher._launch_speech_server(_args(max_tokens=512, seed=7))

    assert launch_server.call_args.kwargs["default_generation_params"] == {
        "temperature": launcher.QWEN35_DEFAULT_TEMPERATURE,
        "top_k": launcher.QWEN35_DEFAULT_TOP_K,
        "top_p": launcher.QWEN35_DEFAULT_TOP_P,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": 512,
        "seed": 7,
    }


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_applies_vllm_ar_server_args(launch_server):
    launcher._launch_speech_server(
        _args(
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=True,
            thinker_quantization="fp8",
            talker_quantization="nvfp4",
            dtype="bfloat16",
            talker_dtype="float16",
            mamba_ssm_dtype="float32",
            max_num_batched_tokens=512,
            page_size=16,
        )
    )

    config = launch_server.call_args.args[0]
    thinker_overrides = _stage(config, "thinker").factory_args[
        "server_args_overrides"
    ]
    talker_overrides = _stage(config, "talker_ar").factory_args[
        "server_args_overrides"
    ]

    assert thinker_overrides["disable_radix_cache"] is True
    assert talker_overrides["disable_radix_cache"] is True
    assert thinker_overrides["chunked_prefill_size"] is None
    assert talker_overrides["chunked_prefill_size"] is None
    assert thinker_overrides["max_prefill_tokens"] == 512
    assert talker_overrides["max_prefill_tokens"] == 512
    assert thinker_overrides["disable_cuda_graph"] is True
    assert talker_overrides["disable_cuda_graph"] is True
    assert thinker_overrides["quantization"] == "fp8"
    assert talker_overrides["quantization"] == "nvfp4"
    assert thinker_overrides["dtype"] == "bfloat16"
    assert talker_overrides["dtype"] == "float16"
    assert thinker_overrides["mamba_ssm_dtype"] == "float32"
    assert talker_overrides["mamba_ssm_dtype"] == "float32"
    assert thinker_overrides["page_size"] == 16
    assert talker_overrides["page_size"] == 16


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_coerces_vllm_bool_strings(launch_server):
    launcher._launch_speech_server(
        _args(
            colocated="false",
            enable_partial_start="false",
            enable_prefix_caching="false",
            enable_chunked_prefill="false",
            enforce_eager="false",
            thinker_enforce_eager="false",
            talker_enforce_eager="false",
            enable_disaggregated_prefilling="false",
            enable_expert_parallel="false",
            is_thinker="true",
            thinker_only="false",
            override_video_max_pixels="false",
            preflight="false",
            validate_xvector_pickle="false",
            compilation_config={
                "use_inductor": "false",
                "pass_config": {"fuse_mla": "false"},
            },
        )
    )

    config = launch_server.call_args.args[0]
    assert isinstance(config, Qwen35OmniSpeechPipelineConfig)
    thinker_overrides = _stage(config, "thinker").factory_args[
        "server_args_overrides"
    ]
    talker_overrides = _stage(config, "talker_ar").factory_args[
        "server_args_overrides"
    ]
    runtime = _stage(config, "preprocessing").runtime
    assert thinker_overrides["disable_radix_cache"] is True
    assert talker_overrides["disable_radix_cache"] is True
    assert thinker_overrides["chunked_prefill_size"] is None
    assert talker_overrides["chunked_prefill_size"] is None
    assert "disable_cuda_graph" not in thinker_overrides
    assert "disable_cuda_graph" not in talker_overrides
    assert runtime.video_override_max_pixels is False
    assert _stage(config, "talker_ar").factory_args["enable_partial_start"] is False


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_bad_bool_string(launch_server):
    with pytest.raises(ValueError, match="enable-partial-start"):
        launcher._launch_speech_server(_args(enable_partial_start="maybe"))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_applies_max_num_batched_tokens(launch_server):
    launcher._launch_speech_server(
        _args(max_num_batched_tokens=512, enable_chunked_prefill=True)
    )

    config = launch_server.call_args.args[0]
    thinker_overrides = _stage(config, "thinker").factory_args[
        "server_args_overrides"
    ]
    talker_overrides = _stage(config, "talker_ar").factory_args[
        "server_args_overrides"
    ]
    assert thinker_overrides["max_prefill_tokens"] == 512
    assert talker_overrides["max_prefill_tokens"] == 512
    assert thinker_overrides["chunked_prefill_size"] == 512
    assert talker_overrides["chunked_prefill_size"] == 512


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_compilation_config_can_disable_cuda_graph(launch_server):
    launcher._launch_speech_server(
        _args(compilation_config={"cudagraph_mode": "none"})
    )

    config = launch_server.call_args.args[0]
    thinker_overrides = _stage(config, "thinker").factory_args[
        "server_args_overrides"
    ]
    talker_overrides = _stage(config, "talker_ar").factory_args[
        "server_args_overrides"
    ]
    assert thinker_overrides["disable_cuda_graph"] is True
    assert talker_overrides["disable_cuda_graph"] is True


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_unsupported_mamba_cache_mode(launch_server):
    with pytest.raises(ValueError, match="mamba-cache-mode"):
        launcher._launch_speech_server(_args(mamba_cache_mode="light"))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_unsupported_kv_transfer_config(launch_server):
    with pytest.raises(ValueError, match="kv-transfer-config"):
        launcher._launch_speech_server(
            _args(kv_transfer_config={"kv_connector": "HybridConnector"})
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_disaggregated_prefilling(launch_server):
    with pytest.raises(ValueError, match="disaggregated-prefilling"):
        launcher._launch_speech_server(
            _args(enable_disaggregated_prefilling=True)
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_unsupported_kv_cache_dtype(launch_server):
    with pytest.raises(ValueError, match="kv-cache-dtype"):
        launcher._launch_speech_server(_args(kv_cache_dtype="tq4"))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_expert_parallel(launch_server):
    with pytest.raises(ValueError, match="expert-parallel"):
        launcher._launch_speech_server(_args(enable_expert_parallel=True))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_speculative_mtp_config(launch_server):
    with pytest.raises(ValueError, match="speculative-config"):
        launcher._launch_speech_server(
            _args(
                speculative_config={
                    "method": "qwen3_omni_next_thinker_mtp",
                    "num_speculative_tokens": 4,
                }
            )
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_ignores_mtp_config_when_disabled(launch_server):
    launcher._launch_speech_server(
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
def test_qwen35_launcher_applies_max_mm_len_to_preprocessing(launch_server):
    launcher._launch_speech_server(_args(max_mm_len=4096))

    config = launch_server.call_args.args[0]
    assert _stage(config, "preprocessing").runtime.max_seq_len == 4096


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_too_large_max_mm_len(launch_server):
    with pytest.raises(ValueError, match="max-mm-len"):
        launcher._launch_speech_server(
            _args(max_mm_len=8193, thinker_max_seq_len=8192)
        )

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_runs_preflight_before_launch(
    launch_server,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        launcher,
        "_run_preflight_or_raise",
        lambda **kwargs: calls.append(kwargs),
    )

    launcher._launch_speech_server(
        _args(
            preflight=True,
            code2wav_model_path="/models/qwen35/code2wav",
            xvector_info_paths=("/voices/ref-a", "/voices/ref-b"),
            validate_xvector_pickle=True,
        )
    )

    assert calls == [
        {
            "model_path": "/models/qwen35",
            "speech": True,
            "code2wav_model_path": "/models/qwen35/code2wav",
            "xvector_info_paths": ("/voices/ref-a", "/voices/ref-b"),
            "validate_xvector_pickle": True,
        }
    ]
    launch_server.assert_called_once()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_preflight_failure_skips_launch(
    launch_server,
    monkeypatch,
):
    def _fail_preflight(**kwargs):
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(launcher, "_run_preflight_or_raise", _fail_preflight)

    with pytest.raises(RuntimeError, match="preflight failed"):
        launcher._launch_speech_server(_args(preflight=True))

    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_uses_colocated_config(launch_server):
    launcher._launch_speech_server(_args(colocated=True))

    config = launch_server.call_args.args[0]
    assert isinstance(config, Qwen35OmniSpeechColocatedPipelineConfig)
    assert _stage(config, "thinker").gpu == 0
    assert _stage(config, "talker_ar").gpu == 0
    assert _stage(config, "code2wav").gpu == 0


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_allows_explicit_code2wav_gpu(launch_server):
    launcher._launch_speech_server(_args(gpu_code2wav=0))

    config = launch_server.call_args.args[0]
    assert _stage(config, "talker_ar").gpu == 1
    assert _stage(config, "code2wav").gpu == 0


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_explicit_talker_model_path(launch_server):
    launcher._launch_speech_server(_args(talker_model_path="/models/qwen35/talker"))

    config = launch_server.call_args.args[0]
    talker_stage = _stage(config, "talker_ar")
    assert talker_stage.factory_args["model_path"] == "/models/qwen35/talker"
    assert talker_stage.factory_args["root_model_path"] == "/models/qwen35"
    assert talker_stage.factory_args["weight_prefix"] == ""


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_global_max_running_requests(launch_server):
    launcher._launch_speech_server(_args(max_running_requests=24))

    config = launch_server.call_args.args[0]
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


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_maps_vllm_gpu_memory_utilization_aliases(launch_server):
    launcher._launch_speech_server(
        _args(
            mem_fraction_static=0.55,
            thinker_mem_fraction_static=0.6,
            talker_mem_fraction_static=0.7,
        )
    )

    config = launch_server.call_args.args[0]
    thinker_overrides = _stage(config, "thinker").factory_args[
        "server_args_overrides"
    ]
    talker_overrides = _stage(config, "talker_ar").factory_args[
        "server_args_overrides"
    ]
    assert thinker_overrides["mem_fraction_static"] == 0.6
    assert talker_overrides["mem_fraction_static"] == 0.7


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_per_stage_max_running_requests(launch_server):
    launcher._launch_speech_server(
        _args(
            max_running_requests=24,
            thinker_max_running_requests=16,
            talker_max_running_requests=8,
        )
    )

    config = launch_server.call_args.args[0]
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


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_explicit_code2wav_model_path(launch_server):
    launcher._launch_speech_server(
        _args(code2wav_model_path="/models/qwen35-code2wav")
    )

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    assert (
        code2wav_stage.factory_args["code2wav_model_path"]
        == "/models/qwen35-code2wav"
    )


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_model_path_from_folder(launch_server):
    args = _args(code2wav_model_folder="qwen3_5_omni_codec_decode_online_0306")
    args.code2wav_model_path = launcher._resolve_code2wav_model_path_from_folder(
        model_path=args.model_path,
        code2wav_model_path=args.code2wav_model_path,
        code2wav_model_folder=args.code2wav_model_folder,
    )

    launcher._launch_speech_server(args)

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    assert code2wav_stage.factory_args["code2wav_model_path"] == (
        "/models/qwen35/qwen3_5_omni_codec_decode_online_0306"
    )


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_torch_compile(launch_server):
    launcher._launch_speech_server(_args(code2wav_enable_torch_compile=True))

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_enable_torch_compile is True
    assert args["enable_torch_compile"] is True


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_coerces_code2wav_bool_strings(launch_server):
    launcher._launch_speech_server(
        _args(
            code2wav_enable_torch_compile="false",
            code2wav_enable_torch_compile_first_chunk="yes",
            code2wav_enable_dynamic_chunk="0",
            code2wav_odeint_method_relaxed="on",
        )
    )

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_enable_torch_compile is False
    assert args["enable_torch_compile"] is False
    assert args["enable_torch_compile_first_chunk"] is True
    assert args["enable_dynamic_chunk"] is False
    assert args["odeint_method_relaxed"] is True


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_rejects_bad_code2wav_bool_string(launch_server):
    with pytest.raises(ValueError, match="code2wav-enable-torch-compile"):
        launcher._launch_speech_server(
            _args(code2wav_enable_torch_compile="maybe")
        )
    launch_server.assert_not_called()


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_codec_eos_token_id(launch_server):
    launcher._launch_speech_server(_args(code2wav_codec_eos_token_id=3000))

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_codec_eos_token_id == 3000
    assert args["codec_eos_token_id"] == 3000


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_sample_rate(launch_server):
    launcher._launch_speech_server(_args(code2wav_sample_rate=48000))

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_sample_rate == 48000
    assert args["sample_rate"] == 48000


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_stream_chunk_size(launch_server):
    launcher._launch_speech_server(_args(code2wav_stream_chunk_size=8))

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_stream_chunk_size == 8
    assert args["stream_chunk_size"] == 8


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_left_context_size(launch_server):
    launcher._launch_speech_server(_args(code2wav_left_context_size=0))

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_left_context_size == 0
    assert args["left_context_size"] == 0


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_dynamic_chunk_args(launch_server):
    launcher._launch_speech_server(
        _args(
            code2wav_enable_dynamic_chunk=True,
            code2wav_dynamic_chunk_sizes=(2, 4),
            code2wav_dynamic_chunk_steps=(1, 1),
        )
    )

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert code2wav_stage.runtime.code2wav_enable_dynamic_chunk is True
    assert args["enable_dynamic_chunk"] is True
    assert args["dynamic_chunk_sizes"] == (2, 4)
    assert args["dynamic_chunk_steps"] == (1, 1)


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_code2wav_perf_runtime_args(launch_server):
    launcher._launch_speech_server(
        _args(
            code2wav_enable_torch_compile_first_chunk=True,
            code2wav_odeint_method="rk4",
            code2wav_odeint_method_relaxed=True,
            code2wav_batched_chunk=2,
            code2wav_frequency="50hz",
            code2wav_dit_quant="fp8",
        )
    )

    config = launch_server.call_args.args[0]
    code2wav_stage = _stage(config, "code2wav")
    args = resolve_stage_factory_args(code2wav_stage, config)
    assert args["enable_torch_compile_first_chunk"] is True
    assert args["odeint_method"] == "rk4"
    assert args["odeint_method_relaxed"] is True
    assert args["batched_chunk"] == 2
    assert args["frequency"] == "50hz"
    assert args["dit_quant"] == "fp8"


@patch("sglang_omni.serve.launch_server")
def test_qwen35_launcher_sets_video_preprocessing_runtime_args(launch_server):
    launcher._launch_speech_server(
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
def test_qwen35_launcher_sets_limit_mm_per_prompt(launch_server):
    launcher._launch_speech_server(
        _args(limit_mm_per_prompt={"image": 2, "video": 1, "audio": 0})
    )

    config = launch_server.call_args.args[0]
    preprocessing_args = _stage(config, "preprocessing").factory_args
    assert preprocessing_args["limit_mm_per_prompt"] == {
        "audio": 0,
        "image": 2,
        "video": 1,
    }


def test_qwen35_launcher_extracts_code2wav_model_path_flag():
    cleaned, code2wav_model_path = launcher._extract_code2wav_model_path(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--code2wav-model-path=/models/code2wav",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert code2wav_model_path == "/models/code2wav"


def test_qwen35_launcher_extracts_code2wav_model_folder_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--code2wav-model-folder",
            "qwen3_5_omni_codec_decode_online_0306",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert (
        qwen35_args["code2wav_model_folder"]
        == "qwen3_5_omni_codec_decode_online_0306"
    )


def test_qwen35_launcher_extracts_preflight_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--preflight",
            "--xvector-info-path=/voices/ref-a",
            "--voice-clone-path",
            "/voices/ref-b",
            "--validate-xvector-pickle",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["preflight"] is True
    assert qwen35_args["xvector_info_paths"] == ["/voices/ref-a", "/voices/ref-b"]
    assert qwen35_args["validate_xvector_pickle"] is True


def test_qwen35_launcher_extracts_talker_model_path_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--talker-model-path=/models/qwen35/talker",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["talker_model_path"] == "/models/qwen35/talker"


def test_qwen35_launcher_extracts_vllm_model_path_aliases():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model",
            "/models/qwen35",
            "--thinker-model=/models/qwen35/thinker",
            "--talker-model",
            "/models/qwen35/talker_lm",
            "--code2wav-model=/models/qwen35/code2wav",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--port", "8008"]
    assert qwen35_args["model_path"] == "/models/qwen35"
    assert qwen35_args["thinker_model_path"] == "/models/qwen35/thinker"
    assert qwen35_args["talker_model_path"] == "/models/qwen35/talker_lm"
    assert qwen35_args["code2wav_model_path"] == "/models/qwen35/code2wav"


def test_qwen35_launcher_extracts_vllm_relative_code2wav_model_as_folder():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model",
            "/models/qwen35",
            "--code2wav-model=code2wav",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--port", "8008"]
    assert qwen35_args["code2wav_model_path"] is None
    assert qwen35_args["code2wav_model_folder"] == "code2wav"


def test_qwen35_launcher_extracts_dtype_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--dtype=bfloat16",
            "--mamba-cache-dtype",
            "float32",
            "--talker-dtype",
            "float16",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["dtype"] == "bfloat16"
    assert qwen35_args["talker_dtype"] == "float16"
    assert qwen35_args["mamba_ssm_dtype"] == "float32"


def test_qwen35_launcher_extracts_max_model_len_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--max-model-len=192000",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["max_model_len"] == 192000


def test_qwen35_launcher_extracts_max_seq_len_to_capture_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--max_seq_len_to_capture=262144",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["max_seq_len_to_capture"] == 262144


def test_qwen35_launcher_extracts_mamba_cache_mode_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--mamba_cache_mode=none",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["mamba_cache_mode"] == "none"


def test_qwen35_launcher_extracts_vllm_kv_transfer_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--thinker-kv-transfer-config",
            '{"kv_connector":"HybridConnector"}',
            "--enable-disaggregated-prefilling",
            "1",
            "--serve-port=29000",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["kv_transfer_config"] == {
        "kv_connector": "HybridConnector"
    }
    assert qwen35_args["enable_disaggregated_prefilling"] is True
    assert qwen35_args["port"] == 29000


def test_qwen35_launcher_extracts_vllm_engine_profile_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--tensor_parallel_size=1",
            "--distributed-executor-backend",
            "mp",
            "--kv-cache-dtype=auto",
            "--enable-expert-parallel",
            "false",
            "--mm-processor-cache-gb=0",
            "--use-omni-engine",
            "--is-thinker",
            "--thinker-only=false",
            "--skip-mm-profiling=true",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["tensor_parallel_size"] == 1
    assert qwen35_args["distributed_executor_backend"] == "mp"
    assert qwen35_args["kv_cache_dtype"] == "auto"
    assert qwen35_args["enable_expert_parallel"] is False
    assert qwen35_args["mm_processor_cache_gb"] == 0
    assert qwen35_args["use_omni_engine"] is True
    assert qwen35_args["is_thinker"] is True
    assert qwen35_args["thinker_only"] is False
    assert qwen35_args["skip_mm_profiling"] is True


def test_qwen35_launcher_extracts_visible_devices_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--thinker_visible_devices=[0,1]",
            "--talker-visible-devices",
            "[2]",
            "--code2wav-visible-devices=2",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["thinker_visible_devices"] == (0, 1)
    assert qwen35_args["talker_visible_devices"] == (2,)
    assert qwen35_args["code2wav_visible_devices"] == (2,)


def test_qwen35_launcher_extracts_vllm_devices_aliases():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--thinker-devices",
            "[0,1]",
            "--talker-devices=2",
            "--code2wav-devices",
            "[3]",
            "--thinker-tensor-parallel-size=2",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["thinker_visible_devices"] == (0, 1)
    assert qwen35_args["talker_visible_devices"] == (2,)
    assert qwen35_args["code2wav_visible_devices"] == (3,)
    assert qwen35_args["tensor_parallel_size"] == 2


def test_qwen35_launcher_extracts_compilation_config_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--compilation_config",
            '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["compilation_config"] == {
        "cudagraph_mode": "FULL_DECODE_ONLY"
    }


def test_qwen35_launcher_extracts_max_num_batched_tokens_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--max-num-batched-tokens",
            "512",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["max_num_batched_tokens"] == 512


def test_qwen35_launcher_extracts_block_size_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--block-size",
            "16",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["page_size"] == 16


def test_qwen35_launcher_extracts_video_preprocessing_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--model-path",
            "/models/qwen35",
            "--image-max-pixels=401408",
            "--video-fps=2",
            "--video-max-frames",
            "128",
            "--video-min-frames=4",
            "--video_max_pixels=401408",
            "--video-seconds-per-chunk",
            "2.5",
            "--position-id-per-seconds=25",
            "--sampling-rate",
            "16000",
            "--timestamp-interval=30",
            "--audio-downsample-times=4",
            "--downsample-chunk-size=100",
            "--port",
            "8008",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35", "--port", "8008"]
    assert qwen35_args["image_max_pixels"] == 401408
    assert qwen35_args["video_fps"] == 2.0
    assert qwen35_args["video_max_frames"] == 128
    assert qwen35_args["video_min_frames"] == 4
    assert qwen35_args["video_max_pixels"] == 401408
    assert qwen35_args["video_seconds_per_chunk"] == 2.5
    assert qwen35_args["video_position_id_per_seconds"] == 25.0
    assert qwen35_args["audio_target_sr"] == 16000
    assert qwen35_args["audio_timestamp_interval"] == 30
    assert qwen35_args["audio_downsample_times"] == 4
    assert qwen35_args["audio_downsample_chunk_size"] == 100


def test_qwen35_launcher_extracts_limit_mm_per_prompt_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--limit-mm-per-prompt",
            '{"image": 2, "video": 1}',
            "--limit-mm-per-prompt-audio=0",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["limit_mm_per_prompt"] == {
        "audio": 0,
        "image": 2,
        "video": 1,
    }


def test_qwen35_launcher_extracts_max_running_request_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--max-running-requests=24",
            "--thinker-max-running-requests",
            "16",
            "--talker_max_running_requests=8",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["max_running_requests"] == 24
    assert qwen35_args["thinker_max_running_requests"] == 16
    assert qwen35_args["talker_max_running_requests"] == 8


def test_qwen35_launcher_extracts_default_talker_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--voice-type=Cherry",
            "--enable-tn=false",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["voice_type"] == "Cherry"
    assert qwen35_args["enable_tn"] is False


def test_qwen35_launcher_extracts_default_generation_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--max-tokens=512",
            "--seed",
            "7",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["max_tokens"] == 512
    assert qwen35_args["seed"] == 7


def test_qwen35_launcher_rejects_bad_max_running_requests():
    with pytest.raises(ValueError, match="thinker-max-running"):
        launcher._launch_speech_server(_args(thinker_max_running_requests=0))


def test_qwen35_launcher_extracts_code2wav_compile_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--enable-torch-compile",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_enable_torch_compile"] is True


def test_qwen35_launcher_extracts_code2wav_codec_eos_token_id_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--code2wav-codec-eos-token-id=3000",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_codec_eos_token_id"] == 3000


def test_qwen35_launcher_extracts_code2wav_sample_rate_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--sample-rate",
            "48000",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_sample_rate"] == 48000


def test_qwen35_launcher_extracts_code2wav_stream_chunk_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--send-chunk-size=8",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_stream_chunk_size"] == 8


def test_qwen35_launcher_extracts_code2wav_left_context_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--code2wav-left-context-size",
            "0",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_left_context_size"] == 0


def test_qwen35_launcher_extracts_code2wav_dynamic_chunk_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--code2wav-enable-dynamic-chunk",
            "--code2wav-dynamic-chunk-sizes=2,4,8",
            "--code2wav-dynamic-chunk-steps",
            "8 4 1",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_enable_dynamic_chunk"] is True
    assert qwen35_args["code2wav_dynamic_chunk_sizes"] == (2, 4, 8)
    assert qwen35_args["code2wav_dynamic_chunk_steps"] == (8, 4, 1)


def test_qwen35_launcher_extracts_vllm_code2wav_runtime_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--enable-torch-compile-first-chunk",
            "--odeint-method",
            "RK4",
            "--odeint-method-relaxed=true",
            "--batched-chunk=2",
            "--code2wav-frequency=1",
            "--code2wav-dit-quantization",
            "FP8",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_enable_torch_compile_first_chunk"] is True
    assert qwen35_args["code2wav_odeint_method"] == "rk4"
    assert qwen35_args["code2wav_odeint_method_relaxed"] is True
    assert qwen35_args["code2wav_batched_chunk"] == 2
    assert qwen35_args["code2wav_frequency"] == "50hz"
    assert qwen35_args["code2wav_dit_quant"] == "fp8"


def test_qwen35_launcher_extracts_vllm_ar_server_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--enable-prefix-caching=false",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "--quantization",
            "NVFP4",
            "--talker-quantization=FP8",
            "--disable-mtp",
            "--max-num-seqs",
            "24",
            "--gpu-memory-utilization",
            "0.55",
            "--talker-gpu-memory-utilization=0.7",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["enable_prefix_caching"] is False
    assert qwen35_args["enable_chunked_prefill"] is False
    assert qwen35_args["enforce_eager"] is True
    assert qwen35_args["thinker_quantization"] == "nvfp4"
    assert qwen35_args["talker_quantization"] == "fp8"
    assert qwen35_args["disable_mtp"] is True
    assert qwen35_args["max_running_requests"] == 24
    assert qwen35_args["mem_fraction_static"] == 0.55
    assert qwen35_args["talker_mem_fraction_static"] == 0.7


def test_qwen35_launcher_extracts_speculative_config_flag():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--speculative-config",
            '{"method":"qwen3_omni_next_thinker_mtp","num_speculative_tokens":4}',
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["speculative_config"] == {
        "method": "qwen3_omni_next_thinker_mtp",
        "num_speculative_tokens": 4,
    }


def test_qwen35_launcher_extracts_namespaced_code2wav_runtime_flags():
    cleaned, qwen35_args = launcher._extract_qwen35_extra_args(
        [
            "server.py",
            "--no-code2wav-torch-compile-first-chunk",
            "--code2wav-odeint-method=euler",
            "--no-code2wav-odeint-method-relaxed",
            "--code2wav-batched-chunk",
            "1",
            "--model-path",
            "/models/qwen35",
        ]
    )

    assert cleaned == ["server.py", "--model-path", "/models/qwen35"]
    assert qwen35_args["code2wav_enable_torch_compile_first_chunk"] is False
    assert qwen35_args["code2wav_odeint_method"] == "euler"
    assert qwen35_args["code2wav_odeint_method_relaxed"] is False
    assert qwen35_args["code2wav_batched_chunk"] == 1
