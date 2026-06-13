# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

from sglang_omni.models.qwen3_5_omni.preflight import (
    format_preflight_report,
    format_vllm_profile_report,
    load_vllm_profile_file,
    load_vllm_profile_payload,
    run_qwen35_preflight,
    run_vllm_profile_preflight,
    shell_join_cli_args,
    suggested_vllm_profile_cli_args,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _touch_code2wav(path, *, codebook_nums=16):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(
        "model:\n"
        "  type: public_v1\n"
        "  dac:\n"
        f"    codebook_nums: {codebook_nums}\n"
    )
    (path / "model_weights.pt").write_bytes(b"placeholder")


def _touch_hf_weights(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"placeholder")


def _touch_xvector_info(path, feat_bytes=b"placeholder", info=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "feat.pkl").write_bytes(feat_bytes)
    _write_json(
        path / "info.json",
        info
        or {
            "talker_system_instruct": "用参考音色自然说话。",
            "language_type": "zh",
        },
    )


def _touch_processor_assets(path):
    _write_json(
        path / "processor_config.json",
        {"processor_class": "Qwen3OmniNextProcessor"},
    )
    _write_json(
        path / "tokenizer_config.json",
        {"chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"},
    )
    _write_json(path / "tokenizer.json", {"version": "1.0"})


def _thinker_config():
    return {
        "model_type": "qwen3_omni_next_thinker",
        "text_config": {
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 32,
        },
        "audio_config": {
            "d_model": 8,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "encoder_layers": 1,
            "num_mel_bins": 4,
            "max_source_positions": 32,
            "n_window": 4,
            "n_window_infer": 4,
            "downsample_hidden_size": 4,
            "output_dim": 8,
            "activation_function": "gelu",
            "downsample_times": 4,
            "downsample_chunk_size": 100,
        },
        "vision_config": {
            "depth": 4,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 4,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [1, 3],
        },
        "audio_token_id": 1,
        "image_token_id": 2,
        "video_token_id": 3,
        "vision_start_token_id": 4,
        "vision_end_token_id": 5,
    }


def _talker_config():
    return {
        "model_type": "qwen3_omni_next_talker",
        "architectures": ["Qwen3OmniNextTalkerModel"],
        "text_config": {
            "vocab_size": 16,
            "text_vocab_size": 128,
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
        },
        "code_predictor_config": {
            "num_code_groups": 16,
            "vocab_size": 16,
            "hidden_size": 4,
            "talker_hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "head_dim": 2,
            "hidden_act": "silu",
        },
        "accept_hidden_layer": 24,
        "codec_bos_id": 10,
        "codec_eos_token_id": 11,
        "codec_nothink_id": 12,
        "codec_think_bos_id": 13,
        "codec_think_eos_id": 14,
        "codec_pad_id": 15,
        "speaker_id": {"tina": 0},
    }


def _root_config(**overrides):
    data = {
        "model_type": "qwen3_omni_next",
        "thinker_config": _thinker_config(),
        "talker_config": _talker_config(),
        "tts_bos_token_id": 20,
        "tts_eos_token_id": 21,
        "tts_pad_token_id": 22,
        "im_start_token_id": 23,
        "im_end_token_id": 24,
        "system_token_id": 25,
        "user_token_id": 26,
        "assistant_token_id": 27,
    }
    data.update(overrides)
    return data


def _profile_checks_by_key(report):
    return {check.key: check for check in report.checks}


def test_qwen35_preflight_script_runs_directly_from_repo_root():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/qwen35_omni_preflight.py",
            "--vllm-engine-args-json",
            '{"dtype":"bfloat16"}',
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Qwen3.5-Omni vLLM profile preflight PASS" in result.stdout


def test_qwen35_preflight_script_disable_mtp_allows_mtp_profile(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    profile = tmp_path / "mtp.config"
    profile.write_text(
        json.dumps(
            {
                "engine_args": {
                    "dtype": "bfloat16",
                    "speculative_config": {
                        "method": "qwen3_omni_next_thinker_mtp",
                        "num_speculative_tokens": 4,
                    },
                }
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/qwen35_omni_preflight.py",
            "--vllm-profile",
            str(profile),
            "--disable-mtp",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Qwen3.5-Omni vLLM profile preflight PASS" in result.stdout
    assert "Qwen3.5 MTP config is ignored" in result.stdout


def test_qwen35_vllm_profile_preflight_accepts_mapped_profile():
    report = run_vllm_profile_preflight(
        {
            "engine_args": {
                "distributed_executor_backend": "mp",
                "model": "/models/qwen35",
                "thinker_model": "/models/qwen35/thinker",
                "talker_model": "/models/qwen35/talker_lm",
                "code2wav_model": "/models/qwen35/code2wav",
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.9,
                "thinker_gpu_memory_utilization": 0.95,
                "talker_gpu_memory_utilization": 0.8,
                "kv_cache_dtype": "auto",
                "thinker_kv_transfer_config": "{}",
                "mamba_cache_dtype": "float32",
                "mamba_cache_mode": "none",
                "quantization": "FP8",
                "mm_processor_cache_gb": 0,
                "max_model_len": 262144,
                "max_mm_len": 256000,
                "max_num_batched_tokens": 32768,
                "max_num_seqs": 32,
                "max_seq_len_to_capture": 262144,
                "tensor_parallel_size": 2,
                "thinker_visible_devices": [0, 1],
                "talker_visible_devices": [2],
                "code2wav_visible_devices": [2],
                "limit_mm_per_prompt": {"audio": 2048, "video": 512},
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "enforce_eager": False,
                "override_video_max_pixels": True,
                "use_omni_engine": True,
                "skip_mm_profiling": True,
                "video_needs_metadata": True,
                "code2wav_model_folder": "qwen3_5_omni_codec_decode_online_0306",
                "code2wav_enable_torch_compile": True,
                "code2wav_enable_torch_compile_first_chunk": True,
                "code2wav_odeint_method": "rk4",
                "code2wav_odeint_method_relaxed": False,
                "code2wav_batched_chunk": 2,
                "code2wav_frequency": 1,
                "code2wav_dit_quant": "fp8",
                "code2wav_sample_rate": 24000,
                "sample_rate": 24000,
                "code2wav_codec_eos_token_id": 2150,
                "code2wav_left_context_size": 25,
                "code2wav_enable_dynamic_chunk": True,
                "code2wav_dynamic_batch": True,
                "code2wav_dynamic_chunk_sizes": [2, 4, 8],
                "code2wav_dynamic_chunk_steps": "8,4,1",
                "send_chunk_size": 8,
                "serve_port": 29000,
                "voice_type": "Cherry",
                "warmup_voice_type": "f245",
                "enable_tn": True,
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "use_inductor": False,
                },
                "speculative_config": {},
            }
        },
        source="unit-profile",
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["dtype"].status == "mapped"
    assert checks["mamba_cache_mode"].status == "noop"
    assert checks["thinker_kv_transfer_config"].status == "noop"
    assert checks["tensor_parallel_size"].status == "mapped"
    assert checks["model"].suggested_args == (
        "--model-path",
        "/models/qwen35",
    )
    assert checks["thinker_model"].status == "noop"
    assert checks["talker_model"].suggested_args == (
        "--talker-model-path",
        "/models/qwen35/talker_lm",
    )
    assert checks["code2wav_model"].suggested_args == (
        "--code2wav-model-path",
        "/models/qwen35/code2wav",
    )
    assert checks["quantization"].suggested_args == (
        "--thinker-quantization",
        "fp8",
    )
    assert checks["thinker_visible_devices"].suggested_args == (
        "--thinker-visible-devices",
        "[0,1]",
    )
    assert checks["override_video_max_pixels"].suggested_args == (
        "--override-video-max-pixels",
        "true",
    )
    assert checks["code2wav_frequency"].suggested_args == (
        "--code2wav-frequency",
        "50hz",
    )
    assert checks["code2wav_sample_rate"].suggested_args == (
        "--code2wav-sample-rate",
        "24000",
    )
    assert checks["sample_rate"].suggested_args == ("--sample-rate", "24000")
    assert checks["voice_type"].suggested_args == ("--voice-type", "Cherry")
    assert checks["warmup_voice_type"].status == "noop"
    assert checks["enable_tn"].suggested_args == ("--enable-tn",)
    assert checks["code2wav_codec_eos_token_id"].suggested_args == (
        "--code2wav-codec-eos-token-id",
        "2150",
    )
    assert checks["code2wav_enable_dynamic_chunk"].suggested_args == (
        "--code2wav-enable-dynamic-chunk",
    )
    assert checks["code2wav_dynamic_batch"].suggested_args == (
        "--code2wav-enable-dynamic-chunk",
    )
    assert checks["code2wav_dynamic_chunk_sizes"].suggested_args == (
        "--code2wav-dynamic-chunk-sizes",
        "2,4,8",
    )
    assert checks["code2wav_dynamic_chunk_steps"].suggested_args == (
        "--code2wav-dynamic-chunk-steps",
        "8,4,1",
    )
    assert checks["serve_port"].suggested_args == ("--port", "29000")
    cli_args = suggested_vllm_profile_cli_args(report)
    assert "--dtype" in cli_args
    assert "--code2wav-odeint-method" in cli_args
    assert "--code2wav-frequency" in cli_args
    assert "--dtype bfloat16" in shell_join_cli_args(cli_args)
    formatted = format_vllm_profile_report(report)
    assert "Qwen3.5-Omni vLLM profile preflight PASS" in formatted
    assert "[mapped-cli]" in formatted


def test_qwen35_vllm_profile_preflight_accepts_perf_v2_h20_shape():
    # 中文说明：字段形状来自 vLLM dev/qwenc_perf_v2 的
    # util/vllmgen/configs/qwen3.5-omni/23b_fp8/h20.config。
    report = run_vllm_profile_preflight(
        {
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
                "code2wav_dynamic_batch": True,
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
            }
        },
        source="perf_v2_23b_fp8_h20",
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["use_omni_rpc_engine"].status == "noop"
    assert checks["thinker_only"].status == "noop"
    assert checks["code2wav_model_folder"].suggested_args == (
        "--code2wav-model-folder",
        "qwen3_5_omni_codec_decode_online_0306",
    )
    assert checks["code2wav_dynamic_batch"].suggested_args == (
        "--code2wav-enable-dynamic-chunk",
    )
    assert not any(check.status in {"unknown", "unsupported"} for check in report.checks)


def test_qwen35_vllm_profile_preflight_accepts_v35_online_speech_shape():
    # 中文说明：字段形状贴近 vLLM dev/qwenc_perf_v2 的
    # examples/offline_inference/qwen_omni_v35.py 在线 speech thinker profile。
    # 它和 256k eval profile 不同，默认使用 192000 context 与 960/960/960
    # 多模态 guard；这些字段到 SGLang 后不应被误判为 unknown。
    report = run_vllm_profile_preflight(
        {
            "engine_args": {
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.9,
                "enable_chunked_prefill": True,
                "tensor_parallel_size": 1,
                "enforce_eager": False,
                "distributed_executor_backend": "mp",
                "limit_mm_per_prompt": {
                    "audio": 960,
                    "image": 960,
                    "video": 960,
                },
                "max_model_len": 192000,
                "max_num_batched_tokens": 32768,
                "max_num_seqs": 32,
                "block_size": 256,
                "enable_prefix_caching": True,
                "enable_prompt_embeds": False,
                "disable_log_stats": False,
                "mm_processor_cache_type": "shm",
                "video_needs_metadata": True,
                "quantization": "none",
                "thinker_visible_devices": [0],
                "talker_visible_devices": [1],
                "code2wav_visible_devices": [1],
                "code2wav_model": "code2wav",
                "code2wav_enable_torch_compile": True,
                "disable_mtp": True,
                "speculative_config": {
                    "method": "qwen3_omni_next_thinker_mtp",
                    "num_speculative_tokens": 4,
                },
            }
        },
        source="qwen_omni_v35_online_speech",
    )

    assert report.ok
    assert not any(check.status in {"unknown", "unsupported"} for check in report.checks)
    checks = _profile_checks_by_key(report)
    assert checks["max_model_len"].suggested_args == ("--max-model-len", "192000")
    assert checks["limit_mm_per_prompt"].suggested_args == (
        "--limit-mm-per-prompt",
        '{"audio":960,"image":960,"video":960}',
    )
    assert checks["mm_processor_cache_type"].status == "noop"
    assert checks["video_needs_metadata"].status == "noop"
    assert checks["enable_prompt_embeds"].status == "noop"
    assert checks["disable_log_stats"].status == "noop"
    assert checks["trust_remote_code"].status == "noop"
    assert checks["code2wav_model"].suggested_args == (
        "--code2wav-model-folder",
        "code2wav",
    )
    assert checks["disable_mtp"].suggested_args == ("--disable-mtp",)
    assert checks["speculative_config"].status == "noop"


def test_qwen35_vllm_profile_preflight_flags_perf_v2_400b_decode_gaps():
    # 中文说明：字段形状来自 vLLM dev/qwenc_perf_v2 的
    # util/vllmgen/configs/qwen3.5-omni/400b_nvfp4_decode/l20d.config。
    report = run_vllm_profile_preflight(
        {
            "engine_args": {
                "thinker_gpu_memory_utilization": "0.9",
                "gpu_memory_utilization": "0.9",
                "code2wav_model_folder": "code2wav",
                "use_omni_rpc_engine": True,
                "is_thinker": True,
                "thinker_only": False,
                "dtype": "bfloat16",
                "send_chunk_size": 8,
                "mamba_cache_mode": "light",
                "mamba_cache_dtype": "float32",
                "max_num_seqs": 256,
                "max_model_len": 262144,
                "max_seq_len_to_capture": 262144,
                "max_num_batched_tokens": 512,
                "enforce_eager": False,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "tensor_parallel_size": 1,
                "enable_expert_parallel": True,
                "thinker_visible_devices": [0],
                "limit_mm_per_prompt": {
                    "audio": 2048,
                    "video": 512,
                    "image": 2048,
                },
                "distributed_executor_backend": "mp",
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "use_inductor": False,
                    "pass_config": {
                        "fuse_norm_quant": False,
                        "fuse_act_quant": False,
                        "fuse_attn_quant": False,
                    },
                },
                "use_zero_shot": True,
                "kv_transfer_config": {
                    "kv_connector": "HybridConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {"backend": "kvt"},
                },
                "enable_disaggregated_prefilling": "1",
                "skip_mm_profiling": True,
                "mm_processor_cache_gb": 0,
                "max_mm_len": 256000,
                "override_video_max_pixels": True,
            }
        },
        source="perf_v2_400b_nvfp4_decode_l20d",
    )

    assert not report.ok
    assert not any(check.status == "unknown" for check in report.checks)
    checks = _profile_checks_by_key(report)
    for key in (
        "mamba_cache_mode",
        "enable_expert_parallel",
        "kv_transfer_config",
        "enable_disaggregated_prefilling",
    ):
        assert checks[key].status == "unsupported"
    assert checks["max_mm_len"].suggested_args == ("--max-mm-len", "256000")
    assert checks["tensor_parallel_size"].suggested_args == (
        "--thinker-tp-size",
        "1",
    )
    assert checks["thinker_visible_devices"].suggested_args == (
        "--thinker-visible-devices",
        "[0]",
    )
    assert checks["override_video_max_pixels"].suggested_args == (
        "--override-video-max-pixels",
        "true",
    )
    assert checks["use_zero_shot"].status == "warning"
    assert "xvector-only zero-shot branch" in checks["use_zero_shot"].message


def test_qwen35_vllm_profile_preflight_maps_disabled_zero_shot_as_noop():
    report = run_vllm_profile_preflight({"use_zero_shot": False})

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["use_zero_shot"].status == "noop"


def test_qwen35_vllm_profile_preflight_accepts_perf_v2_400b_rpc_shape():
    # 中文说明：字段形状来自 vLLM dev/qwenc_perf_v2 的
    # util/vllmgen/configs/qwen3.5-omni/400b_nvfp4_rpc/l20c.config。
    report = run_vllm_profile_preflight(
        {
            "engine_args": {
                "thinker_gpu_memory_utilization": "0.8",
                "code2wav_model_folder": (
                    "qwen3_5_omni_codec_decode_online_0306"
                ),
                "use_omni_rpc_engine": True,
                "is_thinker": True,
                "thinker_only": False,
                "dtype": "bfloat16",
                "send_chunk_size": 8,
                "mamba_cache_dtype": "float32",
                "max_num_seqs": 32,
                "max_model_len": 262144,
                "max_seq_len_to_capture": 262144,
                "max_num_batched_tokens": 32768,
                "enforce_eager": False,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "thinker_visible_devices": [0, 1],
                "limit_mm_per_prompt": {
                    "audio": 2048,
                    "video": 512,
                    "image": 2048,
                },
                "distributed_executor_backend": "mp",
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "use_inductor": False,
                    "pass_config": {
                        "fuse_norm_quant": False,
                        "fuse_act_quant": False,
                        "fuse_attn_quant": False,
                    },
                },
            }
        },
        source="perf_v2_400b_nvfp4_rpc_l20c",
    )

    assert report.ok
    assert not any(check.status in {"unknown", "unsupported"} for check in report.checks)
    checks = _profile_checks_by_key(report)
    assert checks["thinker_visible_devices"].suggested_args == (
        "--thinker-visible-devices",
        "[0,1]",
    )
    assert checks["code2wav_model_folder"].suggested_args == (
        "--code2wav-model-folder",
        "qwen3_5_omni_codec_decode_online_0306",
    )


def test_qwen35_vllm_profile_preflight_maps_standalone_thinker_path():
    report = run_vllm_profile_preflight(
        {"thinker_model": "/models/qwen35/thinker"}
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["thinker_model"].suggested_args == (
        "--model-path",
        "/models/qwen35/thinker",
    )


def test_qwen35_vllm_profile_preflight_maps_vllm_rpc_device_aliases():
    report = run_vllm_profile_preflight(
        {
            "thinker_devices": [0, 1],
            "talker_devices": [2],
            "code2wav_devices": [3],
            "thinker_tensor_parallel_size": 2,
        }
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["thinker_devices"].suggested_args == (
        "--thinker-visible-devices",
        "[0,1]",
    )
    assert checks["talker_devices"].suggested_args == (
        "--talker-visible-devices",
        "[2]",
    )
    assert checks["code2wav_devices"].suggested_args == (
        "--code2wav-visible-devices",
        "[3]",
    )
    assert checks["thinker_tensor_parallel_size"].suggested_args == (
        "--thinker-tp-size",
        "2",
    )


def test_qwen35_vllm_profile_preflight_handles_rpc_script_controls():
    report = run_vllm_profile_preflight(
        {
            "prompt": "audio",
            "num_prompts": 2,
            "max_tokens": 512,
            "seed": 0,
            "talker_quantization": "fp8",
            "do_wave": True,
            "model_to_run": "thinker",
            "thinker_data_parallel_size": 1,
            "code2wav_data_parallelism": 1,
            "id_start_index": 1,
            "pd_prefill_host": "127.0.0.1",
        }
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["prompt"].status == "noop"
    assert checks["num_prompts"].status == "noop"
    assert checks["do_wave"].status == "noop"
    assert checks["model_to_run"].status == "noop"
    assert checks["thinker_data_parallel_size"].status == "noop"
    assert checks["code2wav_data_parallelism"].status == "noop"
    assert checks["id_start_index"].status == "noop"
    assert checks["pd_prefill_host"].status == "noop"
    assert checks["max_tokens"].suggested_args == ("--max-tokens", "512")
    assert checks["seed"].suggested_args == ("--seed", "0")
    assert checks["talker_quantization"].suggested_args == (
        "--talker-quantization",
        "fp8",
    )
    cli_args = suggested_vllm_profile_cli_args(report)
    assert "--max-tokens" in cli_args
    assert "--seed" in cli_args
    assert "--talker-quantization" in cli_args
    assert "--num-prompts" not in cli_args
    assert "--id-start-index" not in cli_args


def test_qwen35_vllm_profile_preflight_handles_server_and_offline_controls():
    report = run_vllm_profile_preflight(
        {
            "host": "127.0.0.1",
            "port": 8008,
            "text_only": True,
            "prompt": "image-video-audio",
            "omni_video_fps": 2,
            "conversation": True,
            "token_prompt": False,
            "use_torchvision": True,
            "do_warmup": True,
            "multi_waveforms": True,
            "translation_prefix": "Translate this:",
            "asr_language": "<lang_chinese>",
            "prompt_file": "cases.jsonl",
            "chat_prompt_log_file": "chat_prompts.jsonl",
            "rtc_test_dir": "rtc_cases",
            "rtc_audio_chunks": "[6,12,20]",
            "rtc_video_chunks": "[6,12,20]",
            "rtc_max_chunks_per_turn": 6,
            "audio_video_alternate_freq": 0,
        }
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["host"].suggested_args == ("--host", "127.0.0.1")
    assert checks["port"].suggested_args == ("--port", "8008")
    assert checks["text_only"].suggested_args == ("--text-only",)
    assert checks["prompt"].status == "noop"
    assert checks["omni_video_fps"].suggested_args == ("--video-fps", "2")
    for key in (
        "conversation",
        "token_prompt",
        "use_torchvision",
        "do_warmup",
        "multi_waveforms",
        "translation_prefix",
        "asr_language",
        "prompt_file",
        "chat_prompt_log_file",
        "rtc_test_dir",
        "rtc_audio_chunks",
        "rtc_video_chunks",
        "rtc_max_chunks_per_turn",
        "audio_video_alternate_freq",
    ):
        assert checks[key].status == "noop"


def test_qwen35_vllm_profile_preflight_rejects_rpc_pd_runtime_controls():
    report = run_vllm_profile_preflight(
        {
            "model_to_run": "talker",
            "thinker_data_parallel_size": 2,
            "code2wav_data_parallelism": 2,
            "v6d_send_socket": "/tmp/send.sock",
            "v6d_recv_socket": "/tmp/recv.sock",
            "v6d_prefill_recv_socket": "/tmp/prefill.sock",
            "remote_endpoint": "127.0.0.1:1234",
            "pd_prefill_host": "10.0.0.5",
            "pd_prefill_rpc_port": 29000,
            "pd_prefill_endpoint": "10.0.0.5:29000",
        }
    )

    assert not report.ok
    checks = _profile_checks_by_key(report)
    for key in (
        "model_to_run",
        "thinker_data_parallel_size",
        "code2wav_data_parallelism",
        "v6d_send_socket",
        "v6d_recv_socket",
        "v6d_prefill_recv_socket",
        "remote_endpoint",
        "pd_prefill_host",
        "pd_prefill_rpc_port",
        "pd_prefill_endpoint",
    ):
        assert checks[key].status == "unsupported"


def test_qwen35_vllm_profile_preflight_rejects_unmapped_video_rtc_modes():
    report = run_vllm_profile_preflight(
        {
            "legacy_omni_video": True,
            "vl_fastv": True,
            "enable_rtc_simulate": True,
        }
    )

    assert not report.ok
    checks = _profile_checks_by_key(report)
    assert checks["legacy_omni_video"].status == "unsupported"
    assert checks["vl_fastv"].status == "unsupported"
    assert checks["enable_rtc_simulate"].status == "unsupported"


def test_qwen35_vllm_profile_preflight_accepts_code2wav_script_aliases():
    report = run_vllm_profile_preflight(
        {
            "enable_torch_compile": "true",
            "enable_torch_compile_first_chunk": "false",
            "odeint_method": "EULER",
            "odeint_method_relaxed": "false",
            "batched_chunk": "1",
            "enable_dynamic_chunk": "false",
            "dynamic_chunk_sizes": "2 4",
            "dynamic_chunk_steps": (8, 1),
            "left_context_size": "0",
        }
    )

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["enable_torch_compile"].suggested_args == (
        "--code2wav-enable-torch-compile",
    )
    assert checks["enable_torch_compile_first_chunk"].suggested_args == (
        "--no-code2wav-torch-compile-first-chunk",
    )
    assert checks["odeint_method"].suggested_args == (
        "--code2wav-odeint-method",
        "euler",
    )
    assert checks["batched_chunk"].suggested_args == (
        "--code2wav-batched-chunk",
        "1",
    )
    assert checks["enable_dynamic_chunk"].suggested_args == (
        "--no-code2wav-dynamic-chunk",
    )
    assert checks["dynamic_chunk_sizes"].suggested_args == (
        "--code2wav-dynamic-chunk-sizes",
        "2,4",
    )
    assert checks["dynamic_chunk_steps"].suggested_args == (
        "--code2wav-dynamic-chunk-steps",
        "8,1",
    )
    assert checks["left_context_size"].suggested_args == (
        "--code2wav-left-context-size",
        "0",
    )


def test_qwen35_vllm_profile_preflight_detects_page_size_conflict():
    report = run_vllm_profile_preflight(
        {
            "block_size": 16,
            "page_size": 32,
        }
    )

    assert not report.ok
    checks = [check for check in report.checks if check.key == "page_size"]
    assert checks[-1].status == "unsupported"
    assert "conflict" in checks[-1].message


def test_qwen35_vllm_profile_preflight_rejects_unknown_keys():
    report = run_vllm_profile_preflight(
        {
            "dtype": "bfloat16",
            "future_perf_v2_flag": True,
        }
    )

    assert not report.ok
    checks = _profile_checks_by_key(report)
    assert checks["future_perf_v2_flag"].status == "unknown"
    assert "profile preflight FAIL" in format_vllm_profile_report(report)


def test_qwen35_vllm_profile_preflight_rejects_unmapped_runtime_paths():
    report = run_vllm_profile_preflight(
        {
            "mamba_cache_mode": "light",
            "kv_transfer_config": {"kv_connector": "HybridConnector"},
            "thinker_kv_transfer_config": {"kv_connector": "HybridConnector"},
            "enable_disaggregated_prefilling": 1,
            "enable_expert_parallel": True,
            "speculative_config": {
                "method": "qwen3_omni_next_thinker_mtp",
                "num_speculative_tokens": 4,
            },
            "compilation_config": {"use_inductor": True},
            "max_model_len": 1024,
            "max_mm_len": 2048,
        }
    )

    assert not report.ok
    checks = _profile_checks_by_key(report)
    for key in (
        "mamba_cache_mode",
        "kv_transfer_config",
        "thinker_kv_transfer_config",
        "enable_disaggregated_prefilling",
        "enable_expert_parallel",
        "speculative_config",
        "compilation_config",
        "max_mm_len",
    ):
        assert checks[key].status == "unsupported"


def test_qwen35_vllm_profile_preflight_rejects_bad_code2wav_runtime_values():
    report = run_vllm_profile_preflight(
        {
            "code2wav_odeint_method": "bogus",
            "code2wav_frequency": "48hz",
            "code2wav_dit_quant": "int4",
            "code2wav_sample_rate": 0,
            "sample_rate": 0,
            "voice_type": " ",
            "code2wav_codec_eos_token_id": -1,
            "code2wav_left_context_size": -1,
            "code2wav_dynamic_chunk_sizes": "2,0",
            "code2wav_dynamic_chunk_steps": [],
        }
    )

    assert not report.ok
    checks = _profile_checks_by_key(report)
    assert checks["code2wav_odeint_method"].status == "unsupported"
    assert checks["code2wav_frequency"].status == "unsupported"
    assert checks["code2wav_dit_quant"].status == "unsupported"
    assert checks["code2wav_sample_rate"].status == "unsupported"
    assert checks["sample_rate"].status == "unsupported"
    assert checks["voice_type"].status == "unsupported"
    assert checks["code2wav_codec_eos_token_id"].status == "unsupported"
    assert checks["code2wav_left_context_size"].status == "unsupported"
    assert checks["code2wav_dynamic_chunk_sizes"].status == "unsupported"
    assert checks["code2wav_dynamic_chunk_steps"].status == "unsupported"


def test_qwen35_vllm_profile_preflight_maps_disabled_tn():
    report = run_vllm_profile_preflight({"enable_tn": False})

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["enable_tn"].suggested_args == ("--disable-tn",)


def test_qwen35_vllm_profile_preflight_loads_engine_args_file(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "name": "h20-smoke",
                "engine_args": {
                    "dtype": "bfloat16",
                    "tensor_parallel_size": 1,
                },
            }
        )
    )

    engine_args = load_vllm_profile_file(str(profile))
    report = run_vllm_profile_preflight(engine_args, source=str(profile))

    assert report.ok
    checks = _profile_checks_by_key(report)
    assert checks["dtype"].suggested_args == ("--dtype", "bfloat16")
    assert checks["tensor_parallel_size"].suggested_args == ("--thinker-tp-size", "1")


def test_qwen35_vllm_profile_preflight_reports_envs_without_failing(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "envs": {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_OMNI_PREPROCESS_ON_GPU": "1",
                    "VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING": "1",
                    "VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE": "0",
                    "CUSTOM_PROFILE_ENV": "x",
                },
                "engine_args": {"dtype": "bfloat16"},
            }
        )
    )

    assert load_vllm_profile_file(str(profile)) == {"dtype": "bfloat16"}
    payload = load_vllm_profile_payload(str(profile))
    report = run_vllm_profile_preflight(payload, source=str(profile))

    assert report.ok
    env_checks = {check.key: check for check in report.env_checks}
    assert env_checks["SGLANG_JIT_DEEPGEMM_PRECOMPILE"].status == "mapped"
    assert env_checks["NCCL_DEBUG"].status == "warning"
    assert env_checks["VLLM_OMNI_PREPROCESS_ON_GPU"].status == "noop"
    assert env_checks["VLLM_OMNI_TALKER_USE_EXTERNAL_EMBEDDING"].status == "noop"
    assert env_checks["VLLM_OMNI_USE_V3_RTC_PROMPT_STYLE"].status == "noop"
    assert env_checks["CUSTOM_PROFILE_ENV"].status == "warning"
    formatted = format_vllm_profile_report(report)
    assert "[envs]" in formatted
    assert "[env:mapped] SGLANG_JIT_DEEPGEMM_PRECOMPILE" in formatted


def test_qwen35_preflight_passes_root_checkpoint_layout(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "qwen3_5_omni_codec_decode_online_0306")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any("using code2wav" in issue.message for issue in report.issues)


def test_qwen35_preflight_passes_split_checkpoint_with_explicit_code2wav(tmp_path):
    model_root = tmp_path / "qwen35"
    code2wav_root = tmp_path / "code2wav"
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", _thinker_config())
    _write_json(model_root / "talker" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(code2wav_root)

    report = run_qwen35_preflight(
        str(model_root),
        code2wav_model_path=str(code2wav_root),
    )

    assert report.ok
    assert not any(issue.severity == "warning" for issue in report.issues)


def test_qwen35_preflight_accepts_explicit_code2wav_checkpoint_file(tmp_path):
    model_root = tmp_path / "qwen35"
    code2wav_root = tmp_path / "code2wav"
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", _thinker_config())
    _write_json(model_root / "talker" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(code2wav_root)

    report = run_qwen35_preflight(
        str(model_root),
        code2wav_model_path=str(code2wav_root / "model_weights.pt"),
    )

    assert report.ok
    assert not any(issue.severity == "warning" for issue in report.issues)


def test_qwen35_preflight_accepts_split_thinker_mtp_config(tmp_path):
    model_root = tmp_path / "qwen35"
    code2wav_root = tmp_path / "code2wav"
    thinker = _thinker_config()
    thinker["model_type"] = "qwen3_omni_next_thinker_mtp"
    thinker["text_config"]["mtp_num_hidden_layers"] = 1
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", thinker)
    _write_json(model_root / "talker" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(code2wav_root)

    report = run_qwen35_preflight(
        str(model_root),
        code2wav_model_path=str(code2wav_root),
    )

    assert report.ok
    assert not any(
        "unsupported model_type" in issue.message or "missing model_type" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.severity == "warning"
        and "runs the base thinker AR model" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_checks_processor_assets_in_split_thinker_dir(tmp_path):
    model_root = tmp_path / "qwen35"
    code2wav_root = tmp_path / "code2wav"
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", _thinker_config())
    _write_json(model_root / "talker" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    # 中文说明：runtime 会从 root/thinker 加载 HF processor/tokenizer；
    # preflight 也必须检查同一路径，避免 split checkpoint 被误报 root 缺资产。
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(code2wav_root)

    report = run_qwen35_preflight(
        str(model_root),
        code2wav_model_path=str(code2wav_root),
    )

    assert report.ok
    assert not any(
        "missing processor config" in issue.message for issue in report.issues
    )
    assert not any(
        "missing tokenizer_config.json" in issue.message for issue in report.issues
    )
    assert not any(
        "missing tokenizer vocabulary" in issue.message for issue in report.issues
    )
    assert not any("missing chat template" in issue.message for issue in report.issues)


def test_qwen35_preflight_accepts_split_talker_lm_checkpoint(tmp_path):
    model_root = tmp_path / "qwen35"
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", _thinker_config())
    _write_json(model_root / "talker_lm" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker_lm")
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert not any(
        "missing split talker_lm/talker config" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_accepts_runtime_prompt_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["talker_language_id"] = {"zh": "7", "en": 8}
    thinker["talker_assistant_prompt_id_mapping"] = {"happy": [70, "71"]}
    talker = _talker_config()
    talker["speaker_system_prompt_id"] = {"f6009": [80, "81"]}
    _write_json(
        model_root / "config.json",
        _root_config(
            thinker_config=thinker,
            talker_config=talker,
            max_thinker_to_talker_mm_tokens="1024",
        ),
    )
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok


def test_qwen35_preflight_accepts_voice_map_matching_speaker_id(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["speaker_id"] = {"custom": 2, "f6009": 3}
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _write_json(model_root / "voice_map.json", {"Cherry": "custom"})
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok


def test_qwen35_preflight_warns_voice_map_unknown_speaker(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _write_json(model_root / "voice_map.json", {"Cherry": "missing"})
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any(
        issue.severity == "warning"
        and "voice_map.json['Cherry'] maps to 'missing'" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_checks_parent_voice_map_for_split_talker(tmp_path):
    model_root = tmp_path / "qwen35"
    code2wav_root = tmp_path / "code2wav"
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", _thinker_config())
    talker = _talker_config()
    talker["speaker_id"] = {"studio_spk": 2}
    _write_json(model_root / "talker_lm" / "config.json", talker)
    _write_json(model_root / "voice_map.json", {"Studio": "studio_spk"})
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker_lm")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(code2wav_root)

    report = run_qwen35_preflight(
        str(model_root),
        code2wav_model_path=str(code2wav_root),
    )

    assert report.ok


def test_qwen35_preflight_rejects_invalid_runtime_prompt_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["talker_language_id"] = {"zh": -1}
    thinker["talker_assistant_prompt_id_mapping"] = {"happy": []}
    talker = _talker_config()
    talker["speaker_id"] = {"f6009": "bad"}
    talker["speaker_system_prompt_id"] = {"f6009": [80, "bad"]}
    _write_json(
        model_root / "config.json",
        _root_config(
            thinker_config=thinker,
            talker_config=talker,
            max_thinker_to_talker_mm_tokens=0,
        ),
    )
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("talker_language_id['zh']" in issue.message for issue in report.issues)
    assert any(
        "talker_assistant_prompt_id_mapping['happy'] must be a non-empty"
        in issue.message
        for issue in report.issues
    )
    assert any("speaker_id['f6009']" in issue.message for issue in report.issues)
    assert any(
        "speaker_system_prompt_id['f6009'][1]" in issue.message
        for issue in report.issues
    )
    assert any(
        "max_thinker_to_talker_mm_tokens must be positive" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_runtime_prompt_vocab_overflow(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["talker_language_id"] = {"zh": 16}
    thinker["talker_assistant_prompt_id_mapping"] = {"happy": [128]}
    talker = _talker_config()
    talker["speaker_id"] = {"tina": 16}
    talker["speaker_system_prompt_id"] = {"tina": [128]}
    _write_json(
        model_root / "config.json",
        _root_config(thinker_config=thinker, talker_config=talker),
    )
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "talker_language_id['zh'] must be in [0, talker codec vocab_size)"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "talker_assistant_prompt_id_mapping['happy'][0] must be in "
        "[0, talker text_vocab_size)"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "speaker_id['tina'] must be in [0, talker codec vocab_size)"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "speaker_system_prompt_id['tina'][0] must be in "
        "[0, talker text_vocab_size)"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_fails_when_code2wav_layout_is_missing(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("could not find code2wav" in issue.message for issue in report.issues)


def test_qwen35_preflight_text_only_skips_speech_requirements(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(
        model_root / "config.json",
        {
            "model_type": "qwen3_omni_next",
            "thinker_config": {
                "text_config": {
                    "hidden_size": 4,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 2,
                }
            },
        },
    )
    _touch_hf_weights(model_root)

    report = run_qwen35_preflight(str(model_root), speech=False)

    assert report.ok
    assert "speech checks skipped" in format_preflight_report(report)


def test_qwen35_preflight_reports_root_inline_thinker_config_path(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(
        model_root / "config.json",
        {
            "model_type": "qwen3_omni_next",
            "thinker_config": {},
        },
    )
    _touch_hf_weights(model_root)

    report = run_qwen35_preflight(str(model_root), speech=False)

    assert not report.ok
    assert any(
        issue.path == str(model_root / "config.json")
        and "thinker_config must provide text_config" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_reports_split_thinker_config_path(tmp_path):
    model_root = tmp_path / "qwen35"
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(
        model_root / "thinker" / "config.json",
        {"model_type": "qwen3_omni_next_thinker", "text_config": {}},
    )
    _write_json(model_root / "talker" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        issue.path == str(model_root / "thinker" / "config.json")
        and "thinker speech metadata missing required fields" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_split_thinker_model_type(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker.pop("model_type")
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", thinker)
    _write_json(model_root / "talker" / "config.json", _talker_config())
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        issue.path == str(model_root / "thinker" / "config.json")
        and "split thinker config missing model_type" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_split_talker_model_type(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker.pop("model_type")
    root_without_subconfigs = _root_config(thinker_config=None, talker_config=None)
    _write_json(model_root / "config.json", root_without_subconfigs)
    _write_json(model_root / "thinker" / "config.json", _thinker_config())
    _write_json(model_root / "talker" / "config.json", talker)
    _touch_hf_weights(model_root / "thinker")
    _touch_hf_weights(model_root / "talker")
    _touch_processor_assets(model_root / "thinker")
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        issue.path == str(model_root / "talker" / "config.json")
        and "split talker config missing model_type" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_qwen35_mrope_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker.pop("vision_start_token_id")
    thinker.pop("vision_end_token_id")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("vision_start_token_id" in issue.message for issue in report.issues)
    assert any(
        "vision_end_token_id or video_end_token_id" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_accepts_video_end_token_alias(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["video_end_token_id"] = thinker.pop("vision_end_token_id")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert not any("vision_end_token_id" in issue.message for issue in report.issues)


def test_qwen35_preflight_requires_encoder_configs(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker.pop("audio_config")
    thinker.pop("vision_config")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "missing required object fields: audio_config, vision_config"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_thinker_text_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["text_config"].pop("num_attention_heads")
    thinker["text_config"].pop("num_key_value_heads")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "thinker text_config missing required fields: "
        "num_attention_heads, num_key_value_heads"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_invalid_text_attention_shape(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    talker = _talker_config()
    thinker["text_config"]["hidden_size"] = 7
    thinker["text_config"]["num_attention_heads"] = 2
    talker["text_config"]["num_attention_heads"] = 3
    talker["text_config"]["num_key_value_heads"] = 2
    _write_json(
        model_root / "config.json",
        _root_config(thinker_config=thinker, talker_config=talker),
    )
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "thinker text_config.hidden_size must be divisible by "
        "num_attention_heads"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "talker text_config.num_attention_heads must be divisible by "
        "num_key_value_heads"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_accepts_qwen3_next_runtime_text_fields(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["text_config"]["layer_types"] = [
        "linear_attention",
        "full_attention",
        "attention",
        *["linear_attention"] * 29,
    ]
    thinker["text_config"]["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": "500000",
        "partial_rotary_factor": "0.25",
    }
    talker = _talker_config()
    talker["text_config"]["layers_block_type"] = [
        "full_attention",
        "linear_attention",
    ]
    talker["text_config"]["full_attention_interval"] = 2
    talker["text_config"]["rope_scaling"] = {
        "type": "default",
        "rope_theta": 1000000.0,
    }
    talker["text_config"]["partial_rotary_factor"] = 0.5
    talker["code_predictor_config"]["layer_types"] = ["full_attention"]
    talker["code_predictor_config"]["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    _write_json(
        model_root / "config.json",
        _root_config(thinker_config=thinker, talker_config=talker),
    )
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert not any("layer type" in issue.message for issue in report.issues)
    assert not any("rope_theta" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_bad_qwen3_next_runtime_text_fields(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["text_config"]["layer_types"] = [
        "linear_attention",
        "bogus_attention",
    ]
    thinker["text_config"]["full_attention_interval"] = 0
    thinker["text_config"]["rope_scaling"] = {
        "rope_theta": -1,
        "partial_rotary_factor": 1.5,
    }
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)

    report = run_qwen35_preflight(str(model_root), speech=False)

    assert not report.ok
    assert any("unsupported layer type" in issue.message for issue in report.issues)
    assert any(
        "layer_types length must equal num_hidden_layers" in issue.message
        for issue in report.issues
    )
    assert any(
        "full_attention_interval must be positive" in issue.message
        for issue in report.issues
    )
    assert any(
        "rope_theta must be positive" in issue.message
        for issue in report.issues
    )
    assert any(
        "partial_rotary_factor must be <= 1" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_audio_encoder_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["audio_config"].pop("d_model")
    thinker["audio_config"].pop("activation_function")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "audio_config missing required fields: d_model" in issue.message
        for issue in report.issues
    )
    assert any("activation_function" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_invalid_audio_encoder_shape(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["audio_config"]["d_model"] = 7
    thinker["audio_config"]["encoder_attention_heads"] = 2
    thinker["audio_config"]["activation_function"] = "mish"
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "audio_config.d_model must be even" in issue.message
        for issue in report.issues
    )
    assert any(
        "must be divisible by encoder_attention_heads" in issue.message
        for issue in report.issues
    )
    assert any(
        "activation_function must be one of" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_invalid_spatial_merge_size(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["vision_config"]["spatial_merge_size"] = 0
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("spatial_merge_size must be positive" in issue.message for issue in report.issues)


def test_qwen35_preflight_requires_vision_encoder_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["vision_config"].pop("patch_size")
    thinker["vision_config"].pop("out_hidden_size")
    thinker["vision_config"].pop("hidden_act")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "vision_config missing required fields: patch_size, out_hidden_size"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "vision_config missing required fields: hidden_act" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_invalid_vision_encoder_shape(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["vision_config"]["hidden_size"] = 7
    thinker["vision_config"]["num_heads"] = 2
    thinker["vision_config"]["deepstack_visual_indexes"] = [1, 4, -1, "bad"]
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "vision_config.hidden_size must be divisible by num_heads" in issue.message
        for issue in report.issues
    )
    assert any("out of range for depth 4" in issue.message for issue in report.issues)
    assert any("non-negative indexes" in issue.message for issue in report.issues)
    assert any("integer layer indexes" in issue.message for issue in report.issues)


def test_qwen35_preflight_warns_missing_deepstack_indexes(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["vision_config"].pop("deepstack_visual_indexes")
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any("deepstack_visual_indexes is missing" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_bad_code_predictor_runtime_fields(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["code_predictor_config"]["layer_types"] = [
        "bogus_attention",
        "linear_attention",
    ]
    talker["code_predictor_config"]["rope_parameters"] = {
        "rope_theta": -1,
        "partial_rotary_factor": 1.5,
    }
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "talker code_predictor_config.layer_types has unsupported layer type"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "talker code_predictor_config.layer_types length must equal "
        "num_hidden_layers"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "talker code_predictor_config.rope_theta must be positive"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "talker code_predictor_config.partial_rotary_factor must be <= 1"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_talker_vocab_size(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["text_config"] = {}
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "talker text_config missing required fields: vocab_size" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_talker_text_vocab_size(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["text_config"].pop("text_vocab_size")
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "talker text_config missing required fields: text_vocab_size"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_requires_talker_num_code_groups(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker.pop("code_predictor_config")
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("missing num_code_groups" in issue.message for issue in report.issues)
    assert any("must provide code_predictor_config" in issue.message for issue in report.issues)


def test_qwen35_preflight_requires_code_predictor_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["code_predictor_config"].pop("vocab_size")
    talker["code_predictor_config"].pop("intermediate_size")
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "code_predictor_config missing required fields: "
        "vocab_size, intermediate_size"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_invalid_code_predictor_shape(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["code_predictor_config"]["hidden_size"] = 7
    talker["code_predictor_config"]["num_attention_heads"] = 2
    talker["code_predictor_config"]["num_key_value_heads"] = 3
    talker["code_predictor_config"]["head_dim"] = 4
    talker["code_predictor_config"]["hidden_act"] = "gelu_pytorch_tanh"
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "code_predictor_config.hidden_size must be divisible by "
        "num_attention_heads"
        in issue.message
        for issue in report.issues
    )
    assert any(
        "code_predictor_config.num_attention_heads must be divisible by "
        "num_key_value_heads"
        in issue.message
        for issue in report.issues
    )
    assert not any(
        "head_dim * num_attention_heads" in issue.message
        for issue in report.issues
    )
    assert any("code_predictor_config.hidden_act" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_talker_code_group_mismatch(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["num_code_groups"] = 8
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "num_code_groups does not match code_predictor_config.num_code_groups"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_code_predictor_hidden_mismatch(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["code_predictor_config"]["talker_hidden_size"] = 8
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "talker_hidden_size/default hidden_size must match "
        "talker text_config.hidden_size"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_code_predictor_vocab_overflow(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["code_predictor_config"]["vocab_size"] = 17
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "code_predictor_config.vocab_size must not exceed "
        "talker text_config.vocab_size"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_codec_token_out_of_vocab(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["codec_eos_token_id"] = 99
    talker["codec_think_id"] = -1
    _write_json(model_root / "config.json", _root_config(talker_config=talker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "codec_eos_token_id must be in [0, vocab_size)" in issue.message
        for issue in report.issues
    )
    assert any(
        "codec_think_id must be a non-negative integer" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_root_text_token_out_of_vocab(tmp_path):
    model_root = tmp_path / "qwen35"
    talker = _talker_config()
    talker["text_config"]["text_vocab_size"] = 24
    _write_json(
        model_root / "config.json",
        _root_config(talker_config=talker, assistant_token_id=27),
    )
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "root speech metadata.assistant_token_id must be in "
        "[0, talker text_vocab_size)"
        in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_code2wav_codebook_mismatch(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav", codebook_nums=8)

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "code2wav dac.codebook_nums does not match" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_allows_code2wav_codebook_sentinel(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav", codebook_nums=-1)

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any(
        issue.severity == "warning"
        and "code2wav dac.codebook_nums is -1" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_warns_when_processor_assets_are_missing(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any("missing processor config" in issue.message for issue in report.issues)
    assert any("missing tokenizer_config.json" in issue.message for issue in report.issues)
    assert any("missing tokenizer vocabulary" in issue.message for issue in report.issues)
    assert any("missing chat template" in issue.message for issue in report.issues)


def test_qwen35_preflight_warns_when_chat_template_is_empty(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _write_json(model_root / "tokenizer_config.json", {"chat_template": "   "})
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any("missing chat template" in issue.message for issue in report.issues)


def test_qwen35_preflight_warns_when_chat_template_file_is_empty(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _write_json(model_root / "tokenizer_config.json", {})
    (model_root / "chat_template.jinja").write_text("")
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any("missing chat template" in issue.message for issue in report.issues)


def test_qwen35_preflight_reports_missing_required_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(
        model_root / "config.json",
        _root_config(tts_bos_token_id=None),
    )
    _touch_hf_weights(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("tts_bos_token_id" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_invalid_accept_hidden_layer(tmp_path):
    model_root = tmp_path / "qwen35"
    bad_talker = _talker_config()
    bad_talker["accept_hidden_layer"] = "embed"
    _write_json(model_root / "config.json", _root_config(talker_config=bad_talker))
    _touch_hf_weights(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("accept_hidden_layer must be" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_out_of_range_accept_hidden_layer(tmp_path):
    model_root = tmp_path / "qwen35"
    bad_talker = _talker_config()
    bad_talker["accept_hidden_layer"] = 40
    _write_json(model_root / "config.json", _root_config(talker_config=bad_talker))
    _touch_hf_weights(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any("out of thinker range" in issue.message for issue in report.issues)


def test_qwen35_preflight_fails_when_sglang_ar_weights_are_missing(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        "missing thinker/SGLang AR weights" in issue.message
        for issue in report.issues
    )
    assert any(
        "missing talker/SGLang AR weights" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_warns_for_unknown_root_identity(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(
        model_root / "config.json",
        _root_config(model_type="not_qwen", architectures=[]),
    )
    _touch_hf_weights(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any(
        issue.severity == "warning"
        and "not obviously Qwen3.5-Omni" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_warns_when_mtp_layers_are_declared(tmp_path):
    model_root = tmp_path / "qwen35"
    thinker = _thinker_config()
    thinker["text_config"]["mtp_num_hidden_layers"] = 1
    _write_json(model_root / "config.json", _root_config(thinker_config=thinker))
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any(
        issue.severity == "warning"
        and "runs the base thinker AR model" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_warns_when_mtp_weights_are_indexed(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _write_json(
        model_root / "model.safetensors.index.json",
        {"weight_map": {"thinker.mtp.fc.weight": "model-00001-of-00001.safetensors"}},
    )
    (model_root / "model-00001-of-00001.safetensors").write_bytes(b"placeholder")
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert report.ok
    assert any(
        issue.severity == "warning"
        and "detected Qwen3.5 MTP weights" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_missing_hf_index_shard(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _write_json(
        model_root / "model.safetensors.index.json",
        {
            "weight_map": {
                "thinker.model.embed_tokens.weight": (
                    "model-00001-of-00002.safetensors"
                ),
                "talker.model.embed_tokens.weight": (
                    "model-00002-of-00002.safetensors"
                ),
            }
        },
    )
    (model_root / "model-00001-of-00002.safetensors").write_bytes(b"placeholder")
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        issue.severity == "error"
        and "index references missing shard files" in issue.message
        and "model-00002-of-00002.safetensors" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_unsafe_hf_index_shard_path(tmp_path):
    model_root = tmp_path / "qwen35"
    _write_json(model_root / "config.json", _root_config())
    _write_json(
        model_root / "model.safetensors.index.json",
        {
            "weight_map": {
                "thinker.model.embed_tokens.weight": "../model-00001.safetensors",
            }
        },
    )
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")

    report = run_qwen35_preflight(str(model_root))

    assert not report.ok
    assert any(
        issue.severity == "error"
        and "shard filenames must be relative" in issue.message
        and "../model-00001.safetensors" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_checks_xvector_info_path(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    _touch_xvector_info(voice_dir)

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
    )

    assert report.ok
    assert any("xvector_info assets found" in issue.message for issue in report.issues)


def test_qwen35_preflight_accepts_xvector_info_alias_metadata(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    _touch_xvector_info(
        voice_dir,
        info={
            "system_instruct": "speak",
            "language": "zh",
        },
    )

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
    )

    assert report.ok
    assert not any(
        "system instruct is missing" in issue.message for issue in report.issues
    )
    assert not any(
        "voice clone language is missing" in issue.message for issue in report.issues
    )


def test_qwen35_preflight_checks_multiple_xvector_info_paths(tmp_path):
    model_root = tmp_path / "qwen35"
    good_voice_dir = tmp_path / "voice_ref_good"
    bad_voice_dir = tmp_path / "voice_ref_bad"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    _touch_xvector_info(good_voice_dir)
    bad_voice_dir.mkdir(parents=True)
    _write_json(
        bad_voice_dir / "info.json",
        {"talker_system_instruct": "speak", "language_type": "zh"},
    )

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(good_voice_dir), str(bad_voice_dir)],
    )

    assert not report.ok
    assert any(
        issue.path == str(good_voice_dir)
        and "xvector_info assets found" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.path == str(bad_voice_dir / "feat.pkl")
        and "missing feat.pkl" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_missing_xvector_files(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    voice_dir.mkdir(parents=True)
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    _write_json(voice_dir / "info.json", {"talker_system_instruct": "speak"})

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
    )

    assert not report.ok
    assert any("missing feat.pkl" in issue.message for issue in report.issues)


def test_qwen35_preflight_rejects_invalid_xvector_info_json(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    voice_dir.mkdir(parents=True)
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    (voice_dir / "feat.pkl").write_bytes(b"placeholder")
    (voice_dir / "info.json").write_text("{not-json")

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
    )

    assert not report.ok
    assert any("invalid info.json" in issue.message for issue in report.issues)


def test_qwen35_preflight_warns_for_empty_xvector_feat(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    _touch_xvector_info(voice_dir, feat_bytes=b"")

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
    )

    assert report.ok
    assert any(
        issue.severity == "warning" and "feat.pkl is empty" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_can_validate_xvector_feat_pickle(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    voice_dir.mkdir(parents=True)
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"ref_code": [[1, 2, 3]]}, handle)
    _write_json(
        voice_dir / "info.json",
        {"talker_system_instruct": "speak", "language_type": "zh"},
    )

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
        validate_xvector_pickle=True,
    )

    assert report.ok
    assert any(
        "validated voice clone prompt code key: ref_code" in issue.message
        for issue in report.issues
    )
    assert any(
        "feat.pkl content was validated" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_xvector_feat_without_prompt_code(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    voice_dir.mkdir(parents=True)
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"xvector": [0.0, 1.0]}, handle)
    _write_json(
        voice_dir / "info.json",
        {"talker_system_instruct": "speak", "language_type": "zh"},
    )

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
        validate_xvector_pickle=True,
    )

    assert not report.ok
    assert any(
        "feat.pkl missing prompt code" in issue.message
        and "xvector-only zero-shot" in issue.message
        for issue in report.issues
    )
    assert not any(
        "feat.pkl content was validated" in issue.message
        for issue in report.issues
    )


def test_qwen35_preflight_rejects_invalid_xvector_feat_pickle(tmp_path):
    model_root = tmp_path / "qwen35"
    voice_dir = tmp_path / "voice_ref"
    _write_json(model_root / "config.json", _root_config())
    _touch_hf_weights(model_root)
    _touch_processor_assets(model_root)
    _touch_code2wav(model_root / "code2wav")
    voice_dir.mkdir(parents=True)
    (voice_dir / "feat.pkl").write_bytes(b"not-a-pickle")
    _write_json(
        voice_dir / "info.json",
        {"talker_system_instruct": "speak", "language_type": "zh"},
    )

    report = run_qwen35_preflight(
        str(model_root),
        xvector_info_paths=[str(voice_dir)],
        validate_xvector_pickle=True,
    )

    assert not report.ok
    assert any("invalid feat.pkl" in issue.message for issue in report.issues)
    assert not any(
        "feat.pkl content was validated" in issue.message
        for issue in report.issues
    )
