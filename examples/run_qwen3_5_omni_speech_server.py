# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Qwen3.5-Omni speech output.

This wrapper intentionally reuses the Qwen3-Omni speech server CLI helpers and
only swaps in the Qwen3.5-Omni pipeline config. Keep argument behavior aligned
with examples/run_qwen3_omni_speech_server.py so benchmarking scripts can
compare both deployments with the same flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from typing import Any

from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
    QWEN3_5_OMNI_MODEL_NAME,
    QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
    normalize_qwen35_omni_model_name,
)
from sglang_omni.models.qwen3_5_omni.preflight import (
    format_vllm_profile_report,
    load_vllm_profile_payload,
    run_vllm_profile_preflight,
    suggested_vllm_profile_cli_args,
)

try:
    from examples.run_qwen3_omni_speech_server import (
        _apply_stage_factory_updates,
        _parse_thinker_tp_gpu_list,
        _set_stage_gpu,
        _set_stage_tp_size,
        _validate_fraction,
        parse_args as _parse_qwen3_args,
    )
except ModuleNotFoundError:
    from run_qwen3_omni_speech_server import (  # type: ignore[no-redef]
        _apply_stage_factory_updates,
        _parse_thinker_tp_gpu_list,
        _set_stage_gpu,
        _set_stage_tp_size,
        _validate_fraction,
        parse_args as _parse_qwen3_args,
    )

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QWEN35_DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-Omni"
QWEN35_DEFAULT_MODEL_NAME = QWEN3_5_OMNI_MODEL_NAME
QWEN35_DEFAULT_THINKER_MAX_SEQ_LEN = QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN
QWEN35_DEFAULT_VOICE_TYPE = "f245"
QWEN35_DEFAULT_MAX_TOKENS = 2048
QWEN35_DEFAULT_SEED = 0
QWEN35_DEFAULT_TEMPERATURE = 0.000001
QWEN35_DEFAULT_TOP_K = 1
QWEN35_DEFAULT_TOP_P = 0.8
QWEN3_DEFAULT_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN3_DEFAULT_MODEL_NAME = "qwen3-omni"
_ROOT_MODEL_PATH_FLAGS = ("--model",)
_THINKER_MODEL_PATH_FLAGS = ("--thinker-model", "--thinker_model")
_TALKER_MODEL_PATH_FLAGS = ("--talker-model", "--talker-model-path", "--talker-path")
_CODE2WAV_MODEL_PATH_FLAGS = (
    "--code2wav-model",
    "--code2wav-model-path",
    "--code2wav-path",
)
_CODE2WAV_MODEL_VLLM_ALIAS_FLAGS = ("--code2wav-model",)
_CODE2WAV_MODEL_FOLDER_FLAGS = (
    "--code2wav-model-folder",
    "--code2wav_model_folder",
)
_CODE2WAV_COMPILE_ENABLE_FLAGS = (
    "--code2wav-enable-torch-compile",
    "--code2wav-torch-compile",
    "--enable-torch-compile",
)
_CODE2WAV_COMPILE_DISABLE_FLAGS = (
    "--no-code2wav-enable-torch-compile",
    "--no-code2wav-torch-compile",
    "--no-enable-torch-compile",
)
_CODE2WAV_COMPILE_FIRST_CHUNK_ENABLE_FLAGS = (
    "--code2wav-enable-torch-compile-first-chunk",
    "--code2wav-torch-compile-first-chunk",
    "--enable-torch-compile-first-chunk",
)
_CODE2WAV_COMPILE_FIRST_CHUNK_DISABLE_FLAGS = (
    "--no-code2wav-enable-torch-compile-first-chunk",
    "--no-code2wav-torch-compile-first-chunk",
    "--no-enable-torch-compile-first-chunk",
)
_CODE2WAV_DYNAMIC_CHUNK_ENABLE_FLAGS = (
    "--code2wav-enable-dynamic-chunk",
    "--code2wav-dynamic-chunk",
)
_CODE2WAV_DYNAMIC_CHUNK_DISABLE_FLAGS = (
    "--no-code2wav-enable-dynamic-chunk",
    "--no-code2wav-dynamic-chunk",
)
_CODE2WAV_CODEC_EOS_TOKEN_ID_FLAGS = ("--code2wav-codec-eos-token-id",)
_CODE2WAV_SAMPLE_RATE_FLAGS = (
    "--code2wav-sample-rate",
    "--sample-rate",
    "--sample_rate",
)
_CODE2WAV_STREAM_CHUNK_SIZE_FLAGS = (
    "--code2wav-stream-chunk-size",
    "--code2wav_stream_chunk_size",
    "--send-chunk-size",
    "--send_chunk_size",
)
_CODE2WAV_LEFT_CONTEXT_SIZE_FLAGS = ("--code2wav-left-context-size",)
_CODE2WAV_DYNAMIC_CHUNK_SIZES_FLAGS = ("--code2wav-dynamic-chunk-sizes",)
_CODE2WAV_DYNAMIC_CHUNK_STEPS_FLAGS = ("--code2wav-dynamic-chunk-steps",)
_CODE2WAV_ODEINT_METHOD_FLAGS = ("--code2wav-odeint-method", "--odeint-method")
_CODE2WAV_ODEINT_RELAXED_ENABLE_FLAGS = (
    "--code2wav-odeint-method-relaxed",
    "--odeint-method-relaxed",
)
_CODE2WAV_ODEINT_RELAXED_DISABLE_FLAGS = (
    "--no-code2wav-odeint-method-relaxed",
    "--no-odeint-method-relaxed",
)
_CODE2WAV_BATCHED_CHUNK_FLAGS = ("--code2wav-batched-chunk", "--batched-chunk")
_CODE2WAV_ODEINT_METHODS = frozenset({"euler", "rk4"})
_CODE2WAV_FREQUENCY_FLAGS = ("--code2wav-frequency", "--code2wav_frequency")
_CODE2WAV_FREQUENCIES = frozenset({"50hz", "25hz"})
_CODE2WAV_FREQUENCY_ALIASES = {
    "1": "50hz",
    "50": "50hz",
    "50hz": "50hz",
    "2": "25hz",
    "25": "25hz",
    "25hz": "25hz",
}
_CODE2WAV_DIT_QUANT_FLAGS = (
    "--code2wav-dit-quantization",
    "--code2wav-dit-quant",
    "--code2wav_dit_quant",
)
_CODE2WAV_DIT_QUANTS = frozenset({"fp8"})
_PREFIX_CACHING_ENABLE_FLAGS = ("--enable-prefix-caching",)
_PREFIX_CACHING_DISABLE_FLAGS = (
    "--disable-prefix-caching",
    "--no-enable-prefix-caching",
)
_CHUNKED_PREFILL_ENABLE_FLAGS = ("--enable-chunked-prefill",)
_CHUNKED_PREFILL_DISABLE_FLAGS = (
    "--disable-chunked-prefill",
    "--no-enable-chunked-prefill",
)
_ENFORCE_EAGER_FLAGS = ("--enforce-eager",)
_THINKER_ENFORCE_EAGER_FLAGS = ("--thinker-enforce-eager",)
_TALKER_ENFORCE_EAGER_FLAGS = ("--talker-enforce-eager",)
_THINKER_QUANTIZATION_FLAGS = (
    "--quantization",
    "--thinker-quantization",
    "--thinker_quantization",
)
_TALKER_QUANTIZATION_FLAGS = ("--talker-quantization", "--talker_quantization")
_THINKER_QUANTIZATIONS = frozenset({"none", "fp8", "nvfp4"})
_TALKER_QUANTIZATIONS = frozenset({"none", "fp8", "nvfp4"})
_DTYPE_FLAGS = ("--dtype",)
_THINKER_DTYPE_FLAGS = ("--thinker-dtype", "--thinker_dtype")
_TALKER_DTYPE_FLAGS = ("--talker-dtype", "--talker_dtype")
_MAMBA_SSM_DTYPE_FLAGS = (
    "--mamba-ssm-dtype",
    "--mamba_ssm_dtype",
    "--mamba-cache-dtype",
    "--mamba_cache_dtype",
)
_MAMBA_CACHE_MODE_FLAGS = ("--mamba-cache-mode", "--mamba_cache_mode")
_MAMBA_CACHE_MODES = frozenset({"none", "light", "all"})
_KV_TRANSFER_CONFIG_FLAGS = (
    "--kv-transfer-config",
    "--kv_transfer_config",
    "--thinker-kv-transfer-config",
    "--thinker_kv_transfer_config",
)
_ENABLE_DISAGGREGATED_PREFILLING_FLAGS = (
    "--enable-disaggregated-prefilling",
    "--enable_disaggregated_prefilling",
)
_TENSOR_PARALLEL_SIZE_FLAGS = (
    "--tensor-parallel-size",
    "--tensor_parallel_size",
    "--thinker-tensor-parallel-size",
    "--thinker_tensor_parallel_size",
)
_DISTRIBUTED_EXECUTOR_BACKEND_FLAGS = (
    "--distributed-executor-backend",
    "--distributed_executor_backend",
)
_KV_CACHE_DTYPE_FLAGS = ("--kv-cache-dtype", "--kv_cache_dtype")
_ENABLE_EXPERT_PARALLEL_FLAGS = (
    "--enable-expert-parallel",
    "--enable_expert_parallel",
)
_MAX_MM_LEN_FLAGS = ("--max-mm-len", "--max_mm_len")
_MM_PROCESSOR_CACHE_GB_FLAGS = (
    "--mm-processor-cache-gb",
    "--mm_processor_cache_gb",
)
_SPECULATIVE_CONFIG_FLAGS = ("--speculative-config", "--speculative_config")
_VLLM_OMNI_BOOL_FLAG_GROUPS = (
    ("use_omni_engine", ("--use-omni-engine", "--use_omni_engine")),
    ("use_omni_rpc_engine", ("--use-omni-rpc-engine", "--use_omni_rpc_engine")),
    ("is_thinker", ("--is-thinker", "--is_thinker")),
    ("thinker_only", ("--thinker-only", "--thinker_only")),
    ("use_zero_shot", ("--use-zero-shot", "--use_zero_shot")),
    ("skip_mm_profiling", ("--skip-mm-profiling", "--skip_mm_profiling")),
    ("video_needs_metadata", ("--video-needs-metadata", "--video_needs_metadata")),
    (
        "override_video_max_pixels",
        ("--override-video-max-pixels", "--override_video_max_pixels"),
    ),
)
_MAX_MODEL_LEN_FLAGS = ("--max-model-len", "--max_model_len")
_MAX_SEQ_LEN_TO_CAPTURE_FLAGS = (
    "--max-seq-len-to-capture",
    "--max_seq_len_to_capture",
)
_HOST_FLAGS = ("--host",)
_PORT_FLAGS = ("--serve-port", "--serve_port")
_VISIBLE_DEVICES_FLAG_GROUPS = (
    (
        "thinker_visible_devices",
        (
            "--thinker-visible-devices",
            "--thinker_visible_devices",
            "--thinker-devices",
            "--thinker_devices",
        ),
    ),
    (
        "talker_visible_devices",
        (
            "--talker-visible-devices",
            "--talker_visible_devices",
            "--talker-devices",
            "--talker_devices",
        ),
    ),
    (
        "code2wav_visible_devices",
        (
            "--code2wav-visible-devices",
            "--code2wav_visible_devices",
            "--code2wav-devices",
            "--code2wav_devices",
        ),
    ),
)
_COMPILATION_CONFIG_FLAGS = ("--compilation-config", "--compilation_config")
_MAX_NUM_BATCHED_TOKENS_FLAGS = (
    "--max-num-batched-tokens",
    "--max_num_batched_tokens",
)
_PAGE_SIZE_FLAGS = (
    "--page-size",
    "--page_size",
    "--block-size",
    "--block_size",
)
_DISABLE_MTP_FLAGS = ("--disable-mtp", "--disable_mtp")
_PREFLIGHT_FLAGS = ("--preflight",)
_LIMIT_MM_PER_PROMPT_FLAGS = ("--limit-mm-per-prompt", "--limit_mm_per_prompt")
_LIMIT_MM_PER_PROMPT_FLAG_GROUPS = (
    (
        "image",
        ("--limit-mm-per-prompt-image", "--limit_mm_per_prompt_image"),
    ),
    (
        "video",
        ("--limit-mm-per-prompt-video", "--limit_mm_per_prompt_video"),
    ),
    (
        "audio",
        ("--limit-mm-per-prompt-audio", "--limit_mm_per_prompt_audio"),
    ),
)
_LIMIT_MM_MODALITIES = frozenset({"audio", "image", "video"})
_XVECTOR_INFO_PATH_FLAGS = (
    "--xvector-info-path",
    "--voice-clone-info-path",
    "--voice-clone-path",
)
_VALIDATE_XVECTOR_PICKLE_FLAGS = ("--validate-xvector-pickle",)
_VOICE_TYPE_FLAGS = ("--voice-type", "--voice_type")
_ENABLE_TN_FLAGS = (
    "--enable-tn",
    "--enable_tn",
    "--enable-text-normalization",
    "--enable_text_normalization",
)
_DISABLE_TN_FLAGS = (
    "--disable-tn",
    "--disable_tn",
    "--no-enable-tn",
    "--no-enable_tn",
    "--disable-text-normalization",
    "--disable_text_normalization",
)
_DEFAULT_MAX_TOKENS_FLAGS = ("--max-tokens", "--max_tokens")
_DEFAULT_SEED_FLAGS = ("--seed",)
_MAX_RUNNING_REQUEST_FLAG_GROUPS = (
    (
        "max_running_requests",
        (
            "--max-running-requests",
            "--max_running_requests",
            "--max-num-seqs",
            "--max_num_seqs",
        ),
    ),
    (
        "thinker_max_running_requests",
        (
            "--thinker-max-running-requests",
            "--thinker_max_running_requests",
        ),
    ),
    (
        "talker_max_running_requests",
        (
            "--talker-max-running-requests",
            "--talker_max_running_requests",
        ),
    ),
)
_VLLM_PROFILE_FLAGS = ("--vllm-profile", "--vllm_profile")
_VIDEO_FPS_FLAGS = ("--video-fps", "--video_fps")
_VIDEO_FLOAT_FLAG_GROUPS = (
    (
        "video_seconds_per_chunk",
        (
            "--video-seconds-per-chunk",
            "--video_seconds_per_chunk",
            "--seconds-per-chunk",
            "--seconds_per_chunk",
        ),
    ),
    (
        "video_position_id_per_seconds",
        (
            "--video-position-id-per-seconds",
            "--video_position_id_per_seconds",
            "--position-id-per-seconds",
            "--position_id_per_seconds",
        ),
    ),
)
_IMAGE_INT_FLAG_GROUPS = (
    ("image_min_pixels", ("--image-min-pixels", "--image_min_pixels")),
    ("image_max_pixels", ("--image-max-pixels", "--image_max_pixels")),
)
_VIDEO_INT_FLAG_GROUPS = (
    ("video_max_frames", ("--video-max-frames", "--video_max_frames")),
    ("video_min_frames", ("--video-min-frames", "--video_min_frames")),
    ("video_min_pixels", ("--video-min-pixels", "--video_min_pixels")),
    ("video_max_pixels", ("--video-max-pixels", "--video_max_pixels")),
    ("video_total_pixels", ("--video-total-pixels", "--video_total_pixels")),
)
_AUDIO_INT_FLAG_GROUPS = (
    (
        "audio_target_sr",
        (
            "--audio-target-sr",
            "--audio_target_sr",
            "--audio-sampling-rate",
            "--audio_sampling_rate",
            "--sampling-rate",
            "--sampling_rate",
        ),
    ),
    (
        "audio_timestamp_interval",
        (
            "--audio-timestamp-interval",
            "--audio_timestamp_interval",
            "--timestamp-interval",
            "--timestamp_interval",
        ),
    ),
    (
        "audio_downsample_times",
        (
            "--audio-downsample-times",
            "--audio_downsample_times",
            "--downsample-times",
            "--downsample_times",
        ),
    ),
    (
        "audio_downsample_chunk_size",
        (
            "--audio-downsample-chunk-size",
            "--audio_downsample_chunk_size",
            "--downsample-chunk-size",
            "--downsample_chunk_size",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    original_argv = sys.argv
    argv = _argv_with_vllm_profile_defaults(original_argv)
    argv, qwen35_args = _extract_qwen35_extra_args(argv)
    old_argv = sys.argv
    try:
        sys.argv = argv
        args = _parse_qwen3_args()
    finally:
        sys.argv = old_argv
    has_sglang_model_path = _has_explicit_flag(original_argv, "--model-path")
    if qwen35_args["model_path"] is not None and not has_sglang_model_path:
        args.model_path = qwen35_args["model_path"]
    elif (
        qwen35_args["thinker_model_path"] is not None
        and not has_sglang_model_path
    ):
        # 中文说明：vLLM 同时传 --model 和 --thinker-model 时，SGLang
        # 保留 root model_path，由 stage resolver 自动切到 root/thinker；
        # 只有单独传 thinker-model 时才把它当作可启动的模型路径。
        args.model_path = qwen35_args["thinker_model_path"]
    args.talker_model_path = qwen35_args["talker_model_path"]
    args.code2wav_model_path = qwen35_args["code2wav_model_path"]
    args.code2wav_model_folder = qwen35_args["code2wav_model_folder"]
    args.code2wav_enable_torch_compile = qwen35_args[
        "code2wav_enable_torch_compile"
    ]
    args.code2wav_enable_torch_compile_first_chunk = qwen35_args[
        "code2wav_enable_torch_compile_first_chunk"
    ]
    args.code2wav_codec_eos_token_id = qwen35_args[
        "code2wav_codec_eos_token_id"
    ]
    args.code2wav_sample_rate = qwen35_args["code2wav_sample_rate"]
    args.code2wav_stream_chunk_size = qwen35_args["code2wav_stream_chunk_size"]
    args.code2wav_left_context_size = qwen35_args["code2wav_left_context_size"]
    args.code2wav_enable_dynamic_chunk = qwen35_args[
        "code2wav_enable_dynamic_chunk"
    ]
    args.code2wav_dynamic_chunk_sizes = qwen35_args[
        "code2wav_dynamic_chunk_sizes"
    ]
    args.code2wav_dynamic_chunk_steps = qwen35_args[
        "code2wav_dynamic_chunk_steps"
    ]
    args.code2wav_odeint_method = qwen35_args["code2wav_odeint_method"]
    args.code2wav_odeint_method_relaxed = qwen35_args[
        "code2wav_odeint_method_relaxed"
    ]
    args.code2wav_batched_chunk = qwen35_args["code2wav_batched_chunk"]
    args.code2wav_frequency = qwen35_args["code2wav_frequency"]
    args.code2wav_dit_quant = qwen35_args["code2wav_dit_quant"]
    args.enable_prefix_caching = qwen35_args["enable_prefix_caching"]
    args.enable_chunked_prefill = qwen35_args["enable_chunked_prefill"]
    args.enforce_eager = qwen35_args["enforce_eager"]
    args.thinker_enforce_eager = qwen35_args["thinker_enforce_eager"]
    args.talker_enforce_eager = qwen35_args["talker_enforce_eager"]
    args.thinker_quantization = qwen35_args["thinker_quantization"]
    args.talker_quantization = qwen35_args["talker_quantization"]
    args.dtype = qwen35_args["dtype"]
    args.thinker_dtype = qwen35_args["thinker_dtype"]
    args.talker_dtype = qwen35_args["talker_dtype"]
    args.mamba_ssm_dtype = qwen35_args["mamba_ssm_dtype"]
    args.mamba_cache_mode = qwen35_args["mamba_cache_mode"]
    args.kv_transfer_config = qwen35_args["kv_transfer_config"]
    if qwen35_args["port"] is not None:
        args.port = qwen35_args["port"]
    if qwen35_args["host"] is not None:
        args.host = qwen35_args["host"]
    args.enable_disaggregated_prefilling = qwen35_args[
        "enable_disaggregated_prefilling"
    ]
    args.tensor_parallel_size = qwen35_args["tensor_parallel_size"]
    args.distributed_executor_backend = qwen35_args[
        "distributed_executor_backend"
    ]
    args.kv_cache_dtype = qwen35_args["kv_cache_dtype"]
    args.enable_expert_parallel = qwen35_args["enable_expert_parallel"]
    args.max_mm_len = qwen35_args["max_mm_len"]
    args.mm_processor_cache_gb = qwen35_args["mm_processor_cache_gb"]
    args.speculative_config = qwen35_args["speculative_config"]
    for key, _ in _VLLM_OMNI_BOOL_FLAG_GROUPS:
        setattr(args, key, qwen35_args[key])
    args.max_seq_len_to_capture = qwen35_args["max_seq_len_to_capture"]
    _apply_visible_devices_args(args, qwen35_args)
    _apply_tensor_parallel_size_arg(args, qwen35_args)
    args.compilation_config = qwen35_args["compilation_config"]
    args.max_num_batched_tokens = qwen35_args["max_num_batched_tokens"]
    args.page_size = qwen35_args["page_size"]
    args.disable_mtp = qwen35_args["disable_mtp"]
    if qwen35_args["mem_fraction_static"] is not None:
        args.mem_fraction_static = qwen35_args["mem_fraction_static"]
    if qwen35_args["thinker_mem_fraction_static"] is not None:
        args.thinker_mem_fraction_static = qwen35_args[
            "thinker_mem_fraction_static"
        ]
    if qwen35_args["talker_mem_fraction_static"] is not None:
        args.talker_mem_fraction_static = qwen35_args["talker_mem_fraction_static"]
    args.image_min_pixels = qwen35_args["image_min_pixels"]
    args.image_max_pixels = qwen35_args["image_max_pixels"]
    args.video_fps = qwen35_args["video_fps"]
    args.video_max_frames = qwen35_args["video_max_frames"]
    args.video_min_frames = qwen35_args["video_min_frames"]
    args.video_min_pixels = qwen35_args["video_min_pixels"]
    args.video_max_pixels = qwen35_args["video_max_pixels"]
    args.video_total_pixels = qwen35_args["video_total_pixels"]
    args.video_seconds_per_chunk = qwen35_args["video_seconds_per_chunk"]
    args.video_position_id_per_seconds = qwen35_args[
        "video_position_id_per_seconds"
    ]
    args.audio_target_sr = qwen35_args["audio_target_sr"]
    args.audio_timestamp_interval = qwen35_args["audio_timestamp_interval"]
    args.audio_downsample_times = qwen35_args["audio_downsample_times"]
    args.audio_downsample_chunk_size = qwen35_args[
        "audio_downsample_chunk_size"
    ]
    args.limit_mm_per_prompt = qwen35_args["limit_mm_per_prompt"]
    args.max_running_requests = qwen35_args["max_running_requests"]
    args.thinker_max_running_requests = qwen35_args[
        "thinker_max_running_requests"
    ]
    args.talker_max_running_requests = qwen35_args[
        "talker_max_running_requests"
    ]
    args.voice_type = qwen35_args["voice_type"]
    args.enable_tn = qwen35_args["enable_tn"]
    args.max_tokens = qwen35_args["max_tokens"]
    args.seed = qwen35_args["seed"]
    args.preflight = qwen35_args["preflight"]
    args.xvector_info_paths = tuple(qwen35_args["xvector_info_paths"])
    args.validate_xvector_pickle = qwen35_args["validate_xvector_pickle"]
    if args.model_path == QWEN3_DEFAULT_MODEL_PATH:
        args.model_path = QWEN35_DEFAULT_MODEL_PATH
    if args.model_name == QWEN3_DEFAULT_MODEL_NAME:
        args.model_name = QWEN35_DEFAULT_MODEL_NAME
    else:
        args.model_name = normalize_qwen35_omni_model_name(args.model_name)
    args.code2wav_model_path = _resolve_code2wav_model_path_from_folder(
        model_path=args.model_path,
        code2wav_model_path=args.code2wav_model_path,
        code2wav_model_folder=args.code2wav_model_folder,
    )
    if qwen35_args["max_model_len"] is not None:
        args.thinker_max_seq_len = qwen35_args["max_model_len"]
    elif not _has_explicit_flag(argv, "--thinker-max-seq-len"):
        # 中文说明：Qwen3.5 vLLM perf_v2 在线 speech runner 默认 thinker
        # max_model_len=192000；256k eval/长上下文场景可显式覆盖。
        args.thinker_max_seq_len = QWEN35_DEFAULT_THINKER_MAX_SEQ_LEN
    return args


def _has_explicit_flag(argv: list[str], flag_name: str) -> bool:
    return any(arg == flag_name or arg.startswith(f"{flag_name}=") for arg in argv[1:])


def _has_any_explicit_flag(argv: list[str], flag_names: tuple[str, ...]) -> bool:
    return any(
        arg == flag_name or arg.startswith(f"{flag_name}=")
        for arg in argv[1:]
        for flag_name in flag_names
    )


def _argv_with_vllm_profile_defaults(argv: list[str]) -> list[str]:
    cleaned, profile_path = _extract_vllm_profile_path(argv)
    if profile_path is None:
        return list(argv)

    payload = load_vllm_profile_payload(profile_path)
    if _has_any_explicit_flag(cleaned, _DISABLE_MTP_FLAGS):
        payload = _profile_payload_with_disable_mtp(payload)
    report = run_vllm_profile_preflight(payload, source=profile_path)
    if not report.ok:
        raise ValueError(
            "--vllm-profile contains settings this launcher cannot map yet:\n"
            + format_vllm_profile_report(report)
        )

    profile_args = [
        arg
        for arg in suggested_vllm_profile_cli_args(report)
        # 中文说明：preflight 的 --text-only 建议面向 generic serve；speech
        # launcher 只需要保留 --thinker-only=true，让后续冲突校验报清楚。
        if arg != "--text-only"
    ]
    # 中文说明：profile 作为默认值放在显式 CLI 参数前面；argparse 和
    # qwen35 手写解析器都会让后面的用户参数覆盖 profile。
    return [cleaned[0], *profile_args, *cleaned[1:]]


def _profile_payload_with_disable_mtp(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("engine_args"), dict):
        normalized = dict(payload)
        engine_args = dict(payload["engine_args"])
        engine_args["disable_mtp"] = True
        normalized["engine_args"] = engine_args
        return normalized
    normalized = dict(payload)
    # 中文说明：用户显式 --disable-mtp 时，允许复用 vLLM MTP profile
    # 里的非 MTP 默认值；preflight 仍会拦住 KV/PD/RPC 等真实未接入能力。
    normalized["disable_mtp"] = True
    return normalized


def _extract_vllm_profile_path(argv: list[str]) -> tuple[list[str], str | None]:
    cleaned = [argv[0]]
    profile_path: str | None = None
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        matched_flag = None
        for flag in _VLLM_PROFILE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_flag = flag
                break
        if matched_flag is None:
            cleaned.append(arg)
            idx += 1
            continue

        if profile_path is not None:
            raise ValueError("--vllm-profile may be passed at most once")
        if arg == matched_flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{matched_flag} requires a file path")
            profile_path = argv[idx + 1]
            idx += 2
        else:
            profile_path = arg.split("=", 1)[1]
            idx += 1
        if not profile_path:
            raise ValueError(f"{matched_flag} requires a non-empty file path")
    return cleaned, profile_path


def _extract_code2wav_model_path(argv: list[str]) -> tuple[list[str], str | None]:
    cleaned, qwen35_args = _extract_qwen35_extra_args(argv)
    return cleaned, qwen35_args["code2wav_model_path"]


def _resolve_code2wav_model_path_from_folder(
    *,
    model_path: str,
    code2wav_model_path: str | None,
    code2wav_model_folder: str | None,
) -> str | None:
    if code2wav_model_path is not None:
        return code2wav_model_path
    if code2wav_model_folder is None:
        return None
    folder = code2wav_model_folder.strip()
    if not folder:
        raise ValueError("--code2wav-model-folder requires a non-empty value")
    if os.path.isabs(folder):
        return folder
    # 中文说明：vLLM perf_v2 配置里写的是 root checkpoint 下的子目录名；
    # 这里解析成 SGLang code2wav factory 能直接加载的完整路径。
    return os.path.join(model_path, folder)


def _looks_like_relative_code2wav_folder(value: str) -> bool:
    value = value.strip()
    if not value or os.path.isabs(value):
        return False
    if value in {".", ".."}:
        return False
    # 中文说明：vLLM qwen_omni_v35.py 的 --code2wav-model 默认值是
    # "code2wav"；若当前目录不存在该相对路径，它会兜底到
    # $MODEL_DIR/code2wav。SGLang 这里把单个相对目录名视为 model root
    # 下的 folder，避免真实启动时把 cwd/code2wav 当成 checkpoint。
    return os.path.basename(value) == value


def _extract_qwen35_extra_args(argv: list[str]) -> tuple[list[str], dict[str, Any]]:
    cleaned = [argv[0]]
    qwen35_args: dict[str, Any] = {
        "model_path": None,
        "thinker_model_path": None,
        "talker_model_path": None,
        "code2wav_model_path": None,
        "code2wav_model_folder": None,
        "code2wav_enable_torch_compile": None,
        "code2wav_enable_torch_compile_first_chunk": None,
        "code2wav_codec_eos_token_id": None,
        "code2wav_sample_rate": None,
        "code2wav_stream_chunk_size": None,
        "code2wav_left_context_size": None,
        "code2wav_enable_dynamic_chunk": None,
        "code2wav_dynamic_chunk_sizes": None,
        "code2wav_dynamic_chunk_steps": None,
        "code2wav_odeint_method": None,
        "code2wav_odeint_method_relaxed": None,
        "code2wav_batched_chunk": None,
        "code2wav_frequency": None,
        "code2wav_dit_quant": None,
        "enable_prefix_caching": None,
        "enable_chunked_prefill": None,
        "enforce_eager": None,
        "thinker_enforce_eager": None,
        "talker_enforce_eager": None,
        "thinker_quantization": None,
        "talker_quantization": None,
        "dtype": None,
        "thinker_dtype": None,
        "talker_dtype": None,
        "mamba_ssm_dtype": None,
        "mamba_cache_mode": None,
        "kv_transfer_config": None,
        "host": None,
        "port": None,
        "enable_disaggregated_prefilling": None,
        "tensor_parallel_size": None,
        "distributed_executor_backend": None,
        "kv_cache_dtype": None,
        "enable_expert_parallel": None,
        "max_mm_len": None,
        "mm_processor_cache_gb": None,
        "speculative_config": None,
        "max_model_len": None,
        "max_seq_len_to_capture": None,
        "thinker_visible_devices": None,
        "talker_visible_devices": None,
        "code2wav_visible_devices": None,
        "compilation_config": None,
        "max_num_batched_tokens": None,
        "page_size": None,
        "disable_mtp": False,
        "mem_fraction_static": None,
        "thinker_mem_fraction_static": None,
        "talker_mem_fraction_static": None,
        "image_min_pixels": None,
        "image_max_pixels": None,
        "video_fps": None,
        "video_max_frames": None,
        "video_min_frames": None,
        "video_min_pixels": None,
        "video_max_pixels": None,
        "video_total_pixels": None,
        "video_seconds_per_chunk": None,
        "video_position_id_per_seconds": None,
        "audio_target_sr": None,
        "audio_timestamp_interval": None,
        "audio_downsample_times": None,
        "audio_downsample_chunk_size": None,
        "limit_mm_per_prompt": None,
        "max_running_requests": None,
        "thinker_max_running_requests": None,
        "talker_max_running_requests": None,
        "voice_type": QWEN35_DEFAULT_VOICE_TYPE,
        "enable_tn": None,
        "max_tokens": QWEN35_DEFAULT_MAX_TOKENS,
        "seed": QWEN35_DEFAULT_SEED,
        "preflight": False,
        "xvector_info_paths": [],
        "validate_xvector_pickle": False,
    }
    for key, _ in _VLLM_OMNI_BOOL_FLAG_GROUPS:
        qwen35_args[key] = None
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        matched_model_key = None
        matched_model_flag = None
        for key, flags in (
            ("model_path", _ROOT_MODEL_PATH_FLAGS),
            ("thinker_model_path", _THINKER_MODEL_PATH_FLAGS),
        ):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_model_key = key
                    matched_model_flag = flag
                    break
            if matched_model_flag is not None:
                break
        if matched_model_flag is not None and matched_model_key is not None:
            if arg == matched_model_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_model_flag} requires a path value")
                qwen35_args[matched_model_key] = argv[idx + 1]
                idx += 2
                continue
            qwen35_args[matched_model_key] = arg.split("=", 1)[1]
            idx += 1
            continue
        if arg in _PREFLIGHT_FLAGS:
            qwen35_args["preflight"] = True
            idx += 1
            continue
        if arg in _VALIDATE_XVECTOR_PICKLE_FLAGS:
            qwen35_args["validate_xvector_pickle"] = True
            idx += 1
            continue
        if arg in _DISABLE_MTP_FLAGS:
            qwen35_args["disable_mtp"] = True
            idx += 1
            continue
        if arg in _ENABLE_TN_FLAGS:
            qwen35_args["enable_tn"] = True
            idx += 1
            continue
        if arg in _DISABLE_TN_FLAGS:
            qwen35_args["enable_tn"] = False
            idx += 1
            continue
        if arg in _PREFIX_CACHING_ENABLE_FLAGS:
            qwen35_args["enable_prefix_caching"] = True
            idx += 1
            continue
        if arg in _PREFIX_CACHING_DISABLE_FLAGS:
            qwen35_args["enable_prefix_caching"] = False
            idx += 1
            continue
        if arg in _CHUNKED_PREFILL_ENABLE_FLAGS:
            qwen35_args["enable_chunked_prefill"] = True
            idx += 1
            continue
        if arg in _CHUNKED_PREFILL_DISABLE_FLAGS:
            qwen35_args["enable_chunked_prefill"] = False
            idx += 1
            continue
        if arg in _ENFORCE_EAGER_FLAGS:
            qwen35_args["enforce_eager"] = True
            idx += 1
            continue
        if arg in _THINKER_ENFORCE_EAGER_FLAGS:
            qwen35_args["thinker_enforce_eager"] = True
            idx += 1
            continue
        if arg in _TALKER_ENFORCE_EAGER_FLAGS:
            qwen35_args["talker_enforce_eager"] = True
            idx += 1
            continue
        if arg in _CODE2WAV_COMPILE_ENABLE_FLAGS:
            qwen35_args["code2wav_enable_torch_compile"] = True
            idx += 1
            continue
        if arg in _CODE2WAV_COMPILE_DISABLE_FLAGS:
            qwen35_args["code2wav_enable_torch_compile"] = False
            idx += 1
            continue
        if arg in _CODE2WAV_COMPILE_FIRST_CHUNK_ENABLE_FLAGS:
            qwen35_args["code2wav_enable_torch_compile_first_chunk"] = True
            idx += 1
            continue
        if arg in _CODE2WAV_COMPILE_FIRST_CHUNK_DISABLE_FLAGS:
            qwen35_args["code2wav_enable_torch_compile_first_chunk"] = False
            idx += 1
            continue
        if arg in _CODE2WAV_DYNAMIC_CHUNK_ENABLE_FLAGS:
            qwen35_args["code2wav_enable_dynamic_chunk"] = True
            idx += 1
            continue
        if arg in _CODE2WAV_DYNAMIC_CHUNK_DISABLE_FLAGS:
            qwen35_args["code2wav_enable_dynamic_chunk"] = False
            idx += 1
            continue
        if arg in _CODE2WAV_ODEINT_RELAXED_ENABLE_FLAGS:
            qwen35_args["code2wav_odeint_method_relaxed"] = True
            idx += 1
            continue
        if arg in _CODE2WAV_ODEINT_RELAXED_DISABLE_FLAGS:
            qwen35_args["code2wav_odeint_method_relaxed"] = False
            idx += 1
            continue
        matched_int_flag = None
        for flag in _CODE2WAV_STREAM_CHUNK_SIZE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_int_flag = flag
                break
        if matched_int_flag is not None:
            if arg == matched_int_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_int_flag} requires an integer value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_stream_chunk_size"] = _parse_positive_int_flag(
                matched_int_flag,
                raw_value,
            )
            continue
        matched_codec_eos_flag = None
        for flag in _CODE2WAV_CODEC_EOS_TOKEN_ID_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_codec_eos_flag = flag
                break
        if matched_codec_eos_flag is not None:
            if arg == matched_codec_eos_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_codec_eos_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_codec_eos_token_id"] = (
                _parse_nonnegative_int_flag(matched_codec_eos_flag, raw_value)
            )
            continue
        matched_sample_rate_flag = None
        for flag in _CODE2WAV_SAMPLE_RATE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_sample_rate_flag = flag
                break
        if matched_sample_rate_flag is not None:
            if arg == matched_sample_rate_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_sample_rate_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_sample_rate"] = _parse_positive_int_flag(
                matched_sample_rate_flag,
                raw_value,
            )
            continue
        matched_default_max_tokens_flag = None
        for flag in _DEFAULT_MAX_TOKENS_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_default_max_tokens_flag = flag
                break
        if matched_default_max_tokens_flag is not None:
            if arg == matched_default_max_tokens_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_default_max_tokens_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["max_tokens"] = _parse_positive_int_flag(
                matched_default_max_tokens_flag,
                raw_value,
            )
            continue
        matched_seed_flag = None
        for flag in _DEFAULT_SEED_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_seed_flag = flag
                break
        if matched_seed_flag is not None:
            if arg == matched_seed_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_seed_flag} requires an integer value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["seed"] = _parse_nonnegative_int_flag(
                matched_seed_flag,
                raw_value,
            )
            continue
        matched_left_context_flag = None
        for flag in _CODE2WAV_LEFT_CONTEXT_SIZE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_left_context_flag = flag
                break
        if matched_left_context_flag is not None:
            if arg == matched_left_context_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_left_context_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_left_context_size"] = (
                _parse_nonnegative_int_flag(matched_left_context_flag, raw_value)
            )
            continue
        matched_batched_chunk_flag = None
        for flag in _CODE2WAV_BATCHED_CHUNK_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_batched_chunk_flag = flag
                break
        if matched_batched_chunk_flag is not None:
            if arg == matched_batched_chunk_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_batched_chunk_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_batched_chunk"] = _parse_positive_int_flag(
                matched_batched_chunk_flag,
                raw_value,
            )
            continue
        matched_odeint_method_flag = None
        for flag in _CODE2WAV_ODEINT_METHOD_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_odeint_method_flag = flag
                break
        if matched_odeint_method_flag is not None:
            if arg == matched_odeint_method_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_odeint_method_flag} requires a method value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_odeint_method"] = _parse_odeint_method_flag(
                matched_odeint_method_flag,
                raw_value,
            )
            continue
        matched_frequency_flag = None
        for flag in _CODE2WAV_FREQUENCY_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_frequency_flag = flag
                break
        if matched_frequency_flag is not None:
            if arg == matched_frequency_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_frequency_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_frequency"] = _parse_code2wav_frequency_flag(
                matched_frequency_flag,
                raw_value,
            )
            continue
        matched_dit_quant_flag = None
        for flag in _CODE2WAV_DIT_QUANT_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_dit_quant_flag = flag
                break
        if matched_dit_quant_flag is not None:
            if arg == matched_dit_quant_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_dit_quant_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["code2wav_dit_quant"] = _parse_code2wav_dit_quant_flag(
                matched_dit_quant_flag,
                raw_value,
            )
            continue
        matched_quantization_key = None
        matched_quantization_flag = None
        for key, flags in (
            ("thinker_quantization", _THINKER_QUANTIZATION_FLAGS),
            ("talker_quantization", _TALKER_QUANTIZATION_FLAGS),
        ):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_quantization_key = key
                    matched_quantization_flag = flag
                    break
            if matched_quantization_flag is not None:
                break
        if matched_quantization_key is not None and matched_quantization_flag:
            if arg == matched_quantization_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_quantization_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_quantization_key] = _parse_quantization_flag(
                matched_quantization_flag,
                raw_value,
                supported=(
                    _THINKER_QUANTIZATIONS
                    if matched_quantization_key == "thinker_quantization"
                    else _TALKER_QUANTIZATIONS
                ),
            )
            continue
        matched_dtype_key = None
        matched_dtype_flag = None
        for key, flags in (
            ("dtype", _DTYPE_FLAGS),
            ("thinker_dtype", _THINKER_DTYPE_FLAGS),
            ("talker_dtype", _TALKER_DTYPE_FLAGS),
            ("mamba_ssm_dtype", _MAMBA_SSM_DTYPE_FLAGS),
        ):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_dtype_key = key
                    matched_dtype_flag = flag
                    break
            if matched_dtype_flag is not None:
                break
        if matched_dtype_key is not None and matched_dtype_flag is not None:
            if arg == matched_dtype_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_dtype_flag} requires a dtype value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_dtype_key] = _parse_dtype_flag(
                matched_dtype_flag,
                raw_value,
            )
            continue
        matched_mamba_cache_mode_flag = None
        for flag in _MAMBA_CACHE_MODE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_mamba_cache_mode_flag = flag
                break
        if matched_mamba_cache_mode_flag is not None:
            if arg == matched_mamba_cache_mode_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_mamba_cache_mode_flag} requires a value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["mamba_cache_mode"] = _parse_mamba_cache_mode_flag(
                matched_mamba_cache_mode_flag,
                raw_value,
            )
            continue
        matched_kv_transfer_flag = None
        for flag in _KV_TRANSFER_CONFIG_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_kv_transfer_flag = flag
                break
        if matched_kv_transfer_flag is not None:
            if arg == matched_kv_transfer_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_kv_transfer_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["kv_transfer_config"] = _parse_json_object_flag(
                matched_kv_transfer_flag,
                raw_value,
            )
            continue
        matched_disaggregated_prefill_flag = None
        for flag in _ENABLE_DISAGGREGATED_PREFILLING_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_disaggregated_prefill_flag = flag
                break
        if matched_disaggregated_prefill_flag is not None:
            if arg == matched_disaggregated_prefill_flag:
                if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                    raw_value = argv[idx + 1]
                    idx += 2
                else:
                    raw_value = "true"
                    idx += 1
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["enable_disaggregated_prefilling"] = _parse_bool_flag(
                matched_disaggregated_prefill_flag,
                raw_value,
            )
            continue
        matched_vllm_bool_key = None
        matched_vllm_bool_flag = None
        for key, flags in _VLLM_OMNI_BOOL_FLAG_GROUPS:
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_vllm_bool_key = key
                    matched_vllm_bool_flag = flag
                    break
            if matched_vllm_bool_flag is not None:
                break
        if matched_vllm_bool_key is not None and matched_vllm_bool_flag:
            if arg == matched_vllm_bool_flag:
                if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                    raw_value = argv[idx + 1]
                    idx += 2
                else:
                    raw_value = "true"
                    idx += 1
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_vllm_bool_key] = _parse_bool_flag(
                matched_vllm_bool_flag,
                raw_value,
            )
            continue
        matched_tensor_parallel_flag = None
        for flag in _TENSOR_PARALLEL_SIZE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_tensor_parallel_flag = flag
                break
        if matched_tensor_parallel_flag is not None:
            if arg == matched_tensor_parallel_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_tensor_parallel_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["tensor_parallel_size"] = _parse_positive_int_flag(
                matched_tensor_parallel_flag,
                raw_value,
            )
            continue
        matched_string_key = None
        matched_string_flag = None
        for key, flags in (
            ("distributed_executor_backend", _DISTRIBUTED_EXECUTOR_BACKEND_FLAGS),
            ("kv_cache_dtype", _KV_CACHE_DTYPE_FLAGS),
            ("host", _HOST_FLAGS),
            ("voice_type", _VOICE_TYPE_FLAGS),
        ):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_string_key = key
                    matched_string_flag = flag
                    break
            if matched_string_flag is not None:
                break
        if matched_string_key is not None and matched_string_flag:
            if arg == matched_string_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_string_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_string_key] = _parse_nonempty_string_flag(
                matched_string_flag,
                raw_value,
            )
            continue
        matched_enable_ep_flag = None
        for flag in _ENABLE_EXPERT_PARALLEL_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_enable_ep_flag = flag
                break
        if matched_enable_ep_flag is not None:
            if arg == matched_enable_ep_flag:
                if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                    raw_value = argv[idx + 1]
                    idx += 2
                else:
                    raw_value = "true"
                    idx += 1
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["enable_expert_parallel"] = _parse_bool_flag(
                matched_enable_ep_flag,
                raw_value,
            )
            continue
        matched_max_mm_len_flag = None
        for flag in _MAX_MM_LEN_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_max_mm_len_flag = flag
                break
        if matched_max_mm_len_flag is not None:
            if arg == matched_max_mm_len_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_max_mm_len_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["max_mm_len"] = _parse_positive_int_flag(
                matched_max_mm_len_flag,
                raw_value,
            )
            continue
        matched_mm_cache_flag = None
        for flag in _MM_PROCESSOR_CACHE_GB_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_mm_cache_flag = flag
                break
        if matched_mm_cache_flag is not None:
            if arg == matched_mm_cache_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_mm_cache_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["mm_processor_cache_gb"] = _parse_nonnegative_float_flag(
                matched_mm_cache_flag,
                raw_value,
            )
            continue
        matched_speculative_config_flag = None
        for flag in _SPECULATIVE_CONFIG_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_speculative_config_flag = flag
                break
        if matched_speculative_config_flag is not None:
            if arg == matched_speculative_config_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_speculative_config_flag} requires a value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["speculative_config"] = _parse_json_object_flag(
                matched_speculative_config_flag,
                raw_value,
            )
            continue
        matched_max_model_len_flag = None
        for flag in _MAX_MODEL_LEN_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_max_model_len_flag = flag
                break
        if matched_max_model_len_flag is not None:
            if arg == matched_max_model_len_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_max_model_len_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["max_model_len"] = _parse_positive_int_flag(
                matched_max_model_len_flag,
                raw_value,
            )
            continue
        matched_capture_len_flag = None
        for flag in _MAX_SEQ_LEN_TO_CAPTURE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_capture_len_flag = flag
                break
        if matched_capture_len_flag is not None:
            if arg == matched_capture_len_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_capture_len_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            # 中文说明：vLLM perf_v2 配置常带 max_seq_len_to_capture。
            # 当前 SGLang 路径不需要单独 capture 长度；这里消费并校验它，
            # 避免迁移启动命令时因为未知参数失败。
            qwen35_args["max_seq_len_to_capture"] = _parse_positive_int_flag(
                matched_capture_len_flag,
                raw_value,
            )
            continue
        matched_port_flag = None
        for flag in _PORT_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_port_flag = flag
                break
        if matched_port_flag is not None:
            if arg == matched_port_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_port_flag} requires a port value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["port"] = _parse_positive_int_flag(
                matched_port_flag,
                raw_value,
            )
            continue
        matched_visible_key = None
        matched_visible_flag = None
        for key, flags in _VISIBLE_DEVICES_FLAG_GROUPS:
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_visible_key = key
                    matched_visible_flag = flag
                    break
            if matched_visible_flag is not None:
                break
        if matched_visible_key is not None and matched_visible_flag is not None:
            if arg == matched_visible_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_visible_flag} requires GPU ids")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_visible_key] = _parse_visible_devices_flag(
                matched_visible_flag,
                raw_value,
            )
            continue
        matched_compilation_flag = None
        for flag in _COMPILATION_CONFIG_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_compilation_flag = flag
                break
        if matched_compilation_flag is not None:
            if arg == matched_compilation_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_compilation_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["compilation_config"] = _parse_json_object_flag(
                matched_compilation_flag,
                raw_value,
            )
            continue
        matched_max_batched_tokens_flag = None
        for flag in _MAX_NUM_BATCHED_TOKENS_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_max_batched_tokens_flag = flag
                break
        if matched_max_batched_tokens_flag is not None:
            if arg == matched_max_batched_tokens_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_max_batched_tokens_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["max_num_batched_tokens"] = _parse_positive_int_flag(
                matched_max_batched_tokens_flag,
                raw_value,
            )
            continue
        matched_page_size_flag = None
        for flag in _PAGE_SIZE_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_page_size_flag = flag
                break
        if matched_page_size_flag is not None:
            if arg == matched_page_size_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_page_size_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["page_size"] = _parse_positive_int_flag(
                matched_page_size_flag,
                raw_value,
            )
            continue
        matched_mem_fraction_key = None
        matched_mem_fraction_flag = None
        for key, flags in (
            (
                "mem_fraction_static",
                ("--gpu-memory-utilization", "--gpu_memory_utilization"),
            ),
            (
                "thinker_mem_fraction_static",
                (
                    "--thinker-gpu-memory-utilization",
                    "--thinker_gpu_memory_utilization",
                ),
            ),
            (
                "talker_mem_fraction_static",
                (
                    "--talker-gpu-memory-utilization",
                    "--talker_gpu_memory_utilization",
                ),
            ),
        ):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_mem_fraction_key = key
                    matched_mem_fraction_flag = flag
                    break
            if matched_mem_fraction_flag is not None:
                break
        if matched_mem_fraction_key is not None and matched_mem_fraction_flag:
            if arg == matched_mem_fraction_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_mem_fraction_flag} requires a fraction value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_mem_fraction_key] = _parse_fraction_flag(
                matched_mem_fraction_flag,
                raw_value,
            )
            continue
        matched_limit_flag = None
        for flag in _LIMIT_MM_PER_PROMPT_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_limit_flag = flag
                break
        if matched_limit_flag is not None:
            if arg == matched_limit_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(f"{matched_limit_flag} requires a value")
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["limit_mm_per_prompt"] = _merge_limit_mm_per_prompt(
                qwen35_args["limit_mm_per_prompt"],
                _parse_limit_mm_per_prompt_flag(matched_limit_flag, raw_value),
            )
            continue
        matched_limit_modality = None
        matched_limit_modality_flag = None
        for modality, flags in _LIMIT_MM_PER_PROMPT_FLAG_GROUPS:
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_limit_modality = modality
                    matched_limit_modality_flag = flag
                    break
            if matched_limit_modality_flag is not None:
                break
        if matched_limit_modality is not None and matched_limit_modality_flag:
            if arg == matched_limit_modality_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_limit_modality_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["limit_mm_per_prompt"] = _merge_limit_mm_per_prompt(
                qwen35_args["limit_mm_per_prompt"],
                {
                    matched_limit_modality: _parse_nonnegative_int_flag(
                        matched_limit_modality_flag,
                        raw_value,
                    )
                },
            )
            continue
        matched_int_list_flag = None
        for flag in (
            *_CODE2WAV_DYNAMIC_CHUNK_SIZES_FLAGS,
            *_CODE2WAV_DYNAMIC_CHUNK_STEPS_FLAGS,
        ):
            if arg == flag or arg.startswith(f"{flag}="):
                matched_int_list_flag = flag
                break
        if matched_int_list_flag is not None:
            if arg == matched_int_list_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_int_list_flag} requires integer values"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            key = (
                "code2wav_dynamic_chunk_sizes"
                if matched_int_list_flag in _CODE2WAV_DYNAMIC_CHUNK_SIZES_FLAGS
                else "code2wav_dynamic_chunk_steps"
            )
            qwen35_args[key] = _parse_positive_int_list_flag(
                matched_int_list_flag,
                raw_value,
            )
            continue
        matched_video_fps_flag = None
        for flag in _VIDEO_FPS_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                matched_video_fps_flag = flag
                break
        if matched_video_fps_flag is not None:
            if arg == matched_video_fps_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_video_fps_flag} requires a float value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args["video_fps"] = _parse_positive_float_flag(
                matched_video_fps_flag,
                raw_value,
            )
            continue
        matched_video_float_key = None
        matched_video_float_flag = None
        for key, flags in _VIDEO_FLOAT_FLAG_GROUPS:
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_video_float_key = key
                    matched_video_float_flag = flag
                    break
            if matched_video_float_flag is not None:
                break
        if matched_video_float_key is not None and matched_video_float_flag:
            if arg == matched_video_float_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_video_float_flag} requires a float value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_video_float_key] = _parse_positive_float_flag(
                matched_video_float_flag,
                raw_value,
            )
            continue
        matched_max_running_key = None
        matched_max_running_flag = None
        for key, flags in _MAX_RUNNING_REQUEST_FLAG_GROUPS:
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_max_running_key = key
                    matched_max_running_flag = flag
                    break
            if matched_max_running_flag is not None:
                break
        if matched_max_running_key is not None and matched_max_running_flag:
            if arg == matched_max_running_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_max_running_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_max_running_key] = _parse_positive_int_flag(
                matched_max_running_flag,
                raw_value,
            )
            continue
        matched_video_int_key = None
        matched_video_int_flag = None
        for key, flags in (*_IMAGE_INT_FLAG_GROUPS, *_VIDEO_INT_FLAG_GROUPS):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_video_int_key = key
                    matched_video_int_flag = flag
                    break
            if matched_video_int_flag is not None:
                break
        if matched_video_int_key is not None and matched_video_int_flag is not None:
            if arg == matched_video_int_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_video_int_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            qwen35_args[matched_video_int_key] = _parse_positive_int_flag(
                matched_video_int_flag,
                raw_value,
            )
            continue
        matched_audio_int_key = None
        matched_audio_int_flag = None
        for key, flags in _AUDIO_INT_FLAG_GROUPS:
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_audio_int_key = key
                    matched_audio_int_flag = flag
                    break
            if matched_audio_int_flag is not None:
                break
        if matched_audio_int_key is not None and matched_audio_int_flag is not None:
            if arg == matched_audio_int_flag:
                if idx + 1 >= len(argv):
                    raise ValueError(
                        f"{matched_audio_int_flag} requires an integer value"
                    )
                raw_value = argv[idx + 1]
                idx += 2
            else:
                raw_value = arg.split("=", 1)[1]
                idx += 1
            parser = (
                _parse_nonnegative_int_flag
                if matched_audio_int_key == "audio_downsample_times"
                else _parse_positive_int_flag
            )
            qwen35_args[matched_audio_int_key] = parser(
                matched_audio_int_flag,
                raw_value,
            )
            continue
        for flag in _CODE2WAV_COMPILE_ENABLE_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["code2wav_enable_torch_compile"] = _parse_bool_flag(
                    flag,
                    arg.split("=", 1)[1],
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        for flag in _CODE2WAV_COMPILE_FIRST_CHUNK_ENABLE_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["code2wav_enable_torch_compile_first_chunk"] = (
                    _parse_bool_flag(
                        flag,
                        arg.split("=", 1)[1],
                    )
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        for flag in _CODE2WAV_DYNAMIC_CHUNK_ENABLE_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["code2wav_enable_dynamic_chunk"] = _parse_bool_flag(
                    flag,
                    arg.split("=", 1)[1],
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        for flag in _CODE2WAV_ODEINT_RELAXED_ENABLE_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["code2wav_odeint_method_relaxed"] = _parse_bool_flag(
                    flag,
                    arg.split("=", 1)[1],
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        for flag in _PREFIX_CACHING_ENABLE_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["enable_prefix_caching"] = _parse_bool_flag(
                    flag,
                    arg.split("=", 1)[1],
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        for flag in _CHUNKED_PREFILL_ENABLE_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["enable_chunked_prefill"] = _parse_bool_flag(
                    flag,
                    arg.split("=", 1)[1],
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        for flag in _ENABLE_TN_FLAGS:
            if arg.startswith(f"{flag}="):
                qwen35_args["enable_tn"] = _parse_bool_flag(
                    flag,
                    arg.split("=", 1)[1],
                )
                idx += 1
                break
        else:
            flag = None
        if flag is not None:
            continue

        matched_path_key = None
        matched_flag = None
        for key, flags in (
            ("talker_model_path", _TALKER_MODEL_PATH_FLAGS),
            ("code2wav_model_path", _CODE2WAV_MODEL_PATH_FLAGS),
            ("code2wav_model_folder", _CODE2WAV_MODEL_FOLDER_FLAGS),
            ("xvector_info_paths", _XVECTOR_INFO_PATH_FLAGS),
        ):
            for flag in flags:
                if arg == flag or arg.startswith(f"{flag}="):
                    matched_path_key = key
                    matched_flag = flag
                    break
            if matched_flag is not None:
                break
        if matched_flag is None or matched_path_key is None:
            cleaned.append(arg)
            idx += 1
            continue

        if arg == matched_flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{matched_flag} requires a path value")
            raw_path = argv[idx + 1]
            if matched_path_key == "xvector_info_paths":
                qwen35_args[matched_path_key].append(raw_path)
            elif (
                matched_path_key == "code2wav_model_path"
                and matched_flag in _CODE2WAV_MODEL_VLLM_ALIAS_FLAGS
                and _looks_like_relative_code2wav_folder(raw_path)
            ):
                qwen35_args["code2wav_model_folder"] = raw_path
            else:
                qwen35_args[matched_path_key] = raw_path
            idx += 2
            continue

        raw_path = arg.split("=", 1)[1]
        if matched_path_key == "xvector_info_paths":
            qwen35_args[matched_path_key].append(raw_path)
        elif (
            matched_path_key == "code2wav_model_path"
            and matched_flag in _CODE2WAV_MODEL_VLLM_ALIAS_FLAGS
            and _looks_like_relative_code2wav_folder(raw_path)
        ):
            qwen35_args["code2wav_model_folder"] = raw_path
        else:
            qwen35_args[matched_path_key] = raw_path
        idx += 1

    return cleaned, qwen35_args


def _apply_visible_devices_args(
    args: argparse.Namespace,
    qwen35_args: dict[str, Any],
) -> None:
    thinker_visible_devices = qwen35_args["thinker_visible_devices"]
    if thinker_visible_devices is not None:
        # 中文说明：vLLM perf 配置里的 thinker_visible_devices 对齐
        # SGLang 的 thinker TP placement。单卡退化成 --gpu-thinker，
        # 多卡则自动填 --thinker-tp-size + --gpu-thinker-tp。
        args.gpu_thinker = int(thinker_visible_devices[0])
        args.thinker_tp_size = len(thinker_visible_devices)
        args.gpu_thinker_tp = (
            None
            if len(thinker_visible_devices) == 1
            else ",".join(str(gpu) for gpu in thinker_visible_devices)
        )

    for key, attr in (
        ("talker_visible_devices", "gpu_talker"),
        ("code2wav_visible_devices", "gpu_code2wav"),
    ):
        devices = qwen35_args[key]
        if devices is None:
            continue
        if len(devices) != 1:
            flag_name = "--" + key.replace("_", "-")
            raise ValueError(f"{flag_name} currently supports exactly one GPU")
        setattr(args, attr, int(devices[0]))


def _apply_tensor_parallel_size_arg(
    args: argparse.Namespace,
    qwen35_args: dict[str, Any],
) -> None:
    tensor_parallel_size = qwen35_args["tensor_parallel_size"]
    if tensor_parallel_size is None:
        return
    if args.thinker_tp_size not in (1, tensor_parallel_size):
        raise ValueError(
            "--tensor-parallel-size conflicts with explicit thinker TP placement"
        )
    if tensor_parallel_size > 1 and args.gpu_thinker_tp is None:
        raise ValueError(
            "--tensor-parallel-size > 1 requires --thinker-visible-devices "
            "so SGLang can place thinker tensor-parallel workers"
        )
    args.thinker_tp_size = int(tensor_parallel_size)


def _parse_bool_flag(flag_name: str, raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{flag_name} expects a boolean value, got {raw!r}")


def _coerce_bool_value(flag_name: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(value)
    # 中文说明：_launch_speech_server 在单测/外部脚本里可能直接收到
    # argparse.Namespace，而不是 parse_args 产出的 bool；显式解析字符串，
    # 避免 bool("false") 这类 Python 语义把关闭开关误当开启。
    return _parse_bool_flag(flag_name, str(value))


def _parse_positive_int_flag(flag_name: str, raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise ValueError(f"{flag_name} must be >= 1, got {value}")
    return value


def _validate_positive_int_value(flag_name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{flag_name} must be >= 1, got {value}")
    return value


def _parse_positive_float_flag(flag_name: str, raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{flag_name} must be > 0, got {value}")
    return value


def _parse_nonnegative_float_flag(flag_name: str, raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise ValueError(f"{flag_name} must be >= 0, got {value}")
    return value


def _parse_nonempty_string_flag(flag_name: str, raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError(f"{flag_name} must not be empty")
    return value


def _parse_positive_int_list_flag(flag_name: str, raw: str) -> tuple[int, ...]:
    pieces = [
        piece.strip()
        for piece in raw.replace(",", " ").split()
        if piece.strip()
    ]
    if not pieces:
        raise ValueError(f"{flag_name} requires at least one integer value")
    return tuple(_parse_positive_int_flag(flag_name, piece) for piece in pieces)


def _parse_visible_devices_flag(flag_name: str, raw: str) -> tuple[int, ...]:
    text = raw.strip()
    if not text:
        raise ValueError(f"{flag_name} requires at least one GPU id")
    if text.startswith("["):
        parsed = json.loads(text)
    else:
        parsed = [piece.strip() for piece in text.replace(",", " ").split()]
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{flag_name} must be an int or list of ints")
    devices: list[int] = []
    for item in parsed:
        gpu = int(item)
        if gpu < 0:
            raise ValueError(f"{flag_name} GPU ids must be >= 0")
        devices.append(gpu)
    return tuple(devices)


def _parse_json_object_flag(flag_name: str, raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} expects a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{flag_name} expects a JSON object")
    return value


def _parse_odeint_method_flag(flag_name: str, raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _CODE2WAV_ODEINT_METHODS:
        supported = ", ".join(sorted(_CODE2WAV_ODEINT_METHODS))
        raise ValueError(f"{flag_name} must be one of: {supported}")
    return normalized


def _parse_code2wav_frequency_flag(flag_name: str, raw: str) -> str:
    normalized = raw.strip().lower()
    value = _CODE2WAV_FREQUENCY_ALIASES.get(normalized)
    if value is None:
        supported = ", ".join(
            sorted((*_CODE2WAV_FREQUENCIES, "1", "2", "25", "50"))
        )
        raise ValueError(f"{flag_name} must be one of: {supported}")
    return value


def _parse_code2wav_dit_quant_flag(flag_name: str, raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _CODE2WAV_DIT_QUANTS:
        supported = ", ".join(sorted(_CODE2WAV_DIT_QUANTS))
        raise ValueError(f"{flag_name} must be one of: {supported}")
    return normalized


def _parse_quantization_flag(
    flag_name: str,
    raw: str,
    *,
    supported: frozenset[str],
) -> str:
    normalized = raw.strip().lower()
    if normalized not in supported:
        supported_values = ", ".join(sorted(supported))
        raise ValueError(f"{flag_name} must be one of: {supported_values}")
    return normalized


def _parse_dtype_flag(flag_name: str, raw: str) -> str:
    normalized = raw.strip().lower()
    if not normalized:
        raise ValueError(f"{flag_name} must not be empty")
    return normalized


def _parse_mamba_cache_mode_flag(flag_name: str, raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in _MAMBA_CACHE_MODES:
        supported_values = ", ".join(sorted(_MAMBA_CACHE_MODES))
        raise ValueError(f"{flag_name} must be one of: {supported_values}")
    return normalized


def _parse_fraction_flag(flag_name: str, raw: str) -> float:
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")
    return value


def _parse_limit_mm_per_prompt_flag(flag_name: str, raw: str) -> dict[str, int]:
    raw = raw.strip()
    if not raw:
        raise ValueError(f"{flag_name} must not be empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = _parse_limit_mm_pairs(flag_name, raw)
    if not isinstance(value, dict):
        raise ValueError(f"{flag_name} expects a JSON object")
    return _normalize_limit_mm_per_prompt_flag(flag_name, value)


def _parse_limit_mm_pairs(flag_name: str, raw: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    pieces = [
        piece.strip()
        for piece in raw.replace(",", " ").split()
        if piece.strip()
    ]
    for piece in pieces:
        if "=" not in piece:
            raise ValueError(f"{flag_name} expects JSON or modality=count pairs")
        key, raw_value = piece.split("=", 1)
        parsed[key.strip()] = int(raw_value)
    return parsed


def _normalize_limit_mm_per_prompt_flag(
    flag_name: str,
    value: dict[object, object],
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_key, raw_limit in value.items():
        modality = str(raw_key).strip().lower()
        if modality.endswith("s"):
            modality = modality[:-1]
        if modality not in _LIMIT_MM_MODALITIES:
            supported = ", ".join(sorted(_LIMIT_MM_MODALITIES))
            raise ValueError(f"{flag_name} modality must be one of: {supported}")
        limit = int(raw_limit)
        if limit < 0:
            raise ValueError(f"{flag_name} values must be >= 0")
        normalized[modality] = limit
    return normalized


def _merge_limit_mm_per_prompt(
    base: dict[str, int] | None,
    updates: dict[str, int],
) -> dict[str, int]:
    merged = dict(base or {})
    merged.update(updates)
    return merged


def _parse_nonnegative_int_flag(flag_name: str, raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise ValueError(f"{flag_name} must be >= 0, got {value}")
    return value


def _apply_video_preprocessing_runtime_args(
    config: Any,
    args: argparse.Namespace,
) -> None:
    updates = {
        "max_seq_len": getattr(args, "max_mm_len", None),
        "image_min_pixels": getattr(args, "image_min_pixels", None),
        "image_max_pixels": getattr(args, "image_max_pixels", None),
        "video_fps": getattr(args, "video_fps", None),
        "video_max_frames": getattr(args, "video_max_frames", None),
        "video_min_frames": getattr(args, "video_min_frames", None),
        "video_min_pixels": getattr(args, "video_min_pixels", None),
        "video_max_pixels": getattr(args, "video_max_pixels", None),
        "video_total_pixels": getattr(args, "video_total_pixels", None),
        "video_override_max_pixels": (
            _coerce_bool_value(
                "--override-video-max-pixels",
                getattr(args, "override_video_max_pixels", False),
            )
            if getattr(args, "override_video_max_pixels", None) is not None
            else None
        ),
        "video_seconds_per_chunk": getattr(args, "video_seconds_per_chunk", None),
        "video_position_id_per_seconds": getattr(
            args,
            "video_position_id_per_seconds",
            None,
        ),
        "audio_target_sr": getattr(args, "audio_target_sr", None),
        "audio_timestamp_interval": getattr(args, "audio_timestamp_interval", None),
        "audio_downsample_times": getattr(args, "audio_downsample_times", None),
        "audio_downsample_chunk_size": getattr(
            args,
            "audio_downsample_chunk_size",
            None,
        ),
    }
    updates = {key: value for key, value in updates.items() if value is not None}
    if not updates:
        return
    for stage in config.stages:
        if stage.name != "preprocessing":
            continue
        # 中文说明：example launcher 的视频参数是服务级默认值，
        # 进入 runtime 后由 runtime_arg_map 转成 preprocessor 构造参数。
        for key, value in updates.items():
            setattr(stage.runtime, key, value)


def _set_stage_max_running_requests(
    config: Any,
    *,
    stage_name: str,
    value: int,
) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.runtime.sglang_server_args.max_running_requests = int(value)


def _apply_stage_runtime_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object],
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        for key, value in updates.items():
            if key not in stage.runtime_arg_map:
                raise ValueError(
                    f"runtime.{key} is not supported by stage {stage_name!r}"
                )
            setattr(stage.runtime, key, value)


def _apply_vllm_ar_server_args(config: Any, args: argparse.Namespace) -> None:
    _validate_vllm_mamba_cache_mode(getattr(args, "mamba_cache_mode", None))
    _validate_vllm_kv_transfer_request(
        enable_disaggregated_prefilling=getattr(
            args,
            "enable_disaggregated_prefilling",
            None,
        ),
        kv_transfer_config=getattr(args, "kv_transfer_config", None),
    )
    _validate_vllm_engine_profile_args(args, speech=True)

    shared_updates: dict[str, object] = {}
    if getattr(args, "enable_prefix_caching", None) is not None:
        # 中文说明：vLLM 的 enable_prefix_caching 对应 SGLang 的 radix
        # cache。SGLang 参数是反向开关 disable_radix_cache。
        shared_updates["disable_radix_cache"] = not _coerce_bool_value(
            "--enable-prefix-caching",
            args.enable_prefix_caching,
        )
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    if max_num_batched_tokens is not None:
        max_num_batched_tokens = _validate_positive_int_value(
            "--max-num-batched-tokens",
            max_num_batched_tokens,
        )
        # 中文说明：vLLM 的 max_num_batched_tokens 控制 prefill 批处理
        # token 上限；SGLang 对应 max_prefill_tokens，并在 chunked prefill
        # 未显式关闭时同步作为 chunked_prefill_size。
        shared_updates["max_prefill_tokens"] = max_num_batched_tokens
        shared_updates["chunked_prefill_size"] = max_num_batched_tokens
    if getattr(args, "enable_chunked_prefill", None) is not None:
        shared_updates["chunked_prefill_size"] = (
            max_num_batched_tokens or QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE
            if _coerce_bool_value(
                "--enable-chunked-prefill",
                args.enable_chunked_prefill,
            )
            else None
        )
    if getattr(args, "dtype", None) is not None:
        # 中文说明：vLLM perf_v2 configs 常设置 dtype=bfloat16；
        # SGLang AR stage 对应 ServerArgs.dtype。
        shared_updates["dtype"] = args.dtype
    if getattr(args, "mamba_ssm_dtype", None) is not None:
        # 中文说明：vLLM 的 mamba_cache_dtype 对应 SGLang core 的
        # mamba_ssm_dtype；两边 Qwen3.5 默认都是 float32。
        shared_updates["mamba_ssm_dtype"] = args.mamba_ssm_dtype
    if getattr(args, "page_size", None) is not None:
        # 中文说明：vLLM block_size 对应 SGLang page_size，控制 KV cache
        # 分页大小；保留 --page-size 作为更贴近 SGLang 的名字。
        shared_updates["page_size"] = _validate_positive_int_value(
            "--block-size",
            args.page_size,
        )
    compilation_updates = _server_args_from_vllm_compilation_config(
        getattr(args, "compilation_config", None)
    )
    shared_updates.update(compilation_updates)

    for stage_name in ("thinker", "talker_ar"):
        if shared_updates:
            _apply_stage_factory_updates(
                config,
                stage_name=stage_name,
                server_arg_updates=shared_updates,
            )

    if _coerce_bool_value(
        "--enforce-eager",
        getattr(args, "enforce_eager", False),
    ) or _coerce_bool_value(
        "--thinker-enforce-eager",
        getattr(args, "thinker_enforce_eager", False),
    ):
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"disable_cuda_graph": True},
        )
    if _coerce_bool_value(
        "--enforce-eager",
        getattr(args, "enforce_eager", False),
    ) or _coerce_bool_value(
        "--talker-enforce-eager",
        getattr(args, "talker_enforce_eager", False),
    ):
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"disable_cuda_graph": True},
        )

    thinker_quantization = getattr(args, "thinker_quantization", None)
    if thinker_quantization is not None and thinker_quantization != "none":
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"quantization": thinker_quantization},
        )
    talker_quantization = getattr(args, "talker_quantization", None)
    if talker_quantization is not None and talker_quantization != "none":
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"quantization": talker_quantization},
        )

    thinker_dtype = getattr(args, "thinker_dtype", None)
    if thinker_dtype is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"dtype": thinker_dtype},
        )
    talker_dtype = getattr(args, "talker_dtype", None)
    if talker_dtype is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"dtype": talker_dtype},
        )


def _server_args_from_vllm_compilation_config(
    compilation_config: dict[str, Any] | None,
) -> dict[str, object]:
    if compilation_config is None:
        return {}
    if not isinstance(compilation_config, dict):
        raise ValueError("--compilation-config expects a JSON object")

    # 中文说明：vLLM perf_v2 Qwen3.5 profiles 默认是 FULL_DECODE_ONLY，
    # use_inductor=false，fuse pass 关闭。SGLang 这里没有逐项 pass_config
    # 开关；只把“显式关闭 cudagraph”的语义映射到 disable_cuda_graph。
    use_inductor = compilation_config.get("use_inductor")
    if use_inductor is not None and _coerce_bool_value(
        "--compilation-config use_inductor",
        use_inductor,
    ):
        raise ValueError("--compilation-config use_inductor=true is not supported")
    pass_config = compilation_config.get("pass_config")
    if pass_config is not None:
        if not isinstance(pass_config, dict):
            raise ValueError("--compilation-config pass_config must be an object")
        enabled = [
            key
            for key, value in pass_config.items()
            if _coerce_bool_value(
                f"--compilation-config pass_config.{key}",
                value,
            )
        ]
        if enabled:
            raise ValueError(
                "--compilation-config fuse pass options are not supported: "
                + ", ".join(sorted(str(key) for key in enabled))
            )

    raw_mode = str(compilation_config.get("cudagraph_mode", "")).strip().lower()
    if not raw_mode:
        return {}
    normalized = raw_mode.replace("-", "_")
    if normalized in {"full_decode_only", "full", "decode_only"}:
        return {}
    if normalized in {"none", "off", "no_cudagraph", "disable", "disabled"}:
        return {"disable_cuda_graph": True}
    raise ValueError(
        "--compilation-config cudagraph_mode must be FULL_DECODE_ONLY or none/off"
    )


def _validate_vllm_mamba_cache_mode(mode: str | None) -> None:
    if mode in (None, "none"):
        return
    # 中文说明：vLLM 的 light/all 是混合 Mamba cache 策略，常和
    # disaggregated prefill/decode profile 一起出现。当前 SGLang core 暴露
    # 的是 mamba_ssm_dtype 等参数，没有等价的 mamba_cache_mode ServerArgs；
    # 这里提前失败，避免把 vLLM perf_v2 配置静默降级。
    raise ValueError(
        "--mamba-cache-mode light/all is a vLLM hybrid Mamba cache setting "
        "and is not supported by the current SGLang Qwen3.5 path; use "
        "--mamba-cache-mode none or omit it until an explicit SGLang mapping "
        "is implemented."
    )


def _validate_vllm_kv_transfer_request(
    *,
    enable_disaggregated_prefilling: bool | None,
    kv_transfer_config: dict[str, Any] | None,
) -> None:
    if enable_disaggregated_prefilling is not None:
        enable_disaggregated_prefilling = _coerce_bool_value(
            "--enable-disaggregated-prefilling",
            enable_disaggregated_prefilling,
        )
    if enable_disaggregated_prefilling:
        # 中文说明：vLLM 的分离式 prefill/decode 需要 KV connector
        # producer/consumer 协议。当前 SGLang-Omni Qwen3.5 还没有把这个
        # profile 映射到 SGLang KV transfer，因此不能当普通调优项忽略。
        raise ValueError(
            "--enable-disaggregated-prefilling is a vLLM KV transfer setting "
            "and is not supported by the current SGLang Qwen3.5 path."
        )
    if kv_transfer_config:
        raise ValueError(
            "--kv-transfer-config is a vLLM KV connector setting and is not "
            "supported by the current SGLang Qwen3.5 path; use the colocated "
            "or non-disaggregated profile until an explicit SGLang KV transfer "
            "mapping is implemented."
        )


def _validate_vllm_speculative_config_request(
    speculative_config: dict[str, Any] | None,
    *,
    disable_mtp: bool | None = False,
) -> None:
    if speculative_config is None:
        return
    if not speculative_config:
        logger.info("--speculative-config={} accepted as a vLLM-compatible no-op")
        return
    method = str(speculative_config.get("method", "")).strip().lower()
    if method == "qwen3_omni_next_thinker_mtp":
        if disable_mtp:
            logger.info(
                "--speculative-config method qwen3_omni_next_thinker_mtp "
                "ignored because --disable-mtp selects the base Qwen3.5 "
                "thinker AR path"
            )
            return
        raise ValueError(
            "--speculative-config method qwen3_omni_next_thinker_mtp requires "
            "Qwen3.5 thinker MTP/draft decoding, which is not enabled in this "
            "SGLang path yet; omit the speculative config or use --disable-mtp "
            "for the current base thinker AR path"
        )
    # 中文说明：非空 speculative_config 会改变解码路径和 worker/KV 形态。
    # 当前 Qwen3.5 launcher 先实现 base thinker AR 主链路，因此提前失败。
    raise ValueError(
        "--speculative-config is not mapped in the current SGLang Qwen3.5 path"
    )


def _validate_vllm_engine_profile_args(
    args: argparse.Namespace,
    *,
    speech: bool,
) -> None:
    tensor_parallel_size = getattr(args, "tensor_parallel_size", None)
    if tensor_parallel_size is not None:
        if int(tensor_parallel_size) < 1:
            raise ValueError("--tensor-parallel-size must be >= 1")
        if int(tensor_parallel_size) > 1 and getattr(args, "gpu_thinker_tp", None) is None:
            raise ValueError(
                "--tensor-parallel-size > 1 requires --thinker-visible-devices "
                "so SGLang can place thinker tensor-parallel workers"
            )

    backend = getattr(args, "distributed_executor_backend", None)
    if backend is not None and backend.strip().lower() != "mp":
        raise ValueError(
            "--distributed-executor-backend currently supports only the vLLM "
            "mp profile value in SGLang Qwen3.5 launchers"
        )

    kv_cache_dtype = getattr(args, "kv_cache_dtype", None)
    if kv_cache_dtype is not None and kv_cache_dtype.strip().lower() != "auto":
        # 中文说明：vLLM 的 tq4/fp8 KV cache dtype 会影响 KV cache 存储和
        # attention backend；SGLang 这里没有等价映射，不能静默降级成 auto。
        raise ValueError(
            "--kv-cache-dtype values other than auto are not supported by the "
            "current SGLang Qwen3.5 path"
        )

    enable_expert_parallel = getattr(args, "enable_expert_parallel", None)
    if enable_expert_parallel is not None and _coerce_bool_value(
        "--enable-expert-parallel",
        enable_expert_parallel,
    ):
        raise ValueError(
            "--enable-expert-parallel is a vLLM MoE parallelism setting and "
            "is not mapped in the current SGLang Qwen3.5 path"
        )

    mm_cache_gb = getattr(args, "mm_processor_cache_gb", None)
    if mm_cache_gb not in (None, 0, 0.0):
        raise ValueError(
            "--mm-processor-cache-gb currently supports only 0 as a "
            "vLLM-compatible no-op in SGLang Qwen3.5 launchers"
        )
    video_needs_metadata = getattr(args, "video_needs_metadata", None)
    if video_needs_metadata is not None:
        _coerce_bool_value("--video-needs-metadata", video_needs_metadata)
        logger.info(
            "--video-needs-metadata accepted as a Qwen3.5 vLLM-compatible "
            "no-op; SGLang Qwen3.5 preprocessing already requests video "
            "metadata when building prompts."
        )

    _validate_vllm_speculative_config_request(
        getattr(args, "speculative_config", None),
        disable_mtp=getattr(args, "disable_mtp", False),
    )

    max_mm_len = getattr(args, "max_mm_len", None)
    if max_mm_len is not None:
        max_mm_len = _validate_positive_int_value("--max-mm-len", int(max_mm_len))
        if max_mm_len > int(args.thinker_max_seq_len):
            raise ValueError(
                "--max-mm-len must be <= thinker context length in the current "
                "SGLang Qwen3.5 path"
            )
    is_thinker = getattr(args, "is_thinker", None)
    if is_thinker is not None:
        is_thinker = _coerce_bool_value("--is-thinker", is_thinker)
    if is_thinker is False:
        raise ValueError("--is-thinker=false is not valid for this Qwen3.5 launcher")
    thinker_only = getattr(args, "thinker_only", None)
    if thinker_only is not None:
        thinker_only = _coerce_bool_value("--thinker-only", thinker_only)
    if speech and thinker_only is True:
        raise ValueError("--thinker-only=true conflicts with the speech launcher")
    if not speech and thinker_only is False:
        raise ValueError("--thinker-only=false conflicts with the thinker-only launcher")


def _run_preflight_or_raise(
    *,
    model_path: str,
    speech: bool,
    code2wav_model_path: str | None = None,
    xvector_info_paths: tuple[str, ...] = (),
    validate_xvector_pickle: bool = False,
) -> None:
    from sglang_omni.models.qwen3_5_omni.preflight import (
        format_preflight_report,
        run_qwen35_preflight,
    )

    extra_kwargs: dict[str, object] = {}
    if xvector_info_paths:
        extra_kwargs["xvector_info_paths"] = xvector_info_paths
    if validate_xvector_pickle:
        extra_kwargs["validate_xvector_pickle"] = True
    report = run_qwen35_preflight(
        model_path,
        speech=speech,
        code2wav_model_path=code2wav_model_path,
        **extra_kwargs,
    )
    message = format_preflight_report(report)
    if not report.ok:
        raise RuntimeError(message)
    logger.info("%s", message)


def _launch_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_5_omni.config import (
        MIN_PARTIAL_START_CHUNKS,
        Qwen35OmniSpeechColocatedPipelineConfig,
        Qwen35OmniSpeechPipelineConfig,
    )
    from sglang_omni.serve import launch_server

    for flag_name, value in (
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--thinker-mem-fraction-static", args.thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", args.talker_mem_fraction_static),
    ):
        _validate_fraction(flag_name, value)

    colocated = _coerce_bool_value("--colocated", getattr(args, "colocated", False))

    enable_partial_start = (
        not colocated
        if args.enable_partial_start is None
        else _coerce_bool_value(
            "--enable-partial-start",
            args.enable_partial_start,
        )
    )

    if (
        enable_partial_start
        and args.partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS
    ):
        raise ValueError(
            f"--partial-start-min-chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
            f"got {args.partial_start_min_chunks}"
        )

    if _coerce_bool_value("--preflight", getattr(args, "preflight", False)):
        _run_preflight_or_raise(
            model_path=args.model_path,
            speech=True,
            code2wav_model_path=getattr(args, "code2wav_model_path", None),
            xvector_info_paths=tuple(getattr(args, "xvector_info_paths", ())),
            validate_xvector_pickle=_coerce_bool_value(
                "--validate-xvector-pickle",
                getattr(args, "validate_xvector_pickle", False),
            ),
        )

    gpu_talker = (
        args.gpu_talker
        if args.gpu_talker is not None
        else (args.gpu_thinker if colocated else 1)
    )
    gpu_code2wav = (
        args.gpu_code2wav
        if args.gpu_code2wav is not None
        # 中文说明：vLLM perf_v2 的 Qwen3.5 参考部署把 talker 和
        # code2wav 放在同一张卡，避免 code2wav 默认压到 thinker GPU。
        else (args.gpu_thinker if colocated else gpu_talker)
    )
    gpu_image_encoder = (
        args.gpu_image_encoder
        if args.gpu_image_encoder is not None
        else (args.gpu_thinker if colocated else 0)
    )
    gpu_audio_encoder = (
        args.gpu_audio_encoder
        if args.gpu_audio_encoder is not None
        else (args.gpu_thinker if colocated else 0)
    )
    if colocated:
        colocated_gpus = {
            "--gpu-thinker": args.gpu_thinker,
            "--gpu-talker": gpu_talker,
            "--gpu-code2wav": gpu_code2wav,
            "--gpu-image-encoder": gpu_image_encoder,
            "--gpu-audio-encoder": gpu_audio_encoder,
        }
        if len(set(colocated_gpus.values())) != 1:
            raise ValueError(
                "--colocated requires all GPU stage flags to use the same GPU, "
                f"got {colocated_gpus}"
            )

    gpu_code_predictor = (
        args.gpu_code_predictor if args.gpu_code_predictor is not None else gpu_talker
    )
    if gpu_code_predictor != gpu_talker:
        raise ValueError(
            "Qwen3.5 speech pipeline does not expose a separate code_predictor "
            "stage. Use the same GPU for --gpu-code-predictor and --gpu-talker."
        )

    config_cls = (
        Qwen35OmniSpeechColocatedPipelineConfig
        if colocated
        else Qwen35OmniSpeechPipelineConfig
    )
    config = config_cls(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    _set_stage_gpu(config, "image_encoder", gpu_image_encoder)
    _set_stage_gpu(config, "audio_encoder", gpu_audio_encoder)

    if args.thinker_tp_size < 1:
        raise ValueError(f"--thinker-tp-size must be >= 1, got {args.thinker_tp_size}")

    if args.thinker_tp_size > 1:
        if args.gpu_thinker_tp is None:
            raise ValueError(
                "--thinker-tp-size > 1 requires --gpu-thinker-tp "
                "(comma-separated GPU ids, one per TP rank)."
            )
        thinker_gpu_ids = _parse_thinker_tp_gpu_list(
            args.gpu_thinker_tp, args.thinker_tp_size
        )
        _set_stage_tp_size(config, "thinker", args.thinker_tp_size)
        _set_stage_gpu(config, "thinker", thinker_gpu_ids)
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"disable_custom_all_reduce": True},
        )
    else:
        if args.gpu_thinker_tp is not None:
            raise ValueError(
                "--gpu-thinker-tp only applies when --thinker-tp-size > 1; "
                "for TP=1, use --gpu-thinker."
            )
        _set_stage_gpu(config, "thinker", args.gpu_thinker)

    _set_stage_gpu(config, "talker_ar", gpu_talker)
    _set_stage_gpu(config, "code2wav", gpu_code2wav)
    if getattr(args, "talker_model_path", None):
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            # 中文说明：显式 talker 子目录通常保存未带 "talker." 前缀的权重；
            # root checkpoint 仍走默认 model_path + weight_prefix="talker."。
            # root_model_path 保留 tokenizer/special token/thinker metadata 来源。
            updates={
                "model_path": args.talker_model_path,
                "root_model_path": args.model_path,
                "weight_prefix": "",
            },
        )
    if getattr(args, "code2wav_model_path", None):
        _apply_stage_factory_updates(
            config,
            stage_name="code2wav",
            updates={"code2wav_model_path": args.code2wav_model_path},
        )
    code2wav_runtime_updates: dict[str, object] = {}
    if getattr(args, "code2wav_enable_torch_compile", None) is not None:
        code2wav_runtime_updates["code2wav_enable_torch_compile"] = (
            _coerce_bool_value(
                "--code2wav-enable-torch-compile",
                args.code2wav_enable_torch_compile,
            )
        )
    if getattr(args, "code2wav_enable_torch_compile_first_chunk", None) is not None:
        code2wav_runtime_updates[
            "code2wav_enable_torch_compile_first_chunk"
        ] = _coerce_bool_value(
            "--code2wav-enable-torch-compile-first-chunk",
            args.code2wav_enable_torch_compile_first_chunk,
        )
    if getattr(args, "code2wav_codec_eos_token_id", None) is not None:
        code2wav_runtime_updates["code2wav_codec_eos_token_id"] = int(
            args.code2wav_codec_eos_token_id
        )
    if getattr(args, "code2wav_sample_rate", None) is not None:
        code2wav_runtime_updates["code2wav_sample_rate"] = int(
            args.code2wav_sample_rate
        )
    if getattr(args, "code2wav_stream_chunk_size", None) is not None:
        code2wav_runtime_updates["code2wav_stream_chunk_size"] = int(
            args.code2wav_stream_chunk_size
        )
    if getattr(args, "code2wav_left_context_size", None) is not None:
        code2wav_runtime_updates["code2wav_left_context_size"] = int(
            args.code2wav_left_context_size
        )
    if getattr(args, "code2wav_enable_dynamic_chunk", None) is not None:
        code2wav_runtime_updates["code2wav_enable_dynamic_chunk"] = (
            _coerce_bool_value(
                "--code2wav-enable-dynamic-chunk",
                args.code2wav_enable_dynamic_chunk,
            )
        )
    if getattr(args, "code2wav_dynamic_chunk_sizes", None) is not None:
        code2wav_runtime_updates["code2wav_dynamic_chunk_sizes"] = (
            args.code2wav_dynamic_chunk_sizes
        )
    if getattr(args, "code2wav_dynamic_chunk_steps", None) is not None:
        code2wav_runtime_updates["code2wav_dynamic_chunk_steps"] = (
            args.code2wav_dynamic_chunk_steps
        )
    if getattr(args, "code2wav_odeint_method", None) is not None:
        code2wav_runtime_updates["code2wav_odeint_method"] = (
            args.code2wav_odeint_method
        )
    if getattr(args, "code2wav_odeint_method_relaxed", None) is not None:
        code2wav_runtime_updates["code2wav_odeint_method_relaxed"] = (
            _coerce_bool_value(
                "--code2wav-odeint-method-relaxed",
                args.code2wav_odeint_method_relaxed,
            )
        )
    if getattr(args, "code2wav_batched_chunk", None) is not None:
        code2wav_runtime_updates["code2wav_batched_chunk"] = int(
            args.code2wav_batched_chunk
        )
    if getattr(args, "code2wav_frequency", None) is not None:
        code2wav_runtime_updates["code2wav_frequency"] = args.code2wav_frequency
    if getattr(args, "code2wav_dit_quant", None) is not None:
        code2wav_runtime_updates["code2wav_dit_quant"] = args.code2wav_dit_quant
    if code2wav_runtime_updates:
        _apply_stage_runtime_updates(
            config,
            stage_name="code2wav",
            updates=code2wav_runtime_updates,
        )

    _apply_vllm_ar_server_args(config, args)
    _apply_max_running_requests_args(config, args)
    _apply_mem_fraction_args(config, args)

    if args.thinker_max_seq_len is not None:
        thinker_seq_len_updates: dict[str, object] = {
            "thinker_max_seq_len": int(args.thinker_max_seq_len)
        }
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=thinker_seq_len_updates,
        )
        _apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=thinker_seq_len_updates,
        )
    _apply_video_preprocessing_runtime_args(config, args)
    _apply_limit_mm_per_prompt_args(config, args)

    talker_partial_start_updates: dict[str, object] = {
        "enable_partial_start": enable_partial_start,
    }
    if enable_partial_start:
        talker_partial_start_updates["partial_start_min_chunks"] = int(
            args.partial_start_min_chunks
        )
    _apply_stage_factory_updates(
        config,
        stage_name="talker_ar",
        updates=talker_partial_start_updates,
    )

    default_talker_params: dict[str, object] = {}
    if getattr(args, "voice_type", None):
        default_talker_params["voice_type"] = args.voice_type
    if getattr(args, "enable_tn", None) is not None:
        default_talker_params["enable_tn"] = _coerce_bool_value(
            "--enable-tn",
            args.enable_tn,
        )
    default_generation_params: dict[str, object] = {
        # 中文说明：对齐 vLLM qwen_omni_v35_server.py 的请求默认值；
        # 单个 OpenAI 请求显式传入的采样参数会在 serve 层覆盖这些默认值。
        "temperature": QWEN35_DEFAULT_TEMPERATURE,
        "top_k": QWEN35_DEFAULT_TOP_K,
        "top_p": QWEN35_DEFAULT_TOP_P,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    }
    if getattr(args, "max_tokens", None) is not None:
        default_generation_params["max_tokens"] = int(args.max_tokens)
    if getattr(args, "seed", None) is not None:
        default_generation_params["seed"] = int(args.seed)

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        default_talker_params=default_talker_params or None,
        default_generation_params=default_generation_params,
    )


def _apply_max_running_requests_args(config: Any, args: argparse.Namespace) -> None:
    thinker_max_running = (
        args.thinker_max_running_requests
        if args.thinker_max_running_requests is not None
        else args.max_running_requests
    )
    talker_max_running = (
        args.talker_max_running_requests
        if args.talker_max_running_requests is not None
        else args.max_running_requests
    )

    if thinker_max_running is not None:
        thinker_max_running = _validate_positive_int_value(
            "--thinker-max-running-requests",
            thinker_max_running,
        )
        _set_stage_max_running_requests(
            config,
            stage_name="thinker",
            value=thinker_max_running,
        )
    if talker_max_running is not None:
        talker_max_running = _validate_positive_int_value(
            "--talker-max-running-requests",
            talker_max_running,
        )
        _set_stage_max_running_requests(
            config,
            stage_name="talker_ar",
            value=talker_max_running,
        )


def _apply_mem_fraction_args(config: Any, args: argparse.Namespace) -> None:
    thinker_mem_fraction = (
        args.thinker_mem_fraction_static
        if args.thinker_mem_fraction_static is not None
        else args.mem_fraction_static
    )
    talker_mem_fraction = (
        args.talker_mem_fraction_static
        if args.talker_mem_fraction_static is not None
        else args.mem_fraction_static
    )

    if thinker_mem_fraction is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"mem_fraction_static": thinker_mem_fraction},
        )
    if talker_mem_fraction is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"mem_fraction_static": talker_mem_fraction},
        )


def _apply_limit_mm_per_prompt_args(config: Any, args: argparse.Namespace) -> None:
    limit_mm_per_prompt = getattr(args, "limit_mm_per_prompt", None)
    if not limit_mm_per_prompt:
        return
    for stage in config.stages:
        if stage.name != "preprocessing":
            continue
        factory_args = dict(stage.factory_args or {})
        current = dict(factory_args.get("limit_mm_per_prompt") or {})
        current.update(limit_mm_per_prompt)
        factory_args["limit_mm_per_prompt"] = current
        stage.factory_args = factory_args


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    _launch_speech_server(args)


if __name__ == "__main__":
    main()
