# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible thinker-only server for Qwen3.5-Omni.

Use this path first while Qwen3.5 talker/code2wav weights are unavailable. It
keeps the same pipeline shape as Qwen3-Omni text mode, but registers the
Qwen3OmniNext architecture and Qwen3.5-specific stage factories.
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

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QWEN35_DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-Omni"
QWEN35_DEFAULT_MODEL_NAME = QWEN3_5_OMNI_MODEL_NAME
_LIMIT_MM_MODALITIES = frozenset({"audio", "image", "video"})
_MAMBA_CACHE_MODES = frozenset({"none", "light", "all"})
_VLLM_DISTRIBUTED_EXECUTOR_BACKENDS = frozenset({"mp"})
_VLLM_PROFILE_FLAGS = ("--vllm-profile", "--vllm_profile")
_DISABLE_MTP_FLAGS = ("--disable-mtp", "--disable_mtp")
_TEXT_ONLY_PROFILE_SKIP_FLAGS = frozenset(
    {
        "--batched-chunk",
        "--code2wav-batched-chunk",
        "--code2wav-codec-eos-token-id",
        "--code2wav-dit-quantization",
        "--code2wav-dynamic-chunk-sizes",
        "--code2wav-dynamic-chunk-steps",
        "--code2wav-enable-dynamic-chunk",
        "--code2wav-enable-torch-compile",
        "--code2wav-enable-torch-compile-first-chunk",
        "--code2wav-frequency",
        "--code2wav-left-context-size",
        "--code2wav-model-folder",
        "--code2wav-model-path",
        "--code2wav-odeint-method",
        "--code2wav-odeint-method-relaxed",
        "--code2wav-sample-rate",
        "--code2wav-stream-chunk-size",
        "--code2wav-visible-devices",
        "--no-code2wav-dynamic-chunk",
        "--no-code2wav-odeint-method-relaxed",
        "--no-code2wav-torch-compile",
        "--no-code2wav-torch-compile-first-chunk",
        "--odeint-method",
        "--send-chunk-size",
        "--talker-gpu-memory-utilization",
        "--talker-model-path",
        "--talker-visible-devices",
        "--text-only",
    }
)


def parse_args() -> argparse.Namespace:
    argv = _argv_with_vllm_profile_defaults(sys.argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vllm-profile",
        "--vllm_profile",
        default=None,
        help=(
            "Load a vLLM perf_v2 profile as default arguments. Explicit CLI "
            "arguments override profile values."
        ),
    )
    parser.add_argument(
        "--model-path",
        "--model",
        "--thinker-model",
        "--thinker_model",
        dest="model_path",
        type=str,
        default=QWEN35_DEFAULT_MODEL_PATH,
        help="Qwen3.5 root or thinker checkpoint path.",
    )
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument(
        "--thinker-visible-devices",
        "--thinker_visible_devices",
        "--thinker-devices",
        "--thinker_devices",
        type=_parse_visible_devices_arg,
        default=None,
        help=(
            "vLLM-compatible alias for thinker GPU placement. This "
            "thinker-only launcher accepts exactly one GPU; use generic "
            "serve for TP placement."
        ),
    )
    parser.add_argument("--gpu-image-encoder", type=int, default=None)
    parser.add_argument("--gpu-audio-encoder", type=int, default=None)
    parser.add_argument(
        "--thinker-max-seq-len",
        "--thinker_max_seq_len",
        "--max-model-len",
        "--max_model_len",
        type=int,
        default=QWEN3_5_OMNI_THINKER_MAX_SEQ_LEN,
        dest="thinker_max_seq_len",
        help=(
            "Set thinker context length. The max-model-len aliases mirror "
            "vLLM Qwen3.5 configs."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for the thinker stage. "
            "The gpu-memory-utilization aliases mirror vLLM configs."
        ),
    )
    parser.add_argument(
        "--max-seq-len-to-capture",
        "--max_seq_len_to_capture",
        type=int,
        default=None,
        help=(
            "vLLM-compatible no-op. SGLang does not require a separate "
            "Qwen3.5 capture length in this launcher."
        ),
    )
    parser.add_argument(
        "--compilation-config",
        "--compilation_config",
        type=_parse_compilation_config_arg,
        default=None,
        help=(
            "vLLM-compatible compilation_config JSON. FULL_DECODE_ONLY is "
            "accepted as the SGLang default; none/off disables CUDA graph."
        ),
    )
    parser.add_argument(
        "--max-running-requests",
        "--max_running_requests",
        "--max-num-seqs",
        "--max_num_seqs",
        type=int,
        default=None,
        help="Set SGLang max_running_requests for the thinker stage.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help=(
            "Set SGLang max_prefill_tokens for the thinker stage. When chunked "
            "prefill is not disabled, also sets chunked_prefill_size."
        ),
    )
    parser.add_argument(
        "--page-size",
        "--page_size",
        "--block-size",
        "--block_size",
        dest="page_size",
        type=int,
        default=None,
        help=(
            "Set SGLang page_size for the thinker stage. The block-size "
            "aliases mirror vLLM KV cache block_size."
        ),
    )
    parser.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_true",
        default=None,
        help="Enable SGLang radix/prefix cache for the thinker stage.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        "--no-enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="Disable SGLang radix/prefix cache for the thinker stage.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_true",
        default=None,
        help="Use Qwen3.5 H20 chunked prefill size for the thinker stage.",
    )
    parser.add_argument(
        "--disable-chunked-prefill",
        "--no-enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_false",
        help="Disable chunked prefill for the thinker stage.",
    )
    parser.add_argument(
        "--enforce-eager",
        "--thinker-enforce-eager",
        dest="thinker_enforce_eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph for the thinker stage.",
    )
    parser.add_argument(
        "--quantization",
        "--thinker-quantization",
        "--thinker_quantization",
        dest="thinker_quantization",
        type=_parse_quantization_arg,
        default=None,
        help="Set SGLang quantization mode for the thinker stage.",
    )
    parser.add_argument(
        "--dtype",
        "--thinker-dtype",
        "--thinker_dtype",
        dest="thinker_dtype",
        type=_parse_dtype_arg,
        default=None,
        help="Set SGLang dtype for the thinker stage, e.g. bfloat16.",
    )
    parser.add_argument(
        "--mamba-ssm-dtype",
        "--mamba_ssm_dtype",
        "--mamba-cache-dtype",
        "--mamba_cache_dtype",
        dest="mamba_ssm_dtype",
        type=_parse_dtype_arg,
        default=None,
        help=(
            "Set SGLang mamba_ssm_dtype for the thinker stage. The "
            "mamba-cache aliases mirror vLLM Qwen3.5 configs."
        ),
    )
    parser.add_argument(
        "--mamba-cache-mode",
        "--mamba_cache_mode",
        type=_parse_mamba_cache_mode_arg,
        default=None,
        help=(
            "vLLM-compatible Qwen3.5 cache mode. Only none is currently "
            "accepted by this SGLang path; light/all fail early."
        ),
    )
    parser.add_argument(
        "--kv-transfer-config",
        "--kv_transfer_config",
        "--thinker-kv-transfer-config",
        "--thinker_kv_transfer_config",
        type=_parse_json_object_arg,
        default=None,
        help=(
            "vLLM-compatible KV connector config. Empty JSON is accepted as "
            "no-op; non-empty configs fail early until SGLang KV transfer is "
            "mapped explicitly."
        ),
    )
    parser.add_argument(
        "--enable-disaggregated-prefilling",
        "--enable_disaggregated_prefilling",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help=(
            "vLLM-compatible disaggregated prefill flag. false/0 is accepted "
            "as no-op; true/1 fails early."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tensor_parallel_size",
        "--thinker-tensor-parallel-size",
        "--thinker_tensor_parallel_size",
        type=int,
        default=None,
        help=(
            "vLLM-compatible alias for thinker tensor parallel size. "
            "The thinker-only launcher supports only 1 GPU; use generic "
            "serve plus --thinker-visible-devices for TP placement."
        ),
    )
    parser.add_argument(
        "--distributed-executor-backend",
        "--distributed_executor_backend",
        type=str,
        default=None,
        help="vLLM-compatible executor backend. Only mp is accepted as no-op.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        "--kv_cache_dtype",
        type=_parse_dtype_arg,
        default=None,
        help=(
            "vLLM-compatible KV cache dtype. Only auto is accepted as no-op; "
            "quantized KV cache modes fail early."
        ),
    )
    parser.add_argument(
        "--enable-expert-parallel",
        "--enable_expert_parallel",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM-compatible expert parallel flag. true/1 fails early.",
    )
    parser.add_argument(
        "--mm-processor-cache-gb",
        "--mm_processor_cache_gb",
        type=float,
        default=None,
        help="vLLM-compatible MM cache size. Only 0 is accepted as no-op.",
    )
    parser.add_argument(
        "--max-mm-len",
        "--max_mm_len",
        type=int,
        default=None,
        help=(
            "vLLM-compatible multimodal capacity setting. Maps to the "
            "Qwen3.5 preprocessing max_seq_len guard."
        ),
    )
    parser.add_argument(
        "--speculative-config",
        "--speculative_config",
        type=_parse_json_object_arg,
        default=None,
        help=(
            "vLLM-compatible speculative_config JSON. Empty JSON is accepted "
            "as no-op; Qwen3.5 MTP configs fail early until SGLang "
            "speculative decoding is mapped."
        ),
    )
    parser.add_argument(
        "--use-omni-engine",
        "--use_omni_engine",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM Qwen3.5 launcher marker accepted as no-op.",
    )
    parser.add_argument(
        "--use-omni-rpc-engine",
        "--use_omni_rpc_engine",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM Qwen3.5 launcher marker accepted as no-op.",
    )
    parser.add_argument(
        "--is-thinker",
        "--is_thinker",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM Qwen3.5 launcher marker; false is rejected.",
    )
    parser.add_argument(
        "--thinker-only",
        "--thinker_only",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM Qwen3.5 launcher marker; false conflicts with this launcher.",
    )
    parser.add_argument(
        "--use-zero-shot",
        "--use_zero_shot",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM Qwen3.5 launcher marker accepted as no-op.",
    )
    parser.add_argument(
        "--skip-mm-profiling",
        "--skip_mm_profiling",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help="vLLM MM profiling marker accepted as no-op.",
    )
    parser.add_argument(
        "--video-needs-metadata",
        "--video_needs_metadata",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help=(
            "vLLM video metadata marker accepted as no-op. Qwen3.5 "
            "preprocessing already requests video metadata."
        ),
    )
    parser.add_argument(
        "--override-video-max-pixels",
        "--override_video_max_pixels",
        nargs="?",
        const="true",
        type=_parse_bool_arg,
        default=None,
        help=(
            "Use max-mm-len-derived video total_pixels as the effective "
            "Qwen3.5 video resize budget."
        ),
    )
    parser.add_argument(
        "--disable-mtp",
        "--disable_mtp",
        action="store_true",
        help=(
            "vLLM-compatible no-op. The current SGLang Qwen3.5 path does not "
            "enable Qwen3.5 thinker MTP."
        ),
    )
    parser.add_argument("--video-fps", "--video_fps", type=float, default=None)
    parser.add_argument(
        "--image-min-pixels",
        "--image_min_pixels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--image-max-pixels",
        "--image_max_pixels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-max-frames",
        "--video_max_frames",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-min-frames",
        "--video_min_frames",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-min-pixels",
        "--video_min_pixels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-max-pixels",
        "--video_max_pixels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-total-pixels",
        "--video_total_pixels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--video-seconds-per-chunk",
        "--video_seconds_per_chunk",
        "--seconds-per-chunk",
        "--seconds_per_chunk",
        dest="video_seconds_per_chunk",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--video-position-id-per-seconds",
        "--video_position_id_per_seconds",
        "--position-id-per-seconds",
        "--position_id_per_seconds",
        dest="video_position_id_per_seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--audio-target-sr",
        "--audio_target_sr",
        "--audio-sampling-rate",
        "--audio_sampling_rate",
        "--sampling-rate",
        "--sampling_rate",
        dest="audio_target_sr",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--audio-timestamp-interval",
        "--audio_timestamp_interval",
        "--timestamp-interval",
        "--timestamp_interval",
        dest="audio_timestamp_interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--audio-downsample-times",
        "--audio_downsample_times",
        "--downsample-times",
        "--downsample_times",
        dest="audio_downsample_times",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--audio-downsample-chunk-size",
        "--audio_downsample_chunk_size",
        "--downsample-chunk-size",
        "--downsample_chunk_size",
        dest="audio_downsample_chunk_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        "--limit_mm_per_prompt",
        type=_parse_limit_mm_per_prompt_arg,
        default=None,
        help="Set Qwen3.5 multimodal count limits, e.g. '{\"image\":2}'.",
    )
    parser.add_argument(
        "--limit-mm-per-prompt-image",
        "--limit_mm_per_prompt_image",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-mm-per-prompt-video",
        "--limit_mm_per_prompt_video",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-mm-per-prompt-audio",
        "--limit_mm_per_prompt_audio",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=["nixl", "shm"],
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port",
        "--serve-port",
        "--serve_port",
        type=int,
        default=8000,
    )
    parser.add_argument("--model-name", type=str, default=QWEN35_DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run local Qwen3.5 checkpoint preflight before launching.",
    )
    args = parser.parse_args(argv[1:])
    args.model_name = normalize_qwen35_omni_model_name(args.model_name)
    return args


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

    profile_args = _filter_vllm_profile_args_for_text(
        suggested_vllm_profile_cli_args(report)
    )
    # 中文说明：profile 只提供默认值，用户显式 CLI 参数放在后面覆盖它。
    return [cleaned[0], *profile_args, *cleaned[1:]]


def _has_any_explicit_flag(argv: list[str], flag_names: tuple[str, ...]) -> bool:
    return any(
        arg == flag_name or arg.startswith(f"{flag_name}=")
        for arg in argv[1:]
        for flag_name in flag_names
    )


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
    # 里的非 MTP 默认值；preflight 仍会拦住 KV/PD 等真实未接入能力。
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


def _filter_vllm_profile_args_for_text(args: tuple[str, ...]) -> list[str]:
    filtered: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        flag = arg.split("=", 1)[0]
        if flag == "--thinker-gpu-memory-utilization":
            filtered.append("--gpu-memory-utilization")
            if "=" in arg:
                filtered.append(arg.split("=", 1)[1])
                idx += 1
            elif idx + 1 < len(args):
                filtered.append(args[idx + 1])
                idx += 2
            else:
                idx += 1
            continue
        if flag in _TEXT_ONLY_PROFILE_SKIP_FLAGS:
            idx += 1
            if "=" not in arg and idx < len(args) and not args[idx].startswith("--"):
                idx += 1
            continue
        filtered.append(arg)
        idx += 1
    return filtered


def _validate_fraction(flag_name: str, value: float | None) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")


def _set_stage_gpu(config: Any, stage_name: str, gpu: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = gpu


def _apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args


def _set_stage_max_running_requests(
    config: Any,
    *,
    stage_name: str,
    value: int,
) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.runtime.sglang_server_args.max_running_requests = int(value)


def _validate_positive_int(flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) < 1:
        raise ValueError(f"{flag_name} must be >= 1, got {value}")
    return int(value)


def _validate_positive_float(flag_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if float(value) <= 0:
        raise ValueError(f"{flag_name} must be > 0, got {value}")
    return float(value)


def _validate_nonnegative_int(flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) < 0:
        raise ValueError(f"{flag_name} must be >= 0, got {value}")
    return int(value)


def _parse_dtype_arg(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("dtype must not be empty")
    return value


def _parse_quantization_arg(raw: str) -> str:
    value = raw.strip().lower()
    if value not in {"none", "fp8", "nvfp4"}:
        raise argparse.ArgumentTypeError("quantization must be one of: fp8, none, nvfp4")
    return value


def _parse_mamba_cache_mode_arg(raw: str) -> str:
    value = raw.strip().lower()
    if value not in _MAMBA_CACHE_MODES:
        supported = ", ".join(sorted(_MAMBA_CACHE_MODES))
        raise argparse.ArgumentTypeError(
            f"mamba_cache_mode must be one of: {supported}"
        )
    return value


def _parse_bool_arg(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {raw!r}")


def _coerce_bool_value(flag_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(value)
    # 中文说明：_launch_server 既会收到 argparse 解析出的 bool，也可能在
    # 单测/YAML/profile 适配层里直接收到字符串。这里显式解析字符串，避免
    # bool("false") 把关闭开关误当成开启。
    try:
        return _parse_bool_arg(str(value))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"{flag_name} {exc}") from exc


def _parse_json_object_arg(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "expected a JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def _parse_visible_devices_arg(raw: str) -> tuple[int, ...]:
    text = raw.strip()
    if not text:
        raise argparse.ArgumentTypeError("visible devices must not be empty")
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                "visible devices must be an int or list of ints"
            ) from exc
    else:
        parsed = [piece.strip() for piece in text.replace(",", " ").split()]
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError(
            "visible devices must be an int or non-empty list"
        )
    devices = tuple(int(item) for item in parsed)
    if any(gpu < 0 for gpu in devices):
        raise argparse.ArgumentTypeError("visible device GPU ids must be >= 0")
    return devices


def _parse_compilation_config_arg(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "compilation_config must be a JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("compilation_config must be a JSON object")
    return value


def _parse_limit_mm_per_prompt_arg(raw: str) -> dict[str, int]:
    raw = raw.strip()
    if not raw:
        raise ValueError("--limit-mm-per-prompt must not be empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = _parse_limit_mm_pairs(raw)
    if not isinstance(value, dict):
        raise ValueError("--limit-mm-per-prompt expects a JSON object")
    return _normalize_limit_mm_per_prompt(value)


def _parse_limit_mm_pairs(raw: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    pieces = [
        piece.strip()
        for piece in raw.replace(",", " ").split()
        if piece.strip()
    ]
    for piece in pieces:
        if "=" not in piece:
            raise ValueError(
                "--limit-mm-per-prompt expects JSON or modality=count pairs"
            )
        key, raw_value = piece.split("=", 1)
        parsed[key.strip()] = int(raw_value)
    return parsed


def _normalize_limit_mm_per_prompt(value: dict[object, object]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_key, raw_limit in value.items():
        modality = str(raw_key).strip().lower()
        if modality.endswith("s"):
            modality = modality[:-1]
        if modality not in _LIMIT_MM_MODALITIES:
            supported = ", ".join(sorted(_LIMIT_MM_MODALITIES))
            raise ValueError(
                f"--limit-mm-per-prompt modality must be one of: {supported}"
            )
        limit = int(raw_limit)
        if limit < 0:
            raise ValueError("--limit-mm-per-prompt values must be >= 0")
        normalized[modality] = limit
    return normalized


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
    _validate_vllm_engine_profile_args(args, speech=False)

    updates: dict[str, object] = {}
    if getattr(args, "enable_prefix_caching", None) is not None:
        # 中文说明：vLLM 的 enable_prefix_caching 对应 SGLang radix cache；
        # SGLang ServerArgs 使用反向开关 disable_radix_cache。
        updates["disable_radix_cache"] = not _coerce_bool_value(
            "--enable-prefix-caching",
            args.enable_prefix_caching,
        )
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    if max_num_batched_tokens is not None:
        max_num_batched_tokens = _validate_positive_int(
            "--max-num-batched-tokens",
            max_num_batched_tokens,
        )
        # 中文说明：vLLM 的 max_num_batched_tokens 对齐 SGLang
        # max_prefill_tokens；chunked prefill 开启时同步调整 chunk 大小。
        updates["max_prefill_tokens"] = max_num_batched_tokens
        updates["chunked_prefill_size"] = max_num_batched_tokens
    if getattr(args, "enable_chunked_prefill", None) is not None:
        updates["chunked_prefill_size"] = (
            max_num_batched_tokens or QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE
            if _coerce_bool_value(
                "--enable-chunked-prefill",
                args.enable_chunked_prefill,
            )
            else None
        )
    if _coerce_bool_value(
        "--thinker-enforce-eager",
        getattr(args, "thinker_enforce_eager", False),
    ):
        updates["disable_cuda_graph"] = True
    if (
        getattr(args, "thinker_quantization", None) is not None
        and args.thinker_quantization != "none"
    ):
        updates["quantization"] = args.thinker_quantization
    if getattr(args, "thinker_dtype", None) is not None:
        # 中文说明：vLLM perf_v2 configs 常设置 dtype=bfloat16；
        # thinker-only 路径直接写入 SGLang ServerArgs.dtype。
        updates["dtype"] = args.thinker_dtype
    if getattr(args, "mamba_ssm_dtype", None) is not None:
        # 中文说明：vLLM 的 mamba_cache_dtype 对应 SGLang core 的
        # mamba_ssm_dtype；默认 float32，这里保留显式覆盖能力。
        updates["mamba_ssm_dtype"] = args.mamba_ssm_dtype
    if getattr(args, "page_size", None) is not None:
        # 中文说明：vLLM block_size 对应 SGLang page_size，控制 KV cache
        # 分页大小；保留 --page-size 作为更贴近 SGLang 的名字。
        updates["page_size"] = _validate_positive_int(
            "--block-size",
            args.page_size,
        )
    updates.update(
        _server_args_from_vllm_compilation_config(
            getattr(args, "compilation_config", None)
        )
    )
    if updates:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=updates,
        )


def _server_args_from_vllm_compilation_config(
    compilation_config: dict[str, Any] | None,
) -> dict[str, object]:
    if compilation_config is None:
        return {}
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
    # 中文说明：vLLM 的 light/all 是混合 Mamba cache 策略；当前
    # SGLang ServerArgs 没有等价 mamba_cache_mode 字段，因此不能静默忽略。
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
        # 中文说明：vLLM 的分离式 prefill/decode 依赖 KV connector
        # producer/consumer 协议。当前 SGLang Qwen3.5 路径尚未显式映射。
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
        if int(tensor_parallel_size) != 1:
            raise ValueError(
                "--tensor-parallel-size > 1 requires generic serve with "
                "--thinker-visible-devices for SGLang TP placement"
            )

    backend = getattr(args, "distributed_executor_backend", None)
    if backend is not None and backend.strip().lower() not in (
        _VLLM_DISTRIBUTED_EXECUTOR_BACKENDS
    ):
        raise ValueError(
            "--distributed-executor-backend currently supports only the vLLM "
            "mp profile value in SGLang Qwen3.5 launchers"
        )

    kv_cache_dtype = getattr(args, "kv_cache_dtype", None)
    if kv_cache_dtype is not None and kv_cache_dtype.strip().lower() != "auto":
        # 中文说明：vLLM 的 tq4/fp8 KV cache dtype 会影响 KV cache 存储和
        # attention backend；当前 SGLang Qwen3.5 没有等价映射。
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
        raise ValueError("--mm-processor-cache-gb currently supports only 0")
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
        max_mm_len = _validate_positive_int("--max-mm-len", int(max_mm_len))
        if int(max_mm_len) > int(args.thinker_max_seq_len):
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


def _apply_video_preprocessing_runtime_args(
    config: Any,
    args: argparse.Namespace,
) -> None:
    updates = {
        "max_seq_len": _validate_positive_int(
            "--max-mm-len",
            args.max_mm_len,
        ),
        "image_min_pixels": _validate_positive_int(
            "--image-min-pixels",
            args.image_min_pixels,
        ),
        "image_max_pixels": _validate_positive_int(
            "--image-max-pixels",
            args.image_max_pixels,
        ),
        "video_fps": _validate_positive_float("--video-fps", args.video_fps),
        "video_max_frames": _validate_positive_int(
            "--video-max-frames",
            args.video_max_frames,
        ),
        "video_min_frames": _validate_positive_int(
            "--video-min-frames",
            args.video_min_frames,
        ),
        "video_min_pixels": _validate_positive_int(
            "--video-min-pixels",
            args.video_min_pixels,
        ),
        "video_max_pixels": _validate_positive_int(
            "--video-max-pixels",
            args.video_max_pixels,
        ),
        "video_total_pixels": _validate_positive_int(
            "--video-total-pixels",
            args.video_total_pixels,
        ),
        "video_override_max_pixels": (
            _coerce_bool_value(
                "--override-video-max-pixels",
                args.override_video_max_pixels,
            )
            if args.override_video_max_pixels is not None
            else None
        ),
        "video_seconds_per_chunk": _validate_positive_float(
            "--video-seconds-per-chunk",
            args.video_seconds_per_chunk,
        ),
        "video_position_id_per_seconds": _validate_positive_float(
            "--video-position-id-per-seconds",
            args.video_position_id_per_seconds,
        ),
        "audio_target_sr": _validate_positive_int(
            "--audio-target-sr",
            args.audio_target_sr,
        ),
        "audio_timestamp_interval": _validate_positive_int(
            "--audio-timestamp-interval",
            args.audio_timestamp_interval,
        ),
        "audio_downsample_times": _validate_nonnegative_int(
            "--audio-downsample-times",
            args.audio_downsample_times,
        ),
        "audio_downsample_chunk_size": _validate_positive_int(
            "--audio-downsample-chunk-size",
            args.audio_downsample_chunk_size,
        ),
    }
    updates = {key: value for key, value in updates.items() if value is not None}
    if not updates:
        return
    for stage in config.stages:
        if stage.name != "preprocessing":
            continue
        # 中文说明：thinker-only 模式也常用于视觉理解压测；这里配置的是
        # 服务级默认视觉采样/像素限制，不改变每个请求可显式覆盖的能力。
        for key, value in updates.items():
            setattr(stage.runtime, key, value)


def _run_preflight_or_raise(
    *,
    model_path: str,
    speech: bool,
    code2wav_model_path: str | None = None,
) -> None:
    from sglang_omni.models.qwen3_5_omni.preflight import (
        format_preflight_report,
        run_qwen35_preflight,
    )

    report = run_qwen35_preflight(
        model_path,
        speech=speech,
        code2wav_model_path=code2wav_model_path,
    )
    message = format_preflight_report(report)
    if not report.ok:
        raise RuntimeError(message)
    logger.info("%s", message)


def _launch_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_5_omni.config import Qwen35OmniPipelineConfig
    from sglang_omni.serve import launch_server

    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    if args.thinker_visible_devices is not None:
        if len(args.thinker_visible_devices) != 1:
            raise ValueError(
                "--thinker-visible-devices supports exactly one GPU in the "
                "thinker-only launcher; use `python -m sglang_omni.cli serve` "
                "with --thinker-visible-devices for TP."
            )
        args.gpu_thinker = int(args.thinker_visible_devices[0])
    _validate_positive_int(
        "--max-seq-len-to-capture",
        args.max_seq_len_to_capture,
    )
    if _coerce_bool_value("--preflight", getattr(args, "preflight", False)):
        _run_preflight_or_raise(
            model_path=args.model_path,
            speech=False,
        )
    config = Qwen35OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    gpu_image_encoder = (
        args.gpu_image_encoder
        if args.gpu_image_encoder is not None
        else args.gpu_thinker
    )
    gpu_audio_encoder = (
        args.gpu_audio_encoder
        if args.gpu_audio_encoder is not None
        else args.gpu_thinker
    )

    _set_stage_gpu(config, "image_encoder", gpu_image_encoder)
    _set_stage_gpu(config, "audio_encoder", gpu_audio_encoder)
    _set_stage_gpu(config, "thinker", args.gpu_thinker)

    thinker_seq_len_updates = {
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
    if args.mem_fraction_static is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"mem_fraction_static": args.mem_fraction_static},
        )
    if args.max_running_requests is not None:
        max_running = _validate_positive_int(
            "--max-running-requests",
            args.max_running_requests,
        )
        _set_stage_max_running_requests(
            config,
            stage_name="thinker",
            value=max_running,
        )
    _apply_vllm_ar_server_args(config, args)

    # 中文说明：thinker-only 入口不启动 talker/code2wav，适合作为 Qwen3.5
    # 模型未齐全时的第一条 bring-up 路径。
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def _apply_limit_mm_per_prompt_args(config: Any, args: argparse.Namespace) -> None:
    updates = dict(getattr(args, "limit_mm_per_prompt", None) or {})
    for modality, value, flag_name in (
        ("image", args.limit_mm_per_prompt_image, "--limit-mm-per-prompt-image"),
        ("video", args.limit_mm_per_prompt_video, "--limit-mm-per-prompt-video"),
        ("audio", args.limit_mm_per_prompt_audio, "--limit-mm-per-prompt-audio"),
    ):
        value = _validate_nonnegative_int(flag_name, value)
        if value is not None:
            updates[modality] = value
    if not updates:
        return
    for stage in config.stages:
        if stage.name != "preprocessing":
            continue
        factory_args = dict(stage.factory_args or {})
        current = dict(factory_args.get("limit_mm_per_prompt") or {})
        current.update(updates)
        factory_args["limit_mm_per_prompt"] = current
        stage.factory_args = factory_args


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    _launch_server(args)


if __name__ == "__main__":
    main()
