from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Literal, NoReturn

import typer
import yaml

from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager

logger = logging.getLogger(__name__)


_STAGE_TOGGLE_MODE = Literal["default", "on", "off"]
_QWEN_COLOCATED_CONFIG_CLASSES = {
    "Qwen3OmniSpeechColocatedPipelineConfig",
    "Qwen35OmniSpeechColocatedPipelineConfig",
}
_QWEN35_CONFIG_CLASSES = {
    "Qwen35OmniPipelineConfig",
    "Qwen35OmniSpeechPipelineConfig",
    "Qwen35OmniSpeechColocatedPipelineConfig",
}
_QWEN35_SPEECH_CONFIG_CLASSES = {
    "Qwen35OmniSpeechPipelineConfig",
    "Qwen35OmniSpeechColocatedPipelineConfig",
}
_QWEN35_DEFAULT_VOICE_TYPE = "f245"
_QWEN35_DEFAULT_MAX_TOKENS = 2048
_QWEN35_DEFAULT_SEED = 0
_QWEN35_DEFAULT_TEMPERATURE = 0.000001
_QWEN35_DEFAULT_TOP_K = 1
_QWEN35_DEFAULT_TOP_P = 0.8
_QWEN35_DEFAULT_REPETITION_PENALTY = 1.0
_QWEN35_DEFAULT_PRESENCE_PENALTY = 0.0
_HIGGS_ASYNC_DECODE_FACTORY = (
    "sglang_omni.models.higgs_tts.stages.create_sglang_tts_engine_executor"
)
_QWEN_PARTIAL_START_TALKER_FACTORIES = {
    "sglang_omni.models.qwen3_omni.stages.create_talker_ar_executor_from_config",
    "sglang_omni.models.qwen3_5_omni.stages.create_talker_ar_executor_from_config",
}
_QWEN_TALKER_MODEL_PATH_FACTORIES = {
    "sglang_omni.models.qwen3_5_omni.stages.create_talker_ar_executor_from_config",
}
_QWEN_PREPROCESSING_VIDEO_FACTORIES = {
    "sglang_omni.models.qwen3_omni.stages.create_preprocessing_executor",
    "sglang_omni.models.qwen3_5_omni.stages.create_preprocessing_executor",
}
_QWEN35_PREPROCESSING_FACTORY = (
    "sglang_omni.models.qwen3_5_omni.stages.create_preprocessing_executor"
)
_LIMIT_MM_MODALITIES = frozenset({"audio", "image", "video"})
_QWEN35_CODE2WAV_ODEINT_METHODS = frozenset({"euler", "rk4"})
_QWEN35_CODE2WAV_FREQUENCIES = frozenset({"50hz", "25hz"})
_QWEN35_CODE2WAV_DIT_QUANTS = frozenset({"fp8"})
_QWEN35_MAMBA_CACHE_MODES = frozenset({"none", "light", "all"})
_QWEN35_VLLM_EXECUTOR_BACKENDS = frozenset({"mp"})
_QWEN35_SPECULATIVE_MTP_METHODS = frozenset({"qwen3_omni_next_thinker_mtp"})
_QWEN35_VLLM_PROFILE_BOOL_FLAGS: dict[str, tuple[str, object]] = {
    "--text-only": ("text_only", True),
    "--disable-mtp": ("disable_mtp", True),
    "--enforce-eager": ("enforce_eager", True),
    "--enable-prefix-caching": ("enable_prefix_caching", True),
    "--disable-prefix-caching": ("disable_prefix_caching", True),
    "--enable-chunked-prefill": ("enable_chunked_prefill", True),
    "--no-enable-chunked-prefill": ("disable_chunked_prefill", True),
    "--code2wav-enable-torch-compile": ("code2wav_enable_torch_compile", True),
    "--no-code2wav-torch-compile": ("code2wav_disable_torch_compile", True),
    "--no-code2wav-dynamic-chunk": ("code2wav_disable_dynamic_chunk", True),
    "--code2wav-enable-dynamic-chunk": ("code2wav_enable_dynamic_chunk", True),
    "--code2wav-enable-torch-compile-first-chunk": (
        "code2wav_enable_torch_compile_first_chunk",
        True,
    ),
    "--no-code2wav-torch-compile-first-chunk": (
        "code2wav_disable_torch_compile_first_chunk",
        True,
    ),
    "--code2wav-odeint-method-relaxed": (
        "code2wav_enable_odeint_method_relaxed",
        True,
    ),
    "--no-code2wav-odeint-method-relaxed": (
        "code2wav_disable_odeint_method_relaxed",
        True,
    ),
    "--enable-tn": ("enable_tn", True),
    "--enable-text-normalization": ("enable_tn", True),
    "--disable-tn": ("disable_tn", True),
    "--no-enable-tn": ("disable_tn", True),
    "--disable-text-normalization": ("disable_tn", True),
}
_QWEN35_VLLM_PROFILE_VALUE_FLAGS: dict[str, tuple[str, type]] = {
    "--model": ("model_path", str),
    "--model-path": ("model_path", str),
    "--thinker-model": ("thinker_model_path", str),
    "--thinker-model-path": ("thinker_model_path", str),
    "--talker-model": ("talker_model_path", str),
    "--talker-model-path": ("talker_model_path", str),
    "--code2wav-model": ("code2wav_model_folder", str),
    "--code2wav-model-path": ("code2wav_model_path", str),
    "--code2wav-model-folder": ("code2wav_model_folder", str),
    "--gpu-memory-utilization": ("mem_fraction_static", float),
    "--thinker-gpu-memory-utilization": ("thinker_mem_fraction_static", float),
    "--talker-gpu-memory-utilization": ("talker_mem_fraction_static", float),
    "--dtype": ("dtype", str),
    "--mamba-cache-dtype": ("mamba_ssm_dtype", str),
    "--thinker-quantization": ("quantization", str),
    "--talker-quantization": ("talker_quantization", str),
    "--max-model-len": ("max_model_len", int),
    "--max-mm-len": ("max_mm_len", int),
    "--max-num-batched-tokens": ("max_num_batched_tokens", int),
    "--max-num-seqs": ("max_running_requests", int),
    "--max-seq-len-to-capture": ("max_seq_len_to_capture", int),
    "--max-tokens": ("max_tokens", int),
    "--seed": ("seed", int),
    "--send-chunk-size": ("code2wav_stream_chunk_size", int),
    "--host": ("host", str),
    "--port": ("port", int),
    "--serve-port": ("port", int),
    "--serve_port": ("port", int),
    "--block-size": ("page_size", int),
    "--page-size": ("page_size", int),
    "--thinker-devices": ("thinker_visible_devices", str),
    "--thinker-visible-devices": ("thinker_visible_devices", str),
    "--talker-devices": ("talker_visible_devices", str),
    "--talker-visible-devices": ("talker_visible_devices", str),
    "--code2wav-devices": ("code2wav_visible_devices", str),
    "--code2wav-visible-devices": ("code2wav_visible_devices", str),
    "--thinker-tp-size": ("thinker_tp_size", int),
    "--thinker-tensor-parallel-size": ("thinker_tp_size", int),
    "--thinker_tensor_parallel_size": ("thinker_tp_size", int),
    "--thinker-only": ("thinker_only_marker", str),
    "--limit-mm-per-prompt": ("limit_mm_per_prompt", str),
    "--code2wav-codec-eos-token-id": ("code2wav_codec_eos_token_id", int),
    "--code2wav-left-context-size": ("code2wav_left_context_size", int),
    "--code2wav-sample-rate": ("code2wav_sample_rate", int),
    "--code2wav-dynamic-chunk-sizes": ("code2wav_dynamic_chunk_sizes", str),
    "--code2wav-dynamic-chunk-steps": ("code2wav_dynamic_chunk_steps", str),
    "--code2wav-odeint-method": ("code2wav_odeint_method", str),
    "--code2wav-batched-chunk": ("code2wav_batched_chunk", int),
    "--code2wav-frequency": ("code2wav_frequency", str),
    "--code2wav-dit-quantization": ("code2wav_dit_quant", str),
    "--sample-rate": ("code2wav_sample_rate", int),
    "--video-fps": ("video_fps", float),
    "--voice-type": ("voice_type", str),
    "--compilation-config": ("compilation_config", str),
    "--override-video-max-pixels": ("override_video_max_pixels", str),
}
_QWEN35_TEXT_ONLY_PROFILE_SKIP_DEFAULT_KEYS = frozenset(
    {
        "talker_mem_fraction_static",
        "talker_visible_devices",
        "talker_model_path",
        "code2wav_visible_devices",
        "code2wav_model_path",
        "code2wav_model_folder",
        "code2wav_enable_torch_compile",
        "code2wav_disable_torch_compile",
        "code2wav_enable_torch_compile_first_chunk",
        "code2wav_disable_torch_compile_first_chunk",
        "code2wav_sample_rate",
        "code2wav_stream_chunk_size",
        "code2wav_left_context_size",
        "code2wav_codec_eos_token_id",
        "code2wav_enable_dynamic_chunk",
        "code2wav_disable_dynamic_chunk",
        "code2wav_dynamic_chunk_sizes",
        "code2wav_dynamic_chunk_steps",
        "code2wav_odeint_method",
        "code2wav_enable_odeint_method_relaxed",
        "code2wav_disable_odeint_method_relaxed",
        "code2wav_batched_chunk",
        "code2wav_frequency",
        "code2wav_dit_quant",
        "voice_type",
        "enable_tn",
        "disable_tn",
    }
)


def launch_server(*args: object, **kwargs: object) -> object:
    from sglang_omni.serve.launcher import launch_server as _launch_server

    return _launch_server(*args, **kwargs)


def _normalize_stage_toggle_mode(flag_name: str, value: str) -> _STAGE_TOGGLE_MODE:
    normalized = value.strip().lower()
    if normalized not in {"default", "on", "off"}:
        raise typer.BadParameter(f"{flag_name} must be one of: default, on, off")
    return normalized  # type: ignore[return-value]


def _normalize_dtype_cli_value(flag_name: str, value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise typer.BadParameter(f"{flag_name} must not be empty")
    return normalized


def _validate_colocate_cli_request(
    *,
    colocate: bool,
    config: str | None,
    text_only: bool,
) -> None:
    if not colocate:
        return
    if text_only:
        raise typer.BadParameter("--colocate cannot be combined with --text-only")
    if not config:
        raise typer.BadParameter("--colocate requires --config")


def _validate_colocate_config(pipeline_config: PipelineConfig) -> None:
    if type(pipeline_config).__name__ not in _QWEN_COLOCATED_CONFIG_CLASSES:
        raise typer.BadParameter(
            "--colocate requires one of: "
            f"{', '.join(sorted(_QWEN_COLOCATED_CONFIG_CLASSES))}"
        )


def _should_print_merged_config(*, colocate: bool, log_level: str) -> bool:
    """Return whether to print the full resolved pipeline config."""

    return colocate or log_level.lower() == "debug"


def _print_merged_config(pipeline_config: PipelineConfig) -> None:
    print("=" * 20, "Merged Configuration", "=" * 20)
    print(
        yaml.dump(
            pipeline_config.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )
    )
    print("=" * 50)


def _is_qwen35_config(pipeline_config: PipelineConfig) -> bool:
    return type(pipeline_config).__name__ in _QWEN35_CONFIG_CLASSES


def _is_qwen35_speech_config(pipeline_config: PipelineConfig) -> bool:
    return type(pipeline_config).__name__ in _QWEN35_SPEECH_CONFIG_CLASSES


def _load_qwen35_vllm_profile_defaults(
    profile_path: str | None,
    *,
    disable_mtp_override: bool = False,
) -> dict[str, object]:
    if profile_path is None:
        return {}

    from sglang_omni.models.qwen3_5_omni import preflight as qwen35_preflight

    payload = qwen35_preflight.load_vllm_profile_payload(profile_path)
    if disable_mtp_override:
        payload = _qwen35_profile_payload_with_disable_mtp(payload)
    report = qwen35_preflight.run_vllm_profile_preflight(
        payload,
        source=profile_path,
    )
    if not report.ok:
        raise typer.BadParameter(
            "--vllm-profile contains settings this launcher cannot map yet:\n"
            + qwen35_preflight.format_vllm_profile_report(report)
        )
    return _parse_qwen35_vllm_profile_defaults(
        qwen35_preflight.suggested_vllm_profile_cli_args(report)
    )


def _qwen35_profile_payload_with_disable_mtp(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("engine_args"), dict):
        normalized = dict(payload)
        engine_args = dict(payload["engine_args"])
        engine_args["disable_mtp"] = True
        normalized["engine_args"] = engine_args
        return normalized
    for key, value in payload.items():
        if not isinstance(value, dict) or not isinstance(value.get("engine_args"), dict):
            continue
        normalized = dict(payload)
        nested = dict(value)
        engine_args = dict(value["engine_args"])
        engine_args["disable_mtp"] = True
        nested["engine_args"] = engine_args
        normalized[key] = nested
        return normalized
    # 中文说明：少数 profile 直接就是 engine_args 对象；这里追加
    # disable_mtp，让通用 serve 和 example launcher 对 MTP profile 行为一致。
    normalized = dict(payload)
    normalized["disable_mtp"] = True
    return normalized


def _parse_qwen35_vllm_profile_defaults(args: tuple[str, ...]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    idx = 0
    while idx < len(args):
        raw_flag = str(args[idx])
        flag, inline_value = (
            raw_flag.split("=", 1) if "=" in raw_flag else (raw_flag, None)
        )
        bool_target = _QWEN35_VLLM_PROFILE_BOOL_FLAGS.get(flag)
        if bool_target is not None:
            key, value = bool_target
            defaults[key] = value
            idx += 1
            continue

        value_target = _QWEN35_VLLM_PROFILE_VALUE_FLAGS.get(flag)
        if value_target is None:
            raise typer.BadParameter(
                f"--vllm-profile suggested unsupported serve arg {flag!r}"
            )
        key, caster = value_target
        if inline_value is not None:
            raw_value = inline_value
            idx += 1
        else:
            if idx + 1 >= len(args):
                raise typer.BadParameter(
                    f"--vllm-profile suggested {flag} without a value"
                )
            raw_value = str(args[idx + 1])
            idx += 2
        try:
            defaults[key] = caster(raw_value)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter(
                f"--vllm-profile suggested invalid value for {flag}: {raw_value!r}"
            ) from exc
    return defaults


def _ctx_param_was_supplied(ctx: typer.Context, name: str) -> bool:
    getter = getattr(ctx, "get_parameter_source", None)
    if getter is None:
        return False
    try:
        source = getter(name)
    except Exception:
        return False
    if source is None:
        return False
    # click.core.ParameterSource.DEFAULT is the only source that should lose to
    # profile defaults. COMMANDLINE/env/default-map values are explicit enough.
    return not str(source).endswith(".DEFAULT")


def _qwen35_profile_default(
    ctx: typer.Context,
    defaults: dict[str, object],
    name: str,
    current: object,
    default: object,
) -> object:
    if name not in defaults:
        return current
    if _ctx_param_was_supplied(ctx, name):
        return current
    if current != default:
        return current
    return defaults[name]


def _resolve_enable_tn_default(enable_tn: bool, disable_tn: bool) -> bool | None:
    if enable_tn and disable_tn:
        raise typer.BadParameter("--enable-tn cannot be combined with --disable-tn")
    if enable_tn:
        return True
    if disable_tn:
        return False
    return None


def _qwen35_thinker_model_matches_root(
    root_model_path: str,
    thinker_model_path: str,
) -> bool:
    root = os.path.normpath(root_model_path)
    thinker = os.path.normpath(thinker_model_path)
    return thinker == os.path.join(root, "thinker")


def _resolve_qwen35_thinker_model_alias(
    *,
    config: str | None,
    model_path: str | None,
    thinker_model_path: str | None,
) -> str | None:
    if thinker_model_path is None:
        return model_path

    normalized_thinker_path = thinker_model_path.strip()
    if not normalized_thinker_path:
        raise typer.BadParameter("--thinker-model must not be empty")

    if model_path is None:
        if config is not None:
            raise typer.BadParameter(
                "--thinker-model with --config also requires --model-path"
            )
        return normalized_thinker_path

    normalized_model_path = model_path.strip()
    if _qwen35_thinker_model_matches_root(
        normalized_model_path,
        normalized_thinker_path,
    ):
        return model_path

    raise typer.BadParameter(
        "--thinker-model can only be used without --model-path, or as "
        "<model-path>/thinker when a root --model-path is provided"
    )


def _resolve_stage_factory_arg(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    key: str,
) -> object | None:
    for stage in pipeline_config.stages:
        if stage.name == stage_name:
            return dict(stage.factory_args or {}).get(key)
    return None


def _resolve_code2wav_model_folder(
    *,
    model_path: str,
    code2wav_model_folder: str,
    flag_name: str,
) -> str:
    folder = code2wav_model_folder.strip()
    if not folder:
        raise typer.BadParameter(f"{flag_name} must not be empty")
    if os.path.isabs(folder):
        return folder
    # 中文说明：vLLM perf_v2 配置常写 root checkpoint 下的 code2wav
    # 子目录名；SGLang stage factory 需要完整目录，所以这里统一补齐。
    return os.path.join(model_path, folder)


def _resolve_preflight_code2wav_model_path(
    pipeline_config: PipelineConfig,
    *,
    code2wav_model_path: str | None,
) -> str | None:
    if code2wav_model_path is not None:
        return code2wav_model_path

    stage_name = type(pipeline_config).code2wav_stage()
    if stage_name is None:
        return None
    value = _resolve_stage_factory_arg(
        pipeline_config,
        stage_name=stage_name,
        key="code2wav_model_path",
    )
    return str(value) if value else None


def _run_qwen35_preflight_or_raise(
    pipeline_config: PipelineConfig,
    *,
    code2wav_model_path: str | None,
    xvector_info_paths: tuple[str, ...] = (),
    validate_xvector_pickle: bool = False,
) -> None:
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--preflight currently supports only Qwen3.5-Omni configs; got "
            f"{type(pipeline_config).__name__}"
        )

    from sglang_omni.models.qwen3_5_omni import preflight as qwen35_preflight

    code2wav_path = _resolve_preflight_code2wav_model_path(
        pipeline_config,
        code2wav_model_path=code2wav_model_path,
    )
    speech = type(pipeline_config).code2wav_stage() is not None
    extra_kwargs: dict[str, object] = {}
    if xvector_info_paths:
        extra_kwargs["xvector_info_paths"] = xvector_info_paths
    if validate_xvector_pickle:
        extra_kwargs["validate_xvector_pickle"] = True
    report = qwen35_preflight.run_qwen35_preflight(
        pipeline_config.model_path,
        speech=speech,
        code2wav_model_path=code2wav_path,
        **extra_kwargs,
    )
    message = qwen35_preflight.format_preflight_report(report)
    if not report.ok:
        raise typer.BadParameter(message)
    logger.info("%s", message)


def _validate_qwen35_disable_mtp_request(
    pipeline_config: PipelineConfig,
    *,
    disable_mtp: bool,
) -> None:
    if not disable_mtp:
        return
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--disable-mtp currently applies only to Qwen3.5-Omni configs"
        )
    # 中文说明：vLLM Qwen3.5 默认可能启用 thinker MTP；当前 SGLang
    # Qwen3.5 先跑 base thinker AR 主链路，没有开启 MTP。接受这个 flag
    # 是为了迁移 vLLM 启动命令时不因未知参数失败。
    logger.info(
        "--disable-mtp accepted: Qwen3.5 thinker MTP is not enabled in this "
        "SGLang path."
    )


def _parse_qwen35_json_object(flag_name: str, value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{flag_name} expects a JSON object") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{flag_name} expects a JSON object")
    return parsed


def _validate_qwen35_speculative_config_request(
    pipeline_config: PipelineConfig,
    *,
    speculative_config: str | None,
) -> None:
    if speculative_config is None:
        return
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--speculative-config currently applies only to Qwen3.5-Omni configs"
        )
    config = _parse_qwen35_json_object("--speculative-config", speculative_config)
    if not config:
        logger.info(
            "--speculative-config={} accepted as a Qwen3.5 vLLM-compatible no-op."
        )
        return
    method = str(config.get("method", "")).strip().lower()
    if method in _QWEN35_SPECULATIVE_MTP_METHODS:
        raise typer.BadParameter(
            "--speculative-config method qwen3_omni_next_thinker_mtp requires "
            "Qwen3.5 thinker MTP/draft decoding, which is not enabled in this "
            "SGLang path yet; omit the speculative config or pass "
            "--disable-mtp for the current base thinker AR path"
        )
    # 中文说明：非空 speculative_config 会改变解码路径和 KV/worker 形态。
    # 当前 Qwen3.5 先实现 base thinker AR 主链路，因此必须提前失败。
    raise typer.BadParameter(
        "--speculative-config is not mapped in the current SGLang Qwen3.5 path"
    )


def _validate_qwen35_max_seq_len_to_capture_request(
    pipeline_config: PipelineConfig,
    *,
    max_seq_len_to_capture: int | None,
) -> None:
    if max_seq_len_to_capture is None:
        return
    if max_seq_len_to_capture < 1:
        raise typer.BadParameter("--max-seq-len-to-capture must be >= 1")
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--max-seq-len-to-capture currently applies only to "
            "Qwen3.5-Omni configs"
        )
    # 中文说明：这是 vLLM cudagraph capture 参数。当前 SGLang Qwen3.5
    # 路径没有单独的 capture length 入口；接受它是为了迁移 vLLM
    # perf 配置时不因未知参数失败。
    logger.info(
        "--max-seq-len-to-capture=%s accepted as a Qwen3.5 vLLM-compatible "
        "no-op; SGLang uses its own CUDA graph capture settings.",
        max_seq_len_to_capture,
    )


def _validate_qwen35_mamba_cache_mode_request(
    pipeline_config: PipelineConfig,
    *,
    mamba_cache_mode: str | None,
) -> None:
    if mamba_cache_mode is None:
        return
    normalized = mamba_cache_mode.strip().lower()
    if normalized not in _QWEN35_MAMBA_CACHE_MODES:
        supported = ", ".join(sorted(_QWEN35_MAMBA_CACHE_MODES))
        raise typer.BadParameter(f"--mamba-cache-mode must be one of: {supported}")
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--mamba-cache-mode currently applies only to Qwen3.5-Omni configs"
        )
    if normalized == "none":
        logger.info(
            "--mamba-cache-mode=none accepted as a Qwen3.5 vLLM-compatible no-op."
        )
        return
    # 中文说明：vLLM 的 light/all 是混合 Mamba cache 策略，常和
    # disaggregated prefill/decode profile 一起出现。SGLang core 当前没有
    # 等价 mamba_cache_mode ServerArgs；这里提前失败，避免静默丢失缓存策略。
    raise typer.BadParameter(
        "--mamba-cache-mode light/all is a vLLM hybrid Mamba cache setting "
        "and is not supported by the current SGLang Qwen3.5 path; use "
        "--mamba-cache-mode none or omit it until an explicit SGLang mapping "
        "is implemented."
    )


def _parse_qwen35_vllm_bool_value(flag_name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise typer.BadParameter(f"{flag_name} expects a boolean value")


def _parse_optional_json_object(flag_name: str, value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    raw = value.strip()
    if not raw or raw.lower() in {"none", "null"}:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{flag_name} expects a JSON object") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{flag_name} expects a JSON object")
    return parsed


def _validate_qwen35_kv_transfer_request(
    pipeline_config: PipelineConfig,
    *,
    enable_disaggregated_prefilling: object | None,
    kv_transfer_config: str | None,
) -> None:
    if enable_disaggregated_prefilling is None and kv_transfer_config is None:
        return
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--kv-transfer-config/--enable-disaggregated-prefilling currently "
            "apply only to Qwen3.5-Omni configs"
        )
    if enable_disaggregated_prefilling is not None:
        enabled = _parse_qwen35_vllm_bool_value(
            "--enable-disaggregated-prefilling",
            enable_disaggregated_prefilling,
        )
        if enabled:
            # 中文说明：vLLM 的 disaggregated prefill/decode profile 需要
            # KV connector producer/consumer 协议；当前 SGLang-Omni
            # Qwen3.5 尚未接这个映射，提前报错比静默降级更可靠。
            raise typer.BadParameter(
                "--enable-disaggregated-prefilling is a vLLM KV transfer "
                "setting and is not supported by the current SGLang Qwen3.5 path"
            )
        logger.info(
            "--enable-disaggregated-prefilling=false accepted as a "
            "Qwen3.5 vLLM-compatible no-op."
        )
    kv_config = _parse_optional_json_object("--kv-transfer-config", kv_transfer_config)
    if kv_config:
        raise typer.BadParameter(
            "--kv-transfer-config is a vLLM KV connector setting and is not "
            "supported by the current SGLang Qwen3.5 path; use the colocated "
            "or non-disaggregated profile until an explicit SGLang KV transfer "
            "mapping is implemented"
        )
    if kv_transfer_config is not None:
        logger.info(
            "--kv-transfer-config={} accepted as a Qwen3.5 vLLM-compatible no-op."
        )


def _validate_qwen35_vllm_engine_profile_request(
    pipeline_config: PipelineConfig,
    *,
    distributed_executor_backend: str | None,
    kv_cache_dtype: str | None,
    enable_expert_parallel: object | None,
    mm_processor_cache_gb: float | None,
    max_mm_len: int | None,
    use_omni_engine: object | None,
    use_omni_rpc_engine: object | None,
    use_zero_shot: object | None,
    skip_mm_profiling: object | None,
    video_needs_metadata: object | None = None,
    override_video_max_pixels: object | None = None,
    is_thinker: object | None = None,
    thinker_only_marker: object | None = None,
    text_only: bool = False,
) -> None:
    requested = any(
        value is not None
        for value in (
            distributed_executor_backend,
            kv_cache_dtype,
            enable_expert_parallel,
            mm_processor_cache_gb,
            max_mm_len,
            use_omni_engine,
            use_omni_rpc_engine,
            use_zero_shot,
            skip_mm_profiling,
            video_needs_metadata,
            override_video_max_pixels,
            is_thinker,
            thinker_only_marker,
        )
    )
    if not requested:
        return
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "vLLM Qwen3.5 engine profile flags currently apply only to "
            "Qwen3.5-Omni configs"
        )
    if (
        distributed_executor_backend is not None
        and distributed_executor_backend.strip().lower()
        not in _QWEN35_VLLM_EXECUTOR_BACKENDS
    ):
        raise typer.BadParameter(
            "--distributed-executor-backend currently supports only mp as a "
            "Qwen3.5 vLLM-compatible no-op"
        )
    if kv_cache_dtype is not None and kv_cache_dtype.strip().lower() != "auto":
        # 中文说明：tq4/fp8 等 KV cache dtype 会改变 KV cache 存储和
        # attention backend。当前 SGLang Qwen3.5 没有等价入口，必须提前失败。
        raise typer.BadParameter(
            "--kv-cache-dtype values other than auto are not supported by the "
            "current SGLang Qwen3.5 path"
        )
    if enable_expert_parallel is not None and _parse_qwen35_vllm_bool_value(
        "--enable-expert-parallel",
        enable_expert_parallel,
    ):
        raise typer.BadParameter(
            "--enable-expert-parallel is a vLLM MoE parallelism setting and is "
            "not mapped in the current SGLang Qwen3.5 path"
        )
    if mm_processor_cache_gb not in (None, 0, 0.0):
        raise typer.BadParameter("--mm-processor-cache-gb currently supports only 0")
    for flag_name, value in (
        ("--use-omni-engine", use_omni_engine),
        ("--use-omni-rpc-engine", use_omni_rpc_engine),
        ("--use-zero-shot", use_zero_shot),
        ("--skip-mm-profiling", skip_mm_profiling),
        ("--video-needs-metadata", video_needs_metadata),
    ):
        if value is not None:
            _parse_qwen35_vllm_bool_value(flag_name, value)
    if override_video_max_pixels is not None:
        _parse_qwen35_vllm_bool_value(
            "--override-video-max-pixels",
            override_video_max_pixels,
        )
    if is_thinker is not None and not _parse_qwen35_vllm_bool_value(
        "--is-thinker",
        is_thinker,
    ):
        raise typer.BadParameter("--is-thinker=false is not valid here")
    if thinker_only_marker is not None:
        marker = _parse_qwen35_vllm_bool_value(
            "--thinker-only",
            thinker_only_marker,
        )
        if text_only and not marker:
            raise typer.BadParameter("--thinker-only=false conflicts with --text-only")
        if not text_only and marker:
            raise typer.BadParameter("--thinker-only=true requires --text-only")


def _server_args_from_vllm_compilation_config(
    compilation_config: str | None,
) -> dict[str, object]:
    if compilation_config is None:
        return {}
    try:
        value = json.loads(compilation_config)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            "--compilation-config expects a JSON object"
        ) from exc
    if not isinstance(value, dict):
        raise typer.BadParameter("--compilation-config expects a JSON object")

    use_inductor = value.get("use_inductor")
    if use_inductor not in (None, False):
        raise typer.BadParameter(
            "--compilation-config use_inductor=true is not supported"
        )
    pass_config = value.get("pass_config")
    if pass_config is not None:
        if not isinstance(pass_config, dict):
            raise typer.BadParameter(
                "--compilation-config pass_config must be an object"
            )
        enabled = [key for key, flag in pass_config.items() if bool(flag)]
        if enabled:
            raise typer.BadParameter(
                "--compilation-config fuse pass options are not supported: "
                + ", ".join(sorted(str(key) for key in enabled))
            )

    raw_mode = str(value.get("cudagraph_mode", "")).strip().lower()
    if not raw_mode:
        return {}
    normalized = raw_mode.replace("-", "_")
    if normalized in {"full_decode_only", "full", "decode_only"}:
        return {}
    if normalized in {"none", "off", "no_cudagraph", "disable", "disabled"}:
        return {"disable_cuda_graph": True}
    raise typer.BadParameter(
        "--compilation-config cudagraph_mode must be FULL_DECODE_ONLY or none/off"
    )


def _find_matching_stages(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    reason: str,
):
    matching_stages = [
        stage for stage in pipeline_config.stages if stage.name == stage_name
    ]
    if not matching_stages:
        raise typer.BadParameter(
            f"Stage {stage_name!r} not found in pipeline; cannot set {reason}"
        )
    return matching_stages


def _raise_unsupported_flag(
    pipeline_config: PipelineConfig,
    flag_name: str,
) -> NoReturn:
    raise typer.BadParameter(
        f"{flag_name} is not supported by {type(pipeline_config).__name__}"
    )


def _resolve_talker_stage(
    pipeline_config: PipelineConfig,
    *,
    flag_name: str,
) -> str:
    stage_name = type(pipeline_config).talker_role_to_stage().get("talker")
    if stage_name is None:
        _raise_unsupported_flag(pipeline_config, flag_name)
    return stage_name


def _resolve_talker_sglang_stage(
    pipeline_config: PipelineConfig,
    *,
    flag_name: str,
) -> str:
    stage_name = type(pipeline_config).talker_sglang_role_to_stage().get("talker")
    if stage_name is None:
        _raise_unsupported_flag(pipeline_config, flag_name)
    return stage_name


def _apply_stage_server_args_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    updates: dict[str, object],
    reason: str,
) -> None:
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=reason,
    )
    for stage in matching_stages:
        factory_args = dict(stage.factory_args or {})
        overrides = dict(factory_args.get("server_args_overrides") or {})
        overrides.update(updates)
        factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args

        stage_runtime_overrides = pipeline_config.runtime_overrides.get(stage.name)
        if stage_runtime_overrides is not None:
            runtime_server_args = stage_runtime_overrides.get("server_args_overrides")
            if isinstance(runtime_server_args, dict):
                runtime_server_args.update(updates)


def _apply_stage_mem_fraction_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    value: float,
) -> None:
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason="SGLang mem_fraction_static override",
    )
    for stage in matching_stages:
        stage.runtime.sglang_server_args.mem_fraction_static = value


def _stage_has_explicit_mem_fraction_static(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    factory_args: dict[str, object],
) -> bool:
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason="mem_fraction_static validation",
    )
    if any(
        stage.runtime.sglang_server_args.mem_fraction_static is not None
        for stage in matching_stages
    ):
        return True

    server_args_overrides = dict(factory_args.get("server_args_overrides") or {})
    if server_args_overrides.get("mem_fraction_static") is not None:
        return True

    runtime_overrides = dict(pipeline_config.runtime_overrides.get(stage_name, {}))
    runtime_server_args_overrides = dict(
        runtime_overrides.get("server_args_overrides") or {}
    )
    return runtime_server_args_overrides.get("mem_fraction_static") is not None


def _validate_mem_fraction_static(flag_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 < value < 1.0:
        raise typer.BadParameter(f"{flag_name} must be > 0 and < 1, got {value}")
    return float(value)


def _validate_encoder_mem_reserve(value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value < 1.0:
        raise typer.BadParameter("--encoder-mem-reserve must be in [0, 1)")
    return float(value)


def apply_mem_fraction_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
) -> PipelineConfig:
    """Apply CLI mem_fraction_static flags to the pipeline config.

    Precedence (per role): a non-None per-role flag wins over the global flag.
    `--thinker-mem-fraction-static` overrides `--mem-fraction-static` for the
    thinker stage; `--talker-mem-fraction-static` overrides it for the talker
    stage. The global `--mem-fraction-static` is the fallback for any role
    whose per-role flag is omitted.

    Validation: out-of-range values raise typer.BadParameter atomically, before
    any stage mutation, so a partially-applied config cannot leak into the
    launch path.
    """
    mem_fraction_static = _validate_mem_fraction_static(
        "--mem-fraction-static", mem_fraction_static
    )
    thinker_mem_fraction_static = _validate_mem_fraction_static(
        "--thinker-mem-fraction-static", thinker_mem_fraction_static
    )
    talker_mem_fraction_static = _validate_mem_fraction_static(
        "--talker-mem-fraction-static", talker_mem_fraction_static
    )

    role_to_stage = type(pipeline_config).mem_fraction_role_to_stage()
    if mem_fraction_static is not None and not role_to_stage:
        raise typer.BadParameter(
            "--mem-fraction-static requires a pipeline with a supported "
            "SGLang AR mem_fraction_static target"
        )
    if thinker_mem_fraction_static is not None and "thinker" not in role_to_stage:
        raise typer.BadParameter(
            "--thinker-mem-fraction-static is not supported by pipeline "
            f"{type(pipeline_config).__name__}."
        )
    if talker_mem_fraction_static is not None and "talker" not in role_to_stage:
        raise typer.BadParameter(
            "--talker-mem-fraction-static is not supported by pipeline "
            f"{type(pipeline_config).__name__}."
        )

    role_values = {
        "thinker": thinker_mem_fraction_static,
        "talker": talker_mem_fraction_static,
    }
    for role, stage_name in role_to_stage.items():
        role_value = role_values.get(role)
        # Precedence: per-role flag wins over the global flag for this role;
        # the global flag is the fallback when no per-role flag was given.
        final_value = role_value if role_value is not None else mem_fraction_static
        if final_value is not None:
            _apply_stage_mem_fraction_override(
                pipeline_config,
                stage_name=stage_name,
                value=final_value,
            )
    return pipeline_config


def _validate_max_running_requests(flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) < 1:
        raise typer.BadParameter(f"{flag_name} must be >= 1, got {value}")
    return int(value)


def apply_max_running_requests_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    max_running_requests: int | None,
    thinker_max_running_requests: int | None,
    talker_max_running_requests: int | None,
) -> PipelineConfig:
    """Apply CLI max_running_requests flags to supported SGLang AR stages."""

    max_running_requests = _validate_max_running_requests(
        "--max-running-requests",
        max_running_requests,
    )
    thinker_max_running_requests = _validate_max_running_requests(
        "--thinker-max-running-requests",
        thinker_max_running_requests,
    )
    talker_max_running_requests = _validate_max_running_requests(
        "--talker-max-running-requests",
        talker_max_running_requests,
    )

    role_to_stage = type(pipeline_config).max_running_requests_role_to_stage()
    if max_running_requests is not None and not role_to_stage:
        raise typer.BadParameter(
            "--max-running-requests requires a pipeline with a supported "
            "SGLang AR max_running_requests target"
        )
    if thinker_max_running_requests is not None and "thinker" not in role_to_stage:
        raise typer.BadParameter(
            "--thinker-max-running-requests is not supported by pipeline "
            f"{type(pipeline_config).__name__}."
        )
    if talker_max_running_requests is not None and "talker" not in role_to_stage:
        raise typer.BadParameter(
            "--talker-max-running-requests is not supported by pipeline "
            f"{type(pipeline_config).__name__}."
        )

    role_values = {
        "thinker": thinker_max_running_requests,
        "talker": talker_max_running_requests,
    }
    for role, stage_name in role_to_stage.items():
        role_value = role_values.get(role)
        final_value = role_value if role_value is not None else max_running_requests
        if final_value is None:
            continue
        matching_stages = _find_matching_stages(
            pipeline_config,
            stage_name=stage_name,
            reason=f"SGLang max_running_requests override for {role}",
        )
        for stage in matching_stages:
            # 中文说明：max_running_requests 属于 typed SGLang runtime
            # intent。写到 runtime 后，YAML/CLI/example launcher 的语义一致，
            # 最终由 config.runtime 统一翻译成 ServerArgs。
            stage.runtime.sglang_server_args.max_running_requests = final_value
    return pipeline_config


def apply_encoder_mem_reserve_cli_override(
    pipeline_config: PipelineConfig,
    *,
    encoder_mem_reserve: float | None,
    mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
) -> PipelineConfig:
    if encoder_mem_reserve is None:
        return pipeline_config
    encoder_mem_reserve = _validate_encoder_mem_reserve(encoder_mem_reserve)

    role_to_stage = type(pipeline_config).encoder_mem_reserve_role_to_stage()
    thinker_stage = role_to_stage.get("thinker")
    if thinker_stage is None:
        _raise_unsupported_flag(pipeline_config, "--encoder-mem-reserve")

    if mem_fraction_static is not None or thinker_mem_fraction_static is not None:
        raise typer.BadParameter(
            "--encoder-mem-reserve is mutually exclusive with "
            "--mem-fraction-static and --thinker-mem-fraction-static"
        )

    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=thinker_stage,
        reason="Qwen thinker encoder memory reserve",
    )
    for stage in matching_stages:
        factory_args = dict(stage.factory_args or {})
        if _stage_has_explicit_mem_fraction_static(
            pipeline_config,
            stage_name=stage.name,
            factory_args=factory_args,
        ):
            raise typer.BadParameter(
                "--encoder-mem-reserve is only valid when thinker "
                "mem_fraction_static is not explicitly pinned"
            )
        factory_args["encoder_mem_reserve"] = encoder_mem_reserve
        stage.factory_args = factory_args

        stage_runtime_overrides = pipeline_config.runtime_overrides.get(stage.name)
        if (
            isinstance(stage_runtime_overrides, dict)
            and "encoder_mem_reserve" in stage_runtime_overrides
        ):
            stage_runtime_overrides["encoder_mem_reserve"] = encoder_mem_reserve
    return pipeline_config


def _parse_gpu_placement(flag_name: str, value: str) -> int | list[int]:
    text = value.strip()
    if not text:
        raise typer.BadParameter(f"{flag_name} must not be empty")

    if text.startswith("["):
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(
                f"{flag_name} must be an int or list of ints"
            ) from exc
    elif "," in text:
        parsed = [part.strip() for part in text.split(",")]
    else:
        try:
            gpu = int(text)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{flag_name} must be an int or list of ints"
            ) from exc
        if gpu < 0:
            raise typer.BadParameter(f"{flag_name} GPU ids must be >= 0")
        return gpu

    if not isinstance(parsed, list) or not parsed:
        raise typer.BadParameter(f"{flag_name} must be an int or non-empty list")

    gpus: list[int] = []
    for item in parsed:
        if isinstance(item, int):
            gpu = item
        elif isinstance(item, str):
            try:
                gpu = int(item.strip())
            except ValueError as exc:
                raise typer.BadParameter(
                    f"{flag_name} must contain only integer GPU ids"
                ) from exc
        else:
            raise typer.BadParameter(f"{flag_name} must contain only integer GPU ids")
        if gpu < 0:
            raise typer.BadParameter(f"{flag_name} GPU ids must be >= 0")
        gpus.append(gpu)

    return gpus[0] if len(gpus) == 1 else gpus


def _parse_single_gpu_visible_device(flag_name: str, value: str | None) -> int | None:
    if value is None:
        return None
    parsed = _parse_gpu_placement(flag_name, value)
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise typer.BadParameter(f"{flag_name} currently supports exactly 1 GPU")
        return int(parsed[0])
    return int(parsed)


def _validate_stage_parallelism_config(stage_name: str, tp_size: int, gpu) -> None:
    if tp_size < 1:
        raise typer.BadParameter(f"{stage_name}_tp_size must be >= 1")
    if tp_size == 1:
        if isinstance(gpu, list) and len(gpu) != 1:
            raise typer.BadParameter(
                f"{stage_name}_gpus must contain exactly 1 GPU id when {stage_name}_tp_size=1"
            )
        return
    if not isinstance(gpu, list):
        raise typer.BadParameter(
            f"{stage_name}_gpus must provide one GPU id per TP rank "
            f"when {stage_name}_tp_size > 1"
        )
    if len(gpu) != tp_size:
        raise typer.BadParameter(
            f"{stage_name}_gpus must contain exactly {tp_size} GPU ids "
            f"when {stage_name}_tp_size={tp_size}"
        )
    if len(set(gpu)) != len(gpu):
        raise typer.BadParameter(
            f"{stage_name}_gpus must not contain duplicate GPU ids"
        )


def _apply_stage_gpu_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    gpu: int | None,
) -> None:
    if gpu is None:
        return
    if gpu < 0:
        raise typer.BadParameter(f"{stage_name}_gpu must be >= 0")
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=f"GPU placement to {gpu}",
    )
    for stage in matching_stages:
        stage.gpu = int(gpu)


def _validate_colocated_gpu_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    flag_name: str,
    gpu: int | None,
) -> None:
    if (
        gpu is None
        or type(pipeline_config).__name__ not in _QWEN_COLOCATED_CONFIG_CLASSES
    ):
        return
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=f"{flag_name} placement validation",
    )
    current_gpu = matching_stages[0].gpu
    if current_gpu != gpu:
        raise typer.BadParameter(
            f"{flag_name} cannot move {stage_name} away from the colocated GPU"
        )


def _apply_tensor_parallel_server_args_overrides(
    pipeline_config: PipelineConfig,
) -> None:
    config_cls = type(pipeline_config)
    for stage in pipeline_config.stages:
        updates = config_cls.tensor_parallel_server_args_overrides(
            stage_name=stage.name,
            tp_size=stage.tp_size,
        )
        if not updates:
            continue
        _apply_stage_server_args_override(
            pipeline_config,
            stage_name=stage.name,
            updates=updates,
            reason=f"tensor parallel server args for {stage.name}",
        )


def _validate_parallelism_config(pipeline_config: PipelineConfig) -> None:
    try:
        type(pipeline_config)(**pipeline_config.model_dump())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def apply_thinker_server_args_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    cpu_offload_gb: int | None,
    quantization: str | None,
) -> PipelineConfig:
    updates: dict[str, object] = {}
    if cpu_offload_gb is not None:
        if cpu_offload_gb < 0:
            raise typer.BadParameter("--cpu-offload-gb must be >= 0")
        updates["cpu_offload_gb"] = int(cpu_offload_gb)
    if quantization is not None:
        quantization = quantization.strip()
        if not quantization:
            raise typer.BadParameter("--quantization must not be empty")
        if quantization.lower() != "none":
            updates["quantization"] = quantization

    if updates:
        _apply_stage_server_args_override(
            pipeline_config,
            stage_name="thinker",
            updates=updates,
            reason="thinker SGLang ServerArgs override",
        )
    return pipeline_config


def apply_talker_server_args_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    talker_quantization: str | None,
) -> PipelineConfig:
    updates: dict[str, object] = {}
    if talker_quantization is not None:
        talker_quantization = talker_quantization.strip()
        if not talker_quantization:
            raise typer.BadParameter("--talker-quantization must not be empty")
        if talker_quantization.lower() != "none":
            updates["quantization"] = talker_quantization

    if not updates:
        return pipeline_config

    role_to_stage = type(pipeline_config).talker_sglang_role_to_stage()
    if "talker" not in role_to_stage:
        raise typer.BadParameter(
            "--talker-quantization is not supported by pipeline "
            f"{type(pipeline_config).__name__}"
        )

    _apply_stage_server_args_override(
        pipeline_config,
        stage_name=role_to_stage["talker"],
        updates=updates,
        reason="talker SGLang ServerArgs override",
    )
    return pipeline_config


def apply_vllm_ar_server_args_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    prefix_caching: str,
    chunked_prefill: str,
    enforce_eager: bool,
    thinker_enforce_eager: bool,
    talker_enforce_eager: bool,
    dtype: str | None = None,
    thinker_dtype: str | None = None,
    talker_dtype: str | None = None,
    mamba_ssm_dtype: str | None = None,
    mamba_cache_mode: str | None = None,
    kv_transfer_config: str | None = None,
    enable_disaggregated_prefilling: object | None = None,
    max_num_batched_tokens: int | None = None,
    page_size: int | None = None,
    compilation_config: str | None = None,
) -> PipelineConfig:
    _validate_qwen35_mamba_cache_mode_request(
        pipeline_config,
        mamba_cache_mode=mamba_cache_mode,
    )
    _validate_qwen35_kv_transfer_request(
        pipeline_config,
        enable_disaggregated_prefilling=enable_disaggregated_prefilling,
        kv_transfer_config=kv_transfer_config,
    )
    prefix_mode = _normalize_stage_toggle_mode("prefix_caching", prefix_caching)
    chunked_mode = _normalize_stage_toggle_mode("chunked_prefill", chunked_prefill)
    shared_updates: dict[str, object] = {}
    if prefix_mode != "default":
        shared_updates["disable_radix_cache"] = prefix_mode != "on"
    max_num_batched_tokens = _validate_positive_int(
        "--max-num-batched-tokens",
        max_num_batched_tokens,
    )
    if max_num_batched_tokens is not None:
        # 中文说明：vLLM 的 max_num_batched_tokens 是 prefill 批处理
        # token 上限；SGLang 对应 max_prefill_tokens，并复用同一个值
        # 作为 chunked_prefill_size，除非用户显式关闭 chunked prefill。
        shared_updates["max_prefill_tokens"] = max_num_batched_tokens
        shared_updates["chunked_prefill_size"] = max_num_batched_tokens
    if chunked_mode != "default":
        if not _is_qwen35_config(pipeline_config):
            raise typer.BadParameter(
                "--chunked-prefill currently supports only Qwen3.5-Omni configs"
            )
        from sglang_omni.models.qwen3_5_omni.config import (
            QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE,
        )

        shared_updates["chunked_prefill_size"] = (
            max_num_batched_tokens or QWEN3_5_OMNI_CHUNKED_PREFILL_SIZE
            if chunked_mode == "on"
            else None
        )
    if dtype is not None:
        shared_updates["dtype"] = _normalize_dtype_cli_value("--dtype", dtype)
    if mamba_ssm_dtype is not None:
        shared_updates["mamba_ssm_dtype"] = _normalize_dtype_cli_value(
            "--mamba-ssm-dtype",
            mamba_ssm_dtype,
        )
    page_size = _validate_positive_int("--block-size", page_size)
    if page_size is not None:
        # 中文说明：vLLM block_size 对应 SGLang page_size，控制 KV cache
        # 分页大小；保留 --page-size 作为更贴近 SGLang 的名字。
        shared_updates["page_size"] = page_size
    shared_updates.update(
        _server_args_from_vllm_compilation_config(compilation_config)
    )

    if shared_updates:
        _apply_stage_server_args_override(
            pipeline_config,
            stage_name="thinker",
            updates=shared_updates,
            reason="vLLM-compatible AR ServerArgs override",
        )
        talker_stage = type(pipeline_config).talker_sglang_role_to_stage().get(
            "talker"
        )
        if talker_stage is not None:
            _apply_stage_server_args_override(
                pipeline_config,
                stage_name=talker_stage,
                updates=shared_updates,
                reason="vLLM-compatible AR ServerArgs override",
            )

    if enforce_eager or thinker_enforce_eager:
        _apply_stage_server_args_override(
            pipeline_config,
            stage_name="thinker",
            updates={"disable_cuda_graph": True},
            reason="vLLM-compatible enforce eager",
        )
    if enforce_eager or talker_enforce_eager:
        talker_stage = type(pipeline_config).talker_sglang_role_to_stage().get(
            "talker"
        )
        if talker_stage is not None:
            _apply_stage_server_args_override(
                pipeline_config,
                stage_name=talker_stage,
                updates={"disable_cuda_graph": True},
                reason="vLLM-compatible talker enforce eager",
            )
        elif talker_enforce_eager:
            _raise_unsupported_flag(pipeline_config, "--talker-enforce-eager")
    if thinker_dtype is not None:
        _apply_stage_server_args_override(
            pipeline_config,
            stage_name="thinker",
            updates={
                "dtype": _normalize_dtype_cli_value(
                    "--thinker-dtype",
                    thinker_dtype,
                )
            },
            reason="vLLM-compatible thinker dtype",
        )
    if talker_dtype is not None:
        talker_stage = type(pipeline_config).talker_sglang_role_to_stage().get(
            "talker"
        )
        if talker_stage is None:
            _raise_unsupported_flag(pipeline_config, "--talker-dtype")
        _apply_stage_server_args_override(
            pipeline_config,
            stage_name=talker_stage,
            updates={
                "dtype": _normalize_dtype_cli_value(
                    "--talker-dtype",
                    talker_dtype,
                )
            },
            reason="vLLM-compatible talker dtype",
        )
    return pipeline_config


def apply_parallelism_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    thinker_tp_size: int | None,
    thinker_gpus: str | None,
    talker_gpu: int | None,
    code2wav_gpu: int | None,
) -> PipelineConfig:
    thinker_gpu_override = (
        _parse_gpu_placement("thinker_gpus", thinker_gpus)
        if thinker_gpus is not None
        else None
    )
    if thinker_tp_size is not None or thinker_gpu_override is not None:
        thinker_stages = _find_matching_stages(
            pipeline_config,
            stage_name="thinker",
            reason="tensor parallel settings",
        )
        for stage in thinker_stages:
            if thinker_tp_size is not None:
                stage.tp_size = int(thinker_tp_size)
                stage.parallelism.tp = stage.tp_size
            if thinker_gpu_override is not None:
                stage.gpu = thinker_gpu_override
            _validate_stage_parallelism_config("thinker", stage.tp_size, stage.gpu)
            if stage.tp_size == 1 and isinstance(stage.gpu, list):
                stage.gpu = int(stage.gpu[0])

    talker_stage = (
        _resolve_talker_stage(
            pipeline_config,
            flag_name="--talker-gpu",
        )
        if talker_gpu is not None
        else None
    )
    code2wav_stage = None
    if code2wav_gpu is not None:
        code2wav_stage = type(pipeline_config).code2wav_stage()
        if code2wav_stage is None:
            _raise_unsupported_flag(pipeline_config, "--code2wav-gpu")

    if talker_stage is not None:
        _validate_colocated_gpu_override(
            pipeline_config,
            stage_name=talker_stage,
            flag_name="--talker-gpu",
            gpu=talker_gpu,
        )
    if code2wav_stage is not None:
        _validate_colocated_gpu_override(
            pipeline_config,
            stage_name=code2wav_stage,
            flag_name="--code2wav-gpu",
            gpu=code2wav_gpu,
        )

    if talker_stage is not None:
        _apply_stage_gpu_override(
            pipeline_config,
            stage_name=talker_stage,
            gpu=talker_gpu,
        )
    if code2wav_stage is not None:
        _apply_stage_gpu_override(
            pipeline_config,
            stage_name=code2wav_stage,
            gpu=code2wav_gpu,
        )
    _apply_tensor_parallel_server_args_overrides(pipeline_config)
    _validate_parallelism_config(pipeline_config)
    return pipeline_config


def _apply_stage_cuda_graph_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    mode: _STAGE_TOGGLE_MODE,
) -> None:
    if mode == "default":
        return

    _apply_stage_server_args_override(
        pipeline_config,
        stage_name=stage_name,
        updates={"disable_cuda_graph": mode != "on"},
        reason=f"CUDA graph mode to {mode!r}",
    )


def _apply_stage_torch_compile_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    mode: _STAGE_TOGGLE_MODE,
    max_bs: int | None,
) -> None:
    if mode == "default" and max_bs is None:
        return

    updates: dict[str, object] = {}
    if mode != "default":
        updates["enable_torch_compile"] = mode == "on"
    if max_bs is not None:
        if int(max_bs) < 1:
            raise typer.BadParameter("torch compile max batch size must be >= 1")
        updates["torch_compile_max_bs"] = int(max_bs)

    _apply_stage_server_args_override(
        pipeline_config,
        stage_name=stage_name,
        updates=updates,
        reason=(f"torch compile settings (mode={mode!r}, max_bs={max_bs})"),
    )


def apply_cuda_graph_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    thinker_cuda_graph: str,
    talker_cuda_graph: str,
) -> PipelineConfig:
    thinker_mode = _normalize_stage_toggle_mode(
        "thinker_cuda_graph", thinker_cuda_graph
    )
    talker_mode = _normalize_stage_toggle_mode("talker_cuda_graph", talker_cuda_graph)
    _apply_stage_cuda_graph_override(
        pipeline_config,
        stage_name="thinker",
        mode=thinker_mode,
    )
    if talker_mode != "default":
        _apply_stage_cuda_graph_override(
            pipeline_config,
            stage_name=_resolve_talker_sglang_stage(
                pipeline_config,
                flag_name="--talker-cuda-graph",
            ),
            mode=talker_mode,
        )
    return pipeline_config


def apply_partial_start_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    talker_partial_start: str,
) -> PipelineConfig:
    mode = _normalize_stage_toggle_mode("talker_partial_start", talker_partial_start)
    if mode == "default":
        return pipeline_config
    stage_name = _resolve_talker_stage(
        pipeline_config,
        flag_name="--talker-partial-start",
    )
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=f"talker partial-start mode to {mode!r}",
    )
    for stage in matching_stages:
        if stage.factory not in _QWEN_PARTIAL_START_TALKER_FACTORIES:
            raise typer.BadParameter(
                "--talker-partial-start currently supports only Qwen Omni "
                f"talker; stage {stage.name!r} uses factory {stage.factory!r}"
            )
    _apply_stage_factory_args_override(
        pipeline_config,
        stage_name=stage_name,
        updates={"enable_partial_start": mode == "on"},
        reason=f"talker partial-start mode to {mode!r}",
        flag_name="--talker-partial-start",
    )
    return pipeline_config


def _apply_stage_factory_args_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    updates: dict[str, object],
    reason: str,
    supported_factory: str | None = None,
    flag_name: str | None = None,
) -> None:
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=reason,
    )
    for stage in matching_stages:
        if supported_factory is not None and stage.factory != supported_factory:
            display_flag = flag_name or reason
            raise typer.BadParameter(
                f"{display_flag} currently supports only Higgs TTS; "
                f"stage {stage.name!r} uses factory {stage.factory!r}"
            )
        factory_args = dict(stage.factory_args or {})
        factory_args.update(updates)
        stage.factory_args = factory_args

        stage_runtime_overrides = pipeline_config.runtime_overrides.get(stage.name)
        if isinstance(stage_runtime_overrides, dict):
            stage_runtime_overrides.update(updates)


def _parse_positive_int_sequence(flag_name: str, value: str) -> tuple[int, ...]:
    pieces = [
        piece.strip()
        for piece in value.replace(",", " ").split()
        if piece.strip()
    ]
    if not pieces:
        raise typer.BadParameter(f"{flag_name} requires at least one integer")
    parsed: list[int] = []
    for piece in pieces:
        try:
            item = int(piece)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{flag_name} must contain only integers"
            ) from exc
        if item < 1:
            raise typer.BadParameter(f"{flag_name} values must be >= 1")
        parsed.append(item)
    return tuple(parsed)


def _validate_positive_int(flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) < 1:
        raise typer.BadParameter(f"{flag_name} must be >= 1")
    return int(value)


def _validate_nonnegative_int(flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if int(value) < 0:
        raise typer.BadParameter(f"{flag_name} must be >= 0")
    return int(value)


def _validate_positive_float(flag_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if float(value) <= 0:
        raise typer.BadParameter(f"{flag_name} must be > 0")
    return float(value)


def _parse_limit_mm_per_prompt(
    value: str | None,
    *,
    flag_name: str = "--limit-mm-per-prompt",
) -> dict[str, int]:
    if value is None:
        return {}
    raw = value.strip()
    if not raw:
        raise typer.BadParameter(f"{flag_name} must not be empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = _parse_limit_mm_pairs(raw, flag_name=flag_name)
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{flag_name} expects a JSON object")
    return _normalize_limit_mm_per_prompt(parsed, flag_name=flag_name)


def _parse_limit_mm_pairs(
    raw: str,
    *,
    flag_name: str,
) -> dict[str, int]:
    parsed: dict[str, int] = {}
    pieces = [
        piece.strip()
        for piece in raw.replace(",", " ").split()
        if piece.strip()
    ]
    for piece in pieces:
        if "=" not in piece:
            raise typer.BadParameter(
                f"{flag_name} expects JSON or modality=count pairs"
            )
        key, raw_value = piece.split("=", 1)
        parsed[key.strip()] = int(raw_value)
    return parsed


def _normalize_limit_mm_per_prompt(
    value: dict[object, object],
    *,
    flag_name: str,
) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_key, raw_limit in value.items():
        modality = str(raw_key).strip().lower()
        if modality.endswith("s"):
            modality = modality[:-1]
        if modality not in _LIMIT_MM_MODALITIES:
            supported = ", ".join(sorted(_LIMIT_MM_MODALITIES))
            raise typer.BadParameter(
                f"{flag_name} modality must be one of: {supported}"
            )
        limit = int(raw_limit)
        if limit < 0:
            raise typer.BadParameter(f"{flag_name} values must be >= 0")
        normalized[modality] = limit
    return normalized


def _resolve_toggle_alias_flags(
    *,
    flag_name: str,
    mode: str,
    enable_alias: bool,
    disable_alias: bool,
) -> str:
    normalized = _normalize_stage_toggle_mode(flag_name, mode)
    if enable_alias and disable_alias:
        raise typer.BadParameter(
            f"{flag_name} enable and disable aliases cannot be combined"
        )
    if enable_alias:
        if normalized == "off":
            raise typer.BadParameter(
                f"{flag_name} enable alias cannot be combined with {flag_name} off"
            )
        return "on"
    if disable_alias:
        if normalized == "on":
            raise typer.BadParameter(
                f"{flag_name} disable alias cannot be combined with {flag_name} on"
            )
        return "off"
    return normalized


def apply_preprocessing_video_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_fps: float | None,
    video_max_frames: int | None,
    video_min_frames: int | None,
    video_min_pixels: int | None,
    video_max_pixels: int | None,
    video_total_pixels: int | None,
    video_override_max_pixels: bool | None = None,
    video_seconds_per_chunk: float | None = None,
    video_position_id_per_seconds: float | None = None,
    audio_target_sr: int | None = None,
    audio_timestamp_interval: int | None = None,
    audio_downsample_times: int | None = None,
    audio_downsample_chunk_size: int | None = None,
) -> PipelineConfig:
    """Apply request-independent Qwen multimodal preprocessing defaults."""

    updates = {
        "image_min_pixels": _validate_positive_int(
            "--image-min-pixels",
            image_min_pixels,
        ),
        "image_max_pixels": _validate_positive_int(
            "--image-max-pixels",
            image_max_pixels,
        ),
        "video_fps": _validate_positive_float("--video-fps", video_fps),
        "video_max_frames": _validate_positive_int(
            "--video-max-frames",
            video_max_frames,
        ),
        "video_min_frames": _validate_positive_int(
            "--video-min-frames",
            video_min_frames,
        ),
        "video_min_pixels": _validate_positive_int(
            "--video-min-pixels",
            video_min_pixels,
        ),
        "video_max_pixels": _validate_positive_int(
            "--video-max-pixels",
            video_max_pixels,
        ),
        "video_total_pixels": _validate_positive_int(
            "--video-total-pixels",
            video_total_pixels,
        ),
        "video_override_max_pixels": video_override_max_pixels,
        "video_seconds_per_chunk": _validate_positive_float(
            "--video-seconds-per-chunk",
            video_seconds_per_chunk,
        ),
        "video_position_id_per_seconds": _validate_positive_float(
            "--video-position-id-per-seconds",
            video_position_id_per_seconds,
        ),
        "audio_target_sr": _validate_positive_int(
            "--audio-target-sr",
            audio_target_sr,
        ),
        "audio_timestamp_interval": _validate_positive_int(
            "--audio-timestamp-interval",
            audio_timestamp_interval,
        ),
        "audio_downsample_times": _validate_nonnegative_int(
            "--audio-downsample-times",
            audio_downsample_times,
        ),
        "audio_downsample_chunk_size": _validate_positive_int(
            "--audio-downsample-chunk-size",
            audio_downsample_chunk_size,
        ),
    }
    updates = {key: value for key, value in updates.items() if value is not None}
    if not updates:
        return pipeline_config

    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name="preprocessing",
        reason="visual preprocessing runtime override",
    )
    for stage in matching_stages:
        if stage.factory not in _QWEN_PREPROCESSING_VIDEO_FACTORIES:
            raise typer.BadParameter(
                "--image-*, --video-* and --audio-* currently support only Qwen Omni "
                "preprocessing; "
                f"stage {stage.name!r} uses factory {stage.factory!r}"
            )

    for stage in matching_stages:
        # 中文说明：这些值是服务级默认多模态预处理参数，放在 runtime
        # schema 中，再由 runtime_arg_map 转成 preprocessor factory args。
        for key, value in updates.items():
            if key not in stage.runtime_arg_map:
                raise typer.BadParameter(
                    f"--{key.replace('_', '-')} is not supported by "
                    f"preprocessing stage {stage.name!r}"
                )
            setattr(stage.runtime, key, value)
    return pipeline_config


def apply_qwen35_max_model_len_cli_override(
    pipeline_config: PipelineConfig,
    *,
    max_model_len: int | None,
    thinker_max_seq_len: int | None,
) -> PipelineConfig:
    if thinker_max_seq_len is not None:
        seq_len = _validate_positive_int(
            "--thinker-max-seq-len",
            thinker_max_seq_len,
        )
    else:
        seq_len = _validate_positive_int("--max-model-len", max_model_len)
    if seq_len is None:
        return pipeline_config
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--max-model-len is currently supported only for Qwen3.5-Omni "
            "pipelines"
        )

    for stage_name in ("preprocessing", "thinker"):
        matching_stages = _find_matching_stages(
            pipeline_config,
            stage_name=stage_name,
            reason="Qwen3.5 max_model_len runtime override",
        )
        for stage in matching_stages:
            if stage.runtime_arg_map.get("max_seq_len") is None:
                raise typer.BadParameter(
                    f"Stage {stage_name!r} cannot accept max_seq_len override"
                )
            # 中文说明：vLLM 的 max_model_len 是 thinker 的上下文上限；
            # 预处理阶段也需要相同 guard，否则长视频/长音频会先被挡住。
            stage.runtime.max_seq_len = seq_len
    return pipeline_config


def apply_qwen35_max_mm_len_cli_override(
    pipeline_config: PipelineConfig,
    *,
    max_mm_len: int | None,
) -> PipelineConfig:
    max_mm_len = _validate_positive_int("--max-mm-len", max_mm_len)
    if max_mm_len is None:
        return pipeline_config
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--max-mm-len is currently supported only for Qwen3.5-Omni pipelines"
        )

    thinker_stages = _find_matching_stages(
        pipeline_config,
        stage_name="thinker",
        reason="Qwen3.5 max_mm_len validation",
    )
    thinker_stage = thinker_stages[0]
    factory_args = dict(thinker_stage.factory_args or {})
    thinker_max_seq_len = (
        thinker_stage.runtime.max_seq_len
        or factory_args.get("thinker_max_seq_len")
    )
    if thinker_max_seq_len is not None and max_mm_len > int(thinker_max_seq_len):
        raise typer.BadParameter(
            "--max-mm-len must be <= the effective thinker context length"
        )

    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name="preprocessing",
        reason="Qwen3.5 max_mm_len runtime override",
    )
    for stage in matching_stages:
        if stage.factory != _QWEN35_PREPROCESSING_FACTORY:
            raise typer.BadParameter(
                "--max-mm-len currently supports only Qwen3.5 preprocessing"
            )
        if stage.runtime_arg_map.get("max_seq_len") is None:
            raise typer.BadParameter(
                "Qwen3.5 preprocessing stage cannot accept max_seq_len override"
            )
        # 中文说明：vLLM max_mm_len 是多模态输入预算/profiling 上限。
        # SGLang 这里先映射到 preprocessing 的 max_seq_len guard，只限制
        # 预处理输入预算，不降低 thinker 后端 context length。
        stage.runtime.max_seq_len = max_mm_len
    return pipeline_config


def apply_limit_mm_per_prompt_cli_override(
    pipeline_config: PipelineConfig,
    *,
    limit_mm_per_prompt: str | None,
    limit_mm_per_prompt_image: int | None,
    limit_mm_per_prompt_video: int | None,
    limit_mm_per_prompt_audio: int | None,
) -> PipelineConfig:
    updates = _parse_limit_mm_per_prompt(limit_mm_per_prompt)
    for modality, value, flag_name in (
        ("image", limit_mm_per_prompt_image, "--limit-mm-per-prompt-image"),
        ("video", limit_mm_per_prompt_video, "--limit-mm-per-prompt-video"),
        ("audio", limit_mm_per_prompt_audio, "--limit-mm-per-prompt-audio"),
    ):
        value = _validate_nonnegative_int(flag_name, value)
        if value is not None:
            updates[modality] = value
    if not updates:
        return pipeline_config
    if not _is_qwen35_config(pipeline_config):
        raise typer.BadParameter(
            "--limit-mm-per-prompt is currently supported only for "
            "Qwen3.5-Omni pipelines"
        )

    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name="preprocessing",
        reason="Qwen3.5 limit_mm_per_prompt override",
    )
    for stage in matching_stages:
        if stage.factory != _QWEN35_PREPROCESSING_FACTORY:
            raise typer.BadParameter(
                "--limit-mm-per-prompt currently supports only Qwen3.5 "
                f"preprocessing; stage {stage.name!r} uses {stage.factory!r}"
            )
        current = dict(stage.factory_args.get("limit_mm_per_prompt") or {})
        current.update(updates)
        _apply_stage_factory_args_override(
            pipeline_config,
            stage_name=stage.name,
            updates={"limit_mm_per_prompt": current},
            reason="Qwen3.5 limit_mm_per_prompt override",
        )
    return pipeline_config


def apply_talker_model_path_cli_override(
    pipeline_config: PipelineConfig,
    *,
    talker_model_path: str | None,
) -> PipelineConfig:
    if talker_model_path is None:
        return pipeline_config
    normalized_path = talker_model_path.strip()
    if not normalized_path:
        raise typer.BadParameter("--talker-model-path must not be empty")

    stage_name = _resolve_talker_stage(
        pipeline_config,
        flag_name="--talker-model-path",
    )
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason="talker split checkpoint path override",
    )
    root_model_path = getattr(pipeline_config, "model_path", None)
    if not root_model_path:
        raise typer.BadParameter(
            "--talker-model-path requires the pipeline config to have model_path"
        )
    for stage in matching_stages:
        if stage.factory not in _QWEN_TALKER_MODEL_PATH_FACTORIES:
            raise typer.BadParameter(
                "--talker-model-path currently supports only Qwen3.5 Omni "
                f"talker; stage {stage.name!r} uses factory {stage.factory!r}"
            )
    # 中文说明：显式 talker 子目录通常保存未带 "talker." 前缀的权重；
    # root_model_path 仍指向 root checkpoint，用于 tokenizer 和 special token。
    _apply_stage_factory_args_override(
        pipeline_config,
        stage_name=stage_name,
        updates={
            "model_path": normalized_path,
            "root_model_path": root_model_path,
            "weight_prefix": "",
        },
        reason="talker split checkpoint path override",
        flag_name="--talker-model-path",
    )
    return pipeline_config


def apply_code2wav_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    code2wav_model_path: str | None,
    code2wav_torch_compile: str,
    code2wav_torch_compile_first_chunk: str,
    code2wav_sample_rate: int | None,
    code2wav_stream_chunk_size: int | None,
    code2wav_left_context_size: int | None,
    code2wav_codec_eos_token_id: int | None,
    code2wav_dynamic_chunk: str,
    code2wav_dynamic_chunk_sizes: str | None,
    code2wav_dynamic_chunk_steps: str | None,
    code2wav_odeint_method: str | None,
    code2wav_odeint_method_relaxed: str,
    code2wav_batched_chunk: int | None,
    code2wav_frequency: str | None,
    code2wav_dit_quant: str | None,
    code2wav_model_folder: str | None = None,
) -> PipelineConfig:
    """Apply Qwen3.5 code2wav runtime args from the generic serve CLI."""

    compile_mode = _normalize_stage_toggle_mode(
        "code2wav_torch_compile", code2wav_torch_compile
    )
    compile_first_chunk_mode = _normalize_stage_toggle_mode(
        "code2wav_torch_compile_first_chunk",
        code2wav_torch_compile_first_chunk,
    )
    dynamic_chunk_mode = _normalize_stage_toggle_mode(
        "code2wav_dynamic_chunk", code2wav_dynamic_chunk
    )
    odeint_relaxed_mode = _normalize_stage_toggle_mode(
        "code2wav_odeint_method_relaxed",
        code2wav_odeint_method_relaxed,
    )
    factory_updates: dict[str, object] = {}
    runtime_updates: dict[str, object] = {}
    if code2wav_model_path is not None:
        normalized_path = code2wav_model_path.strip()
        if not normalized_path:
            raise typer.BadParameter("--code2wav-model-path must not be empty")
        factory_updates["code2wav_model_path"] = normalized_path
    elif code2wav_model_folder is not None:
        factory_updates["code2wav_model_path"] = _resolve_code2wav_model_folder(
            model_path=pipeline_config.model_path,
            code2wav_model_folder=code2wav_model_folder,
            flag_name="--code2wav-model-folder",
        )
    if compile_mode != "default":
        runtime_updates["code2wav_enable_torch_compile"] = compile_mode == "on"
    if compile_first_chunk_mode != "default":
        runtime_updates["code2wav_enable_torch_compile_first_chunk"] = (
            compile_first_chunk_mode == "on"
        )
    if code2wav_sample_rate is not None:
        if int(code2wav_sample_rate) < 1:
            raise typer.BadParameter("--code2wav-sample-rate must be >= 1")
        runtime_updates["code2wav_sample_rate"] = int(code2wav_sample_rate)
    if code2wav_stream_chunk_size is not None:
        if int(code2wav_stream_chunk_size) < 1:
            raise typer.BadParameter("--code2wav-stream-chunk-size must be >= 1")
        runtime_updates["code2wav_stream_chunk_size"] = int(code2wav_stream_chunk_size)
    if code2wav_left_context_size is not None:
        if int(code2wav_left_context_size) < 0:
            raise typer.BadParameter("--code2wav-left-context-size must be >= 0")
        runtime_updates["code2wav_left_context_size"] = int(code2wav_left_context_size)
    if code2wav_codec_eos_token_id is not None:
        if int(code2wav_codec_eos_token_id) < 0:
            raise typer.BadParameter("--code2wav-codec-eos-token-id must be >= 0")
        runtime_updates["code2wav_codec_eos_token_id"] = int(
            code2wav_codec_eos_token_id
        )
    if dynamic_chunk_mode != "default":
        runtime_updates["code2wav_enable_dynamic_chunk"] = (
            dynamic_chunk_mode == "on"
        )
    if code2wav_dynamic_chunk_sizes is not None:
        runtime_updates["code2wav_dynamic_chunk_sizes"] = _parse_positive_int_sequence(
            "--code2wav-dynamic-chunk-sizes",
            code2wav_dynamic_chunk_sizes,
        )
    if code2wav_dynamic_chunk_steps is not None:
        runtime_updates["code2wav_dynamic_chunk_steps"] = _parse_positive_int_sequence(
            "--code2wav-dynamic-chunk-steps",
            code2wav_dynamic_chunk_steps,
        )
    if code2wav_odeint_method is not None:
        normalized_method = code2wav_odeint_method.strip().lower()
        if normalized_method not in _QWEN35_CODE2WAV_ODEINT_METHODS:
            supported = ", ".join(sorted(_QWEN35_CODE2WAV_ODEINT_METHODS))
            raise typer.BadParameter(
                f"--code2wav-odeint-method must be one of: {supported}"
            )
        runtime_updates["code2wav_odeint_method"] = normalized_method
    if odeint_relaxed_mode != "default":
        runtime_updates["code2wav_odeint_method_relaxed"] = (
            odeint_relaxed_mode == "on"
        )
    if code2wav_batched_chunk is not None:
        if int(code2wav_batched_chunk) < 1:
            raise typer.BadParameter("--code2wav-batched-chunk must be >= 1")
        runtime_updates["code2wav_batched_chunk"] = int(code2wav_batched_chunk)
    if code2wav_frequency is not None:
        normalized_frequency = code2wav_frequency.strip().lower()
        if normalized_frequency not in _QWEN35_CODE2WAV_FREQUENCIES:
            supported = ", ".join(sorted(_QWEN35_CODE2WAV_FREQUENCIES))
            raise typer.BadParameter(
                f"--code2wav-frequency must be one of: {supported}"
            )
        runtime_updates["code2wav_frequency"] = normalized_frequency
    if code2wav_dit_quant is not None:
        normalized_quant = code2wav_dit_quant.strip().lower()
        if normalized_quant not in _QWEN35_CODE2WAV_DIT_QUANTS:
            supported = ", ".join(sorted(_QWEN35_CODE2WAV_DIT_QUANTS))
            raise typer.BadParameter(
                f"--code2wav-dit-quantization must be one of: {supported}"
            )
        runtime_updates["code2wav_dit_quant"] = normalized_quant
    if not factory_updates and not runtime_updates:
        return pipeline_config

    stage_name = type(pipeline_config).code2wav_stage()
    if stage_name is None:
        _raise_unsupported_flag(pipeline_config, "--code2wav-*")
    if factory_updates:
        _apply_stage_factory_args_override(
            pipeline_config,
            stage_name=stage_name,
            updates=factory_updates,
            reason="code2wav model path override",
            flag_name="--code2wav-*",
        )
    if runtime_updates:
        matching_stages = _find_matching_stages(
            pipeline_config,
            stage_name=stage_name,
            reason="code2wav runtime override",
        )
        for stage in matching_stages:
            for key, value in runtime_updates.items():
                if key not in stage.runtime_arg_map:
                    raise typer.BadParameter(
                        f"--{key.replace('_', '-')} is not supported by "
                        f"code2wav stage {stage.name!r}"
                    )
                setattr(stage.runtime, key, value)
    return pipeline_config


def _resolve_async_decode_flag(async_decode: str, enable_async_decode: bool) -> str:
    """Map the deprecated bool ``--enable-async-decode`` onto the ``--async-decode``
    tri-state. The legacy flag only expressed "on", so reject it against an
    explicit ``--async-decode off``."""
    if not enable_async_decode:
        return async_decode
    if async_decode == "off":
        raise typer.BadParameter(
            "--enable-async-decode cannot be combined with --async-decode off"
        )
    logger.warning("--enable-async-decode is deprecated; use --async-decode on.")
    return "on"


def apply_async_decode_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    async_decode: str,
    async_decode_min_batch_size: int | None,
) -> PipelineConfig:
    mode = _normalize_stage_toggle_mode("async_decode", async_decode)
    updates: dict[str, object] = {}
    if mode != "default":
        updates["enable_async_decode"] = mode == "on"
    if async_decode_min_batch_size is not None:
        if int(async_decode_min_batch_size) < 1:
            raise typer.BadParameter("--async-decode-min-batch-size must be >= 1")
        updates["async_decode_min_batch_size"] = int(async_decode_min_batch_size)
    if not updates:
        return pipeline_config
    _apply_stage_factory_args_override(
        pipeline_config,
        stage_name="tts_engine",
        updates=updates,
        reason="async decode override",
        supported_factory=_HIGGS_ASYNC_DECODE_FACTORY,
        flag_name="--async-decode/--async-decode-min-batch-size",
    )
    return pipeline_config


def apply_torch_compile_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    thinker_torch_compile: str,
    talker_torch_compile: str,
    thinker_torch_compile_max_bs: int | None,
    talker_torch_compile_max_bs: int | None,
) -> PipelineConfig:
    thinker_mode = _normalize_stage_toggle_mode(
        "thinker_torch_compile", thinker_torch_compile
    )
    talker_mode = _normalize_stage_toggle_mode(
        "talker_torch_compile", talker_torch_compile
    )
    _apply_stage_torch_compile_override(
        pipeline_config,
        stage_name="thinker",
        mode=thinker_mode,
        max_bs=thinker_torch_compile_max_bs,
    )
    if talker_mode != "default" or talker_torch_compile_max_bs is not None:
        flag_name = (
            "--talker-torch-compile"
            if talker_mode != "default"
            else "--talker-torch-compile-max-bs"
        )
        _apply_stage_torch_compile_override(
            pipeline_config,
            stage_name=_resolve_talker_sglang_stage(
                pipeline_config,
                flag_name=flag_name,
            ),
            mode=talker_mode,
            max_bs=talker_torch_compile_max_bs,
        )
    return pipeline_config


def serve(
    ctx: typer.Context,
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model-path",
            "--model_path",
            "--model",
            help=(
                "The Hugging Face model ID or the path to the model directory. "
                "Required unless --config provides model_path."
            )
        ),
    ] = None,
    thinker_model_path: Annotated[
        str | None,
        typer.Option(
            "--thinker-model",
            "--thinker_model",
            "--thinker-model-path",
            "--thinker_model_path",
            help=(
                "vLLM-compatible thinker model path. With --model-path it must "
                "be <model-path>/thinker and is treated as a no-op."
            ),
        ),
    ] = None,
    config: Annotated[
        str | None, typer.Option(help="Path to a pipeline config file.")
    ] = None,
    vllm_profile: Annotated[
        str | None,
        typer.Option(
            "--vllm-profile",
            "--vllm_profile",
            help=(
                "Load a Qwen3.5-Omni vLLM perf_v2 profile as default CLI "
                "arguments. Explicit CLI values override profile values."
            ),
        ),
    ] = None,
    text_only: Annotated[
        bool,
        typer.Option(
            "--text-only",
            help="Use thinker-only pipeline (1 GPU, no talker/speech output).",
        ),
    ] = False,
    colocate: Annotated[
        bool,
        typer.Option(
            "--colocate",
            help="Run Qwen speech with GPU stages colocated on one GPU.",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option(help="Server bind address (default: 0.0.0.0).")
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "--serve-port",
            "--serve_port",
            help="Server bind port (default: 8000).",
        ),
    ] = 8000,
    model_name: Annotated[
        str, typer.Option(help="Model name for /v1/models (default: pipeline name).")
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens",
            "--max_tokens",
            help=(
                "Service-level default max output tokens for OpenAI requests. "
                "Request-level max_tokens/max_completion_tokens still wins."
            ),
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help=(
                "Service-level default sampling seed for OpenAI requests. "
                "Request-level seed still wins."
            ),
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            help=(
                "Service-level default sampling temperature. Request-level "
                "temperature still wins."
            ),
        ),
    ] = None,
    top_p: Annotated[
        float | None,
        typer.Option(
            "--top-p",
            "--top_p",
            help=(
                "Service-level default top_p. Request-level top_p still wins."
            ),
        ),
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "--top_k",
            help=(
                "Service-level default top_k. Request-level top_k still wins."
            ),
        ),
    ] = None,
    repetition_penalty: Annotated[
        float | None,
        typer.Option(
            "--repetition-penalty",
            "--repetition_penalty",
            help=(
                "Service-level default repetition_penalty. Request-level value "
                "still wins."
            ),
        ),
    ] = None,
    frequency_penalty: Annotated[
        float | None,
        typer.Option(
            "--frequency-penalty",
            "--frequency_penalty",
            help=(
                "Service-level default frequency_penalty. Request-level value "
                "still wins."
            ),
        ),
    ] = None,
    presence_penalty: Annotated[
        float | None,
        typer.Option(
            "--presence-penalty",
            "--presence_penalty",
            help=(
                "Service-level default presence_penalty. Request-level value "
                "still wins."
            ),
        ),
    ] = None,
    voice_type: Annotated[
        str | None,
        typer.Option(
            "--voice-type",
            "--voice_type",
            help=(
                "Service-level default Qwen3.5 talker voice. Request-level "
                "audio.voice/voice_type still wins."
            ),
        ),
    ] = None,
    enable_tn: Annotated[
        bool,
        typer.Option(
            "--enable-tn",
            "--enable_tn",
            "--enable-text-normalization",
            "--enable_text_normalization",
            help="Enable service-level Qwen3.5 text normalization for talker.",
        ),
    ] = False,
    disable_tn: Annotated[
        bool,
        typer.Option(
            "--disable-tn",
            "--disable_tn",
            "--no-enable-tn",
            "--no-enable_tn",
            "--disable-text-normalization",
            "--disable_text_normalization",
            help="Disable service-level Qwen3.5 text normalization for talker.",
        ),
    ] = False,
    mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--mem-fraction-static",
            "--gpu-memory-utilization",
            "--gpu_memory_utilization",
            help=(
                "Set SGLang mem_fraction_static for supported SGLang AR stages. "
                "The gpu-memory-utilization aliases mirror vLLM configs. "
                "If omitted, SGLang chooses the value automatically."
            ),
        ),
    ] = None,
    thinker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--thinker-mem-fraction-static",
            "--thinker-gpu-memory-utilization",
            "--thinker_gpu_memory_utilization",
            help=(
                "Set SGLang mem_fraction_static for the thinker stage. Overrides "
                "--mem-fraction-static for thinker."
            ),
        ),
    ] = None,
    talker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--talker-mem-fraction-static",
            "--talker-gpu-memory-utilization",
            "--talker_gpu_memory_utilization",
            help=(
                "Set SGLang mem_fraction_static for supported talker AR stages. "
                "Overrides --mem-fraction-static for talker."
            ),
        ),
    ] = None,
    max_running_requests: Annotated[
        int | None,
        typer.Option(
            "--max-running-requests",
            "--max_running_requests",
            "--max-num-seqs",
            "--max_num_seqs",
            help=(
                "Set SGLang max_running_requests for supported SGLang AR stages."
            ),
        ),
    ] = None,
    thinker_max_running_requests: Annotated[
        int | None,
        typer.Option(
            "--thinker-max-running-requests",
            "--thinker_max_running_requests",
            help=(
                "Set SGLang max_running_requests for the thinker stage. "
                "Overrides --max-running-requests for thinker."
            ),
        ),
    ] = None,
    talker_max_running_requests: Annotated[
        int | None,
        typer.Option(
            "--talker-max-running-requests",
            "--talker_max_running_requests",
            help=(
                "Set SGLang max_running_requests for supported talker AR stages. "
                "Overrides --max-running-requests for talker."
            ),
        ),
    ] = None,
    max_model_len: Annotated[
        int | None,
        typer.Option(
            "--max-model-len",
            "--max_model_len",
            help=(
                "vLLM-compatible Qwen3.5 thinker context length. "
                "--thinker-max-seq-len overrides it."
            ),
        ),
    ] = None,
    max_seq_len_to_capture: Annotated[
        int | None,
        typer.Option(
            "--max-seq-len-to-capture",
            "--max_seq_len_to_capture",
            help=(
                "vLLM-compatible Qwen3.5 no-op. SGLang uses its own CUDA "
                "graph capture settings."
            ),
        ),
    ] = None,
    compilation_config: Annotated[
        str | None,
        typer.Option(
            "--compilation-config",
            "--compilation_config",
            help=(
                "vLLM-compatible Qwen3.5 compilation_config JSON. "
                "FULL_DECODE_ONLY is accepted as the SGLang default; "
                "none/off disables CUDA graph."
            ),
        ),
    ] = None,
    max_num_batched_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-num-batched-tokens",
            "--max_num_batched_tokens",
            help=(
                "vLLM-compatible prefill batch token limit. Maps to SGLang "
                "max_prefill_tokens and chunked_prefill_size when applicable."
            ),
        ),
    ] = None,
    page_size: Annotated[
        int | None,
        typer.Option(
            "--page-size",
            "--page_size",
            "--block-size",
            "--block_size",
            help=(
                "Set SGLang page_size for supported AR stages. The "
                "block-size aliases mirror vLLM KV cache block_size."
            ),
        ),
    ] = None,
    thinker_max_seq_len: Annotated[
        int | None,
        typer.Option(
            "--thinker-max-seq-len",
            "--thinker_max_seq_len",
            help="Set Qwen3.5 thinker context length.",
        ),
    ] = None,
    encoder_mem_reserve: Annotated[
        float | None,
        typer.Option(
            "--encoder-mem-reserve",
            help=(
                "Subtract this fraction from SGLang's auto-picked Qwen thinker "
                "mem_fraction_static for colocated external encoders. Valid only "
                "when thinker mem_fraction_static is not explicitly pinned."
            ),
        ),
    ] = None,
    cpu_offload_gb: Annotated[
        int | None,
        typer.Option(
            "--cpu-offload-gb",
            "--cpu_offload_gb",
            help="Set SGLang cpu_offload_gb for the thinker stage.",
        ),
    ] = None,
    quantization: Annotated[
        str | None,
        typer.Option(
            "--quantization",
            "--thinker-quantization",
            "--thinker_quantization",
            help="Set SGLang quantization mode for the thinker stage.",
        ),
    ] = None,
    talker_quantization: Annotated[
        str | None,
        typer.Option(
            "--talker-quantization",
            "--talker_quantization",
            help="Set SGLang quantization mode for supported talker AR stages.",
        ),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option(
            "--dtype",
            help=(
                "Set SGLang dtype for supported AR stages. This mirrors vLLM "
                "Qwen3.5 configs such as dtype=bfloat16."
            ),
        ),
    ] = None,
    thinker_dtype: Annotated[
        str | None,
        typer.Option(
            "--thinker-dtype",
            "--thinker_dtype",
            help="Set SGLang dtype for the thinker stage. Overrides --dtype.",
        ),
    ] = None,
    talker_dtype: Annotated[
        str | None,
        typer.Option(
            "--talker-dtype",
            "--talker_dtype",
            help="Set SGLang dtype for supported talker AR stage. Overrides --dtype.",
        ),
    ] = None,
    mamba_ssm_dtype: Annotated[
        str | None,
        typer.Option(
            "--mamba-ssm-dtype",
            "--mamba_ssm_dtype",
            "--mamba-cache-dtype",
            "--mamba_cache_dtype",
            help=(
                "Set SGLang mamba_ssm_dtype for supported AR stages. "
                "The mamba-cache aliases mirror vLLM Qwen3.5 configs."
            ),
        ),
    ] = None,
    mamba_cache_mode: Annotated[
        str | None,
        typer.Option(
            "--mamba-cache-mode",
            "--mamba_cache_mode",
            help=(
                "vLLM-compatible Qwen3.5 cache mode. Only none is currently "
                "accepted; light/all fail early because SGLang has no "
                "equivalent mamba_cache_mode ServerArgs yet."
            ),
        ),
    ] = None,
    kv_transfer_config: Annotated[
        str | None,
        typer.Option(
            "--kv-transfer-config",
            "--kv_transfer_config",
            help=(
                "vLLM-compatible Qwen3.5 KV connector JSON. Empty JSON is "
                "accepted as no-op; non-empty configs fail early until "
                "SGLang KV transfer is mapped explicitly."
            ),
        ),
    ] = None,
    enable_disaggregated_prefilling: Annotated[
        str | None,
        typer.Option(
            "--enable-disaggregated-prefilling",
            "--enable_disaggregated_prefilling",
            help=(
                "vLLM-compatible disaggregated prefill flag. false/0 is "
                "accepted as no-op; true/1 fails early."
            ),
        ),
    ] = None,
    tensor_parallel_size: Annotated[
        int | None,
        typer.Option(
            "--tensor-parallel-size",
            "--tensor_parallel_size",
            help=(
                "vLLM-compatible alias for --thinker-tp-size. Combine with "
                "--thinker-visible-devices for multi-GPU placement."
            ),
        ),
    ] = None,
    distributed_executor_backend: Annotated[
        str | None,
        typer.Option(
            "--distributed-executor-backend",
            "--distributed_executor_backend",
            help="vLLM-compatible executor backend. Only mp is accepted as no-op.",
        ),
    ] = None,
    kv_cache_dtype: Annotated[
        str | None,
        typer.Option(
            "--kv-cache-dtype",
            "--kv_cache_dtype",
            help=(
                "vLLM-compatible KV cache dtype. Only auto is accepted as "
                "no-op; quantized KV cache modes fail early."
            ),
        ),
    ] = None,
    enable_expert_parallel: Annotated[
        str | None,
        typer.Option(
            "--enable-expert-parallel",
            "--enable_expert_parallel",
            help="vLLM-compatible expert parallel flag. true/1 fails early.",
        ),
    ] = None,
    mm_processor_cache_gb: Annotated[
        float | None,
        typer.Option(
            "--mm-processor-cache-gb",
            "--mm_processor_cache_gb",
            help="vLLM-compatible MM cache size. Only 0 is accepted as no-op.",
        ),
    ] = None,
    max_mm_len: Annotated[
        int | None,
        typer.Option(
            "--max-mm-len",
            "--max_mm_len",
            help=(
                "vLLM-compatible multimodal capacity setting. Maps to the "
                "Qwen3.5 preprocessing max_seq_len guard."
            ),
        ),
    ] = None,
    speculative_config: Annotated[
        str | None,
        typer.Option(
            "--speculative-config",
            "--speculative_config",
            help=(
                "vLLM-compatible speculative_config JSON. Empty JSON is "
                "accepted as no-op; Qwen3.5 MTP configs fail early until "
                "SGLang speculative decoding is mapped."
            ),
        ),
    ] = None,
    use_omni_engine: Annotated[
        str | None,
        typer.Option(
            "--use-omni-engine",
            "--use_omni_engine",
            help="vLLM Qwen3.5 launcher marker accepted as no-op.",
        ),
    ] = None,
    use_omni_rpc_engine: Annotated[
        str | None,
        typer.Option(
            "--use-omni-rpc-engine",
            "--use_omni_rpc_engine",
            help="vLLM Qwen3.5 launcher marker accepted as no-op.",
        ),
    ] = None,
    is_thinker: Annotated[
        str | None,
        typer.Option(
            "--is-thinker",
            "--is_thinker",
            help="vLLM Qwen3.5 launcher marker; false is rejected.",
        ),
    ] = None,
    thinker_only_marker: Annotated[
        str | None,
        typer.Option(
            "--thinker-only",
            "--thinker_only",
            help="vLLM Qwen3.5 launcher marker; must match --text-only.",
        ),
    ] = None,
    use_zero_shot: Annotated[
        str | None,
        typer.Option(
            "--use-zero-shot",
            "--use_zero_shot",
            help="vLLM Qwen3.5 launcher marker accepted as no-op.",
        ),
    ] = None,
    skip_mm_profiling: Annotated[
        str | None,
        typer.Option(
            "--skip-mm-profiling",
            "--skip_mm_profiling",
            help="vLLM MM profiling marker accepted as no-op.",
        ),
    ] = None,
    video_needs_metadata: Annotated[
        str | None,
        typer.Option(
            "--video-needs-metadata",
            "--video_needs_metadata",
            help=(
                "vLLM video metadata marker accepted as no-op. Qwen3.5 "
                "preprocessing already requests video metadata."
            ),
        ),
    ] = None,
    override_video_max_pixels: Annotated[
        str | None,
        typer.Option(
            "--override-video-max-pixels",
            "--override_video_max_pixels",
            help=(
                "Use max-mm-len-derived video total_pixels as the effective "
                "Qwen3.5 video resize budget."
            ),
        ),
    ] = None,
    image_min_pixels: Annotated[
        int | None,
        typer.Option(
            "--image-min-pixels",
            "--image_min_pixels",
            help="Set default min image pixels for supported Qwen preprocessing.",
        ),
    ] = None,
    image_max_pixels: Annotated[
        int | None,
        typer.Option(
            "--image-max-pixels",
            "--image_max_pixels",
            help="Set default max image pixels for supported Qwen preprocessing.",
        ),
    ] = None,
    video_fps: Annotated[
        float | None,
        typer.Option(
            "--video-fps",
            "--video_fps",
            help="Set default video FPS for supported Qwen preprocessing stages.",
        ),
    ] = None,
    video_max_frames: Annotated[
        int | None,
        typer.Option(
            "--video-max-frames",
            "--video_max_frames",
            help="Set default max video frames for supported Qwen preprocessing.",
        ),
    ] = None,
    video_min_frames: Annotated[
        int | None,
        typer.Option(
            "--video-min-frames",
            "--video_min_frames",
            help="Set default min video frames for supported Qwen preprocessing.",
        ),
    ] = None,
    video_min_pixels: Annotated[
        int | None,
        typer.Option(
            "--video-min-pixels",
            "--video_min_pixels",
            help="Set default min video pixels for supported Qwen preprocessing.",
        ),
    ] = None,
    video_max_pixels: Annotated[
        int | None,
        typer.Option(
            "--video-max-pixels",
            "--video_max_pixels",
            help="Set default max video pixels for supported Qwen preprocessing.",
        ),
    ] = None,
    video_total_pixels: Annotated[
        int | None,
        typer.Option(
            "--video-total-pixels",
            "--video_total_pixels",
            help="Set default total video pixels for supported Qwen preprocessing.",
        ),
    ] = None,
    video_seconds_per_chunk: Annotated[
        float | None,
        typer.Option(
            "--video-seconds-per-chunk",
            "--video_seconds_per_chunk",
            "--seconds-per-chunk",
            "--seconds_per_chunk",
            help="Set default video seconds_per_chunk for supported Qwen preprocessing.",
        ),
    ] = None,
    video_position_id_per_seconds: Annotated[
        float | None,
        typer.Option(
            "--video-position-id-per-seconds",
            "--video_position_id_per_seconds",
            "--position-id-per-seconds",
            "--position_id_per_seconds",
            help=(
                "Set default video position_id_per_seconds for supported Qwen "
                "preprocessing."
            ),
        ),
    ] = None,
    audio_target_sr: Annotated[
        int | None,
        typer.Option(
            "--audio-target-sr",
            "--audio_target_sr",
            "--audio-sampling-rate",
            "--audio_sampling_rate",
            "--sampling-rate",
            "--sampling_rate",
            help="Set default input audio sampling rate for supported Qwen preprocessing.",
        ),
    ] = None,
    audio_timestamp_interval: Annotated[
        int | None,
        typer.Option(
            "--audio-timestamp-interval",
            "--audio_timestamp_interval",
            "--timestamp-interval",
            "--timestamp_interval",
            help=(
                "Set Qwen3.5 audio timestamp interval used when expanding "
                "audio placeholders."
            ),
        ),
    ] = None,
    audio_downsample_times: Annotated[
        int | None,
        typer.Option(
            "--audio-downsample-times",
            "--audio_downsample_times",
            "--downsample-times",
            "--downsample_times",
            help="Set Qwen3.5 audio downsample_times used by the HF processor.",
        ),
    ] = None,
    audio_downsample_chunk_size: Annotated[
        int | None,
        typer.Option(
            "--audio-downsample-chunk-size",
            "--audio_downsample_chunk_size",
            "--downsample-chunk-size",
            "--downsample_chunk_size",
            help="Set Qwen3.5 audio downsample_chunk_size used by the HF processor.",
        ),
    ] = None,
    limit_mm_per_prompt: Annotated[
        str | None,
        typer.Option(
            "--limit-mm-per-prompt",
            "--limit_mm_per_prompt",
            help=(
                "Set Qwen3.5 multimodal count limits as JSON, e.g. "
                "'{\"image\":2,\"video\":1}'."
            ),
        ),
    ] = None,
    limit_mm_per_prompt_image: Annotated[
        int | None,
        typer.Option(
            "--limit-mm-per-prompt-image",
            "--limit_mm_per_prompt_image",
            help="Set Qwen3.5 image count limit per prompt.",
        ),
    ] = None,
    limit_mm_per_prompt_video: Annotated[
        int | None,
        typer.Option(
            "--limit-mm-per-prompt-video",
            "--limit_mm_per_prompt_video",
            help="Set Qwen3.5 video count limit per prompt.",
        ),
    ] = None,
    limit_mm_per_prompt_audio: Annotated[
        int | None,
        typer.Option(
            "--limit-mm-per-prompt-audio",
            "--limit_mm_per_prompt_audio",
            help="Set Qwen3.5 audio count limit per prompt.",
        ),
    ] = None,
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
    preflight: Annotated[
        bool,
        typer.Option(
            "--preflight",
            help="Run Qwen3.5-Omni local checkpoint preflight before launching.",
        ),
    ] = False,
    xvector_info_path: Annotated[
        list[str] | None,
        typer.Option(
            "--xvector-info-path",
            "--voice-clone-info-path",
            "--voice-clone-path",
            help=(
                "Optional Qwen3.5 voice-clone/xvector_info directory to "
                "preflight. May be passed more than once."
            ),
        ),
    ] = None,
    validate_xvector_pickle: Annotated[
        bool,
        typer.Option(
            "--validate-xvector-pickle",
            help=(
                "Also pickle.load Qwen3.5 voice-clone feat.pkl during "
                "preflight. Use only for trusted local assets."
            ),
        ),
    ] = False,
    disable_mtp: Annotated[
        bool,
        typer.Option(
            "--disable-mtp",
            "--disable_mtp",
            help=(
                "vLLM-compatible no-op for Qwen3.5-Omni. This SGLang path "
                "does not enable thinker MTP."
            ),
        ),
    ] = False,
    thinker_tp_size: Annotated[
        int | None,
        typer.Option(
            "--thinker-tp-size",
            "--thinker_tp_size",
            "--thinker-tensor-parallel-size",
            "--thinker_tensor_parallel_size",
            help="Set tensor parallel size for thinker stage.",
        ),
    ] = None,
    thinker_gpus: Annotated[
        str | None,
        typer.Option(
            "--thinker-gpus",
            "--thinker_gpus",
            help="GPU ids for thinker TP ranks, e.g. '0,1' or '[0, 1]'.",
        ),
    ] = None,
    thinker_visible_devices: Annotated[
        str | None,
        typer.Option(
            "--thinker-visible-devices",
            "--thinker_visible_devices",
            "--thinker-devices",
            help="vLLM-compatible alias for --thinker-gpus.",
        ),
    ] = None,
    talker_gpu: Annotated[
        int | None,
        typer.Option(
            "--talker-gpu",
            "--talker_gpu",
            help="Override GPU id for supported talker stage.",
        ),
    ] = None,
    talker_visible_devices: Annotated[
        str | None,
        typer.Option(
            "--talker-visible-devices",
            "--talker_visible_devices",
            "--talker-devices",
            help="vLLM-compatible single-GPU alias for --talker-gpu.",
        ),
    ] = None,
    talker_model_path: Annotated[
        str | None,
        typer.Option(
            "--talker-model-path",
            "--talker-path",
            "--talker-model",
            "--talker_model_path",
            help="Override Qwen3.5 split talker model directory.",
        ),
    ] = None,
    code2wav_gpu: Annotated[
        int | None,
        typer.Option(
            "--code2wav-gpu",
            "--code2wav_gpu",
            help="Override GPU id for supported code2wav stage.",
        ),
    ] = None,
    code2wav_visible_devices: Annotated[
        str | None,
        typer.Option(
            "--code2wav-visible-devices",
            "--code2wav_visible_devices",
            "--code2wav-devices",
            help="vLLM-compatible single-GPU alias for --code2wav-gpu.",
        ),
    ] = None,
    thinker_cuda_graph: Annotated[
        str,
        typer.Option(
            "--thinker-cuda-graph",
            "--thinker_cuda_graph",
            "--thinker_CUDA_graph",
            help="CUDA graph mode for thinker stage: default|on|off.",
        ),
    ] = "default",
    talker_cuda_graph: Annotated[
        str,
        typer.Option(
            "--talker-cuda-graph",
            "--talker_cuda_graph",
            "--talker_CUDA_graph",
            help="CUDA graph mode for supported SGLang talker stage: default|on|off.",
        ),
    ] = "default",
    enforce_eager: Annotated[
        bool,
        typer.Option(
            "--enforce-eager",
            hidden=True,
            help="vLLM-compatible alias that disables CUDA graph on thinker/talker.",
        ),
    ] = False,
    thinker_enforce_eager: Annotated[
        bool,
        typer.Option(
            "--thinker-enforce-eager",
            hidden=True,
            help="vLLM-compatible alias that disables CUDA graph on thinker.",
        ),
    ] = False,
    talker_enforce_eager: Annotated[
        bool,
        typer.Option(
            "--talker-enforce-eager",
            hidden=True,
            help="vLLM-compatible alias that disables CUDA graph on talker.",
        ),
    ] = False,
    prefix_caching: Annotated[
        str,
        typer.Option(
            "--prefix-caching",
            "--prefix_caching",
            help=(
                "Prefix/radix cache mode for supported SGLang AR stages: "
                "default|on|off."
            ),
        ),
    ] = "default",
    enable_prefix_caching: Annotated[
        bool,
        typer.Option(
            "--enable-prefix-caching",
            hidden=True,
            help="vLLM-compatible alias for '--prefix-caching on'.",
        ),
    ] = False,
    disable_prefix_caching: Annotated[
        bool,
        typer.Option(
            "--disable-prefix-caching",
            "--no-enable-prefix-caching",
            hidden=True,
            help="Alias for '--prefix-caching off'.",
        ),
    ] = False,
    chunked_prefill: Annotated[
        str,
        typer.Option(
            "--chunked-prefill",
            "--chunked_prefill",
            help="Qwen3.5 chunked prefill mode: default|on|off.",
        ),
    ] = "default",
    enable_chunked_prefill: Annotated[
        bool,
        typer.Option(
            "--enable-chunked-prefill",
            hidden=True,
            help="vLLM-compatible alias for '--chunked-prefill on'.",
        ),
    ] = False,
    disable_chunked_prefill: Annotated[
        bool,
        typer.Option(
            "--disable-chunked-prefill",
            "--no-enable-chunked-prefill",
            hidden=True,
            help="Alias for '--chunked-prefill off'.",
        ),
    ] = False,
    talker_partial_start: Annotated[
        str,
        typer.Option(
            "--talker-partial-start",
            "--talker_partial_start",
            help=(
                "Partial-start mode for the Qwen3-Omni talker stage: "
                "default|on|off. When on, the talker begins audio generation "
                "from a partial thinker text stream instead of waiting for the "
                "full text. 'default' uses the pipeline config default."
            ),
        ),
    ] = "default",
    thinker_torch_compile: Annotated[
        str,
        typer.Option(
            "--thinker-torch-compile",
            "--thinker_torch_compile",
            help="torch.compile mode for thinker stage: default|on|off.",
        ),
    ] = "default",
    talker_torch_compile: Annotated[
        str,
        typer.Option(
            "--talker-torch-compile",
            "--talker_torch_compile",
            help=(
                "torch.compile mode for supported SGLang talker stage: "
                "default|on|off."
            ),
        ),
    ] = "default",
    thinker_torch_compile_max_bs: Annotated[
        int | None,
        typer.Option(
            "--thinker-torch-compile-max-bs",
            "--thinker_torch_compile_max_bs",
            help="Override torch_compile_max_bs for thinker stage.",
        ),
    ] = None,
    talker_torch_compile_max_bs: Annotated[
        int | None,
        typer.Option(
            "--talker-torch-compile-max-bs",
            "--talker_torch_compile_max_bs",
            help="Override torch_compile_max_bs for supported SGLang talker stage.",
        ),
    ] = None,
    code2wav_model_path: Annotated[
        str | None,
        typer.Option(
            "--code2wav-model-path",
            "--code2wav-path",
            "--code2wav_model_path",
            help="Override the Qwen code2wav/codec decoder model directory.",
        ),
    ] = None,
    code2wav_model_folder: Annotated[
        str | None,
        typer.Option(
            "--code2wav-model-folder",
            "--code2wav_model_folder",
            "--code2wav-model",
            help=(
                "vLLM-compatible relative code2wav folder under the root "
                "checkpoint. Ignored when --code2wav-model-path is set."
            ),
        ),
    ] = None,
    code2wav_torch_compile: Annotated[
        str,
        typer.Option(
            "--code2wav-torch-compile",
            "--code2wav_torch_compile",
            help="torch.compile mode for supported code2wav stage: default|on|off.",
        ),
    ] = "default",
    code2wav_enable_torch_compile: Annotated[
        bool,
        typer.Option(
            "--code2wav-enable-torch-compile",
            "--code2wav_enable_torch_compile",
            "--enable-torch-compile",
            hidden=True,
            help="Alias for '--code2wav-torch-compile on'.",
        ),
    ] = False,
    code2wav_disable_torch_compile: Annotated[
        bool,
        typer.Option(
            "--no-code2wav-enable-torch-compile",
            "--no-code2wav-torch-compile",
            "--no-enable-torch-compile",
            hidden=True,
            help="Alias for '--code2wav-torch-compile off'.",
        ),
    ] = False,
    code2wav_torch_compile_first_chunk: Annotated[
        str,
        typer.Option(
            "--code2wav-torch-compile-first-chunk",
            "--code2wav_torch_compile_first_chunk",
            help=(
                "torch.compile first-chunk mode for supported code2wav stage: "
                "default|on|off."
            ),
        ),
    ] = "default",
    code2wav_enable_torch_compile_first_chunk: Annotated[
        bool,
        typer.Option(
            "--code2wav-enable-torch-compile-first-chunk",
            "--enable-torch-compile-first-chunk",
            hidden=True,
            help="Alias for '--code2wav-torch-compile-first-chunk on'.",
        ),
    ] = False,
    code2wav_disable_torch_compile_first_chunk: Annotated[
        bool,
        typer.Option(
            "--no-code2wav-enable-torch-compile-first-chunk",
            "--no-code2wav-torch-compile-first-chunk",
            "--no-enable-torch-compile-first-chunk",
            hidden=True,
            help="Alias for '--code2wav-torch-compile-first-chunk off'.",
        ),
    ] = False,
    code2wav_sample_rate: Annotated[
        int | None,
        typer.Option(
            "--code2wav-sample-rate",
            "--code2wav_sample_rate",
            "--sample-rate",
            "--sample_rate",
            help="Override output sample rate for supported code2wav stage.",
        ),
    ] = None,
    code2wav_stream_chunk_size: Annotated[
        int | None,
        typer.Option(
            "--code2wav-stream-chunk-size",
            "--code2wav_stream_chunk_size",
            "--send-chunk-size",
            "--send_chunk_size",
            help="Override streaming codec chunk size for supported code2wav stage.",
        ),
    ] = None,
    code2wav_left_context_size: Annotated[
        int | None,
        typer.Option(
            "--code2wav-left-context-size",
            "--code2wav_left_context_size",
            help="Override left-context size for supported code2wav stage.",
        ),
    ] = None,
    code2wav_codec_eos_token_id: Annotated[
        int | None,
        typer.Option(
            "--code2wav-codec-eos-token-id",
            "--code2wav_codec_eos_token_id",
            help="Override codec EOS token id for supported code2wav stage.",
        ),
    ] = None,
    code2wav_dynamic_chunk: Annotated[
        str,
        typer.Option(
            "--code2wav-dynamic-chunk",
            "--code2wav_dynamic_chunk",
            help="Dynamic chunk mode for supported code2wav stage: default|on|off.",
        ),
    ] = "default",
    code2wav_enable_dynamic_chunk: Annotated[
        bool,
        typer.Option(
            "--code2wav-enable-dynamic-chunk",
            "--code2wav_enable_dynamic_chunk",
            hidden=True,
            help="Alias for '--code2wav-dynamic-chunk on'.",
        ),
    ] = False,
    code2wav_disable_dynamic_chunk: Annotated[
        bool,
        typer.Option(
            "--no-code2wav-enable-dynamic-chunk",
            "--no-code2wav-dynamic-chunk",
            hidden=True,
            help="Alias for '--code2wav-dynamic-chunk off'.",
        ),
    ] = False,
    code2wav_dynamic_chunk_sizes: Annotated[
        str | None,
        typer.Option(
            "--code2wav-dynamic-chunk-sizes",
            "--code2wav_dynamic_chunk_sizes",
            help="Comma/space separated positive chunk sizes for code2wav.",
        ),
    ] = None,
    code2wav_dynamic_chunk_steps: Annotated[
        str | None,
        typer.Option(
            "--code2wav-dynamic-chunk-steps",
            "--code2wav_dynamic_chunk_steps",
            help="Comma/space separated positive dynamic chunk steps.",
        ),
    ] = None,
    code2wav_odeint_method: Annotated[
        str | None,
        typer.Option(
            "--code2wav-odeint-method",
            "--odeint-method",
            "--code2wav_odeint_method",
            help="Override vLLM-compatible code2wav ODE solver method: euler|rk4.",
        ),
    ] = None,
    code2wav_odeint_method_relaxed: Annotated[
        str,
        typer.Option(
            "--code2wav-odeint-method-relaxed-mode",
            "--code2wav_odeint_method_relaxed",
            help="code2wav relaxed ODE method mode: default|on|off.",
        ),
    ] = "default",
    code2wav_enable_odeint_method_relaxed: Annotated[
        bool,
        typer.Option(
            "--code2wav-odeint-method-relaxed",
            "--odeint-method-relaxed",
            hidden=True,
            help="Alias for '--code2wav-odeint-method-relaxed-mode on'.",
        ),
    ] = False,
    code2wav_disable_odeint_method_relaxed: Annotated[
        bool,
        typer.Option(
            "--no-code2wav-odeint-method-relaxed",
            "--no-odeint-method-relaxed",
            hidden=True,
            help="Alias for '--code2wav-odeint-method-relaxed-mode off'.",
        ),
    ] = False,
    code2wav_batched_chunk: Annotated[
        int | None,
        typer.Option(
            "--code2wav-batched-chunk",
            "--batched-chunk",
            "--code2wav_batched_chunk",
            help="Override vLLM-compatible code2wav DIT batched chunk count.",
        ),
    ] = None,
    code2wav_frequency: Annotated[
        str | None,
        typer.Option(
            "--code2wav-frequency",
            "--code2wav_frequency",
            help="Override vLLM-compatible code2wav frequency: 50hz|25hz.",
        ),
    ] = None,
    code2wav_dit_quant: Annotated[
        str | None,
        typer.Option(
            "--code2wav-dit-quantization",
            "--code2wav-dit-quant",
            "--code2wav_dit_quant",
            help="Override vLLM-compatible code2wav DIT quantization: fp8.",
        ),
    ] = None,
    enable_realtime: Annotated[
        bool,
        typer.Option(
            "--enable-realtime",
            "--enable_realtime",
            help="Mount the OpenAI Realtime WebSocket endpoint at /v1/realtime.",
        ),
    ] = False,
    async_decode: Annotated[
        str,
        typer.Option(
            "--async-decode",
            "--async_decode",
            help=(
                "One-step-lookahead async decode for the tts_engine stage: "
                "default|on|off. When on, per-step host collect overlaps the "
                "next GPU forward. 'default' uses the pipeline config default "
                "(on for Higgs TTS). Currently supported by Higgs TTS."
            ),
        ),
    ] = "default",
    enable_async_decode: Annotated[
        bool,
        typer.Option(
            "--enable-async-decode",
            "--enable_async_decode",
            hidden=True,
            help="Deprecated alias for '--async-decode on'.",
        ),
    ] = False,
    async_decode_min_batch_size: Annotated[
        int | None,
        typer.Option(
            "--async-decode-min-batch-size",
            "--async_decode_min_batch_size",
            help=(
                "Decode batches smaller than this bypass the async-decode "
                "lookahead and run synchronously (fast path). Default 2."
            ),
        ),
    ] = None,
) -> None:
    """Serve the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    profile_defaults = _load_qwen35_vllm_profile_defaults(
        vllm_profile,
        disable_mtp_override=disable_mtp,
    )
    text_only = _qwen35_profile_default(
        ctx, profile_defaults, "text_only", text_only, False
    )
    if text_only and profile_defaults:
        profile_defaults = {
            key: value
            for key, value in profile_defaults.items()
            if key not in _QWEN35_TEXT_ONLY_PROFILE_SKIP_DEFAULT_KEYS
        }
    model_path = _qwen35_profile_default(
        ctx, profile_defaults, "model_path", model_path, None
    )
    thinker_model_path = _qwen35_profile_default(
        ctx, profile_defaults, "thinker_model_path", thinker_model_path, None
    )
    host = _qwen35_profile_default(ctx, profile_defaults, "host", host, "0.0.0.0")
    port = _qwen35_profile_default(ctx, profile_defaults, "port", port, 8000)
    mem_fraction_static = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "mem_fraction_static",
        mem_fraction_static,
        None,
    )
    thinker_mem_fraction_static = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "thinker_mem_fraction_static",
        thinker_mem_fraction_static,
        None,
    )
    talker_mem_fraction_static = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "talker_mem_fraction_static",
        talker_mem_fraction_static,
        None,
    )
    max_running_requests = _qwen35_profile_default(
        ctx, profile_defaults, "max_running_requests", max_running_requests, None
    )
    max_model_len = _qwen35_profile_default(
        ctx, profile_defaults, "max_model_len", max_model_len, None
    )
    max_seq_len_to_capture = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "max_seq_len_to_capture",
        max_seq_len_to_capture,
        None,
    )
    compilation_config = _qwen35_profile_default(
        ctx, profile_defaults, "compilation_config", compilation_config, None
    )
    max_num_batched_tokens = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "max_num_batched_tokens",
        max_num_batched_tokens,
        None,
    )
    page_size = _qwen35_profile_default(
        ctx, profile_defaults, "page_size", page_size, None
    )
    quantization = _qwen35_profile_default(
        ctx, profile_defaults, "quantization", quantization, None
    )
    talker_quantization = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "talker_quantization",
        talker_quantization,
        None,
    )
    max_tokens = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "max_tokens",
        max_tokens,
        None,
    )
    seed = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "seed",
        seed,
        None,
    )
    dtype = _qwen35_profile_default(ctx, profile_defaults, "dtype", dtype, None)
    mamba_ssm_dtype = _qwen35_profile_default(
        ctx, profile_defaults, "mamba_ssm_dtype", mamba_ssm_dtype, None
    )
    tensor_parallel_size = _qwen35_profile_default(
        ctx, profile_defaults, "thinker_tp_size", tensor_parallel_size, None
    )
    max_mm_len = _qwen35_profile_default(
        ctx, profile_defaults, "max_mm_len", max_mm_len, None
    )
    override_video_max_pixels = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "override_video_max_pixels",
        override_video_max_pixels,
        None,
    )
    thinker_only_marker = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "thinker_only_marker",
        thinker_only_marker,
        None,
    )
    limit_mm_per_prompt = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "limit_mm_per_prompt",
        limit_mm_per_prompt,
        None,
    )
    video_fps = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "video_fps",
        video_fps,
        None,
    )
    if thinker_gpus is None:
        thinker_visible_devices = _qwen35_profile_default(
            ctx,
            profile_defaults,
            "thinker_visible_devices",
            thinker_visible_devices,
            None,
        )
    if talker_gpu is None:
        talker_visible_devices = _qwen35_profile_default(
            ctx,
            profile_defaults,
            "talker_visible_devices",
            talker_visible_devices,
            None,
        )
    if code2wav_gpu is None:
        code2wav_visible_devices = _qwen35_profile_default(
            ctx,
            profile_defaults,
            "code2wav_visible_devices",
            code2wav_visible_devices,
            None,
        )
    talker_model_path = _qwen35_profile_default(
        ctx, profile_defaults, "talker_model_path", talker_model_path, None
    )
    voice_type = _qwen35_profile_default(
        ctx, profile_defaults, "voice_type", voice_type, None
    )
    enable_tn = _qwen35_profile_default(
        ctx, profile_defaults, "enable_tn", enable_tn, False
    )
    disable_tn = _qwen35_profile_default(
        ctx, profile_defaults, "disable_tn", disable_tn, False
    )
    disable_mtp = _qwen35_profile_default(
        ctx, profile_defaults, "disable_mtp", disable_mtp, False
    )
    enable_prefix_caching = _qwen35_profile_default(
        ctx, profile_defaults, "enable_prefix_caching", enable_prefix_caching, False
    )
    disable_prefix_caching = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "disable_prefix_caching",
        disable_prefix_caching,
        False,
    )
    enable_chunked_prefill = _qwen35_profile_default(
        ctx, profile_defaults, "enable_chunked_prefill", enable_chunked_prefill, False
    )
    disable_chunked_prefill = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "disable_chunked_prefill",
        disable_chunked_prefill,
        False,
    )
    enforce_eager = _qwen35_profile_default(
        ctx, profile_defaults, "enforce_eager", enforce_eager, False
    )
    code2wav_model_path = _qwen35_profile_default(
        ctx, profile_defaults, "code2wav_model_path", code2wav_model_path, None
    )
    code2wav_model_folder = _qwen35_profile_default(
        ctx, profile_defaults, "code2wav_model_folder", code2wav_model_folder, None
    )
    code2wav_enable_torch_compile = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_enable_torch_compile",
        code2wav_enable_torch_compile,
        False,
    )
    code2wav_disable_torch_compile = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_disable_torch_compile",
        code2wav_disable_torch_compile,
        False,
    )
    code2wav_enable_torch_compile_first_chunk = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_enable_torch_compile_first_chunk",
        code2wav_enable_torch_compile_first_chunk,
        False,
    )
    code2wav_disable_torch_compile_first_chunk = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_disable_torch_compile_first_chunk",
        code2wav_disable_torch_compile_first_chunk,
        False,
    )
    code2wav_sample_rate = _qwen35_profile_default(
        ctx, profile_defaults, "code2wav_sample_rate", code2wav_sample_rate, None
    )
    code2wav_stream_chunk_size = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_stream_chunk_size",
        code2wav_stream_chunk_size,
        None,
    )
    code2wav_left_context_size = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_left_context_size",
        code2wav_left_context_size,
        None,
    )
    code2wav_codec_eos_token_id = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_codec_eos_token_id",
        code2wav_codec_eos_token_id,
        None,
    )
    code2wav_enable_dynamic_chunk = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_enable_dynamic_chunk",
        code2wav_enable_dynamic_chunk,
        False,
    )
    code2wav_disable_dynamic_chunk = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_disable_dynamic_chunk",
        code2wav_disable_dynamic_chunk,
        False,
    )
    code2wav_dynamic_chunk_sizes = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_dynamic_chunk_sizes",
        code2wav_dynamic_chunk_sizes,
        None,
    )
    code2wav_dynamic_chunk_steps = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_dynamic_chunk_steps",
        code2wav_dynamic_chunk_steps,
        None,
    )
    code2wav_odeint_method = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_odeint_method",
        code2wav_odeint_method,
        None,
    )
    code2wav_enable_odeint_method_relaxed = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_enable_odeint_method_relaxed",
        code2wav_enable_odeint_method_relaxed,
        False,
    )
    code2wav_disable_odeint_method_relaxed = _qwen35_profile_default(
        ctx,
        profile_defaults,
        "code2wav_disable_odeint_method_relaxed",
        code2wav_disable_odeint_method_relaxed,
        False,
    )
    code2wav_batched_chunk = _qwen35_profile_default(
        ctx, profile_defaults, "code2wav_batched_chunk", code2wav_batched_chunk, None
    )
    code2wav_frequency = _qwen35_profile_default(
        ctx, profile_defaults, "code2wav_frequency", code2wav_frequency, None
    )
    code2wav_dit_quant = _qwen35_profile_default(
        ctx, profile_defaults, "code2wav_dit_quant", code2wav_dit_quant, None
    )

    model_path = _resolve_qwen35_thinker_model_alias(
        config=config,
        model_path=model_path,
        thinker_model_path=thinker_model_path,
    )

    _validate_colocate_cli_request(
        colocate=colocate,
        config=config,
        text_only=text_only,
    )

    # --- Resolve config ---
    if config:
        config_manager = ConfigManager.from_file(config)
    elif text_only:
        if model_path is None:
            raise typer.BadParameter("--model-path is required unless --config is set")
        config_manager = ConfigManager.from_model_path(model_path, variant="text")
    else:
        if model_path is None:
            raise typer.BadParameter("--model-path is required unless --config is set")
        config_manager = ConfigManager.from_model_path(model_path)

    # we use ctx to capture the arguments that are used to modify the configuration on the fly
    # we do expect the extra arguments to be pairs of names and values
    extra_args = config_manager.parse_extra_args(ctx.args)
    merged_config = config_manager.merge_config(extra_args)
    if model_path is not None:
        merged_config = merged_config.model_copy(update={"model_path": model_path})
    if colocate:
        _validate_colocate_config(merged_config)
    _validate_qwen35_disable_mtp_request(
        merged_config,
        disable_mtp=disable_mtp,
    )
    _validate_qwen35_max_seq_len_to_capture_request(
        merged_config,
        max_seq_len_to_capture=max_seq_len_to_capture,
    )
    _validate_qwen35_speculative_config_request(
        merged_config,
        speculative_config=speculative_config,
    )
    _validate_qwen35_vllm_engine_profile_request(
        merged_config,
        distributed_executor_backend=distributed_executor_backend,
        kv_cache_dtype=kv_cache_dtype,
        enable_expert_parallel=enable_expert_parallel,
        mm_processor_cache_gb=mm_processor_cache_gb,
        max_mm_len=max_mm_len,
        use_omni_engine=use_omni_engine,
        use_omni_rpc_engine=use_omni_rpc_engine,
        use_zero_shot=use_zero_shot,
        skip_mm_profiling=skip_mm_profiling,
        video_needs_metadata=video_needs_metadata,
        override_video_max_pixels=override_video_max_pixels,
        is_thinker=is_thinker,
        thinker_only_marker=thinker_only_marker,
        text_only=text_only,
    )
    if tensor_parallel_size is not None:
        if tensor_parallel_size < 1:
            raise typer.BadParameter("--tensor-parallel-size must be >= 1")
        if thinker_tp_size is not None and thinker_tp_size != tensor_parallel_size:
            raise typer.BadParameter(
                "--tensor-parallel-size cannot conflict with --thinker-tp-size"
            )
        thinker_tp_size = tensor_parallel_size
    merged_config = apply_mem_fraction_cli_overrides(
        merged_config,
        mem_fraction_static=mem_fraction_static,
        thinker_mem_fraction_static=thinker_mem_fraction_static,
        talker_mem_fraction_static=talker_mem_fraction_static,
    )
    merged_config = apply_max_running_requests_cli_overrides(
        merged_config,
        max_running_requests=max_running_requests,
        thinker_max_running_requests=thinker_max_running_requests,
        talker_max_running_requests=talker_max_running_requests,
    )
    merged_config = apply_encoder_mem_reserve_cli_override(
        merged_config,
        encoder_mem_reserve=encoder_mem_reserve,
        mem_fraction_static=mem_fraction_static,
        thinker_mem_fraction_static=thinker_mem_fraction_static,
    )
    merged_config = apply_thinker_server_args_cli_overrides(
        merged_config,
        cpu_offload_gb=cpu_offload_gb,
        quantization=quantization,
    )
    merged_config = apply_talker_server_args_cli_overrides(
        merged_config,
        talker_quantization=talker_quantization,
    )
    merged_config = apply_qwen35_max_model_len_cli_override(
        merged_config,
        max_model_len=max_model_len,
        thinker_max_seq_len=thinker_max_seq_len,
    )
    merged_config = apply_qwen35_max_mm_len_cli_override(
        merged_config,
        max_mm_len=max_mm_len,
    )
    merged_config = apply_preprocessing_video_cli_overrides(
        merged_config,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        video_fps=video_fps,
        video_max_frames=video_max_frames,
        video_min_frames=video_min_frames,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_total_pixels=video_total_pixels,
        video_override_max_pixels=(
            _parse_qwen35_vllm_bool_value(
                "--override-video-max-pixels",
                override_video_max_pixels,
            )
            if override_video_max_pixels is not None
            else None
        ),
        video_seconds_per_chunk=video_seconds_per_chunk,
        video_position_id_per_seconds=video_position_id_per_seconds,
        audio_target_sr=audio_target_sr,
        audio_timestamp_interval=audio_timestamp_interval,
        audio_downsample_times=audio_downsample_times,
        audio_downsample_chunk_size=audio_downsample_chunk_size,
    )
    merged_config = apply_limit_mm_per_prompt_cli_override(
        merged_config,
        limit_mm_per_prompt=limit_mm_per_prompt,
        limit_mm_per_prompt_image=limit_mm_per_prompt_image,
        limit_mm_per_prompt_video=limit_mm_per_prompt_video,
        limit_mm_per_prompt_audio=limit_mm_per_prompt_audio,
    )
    if thinker_visible_devices is not None:
        if thinker_gpus is not None:
            raise typer.BadParameter(
                "--thinker-visible-devices cannot be combined with --thinker-gpus"
            )
        thinker_gpus = thinker_visible_devices
    if talker_visible_devices is not None:
        if talker_gpu is not None:
            raise typer.BadParameter(
                "--talker-visible-devices cannot be combined with --talker-gpu"
            )
        talker_gpu = _parse_single_gpu_visible_device(
            "--talker-visible-devices",
            talker_visible_devices,
        )
    if code2wav_visible_devices is not None:
        if code2wav_gpu is not None:
            raise typer.BadParameter(
                "--code2wav-visible-devices cannot be combined with --code2wav-gpu"
            )
        code2wav_gpu = _parse_single_gpu_visible_device(
            "--code2wav-visible-devices",
            code2wav_visible_devices,
        )
    merged_config = apply_parallelism_cli_overrides(
        merged_config,
        thinker_tp_size=thinker_tp_size,
        thinker_gpus=thinker_gpus,
        talker_gpu=talker_gpu,
        code2wav_gpu=code2wav_gpu,
    )
    merged_config = apply_talker_model_path_cli_override(
        merged_config,
        talker_model_path=talker_model_path,
    )
    merged_config = apply_cuda_graph_cli_overrides(
        merged_config,
        thinker_cuda_graph=thinker_cuda_graph,
        talker_cuda_graph=talker_cuda_graph,
    )
    merged_config = apply_vllm_ar_server_args_cli_overrides(
        merged_config,
        prefix_caching=_resolve_toggle_alias_flags(
            flag_name="prefix_caching",
            mode=prefix_caching,
            enable_alias=enable_prefix_caching,
            disable_alias=disable_prefix_caching,
        ),
        chunked_prefill=_resolve_toggle_alias_flags(
            flag_name="chunked_prefill",
            mode=chunked_prefill,
            enable_alias=enable_chunked_prefill,
            disable_alias=disable_chunked_prefill,
        ),
        enforce_eager=enforce_eager,
        thinker_enforce_eager=thinker_enforce_eager,
        talker_enforce_eager=talker_enforce_eager,
        dtype=dtype,
        thinker_dtype=thinker_dtype,
        talker_dtype=talker_dtype,
        mamba_ssm_dtype=mamba_ssm_dtype,
        mamba_cache_mode=mamba_cache_mode,
        kv_transfer_config=kv_transfer_config,
        enable_disaggregated_prefilling=enable_disaggregated_prefilling,
        max_num_batched_tokens=max_num_batched_tokens,
        page_size=page_size,
        compilation_config=compilation_config,
    )
    merged_config = apply_torch_compile_cli_overrides(
        merged_config,
        thinker_torch_compile=thinker_torch_compile,
        talker_torch_compile=talker_torch_compile,
        thinker_torch_compile_max_bs=thinker_torch_compile_max_bs,
        talker_torch_compile_max_bs=talker_torch_compile_max_bs,
    )
    merged_config = apply_code2wav_cli_overrides(
        merged_config,
        code2wav_model_path=code2wav_model_path,
        code2wav_torch_compile=_resolve_toggle_alias_flags(
            flag_name="code2wav_torch_compile",
            mode=code2wav_torch_compile,
            enable_alias=code2wav_enable_torch_compile,
            disable_alias=code2wav_disable_torch_compile,
        ),
        code2wav_torch_compile_first_chunk=_resolve_toggle_alias_flags(
            flag_name="code2wav_torch_compile_first_chunk",
            mode=code2wav_torch_compile_first_chunk,
            enable_alias=code2wav_enable_torch_compile_first_chunk,
            disable_alias=code2wav_disable_torch_compile_first_chunk,
        ),
        code2wav_sample_rate=code2wav_sample_rate,
        code2wav_stream_chunk_size=code2wav_stream_chunk_size,
        code2wav_left_context_size=code2wav_left_context_size,
        code2wav_codec_eos_token_id=code2wav_codec_eos_token_id,
        code2wav_dynamic_chunk=_resolve_toggle_alias_flags(
            flag_name="code2wav_dynamic_chunk",
            mode=code2wav_dynamic_chunk,
            enable_alias=code2wav_enable_dynamic_chunk,
            disable_alias=code2wav_disable_dynamic_chunk,
        ),
        code2wav_dynamic_chunk_sizes=code2wav_dynamic_chunk_sizes,
        code2wav_dynamic_chunk_steps=code2wav_dynamic_chunk_steps,
        code2wav_odeint_method=code2wav_odeint_method,
        code2wav_odeint_method_relaxed=_resolve_toggle_alias_flags(
            flag_name="code2wav_odeint_method_relaxed",
            mode=code2wav_odeint_method_relaxed,
            enable_alias=code2wav_enable_odeint_method_relaxed,
            disable_alias=code2wav_disable_odeint_method_relaxed,
        ),
        code2wav_batched_chunk=code2wav_batched_chunk,
        code2wav_frequency=code2wav_frequency,
        code2wav_dit_quant=code2wav_dit_quant,
        code2wav_model_folder=code2wav_model_folder,
    )
    merged_config = apply_async_decode_cli_overrides(
        merged_config,
        async_decode=_resolve_async_decode_flag(async_decode, enable_async_decode),
        async_decode_min_batch_size=async_decode_min_batch_size,
    )
    merged_config = apply_partial_start_cli_overrides(
        merged_config,
        talker_partial_start=talker_partial_start,
    )

    if preflight:
        _run_qwen35_preflight_or_raise(
            merged_config,
            code2wav_model_path=code2wav_model_path,
            xvector_info_paths=tuple(xvector_info_path or ()),
            validate_xvector_pickle=validate_xvector_pickle,
        )

    if _should_print_merged_config(colocate=colocate, log_level=log_level):
        _print_merged_config(merged_config)

    default_generation_params: dict[str, object] = {}
    if _is_qwen35_speech_config(merged_config):
        # 中文说明：通用 serve 的 Qwen3.5 speech/colocated 路径也对齐
        # vLLM qwen_omni_v35_server.py 和 speech launcher 的服务级默认
        # 生成参数；OpenAI 请求体中的 sampling 参数仍然优先。
        default_generation_params.update(
            {
                "temperature": _QWEN35_DEFAULT_TEMPERATURE,
                "top_k": _QWEN35_DEFAULT_TOP_K,
                "top_p": _QWEN35_DEFAULT_TOP_P,
                "repetition_penalty": _QWEN35_DEFAULT_REPETITION_PENALTY,
                "presence_penalty": _QWEN35_DEFAULT_PRESENCE_PENALTY,
                "max_tokens": _QWEN35_DEFAULT_MAX_TOKENS,
                "seed": _QWEN35_DEFAULT_SEED,
            }
        )
    if max_tokens is not None:
        if int(max_tokens) < 1:
            raise typer.BadParameter("--max-tokens must be >= 1")
        default_generation_params["max_tokens"] = int(max_tokens)
    if seed is not None:
        if int(seed) < 0:
            raise typer.BadParameter("--seed must be >= 0")
        default_generation_params["seed"] = int(seed)
    if temperature is not None:
        if float(temperature) < 0.0:
            raise typer.BadParameter("--temperature must be >= 0")
        default_generation_params["temperature"] = float(temperature)
    if top_p is not None:
        if not 0.0 < float(top_p) <= 1.0:
            raise typer.BadParameter("--top-p must be > 0 and <= 1")
        default_generation_params["top_p"] = float(top_p)
    if top_k is not None:
        if int(top_k) < -1:
            raise typer.BadParameter("--top-k must be >= -1")
        default_generation_params["top_k"] = int(top_k)
    if repetition_penalty is not None:
        if float(repetition_penalty) <= 0.0:
            raise typer.BadParameter("--repetition-penalty must be > 0")
        default_generation_params["repetition_penalty"] = float(repetition_penalty)
    if frequency_penalty is not None:
        default_generation_params["frequency_penalty"] = float(frequency_penalty)
    if presence_penalty is not None:
        default_generation_params["presence_penalty"] = float(presence_penalty)

    default_talker_params: dict[str, object] = {}
    resolved_voice_type = str(voice_type).strip() if voice_type is not None else None
    if voice_type is not None and not resolved_voice_type:
        raise typer.BadParameter("--voice-type must not be empty")
    if resolved_voice_type:
        default_talker_params["voice_type"] = resolved_voice_type
    elif _is_qwen35_speech_config(merged_config):
        # 中文说明：通用 serve 的 colocated/YAML 路径也对齐 vLLM
        # qwen_omni_v35_server.py 的默认音色；请求里的 audio.voice 仍优先。
        default_talker_params["voice_type"] = _QWEN35_DEFAULT_VOICE_TYPE

    tn_default = _resolve_enable_tn_default(bool(enable_tn), bool(disable_tn))
    if tn_default is not None:
        default_talker_params["enable_tn"] = tn_default

    launch_server(
        merged_config,
        host=host,
        port=port,
        model_name=model_name,
        log_level=log_level,
        enable_realtime=enable_realtime,
        default_talker_params=default_talker_params or None,
        default_generation_params=default_generation_params or None,
    )
