# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni-specific scheduler construction."""

from __future__ import annotations

import logging
import os
from typing import Any

QWEN3_5_OMNI_THINKER_ARCH_OVERRIDE = "Qwen3OmniNextThinkerForConditionalGeneration"
QWEN3_5_OMNI_TALKER_ARCH_OVERRIDE = "Qwen3OmniNextTalkerModel"
QWEN3_5_OMNI_DEFAULT_CAPTURE_HIDDEN_LAYERS = [0, 24]

logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _subtalker_compile_warmup_batches(server_args: Any) -> list[int]:
    configured = os.environ.get("SGLANG_OMNI_QWEN35_SUBTALKER_WARMUP_BATCHES")
    if configured:
        batches: list[int] = []
        for item in configured.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            size = int(item)
            if size > 0 and size not in batches:
                batches.append(size)
        return batches

    max_running = max(int(getattr(server_args, "max_running_requests", 1) or 1), 1)
    return list(range(1, max_running + 1))


def _warmup_subtalker_code_predictor(
    model: Any,
    server_args: Any,
    *,
    phase: str,
) -> bool:
    if not _env_flag("SGLANG_OMNI_QWEN35_SUBTALKER_COMPILE_WARMUP", default=True):
        return False

    warmup = getattr(model, "warmup_subtalker_code_predictor", None)
    if warmup is None:
        logger.warning(
            "Qwen3.5 talker torch.compile requested, but residual code "
            "predictor warmup hook is unavailable"
        )
        return False

    batches = _subtalker_compile_warmup_batches(server_args)
    warmed_batches = warmup(batch_sizes=batches)
    logger.info(
        "Qwen3.5 residual code predictor %s compile warmup completed "
        "batch_sizes=%s",
        phase,
        warmed_batches,
    )
    return True


def _subtalker_code_predictor_is_compiled(model: Any) -> bool:
    code_predictor = getattr(model, "code_predictor", None)
    return getattr(code_predictor, "_compiled_predict_step", None) is not None


def _load_metadata_config(model_path: str) -> Any:
    from sglang_omni.models.qwen3_5_omni.components.common import (
        load_qwen35_config,
    )

    return load_qwen35_config(model_path)


def _metadata_model_path(model_config: Any, root_model_path: str | None) -> str:
    return root_model_path or model_config.model_path


def _metadata_config(model_config: Any, root_model_path: str | None) -> Any:
    if root_model_path:
        try:
            return _load_metadata_config(root_model_path)
        except Exception as exc:
            logger.warning(
                "failed to load Qwen3.5 root metadata config from %s; "
                "falling back to stage config %s: %s",
                root_model_path,
                model_config.model_path,
                exc,
            )
    return model_config.hf_config


def _resolve_thinker_config(config: Any) -> Any:
    value = getattr(config, "thinker_config", None)
    return value if value is not None else config


def _resolve_talker_config(config: Any) -> Any:
    value = getattr(config, "talker_config", None)
    return value if value is not None else config


def _required_config_value(config: Any, name: str) -> Any:
    value = getattr(config, name, None)
    if value is None:
        raise AttributeError(f"Qwen3.5 config missing required field {name!r}")
    return value


def _optional_config_dict(config: Any, name: str) -> dict[str, Any]:
    value = getattr(config, name, None)
    return value if isinstance(value, dict) else {}


def _maybe_enable_subtalker_torch_compile(model: Any, server_args: Any) -> bool:
    if not bool(getattr(server_args, "enable_torch_compile", False)):
        return False

    hook = getattr(model, "enable_subtalker_torch_compile", None)
    if hook is None:
        code_predictor = getattr(model, "code_predictor", None)
        hook = getattr(code_predictor, "enable_torch_compile", None)
    if hook is None:
        logger.warning(
            "Qwen3.5 talker torch.compile requested, but residual code predictor "
            "does not expose a compile hook"
        )
        return False

    compile_mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "default")
    subtalker_compile_mode = compile_mode
    if compile_mode == "reduce-overhead" and not _env_flag(
        "SGLANG_OMNI_QWEN35_SUBTALKER_ALLOW_REDUCE_OVERHEAD",
        default=False,
    ):
        logger.warning(
            "Qwen3.5 residual code predictor torch.compile mode "
            "reduce-overhead is disabled for this path; using default instead"
        )
        subtalker_compile_mode = "default"

    if subtalker_compile_mode == compile_mode:
        hook()
    else:
        hook(mode=subtalker_compile_mode)
    try:
        _warmup_subtalker_code_predictor(
            model,
            server_args,
            phase="pre-cudagraph",
        )
    except Exception:
        logger.exception("Qwen3.5 residual code predictor compile warmup failed")
        if subtalker_compile_mode != "default":
            logger.warning(
                "Falling back Qwen3.5 residual code predictor "
                "torch.compile mode from %s to default",
                subtalker_compile_mode,
            )
            hook(mode="default")
            try:
                _warmup_subtalker_code_predictor(
                    model,
                    server_args,
                    phase="fallback",
                )
            except Exception:
                logger.exception(
                    "Qwen3.5 residual code predictor default compile "
                    "warmup failed after fallback"
                )
    server_args.enable_torch_compile = False
    return True


def _normalize_hidden_layers(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            return None
        return [int(piece) for piece in pieces]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return None


def _ensure_embed_capture_layer(value: Any) -> list[int]:
    hidden_layers = _normalize_hidden_layers(value) or []
    # The talker needs both text/embed hidden states and the intermediate hidden
    # states selected by accept_hidden_layer. Always include layer 0 even when
    # the model config only lists intermediate layers.
    layers = [0]
    for layer in hidden_layers:
        if layer not in layers:
            layers.append(layer)
    return layers


def _resolve_capture_hidden_layers_from_config(
    server_args: Any,
    *,
    root_model_path: str | None = None,
) -> list[int]:
    model_path = root_model_path or getattr(server_args, "model_path", None)
    if not model_path:
        return list(QWEN3_5_OMNI_DEFAULT_CAPTURE_HIDDEN_LAYERS)
    try:
        config = _load_metadata_config(model_path)
    except Exception as exc:
        logger.debug(
            "fall back to default Qwen3.5 hidden capture layers for %s: %s",
            model_path,
            exc,
        )
        return list(QWEN3_5_OMNI_DEFAULT_CAPTURE_HIDDEN_LAYERS)

    talker_config = getattr(config, "talker_config", None)
    hidden_layers = _normalize_hidden_layers(
        getattr(talker_config, "accept_hidden_layer", None)
    )
    # Qwen3.5 talker consumes intermediate thinker hidden states. Prefer the
    # model config and fall back to the current Qwen3-Omni defaults when the
    # field is missing, so the service can still start.
    if hidden_layers:
        return _ensure_embed_capture_layer(hidden_layers)
    return list(QWEN3_5_OMNI_DEFAULT_CAPTURE_HIDDEN_LAYERS)


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    speech_enabled: bool = False,
    capture_hidden_layers: list[int] | None = None,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    root_model_path: str | None = None,
):
    """Create the Qwen3.5-Omni thinker scheduler."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen3_5_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
        should_generate_audio_output,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    if speech_enabled:
        if capture_hidden_layers is None:
            capture_hidden_layers = _resolve_capture_hidden_layers_from_config(
                server_args,
                root_model_path=root_model_path,
            )
        else:
            capture_hidden_layers = _ensure_embed_capture_layer(capture_hidden_layers)
    else:
        capture_hidden_layers = None
    capture_hidden = speech_enabled
    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    defer_cuda_graph_capture = want_cuda_graph and capture_hidden
    if defer_cuda_graph_capture:
        server_args.enable_return_hidden_states = True
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override=QWEN3_5_OMNI_THINKER_ARCH_OVERRIDE,
        capture_hidden_layers=capture_hidden_layers,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    if defer_cuda_graph_capture:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()

    def _should_generate_audio_output(request: Any) -> bool:
        return should_generate_audio_output(request.data.stage_payload)

    output_proc = SGLangOutputProcessor(
        capture_hidden=capture_hidden,
        capture_hidden_layers=capture_hidden_layers,
        model=model_worker.model_runner.model if capture_hidden_layers else None,
        should_emit_hidden=_should_generate_audio_output,
    )

    model_runner = ThinkerModelRunner(
        model_worker,
        output_proc,
        should_capture_hidden=_should_generate_audio_output,
    )

    metadata_model_path = _metadata_model_path(model_config, root_model_path)
    tokenizer = get_tokenizer(
        metadata_model_path,
        trust_remote_code=True,
    )
    root_config = _metadata_config(model_config, root_model_path)
    thinker_config = _resolve_thinker_config(root_config)
    talker_config = getattr(root_config, "talker_config", None)
    required_aux_hidden_key = getattr(talker_config, "accept_hidden_layer", None)
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        thinker_config=thinker_config,
    )
    stream_output_builder = make_thinker_stream_output_builder(
        required_aux_hidden_key=required_aux_hidden_key,
        vocab_size=model_config.vocab_size,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=stream_output_builder,
    )


def create_talker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    weight_prefix: str = "talker.",
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = 5,
    root_model_path: str | None = None,
):
    """Create the Qwen3.5-Omni talker scheduler."""
    del speech_enabled
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.qwen3_5_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
    from sglang_omni.models.qwen3_omni.talker_scheduler import (
        QwenTalkerScheduler,
        configure_talker_server_args,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    want_cuda_graph = configure_talker_server_args(
        server_args,
        feedback_enabled=feedback_enabled,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override=QWEN3_5_OMNI_TALKER_ARCH_OVERRIDE,
        weight_prefix=weight_prefix,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    if hasattr(model_worker.model_runner, "sampler"):
        model_worker.model_runner.model._sampler = model_worker.model_runner.sampler
    _maybe_enable_subtalker_torch_compile(
        model_worker.model_runner.model,
        server_args,
    )
    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()
        if _subtalker_code_predictor_is_compiled(model_worker.model_runner.model):
            try:
                _warmup_subtalker_code_predictor(
                    model_worker.model_runner.model,
                    server_args,
                    phase="post-cudagraph",
                )
            except Exception:
                logger.exception(
                    "Qwen3.5 residual code predictor post-cudagraph warmup failed"
                )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )

    metadata_model_path = _metadata_model_path(model_config, root_model_path)
    tokenizer = get_tokenizer(
        metadata_model_path,
        trust_remote_code=True,
    )
    root_config = _metadata_config(model_config, root_model_path)
    stage_config = model_config.hf_config
    thinker_config = getattr(root_config, "thinker_config", None)
    if thinker_config is None:
        thinker_config = getattr(stage_config, "thinker_config", None)
    if thinker_config is None:
        raise AttributeError(
            "Qwen3.5 talker scheduler requires thinker_config metadata. "
            "Pass root_model_path when loading a split talker checkpoint."
        )
    talker_config = (
        getattr(root_config, "talker_config", None)
        or _resolve_talker_config(stage_config)
    )
    talker_language_id = getattr(
        thinker_config,
        "talker_language_id",
        getattr(root_config, "talker_language_id", None),
    )
    talker_assistant_prompt_id_mapping = getattr(
        thinker_config,
        "talker_assistant_prompt_id_mapping",
        getattr(root_config, "talker_assistant_prompt_id_mapping", None),
    )
    codec_vocab_size = talker_config.text_config.vocab_size
    (
        request_builder,
        result_adapter,
        stream_chunk_handler,
        stream_done_handler,
    ) = make_talker_scheduler_adapters(
        tokenizer=tokenizer,
        codec_vocab_size=codec_vocab_size,
        model=model_worker.model_runner.model,
        model_path=model_config.model_path,
        thinker_config=thinker_config,
        required_aux_hidden_key=talker_config.accept_hidden_layer,
        codec_bos_id=talker_config.codec_bos_id,
        codec_eos_id=talker_config.codec_eos_token_id,
        codec_nothink_id=talker_config.codec_nothink_id,
        codec_think_id=getattr(talker_config, "codec_think_id", None),
        codec_think_bos_id=talker_config.codec_think_bos_id,
        codec_think_eos_id=talker_config.codec_think_eos_id,
        codec_pad_id=talker_config.codec_pad_id,
        audio_token_id=thinker_config.audio_token_id,
        image_token_id=thinker_config.image_token_id,
        video_token_id=thinker_config.video_token_id,
        tts_bos_token_id=_required_config_value(root_config, "tts_bos_token_id"),
        tts_eos_token_id=_required_config_value(root_config, "tts_eos_token_id"),
        tts_pad_token_id=_required_config_value(root_config, "tts_pad_token_id"),
        im_start_token_id=_required_config_value(root_config, "im_start_token_id"),
        im_end_token_id=_required_config_value(root_config, "im_end_token_id"),
        nl_token_id=getattr(root_config, "nl_token_id", 198),
        system_token_id=_required_config_value(root_config, "system_token_id"),
        user_token_id=_required_config_value(root_config, "user_token_id"),
        assistant_token_id=_required_config_value(root_config, "assistant_token_id"),
        speaker_map=_optional_config_dict(talker_config, "speaker_id"),
        talker_language_id=talker_language_id,
        talker_assistant_prompt_id_mapping=talker_assistant_prompt_id_mapping,
        speaker_system_prompt_id=getattr(
            talker_config,
            "speaker_system_prompt_id",
            None,
        ),
        max_thinker_to_talker_mm_tokens=getattr(
            root_config,
            "max_thinker_to_talker_mm_tokens",
            getattr(talker_config, "max_thinker_to_talker_mm_tokens", None),
        ),
    )

    scheduler = QwenTalkerScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_chunk_handler=stream_chunk_handler,
        stream_done_handler=stream_done_handler,
        enable_partial_start=enable_partial_start,
        partial_start_min_chunks=partial_start_min_chunks,
        im_end_token_id=_required_config_value(root_config, "im_end_token_id"),
    )
    scheduler._model_runner = QwenTalkerModelRunner(
        model_worker,
        output_proc,
        scheduler.outbox,
        feedback_enabled=feedback_enabled,
    )
    return scheduler
