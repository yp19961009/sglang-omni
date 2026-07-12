# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni-specific scheduler construction."""

from __future__ import annotations

from typing import Any

# Register qwen3_omni_next AutoConfig before SGLang ModelConfig loads config.json.
import sglang_omni.models.qwen35_omni.hf_config  # noqa: F401


def create_thinker_scheduler(
    server_args: Any,
    gpu_id: int = 0,
    *,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
):
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
    from sglang_omni.models.qwen35_omni.request_builders import (
        make_thinker_scheduler_adapters,
        make_thinker_stream_output_builder,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

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
        model_arch_override="Qwen35OmniNextThinkerForCausalLM",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = ThinkerModelRunner(model_worker, output_proc)

    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    thinker_config = model_config.hf_config.thinker_config
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        thinker_config=thinker_config,
    )
    stream_output_builder = make_thinker_stream_output_builder()

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
    feedback_enabled: bool = True,
    tp_rank: int = 0,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_partial_start: bool = True,
    partial_start_min_chunks: int = 4,
):
    """Create the Qwen3.5 talker scheduler on the generic feedback runtime."""
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    from sglang_omni.models.qwen3_omni.talker_model_runner import (
        QwenTalkerModelRunner,
    )
    from sglang_omni.models.qwen3_omni.talker_scheduler import (
        configure_talker_server_args,
    )
    from sglang_omni.models.qwen35_omni.request_builders import (
        make_talker_scheduler_adapters,
    )
    from sglang_omni.models.qwen35_omni.talker_scheduler import (
        Qwen35TalkerScheduler,
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
        model_arch_override="Qwen35OmniNextTalker",
        weight_prefix=weight_prefix,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    root_config = getattr(
        model_config,
        "_omni_root_hf_config",
        model_config.hf_config,
    )
    talker_config = root_config.talker_config
    codec_vocab_size = int(talker_config.text_config.vocab_size)
    valid_codec_vocab_size = int(talker_config.code_predictor_config.vocab_size)
    model_config.vocab_size = codec_vocab_size
    runner_config = getattr(model_worker.model_runner, "model_config", None)
    if runner_config is not None and runner_config is not model_config:
        runner_config.vocab_size = codec_vocab_size
    if hasattr(model_worker.model_runner, "sampler"):
        model_worker.model_runner.model._sampler = model_worker.model_runner.sampler
    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()
        model = model_worker.model_runner.model
        if hasattr(model, "init_code_predictor_graphs"):
            model.init_code_predictor_graphs(server_args.cuda_graph_bs)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    tokenizer = get_tokenizer(model_config.model_path, trust_remote_code=True)
    (
        request_builder,
        result_adapter,
        stream_chunk_handler,
        stream_done_handler,
    ) = make_talker_scheduler_adapters(
        tokenizer=tokenizer,
        codec_vocab_size=codec_vocab_size,
        valid_codec_vocab_size=valid_codec_vocab_size,
        model=model_worker.model_runner.model,
        model_path=model_config.model_path,
        root_config=root_config,
        thinker_config=root_config.thinker_config,
    )
    scheduler = Qwen35TalkerScheduler(
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
        im_end_token_id=root_config.im_end_token_id,
        text_chunk_size=4,
        codec_chunk_size=4,
    )
    scheduler._model_runner = QwenTalkerModelRunner(
        model_worker,
        output_proc,
        scheduler.outbox,
        feedback_enabled=feedback_enabled,
    )
    return scheduler
