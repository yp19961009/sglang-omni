# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Qwen3.5-Omni talker and residual code predictor."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniTalker,
    Qwen3OmniMoeTalkerDenseMLP,
    _bind_default_weight_loaders,
    _repeat_kv,
)
from sglang_omni.models.qwen3_omni.quantization import (
    convert_fp8_weight_scale_inv_for_sglang,
)
from sglang_omni.models.qwen35_omni.hf_config import (
    Qwen35OmniNextTalkerConfig,
)
from sglang_omni.vendor.sglang.layers import (
    GemmaRMSNorm,
    QuantizationConfig,
    ReplicatedLinear,
    VocabParallelEmbedding,
    get_rope,
)
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.server_args import get_global_server_args
from sglang_omni.vendor.sglang.utils import add_prefix

logger = logging.getLogger(__name__)


@dataclass
class _CodePredictorGraphEntry:
    graph: torch.cuda.CUDAGraph
    layer0_codes: torch.Tensor
    talker_hidden: torch.Tensor


class _CodePredictorCudaGraphRunner:
    """CUDA graphs for the residual code predictor used after Talker prefill."""

    def __init__(self, model: "Qwen35OmniNextTalker") -> None:
        self._model = model
        self._entries: dict[int, _CodePredictorGraphEntry] = {}
        self._pool = None

    @property
    def has_graphs(self) -> bool:
        return bool(self._entries)

    @torch.inference_mode()
    def capture(self, batch_sizes: Iterable[int]) -> None:
        sizes = sorted({int(size) for size in batch_sizes if int(size) > 0})
        if not sizes:
            return

        device = self._model._output_codes.device
        with torch.cuda.device(device):
            self._pool = torch.cuda.graph_pool_handle()
            for batch_size in reversed(sizes):
                self._capture_one(batch_size, device=device)

    @torch.inference_mode()
    def _capture_one(self, batch_size: int, *, device: torch.device) -> None:
        hidden_size = int(self._model.config.text_config.hidden_size)
        dtype = self._model.model.codec_embedding.weight.dtype
        layer0_codes = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )
        talker_hidden = torch.zeros(
            (batch_size, 1, hidden_size), dtype=dtype, device=device
        )

        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            self._model._code_predictor_forward_eager(
                layer0_codes, talker_hidden
            )
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self._pool,
            capture_error_mode="thread_local",
        ):
            self._model._code_predictor_forward_eager(
                layer0_codes, talker_hidden
            )
        self._entries[batch_size] = _CodePredictorGraphEntry(
            graph=graph,
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
        )
        logger.info(
            "Captured Qwen3.5 code predictor CUDA graph bs=%d", batch_size
        )

    @torch.inference_mode()
    def replay(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        if (
            not layer0_codes.is_cuda
            or layer0_codes.shape[1] != 1
            or talker_hidden.shape[1] != 1
        ):
            return None

        batch_size = int(layer0_codes.shape[0])
        captured_size = next(
            (size for size in sorted(self._entries) if size >= batch_size),
            None,
        )
        if captured_size is None:
            return None

        entry = self._entries[captured_size]
        entry.layer0_codes[:batch_size].copy_(layer0_codes)
        entry.talker_hidden[:batch_size].copy_(talker_hidden)
        if captured_size > batch_size:
            entry.layer0_codes[batch_size:].zero_()
            entry.talker_hidden[batch_size:].zero_()
            self._model._subtalker_frame_positions[
                batch_size:captured_size
            ].zero_()
        entry.graph.replay()
        return (
            self._model._output_codes[:batch_size].unsqueeze(-1),
            self._model._output_embeds[:batch_size].unsqueeze(1),
        )


class Qwen35OmniNextTalkerTextModel(Qwen3NextModel):
    """Qwen3Next backbone with distinct text and codec embedding tables."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.codec_embedding = self.embed_tokens
        self.embed_tokens = VocabParallelEmbedding(
            config.text_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.text_vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )

        max_batch_size = get_global_server_args().max_running_requests
        self._cp_enabled = True
        self._feedback_buffer = torch.zeros(
            max_batch_size,
            config.hidden_size,
            device=self.codec_embedding.weight.device,
            dtype=self.codec_embedding.weight.dtype,
        )
        self._feedback_mask = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=self.codec_embedding.weight.device,
        )

    def get_input_embeddings(self):
        return self.codec_embedding

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
    ):
        if input_embeds is None:
            hidden_states = self.codec_embedding(input_ids)
            if self._cp_enabled:
                batch_size = hidden_states.shape[0]
                feedback_mask = self._feedback_mask[:batch_size]
                hidden_states = torch.where(
                    feedback_mask.unsqueeze(-1),
                    self._feedback_buffer[:batch_size].to(hidden_states.dtype),
                    hidden_states,
                )
                self._feedback_mask[:batch_size] = False
        else:
            hidden_states = input_embeds
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=hidden_states,
        )


class Qwen35CodePredictorAttention(nn.Module):
    """Output-gated attention used by the Qwen3.5 residual code predictor."""

    def __init__(
        self,
        config: Any,
        *,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_size * 2,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ReplicatedLinear(
            config.hidden_size,
            self.kv_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ReplicatedLinear(
            config.hidden_size,
            self.kv_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = ReplicatedLinear(
            self.q_size,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * config.partial_rotary_factor),
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.layer_id = layer_id

    def forward_incremental(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_len: int,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        q_gate, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q_gate = q_gate.view(batch_size, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(batch_size, self.q_size)
        gate = gate.unsqueeze(2)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=None,
            allow_inplace=False,
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=None,
        )

        q = q.view(batch_size, self.num_heads, 1, self.head_dim)
        k = k.view(batch_size, self.num_kv_heads, 1, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, 1, self.head_dim)
        k_cache[:, :, cache_len : cache_len + 1].copy_(k)
        v_cache[:, :, cache_len : cache_len + 1].copy_(v)

        num_kv_groups = self.num_heads // self.num_kv_heads
        cached_k = _repeat_kv(k_cache[:, :, : cache_len + 1], num_kv_groups)
        cached_v = _repeat_kv(v_cache[:, :, : cache_len + 1], num_kv_groups)
        output = F.scaled_dot_product_attention(
            q,
            cached_k,
            cached_v,
            is_causal=False,
        )
        output = output * torch.sigmoid(gate)
        output = output.reshape(batch_size, self.q_size)
        output, _ = self.o_proj(output)
        return output.unsqueeze(1)


class Qwen35CodePredictorLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen35CodePredictorAttention(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3OmniMoeTalkerDenseMLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen35OmniNextTalkerCodePredictor(nn.Module):
    def __init__(
        self,
        config: Qwen35OmniNextTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cp_config = config.code_predictor_config
        self.model = nn.Module()
        self.model.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(cp_config.vocab_size, cp_config.talker_hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )
        self.model.talker_projection = ReplicatedLinear(
            cp_config.talker_hidden_size,
            cp_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("model.talker_projection", prefix),
        )
        self.model.layers = nn.ModuleList(
            [
                Qwen35CodePredictorLayer(
                    cp_config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"model.layers.{layer_id}", prefix),
                )
                for layer_id in range(cp_config.num_hidden_layers)
            ]
        )
        self.model.norm = GemmaRMSNorm(
            cp_config.hidden_size, eps=cp_config.rms_norm_eps
        )
        self.lm_head = nn.ModuleList(
            [
                ReplicatedLinear(
                    cp_config.hidden_size,
                    cp_config.vocab_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix(f"lm_head.{index}", prefix),
                )
                for index in range(config.num_code_groups - 1)
            ]
        )


class _Qwen3NextLoadProxy:
    def __init__(self, model: nn.Module, lm_head: nn.Module, config: Any) -> None:
        self.model = model
        self.lm_head = lm_head
        self.config = config

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield from self.model.named_parameters(prefix="model")
        yield from self.lm_head.named_parameters(prefix="lm_head")


class Qwen35OmniNextTalker(Qwen3OmniTalker):
    """Qwen3.5 talker implementing the generic SGLang talker runner contract."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(config, Qwen35OmniNextTalkerConfig):
            config = Qwen35OmniNextTalkerConfig(**config.talker_config.to_dict())
        self.config = config

        self.hidden_projection = ReplicatedLinear(
            config.thinker_hidden_size,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("hidden_projection", prefix),
        )
        self.model = Qwen35OmniNextTalkerTextModel(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.codec_head = ReplicatedLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("codec_head", prefix),
        )
        self.code_predictor = Qwen35OmniNextTalkerCodePredictor(
            config,
            quant_config=quant_config,
            prefix=add_prefix("code_predictor", prefix),
        )
        self.register_buffer(
            "speaker_codec_embeddings",
            torch.full(
                (
                    config.max_speaker_num,
                    config.num_code_groups,
                    config.speaker_embedding_length,
                ),
                -1,
                dtype=torch.long,
                device=self.model.codec_embedding.weight.device,
            ),
        )

        device = self.model.codec_embedding.weight.device
        talker_hidden_size = config.text_config.hidden_size
        cp_config = config.code_predictor_config
        predictor_len = config.num_code_groups + 1
        max_batch_size = get_global_server_args().max_running_requests
        self._cp_enabled = self.model._cp_enabled
        self._feedback_buffer = self.model._feedback_buffer
        self._feedback_mask = self.model._feedback_mask
        self._predictor_input_buffer = torch.zeros(
            max_batch_size,
            predictor_len,
            talker_hidden_size,
            device=device,
            dtype=self.model.codec_embedding.weight.dtype,
        )
        self._predictor_positions = torch.arange(
            predictor_len, device=device, dtype=torch.long
        )
        self._predictor_k_cache = torch.zeros(
            cp_config.num_hidden_layers,
            max_batch_size,
            cp_config.num_key_value_heads,
            predictor_len,
            cp_config.head_dim,
            device=device,
            dtype=self.model.codec_embedding.weight.dtype,
        )
        self._predictor_v_cache = torch.zeros_like(self._predictor_k_cache)
        self._sampled_token_ids = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self._repetition_mask = torch.zeros(
            max_batch_size,
            config.text_config.vocab_size,
            dtype=torch.bool,
            device=device,
        )
        self._repetition_penalties = torch.ones(
            max_batch_size,
            1,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._suppress_mask = torch.zeros_like(self._repetition_mask)
        self._sampling_temperatures = torch.ones_like(self._repetition_penalties)
        self._sampling_top_ps = torch.ones(
            max_batch_size,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._sampling_top_ks = torch.ones(
            max_batch_size, dtype=torch.int32, device=device
        )
        self._sampling_min_ps = torch.zeros_like(self._sampling_top_ps)
        self._sampling_seeds = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._subtalker_frame_positions = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )
        self._subtalker_sample_index = 0
        self._output_codes = torch.zeros(
            max_batch_size,
            config.num_code_groups,
            dtype=torch.long,
            device=device,
        )
        self._output_embeds = torch.zeros(
            max_batch_size,
            talker_hidden_size,
            dtype=self.model.codec_embedding.weight.dtype,
            device=device,
        )
        self._code_predictor_graph_runner: (
            _CodePredictorCudaGraphRunner | None
        ) = None
        self._sampler = None
        _bind_default_weight_loaders(self)
        self._cached_params_dict = dict(self.named_parameters())

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_text_embeddings(input_ids)

    def prepare_input_embeds(
        self,
        thinker_embeds: torch.Tensor | None = None,
        thinker_hidden_states: torch.Tensor | None = None,
        is_multimodal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del is_multimodal_mask
        if thinker_hidden_states is not None:
            output, _ = self.hidden_projection(thinker_hidden_states)
            return output
        if thinker_embeds is None:
            raise ValueError("Qwen3.5 talker requires projected input embeddings")
        if thinker_embeds.shape[-1] != self.config.text_config.hidden_size:
            raise ValueError(
                "Qwen3.5 talker text embeddings must already be in talker space"
            )
        return thinker_embeds

    def prepare_decode_buffers(self, requests: list) -> None:
        super().prepare_decode_buffers(requests)
        for row_idx, sched_req in enumerate(requests):
            self._subtalker_frame_positions[row_idx] = int(
                getattr(
                    sched_req.data,
                    "codec_generation_steps",
                    len(sched_req.data.req.output_ids),
                )
            )

    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runner = self._code_predictor_graph_runner
        if runner is not None and not torch.cuda.is_current_stream_capturing():
            result = runner.replay(layer0_codes, talker_hidden)
            if result is not None:
                return result
        return self._code_predictor_forward_eager(layer0_codes, talker_hidden)

    def _code_predictor_forward_eager(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._subtalker_sample_index = 0
        return super().code_predictor_forward(layer0_codes, talker_hidden)

    def init_code_predictor_graphs(self, batch_sizes: Iterable[int]) -> None:
        if not torch.cuda.is_available():
            return
        runner = _CodePredictorCudaGraphRunner(self)
        try:
            runner.capture(batch_sizes)
        except Exception:
            logger.warning(
                "Qwen3.5 code predictor CUDA graph capture failed; using eager",
                exc_info=True,
            )
            return
        if runner.has_graphs:
            self._code_predictor_graph_runner = runner

    def _sample_code_predictor_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample residual codec groups with the request's talker policy."""
        logits = logits[:, -1, :]
        if self._sampler is None:
            return torch.argmax(logits, dim=-1, keepdim=True)

        batch_size = logits.shape[0]
        residual_groups = self.config.num_code_groups - 1
        frame_offset, codebook_index = divmod(
            self._subtalker_sample_index,
            residual_groups,
        )
        self._subtalker_sample_index += 1
        positions = (
            self._subtalker_frame_positions[:batch_size] + frame_offset
        ) * residual_groups + codebook_index
        sampling_info = self._build_static_sampling_info(
            batch_size,
            vocab_size=self.config.code_predictor_config.vocab_size,
        )
        sampled = self._sampler(
            LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=None,
            ),
            sampling_info,
            False,
            [0] * batch_size,
            [[] for _ in range(batch_size)],
            positions,
        )
        if sampled.ndim > 1:
            sampled = sampled.squeeze(-1)
        return sampled.to(dtype=torch.long).unsqueeze(-1)

    def _predictor_forward_one_token(
        self,
        *,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
    ) -> torch.Tensor:
        hidden_states, _ = self.code_predictor.model.talker_projection(token_embeds)
        positions = torch.full(
            (batch_size,),
            cache_len,
            dtype=torch.long,
            device=hidden_states.device,
        )
        hidden_size = hidden_states.shape[-1]
        for layer_idx, layer in enumerate(self.code_predictor.model.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states.reshape(-1, hidden_size))
            normed = normed.reshape(batch_size, 1, hidden_size)
            attention_output = layer.self_attn.forward_incremental(
                normed,
                positions=positions,
                k_cache=self._predictor_k_cache[layer_idx, :batch_size],
                v_cache=self._predictor_v_cache[layer_idx, :batch_size],
                cache_len=cache_len,
            )
            hidden_states = residual + attention_output

            residual = hidden_states
            normed = layer.post_attention_layernorm(
                hidden_states.reshape(-1, hidden_size)
            )
            mlp_output = layer.mlp(normed).reshape(batch_size, 1, hidden_size)
            hidden_states = residual + mlp_output

        hidden_states = self.code_predictor.model.norm(
            hidden_states.reshape(-1, hidden_size)
        )
        return hidden_states.reshape(batch_size, 1, hidden_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        main_weights: list[tuple[str, torch.Tensor]] = []
        remaining_weights: list[tuple[str, torch.Tensor]] = []
        for original_name, loaded_weight in weights:
            if not original_name.startswith("talker."):
                continue
            name = original_name[len("talker.") :]
            if name.startswith("model."):
                main_weights.append((name, loaded_weight))
            elif name.startswith("codec_head."):
                main_weights.append(
                    (name.replace("codec_head.", "lm_head.", 1), loaded_weight)
                )
            else:
                remaining_weights.append((name, loaded_weight))

        proxy = _Qwen3NextLoadProxy(
            self.model, self.codec_head, self.config.text_config
        )
        Qwen3NextForCausalLM.load_weights(proxy, main_weights)

        params = self._cached_params_dict
        for name, loaded_weight in remaining_weights:
            if name == "speaker_codec_embeddings":
                self.speaker_codec_embeddings.copy_(loaded_weight)
                continue

            handled = False
            for param_name, weight_name, shard_id in (
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ):
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params.get(mapped)
                if param is not None:
                    converted = convert_fp8_weight_scale_inv_for_sglang(
                        mapped, loaded_weight
                    )
                    param.weight_loader(param, converted, shard_id)
                    handled = True
                    break
            if handled:
                continue

            param = params.get(name)
            if param is not None:
                converted = convert_fp8_weight_scale_inv_for_sglang(name, loaded_weight)
                param.weight_loader(param, converted)


EntryClass = Qwen35OmniNextTalker
