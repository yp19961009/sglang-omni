# SPDX-License-Identifier: Apache-2.0
"""Residual codec predictor for Qwen3.5-Omni talker.

Qwen3.5 uses a small "subtalker" after the main AR codec token.  vLLM
dev/qwenc_perf_v2 implements it as a compact decoder that receives
``[talker_hidden, layer0_codec_embed]`` and autoregressively predicts the
remaining RVQ groups.  The current sglang-omni environment may not ship the
new ``transformers.models.qwen3_omni_next`` package yet, so this module keeps
that Next-style decoder local instead of depending on a future transformers
release.
"""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang_omni.vendor.sglang.layers import ReplicatedLinear

logger = logging.getLogger(__name__)
_NEXT_MODELING_MODULE = "transformers.models.qwen3_omni_next.modeling_qwen3_omni_next"


class _TensorLinear(nn.Module):
    """ReplicatedLinear wrapper that returns only the projected tensor."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear = ReplicatedLinear(
            input_size,
            output_size,
            bias=bias,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.linear(x)
        return out


def _get_int(config: Any, name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _get_float(config: Any, name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _get_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _activation(name: str):
    if name in {"silu", "swish"}:
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported Qwen3.5 subtalker hidden_act={name!r}")


def _expand_sampling_value(
    value: Any,
    *,
    default: float | int,
    batch_size: int,
    seq_len: int,
    flat_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.full((flat_size,), default, device=device, dtype=dtype)

    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == flat_size:
        return tensor
    if tensor.numel() == batch_size:
        return tensor.repeat_interleave(seq_len)
    if tensor.numel() == 1:
        return tensor.expand(flat_size)
    raise ValueError(
        "subtalker sampling tensor must be scalar, batch-sized, or flat-sized: "
        f"got {tensor.numel()} values for batch={batch_size}, seq={seq_len}"
    )


def _expand_optional_seed_value(
    value: Any,
    *,
    batch_size: int,
    seq_len: int,
    flat_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None

    tensor = torch.as_tensor(value, device=device, dtype=torch.long).reshape(-1)
    if tensor.numel() == flat_size:
        return tensor
    if tensor.numel() == batch_size:
        return tensor.repeat_interleave(seq_len)
    if tensor.numel() == 1:
        return tensor.expand(flat_size)
    raise ValueError(
        "subtalker seed tensor must be scalar, batch-sized, or flat-sized: "
        f"got {tensor.numel()} values for batch={batch_size}, seq={seq_len}"
    )


def _apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    top_k = top_k.to(device=logits.device, dtype=torch.long).clamp(min=0)
    active = (top_k > 0) & (top_k < vocab_size)
    if not bool(active.any()):
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
    remove = ranks >= top_k.clamp(max=vocab_size).unsqueeze(1)
    remove = remove & active.unsqueeze(1)
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(1, sorted_indices, sorted_logits)


def _apply_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    top_p = top_p.to(device=logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    active = top_p < 1.0
    if not bool(active.any()):
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p.unsqueeze(1)
    shifted = remove.clone()
    shifted[:, 1:] = remove[:, :-1]
    shifted[:, 0] = False
    remove = shifted & active.unsqueeze(1)
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(1, sorted_indices, sorted_logits)


def _apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    min_p = min_p.to(device=logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    active = min_p > 0
    if not bool(active.any()):
        return logits

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    max_probs = probs.max(dim=-1).values
    threshold = min_p.unsqueeze(1) * max_probs.unsqueeze(1)
    remove = (probs < threshold) & active.unsqueeze(1)
    return logits.masked_fill(remove, float("-inf"))


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor | None = None,
    seed: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = logits.to(torch.float32)
    temperature = temperature.to(device=logits.device, dtype=torch.float32)
    greedy = temperature <= 0
    greedy_ids = torch.argmax(logits, dim=-1)
    if bool(greedy.all()):
        return greedy_ids

    scaled = logits / temperature.clamp_min(1.0e-6).unsqueeze(1)
    scaled = _apply_top_k(scaled, top_k)
    scaled = _apply_top_p(scaled, top_p)
    if min_p is not None:
        scaled = _apply_min_p(scaled, min_p)

    probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
    if seed is None:
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        seed = seed.to(device=logits.device, dtype=torch.long).reshape(-1)
        sampled = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
        for row_idx in range(logits.shape[0]):
            if bool(greedy[row_idx]):
                sampled[row_idx] = greedy_ids[row_idx]
                continue
            if int(seed[row_idx].item()) < 0:
                sampled[row_idx] = torch.multinomial(
                    probs[row_idx],
                    num_samples=1,
                )[0]
                continue
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(int(seed[row_idx].item()))
            sampled[row_idx] = torch.multinomial(
                probs[row_idx],
                num_samples=1,
                generator=generator,
            )[0]
    return torch.where(greedy, greedy_ids, sampled)


def _bind_default_weight_loaders(module: nn.Module) -> None:
    for param in module.parameters():
        if not hasattr(param, "weight_loader"):
            param.weight_loader = default_weight_loader


class _NextKVCache:
    """Lightweight KV cache for local Qwen3.5 subtalker decoding."""

    def __init__(self, num_layers: int) -> None:
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = key_states
            self._values[layer_idx] = value_states
        else:
            self._keys[layer_idx] = torch.cat(
                [self._keys[layer_idx], key_states],
                dim=2,
            )
            self._values[layer_idx] = torch.cat(
                [self._values[layer_idx], value_states],
                dim=2,
            )
        return self._keys[layer_idx], self._values[layer_idx]


def _probe_next_predictor() -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(_NEXT_MODELING_MODULE)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    class_name = "Qwen3OmniNextTalkerCodePredictorModelForConditionalGeneration"
    if getattr(module, class_name, None) is None:
        return False, f"{_NEXT_MODELING_MODULE}.{class_name} missing"
    return True, None


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device)
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _resolve_rotary_dim(config: Any, head_dim: int) -> int:
    rope_parameters = _get_plain_dict(getattr(config, "rope_parameters", None))
    rope_scaling = _get_plain_dict(getattr(config, "rope_scaling", None))
    factor = getattr(config, "partial_rotary_factor", None)
    if factor is None:
        factor = rope_parameters.get(
            "partial_rotary_factor",
            rope_scaling.get("partial_rotary_factor"),
        )
    if factor is None:
        return head_dim

    rotary_dim = max(2, int(head_dim * float(factor)))
    if rotary_dim % 2:
        rotary_dim -= 1
    if rotary_dim <= 0 or rotary_dim > head_dim:
        raise ValueError(
            "subtalker partial_rotary_factor must produce a rotary dim in "
            f"[2, head_dim], got factor={factor!r}, head_dim={head_dim}"
        )
    return rotary_dim


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat(
        (k_rot, k_pass),
        dim=-1,
    )


class _NextSelfAttention(nn.Module):
    """Qwen3Next-style attention with output gating and QK norm."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = _get_int(config, "hidden_size", 0)
        self.num_heads = _get_int(config, "num_attention_heads", 1)
        self.num_kv_heads = _get_int(
            config,
            "num_key_value_heads",
            self.num_heads,
        )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("subtalker num_attention_heads must divide kv heads")
        self.head_dim = _get_int(
            config,
            "head_dim",
            self.hidden_size // self.num_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary_dim = _resolve_rotary_dim(config, self.head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        bias = bool(getattr(config, "attention_bias", False))
        # q_proj outputs query + gate (hence *2)
        self.q_proj = nn.Linear(self.hidden_size, self.q_size * 2, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=bias)
        eps = _get_float(config, "rms_norm_eps", 1e-6)
        self.q_norm = _RMSNorm(self.head_dim, eps)
        self.k_norm = _RMSNorm(self.head_dim, eps)
        self.rotary_emb = _RotaryEmbedding(
            self.rotary_dim,
            _get_float(config, "rope_theta", 10000.0),
        )

    def _shape(
        self,
        tensor: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        bsz, seq_len, _ = tensor.shape
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        past_key_values: _NextKVCache | None = None,
        layer_idx: int | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # q_proj outputs [query, gate] interleaved per head
        qg = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim * 2)
        query_states, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(bsz, seq_len, self.q_size)

        # QK normalization
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = self._shape(self.v_proj(hidden_states), self.num_kv_heads)

        cos, sin = self.rotary_emb(
            position_ids,
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        query_states, key_states = _apply_rotary(
            query_states,
            key_states,
            cos,
            sin,
            self.rotary_dim,
        )
        if use_cache and past_key_values is not None:
            if layer_idx is None:
                raise ValueError("subtalker KV cache requires layer_idx")
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                layer_idx,
            )
        if self.num_kv_groups != 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)
        key_len = key_states.shape[2]
        past_len = key_len - seq_len
        query_positions = torch.arange(
            seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(1) + past_len
        key_positions = torch.arange(
            key_len,
            dtype=torch.long,
            device=hidden_states.device,
        ).unsqueeze(0)
        causal_mask = key_positions > query_positions
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.q_size)
        # Output gating
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class _NextMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = _get_int(config, "hidden_size", 0)
        intermediate_size = _get_int(config, "intermediate_size", hidden_size * 4)
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = _activation(getattr(config, "hidden_act", "silu"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class _NextDecoderLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = _get_int(config, "hidden_size", 0)
        eps = _get_float(config, "rms_norm_eps", 1e-6)
        self.self_attn = _NextSelfAttention(config)
        self.mlp = _NextMLP(config)
        self.input_layernorm = _RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = _RMSNorm(hidden_size, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        past_key_values: _NextKVCache | None = None,
        layer_idx: int | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            position_ids,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class _NextPredictorModel(nn.Module):
    """Local Next-style decoder with HF/vLLM-compatible parameter names."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.num_code_groups = _get_int(config, "num_code_groups", 1)
        self.vocab_size = _get_int(config, "vocab_size", 0)
        self.talker_hidden_size = _get_int(
            config,
            "talker_hidden_size",
            _get_int(config, "hidden_size", 0),
        )
        self.hidden_size = _get_int(config, "hidden_size", self.talker_hidden_size)
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(self.vocab_size, self.talker_hidden_size)
                for _ in range(max(self.num_code_groups - 1, 0))
            ]
        )
        self.talker_projection = _TensorLinear(
            self.talker_hidden_size,
            self.hidden_size,
            prefix="model.talker_projection",
        )
        self.layers = nn.ModuleList(
            [
                _NextDecoderLayer(config)
                for _ in range(_get_int(config, "num_hidden_layers", 0))
            ]
        )
        self.norm = _RMSNorm(
            self.hidden_size,
            _get_float(config, "rms_norm_eps", 1e-6),
        )

    def _default_position_ids(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0).expand(
            batch_size,
            seq_len,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        *,
        past_key_values: _NextKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = inputs_embeds.shape
        hidden_states = self.talker_projection(inputs_embeds)
        if position_ids is None:
            position_ids = self._default_position_ids(
                batch_size,
                seq_len,
                hidden_states.device,
            )
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
                use_cache=use_cache,
            )
        return self.norm(hidden_states)


class Qwen35ResidualCodePredictor(nn.Module):
    """Generate residual RVQ groups after the main talker codec token."""

    def __init__(self, config: Any | None) -> None:
        super().__init__()
        self.config = config
        self.num_code_groups = _get_int(config, "num_code_groups", 1)
        self.vocab_size = _get_int(config, "vocab_size", 0)
        hf_next_available, hf_next_error = _probe_next_predictor()
        self.hf_next_available = hf_next_available
        self.hf_next_error = hf_next_error
        self.exact_next_available = True
        self.exact_next_error = None
        self.implementation = "sglang_next_decoder"
        if not hf_next_available:
            logger.warning(
                "Qwen3.5-Omni HF Next subtalker is unavailable; using "
                "local SGLang Next decoder predictor. import_error=%s",
                hf_next_error,
            )
        self.model = _NextPredictorModel(config)
        self.lm_head = nn.ModuleList(
            [
                _TensorLinear(
                    self.model.hidden_size,
                    self.vocab_size,
                    bias=False,
                    prefix=f"lm_head.{idx}",
                )
                for idx in range(max(self.num_code_groups - 1, 0))
            ]
        )
        _bind_default_weight_loaders(self)

    def generate(
        self,
        *,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        layer0_embed_fn: nn.Module,
        pad_id: int,
        temperature: Any | None = None,
        top_k: Any | None = None,
        top_p: Any | None = None,
        min_p: Any | None = None,
        seed: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate residual code groups and summed codec embeddings.

        中文说明：这里按 vLLM dev/qwenc_perf_v2 的 subtalker 契约执行：
        第一个输入 token 是 talker hidden，第二个输入 token 是主 AR
        预测出的第 0 组 codec embedding；之后每预测一组 residual codec，
        就把该组 embedding 追加到 decoder 输入里继续预测下一组。
        """
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)

        batch_size, seq_len = layer0_codes.shape
        device = layer0_codes.device
        flat_size = batch_size * seq_len
        sampling_temperature = _expand_sampling_value(
            temperature,
            default=0.0,
            batch_size=batch_size,
            seq_len=seq_len,
            flat_size=flat_size,
            device=device,
            dtype=torch.float32,
        )
        sampling_top_k = _expand_sampling_value(
            top_k,
            default=0,
            batch_size=batch_size,
            seq_len=seq_len,
            flat_size=flat_size,
            device=device,
            dtype=torch.long,
        )
        sampling_top_p = _expand_sampling_value(
            top_p,
            default=1.0,
            batch_size=batch_size,
            seq_len=seq_len,
            flat_size=flat_size,
            device=device,
            dtype=torch.float32,
        )
        sampling_min_p = _expand_sampling_value(
            min_p,
            default=0.0,
            batch_size=batch_size,
            seq_len=seq_len,
            flat_size=flat_size,
            device=device,
            dtype=torch.float32,
        )
        sampling_seed = _expand_optional_seed_value(
            seed,
            batch_size=batch_size,
            seq_len=seq_len,
            flat_size=flat_size,
            device=device,
        )
        flat_layer0_codes = layer0_codes.reshape(flat_size)
        flat_talker_hidden = talker_hidden.reshape(flat_size, 1, -1)
        codes = torch.full(
            (flat_size, self.num_code_groups),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        codes[:, 0] = flat_layer0_codes

        layer0_embed = layer0_embed_fn(flat_layer0_codes).unsqueeze(1)
        summed_embeddings = layer0_embed.clone()
        inputs_embeds = torch.cat([flat_talker_hidden, layer0_embed], dim=1)
        kv_cache = _NextKVCache(len(self.model.layers))
        position_ids = torch.arange(
            self.num_code_groups + 1,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(flat_size, -1)

        for group_idx in range(self.num_code_groups - 1):
            if group_idx == 0:
                step_inputs = inputs_embeds
                step_positions = position_ids[:, :2]
            else:
                step_inputs = next_input
                step_positions = position_ids[:, group_idx + 1 : group_idx + 2]
            hidden = self.model(
                step_inputs,
                position_ids=step_positions,
                past_key_values=kv_cache,
                use_cache=True,
            )
            logits = self.lm_head[group_idx](hidden)
            next_code = _sample_logits(
                logits[:, -1, :],
                temperature=sampling_temperature,
                top_k=sampling_top_k,
                top_p=sampling_top_p,
                min_p=sampling_min_p,
                seed=(
                    None
                    if sampling_seed is None
                    else sampling_seed + int(group_idx)
                ),
            )
            codes[:, group_idx + 1] = next_code
            next_input = F.embedding(
                next_code,
                self.model.codec_embedding[group_idx].weight,
            ).unsqueeze(1)
            summed_embeddings = summed_embeddings + next_input.to(
                dtype=summed_embeddings.dtype
            )

        codes = codes.view(batch_size, seq_len, self.num_code_groups).transpose(1, 2)
        embeds = summed_embeddings.view(batch_size, seq_len, -1)
        return codes, embeds
