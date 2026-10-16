# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni Next DAC decoder ported from reference implementation.

This file keeps the Qwen reference pure-PyTorch DAC/codec decoder structure so
the Qwen3.5-Omni code2wav stage can load it locally without a runtime
dependency on the Qwen reference package.
"""

import math
import typing as tp
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import torch
import yaml
from torch import Tensor, nn, pow, sin
from torch.nn import Parameter
from torch.nn import functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope: dict = field(default_factory=lambda: {})
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True
    pos_embed_type: str = "rope"
    max_relative_position: int = 128

    def find_multiple(self, n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = self.find_multiple(n_hidden, 256)
        assert self.pos_embed_type in [
            "rope",
            "conformer",
        ], "pos_embed_type must be either 'rope' or 'conformer'"


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type",
                config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # Force float32
            freqs = (
                inv_freq_expanded.float()
                @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        causal_mask = torch.tril(
            torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool)
        )

        if self.config.rope_base == -1:
            # Only compute RoPE frequencies if using RoPE
            if config.pos_embed_type == "rope":
                self.rotary_emb = Qwen3RotaryEmbedding(config.rope)
            else:
                self.rotary_emb= None
        else:
            if config.pos_embed_type == "rope":
                freqs_cis = precompute_freqs_cis(
                    self.config.block_size, self.config.head_dim, self.config.rope_base
                )
                self.register_buffer("freqs_cis", freqs_cis)
            else:
                self.register_buffer("freqs_cis", None)
            self.register_buffer("causal_mask", causal_mask)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def make_mask(
        self,
        max_length: int,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
        if x_lens is None:
            return mask[None, None]

        lengths = x_lens.to(dtype=torch.long).view(-1)
        col_indices = torch.arange(max_length, device=lengths.device)
        valid_keys = col_indices.view(1, 1, 1, -1) < lengths.view(-1, 1, 1, 1)
        return mask.to(lengths.device)[None, None] & valid_keys

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.config.pos_embed_type == "rope":
            if self.config.rope_base == -1:
                freqs_cis = self.rotary_emb(x, input_pos)
            else:
                assert (
                    self.freqs_cis is not None
                ), "RoPE frequencies must be initialized for RoPE positional embedding"
                freqs_cis = self.freqs_cis[input_pos]
        else:
            if self.config.rope_base == -1:
                freqs_cis = None, None
            else:
                freqs_cis = None

        if mask is None:
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., : input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis,
        mask: Tensor,
    ) -> Tensor:
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        )
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.config = config

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim

        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate
        self.pos_embed_type = config.pos_embed_type

        if self.pos_embed_type == "conformer":
            self.max_relative_position = config.max_relative_position
            num_pos_embeddings = 2 * config.max_relative_position + 1
            self.rel_pos_embeddings = nn.Parameter(
                torch.zeros(num_pos_embeddings, self.head_dim)
            )
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def _normalise_positions(
        self,
        input_pos: Optional[Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        if input_pos is None:
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            return positions.unsqueeze(0).expand(batch_size, -1)
        positions = input_pos.to(device=device, dtype=torch.long)
        if positions.dim() == 1:
            return positions.unsqueeze(0).expand(batch_size, -1)
        if positions.shape[0] == 1 and batch_size > 1:
            return positions.expand(batch_size, -1)
        return positions

    def _relative_position_bias(
        self,
        q: Tensor,
        input_pos: Optional[Tensor],
        *,
        context_len: int,
    ) -> Tensor:
        batch_size, _, query_len, _ = q.shape
        query_pos = self._normalise_positions(
            input_pos,
            batch_size=batch_size,
            seq_len=query_len,
            device=q.device,
        )
        if context_len == query_len:
            key_pos = query_pos
        else:
            key_pos = torch.arange(
                context_len,
                device=q.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        relative = key_pos[:, None, :] - query_pos[:, :, None]
        relative = relative.clamp(
            min=-self.max_relative_position,
            max=self.max_relative_position,
        )
        relative = relative + self.max_relative_position
        rel_embeddings = self.rel_pos_embeddings.to(dtype=q.dtype)[relative]
        # F.scaled_dot_product_attention scales qk scores, so scale the relative
        # position term by head_dim as well before using it as additive bias.
        return torch.einsum("bhtd,btkd->bhtk", q, rel_embeddings) / math.sqrt(
            self.head_dim
        )

    @staticmethod
    def _merge_attention_mask_with_bias(
        mask: Tensor,
        bias: Tensor,
    ) -> Tensor:
        if mask.dtype == torch.bool:
            return bias.masked_fill(~mask.to(device=bias.device), -torch.inf)
        return mask.to(device=bias.device, dtype=bias.dtype) + bias

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope":
            if self.config.rope_base == -1:
                cos_emb, sin_emb = freqs_cis
                q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
                q, k = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
            else:
                q = apply_rotary_emb(q, freqs_cis)
                k = apply_rotary_emb(k, freqs_cis)
                q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        else:
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        attn_mask = mask
        if self.pos_embed_type == "conformer":
            rel_bias = self._relative_position_bias(
                q,
                input_pos,
                context_len=k.shape[-2],
            )
            attn_mask = self._merge_attention_mask_with_bias(mask, rel_bias)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
            attn_mask=attn_mask,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.head_dim * self.n_head)
        )
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-2,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class WindowLimitedTransformer(Transformer):
    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        look_ahead_conv: nn.Module = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.config = config

        self.channels_first = config.channels_first
        self.look_ahead_conv = (
            look_ahead_conv if look_ahead_conv is not None else nn.Identity()
        )
        self.input_proj = (
            nn.Linear(input_dim, config.dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim)
            if input_dim != config.dim
            else nn.Identity()
        )

    def make_window_limited_mask(
        self,
        max_length: int,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        mask = torch.tril(torch.ones(max_length, max_length))
        row_indices = torch.arange(max_length).view(-1, 1)
        window_size = self.window_size or max_length
        valid_range = (row_indices - window_size + 1).clamp(min=0)
        column_indices = torch.arange(max_length)
        mask = (column_indices >= valid_range) & mask.bool()

        mask = mask.bool()[None, None]
        return mask

    def forward(
        self,
        x: Tensor,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)

        input_pos = torch.arange(x.shape[1], device=x.device)
        if self.config.rope_base == -1:
            input_pos = input_pos.unsqueeze(0).expand(x.shape[0], -1)

        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length, x_lens)
        else:
            mask = self.make_mask(max_length, x_lens)
        mask = mask.to(x.device)
        x = super().forward(x, input_pos, mask)
        x = self.output_proj(x)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.float32
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
        padding=None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        pad = self.padding
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()


class CausalTransConvNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=None
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.dwconv = CausalConvNet(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)

        if apply_residual:
            x = input + x

        return x


class Snake1d(nn.Module):
    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super().__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.ModuleList([
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=1),]
        )
        self.causal = causal

    def forward(self, x):

        x_ori = x
        for m in self.block:
            x = m(x)
        pad = x_ori.shape[-1] - x.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2 : -pad // 2]
        return x + x_ori


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False
    ):
        super().__init__()

        self.block = nn.ModuleList(
            [Snake1d(input_dim),
            CausalTransConvNet(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),]
        )

    def forward(self, x):
        for m in self.block:
            x = m(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: Optional[list] = None,
    ):
        super().__init__()
        layers = [CausalConvNet(input_channel, channels, kernel_size=7, padding=3)]

        if n_transformer_layers is None:
            n_transformer_layers = [0, 0, 0, 0]

        # Add upsampling + MRF blocks
        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    causal=causal
                )
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            CausalConvNet(output_dim, d_out, kernel_size=7, padding=3),
        ]

        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return torch.clamp(x,min=-1,max=1)


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon

        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(
            min=self.epsilon
        )[:, None]
        quantized = F.embedding(codes, embedding)
        return quantized


class VectorQuantization(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim

        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            epsilon=epsilon
        )
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = quantized.transpose(1, 2)
        return quantized


class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        input_dimension: tp.Optional[int] = None,
        output_dimension: tp.Optional[int] = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        quantized = self.output_proj(quantized)
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer with separate projections for
    the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers
            used for the semantic quantizer.
        **kwargs: Arguments to the constructor of
            `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


class DAC(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = None,
        pre_transformer: torch.nn.Module = None,
        causal: bool = True,
        decoder_transformer_layers: List[int] = None,
        transformer_general_config=None,
        codebook_size = 2048,
        codebook_nums = 16,
        quantizer = None,
    ):
        if decoder_rates is None:
            decoder_rates = [8, 8, 4, 2]
        if decoder_transformer_layers is None:
            decoder_transformer_layers = [0, 0, 0, 0]
        super().__init__()

        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.decoder_hop = np.prod(decoder_rates)
        self.codebook_nums = int(codebook_nums)
        self.latent_dim = latent_dim

        self.quantizer = quantizer

        if self.quantizer is None:
            self.code_embeddings = nn.ModuleList(
                [
                    nn.Embedding(codebook_size, latent_dim)
                    for _ in range(codebook_nums)
                ]
            )
        else:
            self.pre_conv = CausalConvNet(
                embedding_dim,
                latent_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.pre_transformer = pre_transformer
        pre_decoder_upsample_rates = [2, 2]
        self.total_upsample = int(
            self.decoder_hop * np.prod(pre_decoder_upsample_rates)
        )
        self.upsample = nn.ModuleList(
            [
                nn.ModuleList(
                    [CausalTransConvNet(
                        latent_dim,
                        latent_dim,
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=latent_dim),]
                )
                for idx, factor in reversed(
                    list(enumerate(pre_decoder_upsample_rates))
                )
            ]
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
            n_transformer_layers=decoder_transformer_layers,
        )

    def _decode_inner(self, codes: torch.Tensor):
        if self.quantizer is not None:
            code_embs = self.quantizer.decode(codes.transpose(1, 2))
            code_embs = self.pre_conv(code_embs)
        else:
            code_list = codes.unbind(dim=-1)
            code_embs = [
                emb(code)
                for emb, code in zip(self.code_embeddings, code_list)
            ]
            code_embs = torch.stack(code_embs, dim=0).mean(dim=0).transpose(1, 2)

        x  = self.pre_transformer(code_embs)
        for block in self.upsample:
            for m in block:
                x = m(x)
        rec = self.decoder(x)

        return rec

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor):
        return self._decode_inner(codes)

    @torch.inference_mode()
    def forward(self, codes: torch.Tensor):
        """Decode sglang scheduler codec chunks.

        Qwen reference Next DAC `decode()` accepts [B, T, K], while the existing
        sglang-omni Code2WavScheduler passes [B, K, T]. Transpose only on the
        `model(codes)` path and keep the original reference semantics for
        `decode()`.
        """
        if codes.ndim != 3:
            raise ValueError(
                "Qwen3.5 code2wav expects codec ids with shape [B, K, T]"
            )
        return self.decode(codes.transpose(1, 2).contiguous())

    @torch.inference_mode()
    def decode_overlap_for_long(
        self,
        codes: torch.Tensor,
        chunk_size=250,
        overlap=25,
        upsample_rate=1920,
    ):
        b, t, d = codes.shape

        if t <= chunk_size:
            if self.quantizer is not None:
                code_embs = self.quantizer.decode(codes.transpose(1, 2))
                code_embs = self.pre_conv(code_embs)
            else:
                code_list = codes.unbind(dim=-1)
                code_embs = [
                    emb(code)
                    for emb, code in zip(
                        self.code_embeddings, code_list
                    )
                ]
                code_embs = torch.stack(code_embs, dim=0).mean(dim=0).transpose(1, 2)
            x = self.pre_transformer(code_embs)
            for block in self.upsample:
                for m in block:
                    x = m(x)
            return self.decoder(x)

        chunks = [codes[:, 0:chunk_size]]
        ptr = chunk_size
        while ptr < t:
            chunks.append(codes[:, ptr - overlap : ptr + chunk_size])
            ptr += chunk_size

        output_fragments = []
        trim_len = overlap * upsample_rate

        c = chunks[0]
        if self.quantizer is not None:
            code_embs = self.quantizer.decode(c.transpose(1, 2))
            code_embs = self.pre_conv(code_embs)
        else:
            code_list = c.unbind(dim=-1)
            code_embs = [
                emb(code)
                for emb, code in zip(self.code_embeddings, code_list)
            ]
            code_embs = torch.stack(code_embs, dim=0).mean(dim=0).transpose(1, 2)
        x = self.pre_transformer(code_embs)
        for block in self.upsample:
            for m in block:
                x = m(x)
        output_fragments.append(self.decoder(x))
        torch.cuda.empty_cache()

        if len(chunks) > 2:
            for c in chunks[1:-1]:
                if self.quantizer is not None:
                    code_embs = self.quantizer.decode(c.transpose(1, 2))
                    code_embs = self.pre_conv(code_embs)
                else:
                    code_list = c.unbind(dim=-1)
                    code_embs = [
                        emb(code)
                        for emb, code in zip(self.code_embeddings, code_list)
                    ]
                    code_embs = torch.stack(code_embs, dim=0)
                    code_embs = code_embs.mean(dim=0)
                    code_embs = code_embs.transpose(1, 2)

                x = self.pre_transformer(code_embs)
                for block in self.upsample:
                    for m in block:
                        x = m(x)

                output_fragments.append(self.decoder(x)[:, :, trim_len:])
                torch.cuda.empty_cache()

        if len(chunks) > 1:
            c = chunks[-1]
            if self.quantizer is not None:
                code_embs = self.quantizer.decode(c.transpose(1, 2))
                code_embs = self.pre_conv(code_embs)
            else:
                code_list = c.unbind(dim=-1)
                code_embs = [
                    emb(code)
                    for emb, code in zip(self.code_embeddings, code_list)
                ]
                code_embs = torch.stack(code_embs, dim=0)
                code_embs = code_embs.mean(dim=0)
                code_embs = code_embs.transpose(1, 2)
            x = self.pre_transformer(code_embs)
            for block in self.upsample:
                for m in block:
                    x = m(x)
            output_fragments.append(self.decoder(x)[:, :, trim_len:])
            torch.cuda.empty_cache()

        return torch.cat(output_fragments, dim=2)

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) is dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

def hparams_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return HParams(**fields)

def get_hparams_from_file(config_path):
    yaml.add_constructor('utils.utils.HParams', hparams_constructor)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    hparams = HParams(**config)
    return hparams


def _get_required_hparam(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    joined = " or ".join(names)
    raise AttributeError(f"missing required DAC config field: {joined}")


def _get_optional_hparam(obj, name, default):
    return getattr(obj, name, default)


def _build_post_module_model_args(args):
    return ModelArgs(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dim=args.dim,
        intermediate_size=args.intermediate_size,
        n_local_heads=args.n_local_heads,
        head_dim=args.head_dim,
        rope=args.rope if hasattr(args, "rope") else None,
        rope_base=args.rope_base,
        norm_eps=args.norm_eps,
        dropout_rate=args.dropout_rate,
        attn_dropout_rate=args.attn_dropout_rate,
        channels_first=args.channels_first,
        pos_embed_type=_get_optional_hparam(args, "pos_embed_type", "rope"),
        max_relative_position=int(
            _get_optional_hparam(args, "max_relative_position", 128)
        ),
    )


def _load_checkpoint_state_dict(checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, mmap=True, weights_only=True
        )
    except TypeError:
        # Older torch.load versions may not support mmap/weights_only. Fall back
        # to basic arguments so checkpoints remain loadable.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    state_dict = _strip_state_dict_prefix(state_dict, "module.")

    if any(k.startswith("generator.") for k in state_dict):
        state_dict = {
            k[len("generator.") :]: v
            for k, v in state_dict.items()
            if k.startswith("generator.")
        }
    return state_dict


def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Qwen3.5 code2wav checkpoint must load to a mapping, got "
            f"{type(checkpoint).__name__}"
        )

    for key in ("model", "state_dict", "generator"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            # Exported PyTorch checkpoints are not consistent about their outer
            # key: Qwen reference uses model, while training scripts may use
            # state_dict/generator. Normalize all of them to the flat parameter
            # table expected by DAC model.load_state_dict.
            return dict(value)
    return dict(checkpoint)


def _strip_state_dict_prefix(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _load_model_state_dict(model, state_dict):
    try:
        signature = inspect.signature(model.load_state_dict)
    except (TypeError, ValueError):
        signature = None

    supports_assign = False
    if signature is not None:
        parameters = signature.parameters
        supports_assign = "assign" in parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        )

    if supports_assign:
        try:
            return model.load_state_dict(state_dict, strict=False, assign=True)
        except TypeError as exc:
            if "assign" not in str(exc):
                raise

    # Some older torch versions or lightweight wrapper models do not support the
    # assign argument. Plain load_state_dict still starts correctly, with only a
    # little less parameter-replacement loading optimization.
    return model.load_state_dict(state_dict, strict=False)


def load_qwen3_omni_next_dac_from_config(
    config_path=None,
    checkpoint_path=None,
    device="cuda",
    dtype=torch.bfloat16,
):
    hps = get_hparams_from_file(config_path).model

    if hasattr(hps, "type") and hps.type == "public_v1":
        quantizer = SplitResidualVectorQuantizer(
            dimension=hps.quantizer.dimension,
            n_q=hps.quantizer.n_q,
            n_q_semantic=hps.quantizer.n_q_semantic,
            bins = hps.quantizer.bins,
            input_dimension=hps.quantizer.input_dimension,
            output_dimension=hps.quantizer.output_dimension,
        )
    else:
        quantizer = None

    tranformer_config_quantizer = _build_post_module_model_args(
        hps.post_module.args
    )

    post_module = WindowLimitedTransformer(
        config = tranformer_config_quantizer,
        input_dim = hps.post_module.input_dim,
        window_size = hps.post_module.window_size,
    )

    model = DAC(
        latent_dim=_get_required_hparam(hps.dac, "latet_dim", "latent_dim"),
        decoder_dim = hps.dac.decoder_dim,
        decoder_rates = hps.dac.decoder_rates,
        pre_transformer=post_module,
        causal = hps.dac.causal,
        decoder_transformer_layers = hps.dac.decoder_transformer_layers,
        transformer_general_config=None,
        codebook_nums=hps.dac.codebook_nums,
        codebook_size=hps.dac.codebook_size,
        quantizer=quantizer
    )

    if checkpoint_path is not None:
        state_dict = _load_checkpoint_state_dict(checkpoint_path, device)
        result = _load_model_state_dict(model, state_dict)
        print(f"Loaded model: {result}")


    model.eval()
    model.to(device)
    model.to(dtype)

    return model
