# SPDX-License-Identifier: Apache-2.0
"""Local Qwen3.5-Omni Next encoder components.

中文说明：远端环境当前没有 transformers 的 qwen3_omni_next 模块。
这里先把 audio tower 以纯 PyTorch 形式放到 sglang-omni 内部，保持
HF checkpoint 的 q/k/v 拆分参数名，便于 `load_module(strict=True)` 直接加载。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_feat_extract_output_lengths(
    input_lengths: torch.Tensor,
    downsample_times: int = 4,
    chunk_size: int = 100,
) -> torch.Tensor:
    input_lengths_leave = input_lengths % chunk_size
    for _ in range(downsample_times):
        input_lengths_leave = (input_lengths_leave - 1) // 2 + 1
    return input_lengths_leave + (input_lengths // chunk_size) * math.ceil(
        100 / 2**downsample_times
    )


def _activation(name: str):
    if name in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
        return F.gelu
    if name in {"silu", "swish"}:
        return F.silu
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported Qwen3.5 audio activation_function={name!r}")


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for Qwen3.5 audio encoder."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requires even channels")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        positional_embedding = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)],
            dim=1,
        )
        self.register_buffer(
            "positional_embedding",
            positional_embedding,
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen]


class Qwen3OmniNextAudioAttention(nn.Module):
    """Packed-sequence attention for the Qwen3.5 audio tower."""

    def __init__(self, config: Any):
        super().__init__()
        self.embed_dim = int(config.d_model)
        self.num_heads = int(config.encoder_attention_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("audio d_model must divide encoder_attention_heads")
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len = tensor.shape[0]
        return tensor.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        q = self._shape(self.q_proj(hidden_states))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if lengths.numel() == 0:
            return hidden_states[:0]
        max_len = int(lengths.max().item())
        chunk_count = int(lengths.numel())
        q_padded = q.new_zeros(chunk_count, self.num_heads, max_len, self.head_dim)
        k_padded = q.new_zeros(chunk_count, self.num_heads, max_len, self.head_dim)
        v_padded = q.new_zeros(chunk_count, self.num_heads, max_len, self.head_dim)
        valid_mask = torch.arange(max_len, device=hidden_states.device).unsqueeze(0)
        valid_mask = valid_mask < lengths.to(device=hidden_states.device).unsqueeze(1)

        # 中文说明：把 packed chunks padding 成一个 batch，一次 SDPA 跑完；
        # chunk 之间仍然完全隔离，比逐 chunk 循环更接近 vLLM 的 packed attention。
        for row, (start, end) in enumerate(
            zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
        ):
            chunk_len = end - start
            q_padded[row, :, :chunk_len] = q[:, start:end]
            k_padded[row, :, :chunk_len] = k[:, start:end]
            v_padded[row, :, :chunk_len] = v[:, start:end]

        key_mask = ~valid_mask[:, None, None, :]
        attn_mask = q_padded.new_zeros(chunk_count, 1, max_len, max_len)
        attn_mask = attn_mask.masked_fill(key_mask, float("-inf"))
        out = F.scaled_dot_product_attention(
            q_padded,
            k_padded,
            v_padded,
            attn_mask=attn_mask,
        )
        attn_output = out.transpose(1, 2)[valid_mask].reshape(-1, self.embed_dim)
        return self.out_proj(attn_output)


class Qwen3OmniNextAudioEncoderLayer(nn.Module):
    """Transformer encoder layer for the Qwen3.5 audio tower."""

    def __init__(self, config: Any):
        super().__init__()
        embed_dim = int(config.d_model)
        self.self_attn = Qwen3OmniNextAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = _activation(config.activation_function)
        self.fc1 = nn.Linear(embed_dim, int(config.encoder_ffn_dim), bias=True)
        self.fc2 = nn.Linear(int(config.encoder_ffn_dim), embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cu_seqlens)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states,
                min=-clamp_value,
                max=clamp_value,
            )
        return hidden_states


class Qwen3OmniNextAudioEncoder(nn.Module):
    """Pure PyTorch Qwen3.5-Omni Next audio encoder."""

    def __init__(self, config: Any):
        super().__init__()
        embed_dim = int(config.d_model)
        self.num_mel_bins = int(config.num_mel_bins)
        self.max_source_positions = int(config.max_source_positions)
        self.n_window = int(config.n_window)
        self.n_window_infer = int(config.n_window_infer)
        self.conv_chunksize = int(getattr(config, "conv_chunksize", 32))

        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        hidden = int(config.downsample_hidden_size)
        self.conv2d1 = nn.Conv2d(1, hidden, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv2d4 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)

        freq_after_cnn = (
            (((((self.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2)
        )
        self.conv_out = nn.Linear(hidden * freq_after_cnn, embed_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                Qwen3OmniNextAudioEncoderLayer(config)
                for _ in range(int(config.encoder_layers))
            ]
        )
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.act = _activation(config.activation_function)
        self.proj2 = nn.Linear(embed_dim, int(config.output_dim), bias=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv2d1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv2d1.weight.device

    def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths
        for _ in range(4):
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ) -> torch.Tensor:
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * int(chunk_num.sum().item()),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list,
            batch_first=True,
        ).transpose(1, 2)
        feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
        max_len_after_cnn = int(feature_lens_after_cnn.max().item())
        indices = torch.arange(max_len_after_cnn, device=padded_feature.device)
        padded_mask_after_cnn = indices.unsqueeze(0) < feature_lens_after_cnn[:, None]

        padded_feature = padded_feature.unsqueeze(1)
        conv_inputs = padded_feature.split(self.conv_chunksize, dim=0)
        padded_embeds = []
        for chunk in conv_inputs:
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embed = F.gelu(self.conv2d4(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

        bsz, channels, freq, time = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2)
            .contiguous()
            .view(bsz, time, channels * freq)
        )
        positional_embedding = self.positional_embedding(
            padded_embed.shape[1]
        ).to(device=padded_embed.device, dtype=padded_embed.dtype)
        padded_embed = padded_embed + positional_embedding.unsqueeze(0)
        hidden_states = padded_embed[padded_mask_after_cnn]

        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens.tolist():
            num_full_chunks = int(cnn_len) // window_aftercnn
            remainder = int(cnn_len) % window_aftercnn
            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
            if remainder:
                cu_chunk_lens.append(remainder)
        cu_seqlens = torch.tensor(
            cu_chunk_lens,
            device=aftercnn_lens.device,
            dtype=torch.int32,
        ).cumsum(-1)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.proj2(hidden_states)
