# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.image_encoder import (
    _optimize_patch_embed,
)
from sglang_omni.models.qwen35_omni.components.image_encoder import (
    _hf_grouped_sdpa,
    _hf_split_sdpa,
    _normalize_native_visual_attention_backend,
    _normalize_visual_backend,
    _unpack_visual_outputs,
)


def test_qwen35_visual_backend_validation() -> None:
    assert _normalize_visual_backend(" SGLANG ") == "sglang"
    assert _normalize_visual_backend("hf") == "hf"
    with pytest.raises(ValueError, match="Unsupported Qwen3.5 vision backend"):
        _normalize_visual_backend("unknown")


def test_qwen35_native_visual_attention_backend_validation() -> None:
    assert _normalize_native_visual_attention_backend("sdpa") == "sdpa"
    assert _normalize_native_visual_attention_backend("SDPA_GROUPED") == "sdpa_grouped"
    with pytest.raises(ValueError, match="native vision attention backend"):
        _normalize_native_visual_attention_backend("fa3")


def test_qwen35_native_visual_tensor_output_is_supported() -> None:
    output = torch.randn((4, 8))

    embeds, deepstack = _unpack_visual_outputs(output)

    assert embeds is output
    assert deepstack is None


def test_patch_embed_optimization_can_keep_native_conv_properties() -> None:
    conv = nn.Conv3d(3, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    patch_embed = nn.Module()
    patch_embed.proj = conv
    visual = SimpleNamespace(patch_embed=patch_embed)
    inputs = torch.randn((5, 24))

    expected = torch.nn.functional.linear(
        inputs, conv.weight.view(conv.out_channels, -1), conv.bias
    )
    _optimize_patch_embed(visual, keep_conv=True)

    assert patch_embed.proj is conv
    assert torch.allclose(patch_embed(inputs), expected)


def test_hf_split_sdpa_matches_per_grid_reference() -> None:
    torch.manual_seed(7)
    q = torch.randn(5, 2, 4)
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    backend = SimpleNamespace(dropout=0.0, scale=0.5)

    output = _hf_split_sdpa(
        backend,
        q,
        k,
        v,
        bsz=1,
        cu_seqlens=cu_seqlens,
    )

    expected_parts = []
    for start, end in ((0, 2), (2, 5)):
        q_part, k_part, v_part = [
            tensor[start:end].transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)
        ]
        part = torch.nn.functional.scaled_dot_product_attention(
            q_part,
            k_part,
            v_part,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=0.5,
        )
        expected_parts.append(part.squeeze(0).transpose(0, 1))

    assert torch.equal(output, torch.cat(expected_parts, dim=0))


def test_hf_split_sdpa_rejects_nonflattened_batch() -> None:
    tensor = torch.randn(2, 1, 4)
    backend = SimpleNamespace(dropout=0.0, scale=0.5)

    with pytest.raises(ValueError, match="flattened batch size 1"):
        _hf_split_sdpa(backend, tensor, tensor, tensor, bsz=2)


def test_hf_grouped_sdpa_matches_split_reference_for_equal_grids() -> None:
    torch.manual_seed(11)
    q = torch.randn(12, 2, 4)
    k = torch.randn(12, 2, 4)
    v = torch.randn(12, 2, 4)
    cu_seqlens = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    backend = SimpleNamespace(dropout=0.0, scale=0.5)

    grouped = _hf_grouped_sdpa(
        backend,
        q,
        k,
        v,
        bsz=1,
        cu_seqlens=cu_seqlens,
    )
    split = _hf_split_sdpa(
        backend,
        q,
        k,
        v,
        bsz=1,
        cu_seqlens=cu_seqlens,
    )

    assert torch.equal(grouped, split)


def test_hf_grouped_sdpa_falls_back_for_mixed_grid_lengths() -> None:
    torch.manual_seed(13)
    q = torch.randn(9, 2, 4)
    k = torch.randn(9, 2, 4)
    v = torch.randn(9, 2, 4)
    cu_seqlens = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
    backend = SimpleNamespace(dropout=0.0, scale=0.5)

    grouped = _hf_grouped_sdpa(
        backend,
        q,
        k,
        v,
        bsz=1,
        cu_seqlens=cu_seqlens,
    )
    split = _hf_split_sdpa(
        backend,
        q,
        k,
        v,
        bsz=1,
        cu_seqlens=cu_seqlens,
    )

    assert torch.equal(grouped, split)
