# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.components import subtalker


class _FakeTensorLinear(nn.Module):
    def __init__(self, input_size, output_size, *, bias=True, prefix=""):
        del prefix
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        return self.linear(x)


def test_qwen35_residual_predictor_generates_non_pad_groups(monkeypatch):
    monkeypatch.setattr(subtalker, "_TensorLinear", _FakeTensorLinear)
    config = SimpleNamespace(
        num_code_groups=3,
        vocab_size=5,
        talker_hidden_size=4,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        rms_norm_eps=1e-6,
    )
    predictor = subtalker.Qwen35ResidualCodePredictor(config)
    for head in predictor.lm_head:
        nn.init.zeros_(head.linear.weight)

    layer0_embed = nn.Embedding(5, 4)
    layer0_codes = torch.tensor([[2]])
    talker_hidden = torch.ones(1, 1, 4)

    codes, summed = predictor.generate(
        layer0_codes=layer0_codes,
        talker_hidden=talker_hidden,
        layer0_embed_fn=layer0_embed,
        pad_id=4,
    )

    assert codes.shape == (1, 3, 1)
    assert codes[:, :, 0].tolist() == [[2, 0, 0]]
    assert summed.shape == (1, 1, 4)


def test_qwen35_residual_predictor_uses_next_decoder_weight_names(monkeypatch):
    monkeypatch.setattr(subtalker, "_TensorLinear", _FakeTensorLinear)
    config = SimpleNamespace(
        num_code_groups=3,
        vocab_size=5,
        talker_hidden_size=4,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        rms_norm_eps=1e-6,
    )

    names = set(dict(subtalker.Qwen35ResidualCodePredictor(config).named_parameters()))

    assert {
        "model.codec_embedding.0.weight",
        "model.talker_projection.linear.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.0.linear.weight",
    } <= names


def test_qwen35_residual_predictor_reports_local_next_decoder(monkeypatch):
    monkeypatch.setattr(subtalker, "_TensorLinear", _FakeTensorLinear)
    monkeypatch.setattr(
        subtalker,
        "_probe_next_predictor",
        lambda: (False, "missing qwen3_omni_next"),
    )
    config = SimpleNamespace(
        num_code_groups=2,
        vocab_size=5,
        talker_hidden_size=4,
        hidden_size=4,
        num_hidden_layers=0,
    )

    predictor = subtalker.Qwen35ResidualCodePredictor(config)

    assert predictor.implementation == "sglang_next_decoder"
    assert predictor.exact_next_available is True
    assert predictor.exact_next_error is None
    assert predictor.hf_next_available is False
    assert predictor.hf_next_error == "missing qwen3_omni_next"


def test_qwen35_subtalker_rmsnorm_uses_qwen_next_one_plus_weight():
    norm = subtalker._RMSNorm(2, 1e-6)
    with torch.no_grad():
        norm.weight.fill_(0.5)
    x = torch.tensor([[3.0, 4.0]])

    out = norm(x)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * 1.5

    assert torch.allclose(out, expected)


def test_qwen35_subtalker_uses_partial_rotary_factor():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.5,
    )

    attention = subtalker._NextSelfAttention(config)

    assert attention.rotary_dim == 4
    assert attention.rotary_emb.inv_freq.numel() == 2


def test_qwen35_subtalker_accepts_rope_parameters_partial_rotary_factor():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        rope_parameters={"partial_rotary_factor": 0.25},
    )

    attention = subtalker._NextSelfAttention(config)

    assert attention.rotary_dim == 2


def test_qwen35_subtalker_enables_interleaved_mrope_from_config():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [1, 1, 0]},
    )

    attention = subtalker._NextSelfAttention(config)
    cos, _ = attention.rotary_emb(
        torch.tensor([[1]]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert attention.mrope_interleaved is True
    assert attention.rotary_emb.interleaved is True
    assert attention.rotary_emb.mrope_section == (1, 1, 0)
    assert torch.isclose(cos[0, 0, 0], cos[0, 0, 2])
    assert torch.isclose(cos[0, 0, 1], cos[0, 0, 3])
    assert not torch.isclose(cos[0, 0, 0], cos[0, 0, 1])


def test_qwen35_subtalker_treats_missing_attention_gate_as_enabled():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        partial_rotary_factor=0.5,
        attn_output_gate=None,
    )

    attention = subtalker._NextSelfAttention(config)

    assert attention.attn_output_gate is True
    assert attention.q_proj.out_features == 16


def test_qwen35_subtalker_allows_qwen3_head_dim_larger_than_hidden_slice():
    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=3,
        partial_rotary_factor=1.0,
        rms_norm_eps=1e-6,
    )

    attention = subtalker._NextSelfAttention(config)
    hidden = torch.randn(1, 2, 4)
    position_ids = torch.arange(2).unsqueeze(0)

    out = attention(hidden, position_ids)

    assert attention.q_proj.out_features == 12
    assert attention.o_proj.in_features == 6
    assert out.shape == hidden.shape


def test_qwen35_subtalker_rotary_leaves_unrotated_tail_unchanged():
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cos = torch.zeros(1, 1, 2)
    sin = torch.ones(1, 1, 2)

    q_out, k_out = subtalker._apply_rotary(q, k, cos, sin, rotary_dim=2)

    assert torch.equal(q_out[..., 2:], q[..., 2:])
    assert torch.equal(k_out[..., 2:], k[..., 2:])
    assert not torch.equal(q_out[..., :2], q[..., :2])
    assert not torch.equal(k_out[..., :2], k[..., :2])


def test_qwen35_subtalker_interleaved_mrope_keeps_hf_rotary_layout():
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cos = torch.zeros(1, 1, 4)
    sin = torch.ones(1, 1, 4)

    q_out, k_out = subtalker._apply_rotary(
        q,
        k,
        cos,
        sin,
        rotary_dim=4,
        interleaved=True,
    )

    assert q_out.tolist() == [[[[-3.0, -4.0, 1.0, 2.0]]]]
    assert k_out.tolist() == [[[[-7.0, -8.0, 5.0, 6.0]]]]


def test_qwen35_residual_sampling_respects_top_k():
    logits = torch.tensor([[0.0, 5.0, 1.0], [4.0, 3.0, 2.0]])

    sampled = subtalker._sample_logits(
        logits,
        temperature=torch.ones(2),
        top_k=torch.ones(2, dtype=torch.long),
        top_p=torch.ones(2),
    )

    assert sampled.tolist() == [1, 0]


def test_qwen35_sampling_respects_min_p():
    logits = torch.tensor([[5.0, 4.0, 0.0]])

    sampled = subtalker._sample_logits(
        logits,
        temperature=torch.ones(1),
        top_k=torch.zeros(1, dtype=torch.long),
        top_p=torch.ones(1),
        min_p=torch.ones(1) * 0.5,
    )

    assert sampled.item() in {0, 1}


def test_qwen35_residual_sampling_seed_is_deterministic():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    kwargs = {
        "temperature": torch.ones(1),
        "top_k": torch.zeros(1, dtype=torch.long),
        "top_p": torch.ones(1),
        "seed": torch.tensor([123]),
    }

    first = subtalker._sample_logits(logits, **kwargs)
    second = subtalker._sample_logits(logits, **kwargs)

    assert first.tolist() == second.tolist()


def test_qwen35_residual_predictor_exact_env_uses_local_decoder(monkeypatch):
    monkeypatch.setattr(subtalker, "_TensorLinear", _FakeTensorLinear)
    monkeypatch.setattr(
        subtalker,
        "_probe_next_predictor",
        lambda: (False, "missing qwen3_omni_next"),
    )
    monkeypatch.setenv("SGLANG_OMNI_QWEN35_REQUIRE_EXACT_SUBTALKER", "1")
    config = SimpleNamespace(
        num_code_groups=2,
        vocab_size=5,
        talker_hidden_size=4,
        hidden_size=4,
        num_hidden_layers=0,
    )

    predictor = subtalker.Qwen35ResidualCodePredictor(config)

    assert predictor.implementation == "sglang_next_decoder"
