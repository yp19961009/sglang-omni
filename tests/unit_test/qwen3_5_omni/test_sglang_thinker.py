# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.components import sglang_thinker


class _TextConfig:
    vocab_size = 16
    hidden_size = 4


class _RootConfig:
    thinker_config = SimpleNamespace(text_config=_TextConfig())


class _FakeTextModel(nn.Module):
    def forward(self, input_ids, positions, forward_batch, inputs_embeds=None):
        return inputs_embeds


class _FakeLanguageModel(nn.Module):
    instances = []

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.model = _FakeTextModel()
        self.forward_kwargs = None
        self.loaded_weights = None
        self.instances.append(self)

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return "logits"

    def load_weights(self, weights):
        self.loaded_weights = list(weights)
        return {name for name, _ in self.loaded_weights}


def _install_fake_language_model(monkeypatch):
    _FakeLanguageModel.instances.clear()
    monkeypatch.setattr(
        sglang_thinker,
        "Qwen3NextForCausalLM",
        _FakeLanguageModel,
    )


def test_qwen35_sglang_thinker_exports_mtp_alias():
    assert (
        sglang_thinker.Qwen3OmniNextThinkerMTP
        is sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration
    )


def test_qwen35_sglang_thinker_maps_input_embeds(monkeypatch):
    _install_fake_language_model(monkeypatch)
    model = sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(
        _RootConfig()
    )
    embeds = torch.ones(1, 1, 4)
    mrope_positions = torch.tensor([[0]])
    forward_batch = SimpleNamespace(mrope_positions=mrope_positions)

    assert (
        model(
            input_ids=torch.tensor([[1]]),
            positions=torch.tensor([[9]]),
            forward_batch=forward_batch,
            input_embeds=embeds,
        )
        == "logits"
    )
    language_model = _FakeLanguageModel.instances[-1]

    assert language_model.forward_kwargs["inputs_embeds"] is embeds
    assert language_model.forward_kwargs["positions"] is mrope_positions


def test_qwen35_sglang_thinker_accepts_split_config_with_none_subconfig(
    monkeypatch,
):
    _install_fake_language_model(monkeypatch)
    split_config = SimpleNamespace(thinker_config=None, text_config=_TextConfig())

    model = sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(split_config)

    assert model.thinker_config is split_config
    assert _FakeLanguageModel.instances[-1].config is split_config.text_config


def test_qwen35_sglang_thinker_normalizes_qwen3_next_layer_types(monkeypatch):
    _install_fake_language_model(monkeypatch)
    text_config = SimpleNamespace(
        vocab_size=16,
        hidden_size=4,
        layer_types=("linear_attention", "full_attention"),
    )
    root_config = SimpleNamespace(thinker_config=SimpleNamespace(text_config=text_config))

    sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(root_config)

    assert _FakeLanguageModel.instances[-1].config.layers_block_type == [
        "linear_attention",
        "attention",
    ]
    assert _FakeLanguageModel.instances[-1].config.partial_rotary_factor == 0.25
    assert _FakeLanguageModel.instances[-1].config.torch_dtype is None


def test_qwen35_sglang_thinker_accepts_direct_split_text_config(monkeypatch):
    _install_fake_language_model(monkeypatch)
    split_config = SimpleNamespace(
        thinker_config=None,
        text_config=None,
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=4,
        linear_num_key_heads=16,
        full_attention_interval=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 500000.0,
        },
    )

    model = sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(split_config)

    assert model.config is split_config
    assert _FakeLanguageModel.instances[-1].config.layers_block_type == [
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
    ]
    assert _FakeLanguageModel.instances[-1].config.rope_scaling == {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    assert _FakeLanguageModel.instances[-1].config.rope_theta == 500000.0


def test_qwen35_sglang_thinker_rejects_invalid_full_attention_interval(monkeypatch):
    _install_fake_language_model(monkeypatch)
    split_config = SimpleNamespace(
        thinker_config=None,
        text_config=None,
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=4,
        full_attention_interval=0,
        rope_parameters={"rope_type": "default"},
    )

    with pytest.raises(ValueError, match="full_attention_interval must be positive"):
        sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(split_config)


def test_qwen35_sglang_thinker_applies_deepstack(monkeypatch):
    class FakeForwardMode:
        def is_idle(self):
            return False

    class FakeLayer(nn.Module):
        def forward(
            self,
            *,
            layer_id,
            positions,
            hidden_states,
            residual,
            forward_batch,
        ):
            del layer_id, positions, residual, forward_batch
            return hidden_states + 1, None

    class DeepstackTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer()])
            self.norm = lambda hidden, *args: hidden if not args else (hidden, None)

        def forward(self, input_ids, positions, forward_batch, inputs_embeds=None):
            del input_ids, positions, forward_batch
            return inputs_embeds

        def embed_tokens(self, input_ids):
            return input_ids.to(dtype=torch.float32).unsqueeze(-1)

    class DeepstackLanguageModel(nn.Module):
        def __init__(self, config, quant_config=None, prefix=""):
            super().__init__()
            del config, quant_config, prefix
            self.model = DeepstackTextModel()
            self.lm_head = object()

        def logits_processor(self, input_ids, hidden_states, lm_head, forward_batch):
            del input_ids, lm_head, forward_batch
            return hidden_states

    monkeypatch.setattr(
        sglang_thinker,
        "Qwen3NextForCausalLM",
        DeepstackLanguageModel,
    )
    model = sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(
        _RootConfig()
    )
    output = model(
        input_ids=torch.tensor([3]),
        positions=torch.tensor([0]),
        forward_batch=SimpleNamespace(
            mrope_positions=None,
            forward_mode=FakeForwardMode(),
        ),
        input_deepstack_embeds={"deepstack_input_embeds_0": torch.tensor([[10.0]])},
    )

    assert output.tolist() == [[14.0]]


def test_qwen35_sglang_thinker_load_weights_strips_omni_prefixes(monkeypatch):
    _install_fake_language_model(monkeypatch)
    model = sglang_thinker.Qwen3OmniNextThinkerForConditionalGeneration(
        _RootConfig()
    )
    weight = torch.ones(1)

    model.load_weights(
        [
            ("thinker.model.embed_tokens.weight", weight),
            ("thinker.lm_head.weight", weight),
            ("language_model.model.norm.weight", weight),
            ("thinker.model.layers.0.self_attn.k_scale", weight),
            ("thinker.audio_tower.conv.weight", weight),
            ("thinker.mtp.fc.weight", weight),
            ("visual.patch_embed.weight", weight),
            ("talker.model.embed_tokens.weight", weight),
            ("code2wav.decoder.weight", weight),
        ]
    )
    language_model = _FakeLanguageModel.instances[-1]

    assert [name for name, _ in language_model.loaded_weights] == [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]


def test_qwen35_hidden_capture_forward_returns_aux_states():
    class FakeForwardMode:
        def is_idle(self):
            return False

    class FakeLayer(nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = delta

        def forward(
            self,
            *,
            layer_id,
            positions,
            hidden_states,
            residual,
            forward_batch,
        ):
            del layer_id, positions, residual, forward_batch
            return hidden_states + self.delta, None

    class FakeTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(1), FakeLayer(2)])
            self.norm = lambda hidden, *args: hidden if not args else (hidden, None)

        def forward(self, input_ids, positions, forward_batch, inputs_embeds=None):
            del input_ids, positions, forward_batch
            return inputs_embeds

        def embed_tokens(self, input_ids):
            return input_ids.to(dtype=torch.float32).unsqueeze(-1)

    text_model = FakeTextModel()
    sglang_thinker._install_layers_to_capture_support(text_model)
    text_model.layers_to_capture = [0, 1]

    hidden, aux = text_model(
        torch.tensor([3]),
        torch.tensor([0]),
        SimpleNamespace(forward_mode=FakeForwardMode()),
    )

    assert hidden.tolist() == [[6.0]]
    assert [tensor.tolist() for tensor in aux] == [[[3.0]], [[6.0]]]


def test_qwen35_hidden_capture_records_layer_output_after_deepstack():
    class FakeForwardMode:
        def is_idle(self):
            return False

    class FakeLayer(nn.Module):
        def forward(
            self,
            *,
            layer_id,
            positions,
            hidden_states,
            residual,
            forward_batch,
        ):
            del layer_id, positions, residual, forward_batch
            return hidden_states + 1, None

    class FakeTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
            self.norm = lambda hidden, *args: hidden if not args else (hidden, None)

        def forward(self, input_ids, positions, forward_batch, inputs_embeds=None):
            del input_ids, positions, forward_batch
            return inputs_embeds

        def embed_tokens(self, input_ids):
            return input_ids.to(dtype=torch.float32).unsqueeze(-1)

    text_model = FakeTextModel()
    sglang_thinker._install_layers_to_capture_support(text_model)
    text_model.layers_to_capture = [0, 1]

    hidden, aux = text_model(
        torch.tensor([3]),
        torch.tensor([0]),
        SimpleNamespace(forward_mode=FakeForwardMode()),
        deepstack_input_embeds={"deepstack_input_embeds_1": torch.tensor([[10.0]])},
    )

    assert hidden.tolist() == [[15.0]]
    assert [tensor.tolist() for tensor in aux] == [[[3.0]], [[15.0]]]


def test_qwen35_hidden_capture_delegates_native_capture_without_deepstack():
    class FakeForwardMode:
        def is_idle(self):
            return False

    class NativeCaptureTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers_to_capture = [0, 1]
            self.layers = nn.ModuleList(
                [
                    nn.Module(),
                ]
            )
            self.native_called = False

        def forward(self, input_ids, positions, forward_batch, inputs_embeds=None):
            del input_ids, positions, forward_batch
            layers_to_capture = self.layers_to_capture
            aux_hidden_states = [inputs_embeds + layer for layer in layers_to_capture]
            self.native_called = True
            return inputs_embeds + 10, aux_hidden_states

        def embed_tokens(self, input_ids):
            return input_ids.to(dtype=torch.float32).unsqueeze(-1)

    text_model = NativeCaptureTextModel()
    sglang_thinker._install_layers_to_capture_support(text_model)

    hidden, aux = text_model(
        torch.tensor([3]),
        torch.tensor([0]),
        SimpleNamespace(forward_mode=FakeForwardMode()),
        inputs_embeds=torch.tensor([[2.0]]),
    )

    assert text_model.native_called
    assert hidden.tolist() == [[12.0]]
    assert [tensor.tolist() for tensor in aux] == [[[2.0]], [[3.0]]]
