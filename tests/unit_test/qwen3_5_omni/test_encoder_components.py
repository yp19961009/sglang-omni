# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.components import (
    audio_encoder,
    image_encoder,
)
from sglang_omni.models.qwen3_5_omni.components.qwen3_omni_next_thinker import (
    Qwen3OmniNextAudioAttention,
    Qwen3OmniNextAudioEncoder,
)


def _audio_config():
    return SimpleNamespace(
        d_model=4,
        num_mel_bins=8,
        max_source_positions=8,
        n_window=4,
        n_window_infer=8,
        conv_chunksize=4,
        downsample_hidden_size=2,
        encoder_layers=1,
        encoder_attention_heads=1,
        encoder_ffn_dim=8,
        activation_function="gelu",
        output_dim=6,
    )


def test_image_encoder_resolves_next_vision_class(monkeypatch):
    module_name = "fake_qwen35_vision"
    module = ModuleType(module_name)

    class FakeVision:
        pass

    module.Qwen3OmniNextVisionEncoder = FakeVision
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(image_encoder, "_MODELING_MODULE_CANDIDATES", (module_name,))

    assert image_encoder._resolve_next_class(
        ("Qwen3OmniNextVisionEncoder",)
    ) is FakeVision


def test_image_encoder_resolves_sglang_vision_fallback():
    cls = image_encoder._resolve_next_class(("Qwen3VLMoeVisionModel",))

    assert cls.__name__ == "Qwen3VLMoeVisionModel"


def test_image_encoder_uses_custom_visual_loader(monkeypatch):
    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1)
            self.loaded = None

        def load_weights(self, weights):
            self.loaded = list(weights)
            return {"proj.weight"}

    fake_state = {"blocks.0.attn.q.weight": torch.ones(1, 1)}
    monkeypatch.setattr(
        image_encoder,
        "load_weights_by_prefix",
        lambda model_path, *, prefix: fake_state,
    )

    def _fail_load_module(*args, **kwargs):
        raise AssertionError("load_module should not handle packed visual weights")

    monkeypatch.setattr(image_encoder, "load_module", _fail_load_module)
    visual = FakeVisual()

    loaded = image_encoder._load_visual_weights(
        visual,
        "dummy",
        torch_dtype=None,
        device="cpu",
    )

    assert loaded is visual
    assert visual.loaded == list(fake_state.items())
    assert visual.training is False


def test_image_patch_embed_optimization_preserves_proj_for_dtype():
    class FakePatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv3d(
                1,
                2,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
            )

        def forward(self, hidden_states):
            return self.proj(hidden_states)

    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = FakePatchEmbed()

        @property
        def dtype(self):
            return self.patch_embed.proj.weight.dtype

    visual = FakeVisual()

    image_encoder._optimize_patch_embed(visual)

    assert hasattr(visual.patch_embed, "proj")
    assert hasattr(visual.patch_embed, "linear")
    assert visual.dtype == visual.patch_embed.proj.weight.dtype
    hidden_states = torch.randn(3, 1)
    assert visual.patch_embed(hidden_states).shape == (3, 2)


def test_image_encoder_calls_visual_with_image_grid_kwarg():
    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.seen_grid = None

        def forward(self, pixel_values, *, image_grid_thw):
            self.seen_grid = image_grid_thw
            return pixel_values

    visual = FakeVisual()
    pixel_values = torch.randn(1, 2)
    grid = torch.tensor([[1, 1, 1]])

    output = image_encoder._call_visual(visual, pixel_values, grid)

    assert output is pixel_values
    assert visual.seen_grid is grid


def test_image_encoder_hidden_size_falls_back_to_visual_config():
    visual = SimpleNamespace(config=SimpleNamespace(hidden_size=64))
    vision_cfg = SimpleNamespace(spatial_merge_size=2)

    assert image_encoder._resolve_vision_hidden_size(vision_cfg, visual) == 64


def test_image_encoder_emits_canonical_deepstack_keys():
    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

        def forward(self, pixel_values, grid_thw):
            del grid_thw
            return pixel_values, [pixel_values + 1]

    model = image_encoder.Qwen35OmniImageEncoder.__new__(
        image_encoder.Qwen35OmniImageEncoder
    )
    nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model.visual = FakeVisual()
    model.spatial_merge_size = 1
    model.out_hidden_size = 2
    model.deepstack_layers = 0
    pixel_values = torch.randn(1, 2)

    output = model.forward(
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )

    assert output["image_deepstack_visual_embeds"][0].shape == (1, 2)
    assert output["deepstack_visual_embeds_image"] is output[
        "image_deepstack_visual_embeds"
    ]


def test_image_encoder_splits_packed_deepstack_output():
    embeds = torch.tensor([[1.0, 2.0, 10.0, 20.0, 100.0, 200.0]])

    main, deepstack = image_encoder._split_packed_deepstack_output(
        embeds,
        [],
        base_hidden_size=2,
        deepstack_layers=2,
    )

    assert main.tolist() == [[1.0, 2.0]]
    assert [layer.tolist() for layer in deepstack] == [
        [[10.0, 20.0]],
        [[100.0, 200.0]],
    ]


def test_image_encoder_forward_splits_qwen3_vision_packed_output():
    class FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

        def forward(self, pixel_values, grid_thw):
            del grid_thw
            return torch.cat([pixel_values, pixel_values + 10, pixel_values + 100], -1)

    model = image_encoder.Qwen35OmniImageEncoder.__new__(
        image_encoder.Qwen35OmniImageEncoder
    )
    nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model.visual = FakeVisual()
    model.spatial_merge_size = 1
    model.out_hidden_size = 2
    model.deepstack_layers = 2

    output = model.forward(
        pixel_values=torch.tensor([[1.0, 2.0]]),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )

    assert output["image_embeds"].tolist() == [[1.0, 2.0]]
    assert [layer.tolist() for layer in output["image_deepstack_visual_embeds"]] == [
        [[11.0, 12.0]],
        [[101.0, 102.0]],
    ]


def test_audio_encoder_resolves_next_audio_class(monkeypatch):
    module_name = "fake_qwen35_audio"
    module = ModuleType(module_name)

    class FakeAudio:
        pass

    module.Qwen3OmniNextAudioEncoder = FakeAudio
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(audio_encoder, "_MODELING_MODULE_CANDIDATES", (module_name,))

    assert audio_encoder._resolve_next_class(
        ("Qwen3OmniNextAudioEncoder",)
    ) is FakeAudio


def test_audio_downsample_lengths_matches_qwen35_formula():
    lengths = torch.tensor([100, 101, 200])

    assert audio_encoder._default_downsample_lengths(lengths).tolist() == [7, 8, 14]


def test_audio_encoder_resolves_downsample_config_aliases():
    config = SimpleNamespace(downsample_times=3, chunk_size=80)

    assert audio_encoder._resolve_downsample_config(config) == (3, 80)


def test_audio_encoder_wrapper_passes_aftercnn_lens_to_next_tower():
    class FakeAudioTower(nn.Module):
        @property
        def dtype(self):
            return torch.float32

        def forward(self, input_features, *, feature_lens, aftercnn_lens):
            self.input_features = input_features
            self.feature_lens = feature_lens
            self.aftercnn_lens = aftercnn_lens
            return torch.ones(int(aftercnn_lens.sum().item()), 3)

    model = audio_encoder.Qwen35OmniAudioEncoder.__new__(
        audio_encoder.Qwen35OmniAudioEncoder
    )
    nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model.audio_tower = FakeAudioTower()
    model._downsample_times = 2
    model._downsample_chunk_size = 4
    captured = {}

    def _fake_downsample(lengths, *, downsample_times, chunk_size):
        captured.update(
            downsample_times=downsample_times,
            chunk_size=chunk_size,
        )
        return lengths // downsample_times

    model._downsample_lengths = _fake_downsample

    result = model.forward(
        input_features=torch.ones(1, 4, 4),
        feature_attention_mask=torch.tensor([[True, True, False, False]]),
    )

    assert model.audio_tower.feature_lens.tolist() == [2]
    assert model.audio_tower.aftercnn_lens.tolist() == [1]
    assert captured == {"downsample_times": 2, "chunk_size": 4}
    assert result["audio_output_lengths"].tolist() == [1]
    assert result["audio_embeds"].shape == (1, 3)


def test_audio_encoder_packs_batched_features_without_attention_mask():
    class FakeAudioTower(nn.Module):
        @property
        def dtype(self):
            return torch.float32

        def forward(self, input_features, *, feature_lens, aftercnn_lens):
            self.input_features = input_features
            self.feature_lens = feature_lens
            self.aftercnn_lens = aftercnn_lens
            return torch.ones(int(aftercnn_lens.sum().item()), 3)

    model = audio_encoder.Qwen35OmniAudioEncoder.__new__(
        audio_encoder.Qwen35OmniAudioEncoder
    )
    nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model.audio_tower = FakeAudioTower()
    model._downsample_times = 1
    model._downsample_chunk_size = 4
    model._downsample_lengths = (
        lambda lengths, *, downsample_times, chunk_size: lengths
    )

    input_features = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2, 2, 4)
    result = model.forward(
        input_features=input_features,
        audio_feature_lengths=torch.tensor([2, 3]),
    )

    expected_features = torch.cat(
        [input_features[0, :, :2], input_features[1, :, :3]],
        dim=1,
    )
    assert torch.equal(model.audio_tower.input_features, expected_features)
    assert model.audio_tower.feature_lens.tolist() == [2, 3]
    assert model.audio_tower.aftercnn_lens.tolist() == [2, 3]
    assert result["audio_embeds"].shape == (5, 3)


def test_audio_encoder_accepts_next_input_audio_features_alias():
    class FakeAudioTower(nn.Module):
        @property
        def dtype(self):
            return torch.float32

        def forward(self, input_features, *, feature_lens, aftercnn_lens):
            self.input_features = input_features
            self.feature_lens = feature_lens
            self.aftercnn_lens = aftercnn_lens
            return torch.ones(int(aftercnn_lens.sum().item()), 3)

    model = audio_encoder.Qwen35OmniAudioEncoder.__new__(
        audio_encoder.Qwen35OmniAudioEncoder
    )
    nn.Module.__init__(model)
    model._device = torch.device("cpu")
    model.audio_tower = FakeAudioTower()
    model._downsample_times = 1
    model._downsample_chunk_size = 4
    model._downsample_lengths = (
        lambda lengths, *, downsample_times, chunk_size: lengths
    )

    input_audio_features = torch.arange(1 * 2 * 4, dtype=torch.float32).view(1, 2, 4)
    result = model.forward(
        input_audio_features=input_audio_features,
        feature_attention_mask=torch.tensor([[True, True, False, False]]),
    )

    assert torch.equal(
        model.audio_tower.input_features,
        input_audio_features[0, :, :2],
    )
    assert model.audio_tower.feature_lens.tolist() == [2]
    assert result["audio_embeds"].shape == (2, 3)


def test_local_next_audio_encoder_forward_cpu():
    model = Qwen3OmniNextAudioEncoder(_audio_config())
    input_features = torch.randn(8, 8)
    feature_lens = torch.tensor([8])
    aftercnn_lens = torch.tensor([1])

    output = model(
        input_features,
        feature_lens=feature_lens,
        aftercnn_lens=aftercnn_lens,
    )

    assert output.shape == (1, 6)


def test_local_next_audio_encoder_uses_hf_weight_names():
    names = set(dict(Qwen3OmniNextAudioEncoder(_audio_config()).named_parameters()))

    assert {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.out_proj.weight",
        "layers.0.fc1.weight",
        "layers.0.fc2.weight",
        "conv_out.weight",
        "proj2.bias",
    } <= names
    assert not any(".qkv." in name for name in names)


def test_audio_encoder_splits_packed_qkv_state_dict():
    packed_weight = torch.arange(48, dtype=torch.float32).view(12, 4)
    packed_bias = torch.arange(12, dtype=torch.float32)
    state_dict = {
        "layers.0.self_attn.qkv.weight": packed_weight,
        "layers.0.self_attn.qkv.bias": packed_bias,
    }

    normalized = audio_encoder._split_packed_audio_qkv(state_dict)

    assert "layers.0.self_attn.qkv.weight" not in normalized
    assert torch.equal(
        normalized["layers.0.self_attn.q_proj.weight"],
        packed_weight[:4],
    )
    assert torch.equal(
        normalized["layers.0.self_attn.k_proj.bias"],
        packed_bias[4:8],
    )
    assert torch.equal(
        normalized["layers.0.self_attn.v_proj.weight"],
        packed_weight[8:],
    )


def test_audio_encoder_loads_packed_qkv_checkpoint(monkeypatch):
    source = Qwen3OmniNextAudioEncoder(_audio_config())
    state = dict(source.state_dict())
    for suffix in ("weight", "bias"):
        q_key = f"layers.0.self_attn.q_proj.{suffix}"
        k_key = f"layers.0.self_attn.k_proj.{suffix}"
        v_key = f"layers.0.self_attn.v_proj.{suffix}"
        state[f"layers.0.self_attn.qkv.{suffix}"] = torch.cat(
            [state.pop(q_key), state.pop(k_key), state.pop(v_key)],
            dim=0,
        )
    monkeypatch.setattr(
        audio_encoder,
        "load_weights_by_prefix",
        lambda model_path, *, prefix: state,
    )
    target = Qwen3OmniNextAudioEncoder(_audio_config())

    loaded = audio_encoder._load_audio_weights(
        target,
        "dummy",
        torch_dtype=None,
        device="cpu",
    )

    assert loaded is target
    assert loaded.training is False
    assert torch.equal(
        loaded.layers[0].self_attn.q_proj.weight,
        source.layers[0].self_attn.q_proj.weight,
    )


def test_local_next_audio_attention_handles_multiple_chunks():
    config = SimpleNamespace(d_model=4, encoder_attention_heads=1)
    attention = Qwen3OmniNextAudioAttention(config)
    hidden_states = torch.randn(5, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    output = attention(hidden_states, cu_seqlens)

    assert output.shape == hidden_states.shape
