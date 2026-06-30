# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.request_builders import (
    Qwen35TalkerPrefillBuilder,
)
from sglang_omni.models.qwen3_5_omni import merge as qwen35_merge
from sglang_omni.models.qwen3_5_omni import request_builders
from sglang_omni.models.qwen3_5_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTalkerModel:
    def __init__(self):
        self.activation_dtype = torch.float32
        codec_embedding = nn.Embedding(128, 4)
        with torch.no_grad():
            for token_id in range(codec_embedding.num_embeddings):
                codec_embedding.weight[token_id].fill_(float(token_id))
        self.model = SimpleNamespace(codec_embedding=codec_embedding)
        self.text_projection = nn.Identity()
        self.hidden_projection = nn.Identity()
        self.seen_text_ids = []
        self.seen_speaker_ids = []
        self.seen_prompt_codes = []

    def get_input_embeddings(self):
        return self.model.codec_embedding

    def embed_text_ids(self, token_ids):
        self.seen_text_ids.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1)
        return values.expand(token_ids.shape[0], 4)

    def speaker_codec_input_embeddings(self, speaker_id):
        self.seen_speaker_ids.append(int(speaker_id))
        return torch.tensor(
            [[100.0] * 4, [101.0] * 4],
            dtype=torch.float32,
        )

    def codec_code_embeddings(self, codes):
        code_rows = codes.detach().cpu().to(dtype=torch.float32)
        self.seen_prompt_codes.append(codes.detach().cpu().tolist())
        if code_rows.ndim == 2:
            values = code_rows.sum(dim=1)
        elif code_rows.ndim == 3:
            values = code_rows.sum(dim=1).reshape(-1)
        else:
            raise ValueError("unexpected codes shape")
        return values.unsqueeze(-1).expand(values.shape[0], 4)


class _FakeTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.reshape(-1).detach().cpu().tolist()
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _builder(model, **kwargs):
    model_path = kwargs.pop("model_path", str(Path("/tmp")))
    return Qwen35TalkerPrefillBuilder(
        model=model,
        model_path=model_path,
        **kwargs,
        audio_token_id=1,
        image_token_id=2,
        video_token_id=3,
        tts_bos_token_id=10,
        tts_eos_token_id=11,
        tts_pad_token_id=12,
        im_start_token_id=20,
        im_end_token_id=21,
        system_token_id=30,
        user_token_id=31,
        assistant_token_id=32,
        codec_bos_id=40,
        codec_nothink_id=41,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
    )


def _mrope_config():
    return SimpleNamespace(
        image_token_id=11,
        video_token_id=12,
        vision_end_token_id=13,
        audio_token_id=14,
        vision_start_token_id=10,
        vision_config=SimpleNamespace(spatial_merge_size=1),
    )


def _text_chunks(text: str) -> list[SimpleNamespace]:
    chunks = []
    for char in text:
        token_id = ord(char)
        chunks.append(
            SimpleNamespace(
                data=torch.full((4,), float(token_id)),
                metadata={
                    "token_id": token_id,
                    "layer_hidden": torch.full((4,), float(token_id + 1000)),
                },
            )
        )
    return chunks


def test_qwen35_audio_output_gating_accepts_openai_modalities_param():
    text_only = OmniRequest(inputs={}, params={"modalities": ["text"]})
    audio = OmniRequest(inputs={}, params={"modalities": ["text", "audio"]})
    alias = OmniRequest(inputs={}, params={"output_modalities": "audio"})

    assert request_builders.should_generate_audio_output(None)
    assert not request_builders.should_generate_audio_output(text_only)
    assert request_builders.should_generate_audio_output(audio)
    assert request_builders.should_generate_audio_output(alias)


def test_qwen35_audio_output_gating_normalizes_string_modalities():
    csv = OmniRequest(inputs={}, params={"modalities": " text, audio "})
    whitespace = OmniRequest(inputs={}, params={"modalities": "text audio"})
    list_item_csv = OmniRequest(inputs={}, params={"modalities": ["text,audio"]})
    metadata_list_item_csv = OmniRequest(
        inputs={},
        metadata={"output_modalities": ["text,audio"]},
    )
    blank = OmniRequest(inputs={}, params={"modalities": "   "})

    assert request_builders.output_modalities(csv) == {"text", "audio"}
    assert request_builders.output_modalities(whitespace) == {"text", "audio"}
    assert request_builders.output_modalities(list_item_csv) == {"text", "audio"}
    assert request_builders.output_modalities(metadata_list_item_csv) == {
        "text",
        "audio",
    }
    assert request_builders.should_generate_audio_output(csv)
    assert request_builders.output_modalities(blank) is None


def test_qwen35_audio_output_gating_prefers_metadata_modalities():
    request = OmniRequest(
        inputs={},
        params={"modalities": ["text", "audio"]},
        metadata={"output_modalities": ["text"]},
    )

    assert not request_builders.should_generate_audio_output(request)


def test_qwen35_audio_output_gating_accepts_boolean_flags():
    enabled = OmniRequest(inputs={}, params={"enable_audio_output": True})
    disabled = OmniRequest(inputs={}, params={"enable_audio_output": False})
    alias_enabled = OmniRequest(inputs={}, params={"return_audio": "on"})
    alias_disabled = OmniRequest(inputs={}, params={"audio_output": "text"})
    do_wave_enabled = OmniRequest(inputs={}, params={"do_wave": "true"})
    do_wave_disabled = OmniRequest(inputs={}, params={"do_wave": 0})

    assert request_builders.should_generate_audio_output(enabled)
    assert not request_builders.should_generate_audio_output(disabled)
    assert request_builders.should_generate_audio_output(alias_enabled)
    assert not request_builders.should_generate_audio_output(alias_disabled)
    assert request_builders.should_generate_audio_output(do_wave_enabled)
    assert not request_builders.should_generate_audio_output(do_wave_disabled)


def test_qwen35_audio_output_gating_prefers_modalities_over_boolean_flag():
    request = OmniRequest(
        inputs={},
        params={
            "modalities": ["text"],
            "enable_audio_output": True,
        },
    )

    assert not request_builders.should_generate_audio_output(request)


def test_qwen35_audio_output_resolvers_honor_params_modalities():
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"modalities": ["text"]}),
        data={},
    )

    assert request_builders.resolve_mm_aggregate_next_stages(
        "req-0", payload
    ) == "thinker"
    assert request_builders.resolve_thinker_stream_done_targets(
        "req-0", payload
    ) == ["decode"]
    assert request_builders.resolve_terminal_stages(payload.request) == ["decode"]


def test_qwen35_rtc_prerun_resolvers_terminate_at_thinker_by_default(monkeypatch):
    monkeypatch.delenv("QWEN35_RTC_PRERUN_PREFILL_ONLY", raising=False)
    monkeypatch.delenv("QWEN35_RTC_PRERUN_THINKER_TERMINAL", raising=False)
    payload = StagePayload(
        request_id="req-prerun",
        request=OmniRequest(
            inputs={},
            params={"modalities": ["text"]},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data={},
    )

    assert request_builders.resolve_thinker_next_stages("req-prerun", payload) is None
    assert request_builders.resolve_thinker_stream_done_targets(
        "req-prerun", payload
    ) == []
    assert request_builders.resolve_terminal_stages(payload.request) == ["thinker"]


def test_qwen35_rtc_prerun_resolvers_can_keep_decode_terminal(monkeypatch):
    monkeypatch.delenv("QWEN35_RTC_PRERUN_PREFILL_ONLY", raising=False)
    monkeypatch.setenv("QWEN35_RTC_PRERUN_THINKER_TERMINAL", "0")
    payload = StagePayload(
        request_id="req-prerun",
        request=OmniRequest(
            inputs={},
            params={"modalities": ["text"]},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data={},
    )

    assert request_builders.resolve_thinker_next_stages(
        "req-prerun", payload
    ) == "decode"
    assert request_builders.resolve_thinker_stream_done_targets(
        "req-prerun", payload
    ) == ["decode"]
    assert request_builders.resolve_terminal_stages(payload.request) == ["decode"]


def test_qwen35_rtc_prerun_resolvers_can_keep_generation(monkeypatch):
    monkeypatch.setenv("QWEN35_RTC_PRERUN_PREFILL_ONLY", "0")
    monkeypatch.setenv("QWEN35_RTC_PRERUN_THINKER_TERMINAL", "1")
    payload = StagePayload(
        request_id="req-prerun",
        request=OmniRequest(
            inputs={},
            params={"modalities": ["text", "audio"]},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data={},
    )

    assert request_builders.resolve_thinker_next_stages(
        "req-prerun", payload
    ) == "decode"
    assert request_builders.resolve_thinker_stream_done_targets(
        "req-prerun", payload
    ) == ["talker_ar", "decode"]
    assert request_builders.resolve_terminal_stages(payload.request) == [
        "decode",
        "code2wav",
    ]


def test_qwen35_mm_aggregate_to_thinker_projection_is_isolated():
    model_inputs = {"video_embeds": torch.ones(2, 4)}
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "prompt_text": "hi",
        },
        thinker_inputs={
            "model_inputs": model_inputs,
            "capture_model_output_keys": ("hidden_states",),
        },
        stream_state={"token_ids": []},
        encoder_inputs={"image_encoder": {"large": torch.ones(8)}},
    )
    payload = StagePayload(
        request_id="req-project",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )

    projected = request_builders.project_mm_aggregate_to_thinker(payload)
    projected_state = Qwen3OmniPipelineState.from_dict(projected.data)

    assert projected_state.prompt == state.prompt
    assert projected_state.prompt is not state.prompt
    assert projected_state.thinker_inputs["model_inputs"] is not model_inputs
    assert (
        projected_state.thinker_inputs["model_inputs"]["video_embeds"]
        is model_inputs["video_embeds"]
    )
    assert projected_state.thinker_inputs is not state.thinker_inputs
    assert projected_state.stream_state is not state.stream_state
    assert projected_state.encoder_inputs == {}


def test_qwen35_merge_preserves_audio_is_dependent_mask():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])},
        mm_inputs={
            "image": {},
            "audio": {
                "audio_feature_lengths": torch.tensor([2, 3]),
                "audio_is_dependent": [True, False],
            },
            "video": {"use_audio_in_video": [False, True]},
        },
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )

    merged = qwen35_merge.merge_for_thinker({"preprocessing": payload})
    merged_state = Qwen3OmniPipelineState.from_dict(merged.data)
    model_inputs = merged_state.thinker_inputs["model_inputs"]

    assert model_inputs["audio_is_dependent"].dtype is torch.bool
    assert model_inputs["audio_is_dependent"].tolist() == [True, False]
    assert model_inputs["use_audio_in_video"] == [False, True]


def test_qwen35_merge_preserves_use_audio_in_video_without_audio_mask():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1])},
        mm_inputs={
            "image": {},
            "audio": {},
            "video": {"use_audio_in_video": [False, True]},
        },
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )

    merged = qwen35_merge.merge_for_thinker({"preprocessing": payload})
    merged_state = Qwen3OmniPipelineState.from_dict(merged.data)
    model_inputs = merged_state.thinker_inputs["model_inputs"]

    assert model_inputs["use_audio_in_video"] == [False, True]
    assert "audio_is_dependent" not in model_inputs


def test_qwen35_openai_audio_config_voice_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        metadata={"audio_config": {"voice": "Cherry", "format": "wav"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_type"] == "Cherry"


def test_qwen35_openai_audio_metadata_alias_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        metadata={"audio": {"voice": "Cherry", "format": "wav"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_type"] == "Cherry"


def test_qwen35_openai_audio_metadata_alias_ignores_input_payload():
    request = OmniRequest(
        inputs={},
        metadata={"audio": {"data": "UklGRg==", "format": "wav"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert "voice_type" not in params


def test_qwen35_openai_audio_config_does_not_override_explicit_voice():
    request = OmniRequest(
        inputs={},
        params={"voice_type": "Tina"},
        metadata={"audio_config": {"voice": "Cherry"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_type"] == "Tina"


def test_qwen35_openai_audio_config_language_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        params={"voice_type": "Tina"},
        metadata={"audio_config": {"voice": "Cherry", "language": "zh-CN"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_type"] == "Tina"
    assert params["language"] == "zh-CN"


def test_qwen35_openai_audio_config_language_type_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        metadata={"audio_config": {"language_type": "en"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["language"] == "en"


def test_qwen35_openai_audio_config_language_id_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        metadata={"audio_config": {"language_id": 77}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["language_id"] == 77


def test_qwen35_openai_audio_config_target_lang_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        metadata={"audio_config": {"target_lang": "en"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["language"] == "en"


def test_qwen35_translation_options_prevent_audio_language_override():
    request = OmniRequest(
        inputs={},
        params={"translation_options": {"target_lang": "en", "source_lang": "fr"}},
        metadata={"audio_config": {"language": "zh-CN"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["translation_options"] == {
        "target_lang": "en",
        "source_lang": "fr",
    }
    assert "language" not in params


def test_qwen35_openai_audio_config_does_not_override_explicit_language():
    request = OmniRequest(
        inputs={},
        params={"language": "en-US"},
        metadata={"audio_config": {"language": "zh-CN"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["language"] == "en-US"


def test_qwen35_openai_audio_config_style_and_instruction_flow_to_params():
    request = OmniRequest(
        inputs={},
        metadata={
            "audio_config": {
                "voice_style": "happy",
                "instruction": "Speak softly",
            }
        },
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_style"] == "happy"
    assert params["instruction"] == "Speak softly"


def test_qwen35_openai_audio_config_xvector_flows_to_talker_params():
    request = OmniRequest(
        inputs={},
        metadata={"audio_config": {"xvector_info": "/voices/ref-a"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["xvector_info"] == "/voices/ref-a"


def test_qwen35_openai_audio_config_voice_clone_path_flows_to_params():
    request = OmniRequest(
        inputs={},
        metadata={
            "audio_config": {
                "voice_clone": {
                    "path": "/voices/ref-b",
                }
            }
        },
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_clone_info"] == "/voices/ref-b"


def test_qwen35_openai_audio_config_keeps_voice_clone_path_overrides():
    request = OmniRequest(
        inputs={},
        metadata={
            "audio_config": {
                "voice_clone": {
                    "path": "/voices/ref-b",
                    "language": "en",
                }
            }
        },
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["voice_clone_info"] == {
        "path": "/voices/ref-b",
        "language": "en",
    }


def test_qwen35_openai_audio_config_does_not_override_voice_clone_params():
    request = OmniRequest(
        inputs={},
        params={"xvector_info": "/voices/explicit"},
        metadata={"audio_config": {"xvector_info": "/voices/metadata"}},
    )

    params = request_builders._params_with_openai_audio_config(request)

    assert params["xvector_info"] == "/voices/explicit"


def test_qwen35_talker_prefill_uses_talker_text_embedding():
    model = _FakeTalkerModel()
    builder = _builder(model)

    rows = builder._load_prompt_token_embeddings(torch.tensor([5, 6]))

    assert rows.tolist() == [[5.0] * 4, [6.0] * 4]
    assert model.seen_text_ids == [[5, 6]]


def test_qwen35_talker_reconstruct_maps_media_pad_values_before_embedding():
    model = _FakeTalkerModel()
    builder = _builder(model)
    pad_id = -5364141779887080204
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor([20, 31, pad_id, 32], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.tensor([[900.0] * 4]),
                "pad_values": {"video": pad_id},
            }
        },
    )

    prompt_ids, prompt_embed, prompt_hidden, _ = builder._reconstruct_prompt_states(
        state
    )

    assert prompt_ids.tolist() == [20, 31, 3, 32]
    assert model.seen_text_ids == [[20, 31, 3, 32]]
    assert hidden_seen == [[20, 31, 3, 32]]
    assert prompt_embed[2].tolist() == [3.0] * 4
    assert prompt_hidden[2].tolist() == [900.0] * 4


def test_qwen35_talker_media_pad_values_accept_unsigned_int64_alias():
    builder = _builder(_FakeTalkerModel())
    unsigned_pad = (1 << 63) + 17
    signed_pad = unsigned_pad - (1 << 64)

    canonical = builder._canonicalize_prompt_ids_for_talker(
        torch.tensor([20, signed_pad, 32], dtype=torch.long),
        {"pad_values": {"video": unsigned_pad}},
    )

    assert canonical.tolist() == [20, 3, 32]


def test_qwen35_talker_media_pad_values_accept_multiple_aliases():
    builder = _builder(_FakeTalkerModel())
    unsigned_pad = (1 << 63) + 19
    signed_pad = unsigned_pad - (1 << 64)
    other_pad = -5364141779887080204

    canonical = builder._canonicalize_prompt_ids_for_talker(
        torch.tensor([20, signed_pad, other_pad, 32], dtype=torch.long),
        {"pad_values": {"video": [unsigned_pad, other_pad]}},
    )

    assert canonical.tolist() == [20, 3, 3, 32]


def test_qwen35_talker_maps_cache_pad_runs_by_special_token_context():
    builder = _builder(
        _FakeTalkerModel(),
        thinker_config=SimpleNamespace(
            vision_start_token_id=100,
            vision_end_token_id=101,
            audio_end_token_id=102,
        ),
    )
    video_pad = -5364141779887080204
    audio_pad = -5316291547151614670

    canonical = builder._canonicalize_prompt_ids_for_talker(
        torch.tensor(
            [
                20,
                100,
                video_pad,
                video_pad,
                video_pad,
                101,
                audio_pad,
                audio_pad,
                102,
                32,
            ],
            dtype=torch.long,
        ),
        {
            "video_embeds": torch.ones((3, 4)),
            "audio_embeds": torch.ones((2, 4)),
        },
    )

    assert canonical.tolist() == [20, 100, 3, 3, 3, 101, 1, 1, 102, 32]


def test_qwen35_talker_maps_fully_compressed_media_cache_prompt():
    builder = _builder(_FakeTalkerModel())
    compressed_ids = torch.tensor(
        [
            -5364141779887080204,
            -5364141779887080203,
            -5364141779887080202,
            -5316291547151614670,
            -5316291547151614669,
            -5364141779887080201,
            -5364141779887080200,
            -5364141779887080199,
            -5316291547151614668,
            -5316291547151614667,
        ],
        dtype=torch.long,
    )

    canonical = builder._canonicalize_prompt_ids_for_talker(
        compressed_ids,
        {
            "video_embeds": torch.ones((6, 4)),
            "video_grid_thw": torch.tensor(
                [[1, 1, 3], [1, 1, 3]],
                dtype=torch.long,
            ),
            "audio_embeds": torch.ones((4, 4)),
            "audio_feature_lengths": torch.tensor([200, 200], dtype=torch.long),
        },
    )

    assert canonical.tolist() == [3, 3, 3, 1, 1, 3, 3, 3, 1, 1]


def test_qwen35_talker_maps_compressed_rtc_media_prompt_with_text_filler():
    builder = _builder(_FakeTalkerModel())
    compressed_ids = torch.arange(
        -6000000000000000000,
        -6000000000000000000 + 36560,
        dtype=torch.long,
    )

    canonical = builder._canonicalize_prompt_ids_for_talker(
        compressed_ids,
        {
            "video_embeds": torch.ones((35200, 4)),
            "video_grid_thw": torch.tensor([[1, 40, 88]] * 40, dtype=torch.long),
            "audio_embeds": torch.ones((560, 4)),
            "audio_feature_lengths": torch.tensor([200] * 40, dtype=torch.long),
        },
    )

    assert int((canonical == 3).sum().item()) == 35200
    assert int((canonical == 1).sum().item()) == 560
    assert int((canonical == 0).sum().item()) == 800
    assert canonical[:19].tolist() == [0] * 19
    assert canonical[19:899].tolist() == [3] * 880
    assert canonical[899].item() == 0
    assert canonical[900:914].tolist() == [1] * 14


def test_qwen35_talker_maps_compressed_rtc_media_prompt_without_slot_metadata():
    builder = _builder(_FakeTalkerModel())
    compressed_ids = torch.arange(
        -6000000000000000000,
        -6000000000000000000 + 36560,
        dtype=torch.long,
    )

    canonical = builder._canonicalize_prompt_ids_for_talker(
        compressed_ids,
        {
            "video_embeds": torch.ones((35200, 4)),
            "audio_embeds": torch.ones((560, 4)),
        },
    )

    assert int((canonical == 3).sum().item()) == 35200
    assert int((canonical == 1).sum().item()) == 560
    assert int((canonical == 0).sum().item()) == 800
    assert canonical[:19].tolist() == [0] * 19
    assert canonical[19:899].tolist() == [3] * 880
    assert canonical[899].item() == 0
    assert canonical[900:914].tolist() == [1] * 14


def test_qwen35_talker_preserves_original_text_in_compressed_rtc_prompt():
    builder = _builder(_FakeTalkerModel())
    compressed_ids = torch.arange(
        -6000000000000000000,
        -6000000000000000000 + 36560,
        dtype=torch.long,
    )
    original_ids = torch.full((36560,), 99, dtype=torch.long)

    canonical = builder._canonicalize_prompt_ids_for_talker(
        compressed_ids,
        {
            "original_input_ids": original_ids,
            "video_embeds": torch.ones((35200, 4)),
            "audio_embeds": torch.ones((560, 4)),
        },
    )

    assert int((canonical == 3).sum().item()) == 35200
    assert int((canonical == 1).sum().item()) == 560
    assert int((canonical == 99).sum().item()) == 800
    assert int((canonical == 0).sum().item()) == 0
    assert canonical[:19].tolist() == [99] * 19
    assert canonical[19:899].tolist() == [3] * 880
    assert canonical[899].item() == 99
    assert canonical[900:914].tolist() == [1] * 14


def test_qwen35_talker_restores_original_ids_for_partially_compressed_rtc_prompt():
    builder = _builder(_FakeTalkerModel())
    original_ids = torch.tensor(
        [3, 3, 3, 99, 1, 1, 3, 3, 3, 99, 1, 1],
        dtype=torch.long,
    )
    partial_ids = torch.tensor(
        [
            3,
            -6000000000000000000,
            -6000000000000000001,
            -6000000000000000002,
            1,
            -6000000000000000003,
            -6000000000000000004,
            3,
            3,
            -6000000000000000005,
            -6000000000000000006,
            1,
        ],
        dtype=torch.long,
    )

    canonical = builder._canonicalize_prompt_ids_for_talker(
        partial_ids,
        {
            "original_input_ids": original_ids,
            "video_embeds": torch.ones((6, 4)),
            "video_grid_thw": torch.tensor(
                [[1, 1, 3], [1, 1, 3]],
                dtype=torch.long,
            ),
            "audio_embeds": torch.ones((4, 4)),
            "audio_feature_lengths": torch.tensor([200, 200], dtype=torch.long),
        },
    )

    assert canonical.tolist() == original_ids.tolist()


def test_qwen35_talker_maps_partially_compressed_rtc_prompt_by_slot_layout():
    builder = _builder(_FakeTalkerModel())
    partial_ids = torch.tensor(
        [
            3,
            -6000000000000000000,
            -6000000000000000001,
            -6000000000000000002,
            1,
            -6000000000000000003,
            -6000000000000000004,
            3,
            3,
            -6000000000000000005,
            -6000000000000000006,
            1,
        ],
        dtype=torch.long,
    )

    canonical = builder._canonicalize_prompt_ids_for_talker(
        partial_ids,
        {
            "video_embeds": torch.ones((6, 4)),
            "video_grid_thw": torch.tensor(
                [[1, 1, 3], [1, 1, 3]],
                dtype=torch.long,
            ),
            "audio_embeds": torch.ones((4, 4)),
            "audio_feature_lengths": torch.tensor([200, 200], dtype=torch.long),
        },
    )

    assert canonical.tolist() == [3, 3, 3, 0, 1, 1, 3, 3, 3, 0, 1, 1]


def test_qwen35_talker_reconstruct_maps_missing_pad_values_by_feature_rows():
    model = _FakeTalkerModel()
    builder = _builder(model)
    pad_id = -5316291547151614670
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor([20, pad_id, pad_id, 32], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.tensor([[900.0] * 4, [901.0] * 4]),
            }
        },
    )

    prompt_ids, _, prompt_hidden, _ = builder._reconstruct_prompt_states(state)

    assert prompt_ids.tolist() == [20, 3, 3, 32]
    assert model.seen_text_ids == [[20, 3, 3, 32]]
    assert hidden_seen == [[20, 3, 3, 32]]
    assert prompt_hidden[1].tolist() == [900.0] * 4
    assert prompt_hidden[2].tolist() == [901.0] * 4


def test_qwen35_talker_reconstruct_maps_video_frame_image_placeholders():
    model = _FakeTalkerModel()
    builder = _builder(model)
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor([20, 3, 3, 2, 32], dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.tensor(
                    [[900.0] * 4, [901.0] * 4, [902.0] * 4]
                ),
            }
        },
    )

    prompt_ids, _, prompt_hidden, _ = builder._reconstruct_prompt_states(state)

    assert prompt_ids.tolist() == [20, 3, 3, 3, 32]
    assert model.seen_text_ids == [[20, 3, 3, 3, 32]]
    assert hidden_seen == [[20, 3, 3, 3, 32]]
    assert prompt_hidden[1].tolist() == [900.0] * 4
    assert prompt_hidden[2].tolist() == [901.0] * 4
    assert prompt_hidden[3].tolist() == [902.0] * 4


def test_qwen35_talker_reconstruct_classifies_mixed_media_pad_ids_by_slot_rows():
    model = _FakeTalkerModel()
    builder = _builder(model)
    video_pad_0 = -5327409853779854172
    audio_pad = -5259013824590398178
    video_pad_1 = -5250004939597366257
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor(
                [20, video_pad_0, video_pad_0, audio_pad, video_pad_1, video_pad_1, 32],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(7, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.tensor(
                    [[900.0] * 4, [901.0] * 4, [902.0] * 4, [903.0] * 4]
                ),
                "audio_embeds": torch.tensor([[800.0] * 4]),
                "video_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2]]),
                "audio_feature_lengths": torch.tensor([1]),
            }
        },
    )

    prompt_ids, _, prompt_hidden, _ = builder._reconstruct_prompt_states(state)

    assert prompt_ids.tolist() == [20, 3, 3, 1, 3, 3, 32]
    assert model.seen_text_ids == [[20, 3, 3, 1, 3, 3, 32]]
    assert hidden_seen == [[20, 3, 3, 1, 3, 3, 32]]
    assert prompt_hidden[1].tolist() == [900.0] * 4
    assert prompt_hidden[2].tolist() == [901.0] * 4
    assert prompt_hidden[3].tolist() == [800.0] * 4
    assert prompt_hidden[4].tolist() == [902.0] * 4
    assert prompt_hidden[5].tolist() == [903.0] * 4


def test_qwen35_talker_reconstruct_classifies_video_slots_after_spatial_merge():
    model = _FakeTalkerModel()
    thinker_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    builder = _builder(model, thinker_config=thinker_config)
    video_pad = -5238885191902315364
    audio_pad = -5286455903937610684
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor(
                [
                    20,
                    video_pad,
                    video_pad,
                    video_pad,
                    video_pad,
                    audio_pad,
                    audio_pad,
                    32,
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "audio_embeds": torch.tensor([[800.0] * 4, [801.0] * 4]),
                "video_embeds": torch.tensor(
                    [[900.0] * 4, [901.0] * 4, [902.0] * 4, [903.0] * 4]
                ),
                "audio_feature_lengths": torch.tensor([2]),
                "video_grid_thw": torch.tensor([[1, 4, 4]]),
            }
        },
    )

    prompt_ids, _, prompt_hidden, _ = builder._reconstruct_prompt_states(state)

    assert prompt_ids.tolist() == [20, 3, 3, 3, 3, 1, 1, 32]
    assert model.seen_text_ids == [[20, 3, 3, 3, 3, 1, 1, 32]]
    assert hidden_seen == [[20, 3, 3, 3, 3, 1, 1, 32]]
    assert prompt_hidden[1].tolist() == [900.0] * 4
    assert prompt_hidden[2].tolist() == [901.0] * 4
    assert prompt_hidden[3].tolist() == [902.0] * 4
    assert prompt_hidden[4].tolist() == [903.0] * 4
    assert prompt_hidden[5].tolist() == [800.0] * 4
    assert prompt_hidden[6].tolist() == [801.0] * 4


def test_qwen35_talker_reconstruct_infers_visual_merge_from_feature_rows():
    model = _FakeTalkerModel()
    builder = _builder(model)
    video_pad = -5327409853779854172
    audio_pad = -5259013824590398178
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor(
                [20, video_pad, video_pad, video_pad, video_pad, audio_pad, audio_pad, 32],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "audio_embeds": torch.tensor([[800.0] * 4, [801.0] * 4]),
                "video_embeds": torch.tensor(
                    [[900.0] * 4, [901.0] * 4, [902.0] * 4, [903.0] * 4]
                ),
                "audio_feature_lengths": torch.tensor([2]),
                "video_grid_thw": torch.tensor([[1, 4, 4]]),
            }
        },
    )

    prompt_ids, _, prompt_hidden, _ = builder._reconstruct_prompt_states(state)

    assert prompt_ids.tolist() == [20, 3, 3, 3, 3, 1, 1, 32]
    assert model.seen_text_ids == [[20, 3, 3, 3, 3, 1, 1, 32]]
    assert hidden_seen == [[20, 3, 3, 3, 3, 1, 1, 32]]
    assert prompt_hidden[1].tolist() == [900.0] * 4
    assert prompt_hidden[4].tolist() == [903.0] * 4
    assert prompt_hidden[5].tolist() == [800.0] * 4
    assert prompt_hidden[6].tolist() == [801.0] * 4


def test_qwen35_talker_prefill_handles_empty_assistant_segment():
    model = _FakeTalkerModel()
    builder = _builder(model)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        torch.full((1, 4), 10.0),
        torch.full((1, 4), 11.0),
        torch.full((1, 4), 12.0),
    )

    prefill = builder._build_prefill_input(
        thinker_embed=torch.ones((3, 4)),
        thinker_hidden=torch.ones((3, 4)),
        thinker_input_ids=torch.tensor([20, 31, 21], dtype=torch.long),
        multimodal_mask=torch.zeros(3, dtype=torch.bool),
        assistant_token_count=0,
        text_projection=nn.Identity(),
        hidden_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_bos_embed,
        tts_eos_embed=tts_eos_embed,
        tts_pad_embed=tts_pad_embed,
        assistant_instruct_embed=None,
        im_start_token_id=20,
        system_token_id=30,
        user_token_id=31,
        assistant_token_id=32,
        speaker_id=None,
        codec_nothink_id=41,
        codec_think_id=None,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
        include_assistant_eos=True,
        im_end_token_id=21,
    )

    assert prefill["input_embeds"].shape[0] > 0
    assert prefill["input_ids"].shape[0] == prefill["input_embeds"].shape[0]
    assert prefill["future_text_rows"].shape == (1, 4)


def test_qwen35_talker_reconstruct_classifies_pad_runs_by_boundary_tokens():
    model = _FakeTalkerModel()
    thinker_config = SimpleNamespace(
        vision_start_token_id=70,
        vision_end_token_id=71,
        audio_start_token_id=80,
        audio_end_token_id=81,
    )
    builder = _builder(model, thinker_config=thinker_config)
    video_pad = -5308550691032548132
    audio_pad = -5267736634739934270
    hidden_seen = []

    def fake_hidden_embeddings(token_ids):
        hidden_seen.append(token_ids.detach().cpu().tolist())
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + 1000.0
        return values.expand(token_ids.shape[0], 4).clone()

    builder._load_thinker_hidden_embeddings = fake_hidden_embeddings
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor(
                [70, video_pad, video_pad, video_pad, 71, 80, audio_pad, audio_pad, 81],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(9, dtype=torch.long),
            "prompt_text": "",
        },
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.tensor(
                    [[900.0] * 4, [901.0] * 4, [902.0] * 4]
                ),
                "audio_embeds": torch.tensor([[800.0] * 4, [801.0] * 4]),
                "audio_feature_lengths": torch.tensor([2]),
            }
        },
    )

    prompt_ids, _, prompt_hidden, _ = builder._reconstruct_prompt_states(state)

    assert prompt_ids.tolist() == [70, 3, 3, 3, 71, 80, 1, 1, 81]
    assert model.seen_text_ids == [[70, 3, 3, 3, 71, 80, 1, 1, 81]]
    assert hidden_seen == [[70, 3, 3, 3, 71, 80, 1, 1, 81]]
    assert prompt_hidden[1].tolist() == [900.0] * 4
    assert prompt_hidden[2].tolist() == [901.0] * 4
    assert prompt_hidden[3].tolist() == [902.0] * 4
    assert prompt_hidden[6].tolist() == [800.0] * 4
    assert prompt_hidden[7].tolist() == [801.0] * 4


def test_qwen35_talker_special_embeds_use_talker_text_embedding():
    model = _FakeTalkerModel()
    builder = _builder(model)

    bos, eos, pad = builder.get_tts_special_embeds()

    assert bos.tolist() == [[10.0] * 4]
    assert eos.tolist() == [[11.0] * 4]
    assert pad.tolist() == [[12.0] * 4]


def test_qwen35_talker_prefill_builds_speaker_codec_prefix():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        speaker_map={"f6009": 2},
        speaker_system_prompt_id={"f6009": [70, 71]},
    )

    prefix = builder._build_speaker_prefix({"voice_type": "Tina"})

    assert model.seen_speaker_ids == [2]
    assert prefix.shape == (11, 4)
    assert prefix[:3].tolist() == [[20.0] * 4, [30.0] * 4, [99.0] * 4]
    assert prefix[3:5].tolist() == [[70.0] * 4, [71.0] * 4]
    assert prefix[5].tolist() == [40.0] * 4
    assert prefix[6:8].tolist() == [[100.0] * 4, [101.0] * 4]
    assert prefix[8].tolist() == [45.0] * 4
    assert prefix[9:].tolist() == [[21.0] * 4, [99.0] * 4]


def test_qwen35_talker_prefill_builds_prompt_speaker_code_prefix():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        speaker_map={"f6009": 2},
        speaker_system_prompt_id={"f6009": [70, 71]},
    )

    prefix = builder._build_speaker_prefix(
        {
            "voice_type": "Tina",
            "prompt_speaker_codes": [[1, 2, 3], [4, 5, 6]],
            "system_instruct_ids": [80, 81],
        }
    )

    assert model.seen_speaker_ids == []
    assert model.seen_prompt_codes == [[[1, 2, 3], [4, 5, 6]]]
    assert prefix.shape == (11, 4)
    assert prefix[:3].tolist() == [[20.0] * 4, [30.0] * 4, [99.0] * 4]
    assert prefix[3:5].tolist() == [[80.0] * 4, [81.0] * 4]
    assert prefix[5].tolist() == [40.0] * 4
    assert prefix[6:8].tolist() == [[6.0] * 4, [15.0] * 4]
    assert prefix[8].tolist() == [45.0] * 4
    assert prefix[9:].tolist() == [[21.0] * 4, [99.0] * 4]


def test_qwen35_talker_prefill_uses_xvector_info_dict():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {
            "voice_type": "custom-vc",
            "xvector_info": {
                "prompt_code": [[1, 2, 3]],
                "talker_system_instruct": "Hi",
                "language_type": "en",
            },
        }
    )
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "en"
    assert model.seen_prompt_codes == [[[1, 2, 3]]]
    assert prefix is not None
    assert prefix[3:5].tolist() == [[72.0] * 4, [105.0] * 4]


def test_qwen35_talker_prefill_uses_voice_clone_language_alias():
    model = _FakeTalkerModel()
    builder = _builder(model)

    params = builder._normalize_voice_clone_params(
        {
            "xvector_info": {
                "prompt_code": [[1, 2, 3]],
                "language": "zh-CN",
            },
        }
    )

    assert params["language"] == "zh-CN"


def test_qwen35_talker_prefill_voice_clone_language_does_not_override_target():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        talker_language_id={"en": 11, "zh": 22},
    )

    params = builder._normalize_voice_clone_params(
        {
            "target_lang": "zh-CN",
            "xvector_info": {
                "prompt_code": [[1, 2, 3]],
                "language_type": "en",
            },
        }
    )

    assert params["target_lang"] == "zh-CN"
    assert "language" not in params
    assert builder._resolve_language_id(params) == 22


def test_qwen35_talker_prefill_accepts_ref_code_xvector_alias():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {
            "xvector_info": {
                "ref_code": torch.tensor([[3, 4, 5]]),
                "talker_system_instruct": "R",
            },
        }
    )
    prefix = builder._build_speaker_prefix(params)

    assert model.seen_prompt_codes == [[[3, 4, 5]]]
    assert prefix is not None
    assert prefix[3].tolist() == [82.0] * 4


def test_qwen35_talker_prefill_uses_xvector_info_path(tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_code": torch.tensor([[2, 3, 4]])}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "talker_system_instruct": "V",
                "language_type": "zh",
            }
        ),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {"xvector_info": str(voice_dir)}
    )
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "zh"
    assert model.seen_prompt_codes == [[[2, 3, 4]]]
    assert prefix is not None
    assert prefix[3].tolist() == [86.0] * 4


def test_qwen35_talker_prefill_voice_clone_cache_is_request_isolated(
    tmp_path,
):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_code": [[2, 3, 4]]}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "talker_system_instruct": "V",
                "language_type": "zh",
            }
        ),
        encoding="utf-8",
    )
    builder = _builder(_FakeTalkerModel())

    first = builder._normalize_voice_clone_params(
        {"xvector_info": str(voice_dir)}
    )
    first["prompt_speaker_codes"][0][0] = 99

    second = builder._normalize_voice_clone_params(
        {"xvector_info": str(voice_dir)}
    )

    assert second["prompt_speaker_codes"] == [[2, 3, 4]]


def test_qwen35_talker_prefill_loads_info_aliases_from_xvector_path(tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_code": torch.tensor([[2, 3, 4]])}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "system_instruct": "C",
                "language": "en",
            }
        ),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params({"xvector_info": str(voice_dir)})
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "en"
    assert model.seen_prompt_codes == [[[2, 3, 4]]]
    assert prefix is not None
    assert prefix[3].tolist() == [67.0] * 4


def test_qwen35_talker_prefill_loads_voice_clone_path_with_overrides(tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_code": torch.tensor([[2, 3, 4]])}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "talker_system_instruct": "A",
                "language_type": "zh",
            }
        ),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {
            "voice_clone_info": {
                "path": str(voice_dir),
                "system_instruct": "B",
                "language": "en",
            }
        }
    )
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "en"
    assert model.seen_prompt_codes == [[[2, 3, 4]]]
    assert prefix is not None
    assert prefix[3].tolist() == [66.0] * 4


def test_qwen35_talker_prefill_loads_ref_code_from_xvector_path(tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"ref_code": torch.tensor([[9, 8, 7]])}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "talker_system_instruct": "Z",
                "language_type": "en",
            }
        ),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {"xvector_info": str(voice_dir)}
    )
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "en"
    assert model.seen_prompt_codes == [[[9, 8, 7]]]
    assert prefix is not None
    assert prefix[3].tolist() == [90.0] * 4


def test_qwen35_talker_prefill_loads_numpy_prompt_code_from_xvector_path(tmp_path):
    np = pytest.importorskip("numpy")
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_code": np.array([[11, 12, 13]])}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "prompt_text": "reference text",
                "language_type": "en",
            }
        ),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(model)

    params = builder._normalize_voice_clone_params({"xvector_info": str(voice_dir)})
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "en"
    assert model.seen_prompt_codes == [[[11, 12, 13]]]
    assert prefix is not None


def test_qwen35_talker_prefill_loads_prompt_speaker_codes_from_xvector_path(
    tmp_path,
):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_speaker_codes": [[5, 6, 7]]}, handle)
    (voice_dir / "info.json").write_text(
        json.dumps(
            {
                "talker_system_instruct": "A",
                "language_type": "en",
            }
        ),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {"voice_clone_info": str(voice_dir)}
    )
    prefix = builder._build_speaker_prefix(params)

    assert params["language"] == "en"
    assert model.seen_prompt_codes == [[[5, 6, 7]]]
    assert prefix is not None
    assert prefix[3].tolist() == [65.0] * 4


def test_qwen35_talker_prefill_rejects_non_mapping_xvector_feat(tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump([1, 2, 3], handle)
    (voice_dir / "info.json").write_text(
        json.dumps({"talker_system_instruct": "A", "language_type": "en"}),
        encoding="utf-8",
    )
    builder = _builder(_FakeTalkerModel())

    with pytest.raises(ValueError, match="feat.pkl must contain a mapping"):
        builder._normalize_voice_clone_params({"xvector_info": str(voice_dir)})


def test_qwen35_talker_prefill_rejects_non_object_xvector_info(tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    with (voice_dir / "feat.pkl").open("wb") as handle:
        pickle.dump({"prompt_code": [[1, 2, 3]]}, handle)
    (voice_dir / "info.json").write_text(json.dumps(["en"]), encoding="utf-8")
    builder = _builder(_FakeTalkerModel())

    with pytest.raises(ValueError, match="info.json must contain a JSON object"):
        builder._normalize_voice_clone_params({"xvector_info": str(voice_dir)})


def test_qwen35_talker_prefill_accepts_prompt_codes_voice_clone_alias():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        nl_token_id=99,
        codec_eos_id=45,
        tokenizer=_FakeTokenizer(),
    )

    params = builder._normalize_voice_clone_params(
        {
            "voice_clone_info": {
                "prompt_codes": [[8, 9, 10]],
                "talker_system_instruct": "B",
            }
        }
    )
    prefix = builder._build_speaker_prefix(params)

    assert model.seen_prompt_codes == [[[8, 9, 10]]]
    assert prefix is not None
    assert prefix[3].tolist() == [66.0] * 4


def test_qwen35_talker_prefill_resolves_voice_alias_and_default():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        speaker_map={"f6009": 2, "f568-04": 3, "custom": 4},
    )

    assert builder._resolve_speaker_name_and_id({"voice_type": " tina "}) == (
        "f6009",
        2,
    )
    assert builder._resolve_speaker_name_and_id(
        {"voice_type": "Tina#prefix_caching"}
    ) == ("f6009", 2)
    assert builder._resolve_speaker_name_and_id(
        {"voice_type": "prefix_caching"}
    ) == ("f6009", 2)
    assert builder._resolve_speaker_name_and_id(
        {"voice_type": "f6009prefix_caching"}
    ) == ("f6009", 2)
    assert builder._resolve_speaker_name_and_id({"speaker": "custom"}) == (
        "custom",
        4,
    )
    assert builder._resolve_speaker_name_and_id({"voice_type": "default"}) == (
        "f6009",
        2,
    )


def test_qwen35_talker_prefill_resolves_example_voice_aliases():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        speaker_map={"f245": 1, "f05": 2, "m02": 3, "f030": 4},
    )

    assert builder._resolve_speaker_name_and_id({"voice_type": "Cherry"}) == (
        "f245",
        1,
    )
    assert builder._resolve_speaker_name_and_id({"voice_type": "芊悦"}) == (
        "f245",
        1,
    )
    assert builder._resolve_speaker_name_and_id({"voice_type": "苏瑶"}) == (
        "f05",
        2,
    )
    assert builder._resolve_speaker_name_and_id({"voice_type": "晨煦"}) == (
        "m02",
        3,
    )
    assert builder._resolve_speaker_name_and_id({"voice_type": "Chelsie"}) == (
        "f030",
        4,
    )


def test_qwen35_talker_prefill_uses_model_voice_map(tmp_path):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "voice_map.json").write_text(
        json.dumps({"Cherry": "custom"}),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        model_path=str(model_root),
        speaker_map={"custom": 9, "f245": 1},
    )

    assert builder._resolve_speaker_name_and_id({"voice_type": "Cherry"}) == (
        "custom",
        9,
    )
    assert builder._resolve_speaker_name_and_id({"voice_type": "default"}) == (
        "custom",
        9,
    )


def test_qwen35_talker_prefill_reads_parent_voice_map_for_split_talker(tmp_path):
    model_root = tmp_path / "model"
    talker_dir = model_root / "talker_lm"
    talker_dir.mkdir(parents=True)
    (model_root / "voice_map.json").write_text(
        json.dumps({"Studio": "studio_spk"}),
        encoding="utf-8",
    )
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        model_path=str(talker_dir),
        speaker_map={"studio_spk": 7},
    )

    assert builder._resolve_speaker_name_and_id({"voice_type": "studio"}) == (
        "studio_spk",
        7,
    )


def test_qwen35_talker_prefill_rejects_unknown_voice_when_map_is_known():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f6009": 2})

    with pytest.raises(ValueError, match="voice_type 'unknown'"):
        builder._resolve_speaker_name_and_id({"voice_type": "unknown"})


def test_qwen35_talker_prefill_allows_custom_voice_with_explicit_speaker_id():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f6009": 2})

    assert builder._resolve_speaker_name_and_id(
        {"voice_type": "custom", "speaker_id": 7}
    ) == ("custom", 7)


def test_qwen35_talker_prefill_allows_custom_voice_with_prompt_codes():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f6009": 2})

    prefix = builder._build_speaker_prefix(
        {
            "voice_type": "custom",
            "prompt_speaker_codes": [[1, 2, 3]],
        }
    )

    assert prefix is not None
    assert model.seen_prompt_codes == [[[1, 2, 3]]]


def test_qwen35_talker_prefill_ignores_none_voice():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f6009": 2})

    assert builder._build_speaker_prefix({"voice_type": "none"}) is None
    assert model.seen_speaker_ids == []


def test_qwen35_talker_prefill_prefix_caching_uses_default_speaker():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f6009": 2})

    prefix = builder._build_speaker_prefix({"voice_type": "prefix_caching"})

    assert prefix is not None
    assert model.seen_speaker_ids == [2]


def test_qwen35_talker_prefill_default_uses_sunny_when_tina_missing():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f568-04": 3})

    assert builder._resolve_speaker_name_and_id({}) == ("f568-04", 3)


def test_qwen35_talker_prefill_resolves_language_id():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        codec_think_id=46,
        talker_language_id={
            "zh": 77,
            "en-us": 78,
            "tagalog": 80,
            "yue": 81,
            "wuu": 82,
        },
    )

    assert builder._resolve_language_id({"language": "zh"}) == 77
    assert builder._resolve_language_id({"language": "zh-CN"}) == 77
    assert builder._resolve_language_id({"language": "EN_US"}) == 78
    assert builder._resolve_language_id({"language": "Chinese"}) == 77
    assert builder._resolve_language_id({"language": "English"}) == 78
    assert builder._resolve_language_id({"lang": "79"}) == 79
    assert builder._resolve_language_id({"language_type": "zh-CN"}) == 77
    assert builder._resolve_language_id({"language_type": "Cantonese"}) == 81
    assert builder._resolve_language_id({"target_lang": "zh-CN"}) == 77
    assert (
        builder._resolve_language_id({"translation_options": {"target_lang": "en"}})
        == 78
    )
    assert (
        builder._resolve_language_id(
            {"translation_options": {"target_lang": "Filipino"}}
        )
        == 80
    )
    assert (
        builder._resolve_language_id(
            {"translation_options": {"target_lang": "Chinese_wu"}}
        )
        == 82
    )
    assert (
        builder._resolve_language_id({"translation_options": {"target_lang": "79"}})
        == 79
    )
    assert builder._resolve_language_id({"language_id": "79"}) == 79
    assert builder._resolve_language_id({"language": "auto"}) is None


def test_qwen35_infers_livetranslate_language_from_message_prompt():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        talker_language_id={"tagalog": 80, "zh": 77},
    )
    request = OmniRequest(
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please translate the speech into Filipino.",
                        },
                        {"type": "input_audio", "input_audio": {"data": "..."}},
                    ],
                }
            ]
        },
        params={},
    )

    params = request_builders._params_with_request_language_fallback(request, {})

    assert params["target_language"] == "Tagalog"
    assert builder._resolve_language_id(params) == 80


def test_qwen35_livetranslate_prompt_language_does_not_override_explicit_language():
    request = OmniRequest(
        inputs="Please translate the speech into Chinese.",
        params={"target_lang": "English"},
    )

    params = request_builders._params_with_request_language_fallback(
        request,
        dict(request.params),
    )

    assert params == {"target_lang": "English"}


def test_qwen35_talker_prefill_resolves_assistant_instruct_ids():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        talker_assistant_prompt_id_mapping={"happy": [70, 71]},
    )

    assert builder._resolve_assistant_instruct_ids({"voice_style": "happy"}) == [
        70,
        71,
    ]
    assert builder._resolve_assistant_instruct_ids(
        {"assistant_instruct_ids": "80, 81"}
    ) == [80, 81]
    assert builder._resolve_assistant_instruct_ids({"instruction": "unknown"}) == []


def test_qwen35_tts_generate_mode_default_ignores_instructions(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_QWEN_TTS_GENERATE_MODE", raising=False)

    params = request_builders._apply_tts_generate_mode(
        {
            "tts_generate_mode": "default",
            "voice_type": "tina",
            "voice_style": "happy",
            "assistant_instruct_ids": [70, 71],
        }
    )

    assert params["voice_type"] == "tina"
    assert "voice_style" not in params
    assert "assistant_instruct_ids" not in params


def test_qwen35_tts_generate_mode_voice_design_forces_no_speaker(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_QWEN_TTS_GENERATE_MODE", raising=False)

    params = request_builders._apply_tts_generate_mode(
        {
            "tts_generate_mode": "voice_design",
            "voice_type": "tina",
            "voice_style": "happy",
        }
    )

    assert params["voice_type"] == "none"
    assert params["voice_style"] == "happy"


def test_qwen35_tts_generate_mode_env_instructions_keeps_instructions(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_QWEN_TTS_GENERATE_MODE", "instructions")

    params = request_builders._apply_tts_generate_mode(
        {
            "voice_type": "tina",
            "voice_style": "happy",
        }
    )

    assert params["voice_type"] == "tina"
    assert params["voice_style"] == "happy"


def test_qwen35_assistant_part_uses_language_codec_tokens():
    model = _FakeTalkerModel()
    assistant_embed = torch.arange(5, dtype=torch.float32).view(-1, 1).expand(-1, 4)
    tts_bos = torch.full((1, 4), 10.0)
    tts_eos = torch.full((1, 4), 11.0)
    tts_pad = torch.full((1, 4), 12.0)

    no_lang = request_builders._build_qwen35_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_bos,
        tts_eos_embed=tts_eos,
        tts_pad_embed=tts_pad,
        speaker_id=9,
        codec_nothink_id=41,
        codec_think_id=46,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
    )
    with_lang = request_builders._build_qwen35_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_bos,
        tts_eos_embed=tts_eos,
        tts_pad_embed=tts_pad,
        speaker_id=9,
        codec_nothink_id=41,
        codec_think_id=46,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
        language_id=77,
    )

    assert no_lang["input_embeds"][:, 0].tolist() == [
        0.0,
        1.0,
        2.0,
        41.0,
        42.0,
        43.0,
        10.0,
        40.0,
        3.0,
        4.0,
    ]
    assert no_lang["future_text_rows"][:, 0].tolist() == [11.0]
    assert with_lang["input_embeds"][:, 0].tolist() == [
        0.0,
        1.0,
        2.0,
        46.0,
        42.0,
        77.0,
        43.0,
        10.0,
        40.0,
        3.0,
        4.0,
    ]


def test_qwen35_assistant_part_inserts_instruction_text_rows():
    model = _FakeTalkerModel()
    assistant_embed = torch.zeros(5, 4)
    tts_special = torch.zeros(1, 4)
    instruct_embed = torch.tensor([[70.0] * 4, [71.0] * 4])

    result = request_builders._build_qwen35_assistant_part(
        assistant_embed=assistant_embed,
        assistant_instruct_embed=instruct_embed,
        text_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_special,
        tts_eos_embed=tts_special,
        tts_pad_embed=tts_special,
        speaker_id=9,
        codec_nothink_id=41,
        codec_think_id=46,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
    )

    assert result["input_embeds"].shape == (12, 4)
    assert result["input_embeds"][3:5, 0].tolist() == [70.0, 71.0]
    assert result["input_embeds"][5:10, 0].tolist() == [
        41.0,
        42.0,
        43.0,
        0.0,
        40.0,
    ]


def test_qwen35_assistant_part_omits_speaker_and_codec_pad_rows():
    model = _FakeTalkerModel()
    assistant_embed = torch.zeros(5, 4)
    tts_special = torch.zeros(1, 4)
    instruct_embed = torch.tensor([[70.0] * 4, [71.0] * 4])

    result = request_builders._build_qwen35_assistant_part(
        assistant_embed=assistant_embed,
        assistant_instruct_embed=instruct_embed,
        text_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_special,
        tts_eos_embed=tts_special,
        tts_pad_embed=tts_special,
        speaker_id=None,
        codec_nothink_id=41,
        codec_think_id=46,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
    )

    assert result["input_embeds"].shape == (12, 4)
    assert result["input_embeds"][5:10, 0].tolist() == [
        41.0,
        42.0,
        43.0,
        0.0,
        40.0,
    ]


def test_qwen35_assistant_part_prefills_four_text_rows_and_queues_rest():
    model = _FakeTalkerModel()
    assistant_embed = torch.arange(10, dtype=torch.float32).view(-1, 1).expand(-1, 4)
    tts_bos = torch.full((1, 4), 10.0)
    tts_eos = torch.full((1, 4), 11.0)
    tts_pad = torch.full((1, 4), 12.0)

    result = request_builders._build_qwen35_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_bos,
        tts_eos_embed=tts_eos,
        tts_pad_embed=tts_pad,
        speaker_id=9,
        codec_nothink_id=41,
        codec_think_id=46,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
    )

    assert result["input_embeds"][-4:, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert result["future_text_rows"][:, 0].tolist() == [7.0, 8.0, 9.0, 11.0]


def test_qwen35_prefill_strips_static_assistant_template_suffix():
    model = _FakeTalkerModel()
    builder = _builder(model)
    token_ids = torch.tensor(
        [
            20,
            32,
            198,
            248068,
            271,
            248069,
            271,
            101,
            102,
            103,
            104,
            105,
        ],
        dtype=torch.long,
    )
    thinker_embed = token_ids.to(dtype=torch.float32).view(-1, 1).expand(-1, 4)
    tts_bos = torch.full((1, 4), 10.0)
    tts_eos = torch.full((1, 4), 11.0)
    tts_pad = torch.full((1, 4), 12.0)

    result = builder._build_prefill_input(
        thinker_embed=thinker_embed,
        thinker_hidden=thinker_embed,
        thinker_input_ids=token_ids,
        multimodal_mask=torch.zeros(token_ids.numel(), dtype=torch.bool),
        assistant_token_count=5,
        text_projection=nn.Identity(),
        hidden_projection=nn.Identity(),
        codec_embed_fn=model.get_input_embeddings(),
        tts_bos_embed=tts_bos,
        tts_eos_embed=tts_eos,
        tts_pad_embed=tts_pad,
        assistant_instruct_embed=None,
        im_start_token_id=20,
        system_token_id=30,
        user_token_id=31,
        assistant_token_id=32,
        speaker_id=9,
        codec_nothink_id=41,
        codec_think_id=46,
        codec_think_bos_id=42,
        codec_think_eos_id=43,
        codec_pad_id=44,
        codec_bos_id=40,
        tts_pad_token_id=12,
        im_end_token_id=21,
    )

    assert result["input_embeds"][:, 0].tolist() == [
        20.0,
        32.0,
        198.0,
        41.0,
        42.0,
        43.0,
        10.0,
        40.0,
        101.0,
        102.0,
        103.0,
        104.0,
    ]
    assert result["future_text_rows"][:, 0].tolist() == [105.0, 11.0]


def test_qwen35_user_prefill_limits_multimodal_rows():
    model = _FakeTalkerModel()
    builder = _builder(model, max_thinker_to_talker_mm_tokens=2)
    token_ids = torch.tensor([20, 31, 99, 2, 2, 2, 55])
    thinker_embed = torch.arange(7, dtype=torch.float32).view(-1, 1).expand(-1, 4)
    thinker_hidden = (torch.arange(7, dtype=torch.float32) * 10).view(
        -1,
        1,
    ).expand(-1, 4)
    multimodal_mask = token_ids == 2

    rows, kept_ids = builder._build_user_part(
        thinker_embed=thinker_embed,
        thinker_hidden=thinker_hidden,
        thinker_input_ids=token_ids,
        multimodal_mask=multimodal_mask,
        text_projection=nn.Identity(),
        hidden_projection=nn.Identity(),
    )

    assert kept_ids.tolist() == [20, 31, 99, 2, 2, 55]
    assert rows[:, 0].tolist() == [0.0, 1.0, 2.0, 30.0, 50.0, 6.0]


def test_qwen35_prefill_drops_prompt_model_inputs_for_projected_talker():
    model = _FakeTalkerModel()
    builder = _builder(model, max_thinker_to_talker_mm_tokens=1)
    prompt_ids = torch.tensor([20, 31, 99, 2, 2, 20, 32, 99])
    prompt_embed = torch.ones(prompt_ids.numel(), 4)
    prompt_hidden = torch.ones(prompt_ids.numel(), 4)
    builder._reconstruct_prompt_states = lambda state: (
        prompt_ids,
        prompt_embed,
        prompt_hidden,
        {"image_grid_thw": torch.tensor([[1, 1, 2]])},
    )
    chunks = [
        SimpleNamespace(
            data=torch.ones(4),
            metadata={"token_id": token_id, "layer_hidden": torch.ones(4)},
        )
        for token_id in (60, 61, 62, 63)
    ]
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"voice_type": "tina"}),
        data={},
    )

    prefill = builder.build_prompt_prefill(payload, chunks, thinker_done=True)

    assert prefill["prompt_model_inputs"] == {}
    assert prefill["input_embeds"].shape[0] == prefill["input_ids"].shape[0]


def test_qwen35_prefill_uses_openai_audio_config_voice():
    model = _FakeTalkerModel()
    builder = _builder(model, speaker_map={"f245": 1})
    prompt_ids = torch.tensor([20, 31, 99, 20, 32, 99])
    prompt_embed = torch.ones(prompt_ids.numel(), 4)
    prompt_hidden = torch.ones(prompt_ids.numel(), 4)
    builder._reconstruct_prompt_states = lambda state: (
        prompt_ids,
        prompt_embed,
        prompt_hidden,
        {},
    )
    chunks = [
        SimpleNamespace(
            data=torch.ones(4),
            metadata={"token_id": token_id, "layer_hidden": torch.ones(4)},
        )
        for token_id in (60, 61, 62, 63)
    ]
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            metadata={"audio_config": {"voice": "Cherry"}},
        ),
        data={},
    )

    builder.build_prompt_prefill(payload, chunks, thinker_done=True)

    assert model.seen_speaker_ids == [1]


def test_qwen35_prefill_consumes_thinker_voice_style_tag():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        tokenizer=_FakeTokenizer(),
        talker_assistant_prompt_id_mapping={"happy": [70, 71]},
    )
    prompt_ids = torch.tensor([20, 31, 99, 20, 32, 99])
    prompt_embed = torch.ones(prompt_ids.numel(), 4)
    prompt_hidden = torch.ones(prompt_ids.numel(), 4)
    builder._reconstruct_prompt_states = lambda state: (
        prompt_ids,
        prompt_embed,
        prompt_hidden,
        {},
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"voice_type": "tina"}),
        data={},
    )

    builder.build_prompt_prefill(
        payload,
        _text_chunks("<voice_style>happy</voice_style>Hello"),
        thinker_done=True,
    )

    assert model.seen_text_ids[0] == [ord(char) for char in "Hello"]
    assert [70, 71] in model.seen_text_ids


def test_qwen35_prefill_explicit_voice_style_overrides_thinker_tag():
    model = _FakeTalkerModel()
    builder = _builder(
        model,
        tokenizer=_FakeTokenizer(),
        talker_assistant_prompt_id_mapping={
            "happy": [70],
            "sad": [72],
        },
    )
    prompt_ids = torch.tensor([20, 31, 99, 20, 32, 99])
    prompt_embed = torch.ones(prompt_ids.numel(), 4)
    prompt_hidden = torch.ones(prompt_ids.numel(), 4)
    builder._reconstruct_prompt_states = lambda state: (
        prompt_ids,
        prompt_embed,
        prompt_hidden,
        {},
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"voice_type": "tina", "voice_style": "sad"},
        ),
        data={},
    )

    builder.build_prompt_prefill(
        payload,
        _text_chunks("<voice_style>happy</voice_style>Hello"),
        thinker_done=True,
    )

    assert model.seen_text_ids[0] == [ord(char) for char in "Hello"]
    assert [72] in model.seen_text_ids
    assert [70] not in model.seen_text_ids


def test_qwen35_mrope_positions_include_image_end_position():
    positions, delta = request_builders._compute_qwen35_mrope_positions(
        torch.tensor([10, 11, 11, 11, 11, 13, 99]),
        {"image_grid_thw": torch.tensor([[1, 2, 2]])},
        _mrope_config(),
    )

    assert positions.shape == (3, 7)
    assert positions[:, 0].tolist() == [0, 0, 0]
    assert positions[:, 1:5].tolist() == [
        [1, 1, 1, 1],
        [1, 1, 2, 2],
        [1, 2, 1, 2],
    ]
    assert positions[:, 5].tolist() == [3, 3, 3]
    assert int(delta.item()) == int(positions.max().item() + 1 - 7)


def test_qwen35_mrope_positions_append_video_audio_span():
    positions, delta = request_builders._compute_qwen35_mrope_positions(
        torch.tensor([10, 12, 12, 12, 12, 13, 14, 14, 99]),
        {"video_grid_thw": torch.tensor([[1, 2, 2]])},
        _mrope_config(),
    )

    assert positions.shape == (3, 9)
    assert positions[:, 5].tolist() == [3, 3, 3]
    assert positions[:, 6:8].tolist() == [[1, 2], [1, 2], [1, 2]]
    assert positions[:, 8].tolist() == [3, 3, 3]
    assert int(delta.item()) == int(positions.max().item() + 1 - 9)


def test_qwen35_mrope_positions_delegate_audio_only_to_qwen3_builder(monkeypatch):
    captured = {}

    def fake_base_mrope(input_ids, model_inputs, thinker_config):
        captured["input_ids"] = input_ids
        captured["model_inputs"] = model_inputs
        captured["thinker_config"] = thinker_config
        return torch.ones(3, 4, dtype=torch.long), 5

    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "_compute_mrope_positions",
        fake_base_mrope,
    )
    input_ids = torch.tensor([99, 14, 14, 100])
    model_inputs = {"audio_feature_lengths": torch.tensor([2])}
    config = _mrope_config()

    positions, delta = request_builders._compute_qwen35_mrope_positions(
        input_ids,
        model_inputs,
        config,
    )

    assert captured["input_ids"] is input_ids
    assert captured["model_inputs"] is model_inputs
    assert captured["thinker_config"] is not config
    assert captured["thinker_config"].audio_token_id == config.audio_token_id
    assert captured["thinker_config"].audio_start_token_id == config.audio_token_id
    assert captured["thinker_config"].position_id_per_seconds == 1
    assert positions.tolist() == torch.ones(3, 4, dtype=torch.long).tolist()
    assert int(delta.item()) == 5


def test_qwen35_mrope_positions_skip_text_only_without_audio():
    assert (
        request_builders._compute_qwen35_mrope_positions(
            torch.tensor([99, 100]),
            {},
            _mrope_config(),
        )
        is None
    )


def test_qwen35_thinker_adapter_uses_next_mrope_builder(monkeypatch):
    monkeypatch.delenv("QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA", raising=False)
    monkeypatch.delenv("QWEN35_MAMBA_MEDIA_BRANCH_CACHE", raising=False)
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, result_adapter = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )

    req_data = request_builder(payload)

    assert callable(result_adapter)
    assert req_data.stage_payload is payload
    assert (
        captured["mrope_position_builder"]
        is request_builders._compute_qwen35_mrope_positions
    )
    assert captured["limit_prefix_cache_before_media"] is False
    assert captured["mamba_media_branching_cache"] is True


def test_qwen35_thinker_adapter_can_limit_prefix_cache_before_media(monkeypatch):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setenv("QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA", "1")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["limit_prefix_cache_before_media"] is True


def test_qwen35_thinker_adapter_keeps_rtc_media_prefix_cache_when_omitting_visual_payloads(
    monkeypatch,
):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.delenv("QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS", "1")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["limit_prefix_cache_before_media"] is False


def test_qwen35_thinker_adapter_can_force_prefix_limit_for_rtc_omit(
    monkeypatch,
):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setenv("QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA", "1")
    monkeypatch.setenv("SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS", "1")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["limit_prefix_cache_before_media"] is True


def test_qwen35_thinker_adapter_can_disable_mamba_media_branching_cache(monkeypatch):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setenv("QWEN35_MAMBA_MEDIA_BRANCH_CACHE", "off")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["mamba_media_branching_cache"] is False


def test_qwen35_thinker_adapter_disables_mamba_branching_for_rtc_omit_by_default(
    monkeypatch,
):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.delenv("QWEN35_MAMBA_MEDIA_BRANCH_CACHE", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS", "1")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["mamba_media_branching_cache"] is False


def test_qwen35_thinker_adapter_can_force_mamba_branching_for_rtc_omit(
    monkeypatch,
):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setenv("QWEN35_MAMBA_MEDIA_BRANCH_CACHE", "1")
    monkeypatch.setenv("SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS", "1")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["mamba_media_branching_cache"] is True


def test_qwen35_thinker_adapter_makes_rtc_prerun_prefill_only(monkeypatch):
    monkeypatch.delenv("QWEN35_RTC_PRERUN_PREFILL_ONLY", raising=False)
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"max_tokens": 2, "max_completion_tokens": 2, "max_new_tokens": 2},
            metadata={"pre_run": True},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["params"]["max_tokens"] == 0
    assert captured["params"]["max_completion_tokens"] == 0
    assert captured["params"]["max_new_tokens"] == 0


def test_qwen35_thinker_adapter_returns_lightweight_rtc_prerun_result(monkeypatch):
    monkeypatch.delenv("QWEN35_RTC_PRERUN_PREFILL_ONLY", raising=False)
    monkeypatch.setenv("QWEN35_RTC_PRERUN_THINKER_TERMINAL", "1")
    _, result_adapter = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    payload = StagePayload(
        request_id="req-prerun",
        request=OmniRequest(
            inputs={},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data={"large": torch.ones(4)},
    )
    data = SimpleNamespace(
        stage_payload=payload,
        input_ids=torch.tensor([1, 2, 3]),
        output_ids=[],
        finish_reason=None,
    )

    result = result_adapter(data)

    assert result.request_id == "req-prerun"
    assert result.request is payload.request
    assert result.data == {
        "text": "",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 0,
            "total_tokens": 3,
        },
    }


def test_qwen35_thinker_adapter_marks_rtc_prerun_isolated_before_req_init(
    monkeypatch,
):
    monkeypatch.delenv("QWEN35_RTC_ISOLATE_PRERUN_PREFILL", raising=False)
    monkeypatch.setenv("QWEN35_RTC_PROTECT_PRERUN_PREFIX_CACHE", "1")
    fake_req = SimpleNamespace(is_prefill_only=False)

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(req=fake_req, stage_payload=None)

    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"max_tokens": 0},
            metadata={"pre_run": True, "media_cache_namespace": "rtc:req-0"},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert fake_req._omni_isolate_prefill_batch is True
    assert fake_req._omni_rtc_cache_namespace == "rtc:req-0"
    assert fake_req._omni_protect_latest_prefix_cache is True


def test_qwen35_thinker_adapter_marks_rtc_actual_cache_release(monkeypatch):
    fake_req = SimpleNamespace(
        extra_key="media-cache:audio=rtc:req-0:audio",
        sampling_params=SimpleNamespace(max_new_tokens=64),
    )

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(req=fake_req, stage_payload=None)

    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"max_tokens": 64}, metadata={}),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert fake_req._omni_prioritize_prefill is True
    assert fake_req._omni_rtc_cache_namespace == "rtc:req-0"
    assert fake_req._omni_release_protected_prefix_cache_on_finish is True


def test_qwen35_thinker_adapter_can_keep_rtc_prerun_generation(monkeypatch):
    captured = {}

    def _fake_build_sglang_thinker_request(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(stage_payload=None)

    monkeypatch.setenv("QWEN35_RTC_PRERUN_PREFILL_ONLY", "0")
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "build_sglang_thinker_request",
        _fake_build_sglang_thinker_request,
    )

    request_builder, _ = request_builders.make_thinker_scheduler_adapters(
        tokenizer=object(),
        vocab_size=16,
        thinker_config=SimpleNamespace(),
    )
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1])})
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"max_tokens": 2},
            metadata={"pre_run": True},
        ),
        data=state.to_dict(),
    )

    request_builder(payload)

    assert captured["params"]["max_tokens"] == 2


def test_qwen35_thinker_request_sets_mamba_media_branching_hint(monkeypatch):
    monkeypatch.setattr(
        request_builders.qwen3_request_builders,
        "_mamba_branching_chunk_size",
        lambda: 64,
    )

    video_token_id = 12
    audio_token_id = 14
    input_ids = torch.tensor(
        [101] + [video_token_id] * 150 + [102] + [audio_token_id] * 8,
        dtype=torch.long,
    )
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.ones((150, 4)),
                "audio_embeds": torch.ones((8, 4)),
            },
            "media_cache_keys": {"video": "video:shared", "audio": "audio:q1"},
        },
    )

    req_data = request_builders.qwen3_request_builders.build_sglang_thinker_request(
        state,
        params={"max_tokens": 4},
        tokenizer=None,
        vocab_size=256,
        request_id="req-branch",
        thinker_config=SimpleNamespace(
            image_token_id=11,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
        ),
        mrope_position_builder=lambda *args: None,
        mamba_media_branching_cache=True,
    )

    assert req_data.req._omni_mamba_branching_seqlen == 128


def test_qwen35_mamba_branching_hint_patch_preserves_real_cache_hit():
    class FakeReq:
        def init_next_round_input(self, tree_cache=None):
            del tree_cache
            self.prefix_indices = [0] * 128
            self.mamba_branching_seqlen = None

    request_builders.qwen3_request_builders._install_mamba_branching_hint_patch(
        FakeReq
    )
    req = FakeReq()
    req._omni_mamba_branching_seqlen = 128

    req.init_next_round_input(None)

    assert req.mamba_branching_seqlen is None


def test_qwen35_rtc_actual_detection_falls_back_to_req_media_cache_key():
    request = SimpleNamespace(metadata={})
    req = SimpleNamespace(
        extra_key="media-cache:audio=rtc:sample:audio|video=rtc:sample:video",
        sampling_params=SimpleNamespace(max_new_tokens=64),
    )

    assert request_builders._is_qwen35_rtc_actual_req(request, req)


def test_qwen35_rtc_actual_detection_rejects_prefill_only_cache_key():
    request = SimpleNamespace(metadata={})
    req = SimpleNamespace(
        extra_key="media-cache:audio=rtc:sample:audio",
        sampling_params=SimpleNamespace(max_new_tokens=0),
    )

    assert not request_builders._is_qwen35_rtc_actual_req(request, req)


def test_qwen35_rtc_namespace_from_media_cache_key_strips_modality_suffix():
    req = SimpleNamespace(
        extra_key="media-cache:audio=rtc:sample:audio|video=rtc:sample:video"
    )

    assert request_builders._qwen35_rtc_cache_namespace(
        SimpleNamespace(metadata={}), req
    ) == "rtc:sample"


def test_qwen35_mamba_branching_hint_patch_caps_actual_prefix_match():
    class FakeReq:
        return_logprob = False
        logprob_start_len = -1

        def init_next_round_input(self, tree_cache=None):
            del tree_cache
            self.seen_return_logprob = self.return_logprob
            self.seen_logprob_start_len = self.logprob_start_len
            self.prefix_indices = [0] * int(self.logprob_start_len)
            self.mamba_branching_seqlen = None

    request_builders.qwen3_request_builders._install_mamba_branching_hint_patch(
        FakeReq
    )
    req = FakeReq()
    req._omni_mamba_branching_seqlen = 896
    req._omni_mamba_prefix_cache_limit = 896

    req.init_next_round_input(object())

    assert req.seen_return_logprob is True
    assert req.seen_logprob_start_len == 896
    assert req.return_logprob is False
    assert req.logprob_start_len == -1
    assert len(req.prefix_indices) == 896


def test_qwen35_mamba_branching_hint_patch_preserves_covered_radix_branch():
    class FakeReq:
        def init_next_round_input(self, tree_cache=None):
            del tree_cache
            self.prefix_indices = [0] * 34688
            self.mamba_branching_seqlen = 35584

    request_builders.qwen3_request_builders._install_mamba_branching_hint_patch(
        FakeReq
    )
    req = FakeReq()
    req._omni_mamba_branching_seqlen = 896

    req.init_next_round_input(None)

    assert req.mamba_branching_seqlen == 35584


def test_qwen35_mamba_branching_hint_patch_overrides_uncovered_deeper_branch():
    class FakeReq:
        def init_next_round_input(self, tree_cache=None):
            del tree_cache
            self.prefix_indices = [0] * 64
            self.mamba_branching_seqlen = 256

    request_builders.qwen3_request_builders._install_mamba_branching_hint_patch(
        FakeReq
    )
    req = FakeReq()
    req._omni_mamba_branching_seqlen = 128

    req.init_next_round_input(None)

    assert req.mamba_branching_seqlen == 128


def test_qwen35_mamba_branching_hint_patch_fills_missing_uncovered_branch():
    class FakeReq:
        def init_next_round_input(self, tree_cache=None):
            del tree_cache
            self.prefix_indices = []
            self.mamba_branching_seqlen = None

    request_builders.qwen3_request_builders._install_mamba_branching_hint_patch(
        FakeReq
    )
    req = FakeReq()
    req._omni_mamba_branching_seqlen = 128

    req.init_next_round_input(None)

    assert req.mamba_branching_seqlen == 128


def test_qwen35_mamba_branching_hint_patch_preserves_missing_covered_branch():
    class FakeReq:
        def init_next_round_input(self, tree_cache=None):
            del tree_cache
            self.prefix_indices = [0] * 128
            self.mamba_branching_seqlen = None

    request_builders.qwen3_request_builders._install_mamba_branching_hint_patch(
        FakeReq
    )
    req = FakeReq()
    req._omni_mamba_branching_seqlen = 128

    req.init_next_round_input(None)

    assert req.mamba_branching_seqlen is None


def test_qwen35_thinker_sampling_accepts_openai_max_tokens_alias():
    assert request_builders.qwen3_request_builders._resolve_max_new_tokens(
        {"max_tokens": 33}
    ) == 33
    assert request_builders.qwen3_request_builders._resolve_max_new_tokens(
        {"max_completion_tokens": 55}
    ) == 55
    assert request_builders.qwen3_request_builders._resolve_max_new_tokens(
        {
            "max_new_tokens": 44,
            "max_completion_tokens": 55,
            "max_tokens": 33,
        }
    ) == 44


def test_qwen35_thinker_sampling_passes_openai_penalties():
    state = Qwen3OmniPipelineState(prompt={"input_ids": torch.tensor([1, 2, 3])})

    req_data = request_builders.qwen3_request_builders.build_sglang_thinker_request(
        state,
        params={
            "max_tokens": 8,
            "presence_penalty": 0.7,
            "frequency_penalty": 0.2,
        },
        tokenizer=None,
        vocab_size=128,
        request_id="req-penalty",
    )

    sampling_params = req_data.req.sampling_params
    assert sampling_params.max_new_tokens == 8
    assert sampling_params.presence_penalty == 0.7
    assert sampling_params.frequency_penalty == 0.2


def test_qwen35_prepare_deepstack_inputs_scatter_visual_layers():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([11, 11, 12, 12, 99])},
        thinker_inputs={
            "model_inputs": {
                "image_embeds": torch.ones(2, 4),
                "video_embeds": torch.ones(2, 4),
                "image_deepstack_visual_embeds": [
                    torch.tensor([[1.0], [2.0]]),
                    torch.tensor([[10.0], [20.0]]),
                ],
                "video_deepstack_visual_embeds": [
                    torch.tensor([[3.0], [4.0]]),
                    torch.tensor([[30.0], [40.0]]),
                ],
            }
        },
    )

    request_builders._prepare_qwen35_deepstack_inputs(state, _mrope_config())

    model_inputs = state.thinker_inputs["model_inputs"]
    deepstack = model_inputs["deepstack_input_embeds"]
    assert deepstack["deepstack_input_embeds_0"].tolist() == [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [0.0],
    ]
    assert deepstack["deepstack_input_embeds_1"].tolist() == [
        [10.0],
        [20.0],
        [30.0],
        [40.0],
        [0.0],
    ]
    assert "image_deepstack_visual_embeds" not in model_inputs
    assert "video_deepstack_visual_embeds" not in model_inputs


def test_qwen35_prepare_deepstack_inputs_accepts_encoder_legacy_keys():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([11, 12, 99])},
        thinker_inputs={
            "model_inputs": {
                "image_embeds": torch.ones(1, 4),
                "video_embeds": torch.ones(1, 4),
                "deepstack_visual_embeds_image": [torch.tensor([[1.0]])],
                "deepstack_visual_embeds_video": [torch.tensor([[2.0]])],
            }
        },
    )

    request_builders._prepare_qwen35_deepstack_inputs(state, _mrope_config())

    model_inputs = state.thinker_inputs["model_inputs"]
    assert model_inputs["deepstack_input_embeds"][
        "deepstack_input_embeds_0"
    ].tolist() == [[1.0], [2.0], [0.0]]
    assert "deepstack_visual_embeds_image" not in model_inputs
    assert "deepstack_visual_embeds_video" not in model_inputs


def test_qwen35_prepare_thinker_inputs_trims_trailing_video_item_features():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([12, 12, 12, 99])},
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.arange(5, dtype=torch.float32).reshape(5, 1),
                "video_grid_thw": torch.tensor(
                    [[1, 1, 3], [1, 1, 2]], dtype=torch.long
                ),
                "video_token_counts": torch.tensor([3, 2], dtype=torch.long),
                "video_deepstack_visual_embeds": [
                    torch.arange(10, 15, dtype=torch.float32).reshape(5, 1)
                ],
                "use_audio_in_video": [False, True],
            }
        },
    )

    request_builders._prepare_qwen35_thinker_inputs(state, _mrope_config())

    model_inputs = state.thinker_inputs["model_inputs"]
    assert model_inputs["video_embeds"].tolist() == [[0.0], [1.0], [2.0]]
    assert model_inputs["video_grid_thw"].tolist() == [[1, 1, 3]]
    assert model_inputs["video_token_counts"].tolist() == [3]
    assert model_inputs["use_audio_in_video"] == [False]
    assert model_inputs["deepstack_input_embeds"][
        "deepstack_input_embeds_0"
    ].tolist() == [[10.0], [11.0], [12.0], [0.0]]


def test_qwen35_prepare_thinker_inputs_trims_using_video_grid_counts():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([12, 12, 12, 99])},
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.arange(5, dtype=torch.float32).reshape(5, 1),
                "video_grid_thw": torch.tensor(
                    [[1, 1, 3], [1, 1, 2]], dtype=torch.long
                ),
            }
        },
    )

    request_builders._prepare_qwen35_thinker_inputs(state, _mrope_config())

    model_inputs = state.thinker_inputs["model_inputs"]
    assert model_inputs["video_embeds"].tolist() == [[0.0], [1.0], [2.0]]
    assert model_inputs["video_grid_thw"].tolist() == [[1, 1, 3]]


def test_qwen35_prepare_thinker_inputs_rejects_partial_video_item_mismatch():
    state = Qwen3OmniPipelineState(
        prompt={"input_ids": torch.tensor([12, 12, 12, 12, 99])},
        thinker_inputs={
            "model_inputs": {
                "video_embeds": torch.ones(5, 1),
                "video_grid_thw": torch.tensor(
                    [[1, 1, 3], [1, 1, 2]], dtype=torch.long
                ),
                "video_token_counts": torch.tensor([3, 2], dtype=torch.long),
            }
        },
    )

    with pytest.raises(ValueError, match="video feature/token mismatch"):
        request_builders._prepare_qwen35_thinker_inputs(state, _mrope_config())


def test_qwen35_multimodal_feature_length_validation_passes():
    request_builders._validate_qwen35_multimodal_feature_lengths(
        torch.tensor([11, 11, 12, 14, 14]),
        {
            "image_embeds": torch.ones(2, 4),
            "video_embeds": torch.ones(1, 4),
            "audio_embeds": torch.ones(2, 4),
            "audio_feature_lengths": torch.tensor([2, 3]),
            "audio_is_dependent": torch.tensor([True, False]),
            "video_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "use_audio_in_video": [False, True],
        },
        _mrope_config(),
    )


def test_qwen35_multimodal_feature_length_validation_rejects_mismatch():
    try:
        request_builders._validate_qwen35_multimodal_feature_lengths(
            torch.tensor([11, 14, 14]),
            {
                "image_embeds": torch.ones(1, 4),
                "audio_embeds": torch.ones(1, 4),
            },
            _mrope_config(),
        )
    except ValueError as exc:
        assert "audio feature/token mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatched audio features to fail")


def test_qwen35_multimodal_metadata_validation_rejects_audio_mask_mismatch():
    try:
        request_builders._validate_qwen35_multimodal_feature_lengths(
            torch.tensor([14, 14]),
            {
                "audio_embeds": torch.ones(2, 4),
                "audio_feature_lengths": torch.tensor([2, 3]),
                "audio_is_dependent": torch.tensor([True]),
            },
            _mrope_config(),
        )
    except ValueError as exc:
        assert "audio_is_dependent length mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatched audio_is_dependent to fail")


def test_qwen35_multimodal_metadata_validation_rejects_video_flag_mismatch():
    try:
        request_builders._validate_qwen35_multimodal_feature_lengths(
            torch.tensor([12, 12]),
            {
                "video_embeds": torch.ones(2, 4),
                "video_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                "use_audio_in_video": [True, False, True],
            },
            _mrope_config(),
        )
    except ValueError as exc:
        assert "use_audio_in_video length mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatched video audio flags to fail")


def test_qwen35_stream_builder_selects_required_aux_hidden_key():
    embed = torch.tensor([[1.0, 1.0]])
    hidden_2 = torch.tensor([[2.0, 2.0]])
    hidden_18 = torch.tensor([[18.0, 18.0]])
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key="2,18"
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )
    req_output = SimpleNamespace(
        data=42,
        extra={"hidden_states": {"embed": embed, 2: hidden_2, 18: hidden_18}},
    )

    messages = builder("req-0", req_data, req_output)

    assert len(messages) == 1
    assert messages[0].target == "talker_ar"
    assert torch.equal(messages[0].data, embed[0])
    assert torch.equal(messages[0].metadata["layer_hidden"], hidden_18[0])


def test_qwen35_stream_builder_uses_stream_hidden_as_embed_fallback():
    embed = torch.tensor([[1.0, 1.0]])
    hidden_18 = torch.tensor([[18.0, 18.0]])
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )
    req_output = SimpleNamespace(
        data=42,
        extra={
            "hidden_states": {18: hidden_18},
            "stream_hidden_states": embed,
        },
    )

    messages = builder("req-0", req_data, req_output)

    assert len(messages) == 1
    assert messages[0].target == "talker_ar"
    assert torch.equal(messages[0].data, embed[0])


def test_qwen35_stream_builder_uses_last_hidden_row_for_prefill_suffix():
    embed = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    hidden_18 = torch.tensor([[18.0, 18.0], [19.0, 19.0]])
    stream_hidden = torch.tensor([[101.0, 101.0], [102.0, 102.0]])
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )

    messages = builder(
        "req-0",
        req_data,
        SimpleNamespace(
            data=42,
            extra={
                "hidden_states": {"embed": embed, 18: hidden_18},
                "stream_hidden_states": stream_hidden,
            },
        ),
    )

    assert len(messages) == 1
    assert messages[0].target == "talker_ar"
    assert torch.equal(messages[0].data, embed[-1])
    assert torch.equal(messages[0].metadata["layer_hidden"], hidden_18[-1])


def test_qwen35_stream_builder_inlines_decode_token_when_enabled(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    embed = torch.tensor([[1.0, 1.0]])
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"stream": True}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )
    req_output = SimpleNamespace(
        data=42,
        extra={"stream_hidden_states": embed},
    )

    messages = builder("req-0", req_data, req_output)

    assert [msg.target for msg in messages] == ["decode", "talker_ar"]
    assert messages[0].data == 42
    assert torch.equal(messages[1].data, embed[0])


def test_qwen35_stream_builder_skips_non_text_token_for_talker(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    embed = torch.tensor([[1.0, 1.0]])
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18,
        vocab_size=128,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"stream": True, "modalities": ["text", "audio"]},
        ),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )
    req_output = SimpleNamespace(
        data=1 << 40,
        extra={"stream_hidden_states": embed},
    )

    messages = builder("req-0", req_data, req_output)

    assert [msg.target for msg in messages] == ["decode"]


def test_qwen35_stream_builder_batches_decode_tokens(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    monkeypatch.setenv("SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE", "3")
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"stream": True}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )

    per_token_messages = [
        builder(
            "req-0",
            req_data,
            SimpleNamespace(data=token_id, extra={}),
        )
        for token_id in [42, 43, 44, 45, 46]
    ]
    decode_chunks = [
        [msg for msg in messages if msg.target == "decode"]
        for messages in per_token_messages
    ]

    assert decode_chunks[0][0].data == 42
    assert decode_chunks[1] == []
    assert decode_chunks[2] == []
    assert decode_chunks[3][0].data == [43, 44, 45]
    assert decode_chunks[3][0].metadata["token_count"] == 3
    assert decode_chunks[4] == []


def test_qwen35_stream_builder_batches_rtc_audio_decode_tokens_by_default(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    monkeypatch.delenv("SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("QWEN35_RTC_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"stream": True, "modalities": ["text", "audio"]},
            metadata={"media_cache_namespace": "rtc:req-0"},
        ),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )

    per_token_messages = [
        builder(
            "req-0",
            req_data,
            SimpleNamespace(data=token_id, extra={}),
        )
        for token_id in range(1, 10)
    ]
    decode_chunks = [
        [msg for msg in messages if msg.target == "decode"]
        for messages in per_token_messages
    ]

    assert decode_chunks[0][0].data == 1
    assert all(chunk == [] for chunk in decode_chunks[1:8])
    assert decode_chunks[8][0].data == list(range(2, 10))
    assert decode_chunks[8][0].metadata["token_count"] == 8


def test_qwen35_stream_builder_batches_rtc_decode_tokens_from_req_extra_key(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    monkeypatch.delenv("SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("QWEN35_RTC_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"stream": True, "modalities": ["text", "audio"]},
        ),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0, extra_key="cache=rtc:req-0:video"),
        stage_payload=payload,
    )

    per_token_messages = [
        builder(
            "req-0",
            req_data,
            SimpleNamespace(data=token_id, extra={}),
        )
        for token_id in range(1, 10)
    ]
    decode_chunks = [
        [msg for msg in messages if msg.target == "decode"]
        for messages in per_token_messages
    ]

    assert decode_chunks[0][0].data == 1
    assert all(chunk == [] for chunk in decode_chunks[1:8])
    assert decode_chunks[8][0].data == list(range(2, 10))
    assert decode_chunks[8][0].metadata["token_count"] == 8


def test_qwen35_stream_builder_batches_rtc_text_only_decode_tokens(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    monkeypatch.delenv("SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("QWEN35_RTC_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"stream": True, "modalities": ["text"]},
            metadata={"media_cache_namespace": "rtc:req-0"},
        ),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )

    per_token_messages = [
        builder(
            "req-0",
            req_data,
            SimpleNamespace(data=token_id, extra={}),
        )
        for token_id in range(1, 10)
    ]
    decode_chunks = [
        [msg for msg in messages if msg.target == "decode"]
        for messages in per_token_messages
    ]

    assert decode_chunks[0][0].data == 1
    assert all(chunk == [] for chunk in decode_chunks[1:8])
    assert decode_chunks[8][0].data == list(range(2, 10))


def test_qwen35_stream_builder_keeps_non_rtc_decode_tokens_unbatched(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES", "4096")
    monkeypatch.delenv("SGLANG_OMNI_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("QWEN35_RTC_DECODE_STREAM_TOKEN_BATCH_SIZE", raising=False)
    builder = request_builders.make_thinker_stream_output_builder(
        required_aux_hidden_key=18
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"stream": True, "modalities": ["text", "audio"]},
        ),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(is_chunked=0),
        stage_payload=payload,
    )

    first = builder("req-0", req_data, SimpleNamespace(data=1, extra={}))
    second = builder("req-0", req_data, SimpleNamespace(data=2, extra={}))

    assert [msg.data for msg in first if msg.target == "decode"] == [1]
    assert [msg.data for msg in second if msg.target == "decode"] == [2]


def test_qwen35_talker_adapter_prioritizes_rtc_actual_prefill(monkeypatch):
    fake_req = SimpleNamespace()

    def _fake_build_talker_request_data(payload, **kwargs):
        del kwargs
        return SimpleNamespace(req=fake_req, stage_payload=payload)

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace()

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"stream": True, "modalities": ["text", "audio"]},
            metadata={"media_cache_namespace": "rtc:req-0"},
        ),
        data={},
    )

    request_builder(payload)

    assert fake_req._omni_prioritize_prefill is True


def test_qwen35_talker_adapter_uses_text_chunk_size_env(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        del payload, kwargs
        req_data = SimpleNamespace(req=SimpleNamespace(), stage_payload=None)
        captured["req_data"] = req_data
        return req_data

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    monkeypatch.setenv("QWEN35_TALKER_TEXT_CHUNK_SIZE", "1")
    monkeypatch.setenv("QWEN35_TALKER_TEXT_FEEDBACK_STRIDE", "1")
    model = _FakeTalkerModel()
    model.config = SimpleNamespace()

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )

    request_builder(
        StagePayload(
            request_id="req-0",
            request=OmniRequest(
                inputs={},
                params={"stream": True, "modalities": ["text", "audio"]},
            ),
            data={},
        )
    )

    req_data = captured["req_data"]
    assert req_data.talker_text_chunk_size == 1
    assert req_data.talker_text_feedback_stride == 1
    assert req_data.talker_text_feedback_countdown == 1


def test_qwen35_talker_sampling_uses_passed_codec_eos_id(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        del payload
        captured["mrope_position_builder"] = kwargs["mrope_position_builder"]
        captured.update(kwargs["resolve_sampling_config"]({}))
        return captured

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace()

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={}),
        data={},
    )
    sampling_config = request_builder(payload)

    assert sampling_config["codec_eos_id"] == 7
    assert sampling_config["max_new_tokens"] == 2048
    assert sampling_config["suppress_tokens"] == []
    assert (
        sampling_config["mrope_position_builder"]
        is request_builders._compute_qwen35_mrope_positions
    )


def test_qwen35_talker_max_tokens_scales_from_text_limit(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"max_tokens": 50}),
        data={},
    )

    request_builder(payload)

    assert captured["max_new_tokens"] == 2000


def test_qwen35_talker_max_tokens_scales_from_completion_limit(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"max_completion_tokens": 50}),
        data={},
    )

    request_builder(payload)

    assert captured["max_new_tokens"] == 2000


def test_qwen35_talker_default_cap_wins_over_large_text_limit(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"max_tokens": 2048}),
        data={},
    )

    request_builder(payload)

    assert captured["max_new_tokens"] == 2048


def test_qwen35_talker_and_subtalker_require_explicit_stage_seed(
    monkeypatch,
):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params={"seed": 0}),
        data={},
    )

    req_data = request_builder(payload)
    sub_sp = req_data.req._qwen35_subtalker_sampling_params

    assert captured["seed"] is None
    assert sub_sp.temperature == 0.1
    assert sub_sp.top_k == 5
    assert sub_sp.top_p == 1.0
    assert sub_sp.repetition_penalty == 1.05
    assert sub_sp.sampling_seed is None


def test_qwen35_talker_max_tokens_uses_talker_cap_with_text_limit(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={"max_tokens": 50, "talker_max_tokens": 1200},
        ),
        data={},
    )

    request_builder(payload)

    assert captured["max_new_tokens"] == 1200


def test_qwen35_talker_adapter_accepts_nested_sampling_params(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        captured["mrope_position_builder"] = kwargs["mrope_position_builder"]
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={
                "talker_params": SimpleNamespace(
                    max_completion_tokens=12,
                    temperature=0.2,
                    top_k=3,
                    top_p=0.7,
                    min_p=0.04,
                    repetition_penalty=1.2,
                    seed=123,
                ),
                "subtalker_params": SimpleNamespace(
                    temperature=0.1,
                    top_k=5,
                    top_p=0.95,
                    min_p=0.03,
                    seed=456,
                ),
            },
        ),
        data={},
    )

    req_data = request_builder(payload)
    sub_sp = req_data.req._qwen35_subtalker_sampling_params

    assert captured["max_new_tokens"] == 12
    assert captured["temperature"] == 0.2
    assert captured["top_k"] == 3
    assert captured["top_p"] == 0.7
    assert captured["min_p"] == 0.04
    assert captured["repetition_penalty"] == 1.2
    assert captured["seed"] == 123
    assert sub_sp.temperature == 0.1
    assert sub_sp.top_k == 5
    assert sub_sp.top_p == 0.95
    assert sub_sp.min_p == 0.03
    assert sub_sp.sampling_seed == 456
    assert (
        captured["mrope_position_builder"]
        is request_builders._compute_qwen35_mrope_positions
    )


def test_qwen35_talker_adapter_accepts_prefixed_sampling_params(monkeypatch):
    captured = {}

    def _fake_build_talker_request_data(payload, **kwargs):
        captured.update(kwargs["resolve_sampling_config"](payload.request.params))
        return SimpleNamespace(req=SimpleNamespace())

    monkeypatch.setattr(
        request_builders,
        "_build_talker_request_data",
        _fake_build_talker_request_data,
    )
    model = _FakeTalkerModel()
    model.config = SimpleNamespace(codec_eos_token_id=7)

    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=object(),
        codec_vocab_size=16,
        model=model,
        model_path=str(Path("/tmp")),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=0,
        codec_eos_id=7,
    )
    payload = StagePayload(
        request_id="req-0",
        request=OmniRequest(
            inputs={},
            params={
                "max_tokens": 2,
                "talker_max_tokens": 12,
                "talker_temperature": 0.25,
                "talker_top_k": 4,
                "talker_top_p": 0.75,
                "talker_min_p": 0.05,
                "talker_repetition_penalty": 1.3,
                "talker_seed": 123,
                "subtalker_temperature": 0.15,
                "subtalker_top_k": 6,
                "subtalker_top_p": 0.85,
                "subtalker_min_p": 0.02,
                "subtalker_seed": 456,
            },
        ),
        data={},
    )

    req_data = request_builder(payload)
    sub_sp = req_data.req._qwen35_subtalker_sampling_params

    # A prefixed talker_max_tokens directly caps codec tokens; when it is
    # omitted, the adapter falls back to text max_tokens * 40.
    assert captured["max_new_tokens"] == 12
    assert captured["temperature"] == 0.25
    assert captured["top_k"] == 4
    assert captured["top_p"] == 0.75
    assert captured["min_p"] == 0.05
    assert captured["repetition_penalty"] == 1.3
    assert captured["seed"] == 123
    assert sub_sp.temperature == 0.15
    assert sub_sp.top_k == 6
    assert sub_sp.top_p == 0.85
    assert sub_sp.min_p == 0.02
    assert sub_sp.sampling_seed == 456
