# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import inspect
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen35_omni.components.talker_prefill import (
    Qwen35TalkerPrefillBuilder,
)
from sglang_omni.models.qwen35_omni.components.talker import (
    Qwen35CodePredictorAttention,
    Qwen35CodePredictorLayer,
    Qwen35OmniNextTalker,
    Qwen35OmniNextTalkerCodePredictor,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import (
    QwenTalkerModelRunner,
)
from sglang_omni.models.qwen35_omni.config import (
    Qwen35OmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen35_omni.hf_config import Qwen35OmniNextConfig
from sglang_omni.models.qwen35_omni.request_builders import (
    make_thinker_stream_output_builder,
)
from sglang_omni.models.qwen35_omni.stages import (
    create_talker_ar_executor_from_config,
)
from sglang_omni.models.qwen35_omni.talker_scheduler import (
    Qwen35TalkerScheduler,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.vendor.sglang.layers import GemmaRMSNorm


class _Embedding:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.weight = torch.empty((1, 2), dtype=torch.float32)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        values = token_ids.float().reshape(-1, 1) + self.offset
        return torch.cat([values, values + 0.25], dim=-1)


class _FakeTalker:
    activation_dtype = torch.float32

    def __init__(self) -> None:
        self.model = SimpleNamespace(codec_embedding=_Embedding(100.0))
        self.code_predictor = SimpleNamespace(
            model=SimpleNamespace(
                codec_embedding=[_Embedding(200.0 + i) for i in range(15)]
            )
        )
        self.speaker_codec_embeddings = torch.arange(32).reshape(1, 16, 2)

    def get_input_embeddings(self):
        return self.model.codec_embedding

    @staticmethod
    def get_text_embeddings(token_ids: torch.Tensor) -> torch.Tensor:
        values = token_ids.float().reshape(-1, 1)
        return torch.cat([values, values + 0.5], dim=-1)


def _root_config() -> SimpleNamespace:
    talker = SimpleNamespace(
        speaker_id={"m02": 0},
        speaker_system_prompt_id={"m02": [10, 11]},
        codec_bos_id=4197,
        codec_eos_token_id=4198,
        codec_think_id=4202,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
    )
    return SimpleNamespace(
        talker_config=talker,
        im_start_token_id=1,
        im_end_token_id=2,
        system_token_id=3,
        assistant_token_id=4,
        nl_token_id=5,
        tts_bos_token_id=6,
        tts_eos_token_id=7,
        tts_pad_token_id=8,
        talker_language_id={"english": 2050},
        talker_assistant_prompt_id_mapping={"whispering": [12, 13]},
    )


def _chunk(token_id: int) -> SimpleNamespace:
    return SimpleNamespace(metadata={"token_id": token_id}, data=torch.tensor([0]))


def test_qwen35_nested_talker_configs_are_typed() -> None:
    config = Qwen35OmniNextConfig(
        talker_config={
            "text_config": {
                "hidden_size": 1280,
                "vocab_size": 5120,
                "text_vocab_size": 248320,
                "num_hidden_layers": 20,
                "layer_types": ["linear_attention"] * 20,
            },
            "code_predictor_config": {
                "hidden_size": 1024,
                "talker_hidden_size": 1280,
                "vocab_size": 2048,
            },
        }
    )

    assert config.talker_config.text_config.hidden_size == 1280
    assert config.talker_config.text_config.text_vocab_size == 248320
    assert config.talker_config.code_predictor_config.talker_hidden_size == 1280


def test_qwen35_code_predictor_uses_offset_rms_norm(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.qwen35_omni.components.talker.get_rope",
        lambda *args, **kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen35_omni.components.talker.Qwen3OmniMoeTalkerDenseMLP",
        lambda *args, **kwargs: nn.Identity(),
    )
    predictor_config = SimpleNamespace(
        hidden_size=8,
        talker_hidden_size=8,
        intermediate_size=16,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.5,
        max_position_embeddings=32,
        rope_theta=10_000,
        rope_scaling=None,
    )
    talker_config = SimpleNamespace(
        code_predictor_config=predictor_config,
        num_code_groups=4,
    )

    attention = Qwen35CodePredictorAttention(
        predictor_config,
        layer_id=0,
        quant_config=None,
        prefix="attention",
    )
    layer = Qwen35CodePredictorLayer(
        predictor_config,
        layer_id=0,
        quant_config=None,
        prefix="layer",
    )
    predictor = Qwen35OmniNextTalkerCodePredictor(talker_config)

    norms = (
        attention.q_norm,
        attention.k_norm,
        layer.input_layernorm,
        layer.post_attention_layernorm,
        predictor.model.norm,
    )
    assert all(isinstance(norm, GemmaRMSNorm) for norm in norms)

    value = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    expected = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    attention.q_norm.weight.data.zero_()
    assert torch.allclose(attention.q_norm.forward_native(value), expected)


def test_qwen35_speech_topology_keeps_thinker_and_talker_separate() -> None:
    config = Qwen35OmniSpeechPipelineConfig(model_path="/tmp/model")
    stages = {stage.name: stage for stage in config.stages}

    assert stages["thinker"].gpu == 0
    assert stages["talker_ar"].gpu == 1
    assert stages["code2wav"].gpu == 1
    assert stages["thinker"].stream_to == ["talker_ar", "decode"]
    assert stages["talker_ar"].factory_args["enable_partial_start"] is True
    assert stages["talker_ar"].factory_args["partial_start_min_chunks"] == 4


def test_qwen35_talker_partial_start_defaults_to_enabled() -> None:
    signature = inspect.signature(create_talker_ar_executor_from_config)
    assert signature.parameters["enable_partial_start"].default is True


def test_qwen35_talker_override_exposes_qwen3_next_as_hf_config() -> None:
    root_config = Qwen35OmniNextConfig(
        talker_config={
            "text_config": {
                "hidden_size": 1280,
                "vocab_size": 5120,
                "num_hidden_layers": 20,
                "layer_types": ["linear_attention"] * 20,
            }
        }
    )
    model_config = SimpleNamespace(hf_config=root_config)

    ModelWorker._apply_arch_override(model_config, "Qwen35OmniNextTalker")

    assert model_config.hf_config is root_config.talker_config.text_config
    assert model_config.hf_text_config is model_config.hf_config
    assert model_config.hf_config.talker_config is root_config.talker_config
    assert model_config._omni_root_hf_config is root_config


def test_qwen35_prefill_uses_voice_map_and_four_token_trigger(tmp_path) -> None:
    (tmp_path / "voice_map.json").write_text(json.dumps({"Ethan": "m02"}))
    builder = Qwen35TalkerPrefillBuilder(
        model=_FakeTalker(),
        root_config=_root_config(),
        model_path=str(tmp_path),
    )
    payload = SimpleNamespace(
        request=SimpleNamespace(
            params={},
            metadata={
                "audio_config": {
                    "voice": "Ethan",
                    "language": "english",
                    "style": "whispering",
                }
            },
        )
    )
    result = builder.build_prompt_prefill(
        payload,
        [_chunk(token_id) for token_id in (20, 21, 22, 23, 24)],
        thinker_done=True,
    )

    assert result["voice"] == "m02"
    assert result["language"] == "english"
    assert result["input_embeds"].shape == (28, 2)
    assert result["input_ids"].tolist() == [8] * 28
    assert len(result["pending_text_queue"]) == 0
    assert result["feedback_only_decode"] is True
    assert result["interleaved_streaming"] is False


def test_qwen35_partial_prefill_starts_with_four_text_tokens(tmp_path) -> None:
    (tmp_path / "voice_map.json").write_text(json.dumps({"Ethan": "m02"}))
    builder = Qwen35TalkerPrefillBuilder(
        model=_FakeTalker(),
        root_config=_root_config(),
        model_path=str(tmp_path),
    )
    payload = SimpleNamespace(
        request=SimpleNamespace(params={}, metadata={"audio_config": {}})
    )

    result = builder.build_prompt_prefill(
        payload,
        [_chunk(token_id) for token_id in (20, 21, 22, 23, 24)],
        thinker_done=False,
    )

    assert result["input_embeds"].shape == (23, 2)
    assert len(result["pending_text_queue"]) == 1
    assert result["feedback_only_decode"] is False
    assert result["interleaved_streaming"] is True


def test_qwen35_thinker_streams_token_to_talker_without_hidden_states() -> None:
    request = OmniRequest(inputs={}, params={"stream": False}, metadata={})
    payload = StagePayload(
        request_id="req-1",
        request=request,
        data={},
    )
    req_data = SimpleNamespace(req=SimpleNamespace(is_chunked=0), stage_payload=payload)
    output = SimpleNamespace(data=123)

    messages = make_thinker_stream_output_builder()("req-1", req_data, output)

    assert len(messages) == 1
    assert messages[0].target == "talker_ar"
    assert messages[0].metadata == {"token_id": 123}


def test_qwen35_sampler_updates_inference_logits_inside_inference_mode(
    monkeypatch,
) -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    with torch.inference_mode():
        logits = torch.ones(1)

    def _sample(self, logits_output, *args):
        assert torch.is_inference_mode_enabled()
        logits_output.div_(2)
        return logits_output

    monkeypatch.setattr(ModelRunner, "_sample_next_token_ids", _sample)
    result = runner._sample_next_token_ids(logits, None, None, [])

    assert result.item() == 0.5


def test_qwen35_residual_predictor_uses_request_sampler_and_frame_position() -> None:
    talker = Qwen35OmniNextTalker.__new__(Qwen35OmniNextTalker)
    nn.Module.__init__(talker)
    talker.config = SimpleNamespace(
        num_code_groups=16,
        code_predictor_config=SimpleNamespace(vocab_size=2048),
    )
    talker._subtalker_frame_positions = torch.tensor([5])
    talker._subtalker_sample_index = 0
    captured = {}

    def _sampling_info(batch_size: int, *, vocab_size: int | None = None):
        captured["batch_size"] = batch_size
        captured["vocab_size"] = vocab_size
        return SimpleNamespace()

    def _sampler(_logits, _info, _return_logprob, _top_logprobs, _ids, positions):
        captured["positions"] = positions.clone()
        return torch.tensor([[1159]])

    talker._build_static_sampling_info = _sampling_info
    talker._sampler = _sampler

    sampled = talker._sample_code_predictor_token(torch.zeros(1, 1, 2048))

    assert sampled.tolist() == [[1159]]
    assert captured["batch_size"] == 1
    assert captured["vocab_size"] == 2048
    assert torch.equal(captured["positions"], torch.tensor([75]))


def test_qwen35_residual_frame_position_ignores_interleaved_placeholders() -> None:
    talker = Qwen35OmniNextTalker.__new__(Qwen35OmniNextTalker)
    nn.Module.__init__(talker)
    talker._subtalker_frame_positions = torch.zeros(1, dtype=torch.long)
    talker.prepare_decode_buffers = Qwen35OmniNextTalker.prepare_decode_buffers.__get__(
        talker
    )
    base_prepare = Qwen35OmniNextTalker.__mro__[1].prepare_decode_buffers
    request = SimpleNamespace(
        data=SimpleNamespace(
            codec_generation_steps=4,
            req=SimpleNamespace(output_ids=[1, 2, 3, 4, 8, 8, 8]),
        )
    )

    original = base_prepare
    try:
        Qwen35OmniNextTalker.__mro__[1].prepare_decode_buffers = lambda self, reqs: None
        talker.prepare_decode_buffers([request])
    finally:
        Qwen35OmniNextTalker.__mro__[1].prepare_decode_buffers = original

    assert talker._subtalker_frame_positions.tolist() == [4]


def test_qwen35_interleave_waits_at_codec_boundary() -> None:
    data = SimpleNamespace(
        interleaved_codec_chunk_size=4,
        interleaved_codec_steps=4,
        interleaved_final=False,
        interleaved_boundary_ready=False,
        interleaved_text_chunk_size=4,
        pending_text_queue=deque([torch.ones(2)] * 3),
        pending_feedback_queue=deque([torch.zeros(2)]),
        thinker_chunks_done=False,
    )

    assert Qwen35TalkerScheduler._awaiting_boundary_step(data)
    assert not Qwen35TalkerScheduler._needs_text_extend(data)
    assert QwenTalkerModelRunner._data_has_next_decode_input(data)

    data.interleaved_boundary_ready = True
    assert Qwen35TalkerScheduler._needs_text_extend(data)
    assert not Qwen35TalkerScheduler._text_chunk_ready(data)
    assert not QwenTalkerModelRunner._data_has_next_decode_input(data)

    data.pending_text_queue.append(torch.ones(2))
    assert Qwen35TalkerScheduler._text_chunk_ready(data)


def test_qwen35_interleave_decode_does_not_consume_future_text() -> None:
    feedback = torch.tensor([1.0, 2.0])
    future_text = deque([torch.full((2,), float(index)) for index in range(4)])
    data = SimpleNamespace(
        feedback_only_decode=False,
        interleaved_text_chunk_size=4,
        pending_feedback_queue=deque([feedback]),
        pending_text_queue=future_text,
    )
    request = SimpleNamespace(data=data)

    combined = QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=request,
        device=feedback.device,
        dtype=feedback.dtype,
    )

    assert torch.equal(combined, feedback)
    assert len(data.pending_feedback_queue) == 0
    assert len(data.pending_text_queue) == 4


def test_qwen35_boundary_decode_is_not_emitted_or_counted() -> None:
    emitted = []
    model = SimpleNamespace(
        _output_codes=torch.tensor([[4198, 2, 3]]),
        _output_embeds=torch.tensor([[1.0, 2.0]]),
        config=SimpleNamespace(codec_eos_token_id=4198),
    )
    runner = QwenTalkerModelRunner.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._outbox = SimpleNamespace(put=emitted.append)
    runner._code2wav_target = "code2wav"
    data = SimpleNamespace(
        stage_payload=None,
        pending_feedback_queue=deque(),
        codec_generation_steps=4,
        interleaved_codec_chunk_size=4,
        interleaved_codec_steps=4,
        interleaved_drop_next_output=True,
        interleaved_boundary_ready=False,
    )
    schedule_batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r0")],
        output_ids=torch.tensor([4198]),
    )

    runner._emit_code_chunks_and_feedback(
        schedule_batch=schedule_batch,
        requests=[SimpleNamespace(data=data)],
    )

    assert emitted == []
    assert len(data.pending_feedback_queue) == 1
    assert data.codec_generation_steps == 4
    assert data.interleaved_codec_steps == 4
    assert not data.interleaved_drop_next_output
    assert data.interleaved_boundary_ready
    assert schedule_batch.output_ids.tolist() == [0]


def test_qwen35_boundary_decode_does_not_consume_codec_budget() -> None:
    scheduler = object.__new__(Qwen35TalkerScheduler)
    data = SimpleNamespace(
        interleaved_codec_chunk_size=4,
        interleaved_codec_steps=4,
        interleaved_final=False,
        interleaved_boundary_ready=False,
        interleaved_drop_next_output=False,
    )
    req = SimpleNamespace(
        _omni_data=data,
        sampling_params=SimpleNamespace(max_new_tokens=5),
    )

    scheduler._mark_boundary_steps(SimpleNamespace(reqs=[req]))
    scheduler._mark_boundary_steps(SimpleNamespace(reqs=[req]))

    assert data.interleaved_drop_next_output
    assert req.sampling_params.max_new_tokens == 6


def test_qwen35_interleave_extend_replaces_boundary_feedback() -> None:
    scheduler = object.__new__(Qwen35TalkerScheduler)
    freed_slots = []
    scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int32).reshape(1, 64),
        free=lambda index: freed_slots.append(index),
    )
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(mambaish_config=None)
    )
    data = SimpleNamespace(
        pending_text_queue=deque(
            [torch.full((2,), float(index)) for index in range(4)]
        ),
        pending_feedback_queue=deque([torch.tensor([99.0, 99.0])]),
        prefill_input_embeds=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        interleaved_text_chunk_size=4,
        interleaved_codec_steps=4,
        interleaved_final=False,
        interleaved_boundary_ready=True,
        interleaved_placeholder_count=0,
        thinker_chunks_done=False,
        feedback_only_decode=False,
        codec_generation_steps=4,
    )
    req = SimpleNamespace(
        rid="r0",
        _omni_data=data,
        kv_committed_len=5,
        req_pool_idx=0,
        origin_input_ids=[8, 8, 8],
        output_ids=[100, 101, 102],
        sampling_params=SimpleNamespace(max_new_tokens=10),
        fill_ids=[],
    )
    req.set_extend_input_len = lambda length: setattr(req, "extend_input_len", length)

    scheduler._prepare_text_extend(req)

    assert data.prefill_input_embeds.shape == (9, 2)
    assert len(data.pending_feedback_queue) == 0
    assert len(req.output_ids) == 6
    assert req.sampling_params.max_new_tokens == 13
    assert data.interleaved_placeholder_count == 3
    assert req.extend_input_len == 4
    assert req.req_pool_idx is None
    assert freed_slots == [0]
    assert len(req.prefix_indices) == 5
    assert data.interleaved_codec_steps == 0
    assert not data.interleaved_boundary_ready
    assert not data.interleaved_final


def test_qwen35_interleave_final_tail_switches_to_feedback_only() -> None:
    data = SimpleNamespace(
        pending_text_queue=deque([torch.ones(2), torch.full((2,), 2.0)]),
        interleaved_text_chunk_size=4,
        thinker_chunks_done=True,
    )

    rows, is_final = Qwen35TalkerScheduler._take_text_rows(data)

    assert rows.shape == (2, 2)
    assert is_final
    assert len(data.pending_text_queue) == 0


def test_qwen35_prefill_configures_residual_sampler_before_predictor() -> None:
    calls = []
    model = SimpleNamespace(
        prepare_decode_buffers=lambda requests: calls.append(("prepare", requests)),
        code_predictor_forward=lambda codes, hidden: calls.append(
            ("predict", codes.clone(), hidden.clone())
        ),
    )
    runner = QwenTalkerModelRunner.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._emit_code_chunks_and_feedback = lambda **kwargs: None
    requests = [SimpleNamespace()]
    schedule_batch = SimpleNamespace(output_ids=None)
    result = SimpleNamespace(
        next_token_ids=torch.tensor([1995]),
        logits_output=SimpleNamespace(hidden_states=torch.ones(1, 1280)),
    )

    runner.post_prefill(result, None, schedule_batch, requests)

    assert calls[0] == ("prepare", requests)
    assert calls[1][0] == "predict"
