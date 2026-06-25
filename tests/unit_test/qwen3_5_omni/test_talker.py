# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_5_omni.components import subtalker, talker


class _FakeBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.last_inputs_embeds = None

    def forward(self, input_ids, positions, forward_batch, inputs_embeds=None):
        del positions, forward_batch
        self.last_inputs_embeds = inputs_embeds
        if inputs_embeds is not None:
            return inputs_embeds
        return self.embed_tokens(input_ids)


class _FakeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.proj(hidden_states), None


class _FakeLanguageModel(nn.Module):
    instances = []

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.model = _FakeBackbone(config)
        self.lm_head = _FakeHead(config)
        self.loaded_weights = None
        self.instances.append(self)

    def load_weights(self, weights):
        self.loaded_weights = list(weights)
        return {name for name, _ in self.loaded_weights}


class _FakeVocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        del kwargs
        super().__init__(num_embeddings, embedding_dim)


class _FakeProjection(nn.Module):
    instances = []

    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.linear = nn.Linear(in_size, out_size)
        self.instances.append(self)

    def forward(self, x):
        return self.linear(x)


def test_qwen35_talker_moe_arch_alias_matches_next_talker():
    assert (
        talker.Qwen3OmniNextMoeTalkerForConditionalGeneration
        is talker.Qwen3OmniNextTalkerForConditionalGeneration
    )
    assert (
        talker.Qwen3OmniNextTalkerModel
        is talker.Qwen3OmniNextTalkerForConditionalGeneration
    )


class _ForwardMode:
    def __init__(self, *, extend=False, decode=False):
        self._extend = extend
        self._decode = decode

    def is_extend(self):
        return self._extend

    def is_decode(self):
        return self._decode


def _config():
    text_config = SimpleNamespace(
        vocab_size=8,
        text_vocab_size=13,
        hidden_size=4,
    )
    code_predictor_config = SimpleNamespace(num_code_groups=3, vocab_size=6)
    code_predictor_config.hidden_size = 4
    code_predictor_config.talker_hidden_size = 4
    talker_config = SimpleNamespace(
        text_config=text_config,
        thinker_hidden_size=4,
        code_predictor_config=code_predictor_config,
        num_code_groups=3,
        codec_pad_id=2,
        codec_eos_token_id=5,
    )
    return SimpleNamespace(talker_config=talker_config)


def _split_config_with_none_talker_config():
    return SimpleNamespace(talker_config=None, **vars(_config().talker_config))


def _install_fakes(monkeypatch):
    _FakeLanguageModel.instances.clear()
    _FakeProjection.instances.clear()
    monkeypatch.setattr(talker, "Qwen3NextForCausalLM", _FakeLanguageModel)
    monkeypatch.setattr(talker, "VocabParallelEmbedding", _FakeVocabParallelEmbedding)
    monkeypatch.setattr(talker, "_LinearProjection", _FakeProjection)


def _decode_sched_req(**data_overrides):
    sampling_params = SimpleNamespace(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        sampling_seed=None,
        repetition_penalty=1.0,
    )
    data = SimpleNamespace(
        req=SimpleNamespace(
            sampling_params=sampling_params,
            output_ids=[],
            _codec_suppress_tokens=None,
        ),
        suppress_tokens=None,
        pending_text_queue=[],
        thinker_chunks_done=False,
    )
    for key, value in data_overrides.items():
        setattr(data, key, value)
    return SimpleNamespace(data=data)


def test_qwen35_talker_exposes_runtime_contract(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())

    assert model.get_input_embeddings() is model.model.codec_embedding
    assert model.activation_dtype == model.model.codec_embedding.weight.dtype
    assert model.code_predictor.implementation == "sglang_next_decoder"
    assert model._output_codes.shape == (1, 3)
    assert model._feedback_buffer.shape == (1, 4)


def test_qwen35_talker_warms_subtalker_code_predictor(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(talker, "_max_running_requests", lambda: 8)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    calls = []

    def _fake_code_predictor_forward(layer0_codes, talker_hidden, requests=None):
        calls.append((tuple(layer0_codes.shape), tuple(talker_hidden.shape), requests))
        return None

    monkeypatch.setattr(
        model,
        "code_predictor_forward",
        _fake_code_predictor_forward,
    )

    warmed = model.warmup_subtalker_code_predictor(
        batch_sizes=[1, 2, 4, 8, 8, 99, 0]
    )

    def _request_temperature(reqs):
        if reqs is None:
            return None
        params = reqs[0].data.req._qwen35_subtalker_sampling_params
        return None if params is None else params.temperature

    def _request_top_k(reqs):
        if reqs is None:
            return None
        params = reqs[0].data.req._qwen35_subtalker_sampling_params
        return None if params is None else params.top_k

    assert warmed == [1, 2, 4, 8]
    assert [(shape, hidden_shape, reqs is None) for shape, hidden_shape, reqs in calls] == [
        ((1,), (1, 4), True),
        ((1,), (1, 4), False),
        ((1,), (1, 4), True),
        ((1,), (1, 4), False),
        ((2,), (2, 4), True),
        ((2,), (2, 4), False),
        ((2,), (2, 4), True),
        ((2,), (2, 4), False),
        ((4,), (4, 4), True),
        ((4,), (4, 4), False),
        ((4,), (4, 4), True),
        ((4,), (4, 4), False),
        ((8,), (8, 4), True),
        ((8,), (8, 4), False),
        ((8,), (8, 4), True),
        ((8,), (8, 4), False),
    ]
    assert [len(reqs) for _, _, reqs in calls if reqs is not None] == [
        1,
        1,
        2,
        2,
        4,
        4,
        8,
        8,
    ]
    assert [_request_temperature(reqs) for _, _, reqs in calls] == [
        None,
        None,
        None,
        0.1,
        None,
        None,
        None,
        0.1,
        None,
        None,
        None,
        0.1,
        None,
        None,
        None,
        0.1,
    ]
    assert [_request_top_k(reqs) for _, _, reqs in calls] == [
        None,
        None,
        None,
        5,
        None,
        None,
        None,
        5,
        None,
        None,
        None,
        5,
        None,
        None,
        None,
        5,
    ]


def test_qwen35_talker_accepts_split_config_with_none_subconfig(monkeypatch):
    _install_fakes(monkeypatch)
    config = _split_config_with_none_talker_config()

    model = talker.Qwen3OmniNextTalkerModel(config)

    assert model.config is config
    assert _FakeLanguageModel.instances[-1].config is config.text_config


def test_qwen35_talker_normalizes_qwen3_next_layer_types(monkeypatch):
    _install_fakes(monkeypatch)
    config = _config()
    config.talker_config.text_config.layer_types = [
        "full_attention",
        "linear_attention",
    ]

    talker.Qwen3OmniNextTalkerModel(config)

    assert config.talker_config.text_config.layer_types == [
        "full_attention",
        "linear_attention",
    ]
    assert _FakeLanguageModel.instances[-1].config.layers_block_type == [
        "attention",
        "linear_attention",
    ]
    assert _FakeLanguageModel.instances[-1].config.partial_rotary_factor == 0.25
    assert _FakeLanguageModel.instances[-1].config.torch_dtype is None


def test_qwen35_talker_accepts_direct_split_text_config(monkeypatch):
    _install_fakes(monkeypatch)
    config = _split_config_with_none_talker_config()
    config.text_config = None
    config.vocab_size = 8
    config.text_vocab_size = 13
    config.hidden_size = 4
    config.num_hidden_layers = 4
    config.linear_num_key_heads = 16
    config.full_attention_interval = 2
    config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }

    model = talker.Qwen3OmniNextTalkerModel(config)

    assert model.text_config is config
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


def test_qwen35_talker_normalizes_code_predictor_config(monkeypatch):
    _install_fakes(monkeypatch)
    config = _config()
    config.talker_config.code_predictor_config.model_type = "qwen3_5_text"
    config.talker_config.code_predictor_config.num_hidden_layers = 2
    config.talker_config.code_predictor_config.full_attention_interval = 1
    config.talker_config.code_predictor_config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }

    model = talker.Qwen3OmniNextTalkerModel(config)

    assert model.code_predictor_config.layer_types == [
        "full_attention",
        "full_attention",
    ]
    assert model.code_predictor_config.layers_block_type == [
        "attention",
        "attention",
    ]
    assert model.code_predictor_config.rope_scaling == {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    assert model.code_predictor_config.rope_theta == 500000.0
    assert model.code_predictor_config.partial_rotary_factor == 0.25


def test_qwen35_talker_recognizes_next_code_predictor_model_type(monkeypatch):
    _install_fakes(monkeypatch)
    config = _config()
    config.talker_config.code_predictor_config.model_type = (
        "qwen3_omni_next_talker_code_predictor"
    )
    config.talker_config.code_predictor_config.num_hidden_layers = 2

    model = talker.Qwen3OmniNextTalkerModel(config)

    assert model.code_predictor_config.layer_types == [
        "linear_attention",
        "linear_attention",
    ]
    assert model.code_predictor_config.layers_block_type == [
        "linear_attention",
        "linear_attention",
    ]


def test_qwen35_talker_backfills_layer_types_from_sglang_alias(monkeypatch):
    _install_fakes(monkeypatch)
    config = _config()
    config.talker_config.text_config.layers_block_type = [
        "attention",
        "linear_attention",
    ]

    talker.Qwen3OmniNextTalkerModel(config)

    assert _FakeLanguageModel.instances[-1].config.layer_types == [
        "full_attention",
        "linear_attention",
    ]
    assert _FakeLanguageModel.instances[-1].config.layers_block_type == [
        "attention",
        "linear_attention",
    ]


def test_qwen35_talker_drops_inherited_root_quant_config(monkeypatch):
    _install_fakes(monkeypatch)
    quant_config = object()

    talker.Qwen3OmniNextTalkerModel(_config(), quant_config=quant_config)

    assert _FakeLanguageModel.instances[-1].quant_config is None
    assert _FakeProjection.instances[-1].kwargs["quant_config"] is None


def test_qwen35_talker_keeps_text_config_quant_config(monkeypatch):
    _install_fakes(monkeypatch)
    config = _config()
    config.talker_config.text_config.quantization_config = {"bits": 8}
    quant_config = object()

    talker.Qwen3OmniNextTalkerModel(config, quant_config=quant_config)

    assert _FakeLanguageModel.instances[-1].quant_config is quant_config
    assert _FakeProjection.instances[-1].kwargs["quant_config"] is quant_config


def test_qwen35_talker_forward_projected_prefill(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    input_embeds = torch.ones(2, 4)
    forward_batch = SimpleNamespace(
        mrope_positions=None,
        forward_mode=_ForwardMode(extend=True),
        input_ids=torch.tensor([1, 2]),
        extend_seq_lens=None,
        padded_static_len=None,
    )

    output = model(
        input_ids=forward_batch.input_ids,
        positions=torch.tensor([0, 1]),
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        input_embeds_are_projected=True,
    )

    assert output.next_token_logits.shape == (1, 8)
    assert output.hidden_states.shape == (1, 4)
    assert model.model.last_inputs_embeds is input_embeds


def test_qwen35_talker_decode_uses_feedback_buffer(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    feedback = torch.full((4,), 7.0, dtype=model._feedback_buffer.dtype)
    model._feedback_buffer[0].copy_(feedback)
    model._feedback_mask[0] = True
    forward_batch = SimpleNamespace(
        mrope_positions=None,
        forward_mode=_ForwardMode(decode=True),
        positions=torch.tensor([0]),
    )

    model(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        forward_batch=forward_batch,
    )

    assert torch.equal(model.model.last_inputs_embeds[0], feedback)
    assert not bool(model._feedback_mask[0].item())


def test_qwen35_talker_code_predictor_fills_runtime_buffers(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    for head in model.code_predictor.lm_head:
        nn.init.zeros_(head.linear.weight)
    model._subtalker_temperature.zero_()
    layer0_codes = torch.tensor([[4]])
    hidden = torch.ones(1, 1, 4)

    codes, embeds = model.code_predictor_forward(layer0_codes, hidden)

    assert codes.shape == (1, 3, 1)
    assert codes[:, :, 0].tolist() == [[4, 0, 0]]
    assert model._output_codes.tolist() == [[4, 0, 0]]
    assert torch.allclose(model._output_embeds[0], embeds[0, 0])


def test_qwen35_talker_code_predictor_uses_incremental_kv_cache(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    calls = []
    original_forward = model.code_predictor.model.forward

    def _wrapped_forward(inputs_embeds, *args, **kwargs):
        calls.append(
            {
                "seq_len": inputs_embeds.shape[1],
                "use_cache": kwargs.get("use_cache"),
                "past_key_values": kwargs.get("past_key_values"),
            }
        )
        return original_forward(inputs_embeds, *args, **kwargs)

    monkeypatch.setattr(model.code_predictor.model, "forward", _wrapped_forward)

    model.code_predictor_forward(
        torch.tensor([[4]]),
        torch.ones(1, 1, 4),
    )

    assert [call["seq_len"] for call in calls] == [2, 1]
    assert all(call["use_cache"] is True for call in calls)
    assert calls[0]["past_key_values"] is calls[1]["past_key_values"]


def test_qwen35_subtalker_incremental_cache_matches_full_recompute():
    torch.manual_seed(0)
    config = SimpleNamespace(
        num_code_groups=3,
        vocab_size=8,
        hidden_size=4,
        talker_hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        intermediate_size=8,
        head_dim=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    model = subtalker._NextPredictorModel(config)
    inputs = torch.randn(2, 3, 4)
    positions = torch.arange(3).unsqueeze(0).expand(2, -1)

    full = model(inputs, position_ids=positions)
    cache = subtalker._NextKVCache(len(model.layers))
    first = model(
        inputs[:, :2],
        position_ids=positions[:, :2],
        past_key_values=cache,
        use_cache=True,
    )
    second = model(
        inputs[:, 2:],
        position_ids=positions[:, 2:],
        past_key_values=cache,
        use_cache=True,
    )

    assert torch.allclose(first, full[:, :2], atol=1e-5, rtol=1e-5)
    assert torch.allclose(second, full[:, 2:], atol=1e-5, rtol=1e-5)


def test_qwen35_talker_embeds_speaker_codec_codes(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    with torch.no_grad():
        for token_id in range(model.model.codec_embedding.num_embeddings):
            model.model.codec_embedding.weight[token_id].fill_(float(token_id))
        for token_id in range(
            model.code_predictor.model.codec_embedding[0].num_embeddings
        ):
            model.code_predictor.model.codec_embedding[0].weight[token_id].fill_(
                10.0 + token_id
            )
            model.code_predictor.model.codec_embedding[1].weight[token_id].fill_(
                20.0 + token_id
            )
    model.speaker_codec_embeddings = nn.Parameter(
        torch.tensor([[[1, 2, 3], [4, 5, 0], [-1, -1, -1]]]),
        requires_grad=False,
    )

    codes = model.speaker_codec_codes(0)
    embeds = model.speaker_codec_input_embeddings(0)

    assert codes.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert embeds.tolist() == [[36.0] * 4, [39.0] * 4]


def test_qwen35_talker_transposes_checkpoint_speaker_codec_codes(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())

    loaded = model.load_weights(
        [
            (
                "talker.speaker_codec_embeddings",
                torch.tensor([[[1, 4], [2, 5], [3, 0]]]),
            )
        ]
    )

    assert "speaker_codec_embeddings" in loaded
    assert model.speaker_codec_embeddings.shape == (1, 2, 3)
    assert model.speaker_codec_codes(0).tolist() == [[1, 2, 3], [4, 5, 0]]


def test_qwen35_talker_embeds_clone_codec_codes_layout(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    with torch.no_grad():
        for token_id in range(model.model.codec_embedding.num_embeddings):
            model.model.codec_embedding.weight[token_id].fill_(float(token_id))
        for token_id in range(
            model.code_predictor.model.codec_embedding[0].num_embeddings
        ):
            model.code_predictor.model.codec_embedding[0].weight[token_id].fill_(
                10.0 + token_id
            )
            model.code_predictor.model.codec_embedding[1].weight[token_id].fill_(
                20.0 + token_id
            )

    embeds = model.codec_code_embeddings(torch.tensor([[1, 4], [2, 5], [3, 0]]))

    assert embeds.tolist() == [[36.0] * 4, [39.0] * 4]


def test_qwen35_talker_uses_subtalker_defaults_without_subtalker_params(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        layer0_codes = kwargs["layer0_codes"]
        batch_size, seq_len = layer0_codes.shape
        codes = torch.zeros(batch_size, 3, seq_len, dtype=torch.long)
        codes[:, 0, :] = layer0_codes
        embeds = torch.zeros(batch_size, seq_len, 4)
        return codes, embeds

    model.code_predictor.generate = _fake_generate
    sampling_params = SimpleNamespace(temperature=0.7, top_k=3, top_p=0.8)
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(sampling_params=sampling_params)
            )
        )
    ]

    model.code_predictor_forward(
        torch.tensor([[4]]),
        torch.ones(1, 1, 4),
        requests=requests,
    )

    assert torch.allclose(
        captured["temperature"].cpu(),
        torch.tensor([model._subtalker_default_temperature]),
    )
    assert captured["top_k"].tolist() == [model._subtalker_default_top_k]
    assert torch.allclose(
        captured["top_p"].cpu(),
        torch.tensor([model._subtalker_default_top_p]),
    )
    assert torch.allclose(
        captured["min_p"].cpu(),
        torch.tensor([model._subtalker_default_min_p]),
    )


def test_qwen35_talker_prefers_subtalker_sampling_params(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        layer0_codes = kwargs["layer0_codes"]
        batch_size, seq_len = layer0_codes.shape
        codes = torch.zeros(batch_size, 3, seq_len, dtype=torch.long)
        codes[:, 0, :] = layer0_codes
        embeds = torch.zeros(batch_size, seq_len, 4)
        return codes, embeds

    model.code_predictor.generate = _fake_generate
    talker_params = SimpleNamespace(temperature=0.9, top_k=50, top_p=1.0)
    subtalker_params = SimpleNamespace(
        temperature=0.1,
        top_k=5,
        top_p=0.95,
        min_p=0.03,
        sampling_seed=456,
    )
    req = SimpleNamespace(
        sampling_params=talker_params,
        _qwen35_subtalker_sampling_params=subtalker_params,
    )
    requests = [SimpleNamespace(data=SimpleNamespace(req=req))]

    model.code_predictor_forward(
        torch.tensor([[4]]),
        torch.ones(1, 1, 4),
        requests=requests,
    )

    assert torch.allclose(captured["temperature"].cpu(), torch.tensor([0.1]))
    assert captured["top_k"].tolist() == [5]
    assert torch.allclose(captured["top_p"].cpu(), torch.tensor([0.95]))
    assert torch.allclose(captured["min_p"].cpu(), torch.tensor([0.03]))
    assert captured["seed"].tolist() == [456]


def test_qwen35_talker_decode_suppresses_codec_eos_while_text_pending(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    requests = [
        _decode_sched_req(
            pending_text_queue=[torch.ones(4)],
            thinker_chunks_done=False,
        )
    ]
    logits = torch.zeros(1, 8)
    logits[0, 5] = 10.0
    logits[0, 4] = 8.0

    model.prepare_decode_buffers(requests)
    sampled = model._sample_decode_tokens(logits)

    assert sampled.tolist() == [4]


def test_qwen35_talker_decode_allows_codec_eos_after_text_done(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    requests = [
        _decode_sched_req(
            pending_text_queue=[],
            thinker_chunks_done=True,
        )
    ]
    logits = torch.zeros(1, 8)
    logits[0, 5] = 10.0
    logits[0, 4] = 8.0

    model.prepare_decode_buffers(requests)
    sampled = model._sample_decode_tokens(logits)

    assert sampled.tolist() == [5]


def test_qwen35_talker_decode_config_cache_keeps_dynamic_masks(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    apply_calls = []
    apply_decode_buffer_config = model._apply_decode_buffer_config

    def wrapped_apply_decode_buffer_config(**kwargs):
        apply_calls.append(kwargs)
        return apply_decode_buffer_config(**kwargs)

    model._apply_decode_buffer_config = wrapped_apply_decode_buffer_config
    sched_req = _decode_sched_req(thinker_chunks_done=True)
    sched_req.data.req.sampling_params.repetition_penalty = 2.0
    sched_req.data.req.output_ids = [5]
    logits = torch.zeros(1, 8)
    logits[0, 5] = 10.0
    logits[0, 4] = 8.0

    model.prepare_decode_buffers([sched_req])
    assert model._sample_decode_tokens(logits).tolist() == [4]

    sched_req.data.req.output_ids = [4]
    model.prepare_decode_buffers([sched_req])
    assert model._sample_decode_tokens(logits).tolist() == [5]
    assert len(apply_calls) == 1

    sched_req.data.req.sampling_params.top_k = 4
    model.prepare_decode_buffers([sched_req])
    assert len(apply_calls) == 2


def test_qwen35_talker_decode_uses_sglang_sampler_metadata(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    captured = {}

    class FakeSampler:
        def __call__(
            self,
            logits_output,
            sampling_info,
            *args,
        ):
            del logits_output, args
            captured["temperatures"] = sampling_info.temperatures.clone()
            captured["top_ks"] = sampling_info.top_ks.clone()
            captured["top_ps"] = sampling_info.top_ps.clone()
            captured["min_ps"] = sampling_info.min_ps.clone()
            captured["sampling_seed"] = sampling_info.sampling_seed.clone()
            captured["is_all_greedy"] = sampling_info.is_all_greedy
            captured["need_top_p_sampling"] = sampling_info.need_top_p_sampling
            captured["need_top_k_sampling"] = sampling_info.need_top_k_sampling
            captured["need_min_p_sampling"] = sampling_info.need_min_p_sampling
            return torch.tensor([3])

    sampling_params = SimpleNamespace(
        temperature=0.6,
        top_k=4,
        top_p=0.75,
        min_p=0.1,
        sampling_seed=123,
        repetition_penalty=1.0,
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    sampling_params=sampling_params,
                    output_ids=[],
                    _codec_suppress_tokens=None,
                ),
                suppress_tokens=None,
            )
        )
    ]
    model.prepare_decode_buffers(requests)
    model._sampler = FakeSampler()

    sampled = model._sample_decode_tokens(
        torch.zeros(1, 8),
        SimpleNamespace(positions=torch.tensor([0])),
    )

    assert sampled.tolist() == [3]
    assert torch.allclose(captured["temperatures"].cpu(), torch.tensor([[0.6]]))
    assert captured["top_ks"].tolist() == [4]
    assert torch.allclose(captured["top_ps"].cpu(), torch.tensor([0.75]))
    assert torch.allclose(captured["min_ps"].cpu(), torch.tensor([0.1]))
    assert captured["sampling_seed"].tolist() == [123]
    assert captured["is_all_greedy"] is False
    assert captured["need_top_p_sampling"] is True
    assert captured["need_top_k_sampling"] is True
    assert captured["need_min_p_sampling"] is True


def test_qwen35_talker_sampling_metadata_disables_unused_filters(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    sampling_params = SimpleNamespace(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        sampling_seed=None,
        repetition_penalty=1.0,
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    sampling_params=sampling_params,
                    output_ids=[],
                    _codec_suppress_tokens=None,
                ),
                suppress_tokens=None,
            )
        )
    ]

    model.prepare_decode_buffers(requests)
    sampling_info = model._build_static_sampling_info(1)

    assert sampling_info.is_all_greedy is True
    assert sampling_info.need_top_p_sampling is False
    assert sampling_info.need_top_k_sampling is False
    assert sampling_info.need_min_p_sampling is False


def test_qwen_talker_runner_passes_requests_when_supported():
    class Model:
        def __init__(self):
            self.seen_requests = None

        def code_predictor_forward(self, layer0_codes, talker_hidden, requests=None):
            del layer0_codes, talker_hidden
            self.seen_requests = requests
            return "ok"

    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = Model()
    runner._code_predictor_accepts_requests = None
    requests = [object()]

    result = runner._run_code_predictor_forward(
        torch.tensor([[1]]),
        torch.ones(1, 1, 4),
        requests,
    )

    assert result == "ok"
    assert runner.model.seen_requests is requests


def test_qwen35_talker_load_weights_maps_prefixes(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    weight = torch.ones(1)
    text_weight = torch.ones_like(model.text_embedding.weight)
    hidden_weight = torch.ones_like(model.hidden_projection.linear.weight)
    cp_embed_weight = torch.ones_like(
        model.code_predictor.model.codec_embedding[0].weight
    )
    cp_head_weight = torch.ones_like(model.code_predictor.lm_head[0].linear.weight)

    loaded = model.load_weights(
        [
            ("talker.codec_head.weight", weight),
            ("talker.model.codec_embedding.weight", weight),
            ("talker.model.embed_tokens.weight", text_weight),
            ("talker.hidden_projection.weight", hidden_weight),
            ("talker.code_predictor.model.codec_embedding.0.weight", cp_embed_weight),
            ("talker.code_predictor.lm_head.0.weight", cp_head_weight),
            ("thinker.model.embed_tokens.weight", weight),
        ]
    )
    language_model = _FakeLanguageModel.instances[-1]

    assert [name for name, _ in language_model.loaded_weights] == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    assert "text_embedding.weight" in loaded
    assert "hidden_projection.linear.weight" in loaded
    assert "code_predictor.model.codec_embedding.0.weight" in loaded
    assert "code_predictor.lm_head.0.linear.weight" in loaded


def test_qwen35_talker_skips_derived_rotary_inv_freq(monkeypatch):
    _install_fakes(monkeypatch)
    model = talker.Qwen3OmniNextTalkerModel(_config())
    weight = torch.ones(1)

    loaded = model.load_weights(
        [
            ("talker.model.rotary_emb.inv_freq", weight),
            ("talker.code_predictor.model.rotary_emb.inv_freq", weight),
            ("talker.codec_head.weight", weight),
        ]
    )
    language_model = _FakeLanguageModel.instances[-1]

    assert [name for name, _ in language_model.loaded_weights] == [
        "lm_head.weight"
    ]
    assert not any("rotary_emb.inv_freq" in name for name in loaded)
