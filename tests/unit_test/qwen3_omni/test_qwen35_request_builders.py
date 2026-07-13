# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen35_omni import request_builders


def _patch_sampling_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def _build_thinker_request(monkeypatch: pytest.MonkeyPatch):
    _patch_sampling_validation(monkeypatch)
    state = Qwen3OmniPipelineState(
        prompt={
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
        }
    )
    return request_builders.build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 7, "min_new_tokens": 2},
        tokenizer=SimpleNamespace(),
        vocab_size=256,
    )


def test_thinker_request_preserves_requested_token_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SGLANG_OMNI_BENCHMARK_FIXED_TEXT_TOKENS", raising=False)

    request = _build_thinker_request(monkeypatch)

    assert request.req.sampling_params.max_new_tokens == 7
    assert request.req.sampling_params.min_new_tokens == 2


def test_thinker_request_can_fix_text_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_BENCHMARK_FIXED_TEXT_TOKENS", "5")

    request = _build_thinker_request(monkeypatch)

    assert request.req.sampling_params.max_new_tokens == 5
    assert request.req.sampling_params.min_new_tokens == 5


def test_fixed_token_environment_requires_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_BENCHMARK_FIXED_TEXT_TOKENS", "0")

    with pytest.raises(ValueError, match="must be positive"):
        _build_thinker_request(monkeypatch)


class _FakeTalkerPrefillBuilder:
    def append_text_chunk(self, *args, **kwargs) -> None:
        return None

    def mark_thinker_done(self, *args, **kwargs) -> None:
        return None


def test_talker_request_can_fix_codec_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("SGLANG_OMNI_BENCHMARK_FIXED_CODEC_FRAMES", "12")
    monkeypatch.setattr(
        request_builders,
        "Qwen35TalkerPrefillBuilder",
        lambda **kwargs: _FakeTalkerPrefillBuilder(),
    )

    def _build_talker_request_data(
        payload,
        *,
        resolve_sampling_config,
        **kwargs,
    ):
        del kwargs
        captured.update(resolve_sampling_config(payload.request.params))
        return SimpleNamespace(thinker_chunks_done=True, stage_payload=payload)

    monkeypatch.setattr(
        request_builders.qwen3_builders,
        "_build_talker_request_data",
        _build_talker_request_data,
    )
    request_builder, *_ = request_builders.make_talker_scheduler_adapters(
        tokenizer=SimpleNamespace(),
        codec_vocab_size=10,
        valid_codec_vocab_size=8,
        model=SimpleNamespace(),
        model_path="/tmp/model",
        root_config=SimpleNamespace(
            talker_config=SimpleNamespace(
                codec_bos_id=7,
                codec_eos_token_id=9,
            )
        ),
        thinker_config=SimpleNamespace(
            audio_token_id=20,
            image_token_id=21,
            video_token_id=22,
        ),
    )

    request_builder(
        SimpleNamespace(
            request=SimpleNamespace(params={"talker_max_new_tokens": 99})
        )
    )

    assert captured["max_new_tokens"] == 12
    assert captured["suppress_tokens"] == [8, 9]
