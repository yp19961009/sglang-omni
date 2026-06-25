# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_5_omni.components.code2wav_scheduler import (
    Qwen35Code2WavScheduler,
    _warmup_code2wav_decode,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem


def test_qwen35_code2wav_skips_out_of_range_codec_rows():
    model = SimpleNamespace(total_upsample=1, codebook_size=2048)
    scheduler = Qwen35Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        sample_rate=24000,
        codec_eos_token_id=2150,
    )

    chunk = StreamItem(
        chunk_id=0,
        from_stage="talker_ar",
        data=torch.tensor([12, 2150, 4, 5], dtype=torch.long),
        metadata={"stream": True},
    )

    assert scheduler._should_skip_invalid_codec_row("req-1", chunk)


def test_qwen35_code2wav_accepts_valid_codec_rows():
    model = SimpleNamespace(total_upsample=1, codebook_size=2048)
    scheduler = Qwen35Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        sample_rate=24000,
        codec_eos_token_id=2150,
    )

    chunk = StreamItem(
        chunk_id=0,
        from_stage="talker_ar",
        data=torch.tensor([12, 2047, 4, 5], dtype=torch.long),
        metadata={"stream": True},
    )

    assert not scheduler._should_skip_invalid_codec_row("req-1", chunk)


def test_qwen35_code2wav_reads_nested_quantizer_bins():
    quantizer = SimpleNamespace(
        rvq_first=SimpleNamespace(bins=2048),
        rvq_rest=SimpleNamespace(bins=2048),
    )
    model = SimpleNamespace(total_upsample=1, quantizer=quantizer)
    scheduler = Qwen35Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        sample_rate=24000,
        codec_eos_token_id=2150,
    )

    chunk = StreamItem(
        chunk_id=0,
        from_stage="talker_ar",
        data=torch.tensor([12, 2150, 4, 5], dtype=torch.long),
        metadata={"stream": True},
    )

    assert scheduler._codec_codebook_size() == 2048
    assert scheduler._should_skip_invalid_codec_row("req-1", chunk)


class _WarmupModel:
    total_upsample = 1
    codebook_nums = 16

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.detach().clone())
        return torch.zeros(1, 1)


def test_qwen35_code2wav_compile_warmup_matches_first_chunk_shape(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_QWEN35_CODE2WAV_COMPILE_WARMUP", "1")
    model = _WarmupModel()

    assert _warmup_code2wav_decode(
        model,
        device="cpu",
        stream_chunk_size=4,
        left_context_size=25,
    )

    assert len(model.calls) == 1
    assert tuple(model.calls[0].shape) == (1, 32, 16)
    assert model.calls[0].dtype == torch.long


def test_qwen35_code2wav_compile_warmup_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_QWEN35_CODE2WAV_COMPILE_WARMUP", raising=False)
    model = _WarmupModel()

    assert not _warmup_code2wav_decode(
        model,
        device="cpu",
        stream_chunk_size=4,
        left_context_size=25,
    )

    assert model.calls == []


def test_qwen35_code2wav_compile_warmup_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_QWEN35_CODE2WAV_COMPILE_WARMUP", "0")
    model = _WarmupModel()

    assert not _warmup_code2wav_decode(
        model,
        device="cpu",
        stream_chunk_size=4,
        left_context_size=25,
    )

    assert model.calls == []
