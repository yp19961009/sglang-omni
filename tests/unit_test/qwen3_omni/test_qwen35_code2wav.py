# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen35_omni.components.code2wav_scheduler import (
    Qwen35Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload


class _FakeQwen35CodecDecoder:
    def __init__(self, upsample: int = 2) -> None:
        self.upsample = upsample
        self.calls: list[tuple[int, ...]] = []

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(codes.shape))
        samples = codes.shape[1] * self.upsample
        return torch.arange(samples, dtype=torch.float32).reshape(1, 1, samples)


def _code_chunk(chunk_id: int, *, metadata: dict | None) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=torch.arange(16, dtype=torch.long) + chunk_id,
        from_stage="talker_ar",
        metadata=metadata,
    )


def test_qwen35_code2wav_decodes_incrementally_and_clears_state() -> None:
    model = _FakeQwen35CodecDecoder()
    scheduler = Qwen35Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        codec_upsample_rate=2,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(
        request_id="req-1", params={"stream": False}
    )
    scheduler._ensure_request_state("req-1")

    metadata = {"stream": False}
    scheduler._on_chunk("req-1", _code_chunk(0, metadata=metadata))
    scheduler._on_chunk("req-1", _code_chunk(1, metadata=metadata))
    scheduler._on_chunk("req-1", _code_chunk(2, metadata=metadata))
    scheduler._on_done("req-1")

    result = scheduler.outbox.get_nowait()
    audio = np.frombuffer(result.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 8, 16), (1, 8, 16)]
    assert audio.shape == (6,)
    assert "req-1" not in scheduler._code_chunks
    assert "req-1" not in scheduler._payloads


def test_qwen35_code2wav_rejects_missing_stream_metadata() -> None:
    scheduler = Qwen35Code2WavScheduler(
        _FakeQwen35CodecDecoder(),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        codec_upsample_rate=2,
    )

    scheduler._on_chunk("req-1", _code_chunk(0, metadata=None))

    error = scheduler.outbox.get_nowait()
    assert error.type == "error"
    assert "metadata['stream']" in str(error.data)
    assert "req-1" not in scheduler._code_chunks
