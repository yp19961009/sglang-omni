# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components import (
    code2wav_scheduler as code2wav_module,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform emission and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._ensure_request_state("req-1")

    chunk_meta = {"stream": False}  # non-streaming: final result carries full PCM
    scheduler._on_chunk(
        "req-1",
        StreamItem(0, torch.tensor([1, 10]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(1, torch.tensor([2, 20]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(2, torch.tensor([3, 30]), "talker", metadata=chunk_meta),
    )
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._ensure_request_state("req-2")
    scheduler._pending_done.add("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._code_chunks
    assert "req-2" not in scheduler._payloads
    assert "req-2" not in scheduler._pending_done


def test_qwen_code2wav_profile_events_cover_collect_decode_and_finalize(
    monkeypatch,
) -> None:
    events = []
    monkeypatch.setattr(
        code2wav_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._ensure_request_state("req-1")

    chunk_meta = {"stream": False}
    scheduler._on_chunk(
        "req-1",
        StreamItem(0, torch.tensor([1, 10]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(1, torch.tensor([2, 20]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(2, torch.tensor([3, 30]), "talker", metadata=chunk_meta),
    )
    scheduler._on_done("req-1")

    names = [event["event_name"] for event in events]
    assert names.count("code2wav_chunk_received") == 3
    assert names.count("code2wav_window_collect_start") == 2
    assert names.count("code2wav_window_collect_end") == 2
    assert names.count("code2wav_decode_start") == 2
    assert names.count("code2wav_decode_end") == 2
    assert names.count("code2wav_finalize_start") == 1
    assert names.count("code2wav_finalize_end") == 1

    decode_end_events = [
        event for event in events if event["event_name"] == "code2wav_decode_end"
    ]
    assert [event["metadata"]["samples"] for event in decode_end_events] == [4, 2]
