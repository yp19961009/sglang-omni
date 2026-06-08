# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from benchmarks.eval.benchmark_omni_streaming_ttft import (
    RunResult,
    Summary,
    _event_audio_data,
    _event_text_delta,
    _measure_one,
    _summary_payload,
)


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamClient:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.payload = None

    def stream(self, method, url, *, json, timeout):
        del method, url, timeout
        self.payload = json
        return _FakeStreamResponse(self._lines)


def test_streaming_ttft_detects_delta_and_vllm_top_level_audio_events():
    assert (
        _event_text_delta(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "hello",
                        }
                    }
                ]
            }
        )
        == "hello"
    )
    assert (
        _event_audio_data(
            {
                "choices": [
                    {
                        "delta": {
                            "audio": {"data": "delta-audio", "format": "wav"},
                        }
                    }
                ]
            }
        )
        == "delta-audio"
    )
    assert (
        _event_audio_data(
            {
                "object": "chat.completion.audio",
                "audio": {"data": "top-level-audio", "format": "wav"},
            }
        )
        == "top-level-audio"
    )


def test_streaming_ttft_payload_works_for_vllm_and_sglang():
    client = _FakeStreamClient(
        [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            (
                'data: {"object":"chat.completion.audio",'
                '"audio":{"data":"abc","format":"wav"}}'
            ),
            "data: [DONE]",
        ]
    )

    ttft, text_ttft, total, audio_chunks, status_code = asyncio.run(
        _measure_one(
            client,
            "http://localhost:8008",
            "qwen3.5-omni",
            "hello",
            request_id_hint="req-1",
            seed=7,
            timeout_s=30,
            max_tokens=128,
            voice="Cherry",
        )
    )

    assert ttft >= 0
    assert text_ttft is not None
    assert text_ttft <= ttft
    assert total >= ttft
    assert audio_chunks == 1
    assert status_code == 200
    assert client.payload["modalities"] == ["text", "audio"]
    assert client.payload["enable_audio_output"] is True
    assert client.payload["max_tokens"] == 128
    assert client.payload["audio"] == {"format": "wav", "voice": "Cherry"}
    assert client.payload["voice_type"] == "Cherry"


def test_streaming_ttft_summary_payload_records_config():
    summary = Summary(label="q35", base_url="http://localhost:8008")
    summary.per_run.append(
        RunResult(
            label="q35",
            prompt_id="short",
            repeat=0,
            ttft_seconds=1.2,
            text_ttft_seconds=0.8,
            total_seconds=3.4,
            audio_chunks=2,
            status_code=200,
        )
    )
    summary.aggregate["short"] = {"ttft_mean": 1.2, "total_mean": 3.4}
    args = SimpleNamespace(
        model="qwen3.5-omni",
        audio_format="wav",
        voice=None,
        max_tokens=128,
        warmup=1,
        repeats=2,
        timeout_s=30.0,
    )

    payload = _summary_payload(summary, args)

    assert payload["config"] == {
        "model": "qwen3.5-omni",
        "audio_format": "wav",
        "voice": None,
        "max_tokens": 128,
        "warmup": 1,
        "repeats": 2,
        "timeout_s": 30.0,
    }
    assert payload["per_run"][0]["audio_chunks"] == 2
    assert payload["per_run"][0]["text_ttft_seconds"] == 0.8
