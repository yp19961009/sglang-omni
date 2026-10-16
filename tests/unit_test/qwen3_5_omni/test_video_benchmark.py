# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import wave

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.videomme import VideoAMMESample
from benchmarks.eval.benchmark_omni_videomme import (
    add_video_eval_args,
    video_eval_config_from_args,
)
from benchmarks.tasks.video_understanding import (
    _apply_chat_completion_response,
    make_video_send_fn,
)


def _wav_b64() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 240)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _FakeResponse:
    def raise_for_status(self):
        return None

    async def json(self):
        return {
            "choices": [{"message": {"content": "Answer: A"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }


class _FakeStreamContent:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeStreamResponse:
    def __init__(self, lines):
        self.content = _FakeStreamContent(lines)

    def raise_for_status(self):
        return None


class _FakePost:
    def __init__(self, response=None):
        self._response = response or _FakeResponse()

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, response=None):
        self.payload = None
        self._response = response

    def post(self, url, *, json):
        del url
        self.payload = json
        return _FakePost(self._response)


def test_qwen35_video_benchmark_passes_audio_request_config():
    send_fn = make_video_send_fn(
        "qwen3.5-omni",
        "http://localhost:8008/v1/chat/completions",
        enable_audio_input=True,
        audio_output_dir="/tmp/videoamme_audio",
        audio_format="wav",
        audio_voice="Cherry",
        audio_language="zh",
    )
    sample = VideoAMMESample(
        sample_id="sample-1",
        video_path="/tmp/video.mp4",
        question="question",
        options=["yes", "no"],
        answer="A",
        prompt="fixed prompt",
        all_choices=["A", "B"],
        index2ans={"A": "yes", "B": "no"},
        audio_path="/tmp/question.wav",
    )
    session = _FakeSession()

    asyncio.run(send_fn(session, sample))

    assert session.payload["model"] == "qwen3.5-omni"
    assert session.payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/tmp/video.mp4"},
                {"type": "audio", "audio": "/tmp/question.wav"},
                {"type": "text", "text": "fixed prompt"},
            ],
        }
    ]
    assert "videos" not in session.payload
    assert "audios" not in session.payload
    assert session.payload["modalities"] == ["text", "audio"]
    assert session.payload["metadata"] == {"sample_id": "sample-1"}
    assert session.payload["audio"] == {
        "format": "wav",
        "voice": "Cherry",
        "language": "zh",
    }
    assert "enable_audio_output" not in session.payload
    assert "do_wave" not in session.payload
    assert "voice_type" not in session.payload


def test_qwen35_video_benchmark_rejects_non_wav_audio_metrics():
    with pytest.raises(ValueError, match="require wav output"):
        make_video_send_fn(
            "qwen3.5-omni",
            "http://localhost:8008/v1/chat/completions",
            audio_output_dir="/tmp/videoamme_audio",
            audio_format="mp3",
        )


def test_qwen35_video_benchmark_text_only_omits_audio_output_flag():
    send_fn = make_video_send_fn(
        "qwen3.5-omni",
        "http://localhost:8008/v1/chat/completions",
    )
    sample = VideoAMMESample(
        sample_id="sample-1",
        video_path="/tmp/video.mp4",
        question="question",
        options=["yes", "no"],
        answer="A",
        prompt="fixed prompt",
        all_choices=["A", "B"],
        index2ans={"A": "yes", "B": "no"},
        audio_path="/tmp/question.wav",
    )
    session = _FakeSession()

    asyncio.run(send_fn(session, sample))

    assert session.payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/tmp/video.mp4"},
                {"type": "text", "text": "fixed prompt"},
            ],
        }
    ]
    assert session.payload["modalities"] == ["text"]
    assert session.payload["metadata"] == {"sample_id": "sample-1"}
    assert "enable_audio_output" not in session.payload
    assert "do_wave" not in session.payload
    assert "audio" not in session.payload


def test_qwen35_video_benchmark_accepts_message_audio(tmp_path):
    result = RequestResult(request_id="sample-top")
    body = {
        "choices": [
            {
                "message": {
                    "content": "Answer: A",
                    "audio": {"data": _wav_b64(), "format": "wav"},
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }

    assert _apply_chat_completion_response(
        result,
        body,
        audio_output_dir=str(tmp_path),
        sample_id="sample-top",
    )
    assert result.text == "Answer: A"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 3
    assert result.audio_duration_s > 0
    assert result.wav_path == str(tmp_path / "sample-top.wav")
    assert (tmp_path / "sample-top.wav").exists()


def test_qwen35_video_benchmark_stream_records_audio_ttfp(tmp_path):
    audio_event = {
        "choices": [
            {
                "delta": {
                    "audio": {
                        "data": _wav_b64(),
                        "format": "wav",
                    }
                }
            }
        ]
    }
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        'data: {"choices":[{"delta":{"content":"Answer: "}}]}\n',
        f"data: {json.dumps(audio_event)}\n",
        f"data: {json.dumps(audio_event)}\n",
        (
            'data: {"choices":[{"delta":{"content":"A"}}],'
            '"usage":{"prompt_tokens":11,"completion_tokens":3}}\n'
        ),
        "data: [DONE]\n",
    ]
    send_fn = make_video_send_fn(
        "qwen3.5-omni",
        "http://localhost:8008/v1/chat/completions",
        enable_audio_input=True,
        audio_output_dir=str(tmp_path),
        audio_format="wav",
        audio_voice="Cherry",
        stream_output=True,
    )
    sample = VideoAMMESample(
        sample_id="sample-1",
        video_path="/tmp/video.mp4",
        question="question",
        options=["yes", "no"],
        answer="A",
        prompt="fixed prompt",
        all_choices=["A", "B"],
        index2ans={"A": "yes", "B": "no"},
        audio_path="/tmp/question.wav",
    )
    session = _FakeSession(_FakeStreamResponse(lines))

    result = asyncio.run(send_fn(session, sample))

    assert session.payload["stream"] is True
    assert "enable_audio_output" not in session.payload
    assert "do_wave" not in session.payload
    assert result.is_success
    assert result.text == "Answer: A"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 3
    assert result.audio_ttfp_s is not None
    assert result.text_ttft_s is not None
    assert result.audio_duration_s > 0
    assert len(result.inter_chunk_s) == 1
    assert result.wav_path == str(tmp_path / "sample-1.wav")
    assert (tmp_path / "sample-1.wav").exists()


def test_qwen35_video_benchmark_stream_saves_delta_audio_chunks(tmp_path):
    audio_event = {
        "choices": [
            {
                "delta": {
                    "audio": {
                        "data": _wav_b64(),
                        "format": "wav",
                    }
                }
            }
        ]
    }
    lines = [
        f"data: {json.dumps(audio_event)}\n",
        f"data: {json.dumps(audio_event)}\n",
        "data: [DONE]\n",
    ]
    send_fn = make_video_send_fn(
        "qwen3.5-omni",
        "http://localhost:8008/v1/chat/completions",
        audio_output_dir=str(tmp_path),
        stream_output=True,
    )
    sample = VideoAMMESample(
        sample_id="sample-delta",
        video_path="/tmp/video.mp4",
        question="question",
        options=["yes", "no"],
        answer="A",
        prompt="fixed prompt",
        all_choices=["A", "B"],
        index2ans={"A": "yes", "B": "no"},
        audio_path="/tmp/question.wav",
    )
    session = _FakeSession(_FakeStreamResponse(lines))

    result = asyncio.run(send_fn(session, sample))

    assert result.is_success
    assert result.audio_duration_s > 0
    assert result.wav_path == str(tmp_path / "sample-delta.wav")
    with wave.open(result.wav_path, "rb") as wav_file:
        assert wav_file.getnframes() == 480


def test_qwen35_video_benchmark_cli_accepts_audio_request_config():
    parser = argparse.ArgumentParser()
    add_video_eval_args(parser, repo_help="repo")

    args = parser.parse_args(
        [
            "--model",
            "qwen3.5-omni",
            "--sample-offset",
            "50",
            "--enable-audio",
            "--audio-format",
            "wav",
            "--audio-voice",
            "Cherry",
            "--audio-language",
            "zh",
            "--stream",
        ]
    )
    config = video_eval_config_from_args(args)

    assert config.model == "qwen3.5-omni"
    assert config.sample_offset == 50
    assert config.enable_audio is True
    assert config.audio_format == "wav"
    assert config.audio_voice == "Cherry"
    assert config.audio_language == "zh"
    assert config.stream is True


def test_qwen35_video_benchmark_cli_rejects_non_wav_audio_format():
    parser = argparse.ArgumentParser()
    add_video_eval_args(parser, repo_help="repo")

    with pytest.raises(SystemExit):
        parser.parse_args(["--enable-audio", "--audio-format", "mp3"])
