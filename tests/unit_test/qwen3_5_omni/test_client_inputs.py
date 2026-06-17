# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import numpy as np

from sglang_omni.client import client as client_module
from sglang_omni.client.client import Client, _extract_inputs
from sglang_omni.client.types import (
    GenerateChunk,
    GenerateRequest,
    Message,
    SamplingParams,
)


class _AudioCompletionClient(Client):
    def __init__(self, *, sample_rate: int | None = None) -> None:
        self.sample_rate = sample_rate

    async def generate(self, request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id or "req-audio",
            text="hello",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=self.sample_rate,
            finish_reason="stop",
        )


def test_client_extract_inputs_keeps_plain_prompt_backward_compatible():
    request = GenerateRequest(prompt="hello")

    assert _extract_inputs(request) == "hello"


def test_client_extract_inputs_preserves_openai_tool_message_fields():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }
    ]
    request = GenerateRequest(
        messages=[
            Message(role="assistant", content=None, tool_calls=tool_calls),
            Message(
                role="tool",
                content='{"temperature": 21}',
                name="get_weather",
                tool_call_id="call_1",
            ),
        ],
    )

    assert _extract_inputs(request) == [
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {
            "role": "tool",
            "content": '{"temperature": 21}',
            "name": "get_weather",
            "tool_call_id": "call_1",
        },
    ]


def test_client_extract_inputs_preserves_legacy_function_call_messages():
    function_call = {"name": "get_weather", "arguments": "{}"}
    request = GenerateRequest(
        messages=[
            Message(role="assistant", content=None, function_call=function_call),
            Message(
                role="function",
                content='{"temperature": 21}',
                name="get_weather",
            ),
        ],
    )

    assert _extract_inputs(request) == [
        {"role": "assistant", "content": None, "function_call": function_call},
        {
            "role": "function",
            "content": '{"temperature": 21}',
            "name": "get_weather",
        },
    ]


def test_client_extract_inputs_preserves_raw_prompt_multimodal_metadata():
    request = GenerateRequest(
        prompt="<|im_start|>user\n<|vision_start|><|video_pad|>",
        metadata={
            "videos": ["clip.mp4"],
            "audios": ["voice.wav"],
            "video_fps": 2,
            "video_min_frames": 4,
            "video_max_frames": 128,
            "use_audio_in_video": True,
            "dependent_audio": [0],
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": request.prompt,
        "videos": ["clip.mp4"],
        "audios": ["voice.wav"],
        "video_fps": 2,
        "video_min_frames": 4,
        "video_max_frames": 128,
        "use_audio_in_video": True,
        "dependent_audio": [0],
    }


def test_client_extract_inputs_preserves_token_prompt_multimodal_metadata():
    request = GenerateRequest(
        prompt_token_ids=[1, 2, 3],
        metadata={
            "images": ["frame.png"],
            "audios": ["speech.wav"],
            "image_max_pixels": 401408,
            "audio_timestamp_interval": 15,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt_token_ids": [1, 2, 3],
        "images": ["frame.png"],
        "audios": ["speech.wav"],
        "image_max_pixels": 401408,
        "audio_timestamp_interval": 15,
    }


def test_client_extract_inputs_preserves_chat_template_tools_metadata():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    request = GenerateRequest(
        messages=[Message(role="user", content="weather?")],
        metadata={"tools": tools, "tool_choice": "auto"},
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": tools,
        "tool_choice": "auto",
    }


def test_client_extract_inputs_preserves_singular_media_aliases():
    request = GenerateRequest(
        prompt="<rendered qwen3.5 prompt>",
        metadata={
            "image": "frame.png",
            "video": "clip.mp4",
            "audio": "speech.wav",
            "video_fps": 2,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": "<rendered qwen3.5 prompt>",
        "image": "frame.png",
        "video": "clip.mp4",
        "audio": "speech.wav",
        "video_fps": 2,
    }


def test_client_extract_inputs_skips_openai_audio_output_config_alias():
    request = GenerateRequest(
        prompt="<rendered qwen3.5 prompt>",
        metadata={
            "audio": {"voice": "Cherry", "format": "wav"},
            "video": "clip.mp4",
            "video_fps": 2,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": "<rendered qwen3.5 prompt>",
        "video": "clip.mp4",
        "video_fps": 2,
    }


def test_client_extract_inputs_keeps_audio_input_dict_alias_with_payload():
    input_audio = {"data": "UklGRg==", "format": "wav"}
    request = GenerateRequest(
        prompt="<rendered qwen3.5 prompt>",
        metadata={
            "audio": input_audio,
            "audio_timestamp_interval": 15,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": "<rendered qwen3.5 prompt>",
        "audio": input_audio,
        "audio_timestamp_interval": 15,
    }


def test_client_extract_inputs_preserves_qwen35_image_video_url_aliases():
    request = GenerateRequest(
        prompt="<rendered qwen3.5 prompt>",
        metadata={
            "image_url": "frame.png",
            "input_video": "clip.mp4",
            "video_url": ["clip-b.mp4"],
            "video_fps": 2,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": "<rendered qwen3.5 prompt>",
        "image_url": "frame.png",
        "input_video": "clip.mp4",
        "video_url": ["clip-b.mp4"],
        "video_fps": 2,
    }


def test_client_extract_inputs_preserves_qwen35_audio_input_aliases():
    input_audio = {"data": "UklGRg==", "format": "wav"}
    request = GenerateRequest(
        prompt="<rendered qwen3.5 prompt>",
        metadata={
            "input_audio": input_audio,
            "audio_url": "speech.wav",
            "audio_timestamp_interval": 15,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": "<rendered qwen3.5 prompt>",
        "input_audio": input_audio,
        "audio_url": "speech.wav",
        "audio_timestamp_interval": 15,
    }


def test_client_extract_inputs_accepts_array_media_metadata():
    audio = np.zeros(16000, dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    request = GenerateRequest(
        prompt_token_ids=[1, 2, 3],
        metadata={
            "audios": audio,
            "images": image,
            "audio_target_sr": 16000,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs["prompt_token_ids"] == [1, 2, 3]
    assert inputs["audios"] is audio
    assert inputs["images"] is image
    assert inputs["audio_target_sr"] == 16000


def test_client_extract_inputs_preserves_audio_sampling_rate_alias():
    request = GenerateRequest(
        prompt="<rendered qwen3.5 prompt>",
        metadata={
            "audios": ["speech.wav"],
            "sampling_rate": 16000,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "prompt": "<rendered qwen3.5 prompt>",
        "audios": ["speech.wav"],
        "sampling_rate": 16000,
    }


def test_client_extract_inputs_keeps_message_media_path():
    request = GenerateRequest(
        messages=[Message(role="user", content="describe")],
        metadata={
            "images": ["frame.png"],
            "video_total_pixels": 32768 * 768,
        },
    )

    inputs = _extract_inputs(request)

    assert inputs == {
        "messages": [{"role": "user", "content": "describe"}],
        "images": ["frame.png"],
        "video_total_pixels": 32768 * 768,
    }


def test_client_build_omni_request_preserves_qwen35_runtime_contract():
    request = GenerateRequest(
        model="qwen3.5-omni",
        prompt="<rendered>",
        sampling=SamplingParams(
            max_new_tokens=64,
            temperature=0.5,
            frequency_penalty=0.25,
            presence_penalty=1.5,
        ),
        output_modalities=["text", "audio"],
        extra_params={
            "voice_type": "Cherry",
            "xvector_info": "/voices/ref-a",
        },
        metadata={
            "videos": ["clip.mp4"],
            "video_fps": 2,
            "use_audio_in_video": True,
        },
    )

    omni_request = Client._build_omni_request(request)

    assert omni_request.inputs == {
        "prompt": "<rendered>",
        "videos": ["clip.mp4"],
        "video_fps": 2,
        "use_audio_in_video": True,
    }
    assert omni_request.params["max_new_tokens"] == 64
    assert omni_request.params["temperature"] == 0.5
    assert omni_request.params["frequency_penalty"] == 0.25
    assert omni_request.params["presence_penalty"] == 1.5
    assert omni_request.params["voice_type"] == "Cherry"
    assert omni_request.params["xvector_info"] == "/voices/ref-a"
    assert omni_request.metadata["model"] == "qwen3.5-omni"
    assert omni_request.metadata["output_modalities"] == ["text", "audio"]


def test_client_build_omni_request_keeps_audio_output_config_in_metadata_only():
    audio_config = {"voice": "Cherry", "format": "wav"}
    request = GenerateRequest(
        model="qwen3.5-omni",
        prompt="<rendered>",
        output_modalities=["text", "audio"],
        metadata={
            "audio": audio_config,
            "video": "clip.mp4",
            "video_fps": 2,
        },
    )

    omni_request = Client._build_omni_request(request)

    assert omni_request.inputs == {
        "prompt": "<rendered>",
        "video": "clip.mp4",
        "video_fps": 2,
    }
    assert omni_request.metadata["audio"] == audio_config
    assert omni_request.metadata["audio_config"] == audio_config
    assert omni_request.metadata["output_modalities"] == ["text", "audio"]


def test_client_build_omni_request_does_not_promote_audio_input_as_config():
    input_audio = {"data": "UklGRg==", "format": "wav"}
    request = GenerateRequest(
        model="qwen3.5-omni",
        prompt="<rendered>",
        metadata={
            "audio": input_audio,
            "audio_timestamp_interval": 15,
        },
    )

    omni_request = Client._build_omni_request(request)

    assert omni_request.inputs == {
        "prompt": "<rendered>",
        "audio": input_audio,
        "audio_timestamp_interval": 15,
    }
    assert omni_request.metadata["audio"] == input_audio
    assert "audio_config" not in omni_request.metadata


def test_client_completion_audio_preserves_response_format():
    result = asyncio.run(
        _AudioCompletionClient().completion(
            GenerateRequest(model="qwen3.5-omni", prompt="hello"),
            request_id="req-audio",
            audio_format="wav",
        )
    )

    assert result.audio is not None
    assert result.audio.format == "wav"
    assert result.audio.transcript == "hello"


def test_client_completion_audio_uses_chunk_sample_rate(monkeypatch):
    calls = []

    def _fake_encode_audio(
        audio,
        *,
        response_format="wav",
        sample_rate=24000,
        speed=1.0,
    ):
        calls.append(
            {
                "audio": list(audio),
                "sample_rate": sample_rate,
                "response_format": response_format,
                "speed": speed,
            }
        )
        return b"encoded", client_module.FORMAT_MIME_TYPES["wav"]

    monkeypatch.setattr(client_module, "encode_audio", _fake_encode_audio)

    result = asyncio.run(
        _AudioCompletionClient(sample_rate=48000).completion(
            GenerateRequest(model="qwen3.5-omni", prompt="hello"),
            request_id="req-audio",
            audio_format="opus",
        )
    )

    assert calls == [
        {
            "audio": [0.0, 0.1, -0.1, 0.0],
            "sample_rate": 48000,
            "response_format": "opus",
            "speed": 1.0,
        }
    ]
    assert result.audio is not None
    assert result.audio.data == "ZW5jb2RlZA=="
    assert result.audio.format == "wav"
    assert result.audio.sample_rate == 48000


def test_client_completion_stream_audio_uses_chunk_sample_rate(monkeypatch):
    calls = []

    def _fake_encode_audio(
        audio,
        *,
        response_format="wav",
        sample_rate=24000,
        speed=1.0,
    ):
        calls.append(
            {
                "audio": list(audio),
                "sample_rate": sample_rate,
                "response_format": response_format,
                "speed": speed,
            }
        )
        return b"encoded", client_module.FORMAT_MIME_TYPES["wav"]

    monkeypatch.setattr(client_module, "encode_audio", _fake_encode_audio)

    async def _collect():
        chunks = []
        async for chunk in _AudioCompletionClient(
            sample_rate=48000
        ).completion_stream(
            GenerateRequest(model="qwen3.5-omni", prompt="hello"),
            request_id="req-audio",
            audio_format="opus",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert calls == [
        {
            "audio": [0.0, 0.1, -0.1, 0.0],
            "sample_rate": 48000,
            "response_format": "opus",
            "speed": 1.0,
        }
    ]
    assert chunks[0].audio_format == "wav"
    assert chunks[0].sample_rate == 48000
    assert chunks[0].audio_b64 == "ZW5jb2RlZA=="
