# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client, GenerateChunk
from sglang_omni.client.audio import encode_pcm
from sglang_omni.client.types import GenerateRequest
from sglang_omni.client.types import (
    CompletionAudio,
    CompletionResult,
    CompletionStreamChunk,
    GenerateRequest,
    UsageInfo,
)
from sglang_omni.models.qwen3_5_omni import request_builders as qwen35_builders
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_chat_generate_request,
    _build_speech_generate_request,
    _chat_non_stream,
    _chat_stream,
    _speech_stream,
    build_speech_generate_request,
    build_transcription_generate_request,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, CreateSpeechRequest
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

MODEL_FAMILIES = {
    "qwen3-omni": "code2wav",
    "ming-omni": "talker",
    "s2-pro": "vocoder",
    "voxtral": "vocoder",
}


class FaultInjectingCoordinator(Coordinator):
    """Inject a model-stage failure through the real Coordinator/Client path."""

    def __init__(self, terminal_stage: str):
        super().__init__(
            completion_endpoint="inproc://complete",
            abort_endpoint="inproc://abort",
            entry_stage="preprocess",
            terminal_stages=[terminal_stage],
        )
        self.control_plane = RecordingCoordinatorControlPlane()
        self.terminal_stage = terminal_stage
        self.register_stage("preprocess", "inproc://preprocess")

    async def _submit_request(
        self, request_id: str, request: OmniRequest | Any
    ) -> None:
        await super()._submit_request(request_id, request)
        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        if bool(request.params.get("stream", False)):
            await self._handle_stream(self._partial_stream_message(request_id, request))
        await self._handle_completion(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.terminal_stage,
                success=False,
                error="cuda out of memory",
            )
        )

    def _partial_stream_message(
        self, request_id: str, request: OmniRequest
    ) -> StreamMessage:
        if "tts_params" in request.metadata:
            chunk = {
                "audio_data": [0.0, 0.1],
                "sample_rate": 24000,
                "modality": "audio",
            }
            modality = "audio"
        else:
            chunk = {"text": "partial", "modality": "text"}
            modality = "text"
        return StreamMessage(
            request_id=request_id,
            from_stage=self.terminal_stage,
            chunk=chunk,
            stage_name=self.terminal_stage,
            modality=modality,
        )


def _fault_client(model_name: str) -> Client:
    return Client(FaultInjectingCoordinator(MODEL_FAMILIES[model_name]))


class SuccessfulSpeechClient:
    def __init__(self, *, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=self.sample_rate,
            finish_reason="stop",
        )


class SuccessfulTranscriptionClient:
    def __init__(self) -> None:
        self.requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        from sglang_omni.client.types import CompletionResult

        del request_id, audio_format
        self.requests.append(request)
        return CompletionResult(request_id="transcription-1", text="hello world")


class SuccessfulChatClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del request, audio_format
        return CompletionResult(
            request_id=request_id,
            text="hello",
            audio=CompletionAudio(
                id=f"audio-{request_id}",
                data="UklGRg==",
                transcript="hello",
            ),
        )


class SampleRateChatClient(SuccessfulChatClient):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del request, audio_format
        return CompletionResult(
            request_id=request_id,
            text="hello",
            audio=CompletionAudio(
                id=f"audio-{request_id}",
                data="UklGRg==",
                format="wav",
                sample_rate=48000,
                transcript="hello",
            ),
        )


class RecordingChatClient(SuccessfulChatClient):
    def __init__(self) -> None:
        self.requests: list[GenerateRequest] = []

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        self.requests.append(request)
        return await super().completion(
            request,
            request_id=request_id,
            audio_format=audio_format,
        )


class StreamingAudioChatClient:
    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        del request, audio_format
        yield CompletionStreamChunk(
            request_id=request_id,
            modality="audio",
            audio_b64="UklGRg==",
            audio_format="wav",
            sample_rate=48000,
        )


class StreamingUsageChatClient:
    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        del request, audio_format
        yield CompletionStreamChunk(request_id=request_id, text="hello")
        yield CompletionStreamChunk(
            request_id=request_id,
            finish_reason="stop",
            usage=UsageInfo(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )


@pytest.mark.parametrize("model_name", MODEL_FAMILIES)
def test_non_streaming_http_faults_return_500(model_name: str) -> None:
    client = TestClient(create_app(_fault_client(model_name), model_name=model_name))

    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 500
    assert "cuda out of memory" in chat_resp.json()["detail"]

    speech_resp = client.post(
        "/v1/audio/speech",
        json={
            "model": model_name,
            "input": "hello",
            "stream": False,
            "response_format": "wav",
        },
    )
    assert speech_resp.status_code == 500
    assert "cuda out of memory" in speech_resp.json()["detail"]


def test_chat_stream_failure_closes_without_done_sentinel() -> None:
    chunks: list[str] = []
    client = _fault_client("qwen3-omni")
    req = ChatCompletionRequest(
        model="qwen3-omni",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    async def _drive() -> None:
        async for chunk in _chat_stream(
            client=client,
            gen_req=GenerateRequest(model="qwen3-omni", prompt="hello", stream=True),
            request_id="req-1",
            response_id="chatcmpl-req-1",
            created=0,
            model="qwen3-omni",
            req=req,
            audio_format="opus",
        ):
            chunks.append(chunk)

    with pytest.raises(RuntimeError, match="cuda out of memory"):
        asyncio.run(_drive())

    assert chunks
    assert all(chunk != "data: [DONE]\n\n" for chunk in chunks)


def test_chat_stream_audio_delta_includes_format() -> None:
    chunks: list[str] = []
    req = ChatCompletionRequest(
        model="qwen3.5-omni",
        messages=[{"role": "user", "content": "say hi"}],
        stream=True,
        modalities=["text", "audio"],
    )

    async def _drive() -> None:
        async for chunk in _chat_stream(
            client=StreamingAudioChatClient(),
            gen_req=GenerateRequest(model="qwen3.5-omni", prompt="say hi", stream=True),
            request_id="req-audio",
            response_id="chatcmpl-req-audio",
            created=0,
            model="qwen3.5-omni",
            req=req,
            audio_format="wav",
        ):
            chunks.append(chunk)

    asyncio.run(_drive())

    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: {")
    ]
    audio_payload = next(
        payload
        for payload in payloads
        if payload["choices"][0]["delta"].get("audio") is not None
    )
    assert audio_payload["choices"][0]["delta"]["audio"] == {
        "id": "audio-req-audio",
        "data": "UklGRg==",
        "format": "wav",
        "sample_rate": 48000,
    }


def test_chat_stream_options_include_usage_emits_usage_chunk() -> None:
    chunks: list[str] = []
    req = ChatCompletionRequest(
        model="qwen3.5-omni",
        messages=[{"role": "user", "content": "say hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    async def _drive() -> None:
        async for chunk in _chat_stream(
            client=StreamingUsageChatClient(),
            gen_req=GenerateRequest(model="qwen3.5-omni", prompt="say hi", stream=True),
            request_id="req-usage",
            response_id="chatcmpl-req-usage",
            created=0,
            model="qwen3.5-omni",
            req=req,
            audio_format="wav",
        ):
            chunks.append(chunk)

    asyncio.run(_drive())

    assert chunks[-1] == "data: [DONE]\n\n"
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: {")
    ]
    finish_payload = payloads[-2]
    usage_payload = payloads[-1]

    assert finish_payload["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in finish_payload
    assert usage_payload["choices"] == []
    assert usage_payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


async def _collect_speech_stream(client: Any) -> list[str]:
    chunks: list[str] = []
    async for chunk in _speech_stream(
        client=client,
        gen_req=GenerateRequest(model="s2-pro", prompt="hello", stream=True),
        request_id="req-1",
        response_format="wav",
        speed=1.0,
    ):
        chunks.append(chunk)
    return chunks


def test_speech_stream_success_emits_done_sentinel() -> None:
    chunks = asyncio.run(_collect_speech_stream(SuccessfulSpeechClient()))

    assert chunks[-1] == "data: [DONE]\n\n"
    payload = json.loads(chunks[-2][len("data: ") :])
    assert payload["audio"] is None
    assert payload["finish_reason"] == "stop"


def test_speech_stream_defaults_to_sse_for_compatibility() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "audio.speech.chunk" in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_speech_stream_audio_format_returns_raw_pcm_bytes() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=24000)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_speech_stream_audio_format_headers_use_chunk_sample_rate() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(sample_rate=44100), model_name="s2-pro")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
        },
    )

    expected = encode_pcm([0.0, 0.1, -0.1, 0.0], sample_rate=44100)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-sample-rate"] == "44100"
    assert response.headers["x-channels"] == "1"
    assert response.headers["x-bit-depth"] == "16"
    assert response.content == expected


def test_speech_stream_audio_format_rejects_non_pcm_response_format() -> None:
    client = TestClient(
        create_app(SuccessfulSpeechClient(), model_name="higgs-audio-v2")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "stream": True,
            "stream_format": "audio",
            "response_format": "wav",
        },
    )

    assert 400 <= response.status_code < 500
    assert "stream_format" in response.text
    assert "pcm" in response.text.lower()


def test_speech_request_carries_initial_codec_chunk_frames() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
        initial_codec_chunk_frames=4,
    )

    gen_req = build_speech_generate_request(req, default_model="higgs-audio-v2")

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 4


def test_raw_pcm_speech_request_defaults_initial_codec_chunk_frames() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        stream_format="audio",
        response_format="pcm",
    )

    gen_req = build_speech_generate_request(req, default_model="higgs-audio-v2")

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 1


def test_sse_speech_request_does_not_default_initial_codec_chunk_frames() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        response_format="pcm",
    )

    gen_req = build_speech_generate_request(req, default_model="higgs-audio-v2")

    assert "initial_codec_chunk_frames" not in gen_req.extra_params


def test_raw_pcm_speech_request_respects_explicit_initial_zero() -> None:
    req = CreateSpeechRequest(
        input="hello",
        stream=True,
        stream_format="audio",
        response_format="pcm",
        initial_codec_chunk_frames=0,
    )

    gen_req = build_speech_generate_request(req, default_model="higgs-audio-v2")

    assert gen_req.extra_params["initial_codec_chunk_frames"] == 0


def test_speech_stream_failure_closes_without_done_sentinel() -> None:
    """A mid-stream failure must not be reported as a successful SSE finish."""

    chunks: list[str] = []
    client = _fault_client("s2-pro")

    async def _drive() -> None:
        async for chunk in _speech_stream(
            client=client,
            gen_req=GenerateRequest(
                model="s2-pro",
                prompt="hello",
                stream=True,
                metadata={"tts_params": {}},
            ),
            request_id="req-1",
            response_format="wav",
            speed=1.0,
        ):
            chunks.append(chunk)

    with pytest.raises(RuntimeError, match="cuda out of memory"):
        asyncio.run(_drive())

    assert chunks
    assert all(chunk != "data: [DONE]\n\n" for chunk in chunks)
    payload = json.loads(chunks[0][len("data: ") :])
    assert payload["audio"] is not None
    assert payload["finish_reason"] is None


def test_speech_request_records_explicit_generation_params() -> None:
    req = CreateSpeechRequest(
        input="hello",
        temperature=0.8,
        top_k=30,
        seed=123,
    )

    gen_req = build_speech_generate_request(req, "qwen3-tts")

    assert _build_speech_generate_request is build_speech_generate_request
    assert gen_req.sampling.temperature == 0.8
    assert gen_req.sampling.top_k == 30
    assert gen_req.sampling.seed == 123
    assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
        "seed",
        "temperature",
        "top_k",
    ]


def test_chat_request_preserves_requested_model_name() -> None:
    req = ChatCompletionRequest(
        model="qwen_3_5_omni",
        messages=[{"role": "user", "content": "hello"}],
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.model == "qwen_3_5_omni"


def test_chat_request_uses_default_model_name() -> None:
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
    )

    gen_req = _build_chat_generate_request(
        req,
        default_model="qwen3_5_omni",
    )

    assert gen_req.model == "qwen3_5_omni"


def test_speech_request_preserves_requested_model_name() -> None:
    req = CreateSpeechRequest.model_validate(
        {
            "model": "qwen3_5_omni",
            "input": "say hi",
            "voice": "Cherry",
        }
    )

    gen_req = build_speech_generate_request(req, "qwen3-omni")

    assert gen_req.model == "qwen3_5_omni"


def test_http_chat_preserves_model_name_in_response() -> None:
    client = TestClient(
        create_app(SuccessfulChatClient(), model_name="qwen3_5_omni")
    )

    models_resp = client.get("/v1/models")
    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen_3_5_omni",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert models_resp.json()["data"][0]["id"] == "qwen3_5_omni"
    assert chat_resp.json()["model"] == "qwen_3_5_omni"


def test_http_chat_uses_default_qwen35_model_when_request_omits_model() -> None:
    recording_client = RecordingChatClient()
    client = TestClient(create_app(recording_client, model_name="qwen3_5_omni"))

    chat_resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert chat_resp.json()["model"] == "qwen3_5_omni"
    assert recording_client.requests[-1].model == "qwen3_5_omni"


def test_http_chat_defaults_missing_finish_reason_to_stop() -> None:
    client = TestClient(create_app(SuccessfulChatClient(), model_name="qwen3_5_omni"))

    chat_resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert chat_resp.status_code == 200
    assert chat_resp.json()["choices"][0]["finish_reason"] == "stop"


def test_chat_request_merges_default_talker_params() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "modalities": ["text", "audio"],
        }
    )

    gen_req = _build_chat_generate_request(
        req,
        default_talker_params={"voice_type": "f245", "enable_tn": True},
    )

    assert gen_req.extra_params == {"voice_type": "f245", "enable_tn": True}

    override_req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "audio": {"voice": "Cherry", "format": "wav"},
        }
    )

    gen_req = _build_chat_generate_request(
        override_req,
        default_talker_params={"voice_type": "f245", "enable_tn": True},
    )

    assert gen_req.extra_params == {"enable_tn": True}
    omni_request = OmniRequest(
        inputs={},
        params=gen_req.extra_params,
        metadata=gen_req.metadata,
    )
    params = qwen35_builders._params_with_openai_audio_config(omni_request)
    assert params["voice_type"] == "Cherry"
    assert params["enable_tn"] is True


def test_speech_request_merges_default_talker_params() -> None:
    req = CreateSpeechRequest.model_validate({"input": "say hi"})

    gen_req = build_speech_generate_request(
        req,
        "qwen3.5-omni",
        default_talker_params={"voice_type": "f245", "enable_tn": True},
    )

    assert gen_req.metadata["tts_params"]["voice"] == "default"
    assert gen_req.extra_params == {"voice_type": "f245", "enable_tn": True}

    override_req = CreateSpeechRequest.model_validate(
        {"input": "say hi", "voice": "Cherry"}
    )

    gen_req = build_speech_generate_request(
        override_req,
        "qwen3.5-omni",
        default_talker_params={"voice_type": "f245", "enable_tn": True},
    )

    assert gen_req.extra_params == {"voice_type": "f245", "enable_tn": True}
    assert gen_req.metadata["tts_params"]["voice"] == "Cherry"


def test_http_chat_applies_default_talker_params() -> None:
    recording_client = RecordingChatClient()
    client = TestClient(
        create_app(
            recording_client,
            model_name="qwen3_5_omni",
            default_talker_params={"voice_type": "f245"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert recording_client.requests[-1].extra_params == {"voice_type": "f245"}


def test_chat_request_merges_default_generation_params() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
        }
    )

    gen_req = _build_chat_generate_request(
        req,
        default_generation_params={
            "temperature": 0.000001,
            "top_k": 1,
            "top_p": 0.8,
            "max_tokens": 2048,
            "seed": 0,
        },
    )

    assert gen_req.sampling.temperature == 0.000001
    assert gen_req.sampling.top_k == 1
    assert gen_req.sampling.top_p == 0.8
    assert gen_req.sampling.max_new_tokens == 2048
    assert gen_req.sampling.seed == 0
    assert gen_req.max_tokens == 2048

    override_req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "temperature": 0.2,
            "top_k": 4,
            "max_tokens": 128,
            "seed": 99,
        }
    )

    gen_req = _build_chat_generate_request(
        override_req,
        default_generation_params={
            "temperature": 0.000001,
            "top_k": 1,
            "max_tokens": 2048,
            "seed": 0,
        },
    )

    assert gen_req.sampling.temperature == 0.2
    assert gen_req.sampling.top_k == 4
    assert gen_req.sampling.max_new_tokens == 128
    assert gen_req.sampling.seed == 99
    assert gen_req.max_tokens == 128


def test_chat_request_stage_overrides_prefer_top_level_fields() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "stage_sampling": {
                "decode": {
                    "temperature": 0.1,
                }
            },
            "stage_params": {"decode": {"max_prefill_tokens": 64}},
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.stage_sampling is not None
    assert set(gen_req.stage_sampling) == {"decode"}
    assert gen_req.stage_sampling["decode"].temperature == 0.1
    assert gen_req.stage_params == {"decode": {"max_prefill_tokens": 64}}


def test_speech_request_merges_default_generation_params() -> None:
    req = CreateSpeechRequest.model_validate({"input": "say hi"})

    gen_req = build_speech_generate_request(
        req,
        "qwen3.5-omni",
        default_generation_params={
            "temperature": 0.000001,
            "top_k": 1,
            "top_p": 0.8,
            "max_tokens": 2048,
            "seed": 0,
        },
    )

    assert gen_req.sampling.temperature == 0.000001
    assert gen_req.sampling.top_k == 1
    assert gen_req.sampling.top_p == 0.8
    assert gen_req.sampling.max_new_tokens == 2048
    assert gen_req.sampling.seed == 0
    assert gen_req.metadata["tts_params"]["seed"] == 0
    assert "explicit_generation_params" not in gen_req.metadata["tts_params"]

    override_req = CreateSpeechRequest.model_validate(
        {
            "input": "say hi",
            "temperature": 0.2,
            "max_tokens": 128,
            "seed": 99,
        }
    )

    gen_req = build_speech_generate_request(
        override_req,
        "qwen3.5-omni",
        default_generation_params={
            "temperature": 0.000001,
            "max_tokens": 2048,
            "seed": 0,
        },
    )

    assert gen_req.sampling.temperature == 0.2
    assert gen_req.sampling.max_new_tokens == 128
    assert gen_req.sampling.seed == 99
    assert gen_req.metadata["tts_params"]["seed"] == 99
    assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
        "max_tokens",
        "seed",
        "temperature",
    ]


def test_speech_request_accepts_top_level_stage_sampling() -> None:
    req = CreateSpeechRequest.model_validate(
        {
            "input": "say hi",
            "stage_sampling": {
                "talker_ar": {
                    "temperature": 0.25,
                }
            },
            "stage_params": {"code2wav": {"frequency": 25}},
        }
    )

    gen_req = build_speech_generate_request(req, "qwen3.5-omni")

    assert gen_req.stage_sampling is not None
    assert gen_req.stage_sampling["talker_ar"].temperature == 0.25
    assert gen_req.stage_params == {"code2wav": {"frequency": 25}}


def test_http_chat_applies_default_generation_params() -> None:
    recording_client = RecordingChatClient()
    client = TestClient(
        create_app(
            recording_client,
            model_name="qwen3_5_omni",
            default_generation_params={"max_tokens": 2048, "seed": 0},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    request = recording_client.requests[-1]
    assert request.max_tokens == 2048
    assert request.sampling.max_new_tokens == 2048
    assert request.sampling.seed == 0


def test_chat_request_passes_qwen35_preprocessing_params() -> None:
    req = ChatCompletionRequest(
        model="qwen3.5-omni",
        messages=[{"role": "user", "content": "watch"}],
        modalities=["text", "audio"],
        images=["frame.png"],
        videos=["clip.mp4"],
        audios=["speech.wav"],
        image_min_pixels=100352,
        image_max_pixels=401408,
        video_fps=2,
        video_min_frames=4,
        video_max_frames=128,
        video_min_pixels=100352,
        video_max_pixels=401408,
        video_total_pixels=32768 * 768,
        use_audio_in_video=[True],
        dependent_audio=[0],
        video_seconds_per_chunk=2.0,
        video_position_id_per_seconds=25,
        return_video_metadata=True,
        video_metadata=[{"fps": 2.0}],
        audio_target_sr=16000,
        audio_timestamp_interval=15,
        audio_downsample_times=4,
        audio_downsample_chunk_size=100,
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.output_modalities == ["text", "audio"]
    assert gen_req.metadata == {
        "audios": ["speech.wav"],
        "images": ["frame.png"],
        "videos": ["clip.mp4"],
        "image_min_pixels": 100352,
        "image_max_pixels": 401408,
        "video_fps": 2,
        "video_min_frames": 4,
        "video_max_frames": 128,
        "video_min_pixels": 100352,
        "video_max_pixels": 401408,
        "video_total_pixels": 32768 * 768,
        "use_audio_in_video": [True],
        "dependent_audio": [0],
        "video_seconds_per_chunk": 2.0,
        "video_position_id_per_seconds": 25,
        "return_video_metadata": True,
        "video_metadata": [{"fps": 2.0}],
        "audio_target_sr": 16000,
        "audio_timestamp_interval": 15,
        "audio_downsample_times": 4,
        "audio_downsample_chunk_size": 100,
    }


def test_chat_request_keeps_audio_input_separate_from_output_config() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "listen"}],
            "audios": ["speech.wav"],
            "audio": {"voice": "Cherry", "format": "wav"},
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.metadata["audios"] == ["speech.wav"]
    assert gen_req.metadata["audio_config"] == {"voice": "Cherry", "format": "wav"}


def test_chat_request_uses_top_level_sampling_params() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "temperature": 0.2,
            "top_p": 0.7,
            "top_k": 5,
            "min_p": 0.1,
            "repetition_penalty": 1.05,
            "frequency_penalty": 0.25,
            "presence_penalty": 1.5,
            "max_tokens": 77,
            "seed": 123,
            "stop": ["<stop>"],
        }
    )

    gen_req = _build_chat_generate_request(req)
    sampling = gen_req.sampling.to_dict()

    assert sampling["temperature"] == 0.2
    assert sampling["top_p"] == 0.7
    assert sampling["top_k"] == 5
    assert sampling["min_p"] == 0.1
    assert sampling["repetition_penalty"] == 1.05
    assert sampling["frequency_penalty"] == 0.25
    assert sampling["presence_penalty"] == 1.5
    assert sampling["max_new_tokens"] == 77
    assert sampling["seed"] == 123
    assert sampling["stop"] == ["<stop>"]
    assert gen_req.max_tokens == 77


def test_chat_request_uses_modalities_field_for_audio_output() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "modalities": ["text", "audio"],
        }
    )

    assert _build_chat_generate_request(req).output_modalities == ["text", "audio"]


def test_chat_request_passes_tools_to_preprocessor_metadata() -> None:
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
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": tools,
            "tool_choice": "auto",
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.metadata["tools"] == tools
    assert gen_req.metadata["tool_choice"] == "auto"


def test_chat_request_wraps_legacy_functions_as_tools() -> None:
    functions = [
        {
            "name": "get_weather",
            "description": "Return the weather.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "weather?"}],
            "functions": functions,
            "function_call": "auto",
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.metadata["tools"] == [
        {"type": "function", "function": functions[0]}
    ]
    assert gen_req.metadata["function_call"] == "auto"


def test_chat_request_preserves_tool_call_messages() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }
    ]
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {
                    "role": "tool",
                    "content": '{"temperature": 21}',
                    "name": "get_weather",
                    "tool_call_id": "call_1",
                },
            ],
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert [message.to_dict() for message in gen_req.messages] == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {
            "role": "tool",
            "content": '{"temperature": 21}',
            "name": "get_weather",
            "tool_call_id": "call_1",
        },
    ]


def test_chat_request_preserves_legacy_function_call_messages() -> None:
    function_call = {"name": "get_weather", "arguments": "{}"}
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call,
                },
                {
                    "role": "function",
                    "content": '{"temperature": 21}',
                    "name": "get_weather",
                },
            ],
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert [message.to_dict() for message in gen_req.messages] == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "function_call": function_call},
        {
            "role": "function",
            "content": '{"temperature": 21}',
            "name": "get_weather",
        },
    ]


def test_chat_request_preserves_user_metadata_without_overriding_internal_fields() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "metadata": {
                "client_label": "ttft-smoke-0",
                "audio_config": {"voice": "metadata-voice"},
            },
            "audio": {"voice": "Cherry", "format": "wav"},
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.metadata["client_label"] == "ttft-smoke-0"
    assert gen_req.metadata["audio_config"] == {"voice": "Cherry", "format": "wav"}


def test_chat_request_preserves_tracking_fields_in_metadata() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "metadata": {
                "request_id": "metadata-request",
                "user": "metadata-user",
                "client_label": "profile-run-0",
            },
            "request_id": "client-request-1",
            "user": "user-1",
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.metadata["request_id"] == "client-request-1"
    assert gen_req.metadata["user"] == "user-1"
    assert gen_req.metadata["client_label"] == "profile-run-0"


def test_chat_request_passes_qwen3_talker_overrides() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "talker_temperature": 0.2,
            "talker_top_p": 0.7,
            "talker_top_k": 3,
            "talker_repetition_penalty": 1.2,
            "talker_max_new_tokens": 88,
        }
    )

    gen_req = _build_chat_generate_request(req)

    assert gen_req.extra_params == {
        "talker_temperature": 0.2,
        "talker_top_p": 0.7,
        "talker_top_k": 3,
        "talker_repetition_penalty": 1.2,
        "talker_max_new_tokens": 88,
    }


def test_chat_request_audio_config_flows_to_qwen35_talker_params() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "say hi"}],
            "audio": {
                "voice": "Cherry",
                "language": "zh-CN",
                "voice_style": "happy",
                "instruction": "Speak softly",
                "xvector_info": "/voices/from-audio",
            },
        }
    )

    gen_req = _build_chat_generate_request(req)
    omni_request = OmniRequest(
        inputs={},
        params=gen_req.extra_params,
        metadata=gen_req.metadata,
    )

    params = qwen35_builders._params_with_openai_audio_config(omni_request)

    assert gen_req.extra_params == {}
    assert params["voice_type"] == "Cherry"
    assert params["language"] == "zh-CN"
    assert params["voice_style"] == "happy"
    assert params["instruction"] == "Speak softly"
    assert params["xvector_info"] == "/voices/from-audio"


def test_chat_request_audio_target_language_flows_to_talker_params() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.5-omni",
            "messages": [{"role": "user", "content": "translate this"}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "Cherry", "target_language": "en"},
        }
    )

    gen_req = _build_chat_generate_request(req)
    omni_request = OmniRequest(
        inputs={},
        params=gen_req.extra_params,
        metadata=gen_req.metadata,
    )
    params = qwen35_builders._params_with_openai_audio_config(omni_request)

    assert params["voice_type"] == "Cherry"
    assert params["language"] == "en"


def test_chat_non_stream_modalities_returns_audio() -> None:
    req = ChatCompletionRequest(
        model="qwen3.5-omni",
        messages=[{"role": "user", "content": "say hi"}],
        modalities=["text", "audio"],
    )

    response = asyncio.run(
        _chat_non_stream(
            client=SuccessfulChatClient(),
            gen_req=GenerateRequest(model="qwen3.5-omni", prompt="say hi"),
            request_id="req-audio",
            response_id="chatcmpl-req-audio",
            created=0,
            model="qwen3.5-omni",
            req=req,
            audio_format="wav",
        )
    )

    payload = json.loads(response.body)
    message = payload["choices"][0]["message"]
    assert message["content"] == "hello"
    assert message["audio"] == {
        "id": "audio-req-audio",
        "data": "UklGRg==",
        "format": "wav",
        "transcript": "hello",
    }
    assert "audio" not in payload


def test_chat_non_stream_audio_includes_sample_rate_when_available() -> None:
    req = ChatCompletionRequest(
        model="qwen3.5-omni",
        messages=[{"role": "user", "content": "say hi"}],
        modalities=["text", "audio"],
    )

    response = asyncio.run(
        _chat_non_stream(
            client=SampleRateChatClient(),
            gen_req=GenerateRequest(model="qwen3.5-omni", prompt="say hi"),
            request_id="req-audio",
            response_id="chatcmpl-req-audio",
            created=0,
            model="qwen3.5-omni",
            req=req,
            audio_format="wav",
        )
    )

    payload = json.loads(response.body)
    message = payload["choices"][0]["message"]
    assert message["audio"]["sample_rate"] == 48000
    assert "audio" not in payload


def test_transcription_request_builds_asr_generate_request() -> None:
    gen_req = build_transcription_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="openai/whisper-large-v3",
        language="en",
        prompt=None,
        temperature=None,
    )

    assert gen_req.model == "openai/whisper-large-v3"
    assert gen_req.prompt == {
        "audio_bytes": b"RIFF",
        "filename": "sample.wav",
        "content_type": "audio/wav",
    }
    assert gen_req.extra_params == {"task": "transcribe", "language": "en"}
    assert gen_req.metadata == {"task": "asr"}
    assert gen_req.output_modalities == ["text"]
    assert gen_req.stream is False


def test_transcription_endpoint_returns_text_json() -> None:
    transcription_client = SuccessfulTranscriptionClient()
    client = TestClient(
        create_app(transcription_client, model_name="openai/whisper-large-v3")
    )

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-large-v3", "language": "en"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
    assert transcription_client.requests
    request = transcription_client.requests[0]
    assert request.model == "openai/whisper-large-v3"
    assert request.prompt["filename"] == "sample.wav"
    assert request.extra_params["language"] == "en"


def test_speech_request_passes_moss_token_count() -> None:
    req = CreateSpeechRequest(input="hello", token_count=180)

    gen_req = build_speech_generate_request(req, "moss-tts")

    assert gen_req.metadata["tts_params"]["token_count"] == 180
