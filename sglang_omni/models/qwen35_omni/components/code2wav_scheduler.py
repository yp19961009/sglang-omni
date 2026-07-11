# SPDX-License-Identifier: Apache-2.0
"""Streaming scheduler for the external Qwen3.5 public_v1 codec decoder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.qwen35_omni.components.code2wav import (
    load_qwen3_omni_next_dac_from_config,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

DEFAULT_CODEC_DIR = "qwen3_5_omni_codec_decode_online_0306"


def resolve_code2wav_model_path(
    model_path: str, code2wav_model_path: str | None = None
) -> Path:
    path = Path(code2wav_model_path) if code2wav_model_path else Path(model_path)
    if not (path / "config.yaml").is_file():
        path = path / DEFAULT_CODEC_DIR
    missing = [
        name
        for name in ("config.yaml", "model_weights.pt")
        if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Qwen3.5 code2wav assets missing from {path}: {', '.join(missing)}"
        )
    return path


class Qwen35Code2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        model: Any,
        *,
        device: str,
        stream_chunk_size: int = 4,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_upsample_rate: int = 1920,
        codec_eos_token_id: int = 4198,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(1, int(stream_chunk_size))
        self._left_context_size = max(0, int(left_context_size))
        self._sample_rate = int(sample_rate)
        self._codec_upsample_rate = int(codec_upsample_rate)
        self._codec_eos_token_id = int(codec_eos_token_id)
        max_window = self._left_context_size + self._stream_chunk_size
        self._padding_interval = (max_window // 8 + 1) * 8
        self._code_chunks: dict[str, list[torch.Tensor]] = {}
        self._emitted: dict[str, int] = {}
        self._audio_chunks: dict[str, list[np.ndarray]] = {}
        self._stream_enabled: dict[str, bool] = {}
        super().__init__(compute_fn=None)
        self._payloads = self._stream_payloads

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        del payload
        self._ensure_request_state(request_id)

    def clear_stream_state(self, request_id: str) -> None:
        self._code_chunks.pop(request_id, None)
        self._emitted.pop(request_id, None)
        self._audio_chunks.pop(request_id, None)
        self._stream_enabled.pop(request_id, None)

    def _fail_request(self, request_id: str, error: Exception) -> None:
        self.outbox.put(
            OutgoingMessage(request_id=request_id, type="error", data=error)
        )
        self.abort(request_id)

    def _ensure_request_state(self, request_id: str) -> None:
        if request_id in self._code_chunks:
            return
        self._code_chunks[request_id] = []
        self._emitted[request_id] = 0
        self._audio_chunks[request_id] = []

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        self._ensure_request_state(request_id)
        if request_id not in self._stream_enabled:
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else None
            if metadata is None or "stream" not in metadata:
                self._fail_request(
                    request_id,
                    RuntimeError(
                        f"Qwen3.5 code2wav got a chunk for {request_id!r} "
                        "without metadata['stream']"
                    ),
                )
                return []
            self._stream_enabled[request_id] = bool(metadata["stream"])

        codes = chunk.data.to(device=self._device, dtype=torch.long).reshape(-1)
        if codes.shape[0] != 16:
            raise ValueError(
                f"Qwen3.5 code2wav expects 16 code groups, got {codes.shape[0]}"
            )
        if int(codes[0].item()) == self._codec_eos_token_id:
            return []
        logger.debug(
            "qwen35_codec_chunk request_id=%s index=%d codes=%s",
            request_id,
            len(self._code_chunks[request_id]),
            codes.tolist(),
        )
        self._code_chunks[request_id].append(codes)
        ready = len(self._code_chunks[request_id]) - self._emitted[request_id]
        if ready >= self._stream_chunk_size:
            return self._decode_and_emit(request_id)
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        self._ensure_request_state(request_id)
        messages: list[OutgoingMessage] = []
        if self._emitted[request_id] < len(self._code_chunks[request_id]):
            messages.extend(self._decode_and_emit(request_id))

        audio_parts = self._audio_chunks[request_id]
        if not audio_parts:
            self._fail_request(
                request_id,
                RuntimeError("Qwen3.5 code2wav produced no audio"),
            )
            return []

        full_audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        payload = self._payloads[request_id]
        final_data = (
            {"modality": "audio", "sample_rate": self._sample_rate}
            if self._stream_enabled.get(request_id, False)
            else self._build_audio_payload(full_audio)
        )
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def _decode_and_emit(self, request_id: str) -> list[OutgoingMessage]:
        chunks = self._code_chunks[request_id]
        start = self._emitted[request_id]
        end = len(chunks)
        context = min(self._left_context_size, start)
        window = torch.stack(chunks[start - context : end], dim=0)
        valid_window_len = int(window.shape[0])
        if valid_window_len > self._padding_interval:
            window = window[-self._padding_interval :]
            valid_window_len = int(window.shape[0])
            context = valid_window_len - (end - start)
        padded = torch.zeros(
            (1, self._padding_interval, 16),
            dtype=torch.long,
            device=self._device,
        )
        padded[0, :valid_window_len].copy_(window)

        with torch.inference_mode():
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            waveform = self._model.decode(padded).float()
        audio_start = context * self._codec_upsample_rate
        audio_end = valid_window_len * self._codec_upsample_rate
        audio = (
            waveform[0, 0, audio_start:audio_end]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        self._emitted[request_id] = end
        if audio.size == 0:
            return []

        is_first = not self._audio_chunks[request_id]
        self._audio_chunks[request_id].append(audio)
        if is_first:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(audio.shape[0])},
            )
        if not self._stream_enabled.get(request_id, False):
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data=self._build_audio_payload(audio),
                metadata={"modality": "audio"},
            )
        ]

    def _build_audio_payload(self, audio: np.ndarray) -> dict[str, Any]:
        return audio_waveform_payload(
            audio,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Qwen3.5-Omni public_v1 code2wav",
        )


def create_code2wav_scheduler(
    model_path: str,
    *,
    code2wav_model_path: str | None = None,
    device: str = "cuda",
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 4,
    left_context_size: int = 25,
):
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    codec_path = resolve_code2wav_model_path(model_path, code2wav_model_path)
    torch_dtype = getattr(torch, dtype) if dtype else torch.bfloat16
    model = load_qwen3_omni_next_dac_from_config(
        codec_path / "config.yaml",
        codec_path / "model_weights.pt",
        device=device,
        dtype=torch_dtype,
    )
    logger.info("Loaded Qwen3.5 code2wav from %s", codec_path)
    return Qwen35Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
    )
