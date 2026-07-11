# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-specific talker prompt construction and text-stream updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)


class Qwen35TalkerPrefillBuilder:
    """Build the production Qwen3.5 talker layout used by the vLLM reference."""

    def __init__(
        self,
        *,
        model: Any,
        root_config: Any,
        model_path: str,
        first_text_tokens: int = 4,
    ) -> None:
        self._model = model
        self._root_config = root_config
        self._talker_config = root_config.talker_config
        self._device = model.model.codec_embedding.weight.device
        self._dtype = model.activation_dtype
        self._first_text_tokens = max(1, int(first_text_tokens))
        self._voice_map = self._load_voice_map(model_path)
        self._special_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _load_voice_map(model_path: str) -> dict[str, str]:
        path = Path(model_path) / "voice_map.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        return {str(name).lower(): str(value) for name, value in data.items()}

    def build_prompt_prefill(
        self,
        payload: Any,
        thinker_chunks: list[Any],
        *,
        thinker_done: bool,
    ) -> dict[str, Any]:
        chunks = [
            chunk
            for chunk in thinker_chunks
            if self._chunk_token_id(chunk) != self._root_config.im_end_token_id
        ]
        if not chunks:
            raise ValueError("Qwen3.5 talker prefill requires generated text tokens")

        if thinker_done:
            first_chunks = chunks
            future_chunks = []
        else:
            first_chunks = chunks[: self._first_text_tokens]
            future_chunks = chunks[self._first_text_tokens :]
        first_ids = torch.tensor(
            [self._chunk_token_id(chunk) for chunk in first_chunks],
            dtype=torch.long,
            device=self._device,
        )

        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        voice = self._resolve_voice(params=params, metadata=metadata)
        language = self._resolve_language(params=params, metadata=metadata)
        style = self._resolve_style(params=params, metadata=metadata)

        parts = [
            self._system_part(voice),
            self._assistant_prefix(language=language, style=style),
            self._text_embeddings(first_ids),
        ]
        if thinker_done:
            # Full-buffer mode matches the HF single-chunk layout: all text and
            # tts_eos are in the prompt, then decode feeds back codec embeddings only.
            parts.append(self._tts_eos_embed())
        input_embeds = torch.cat(parts, dim=0)
        input_ids = torch.full(
            (input_embeds.shape[0],),
            int(self._root_config.tts_pad_token_id),
            dtype=torch.long,
            device=self._device,
        )

        pending = PendingTextTensorQueue()
        if future_chunks:
            future_ids = torch.tensor(
                [self._chunk_token_id(chunk) for chunk in future_chunks],
                dtype=torch.long,
                device=self._device,
            )
            pending.append_rows(self._text_embeddings(future_ids))
        return {
            "input_embeds": input_embeds,
            "input_ids": input_ids,
            "pending_text_queue": pending,
            "tts_pad_embed": self._tts_pad_embed()[0],
            "tts_eos_embed": self._tts_eos_embed()[0],
            "prompt_model_inputs": {},
            "voice": voice,
            "language": language,
            "feedback_only_decode": thinker_done,
            "interleaved_streaming": not thinker_done,
        }

    def append_text_chunk(self, req_data: Any, chunk: Any) -> None:
        if getattr(req_data, "thinker_chunks_done", False):
            return
        token_id = self._chunk_token_id(chunk)
        if token_id == self._root_config.im_end_token_id:
            return
        queue = coerce_pending_text_queue(req_data.pending_text_queue)
        req_data.pending_text_queue = queue
        token = torch.tensor([token_id], dtype=torch.long, device=self._device)
        queue.append(self._text_embeddings(token)[0])

    def mark_thinker_done(self, req_data: Any) -> None:
        if getattr(req_data, "thinker_chunks_done", False):
            return
        req_data.thinker_chunks_done = True
        queue = coerce_pending_text_queue(req_data.pending_text_queue)
        req_data.pending_text_queue = queue
        queue.append(self._tts_eos_embed()[0])

    @staticmethod
    def _chunk_token_id(chunk: Any) -> int:
        metadata = chunk.metadata or {}
        if "token_id" not in metadata:
            raise ValueError("thinker stream chunk is missing token_id")
        return int(metadata["token_id"])

    def _resolve_voice(
        self, *, params: dict[str, Any], metadata: dict[str, Any]
    ) -> str:
        audio_config = metadata.get("audio_config")
        audio_config = audio_config if isinstance(audio_config, dict) else {}
        requested = (
            params.get("speaker")
            or params.get("voice")
            or audio_config.get("voice")
            or "Ethan"
        )
        requested = str(requested)
        if requested.lower() == "default":
            requested = "Ethan"
        internal = self._voice_map.get(requested.lower(), requested)
        speaker_ids = self._talker_config.speaker_id
        if internal not in speaker_ids:
            by_lower = {str(name).lower(): str(name) for name in speaker_ids}
            internal = by_lower.get(internal.lower(), internal)
        if internal not in speaker_ids:
            raise ValueError(f"unknown Qwen3.5 voice: {requested!r}")
        return internal

    @staticmethod
    def _resolve_language(*, params: dict[str, Any], metadata: dict[str, Any]) -> str:
        audio_config = metadata.get("audio_config")
        audio_config = audio_config if isinstance(audio_config, dict) else {}
        return str(
            params.get("language") or audio_config.get("language") or "auto"
        ).lower()

    @staticmethod
    def _resolve_style(
        *, params: dict[str, Any], metadata: dict[str, Any]
    ) -> str | None:
        audio_config = metadata.get("audio_config")
        audio_config = audio_config if isinstance(audio_config, dict) else {}
        value = params.get("voice_style") or audio_config.get("style")
        return str(value).lower() if value else None

    def _system_part(self, voice: str) -> torch.Tensor:
        text_prefix = self._text_embeddings(
            torch.tensor(
                [
                    self._root_config.im_start_token_id,
                    self._root_config.system_token_id,
                    self._root_config.nl_token_id,
                ],
                dtype=torch.long,
                device=self._device,
            )
        )
        prompt_ids = self._talker_config.speaker_system_prompt_id[voice]
        speaker_prompt = self._text_embeddings(
            torch.tensor(prompt_ids, dtype=torch.long, device=self._device)
        )
        codec_bos = self._codec_embeddings(
            torch.tensor(
                [self._talker_config.codec_bos_id],
                dtype=torch.long,
                device=self._device,
            )
        )
        speaker_codes = self._speaker_embeddings(voice)
        codec_eos = self._codec_embeddings(
            torch.tensor(
                [self._talker_config.codec_eos_token_id],
                dtype=torch.long,
                device=self._device,
            )
        )
        chat_end = self._text_embeddings(
            torch.tensor(
                [self._root_config.im_end_token_id, self._root_config.nl_token_id],
                dtype=torch.long,
                device=self._device,
            )
        )
        return torch.cat(
            [
                text_prefix,
                speaker_prompt,
                codec_bos,
                speaker_codes,
                codec_eos,
                chat_end,
            ],
            dim=0,
        )

    def _assistant_prefix(self, *, language: str, style: str | None) -> torch.Tensor:
        parts = [
            self._text_embeddings(
                torch.tensor(
                    [
                        self._root_config.im_start_token_id,
                        self._root_config.assistant_token_id,
                        self._root_config.nl_token_id,
                    ],
                    dtype=torch.long,
                    device=self._device,
                )
            )
        ]
        style_mapping = getattr(
            self._root_config, "talker_assistant_prompt_id_mapping", {}
        )
        if style and style in style_mapping:
            parts.append(
                self._text_embeddings(
                    torch.tensor(
                        style_mapping[style], dtype=torch.long, device=self._device
                    )
                )
            )

        language_mapping = getattr(self._root_config, "talker_language_id", {})
        if language != "auto" and language in language_mapping:
            codec_ids = [
                self._talker_config.codec_think_id,
                self._talker_config.codec_think_bos_id,
                language_mapping[language],
                self._talker_config.codec_think_eos_id,
            ]
        else:
            codec_ids = [
                self._talker_config.codec_nothink_id,
                self._talker_config.codec_think_bos_id,
                self._talker_config.codec_think_eos_id,
            ]
        parts.extend(
            [
                self._codec_embeddings(
                    torch.tensor(codec_ids, dtype=torch.long, device=self._device)
                ),
                self._tts_bos_embed(),
                self._codec_embeddings(
                    torch.tensor(
                        [self._talker_config.codec_bos_id],
                        dtype=torch.long,
                        device=self._device,
                    )
                ),
            ]
        )
        return torch.cat(parts, dim=0)

    def _speaker_embeddings(self, voice: str) -> torch.Tensor:
        speaker_id = int(self._talker_config.speaker_id[voice])
        codes = self._model.speaker_codec_embeddings[speaker_id].transpose(0, 1)
        layer0 = self._model.get_input_embeddings()(codes[:, 0])
        residual = [
            embedding(codes[:, index + 1])
            for index, embedding in enumerate(
                self._model.code_predictor.model.codec_embedding
            )
        ]
        return torch.stack([layer0, *residual], dim=0).sum(dim=0)

    def _text_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._model.get_text_embeddings(token_ids).to(dtype=self._dtype)

    def _codec_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._model.get_input_embeddings()(token_ids).to(dtype=self._dtype)

    def _cached_text_special(self, name: str, token_id: int) -> torch.Tensor:
        value = self._special_cache.get(name)
        if value is None:
            value = self._text_embeddings(
                torch.tensor([token_id], dtype=torch.long, device=self._device)
            )
            self._special_cache[name] = value
        return value

    def _tts_bos_embed(self) -> torch.Tensor:
        return self._cached_text_special("tts_bos", self._root_config.tts_bos_token_id)

    def _tts_eos_embed(self) -> torch.Tensor:
        return self._cached_text_special("tts_eos", self._root_config.tts_eos_token_id)

    def _tts_pad_embed(self) -> torch.Tensor:
        return self._cached_text_special("tts_pad", self._root_config.tts_pad_token_id)
