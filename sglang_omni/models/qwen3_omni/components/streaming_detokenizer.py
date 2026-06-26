# SPDX-License-Identifier: Apache-2.0
"""Streaming detokenizer scheduler for the Qwen3-Omni decode stage.

Replaces the one-shot SimpleScheduler-based decode. Consumes ``stream_chunk``
IncomingMessages from the thinker (each carrying one or more token ids),
incrementally detokenizes via HF tokenizer with UTF-8 boundary safety, and
emits text deltas as ``OutgoingMessage(type="stream", target=None)`` which the
stage runtime forwards to the Coordinator. Final result is emitted on
``new_request`` (the thinker's terminal payload via ``next``), preserving the
existing non-streaming result shape.
"""
from __future__ import annotations

import logging
import os
import queue as _queue_mod
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

from sglang_omni.models.qwen3_omni.merge import (
    decode_events,
    is_decodable_token_id,
    safe_decode_token_ids,
)
from sglang_omni.models.qwen3_omni.payload_types import (
    Qwen3OmniEvent,
    Qwen3OmniPipelineState,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

THINKER_STAGE = "thinker"

# Cap on orphan stream_done entries (zero-token race + late-done leak).
# When exceeded, evict oldest first down to _DONE_SEEN_EVICT_TO.
_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000
_PRIORITY_FIRST_STREAM_ENV = "SGLANG_OMNI_DECODE_PRIORITY_FIRST_STREAM"
_PRIORITY_STREAM_OUTBOX_ENV = "SGLANG_OMNI_DECODE_PRIORITY_STREAM_OUTBOX"


def _event_to_dict(event: Qwen3OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


@dataclass
class _RequestState:
    pending_tokens: list[int] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False
    skipped_token_count: int = 0
    stream_token_count: int = 0
    empty_delta_count: int = 0
    utf8_hold_count: int = 0
    emitted_text_count: int = 0
    emitted_text: str = ""


def _coerce_stream_token_ids(data: Any) -> list[int]:
    if isinstance(data, (list, tuple)):
        return [int(item) for item in data]
    if hasattr(data, "detach") and hasattr(data, "reshape"):
        values = data.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in values]
    if hasattr(data, "tolist") and not hasattr(data, "item"):
        values = data.tolist()
        if isinstance(values, list):
            return [int(item) for item in values]
    if hasattr(data, "item"):
        return [int(data.item())]
    return [int(data)]


class _PriorityFirstStreamInbox:
    """Queue wrapper that keeps text streams ahead of pre-run terminal payloads."""

    def __init__(self):
        self._queue: _queue_mod.PriorityQueue[tuple[int, int, IncomingMessage]] = (
            _queue_mod.PriorityQueue()
        )
        self._seq = 0

    @staticmethod
    def _priority(msg: IncomingMessage) -> int:
        if msg.type == "stream_chunk":
            item = msg.data
            return 0 if getattr(item, "chunk_id", None) == 0 else 1
        if msg.type == "stream_done":
            return 1
        if _is_streaming_actual_payload(msg):
            return 1
        return 3

    def put(
        self,
        item: IncomingMessage,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        self._seq += 1
        self._queue.put((self._priority(item), self._seq, item), block, timeout)

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> IncomingMessage:
        return self._queue.get(block=block, timeout=timeout)[2]

    def get_nowait(self) -> IncomingMessage:
        return self.get(block=False)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


def _is_streaming_actual_payload(msg: IncomingMessage) -> bool:
    if msg.type != "new_request":
        return False
    payload = msg.data
    request = getattr(payload, "request", None)
    params = getattr(request, "params", None)
    if not bool((params or {}).get("stream", False)):
        return False
    metadata = getattr(request, "metadata", None)
    return not (isinstance(metadata, dict) and bool(metadata.get("pre_run")))


class _PriorityStreamOutbox:
    """Outbox wrapper that prevents completed pre-run results blocking TTFT.

    The RTC benchmark sends many non-streaming pre-run requests before the
    measured streaming request. Under burst load, those pre-run terminal
    results can fill the decode scheduler outbox and delay the first visible
    text chunk for measured requests by seconds. Give a request's first text
    delta the highest priority, keep later stream deltas ahead of terminal
    results, and preserve FIFO order within each priority class.
    """

    def __init__(self):
        self._queue: _queue_mod.PriorityQueue[tuple[int, int, OutgoingMessage]] = (
            _queue_mod.PriorityQueue()
        )
        self._seq = 0
        self._first_stream_seen: set[str] = set()
        self._lock = threading.Lock()

    def _priority(self, msg: OutgoingMessage) -> int:
        if msg.type == "error":
            return 0
        if msg.type == "stream":
            if msg.request_id not in self._first_stream_seen:
                self._first_stream_seen.add(msg.request_id)
                return 0
            return 1
        return 2

    def put(
        self,
        item: OutgoingMessage,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        with self._lock:
            self._seq += 1
            entry = (self._priority(item), self._seq, item)
        self._queue.put(entry, block, timeout)

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> OutgoingMessage:
        item = self._queue.get(block=block, timeout=timeout)[2]
        if item.type in {"result", "error"}:
            with self._lock:
                self._first_stream_seen.discard(item.request_id)
        return item

    def get_nowait(self) -> OutgoingMessage:
        return self.get(block=False)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


def _priority_first_stream_enabled() -> bool:
    raw = os.getenv(_PRIORITY_FIRST_STREAM_ENV)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _priority_stream_outbox_enabled() -> bool:
    raw = os.getenv(_PRIORITY_STREAM_OUTBOX_ENV)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class StreamingDetokenizeScheduler:
    """Stream-aware decode stage."""

    def __init__(
        self,
        tokenizer: Any,
        eos_token_id: int | None,
        *,
        stage_name: str = "decode",
    ):
        self.inbox = (
            _PriorityFirstStreamInbox()
            if _priority_first_stream_enabled()
            else _queue_mod.Queue()
        )
        self.outbox = (
            _PriorityStreamOutbox()
            if _priority_stream_outbox_enabled()
            else _queue_mod.Queue()
        )
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self.stage_name = stage_name
        self._running = False
        self._state: dict[str, _RequestState] = {}
        self._done_seen: OrderedDict[str, None] = OrderedDict()
        self._first_stream_chunk_dequeue_seen: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue

            # Per-request failure isolation: a malformed payload, tokenizer
            # edge case, or Qwen3OmniPipelineState/decode_events bug must fail only
            # the offending request — letting the exception escape `start()`
            # trips `Stage._handle_scheduler_crash`, which fails every
            # active request on the decode stage. Mirrors the
            # SimpleScheduler / FishScheduler / Code2WavScheduler contract.
            try:
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    if msg.request_id not in self._first_stream_chunk_dequeue_seen:
                        self._first_stream_chunk_dequeue_seen.add(msg.request_id)
                        _emit_event(
                            request_id=msg.request_id,
                            stage=self.stage_name,
                            event_name="scheduler_first_stream_chunk_dequeued",
                            metadata={
                                "chunk_id": getattr(msg.data, "chunk_id", None),
                                "token_id": (
                                    msg.data.metadata.get("token_id")
                                    if isinstance(
                                        getattr(msg.data, "metadata", None), dict
                                    )
                                    else None
                                ),
                                "inbox_qsize": (
                                    self.inbox.qsize()
                                    if hasattr(self.inbox, "qsize")
                                    else None
                                ),
                            },
                        )
                    self._on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_stream_done(msg.request_id)
            except Exception as exc:
                logger.exception(
                    "StreamingDetokenizeScheduler failed request %s",
                    msg.request_id,
                )
                self.abort(msg.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        self._first_stream_chunk_dequeue_seen.discard(request_id)

    def _ensure_state(self, request_id: str) -> _RequestState:
        s = self._state.get(request_id)
        if s is None:
            s = _RequestState()
            self._state[request_id] = s
        return s

    def _emit_text_delta(
        self,
        request_id: str,
        s: _RequestState,
        text: str,
        *,
        token_id: int | None,
    ) -> None:
        if not text:
            return
        if s.emitted_text_count == 0:
            _emit_event(
                request_id=request_id,
                stage=self.stage_name,
                event_name="scheduler_first_text_delta_built",
                metadata={
                    "token_id": token_id,
                    "stream_token_count": s.stream_token_count,
                    "text_len": len(text),
                    "text_preview": text[:32],
                },
            )
        s.emitted_text_count += 1
        s.emitted_text += text
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,  # terminal stream → Coordinator
                data={
                    "text": text,
                    "modality": "text",
                    "stage_name": self.stage_name,
                },
                metadata={"modality": "text"},
            )
        )

    def _on_stream_chunk(self, request_id: str, item: Any) -> None:
        s = self._ensure_state(request_id)
        text_parts: list[str] = []
        last_text_token_id: int | None = None
        for token_id in _coerce_stream_token_ids(item.data):
            s.stream_token_count += 1
            if not is_decodable_token_id(self._tokenizer, token_id):
                s.skipped_token_count += 1
                if s.skipped_token_count == 1:
                    _emit_event(
                        request_id=request_id,
                        stage=self.stage_name,
                        event_name="scheduler_stream_token_skipped",
                        metadata={
                            "token_id": token_id,
                            "stream_token_count": s.stream_token_count,
                        },
                    )
                    logger.warning(
                        "Skipping non-text token id %s in streaming decode for request %s",
                        token_id,
                        request_id,
                    )
                continue
            s.pending_tokens.append(token_id)

            candidate = safe_decode_token_ids(
                self._tokenizer,
                s.pending_tokens,
                skip_special_tokens=True,
                request_id=request_id,
                context="streaming_delta",
            )
            # Incomplete multi-byte UTF-8 surfaces as U+FFFD; hold pending
            # until the next token completes the byte sequence.
            if "�" in candidate:
                s.utf8_hold_count += 1
                if s.utf8_hold_count == 1:
                    _emit_event(
                        request_id=request_id,
                        stage=self.stage_name,
                        event_name="scheduler_stream_utf8_hold",
                        metadata={
                            "token_id": token_id,
                            "pending_len": len(s.pending_tokens),
                            "stream_token_count": s.stream_token_count,
                        },
                    )
                continue

            s.pending_tokens.clear()
            if not candidate:
                s.empty_delta_count += 1
                if s.empty_delta_count == 1:
                    _emit_event(
                        request_id=request_id,
                        stage=self.stage_name,
                        event_name="scheduler_stream_empty_delta",
                        metadata={
                            "token_id": token_id,
                            "stream_token_count": s.stream_token_count,
                        },
                    )
                continue  # special tokens suppressed; nothing to emit

            text_parts.append(candidate)
            last_text_token_id = token_id

        if text_parts:
            self._emit_text_delta(
                request_id,
                s,
                "".join(text_parts),
                token_id=last_text_token_id,
            )

    def _on_stream_done(self, request_id: str) -> None:
        # No state row means either zero-token generation (no chunk created
        # state) or a late duplicate done after _finalize. Latch both;
        # _on_new_request consumes the zero-token case, the FIFO cap below
        # evicts duplicates.
        s = self._state.get(request_id)
        if s is None:
            self._done_seen[request_id] = None
            if len(self._done_seen) > _DONE_SEEN_MAX:
                for _ in range(len(self._done_seen) - _DONE_SEEN_EVICT_TO):
                    self._done_seen.popitem(last=False)
            return
        s.done = True
        if s.payload is not None:
            self._finalize(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        s = self._ensure_state(request_id)
        s.payload = payload
        if request_id in self._done_seen:
            s.done = True
            self._done_seen.pop(request_id, None)
        is_streaming = bool((payload.request.params or {}).get("stream", False))
        if s.done or not is_streaming:
            self._finalize(request_id)

    def _final_output_text(
        self,
        request_id: str,
        payload: StagePayload,
    ) -> str | None:
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            return None
        output_ids = thinker_out.get("output_ids")
        if not isinstance(output_ids, list) or not output_ids:
            return None
        return safe_decode_token_ids(
            self._tokenizer,
            output_ids,
            skip_special_tokens=True,
            request_id=request_id,
            context="streaming_final_suffix",
        )

    def _emit_unstreamed_final_suffix(
        self,
        request_id: str,
        s: _RequestState,
    ) -> None:
        if s.payload is None:
            return
        final_text = self._final_output_text(request_id, s.payload)
        if not final_text or final_text == s.emitted_text:
            return
        if final_text.startswith(s.emitted_text):
            self._emit_text_delta(
                request_id,
                s,
                final_text[len(s.emitted_text) :],
                token_id=None,
            )
            return

        _emit_event(
            request_id=request_id,
            stage=self.stage_name,
            event_name="scheduler_stream_suffix_mismatch",
            metadata={
                "emitted_text_len": len(s.emitted_text),
                "final_text_len": len(final_text),
                "emitted_preview": s.emitted_text[-32:],
                "final_preview": final_text[:32],
            },
        )
        logger.warning(
            "Skipping streaming final suffix for request %s because final text "
            "does not extend already emitted text",
            request_id,
        )

    def _finalize(self, request_id: str) -> None:
        s = self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        self._first_stream_chunk_dequeue_seen.discard(request_id)
        if s is None or s.payload is None:
            return
        # Flush leftover pending — UTF-8 may be truncated mid-char (e.g. on
        # max_tokens); without this the streaming client misses trailing
        # bytes that non-streaming clients still see in the final result.
        if s.pending_tokens:
            leftover = safe_decode_token_ids(
                self._tokenizer,
                s.pending_tokens,
                skip_special_tokens=True,
                request_id=request_id,
                context="streaming_finalize",
            )
            if leftover:
                self._emit_text_delta(request_id, s, leftover, token_id=None)
        is_streaming = bool((s.payload.request.params or {}).get("stream", False))
        if is_streaming:
            self._emit_unstreamed_final_suffix(request_id, s)
        result = self._build_result(s.payload, is_streaming=is_streaming)
        s.payload.data = result
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=s.payload,
            )
        )

    def _build_result(
        self, payload: StagePayload, *, is_streaming: bool = False
    ) -> dict[str, Any]:
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=self._tokenizer,
                eos_token_id=self._eos_token_id,
                step=step,
            )
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        final_event = next(
            (
                e
                for e in reversed(events)
                if e.is_final or e.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        # Streaming clients already received the full output as per-token
        # text deltas via OutgoingMessage(type="stream"). The terminal
        # result must NOT carry the reconstructed full text — direct
        # consumers of Client.completion_stream() append every chunk's
        # "text" field and would otherwise emit the whole response twice.
        # Mirrors the code2wav slim-final contract for audio.
        if is_streaming:
            result.pop("text", None)
        elif "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if isinstance(output_ids, list) and output_ids:
                result["text"] = safe_decode_token_ids(
                    self._tokenizer,
                    output_ids,
                    skip_special_tokens=True,
                    request_id=payload.request_id,
                    context="non_streaming_final",
                )
                result.setdefault("modality", "text")

        finish_reason = thinker_out.get("finish_reason")
        if finish_reason is not None:
            result.setdefault("finish_reason", finish_reason)

        prompt_tokens_value = (
            state.prompt.get("prompt_tokens") if isinstance(state.prompt, dict) else None
        )
        if prompt_tokens_value is not None:
            prompt_tokens = int(prompt_tokens_value)
        else:
            input_ids = (
                state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
            )
            if input_ids is None:
                prompt_tokens = 0
            elif hasattr(input_ids, "numel"):
                prompt_tokens = int(input_ids.numel())
            else:
                prompt_tokens = len(input_ids)
        if prompt_tokens < 0:
            prompt_tokens = 0

        completion_ids = thinker_out.get("output_ids") or []
        completion_tokens = len(completion_ids)

        result.setdefault(
            "usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

        return result


def create_streaming_detokenize_scheduler(
    model_path: str,
    *,
    stage_name: str = "decode",
) -> StreamingDetokenizeScheduler:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return StreamingDetokenizeScheduler(
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        stage_name=stage_name,
    )
