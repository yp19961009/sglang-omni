# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 talker scheduler with 4-text/4-codec streaming interleave."""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler

logger = logging.getLogger(__name__)


class Qwen35TalkerScheduler(QwenTalkerScheduler):
    """Insert text chunks into a live talker KV stream between codec groups.

    Qwen3.5 consumes four text embeddings and emits four codec frames. It then
    consumes the fourth codec feedback in one extra boundary step, drops that
    candidate output, and replaces the dropped candidate with the next text
    group. This matches vLLM's external-data scheduling contract.
    """

    def __init__(
        self,
        *args: Any,
        text_chunk_size: int = 4,
        codec_chunk_size: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text_chunk_size = int(text_chunk_size)
        self._codec_chunk_size = int(codec_chunk_size)
        if self._text_chunk_size <= 0 or self._codec_chunk_size <= 0:
            raise ValueError("Qwen3.5 interleave chunk sizes must be positive")
        self._parked_interleaved_reqs: dict[str, Any] = {}

    def get_next_batch_to_run(self) -> Any | None:
        # Call upstream directly: QwenTalkerScheduler's legacy readiness gate
        # assumes one text row is consumed on every codec decode step.
        batch = _Upstream.get_next_batch_to_run(self)
        parked_ready = self._ready_parked_requests()

        if batch is None:
            return self._build_text_extend_batch(parked_ready) if parked_ready else None
        if not batch.forward_mode.is_decode():
            return batch

        decode_ready: list[Any] = []
        park_now: list[Any] = []
        extend_now: list[Any] = []
        for req in list(batch.reqs):
            data = getattr(req, "_omni_data", None)
            if self._needs_text_extend(data):
                park_now.append(req)
                if self._text_chunk_ready(data):
                    extend_now.append(req)
            elif self._model_runner.is_decode_batch_ready(
                self._single_req_batch_view(batch, req)
            ):
                decode_ready.append(req)
            else:
                park_now.append(req)

        if not park_now and not parked_ready:
            self._mark_boundary_steps(batch)
            return batch

        self._rollback_decode_prep_after_skip(batch)
        self._park_running_requests(batch, park_now)

        extend_reqs = parked_ready + extend_now
        if extend_reqs:
            return self._build_text_extend_batch(extend_reqs)

        if not decode_ready:
            return None
        batch = _Upstream.update_running_batch(self, self.running_batch)
        self._mark_boundary_steps(batch)
        return batch

    @staticmethod
    def _single_req_batch_view(batch: Any, req: Any) -> Any:
        return type(
            "TalkerDecodeView",
            (),
            {
                "forward_mode": batch.forward_mode,
                "reqs": [req],
            },
        )()

    @staticmethod
    def _needs_text_extend(data: Any) -> bool:
        if data is None or bool(getattr(data, "interleaved_final", False)):
            return False
        chunk_size = int(getattr(data, "interleaved_codec_chunk_size", 0) or 0)
        return (
            chunk_size > 0
            and int(getattr(data, "interleaved_codec_steps", 0)) >= chunk_size
            and bool(getattr(data, "interleaved_boundary_ready", False))
        )

    @staticmethod
    def _awaiting_boundary_step(data: Any) -> bool:
        if data is None or bool(getattr(data, "interleaved_final", False)):
            return False
        chunk_size = int(getattr(data, "interleaved_codec_chunk_size", 0) or 0)
        return (
            chunk_size > 0
            and int(getattr(data, "interleaved_codec_steps", 0)) >= chunk_size
            and not bool(getattr(data, "interleaved_boundary_ready", False))
        )

    def _mark_boundary_steps(self, batch: Any | None) -> None:
        if batch is None:
            return
        for req in batch.reqs:
            data = getattr(req, "_omni_data", None)
            if not self._awaiting_boundary_step(data):
                continue
            if bool(getattr(data, "interleaved_drop_next_output", False)):
                continue
            data.interleaved_drop_next_output = True
            # The sampled boundary candidate is a logical output token but not
            # an emitted codec frame, so exclude it from the user's codec limit.
            req.sampling_params.max_new_tokens += 1

    @staticmethod
    def _text_chunk_ready(data: Any) -> bool:
        if not Qwen35TalkerScheduler._needs_text_extend(data):
            return False
        queue = getattr(data, "pending_text_queue", None)
        available = len(queue) if queue is not None else 0
        chunk_size = int(getattr(data, "interleaved_text_chunk_size", 0) or 0)
        return available >= chunk_size or (
            bool(getattr(data, "thinker_chunks_done", False)) and available > 0
        )

    def _ready_parked_requests(self) -> list[Any]:
        return [
            req
            for req in self._parked_interleaved_reqs.values()
            if self._text_chunk_ready(getattr(req, "_omni_data", None))
        ]

    def _park_running_requests(self, batch: Any, reqs: list[Any]) -> None:
        if not reqs:
            return
        parked_ids = {id(req) for req in reqs}
        for req in reqs:
            self._parked_interleaved_reqs[req.rid] = req
        keep = [idx for idx, req in enumerate(batch.reqs) if id(req) not in parked_ids]
        batch.filter_batch(keep_indices=keep)
        batch.batch_is_full = False
        self.running_batch = batch

    def _build_text_extend_batch(self, reqs: list[Any]) -> ScheduleBatch:
        if not reqs:
            raise ValueError("text extend batch requires at least one request")
        for req in reqs:
            self._parked_interleaved_reqs.pop(req.rid, None)
            self._prepare_text_extend(req)

        batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        batch.prepare_for_extend()
        batch.decoding_reqs = None
        return batch

    def _prepare_text_extend(self, req: Any) -> None:
        data = req._omni_data
        rows, is_final = self._take_text_rows(data)
        prefix_len = int(req.kv_committed_len)
        logical_len = len(req.origin_input_ids) + len(req.output_ids)
        if logical_len != prefix_len + 1:
            raise RuntimeError(
                f"Qwen3.5 interleave sequence drift for {req.rid}: "
                f"logical_len={logical_len}, committed={prefix_len}"
            )

        sequence = data.prefill_input_embeds
        if not isinstance(sequence, torch.Tensor) or sequence.ndim != 2:
            raise RuntimeError("Qwen3.5 interleave requires a 2D input history")
        if int(sequence.shape[0]) != prefix_len:
            raise RuntimeError(
                f"Qwen3.5 interleave embedding history drift for {req.rid}: "
                f"rows={sequence.shape[0]}, committed={prefix_len}"
            )
        rows = rows.to(device=sequence.device, dtype=sequence.dtype)
        data.prefill_input_embeds = torch.cat([sequence, rows], dim=0)

        feedback_queue = data.pending_feedback_queue
        if len(feedback_queue) != 1:
            raise RuntimeError(
                f"Qwen3.5 interleave expected one boundary feedback row for "
                f"{req.rid}, got {len(feedback_queue)}"
            )
        feedback_queue.popleft()

        placeholder_count = int(rows.shape[0]) - 1
        if placeholder_count > 0:
            placeholder_id = int(req.origin_input_ids[-1])
            req.output_ids.extend([placeholder_id] * placeholder_count)
            req.sampling_params.max_new_tokens += placeholder_count
            data.interleaved_placeholder_count += placeholder_count

        end = prefix_len + int(rows.shape[0])
        req.fill_ids = req.origin_input_ids + req.output_ids
        if len(req.fill_ids) != end:
            raise RuntimeError(
                f"Qwen3.5 interleave fill length drift for {req.rid}: "
                f"fill={len(req.fill_ids)}, expected={end}"
            )
        old_req_pool_idx = req.req_pool_idx
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            old_req_pool_idx, :prefix_len
        ].to(dtype=torch.int64, copy=True)
        if getattr(self.tp_worker.model_runner, "mambaish_config", None) is not None:
            self.req_to_token_pool.free(
                old_req_pool_idx,
                free_mamba_cache=False,
            )
        else:
            self.req_to_token_pool.free(old_req_pool_idx)
        req.req_pool_idx = None
        req.set_extend_input_len(int(rows.shape[0]))

        data.interleaved_codec_steps = 0
        data.interleaved_boundary_ready = False
        data.interleaved_final = bool(is_final)
        data.feedback_only_decode = bool(is_final)
        logger.debug(
            "qwen35_talker_text_extend request_id=%s rows=%d final=%s "
            "prefix_len=%d codec_frames=%d",
            req.rid,
            int(rows.shape[0]),
            is_final,
            prefix_len,
            int(data.codec_generation_steps),
        )

    @staticmethod
    def _take_text_rows(data: Any) -> tuple[torch.Tensor, bool]:
        queue = data.pending_text_queue
        available = len(queue)
        chunk_size = int(data.interleaved_text_chunk_size)
        if available >= chunk_size:
            count = chunk_size
        elif data.thinker_chunks_done and available > 0:
            count = available
        else:
            raise RuntimeError(
                "Qwen3.5 interleave text extend was scheduled before a chunk was ready"
            )
        rows = torch.stack([queue.popleft() for _ in range(count)], dim=0)
        return rows, bool(data.thinker_chunks_done and len(queue) == 0)

    def _find_request_data(self, request_id: str) -> Any | None:
        data = super()._find_request_data(request_id)
        if data is not None:
            return data
        req = self._parked_interleaved_reqs.get(request_id)
        return getattr(req, "_omni_data", None)

    def _active_request_ids(self) -> list[str]:
        request_ids = set(super()._active_request_ids())
        request_ids.update(self._parked_interleaved_reqs)
        return sorted(request_ids)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        parked = self._parked_interleaved_reqs.pop(request_id, None)
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if parked is not None and parked.req_pool_idx is not None:
            self._release_request_kv_cache(parked)

    def self_check_during_idle(self) -> None:
        if self._parked_interleaved_reqs:
            return
        super().self_check_during_idle()
