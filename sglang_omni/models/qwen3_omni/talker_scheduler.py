# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import deque
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)
_READY_SUBSET_MIN_SIZE_ENV = "SGLANG_OMNI_TALKER_READY_SUBSET_MIN_SIZE"
_READY_SUBSET_STATS_ENV = "SGLANG_OMNI_TALKER_READY_SUBSET_STATS"
_READY_SUBSET_STATS_LOG_INTERVAL_NS = 1_000_000_000


def _ready_subset_min_size() -> int:
    raw = os.getenv(_READY_SUBSET_MIN_SIZE_ENV)
    if raw is None or raw == "":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", _READY_SUBSET_MIN_SIZE_ENV, raw)
        return 0


def _ready_subset_stats_enabled() -> bool:
    return os.getenv(_READY_SUBSET_STATS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configure_talker_server_args(
    server_args: Any,
    *,
    feedback_enabled: bool = True,
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were originally requested so the caller can
    re-enable graph capture after the model worker is constructed.
    """

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if feedback_enabled:
        server_args.disable_overlap_schedule = True
        if want_cuda_graph:
            server_args.disable_cuda_graph = True
    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = 0
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    def __init__(
        self,
        *args: Any,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
                f"got {partial_start_min_chunks}"
            )
        self._enable_partial_start = bool(enable_partial_start)
        self._partial_start_min_chunks = int(partial_start_min_chunks)
        self._im_end_token_id = im_end_token_id
        self._ready_subset_deferred_batch: Any | None = None
        self._ready_subset_stats = self._new_ready_subset_stats()
        self._ready_subset_stats_last_log_ns = 0

    def _count_usable_prefetched_chunks(self, prefetched: list[Any]) -> int:
        im_end = self._im_end_token_id
        if im_end is None or not prefetched:
            return len(prefetched)
        metadata = getattr(prefetched[-1], "metadata", None) or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == int(im_end):
            return len(prefetched) - 1
        return len(prefetched)

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        if pending_stream_done:
            return True
        if not self._enable_partial_start:
            return False
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return (
            self._count_usable_prefetched_chunks(prefetched)
            >= self._partial_start_min_chunks
        )

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        del req_data, payload
        return None

    def _should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: Any
    ) -> bool:
        del request_id, chunk
        return self._enable_partial_start

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and self._model_runner is not None
            and hasattr(self._model_runner, "is_decode_batch_ready")
            and not self._model_runner.is_decode_batch_ready(batch)
        ):
            logger.debug(
                "Deferring decode batch until talker feedback/text input is ready"
            )
            return False
        return True

    def _restore_ready_subset_deferred_batch(self) -> None:
        deferred = getattr(self, "_ready_subset_deferred_batch", None)
        if deferred is None or deferred.is_empty():
            self._ready_subset_deferred_batch = None
            return
        restored_reqs = len(deferred.reqs)
        if self.running_batch is None or self.running_batch.is_empty():
            self.running_batch = deferred
        else:
            self.running_batch.merge_batch(deferred)
            self.running_batch.batch_is_full = False
        self._record_ready_subset_restore(restored_reqs)
        self._ready_subset_deferred_batch = None

    def recv_requests(self) -> list[Any]:
        self._restore_ready_subset_deferred_batch()
        return super().recv_requests()

    def get_next_batch_to_run(self) -> Any | None:
        self._restore_ready_subset_deferred_batch()
        batch = _Upstream.get_next_batch_to_run(self)
        if batch is not None and not self._is_batch_ready_to_run(batch):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        min_size = _ready_subset_min_size()
        if min_size <= 0 or not self._ready_subset_supported(batch):
            return _Upstream.update_running_batch(self, batch)

        batch.filter_batch(v1_spec_info_filtered=True)
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        split = self._split_ready_decode_batch(batch, min_size=min_size)
        if split is None:
            return _Upstream.update_running_batch(self, batch)

        ready_batch, deferred_batch = split
        if self._ready_subset_deferred_batch is None:
            self._ready_subset_deferred_batch = deferred_batch
        else:
            self._ready_subset_deferred_batch.merge_batch(deferred_batch)
            self._ready_subset_deferred_batch.batch_is_full = False
        return _Upstream.update_running_batch(self, ready_batch)

    def _ready_subset_supported(self, batch: Any) -> bool:
        if batch is None or not batch.forward_mode.is_decode():
            return False
        if getattr(batch, "enable_overlap", False):
            return False
        spec_algorithm = getattr(batch, "spec_algorithm", None)
        if spec_algorithm is not None and not spec_algorithm.is_none():
            return False
        model_config = getattr(batch, "model_config", None)
        if bool(getattr(model_config, "is_encoder_decoder", False)):
            return False
        model_runner = getattr(self, "_model_runner", None)
        return model_runner is not None and hasattr(
            model_runner,
            "_decode_input_readiness",
        )

    def _split_ready_decode_batch(
        self,
        batch: ScheduleBatch,
        *,
        min_size: int,
    ) -> tuple[ScheduleBatch, ScheduleBatch] | None:
        model_runner = self._model_runner
        readiness = [
            model_runner._decode_input_readiness(getattr(req, "_omni_data", None))[0]
            for req in batch.reqs
        ]
        ready_indices = [i for i, ready in enumerate(readiness) if ready]
        if len(ready_indices) == len(batch.reqs):
            return None
        if len(ready_indices) < min_size:
            return None
        deferred_indices = [i for i, ready in enumerate(readiness) if not ready]
        if not deferred_indices:
            return None
        ready_batch = self._make_decode_batch_subset(batch, ready_indices)
        deferred_batch = self._make_decode_batch_subset(batch, deferred_indices)
        self._record_ready_subset_split(
            input_size=len(batch.reqs),
            ready_size=len(ready_batch.reqs),
            deferred_size=len(deferred_batch.reqs),
        )
        logger.debug(
            "Running talker ready decode subset: ready=%d deferred=%d",
            len(ready_batch.reqs),
            len(deferred_batch.reqs),
        )
        return ready_batch, deferred_batch

    @staticmethod
    def _new_ready_subset_stats() -> dict[str, Any]:
        return {
            "split_batches": 0,
            "ready_reqs": 0,
            "deferred_reqs": 0,
            "input_size_hist": {},
            "ready_size_hist": {},
            "deferred_size_hist": {},
            "restored_batches": 0,
            "restored_reqs": 0,
        }

    def _record_ready_subset_split(
        self,
        *,
        input_size: int,
        ready_size: int,
        deferred_size: int,
    ) -> None:
        if not _ready_subset_stats_enabled():
            return
        stats = getattr(self, "_ready_subset_stats", None)
        if stats is None:
            stats = self._new_ready_subset_stats()
            self._ready_subset_stats = stats
        stats["split_batches"] += 1
        stats["ready_reqs"] += ready_size
        stats["deferred_reqs"] += deferred_size
        self._inc_ready_subset_hist(stats["input_size_hist"], input_size)
        self._inc_ready_subset_hist(stats["ready_size_hist"], ready_size)
        self._inc_ready_subset_hist(stats["deferred_size_hist"], deferred_size)
        self._maybe_log_ready_subset_stats()

    def _record_ready_subset_restore(self, restored_reqs: int) -> None:
        if not _ready_subset_stats_enabled():
            return
        stats = getattr(self, "_ready_subset_stats", None)
        if stats is None:
            stats = self._new_ready_subset_stats()
            self._ready_subset_stats = stats
        stats["restored_batches"] += 1
        stats["restored_reqs"] += restored_reqs
        self._maybe_log_ready_subset_stats()

    @staticmethod
    def _inc_ready_subset_hist(hist: dict[int, int], value: int) -> None:
        hist[value] = hist.get(value, 0) + 1

    def _maybe_log_ready_subset_stats(self) -> None:
        now_ns = time.monotonic_ns()
        last_log_ns = getattr(self, "_ready_subset_stats_last_log_ns", 0)
        if now_ns - last_log_ns < _READY_SUBSET_STATS_LOG_INTERVAL_NS:
            return
        self._ready_subset_stats_last_log_ns = now_ns
        stats = getattr(self, "_ready_subset_stats", None)
        if not stats:
            return
        logger.info(
            "talker_ready_subset_stats split_batches=%d ready_reqs=%d "
            "deferred_reqs=%d input_size_hist=%s ready_size_hist=%s "
            "deferred_size_hist=%s restored_batches=%d restored_reqs=%d",
            stats["split_batches"],
            stats["ready_reqs"],
            stats["deferred_reqs"],
            stats["input_size_hist"],
            stats["ready_size_hist"],
            stats["deferred_size_hist"],
            stats["restored_batches"],
            stats["restored_reqs"],
        )

    def _make_decode_batch_subset(
        self,
        batch: ScheduleBatch,
        keep_indices: list[int],
    ) -> ScheduleBatch:
        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.long,
            device=batch.req_pool_indices.device,
        )
        # Reuse upstream ScheduleBatch.filter_batch so every scheduler tensor stays
        # aligned with the selected requests. Hand-constructing ScheduleBatch here
        # is fragile because upstream adds fields over time.
        subset = copy.copy(batch)
        subset.sampling_info = SamplingBatchInfo.from_schedule_batch(
            subset,
            subset.model_config.vocab_size,
        )
        subset.filter_batch(
            keep_indices=keep_indices,
            v1_spec_info_filtered=True,
        )
        if subset.output_ids is None:
            subset.output_ids = self._subset_output_ids(
                batch,
                keep_indices,
                keep_indices_device,
            )
        subset.input_ids = subset.output_ids
        subset.out_cache_loc = None
        subset.batch_is_full = False
        return subset

    @staticmethod
    def _subset_output_ids(
        batch: ScheduleBatch,
        keep_indices: list[int],
        keep_indices_device: torch.Tensor,
    ) -> torch.Tensor:
        if batch.output_ids is not None:
            return batch.output_ids[keep_indices_device]
        token_ids: list[int] = []
        for req in (batch.reqs[i] for i in keep_indices):
            if req.output_ids:
                token_ids.append(int(req.output_ids[-1]))
            else:
                token_ids.append(int(req.origin_input_ids[-1]))
        return torch.tensor(token_ids, dtype=torch.long, device=batch.device)

    def _rollback_decode_prep_after_skip(self, batch: Any) -> None:
        # Note(Chenchen Hong, Xuesong): This is talker-only. It does not fully
        # invert prepare_for_decode; talker disables overlap/spec/Mamba/hisparse,
        # and its SamplingParams defaults keep the upstream penalizer branch
        # inactive. Also zero the req_to_token_pool cell that alloc_for_decode
        # wrote at (req_pool_indices, pre-increment seq_lens).
        if not batch.forward_mode.is_decode():
            return
        if not isinstance(batch.seq_lens_sum, int):
            raise TypeError(
                f"seq_lens_sum is {type(batch.seq_lens_sum).__name__}, expected int; "
                "sglang upstream prepare_for_decode changed; update rollback."
            )
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        if batch.output_ids is None:
            batch.output_ids = batch.input_ids
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.seq_lens_sum -= len(batch.reqs)
        batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        super().self_check_during_idle()

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        self._stream_done_handler(req_data)
