# SPDX-License-Identifier: Apache-2.0
"""Stage — IO shell for pipeline processing.

Handles: control plane messaging, data plane (relay) IO, input aggregation,
stream chunk routing, abort tracking, profiling.

Dispatches all compute to scheduler (OmniScheduler or SimpleScheduler).
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue as _queue_mod
import threading
from contextlib import suppress
from typing import Any, Callable, Literal

from sglang_omni.pipeline import relay_io
from sglang_omni.pipeline.stage.input import DirectInput, InputHandler
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.pipeline.tp_control import TPLeaderFanout, TPWorkMessage
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_recorder
from sglang_omni.profiler.event_recorder import set_active_stage as _set_active_stage
from sglang_omni.profiler.torch_profiler import TorchProfiler
from sglang_omni.proto import (
    CompleteMessage,
    DataReadyMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
)
from sglang_omni.relay.base import Relay, create_relay
from sglang_omni.scheduling.messages import IncomingMessage

logger = logging.getLogger(__name__)

GetNextFn = Callable[[str, Any], str | list[str] | None]
GetStreamDoneTargetsFn = Callable[[str, Any], str | list[str] | None]
_STREAM_RELAY_COMPLETION_TIMEOUT_ENV = "SGLANG_OMNI_STREAM_RELAY_COMPLETION_TIMEOUT_S"
_STREAM_RELAY_COMPLETION_DEFAULT_TIMEOUT_S = 300.0
_DEFER_NONSTREAM_COMPLETES_ENV = "SGLANG_OMNI_DEFER_NONSTREAM_COMPLETES"
_STAGE_IO_YIELD_EVERY_MESSAGES_ENV = "SGLANG_OMNI_STAGE_IO_YIELD_EVERY_MESSAGES"
_ASYNC_STREAM_INGEST_ENV = "SGLANG_OMNI_ASYNC_STREAM_INGEST"


def _stream_relay_completion_timeout() -> float | None:
    raw = os.getenv(_STREAM_RELAY_COMPLETION_TIMEOUT_ENV)
    if raw is None or raw == "":
        return _STREAM_RELAY_COMPLETION_DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %.1fs",
            _STREAM_RELAY_COMPLETION_TIMEOUT_ENV,
            raw,
            _STREAM_RELAY_COMPLETION_DEFAULT_TIMEOUT_S,
        )
        return _STREAM_RELAY_COMPLETION_DEFAULT_TIMEOUT_S
    if value <= 0:
        return None
    return value


def _defer_nonstream_completes_enabled() -> bool:
    raw = os.getenv(_DEFER_NONSTREAM_COMPLETES_ENV)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _stage_io_yield_every_messages() -> int:
    raw = os.getenv(_STAGE_IO_YIELD_EVERY_MESSAGES_ENV)
    if raw is None or raw == "":
        return 1
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using 1",
            _STAGE_IO_YIELD_EVERY_MESSAGES_ENV,
            raw,
        )
        return 1
    return max(0, value)


def _async_stream_ingest_enabled() -> bool:
    raw = os.getenv(_ASYNC_STREAM_INGEST_ENV)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class Stage:
    """IO shell for one pipeline stage.

    All stage compute is dispatched through the scheduler inbox/outbox
    contract, independent of scheduler implementation.

    Note on ``role``: ``role="single"`` means this stage owns its own ZMQ
    control plane and relay reader (i.e. it is NOT a TP follower). It does
    **not** imply this stage has its OS process to itself — since the
    declarative topology PR, multiple ``role="single"`` stages can share
    one OS process (and one asyncio event loop). When they do, they share
    a failure domain: see ``_run_process`` in ``stage_workers.py``.
    ``role="leader"`` / ``role="follower"`` continue to denote TP rank 0
    vs rank > 0 within a multi-rank TP stage; TP stages must own their OS
    process exclusively.
    """

    def __init__(
        self,
        name: str,
        role: Literal["single", "leader", "follower"],
        get_next: GetNextFn,
        gpu_id: int | None,
        endpoints: dict[str, str],
        control_plane: Any,
        stream_endpoints: dict[str, str] | None = None,
        input_handler: InputHandler | None = None,
        relay: Relay | None = None,
        relay_config: dict[str, Any] | None = None,
        scheduler: Any = None,
        project_payload: dict[str, Callable[[Any], Any]] | None = None,
        stream_targets: list[str] | None = None,
        get_stream_done_targets: GetStreamDoneTargetsFn | None = None,
        same_gpu_targets: set[str] | None = None,
        same_gpu_payload_targets: set[str] | None = None,
        same_process_targets: set[str] | None = None,
        local_dispatcher: Any | None = None,
        can_accept_stream_before_payload: bool = False,
        tp_fanout: TPLeaderFanout | None = None,
        is_terminal: bool = False,
    ):
        self.name = name
        self.role = role
        self.get_next = get_next
        self.gpu_id = gpu_id
        self.endpoints = endpoints
        self.stream_endpoints = stream_endpoints or endpoints
        self.control_plane = control_plane
        self.input_handler = input_handler or DirectInput()
        self.scheduler = scheduler
        self._project_payload = project_payload or {}
        self._stream_targets = stream_targets or []
        self.get_stream_done_targets = get_stream_done_targets
        self._same_gpu_targets = same_gpu_targets or set()
        self._same_gpu_payload_targets = same_gpu_payload_targets or set()
        self._same_process_targets = same_process_targets or set()
        self._local_dispatcher = local_dispatcher
        self._can_accept_stream_before_payload = can_accept_stream_before_payload
        self._tp_fanout = tp_fanout
        self._is_terminal = is_terminal
        self._owns_external_io = role in {"single", "leader"}

        # --- Relay ---
        if relay is not None:
            self.relay = relay
        else:
            config = relay_config or {}
            engine_id = config.get("worker_id", f"{name}_relay")
            relay_type = config.get("relay_type", "nixl").lower()
            gpu_id = config.get("gpu_id")
            if gpu_id is not None:
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
                if relay_type == "nccl":
                    device = "cuda"
            self.relay = create_relay(
                relay_type,
                engine_id=engine_id,
                slot_size_mb=config.get("slot_size_mb", 64),
                credits=config.get("credits", 2),
                device=device,
                rank=config.get("rank"),
                world_size=config.get("world_size"),
                send_to_ranks=config.get("send_to_ranks", []),
                recv_from_ranks=config.get("recv_from_ranks", []),
            )

        # --- State ---
        self._running = False
        self._aborted: set[str] = set()
        self._active_requests: set[str] = set()
        self._stream_queue: StreamQueue | None = None
        self._stream_chunk_counters: dict[tuple[str, str], int] = {}
        # Per-request: did we already emit the first stream-chunk event?
        self._first_stream_chunk_seen: set[str] = set()
        self._first_outbox_stream_dequeue_seen: set[str] = set()
        self._first_scheduler_stream_enqueue_seen: set[str] = set()
        self._local_stream_targets: dict[str, set[str]] = {}
        self._nonlocal_stream_targets: dict[str, set[str]] = {}
        self._scheduler_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._scheduler_crash_error: BaseException | None = None
        self._background_task_error: BaseException | None = None
        self._stream_relay_completion_timeout = _stream_relay_completion_timeout()
        self._stream_relay_completion_tasks: set[asyncio.Task] = set()
        self._defer_nonstream_completes = _defer_nonstream_completes_enabled()
        self._deferred_complete_queue: asyncio.Queue[CompleteMessage] | None = None
        self._stage_io_yield_every_messages = _stage_io_yield_every_messages()
        self._async_stream_ingest = _async_stream_ingest_enabled()
        self._stream_ingest_queues: dict[
            str, asyncio.Queue[DataReadyMessage | None]
        ] = {}
        self._stream_ingest_tasks: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if self._running:
            return
        await self.control_plane.start()
        self._loop = asyncio.get_running_loop()
        if self._owns_external_io and self._defer_nonstream_completes:
            self._deferred_complete_queue = asyncio.Queue()
        self._running = True

        # Start scheduler in dedicated thread
        if self.scheduler is not None:

            def _run_scheduler():
                # Active-stage binding so ``emit(stage=None)`` from
                # scheduler-thread descendants resolves to this stage.
                _set_active_stage(self.name)
                try:
                    if self.gpu_id is not None:
                        import torch

                        torch.cuda.set_device(int(self.gpu_id))
                        logger.info(
                            "Scheduler thread for stage %s set CUDA device to %s",
                            self.name,
                            self.gpu_id,
                        )
                    self.scheduler.start()
                except Exception as exc:
                    logger.exception("Scheduler thread for stage %s crashed", self.name)
                    self._running = False
                    loop = self._loop
                    if loop is not None and not loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._handle_scheduler_crash(exc),
                            loop,
                        )

            self._scheduler_thread = threading.Thread(
                target=_run_scheduler,
                name=f"scheduler-{self.name}",
                daemon=True,
            )
            self._scheduler_thread.start()

        logger.info("Stage %s started", self.name)

    async def stop(self) -> None:
        self._running = False
        if self.scheduler is not None:
            self.scheduler.stop()
        self.control_plane.close()
        for queue in list(self._stream_ingest_queues.values()):
            queue.put_nowait(None)
        for task in list(self._stream_ingest_tasks.values()):
            task.cancel()
        if self._stream_ingest_tasks:
            stream_ingest_tasks = list(self._stream_ingest_tasks.values())
            await asyncio.gather(
                *stream_ingest_tasks,
                return_exceptions=True,
            )
            self._stream_ingest_tasks.clear()
            self._stream_ingest_queues.clear()
        for task in list(self._stream_relay_completion_tasks):
            task.cancel()
        if self._stream_relay_completion_tasks:
            await asyncio.gather(
                *self._stream_relay_completion_tasks,
                return_exceptions=True,
            )
            self._stream_relay_completion_tasks.clear()
        if self._tp_fanout is not None:
            self._tp_fanout.close()
        self.relay.close()
        logger.info("Stage %s stopped", self.name)

    async def run(self) -> None:
        await self.start()

        abort_task = asyncio.create_task(self._abort_listener())
        outbox_task = asyncio.create_task(self._drain_outbox())
        deferred_complete_task = (
            asyncio.create_task(self._drain_deferred_completes())
            if self._deferred_complete_queue is not None
            else None
        )
        abort_task.add_done_callback(
            lambda task: self._on_background_task_done(task, "abort listener")
        )
        outbox_task.add_done_callback(
            lambda task: self._on_background_task_done(task, "outbox drain")
        )
        if deferred_complete_task is not None:
            deferred_complete_task.add_done_callback(
                lambda task: self._on_background_task_done(
                    task, "deferred complete drain"
                )
            )

        try:
            handled_messages = 0
            while self._running:
                msg = await self.control_plane.recv()
                if (
                    self.role == "leader"
                    and self._tp_fanout is not None
                    and isinstance(
                        msg,
                        (ShutdownMessage, ProfilerStartMessage, ProfilerStopMessage),
                    )
                ):
                    await self._tp_fanout.fanout_control(msg)
                if isinstance(msg, ShutdownMessage):
                    break
                if isinstance(msg, TPWorkMessage):
                    await self._execute(msg.data)
                    handled_messages += 1
                    if (
                        self._stage_io_yield_every_messages > 0
                        and handled_messages
                        % self._stage_io_yield_every_messages
                        == 0
                    ):
                        await asyncio.sleep(0)
                    continue
                await self._handle_message(msg)
                handled_messages += 1
                if (
                    self._stage_io_yield_every_messages > 0
                    and handled_messages % self._stage_io_yield_every_messages == 0
                ):
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception:
            if self._scheduler_crash_error is None:
                raise
        finally:
            await self.stop()
            abort_task.cancel()
            outbox_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task
            with suppress(asyncio.CancelledError):
                await outbox_task
            if deferred_complete_task is not None:
                deferred_complete_task.cancel()
                with suppress(asyncio.CancelledError):
                    await deferred_complete_task
            if self._background_task_error is not None:
                raise self._background_task_error
            if self._scheduler_crash_error is not None:
                raise RuntimeError(
                    f"Scheduler thread for stage {self.name} crashed"
                ) from self._scheduler_crash_error

    async def _handle_message(self, msg: Any) -> None:
        if isinstance(msg, SubmitMessage):
            await self._on_submit(msg)
        elif isinstance(msg, DataReadyMessage):
            if (
                self._async_stream_ingest
                and (msg.is_done or msg.error or msg.chunk_id is not None)
            ):
                self._enqueue_stream_control_message(msg)
            elif msg.is_done or msg.error:
                await self._on_stream_signal(msg)
            elif msg.chunk_id is not None:
                await self._on_stream_chunk(msg)
            else:
                await self._on_data_ready(msg)
        elif isinstance(msg, ProfilerStartMessage):
            self._on_profiler_start(msg)
        elif isinstance(msg, ProfilerStopMessage):
            self._on_profiler_stop(msg)

    async def _on_submit(self, msg: SubmitMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_input_received",
            metadata={"from_stage": "coordinator", "kind": "submit"},
        )

        payload = msg.data  # StagePayload from coordinator
        await self._execute(payload)

    async def _on_data_ready(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            await self._discard_payload_data(msg)
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        # Read payload from relay
        try:
            if (
                isinstance(msg.shm_metadata, dict)
                and msg.shm_metadata.get("_payload_ipc")
            ):
                failure_prefix = "payload read failed"
                payload = relay_io.deserialize_ipc_payload(msg.shm_metadata)
            else:
                failure_prefix = "relay read failed"
                payload = await relay_io.read_payload(
                    self.relay, request_id, msg.shm_metadata
                )
        except Exception as exc:
            logger.exception(
                "Stage %s: %s for %s", self.name, failure_prefix, request_id
            )
            self.relay.cleanup(request_id)
            await self._send_failure(request_id, f"{failure_prefix}: {exc}")
            return

        await self._receive_payload_from_stage(request_id, msg.from_stage, payload)

    async def receive_local_payload(
        self,
        request_id: str,
        from_stage: str,
        payload: Any,
    ) -> None:
        await self._receive_payload_from_stage(request_id, from_stage, payload)

    async def receive_local_stream_chunk(
        self,
        request_id: str,
        from_stage: str,
        chunk_id: int,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        item = StreamItem(
            chunk_id=chunk_id,
            data=data,
            from_stage=from_stage,
            metadata=metadata,
        )
        self._emit_stream_chunk_received(
            request_id=request_id,
            from_stage=from_stage,
            chunk_id=chunk_id,
        )
        await self._route_stream_item_or_fail(request_id, item)

    async def receive_local_stream_signal(
        self,
        request_id: str,
        from_stage: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        await self._receive_stream_signal(
            request_id,
            from_stage,
            is_done=is_done,
            error=error,
        )

    async def _receive_payload_from_stage(
        self,
        request_id: str,
        from_stage: str,
        payload: Any,
    ) -> None:
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        if request_id in self._aborted:
            return

        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_input_received",
            metadata={"from_stage": from_stage, "kind": "payload"},
        )
        merged = self.input_handler.receive(request_id, from_stage, payload)
        if merged is not None:
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_aggregate_ready",
                metadata={"from_stage": from_stage},
            )
            await self._execute(merged)

    async def _on_stream_chunk(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            await self._discard_stream_chunk_data(msg)
            return
        self._active_requests.add(request_id)

        # Tiny CPU-only chunks, such as text token ids for the decode stage, can
        # ride in the control message and avoid the relay data path entirely.
        if isinstance(msg.shm_metadata, dict) and msg.shm_metadata.get("_inline_cpu"):
            try:
                data, metadata = relay_io.deserialize_inline_cpu_chunk(
                    msg.shm_metadata
                )
            except Exception as exc:
                logger.error(
                    "Stage %s: inline CPU stream deserialize failed for %s: %s",
                    self.name,
                    request_id,
                    exc,
                )
                await self._queue_stream_error(request_id, msg.from_stage, exc)
                return
            if request_id in self._aborted:
                return
            item = StreamItem(
                chunk_id=msg.chunk_id,
                data=data,
                from_stage=msg.from_stage,
                metadata=metadata,
            )
            self._emit_stream_chunk_received(
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                chunk_id=msg.chunk_id,
            )
            await self._route_stream_item_or_fail(request_id, item)
            return

        # Same-GPU CUDA IPC
        if isinstance(msg.shm_metadata, dict) and msg.shm_metadata.get("_ipc"):
            try:
                item = self._deserialize_ipc_chunk(msg)
            except Exception as exc:
                logger.error(
                    "Stage %s: IPC deserialize failed for %s: %s",
                    self.name,
                    request_id,
                    exc,
                )
                await self._queue_stream_error(request_id, msg.from_stage, exc)
                return
            if request_id in self._aborted:
                return
            self._emit_stream_chunk_received(
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                chunk_id=msg.chunk_id,
            )
            await self._route_stream_item_or_fail(request_id, item)
            return

        # Cross-GPU: relay
        blob_key = f"{request_id}:stream:{msg.from_stage}:{msg.to_stage}:{msg.chunk_id}"
        try:
            data = await relay_io.read_blob(self.relay, blob_key, msg.shm_metadata)
            metadata = await self._read_chunk_metadata(msg.shm_metadata, blob_key)
        except Exception as exc:
            logger.error(
                "Stage %s: stream chunk read failed for %s: %s",
                self.name,
                request_id,
                exc,
            )
            await self._queue_stream_error(request_id, msg.from_stage, exc)
            return

        if request_id in self._aborted:
            return

        item = StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata,
        )
        self._emit_stream_chunk_received(
            request_id=msg.request_id,
            from_stage=msg.from_stage,
            chunk_id=msg.chunk_id,
        )
        await self._route_stream_item_or_fail(request_id, item)

    def _enqueue_stream_control_message(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        queue = self._stream_ingest_queues.get(request_id)
        if queue is None:
            queue = asyncio.Queue()
            self._stream_ingest_queues[request_id] = queue
            task = asyncio.create_task(
                self._drain_stream_ingest_queue(request_id, queue)
            )
            self._stream_ingest_tasks[request_id] = task
            task.add_done_callback(
                lambda done, rid=request_id: self._on_stream_ingest_done(rid, done)
            )
        queue.put_nowait(msg)

    async def _drain_stream_ingest_queue(
        self,
        request_id: str,
        queue: asyncio.Queue[DataReadyMessage | None],
    ) -> None:
        while self._running or not queue.empty():
            msg = await queue.get()
            if msg is None:
                break
            if msg.is_done or msg.error:
                await self._on_stream_signal(msg)
                break
            await self._on_stream_chunk(msg)
            await asyncio.sleep(0)

    def _on_stream_ingest_done(self, request_id: str, task: asyncio.Task) -> None:
        current = self._stream_ingest_tasks.get(request_id)
        if current is task:
            self._stream_ingest_tasks.pop(request_id, None)
            self._stream_ingest_queues.pop(request_id, None)
        self._on_background_task_done(task, f"stream ingest {request_id}")

    def _emit_stream_chunk_received(
        self,
        *,
        request_id: str,
        from_stage: str,
        chunk_id: int | None,
    ) -> None:
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_stream_chunk_received",
            metadata={"from_stage": from_stage, "chunk_id": chunk_id},
        )

    async def _route_stream_item_or_fail(
        self, request_id: str, item: StreamItem
    ) -> None:
        if self._open_pre_payload_stream_if_allowed(request_id):
            self._route_stream_item(request_id, item)
            return
        with suppress(Exception):
            self.scheduler.abort(request_id)
        await self._send_failure(
            request_id,
            (
                f"Stage {self.name}: stream chunk from {item.from_stage!r} arrived "
                "before the request payload, but this stage is not configured to "
                "accept pre-payload stream data"
            ),
        )

    async def _queue_stream_error(
        self,
        request_id: str,
        from_stage: str | None,
        error: BaseException,
    ) -> None:
        if request_id in self._aborted:
            return
        logger.error(
            "Stage %s: stream error from %s for %s: %s",
            self.name,
            from_stage,
            request_id,
            error,
        )
        with suppress(Exception):
            self.scheduler.abort(request_id)
        await self._send_failure(request_id, str(error))

    async def _read_chunk_metadata(
        self, shm_metadata: dict, blob_key: str
    ) -> dict | None:
        metadata = {}
        chunk_meta = (
            shm_metadata.get("chunk_metadata")
            if isinstance(shm_metadata, dict)
            else None
        )
        if isinstance(chunk_meta, dict):
            metadata.update(chunk_meta)
        tensor_blobs = (
            shm_metadata.get("chunk_metadata_tensors", {})
            if isinstance(shm_metadata, dict)
            else {}
        )
        if isinstance(tensor_blobs, dict):
            tensor_dict = {}
            for path, info in tensor_blobs.items():
                if not isinstance(info, dict):
                    continue
                meta_blob_key = info.get("blob_key")
                meta_metadata = info.get("relay_metadata")
                if isinstance(meta_blob_key, str) and isinstance(meta_metadata, dict):
                    tensor_dict[path] = await relay_io.read_blob(
                        self.relay, meta_blob_key, meta_metadata
                    )
            if tensor_dict:
                metadata = relay_io.restore_tensors(metadata, tensor_dict)
        return metadata or None

    async def _discard_payload_data(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        if isinstance(msg.shm_metadata, dict) and msg.shm_metadata.get("_payload_ipc"):
            return
        try:
            await relay_io.read_payload(self.relay, request_id, msg.shm_metadata)
        except Exception:
            logger.debug(
                "Stage %s: failed to drain aborted payload for %s",
                self.name,
                request_id,
                exc_info=True,
            )
            self.relay.cleanup(request_id)

    async def _discard_stream_chunk_data(self, msg: DataReadyMessage) -> None:
        if isinstance(msg.shm_metadata, dict) and (
            msg.shm_metadata.get("_ipc") or msg.shm_metadata.get("_inline_cpu")
        ):
            return
        if msg.chunk_id is None:
            return
        blob_key = (
            f"{msg.request_id}:stream:{msg.from_stage}:{msg.to_stage}:{msg.chunk_id}"
        )
        try:
            await relay_io.read_blob(self.relay, blob_key, msg.shm_metadata)
            await self._read_chunk_metadata(msg.shm_metadata, blob_key)
        except Exception:
            logger.debug(
                "Stage %s: failed to drain aborted stream chunk for %s",
                self.name,
                msg.request_id,
                exc_info=True,
            )

    async def _on_stream_signal(self, msg: DataReadyMessage) -> None:
        await self._receive_stream_signal(
            msg.request_id,
            msg.from_stage,
            is_done=msg.is_done,
            error=msg.error,
        )

    async def _receive_stream_signal(
        self,
        request_id: str,
        from_stage: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if error:
            await self._queue_stream_error(
                request_id,
                from_stage,
                RuntimeError(error),
            )
            return

        if is_done:
            if not self._open_pre_payload_stream_if_allowed(request_id):
                with suppress(Exception):
                    self.scheduler.abort(request_id)
                await self._send_failure(
                    request_id,
                    (
                        f"Stage {self.name}: stream_done from {from_stage!r} "
                        "arrived before the request payload, but this stage is not "
                        "configured to accept pre-payload stream data"
                    ),
                )
                return
            self._stream_queue.put_done(request_id, from_stage=from_stage)
            self.scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_done",
                )
            )

    def _open_pre_payload_stream_if_allowed(self, request_id: str) -> bool:
        if self._stream_queue is None:
            return False
        if self._stream_queue.has(request_id):
            return True
        if not self._can_accept_stream_before_payload:
            return False
        self._active_requests.add(request_id)
        self._stream_queue.open(request_id)
        return True

    @staticmethod
    def _deserialize_ipc_chunk(msg: DataReadyMessage) -> StreamItem:
        import pickle as _pickle

        ipc_meta = msg.shm_metadata
        data = _pickle.loads(ipc_meta["tensor_bytes"])
        metadata = {}
        raw_meta = ipc_meta.get("metadata", {})
        if isinstance(raw_meta, dict):
            metadata = relay_io.deserialize_ipc_metadata(raw_meta)
        return StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata or None,
        )

    def _route_stream_item(self, request_id: str, item: StreamItem) -> None:
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="stream_chunk", data=item)
        )
        if request_id not in self._first_scheduler_stream_enqueue_seen:
            self._first_scheduler_stream_enqueue_seen.add(request_id)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_first_stream_chunk_enqueued",
                metadata={
                    "from_stage": item.from_stage,
                    "chunk_id": item.chunk_id,
                    "token_id": (
                        item.metadata.get("token_id")
                        if isinstance(item.metadata, dict)
                        else None
                    ),
                    "inbox_qsize": (
                        self.scheduler.inbox.qsize()
                        if hasattr(self.scheduler.inbox, "qsize")
                        else None
                    ),
                },
            )

    async def _execute(self, payload: Any) -> None:
        request_id = payload.request_id
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_dispatch",
        )
        if (
            self.role == "leader"
            and self._tp_fanout is not None
            and getattr(self.scheduler, "requires_tp_work_fanout", False)
        ):
            self._tp_fanout.fanout_work(payload)
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )

    # ------------------------------------------------------------------
    # Outbox drain: scheduler results → route downstream
    # ------------------------------------------------------------------

    async def _drain_outbox(self) -> None:
        if self._owns_external_io:
            await self._drain_outbox_external()
        else:
            await self._drain_outbox_follower()

    async def _drain_outbox_external(self) -> None:
        """Drain scheduler outbox and route results downstream."""
        handled_messages = 0
        while self._running or not self.scheduler.outbox.empty():
            try:
                out = self.scheduler.outbox.get_nowait()
            except _queue_mod.Empty:
                await asyncio.sleep(0.001)
                continue

            if out.request_id not in self._active_requests:
                if out.type == "stream":
                    _emit_event(
                        request_id=out.request_id,
                        stage=self.name,
                        event_name="stage_outbox_stream_skipped_inactive",
                        metadata={
                            "target": out.target,
                            "modality": (
                                out.metadata.get("modality")
                                if isinstance(out.metadata, dict)
                                else None
                            ),
                        },
                    )
                handled_messages += 1
                if (
                    self._stage_io_yield_every_messages > 0
                    and handled_messages % self._stage_io_yield_every_messages == 0
                ):
                    await asyncio.sleep(0)
                continue

            if out.type == "result":
                await self._route_result(out.request_id, out.data)
            elif out.type == "stream":
                if out.request_id not in self._first_outbox_stream_dequeue_seen:
                    self._first_outbox_stream_dequeue_seen.add(out.request_id)
                    _emit_event(
                        request_id=out.request_id,
                        stage=self.name,
                        event_name="stage_first_outbox_stream_dequeued",
                        metadata={
                            "target": out.target,
                            "modality": (
                                out.metadata.get("modality")
                                if isinstance(out.metadata, dict)
                                else None
                            ),
                            "outbox_qsize": (
                                self.scheduler.outbox.qsize()
                                if hasattr(self.scheduler.outbox, "qsize")
                                else None
                            ),
                        },
                    )
                if out.target is None:
                    if self._stream_targets:
                        for target in self._stream_targets:
                            await self._send_stream_to_target(
                                out.request_id,
                                out.data,
                                target,
                                out.metadata,
                            )
                    else:
                        await self._send_stream_to_coordinator(
                            out.request_id,
                            out.data,
                            out.metadata,
                        )
                else:
                    await self._send_stream_to_target(
                        out.request_id,
                        out.data,
                        out.target,
                        out.metadata,
                    )
            elif out.type == "error":
                await self._send_failure(out.request_id, str(out.data))

            handled_messages += 1
            if (
                self._stage_io_yield_every_messages > 0
                and handled_messages % self._stage_io_yield_every_messages == 0
            ):
                await asyncio.sleep(0)

    async def _drain_outbox_follower(self) -> None:
        """Drain follower outbox without emitting external stage traffic."""
        handled_messages = 0
        while self._running or not self.scheduler.outbox.empty():
            try:
                out = self.scheduler.outbox.get_nowait()
            except _queue_mod.Empty:
                await asyncio.sleep(0.001)
                continue

            if out.type == "result":
                self._clear_request_state(out.request_id)
            elif out.type == "stream":
                pass
            elif out.type == "error":
                raise RuntimeError(
                    f"TP follower stage {self.name} received scheduler error: {out.data}"
                )
            handled_messages += 1
            if (
                self._stage_io_yield_every_messages > 0
                and handled_messages % self._stage_io_yield_every_messages == 0
            ):
                await asyncio.sleep(0)

    async def _route_result(self, request_id: str, result: Any) -> None:
        """Route a completed result to next stage(s) or complete at coordinator."""
        if not self._owns_external_io:
            self._clear_request_state(request_id)
            return
        # Send stream done to the active stream targets for this request.
        stream_targets = self._stream_targets
        if self.get_stream_done_targets is not None:
            resolved = self.get_stream_done_targets(request_id, result)
            if isinstance(resolved, str):
                stream_targets = [resolved]
            elif isinstance(resolved, list):
                stream_targets = resolved
            elif resolved is None:
                stream_targets = []
        stream_targets_for_request = set(stream_targets)
        for target in stream_targets:
            await self._send_stream_signal_to_target(
                request_id,
                target,
                is_done=True,
            )

        next_stages = self.get_next(request_id, result)
        if next_stages is None:
            # Terminal: notify coordinator
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": True},
            )
            complete_msg = CompleteMessage(
                request_id=request_id,
                from_stage=self.name,
                success=True,
                result=result.data,
            )
            if self._should_defer_terminal_complete(result):
                self._enqueue_deferred_complete(complete_msg)
            else:
                await self.control_plane.send_complete(complete_msg)
        else:
            if isinstance(next_stages, str):
                next_stages = [next_stages]
            is_single_target = len(next_stages) == 1
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": False, "next": list(next_stages)},
            )
            for target in next_stages:
                await self._send_to_stage(
                    request_id,
                    target,
                    result,
                    allow_local_object=is_single_target,
                    allow_projected_local_object=not is_single_target,
                    stream_targets_for_request=stream_targets_for_request,
                )

        self._clear_request_state(request_id)

    def _should_defer_terminal_complete(self, result: Any) -> bool:
        if self._deferred_complete_queue is None:
            return False
        request = getattr(result, "request", None)
        params = getattr(request, "params", None)
        return not bool((params or {}).get("stream", False))

    def _enqueue_deferred_complete(self, msg: CompleteMessage) -> None:
        queue = self._deferred_complete_queue
        if queue is None:
            return
        queue.put_nowait(msg)
        _emit_event(
            request_id=msg.request_id,
            stage=self.name,
            event_name="stage_complete_deferred",
            metadata={"queue_size": queue.qsize()},
        )

    async def _drain_deferred_completes(self) -> None:
        queue = self._deferred_complete_queue
        if queue is None:
            return
        while self._running or not queue.empty():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            await self.control_plane.send_complete_low_priority(msg)

    async def _send_to_stage(
        self,
        request_id: str,
        target: str,
        payload: Any,
        *,
        allow_local_object: bool = False,
        allow_projected_local_object: bool = False,
        stream_targets_for_request: set[str] | None = None,
    ) -> None:
        if not self._owns_external_io:
            raise RuntimeError(
                f"Follower stage {self.name} cannot send downstream data"
            )
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            logger.warning("Stage %s: no endpoint for %s", self.name, target)
            return
        projector = self._project_payload.get(target)
        projected_payload = projector(payload) if projector is not None else payload
        use_local_object = allow_local_object or (
            allow_projected_local_object
            and self._is_isolated_projected_payload(
                payload,
                projected_payload,
                projector_present=projector is not None,
            )
        )

        if (
            use_local_object
            and target in self._same_process_targets
            and self._can_send_full_payload_locally(
                request_id,
                target,
                (
                    set(self._stream_targets)
                    if stream_targets_for_request is None
                    else stream_targets_for_request
                ),
            )
        ):
            if self._local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process target {target!r} requires "
                    "a local dispatcher"
                )

            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_hop_sent",
                metadata={"to_stage": target, "transport": "local_object"},
            )
            await self._local_dispatcher.send_payload(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                payload=projected_payload,
            )
            return

        if (
            target in self._same_gpu_payload_targets
            and relay_io.should_use_cuda_ipc_payload(projected_payload)
        ):
            msg = DataReadyMessage(
                request_id=request_id,
                from_stage=self.name,
                to_stage=target,
                shm_metadata=relay_io.serialize_ipc_payload(projected_payload),
            )
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_hop_sent",
                metadata={"to_stage": target, "transport": "cuda_ipc"},
            )
            await self.control_plane.send_to_stage(target, endpoint, msg)
            return

        metadata, op = await relay_io.write_payload(
            self.relay, request_id, projected_payload
        )
        msg = DataReadyMessage(
            request_id=request_id,
            from_stage=self.name,
            to_stage=target,
            shm_metadata=metadata,
        )
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_hop_sent",
            metadata={
                "to_stage": target,
                "tensor_count": metadata.get("tensor_count"),
                "tensor_bytes": metadata.get("tensor_bytes"),
                "relay_bytes": metadata.get("relay_bytes"),
                "payload_pickle_bytes": metadata.get("payload_pickle_bytes"),
            },
        )
        await self.control_plane.send_to_stage(target, endpoint, msg)
        self._track_relay_completion(
            kind="payload",
            request_id=request_id,
            target=target,
            pending_ops=[op],
        )

    @staticmethod
    def _is_isolated_projected_payload(
        original_payload: Any,
        projected_payload: Any,
        *,
        projector_present: bool,
    ) -> bool:
        if not projector_present or projected_payload is original_payload:
            return False
        if not isinstance(original_payload, StagePayload):
            raise TypeError(
                "projected local-object dispatch requires the original payload "
                f"to be StagePayload, got {type(original_payload).__name__}"
            )
        if not isinstance(projected_payload, StagePayload):
            raise TypeError(
                "projected local-object dispatch requires projectors to return "
                f"StagePayload, got {type(projected_payload).__name__}"
            )
        # A fan-out edge may use process-local dispatch only when projection
        # gives the target its own mutable payload/data containers. Tensor leaves
        # inside those containers may still be shared intentionally.
        if projected_payload.data is original_payload.data:
            return False
        return not Stage._shares_mutable_container(
            original_payload.data, projected_payload.data
        )

    @staticmethod
    def _shares_mutable_container(original: Any, projected: Any) -> bool:
        original_ids = Stage._collect_mutable_container_ids(original)
        if not original_ids:
            return False
        return Stage._contains_mutable_container_id(projected, original_ids)

    @staticmethod
    def _collect_mutable_container_ids(
        obj: Any, seen: set[int] | None = None
    ) -> set[int]:
        seen = set() if seen is None else seen
        obj_id = id(obj)
        if obj_id in seen:
            return set()
        seen.add(obj_id)

        ids: set[int] = set()
        if isinstance(obj, (dict, list, set, bytearray)):
            ids.add(obj_id)

        for child in Stage._iter_container_children(obj):
            ids.update(Stage._collect_mutable_container_ids(child, seen))
        return ids

    @staticmethod
    def _contains_mutable_container_id(
        obj: Any, original_ids: set[int], seen: set[int] | None = None
    ) -> bool:
        seen = set() if seen is None else seen
        obj_id = id(obj)
        if obj_id in seen:
            return False
        seen.add(obj_id)

        if isinstance(obj, (dict, list, set, bytearray)) and obj_id in original_ids:
            return True
        return any(
            Stage._contains_mutable_container_id(child, original_ids, seen)
            for child in Stage._iter_container_children(obj)
        )

    @staticmethod
    def _iter_container_children(obj: Any):
        if isinstance(obj, dict):
            return obj.values()
        if isinstance(obj, (list, tuple, set, frozenset)):
            return obj
        return ()

    def _can_send_full_payload_locally(
        self,
        request_id: str,
        target: str,
        stream_targets_for_request: set[str],
    ) -> bool:
        if target in self._nonlocal_stream_targets.get(request_id, set()):
            return False
        if target not in stream_targets_for_request:
            return True
        return target in self._local_stream_targets.get(request_id, set())

    def _record_local_stream_target(self, request_id: str, target: str) -> None:
        self._local_stream_targets.setdefault(request_id, set()).add(target)

    def _record_nonlocal_stream_target(self, request_id: str, target: str) -> None:
        self._nonlocal_stream_targets.setdefault(request_id, set()).add(target)

    async def _send_stream_to_target(
        self,
        request_id: str,
        data: Any,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._owns_external_io:
            return
        endpoint = self.stream_endpoints.get(target) or self.endpoints.get(target)
        if endpoint is None:
            return
        key = (request_id, target)
        chunk_id = self._stream_chunk_counters.get(key, 0)
        self._stream_chunk_counters[key] = chunk_id + 1
        chunk_modality = (
            metadata.get("modality") if isinstance(metadata, dict) else None
        )
        if request_id not in self._first_stream_chunk_seen:
            self._first_stream_chunk_seen.add(request_id)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_first_stream_chunk_sent",
                metadata={"to_stage": target, "modality": chunk_modality},
            )
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_stream_chunk_sent",
            metadata={
                "to_stage": target,
                "chunk_id": chunk_id,
                "modality": chunk_modality,
            },
        )
        if target in self._same_process_targets:
            if self._local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process stream target {target!r} "
                    "requires a local dispatcher"
                )
            self._record_local_stream_target(request_id, target)
            await self._local_dispatcher.send_stream_chunk(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                chunk_id=chunk_id,
                data=data,
                metadata=metadata,
            )
            return
        self._record_nonlocal_stream_target(request_id, target)
        pending_ops = await relay_io.send_stream_chunk(
            self.relay,
            self.control_plane,
            request_id=request_id,
            data=data,
            target_stage=target,
            target_endpoint=endpoint,
            from_stage=self.name,
            chunk_id=chunk_id,
            metadata=metadata,
            same_gpu_targets=self._same_gpu_targets,
            await_completion=False,
        )
        self._track_relay_completion(
            kind="stream",
            request_id=request_id,
            target=target,
            chunk_id=chunk_id,
            pending_ops=pending_ops,
        )

    def _track_relay_completion(
        self,
        *,
        kind: str,
        request_id: str,
        target: str,
        pending_ops: list[Any],
        chunk_id: int | None = None,
    ) -> None:
        if not pending_ops:
            return
        task = asyncio.create_task(
            self._wait_relay_completion(
                request_id=request_id,
                target=target,
                chunk_id=chunk_id,
                pending_ops=pending_ops,
            )
        )
        self._stream_relay_completion_tasks.add(task)
        task.add_done_callback(
            lambda done: self._on_relay_completion_done(
                done,
                kind=kind,
                request_id=request_id,
                target=target,
                chunk_id=chunk_id,
            )
        )

    async def _wait_relay_completion(
        self,
        *,
        request_id: str,
        target: str,
        chunk_id: int | None,
        pending_ops: list[Any],
    ) -> None:
        del request_id, target, chunk_id
        await relay_io.wait_for_relay_ops(
            pending_ops,
            timeout=self._stream_relay_completion_timeout,
        )

    def _on_relay_completion_done(
        self,
        task: asyncio.Task,
        *,
        kind: str,
        request_id: str,
        target: str,
        chunk_id: int | None,
    ) -> None:
        self._stream_relay_completion_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.warning(
                "Stage %s: %s relay completion wait failed "
                "(request_id=%s target=%s chunk_id=%s)",
                self.name,
                kind,
                request_id,
                target,
                chunk_id,
                exc_info=True,
            )

    async def _send_stream_signal_to_target(
        self,
        request_id: str,
        target: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        if not self._owns_external_io:
            return
        endpoint = self.stream_endpoints.get(target) or self.endpoints.get(target)
        if endpoint is None:
            return
        if target in self._same_process_targets:
            if self._local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process stream target {target!r} "
                    "requires a local dispatcher"
                )
            self._record_local_stream_target(request_id, target)
            await self._local_dispatcher.send_stream_signal(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                is_done=is_done,
                error=error,
            )
            return
        self._record_nonlocal_stream_target(request_id, target)
        await relay_io.send_stream_signal(
            self.control_plane,
            request_id=request_id,
            target_stage=target,
            target_endpoint=endpoint,
            from_stage=self.name,
            is_done=is_done,
            error=error,
        )

    async def _send_stream_to_coordinator(
        self,
        request_id: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Forward a terminal stage's stream chunk to the Coordinator."""
        if not self._is_terminal:
            raise RuntimeError(
                f"Stage {self.name!r} emitted untargeted stream chunk but isn't "
                "terminal. Set ``terminal=True``, or use ``target=...`` / "
                "``stream_to=[...]``."
            )
        if not self._owns_external_io:
            return
        if request_id in self._aborted:
            return
        modality = metadata.get("modality") if isinstance(metadata, dict) else None
        if modality is None and isinstance(data, dict):
            modality = data.get("modality")
        key = (request_id, "coordinator")
        chunk_id = self._stream_chunk_counters.get(key, 0)
        self._stream_chunk_counters[key] = chunk_id + 1
        msg = StreamMessage(
            request_id=request_id,
            from_stage=self.name,
            chunk=data,
            stage_name=self.name,
            modality=modality,
            chunk_id=chunk_id,
        )
        if request_id not in self._first_stream_chunk_seen:
            self._first_stream_chunk_seen.add(request_id)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_first_stream_chunk_sent",
                metadata={
                    "to_stage": "coordinator",
                    "chunk_id": chunk_id,
                    "modality": modality,
                },
            )
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_stream_chunk_sent",
            metadata={
                "to_stage": "coordinator",
                "chunk_id": chunk_id,
                "modality": modality,
            },
        )
        await self.control_plane.send_stream(msg)

    async def _send_failure(self, request_id: str, error: str) -> None:
        self._record_aborted_request_id(request_id)
        if not self._owns_external_io:
            self._clear_request_state(request_id)
            raise RuntimeError(f"Follower stage {self.name} failed: {error}")
        await self.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.name,
                success=False,
                error=error,
            )
        )
        self._clear_request_state(request_id)

    def _clear_request_state(self, request_id: str) -> None:
        self._active_requests.discard(request_id)
        self.input_handler.cancel(request_id)
        if self._stream_queue is not None:
            self._stream_queue.close(request_id)
        stream_ingest_queue = self._stream_ingest_queues.pop(request_id, None)
        if stream_ingest_queue is not None:
            stream_ingest_queue.put_nowait(None)
        self._stream_ingest_tasks.pop(request_id, None)
        stale_keys = [
            key for key in self._stream_chunk_counters if key[0] == request_id
        ]
        for key in stale_keys:
            self._stream_chunk_counters.pop(key, None)
        self._first_stream_chunk_seen.discard(request_id)
        first_outbox_seen = getattr(self, "_first_outbox_stream_dequeue_seen", None)
        if first_outbox_seen is not None:
            first_outbox_seen.discard(request_id)
        first_enqueue_seen = getattr(
            self, "_first_scheduler_stream_enqueue_seen", None
        )
        if first_enqueue_seen is not None:
            first_enqueue_seen.discard(request_id)
        self._local_stream_targets.pop(request_id, None)
        self._nonlocal_stream_targets.pop(request_id, None)

    async def _handle_scheduler_crash(self, exc: BaseException) -> None:
        if self._scheduler_crash_error is not None:
            return
        self._scheduler_crash_error = exc
        if not self._owns_external_io:
            self.control_plane.close()
            return
        error = f"scheduler crashed: {exc}"
        active_request_ids = [
            request_id
            for request_id in list(self._active_requests)
            if request_id not in self._aborted
        ]
        for request_id in active_request_ids:
            with suppress(Exception):
                self.scheduler.abort(request_id)
            await self._send_failure(request_id, error)
            with suppress(Exception):
                self.relay.cleanup(request_id)
        self.control_plane.close()

    async def _abort_listener(self) -> None:
        try:
            while self._running:
                abort_msg = await self.control_plane.recv_abort()
                if self.role == "leader" and self._tp_fanout is not None:
                    await self._tp_fanout.fanout_abort(abort_msg)
                self._on_abort(abort_msg.request_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            if self._scheduler_crash_error is None and self._running:
                logger.exception("Stage %s abort listener crashed", self.name)

    def _record_aborted_request_id(self, request_id: str) -> None:
        self._aborted.add(request_id)
        if len(self._aborted) > 10000:
            excess = len(self._aborted) - 5000
            it = iter(self._aborted)
            to_remove = [next(it) for _ in range(excess)]
            self._aborted -= set(to_remove)

    def _on_abort(self, request_id: str) -> None:
        self._record_aborted_request_id(request_id)
        self.relay.cleanup(request_id)
        self._clear_request_state(request_id)
        self.scheduler.abort(request_id)

    def _on_profiler_start(self, msg: ProfilerStartMessage) -> None:
        run_id = msg.run_id
        if msg.enable_torch and not TorchProfiler.is_active():
            base_tpl = msg.trace_path_template.format(run_id=run_id, stage=self.name)
            template = f"{base_tpl}_pid{os.getpid()}"
            prof_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
            if prof_dir and not os.path.isabs(template):
                template = os.path.join(prof_dir, template)
            TorchProfiler.start(template, run_id=run_id)
        if msg.event_dir is not None:
            try:
                _get_recorder().start(
                    run_id=run_id, event_dir=msg.event_dir, stage=self.name
                )
            except Exception:
                logger.warning(
                    "Stage %s failed to start request event recorder",
                    self.name,
                    exc_info=True,
                )

    def _on_profiler_stop(self, msg: ProfilerStopMessage) -> None:
        # run_id=None is a wildcard (stop whatever's active).
        if TorchProfiler.is_active() and (
            msg.run_id is None or TorchProfiler.get_active_run_id() == msg.run_id
        ):
            TorchProfiler.stop(run_id=msg.run_id)
        recorder = _get_recorder()
        if recorder.is_active() and (
            msg.run_id is None or recorder.active_run_id() == msg.run_id
        ):
            recorder.stop(run_id=msg.run_id)

    def _on_background_task_done(self, task: asyncio.Task, label: str) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        logger.exception(
            "Stage %s %s task crashed",
            self.name,
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        if self._background_task_error is None:
            self._background_task_error = exc
        self._running = False
        self.control_plane.close()

    def info(self) -> StageInfo:
        return StageInfo(
            name=self.name,
            control_endpoint=self.control_plane.recv_endpoint,
        )
