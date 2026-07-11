# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from multiprocessing import shared_memory as _shm
from typing import Any

import numpy as np
import torch

from .base import Relay, RelayOperation, register_relay

logger = logging.getLogger(__name__)


def shm_create_from_tensor(tensor: torch.Tensor) -> _shm.SharedMemory:
    """Creates a SHM block and writes tensor data into it (optimized single copy)."""
    t_cpu = tensor.cpu() if tensor.is_cuda else tensor
    t_np = t_cpu.numpy().reshape(-1)
    size = t_np.nbytes

    # 1. Create SHM directly
    shm = _shm.SharedMemory(create=True, size=size)

    # 2. Create a numpy view based on SHM memory
    # This step is instantaneous and involves no copying
    shm_view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)

    # 3. Direct data copy (Only One Copy)
    # Uses low-level C memcpy to write directly from source Tensor to SHM
    shm_view[:] = t_np[:]

    return shm


def shm_create_from_segments(
    segments: list[tuple[int, torch.Tensor]], total_size: int
) -> _shm.SharedMemory:
    """Create one SHM block and copy tensor byte segments into final offsets."""
    if total_size <= 0:
        raise ValueError(f"total_size must be positive, got {total_size}")

    shm = _shm.SharedMemory(create=True, size=total_size)
    try:
        shm_view = torch.frombuffer(shm.buf, dtype=torch.uint8)
        for offset, tensor in segments:
            flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            if flat.is_cuda:
                flat = flat.cpu()
            end = offset + flat.numel()
            if offset < 0 or end > total_size:
                raise ValueError(
                    f"SHM segment [{offset}, {end}) exceeds total size {total_size}"
                )
            shm_view[offset:end].copy_(flat)
    except Exception:
        shm.close()
        shm.unlink()
        raise
    return shm


class ShmOperation(RelayOperation):
    """Base class implementation for SHM operations."""

    def __init__(self, metadata: Any):
        self._metadata = metadata
        self._completed = False

    @property
    def metadata(self) -> Any:
        return self._metadata

    # wait_for_completion is implemented by subclasses


class ShmPutOperation(ShmOperation):
    """
    Handle for Put.
    In this simplified SHM model, writing is synchronous during creation,
    so the operation is effectively complete immediately.
    """

    def __init__(self, metadata: Any, shm_obj: _shm.SharedMemory):
        super().__init__(metadata)
        self._shm_obj = shm_obj

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        # Sender simply closes the local handle; Receiver is responsible for unlinking.
        if not self._completed:
            self._shm_obj.close()
            self._completed = True
        return


class ShmGetOperation(ShmOperation):
    """
    Handle for Get.
    Performs copy from SHM to destination tensor and unlinks the shared memory.
    """

    def __init__(self, metadata: Any, dest_tensor: torch.Tensor):
        super().__init__(metadata)
        self._transfer_info = metadata["transfer_info"]
        self._dest_tensor = dest_tensor

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self._completed:
            return

        shm_name = self._transfer_info["shm_name"]
        size = self._transfer_info["size"]

        try:
            # 1. Open SHM
            try:
                existing_shm = _shm.SharedMemory(name=shm_name)
            except FileNotFoundError:
                raise RuntimeError(f"SHM block {shm_name} not found.")

            try:
                # 2. Zero-copy Read -> Copy to Dest
                shm_array = np.ndarray((size,), dtype=np.uint8, buffer=existing_shm.buf)
                src_tensor = torch.from_numpy(shm_array)

                dest_view = self._dest_tensor.view(torch.uint8).reshape(-1)
                copy_len = min(dest_view.numel(), size)
                dest_view[:copy_len].copy_(src_tensor[:copy_len])

                if self._dest_tensor.is_cuda:
                    torch.cuda.synchronize(self._dest_tensor.device)

            finally:
                # 3. Cleanup (Receiver owns lifecycle)
                existing_shm.close()
                existing_shm.unlink()

        finally:
            self._completed = True


@register_relay("shm")
class ShmRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cpu",
    ):
        self.engine_id = engine_id
        self.device = device
        # Semaphore mimics the 'credits' flow control
        self._sem = asyncio.Semaphore(credits)
        self._slot_size_bytes = slot_size_mb * 1024 * 1024

    async def put_async(
        self, tensor: torch.Tensor, request_id: str = None, dst_rank: int = None
    ) -> RelayOperation:
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Flow control
        await self._sem.acquire()

        try:
            # 1. Create SHM and write data
            shm = shm_create_from_tensor(tensor)
            size_bytes = shm.size

            # 2. Construct Metadata
            metadata = {
                "engine_id": self.engine_id,
                "transfer_info": {
                    "shm_name": shm.name,
                    "size": size_bytes,
                    "req_id": request_id,
                },
            }

            # 3. Release semaphore immediately (Fire-and-Forget model)
            self._sem.release()

            return ShmPutOperation(metadata, shm)

        except Exception as e:
            self._sem.release()
            raise e

    async def put_many_async(
        self,
        segments: list[tuple[int, torch.Tensor]],
        total_size: int,
        request_id: str | None = None,
        dst_rank: int | None = None,
    ) -> RelayOperation:
        """Write pre-positioned tensors directly into one SHM allocation."""
        del dst_rank
        if request_id is None:
            request_id = str(uuid.uuid4())

        await self._sem.acquire()
        try:
            shm = shm_create_from_segments(segments, total_size)
            metadata = {
                "engine_id": self.engine_id,
                "transfer_info": {
                    "shm_name": shm.name,
                    "size": shm.size,
                    "req_id": request_id,
                },
            }
            self._sem.release()
            return ShmPutOperation(metadata, shm)
        except Exception:
            self._sem.release()
            raise

    def map_tensor(
        self, metadata: Any, request_id: str | None = None
    ) -> torch.Tensor | None:
        """Map a CPU SHM transfer as a tensor without copying its bytes."""
        del request_id
        if torch.device(self.device).type != "cpu":
            return None

        transfer_info = metadata["transfer_info"]
        shm_name = transfer_info["shm_name"]
        size = int(transfer_info["size"])
        shm_path = os.path.join("/dev/shm", shm_name.lstrip("/"))
        if not os.path.exists(shm_path):
            return None

        existing_shm = _shm.SharedMemory(name=shm_name)
        try:
            tensor = torch.from_file(
                shm_path,
                shared=True,
                size=size,
                dtype=torch.uint8,
            )
            existing_shm.unlink()
            return tensor
        finally:
            existing_shm.close()

    async def get_async(
        self, metadata: Any, dest_tensor: torch.Tensor, request_id: str = None
    ) -> RelayOperation:
        # Note: metadata validation is implicit here based on usage in test
        return ShmGetOperation(metadata=metadata, dest_tensor=dest_tensor)

    def cleanup(self, request_id: str) -> None:
        # In this pattern, cleanup is handled inside wait_for_completion (unlink)
        # or via garbage collection if the process dies.
        pass

    def close(self) -> None:
        pass

    # Optional hook for tests
    def reset_pool(self):
        pass
