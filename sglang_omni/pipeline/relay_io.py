# SPDX-License-Identifier: Apache-2.0
"""Relay IO utilities for inter-stage data transfer.

Handles payload serialization (tensor extraction/restoration), relay read/write,
streaming chunk transfer, and NIXL credit deadlock avoidance.

Extracted from worker/data_plane.py and worker/runtime.py.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pickle
from dataclasses import fields, is_dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any

import torch

from sglang_omni.proto import DataReadyMessage, StagePayload
from sglang_omni.relay.base import Relay

_FULL_PAYLOAD_CUDA_IPC_ENV = "SGLANG_OMNI_FULL_PAYLOAD_CUDA_IPC"
_FULL_PAYLOAD_CUDA_IPC_MAX_CONTROL_BYTES_ENV = (
    "SGLANG_OMNI_FULL_PAYLOAD_CUDA_IPC_MAX_CONTROL_BYTES"
)
_FULL_PAYLOAD_CUDA_IPC_DEFAULT_MAX_CONTROL_BYTES = 4 * 1024 * 1024
_PAYLOAD_PREP_EXECUTOR_ENV = "SGLANG_OMNI_RELAY_PAYLOAD_PREP_EXECUTOR"
_DEFAULT_RELAY_COMPLETION_TIMEOUT_S = 30.0
_INLINE_CPU_STREAM_CHUNK_MAX_BYTES_ENV = (
    "SGLANG_OMNI_INLINE_CPU_STREAM_CHUNK_MAX_BYTES"
)
_INLINE_CPU_STREAM_CHUNK_DEFAULT_MAX_BYTES = 0


def _dtype_alignment(dtype: torch.dtype) -> int:
    return max(torch.empty((), dtype=dtype).element_size(), 1)


def _pad_offset(offset: int, alignment: int) -> int:
    return (-offset) % alignment


# ---------------------------------------------------------------------------
# Tensor extraction / restoration (recursive, nested dicts/lists)
# ---------------------------------------------------------------------------


def extract_tensors(obj: Any, path: str = "") -> tuple[Any, dict[str, torch.Tensor]]:
    """Recursively extract tensors from nested structure, replacing with placeholders."""
    tensors = {}

    if isinstance(obj, torch.Tensor):
        placeholder = {
            "_tensor_placeholder": path,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
        tensors[path] = obj
        return placeholder, tensors

    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            new_value, sub_tensors = extract_tensors(value, new_path)
            new_dict[key] = new_value
            tensors.update(sub_tensors)
        return new_dict, tensors

    elif isinstance(obj, (list, tuple)):
        new_list = []
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            new_item, sub_tensors = extract_tensors(item, new_path)
            new_list.append(new_item)
            tensors.update(sub_tensors)
        return (type(obj)(new_list), tensors)

    else:
        return obj, tensors


def restore_tensors(obj: Any, tensor_dict: dict[str, torch.Tensor]) -> Any:
    """Recursively restore tensors from placeholders."""
    if isinstance(obj, dict):
        if "_tensor_placeholder" in obj:
            path = obj["_tensor_placeholder"]
            return tensor_dict.get(path)
        else:
            return {
                key: restore_tensors(value, tensor_dict) for key, value in obj.items()
            }
    elif isinstance(obj, (list, tuple)):
        return type(obj)(restore_tensors(item, tensor_dict) for item in obj)
    else:
        return obj


# ---------------------------------------------------------------------------
# Payload read/write (full StagePayload via relay)
# ---------------------------------------------------------------------------


async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
) -> tuple[dict[str, Any], Any]:
    """Write a StagePayload to relay. Returns (control_plane_metadata, relay_op)."""
    device = getattr(relay, "device", "cpu")
    if _env_enabled(_PAYLOAD_PREP_EXECUTOR_ENV):
        loop = asyncio.get_running_loop()
        metadata_bytes, tensor_info, all_tensors = await loop.run_in_executor(
            None,
            _prepare_payload_for_relay,
            device,
            payload,
        )
    else:
        metadata_bytes, tensor_info, all_tensors = _prepare_payload_for_relay(
            device,
            payload,
        )

    op = await relay.put_async(all_tensors, request_id=request_id)
    tensor_bytes = sum(int(info.get("size", 0)) for info in tensor_info)

    return {
        "relay_info": op.metadata,
        "payload_pickle": base64.b64encode(metadata_bytes).decode("ascii"),
        "tensor_info": tensor_info,
        "payload_pickle_bytes": len(metadata_bytes),
        "tensor_count": len(tensor_info),
        "tensor_bytes": tensor_bytes,
        "relay_bytes": int(all_tensors.numel()),
    }, op


def _prepare_payload_for_relay(
    device: str,
    payload: StagePayload,
) -> tuple[bytes, list[dict[str, Any]], torch.Tensor]:
    transport_device = torch.device(device)
    modified_data, tensor_dict = extract_tensors(payload.data)
    payload_no_tensors = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=modified_data,
    )
    metadata_bytes = pickle.dumps(payload_no_tensors)

    if tensor_dict:
        tensor_buffers = []
        tensor_info = []
        offset = 0
        for path, tensor in tensor_dict.items():
            flat = tensor.contiguous().view(torch.uint8).reshape(-1)
            if flat.device != transport_device:
                flat = flat.to(device=transport_device)
            padding = _pad_offset(offset, _dtype_alignment(tensor.dtype))
            if padding:
                tensor_buffers.append(
                    torch.zeros(padding, dtype=torch.uint8, device=transport_device)
                )
                offset += padding
            tensor_buffers.append(flat)
            tensor_info.append(
                {
                    "path": path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "offset": offset,
                    "size": flat.numel(),
                }
            )
            offset += flat.numel()
        all_tensors = torch.cat(tensor_buffers)
    else:
        all_tensors = torch.zeros(1, dtype=torch.uint8, device=device)
        tensor_info = []
    if transport_device.type == "cuda":
        torch.cuda.synchronize(transport_device)
    return metadata_bytes, tensor_info, all_tensors


async def read_payload(
    relay: Relay,
    request_id: str,
    metadata: dict[str, Any],
) -> StagePayload:
    """Read a StagePayload from relay using control_plane metadata."""
    device = getattr(relay, "device", "cpu")

    payload_bytes = base64.b64decode(metadata["payload_pickle"])
    payload_no_tensors = pickle.loads(payload_bytes)

    relay_info = metadata["relay_info"]
    tensor_info = metadata.get("tensor_info", [])
    tensor_dict = {}

    data_size = relay_info["transfer_info"]["size"]
    recv_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
    op = await relay.get_async(
        metadata=relay_info, dest_tensor=recv_tensor, request_id=request_id
    )
    await op.wait_for_completion()

    if tensor_info:
        for info in tensor_info:
            path = info["path"]
            shape = info["shape"]
            dtype_str = info["dtype"]
            offset = info["offset"]
            size = info["size"]
            tensor_bytes = recv_tensor[offset : offset + size]
            dtype = getattr(torch, dtype_str.replace("torch.", ""))
            tensor = tensor_bytes.view(dtype).reshape(shape)
            tensor_dict[path] = tensor

    restored_data = restore_tensors(payload_no_tensors.data, tensor_dict)
    payload = StagePayload(
        request_id=payload_no_tensors.request_id,
        request=payload_no_tensors.request,
        data=restored_data,
    )
    relay.cleanup(request_id)
    return payload


# ---------------------------------------------------------------------------
# Blob read/write (raw tensor via relay, for streaming chunks)
# ---------------------------------------------------------------------------


async def write_blob(
    relay: Relay,
    key: str,
    tensor: torch.Tensor,
) -> tuple[dict[str, Any], Any]:
    """Write a raw tensor to relay. Returns (metadata, relay_op)."""
    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    transport_device = torch.device(getattr(relay, "device", "cpu"))
    if flat.device != transport_device:
        flat = flat.to(device=transport_device)
    padding = _pad_offset(0, _dtype_alignment(tensor.dtype))
    if padding:
        flat = torch.cat(
            [
                torch.zeros(padding, dtype=torch.uint8, device=transport_device),
                flat,
            ]
        )
    op = await relay.put_async(flat, request_id=key)
    metadata = {
        "relay_info": op.metadata,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_offset": padding,
    }
    return metadata, op


async def read_blob(
    relay: Relay,
    key: str,
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Read a raw tensor from relay."""
    device = getattr(relay, "device", "cpu")
    relay_info = metadata["relay_info"]
    shape = metadata["tensor_shape"]
    dtype_str = metadata["tensor_dtype"]
    offset = int(metadata.get("tensor_offset", 0))

    data_size = relay_info["transfer_info"]["size"]
    recv_buf = torch.zeros(data_size, dtype=torch.uint8, device=device)
    op = await relay.get_async(
        metadata=relay_info, dest_tensor=recv_buf, request_id=key
    )
    await op.wait_for_completion()

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return recv_buf[offset:].view(dtype).reshape(shape)


async def wait_for_relay_op(
    op: Any,
    *,
    timeout: float | None = _DEFAULT_RELAY_COMPLETION_TIMEOUT_S,
) -> None:
    """Wait for a relay operation.

    ``timeout=None`` means wait indefinitely. A few unit-test fakes implement
    the older no-argument method, so retry without the timeout only when the
    call signature rejects the keyword.
    """

    try:
        if timeout is None:
            await op.wait_for_completion(timeout=float("inf"))
        else:
            await op.wait_for_completion(timeout=timeout)
    except TypeError:
        await op.wait_for_completion()


async def wait_for_relay_ops(
    ops: list[Any],
    *,
    timeout: float | None = _DEFAULT_RELAY_COMPLETION_TIMEOUT_S,
) -> None:
    for op in ops:
        await wait_for_relay_op(op, timeout=timeout)


# ---------------------------------------------------------------------------
# Stream chunk send
# ---------------------------------------------------------------------------

_IPC_INLINE_CPU_BYTES_LIMIT = 64 * 1024


def _is_cuda_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor) and obj.is_cuda


def _contains_cuda_tensor(obj: Any) -> bool:
    if _is_cuda_tensor(obj):
        return True
    if isinstance(obj, torch.Tensor):
        return False
    if isinstance(obj, dict):
        return any(_contains_cuda_tensor(value) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_cuda_tensor(value) for value in obj)
    return False


def _contains_cpu_tensor(obj: Any, seen: set[int] | None = None) -> bool:
    if obj is None:
        return False
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return False
    seen.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return not _is_cuda_tensor(obj)
    if isinstance(obj, dict):
        return any(_contains_cpu_tensor(value, seen) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_cpu_tensor(value, seen) for value in obj)
    return False


def _cpu_tensor_bytes(obj: Any, seen: set[int] | None = None) -> int:
    if obj is None:
        return 0
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return 0 if obj.is_cuda else obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(
            _cpu_tensor_bytes(key, seen) + _cpu_tensor_bytes(value, seen)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_cpu_tensor_bytes(value, seen) for value in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return sum(_cpu_tensor_bytes(getattr(obj, field.name), seen) for field in fields(obj))
    return 0


def _inline_cpu_pickle_size(obj: Any, seen: set[int] | None = None) -> int:
    if obj is None:
        return 0
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return 0
    if isinstance(obj, dict):
        return sum(
            _inline_cpu_pickle_size(key, seen) + _inline_cpu_pickle_size(value, seen)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_inline_cpu_pickle_size(value, seen) for value in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return sum(
            _inline_cpu_pickle_size(getattr(obj, field.name), seen)
            for field in fields(obj)
        )

    try:
        return len(pickle.dumps(obj))
    except Exception:
        return _IPC_INLINE_CPU_BYTES_LIMIT + 1


def _should_use_cuda_ipc_stream_chunk(data: Any, metadata: dict | None) -> bool:
    if not _contains_cuda_tensor(data):
        return False
    if _contains_cpu_tensor(data) or _contains_cpu_tensor(metadata):
        return False
    inline_size = _inline_cpu_pickle_size(data) + _inline_cpu_pickle_size(metadata)
    return inline_size <= _IPC_INLINE_CPU_BYTES_LIMIT


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _inline_cpu_stream_chunk_max_bytes() -> int:
    raw = os.getenv(_INLINE_CPU_STREAM_CHUNK_MAX_BYTES_ENV)
    if raw is None or raw == "":
        return _INLINE_CPU_STREAM_CHUNK_DEFAULT_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _INLINE_CPU_STREAM_CHUNK_DEFAULT_MAX_BYTES


def _full_payload_cuda_ipc_max_control_bytes() -> int:
    raw = os.getenv(_FULL_PAYLOAD_CUDA_IPC_MAX_CONTROL_BYTES_ENV)
    if raw is None:
        return _FULL_PAYLOAD_CUDA_IPC_DEFAULT_MAX_CONTROL_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _FULL_PAYLOAD_CUDA_IPC_DEFAULT_MAX_CONTROL_BYTES


def should_use_cuda_ipc_payload(payload: Any) -> bool:
    if not _env_enabled(_FULL_PAYLOAD_CUDA_IPC_ENV):
        return False
    if not isinstance(payload, StagePayload):
        return False
    if not _contains_cuda_tensor(payload.data):
        return False
    inline_size = _inline_cpu_pickle_size(payload)
    inline_size += _cpu_tensor_bytes(payload)
    return inline_size <= _full_payload_cuda_ipc_max_control_bytes()


def ipc_pickle(obj: Any) -> bytes:
    """Serialize via ForkingPickler only when CUDA IPC tensor handles are needed."""
    if not _contains_cuda_tensor(obj):
        return pickle.dumps(obj)
    buf = io.BytesIO()
    ForkingPickler(buf, 2).dump(obj)
    return buf.getvalue()


def serialize_ipc_payload(payload: StagePayload) -> dict[str, Any]:
    return {
        "_payload_ipc": True,
        "payload_bytes": ipc_pickle(payload),
    }


def deserialize_ipc_payload(metadata: dict[str, Any]) -> StagePayload:
    payload = pickle.loads(metadata["payload_bytes"])
    if not isinstance(payload, StagePayload):
        raise TypeError(
            "CUDA IPC payload metadata did not contain a StagePayload, got "
            f"{type(payload).__name__}"
        )
    return payload


def _serialize_ipc_metadata_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"_ipc_tensor": ipc_pickle(value)}
    if isinstance(value, dict):
        return {key: _serialize_ipc_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_ipc_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return {"_ipc_tuple": [_serialize_ipc_metadata_value(item) for item in value]}
    return value


def serialize_ipc_chunk(
    data: Any,
    metadata: dict | None,
) -> dict[str, Any]:
    ipc_metadata: dict[str, Any] = {"_ipc": True}
    ipc_metadata["tensor_bytes"] = ipc_pickle(data)

    if metadata:
        ipc_metadata["metadata"] = _serialize_ipc_metadata_value(metadata)

    return ipc_metadata


def serialize_inline_cpu_chunk(
    data: Any,
    metadata: dict | None,
) -> dict[str, Any]:
    payload_bytes = pickle.dumps(
        {"data": data, "metadata": metadata},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return {
        "_inline_cpu": True,
        "payload_bytes": payload_bytes,
        "payload_pickle_bytes": len(payload_bytes),
    }


def deserialize_inline_cpu_chunk(
    inline_metadata: dict[str, Any],
) -> tuple[Any, dict | None]:
    payload = pickle.loads(inline_metadata["payload_bytes"])
    return payload["data"], payload.get("metadata")


def _try_serialize_inline_cpu_chunk(
    data: Any,
    metadata: dict | None,
) -> dict[str, Any] | None:
    max_bytes = _inline_cpu_stream_chunk_max_bytes()
    if max_bytes <= 0:
        return None
    if _contains_cuda_tensor(data) or _contains_cuda_tensor(metadata):
        return None
    # Metadata tensors are often hidden states or codec-side side channels; keep
    # them on the relay path so the control plane only carries tiny token data.
    if _contains_cpu_tensor(metadata):
        return None
    try:
        inline_metadata = serialize_inline_cpu_chunk(data, metadata)
    except Exception:
        return None
    if inline_metadata["payload_pickle_bytes"] > max_bytes:
        return None
    return inline_metadata


def deserialize_ipc_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"_ipc_tensor"}:
            return pickle.loads(value["_ipc_tensor"])
        if set(value) == {"_ipc_tuple"}:
            return tuple(deserialize_ipc_metadata(item) for item in value["_ipc_tuple"])
        return {key: deserialize_ipc_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deserialize_ipc_metadata(item) for item in value]
    return value


async def _send_stream_control_message(
    control_plane: Any,
    target_stage: str,
    target_endpoint: str,
    msg: DataReadyMessage,
) -> None:
    send_stream = getattr(control_plane, "send_stream_to_stage", None)
    if callable(send_stream):
        await send_stream(target_stage, target_endpoint, msg)
        return
    await control_plane.send_to_stage(target_stage, target_endpoint, msg)


async def send_stream_chunk(
    relay: Relay,
    control_plane: Any,
    *,
    request_id: str,
    data: Any,
    target_stage: str,
    target_endpoint: str,
    from_stage: str,
    chunk_id: int,
    metadata: dict | None = None,
    same_gpu_targets: set[str] | None = None,
    await_completion: bool = True,
    completion_timeout: float | None = _DEFAULT_RELAY_COMPLETION_TIMEOUT_S,
) -> list[Any]:
    """Send a streaming chunk to a downstream stage."""
    inline_metadata = _try_serialize_inline_cpu_chunk(data, metadata)
    if inline_metadata is not None:
        msg = DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            shm_metadata=inline_metadata,
            chunk_id=chunk_id,
        )
        await _send_stream_control_message(
            control_plane,
            target_stage,
            target_endpoint,
            msg,
        )
        return []

    # Keep CUDA IPC limited to CUDA-dominant chunks with no CPU tensors and only
    # small inline Python metadata; otherwise the relay path keeps CPU-heavy
    # pieces out of the IPC control-plane pickle.
    if (
        same_gpu_targets
        and target_stage in same_gpu_targets
        and _should_use_cuda_ipc_stream_chunk(data, metadata)
    ):
        msg = DataReadyMessage(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            shm_metadata=serialize_ipc_chunk(data, metadata),
            chunk_id=chunk_id,
        )
        await _send_stream_control_message(
            control_plane,
            target_stage,
            target_endpoint,
            msg,
        )
        return []

    if (
        same_gpu_targets
        and target_stage in same_gpu_targets
        and _contains_cuda_tensor(data)
        and not isinstance(data, torch.Tensor)
    ):
        raise ValueError(
            "CUDA IPC stream chunks with mixed object graphs must not carry "
            "CPU-heavy data through the control plane; use tensor data with "
            "relay-backed metadata instead"
        )

    blob_key = f"{request_id}:stream:{from_stage}:{target_stage}:{chunk_id}"

    pending_ops = []
    relay_metadata, op = await write_blob(relay, blob_key, data)
    pending_ops.append(op)

    if metadata:
        cleaned_meta, tensor_dict = extract_tensors(metadata)
        relay_metadata["chunk_metadata"] = cleaned_meta
        if tensor_dict:
            metadata_refs: dict[str, Any] = {}
            for meta_idx, (tkey, tensor) in enumerate(tensor_dict.items()):
                meta_blob_key = f"{blob_key}:meta:{meta_idx}"
                meta_relay_info, meta_op = await write_blob(
                    relay, meta_blob_key, tensor
                )
                pending_ops.append(meta_op)
                metadata_refs[tkey] = {
                    "blob_key": meta_blob_key,
                    "relay_metadata": meta_relay_info,
                }
            relay_metadata["chunk_metadata_tensors"] = metadata_refs

    # Send control message FIRST — receiver starts reading immediately.
    # NIXL credit deadlock avoidance: if we wait_for_completion before notifying,
    # the receiver never starts reading, never triggers RDMA notification, deadlock.
    msg = DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
        shm_metadata=relay_metadata,
        chunk_id=chunk_id,
    )
    await _send_stream_control_message(control_plane, target_stage, target_endpoint, msg)

    if await_completion:
        await wait_for_relay_ops(pending_ops, timeout=completion_timeout)
    return pending_ops


async def send_stream_signal(
    control_plane: Any,
    *,
    request_id: str,
    target_stage: str,
    target_endpoint: str,
    from_stage: str,
    is_done: bool = False,
    error: str | None = None,
) -> None:
    """Send stream done/error signal to downstream stage."""
    msg = DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=target_stage,
        shm_metadata={},
        is_done=is_done,
        error=error,
    )
    await _send_stream_control_message(control_plane, target_stage, target_endpoint, msg)
