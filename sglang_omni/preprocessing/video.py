# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic video preprocessing utilities."""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import av
import librosa
import torch
from qwen_vl_utils import vision_process as qwen_vision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_f

from .base import MediaIO, _is_url
from .cache_key import compute_media_cache_key
from .resource_connector import global_thread_pool

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_PATCH_SIZE = 14
_DEFAULT_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES = 32
_DEFAULT_VIDEO_PREPROCESS_CACHE_MAX_BYTES = 4 * 1024**3


@dataclass
class _VideoPreprocessCacheEntry:
    value: tuple[Any, float, Any | None]
    size_bytes: int


_VIDEO_PREPROCESS_CACHE: OrderedDict[str, _VideoPreprocessCacheEntry] = OrderedDict()
_VIDEO_PREPROCESS_INFLIGHT: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Task]] = {}
_VIDEO_PREPROCESS_CACHE_LOCK = threading.Lock()
_VIDEO_PREPROCESS_CACHE_BYTES = 0


def _qwen_image_factor() -> int:
    image_factor = getattr(qwen_vision, "IMAGE_FACTOR", None)
    if image_factor is not None:
        return int(image_factor)
    return _DEFAULT_IMAGE_PATCH_SIZE * int(qwen_vision.SPATIAL_MERGE_SIZE)


def _qwen_video_min_pixels() -> int:
    min_pixels = getattr(qwen_vision, "VIDEO_MIN_PIXELS", None)
    if min_pixels is not None:
        return int(min_pixels)
    return int(qwen_vision.VIDEO_MIN_TOKEN_NUM) * _qwen_image_factor() ** 2


def _qwen_video_max_pixels() -> int:
    max_pixels = getattr(qwen_vision, "VIDEO_MAX_PIXELS", None)
    if max_pixels is not None:
        return int(max_pixels)
    return int(qwen_vision.VIDEO_MAX_TOKEN_NUM) * _qwen_image_factor() ** 2


def _qwen_video_total_pixels() -> int:
    total_pixels = getattr(qwen_vision, "VIDEO_TOTAL_PIXELS", None)
    if total_pixels is not None:
        return int(total_pixels)
    return int(qwen_vision.MODEL_SEQ_LEN * _qwen_image_factor() ** 2 * 0.9)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return 0
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _trace_video_preprocess_cache_enabled() -> bool:
    return _env_bool("SGLANG_OMNI_TRACE_VIDEO_PREPROCESS_CACHE", default=False)


def _short_cache_key(cache_key: str | None) -> str:
    if not cache_key:
        return "-"
    if len(cache_key) <= 32:
        return cache_key
    return f"{cache_key[:16]}...{cache_key[-8:]}"


def _trace_video_preprocess_cache(
    action: str,
    *,
    cache_key: str | None,
    path: str | Path | None = None,
    size_bytes: int | None = None,
    detail: str | None = None,
) -> None:
    if not _trace_video_preprocess_cache_enabled():
        return
    logger.info(
        "video_preprocess_cache action=%s key=%s path=%s size_bytes=%s detail=%s",
        action,
        _short_cache_key(cache_key),
        str(path) if path is not None else "-",
        size_bytes if size_bytes is not None else "-",
        detail or "-",
    )


def _video_preprocess_cache_max_entries() -> int:
    return _env_int(
        "SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES",
        _DEFAULT_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES,
    )


def _video_preprocess_cache_max_bytes() -> int:
    return _env_int(
        "SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES",
        _DEFAULT_VIDEO_PREPROCESS_CACHE_MAX_BYTES,
    )


def _value_size_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            return 0
    if isinstance(value, dict):
        return sum(_value_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_size_bytes(item) for item in value)
    return 0


def _video_result_size_bytes(value: tuple[Any, float, Any | None]) -> int:
    video, _, audio = value
    return _value_size_bytes(video) + _value_size_bytes(audio)


def _video_preprocess_cache_get(
    cache_key: str,
) -> tuple[Any, float, Any | None] | None:
    max_entries = _video_preprocess_cache_max_entries()
    max_bytes = _video_preprocess_cache_max_bytes()
    if max_entries <= 0 or max_bytes <= 0:
        return None
    with _VIDEO_PREPROCESS_CACHE_LOCK:
        entry = _VIDEO_PREPROCESS_CACHE.get(cache_key)
        if entry is None:
            _trace_video_preprocess_cache("miss", cache_key=cache_key)
            return None
        _VIDEO_PREPROCESS_CACHE.move_to_end(cache_key)
        _trace_video_preprocess_cache(
            "hit", cache_key=cache_key, size_bytes=entry.size_bytes
        )
        return entry.value


def _video_preprocess_cache_put(
    cache_key: str,
    value: tuple[Any, float, Any | None],
) -> None:
    global _VIDEO_PREPROCESS_CACHE_BYTES

    max_entries = _video_preprocess_cache_max_entries()
    max_bytes = _video_preprocess_cache_max_bytes()
    if max_entries <= 0 or max_bytes <= 0:
        return
    size_bytes = _video_result_size_bytes(value)
    if size_bytes > max_bytes:
        _trace_video_preprocess_cache(
            "skip_store_oversize",
            cache_key=cache_key,
            size_bytes=size_bytes,
            detail=f"max_bytes={max_bytes}",
        )
        return
    with _VIDEO_PREPROCESS_CACHE_LOCK:
        old_entry = _VIDEO_PREPROCESS_CACHE.pop(cache_key, None)
        if old_entry is not None:
            _VIDEO_PREPROCESS_CACHE_BYTES -= old_entry.size_bytes
        _VIDEO_PREPROCESS_CACHE[cache_key] = _VideoPreprocessCacheEntry(
            value=value,
            size_bytes=size_bytes,
        )
        _VIDEO_PREPROCESS_CACHE_BYTES += size_bytes
        _VIDEO_PREPROCESS_CACHE.move_to_end(cache_key)
        _trace_video_preprocess_cache(
            "store", cache_key=cache_key, size_bytes=size_bytes
        )
        while len(_VIDEO_PREPROCESS_CACHE) > max_entries:
            evicted_key, entry = _VIDEO_PREPROCESS_CACHE.popitem(last=False)
            _VIDEO_PREPROCESS_CACHE_BYTES -= entry.size_bytes
            _trace_video_preprocess_cache(
                "evict_entries",
                cache_key=evicted_key,
                size_bytes=entry.size_bytes,
                detail=f"max_entries={max_entries}",
            )
        while _VIDEO_PREPROCESS_CACHE_BYTES > max_bytes and _VIDEO_PREPROCESS_CACHE:
            evicted_key, entry = _VIDEO_PREPROCESS_CACHE.popitem(last=False)
            _VIDEO_PREPROCESS_CACHE_BYTES -= entry.size_bytes
            _trace_video_preprocess_cache(
                "evict_bytes",
                cache_key=evicted_key,
                size_bytes=entry.size_bytes,
                detail=f"max_bytes={max_bytes}",
            )


def clear_video_preprocess_cache() -> None:
    """Clear the local decoded-video preprocessing cache."""

    global _VIDEO_PREPROCESS_CACHE_BYTES
    with _VIDEO_PREPROCESS_CACHE_LOCK:
        _VIDEO_PREPROCESS_CACHE.clear()
        _VIDEO_PREPROCESS_INFLIGHT.clear()
        _VIDEO_PREPROCESS_CACHE_BYTES = 0


def _local_video_preprocess_cache_key(
    video_item: str | Path,
    *,
    fps: float | None,
    max_frames: int | None,
    min_frames: int | None,
    min_pixels: int | None,
    max_pixels: int | None,
    total_pixels: int | None,
    override_max_pixels: bool,
    extract_audio: bool,
    audio_target_sr: int,
    image_mode: str,
) -> str | None:
    cache_key = compute_video_cache_key(
        video_item,
        fps=fps,
        max_frames=max_frames,
        min_frames=min_frames,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
        override_max_pixels=override_max_pixels,
    )
    if cache_key is None:
        return None
    backend = qwen_vision.get_video_reader_backend()
    return (
        f"{cache_key}|backend={backend}|image_mode={image_mode}"
        f"|extract_audio={bool(extract_audio)}|audio_target_sr={int(audio_target_sr)}"
    )


class VideoDecodeError(RuntimeError):
    """Raised when video decoding fails."""


def _unpack_qwen_video_reader_output(output: Any) -> tuple[torch.Tensor, float]:
    """Normalize qwen-vl-utils video reader outputs across versions."""

    if not isinstance(output, tuple):
        raise ValueError(
            "Qwen video reader must return a tuple, "
            f"got {type(output).__name__}"
        )
    if len(output) == 2:
        video, sample_fps = output
    elif len(output) >= 3:
        video, _, sample_fps = output[:3]
    else:
        raise ValueError(
            "Qwen video reader must return at least 2 values, "
            f"got {len(output)}"
        )
    return video, float(sample_fps)


class VideoMediaIO(MediaIO[tuple[torch.Tensor, float, Any | None]]):
    """MediaIO implementation for video files with optional audio extraction."""

    def __init__(
        self,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        min_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        override_max_pixels: bool = False,
        image_mode: str = "RGB",
        extract_audio: bool = False,
        audio_target_sr: int = 16000,
        **kwargs,
    ) -> None:
        """Initialize VideoMediaIO.

        Args:
            fps: Target FPS for video loading.
            max_frames: Optional frame cap passed to the video reader backend.
            min_frames: Optional minimum frame count passed to the video reader backend.
            min_pixels: Optional lower resize budget per frame.
            max_pixels: Optional upper resize budget per frame.
            total_pixels: Optional total video pixel budget.
            override_max_pixels: If True, let ``total_pixels`` or explicit
                ``max_pixels`` override the default Qwen video max-pixel cap.
            image_mode: Target image mode (default: "RGB").
            extract_audio: If True, extract audio from video and return as third element.
            audio_target_sr: Target sample rate for audio extraction (default: 16000).
            **kwargs: Additional arguments (for compatibility with MultiModalResourceConnector).
        """
        super().__init__()
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.override_max_pixels = override_max_pixels
        self.image_mode = image_mode
        self.extract_audio = extract_audio
        self.audio_target_sr = audio_target_sr
        self.kwargs = kwargs

    def _load_path(self, filepath: Path) -> tuple[torch.Tensor, float]:
        return load_video_path(
            filepath,
            fps=self.fps,
            max_frames=self.max_frames,
            min_frames=self.min_frames,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            total_pixels=self.total_pixels,
            override_max_pixels=self.override_max_pixels,
        )

    def load_bytes(self, data: bytes) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from raw bytes, optionally extracting audio.

        Returns:
            Tuple of (video_tensor, sample_fps, audio_or_None).
            If extract_audio is False, the third element is None.
        """
        # qwen_vision._read_video_torchvision requires a file path,
        # so we need to write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(data)

        try:
            if self.extract_audio:
                # Load video and extract audio from the same file
                video, sample_fps = self._load_path(tmp_path)
                audio = _extract_audio_from_path(tmp_path, self.audio_target_sr)
                return video, sample_fps, audio
            else:
                video, sample_fps = self._load_path(tmp_path)
                return video, sample_fps, None
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from base64-encoded data, optionally extracting audio."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from a local file path, optionally extracting audio."""
        if self.extract_audio:
            # Load video and extract audio from the same file
            video, sample_fps = self._load_path(filepath)
            audio = _extract_audio_from_path(filepath, self.audio_target_sr)
            return video, sample_fps, audio
        else:
            video, sample_fps = self._load_path(filepath)
            return video, sample_fps, None


async def _load_local_video_uncached(
    video_path: Path,
    *,
    fps: float | None,
    max_frames: int | None,
    min_frames: int | None,
    min_pixels: int | None,
    max_pixels: int | None,
    total_pixels: int | None,
    override_max_pixels: bool,
    extract_audio: bool,
    audio_target_sr: int,
) -> tuple[Any, float, Any | None]:
    loop = asyncio.get_running_loop()
    if extract_audio:
        video_task = loop.run_in_executor(
            global_thread_pool,
            load_video_path,
            video_path,
            fps,
            max_frames,
            min_frames,
            min_pixels,
            max_pixels,
            total_pixels,
            override_max_pixels,
        )
        audio_task = loop.run_in_executor(
            global_thread_pool,
            _extract_audio_from_path,
            video_path,
            audio_target_sr,
        )
        (video, sample_fps), audio = await asyncio.gather(video_task, audio_task)
        return video, sample_fps, audio

    video, sample_fps = await loop.run_in_executor(
        global_thread_pool,
        load_video_path,
        video_path,
        fps,
        max_frames,
        min_frames,
        min_pixels,
        max_pixels,
        total_pixels,
        override_max_pixels,
    )
    return video, sample_fps, None


async def _load_local_video_with_cache(
    video_path: Path,
    *,
    fps: float | None,
    max_frames: int | None,
    min_frames: int | None,
    min_pixels: int | None,
    max_pixels: int | None,
    total_pixels: int | None,
    override_max_pixels: bool,
    image_mode: str,
    extract_audio: bool,
    audio_target_sr: int,
) -> tuple[Any, float, Any | None]:
    cache_key = _local_video_preprocess_cache_key(
        video_path,
        fps=fps,
        max_frames=max_frames,
        min_frames=min_frames,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
        override_max_pixels=override_max_pixels,
        extract_audio=extract_audio,
        audio_target_sr=audio_target_sr,
        image_mode=image_mode,
    )
    if cache_key is None:
        return await _load_local_video_uncached(
            video_path,
            fps=fps,
            max_frames=max_frames,
            min_frames=min_frames,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            override_max_pixels=override_max_pixels,
            extract_audio=extract_audio,
            audio_target_sr=audio_target_sr,
        )

    cached = _video_preprocess_cache_get(cache_key)
    if cached is not None:
        _trace_video_preprocess_cache("return_hit", cache_key=cache_key, path=video_path)
        return cached

    loop = asyncio.get_running_loop()
    is_leader = False
    with _VIDEO_PREPROCESS_CACHE_LOCK:
        inflight = _VIDEO_PREPROCESS_INFLIGHT.get(cache_key)
        if inflight is not None and inflight[0] is loop and not inflight[1].done():
            task = inflight[1]
            _trace_video_preprocess_cache(
                "wait_inflight", cache_key=cache_key, path=video_path
            )
        else:
            task = loop.create_task(
                _load_local_video_uncached(
                    video_path,
                    fps=fps,
                    max_frames=max_frames,
                    min_frames=min_frames,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    total_pixels=total_pixels,
                    override_max_pixels=override_max_pixels,
                    extract_audio=extract_audio,
                    audio_target_sr=audio_target_sr,
                )
            )
            _VIDEO_PREPROCESS_INFLIGHT[cache_key] = (loop, task)
            is_leader = True
            _trace_video_preprocess_cache(
                "decode_start", cache_key=cache_key, path=video_path
            )

    try:
        result = await asyncio.shield(task)
    except Exception:
        if is_leader:
            with _VIDEO_PREPROCESS_CACHE_LOCK:
                if _VIDEO_PREPROCESS_INFLIGHT.get(cache_key) == (loop, task):
                    _VIDEO_PREPROCESS_INFLIGHT.pop(cache_key, None)
        raise

    if is_leader:
        _video_preprocess_cache_put(cache_key, result)
        with _VIDEO_PREPROCESS_CACHE_LOCK:
            if _VIDEO_PREPROCESS_INFLIGHT.get(cache_key) == (loop, task):
                _VIDEO_PREPROCESS_INFLIGHT.pop(cache_key, None)
    return result


async def ensure_video_list_async(
    videos: Any,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    min_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    override_max_pixels: bool = False,
    image_mode: str = "RGB",
    resource_connector: Any | None = None,
    extract_audio: bool | list[bool] | tuple[bool, ...] = False,
    audio_target_sr: int = 16000,
) -> tuple[list[Any], list[float] | None, list[Any] | None]:
    """Asynchronously normalize video inputs into a list.

    Args:
        videos: Video input(s) - can be a path, URL, torch Tensor, or list.
        fps: Target FPS for video loading.
        max_frames: Optional frame cap passed to the video reader backend.
        min_frames: Optional minimum frame count passed to the video reader backend.
        min_pixels: Optional lower resize budget per frame.
        max_pixels: Optional upper resize budget per frame.
        total_pixels: Optional total video pixel budget.
        override_max_pixels: If True, let ``total_pixels`` or explicit
            ``max_pixels`` override the default Qwen video max-pixel cap.
        image_mode: Target image mode (default: "RGB").
        resource_connector: Optional MultiModalResourceConnector instance. If None, uses
                        the global connector.
        extract_audio: If True, extract audio from videos and return as third element.
                    A bool list enables per-video extraction.
        audio_target_sr: Target sample rate for audio extraction (default: 16000).

    Returns:
        Tuple of (normalized video list, sample_fps_list or None, extracted_audio_list or None).
        If extract_audio is False, the third element is None.
    """
    if videos is None:
        return [], None, None
    if isinstance(videos, list):
        items = videos
    else:
        items = [videos]
    extract_audio_flags = _normalize_extract_audio_flags(extract_audio, len(items))
    should_return_audio = any(extract_audio_flags)
    normalized: list[Any] = []
    sample_fps_list: list[float] = []
    extracted_audios: list[Any] = [] if should_return_audio else []
    all_paths = True

    # Import here to avoid circular dependency
    if resource_connector is None:
        from .resource_connector import get_global_resource_connector

        resource_connector = get_global_resource_connector()

    async def _load_video_with_audio(
        video_item: str | Path,
        *,
        is_url: bool,
        extract_audio_for_item: bool,
    ) -> tuple[Any, float, Any | None]:
        """Load video and optionally extract audio."""
        if is_url:
            # Use fetch_video_async for URL videos, similar to fetch_image_async
            return await resource_connector.fetch_video_async(
                str(video_item),
                fps=fps,
                max_frames=max_frames,
                min_frames=min_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                total_pixels=total_pixels,
                override_max_pixels=override_max_pixels,
                image_mode=image_mode,
                extract_audio=extract_audio_for_item,
                audio_target_sr=audio_target_sr,
            )
        else:
            # Local file path
            return await _load_local_video_with_cache(
                Path(video_item),
                fps=fps,
                max_frames=max_frames,
                min_frames=min_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                total_pixels=total_pixels,
                override_max_pixels=override_max_pixels,
                image_mode=image_mode,
                extract_audio=extract_audio_for_item,
                audio_target_sr=audio_target_sr,
            )

    # Collect coroutines for URL and local file items
    coroutines: list[asyncio.Task[tuple[Any, float, Any | None]] | None] = []
    url_indices: list[int] = []

    # First pass: identify items that need loading
    for idx, video_item in enumerate(items):
        extract_audio_for_item = extract_audio_flags[idx]
        if isinstance(video_item, (str, Path)):
            if _is_url(video_item):
                # Create coroutine for async URL fetching with optional audio extraction
                coro = _load_video_with_audio(
                    video_item,
                    is_url=True,
                    extract_audio_for_item=extract_audio_for_item,
                )
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder for video
                sample_fps_list.append(0.0)  # Placeholder for fps
                if should_return_audio:
                    extracted_audios.append(None)  # Placeholder for audio
            elif Path(video_item).exists():
                # Load from local path with optional audio extraction
                coro = _load_video_with_audio(
                    video_item,
                    is_url=False,
                    extract_audio_for_item=extract_audio_for_item,
                )
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder for video
                sample_fps_list.append(0.0)  # Placeholder for fps
                if should_return_audio:
                    extracted_audios.append(None)  # Placeholder for audio
            else:
                # Path doesn't exist, treat as already processed
                normalized.append(video_item)
                all_paths = False
                if should_return_audio:
                    extracted_audios.append(None)
        else:
            # Already processed (torch Tensor, etc.)
            normalized.append(video_item)
            all_paths = False
            if should_return_audio:
                extracted_audios.append(None)

    # Wait for all loads to complete
    if coroutines:
        results = await asyncio.gather(*coroutines)
        # Fill in the results at the correct indices
        for url_idx, (video, sample_fps, audio) in zip(url_indices, results):
            normalized[url_idx] = video
            sample_fps_list[url_idx] = sample_fps
            if should_return_audio:
                extracted_audios[url_idx] = audio

    if all_paths:
        return (
            normalized,
            sample_fps_list,
            extracted_audios if should_return_audio else None,
        )
    return normalized, None, extracted_audios if should_return_audio else None


def _normalize_extract_audio_flags(
    extract_audio: bool | list[bool] | tuple[bool, ...],
    item_count: int,
) -> list[bool]:
    if isinstance(extract_audio, (list, tuple)):
        flags = [bool(item) for item in extract_audio[:item_count]]
        if len(flags) < item_count:
            flags.extend([False] * (item_count - len(flags)))
        return flags
    return [bool(extract_audio)] * item_count


def _extract_audio_from_path(video_path: Path, target_sr: int) -> Any | None:
    """Extract audio from a video file path."""
    if not _check_if_video_has_audio(video_path):
        return None
    try:
        audio, _ = librosa.load(str(video_path), sr=target_sr)
        return audio
    except Exception as e:
        logger.debug(f"Failed to extract audio from {video_path}: {e}")
        return None


def load_video_path(
    path: str | Path,
    fps: float | None = None,
    max_frames: int | None = None,
    min_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    override_max_pixels: bool = False,
) -> tuple[torch.Tensor, float]:
    """Load a local video into a torch tensor (T, C, H, W) on CPU."""
    path = Path(path)
    ele: dict[str, Any] = {"video": str(path)}
    if fps is not None:
        ele["fps"] = float(fps)
    if max_frames is not None:
        ele["max_frames"] = int(max_frames)
    if min_frames is not None:
        ele["min_frames"] = int(min_frames)
    if min_pixels is not None:
        ele["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        ele["max_pixels"] = int(max_pixels)
    if total_pixels is not None:
        ele["total_pixels"] = int(total_pixels)
    backend = qwen_vision.get_video_reader_backend()
    try:
        video, sample_fps = _unpack_qwen_video_reader_output(
            qwen_vision.VIDEO_READER_BACKENDS[backend](ele)
        )
    except Exception as backend_exc:
        if backend == "torchvision":
            raise VideoDecodeError(
                f"Failed to decode video path={path}; torchvision failed with "
                f"{type(backend_exc).__name__}: {backend_exc}"
            ) from backend_exc
        logger.warning("Video reader %s failed, falling back to torchvision", backend)
        try:
            video, sample_fps = _unpack_qwen_video_reader_output(
                qwen_vision.VIDEO_READER_BACKENDS["torchvision"](ele)
            )
        except Exception as fallback_exc:
            raise VideoDecodeError(
                f"Failed to decode video path={path}; {backend} failed with "
                f"{type(backend_exc).__name__}: {backend_exc}; "
                f"torchvision failed with {type(fallback_exc).__name__}: "
                f"{fallback_exc}"
            ) from fallback_exc
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", _qwen_video_min_pixels())
    total_pixels = ele.get("total_pixels", _qwen_video_total_pixels())
    max_pixels_from_total = total_pixels / nframes * qwen_vision.FRAME_FACTOR
    if not override_max_pixels:
        max_pixels_from_total = min(
            _qwen_video_max_pixels(),
            max_pixels_from_total,
        )
    max_pixels = max(
        max_pixels_from_total,
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels")
    if max_pixels_supposed is not None:
        max_pixels = (
            int(max_pixels_supposed)
            if override_max_pixels
            else min(int(max_pixels_supposed), max_pixels)
        )
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = qwen_vision.smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=_qwen_image_factor(),
        )
    else:
        resized_height, resized_width = qwen_vision.smart_resize(
            height,
            width,
            factor=_qwen_image_factor(),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = tv_f.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video, sample_fps


def build_video_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "pixel_values_videos": hf_inputs.get("pixel_values_videos"),
        "video_grid_thw": hf_inputs.get("video_grid_thw"),
        "video_second_per_grid": hf_inputs.get("video_second_per_grid"),
        "video_item_pixel_present": hf_inputs.get("video_item_pixel_present"),
    }


def compute_video_cache_key(
    videos: Any,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    min_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    override_max_pixels: bool = False,
) -> str | None:
    """Compute cache key from raw video inputs + effective decode params.

    Decode params change the resulting frame count and thus the encoder
    output length. They must be part of the cache key — otherwise an entry
    produced under one (fps, max_frames, pixel-limit) tuple could be
    returned for a request with different params, yielding ``video_embeds``
    whose length no longer matches the prompt placeholders.
    """
    base = compute_media_cache_key(videos, prefix="video")
    if base is None:
        return None
    decode_sig = (
        f"|fps={fps}|max_frames={max_frames}|min_frames={min_frames}"
        f"|min_px={min_pixels}|max_px={max_pixels}|total_px={total_pixels}"
        f"|override_max_px={override_max_pixels}"
    )
    return base + decode_sig


def derive_video_total_pixels_from_mm_len(max_mm_len: int) -> int:
    """Convert a Qwen-style multimodal token budget to a 3-D pixel budget."""

    if max_mm_len <= 0:
        raise ValueError("max_mm_len must be positive")
    # Qwen video token count is roughly the 3D pixel budget divided by
    # IMAGE_FACTOR^2. Qwen3-VL effective max pixels uses the same unit^2
    # conversion, and the Omni loader then spreads it across frames with
    # FRAME_FACTOR to get per-frame max_pixels.
    return int(max_mm_len) * _qwen_image_factor() ** 2


def _check_if_video_has_audio(video_path: str | Path) -> bool:
    try:
        container = av.open(str(video_path))
        audio_streams = [
            stream for stream in container.streams if stream.type == "audio"
        ]
        container.close()
        return len(audio_streams) > 0
    except Exception as e:
        logger.debug(f"Failed to check audio in video {video_path}: {e}")
        return False
