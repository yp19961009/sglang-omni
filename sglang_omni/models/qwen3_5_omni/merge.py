# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni payload merge helpers."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.qwen3_omni import merge as qwen3_merge
from sglang_omni.models.qwen3_omni.merge import *  # noqa: F401,F403
from sglang_omni.models.qwen3_5_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto import StagePayload


def merge_for_thinker(payloads: dict[str, StagePayload]) -> StagePayload:
    """Aggregate encoder outputs and preserve Qwen3.5 video-audio slot metadata."""

    audio_is_dependent, use_audio_in_video = _extract_qwen35_video_audio_metadata(
        payloads
    )
    merged = qwen3_merge.merge_for_thinker(payloads)
    if audio_is_dependent is None and use_audio_in_video is None:
        return merged

    state = Qwen3OmniPipelineState.from_dict(merged.data)
    thinker_inputs = dict(state.thinker_inputs or {})
    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if audio_is_dependent is not None:
        # 中文说明：对齐 vLLM perf_v2 的 audio_is_dependent 语义：
        # audio feature 列表中 True 表示该槽位来自 video 内置音轨。
        model_inputs["audio_is_dependent"] = _to_bool_tensor(audio_is_dependent)
    if use_audio_in_video is not None:
        # 中文说明：Qwen3.5 支持 per-video audio flags。base Qwen3 merge
        # 只保留标量 True；这里把 [False, True] 这类细粒度配置继续传下去。
        model_inputs["use_audio_in_video"] = _normalize_use_audio_in_video(
            use_audio_in_video
        )
    thinker_inputs["model_inputs"] = model_inputs
    state.thinker_inputs = thinker_inputs

    if isinstance(state.mm_inputs, dict):
        mm_inputs = dict(state.mm_inputs)
        if audio_is_dependent is not None:
            audio_inputs = dict(mm_inputs.get("audio", {}))
            audio_inputs["audio_is_dependent"] = audio_is_dependent
            mm_inputs["audio"] = audio_inputs
        if use_audio_in_video is not None:
            video_inputs = dict(mm_inputs.get("video", {}))
            video_inputs["use_audio_in_video"] = _normalize_use_audio_in_video(
                use_audio_in_video
            )
            mm_inputs["video"] = video_inputs
        state.mm_inputs = mm_inputs

    merged.data = state.to_dict()
    return merged


def _extract_qwen35_video_audio_metadata(
    payloads: dict[str, StagePayload],
) -> tuple[Any | None, Any | None]:
    if not payloads:
        return None, None
    base = payloads.get("preprocessing") or next(iter(payloads.values()))
    state = Qwen3OmniPipelineState.from_dict(base.data)
    if not isinstance(state.mm_inputs, dict):
        return None, None
    audio_inputs = state.mm_inputs.get("audio", {})
    video_inputs = state.mm_inputs.get("video", {})
    audio_is_dependent = (
        audio_inputs.get("audio_is_dependent")
        if isinstance(audio_inputs, dict)
        else None
    )
    use_audio_in_video = (
        video_inputs.get("use_audio_in_video")
        if isinstance(video_inputs, dict)
        else None
    )
    return audio_is_dependent, use_audio_in_video


def _to_bool_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.bool).reshape(-1)
    if isinstance(value, (list, tuple)):
        return torch.tensor([bool(item) for item in value], dtype=torch.bool)
    return torch.tensor([bool(value)], dtype=torch.bool)


def _normalize_use_audio_in_video(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        values = value.detach().to(dtype=torch.bool).reshape(-1).tolist()
        return bool(values[0]) if len(values) == 1 else [bool(item) for item in values]
    if isinstance(value, (list, tuple)):
        normalized = [bool(item) for item in value]
        return normalized[0] if len(normalized) == 1 else normalized
    return bool(value)
