# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for Qwen3.5-Omni."""

from __future__ import annotations

import importlib
import math
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.components.common import (
    load_qwen35_thinker_config,
)
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype
from sglang_omni.utils import instantiate_module

AUDIO_TOWER_PREFIX = ("thinker.audio_tower.", "audio_tower.")
_MODELING_MODULE_CANDIDATES = (
    "transformers.models.qwen3_omni_next.modeling_qwen3_omni_next",
    "sglang_omni.models.qwen3_5_omni.components.qwen3_omni_next_thinker",
)
_AUDIO_CLASS_CANDIDATES = ("Qwen3OmniNextAudioEncoder",)
DEFAULT_DOWNSAMPLE_TIMES = 4
DEFAULT_DOWNSAMPLE_CHUNK_SIZE = 100


def _resolve_next_class(class_names: tuple[str, ...]):
    for module_name in _MODELING_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for class_name in class_names:
            model_cls = getattr(module, class_name, None)
            if model_cls is not None:
                return model_cls
    raise ImportError(
        "Qwen3.5-Omni audio encoder requires Qwen3OmniNextAudioEncoder. "
        "Install a transformers build with qwen3_omni_next or vendor the "
        "Qwen3.5 thinker implementation under qwen3_5_omni.components."
    )


def _default_downsample_lengths(
    input_lengths: torch.Tensor,
    downsample_times: int = DEFAULT_DOWNSAMPLE_TIMES,
    chunk_size: int = DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
) -> torch.Tensor:
    input_lengths_leave = input_lengths % chunk_size
    for _ in range(downsample_times):
        input_lengths_leave = (input_lengths_leave - 1) // 2 + 1
    return input_lengths_leave + (input_lengths // chunk_size) * math.ceil(
        100 / 2**downsample_times
    )


def _get_int_config(config: Any, names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return int(default)


def _resolve_downsample_config(audio_cfg: Any) -> tuple[int, int]:
    return (
        _get_int_config(
            audio_cfg,
            ("downsample_times",),
            DEFAULT_DOWNSAMPLE_TIMES,
        ),
        _get_int_config(
            audio_cfg,
            ("downsample_chunk_size", "chunk_size"),
            DEFAULT_DOWNSAMPLE_CHUNK_SIZE,
        ),
    )


def _call_downsample_lengths(
    downsample_fn: Any,
    input_lengths: torch.Tensor,
    *,
    downsample_times: int,
    chunk_size: int,
) -> torch.Tensor:
    try:
        return downsample_fn(
            input_lengths,
            downsample_times=downsample_times,
            chunk_size=chunk_size,
        )
    except TypeError:
        return downsample_fn(input_lengths, downsample_times, chunk_size)


def _pack_audio_features(
    input_features: torch.Tensor,
    *,
    feature_attention_mask: torch.Tensor | None = None,
    audio_feature_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_attention_mask is not None:
        if input_features.ndim != 3:
            raise ValueError(
                "feature_attention_mask expects batched 3D audio features"
            )
        if feature_attention_mask.ndim != 2:
            raise ValueError("feature_attention_mask must be a 2D tensor")
        if input_features.shape[0] != feature_attention_mask.shape[0]:
            raise ValueError(
                "audio batch size does not match feature_attention_mask"
            )
        if input_features.shape[2] != feature_attention_mask.shape[1]:
            raise ValueError(
                "audio time dimension does not match feature_attention_mask"
            )
        feature_attention_mask = feature_attention_mask.to(
            device=input_features.device,
            dtype=torch.bool,
        )
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = (
            input_features.permute(0, 2, 1)[feature_attention_mask]
            .permute(1, 0)
            .contiguous()
        )
        return input_features, audio_feature_lengths.to(dtype=torch.long)

    if audio_feature_lengths is None:
        raise ValueError("audio_feature_lengths or feature_attention_mask is required")

    audio_feature_lengths = audio_feature_lengths.to(dtype=torch.long)
    if input_features.ndim == 2:
        return input_features.contiguous(), audio_feature_lengths
    if input_features.ndim != 3:
        raise ValueError("audio features must be a 2D packed or 3D batched tensor")
    if audio_feature_lengths.numel() != input_features.shape[0]:
        raise ValueError("audio_feature_lengths must have one value per audio sample")
    if torch.any(audio_feature_lengths < 0) or torch.any(
        audio_feature_lengths > input_features.shape[2]
    ):
        raise ValueError("audio_feature_lengths contains invalid frame counts")

    # 中文说明：部分 processor/cache 路径只给 lengths，不给 attention mask。
    # audio tower 需要的是按有效帧拼好的 [num_mel, total_frames]。
    frame_ids = torch.arange(input_features.shape[2], device=input_features.device)
    keep_frames = frame_ids.unsqueeze(0) < audio_feature_lengths.to(
        input_features.device
    ).unsqueeze(1)
    input_features = (
        input_features.permute(0, 2, 1)[keep_frames].permute(1, 0).contiguous()
    )
    return input_features, audio_feature_lengths


def _move_eval(
    module: nn.Module,
    *,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    module.eval()
    if torch_dtype is not None:
        return module.to(device=device, dtype=torch_dtype)
    return module.to(device=device)


def _split_packed_audio_qkv(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    for name, tensor in list(state_dict.items()):
        marker = "self_attn.qkv."
        if marker not in name:
            continue
        prefix, suffix = name.split(marker, 1)
        if suffix not in {"weight", "bias"}:
            continue
        if tensor.shape[0] % 3 != 0:
            raise ValueError(f"packed audio qkv tensor has invalid shape: {name}")
        q, k, v = tensor.chunk(3, dim=0)
        del normalized[name]
        for shard_name, shard in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
            normalized[f"{prefix}self_attn.{shard_name}.{suffix}"] = shard
    return normalized


def _load_audio_weights(
    audio_tower: nn.Module,
    model_path: str,
    *,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    state_dict = load_weights_by_prefix(model_path, prefix=AUDIO_TOWER_PREFIX)
    # 中文说明：兼容 vLLM native audio tower 保存的 packed qkv 权重；
    # 本地/HF-style tower 使用 q_proj/k_proj/v_proj 三组参数。
    state_dict = _split_packed_audio_qkv(state_dict)
    try:
        audio_tower.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        audio_tower.load_state_dict(state_dict, strict=True)
    return _move_eval(audio_tower, torch_dtype=torch_dtype, device=device)


def _build_audio_tower(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    audio_cfg = thinker_cfg.audio_config
    audio_cls = _resolve_next_class(_AUDIO_CLASS_CANDIDATES)
    audio_tower = instantiate_module(audio_cls, audio_cfg)
    return _load_audio_weights(
        audio_tower,
        model_path,
        device=device,
        torch_dtype=torch_dtype,
    )


class Qwen35OmniAudioEncoder(nn.Module):
    """Audio tower extracted from a Qwen3.5-Omni thinker checkpoint."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_qwen35_thinker_config(model_path)
        audio_cfg = thinker_cfg.audio_config
        (
            self._downsample_times,
            self._downsample_chunk_size,
        ) = _resolve_downsample_config(audio_cfg)
        self._device = torch.device(device)
        self.audio_tower = _build_audio_tower(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        try:
            module = importlib.import_module(
                "transformers.models.qwen3_omni_next.modeling_qwen3_omni_next"
            )
            self._downsample_lengths = getattr(
                module,
                "_get_feat_extract_output_lengths",
                _default_downsample_lengths,
            )
        except ImportError:
            self._downsample_lengths = _default_downsample_lengths

    def forward(
        self,
        *,
        input_features: torch.Tensor | None = None,
        input_audio_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_features is None:
            input_features = input_audio_features
        if input_features is None:
            raise ValueError("input_features or input_audio_features is required")

        input_features, audio_feature_lengths = _pack_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )

        audio_feature_lengths = audio_feature_lengths.to(self._device, dtype=torch.long)
        tower_dtype = getattr(self.audio_tower, "dtype", torch.bfloat16)
        # 中文说明：这里必须和 Qwen3OmniNextProcessor 的 audio_kwargs 对齐；
        # 否则 audio placeholder 数量和 audio tower 输出长度会在真实权重下错位。
        audio_output_lengths = _call_downsample_lengths(
            self._downsample_lengths,
            audio_feature_lengths,
            downsample_times=self._downsample_times,
            chunk_size=self._downsample_chunk_size,
        )
        outputs: Any = self.audio_tower(
            input_features.to(device=self._device, dtype=tower_dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        audio_embeds = getattr(outputs, "last_hidden_state", outputs)
        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }


Qwen3OmniNextAudioEncoder = Qwen35OmniAudioEncoder
