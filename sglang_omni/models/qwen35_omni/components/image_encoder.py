# SPDX-License-Identifier: Apache-2.0
"""Image/video encoder component for Qwen3.5-Omni."""

from __future__ import annotations

import copy
import logging
import threading
from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.image_encoder import _optimize_patch_embed
from sglang_omni.models.qwen35_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype
from sglang_omni.utils import instantiate_module

VISUAL_PREFIX = ("thinker.visual.", "visual.")
VISUAL_CLASS = hf_modeling.Qwen3OmniMoeVisionEncoder
VISUAL_BACKENDS = frozenset({"hf", "sglang"})
NATIVE_VISUAL_ATTENTION_BACKENDS = frozenset({"sdpa", "sdpa_grouped"})

logger = logging.getLogger(__name__)


def _normalize_visual_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in VISUAL_BACKENDS:
        raise ValueError(
            f"Unsupported Qwen3.5 vision backend {backend!r}; "
            f"expected one of {sorted(VISUAL_BACKENDS)}"
        )
    return normalized


def _normalize_native_visual_attention_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in NATIVE_VISUAL_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported Qwen3.5 native vision attention backend {backend!r}; "
            f"expected one of {sorted(NATIVE_VISUAL_ATTENTION_BACKENDS)}"
        )
    return normalized


def _unpack_visual_outputs(outputs: object) -> tuple[torch.Tensor, object | None]:
    if isinstance(outputs, torch.Tensor):
        return outputs, None
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output, getattr(outputs, "deepstack_features", None)
    if isinstance(outputs, tuple):
        embeds = outputs[0]
        deepstack = outputs[1] if len(outputs) > 1 else None
        return embeds, deepstack
    raise TypeError(f"Unsupported visual encoder output type: {type(outputs)!r}")


def _remap_vision_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        mapped = name
        if mapped.startswith("merger.norm."):
            mapped = mapped.replace("merger.norm.", "merger.ln_q.", 1)
        elif mapped.startswith("merger.linear_fc1."):
            mapped = mapped.replace("merger.linear_fc1.", "merger.mlp.0.", 1)
        elif mapped.startswith("merger.linear_fc2."):
            mapped = mapped.replace("merger.linear_fc2.", "merger.mlp.2.", 1)
        remapped[mapped] = tensor
    return remapped


def _build_hf_visual(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    vision_config = copy.deepcopy(thinker_cfg.vision_config)
    vision_config.deepstack_visual_indexes = []
    visual = instantiate_module(VISUAL_CLASS, vision_config)
    state_dict = load_weights_by_prefix(
        model_path,
        prefix=VISUAL_PREFIX,
        local_files_only=True,
    )
    remapped = {
        name: tensor
        for name, tensor in _remap_vision_state_dict(state_dict).items()
        if not name.startswith(("merger_list.", "deepstack_merger_list."))
    }
    visual.load_state_dict(remapped, strict=True)
    visual.eval()
    if torch_dtype is not None:
        visual = visual.to(dtype=torch_dtype)
    visual = visual.to(device=device)
    _optimize_patch_embed(visual)
    return visual


def _hf_split_sdpa(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bsz: int,
    cu_seqlens: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    """Match HF's per-grid SDPA calls instead of using a boolean block mask."""
    if bsz != 1:
        raise ValueError(
            f"Qwen3.5 vision SDPA expects flattened batch size 1, got {bsz}"
        )
    boundaries = [0, q.shape[0]] if cu_seqlens is None else cu_seqlens.cpu().tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q_part, k_part, v_part = [
            tensor[start:end].transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)
        ]
        output = torch.nn.functional.scaled_dot_product_attention(
            q_part,
            k_part,
            v_part,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False,
            scale=self.scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _hf_grouped_sdpa(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bsz: int,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """Batch equal-length temporal grids while retaining HF SDPA semantics."""
    if bsz != 1 or cu_seqlens is None:
        return _hf_split_sdpa(
            self,
            q,
            k,
            v,
            bsz=bsz,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

    boundaries = cu_seqlens.cpu().tolist()
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if not lengths or any(length != lengths[0] for length in lengths):
        return _hf_split_sdpa(
            self,
            q,
            k,
            v,
            bsz=bsz,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

    grid_count = len(lengths)
    grid_length = lengths[0]
    q_batch, k_batch, v_batch = [
        tensor.reshape(grid_count, grid_length, *tensor.shape[1:]).transpose(1, 2)
        for tensor in (q, k, v)
    ]
    output = torch.nn.functional.scaled_dot_product_attention(
        q_batch,
        k_batch,
        v_batch,
        attn_mask=None,
        dropout_p=self.dropout,
        is_causal=False,
        scale=self.scale,
    )
    return output.transpose(1, 2).reshape_as(q)


def _build_sglang_visual(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype,
    device: str,
    attention_backend: str,
) -> nn.Module:
    from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
    from sglang.srt.server_args import get_global_server_args

    from sglang_omni.model_runner._sglang_qwen3_vl_patches import (
        apply_qwen3_vl_hf_parity_patches,
    )

    apply_qwen3_vl_hf_parity_patches()
    vision_config = Qwen3VLVisionConfig(**thinker_cfg.vision_config.to_dict())
    server_args = get_global_server_args()
    previous_mm_attention_backend = server_args.mm_attention_backend
    server_args.mm_attention_backend = (
        "sdpa" if attention_backend == "sdpa_grouped" else attention_backend
    )
    try:
        visual = Qwen3VLMoeVisionModel(
            vision_config,
            norm_eps=float(getattr(thinker_cfg.text_config, "rms_norm_eps", 1e-6)),
            use_data_parallel=True,
        )
    finally:
        server_args.mm_attention_backend = previous_mm_attention_backend

    visual.config = thinker_cfg.vision_config
    visual.fast_pos_embed_interpolate = MethodType(
        VISUAL_CLASS.fast_pos_embed_interpolate, visual
    )
    if attention_backend in {"sdpa", "sdpa_grouped"}:
        forward = (
            _hf_grouped_sdpa if attention_backend == "sdpa_grouped" else _hf_split_sdpa
        )
        for block in visual.blocks:
            block.attn.qkv_backend.forward = MethodType(forward, block.attn.qkv_backend)
    state_dict = load_weights_by_prefix(
        model_path,
        prefix=VISUAL_PREFIX,
        local_files_only=True,
    )
    state_dict = {
        name.replace(".attn.qkv.", ".attn.qkv_proj."): tensor
        for name, tensor in state_dict.items()
    }
    visual.load_state_dict(state_dict, strict=True)
    visual.eval()
    visual = visual.to(device=device, dtype=torch_dtype)
    # Native dtype/device properties still reference patch_embed.proj.
    _optimize_patch_embed(visual, keep_conv=True)
    return visual


class Qwen35OmniImageEncoder(nn.Module):
    """Qwen3.5-Omni vision tower extracted as an encoder stage."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        backend: str = "hf",
        attention_backend: str = "sdpa_grouped",
    ) -> None:
        super().__init__()
        self._backend = _normalize_visual_backend(backend)
        self._attention_backend = _normalize_native_visual_attention_backend(
            attention_backend
        )
        self._model_path = model_path
        self._torch_dtype = resolve_dtype(dtype) or torch.bfloat16
        self._thinker_cfg = load_thinker_config(model_path)
        vision_cfg = self._thinker_cfg.vision_config
        self._device = torch.device(device)
        self._visual_lock = threading.Lock()
        self.visual = (
            _build_hf_visual(
                model_path,
                thinker_cfg=self._thinker_cfg,
                torch_dtype=self._torch_dtype,
                device=device,
            )
            if self._backend == "hf"
            else None
        )
        self.spatial_merge_size = int(vision_cfg.spatial_merge_size)
        self.out_hidden_size = int(vision_cfg.out_hidden_size)
        self.deepstack_layers = 0
        self.visual_dtype_bytes = torch.empty(
            (), dtype=self._torch_dtype
        ).element_size()
        logger.info(
            "Qwen3.5 image encoder backend=%s attention_backend=%s",
            self._backend,
            self._attention_backend,
        )

    def _ensure_visual(self) -> nn.Module:
        if self.visual is not None:
            return self.visual
        with self._visual_lock:
            if self.visual is None:
                self.visual = _build_sglang_visual(
                    self._model_path,
                    thinker_cfg=self._thinker_cfg,
                    torch_dtype=self._torch_dtype,
                    device=str(self._device),
                    attention_backend=self._attention_backend,
                )
        return self.visual

    def _encode_pixels(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, object | None, torch.Tensor]:
        visual = self._ensure_visual()
        pixel_values = pixel_values.to(device=self._device, dtype=self._torch_dtype)
        output_grid = grid_thw.to(self._device, dtype=torch.long)
        visual_grid = (
            grid_thw.to(device="cpu", dtype=torch.int32)
            if self._backend == "sglang"
            else output_grid
        )
        visual_outputs = visual(pixel_values, grid_thw=visual_grid)
        embeds, multiscale = _unpack_visual_outputs(visual_outputs)
        return embeds, multiscale, output_grid

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        merge = self.spatial_merge_size**2

        if isinstance(pixel_values, torch.Tensor) and isinstance(
            image_grid_thw, torch.Tensor
        ):
            image_embeds, image_multiscale, image_grid_thw = self._encode_pixels(
                pixel_values, image_grid_thw
            )
            image_counts = image_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_token_counts": image_counts.to(device=self._device),
                }
            )

        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
            video_grid_thw, torch.Tensor
        ):
            video_embeds, video_multiscale, video_grid_thw = self._encode_pixels(
                pixel_values_videos, video_grid_thw
            )
            video_counts = video_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "video_embeds": video_embeds,
                    "video_grid_thw": video_grid_thw,
                    "video_token_counts": video_counts.to(device=self._device),
                }
            )

        return outputs
