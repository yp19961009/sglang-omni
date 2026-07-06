# SPDX-License-Identifier: Apache-2.0
"""Image/video encoder component for Qwen3.5-Omni."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.image_encoder import _optimize_patch_embed
from sglang_omni.models.qwen35_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import load_weights_by_prefix, resolve_dtype
from sglang_omni.utils import instantiate_module

VISUAL_PREFIX = ("thinker.visual.", "visual.")
VISUAL_CLASS = hf_modeling.Qwen3OmniMoeVisionEncoder


def _unpack_visual_outputs(outputs: object) -> tuple[torch.Tensor, object | None]:
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output, getattr(outputs, "deepstack_features", None)
    if isinstance(outputs, tuple):
        embeds = outputs[0]
        deepstack = outputs[1] if len(outputs) > 1 else None
        return embeds, deepstack
    raise TypeError(f"Unsupported visual encoder output type: {type(outputs)!r}")


def _remap_vision_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _build_visual(
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


class Qwen35OmniImageEncoder(nn.Module):
    """Qwen3.5-Omni vision tower extracted as an encoder stage."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_path)
        vision_cfg = thinker_cfg.vision_config
        self._device = torch.device(device)
        self.visual = _build_visual(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.spatial_merge_size = int(vision_cfg.spatial_merge_size)
        self.out_hidden_size = int(vision_cfg.out_hidden_size)
        self.deepstack_layers = 0
        self.visual_dtype_bytes = torch.empty((), dtype=self.visual.dtype).element_size()

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

        if isinstance(pixel_values, torch.Tensor) and isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
            pixel_values = pixel_values.to(device=self._device, dtype=self.visual.dtype)
            vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds, image_multiscale = _unpack_visual_outputs(vision_outputs)
            image_counts = image_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_token_counts": image_counts.to(device=self._device),
                }
            )

        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.to(self._device, dtype=torch.long)
            pixel_values_videos = pixel_values_videos.to(
                device=self._device, dtype=self.visual.dtype
            )
            video_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_embeds, video_multiscale = _unpack_visual_outputs(video_outputs)
            video_counts = video_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "video_embeds": video_embeds,
                    "video_grid_thw": video_grid_thw,
                    "video_token_counts": video_counts.to(device=self._device),
                }
            )

        return outputs
