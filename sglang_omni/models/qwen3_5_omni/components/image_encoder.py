# SPDX-License-Identifier: Apache-2.0
"""Image/video encoder component for Qwen3.5-Omni."""

from __future__ import annotations

import importlib
import inspect
import logging
import types
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.components.common import (
    load_qwen35_thinker_config,
)
from sglang_omni.models.weight_loader import (
    load_module,
    load_weights_by_prefix,
    resolve_dtype,
)
from sglang_omni.utils import instantiate_module

logger = logging.getLogger(__name__)

VISUAL_PREFIX = ("thinker.visual.", "visual.")
_MODELING_MODULE_CANDIDATES = (
    "transformers.models.qwen3_omni_next.modeling_qwen3_omni_next",
    "sglang_omni.models.qwen3_5_omni.components.qwen3_omni_next_thinker",
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "sglang.srt.models.qwen3_vl",
)
_VISION_CLASS_CANDIDATES = (
    "Qwen3OmniNextVisionEncoder",
    "Qwen3OmniNextVisionTransformer",
    "Qwen3VLVisionModel",
    "Qwen3_VisionTransformer",
    "Qwen3VLMoeVisionModel",
)


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
        "Qwen3.5-Omni vision encoder requires a Qwen3OmniNext vision class. "
        "Install a transformers build with qwen3_omni_next or vendor the "
        "Qwen3.5 thinker implementation under qwen3_5_omni.components."
    )


def _module_dtype(module: nn.Module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    param = next(module.parameters(), None)
    return param.dtype if param is not None else torch.bfloat16


def _patch_embed_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.linear(hidden_states.to(dtype=self.linear.weight.dtype))


def _optimize_patch_embed(visual: nn.Module) -> None:
    """Replace non-sliding Conv3d patch embed with an equivalent Linear."""
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        return
    conv = getattr(patch_embed, "proj", None)
    if conv is None or not isinstance(conv, nn.Conv3d):
        return
    if list(conv.kernel_size) != list(conv.stride):
        return
    if conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1) or conv.groups != 1:
        return

    embed_dim = conv.out_channels
    in_features = (
        conv.in_channels
        * conv.kernel_size[0]
        * conv.kernel_size[1]
        * conv.kernel_size[2]
    )
    linear = nn.Linear(
        in_features,
        embed_dim,
        bias=conv.bias is not None,
        dtype=conv.weight.dtype,
        device=conv.weight.device,
    )
    with torch.no_grad():
        linear.weight.copy_(conv.weight.view(embed_dim, -1))
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)

    # 中文说明：不要删除原始 Conv3d。SGLang core vision 的 dtype/device
    # property 会访问 patch_embed.proj.weight；forward 改走 Linear 即可。
    patch_embed.linear = linear
    patch_embed.forward = types.MethodType(_patch_embed_forward, patch_embed)
    logger.info("Qwen3.5 PatchEmbed Conv3d optimized to Linear")


def _split_visual_output(outputs: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(outputs, tuple):
        embeds = outputs[0]
        multiscale = outputs[1] if len(outputs) > 1 else []
        return embeds, multiscale
    if isinstance(outputs, dict):
        embeds = outputs.get("last_hidden_state")
        if embeds is None:
            embeds = outputs.get("hidden_states")
        multiscale = outputs.get("deepstack_visual_embeds")
        if multiscale is None:
            multiscale = []
        return embeds, multiscale
    embeds = getattr(outputs, "last_hidden_state", outputs)
    multiscale = getattr(outputs, "deepstack_visual_embeds", [])
    return embeds, multiscale


def _has_deepstack(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, (list, tuple)):
        return any(_has_deepstack(item) for item in value)
    return True


def _split_packed_deepstack_output(
    embeds: torch.Tensor,
    multiscale: Any,
    *,
    base_hidden_size: int,
    deepstack_layers: int,
) -> tuple[torch.Tensor, Any]:
    if (
        not isinstance(embeds, torch.Tensor)
        or _has_deepstack(multiscale)
        or base_hidden_size <= 0
        or deepstack_layers <= 0
    ):
        return embeds, multiscale

    expected = base_hidden_size * (deepstack_layers + 1)
    if embeds.shape[-1] != expected:
        return embeds, multiscale

    # 中文说明：SGLang/vLLM 的 Qwen3_VisionTransformer 会把主视觉特征
    # 和 deepstack 多尺度特征拼在最后一维返回。分离后，主特征继续作为
    # image/video_embeds，deepstack 层按 request builder 需要的 list 透传。
    chunks = torch.split(embeds, base_hidden_size, dim=-1)
    return chunks[0], list(chunks[1:])


def _call_visual(
    visual: nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> Any:
    try:
        signature = inspect.signature(visual.forward)
    except (TypeError, ValueError):
        return visual(pixel_values, grid_thw)

    parameters = signature.parameters
    if "grid_thw" in parameters:
        return visual(pixel_values, grid_thw=grid_thw)
    if "image_grid_thw" in parameters:
        return visual(pixel_values, image_grid_thw=grid_thw)
    return visual(pixel_values, grid_thw)


def _first_int_attr(source: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return int(value)
    return None


def _resolve_vision_hidden_size(vision_cfg: Any, visual: nn.Module) -> int:
    for source in (vision_cfg, getattr(visual, "config", None), visual):
        value = _first_int_attr(
            source,
            ("out_hidden_size", "hidden_size", "embed_dim"),
        )
        if value is not None:
            return value
    return 0


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


def _load_visual_weights(
    visual: nn.Module,
    model_path: str,
    *,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    load_weights = getattr(visual, "load_weights", None)
    if callable(load_weights):
        # 中文说明：SGLang/vLLM vision tower 会把 HF 的 attn.q/k/v 权重
        # 映射到 packed attn.qkv 参数。直接 load_state_dict 会绕过这个映射。
        state_dict = load_weights_by_prefix(model_path, prefix=VISUAL_PREFIX)
        load_weights(state_dict.items())
        return _move_eval(visual, torch_dtype=torch_dtype, device=device)

    return load_module(
        visual,
        model_path,
        prefix=VISUAL_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )


def _ensure_pretrained_config(vision_cfg):
    """Wrap SimpleNamespace into Qwen3VLVisionConfig for HF PretrainedModel."""
    from transformers import PretrainedConfig

    if isinstance(vision_cfg, PretrainedConfig):
        return vision_cfg
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    attrs = {
        k: v
        for k, v in vars(vision_cfg).items()
        if not (k in ("label2id", "id2label") and not isinstance(v, dict))
    }
    return Qwen3VLVisionConfig(**attrs)


def _build_visual(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    vision_cfg = thinker_cfg.vision_config
    vision_cfg = _ensure_pretrained_config(vision_cfg)
    visual_cls = _resolve_next_class(_VISION_CLASS_CANDIDATES)
    visual = instantiate_module(visual_cls, vision_cfg)
    visual = _load_visual_weights(
        visual,
        model_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    _optimize_patch_embed(visual)
    return visual


class Qwen35OmniImageEncoder(nn.Module):
    """Vision tower extracted from a Qwen3.5-Omni thinker checkpoint."""

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
        vision_cfg = thinker_cfg.vision_config
        self._device = torch.device(device)
        self.visual = _build_visual(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
        self.out_hidden_size = _resolve_vision_hidden_size(vision_cfg, self.visual)
        self.deepstack_layers = len(
            getattr(vision_cfg, "deepstack_visual_indexes", []) or []
        )

    @property
    def visual_dtype_bytes(self) -> int:
        return torch.empty((), dtype=_module_dtype(self.visual)).element_size()

    def _encode_pixels(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        grid_thw = grid_thw.to(self._device, dtype=torch.long)
        visual_dtype = _module_dtype(self.visual)
        pixel_values = pixel_values.to(device=self._device, dtype=visual_dtype)
        outputs = _call_visual(self.visual, pixel_values, grid_thw)
        embeds, multiscale = _split_visual_output(outputs)
        embeds, multiscale = _split_packed_deepstack_output(
            embeds,
            multiscale,
            base_hidden_size=self.out_hidden_size,
            deepstack_layers=self.deepstack_layers,
        )
        token_counts = grid_thw.prod(-1) // (self.spatial_merge_size**2)
        return embeds, multiscale, token_counts.to(device=self._device)

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

        if isinstance(pixel_values, torch.Tensor) and isinstance(
            image_grid_thw, torch.Tensor
        ):
            embeds, multiscale, token_counts = self._encode_pixels(
                pixel_values,
                image_grid_thw,
            )
            outputs.update(
                {
                    "image_embeds": embeds,
                    "image_grid_thw": image_grid_thw.to(self._device),
                    "image_token_counts": token_counts,
                    "image_deepstack_visual_embeds": multiscale,
                    # 中文说明：保留旧 key 兼容已经接入的 stage/test；
                    # request builder 会优先读取 canonical key。
                    "deepstack_visual_embeds_image": multiscale,
                }
            )

        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
            video_grid_thw, torch.Tensor
        ):
            embeds, multiscale, token_counts = self._encode_pixels(
                pixel_values_videos,
                video_grid_thw,
            )
            outputs.update(
                {
                    "video_embeds": embeds,
                    "video_grid_thw": video_grid_thw.to(self._device),
                    "video_token_counts": token_counts,
                    "video_deepstack_visual_embeds": multiscale,
                    # 中文说明：同 image 分支，兼容历史 key。
                    "deepstack_visual_embeds_video": multiscale,
                }
            )

        return outputs


Qwen3OmniNextImageEncoder = Qwen35OmniImageEncoder
