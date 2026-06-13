# SPDX-License-Identifier: Apache-2.0
"""SGLang thinker wrapper for Qwen3.5-Omni.

The Qwen3.5-Omni root checkpoint owns thinker/talker/code2wav/encoder weights.
This wrapper keeps only the Qwen3-Next language model used by the thinker stage;
the Omni pipeline runs image/audio encoders as separate stages and injects their
embeddings before SGLang prefill.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.utils import logger as sglang_logger

from sglang_omni.models.qwen3_omni.quantization import (
    convert_fp8_weight_scale_inv_for_sglang,
)
from sglang_omni.models.qwen3_5_omni.components.common import (
    ensure_sglang_qwen3_next_text_config,
    sub_config_or_self,
)

logger = logging.getLogger(__name__)

_SKIP_PREFIXES = (
    "audio_tower.",
    "visual.",
    "talker.",
    "code2wav.",
    "mtp.",
)
_LANGUAGE_PREFIXES = ("model.", "lm_head.")
_IGNORED_SUFFIXES = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    ".weight_scale",
    "_weight_scale",
    ".input_scale",
    "_input_scale",
)


def _normalize_language_weight_name(name: str) -> str | None:
    """Map Omni checkpoint names to Qwen3NextForCausalLM-local names."""
    if name.startswith("thinker.lm_head."):
        name = "lm_head." + name[len("thinker.lm_head.") :]
    elif name.startswith("thinker.model."):
        name = "model." + name[len("thinker.model.") :]
    elif name.startswith("thinker.language_model."):
        name = name[len("thinker.language_model.") :]
    elif name.startswith("thinker."):
        name = name[len("thinker.") :]
    elif name.startswith("language_model."):
        name = name[len("language_model.") :]
    elif name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]

    if name.startswith(_SKIP_PREFIXES):
        return None
    if not name.startswith(_LANGUAGE_PREFIXES):
        return None
    return name


def _iter_language_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    params_dict: dict[str, nn.Parameter] | None = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    for name, loaded_weight in weights:
        mapped = _normalize_language_weight_name(name)
        if mapped is None:
            continue
        if (
            params_dict is not None
            and mapped.endswith(_IGNORED_SUFFIXES)
            and mapped not in params_dict
        ):
            continue
        loaded_weight = convert_fp8_weight_scale_inv_for_sglang(
            mapped,
            loaded_weight,
        )
        yield mapped, loaded_weight


def _has_real_deepstack(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, dict):
        return any(_has_real_deepstack(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_real_deepstack(item) for item in value)
    return True


def _native_layers_to_capture_supported(forward_fn: Any) -> bool:
    try:
        source = inspect.getsource(forward_fn)
    except (OSError, TypeError):
        return False
    return "layers_to_capture" in source and "aux_hidden_states" in source


def _deepstack_tensor_for_layer(value: Any, layer_id: int) -> torch.Tensor | None:
    if value is None:
        return None
    key = f"deepstack_input_embeds_{layer_id}"
    if isinstance(value, dict):
        if key in value:
            return value[key]
        if layer_id in value:
            return value[layer_id]
        return None
    if isinstance(value, (list, tuple)):
        return value[layer_id] if layer_id < len(value) else None
    if isinstance(value, torch.Tensor):
        if value.ndim >= 3 and layer_id < value.shape[0]:
            return value[layer_id]
        return value if layer_id == 0 else None
    try:
        return value[key]
    except Exception:
        return None


def _add_deepstack_for_layer(
    hidden_states: torch.Tensor,
    deepstack_input_embeds: Any,
    layer_id: int,
) -> torch.Tensor:
    layer_deepstack = _deepstack_tensor_for_layer(deepstack_input_embeds, layer_id)
    if layer_deepstack is None:
        return hidden_states
    layer_deepstack = layer_deepstack.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # 中文说明：vLLM Qwen3NextVLModel 在对应层之后把 deepstack visual
    # features 加回 hidden_states；这里保持同样位置，避免影响 attention 输入。
    return hidden_states + layer_deepstack


def _tensor_stats(tensor: torch.Tensor, sample_size: int = 8) -> dict[str, Any]:
    data = tensor.detach()
    stats_data = data.float()
    last = stats_data.reshape(-1, stats_data.shape[-1])[-1]
    stats = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "mean": float(stats_data.mean().cpu()),
        "std": float(stats_data.std(unbiased=False).cpu()),
        "norm": float(torch.linalg.vector_norm(stats_data).cpu()),
        "last_mean": float(last.mean().cpu()),
        "last_std": float(last.std(unbiased=False).cpu()),
        "last_norm": float(torch.linalg.vector_norm(last).cpu()),
        "last_first_values": [
            float(x) for x in last[:sample_size].detach().cpu().tolist()
        ],
    }
    if os.getenv("QWEN35_THINKER_HIDDEN_DUMP_LAST_VALUES"):
        stats["last_values"] = [float(x) for x in last.detach().cpu().tolist()]
    if os.getenv("QWEN35_THINKER_HIDDEN_DUMP_TOKEN_NORMS"):
        flat = stats_data.reshape(-1, stats_data.shape[-1])
        stats["token_norms"] = [
            float(x) for x in torch.linalg.vector_norm(flat, dim=-1).cpu().tolist()
        ]
    return stats


def _hidden_dump_layers() -> set[int] | None:
    raw = os.getenv("QWEN35_THINKER_HIDDEN_DUMP_LAYERS")
    if not raw:
        return None
    layers: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layers.add(int(item))
        except ValueError:
            logger.warning("invalid QWEN35_THINKER_HIDDEN_DUMP_LAYERS item: %s", item)
    return layers


def _write_hidden_dump(path: str, record: dict[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("failed to write Qwen3.5 thinker hidden dump to %s", path)


def _install_layers_to_capture_support(text_model: nn.Module) -> None:
    """Teach SGLang Qwen3NextModel to return aux hidden states when requested."""
    if getattr(text_model, "_qwen35_capture_forward_installed", False):
        return

    original_forward = text_model.forward
    native_capture_supported = _native_layers_to_capture_supported(original_forward)

    @functools.wraps(original_forward)
    def _forward_with_capture(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_input_embeds: Any | None = None,
    ):
        capture_layers = list(getattr(text_model, "layers_to_capture", []) or [])
        has_deepstack = _has_real_deepstack(deepstack_input_embeds)
        hidden_dump_path = os.getenv("QWEN35_THINKER_HIDDEN_DUMP")
        hidden_dump_prefill_only = os.getenv(
            "QWEN35_THINKER_HIDDEN_DUMP_DECODE"
        ) not in {"1", "true", "TRUE", "yes", "YES"}
        hidden_dump_enabled = bool(hidden_dump_path) and (
            not hidden_dump_prefill_only or forward_batch.forward_mode.is_extend()
        )
        if (
            not hidden_dump_enabled
            and (not has_deepstack)
            and (not capture_layers or native_capture_supported)
        ):
            return original_forward(
                input_ids,
                positions,
                forward_batch,
                inputs_embeds,
            )

        # 中文说明：现有 hidden-capture hook 期望 text_model.forward 在设置
        # layers_to_capture 后返回 (hidden_states, aux_hidden_states)。SGLang
        # core 的 Qwen3Next 暂未内建这个返回，所以这里按原 forward 逻辑补齐。
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = text_model.embed_tokens(input_ids)

        capture_set = set(capture_layers)
        aux_hidden_by_layer = {}
        if 0 in capture_set:
            aux_hidden_by_layer[0] = hidden_states.clone()
        dump_layers = _hidden_dump_layers() if hidden_dump_enabled else None
        dump_base: dict[str, Any] = {}
        if hidden_dump_enabled:
            mrope_positions = getattr(forward_batch, "mrope_positions", None)
            seq_lens = getattr(forward_batch, "seq_lens", None)
            extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
            dump_base = {
                "forward_mode": str(getattr(forward_batch, "forward_mode", None)),
                "input_ids_shape": (
                    list(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None
                ),
                "positions_shape": list(positions.shape),
                "mrope_positions_shape": (
                    list(mrope_positions.shape)
                    if isinstance(mrope_positions, torch.Tensor)
                    else None
                ),
                "seq_lens": (
                    [int(x) for x in seq_lens.detach().cpu().tolist()]
                    if isinstance(seq_lens, torch.Tensor)
                    else None
                ),
                "extend_seq_lens_cpu": (
                    [int(x) for x in extend_lens]
                    if extend_lens is not None
                    else None
                ),
            }
            _write_hidden_dump(
                hidden_dump_path,
                {
                    **dump_base,
                    "stage": "embed",
                    "hidden": _tensor_stats(hidden_states),
                },
            )
        residual = None
        for layer_id, layer in enumerate(text_model.layers):
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_id
            ):
                hidden_states, residual = layer(
                    layer_id=layer_id,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                )
            if has_deepstack:
                hidden_states = _add_deepstack_for_layer(
                    hidden_states,
                    deepstack_input_embeds,
                    layer_id,
                )
            if layer_id != 0 and layer_id in capture_set:
                # 中文说明：Qwen3.5 vLLM 的 accept_hidden_layer 语义是该层
                # 输出（并包含 deepstack 注入）后的 hidden；只有 0 保留为
                # 进入首层前的 text/embed hidden。
                aux_hidden_by_layer[layer_id] = hidden_states.clone()
            if hidden_dump_enabled and (
                dump_layers is None or layer_id in dump_layers
            ):
                record = {
                    **dump_base,
                    "stage": "layer",
                    "layer_id": layer_id,
                    "hidden": _tensor_stats(hidden_states),
                }
                if isinstance(residual, torch.Tensor):
                    record["residual"] = _tensor_stats(residual)
                _write_hidden_dump(hidden_dump_path, record)

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = text_model.norm(hidden_states)
            else:
                hidden_states, _ = text_model.norm(hidden_states, residual)
        if hidden_dump_enabled:
            _write_hidden_dump(
                hidden_dump_path,
                {
                    **dump_base,
                    "stage": "norm",
                    "hidden": _tensor_stats(hidden_states),
                },
            )

        aux_hidden_states = [
            aux_hidden_by_layer[layer_id]
            for layer_id in capture_layers
            if layer_id in aux_hidden_by_layer
        ]
        return hidden_states, aux_hidden_states

    text_model.forward = _forward_with_capture
    text_model._qwen35_capture_forward_installed = True


class Qwen3OmniNextThinkerForConditionalGeneration(nn.Module):
    """Qwen3.5-Omni thinker backed by SGLang's Qwen3NextForCausalLM."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        ) -> None:
        super().__init__()
        self.root_config = config
        self.thinker_config = sub_config_or_self(config, "thinker_config")
        self.config = ensure_sglang_qwen3_next_text_config(
            getattr(self.thinker_config, "text_config", None) or self.thinker_config
        )

        self.language_model = Qwen3NextForCausalLM(
            self.config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.model = self.language_model.model
        _install_layers_to_capture_support(self.model)

    @property
    def thinker(self) -> "Qwen3OmniNextThinkerForConditionalGeneration":
        # Existing Omni hidden-capture hooks navigate through model.thinker.model.
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any | None = None,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        del get_embedding, pp_proxy_tensors
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        if input_embeds is not None and inputs_embeds is not None:
            raise ValueError("Pass only one of input_embeds or inputs_embeds.")
        embeds = inputs_embeds if inputs_embeds is not None else input_embeds
        deepstack_input_embeds = kwargs.pop(
            "deepstack_input_embeds",
            input_deepstack_embeds,
        )

        if _has_real_deepstack(deepstack_input_embeds):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                inputs_embeds=embeds,
                deepstack_input_embeds=deepstack_input_embeds,
            )
            if isinstance(hidden_states, tuple):
                hidden_states, _ = hidden_states
            return self.language_model.logits_processor(
                input_ids,
                hidden_states,
                self.language_model.lm_head,
                forward_batch,
            )

        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=embeds,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load thinker language-model weights and ignore other Omni modules."""
        params_dict = dict(self.language_model.named_parameters())
        loaded = self.language_model.load_weights(
            _iter_language_weights(weights, params_dict=params_dict)
        )
        for name in loaded:
            if not name.startswith(_LANGUAGE_PREFIXES):
                sglang_logger.warning(
                    "Loaded unexpected Qwen3.5 thinker weight %s",
                    name,
                )
        return loaded


Qwen3OmniNextForConditionalGeneration = Qwen3OmniNextThinkerForConditionalGeneration
Qwen3OmniNextThinkerMTP = Qwen3OmniNextThinkerForConditionalGeneration
EntryClass = Qwen3OmniNextThinkerForConditionalGeneration
