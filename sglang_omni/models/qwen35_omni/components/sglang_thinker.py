# SPDX-License-Identifier: Apache-2.0
"""SGLang thinker wrapper for Qwen3.5-Omni Next text output."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.utils import logger


_MOE_SUM_REDUCE_PATCHED = False


def _patch_qwen35_moe_sum_reduce() -> None:
    """Avoid a torch.compile/autograd fake-tensor issue in small-token MoE reduce."""

    global _MOE_SUM_REDUCE_PATCHED
    if _MOE_SUM_REDUCE_PATCHED:
        return

    def _moe_sum_reduce_eager(
        x: torch.Tensor,
        out: torch.Tensor,
        routed_scaling_factor: float,
    ) -> None:
        with torch.no_grad():
            reduced = torch.sum(x, dim=1)
            if routed_scaling_factor != 1.0:
                reduced = reduced * routed_scaling_factor
            out.copy_(reduced)

    for module_name in (
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe",
        "sglang.srt.layers.moe.moe_runner.triton",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive import for SGLang variants.
            logger.debug("Qwen35 thinker could not patch %s: %s", module_name, exc)
            continue
        if hasattr(module, "moe_sum_reduce_torch_compile"):
            setattr(module, "moe_sum_reduce_torch_compile", _moe_sum_reduce_eager)

    _MOE_SUM_REDUCE_PATCHED = True


class _Qwen3NextModelAdapter(nn.Module):
    """Expose Qwen3NextModel with the input_embeds name used by Omni runner."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.inner = model
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Any | None = None,
    ) -> torch.Tensor:
        del input_deepstack_embeds, pp_proxy_tensors
        return self.inner(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
        )


class Qwen35OmniNextThinkerForCausalLM(nn.Module):
    """Qwen3.5-Omni thinker text model without duplicated encoders."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _patch_qwen35_moe_sum_reduce()
        self.root_config = config
        self.thinker_config = getattr(config, "thinker_config", config)
        self.config = getattr(self.thinker_config, "text_config", self.thinker_config)
        self._next_lm = Qwen3NextForCausalLM(
            config=self.config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.model = _Qwen3NextModelAdapter(self._next_lm.model)
        self.lm_head = self._next_lm.lm_head
        self.logits_processor = self._next_lm.logits_processor

    @property
    def thinker(self) -> "Qwen35OmniNextThinkerForCausalLM":
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any | None = None,
        input_embeds: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ):
        del get_embedding, pp_proxy_tensors, input_deepstack_embeds
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        return self._next_lm(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds if inputs_embeds is not None else input_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        cleaned = []
        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")
            if name.startswith("thinker."):
                name = name[len("thinker.") :]
            elif name.startswith(("talker.", "code2wav.")):
                continue
            if name.startswith(("audio_tower.", "visual.")):
                continue
            if name.startswith(("model.", "lm_head.")) or "mtp" in name:
                cleaned.append((name, loaded_weight))
        try:
            self._next_lm.load_weights(cleaned)
        except KeyError as exc:
            logger.warning("Qwen35 thinker skipped missing weight during load: %s", exc)
            raise


EntryClass = Qwen35OmniNextThinkerForCausalLM
