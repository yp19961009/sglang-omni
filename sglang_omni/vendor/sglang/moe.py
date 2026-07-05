"""Vendor patches for SGLang MoE helpers.

Qwen3.5 full-chain audio captures thinker hidden states for the talker. In
that path, SGLang can run the small-token MoE reduce helper while FX/Dynamo is
already tracing the surrounding graph. The upstream helper is itself wrapped in
``torch.compile``; calling a compiled function from inside FX tracing raises and
cuts the HTTP stream mid-response. Keep the fast compiled helper for normal
eager execution, but use the equivalent eager reduce while tracing.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_FUSED_MOE_MODULE = "sglang.srt.layers.moe.fused_moe_triton.fused_moe"
_TRITON_RUNNER_MODULE = "sglang.srt.layers.moe.moe_runner.triton"
_FORCE_EAGER_ENV = "SGLANG_OMNI_MOE_SUM_REDUCE_EAGER"
_PATCHED_FLAG = "_sglang_omni_moe_sum_reduce_fx_guard_patch"
_ORIGINAL_ATTR = "_sglang_omni_original_moe_sum_reduce_torch_compile"


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _force_eager_reduce() -> bool:
    return _env_flag_enabled(_FORCE_EAGER_ENV, default=False)


def _is_fx_or_dynamo_tracing() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None:
        try:
            if bool(is_compiling()):
                return True
        except Exception:
            pass

    try:
        import torch._dynamo as dynamo  # type: ignore[attr-defined]

        if bool(dynamo.is_compiling()):
            return True
    except Exception:
        pass

    try:
        from torch.fx._symbolic_trace import is_fx_symbolic_tracing

        if bool(is_fx_symbolic_tracing()):
            return True
    except Exception:
        pass

    return False


def _eager_moe_sum_reduce(
    x: torch.Tensor,
    out: torch.Tensor,
    routed_scaling_factor: float | int | None,
) -> None:
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def _make_guarded_moe_sum_reduce(
    original: Callable[[torch.Tensor, torch.Tensor, Any], Any],
) -> Callable[[torch.Tensor, torch.Tensor, Any], Any]:
    def guarded_moe_sum_reduce_torch_compile(
        x: torch.Tensor,
        out: torch.Tensor,
        routed_scaling_factor: Any,
    ) -> Any:
        if _force_eager_reduce() or _is_fx_or_dynamo_tracing():
            return _eager_moe_sum_reduce(x, out, routed_scaling_factor)
        return original(x, out, routed_scaling_factor)

    setattr(guarded_moe_sum_reduce_torch_compile, _PATCHED_FLAG, True)
    setattr(guarded_moe_sum_reduce_torch_compile, _ORIGINAL_ATTR, original)
    return guarded_moe_sum_reduce_torch_compile


def apply_moe_sum_reduce_fx_guard_patch() -> bool:
    """Patch SGLang's compiled MoE reduce helper when the module is available."""
    try:
        fused_moe = importlib.import_module(_FUSED_MOE_MODULE)
    except Exception:
        logger.debug("SGLang fused MoE module is unavailable", exc_info=True)
        return False

    current = getattr(fused_moe, "moe_sum_reduce_torch_compile", None)
    if current is None:
        logger.debug("SGLang fused MoE module has no sum-reduce helper to patch")
        return False
    if getattr(current, _PATCHED_FLAG, False):
        return True

    original = getattr(fused_moe, _ORIGINAL_ATTR, current)
    guarded = _make_guarded_moe_sum_reduce(original)
    setattr(fused_moe, _ORIGINAL_ATTR, original)
    setattr(fused_moe, "moe_sum_reduce_torch_compile", guarded)

    triton_runner = sys.modules.get(_TRITON_RUNNER_MODULE)
    if triton_runner is not None and hasattr(
        triton_runner,
        "moe_sum_reduce_torch_compile",
    ):
        setattr(triton_runner, "moe_sum_reduce_torch_compile", guarded)

    logger.info("applied SGLang MoE sum-reduce FX guard patch")
    return True


__all__ = [
    "apply_moe_sum_reduce_fx_guard_patch",
]
