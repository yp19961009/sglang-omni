# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni model components and pipeline helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_5_omni import config, preflight
    from sglang_omni.models.qwen3_5_omni.components.preprocessor import (
        Qwen35OmniPreprocessor,
    )

__all__ = [
    "Qwen35OmniPreprocessor",
    "config",
    "preflight",
]


def __getattr__(name: str):
    if name == "config":
        return import_module(f"{__name__}.config")
    if name == "preflight":
        return import_module(f"{__name__}.preflight")
    if name == "Qwen35OmniPreprocessor":
        from sglang_omni.models.qwen3_5_omni.components.preprocessor import (
            Qwen35OmniPreprocessor,
        )

        return Qwen35OmniPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
