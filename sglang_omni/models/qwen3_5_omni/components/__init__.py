# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni component entry points."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_5_omni.components.audio_encoder import (
        Qwen35OmniAudioEncoder,
    )
    from sglang_omni.models.qwen3_5_omni.components.image_encoder import (
        Qwen35OmniImageEncoder,
    )
    from sglang_omni.models.qwen3_5_omni.components.preprocessor import (
        Qwen35OmniPreprocessor,
    )
    from sglang_omni.models.qwen3_5_omni.components.sglang_thinker import (
        Qwen3OmniNextThinkerForConditionalGeneration,
    )
    from sglang_omni.models.qwen3_5_omni.components.subtalker import (
        Qwen35ResidualCodePredictor,
    )
    from sglang_omni.models.qwen3_5_omni.components.talker import (
        Qwen3OmniNextMoeTalkerForConditionalGeneration,
        Qwen3OmniNextTalkerForConditionalGeneration,
    )

__all__ = [
    "Qwen35OmniAudioEncoder",
    "Qwen35OmniImageEncoder",
    "Qwen35OmniPreprocessor",
    "Qwen35ResidualCodePredictor",
    "Qwen3OmniNextThinkerForConditionalGeneration",
    "Qwen3OmniNextMoeTalkerForConditionalGeneration",
    "Qwen3OmniNextTalkerForConditionalGeneration",
]


def __getattr__(name: str):
    if name == "Qwen35OmniAudioEncoder":
        module = import_module(f"{__name__}.audio_encoder")
        return module.Qwen35OmniAudioEncoder
    if name == "Qwen35OmniImageEncoder":
        module = import_module(f"{__name__}.image_encoder")
        return module.Qwen35OmniImageEncoder
    if name == "Qwen35OmniPreprocessor":
        module = import_module(f"{__name__}.preprocessor")
        return module.Qwen35OmniPreprocessor
    if name == "Qwen3OmniNextThinkerForConditionalGeneration":
        module = import_module(f"{__name__}.sglang_thinker")
        return module.Qwen3OmniNextThinkerForConditionalGeneration
    if name == "Qwen35ResidualCodePredictor":
        module = import_module(f"{__name__}.subtalker")
        return module.Qwen35ResidualCodePredictor
    if name == "Qwen3OmniNextTalkerForConditionalGeneration":
        module = import_module(f"{__name__}.talker")
        return module.Qwen3OmniNextTalkerForConditionalGeneration
    if name == "Qwen3OmniNextMoeTalkerForConditionalGeneration":
        module = import_module(f"{__name__}.talker")
        return module.Qwen3OmniNextMoeTalkerForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
