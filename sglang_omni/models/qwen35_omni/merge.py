# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni reuses the Qwen3-Omni multimodal merge stage."""

from sglang_omni.models.qwen3_omni.merge import *  # noqa: F401,F403
from sglang_omni.models.qwen3_omni.merge import (
    merge_for_thinker as _qwen3_merge_for_thinker,
)


def merge_for_thinker(payloads):
    """Merge multimodal outputs without qwen3 encoder/prefix cache metadata."""
    result = _qwen3_merge_for_thinker(payloads)
    if isinstance(result.data, dict):
        thinker_inputs = result.data.get("thinker_inputs")
        if isinstance(thinker_inputs, dict):
            thinker_inputs.pop("media_cache_keys", None)
    return result
