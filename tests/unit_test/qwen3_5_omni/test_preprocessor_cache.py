# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from sglang_omni.models.qwen3_5_omni.components.preprocessor import (
    _Qwen35ProcessorShim,
)


def test_processor_item_cache_returns_independent_values():
    shim = object.__new__(_Qwen35ProcessorShim)
    cache = OrderedDict()
    key = ("video", "k")
    value = {
        "tensor": torch.tensor([1, 2]),
        "array": np.array([3, 4]),
        "nested": (torch.tensor([5]),),
    }

    shim._processor_cache_set(cache, key, value)
    value["tensor"][0] = 99
    value["array"][0] = 99

    first = shim._processor_cache_get(cache, key)
    assert first is not None
    first["tensor"][0] = 77
    first["array"][0] = 77
    first["nested"][0][0] = 77

    second = shim._processor_cache_get(cache, key)
    assert second is not None
    assert second["tensor"].tolist() == [1, 2]
    assert second["array"].tolist() == [3, 4]
    assert second["nested"][0].tolist() == [5]


def test_processor_item_cache_can_return_shared_value_when_clone_disabled():
    shim = object.__new__(_Qwen35ProcessorShim)
    cache = OrderedDict()
    key = ("video", "k")
    value = {"tensor": torch.tensor([1, 2])}

    shim._processor_cache_set(cache, key, value)
    first = shim._processor_cache_get(cache, key, clone=False)
    assert first is not None
    first["tensor"][0] = 77

    second = shim._processor_cache_get(cache, key, clone=False)
    assert second is not None
    assert second["tensor"].tolist() == [77, 2]


def test_get_video_tokens_does_not_mutate_frame_indices():
    shim = object.__new__(_Qwen35ProcessorShim)
    shim.vision_bos_token = "<vision_bos>"
    shim.vision_eos_token = "<vision_eos>"
    shim.video_token = "<video>"
    indices = [0, 1, 2]

    text = shim._get_video_tokens(
        indices,
        video_fps=1.0,
        video_grid_thw=torch.tensor([2, 2, 2]),
        merge_size=2,
    )

    assert indices == [0, 1, 2]
    assert "<video>" in text
