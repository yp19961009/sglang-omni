# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache


def test_stage_output_cache_returns_independent_tensor_values():
    cache = StageOutputCache()
    cache.put("k", {"x": torch.tensor([1, 2, 3])})

    first = cache.get("k")
    assert first is not None
    first["x"][0] = 99

    second = cache.get("k")
    assert second is not None
    assert second["x"].tolist() == [1, 2, 3]


def test_stage_output_cache_clones_nested_sequences():
    cache = StageOutputCache()
    cache.put("k", {"layers": [torch.tensor([1.0]), (torch.tensor([2.0]),)]})

    first = cache.get("k")
    assert first is not None
    first["layers"][0][0] = 9.0
    first["layers"][1][0][0] = 8.0

    second = cache.get("k")
    assert second is not None
    assert second["layers"][0].item() == 1.0
    assert second["layers"][1][0].item() == 2.0
