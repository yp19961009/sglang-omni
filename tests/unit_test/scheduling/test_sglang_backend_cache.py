# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.scheduling.sglang_backend import cache


class _ServerArgs:
    disable_radix_cache = False
    chunked_prefill_size = None

    def __init__(self, *, enable_mamba_extra_buffer: bool = False):
        self._enable_mamba_extra_buffer = enable_mamba_extra_buffer

    def enable_mamba_extra_buffer(self) -> bool:
        return self._enable_mamba_extra_buffer


def test_create_tree_cache_uses_radix_cache_for_plain_pool(monkeypatch):
    seen = {}

    def fake_radix_cache(params):
        seen["params"] = params
        return "radix"

    monkeypatch.setattr(cache, "RadixCache", fake_radix_cache)

    result = cache.create_tree_cache(
        _ServerArgs(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        page_size=1,
    )

    assert result == "radix"
    assert seen["params"].enable_mamba_extra_buffer is False


def test_create_tree_cache_uses_mamba_radix_cache_for_hybrid_pool(monkeypatch):
    seen = {}

    def fake_mamba_radix_cache(params):
        seen["params"] = params
        return "mamba"

    monkeypatch.setattr(cache, "MambaRadixCache", fake_mamba_radix_cache)

    result = cache.create_tree_cache(
        _ServerArgs(enable_mamba_extra_buffer=True),
        req_to_token_pool=SimpleNamespace(mamba_pool=object()),
        token_to_kv_pool_allocator=SimpleNamespace(),
        page_size=1,
    )

    assert result == "mamba"
    assert seen["params"].enable_mamba_extra_buffer is True



def _mamba_guard_cache(*, temporal: torch.Tensor, conv: torch.Tensor | None = None):
    conv_tensor = conv if conv is not None else torch.zeros_like(temporal)
    mamba_cache = SimpleNamespace(temporal=temporal, conv=[conv_tensor])
    mamba_pool = SimpleNamespace(mamba_cache=mamba_cache)
    return SimpleNamespace(
        disable=False,
        enable_mamba_extra_buffer=False,
        req_to_token_pool=SimpleNamespace(
            mamba_pool=mamba_pool,
            req_to_token=torch.arange(8, dtype=torch.long).reshape(1, 8),
            get_mamba_indices=lambda req_pool_idx: torch.tensor(0, dtype=torch.long),
        ),
    )


def test_mamba_guard_finished_skips_nonfinite_rtc_cache(monkeypatch):
    from sglang_omni.vendor.sglang import mamba_radix_cache

    monkeypatch.delenv("SGLANG_OMNI_SKIP_NAN_MAMBA_CACHE", raising=False)
    cache_obj = _mamba_guard_cache(temporal=torch.tensor([[float("nan")]]))
    req = SimpleNamespace(
        rid="rid-1",
        extra_key="media-cache:audio=rtc:rid-1:audio",
        mamba_pool_idx=torch.tensor(0, dtype=torch.long),
        extend_input_len=1,
        prefix_indices=None,
    )
    calls = []

    def original(cache_arg, req_arg, *, is_insert=True):
        calls.append(is_insert)
        return "done"

    result = mamba_radix_cache._cache_finished_req_with_guard(
        cache_obj,
        req,
        is_insert=True,
        original=original,
    )

    assert result == "done"
    assert calls == [False]


def test_mamba_guard_finished_respects_disable_env(monkeypatch):
    from sglang_omni.vendor.sglang import mamba_radix_cache

    monkeypatch.setenv("SGLANG_OMNI_SKIP_NAN_MAMBA_CACHE", "0")
    cache_obj = _mamba_guard_cache(temporal=torch.tensor([[float("nan")]]))
    req = SimpleNamespace(
        rid="rid-1",
        extra_key="media-cache:audio=rtc:rid-1:audio",
        mamba_pool_idx=torch.tensor(0, dtype=torch.long),
    )
    calls = []

    def original(cache_arg, req_arg, *, is_insert=True):
        calls.append(is_insert)
        return "done"

    result = mamba_radix_cache._cache_finished_req_with_guard(
        cache_obj,
        req,
        is_insert=True,
        original=original,
    )

    assert result == "done"
    assert calls == [True]


def test_mamba_guard_unfinished_skips_nonfinite_rtc_cache(monkeypatch):
    from sglang_omni.vendor.sglang import mamba_radix_cache

    monkeypatch.delenv("SGLANG_OMNI_SKIP_NAN_MAMBA_CACHE", raising=False)
    cache_obj = _mamba_guard_cache(temporal=torch.tensor([[1.0]]), conv=torch.tensor([[float("nan")]]))
    req = SimpleNamespace(
        rid="rid-2",
        extra_key="media-cache:video=rtc:rid-2:video",
        req_pool_idx=0,
        fill_ids=[1, 2, 3, 4],
        prefix_indices=None,
        mamba_last_track_seqlen=4,
        extend_input_len=4,
    )
    calls = []

    def original(cache_arg, req_arg, *, chunked=False):
        calls.append(chunked)
        return "cached"

    result = mamba_radix_cache._cache_unfinished_req_with_guard(
        cache_obj,
        req,
        chunked=True,
        original=original,
    )

    assert result is None
    assert calls == []
    assert torch.equal(req.prefix_indices, torch.arange(4, dtype=torch.long))
