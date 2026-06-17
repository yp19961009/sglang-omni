# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

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
