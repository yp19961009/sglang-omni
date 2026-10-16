# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.vendor.sglang import mamba_radix_cache


class _FakeLRU:
    def __init__(self):
        self.nodes = []

    def in_list(self, node):
        return node in self.nodes

    def remove_node(self, node):
        self.nodes.remove(node)


class _FakeMambaPool:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.append(value.clone())


class _FakeTokenAllocator:
    def __init__(self):
        self.freed = []

    def free(self, value):
        self.freed.append(value.clone())


def _cache(root, node):
    mamba_lru = _FakeLRU()
    full_lru = _FakeLRU()
    mamba_lru.nodes.append(node)
    full_lru.nodes.append(node)
    return SimpleNamespace(
        root_node=root,
        mamba_lru_list=mamba_lru,
        full_lru_list=full_lru,
        mamba_evictable_size_=1,
        req_to_token_pool=SimpleNamespace(mamba_pool=_FakeMambaPool()),
        token_to_kv_pool_allocator=_FakeTokenAllocator(),
        get_child_key_fn=lambda key: tuple(key.token_ids[:1]),
        _delete_leaf=lambda n: n.parent.children.pop(tuple(n.key.token_ids[:1])),
        _iteratively_delete_tombstone_leaf=lambda n: None,
    )


def test_drop_bad_node_mamba_value_tombstones_internal_node():
    root = SimpleNamespace()
    node = SimpleNamespace(
        mamba_value=torch.tensor([3]),
        mamba_lock_ref=0,
        full_lock_ref=0,
        children={"child": object()},
    )
    cache = _cache(root, node)

    assert mamba_radix_cache._drop_bad_node_mamba_value(cache, node)

    assert node.mamba_value is None
    assert node not in cache.mamba_lru_list.nodes
    assert cache.mamba_evictable_size_ == 0
    assert cache.req_to_token_pool.mamba_pool.freed[0].tolist() == [3]
    assert cache.token_to_kv_pool_allocator.freed == []
    assert node in cache.full_lru_list.nodes


def test_drop_bad_node_mamba_value_removes_unlocked_leaf():
    root = SimpleNamespace()
    node = SimpleNamespace(
        mamba_value=torch.tensor([5]),
        value=torch.tensor([11, 12]),
        mamba_lock_ref=0,
        full_lock_ref=0,
        children={},
        key=SimpleNamespace(token_ids=[7]),
    )
    parent = SimpleNamespace(children={tuple(node.key.token_ids[:1]): node})
    node.parent = parent
    cache = _cache(root, node)

    assert mamba_radix_cache._drop_bad_node_mamba_value(cache, node)

    assert tuple(node.key.token_ids[:1]) not in parent.children
    assert node not in cache.mamba_lru_list.nodes
    assert node not in cache.full_lru_list.nodes
    assert cache.req_to_token_pool.mamba_pool.freed[0].tolist() == [5]
    assert cache.token_to_kv_pool_allocator.freed[0].tolist() == [11, 12]


def test_drop_bad_node_mamba_value_leaves_locked_leaf_unchanged():
    root = SimpleNamespace()
    node = SimpleNamespace(
        mamba_value=torch.tensor([8]),
        value=torch.tensor([21]),
        mamba_lock_ref=0,
        full_lock_ref=1,
        children={},
    )
    cache = _cache(root, node)

    assert not mamba_radix_cache._drop_bad_node_mamba_value(cache, node)

    assert node.mamba_value.tolist() == [8]
    assert node in cache.mamba_lru_list.nodes
    assert node in cache.full_lru_list.nodes
    assert cache.req_to_token_pool.mamba_pool.freed == []
    assert cache.token_to_kv_pool_allocator.freed == []


def test_cache_finished_req_skips_rtc_actual_mamba_insert():
    req = SimpleNamespace(
        _omni_rtc_cache_namespace="rtc:req-0",
        sampling_params=SimpleNamespace(max_new_tokens=64),
    )
    cache = SimpleNamespace()
    calls = []

    def _original(cache_arg, req_arg, is_insert=True):
        calls.append((cache_arg, req_arg, is_insert))
        return "done"

    assert (
        mamba_radix_cache._cache_finished_req_with_guard(
            cache, req, is_insert=True, original=_original
        )
        == "done"
    )
    assert calls == [(cache, req, False)]


def test_cache_unfinished_req_skips_rtc_actual_mamba_insert():
    req = SimpleNamespace(
        _omni_rtc_cache_namespace="rtc:req-0",
        sampling_params=SimpleNamespace(max_new_tokens=64),
        req_pool_idx=0,
        fill_ids=[1, 2, 3, 4],
    )
    cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64)
        )
    )

    def _original(*args, **kwargs):
        raise AssertionError("actual RTC requests should not be inserted")

    assert (
        mamba_radix_cache._cache_unfinished_req_with_guard(
            cache, req, chunked=True, original=_original
        )
        is None
    )
    assert req.prefix_indices.tolist() == [11, 12, 13, 14]
