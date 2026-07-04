"""Vendor wrapper for SGLang Mamba radix cache.

Applies a narrow guard for RTC Mamba prefix-cache entries whose state has
become non-finite. Such entries poison later actual requests when copied from
radix cache, so skip caching them and let the request fall back to normal KV
state instead.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

logger = logging.getLogger(__name__)
_SKIP_NONFINITE_MAMBA_CACHE_ENV = "SGLANG_OMNI_SKIP_NAN_MAMBA_CACHE"
_SKIP_RTC_ACTUAL_MAMBA_CACHE_INSERT_ENV = (
    "SGLANG_OMNI_SKIP_RTC_ACTUAL_MAMBA_CACHE_INSERT"
)
_PATCHED_FLAG = "_sglang_omni_skip_nan_mamba_cache_patch"


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_rtc_req(req: Any) -> bool:
    namespace = getattr(req, "_omni_rtc_cache_namespace", None)
    if isinstance(namespace, str) and namespace.startswith("rtc:"):
        return True
    return "rtc:" in str(getattr(req, "extra_key", ""))


def _is_direct_full_req(req: Any) -> bool:
    if bool(getattr(req, "_omni_direct_full_actual", False)):
        return True
    return "direct-steady:" in str(getattr(req, "extra_key", ""))


def _is_guarded_req(req: Any) -> bool:
    return _is_rtc_req(req) or _is_direct_full_req(req)


def _is_guarded_actual_req(req: Any) -> bool:
    if not _is_guarded_req(req):
        return False
    sampling_params = getattr(req, "sampling_params", None)
    max_new_tokens = getattr(sampling_params, "max_new_tokens", None)
    try:
        return max_new_tokens is not None and int(max_new_tokens) > 0
    except (TypeError, ValueError):
        return False


def _should_skip_nonfinite_mamba_cache(req: Any) -> bool:
    return _env_flag_enabled(
        _SKIP_NONFINITE_MAMBA_CACHE_ENV,
        default=True,
    ) and _is_guarded_req(req)


def _should_skip_rtc_actual_mamba_cache_insert(req: Any) -> bool:
    return _env_flag_enabled(
        _SKIP_RTC_ACTUAL_MAMBA_CACHE_INSERT_ENV,
        default=True,
    ) and _is_guarded_actual_req(req)


def _mamba_value_is_nonfinite(cache: Any, mamba_value: torch.Tensor | None) -> bool:
    if mamba_value is None:
        return False
    pool = cache.req_to_token_pool.mamba_pool
    temporal = pool.mamba_cache.temporal[:, mamba_value]
    if not bool(torch.isfinite(temporal).all().detach().cpu().item()):
        return True
    for conv in pool.mamba_cache.conv:
        conv_slice = conv[:, mamba_value]
        if not bool(torch.isfinite(conv_slice).all().detach().cpu().item()):
            return True
    return False


def _finished_mamba_value(cache: Any, req: Any) -> torch.Tensor | None:
    if cache.enable_mamba_extra_buffer:
        buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if buffer is None:
            return None
        keep_idx = cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
            req.mamba_next_track_idx
        )
        return buffer[keep_idx].unsqueeze(-1).clone()
    mamba_pool_idx = getattr(req, "mamba_pool_idx", None)
    if mamba_pool_idx is None:
        return None
    return mamba_pool_idx.unsqueeze(-1).clone()


def _unfinished_mamba_value(cache: Any, req: Any) -> torch.Tensor | None:
    if cache.enable_mamba_extra_buffer:
        buffer = getattr(req, "mamba_ping_pong_track_buffer", None)
        if buffer is None:
            return None
        keep_idx = cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
            req.mamba_next_track_idx
        )
        return buffer[keep_idx].unsqueeze(-1).clone()
    req_pool_idx = getattr(req, "req_pool_idx", None)
    if req_pool_idx is None:
        return None
    return cache.req_to_token_pool.get_mamba_indices(req_pool_idx).unsqueeze(-1)


def _local_mamba_value(req: Any) -> torch.Tensor | None:
    mamba_pool_idx = getattr(req, "mamba_pool_idx", None)
    if mamba_pool_idx is None:
        return None
    return mamba_pool_idx.unsqueeze(-1).clone()


def _prefix_len(req: Any) -> int:
    prefix_indices = getattr(req, "prefix_indices", None)
    return len(prefix_indices) if prefix_indices is not None else 0


def _log_skip(where: str, req: Any, cache_len: int | None) -> None:
    logger.warning(
        "skip_nonfinite_mamba_cache_%s rid=%s cache_len=%s prefix_len=%s "
        "extend_input_len=%s extra_key=%s",
        where,
        getattr(req, "rid", None),
        cache_len,
        _prefix_len(req),
        getattr(req, "extend_input_len", None),
        getattr(req, "extra_key", None),
    )


def _empty_match_result(cache: Any) -> MatchResult:
    return MatchResult(
        device_indices=torch.empty(
            (0,),
            dtype=torch.int64,
            device=cache.device,
        ),
        last_device_node=cache.root_node,
        last_host_node=cache.root_node,
    )


def _free_local_mamba_value(cache: Any, req: Any) -> None:
    mamba_pool_idx = getattr(req, "mamba_pool_idx", None)
    if mamba_pool_idx is None:
        return
    try:
        cache.req_to_token_pool.mamba_pool.free(mamba_pool_idx.unsqueeze(-1))
    finally:
        req.mamba_pool_idx = None


def _drop_bad_node_mamba_value(cache: Any, node: Any) -> bool:
    if node is None or node == cache.root_node:
        return False
    mamba_value = getattr(node, "mamba_value", None)
    if mamba_value is None or getattr(node, "mamba_lock_ref", 0) > 0:
        return False

    has_children = len(getattr(node, "children", {}) or {}) > 0
    if not has_children and getattr(node, "full_lock_ref", 0) > 0:
        return False

    if cache.mamba_lru_list.in_list(node):
        cache.mamba_lru_list.remove_node(node)
        cache.mamba_evictable_size_ -= len(mamba_value)
    cache.req_to_token_pool.mamba_pool.free(mamba_value)

    if has_children:
        node.mamba_value = None
        return True

    if cache.full_lru_list.in_list(node):
        cache.full_lru_list.remove_node(node)
    cache.token_to_kv_pool_allocator.free(node.value)
    cache._delete_leaf(node)
    cache._iteratively_delete_tombstone_leaf(node)
    return True


def _match_len(result: Any) -> int:
    device_indices = getattr(result, "device_indices", None)
    if device_indices is None:
        return 0
    return len(device_indices)


def _match_prefix_with_guard(
    cache: Any,
    key: Any,
    *,
    original: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    req = kwargs.get("req")
    result = original(cache, key, **kwargs)
    if (
        req is None
        or not kwargs.get("cow_mamba", False)
        or not _should_skip_nonfinite_mamba_cache(req)
        or _match_len(result) == 0
    ):
        return result

    mamba_value = _local_mamba_value(req)
    if not _mamba_value_is_nonfinite(cache, mamba_value):
        return result

    logger.warning(
        "skip_nonfinite_mamba_cache_match rid=%s matched_len=%s "
        "prefix_len=%s extend_input_len=%s dropped_node=%s extra_key=%s",
        getattr(req, "rid", None),
        _match_len(result),
        _prefix_len(req),
        getattr(req, "extend_input_len", None),
        _drop_bad_node_mamba_value(
            cache, getattr(result, "last_device_node", None)
        ),
        getattr(req, "extra_key", None),
    )
    _free_local_mamba_value(cache, req)
    return _empty_match_result(cache)


def _cache_finished_req_with_guard(
    cache: Any,
    req: Any,
    *,
    is_insert: bool,
    original: Callable[..., Any],
) -> Any:
    if is_insert and _should_skip_rtc_actual_mamba_cache_insert(req):
        return original(cache, req, is_insert=False)
    if is_insert and _should_skip_nonfinite_mamba_cache(req):
        cache_len = (
            getattr(req, "mamba_last_track_seqlen", None)
            if cache.enable_mamba_extra_buffer
            else None
        )
        mamba_value = _finished_mamba_value(cache, req)
        if _mamba_value_is_nonfinite(cache, mamba_value):
            _log_skip("finished", req, cache_len)
            return original(cache, req, is_insert=False)
    return original(cache, req, is_insert=is_insert)


def _skip_cache_unfinished_req(cache: Any, req: Any) -> None:
    kv_indices = cache.req_to_token_pool.req_to_token[
        req.req_pool_idx,
        : len(req.fill_ids),
    ]
    req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)


def _cache_unfinished_req_with_guard(
    cache: Any,
    req: Any,
    *,
    chunked: bool,
    original: Callable[..., Any],
) -> Any:
    if _should_skip_rtc_actual_mamba_cache_insert(req):
        _skip_cache_unfinished_req(cache, req)
        return None
    cache_len = (
        getattr(req, "mamba_last_track_seqlen", None)
        if cache.enable_mamba_extra_buffer
        else len(getattr(req, "fill_ids", []) or [])
    )
    if (
        not getattr(cache, "disable", False)
        and cache_len is not None
        and _should_skip_nonfinite_mamba_cache(req)
    ):
        mamba_value = _unfinished_mamba_value(cache, req)
        if _mamba_value_is_nonfinite(cache, mamba_value):
            _log_skip("unfinished", req, cache_len)
            _skip_cache_unfinished_req(cache, req)
            return None
    return original(cache, req, chunked=chunked)


def apply_skip_nonfinite_mamba_cache_patch() -> None:
    if getattr(MambaRadixCache, _PATCHED_FLAG, False):
        return

    original_cache_finished_req = MambaRadixCache.cache_finished_req
    original_cache_unfinished_req = MambaRadixCache.cache_unfinished_req
    original_match_prefix = MambaRadixCache.match_prefix

    def cache_finished_req(self, req, is_insert: bool = True):
        return _cache_finished_req_with_guard(
            self,
            req,
            is_insert=is_insert,
            original=original_cache_finished_req,
        )

    def cache_unfinished_req(self, req, chunked=False):
        return _cache_unfinished_req_with_guard(
            self,
            req,
            chunked=chunked,
            original=original_cache_unfinished_req,
        )

    def match_prefix(self, key, **kwargs):
        return _match_prefix_with_guard(
            self,
            key,
            original=original_match_prefix,
            **kwargs,
        )

    MambaRadixCache.cache_finished_req = cache_finished_req
    MambaRadixCache.cache_unfinished_req = cache_unfinished_req
    MambaRadixCache.match_prefix = match_prefix
    setattr(MambaRadixCache, _PATCHED_FLAG, True)


apply_skip_nonfinite_mamba_cache_patch()

__all__ = [
    "MambaRadixCache",
    "apply_skip_nonfinite_mamba_cache_patch",
]
