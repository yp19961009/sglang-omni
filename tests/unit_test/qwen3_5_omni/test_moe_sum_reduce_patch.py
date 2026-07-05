# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from sglang_omni.vendor.sglang import moe


def test_moe_sum_reduce_patch_uses_eager_reduce_while_tracing(monkeypatch):
    calls = []

    def compiled_helper(x, out, routed_scaling_factor):
        calls.append("compiled")
        torch.sum(x, dim=1, out=out)
        out.mul_(routed_scaling_factor)

    fused_module = SimpleNamespace(moe_sum_reduce_torch_compile=compiled_helper)
    triton_module = SimpleNamespace(moe_sum_reduce_torch_compile=compiled_helper)
    monkeypatch.setitem(sys.modules, moe._FUSED_MOE_MODULE, fused_module)
    monkeypatch.setitem(sys.modules, moe._TRITON_RUNNER_MODULE, triton_module)
    monkeypatch.setattr(moe, "_is_fx_or_dynamo_tracing", lambda: True)

    assert moe.apply_moe_sum_reduce_fx_guard_patch()
    assert fused_module.moe_sum_reduce_torch_compile is triton_module.moe_sum_reduce_torch_compile

    x = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    out = torch.empty(2, 2, dtype=torch.float32)
    fused_module.moe_sum_reduce_torch_compile(x, out, 0.5)

    assert calls == []
    assert torch.equal(out, torch.sum(x, dim=1) * 0.5)


def test_moe_sum_reduce_patch_keeps_compiled_helper_for_eager_path(monkeypatch):
    calls = []

    def compiled_helper(x, out, routed_scaling_factor):
        calls.append("compiled")
        torch.sum(x, dim=1, out=out)
        out.mul_(routed_scaling_factor)

    fused_module = SimpleNamespace(moe_sum_reduce_torch_compile=compiled_helper)
    monkeypatch.setitem(sys.modules, moe._FUSED_MOE_MODULE, fused_module)
    monkeypatch.delitem(sys.modules, moe._TRITON_RUNNER_MODULE, raising=False)
    monkeypatch.setattr(moe, "_is_fx_or_dynamo_tracing", lambda: False)

    assert moe.apply_moe_sum_reduce_fx_guard_patch()

    x = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    out = torch.empty(2, 2, dtype=torch.float32)
    fused_module.moe_sum_reduce_torch_compile(x, out, 0.25)

    assert calls == ["compiled"]
    assert torch.equal(out, torch.sum(x, dim=1) * 0.25)


def test_moe_sum_reduce_patch_is_idempotent(monkeypatch):
    def compiled_helper(x, out, routed_scaling_factor):
        torch.sum(x, dim=1, out=out)
        out.mul_(routed_scaling_factor)

    fused_module = SimpleNamespace(moe_sum_reduce_torch_compile=compiled_helper)
    monkeypatch.setitem(sys.modules, moe._FUSED_MOE_MODULE, fused_module)

    assert moe.apply_moe_sum_reduce_fx_guard_patch()
    patched = fused_module.moe_sum_reduce_torch_compile
    assert moe.apply_moe_sum_reduce_fx_guard_patch()

    assert fused_module.moe_sum_reduce_torch_compile is patched
    assert getattr(patched, moe._PATCHED_FLAG)
    assert getattr(patched, moe._ORIGINAL_ATTR) is compiled_helper
