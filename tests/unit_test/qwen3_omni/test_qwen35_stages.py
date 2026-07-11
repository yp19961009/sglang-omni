# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen35_omni import stages


@pytest.mark.parametrize(
    ("hopper_available", "expected"),
    ((True, "fa3"), (False, "triton")),
)
def test_qwen35_thinker_attention_backend_tracks_hopper_support(
    monkeypatch: pytest.MonkeyPatch,
    hopper_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(
        stages,
        "is_hopper_with_cuda_12_3",
        lambda: hopper_available,
    )

    assert stages._default_thinker_attention_backend() == expected


def test_qwen35_thinker_disables_radix_cache_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _build_server_args(model_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            mem_fraction_static=kwargs.get("mem_fraction_static", 0.5),
            **kwargs,
        )

    memory_contract = SimpleNamespace(
        mem_fraction_static_pinned=False,
        effective_total_gpu_memory_fraction=None,
        applied_encoder_mem_reserve=0.0,
    )
    monkeypatch.setattr(stages, "build_sglang_server_args", _build_server_args)
    monkeypatch.setattr(
        stages, "create_thinker_scheduler", lambda *args, **kwargs: "ok"
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda gpu_id: 90.0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda gpu_id: 0)
    monkeypatch.setattr(
        stages.qwen3_stages,
        "_apply_colocated_ar_memory_contract",
        lambda *args, **kwargs: memory_contract,
    )
    monkeypatch.setattr(
        stages.qwen3_stages,
        "_apply_qwen_thinker_encoder_reserve",
        lambda *args, **kwargs: False,
    )

    assert stages.create_sglang_thinker_executor_from_config("/tmp/model") == "ok"
    assert captured["disable_radix_cache"] is True
