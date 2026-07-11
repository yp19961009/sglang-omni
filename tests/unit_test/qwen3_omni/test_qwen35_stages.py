# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
