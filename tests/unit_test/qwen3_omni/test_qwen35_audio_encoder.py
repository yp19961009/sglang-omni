# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen35_omni.components.audio_encoder import (
    Qwen35OmniNextAudioAttention,
)


def test_qwen35_audio_attention_cpu_boundaries_preserve_output() -> None:
    torch.manual_seed(17)
    attention = Qwen35OmniNextAudioAttention(
        SimpleNamespace(d_model=8, encoder_attention_heads=2)
    )
    hidden_states = torch.randn((7, 8))
    tensor_boundaries = torch.tensor([0, 3, 7], dtype=torch.int32)

    with torch.inference_mode():
        tensor_output = attention(hidden_states, tensor_boundaries)
        list_output = attention(hidden_states, tensor_boundaries.tolist())

    assert torch.equal(list_output, tensor_output)
