# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen35_omni.components.preprocessor import (
    Qwen35OmniNextProcessor,
)


class _FakeVideoProcessor:
    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            "pixel_values_videos": torch.ones((8, 16)),
            "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
            "video_metadata": [object()],
        }


def test_qwen35_video_processor_uses_presampled_video_path() -> None:
    processor = object.__new__(Qwen35OmniNextProcessor)
    video_processor = _FakeVideoProcessor()
    processor.video_processor = video_processor
    processor.temporal_patch_size = 2
    video = torch.ones((4, 3, 32, 64))

    output = processor._process_videos(
        [video],
        {"fps": 1.25, "device": "cpu"},
    )

    assert video_processor.kwargs is not None
    assert len(video_processor.kwargs["videos"]) == 1
    assert video_processor.kwargs["videos"][0] is video
    assert video_processor.kwargs["do_sample_frames"] is False
    assert video_processor.kwargs["return_metadata"] is True
    assert video_processor.kwargs["device"] == "cpu"
    metadata = video_processor.kwargs["video_metadata"]
    assert len(metadata) == 1
    assert metadata[0]["fps"] == 1.25
    assert torch.equal(metadata[0]["frames_indices"], torch.arange(4))
    assert metadata[0]["total_num_frames"] == 4
    assert metadata[0]["video_backend"] == "preprocessed"
    assert "video_metadata" not in output
    assert output["video_second_per_grid"].tolist() == pytest.approx([1.6])
