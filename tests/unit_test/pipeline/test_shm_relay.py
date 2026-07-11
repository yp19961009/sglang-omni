# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os

import torch

from sglang_omni.pipeline import relay_io
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.relay.shm import ShmRelay


class _FastPathOnlyShmRelay(ShmRelay):
    async def put_async(self, *args, **kwargs):
        raise AssertionError("payload should use put_many_async")

    async def get_async(self, *args, **kwargs):
        raise AssertionError("payload should use map_tensor")


def test_shm_payload_round_trip_uses_segmented_write_and_mmap_read() -> None:
    async def _run() -> None:
        relay = _FastPathOnlyShmRelay(engine_id="test")
        video = torch.arange(96, dtype=torch.float32).reshape(12, 8).bfloat16()
        grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
        payload = StagePayload(
            request_id="req-shm-fast-path",
            request=OmniRequest(inputs={"model": "test"}),
            data={"pixel_values_videos": video, "video_grid_thw": grid},
        )

        metadata, op = await relay_io.write_payload(relay, payload.request_id, payload)
        await op.wait_for_completion()
        shm_name = metadata["relay_info"]["transfer_info"]["shm_name"]
        shm_path = os.path.join("/dev/shm", shm_name.lstrip("/"))
        assert os.path.exists(shm_path)

        restored = await relay_io.read_payload(relay, payload.request_id, metadata)

        assert not os.path.exists(shm_path)
        assert torch.equal(restored.data["pixel_values_videos"], video)
        assert torch.equal(restored.data["video_grid_thw"], grid)
        assert restored.data["pixel_values_videos"].dtype == torch.bfloat16

    asyncio.run(_run())
