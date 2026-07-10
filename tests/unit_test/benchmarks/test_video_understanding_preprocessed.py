# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from benchmarks.dataset.videomme import VideoAMMESample
from benchmarks.tasks.video_understanding import make_video_send_fn


class _FakeResponse:
    def raise_for_status(self):
        return None

    async def json(self):
        return {
            "choices": [{"message": {"content": "Answer: A"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self):
        self.payload = None

    def post(self, _url, *, json):
        self.payload = json
        return _FakeResponse()


def _sample() -> VideoAMMESample:
    return VideoAMMESample(
        sample_id="001-1",
        video_path="/tmp/raw-video.mp4",
        audio_path="/tmp/raw-audio.wav",
        question="Question?",
        options=["a", "b", "c", "d"],
        answer="A",
        video_id="001",
        question_id="001-1",
        prompt="Prompt",
    )


@pytest.mark.parametrize("reuse_loaded", [False, True])
def test_video_send_fn_can_send_preprocessed_videoamme_payload(
    tmp_path, reuse_loaded
):
    video_dir = tmp_path / "videos"
    audio_dir = tmp_path / "audios"
    video_dir.mkdir()
    audio_dir.mkdir()
    video_path = video_dir / "001_fps1p0_frames128_px401408.pt"
    audio_path = audio_dir / "001-1_sr16000.pt"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")

    send_fn = make_video_send_fn(
        "qwen35-omni-s2t",
        "http://127.0.0.1:8011/v1/chat/completions",
        video_fps=1.0,
        video_max_frames=128,
        video_max_pixels=401408,
        preprocessed_video_dir=str(video_dir),
        preprocessed_audio_dir=str(audio_dir),
        reuse_preprocessed_media=reuse_loaded,
        enable_audio_input=True,
        fixed_prompt="Prompt",
    )
    session = _FakeSession()

    result = asyncio.run(send_fn(session, _sample()))

    assert result.is_success
    expected_video = {"path": str(video_path)}
    expected_audio = {"path": str(audio_path)}
    if reuse_loaded:
        expected_video["reuse_loaded"] = True
        expected_audio["reuse_loaded"] = True
    assert session.payload["preprocessed_videos"] == [expected_video]
    assert session.payload["preprocessed_audios"] == [expected_audio]
    assert "videos" not in session.payload
    assert "audios" not in session.payload
    assert "video_fps" not in session.payload
    assert "video_max_frames" not in session.payload
    assert "video_max_pixels" not in session.payload
