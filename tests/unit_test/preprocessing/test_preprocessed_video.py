# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

import numpy as np
import torch

from sglang_omni.client.client import _extract_inputs
from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.preprocessing import audio as audio_mod
from sglang_omni.preprocessing import video as video_mod
from sglang_omni.preprocessing.audio import materialize_preprocessed_audio_list
from sglang_omni.preprocessing.video import materialize_preprocessed_video_list
from sglang_omni.serve.protocol import ChatCompletionRequest


def test_materialize_preprocessed_video_frames_spec() -> None:
    frames = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)

    videos, fps, extracted_audio = materialize_preprocessed_video_list(
        {"frames": frames, "sample_fps": 1.5}
    )

    assert extracted_audio is None
    assert fps == [1.5]
    assert len(videos) == 1
    assert torch.equal(videos[0], torch.from_numpy(frames))


def test_materialize_preprocessed_video_base64_spec() -> None:
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
    encoded = base64.b64encode(frames.tobytes()).decode("ascii")

    videos, fps, _ = materialize_preprocessed_video_list(
        {
            "data": encoded,
            "shape": list(frames.shape),
            "dtype": "float32",
            "layout": "THWC",
            "sample_fps": 2.0,
        }
    )

    assert fps == [2.0]
    assert videos[0].shape == (2, 3, 4, 5)
    assert torch.equal(videos[0], torch.from_numpy(frames).permute(0, 3, 1, 2))


def test_materialize_preprocessed_video_file_spec(tmp_path) -> None:
    frames = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    video_path = tmp_path / "frames.pt"
    torch.save({"video": frames, "sample_fps": 3.0}, video_path)

    videos, fps, _ = materialize_preprocessed_video_list({"path": str(video_path)})

    assert fps == [3.0]
    assert torch.equal(videos[0], frames)


def test_client_extract_inputs_passes_preprocessed_media() -> None:
    video_spec = {"path": "/tmp/frames.pt"}
    audio_spec = {"path": "/tmp/audio.pt"}
    request = GenerateRequest(
        messages=[Message(role="user", content="hi")],
        metadata={
            "preprocessed_videos": [video_spec],
            "preprocessed_video_fps": [1.0],
            "preprocessed_audios": [audio_spec],
            "preprocessed_audio_sample_rate": [16000],
        },
    )

    inputs = _extract_inputs(request)

    assert inputs["preprocessed_videos"] == [video_spec]
    assert inputs["preprocessed_video_fps"] == [1.0]
    assert inputs["preprocessed_audios"] == [audio_spec]
    assert inputs["preprocessed_audio_sample_rate"] == [16000]
    assert "videos" not in inputs
    assert "audios" not in inputs


def test_chat_request_accepts_loaded_videos_alias() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "loaded_videos": [{"path": "/tmp/frames.pt"}],
            "loaded_video_fps": 1.0,
            "loaded_audios": [{"path": "/tmp/audio.pt"}],
            "loaded_audio_sample_rate": 16000,
        }
    )

    assert request.preprocessed_videos == [{"path": "/tmp/frames.pt"}]
    assert request.preprocessed_video_fps == 1.0
    assert request.preprocessed_audios == [{"path": "/tmp/audio.pt"}]
    assert request.preprocessed_audio_sample_rate == 16000



def test_materialize_preprocessed_audio_base64_spec_resamples() -> None:
    audio = np.linspace(-1.0, 1.0, num=8000, dtype=np.float32)
    encoded = base64.b64encode(audio.tobytes()).decode("ascii")

    audios = materialize_preprocessed_audio_list(
        {
            "data": encoded,
            "shape": [8000],
            "dtype": "float32",
            "sample_rate": 8000,
        },
        target_sr=16000,
    )

    assert len(audios) == 1
    assert audios[0].dtype == np.float32
    assert audios[0].shape == (16000,)


def test_materialize_preprocessed_audio_file_spec(tmp_path) -> None:
    audio = np.linspace(-0.5, 0.5, num=16000, dtype=np.float32)
    audio_path = tmp_path / "audio.npz"
    np.savez(audio_path, audio=audio, sample_rate=np.array([16000]))

    audios = materialize_preprocessed_audio_list({"path": str(audio_path)})

    assert len(audios) == 1
    assert np.allclose(audios[0], audio)


def test_preprocessed_video_path_can_reuse_loaded_object(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "video.pt"
    video_path.touch()
    frames = torch.ones((2, 3, 4, 5), dtype=torch.float32)
    calls = 0

    def fake_load(_path):
        nonlocal calls
        calls += 1
        return frames, 1.0

    video_mod._load_preprocessed_video_path_cached.cache_clear()
    monkeypatch.setattr(video_mod, "_load_preprocessed_video_path", fake_load)
    spec = {"path": str(video_path), "reuse_loaded": True}

    first, _, _ = materialize_preprocessed_video_list(spec)
    second, _, _ = materialize_preprocessed_video_list(spec)

    assert calls == 1
    assert first[0].data_ptr() == second[0].data_ptr()
    video_mod._load_preprocessed_video_path_cached.cache_clear()


def test_preprocessed_audio_path_can_reuse_loaded_object(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.pt"
    audio_path.touch()
    audio = np.ones(16000, dtype=np.float32)
    calls = 0

    def fake_load(_path):
        nonlocal calls
        calls += 1
        return audio, 16000

    audio_mod._load_preprocessed_audio_path_cached.cache_clear()
    monkeypatch.setattr(audio_mod, "_load_preprocessed_audio_path", fake_load)
    spec = {"path": str(audio_path), "reuse_loaded": True}

    first = materialize_preprocessed_audio_list(spec)
    second = materialize_preprocessed_audio_list(spec)

    assert calls == 1
    assert np.shares_memory(first[0], second[0])
    audio_mod._load_preprocessed_audio_path_cached.cache_clear()
