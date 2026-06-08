# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.qwen3_5_omni.components import code2wav_scheduler
from sglang_omni.models.qwen3_5_omni.components import (
    qwen3_omni_next_codec_decoder as codec_decoder,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _touch_code2wav_files(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("model:\n  type: public_v1\n")
    (path / "model_weights.pt").write_bytes(b"placeholder")


def _payload(params):
    return StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params=params),
        data={},
    )


def _payload_with_metadata(params, metadata):
    return StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}, params=params, metadata=metadata),
        data={},
    )


class _FakeCode2WavModel:
    total_upsample = 1

    def __call__(self, codes):
        return torch.ones((1, 1, codes.shape[-1]), dtype=torch.float32)


def test_resolve_code2wav_model_dir_prefers_subdir(tmp_path):
    root = tmp_path / "model"
    nested = root / "code2wav"
    _touch_code2wav_files(nested)
    _touch_code2wav_files(root)

    assert code2wav_scheduler._resolve_code2wav_model_dir(str(root)) == nested


def test_resolve_code2wav_model_dir_prefers_qwen35_online_decoder(tmp_path):
    root = tmp_path / "model"
    online_decoder = root / "qwen3_5_omni_codec_decode_online_0306"
    generic_decoder = root / "code2wav"
    _touch_code2wav_files(generic_decoder)
    _touch_code2wav_files(online_decoder)

    assert (
        code2wav_scheduler._resolve_code2wav_model_dir(str(root)) == online_decoder
    )


def test_resolve_code2wav_model_dir_accepts_qwen35_0226_decoder(tmp_path):
    root = tmp_path / "model"
    legacy_decoder = root / "qwen3_5_omni_codec_decode_online_0226"
    generic_decoder = root / "code2wav"
    _touch_code2wav_files(generic_decoder)
    _touch_code2wav_files(legacy_decoder)

    assert (
        code2wav_scheduler._resolve_code2wav_model_dir(str(root)) == legacy_decoder
    )


def test_resolve_code2wav_model_dir_prefers_0306_over_0226(tmp_path):
    root = tmp_path / "model"
    decoder_0306 = root / "qwen3_5_omni_codec_decode_online_0306"
    decoder_0226 = root / "qwen3_5_omni_codec_decode_online_0226"
    _touch_code2wav_files(decoder_0226)
    _touch_code2wav_files(decoder_0306)

    assert code2wav_scheduler._resolve_code2wav_model_dir(str(root)) == decoder_0306


def test_resolve_code2wav_files_accepts_codec_decoder_layout(tmp_path):
    model_root = tmp_path / "model"
    codec_dir = model_root / "codec_decoder"
    codec_dir.mkdir(parents=True)
    config_path = codec_dir / "codec_decoder.yaml"
    checkpoint_path = codec_dir / "checkpoint.pt"
    config_path.write_text("model:\n  type: public_v1\n")
    checkpoint_path.write_bytes(b"placeholder")

    files = code2wav_scheduler._resolve_code2wav_files(str(model_root))

    assert files is not None
    assert files.model_dir == codec_dir
    assert files.config_path == config_path
    assert files.checkpoint_path == checkpoint_path


def test_resolve_code2wav_files_accepts_pytorch_checkpoint_alias(tmp_path):
    model_root = tmp_path / "model"
    dac_dir = model_root / "dac"
    dac_dir.mkdir(parents=True)
    config_path = dac_dir / "dac.yaml"
    checkpoint_path = dac_dir / "model.pth"
    config_path.write_text("model:\n  type: public_v1\n")
    checkpoint_path.write_bytes(b"placeholder")

    files = code2wav_scheduler._resolve_code2wav_files(str(model_root))

    assert files is not None
    assert files.model_dir == dac_dir
    assert files.config_path == config_path
    assert files.checkpoint_path == checkpoint_path


def test_resolve_code2wav_files_accepts_explicit_checkpoint_file(tmp_path):
    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()
    config_path = codec_dir / "config.yaml"
    checkpoint_path = codec_dir / "my_codec_weights.pth"
    config_path.write_text("model:\n  type: public_v1\n")
    checkpoint_path.write_bytes(b"placeholder")

    files = code2wav_scheduler._resolve_code2wav_files(str(checkpoint_path))

    assert files is not None
    assert files.model_dir == codec_dir
    assert files.config_path == config_path
    assert files.checkpoint_path == checkpoint_path


def test_resolve_code2wav_files_ignores_checkpoint_directory(tmp_path):
    model_root = tmp_path / "model"
    code2wav_dir = model_root / "code2wav"
    code2wav_dir.mkdir(parents=True)
    (code2wav_dir / "config.yaml").write_text("model:\n  type: public_v1\n")
    (code2wav_dir / "model_weights.pt").mkdir()

    assert code2wav_scheduler._resolve_code2wav_files(str(model_root)) is None


def test_resolve_code2wav_files_ignores_config_directory(tmp_path):
    model_root = tmp_path / "model"
    code2wav_dir = model_root / "code2wav"
    code2wav_dir.mkdir(parents=True)
    (code2wav_dir / "config.yaml").mkdir()
    (code2wav_dir / "model_weights.pt").write_bytes(b"placeholder")

    assert code2wav_scheduler._resolve_code2wav_files(str(model_root)) is None


def test_missing_code2wav_message_lists_search_layout(tmp_path):
    model_root = tmp_path / "model"
    message = code2wav_scheduler._missing_code2wav_message(
        str(model_root),
        loader_available=False,
    )

    assert str(model_root / "qwen3_5_omni_codec_decode_online_0306") in message
    assert str(model_root / "qwen3_5_omni_codec_decode_online_0226") in message
    assert str(model_root / "code2wav") in message
    assert "config.yaml" in message
    assert "model_weights.pt" in message
    assert "Next DAC loader was not importable" in message


def test_read_codec_eos_token_id_from_talker_config(tmp_path):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        '{"talker_config": {"codec_eos_token_id": 3000}}'
    )

    assert code2wav_scheduler._read_codec_eos_from_config(str(model_root)) == 3000


def test_read_codec_eos_token_id_from_split_talker_config(tmp_path):
    model_root = tmp_path / "model"
    talker_dir = model_root / "talker"
    talker_dir.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model_type": "qwen3_omni_next"}')
    (talker_dir / "config.json").write_text(
        '{"text_config": {"codec_eos_token_id": 3001}}'
    )

    assert code2wav_scheduler._read_codec_eos_from_config(str(model_root)) == 3001


def test_read_codec_eos_token_id_from_split_talker_lm_config(tmp_path):
    model_root = tmp_path / "model"
    talker_lm_dir = model_root / "talker_lm"
    talker_lm_dir.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model_type": "qwen3_omni_next"}')
    (talker_lm_dir / "config.json").write_text(
        '{"text_config": {"codec_eos_token_id": 3002}}'
    )

    assert code2wav_scheduler._read_codec_eos_from_config(str(model_root)) == 3002


def test_missing_code2wav_scheduler_when_next_decoder_unavailable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(code2wav_scheduler, "_load_next_dac_loader", lambda: None)

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(tmp_path),
        device="cpu",
    )

    assert isinstance(scheduler, code2wav_scheduler.MissingQwen35Code2WavScheduler)


def test_missing_code2wav_scheduler_error_includes_model_path(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    monkeypatch.setattr(code2wav_scheduler, "_load_next_dac_loader", lambda: None)
    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(model_root),
        device="cpu",
    )

    scheduler.on_streaming_new_request("req-0", _payload({}))
    messages = scheduler.on_stream_done("req-0")

    assert len(messages) == 1
    assert messages[0].type == "error"
    message = str(messages[0].data)
    assert str(model_root / "qwen3_5_omni_codec_decode_online_0306") in message
    assert str(model_root / "qwen3_5_omni_codec_decode_online_0226") in message
    assert "Accepted config filenames" in message
    assert "Accepted checkpoint filenames" in message


def test_load_code2wav_voice_types_from_legacy_speaker_inputs(tmp_path):
    codec_dir = tmp_path / "code2wav"
    _touch_code2wav_files(codec_dir)
    inputs = codec_dir / "inputs"
    inputs_sft = codec_dir / "inputs_sft4spks"
    inputs.mkdir()
    inputs_sft.mkdir()
    (inputs / "f245_spk_emb.npy").write_bytes(b"placeholder")
    (inputs_sft / "m02_spk_emb.npy").write_bytes(b"placeholder")

    assert code2wav_scheduler._load_code2wav_voice_types(str(codec_dir)) == (
        "f245",
        "m02",
    )


def test_load_code2wav_voice_types_from_spk_dict_maps_display_names(tmp_path):
    codec_dir = tmp_path / "code2wav"
    _touch_code2wav_files(codec_dir)
    torch.save({"Cherry": torch.ones(1), "custom": torch.ones(1)}, codec_dir / "spk_dict.pt")

    assert code2wav_scheduler._load_code2wav_voice_types(str(codec_dir)) == (
        "f245",
        "custom",
    )


def test_prefix_caching_voice_skips_missing_code2wav_model(tmp_path, monkeypatch):
    monkeypatch.setattr(code2wav_scheduler, "_load_next_dac_loader", lambda: None)
    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(tmp_path),
        device="cpu",
    )
    payload = _payload({"voice_type": "tina#prefix_caching"})

    scheduler.on_streaming_new_request("req-0", payload)
    assert scheduler.on_stream_chunk("req-0", SimpleNamespace()) == []
    messages = scheduler.on_stream_done("req-0")

    assert len(messages) == 1
    assert messages[0].type == "result"
    assert messages[0].data.data == {
        "modality": "audio",
        "sample_rate": 24000,
        "skipped": True,
        "reason": "prefix_caching",
    }


def test_prefix_caching_audio_config_skips_missing_code2wav_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(code2wav_scheduler, "_load_next_dac_loader", lambda: None)
    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(tmp_path),
        device="cpu",
    )
    payload = _payload_with_metadata(
        {},
        {"audio_config": {"voice": "tina#prefix_caching"}},
    )

    scheduler.on_streaming_new_request("req-0", payload)
    messages = scheduler.on_stream_done("req-0")

    assert len(messages) == 1
    assert messages[0].type == "result"
    assert messages[0].data.data["skipped"] is True


def test_prefix_caching_audio_metadata_alias_skips_missing_code2wav_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(code2wav_scheduler, "_load_next_dac_loader", lambda: None)
    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(tmp_path),
        device="cpu",
    )
    payload = _payload_with_metadata(
        {},
        {"audio": {"voice": "tina#prefix_caching", "format": "wav"}},
    )

    scheduler.on_streaming_new_request("req-0", payload)
    messages = scheduler.on_stream_done("req-0")

    assert len(messages) == 1
    assert messages[0].type == "result"
    assert messages[0].data.data["skipped"] is True


def test_qwen35_code2wav_scheduler_skips_prefix_caching_chunks():
    scheduler = code2wav_scheduler.Qwen35Code2WavScheduler(
        SimpleNamespace(total_upsample=1),
        device="cpu",
        sample_rate=48000,
    )
    payload = _payload({"voice_type": "prefix_caching"})

    scheduler.on_streaming_new_request("req-0", payload)
    assert scheduler.on_stream_chunk("req-0", SimpleNamespace()) == []
    messages = scheduler.on_stream_done("req-0")

    assert messages[0].data.data["sample_rate"] == 48000
    assert messages[0].data.data["skipped"] is True


def test_qwen35_code2wav_scheduler_validates_available_voice_aliases():
    scheduler = code2wav_scheduler.Qwen35Code2WavScheduler(
        SimpleNamespace(total_upsample=1),
        device="cpu",
        sample_rate=24000,
        code2wav_voice_types=("f245",),
    )

    scheduler.on_streaming_new_request("req-ok", _payload({"voice_type": "Cherry"}))
    scheduler.on_streaming_new_request("req-default", _payload({"voice_type": "default"}))

    with pytest.raises(ValueError, match="available voices"):
        scheduler.on_streaming_new_request(
            "req-bad",
            _payload({"voice_type": "unknown"}),
        )


def test_qwen35_code2wav_scheduler_ignores_audio_input_metadata_payload():
    scheduler = code2wav_scheduler.Qwen35Code2WavScheduler(
        SimpleNamespace(total_upsample=1),
        device="cpu",
        sample_rate=24000,
        code2wav_voice_types=("f245",),
    )

    scheduler.on_streaming_new_request(
        "req-audio-input",
        _payload_with_metadata(
            {},
            {"audio": {"data": "UklGRg==", "format": "wav"}},
        ),
    )


def test_qwen35_code2wav_scheduler_uses_dynamic_chunk_schedule():
    scheduler = code2wav_scheduler.Qwen35Code2WavScheduler(
        _FakeCode2WavModel(),
        device="cpu",
        sample_rate=24000,
        stream_chunk_size=8,
        enable_dynamic_chunk=True,
        dynamic_chunk_sizes=(2, 4),
        dynamic_chunk_steps=(1, 1),
    )
    scheduler.on_streaming_new_request("req-0", _payload({}))

    outputs = []
    for token_id in range(1, 7):
        chunk = SimpleNamespace(
            data=torch.tensor([token_id, token_id, token_id]),
            metadata={"stream": True},
        )
        outputs.append(scheduler.on_stream_chunk("req-0", chunk))

    assert [len(messages) for messages in outputs] == [0, 1, 0, 0, 0, 1]
    assert scheduler._qwen35_dynamic_chunk_index["req-0"] == 2


def test_load_next_dac_loader_finds_local_decoder():
    loader = code2wav_scheduler._load_next_dac_loader()

    assert callable(loader)
    assert loader.__module__.endswith(".qwen3_omni_next_codec_decoder")


def test_load_next_code2wav_model_invokes_available_loader(tmp_path, monkeypatch):
    _touch_code2wav_files(tmp_path)
    calls = {}
    expected_model = object()

    def _fake_loader(*, config_path, checkpoint_path, device, dtype):
        calls.update(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
        )
        return expected_model

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_dac_loader",
        lambda: _fake_loader,
    )

    model = code2wav_scheduler._load_next_code2wav_model(
        str(tmp_path),
        device="cpu",
        dtype=None,
    )

    assert model is expected_model
    assert calls["config_path"].endswith("config.yaml")
    assert calls["checkpoint_path"].endswith("model_weights.pt")
    assert calls["device"] == "cpu"
    assert calls["dtype"] == torch.bfloat16


def test_load_next_code2wav_model_filters_supported_loader_kwargs(
    tmp_path,
    monkeypatch,
):
    _touch_code2wav_files(tmp_path)
    calls = {}
    expected_model = object()

    def _fake_loader(
        *,
        config_path,
        checkpoint_path,
        device,
        dtype,
        odeint_method=None,
    ):
        calls.update(
            odeint_method=odeint_method,
            extra_keys=(
                config_path,
                checkpoint_path,
                device,
                dtype,
            ),
        )
        return expected_model

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_dac_loader",
        lambda: _fake_loader,
    )

    model = code2wav_scheduler._load_next_code2wav_model(
        str(tmp_path),
        device="cpu",
        dtype=None,
        loader_kwargs={"odeint_method": "rk4", "unsupported": "ignored"},
    )

    assert model is expected_model
    assert calls["odeint_method"] == "rk4"


def test_create_code2wav_scheduler_uses_explicit_model_path(tmp_path, monkeypatch):
    root = tmp_path / "model"
    codec_dir = tmp_path / "standalone-code2wav"
    _touch_code2wav_files(codec_dir)
    calls = {}

    def _fake_load_next_code2wav_model(model_path, *, device, dtype):
        calls.update(model_path=model_path, device=device, dtype=dtype)
        return object()

    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        _fake_load_next_code2wav_model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(root),
        code2wav_model_path=str(codec_dir),
        device="cpu",
    )

    assert isinstance(scheduler, FakeCode2WavScheduler)
    assert calls == {
        "model_path": str(codec_dir),
        "device": "cpu",
        "dtype": None,
    }
    assert scheduler.kwargs["sample_rate"] == 24000
    assert scheduler.kwargs["codec_eos_token_id"] == 2150
    assert scheduler.kwargs["stream_chunk_size"] == 8


def test_create_code2wav_scheduler_passes_available_voice_types(tmp_path, monkeypatch):
    codec_dir = tmp_path / "code2wav"
    _touch_code2wav_files(codec_dir)
    torch.save({"Cherry": torch.ones(1)}, codec_dir / "spk_dict.pt")

    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        code2wav_model_path=str(codec_dir),
        device="cpu",
        enable_torch_compile=False,
    )

    assert scheduler.kwargs["code2wav_voice_types"] == ("f245",)


def test_create_code2wav_scheduler_uses_config_codec_eos(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        '{"talker_config": {"codec_eos_token_id": 3000}}'
    )

    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(root),
        device="cpu",
    )

    assert scheduler.kwargs["codec_eos_token_id"] == 3000


def test_create_code2wav_scheduler_uses_split_talker_codec_eos(tmp_path, monkeypatch):
    root = tmp_path / "model"
    talker_dir = root / "talker"
    talker_dir.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type": "qwen3_omni_next"}')
    (talker_dir / "config.json").write_text(
        '{"text_config": {"codec_eos_token_id": 3001}}'
    )

    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        str(root),
        device="cpu",
    )

    assert scheduler.kwargs["codec_eos_token_id"] == 3001


def test_create_code2wav_scheduler_allows_codec_eos_override(monkeypatch):
    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        codec_eos_token_id=3000,
        device="cpu",
    )

    assert scheduler.kwargs["codec_eos_token_id"] == 3000


def test_create_code2wav_scheduler_allows_sample_rate_override(monkeypatch):
    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        sample_rate=48000,
    )

    assert scheduler.kwargs["sample_rate"] == 48000


def test_create_code2wav_scheduler_allows_stream_chunk_override(monkeypatch):
    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        stream_chunk_size=8,
    )

    assert scheduler.kwargs["stream_chunk_size"] == 8


def test_create_code2wav_scheduler_allows_left_context_override(monkeypatch):
    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        left_context_size=0,
    )

    assert scheduler.kwargs["left_context_size"] == 0


def test_create_code2wav_scheduler_allows_dynamic_chunk_override(monkeypatch):
    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_dynamic_chunk=True,
        dynamic_chunk_sizes=(2, 4),
        dynamic_chunk_steps=(1, 1),
    )

    assert scheduler.kwargs["enable_dynamic_chunk"] is True
    assert scheduler.kwargs["dynamic_chunk_sizes"] == (2, 4)
    assert scheduler.kwargs["dynamic_chunk_steps"] == (1, 1)


def test_create_code2wav_scheduler_normalizes_scalar_dynamic_chunk_args(monkeypatch):
    class FakeCode2WavScheduler:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_dynamic_chunk=True,
        dynamic_chunk_sizes=4,
        dynamic_chunk_steps=2,
    )

    assert scheduler.kwargs["enable_dynamic_chunk"] is True
    assert scheduler.kwargs["dynamic_chunk_sizes"] == (4,)
    assert scheduler.kwargs["dynamic_chunk_steps"] == (2,)


def test_create_code2wav_scheduler_validates_dynamic_chunk_before_decoder_load(
    monkeypatch,
):
    calls = {}

    def _fake_load_next_code2wav_model(model_path, *, device, dtype):
        calls["loaded"] = True
        return None

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        _fake_load_next_code2wav_model,
    )

    with pytest.raises(ValueError, match="same length"):
        code2wav_scheduler.create_code2wav_scheduler(
            "/models/qwen35",
            device="cpu",
            enable_dynamic_chunk=True,
            dynamic_chunk_sizes=(2, 4),
            dynamic_chunk_steps=(1,),
        )

    assert calls == {}


def test_create_code2wav_scheduler_validates_audio_args_before_decoder_load(
    monkeypatch,
):
    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: None,
    )

    with pytest.raises(ValueError, match="code2wav_sample_rate"):
        code2wav_scheduler.create_code2wav_scheduler(
            "/models/qwen35",
            device="cpu",
            sample_rate=0,
        )


def test_apply_code2wav_compile_prefers_decode_inner(monkeypatch):
    compiled = []

    def _fake_compile(fn):
        compiled.append(fn.__name__)
        return f"compiled-{fn.__name__}"

    class Model:
        def _decode_inner(self):
            return None

        def decode(self):
            return None

    monkeypatch.setattr(code2wav_scheduler.torch, "compile", _fake_compile)
    model = Model()

    assert code2wav_scheduler._apply_code2wav_torch_compile(model) is model
    assert compiled == ["_decode_inner"]
    assert model._decode_inner == "compiled-_decode_inner"


def test_apply_code2wav_compile_uses_model_method_before_decode(monkeypatch):
    compiled = []

    def _fake_compile(fn):
        compiled.append(fn.__name__)
        return fn

    class Model:
        def __init__(self):
            self.enabled = False

        def enable_torch_compile(self):
            self.enabled = True

        def decode(self):
            return None

    monkeypatch.setattr(code2wav_scheduler.torch, "compile", _fake_compile)
    model = Model()

    code2wav_scheduler._apply_code2wav_torch_compile(model)

    assert model.enabled is True
    assert compiled == []


def test_apply_code2wav_compile_forwards_first_chunk_flag(monkeypatch):
    compiled = []

    def _fake_compile(fn):
        compiled.append(fn.__name__)
        return fn

    class Model:
        def __init__(self):
            self.compile_first_chunk = None

        def enable_torch_compile(self, compile_first_chunk=False):
            self.compile_first_chunk = compile_first_chunk

    monkeypatch.setattr(code2wav_scheduler.torch, "compile", _fake_compile)
    model = Model()

    code2wav_scheduler._apply_code2wav_torch_compile(
        model,
        compile_first_chunk=True,
    )

    assert model.compile_first_chunk is True
    assert compiled == []


def test_create_code2wav_scheduler_applies_compile_flag(monkeypatch):
    model = SimpleNamespace(total_upsample=1)
    calls = {}

    def _fake_load_next_code2wav_model(model_path, *, device, dtype):
        calls.update(model_path=model_path, device=device, dtype=dtype)
        return model

    def _fake_apply_compile(loaded_model, **kwargs):
        calls["compiled"] = loaded_model
        calls["compile_kwargs"] = kwargs
        return loaded_model

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        _fake_load_next_code2wav_model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "_apply_code2wav_torch_compile",
        _fake_apply_compile,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_torch_compile=True,
    )

    assert scheduler.model is model
    assert calls["compiled"] is model
    assert calls["compile_kwargs"] == {"compile_first_chunk": False}


def test_create_code2wav_scheduler_applies_compile_first_chunk_flag(monkeypatch):
    model = SimpleNamespace(total_upsample=1)
    calls = {}

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype, loader_kwargs=None: model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "_apply_code2wav_torch_compile",
        lambda loaded_model, **kwargs: calls.setdefault(
            "compile_kwargs",
            kwargs,
        )
        and loaded_model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        code2wav_enable_torch_compile_first_chunk=True,
    )

    assert scheduler.model is model
    assert calls["compile_kwargs"] == {"compile_first_chunk": True}


def test_create_code2wav_scheduler_applies_vllm_runtime_options(monkeypatch):
    odeint_kwargs = {"method": "euler"}
    model = SimpleNamespace(
        total_upsample=1,
        code2wav_dit_model=SimpleNamespace(
            cfm_model=SimpleNamespace(odeint_kwargs=odeint_kwargs),
        ),
        odeint_method_relaxed=False,
        bs_mel=24,
        chunk_size=24,
    )

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype, loader_kwargs=None: model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_torch_compile=False,
        code2wav_odeint_method="RK4",
        code2wav_odeint_method_relaxed=True,
        code2wav_batched_chunk=2,
    )

    assert scheduler.model is model
    assert scheduler.kwargs["sample_rate"] == 24000
    assert odeint_kwargs["method"] == "rk4"
    assert model.odeint_method_relaxed is True
    assert model.chunk_size == 48


def test_create_code2wav_scheduler_applies_frequency_default_chunk(monkeypatch):
    model = SimpleNamespace(
        total_upsample=1,
        frequency="50hz",
        code2wav_frequency="50hz",
        dit_quant=None,
        code2wav_dit_quant=None,
        bs_mel=32,
        chunk_size=64,
    )
    calls = {}

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    def _fake_load_next_code2wav_model(
        model_path,
        *,
        device,
        dtype,
        loader_kwargs=None,
    ):
        calls["loader_kwargs"] = loader_kwargs
        return model

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        _fake_load_next_code2wav_model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_torch_compile=False,
        code2wav_frequency="25HZ",
        code2wav_dit_quant="FP8",
    )

    assert scheduler.model is model
    assert calls["loader_kwargs"] == {
        "batched_chunk": 1,
        "frequency": "25hz",
        "dit_quant": "fp8",
    }
    assert model.frequency == "25hz"
    assert model.code2wav_frequency == "25hz"
    assert model.dit_quant == "fp8"
    assert model.code2wav_dit_quant == "fp8"
    assert model.chunk_size == 32


def test_create_code2wav_scheduler_accepts_vllm_frequency_alias(monkeypatch):
    model = SimpleNamespace(
        total_upsample=1,
        frequency="25hz",
        code2wav_frequency="25hz",
        bs_mel=32,
        chunk_size=64,
    )
    calls = {}

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    def _fake_load_next_code2wav_model(
        model_path,
        *,
        device,
        dtype,
        loader_kwargs=None,
    ):
        calls["loader_kwargs"] = loader_kwargs
        return model

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        _fake_load_next_code2wav_model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_torch_compile=False,
        frequency=1,
        code2wav_frequency="50hz",
    )

    assert scheduler.model is model
    assert calls["loader_kwargs"] == {
        "batched_chunk": 2,
        "frequency": "50hz",
    }
    assert model.frequency == "50hz"
    assert model.code2wav_frequency == "50hz"
    assert model.chunk_size == 64


def test_create_code2wav_scheduler_rejects_bad_runtime_options(monkeypatch):
    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: SimpleNamespace(total_upsample=1),
    )

    try:
        code2wav_scheduler.create_code2wav_scheduler(
            "/models/qwen35",
            device="cpu",
            enable_torch_compile=False,
            code2wav_odeint_method="bogus",
        )
    except ValueError as exc:
        assert "odeint_method" in str(exc)
    else:
        raise AssertionError("expected invalid odeint method to raise")


def test_create_code2wav_scheduler_defaults_to_torch_compile(monkeypatch):
    model = SimpleNamespace(total_upsample=1)
    calls = {}

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "_apply_code2wav_torch_compile",
        lambda loaded_model, **kwargs: calls.setdefault("compiled", loaded_model),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
    )

    assert scheduler.model is model
    assert calls["compiled"] is model


def test_create_code2wav_scheduler_allows_disabling_default_compile(monkeypatch):
    model = SimpleNamespace(total_upsample=1)
    calls = {}

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda model_path, *, device, dtype: model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "_apply_code2wav_torch_compile",
        lambda loaded_model, **kwargs: calls.setdefault("compiled", loaded_model),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        enable_torch_compile=False,
    )

    assert scheduler.model is model
    assert "compiled" not in calls


def test_create_code2wav_scheduler_coerces_string_bool_options(monkeypatch):
    model = SimpleNamespace(total_upsample=1, odeint_method_relaxed=False)
    calls = {}

    class FakeCode2WavScheduler:
        def __init__(self, loaded_model, **kwargs):
            self.model = loaded_model
            self.kwargs = kwargs

    def _fake_load_next_code2wav_model(model_path, *, device, dtype, loader_kwargs=None):
        calls["loader_kwargs"] = loader_kwargs
        return model

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        _fake_load_next_code2wav_model,
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "_apply_code2wav_torch_compile",
        lambda loaded_model, **kwargs: calls.setdefault("compiled", loaded_model),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Qwen35Code2WavScheduler",
        FakeCode2WavScheduler,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "/models/qwen35",
        device="cpu",
        code2wav_enable_torch_compile="off",
        code2wav_enable_torch_compile_first_chunk="yes",
        code2wav_odeint_method_relaxed="on",
        enable_dynamic_chunk="true",
    )

    assert scheduler.model is model
    assert scheduler.kwargs["enable_dynamic_chunk"] is True
    assert model.odeint_method_relaxed is True
    assert calls["loader_kwargs"] == {
        "enable_torch_compile_first_chunk": True,
        "odeint_method_relaxed": True,
    }
    assert "compiled" not in calls


def test_create_code2wav_scheduler_rejects_bad_bool_string(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        code2wav_scheduler,
        "_load_next_code2wav_model",
        lambda *args, **kwargs: calls.setdefault("loaded", True),
    )

    with pytest.raises(ValueError, match="code2wav_enable_dynamic_chunk"):
        code2wav_scheduler.create_code2wav_scheduler(
            "/models/qwen35",
            device="cpu",
            enable_dynamic_chunk="maybe",
        )

    assert "loaded" not in calls


def test_next_dac_loader_accepts_latent_dim_aliases():
    assert (
        codec_decoder._get_required_hparam(
            SimpleNamespace(latet_dim=8),
            "latet_dim",
            "latent_dim",
        )
        == 8
    )
    assert (
        codec_decoder._get_required_hparam(
            SimpleNamespace(latent_dim=9),
            "latet_dim",
            "latent_dim",
        )
        == 9
    )


def test_next_dac_checkpoint_loader_falls_back_for_older_torch(monkeypatch):
    calls = []
    tensor = torch.ones(1)

    def _fake_torch_load(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("mmap") is True:
            raise TypeError("mmap is unsupported")
        return {"model": {"generator.proj.weight": tensor, "skip.weight": tensor}}

    monkeypatch.setattr(codec_decoder.torch, "load", _fake_torch_load)

    state_dict = codec_decoder._load_checkpoint_state_dict("dummy.pt", "cpu")

    assert calls[0]["mmap"] is True
    assert "mmap" not in calls[1]
    assert state_dict == {"proj.weight": tensor}


def test_next_dac_checkpoint_loader_accepts_state_dict_wrapper(monkeypatch):
    tensor = torch.ones(1)

    def _fake_torch_load(path, **kwargs):
        del path, kwargs
        return {
            "state_dict": {
                "module.generator.proj.weight": tensor,
                "module.skip.weight": tensor,
            }
        }

    monkeypatch.setattr(codec_decoder.torch, "load", _fake_torch_load)

    state_dict = codec_decoder._load_checkpoint_state_dict("dummy.pth", "cpu")

    assert state_dict == {"proj.weight": tensor}


def test_next_dac_checkpoint_loader_accepts_generator_wrapper(monkeypatch):
    tensor = torch.ones(1)

    def _fake_torch_load(path, **kwargs):
        del path, kwargs
        return {"generator": {"proj.weight": tensor}}

    monkeypatch.setattr(codec_decoder.torch, "load", _fake_torch_load)

    state_dict = codec_decoder._load_checkpoint_state_dict("dummy.pth", "cpu")

    assert state_dict == {"proj.weight": tensor}


def test_next_dac_model_loader_uses_assign_when_supported():
    calls = []

    class Model:
        def load_state_dict(self, state_dict, *, strict=False, assign=False):
            calls.append((state_dict, strict, assign))
            return "loaded"

    state_dict = {"proj.weight": torch.ones(1)}

    result = codec_decoder._load_model_state_dict(Model(), state_dict)

    assert result == "loaded"
    assert calls == [(state_dict, False, True)]


def test_next_dac_model_loader_falls_back_without_assign():
    calls = []

    class Model:
        def load_state_dict(self, state_dict, strict=False):
            calls.append((state_dict, strict))
            return "loaded"

    state_dict = {"proj.weight": torch.ones(1)}

    result = codec_decoder._load_model_state_dict(Model(), state_dict)

    assert result == "loaded"
    assert calls == [(state_dict, False)]


def _post_module_args(**overrides):
    values = {
        "block_size": 8,
        "n_layer": 1,
        "n_head": 2,
        "dim": 4,
        "intermediate_size": 8,
        "n_local_heads": 2,
        "head_dim": 2,
        "rope_base": 10000,
        "norm_eps": 1e-5,
        "dropout_rate": 0.0,
        "attn_dropout_rate": 0.0,
        "channels_first": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_next_dac_model_args_preserve_conformer_config():
    args = _post_module_args(
        pos_embed_type="conformer",
        max_relative_position=3,
    )

    config = codec_decoder._build_post_module_model_args(args)

    assert config.pos_embed_type == "conformer"
    assert config.max_relative_position == 3


def test_next_dac_conformer_attention_forward_cpu():
    config = codec_decoder.ModelArgs(
        block_size=8,
        n_layer=1,
        n_head=2,
        dim=4,
        intermediate_size=8,
        n_local_heads=2,
        head_dim=2,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        pos_embed_type="conformer",
        max_relative_position=2,
    )
    attention = codec_decoder.Attention(config).eval()
    hidden_states = torch.randn(1, 3, 4)
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)

    output = attention(
        hidden_states,
        freqs_cis=None,
        mask=mask,
        input_pos=torch.arange(3),
    )

    assert output.shape == hidden_states.shape


def test_next_dac_conformer_window_transformer_forward_cpu():
    config = codec_decoder._build_post_module_model_args(
        _post_module_args(
            pos_embed_type="conformer",
            max_relative_position=4,
        )
    )
    model = codec_decoder.WindowLimitedTransformer(
        config=config,
        input_dim=4,
        window_size=None,
    ).eval()
    hidden_states = torch.randn(1, 3, 4)

    output = model(hidden_states)

    assert output.shape == hidden_states.shape


def test_next_dac_exposes_sglang_total_upsample_contract():
    model = codec_decoder.DAC(
        latent_dim=4,
        decoder_dim=16,
        decoder_rates=[8, 5, 4, 3],
        decoder_transformer_layers=[0, 0, 0, 0],
        pre_transformer=nn.Identity(),
        codebook_size=8,
        codebook_nums=3,
    )

    assert model.total_upsample == 1920


def test_next_dac_forward_adapts_sglang_codec_layout():
    model = codec_decoder.DAC(
        latent_dim=4,
        decoder_dim=8,
        decoder_rates=[2],
        decoder_transformer_layers=[0],
        pre_transformer=nn.Identity(),
        codebook_size=8,
        codebook_nums=3,
    )
    seen = {}

    def _fake_decode(codes):
        seen["codes"] = codes.clone()
        return torch.zeros(codes.shape[0], 1, codes.shape[1])

    model.decode = _fake_decode
    scheduler_codes = torch.tensor([[[1, 2], [3, 4], [5, 6]]])

    output = model(scheduler_codes)

    assert torch.equal(seen["codes"], torch.tensor([[[1, 3, 5], [2, 4, 6]]]))
    assert output.shape == (1, 1, 2)
