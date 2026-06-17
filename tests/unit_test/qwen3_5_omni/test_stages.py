# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_5_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_5_omni import stages
from sglang_omni.proto import OmniRequest, StagePayload


def _split_root(tmp_path):
    root = tmp_path / "qwen35"
    thinker = root / "thinker"
    thinker.mkdir(parents=True)
    (thinker / "config.json").write_text("{}", encoding="utf-8")
    return root, thinker


def _split_root_with_talker(tmp_path):
    root, thinker = _split_root(tmp_path)
    talker = root / "talker"
    talker.mkdir(parents=True)
    (talker / "config.json").write_text("{}", encoding="utf-8")
    return root, thinker, talker


def _split_root_with_talker_lm(tmp_path):
    root, thinker = _split_root(tmp_path)
    talker_lm = root / "talker_lm"
    talker_lm.mkdir(parents=True)
    (talker_lm / "config.json").write_text("{}", encoding="utf-8")
    return root, thinker, talker_lm


def test_resolve_qwen35_stage_model_path_uses_local_thinker_subdir(tmp_path):
    root, thinker = _split_root(tmp_path)

    assert (
        stages._resolve_qwen35_stage_model_path(str(root), stages.THINKER_STAGE)
        == str(thinker)
    )
    assert (
        stages._resolve_qwen35_stage_model_path(str(root), stages.IMAGE_STAGE)
        == str(thinker)
    )
    assert (
        stages._resolve_qwen35_stage_model_path(str(root), "talker_ar")
        == str(root)
    )
    assert (
        stages._resolve_qwen35_stage_model_path(
            "Qwen/Qwen3.5-Omni",
            stages.THINKER_STAGE,
        )
        == "Qwen/Qwen3.5-Omni"
    )


def test_resolve_qwen35_stage_model_path_uses_local_talker_subdir(tmp_path):
    root, _, talker = _split_root_with_talker(tmp_path)

    assert (
        stages._resolve_qwen35_stage_model_path(str(root), stages.TALKER_STAGE)
        == str(talker)
    )


def test_resolve_qwen35_stage_model_path_prefers_talker_lm_subdir(tmp_path):
    root, _, talker = _split_root_with_talker(tmp_path)
    talker_lm = root / "talker_lm"
    talker_lm.mkdir(parents=True)
    (talker_lm / "config.json").write_text("{}", encoding="utf-8")

    assert (
        stages._resolve_qwen35_stage_model_path(str(root), stages.TALKER_STAGE)
        == str(talker_lm)
    )
    assert talker != talker_lm


def test_resolve_qwen35_stage_model_path_requires_config_json(tmp_path):
    root = tmp_path / "qwen35"
    (root / "thinker").mkdir(parents=True)

    assert (
        stages._resolve_qwen35_stage_model_path(str(root), stages.THINKER_STAGE)
        == str(root)
    )


def test_qwen35_preprocessing_executor_uses_thinker_subdir(monkeypatch, tmp_path):
    root, thinker = _split_root(tmp_path)
    seen = {}

    class FakePreprocessor:
        def __init__(self, *, model_path, **kwargs):
            seen["model_path"] = model_path
            seen["kwargs"] = kwargs

        async def __call__(self, payload):
            return payload

    monkeypatch.setattr(stages, "Qwen35OmniPreprocessor", FakePreprocessor)

    stages.create_preprocessing_executor(
        str(root),
        thinker_max_seq_len=123,
        video_seconds_per_chunk=2.0,
        video_position_id_per_seconds=25.0,
        audio_target_sr=16000,
        audio_timestamp_interval=30,
        audio_downsample_times=4,
        audio_downsample_chunk_size=100,
    )

    assert seen["model_path"] == str(thinker)
    assert seen["kwargs"]["max_seq_len"] == 123
    assert seen["kwargs"]["video_seconds_per_chunk"] == 2.0
    assert seen["kwargs"]["video_position_id_per_seconds"] == 25.0
    assert seen["kwargs"]["audio_target_sr"] == 16000
    assert seen["kwargs"]["audio_timestamp_interval"] == 30
    assert seen["kwargs"]["audio_downsample_times"] == 4
    assert seen["kwargs"]["audio_downsample_chunk_size"] == 100


def test_load_qwen35_encoder_uses_local_image_impl(monkeypatch):
    module_name = "sglang_omni.models.qwen3_5_omni.components.image_encoder"
    module = ModuleType(module_name)

    class FakeImageEncoder:
        def __init__(self, *, model_path, device, dtype):
            self.model_path = model_path
            self.device = device
            self.dtype = dtype

    module.Qwen35OmniImageEncoder = FakeImageEncoder
    monkeypatch.setitem(sys.modules, module_name, module)

    model = stages._load_qwen35_encoder(
        stages.IMAGE_STAGE,
        model_path="/models/qwen35",
        device="cpu",
        dtype="float16",
    )

    assert isinstance(model, FakeImageEncoder)
    assert model.model_path == "/models/qwen35"
    assert model.device == "cpu"
    assert model.dtype == "float16"


def test_load_qwen35_encoder_returns_none_without_impl():
    assert (
        stages._load_qwen35_encoder(
            "unknown",
            model_path="/models/qwen35",
            device="cpu",
            dtype=None,
        )
        is None
    )


def test_load_qwen35_encoder_raises_for_known_missing_impl(monkeypatch):
    monkeypatch.setitem(
        stages._ENCODER_IMPL_CANDIDATES,
        stages.IMAGE_STAGE,
        (("missing.qwen35.encoder", ("Nope",)),),
    )

    with pytest.raises(ImportError, match=stages.IMAGE_STAGE):
        stages._load_qwen35_encoder(
            stages.IMAGE_STAGE,
            model_path="/models/qwen35",
            device="cpu",
            dtype=None,
        )


class _FakeImageEncoder:
    spatial_merge_size = 1
    out_hidden_size = 8
    deepstack_layers = 0
    visual_dtype_bytes = 2

    def __call__(self, **model_inputs):
        return {
            "image_embeds": model_inputs["pixel_values"].new_zeros((1, 8)),
            "image_grid_thw": model_inputs["image_grid_thw"],
            "image_token_counts": torch.tensor([1]),
            "deepstack_visual_embeds_image": [],
        }


class _FakeAudioEncoder:
    def __call__(self, **model_inputs):
        lengths = model_inputs["audio_feature_lengths"]
        return {
            "audio_embeds": model_inputs["input_features"].new_zeros((1, 4)),
            "audio_feature_lengths": lengths,
            "audio_output_lengths": torch.ones_like(lengths),
        }


def _payload(stage_name, model_inputs):
    state = Qwen3OmniPipelineState(
        encoder_inputs={stage_name: dict(model_inputs)},
    )
    return StagePayload(
        request_id="req-0",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )


def test_qwen35_image_encoder_executor_reuses_batched_qwen_path(monkeypatch):
    monkeypatch.setattr(stages, "_emit_event", lambda **_: None)
    monkeypatch.setattr(
        stages,
        "_load_qwen35_encoder",
        lambda *_, **__: _FakeImageEncoder(),
    )

    scheduler = stages.create_image_encoder_executor("/models/qwen35", device="cpu")

    assert scheduler._batch_fn is not None
    assert scheduler._max_batch_size == 32
    assert scheduler._max_batch_wait_s == pytest.approx(0.05)
    assert scheduler._request_cost_fn is not None
    assert scheduler._max_batch_cost == stages.QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES


def test_qwen35_image_encoder_executor_uses_thinker_subdir(monkeypatch, tmp_path):
    root, thinker = _split_root(tmp_path)
    seen = {}

    def fake_load(stage_name, *, model_path, device, dtype):
        seen["stage_name"] = stage_name
        seen["model_path"] = model_path
        return _FakeImageEncoder()

    monkeypatch.setattr(stages, "_emit_event", lambda **_: None)
    monkeypatch.setattr(stages, "_load_qwen35_encoder", fake_load)

    stages.create_image_encoder_executor(str(root), device="cpu")

    assert seen["stage_name"] == stages.IMAGE_STAGE
    assert seen["model_path"] == str(thinker)


def test_qwen35_audio_encoder_executor_writes_encoder_outputs(monkeypatch):
    monkeypatch.setattr(stages, "_emit_event", lambda **_: None)
    monkeypatch.setattr(
        stages,
        "_load_qwen35_encoder",
        lambda *_, **__: _FakeAudioEncoder(),
    )
    scheduler = stages.create_audio_encoder_executor("/models/qwen35", device="cpu")
    payload = _payload(
        stages.AUDIO_STAGE,
        {
            "input_features": torch.ones(1, 4),
            "audio_feature_lengths": torch.tensor([4]),
        },
    )

    result = scheduler._fn(payload)
    state = Qwen3OmniPipelineState.from_dict(result.data)
    audio_out = state.encoder_outs[stages.AUDIO_STAGE]

    assert scheduler._batch_fn is not None
    assert audio_out["audio_embeds"].shape == (1, 4)
    assert audio_out["audio_output_lengths"].tolist() == [1]


def test_qwen35_thinker_executor_uses_thinker_subdir(monkeypatch, tmp_path):
    root, thinker = _split_root(tmp_path)
    seen = {}

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        seen["model_path"] = model_path
        seen["context_length"] = context_length
        seen["overrides"] = overrides
        return SimpleNamespace(mem_fraction_static=None)

    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        stages,
        "_apply_colocated_ar_memory_contract",
        lambda *_, **__: SimpleNamespace(
            mem_fraction_static_pinned=False,
            effective_total_gpu_memory_fraction=None,
            applied_encoder_mem_reserve=0.0,
        ),
    )
    monkeypatch.setattr(
        stages,
        "_apply_qwen_thinker_encoder_reserve",
        lambda *_, **__: False,
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)

    def fake_create_thinker_scheduler(*args, **kwargs):
        seen["scheduler_args"] = args
        seen["scheduler_kwargs"] = kwargs
        return "scheduler"

    monkeypatch.setattr(
        stages,
        "create_thinker_scheduler",
        fake_create_thinker_scheduler,
    )

    result = stages.create_sglang_thinker_executor_from_config(
        str(root),
        thinker_max_seq_len=4096,
    )

    assert result == "scheduler"
    assert seen["model_path"] == str(thinker)
    assert seen["context_length"] == 4096
    assert seen["scheduler_kwargs"]["root_model_path"] == str(root)


def test_qwen35_thinker_executor_reads_mamba_strategy_env(monkeypatch):
    seen = {}

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        seen["model_path"] = model_path
        seen["context_length"] = context_length
        seen["overrides"] = overrides
        return SimpleNamespace(mem_fraction_static=None)

    monkeypatch.setenv("QWEN35_THINKER_MAMBA_SCHEDULER_STRATEGY", "extra_buffer")
    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        stages,
        "_apply_colocated_ar_memory_contract",
        lambda *_, **__: SimpleNamespace(
            mem_fraction_static_pinned=False,
            effective_total_gpu_memory_fraction=None,
            applied_encoder_mem_reserve=0.0,
        ),
    )
    monkeypatch.setattr(
        stages,
        "_apply_qwen_thinker_encoder_reserve",
        lambda *_, **__: False,
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)
    monkeypatch.setattr(
        stages,
        "create_thinker_scheduler",
        lambda *_, **__: "scheduler",
    )

    result = stages.create_sglang_thinker_executor_from_config("/models/qwen35")

    assert result == "scheduler"
    assert seen["model_path"] == "/models/qwen35"
    assert seen["overrides"]["mamba_scheduler_strategy"] == "extra_buffer"


def test_qwen35_thinker_colocated_reserves_encoder_memory(monkeypatch):
    seen = {}
    monkeypatch.delenv("QWEN35_THINKER_MAMBA_SCHEDULER_STRATEGY", raising=False)

    def fake_apply_colocated_contract(overrides, **kwargs):
        seen["contract_overrides_before"] = dict(overrides)
        seen["contract_kwargs"] = kwargs
        overrides["mem_fraction_static"] = 0.70
        return SimpleNamespace(
            mem_fraction_static_pinned=True,
            effective_total_gpu_memory_fraction=0.70,
            applied_encoder_mem_reserve=0.05,
        )

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        seen["model_path"] = model_path
        seen["context_length"] = context_length
        seen["overrides"] = overrides
        return SimpleNamespace(
            mem_fraction_static=overrides.get("mem_fraction_static"),
        )

    def fake_create_thinker_scheduler(*args, **kwargs):
        seen["scheduler_args"] = args
        seen["scheduler_kwargs"] = kwargs
        return "scheduler"

    monkeypatch.setattr(
        stages,
        "_apply_colocated_ar_memory_contract",
        fake_apply_colocated_contract,
    )
    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)
    monkeypatch.setattr(
        stages,
        "create_thinker_scheduler",
        fake_create_thinker_scheduler,
    )

    result = stages.create_sglang_thinker_executor_from_config(
        "/models/qwen35",
        total_gpu_memory_fraction=0.75,
        encoder_mem_reserve=0.05,
    )

    assert result == "scheduler"
    assert seen["model_path"] == "/models/qwen35"
    assert seen["contract_kwargs"]["stage_name"] == "thinker"
    assert seen["contract_kwargs"]["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert seen["contract_kwargs"]["encoder_mem_reserve"] == pytest.approx(0.05)
    assert seen["overrides"]["mamba_scheduler_strategy"] == "extra_buffer"
    assert seen["overrides"]["mem_fraction_static"] == pytest.approx(0.70)
    assert seen["scheduler_kwargs"]["total_gpu_memory_fraction"] == pytest.approx(0.70)


def test_qwen35_thinker_colocated_explicit_mem_fraction_skips_reserve(
    monkeypatch,
):
    seen = {}

    def fake_apply_colocated_contract(overrides, **kwargs):
        seen["contract_overrides_before"] = dict(overrides)
        seen["contract_kwargs"] = kwargs
        return SimpleNamespace(
            mem_fraction_static_pinned=True,
            effective_total_gpu_memory_fraction=0.75,
            applied_encoder_mem_reserve=0.0,
        )

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        seen["model_path"] = model_path
        seen["context_length"] = context_length
        seen["overrides"] = overrides
        return SimpleNamespace(
            mem_fraction_static=overrides.get("mem_fraction_static"),
        )

    def fake_create_thinker_scheduler(*args, **kwargs):
        seen["scheduler_args"] = args
        seen["scheduler_kwargs"] = kwargs
        return "scheduler"

    monkeypatch.setattr(
        stages,
        "_apply_colocated_ar_memory_contract",
        fake_apply_colocated_contract,
    )
    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)
    monkeypatch.setattr(
        stages,
        "create_thinker_scheduler",
        fake_create_thinker_scheduler,
    )

    result = stages.create_sglang_thinker_executor_from_config(
        "/models/qwen35",
        total_gpu_memory_fraction=0.75,
        encoder_mem_reserve=0.05,
        server_args_overrides={"mem_fraction_static": 0.75},
    )

    assert result == "scheduler"
    assert seen["contract_overrides_before"]["mem_fraction_static"] == pytest.approx(
        0.75,
    )
    assert seen["contract_kwargs"]["stage_name"] == "thinker"
    assert seen["contract_kwargs"]["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert seen["contract_kwargs"]["encoder_mem_reserve"] == pytest.approx(0.0)
    assert seen["scheduler_kwargs"]["total_gpu_memory_fraction"] == pytest.approx(0.75)


def test_qwen35_talker_executor_uses_talker_subdir(monkeypatch, tmp_path):
    root, _, talker = _split_root_with_talker(tmp_path)
    seen = {}

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        seen["model_path"] = model_path
        seen["context_length"] = context_length
        seen["overrides"] = overrides
        return SimpleNamespace(mem_fraction_static=None)

    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        stages,
        "_apply_colocated_ar_memory_contract",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)

    def fake_create_talker_scheduler(*args, **kwargs):
        seen["scheduler_args"] = args
        seen["scheduler_kwargs"] = kwargs
        return "talker-scheduler"

    module = ModuleType("sglang_omni.models.qwen3_5_omni.bootstrap")
    module.create_talker_scheduler = fake_create_talker_scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = stages.create_talker_ar_executor_from_config(
        str(root),
        talker_max_seq_len=2048,
    )

    assert result == "talker-scheduler"
    assert seen["model_path"] == str(talker)
    assert seen["context_length"] == 2048
    assert seen["scheduler_kwargs"]["root_model_path"] == str(root)
    assert seen["scheduler_kwargs"]["weight_prefix"] == ""


def test_qwen35_talker_executor_uses_talker_lm_subdir(monkeypatch, tmp_path):
    root, _, talker_lm = _split_root_with_talker_lm(tmp_path)
    seen = {}

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        seen["model_path"] = model_path
        seen["context_length"] = context_length
        seen["overrides"] = overrides
        return SimpleNamespace(mem_fraction_static=None)

    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        stages,
        "_apply_colocated_ar_memory_contract",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(stages, "avail_gpu_mem", lambda *_: 0)
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda *_: 0)

    def fake_create_talker_scheduler(*args, **kwargs):
        seen["scheduler_args"] = args
        seen["scheduler_kwargs"] = kwargs
        return "talker-scheduler"

    module = ModuleType("sglang_omni.models.qwen3_5_omni.bootstrap")
    module.create_talker_scheduler = fake_create_talker_scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = stages.create_talker_ar_executor_from_config(str(root))

    assert result == "talker-scheduler"
    assert seen["model_path"] == str(talker_lm)
    assert seen["scheduler_kwargs"]["root_model_path"] == str(root)
    assert seen["scheduler_kwargs"]["weight_prefix"] == ""
