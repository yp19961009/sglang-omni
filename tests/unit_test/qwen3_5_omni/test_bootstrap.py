# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.models.qwen3_5_omni import bootstrap


def test_normalize_hidden_layers():
    assert bootstrap._normalize_hidden_layers(24) == [24]
    assert bootstrap._normalize_hidden_layers("0,24") == [0, 24]
    assert bootstrap._normalize_hidden_layers([0, "24"]) == [0, 24]
    assert bootstrap._normalize_hidden_layers(None) is None


def test_ensure_embed_capture_layer():
    assert bootstrap._ensure_embed_capture_layer([18]) == [0, 18]
    assert bootstrap._ensure_embed_capture_layer("0,18") == [0, 18]
    assert bootstrap._ensure_embed_capture_layer([0, "18", 18]) == [0, 18]


def test_resolve_capture_hidden_layers_falls_back_without_model_path():
    assert bootstrap._resolve_capture_hidden_layers_from_config(
        SimpleNamespace()
    ) == bootstrap.QWEN3_5_OMNI_DEFAULT_CAPTURE_HIDDEN_LAYERS


def test_resolve_capture_hidden_layers_from_talker_config(monkeypatch):
    from transformers import AutoConfig

    def _from_pretrained(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            talker_config=SimpleNamespace(accept_hidden_layer="2, 18")
        )

    monkeypatch.setattr(AutoConfig, "from_pretrained", _from_pretrained)

    assert bootstrap._resolve_capture_hidden_layers_from_config(
        SimpleNamespace(model_path="/tmp/model")
    ) == [0, 2, 18]


def test_resolve_capture_hidden_layers_prefers_root_metadata(monkeypatch):
    seen = []

    def _load_metadata_config(model_path):
        seen.append(model_path)
        if model_path == "/models/qwen35":
            return SimpleNamespace(
                talker_config=SimpleNamespace(accept_hidden_layer=18)
            )
        return SimpleNamespace()

    monkeypatch.setattr(bootstrap, "_load_metadata_config", _load_metadata_config)

    layers = bootstrap._resolve_capture_hidden_layers_from_config(
        SimpleNamespace(model_path="/models/qwen35/thinker"),
        root_model_path="/models/qwen35",
    )

    assert layers == [0, 18]
    assert seen == ["/models/qwen35"]


def test_metadata_config_loads_root_model_path(monkeypatch):
    root_config = SimpleNamespace(source="root")

    monkeypatch.setattr(
        bootstrap,
        "_load_metadata_config",
        lambda model_path: root_config,
    )

    assert (
        bootstrap._metadata_config(
            SimpleNamespace(model_path="/models/qwen35/talker", hf_config="stage"),
            "/models/qwen35",
        )
        is root_config
    )


def test_metadata_config_falls_back_to_stage_config(monkeypatch):
    stage_config = SimpleNamespace(source="stage")

    def _raise(model_path):
        raise OSError(model_path)

    monkeypatch.setattr(bootstrap, "_load_metadata_config", _raise)

    assert (
        bootstrap._metadata_config(
            SimpleNamespace(
                model_path="/models/qwen35/talker",
                hf_config=stage_config,
            ),
            "/models/qwen35",
        )
        is stage_config
    )


def test_resolve_direct_qwen35_configs():
    direct_config = SimpleNamespace(text_config=SimpleNamespace())
    split_thinker_config = SimpleNamespace(
        thinker_config=None,
        text_config=SimpleNamespace(),
    )
    split_talker_config = SimpleNamespace(
        talker_config=None,
        text_config=SimpleNamespace(),
    )
    root_config = SimpleNamespace(
        thinker_config=SimpleNamespace(name="thinker"),
        talker_config=SimpleNamespace(name="talker"),
    )

    assert bootstrap._resolve_thinker_config(direct_config) is direct_config
    assert bootstrap._resolve_talker_config(direct_config) is direct_config
    assert bootstrap._resolve_thinker_config(split_thinker_config) is split_thinker_config
    assert bootstrap._resolve_talker_config(split_talker_config) is split_talker_config
    assert bootstrap._resolve_thinker_config(root_config).name == "thinker"
    assert bootstrap._resolve_talker_config(root_config).name == "talker"


def test_optional_config_dict_allows_missing_speaker_id():
    speaker_map = {"f6009": 1}

    assert bootstrap._optional_config_dict(SimpleNamespace(), "speaker_id") == {}
    assert (
        bootstrap._optional_config_dict(
            SimpleNamespace(speaker_id="not-a-dict"),
            "speaker_id",
        )
        == {}
    )
    assert (
        bootstrap._optional_config_dict(
            SimpleNamespace(speaker_id=speaker_map),
            "speaker_id",
            )
        is speaker_map
    )


def test_maybe_enable_subtalker_torch_compile_consumes_talker_flag():
    calls = []
    model = SimpleNamespace(
        enable_subtalker_torch_compile=lambda: calls.append("model_hook")
    )
    server_args = SimpleNamespace(enable_torch_compile=True)

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == ["model_hook"]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_warms_predictor(monkeypatch):
    calls = []
    model = SimpleNamespace(
        enable_subtalker_torch_compile=lambda: calls.append(("compile", None)),
        warmup_subtalker_code_predictor=lambda *, batch_sizes: calls.append(
            ("warmup", list(batch_sizes))
        )
        or list(batch_sizes),
    )
    server_args = SimpleNamespace(enable_torch_compile=True, max_running_requests=8)

    monkeypatch.setenv("SGLANG_OMNI_QWEN35_SUBTALKER_COMPILE_WARMUP", "1")
    monkeypatch.delenv(
        "SGLANG_OMNI_QWEN35_SUBTALKER_WARMUP_BATCHES",
        raising=False,
    )

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == [("compile", None), ("warmup", [1, 2, 3, 4, 5, 6, 7, 8])]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_avoids_reduce_overhead(monkeypatch):
    calls = []

    def compile_hook(*, mode=None):
        calls.append(("compile", mode))

    model = SimpleNamespace(
        enable_subtalker_torch_compile=compile_hook,
        warmup_subtalker_code_predictor=lambda *, batch_sizes: calls.append(
            ("warmup", list(batch_sizes))
        )
        or list(batch_sizes),
    )
    server_args = SimpleNamespace(enable_torch_compile=True, max_running_requests=2)

    monkeypatch.setenv("SGLANG_TORCH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("SGLANG_OMNI_QWEN35_SUBTALKER_COMPILE_WARMUP", "1")
    monkeypatch.delenv(
        "SGLANG_OMNI_QWEN35_SUBTALKER_WARMUP_BATCHES",
        raising=False,
    )

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == [("compile", "default"), ("warmup", [1, 2])]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_fallbacks_to_default_mode(monkeypatch):
    calls = []

    def compile_hook(*, mode=None):
        calls.append(("compile", mode))

    def warmup_hook(*, batch_sizes):
        calls.append(("warmup", list(batch_sizes)))
        if len([call for call in calls if call[0] == "warmup"]) == 1:
            raise RuntimeError("compile mode failed")
        return list(batch_sizes)

    model = SimpleNamespace(
        enable_subtalker_torch_compile=compile_hook,
        warmup_subtalker_code_predictor=warmup_hook,
    )
    server_args = SimpleNamespace(enable_torch_compile=True, max_running_requests=2)

    monkeypatch.setenv("SGLANG_TORCH_COMPILE_MODE", "max-autotune")
    monkeypatch.setenv("SGLANG_OMNI_QWEN35_SUBTALKER_COMPILE_WARMUP", "1")
    monkeypatch.delenv(
        "SGLANG_OMNI_QWEN35_SUBTALKER_WARMUP_BATCHES",
        raising=False,
    )

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == [
        ("compile", None),
        ("warmup", [1, 2]),
        ("compile", "default"),
        ("warmup", [1, 2]),
    ]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_warmup_defaults_on(monkeypatch):
    calls = []
    model = SimpleNamespace(
        enable_subtalker_torch_compile=lambda: calls.append("compile"),
        warmup_subtalker_code_predictor=lambda *, batch_sizes: calls.append(
            ("warmup", list(batch_sizes))
        )
        or list(batch_sizes),
    )
    server_args = SimpleNamespace(enable_torch_compile=True, max_running_requests=3)

    monkeypatch.delenv(
        "SGLANG_OMNI_QWEN35_SUBTALKER_COMPILE_WARMUP",
        raising=False,
    )

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == ["compile", ("warmup", [1, 2, 3])]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_warmup_can_be_disabled(monkeypatch):
    calls = []
    model = SimpleNamespace(
        enable_subtalker_torch_compile=lambda: calls.append("compile"),
        warmup_subtalker_code_predictor=lambda *, batch_sizes: calls.append("warmup"),
    )
    server_args = SimpleNamespace(enable_torch_compile=True, max_running_requests=8)

    monkeypatch.setenv("SGLANG_OMNI_QWEN35_SUBTALKER_COMPILE_WARMUP", "0")

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == ["compile"]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_falls_back_to_code_predictor():
    calls = []
    model = SimpleNamespace(
        code_predictor=SimpleNamespace(
            enable_torch_compile=lambda: calls.append("predictor_hook")
        )
    )
    server_args = SimpleNamespace(enable_torch_compile=True)

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is True

    assert calls == ["predictor_hook"]
    assert server_args.enable_torch_compile is False


def test_maybe_enable_subtalker_torch_compile_ignores_disabled_flag():
    model = SimpleNamespace(
        enable_subtalker_torch_compile=lambda: (_ for _ in ()).throw(
            AssertionError("compile hook should not be called")
        )
    )
    server_args = SimpleNamespace(enable_torch_compile=False)

    assert bootstrap._maybe_enable_subtalker_torch_compile(model, server_args) is False
    assert server_args.enable_torch_compile is False
