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
