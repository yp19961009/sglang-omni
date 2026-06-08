# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

from sglang_omni.model_runner import model_worker
from sglang_omni.models.qwen3_5_omni import hf_config as qwen35_hf_config
from sglang_omni.utils import hf as hf_utils


def test_model_worker_maps_qwen35_thinker_to_text_config():
    assert model_worker._ARCH_CONFIG_MAP[
        "Qwen3OmniNextThinkerForConditionalGeneration"
    ] == ("thinker_config", "text_config")
    assert model_worker._ARCH_CONFIG_MAP["Qwen3OmniNextThinkerMTP"] == (
        "thinker_config",
        "text_config",
    )
    assert model_worker._ARCH_CONFIG_MAP[
        "Qwen3OmniNextForConditionalGeneration"
    ] == ("thinker_config", "text_config")
    assert "Qwen3OmniNextThinkerMTP" in model_worker._QWEN_OMNI_ARCHES
    assert "Qwen3OmniNextThinkerMTP" in model_worker._QWEN35_OMNI_ARCHES


def test_model_worker_maps_qwen35_talker_to_text_config():
    assert model_worker._ARCH_CONFIG_MAP["Qwen3OmniNextTalkerModel"] == (
        "talker_config",
        "text_config",
    )
    assert model_worker._ARCH_CONFIG_MAP[
        "Qwen3OmniNextMoeTalkerForConditionalGeneration"
    ] == ("talker_config", "text_config")
    assert (
        "Qwen3OmniNextTalkerModel" in model_worker._QWEN_OMNI_TALKER_ARCHES
    )
    assert (
        "Qwen3OmniNextMoeTalkerForConditionalGeneration"
        in model_worker._QWEN_OMNI_TALKER_ARCHES
    )


def test_qwen35_hf_config_registers_auto_config():
    from transformers import AutoConfig

    qwen35_hf_config.register_qwen35_hf_config()

    cfg = AutoConfig.for_model(
        "qwen3_omni_next",
        thinker_config={
            "text_config": {
                "hidden_size": 64,
                "num_attention_heads": 4,
            }
        },
        talker_config={"text_config": {"vocab_size": 16}},
    )

    assert isinstance(cfg, qwen35_hf_config.Qwen3OmniNextConfig)
    assert cfg.thinker_config.text_config.hidden_size == 64
    assert cfg.get_text_config().num_attention_heads == 4

    talker_cfg = AutoConfig.for_model(
        "qwen3_omni_next_talker",
        text_config={"vocab_size": 32},
    )

    assert isinstance(talker_cfg, qwen35_hf_config.Qwen3OmniNextTalkerConfig)
    assert talker_cfg.get_text_config().vocab_size == 32

    code_predictor_cfg = AutoConfig.for_model(
        "qwen3_omni_next_talker_code_predictor",
        num_hidden_layers=2,
    )

    assert isinstance(
        code_predictor_cfg,
        qwen35_hf_config.Qwen3OmniNextTalkerCodePredictorConfig,
    )
    assert code_predictor_cfg.layer_types == [
        "linear_attention",
        "linear_attention",
    ]
    assert code_predictor_cfg.layers_block_type == [
        "linear_attention",
        "linear_attention",
    ]


def test_qwen35_hf_config_loads_root_from_pretrained(tmp_path):
    from transformers import AutoConfig

    qwen35_hf_config.register_qwen35_hf_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_next",
                "thinker_config": {
                    "text_config": {
                        "hidden_size": 128,
                        "num_hidden_layers": 4,
                        "num_attention_heads": 8,
                        "layer_types": [
                            "linear_attention",
                            "full_attention",
                            "attention",
                            "linear_attention",
                        ],
                        "quantization_config": {
                            "quant_method": "fp8",
                            "weight_block_size": [128, 128],
                        },
                        "rope_scaling": {"type": "yarn", "factor": 4.0},
                        "rope_parameters": {
                            "rope_type": "default",
                            "rope_theta": 1000000.0,
                        },
                    },
                    "audio_config": {"downsample_times": 4},
                    "talker_language_id": {"zh": 77},
                    "talker_assistant_prompt_id_mapping": {
                        "happy": [90, 91],
                    },
                },
                "talker_config": {
                    "text_config": {
                        "vocab_size": 256,
                        "compression_config": {"format": "nvfp4"},
                    },
                    "code_predictor_config": {
                        "rope_scaling": {"rope_type": "default"}
                    },
                    "speaker_id": {"f6009": 2},
                    "speaker_system_prompt_id": {"f6009": [70, 71]},
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)

    assert isinstance(cfg, qwen35_hf_config.Qwen3OmniNextConfig)
    assert cfg.thinker_config.text_config.hidden_size == 128
    assert cfg.talker_config.text_config.vocab_size == 256
    assert cfg.get_text_config().num_attention_heads == 8
    assert cfg.thinker_config.text_config.quantization_config == {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    assert cfg.thinker_config.text_config.rope_scaling == {
        "type": "yarn",
        "factor": 4.0,
    }
    assert cfg.thinker_config.text_config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 1000000.0,
    }
    assert cfg.thinker_config.text_config.rope_theta == 1000000.0
    assert cfg.thinker_config.text_config.partial_rotary_factor == 0.25
    assert hasattr(cfg.thinker_config.text_config, "torch_dtype")
    assert cfg.thinker_config.text_config.layers_block_type == [
        "linear_attention",
        "attention",
        "attention",
        "linear_attention",
    ]
    assert cfg.thinker_config.text_config.layer_types == [
        "linear_attention",
        "full_attention",
        "full_attention",
        "linear_attention",
    ]
    assert isinstance(cfg.thinker_config.text_config.quantization_config, dict)
    assert isinstance(cfg.thinker_config.text_config.rope_scaling, dict)
    assert isinstance(cfg.thinker_config.text_config.rope_parameters, dict)
    assert cfg.talker_config.text_config.compression_config == {"format": "nvfp4"}
    assert cfg.talker_config.code_predictor_config.rope_scaling == {
        "rope_type": "default"
    }
    assert isinstance(cfg.talker_config.text_config.compression_config, dict)
    assert isinstance(cfg.talker_config.code_predictor_config.rope_scaling, dict)
    assert cfg.thinker_config.talker_language_id == {"zh": 77}
    assert cfg.thinker_config.talker_assistant_prompt_id_mapping == {
        "happy": [90, 91],
    }
    assert cfg.talker_config.speaker_id == {"f6009": 2}
    assert cfg.talker_config.speaker_system_prompt_id == {"f6009": [70, 71]}


def test_qwen35_hf_config_normalizes_vllm_qwen35_text_model_types(tmp_path):
    from transformers import AutoConfig

    qwen35_hf_config.register_qwen35_hf_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_next",
                "thinker_config": {
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 128,
                        "num_hidden_layers": 4,
                        "rope_parameters": {
                            "rope_type": "default",
                            "rope_theta": 1000000.0,
                        },
                    },
                },
                "talker_config": {
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "vocab_size": 256,
                        "num_hidden_layers": 4,
                        "full_attention_interval": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)

    thinker_text = cfg.thinker_config.text_config
    talker_text = cfg.talker_config.text_config
    assert thinker_text.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert thinker_text.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert thinker_text.rope_scaling == {
        "rope_type": "default",
        "rope_theta": 1000000.0,
    }
    assert thinker_text.rope_theta == 1000000.0
    assert thinker_text.partial_rotary_factor == 0.25
    assert hasattr(thinker_text, "torch_dtype")
    assert talker_text.layers_block_type == [
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
    ]
    assert talker_text.layer_types == [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ]
    assert talker_text.partial_rotary_factor == 0.25


def test_qwen35_hf_config_loads_split_configs_from_pretrained(tmp_path):
    from transformers import AutoConfig

    qwen35_hf_config.register_qwen35_hf_config()
    thinker_dir = tmp_path / "thinker"
    talker_dir = tmp_path / "talker_lm"
    thinker_dir.mkdir()
    talker_dir.mkdir()
    (thinker_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_next_thinker",
                "text_config": {
                    "hidden_size": 64,
                    "num_hidden_layers": 4,
                    "linear_num_key_heads": 16,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 500000.0,
                    },
                },
                "vision_config": {"spatial_merge_size": 2},
            }
        ),
        encoding="utf-8",
    )
    (talker_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_next_talker",
                "text_config": {"vocab_size": 512},
                "speaker_id": {"f6009": 2},
                "speaker_system_prompt_id": {"f6009": [70, 71]},
            }
        ),
        encoding="utf-8",
    )

    thinker_cfg = AutoConfig.from_pretrained(
        str(thinker_dir),
        local_files_only=True,
    )
    talker_cfg = AutoConfig.from_pretrained(str(talker_dir), local_files_only=True)

    assert isinstance(thinker_cfg, qwen35_hf_config.Qwen3OmniNextThinkerConfig)
    assert thinker_cfg.get_text_config().hidden_size == 64
    assert thinker_cfg.get_text_config().layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert thinker_cfg.get_text_config().rope_scaling == {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    assert thinker_cfg.get_text_config().rope_theta == 500000.0
    assert thinker_cfg.get_text_config().partial_rotary_factor == 0.25
    assert thinker_cfg.vision_config.spatial_merge_size == 2
    assert isinstance(talker_cfg, qwen35_hf_config.Qwen3OmniNextTalkerConfig)
    assert talker_cfg.get_text_config().vocab_size == 512
    assert talker_cfg.speaker_id == {"f6009": 2}
    assert talker_cfg.speaker_system_prompt_id == {"f6009": [70, 71]}


def test_qwen35_hf_config_loads_thinker_mtp_from_pretrained(tmp_path):
    from transformers import AutoConfig

    qwen35_hf_config.register_qwen35_hf_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_next_thinker_mtp",
                "text_config": {
                    "hidden_size": 96,
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)

    assert isinstance(cfg, qwen35_hf_config.Qwen3OmniNextThinkerMTPConfig)
    assert cfg.architectures == ["Qwen3OmniNextThinkerMTP"]
    assert cfg.get_text_config().hidden_size == 96
    assert cfg.get_text_config().mtp_num_hidden_layers == 1


def test_qwen35_raw_config_resolves_split_talker_arch(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_omni_next_talker"}),
        encoding="utf-8",
    )

    assert (
        hf_utils.try_resolve_arch_from_raw_config(str(tmp_path))
        == "Qwen3OmniNextTalkerModel"
    )


def test_qwen35_hf_config_does_not_override_existing_config():
    from transformers import AutoConfig, PretrainedConfig

    model_type = "qwen3_omni_next_existing_config_test"

    class ExistingConfig(PretrainedConfig):
        model_type = "qwen3_omni_next_existing_config_test"

    class ReplacementConfig(PretrainedConfig):
        model_type = "qwen3_omni_next_existing_config_test"

    AutoConfig.register(model_type, ExistingConfig, exist_ok=True)

    qwen35_hf_config._register_config_if_missing(model_type, ReplacementConfig)

    cfg = AutoConfig.for_model(model_type)
    assert isinstance(cfg, ExistingConfig)


def test_model_worker_registers_qwen35_config_before_model_config(monkeypatch):
    from sglang.srt.configs import model_config as sglang_model_config

    order = []

    def fake_register():
        order.append("register")

    def fake_from_server_args(**kwargs):
        del kwargs
        assert order == ["register"]
        order.append("model_config")
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[],
                thinker_config=SimpleNamespace(text_config=_text_config()),
            )
        )

    monkeypatch.setattr(
        qwen35_hf_config,
        "register_qwen35_hf_config",
        fake_register,
    )
    monkeypatch.setattr(
        sglang_model_config.ModelConfig,
        "from_server_args",
        staticmethod(fake_from_server_args),
    )
    worker = model_worker.ModelWorker.__new__(model_worker.ModelWorker)
    worker.server_args = SimpleNamespace(model_path="/models/qwen35", revision=None)
    worker.model_arch_override = "Qwen3OmniNextThinkerForConditionalGeneration"

    worker._init_model_config()

    assert order == ["register", "model_config"]
    assert worker.model_config.hidden_size == 4096


def test_model_worker_registers_qwen35_mtp_config_before_model_config(monkeypatch):
    from sglang.srt.configs import model_config as sglang_model_config

    order = []

    def fake_register():
        order.append("register")

    def fake_from_server_args(**kwargs):
        del kwargs
        assert order == ["register"]
        order.append("model_config")
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[],
                text_config=_text_config(),
            )
        )

    monkeypatch.setattr(
        qwen35_hf_config,
        "register_qwen35_hf_config",
        fake_register,
    )
    monkeypatch.setattr(
        sglang_model_config.ModelConfig,
        "from_server_args",
        staticmethod(fake_from_server_args),
    )
    worker = model_worker.ModelWorker.__new__(model_worker.ModelWorker)
    worker.server_args = SimpleNamespace(model_path="/models/qwen35", revision=None)
    worker.model_arch_override = "Qwen3OmniNextThinkerMTP"

    worker._init_model_config()

    assert order == ["register", "model_config"]
    assert worker.model_config.hidden_size == 4096


def _text_config():
    return SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=32000,
    )


def test_model_worker_applies_qwen35_direct_talker_config():
    text_config = _text_config()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[],
            text_config=text_config,
        )
    )

    model_worker.ModelWorker._apply_arch_override(
        config,
        "Qwen3OmniNextTalkerModel",
    )

    assert config.hf_text_config is text_config
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 2
    assert config.num_hidden_layers == 32
    assert config.vocab_size == 32000
    assert config.num_attention_layers == 32
    assert config.head_dim == 512
    assert config.v_head_dim == 512


def test_model_worker_applies_qwen35_moe_talker_alias_config():
    text_config = _text_config()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[],
            text_config=text_config,
        )
    )

    model_worker.ModelWorker._apply_arch_override(
        config,
        "Qwen3OmniNextMoeTalkerForConditionalGeneration",
    )

    assert config.hf_text_config is text_config
    assert config.hidden_size == 4096
    assert config.vocab_size == 32000
    assert config.head_dim == 512


def test_model_worker_applies_qwen35_direct_thinker_config():
    text_config = _text_config()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[],
            text_config=text_config,
        )
    )

    model_worker.ModelWorker._apply_arch_override(
        config,
        "Qwen3OmniNextThinkerForConditionalGeneration",
    )

    assert config.hf_text_config is text_config
    assert config.vocab_size == 32000
    assert config.num_attention_layers == 32


def test_model_worker_applies_qwen35_thinker_mtp_alias_config():
    text_config = _text_config()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[],
            text_config=text_config,
        )
    )

    model_worker.ModelWorker._apply_arch_override(
        config,
        "Qwen3OmniNextThinkerMTP",
    )

    assert config.hf_text_config is text_config
    assert config.hidden_size == 4096
    assert config.vocab_size == 32000
    assert config.num_attention_layers == 32
