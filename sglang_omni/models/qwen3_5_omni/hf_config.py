# SPDX-License-Identifier: Apache-2.0
"""Hugging Face config shim for Qwen3.5-Omni.

The current runtime image may not yet ship transformers.models.qwen3_omni_next.
SGLang still asks AutoConfig to parse config.json before the Qwen3.5 model
wrapper is selected, so we register a small local config class that preserves the
fields needed by ModelConfig, encoder stages, and request builders.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang_omni.models.qwen3_5_omni.components.common import (
    ensure_sglang_qwen3_next_text_config,
)

_PRESERVE_DICT_KEYS = frozenset(
    {
        "compression_config",
        "quantization_config",
        "rope_parameters",
        "rope_scaling",
        "speaker_id",
        "speaker_system_prompt_id",
        "talker_language_id",
        "talker_assistant_prompt_id_mapping",
        "id2label",
        "label2id",
    }
)
_QWEN3_NEXT_TEXT_ATTRS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "partial_rotary_factor",
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "decoder_sparse_step",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts_per_tok",
    "num_experts",
    "norm_topk_prob",
    "output_router_logits",
    "router_aux_loss_coef",
    "mlp_only_layers",
    "full_attention_interval",
)


def _as_plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_value(item) for item in value]
    return value


def _as_config(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Qwen3NextConfig):
        return value
    if isinstance(value, PretrainedConfig):
        return ensure_sglang_qwen3_next_text_config(value)
    if isinstance(value, dict):
        if key in _PRESERVE_DICT_KEYS:
            # 中文说明：这些字段是运行期映射表或底层框架约定的普通 dict
            # 配置，不是 HF 子配置；若转成 PretrainedConfig，后续
            # .items()/dict lookup 或 isinstance(..., dict) 判断会失效。
            return _as_plain_value(value)
        cfg = PretrainedConfig(
            **{
                item_key: _as_config(item, key=item_key)
                for item_key, item in value.items()
            }
        )
        return ensure_sglang_qwen3_next_text_config(cfg)
    if isinstance(value, list):
        return [_as_config(item, key=key) for item in value]
    return value


def _nested_text_config(config: Any) -> Any | None:
    return getattr(config, "text_config", None) if config is not None else None


def _copy_qwen3_next_text_attrs(kwargs: dict[str, Any], text_config: Any | None) -> None:
    if text_config is None:
        return
    for attr in _QWEN3_NEXT_TEXT_ATTRS:
        if attr not in kwargs and hasattr(text_config, attr):
            kwargs[attr] = getattr(text_config, attr)


def _to_sglang_layer_block_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized == "full_attention":
        return "attention"
    return normalized


def _explicit_layers_block_type(text_config: Any | None) -> list[str] | None:
    if text_config is None:
        return None
    raw = getattr(text_config, "layers_block_type", None)
    if raw is None:
        raw = getattr(text_config, "layer_types", None)
    if raw is None:
        return None
    return [_to_sglang_layer_block_type(item) for item in raw]


class Qwen3OmniNextConfig(Qwen3NextConfig):
    """Minimal attr-preserving config for Qwen3.5-Omni root checkpoints."""

    model_type = "qwen3_omni_next"

    def __init__(
        self,
        thinker_config: dict[str, Any] | PretrainedConfig | None = None,
        talker_config: dict[str, Any] | PretrainedConfig | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_config: dict[str, Any] | PretrainedConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault(
            "architectures",
            ["Qwen3OmniNextForConditionalGeneration"],
        )
        thinker_cfg = _as_config(thinker_config)
        talker_cfg = _as_config(talker_config)
        text_cfg = _as_config(text_config)
        qwen3_next_text_cfg = (
            _nested_text_config(thinker_cfg)
            or text_cfg
            or _nested_text_config(talker_cfg)
        )
        kwargs.setdefault("full_attention_interval", 4)
        _copy_qwen3_next_text_attrs(
            kwargs,
            qwen3_next_text_cfg,
        )
        super().__init__(**kwargs)
        self._omni_layers_block_type = _explicit_layers_block_type(
            qwen3_next_text_cfg
        )
        self.thinker_config = thinker_cfg
        self.talker_config = talker_cfg
        self.text_config = text_cfg if text_cfg is not None else qwen3_next_text_cfg
        self.audio_config = _as_config(audio_config)
        self.vision_config = _as_config(vision_config)
        self.layer_types = [
            "full_attention" if item == "attention" else item
            for item in self.layers_block_type
        ]

    @property
    def layers_block_type(self) -> list[str]:
        explicit = getattr(self, "_omni_layers_block_type", None)
        if explicit is not None:
            return list(explicit)
        return super().layers_block_type

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        for candidate in (
            getattr(self.thinker_config, "text_config", None),
            getattr(self.talker_config, "text_config", None),
            self.text_config,
            self,
        ):
            if candidate is not None:
                return candidate
        return self


class Qwen3OmniNextThinkerConfig(Qwen3OmniNextConfig):
    """Direct split-checkpoint config for root/thinker."""

    model_type = "qwen3_omni_next_thinker"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "architectures",
            ["Qwen3OmniNextThinkerForConditionalGeneration"],
        )
        super().__init__(**kwargs)


class Qwen3OmniNextThinkerMTPConfig(Qwen3OmniNextConfig):
    """Config alias used by the reference Qwen3.5 thinker MTP path."""

    model_type = "qwen3_omni_next_thinker_mtp"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("architectures", ["Qwen3OmniNextThinkerMTP"])
        super().__init__(**kwargs)


class Qwen3OmniNextTalkerConfig(Qwen3OmniNextConfig):
    """Direct split-checkpoint config for root/talker_lm or root/talker."""

    model_type = "qwen3_omni_next_talker"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("architectures", ["Qwen3OmniNextTalkerModel"])
        super().__init__(**kwargs)


class Qwen3OmniNextTalkerCodePredictorConfig(Qwen3OmniNextConfig):
    """Config alias used by the Qwen3.5-Omni residual code predictor."""

    model_type = "qwen3_omni_next_talker_code_predictor"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "architectures",
            ["Qwen3OmniNextTalkerCodePredictorModelForConditionalGeneration"],
        )
        super().__init__(**kwargs)


def _is_model_type_registered(model_type: str) -> bool:
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        return model_type in CONFIG_MAPPING
    except Exception:
        try:
            AutoConfig.for_model(model_type)
        except Exception:
            return False
        return True


def _register_config_if_missing(
    model_type: str,
    config_cls: type[PretrainedConfig],
) -> None:
    if _is_model_type_registered(model_type):
        return
    AutoConfig.register(model_type, config_cls)


def register_qwen35_hf_config() -> None:
    """Register local config aliases used by Qwen3.5-Omni checkpoints."""

    # 中文说明：如果未来 transformers 已内置官方 qwen3_omni_next config，
    # 不覆盖官方实现；当前老镜像没有这些 model_type 时才注册轻量 shim。
    _register_config_if_missing(
        Qwen3OmniNextConfig.model_type,
        Qwen3OmniNextConfig,
    )
    _register_config_if_missing(
        Qwen3OmniNextThinkerConfig.model_type,
        Qwen3OmniNextThinkerConfig,
    )
    _register_config_if_missing(
        Qwen3OmniNextThinkerMTPConfig.model_type,
        Qwen3OmniNextThinkerMTPConfig,
    )
    _register_config_if_missing(
        Qwen3OmniNextTalkerConfig.model_type,
        Qwen3OmniNextTalkerConfig,
    )
    _register_config_if_missing(
        Qwen3OmniNextTalkerCodePredictorConfig.model_type,
        Qwen3OmniNextTalkerCodePredictorConfig,
    )
