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
    }
)


def _as_plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_value(item) for item in value]
    return value


def _as_config(value: Any, *, key: str | None = None) -> Any:
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


class Qwen3OmniNextConfig(PretrainedConfig):
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
        super().__init__(**kwargs)
        self.thinker_config = _as_config(thinker_config)
        self.talker_config = _as_config(talker_config)
        self.text_config = _as_config(text_config)
        self.audio_config = _as_config(audio_config)
        self.vision_config = _as_config(vision_config)
        ensure_sglang_qwen3_next_text_config(self)

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
    """Config alias used by vLLM's Qwen3.5 thinker MTP path."""

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
