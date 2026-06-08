# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3.5-Omni components."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sglang_omni.utils import load_hf_config

_QWEN3_NEXT_HINT_FIELDS = (
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "decoder_sparse_step",
    "mlp_only_layers",
    "layer_types",
    "layers_block_type",
    "full_attention_interval",
    "rope_parameters",
    "rope_scaling",
    "partial_rotary_factor",
)
_QWEN3_NEXT_TEXT_MODEL_TYPES = {
    "qwen3_next",
    "qwen3_next_vl_text",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_omni_next_thinker",
    "qwen3_omni_next_talker_code_predictor",
    "qwen3_omni_next_code_predictor",
    "qwen3_5_talker_code_predictor",
    "qwen3_5_code_predictor",
}
_RAW_CONFIG_PRESERVE_DICT_KEYS = frozenset(
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


def _to_namespace(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        if key in _RAW_CONFIG_PRESERVE_DICT_KEYS:
            # 中文说明：这些字段是运行期查表/量化/rope 配置，后续代码会按
            # 普通 dict 读取；raw config fallback 不能把它们转成 namespace。
            return {
                item_key: _to_namespace(item, key=str(item_key))
                for item_key, item in value.items()
            }
        namespace = SimpleNamespace()
        for key, item in value.items():
            if isinstance(key, str):
                setattr(namespace, key, _to_namespace(item, key=key))
        return namespace
    if isinstance(value, list):
        return [_to_namespace(item, key=key) for item in value]
    return value


def _normalize_qwen35_config_tree(
    config: Any,
    *,
    _seen: set[int] | None = None,
) -> Any:
    """Apply SGLang runtime aliases after raw config.json fallback loading."""

    if config is None or isinstance(config, (str, int, float, bool, dict)):
        return config
    seen = _seen if _seen is not None else set()
    obj_id = id(config)
    if obj_id in seen:
        return config
    seen.add(obj_id)

    # 中文说明：raw fallback 绕过了 HF shim 的 PretrainedConfig 归一化。
    # 这里补齐 thinker/talker/code predictor 需要的 SGLang 字段，避免旧
    # transformers 环境下真实权重启动时才遇到 layers_block_type/rope 缺失。
    ensure_sglang_qwen3_next_text_config(config)
    for attr in (
        "text_config",
        "thinker_config",
        "talker_config",
        "code_predictor_config",
    ):
        _normalize_qwen35_config_tree(getattr(config, attr, None), _seen=seen)
    return config


def _load_raw_config(model_path: str) -> Any:
    config_path = Path(model_path) / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return _normalize_qwen35_config_tree(_to_namespace(data))


def load_qwen35_config(model_path: str) -> Any:
    try:
        return load_hf_config(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as hf_exc:
        try:
            # 中文说明：当前基础环境可能还没有 transformers
            # qwen3_omni_next config。真实模型通常至少带本地 config.json，
            # 这里转成 attr-style namespace，让 encoder/preflight 先能读配置。
            return _load_raw_config(model_path)
        except Exception as raw_exc:
            raise OSError(
                "failed to load Qwen3.5 config via AutoConfig or raw config.json: "
                f"AutoConfig={hf_exc}; raw={raw_exc}"
            ) from raw_exc


def sub_config_or_self(config: Any, attr: str) -> Any:
    value = getattr(config, attr, None)
    # 中文说明：本地 HF config shim 会为 split thinker/talker config 保留
    # thinker_config/talker_config 属性，但值可能是 None。此时它本身就是
    # 子配置，不能把 None 继续传给后续 stage/model wrapper。
    return value if value is not None else config


def ensure_sglang_qwen3_next_text_config(config: Any) -> Any:
    """Normalize Qwen3-Next text config fields consumed by SGLang core."""

    if config is None:
        return config

    _ensure_sglang_qwen3_next_runtime_defaults(config)

    layers_block_type = getattr(config, "layers_block_type", None)
    if layers_block_type is not None:
        sglang_layer_types = [
            _to_sglang_layer_block_type(item) for item in layers_block_type
        ]
        setattr(config, "layers_block_type", sglang_layer_types)
        if getattr(config, "layer_types", None) is None:
            # 中文说明：SGLang core 读 layers_block_type，HF/vLLM 新
            # subtalker 代码会读 layer_types；raw 配置若只带前者，这里
            # 回填 HF 命名，避免 code predictor 初始化时访问缺失字段。
            setattr(
                config,
                "layer_types",
                [_to_hf_layer_type(item) for item in sglang_layer_types],
            )
        return config

    layer_types = getattr(config, "layer_types", None)
    if layer_types is None and _looks_like_qwen3_next_text_config(config):
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if num_hidden_layers is not None:
            # 中文说明：vLLM/transformers Qwen3NextConfig 的默认 layer_types
            # 是前三层 linear_attention、每第四层 full_attention。当前
            # SGLang core 使用字段名 layers_block_type，且把 full_attention
            # 叫作 attention；这里补齐别名，避免真实模型加载到第一层时报
            # AttributeError 或 KeyError。
            interval_pattern = _full_attention_interval(config)
            layer_types = [
                "linear_attention"
                if bool((idx + 1) % interval_pattern)
                else "full_attention"
                for idx in range(int(num_hidden_layers))
            ]

    if layer_types is not None:
        hf_layer_types = [_to_hf_layer_type(item) for item in layer_types]
        setattr(config, "layer_types", hf_layer_types)
        setattr(
            config,
            "layers_block_type",
            [_to_sglang_layer_block_type(item) for item in hf_layer_types],
        )
    return config


def _ensure_sglang_qwen3_next_runtime_defaults(config: Any) -> None:
    if not _looks_like_qwen3_next_text_config(config):
        return

    rope_scaling = _plain_dict_or_none(getattr(config, "rope_scaling", None))
    rope_parameters = _plain_dict_or_none(getattr(config, "rope_parameters", None))
    rope_source: dict[str, Any] = {}
    if rope_parameters:
        rope_source.update(rope_parameters)
    if rope_scaling:
        rope_source.update(rope_scaling)

    if rope_scaling is None and rope_parameters is not None:
        # 中文说明：vLLM/transformers 新 config 用 rope_parameters；当前
        # SGLang core 仍读取 rope_scaling。这里保留原字段，同时补一个
        # SGLang 可消费的别名，避免真实模型用错默认 RoPE。
        setattr(config, "rope_scaling", dict(rope_parameters))

    if getattr(config, "rope_theta", None) is None and rope_source.get(
        "rope_theta"
    ) is not None:
        setattr(config, "rope_theta", rope_source["rope_theta"])

    if getattr(config, "partial_rotary_factor", None) is None:
        setattr(
            config,
            "partial_rotary_factor",
            rope_source.get("partial_rotary_factor", 0.25),
        )

    if not hasattr(config, "torch_dtype"):
        # 中文说明：SGLang Qwen3Next linear attention 会直接访问
        # config.torch_dtype。PretrainedConfig 通常自带该属性，但 raw/split
        # namespace 不一定有；补 None 让底层按默认 dtype 处理。
        setattr(config, "torch_dtype", None)


def _plain_dict_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value).copy()
    return None


def _full_attention_interval(config: Any) -> int:
    raw_interval = getattr(config, "full_attention_interval", 4)
    if raw_interval is None:
        return 4

    try:
        interval = int(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("Qwen3Next full_attention_interval must be an integer") from exc

    if interval <= 0:
        raise ValueError("Qwen3Next full_attention_interval must be positive")
    return interval


def _looks_like_qwen3_next_text_config(config: Any) -> bool:
    model_type = getattr(config, "model_type", None)
    if model_type in _QWEN3_NEXT_TEXT_MODEL_TYPES:
        return True
    return any(
        getattr(config, field, None) is not None
        for field in _QWEN3_NEXT_HINT_FIELDS
    )


def _to_sglang_layer_block_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized == "full_attention":
        return "attention"
    return normalized


def _to_hf_layer_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized == "attention":
        return "full_attention"
    return normalized


def load_qwen35_thinker_config(model_path: str) -> Any:
    cfg = load_qwen35_config(model_path)
    # 中文说明：root checkpoint 使用 cfg.thinker_config；split checkpoint 的
    # thinker 子目录则可能直接就是 thinker config。这里同时兼容两种布局。
    return sub_config_or_self(cfg, "thinker_config")
