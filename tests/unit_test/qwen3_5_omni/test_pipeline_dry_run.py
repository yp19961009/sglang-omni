# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from collections.abc import Iterable

from sglang_omni.config import PipelineConfig
from sglang_omni.models.qwen3_5_omni.config import (
    Qwen35OmniPipelineConfig,
    Qwen35OmniSpeechColocatedPipelineConfig,
    Qwen35OmniSpeechPipelineConfig,
)


def _resolve(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _iter_config_paths(config: PipelineConfig) -> Iterable[tuple[str, str]]:
    if config.placement_policy:
        yield "placement_policy", config.placement_policy
    if config.terminal_stages_fn:
        yield "terminal_stages_fn", config.terminal_stages_fn

    for stage in config.stages:
        yield f"{stage.name}.factory", stage.factory
        for field in ("route_fn", "wait_for_fn", "merge_fn", "stream_done_to_fn"):
            path = getattr(stage, field)
            if path:
                yield f"{stage.name}.{field}", path
        for target, path in stage.project_payload.items():
            yield f"{stage.name}.project_payload.{target}", path


def test_qwen35_pipeline_runtime_paths_are_importable():
    """Dry-run dotted paths before the launcher imports them at runtime."""
    configs = [
        Qwen35OmniPipelineConfig(model_path="dummy"),
        Qwen35OmniSpeechPipelineConfig(model_path="dummy"),
        Qwen35OmniSpeechColocatedPipelineConfig(model_path="dummy"),
    ]

    checked_paths: set[str] = set()
    for config in configs:
        for field, path in _iter_config_paths(config):
            target = _resolve(path)
            checked_paths.add(path)
            assert callable(target), f"{config.config_cls}.{field} is not callable"

    assert (
        "sglang_omni.models.qwen3_5_omni.stages."
        "create_talker_ar_executor_from_config"
        in checked_paths
    )
    assert (
        "sglang_omni.models.qwen3_5_omni.request_builders."
        "resolve_terminal_stages"
        in checked_paths
    )
