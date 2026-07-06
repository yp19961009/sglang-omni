# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3.5-Omni components."""

from __future__ import annotations

from typing import Any

# Import registers the qwen3_omni_next AutoConfig shims before load_hf_config.
import sglang_omni.models.qwen35_omni.hf_config  # noqa: F401
from sglang_omni.utils import load_hf_config


def load_thinker_config(model_path: str) -> Any:
    cfg = load_hf_config(model_path, trust_remote_code=True, local_files_only=True)
    return cfg.thinker_config
