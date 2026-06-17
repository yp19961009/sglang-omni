# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from examples import run_qwen3_5_omni_server as launcher


def test_qwen35_text_example_adds_text_only_before_delegating():
    argv = launcher.build_cli_argv(["--model-path", "/models/qwen35"])

    assert argv == ["serve", "--text-only", "--model-path", "/models/qwen35"]


def test_qwen35_text_example_keeps_explicit_text_only_once():
    argv = launcher.build_cli_argv(
        ["--model-path", "/models/qwen35", "--text_only"]
    )

    assert argv == ["serve", "--model-path", "/models/qwen35", "--text_only"]
