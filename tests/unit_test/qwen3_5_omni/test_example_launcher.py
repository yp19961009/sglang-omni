# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from examples import run_qwen3_5_omni_speech_server as launcher


def test_qwen35_speech_example_delegates_to_native_cli():
    argv = launcher.build_cli_argv(
        [
            "--model-path",
            "/models/qwen35",
            "--thinker-gpus",
            "0",
            "--talker-gpu",
            "1",
        ]
    )

    assert argv == [
        "serve",
        "--model-path",
        "/models/qwen35",
        "--thinker-gpus",
        "0",
        "--talker-gpu",
        "1",
    ]
