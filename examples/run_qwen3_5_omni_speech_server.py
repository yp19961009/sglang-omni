# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the native Qwen3.5-Omni speech server CLI."""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_cli_argv(argv: list[str]) -> list[str]:
    """Build the native Typer argv for this historical example path."""

    return ["serve", *argv]


def main(argv: list[str] | None = None) -> None:
    from sglang_omni.cli import app

    args = list(sys.argv[1:] if argv is None else argv)
    sys.argv = [sys.argv[0], *build_cli_argv(args)]
    app()


if __name__ == "__main__":
    main()
