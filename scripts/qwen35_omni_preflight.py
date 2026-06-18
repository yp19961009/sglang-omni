# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3.5-Omni checkpoint preflight checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Allow direct execution from a source tree with
    # `python scripts/qwen35_omni_preflight.py`, without requiring an editable
    # install first.
    sys.path.insert(0, str(_REPO_ROOT))

from sglang_omni.models.qwen3_5_omni.preflight import (
    format_preflight_report,
    run_qwen35_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--code2wav-model-path", default=None)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip talker/code2wav speech checks.",
    )
    parser.add_argument(
        "--xvector-info-path",
        "--voice-clone-info-path",
        "--voice-clone-path",
        dest="xvector_info_paths",
        action="append",
        default=[],
        help=(
            "Optional voice-clone/xvector_info directory to preflight. "
            "May be passed more than once."
        ),
    )
    parser.add_argument(
        "--validate-xvector-pickle",
        action="store_true",
        help=(
            "Also pickle.load feat.pkl and check for supported prompt-code "
            "keys. Only use this for trusted local voice-clone assets."
        ),
    )
    args = parser.parse_args()
    if args.model_path is None:
        parser.error("--model-path is required")
    return args


def main() -> None:
    args = parse_args()
    report = run_qwen35_preflight(
        args.model_path,
        speech=not args.text_only,
        code2wav_model_path=args.code2wav_model_path,
        xvector_info_paths=tuple(args.xvector_info_paths or ()),
        validate_xvector_pickle=args.validate_xvector_pickle,
    )
    print(format_preflight_report(report))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
