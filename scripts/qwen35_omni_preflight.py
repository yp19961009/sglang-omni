# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3.5-Omni checkpoint preflight checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # 中文说明：支持从源码树直接执行
    # `python scripts/qwen35_omni_preflight.py`，不要求先 editable install。
    sys.path.insert(0, str(_REPO_ROOT))

from sglang_omni.models.qwen3_5_omni.preflight import (
    format_preflight_report,
    format_vllm_profile_report,
    load_vllm_profile_payload,
    run_qwen35_preflight,
    run_vllm_profile_preflight,
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
    parser.add_argument(
        "--vllm-profile",
        default=None,
        help=(
            "Optional vLLM perf_v2 profile file. The file may contain an "
            "engine_args object or be the engine_args object itself."
        ),
    )
    parser.add_argument(
        "--vllm-engine-args-json",
        default=None,
        help="Inline JSON object for vLLM engine_args compatibility preflight.",
    )
    parser.add_argument(
        "--disable-mtp",
        action="store_true",
        help=(
            "When checking a vLLM profile, ignore Qwen3.5 thinker MTP "
            "speculative_config and validate the base thinker AR path."
        ),
    )
    args = parser.parse_args()
    if args.model_path is None and not (
        args.vllm_profile or args.vllm_engine_args_json
    ):
        parser.error("--model-path or --vllm-profile is required")
    if args.vllm_profile and args.vllm_engine_args_json:
        parser.error("--vllm-profile and --vllm-engine-args-json are mutually exclusive")
    return args


def _payload_with_disable_mtp(payload):
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("engine_args"), dict):
        normalized = dict(payload)
        engine_args = dict(payload["engine_args"])
        engine_args["disable_mtp"] = True
        normalized["engine_args"] = engine_args
        return normalized
    for key, value in payload.items():
        if not isinstance(value, dict) or not isinstance(value.get("engine_args"), dict):
            continue
        normalized = dict(payload)
        nested = dict(value)
        engine_args = dict(value["engine_args"])
        engine_args["disable_mtp"] = True
        nested["engine_args"] = engine_args
        normalized[key] = nested
        return normalized
    # 中文说明：有些 perf_v2 文件直接就是 engine_args 对象；命令行加
    # --disable-mtp 时在检查前注入这个 no-op 兼容开关，和 launcher/serve 对齐。
    normalized = dict(payload)
    normalized["disable_mtp"] = True
    return normalized


def main() -> None:
    args = parse_args()
    ok = True
    outputs: list[str] = []
    if args.model_path is not None:
        report = run_qwen35_preflight(
            args.model_path,
            speech=not args.text_only,
            code2wav_model_path=args.code2wav_model_path,
            xvector_info_paths=tuple(args.xvector_info_paths or ()),
            validate_xvector_pickle=args.validate_xvector_pickle,
        )
        outputs.append(format_preflight_report(report))
        ok = ok and report.ok

    if args.vllm_profile or args.vllm_engine_args_json:
        if args.vllm_profile:
            engine_args = load_vllm_profile_payload(args.vllm_profile)
            source = args.vllm_profile
        else:
            engine_args = json.loads(args.vllm_engine_args_json)
            source = "--vllm-engine-args-json"
        if args.disable_mtp:
            engine_args = _payload_with_disable_mtp(engine_args)
        profile_report = run_vllm_profile_preflight(engine_args, source=source)
        outputs.append(format_vllm_profile_report(profile_report))
        ok = ok and profile_report.ok

    print("\n\n".join(outputs))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
