#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ASR-check generated Qwen3.5-Omni RTC audio artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import wave
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_TEXT = (
    "我叫千问，是阿里巴巴集团旗下的通义实验室自主研发的多模态"
    "超大规模语言模型，有什么我可以帮助你的吗？"
)


def _load_first(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"empty per-request artifact: {path}")
        data = data[0]
    if not isinstance(data, dict):
        raise TypeError(f"unsupported artifact shape: {path}")
    return data


def _norm(text: str) -> str:
    return re.sub(
        r'[\s,，。.!！?？、:：;；"“”~·（）()\[\]{}<>《》-]',
        "",
        text,
    ).lower()


def _edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nd = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            nd[j] = min(dp[j] + 1, nd[j - 1] + 1, dp[j - 1] + (ca != cb))
        dp = nd
    return dp[-1]


def _per_second_rms(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    if channels > 1:
        values = values[::channels]
    rms_values = []
    for start in range(0, len(values), sample_rate):
        chunk = values[start : start + sample_rate]
        if not chunk:
            continue
        rms = math.sqrt(sum((x / 32768) ** 2 for x in chunk) / len(chunk))
        rms_values.append(round(rms, 4))
    return rms_values


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "openai-whisper is required for ASR checks; install whisper or run "
            "without RUN_ASR=1"
        ) from exc

    row = _load_first(Path(args.per_request))
    wav_path = Path(row["wav_path"])
    if not wav_path.is_absolute():
        wav_path = Path(args.per_request).resolve().parent / wav_path
        if not wav_path.exists():
            wav_path = Path.cwd() / row["wav_path"]
    if not wav_path.exists():
        raise FileNotFoundError(row["wav_path"])

    model = whisper.load_model(args.model)
    result = model.transcribe(
        str(wav_path),
        language=args.language,
        fp16=args.fp16,
        verbose=False,
    )
    asr_text = str(result.get("text") or "").strip()
    expected_norm = _norm(args.expected)
    asr_norm = _norm(asr_text)
    cer = _edit_distance(expected_norm, asr_norm) / max(1, len(expected_norm))
    rms = _per_second_rms(wav_path)

    output = {
        "per_request": str(Path(args.per_request)),
        "wav_path": str(wav_path),
        "expected": args.expected,
        "predicted_text": row.get("text") or row.get("output_text"),
        "asr_text": asr_text,
        "cer": cer,
        "pass": cer <= args.max_cer and sum(x < args.silence_rms for x in rms) <= args.max_silent_seconds,
        "max_cer": args.max_cer,
        "sec_rms": rms,
        "silence_rms": args.silence_rms,
        "silent_seconds": sum(x < args.silence_rms for x in rms),
        "max_silent_seconds": args.max_silent_seconds,
        "metrics": {
            "ttft_ms": row.get("ttft_ms"),
            "ttfa_ms": row.get("ttfa_ms"),
            "e2e_total_ms": row.get("e2e_total_ms"),
            "audio_duration_s": row.get("audio_duration_s"),
            "audio_chunk_count": row.get("audio_chunk_count"),
        },
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-request", required=True, help="per_request.json path")
    parser.add_argument("--expected", default=DEFAULT_EXPECTED_TEXT)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-cer", type=float, default=0.10)
    parser.add_argument("--silence-rms", type=float, default=0.005)
    parser.add_argument("--max-silent-seconds", type=int, default=0)
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--print", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_check(args)
    if args.out:
        Path(args.out).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.print or not args.out:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    if not output["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
