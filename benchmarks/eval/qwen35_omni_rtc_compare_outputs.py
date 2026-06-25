#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen3.5-Omni RTC C1 output-capture artifacts.

The script intentionally checks output sanity instead of exact text parity:
the RTC prompt is open-ended, so the useful acceptance criteria are non-empty
text, no pure-bang degeneration, saved playable audio, and comparable latency
metrics.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any


def _load_first(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        return data
    raise TypeError(f"unsupported artifact shape: {path}")


def _is_pure_bang(text: str | None) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and set(stripped) <= {"!"}


def _wav_duration(path: str | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with wave.open(str(p), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / rate if rate else None
    except wave.Error:
        return None


def _num(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _text(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("text") or row.get("raw_response") or "")


def _status(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    text = _text(row)
    wav_path = row.get("wav_path")
    if not text.strip():
        return "bad: empty text"
    if _is_pure_bang(text):
        return "bad: pure bang text"
    if wav_path and _wav_duration(str(wav_path)) is None:
        return "bad: wav missing/unreadable"
    return "ok"


def _row(name: str, row: dict[str, Any] | None) -> list[str]:
    if not row:
        return [name, "missing", "-", "-", "-", "-", "-", "-", "-"]
    wav_duration = _wav_duration(row.get("wav_path"))
    audio_duration = _num(row, "audio_duration_s")
    if audio_duration is None:
        audio_duration = wav_duration
    return [
        name,
        _status(row),
        _fmt(_num(row, "ttft_ms", "thinker_ttft_ms", "avg_ttft_ms")),
        _fmt(_num(row, "ttfa_ms", "e2e_ttfa_ms", "avg_ttfa_ms")),
        _fmt(_num(row, "e2e_total_ms")),
        _fmt(audio_duration, precision=3),
        str(row.get("prompt_tokens") or row.get("num_prompt_tokens") or "-"),
        str(row.get("completion_tokens") or row.get("thinker_token_count") or "-"),
        str(row.get("wav_path") or "-"),
    ]


def _fmt(value: float | None, precision: int = 2) -> str:
    return "-" if value is None else f"{value:.{precision}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    sglang = _load_first(args.sglang)
    vllm = _load_first(args.vllm)
    rows = [_row("SGLang-Omni", sglang), _row("vLLM", vllm)]

    summary = {
        "sglang_status": _status(sglang),
        "vllm_status": _status(vllm),
        "sglang_text": _text(sglang),
        "vllm_text": _text(vllm),
        "sglang_wav_duration_s": _wav_duration(sglang.get("wav_path") if sglang else None),
        "vllm_wav_duration_s": _wav_duration(vllm.get("wav_path") if vllm else None),
    }

    lines = [
        "# Qwen3.5-Omni RTC C1 Output Check",
        "",
        _md_table(
            [
                "Runtime",
                "status",
                "TTFT ms",
                "TTFA ms",
                "E2E ms",
                "audio s",
                "prompt tok",
                "output tok",
                "wav",
            ],
            rows,
        ),
        "",
        "## Text Preview",
        "",
        f"- SGLang-Omni: `{_preview(summary['sglang_text']) or '-'}`",
        f"- vLLM: `{_preview(summary['vllm_text']) or '-'}`",
        "",
        "## Acceptance",
        "",
        "- Pass if each runtime has non-empty non-bang text and readable generated audio.",
        "- Do not require exact text or waveform equality for this open-ended RTC prompt.",
    ]
    return "\n".join(lines) + "\n", summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang", help="SGLang per_request.json")
    parser.add_argument("--vllm", help="vLLM per_request.json")
    parser.add_argument("--out", help="Markdown report path")
    parser.add_argument("--json-out", help="JSON summary path")
    parser.add_argument("--print", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, summary = build_report(args)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.print or not args.out:
        print(report)


if __name__ == "__main__":
    main()
