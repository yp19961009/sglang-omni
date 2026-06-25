# SPDX-License-Identifier: Apache-2.0
"""Build a SeedTTS-compatible meta.lst from the local Video-AMME CI cache.

The Qwen3.5-Omni performance report uses Video-AMME as the aligned SGLang/vLLM
workload. This helper reuses the same spoken-question audio files as a local
reference-audio smoke set for ``benchmark_omni_seedtts.py`` when the official
SeedTTS cache is not available on the host.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_VIDEOAMME_CACHE = Path("/home/gangouyu/data/videoamme")
DEFAULT_OUTPUT = Path("results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst")


def _sanitize_field(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "/")
    return " ".join(text.split())


def _find_snapshot(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "data/test.jsonl").is_file() and (path / "audios").is_dir():
        return path

    candidates = sorted(
        path.glob(
            "hub/datasets--zhaochenyang20--Video_AMME_ci/snapshots/*/data/test.jsonl"
        )
    )
    if not candidates:
        candidates = sorted(
            path.glob("datasets--zhaochenyang20--Video_AMME_ci/snapshots/*/data/test.jsonl")
        )
    if not candidates:
        raise FileNotFoundError(
            "Could not find Video-AMME CI data/test.jsonl under "
            f"{path}. Pass --snapshot-dir explicitly."
        )
    return candidates[-1].parents[1]


def _target_text(row: dict[str, Any], mode: str) -> str:
    if mode == "audio_text":
        return _sanitize_field(row.get("audio_text"))
    if mode == "question":
        return _sanitize_field(row.get("question"))
    if mode == "qa_prompt":
        options = row.get("options") or []
        option_text = " ".join(
            f"{chr(ord('A') + idx)}. {_sanitize_field(option)}"
            for idx, option in enumerate(options)
        )
        return _sanitize_field(
            f"Question: {row.get('question', '')} Options: {option_text} "
            "Briefly answer and end with: Answer: letter."
        )
    raise ValueError(f"Unsupported target mode: {mode}")


def build_meta(
    *,
    snapshot_dir: Path,
    output: Path,
    max_samples: int | None,
    min_audio_duration_s: float,
    max_audio_duration_s: float | None,
    target_mode: str,
) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.resolve()
    jsonl_path = snapshot_dir / "data/test.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    durations: list[float] = []
    target_chars: list[int] = []
    skipped_missing_audio = 0
    skipped_duration = 0

    with jsonl_path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            duration = float(row.get("audio_duration_s") or 0.0)
            if duration < min_audio_duration_s:
                skipped_duration += 1
                continue
            if max_audio_duration_s is not None and duration > max_audio_duration_s:
                skipped_duration += 1
                continue

            audio_rel = row.get("audio_path")
            audio_path = (snapshot_dir / audio_rel).resolve() if audio_rel else Path()
            if not audio_path.is_file():
                skipped_missing_audio += 1
                continue

            ref_text = _sanitize_field(row.get("audio_text"))
            target_text = _target_text(row, target_mode)
            if not ref_text or not target_text:
                continue

            sample_id = _sanitize_field(row.get("sample_id"))
            rows.append(f"{sample_id}|{ref_text}|{audio_path}|{target_text}")
            durations.append(duration)
            target_chars.append(len(target_text))
            if max_samples is not None and len(rows) >= max_samples:
                break

    output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def stat(values: list[float | int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    return {
        "snapshot_dir": str(snapshot_dir),
        "source_jsonl": str(jsonl_path),
        "output": str(output),
        "target_mode": target_mode,
        "rows": len(rows),
        "max_samples": max_samples,
        "skipped_missing_audio": skipped_missing_audio,
        "skipped_duration": skipped_duration,
        "audio_duration_s": stat(durations),
        "target_chars": stat(target_chars),
    }


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a SeedTTS-compatible meta.lst from local Video-AMME CI."
    )
    parser.add_argument(
        "--videoamme-cache",
        type=Path,
        default=DEFAULT_VIDEOAMME_CACHE,
        help="Video-AMME cache root containing hub/datasets--... snapshots.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Direct Video-AMME CI snapshot directory with data/test.jsonl.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--min-audio-duration-s", type=float, default=0.0)
    parser.add_argument("--max-audio-duration-s", type=float, default=None)
    parser.add_argument(
        "--target-mode",
        choices=["audio_text", "question", "qa_prompt"],
        default="audio_text",
        help="Text field to synthesize in benchmark_omni_seedtts.py.",
    )
    args = parser.parse_args()

    snapshot_dir = (
        args.snapshot_dir.expanduser().resolve()
        if args.snapshot_dir is not None
        else _find_snapshot(args.videoamme_cache)
    )
    summary = build_meta(
        snapshot_dir=snapshot_dir,
        output=args.output,
        max_samples=args.max_samples,
        min_audio_duration_s=args.min_audio_duration_s,
        max_audio_duration_s=args.max_audio_duration_s,
        target_mode=args.target_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_output is not None:
        _save_json(summary, args.summary_output)


if __name__ == "__main__":
    main()
