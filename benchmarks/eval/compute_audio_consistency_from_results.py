# SPDX-License-Identifier: Apache-2.0
"""Compute text/audio consistency WER from saved benchmark result JSON files.

This post-processing helper lets serving performance runs skip ASR during the
hot path and run WER later through either an OpenAI-compatible ASR router or a
local openai-whisper model.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path
from typing import Any

from jiwer import process_words

from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.tts import (
    SampleOutput,
    compute_text_audio_consistency_from_records,
    normalize_text,
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def _normalize_records(
    records: list[dict[str, Any]],
    *,
    path_root: Path,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        wav_path = str(item.get("wav_path") or "")
        if wav_path and not os.path.isabs(wav_path):
            item["wav_path"] = str((path_root / wav_path).resolve())
        normalized.append(item)
    return normalized


@functools.lru_cache(maxsize=2)
def _load_local_whisper_model(model_name: str, device: str):
    import torch
    import whisper

    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    return whisper.load_model(model_name, device=device)


def _compute_with_local_whisper(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        model = _load_local_whisper_model(args.local_whisper_model, args.asr_device)
    except ImportError as exc:
        raise RuntimeError(
            "Local Whisper mode requires openai-whisper and torch in the active "
            "Python environment."
        ) from exc

    device = args.asr_device
    outputs: list[SampleOutput] = []
    whisper_kwargs: dict[str, Any] = {"task": "transcribe"}
    if args.lang:
        whisper_kwargs["language"] = "en" if args.lang == "en" else args.lang
    if device.startswith("cuda"):
        whisper_kwargs["fp16"] = True

    for idx, record in enumerate(records, start=1):
        sample_id = str(record.get(args.sample_id_key) or "")
        output = SampleOutput(
            sample_id=sample_id,
            target_text=" ".join(str(record.get(args.text_key) or "").split()),
            latency_s=float(record.get("latency_s") or 0),
            audio_duration_s=float(record.get("audio_duration_s") or 0),
        )
        wav_path = str(record.get("wav_path") or "")
        if not bool(record.get("is_success")):
            output.error = str(record.get("error") or "Request was not successful")
            outputs.append(output)
            continue
        if not wav_path or not os.path.isfile(wav_path):
            output.error = f"Missing wav_path: {wav_path}"
            outputs.append(output)
            continue

        asr_t0 = time.perf_counter()
        try:
            transcript = model.transcribe(wav_path, **whisper_kwargs)
            output.asr_latency_s = time.perf_counter() - asr_t0
            output.whisper_text = str(transcript.get("text") or "").strip()
            output.ref_norm = normalize_text(output.target_text, args.lang)
            output.hyp_norm = normalize_text(output.whisper_text, args.lang)
            if not output.ref_norm:
                output.error = "Empty reference after normalization"
            else:
                measures = process_words(output.ref_norm, output.hyp_norm)
                output.wer = measures.wer
                output.substitutions = measures.substitutions
                output.deletions = measures.deletions
                output.insertions = measures.insertions
                output.hits = measures.hits
                output.is_success = True
        except Exception as exc:
            output.asr_latency_s = time.perf_counter() - asr_t0
            output.error = f"Transcription failed: {exc}"
        outputs.append(output)

        if args.progress_interval and idx % args.progress_interval == 0:
            print(f"processed {idx}/{len(records)}")

    per_sample = [
        {
            "id": output.sample_id,
            "is_success": output.is_success,
            "wer": output.wer if output.is_success else None,
            "ref_text": output.target_text[:100],
            "hyp_text": output.whisper_text[:100],
            "ref_norm": output.ref_norm,
            "hyp_norm": output.hyp_norm,
            "audio_duration_s": output.audio_duration_s,
            "latency_s": output.latency_s,
            "asr_latency_s": output.asr_latency_s,
            "error": output.error,
        }
        for output in outputs
    ]
    return {
        "summary": _calculate_local_whisper_metrics(outputs, args),
        "per_sample": per_sample,
    }


def _calculate_local_whisper_metrics(
    outputs: list[SampleOutput],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from benchmarks.metrics.wer import calculate_wer_metrics

    metrics = calculate_wer_metrics(outputs, args.lang)
    metrics.update(
        {
            "asr_model": args.local_whisper_model,
            "asr_backend": "local_whisper",
            "asr_device": args.asr_device,
        }
    )
    return metrics


def _compute_for_file(args: argparse.Namespace, result_path: Path) -> dict[str, Any]:
    result_path = result_path.resolve()
    result_json = _load_json(result_path)
    per_sample = result_json.get("per_sample")
    if not isinstance(per_sample, list):
        raise ValueError(f"{result_path} does not contain a per_sample list")

    records = _normalize_records(per_sample, path_root=args.path_root.resolve())
    output_path = (
        Path(args.output)
        if args.output and len(args.results) == 1
        else result_path.parent / args.output_name
    )

    start = time.perf_counter()
    if args.local_whisper_model:
        wer = _compute_with_local_whisper(records, args)
    else:
        wer = compute_text_audio_consistency_from_records(
            records,
            args.lang,
            args.asr_device,
            sample_id_key=args.sample_id_key,
            text_key=args.text_key,
            asr_router_port=args.asr_router_port,
            asr_model_path=args.asr_model_path,
            asr_concurrency=args.asr_concurrency,
        )
    wall_time_s = round(time.perf_counter() - start, 4)
    wer["summary"].update(
        {
            "asr_model": args.local_whisper_model or args.asr_model_path,
            "asr_router_port": args.asr_router_port,
            "asr_device": args.asr_device,
            "wall_time_s": wall_time_s,
            "source_result_json": str(result_path),
        }
    )
    _save_json(wer, output_path)

    if args.update_result_json:
        result_json["wer"] = wer
        _save_json(result_json, result_path)

    print(f"\nSaved WER: {output_path}")
    print_wer_summary(wer["summary"], result_json.get("config", {}).get("model", "model"))
    return wer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER from saved benchmark per_sample records."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to benchmark result JSON files with a per_sample list.",
    )
    parser.add_argument(
        "--asr-router-port",
        type=int,
        help="Port of a running SGLang Omni ASR router.",
    )
    parser.add_argument(
        "--asr-model-path",
        default="openai/whisper-large-v3",
        help="ASR model name served by the router.",
    )
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument("--asr-concurrency", type=int, default=1)
    parser.add_argument(
        "--local-whisper-model",
        default=None,
        help=(
            "Use openai-whisper in-process instead of an ASR router. For example, "
            "`large-v3` reuses /root/.cache/whisper/large-v3.pt when present."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print local Whisper progress every N records; use 0 to disable.",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--sample-id-key", default="sample_id")
    parser.add_argument("--text-key", default="raw_response")
    parser.add_argument(
        "--path-root",
        type=Path,
        default=Path.cwd(),
        help="Root used to resolve relative wav_path values.",
    )
    parser.add_argument(
        "--output-name",
        default="whisper_large_v3_wer.json",
        help="Output filename next to each input result JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output path. Only valid with a single input result JSON.",
    )
    parser.add_argument(
        "--update-result-json",
        action="store_true",
        help="Also embed the computed WER object into the source result JSON.",
    )
    args = parser.parse_args()

    if args.output and len(args.results) != 1:
        parser.error("--output can only be used with one input result JSON")
    if args.local_whisper_model is None and args.asr_router_port is None:
        parser.error("--asr-router-port is required unless --local-whisper-model is set")

    for path in args.results:
        _compute_for_file(args, Path(path))


if __name__ == "__main__":
    main()
