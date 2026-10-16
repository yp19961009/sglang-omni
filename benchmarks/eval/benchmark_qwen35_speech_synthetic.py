# SPDX-License-Identifier: Apache-2.0
"""Synthetic text-to-speech stress benchmark for Qwen3.5-Omni.

The benchmark uses local fixed prompts so performance sweeps do not depend on
external datasets. It calls the OpenAI-compatible chat completions endpoint with
``modalities=["text", "audio"]`` and computes latency, RTF, audio throughput,
and token throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.benchmarker.utils import get_wav_duration, save_json_results
from benchmarks.metrics.performance import compute_speed_metrics, print_speed_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


SHORT_TEXT = (
    "The system is ready. Please speak this single sentence clearly and calmly."
)

MEDIUM_TEXT = (
    "Today we are measuring a multimodal speech generation pipeline under "
    "controlled load. The input is plain text, the output is speech, and the "
    "important metrics are end to end latency, real time factor, audio "
    "throughput, and the stability of each internal stage."
)

LONG_TEXT = (
    "This long-form request is designed to stress the talker and code to wave "
    "path without involving video or audio encoders. Read it at a natural pace. "
    "A useful performance report should separate cold start time from warmed "
    "steady state, distinguish prefill from autoregressive decoding, and track "
    "whether downstream audio synthesis waits on the talker or becomes a "
    "compute bottleneck by itself. In a healthy deployment, short answers should "
    "complete quickly, moderate concurrency should improve throughput without "
    "large tail latency, and very high concurrency should reveal the saturation "
    "point where queueing dominates. This paragraph intentionally contains "
    "several sentences so that generated audio lasts much longer than the short "
    "Video-AMME answers. The goal is not to test language accuracy; the goal is "
    "to produce a repeatable, length-controlled load that makes stage-level "
    "performance easy to compare across runs and across serving engines."
)

SCENARIO_TEXTS = {
    "short": SHORT_TEXT,
    "medium": MEDIUM_TEXT,
    "long": LONG_TEXT,
}


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    scenario: str
    target_text: str


def _make_samples(scenario: str, samples_per_scenario: int) -> list[SyntheticSample]:
    scenarios = list(SCENARIO_TEXTS) if scenario == "all" else [scenario]
    samples: list[SyntheticSample] = []
    for name in scenarios:
        text = SCENARIO_TEXTS[name]
        for idx in range(samples_per_scenario):
            samples.append(
                SyntheticSample(
                    sample_id=f"{name}_{idx:03d}",
                    scenario=name,
                    target_text=text,
                )
            )
    return samples


def _build_prompt(target_text: str) -> str:
    return (
        "Read the following text out loud exactly once. Do not add an "
        "introduction or commentary.\n\n"
        f"{target_text}"
    )


def _apply_response(
    result: RequestResult,
    body: dict[str, Any],
    *,
    audio_output_dir: str,
    sample_id: str,
) -> bool:
    message = body.get("choices", [{}])[0].get("message", {})
    result.text = message.get("content", "") or ""
    audio_obj = message.get("audio")
    if not isinstance(audio_obj, dict):
        result.error = "No audio in response"
        return False
    audio_b64 = audio_obj.get("data", "")
    if not audio_b64:
        result.error = "Empty audio data in response"
        return False
    try:
        wav_bytes = base64.b64decode(audio_b64, validate=True)
        result.audio_duration_s = round(get_wav_duration(wav_bytes), 4)
    except (binascii.Error, ValueError, struct.error) as exc:
        result.error = f"Invalid audio data: {exc}"
        return False

    usage = body.get("usage", {})
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)

    os.makedirs(audio_output_dir, exist_ok=True)
    wav_path = os.path.join(audio_output_dir, f"{sample_id}.wav")
    with open(wav_path, "wb") as fp:
        fp.write(wav_bytes)
    result.wav_path = wav_path
    result.is_success = True
    return True


def make_send_fn(
    *,
    model: str,
    api_url: str,
    audio_output_dir: str,
    voice: str | None,
    max_tokens: int,
    temperature: float,
) -> SendFn:
    async def send_fn(
        session: aiohttp.ClientSession,
        sample: SyntheticSample,
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=sample.target_text[:80],
        )
        audio_config: dict[str, Any] = {"format": "wav"}
        if voice:
            audio_config["voice"] = voice
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": _build_prompt(sample.target_text)}],
            "modalities": ["text", "audio"],
            "audio": audio_config,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": {
                "sample_id": sample.sample_id,
                "scenario": sample.scenario,
            },
        }

        start = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                body = await response.json()
            if not _apply_response(
                result,
                body,
                audio_output_dir=audio_output_dir,
                sample_id=sample.sample_id,
            ):
                return result
            elapsed = time.perf_counter() - start
            result.engine_time_s = elapsed
            if result.audio_duration_s > 0:
                result.rtf = elapsed / result.audio_duration_s
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start
        return result

    return send_fn


def _record(sample: SyntheticSample, result: RequestResult) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "scenario": sample.scenario,
        "target_chars": len(sample.target_text),
        "target_words": len(sample.target_text.split()),
        "latency_s": round(result.latency_s, 4),
        "audio_duration_s": (
            round(result.audio_duration_s, 4) if result.audio_duration_s > 0 else None
        ),
        "rtf": round(result.rtf, 4) if result.rtf > 0 else None,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "wav_path": result.wav_path,
        "is_success": result.is_success,
        "error": result.error,
        "raw_response": result.text,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    samples = _make_samples(args.scenario, args.samples_per_scenario)
    audio_output_dir = os.path.abspath(os.path.join(args.output_dir, "audio"))
    send_fn = make_send_fn(
        model=args.model,
        api_url=api_url,
        audio_output_dir=audio_output_dir,
        voice=args.voice,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            disable_tqdm=args.disable_tqdm,
            timeout_s=args.timeout_s,
        )
    )
    results = await runner.run(samples, send_fn)
    speed = compute_speed_metrics(results, wall_clock_s=runner.wall_clock_s)
    per_sample = [_record(sample, result) for sample, result in zip(samples, results)]
    output = {
        "summary": {
            "total_samples": len(samples),
            "completed": speed.get("completed_requests", 0),
            "failed": speed.get("failed_requests", len(samples)),
        },
        "speed": speed,
        "config": {
            "model": args.model,
            "base_url": base_url,
            "scenario": args.scenario,
            "samples_per_scenario": args.samples_per_scenario,
            "max_concurrency": args.max_concurrency,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "voice": args.voice,
        },
        "per_sample": per_sample,
    }
    save_json_results(output, args.output_dir, "synthetic_speech_results.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Qwen3.5-Omni text-to-speech stress benchmark."
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3_5-omni")
    parser.add_argument(
        "--scenario",
        choices=["short", "medium", "long", "all"],
        default="all",
    )
    parser.add_argument("--samples-per-scenario", type=int, default=8)
    parser.add_argument("--output-dir", default="results/qwen35_synthetic_speech")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--voice", default="m02")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args))
    result_path = os.path.join(args.output_dir, "synthetic_speech_results.json")
    with open(result_path, encoding="utf-8") as fp:
        output = json.load(fp)
    print_speed_summary(
        output["speed"],
        args.model,
        concurrency=args.max_concurrency,
        title="Synthetic Speech Speed",
    )


if __name__ == "__main__":
    main()
