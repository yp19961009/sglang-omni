# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni S2T single-concurrency SGLang/vLLM alignment runner.

This module is intentionally self-contained.  The default ``orchestrate`` mode
runs on the host and controls the two required containers with ``docker exec``.
The ``client`` mode runs inside either container and sends the same Video-AMME
cases to the server local to that container.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("qwen35_omni_single_s2t")

DEFAULT_MODEL_PATH = (
    "/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315"
)
DEFAULT_DATA_ROOT = "/myapp/data/videoamme"
DEFAULT_HOST_OUTPUT_ROOT = "/home/gangouyu/benchmarks/qwen35_s2t_align"
DEFAULT_SGLANG_CONTAINER = "sglang-omni-dev"
DEFAULT_VLLM_CONTAINER = "qwen35-videoamme-test"
DEFAULT_GPU = "7"
DEFAULT_SGLANG_PORT = 8010
DEFAULT_VLLM_PORT = 8020
DEFAULT_PROMPT = (
    "Use the video and the audio question to answer. "
    "Return the final answer as Answer: $LETTER."
)

VIDEOAMME_REPO_DIR = "datasets--zhaochenyang20--Video_AMME_ci"
DEFAULT_SOURCE_REPO_DIR = "datasets--zhaochenyang20--Video_MME_ci"
ANSWER_RE = re.compile(
    r"(?:answer|final answer)\s*:?\s*(?:option\s*)?[\(\[]?\s*([ABCD])\b",
    re.IGNORECASE,
)
LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
TIMESTAMP_LOOP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?:\s+\1){2,}")
VLLM_TIMING_RE = re.compile(
    r"\[(?P<request_id>chatcmpl-[^\]]+)\]\s+TIMING\s+"
    r"stage=(?P<stage>\S+)\s+ms=(?P<ms>[0-9.]+)\s+"
    r"since_start_ms=(?P<since_start_ms>[0-9.]+)(?P<extras>.*)$"
)
VLLM_INPUT_PREPROCESSOR_RE = re.compile(
    r"thinker input_preprocessor finished,\s+rid:\s+"
    r"(?P<request_id>\S+),\s+cost:(?P<ms>[0-9.]+)"
)
VLLM_HFPREP_RE = re.compile(
    r"HFPREP_PROFILE\s+(?P<kind>\w+):\s+(?P<fields>.*)$"
)
VLLM_PROCESSOR_STATS_RE = re.compile(
    r"Qwen3OmniNextProcessor preprocessing stats:\s+(?P<fields>.*)$"
)
VLLM_MM_ENCODER_RE = re.compile(
    r"encode all mm inputs done,\s+cost:\s+(?P<ms>[0-9.]+)\s+ms,.*"
    r"\(request_id,\s+item_cnt\):\s+(?P<items>\{.*\})"
)
VLLM_REQUEST_FINISHED_RE = re.compile(
    r"Request\s+(?P<request_id>chatcmpl-\S+)\s+finished,\s+"
    r"output length:\s+(?P<output_length>\d+),\s+reason:\s+(?P<reason>\S+),"
)
VLLM_KV_RE = re.compile(
    r"(?P<key>[A-Za-z_][\w.-]*)="
    r"(?P<value>\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|\"[^\"]*\"|'[^']*'|[^ ]+)"
)


@dataclass(frozen=True)
class VideoAMMESample:
    sample_id: str
    question_id: str
    video_id: str
    video_path: str
    audio_path: str
    question: str
    options: list[str]
    expected: str
    duration: str = ""
    domain: str = ""
    sub_category: str = ""
    task_type: str = ""


@dataclass(frozen=True)
class Case:
    case_id: str
    probe_kind: str
    sample: VideoAMMESample


def _container_path(path: str) -> str:
    if path.startswith("/home/gangouyu/"):
        return "/myapp/" + path[len("/home/gangouyu/") :]
    return path


def _host_path(path: str) -> str:
    if path.startswith("/myapp/"):
        return "/home/gangouyu/" + path[len("/myapp/") :]
    return path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _client_profile_run_id(args: argparse.Namespace) -> str:
    return args.profile_run_id or time.strftime("qwen35-client-%Y%m%d-%H%M%S")


def _client_profile_paths(args: argparse.Namespace, *, run_id: str) -> tuple[str, str]:
    output_path = Path(args.output)
    stem_path = output_path.with_suffix("")
    event_dir = args.profile_event_dir or str(
        stem_path.parent / f"{stem_path.name}_events" / run_id
    )
    report_path = args.profile_output or f"{stem_path}_profile.json"
    return str(event_dir), str(report_path)


def _start_client_profile(
    session: Any,
    args: argparse.Namespace,
    *,
    run_id: str,
    event_dir: str,
) -> str | None:
    url = f"{args.base_url.rstrip('/')}/start_request_profile"
    response = session.post(
        url,
        json={"run_id": run_id, "event_dir": event_dir},
        timeout=args.profile_timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    return str(body.get("run_id") or run_id)


def _stop_client_profile(
    session: Any,
    args: argparse.Namespace,
    *,
    run_id: str,
) -> None:
    url = f"{args.base_url.rstrip('/')}/stop_request_profile"
    response = session.post(
        url,
        json={"run_id": run_id},
        timeout=args.profile_timeout_s,
    )
    response.raise_for_status()


def _render_client_profile(event_dir: str, report_path: str) -> dict[str, Any]:
    from sglang_omni.profiler.views import build_report

    report = build_report(event_dir)
    _write_json(Path(report_path), report)
    return report


def _log_client_profile(
    *,
    engine: str,
    event_dir: str,
    report_path: str,
    report: dict[str, Any],
) -> None:
    from sglang_omni.profiler.views import format_table

    LOGGER.info(
        "[%s] stage_profile request_count=%s event_dir=%s report=%s",
        engine,
        report.get("request_count"),
        event_dir,
        report_path,
    )
    stage_rows = report.get("stage_breakdown", [])
    if stage_rows:
        LOGGER.info(
            "[%s] stage_profile:\n%s",
            engine,
            format_table(
                stage_rows,
                ["stage", "interval", "count", "total_ms", "avg_ms", "p95_ms"],
            ).rstrip(),
        )
    hop_rows = report.get("hop_breakdown", [])
    if hop_rows:
        LOGGER.info(
            "[%s] hop_profile:\n%s",
            engine,
            format_table(
                hop_rows,
                ["src", "dst", "kind", "count", "total_ms", "avg_ms", "p95_ms"],
            ).rstrip(),
        )


def _find_snapshot(root: Path, repo_dir: str) -> Path:
    snapshots_dir = root / "hub" / repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(f"Missing snapshots directory: {snapshots_dir}")
    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under: {snapshots_dir}")
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def _repo_id_to_cache_dir(repo_id: str | None) -> str:
    if not repo_id:
        return DEFAULT_SOURCE_REPO_DIR
    return "datasets--" + repo_id.replace("/", "--")


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-D][\.\)]\s*", "", option.strip(), flags=re.IGNORECASE)


def load_videoamme_samples(
    data_root: str,
    *,
    split_json: str = "data/test.jsonl",
    max_scan: int | None = None,
) -> list[VideoAMMESample]:
    root = Path(data_root)
    amme_snapshot = _find_snapshot(root, VIDEOAMME_REPO_DIR)
    source_snapshots: dict[str, Path] = {}
    jsonl_path = amme_snapshot / split_json
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Missing Video-AMME jsonl: {jsonl_path}")

    samples: list[VideoAMMESample] = []
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("question_id")).strip()
            source_repo_id = str(row.get("source_repo_id") or "").strip() or None
            source_repo_dir = _repo_id_to_cache_dir(source_repo_id)
            if source_repo_dir not in source_snapshots:
                source_snapshots[source_repo_dir] = _find_snapshot(root, source_repo_dir)

            audio_path = amme_snapshot / str(row["audio_path"])
            video_rel = str(row.get("source_video_path") or row.get("video_path") or "")
            video_path = source_snapshots[source_repo_dir] / video_rel
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio file for {sample_id}: {audio_path}")
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing video file for {sample_id}: {video_path}")

            samples.append(
                VideoAMMESample(
                    sample_id=sample_id,
                    question_id=str(row.get("question_id", sample_id)).strip(),
                    video_id=str(row.get("video_id", "")).strip(),
                    video_path=str(video_path),
                    audio_path=str(audio_path),
                    question=str(row.get("question", "")).strip(),
                    options=[_strip_option_prefix(str(x)) for x in row.get("options", [])],
                    expected=str(row.get("answer", "")).strip().upper(),
                    duration=str(row.get("duration", "")).strip(),
                    domain=str(row.get("domain", "")).strip(),
                    sub_category=str(row.get("sub_category", "")).strip(),
                    task_type=str(row.get("task_type", "")).strip(),
                )
            )
            if max_scan is not None and len(samples) >= max_scan:
                break
    return samples


def build_cases(
    samples: list[VideoAMMESample],
    *,
    suite: str,
    sample_ids: list[str] | None,
    max_samples: int | None,
    include_cache_probes: bool,
) -> list[Case]:
    by_id = {sample.sample_id: sample for sample in samples}

    if suite == "smoke":
        requested = sample_ids or ["001-1", "001-2", "002-1"]
        selected = [by_id[sid] for sid in requested]
    elif suite == "short":
        selected = samples[: max_samples or 10]
    elif suite == "acceptance":
        selected = samples[: max_samples or 50]
    elif suite == "custom":
        if sample_ids:
            selected = [by_id[sid] for sid in sample_ids]
        else:
            selected = samples[: max_samples or 1]
    else:
        raise ValueError(f"Unknown suite: {suite}")

    cases: list[Case] = []
    if include_cache_probes:
        for sid in ("001-1", "001-2", "002-1"):
            if sid not in by_id:
                raise KeyError(f"Cache probe sample {sid} is not available")
        cases.extend(
            [
                Case("warmup_001_1", "warmup", by_id["001-1"]),
                Case("exact_repeat_001_1", "exact_repeat", by_id["001-1"]),
                Case("same_video_001_2", "same_video_different_question", by_id["001-2"]),
                Case("different_video_002_1", "different_video", by_id["002-1"]),
            ]
        )

    seen_case_ids = {case.case_id for case in cases}
    for idx, sample in enumerate(selected, 1):
        case_id = f"eval_{idx:03d}_{sample.sample_id}"
        if case_id in seen_case_ids:
            continue
        cases.append(Case(case_id, "eval", sample))
    return cases


def parse_prediction(text: str, options: list[str]) -> tuple[str, bool]:
    if not text:
        return "", False
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper(), False
    letter_matches = LETTER_RE.findall(text)
    if letter_matches:
        return letter_matches[-1].upper(), False

    normalized = _normalize_text(text)
    for idx, option in enumerate(options[:4]):
        if _normalize_text(option) in normalized:
            return "ABCD"[idx], True
    return "", False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _sglang_payload(args: argparse.Namespace, sample: VideoAMMESample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "videos": [sample.video_path],
        "audios": [sample.audio_path],
        "modalities": ["text"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.video_fps is not None:
        payload["video_fps"] = args.video_fps
    if args.video_max_frames is not None:
        payload["video_max_frames"] = args.video_max_frames
    if args.video_max_pixels is not None:
        payload["video_max_pixels"] = args.video_max_pixels
    return payload


def _vllm_payload(args: argparse.Namespace, sample: VideoAMMESample) -> dict[str, Any]:
    content = [
        {"type": "video", "video_url": sample.video_path},
        {"type": "audio", "audio_url": sample.audio_path},
        {"type": "text", "text": args.prompt},
    ]
    payload: dict[str, Any] = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stream": False,
        "seed": args.seed,
        "enable_audio_output": False,
        "use_audio_in_video": False,
    }
    if args.video_fps is not None:
        payload["video_fps"] = args.video_fps
    if args.video_max_frames is not None:
        payload["video_max_frames"] = args.video_max_frames
    if args.video_max_pixels is not None:
        payload["video_max_pixels"] = args.video_max_pixels
    return payload


def run_client(args: argparse.Namespace) -> dict[str, Any]:
    import requests

    samples = load_videoamme_samples(args.data_root, max_scan=args.max_scan)
    cases = build_cases(
        samples,
        suite=args.suite,
        sample_ids=args.sample_id,
        max_samples=args.max_samples,
        include_cache_probes=args.include_cache_probes,
    )
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    payload_builder = _sglang_payload if args.engine == "sglang" else _vllm_payload

    records: list[dict[str, Any]] = []
    session = requests.Session()
    profile_info: dict[str, str] | None = None
    profile_report: dict[str, Any] | None = None
    if args.profile and args.engine == "sglang":
        profile_run_id = _client_profile_run_id(args)
        event_dir, report_path = _client_profile_paths(args, run_id=profile_run_id)
        try:
            run_id = _start_client_profile(
                session,
                args,
                run_id=profile_run_id,
                event_dir=event_dir,
            )
            if run_id is not None:
                profile_info = {
                    "run_id": run_id,
                    "event_dir": event_dir,
                    "report_path": report_path,
                }
                LOGGER.info(
                    "[%s] request profiler started run_id=%s event_dir=%s",
                    args.engine,
                    run_id,
                    event_dir,
                )
        except Exception as exc:  # noqa: BLE001 - profiler should not block eval.
            LOGGER.warning("[%s] request profiler start failed: %s", args.engine, exc)
    elif args.profile:
        LOGGER.info("[%s] request profiler is only available for SGLang", args.engine)

    try:
        for idx, case in enumerate(cases, 1):
            payload = payload_builder(args, case.sample)
            start = time.perf_counter()
            error = ""
            body: dict[str, Any] = {}
            text = ""
            usage: dict[str, Any] = {}
            try:
                response = session.post(url, json=payload, timeout=args.timeout_s)
                latency_s = time.perf_counter() - start
                response.raise_for_status()
                body = response.json()
                message = body.get("choices", [{}])[0].get("message", {})
                text = message.get("content") or ""
                usage = body.get("usage") or {}
            except Exception as exc:  # noqa: BLE001 - persisted as benchmark output.
                latency_s = time.perf_counter() - start
                error = str(exc)

            predicted, fallback = parse_prediction(text, case.sample.options)
            is_success = error == ""
            is_correct = is_success and predicted == case.sample.expected
            record = {
                "case_id": case.case_id,
                "probe_kind": case.probe_kind,
                "sample_id": case.sample.sample_id,
                "video_id": case.sample.video_id,
                "question_id": case.sample.question_id,
                "duration": case.sample.duration,
                "domain": case.sample.domain,
                "sub_category": case.sample.sub_category,
                "task_type": case.sample.task_type,
                "video_path": case.sample.video_path,
                "audio_path": case.sample.audio_path,
                "expected": case.sample.expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "is_success": is_success,
                "is_mc_fallback": fallback,
                "has_timestamp_loop": bool(TIMESTAMP_LOOP_RE.search(text)),
                "latency_s": round(latency_s, 4),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "raw_response": text,
                "error": error,
            }
            records.append(record)
            LOGGER.info(
                "[%s] %d/%d %s expected=%s predicted=%s correct=%s latency=%.3fs error=%s",
                args.engine,
                idx,
                len(cases),
                case.case_id,
                case.sample.expected,
                predicted or "-",
                is_correct,
                latency_s,
                error or "-",
            )
            if args.print_raw_response:
                LOGGER.info(
                    "[%s] %s raw_response:\n%s",
                    args.engine,
                    case.case_id,
                    text or "",
                )
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    finally:
        if profile_info is not None:
            try:
                _stop_client_profile(session, args, run_id=profile_info["run_id"])
            except Exception as exc:  # noqa: BLE001 - keep benchmark output.
                LOGGER.warning("[%s] request profiler stop failed: %s", args.engine, exc)
            try:
                profile_report = _render_client_profile(
                    profile_info["event_dir"],
                    profile_info["report_path"],
                )
                _log_client_profile(
                    engine=args.engine,
                    event_dir=profile_info["event_dir"],
                    report_path=profile_info["report_path"],
                    report=profile_report,
                )
            except Exception as exc:  # noqa: BLE001 - keep benchmark output.
                LOGGER.warning("[%s] request profiler render failed: %s", args.engine, exc)

    result = {
        "engine": args.engine,
        "config": {
            "suite": args.suite,
            "base_url": args.base_url,
            "data_root": args.data_root,
            "max_samples": args.max_samples,
            "include_cache_probes": args.include_cache_probes,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "video_fps": args.video_fps,
            "video_max_frames": args.video_max_frames,
            "video_max_pixels": args.video_max_pixels,
        },
        "summary": summarize_records(records),
        "records": records,
    }
    if profile_info is not None and profile_report is not None:
        result["profile"] = {
            **profile_info,
            "request_count": profile_report.get("request_count"),
            "stage_breakdown": profile_report.get("stage_breakdown", []),
            "hop_breakdown": profile_report.get("hop_breakdown", []),
        }
    if args.output:
        _write_json(Path(args.output), result)
    return result


def client_stdout_payload(
    result: dict[str, Any],
    *,
    print_raw_response: bool,
    print_profile: bool = True,
) -> dict[str, Any]:
    payload = dict(result["summary"])
    if print_raw_response:
        payload["raw_responses"] = [
            {
                "case_id": record["case_id"],
                "sample_id": record["sample_id"],
                "expected": record["expected"],
                "predicted": record["predicted"],
                "raw_response": record["raw_response"],
            }
            for record in result.get("records", [])
        ]
    if print_profile and result.get("profile"):
        payload["profile"] = result["profile"]
    return payload


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in records if r["is_success"]]
    eval_records = [r for r in records if r["probe_kind"] == "eval"]
    latencies = [float(r["latency_s"]) for r in successes]
    return {
        "total_cases": len(records),
        "completed_cases": len(successes),
        "failed_cases": len(records) - len(successes),
        "correct_cases": sum(bool(r["is_correct"]) for r in records),
        "accuracy": _ratio(sum(bool(r["is_correct"]) for r in records), len(records)),
        "eval_total": len(eval_records),
        "eval_correct": sum(bool(r["is_correct"]) for r in eval_records),
        "eval_accuracy": _ratio(
            sum(bool(r["is_correct"]) for r in eval_records), len(eval_records)
        ),
        "mc_fallback": sum(bool(r["is_mc_fallback"]) for r in records),
        "timestamp_loop_cases": [r["case_id"] for r in records if r["has_timestamp_loop"]],
        "latency_mean_s": round(sum(latencies) / len(latencies), 4) if latencies else 0,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "completion_tokens_total": sum(int(r["completion_tokens"]) for r in successes),
        "prompt_tokens_total": sum(int(r["prompt_tokens"]) for r in successes),
    }


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return round(ordered[int(idx)], 4)
    return round(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo), 4)


def _coerce_vllm_value(raw: str) -> Any:
    raw = raw.strip().rstrip(",")
    if raw.endswith("ms"):
        with suppress(ValueError):
            return round(float(raw[:-2]), 3)
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"none", "null"}:
        return None
    if raw and raw[0] in "[{\"'":
        with suppress(Exception):
            return ast.literal_eval(raw)
    with suppress(ValueError):
        return int(raw)
    with suppress(ValueError):
        return float(raw)
    return raw


def _parse_vllm_kv_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for match in VLLM_KV_RE.finditer(text):
        fields[match.group("key")] = _coerce_vllm_value(match.group("value"))
    return fields


def _parse_vllm_processor_stats(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {"raw": text}
    for name, count, seconds in re.findall(
        r"(\w+)_items=(\d+)\s+\(([0-9.]+)s\)", text
    ):
        fields[f"{name}_items"] = int(count)
        fields[f"{name}_ms"] = round(float(seconds) * 1000, 3)
    for name, seconds in re.findall(r"(\w+)=([0-9.]+)s", text):
        if name.endswith("_items"):
            continue
        fields[f"{name}_ms"] = round(float(seconds) * 1000, 3)
    return fields


def _vllm_request_entry(
    requests: dict[str, dict[str, Any]], request_id: str
) -> dict[str, Any]:
    return requests.setdefault(
        request_id,
        {
            "timings": [],
            "events": [],
            "scheduler_events": [],
            "finished": None,
        },
    )


def _add_vllm_event(
    requests: dict[str, dict[str, Any]],
    request_id: str,
    *,
    stage: str,
    ms: float | None,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    event: dict[str, Any] = {
        "stage": stage,
        "kind": kind,
        "metadata": dict(metadata or {}),
    }
    if ms is not None:
        event["ms"] = round(float(ms), 3)
    _vllm_request_entry(requests, request_id)["events"].append(event)


def _aggregate_vllm_profile_rows(
    requests: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for request in requests.values():
        for event in request.get("events", []):
            if "ms" not in event:
                continue
            bucket[(str(event["stage"]), str(event["kind"]))].append(float(event["ms"]))

    rows: list[dict[str, Any]] = []
    for (stage, kind), durations in bucket.items():
        durations.sort()
        rows.append(
            {
                "stage": stage,
                "kind": kind,
                "count": len(durations),
                "total_ms": round(sum(durations), 3),
                "avg_ms": round(sum(durations) / len(durations), 3),
                "p50_ms": _percentile(durations, 50),
                "p95_ms": _percentile(durations, 95),
                "max_ms": round(durations[-1], 3),
            }
        )
    rows.sort(key=lambda row: (-float(row["total_ms"]), row["stage"], row["kind"]))
    return rows


def _add_vllm_derived_events(requests: dict[str, dict[str, Any]]) -> None:
    for request in requests.values():
        timings_by_stage = {
            timing.get("stage"): timing for timing in request.get("timings", [])
        }
        first_output = timings_by_stage.get("engine.thinker_first_output")
        final_output = timings_by_stage.get("engine.thinker_final_output")
        if not first_output or not final_output:
            continue

        first_since = first_output.get("since_start_ms")
        final_since = final_output.get("since_start_ms")
        if not isinstance(first_since, (int, float)) or not isinstance(
            final_since, (int, float)
        ):
            continue

        decode_ms = round(float(final_since) - float(first_since), 3)
        if decode_ms < 0:
            continue

        request["events"].append(
            {
                "stage": "engine.thinker_decode_after_first",
                "kind": "derived",
                "ms": decode_ms,
                "metadata": {
                    "from": "engine.thinker_first_output",
                    "to": "engine.thinker_final_output",
                },
            }
        )


def parse_vllm_server_profile(log_path: Path) -> dict[str, Any]:
    """Parse vLLM Qwen3.5-Omni server logs into a request-level profile."""
    requests: dict[str, dict[str, Any]] = {}
    unattributed_events: list[dict[str, Any]] = []
    current_request_id: str | None = None

    if not log_path.is_file():
        return {
            "source_log": str(log_path),
            "request_count": 0,
            "requests": {},
            "stage_breakdown": [],
            "unattributed_events": [],
        }

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        timing = VLLM_TIMING_RE.search(line)
        if timing:
            request_id = timing.group("request_id")
            current_request_id = request_id
            metadata = _parse_vllm_kv_fields(timing.group("extras"))
            event = {
                "stage": timing.group("stage"),
                "kind": "server_timing",
                "ms": round(float(timing.group("ms")), 3),
                "since_start_ms": round(float(timing.group("since_start_ms")), 3),
                "metadata": metadata,
            }
            request = _vllm_request_entry(requests, request_id)
            request["timings"].append(event)
            request["events"].append(event)
            if event["stage"] == "http.total":
                current_request_id = None
            continue

        input_preprocessor = VLLM_INPUT_PREPROCESSOR_RE.search(line)
        if input_preprocessor:
            request_id = input_preprocessor.group("request_id")
            _add_vllm_event(
                requests,
                request_id,
                stage="engine.input_preprocessor",
                ms=float(input_preprocessor.group("ms")),
                kind="engine",
            )
            current_request_id = request_id
            continue

        hfprep = VLLM_HFPREP_RE.search(line)
        if hfprep:
            fields = _parse_vllm_kv_fields(hfprep.group("fields"))
            total_ms = fields.get("total")
            event = {
                "stage": f"hfprep.{hfprep.group('kind')}",
                "kind": "processor",
                "metadata": fields,
            }
            if isinstance(total_ms, (int, float)):
                event["ms"] = round(float(total_ms), 3)
            if current_request_id is None:
                unattributed_events.append(event)
            else:
                _vllm_request_entry(requests, current_request_id)["events"].append(event)
            continue

        processor_stats = VLLM_PROCESSOR_STATS_RE.search(line)
        if processor_stats:
            fields = _parse_vllm_processor_stats(processor_stats.group("fields"))
            total_ms = fields.get("total_preprocess_ms")
            event = {
                "stage": "qwen_processor.preprocess",
                "kind": "processor",
                "metadata": fields,
            }
            if isinstance(total_ms, (int, float)):
                event["ms"] = round(float(total_ms), 3)
            if current_request_id is None:
                unattributed_events.append(event)
            else:
                _vllm_request_entry(requests, current_request_id)["events"].append(event)
            continue

        mm_encoder = VLLM_MM_ENCODER_RE.search(line)
        if mm_encoder:
            with suppress(Exception):
                item_counts = ast.literal_eval(mm_encoder.group("items"))
                if isinstance(item_counts, dict):
                    for request_id, item_count in item_counts.items():
                        _add_vllm_event(
                            requests,
                            str(request_id),
                            stage="engine.mm_encoder",
                            ms=float(mm_encoder.group("ms")),
                            kind="engine",
                            metadata={"item_count": item_count},
                        )
                    continue
            if current_request_id is not None:
                _add_vllm_event(
                    requests,
                    current_request_id,
                    stage="engine.mm_encoder",
                    ms=float(mm_encoder.group("ms")),
                    kind="engine",
                    metadata={"raw_items": mm_encoder.group("items")},
                )
            continue

        if "SCHED_STEP" in line and current_request_id is not None:
            _vllm_request_entry(requests, current_request_id)["scheduler_events"].append(
                {"raw": line.split("SCHED_STEP", 1)[1].strip()}
            )
            continue

        finished = VLLM_REQUEST_FINISHED_RE.search(line)
        if finished:
            request_id = finished.group("request_id")
            _vllm_request_entry(requests, request_id)["finished"] = {
                "output_length": int(finished.group("output_length")),
                "reason": finished.group("reason"),
            }
            current_request_id = request_id

    _add_vllm_derived_events(requests)
    return {
        "source_log": str(log_path),
        "request_count": len(requests),
        "requests": requests,
        "stage_breakdown": _aggregate_vllm_profile_rows(requests),
        "unattributed_events": unattributed_events,
    }


def vllm_profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_count": profile.get("request_count", 0),
        "stage_breakdown": profile.get("stage_breakdown", []),
        "unattributed_events": profile.get("unattributed_events", []),
    }


def wait_for_container_health(
    container: str,
    base_url: str,
    timeout_s: int,
    *,
    server_proc: subprocess.Popen | None = None,
) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    url = f"{base_url.rstrip('/')}/health"
    probe = (
        "python3 - <<'PY'\n"
        "import sys\n"
        "import urllib.error\n"
        "import urllib.request\n"
        f"url = {json.dumps(url)}\n"
        "try:\n"
        "    with urllib.request.urlopen(url, timeout=5) as response:\n"
        "        sys.exit(0 if 200 <= response.status < 300 else 1)\n"
        "except Exception as exc:\n"
        "    print(exc)\n"
        "    sys.exit(1)\n"
        "PY"
    )
    while time.time() < deadline:
        if server_proc is not None and server_proc.poll() is not None:
            raise RuntimeError(
                f"Server process for {container} exited with code "
                f"{server_proc.returncode} while waiting for {url}. "
                f"Last health error: {last_error}"
            )
        try:
            result = subprocess.run(
                ["docker", "exec", container, "bash", "-lc", probe],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            if result.returncode == 0:
                return
            last_error = result.stdout.strip()
        except subprocess.SubprocessError as exc:
            last_error = str(exc)
        time.sleep(2)
    if server_proc is not None and server_proc.poll() is not None:
        raise RuntimeError(
            f"Server process for {container} exited with code "
            f"{server_proc.returncode} while waiting for {url}. "
            f"Last health error: {last_error}"
        )
    raise TimeoutError(
        f"Timed out waiting for {url} inside container {container}: {last_error}"
    )


class DockerServer:
    def __init__(
        self,
        *,
        container: str,
        command: str,
        log_path: Path,
        base_url: str,
        stop_pattern: str,
        startup_timeout_s: int,
    ) -> None:
        self.container = container
        self.command = command
        self.log_path = log_path
        self.base_url = base_url
        self.stop_pattern = stop_pattern
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None
        self._log_handle = None

    def __enter__(self) -> "DockerServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        cmd = ["docker", "exec", "-i", self.container, "bash", "-lc", self.command]
        LOGGER.info("Starting %s server: %s", self.container, self.command)
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            wait_for_container_health(
                self.container,
                self.base_url,
                self.startup_timeout_s,
                server_proc=self.proc,
            )
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def stop(self) -> None:
        stop_script = (
            "python3 - <<'PY'\n"
            "import os\n"
            "import signal\n"
            "import subprocess\n"
            "import time\n"
            f"pattern = {json.dumps(self.stop_pattern)}\n"
            "\n"
            "def pgrep(args):\n"
            "    out = subprocess.run(\n"
            "        args,\n"
            "        check=False,\n"
            "        text=True,\n"
            "        stdout=subprocess.PIPE,\n"
            "        stderr=subprocess.DEVNULL,\n"
            "    ).stdout\n"
            "    return [int(x) for x in out.split() if x.isdigit() and int(x) != os.getpid()]\n"
            "\n"
            "def children(pid):\n"
            "    return pgrep(['pgrep', '-P', str(pid)])\n"
            "\n"
            "def kill_tree(pid, sig):\n"
            "    for child in children(pid):\n"
            "        kill_tree(child, sig)\n"
            "    try:\n"
            "        os.kill(pid, sig)\n"
            "    except ProcessLookupError:\n"
            "        pass\n"
            "\n"
            "roots = pgrep(['pgrep', '-f', pattern])\n"
            "for pid in roots:\n"
            "    kill_tree(pid, signal.SIGTERM)\n"
            "time.sleep(5)\n"
            "for pid in roots:\n"
            "    kill_tree(pid, signal.SIGKILL)\n"
            "PY"
        )
        with suppress(Exception):
            subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container,
                    "bash",
                    "-lc",
                    stop_script,
                ],
                timeout=30,
                check=False,
            )
        if self.proc is not None and self.proc.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=20)
        if self.proc is not None and self.proc.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()


def docker_exec(container: str, command: str, *, timeout_s: int | None = None) -> None:
    LOGGER.info("[%s] %s", container, command)
    subprocess.run(
        ["docker", "exec", "-i", container, "bash", "-lc", command],
        check=True,
        timeout=timeout_s,
    )


def _client_command(args: argparse.Namespace, *, engine: str, output_path: str) -> str:
    base_url = (
        f"http://127.0.0.1:{args.sglang_port}"
        if engine == "sglang"
        else f"http://127.0.0.1:{args.vllm_port}"
    )
    script = "/myapp/sglang-omni/benchmarks/eval/qwen35_omni_single_s2t.py"
    pieces = [
        "python3",
        script,
        "client",
        "--engine",
        engine,
        "--base-url",
        base_url,
        "--data-root",
        args.data_root,
        "--output",
        output_path,
        "--suite",
        args.suite,
        "--model-name",
        args.model_name,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--timeout-s",
        str(args.request_timeout_s),
        "--max-scan",
        str(args.max_scan),
    ]
    if args.max_samples is not None:
        pieces.extend(["--max-samples", str(args.max_samples)])
    for sample_id in args.sample_id or []:
        pieces.extend(["--sample-id", sample_id])
    if args.no_cache_probes:
        pieces.append("--no-cache-probes")
    if args.top_p is not None:
        pieces.extend(["--top-p", str(args.top_p)])
    if args.top_k is not None:
        pieces.extend(["--top-k", str(args.top_k)])
    if args.seed is not None:
        pieces.extend(["--seed", str(args.seed)])
    if args.video_fps is not None:
        pieces.extend(["--video-fps", str(args.video_fps)])
    if args.video_max_frames is not None:
        pieces.extend(["--video-max-frames", str(args.video_max_frames)])
    if args.video_max_pixels is not None:
        pieces.extend(["--video-max-pixels", str(args.video_max_pixels)])
    return "cd /myapp/sglang-omni && " + " ".join(
        _shell_quote(piece) for piece in pieces
    )


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _vllm_server_command(args: argparse.Namespace) -> str:
    return " ".join(
        [
            "cd /myapp/vllm/examples/offline_inference &&",
            f"CUDA_VISIBLE_DEVICES={_shell_quote(args.gpu)}",
            "VLLM_FORCE_DETOKENIZE=1",
            "VLLM_FLASH_ATTN_USE_UPSTREAM=0",
            "python3 qwen_omni_v35_server.py",
            "--thinker-only",
            "--thinker-enforce-eager",
            "--video-needs-metadata",
            f"--model {_shell_quote(args.model_path)}",
            f"--thinker-model {_shell_quote(args.model_path)}",
            "--host 127.0.0.1",
            f"--port {args.vllm_port}",
            "--thinker-devices '[0]'",
            "--max-num-seqs 1",
            "--max-model-len 32768",
            "--block-size 256",
            "--enable-chunked-prefill",
            "--disable-mtp",
            "--mm-processor-cache-type lru",
            f"--gpu-memory-utilization {args.vllm_gpu_memory_utilization}",
        ]
    )


def _sglang_server_command(args: argparse.Namespace, event_dir: str) -> str:
    command = [
        "cd /myapp/sglang-omni &&",
        f"CUDA_VISIBLE_DEVICES={_shell_quote(args.gpu)}",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        f"SGLANG_TORCH_PROFILER_DIR={_shell_quote(str(Path(event_dir).parent / 'torch'))}",
        "python3 -m sglang_omni.cli serve",
        f"--model-path {_shell_quote(args.model_path)}",
        "--text-only",
        f"--model-name {_shell_quote(args.model_name)}",
        "--host 127.0.0.1",
        f"--port {args.sglang_port}",
        "--thinker-gpus 0",
        f"--thinker-cuda-graph {args.sglang_thinker_cuda_graph}",
        f"--thinker-torch-compile {args.sglang_thinker_torch_compile}",
        "--max-running-requests 1",
        "--stages.0.runtime.max_seq_len 32768",
        "--stages.4.runtime.max_seq_len 32768",
        "--stages.4.runtime.sglang_server_args.disable_radix_cache true",
    ]
    if args.sglang_thinker_mem_fraction_static is not None:
        command.append(
            f"--thinker-mem-fraction-static {args.sglang_thinker_mem_fraction_static}"
        )
    else:
        command.append("--encoder-mem-reserve 0.30")
    if args.sglang_max_prefill_tokens is not None:
        command.append(
            "--stages.4.runtime.sglang_server_args.max_prefill_tokens "
            f"{args.sglang_max_prefill_tokens}"
        )
    return " ".join(command)


def _profile_start(container: str, base_url: str, run_id: str, event_dir: str) -> None:
    cmd = (
        "python3 - <<'PY'\n"
        "import requests\n"
        f"requests.post('{base_url}/start_request_profile', json={{'run_id': '{run_id}', 'event_dir': '{event_dir}'}}).raise_for_status()\n"
        "PY"
    )
    docker_exec(container, cmd, timeout_s=30)


def _profile_stop(container: str, base_url: str, run_id: str) -> None:
    cmd = (
        "python3 - <<'PY'\n"
        "import requests\n"
        f"requests.post('{base_url}/stop_request_profile', json={{'run_id': '{run_id}'}}).raise_for_status()\n"
        "PY"
    )
    docker_exec(container, cmd, timeout_s=30)


def _render_profile(container: str, event_dir: str, output_path: str) -> None:
    cmd = (
        "cd /myapp/sglang-omni && "
        f"python3 -m sglang_omni.profiler {_shell_quote(event_dir)} "
        f"--format json --out {_shell_quote(output_path)}"
    )
    docker_exec(container, cmd, timeout_s=120)


def _load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_alignment_report(
    *,
    sglang_result: dict[str, Any],
    vllm_result: dict[str, Any],
    sglang_log_path: Path,
    profile_report_path: Path,
    vllm_profile_report_path: Path | None = None,
) -> dict[str, Any]:
    s_records = {r["case_id"]: r for r in sglang_result.get("records", [])}
    v_records = {r["case_id"]: r for r in vllm_result.get("records", [])}
    case_ids = sorted(set(s_records) | set(v_records))
    diffs = []
    for case_id in case_ids:
        s = s_records.get(case_id)
        v = v_records.get(case_id)
        diffs.append(
            {
                "case_id": case_id,
                "sample_id": (s or v or {}).get("sample_id"),
                "probe_kind": (s or v or {}).get("probe_kind"),
                "expected": (s or v or {}).get("expected"),
                "sglang_predicted": None if s is None else s.get("predicted"),
                "vllm_predicted": None if v is None else v.get("predicted"),
                "prediction_match": (
                    s is not None
                    and v is not None
                    and s.get("predicted") == v.get("predicted")
                ),
                "sglang_correct": None if s is None else s.get("is_correct"),
                "vllm_correct": None if v is None else v.get("is_correct"),
                "sglang_latency_s": None if s is None else s.get("latency_s"),
                "vllm_latency_s": None if v is None else v.get("latency_s"),
                "sglang_timestamp_loop": None if s is None else s.get("has_timestamp_loop"),
                "vllm_timestamp_loop": None if v is None else v.get("has_timestamp_loop"),
                "sglang_error": None if s is None else s.get("error"),
                "vllm_error": None if v is None else v.get("error"),
            }
        )
    s_acc = float(sglang_result.get("summary", {}).get("eval_accuracy", 0.0))
    v_acc = float(vllm_result.get("summary", {}).get("eval_accuracy", 0.0))
    profile_summary = {}
    if profile_report_path.is_file():
        profile = _load_result(profile_report_path)
        profile_summary = {
            "request_count": profile.get("request_count"),
            "stage_breakdown": profile.get("stage_breakdown", []),
            "hop_breakdown": profile.get("hop_breakdown", []),
        }
    vllm_profile = {}
    if vllm_profile_report_path is not None and vllm_profile_report_path.is_file():
        vllm_profile = vllm_profile_summary(_load_result(vllm_profile_report_path))
    return {
        "summary": {
            "sglang_eval_accuracy": s_acc,
            "vllm_eval_accuracy": v_acc,
            "eval_accuracy_delta": round(s_acc - v_acc, 4),
            "prediction_matches": sum(bool(d["prediction_match"]) for d in diffs),
            "total_compared_cases": len(diffs),
            "sglang_timestamp_loop_cases": sglang_result.get("summary", {}).get(
                "timestamp_loop_cases", []
            ),
            "vllm_timestamp_loop_cases": vllm_result.get("summary", {}).get(
                "timestamp_loop_cases", []
            ),
        },
        "diffs": diffs,
        "sglang_summary": sglang_result.get("summary", {}),
        "vllm_summary": vllm_result.get("summary", {}),
        "profile": profile_summary,
        "vllm_profile": vllm_profile,
    }


def run_orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or time.strftime("qwen35-s2t-%Y%m%d-%H%M%S")
    host_output_root = Path(_host_path(args.output_root)).resolve()
    host_run_dir = host_output_root / run_id
    container_run_dir = _container_path(str(host_run_dir))
    host_run_dir.mkdir(parents=True, exist_ok=True)

    sglang_dir = host_run_dir / "sglang"
    vllm_dir = host_run_dir / "vllm"
    sglang_dir.mkdir(exist_ok=True)
    vllm_dir.mkdir(exist_ok=True)

    manifest = {
        "run_id": run_id,
        "host_run_dir": str(host_run_dir),
        "container_run_dir": container_run_dir,
        "sglang_container": args.sglang_container,
        "vllm_container": args.vllm_container,
        "gpu": args.gpu,
        "suite": args.suite,
        "model_path": args.model_path,
        "data_root": args.data_root,
        "sglang_port": args.sglang_port,
        "vllm_port": args.vllm_port,
        "sglang_thinker_cuda_graph": args.sglang_thinker_cuda_graph,
        "sglang_thinker_torch_compile": args.sglang_thinker_torch_compile,
        "sglang_thinker_mem_fraction_static": args.sglang_thinker_mem_fraction_static,
        "sglang_max_prefill_tokens": args.sglang_max_prefill_tokens,
    }
    _write_json(host_run_dir / "manifest.json", manifest)

    vllm_result_path = f"{container_run_dir}/vllm/results.json"
    sglang_result_path = f"{container_run_dir}/sglang/results.json"
    sglang_events_dir = f"{container_run_dir}/sglang/events"
    sglang_profile_path = f"{container_run_dir}/sglang/profile_report.json"
    vllm_profile_path = f"{container_run_dir}/vllm/profile_report.json"

    if args.skip_vllm:
        LOGGER.warning("Skipping vLLM reference run by request")
    else:
        with DockerServer(
            container=args.vllm_container,
            command=_vllm_server_command(args),
            log_path=vllm_dir / "server.log",
            base_url=f"http://127.0.0.1:{args.vllm_port}",
            stop_pattern=f"qwen_omni_v35_server.py.*--port {args.vllm_port}",
            startup_timeout_s=args.startup_timeout_s,
        ):
            docker_exec(
                args.vllm_container,
                _client_command(args, engine="vllm", output_path=vllm_result_path),
                timeout_s=args.client_timeout_s,
            )
        try:
            vllm_profile = parse_vllm_server_profile(vllm_dir / "server.log")
            _write_json(Path(_host_path(vllm_profile_path)), vllm_profile)
            LOGGER.info("vLLM profile written to %s", _host_path(vllm_profile_path))
        except Exception as exc:  # noqa: BLE001 - benchmark output is still useful.
            LOGGER.warning("Failed to render vLLM profile: %s", exc)

    if args.skip_sglang:
        LOGGER.warning("Skipping SGLang run by request")
    else:
        with DockerServer(
            container=args.sglang_container,
            command=_sglang_server_command(args, sglang_events_dir),
            log_path=sglang_dir / "server.log",
            base_url=f"http://127.0.0.1:{args.sglang_port}",
            stop_pattern=f"sglang_omni.cli.*serve.*--port {args.sglang_port}",
            startup_timeout_s=args.startup_timeout_s,
        ):
            if args.profile:
                _profile_start(
                    args.sglang_container,
                    f"http://127.0.0.1:{args.sglang_port}",
                    run_id,
                    sglang_events_dir,
                )
            try:
                docker_exec(
                    args.sglang_container,
                    _client_command(
                        args, engine="sglang", output_path=sglang_result_path
                    ),
                    timeout_s=args.client_timeout_s,
                )
            finally:
                if args.profile:
                    with suppress(Exception):
                        _profile_stop(
                            args.sglang_container,
                            f"http://127.0.0.1:{args.sglang_port}",
                            run_id,
                        )
                    with suppress(Exception):
                        _render_profile(
                            args.sglang_container,
                            sglang_events_dir,
                            sglang_profile_path,
                        )

    host_vllm_result_path = Path(_host_path(vllm_result_path))
    host_sglang_result_path = Path(_host_path(sglang_result_path))
    host_vllm_profile_path = Path(_host_path(vllm_profile_path))
    if host_vllm_result_path.is_file() and host_sglang_result_path.is_file():
        report = build_alignment_report(
            sglang_result=_load_result(host_sglang_result_path),
            vllm_result=_load_result(host_vllm_result_path),
            sglang_log_path=sglang_dir / "server.log",
            profile_report_path=Path(_host_path(sglang_profile_path)),
            vllm_profile_report_path=host_vllm_profile_path,
        )
        _write_json(host_run_dir / "alignment_report.json", report)
        LOGGER.info("Alignment report written to %s", host_run_dir / "alignment_report.json")
        return report

    single_engine_report: dict[str, Any] = {"manifest": manifest}
    if host_vllm_result_path.is_file():
        single_engine_report["vllm_summary"] = _load_result(host_vllm_result_path).get(
            "summary", {}
        )
    if host_vllm_profile_path.is_file():
        single_engine_report["vllm_profile"] = vllm_profile_summary(
            _load_result(host_vllm_profile_path)
        )
    if host_sglang_result_path.is_file():
        single_engine_report["sglang_summary"] = _load_result(
            host_sglang_result_path
        ).get("summary", {})
    if len(single_engine_report) > 1:
        LOGGER.info("Run artifacts written to %s", host_run_dir)
        return single_engine_report

    LOGGER.info("Run artifacts written to %s", host_run_dir)
    return {"manifest": manifest}


def add_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=("smoke", "short", "acceptance", "custom"), default="smoke")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-scan", type=int, default=256)
    parser.add_argument("--no-cache-probes", dest="include_cache_probes", action="store_false")
    parser.set_defaults(include_cache_probes=True)
    parser.add_argument("--model-name", default="qwen35-omni-s2t")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=128)
    parser.add_argument("--video-max-pixels", type=int, default=401408)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--print-raw-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print raw model responses in the client stdout payload and logs.",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect SGLang request profile events and print stage breakdown.",
    )
    parser.add_argument(
        "--profile-run-id",
        default=None,
        help="Run id for client request profiling; defaults to a timestamp.",
    )
    parser.add_argument(
        "--profile-event-dir",
        default=None,
        help="Request profile event directory; defaults next to --output.",
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        help="Request profile report JSON path; defaults next to --output.",
    )
    parser.add_argument("--profile-timeout-s", type=int, default=30)
    parser.add_argument(
        "--print-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print profile summary in the client stdout payload.",
    )


def add_orchestrate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=DEFAULT_HOST_OUTPUT_ROOT)
    parser.add_argument("--sglang-container", default=DEFAULT_SGLANG_CONTAINER)
    parser.add_argument("--vllm-container", default=DEFAULT_VLLM_CONTAINER)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--sglang-port", type=int, default=DEFAULT_SGLANG_PORT)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--suite", choices=("smoke", "short", "acceptance", "custom"), default="smoke")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-scan", type=int, default=256)
    parser.add_argument("--no-cache-probes", action="store_true")
    parser.add_argument("--model-name", default="qwen35-omni-s2t")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=128)
    parser.add_argument("--video-max-pixels", type=int, default=401408)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sglang-thinker-cuda-graph", choices=("on", "off"), default="on")
    parser.add_argument("--sglang-thinker-torch-compile", choices=("on", "off"), default="off")
    parser.add_argument("--sglang-thinker-mem-fraction-static", type=float, default=None)
    parser.add_argument("--sglang-max-prefill-tokens", type=int, default=None)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-sglang", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--startup-timeout-s", type=int, default=1800)
    parser.add_argument("--request-timeout-s", type=int, default=300)
    parser.add_argument("--client-timeout-s", type=int, default=3600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command")

    orchestrate = subparsers.add_parser("orchestrate", help="Run both container engines")
    add_orchestrate_args(orchestrate)

    client = subparsers.add_parser("client", help="Send Video-AMME requests")
    add_client_args(client)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"orchestrate", "client", "-h", "--help"}:
        argv = ["orchestrate", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.command is None:
        args.command = "orchestrate"
    if args.command == "client":
        result = run_client(args)
        print(
            json.dumps(
                client_stdout_payload(
                    result,
                    print_raw_response=args.print_raw_response,
                    print_profile=args.print_profile,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "orchestrate":
        report = run_orchestrate(args)
        print(json.dumps(report.get("summary", report), ensure_ascii=False, indent=2))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
