# SPDX-License-Identifier: Apache-2.0
"""Regenerate key Qwen3.5-Omni performance-report tables from artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval.summarize_vllm_offline_runner_overhead import (
    summarize as summarize_vllm_overhead,
)
from benchmarks.eval.summarize_vllm_omni_log_stages import (
    summarize as summarize_vllm_log_stages,
)
from benchmarks.eval.diagnose_vllm_offline_admission import (
    diagnose_case as diagnose_vllm_admission,
)


@dataclass(frozen=True)
class StressCase:
    concurrency: int
    result_dir: str


@dataclass(frozen=True)
class SyntheticCase:
    scenario: str
    concurrency: int


STRESS_CASES = [
    StressCase(1, "benchmark_audio_50_c1_warm_profile_skipwer"),
    StressCase(2, "benchmark_audio_50_c2_warm_profile_skipwer"),
    StressCase(4, "benchmark_audio_50_c4_profile_skipwer"),
    StressCase(8, "benchmark_audio_50_c8_profile_skipwer"),
    StressCase(16, "benchmark_audio_50_c16_profile_skipwer"),
]

SYNTHETIC_CASES = [
    SyntheticCase("short", 1),
    SyntheticCase("short", 4),
    SyntheticCase("short", 8),
    SyntheticCase("long", 1),
    SyntheticCase("long", 4),
    SyntheticCase("long", 8),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.{digits}f}{suffix}"


def _percent(value: Any, digits: int = 1) -> str:
    try:
        numeric = float(value) * 100.0
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.{digits}f}%"


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _missing_required(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _stress_paths(root: Path) -> list[Path]:
    stress_root = root / "results/qwen35_sglang_mr8_stress_20260619"
    paths: list[Path] = []
    for case in STRESS_CASES:
        case_dir = stress_root / case.result_dir
        paths.append(case_dir / "videoamme_results.json")
        paths.append(case_dir / "whisper_large_v3_local_wer.json")
        paths.append(stress_root / f"request_profile_c{case.concurrency}_profile_skipwer.json")
    # c1/c2 use warmed profile ids in this checkpoint.
    paths.remove(stress_root / "request_profile_c1_profile_skipwer.json")
    paths.remove(stress_root / "request_profile_c2_profile_skipwer.json")
    paths.extend(
        [
            stress_root / "request_profile_c1_warm_profile_skipwer.json",
            stress_root / "request_profile_c2_warm_profile_skipwer.json",
        ]
    )
    return paths


def _synthetic_paths(root: Path) -> list[Path]:
    synthetic_root = root / "results/qwen35_synthetic_speech_20260619"
    paths: list[Path] = []
    for case in SYNTHETIC_CASES:
        label = f"{case.scenario}_c{case.concurrency}"
        paths.append(synthetic_root / label / "synthetic_speech_results.json")
        paths.append(synthetic_root / f"request_profile_{label}_profile.json")
    return paths


def _vllm_paths(root: Path) -> list[Path]:
    return [
        root
        / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
        root
        / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/whisper_large_v3_wer.json",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/vllm_videoamme_report.md",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/vllm_videoamme_report.md",
        root
        / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
        root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
        root
        / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh",
        root
        / "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py",
    ]


def _preproc_paths(root: Path) -> list[Path]:
    return [
        root
        / "results/qwen35_sglang_preproc2_mr8_c8_20260619/benchmark_audio_50_c8_preproc2_profile_skipwer/videoamme_results.json",
        root
        / "results/qwen35_sglang_preproc2_mr8_c8_20260619/request_profile_c8_preproc2_profile_skipwer.json",
        root / "results/qwen35_sglang_mr8_preproc4_stress_20260619",
    ]


def artifact_status(root: Path) -> list[tuple[str, int, int]]:
    groups = [
        ("SGLang stress", _stress_paths(root)),
        ("Synthetic speech", _synthetic_paths(root)),
        ("vLLM comparison", _vllm_paths(root)),
        ("Preprocessing negative", _preproc_paths(root)),
    ]
    return [
        (label, len(paths) - len(_missing_required(paths)), len(paths))
        for label, paths in groups
    ]


def artifact_status_payload(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "group": label,
            "present": present,
            "total": total,
            "missing": total - present,
        }
        for label, present, total in artifact_status(root)
    ]


def print_artifact_status(root: Path) -> list[Path]:
    groups = [
        ("SGLang stress", _stress_paths(root)),
        ("Synthetic speech", _synthetic_paths(root)),
        ("vLLM comparison", _vllm_paths(root)),
        ("Preprocessing negative", _preproc_paths(root)),
    ]
    missing: list[Path] = []
    print("## Artifact Status\n")
    print("| Group | Present | Missing |")
    print("| --- | ---: | ---: |")
    for label, paths in groups:
        group_missing = _missing_required(paths)
        missing.extend(group_missing)
        print(f"| {label} | {len(paths) - len(group_missing)}/{len(paths)} | {len(group_missing)} |")
    if missing:
        print("\nMissing artifacts:")
        for path in missing:
            print(f"- `{path}`")
    print()
    return missing


def sglang_stress_rows(root: Path) -> list[dict[str, Any]]:
    stress_root = root / "results/qwen35_sglang_mr8_stress_20260619"
    rows: list[dict[str, Any]] = []
    for case in STRESS_CASES:
        case_dir = stress_root / case.result_dir
        result = _load_json(case_dir / "videoamme_results.json")
        wer = _load_json(case_dir / "whisper_large_v3_local_wer.json")
        summary = result["summary"]
        speed = result["speed"]
        wer_summary = wer["summary"]
        rows.append(
            {
                "concurrency": case.concurrency,
                "n": summary["total_samples"],
                "accuracy": summary["accuracy"],
                "latency_mean_s": speed["latency_mean_s"],
                "latency_p95_s": speed["latency_p95_s"],
                "rtf_mean": speed["rtf_mean"],
                "rtf_p95": speed["rtf_p95"],
                "throughput_qps": speed["throughput_qps"],
                "audio_throughput_s_per_s": speed["audio_throughput_s_per_s"],
                "wer_corpus": wer_summary["wer_corpus"],
                "result_json": str(case_dir / "videoamme_results.json"),
                "wer_json": str(case_dir / "whisper_large_v3_local_wer.json"),
            }
        )
    return rows


def print_sglang_stress_table(root: Path) -> None:
    print("## SGLang Video-AMME Stress + WER\n")
    print(
        "| c | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | "
        "QPS | Audio Thr | WER Corpus |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sglang_stress_rows(root):
        print(
            f"| {row['concurrency']} | {row['n']} | "
            f"{_percent(row['accuracy'])} | "
            f"{_fmt(row['latency_mean_s'], 3)}s | "
            f"{_fmt(row['latency_p95_s'], 3)}s | "
            f"{_fmt(row['rtf_mean'], 4)} | "
            f"{_fmt(row['rtf_p95'], 4)} | "
            f"{_fmt(row['throughput_qps'], 3)} | "
            f"{_fmt(row['audio_throughput_s_per_s'], 3)} | "
            f"{_percent(row['wer_corpus'], 2)} |"
        )
    print()


def _profile_row(
    profile: dict[str, Any],
    stage: str,
    interval: str,
) -> dict[str, Any]:
    for row in profile.get("stage_breakdown", []):
        if row.get("stage") == stage and row.get("interval") == interval:
            return row
    return {}


def _hop_row(
    profile: dict[str, Any],
    src: str,
    dst: str,
    kind: str,
) -> dict[str, Any]:
    for row in profile.get("hop_breakdown", []):
        if row.get("src") == src and row.get("dst") == dst and row.get("kind") == kind:
            return row
    return {}


def _avg_p95(row: dict[str, Any], digits: int = 0) -> str:
    if not row:
        return ""
    return f"{_fmt(row.get('avg_ms'), digits)}/{_fmt(row.get('p95_ms'), digits)}ms"


def _top_lifecycle(profile: dict[str, Any]) -> str:
    candidates = [
        row
        for row in profile.get("stage_breakdown", [])
        if row.get("interval") == "stage_input_received->stage_complete"
        and row.get("stage") != "coordinator"
    ]
    if not candidates:
        return ""
    top = max(candidates, key=lambda row: float(row.get("avg_ms") or 0))
    return f"{top['stage']} {_avg_p95(top)}"


def _stress_profile_path(root: Path, case: StressCase) -> Path:
    stress_root = root / "results/qwen35_sglang_mr8_stress_20260619"
    if case.concurrency == 1:
        return stress_root / "request_profile_c1_warm_profile_skipwer.json"
    if case.concurrency == 2:
        return stress_root / "request_profile_c2_warm_profile_skipwer.json"
    return stress_root / f"request_profile_c{case.concurrency}_profile_skipwer.json"


def sglang_stage_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in STRESS_CASES:
        profile_path = _stress_profile_path(root, case)
        profile = _load_json(profile_path)
        preproc = _profile_row(
            profile, "preprocessing", "stage_input_received->stage_complete"
        )
        talker = _profile_row(
            profile, "talker_ar", "stage_input_received->stage_complete"
        )
        code2wav = _profile_row(
            profile, "code2wav", "stage_input_received->stage_complete"
        )
        decode = _profile_row(
            profile, "code2wav", "code2wav_decode_start->code2wav_decode_end"
        )
        window = _profile_row(
            profile,
            "code2wav",
            "code2wav_window_collect_start->code2wav_window_collect_end",
        )
        hop = _hop_row(profile, "talker_ar", "code2wav", "stream_chunk")
        rows.append(
            {
                "concurrency": case.concurrency,
                "top_stage": _top_lifecycle(profile),
                "preproc_stage_avg_ms": preproc.get("avg_ms"),
                "preproc_stage_p95_ms": preproc.get("p95_ms"),
                "talker_avg_ms": talker.get("avg_ms"),
                "talker_p95_ms": talker.get("p95_ms"),
                "code2wav_stage_avg_ms": code2wav.get("avg_ms"),
                "code2wav_stage_p95_ms": code2wav.get("p95_ms"),
                "code2wav_decode_avg_ms": decode.get("avg_ms"),
                "code2wav_decode_p95_ms": decode.get("p95_ms"),
                "code2wav_window_avg_ms": window.get("avg_ms"),
                "code2wav_window_p95_ms": window.get("p95_ms"),
                "talker_to_code2wav_hop_avg_ms": hop.get("avg_ms"),
                "talker_to_code2wav_hop_p95_ms": hop.get("p95_ms"),
                "profile_json": str(profile_path),
            }
        )
    return rows


def sglang_preproc_split_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in STRESS_CASES:
        if case.concurrency not in {4, 8, 16}:
            continue
        profile = _load_json(_stress_profile_path(root, case))
        preproc = _profile_row(
            profile, "preprocessing", "stage_input_received->stage_complete"
        )
        actual = _profile_row(
            profile, "preprocessing", "preprocess_start->preprocess_end"
        )
        hf_processor = _profile_row(
            profile,
            "preprocessing",
            "preprocess_hf_processor_start->preprocess_hf_processor_end",
        )
        rows.append(
            {
                "concurrency": case.concurrency,
                "preproc_stage_avg_ms": preproc.get("avg_ms"),
                "preproc_stage_p95_ms": preproc.get("p95_ms"),
                "actual_preprocess_avg_ms": actual.get("avg_ms"),
                "actual_preprocess_p95_ms": actual.get("p95_ms"),
                "hf_processor_avg_ms": hf_processor.get("avg_ms"),
                "hf_processor_p95_ms": hf_processor.get("p95_ms"),
            }
        )
    return rows


def print_sglang_stage_table(root: Path) -> None:
    print("## SGLang Video-AMME Stage Breakdown\n")
    print(
        "| c | Top Stage Avg/P95 | Preproc Stage | Talker | Code2wav Stage | "
        "Decode | Window Collect | Talker->Code2wav Hop |"
    )
    print("| ---: | --- | --- | --- | --- | --- | --- | --- |")
    for row in sglang_stage_rows(root):
        print(
            f"| {row['concurrency']} | {row['top_stage']} | "
            f"{_fmt(row['preproc_stage_avg_ms'], 0)}/{_fmt(row['preproc_stage_p95_ms'], 0)}ms | "
            f"{_fmt(row['talker_avg_ms'], 0)}/{_fmt(row['talker_p95_ms'], 0)}ms | "
            f"{_fmt(row['code2wav_stage_avg_ms'], 0)}/{_fmt(row['code2wav_stage_p95_ms'], 0)}ms | "
            f"{_fmt(row['code2wav_decode_avg_ms'], 0)}/{_fmt(row['code2wav_decode_p95_ms'], 0)}ms | "
            f"{_fmt(row['code2wav_window_avg_ms'], 0)}/{_fmt(row['code2wav_window_p95_ms'], 0)}ms | "
            f"{_fmt(row['talker_to_code2wav_hop_avg_ms'], 1)}/"
            f"{_fmt(row['talker_to_code2wav_hop_p95_ms'], 1)}ms |"
        )
    print()

    print("### SGLang Preprocessing Compute Split\n")
    print("| c | Preproc Stage | Actual Preprocess | HF Processor |")
    print("| ---: | ---: | ---: | ---: |")
    for row in sglang_preproc_split_rows(root):
        print(
            f"| {row['concurrency']} | "
            f"{_fmt(row['preproc_stage_avg_ms'], 0)}/{_fmt(row['preproc_stage_p95_ms'], 0)}ms | "
            f"{_fmt(row['actual_preprocess_avg_ms'], 0)}/{_fmt(row['actual_preprocess_p95_ms'], 0)}ms | "
            f"{_fmt(row['hf_processor_avg_ms'], 0)}/{_fmt(row['hf_processor_p95_ms'], 0)}ms |"
        )
    print()


def print_synthetic_table(root: Path) -> None:
    print("## Synthetic Text-Length + Speech Stress\n")
    print(
        "| Scenario | c | n | Target Chars | Target Words | Audio Mean | "
        "Latency Mean | Latency P95 | RTF Mean | RTF P95 | QPS | Audio Thr |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |"
    )
    for row in synthetic_rows(root):
        print(
            f"| {row['scenario']} | {row['concurrency']} | {row['n']} | "
            f"{_fmt(row['target_chars'], 0)} | "
            f"{_fmt(row['target_words'], 0)} | "
            f"{_fmt(row['audio_duration_mean_s'], 2)}s | "
            f"{_fmt(row['latency_mean_s'], 3)}s | "
            f"{_fmt(row['latency_p95_s'], 3)}s | "
            f"{_fmt(row['rtf_mean'], 4)} | "
            f"{_fmt(row['rtf_p95'], 4)} | "
            f"{_fmt(row['throughput_qps'], 3)} | "
            f"{_fmt(row['audio_throughput_s_per_s'], 3)} |"
        )
    print()


def synthetic_rows(root: Path) -> list[dict[str, Any]]:
    synthetic_root = root / "results/qwen35_synthetic_speech_20260619"
    rows: list[dict[str, Any]] = []
    for case in SYNTHETIC_CASES:
        result_path = (
            synthetic_root
            / f"{case.scenario}_c{case.concurrency}"
            / "synthetic_speech_results.json"
        )
        result = _load_json(result_path)
        speed = result["speed"]
        per_sample = result.get("per_sample", [])
        rows.append(
            {
                "scenario": case.scenario,
                "concurrency": case.concurrency,
                "n": speed["total_requests"],
                "target_chars": _mean_numeric(per_sample, "target_chars"),
                "target_words": _mean_numeric(per_sample, "target_words"),
                "audio_duration_mean_s": speed["audio_duration_mean_s"],
                "latency_mean_s": speed["latency_mean_s"],
                "latency_p95_s": speed["latency_p95_s"],
                "rtf_mean": speed["rtf_mean"],
                "rtf_p95": speed["rtf_p95"],
                "throughput_qps": speed["throughput_qps"],
                "audio_throughput_s_per_s": speed["audio_throughput_s_per_s"],
                "result_json": str(result_path),
            }
        )
    return rows


def synthetic_stage_rows(root: Path) -> list[dict[str, Any]]:
    synthetic_root = root / "results/qwen35_synthetic_speech_20260619"
    rows: list[dict[str, Any]] = []
    for case in SYNTHETIC_CASES:
        profile_path = (
            synthetic_root
            / f"request_profile_{case.scenario}_c{case.concurrency}_profile.json"
        )
        profile = _load_json(profile_path)
        talker = _profile_row(
            profile, "talker_ar", "stage_input_received->stage_complete"
        )
        code2wav = _profile_row(
            profile, "code2wav", "stage_input_received->stage_complete"
        )
        decode = _profile_row(
            profile, "code2wav", "code2wav_decode_start->code2wav_decode_end"
        )
        window = _profile_row(
            profile,
            "code2wav",
            "code2wav_window_collect_start->code2wav_window_collect_end",
        )
        hop = _hop_row(profile, "talker_ar", "code2wav", "stream_chunk")
        rows.append(
            {
                "scenario": case.scenario,
                "concurrency": case.concurrency,
                "talker_avg_ms": talker.get("avg_ms"),
                "talker_p95_ms": talker.get("p95_ms"),
                "code2wav_stage_avg_ms": code2wav.get("avg_ms"),
                "code2wav_stage_p95_ms": code2wav.get("p95_ms"),
                "code2wav_decode_avg_ms": decode.get("avg_ms"),
                "code2wav_decode_p95_ms": decode.get("p95_ms"),
                "code2wav_window_avg_ms": window.get("avg_ms"),
                "code2wav_window_p95_ms": window.get("p95_ms"),
                "talker_to_code2wav_hop_avg_ms": hop.get("avg_ms"),
                "talker_to_code2wav_hop_p95_ms": hop.get("p95_ms"),
                "profile_json": str(profile_path),
            }
        )
    return rows


def print_synthetic_stage_table(root: Path) -> None:
    print("## Synthetic Speech Stage Breakdown\n")
    print(
        "| Scenario | c | Talker | Code2wav Stage | Decode | Window Collect | "
        "Talker->Code2wav Hop |"
    )
    print("| --- | ---: | --- | --- | --- | --- | --- |")
    for row in synthetic_stage_rows(root):
        print(
            f"| {row['scenario']} | {row['concurrency']} | "
            f"{_fmt(row['talker_avg_ms'], 0)}/{_fmt(row['talker_p95_ms'], 0)}ms | "
            f"{_fmt(row['code2wav_stage_avg_ms'], 0)}/{_fmt(row['code2wav_stage_p95_ms'], 0)}ms | "
            f"{_fmt(row['code2wav_decode_avg_ms'], 0)}/{_fmt(row['code2wav_decode_p95_ms'], 0)}ms | "
            f"{_fmt(row['code2wav_window_avg_ms'], 0)}/{_fmt(row['code2wav_window_p95_ms'], 0)}ms | "
            f"{_fmt(row['talker_to_code2wav_hop_avg_ms'], 1)}/"
            f"{_fmt(row['talker_to_code2wav_hop_p95_ms'], 1)}ms |"
        )
    print()


def preproc_rows(root: Path) -> list[dict[str, Any]]:
    baseline = _load_json(
        root
        / "results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json"
    )
    preproc2 = _load_json(
        root
        / "results/qwen35_sglang_preproc2_mr8_c8_20260619/benchmark_audio_50_c8_preproc2_profile_skipwer/videoamme_results.json"
    )
    rows: list[dict[str, Any]] = []
    for label, payload in [("preproc=1 baseline", baseline), ("preproc=2", preproc2)]:
        summary = payload["summary"]
        speed = payload["speed"]
        rows.append(
            {
                "setting": label,
                "completed": speed["completed_requests"],
                "failed": speed["failed_requests"],
                "accuracy": summary["accuracy"],
                "latency_mean_s": speed["latency_mean_s"],
                "latency_p95_s": speed["latency_p95_s"],
                "rtf_mean": speed["rtf_mean"],
                "rtf_p95": speed["rtf_p95"],
                "throughput_qps": speed["throughput_qps"],
            }
        )
    return rows


def print_preproc_table(root: Path) -> None:
    print("## Preprocessing Concurrency c8 Comparison\n")
    print(
        "| Setting | Completed | Failed | Accuracy | Latency Mean | Latency P95 | "
        "RTF Mean | RTF P95 | QPS |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in preproc_rows(root):
        print(
            f"| {row['setting']} | {row['completed']} | {row['failed']} | "
            f"{_percent(row['accuracy'])} | "
            f"{_fmt(row['latency_mean_s'], 3)}s | "
            f"{_fmt(row['latency_p95_s'], 3)}s | "
            f"{_fmt(row['rtf_mean'], 4)} | "
            f"{_fmt(row['rtf_p95'], 4)} | "
            f"{_fmt(row['throughput_qps'], 3)} |"
        )
    print()


def vllm_overhead_rows(root: Path) -> list[dict[str, Any]]:
    return [
        summarize_vllm_overhead(
            root
            / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
            "vLLM-c4",
        ),
        summarize_vllm_overhead(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            "vLLM-c8",
        ),
        summarize_vllm_overhead(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            "vLLM-c8-prebuild-w1",
        ),
        summarize_vllm_overhead(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            "vLLM-c8-prebuild-w4",
        ),
    ]


def vllm_log_stage_rows(root: Path) -> list[dict[str, Any]]:
    return [
        summarize_vllm_log_stages(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log",
            label="vLLM-c1",
            skip_first_requests=4,
        ),
        summarize_vllm_log_stages(
            root
            / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
            label="vLLM-c4",
            skip_first_requests=4,
        ),
        summarize_vllm_log_stages(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
            label="vLLM-c8",
            skip_first_requests=8,
        ),
        summarize_vllm_log_stages(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
            label="vLLM-c8-prebuild-w1",
            skip_first_requests=8,
        ),
        summarize_vllm_log_stages(
            root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
            label="vLLM-c8-prebuild-w4",
            skip_first_requests=8,
        ),
    ]


def vllm_admission_diagnosis_rows(root: Path) -> list[dict[str, Any]]:
    return [
        diagnose_vllm_admission(
            label="vLLM-c4",
            result_json=root
            / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
            run_log=root
            / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
            skip_first_requests=4,
        ),
        diagnose_vllm_admission(
            label="vLLM-c8",
            result_json=root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            run_log=root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
            skip_first_requests=8,
        ),
        diagnose_vllm_admission(
            label="vLLM-c8-prebuild-w1",
            result_json=root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            run_log=root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
            skip_first_requests=8,
        ),
        diagnose_vllm_admission(
            label="vLLM-c8-prebuild-w4",
            result_json=root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            run_log=root
            / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
            skip_first_requests=8,
        ),
    ]


def print_vllm_overhead_table(root: Path) -> None:
    rows = vllm_overhead_rows(root)
    print("## vLLM Offline Runner Overhead\n")
    print(
        "| Label | c | Requests | Runner Wall s | Engine Wall s | Prompt Build s | "
        "Batch-max Sum s | Runner Overhead | Engine Overhead | Runner QPS | "
        "Engine QPS | Batch-max QPS | Runner Audio Thr | Engine Audio Thr |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        print(
            f"| {row['label']} | {row['concurrency']} | {row['requests']} | "
            f"{_fmt(row['runner_wall_time_s'], 1)} | "
            f"{_fmt(row['engine_wall_time_s'], 1)} | "
            f"{_fmt(row['prompt_build_wall_s'], 1)} | "
            f"{_fmt(row['batch_max_sum_s'], 1)} | "
            f"{_fmt(row['runner_overhead_s'], 1)}s "
            f"({_fmt(row['runner_overhead_pct_wall'], 1)}%) | "
            f"{_fmt(row['engine_overhead_s'], 1)}s "
            f"({_fmt(row['engine_overhead_pct_wall'], 1)}%) | "
            f"{_fmt(row['runner_qps'], 4)} | "
            f"{_fmt(row['engine_qps'], 4)} | "
            f"{_fmt(row['batch_max_qps'], 4)} | "
            f"{_fmt(row['runner_audio_throughput_s_per_s'], 4)} | "
            f"{_fmt(row['engine_audio_throughput_s_per_s'], 4)} |"
        )
    print()


def print_vllm_log_stage_table(root: Path) -> None:
    rows = vllm_log_stage_rows(root)
    print("## vLLM Log-Derived Stage Signals\n")
    print(
        "| Run | Skip | Request IDs | Processor Total Avg/P95 | "
        "Input Preproc Avg/P95 | Encoder Avg/P95 | Thinker->Talker Avg/P95 | "
        "Feed->Codec Avg/P95 | Codec Gap Avg/P95 | Talker->C2W Drain Avg/P95 |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        processor = row["processor_total_ms"]
        input_preproc = row["input_preprocessor_lifecycle_ms"]
        encoder = row["encoder_mm_ms"]
        thinker_to_talker = row["thinker_to_talker_feed_ms"]
        first_codec = row["talker_feed_to_first_codec_ms"]
        codec_gap = row["talker_codec_avg_gap_ms"]
        drain = row["talker_to_code2wav_drain_ms"]
        print(
            f"| {row['label']} | {row['skip_first_requests']} | "
            f"{row['included_request_ids']}/{row['total_request_ids']} | "
            f"{_fmt(processor['avg'], 1)}/{_fmt(processor['p95'], 1)}ms | "
            f"{_fmt(input_preproc['avg'], 1)}/{_fmt(input_preproc['p95'], 1)}ms | "
            f"{_fmt(encoder['avg'], 1)}/{_fmt(encoder['p95'], 1)}ms | "
            f"{_fmt(thinker_to_talker['avg'], 1)}/{_fmt(thinker_to_talker['p95'], 1)}ms | "
            f"{_fmt(first_codec['avg'], 1)}/{_fmt(first_codec['p95'], 1)}ms | "
            f"{_fmt(codec_gap['avg'], 1)}/{_fmt(codec_gap['p95'], 1)}ms | "
            f"{_fmt(drain['avg'], 1)}/{_fmt(drain['p95'], 1)}ms |"
        )
    print()
    print(
        "| Run | Included Batches | Req/Batch Avg/P95 | First Engine Lag Avg/P95 | "
        "Last Engine Lag Avg/P95 | Batch Admission Span Avg/P95 |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        request_count = row["batch_request_count"]
        first_lag = row["batch_first_engine_lag_ms"]
        last_lag = row["batch_last_engine_lag_ms"]
        span = row["batch_admission_span_ms"]
        print(
            f"| {row['label']} | {row['included_batch_count']}/{row['batch_count']} | "
            f"{_fmt(request_count['avg'], 1)}/{_fmt(request_count['p95'], 1)} | "
            f"{_fmt(first_lag['avg'], 1)}/{_fmt(first_lag['p95'], 1)}ms | "
            f"{_fmt(last_lag['avg'], 1)}/{_fmt(last_lag['p95'], 1)}ms | "
            f"{_fmt(span['avg'], 1)}/{_fmt(span['p95'], 1)}ms |"
        )
    print()


def print_vllm_admission_diagnosis_table(root: Path) -> None:
    rows = vllm_admission_diagnosis_rows(root)
    print("## vLLM Offline Admission Diagnosis\n")
    print(
        "| Label | c | Runner QPS | Engine QPS | Batch-max QPS | Runner Overhead | "
        "Engine Overhead | "
        "Admission Span Avg/P95 | Last Engine Lag Avg/P95 | "
        "Boundary P95 Encoder/C2W | Diagnosis |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            f"| {row['label']} | {row['concurrency']} | "
            f"{_fmt(row['runner_qps'], 4)} | "
            f"{_fmt(row['engine_qps'], 4)} | "
            f"{_fmt(row['batch_max_qps'], 4)} | "
            f"{_fmt(row['runner_overhead_s'], 1)}s "
            f"({_fmt(row['runner_overhead_pct_wall'], 1)}%) | "
            f"{_fmt(row['engine_overhead_s'], 1)}s "
            f"({_fmt(row['engine_overhead_pct_wall'], 1)}%) | "
            f"{_fmt(row['batch_admission_span_avg_ms'], 1)}/"
            f"{_fmt(row['batch_admission_span_p95_ms'], 1)}ms | "
            f"{_fmt(row['batch_last_engine_lag_avg_ms'], 1)}/"
            f"{_fmt(row['batch_last_engine_lag_p95_ms'], 1)}ms | "
            f"{_fmt(row['encoder_p95_ms'], 1)}/"
            f"{_fmt(row['talker_to_code2wav_drain_p95_ms'], 1)}ms | "
            f"{row['diagnosis']} |"
        )
    print()


def build_json_payload(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "artifact_status": artifact_status_payload(root),
        "tables": {
            "sglang_stress": sglang_stress_rows(root),
            "sglang_stage_breakdown": sglang_stage_rows(root),
            "sglang_preprocessing_split": sglang_preproc_split_rows(root),
            "synthetic_speech": synthetic_rows(root),
            "synthetic_stage_breakdown": synthetic_stage_rows(root),
            "preprocessing_concurrency": preproc_rows(root),
            "vllm_offline_runner_overhead": vllm_overhead_rows(root),
            "vllm_log_stage_signals": vllm_log_stage_rows(root),
            "vllm_admission_diagnosis": vllm_admission_diagnosis_rows(root),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check Qwen3.5-Omni report artifacts and regenerate key Markdown tables."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing results/ and benchmarks/.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check expected artifact presence.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for machine-readable artifact/table summary.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    missing = print_artifact_status(root)
    if missing:
        print(f"ERROR: {len(missing)} expected artifact(s) are missing.", file=sys.stderr)
        raise SystemExit(1)
    if args.json_output is not None:
        _save_json(build_json_payload(root), args.json_output)
    if args.check_only:
        return

    print_sglang_stress_table(root)
    print_sglang_stage_table(root)
    print_synthetic_table(root)
    print_synthetic_stage_table(root)
    print_preproc_table(root)
    print_vllm_overhead_table(root)
    print_vllm_log_stage_table(root)
    print_vllm_admission_diagnosis_table(root)


if __name__ == "__main__":
    main()
