# SPDX-License-Identifier: Apache-2.0
"""Verify key Qwen3.5-Omni performance-report claims from local artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval.summarize_qwen35_omni_report_artifacts import (
    STRESS_CASES,
    SYNTHETIC_CASES,
    _hop_row,
    _load_json,
    _profile_row,
    _stress_profile_path,
)
from benchmarks.eval.summarize_vllm_omni_log_stages import (
    summarize as summarize_vllm_log_stages,
)
from benchmarks.eval.diagnose_vllm_offline_admission import diagnose_case


@dataclass
class CheckResult:
    name: str
    passed: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence": self.evidence,
        }


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    floor = int(pos)
    ceil = min(floor + 1, len(values) - 1)
    if floor == ceil:
        return values[floor]
    return values[floor] + (values[ceil] - values[floor]) * (pos - floor)


def _request_slice_metrics(result_json: Path, *, skip_first: int = 0) -> dict[str, float]:
    payload = _load_json(result_json)
    records = payload["per_sample"][skip_first:]
    success = [record for record in records if record.get("is_success")]
    latencies = [float(record.get("latency_s") or 0) for record in success]
    rtfs = [float(record.get("rtf") or 0) for record in success]
    correct = sum(1 for record in records if record.get("is_correct"))
    return {
        "n": float(len(records)),
        "success": float(len(success)),
        "accuracy": correct / len(records) if records else 0.0,
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95_s": _percentile(latencies, 0.95),
        "rtf_mean": sum(rtfs) / len(rtfs) if rtfs else 0.0,
        "rtf_p95": _percentile(rtfs, 0.95),
    }


def _top_lifecycle_stage(profile: dict[str, Any]) -> str:
    rows = [
        row
        for row in profile.get("stage_breakdown", [])
        if row.get("interval") == "stage_input_received->stage_complete"
        and row.get("stage") != "coordinator"
    ]
    if not rows:
        return ""
    return str(max(rows, key=lambda row: float(row.get("avg_ms") or 0))["stage"])


def _check(name: str, condition: bool, evidence: str) -> CheckResult:
    return CheckResult(name, bool(condition), evidence)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "vllm_c4": root
        / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
        "vllm_c4_wer": root
        / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/whisper_large_v3_wer.json",
        "vllm_c8": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        "vllm_c8_prebuild": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        "vllm_c8_prebuild_w4": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
        "vllm_c1_log": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log",
        "vllm_c4_log": root
        / "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log",
        "vllm_c8_log": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log",
        "vllm_c8_prebuild_log": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log",
        "vllm_c8_prebuild_w4_log": root
        / "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log",
        "sglang_c4": root
        / "results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/videoamme_results.json",
        "sglang_c4_wer": root
        / "results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/whisper_large_v3_wer.json",
        "stress_root": root / "results/qwen35_sglang_mr8_stress_20260619",
        "synthetic_root": root / "results/qwen35_synthetic_speech_20260619",
        "preproc2": root
        / "results/qwen35_sglang_preproc2_mr8_c8_20260619/benchmark_audio_50_c8_preproc2_profile_skipwer/videoamme_results.json",
        "preproc4": root
        / "results/qwen35_sglang_mr8_preproc4_stress_20260619/warmup_audio_50_c8_skipwer/videoamme_results.json",
    }


def verify(root: Path) -> list[CheckResult]:
    paths = _paths(root)
    results: list[CheckResult] = []

    missing = [label for label, path in paths.items() if not path.exists()]
    results.append(
        _check(
            "required artifacts exist",
            not missing,
            "missing=" + ",".join(missing) if missing else "all primary artifacts present",
        )
    )
    if missing:
        return results

    # Strict cross-runtime c=4 comparison uses warmed skip-first-4 request slices.
    vllm_c4 = _request_slice_metrics(paths["vllm_c4"], skip_first=4)
    sglang_c4 = _request_slice_metrics(paths["sglang_c4"], skip_first=4)
    vllm_wer = _load_json(paths["vllm_c4_wer"])["summary"]["wer_corpus"]
    sglang_wer = _load_json(paths["sglang_c4_wer"])["summary"]["wer_corpus"]
    results.append(
        _check(
            "SGLang warmed c4 beats vLLM warmed c4 latency/RTF",
            sglang_c4["latency_mean_s"] < vllm_c4["latency_mean_s"]
            and sglang_c4["latency_p95_s"] < vllm_c4["latency_p95_s"]
            and sglang_c4["rtf_mean"] < vllm_c4["rtf_mean"]
            and sglang_c4["rtf_p95"] < vllm_c4["rtf_p95"],
            (
                f"lat mean {sglang_c4['latency_mean_s']:.3f}<{vllm_c4['latency_mean_s']:.3f}, "
                f"lat p95 {sglang_c4['latency_p95_s']:.3f}<{vllm_c4['latency_p95_s']:.3f}, "
                f"rtf mean {sglang_c4['rtf_mean']:.4f}<{vllm_c4['rtf_mean']:.4f}, "
                f"rtf p95 {sglang_c4['rtf_p95']:.4f}<{vllm_c4['rtf_p95']:.4f}"
            ),
        )
    )
    results.append(
        _check(
            "SGLang warmed c4 preserves accuracy/WER vs vLLM",
            sglang_c4["accuracy"] >= vllm_c4["accuracy"] and sglang_wer <= vllm_wer,
            (
                f"accuracy {sglang_c4['accuracy'] * 100:.1f}%>="
                f"{vllm_c4['accuracy'] * 100:.1f}%, "
                f"WER {sglang_wer * 100:.2f}%<={vllm_wer * 100:.2f}%"
            ),
        )
    )

    vllm_log_rows = [
        summarize_vllm_log_stages(paths["vllm_c1_log"], label="vLLM-c1", skip_first_requests=4),
        summarize_vllm_log_stages(paths["vllm_c4_log"], label="vLLM-c4", skip_first_requests=4),
        summarize_vllm_log_stages(paths["vllm_c8_log"], label="vLLM-c8", skip_first_requests=8),
    ]
    vllm_c8_prebuild_log = summarize_vllm_log_stages(
        paths["vllm_c8_prebuild_log"],
        label="vLLM-c8-prebuild-w1",
        skip_first_requests=8,
    )
    vllm_c8_prebuild_w4_log = summarize_vllm_log_stages(
        paths["vllm_c8_prebuild_w4_log"],
        label="vLLM-c8-prebuild-w4",
        skip_first_requests=8,
    )
    max_encoder_p95 = max(
        float(row["encoder_mm_ms"]["p95"] or 0) for row in vllm_log_rows
    )
    max_thinker_to_talker_p95 = max(
        float(row["thinker_to_talker_feed_ms"]["p95"] or 0)
        for row in vllm_log_rows
    )
    max_talker_to_c2w_p95 = max(
        float(row["talker_to_code2wav_drain_ms"]["p95"] or 0)
        for row in vllm_log_rows
    )
    results.append(
        _check(
            "vLLM log-derived stage boundaries are not hidden bottlenecks",
            max_encoder_p95 < 60
            and max_thinker_to_talker_p95 <= 2
            and max_talker_to_c2w_p95 < 25,
            (
                "encoder_p95="
                + ", ".join(
                    f"{row['label']}={row['encoder_mm_ms']['p95']:.1f}ms"
                    for row in vllm_log_rows
                )
                + "; thinker_to_talker_p95="
                + ", ".join(
                    f"{row['label']}={row['thinker_to_talker_feed_ms']['p95']:.1f}ms"
                    for row in vllm_log_rows
                )
                + "; talker_to_c2w_drain_p95="
                + ", ".join(
                    f"{row['label']}={row['talker_to_code2wav_drain_ms']['p95']:.1f}ms"
                    for row in vllm_log_rows
                )
            ),
        )
    )
    c8_log = next(row for row in vllm_log_rows if row["label"] == "vLLM-c8")
    c4_log = next(row for row in vllm_log_rows if row["label"] == "vLLM-c4")
    c8_admission_span_avg = float(c8_log["batch_admission_span_ms"]["avg"] or 0)
    c8_admission_span_p95 = float(c8_log["batch_admission_span_ms"]["p95"] or 0)
    c4_admission_span_avg = float(c4_log["batch_admission_span_ms"]["avg"] or 0)
    results.append(
        _check(
            "vLLM c8 offline runner is prompt-feed/admission limited",
            c8_admission_span_avg > 30000
            and c8_admission_span_p95 > 40000
            and c8_admission_span_avg > c4_admission_span_avg,
            (
                f"c8 admission span avg/p95={c8_admission_span_avg:.1f}/"
                f"{c8_admission_span_p95:.1f}ms; "
                f"c4 admission span avg={c4_admission_span_avg:.1f}ms"
            ),
        )
    )
    vllm_diagnoses = [
        diagnose_case(
            label="vLLM-c4",
            result_json=paths["vllm_c4"],
            run_log=paths["vllm_c4_log"],
            skip_first_requests=4,
        ),
        diagnose_case(
            label="vLLM-c8",
            result_json=paths["vllm_c8"],
            run_log=paths["vllm_c8_log"],
            skip_first_requests=8,
        ),
    ]
    results.append(
        _check(
            "vLLM offline admission diagnosis classifies c4/c8 as prompt-feed limited",
            all(row["diagnosis"] == "prompt_feed_limited" for row in vllm_diagnoses),
            "; ".join(
                f"{row['label']}={row['diagnosis']} "
                f"overhead={row['runner_overhead_pct_wall']:.1f}% "
                f"span_avg={row['batch_admission_span_avg_ms']:.1f}ms"
                for row in vllm_diagnoses
            ),
        )
    )
    prebuild_payload = _load_json(paths["vllm_c8_prebuild"])
    prebuild_config = prebuild_payload["config"]
    prebuild_speed = prebuild_payload["speed"]
    prebuild_summary = prebuild_payload["summary"]
    results.append(
        _check(
            "vLLM c8 prebuilt-prompt artifact is successful and separates clocks",
            bool(prebuild_config.get("prebuild_prompts"))
            and prebuild_speed.get("completed_requests") == 50
            and prebuild_summary.get("failed") == 0
            and float(prebuild_config.get("runner_wall_clock_s") or 0)
            > float(prebuild_config.get("engine_wall_clock_s") or 0)
            and float(prebuild_config.get("prompt_build_wall_s") or 0) > 0,
            (
                f"completed={prebuild_speed.get('completed_requests')}, "
                f"failed={prebuild_summary.get('failed')}, "
                f"prompt_build_wall={prebuild_config.get('prompt_build_wall_s')}s, "
                f"engine_wall={prebuild_config.get('engine_wall_clock_s')}s, "
                f"runner_wall={prebuild_config.get('runner_wall_clock_s')}s"
            ),
        )
    )
    prebuild_admission_span_avg = float(
        vllm_c8_prebuild_log["batch_admission_span_ms"]["avg"] or 0
    )
    results.append(
        _check(
            "vLLM c8 prebuilt prompts reduce engine admission span",
            prebuild_admission_span_avg < 6000
            and c8_admission_span_avg / max(prebuild_admission_span_avg, 1.0) >= 5,
            (
                f"old_c8_span_avg={c8_admission_span_avg:.1f}ms, "
                f"prebuild_span_avg={prebuild_admission_span_avg:.1f}ms"
            ),
        )
    )
    prebuild_w4_payload = _load_json(paths["vllm_c8_prebuild_w4"])
    prebuild_w4_config = prebuild_w4_payload["config"]
    prebuild_w4_speed = prebuild_w4_payload["speed"]
    prebuild_w4_summary = prebuild_w4_payload["summary"]
    prebuild_w1_runner_wall = float(prebuild_config.get("runner_wall_clock_s") or 0)
    prebuild_w4_runner_wall = float(prebuild_w4_config.get("runner_wall_clock_s") or 0)
    prebuild_w1_prompt_wall = float(prebuild_config.get("prompt_build_wall_s") or 0)
    prebuild_w4_prompt_wall = float(prebuild_w4_config.get("prompt_build_wall_s") or 0)
    prebuild_w4_engine_qps = float(prebuild_w4_speed.get("throughput_qps") or 0)
    prebuild_w1_runner_qps = (
        float(prebuild_speed.get("completed_requests") or 0) / prebuild_w1_runner_wall
        if prebuild_w1_runner_wall
        else 0.0
    )
    prebuild_w4_runner_qps = (
        float(prebuild_w4_speed.get("completed_requests") or 0) / prebuild_w4_runner_wall
        if prebuild_w4_runner_wall
        else 0.0
    )
    prebuild_w4_admission_span_avg = float(
        vllm_c8_prebuild_w4_log["batch_admission_span_ms"]["avg"] or 0
    )
    results.append(
        _check(
            "vLLM c8 prebuilt prompts with 4 workers improve runner-side wall clock",
            bool(prebuild_w4_config.get("prebuild_prompts"))
            and prebuild_w4_config.get("prebuild_workers") == 4
            and prebuild_w4_speed.get("completed_requests") == 50
            and prebuild_w4_summary.get("failed") == 0
            and prebuild_w4_prompt_wall < prebuild_w1_prompt_wall * 0.7
            and prebuild_w4_runner_wall < prebuild_w1_runner_wall * 0.75
            and prebuild_w4_runner_qps > prebuild_w1_runner_qps * 1.4
            and prebuild_w4_engine_qps > 0.5
            and prebuild_w4_admission_span_avg < 5000,
            (
                f"w1_prompt={prebuild_w1_prompt_wall:.1f}s, "
                f"w4_prompt={prebuild_w4_prompt_wall:.1f}s; "
                f"w1_runner={prebuild_w1_runner_wall:.1f}s, "
                f"w4_runner={prebuild_w4_runner_wall:.1f}s; "
                f"w1_runner_qps={prebuild_w1_runner_qps:.4f}, "
                f"w4_runner_qps={prebuild_w4_runner_qps:.4f}, "
                f"w4_engine_qps={prebuild_w4_engine_qps:.4f}, "
                f"w4_span_avg={prebuild_w4_admission_span_avg:.1f}ms"
            ),
        )
    )

    stress_rows: dict[int, dict[str, Any]] = {}
    stress_wers: dict[int, float] = {}
    for case in STRESS_CASES:
        case_dir = paths["stress_root"] / case.result_dir
        stress_rows[case.concurrency] = _load_json(case_dir / "videoamme_results.json")
        stress_wers[case.concurrency] = _load_json(
            case_dir / "whisper_large_v3_local_wer.json"
        )["summary"]["wer_corpus"]

    qps_by_c = {
        concurrency: payload["speed"]["throughput_qps"]
        for concurrency, payload in stress_rows.items()
    }
    c8_qps = qps_by_c[8]
    results.append(
        _check(
            "SGLang stress c8 is throughput peak",
            c8_qps == max(qps_by_c.values()) and stress_rows[16]["speed"]["throughput_qps"] < c8_qps,
            "qps_by_c=" + ", ".join(f"c{c}={qps:.3f}" for c, qps in qps_by_c.items()),
        )
    )
    results.append(
        _check(
            "SGLang stress accuracy/failure stable",
            all(
                payload["summary"]["accuracy"] == 0.7
                and payload["speed"]["failed_requests"] == 0
                for payload in stress_rows.values()
            ),
            "accuracy_by_c="
            + ", ".join(
                f"c{c}={payload['summary']['accuracy'] * 100:.1f}%/"
                f"fail{payload['speed']['failed_requests']}"
                for c, payload in stress_rows.items()
            ),
        )
    )
    wer_values = list(stress_wers.values())
    results.append(
        _check(
            "SGLang stress WER remains stable",
            max(wer_values) <= 0.05 and (max(wer_values) - min(wer_values)) <= 0.02,
            "wer_by_c="
            + ", ".join(f"c{c}={wer * 100:.2f}%" for c, wer in stress_wers.items()),
        )
    )

    top_by_c: dict[int, str] = {}
    code2wav_decode: dict[int, tuple[float, float]] = {}
    hop_p95: dict[int, float] = {}
    actual_preproc_avg: dict[int, float] = {}
    for case in STRESS_CASES:
        profile = _load_json(_stress_profile_path(root, case))
        top_by_c[case.concurrency] = _top_lifecycle_stage(profile)
        decode = _profile_row(
            profile, "code2wav", "code2wav_decode_start->code2wav_decode_end"
        )
        code2wav_decode[case.concurrency] = (
            float(decode.get("avg_ms") or 0),
            float(decode.get("p95_ms") or 0),
        )
        hop = _hop_row(profile, "talker_ar", "code2wav", "stream_chunk")
        hop_p95[case.concurrency] = float(hop.get("p95_ms") or 0)
        actual = _profile_row(
            profile, "preprocessing", "preprocess_start->preprocess_end"
        )
        actual_preproc_avg[case.concurrency] = float(actual.get("avg_ms") or 0)

    results.append(
        _check(
            "stress stage transition matches report",
            all(top_by_c[c] == "talker_ar" for c in [1, 2, 4])
            and all(top_by_c[c] == "preprocessing" for c in [8, 16]),
            "top_by_c=" + ", ".join(f"c{c}={stage}" for c, stage in top_by_c.items()),
        )
    )
    results.append(
        _check(
            "code2wav decode and talker->code2wav hop are not bottlenecks",
            all(avg <= 20 and p95 <= 32 for avg, p95 in code2wav_decode.values())
            and all(p95 <= 25 for p95 in hop_p95.values()),
            "decode(avg/p95)="
            + ", ".join(
                f"c{c}={avg:.1f}/{p95:.1f}ms"
                for c, (avg, p95) in code2wav_decode.items()
            )
            + "; hop_p95="
            + ", ".join(f"c{c}={p95:.1f}ms" for c, p95 in hop_p95.items()),
        )
    )
    results.append(
        _check(
            "preprocessing lifecycle growth is queue/admission dominated",
            all(actual_preproc_avg[c] <= 350 for c in [4, 8, 16]),
            "actual_preproc_avg="
            + ", ".join(f"c{c}={avg:.1f}ms" for c, avg in actual_preproc_avg.items()),
        )
    )

    synthetic_root = paths["synthetic_root"]
    synthetic_rows = {
        (case.scenario, case.concurrency): _load_json(
            synthetic_root
            / f"{case.scenario}_c{case.concurrency}"
            / "synthetic_speech_results.json"
        )
        for case in SYNTHETIC_CASES
    }
    long_rows = {
        concurrency: synthetic_rows[("long", concurrency)]["speed"]
        for concurrency in [1, 4, 8]
    }
    results.append(
        _check(
            "long synthetic speech remains faster than real time",
            all(row["rtf_mean"] < 1.0 and row["audio_duration_mean_s"] > 45 for row in long_rows.values()),
            "long="
            + ", ".join(
                f"c{c}:audio{row['audio_duration_mean_s']:.1f}s/rtf{row['rtf_mean']:.4f}"
                for c, row in long_rows.items()
            ),
        )
    )

    baseline_c8 = stress_rows[8]["speed"]
    preproc2 = _load_json(paths["preproc2"])["speed"]
    preproc4_payload = _load_json(paths["preproc4"])
    preproc4 = preproc4_payload["speed"]
    results.append(
        _check(
            "naive preprocessing parallelism regresses or fails",
            preproc2["throughput_qps"] <= baseline_c8["throughput_qps"] * 0.8
            and preproc2["latency_mean_s"] >= baseline_c8["latency_mean_s"] * 1.2
            and preproc4["failed_requests"] >= 1,
            (
                f"preproc2 qps {preproc2['throughput_qps']:.3f} vs "
                f"baseline {baseline_c8['throughput_qps']:.3f}; "
                f"preproc2 latency {preproc2['latency_mean_s']:.3f} vs "
                f"baseline {baseline_c8['latency_mean_s']:.3f}; "
                f"preproc4 failed {preproc4['failed_requests']}"
            ),
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Qwen3.5-Omni report claims against local artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing results/ and benchmarks/.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for machine-readable claim verification results.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    checks = verify(root)
    print("| Claim | Result | Evidence |")
    print("| --- | --- | --- |")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"| {check.name} | {status} | {check.evidence} |")

    failed = [check for check in checks if not check.passed]
    if args.json_output is not None:
        _save_json(
            {
                "root": str(root),
                "passed": len(failed) == 0,
                "total_checks": len(checks),
                "failed_checks": len(failed),
                "checks": [check.to_dict() for check in checks],
            },
            args.json_output,
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
