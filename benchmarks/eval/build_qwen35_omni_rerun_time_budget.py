# SPDX-License-Identifier: Apache-2.0
"""Build the rerun wall-time and compute-budget matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "rerun_time_budget.json"

REQUIRED_COMMAND_IDS = {
    "run_full_audit",
    "launch_sglang_optimized",
    "sglang_videoamme_stress",
    "sglang_synthetic_text_to_speech",
    "sglang_recompute_wer",
    "check_wer_asr_path",
    "vllm_c1_original",
    "vllm_c4_original",
    "vllm_c8_original",
    "vllm_c8_prebuild_w4",
    "validate_share_bundle_receiver_smoke",
}

VLLM_LABEL_TO_COMMAND = {
    "vLLM-c1": "vllm_c1_original",
    "vLLM-c4": "vllm_c4_original",
    "vLLM-c8": "vllm_c8_original",
    "vLLM-c8-prebuild-w4": "vllm_c8_prebuild_w4",
}

VLLM_C1_RESULT = Path(
    "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/"
    "benchmark_audio_50_c1_offline_compile/videoamme_results.json"
)


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _num(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("\n", " ").replace("|", "\\|")


def _seconds_from_qps(row: dict[str, Any]) -> float:
    n = _float(row.get("n"))
    qps = _float(row.get("throughput_qps"))
    if n <= 0 or qps <= 0:
        return 0.0
    return n / qps


def _command_ids(repro: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for command in repro.get("commands", []):
        if not isinstance(command, dict):
            continue
        command_id = str(command.get("id") or "")
        if command_id:
            ids.add(command_id)
    return ids


def _command_map(repro: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for command in repro.get("commands", []):
        if not isinstance(command, dict):
            continue
        command_id = str(command.get("id") or "")
        if command_id:
            result[command_id] = command
    return result


def _rel_path(root: Path, value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return text
    return text


def _evidence_exists(root: Path, rel_path: str) -> bool:
    if not rel_path or "*" in rel_path:
        return True
    return (root / rel_path).exists()


def _detail_row_from_sglang(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    timed_s = _seconds_from_qps(row)
    concurrency = _int(row.get("concurrency"))
    result_json = _rel_path(root, row.get("result_json"))
    wer_json = _rel_path(root, row.get("wer_json"))
    return {
        "detail_id": f"sglang_videoamme_c{concurrency}",
        "budget_id": "sglang_videoamme_stress_ci50_c1_c16",
        "runtime": "sglang",
        "scenario": "Video-AMME ci-50",
        "concurrency": concurrency,
        "requests": _int(row.get("n")),
        "throughput_qps": _float(row.get("throughput_qps")),
        "timed_wall_lower_bound_s": timed_s,
        "latency_mean_s": _float(row.get("latency_mean_s")),
        "latency_p95_s": _float(row.get("latency_p95_s")),
        "rtf_mean": _float(row.get("rtf_mean")),
        "rtf_p95": _float(row.get("rtf_p95")),
        "evidence_files": [result_json, wer_json],
    }


def _detail_row_from_synthetic(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    timed_s = _seconds_from_qps(row)
    scenario = str(row.get("scenario") or "")
    concurrency = _int(row.get("concurrency"))
    result_json = _rel_path(root, row.get("result_json"))
    return {
        "detail_id": f"sglang_synthetic_{scenario}_c{concurrency}",
        "budget_id": "sglang_synthetic_short_long_c1_c8",
        "runtime": "sglang",
        "scenario": f"synthetic {scenario}",
        "concurrency": concurrency,
        "requests": _int(row.get("n")),
        "target_chars": _float(row.get("target_chars")),
        "target_words": _float(row.get("target_words")),
        "audio_duration_mean_s": _float(row.get("audio_duration_mean_s")),
        "throughput_qps": _float(row.get("throughput_qps")),
        "timed_wall_lower_bound_s": timed_s,
        "latency_mean_s": _float(row.get("latency_mean_s")),
        "latency_p95_s": _float(row.get("latency_p95_s")),
        "rtf_mean": _float(row.get("rtf_mean")),
        "rtf_p95": _float(row.get("rtf_p95")),
        "evidence_files": [result_json],
    }


def _detail_row_from_vllm(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    label = str(row.get("label") or "")
    result_json = _rel_path(root, row.get("result_json"))
    run_log = _rel_path(root, row.get("run_log"))
    return {
        "detail_id": label.lower().replace("=", "").replace(" ", "_"),
        "budget_id": f"vllm_{label.lower().replace('vllm-', '').replace('-', '_')}",
        "runtime": "vllm",
        "scenario": label,
        "concurrency": _int(row.get("concurrency")),
        "requests": _int(row.get("requests")),
        "timed_wall_lower_bound_s": _float(row.get("wall_time_s")),
        "runner_wall_time_s": _float(row.get("runner_wall_time_s")),
        "engine_wall_time_s": _float(row.get("engine_wall_time_s")),
        "wall_qps": _float(row.get("wall_qps")),
        "runner_overhead_pct_wall": _float(row.get("runner_overhead_pct_wall")),
        "diagnosis": str(row.get("diagnosis") or ""),
        "evidence_files": [result_json, run_log],
    }


def _vllm_c1_from_raw_result(root: Path) -> dict[str, Any]:
    result_json = root / VLLM_C1_RESULT
    payload = _load_json_optional(result_json)
    speed = payload.get("speed", {}) if isinstance(payload.get("speed"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    requests = _int(speed.get("total_requests") or summary.get("total_samples"))
    qps = _float(speed.get("throughput_qps"))
    timed_wall_s = requests / qps if requests > 0 and qps > 0 else 0.0
    return {
        "label": "vLLM-c1",
        "result_json": str(VLLM_C1_RESULT),
        "run_log": (
            "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/"
            "run.log"
        ),
        "concurrency": 1,
        "requests": requests,
        "wall_time_s": timed_wall_s,
        "runner_wall_time_s": timed_wall_s,
        "engine_wall_time_s": timed_wall_s,
        "wall_qps": qps,
        "runner_overhead_pct_wall": None,
        "diagnosis": "raw_result_speed_summary",
        "source": "raw videoamme_results.speed.total_requests / throughput_qps",
    }


def _budget_row(
    *,
    budget_id: str,
    title: str,
    command_ids: list[str],
    runs: int,
    requests: int,
    timed_wall_lower_bound_s: float | None,
    gpu_count: int,
    gpu_scope: str,
    evidence_files: list[str],
    source: str,
    boundary: str,
    detail_ids: list[str],
) -> dict[str, Any]:
    gpu_hours = None
    if timed_wall_lower_bound_s is not None and gpu_count > 0:
        gpu_hours = timed_wall_lower_bound_s * gpu_count / 3600.0
    return {
        "budget_id": budget_id,
        "title": title,
        "command_ids": command_ids,
        "runs": runs,
        "requests": requests,
        "timed_wall_lower_bound_s": timed_wall_lower_bound_s,
        "gpu_count": gpu_count,
        "equivalent_gpu_hours_lower_bound": gpu_hours,
        "gpu_scope": gpu_scope,
        "evidence_files": evidence_files,
        "source": source,
        "boundary": boundary,
        "detail_ids": detail_ids,
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    tables = _load_json_optional(audit_dir / "tables_summary.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    command_ids = _command_ids(repro)
    commands = _command_map(repro)
    table_groups = tables.get("tables", {}) if isinstance(tables.get("tables"), dict) else {}
    sglang_rows = [
        row for row in table_groups.get("sglang_stress", []) if isinstance(row, dict)
    ]
    synthetic_rows = [
        row for row in table_groups.get("synthetic_speech", []) if isinstance(row, dict)
    ]
    vllm_rows = [
        row
        for row in table_groups.get("vllm_admission_diagnosis", [])
        if isinstance(row, dict)
    ]
    vllm_c1_row = _vllm_c1_from_raw_result(root)

    sglang_detail = [_detail_row_from_sglang(root, row) for row in sglang_rows]
    synthetic_detail = [_detail_row_from_synthetic(root, row) for row in synthetic_rows]
    vllm_detail = [
        _detail_row_from_vllm(root, row)
        for row in [vllm_c1_row, *vllm_rows]
    ]
    detail_rows = [*sglang_detail, *synthetic_detail, *vllm_detail]

    sglang_total_s = sum(_float(row.get("timed_wall_lower_bound_s")) for row in sglang_detail)
    synthetic_total_s = sum(
        _float(row.get("timed_wall_lower_bound_s")) for row in synthetic_detail
    )
    sglang_requests = sum(_int(row.get("requests")) for row in sglang_detail)
    synthetic_requests = sum(_int(row.get("requests")) for row in synthetic_detail)

    vllm_by_label = {
        str(row.get("label") or ""): row
        for row in [vllm_c1_row, *vllm_rows]
    }
    rows = [
        _budget_row(
            budget_id="host_receiver_quickcheck_and_evidence_smoke",
            title="收包 quickcheck 与 evidence-query smoke",
            command_ids=["validate_share_bundle_receiver_smoke"],
            runs=1,
            requests=0,
            timed_wall_lower_bound_s=None,
            gpu_count=0,
            gpu_scope="host-only package validation",
            evidence_files=[
                "results/qwen35_report_audit_20260619/share_package_validation.json",
                "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
                "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh",
            ],
            source="receiver quickcheck wrapper and evidence-query smoke; wall time is environment dependent",
            boundary="audit-only packaging check; no model-serving benchmark is rerun",
            detail_ids=[],
        ),
        _budget_row(
            budget_id="full_audit_regeneration",
            title="完整报告审计再生成",
            command_ids=["run_full_audit"],
            runs=1,
            requests=0,
            timed_wall_lower_bound_s=None,
            gpu_count=0,
            gpu_scope="host audit pipeline",
            evidence_files=[
                "results/qwen35_report_audit_20260619/audit_run_summary.json",
                "results/qwen35_report_audit_20260619/final_readiness_audit.json",
                "results/qwen35_report_audit_20260619/manifest.json",
            ],
            source=str(commands.get("run_full_audit", {}).get("rerun_cost") or "audit-only"),
            boundary="audit-only, no benchmark rerun; it regenerates reports and package validation evidence",
            detail_ids=[],
        ),
        _budget_row(
            budget_id="sglang_videoamme_stress_ci50_c1_c16",
            title="SGLang Video-AMME ci-50 c=1/2/4/8/16",
            command_ids=["launch_sglang_optimized", "sglang_videoamme_stress"],
            runs=len(sglang_detail),
            requests=sglang_requests,
            timed_wall_lower_bound_s=sglang_total_s,
            gpu_count=8,
            gpu_scope="8x H20 SGLang serving session",
            evidence_files=[
                "results/qwen35_report_audit_20260619/tables_summary.json",
                "results/qwen35_report_audit_20260619/headline_scorecard.json",
                "results/qwen35_report_audit_20260619/stage_latency_budget.json",
                "results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_profile_skipwer/videoamme_results.json",
            ],
            source="tables_summary.sglang_stress: sum(n / throughput_qps)",
            boundary="不包含 server launch/warmup/WER/ASR；c=16 是 saturation boundary，不是默认 serving point",
            detail_ids=[row["detail_id"] for row in sglang_detail],
        ),
        _budget_row(
            budget_id="sglang_synthetic_short_long_c1_c8",
            title="SGLang short/long synthetic speech c=1/4/8",
            command_ids=["launch_sglang_optimized", "sglang_synthetic_text_to_speech"],
            runs=len(synthetic_detail),
            requests=synthetic_requests,
            timed_wall_lower_bound_s=synthetic_total_s,
            gpu_count=8,
            gpu_scope="8x H20 SGLang serving session",
            evidence_files=[
                "results/qwen35_report_audit_20260619/length_regime_coverage.json",
                "results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv",
                "results/qwen35_synthetic_speech_20260619/*/synthetic_speech_results.json",
            ],
            source="tables_summary.synthetic_speech: sum(n / throughput_qps)",
            boundary="不包含 server launch/warmup/WER/ASR；short=74 chars/12 words, long=944 chars/139 words",
            detail_ids=[row["detail_id"] for row in synthetic_detail],
        ),
        _budget_row(
            budget_id="sglang_wer_recompute",
            title="SGLang saved-output WER/ASR recompute",
            command_ids=["check_wer_asr_path", "sglang_recompute_wer"],
            runs=1,
            requests=sglang_requests,
            timed_wall_lower_bound_s=None,
            gpu_count=1,
            gpu_scope="ASR GPU or cached Whisper path",
            evidence_files=[
                "results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_*/whisper_large_v3_local_wer.json",
                "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md",
            ],
            source="command manifest only; current package stores WER JSON but does not time ASR recompute",
            boundary="WER/ASR is quality validation, not serving throughput; avoid co-running with serving benchmark",
            detail_ids=[],
        ),
    ]

    for label in ["vLLM-c1", "vLLM-c4", "vLLM-c8", "vLLM-c8-prebuild-w4"]:
        source_row = vllm_by_label.get(label, {})
        command_id = VLLM_LABEL_TO_COMMAND[label]
        detail_id = label.lower().replace("=", "").replace(" ", "_")
        rows.append(
            _budget_row(
                budget_id=f"vllm_{label.lower().replace('vllm-', '').replace('-', '_')}",
                title=f"{label} offline benchmark",
                command_ids=[command_id],
                runs=1 if source_row else 0,
                requests=_int(source_row.get("requests")),
                timed_wall_lower_bound_s=_float(source_row.get("wall_time_s")) or None,
                gpu_count=8,
                gpu_scope="8x H20 vLLM offline runner",
                evidence_files=[
                    _rel_path(root, source_row.get("result_json")),
                    _rel_path(root, source_row.get("run_log")),
                    "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
                ],
                source=(
                    "raw videoamme_results.speed.total_requests / throughput_qps"
                    if label == "vLLM-c1"
                    else "tables_summary.vllm_admission_diagnosis.wall_time_s"
                ),
                boundary=(
                    "strict headline slice"
                    if label == "vLLM-c4"
                    else "offline diagnostic; not online serving parity"
                ),
                detail_ids=[detail_id] if source_row else [],
            )
        )

    required_missing = sorted(REQUIRED_COMMAND_IDS - command_ids)
    all_command_refs = [
        command_id
        for row in rows
        for command_id in row["command_ids"]
    ]
    missing_evidence_files = sorted(
        {
            evidence
            for row in rows
            for evidence in row.get("evidence_files", [])
            if evidence and not _evidence_exists(root, evidence)
        }
    )
    timed_rows = [
        row
        for row in rows
        if _float(row.get("timed_wall_lower_bound_s")) > 0
    ]
    total_timed_wall_s = sum(
        _float(row.get("timed_wall_lower_bound_s")) for row in timed_rows
    )
    equivalent_8gpu_gpu_hours = sum(
        _float(row.get("equivalent_gpu_hours_lower_bound")) for row in timed_rows
    )
    vllm_budget_ids = {
        "vllm_c1",
        "vllm_c4",
        "vllm_c8",
        "vllm_c8_prebuild_w4",
    }
    vllm_total_s = sum(
        _float(row.get("timed_wall_lower_bound_s"))
        for row in rows
        if row.get("budget_id") in vllm_budget_ids
    )

    checks = {
        "required_command_ids_present": not required_missing,
        "sglang_stress_rows_cover_c1_c16": {
            1,
            2,
            4,
            8,
            16,
        }.issubset({_int(row.get("concurrency")) for row in sglang_rows}),
        "synthetic_short_long_rows_cover_c1_c8": len(synthetic_rows) >= 6
        and {"short", "long"}.issubset(
            {str(row.get("scenario") or "") for row in synthetic_rows}
        )
        and {1, 4, 8}.issubset({_int(row.get("concurrency")) for row in synthetic_rows}),
        "vllm_timed_rows_cover_baseline_and_diagnostics": {
            "vLLM-c1",
            "vLLM-c4",
            "vLLM-c8",
            "vLLM-c8-prebuild-w4",
        }.issubset(set(vllm_by_label)),
        "timed_lower_bound_positive": total_timed_wall_s > 0
        and equivalent_8gpu_gpu_hours > 0,
        "wer_asr_boundary_explicit": any(
            row["budget_id"] == "sglang_wer_recompute"
            and row["timed_wall_lower_bound_s"] is None
            and "WER/ASR" in row["boundary"]
            for row in rows
        ),
        "audit_rows_explicitly_unmeasured": all(
            row["timed_wall_lower_bound_s"] is None and row["gpu_count"] == 0
            for row in rows[:2]
        ),
        "evidence_files_present_or_globbed": not missing_evidence_files,
    }
    required_failures = [name for name, ok in checks.items() if not ok]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "detail_rows_total": len(detail_rows),
            "timed_rows": len(timed_rows),
            "unmeasured_rows": len(rows) - len(timed_rows),
            "command_refs_total": len(all_command_refs),
            "required_command_ids_total": len(REQUIRED_COMMAND_IDS),
            "required_command_ids_present": not required_missing,
            "missing_command_ids": required_missing,
            "total_timed_benchmark_wall_s": total_timed_wall_s,
            "equivalent_8gpu_timed_lower_bound_gpu_hours": equivalent_8gpu_gpu_hours,
            "sglang_videoamme_timed_wall_s": sglang_total_s,
            "synthetic_timed_wall_s": synthetic_total_s,
            "vllm_timed_wall_s": vllm_total_s,
            "missing_evidence_files": missing_evidence_files,
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "caveat": (
                "timed benchmark lower bound excludes server launch/warmup/WER/ASR/"
                "downloads/cache population/manual review"
            ),
        },
        "checks": checks,
        "rows": rows,
        "detail_rows": detail_rows,
        "command_rerun_costs": {
            command_id: commands.get(command_id, {}).get("rerun_cost")
            for command_id in sorted(REQUIRED_COMMAND_IDS)
        },
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    detail_rows = payload.get("detail_rows", [])

    lines = [
        "# Qwen3.5-Omni 复跑耗时/算力预算",
        "",
        f"生成时间 UTC：`{payload.get('generated_at_utc')}`。",
        f"工作目录：`{payload.get('root')}`。",
        "",
        "这张表给合作方估算复跑排期和算力占用。计时列是从已归档结果反推的",
        "`timed benchmark lower bound`：SGLang 用 `n / throughput_qps`，vLLM 用",
        "`vllm_admission_diagnosis.wall_time_s`。它不包含 server launch/warmup/WER/ASR、",
        "镜像或模型下载、缓存填充、人工审阅和结果打包时间，因此实际预约窗口应在下界上留 buffer。",
        "",
        "## 1. 总览",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| ready | `{summary.get('ready')}` |",
        f"| budget rows | `{summary.get('rows_total')}` |",
        f"| measured detail rows | `{summary.get('detail_rows_total')}` |",
        f"| timed rows | `{summary.get('timed_rows')}` |",
        f"| unmeasured/audit rows | `{summary.get('unmeasured_rows')}` |",
        f"| command refs | `{summary.get('command_refs_total')}` |",
        f"| required command ids present | `{summary.get('required_command_ids_present')}` |",
        f"| total timed lower bound | `{_num(summary.get('total_timed_benchmark_wall_s'), 1, 's')}` |",
        f"| 8-GPU equivalent lower bound | `{_num(summary.get('equivalent_8gpu_timed_lower_bound_gpu_hours'), 3, ' GPUh')}` |",
        f"| SGLang Video-AMME lower bound | `{_num(summary.get('sglang_videoamme_timed_wall_s'), 1, 's')}` |",
        f"| SGLang synthetic lower bound | `{_num(summary.get('synthetic_timed_wall_s'), 1, 's')}` |",
        f"| vLLM lower bound | `{_num(summary.get('vllm_timed_wall_s'), 1, 's')}` |",
        "",
        "## 2. 预算矩阵",
        "",
        "| Budget | Command IDs | Runs / requests | Timed lower bound | 8-GPU lower bound | GPU scope | Evidence | Boundary |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        timed = row.get("timed_wall_lower_bound_s")
        gpu_hours = row.get("equivalent_gpu_hours_lower_bound")
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("title")),
                    _cell(", ".join(f"`{item}`" for item in row.get("command_ids", []))),
                    _cell(f"{row.get('runs')} / {row.get('requests')}"),
                    _cell(_num(timed, 1, "s") if timed is not None else "not timed"),
                    _cell(_num(gpu_hours, 3, " GPUh") if gpu_hours is not None else "n/a"),
                    _cell(row.get("gpu_scope")),
                    _cell("; ".join(f"`{item}`" for item in row.get("evidence_files", [])[:3])),
                    _cell(row.get("boundary")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 3. 计时明细",
            "",
            "| Detail | Runtime | Scenario | C | Requests | Timed lower bound | QPS / wall QPS | Lat p95 / RTF p95 | Evidence |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in detail_rows:
        qps = row.get("throughput_qps", row.get("wall_qps"))
        lat_rtf = (
            f"{_num(row.get('latency_p95_s'), 3, 's')} / {_num(row.get('rtf_p95'), 4)}"
            if row.get("runtime") == "sglang"
            else f"runner overhead {_num(row.get('runner_overhead_pct_wall'), 1, '%')}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row.get("detail_id")),
                    _cell(row.get("runtime")),
                    _cell(row.get("scenario")),
                    _cell(row.get("concurrency")),
                    _cell(row.get("requests")),
                    _cell(_num(row.get("timed_wall_lower_bound_s"), 1, "s")),
                    _cell(_num(qps, 3)),
                    _cell(lat_rtf),
                    _cell("; ".join(f"`{item}`" for item in row.get("evidence_files", [])[:2])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 4. 查询命令",
            "",
            "```bash",
            "jq '.summary' results/qwen35_report_audit_20260619/rerun_time_budget.json",
            "jq '.rows[] | {budget_id, command_ids, timed_wall_lower_bound_s, boundary}' results/qwen35_report_audit_20260619/rerun_time_budget.json",
            "jq '.detail_rows[] | select(.runtime==\"sglang\") | {detail_id, concurrency, timed_wall_lower_bound_s, throughput_qps}' results/qwen35_report_audit_20260619/rerun_time_budget.json",
            "```",
            "",
            "## 5. 使用边界",
            "",
            "- `rerun_time_budget.json` 是排期预算和复跑解释证据，不替代 `rerun_acceptance_contract.json` 的 34 项返回证据硬合同。",
            "- SGLang 预算默认共用同一个 8 卡 serving session；若每个压力点都冷启动，实际 wall time 会高于表中下界。",
            "- vLLM c=8 与 c=8 prebuild w4 是 offline diagnostic；升级成 online parity headline 前仍要按 `qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` 补在线入口证据。",
            "- WER/ASR 复算没有纳入 timed benchmark lower bound；合作方如果要求重新跑 WER，请单独预约 ASR GPU 或确认本地 Whisper cache。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni rerun time and compute budget."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_payload(root)
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(payload), encoding="utf-8")
    _save_json(payload, json_output)
    print(f"Rerun time budget written: {output}")
    print(f"Rerun time budget JSON written: {json_output}")
    if payload.get("summary", {}).get("required_failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
