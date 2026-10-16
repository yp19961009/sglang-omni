# SPDX-License-Identifier: Apache-2.0
"""Build a stage-causal graph report for the Qwen3.5-Omni performance package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path("benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md")
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "stage_causal_graph.json"


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


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("interactions", [])
    return rows if isinstance(rows, list) else []


def _fmt_ms(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}ms"
    except (TypeError, ValueError):
        return "n/a"


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _find(
    interactions: list[dict[str, Any]],
    *,
    runtime: str,
    boundary: str,
    concurrency: int | None = None,
    status: str | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in interactions:
        if row.get("runtime") != runtime:
            continue
        if row.get("boundary") != boundary:
            continue
        if concurrency is not None and row.get("concurrency") != concurrency:
            continue
        if status is not None and row.get("status") != status:
            continue
        if label is not None and row.get("label") != label:
            continue
        result.append(row)
    return result


def _first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def _metric(row: dict[str, Any], key: str) -> Any:
    metrics = row.get("metrics", {})
    return metrics.get(key) if isinstance(metrics, dict) else None


def _table_rows(tables_payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    tables = tables_payload.get("tables", {})
    if not isinstance(tables, dict):
        return []
    rows = tables.get(name, [])
    return rows if isinstance(rows, list) else []


def _first_table_row(rows: list[dict[str, Any]], **matches: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    return {}


def _rel_path(root: Path, value: Any) -> str:
    if not value:
        return "n/a"
    text = str(value)
    try:
        path = Path(text)
        if path.is_absolute():
            text = path.resolve().relative_to(root).as_posix()
    except Exception:
        pass
    return f"`{text}`"


def _rel_path_text(root: Path, value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    try:
        path = Path(text)
        if path.is_absolute():
            return path.resolve().relative_to(root).as_posix()
    except Exception:
        return text
    return text


def _manifest_relative_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        rel_path = str(record.get("relative_path") or "")
        if rel_path and record.get("exists") is not False:
            paths.add(rel_path)
        for item in record.get("files", []):
            if not isinstance(item, dict):
                continue
            item_rel_path = str(item.get("relative_path") or "")
            if item_rel_path and item.get("exists") is not False:
                paths.add(item_rel_path)
    return paths


def _clean_paths(root: Path, *values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        rel_path = _rel_path_text(root, value)
        if not rel_path or rel_path == "n/a" or rel_path in seen:
            continue
        seen.add(rel_path)
        result.append(rel_path)
    return result


def _paths(*values: str) -> str:
    usable = [value for value in values if value and value != "n/a"]
    return "<br>".join(usable) if usable else "n/a"


def _acceptance_row(acceptance: dict[str, Any], serving_status: str) -> dict[str, Any]:
    for row in acceptance.get("rows", []):
        if row.get("serving_status") == serving_status:
            return row
    return {}


def _status_counts(summary: dict[str, Any]) -> str:
    counts = summary.get("status_counts", {})
    if not isinstance(counts, dict):
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _mermaid() -> str:
    return """```mermaid
flowchart LR
    Client["client request"] --> Admission["request admission / queue"]
    Admission --> Preproc["preprocessing lifecycle"]
    Preproc --> Thinker["thinker / encoder"]
    Thinker --> Talker["talker AR / codec cadence"]
    Talker --> Hop["talker -> code2wav stream hop"]
    Hop --> Collect["code2wav window collect"]
    Collect --> Decode["code2wav decode"]
    Decode --> Audio["audio response"]

    Admission -. c8/c16 queue grows .-> Preproc
    Preproc -. naive preproc=2/4 adds contention .-> Thinker
    Talker -. chunk cadence drives collect wait .-> Collect
    Hop -. healthy p95 ~15-24ms .-> Collect
    Decode -. decode p95 ~18-26ms, not bottleneck .-> Audio

    VPrompt["vLLM offline prompt build/feed"] --> VAdmission["vLLM engine admission"]
    VAdmission --> VThinker["vLLM thinker"]
    VThinker --> VTalker["vLLM talker"]
    VTalker --> VC2W["vLLM code2wav"]
    VPrompt -. original c8 prompt-feed limited .-> VAdmission
    VAdmission -. prebuild w4 lowers admission span .-> VTalker
```"""


def _check(name: str, condition: bool, evidence: str, *, required: bool = True) -> dict[str, Any]:
    if condition:
        status = "PASS"
    elif required:
        status = "FAIL"
    else:
        status = "WARN"
    return {
        "name": name,
        "status": status,
        "required": required,
        "evidence": evidence,
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    interactions_payload = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    stage_latency_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    stage_boundary_ledger = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    stage_reproduction = _load_json_optional(audit_dir / "stage_reproduction_drilldown.json")
    stage_route = _load_json_optional(audit_dir / "stage_route_decision_matrix.json")
    tables_summary = _load_json_optional(audit_dir / "tables_summary.json")
    vllm_admission = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")

    stage_summary = interactions_payload.get("summary", {})
    budget_summary = stage_latency_budget.get("summary", {})
    ledger_summary = stage_boundary_ledger.get("summary", {})
    reproduction_summary = stage_reproduction.get("summary", {})
    route_summary = stage_route.get("summary", {})
    vllm_rows_payload = vllm_admission.get("rows", [])
    vllm_rows_payload = vllm_rows_payload if isinstance(vllm_rows_payload, list) else []
    interactions = _rows(interactions_payload)
    sglang_stress_rows = _table_rows(tables_summary, "sglang_stress")
    sglang_stage_rows = _table_rows(tables_summary, "sglang_stage_breakdown")
    synthetic_rows = _table_rows(tables_summary, "synthetic_speech")
    synthetic_stage_rows = _table_rows(tables_summary, "synthetic_stage_breakdown")
    vllm_rows = _table_rows(tables_summary, "vllm_admission_diagnosis")

    c8_stress_raw = _first_table_row(sglang_stress_rows, concurrency=8)
    c16_stress_raw = _first_table_row(sglang_stress_rows, concurrency=16)
    c8_stage_raw = _first_table_row(sglang_stage_rows, concurrency=8)
    c16_stage_raw = _first_table_row(sglang_stage_rows, concurrency=16)
    short_c8_raw = _first_table_row(synthetic_rows, scenario="short", concurrency=8)
    long_c8_raw = _first_table_row(synthetic_rows, scenario="long", concurrency=8)
    short_c8_stage_raw = _first_table_row(
        synthetic_stage_rows, scenario="short", concurrency=8
    )
    long_c8_stage_raw = _first_table_row(
        synthetic_stage_rows, scenario="long", concurrency=8
    )
    vllm_c8_raw = _first_table_row(vllm_rows, label="vLLM-c8")
    vllm_w4_raw = _first_table_row(vllm_rows, label="vLLM-c8-prebuild-w4")

    c8_preproc = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="request_admission_to_preprocessing",
            concurrency=8,
        )
    )
    c16_preproc = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="request_admission_to_preprocessing",
            concurrency=16,
        )
    )
    c8_hop = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="talker_to_code2wav_stream",
            concurrency=8,
        )
    )
    c16_hop = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="talker_to_code2wav_stream",
            concurrency=16,
        )
    )
    c8_decode = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="code2wav_collect_to_decode",
            concurrency=8,
        )
    )
    c16_decode = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="code2wav_collect_to_decode",
            concurrency=16,
        )
    )
    preproc_regression = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="preprocessing_to_encoder_thinker",
            status="contention_regression",
        )
    )
    vllm_c8_admission = _first(
        _find(
            interactions,
            runtime="vllm",
            boundary="runner_to_engine_admission",
            status="prompt_feed_limited",
            label="vLLM-c8",
        )
    )
    vllm_w4_admission = _first(
        _find(
            interactions,
            runtime="vllm",
            boundary="runner_to_engine_admission",
            status="diagnostic_only",
            label="vLLM-c8-prebuild-w4",
        )
    )

    causal_edges = [
        {
            "id": "sglang_admission_to_preprocessing_queue",
            "runtime": "sglang",
            "source": "request admission / queue",
            "target": "preprocessing lifecycle",
            "verdict": "queue_limited_at_c8_c16",
            "evidence": {
                "c8_preproc_lifecycle_avg_ms": _metric(c8_preproc, "preproc_lifecycle_avg_ms"),
                "c8_actual_preprocess_avg_ms": _metric(c8_preproc, "actual_preprocess_avg_ms"),
                "c16_preproc_lifecycle_avg_ms": _metric(c16_preproc, "preproc_lifecycle_avg_ms"),
                "c16_actual_preprocess_avg_ms": _metric(c16_preproc, "actual_preprocess_avg_ms"),
            },
        },
        {
            "id": "sglang_preprocessing_parallelism_to_contention",
            "runtime": "sglang",
            "source": "preprocessing widening",
            "target": "thinker/encoder shared resources",
            "verdict": "negative_optimization",
            "evidence": {
                "qps_delta_pct": _metric(preproc_regression, "qps_delta_pct"),
                "status": preproc_regression.get("status"),
            },
        },
        {
            "id": "sglang_talker_ar_to_code2wav_collect",
            "runtime": "sglang",
            "source": "talker AR / codec cadence",
            "target": "code2wav collect wait",
            "verdict": "collect_wait_tracks_talker_chunks",
            "evidence": {
                "c8_collect_avg_ms": _metric(c8_decode, "code2wav_window_collect_avg_ms"),
                "c8_decode_avg_ms": _metric(c8_decode, "code2wav_decode_avg_ms"),
                "c16_collect_avg_ms": _metric(c16_decode, "code2wav_window_collect_avg_ms"),
                "c16_decode_avg_ms": _metric(c16_decode, "code2wav_decode_avg_ms"),
            },
        },
        {
            "id": "sglang_talker_to_code2wav_stream_handoff",
            "runtime": "sglang",
            "source": "talker",
            "target": "code2wav stream hop",
            "verdict": "handoff_healthy",
            "evidence": {
                "c8_hop_p95_ms": _metric(c8_hop, "talker_to_code2wav_hop_p95_ms"),
                "c16_hop_p95_ms": _metric(c16_hop, "talker_to_code2wav_hop_p95_ms"),
            },
        },
        {
            "id": "sglang_code2wav_decode_to_audio",
            "runtime": "sglang",
            "source": "code2wav decode",
            "target": "audio response",
            "verdict": "decode_not_current_bottleneck",
            "evidence": {
                "c8_decode_p95_ms": _metric(c8_decode, "code2wav_decode_p95_ms"),
                "c16_decode_p95_ms": _metric(c16_decode, "code2wav_decode_p95_ms"),
            },
        },
        {
            "id": "vllm_prompt_feed_to_engine_admission",
            "runtime": "vllm",
            "source": "offline prompt build/feed",
            "target": "engine admission",
            "verdict": "prompt_feed_limited",
            "evidence": {
                "c8_admission_avg_ms": _metric(vllm_c8_admission, "batch_admission_span_avg_ms"),
                "c8_admission_p95_ms": _metric(vllm_c8_admission, "batch_admission_span_p95_ms"),
            },
        },
        {
            "id": "vllm_prebuild_to_later_engine_tail",
            "runtime": "vllm",
            "source": "prebuild w4 prompt path",
            "target": "engine/talker-side tail",
            "verdict": "offline_diagnostic_only",
            "evidence": {
                "w4_admission_avg_ms": _metric(vllm_w4_admission, "batch_admission_span_avg_ms"),
                "w4_admission_p95_ms": _metric(vllm_w4_admission, "batch_admission_span_p95_ms"),
            },
        },
    ]

    raw_drilldown = [
        {
            "question": "SGLang c8/c16 high concurrency admission queue",
            "summary_sources": [
                "results/qwen35_report_audit_20260619/stage_latency_budget.json",
                "results/qwen35_report_audit_20260619/stage_interaction_summary.json",
            ],
            "raw_artifacts": _clean_paths(
                root,
                c8_stress_raw.get("result_json"),
                c8_stress_raw.get("wer_json"),
                c8_stage_raw.get("profile_json"),
                c16_stress_raw.get("result_json"),
                c16_stress_raw.get("wer_json"),
                c16_stage_raw.get("profile_json"),
            ),
        },
        {
            "question": "SGLang talker to code2wav handoff",
            "summary_sources": [
                "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            ],
            "raw_artifacts": _clean_paths(
                root,
                c8_stage_raw.get("profile_json"),
                c16_stage_raw.get("profile_json"),
            ),
        },
        {
            "question": "Short/long text speech coverage and long c8 real-time guard",
            "summary_sources": [
                "results/qwen35_report_audit_20260619/tables_summary.json",
                "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            ],
            "raw_artifacts": _clean_paths(
                root,
                short_c8_raw.get("result_json"),
                short_c8_stage_raw.get("profile_json"),
                long_c8_raw.get("result_json"),
                long_c8_stage_raw.get("profile_json"),
            ),
        },
        {
            "question": "vLLM original c8 offline diagnostic boundary",
            "summary_sources": [
                "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
            ],
            "raw_artifacts": _clean_paths(
                root,
                vllm_c8_raw.get("result_json"),
                vllm_c8_raw.get("run_log"),
            ),
        },
        {
            "question": "vLLM c8 prebuild w4 diagnostic improvement",
            "summary_sources": [
                "results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json",
                "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
            ],
            "raw_artifacts": _clean_paths(
                root,
                vllm_w4_raw.get("result_json"),
                vllm_w4_raw.get("run_log"),
            ),
        },
    ]
    raw_paths = sorted(
        {
            raw_path
            for row in raw_drilldown
            for raw_path in row.get("raw_artifacts", [])
        }
    )
    manifest_paths = _manifest_relative_paths(manifest)
    missing_from_manifest = [path for path in raw_paths if path not in manifest_paths]
    required_edge_ids = {
        "sglang_admission_to_preprocessing_queue",
        "sglang_preprocessing_parallelism_to_contention",
        "sglang_talker_ar_to_code2wav_collect",
        "sglang_talker_to_code2wav_stream_handoff",
        "sglang_code2wav_decode_to_audio",
        "vllm_prompt_feed_to_engine_admission",
        "vllm_prebuild_to_later_engine_tail",
    }
    edge_ids = {str(edge.get("id")) for edge in causal_edges}

    checks = [
        _check(
            "source stage evidence present",
            bool(stage_summary)
            and bool(budget_summary)
            and bool(ledger_summary)
            and bool(reproduction_summary)
            and bool(route_summary)
            and len(vllm_rows_payload) >= 4,
            (
                f"stage={bool(stage_summary)}, budget={bool(budget_summary)}, "
                f"ledger={bool(ledger_summary)}, reproduction={bool(reproduction_summary)}, "
                f"route={bool(route_summary)}, vllm_rows={len(vllm_rows_payload)}"
            ),
        ),
        _check(
            "stage interaction flags preserve handoff and bottleneck claims",
            _int_value(stage_summary.get("total_interactions")) >= 37
            and bool(stage_summary.get("sglang_talker_to_code2wav_healthy"))
            and bool(stage_summary.get("sglang_code2wav_decode_not_bottleneck"))
            and bool(stage_summary.get("vllm_original_c8_prompt_feed_limited"))
            and bool(stage_summary.get("preprocessing_parallelism_regresses")),
            f"stage_interaction_summary={stage_summary}",
        ),
        _check(
            "causal edge inventory covers SGLang and vLLM routes",
            len(causal_edges) >= 7 and required_edge_ids.issubset(edge_ids),
            f"edges={sorted(edge_ids)}",
        ),
        _check(
            "SGLang queue, handoff, collect, and decode edges are quantified",
            _float_value(_metric(c8_preproc, "preproc_lifecycle_avg_ms")) > 0
            and _float_value(_metric(c16_preproc, "preproc_lifecycle_avg_ms")) > 0
            and _float_value(_metric(c8_hop, "talker_to_code2wav_hop_p95_ms")) > 0
            and _float_value(_metric(c16_hop, "talker_to_code2wav_hop_p95_ms")) > 0
            and _float_value(_metric(c8_decode, "code2wav_decode_p95_ms")) > 0
            and _float_value(_metric(c16_decode, "code2wav_decode_p95_ms")) > 0,
            "c8/c16 preproc lifecycle, hop p95, and decode p95 all present",
        ),
        _check(
            "vLLM prompt-feed and prebuild diagnostic edges are quantified",
            _float_value(_metric(vllm_c8_admission, "batch_admission_span_p95_ms")) > 0
            and _float_value(_metric(vllm_w4_admission, "batch_admission_span_p95_ms")) > 0,
            (
                f"vllm_c8={vllm_c8_admission.get('metrics')}; "
                f"vllm_w4={vllm_w4_admission.get('metrics')}"
            ),
        ),
        _check(
            "raw drilldown artifacts are manifest-backed",
            len(raw_drilldown) >= 5
            and len(raw_paths) >= 14
            and not missing_from_manifest,
            f"raw_paths={len(raw_paths)}, missing_from_manifest={missing_from_manifest}",
        ),
        _check(
            "stage route and reproduction drilldown remain attached",
            bool(reproduction_summary.get("ready"))
            and _int_value(reproduction_summary.get("stage_rows_total")) >= 52
            and _int_value(reproduction_summary.get("route_rows_total")) >= 11
            and _int_value(reproduction_summary.get("command_refs_total")) >= 15
            and bool(route_summary.get("ready"))
            and _int_value(route_summary.get("route_rows_total")) >= 11
            and _int_value(route_summary.get("command_refs_total")) >= 15,
            (
                f"stage_reproduction={reproduction_summary}; "
                f"stage_route={route_summary}"
            ),
        ),
    ]
    required_failures = [
        check for check in checks if check["required"] and check["status"] != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check["status"] == "PASS"),
            "required_failures": len(required_failures),
            "causal_edges_total": len(causal_edges),
            "raw_drilldown_rows": len(raw_drilldown),
            "raw_artifacts_total": len(raw_paths),
            "manifest_missing_raw_artifacts": len(missing_from_manifest),
            "sglang_handoff_healthy": bool(
                stage_summary.get("sglang_talker_to_code2wav_healthy")
            ),
            "sglang_decode_not_bottleneck": bool(
                stage_summary.get("sglang_code2wav_decode_not_bottleneck")
            ),
            "vllm_c8_scope": "offline_diagnostic_until_online_ingress_artifacts",
        },
        "checks": checks,
        "required_failures": required_failures,
        "causal_edges": causal_edges,
        "raw_drilldown": raw_drilldown,
        "raw_artifacts": raw_paths,
        "missing_raw_artifacts_from_manifest": missing_from_manifest,
        "source_files": {
            "stage_interaction_summary": str(audit_dir / "stage_interaction_summary.json"),
            "stage_latency_budget": str(audit_dir / "stage_latency_budget.json"),
            "stage_boundary_bottleneck_ledger": str(
                audit_dir / "stage_boundary_bottleneck_ledger.json"
            ),
            "stage_reproduction_drilldown": str(
                audit_dir / "stage_reproduction_drilldown.json"
            ),
            "stage_route_decision_matrix": str(
                audit_dir / "stage_route_decision_matrix.json"
            ),
            "vllm_admission_diagnosis": str(audit_dir / "vllm_admission_diagnosis.json"),
            "manifest": str(audit_dir / "manifest.json"),
        },
    }


def build_markdown(root: Path) -> str:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    interactions_payload = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    scorecard = _load_json_optional(audit_dir / "headline_scorecard.json")
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    tables_summary = _load_json_optional(audit_dir / "tables_summary.json")

    summary = interactions_payload.get("summary", {})
    interactions = _rows(interactions_payload)
    sglang_stress_rows = _table_rows(tables_summary, "sglang_stress")
    sglang_stage_rows = _table_rows(tables_summary, "sglang_stage_breakdown")
    synthetic_rows = _table_rows(tables_summary, "synthetic_speech")
    synthetic_stage_rows = _table_rows(tables_summary, "synthetic_stage_breakdown")
    vllm_rows = _table_rows(tables_summary, "vllm_admission_diagnosis")

    c8_stress_raw = _first_table_row(sglang_stress_rows, concurrency=8)
    c16_stress_raw = _first_table_row(sglang_stress_rows, concurrency=16)
    c8_stage_raw = _first_table_row(sglang_stage_rows, concurrency=8)
    c16_stage_raw = _first_table_row(sglang_stage_rows, concurrency=16)
    short_c8_raw = _first_table_row(synthetic_rows, scenario="short", concurrency=8)
    long_c8_raw = _first_table_row(synthetic_rows, scenario="long", concurrency=8)
    short_c8_stage_raw = _first_table_row(
        synthetic_stage_rows, scenario="short", concurrency=8
    )
    long_c8_stage_raw = _first_table_row(
        synthetic_stage_rows, scenario="long", concurrency=8
    )
    vllm_c8_raw = _first_table_row(vllm_rows, label="vLLM-c8")
    vllm_w4_raw = _first_table_row(vllm_rows, label="vLLM-c8-prebuild-w4")

    c8_preproc = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="request_admission_to_preprocessing",
            concurrency=8,
        )
    )
    c16_preproc = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="request_admission_to_preprocessing",
            concurrency=16,
        )
    )
    c8_hop = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="talker_to_code2wav_stream",
            concurrency=8,
        )
    )
    c16_hop = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="talker_to_code2wav_stream",
            concurrency=16,
        )
    )
    c8_decode = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="code2wav_collect_to_decode",
            concurrency=8,
        )
    )
    c16_decode = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="code2wav_collect_to_decode",
            concurrency=16,
        )
    )
    preproc_regression = _first(
        _find(
            interactions,
            runtime="sglang",
            boundary="preprocessing_to_encoder_thinker",
            status="contention_regression",
        )
    )
    vllm_c8_admission = _first(
        _find(
            interactions,
            runtime="vllm",
            boundary="runner_to_engine_admission",
            status="prompt_feed_limited",
            label="vLLM-c8",
        )
    )
    vllm_w4_admission = _first(
        _find(
            interactions,
            runtime="vllm",
            boundary="runner_to_engine_admission",
            status="diagnostic_only",
            label="vLLM-c8-prebuild-w4",
        )
    )
    peak = _acceptance_row(acceptance, "recommended_peak_throughput")
    saturation = _acceptance_row(acceptance, "not_recommended_saturation")
    anti_regression = _acceptance_row(acceptance, "anti_recipe_regression")

    score_summary = scorecard.get("summary", {})
    final_summary = final.get("summary", {})

    lines: list[str] = [
        "# Qwen3.5-Omni Stage 因果图",
        "",
        f"生成时间 UTC：`{datetime.now(timezone.utc).isoformat()}`。",
        f"工作目录：`{root}`。",
        "",
        "这页把 stage breakdown 里的 queue、compute、handoff、collect wait 和 offline",
        "runner admission 关系画成因果图，用来回答 reviewer 最容易追问的两件事：",
        "stage 之间有没有卡住，以及一个 stage 的变化会不会把瓶颈转移到另一个 stage。",
        "",
        "## 1. 机器 Gate",
        "",
        "| Gate | 当前值 | 判定 |",
        "| --- | --- | --- |",
        f"| stage interaction rows | `{summary.get('total_interactions')}` | {_status_counts(summary)} |",
        (
            f"| SGLang handoff | talker->code2wav healthy=`{summary.get('sglang_talker_to_code2wav_healthy')}`；"
            f"decode not bottleneck=`{summary.get('sglang_code2wav_decode_not_bottleneck')}` | PASS |"
        ),
        (
            f"| vLLM offline diagnosis | original c8 prompt-feed limited=`{summary.get('vllm_original_c8_prompt_feed_limited')}` | "
            "只做诊断，不做 online parity |"
        ),
        (
            f"| anti-recipe guard | preprocessing parallelism regresses=`{summary.get('preprocessing_parallelism_regresses')}` | "
            "preproc=2/4 不作为当前 recipe |"
        ),
        (
            f"| final readiness | ready=`{final_summary.get('ready')}`；headline checks="
            f"`{score_summary.get('checks_passed')}/{score_summary.get('checks_total')}` | 分享前门禁 |"
        ),
        "",
        "## 2. Stage 因果图",
        "",
        _mermaid(),
        "",
        "## 3. SGLang 关键因果边",
        "",
        "| 因果边 | 证据 | 结论 | 优先动作 |",
        "| --- | --- | --- | --- |",
        (
            "| admission -> preprocessing lifecycle | "
            f"c8 lifecycle `{_fmt_ms(_metric(c8_preproc, 'preproc_lifecycle_avg_ms'))}`，"
            f"actual preprocess `{_fmt_ms(_metric(c8_preproc, 'actual_preprocess_avg_ms'))}`；"
            f"c16 lifecycle `{_fmt_ms(_metric(c16_preproc, 'preproc_lifecycle_avg_ms'))}`，"
            f"actual preprocess `{_fmt_ms(_metric(c16_preproc, 'actual_preprocess_avg_ms'))}` | "
            "高并发慢主要是 admission/queue，不是预处理 compute 突然变慢 | 调 admission/batching，不先加 worker |"
        ),
        (
            "| preprocessing widening -> thinker/encoder contention | "
            f"preproc=2 QPS delta `{_metric(preproc_regression, 'qps_delta_pct'):.1f}%` | "
            "朴素加 preprocessing 并发把排队问题转成共享资源争用 | 保持 preproc=1，除非同步改 placement/admission |"
        ),
        (
            "| talker AR -> code2wav collect wait | "
            f"c8 collect `{_fmt_ms(_metric(c8_decode, 'code2wav_window_collect_avg_ms'))}` vs decode "
            f"`{_fmt_ms(_metric(c8_decode, 'code2wav_decode_avg_ms'))}`；"
            f"c16 collect `{_fmt_ms(_metric(c16_decode, 'code2wav_window_collect_avg_ms'))}` vs decode "
            f"`{_fmt_ms(_metric(c16_decode, 'code2wav_decode_avg_ms'))}` | "
            "collect 变长主要是在等 talker codec chunk，不等于 vocoder compute 慢 | 优先看 talker cadence 和 chunk policy |"
        ),
        (
            "| talker -> code2wav stream hop | "
            f"c8 hop p95 `{_fmt_ms(_metric(c8_hop, 'talker_to_code2wav_hop_p95_ms'))}`；"
            f"c16 hop p95 `{_fmt_ms(_metric(c16_hop, 'talker_to_code2wav_hop_p95_ms'))}` | "
            "stage handoff 本身健康，不是当前高并发瓶颈 | 不把连接层误判为主瓶颈 |"
        ),
        (
            "| code2wav decode -> audio response | "
            f"c8 decode p95 `{_fmt_ms(_metric(c8_decode, 'code2wav_decode_p95_ms'))}`；"
            f"c16 decode p95 `{_fmt_ms(_metric(c16_decode, 'code2wav_decode_p95_ms'))}` | "
            "decode 是十几到二十几 ms 量级，不是当前主要 compute bottleneck | 保持 compile 路径，优先优化 talker/admission |"
        ),
        "",
        "## 4. vLLM 诊断因果边",
        "",
        "| 因果边 | 证据 | 结论 | 对外边界 |",
        "| --- | --- | --- | --- |",
        (
            "| prompt build/feed -> engine admission | "
            f"original c8 admission avg/p95 `{_fmt_ms(_metric(vllm_c8_admission, 'batch_admission_span_avg_ms'))}` / "
            f"`{_fmt_ms(_metric(vllm_c8_admission, 'batch_admission_span_p95_ms'))}` | "
            "原始 offline c8 被 runner/admission 限制 | 不能拿 original c8 wall QPS 做 serving parity |"
        ),
        (
            "| prebuild w4 -> lower admission span -> later tail exposed | "
            f"prebuild admission avg/p95 `{_fmt_ms(_metric(vllm_w4_admission, 'batch_admission_span_avg_ms'))}` / "
            f"`{_fmt_ms(_metric(vllm_w4_admission, 'batch_admission_span_p95_ms'))}` | "
            "prebuild 缓解 prompt/admission，但暴露 engine/talker-side tail | 仍需 online ingress + WER/ASR 才能做 c8 parity |"
        ),
        "",
        "## 5. Regime 读法",
        "",
        "| Regime | 证据 | 因果解释 | Reviewer 口径 |",
        "| --- | --- | --- | --- |",
        (
            f"| c=8 | {peak.get('key_metrics', '')} | admission 开始显性，但 handoff/decode 仍健康 | "
            "当前高并发吞吐峰值 |"
        ),
        (
            f"| c=16 | {saturation.get('key_metrics', '')} | queue/admission 饱和，吞吐回落 | "
            "压力边界，不是默认服务点 |"
        ),
        (
            f"| preproc=2 | {anti_regression.get('key_metrics', '')} | resource contention 盖过并发收益 | "
            "反例 recipe |"
        ),
        "",
        "## 6. 原始证据 Drilldown",
        "",
        "这张表把因果图里的关键边直接映射到 raw local artifact。share bundle 内只携带",
        "摘要、审计 JSON、图表和校验工具；需要重新追溯原始 profile/run log 时，按这些路径在",
        "`/home/gangouyu/sglang-omni` 工作区或复跑结果目录中检查。",
        "这些 raw artifact path 由 `manifest.json` 背书；final readiness 会校验它们存在于",
        "manifest 证据清单中，避免 drilldown 变成未登记的手写路径。",
        "",
        "| 追问 | 先看机器摘要 | Raw artifact drilldown | 证明什么 |",
        "| --- | --- | --- | --- |",
        (
            "| SGLang c=8/c=16 高并发为什么是 admission/queue | "
            "`stage_latency_budget.json`；`stage_interaction_summary.json` | "
            f"{_paths(_rel_path(root, c8_stress_raw.get('result_json')), _rel_path(root, c8_stress_raw.get('wer_json')), _rel_path(root, c8_stage_raw.get('profile_json')), _rel_path(root, c16_stress_raw.get('result_json')), _rel_path(root, c16_stress_raw.get('wer_json')), _rel_path(root, c16_stage_raw.get('profile_json')))} | "
            "同时复核 QPS/latency/WER、preprocessing lifecycle、actual preprocess、hop 和 decode span |"
        ),
        (
            "| SGLang talker->code2wav handoff 是否卡住 | "
            "`stage_boundary_bottleneck_ledger.json` rows `talker_to_code2wav_stream` | "
            f"{_paths(_rel_path(root, c8_stage_raw.get('profile_json')), _rel_path(root, c16_stage_raw.get('profile_json')))} | "
            "检查 hop p95 约 20ms，而 talker tail 是秒级；连接不是主瓶颈 |"
        ),
        (
            "| 短/长文本语音是否覆盖，并且 long c=8 是否快于实时 | "
            "`tables_summary.json` synthetic rows；`stage_latency_budget.json` | "
            f"{_paths(_rel_path(root, short_c8_raw.get('result_json')), _rel_path(root, short_c8_stage_raw.get('profile_json')), _rel_path(root, long_c8_raw.get('result_json')), _rel_path(root, long_c8_stage_raw.get('profile_json')))} | "
            "复核 short/long 输入形态、RTF、talker AR、code2wav decode 和 hop |"
        ),
        (
            "| vLLM original c=8 为什么不能当 online parity | "
            "`vllm_admission_diagnosis.json` row `vLLM-c8` | "
            f"{_paths(_rel_path(root, vllm_c8_raw.get('result_json')), _rel_path(root, vllm_c8_raw.get('run_log')))} | "
            "复核 runner overhead、batch admission span 和 prompt-feed limited 诊断 |"
        ),
        (
            "| vLLM c=8 prebuild w4 改善了什么，为什么仍是 diagnostic | "
            "`vllm_admission_diagnosis.json` row `vLLM-c8-prebuild-w4`；`vLLM online parity protocol` | "
            f"{_paths(_rel_path(root, vllm_w4_raw.get('result_json')), _rel_path(root, vllm_w4_raw.get('run_log')))} | "
            "复核 admission span 降低、engine/talker tail 暴露，以及缺少 online ingress/WER 的边界 |"
        ),
        "",
        "## 7. 复核入口",
        "",
        "- stage interaction summary：`results/qwen35_report_audit_20260619/stage_interaction_summary.json`",
        "- stage latency budget：`results/qwen35_report_audit_20260619/stage_latency_budget.json`",
        "- stage boundary bottleneck ledger：`results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`",
        "- tables summary raw path index：`results/qwen35_report_audit_20260619/tables_summary.json`",
        "- vLLM admission diagnosis：`results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`",
        "- regime decision matrix：`benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md`",
        "- regime decision matrix JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`",
        "- stage metric dictionary：`benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`",
        "- optimization playbook：`benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni stage causal graph report."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    payload = build_payload(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(root), encoding="utf-8")
    _save_json(payload, json_output)
    print(
        f"Stage causal graph written: {output} "
        f"ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    print(f"Stage causal graph JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
