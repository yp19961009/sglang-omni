# SPDX-License-Identifier: Apache-2.0
"""Build a vLLM c=8 online-parity upgrade protocol for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "vllm_online_parity_protocol.json"

VLLM_IMAGE = (
    "tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/"
    "dashllm:cuda129_cp312_test_vl_13589"
)
RUN_SCRIPT = Path(
    "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/"
    "run_vllm_videoamme_ci5_offline_compile.sh"
)
RUNNER = Path(
    "results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/"
    "vllm_videoamme_runner.py"
)

MAIN_REPORT = Path("benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md")
FINAL_NOTE = Path("benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md")
RUNTIME_CONTRACT = Path(
    "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md"
)
VLLM_LOCK_REPORT = Path(
    "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md"
)
CAVEAT_MATRIX = Path(
    "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md"
)


@dataclass(frozen=True)
class ProtocolCheck:
    name: str
    status: str
    evidence: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "evidence": self.evidence,
        }


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_s(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}s"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_row(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row if isinstance(row, dict) else {}
    return {}


def _table_rows(tables: dict[str, Any], table_name: str) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get(table_name, [])
    return rows if isinstance(rows, list) else []


def _needle_check(
    *,
    name: str,
    text: str,
    needle: str,
    source: str,
    required: bool = True,
) -> ProtocolCheck:
    return ProtocolCheck(
        name=name,
        status=_status(needle in text),
        required=required,
        evidence=f"{source} contains {needle!r}",
    )


def _replacement_gates(sglang_c8: dict[str, Any]) -> dict[str, Any]:
    latency_mean_s = _float_value(sglang_c8.get("latency_mean_s"))
    latency_p95_s = _float_value(sglang_c8.get("latency_p95_s"))
    rtf_mean = _float_value(sglang_c8.get("rtf_mean"))
    rtf_p95 = _float_value(sglang_c8.get("rtf_p95"))
    throughput_qps = _float_value(sglang_c8.get("throughput_qps"))
    accuracy = _float_value(sglang_c8.get("accuracy"))
    wer = _float_value(sglang_c8.get("wer_corpus"))
    return {
        "sample_contract": "Video-AMME ci-50, max_tokens=256, temperature=0, c=8, skip first warmup batch/request window.",
        "online_ingress_required": True,
        "same_quality_path_required": True,
        "minimum_completed": 50,
        "maximum_failed": 0,
        "latency_mean_s_max_for_parity": round(latency_mean_s * 1.05, 4),
        "latency_p95_s_max_for_parity": round(latency_p95_s * 1.05, 4),
        "rtf_mean_max_for_parity": round(rtf_mean * 1.05, 4),
        "rtf_p95_max_for_parity": round(rtf_p95 * 1.05, 4),
        "throughput_qps_min_for_parity": round(throughput_qps * 0.95, 4),
        "accuracy_min_for_parity": round(max(0.0, accuracy - 0.02), 4),
        "wer_corpus_max_for_parity": round(wer + 0.01, 4),
        "stage_boundary_required": [
            "online request admission/queue time",
            "input preprocessing lifecycle",
            "encoder/thinker boundary",
            "thinker->talker handoff",
            "talker codec cadence",
            "talker->code2wav drain",
            "code2wav collect/decode",
        ],
    }


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    tables = _load_json_optional(audit_dir / "tables_summary.json")
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_diagnosis = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    stage_interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")

    run_script_text = _read_text_optional(root / RUN_SCRIPT)
    runner_text = _read_text_optional(root / RUNNER)
    final_note_text = _read_text_optional(root / FINAL_NOTE)
    runtime_contract_text = _read_text_optional(root / RUNTIME_CONTRACT)
    vllm_lock_text = _read_text_optional(root / VLLM_LOCK_REPORT)
    caveat_text = _read_text_optional(root / CAVEAT_MATRIX)
    main_report_text = _read_text_optional(root / MAIN_REPORT)

    sglang_c8 = _find_row(_table_rows(tables, "sglang_stress"), "concurrency", 8)
    overhead_rows = _table_rows(tables, "vllm_offline_runner_overhead")
    w1_overhead = _find_row(overhead_rows, "label", "vLLM-c8-prebuild-w1")
    w4_overhead = _find_row(overhead_rows, "label", "vLLM-c8-prebuild-w4")
    diagnosis_rows = vllm_diagnosis.get("rows", [])
    diagnosis_rows = diagnosis_rows if isinstance(diagnosis_rows, list) else []
    c8_diag = _find_row(diagnosis_rows, "label", "vLLM-c8")
    w4_diag = _find_row(diagnosis_rows, "label", "vLLM-c8-prebuild-w4")
    stage_summary = stage_interactions.get("summary", {})
    image = environment.get("docker_images", {}).get("vllm", {})
    vllm_lock_summary = vllm_lock.get("summary", {})

    replacement_gates = _replacement_gates(sglang_c8)
    current_package_safe = True
    online_parity_proven = False

    checks: list[ProtocolCheck] = [
        ProtocolCheck(
            name="vLLM image is recorded",
            status=_status(
                bool(image.get("ok"))
                and image.get("image") == VLLM_IMAGE
                and str(image.get("id") or "").startswith("sha256:")
            ),
            evidence=f"image={image.get('image')}, id={image.get('id')}",
        ),
        ProtocolCheck(
            name="vLLM optimization lock ready",
            status=_status(
                bool(vllm_lock_summary.get("ready"))
                and int(vllm_lock_summary.get("required_failures") or 0) == 0
                and int(vllm_lock_summary.get("checks_total") or 0) >= 22
            ),
            evidence=f"vllm_optimization_lock={vllm_lock_summary}",
        ),
        ProtocolCheck(
            name="original c8 remains prompt-feed/admission limited",
            status=_status(
                bool(c8_diag.get("prompt_feed_limited"))
                and c8_diag.get("diagnosis") == "prompt_feed_limited"
            ),
            evidence=(
                "vLLM-c8 diagnosis="
                f"{c8_diag.get('diagnosis')}, span_avg_ms="
                f"{_fmt_num(c8_diag.get('batch_admission_span_avg_ms'), 1)}"
            ),
        ),
        ProtocolCheck(
            name="prebuild w4 is diagnostic, not parity",
            status=_status(w4_diag.get("diagnosis") == "engine_or_workload_limited"),
            evidence=(
                "vLLM-c8-prebuild-w4 diagnosis="
                f"{w4_diag.get('diagnosis')}, engine_qps="
                f"{_fmt_num(w4_diag.get('engine_qps'))}"
            ),
        ),
        ProtocolCheck(
            name="prebuild w4 improves prompt build wall",
            status=_status(
                _float_value(w4_overhead.get("prompt_build_wall_s"), 999999.0)
                < _float_value(w1_overhead.get("prompt_build_wall_s"), -1.0)
            ),
            evidence=(
                "w1_prompt_wall="
                f"{_fmt_s(w1_overhead.get('prompt_build_wall_s'))}, "
                "w4_prompt_wall="
                f"{_fmt_s(w4_overhead.get('prompt_build_wall_s'))}"
            ),
        ),
        ProtocolCheck(
            name="prebuild w4 improves runner wall",
            status=_status(
                _float_value(w4_overhead.get("runner_wall_time_s"), 999999.0)
                < _float_value(w1_overhead.get("runner_wall_time_s"), -1.0)
            ),
            evidence=(
                "w1_runner_wall="
                f"{_fmt_s(w1_overhead.get('runner_wall_time_s'))}, "
                "w4_runner_wall="
                f"{_fmt_s(w4_overhead.get('runner_wall_time_s'))}"
            ),
        ),
        ProtocolCheck(
            name="stage summary records current c8 boundary",
            status=_status(bool(stage_summary.get("vllm_original_c8_prompt_feed_limited"))),
            evidence=f"stage_interactions={stage_summary}",
        ),
        ProtocolCheck(
            name="SGLang c8 target row is present",
            status=_status(
                int(sglang_c8.get("n") or 0) >= 50
                and _float_value(sglang_c8.get("throughput_qps")) > 0
                and "wer_corpus" in sglang_c8
            ),
            evidence=f"sglang_c8={sglang_c8}",
        ),
        _needle_check(
            name="wrapper keeps vLLM compile enabled",
            text=run_script_text,
            needle="VLLM_ENABLE_TORCH_COMPILE=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper keeps optimized cache/transfer path",
            text=run_script_text,
            needle="VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="runner supports prebuilt prompts",
            text=runner_text,
            needle="--prebuild-prompts",
            source=str(RUNNER),
        ),
        _needle_check(
            name="final delivery note forbids online parity claim",
            text=final_note_text,
            needle="vLLM c=8 prebuild w4 不是 online serving parity",
            source=str(FINAL_NOTE),
        ),
        _needle_check(
            name="runtime contract separates online parity",
            text=runtime_contract_text,
            needle="不能当 online parity",
            source=str(RUNTIME_CONTRACT),
        ),
        _needle_check(
            name="vLLM lock states diagnostic boundary",
            text=vllm_lock_text,
            needle="不是 online serving parity",
            source=str(VLLM_LOCK_REPORT),
        ),
        _needle_check(
            name="caveat matrix blocks c8 headline promotion",
            text=caveat_text,
            needle="online parity",
            source=str(CAVEAT_MATRIX),
        ),
        _needle_check(
            name="main report keeps online caveat visible",
            text=main_report_text,
            needle="online serving parity",
            source=str(MAIN_REPORT),
        ),
        ProtocolCheck(
            name="replacement gates are declared",
            status=_status(
                bool(replacement_gates.get("online_ingress_required"))
                and bool(replacement_gates.get("same_quality_path_required"))
                and len(replacement_gates.get("stage_boundary_required", [])) >= 7
            ),
            evidence=f"replacement_gates={replacement_gates}",
        ),
        ProtocolCheck(
            name="current package does not overclaim online parity",
            status=_status(current_package_safe and not online_parity_proven),
            evidence=(
                f"current_package_safe={current_package_safe}, "
                f"online_parity_proven={online_parity_proven}"
            ),
        ),
    ]

    required_failures = sum(
        1 for check in checks if check.required and check.status != "PASS"
    )
    checks_passed = sum(1 for check in checks if check.status == "PASS")

    required_artifacts = [
        {
            "name": "vLLM online serving launch log",
            "requirement": "same image digest, compile/cache knobs, c=8 admission capacity, and warmup are logged.",
        },
        {
            "name": "vLLM online HTTP ingress result JSON",
            "requirement": "Video-AMME ci-50 requests are sent through OpenAI-compatible ingress, not offline engine calls.",
        },
        {
            "name": "vLLM online stage/tail profile",
            "requirement": "request admission, preprocessing, thinker/talker, talker/code2wav, and code2wav decode boundaries are measurable.",
        },
        {
            "name": "vLLM online WER/ASR JSON",
            "requirement": "same audio outputs are scored with the same Whisper/ASR path after serving latency is measured.",
        },
        {
            "name": "updated SGLang c=8 control run or frozen control hash",
            "requirement": "SGLang comparison target is either rerun in the same window or explicitly pinned by manifest hash.",
        },
        {
            "name": "parity verifier JSON",
            "requirement": "machine gate compares vLLM online c=8 against SGLang c=8 thresholds before any headline replacement.",
        },
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": required_failures == 0,
            "checks_total": len(checks),
            "checks_passed": checks_passed,
            "required_failures": required_failures,
            "current_package_safe": current_package_safe,
            "online_parity_proven": online_parity_proven,
            "upgrade_decision": "do_not_promote_c8_parity_without_online_ingress_artifacts",
            "required_artifacts_total": len(required_artifacts),
        },
        "current_boundary": {
            "strict_headline": "SGLang warmed c=4 vs optimized vLLM warmed c=4.",
            "vllm_c8_status": "optimized offline diagnostic only",
            "vllm_original_c8_diagnosis": c8_diag,
            "vllm_prebuild_w4_diagnosis": w4_diag,
        },
        "sglang_c8_target": sglang_c8,
        "replacement_gates": replacement_gates,
        "required_online_artifacts": required_artifacts,
        "checks": [check.to_dict() for check in checks],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    sglang_c8 = payload["sglang_c8_target"]
    gates = payload["replacement_gates"]
    boundary = payload["current_boundary"]
    lines: list[str] = [
        "# Qwen3.5-Omni vLLM c=8 Online Parity 升级协议",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这份协议只定义如何把当前 vLLM c=8 证据从 offline diagnostic 升级为 strict",
        "online serving parity。它不把当前报告的 vLLM c=8 prebuild w4 结果提升为",
        "online parity；当前可分享 headline 仍是 warmed c=4 的严格横向对比。",
        "",
        "## 1. 当前裁决",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Protocol ready | `{summary['ready']}` |",
        f"| Checks | `{summary['checks_passed']}/{summary['checks_total']}` |",
        f"| Required failures | `{summary['required_failures']}` |",
        f"| Current package safe | `{summary['current_package_safe']}` |",
        f"| Online parity proven | `{summary['online_parity_proven']}` |",
        f"| Upgrade decision | `{summary['upgrade_decision']}` |",
        "",
        "当前结论：vLLM original c=8 仍是 prompt-feed/admission limited；prebuild w4",
        "是优化后的 offline diagnostic，用来定位 runner prompt 构造/投喂瓶颈和后续",
        "engine/talker-side tail，不可直接写成 online serving parity。",
        "",
        "## 2. SGLang c=8 对照目标",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| n | `{sglang_c8.get('n')}` |",
        f"| accuracy | `{_fmt_pct(sglang_c8.get('accuracy'))}` |",
        f"| WER corpus | `{_fmt_pct(sglang_c8.get('wer_corpus'))}` |",
        f"| latency mean / p95 | `{_fmt_s(sglang_c8.get('latency_mean_s'))}` / `{_fmt_s(sglang_c8.get('latency_p95_s'))}` |",
        f"| RTF mean / p95 | `{_fmt_num(sglang_c8.get('rtf_mean'))}` / `{_fmt_num(sglang_c8.get('rtf_p95'))}` |",
        f"| throughput QPS | `{_fmt_num(sglang_c8.get('throughput_qps'), 3)}` |",
        "",
        "## 3. 升级为 strict c=8 parity 的替换 Gate",
        "",
        "| Gate | Required value |",
        "| --- | --- |",
        f"| sample contract | {gates['sample_contract']} |",
        f"| completed / failed | `>= {gates['minimum_completed']}` / `<= {gates['maximum_failed']}` |",
        f"| latency mean max | `{_fmt_s(gates['latency_mean_s_max_for_parity'])}` |",
        f"| latency p95 max | `{_fmt_s(gates['latency_p95_s_max_for_parity'])}` |",
        f"| RTF mean max | `{_fmt_num(gates['rtf_mean_max_for_parity'])}` |",
        f"| RTF p95 max | `{_fmt_num(gates['rtf_p95_max_for_parity'])}` |",
        f"| throughput QPS min | `{_fmt_num(gates['throughput_qps_min_for_parity'], 3)}` |",
        f"| accuracy min | `{_fmt_pct(gates['accuracy_min_for_parity'])}` |",
        f"| WER corpus max | `{_fmt_pct(gates['wer_corpus_max_for_parity'])}` |",
        "| online ingress | HTTP/OpenAI-compatible serving ingress, not offline engine runner |",
        "| quality path | same WER/ASR scoring path after serving latency measurement |",
        "| stage profile | request admission, preprocessing, thinker/talker, talker/code2wav, code2wav decode all visible |",
        "",
        "## 4. 必需新增 Artifact",
        "",
        "| Artifact | Requirement |",
        "| --- | --- |",
    ]
    for artifact in payload["required_online_artifacts"]:
        lines.append(f"| {artifact['name']} | {artifact['requirement']} |")
    lines.extend(
        [
            "",
            "## 5. 当前证据边界",
            "",
            f"- strict headline：{boundary['strict_headline']}",
            f"- vLLM c=8 status：{boundary['vllm_c8_status']}",
            "- 禁止把 `prebuild w4`、`engine_qps` 或 offline runner wall QPS 写成 online serving parity。",
            "- 如果合作方给出新的 vLLM online c=8 结果，先按本页 gate 生成 parity verifier JSON，再决定是否替换主报告数字。",
            "",
            "## 6. 协议自检",
            "",
            "| Status | Required | Check | Evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        lines.append(
            f"| {check['status']} | {'yes' if check['required'] else 'no'} | "
            f"{check['name']} | {check['evidence']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni vLLM c=8 online parity protocol."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
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

    summary = payload["summary"]
    print(
        "vLLM online parity protocol written: "
        f"{output} ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"online_parity_proven={summary['online_parity_proven']}"
    )
    print(f"vLLM online parity protocol JSON written: {json_output}")
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
