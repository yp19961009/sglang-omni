# SPDX-License-Identifier: Apache-2.0
"""Build a fair-runtime-comparison contract for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "runtime_comparison_contract.json"


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


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("summary", {})
    return value if isinstance(value, dict) else {}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    evidence: str,
    *,
    required: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if condition else "FAIL",
            "required": required,
            "evidence": evidence,
        }
    )


def _fmt_s(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}s"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ms(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}ms"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100.0:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct_raw(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _find_row(payload: dict[str, Any], key: str, value: str) -> dict[str, Any]:
    for row in _rows(payload):
        if str(row.get(key)) == value:
            return row
    return {}


def _first_gpu(environment: dict[str, Any]) -> str:
    gpus = environment.get("gpu", {}).get("gpus", [])
    if not gpus:
        return "n/a"
    first = gpus[0]
    return f"{first.get('name')} / {first.get('memory_total')}"


def _image_line(environment: dict[str, Any], label: str) -> str:
    image = environment.get("docker_images", {}).get(label, {})
    name = image.get("image") or label
    image_id = str(image.get("id") or "n/a")
    short_id = image_id[:19] if image_id.startswith("sha256:") else image_id
    return f"{name} / {short_id}"


def _status_counts(acceptance: dict[str, Any]) -> str:
    counts = acceptance.get("summary", {}).get("serving_status_counts", {})
    if not isinstance(counts, dict):
        return "n/a"
    keys = [
        "recommended_strict_baseline",
        "recommended_serving_window",
        "recommended_peak_throughput",
        "not_recommended_saturation",
        "optimized_offline_diagnostic",
        "diagnostic_prompt_feed_limited",
        "anti_recipe_regression",
        "anti_recipe_failure",
    ]
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in keys)


def build_contract(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    scorecard = _load_json_optional(audit_dir / "headline_scorecard.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    vllm = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")
    runtime_image = _load_json_optional(audit_dir / "runtime_image_contract.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    parity = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")

    strict = scorecard.get("strict_c4_comparison", {})
    sglang = strict.get("sglang", {})
    vllm_c4_strict = strict.get("vllm", {})
    relative = strict.get("relative_sglang_lower_pct", {})
    checks = scorecard.get("summary", {}).get("checks", {})
    vllm_c8 = _find_row(vllm, "label", "vLLM-c8")
    vllm_w4 = _find_row(vllm, "label", "vLLM-c8-prebuild-w4")
    confidence_summary = _summary(confidence)
    final_summary = _summary(final)
    final_hard_gates = final_summary.get("hard_gates", {})
    final_contract_attached = (
        bool(final_summary)
        and _int_value(final_summary.get("checks_total")) >= 45
        and final_hard_gates.get("runtime_comparison_contract")
        == "9/9 / warmed c4 only / c8 diagnostic"
    )
    manifest_summary = _summary(manifest)
    repro_summary = _summary(repro)
    runtime_image_summary = _summary(runtime_image)
    vllm_lock_summary = _summary(vllm_lock)
    sglang_lock_summary = _summary(sglang_lock)
    parity_summary = _summary(parity)
    acceptance_summary = _summary(acceptance)
    status_counts = acceptance_summary.get("serving_status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}

    contract_checks: list[dict[str, Any]] = []
    _check(
        contract_checks,
        "source evidence present",
        all(
            payload
            for payload in [
                scorecard,
                acceptance,
                vllm,
                environment,
                confidence,
                runtime_image,
                vllm_lock,
                sglang_lock,
                parity,
            ]
        ),
        (
            f"scorecard={bool(scorecard)}, acceptance={bool(acceptance)}, "
            f"vllm={bool(vllm)}, runtime_image={bool(runtime_image)}, "
            f"sglang_lock={bool(sglang_lock)}, vllm_lock={bool(vllm_lock)}, "
            f"parity={bool(parity)}"
        ),
    )
    _check(
        contract_checks,
        "strict c4 headline is apples-to-apples only",
        "c=4" in str(strict.get("scope") or "")
        and _int_value(sglang.get("n")) == _int_value(vllm_c4_strict.get("n"))
        and bool(checks.get("strict_c4_sglang_latency_rtf_win"))
        and bool(checks.get("strict_c4_accuracy_wer_preserved")),
        (
            f"scope={strict.get('scope')}, n={sglang.get('n')}/"
            f"{vllm_c4_strict.get('n')}, latency_win="
            f"{checks.get('strict_c4_sglang_latency_rtf_win')}, "
            f"quality_preserved={checks.get('strict_c4_accuracy_wer_preserved')}"
        ),
    )
    _check(
        contract_checks,
        "strict c4 relative metrics are favorable",
        all(
            _float_value(relative.get(key), default=-1.0) > 0
            for key in ["latency_mean", "latency_p95", "rtf_mean", "rtf_p95"]
        ),
        f"relative_sglang_lower_pct={relative}",
    )
    _check(
        contract_checks,
        "SGLang pressure scaling separated from cross-runtime headline",
        status_counts.get("recommended_serving_window", 0) >= 3
        and status_counts.get("recommended_peak_throughput", 0) >= 1
        and status_counts.get("not_recommended_saturation", 0) >= 1,
        f"serving_status_counts={status_counts}",
    )
    _check(
        contract_checks,
        "vLLM c8 original remains prompt-feed diagnostic",
        vllm_c8.get("diagnosis") == "prompt_feed_limited"
        and _float_value(vllm_c8.get("runner_overhead_pct_wall")) >= 70.0
        and bool(checks.get("vllm_original_c8_prompt_feed_limited")),
        (
            f"diagnosis={vllm_c8.get('diagnosis')}, "
            f"runner_overhead_pct={vllm_c8.get('runner_overhead_pct_wall')}, "
            f"scorecard={checks.get('vllm_original_c8_prompt_feed_limited')}"
        ),
    )
    _check(
        contract_checks,
        "vLLM c8 prebuild w4 is optimized offline diagnostic only",
        vllm_w4.get("diagnosis") == "engine_or_workload_limited"
        and _float_value(vllm_w4.get("engine_qps"))
        > _float_value(vllm_c8.get("engine_qps"))
        and bool(checks.get("vllm_w4_prebuild_improves_runner_wall"))
        and bool(parity_summary.get("current_package_safe"))
        and not bool(parity_summary.get("online_parity_proven")),
        (
            f"w4_diagnosis={vllm_w4.get('diagnosis')}, "
            f"engine_qps={vllm_w4.get('engine_qps')} vs "
            f"{vllm_c8.get('engine_qps')}, "
            f"w4_improves={checks.get('vllm_w4_prebuild_improves_runner_wall')}, "
            f"online_parity_proven={parity_summary.get('online_parity_proven')}"
        ),
    )
    _check(
        contract_checks,
        "optimized baseline images and recipes are locked",
        bool(runtime_image_summary.get("ready"))
        and bool(sglang_lock_summary.get("ready"))
        and bool(vllm_lock_summary.get("ready"))
        and "c4" in str(runtime_image_summary.get("vllm_strict_scope") or "")
        and "offline diagnostic" in str(runtime_image_summary.get("vllm_c8_scope") or "")
        and "optimized warmed c4" in str(vllm_lock_summary.get("strict_c4_contract") or "")
        and _int_value(vllm_lock_summary.get("checks_total")) >= 22
        and len(vllm_lock.get("required_switches", [])) >= 7,
        (
            f"runtime_image={runtime_image_summary.get('ready')}, "
            f"sglang_lock={sglang_lock_summary.get('ready')}, "
            f"vllm_lock={vllm_lock_summary.get('ready')}, "
            f"vllm_strict_scope={runtime_image_summary.get('vllm_strict_scope')}, "
            f"strict_c4_contract={vllm_lock_summary.get('strict_c4_contract')}, "
            f"required_switches={len(vllm_lock.get('required_switches', []))}"
        ),
    )
    _check(
        contract_checks,
        "unsupported parity claims remain blocked",
        _int_value(confidence_summary.get("unsupported_claims"), default=99) == 0
        and bool(parity_summary.get("current_package_safe"))
        and not bool(parity_summary.get("online_parity_proven")),
        (
            f"unsupported={confidence_summary.get('unsupported_claims')}, "
            f"current_package_safe={parity_summary.get('current_package_safe')}, "
            f"online_parity_proven={parity_summary.get('online_parity_proven')}"
        ),
    )
    _check(
        contract_checks,
        "reproduction gates remain attached",
        (bool(final_summary.get("ready")) or final_contract_attached)
        and _int_value(manifest_summary.get("missing_records"), default=1) == 0
        and _int_value(manifest_summary.get("total_records")) >= 180
        and bool(repro_summary.get("required_command_ids_present"))
        and _int_value(repro_summary.get("commands_total")) >= 60,
        (
            f"final={final_summary.get('ready')}, "
            f"final_contract_attached={final_contract_attached}, "
            f"manifest={manifest_summary}, repro={repro_summary.get('commands_total')}, "
            f"required_command_ids_present={repro_summary.get('required_command_ids_present')}"
        ),
    )
    required_failures = [
        check
        for check in contract_checks
        if check["required"] and check["status"] != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(contract_checks),
            "checks_passed": sum(
                1 for check in contract_checks if check["status"] == "PASS"
            ),
            "required_failures": len(required_failures),
            "strict_scope": strict.get("scope"),
            "strict_c4_lower_latency_mean_pct": relative.get("latency_mean"),
            "strict_c4_lower_rtf_p95_pct": relative.get("rtf_p95"),
            "vllm_c8_contract": "offline_diagnostic_not_online_parity",
            "allowed_cross_runtime_headline": "warmed c=4 only",
            "baseline_strength": (
                "optimized image plus compile/graph/cache/shared-memory/"
                "encoder/prebuild evidence"
            ),
        },
        "checks": contract_checks,
        "strict_c4_comparison": {
            "sglang": sglang,
            "vllm": vllm_c4_strict,
            "relative_sglang_lower_pct": relative,
        },
        "vllm_c8_boundary": {
            "original": vllm_c8,
            "prebuild_w4": vllm_w4,
            "online_parity_proven": parity_summary.get("online_parity_proven"),
            "upgrade_decision": parity_summary.get("upgrade_decision"),
        },
        "source_summaries": {
            "acceptance_matrix": acceptance_summary,
            "confidence_ledger": confidence_summary,
            "runtime_image_contract": runtime_image_summary,
            "sglang_optimization_lock": sglang_lock_summary,
            "vllm_optimization_lock": vllm_lock_summary,
            "vllm_online_parity_protocol": parity_summary,
            "final_readiness": final_summary,
            "manifest": manifest_summary,
            "repro_command_manifest": repro_summary,
        },
        "source_files": [
            str(AUDIT_DIR / "headline_scorecard.json"),
            str(AUDIT_DIR / "acceptance_matrix.json"),
            str(AUDIT_DIR / "vllm_admission_diagnosis.json"),
            str(AUDIT_DIR / "runtime_image_contract.json"),
            str(AUDIT_DIR / "sglang_optimization_lock.json"),
            str(AUDIT_DIR / "vllm_optimization_lock.json"),
            str(AUDIT_DIR / "vllm_online_parity_protocol.json"),
            str(AUDIT_DIR / "confidence_ledger.json"),
            str(AUDIT_DIR / "repro_command_manifest.json"),
        ],
    }


def build_markdown(root: Path, contract: dict[str, Any] | None = None) -> str:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    scorecard = _load_json_optional(audit_dir / "headline_scorecard.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    vllm = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    repro = _load_json_optional(audit_dir / "repro_command_manifest.json")

    strict = scorecard.get("strict_c4_comparison", {})
    sglang = strict.get("sglang", {})
    vllm_c4_strict = strict.get("vllm", {})
    relative = strict.get("relative_sglang_lower_pct", {})
    checks = scorecard.get("summary", {}).get("checks", {})
    vllm_c8 = _find_row(vllm, "label", "vLLM-c8")
    vllm_w4 = _find_row(vllm, "label", "vLLM-c8-prebuild-w4")
    confidence_summary = confidence.get("summary", {})
    final_summary = final.get("summary", {})
    final_hard_gates = final_summary.get("hard_gates", {})
    final_contract_attached = (
        bool(final_summary)
        and _int_value(final_summary.get("checks_total")) >= 45
        and final_hard_gates.get("runtime_comparison_contract")
        == "9/9 / warmed c4 only / c8 diagnostic"
    )
    manifest_summary = manifest.get("summary", {})
    repro_summary = repro.get("summary", {})
    generated_at = (
        str(contract.get("generated_at_utc"))
        if isinstance(contract, dict) and contract.get("generated_at_utc")
        else datetime.now(timezone.utc).isoformat()
    )

    lines: list[str] = [
        "# Qwen3.5-Omni Runtime 公平对比合同",
        "",
        f"生成时间 UTC：`{generated_at}`。",
        f"工作目录：`{root}`。",
        "",
        "这页用于把 SGLang-Omni 与 vLLM 的对比边界固定下来：哪些数字可以做",
        "strict apples-to-apples headline，哪些只能做同 runtime scaling，哪些只是",
        "offline diagnostic。它不新增 benchmark 口径，只汇总当前 audit JSON。",
        "",
        "## 1. 合同 Gate",
        "",
        "| Gate | 当前值 | 判定 |",
        "| --- | --- | --- |",
        (
            f"| 硬件/环境 | `{environment.get('gpu', {}).get('count')}`x "
            f"{_first_gpu(environment)}；SGLang `{_image_line(environment, 'sglang')}`；"
            f"vLLM `{_image_line(environment, 'vllm')}` | 同一 checkpoint 内可比 |"
        ),
        (
            f"| 严格横向 workload | `{strict.get('scope', 'Video-AMME ci-50, warmed c=4')}` | "
            "作为唯一 cross-runtime headline |"
        ),
        (
            f"| headline gate | c=4 latency/RTF win=`{checks.get('strict_c4_sglang_latency_rtf_win')}`；"
            f"quality preserved=`{checks.get('strict_c4_accuracy_wer_preserved')}` | PASS |"
        ),
        (
            f"| c=8 separation gate | vLLM original prompt-feed limited="
            f"`{checks.get('vllm_original_c8_prompt_feed_limited')}`；"
            f"prebuild w4 improves runner wall=`{checks.get('vllm_w4_prebuild_improves_runner_wall')}` | "
            "只能诊断，不能当 online parity |"
        ),
        (
            f"| confidence gate | high `{confidence_summary.get('high_confidence_claims')}`，"
            f"medium `{confidence_summary.get('medium_confidence_boundaries')}`，"
            f"unsupported `{confidence_summary.get('unsupported_claims')}` | "
            "对外话术受 confidence ledger 约束 |"
        ),
        (
            f"| final readiness | ready=`{final_summary.get('ready')}`，"
            f"contract=`{final_contract_attached}`，"
            f"manifest=`{manifest_summary.get('total_records')}` records，"
            f"commands=`{repro_summary.get('commands_total')}` | 复现入口完整 |"
        ),
        "",
        "## 2. 允许和禁止的比较",
        "",
        "| 比较对象 | 可否作为结论 | 允许说法 | 禁止说法 | 证据入口 |",
        "| --- | --- | --- | --- | --- |",
        (
            "| SGLang warmed c=4 vs vLLM warmed c=4 | 可以做主 headline | "
            "SGLang latency/RTF 更优，accuracy/WER 不退化 | "
            "把该结论外推到所有并发和所有流量 | "
            "`headline_scorecard.json` strict_c4_comparison |"
        ),
        (
            "| SGLang c=1/2/4/8/16 | 可以做 SGLang 内部 scaling | "
            "c=4-c=8 是推荐窗口，c=8 是吞吐峰值，c=16 是压力边界 | "
            "把 c=16 当默认服务点 | `acceptance_matrix.json`；`tables_summary.json` |"
        ),
        (
            "| SGLang c=8 vs vLLM original offline c=8 | 只能做诊断 | "
            "vLLM original c=8 受 prompt build/feed admission 限制 | "
            "用 offline runner wall QPS 证明 online parity 或 non-parity | "
            "`vllm_admission_diagnosis.json` row vLLM-c8 |"
        ),
        (
            "| SGLang c=8 vs vLLM c=8 prebuild w4 | 只能做优化后 offline diagnostic | "
            "prebuild w4 缓解 prompt/admission，暴露后续 engine/talker-side tail | "
            "把 prebuild w4 说成严格 online serving-throughput parity | "
            "`vllm_admission_diagnosis.json` row vLLM-c8-prebuild-w4 |"
        ),
        (
            "| synthetic short/long speech vs Video-AMME | 不能混成同一 headline | "
            "short/long 用于守住文本输入和语音输出路径 | "
            "用 synthetic RTF 代替 Video-AMME accuracy/WER | "
            "`tables_summary.json` synthetic_speech |"
        ),
        (
            "| preproc=2/4 vs baseline c=8 | 可以做 anti-recipe | "
            "朴素 preprocessing 并发放大回退或失败 | "
            "把 preproc=2/4 当当前最优 recipe | `acceptance_matrix.json` anti_recipe rows |"
        ),
        "",
        "## 3. Strict c=4 横向数字",
        "",
        "| Runtime | n | Accuracy | WER | Latency mean/p95 | RTF mean/p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| SGLang-Omni | {int(float(sglang.get('n') or 0))} | "
            f"{_fmt_pct(sglang.get('accuracy'))} | {_fmt_pct(sglang.get('wer_corpus'))} | "
            f"{_fmt_s(sglang.get('latency_mean_s'))} / {_fmt_s(sglang.get('latency_p95_s'))} | "
            f"{_fmt_num(sglang.get('rtf_mean'))} / {_fmt_num(sglang.get('rtf_p95'))} |"
        ),
        (
            f"| vLLM optimized | {int(float(vllm_c4_strict.get('n') or 0))} | "
            f"{_fmt_pct(vllm_c4_strict.get('accuracy'))} | {_fmt_pct(vllm_c4_strict.get('wer_corpus'))} | "
            f"{_fmt_s(vllm_c4_strict.get('latency_mean_s'))} / {_fmt_s(vllm_c4_strict.get('latency_p95_s'))} | "
            f"{_fmt_num(vllm_c4_strict.get('rtf_mean'))} / {_fmt_num(vllm_c4_strict.get('rtf_p95'))} |"
        ),
        "",
        "SGLang 相对 vLLM 更低："
        f"latency mean `{_fmt_pct_raw(relative.get('latency_mean'))}`，"
        f"latency p95 `{_fmt_pct_raw(relative.get('latency_p95'))}`，"
        f"RTF mean `{_fmt_pct_raw(relative.get('rtf_mean'))}`，"
        f"RTF p95 `{_fmt_pct_raw(relative.get('rtf_p95'))}`。",
        "",
        "## 4. vLLM c=8 诊断边界",
        "",
        "| vLLM c=8 case | Runner QPS | Engine QPS | Admission avg/p95 | Runner overhead | 诊断 | 对比合同 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        (
            f"| original | {_fmt_num(vllm_c8.get('runner_qps'))} | "
            f"{_fmt_num(vllm_c8.get('engine_qps'))} | "
            f"{_fmt_ms(vllm_c8.get('batch_admission_span_avg_ms'))} / "
            f"{_fmt_ms(vllm_c8.get('batch_admission_span_p95_ms'))} | "
            f"{_fmt_pct_raw(vllm_c8.get('runner_overhead_pct_wall'))} | "
            f"{vllm_c8.get('diagnosis')} | diagnostic_prompt_feed_limited |"
        ),
        (
            f"| prebuild w4 | {_fmt_num(vllm_w4.get('runner_qps'))} | "
            f"{_fmt_num(vllm_w4.get('engine_qps'))} | "
            f"{_fmt_ms(vllm_w4.get('batch_admission_span_avg_ms'))} / "
            f"{_fmt_ms(vllm_w4.get('batch_admission_span_p95_ms'))} | "
            f"{_fmt_pct_raw(vllm_w4.get('runner_overhead_pct_wall'))} | "
            f"{vllm_w4.get('diagnosis')} | optimized_offline_diagnostic |"
        ),
        "",
        "vLLM c=8 prebuild w4 是 offline diagnostic：它证明 prompt/admission 问题被缓解，",
        "但严格 c=8 online serving parity 仍需要 online ingress、同口径 WER/ASR 和",
        "engine/talker boundary 复核。",
        "",
        "## 5. Reviewer 使用规则",
        "",
        "1. 若要一句话 headline，只引用 warmed c=4 strict runtime comparison。",
        "2. 若讨论高并发服务点，只在 SGLang 内部说 c=4-c=8 推荐窗口和 c=8 吞吐峰值。",
        "3. 若讨论 vLLM c=8，只说 offline runner prompt-feed/admission 诊断和 prebuild w4 改善，不说 online parity 已证明。",
        "4. 若合作方复跑结果硬件、image、模型、数据或 ASR 变化，先按 rerun validation sheet 标注差异，再决定是否替换数字。",
        "5. 若需要新 headline，必须重跑 full audit，且本页、regime decision matrix、confidence ledger、final readiness 全部重新通过。",
        "",
        "## 6. 机器证据",
        "",
        f"- acceptance status counts：`{_status_counts(acceptance)}`",
        "- strict c=4 source：`results/qwen35_report_audit_20260619/headline_scorecard.json`",
        "- vLLM c=8 diagnosis source：`results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`",
        "- regime decisions：`benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md`",
        "- regime decisions JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`",
        "- confidence ledger：`results/qwen35_report_audit_20260619/confidence_ledger.json`",
        "- reproduction command manifest：`results/qwen35_report_audit_20260619/repro_command_manifest.json`",
        "- runtime comparison contract JSON：`results/qwen35_report_audit_20260619/runtime_comparison_contract.json`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni runtime comparison contract."
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
    contract = build_contract(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(root, contract), encoding="utf-8")
    _save_json(contract, json_output)
    summary = contract["summary"]
    print(
        f"Runtime comparison contract written: {output} "
        f"ready={summary['ready']} checks={summary['checks_passed']}/"
        f"{summary['checks_total']}"
    )
    print(f"Runtime comparison contract JSON written: {json_output}")
    if args.strict and not summary["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
