# SPDX-License-Identifier: Apache-2.0
"""Build a vLLM optimization lock matrix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "vllm_optimization_lock.json"

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

LOGS = {
    "c4": Path(
        "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/"
        "run.log"
    ),
    "c8": Path(
        "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/"
        "run.log"
    ),
    "c8_prebuild_w4": Path(
        "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/"
        "run.log"
    ),
}


@dataclass(frozen=True)
class LockCheck:
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


def _fmt_ms(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}ms"
    except (TypeError, ValueError):
        return "n/a"


def _needle_check(
    *,
    name: str,
    text: str,
    needle: str,
    source: str,
    required: bool = True,
) -> LockCheck:
    return LockCheck(
        name=name,
        status=_status(needle in text),
        required=required,
        evidence=f"{source} contains {needle!r}",
    )


def _find_row(payload: dict[str, Any], label: str) -> dict[str, Any]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if str(row.get("label")) == label:
            return row if isinstance(row, dict) else {}
    return {}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _case_rows(vllm_admission: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ["vLLM-c4", "vLLM-c8", "vLLM-c8-prebuild-w4"]:
        row = _find_row(vllm_admission, label)
        if row:
            rows.append(row)
    return rows


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    vllm_admission = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")

    run_script_text = _read_text_optional(root / RUN_SCRIPT)
    runner_text = _read_text_optional(root / RUNNER)
    c4_log_text = _read_text_optional(root / LOGS["c4"])
    c8_log_text = _read_text_optional(root / LOGS["c8"])
    w4_log_text = _read_text_optional(root / LOGS["c8_prebuild_w4"])
    combined_log_text = "\n".join([c4_log_text, c8_log_text, w4_log_text])

    image = environment.get("docker_images", {}).get("vllm", {})
    c8 = _find_row(vllm_admission, "vLLM-c8")
    w4 = _find_row(vllm_admission, "vLLM-c8-prebuild-w4")

    checks: list[LockCheck] = [
        LockCheck(
            name="vLLM image locked",
            status=_status(
                bool(image.get("ok"))
                and image.get("image") == VLLM_IMAGE
                and str(image.get("id") or "").startswith("sha256:")
            ),
            evidence=(
                f"image={image.get('image')}, id={image.get('id')}, "
                f"created={image.get('created')}"
            ),
        ),
        LockCheck(
            name="vLLM wrapper present",
            status=_status((root / RUN_SCRIPT).is_file()),
            evidence=str(root / RUN_SCRIPT),
        ),
        LockCheck(
            name="vLLM runner present",
            status=_status((root / RUNNER).is_file()),
            evidence=str(root / RUNNER),
        ),
        _needle_check(
            name="wrapper uses locked image",
            text=run_script_text,
            needle=VLLM_IMAGE,
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper enables torch compile",
            text=run_script_text,
            needle="VLLM_ENABLE_TORCH_COMPILE=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper enables hidden-buffer fast transfer",
            text=run_script_text,
            needle="VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper enables shared-memory hidden buffer",
            text=run_script_text,
            needle="VLLM_HIDDEN_BUFFER_BACKEND=shm",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper enables encoder torch compile",
            text=run_script_text,
            needle="VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper enables encoder batching",
            text=run_script_text,
            needle="VLLM_OMNI_ENABLE_ENCODER_BATCH=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="wrapper reuses thinker preprocessing for talker",
            text=run_script_text,
            needle="VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS=True",
            source=str(RUN_SCRIPT),
        ),
        _needle_check(
            name="runner supports prebuild prompts",
            text=runner_text,
            needle="--prebuild-prompts",
            source=str(RUNNER),
        ),
        _needle_check(
            name="runner supports prebuild workers",
            text=runner_text,
            needle="--prebuild-workers",
            source=str(RUNNER),
        ),
        _needle_check(
            name="log proves torch.compile preflight",
            text=combined_log_text,
            needle="preflight torch.compile ok",
            source="vLLM run logs",
        ),
        _needle_check(
            name="log proves enforce_eager false",
            text=combined_log_text,
            needle="'enforce_eager': False",
            source="vLLM run logs",
        ),
        _needle_check(
            name="log proves VLLM compile mode",
            text=combined_log_text,
            needle="CompilationMode.VLLM_COMPILE",
            source="vLLM run logs",
        ),
        _needle_check(
            name="log proves FULL_AND_PIECEWISE CUDA graph",
            text=combined_log_text,
            needle="FULL_AND_PIECEWISE",
            source="vLLM run logs",
        ),
        _needle_check(
            name="log proves chunked prefill",
            text=combined_log_text,
            needle="'enable_chunked_prefill': True",
            source="vLLM run logs",
        ),
        _needle_check(
            name="log proves prefix caching",
            text=combined_log_text,
            needle="'enable_prefix_caching': True",
            source="vLLM run logs",
        ),
        _needle_check(
            name="w4 diagnostic uses prebuild args",
            text=w4_log_text,
            needle="EXTRA_ARGS=--prebuild-prompts --prebuild-workers 4",
            source=str(LOGS["c8_prebuild_w4"]),
        ),
        _needle_check(
            name="w4 diagnostic logs prebuilt batches",
            text=w4_log_text,
            needle="Prebuilt batch 0 prompts",
            source=str(LOGS["c8_prebuild_w4"]),
        ),
        LockCheck(
            name="original c8 prompt-feed diagnosis locked",
            status=_status(bool(c8.get("prompt_feed_limited"))),
            evidence=(
                f"label={c8.get('label')}, diagnosis={c8.get('diagnosis')}, "
                f"admission_p95={_fmt_ms(c8.get('batch_admission_span_p95_ms'))}"
            ),
        ),
        LockCheck(
            name="prebuild w4 improves runner wall",
            status=_status(
                _float_value(w4.get("runner_qps"))
                > _float_value(c8.get("runner_qps"))
                and not bool(w4.get("prompt_feed_limited"))
            ),
            evidence=(
                f"c8_runner_qps={_fmt_num(c8.get('runner_qps'))}, "
                f"w4_runner_qps={_fmt_num(w4.get('runner_qps'))}, "
                f"w4_diagnosis={w4.get('diagnosis')}"
            ),
        ),
    ]

    required_failures = [
        check for check in checks if check.required and check.status != "PASS"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "PASS"),
            "required_failures": len(required_failures),
            "vllm_image": image.get("image"),
            "vllm_image_id": image.get("id"),
            "strict_c4_contract": "optimized warmed c4 apples-to-apples headline only",
            "c8_contract": "prebuild w4 is optimized offline diagnostic, not online parity",
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "case_rows": _case_rows(vllm_admission),
        "required_switches": [
            {
                "switch": "VLLM_ENABLE_TORCH_COMPILE=True",
                "purpose": "avoid conservative eager baseline",
                "source": str(RUN_SCRIPT),
            },
            {
                "switch": "enforce_eager=False",
                "purpose": "engine runs on compile/graph path",
                "source": "vLLM run logs",
            },
            {
                "switch": "FULL_AND_PIECEWISE CUDA graph",
                "purpose": "lock optimized cudagraph mode",
                "source": "vLLM run logs",
            },
            {
                "switch": "enable_prefix_caching=True; enable_chunked_prefill=True",
                "purpose": "lock vLLM-Omni prefill/cache behavior",
                "source": "vLLM run logs",
            },
            {
                "switch": "VLLM_HIDDEN_BUFFER_BACKEND=shm; VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True",
                "purpose": "lock inter-stage shared-memory transfer path",
                "source": str(RUN_SCRIPT),
            },
            {
                "switch": "VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=True; VLLM_OMNI_ENABLE_ENCODER_BATCH=True",
                "purpose": "lock optimized multimodal encoder path",
                "source": str(RUN_SCRIPT),
            },
            {
                "switch": "--prebuild-prompts --prebuild-workers 4",
                "purpose": "strongest current c=8 offline diagnostic",
                "source": str(LOGS["c8_prebuild_w4"]),
            },
        ],
        "source_files": {
            "environment_snapshot": str(audit_dir / "environment_snapshot.json"),
            "vllm_admission_diagnosis": str(audit_dir / "vllm_admission_diagnosis.json"),
            "run_script": str(root / RUN_SCRIPT),
            "runner": str(root / RUNNER),
            "c4_log": str(root / LOGS["c4"]),
            "c8_log": str(root / LOGS["c8"]),
            "c8_prebuild_w4_log": str(root / LOGS["c8_prebuild_w4"]),
        },
    }


def _gate_table(checks: list[dict[str, Any]]) -> list[str]:
    lines = ["| Status | Required | Gate | Evidence |", "| --- | --- | --- | --- |"]
    for check in checks:
        evidence = str(check["evidence"]).replace("|", "\\|")
        required = "yes" if check["required"] else "no"
        lines.append(
            f"| {check['status']} | {required} | {check['name']} | {evidence} |"
        )
    return lines


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines: list[str] = [
        "# Qwen3.5-Omni vLLM 优化锁定矩阵",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这页专门锁定 vLLM baseline 的镜像、run wrapper、compile/cudagraph/cache",
        "优化开关和 c=8 prebuild diagnostic 边界。目的不是新增 benchmark 口径，",
        "而是防止 reviewer 误以为报告拿 SGLang 优化版去比较一个保守 vLLM baseline。",
        "",
        "## 1. 锁定结论",
        "",
        f"- ready：`{summary['ready']}`，checks：`{summary['checks_passed']}/{summary['checks_total']}`，required failures：`{summary['required_failures']}`。",
        f"- vLLM image：`{summary['vllm_image']}`。",
        f"- vLLM image id：`{summary['vllm_image_id']}`。",
        "- strict headline 只使用 warmed c=4 apples-to-apples 对比。",
        "- c=8 prebuild w4 是当前最强 vLLM offline diagnostic，不是 online serving parity。",
        "",
        "## 2. Gate 明细",
        "",
    ]
    lines.extend(_gate_table(payload["checks"]))
    lines.extend(
        [
            "",
            "## 3. 必须锁定的 vLLM 优化开关",
            "",
            "| Switch / evidence | 用途 | 来源 |",
            "| --- | --- | --- |",
        ]
    )
    for item in payload["required_switches"]:
        lines.append(
            f"| `{item['switch']}` | {item['purpose']} | `{item['source']}` |"
        )
    lines.extend(
        [
            "",
            "## 4. vLLM case 锁定表",
            "",
            "| Case | c | Runner QPS | Engine QPS | Admission p95 | Runner overhead | Diagnosis | 使用边界 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["case_rows"]:
        label = row.get("label")
        boundary = (
            "strict warmed baseline"
            if label == "vLLM-c4"
            else "prompt-feed diagnostic"
            if label == "vLLM-c8"
            else "optimized offline diagnostic"
        )
        lines.append(
            "| "
            f"{label} | "
            f"{row.get('concurrency')} | "
            f"{_fmt_num(row.get('runner_qps'))} | "
            f"{_fmt_num(row.get('engine_qps'))} | "
            f"{_fmt_ms(row.get('batch_admission_span_p95_ms'))} | "
            f"{_fmt_num(row.get('runner_overhead_pct_wall'), 1)}% | "
            f"{row.get('diagnosis')} | "
            f"{boundary} |"
        )
    lines.extend(
        [
            "",
            "## 5. 对外使用规则",
            "",
            "- 可以说：vLLM baseline 使用 Qwen3.5-capable 镜像，且 compile、CUDA graph、prefix/chunked prefill、shared-memory transfer 和 encoder compile/batch 等优化路径均有 run script 或 log 证据。",
            "- 可以说：c=8 prebuild w4 已经把 offline runner 的 prompt-feed admission 问题明显缓解，是当前最强 vLLM offline diagnostic。",
            "- 禁止说：已经完成严格 vLLM c=8 online serving parity；该结论需要 online ingress、同口径 WER/ASR 和 engine/talker boundary 复核。",
            "- 禁止把缺少这些开关的新 vLLM 重跑结果直接替换为报告 baseline；必须先刷新本页 JSON、runtime contract、confidence ledger、final readiness 和 full audit。",
            "",
            "## 6. 机器证据",
            "",
        ]
    )
    for name, path in payload["source_files"].items():
        lines.append(f"- {name}：`{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni vLLM optimization lock matrix."
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
    print(
        "vLLM optimization lock written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    print(f"vLLM optimization lock JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
