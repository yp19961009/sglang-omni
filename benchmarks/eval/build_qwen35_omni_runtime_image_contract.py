# SPDX-License-Identifier: Apache-2.0
"""Build a runtime image and optimization contract for Qwen3.5-Omni handoff."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
AUDIT_SUMMARY = AUDIT_DIR / "audit_run_summary.json"
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "runtime_image_contract.json"

SGLANG_IMAGE = "frankleeeee/sglang-omni:dev"
VLLM_IMAGE = (
    "tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/"
    "dashllm:cuda129_cp312_test_vl_13589"
)
SGLANG_IMAGE_ID = "sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84"

REQUIRED_COMMAND_IDS = {
    "launch_sglang_optimized",
    "sglang_videoamme_stress",
    "sglang_synthetic_text_to_speech",
    "vllm_c1_original",
    "vllm_c8_original",
    "vllm_c8_prebuild_w4",
}


@dataclass(frozen=True)
class ContractCheck:
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _audit_green_or_in_progress(payload: dict[str, Any]) -> bool:
    if bool(payload.get("ok")):
        return True
    if not bool(payload.get("in_progress")):
        return False
    steps = payload.get("steps", [])
    return isinstance(steps, list) and all(
        isinstance(step, dict) and bool(step.get("ok")) for step in steps
    )


def _failed_required_check_names(payload: dict[str, Any]) -> set[str]:
    failed: set[str] = set()
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        return failed
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("required") is False:
            continue
        if check.get("status") == "FAIL":
            failed.add(str(check.get("name") or ""))
    return failed


def _preflight_pending_in_full_audit(
    preflight: dict[str, Any], audit_summary: dict[str, Any]
) -> bool:
    summary = preflight.get("summary", {})
    if not isinstance(summary, dict):
        return False
    return (
        _audit_green_or_in_progress(audit_summary)
        and bool(audit_summary.get("in_progress"))
        and not bool(summary.get("ready"))
        and _int_value(summary.get("total_checks")) >= 62
        and _int_value(summary.get("required_failures"), default=99) == 1
        and _failed_required_check_names(preflight)
        == {"final checkpoint watchlist JSON"}
    )


def _environment_ready_or_pending(
    environment_summary: dict[str, Any],
    preflight: dict[str, Any],
    audit_summary: dict[str, Any],
) -> tuple[bool, bool]:
    if bool(environment_summary.get("ready")):
        return True, False
    claims = environment_summary.get("claims", {})
    coverage = environment_summary.get("coverage", {})
    manifest = environment_summary.get("manifest", {})
    preflight_pending = _preflight_pending_in_full_audit(preflight, audit_summary)
    pending = (
        preflight_pending
        and bool(claims.get("passed"))
        and bool(coverage.get("complete"))
        and _int_value(manifest.get("missing_records"), default=1) == 0
        and _int_value(manifest.get("total_records")) >= 180
    )
    return pending, pending


def _switch_text(lock_payload: dict[str, Any], key: str) -> str:
    rows = lock_payload.get(key, [])
    if not isinstance(rows, list):
        return ""
    parts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            parts.append(str(row.get("switch") or ""))
            parts.append(str(row.get("purpose") or ""))
            parts.append(str(row.get("source") or ""))
    return "\n".join(parts)


def _command_ids(repro_manifest: dict[str, Any]) -> set[str]:
    commands = repro_manifest.get("commands", [])
    if not isinstance(commands, list):
        return set()
    return {str(row.get("id")) for row in commands if isinstance(row, dict)}


def _all_needles(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    online_protocol = _load_json_optional(
        audit_dir / "vllm_online_parity_protocol.json"
    )
    repro_manifest = _load_json_optional(audit_dir / "repro_command_manifest.json")
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    audit_summary = _load_json_optional(root / AUDIT_SUMMARY)

    docker_images = environment.get("docker_images", {})
    sglang_image = docker_images.get("sglang", {})
    vllm_image = docker_images.get("vllm", {})
    gpu = environment.get("gpu", {})
    environment_summary = environment.get("audit", {}).get("summary", {})
    sglang_summary = sglang_lock.get("summary", {})
    vllm_summary = vllm_lock.get("summary", {})
    online_summary = online_protocol.get("summary", {})
    repro_summary = repro_manifest.get("summary", {})
    environment_ready, environment_pending = _environment_ready_or_pending(
        environment_summary,
        preflight,
        audit_summary,
    )

    sglang_switch_text = _switch_text(sglang_lock, "recipe_switches")
    vllm_switch_text = _switch_text(vllm_lock, "required_switches")
    sglang_switch_ok, sglang_switch_missing = _all_needles(
        sglang_switch_text,
        [
            "NO_CODE2WAV_TORCH_COMPILE=0",
            "TORCHDYNAMO_DISABLE=0",
            "--thinker-cuda-graph on",
            "--talker-cuda-graph on",
            "--talker-torch-compile on",
            "--thinker-max-running-requests 8",
            "SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184",
            "PREPROCESSING_MAX_CONCURRENCY=1",
        ],
    )
    vllm_switch_ok, vllm_switch_missing = _all_needles(
        vllm_switch_text,
        [
            "VLLM_ENABLE_TORCH_COMPILE=True",
            "enforce_eager=False",
            "FULL_AND_PIECEWISE CUDA graph",
            "enable_prefix_caching=True",
            "enable_chunked_prefill=True",
            "VLLM_HIDDEN_BUFFER_BACKEND=shm",
            "VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True",
            "VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=True",
            "VLLM_OMNI_ENABLE_ENCODER_BATCH=True",
            "--prebuild-prompts --prebuild-workers 4",
        ],
    )
    command_ids = _command_ids(repro_manifest)
    missing_commands = sorted(REQUIRED_COMMAND_IDS - command_ids)

    checks: list[ContractCheck] = [
        ContractCheck(
            "environment snapshot ready",
            _status(environment_ready),
            (
                f"environment_summary={environment_summary}; "
                f"preflight_pending_in_full_audit={environment_pending}"
            ),
        ),
        ContractCheck(
            "8x H20 CUDA environment captured",
            _status(
                bool(gpu.get("ok"))
                and _int_value(gpu.get("count")) >= 8
                and str(gpu.get("cuda_version") or "").startswith("12.")
                and all(
                    "H20" in str(row.get("name") or "")
                    for row in gpu.get("gpus", [])
                    if isinstance(row, dict)
                )
            ),
            (
                f"count={gpu.get('count')}, cuda={gpu.get('cuda_version')}, "
                f"first_gpu={(gpu.get('gpus') or [{}])[0].get('name') if gpu.get('gpus') else None}"
            ),
        ),
        ContractCheck(
            "SGLang image digest locked",
            _status(
                bool(sglang_image.get("ok"))
                and sglang_image.get("image") == SGLANG_IMAGE
                and sglang_image.get("id") == SGLANG_IMAGE_ID
                and sglang_summary.get("sglang_image_id") == SGLANG_IMAGE_ID
            ),
            (
                f"env={sglang_image.get('image')} {sglang_image.get('id')}; "
                f"lock={sglang_summary.get('sglang_image')} "
                f"{sglang_summary.get('sglang_image_id')}"
            ),
        ),
        ContractCheck(
            "vLLM image digest captured",
            _status(
                bool(vllm_image.get("ok"))
                and vllm_image.get("image") == VLLM_IMAGE
                and str(vllm_image.get("id") or "").startswith("sha256:")
                and vllm_summary.get("vllm_image") == VLLM_IMAGE
                and vllm_summary.get("vllm_image_id") == vllm_image.get("id")
            ),
            (
                f"env={vllm_image.get('image')} {vllm_image.get('id')}; "
                f"lock={vllm_summary.get('vllm_image')} "
                f"{vllm_summary.get('vllm_image_id')}"
            ),
        ),
        ContractCheck(
            "SGLang optimization lock ready",
            _status(
                bool(sglang_summary.get("ready"))
                and _int_value(sglang_summary.get("checks_total")) >= 26
                and _int_value(sglang_summary.get("required_failures"), 1) == 0
            ),
            f"sglang_optimization_lock={sglang_summary}",
        ),
        ContractCheck(
            "vLLM optimization lock ready",
            _status(
                bool(vllm_summary.get("ready"))
                and _int_value(vllm_summary.get("checks_total")) >= 22
                and _int_value(vllm_summary.get("required_failures"), 1) == 0
            ),
            f"vllm_optimization_lock={vllm_summary}",
        ),
        ContractCheck(
            "SGLang optimized recipe switches preserved",
            _status(sglang_switch_ok),
            "missing=" + ", ".join(sglang_switch_missing),
        ),
        ContractCheck(
            "vLLM optimized recipe switches preserved",
            _status(vllm_switch_ok),
            "missing=" + ", ".join(vllm_switch_missing),
        ),
        ContractCheck(
            "reproduction commands cover runtime recipes",
            _status(
                _int_value(repro_summary.get("commands_total")) >= 52
                and not missing_commands
            ),
            (
                f"commands={repro_summary.get('commands_total')}, "
                f"repro_ready={repro_summary.get('ready')}, "
                f"missing={missing_commands}"
            ),
        ),
        ContractCheck(
            "SGLang serving scope explicit",
            _status(
                "c4-c8" in str(sglang_summary.get("recommended_window") or "")
                and "c16" in str(sglang_summary.get("recommended_window") or "")
            ),
            f"recommended_window={sglang_summary.get('recommended_window')}",
        ),
        ContractCheck(
            "vLLM strict c4 scope explicit",
            _status(
                "c4" in str(vllm_summary.get("strict_c4_contract") or "")
                and "headline" in str(vllm_summary.get("strict_c4_contract") or "")
            ),
            f"strict_c4_contract={vllm_summary.get('strict_c4_contract')}",
        ),
        ContractCheck(
            "vLLM c8 online caveat explicit",
            _status(
                bool(online_summary.get("ready"))
                and bool(online_summary.get("current_package_safe"))
                and not bool(online_summary.get("online_parity_proven"))
                and "do_not_promote" in str(online_summary.get("upgrade_decision") or "")
            ),
            f"vllm_online_parity_protocol={online_summary}",
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
            "sglang_image": sglang_image.get("image"),
            "sglang_image_id": sglang_image.get("id"),
            "vllm_image": vllm_image.get("image"),
            "vllm_image_id": vllm_image.get("id"),
            "gpu_contract": f"{gpu.get('count')}x {((gpu.get('gpus') or [{}])[0]).get('name') if gpu.get('gpus') else 'unknown'} / CUDA {gpu.get('cuda_version')}",
            "sglang_scope": sglang_summary.get("recommended_window"),
            "vllm_strict_scope": vllm_summary.get("strict_c4_contract"),
            "vllm_c8_scope": vllm_summary.get("c8_contract"),
            "environment_pending_in_full_audit": environment_pending,
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "runtime_rows": [
            {
                "runtime": "SGLang-Omni",
                "image": sglang_image.get("image"),
                "image_id": sglang_image.get("id"),
                "created": sglang_image.get("created"),
                "optimization_gate": f"{sglang_summary.get('checks_passed')}/{sglang_summary.get('checks_total')}",
                "claim_scope": sglang_summary.get("recommended_window"),
            },
            {
                "runtime": "vLLM-Omni",
                "image": vllm_image.get("image"),
                "image_id": vllm_image.get("id"),
                "created": vllm_image.get("created"),
                "optimization_gate": f"{vllm_summary.get('checks_passed')}/{vllm_summary.get('checks_total')}",
                "claim_scope": (
                    f"{vllm_summary.get('strict_c4_contract')}; "
                    f"{vllm_summary.get('c8_contract')}"
                ),
            },
        ],
        "sglang_required_switches": sglang_lock.get("recipe_switches", []),
        "vllm_required_switches": vllm_lock.get("required_switches", []),
        "reproduction_command_ids": sorted(command_ids),
        "required_reproduction_command_ids": sorted(REQUIRED_COMMAND_IDS),
        "source_files": {
            "environment_snapshot": str(audit_dir / "environment_snapshot.json"),
            "sglang_optimization_lock": str(audit_dir / "sglang_optimization_lock.json"),
            "vllm_optimization_lock": str(audit_dir / "vllm_optimization_lock.json"),
            "vllm_online_parity_protocol": str(
                audit_dir / "vllm_online_parity_protocol.json"
            ),
            "repro_command_manifest": str(audit_dir / "repro_command_manifest.json"),
        },
        "handoff_rules": [
            "If either image digest changes, rerun the relevant optimization lock and full audit before replacing headline numbers.",
            "If any required optimization switch is removed, treat that run as a new baseline instead of the current optimized contract.",
            "SGLang c4-c8 is the current serving window; c16 is saturation evidence, not the recommended operating point.",
            "vLLM c8 prebuild w4 is an optimized offline diagnostic until online ingress, WER/ASR, and engine/talker boundary artifacts are collected.",
        ],
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
        "# Qwen3.5-Omni Runtime Image Contract",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这页把 SGLang-Omni 与 vLLM-Omni 的镜像、digest、硬件环境、优化开关和",
        "可对外声明边界合并成一个 handoff contract。它不新增 benchmark 口径，",
        "只把已有审计证据压成 reviewer 可以快速核对的一页。",
        "",
        "## 1. 结论",
        "",
        f"- ready：`{summary['ready']}`，checks：`{summary['checks_passed']}/{summary['checks_total']}`，required failures：`{summary['required_failures']}`。",
        f"- GPU contract：`{summary['gpu_contract']}`。",
        f"- SGLang image：`{summary['sglang_image']}`。",
        f"- SGLang image id：`{summary['sglang_image_id']}`。",
        f"- vLLM image：`{summary['vllm_image']}`。",
        f"- vLLM image id：`{summary['vllm_image_id']}`。",
        f"- SGLang scope：`{summary['sglang_scope']}`。",
        f"- vLLM strict scope：`{summary['vllm_strict_scope']}`。",
        f"- vLLM c8 scope：`{summary['vllm_c8_scope']}`。",
        "",
        "## 2. Runtime Matrix",
        "",
        "| Runtime | Image | Image ID | Created | Optimization Gate | Claim Scope |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in payload["runtime_rows"]:
        lines.append(
            "| "
            f"{row['runtime']} | "
            f"`{row['image']}` | "
            f"`{row['image_id']}` | "
            f"`{row['created']}` | "
            f"{row['optimization_gate']} | "
            f"{row['claim_scope']} |"
        )
    lines.extend(["", "## 3. Gate 明细", ""])
    lines.extend(_gate_table(payload["checks"]))
    lines.extend(
        [
            "",
            "## 4. SGLang 必须保留的优化开关",
            "",
            "| Switch | 用途 | 来源 |",
            "| --- | --- | --- |",
        ]
    )
    for item in payload["sglang_required_switches"]:
        lines.append(
            f"| `{item.get('switch')}` | {item.get('purpose')} | `{item.get('source')}` |"
        )
    lines.extend(
        [
            "",
            "## 5. vLLM 必须保留的优化开关",
            "",
            "| Switch / evidence | 用途 | 来源 |",
            "| --- | --- | --- |",
        ]
    )
    for item in payload["vllm_required_switches"]:
        lines.append(
            f"| `{item.get('switch')}` | {item.get('purpose')} | `{item.get('source')}` |"
        )
    lines.extend(
        [
            "",
            "## 6. 对外声明规则",
            "",
        ]
    )
    for rule in payload["handoff_rules"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "## 7. 复现命令覆盖",
            "",
            "本 contract 要求以下 reproduction command ids 存在于 `repro_command_manifest.json`：",
            "",
        ]
    )
    for command_id in payload["required_reproduction_command_ids"]:
        lines.append(f"- `{command_id}`")
    lines.extend(["", "## 8. 机器证据", ""])
    for name, path in payload["source_files"].items():
        lines.append(f"- {name}：`{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni runtime image/optimization contract."
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
        "Runtime image contract written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    print(f"Runtime image contract JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
