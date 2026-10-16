# SPDX-License-Identifier: Apache-2.0
"""Build an SGLang optimization lock matrix for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "sglang_optimization_lock.json"

SGLANG_IMAGE = "frankleeeee/sglang-omni:dev"
SGLANG_IMAGE_ID = "sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84"
LAUNCH_SCRIPT = Path("examples/launch_qwen35_omni_speech_server_container.sh")
MAIN_REPORT = Path("benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md")
REPRO_CHECKLIST = Path(
    "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md"
)


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


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _row_by_key(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def _accepted_pressure(rows: list[dict[str, Any]], pressure: str) -> bool:
    for row in rows:
        if row.get("pressure") == pressure:
            return bool(row.get("accepted")) and row.get("evidence_status") == "PASS"
    return False


def _stress_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get("sglang_stress", [])
    return rows if isinstance(rows, list) else []


def _stage_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get("sglang_stage_breakdown", [])
    return rows if isinstance(rows, list) else []


def _preproc_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get("sglang_preprocessing_split", [])
    return rows if isinstance(rows, list) else []


def _synthetic_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get("synthetic_speech", [])
    return rows if isinstance(rows, list) else []


def _synthetic_stage_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get("synthetic_stage_breakdown", [])
    return rows if isinstance(rows, list) else []


def _preproc_negative_rows(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get("preprocessing_concurrency", [])
    return rows if isinstance(rows, list) else []


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    environment = _load_json_optional(audit_dir / "environment_snapshot.json")
    tables = _load_json_optional(audit_dir / "tables_summary.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    stage_interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")

    launch_text = _read_text_optional(root / LAUNCH_SCRIPT)
    main_report_text = _read_text_optional(root / MAIN_REPORT)
    repro_text = _read_text_optional(root / REPRO_CHECKLIST)

    image = environment.get("docker_images", {}).get("sglang", {})
    stress = _stress_rows(tables)
    stages = _stage_rows(tables)
    preproc_split = _preproc_rows(tables)
    synthetic = _synthetic_rows(tables)
    synthetic_stages = _synthetic_stage_rows(tables)
    preproc_negative = _preproc_negative_rows(tables)
    acceptance_rows = acceptance.get("rows", [])
    stage_summary = stage_interactions.get("summary", {})

    c4 = _row_by_key(stress, "concurrency", 4)
    c8 = _row_by_key(stress, "concurrency", 8)
    c16 = _row_by_key(stress, "concurrency", 16)
    preproc1 = _row_by_key(preproc_negative, "setting", "preproc=1 baseline")
    preproc2 = _row_by_key(preproc_negative, "setting", "preproc=2")
    c8_split = _row_by_key(preproc_split, "concurrency", 8)
    c16_split = _row_by_key(preproc_split, "concurrency", 16)
    long_c8 = next(
        (
            row
            for row in synthetic
            if row.get("scenario") == "long" and row.get("concurrency") == 8
        ),
        {},
    )

    checks: list[LockCheck] = [
        LockCheck(
            name="SGLang image locked",
            status=_status(
                bool(image.get("ok"))
                and image.get("image") == SGLANG_IMAGE
                and image.get("id") == SGLANG_IMAGE_ID
            ),
            evidence=(
                f"image={image.get('image')}, id={image.get('id')}, "
                f"created={image.get('created')}"
            ),
        ),
        LockCheck(
            name="launch script present",
            status=_status((root / LAUNCH_SCRIPT).is_file()),
            evidence=str(root / LAUNCH_SCRIPT),
        ),
        _needle_check(
            name="launch defaults to serial preprocessing",
            text=launch_text,
            needle='PREPROCESSING_MAX_CONCURRENCY="${PREPROCESSING_MAX_CONCURRENCY:-1}"',
            source=str(LAUNCH_SCRIPT),
        ),
        _needle_check(
            name="launch exposes code2wav compile toggle",
            text=launch_text,
            needle='NO_CODE2WAV_TORCH_COMPILE="${NO_CODE2WAV_TORCH_COMPILE:-1}"',
            source=str(LAUNCH_SCRIPT),
        ),
        _needle_check(
            name="launch exposes TorchDynamo override",
            text=launch_text,
            needle='export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"',
            source=str(LAUNCH_SCRIPT),
        ),
        _needle_check(
            name="launch appends EXTRA_ARGS",
            text=launch_text,
            needle='server_args+=("${extra_args_array[@]}")',
            source=str(LAUNCH_SCRIPT),
        ),
        _needle_check(
            name="performance recipe keeps code2wav compile on",
            text=main_report_text + "\n" + repro_text,
            needle="NO_CODE2WAV_TORCH_COMPILE=0",
            source="main report / reproduction checklist",
        ),
        _needle_check(
            name="performance recipe keeps TorchDynamo on",
            text=main_report_text + "\n" + repro_text,
            needle="TORCHDYNAMO_DISABLE=0",
            source="main report / reproduction checklist",
        ),
        _needle_check(
            name="performance recipe sets video preprocessing cache",
            text=main_report_text + "\n" + repro_text,
            needle="SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184",
            source="main report / reproduction checklist",
        ),
        _needle_check(
            name="performance recipe enables Thinker CUDA graph",
            text=main_report_text + "\n" + repro_text,
            needle="--thinker-cuda-graph on",
            source="main report / reproduction checklist",
        ),
        _needle_check(
            name="performance recipe enables Talker CUDA graph",
            text=main_report_text + "\n" + repro_text,
            needle="--talker-cuda-graph on",
            source="main report / reproduction checklist",
        ),
        _needle_check(
            name="performance recipe enables Talker torch compile",
            text=main_report_text + "\n" + repro_text,
            needle="--talker-torch-compile on",
            source="main report / reproduction checklist",
        ),
        _needle_check(
            name="performance recipe uses max-running 8",
            text=main_report_text + "\n" + repro_text + "\n" + launch_text,
            needle="--thinker-max-running-requests 8 --talker-max-running-requests 8",
            source="main report / reproduction checklist / launch script",
        ),
        LockCheck(
            name="stress sweep covers c1/c2/c4/c8/c16",
            status=_status({row.get("concurrency") for row in stress} >= {1, 2, 4, 8, 16}),
            evidence=f"concurrency={sorted(row.get('concurrency') for row in stress)}",
        ),
        LockCheck(
            name="stress quality is stable",
            status=_status(
                len(stress) >= 5
                and all(_float_value(row.get("accuracy")) >= 0.70 for row in stress)
                and all(_float_value(row.get("wer_corpus")) <= 0.04 for row in stress)
            ),
            evidence=(
                "acc="
                + ",".join(
                    f"c{row.get('concurrency')}={_fmt_num(row.get('accuracy'), 3)}"
                    for row in stress
                )
                + "; WER="
                + ",".join(
                    f"c{row.get('concurrency')}={_fmt_num(row.get('wer_corpus'), 4)}"
                    for row in stress
                )
            ),
        ),
        LockCheck(
            name="c8 is the current throughput peak",
            status=_status(
                _float_value(c8.get("throughput_qps"))
                > _float_value(c4.get("throughput_qps"))
                and _float_value(c8.get("throughput_qps"))
                > _float_value(c16.get("throughput_qps"))
            ),
            evidence=(
                f"c4={_fmt_num(c4.get('throughput_qps'))}, "
                f"c8={_fmt_num(c8.get('throughput_qps'))}, "
                f"c16={_fmt_num(c16.get('throughput_qps'))} QPS"
            ),
        ),
        LockCheck(
            name="c16 is a saturation boundary",
            status=_status(
                _float_value(c16.get("throughput_qps"))
                < _float_value(c8.get("throughput_qps"))
                and _float_value(c16.get("latency_mean_s"))
                > _float_value(c8.get("latency_mean_s"))
            ),
            evidence=(
                f"c8_qps={_fmt_num(c8.get('throughput_qps'))}, "
                f"c16_qps={_fmt_num(c16.get('throughput_qps'))}, "
                f"c8_lat={_fmt_num(c8.get('latency_mean_s'), 3)}s, "
                f"c16_lat={_fmt_num(c16.get('latency_mean_s'), 3)}s"
            ),
        ),
        LockCheck(
            name="code2wav decode is not the stress bottleneck",
            status=_status(
                stages
                and max(_float_value(row.get("code2wav_decode_p95_ms")) for row in stages)
                <= 30.0
            ),
            evidence=(
                "decode_p95="
                + ",".join(
                    f"c{row.get('concurrency')}={_fmt_ms(row.get('code2wav_decode_p95_ms'))}"
                    for row in stages
                )
            ),
        ),
        LockCheck(
            name="talker to code2wav stream handoff is healthy",
            status=_status(
                stages
                and max(
                    _float_value(row.get("talker_to_code2wav_hop_p95_ms"))
                    for row in stages
                )
                <= 25.0
            ),
            evidence=(
                "hop_p95="
                + ",".join(
                    f"c{row.get('concurrency')}={_fmt_ms(row.get('talker_to_code2wav_hop_p95_ms'))}"
                    for row in stages
                )
            ),
        ),
        LockCheck(
            name="preprocessing compute stays stable while lifecycle queues",
            status=_status(
                preproc_split
                and max(
                    _float_value(row.get("actual_preprocess_avg_ms"))
                    for row in preproc_split
                )
                <= 320.0
                and (
                    _float_value(c8_split.get("preproc_stage_avg_ms"))
                    - _float_value(c8_split.get("actual_preprocess_avg_ms"))
                )
                >= 900.0
                and (
                    _float_value(c16_split.get("preproc_stage_avg_ms"))
                    - _float_value(c16_split.get("actual_preprocess_avg_ms"))
                )
                >= 4000.0
            ),
            evidence=(
                f"actual_avg_max={_fmt_ms(max((_float_value(row.get('actual_preprocess_avg_ms')) for row in preproc_split), default=0.0))}, "
                f"c8_queue_gap={_fmt_ms(_float_value(c8_split.get('preproc_stage_avg_ms')) - _float_value(c8_split.get('actual_preprocess_avg_ms')))}, "
                f"c16_queue_gap={_fmt_ms(_float_value(c16_split.get('preproc_stage_avg_ms')) - _float_value(c16_split.get('actual_preprocess_avg_ms')))}"
            ),
        ),
        LockCheck(
            name="preproc=2 is locked as a negative optimization",
            status=_status(
                _float_value(preproc2.get("throughput_qps"))
                < _float_value(preproc1.get("throughput_qps")) * 0.8
                and _float_value(preproc2.get("latency_mean_s"))
                > _float_value(preproc1.get("latency_mean_s"))
            ),
            evidence=(
                f"baseline_qps={_fmt_num(preproc1.get('throughput_qps'))}, "
                f"preproc2_qps={_fmt_num(preproc2.get('throughput_qps'))}, "
                f"baseline_lat={_fmt_num(preproc1.get('latency_mean_s'), 3)}s, "
                f"preproc2_lat={_fmt_num(preproc2.get('latency_mean_s'), 3)}s"
            ),
        ),
        LockCheck(
            name="preproc=4 failure boundary is locked",
            status=_status(
                _accepted_pressure(
                    acceptance_rows, "PREPROCESSING_MAX_CONCURRENCY=4 at c=8"
                )
            ),
            evidence="acceptance matrix row PREPROCESSING_MAX_CONCURRENCY=4 at c=8 is PASS",
        ),
        LockCheck(
            name="synthetic short and long speech regimes are covered",
            status=_status(
                {
                    (row.get("scenario"), row.get("concurrency"))
                    for row in synthetic
                }
                >= {
                    ("short", 1),
                    ("short", 4),
                    ("short", 8),
                    ("long", 1),
                    ("long", 4),
                    ("long", 8),
                }
            ),
            evidence=f"synthetic_rows={len(synthetic)}",
        ),
        LockCheck(
            name="long synthetic c8 remains faster than real time",
            status=_status(
                _float_value(long_c8.get("audio_duration_mean_s")) >= 50.0
                and _float_value(long_c8.get("rtf_mean")) < 0.50
            ),
            evidence=(
                f"audio={_fmt_num(long_c8.get('audio_duration_mean_s'), 1)}s, "
                f"lat={_fmt_num(long_c8.get('latency_mean_s'), 3)}s, "
                f"rtf={_fmt_num(long_c8.get('rtf_mean'))}"
            ),
        ),
        LockCheck(
            name="synthetic speech handoff remains healthy",
            status=_status(
                synthetic_stages
                and max(
                    _float_value(row.get("talker_to_code2wav_hop_p95_ms"))
                    for row in synthetic_stages
                )
                <= 25.0
                and max(
                    _float_value(row.get("code2wav_decode_p95_ms"))
                    for row in synthetic_stages
                )
                <= 25.0
            ),
            evidence=(
                f"max_hop_p95={_fmt_ms(max((_float_value(row.get('talker_to_code2wav_hop_p95_ms')) for row in synthetic_stages), default=0.0))}, "
                f"max_decode_p95={_fmt_ms(max((_float_value(row.get('code2wav_decode_p95_ms')) for row in synthetic_stages), default=0.0))}"
            ),
        ),
        LockCheck(
            name="stage interaction flags are locked",
            status=_status(
                bool(stage_summary.get("sglang_talker_to_code2wav_healthy"))
                and bool(stage_summary.get("sglang_code2wav_decode_not_bottleneck"))
                and bool(stage_summary.get("preprocessing_parallelism_regresses"))
            ),
            evidence=f"stage_interactions={stage_summary}",
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
            "sglang_image": image.get("image"),
            "sglang_image_id": image.get("id"),
            "recommended_window": "c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary",
            "recipe_contract": "compiled/graph SGLang recipe with serial preprocessing and 16GiB preprocessing cache",
        },
        "checks": [check.to_dict() for check in checks],
        "required_failures": [check.to_dict() for check in required_failures],
        "recipe_switches": [
            {
                "switch": "NO_CODE2WAV_TORCH_COMPILE=0; TORCHDYNAMO_DISABLE=0",
                "purpose": "keep code2wav compile path enabled for performance runs",
                "source": str(MAIN_REPORT),
            },
            {
                "switch": "--thinker-cuda-graph on; --talker-cuda-graph on; --talker-torch-compile on",
                "purpose": "lock warmed Thinker/Talker graph and compile path",
                "source": str(MAIN_REPORT),
            },
            {
                "switch": "--thinker-max-running-requests 8; --talker-max-running-requests 8",
                "purpose": "expose the current c4-c8 operating window and c16 saturation boundary",
                "source": str(MAIN_REPORT),
            },
            {
                "switch": "SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184; SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64",
                "purpose": "stabilize repeated Video-AMME preprocessing during stress sweeps",
                "source": str(MAIN_REPORT),
            },
            {
                "switch": "PREPROCESSING_MAX_CONCURRENCY=1",
                "purpose": "current safe admission point; wider preprocessing is a measured anti-recipe",
                "source": str(LAUNCH_SCRIPT),
            },
        ],
        "stress_rows": stress,
        "stage_rows": stages,
        "source_files": {
            "environment_snapshot": str(audit_dir / "environment_snapshot.json"),
            "tables_summary": str(audit_dir / "tables_summary.json"),
            "claims_verification": str(audit_dir / "claims_verification.json"),
            "acceptance_matrix": str(audit_dir / "acceptance_matrix.json"),
            "stage_interaction_summary": str(audit_dir / "stage_interaction_summary.json"),
            "launch_script": str(root / LAUNCH_SCRIPT),
            "main_report": str(root / MAIN_REPORT),
            "reproduction_checklist": str(root / REPRO_CHECKLIST),
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
        "# Qwen3.5-Omni SGLang 优化锁定矩阵",
        "",
        f"生成时间 UTC：`{payload['generated_at_utc']}`。",
        f"工作目录：`{payload['root']}`。",
        "",
        "这页专门锁定 SGLang-Omni 当前 best recipe、推荐运行窗口、stage",
        "连接健康度和反例实验。目的不是新增 benchmark 数字，而是防止 reviewer",
        "误以为 SGLang 侧只是偶然跑快，或把负优化配置误当成下一步主线。",
        "",
        "## 1. 锁定结论",
        "",
        f"- ready：`{summary['ready']}`，checks：`{summary['checks_passed']}/{summary['checks_total']}`，required failures：`{summary['required_failures']}`。",
        f"- SGLang image：`{summary['sglang_image']}`。",
        f"- SGLang image id：`{summary['sglang_image_id']}`。",
        f"- 推荐窗口：{summary['recommended_window']}。",
        f"- recipe contract：{summary['recipe_contract']}。",
        "- c=8 是当前吞吐峰值；c=16 是饱和边界，不作为推荐 serving 点。",
        "- `PREPROCESSING_MAX_CONCURRENCY=2/4` 已被锁为反例，不应作为默认优化方向。",
        "",
        "## 2. Gate 明细",
        "",
    ]
    lines.extend(_gate_table(payload["checks"]))
    lines.extend(
        [
            "",
            "## 3. 必须锁定的 SGLang 优化开关",
            "",
            "| Switch / evidence | 用途 | 来源 |",
            "| --- | --- | --- |",
        ]
    )
    for item in payload["recipe_switches"]:
        lines.append(
            f"| `{item['switch']}` | {item['purpose']} | `{item['source']}` |"
        )
    lines.extend(
        [
            "",
            "## 4. SGLang 压力窗口锁定表",
            "",
            "| c | Accuracy | WER | Lat mean | Lat p95 | RTF mean | QPS | Audio throughput | 使用边界 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["stress_rows"]:
        c = row.get("concurrency")
        boundary = (
            "recommended serving window"
            if c in {1, 2, 4}
            else "current peak throughput"
            if c == 8
            else "saturation boundary"
        )
        lines.append(
            "| "
            f"{c} | "
            f"{_fmt_num(_float_value(row.get('accuracy')) * 100, 1)}% | "
            f"{_fmt_num(_float_value(row.get('wer_corpus')) * 100, 2)}% | "
            f"{_fmt_num(row.get('latency_mean_s'), 3)}s | "
            f"{_fmt_num(row.get('latency_p95_s'), 3)}s | "
            f"{_fmt_num(row.get('rtf_mean'))} | "
            f"{_fmt_num(row.get('throughput_qps'))} | "
            f"{_fmt_num(row.get('audio_throughput_s_per_s'))} | "
            f"{boundary} |"
        )
    lines.extend(
        [
            "",
            "## 5. Stage 连接锁定表",
            "",
            "| c | Top stage | Preproc lifecycle avg/p95 | Talker avg/p95 | Code2wav decode avg/p95 | Talker->Code2wav hop p95 | 解释 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["stage_rows"]:
        c = row.get("concurrency")
        explanation = (
            "talker tail dominates"
            if c in {1, 2, 4}
            else "preprocessing lifecycle queue joins the tail"
            if c == 8
            else "queue-dominated saturation"
        )
        lines.append(
            "| "
            f"{c} | "
            f"{row.get('top_stage')} | "
            f"{_fmt_ms(row.get('preproc_stage_avg_ms'))}/{_fmt_ms(row.get('preproc_stage_p95_ms'))} | "
            f"{_fmt_ms(row.get('talker_avg_ms'))}/{_fmt_ms(row.get('talker_p95_ms'))} | "
            f"{_fmt_ms(row.get('code2wav_decode_avg_ms'))}/{_fmt_ms(row.get('code2wav_decode_p95_ms'))} | "
            f"{_fmt_ms(row.get('talker_to_code2wav_hop_p95_ms'))} | "
            f"{explanation} |"
        )
    lines.extend(
        [
            "",
            "## 6. 对外使用规则",
            "",
            "- 可以说：SGLang 当前 best recipe 是 compiled/graph path + 16GiB preprocessing cache + serial preprocessing admission。",
            "- 可以说：在当前 8x H20、Video-AMME ci-50、warmed pressure sweep 下，c=8 是吞吐峰值，c=16 是压力边界。",
            "- 可以说：stage 间连接不是主要瓶颈；主要瓶颈来自 high-concurrency preprocessing lifecycle queue 和 talker AR tail。",
            "- 禁止说：简单扩大 preprocessing 并发能优化当前 recipe；preproc=2/4 已经被反例锁定。",
            "- 禁止把没有通过本页 JSON、claims、coverage、preflight、final readiness 的新 SGLang 数字替换进报告。",
            "",
            "## 7. 机器证据",
            "",
        ]
    )
    for name, path in payload["source_files"].items():
        lines.append(f"- {name}：`{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni SGLang optimization lock matrix."
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
        "SGLang optimization lock written: "
        f"{output} ready={payload['summary']['ready']} "
        f"checks={payload['summary']['checks_passed']}/"
        f"{payload['summary']['checks_total']}"
    )
    print(f"SGLang optimization lock JSON written: {json_output}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
