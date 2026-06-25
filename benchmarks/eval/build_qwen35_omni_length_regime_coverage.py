# SPDX-License-Identifier: Apache-2.0
"""Build the short/long length-regime coverage matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "length_regime_coverage.json"

SHORT_CHARS = 74.0
SHORT_WORDS = 12.0
LONG_CHARS = 944.0
LONG_WORDS = 139.0
REQUIRED_CONCURRENCY = {1, 4, 8}


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


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


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


def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("\n", " ").replace("|", "\\|")


def _num(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _ms(value: Any) -> str:
    return _num(value, 1, "ms")


def _stage_budget_map(stage_budget: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in stage_budget.get("synthetic_speech_budget", []):
        if not isinstance(row, dict):
            continue
        scenario = str(row.get("scenario") or "")
        concurrency = _int(row.get("concurrency"))
        if scenario and concurrency:
            result[(scenario, concurrency)] = row
    return result


def _csv_scenario_rows(rows: list[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("scenario") or "") == scenario
    ]


def _shape_ok(rows: list[dict[str, Any]], scenario: str, chars: float, words: float) -> bool:
    subset = _csv_scenario_rows(rows, scenario)
    if len(subset) != len(REQUIRED_CONCURRENCY):
        return False
    return all(
        _float(row.get("target_chars")) == chars
        and _float(row.get("target_words")) == words
        for row in subset
    )


def _synthetic_rows(
    csv_rows: list[dict[str, Any]], stage_budget: dict[str, Any]
) -> list[dict[str, Any]]:
    budget_by_key = _stage_budget_map(stage_budget)
    scenario_order = {"short": 0, "long": 1}
    sorted_rows = sorted(
        csv_rows,
        key=lambda row: (
            scenario_order.get(str(row.get("scenario") or ""), 99),
            _int(row.get("concurrency")),
        ),
    )
    result: list[dict[str, Any]] = []
    for row in sorted_rows:
        scenario = str(row.get("scenario") or "")
        concurrency = _int(row.get("concurrency"))
        budget = budget_by_key.get((scenario, concurrency), {})
        is_long = scenario == "long"
        result.append(
            {
                "coverage_id": f"synthetic_{scenario}_c{concurrency}",
                "workload": f"synthetic {scenario} text-to-speech",
                "concurrency": concurrency,
                "input_shape": (
                    f"{_int(row.get('target_chars'))} chars / "
                    f"{_int(row.get('target_words'))} words"
                ),
                "output_shape": f"audio_mean={_num(row.get('audio_duration_mean_s'), 1, 's')}",
                "stage_pressure": (
                    "long-form Talker AR cadence and chunk cadence"
                    if is_long
                    else "short-output thinker/talker/code2wav guard"
                ),
                "key_metrics": (
                    f"lat_mean={_num(row.get('latency_mean_s'), 3, 's')}, "
                    f"lat_p95={_num(row.get('latency_p95_s'), 3, 's')}, "
                    f"RTF={_num(row.get('rtf_mean'), 4)}, "
                    f"RTF_p95={_num(row.get('rtf_p95'), 4)}, "
                    f"QPS={_num(row.get('throughput_qps'), 3)}"
                ),
                "stage_metrics": (
                    f"talker_pct={_num(budget.get('talker_pct_of_latency'), 1, '%')}, "
                    f"hop_p95={_ms(row.get('talker_to_code2wav_hop_p95_ms'))}, "
                    f"decode_p95={_ms(row.get('code2wav_decode_p95_ms'))}"
                ),
                "allowed_claim": (
                    "long c=8 remains faster than real time"
                    if is_long and concurrency == 8
                    else "short/long speech guard remains faster than real time"
                ),
                "boundary": (
                    "synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic"
                ),
                "evidence_files": [
                    "results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv",
                    "results/qwen35_report_audit_20260619/stage_latency_budget.json",
                    str(row.get("result_json") or ""),
                    str(row.get("profile_json") or ""),
                ],
            }
        )
    return result


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    video_meta = _load_json_optional(audit_dir / "videoamme_seedtts_meta_summary.json")
    stage_budget = _load_json_optional(audit_dir / "stage_latency_budget.json")
    regime = _load_json_optional(audit_dir / "regime_decision_matrix.json")
    csv_path = audit_dir / "share_charts/synthetic_short_long_speech.csv"
    csv_rows = _read_csv_rows(csv_path)

    synthetic = _synthetic_rows(csv_rows, stage_budget)
    target_chars = video_meta.get("target_chars", {})
    audio_duration = video_meta.get("audio_duration_s", {})
    video_row = {
        "coverage_id": "videoamme_ci50_real_multimodal",
        "workload": "Video-AMME ci-50 video + spoken question -> text + speech",
        "concurrency": "1/2/4/8/16",
        "input_shape": (
            f"target_chars={_int(target_chars.get('min'))}-"
            f"{_int(target_chars.get('max'))}, mean={_num(target_chars.get('mean'), 1)}"
        ),
        "output_shape": (
            f"audio_duration={_num(audio_duration.get('min'), 1, 's')}-"
            f"{_num(audio_duration.get('max'), 1, 's')}, "
            f"mean={_num(audio_duration.get('mean'), 1, 's')}"
        ),
        "stage_pressure": "real multimodal preprocessing, encoder, Talker, code2wav, and admission queue",
        "key_metrics": "SGLang stress c=1/2/4/8/16; c=8 current throughput peak; c=16 saturation boundary",
        "stage_metrics": "stage latency budget and stage boundary ledger carry per-stage percentages",
        "allowed_claim": "Video-AMME ci-50 anchors the main single/high-concurrency performance shape",
        "boundary": "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
        "evidence_files": [
            "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json",
            "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
        ],
    }
    rows = [video_row, *synthetic]

    synthetic_concurrency = {
        _int(row.get("concurrency"))
        for row in csv_rows
        if str(row.get("scenario") or "") in {"short", "long"}
    }
    long_c8 = next(
        (
            row
            for row in csv_rows
            if row.get("scenario") == "long" and _int(row.get("concurrency")) == 8
        ),
        {},
    )
    max_hop = max((_float(row.get("talker_to_code2wav_hop_p95_ms")) for row in csv_rows), default=0.0)
    max_decode = max((_float(row.get("code2wav_decode_p95_ms")) for row in csv_rows), default=0.0)
    stage_synthetic_rows = stage_budget.get("synthetic_speech_budget", [])
    checks = {
        "videoamme_meta_ready": _int(video_meta.get("rows")) == 50
        and _int(video_meta.get("skipped_missing_audio")) == 0
        and _float(target_chars.get("min")) > 0
        and _float(target_chars.get("max")) > _float(target_chars.get("min"))
        and _float(audio_duration.get("mean")) > 0,
        "synthetic_csv_rows_ready": len(csv_rows) == 6,
        "synthetic_short_shape_fixed": _shape_ok(csv_rows, "short", SHORT_CHARS, SHORT_WORDS),
        "synthetic_long_shape_fixed": _shape_ok(csv_rows, "long", LONG_CHARS, LONG_WORDS),
        "synthetic_concurrency_coverage": REQUIRED_CONCURRENCY.issubset(synthetic_concurrency),
        "synthetic_all_faster_than_realtime": bool(csv_rows)
        and all(_float(row.get("rtf_p95")) < 1.0 for row in csv_rows),
        "long_c8_faster_than_realtime": _float(long_c8.get("rtf_mean")) < 1.0
        and _float(long_c8.get("rtf_p95")) < 1.0,
        "handoff_decode_guard": max_hop <= 30.0 and max_decode <= 30.0,
        "stage_budget_synthetic_rows": isinstance(stage_synthetic_rows, list)
        and len(stage_synthetic_rows) >= 6,
        "regime_decision_links_short_long": {
            "short c=1",
            "short c=4",
            "short c=8",
            "long c=1",
            "long c=4",
            "long c=8",
        }.issubset(
            {
                str(row.get("pressure") or "")
                for row in regime.get("rows", [])
                if isinstance(row, dict)
                and row.get("regime") == "sglang_synthetic_speech"
            }
        ),
    }
    required_failures = sum(1 for value in checks.values() if not value)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": required_failures == 0,
            "checks_total": len(checks),
            "checks_passed": sum(1 for value in checks.values() if value),
            "required_failures": required_failures,
            "rows_total": len(rows),
            "videoamme_rows": 1,
            "synthetic_rows": len(synthetic),
            "short_rows": len(_csv_scenario_rows(csv_rows, "short")),
            "long_rows": len(_csv_scenario_rows(csv_rows, "long")),
            "videoamme_target_chars_min": target_chars.get("min"),
            "videoamme_target_chars_max": target_chars.get("max"),
            "videoamme_audio_duration_mean_s": audio_duration.get("mean"),
            "short_target_chars": SHORT_CHARS,
            "long_target_chars": LONG_CHARS,
            "long_c8_rtf_mean": _float(long_c8.get("rtf_mean")),
            "long_c8_rtf_p95": _float(long_c8.get("rtf_p95")),
            "max_talker_to_code2wav_hop_p95_ms": max_hop,
            "max_code2wav_decode_p95_ms": max_decode,
            "entry_scope_redline": "ci-50/stress/synthetic 证据不能直接外推到完整线上流量",
            "share_scope": (
                "Standalone short/long length-regime coverage matrix tying Video-AMME "
                "ci-50 target length, synthetic short/long fixed prompts, stage pressure, "
                "handoff/decode guardrails, and non-extrapolation boundaries."
            ),
        },
        "checks": checks,
        "rows": rows,
        "sources": {
            "videoamme_seedtts_meta_summary": video_meta,
            "stage_latency_budget": stage_budget.get("summary", {}),
            "regime_decision_matrix": regime.get("summary", {}),
            "synthetic_csv": str(csv_path),
        },
    }


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines: list[str] = [
        "# Qwen3.5-Omni 长短输入/输出 Length-Regime 覆盖矩阵",
        "",
        "## 1. 结论",
        "",
        (
            f"- Video-AMME ci-50 覆盖真实多模态样本：目标文本 "
            f"{_int(summary['videoamme_target_chars_min'])}-"
            f"{_int(summary['videoamme_target_chars_max'])} chars，平均音频 "
            f"{_num(summary['videoamme_audio_duration_mean_s'], 1, 's')}。"
        ),
        (
            f"- Synthetic short/long speech 覆盖固定短输入 "
            f"{_int(summary['short_target_chars'])} chars / 12 words 和固定长输入 "
            f"{_int(summary['long_target_chars'])} chars / 139 words，c=1/4/8 全部保留。"
        ),
        (
            f"- long c=8 仍快于实时：RTF mean={_num(summary['long_c8_rtf_mean'], 4)}，"
            f"RTF p95={_num(summary['long_c8_rtf_p95'], 4)}。"
        ),
        (
            f"- Handoff/decode guard 仍健康：最大 talker->code2wav hop p95="
            f"{_ms(summary['max_talker_to_code2wav_hop_p95_ms'])}，最大 code2wav decode p95="
            f"{_ms(summary['max_code2wav_decode_p95_ms'])}。"
        ),
        "- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑和 gate 全绿。",
        "",
        "## 2. 覆盖矩阵",
        "",
        "| 覆盖项 | 并发 | 输入形状 | 输出形状 | Stage 压力 | 关键指标 | Stage 读法 | 可说结论 | 边界 |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row["workload"]),
                    _cell(row["concurrency"]),
                    _cell(row["input_shape"]),
                    _cell(row["output_shape"]),
                    _cell(row["stage_pressure"]),
                    _cell(row["key_metrics"]),
                    _cell(row["stage_metrics"]),
                    _cell(row["allowed_claim"]),
                    _cell(row["boundary"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. 复现入口",
            "",
            "- 形状证据：`jq '.summary' results/qwen35_report_audit_20260619/length_regime_coverage.json`。",
            "- Video-AMME ci-50 分布：`jq '.target_chars, .audio_duration_s' results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json`。",
            "- Synthetic short/long CSV：`column -s, -t results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv | head`。",
            "- Stage budget：`jq '.synthetic_speech_budget' results/qwen35_report_audit_20260619/stage_latency_budget.json`。",
            "- 复跑裁决：`jq '.rules[] | select(.id | test(\"synthetic_(short|long)\"))' results/qwen35_report_audit_20260619/rerun_acceptance_contract.json`。",
            "",
            "## 4. 不可越界说法",
            "",
            "- 不说“synthetic long c=8 代表所有长文线上流量”。",
            "- 不说“Video-AMME ci-50 已经覆盖完整线上流量”。",
            "- 不说“短/长输入形状改变后仍可直接复用当前 headline 数字”。",
            "- 任何替换 headline 的长短文数据都必须保持固定输入形状、同一镜像、同一硬件、同一 WER/ASR 口径，并重跑 full audit。",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Length-Regime Coverage\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Checks | {summary['checks_passed']}/{summary['checks_total']} |")
    print(f"| Rows | {summary['rows_total']} |")
    print(f"| Required failures | {summary['required_failures']} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build short/long length-regime coverage evidence for Qwen3.5-Omni."
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
    write_markdown(payload, output)
    _save_json(payload, json_output)
    print_markdown(payload)
    print(f"Length-regime coverage written: {output} ready={payload['summary']['ready']}")
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
