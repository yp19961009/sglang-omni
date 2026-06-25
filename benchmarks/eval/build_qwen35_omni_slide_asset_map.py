# SPDX-License-Identifier: Apache-2.0
"""Build a slide-to-asset map for the Qwen3.5-Omni share deck."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
CHART_DIR = AUDIT_DIR / "share_charts"
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "slide_asset_map.json"


@dataclass(frozen=True)
class SlideAssetRow:
    deck_section: str
    claim: str
    primary_asset: str
    data_asset: str
    speaker_note: str

    def to_dict(self) -> dict[str, str]:
        return {
            "deck_section": self.deck_section,
            "claim": self.claim,
            "primary_asset": self.primary_asset,
            "data_asset": self.data_asset,
            "speaker_note": self.speaker_note,
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


def _rows() -> list[SlideAssetRow]:
    return [
        SlideAssetRow(
            deck_section="3. Headline: warmed c=4 严格横向对比",
            claim="SGLang latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/strict_c4_latency_rtf.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/strict_c4_runtime_comparison.csv",
            speaker_note="先讲 warmed skip-first-4，再讲 latency/RTF 四项优势和 WER 边界。",
        ),
        SlideAssetRow(
            deck_section="4. SGLang 压测曲线：c=1/2/4/8/16",
            claim="c=8 是当前吞吐峰值，c=16 是 saturation boundary。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/sglang_pressure_qps.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/sglang_pressure_sweep.csv",
            speaker_note="用 QPS 图讲峰值，再用表格解释 c=16 不推荐。",
        ),
        SlideAssetRow(
            deck_section="4. SGLang 压测曲线：latency tail",
            claim="c=16 tail 变差是 admission/queueing 饱和信号。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/sglang_pressure_latency.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/sglang_pressure_sweep.csv",
            speaker_note="不要把 c=16 说成失败；它是压力边界证据。",
        ),
        SlideAssetRow(
            deck_section="5. Stage breakdown：哪里慢",
            claim="c=8/c=16 preprocessing lifecycle 增长主要是 queue/admission。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget_pct.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/sglang_stage_latency_budget.csv",
            speaker_note="强调 lifecycle 和 actual compute 的区别。",
        ),
        SlideAssetRow(
            deck_section="6. Stage 连接：是不是卡在 stage 之间",
            claim="talker_ar -> code2wav handoff 健康，code2wav decode 不是主瓶颈。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/sglang_handoff_decode_ms.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/stage_connection_health.csv",
            speaker_note="用 hop p95 和 decode ms 回答 stage boundary 追问。",
        ),
        SlideAssetRow(
            deck_section="7. 短/长文本输入 + 语音输出 guardrail",
            claim="short/long c=1/4/8 均覆盖，long c=8 仍快于实时。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_rtf.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv",
            speaker_note="同步说清 synthetic long 不是 official SeedTTS full-set headline。",
        ),
        SlideAssetRow(
            deck_section="8. 负优化：为什么不直接放大 preprocessing 并发",
            claim="preproc=2 回退，preproc=4 OOM/失败，朴素加并发不是当前 recipe。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/preprocessing_antirecipe.csv",
            data_asset="results/qwen35_report_audit_20260619/share_charts/preprocessing_antirecipe.csv",
            speaker_note="这一页用 CSV 表即可；重点讲共享资源争用。",
        ),
        SlideAssetRow(
            deck_section="9. vLLM c=8：offline diagnostic 边界",
            claim="prebuild w4 是当前最强 offline diagnostic，不是 online serving parity。",
            primary_asset="results/qwen35_report_audit_20260619/share_charts/vllm_c8_diagnostic_qps.svg",
            data_asset="results/qwen35_report_audit_20260619/share_charts/vllm_admission_diagnosis.csv",
            speaker_note="分清 runner QPS、engine QPS 和 admission span。",
        ),
        SlideAssetRow(
            deck_section="11. 复现路径",
            claim="先 full audit，再 receiver smoke/extracted-only，再重跑性能。",
            primary_asset="results/qwen35_report_audit_20260619/audit_run_summary.json",
            data_asset="results/qwen35_report_audit_20260619/repro_command_manifest.json",
            speaker_note="这页不放性能图；放命令和 expected gates。",
        ),
        SlideAssetRow(
            deck_section="14. 被追问时的证据跳转",
            claim="复跑数字偏移时先用 delta triage，不要直接改 headline。",
            primary_asset="benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
            data_asset="results/qwen35_report_audit_20260619/rerun_delta_triage.json",
            speaker_note="现场把症状映射到 stage/boundary、裁决和下一步动作。",
        ),
    ]


def _manifest_paths(root: Path, manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for item in manifest.get("generated_files", []):
        path_text = str(item.get("path") or "")
        if not path_text:
            continue
        path = Path(path_text)
        try:
            paths.add(path.resolve().relative_to(root).as_posix())
        except ValueError:
            paths.add(path_text)
    return paths


def build_slide_asset_map(root: Path) -> dict[str, Any]:
    root = root.resolve()
    chart_manifest = _load_json_optional(root / CHART_DIR / "chart_pack_manifest.json")
    chart_summary = chart_manifest.get("summary", {})
    manifest_paths = _manifest_paths(root, chart_manifest)
    rows = _rows()
    asset_paths = sorted(
        {
            row.primary_asset
            for row in rows
            if row.primary_asset.startswith("results/qwen35_report_audit_20260619/share_charts/")
        }
        | {
            row.data_asset
            for row in rows
            if row.data_asset.startswith("results/qwen35_report_audit_20260619/share_charts/")
        }
    )
    missing_assets = [path for path in asset_paths if not (root / path).is_file()]
    missing_from_chart_manifest = [
        path for path in asset_paths if path not in manifest_paths
    ]
    checks = {
        "rows_total": len(rows) >= 10,
        "chart_pack_ready": bool(chart_summary.get("ready"))
        and int(chart_summary.get("svg_files") or 0) >= 7
        and int(chart_summary.get("csv_files") or 0) >= 7,
        "all_chart_assets_exist": not missing_assets,
        "all_chart_assets_in_manifest": not missing_from_chart_manifest,
        "reproduction_and_triage_routes_present": any(
            row.data_asset.endswith("repro_command_manifest.json") for row in rows
        )
        and any(row.data_asset.endswith("rerun_delta_triage.json") for row in rows),
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "required_failures": len(required_failures),
            "chart_assets_total": len(asset_paths),
            "missing_assets": missing_assets,
            "missing_from_chart_manifest": missing_from_chart_manifest,
        },
        "checks": checks,
        "chart_pack_summary": chart_summary,
        "rows": [row.to_dict() for row in rows],
        "required_failures": required_failures,
    }


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni 分享 Deck 图表资产映射",
        "",
        "状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。",
        "工作目录：`/home/gangouyu/sglang-omni`。",
        "用途：把 deck 提纲中的每个核心 claim 映射到可直接放 PPT 的 SVG/CSV 或现场证据 JSON，避免手工挑图或手改数字。",
        "",
        "## 1. 使用规则",
        "",
        "1. PPT 中优先嵌入 SVG；需要表格或复核数字时打开同名 CSV。",
        "2. 不要手工改图里的数字；图表来自 `share_charts/chart_pack_manifest.json` 记录的审计 JSON。",
        "3. 没有图的复现/追问页使用 JSON 或 Markdown 证据，不伪造图。",
        "4. 若重跑后数字变化，先重建 share chart pack、slide asset map、share bundle，再跑 package validation。",
        "",
        "## 2. 当前 Gate",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | {summary.get('ready')} |",
        f"| Rows | {summary.get('rows_total')} |",
        f"| Checks | {summary.get('checks_passed')}/{summary.get('checks_total')} |",
        f"| Chart assets | {summary.get('chart_assets_total')} |",
        f"| Missing assets | {len(summary.get('missing_assets') or [])} |",
        f"| Missing from chart manifest | {len(summary.get('missing_from_chart_manifest') or [])} |",
        "",
        "## 3. Slide 资产映射",
        "",
        "| Deck section | Claim | Primary asset | Data / proof asset | Speaker note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(row["deck_section"]),
                    _md_escape(row["claim"]),
                    f"`{_md_escape(row['primary_asset'])}`",
                    f"`{_md_escape(row['data_asset'])}`",
                    _md_escape(row["speaker_note"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 4. 发送前检查",
            "",
            "```bash",
            "HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"",
            "cd \"$HOST_REPO\"",
            "python3 -m benchmarks.eval.build_qwen35_omni_slide_asset_map \\",
            "  --root \"$HOST_REPO\" \\",
            "  --strict \\",
            "  --output benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md \\",
            "  --json-output results/qwen35_report_audit_20260619/slide_asset_map.json",
            "python3 -m benchmarks.eval.run_qwen35_omni_report_audit \\",
            "  --root \"$HOST_REPO\" \\",
            "  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.5-Omni slide asset map."
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
    payload = build_slide_asset_map(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")
    _save_json(payload, json_output)
    if args.strict and payload["summary"]["required_failures"]:
        raise SystemExit(
            "Slide asset map failed: "
            + ", ".join(payload.get("required_failures", []))
        )
    print(
        "slide_asset_map_ready="
        f"{payload['summary']['ready']} rows={payload['summary']['rows_total']} "
        f"checks={payload['summary']['checks_passed']}/{payload['summary']['checks_total']}"
    )


if __name__ == "__main__":
    main()
