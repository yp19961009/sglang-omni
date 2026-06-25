# SPDX-License-Identifier: Apache-2.0
"""Validate that Qwen3.5-Omni share charts match their audit JSON sources."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval.build_qwen35_omni_share_charts import (
    AUDIT_DIR,
    build_chart_pack,
)


DEFAULT_JSON_OUTPUT = AUDIT_DIR / "chart_source_consistency.json"
DEFAULT_REPORT_OUTPUT = (
    Path("benchmarks/reports")
    / "qwen35_omni_chart_source_consistency_zh_20260621.md"
)
SOURCE_FILES = [
    AUDIT_DIR / "tables_summary.json",
    AUDIT_DIR / "headline_scorecard.json",
    AUDIT_DIR / "stage_interaction_summary.json",
    AUDIT_DIR / "vllm_admission_diagnosis.json",
    AUDIT_DIR / "stage_latency_budget.json",
]
CHART_DIR = AUDIT_DIR / "share_charts"
CHART_MANIFEST = CHART_DIR / "chart_pack_manifest.json"


@dataclass(frozen=True)
class Check:
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else {}


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as fp:
        return fp.read()


def _relative_chart_files(manifest: dict[str, Any], root: Path) -> list[str]:
    rel_paths: list[str] = []
    for item in manifest.get("generated_files", []):
        path_text = str(item.get("path") or "")
        if not path_text:
            continue
        path = Path(path_text)
        try:
            rel_paths.append(str(path.resolve().relative_to(root)))
        except ValueError:
            rel_paths.append(str(path))
    return sorted(rel_paths)


def _generated_by_filename(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.get("generated_files", []):
        path_text = str(item.get("path") or "")
        if path_text:
            result[Path(path_text).name] = dict(item)
    return result


def _csv_shape(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    return {
        "columns": list(reader.fieldnames or []),
        "rows": len(rows),
    }


def _check(checks: list[Check], name: str, condition: bool, evidence: str) -> None:
    checks.append(Check(name, "PASS" if condition else "FAIL", evidence))


def build_payload(root: Path) -> dict[str, Any]:
    root = root.resolve()
    checks: list[Check] = []

    source_status = [
        {
            "path": str((root / path).resolve()),
            "relative_path": str(path),
            "exists": (root / path).is_file(),
            "size_bytes": (root / path).stat().st_size if (root / path).is_file() else None,
        }
        for path in SOURCE_FILES
    ]
    _check(
        checks,
        "source audit JSONs exist",
        all(item["exists"] for item in source_status),
        f"sources={sum(1 for item in source_status if item['exists'])}/{len(source_status)}",
    )

    chart_manifest_path = root / CHART_MANIFEST
    actual_manifest = _load_json(chart_manifest_path) if chart_manifest_path.is_file() else {}
    actual_summary = actual_manifest.get("summary", {})
    _check(
        checks,
        "chart pack manifest ready",
        bool(actual_summary.get("ready"))
        and int(actual_summary.get("csv_files") or 0) >= 7
        and int(actual_summary.get("svg_files") or 0) >= 7
        and int(actual_summary.get("generated_files") or 0) >= 14,
        f"chart_summary={actual_summary}",
    )

    with tempfile.TemporaryDirectory(prefix="qwen35_chart_source_") as tmpdir:
        expected_dir = Path(tmpdir) / "share_charts"
        expected_manifest = build_chart_pack(root, expected_dir)
        expected_summary = expected_manifest.get("summary", {})
        _check(
            checks,
            "expected chart regeneration ready",
            bool(expected_summary.get("ready"))
            and int(expected_summary.get("csv_files") or 0) >= 7
            and int(expected_summary.get("svg_files") or 0) >= 7,
            f"expected_summary={expected_summary}",
        )

        actual_by_name = _generated_by_filename(actual_manifest)
        expected_by_name = _generated_by_filename(expected_manifest)
        actual_names = sorted(actual_by_name)
        expected_names = sorted(expected_by_name)
        _check(
            checks,
            "chart file set matches regenerated source",
            actual_names == expected_names and len(actual_names) >= 14,
            f"actual={actual_names}; expected={expected_names}",
        )

        file_rows: list[dict[str, Any]] = []
        csv_exact = 0
        svg_exact = 0
        csv_parseable = 0
        svg_parseable = 0
        metadata_matches = 0
        for name in expected_names:
            expected_path = expected_dir / name
            actual_path = root / CHART_DIR / name
            expected_item = expected_by_name.get(name, {})
            actual_item = actual_by_name.get(name, {})
            kind = str(expected_item.get("kind") or actual_item.get("kind") or "")
            exists = actual_path.is_file()
            expected_bytes = _read_bytes(expected_path) if expected_path.is_file() else b""
            actual_bytes = _read_bytes(actual_path) if exists else b""
            byte_exact = exists and actual_bytes == expected_bytes

            expected_meta = {
                key: expected_item.get(key)
                for key in ["kind", "label", "rows"]
                if key in expected_item
            }
            actual_meta = {
                key: actual_item.get(key)
                for key in ["kind", "label", "rows"]
                if key in actual_item
            }
            meta_match = expected_meta == actual_meta
            metadata_matches += int(meta_match)

            parseable = False
            shape: dict[str, Any] = {}
            if kind == "csv" and exists:
                shape = _csv_shape(actual_path)
                parseable = bool(shape["columns"]) and int(shape["rows"]) == int(
                    actual_item.get("rows") or 0
                )
                csv_exact += int(byte_exact)
                csv_parseable += int(parseable)
            elif kind == "svg" and exists:
                text = actual_path.read_text(encoding="utf-8")
                parseable = text.startswith("<svg ") and text.rstrip().endswith("</svg>")
                svg_exact += int(byte_exact)
                svg_parseable += int(parseable)

            file_rows.append(
                {
                    "file": name,
                    "kind": kind,
                    "exists": exists,
                    "byte_exact_with_regenerated_source": byte_exact,
                    "manifest_metadata_matches_regenerated_source": meta_match,
                    "parseable": parseable,
                    "actual_size_bytes": actual_path.stat().st_size if exists else None,
                    "expected_size_bytes": len(expected_bytes),
                    "csv_shape": shape,
                }
            )

        _check(
            checks,
            "all chart assets byte-exact with regenerated source",
            len(file_rows) >= 14 and all(row["byte_exact_with_regenerated_source"] for row in file_rows),
            f"checked={len(file_rows)}, csv_exact={csv_exact}, svg_exact={svg_exact}",
        )
        _check(
            checks,
            "all chart manifest metadata matches regenerated source",
            len(file_rows) >= 14
            and metadata_matches == len(file_rows)
            and _relative_chart_files(actual_manifest, root)
            == sorted(str(CHART_DIR / name) for name in expected_names),
            f"metadata_matches={metadata_matches}/{len(file_rows)}",
        )
        _check(
            checks,
            "all CSV assets parse with manifest row counts",
            csv_parseable >= 7,
            f"csv_parseable={csv_parseable}/7",
        )
        _check(
            checks,
            "all SVG assets have valid svg envelope",
            svg_parseable >= 7,
            f"svg_parseable={svg_parseable}/7",
        )

    required_failures = [check.to_dict() for check in checks if check.required and check.status != "PASS"]
    summary = {
        "ready": not required_failures,
        "checks_total": len(checks),
        "checks_passed": sum(1 for check in checks if check.status == "PASS"),
        "required_failures": len(required_failures),
        "csv_files_checked": sum(1 for row in file_rows if row["kind"] == "csv"),
        "svg_files_checked": sum(1 for row in file_rows if row["kind"] == "svg"),
        "byte_exact_files": sum(
            1 for row in file_rows if row["byte_exact_with_regenerated_source"]
        ),
        "source_files_total": len(source_status),
        "share_scope": (
            "Verifies slide CSV/SVG assets are byte-exact regenerations from audited "
            "JSON sources, so public figures have not been hand-edited."
        ),
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": summary,
        "sources": source_status,
        "chart_manifest": {
            "path": str(chart_manifest_path),
            "summary": actual_summary,
        },
        "checks": [check.to_dict() for check in checks],
        "file_rows": file_rows,
        "required_failures": required_failures,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Qwen3.5-Omni Chart Source Consistency",
        "",
        "用途：校验分享包里的 PPT CSV/SVG 图表是否仍然是从审计 JSON 重新生成的版本，",
        "避免 slide 数字被手工改动后继续对外分享。",
        "",
        "## 1. Summary",
        "",
        "| Gate | Value |",
        "| --- | ---: |",
        f"| Ready | `{summary['ready']}` |",
        f"| Checks | `{summary['checks_passed']}/{summary['checks_total']}` |",
        f"| Required failures | `{summary['required_failures']}` |",
        f"| CSV files checked | `{summary['csv_files_checked']}` |",
        f"| SVG files checked | `{summary['svg_files_checked']}` |",
        f"| Byte-exact files | `{summary['byte_exact_files']}` |",
        "",
        "## 2. Checks",
        "",
        "| Status | Required | Check | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(
            "| {status} | {required} | {name} | {evidence} |".format(
                status=check["status"],
                required="yes" if check["required"] else "no",
                name=check["name"],
                evidence=str(check["evidence"]).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## 3. Chart Files",
            "",
            "| File | Kind | Byte-exact | Parseable | Size | Shape |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in payload["file_rows"]:
        shape = row.get("csv_shape") or {}
        shape_text = (
            f"{len(shape.get('columns', []))} columns / {shape.get('rows')} rows"
            if shape
            else ""
        )
        lines.append(
            "| {file} | {kind} | {exact} | {parseable} | {size} | {shape} |".format(
                file=row["file"],
                kind=row["kind"],
                exact=row["byte_exact_with_regenerated_source"],
                parseable=row["parseable"],
                size=row["actual_size_bytes"],
                shape=shape_text,
            )
        )
    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "- 这个 gate 证明图表文件与当前审计 JSON 的生成结果 byte-exact 一致。",
            "- 如果合作方改了 PPT 数字或 CSV/SVG，必须重跑 chart pack 和本一致性检查。",
            "- 它不替代 benchmark 复跑；benchmark 复跑仍以 `repro_command_manifest.json` 和 full audit 为准。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate share chart CSV/SVG assets against regenerated audit JSON sources."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_payload(root)

    json_output = args.json_output
    if not json_output.is_absolute():
        json_output = root / json_output
    _save_json(payload, json_output)

    output = args.output
    if not output.is_absolute():
        output = root / output
    write_markdown(payload, output)

    summary = payload["summary"]
    print(
        "Chart source consistency: "
        f"ready={summary['ready']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"byte_exact={summary['byte_exact_files']} "
        f"csv={summary['csv_files_checked']} svg={summary['svg_files_checked']}"
    )
    if args.strict and not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
