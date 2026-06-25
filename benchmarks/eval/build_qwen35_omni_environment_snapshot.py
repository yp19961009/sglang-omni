# SPDX-License-Identifier: Apache-2.0
"""Build a reproducibility environment snapshot for the Qwen3.5-Omni report."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval.preflight_qwen35_omni_repro import (
    SGLANG_IMAGE,
    SGLANG_IMAGE_ID,
    VLLM_IMAGE,
)


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 15) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "command": cmd,
            "returncode": 1,
            "ok": False,
            "output": str(exc),
        }
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "output": proc.stdout.strip(),
    }


def _git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "head": _run(["git", "rev-parse", "HEAD"], cwd=root).get("output"),
        "status_short": _run(["git", "status", "--short"], cwd=root).get("output"),
    }


def _docker_image(image: str, *, expected_id: str | None = None) -> dict[str, Any]:
    if shutil.which("docker") is None:
        return {"image": image, "available": False, "ok": False, "reason": "docker CLI not found"}
    result = _run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}} {{.Created}}"],
        timeout=20,
    )
    fields = result.get("output", "").split()
    image_id = fields[0] if fields else ""
    created = fields[1] if len(fields) > 1 else ""
    expected_ok = expected_id is None or image_id == expected_id
    return {
        "image": image,
        "available": bool(result["ok"]),
        "ok": bool(result["ok"]) and expected_ok,
        "id": image_id,
        "created": created,
        "expected_id": expected_id,
        "expected_id_match": expected_ok,
        "inspect_returncode": result["returncode"],
        "inspect_output": result["output"],
    }


def _gpu_snapshot(min_gpus: int) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "ok": False, "reason": "nvidia-smi not found"}
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout=20,
    )
    version_query = _run(["nvidia-smi"], timeout=20)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", version_query.get("output", ""))
    cuda_version = cuda_match.group(1) if cuda_match else ""
    rows: list[dict[str, str]] = []
    for line in query.get("output", "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total": parts[2],
                    "driver_version": parts[3],
                }
            )
    return {
        "available": bool(query["ok"]),
        "ok": bool(query["ok"]) and len(rows) >= min_gpus,
        "required_gpus": min_gpus,
        "count": len(rows),
        "gpus": rows,
        "query_returncode": query["returncode"],
        "query_output": query["output"],
        "cuda_version": cuda_version,
        "nvidia_smi_output": version_query.get("output"),
    }


def _path_record(name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "type": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
    }


def _load_json_optional(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    audit_dir = root / "results/qwen35_report_audit_20260619"
    preflight = _load_json_optional(audit_dir / "preflight_repro.json")
    manifest = _load_json_optional(audit_dir / "manifest.json")
    coverage = _load_json_optional(audit_dir / "coverage_matrix.json")
    claims = _load_json_optional(audit_dir / "claims_verification.json")
    missing_records = manifest.get("summary", {}).get("missing_records")
    manifest_missing = int(missing_records) if missing_records is not None else 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "git": _git_snapshot(root),
        "docker_images": {
            "sglang": _docker_image(SGLANG_IMAGE, expected_id=SGLANG_IMAGE_ID),
            "vllm": _docker_image(VLLM_IMAGE),
        },
        "gpu": _gpu_snapshot(args.min_gpus),
        "paths": [
            _path_record("workspace", root),
            _path_record("model", args.model_path),
            _path_record("videoamme_cache", args.videoamme_cache),
            _path_record("report", root / "benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md"),
            _path_record("brief_zh", root / "benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md"),
        ],
        "audit": {
            "summary": {
                "ready": bool(claims.get("passed"))
                and bool(preflight.get("summary", {}).get("ready"))
                and bool(coverage.get("summary", {}).get("complete"))
                and manifest_missing == 0,
                "claims": {
                    "passed": claims.get("passed"),
                    "total_checks": claims.get("total_checks"),
                    "failed_checks": claims.get("failed_checks"),
                },
                "coverage": coverage.get("summary"),
                "preflight": preflight.get("summary"),
                "manifest": manifest.get("summary"),
            },
            "preflight": preflight.get("summary"),
            "manifest": manifest.get("summary"),
            "coverage": coverage.get("summary"),
            "claims": {
                "passed": claims.get("passed"),
                "total_checks": claims.get("total_checks"),
                "failed_checks": claims.get("failed_checks"),
            },
        },
    }


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def print_markdown(snapshot: dict[str, Any]) -> None:
    print("| Item | Value |")
    print("| --- | --- |")
    print(f"| Root | `{snapshot['root']}` |")
    print(f"| Git HEAD | `{snapshot['git'].get('head')}` |")
    gpu = snapshot.get("gpu", {})
    print(f"| GPU count | {gpu.get('count')}/{gpu.get('required_gpus')} |")
    first_gpu = (gpu.get("gpus") or [{}])[0]
    if first_gpu:
        print(
            "| First GPU | "
            f"{first_gpu.get('name')} / {first_gpu.get('memory_total')} / "
            f"driver {first_gpu.get('driver_version')} / CUDA {gpu.get('cuda_version')} |"
        )
    for label, image in snapshot.get("docker_images", {}).items():
        print(
            f"| Docker {label} | `{image.get('image')}` / `{image.get('id')}` / "
            f"ok={image.get('ok')} |"
        )
    audit = snapshot.get("audit", {}).get("summary", {})
    print(f"| Audit ready | {audit.get('ready')} |")
    print(f"| Claims | {audit.get('claims')} |")
    print(f"| Coverage | {audit.get('coverage')} |")
    print(f"| Manifest | {audit.get('manifest')} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni report reproducibility environment snapshot."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/gangouyu/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315"),
    )
    parser.add_argument("--videoamme-cache", type=Path, default=Path("/home/gangouyu/data/videoamme"))
    parser.add_argument("--min-gpus", type=int, default=8)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    snapshot = build_snapshot(args)
    print_markdown(snapshot)
    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = args.root.resolve() / output
        _save_json(snapshot, output)


if __name__ == "__main__":
    main()
