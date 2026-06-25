# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic convenience tarball for the Qwen3.5-Omni share package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_SOURCE_MANIFEST = AUDIT_DIR / "share_bundle_manifest.json"
DEFAULT_OUTPUT = AUDIT_DIR / "qwen35_omni_share_bundle_20260621.tar.gz"
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "share_bundle_package_manifest.json"
DEFAULT_ARC_PREFIX = "qwen35_omni_share_bundle_20260621"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _checksum_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".sha256")


def _require_inside_root(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Share package source is outside root: {resolved}") from exc
    return resolved


def _source_files(
    root: Path,
    payload: dict[str, Any],
    *,
    source_manifest: Path,
) -> list[Path]:
    paths: list[Path] = []
    for record in payload.get("records", []):
        if record.get("type") == "file":
            paths.append(root / str(record["relative_path"]))
        elif record.get("type") == "directory":
            for item in record.get("files", []):
                paths.append(root / str(item["relative_path"]))
    paths.append(source_manifest)

    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = _require_inside_root(root, path)
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing share package source: {resolved}")
        seen.add(resolved)
        result.append(resolved)
    return result


def _file_records(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _add_bytes(tf: tarfile.TarFile, arcname: str, content: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(content)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tf.addfile(info, io.BytesIO(content))


def _add_file(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tf.gettarinfo(str(path), arcname=arcname)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with path.open("rb") as fp:
        tf.addfile(info, fp)


def _build_readme() -> str:
    return (
        "Qwen3.5-Omni SGLang vs vLLM share bundle\n"
        "\n"
        "Start with benchmarks/reports/qwen35_omni_start_here_zh_20260621.md "
        "for the 30-second gate summary, three receiver verification commands, "
        "first-read files, and caveat red lines.\n"
        "For a copy/paste university email or chat message, open "
        "benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md.\n"
        "Then open benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md "
        "for the full reading order, evidence map, and defense-question routes.\n"
        "For a compact 15-minute university review packet, open "
        "benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md "
        "and results/qwen35_report_audit_20260619/university_review_packet.json.\n"
        "If you are reading an extracted bundle on another machine, first open "
        "benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md "
        "to separate HOST_REPO, BUNDLE_ROOT, and container paths.\n"
        "For a one-page copy/paste receiver command card, open "
        "benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md; "
        "it chains checksum, receiver quickcheck, standalone validation, full audit, "
        "performance rerun entry points, and headline replacement boundaries.\n"
        "For the machine-checked receiver quickcheck contract, open "
        "benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md "
        "and results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json.\n"
        "For the single Chinese technical-report entry point, open "
        "benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md "
        "and results/qwen35_report_audit_20260619/university_technical_report.json.\n"
        "For pressure-by-pressure rerun command IDs, evidence files, and headline "
        "replacement boundaries, open "
        "benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md.\n"
        "For copy/paste jq proof cards for each core headline and gate, open "
        "benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md.\n"
        "To execute those proof cards and assert the core machine gates in one read-only "
        "smoke pass, run benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh "
        "--root \"$HOST_REPO\" --mode host from the host repository; after extracting "
        "the bundle, run \"$BUNDLE_ROOT/benchmarks/eval/"
        "qwen35_omni_evidence_query_cards_smoke.sh\" --root \"$BUNDLE_ROOT\" "
        "--mode portable.\n"
        "Before judging baseline strength, open "
        "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md and "
        "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md.\n"
        "For the measured-best recipe, anti-recipes, and vLLM diagnostic verdicts, "
        "open benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md "
        "and results/qwen35_report_audit_20260619/optimization_candidate_ledger.json.\n"
        "The vLLM baseline uses a Qwen3.5-capable image with compile/CUDA graph, "
        "prefix/chunked prefill, shared-memory transfer, encoder compile/batch, "
        "and prebuild w4 evidence locked; it is not a weak baseline.\n"
        "vLLM c=8 prebuild w4 remains an optimized offline diagnostic, not online serving parity.\n"
        "For stage breakdown and stage-to-stage bottlenecks, open "
        "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md "
        "section 2 first; its quick_reproduction_map gives 5 defense quick routes "
        "before drilling into every stage row. Then open "
        "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md, "
        "results/qwen35_report_audit_20260619/stage_causal_graph.json, "
        "results/qwen35_report_audit_20260619/stage_drilldown_index.json, "
        "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json, "
        "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json, "
        "and results/qwen35_report_audit_20260619/tail_confidence_appendix.json.\n"
        "For single/high concurrency and short/long text regimes, open "
        "benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md and "
        "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md, "
        "plus results/qwen35_report_audit_20260619/regime_decision_matrix.json.\n"
        "For one-page headline numbers, presentation flow, and PPT-ready figures, open "
        "benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md, "
        "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json, "
        "results/qwen35_report_audit_20260619/chart_source_consistency.json, "
        "results/qwen35_report_audit_20260619/metric_provenance_index.json, "
        "and results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json.\n"
        "For safe external wording, caveats that must travel, and defense questions, open "
        "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md, "
        "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json, "
        "results/qwen35_report_audit_20260619/defense_claim_matrix.json, "
        "results/qwen35_report_audit_20260619/claim_metric_crosswalk.json, "
        "and results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json.\n"
        "For external reproduction commands and handoff steps, open "
        "benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md, "
        "benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md, "
        "and results/qwen35_report_audit_20260619/repro_command_manifest.json.\n"
        "For package/raw path reference hygiene, open "
        "results/qwen35_report_audit_20260619/share_path_hygiene.json; "
        "it should show package_offenders_total=0, raw_offenders_total=0, "
        "and legacy_hits_total=0.\n"
        "For command-reference hygiene, open "
        "results/qwen35_report_audit_20260619/command_reference_hygiene.json; "
        "it should show all structured rerun command IDs resolve against "
        "repro_command_manifest.json and critical SGLang/vLLM commands are "
        "documented in public reports.\n"
        "For reviewer reruns and delta triage, also open "
        "benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md and "
        "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md.\n"
        "For a returned full-repository rerun evidence bundle, run "
        "benchmarks/eval/qwen35_omni_collaborator_return_check.py "
        "--root \"$HOST_REPO\" --strict --json-output "
        "results/qwen35_report_audit_20260619/collaborator_return_check.json; "
        "it verifies the 34 required return-evidence files, 27 command-matrix rows, "
        "same 8x H20/image contract, stage gates, vLLM c=8 online-parity caveat, "
        "and share-package validation chain before headline replacement review.\n"
        "For a one-command receiver quickcheck, run "
        "benchmarks/eval/qwen35_omni_receiver_quickcheck.sh from the repository root; "
        "it chains checksum, tarball-mode validation, receiver-smoke validation, "
        "extracted-only validation, and external standalone validation.\n"
        "For a one-command evidence-card smoke in the host repository, run "
        "benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh "
        "--root \"$HOST_REPO\" --mode host; after extracting the bundle, run "
        "\"$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh\" "
        "--root \"$BUNDLE_ROOT\" --mode portable. It extracts the bash blocks from "
        "the proof-card document, executes them against the selected root, and asserts "
        "the main SGLang/vLLM, stage, rerun, and receiver gates.\n"
        "For a manual receiver-side sanity check, run "
        "benchmarks/eval/validate_qwen35_omni_share_package.py from the repository root.\n"
        "After extracting this tarball, run the same script with --extracted-only "
        "from the extracted bundle root.\n"
        "The extracted bundle can contain an in_progress=true audit_run_summary.json "
        "because audit_run_summary.json is a tarball member written before final "
        "self-referential package hashes settle. This is expected only for "
        "extracted-only validation. The validator accepts that state only when "
        "direct machine gates in the bundle are green, including "
        "results/qwen35_report_audit_20260619/rerun_delta_triage.json "
        "rows_total>=19 and checks_passed>=8. Host-side final validation must use "
        "a completion-state audit_run_summary.json with top-level rerun_delta_triage.\n"
        "If the repository is mounted elsewhere, set HOST_REPO before copying commands.\n"
        "\n"
        "Headline replacement boundary:\n"
        "  Green / replacement review: same 8x H20 hardware, same SGLang and vLLM "
        "image digests, same model, same Video-AMME cache, same ASR/WER path, "
        "and full audit plus rerun acceptance contract are green.\n"
        "  Yellow / appendix only: full audit is green but hardware, image digest, "
        "model path, data cache, or ASR/WER path differs.\n"
        "  Red / do not replace: any required gate fails, share package validation "
        "fails, or vLLM c=8 evidence is still offline diagnostic only.\n"
        "Before replacing any report number, update the machine evidence JSON, "
        "charts, main report, Chinese university technical report, pressure matrix, stage causal graph, Q&A, defense "
        "claim matrix, claim metric crosswalk, objective requirement crosswalk, "
        "regime decision matrix, "
        "stage reproduction drilldown, stage route decision matrix, "
        "share package index, final delivery note, share bundle, "
        "checksum, receiver smoke validation, extracted-only validation, external standalone validation, and release seal "
        "together.\n"
        "Do not claim SGLang/vLLM headline replacement or vLLM c=8 online parity "
        "from a yellow/red rerun.\n"
        "\n"
        "Quick receiver checks from the repository root when this package is current:\n"
        "  HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"\n"
        "  SMOKE_DIR=\"${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_readme}\"\n"
        "  EXTRACT_DIR=\"${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_readme}\"\n"
        "  STANDALONE_DIR=\"${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_readme}\"\n"
        "  cd \"$HOST_REPO\"\n"
        "  bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh\n"
        "  bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root \"$HOST_REPO\" --mode host\n"
        "  python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check "
        "--root \"$HOST_REPO\" --strict "
        "--json-output results/qwen35_report_audit_20260619/"
        "collaborator_return_check.json\n"
        "  # Manual split commands:\n"
        "  sha256sum -c results/qwen35_report_audit_20260619/"
        "qwen35_omni_share_bundle_20260621.tar.gz.sha256\n"
        "  python3 -m benchmarks.eval.validate_qwen35_omni_share_package "
        "--root \"$HOST_REPO\" --strict "
        "--json-output results/qwen35_report_audit_20260619/share_package_validation.json\n"
        "  python3 -m benchmarks.eval.validate_qwen35_omni_share_package "
        "--root \"$HOST_REPO\" --strict "
        "--receiver-smoke-dir \"$SMOKE_DIR\" "
        "--json-output results/qwen35_report_audit_20260619/"
        "share_package_receiver_smoke_validation.json\n"
        "\n"
        "Manual extracted-root check after unpacking:\n"
        "  HOST_REPO=\"${HOST_REPO:-/home/gangouyu/sglang-omni}\"\n"
        "  EXTRACT_DIR=\"${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_readme}\"\n"
        "  cd \"$HOST_REPO\"\n"
        "  rm -rf \"$EXTRACT_DIR\"\n"
        "  mkdir -p \"$EXTRACT_DIR\"\n"
        "  tar -xzf results/qwen35_report_audit_20260619/"
        "qwen35_omni_share_bundle_20260621.tar.gz "
        "-C \"$EXTRACT_DIR\"\n"
        "  cd \"$EXTRACT_DIR/qwen35_omni_share_bundle_20260621\"\n"
        "  BUNDLE_ROOT=\"$PWD\"\n"
        "  bash \"$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh\" "
        "--root \"$BUNDLE_ROOT\" --mode portable\n"
        "  python3 benchmarks/eval/validate_qwen35_omni_share_package.py "
        "--root \"$PWD\" --extracted-only --strict "
        "--json-output \"$HOST_REPO/results/qwen35_report_audit_20260619/"
        "share_package_validation_extracted.json\"\n"
        "  cd \"$HOST_REPO\"\n"
        "  python3 -m benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation "
        "--root \"$HOST_REPO\" --strict "
        "--work-dir /tmp/qwen35_omni_external_standalone_bundle_validation_readme "
        "--json-output results/qwen35_report_audit_20260619/"
        "share_package_external_standalone_validation.json\n"
        "Expected gates when this package is current: final readiness 49/49, "
        "tarball validation 17/17, extracted-only validation 13/13, "
        "external standalone validation 8/8, release seal ready=true.\n"
        "Final readiness should expose public_doc_quality_guard="
        "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift, "
        "with hash/table/token/duplicate-heading/semantic-count offenders all empty.\n"
        "The receiver quickcheck script runs checksum, tarball validation, safe extraction, "
        "extracted-only validation, and external standalone validation in one host-side step.\n"
        "The validator directly scans packaged share_report Markdown for "
        "bare hashes, malformed tables, duplicate headings, semantic count drift, and malformed display tokens in both "
        "tarball mode and extracted-only mode.\n"
        "It also validates packaged share_charts CSV/SVG assets are parseable, non-empty, "
        "and render-structured.\n"
        "Passing validation evidence should show report_quality_offenders=[] and "
        "chart_quality_offenders=[].\n"
        "\n"
        "Wrong-root hints:\n"
        "  Tarball mode expects the repository root that contains results/qwen35_report_audit_20260619/.\n"
        "  Extracted-only mode expects the extracted bundle root that contains PACKAGE_README.txt "
        "and PACKAGE_FILE_SHA256SUMS.txt.\n"
        "\n"
        "Internal file hash list:\n"
        "  PACKAGE_FILE_SHA256SUMS.txt lists every bundled source file with "
        "repository-root-relative paths and SHA-256 hashes.\n"
        "  It is used by tarball validation and --extracted-only validation to verify reports, "
        "evidence JSON, tools, and chart assets before or after unpacking.\n"
        "\n"
        "The authoritative machine-readable file list is "
        "results/qwen35_report_audit_20260619/share_bundle_manifest.json.\n"
        "The full audit summary is "
        "results/qwen35_report_audit_20260619/audit_run_summary.json.\n"
        "Keep share_bundle_package_manifest.json, share_package_validation.json, "
        "share_package_validation_extracted.json, and "
        "share_package_receiver_smoke_validation.json, and "
        "share_package_external_standalone_validation.json, and "
        "share_release_seal.json adjacent to the tarball after "
        "generation. They describe this tarball and are intentionally not tarball "
        "members to avoid self-referential hashes.\n"
        "Keep benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md "
        "adjacent to the tarball as the human-readable release seal.\n"
        "This tarball intentionally excludes raw large benchmark outputs; hashes for "
        "the full local evidence inventory remain in manifest.json.\n"
    )


def build_package(
    root: Path,
    *,
    source_manifest: Path,
    output: Path,
    json_output: Path,
    arc_prefix: str,
) -> dict[str, Any]:
    root = root.resolve()
    source_manifest = (
        source_manifest if source_manifest.is_absolute() else root / source_manifest
    ).resolve()
    output = (output if output.is_absolute() else root / output).resolve()
    json_output = (json_output if json_output.is_absolute() else root / json_output).resolve()
    checksum_output = _checksum_path(output)

    source_payload = _load_json(source_manifest)
    source_summary = source_payload.get("summary", {})
    source_ready = bool(source_summary.get("ready"))
    source_missing = int(source_summary.get("missing_required") or 0)
    files = _source_files(root, source_payload, source_manifest=source_manifest)
    file_records = _file_records(root, files)
    source_filelist = "".join(
        f"{record['sha256']}  {record['relative_path']}\n"
        for record in file_records
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                _add_bytes(
                    tf,
                    f"{arc_prefix}/PACKAGE_README.txt",
                    _build_readme().encode("utf-8"),
                )
                _add_bytes(
                    tf,
                    f"{arc_prefix}/PACKAGE_FILE_SHA256SUMS.txt",
                    source_filelist.encode("utf-8"),
                )
                for path in files:
                    rel_path = path.relative_to(root).as_posix()
                    _add_file(tf, path, f"{arc_prefix}/{rel_path}")

    tarball_sha = _sha256(output)
    checksum_output.write_text(
        f"{tarball_sha}  {output.relative_to(root).as_posix()}\n",
        encoding="utf-8",
    )

    checks = {
        "source_manifest_ready": source_ready and source_missing == 0,
        "source_file_count_sufficient": len(file_records) >= 40,
        "tarball_written": output.is_file() and output.stat().st_size > 0,
        "checksum_written": checksum_output.is_file(),
        "checksum_matches_tarball": checksum_output.read_text(encoding="utf-8").split()[0]
        == tarball_sha,
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "tarball": str(output),
        "tarball_relative_path": output.relative_to(root).as_posix(),
        "tarball_size_bytes": output.stat().st_size,
        "tarball_sha256": tarball_sha,
        "checksum_file": str(checksum_output),
        "checksum_relative_path": checksum_output.relative_to(root).as_posix(),
        "arc_prefix": arc_prefix,
        "summary": {
            "ready": all(checks.values()),
            "file_count": len(file_records),
            "total_source_bytes": sum(record["size_bytes"] for record in file_records),
            "source_manifest_records": source_summary.get("records_total"),
            "checks": checks,
        },
        "source_summary": source_summary,
        "files": file_records,
    }
    _save_json(payload, json_output)
    return payload


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Share Bundle Package\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Source files | {summary['file_count']} |")
    print(f"| Source bytes | {summary['total_source_bytes']} |")
    print(f"| Tarball bytes | {payload['tarball_size_bytes']} |")
    print(f"| Tarball SHA-256 | `{payload['tarball_sha256']}` |")
    print("\n| Check | Value |")
    print("| --- | ---: |")
    for name, value in summary["checks"].items():
        print(f"| {name} | {value} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic Qwen3.5-Omni share tarball."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--arc-prefix", default=DEFAULT_ARC_PREFIX)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = build_package(
        args.root,
        source_manifest=args.source_manifest,
        output=args.output,
        json_output=args.json_output,
        arc_prefix=args.arc_prefix,
    )
    print_markdown(payload)
    print(
        "Share bundle package written: "
        f"{payload['tarball']} ready={payload['summary']['ready']} "
        f"files={payload['summary']['file_count']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
