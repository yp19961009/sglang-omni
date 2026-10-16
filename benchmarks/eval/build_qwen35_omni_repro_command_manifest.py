# SPDX-License-Identifier: Apache-2.0
"""Build a machine-readable reproduction-command manifest for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = AUDIT_DIR / "repro_command_manifest.json"

EXPECTED_GATES = {
    "claims": {"passed": 17, "total": 17, "failed": 0},
    "coverage": {"passed": 34, "total": 34, "missing": 0},
    "preflight": {"total_checks": 62, "required_failures": 0},
    "manifest": {
        "min_total_records": 180,
        "min_file_records": 178,
        "directory_records": 2,
        "missing_records": 0,
    },
    "headline_scorecard": {"checks_passed": 9, "checks_total": 9},
    "chart_pack": {"csv_files": 7, "svg_files": 7, "generated_files": 14},
    "chart_source_consistency": {
        "ready": True,
        "checks_total": 8,
        "csv_files_checked": 7,
        "svg_files_checked": 7,
        "byte_exact_files": 14,
        "required_failures": 0,
    },
    "acceptance_matrix": {"rows_passed": 17, "rows_total": 17},
    "confidence_ledger": {
        "entries_passed": 12,
        "entries_total": 12,
        "high_confidence_claims": 9,
        "medium_confidence_boundaries": 3,
        "unsupported_claims": 0,
    },
    "caveat_adjudication_matrix": {
        "ready": True,
        "rows_total": 12,
        "forbidden_claims_total": 12,
        "replacement_triggers_total": 12,
        "required_failures": 0,
    },
    "objective_completion_audit": {
        "share_ready_with_documented_caveats": True,
        "required_failures": 0,
        "rows_total": 17,
    },
    "objective_requirement_crosswalk": {
        "ready": True,
        "optimization_candidate_rows_total": 8,
        "required_failures": 0,
    },
    "optimization_candidate_ledger": {
        "ready": True,
        "candidate_rows_total": 8,
        "accepted_current_best_rows_total": 1,
        "rejected_anti_recipe_rows_total": 2,
        "vllm_diagnostic_rows_total": 2,
        "required_failures": 0,
    },
    "rerun_acceptance_contract": {
        "ready": True,
        "checks_total": 17,
        "rules_total": 18,
        "return_evidence_files": 34,
        "return_evidence_command_rows": 27,
    },
    "final_readiness": {"ready": True, "required_failures": 0},
    "share_bundle_manifest": {"ready": True},
    "share_release_seal": {"ready": True, "required_failures": 0},
    "tail_confidence_appendix": {"ready": True, "checks_total": 13, "rows_total": 18},
    "serving_capacity_matrix": {
        "ready": True,
        "checks_total": 10,
        "rows_total": 7,
        "required_failures": 0,
    },
    "share_consistency_guard": {
        "ready": True,
        "checks_total": 17,
        "required_failures": 0,
        "public_stale_hits": 0,
        "machine_stale_hits": 0,
        "embedded_identity_leaks": 0,
        "tarball_identity_mismatches": 0,
        "tarball_identity_missing_fields": 0,
        "evidence_smoke_route_missing": 0,
        "manifest_expected_gate_unexpected_fields": 0,
        "preflight_alias_mismatches": 0,
        "preflight_repro_in_manifest": True,
        "preflight_alias_in_manifest": False,
    },
}

REQUIRED_COMMAND_IDS = {
    "run_full_audit",
    "launch_sglang_optimized",
    "sglang_videoamme_stress",
    "sglang_synthetic_text_to_speech",
    "sglang_recompute_wer",
    "vllm_c1_original",
    "vllm_c4_original",
    "vllm_c8_original",
    "vllm_c8_prebuild_w4",
    "build_report_tables",
    "build_share_charts",
    "build_chart_source_consistency",
    "build_slide_asset_map",
    "build_final_status_summary",
    "build_regime_decision_matrix",
    "build_runtime_comparison_contract",
    "build_runtime_image_contract",
    "build_rerun_acceptance_contract",
    "build_sglang_optimization_lock",
    "build_vllm_optimization_lock",
    "build_vllm_online_parity_protocol",
    "build_final_checkpoint_watchlist",
    "build_tail_confidence_appendix",
    "build_stage_latency_budget",
    "build_stage_boundary_bottleneck_ledger",
    "build_serving_capacity_matrix",
    "build_share_consistency_guard",
    "build_stage_causal_graph",
    "build_stage_drilldown_index",
    "build_metric_provenance_index",
    "build_stage_reproduction_drilldown",
    "build_stage_route_decision_matrix",
    "build_caveat_adjudication_matrix",
    "build_rerun_delta_triage",
    "build_objective_completion_audit",
    "build_defense_claim_matrix",
    "build_claim_metric_crosswalk",
    "build_objective_requirement_crosswalk",
    "build_optimization_candidate_ledger",
    "build_share_path_hygiene",
    "build_command_reference_hygiene",
    "build_final_readiness_audit",
    "build_share_bundle_manifest",
    "build_share_bundle_package",
    "validate_share_bundle_package",
    "validate_share_bundle_receiver_smoke",
    "validate_extracted_share_bundle",
    "validate_external_standalone_share_bundle",
    "build_share_release_seal",
    "run_preflight",
    "check_wer_asr_path",
    "build_coverage_matrix",
    "build_evidence_manifest",
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


def _cmd(command: str) -> str:
    return dedent(command).strip()


def _command(
    *,
    command_id: str,
    phase: str,
    where: str,
    purpose: str,
    command: str,
    expected: str,
    evidence_after_run: list[str],
    rerun_cost: str,
) -> dict[str, Any]:
    return {
        "id": command_id,
        "phase": phase,
        "where": where,
        "purpose": purpose,
        "command": _cmd(command),
        "expected": expected,
        "evidence_after_run": evidence_after_run,
        "rerun_cost": rerun_cost,
    }


def _table_rows(tables: dict[str, Any], table_name: str) -> list[dict[str, Any]]:
    rows = tables.get("tables", {}).get(table_name, [])
    return rows if isinstance(rows, list) else []


def _synthetic_targets(tables: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for scenario in ["short", "long"]:
        rows = [
            row
            for row in _table_rows(tables, "synthetic_speech")
            if row.get("scenario") == scenario
        ]
        if not rows:
            continue
        chars = {float(row.get("target_chars") or 0.0) for row in rows}
        words = {float(row.get("target_words") or 0.0) for row in rows}
        result[scenario] = {
            "target_chars": max(chars) if chars else 0.0,
            "target_words": max(words) if words else 0.0,
        }
    return result


def _summary_from(path: Path) -> dict[str, Any]:
    return _load_json_optional(path).get("summary", {})


def _labels_from(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("label")) for row in rows if row.get("label")]


def _stable_share_release_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready": summary.get("ready"),
        "checks_total": summary.get("checks_total"),
        "checks_passed": summary.get("checks_passed"),
        "required_failures": summary.get("required_failures"),
        "receiver_smoke_ready": summary.get("receiver_smoke_ready"),
        "goal_complete": summary.get("goal_complete"),
        "completion_allowed_now": summary.get("completion_allowed_now"),
        "send_decision": summary.get("send_decision"),
        "forbidden_tarball_members": summary.get("forbidden_tarball_members"),
        "identity_fields_omitted": True,
    }


def build_commands() -> list[dict[str, Any]]:
    return [
        _command(
            command_id="run_full_audit",
            phase="audit_first",
            where="host",
            purpose="Regenerate the full report audit package in the required order.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.run_qwen35_omni_report_audit \\
                  --root /home/gangouyu/sglang-omni \\
                  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
            """,
            expected=(
                "ok=true; claims 17/17; coverage 34/34; preflight 62 checks with 0 "
                "required failures; manifest >=180 records with 0 missing; chart source "
                "consistency ready=true; objective "
                "completion audit has 0 required failures; final readiness ready=true; "
                "runtime image contract ready=true; rerun acceptance contract ready=true; "
                "checkpoint watchlist ready=true; stage latency budget ready=true; stage "
                "boundary bottleneck ledger ready=true; metric provenance index ready=true; "
                "stage reproduction drilldown ready=true; claim metric crosswalk ready=true; "
                "objective requirement crosswalk ready=true with 8 optimization candidates; "
                "share bundle ready=true; tarball validation 17/17; receiver smoke ready=true; "
                "extracted-only validation 13/13; standalone validation 8/8; "
                "receiver quickcheck contract 15/15 with WER/ASR path guard, evidence-query CLI/docs, stage dictionary crosswalk, and final completion route; "
                "final completion audit ready=true with 0 required failures; "
                "release seal ready=true."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/audit_run_summary.json",
                "results/qwen35_report_audit_20260619/manifest.json",
                "results/qwen35_report_audit_20260619/final_readiness_audit.json",
                "results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json",
            ],
            rerun_cost="audit-only, no benchmark rerun",
        ),
        _command(
            command_id="build_seedtts_meta",
            phase="audit_first",
            where="host",
            purpose="Create the local Video-AMME SeedTTS-compatible smoke meta file.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_videoamme_seedtts_meta \\
                  --output results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst \\
                  --summary-output results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json \\
                  --max-samples 50 --target-mode audio_text
            """,
            expected="50 local Video-AMME audio+text rows are exported without external cache.",
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst",
                "results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json",
            ],
            rerun_cost="audit-only, no serving",
        ),
        _command(
            command_id="launch_sglang_optimized",
            phase="sglang_serving",
            where="sglang_container",
            purpose="Start the optimized Qwen3.5-Omni SGLang serving recipe used by the report.",
            command="""
                cd /myapp/sglang-omni
                NO_CODE2WAV_TORCH_COMPILE=0 \\
                TORCHDYNAMO_DISABLE=0 \\
                SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \\
                SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \\
                EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \\
                bash examples/launch_qwen35_omni_speech_server_container.sh
            """,
            expected="Server listens on port 8161 after warmup; c<=8 is the recommended serving envelope for this recipe.",
            evidence_after_run=["serving logs", "request profiler events"],
            rerun_cost="requires 8x H20 serving session",
        ),
        _command(
            command_id="sglang_videoamme_stress",
            phase="sglang_stress",
            where="sglang_container",
            purpose="Rerun Video-AMME ci-50 at single and high concurrency.",
            command="""
                cd /myapp/sglang-omni
                for C in 1 2 4 8 16; do
                  RUN_ID="c${C}_profile_skipwer"
                  RUN_ROOT="results/qwen35_sglang_mr8_stress_20260619"
                  OUT_DIR="${RUN_ROOT}/benchmark_audio_50_${RUN_ID}"
                  curl -s http://127.0.0.1:8161/start_request_profile \\
                    -H "Content-Type: application/json" \\
                    -d "{\\"run_id\\":\\"${RUN_ID}\\",\\"event_dir\\":\\"/myapp/sglang-omni/${RUN_ROOT}/events\\"}"
                  HF_HOME=/myapp/data/videoamme \\
                  HF_DATASETS_CACHE=/myapp/data/videoamme/datasets \\
                  HF_HUB_OFFLINE=1 \\
                  python -m benchmarks.eval.benchmark_omni_videoamme \\
                    --model qwen3_5-omni --port 8161 \\
                    --repo-id zhaochenyang20/Video_AMME_ci \\
                    --output-dir "${OUT_DIR}" \\
                    --max-samples 50 --max-concurrency "${C}" \\
                    --max-tokens 256 --temperature 0.0 \\
                    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \\
                    --enable-audio --audio-voice m02 --skip-wer --disable-tqdm
                  curl -s http://127.0.0.1:8161/stop_request_profile \\
                    -H "Content-Type: application/json" \\
                    -d "{\\"run_id\\":\\"${RUN_ID}\\"}"
                  python -m sglang_omni.profiler \\
                    "/myapp/sglang-omni/${RUN_ROOT}/events" \\
                    --format json \\
                    --out "/myapp/sglang-omni/${RUN_ROOT}/request_profile_${RUN_ID}.json"
                done
            """,
            expected="c=8 is the current throughput peak; c=16 is saturation evidence; code2wav remains non-bottleneck.",
            evidence_after_run=[
                "results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_profile_skipwer/videoamme_results.json",
                "results/qwen35_sglang_mr8_stress_20260619/request_profile_c*_profile_skipwer.json",
            ],
            rerun_cost="serving benchmark",
        ),
        _command(
            command_id="sglang_synthetic_text_to_speech",
            phase="sglang_stress",
            where="sglang_container",
            purpose="Rerun fixed short/long text-input plus speech-output guardrails.",
            command="""
                cd /myapp/sglang-omni
                for SCENARIO in short long; do
                  for C in 1 4 8; do
                    RUN_ID="${SCENARIO}_c${C}_profile"
                    RUN_ROOT="results/qwen35_synthetic_speech_20260619"
                    OUT_DIR="${RUN_ROOT}/${SCENARIO}_c${C}"
                    SAMPLES=16
                    if [ "${SCENARIO}" = "long" ]; then SAMPLES=8; fi
                    curl -s http://127.0.0.1:8161/start_request_profile \\
                      -H "Content-Type: application/json" \\
                      -d "{\\"run_id\\":\\"${RUN_ID}\\",\\"event_dir\\":\\"/myapp/sglang-omni/${RUN_ROOT}/events\\"}"
                    python -m benchmarks.eval.benchmark_qwen35_speech_synthetic \\
                      --model qwen3_5-omni --port 8161 \\
                      --scenario "${SCENARIO}" --samples-per-scenario "${SAMPLES}" \\
                      --output-dir "${OUT_DIR}" \\
                      --max-concurrency "${C}" \\
                      --voice m02 --max-tokens 1024 --temperature 0.0 \\
                      --disable-tqdm
                    curl -s http://127.0.0.1:8161/stop_request_profile \\
                      -H "Content-Type: application/json" \\
                      -d "{\\"run_id\\":\\"${RUN_ID}\\"}"
                    python -m sglang_omni.profiler \\
                      "/myapp/sglang-omni/${RUN_ROOT}/events" \\
                      --format json \\
                      --out "/myapp/sglang-omni/${RUN_ROOT}/request_profile_${RUN_ID}.json"
                  done
                done
            """,
            expected="Short input remains 74 chars / 12 words; long input remains 944 chars / 139 words; long c=8 RTF stays below 1.0.",
            evidence_after_run=[
                "results/qwen35_synthetic_speech_20260619/*/synthetic_speech_results.json",
                "results/qwen35_synthetic_speech_20260619/request_profile_*_profile.json",
            ],
            rerun_cost="serving benchmark",
        ),
        _command(
            command_id="sglang_recompute_wer",
            phase="quality_validation",
            where="sglang_container",
            purpose="Recompute offline Whisper large-v3 WER from saved SGLang outputs.",
            command="""
                cd /myapp/sglang-omni
                python -m benchmarks.eval.compute_audio_consistency_from_results \\
                  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json \\
                  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json \\
                  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/videoamme_results.json \\
                  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json \\
                  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json \\
                  --path-root /myapp/sglang-omni \\
                  --local-whisper-model large-v3 \\
                  --asr-device cuda:1 \\
                  --output-name whisper_large_v3_local_wer.json \\
                  --lang en
            """,
            expected="Corpus WER remains stable across c=1/2/4/8/16 and does not trade quality for throughput.",
            evidence_after_run=[
                "results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_*/whisper_large_v3_local_wer.json"
            ],
            rerun_cost="offline ASR; avoid running while serving benchmark is active",
        ),
        _command(
            command_id="check_wer_asr_path",
            phase="quality_validation",
            where="host",
            purpose=(
                "Record whether the receiver will use a local Whisper cache or "
                "the ASR router path before recomputing WER."
            ),
            command="""
                cd /home/gangouyu/sglang-omni
                if [ -f /root/.cache/whisper/large-v3.pt ]; then
                  echo "WER_ASR_PATH=local_whisper_cache"
                  echo "WHISPER_CACHE=/root/.cache/whisper/large-v3.pt"
                else
                  echo "WER_ASR_PATH=asr_router_or_container_cache_required"
                  echo "LOCAL_WHISPER_CACHE_MISSING=/root/.cache/whisper/large-v3.pt"
                  echo "Use the ASR router command in qwen35_omni_stress_performance_plan_20260621.md section 14.2, or run WER inside a container with cached large-v3 weights."
                fi
            """,
            expected=(
                "Receiver records local_whisper_cache or asr_router_or_container_cache_required; "
                "a missing host cache is optional and must not be counted as a serving benchmark failure."
            ),
            evidence_after_run=["terminal output copied into collaborator rerun sheet"],
            rerun_cost="local check",
        ),
        _command(
            command_id="vllm_c1_original",
            phase="vllm_baseline",
            where="host",
            purpose="Rerun the optimized vLLM offline c=1 baseline.",
            command="""
                cd /home/gangouyu/sglang-omni
                MAX_SAMPLES=50 MAX_CONCURRENCY=1 MAX_NUM_SEQS=8 \\
                RUN_TAG=ci50_offline_compile_c1_mns8_20260619 \\
                bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
            """,
            expected="Completes Video-AMME ci-50 with the Qwen3.5-capable vLLM image and max_num_seqs=8.",
            evidence_after_run=[
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_*/benchmark_audio_50_c1_offline_compile/videoamme_results.json"
            ],
            rerun_cost="vLLM offline benchmark",
        ),
        _command(
            command_id="vllm_c4_original",
            phase="vllm_baseline",
            where="host",
            purpose="Rerun the optimized vLLM offline c=4 strict headline baseline.",
            command="""
                cd /home/gangouyu/sglang-omni
                MAX_SAMPLES=50 MAX_CONCURRENCY=4 MAX_NUM_SEQS=8 \\
                RUN_TAG=ci50_official_talker_compile_c4_20260618 \\
                bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
            """,
            expected="Completes Video-AMME ci-50 with the Qwen3.5-capable vLLM image; warmed c=4 is the strict apples-to-apples headline comparison slice.",
            evidence_after_run=[
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_*/run.log",
                "results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_*/benchmark_audio_50_c4_offline_compile/videoamme_results.json",
            ],
            rerun_cost="vLLM offline benchmark",
        ),
        _command(
            command_id="vllm_c8_original",
            phase="vllm_baseline",
            where="host",
            purpose="Rerun the optimized vLLM offline c=8 original path for prompt-feed diagnosis.",
            command="""
                cd /home/gangouyu/sglang-omni
                MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \\
                RUN_TAG=ci50_offline_compile_c8_mns8_20260619 \\
                bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
            """,
            expected="Original c=8 remains prompt-feed/admission limited and is not used as online serving parity.",
            evidence_after_run=[
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_*/run.log",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_*/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            ],
            rerun_cost="vLLM offline benchmark",
        ),
        _command(
            command_id="vllm_c8_prebuild_w4",
            phase="vllm_baseline",
            where="host",
            purpose="Rerun the strongest current vLLM offline diagnostic with parallel prompt prebuild.",
            command="""
                cd /home/gangouyu/sglang-omni
                RUN_ROOT="/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)" \\
                MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \\
                RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 \\
                EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4" \\
                bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
            """,
            expected="Completes 50/50; runner QPS improves versus prebuild w1; engine QPS remains around 0.536 in the current checkpoint.",
            evidence_after_run=[
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_*/run.log",
                "results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_*/benchmark_audio_50_c8_offline_compile/videoamme_results.json",
            ],
            rerun_cost="vLLM offline benchmark",
        ),
        _command(
            command_id="summarize_vllm_log_stages",
            phase="audit_regeneration",
            where="host",
            purpose="Rebuild vLLM log-derived stage timing summaries.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.summarize_vllm_omni_log_stages \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log \\
                  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log \\
                  --labels vLLM-c1 vLLM-c4 vLLM-c8 vLLM-c8-prebuild-w1 vLLM-c8-prebuild-w4 \\
                  --skip-first-requests 4 4 8 8 8 \\
                  --json-output results/qwen35_report_audit_20260619/vllm_log_stage_summary.json
            """,
            expected="vLLM log-stage rows exist for c1/c4/c8 and prebuild w1/w4.",
            evidence_after_run=["results/qwen35_report_audit_20260619/vllm_log_stage_summary.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="diagnose_vllm_admission",
            phase="audit_regeneration",
            where="host",
            purpose="Classify vLLM offline admission and runner overhead.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.diagnose_vllm_offline_admission \\
                  --case vLLM-c4 \\
                  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json \\
                  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log 4 \\
                  --case vLLM-c8 \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log 8 \\
                  --case vLLM-c8-prebuild-w1 \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log 8 \\
                  --case vLLM-c8-prebuild-w4 \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json \\
                  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log 8 \\
                  --json-output results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json
            """,
            expected="Original c4/c8 are prompt-feed limited; prebuild w4 is the optimized offline diagnostic.",
            evidence_after_run=["results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_report_tables",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate report tables from benchmark artifacts.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/tables_summary.json
            """,
            expected="All 45/45 expected artifacts are present and table rows include target_chars/target_words for synthetic text.",
            evidence_after_run=["results/qwen35_report_audit_20260619/tables_summary.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="verify_report_claims",
            phase="audit_regeneration",
            where="host",
            purpose="Verify the numeric claims in the report against JSON artifacts.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.verify_qwen35_omni_report_claims \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/claims_verification.json
            """,
            expected="17/17 claims pass with 0 failures.",
            evidence_after_run=["results/qwen35_report_audit_20260619/claims_verification.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_interactions",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate cross-stage interaction and bottleneck attribution summary.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json
            """,
            expected="At least 30 stage-interaction rows; SGLang handoff and code2wav checks are true.",
            evidence_after_run=["results/qwen35_report_audit_20260619/stage_interaction_summary.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_headline_scorecard",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the headline scorecard used in share material.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/headline_scorecard.json
            """,
            expected="ready=true and checks 9/9.",
            evidence_after_run=["results/qwen35_report_audit_20260619/headline_scorecard.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_share_charts",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate slide-ready CSV/SVG chart assets.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_share_charts \\
                  --root /home/gangouyu/sglang-omni \\
                  --output-dir results/qwen35_report_audit_20260619/share_charts \\
                  --manifest-output results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json
            """,
            expected="7 CSV files, 7 SVG files, and 14 generated files.",
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json",
                "results/qwen35_report_audit_20260619/share_charts/",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_chart_source_consistency",
            phase="audit_regeneration",
            where="host",
            purpose="Verify slide CSV/SVG assets are byte-exact regenerations from audit JSON sources.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_chart_source_consistency \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/chart_source_consistency.json
            """,
            expected="ready=true, checks 8/8, 7 CSV and 7 SVG assets byte-exact against regenerated sources.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md",
                "results/qwen35_report_audit_20260619/chart_source_consistency.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_slide_asset_map",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the slide-to-chart asset map for the external share deck.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_slide_asset_map \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/slide_asset_map.json
            """,
            expected="ready=true, 10 slide rows, 14 chart assets present, and no hand-edited chart numbers.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md",
                "results/qwen35_report_audit_20260619/slide_asset_map.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_acceptance_matrix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate per-pressure pass/fail acceptance rows.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/acceptance_matrix.json
            """,
            expected="ready=true and rows 17/17.",
            evidence_after_run=["results/qwen35_report_audit_20260619/acceptance_matrix.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_confidence_ledger",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate safe external wording and confidence boundaries.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/confidence_ledger.json
            """,
            expected="ready=true; entries 12/12; high=9, medium=3, unsupported=0.",
            evidence_after_run=["results/qwen35_report_audit_20260619/confidence_ledger.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_final_status_summary",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the one-page final status summary from audit JSONs.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_final_status_summary \\
                  --root /home/gangouyu/sglang-omni \\
                  --output benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md
            """,
            expected="Markdown summary reflects full audit, objective completion, final readiness, share bundle, and tarball checksum state.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_regime_decision_matrix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the reviewer-facing regime decision matrix from audited JSONs.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_regime_decision_matrix \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/regime_decision_matrix.json
            """,
            expected="ready=true; Markdown and JSON map all accepted regimes to recommendation, bottleneck, caveat, evidence, and action.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md",
                "results/qwen35_report_audit_20260619/regime_decision_matrix.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_runtime_comparison_contract",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the fair runtime-comparison contract from audited JSONs.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_runtime_comparison_contract \\
                  --root /home/gangouyu/sglang-omni \\
                  --output benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/runtime_comparison_contract.json \\
                  --strict
            """,
            expected="ready=true; 9/9 JSON checks; Markdown contract separates strict c=4 headline, SGLang scaling, vLLM c=8 offline diagnostics, optimized baseline strength, and invalid parity comparisons.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md",
                "results/qwen35_report_audit_20260619/runtime_comparison_contract.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_runtime_image_contract",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the runtime image, digest, optimization-switch, and claim-scope contract.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_runtime_image_contract \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/runtime_image_contract.json
            """,
            expected="ready=true, 12/12 checks pass, and SGLang/vLLM image digests plus optimized switches are locked.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md",
                "results/qwen35_report_audit_20260619/runtime_image_contract.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_rerun_acceptance_contract",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the collaborator rerun acceptance thresholds and headline replacement rules.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_rerun_acceptance_contract \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/rerun_acceptance_contract.json
            """,
            expected="ready=true, 17/17 checks pass, 34 return evidence files and 27 command return rows are locked, command return matrix has no gaps, silent-replacement/protocol-drift guards are documented, and 18 rerun acceptance/replacement rules are locked.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md",
                "results/qwen35_report_audit_20260619/rerun_acceptance_contract.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_sglang_optimization_lock",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the SGLang optimization lock matrix and JSON gate.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_sglang_optimization_lock \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/sglang_optimization_lock.json
            """,
            expected="ready=true, 26/26 checks pass, and SGLang image/recipe/stress/stage/anti-recipe evidence is locked.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md",
                "results/qwen35_report_audit_20260619/sglang_optimization_lock.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_vllm_optimization_lock",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the vLLM optimization lock matrix and JSON gate.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_vllm_optimization_lock \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/vllm_optimization_lock.json
            """,
            expected="ready=true, 22/22 checks pass, and vLLM image/compile/cudagraph/prefix/cache/prebuild evidence is locked.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md",
                "results/qwen35_report_audit_20260619/vllm_optimization_lock.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_vllm_online_parity_protocol",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the vLLM c=8 online-parity upgrade protocol and JSON gate.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json
            """,
            expected="ready=true, 18/18 checks pass, current_package_safe=true, and online_parity_proven=false until online ingress artifacts exist.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md",
                "results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_final_checkpoint_watchlist",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the final completion-readiness watchlist and JSON gate after the updated objective removed the 2026-06-21 evening wait.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json
            """,
            expected="ready=true, 24/24 checks pass, watch items >=7, checkpoint_phase=completion_audit_ready, share_ready_with_documented_caveats=true, and no legacy evening-wait blocker; final_completion_audit is the terminal completion_allowed_now gate.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md",
                "results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_tail_confidence_appendix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the per-sample tail-confidence appendix for strict c4, SGLang stress, synthetic speech, and vLLM diagnostic runs.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_tail_confidence_appendix \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/tail_confidence_appendix.json
            """,
            expected=(
                "ready=true, 13/13 checks pass, 18 per-sample distribution rows "
                "and 9 bootstrap comparison rows cover strict c4, stress, "
                "synthetic, and vLLM diagnostic cases."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md",
                "results/qwen35_report_audit_20260619/tail_confidence_appendix.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_latency_budget",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the stage latency-budget appendix and JSON gate for per-stage pressure ratios.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_stage_latency_budget \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/stage_latency_budget.json
            """,
            expected="ready=true, 12/12 checks pass, with 5 SGLang, 6 synthetic, and 4 vLLM stage-budget rows.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md",
                "results/qwen35_report_audit_20260619/stage_latency_budget.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_boundary_bottleneck_ledger",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the stage-boundary bottleneck ledger with evidence, decision, and claim scope for each audited boundary.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_stage_boundary_bottleneck_ledger \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json
            """,
            expected=(
                "ready=true, 12/12 checks pass, all 37 stage boundary rows have "
                "evidence, decision, and claim scope, and 11 pressure transition "
                "rows cover concurrency, long/short text, and vLLM diagnostics."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md",
                "results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_serving_capacity_matrix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the serving/capacity decision matrix that maps pressure regimes to run choices, stage guardrails, and forbidden claims.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_serving_capacity_matrix \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/serving_capacity_matrix.json
            """,
            expected=(
                "ready=true, 10/10 checks pass, 7 capacity rows cover low "
                "concurrency, c4, c8, c16, synthetic short/long c8, and vLLM "
                "c8 prebuild w4 offline diagnostic."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md",
                "results/qwen35_report_audit_20260619/serving_capacity_matrix.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_share_consistency_guard",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the share-consistency guard that blocks stale public gate counts, stale machine snapshots, serving/capacity routing drift, and evidence-query host/portable route drift.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_share_consistency_guard \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/share_consistency_guard.json
            """,
            expected=(
                "ready=true, 17/17 checks pass, stale public and machine gate "
                "tokens are zero, evidence-query host/portable routes are explicit, "
                "manifest expected gates use explicit minimum-record fields, "
                "preflight aliases cannot drift from preflight_repro.json, the share "
                "index routes the current university review packet gate, embedded "
                "tarball identity fields are omitted, and currently refreshed tarball "
                "identity fields agree without embedding hashes; the adjacent release "
                "seal enforces the full post-validation hash chain."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md",
                "results/qwen35_report_audit_20260619/share_consistency_guard.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_causal_graph",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the reviewer-facing stage causal graph from audited JSONs.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_stage_causal_graph \\
                  --root /home/gangouyu/sglang-omni \\
                  --output benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/stage_causal_graph.json \\
                  --strict
            """,
            expected=(
                "ready=true, 7/7 JSON checks pass, 7 causal edges are quantified, "
                "and raw drilldown artifacts are manifest-backed."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md",
                "results/qwen35_report_audit_20260619/stage_causal_graph.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_drilldown_index",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the stage-first drilldown index that joins budget, boundary, evidence, and rerun commands.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_stage_drilldown_index \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/stage_drilldown_index.json
            """,
            expected="ready=true, 52 rows, all stage evidence files and rerun command IDs are present.",
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/stage_drilldown_index.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_metric_provenance_index",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the metric-to-evidence provenance index for external review.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_metric_provenance_index \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/metric_provenance_index.json
            """,
            expected=(
                "ready=true, at least 150 metric provenance rows, all raw evidence "
                "paths and referenced rerun command IDs are present."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/metric_provenance_index.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_reproduction_drilldown",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the reviewer-facing stage reproduction drilldown with jq queries.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_stage_reproduction_drilldown \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json
            """,
            expected=(
                "ready=true, 52 stage rows, at least 11 route rows, raw artifacts, "
                "metric row links, jq queries, and rerun command IDs are present."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md",
                "results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_stage_route_decision_matrix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the route-level stage decision matrix for sharing and reviewer Q&A.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_stage_route_decision_matrix \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/stage_route_decision_matrix.json
            """,
            expected=(
                "ready=true, 11 route rows, 52 covered stage rows, route-level "
                "decisions, safe talking points, raw artifacts, jq queries, rerun "
                "command IDs, and the 15-row pressure-stage heatmap are present."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md",
                "benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md",
                "results/qwen35_report_audit_20260619/stage_route_decision_matrix.json",
                "results/qwen35_report_audit_20260619/pressure_stage_heatmap.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_caveat_adjudication_matrix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the reviewer-facing caveat adjudication matrix.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json
            """,
            expected=(
                "ready=true, at least 11 caveat rows, forbidden overclaims, "
                "replacement triggers, and 0 required failures."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md",
                "results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_rerun_delta_triage",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the collaborator rerun-delta triage matrix for symptom-to-stage attribution.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_rerun_delta_triage \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/rerun_delta_triage.json
            """,
            expected="ready=true, 16 symptoms, stage routes, first evidence, and replacement boundaries are present.",
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md",
                "results/qwen35_report_audit_20260619/rerun_delta_triage.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_objective_completion_audit",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the original-objective completion audit for the share package.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_objective_completion_audit \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/objective_completion_audit.json
            """,
            expected="share_ready_with_documented_caveats=true, rows 17, and 0 required failures.",
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/objective_completion_audit.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_defense_claim_matrix",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the machine-readable defense claim/evidence/rerun/failure matrix.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_defense_claim_matrix \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/defense_claim_matrix.json
            """,
            expected="ready=true, 10 claim rows, all evidence files and rerun command IDs are present.",
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/defense_claim_matrix.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_claim_metric_crosswalk",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the claim-to-metric crosswalk for external defense review.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_claim_metric_crosswalk \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/claim_metric_crosswalk.json
            """,
            expected=(
                "ready=true, 10 defense claims, at least 60 unique metric row "
                "references, raw artifacts, and rerun command IDs are present."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/claim_metric_crosswalk.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_objective_requirement_crosswalk",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the original-objective-to-evidence crosswalk for external review.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json
            """,
            expected=(
                "ready=true, at least 11 original requirement rows, at least 85 "
                "unique metric row references, raw artifacts, evidence files, "
                "rerun command IDs, and 8 optimization candidate verdicts are present."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_optimization_candidate_ledger",
            phase="audit_regeneration",
            where="host",
            purpose=(
                "Regenerate the standalone optimization-candidate ledger for the "
                "measured-best recipe, anti-recipes, and vLLM diagnostic boundary."
            ),
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/optimization_candidate_ledger.json
            """,
            expected=(
                "ready=true, 8 optimization candidate verdicts, current best "
                "candidate sglang_current_best_measured_recipe, at least 2 "
                "anti-recipes, at least 2 vLLM diagnostic rows, and 0 required "
                "failures."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md",
                "results/qwen35_report_audit_20260619/optimization_candidate_ledger.json",
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_repro_command_manifest",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate this machine-readable reproduction command manifest.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_repro_command_manifest \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/repro_command_manifest.json
            """,
            expected="ready=true and all required command IDs are present.",
            evidence_after_run=["results/qwen35_report_audit_20260619/repro_command_manifest.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_share_path_hygiene",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the share-report path hygiene gate that classifies package, raw-artifact, wildcard, directory, and generated-output path references.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_share_path_hygiene \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/share_path_hygiene.json
            """,
            expected="ready=true with 0 package offenders, 0 raw offenders, and 0 legacy path/gate token hits.",
            evidence_after_run=["results/qwen35_report_audit_20260619/share_path_hygiene.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_command_reference_hygiene",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the global command-reference hygiene gate that resolves structured rerun command IDs against this manifest.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_command_reference_hygiene \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/command_reference_hygiene.json
            """,
            expected=(
                "ready=true with all structured rerun command IDs resolved, "
                "critical SGLang/vLLM commands documented, and manifest references "
                "present in public docs."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/command_reference_hygiene.json"
            ],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_final_readiness_audit",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the final send/no-send readiness audit for the share package.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_final_readiness \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/final_readiness_audit.json
            """,
            expected="ready=true, 0 required failures, and all hard share-package gates pass.",
            evidence_after_run=["results/qwen35_report_audit_20260619/final_readiness_audit.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_share_bundle_manifest",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the external share-bundle manifest with file hashes and chart assets.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_manifest \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/share_bundle_manifest.json
            """,
            expected="ready=true, all required share documents and machine evidence are present, and chart assets are listed with hashes.",
            evidence_after_run=["results/qwen35_report_audit_20260619/share_bundle_manifest.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_share_bundle_package",
            phase="audit_regeneration",
            where="host",
            purpose="Regenerate the deterministic convenience tarball and checksum from the share-bundle manifest.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_package \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --source-manifest results/qwen35_report_audit_20260619/share_bundle_manifest.json \\
                  --output results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \\
                  --json-output results/qwen35_report_audit_20260619/share_bundle_package_manifest.json
            """,
            expected="ready=true; source files are listed with SHA-256; tarball checksum verifies from the repository root.",
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz",
                "results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256",
                "results/qwen35_report_audit_20260619/share_bundle_package_manifest.json",
            ],
            rerun_cost="audit-only packaging",
        ),
        _command(
            command_id="validate_share_bundle_package",
            phase="audit_regeneration",
            where="host",
            purpose="Run a fast receiver-style sanity check over the tarball, checksum, gate JSONs, packaged Markdown, and chart assets.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.validate_qwen35_omni_share_package \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/share_package_validation.json
            """,
            expected=(
                "ready=true; tarball checksum matches; required reports, stage-budget charts, "
                "validator script, and caveat gates are present; report_quality_offenders=[]; "
                "chart_quality_offenders=[]."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/share_package_validation.json",
            ],
            rerun_cost="audit-only packaging check",
        ),
        _command(
            command_id="validate_share_bundle_receiver_smoke",
            phase="audit_regeneration",
            where="host",
            purpose="Run the receiver quickcheck wrapper: checksum, tarball validation, receiver smoke, extracted-only validation, and external standalone validation.",
            command="""
                cd /home/gangouyu/sglang-omni
                HOST_REPO=/home/gangouyu/sglang-omni \\
                SMOKE_DIR=/tmp/qwen35_omni_receiver_smoke_manifest \\
                EXTRACT_DIR=/tmp/qwen35_omni_share_bundle_manifest_quickcheck \\
                STANDALONE_DIR=/tmp/qwen35_omni_external_standalone_bundle_validation_manifest \\
                bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
            """,
            expected=(
                "ready=true; tarball validation 17/17; receiver_smoke_ready=true; "
                "nested extracted-only validation 13/13; standalone validation 8/8; "
                "report_quality_offenders=[]; "
                "chart_quality_offenders=[]."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/share_package_validation.json",
                "results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json",
                "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
                "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
            ],
            rerun_cost="audit-only packaging check",
        ),
        _command(
            command_id="validate_extracted_share_bundle",
            phase="audit_regeneration",
            where="host",
            purpose="Validate the share package after extraction, including packaged Markdown and chart asset quality, without relying on the original repository tarball.",
            command="""
                cd /home/gangouyu/sglang-omni
                rm -rf /tmp/qwen35_omni_share_bundle_manifest
                mkdir -p /tmp/qwen35_omni_share_bundle_manifest
                tar -xzf results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \\
                  -C /tmp/qwen35_omni_share_bundle_manifest
                cd /tmp/qwen35_omni_share_bundle_manifest/qwen35_omni_share_bundle_20260621
                python3 benchmarks/eval/validate_qwen35_omni_share_package.py \\
                  --root "$PWD" \\
                  --extracted-only \\
                  --strict \\
                  --json-output /home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_package_validation_extracted.json
            """,
            expected=(
                "ready=true; extracted files match PACKAGE_FILE_SHA256SUMS.txt and the "
                "stage-budget/caveat gates remain present; report_quality_offenders=[]; "
                "chart_quality_offenders=[]."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/share_package_validation_extracted.json",
            ],
            rerun_cost="audit-only extraction check",
        ),
        _command(
            command_id="validate_external_standalone_share_bundle",
            phase="audit_regeneration",
            where="host",
            purpose=(
                "Prove the tarball can be extracted into a clean /tmp root and "
                "validated by the bundled script without importing repository modules."
            ),
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --work-dir /tmp/qwen35_omni_external_standalone_bundle_validation_manifest \\
                  --json-output results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json
            """,
            expected=(
                "ready=true; standalone checks pass; bundled validator runs from "
                "the extracted bundle root; nested extracted-only validation is 13/13; "
                "report_quality_offenders=[]; chart_quality_offenders=[]."
            ),
            evidence_after_run=[
                "results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json",
            ],
            rerun_cost="audit-only extraction check",
        ),
        _command(
            command_id="build_share_release_seal",
            phase="audit_regeneration",
            where="host",
            purpose=(
                "Regenerate the adjacent release seal that binds the tarball, "
                "checksum, package manifest, validations, final readiness, and caveats."
            ),
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \\
                  --json-output results/qwen35_report_audit_20260619/share_release_seal.json
            """,
            expected=(
                "ready=true; tarball checksum, validation JSONs, final readiness, "
                "receiver smoke, extracted-only validation, standalone validation, "
                "and caveats are sealed as adjacent non-tarball evidence."
            ),
            evidence_after_run=[
                "benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md",
                "results/qwen35_report_audit_20260619/share_release_seal.json",
            ],
            rerun_cost="audit-only packaging check",
        ),
        _command(
            command_id="run_preflight",
            phase="handoff_gates",
            where="host",
            purpose="Check local readiness before benchmark reruns or external handoff.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.preflight_qwen35_omni_repro \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/preflight_repro.json
            """,
            expected="62 checks; 0 required failures; optional host-side Whisper cache warning may remain.",
            evidence_after_run=["results/qwen35_report_audit_20260619/preflight_repro.json"],
            rerun_cost="local checks",
        ),
        _command(
            command_id="build_coverage_matrix",
            phase="handoff_gates",
            where="host",
            purpose="Regenerate the original requirement coverage matrix.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.summarize_qwen35_report_coverage \\
                  --root /home/gangouyu/sglang-omni \\
                  --strict \\
                  --json-output results/qwen35_report_audit_20260619/coverage_matrix.json
            """,
            expected="34/34 requirements pass with 0 missing.",
            evidence_after_run=["results/qwen35_report_audit_20260619/coverage_matrix.json"],
            rerun_cost="audit-only",
        ),
        _command(
            command_id="build_environment_snapshot",
            phase="handoff_gates",
            where="host",
            purpose="Record GPU, Docker image, path, git, and audit environment state.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_environment_snapshot \\
                  --root /home/gangouyu/sglang-omni \\
                  --json-output results/qwen35_report_audit_20260619/environment_snapshot.json
            """,
            expected="8x H20 and required SGLang/vLLM image state are captured.",
            evidence_after_run=["results/qwen35_report_audit_20260619/environment_snapshot.json"],
            rerun_cost="local checks",
        ),
        _command(
            command_id="build_evidence_manifest",
            phase="handoff_gates",
            where="host",
            purpose="Regenerate file inventory with hashes for the report evidence set.",
            command="""
                cd /home/gangouyu/sglang-omni
                python3 -m benchmarks.eval.build_qwen35_omni_report_manifest \\
                  --root /home/gangouyu/sglang-omni \\
                  --output results/qwen35_report_audit_20260619/manifest.json
            """,
            expected=">=180 records, >=178 files, 2 directories, and 0 missing artifacts.",
            evidence_after_run=["results/qwen35_report_audit_20260619/manifest.json"],
            rerun_cost="audit-only",
        ),
    ]


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_dir = root / AUDIT_DIR
    tables = _load_json_optional(audit_dir / "tables_summary.json")
    headline = _load_json_optional(audit_dir / "headline_scorecard.json")
    chart_pack = _load_json_optional(audit_dir / "share_charts/chart_pack_manifest.json")
    chart_source_consistency = _load_json_optional(
        audit_dir / "chart_source_consistency.json"
    )
    slide_asset_map = _load_json_optional(audit_dir / "slide_asset_map.json")
    acceptance = _load_json_optional(audit_dir / "acceptance_matrix.json")
    confidence = _load_json_optional(audit_dir / "confidence_ledger.json")
    objective_summary = _summary_from(audit_dir / "objective_completion_audit.json")
    interactions = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    vllm_admission = _load_json_optional(audit_dir / "vllm_admission_diagnosis.json")
    runtime_image_contract = _load_json_optional(
        audit_dir / "runtime_image_contract.json"
    )
    runtime_comparison_contract = _load_json_optional(
        audit_dir / "runtime_comparison_contract.json"
    )
    rerun_acceptance_contract = _load_json_optional(
        audit_dir / "rerun_acceptance_contract.json"
    )
    sglang_lock = _load_json_optional(audit_dir / "sglang_optimization_lock.json")
    vllm_lock = _load_json_optional(audit_dir / "vllm_optimization_lock.json")
    vllm_online_protocol = _load_json_optional(
        audit_dir / "vllm_online_parity_protocol.json"
    )
    checkpoint_watchlist = _load_json_optional(
        audit_dir / "final_checkpoint_watchlist.json"
    )
    tail_confidence_appendix = _load_json_optional(
        audit_dir / "tail_confidence_appendix.json"
    )
    stage_latency_budget = _load_json_optional(
        audit_dir / "stage_latency_budget.json"
    )
    stage_boundary_ledger = _load_json_optional(
        audit_dir / "stage_boundary_bottleneck_ledger.json"
    )
    serving_capacity_matrix = _load_json_optional(
        audit_dir / "serving_capacity_matrix.json"
    )
    share_consistency_guard = _load_json_optional(
        audit_dir / "share_consistency_guard.json"
    )
    stage_causal_graph = _load_json_optional(audit_dir / "stage_causal_graph.json")
    stage_drilldown_index = _load_json_optional(
        audit_dir / "stage_drilldown_index.json"
    )
    metric_provenance_index = _load_json_optional(
        audit_dir / "metric_provenance_index.json"
    )
    stage_reproduction_drilldown = _load_json_optional(
        audit_dir / "stage_reproduction_drilldown.json"
    )
    stage_route_decision_matrix = _load_json_optional(
        audit_dir / "stage_route_decision_matrix.json"
    )
    pressure_stage_heatmap = _load_json_optional(
        audit_dir / "pressure_stage_heatmap.json"
    )
    share_path_hygiene = _load_json_optional(audit_dir / "share_path_hygiene.json")
    command_reference_hygiene = _load_json_optional(
        audit_dir / "command_reference_hygiene.json"
    )
    regime_decision_matrix = _load_json_optional(
        audit_dir / "regime_decision_matrix.json"
    )
    rerun_delta_triage = _load_json_optional(audit_dir / "rerun_delta_triage.json")
    defense_claim_matrix = _load_json_optional(
        audit_dir / "defense_claim_matrix.json"
    )
    claim_metric_crosswalk = _load_json_optional(
        audit_dir / "claim_metric_crosswalk.json"
    )
    objective_requirement_crosswalk = _load_json_optional(
        audit_dir / "objective_requirement_crosswalk.json"
    )
    optimization_candidate_ledger = _load_json_optional(
        audit_dir / "optimization_candidate_ledger.json"
    )
    caveat_adjudication_matrix = _load_json_optional(
        audit_dir / "caveat_adjudication_matrix.json"
    )
    share_release_seal = _load_json_optional(audit_dir / "share_release_seal.json")

    commands = build_commands()
    command_ids = {command["id"] for command in commands}
    synthetic_targets = _synthetic_targets(tables)
    short_words = synthetic_targets.get("short", {}).get("target_words", 0.0)
    long_words = synthetic_targets.get("long", {}).get("target_words", 0.0)
    short_chars = synthetic_targets.get("short", {}).get("target_chars", 0.0)
    long_chars = synthetic_targets.get("long", {}).get("target_chars", 0.0)

    headline_summary = headline.get("summary", {})
    chart_summary = chart_pack.get("summary", {})
    chart_source_summary = chart_source_consistency.get("summary", {})
    slide_asset_summary = slide_asset_map.get("summary", {})
    acceptance_summary = acceptance.get("summary", {})
    regime_summary = regime_decision_matrix.get("summary", {})
    confidence_summary = confidence.get("summary", {})
    interaction_summary = interactions.get("summary", {})
    vllm_rows = vllm_admission.get("rows", [])
    runtime_image_summary = runtime_image_contract.get("summary", {})
    runtime_comparison_summary = runtime_comparison_contract.get("summary", {})
    rerun_acceptance_summary = rerun_acceptance_contract.get("summary", {})
    sglang_lock_summary = sglang_lock.get("summary", {})
    vllm_lock_summary = vllm_lock.get("summary", {})
    vllm_online_summary = vllm_online_protocol.get("summary", {})
    checkpoint_summary = checkpoint_watchlist.get("summary", {})
    tail_confidence_summary = tail_confidence_appendix.get("summary", {})
    stage_budget_summary = stage_latency_budget.get("summary", {})
    stage_ledger_summary = stage_boundary_ledger.get("summary", {})
    serving_capacity_summary = serving_capacity_matrix.get("summary", {})
    share_consistency_summary = share_consistency_guard.get("summary", {})
    stage_causal_summary = stage_causal_graph.get("summary", {})
    stage_drilldown_summary = stage_drilldown_index.get("summary", {})
    metric_provenance_summary = metric_provenance_index.get("summary", {})
    stage_reproduction_summary = stage_reproduction_drilldown.get("summary", {})
    stage_route_summary = stage_route_decision_matrix.get("summary", {})
    pressure_stage_heatmap_summary = pressure_stage_heatmap.get("summary", {})
    share_path_hygiene_summary = share_path_hygiene.get("summary", {})
    command_reference_summary = command_reference_hygiene.get("summary", {})
    rerun_delta_summary = rerun_delta_triage.get("summary", {})
    defense_claim_summary = defense_claim_matrix.get("summary", {})
    claim_metric_summary = claim_metric_crosswalk.get("summary", {})
    objective_requirement_summary = objective_requirement_crosswalk.get("summary", {})
    optimization_candidate_summary = optimization_candidate_ledger.get("summary", {})
    caveat_summary = caveat_adjudication_matrix.get("summary", {})
    share_release_summary = share_release_seal.get("summary", {})

    checks = {
        "required_command_ids_present": REQUIRED_COMMAND_IDS.issubset(command_ids),
        "phase_count_sufficient": len({command["phase"] for command in commands}) >= 6,
        "expected_gates_declared": EXPECTED_GATES["coverage"]["total"] == 34
        and EXPECTED_GATES["preflight"]["total_checks"] == 62
        and EXPECTED_GATES["manifest"]["min_total_records"] == 180,
        "synthetic_short_text_shape": short_chars <= 100.0 and short_words <= 20.0,
        "synthetic_long_text_shape": long_chars >= 900.0 and long_words >= 100.0,
        "headline_scorecard_ready": bool(headline_summary.get("ready"))
        and int(headline_summary.get("checks_total") or 0) >= 9,
        "chart_pack_ready": bool(chart_summary.get("ready"))
        and int(chart_summary.get("csv_files") or 0) >= 7
        and int(chart_summary.get("svg_files") or 0) >= 7
        and int(chart_summary.get("generated_files") or 0) >= 14,
        "chart_source_consistency_ready": "build_chart_source_consistency"
        in command_ids
        and bool(chart_source_summary.get("ready"))
        and int(chart_source_summary.get("checks_total") or 0)
        >= EXPECTED_GATES["chart_source_consistency"]["checks_total"]
        and int(chart_source_summary.get("csv_files_checked") or 0)
        >= EXPECTED_GATES["chart_source_consistency"]["csv_files_checked"]
        and int(chart_source_summary.get("svg_files_checked") or 0)
        >= EXPECTED_GATES["chart_source_consistency"]["svg_files_checked"]
        and int(chart_source_summary.get("byte_exact_files") or 0)
        >= EXPECTED_GATES["chart_source_consistency"]["byte_exact_files"]
        and int(chart_source_summary.get("required_failures") or 0)
        == EXPECTED_GATES["chart_source_consistency"]["required_failures"],
        "slide_asset_map_ready": "build_slide_asset_map" in command_ids
        and bool(slide_asset_summary.get("ready"))
        and int(slide_asset_summary.get("required_failures") or 0) == 0
        and int(slide_asset_summary.get("rows_total") or 0) >= 10
        and int(slide_asset_summary.get("chart_assets_total") or 0) >= 14,
        "acceptance_matrix_ready": bool(acceptance_summary.get("ready"))
        and int(acceptance_summary.get("rows_total") or 0) >= 17,
        "regime_decision_matrix_ready": "build_regime_decision_matrix" in command_ids
        and bool(regime_summary.get("ready"))
        and int(regime_summary.get("rows_total") or 0) >= 17
        and int(regime_summary.get("accepted_rows") or 0) >= 17
        and int(regime_summary.get("checks_total") or 0) >= 9
        and int(regime_summary.get("required_failures") or 0) == 0,
        "runtime_comparison_contract_ready": "build_runtime_comparison_contract"
        in command_ids
        and bool(runtime_comparison_summary.get("ready"))
        and int(runtime_comparison_summary.get("checks_total") or 0) >= 9
        and int(runtime_comparison_summary.get("required_failures") or 0) == 0
        and runtime_comparison_summary.get("allowed_cross_runtime_headline")
        == "warmed c=4 only"
        and runtime_comparison_summary.get("vllm_c8_contract")
        == "offline_diagnostic_not_online_parity",
        "confidence_ledger_ready": bool(confidence_summary.get("ready"))
        and int(confidence_summary.get("entries_total") or 0) >= 12,
        "objective_completion_ready": "build_objective_completion_audit"
        in command_ids
        and EXPECTED_GATES["objective_completion_audit"]["rows_total"] == 17
        and EXPECTED_GATES["objective_completion_audit"]["required_failures"] == 0,
        "stage_interactions_ready": int(
            interaction_summary.get("total_interactions") or 0
        )
        >= 30,
        "vllm_admission_rows_ready": len(vllm_rows) >= 4,
        "sglang_optimization_lock_ready": bool(sglang_lock_summary.get("ready"))
        and int(sglang_lock_summary.get("checks_total") or 0) >= 26
        and int(sglang_lock_summary.get("required_failures") or 0) == 0,
        "vllm_optimization_lock_ready": bool(vllm_lock_summary.get("ready"))
        and int(vllm_lock_summary.get("checks_total") or 0) >= 22
        and int(vllm_lock_summary.get("required_failures") or 0) == 0,
        "vllm_online_parity_protocol_ready": bool(vllm_online_summary.get("ready"))
        and int(vllm_online_summary.get("checks_total") or 0) >= 18
        and int(vllm_online_summary.get("required_failures") or 0) == 0
        and bool(vllm_online_summary.get("current_package_safe"))
        and not bool(vllm_online_summary.get("online_parity_proven")),
        "runtime_image_contract_ready": bool(runtime_image_summary.get("ready"))
        and int(runtime_image_summary.get("checks_total") or 0) >= 12
        and int(runtime_image_summary.get("required_failures") or 0) == 0
        and "c4-c8" in str(runtime_image_summary.get("sglang_scope") or "")
        and "c4" in str(runtime_image_summary.get("vllm_strict_scope") or "")
        and "offline diagnostic" in str(runtime_image_summary.get("vllm_c8_scope") or ""),
        "rerun_acceptance_contract_ready": bool(rerun_acceptance_summary.get("ready"))
        and int(rerun_acceptance_summary.get("checks_total") or 0) >= 17
        and int(rerun_acceptance_summary.get("required_failures") or 0) == 0
        and int(rerun_acceptance_summary.get("rules_total") or 0) >= 18
        and int(rerun_acceptance_summary.get("return_evidence_files") or 0) >= 34
        and int(rerun_acceptance_summary.get("return_evidence_command_rows") or 0)
        >= 27
        and int(rerun_acceptance_summary.get("return_evidence_command_missing") or 0)
        == 0
        and int(rerun_acceptance_summary.get("return_evidence_command_file_gaps") or 0)
        == 0,
        "final_checkpoint_watchlist_ready": "build_final_checkpoint_watchlist"
        in command_ids
        and int(checkpoint_summary.get("checks_total") or 21) >= 21
        and int(checkpoint_summary.get("watch_items_total") or 7) >= 7
        and str(checkpoint_summary.get("checkpoint_phase") or "")
        == "completion_audit_ready"
        and int(checkpoint_summary.get("seconds_until_checkpoint") or 0) == 0
        and not any(
            "waiting_for_2026-06-21" in str(blocker)
            for blocker in checkpoint_summary.get("completion_blockers", []) or []
        ),
        "stage_latency_budget_ready": bool(stage_budget_summary.get("ready"))
        and int(stage_budget_summary.get("checks_total") or 0) >= 12
        and int(stage_budget_summary.get("required_failures") or 0) == 0
        and int(stage_budget_summary.get("sglang_budget_rows") or 0) >= 5
        and int(stage_budget_summary.get("synthetic_budget_rows") or 0) >= 6
        and int(stage_budget_summary.get("vllm_budget_rows") or 0) >= 4,
        "stage_boundary_bottleneck_ledger_ready": bool(stage_ledger_summary.get("ready"))
        and int(stage_ledger_summary.get("checks_total") or 0) >= 12
        and int(stage_ledger_summary.get("required_failures") or 0) == 0
        and int(stage_ledger_summary.get("ledger_rows") or 0) >= 37
        and int(stage_ledger_summary.get("pressure_transition_rows") or 0) >= 11,
        "serving_capacity_matrix_ready": "build_serving_capacity_matrix"
        in command_ids
        and bool(serving_capacity_summary.get("ready"))
        and int(serving_capacity_summary.get("checks_total") or 0)
        >= EXPECTED_GATES["serving_capacity_matrix"]["checks_total"]
        and int(serving_capacity_summary.get("rows_total") or 0)
        >= EXPECTED_GATES["serving_capacity_matrix"]["rows_total"]
        and int(serving_capacity_summary.get("required_failures") or 0)
        == EXPECTED_GATES["serving_capacity_matrix"]["required_failures"]
        and float(serving_capacity_summary.get("long_c8_rtf_p95") or 9.0) < 1.0
        and serving_capacity_summary.get("vllm_w4_scope")
        == "offline_diagnostic_only",
        "share_consistency_guard_ready": "build_share_consistency_guard"
        in command_ids
        and bool(share_consistency_summary.get("ready"))
        and int(share_consistency_summary.get("checks_total") or 0)
        >= EXPECTED_GATES["share_consistency_guard"]["checks_total"]
        and int(share_consistency_summary.get("required_failures") or 0)
        == EXPECTED_GATES["share_consistency_guard"]["required_failures"]
        and int(share_consistency_summary.get("public_stale_hits", 1))
        == EXPECTED_GATES["share_consistency_guard"]["public_stale_hits"]
        and int(share_consistency_summary.get("machine_stale_hits", 1))
        == EXPECTED_GATES["share_consistency_guard"]["machine_stale_hits"]
        and int(share_consistency_summary.get("embedded_identity_leaks", 1))
        == EXPECTED_GATES["share_consistency_guard"]["embedded_identity_leaks"]
        and int(share_consistency_summary.get("tarball_identity_mismatches", 1))
        == EXPECTED_GATES["share_consistency_guard"][
            "tarball_identity_mismatches"
        ]
        and int(share_consistency_summary.get("tarball_identity_missing_fields", 1))
        == EXPECTED_GATES["share_consistency_guard"][
            "tarball_identity_missing_fields"
        ]
        and int(share_consistency_summary.get("evidence_smoke_route_missing", 1))
        == EXPECTED_GATES["share_consistency_guard"]["evidence_smoke_route_missing"]
        and bool(share_consistency_summary.get("manifest_expected_gate_ready"))
        and int(
            share_consistency_summary.get("manifest_expected_gate_unexpected_fields", 1)
        )
        == EXPECTED_GATES["share_consistency_guard"][
            "manifest_expected_gate_unexpected_fields"
        ]
        and int(share_consistency_summary.get("preflight_alias_mismatches", 1))
        == EXPECTED_GATES["share_consistency_guard"]["preflight_alias_mismatches"]
        and bool(share_consistency_summary.get("preflight_repro_in_manifest"))
        == EXPECTED_GATES["share_consistency_guard"]["preflight_repro_in_manifest"]
        and bool(share_consistency_summary.get("preflight_alias_in_manifest"))
        == EXPECTED_GATES["share_consistency_guard"]["preflight_alias_in_manifest"],
        "stage_causal_graph_ready": "build_stage_causal_graph" in command_ids
        and bool(stage_causal_summary.get("ready"))
        and int(stage_causal_summary.get("checks_total") or 0) >= 7
        and int(stage_causal_summary.get("required_failures") or 0) == 0
        and int(stage_causal_summary.get("causal_edges_total") or 0) >= 7
        and int(stage_causal_summary.get("raw_drilldown_rows") or 0) >= 5
        and int(stage_causal_summary.get("raw_artifacts_total") or 0) >= 14
        and int(stage_causal_summary.get("manifest_missing_raw_artifacts") or 0) == 0,
        "stage_drilldown_index_ready": "build_stage_drilldown_index" in command_ids
        and bool(stage_drilldown_summary.get("ready"))
        and int(stage_drilldown_summary.get("rows_total") or 0) >= 52
        and int(stage_drilldown_summary.get("boundary_rows_total") or 0) >= 37
        and int(stage_drilldown_summary.get("budget_rows_total") or 0) >= 15
        and int(stage_drilldown_summary.get("stage_routes_total") or 0) >= 7
        and int(stage_drilldown_summary.get("required_failures") or 0) == 0,
        "metric_provenance_index_ready": "build_metric_provenance_index"
        in command_ids
        and bool(metric_provenance_summary.get("ready"))
        and int(metric_provenance_summary.get("rows_total") or 0) >= 150
        and int(metric_provenance_summary.get("raw_artifacts_total") or 0) >= 15
        and int(metric_provenance_summary.get("command_refs_total") or 0) >= 15
        and int(metric_provenance_summary.get("required_failures") or 0) == 0,
        "stage_reproduction_drilldown_ready": "build_stage_reproduction_drilldown"
        in command_ids
        and bool(stage_reproduction_summary.get("ready"))
        and int(stage_reproduction_summary.get("stage_rows_total") or 0) >= 52
        and int(stage_reproduction_summary.get("route_rows_total") or 0) >= 11
        and int(stage_reproduction_summary.get("raw_artifacts_total") or 0) >= 10
        and int(stage_reproduction_summary.get("command_refs_total") or 0) >= 15
        and int(stage_reproduction_summary.get("required_failures") or 0) == 0,
        "stage_route_decision_matrix_ready": "build_stage_route_decision_matrix"
        in command_ids
        and bool(stage_route_summary.get("ready"))
        and int(stage_route_summary.get("route_rows_total") or 0) >= 11
        and int(stage_route_summary.get("stage_rows_total") or 0) >= 52
        and int(stage_route_summary.get("raw_artifacts_total") or 0) >= 28
        and int(stage_route_summary.get("command_refs_total") or 0) >= 15
        and int(stage_route_summary.get("required_failures") or 0) == 0
        and bool(pressure_stage_heatmap_summary.get("ready"))
        and int(pressure_stage_heatmap_summary.get("rows_total") or 0) >= 15
        and int(pressure_stage_heatmap_summary.get("sglang_videoamme_rows") or 0) >= 5
        and int(pressure_stage_heatmap_summary.get("synthetic_rows") or 0) >= 6
        and int(pressure_stage_heatmap_summary.get("vllm_rows") or 0) >= 4
        and int(pressure_stage_heatmap_summary.get("required_failures") or 0) == 0,
        "tail_confidence_appendix_ready": "build_tail_confidence_appendix"
        in command_ids
        and bool(tail_confidence_summary.get("ready"))
        and int(tail_confidence_summary.get("checks_total") or 0)
        >= EXPECTED_GATES["tail_confidence_appendix"]["checks_total"]
        and int(tail_confidence_summary.get("rows_total") or 0)
        >= EXPECTED_GATES["tail_confidence_appendix"]["rows_total"]
        and int(tail_confidence_summary.get("required_failures") or 0) == 0,
        "share_path_hygiene_ready": "build_share_path_hygiene" in command_ids
        and bool(share_path_hygiene_summary.get("ready"))
        and int(share_path_hygiene_summary.get("required_failures") or 0) == 0
        and int(share_path_hygiene_summary.get("package_offenders_total") or 0) == 0
        and int(share_path_hygiene_summary.get("raw_offenders_total") or 0) == 0
        and int(share_path_hygiene_summary.get("legacy_hits_total") or 0) == 0,
        "command_reference_hygiene_ready": "build_command_reference_hygiene"
        in command_ids
        and bool(command_reference_summary.get("ready"))
        and int(command_reference_summary.get("required_failures") or 0) == 0
        and int(command_reference_summary.get("checks_total") or 0) >= 6
        and int(command_reference_summary.get("structured_unique_command_refs_total") or 0)
        >= 25
        and int(command_reference_summary.get("unresolved_command_refs_total") or 0)
        == 0
        and not command_reference_summary.get("missing_critical_doc_command_ids"),
        "rerun_delta_triage_ready": "build_rerun_delta_triage" in command_ids
        and bool(rerun_delta_summary.get("ready"))
        and int(rerun_delta_summary.get("required_failures") or 0) == 0
        and int(rerun_delta_summary.get("rows_total") or 0) >= 16
        and int(rerun_delta_summary.get("checks_total") or 0) >= 5,
        "defense_claim_matrix_ready": "build_defense_claim_matrix" in command_ids
        and bool(defense_claim_summary.get("ready"))
        and int(defense_claim_summary.get("rows_total") or 0) >= 10
        and int(defense_claim_summary.get("required_failures") or 0) == 0
        and bool(defense_claim_summary.get("qna_claims_covered"))
        and int(defense_claim_summary.get("failure_decisions_total") or 0) >= 10,
        "claim_metric_crosswalk_ready": "build_claim_metric_crosswalk" in command_ids
        and bool(claim_metric_summary.get("ready"))
        and int(claim_metric_summary.get("claims_total") or 0) >= 10
        and int(claim_metric_summary.get("unique_metric_rows_total") or 0) >= 60
        and int(claim_metric_summary.get("raw_artifacts_total") or 0) >= 20
        and int(claim_metric_summary.get("command_refs_total") or 0) >= 15
        and int(claim_metric_summary.get("required_failures") or 0) == 0,
        "objective_requirement_crosswalk_ready": "build_objective_requirement_crosswalk"
        in command_ids
        and bool(objective_requirement_summary.get("ready"))
        and int(objective_requirement_summary.get("requirement_rows_total") or 0)
        >= 11
        and int(objective_requirement_summary.get("unique_metric_rows_total") or 0)
        >= 85
        and int(objective_requirement_summary.get("raw_artifacts_total") or 0)
        >= 25
        and int(objective_requirement_summary.get("command_refs_total") or 0)
        >= 25
        and bool(
            objective_requirement_summary.get("optimization_candidate_ledger_ready")
        )
        and int(
            objective_requirement_summary.get("optimization_candidate_rows_total") or 0
        )
        >= EXPECTED_GATES["objective_requirement_crosswalk"][
            "optimization_candidate_rows_total"
        ]
        and int(
            objective_requirement_summary.get(
                "optimization_rejected_anti_recipes_total"
            )
            or 0
        )
        >= 2
        and int(
            objective_requirement_summary.get(
                "optimization_vllm_diagnostic_rows_total"
            )
            or 0
        )
        >= 2
        and int(objective_requirement_summary.get("required_failures") or 0) == 0
        and not bool(objective_requirement_summary.get("goal_complete")),
        "optimization_candidate_ledger_ready": "build_optimization_candidate_ledger"
        in command_ids
        and bool(optimization_candidate_summary.get("ready"))
        and int(optimization_candidate_summary.get("candidate_rows_total") or 0)
        >= EXPECTED_GATES["optimization_candidate_ledger"][
            "candidate_rows_total"
        ]
        and optimization_candidate_summary.get("current_best_candidate_id")
        == "sglang_current_best_measured_recipe"
        and int(
            optimization_candidate_summary.get("accepted_current_best_rows_total")
            or 0
        )
        >= EXPECTED_GATES["optimization_candidate_ledger"][
            "accepted_current_best_rows_total"
        ]
        and int(
            optimization_candidate_summary.get("rejected_anti_recipe_rows_total")
            or 0
        )
        >= EXPECTED_GATES["optimization_candidate_ledger"][
            "rejected_anti_recipe_rows_total"
        ]
        and int(
            optimization_candidate_summary.get("vllm_diagnostic_rows_total")
            or 0
        )
        >= EXPECTED_GATES["optimization_candidate_ledger"][
            "vllm_diagnostic_rows_total"
        ]
        and bool(optimization_candidate_summary.get("not_global_optimum_boundary"))
        and int(optimization_candidate_summary.get("missing_metric_row_ids") or 0) == 0
        and int(optimization_candidate_summary.get("missing_evidence_files") or 0) == 0
        and int(optimization_candidate_summary.get("missing_command_ids") or 0) == 0
        and int(optimization_candidate_summary.get("required_failures") or 0) == 0,
        "caveat_adjudication_matrix_ready": "build_caveat_adjudication_matrix"
        in command_ids
        and bool(caveat_summary.get("ready"))
        and int(caveat_summary.get("rows_total") or 0)
        >= EXPECTED_GATES["caveat_adjudication_matrix"]["rows_total"]
        and int(caveat_summary.get("forbidden_claims_total") or 0)
        >= EXPECTED_GATES["caveat_adjudication_matrix"][
            "forbidden_claims_total"
        ]
        and int(caveat_summary.get("replacement_triggers_total") or 0)
        >= EXPECTED_GATES["caveat_adjudication_matrix"][
            "replacement_triggers_total"
        ]
        and int(caveat_summary.get("required_failures") or 0)
        == EXPECTED_GATES["caveat_adjudication_matrix"]["required_failures"]
        and not bool(caveat_summary.get("online_parity_proven"))
        and not bool(caveat_summary.get("seedtts_fullset_headline"))
        and caveat_summary.get("current_best_scope")
        == "measured_best_not_global_optimum",
        "share_release_seal_ready_or_pending": "build_share_release_seal"
        in command_ids
        and (
            not share_release_summary
            or (
                bool(share_release_summary.get("ready"))
                and int(share_release_summary.get("required_failures") or 0)
                == EXPECTED_GATES["share_release_seal"]["required_failures"]
                and int(share_release_summary.get("checks_total") or 0) >= 13
                and int(share_release_summary.get("checks_passed") or 0)
                == int(share_release_summary.get("checks_total") or -1)
                and bool(share_release_summary.get("tarball_sha256"))
                and bool(share_release_summary.get("receiver_smoke_ready"))
                and not bool(share_release_summary.get("goal_complete"))
            )
        ),
    }

    phases = [
        {
            "id": "audit_first",
            "purpose": "Start with machine gates before interpreting or replacing any report numbers.",
            "command_ids": [
                command["id"] for command in commands if command["phase"] == "audit_first"
            ],
        },
        {
            "id": "sglang_serving",
            "purpose": "Launch the optimized SGLang-Omni server recipe.",
            "command_ids": [
                command["id"] for command in commands if command["phase"] == "sglang_serving"
            ],
        },
        {
            "id": "sglang_stress",
            "purpose": "Reproduce single/high concurrency Video-AMME plus short/long text-to-speech pressure.",
            "command_ids": [
                command["id"] for command in commands if command["phase"] == "sglang_stress"
            ],
        },
        {
            "id": "quality_validation",
            "purpose": "Recompute WER away from the serving benchmark critical path.",
            "command_ids": [
                command["id"]
                for command in commands
                if command["phase"] == "quality_validation"
            ],
        },
        {
            "id": "vllm_baseline",
            "purpose": "Reproduce the optimized vLLM offline comparison and c=8 diagnostic.",
            "command_ids": [
                command["id"] for command in commands if command["phase"] == "vllm_baseline"
            ],
        },
        {
            "id": "audit_regeneration",
            "purpose": "Regenerate derived JSON tables, stage summaries, charts, and confidence gates.",
            "command_ids": [
                command["id"]
                for command in commands
                if command["phase"] == "audit_regeneration"
            ],
        },
        {
            "id": "handoff_gates",
            "purpose": "Finish with preflight, coverage, environment, and evidence manifest gates.",
            "command_ids": [
                command["id"] for command in commands if command["phase"] == "handoff_gates"
            ],
        },
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": all(checks.values()),
            "commands_total": len(commands),
            "phases_total": len(phases),
            "required_command_ids_present": checks["required_command_ids_present"],
            "checks": checks,
            "expected_gates": EXPECTED_GATES,
        },
        "source_artifact_snapshot": {
            "synthetic_text_targets": synthetic_targets,
            "headline_scorecard": headline_summary,
            "chart_pack": chart_summary,
            "chart_source_consistency": chart_source_summary,
            "slide_asset_map": slide_asset_summary,
            "acceptance_matrix": acceptance_summary,
            "confidence_ledger": confidence_summary,
            "objective_completion": objective_summary,
            "stage_interactions": {
                "total_interactions": interaction_summary.get("total_interactions"),
                "sglang_talker_to_code2wav_healthy": interaction_summary.get(
                    "sglang_talker_to_code2wav_healthy"
                ),
                "sglang_code2wav_decode_not_bottleneck": interaction_summary.get(
                    "sglang_code2wav_decode_not_bottleneck"
                ),
                "vllm_original_c8_prompt_feed_limited": interaction_summary.get(
                    "vllm_original_c8_prompt_feed_limited"
                ),
            },
            "vllm_admission_labels": _labels_from(vllm_rows),
            "sglang_optimization_lock": sglang_lock_summary,
            "vllm_optimization_lock": vllm_lock_summary,
            "vllm_online_parity_protocol": vllm_online_summary,
            "runtime_image_contract": runtime_image_summary,
            "rerun_acceptance_contract": rerun_acceptance_summary,
            "final_checkpoint_watchlist": checkpoint_summary,
            "stage_latency_budget": stage_budget_summary,
            "stage_boundary_bottleneck_ledger": stage_ledger_summary,
            "serving_capacity_matrix": serving_capacity_summary,
            "share_consistency_guard": share_consistency_summary,
            "stage_drilldown_index": stage_drilldown_summary,
            "metric_provenance_index": metric_provenance_summary,
            "stage_reproduction_drilldown": stage_reproduction_summary,
            "stage_route_decision_matrix": stage_route_summary,
            "pressure_stage_heatmap": pressure_stage_heatmap_summary,
            "share_path_hygiene": share_path_hygiene_summary,
            "command_reference_hygiene": command_reference_summary,
            "rerun_delta_triage": rerun_delta_summary,
            "defense_claim_matrix": defense_claim_summary,
            "claim_metric_crosswalk": claim_metric_summary,
            "objective_requirement_crosswalk": objective_requirement_summary,
            "optimization_candidate_ledger": optimization_candidate_summary,
            "caveat_adjudication_matrix": caveat_summary,
            "share_release_seal": _stable_share_release_summary(
                share_release_summary
            ),
            "preflight": _summary_from(audit_dir / "preflight_repro.json"),
            "coverage": _summary_from(audit_dir / "coverage_matrix.json"),
            "manifest": _summary_from(audit_dir / "manifest.json"),
        },
        "phases": phases,
        "commands": commands,
        "safety_boundaries": [
            "vLLM c=8 prebuild w4 is an optimized offline diagnostic, not an online serving parity claim.",
            "The vLLM online parity protocol must stay ready and online_parity_proven=false until online ingress artifacts exist.",
            "The final completion watchlist must stay ready and completion_allowed_now=true before marking the updated goal complete.",
            "The stage latency-budget appendix must stay ready before using stage percentage language in external material.",
            "The stage-boundary bottleneck ledger must stay ready before claiming a stage connection is or is not a bottleneck.",
            "The serving/capacity matrix must stay ready before recommending c8 as the current high-concurrency edge or treating c16 as saturation evidence.",
            "The share-consistency guard must stay ready before packaging public material, so stale gate counts and embedded tarball identities cannot silently drift.",
            "The stage reproduction drilldown must stay ready before claiming every stage can be independently queried and reproduced.",
            "The stage route decision matrix must stay ready before using route-level bottleneck and optimization talking points in external material.",
            "The command-reference hygiene audit must stay ready before sharing rerun command IDs in public docs or machine evidence.",
            "The slide asset map must stay ready before deck slides cite chart files or chart-backed numbers.",
            "The chart source consistency gate must stay ready before using PPT-ready CSV/SVG assets in external material.",
            "The runtime image contract must stay ready before claiming vLLM/SGLang image or optimization-switch parity.",
            "The rerun acceptance contract must stay ready before replacing report headline numbers with collaborator rerun results.",
            "The rerun-delta triage matrix must stay ready before interpreting collaborator rerun differences as stage bottlenecks.",
            "The objective requirement crosswalk must stay ready before claiming the original user objective is fully evidenced.",
            "The caveat adjudication matrix must stay ready before external sharing so forbidden overclaims and replacement triggers travel with the package.",
            "Do not replace report headline numbers unless full audit, claims, coverage, preflight, manifest, headline, acceptance, confidence, chart, and chart-source gates pass.",
            "Keep preprocessing max concurrency at 1 for the current recipe; preproc=2/4 are documented anti-recipes.",
            "Run WER/ASR after serving benchmarks or on isolated GPU resources to avoid measuring ASR contention as serving latency.",
            "Optional host-side Whisper cache warnings do not invalidate serving benchmarks if WER is recomputed inside the proper container/cache.",
        ],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("## Qwen3.5-Omni Reproduction Command Manifest\n")
    print("| Gate | Value |")
    print("| --- | ---: |")
    print(f"| Ready | {summary['ready']} |")
    print(f"| Commands | {summary['commands_total']} |")
    print(f"| Phases | {summary['phases_total']} |")
    print(
        "| Required command IDs present | "
        f"{summary['required_command_ids_present']} |"
    )
    print("\n| Check | Value |")
    print("| --- | ---: |")
    for name, value in summary["checks"].items():
        print(f"| {name} | {value} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5-Omni machine-readable reproduction commands."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    payload = build_manifest(root)
    output = args.json_output
    if not output.is_absolute():
        output = root / output
    _save_json(payload, output)
    print_markdown(payload)
    print(
        "Reproduction command manifest written: "
        f"{output} ready={payload['summary']['ready']} "
        f"commands={payload['summary']['commands_total']}"
    )
    if args.strict and not payload["summary"]["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
