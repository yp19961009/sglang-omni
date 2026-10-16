#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="${HOST_REPO:-${BUNDLE_ROOT:-$(pwd)}}"
AUDIT_DIR="${AUDIT_DIR:-results/qwen35_report_audit_20260619}"
CARD="${CARD:-}"
MODE="${MODE:-auto}"

die() {
  printf 'evidence-card smoke error: %s\n' "$*" >&2
  exit 1
}

need_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

usage() {
  cat <<'EOF'
Usage: qwen35_omni_evidence_query_cards_smoke.sh [options]

Options:
  --root PATH        Repository or extracted bundle root. Overrides HOST_REPO/BUNDLE_ROOT.
  --bundle-root PATH Alias for --root.
  --audit-dir PATH   Audit directory under root. Default: results/qwen35_report_audit_20260619.
  --card PATH        Evidence query card markdown path.
  --mode MODE        auto, host, or portable.
  --output PATH      File used to capture query-card command output; PASS/skip summary stays on stdout.
  -h, --help         Show this help.
EOF
}

run_jq_check() {
  local label="$1"
  local filter="$2"
  local file="$3"
  need_file "$file"
  jq -e "$filter" "$file" >/dev/null || die "jq check failed: $label"
  printf 'PASS %s\n' "$label"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root|--bundle-root)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ROOT="$2"
      shift 2
      ;;
    --audit-dir)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      AUDIT_DIR="$2"
      shift 2
      ;;
    --card)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      CARD="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      MODE="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      OUTPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

command -v jq >/dev/null 2>&1 || die "jq is required"

ROOT="$(cd "$ROOT" && pwd)"
CARD="${CARD:-${ROOT}/benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md}"
CARD="${CARD/#\~/$HOME}"
if [[ -z "${OUTPUT:-}" ]]; then
  OUTPUT="$(mktemp /tmp/qwen35_omni_evidence_query_cards_smoke.XXXXXX.out)"
else
  OUTPUT="${OUTPUT/#\~/$HOME}"
fi
mkdir -p "$(dirname "$OUTPUT")"
need_file "$CARD"

share_package_json="${ROOT}/${AUDIT_DIR}/share_package_validation.json"
receiver_smoke_json="${ROOT}/${AUDIT_DIR}/share_package_receiver_smoke_validation.json"
if [[ "$MODE" == "auto" ]]; then
  if [[ -f "$share_package_json" && -f "$receiver_smoke_json" ]]; then
    MODE="host"
  else
    MODE="portable"
  fi
fi
[[ "$MODE" == "host" || "$MODE" == "portable" ]] || die "MODE must be host, portable, or auto"

tmp_script="$(mktemp /tmp/qwen35_evidence_cards.XXXXXX.sh)"
trap 'rm -f "$tmp_script"' EXIT
root_escaped="$(printf '%q' "$ROOT")"

awk -v mode="$MODE" -v root_escaped="$root_escaped" '
  /^```bash$/ {
    in_block = 1
    block += 1
    next
  }
  /^```$/ {
    in_block = 0
    next
  }
  in_block {
    if (mode == "portable" && block == 10) {
      next
    }
    if ($0 == "cd /home/gangouyu/sglang-omni") {
      print "cd " root_escaped
      next
    }
    print
  }
' "$CARD" > "$tmp_script"

printf 'evidence query card smoke\n'
printf 'root=%s\n' "$ROOT"
printf 'card=%s\n' "$CARD"
printf 'mode=%s\n' "$MODE"
printf 'output=%s\n' "$OUTPUT"
printf 'summary_output=stdout (--output captures query-card command output only)\n'
if [[ "$MODE" == "portable" ]]; then
  printf 'portable mode skips card 10 because adjacent package-validation JSONs are not bundled.\n'
fi

bash "$tmp_script" > "$OUTPUT"
printf 'query_blocks_executed=OK\n'
printf 'query_output_lines='
wc -l < "$OUTPUT"

audit_json="${ROOT}/${AUDIT_DIR}/audit_run_summary.json"
final_readiness_json="${ROOT}/${AUDIT_DIR}/final_readiness_audit.json"
scorecard_json="${ROOT}/${AUDIT_DIR}/headline_scorecard.json"
length_regime_json="${ROOT}/${AUDIT_DIR}/length_regime_coverage.json"
tables_json="${ROOT}/${AUDIT_DIR}/tables_summary.json"
stage_interaction_json="${ROOT}/${AUDIT_DIR}/stage_interaction_summary.json"
stage_budget_json="${ROOT}/${AUDIT_DIR}/stage_latency_budget.json"
stage_boundary_json="${ROOT}/${AUDIT_DIR}/stage_boundary_bottleneck_ledger.json"
stage_reproduction_json="${ROOT}/${AUDIT_DIR}/stage_reproduction_drilldown.json"
vllm_lock_json="${ROOT}/${AUDIT_DIR}/vllm_optimization_lock.json"
vllm_parity_json="${ROOT}/${AUDIT_DIR}/vllm_online_parity_protocol.json"
rerun_contract_json="${ROOT}/${AUDIT_DIR}/rerun_acceptance_contract.json"
rerun_time_budget_json="${ROOT}/${AUDIT_DIR}/rerun_time_budget.json"
receiver_contract_json="${ROOT}/${AUDIT_DIR}/receiver_quickcheck_contract.json"
share_consistency_json="${ROOT}/${AUDIT_DIR}/share_consistency_guard.json"
sglang_lock_json="${ROOT}/${AUDIT_DIR}/sglang_optimization_lock.json"
optimization_ledger_json="${ROOT}/${AUDIT_DIR}/optimization_candidate_ledger.json"

run_jq_check \
  "audit summary is readable" \
  '.ok == true' \
  "$audit_json"
run_jq_check \
  "final readiness direct evidence" \
  '.summary.ready == true and .summary.checks_passed == 49 and .summary.required_failures == 0' \
  "$final_readiness_json"
run_jq_check \
  "public docs semantic count guard is present" \
  '.summary.hard_gates.public_doc_quality_guard == "no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift" and ([.checks[] | select(.name == "public share docs quality guard" and .status == "PASS" and (.evidence | contains("semantic_offenders=[]")) and (.evidence | contains("duplicate_heading_offenders=[]")))] | length) == 1' \
  "$final_readiness_json"
run_jq_check \
  "strict c4 SGLang beats optimized vLLM without quality regression" \
  '.strict_c4_comparison.sglang.latency_mean_s < .strict_c4_comparison.vllm.latency_mean_s and .strict_c4_comparison.sglang.rtf_mean < .strict_c4_comparison.vllm.rtf_mean and .strict_c4_comparison.sglang.accuracy >= .strict_c4_comparison.vllm.accuracy and .strict_c4_comparison.sglang.wer_corpus <= .strict_c4_comparison.vllm.wer_corpus' \
  "$scorecard_json"
run_jq_check \
  "SGLang c8 is peak and c16 is saturation boundary" \
  '.sglang_stress.throughput_peak.concurrency == 8 and .sglang_stress.c16_vs_c8_qps_delta_pct < 0' \
  "$scorecard_json"
run_jq_check \
  "length-regime coverage gate is present" \
  '.summary.ready == true and .summary.checks_passed == 10 and .summary.required_failures == 0 and .summary.rows_total >= 7 and .summary.short_target_chars == 74 and .summary.long_target_chars == 944 and .summary.long_c8_rtf_p95 < 1 and .summary.max_talker_to_code2wav_hop_p95_ms <= 30 and .summary.max_code2wav_decode_p95_ms <= 30' \
  "$length_regime_json"
run_jq_check \
  "short and long synthetic speech table coverage is present" \
  '(.tables.synthetic_speech | length) >= 6 and ([.tables.synthetic_speech[] | select(.scenario == "long" and .concurrency == 8 and .rtf_mean < 1)] | length) >= 1' \
  "$tables_json"
run_jq_check \
  "stage connections and pressure diagnosis are healthy" \
  '.summary.sglang_talker_to_code2wav_healthy == true and .summary.sglang_code2wav_decode_not_bottleneck == true and .summary.vllm_original_c8_prompt_feed_limited == true and .summary.preprocessing_parallelism_regresses == true' \
  "$stage_interaction_json"
run_jq_check \
  "stage latency budget gate" \
  '.summary.ready == true and .summary.checks_passed == 12 and .summary.required_failures == 0' \
  "$stage_budget_json"
run_jq_check \
  "stage boundary bottleneck ledger gate" \
  '.summary.ready == true and .summary.checks_passed == 12 and .summary.required_failures == 0 and .summary.recommended_sglang_window == "c4-c8" and .summary.saturation_boundary == "c16"' \
  "$stage_boundary_json"
run_jq_check \
  "stage reproduction quick routes are present" \
  '.summary.ready == true and .summary.checks_passed == .summary.checks_total and .summary.checks_total >= 17 and .summary.required_failures == 0 and .summary.quick_reproduction_routes_total >= 5 and ([.quick_reproduction_map[] | select((.question // "") != "" and (.stage_row_id // "") != "" and (.metric_row_id // "") != "" and (.first_rerun_command_id // "") != "" and ((.stage_query // "") | contains("stage_drilldown_index.json")) and ((.metric_query // "") | contains("metric_provenance_index.json")))] | length) >= 5' \
  "$stage_reproduction_json"
run_jq_check \
  "optimized vLLM baseline lock" \
  '.summary.ready == true and .summary.checks_passed == 22 and .summary.required_failures == 0' \
  "$vllm_lock_json"
run_jq_check \
  "vLLM c8 stays diagnostic until online parity artifacts exist" \
  '.summary.ready == true and .summary.checks_passed == 18 and .summary.online_parity_proven == false' \
  "$vllm_parity_json"
run_jq_check \
  "rerun replacement contract is present" \
  '.summary.ready == true and .summary.checks_passed == 17 and .summary.return_evidence_files == 34 and .summary.return_evidence_command_rows == 27' \
  "$rerun_contract_json"
run_jq_check \
  "rerun time and compute budget is present" \
  '.summary.ready == true and .summary.rows_total == 9 and .summary.timed_rows == 6 and .summary.required_failures == 0 and .summary.total_timed_benchmark_wall_s > 0 and .summary.equivalent_8gpu_timed_lower_bound_gpu_hours > 0 and (.summary.caveat | contains("WER/ASR"))' \
  "$rerun_time_budget_json"
run_jq_check \
  "receiver quickcheck contract is present" \
  '.summary.ready == true and .summary.checks_passed == .summary.checks_total and .summary.checks_total >= 14 and .summary.required_failures == 0' \
  "$receiver_contract_json"
run_jq_check \
  "share consistency and preflight alias guard is present" \
  '.summary.ready == true and .summary.checks_passed == .summary.checks_total and .summary.checks_total >= 17 and .summary.required_failures == 0 and .summary.public_stale_hits == 0 and .summary.machine_stale_hits == 0 and .summary.university_review_packet_checks == "14/14 checks" and (.summary.university_review_packet_glossary_missing | length) == 0 and .summary.manifest_expected_gate_unexpected_fields == 0 and .summary.preflight_alias_mismatches == 0 and .summary.preflight_repro_in_manifest == true and .summary.preflight_alias_in_manifest == false and .summary.evidence_smoke_route_missing == 0' \
  "$share_consistency_json"
run_jq_check \
  "SGLang optimization lock is present" \
  '.summary.ready == true and .summary.checks_passed == 26 and .summary.required_failures == 0 and (.summary.recommended_window | contains("c4-c8")) and (.summary.recommended_window | contains("c16")) and (.recipe_switches | length) >= 5 and ([.recipe_switches[].switch | select(contains("PREPROCESSING_MAX_CONCURRENCY=1"))] | length) == 1 and ([.stress_rows[] | select(.concurrency == 8 and .throughput_qps > 2.5)] | length) == 1 and ([.stress_rows[] | select(.concurrency == 16 and .throughput_qps < 2.5)] | length) == 1' \
  "$sglang_lock_json"
run_jq_check \
  "SGLang current-best and anti-recipes are locked" \
  '.summary.ready == true and .summary.current_best_candidate_id == "sglang_current_best_measured_recipe" and .summary.accepted_current_best_rows_total == 1 and .summary.rejected_anti_recipe_rows_total == 2 and ([.rows[] | select(.candidate_id == "sglang_current_best_measured_recipe" and .decision == "accept_current_recipe")] | length) == 1 and ([.rows[] | select(.decision_class == "rejected_anti_recipe")] | length) == 2' \
  "$optimization_ledger_json"

if [[ "$MODE" == "host" ]]; then
  run_jq_check \
    "host tarball validation is green" \
    '.summary.ready == true and .summary.checks_passed == 17 and .summary.required_failures == 0' \
    "$share_package_json"
  run_jq_check \
    "host receiver smoke validation is green" \
    '.summary.ready == true and .summary.checks_passed == 17 and .summary.receiver_smoke_ready == true' \
    "$receiver_smoke_json"
fi

printf 'evidence query card smoke complete\n'
