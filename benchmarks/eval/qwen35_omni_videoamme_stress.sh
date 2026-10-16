#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Qwen3.5-Omni Video-AMME stress benchmarks and summarize performance/stages.

Default:
  benchmarks/eval/qwen35_omni_videoamme_stress.sh

Useful overrides:
  CONCURRENCIES="1 4 8" MAX_SAMPLES=10 benchmarks/eval/qwen35_omni_videoamme_stress.sh
  WARMUP_SAMPLES=8 WARMUP_REPO_ID=/path/to/holdout_videoamme WARMUP_CONCURRENCIES="1 8" benchmarks/eval/qwen35_omni_videoamme_stress.sh
  WARMUP_SAMPLES=8 WARMUP_SAMPLE_OFFSET=50 WARMUP_CONCURRENCIES="1 8" benchmarks/eval/qwen35_omni_videoamme_stress.sh
  RUN_ROOT=results/my_videoamme_run benchmarks/eval/qwen35_omni_videoamme_stress.sh
  ANALYZE_ONLY=1 RUN_ROOT=results/my_videoamme_run benchmarks/eval/qwen35_omni_videoamme_stress.sh

Outputs:
  ${RUN_ROOT}/performance_table.tsv
  ${RUN_ROOT}/performance_table.md
  ${RUN_ROOT}/stage_breakdown.tsv
  ${RUN_ROOT}/stage_breakdown_top_by_avg_ms.tsv
  ${RUN_ROOT}/stage_breakdown_top_by_p95_ms.tsv
  ${RUN_ROOT}/hop_breakdown.tsv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

find_repo_root() {
  if [[ -n "${ROOT:-}" ]]; then
    printf '%s\n' "${ROOT}"
    return
  fi
  if [[ -d /myapp/sglang-omni && -f /myapp/sglang-omni/benchmarks/eval/benchmark_omni_videoamme.py ]]; then
    printf '%s\n' /myapp/sglang-omni
    return
  fi
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

ROOT="$(find_repo_root)"
cd "${ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8161}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-qwen3_5-omni}"
REPO_ID="${REPO_ID:-zhaochenyang20/Video_AMME_ci}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16}"
SWEEP_CACHE_NOTE="${SWEEP_CACHE_NOTE:-same-server sequential sweep; repeated measured samples can hit server-side video preprocessing/prefix caches after the first concurrency}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-0}"
WARMUP_REPO_ID="${WARMUP_REPO_ID:-$REPO_ID}"
WARMUP_SAMPLE_OFFSET="${WARMUP_SAMPLE_OFFSET:-}"
WARMUP_ALLOW_MEASURED_OVERLAP="${WARMUP_ALLOW_MEASURED_OVERLAP:-0}"
WARMUP_CONCURRENCIES="${WARMUP_CONCURRENCIES:-1}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-${MAX_TOKENS:-256}}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-401408}"
AUDIO_VOICE="${AUDIO_VOICE:-m02}"
SKIP_WER="${SKIP_WER:-1}"
PROFILE="${PROFILE:-1}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
ANALYZER="${ANALYZER:-1}"
RUN_ROOT="${RUN_ROOT:-results/qwen35_sglang_videoamme_stress_$(date +%Y%m%d_%H%M%S)}"
EVENT_DIR="${EVENT_DIR:-$(pwd)/${RUN_ROOT}/events}"
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

export HF_HOME="${HF_HOME:-/myapp/data/videoamme}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

check_deps() {
  require_cmd "${PYTHON}"
  if [[ "${ANALYZE_ONLY}" != "1" ]]; then
    require_cmd curl
  fi
}

check_server() {
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    return
  fi
  echo "ERROR: server is not reachable at ${BASE_URL}/v1/models" >&2
  echo "Start the qwen35-omni SGLang server on PORT=${PORT}, then rerun this script." >&2
  exit 1
}

check_warmup_overlap_policy() {
  [[ "${WARMUP_SAMPLES}" != "0" ]] || return 0
  [[ "${WARMUP_ALLOW_MEASURED_OVERLAP}" != "1" ]] || return 0
  [[ "${WARMUP_REPO_ID}" == "${REPO_ID}" ]] || return 0

  if [[ -z "${WARMUP_SAMPLE_OFFSET}" ]]; then
    echo "ERROR: WARMUP_SAMPLE_OFFSET is required for dataset-clean warmup." >&2
    echo "Set WARMUP_SAMPLE_OFFSET outside the measured slice, or set WARMUP_ALLOW_MEASURED_OVERLAP=1 for cache-hot experiments only." >&2
    exit 1
  fi

  local measured_start measured_end warmup_start warmup_end
  measured_start=$((SAMPLE_OFFSET))
  measured_end=$((SAMPLE_OFFSET + MAX_SAMPLES))
  warmup_start=$((WARMUP_SAMPLE_OFFSET))
  warmup_end=$((WARMUP_SAMPLE_OFFSET + WARMUP_SAMPLES))

  if (( warmup_start < measured_end && measured_start < warmup_end )); then
    echo "ERROR: warmup sample slice overlaps measured slice." >&2
    echo "Measured slice: [${measured_start}, ${measured_end}); warmup slice: [${warmup_start}, ${warmup_end})." >&2
    echo "Use non-overlapping WARMUP_SAMPLE_OFFSET, or set WARMUP_ALLOW_MEASURED_OVERLAP=1 for cache-hot experiments only." >&2
    exit 1
  fi
}

start_request_profile() {
  local run_id="$1"
  [[ "${PROFILE}" == "1" ]] || return 0
  curl -fsS "${BASE_URL}/start_request_profile" \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${run_id}\",\"event_dir\":\"${EVENT_DIR}\"}" \
    >/dev/null || echo "WARN: failed to start request profile for ${run_id}" >&2
}

stop_request_profile() {
  local run_id="$1"
  [[ "${PROFILE}" == "1" ]] || return 0
  curl -fsS "${BASE_URL}/stop_request_profile" \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${run_id}\"}" \
    >/dev/null || echo "WARN: failed to stop request profile for ${run_id}" >&2
}

build_benchmark_args() {
  local concurrency="$1"
  local out_dir="$2"
  BENCH_ARGS=(
    "${PYTHON}" -m benchmarks.eval.benchmark_omni_videoamme
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --repo-id "${REPO_ID}"
    --output-dir "${out_dir}"
    --max-samples "${MAX_SAMPLES}"
    --sample-offset "${SAMPLE_OFFSET}"
    --max-concurrency "${concurrency}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --video-fps "${VIDEO_FPS}"
    --video-max-frames "${VIDEO_MAX_FRAMES}"
    --video-max-pixels "${VIDEO_MAX_PIXELS}"
    --enable-audio
    --audio-voice "${AUDIO_VOICE}"
  )
  [[ "${SKIP_WER}" == "1" ]] && BENCH_ARGS+=(--skip-wer)
  [[ "${DISABLE_TQDM}" == "1" ]] && BENCH_ARGS+=(--disable-tqdm)
}

run_one_concurrency() {
  local concurrency="$1"
  local run_id="c${concurrency}_profile"
  [[ "${SKIP_WER}" == "1" ]] && run_id="${run_id}_skipwer"
  local out_dir="${RUN_ROOT}/benchmark_audio_${MAX_SAMPLES}_${run_id}"
  local profile_out
  profile_out="$(pwd)/${RUN_ROOT}/request_profile_${run_id}.json"

  mkdir -p "${out_dir}" "${EVENT_DIR}"

  echo
  echo "=== Video-AMME c=${concurrency}, samples=${MAX_SAMPLES}, output=${out_dir} ==="
  start_request_profile "${run_id}"

  build_benchmark_args "${concurrency}" "${out_dir}"

  set +e
  "${BENCH_ARGS[@]}" 2>&1 | tee "${out_dir}/run.log"
  local bench_status=${PIPESTATUS[0]}
  set -e

  stop_request_profile "${run_id}"
  if [[ "${bench_status}" -ne 0 ]]; then
    echo "ERROR: benchmark failed for c=${concurrency}; log=${out_dir}/run.log" >&2
    exit "${bench_status}"
  fi

  if [[ "${PROFILE}" == "1" ]]; then
    "${PYTHON}" -m sglang_omni.profiler \
      "${EVENT_DIR}" \
      --format json \
      --out "${profile_out}"
  fi
}

run_warmup_one_concurrency() {
  local concurrency="$1"
  local out_dir="${RUN_ROOT}/warmup_audio_${WARMUP_SAMPLES}_c${concurrency}_skipwer"
  mkdir -p "${out_dir}"

  echo
  echo "=== Warmup Video-AMME c=${concurrency}, samples=${WARMUP_SAMPLES}, output=${out_dir} ==="

  local warmup_args=(
    "${PYTHON}" -m benchmarks.eval.benchmark_omni_videoamme
    --model "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --repo-id "${WARMUP_REPO_ID}"
    --output-dir "${out_dir}"
    --max-samples "${WARMUP_SAMPLES}"
    --sample-offset "${WARMUP_SAMPLE_OFFSET:-0}"
    --max-concurrency "${concurrency}"
    --max-tokens "${WARMUP_MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --video-fps "${VIDEO_FPS}"
    --video-max-frames "${VIDEO_MAX_FRAMES}"
    --video-max-pixels "${VIDEO_MAX_PIXELS}"
    --enable-audio
    --audio-voice "${AUDIO_VOICE}"
    --skip-wer
  )
  [[ "${DISABLE_TQDM}" == "1" ]] && warmup_args+=(--disable-tqdm)

  set +e
  "${warmup_args[@]}" 2>&1 | tee "${out_dir}/run.log"
  local warmup_status=${PIPESTATUS[0]}
  set -e

  if [[ "${warmup_status}" -ne 0 ]]; then
    echo "ERROR: warmup failed for c=${concurrency}; log=${out_dir}/run.log" >&2
    exit "${warmup_status}"
  fi
}

result_json_for_c() {
  local concurrency="$1"
  local run_id="c${concurrency}_profile"
  [[ "${SKIP_WER}" == "1" ]] && run_id="${run_id}_skipwer"
  printf '%s\n' "${RUN_ROOT}/benchmark_audio_${MAX_SAMPLES}_${run_id}/videoamme_results.json"
}

profile_json_for_c() {
  local concurrency="$1"
  local run_id="c${concurrency}_profile"
  [[ "${SKIP_WER}" == "1" ]] && run_id="${run_id}_skipwer"
  printf '%s\n' "${RUN_ROOT}/request_profile_${run_id}.json"
}

write_performance_table() {
  local out_tsv="${RUN_ROOT}/performance_table.tsv"
  local out_md="${RUN_ROOT}/performance_table.md"

  printf 'C\tcompleted\tfailed\taccuracy\tqps\tlat_mean_s\tlat_median_s\tlat_p95_s\tlat_p99_s\trtf_mean\toutput_tok_per_req_s\toutput_throughput\tprompt_tokens_total\toutput_tokens_total\tresult_json\n' >"${out_tsv}"

  for concurrency in ${CONCURRENCIES}; do
    local result_json
    result_json="$(result_json_for_c "${concurrency}")"
    if [[ ! -f "${result_json}" ]]; then
      echo "WARN: missing result json for c=${concurrency}: ${result_json}" >&2
      continue
    fi
    "${PYTHON}" - "${concurrency}" "${result_json}" <<'PY' >>"${out_tsv}"
import json
import sys

concurrency, path = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

speed = payload.get("speed") or {}
summary = payload.get("summary") or {}

fields = [
    concurrency,
    speed.get("completed_requests"),
    speed.get("failed_requests"),
    summary.get("accuracy"),
    speed.get("throughput_qps"),
    speed.get("latency_mean_s"),
    speed.get("latency_median_s"),
    speed.get("latency_p95_s"),
    speed.get("latency_p99_s"),
    speed.get("rtf_mean"),
    speed.get("output_tok_per_req_s"),
    speed.get("output_throughput"),
    speed.get("prompt_tokens_total"),
    speed.get("output_tokens_total"),
    path,
]

print("\t".join("" if value is None else str(value) for value in fields))
PY
  done

  {
    echo '| C | completed | failed | accuracy | qps | lat_mean_s | lat_p95_s | rtf_mean |'
    echo '|---:|---:|---:|---:|---:|---:|---:|---:|'
    awk -F '\t' 'NR > 1 {
      printf("| %s | %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6, $8, $10)
    }' "${out_tsv}"
  } >"${out_md}"
}

write_stage_breakdown() {
  local stage_tsv="${RUN_ROOT}/stage_breakdown.tsv"
  local hop_tsv="${RUN_ROOT}/hop_breakdown.tsv"
  local top_avg="${RUN_ROOT}/stage_breakdown_top_by_avg_ms.tsv"
  local top_p95="${RUN_ROOT}/stage_breakdown_top_by_p95_ms.tsv"

  printf 'C\tstage\tinterval\tcount\tavg_ms\tp50_ms\tp95_ms\tmax_ms\ttotal_ms\tprofile_json\n' >"${stage_tsv}"
  printf 'C\tsrc\tdst\tkind\tcount\tavg_ms\tp50_ms\tp95_ms\tmax_ms\ttotal_ms\tprofile_json\n' >"${hop_tsv}"

  for concurrency in ${CONCURRENCIES}; do
    local profile_json
    profile_json="$(profile_json_for_c "${concurrency}")"
    if [[ ! -f "${profile_json}" ]]; then
      echo "WARN: missing profile json for c=${concurrency}: ${profile_json}" >&2
      continue
    fi
    "${PYTHON}" - "${concurrency}" "${profile_json}" stage <<'PY' >>"${stage_tsv}"
import json
import sys

concurrency, path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if mode == "stage":
    rows = payload.get("stage_breakdown") or []
    keys = ("stage", "interval", "count", "avg_ms", "p50_ms", "p95_ms", "max_ms", "total_ms")
else:
    rows = payload.get("hop_breakdown") or []
    keys = ("src", "dst", "kind", "count", "avg_ms", "p50_ms", "p95_ms", "max_ms", "total_ms")

for row in rows:
    fields = [concurrency] + [row.get(key) for key in keys] + [path]
    print("\t".join("" if value is None else str(value) for value in fields))
PY

    "${PYTHON}" - "${concurrency}" "${profile_json}" hop <<'PY' >>"${hop_tsv}"
import json
import sys

concurrency, path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if mode == "stage":
    rows = payload.get("stage_breakdown") or []
    keys = ("stage", "interval", "count", "avg_ms", "p50_ms", "p95_ms", "max_ms", "total_ms")
else:
    rows = payload.get("hop_breakdown") or []
    keys = ("src", "dst", "kind", "count", "avg_ms", "p50_ms", "p95_ms", "max_ms", "total_ms")

for row in rows:
    fields = [concurrency] + [row.get(key) for key in keys] + [path]
    print("\t".join("" if value is None else str(value) for value in fields))
PY
  done

  {
    head -n 1 "${stage_tsv}"
    tail -n +2 "${stage_tsv}" | sort -t "$(printf '\t')" -k5,5nr | head -80
  } >"${top_avg}"

  {
    head -n 1 "${stage_tsv}"
    tail -n +2 "${stage_tsv}" | sort -t "$(printf '\t')" -k7,7nr | head -80
  } >"${top_p95}"
}

print_summary() {
  echo
  echo "=== Performance table ==="
  column -t -s "$(printf '\t')" "${RUN_ROOT}/performance_table.tsv" || cat "${RUN_ROOT}/performance_table.tsv"

  echo
  echo "=== Top stages by avg_ms ==="
  column -t -s "$(printf '\t')" "${RUN_ROOT}/stage_breakdown_top_by_avg_ms.tsv" || cat "${RUN_ROOT}/stage_breakdown_top_by_avg_ms.tsv"

  echo
  echo "=== Output files ==="
  printf '%s\n' \
    "${RUN_ROOT}/performance_table.tsv" \
    "${RUN_ROOT}/performance_table.md" \
    "${RUN_ROOT}/stage_breakdown.tsv" \
    "${RUN_ROOT}/stage_breakdown_top_by_avg_ms.tsv" \
    "${RUN_ROOT}/stage_breakdown_top_by_p95_ms.tsv" \
    "${RUN_ROOT}/hop_breakdown.tsv"
}

main() {
  check_deps
  mkdir -p "${RUN_ROOT}"

  if [[ "${ANALYZE_ONLY}" != "1" ]]; then
    check_server
    check_warmup_overlap_policy
    echo "=== Cache policy ==="
    echo "${SWEEP_CACHE_NOTE}"
    echo "Measured repo=${REPO_ID} sample_offset=${SAMPLE_OFFSET} max_samples=${MAX_SAMPLES}"
    if [[ "${WARMUP_SAMPLES}" != "0" ]]; then
      echo "Warmup repo=${WARMUP_REPO_ID} sample_offset=${WARMUP_SAMPLE_OFFSET:-0} samples=${WARMUP_SAMPLES}"
    fi
    if [[ "${WARMUP_SAMPLES}" != "0" ]]; then
      for concurrency in ${WARMUP_CONCURRENCIES}; do
        run_warmup_one_concurrency "${concurrency}"
      done
    fi
    for concurrency in ${CONCURRENCIES}; do
      run_one_concurrency "${concurrency}"
    done
    ln -sfn "$(basename "${RUN_ROOT}")" results/qwen35_sglang_videoamme_stress_latest
  fi

  write_performance_table
  write_stage_breakdown
  if [[ "${ANALYZER}" == "1" ]]; then
    "${PYTHON}" -m benchmarks.eval.qwen35_omni_videoamme_analyze \
      --run-root "${RUN_ROOT}" || echo "WARN: analysis report generation failed" >&2
  fi
  print_summary
}

main "$@"
