#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a strict per-concurrency restart sweep for Qwen3.5-Omni Video-AMME.

Each concurrency point starts a fresh SGLang server, waits for readiness, runs
non-overlapping warmup requests, runs the measured benchmark for that single C,
then shuts the server down. This avoids C=2/4/8/16 inheriting sample-specific
server caches from earlier concurrency points.

Default strict ci-50 split:
  WARMUP_SAMPLES=1, WARMUP_SAMPLE_OFFSET=0, SAMPLE_OFFSET=1, MAX_SAMPLES=49

Usage:
  benchmarks/eval/qwen35_omni_videoamme_restart_sweep.sh

Useful overrides:
  CONCURRENCIES="1 4 8" benchmarks/eval/qwen35_omni_videoamme_restart_sweep.sh
  WARMUP_SAMPLES=8 SAMPLE_OFFSET=8 MAX_SAMPLES=42 benchmarks/eval/qwen35_omni_videoamme_restart_sweep.sh
  WARMUP_REPO_ID=/path/to/holdout_videoamme SAMPLE_OFFSET=0 MAX_SAMPLES=50 benchmarks/eval/qwen35_omni_videoamme_restart_sweep.sh
  RUN_ROOT=results/my_restart_sweep benchmarks/eval/qwen35_omni_videoamme_restart_sweep.sh

Outputs:
  ${RUN_ROOT}/performance_table.tsv
  ${RUN_ROOT}/performance_table.md
  ${RUN_ROOT}/stage_breakdown.tsv
  ${RUN_ROOT}/stage_breakdown_top_by_avg_ms.tsv
  ${RUN_ROOT}/stage_breakdown_top_by_p95_ms.tsv
  ${RUN_ROOT}/hop_breakdown.tsv
  ${RUN_ROOT}/analysis_report.md
  ${RUN_ROOT}/analysis_summary.json
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
  if [[ -d /myapp/sglang-omni && -f /myapp/sglang-omni/benchmarks/eval/qwen35_omni_videoamme_stress.sh ]]; then
    printf '%s\n' /myapp/sglang-omni
    return
  fi
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

ROOT="$(find_repo_root)"
cd "${ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8161}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-qwen3_5-omni}"
REPO_ID="${REPO_ID:-zhaochenyang20/Video_AMME_ci}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16}"
TOTAL_CI_SAMPLES="${TOTAL_CI_SAMPLES:-50}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
WARMUP_SAMPLE_OFFSET="${WARMUP_SAMPLE_OFFSET:-0}"
WARMUP_REPO_ID="${WARMUP_REPO_ID:-$REPO_ID}"
WARMUP_CONCURRENCIES="${WARMUP_CONCURRENCIES:-CURRENT}"
WARMUP_ALLOW_MEASURED_OVERLAP="${WARMUP_ALLOW_MEASURED_OVERLAP:-0}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-$((WARMUP_SAMPLE_OFFSET + WARMUP_SAMPLES))}"
MAX_SAMPLES="${MAX_SAMPLES:-$((TOTAL_CI_SAMPLES - SAMPLE_OFFSET))}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-401408}"
AUDIO_VOICE="${AUDIO_VOICE:-m02}"
SKIP_WER="${SKIP_WER:-1}"
PROFILE="${PROFILE:-1}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
RUN_ROOT="${RUN_ROOT:-results/qwen35_sglang_videoamme_restart_sweep_$(date +%Y%m%d_%H%M%S)}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${RUN_ROOT}/server_logs}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-180}"
SERVER_WAIT_INTERVAL_S="${SERVER_WAIT_INTERVAL_S:-10}"
SERVER_SHUTDOWN_GRACE_S="${SERVER_SHUTDOWN_GRACE_S:-30}"
GPU_COOLDOWN_S="${GPU_COOLDOWN_S:-10}"
KILL_STALE_PORT_LISTENER="${KILL_STALE_PORT_LISTENER:-1}"
START_SERVER_CMD="${START_SERVER_CMD:-bash examples/launch_qwen35_omni_speech_server_container.sh}"
SWEEP_CACHE_NOTE="${SWEEP_CACHE_NOTE:-per-concurrency restart sweep; server-side video preprocessing/prefix caches are reset before each measured C; warmup uses a non-overlapping sample slice}"

# Performance recipe defaults. Caller-provided env wins.
export NO_CODE2WAV_TORCH_COMPILE="${NO_CODE2WAV_TORCH_COMPILE:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-0}"
export SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES="${SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES:-17179869184}"
export SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES="${SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES:-64}"
export EXTRA_ARGS="${EXTRA_ARGS:---thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8}"

export PYTHON
export HOST PORT BASE_URL MODEL REPO_ID
export MAX_SAMPLES SAMPLE_OFFSET MAX_TOKENS TEMPERATURE
export VIDEO_FPS VIDEO_MAX_FRAMES VIDEO_MAX_PIXELS AUDIO_VOICE
export SKIP_WER PROFILE DISABLE_TQDM
export WARMUP_SAMPLES WARMUP_SAMPLE_OFFSET WARMUP_REPO_ID WARMUP_ALLOW_MEASURED_OVERLAP
export HF_HOME="${HF_HOME:-/myapp/data/videoamme}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SWEEP_CACHE_NOTE

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

port_listener_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true
    return
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${PORT}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u || true
  fi
}

terminate_pids() {
  local pids=("$@")
  [[ "${#pids[@]}" -gt 0 ]] || return 0

  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 3

  local alive=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      alive+=("$pid")
    fi
  done

  if [[ "${#alive[@]}" -gt 0 ]]; then
    kill -KILL "${alive[@]}" 2>/dev/null || true
    sleep 1
  fi
}

kill_stale_port_listener() {
  [[ "${KILL_STALE_PORT_LISTENER}" == "1" ]] || return 0

  mapfile -t pids < <(port_listener_pids)
  [[ "${#pids[@]}" -gt 0 ]] || return 0

  echo "[restart-sweep] killing stale listener(s) on ${BASE_URL}: ${pids[*]}"
  terminate_pids "${pids[@]}"
}

wait_for_server_ready() {
  local log_file="$1"
  local attempt
  for ((attempt = 1; attempt <= SERVER_WAIT_ATTEMPTS; attempt++)); do
    if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
      echo "[restart-sweep] server ready: ${BASE_URL}"
      return 0
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "[restart-sweep] server exited before readiness; last log lines:" >&2
      tail -80 "${log_file}" >&2 || true
      return 1
    fi
    echo "[restart-sweep] waiting for server at ${BASE_URL} (${attempt}/${SERVER_WAIT_ATTEMPTS})"
    sleep "${SERVER_WAIT_INTERVAL_S}"
  done

  echo "[restart-sweep] server did not become ready; last log lines:" >&2
  tail -80 "${log_file}" >&2 || true
  return 1
}

start_server_for_c() {
  local concurrency="$1"
  local log_file="${SERVER_LOG_DIR}/server_c${concurrency}.log"

  mkdir -p "${SERVER_LOG_DIR}"
  kill_stale_port_listener

  echo
  echo "=== Starting fresh SGLang server for C=${concurrency} ==="
  echo "[restart-sweep] log=${log_file}"
  setsid bash -lc "${START_SERVER_CMD}" >"${log_file}" 2>&1 &
  server_pid=$!
  wait_for_server_ready "${log_file}"
}

stop_server_for_c() {
  local concurrency="$1"
  [[ -n "${server_pid:-}" ]] || return 0

  echo "=== Stopping SGLang server for C=${concurrency} pid=${server_pid} ==="
  if kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || kill -TERM "${server_pid}" 2>/dev/null || true

    local waited=0
    while kill -0 "${server_pid}" >/dev/null 2>&1 && (( waited < SERVER_SHUTDOWN_GRACE_S )); do
      sleep 1
      waited=$((waited + 1))
    done

    if kill -0 "${server_pid}" >/dev/null 2>&1; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || kill -KILL "${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi

  server_pid=""
  if [[ "${GPU_COOLDOWN_S}" != "0" ]]; then
    sleep "${GPU_COOLDOWN_S}"
  fi
}

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    stop_server_for_c "${current_concurrency:-unknown}" || true
  fi
}
trap cleanup EXIT INT TERM

run_measured_c() {
  local concurrency="$1"
  local event_dir
  local warmup_concurrencies
  if [[ "${WARMUP_CONCURRENCIES}" == "CURRENT" ]]; then
    warmup_concurrencies="${concurrency}"
  else
    warmup_concurrencies="${WARMUP_CONCURRENCIES}"
  fi
  event_dir="$(pwd)/${RUN_ROOT}/events_c${concurrency}"

  echo
  echo "=== Running strict measured C=${concurrency} ==="
  echo "[restart-sweep] warmup_concurrencies=${warmup_concurrencies}"
  echo "[restart-sweep] event_dir=${event_dir}"
  RUN_ROOT="${RUN_ROOT}" \
    EVENT_DIR="${event_dir}" \
    CONCURRENCIES="${concurrency}" \
    WARMUP_CONCURRENCIES="${warmup_concurrencies}" \
    ANALYZER=0 \
    benchmarks/eval/qwen35_omni_videoamme_stress.sh
}

write_run_contract() {
  mkdir -p "${RUN_ROOT}"
  {
    echo "run_root=${RUN_ROOT}"
    echo "cache_policy=${SWEEP_CACHE_NOTE}"
    echo "concurrencies=${CONCURRENCIES}"
    echo "repo_id=${REPO_ID}"
    echo "measured_sample_offset=${SAMPLE_OFFSET}"
    echo "measured_max_samples=${MAX_SAMPLES}"
    echo "warmup_repo_id=${WARMUP_REPO_ID}"
    echo "warmup_sample_offset=${WARMUP_SAMPLE_OFFSET}"
    echo "warmup_samples=${WARMUP_SAMPLES}"
    echo "warmup_concurrencies=${WARMUP_CONCURRENCIES}"
    echo "event_dir_policy=one event directory per measured concurrency"
    echo "server_cmd=${START_SERVER_CMD}"
    echo "no_code2wav_torch_compile=${NO_CODE2WAV_TORCH_COMPILE}"
    echo "torchdynamo_disable=${TORCHDYNAMO_DISABLE}"
    echo "video_preprocess_cache_max_bytes=${SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES}"
    echo "video_preprocess_cache_max_entries=${SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES}"
    echo "extra_args=${EXTRA_ARGS}"
  } >"${RUN_ROOT}/restart_sweep_contract.env"
}

main() {
  require_cmd "${PYTHON}"
  require_cmd curl
  require_cmd setsid

  if (( MAX_SAMPLES <= 0 )); then
    echo "ERROR: MAX_SAMPLES must be positive; got ${MAX_SAMPLES}" >&2
    exit 1
  fi

  write_run_contract

  echo "=== Restart Sweep Contract ==="
  cat "${RUN_ROOT}/restart_sweep_contract.env"

  local concurrency
  for concurrency in ${CONCURRENCIES}; do
    current_concurrency="${concurrency}"
    start_server_for_c "${concurrency}"
    run_measured_c "${concurrency}"
    stop_server_for_c "${concurrency}"
  done

  echo
  echo "=== Aggregating all concurrency points ==="
  RUN_ROOT="${RUN_ROOT}" \
    CONCURRENCIES="${CONCURRENCIES}" \
    ANALYZE_ONLY=1 \
    ANALYZER=1 \
    benchmarks/eval/qwen35_omni_videoamme_stress.sh

  ln -sfn "$(basename "${RUN_ROOT}")" results/qwen35_sglang_videoamme_restart_sweep_latest
  echo
  echo "=== Restart sweep complete ==="
  echo "${RUN_ROOT}"
}

main "$@"
