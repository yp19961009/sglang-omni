#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-H20-ecs-gangouyu}"
CONTAINER="${CONTAINER:-b5f665f3d883}"
REMOTE_REPO="${REMOTE_REPO:-/myapp/sglang-omni}"
PORT="${PORT:-8162}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-6115ffd}"

echo "[local] checking report artifacts"
required_files=(
  "$ROOT_DIR/reports/qwen35_c12_final_delivery_note_20260701.md"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_stable_lock_20260630.md"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_delivery_audit_20260630.md"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_perf_report_20260630.md"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_benchmark_manifest_20260630.json"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_benchmark_summary_20260630.csv"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_stage_summary_20260630.csv"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_audio_index_20260630.md"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_audio_outputs_20260630.csv"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_audio_validation_20260630.md"
  "$ROOT_DIR/reports/sglang_omni_qwen35_c12_audio_validation_20260630.csv"
  "$ROOT_DIR/reports/qwen35_c12_stable_best_audio_player_20260630.html"
  "$ROOT_DIR/reports/qwen35_c12_delivery_bundle_20260630.tar.gz"
  "$ROOT_DIR/reports/qwen35_c12_delivery_bundle_20260630.tar.gz.sha256"
  "$ROOT_DIR/reports/delivery_ready_snapshot_20260630.txt"
  "$ROOT_DIR/reports/run_sglang_qwen35_stable_server.sh"
  "$ROOT_DIR/reports/run_sglang_qwen35_stable_c12_benchmark.sh"
)
for file in "${required_files[@]}"; do
  test -s "$file"
  echo "  present: ${file#$ROOT_DIR/}"
done

python3 -m json.tool "$ROOT_DIR/reports/sglang_omni_qwen35_c12_benchmark_manifest_20260630.json" >/tmp/qwen35_manifest.pretty

(
  cd "$ROOT_DIR/reports"
  shasum -a 256 -c qwen35_c12_delivery_bundle_20260630.tar.gz.sha256
  tar -tzf qwen35_c12_delivery_bundle_20260630.tar.gz >/tmp/qwen35_delivery_bundle.files
)

python3 - "$ROOT_DIR/reports/qwen35_c12_delivery_bundle_20260630.tar.gz" <<'PY'
import sys
import tarfile
from pathlib import Path

bundle_path = Path(sys.argv[1])
files = Path("/tmp/qwen35_delivery_bundle.files").read_text().splitlines()
with tarfile.open(bundle_path) as bundle:
    tar_names = bundle.getnames()
appledouble = [
    name for name in tar_names
    if name.startswith("._") or "/._" in name
]
if appledouble:
    raise SystemExit(f"bundle contains AppleDouble metadata entries: {appledouble[:8]}")
required = {
    "README_qwen35_c12_delivery_20260630.md",
    "qwen35_c12_final_delivery_note_20260701.md",
    "check_sglang_qwen35_delivery_ready.sh",
    "sglang_omni_qwen35_c12_perf_report_20260630.md",
    "sglang_omni_qwen35_c12_benchmark_manifest_20260630.json",
    "sglang_omni_qwen35_c12_audio_index_20260630.md",
    "sglang_omni_qwen35_c12_audio_outputs_20260630.csv",
    "sglang_omni_qwen35_c12_audio_validation_20260630.md",
    "sglang_omni_qwen35_c12_audio_validation_20260630.csv",
    "qwen35_c12_stable_best_audio_player_20260630.html",
}
missing = sorted(required.difference(files))
if missing:
    raise SystemExit(f"bundle missing files: {missing}")
wav_count = sum(
    name.startswith("audio/stable_best_sil672/") and name.endswith(".wav")
    for name in files
)
result_count = sum(
    name.startswith("audio/stable_best_sil672/") and name.endswith("/result.json")
    for name in files
)
if wav_count != 12 or result_count != 12:
    raise SystemExit(f"bundle audio mismatch: wav={wav_count} result={result_count}")
print(f"  bundle ok: files={len(files)} wav={wav_count} result={result_count}")
PY

python3 - "$ROOT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rel in [
    "reports/sglang_omni_qwen35_c12_benchmark_summary_20260630.csv",
    "reports/sglang_omni_qwen35_c12_stage_summary_20260630.csv",
    "reports/sglang_omni_qwen35_c12_audio_outputs_20260630.csv",
    "reports/sglang_omni_qwen35_c12_audio_validation_20260630.csv",
]:
    path = root / rel
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise SystemExit(f"{rel} has inconsistent column counts: {sorted(widths)}")
    print(f"  csv ok: {rel} rows={len(rows) - 1} columns={next(iter(widths))}")

audio_dir = root / "reports/audio/stable_best_sil672"
wav_count = len(list(audio_dir.glob("sample_*/*.wav")))
result_count = len(list(audio_dir.glob("sample_*/result.json")))
if wav_count != 12 or result_count != 12:
    raise SystemExit(f"local audio count mismatch: wav={wav_count} result={result_count}")
print(f"  local audio ok: wav={wav_count} result={result_count}")

html = (root / "reports/qwen35_c12_stable_best_audio_player_20260630.html").read_text()
audio_tags = html.count("<audio ")
if audio_tags != 12:
    raise SystemExit(f"audio player tag mismatch: audio_tags={audio_tags}")
print(f"  audio player ok: audio_tags={audio_tags}")

validation_path = root / "reports/sglang_omni_qwen35_c12_audio_validation_20260630.csv"
with validation_path.open(newline="") as f:
    validation_rows = list(csv.DictReader(f))
if len(validation_rows) != 12:
    raise SystemExit(f"audio validation row mismatch: rows={len(validation_rows)}")
for row in validation_rows:
    if row.get("valid_wav") != "True":
        raise SystemExit(f"invalid wav in audio validation: {row}")
    if row.get("channels") != "1" or row.get("sample_rate") != "24000" or row.get("sample_width_bytes") != "2":
        raise SystemExit(f"unexpected wav format in audio validation: {row}")
    if abs(float(row.get("duration_delta_ms", "999999"))) > 1.0:
        raise SystemExit(f"audio duration mismatch in audio validation: {row}")
print("  audio validation ok: rows=12 valid=12 format=mono/24k/16bit max_delta_ms<=1")
PY

bash -n "$ROOT_DIR/reports/run_sglang_qwen35_stable_server.sh"
bash -n "$ROOT_DIR/reports/run_sglang_qwen35_stable_c12_benchmark.sh"
test -x "$ROOT_DIR/reports/run_sglang_qwen35_stable_server.sh"
test -x "$ROOT_DIR/reports/run_sglang_qwen35_stable_c12_benchmark.sh"
echo "[local] scripts ok"

echo "[remote] checking repo, service, and benchmark evidence"
ssh "$REMOTE_HOST" "docker exec -i $CONTAINER bash" <<SH
set -euo pipefail
cd "$REMOTE_REPO"

EXPECTED_COMMIT="$EXPECTED_COMMIT"
PORT="$PORT"

actual_commit="\$(git rev-parse --short HEAD)"
if [ "\$actual_commit" != "\$EXPECTED_COMMIT" ]; then
  echo "Unexpected remote commit: \$actual_commit != \$EXPECTED_COMMIT" >&2
  exit 1
fi

if [ -n "\$(git status --short)" ]; then
  echo "Remote repo is dirty:" >&2
  git status --short >&2
  exit 1
fi
echo "  git ok: \$(git log -1 --oneline)"

service_line="\$(ps -eo pid,etimes,args | grep 'sglang_omni.cli serve' | grep -- "--port \$PORT" | grep -v grep || true)"
if [ -z "\$service_line" ]; then
  echo "No stable service on port \$PORT" >&2
  exit 1
fi
echo "  service ok: \$service_line"

client_line="\$(ps -eo pid,etimes,args | grep 'qwen35_omni_sglang_rtc_concurrency' | grep -v grep || true)"
if [ -n "\$client_line" ]; then
  echo "Unexpected benchmark client still running:" >&2
  echo "\$client_line" >&2
  exit 1
fi
echo "  no benchmark client running"

curl -fsS "http://127.0.0.1:\$PORT/v1/models" >/tmp/qwen35_delivery_models.json
python - <<'PY'
import json
models = json.load(open("/tmp/qwen35_delivery_models.json"))
ids = [item.get("id") for item in models.get("data", [])]
assert "qwen3_5-omni" in ids, ids
print("  health ok: qwen3_5-omni")
PY

python - <<'PY'
import glob
import json
import os

checks = [
    (
        "stable_best",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_relay1024_cvd345_8162_20260630_103902/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_104103",
        12,
        0,
    ),
    (
        "post_commit_sil672",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_142334",
        12,
        0,
    ),
    (
        "warmed_sil672_same_offset",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil672_stagger0_temp1_barrier_validation_143808",
        12,
        0,
    ),
    (
        "sil680_independent_failed",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_142141/client_c12_realtime_audio_vllmstyle_sil680_stagger0_temp1_barrier_validation_144121",
        0,
        12,
    ),
    (
        "profile_run",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_20260630_122447/client_c12_realtime_audio_vllmstyle_sil700_stagger0_temp1_barrier_profile_stable_c12_c2w4_122641",
        12,
        0,
    ),
    (
        "post_fix_sil712",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_124917/client_c12_realtime_audio_vllmstyle_sil712_stagger0_temp1_barrier_validation_125117",
        12,
        0,
    ),
    (
        "post_fix_sil724",
        "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_124917/client_c12_realtime_audio_vllmstyle_sil724_stagger0_temp1_barrier_validation_130217",
        12,
        0,
    ),
]

for name, out_dir, completed, failed in checks:
    metrics_path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise SystemExit(f"{name}: missing metrics.json at {metrics_path}")
    metrics = json.load(open(metrics_path))
    if metrics.get("completed") != completed or metrics.get("failed") != failed:
        raise SystemExit(
            f"{name}: bad completion metrics completed={metrics.get('completed')} failed={metrics.get('failed')}"
        )
    wav_count = len(glob.glob(os.path.join(out_dir, "sample_*", "*.wav")))
    result_count = len(glob.glob(os.path.join(out_dir, "sample_*", "result.json")))
    if wav_count != completed or result_count != completed:
        raise SystemExit(f"{name}: wav/result count mismatch wav={wav_count} result={result_count}")
    print(
        f"  evidence ok: {name} completed={completed} actual={metrics.get('actual_elapsed_s'):.3f} "
        f"wav={wav_count} result={result_count}"
    )

profile_json = (
    "results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimfix_cache2048_64g_run12_c2w4_profile_relay1024_cvd345_8162_20260630_122447/"
    "client_c12_realtime_audio_vllmstyle_sil700_stagger0_temp1_barrier_profile_stable_c12_c2w4_122641/"
    "request_profile_stable_c12_c2w4_122641.json"
)
if not os.path.exists(profile_json):
    raise SystemExit(f"missing profile json: {profile_json}")
json.load(open(profile_json))
print("  profile json ok")
PY

ACTIVE_RUN="results/sg_realtime_c12_decodebatch8_ready_subset1_mem080_item256m_omitcached_trimpartial_cache2048_64g_run12_c2w4_relay1024_cvd345_8162_20260630_145321"
grep -q '^commit=6115ffd$' "\$ACTIVE_RUN/config.env"
echo "  active config ok: commit=6115ffd"
ACTIVE_RUN="\$ACTIVE_RUN" python - <<'PY'
import os
from pathlib import Path

log_text = Path(os.environ["ACTIVE_RUN"], "server.log").read_text(errors="ignore")
checks = {
    "HTTP 500": 'HTTP/1.1" 500',
    "OOM": "out of memory",
    "feature/token mismatch": "feature/token mismatch",
    "cache miss": "Visual item payload was omitted but encoder item cache missed",
}
for label, needle in checks.items():
    count = log_text.lower().count(needle.lower())
    print(f"  active log {label} count: {count}")
    if count:
        raise SystemExit(f"active log has {label}: {count}")
PY
SH

echo "delivery_ready=1"
