#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

HOST_REPO="${HOST_REPO:-$(pwd)}"
AUDIT_DIR="${AUDIT_DIR:-results/qwen35_report_audit_20260619}"
TARBALL="${TARBALL:-${AUDIT_DIR}/qwen35_omni_share_bundle_20260621.tar.gz}"
CHECKSUM="${CHECKSUM:-${TARBALL}.sha256}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_quickcheck}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_quickcheck}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_quickcheck}"
ALLOW_NON_TMP_EXTRACT="${ALLOW_NON_TMP_EXTRACT:-0}"
ARC_PREFIX="${ARC_PREFIX:-qwen35_omni_share_bundle_20260621}"

die() {
  printf 'quickcheck error: %s\n' "$*" >&2
  exit 1
}

prepare_tmp_dir() {
  local target="$1"
  [[ -n "$target" && "$target" != "/" ]] || die "unsafe temp directory: ${target:-<empty>}"
  if [[ "$ALLOW_NON_TMP_EXTRACT" != "1" && "$target" != /tmp/* ]]; then
    die "refusing to remove non-/tmp directory '$target'; set ALLOW_NON_TMP_EXTRACT=1 to override"
  fi
  rm -rf "$target"
  mkdir -p "$target"
}

cd "$HOST_REPO"
if [[ "$AUDIT_DIR" = /* ]]; then
  AUDIT_DIR_ABS="$AUDIT_DIR"
else
  AUDIT_DIR_ABS="${HOST_REPO}/${AUDIT_DIR}"
fi
mkdir -p "$AUDIT_DIR_ABS"

[[ -f "$TARBALL" ]] || die "missing tarball: $TARBALL"
[[ -f "$CHECKSUM" ]] || die "missing checksum: $CHECKSUM"
[[ -f benchmarks/eval/validate_qwen35_omni_share_package.py ]] || die "run from repository root or set HOST_REPO"

printf '[1/6] checksum\n'
sha256sum -c "$CHECKSUM"

printf '[2/6] tarball-mode validation\n'
python3 -m benchmarks.eval.validate_qwen35_omni_share_package \
  --root "$HOST_REPO" \
  --strict \
  --json-output "${AUDIT_DIR_ABS}/share_package_validation.json"

printf '[3/6] receiver-smoke validation\n'
prepare_tmp_dir "$SMOKE_DIR"
python3 -m benchmarks.eval.validate_qwen35_omni_share_package \
  --root "$HOST_REPO" \
  --strict \
  --receiver-smoke-dir "$SMOKE_DIR" \
  --json-output "${AUDIT_DIR_ABS}/share_package_receiver_smoke_validation.json"

printf '[4/6] extracted-only validation\n'
prepare_tmp_dir "$EXTRACT_DIR"
tar -xzf "$TARBALL" -C "$EXTRACT_DIR"
EXTRACTED_BUNDLE="${EXTRACT_DIR}/${ARC_PREFIX}"
[[ -d "$EXTRACTED_BUNDLE" ]] || die "missing extracted bundle: $EXTRACTED_BUNDLE"
(
  cd "$EXTRACTED_BUNDLE"
  python3 benchmarks/eval/validate_qwen35_omni_share_package.py \
    --root "$PWD" \
    --extracted-only \
    --strict \
    --json-output "${AUDIT_DIR_ABS}/share_package_validation_extracted.json"
)

printf '[5/6] external standalone validation\n'
python3 -m benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation \
  --root "$HOST_REPO" \
  --strict \
  --work-dir "$STANDALONE_DIR" \
  --json-output "${AUDIT_DIR_ABS}/share_package_external_standalone_validation.json"

printf '[6/6] summary\n'
python3 - "$AUDIT_DIR_ABS" <<'PY'
import json
import sys
from pathlib import Path

audit_dir = Path(sys.argv[1])
paths = {
    "tarball": audit_dir / "share_package_validation.json",
    "receiver_smoke": audit_dir / "share_package_receiver_smoke_validation.json",
    "extracted": audit_dir / "share_package_validation_extracted.json",
    "standalone": audit_dir / "share_package_external_standalone_validation.json",
}
for label, path in paths.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    print(
        f"{label}: ready={summary.get('ready')} "
        f"checks={summary.get('checks_passed')}/{summary.get('checks_total')} "
        f"required_failures={summary.get('required_failures')}"
    )
PY

printf 'quickcheck complete: open benchmarks/reports/qwen35_omni_start_here_zh_20260621.md, benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md, or benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md next\n'
