#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

"$PYTHON" -m pytest tests/unit_test/qwen3_5_omni -q
"$PYTHON" -m pytest tests/unit_test/qwen3_omni/test_cli.py -q
"$PYTHON" -m pytest tests/unit_test/serve/test_openai_api.py -q

PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/codex_pycache_q35}" \
"$PYTHON" -m py_compile \
  benchmarks/eval/benchmark_omni_videomme.py \
  benchmarks/tasks/video_understanding.py \
  examples/run_qwen3_5_omni_server.py \
  examples/run_qwen3_5_omni_speech_server.py \
  scripts/qwen35_omni_alignment.py \
  scripts/qwen35_omni_preflight.py \
  sglang_omni/config/runtime.py \
  sglang_omni/config/schema.py \
  sglang_omni/client/client.py \
  sglang_omni/client/types.py \
  sglang_omni/cli/serve.py \
  sglang_omni/serve/launcher.py \
  sglang_omni/serve/openai_api.py \
  sglang_omni/serve/protocol.py \
  sglang_omni/model_runner/model_worker.py \
  sglang_omni/model_runner/sglang_model_runner.py \
  sglang_omni/utils/hf.py \
  sglang_omni/models/qwen3_omni/request_builders.py \
  sglang_omni/models/qwen3_omni/talker_model_runner.py \
  sglang_omni/models/qwen3_omni/components/preprocessor.py \
  $(find sglang_omni/models/qwen3_5_omni tests/unit_test/qwen3_5_omni -name '*.py' -print)

"$PYTHON" - <<'PY'
from pathlib import Path

paths = list(Path("sglang_omni/models/qwen3_5_omni").rglob("*.py"))
paths += list(Path("tests/unit_test/qwen3_5_omni").glob("*.py"))
paths += [
    Path("examples/run_qwen3_5_omni_server.py"),
    Path("examples/run_qwen3_5_omni_speech_server.py"),
    Path("examples/qwen3_5_omni_README.md"),
    Path("examples/configs/qwen3_5_omni_colocated_h20.yaml"),
    Path("scripts/validate_qwen35_omni.sh"),
    Path("scripts/qwen35_omni_alignment.py"),
    Path("scripts/qwen35_omni_preflight.py"),
    Path("sglang_omni/config/runtime.py"),
    Path("sglang_omni/config/schema.py"),
    Path("sglang_omni/client/client.py"),
    Path("sglang_omni/client/types.py"),
    Path("sglang_omni/cli/serve.py"),
    Path("sglang_omni/serve/launcher.py"),
    Path("sglang_omni/serve/openai_api.py"),
    Path("sglang_omni/serve/protocol.py"),
    Path("tests/unit_test/serve/test_openai_api.py"),
    Path("sglang_omni/model_runner/model_worker.py"),
    Path("sglang_omni/model_runner/sglang_model_runner.py"),
    Path("sglang_omni/utils/hf.py"),
    Path("sglang_omni/models/qwen3_omni/request_builders.py"),
    Path("sglang_omni/models/qwen3_omni/talker_model_runner.py"),
    Path("sglang_omni/models/qwen3_omni/components/preprocessor.py"),
]

too_long = []
trailing_ws = []
for path in paths:
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if len(line) > 100:
            too_long.append(f"{path}:{lineno}:{len(line)}")
        if line.rstrip(" \t") != line:
            trailing_ws.append(f"{path}:{lineno}")

if too_long:
    raise SystemExit("\n".join(too_long))
if trailing_ws:
    raise SystemExit("trailing whitespace:\n" + "\n".join(trailing_ws))
PY

echo "Qwen3.5-Omni validation passed."
