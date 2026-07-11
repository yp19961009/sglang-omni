# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[3] / "benchmarks" / "eval" / "qwen35_omni_talker_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen35_talker_contract", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _write_fake_model(path: Path, weight: bytes) -> None:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"qwen3_omni_next"}\n')
    (path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"model-00001-of-00001.safetensors"}}\n'
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(weight)
    codec = path / "qwen3_5_omni_codec_decode_online_0306"
    codec.mkdir()
    (codec / "config.yaml").write_text("sample_rate: 24000\n")
    (codec / "model_weights.pt").write_bytes(b"codec-weights")


def test_fingerprint_changes_when_weights_change(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_fake_model(model, b"first-weight")
    first = contract.build_model_fingerprint(model, engine="sglang")

    (model / "model-00001-of-00001.safetensors").write_bytes(b"other-weight")
    second = contract.build_model_fingerprint(model, engine="sglang")

    assert first["model"]["model_id"] != second["model"]["model_id"]
    assert first["model"]["weight_hash_mode"] == "sampled-first-last-1mib"


def test_comparison_contract_rejects_exact_alignment_for_different_models(
    tmp_path: Path,
) -> None:
    first_model = tmp_path / "first"
    second_model = tmp_path / "second"
    _write_fake_model(first_model, b"first")
    _write_fake_model(second_model, b"second")
    candidate = contract.build_model_fingerprint(first_model, engine="sglang")
    reference = contract.build_model_fingerprint(second_model, engine="vllm")

    comparison = contract.build_comparison_contract(candidate, reference)

    assert comparison["comparison_mode"] == "different_model_reference"
    assert "exact thinker token equality" in comparison["forbidden_assertions"]


def test_comparison_contract_allows_same_model_regression(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_fake_model(model, b"same")
    fingerprint = contract.build_model_fingerprint(model, engine="sglang")

    comparison = contract.build_comparison_contract(fingerprint, fingerprint)

    assert comparison["comparison_mode"] == "same_model_regression"
    assert comparison["forbidden_assertions"] == []


def test_comparison_contract_detects_same_weights_with_config_drift(
    tmp_path: Path,
) -> None:
    first_model = tmp_path / "first"
    second_model = tmp_path / "second"
    _write_fake_model(first_model, b"same")
    _write_fake_model(second_model, b"same")
    (second_model / "voice_map.json").write_text('{"Ethan":"m02"}\n')
    candidate = contract.build_model_fingerprint(first_model, engine="sglang")
    reference = contract.build_model_fingerprint(second_model, engine="vllm")

    comparison = contract.build_comparison_contract(candidate, reference)

    assert comparison["comparison_mode"] == "same_weights_different_runtime_config"
    assert comparison["same_weight_set"] is True
    assert comparison["identity_differences"][0]["path"] == "voice_map.json"
    assert (
        "unqualified exact talker codec or PCM equality"
        in comparison["forbidden_assertions"]
    )


def test_replay_contract_requires_four_thinker_tokens(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_fake_model(model, b"same")
    fingerprint = contract.build_model_fingerprint(model, engine="sglang")

    with pytest.raises(ValueError, match="shorter"):
        contract.build_replay_contract(
            fingerprint,
            [1, 2, 3],
            voice="Ethan",
            language="auto",
            style=None,
            partial_start_min_tokens=4,
            seed=1234,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.05,
            max_new_tokens=64,
        )

    replay = contract.build_replay_contract(
        fingerprint,
        [1, 2, 3, 4],
        voice="Ethan",
        language="auto",
        style=None,
        partial_start_min_tokens=4,
        seed=1234,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        max_new_tokens=64,
    )
    assert replay["kind"] == "fixed_thinker_token_replay"
    assert replay["thinker_token_ids"] == [1, 2, 3, 4]


def test_extract_thinker_tokens_orders_profiler_events(tmp_path: Path) -> None:
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    events = [
        {
            "request_id": "req-1",
            "event_name": "thinker_token_emit",
            "timestamp_ns": 20,
            "metadata": {"token_id": 102, "token_index": 1},
        },
        {
            "request_id": "req-1",
            "event_name": "thinker_token_emit",
            "timestamp_ns": 10,
            "metadata": {"token_id": 101, "token_index": 0},
        },
    ]
    (event_dir / "events_thinker_1.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    extracted = contract.extract_thinker_tokens(event_dir)

    assert extracted["request_id"] == "req-1"
    assert extracted["thinker_token_ids"] == [101, 102]
