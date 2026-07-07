# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from pathlib import Path

from benchmarks.eval.qwen35_omni_single_s2t import (
    VideoAMMESample,
    _sglang_server_command,
    _vllm_server_command,
    build_alignment_report,
    build_parser,
    build_cases,
    client_stdout_payload,
    parse_prediction,
)


def _sample(sample_id: str, *, video_id: str | None = None) -> VideoAMMESample:
    return VideoAMMESample(
        sample_id=sample_id,
        question_id=sample_id,
        video_id=video_id or sample_id.split("-")[0],
        video_path=f"/myapp/data/videoamme/{sample_id}.mp4",
        audio_path=f"/myapp/data/videoamme/{sample_id}.wav",
        question=f"Question {sample_id}?",
        options=["red car", "blue cup", "yellow bike", "quiet room"],
        expected="A",
    )


def test_build_cases_smoke_keeps_required_cache_probe_order():
    samples = [_sample("001-1"), _sample("001-2", video_id="001"), _sample("002-1")]

    cases = build_cases(
        samples,
        suite="smoke",
        sample_ids=None,
        max_samples=None,
        include_cache_probes=True,
    )

    assert [case.case_id for case in cases] == [
        "warmup_001_1",
        "exact_repeat_001_1",
        "same_video_001_2",
        "different_video_002_1",
        "eval_001_001-1",
        "eval_002_001-2",
        "eval_003_002-1",
    ]
    assert [case.probe_kind for case in cases[:4]] == [
        "warmup",
        "exact_repeat",
        "same_video_different_question",
        "different_video",
    ]


def test_build_cases_short_defaults_to_ten_eval_samples_with_cache_probes():
    samples = [_sample("001-1"), _sample("001-2", video_id="001"), _sample("002-1")]
    samples.extend(_sample(f"{idx:03d}-1") for idx in range(3, 13))

    cases = build_cases(
        samples,
        suite="short",
        sample_ids=None,
        max_samples=None,
        include_cache_probes=True,
    )

    assert len(cases) == 14
    assert cases[0].case_id == "warmup_001_1"
    assert [case.case_id for case in cases[4:7]] == [
        "eval_001_001-1",
        "eval_002_001-2",
        "eval_003_002-1",
    ]
    assert cases[-1].case_id == "eval_010_009-1"


def test_parse_prediction_prefers_answer_tag_then_option_text_fallback():
    options = ["red car", "blue cup", "yellow bike", "quiet room"]

    assert parse_prediction("Answer: C", options) == ("C", False)
    assert parse_prediction("The final choice is the yellow bike.", options) == (
        "C",
        True,
    )


def test_client_stdout_payload_optionally_includes_raw_responses():
    result = {
        "summary": {"total_cases": 1},
        "records": [
            {
                "case_id": "eval_001_002-2",
                "sample_id": "002-2",
                "expected": "D",
                "predicted": "A",
                "raw_response": "Answer: A",
            }
        ],
    }

    assert client_stdout_payload(result, print_raw_response=False) == {
        "total_cases": 1
    }
    assert client_stdout_payload(result, print_raw_response=True) == {
        "total_cases": 1,
        "raw_responses": [
            {
                "case_id": "eval_001_002-2",
                "sample_id": "002-2",
                "expected": "D",
                "predicted": "A",
                "raw_response": "Answer: A",
            }
        ],
    }


def test_client_parser_prints_raw_response_by_default():
    parser = build_parser()

    default_args = parser.parse_args(
        [
            "client",
            "--engine",
            "sglang",
            "--base-url",
            "http://127.0.0.1:8011",
            "--output",
            "/tmp/results.json",
        ]
    )
    disabled_args = parser.parse_args(
        [
            "client",
            "--engine",
            "sglang",
            "--base-url",
            "http://127.0.0.1:8011",
            "--output",
            "/tmp/results.json",
            "--no-print-raw-response",
        ]
    )

    assert default_args.print_raw_response is True
    assert disabled_args.print_raw_response is False


def test_sglang_baseline_command_disables_radix_cache():
    command = _sglang_server_command(
        Namespace(
            gpu="7",
            model_path="/myapp/models/qwen35",
            model_name="qwen35-omni-s2t",
            sglang_port=8010,
            sglang_thinker_cuda_graph="on",
            sglang_thinker_torch_compile="off",
            sglang_thinker_mem_fraction_static=None,
            sglang_max_prefill_tokens=None,
        ),
        "/myapp/benchmarks/qwen35_s2t_align/run/sglang/events",
    )

    assert "--thinker-cuda-graph on" in command
    assert "--thinker-torch-compile off" in command
    assert "--encoder-mem-reserve 0.30" in command
    assert "--stages.4.runtime.sglang_server_args.disable_radix_cache true" in command


def test_sglang_command_accepts_diagnostic_overrides():
    command = _sglang_server_command(
        Namespace(
            gpu="7",
            model_path="/myapp/models/qwen35",
            model_name="qwen35-omni-s2t",
            sglang_port=8011,
            sglang_thinker_cuda_graph="off",
            sglang_thinker_torch_compile="on",
            sglang_thinker_mem_fraction_static=0.5,
            sglang_max_prefill_tokens=4096,
        ),
        "/myapp/benchmarks/qwen35_s2t_align/run/sglang/events",
    )

    assert "--thinker-cuda-graph off" in command
    assert "--thinker-torch-compile on" in command
    assert "--thinker-mem-fraction-static 0.5" in command
    assert "--encoder-mem-reserve" not in command
    assert "--stages.4.runtime.sglang_server_args.max_prefill_tokens 4096" in command


def test_vllm_reference_command_uses_installed_package_cwd_and_disables_mtp():
    command = _vllm_server_command(
        Namespace(
            gpu="7",
            model_path="/myapp/models/qwen35",
            vllm_port=8020,
            vllm_gpu_memory_utilization=0.9,
        )
    )

    assert command.startswith("cd /myapp/vllm/examples/offline_inference &&")
    assert "VLLM_FLASH_ATTN_USE_UPSTREAM=0" in command
    assert "python3 qwen_omni_v35_server.py" in command
    assert "--thinker-enforce-eager" in command
    assert "--disable-mtp" in command
    assert "--mm-processor-cache-type lru" in command


def test_alignment_report_compares_predictions(tmp_path: Path):
    sglang_result = {
        "summary": {"eval_accuracy": 1.0, "timestamp_loop_cases": []},
        "records": [
            {
                "case_id": "eval_001_001-1",
                "sample_id": "001-1",
                "probe_kind": "eval",
                "expected": "A",
                "predicted": "A",
                "is_correct": True,
                "latency_s": 1.23,
                "has_timestamp_loop": False,
                "error": "",
            }
        ],
    }
    vllm_result = {
        "summary": {"eval_accuracy": 0.0, "timestamp_loop_cases": []},
        "records": [
            {
                "case_id": "eval_001_001-1",
                "sample_id": "001-1",
                "probe_kind": "eval",
                "expected": "A",
                "predicted": "B",
                "is_correct": False,
                "latency_s": 2.34,
                "has_timestamp_loop": False,
                "error": "",
            }
        ],
    }
    log_path = tmp_path / "server.log"
    log_path.write_text("", encoding="utf-8")

    report = build_alignment_report(
        sglang_result=sglang_result,
        vllm_result=vllm_result,
        sglang_log_path=log_path,
        profile_report_path=tmp_path / "missing_profile.json",
    )

    assert report["summary"]["eval_accuracy_delta"] == 1.0
    assert report["summary"]["prediction_matches"] == 0
    assert report["diffs"][0]["sglang_predicted"] == "A"
    assert report["diffs"][0]["vllm_predicted"] == "B"
