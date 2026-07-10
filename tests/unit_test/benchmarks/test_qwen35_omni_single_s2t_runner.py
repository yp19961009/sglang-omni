# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from pathlib import Path

from benchmarks.eval.qwen35_omni_single_s2t import (
    VideoAMMESample,
    _client_profile_paths,
    _preprocessed_audio_cache_path,
    _preprocessed_video_cache_path,
    _sglang_payload,
    _sglang_server_command,
    _vllm_payload,
    _vllm_server_command,
    build_alignment_report,
    build_parser,
    build_cases,
    client_stdout_payload,
    parse_prediction,
    parse_vllm_server_profile,
    vllm_profile_summary,
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


def test_vllm_payload_forwards_video_sampling_options():
    args = Namespace(
        model_name="qwen35-omni-s2t",
        prompt="Answer with one option.",
        temperature=0.0,
        top_p=0.8,
        top_k=1,
        max_tokens=4,
        seed=0,
        video_fps=2.0,
        video_max_frames=128,
        video_max_pixels=401408,
    )

    payload = _vllm_payload(args, _sample("001-1"))

    assert payload["video_fps"] == 2.0
    assert payload["video_max_frames"] == 128
    assert payload["video_max_pixels"] == 401408
    assert payload["messages"][0]["content"][0]["video_url"].endswith("001-1.mp4")


def test_sglang_payload_can_use_preprocessed_video_and_audio_dirs(tmp_path):
    args = Namespace(
        model_name="qwen35-omni-s2t",
        prompt="Answer with one option.",
        temperature=0.0,
        top_p=0.8,
        top_k=1,
        max_tokens=4,
        seed=0,
        video_fps=1.0,
        video_max_frames=128,
        video_max_pixels=401408,
        preprocessed_video_dir=str(tmp_path / "video"),
        preprocessed_audio_dir=str(tmp_path / "audio"),
    )
    sample = _sample("001-1")
    video_cache_path = _preprocessed_video_cache_path(args, sample)
    audio_cache_path = _preprocessed_audio_cache_path(args, sample)
    assert video_cache_path is not None
    assert audio_cache_path is not None
    video_cache_path.parent.mkdir(parents=True, exist_ok=True)
    audio_cache_path.parent.mkdir(parents=True, exist_ok=True)
    video_cache_path.write_bytes(b"cached-video")
    audio_cache_path.write_bytes(b"cached-audio")

    payload = _sglang_payload(args, sample)

    assert payload["preprocessed_videos"] == [{"path": str(video_cache_path)}]
    assert payload["preprocessed_audios"] == [{"path": str(audio_cache_path)}]
    assert "videos" not in payload
    assert "audios" not in payload
    assert "video_fps" not in payload
    assert "video_max_frames" not in payload
    assert "video_max_pixels" not in payload


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


def test_client_stdout_payload_optionally_includes_profile():
    result = {
        "summary": {"total_cases": 1},
        "records": [],
        "profile": {
            "run_id": "run-1",
            "event_dir": "/tmp/results_events",
            "report_path": "/tmp/results_profile.json",
            "request_count": 1,
            "stage_breakdown": [{"stage": "thinker", "total_ms": 123.4}],
            "hop_breakdown": [],
        },
    }

    assert client_stdout_payload(
        result, print_raw_response=False, print_profile=False
    ) == {"total_cases": 1}
    assert client_stdout_payload(
        result, print_raw_response=False, print_profile=True
    ) == {
        "total_cases": 1,
        "profile": result["profile"],
    }


def test_client_profile_paths_default_next_to_output():
    args = Namespace(
        output="/myapp/benchmarks/qwen35_s2t_align/manual-single/results.json",
        profile_event_dir=None,
        profile_output=None,
    )

    assert _client_profile_paths(args, run_id="run-1") == (
        "/myapp/benchmarks/qwen35_s2t_align/manual-single/results_events/run-1",
        "/myapp/benchmarks/qwen35_s2t_align/manual-single/results_profile.json",
    )


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
    assert default_args.profile is True
    assert default_args.print_profile is True


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


def test_parse_vllm_server_profile_extracts_request_breakdown(tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            [
                "INFO [qwen_omni_v35_server.py:420] [chatcmpl-abc123] TIMING stage=http.request_json ms=0.070 since_start_ms=0.091 content_length=692",
                "INFO [qwen_omni_v35_server.py:420] [chatcmpl-abc123] TIMING stage=build_prompt.process_mm_info ms=6110.834 since_start_ms=7237.020 audios=1 images=0 videos=1 video_kwargs=['do_sample_frames']",
                "INFO [qwen3_omni_next.py:394] Qwen3OmniNextProcessor preprocessing stats: audio_items=1 (0.012s), image_items=0 (0.000s), video_items=1 (0.150s), total_preprocess=0.162s, replace_multimodal_special_tokens=0.002s",
                "INFO [processing.py:2482] HFPREP_PROFILE apply: total=840.5ms cached_hf=824.2ms prompt_updates=16.4ms is_update_applied=False",
                "INFO [async_llm.py:707] thinker input_preprocessor finished, rid: chatcmpl-abc123, cost:5664.1",
                "[0/1][pid=1144] INFO [gpu_model_runner.py:2936] encode all mm inputs done, cost: 367.483642578125 ms, (request_id, item_cnt): {'chatcmpl-abc123': 2}",
                "INFO [qwen_omni_v35_server.py:420] [chatcmpl-abc123] TIMING stage=engine.thinker_first_output ms=7340.500 since_start_ms=20245.100 prompt_tokens=8138 output_tokens=1 finished=False",
                "INFO [qwen_omni_v35_server.py:420] [chatcmpl-abc123] TIMING stage=engine.thinker_final_output ms=7573.070 since_start_ms=20498.100 prompt_tokens=8138 output_tokens=4 finish_reason=stop",
                "[EngineCore_DP0][pid=777] INFO [scheduler.py:1212] SCHED_STEP prefill_reqs=1(8138 tok) decode_reqs=0(0 tok) encoder_reqs=1 total=8138",
                "[EngineCore_DP0][pid=777] INFO [scheduler.py:2330] Request chatcmpl-abc123 finished, output length: 4, reason: stop, stop reason: None",
                "INFO [qwen_omni_v35_server.py:420] [chatcmpl-abc123] TIMING stage=http.total ms=20498.742 since_start_ms=20498.742",
            ]
        ),
        encoding="utf-8",
    )

    profile = parse_vllm_server_profile(log_path)
    stages = {row["stage"]: row for row in profile["stage_breakdown"]}

    assert profile["request_count"] == 1
    assert stages["http.total"]["total_ms"] == 20498.742
    assert stages["build_prompt.process_mm_info"]["total_ms"] == 6110.834
    assert stages["engine.input_preprocessor"]["total_ms"] == 5664.1
    assert stages["hfprep.apply"]["total_ms"] == 840.5
    assert stages["engine.mm_encoder"]["total_ms"] == 367.484
    assert stages["engine.thinker_first_output"]["total_ms"] == 7340.5
    assert stages["engine.thinker_final_output"]["total_ms"] == 7573.07
    assert stages["engine.thinker_decode_after_first"]["total_ms"] == 253.0
    request = profile["requests"]["chatcmpl-abc123"]
    assert request["finished"] == {"output_length": 4, "reason": "stop"}
    assert len(request["scheduler_events"]) == 1


def test_vllm_profile_summary_keeps_breakdown_and_unattributed_events():
    profile = {
        "request_count": 1,
        "stage_breakdown": [{"stage": "http.total", "total_ms": 1.0}],
        "unattributed_events": [{"stage": "hfprep.apply"}],
        "requests": {"chatcmpl-abc123": {}},
    }

    assert vllm_profile_summary(profile) == {
        "request_count": 1,
        "stage_breakdown": [{"stage": "http.total", "total_ms": 1.0}],
        "unattributed_events": [{"stage": "hfprep.apply"}],
    }


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
    vllm_profile_path = tmp_path / "vllm_profile.json"
    vllm_profile_path.write_text(
        '{"request_count": 1, "stage_breakdown": [{"stage": "http.total", "total_ms": 2.0}], "unattributed_events": []}',
        encoding="utf-8",
    )

    report = build_alignment_report(
        sglang_result=sglang_result,
        vllm_result=vllm_result,
        sglang_log_path=log_path,
        profile_report_path=tmp_path / "missing_profile.json",
        vllm_profile_report_path=vllm_profile_path,
    )

    assert report["summary"]["eval_accuracy_delta"] == 1.0
    assert report["summary"]["prediction_matches"] == 0
    assert report["diffs"][0]["sglang_predicted"] == "A"
    assert report["diffs"][0]["vllm_predicted"] == "B"
    assert report["vllm_profile"]["request_count"] == 1
