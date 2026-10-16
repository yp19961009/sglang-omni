from benchmarks.eval.qwen35_omni_sglang_rtc_concurrency import (
    _extract_vllm_style_profile_rows,
)


def _event(stage, event_name, t_rel_ms, metadata=None):
    return {
        "stage": stage,
        "event_name": event_name,
        "t_rel_ms": t_rel_ms,
        "timestamp_ns": int(t_rel_ms * 1_000_000),
        "metadata": metadata or {},
    }


def test_profile_rows_match_client_request_id_from_metadata():
    profile = {
        "timelines": {
            "internal-prefix": [
                _event(
                    "coordinator",
                    "request_admission",
                    0.0,
                    {
                        "metadata_request_id": "__prefix__client-a_t39",
                        "pre_run": True,
                    },
                ),
                _event("thinker", "scheduler_first_emit", 1.0),
            ],
            "internal-b": [
                _event(
                    "coordinator",
                    "request_admission",
                    0.0,
                    {"metadata_request_id": "client-b", "trunk_size": 40},
                ),
                _event("thinker", "scheduler_first_emit", 222.0),
                _event(
                    "coordinator",
                    "stage_stream_chunk_received",
                    333.0,
                    {"from_stage": "code2wav", "modality": "audio"},
                ),
            ],
            "internal-a": [
                _event(
                    "coordinator",
                    "request_admission",
                    0.0,
                    {"metadata_request_id": "client-a", "trunk_size": 40},
                ),
                _event("thinker", "scheduler_first_emit", 111.0),
                _event(
                    "coordinator",
                    "stage_stream_chunk_received",
                    222.0,
                    {"from_stage": "code2wav", "modality": "audio"},
                ),
            ],
        }
    }

    rows = _extract_vllm_style_profile_rows(profile, {"client-a", "client-b"})
    by_client_id = {row["profile_metadata_request_id"]: row for row in rows}

    assert set(by_client_id) == {"client-a", "client-b"}
    assert by_client_id["client-a"]["profile_request_id"] == "internal-a"
    assert by_client_id["client-a"]["profile_thinker_ttft_ms"] == 111.0
    assert by_client_id["client-b"]["profile_request_id"] == "internal-b"
    assert by_client_id["client-b"]["profile_thinker_ttft_ms"] == 222.0
