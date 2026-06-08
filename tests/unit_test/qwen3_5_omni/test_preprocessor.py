# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components import preprocessor as qwen3_preprocessor
from sglang_omni.models.qwen3_5_omni.components import preprocessor
from sglang_omni.models.qwen3_5_omni.components import common as qwen35_common
from sglang_omni.models.qwen3_5_omni.components.preprocessor import (
    Qwen35OmniPreprocessor,
    _load_qwen3_omni_next_processor,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.preprocessing import video as video_preprocessing


class _FakeProcessor:
    def __init__(self, hf_input_ids: torch.Tensor | None = None) -> None:
        self.hf_input_ids = (
            hf_input_ids
            if hf_input_ids is not None
            else torch.tensor([[101, 102]], dtype=torch.long)
        )
        self.calls: list[dict[str, object]] = []
        self.template_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, "kwargs": kwargs})
        return "<templated>"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        data = {
            "input_ids": self.hf_input_ids,
            "attention_mask": torch.ones_like(self.hf_input_ids),
        }
        if kwargs.get("audio") is not None:
            data.update(
                {
                    "input_audio_features": torch.ones(1, 4, 3),
                    "feature_attention_mask": torch.ones(1, 3, dtype=torch.bool),
                }
            )
        return data


class _FakeMediaIO:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def load_base64(self, media_type: str, data: str):
        self.calls.append(("base64", (media_type, data)))
        return ("base64", media_type, data)

    def load_bytes(self, data: bytes):
        self.calls.append(("bytes", data))
        return ("bytes", data)


def _new_preprocessor_for_unit() -> Qwen35OmniPreprocessor:
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj.max_seq_len = None
    obj.default_image_min_pixels = None
    obj.default_image_max_pixels = None
    obj.default_video_fps = None
    obj.default_video_max_frames = None
    obj.default_video_min_frames = None
    obj.default_video_min_pixels = None
    obj.default_video_max_pixels = None
    obj.default_video_total_pixels = None
    obj.default_video_override_max_pixels = False
    obj.limit_mm_per_prompt = {}
    obj._audio_processor_defaults = {}
    obj.processor = _FakeProcessor()
    return obj


def test_qwen35_preprocessor_loader_returns_from_pretrained_class():
    processor_cls = _load_qwen3_omni_next_processor()

    assert hasattr(processor_cls, "from_pretrained")


def test_qwen35_preprocessor_uses_remote_processor_fallback():
    # 中文说明：当前测试环境可能还没有内置 qwen3_omni_next。
    # 这里固定子类接口，保证真实模型带 remote processor 时可以兜底加载。
    assert Qwen35OmniPreprocessor.chat_template_fallback_model_paths == ()


def test_qwen35_preprocessor_accepts_next_audio_feature_alias():
    input_audio_features = torch.ones(1, 4, 8)
    audio_inputs = qwen3_preprocessor._build_audio_mm_inputs_compat(
        {
            "input_audio_features": input_audio_features,
            "feature_attention_mask": torch.ones(1, 8, dtype=torch.bool),
        }
    )

    assert audio_inputs["input_features"] is input_audio_features
    assert audio_inputs["audio_feature_lengths"].tolist() == [8]


def test_qwen35_audio_cache_context_distinguishes_video_audio_order():
    explicit_then_video = qwen3_preprocessor._contextualize_cache_key(
        "audio:explicit|video:clip",
        audio_is_dependent=[False, True],
        target_sr=16000,
        use_audio_in_video=True,
    )
    video_then_explicit = qwen3_preprocessor._contextualize_cache_key(
        "audio:explicit|video:clip",
        audio_is_dependent=[True, False],
        target_sr=16000,
        use_audio_in_video=True,
    )
    per_video_flags = qwen3_preprocessor._contextualize_cache_key(
        "video:a,b",
        audio_is_dependent=torch.tensor([True]),
        target_sr=16000,
        use_audio_in_video=[False, True],
    )

    assert explicit_then_video != video_then_explicit
    assert "audio_is_dependent=(False, True)" in explicit_then_video
    assert "audio_is_dependent=(True, False)" in video_then_explicit
    assert "use_audio_in_video=(False, True)" in per_video_flags


def test_qwen_preprocessor_media_input_helpers_are_tensor_safe():
    tensor_video = torch.zeros(2, 3, 8, 8)
    tensor_audio = torch.zeros(16000)

    assert (
        qwen3_preprocessor._first_present_media_input(
            {"videos": tensor_video, "video": "fallback.mp4"},
            ("videos", "video"),
        )
        is tensor_video
    )
    assert (
        qwen3_preprocessor._first_present_media_input(
            {"videos": [], "video": tensor_video},
            ("videos", "video"),
        )
        is tensor_video
    )
    assert (
        qwen3_preprocessor._first_present_media_input(
            {"images": [], "image": tensor_video},
            ("images", "image"),
        )
        is tensor_video
    )
    assert qwen3_preprocessor._media_item_count(tensor_audio) == 1
    assert qwen3_preprocessor._media_item_count([tensor_audio, tensor_audio]) == 2
    assert qwen3_preprocessor._media_item_count([]) == 0


def test_qwen_preprocessor_distinguishes_audio_output_config_from_audio_input():
    output_audio_config = {"voice": "Cherry", "format": "wav", "language": "zh-CN"}
    input_audio_dict = {"url": "https://example.test/input.wav"}

    assert (
        qwen3_preprocessor._first_present_audio_input({"audio": output_audio_config})
        is None
    )
    assert qwen3_preprocessor._first_present_audio_input(
        {
            "audio": output_audio_config,
            "audios": ["input.wav"],
        }
    ) == ["input.wav"]
    assert (
        qwen3_preprocessor._first_present_audio_input({"audio": input_audio_dict})
        is input_audio_dict
    )


def test_qwen_preprocessor_accepts_rendered_vllm_prompt():
    assert (
        qwen3_preprocessor._request_raw_prompt_text({"prompt": "<rendered>"})
        == "<rendered>"
    )
    assert (
        qwen3_preprocessor._request_raw_prompt_text({"prompt": ["<rendered>"]})
        == "<rendered>"
    )
    assert (
        qwen3_preprocessor._request_raw_prompt_text(
            {
                "prompt": "<rendered>",
                "messages": [{"role": "user", "content": "plain"}],
            }
        )
        is None
    )


def test_qwen_preprocessor_accepts_vllm_prompt_token_ids():
    token_ids = qwen3_preprocessor._request_prompt_token_ids(
        {"prompt_token_ids": [1, 2, 3]}
    )
    batched_token_ids = qwen3_preprocessor._request_prompt_token_ids(
        {"prompt_token_ids": [[4, 5, 6]]}
    )

    assert torch.equal(token_ids, torch.tensor([1, 2, 3]))
    assert torch.equal(batched_token_ids, torch.tensor([4, 5, 6]))
    assert qwen3_preprocessor._request_prompt_token_ids({"prompt": "text"}) is None


def test_qwen35_preprocessor_uses_vllm_raw_prompt_without_template():
    obj = _new_preprocessor_for_unit()
    fake_processor = obj.processor
    payload = StagePayload(
        request_id="req-raw-prompt",
        request=OmniRequest(
            inputs={
                "prompt": "<|im_start|>user\nrendered prompt<|im_end|>\n",
            },
            params={"max_tokens": 8},
        ),
        data={},
    )

    result = asyncio.run(obj._call_impl(payload))

    assert fake_processor.template_calls == []
    assert fake_processor.calls[0]["text"] == payload.request.inputs["prompt"]
    prompt = result.data["prompt"]
    assert prompt["prompt_text"] == payload.request.inputs["prompt"]
    assert torch.equal(prompt["input_ids"], torch.tensor([101, 102]))


def test_qwen35_preprocessor_uses_vllm_prompt_token_ids_over_hf_ids():
    obj = _new_preprocessor_for_unit()
    fake_processor = obj.processor
    fake_processor.hf_input_ids = torch.tensor([[101, 102]], dtype=torch.long)
    audio = torch.zeros(8, dtype=torch.float32)
    payload = StagePayload(
        request_id="req-token-prompt",
        request=OmniRequest(
            inputs={
                "prompt_token_ids": [7, 8, 9],
                "multi_modal_data": {"audio": audio},
                "multi_modal_uuids": {"audio": ["audio-hash-1"]},
                "mm_processor_kwargs": {
                    "audio_kwargs": {"timestamp_interval": 15},
                },
            },
            params={"max_tokens": 8},
        ),
        data={},
    )

    result = asyncio.run(obj._call_impl(payload))

    assert fake_processor.template_calls == []
    assert fake_processor.calls[0]["text"] == ""
    assert fake_processor.calls[0]["audio"] == [audio]
    assert fake_processor.calls[0]["audio_kwargs"]["timestamp_interval"] == 15
    prompt = result.data["prompt"]
    assert prompt["prompt_text"] == ""
    assert torch.equal(prompt["input_ids"], torch.tensor([7, 8, 9]))
    assert torch.equal(prompt["attention_mask"], torch.ones(3, dtype=torch.long))
    assert result.data["mm_inputs"]["audio"]["audio_feature_lengths"].tolist() == [3]
    assert result.data["encoder_inputs"]["audio_encoder"]["cache_key"].startswith(
        'vllm_uuid:audio:["audio-hash-1"]'
    )


def test_qwen35_preprocessor_builds_vllm_uuid_cache_keys():
    inputs = {
        "multi_modal_uuids": [
            {"video": "video-hash-1"},
            {"videos": ["video-hash-2"]},
        ]
    }

    assert preprocessor._vllm_uuid_cache_key(inputs, "video") == (
        'vllm_uuid:video:["video-hash-1","video-hash-2"]'
    )
    assert preprocessor._vllm_uuid_cache_key(
        inputs,
        "video",
        expected_count=2,
    ) == 'vllm_uuid:video:["video-hash-1","video-hash-2"]'
    assert (
        preprocessor._vllm_uuid_cache_key(inputs, "video", expected_count=3)
        is None
    )
    assert preprocessor._vllm_uuid_cache_key(inputs, "audio") is None


def test_qwen35_preprocessor_falls_back_when_vllm_uuid_count_mismatches():
    obj = _new_preprocessor_for_unit()
    raw_videos = ["clip-a.mp4", "clip-b.mp4"]
    request_inputs = {
        "multi_modal_uuids": {
            "video": ["video-hash-a"],
        }
    }

    cache_key = obj._media_cache_key_for_request(
        request_inputs=request_inputs,
        modality="video",
        raw_value=raw_videos,
        compute_cache_key=lambda value: "raw:" + ",".join(value),
    )

    assert cache_key == "raw:clip-a.mp4,clip-b.mp4"


def test_qwen35_preprocessor_derives_video_total_pixels_for_override(monkeypatch):
    obj = _new_preprocessor_for_unit()
    obj.max_seq_len = 256
    obj.default_video_override_max_pixels = True
    captured: dict[str, object] = {}

    async def fake_ensure_video_list_async(videos, **kwargs):
        captured.update(kwargs)
        return [torch.zeros(2, 3, 4, 4)], [2.0], None

    monkeypatch.setattr(
        qwen3_preprocessor,
        "ensure_video_list_async",
        fake_ensure_video_list_async,
    )
    payload = StagePayload(
        request_id="req-video-override",
        request=OmniRequest(
            inputs={
                "prompt": "watch",
                "multi_modal_data": {"video": "clip.mp4"},
            },
            params={"max_tokens": 8},
        ),
        data={},
    )

    asyncio.run(obj._call_impl(payload))

    assert captured["override_max_pixels"] is True
    assert captured["total_pixels"] == (
        video_preprocessing.derive_video_total_pixels_from_mm_len(256)
    )


def test_qwen35_preprocessor_limit_mm_rejects_openai_content_parts():
    obj = _new_preprocessor_for_unit()
    obj.limit_mm_per_prompt = {"image": 1}
    payload = StagePayload(
        request_id="req-limit-openai",
        request=OmniRequest(
            inputs={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "a.png"}},
                            {"type": "image_url", "image_url": {"url": "b.png"}},
                        ],
                    }
                ]
            },
            params={"max_tokens": 8},
        ),
        data={},
    )

    with pytest.raises(ValueError, match="image: 2 > 1"):
        asyncio.run(obj._call_impl(payload))


def test_qwen35_preprocessor_limit_mm_rejects_vllm_multimodal_data():
    obj = _new_preprocessor_for_unit()
    obj.limit_mm_per_prompt = {"video": 1}
    payload = StagePayload(
        request_id="req-limit-vllm",
        request=OmniRequest(
            inputs={
                "prompt": "watch",
                "multi_modal_data": {
                    "video": ["a.mp4", "b.mp4"],
                },
            },
            params={"max_tokens": 8},
        ),
        data={},
    )

    with pytest.raises(ValueError, match="video: 2 > 1"):
        asyncio.run(obj._call_impl(payload))


def test_qwen35_preprocessor_does_not_treat_openai_audio_config_as_input():
    obj = _new_preprocessor_for_unit()
    fake_processor = obj.processor
    payload = StagePayload(
        request_id="req-audio-config",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "say hi"}],
                "audio": {"voice": "Cherry", "format": "wav"},
            },
            params={"max_tokens": 8},
            metadata={"audio_config": {"voice": "Cherry", "format": "wav"}},
        ),
        data={},
    )

    result = asyncio.run(obj._call_impl(payload))

    assert fake_processor.calls[0]["audio"] is None
    assert result.data["encoder_inputs"]["audio_encoder"] == {
        "_skip": True,
        "_result": {},
    }
    assert result.data["mm_inputs"].get("audio") == {}


def test_qwen35_preprocessor_passes_tools_to_chat_template():
    obj = _new_preprocessor_for_unit()
    fake_processor = obj.processor
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    payload = StagePayload(
        request_id="req-tools",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": tools,
            },
            params={"max_tokens": 8},
        ),
        data={},
    )

    asyncio.run(obj._call_impl(payload))

    assert fake_processor.template_calls[0]["kwargs"]["tools"] == tools
    assert fake_processor.template_calls[0]["kwargs"]["add_generation_prompt"] is True


def test_qwen_preprocessor_resolves_openai_max_tokens_for_context_check():
    assert qwen3_preprocessor._resolve_request_max_new_tokens({"max_tokens": 17}) == 17
    assert (
        qwen3_preprocessor._resolve_request_max_new_tokens(
            {
                "max_new_tokens": 19,
                "max_tokens": 17,
            }
        )
        == 19
    )
    assert (
        qwen3_preprocessor._resolve_request_max_new_tokens({})
        == qwen3_preprocessor.DEFAULT_THINKER_MAX_NEW_TOKENS
    )


def test_qwen35_preprocessor_requires_explicit_audio_in_video_for_video_inputs():
    obj = object.__new__(Qwen35OmniPreprocessor)

    assert obj._resolve_use_audio_in_video(
        {"videos": ["clip.mp4"]},
        ["clip.mp4"],
    ) is None
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "use_audio_in_video": True,
        },
        ["clip.mp4"],
    ) is True
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "use_audio_in_video": False,
        },
        ["clip.mp4"],
    ) is False
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "use_audio_in_video": "false",
        },
        ["clip.mp4"],
    ) is False
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "dependent_audio": [0],
        },
        ["clip.mp4"],
    ) is True
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "dependent_audio": [],
        },
        ["clip.mp4"],
    ) is None
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "dependent_audio": [False, False],
        },
        ["clip.mp4"],
    ) is None
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "dependent_audio": True,
        },
        ["clip.mp4"],
    ) is True
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "dependent_audio": "false",
        },
        ["clip.mp4"],
    ) is None
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "dependent_audio": "0",
        },
        ["clip.mp4"],
    ) is True
    assert obj._resolve_use_audio_in_video(
        {
            "videos": ["clip.mp4"],
            "use_audio_in_video": False,
            "dependent_audio": [0],
        },
        ["clip.mp4"],
    ) is False
    assert obj._resolve_use_audio_in_video({"messages": []}, None) is None


def test_qwen35_preprocessor_loads_audio_processor_defaults(monkeypatch):
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj.model_dir = "/models/qwen35"

    monkeypatch.setattr(
        preprocessor,
        "load_qwen35_thinker_config",
        lambda path: SimpleNamespace(
            audio_config=SimpleNamespace(
                downsample_times=3,
                downsample_chunk_size=80,
                timestamp_interval=12,
            )
        ),
    )

    assert obj._load_audio_processor_defaults() == {
        "downsample_times": 3,
        "downsample_chunk_size": 80,
        "timestamp_interval": 12,
    }


def test_qwen35_preprocessor_keeps_audio_processor_defaults_for_partial_config(
    monkeypatch,
):
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj.model_dir = "/models/qwen35"

    monkeypatch.setattr(
        preprocessor,
        "load_qwen35_thinker_config",
        lambda path: SimpleNamespace(
            audio_config=SimpleNamespace(downsample_times=3)
        ),
    )

    assert obj._load_audio_processor_defaults() == {
        "downsample_times": 3,
        "downsample_chunk_size": 100,
        "timestamp_interval": 60,
    }


def test_qwen35_preprocessor_uses_audio_processor_defaults_on_config_failure(
    monkeypatch,
):
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj.model_dir = "/models/qwen35"

    def fail_load_config(path):
        raise OSError(path)

    monkeypatch.setattr(preprocessor, "load_qwen35_thinker_config", fail_load_config)

    assert obj._load_audio_processor_defaults() == {
        "downsample_times": 4,
        "downsample_chunk_size": 100,
        "timestamp_interval": 60,
    }


def test_load_qwen35_thinker_config_accepts_root_and_direct_config(monkeypatch):
    root_thinker = SimpleNamespace(name="root-thinker")
    direct_thinker = SimpleNamespace(name="direct-thinker")
    split_thinker = SimpleNamespace(
        thinker_config=None,
        text_config=SimpleNamespace(hidden_size=8),
    )
    seen = []

    def fake_load_hf_config(path, **kwargs):
        seen.append((path, kwargs))
        if path == "/models/root":
            return SimpleNamespace(thinker_config=root_thinker)
        if path == "/models/root/thinker-shim":
            return split_thinker
        return direct_thinker

    monkeypatch.setattr(qwen35_common, "load_hf_config", fake_load_hf_config)

    assert qwen35_common.load_qwen35_thinker_config("/models/root") is root_thinker
    assert (
        qwen35_common.load_qwen35_thinker_config("/models/root/thinker")
        is direct_thinker
    )
    assert (
        qwen35_common.load_qwen35_thinker_config("/models/root/thinker-shim")
        is split_thinker
    )
    assert seen[0][1] == {"trust_remote_code": True, "local_files_only": True}


def test_load_qwen35_thinker_config_falls_back_to_raw_root_config(
    tmp_path,
    monkeypatch,
):
    model_root = tmp_path / "qwen35"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_next",
                "thinker_config": {
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "num_hidden_layers": 4,
                        "rope_parameters": {
                            "rope_type": "default",
                            "rope_theta": 1000000.0,
                        },
                        "quantization_config": {
                            "quant_method": "fp8",
                            "weight_block_size": [128, 128],
                        },
                    },
                    "audio_config": {"downsample_times": 3},
                    "vision_config": {"spatial_merge_size": 2},
                    "talker_language_id": {"zh": 77},
                },
                "talker_config": {
                    "code_predictor_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 2,
                        "rope_parameters": {
                            "rope_type": "default",
                            "rope_theta": 500000.0,
                        },
                    },
                    "speaker_id": {"f245": 0},
                    "speaker_system_prompt_id": {"f245": [1, 2]},
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_hf_config(path, **kwargs):
        del kwargs
        raise ValueError(path)

    monkeypatch.setattr(qwen35_common, "load_hf_config", fail_hf_config)

    thinker_config = qwen35_common.load_qwen35_thinker_config(str(model_root))

    assert thinker_config.audio_config.downsample_times == 3
    assert thinker_config.vision_config.spatial_merge_size == 2
    assert thinker_config.text_config.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert thinker_config.text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert thinker_config.text_config.rope_scaling == {
        "rope_type": "default",
        "rope_theta": 1000000.0,
    }
    assert thinker_config.text_config.rope_theta == 1000000.0
    assert thinker_config.text_config.partial_rotary_factor == 0.25
    assert thinker_config.text_config.quantization_config == {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    assert thinker_config.talker_language_id == {"zh": 77}
    root_config = qwen35_common.load_qwen35_config(str(model_root))
    assert root_config.talker_config.speaker_id == {"f245": 0}
    assert root_config.talker_config.speaker_system_prompt_id == {"f245": [1, 2]}
    assert root_config.talker_config.code_predictor_config.rope_scaling == {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }
    assert root_config.talker_config.code_predictor_config.rope_theta == 500000.0
    assert root_config.talker_config.code_predictor_config.layers_block_type == [
        "linear_attention",
        "linear_attention",
    ]
    assert root_config.talker_config.code_predictor_config.layer_types == [
        "linear_attention",
        "linear_attention",
    ]


def test_load_qwen35_thinker_config_falls_back_to_raw_split_config(
    tmp_path,
    monkeypatch,
):
    thinker_dir = tmp_path / "qwen35" / "thinker"
    thinker_dir.mkdir(parents=True)
    (thinker_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 8,
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 2,
                    "full_attention_interval": 1,
                },
                "audio_config": {"downsample_chunk_size": 80},
            }
        ),
        encoding="utf-8",
    )

    def fail_hf_config(path, **kwargs):
        del kwargs
        raise ValueError(path)

    monkeypatch.setattr(qwen35_common, "load_hf_config", fail_hf_config)

    thinker_config = qwen35_common.load_qwen35_thinker_config(str(thinker_dir))

    assert thinker_config.text_config.hidden_size == 8
    assert thinker_config.audio_config.downsample_chunk_size == 80
    assert thinker_config.text_config.layers_block_type == [
        "attention",
        "attention",
    ]
    assert thinker_config.text_config.layer_types == [
        "full_attention",
        "full_attention",
    ]


def test_qwen35_preprocessor_normalizes_openai_text_content_parts():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized == [{"role": "user", "content": "hello\nworld"}]


def test_qwen35_preprocessor_normalizes_openai_input_text_content_parts():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "input_text": "hello"},
                {"type": "input_text", "text": "world"},
                {"input_text": "again"},
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized == [{"role": "user", "content": "hello\nworld\nagain"}]


def test_qwen35_preprocessor_extracts_openai_multimodal_content_parts():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://a/img.png"}},
                {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                {"type": "input_video", "input_video": {"data": "def", "format": "mp4"}},
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["messages"] == [
        {
            "role": "user",
            "content": (
                "describe"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['image']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['audio']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
            ),
        }
    ]
    assert normalized["images"] == ["https://a/img.png"]
    assert normalized["audios"] == ["data:audio/wav;base64,abc"]
    assert normalized["videos"] == ["data:video/mp4;base64,def"]


def test_qwen35_preprocessor_extracts_top_level_media_part_payloads():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "url": "https://a/img.png"},
                {"type": "audio", "path": "speech.wav"},
                {"type": "input_video", "data": "AAAA", "format": "mp4"},
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["messages"] == [
        {
            "role": "user",
            "content": (
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['image']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['audio']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
            ),
        }
    ]
    assert normalized["images"] == ["https://a/img.png"]
    assert normalized["audios"] == ["speech.wav"]
    assert normalized["videos"] == ["data:video/mp4;base64,AAAA"]


def test_qwen35_preprocessor_normalizes_request_level_audio_aliases():
    inputs = {
        "messages": [{"role": "user", "content": "listen"}],
        "input_audio": {"url": "speech.wav"},
    }

    normalized = preprocessor._normalize_request_level_media_aliases(inputs)

    assert normalized is not inputs
    assert normalized["audios"] == "speech.wav"

    output_audio_config = {
        "messages": [{"role": "user", "content": "say hi"}],
        "audio": {"voice": "Cherry", "format": "wav"},
    }

    assert (
        preprocessor._normalize_request_level_media_aliases(output_audio_config)
        is output_audio_config
    )


def test_qwen35_preprocessor_normalizes_request_level_image_video_aliases():
    inputs = {
        "messages": [{"role": "user", "content": "watch"}],
        "image_url": {"data": "AAAA", "format": "png"},
        "input_video": {"url": "clip.mp4"},
    }

    normalized = preprocessor._normalize_request_level_media_aliases(inputs)

    assert normalized is not inputs
    assert normalized["images"] == "data:image/png;base64,AAAA"
    assert normalized["videos"] == "clip.mp4"


def test_qwen35_preprocessor_normalizes_existing_media_payloads():
    inputs = {
        "messages": [{"role": "user", "content": "watch"}],
        "images": [{"url": "frame.png"}],
        "videos": [{"data": "BBBB", "format": "mp4"}],
        "audios": [{"data": "CCCC", "format": "wav"}],
    }

    normalized = preprocessor._normalize_request_level_media_aliases(inputs)

    assert normalized["images"] == ["frame.png"]
    assert normalized["videos"] == ["data:video/mp4;base64,BBBB"]
    assert normalized["audios"] == ["data:audio/wav;base64,CCCC"]


def test_qwen35_preprocessor_merges_openai_audio_parts_with_request_alias():
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "listen"},
                    {"type": "input_audio", "input_audio": "inline.wav"},
                ],
            }
        ],
        "audio_url": "request.wav",
    }

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)
    normalized = preprocessor._normalize_request_level_media_aliases(normalized)

    assert normalized["audios"] == ["request.wav", "inline.wav"]


def test_qwen35_resource_connector_decodes_percent_encoded_base64_data_url():
    connector = MultiModalResourceConnector()
    media_io = _FakeMediaIO()

    result = connector.load_resource(
        "data:audio/wav;base64,YWJj%2BZA%3D%3D",
        media_io,
    )

    assert result == ("base64", "audio/wav", "YWJj+ZA==")
    assert media_io.calls == [("base64", ("audio/wav", "YWJj+ZA=="))]


def test_qwen35_resource_connector_supports_plain_data_url_bytes():
    connector = MultiModalResourceConnector()
    media_io = _FakeMediaIO()

    result = connector.load_resource("data:text/plain,hello%20world", media_io)

    assert result == ("bytes", b"hello world")
    assert media_io.calls == [("bytes", b"hello world")]


def test_qwen35_preprocessor_restores_openai_media_part_order():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "watch"},
                {"type": "video", "video": "clip.mp4"},
                {"type": "text", "text": "then listen"},
                {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                {"type": "image", "image": "frame.png"},
            ],
        }
    ]
    obj = object.__new__(Qwen35OmniPreprocessor)

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)
    messages_mm = obj._build_multimodal_messages(
        normalized["messages"],
        num_images=1,
        num_audios=1,
        num_videos=1,
    )

    assert messages_mm == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "watch"},
                {"type": "video"},
                {"type": "text", "text": "then listen"},
                {"type": "audio"},
                {"type": "image"},
            ],
        }
    ]


def test_qwen35_preprocessor_merges_openai_media_with_top_level_media():
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image", "image": "inline.png"},
                    {"type": "video_url", "video_url": {"url": "inline.mp4"}},
                    {"type": "audio_url", "audio_url": {"url": "inline.wav"}},
                ],
            }
        ],
        "images": ["top.png"],
        "videos": ["top.mp4"],
        "audios": ["top.wav"],
        "video_fps": 2,
    }

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["messages"] == [{"role": "user", "content": "compare"}]
    assert normalized["images"] == ["top.png", "inline.png"]
    assert normalized["videos"] == ["top.mp4", "inline.mp4"]
    assert normalized["audios"] == ["top.wav", "inline.wav"]
    assert normalized["video_fps"] == 2


def test_qwen35_preprocessor_lifts_openai_video_part_options():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "video",
                    "video": "clip.mp4",
                    "fps": 2,
                    "min_frames": 4,
                    "max_frames": 128,
                    "min_pixels": 4096,
                    "max_pixels": 401408,
                    "total_pixels": 32768 * 768,
                    "use_audio_in_video": True,
                    "seconds_per_chunk": 2.0,
                    "position_id_per_seconds": 25,
                },
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["messages"] == [
        {
            "role": "user",
            "content": (
                "describe"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
            ),
        }
    ]
    assert normalized["videos"] == ["clip.mp4"]
    assert normalized["video_fps"] == 2
    assert normalized["video_min_frames"] == 4
    assert normalized["video_max_frames"] == 128
    assert normalized["video_min_pixels"] == 4096
    assert normalized["video_max_pixels"] == 401408
    assert normalized["video_total_pixels"] == 32768 * 768
    assert normalized["use_audio_in_video"] is True
    assert normalized["video_seconds_per_chunk"] == 2.0
    assert normalized["video_position_id_per_seconds"] == 25


def test_qwen35_preprocessor_lifts_openai_image_part_options():
    inputs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image",
                    "image": "frame.png",
                    "min_pixels": 100352,
                    "max_pixels": 401408,
                },
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["messages"] == [
        {
            "role": "user",
            "content": (
                "describe"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['image']}"
            ),
        }
    ]
    assert normalized["images"] == ["frame.png"]
    assert normalized["image_min_pixels"] == 100352
    assert normalized["image_max_pixels"] == 401408


def test_qwen35_preprocessor_lifts_openai_audio_part_options():
    inputs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "abc",
                        "format": "wav",
                        "sampling_rate": 24000,
                        "timestamp_interval": 20,
                        "downsample_times": 3,
                        "downsample_chunk_size": 80,
                    },
                }
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["audios"] == ["data:audio/wav;base64,abc"]
    assert normalized["audio_target_sr"] == 24000
    assert normalized["audio_timestamp_interval"] == 20
    assert normalized["audio_downsample_times"] == 3
    assert normalized["audio_downsample_chunk_size"] == 80


def test_qwen35_preprocessor_keeps_top_level_audio_options():
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "inline.wav",
                            "audio_sampling_rate": 24000,
                            "timestamp_interval": 20,
                        },
                    }
                ],
            }
        ],
        "audio_target_sr": 16000,
    }

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["audios"] == ["inline.wav"]
    assert normalized["audio_target_sr"] == 16000
    assert normalized["audio_timestamp_interval"] == 20


def test_qwen35_preprocessor_lifts_nested_openai_image_options():
    inputs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.test/image.png",
                        "max_pixels": 200704,
                    },
                }
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["images"] == ["https://example.test/image.png"]
    assert normalized["image_max_pixels"] == 200704


def test_qwen35_preprocessor_lifts_per_video_audio_flags():
    inputs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "silent.mp4",
                    "use_audio_in_video": "false",
                },
                {
                    "type": "video",
                    "video": "talking.mp4",
                    "use_audio_in_video": "true",
                },
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["videos"] == ["silent.mp4", "talking.mp4"]
    assert normalized["use_audio_in_video"] == [False, True]


def test_qwen35_preprocessor_lifts_nested_openai_video_options():
    inputs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://example.test/clip.mp4",
                        "fps": 1,
                        "max_pixels": 200704,
                    },
                }
            ],
        }
    ]

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["videos"] == ["https://example.test/clip.mp4"]
    assert normalized["video_fps"] == 1
    assert normalized["video_max_pixels"] == 200704


def test_qwen35_preprocessor_keeps_top_level_video_options():
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "clip.mp4",
                        "fps": 2,
                        "max_frames": 128,
                    }
                ],
            }
        ],
        "video_fps": 4,
    }

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["videos"] == ["clip.mp4"]
    assert normalized["video_fps"] == 4
    assert normalized["video_max_frames"] == 128


def test_qwen35_preprocessor_keeps_top_level_audio_in_video_option():
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "clip.mp4",
                        "use_audio_in_video": False,
                    }
                ],
            }
        ],
        "use_audio_in_video": True,
    }

    normalized = preprocessor._normalize_openai_multimodal_inputs(inputs)

    assert normalized["videos"] == ["clip.mp4"]
    assert normalized["use_audio_in_video"] is True


def test_qwen35_preprocessor_merges_request_params_into_inputs():
    inputs = {"messages": [{"role": "user", "content": "watch"}]}

    merged = preprocessor._merge_request_params_into_inputs(
        inputs,
        {
            "use_audio_in_video": True,
            "max_frames": 128,
            "image_max_pixels": 401408,
            "dependent_audio": [0],
            "temperature": 0.2,
        },
    )

    assert merged is not inputs
    assert merged["use_audio_in_video"] is True
    assert merged["video_max_frames"] == 128
    assert merged["image_max_pixels"] == 401408
    assert merged["dependent_audio"] == [0]
    assert "temperature" not in merged


def test_qwen35_preprocessor_keeps_input_values_over_request_params():
    inputs = {
        "messages": [{"role": "user", "content": "watch"}],
        "use_audio_in_video": False,
        "video_max_frames": 64,
    }

    merged = preprocessor._merge_request_params_into_inputs(
        inputs,
        {
            "use_audio_in_video": True,
            "max_frames": 128,
            "video_dependent_audio": [0],
        },
    )

    assert merged["use_audio_in_video"] is False
    assert merged["video_max_frames"] == 64
    assert merged["dependent_audio"] == [0]


def test_qwen35_preprocessor_merges_params_before_openai_video_options():
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "clip.mp4",
                        "use_audio_in_video": False,
                        "max_frames": 16,
                    }
                ],
            }
        ],
    }

    merged = preprocessor._merge_request_params_into_inputs(
        inputs,
        {
            "use_audio_in_video": True,
            "video_max_frames": 128,
        },
    )
    normalized = preprocessor._normalize_openai_multimodal_inputs(merged)

    assert normalized["videos"] == ["clip.mp4"]
    assert normalized["use_audio_in_video"] is True
    assert normalized["video_max_frames"] == 128


def test_qwen35_preprocessor_wraps_openai_message_list_for_params():
    inputs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "clip.mp4",
                    "use_audio_in_video": False,
                }
            ],
        }
    ]

    merged = preprocessor._merge_request_params_into_inputs(
        inputs,
        {"use_audio_in_video": True},
    )
    normalized = preprocessor._normalize_openai_multimodal_inputs(merged)

    assert normalized["messages"] == [
        {
            "role": "user",
            "content": preprocessor._OPENAI_MEDIA_PLACEHOLDERS["video"],
        }
    ]
    assert normalized["videos"] == ["clip.mp4"]
    assert normalized["use_audio_in_video"] is True


def test_qwen35_preprocessor_lifts_vllm_text_prompt_multimodal_fields():
    inputs = {
        "prompt": "<|im_start|>user\n<|vision_start|><|video_pad|>",
        "multi_modal_data": {
            "image": ["frame.png"],
            "video": ["clip_tensor"],
            "audio": ["video_audio"],
        },
        "mm_processor_kwargs": {
            "fps": [2.0],
            "use_audio_in_video": [True],
            "dependent_audio": [0],
            "videos_kwargs": {
                "max_frames": 128,
                "return_metadata": True,
                "video_metadata": [{"fps": 2.0, "frames": 8}],
            },
            "images_kwargs": {"max_pixels": 401408},
            "audio_kwargs": {"timestamp_interval": 15},
        },
        "multi_modal_uuids": {
            "video": ["video-hash-1"],
            "audio": ["audio-hash-1"],
        },
    }

    normalized = preprocessor._normalize_vllm_multimodal_inputs(inputs)

    assert normalized is not inputs
    assert normalized["prompt"] == inputs["prompt"]
    assert "messages" not in normalized
    assert normalized["images"] == ["frame.png"]
    assert normalized["videos"] == ["clip_tensor"]
    assert normalized["audios"] == ["video_audio"]
    assert normalized["video_fps"] == 2.0
    assert normalized["video_max_frames"] == 128
    assert normalized["image_max_pixels"] == 401408
    assert normalized["audio_timestamp_interval"] == 15
    assert normalized["use_audio_in_video"] == [True]
    assert normalized["dependent_audio"] == [0]
    assert normalized["return_video_metadata"] is True
    assert normalized["video_metadata"] == [{"fps": 2.0, "frames": 8}]
    assert normalized["multi_modal_uuids"] == {
        "video": ["video-hash-1"],
        "audio": ["audio-hash-1"],
    }


def test_qwen35_preprocessor_lifts_vllm_list_multimodal_fields():
    inputs = {
        "prompt": "<rendered>",
        "multi_modal_data": [
            {"image": ["frame-a.png"]},
            {
                "video": "clip.mp4",
                "audio": ["speech.wav"],
            },
        ],
        "mm_processor_kwargs": [
            {"fps": [2.0]},
            {
                "videos_kwargs": {
                    "max_frames": 128,
                    "return_metadata": True,
                },
                "audio_kwargs": {"timestamp_interval": 15},
            },
        ],
    }

    normalized = preprocessor._normalize_vllm_multimodal_inputs(inputs)

    assert normalized is not inputs
    assert normalized["images"] == ["frame-a.png"]
    assert normalized["videos"] == ["clip.mp4"]
    assert normalized["audios"] == ["speech.wav"]
    assert normalized["video_fps"] == 2.0
    assert normalized["video_max_frames"] == 128
    assert normalized["return_video_metadata"] is True
    assert normalized["audio_timestamp_interval"] == 15


def test_qwen35_preprocessor_vllm_audio_data_not_shadowed_by_output_config():
    inputs = {
        "prompt": "<rendered>",
        "audio": {"voice": "Cherry", "format": "wav"},
        "multi_modal_data": {"audio": ["speech.wav"]},
    }

    normalized = preprocessor._normalize_vllm_multimodal_inputs(inputs)
    normalized = preprocessor._normalize_request_level_media_aliases(normalized)

    assert normalized["audio"] == {"voice": "Cherry", "format": "wav"}
    assert normalized["audios"] == ["speech.wav"]


def test_qwen35_preprocessor_keeps_top_level_values_over_vllm_kwargs():
    inputs = {
        "prompt": "rendered",
        "videos": ["top.mp4"],
        "video_fps": 4,
        "multi_modal_data": {"video": ["from_vllm.mp4"]},
        "mm_processor_kwargs": {
            "fps": 2,
            "videos_kwargs": {"max_frames": 128},
        },
    }

    normalized = preprocessor._normalize_vllm_multimodal_inputs(inputs)

    assert normalized["videos"] == ["top.mp4"]
    assert normalized["video_fps"] == 4
    assert normalized["video_max_frames"] == 128


def test_qwen35_preprocessor_passes_next_audio_kwargs():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {
        "downsample_times": 4,
        "downsample_chunk_size": 100,
        "timestamp_interval": 60,
    }

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "audio_timestamp_interval": 15,
            "audio_downsample_times": 5,
        },
        images_kwargs={},
        videos_kwargs={"use_audio_in_video": True},
        audio_target_sr=22050,
    )

    assert processor_kwargs["videos_kwargs"] == {
        "use_audio_in_video": True,
        "return_metadata": True,
    }
    assert processor_kwargs["audio_kwargs"] == {
        "sampling_rate": 22050,
        "padding": True,
        "return_attention_mask": True,
        "truncation": False,
        "downsample_times": 5,
        "downsample_chunk_size": 100,
        "timestamp_interval": 15,
    }


def test_qwen35_preprocessor_passes_image_kwargs():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={"images": ["frame.png"]},
        images_kwargs={
            "min_pixels": 100352,
            "max_pixels": 401408,
        },
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["images_kwargs"] == {
        "min_pixels": 100352,
        "max_pixels": 401408,
    }
    assert processor_kwargs["audio_kwargs"]["sampling_rate"] == 16000


def test_qwen35_preprocessor_requests_video_metadata_for_video_inputs():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={"videos": ["clip.mp4"]},
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {"return_metadata": True}


def test_qwen35_preprocessor_passes_vllm_video_metadata_kwargs():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip.mp4"],
            "video_metadata": [{"fps": 2.0}],
            "return_video_metadata": False,
        },
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {
        "video_metadata": [{"fps": 2.0}],
        "return_metadata": False,
    }


def test_qwen35_preprocessor_uses_dependent_audio_as_video_audio_signal():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip.mp4"],
            "use_audio_in_video": True,
            "dependent_audio": [0],
        },
        images_kwargs={},
        videos_kwargs={"use_audio_in_video": True},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {
        "use_audio_in_video": True,
        "return_metadata": True,
    }


def test_qwen35_preprocessor_does_not_pass_dependent_audio_to_hf_processor():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip-a.mp4", "clip-b.mp4"],
            "audios": ["explicit.wav"],
            "use_audio_in_video": True,
        },
        images_kwargs={},
        videos_kwargs={"use_audio_in_video": True},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {
        "use_audio_in_video": True,
        "return_metadata": True,
    }


def test_qwen35_preprocessor_does_not_auto_fill_dependent_audio_when_disabled():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip-a.mp4"],
            "audios": ["explicit.wav"],
            "use_audio_in_video": False,
        },
        images_kwargs={},
        videos_kwargs={"use_audio_in_video": False},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {
        "use_audio_in_video": False,
        "return_metadata": True,
    }


def test_qwen35_preprocessor_accepts_dependent_audio_alias():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip.mp4"],
            "video_dependent_audio": [1, 3],
        },
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {
        "use_audio_in_video": True,
        "return_metadata": True,
    }


def test_qwen35_preprocessor_ignores_empty_dependent_audio_alias():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip.mp4"],
            "video_dependent_audio": [],
        },
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {"return_metadata": True}


def test_qwen35_preprocessor_ignores_false_dependent_audio_mask():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={
            "videos": ["clip.mp4"],
            "dependent_audio": [False, False],
        },
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {"return_metadata": True}


def test_qwen35_preprocessor_orders_openai_video_audio_by_prompt_order():
    obj = object.__new__(Qwen35OmniPreprocessor)
    messages = [
        {
            "role": "user",
            "content": (
                "watch"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
                "then listen"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['audio']}"
            ),
        }
    ]

    audios, used_video_audio = obj._merge_audio_inputs_for_request(
        messages=messages,
        explicit_audios=["explicit.wav"],
        video_audios=["video-track.wav"],
        use_audio_in_video=True,
    )

    assert audios == ["video-track.wav", "explicit.wav"]
    assert used_video_audio is True


def test_qwen35_preprocessor_orders_openai_audio_before_video_audio():
    obj = object.__new__(Qwen35OmniPreprocessor)
    messages = [
        {
            "role": "user",
            "content": (
                "listen"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['audio']}"
                "then watch"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
            ),
        }
    ]

    audios, used_video_audio = obj._merge_audio_inputs_for_request(
        messages=messages,
        explicit_audios=["explicit.wav"],
        video_audios=["video-track.wav"],
        use_audio_in_video=True,
    )

    assert audios == ["explicit.wav", "video-track.wav"]
    assert used_video_audio is True


def test_qwen35_preprocessor_marks_openai_video_audio_as_dependent():
    obj = object.__new__(Qwen35OmniPreprocessor)
    messages = [
        {
            "role": "user",
            "content": (
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['audio']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
            ),
        }
    ]

    audio_is_dependent = obj._audio_is_dependent_for_request(
        messages=messages,
        explicit_audios=["explicit.wav"],
        video_audios=["video-track.wav"],
        use_audio_in_video=True,
    )

    assert audio_is_dependent == [False, True]


def test_qwen35_preprocessor_respects_per_video_audio_flags():
    obj = object.__new__(Qwen35OmniPreprocessor)
    messages = [
        {
            "role": "user",
            "content": (
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['video']}"
                f"{preprocessor._OPENAI_MEDIA_PLACEHOLDERS['audio']}"
            ),
        }
    ]

    audios, used_video_audio = obj._merge_audio_inputs_for_request(
        messages=messages,
        explicit_audios=["explicit.wav"],
        video_audios=["ignored-track.wav", "used-track.wav"],
        use_audio_in_video=[False, True],
    )

    assert audios == ["used-track.wav", "explicit.wav"]
    assert used_video_audio is True


def test_qwen35_preprocessor_orders_default_video_audio_before_explicit_audio():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audios, used_video_audio = obj._merge_audio_inputs_for_request(
        messages=[{"role": "user", "content": "describe"}],
        explicit_audios=["explicit.wav"],
        video_audios=["video-track.wav"],
        use_audio_in_video=True,
    )

    assert audios == ["video-track.wav", "explicit.wav"]
    assert used_video_audio is True


def test_qwen35_preprocessor_marks_default_video_audio_as_dependent():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audio_is_dependent = obj._audio_is_dependent_for_request(
        messages=[{"role": "user", "content": "describe"}],
        explicit_audios=["explicit.wav"],
        video_audios=["video-track.wav"],
        use_audio_in_video=True,
    )

    assert audio_is_dependent == [True, False]


def test_qwen35_preprocessor_respects_default_per_video_audio_flags():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audios, used_video_audio = obj._merge_audio_inputs_for_request(
        messages=[{"role": "user", "content": "describe"}],
        explicit_audios=["explicit.wav"],
        video_audios=["ignored-track.wav", "used-track.wav"],
        use_audio_in_video=[False, True],
    )

    assert audios == ["used-track.wav", "explicit.wav"]
    assert used_video_audio is True


def test_qwen35_preprocessor_keeps_default_explicit_audio_without_video_audio():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audios, used_video_audio = obj._merge_audio_inputs_for_request(
        messages=[{"role": "user", "content": "describe"}],
        explicit_audios=["explicit.wav"],
        video_audios=["video-track.wav"],
        use_audio_in_video=False,
    )

    assert audios == ["explicit.wav"]
    assert used_video_audio is False


def test_qwen35_preprocessor_preserves_per_video_audio_flags_for_processor():
    obj = object.__new__(Qwen35OmniPreprocessor)

    assert obj._processor_use_audio_in_video_value([False, True]) == [False, True]
    assert obj._processor_use_audio_in_video_value(["false", "true"]) == [
        False,
        True,
    ]
    assert obj._processor_use_audio_in_video_value(True) is True
    assert obj._processor_use_audio_in_video_value("false") is False


def test_qwen35_preprocessor_marks_vllm_dependent_audio_indices():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audio_is_dependent = obj._audio_is_dependent_for_request(
        request_inputs={"dependent_audio": [0, 2]},
        messages=[],
        explicit_audios=["video-audio-a", "explicit.wav", "video-audio-b"],
        video_audios=None,
        use_audio_in_video=True,
    )

    assert audio_is_dependent == [True, False, True]


def test_qwen35_preprocessor_accepts_dependent_audio_bool_strings():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audio_is_dependent = obj._audio_is_dependent_for_request(
        request_inputs={"dependent_audio": ["true", "false"]},
        messages=[],
        explicit_audios=["video-audio", "explicit.wav", "tail.wav"],
        video_audios=None,
        use_audio_in_video=True,
    )

    assert audio_is_dependent == [True, False, False]


def test_qwen35_preprocessor_accepts_dependent_audio_boolean_mask():
    obj = object.__new__(Qwen35OmniPreprocessor)

    audio_is_dependent = obj._audio_is_dependent_for_request(
        request_inputs={"dependent_audio": [True, False]},
        messages=[],
        explicit_audios=["video-audio", "explicit.wav", "tail.wav"],
        video_audios=None,
        use_audio_in_video=True,
    )

    assert audio_is_dependent == [True, False, False]


def test_video_loader_uses_per_video_audio_extraction_flags():
    class _FakeConnector:
        def __init__(self):
            self.calls = []

        async def fetch_video_async(self, url, **kwargs):
            extract_audio = kwargs["extract_audio"]
            self.calls.append((url, extract_audio))
            return f"video:{url}", 2.0, f"audio:{url}" if extract_audio else None

    async def _run():
        connector = _FakeConnector()
        videos, sampled_fps, audios = await video_preprocessing.ensure_video_list_async(
            ["https://example.test/a.mp4", "https://example.test/b.mp4"],
            resource_connector=connector,
            extract_audio=[False, True],
        )
        return connector, videos, sampled_fps, audios

    connector, videos, sampled_fps, audios = asyncio.run(_run())

    assert connector.calls == [
        ("https://example.test/a.mp4", False),
        ("https://example.test/b.mp4", True),
    ]
    assert videos == [
        "video:https://example.test/a.mp4",
        "video:https://example.test/b.mp4",
    ]
    assert sampled_fps == [2.0, 2.0]
    assert audios == [None, "audio:https://example.test/b.mp4"]


def test_qwen35_preprocessor_detects_tensor_video_without_bool_error():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={"video": torch.zeros(2, 3, 8, 8)},
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert processor_kwargs["videos_kwargs"] == {"return_metadata": True}


def test_qwen35_preprocessor_ignores_empty_video_list():
    obj = object.__new__(Qwen35OmniPreprocessor)
    obj._audio_processor_defaults = {}

    processor_kwargs = obj._processor_kwargs_for_request(
        request_inputs={"videos": []},
        images_kwargs={},
        videos_kwargs={},
        audio_target_sr=16000,
    )

    assert "videos_kwargs" not in processor_kwargs
