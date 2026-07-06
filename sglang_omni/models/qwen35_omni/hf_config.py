# SPDX-License-Identifier: Apache-2.0
"""Local HF config shims for Qwen3.5-Omni Next checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang_omni.models.qwen3_omni.hf_config import (
    Qwen3OmniMoeVisionEncoderConfig,
)


class Qwen35OmniNextAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_omni_next_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0,
        attention_dropout: float = 0,
        activation_function: str = "gelu",
        activation_dropout: float = 0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 50,
        output_dim: int = 2048,
        n_window_infer: int = 200,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen35OmniNextVisionEncoderConfig(Qwen3OmniMoeVisionEncoderConfig):
    model_type = "qwen3_omni_next_vision_encoder"


class Qwen35OmniNextTextConfig(Qwen3NextConfig):
    model_type = "qwen3_omni_next_text"


class Qwen35OmniNextThinkerConfig(PretrainedConfig):
    model_type = "qwen3_omni_next_thinker"

    def __init__(
        self,
        audio_config: Any | None = None,
        vision_config: Any | None = None,
        text_config: Any | None = None,
        audio_token_id: int = 248076,
        audio_start_token_id: int = 248070,
        audio_end_token_id: int = 248071,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        position_id_per_seconds: int = 13,
        user_token_id: int = 872,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.initializer_range = initializer_range

        if isinstance(vision_config, dict):
            vision_config = Qwen35OmniNextVisionEncoderConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen35OmniNextVisionEncoderConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Qwen35OmniNextAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen35OmniNextAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen35OmniNextTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen35OmniNextTextConfig()
        self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


class Qwen35OmniNextConfig(PretrainedConfig):
    model_type = "qwen3_omni_next"

    def __init__(
        self,
        thinker_config: Any | None = None,
        talker_config: Any | None = None,
        code2wav_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(thinker_config, dict):
            thinker_config = Qwen35OmniNextThinkerConfig(**thinker_config)
        elif thinker_config is None:
            thinker_config = Qwen35OmniNextThinkerConfig()
        self.thinker_config = thinker_config
        self.talker_config = talker_config
        self.code2wav_config = code2wav_config
        if not getattr(self, "architectures", None):
            self.architectures = ["Qwen3OmniNextForConditionalGeneration"]


_CONFIGS = (
    ("qwen3_omni_next", Qwen35OmniNextConfig),
    ("qwen3_omni_next_thinker", Qwen35OmniNextThinkerConfig),
    ("qwen3_omni_next_text", Qwen35OmniNextTextConfig),
    ("qwen3_omni_next_audio_encoder", Qwen35OmniNextAudioEncoderConfig),
    ("qwen3_omni_next_vision_encoder", Qwen35OmniNextVisionEncoderConfig),
)

for _model_type, _config_cls in _CONFIGS:
    try:
        AutoConfig.register(_model_type, _config_cls, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register(_model_type, _config_cls)
        except ValueError:
            pass
    except ValueError:
        pass
