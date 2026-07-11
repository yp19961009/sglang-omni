# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni processor shim used by the sglang-omni pipeline."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, WhisperFeatureExtractor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

from sglang_omni.models.qwen3_omni.components.preprocessor import (
    Qwen3OmniPreprocessor,
    _resolve_local_model_dir,
)
from sglang_omni.preprocessing import ensure_chat_template


def get_feat_extract_output_lengths(
    input_lengths: torch.Tensor, downsample_times: int = 4, chunk_size: int = 100
) -> torch.Tensor:
    input_lengths_leave = input_lengths % chunk_size
    for _ in range(downsample_times):
        input_lengths_leave = (input_lengths_leave - 1) // 2 + 1
    return input_lengths_leave + (input_lengths // chunk_size) * math.ceil(
        chunk_size / 2**downsample_times
    )


class Qwen35OmniNextProcessor:
    """Small local replacement for missing Transformers Qwen3OmniNextProcessor."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.video_processor = Qwen3VLVideoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.chat_template = getattr(self.tokenizer, "chat_template", None)

        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.merge_size = int(getattr(self.video_processor, "merge_size", 2))
        self.temporal_patch_size = int(
            getattr(self.video_processor, "temporal_patch_size", 2)
        )
        self.mm_token_pattern = re.compile(
            "|".join(
                re.escape(tok)
                for tok in (self.audio_token, self.image_token, self.video_token)
            )
        )

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(
        self,
        *,
        text: str | list[str],
        images: Any = None,
        videos: Any = None,
        audio: Any = None,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        videos_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        videos_kwargs = dict(videos_kwargs or {})
        audio_inputs, audio_lengths = self._process_audio(audio)
        image_inputs = self._process_images(images)
        video_inputs = self._process_videos(videos, videos_kwargs)

        texts = [text] if isinstance(text, str) else list(text)
        texts = self.replace_multimodal_special_tokens(
            texts,
            audio_lengths=iter(audio_lengths),
            image_grid_thw=iter(image_inputs.get("image_grid_thw", [])),
            video_grid_thw=iter(video_inputs.get("video_grid_thw", [])),
            video_fps=videos_kwargs.get("fps"),
        )
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=False,
        )
        return {**tokenized, **image_inputs, **video_inputs, **audio_inputs}

    def _process_audio(self, audio: Any) -> tuple[dict[str, Any], torch.Tensor]:
        if audio is None:
            return {}, torch.empty(0, dtype=torch.long)
        audios = audio if isinstance(audio, list) else [audio]
        padded = []
        sr = int(self.feature_extractor.sampling_rate)
        for item in audios:
            arr = np.asarray(item, dtype=np.float32)
            pad = sr - arr.shape[-1] % sr
            if pad != sr:
                arr = np.pad(arr, (0, pad))
            padded.append(arr)
        audio_inputs = self.feature_extractor(
            padded,
            sampling_rate=sr,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
        feature_lengths = audio_inputs["feature_attention_mask"].sum(-1).to(torch.long)
        audio_lengths = get_feat_extract_output_lengths(feature_lengths)
        return dict(audio_inputs), audio_lengths

    def _process_images(self, images: Any) -> dict[str, Any]:
        if images is None:
            return {}
        return dict(self.image_processor(images=images, return_tensors="pt"))

    def _process_videos(self, videos: Any, videos_kwargs: dict[str, Any]) -> dict[str, Any]:
        if videos is None:
            return {}
        video_items = videos if isinstance(videos, list) else [videos]
        fps_values = self._coerce_fps_values(
            videos_kwargs.get("fps"), len(video_items)
        )
        video_metadata = []
        for video, fps in zip(video_items, fps_values):
            num_frames = int(video.shape[0])
            video_metadata.append(
                {
                    "fps": fps,
                    "frames_indices": torch.arange(num_frames),
                    "total_num_frames": num_frames,
                    "video_backend": "preprocessed",
                }
            )
        proc_kwargs = {
            k: v
            for k, v in videos_kwargs.items()
            if k
            in {
                "min_pixels",
                "max_pixels",
                "size",
                "device",
                "do_resize",
                "do_rescale",
                "do_normalize",
            }
        }
        out = dict(
            self.video_processor(
                videos=video_items,
                video_metadata=video_metadata,
                do_sample_frames=False,
                return_metadata=True,
                return_tensors="pt",
                **proc_kwargs,
            )
        )
        out.pop("video_metadata", None)
        grid = out.get("video_grid_thw")
        if isinstance(grid, torch.Tensor):
            seconds = [self.temporal_patch_size / max(float(fps), 1e-6) for fps in fps_values]
            out["video_second_per_grid"] = torch.tensor(seconds, dtype=torch.float32)
        return out

    def _coerce_fps_values(self, fps: Any, count: int) -> list[float]:
        if isinstance(fps, torch.Tensor):
            fps = fps.detach().cpu().tolist()
        if isinstance(fps, (list, tuple)):
            values = [float(v) for v in fps]
        elif fps is None:
            values = [24.0]
        else:
            values = [float(fps)]
        if len(values) < count:
            values.extend([values[-1]] * (count - len(values)))
        return values[:count]

    def replace_multimodal_special_tokens(
        self,
        text: list[str],
        *,
        audio_lengths: Iterable[torch.Tensor | int],
        image_grid_thw: Iterable[torch.Tensor],
        video_grid_thw: Iterable[torch.Tensor],
        video_fps: Any,
    ) -> list[str]:
        processed: list[str] = []
        fps_iter = iter(self._fps_list(video_fps))
        for sample in text:
            special_tokens = self.mm_token_pattern.findall(sample)
            for special_token in special_tokens:
                if special_token == self.audio_token:
                    sample = sample.replace(
                        self.audio_token,
                        self._get_audio_tokens(int(next(audio_lengths))),
                        1,
                    )
                elif special_token == self.image_token:
                    grid = next(image_grid_thw)
                    image_seq_length = int(grid.prod().item()) // (self.merge_size**2)
                    sample = sample.replace(
                        self.image_token,
                        "<|image_placeholder|>" * image_seq_length,
                        1,
                    )
                elif special_token == self.video_token:
                    grid = next(video_grid_thw)
                    fps = next(fps_iter, 24.0)
                    target = self.vision_bos_token + self.video_token + self.vision_eos_token
                    replacement = self._get_video_tokens(grid, fps)
                    if target in sample:
                        sample = sample.replace(target, replacement, 1)
                    else:
                        sample = sample.replace(self.video_token, replacement, 1)
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed.append(sample)
        return processed

    def _fps_list(self, fps: Any) -> list[float]:
        if isinstance(fps, torch.Tensor):
            fps = fps.detach().cpu().tolist()
        if isinstance(fps, (list, tuple)):
            return [float(v) for v in fps]
        if fps is None:
            return [24.0]
        return [float(fps)]

    def _get_audio_tokens(
        self,
        audio_length: int,
        *,
        audio_tokens_per_second: int = 25,
        timestamp_interval: int = 60,
    ) -> str:
        tokens_interval = audio_tokens_per_second * timestamp_interval
        num_full_chunks = math.floor(audio_length / tokens_interval)
        num_residual_tokens = audio_length % tokens_interval
        pieces = []
        for i in range(num_full_chunks):
            pieces.append(f"<{i * timestamp_interval:.1f} seconds>")
            pieces.append("<|audio_placeholder|>" * tokens_interval)
        if num_residual_tokens > 0:
            pieces.append(f"<{num_full_chunks * timestamp_interval:.1f} seconds>")
            pieces.append("<|audio_placeholder|>" * num_residual_tokens)
        return "".join(pieces)

    def _get_video_tokens(self, video_grid_thw: torch.Tensor, fps: float) -> str:
        grid_t = int(video_grid_thw[0].item())
        grid_h = int(video_grid_thw[1].item())
        grid_w = int(video_grid_thw[2].item())
        frame_seq_len = grid_h * grid_w // (self.merge_size**2)
        pieces = []
        for frame_idx in range(grid_t):
            curr_time = (
                frame_idx * self.temporal_patch_size
                + (self.temporal_patch_size - 1) / 2.0
            ) / max(float(fps), 1e-6)
            pieces.append(f"<{curr_time:.1f} seconds>")
            pieces.append(self.vision_bos_token)
            pieces.append("<|video_placeholder|>" * frame_seq_len)
            pieces.append(self.vision_eos_token)
        return "".join(pieces)


class Qwen35OmniPreprocessor(Qwen3OmniPreprocessor):
    """Qwen3.5-Omni preprocessing with a local Qwen3OmniNext processor shim."""

    def __init__(
        self,
        model_path: str,
        max_seq_len: int | None = None,
        *,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_pixels: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.default_video_fps = float(video_fps) if video_fps is not None else None
        self.default_video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.default_video_min_pixels = int(video_min_pixels) if video_min_pixels is not None else None
        self.default_video_max_pixels = int(video_max_pixels) if video_max_pixels is not None else None
        self.default_video_total_pixels = int(video_total_pixels) if video_total_pixels is not None else None
        self.model_dir = _resolve_local_model_dir(model_path)
        self.processor = Qwen35OmniNextProcessor(self.model_dir)
        self.video_resize_factor = int(self.processor.video_processor.patch_size) * int(
            self.processor.video_processor.merge_size
        )
        self.image_encoder_input_dtype = torch.bfloat16
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_path=self.model_dir, fallback_model_paths=())
        if not getattr(self.processor, "chat_template", None) and getattr(
            self.tokenizer, "chat_template", None
        ):
            self.processor.chat_template = self.tokenizer.chat_template
