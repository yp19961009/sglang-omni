# SPDX-License-Identifier: Apache-2.0
"""Request helpers for Qwen3.5-Omni stages."""

from __future__ import annotations

from typing import Any

import torch
import xxhash

from sglang_omni.models.qwen3_omni import request_builders as qwen3_builders
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

IMAGE_STAGE = qwen3_builders.IMAGE_STAGE
AUDIO_STAGE = qwen3_builders.AUDIO_STAGE
THINKER_STAGE = qwen3_builders.THINKER_STAGE
DECODE_STAGE = qwen3_builders.DECODE_STAGE
MM_AGGREGATE_STAGE = qwen3_builders.MM_AGGREGATE_STAGE

output_modalities = qwen3_builders.output_modalities
should_generate_audio_output = qwen3_builders.should_generate_audio_output
resolve_preprocessing_next_stages = qwen3_builders.resolve_preprocessing_next_stages
resolve_mm_aggregate_wait_sources = qwen3_builders.resolve_mm_aggregate_wait_sources
project_preprocessing_to_image_encoder = qwen3_builders.project_preprocessing_to_image_encoder
project_preprocessing_to_audio_encoder = qwen3_builders.project_preprocessing_to_audio_encoder
project_preprocessing_to_mm_aggregate = qwen3_builders.project_preprocessing_to_mm_aggregate
project_encoder_to_mm_aggregate = qwen3_builders.project_encoder_to_mm_aggregate
project_thinker_to_decode = qwen3_builders.project_thinker_to_decode
build_encoder_request = qwen3_builders.build_encoder_request
apply_encoder_result = qwen3_builders.apply_encoder_result
build_lightweight_mm_inputs = qwen3_builders.build_lightweight_mm_inputs
apply_thinker_result = qwen3_builders.apply_thinker_result
make_thinker_stream_output_builder = qwen3_builders.make_thinker_stream_output_builder


def _as_grid_rows(value: Any) -> list[list[int]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [[int(x) for x in row] for row in value]


def _flatten_video_grid_rows(video_grid_thw: Any) -> list[list[int]]:
    rows = []
    for t, h, w in _as_grid_rows(video_grid_thw):
        rows.extend([[1, h, w] for _ in range(int(t))])
    return rows


def _append_text_positions(
    parts: list[torch.Tensor],
    *,
    length: int,
    start: int,
) -> int:
    if length <= 0:
        return start
    parts.append(torch.arange(length).view(1, -1).expand(3, -1) + start)
    return start + length


def _compute_mrope_positions(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    image_grid_rows = _as_grid_rows(model_inputs.get("image_grid_thw"))
    video_grid_rows = _flatten_video_grid_rows(model_inputs.get("video_grid_thw"))
    ids_2d = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    ids_2d = ids_2d.cpu().to(dtype=torch.long)

    if not image_grid_rows and not video_grid_rows:
        seq_len = ids_2d.shape[1]
        pos = torch.arange(seq_len, dtype=torch.long).view(1, 1, -1).expand(3, ids_2d.shape[0], -1)
        delta = pos.max(0, keepdim=False)[0].max(-1, keepdim=True)[0] + 1 - seq_len
        return pos.squeeze(1), delta

    image_token_id = int(thinker_config.image_token_id)
    video_token_id = int(thinker_config.video_token_id)
    audio_token_id = int(thinker_config.audio_token_id)
    vision_start_token_id = int(thinker_config.vision_start_token_id)
    vision_end_token_id = int(thinker_config.vision_end_token_id)
    spatial_merge_size = int(thinker_config.vision_config.spatial_merge_size)

    position_ids = torch.zeros(
        3,
        ids_2d.shape[0],
        ids_2d.shape[1],
        dtype=torch.long,
        device=ids_2d.device,
    )
    deltas = []
    image_index = 0
    video_index = 0

    for batch_idx, current_input_ids in enumerate(ids_2d):
        input_tokens = [int(x) for x in current_input_ids.tolist()]
        input_tokens_tensor = current_input_ids
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = input_tokens_tensor[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int((vision_tokens == video_token_id).sum().item())
        else:
            image_nums = 0
            video_nums = 0

        parts: list[torch.Tensor] = []
        st = 0
        remain_images = image_nums
        remain_videos = video_nums

        for _ in range(image_nums + video_nums):
            ed_image = (
                input_tokens.index(image_token_id, st)
                if image_token_id in input_tokens[st:] and remain_images > 0
                else len(input_tokens) + 1
            )
            ed_video = (
                input_tokens.index(video_token_id, st)
                if video_token_id in input_tokens[st:] and remain_videos > 0
                else len(input_tokens) + 1
            )
            if ed_image < ed_video:
                t, h, w = image_grid_rows[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_rows[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            text_len = ed - st
            st_idx = int(parts[-1].max().item()) + 1 if parts else 0
            st_idx = _append_text_positions(parts, length=text_len, start=st_idx)

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            vision_pos = torch.stack([t_index, h_index, w_index]) + st_idx
            vision_pos = torch.cat(
                [vision_pos, torch.full((3, 1), int(vision_pos.max().item()) + 1)],
                dim=-1,
            )

            try:
                vision_end_index = input_tokens.index(vision_end_token_id, st)
            except ValueError:
                vision_end_index = ed + llm_grid_t * llm_grid_h * llm_grid_w
            audio_start = vision_end_index + 1
            audio_length = 0
            for token in input_tokens[audio_start:]:
                if token != audio_token_id:
                    break
                audio_length += 1
            if audio_length > 0:
                audio_pos = (
                    torch.arange(audio_length).view(1, -1).expand(3, -1) + st_idx
                )
                vision_pos = torch.cat([vision_pos, audio_pos], dim=-1)

            parts.append(vision_pos)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w + 1 + audio_length

        if st < len(input_tokens):
            st_idx = int(parts[-1].max().item()) + 1 if parts else 0
            _append_text_positions(parts, length=len(input_tokens) - st, start=st_idx)

        llm_positions = torch.cat(parts, dim=1).reshape(3, -1)
        if llm_positions.shape[-1] != len(input_tokens):
            raise ValueError(
                "Qwen35 M-RoPE position length mismatch: "
                f"positions={llm_positions.shape[-1]} tokens={len(input_tokens)}"
            )
        position_ids[:, batch_idx, :] = llm_positions.to(position_ids.device)
        deltas.append(int(llm_positions.max().item()) + 1 - len(input_tokens))

    delta_tensor = torch.tensor(deltas, device=ids_2d.device, dtype=torch.long).unsqueeze(1)
    return position_ids.squeeze(1), delta_tensor


def build_sglang_thinker_request(
    state: Qwen3OmniPipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    request_id: str | None = None,
    thinker_config: Any = None,
) -> SGLangARRequestData:
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    prompt = state.prompt
    input_ids = prompt["input_ids"]
    original_input_ids = input_ids
    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v
            for k, v in thinker_inputs.items()
            if k not in ("capture_model_output_keys", "media_cache_keys")
        }
    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    media_cache_keys = thinker_inputs.get("media_cache_keys", {})
    pad_values: dict[str, int] = {}
    if media_cache_keys and thinker_config is not None:
        token_id_map: dict[int, int] = {}
        for modality, orig_token_id in [
            ("image", thinker_config.image_token_id),
            ("video", thinker_config.video_token_id),
            ("audio", thinker_config.audio_token_id),
        ]:
            cache_key = media_cache_keys.get(modality)
            if cache_key is None:
                continue
            h = xxhash.xxh3_64(cache_key.encode()).intdigest()
            pad_val = vocab_size + h % (1 << 62)
            pad_values[modality] = pad_val
            token_id_map[int(orig_token_id)] = pad_val
        if token_id_map:
            input_ids = input_ids.clone()
            for orig_id, pad_val in token_id_map.items():
                input_ids[input_ids == orig_id] = pad_val
        if pad_values:
            model_inputs["pad_values"] = pad_values
    model_inputs.pop("attention_mask", None)
    input_ids_list = input_ids.to(dtype=torch.long).tolist()

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)
    top_p = params.get("top_p", 1.0)
    top_k = params.get("top_k", -1)
    min_p = params.get("min_p", 0.0)
    repetition_penalty = params.get("repetition_penalty", 1.0)
    stop = params.get("stop") or []
    stop_token_ids = params.get("stop_token_ids") or []
    seed = qwen3_builders._resolve_seed(params)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop=stop,
        stop_token_ids=stop_token_ids,
        sampling_seed=seed,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=request_id or "req-0",
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer

    if thinker_config is not None and model_inputs:
        mrope_result = _compute_mrope_positions(
            original_input_ids.to(dtype=torch.long), model_inputs, thinker_config
        )
        if mrope_result is not None:
            mrope_positions, mrope_position_delta = mrope_result
            mm_inputs = MultimodalInputs(mm_items=[])
            mm_inputs.mrope_positions = mrope_positions
            mm_inputs.mrope_position_delta = mrope_position_delta
            req.multimodal_inputs = mm_inputs

    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None
    req._codec_suppress_tokens = None

    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    data.return_logprob = bool(params.get("return_logprob"))
    return data


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    thinker_config: Any = None,
    stage_name: str = "thinker",
):
    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        params = payload.request.params or {}
        req_data = build_sglang_thinker_request(
            state,
            params=params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
            thinker_config=thinker_config,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        apply_thinker_result(state, stage_name=stage_name, result=data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter
