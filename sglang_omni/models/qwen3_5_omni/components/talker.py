# SPDX-License-Identifier: Apache-2.0
"""SGLang talker wrapper for Qwen3.5-Omni.

This module ports the Qwen3.5 talker far enough to satisfy sglang-omni's
existing speech runtime contract: the main AR model runs on SGLang's
Qwen3NextForCausalLM, while feedback/code buffers expose the same surface used
by QwenTalkerModelRunner.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import add_prefix

from sglang_omni.models.qwen3_5_omni.components.subtalker import (
    Qwen35ResidualCodePredictor,
    _sample_logits,
)
from sglang_omni.models.qwen3_omni.quantization import (
    convert_fp8_weight_scale_inv_for_sglang,
)
from sglang_omni.models.qwen3_5_omni.components.common import (
    ensure_sglang_qwen3_next_text_config,
    sub_config_or_self,
)
from sglang_omni.vendor.sglang.layers import ReplicatedLinear
from sglang_omni.vendor.sglang.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_SKIP_PREFIXES = ("thinker.", "code2wav.", "mtp.")
_IGNORED_SUFFIXES = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    ".weight_scale",
    "_weight_scale",
    ".input_scale",
    "_input_scale",
)


class _LinearProjection(nn.Module):
    """ReplicatedLinear wrapper with a plain tensor-returning forward."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        quant_config: Any | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.linear = ReplicatedLinear(
            in_size,
            out_size,
            bias=True,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.linear(x)
        return out


class _IdentityProjection(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _get_int(config: Any, name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _resolve_talker_quant_config(
    quant_config: Any | None,
    text_config: Any,
) -> Any | None:
    if quant_config is None:
        return None
    if getattr(text_config, "quantization_config", None) or getattr(
        text_config,
        "compression_config",
        None,
    ):
        return quant_config

    # 中文说明：Qwen3.5 root checkpoint 可能带 root-level quant_config，
    # 但 talker 子模型未必量化。vLLM perf_v2 会只在 talker text_config
    # 自己声明量化/压缩时保留 quant_config；这里保持一致，避免 root
    # quant 误套到 talker 权重映射后的子模型上。
    return None


def _max_running_requests() -> int:
    try:
        return int(get_global_server_args().max_running_requests)
    except Exception:
        return 1


def _bind_default_weight_loaders(module: nn.Module) -> None:
    for param in module.parameters():
        if not hasattr(param, "weight_loader"):
            param.weight_loader = default_weight_loader


def _should_suppress_codec_eos_for_decode(data: Any) -> bool:
    pending_text_queue = getattr(data, "pending_text_queue", None)
    if pending_text_queue:
        return True
    return not bool(getattr(data, "thinker_chunks_done", False))


def _direct_weight_loader(param: nn.Parameter):
    return getattr(param, "weight_loader", default_weight_loader)


def _normalize_speaker_codec_embeddings(
    loaded_weight: torch.Tensor,
    *,
    num_code_groups: int,
) -> torch.Tensor:
    if loaded_weight.ndim != 3:
        raise ValueError(
            "Qwen3.5 speaker_codec_embeddings must be a 3D tensor, got "
            f"shape={tuple(loaded_weight.shape)}"
        )

    if num_code_groups <= 0:
        return loaded_weight

    if loaded_weight.shape[2] == num_code_groups:
        return loaded_weight
    if loaded_weight.shape[1] == num_code_groups:
        # 中文说明：vLLM perf_v2 的原始 checkpoint 存的是 [S,K,T]，
        # engine 加载时转成运行时使用的 [S,T,K]。这里同步这个约定，
        # 避免 speaker prompt 的时间维和 codec group 维被反着解释。
        return loaded_weight.transpose(1, 2).contiguous()

    raise ValueError(
        "Qwen3.5 speaker_codec_embeddings last or middle dimension must match "
        f"num_code_groups={num_code_groups}, got shape={tuple(loaded_weight.shape)}"
    )


def _normalize_weight_name(name: str) -> str | None:
    """Map HF/vLLM Qwen3.5-Omni talker names to this wrapper's names."""
    if name.startswith(_SKIP_PREFIXES):
        return None
    if name.startswith("talker."):
        name = name[len("talker.") :]
    if name.endswith("rotary_emb.inv_freq"):
        # 中文说明：Qwen3.5 subtalker/Next 模型的 rotary inv_freq 是由
        # config 现场计算的 buffer。不同 HF/vLLM 版本可能把它暴露在
        # state_dict 里，但它不应该作为真实权重强制加载。
        return None

    if name.startswith("codec_head."):
        return "language_model.lm_head." + name[len("codec_head.") :]
    if name.startswith("model.codec_embedding."):
        return "language_model.model.embed_tokens." + name[
            len("model.codec_embedding.") :
        ]
    if name.startswith("model.embed_tokens."):
        return "text_embedding." + name[len("model.embed_tokens.") :]
    if name.startswith("model."):
        return "language_model.model." + name[len("model.") :]
    if name.startswith("hidden_projection."):
        return "hidden_projection.linear." + name[len("hidden_projection.") :]
    if name.startswith("text_projection."):
        return "text_projection.linear." + name[len("text_projection.") :]
    if name.startswith("speaker_codec_embeddings"):
        return name
    if name.startswith("code_predictor."):
        if name.startswith("code_predictor.model.talker_projection."):
            return "code_predictor.model.talker_projection.linear." + name[
                len("code_predictor.model.talker_projection.") :
            ]
        if name.startswith("code_predictor.model.context_projection."):
            return "code_predictor.model.context_projection.linear." + name[
                len("code_predictor.model.context_projection.") :
            ]
        if name.startswith("code_predictor.lm_head."):
            parts = name.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                rest = ".".join(parts[3:])
                return f"code_predictor.lm_head.{parts[2]}.linear.{rest}"
        return name
    return name


def _iter_language_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    language_params: dict[str, nn.Parameter],
) -> Iterator[Tuple[str, torch.Tensor]]:
    for name, loaded_weight in weights:
        mapped = _normalize_weight_name(name)
        if mapped is None or not mapped.startswith("language_model."):
            continue
        local_name = mapped[len("language_model.") :]
        if local_name.endswith(_IGNORED_SUFFIXES) and local_name not in language_params:
            continue
        loaded_weight = convert_fp8_weight_scale_inv_for_sglang(
            local_name,
            loaded_weight,
        )
        yield local_name, loaded_weight


class Qwen3OmniNextTalkerForConditionalGeneration(nn.Module):
    """Qwen3.5-Omni talker adapted to sglang-omni's Qwen talker runtime."""

    def __init__(
        self,
        config: Any,
        quant_config: Any | None = None,
        prefix: str = "",
        ) -> None:
        super().__init__()
        self.root_config = config
        self.config = sub_config_or_self(config, "talker_config")
        self.text_config = ensure_sglang_qwen3_next_text_config(
            getattr(self.config, "text_config", None) or self.config
        )
        self.code_predictor_config = getattr(
            self.config,
            "code_predictor_config",
            None,
        )
        if self.code_predictor_config is not None:
            self.code_predictor_config = ensure_sglang_qwen3_next_text_config(
                self.code_predictor_config
            )
        talker_quant_config = _resolve_talker_quant_config(
            quant_config,
            self.text_config,
        )

        self.language_model = Qwen3NextForCausalLM(
            self.text_config,
            quant_config=talker_quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.model = self.language_model.model
        self.codec_head = self.language_model.lm_head
        self.model.codec_embedding = self.model.embed_tokens

        text_vocab_size = _get_int(
            self.text_config,
            "text_vocab_size",
            _get_int(self.text_config, "vocab_size", 0),
        )
        self.text_embedding = VocabParallelEmbedding(
            text_vocab_size,
            self.text_config.hidden_size,
            prefix=add_prefix("text_embedding", prefix),
        )

        thinker_hidden_size = _get_int(
            self.config,
            "thinker_hidden_size",
            self.text_config.hidden_size,
        )
        # 中文说明：Qwen3.5 的文本侧已经有 talker.model.embed_tokens，
        # 不再像 Qwen3-Omni 那样用 thinker embedding 经过 text_projection。
        self.text_projection = _IdentityProjection()
        self.hidden_projection = _LinearProjection(
            thinker_hidden_size,
            self.text_config.hidden_size,
            quant_config=talker_quant_config,
            prefix=add_prefix("hidden_projection", prefix),
        )

        self.num_code_groups = _get_int(
            self.config,
            "num_code_groups",
            _get_int(self.code_predictor_config, "num_code_groups", 1),
        )
        self._subtalker_default_temperature = float(
            getattr(self.code_predictor_config, "temperature", 1.0)
        )
        self._subtalker_default_top_k = int(
            getattr(self.code_predictor_config, "top_k", 50) or 0
        )
        self._subtalker_default_top_p = float(
            getattr(self.code_predictor_config, "top_p", 1.0)
        )
        self._subtalker_default_min_p = float(
            getattr(self.code_predictor_config, "min_p", 0.0)
        )
        self.codec_pad_id = _get_int(self.config, "codec_pad_id", 0)
        self.codec_eos_token_id = getattr(self.config, "codec_eos_token_id", None)
        self.codec_vocab_size = _get_int(
            self.code_predictor_config,
            "vocab_size",
            _get_int(self.text_config, "vocab_size", 0),
        )
        self.speaker_codec_embeddings = nn.Parameter(
            torch.empty(0),
            requires_grad=False,
        )
        self.code_predictor = Qwen35ResidualCodePredictor(
            self.code_predictor_config
        )

        self._init_runtime_buffers()
        _bind_default_weight_loaders(self)
        self._cached_params_dict = dict(self.named_parameters())
        self._sampler = None

    def _init_runtime_buffers(self) -> None:
        weight = self.model.codec_embedding.weight
        device = weight.device
        dtype = weight.dtype
        hidden_size = self.text_config.hidden_size
        max_batch_size = _max_running_requests()

        self._feedback_buffer = torch.zeros(
            max_batch_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self._feedback_mask = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=device,
        )
        self._sampled_token_ids = torch.zeros(
            max_batch_size,
            dtype=torch.long,
            device=device,
        )
        self._output_codes = torch.full(
            (max_batch_size, self.num_code_groups),
            self.codec_pad_id,
            dtype=torch.long,
            device=device,
        )
        self._output_embeds = torch.zeros(
            max_batch_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        vocab_size = self.text_config.vocab_size
        self._repetition_mask = torch.zeros(
            max_batch_size,
            vocab_size,
            dtype=torch.bool,
            device=device,
        )
        self._suppress_mask = torch.zeros_like(self._repetition_mask)
        self._repetition_penalties = torch.ones(
            max_batch_size,
            1,
            dtype=dtype,
            device=device,
        )
        self._sampling_temperatures = torch.zeros(
            max_batch_size,
            1,
            dtype=dtype,
            device=device,
        )
        self._sampling_top_ps = torch.ones(
            max_batch_size,
            dtype=dtype,
            device=device,
        )
        self._sampling_top_ks = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device=device,
        )
        self._sampling_min_ps = torch.zeros(
            max_batch_size,
            dtype=dtype,
            device=device,
        )
        self._sampling_seeds = torch.zeros(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self._sampling_seed_enabled = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=device,
        )
        self._subtalker_temperature = torch.full(
            (max_batch_size,),
            self._subtalker_default_temperature,
            dtype=torch.float32,
            device=device,
        )
        self._subtalker_top_k = torch.full(
            (max_batch_size,),
            self._subtalker_default_top_k,
            dtype=torch.long,
            device=device,
        )
        self._subtalker_top_p = torch.full(
            (max_batch_size,),
            self._subtalker_default_top_p,
            dtype=torch.float32,
            device=device,
        )
        self._subtalker_min_p = torch.full(
            (max_batch_size,),
            self._subtalker_default_min_p,
            dtype=torch.float32,
            device=device,
        )
        self._subtalker_seeds = torch.zeros(
            max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self._subtalker_seed_enabled = torch.zeros(
            max_batch_size,
            dtype=torch.bool,
            device=device,
        )

    @property
    def activation_dtype(self) -> torch.dtype:
        return self.model.codec_embedding.weight.dtype

    def get_input_embeddings(self):
        return self.model.codec_embedding

    def embed_text_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_embedding(input_ids)

    def codec_code_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed codec groups and sum them into talker hidden rows.

        中文说明：vLLM Qwen3.5 的 speaker code 使用 [T, K] 或
        [B, K, T] 的多码本 codec id。第 0 组走主 talker codec embedding，
        后续 residual 组走 subtalker/code_predictor 的 codec embedding。
        """
        if codes.ndim == 2:
            code_rows = codes.to(device=self.activation_dtype_device, dtype=torch.long)
            if (
                code_rows.shape[0] == self.num_code_groups
                and code_rows.shape[1] != self.num_code_groups
            ):
                # 中文说明：voice clone 传入的 prompt_speaker_codes 常见布局是
                # [K,T]，而本函数内部统一按 [T,K] 处理。
                code_rows = code_rows.transpose(0, 1).contiguous()
            groups = code_rows.shape[1]
            parts = [self.model.codec_embedding(code_rows[:, 0:1])]
            for group_idx in range(1, min(groups, self.num_code_groups)):
                embed = self.code_predictor.model.codec_embedding[group_idx - 1]
                parts.append(embed(code_rows[:, group_idx : group_idx + 1]))
            return torch.cat(parts, dim=1).sum(1)

        if codes.ndim == 3:
            code_rows = codes.to(device=self.activation_dtype_device, dtype=torch.long)
            groups = code_rows.shape[1]
            parts = [self.model.codec_embedding(code_rows[:, 0:1])]
            for group_idx in range(1, min(groups, self.num_code_groups)):
                embed = self.code_predictor.model.codec_embedding[group_idx - 1]
                parts.append(embed(code_rows[:, group_idx : group_idx + 1]))
            return torch.cat(parts, dim=1).sum(1)

        raise ValueError("codec_code_embeddings expects [T,K] or [B,K,T] codes")

    @property
    def activation_dtype_device(self) -> torch.device:
        return self.model.codec_embedding.weight.device

    def speaker_codec_codes(self, speaker_id: int) -> torch.Tensor | None:
        speaker_codes = self.speaker_codec_embeddings
        if not isinstance(speaker_codes, torch.Tensor) or speaker_codes.numel() == 0:
            return None
        if speaker_codes.ndim != 3:
            raise ValueError(
                "Qwen3.5 speaker_codec_embeddings must have shape [S,T,K]"
            )
        speaker_id = int(speaker_id)
        if speaker_id < 0 or speaker_id >= speaker_codes.shape[0]:
            raise IndexError(
                f"speaker_id {speaker_id} out of range for "
                f"{speaker_codes.shape[0]} speakers"
            )
        codes = speaker_codes[speaker_id].detach().to(
            device=self.activation_dtype_device,
            dtype=torch.long,
        )
        valid = codes.ne(-1).any(dim=1)
        return codes[valid]

    def speaker_codec_input_embeddings(self, speaker_id: int) -> torch.Tensor | None:
        codes = self.speaker_codec_codes(speaker_id)
        if codes is None or codes.numel() == 0:
            return None
        return self.codec_code_embeddings(codes).to(dtype=self.activation_dtype)

    def prepare_decode_buffers(self, requests: list[Any]) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        self._repetition_mask[:batch_size] = False
        self._suppress_mask[:batch_size] = False
        self._repetition_penalties[:batch_size, 0] = 1.0
        self._sampling_temperatures[:batch_size, 0] = 0.0
        self._sampling_top_ps[:batch_size] = 1.0
        self._sampling_top_ks[:batch_size] = 0
        self._sampling_min_ps[:batch_size] = 0.0
        self._sampling_seeds[:batch_size] = 0
        self._sampling_seed_enabled[:batch_size] = False
        self._subtalker_temperature[:batch_size] = (
            self._subtalker_default_temperature
        )
        self._subtalker_top_k[:batch_size] = self._subtalker_default_top_k
        self._subtalker_top_p[:batch_size] = self._subtalker_default_top_p
        self._subtalker_min_p[:batch_size] = self._subtalker_default_min_p
        self._subtalker_seeds[:batch_size] = 0
        self._subtalker_seed_enabled[:batch_size] = False

        rep_rows: list[int] = []
        rep_toks: list[int] = []
        sup_rows: list[int] = []
        sup_toks: list[int] = []
        rep_vocab = self._repetition_mask.shape[1]
        sup_vocab = self._suppress_mask.shape[1]

        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            sp = req.sampling_params
            temperature = float(getattr(sp, "temperature", 0.0))
            top_k = int(getattr(sp, "top_k", 0) or 0)
            top_p = float(getattr(sp, "top_p", 1.0))
            min_p = float(getattr(sp, "min_p", 0.0))
            seed = getattr(sp, "sampling_seed", None)
            self._sampling_temperatures[row_idx, 0] = temperature
            self._sampling_top_ks[row_idx] = top_k
            self._sampling_top_ps[row_idx] = top_p
            self._sampling_min_ps[row_idx] = min_p
            self._sampling_seeds[row_idx] = int(seed) if seed is not None else 0
            self._sampling_seed_enabled[row_idx] = seed is not None
            sub_sp = getattr(req, "_qwen35_subtalker_sampling_params", None)
            if sub_sp is None:
                sub_sp = getattr(data, "subtalker_sampling_params", None)
            # 中文说明：Qwen3.5 的 residual codec predictor 有自己的
            # generation defaults（vLLM CodePredictor 默认
            # temperature=0.9/top_k=50）。未显式传 subtalker_params 时，
            # 必须使用这些默认值；若跟主 talker 的 greedy 参数绑定，会让
            # residual codec 坍缩，code2wav 输出不可懂的噪音。
            if sub_sp is not None:
                self._subtalker_temperature[row_idx] = float(
                    getattr(
                        sub_sp,
                        "temperature",
                        self._subtalker_default_temperature,
                    )
                )
                self._subtalker_top_k[row_idx] = int(
                    getattr(sub_sp, "top_k", self._subtalker_default_top_k) or 0
                )
                self._subtalker_top_p[row_idx] = float(
                    getattr(sub_sp, "top_p", self._subtalker_default_top_p)
                )
                self._subtalker_min_p[row_idx] = float(
                    getattr(sub_sp, "min_p", self._subtalker_default_min_p)
                )
                sub_seed = getattr(sub_sp, "sampling_seed", None)
                if sub_seed is None:
                    sub_seed = getattr(sub_sp, "seed", None)
                if sub_seed is not None:
                    self._subtalker_seeds[row_idx] = int(sub_seed)
                    self._subtalker_seed_enabled[row_idx] = True
            penalty = float(getattr(sp, "repetition_penalty", 1.0))
            self._repetition_penalties[row_idx, 0] = penalty
            if penalty != 1.0 and req.output_ids:
                unique = {
                    int(tok)
                    for tok in req.output_ids
                    if 0 <= int(tok) < rep_vocab
                }
                rep_rows.extend([row_idx] * len(unique))
                rep_toks.extend(unique)

            suppress_tokens = data.suppress_tokens or req._codec_suppress_tokens
            if _should_suppress_codec_eos_for_decode(data):
                codec_eos = self.codec_eos_token_id
                if codec_eos is not None:
                    suppress_tokens = [
                        *(suppress_tokens or ()),
                        int(codec_eos),
                    ]
            if suppress_tokens:
                valid = [
                    int(tok)
                    for tok in suppress_tokens
                    if 0 <= int(tok) < sup_vocab
                ]
                sup_rows.extend([row_idx] * len(valid))
                sup_toks.extend(valid)

        device = self._repetition_mask.device
        if rep_rows:
            self._repetition_mask[
                torch.tensor(rep_rows, dtype=torch.long, device=device),
                torch.tensor(rep_toks, dtype=torch.long, device=device),
            ] = True
        if sup_rows:
            self._suppress_mask[
                torch.tensor(sup_rows, dtype=torch.long, device=device),
                torch.tensor(sup_toks, dtype=torch.long, device=device),
            ] = True

    def prepare_input_embeds(
        self,
        thinker_embeds: torch.Tensor | None = None,
        thinker_hidden_states: torch.Tensor | None = None,
        is_multimodal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if thinker_hidden_states is None or is_multimodal_mask is None:
            return self.text_projection(thinker_embeds)
        if thinker_embeds is None:
            return self.hidden_projection(thinker_hidden_states)

        output = torch.empty(
            (*thinker_embeds.shape[:-1], self.text_config.hidden_size),
            device=thinker_embeds.device,
            dtype=thinker_embeds.dtype,
        )
        if is_multimodal_mask.any():
            output[is_multimodal_mask] = self.hidden_projection(
                thinker_hidden_states[is_multimodal_mask]
            )
        text_mask = ~is_multimodal_mask
        if text_mask.any():
            output[text_mask] = self.text_projection(thinker_embeds[text_mask])
        return output

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
        input_deepstack_mask: torch.Tensor | None = None,
        input_embeds_are_projected: bool = False,
    ):
        if input_embeds is not None and not input_embeds_are_projected:
            input_embeds = self.prepare_input_embeds(
                thinker_embeds=input_embeds,
                thinker_hidden_states=input_deepstack_embeds,
                is_multimodal_mask=input_deepstack_mask,
            )
        elif input_embeds is None:
            batch_size = int(input_ids.shape[0])
            feedback_mask = self._feedback_mask[:batch_size]
            if bool(feedback_mask.any().item()):
                codec_embeds = self.model.codec_embedding(input_ids)
                input_embeds = torch.where(
                    feedback_mask.unsqueeze(-1),
                    self._feedback_buffer[:batch_size].to(codec_embeds.dtype),
                    codec_embeds,
                )
                self._feedback_mask[:batch_size] = False

        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
        )
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states

        if forward_batch.forward_mode.is_extend() and input_embeds is not None:
            return self._manual_extend_logits(hidden_states, forward_batch)

        logits_output = self._manual_decode_logits(hidden_states)
        if forward_batch.forward_mode.is_decode():
            sampled_token_ids = self._sample_decode_tokens(
                logits_output.next_token_logits,
                forward_batch,
            )
            batch_size = sampled_token_ids.shape[0]
            self._sampled_token_ids[:batch_size].copy_(sampled_token_ids)
            self.code_predictor_forward(
                sampled_token_ids.unsqueeze(1),
                hidden_states.unsqueeze(1),
            )
        return logits_output

    def _manual_extend_logits(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> LogitsProcessorOutput:
        last_index = self._extend_last_index(forward_batch, hidden_states.device)
        pruned_states = hidden_states[last_index]
        next_token_logits = self.compute_logits(pruned_states)
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=pruned_states,
        )

    def _manual_decode_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> LogitsProcessorOutput:
        next_token_logits = self.compute_logits(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=hidden_states,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        head_weight = getattr(self.codec_head, "weight", None)
        if head_weight is not None:
            logits = torch.matmul(
                hidden_states.to(head_weight.dtype),
                head_weight.T,
            )
        else:
            logits_output = self.codec_head(hidden_states)
            logits = (
                logits_output[0]
                if isinstance(logits_output, tuple)
                else logits_output
            )
        return self._mask_invalid_codec_logits(logits)

    def _mask_invalid_codec_logits(self, logits: torch.Tensor) -> torch.Tensor:
        codec_eos = self.codec_eos_token_id
        if self.codec_vocab_size <= 0 or self.codec_vocab_size >= logits.shape[-1]:
            return logits
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[: self.codec_vocab_size] = False
        if codec_eos is not None and 0 <= int(codec_eos) < logits.shape[-1]:
            mask[int(codec_eos)] = False
        return logits.masked_fill(mask, -1e8)

    def _sample_decode_tokens(
        self,
        logits: torch.Tensor,
        forward_batch: Any | None = None,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        logits = logits.clone()
        penalties = self._repetition_penalties[:batch_size].to(dtype=logits.dtype)
        penalized = torch.where(logits > 0, logits / penalties, logits * penalties)
        logits = torch.where(self._repetition_mask[:batch_size], penalized, logits)
        logits = logits.masked_fill(self._suppress_mask[:batch_size], float("-inf"))

        logits_output = LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=None,
        )
        if self._sampler is not None and forward_batch is not None:
            sampled = self._sampler(
                logits_output,
                self._build_static_sampling_info(batch_size),
                False,
                [0] * batch_size,
                [[] for _ in range(batch_size)],
                forward_batch.positions,
            )
            if sampled.ndim > 1:
                sampled = sampled.squeeze(-1)
            return sampled

        # 中文说明：没有 SGLang sampler 的单测/降级场景仍按请求采样参数执行，
        # 真实服务里 bootstrap 会把 model_runner.sampler 注入到 self._sampler。
        return _sample_logits(
            logits,
            temperature=self._sampling_temperatures[:batch_size, 0],
            top_k=self._sampling_top_ks[:batch_size],
            top_p=self._sampling_top_ps[:batch_size],
            min_p=self._sampling_min_ps[:batch_size],
        )

    def _build_static_sampling_info(self, batch_size: int) -> SamplingBatchInfo:
        temperatures = self._sampling_temperatures[:batch_size]
        top_ps = self._sampling_top_ps[:batch_size]
        top_ks = self._sampling_top_ks[:batch_size]
        min_ps = self._sampling_min_ps[:batch_size]
        # 中文说明：这些 flag 会影响 SGLang sampler 是否进入对应采样分支。
        # 按当前 batch 动态设置，既保证 min_p 真正生效，也避免贪心请求走多余路径。
        is_all_greedy = bool((temperatures <= 0).all().item())
        need_top_p_sampling = bool((top_ps < 1.0).any().item())
        need_top_k_sampling = bool((top_ks > 0).any().item())
        need_min_p_sampling = bool((min_ps > 0.0).any().item())
        seed_enabled = self._sampling_seed_enabled[:batch_size]
        return SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=is_all_greedy,
            need_top_p_sampling=need_top_p_sampling,
            need_top_k_sampling=need_top_k_sampling,
            need_min_p_sampling=need_min_p_sampling,
            vocab_size=self.text_config.vocab_size,
            grammars=[],
            vocab_mask=None,
            apply_mask_func=None,
            penalizer_orchestrator=None,
            acc_linear_penalties=None,
            has_custom_logit_processor=False,
            custom_params=None,
            custom_logit_processor=None,
            sampling_seed=(
                self._sampling_seeds[:batch_size]
                if bool(seed_enabled.any().item())
                else None
            ),
            device=self._sampling_temperatures.device.type,
            logit_bias=None,
        )

    @staticmethod
    def _extend_last_index(forward_batch: Any, device: torch.device) -> torch.Tensor:
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)
        padded_static_len = getattr(forward_batch, "padded_static_len", None)
        if padded_static_len is not None and padded_static_len >= 0:
            idx = torch.arange(
                len(extend_seq_lens),
                device=device,
                dtype=extend_seq_lens.dtype,
            )
            return idx * padded_static_len + extend_seq_lens.to(device=device) - 1
        seq_lens = extend_seq_lens.to(device=device)
        return torch.cumsum(seq_lens, dim=0) - 1

    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
        requests: list[Any] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a code chunk compatible with code2wav."""
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        if talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)

        temperature, top_k, top_p, min_p, seed = self._resolve_subtalker_sampling(
            requests=requests,
            batch_size=layer0_codes.shape[0],
        )
        codes, embeds = self.code_predictor.generate(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
            layer0_embed_fn=self.get_input_embeddings(),
            pad_id=self.codec_pad_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
        )
        batch_size, seq_len = layer0_codes.shape
        if seq_len == 1:
            self._output_codes[:batch_size].copy_(codes[:, :, 0])
            self._output_embeds[:batch_size].copy_(embeds[:, 0, :])
        return codes, embeds

    def _resolve_subtalker_sampling(
        self,
        *,
        requests: list[Any] | None,
        batch_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if requests is None:
            seed_enabled = self._subtalker_seed_enabled[:batch_size]
            seed = torch.where(
                seed_enabled,
                self._subtalker_seeds[:batch_size],
                torch.full_like(self._subtalker_seeds[:batch_size], -1),
            )
            return (
                self._subtalker_temperature[:batch_size],
                self._subtalker_top_k[:batch_size],
                self._subtalker_top_p[:batch_size],
                self._subtalker_min_p[:batch_size],
                seed if bool(seed_enabled.any().item()) else None,
            )

        temperature = torch.full(
            (batch_size,),
            self._subtalker_default_temperature,
            dtype=torch.float32,
            device=self._subtalker_temperature.device,
        )
        top_k = torch.full(
            (batch_size,),
            self._subtalker_default_top_k,
            dtype=torch.long,
            device=self._subtalker_top_k.device,
        )
        top_p = torch.full(
            (batch_size,),
            self._subtalker_default_top_p,
            dtype=torch.float32,
            device=self._subtalker_top_p.device,
        )
        min_p = torch.full(
            (batch_size,),
            self._subtalker_default_min_p,
            dtype=torch.float32,
            device=self._subtalker_min_p.device,
        )
        seed = torch.full(
            (batch_size,),
            -1,
            dtype=torch.int64,
            device=self._subtalker_seeds.device,
        )
        seed_enabled = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self._subtalker_seed_enabled.device,
        )
        for row_idx, sched_req in enumerate(requests[:batch_size]):
            data = sched_req.data
            req = data.req
            sp = getattr(req, "_qwen35_subtalker_sampling_params", None)
            if sp is None:
                sp = getattr(data, "subtalker_sampling_params", None)
            if sp is None:
                continue
            temperature[row_idx] = float(
                getattr(sp, "temperature", self._subtalker_default_temperature)
            )
            top_k[row_idx] = int(
                getattr(sp, "top_k", self._subtalker_default_top_k) or 0
            )
            top_p[row_idx] = float(getattr(sp, "top_p", self._subtalker_default_top_p))
            min_p[row_idx] = float(
                getattr(sp, "min_p", self._subtalker_default_min_p)
            )
            sp_seed = getattr(sp, "sampling_seed", None)
            if sp_seed is None:
                sp_seed = getattr(sp, "seed", None)
            if sp_seed is not None:
                seed[row_idx] = int(sp_seed)
                seed_enabled[row_idx] = True
        return temperature, top_k, top_p, min_p, (
            seed if bool(seed_enabled.any().item()) else None
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        language_params = dict(self.language_model.named_parameters())
        direct_params = dict(self.named_parameters())
        direct_loaded: set[str] = set()

        def _mapped_language_weights() -> Iterator[Tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                mapped = _normalize_weight_name(name)
                if mapped is None:
                    continue
                if mapped.startswith("language_model."):
                    local_name = mapped[len("language_model.") :]
                    if (
                        local_name.endswith(_IGNORED_SUFFIXES)
                        and local_name not in language_params
                    ):
                        continue
                    loaded_weight = convert_fp8_weight_scale_inv_for_sglang(
                        local_name,
                        loaded_weight,
                    )
                    yield local_name, loaded_weight
                elif self._load_direct_weight(
                    mapped,
                    loaded_weight,
                    direct_params,
                ):
                    direct_loaded.add(mapped)

        loaded = {
            f"language_model.{name}"
            for name in self.language_model.load_weights(
                _mapped_language_weights()
            )
        }
        loaded.update(direct_loaded)
        return loaded

    def _load_direct_weight(
        self,
        mapped: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        if mapped == "speaker_codec_embeddings":
            loaded_weight = _normalize_speaker_codec_embeddings(
                loaded_weight,
                num_code_groups=self.num_code_groups,
            )
            self.speaker_codec_embeddings = nn.Parameter(
                loaded_weight.detach(),
                requires_grad=False,
            )
            params_dict[mapped] = self.speaker_codec_embeddings
            return True

        param = params_dict.get(mapped)
        if param is None:
            return False
        loaded_weight = convert_fp8_weight_scale_inv_for_sglang(
            mapped,
            loaded_weight,
        )
        weight_loader = _direct_weight_loader(param)
        weight_loader(param, loaded_weight)
        return True


Qwen3OmniNextTalkerModel = Qwen3OmniNextTalkerForConditionalGeneration
Qwen3OmniNextMoeTalkerForConditionalGeneration = (
    Qwen3OmniNextTalkerForConditionalGeneration
)
EntryClass = Qwen3OmniNextTalkerForConditionalGeneration
