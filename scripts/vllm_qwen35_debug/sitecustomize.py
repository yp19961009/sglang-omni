# SPDX-License-Identifier: Apache-2.0
"""Debug-only vLLM Qwen3.5 hidden-state hook.

Put this directory before vLLM on PYTHONPATH and set VLLM_QWEN35_HIDDEN_DUMP
to make spawned vLLM workers dump Qwen3NextModel prefill hidden stats.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
from pathlib import Path
from typing import Any


_TARGET = "vllm.model_executor.models.qwen3_next"
_VL_TARGET = "vllm.model_executor.models.qwen3_omni_next_thinker"
_CODE_PREDICTOR_TARGET = "vllm.v1.spec_decode.code_predictor"
_ORIGINAL_IMPORT = builtins.__import__


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _tensor_stats(torch: Any, tensor: Any, sample_size: int = 8) -> dict[str, Any] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    data = tensor.detach()
    stats_data = data.float()
    last = stats_data.reshape(-1, stats_data.shape[-1])[-1]
    stats = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "mean": float(stats_data.mean().cpu()),
        "std": float(stats_data.std(unbiased=False).cpu()),
        "norm": float(torch.linalg.vector_norm(stats_data).cpu()),
        "last_mean": float(last.mean().cpu()),
        "last_std": float(last.std(unbiased=False).cpu()),
        "last_norm": float(torch.linalg.vector_norm(last).cpu()),
        "last_first_values": [
            float(x) for x in last[:sample_size].detach().cpu().tolist()
        ],
    }
    if os.getenv("VLLM_QWEN35_HIDDEN_DUMP_LAST_VALUES"):
        stats["last_values"] = [float(x) for x in last.detach().cpu().tolist()]
    if (
        os.getenv("VLLM_QWEN35_HIDDEN_DUMP_TOKEN_NORMS")
        or os.getenv("VLLM_QWEN35_AUDIO_DUMP_TOKEN_NORMS")
    ):
        flat = stats_data.reshape(-1, stats_data.shape[-1])
        stats["token_norms"] = [
            float(x) for x in torch.linalg.vector_norm(flat, dim=-1).cpu().tolist()
        ]
    return stats


def _tensor_values(torch: Any, tensor: Any, max_items: int = 512) -> Any | None:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() > max_items:
        return None
    return tensor.detach().cpu().tolist()


def _audio_feature_lengths(torch: Any, mask: Any, device: Any) -> Any | None:
    if isinstance(mask, torch.Tensor):
        return [int(x) for x in mask.sum(-1).detach().cpu().tolist()]
    if isinstance(mask, list):
        lengths = []
        for item in mask:
            if isinstance(item, torch.Tensor):
                lengths.append(int(item.sum(-1).detach().cpu().item()))
        return lengths
    return None


def _audio_input_get(value: Any, key: str) -> Any | None:
    try:
        return value[key]
    except Exception:
        return getattr(value, key, None)


def _concat_tensor_list(torch: Any, value: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, (list, tuple)):
        return None
    tensors = [item for item in value if isinstance(item, torch.Tensor)]
    if not tensors:
        return None
    try:
        return torch.cat(tensors, dim=0)
    except RuntimeError:
        return None


def _audio_layer_filter() -> set[int] | None:
    value = os.getenv("VLLM_QWEN35_AUDIO_DUMP_LAYERS", "").strip()
    if not value:
        return set()
    if value.lower() == "all":
        return None
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            selected.add(int(item))
        except ValueError:
            continue
    return selected


def _audio_layer_selected(selected: set[int] | None, layer_id: int) -> bool:
    return selected is None or layer_id in selected


def _patch_qwen35_audio_encoder_layers(module: Any) -> None:
    dump_path = os.getenv("VLLM_QWEN35_AUDIO_DUMP")
    if not dump_path or not os.getenv("VLLM_QWEN35_AUDIO_DUMP_LAYERS"):
        return
    cls = getattr(module, "Qwen3OmniNextAudioEncoder", None)
    if cls is None or getattr(cls, "_qwen35_audio_layer_dump_installed", False):
        return

    try:
        max_calls = int(os.getenv("VLLM_QWEN35_AUDIO_DUMP_MAX_CALLS", "1"))
    except ValueError:
        max_calls = 1

    torch = module.torch
    original_forward = cls.forward

    def _ensure_layer_wrappers(self: Any) -> None:
        if getattr(self, "_qwen35_audio_layers_wrapped", False):
            return
        for layer_id, layer in enumerate(getattr(self, "layers", [])):
            original_layer_forward = layer.forward

            def _layer_forward(
                hidden_states: Any,
                cu_seqlens: Any,
                max_seqlen: Any = None,
                *,
                _original_layer_forward: Any = original_layer_forward,
                _layer_id: int = layer_id,
                _encoder: Any = self,
            ) -> Any:
                active = bool(getattr(_encoder, "_qwen35_audio_layer_dump_active", False))
                selected = getattr(_encoder, "_qwen35_audio_layer_dump_filter", set())
                path = getattr(_encoder, "_qwen35_audio_layer_dump_path", "")
                if (
                    active
                    and path
                    and _layer_id == 0
                    and not getattr(_encoder, "_qwen35_audio_layer_input_dumped", False)
                ):
                    _encoder._qwen35_audio_layer_input_dumped = True
                    _append_jsonl(
                        path,
                        {
                            "source": "vllm-sitecustomize",
                            "pid": os.getpid(),
                            "stage": "audio_layer_input",
                            "layer_id": 0,
                            "hidden": _tensor_stats(torch, hidden_states),
                            "cu_seqlens": _tensor_values(torch, cu_seqlens),
                        },
                    )
                output = _original_layer_forward(hidden_states, cu_seqlens, max_seqlen)
                if active and path and _audio_layer_selected(selected, _layer_id):
                    _append_jsonl(
                        path,
                        {
                            "source": "vllm-sitecustomize",
                            "pid": os.getpid(),
                            "stage": "audio_layer",
                            "layer_id": int(_layer_id),
                            "hidden": _tensor_stats(torch, output),
                        },
                    )
                return output

            layer.forward = _layer_forward
        self._qwen35_audio_layers_wrapped = True

    def _forward_with_audio_layers(
        self: Any,
        input_features: Any,
        feature_lens: Any,
        aftercnn_lens: Any,
    ) -> Any:
        _ensure_layer_wrappers(self)
        dumped_calls = int(getattr(self, "_qwen35_audio_layer_dump_calls", 0))
        should_dump = dumped_calls < max_calls
        if should_dump:
            self._qwen35_audio_layer_dump_calls = dumped_calls + 1
            self._qwen35_audio_layer_dump_active = True
            self._qwen35_audio_layer_dump_path = dump_path
            self._qwen35_audio_layer_dump_filter = _audio_layer_filter()
            self._qwen35_audio_layer_input_dumped = False
            _append_jsonl(
                dump_path,
                {
                    "source": "vllm-sitecustomize",
                    "pid": os.getpid(),
                    "stage": "audio_tower_input",
                    "input_features": _tensor_stats(torch, input_features),
                    "feature_lens": _tensor_values(torch, feature_lens),
                    "aftercnn_lens": _tensor_values(torch, aftercnn_lens),
                },
            )
        try:
            output = original_forward(self, input_features, feature_lens, aftercnn_lens)
        finally:
            if should_dump:
                self._qwen35_audio_layer_dump_active = False
        if should_dump:
            _append_jsonl(
                dump_path,
                {
                    "source": "vllm-sitecustomize",
                    "pid": os.getpid(),
                    "stage": "audio_tower_output",
                    "hidden": _tensor_stats(torch, output),
                },
            )
        return output

    cls.forward = _forward_with_audio_layers
    cls._qwen35_audio_layer_dump_installed = True
    _append_jsonl(
        dump_path,
        {
            "source": "vllm-sitecustomize",
            "pid": os.getpid(),
            "stage": "audio_layer_hook_installed",
        },
    )


def _patch_qwen35_audio_process(module: Any) -> None:
    dump_path = os.getenv("VLLM_QWEN35_AUDIO_DUMP")
    if not dump_path:
        return
    cls = getattr(module, "Qwen3OmniNextConditionalGenerationMixin", None)
    if cls is None or getattr(cls, "_qwen35_audio_dump_installed", False):
        return

    try:
        max_calls = int(os.getenv("VLLM_QWEN35_AUDIO_DUMP_MAX_CALLS", "1"))
    except ValueError:
        max_calls = 1

    torch = module.torch
    original_process = cls._process_audio_input

    def _process_audio_input_with_dump(
        self: Any,
        audio_input: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        dumped_calls = int(getattr(self, "_qwen35_audio_dump_calls", 0))
        should_dump = dumped_calls < max_calls
        if should_dump:
            self._qwen35_audio_dump_calls = dumped_calls + 1

        input_features = _audio_input_get(audio_input, "input_features")
        feature_attention_mask = _audio_input_get(audio_input, "feature_attention_mask")

        feature_lengths = _audio_feature_lengths(
            torch,
            feature_attention_mask,
            getattr(input_features, "device", None),
        )
        if should_dump:
            _append_jsonl(
                dump_path,
                {
                    "source": "vllm-sitecustomize",
                    "pid": os.getpid(),
                    "stage": "audio_input",
                    "input_features": _tensor_stats(torch, input_features),
                    "feature_attention_mask_shape": (
                        list(feature_attention_mask.shape)
                        if isinstance(feature_attention_mask, torch.Tensor)
                        else None
                    ),
                    "feature_lengths": feature_lengths,
                },
            )

        outputs = original_process(self, audio_input, *args, **kwargs)
        if should_dump:
            audio_features = _concat_tensor_list(torch, outputs)
            split_shapes = None
            if isinstance(outputs, (list, tuple)):
                split_shapes = [
                    list(item.shape) if isinstance(item, torch.Tensor) else None
                    for item in outputs
                ]
            _append_jsonl(
                dump_path,
                {
                    "source": "vllm-sitecustomize",
                    "pid": os.getpid(),
                    "stage": "audio_output",
                    "audio_features": _tensor_stats(torch, audio_features),
                    "split_shapes": split_shapes,
                },
            )
        return outputs

    cls._process_audio_input = _process_audio_input_with_dump
    cls._qwen35_audio_dump_installed = True
    _append_jsonl(
        dump_path,
        {
            "source": "vllm-sitecustomize",
            "pid": os.getpid(),
            "stage": "audio_hook_installed",
        },
    )


def _patch_qwen3_next(module: Any) -> None:
    dump_path = os.getenv("VLLM_QWEN35_HIDDEN_DUMP")
    if not dump_path:
        return
    cls = getattr(module, "Qwen3NextModel", None)
    if cls is None or getattr(cls, "_qwen35_hidden_dump_installed", False):
        return

    try:
        expected_tokens = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_EXPECT_TOKENS", "0"))
    except ValueError:
        expected_tokens = 0
    try:
        max_calls = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_MAX_CALLS", "1"))
    except ValueError:
        max_calls = 1
    try:
        max_skips = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_MAX_SKIPS", "20"))
    except ValueError:
        max_skips = 20

    torch = module.torch
    original_forward = cls.forward

    def _forward_with_hidden_dump(
        self: Any,
        input_ids: Any,
        positions: Any,
        intermediate_tensors: Any | None = None,
        inputs_embeds: Any | None = None,
    ) -> Any:
        dumped_calls = int(getattr(self, "_qwen35_hidden_dump_calls", 0))
        token_count = None
        if isinstance(inputs_embeds, torch.Tensor):
            token_count = int(inputs_embeds.shape[0])
        elif isinstance(input_ids, torch.Tensor):
            token_count = int(input_ids.shape[0])
        should_dump = dumped_calls < max_calls and (
            not expected_tokens or token_count == expected_tokens
        )
        if not should_dump:
            if os.getenv("VLLM_QWEN35_HIDDEN_DUMP_SKIPS"):
                skipped_calls = int(getattr(self, "_qwen35_hidden_dump_skip_calls", 0))
                if skipped_calls < max_skips:
                    self._qwen35_hidden_dump_skip_calls = skipped_calls + 1
                    _append_jsonl(
                        dump_path,
                        {
                            "source": "vllm-sitecustomize",
                            "pid": os.getpid(),
                            "stage": "skip",
                            "token_count": token_count,
                            "input_ids_shape": (
                                list(input_ids.shape)
                                if isinstance(input_ids, torch.Tensor)
                                else None
                            ),
                            "positions_shape": (
                                list(positions.shape)
                                if isinstance(positions, torch.Tensor)
                                else None
                            ),
                            "inputs_embeds_shape": (
                                list(inputs_embeds.shape)
                                if isinstance(inputs_embeds, torch.Tensor)
                                else None
                            ),
                        },
                    )
            return original_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        self._qwen35_hidden_dump_calls = dumped_calls + 1
        get_pp_group = module.get_pp_group
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        base_record = {
            "source": "vllm-sitecustomize",
            "pid": os.getpid(),
            "input_ids_shape": (
                list(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None
            ),
            "positions_shape": (
                list(positions.shape) if isinstance(positions, torch.Tensor) else None
            ),
            "inputs_embeds_shape": (
                list(inputs_embeds.shape)
                if isinstance(inputs_embeds, torch.Tensor)
                else None
            ),
            "input_ids": _tensor_values(torch, input_ids),
            "positions": _tensor_values(torch, positions),
        }
        _append_jsonl(
            dump_path,
            {
                **base_record,
                "stage": "embed",
                "hidden": _tensor_stats(torch, hidden_states),
            },
        )

        for layer in module.islice(self.layers, self.start_layer, self.end_layer):
            layer_id = getattr(layer, "layer_idx", None)
            if layer_id is None:
                layer_id = module.extract_layer_index(getattr(layer, "prefix", ""))
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            record = {
                **base_record,
                "stage": "layer",
                "layer_id": int(layer_id),
                "hidden": _tensor_stats(torch, hidden_states),
            }
            if isinstance(residual, torch.Tensor):
                record["residual"] = _tensor_stats(torch, residual)
            _append_jsonl(dump_path, record)

        if not get_pp_group().is_last_rank:
            return module.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        _append_jsonl(
            dump_path,
            {
                **base_record,
                "stage": "norm",
                "hidden": _tensor_stats(torch, hidden_states),
            },
        )
        return hidden_states

    cls.forward = _forward_with_hidden_dump
    cls._qwen35_hidden_dump_installed = True
    _append_jsonl(
        dump_path,
        {"source": "vllm-sitecustomize", "pid": os.getpid(), "stage": "hook_installed"},
    )


def _patch_qwen3_next_vl(module: Any) -> None:
    dump_path = os.getenv("VLLM_QWEN35_HIDDEN_DUMP")
    if not dump_path:
        return
    cls = getattr(module, "Qwen3NextVLModel", None)
    if cls is None or cls.__dict__.get("_qwen35_vl_hidden_dump_installed", False):
        return

    try:
        expected_tokens = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_EXPECT_TOKENS", "0"))
    except ValueError:
        expected_tokens = 0
    try:
        max_calls = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_MAX_CALLS", "1"))
    except ValueError:
        max_calls = 1
    try:
        max_skips = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_MAX_SKIPS", "20"))
    except ValueError:
        max_skips = 20

    torch = module.torch
    original_forward = cls.forward

    def _forward_with_hidden_dump(
        self: Any,
        input_ids: Any,
        positions: Any,
        intermediate_tensors: Any | None = None,
        inputs_embeds: Any | None = None,
        deepstack_input_embeds: Any | None = None,
    ) -> Any:
        dumped_calls = int(getattr(self, "_qwen35_vl_hidden_dump_calls", 0))
        token_count = None
        if isinstance(inputs_embeds, torch.Tensor):
            token_count = int(inputs_embeds.shape[0])
        elif isinstance(input_ids, torch.Tensor):
            token_count = int(input_ids.shape[0])
        should_dump = dumped_calls < max_calls and (
            not expected_tokens or token_count == expected_tokens
        )
        if not should_dump:
            if os.getenv("VLLM_QWEN35_HIDDEN_DUMP_SKIPS"):
                skipped_calls = int(
                    getattr(self, "_qwen35_vl_hidden_dump_skip_calls", 0)
                )
                if skipped_calls < max_skips:
                    self._qwen35_vl_hidden_dump_skip_calls = skipped_calls + 1
                    _append_jsonl(
                        dump_path,
                        {
                            "source": "vllm-sitecustomize",
                            "class": "Qwen3NextVLModel",
                            "pid": os.getpid(),
                            "stage": "skip",
                            "token_count": token_count,
                            "input_ids_shape": (
                                list(input_ids.shape)
                                if isinstance(input_ids, torch.Tensor)
                                else None
                            ),
                            "positions_shape": (
                                list(positions.shape)
                                if isinstance(positions, torch.Tensor)
                                else None
                            ),
                            "inputs_embeds_shape": (
                                list(inputs_embeds.shape)
                                if isinstance(inputs_embeds, torch.Tensor)
                                else None
                            ),
                        },
                    )
            return original_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                deepstack_input_embeds,
            )

        self._qwen35_vl_hidden_dump_calls = dumped_calls + 1
        get_pp_group = module.get_pp_group
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        base_record = {
            "source": "vllm-sitecustomize",
            "class": "Qwen3NextVLModel",
            "pid": os.getpid(),
            "input_ids_shape": (
                list(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None
            ),
            "positions_shape": (
                list(positions.shape) if isinstance(positions, torch.Tensor) else None
            ),
            "inputs_embeds_shape": (
                list(inputs_embeds.shape)
                if isinstance(inputs_embeds, torch.Tensor)
                else None
            ),
            "input_ids": _tensor_values(torch, input_ids),
            "positions": _tensor_values(torch, positions),
        }
        _append_jsonl(
            dump_path,
            {
                **base_record,
                "stage": "embed",
                "hidden": _tensor_stats(torch, hidden_states),
            },
        )

        accept_hidden_states = None
        for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            layer_idx = layer_idx + self.start_layer
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )
            if (
                self.accept_hidden_layer is not None
                and self.accept_hidden_layer == layer_idx
            ):
                accept_hidden_states = hidden_states
            record = {
                **base_record,
                "stage": "layer",
                "layer_id": int(layer_idx),
                "hidden": _tensor_stats(torch, hidden_states),
            }
            if isinstance(residual, torch.Tensor):
                record["residual"] = _tensor_stats(torch, residual)
            _append_jsonl(dump_path, record)

        if not get_pp_group().is_last_rank:
            return module.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_shape = hidden_states.shape
        if residual is not None:
            hidden_states, _ = self.norm(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                residual.reshape(-1, residual.shape[-1]),
            )
        else:
            hidden_states, _ = self.norm(
                hidden_states.reshape(-1, hidden_states.shape[-1])
            )
        hidden_states = hidden_states.view(hidden_shape)
        _append_jsonl(
            dump_path,
            {
                **base_record,
                "stage": "norm",
                "hidden": _tensor_stats(torch, hidden_states),
            },
        )
        if accept_hidden_states is not None:
            return hidden_states, accept_hidden_states
        return hidden_states

    cls.forward = _forward_with_hidden_dump
    cls._qwen35_vl_hidden_dump_installed = True
    _append_jsonl(
        dump_path,
        {
            "source": "vllm-sitecustomize",
            "class": "Qwen3NextVLModel",
            "pid": os.getpid(),
            "stage": "hook_installed",
        },
    )


def _patch_code_predictor_feedback(module: Any) -> None:
    dump_path = os.getenv("VLLM_QWEN35_TALKER_FEEDBACK_DUMP")
    if not dump_path:
        return
    cls = getattr(module, "CodePredictor", None)
    if cls is None or getattr(cls, "_qwen35_feedback_dump_installed", False):
        return

    try:
        max_calls = int(os.getenv("VLLM_QWEN35_TALKER_FEEDBACK_DUMP_MAX_CALLS", "8"))
    except ValueError:
        max_calls = 8

    torch = module.torch
    original = cls.post_process_hidden_states

    def _post_process_hidden_states_with_dump(
        self: Any,
        text_embeddings: Any,
        subtalker_codec: Any,
    ) -> Any:
        output = original(self, text_embeddings, subtalker_codec)
        dumped_calls = int(getattr(self, "_qwen35_feedback_dump_calls", 0))
        if dumped_calls < max_calls:
            self._qwen35_feedback_dump_calls = dumped_calls + 1
            _append_jsonl(
                dump_path,
                {
                    "source": "vllm-sitecustomize",
                    "pid": os.getpid(),
                    "stage": "talker_feedback",
                    "call": dumped_calls,
                    "text_embeddings": _tensor_stats(torch, text_embeddings),
                    "subtalker_codec": _tensor_values(torch, subtalker_codec),
                    "feedback": _tensor_stats(torch, output),
                },
            )
        return output

    cls.post_process_hidden_states = _post_process_hidden_states_with_dump
    cls._qwen35_feedback_dump_installed = True
    _append_jsonl(
        dump_path,
        {
            "source": "vllm-sitecustomize",
            "pid": os.getpid(),
            "stage": "talker_feedback_hook_installed",
        },
    )


def _maybe_patch_loaded() -> None:
    module = sys.modules.get(_TARGET)
    if module is not None:
        _patch_qwen3_next(module)
    vl_module = sys.modules.get(_VL_TARGET)
    if vl_module is not None:
        _patch_qwen35_audio_encoder_layers(vl_module)
        _patch_qwen35_audio_process(vl_module)
        _patch_qwen3_next_vl(vl_module)
    code_predictor_module = sys.modules.get(_CODE_PREDICTOR_TARGET)
    if code_predictor_module is not None:
        _patch_code_predictor_feedback(code_predictor_module)


def _debug_import(name: str, globals: Any = None, locals: Any = None,
                  fromlist: Any = (), level: int = 0) -> Any:
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if name == _TARGET or name.startswith(_TARGET + "."):
        _maybe_patch_loaded()
    elif name == _VL_TARGET or name.startswith(_VL_TARGET + "."):
        _maybe_patch_loaded()
    elif name == "vllm.model_executor.models" and "qwen3_next" in (fromlist or ()):
        _maybe_patch_loaded()
    elif (
        name == "vllm.model_executor.models"
        and "qwen3_omni_next_thinker" in (fromlist or ())
    ):
        _maybe_patch_loaded()
    elif name == _CODE_PREDICTOR_TARGET or name.startswith(
        _CODE_PREDICTOR_TARGET + "."
    ):
        _maybe_patch_loaded()
    return module


if (
    os.getenv("VLLM_QWEN35_HIDDEN_DUMP")
    or os.getenv("VLLM_QWEN35_AUDIO_DUMP")
    or os.getenv("VLLM_QWEN35_TALKER_FEEDBACK_DUMP")
):
    builtins.__import__ = _debug_import
    _maybe_patch_loaded()
