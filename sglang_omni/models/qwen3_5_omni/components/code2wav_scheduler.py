# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni Code2Wav scheduler.

Qwen3.5-Omni uses the Qwen3OmniNext DAC/codec decoder path from the vLLM
perf_v2 branch. If that decoder is available, this module plugs it into
sglang-omni's existing streaming Code2WavScheduler. Otherwise the stage remains
importable and fails clearly when audio output is requested.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)

QWEN35_CODEC_EOS_TOKEN_ID = 2150
QWEN35_CODE2WAV_SAMPLE_RATE = 24000
# 中文说明：对齐 vLLM dev/qwenc_perf_v2 H20 profile 的 send_chunk_size=8。
# 用户仍可通过 --code2wav-stream-chunk-size 或 YAML factory_args 覆盖。
QWEN35_CODE2WAV_STREAM_CHUNK_SIZE = 8
QWEN35_CODE2WAV_DYNAMIC_CHUNK_SIZES = (2, 4, 6, 8)
QWEN35_CODE2WAV_DYNAMIC_CHUNK_STEPS = (8, 4, 2, 1)
QWEN35_CODE2WAV_ODEINT_METHODS = frozenset({"euler", "rk4"})
QWEN35_CODE2WAV_FREQUENCIES = frozenset({"50hz", "25hz"})
QWEN35_CODE2WAV_FREQUENCY_ALIASES = {
    "1": "50hz",
    "50": "50hz",
    "50hz": "50hz",
    "2": "25hz",
    "25": "25hz",
    "25hz": "25hz",
}
QWEN35_CODE2WAV_DIT_QUANTS = frozenset({"fp8"})


_MISSING_CODE2WAV_MESSAGE = (
    "Qwen3.5-Omni code2wav could not find a loadable Next DAC config and "
    "checkpoint under the model directory. Expected files such as "
    "code2wav/config.yaml + code2wav/model_weights.pt, or equivalent "
    "codec_decoder/dac layouts, before requesting audio output."
)

_CODE2WAV_DIR_NAMES = (
    # 中文说明：vLLM perf_v2/HF reference 里出现过 0306 和 0226
    # 两版在线解码器目录名；优先新目录，但保留旧目录兼容。
    "qwen3_5_omni_codec_decode_online_0306",
    "qwen3_5_omni_codec_decode_online_0226",
    "code2wav",
    "codec_decoder",
    "dac",
    "codec",
)
_CODE2WAV_CONFIG_NAMES = (
    "config.yaml",
    "config.yml",
    "codec_decoder.yaml",
    "dac.yaml",
)
_CODE2WAV_CHECKPOINT_NAMES = (
    "model_weights.pt",
    "checkpoint.pt",
    "model.pt",
    "model.pth",
    "state_dict.pt",
    "state_dict.pth",
    "pytorch_model.bin",
    "generator.pt",
    "generator.pth",
    "codec_decoder.pt",
    "codec_decoder.pth",
    "dac.pt",
    "dac.pth",
)
_CODE2WAV_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin")
_CODE2WAV_SKIP_VOICE_SUFFIXES = ("prefix_caching",)
_CODE2WAV_IGNORED_VOICE_TYPES = frozenset({"", "default", "none", "null"})
_CODE2WAV_AUDIO_OUTPUT_CONFIG_KEYS = frozenset(
    {
        "speaker",
        "voice",
        "voice_type",
    }
)
_CODE2WAV_AUDIO_MEDIA_PAYLOAD_KEYS = frozenset(
    {
        "audio",
        "audio_url",
        "data",
        "input_audio",
        "path",
        "samples",
        "url",
    }
)
_CODE2WAV_VOICE_TYPE_MAPPING = {
    "cherry": "f245",
    "ethan": "m02",
    "serena": "f05",
    "chelsie": "f030",
}


@dataclass(frozen=True)
class Code2WavFiles:
    model_dir: Path
    config_path: Path
    checkpoint_path: Path


def _load_next_dac_loader():
    for module_name in (
        "sglang_omni.models.qwen3_5_omni.components.qwen3_omni_next_codec_decoder",
        "vllm.model_executor.models.qwen3_omni_next_codec_decoder",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        loader = getattr(module, "load_qwen3_omni_next_dac_from_config", None)
        if loader is not None:
            return loader
    return None


def _iter_code2wav_candidate_dirs(model_path: str) -> tuple[Path, ...]:
    root = Path(model_path)
    if root.is_file():
        return (root.parent,)
    return tuple(root / name for name in _CODE2WAV_DIR_NAMES) + (root,)


def _resolve_code2wav_files(model_path: str) -> Code2WavFiles | None:
    root = Path(model_path)
    if root.is_file() and _is_code2wav_checkpoint_file(root):
        config_path = _first_existing(root.parent, _CODE2WAV_CONFIG_NAMES)
        if config_path is not None:
            # 中文说明：用户可能直接传 --code2wav-model-path 到
            # model_weights.pt/model.pth。此时 checkpoint 已经明确，
            # 只需要在同目录找 DAC config。
            return Code2WavFiles(
                model_dir=root.parent,
                config_path=config_path,
                checkpoint_path=root,
            )

    for candidate in _iter_code2wav_candidate_dirs(model_path):
        if not candidate.exists():
            continue
        config_path = _first_existing(candidate, _CODE2WAV_CONFIG_NAMES)
        checkpoint_path = _first_existing(candidate, _CODE2WAV_CHECKPOINT_NAMES)
        if config_path is not None and checkpoint_path is not None:
            return Code2WavFiles(
                model_dir=candidate,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
    return None


def _resolve_code2wav_model_dir(model_path: str) -> Path | None:
    files = _resolve_code2wav_files(model_path)
    return files.model_dir if files is not None else None


def _is_code2wav_checkpoint_file(path: Path) -> bool:
    return (
        path.name in _CODE2WAV_CHECKPOINT_NAMES
        or path.suffix.lower() in _CODE2WAV_CHECKPOINT_SUFFIXES
    )


def _missing_code2wav_message(
    model_path: str,
    *,
    loader_available: bool | None = None,
) -> str:
    lines = [
        _MISSING_CODE2WAV_MESSAGE,
        f"Searched model path: {model_path}",
        "Candidate directories:",
    ]
    lines.extend(f"  - {path}" for path in _iter_code2wav_candidate_dirs(model_path))
    lines.extend(
        [
            "Accepted config filenames: " + ", ".join(_CODE2WAV_CONFIG_NAMES),
            "Accepted checkpoint filenames: "
            + ", ".join(_CODE2WAV_CHECKPOINT_NAMES),
        ]
    )
    if loader_available is False:
        lines.append(
            "Next DAC loader was not importable from the local Qwen3.5 "
            "implementation or the vLLM perf_v2 module."
        )
    # 中文说明：真实 checkpoint 目录不完整时，这条错误会直接列出查找路径和
    # 期望文件名，避免只看到 Missing scheduler 后还要回源码里反查布局。
    return "\n".join(lines)


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = directory / name
        # 中文说明：只接受真实文件。上传中断时可能留下同名目录，
        # 如果用 exists() 会让 preflight/launcher 误判 code2wav 已完整。
        if path.is_file():
            return path
    return None


def _iter_codec_eos_config_paths(model_path: str) -> tuple[Path, ...]:
    root = Path(model_path)
    paths = [root / "config.json"]
    # 中文说明：split checkpoint 场景下 codec_eos_token_id 常在
    # root/talker_lm 或 root/talker/config.json，root config 未必带完整
    # talker_config。
    paths.append(root / "talker_lm" / "config.json")
    paths.append(root / "talker" / "config.json")
    return tuple(paths)


def _read_codec_eos_from_config(model_path: str) -> int | None:
    for config_path in _iter_codec_eos_config_paths(model_path):
        value = _read_codec_eos_from_config_path(config_path)
        if value is not None:
            return value
    return None


def _read_codec_eos_from_config_path(config_path: Path) -> int | None:
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except Exception as exc:
        logger.warning(
            "failed to read Qwen3.5-Omni config for codec eos from %s: %s",
            config_path,
            exc,
        )
        return None

    for container in (
        config.get("talker_config"),
        config.get("text_config"),
        config,
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("codec_eos_token_id")
        if value is not None:
            return int(value)
    return None


def _resolve_codec_eos_token_id(
    model_path: str,
    codec_eos_token_id: int | None,
) -> int:
    if codec_eos_token_id is not None:
        return int(codec_eos_token_id)
    return _read_codec_eos_from_config(model_path) or QWEN35_CODEC_EOS_TOKEN_ID


def _coerce_positive_int_tuple(
    value: Any,
    default: tuple[int, ...],
    *,
    option_name: str,
) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        pieces = [
            piece.strip()
            for piece in value.replace(",", " ").split()
            if piece.strip()
        ]
        values = tuple(int(piece) for piece in pieces)
    elif isinstance(value, int) and not isinstance(value, bool):
        values = (int(value),)
    else:
        try:
            values = tuple(int(item) for item in value)
        except TypeError as exc:
            raise ValueError(
                f"{option_name} must be a positive integer or sequence of "
                f"positive integers, got {value!r}"
            ) from exc
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{option_name} must contain positive integers, got {value!r}")
    return values


def _coerce_positive_int(value: Any, *, option_name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{option_name} must be >= 1, got {value}")
    return value


def _coerce_nonnegative_int(value: Any, *, option_name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{option_name} must be >= 0, got {value}")
    return value


def _coerce_bool(value: Any, *, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
    raise ValueError(f"{option_name} must be a boolean value, got {value!r}")


def _coerce_optional_bool(value: Any, *, option_name: str) -> bool | None:
    if value is None:
        return None
    return _coerce_bool(value, option_name=option_name)


def _coalesce_alias_value(
    primary: Any,
    alias: Any,
    *,
    primary_name: str,
    alias_name: str,
) -> Any:
    if primary is None:
        return alias
    if alias is None or alias == primary:
        return primary
    raise ValueError(
        f"{primary_name} and {alias_name} conflict: {primary!r} != {alias!r}"
    )


def _coerce_code2wav_odeint_method(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in QWEN35_CODE2WAV_ODEINT_METHODS:
        supported = ", ".join(sorted(QWEN35_CODE2WAV_ODEINT_METHODS))
        raise ValueError(f"code2wav odeint_method must be one of: {supported}")
    return normalized


def _coerce_code2wav_frequency(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    frequency = QWEN35_CODE2WAV_FREQUENCY_ALIASES.get(normalized)
    if frequency is None:
        supported = ", ".join(
            sorted((*QWEN35_CODE2WAV_FREQUENCIES, "1", "2", "25", "50"))
        )
        raise ValueError(f"code2wav frequency must be one of: {supported}")
    return frequency


def _coerce_code2wav_dit_quant(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in QWEN35_CODE2WAV_DIT_QUANTS:
        supported = ", ".join(sorted(QWEN35_CODE2WAV_DIT_QUANTS))
        raise ValueError(f"code2wav dit_quant must be one of: {supported}")
    return normalized


def _coerce_positive_optional_int(value: Any, *, option_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{option_name} must be >= 1, got {value}")
    return value


def _expand_dynamic_chunk_sizes(
    sizes: tuple[int, ...],
    steps: tuple[int, ...],
) -> tuple[int, ...]:
    if len(sizes) != len(steps):
        raise ValueError(
            "dynamic code2wav chunk sizes and steps must have the same length"
        )
    expanded = tuple(
        size
        for size, count in zip(sizes, steps)
        for _ in range(int(count))
    )
    if not expanded:
        raise ValueError("dynamic code2wav chunk schedule cannot be empty")
    return expanded


def _should_skip_code2wav_for_request(payload: StagePayload) -> bool:
    for value in _request_voice_values(payload):
        voice_name = str(value).lower().strip()
        if any(
            voice_name == suffix or voice_name.endswith(suffix)
            for suffix in _CODE2WAV_SKIP_VOICE_SUFFIXES
        ):
            return True
    return False


def _code2wav_voice_candidates(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    normalized = str(value).strip().lower()
    if not normalized:
        return ()
    candidates = [normalized]
    if "#" in normalized:
        base = normalized.split("#", 1)[0].strip()
        if base:
            candidates.append(base)
    for candidate in tuple(candidates):
        mapped = _CODE2WAV_VOICE_TYPE_MAPPING.get(candidate)
        if mapped:
            candidates.append(mapped)
    return tuple(dict.fromkeys(candidates))


def _is_ignored_code2wav_voice(value: Any) -> bool:
    candidates = _code2wav_voice_candidates(value)
    if not candidates:
        return True
    return any(candidate in _CODE2WAV_IGNORED_VOICE_TYPES for candidate in candidates)


def _normalize_code2wav_voice_types(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raw_values = (values,)
    else:
        raw_values = tuple(values)
    normalized: set[str] = set()
    for value in raw_values:
        normalized.update(_code2wav_voice_candidates(value))
    normalized.difference_update(_CODE2WAV_IGNORED_VOICE_TYPES)
    return frozenset(normalized)


def _parse_legacy_code2wav_voice_key(path: Path) -> str:
    filename = path.name
    return "default" if filename == "spk_emb.npy" else filename.split("_", 1)[0].lower()


def _load_code2wav_voice_types(model_path: str) -> tuple[str, ...]:
    files = _resolve_code2wav_files(model_path)
    model_dir = files.model_dir if files is not None else Path(model_path)

    legacy_voice_types = [
        _parse_legacy_code2wav_voice_key(path)
        for subdir_name in ("inputs", "inputs_sft4spks")
        for path in sorted((model_dir / subdir_name).glob("*spk_emb.npy"))
    ]
    if legacy_voice_types:
        return tuple(dict.fromkeys(legacy_voice_types))

    spk_dict_path = model_dir / "spk_dict.pt"
    if not spk_dict_path.is_file():
        return ()
    try:
        spk_dict = torch.load(spk_dict_path, map_location="cpu")
    except TypeError:
        spk_dict = torch.load(spk_dict_path)
    if not isinstance(spk_dict, dict):
        logger.warning(
            "Qwen3.5 code2wav spk_dict.pt is not a mapping: %s",
            spk_dict_path,
        )
        return ()

    voice_types = []
    for key in spk_dict:
        normalized = str(key).strip()
        if not normalized:
            continue
        # 中文说明：vLLM perf_v2 会把 spk_dict.pt 里的展示名映射成
        # code2wav 内部 speaker id；这里保留同一规则，避免 Cherry 等
        # 用户可见音色在 code2wav 阶段被误判为未知。
        voice_types.append(_CODE2WAV_VOICE_TYPE_MAPPING.get(normalized.lower(), normalized))
    return tuple(dict.fromkeys(voice_types))


def _request_voice_values(payload: StagePayload) -> tuple[Any, ...]:
    params = payload.request.params if payload.request is not None else None
    values: list[Any] = []
    if isinstance(params, dict):
        for key in ("voice_type", "voice", "speaker"):
            value = params.get(key)
            if value:
                values.append(value)
    metadata = payload.request.metadata if payload.request is not None else None
    audio_config = _metadata_audio_output_config(metadata)
    if isinstance(audio_config, dict):
        for key in ("voice_type", "voice", "speaker"):
            value = audio_config.get(key)
            if value:
                values.append(value)
    return tuple(values)


def _metadata_audio_output_config(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    audio_config = metadata.get("audio_config")
    if isinstance(audio_config, dict):
        return audio_config
    audio_value = metadata.get("audio")
    if not isinstance(audio_value, dict) or not audio_value:
        return None
    keys = set(audio_value)
    if keys & _CODE2WAV_AUDIO_MEDIA_PAYLOAD_KEYS:
        return None
    if not keys & _CODE2WAV_AUDIO_OUTPUT_CONFIG_KEYS:
        return None
    # 中文说明：直连 OmniRequest 时可能只有 metadata.audio={voice,...}。
    # code2wav 只需要音色来处理 prefix_caching 和 speaker 校验；真正的
    # 输入音频 payload(data/path/url) 已在上面排除，避免误判。
    return audio_value


def _skipped_code2wav_result(
    request_id: str,
    payload: StagePayload,
    *,
    sample_rate: int,
) -> list[OutgoingMessage]:
    return [
        OutgoingMessage(
            request_id=request_id,
            type="result",
            data=StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={
                    "modality": "audio",
                    "sample_rate": int(sample_rate),
                    "skipped": True,
                    "reason": "prefix_caching",
                },
            ),
        )
    ]


def _load_next_code2wav_model(
    model_path: str,
    *,
    device: str,
    dtype: str | None,
    loader_kwargs: dict[str, Any] | None = None,
):
    loader = _load_next_dac_loader()
    files = _resolve_code2wav_files(model_path)
    if loader is None or files is None:
        return None

    torch_dtype = resolve_dtype(dtype) or torch.bfloat16
    logger.info("loading Qwen3.5-Omni code2wav from %s", files.model_dir)
    extra_kwargs = _filter_supported_loader_kwargs(loader, loader_kwargs or {})
    return loader(
        config_path=str(files.config_path),
        checkpoint_path=str(files.checkpoint_path),
        device=device,
        dtype=torch_dtype,
        **extra_kwargs,
    )


def _filter_supported_loader_kwargs(
    loader: Any,
    loader_kwargs: dict[str, Any],
) -> dict[str, Any]:
    requested = {key: value for key, value in loader_kwargs.items() if value is not None}
    if not requested:
        return {}
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return {}
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return requested
    return {key: value for key, value in requested.items() if key in parameters}


class Qwen35Code2WavScheduler(Code2WavScheduler):
    """Qwen3.5 code2wav scheduler with vLLM-style prefix-cache skip."""

    def __init__(
        self,
        *args: Any,
        sample_rate: int,
        enable_dynamic_chunk: bool = False,
        dynamic_chunk_sizes: Any = None,
        dynamic_chunk_steps: Any = None,
        code2wav_voice_types: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, sample_rate=sample_rate, **kwargs)
        self._qwen35_skip_code2wav: dict[str, bool] = {}
        self._qwen35_dynamic_chunk_index: dict[str, int] = {}
        self._qwen35_code2wav_voice_types = _normalize_code2wav_voice_types(
            code2wav_voice_types
        )
        self._qwen35_enable_dynamic_chunk = _coerce_bool(
            enable_dynamic_chunk,
            option_name="code2wav_enable_dynamic_chunk",
        )
        chunk_sizes = _coerce_positive_int_tuple(
            dynamic_chunk_sizes,
            QWEN35_CODE2WAV_DYNAMIC_CHUNK_SIZES,
            option_name="dynamic_chunk_sizes",
        )
        chunk_steps = _coerce_positive_int_tuple(
            dynamic_chunk_steps,
            QWEN35_CODE2WAV_DYNAMIC_CHUNK_STEPS,
            option_name="dynamic_chunk_steps",
        )
        self._qwen35_dynamic_chunks = _expand_dynamic_chunk_sizes(
            chunk_sizes,
            chunk_steps,
        )

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        super().on_streaming_new_request(request_id, payload)
        if isinstance(getattr(self, "_payloads", None), dict):
            self._payloads[request_id] = payload
        skip_code2wav = _should_skip_code2wav_for_request(payload)
        self._qwen35_skip_code2wav[request_id] = skip_code2wav
        if not skip_code2wav:
            self._validate_request_voice_type(payload)
        self._qwen35_dynamic_chunk_index[request_id] = 0

    def clear_stream_state(self, request_id: str) -> None:
        super().clear_stream_state(request_id)
        self._qwen35_skip_code2wav.pop(request_id, None)
        self._qwen35_dynamic_chunk_index.pop(request_id, None)

    def _current_dynamic_chunk_size(self, request_id: str) -> int:
        index = self._qwen35_dynamic_chunk_index.get(request_id, 0)
        index = min(index, len(self._qwen35_dynamic_chunks) - 1)
        return int(self._qwen35_dynamic_chunks[index])

    def _validate_request_voice_type(self, payload: StagePayload) -> None:
        if not self._qwen35_code2wav_voice_types:
            return
        for value in _request_voice_values(payload):
            if _is_ignored_code2wav_voice(value):
                continue
            if any(
                candidate in self._qwen35_code2wav_voice_types
                for candidate in _code2wav_voice_candidates(value)
            ):
                continue
            supported = ", ".join(sorted(self._qwen35_code2wav_voice_types))
            raise ValueError(
                f"Qwen3.5 code2wav voice_type {value!r} is not in "
                f"available voices: [{supported}]"
            )

    def on_stream_chunk(
        self,
        request_id: str,
        chunk: StreamItem,
    ) -> list[OutgoingMessage]:
        if self._qwen35_skip_code2wav.get(request_id, False):
            return []
        if not self._qwen35_enable_dynamic_chunk:
            return super().on_stream_chunk(request_id, chunk)

        before_parts = len(self._audio_chunks.get(request_id, []))
        original_chunk_size = self._stream_chunk_size
        self._stream_chunk_size = self._current_dynamic_chunk_size(request_id)
        try:
            messages = super().on_stream_chunk(request_id, chunk)
        finally:
            self._stream_chunk_size = original_chunk_size

        after_parts = len(self._audio_chunks.get(request_id, []))
        if after_parts > before_parts:
            # 中文说明：每完成一次 code2wav decode，就推进到下一档 chunk
            # size。前几档小 chunk 降首音频延迟，后几档大 chunk 降解码开销。
            self._qwen35_dynamic_chunk_index[request_id] = (
                self._qwen35_dynamic_chunk_index.get(request_id, 0) + 1
            )
        return messages

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        if self._qwen35_skip_code2wav.get(request_id, False):
            return _skipped_code2wav_result(
                request_id,
                self._payloads[request_id],
                sample_rate=self._sample_rate,
            )
        return super().on_stream_done(request_id)


def _call_enable_torch_compile(model: Any, *, compile_first_chunk: bool) -> None:
    method = getattr(model, "enable_torch_compile")
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = tuple(signature.parameters.values())
        supports_arg = any(
            param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for param in params
        )
        if supports_arg:
            method(bool(compile_first_chunk))
            return
        method()
        return
    try:
        method(bool(compile_first_chunk))
    except TypeError:
        method()


def _apply_code2wav_torch_compile(
    model: Any,
    *,
    compile_first_chunk: bool = False,
) -> Any:
    if hasattr(model, "_decode_inner"):
        logger.info("applying torch.compile to Qwen3.5 code2wav _decode_inner")
        model._decode_inner = torch.compile(model._decode_inner)
    elif hasattr(model, "enable_torch_compile"):
        logger.info("enabling Qwen3.5 code2wav model-provided torch.compile")
        _call_enable_torch_compile(
            model,
            compile_first_chunk=compile_first_chunk,
        )
    elif hasattr(model, "decode"):
        logger.info("applying torch.compile to Qwen3.5 code2wav decode")
        model.decode = torch.compile(model.decode)
    else:
        logger.info(
            "Qwen3.5 code2wav has no known torch.compile entry point; "
            "leaving model unchanged"
        )
    # 中文说明：对齐 vLLM perf_v2 的 compile 优先级。真实模型如果暴露
    # _decode_inner 就只编译内部热路径；否则优先走模型自带开关，最后才
    # 编译 decode 方法。Qwen3.5 默认开启，用户可用 CLI/YAML 显式关闭。
    return model


def _get_nested_attr(obj: Any, path: tuple[str, ...]) -> Any | None:
    current = obj
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _apply_code2wav_odeint_method(model: Any, method: str) -> bool:
    setter = getattr(model, "set_odeint_method", None)
    if callable(setter):
        setter(method)
        return True

    odeint_kwargs = _get_nested_attr(
        model,
        ("code2wav_dit_model", "cfm_model", "odeint_kwargs"),
    )
    if isinstance(odeint_kwargs, dict):
        odeint_kwargs["method"] = method
        return True

    if hasattr(model, "odeint_method"):
        setattr(model, "odeint_method", method)
        return True
    return False


def _apply_code2wav_odeint_relaxed(model: Any, relaxed: bool) -> bool:
    setter = getattr(model, "set_odeint_method_relaxed", None)
    if callable(setter):
        setter(bool(relaxed))
        return True
    if hasattr(model, "odeint_method_relaxed"):
        setattr(model, "odeint_method_relaxed", bool(relaxed))
        return True
    return False


def _apply_code2wav_batched_chunk(model: Any, batched_chunk: int) -> bool:
    setter = getattr(model, "set_batched_chunk", None)
    if callable(setter):
        setter(int(batched_chunk))
        return True
    if hasattr(model, "batched_chunk"):
        setattr(model, "batched_chunk", int(batched_chunk))
        return True
    bs_mel = getattr(model, "bs_mel", None)
    if hasattr(model, "chunk_size") and bs_mel is not None:
        setattr(model, "chunk_size", int(bs_mel) * int(batched_chunk))
        return True
    return False


def _apply_code2wav_frequency(model: Any, frequency: str) -> bool:
    setter = getattr(model, "set_frequency", None)
    if callable(setter):
        setter(frequency)
        return True
    applied = False
    for attr_name in ("frequency", "code2wav_frequency"):
        if hasattr(model, attr_name):
            setattr(model, attr_name, frequency)
            applied = True
    return applied


def _apply_code2wav_dit_quant(model: Any, dit_quant: str) -> bool:
    setter = getattr(model, "set_dit_quant", None)
    if callable(setter):
        setter(dit_quant)
        return True
    applied = False
    for attr_name in ("dit_quant", "code2wav_dit_quant"):
        if hasattr(model, attr_name):
            setattr(model, attr_name, dit_quant)
            applied = True
    return applied


def _apply_code2wav_runtime_options(
    model: Any,
    *,
    odeint_method: str | None = None,
    odeint_method_relaxed: bool | None = None,
    batched_chunk: int | None = None,
    frequency: str | None = None,
    dit_quant: str | None = None,
) -> Any:
    """Apply vLLM-compatible Code2Wav runtime knobs when the model supports them."""

    if odeint_method is not None and not _apply_code2wav_odeint_method(
        model,
        odeint_method,
    ):
        logger.info(
            "Qwen3.5 code2wav model does not expose an odeint_method hook; "
            "ignoring requested method=%s",
            odeint_method,
        )
    if odeint_method_relaxed is not None:
        odeint_method_relaxed = _coerce_bool(
            odeint_method_relaxed,
            option_name="code2wav_odeint_method_relaxed",
        )
        if not _apply_code2wav_odeint_relaxed(model, odeint_method_relaxed):
            logger.info(
                "Qwen3.5 code2wav model does not expose an odeint relaxed hook; "
                "ignoring requested relaxed=%s",
                odeint_method_relaxed,
            )
    if batched_chunk is not None and not _apply_code2wav_batched_chunk(
        model,
        batched_chunk,
    ):
        logger.info(
            "Qwen3.5 code2wav model does not expose a batched_chunk hook; "
            "ignoring requested batched_chunk=%s",
            batched_chunk,
        )
    if frequency is not None and not _apply_code2wav_frequency(model, frequency):
        logger.info(
            "Qwen3.5 code2wav model does not expose a frequency hook; "
            "ignoring requested frequency=%s",
            frequency,
        )
    if dit_quant is not None and not _apply_code2wav_dit_quant(model, dit_quant):
        logger.info(
            "Qwen3.5 code2wav model does not expose a DIT quant hook; "
            "ignoring requested dit_quant=%s",
            dit_quant,
        )
    # 中文说明：这些参数来自 vLLM perf_v2 的 code2wav DIT 路线。当前内置
    # Next DAC 如果没有对应属性就安全忽略；后续接入真实 DIT/在线 decoder
    # 时，同名 setter/属性会自动生效。
    return model


class MissingQwen35Code2WavScheduler(StreamingSimpleScheduler):
    """Streaming scheduler that reports the missing Next codec path clearly."""

    def __init__(
        self,
        *,
        sample_rate: int = QWEN35_CODE2WAV_SAMPLE_RATE,
        missing_message: str = _MISSING_CODE2WAV_MESSAGE,
    ) -> None:
        super().__init__(compute_fn=None)
        self._payloads: dict[str, StagePayload] = {}
        self._sample_rate = int(sample_rate)
        self._skip_code2wav: dict[str, bool] = {}
        self._missing_message = missing_message

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self._payloads[request_id] = payload
        self._skip_code2wav[request_id] = _should_skip_code2wav_for_request(payload)

    def clear_stream_state(self, request_id: str) -> None:
        self._payloads.pop(request_id, None)
        self._skip_code2wav.pop(request_id, None)

    def _error(self, request_id: str) -> list[OutgoingMessage]:
        self.abort(request_id)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=NotImplementedError(self._missing_message),
            )
        ]

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        del chunk
        if self._skip_code2wav.get(request_id, False):
            return []
        return self._error(request_id)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        if self._skip_code2wav.get(request_id, False):
            return _skipped_code2wav_result(
                request_id,
                self._payloads[request_id],
                sample_rate=self._sample_rate,
            )
        return self._error(request_id)


def create_code2wav_scheduler(
    model_path: str,
    *,
    code2wav_model_path: str | None = None,
    enable_torch_compile: bool = True,
    code2wav_enable_torch_compile: bool | None = None,
    enable_torch_compile_first_chunk: bool | None = None,
    code2wav_enable_torch_compile_first_chunk: bool | None = None,
    odeint_method: str | None = None,
    code2wav_odeint_method: str | None = None,
    odeint_method_relaxed: bool | None = None,
    code2wav_odeint_method_relaxed: bool | None = None,
    batched_chunk: int | None = None,
    code2wav_batched_chunk: int | None = None,
    frequency: str | None = None,
    code2wav_frequency: str | None = None,
    dit_quant: str | None = None,
    code2wav_dit_quant: str | None = None,
    device: str = "cuda",
    dtype: str | None = None,
    gpu_id: int | None = None,
    codec_eos_token_id: int | None = None,
    sample_rate: int = QWEN35_CODE2WAV_SAMPLE_RATE,
    stream_chunk_size: int = QWEN35_CODE2WAV_STREAM_CHUNK_SIZE,
    left_context_size: int = 25,
    enable_dynamic_chunk: bool = False,
    dynamic_chunk_sizes: Any = None,
    dynamic_chunk_steps: Any = None,
    **_: Any,
):
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    sample_rate = _coerce_positive_int(
        sample_rate,
        option_name="code2wav_sample_rate",
    )
    stream_chunk_size = _coerce_positive_int(
        stream_chunk_size,
        option_name="code2wav_stream_chunk_size",
    )
    left_context_size = _coerce_nonnegative_int(
        left_context_size,
        option_name="code2wav_left_context_size",
    )
    codec_eos_token_id = _coerce_nonnegative_int(
        _resolve_codec_eos_token_id(model_path, codec_eos_token_id),
        option_name="code2wav_codec_eos_token_id",
    )
    enable_dynamic_chunk = _coerce_bool(
        enable_dynamic_chunk,
        option_name="code2wav_enable_dynamic_chunk",
    )
    dynamic_chunk_sizes = _coerce_positive_int_tuple(
        dynamic_chunk_sizes,
        QWEN35_CODE2WAV_DYNAMIC_CHUNK_SIZES,
        option_name="dynamic_chunk_sizes",
    )
    dynamic_chunk_steps = _coerce_positive_int_tuple(
        dynamic_chunk_steps,
        QWEN35_CODE2WAV_DYNAMIC_CHUNK_STEPS,
        option_name="dynamic_chunk_steps",
    )
    _expand_dynamic_chunk_sizes(dynamic_chunk_sizes, dynamic_chunk_steps)

    compile_first_chunk = _coalesce_alias_value(
        enable_torch_compile_first_chunk,
        code2wav_enable_torch_compile_first_chunk,
        primary_name="enable_torch_compile_first_chunk",
        alias_name="code2wav_enable_torch_compile_first_chunk",
    )
    compile_first_chunk = _coerce_optional_bool(
        compile_first_chunk,
        option_name="code2wav_enable_torch_compile_first_chunk",
    )
    odeint_method = _coerce_code2wav_odeint_method(
        _coalesce_alias_value(
            odeint_method,
            code2wav_odeint_method,
            primary_name="odeint_method",
            alias_name="code2wav_odeint_method",
        )
    )
    odeint_method_relaxed = _coalesce_alias_value(
        odeint_method_relaxed,
        code2wav_odeint_method_relaxed,
        primary_name="odeint_method_relaxed",
        alias_name="code2wav_odeint_method_relaxed",
    )
    odeint_method_relaxed = _coerce_optional_bool(
        odeint_method_relaxed,
        option_name="code2wav_odeint_method_relaxed",
    )
    batched_chunk = _coerce_positive_optional_int(
        _coalesce_alias_value(
            batched_chunk,
            code2wav_batched_chunk,
            primary_name="batched_chunk",
            alias_name="code2wav_batched_chunk",
        ),
        option_name="code2wav_batched_chunk",
    )
    frequency = _coalesce_alias_value(
        _coerce_code2wav_frequency(frequency),
        _coerce_code2wav_frequency(code2wav_frequency),
        primary_name="frequency",
        alias_name="code2wav_frequency",
    )
    if batched_chunk is None and frequency is not None:
        batched_chunk = 2 if frequency == "50hz" else 1
    dit_quant = _coerce_code2wav_dit_quant(
        _coalesce_alias_value(
            dit_quant,
            code2wav_dit_quant,
            primary_name="dit_quant",
            alias_name="code2wav_dit_quant",
        )
    )

    # 中文说明：优先复用 sglang-omni 现有 streaming Code2WavScheduler。
    # 只要 Next DAC decoder 和权重目录就绪，这里就能进入真实音频解码路径。
    resolved_model_path = code2wav_model_path or model_path
    loader_kwargs = {
        "enable_torch_compile_first_chunk": compile_first_chunk,
        "odeint_method": odeint_method,
        "odeint_method_relaxed": odeint_method_relaxed,
        "batched_chunk": batched_chunk,
        "frequency": frequency,
        "dit_quant": dit_quant,
    }
    loader_kwargs = {
        key: value for key, value in loader_kwargs.items() if value is not None
    }
    load_kwargs = {"device": device, "dtype": dtype}
    if loader_kwargs:
        load_kwargs["loader_kwargs"] = loader_kwargs
    model = _load_next_code2wav_model(resolved_model_path, **load_kwargs)
    if model is not None:
        enable_torch_compile = _coerce_bool(
            enable_torch_compile,
            option_name="code2wav_enable_torch_compile",
        )
        if code2wav_enable_torch_compile is not None:
            enable_torch_compile = _coerce_bool(
                code2wav_enable_torch_compile,
                option_name="code2wav_enable_torch_compile",
            )
        model = _apply_code2wav_runtime_options(
            model,
            odeint_method=odeint_method,
            odeint_method_relaxed=odeint_method_relaxed,
            batched_chunk=batched_chunk,
            frequency=frequency,
            dit_quant=dit_quant,
        )
        if enable_torch_compile:
            model = _apply_code2wav_torch_compile(
                model,
                compile_first_chunk=bool(compile_first_chunk),
            )
        return Qwen35Code2WavScheduler(
            model,
            device=device,
            codec_eos_token_id=codec_eos_token_id,
            sample_rate=sample_rate,
            stream_chunk_size=stream_chunk_size,
            left_context_size=left_context_size,
            enable_dynamic_chunk=enable_dynamic_chunk,
            dynamic_chunk_sizes=dynamic_chunk_sizes,
            dynamic_chunk_steps=dynamic_chunk_steps,
            code2wav_voice_types=_load_code2wav_voice_types(resolved_model_path),
        )

    return MissingQwen35Code2WavScheduler(
        sample_rate=sample_rate,
        missing_message=_missing_code2wav_message(
            resolved_model_path,
            loader_available=_load_next_dac_loader() is not None,
        ),
    )
