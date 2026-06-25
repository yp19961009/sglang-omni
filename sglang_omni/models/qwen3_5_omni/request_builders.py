# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-Omni request builders."""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.qwen3_omni import request_builders as qwen3_request_builders
from sglang_omni.models.qwen3_omni.components.talker_input import (
    segment_chat_template,
)
from sglang_omni.models.qwen3_omni.request_builders import *  # noqa: F401,F403
from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    TalkerPrefillBuilder,
    coerce_feature_tensor,
    resolve_speaker_id,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    _build_talker_request_data,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage

logger = logging.getLogger(__name__)

_VOICE_TO_SPK_MAPPING = {
    # Common user-facing voice aliases used by Qwen3.5-Omni.
    "芊悦": "f245",
    "cherry": "f245",
    "苏瑶": "f05",
    "晨煦": "m02",
    "千雪": "f030",
    "chelsie": "f030",
    "tina": "f6009",
    "qiao": "f666",
    "liora mira": "czr-f2810-cluster4",
    "evan": "m6005",
    "katerina": "f37",
    "ryan": "m36",
    "mione": "f1015",
    "harvey": "m905",
    "mia": "br_f094",
    "emilien": "m1001",
    "marina": "f2003",
    "rizky": "m102",
    "sohee": "f02",
    "jakub": "m109",
    "alek": "m03",
    "sonrisa": "f3002",
    "bodega": "m10",
    "bea": "f01",
    "hana": "f1001",
    "serena": "f05",
    "ethan": "m02",
    "cindy": "f1003_b",
    "sunnybobi": "czr-f1818-cluster2",
    "raymond": "czr-m2831-cluster1",
    "maia": "f20",
    "momo": "f30",
    "theo calm": "m789",
    "wil": "m1012",
    "eliska": "f2001",
    "griet": "f36",
    "jennifer": "f04",
    "aiden": "m11",
    "gold": "m1005",
    "li cassian": "br_m028",
    "joyner": "br_m079",
    "angel": "br_f027",
    "siiri": "f6001",
    "lenn": "m06",
    "sigga": "f1006",
    "dolce": "m04",
    "ono anna": "f3001",
    "chloe": "f07_msa",
    "ingrid": "f1002",
    "roya": "f2008",
    "andre": "m40",
    "radio gol": "m034-23",
    "arda": "m6010",
    "sunny": "f568-04",
    "dylan": "m325-75",
    "li": "m680",
    "marcus": "m987",
    "peter": "m952",
    "eric": "m002",
    "rocky": "jm555",
    "joseph chen": "m103_minnan",
}
_VOICE_CONTROL_SUFFIXES = ("prefix_caching",)
_QWEN35_TALKER_NUM_OUTPUT_IN_CHUNK = 4
_QWEN35_TALKER_TEXT_FEEDBACK_STRIDE = _QWEN35_TALKER_NUM_OUTPUT_IN_CHUNK
_QWEN35_RTC_TEXT_FILLER_PER_CHUNK = 20
_QWEN35_RTC_AUDIO_ROWS_PER_CHUNK = 14
_VOICE_PARAM_KEYS = ("speaker", "voice", "voice_type")
_OPENAI_AUDIO_VOICE_KEYS = ("voice_type", "voice", "speaker")
_VOICE_STYLE_PATTERN = re.compile(
    r"^\s*<?\s*voice_style\s*>?\s*"
    r"(.*?)"
    r"\s*<?\s*/\s*voice_style\s*>?\s*",
    re.IGNORECASE | re.DOTALL,
)
_VOICE_STYLE_CONTAINS = re.compile(
    r"<?\s*/?voice_style",
    re.IGNORECASE,
)


def _qwen35_talker_text_feedback_stride() -> int:
    raw = os.environ.get("QWEN35_TALKER_TEXT_FEEDBACK_STRIDE")
    if raw is None:
        # Align Qwen3.5 external-text handoff with vLLM's
        # VLLM_OMNI_TALKER_NUM_OUTPUT_IN_CHUNK. Text rows are consumed in
        # 4-token chunks; between chunks the talker runs four feedback steps
        # before the boundary drop/next external text chunk.
        return _QWEN35_TALKER_TEXT_FEEDBACK_STRIDE
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring invalid QWEN35_TALKER_TEXT_FEEDBACK_STRIDE=%r",
            raw,
        )
        return _QWEN35_TALKER_TEXT_FEEDBACK_STRIDE


_VOICE_STYLE_TRAILING_PREFIX = re.compile(r"^(?:[>\s]|\\[nrt])*")
_LIVETRANSLATE_TARGET_PATTERN = re.compile(
    r"\bspeech\s+into\b\s+([A-Za-z_][\w-]*)",
    re.IGNORECASE,
)
_LANGUAGE_PARAM_KEYS = (
    "language_id",
    "language",
    "lang",
    "language_type",
    "target_language",
    "target_lang",
)
_OPENAI_AUDIO_LANGUAGE_KEYS = (
    "language_id",
    "language",
    "lang",
    "language_type",
    "target_language",
    "target_lang",
)
_TRANSLATION_TARGET_KEYS = (
    "target_language",
    "target_lang",
    "targetLanguage",
    "language",
    "lang",
)
# Live-translate/TTS requests may expose human-readable language names such as
# Chinese, English, and Filipino, while Qwen3.5 talker configs commonly use
# codes such as zh/en/tagalog/yue in talker_language_id. Expand candidates
# during request parsing so users do not have to provide language_id manually.
_LANGUAGE_NAME_ALIASES = {
    "chinese": ("zh", "zh-cn", "cmn"),
    "mandarin": ("zh", "zh-cn", "cmn"),
    "chinese-tw": ("zh-tw", "zh-hant"),
    "chinese-wu": ("wuu",),
    "chinese-dialect": ("dialect-zh", "zh"),
    "cantonese": ("yue", "zh-yue"),
    "hokkien": ("nan",),
    "english": ("en", "eng"),
    "arabic": ("ar", "ara"),
    "german": ("de", "deu"),
    "french": ("fr", "fra"),
    "spanish": ("es", "spa"),
    "portuguese": ("pt", "por"),
    "indonesian": ("id", "ind"),
    "italian": ("it", "ita"),
    "japanese": ("ja", "jpn"),
    "korean": ("ko", "kor"),
    "russian": ("ru", "rus"),
    "thai": ("th", "tha"),
    "vietnamese": ("vi", "vie"),
    "turkish": ("tr", "tur"),
    "hindi": ("hi", "hin"),
    "dutch": ("nl", "nld"),
    "polish": ("pl", "pol"),
    "bulgarian": ("bg", "bul"),
    "romanian": ("ro", "ron"),
    "hebrew": ("he", "iw", "heb"),
    "ukrainian": ("uk", "ukr"),
    "serbian": ("sr", "srp"),
    "swedish": ("sv", "swe"),
    "czech": ("cs", "ces"),
    "norwegian": ("no",),
    "danish": ("da",),
    "malay": ("ms",),
    "urdu": ("ur",),
    "finnish": ("fi",),
    "persian": ("fa",),
    "greek": ("el",),
    "filipino": ("tagalog", "tl"),
    "tagalog": ("filipino", "tl"),
    "afrikaans": ("af",),
    "asturian": ("ast",),
    "belarusian": ("be",),
    "bengali": ("bn",),
    "bosnian": ("bs",),
    "catalan": ("ca",),
    "cebuano": ("ceb",),
    "estonian": ("et",),
    "galician": ("gl",),
    "gujarati": ("gu",),
    "croatian": ("hr",),
    "hungarian": ("hu",),
    "javanese": ("jv",),
    "kazakh": ("kk",),
    "kannada": ("kn",),
    "kyrgyz": ("ky",),
    "latvian": ("lv",),
    "macedonian": ("mk",),
    "malayalam": ("ml",),
    "marathi": ("mr",),
    "punjabi": ("pa",),
    "slovak": ("sk",),
    "slovenian": ("sl",),
    "swahili": ("sw",),
    "tajik": ("tg",),
    "azerbaijani": ("az",),
}
_ASSISTANT_INSTRUCT_ID_KEYS = (
    "assistant_instruct_ids",
    "talker_assistant_prompt_ids",
    "voice_style_ids",
    "instruct_ids",
)
_STYLE_PARAM_KEYS = (*_ASSISTANT_INSTRUCT_ID_KEYS, "voice_style", "style")
_INSTRUCTION_PARAM_KEYS = (
    *_ASSISTANT_INSTRUCT_ID_KEYS,
    "instruction",
    "talker_instruction",
)
_VOICE_CLONE_PARAM_KEYS = ("xvector_info", "voice_clone_info")
_VOICE_CLONE_PATH_KEYS = ("path", "xvector_info_path", "voice_clone_path")
_OPENAI_AUDIO_VOICE_CLONE_KEYS = (
    "xvector_info",
    "voice_clone_info",
    "xvector_info_path",
    "voice_clone_path",
    "voice_clone",
)
_OPENAI_AUDIO_OUTPUT_CONFIG_KEYS = frozenset(
    {
        "format",
        "instruction",
        "lang",
        "language",
        "language_id",
        "speaker",
        "style",
        "target_lang",
        "target_language",
        "voice",
        "voice_clone",
        "voice_clone_info",
        "voice_clone_path",
        "voice_style",
        "voice_type",
        "xvector_info",
        "xvector_info_path",
    }
)
_AUDIO_MEDIA_PAYLOAD_KEYS = frozenset(
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
_VOICE_CLONE_SYSTEM_INSTRUCT_KEYS = (
    "system_instruct",
    "voice_clone_system_instruct",
    "system_instruct_ids",
    "talker_system_instruct",
)
_VOICE_CLONE_LANGUAGE_KEYS = (
    "language",
    "lang",
    "target_language",
    "target_lang",
    "language_type",
    "prompt_language",
    "ref_language",
)
_TTS_GENERATE_MODE_CONFIG = {
    "default": {
        "ignore_instructions": True,
        "force_voice_type": None,
    },
    "voice_design": {
        "ignore_instructions": False,
        "force_voice_type": "none",
    },
    "instructions": {
        "ignore_instructions": False,
        "force_voice_type": None,
    },
}


def _coerce_output_modalities(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        # The official OpenAI field is usually a list, but benchmark scripts and
        # curl calls often pass "text,audio" or "text audio" strings. Split both
        # forms into tokens.
        values = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            if isinstance(item, str):
                values.extend(item.replace(",", " ").split())
            else:
                values.append(item)
    else:
        return None
    modalities = {
        str(modality).strip().lower()
        for modality in values
        if str(modality).strip()
    }
    return modalities or None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "audio"}:
            return True
        if normalized in {"0", "false", "no", "off", "text", "none"}:
            return False
    return None


def _explicit_audio_output_enabled(request: OmniRequest | None) -> bool | None:
    params = getattr(request, "params", None)
    if not isinstance(params, dict):
        return None
    for key in ("enable_audio_output", "audio_output", "return_audio", "do_wave"):
        value = _coerce_optional_bool(params.get(key))
        if value is not None:
            # enable_audio_output is the Qwen3.5 audio-output switch. do_wave is
            # a common offline/benchmark-script flag meaning "synthesize a
            # waveform", so support both.
            return value
    return None


def output_modalities(request: OmniRequest | None) -> set[str] | None:
    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, dict):
        modalities = _coerce_output_modalities(metadata.get("output_modalities"))
        if modalities is not None:
            # The Qwen3 base class only lowercases values; Qwen3.5 also needs to
            # support the ["text,audio"] form commonly used by OpenAI benchmarks.
            return modalities

    metadata_modalities = qwen3_request_builders.output_modalities(request)
    if metadata_modalities is not None:
        return metadata_modalities

    params = getattr(request, "params", None)
    if not isinstance(params, dict):
        return None
    for key in ("modalities", "output_modalities", "response_modalities"):
        modalities = _coerce_output_modalities(params.get(key))
        if modalities is not None:
            # OpenAI-compatible requests usually put modalities at the top level
            # of the body. After conversion to OmniRequest it may land in
            # params, so recognize that path as well.
            return modalities
    return None


def should_generate_audio_output(
    payload_or_request: StagePayload | OmniRequest | None,
) -> bool:
    request = (
        payload_or_request.request
        if isinstance(payload_or_request, StagePayload)
        else payload_or_request
    )
    modalities = output_modalities(request)
    if modalities is not None:
        return "audio" in modalities
    enabled = _explicit_audio_output_enabled(request)
    if enabled is not None:
        return enabled
    return True


def resolve_mm_aggregate_next_stages(
    request_id: str, output: StagePayload
) -> str | list[str]:
    del request_id
    if should_generate_audio_output(output):
        return [
            qwen3_request_builders.THINKER_STAGE,
            qwen3_request_builders.TALKER_STAGE,
        ]
    return qwen3_request_builders.THINKER_STAGE


def resolve_thinker_next_stages(
    request_id: str, output: StagePayload
) -> str | list[str] | None:
    if _is_qwen35_rtc_thinker_terminal_prerun(output.request):
        return None
    return qwen3_request_builders.resolve_thinker_next_stages(request_id, output)


def resolve_thinker_stream_done_targets(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    if _is_qwen35_rtc_thinker_terminal_prerun(output.request):
        return []
    if should_generate_audio_output(output):
        return [
            qwen3_request_builders.TALKER_STAGE,
            qwen3_request_builders.DECODE_STAGE,
        ]
    return [qwen3_request_builders.DECODE_STAGE]


def resolve_terminal_stages(request: OmniRequest) -> list[str]:
    if _is_qwen35_rtc_thinker_terminal_prerun(request):
        return [qwen3_request_builders.THINKER_STAGE]
    if should_generate_audio_output(request):
        return [
            qwen3_request_builders.DECODE_STAGE,
            qwen3_request_builders.CODE2WAV_STAGE,
        ]
    return [qwen3_request_builders.DECODE_STAGE]


def _as_grid_list(value: Any) -> list[list[int]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    rows = list(value)
    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]
    return [[int(part) for part in row] for row in rows]


def _strip_voice_control_suffix(voice_name: str) -> str:
    normalized = voice_name.lower().strip()
    for suffix in _VOICE_CONTROL_SUFFIXES:
        if normalized == suffix:
            return ""
        if not normalized.endswith(suffix):
            continue
        if "#" in normalized:
            normalized = normalized.rsplit("#", 1)[0].strip()
        else:
            normalized = normalized[: -len(suffix)].rstrip("#:_- ").strip()
    return normalized


def _raw_voice_name(params: dict[str, Any]) -> str | None:
    for key in _VOICE_PARAM_KEYS:
        if key not in params or params[key] is None:
            continue
        return str(params[key]).lower().strip()
    return None


def _has_any_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in mapping and mapping[key] is not None for key in keys)


def _first_present_value(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[str, Any] | tuple[None, None]:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return key, value
    return None, None


def _normalize_openai_voice_clone_value(key: str | None, value: Any) -> Any:
    if isinstance(value, dict):
        for path_key in _VOICE_CLONE_PATH_KEYS:
            path_value = value.get(path_key)
            if path_value is not None and len(value) == 1:
                return path_value
        if key in {"xvector_info_path", "voice_clone_path"}:
            for path_key in _VOICE_CLONE_PATH_KEYS:
                path_value = value.get(path_key)
                if path_value is not None:
                    return path_value
    return value


def _looks_like_openai_audio_output_config(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    keys = set(value)
    if keys & _AUDIO_MEDIA_PAYLOAD_KEYS:
        return False
    return bool(keys & _OPENAI_AUDIO_OUTPUT_CONFIG_KEYS)


def _translation_target_value(params: dict[str, Any]) -> Any:
    options = params.get("translation_options")
    if not isinstance(options, dict):
        return None
    _, value = _first_present_value(options, _TRANSLATION_TARGET_KEYS)
    return value


def _has_language_param_value(params: dict[str, Any]) -> bool:
    return _has_any_value(params, _LANGUAGE_PARAM_KEYS) or (
        _translation_target_value(params) is not None
    )


def _params_with_openai_audio_config(request: Any) -> dict[str, Any]:
    params = dict(getattr(request, "params", None) or {})

    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, dict):
        return params
    audio_config = metadata.get("audio_config")
    if not isinstance(audio_config, dict):
        audio_value = metadata.get("audio")
        if _looks_like_openai_audio_output_config(audio_value):
            # The serve layer normalizes top-level OpenAI audio into
            # audio_config. This fallback also supports directly constructed
            # OmniRequest objects or older internal Client paths that pass
            # metadata.audio, so the talker does not lose voice settings.
            audio_config = audio_value
    if not isinstance(audio_config, dict):
        return params

    if _raw_voice_name(params) is None:
        _, value = _first_present_value(audio_config, _OPENAI_AUDIO_VOICE_KEYS)
        if value is not None:
            # OpenAI chat completions audio.voice lands in
            # request.metadata["audio_config"] at the serve layer. The talker
            # only reads params, so bridge it once without overriding explicitly
            # provided voice_type/speaker values.
            params["voice_type"] = value

    if not _has_language_param_value(params):
        key, value = _first_present_value(audio_config, _OPENAI_AUDIO_LANGUAGE_KEYS)
        if value is not None:
            # language_id carries a numeric token directly. Natural language
            # names and locale codes should use language; _resolve_language_id
            # handles aliases such as zh-CN/en_US later.
            params["language_id" if key == "language_id" else "language"] = value

    if not _has_any_value(params, _STYLE_PARAM_KEYS):
        _, value = _first_present_value(audio_config, ("voice_style", "style"))
        if value is not None:
            params["voice_style"] = value

    if not _has_any_value(params, _INSTRUCTION_PARAM_KEYS):
        _, value = _first_present_value(
            audio_config,
            ("instruction", "talker_instruction"),
        )
        if value is not None:
            params["instruction"] = value

    if not _has_any_value(params, _VOICE_CLONE_PARAM_KEYS):
        key, value = _first_present_value(audio_config, _OPENAI_AUDIO_VOICE_CLONE_KEYS)
        if value is not None:
            target_key = "xvector_info" if key and key.startswith("xvector") else "voice_clone_info"
            # Voice clone inputs are normalized in Qwen35TalkerPrefillBuilder
            # into prompt_speaker_codes/system instruct. Here we only connect
            # OpenAI audio config to the same params path, avoiding a separate
            # scheduling branch.
            params[target_key] = _normalize_openai_voice_clone_value(key, value)
    return params


def _iter_request_text_fragments(value: Any, *, depth: int = 0) -> list[str]:
    if value is None or depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        fragments: list[str] = []
        for key in (
            "prompt",
            "messages",
            "message",
            "content",
            "text",
            "input_text",
        ):
            if key in value:
                fragments.extend(
                    _iter_request_text_fragments(value[key], depth=depth + 1)
                )
        return fragments
    if isinstance(value, (list, tuple)):
        fragments = []
        for item in value:
            fragments.extend(_iter_request_text_fragments(item, depth=depth + 1))
        return fragments
    return []


def _infer_livetranslate_target_language(request: Any) -> str | None:
    params = getattr(request, "params", None)
    metadata = getattr(request, "metadata", None)
    sources = [
        getattr(request, "inputs", None),
        params.get("prompt") if isinstance(params, dict) else None,
        metadata.get("prompt") if isinstance(metadata, dict) else None,
    ]
    for source in sources:
        for text in _iter_request_text_fragments(source):
            match = _LIVETRANSLATE_TARGET_PATTERN.search(text)
            if match is None:
                continue
            language = match.group(1).strip()
            if not language:
                continue
            # Qwen3.5 live-translate normalizes Filipino to Tagalog, and
            # talker_language_id usually uses tagalog as the key as well.
            return "Tagalog" if language.lower() == "filipino" else language
    return None


def _params_with_request_language_fallback(
    request: Any,
    params: dict[str, Any],
) -> dict[str, Any]:
    if _has_language_param_value(params):
        return params
    target_language = _infer_livetranslate_target_language(request)
    if target_language is None:
        return params
    normalized = dict(params)
    # Support fixed Qwen3.5 live-translate prompts such as
    # "... speech into Chinese ...". Infer the language from the prompt only
    # when the request does not specify one, so normal TTS or explicit
    # translation requests are not overridden by text content.
    normalized["target_language"] = target_language
    return normalized


def _should_ignore_voice(params: dict[str, Any]) -> bool:
    return _raw_voice_name(params) in {"none", "null"}


def _resolve_tts_generate_mode(params: dict[str, Any]) -> str | None:
    for key in ("tts_generate_mode", "qwen_tts_generate_mode"):
        value = params.get(key)
        if value:
            return str(value).lower().strip()
    env_value = os.environ.get("SGLANG_OMNI_QWEN_TTS_GENERATE_MODE")
    return env_value.lower().strip() if env_value else None


def _apply_tts_generate_mode(params: dict[str, Any]) -> dict[str, Any]:
    mode = _resolve_tts_generate_mode(params)
    if not mode:
        return params
    config = _TTS_GENERATE_MODE_CONFIG.get(
        mode,
        _TTS_GENERATE_MODE_CONFIG["default"],
    )
    normalized = dict(params)
    if config["force_voice_type"] is not None:
        normalized["voice_type"] = config["force_voice_type"]
        normalized.pop("voice", None)
        normalized.pop("speaker", None)
    if config["ignore_instructions"]:
        for key in (
            "assistant_instruct_ids",
            "talker_assistant_prompt_ids",
            "voice_style_ids",
            "instruct_ids",
            "voice_style",
            "instruction",
            "talker_instruction",
            "style",
        ):
            normalized.pop(key, None)
    # Apply this only when mode is provided explicitly or the matching SGLang
    # env is set. Otherwise keep the current sglang-omni request behavior so
    # existing instruction requests are not silently dropped.
    return normalized


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str | None:
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None
    decode_kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(decode(token_ids, **decode_kwargs))
    except TypeError:
        try:
            return str(decode(token_ids, skip_special_tokens=False))
        except TypeError:
            return str(decode(token_ids))


def _looks_like_chinese_text(text: str | None) -> bool:
    if not text:
        return False
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _parse_voice_style_prefix(text: str) -> tuple[str, str]:
    match = _VOICE_STYLE_PATTERN.match(text)
    if match is None:
        return text, ""
    instruction = match.group(1).strip()
    content = text[match.end() :]
    content = _VOICE_STYLE_TRAILING_PREFIX.sub("", content, count=1)
    if not instruction or _VOICE_STYLE_CONTAINS.search(content):
        return text, ""
    return content, instruction


def _voice_style_drop_count(
    tokenizer: Any,
    token_ids: list[int],
    *,
    content: str,
) -> int | None:
    if not content:
        return len(token_ids)

    for index in range(len(token_ids) + 1):
        suffix = _decode_token_ids(tokenizer, token_ids[index:])
        if suffix == content:
            return index
    return None


def _index_after(tokens: list[int], token_id: int, start: int) -> int | None:
    try:
        return tokens.index(int(token_id), start)
    except ValueError:
        return None


def _count_consecutive_tokens(
    tokens: list[int],
    *,
    token_id: int,
    start: int,
) -> int:
    count = 0
    while start + count < len(tokens) and tokens[start + count] == token_id:
        count += 1
    return count


def _get_token_id(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if value is None:
        raise AttributeError(f"Qwen3.5-Omni config missing {name}")
    return int(value)


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.reshape(-1).detach().cpu().tolist()]
    if isinstance(value, str):
        pieces = [
            piece.strip()
            for piece in value.replace(",", " ").split()
            if piece.strip()
        ]
        return [int(piece) for piece in pieces]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _coerce_long_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.long)
    if hasattr(value, "__array__"):
        return torch.as_tensor(value, dtype=torch.long)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.long)
    return None


def _as_params_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _first_param_value(
    params: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        value = params.get(key)
        if value is not None:
            return value
    return default


def _first_present_param(
    params: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in params and params[key] is not None:
            return params[key]
    return None


def _language_key_candidates(value: Any) -> tuple[str, ...]:
    language = "-".join(str(value).lower().strip().replace("_", "-").split())
    if not language:
        return ()
    candidates = [language]
    if "-" in language:
        candidates.append(language.split("-", 1)[0])
    for candidate in tuple(candidates):
        candidates.extend(_LANGUAGE_NAME_ALIASES.get(candidate, ()))
    return tuple(dict.fromkeys(candidates))


def _load_voice_clone_info_from_path(path: str) -> dict[str, Any]:
    voice_dir = Path(path)
    feat_path = voice_dir / "feat.pkl"
    info_path = voice_dir / "info.json"
    if not feat_path.is_file():
        raise FileNotFoundError(f"Qwen3.5 voice clone feat.pkl not found: {feat_path}")
    if not info_path.is_file():
        raise FileNotFoundError(f"Qwen3.5 voice clone info.json not found: {info_path}")

    with feat_path.open("rb") as handle:
        feat_data = pickle.load(handle)
    if not isinstance(feat_data, dict):
        raise ValueError(
            "Qwen3.5 voice clone feat.pkl must contain a mapping, got "
            f"{type(feat_data).__name__}: {feat_path}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError(
            "Qwen3.5 voice clone info.json must contain a JSON object, got "
            f"{type(info).__name__}: {info_path}"
        )
    return {
        # Different reference scripts save feat.pkl with slightly different key
        # names. Parse them through _voice_clone_prompt_code so real xvector
        # assets using prompt_speaker_codes/prompt_codes do not silently lose the
        # speaker prefix.
        "prompt_code": _voice_clone_prompt_code(feat_data),
        "talker_system_instruct": _voice_clone_system_instruct(info) or "",
        "language_type": _voice_clone_language(info) or "",
    }


def _voice_clone_prompt_code(raw: dict[str, Any]) -> Any:
    for key in (
        "prompt_code",
        "prompt_speaker_codes",
        "prompt_codes",
        # Older zero-shot xvector assets may use ref_code, and Qwen3.5
        # voice-clone directories may reuse the same feat.pkl layout.
        "ref_code",
        "speaker_codec_codes",
        "voice_clone_codes",
    ):
        value = raw.get(key)
        if value is not None:
            return value
    return None


def _voice_clone_system_instruct(raw: dict[str, Any]) -> Any:
    for key in _VOICE_CLONE_SYSTEM_INSTRUCT_KEYS:
        value = raw.get(key)
        if value is not None and value != "":
            return value
    return None


def _voice_clone_language(raw: dict[str, Any]) -> Any:
    for key in _VOICE_CLONE_LANGUAGE_KEYS:
        value = raw.get(key)
        if value is not None and value != "":
            return value
    return None


def _voice_clone_path(raw: dict[str, Any]) -> str | None:
    for key in _VOICE_CLONE_PATH_KEYS:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _load_voice_map(model_path: str) -> dict[str, str]:
    model_dir = Path(model_path)
    candidates = (model_dir / "voice_map.json", model_dir.parent / "voice_map.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        voice_map: dict[str, str] = {}
        for name, code in data.items():
            if not isinstance(name, str) or not isinstance(code, str):
                continue
            normalized_name = name.lower().strip()
            normalized_code = code.lower().strip()
            if normalized_name and normalized_code:
                voice_map[normalized_name] = normalized_code
        # Real Qwen3.5 weights may ship voice_map.json with code2wav/speaker
        # assets. Prefer the model-provided mapping instead of relying only on
        # hard-coded aliases.
        return voice_map
    return {}


def _resolve_prefixed_sampling_value(
    base_params: dict[str, Any],
    nested_params: dict[str, Any],
    *,
    prefix: str,
    names: tuple[str, ...],
    default: Any,
) -> Any:
    prefixed = tuple(f"{prefix}_{name}" for name in names)
    value = _first_param_value(base_params, prefixed, None)
    if value is not None:
        return value
    return _first_param_value(nested_params, names, default)


def _resolve_prefixed_sampling_seed(
    base_params: dict[str, Any],
    nested_params: dict[str, Any],
    *,
    prefix: str,
    fallback_base_seed: bool = True,
) -> int | None:
    value = _first_param_value(
        base_params,
        (f"{prefix}_seed", f"{prefix}_sampling_seed"),
        None,
    )
    if value is None:
        value = _first_param_value(nested_params, ("seed", "sampling_seed"), None)
    if value is None and fallback_base_seed:
        value = _first_param_value(base_params, ("seed", "sampling_seed"), None)
    if value is None:
        return None
    return int(value)


def _resolve_nested_sampling_params(
    params: dict[str, Any],
    *,
    nested_key: str,
    prefix: str,
    defaults: dict[str, Any],
    fallback_base_seed: bool = True,
) -> dict[str, Any]:
    nested = _as_params_mapping(params.get(nested_key))
    return {
        "max_new_tokens": int(
            _resolve_prefixed_sampling_value(
                params,
                nested,
                prefix=prefix,
                names=("max_new_tokens", "max_completion_tokens", "max_tokens"),
                default=defaults.get("max_new_tokens", 0),
            )
        ),
        "temperature": float(
            _resolve_prefixed_sampling_value(
                params,
                nested,
                prefix=prefix,
                names=("temperature",),
                default=defaults.get("temperature", 0.0),
            )
        ),
        "top_k": int(
            _resolve_prefixed_sampling_value(
                params,
                nested,
                prefix=prefix,
                names=("top_k",),
                default=defaults.get("top_k", 0),
            )
        ),
        "top_p": float(
            _resolve_prefixed_sampling_value(
                params,
                nested,
                prefix=prefix,
                names=("top_p",),
                default=defaults.get("top_p", 1.0),
            )
        ),
        "min_p": float(
            _resolve_prefixed_sampling_value(
                params,
                nested,
                prefix=prefix,
                names=("min_p",),
                default=defaults.get("min_p", 0.0),
            )
        ),
        "repetition_penalty": float(
            _resolve_prefixed_sampling_value(
                params,
                nested,
                prefix=prefix,
                names=("repetition_penalty",),
                default=defaults.get("repetition_penalty", 1.0),
            )
        ),
        "seed": _resolve_prefixed_sampling_seed(
            params,
            nested,
            prefix=prefix,
            fallback_base_seed=fallback_base_seed,
        ),
    }


def _resolve_qwen35_talker_max_new_tokens(
    params: dict[str, Any],
    resolved_talker_max_new_tokens: int,
    *,
    audio_tokens_per_text_token: int = 40,
) -> int:
    base_max_tokens = _first_param_value(
        params,
        ("max_new_tokens", "max_completion_tokens", "max_tokens"),
        None,
    )
    if base_max_tokens is None:
        return int(resolved_talker_max_new_tokens)

    # Qwen3.5 talker uses text-side max_tokens/max_completion_tokens * 40 as the
    # audio codec limit, then takes the minimum with the talker's own limit. This
    # keeps audio bounded when users limit text length without truncating too
    # early with a 1:1 token limit.
    return min(
        int(resolved_talker_max_new_tokens),
        int(base_max_tokens) * int(audio_tokens_per_text_token),
    )


def _to_subtalker_sampling_namespace(config: dict[str, Any]) -> SimpleNamespace:
    # The subtalker samples residual codec groups inside the model and does not
    # need a separate SGLang Req. Keep a lightweight object for decode and
    # code_predictor paths to read.
    return SimpleNamespace(
        temperature=float(config["temperature"]),
        top_k=int(config["top_k"]),
        top_p=float(config["top_p"]),
        min_p=float(config.get("min_p", 0.0)),
        repetition_penalty=float(config.get("repetition_penalty", 1.0)),
        sampling_seed=config.get("seed"),
    )


def _get_vision_end_token_id(config: Any) -> int:
    for name in ("vision_end_token_id", "video_end_token_id"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError("Qwen3.5-Omni config missing vision_end_token_id")


def _normalize_aux_hidden_key(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            return None
        piece = pieces[-1]
        return int(piece) if piece.lstrip("-").isdigit() else piece
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            normalized = _normalize_aux_hidden_key(item)
            if normalized is not None:
                return normalized
        return None
    return None


def _hidden_key_candidates(key: int | str | None) -> tuple[Any, ...]:
    if key is None:
        return ()
    candidates: list[Any] = [key]
    if isinstance(key, int):
        candidates.append(str(key))
        if key == 0:
            candidates.append("embed")
    elif isinstance(key, str):
        if key.lstrip("-").isdigit():
            candidates.append(int(key))
        if key == "0":
            candidates.append("embed")
    return tuple(dict.fromkeys(candidates))


def _select_hidden_by_key(
    hidden: dict[Any, torch.Tensor],
    key: int | str | None,
) -> torch.Tensor | None:
    for candidate in _hidden_key_candidates(key):
        value = hidden.get(candidate)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _normalize_chunk_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
    if hidden is None:
        return None
    if hidden.ndim == 1:
        return hidden
    if hidden.ndim == 2:
        return hidden[0]
    return None


def _make_text_positions(length: int, start: int) -> torch.Tensor:
    return torch.arange(length, dtype=torch.long).view(1, -1).expand(3, -1) + start


def _make_vision_positions(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    start: int,
) -> torch.Tensor:
    t_index = (
        torch.arange(grid_t, dtype=torch.long)
        .view(-1, 1)
        .expand(-1, grid_h * grid_w)
        .reshape(-1)
    )
    h_index = (
        torch.arange(grid_h, dtype=torch.long)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .reshape(-1)
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .reshape(-1)
    )
    return torch.stack([t_index, h_index, w_index]) + start


def _compute_qwen35_mrope_positions(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compute Qwen3.5-Omni Next M-RoPE positions.

    Qwen3.5 video grids are first expanded along the temporal dimension into
    per-frame grids, then an explicit vision_end position is appended according
    to the Next thinker/talker 3D RoPE rules.
    """
    image_grid_thw = _as_grid_list(model_inputs.get("image_grid_thw"))
    raw_video_grid_thw = _as_grid_list(model_inputs.get("video_grid_thw"))
    if not image_grid_thw and not raw_video_grid_thw:
        if _has_audio_mrope_inputs(model_inputs):
            # Audio-only requests do not have Qwen3.5 Next per-frame vision
            # grids. Reuse the existing Qwen3 audio M-RoPE logic so pure-audio
            # requests do not miss positions.
            base_result = qwen3_request_builders._compute_mrope_positions(
                input_ids,
                model_inputs,
                _audio_mrope_compat_config(thinker_config),
            )
            if base_result is None:
                return None
            positions, position_delta = base_result
            return positions, _mrope_delta_tensor(position_delta)
        return None

    video_grid_thw = [
        [1, int(grid_h), int(grid_w)]
        for grid_t, grid_h, grid_w in raw_video_grid_thw
        for _ in range(int(grid_t))
    ]

    input_tokens = [
        int(token_id)
        for token_id in input_ids.reshape(-1).detach().cpu().tolist()
    ]
    if not input_tokens:
        return None

    vision_config = getattr(thinker_config, "vision_config", thinker_config)
    spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
    if spatial_merge_size <= 0:
        raise ValueError("Qwen3.5-Omni spatial_merge_size must be positive")

    image_token_id = _get_token_id(thinker_config, "image_token_id")
    video_token_id = _get_token_id(thinker_config, "video_token_id")
    audio_token_id = _get_token_id(thinker_config, "audio_token_id")
    vision_start_token_id = _get_token_id(thinker_config, "vision_start_token_id")
    vision_end_token_id = _get_vision_end_token_id(thinker_config)

    vision_followers = [
        input_tokens[index + 1]
        for index, token_id in enumerate(input_tokens[:-1])
        if token_id == vision_start_token_id
    ]
    image_count = sum(token_id == image_token_id for token_id in vision_followers)
    video_count = sum(token_id == video_token_id for token_id in vision_followers)
    if image_count + video_count == 0:
        return None

    image_index = 0
    video_index = 0
    remaining_images = image_count
    remaining_videos = video_count
    start = 0
    position_chunks: list[torch.Tensor] = []

    for _ in range(image_count + video_count):
        image_pos = (
            _index_after(input_tokens, image_token_id, start)
            if remaining_images > 0
            else None
        )
        video_pos = (
            _index_after(input_tokens, video_token_id, start)
            if remaining_videos > 0
            else None
        )
        if image_pos is None and video_pos is None:
            break

        is_image = video_pos is None or (
            image_pos is not None and image_pos < video_pos
        )
        if is_image:
            grid = image_grid_thw[image_index]
            image_index += 1
            remaining_images -= 1
            token_pos = int(image_pos)
            media_token_id = image_token_id
        else:
            grid = video_grid_thw[video_index]
            video_index += 1
            remaining_videos -= 1
            token_pos = int(video_pos)
            media_token_id = video_token_id

        grid_t, grid_h, grid_w = (int(part) for part in grid)
        grid_h //= spatial_merge_size
        grid_w //= spatial_merge_size
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"invalid Qwen3.5-Omni grid after merge: {grid}")

        text_len = token_pos - start
        next_pos = (
            int(position_chunks[-1].max().item()) + 1
            if position_chunks
            else 0
        )
        if text_len > 0:
            position_chunks.append(_make_text_positions(text_len, next_pos))

        vision_start_pos = next_pos + text_len
        vision_positions = _make_vision_positions(
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            start=vision_start_pos,
        )
        position_chunks.append(vision_positions)
        vision_end_pos = int(vision_positions.max().item()) + 1
        position_chunks.append(
            torch.full((3, 1), vision_end_pos, dtype=torch.long)
        )

        grid_tokens = int(grid_t * grid_h * grid_w)
        audio_len = 0
        if media_token_id == video_token_id:
            expected_end = token_pos + grid_tokens
            actual_end = _index_after(input_tokens, vision_end_token_id, start)
            if actual_end == expected_end:
                audio_len = _count_consecutive_tokens(
                    input_tokens,
                    token_id=audio_token_id,
                    start=actual_end + 1,
                )
        if audio_len > 0:
            position_chunks.append(
                _make_text_positions(audio_len, vision_start_pos)
            )

        start = token_pos + grid_tokens + 1 + audio_len

    if start < len(input_tokens):
        next_pos = (
            int(position_chunks[-1].max().item()) + 1
            if position_chunks
            else 0
        )
        position_chunks.append(
            _make_text_positions(len(input_tokens) - start, next_pos)
        )

    if not position_chunks:
        return None

    positions = torch.cat(position_chunks, dim=1).reshape(3, -1)
    if positions.shape[1] != len(input_tokens):
        raise ValueError(
            "Qwen3.5-Omni M-RoPE position length mismatch: "
            f"positions={positions.shape[1]} input={len(input_tokens)}"
        )
    position_delta = _mrope_delta_tensor(
        int(positions.max().item() + 1 - len(input_tokens))
    )
    return positions, position_delta


def _mrope_delta_tensor(position_delta: Any) -> torch.Tensor:
    if isinstance(position_delta, torch.Tensor):
        return position_delta.to(dtype=torch.long).reshape(-1)
    return torch.tensor([int(position_delta)], dtype=torch.long)


def _has_audio_mrope_inputs(model_inputs: dict[str, Any]) -> bool:
    audio_feature_lengths = model_inputs.get("audio_feature_lengths")
    if audio_feature_lengths is None:
        return False
    if isinstance(audio_feature_lengths, torch.Tensor):
        return audio_feature_lengths.numel() > 0
    if isinstance(audio_feature_lengths, (list, tuple)):
        return len(audio_feature_lengths) > 0
    return True


def _audio_mrope_compat_config(thinker_config: Any) -> SimpleNamespace:
    vision_config = getattr(thinker_config, "vision_config", SimpleNamespace())
    audio_token_id = _get_token_id(thinker_config, "audio_token_id")
    return SimpleNamespace(
        image_token_id=_get_token_id(thinker_config, "image_token_id"),
        video_token_id=_get_token_id(thinker_config, "video_token_id"),
        vision_start_token_id=_get_token_id(thinker_config, "vision_start_token_id"),
        audio_token_id=audio_token_id,
        audio_start_token_id=int(
            getattr(thinker_config, "audio_start_token_id", audio_token_id)
        ),
        audio_end_token_id=int(
            getattr(thinker_config, "audio_end_token_id", audio_token_id)
        ),
        position_id_per_seconds=int(
            getattr(thinker_config, "position_id_per_seconds", 1)
        ),
        vision_config=SimpleNamespace(
            spatial_merge_size=int(getattr(vision_config, "spatial_merge_size", 1)),
            tokens_per_second=getattr(vision_config, "tokens_per_second", None),
        ),
    )


def _normalize_deepstack_layers(value: Any) -> list[torch.Tensor]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            return [value[idx] for idx in range(value.shape[0])]
        if value.ndim == 2:
            return [value]
        return []
    if isinstance(value, (list, tuple)):
        layers: list[torch.Tensor] = []
        for item in value:
            layers.extend(_normalize_deepstack_layers(item))
        return layers
    return []


def _count_feature_rows(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1, value.shape[-1]).shape[0]) if value.ndim else 0
    if isinstance(value, (list, tuple)):
        total = 0
        saw_tensor = False
        for item in value:
            rows = _count_feature_rows(item)
            if rows is None:
                continue
            saw_tensor = True
            total += rows
        return total if saw_tensor else None
    return None


def _validate_feature_rows(
    *,
    model_inputs: dict[str, Any],
    input_ids: torch.Tensor,
    thinker_config: Any,
    feature_key: str,
    token_attr: str,
    modality: str,
) -> None:
    rows = _count_feature_rows(model_inputs.get(feature_key))
    if rows is None:
        return
    token_id = getattr(thinker_config, token_attr, None)
    if token_id is None:
        return
    expected = int((input_ids == int(token_id)).sum().item())
    if rows != expected:
        raise ValueError(
            f"Qwen3.5 {modality} feature/token mismatch: "
            f"{feature_key} rows={rows}, prompt {token_attr} count={expected}"
        )


def _validate_qwen35_multimodal_feature_lengths(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> None:
    input_ids = input_ids.to(dtype=torch.long).reshape(-1)
    # External encoders have already produced image/video/audio features.
    # Validate placeholder counts before entering SGLang AR so real models catch
    # processor/encoder length mismatches earlier, instead of failing later
    # during embedding scatter.
    for feature_key, token_attr, modality in (
        ("image_embeds", "image_token_id", "image"),
        ("video_embeds", "video_token_id", "video"),
        ("audio_embeds", "audio_token_id", "audio"),
    ):
        _validate_feature_rows(
            model_inputs=model_inputs,
            input_ids=input_ids,
            thinker_config=thinker_config,
            feature_key=feature_key,
            token_attr=token_attr,
            modality=modality,
        )
    _validate_qwen35_multimodal_metadata_lengths(model_inputs)


def _env_flag_enabled(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or raw == "":
        return default
    return raw not in {"0", "false", "False", "no", "NO", "off", "OFF"}


def _omit_cached_visual_item_payloads_enabled() -> bool:
    return _env_flag_enabled(
        os.getenv("SGLANG_OMNI_OMIT_CACHED_VISUAL_ITEM_PAYLOADS"),
    )


def _limit_prefix_cache_before_media_enabled(request: Any | None = None) -> bool:
    raw = os.getenv("QWEN35_LIMIT_PREFIX_CACHE_BEFORE_MEDIA")
    if raw is not None:
        return _env_flag_enabled(raw)
    # RTC pre-runs intentionally build growing multimedia prefixes. Let those
    # prefixes populate SGLang's radix cache even when cached visual payloads
    # are omitted from later requests; pad_values keep media tokens stable
    # across the session, and embed injection skips already-cached rows.
    return False


def _mamba_media_branching_cache_enabled(request: Any | None = None) -> bool:
    raw = os.getenv("QWEN35_MAMBA_MEDIA_BRANCH_CACHE")
    if raw is None and (
        request is not None
        and (_is_qwen35_rtc_prerun(request) or _is_qwen35_rtc_actual(request))
        and _omit_cached_visual_item_payloads_enabled()
    ):
        return False
    return _env_flag_enabled(raw, default=True)


def _rtc_prerun_prefill_only_enabled() -> bool:
    raw = os.getenv("QWEN35_RTC_PRERUN_PREFILL_ONLY")
    return _env_flag_enabled(raw, default=True)


def _rtc_isolate_prerun_prefill_enabled() -> bool:
    raw = os.getenv("QWEN35_RTC_ISOLATE_PRERUN_PREFILL")
    return _env_flag_enabled(raw, default=True)


def _rtc_prerun_thinker_terminal_enabled() -> bool:
    raw = os.getenv("QWEN35_RTC_PRERUN_THINKER_TERMINAL")
    return _env_flag_enabled(raw, default=False)


def _is_qwen35_rtc_prerun(request: Any) -> bool:
    metadata = getattr(request, "metadata", None)
    return isinstance(metadata, dict) and bool(metadata.get("pre_run"))


def _is_qwen35_rtc_prefill_only_prerun(request: Any) -> bool:
    return _rtc_prerun_prefill_only_enabled() and _is_qwen35_rtc_prerun(request)


def _is_qwen35_rtc_thinker_terminal_prerun(request: Any) -> bool:
    return (
        _is_qwen35_rtc_prefill_only_prerun(request)
        and _rtc_prerun_thinker_terminal_enabled()
    )


def _is_qwen35_rtc_actual(request: Any) -> bool:
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, dict) or bool(metadata.get("pre_run")):
        return False
    namespace = metadata.get("media_cache_namespace")
    return isinstance(namespace, str) and namespace.startswith("rtc:")


def _params_for_qwen35_prerun(params: dict[str, Any], request: Any) -> dict[str, Any]:
    if not _is_qwen35_rtc_prefill_only_prerun(request):
        return params

    # RTC pre_run is a cache-warm/prefix-extension request. Letting it generate
    # tokens commits non-prompt tokens into the hybrid Mamba radix cache, which
    # can poison later growing-trunk prefix hits under concurrency.
    adjusted = dict(params)
    adjusted["max_tokens"] = 0
    adjusted["max_completion_tokens"] = 0
    adjusted["max_new_tokens"] = 0
    return adjusted


def _sequence_length(value: Any) -> int | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError):
            return None
    try:
        return len(value)
    except TypeError:
        return None


def _qwen35_prerun_terminal_result(data: Any) -> dict[str, Any]:
    prompt_tokens = _sequence_length(getattr(data, "input_ids", None))
    completion_tokens = _sequence_length(getattr(data, "output_ids", None)) or 0
    usage: dict[str, int] = {"completion_tokens": completion_tokens}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
        usage["total_tokens"] = prompt_tokens + completion_tokens

    return {
        "text": "",
        "finish_reason": getattr(data, "finish_reason", None) or "stop",
        "usage": usage,
    }


def _metadata_length(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def _validate_qwen35_multimodal_metadata_lengths(
    model_inputs: dict[str, Any],
) -> None:
    audio_slots = _metadata_length(model_inputs.get("audio_feature_lengths"))
    dependent_slots = _metadata_length(model_inputs.get("audio_is_dependent"))
    if (
        audio_slots is not None
        and dependent_slots is not None
        and audio_slots != dependent_slots
    ):
        raise ValueError(
            "Qwen3.5 audio_is_dependent length mismatch: "
            f"audio slots={audio_slots}, mask slots={dependent_slots}"
        )

    use_audio_in_video = model_inputs.get("use_audio_in_video")
    if use_audio_in_video is None or isinstance(use_audio_in_video, bool):
        return
    flag_slots = _metadata_length(use_audio_in_video)
    if flag_slots in (None, 1):
        return
    video_slots = len(_as_grid_list(model_inputs.get("video_grid_thw")))
    if video_slots and flag_slots != video_slots:
        # Qwen3.5 supports per-video use_audio_in_video. If lengths diverge it
        # becomes ambiguous which video should consume audio, so fail before
        # entering AR.
        raise ValueError(
            "Qwen3.5 use_audio_in_video length mismatch: "
            f"video slots={video_slots}, flag slots={flag_slots}"
        )


def _scatter_deepstack_layers(
    *,
    output_layers: dict[int, torch.Tensor],
    layers: list[torch.Tensor],
    positions: torch.Tensor,
    seq_len: int,
) -> None:
    if not layers or positions.numel() == 0:
        return
    for layer_idx, layer_tensor in enumerate(layers):
        layer_tensor = layer_tensor.reshape(-1, layer_tensor.shape[-1])
        if layer_tensor.shape[0] != positions.numel():
            raise ValueError(
                "Qwen3.5 deepstack token count mismatch: "
                f"layer={layer_idx} features={layer_tensor.shape[0]} "
                f"positions={positions.numel()}"
            )
        full = output_layers.get(layer_idx)
        if full is None:
            full = layer_tensor.new_zeros((seq_len, layer_tensor.shape[-1]))
            output_layers[layer_idx] = full
        full.index_copy_(0, positions.to(device=full.device), layer_tensor)


def _build_qwen35_deepstack_input_embeds(
    input_ids: torch.Tensor,
    model_inputs: dict[str, Any],
    thinker_config: Any,
) -> dict[str, torch.Tensor]:
    input_ids = input_ids.to(dtype=torch.long).reshape(-1)
    seq_len = int(input_ids.numel())
    if seq_len == 0:
        return {}

    image_token_id = getattr(thinker_config, "image_token_id", None)
    video_token_id = getattr(thinker_config, "video_token_id", None)
    image_positions = (
        (input_ids == int(image_token_id)).nonzero(as_tuple=True)[0]
        if image_token_id is not None
        else input_ids.new_empty((0,))
    )
    video_positions = (
        (input_ids == int(video_token_id)).nonzero(as_tuple=True)[0]
        if video_token_id is not None
        else input_ids.new_empty((0,))
    )

    image_layers = _normalize_deepstack_layers(
        model_inputs.get(
            "image_deepstack_visual_embeds",
            model_inputs.get("deepstack_visual_embeds_image"),
        )
    )
    video_layers = _normalize_deepstack_layers(
        model_inputs.get(
            "video_deepstack_visual_embeds",
            model_inputs.get("deepstack_visual_embeds_video"),
        )
    )
    shared_layers = _normalize_deepstack_layers(
        model_inputs.get("deepstack_visual_embeds")
    )
    if shared_layers and not image_layers and image_positions.numel() > 0:
        image_layers = shared_layers
    elif shared_layers and not video_layers and video_positions.numel() > 0:
        video_layers = shared_layers

    layers: dict[int, torch.Tensor] = {}
    _scatter_deepstack_layers(
        output_layers=layers,
        layers=image_layers,
        positions=image_positions,
        seq_len=seq_len,
    )
    _scatter_deepstack_layers(
        output_layers=layers,
        layers=video_layers,
        positions=video_positions,
        seq_len=seq_len,
    )
    return {
        f"deepstack_input_embeds_{layer_idx}": layers[layer_idx]
        for layer_idx in sorted(layers)
    }


def _prepare_qwen35_deepstack_inputs(
    state: Qwen3OmniPipelineState,
    thinker_config: Any,
) -> None:
    prompt = state.prompt or {}
    input_ids = prompt.get("input_ids")
    if input_ids is None or thinker_config is None:
        return

    thinker_inputs = dict(state.thinker_inputs or {})
    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        return

    deepstack_input_embeds = _build_qwen35_deepstack_input_embeds(
        torch.as_tensor(input_ids),
        model_inputs,
        thinker_config,
    )
    if not deepstack_input_embeds:
        return

    # The external vision encoder has already computed multiscale visual
    # features. Scatter them into full deepstack_input_embeds at image/video
    # token positions in the prompt so the thinker forward receives more than
    # just the main vision embedding.
    model_inputs["deepstack_input_embeds"] = deepstack_input_embeds
    for key in (
        "deepstack_visual_embeds",
        "image_deepstack_visual_embeds",
        "video_deepstack_visual_embeds",
        "deepstack_visual_embeds_image",
        "deepstack_visual_embeds_video",
    ):
        model_inputs.pop(key, None)
    thinker_inputs["model_inputs"] = model_inputs
    state.thinker_inputs = thinker_inputs


def _prepare_qwen35_thinker_inputs(
    state: Qwen3OmniPipelineState,
    thinker_config: Any,
) -> None:
    prompt = state.prompt or {}
    input_ids = prompt.get("input_ids")
    if input_ids is None or thinker_config is None:
        return
    thinker_inputs = state.thinker_inputs or {}
    model_inputs = thinker_inputs.get("model_inputs", {})
    if isinstance(model_inputs, dict):
        _validate_qwen35_multimodal_feature_lengths(
            torch.as_tensor(input_ids),
            model_inputs,
            thinker_config,
        )
    _prepare_qwen35_deepstack_inputs(state, thinker_config)


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    thinker_config: Any = None,
    stage_name: str = "thinker",
    mrope_position_builder: Callable[
        [torch.Tensor, dict[str, Any], Any], Any
    ] | None = None,
):
    """Build Qwen3.5 thinker adapters with Next M-RoPE positions."""
    qwen35_mrope_builder = mrope_position_builder or _compute_qwen35_mrope_positions

    def request_builder(payload):
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        _prepare_qwen35_thinker_inputs(state, thinker_config)
        params = _params_for_qwen35_prerun(
            payload.request.params or {},
            payload.request,
        )
        req_data = qwen3_request_builders.build_sglang_thinker_request(
            state,
            params=params,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            request_id=payload.request_id,
            thinker_config=thinker_config,
            mrope_position_builder=qwen35_mrope_builder,
            limit_prefix_cache_before_media=_limit_prefix_cache_before_media_enabled(
                payload.request
            ),
            mamba_media_branching_cache=_mamba_media_branching_cache_enabled(
                payload.request
            ),
        )
        req = getattr(req_data, "req", None)
        if (
            _rtc_isolate_prerun_prefill_enabled()
            and _is_qwen35_rtc_prerun(payload.request)
            and req is not None
        ):
            req._omni_isolate_prefill_batch = True
        if _is_qwen35_rtc_actual(payload.request) and req is not None:
            req._omni_prioritize_prefill = True
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data):
        payload = data.stage_payload
        if _is_qwen35_rtc_thinker_terminal_prerun(payload.request):
            return payload.__class__(
                request_id=payload.request_id,
                request=payload.request,
                data=_qwen35_prerun_terminal_result(data),
            )
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        qwen3_request_builders.apply_thinker_result(
            state,
            stage_name=stage_name,
            result=data,
        )
        return payload.__class__(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


def make_thinker_stream_output_builder(required_aux_hidden_key: Any = None):
    aux_hidden_key = _normalize_aux_hidden_key(required_aux_hidden_key)
    decode_stream_batcher = qwen3_request_builders._DecodeStreamTokenBatcher()

    def _select_stream_hidden_fallback(extra: dict[str, Any]) -> torch.Tensor | None:
        hidden = extra.get("stream_hidden_states")
        if isinstance(hidden, torch.Tensor):
            return _normalize_chunk_hidden(hidden)
        return None

    def _split_qwen35_hidden(
        hidden: dict[Any, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if isinstance(hidden, torch.Tensor):
            return _normalize_chunk_hidden(hidden), None

        embed = _select_hidden_by_key(hidden, 0)
        layer_hidden = _select_hidden_by_key(hidden, aux_hidden_key)
        if layer_hidden is None:
            # If the model config or test stub lacks the target layer, fall back
            # to the original Qwen3 logic and take the first non-embed hidden
            # state so the speech path still has output.
            for key, value in hidden.items():
                if key in ("embed", 0, "0"):
                    continue
                if isinstance(value, torch.Tensor):
                    layer_hidden = value
                    break
        return _normalize_chunk_hidden(embed), _normalize_chunk_hidden(layer_hidden)

    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = getattr(req_data, "req", None)
        if req is not None and int(getattr(req, "is_chunked", 0) or 0) > 0:
            return []
        if req_output.data is None:
            return []

        token_id = int(req_output.data)
        messages: list[OutgoingMessage] = []

        stage_payload = req_data.stage_payload
        is_streaming = bool(
            stage_payload is not None
            and (stage_payload.request.params or {}).get("stream", False)
        )
        if is_streaming:
            decode_msg = decode_stream_batcher.build(request_id, token_id)
            if decode_msg is not None:
                messages.append(decode_msg)

        if not should_generate_audio_output(stage_payload):
            return messages

        extra = req_output.extra
        if isinstance(extra, dict):
            embed = None
            layer_hidden = None
            if "hidden_states" in extra:
                embed, layer_hidden = _split_qwen35_hidden(extra["hidden_states"])
            if embed is None:
                # SGLang may return the current token stream hidden separately.
                # Use it as a fallback when aux hidden lacks embed, avoiding
                # treating an intermediate layer hidden state as the talker's
                # text/embed hidden state.
                embed = _select_stream_hidden_fallback(extra)
            if embed is not None:
                metadata = {"token_id": token_id}
                if layer_hidden is not None:
                    metadata["layer_hidden"] = layer_hidden
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=embed,
                        target="talker_ar",
                        metadata=metadata,
                    )
                )
            elif layer_hidden is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=layer_hidden,
                        target="talker_ar",
                        metadata={"token_id": token_id},
                    )
                )

        return messages

    return _build_stream_output


def _build_qwen35_assistant_part(
    *,
    assistant_embed: torch.Tensor,
    assistant_instruct_embed: torch.Tensor | None = None,
    text_projection,
    codec_embed_fn,
    tts_bos_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    speaker_id: int | None,
    codec_nothink_id: int,
    codec_think_id: int | None,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    tts_pad_token_id: int,
    language_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build Qwen3.5 assistant rows in the omni3_5 layout."""
    device = assistant_embed.device
    dtype = assistant_embed.dtype
    projected = text_projection(assistant_embed)
    if assistant_instruct_embed is None:
        assistant_instruct_embed = torch.empty(
            (0, projected.shape[-1]),
            device=device,
            dtype=dtype,
        )
    else:
        assistant_instruct_embed = assistant_instruct_embed.to(
            device=device,
            dtype=dtype,
        )

    if language_id is not None and language_id >= 0 and codec_think_id is not None:
        leading_codec_ids = (
            int(codec_think_id),
            int(codec_think_bos_id),
            int(language_id),
            int(codec_think_eos_id),
        )
    else:
        leading_codec_ids = (
            int(codec_nothink_id),
            int(codec_think_bos_id),
            int(codec_think_eos_id),
        )
    del speaker_id, codec_pad_id

    codec_special_ids = torch.tensor(
        [*leading_codec_ids],
        device=device,
        dtype=torch.long,
    )
    spoken_rows = (
        projected[3:]
        if projected.shape[0] > 3
        else projected.new_empty((0, projected.shape[-1]))
    )
    initial_text_rows = spoken_rows[:_QWEN35_TALKER_NUM_OUTPUT_IN_CHUNK]
    future_text_rows = spoken_rows[_QWEN35_TALKER_NUM_OUTPUT_IN_CHUNK:]

    text_parts = [
        projected[:3],
        assistant_instruct_embed,
        codec_embed_fn(codec_special_ids),
        tts_bos_embed,
        codec_embed_fn(
            torch.tensor([int(codec_bos_id)], device=device, dtype=torch.long)
        ),
        initial_text_rows,
    ]
    assistant_text_hidden = torch.cat(
        [part.to(device=device, dtype=dtype) for part in text_parts if part.shape[0] > 0],
        dim=0,
    )
    input_ids = torch.full(
        (assistant_text_hidden.shape[0],),
        tts_pad_token_id,
        dtype=torch.long,
        device=device,
    )

    if future_text_rows.shape[0] > 0:
        future_text_rows = torch.cat([future_text_rows, tts_eos_embed], dim=0)
    else:
        future_text_rows = tts_eos_embed.clone()
    return {
        "input_embeds": assistant_text_hidden,
        "input_ids": input_ids,
        "future_text_rows": future_text_rows,
    }


class Qwen35TalkerPrefillBuilder(TalkerPrefillBuilder):
    """Qwen3.5 text-side prefill uses talker's own text embedding table."""

    def __init__(
        self,
        *,
        nl_token_id: int = 198,
        codec_eos_id: int | None = None,
        codec_think_id: int | None = None,
        talker_language_id: dict[str, int] | None = None,
        talker_assistant_prompt_id_mapping: dict[str, list[int]] | None = None,
        speaker_system_prompt_id: dict[str, list[int]] | None = None,
        max_thinker_to_talker_mm_tokens: int | None = None,
        tokenizer: Any | None = None,
        thinker_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._thinker_config = thinker_config
        self._voice_clone_cache: dict[str, dict[str, Any]] = {}
        self._voice_map = _load_voice_map(self._model_path)
        self._nl_token_id = int(nl_token_id)
        self._codec_eos_id = (
            int(codec_eos_id) if codec_eos_id is not None else None
        )
        self._codec_think_id = (
            int(codec_think_id) if codec_think_id is not None else None
        )
        self._talker_language_id: dict[str, int] = {}
        for name, token_id in (talker_language_id or {}).items():
            candidates = _language_key_candidates(name)
            if not candidates:
                continue
            self._talker_language_id[candidates[0]] = int(token_id)
            for alias in candidates[1:]:
                self._talker_language_id.setdefault(alias, int(token_id))
        self._talker_assistant_prompt_id_mapping = {
            str(name).lower(): [int(token_id) for token_id in token_ids]
            for name, token_ids in (
                talker_assistant_prompt_id_mapping or {}
            ).items()
        }
        self._speaker_system_prompt_id = {
            str(name).lower(): [int(token_id) for token_id in token_ids]
            for name, token_ids in (speaker_system_prompt_id or {}).items()
        }
        self._max_thinker_to_talker_mm_tokens = (
            int(max_thinker_to_talker_mm_tokens)
            if max_thinker_to_talker_mm_tokens is not None
            else None
        )

    def _normalize_voice_clone_params(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        info = _first_present_param(
            params,
            ("xvector_info", "voice_clone_info"),
        )
        if info is None:
            return params

        if isinstance(info, str):
            if info not in self._voice_clone_cache:
                self._voice_clone_cache[info] = _load_voice_clone_info_from_path(info)
            raw = copy.deepcopy(self._voice_clone_cache[info])
        elif isinstance(info, dict):
            info_path = _voice_clone_path(info)
            if info_path:
                if info_path not in self._voice_clone_cache:
                    self._voice_clone_cache[info_path] = (
                        _load_voice_clone_info_from_path(info_path)
                    )
                raw = copy.deepcopy(self._voice_clone_cache[info_path])
                raw.update(
                    {
                        key: value
                        for key, value in info.items()
                        if key not in _VOICE_CLONE_PATH_KEYS
                    }
                )
            else:
                raw = dict(info)
        else:
            raise ValueError(
                "Qwen3.5 voice clone xvector_info/voice_clone_info must be "
                f"a dict or path string, got {type(info).__name__}"
            )

        normalized = dict(params)
        prompt_code = _voice_clone_prompt_code(raw)
        if prompt_code is not None and not any(
            key in normalized
            for key in (
                "prompt_speaker_codes",
                "speaker_codec_codes",
                "voice_clone_codes",
            )
        ):
            normalized["prompt_speaker_codes"] = prompt_code

        system_instruct = _voice_clone_system_instruct(raw)
        if system_instruct is not None and not any(
            key in normalized
            for key in (
                "system_instruct_ids",
                "talker_system_instruct_ids",
                "speaker_system_instruct_ids",
            )
        ):
            normalized["talker_system_instruct"] = system_instruct

        language = _voice_clone_language(raw)
        if language and not _has_language_param_value(normalized):
            # Language stored in voice-clone assets is only a fallback; explicit
            # user-provided language_id/target_lang/translation_options must
            # take precedence.
            normalized["language"] = language

        # xvector_info does not add a separate scheduling branch; later stages
        # still reuse the speaker prefix from prompt_speaker_codes and
        # system_instruct_ids.
        return normalized

    def _load_prompt_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_text_ids = getattr(self._model, "embed_text_ids", None)
        if embed_text_ids is None:
            return super()._load_prompt_token_embeddings(token_ids)

        token_ids = token_ids.to(
            device=self._device,
            dtype=torch.long,
        ).view(-1)
        return embed_text_ids(token_ids).to(device=self._device, dtype=self._dtype)

    def _load_thinker_hidden_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Load thinker-dimension embeddings for prompt_hidden (2048d)."""
        return super()._load_prompt_token_embeddings(token_ids)

    @staticmethod
    def _signed_int64_aliases(value: Any) -> tuple[int, ...]:
        if isinstance(value, torch.Tensor):
            aliases: list[int] = []
            for item in value.detach().cpu().reshape(-1).tolist():
                aliases.extend(Qwen35TalkerPrefillBuilder._signed_int64_aliases(item))
            return tuple(dict.fromkeys(aliases))
        if isinstance(value, (list, tuple, set)):
            aliases = []
            for item in value:
                aliases.extend(Qwen35TalkerPrefillBuilder._signed_int64_aliases(item))
            return tuple(dict.fromkeys(aliases))
        try:
            raw = int(value.item() if hasattr(value, "item") else value)
        except (TypeError, ValueError, OverflowError):
            return ()
        wrapped = ((raw + (1 << 63)) % (1 << 64)) - (1 << 63)
        return (raw,) if wrapped == raw else (raw, wrapped)

    def _talker_text_vocab_size(self) -> int | None:
        text_embedding = getattr(self._model, "text_embedding", None)
        for attr in ("num_embeddings", "org_num_embeddings"):
            value = getattr(text_embedding, attr, None)
            if value is not None:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return parsed

        for config in (
            getattr(self._model, "text_config", None),
            getattr(getattr(self._model, "config", None), "text_config", None),
        ):
            if config is None:
                continue
            for attr in ("text_vocab_size", "vocab_size"):
                value = getattr(config, attr, None)
                if value is None and isinstance(config, dict):
                    value = config.get(attr)
                if value is None:
                    continue
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return parsed
        return None

    def _invalid_prompt_id_mask(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        invalid = prompt_ids < 0
        vocab_size = self._talker_text_vocab_size()
        if vocab_size is not None:
            invalid = invalid | (prompt_ids >= int(vocab_size))
        else:
            # Media cache pad values are xxhash-derived int64s. In unit tests
            # and odd configs where vocab metadata is unavailable, still catch
            # the wrapped/large values without treating normal tokenizer ids as
            # media placeholders.
            invalid = invalid | (prompt_ids.abs() >= (1 << 31))
        return invalid

    def _fallback_prompt_text_token_id(self) -> int:
        vocab_size = self._talker_text_vocab_size()
        candidates = [
            getattr(self._tokenizer, "pad_token_id", None),
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._tokenizer, "bos_token_id", None),
            0,
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                token_id = int(value)
            except (TypeError, ValueError, RuntimeError):
                continue
            if token_id < 0:
                continue
            if vocab_size is not None and token_id >= int(vocab_size):
                continue
            return token_id
        return 0

    @staticmethod
    def _distribute_prompt_extra(total: int, slots: int) -> list[int]:
        if total <= 0 or slots <= 0:
            return [0] * max(slots, 0)
        base, remainder = divmod(int(total), int(slots))
        return [base + (1 if idx < remainder else 0) for idx in range(slots)]

    @staticmethod
    def _slot_lengths_from_slot_count(
        missing_by_modality: dict[str, int],
        slot_count: int,
    ) -> dict[str, list[int]] | None:
        if slot_count <= 0:
            return None
        inferred: dict[str, list[int]] = {}
        for modality, rows in missing_by_modality.items():
            if rows <= 0 or rows % slot_count != 0:
                return None
            inferred[modality] = [rows // slot_count] * slot_count
        return inferred

    @classmethod
    def _infer_rtc_media_slot_lengths(
        cls,
        *,
        prompt_len: int,
        missing_by_modality: dict[str, int],
    ) -> dict[str, list[int]] | None:
        if set(missing_by_modality) != {"audio", "video"}:
            return None

        total_media = sum(missing_by_modality.values())
        extra_total = int(prompt_len) - int(total_media)
        if extra_total < 0:
            return None

        candidates: list[tuple[int, int]] = []

        def add_candidate(priority: int, slot_count: int) -> None:
            if slot_count <= 0:
                return
            if extra_total and extra_total % slot_count != 0:
                return
            if not cls._slot_lengths_from_slot_count(
                missing_by_modality,
                slot_count,
            ):
                return
            candidates.append((priority, slot_count))

        if extra_total > 0:
            add_candidate(
                0,
                extra_total // _QWEN35_RTC_TEXT_FILLER_PER_CHUNK,
            )

        audio_rows = missing_by_modality.get("audio", 0)
        if audio_rows > 0:
            add_candidate(
                1,
                audio_rows // _QWEN35_RTC_AUDIO_ROWS_PER_CHUNK,
            )

        if not candidates:
            return None

        candidates = sorted(set(candidates), key=lambda item: item[0])
        return cls._slot_lengths_from_slot_count(
            missing_by_modality,
            candidates[0][1],
        )

    @staticmethod
    def _feature_rows_for_prompt_model_inputs(
        prompt_model_inputs: dict[str, Any],
        feature_key: str,
    ) -> int | None:
        feature_tensor = coerce_feature_tensor(prompt_model_inputs.get(feature_key))
        if feature_tensor is None:
            return None
        return int(feature_tensor.reshape(-1, feature_tensor.shape[-1]).shape[0])

    @staticmethod
    def _int_sequence(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            result: list[int] = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    result.extend(
                        int(part) for part in item.detach().cpu().reshape(-1).tolist()
                    )
                    continue
                if isinstance(item, (list, tuple)):
                    result.extend(Qwen35TalkerPrefillBuilder._int_sequence(item))
                    continue
                try:
                    result.append(int(item))
                except (TypeError, ValueError):
                    continue
            return result
        try:
            return [int(value)]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _matching_original_prompt_ids(
        prompt_model_inputs: dict[str, Any],
        prompt_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        for key in ("original_input_ids", "prompt_original_input_ids"):
            value = prompt_model_inputs.get(key)
            if value is None:
                continue
            try:
                original = torch.as_tensor(value, dtype=torch.long).reshape(-1).cpu()
            except (TypeError, ValueError):
                continue
            if original.numel() == prompt_ids.numel():
                return original
        return None

    def _feature_rows_by_modality(
        self,
        prompt_model_inputs: dict[str, Any],
    ) -> dict[str, int]:
        rows_by_modality: dict[str, int] = {}
        for modality, feature_key in (
            ("audio", "audio_embeds"),
            ("image", "image_embeds"),
            ("video", "video_embeds"),
        ):
            rows = self._feature_rows_for_prompt_model_inputs(
                prompt_model_inputs,
                feature_key,
            )
            if rows is not None and rows > 0:
                rows_by_modality[modality] = int(rows)
        return rows_by_modality

    def _media_token_counts_match_feature_rows(
        self,
        prompt_ids: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
    ) -> bool:
        rows_by_modality = self._feature_rows_by_modality(prompt_model_inputs)
        if not rows_by_modality:
            return False
        token_ids = {
            "audio": self._audio_token_id,
            "image": self._image_token_id,
            "video": self._video_token_id,
        }
        for modality, rows in rows_by_modality.items():
            token_id = token_ids.get(modality)
            if token_id is None:
                return False
            present = int((prompt_ids == int(token_id)).sum().item())
            if present != int(rows):
                return False
        return True

    def _original_prompt_ids_if_valid_for_media(
        self,
        prompt_ids: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
    ) -> torch.Tensor | None:
        original_prompt_ids = self._matching_original_prompt_ids(
            prompt_model_inputs,
            prompt_ids,
        )
        if original_prompt_ids is None:
            return None
        if self._invalid_prompt_id_mask(original_prompt_ids).any():
            return None
        if not self._media_token_counts_match_feature_rows(
            original_prompt_ids,
            prompt_model_inputs,
        ):
            return None
        return original_prompt_ids.clone()

    def _vision_spatial_merge_size(self) -> int:
        configs = [
            self._thinker_config,
            getattr(self._thinker_config, "vision_config", None),
            getattr(self._model, "config", None),
            getattr(getattr(self._model, "config", None), "thinker_config", None),
            getattr(getattr(self._model, "config", None), "vision_config", None),
            getattr(
                getattr(getattr(self._model, "config", None), "thinker_config", None),
                "vision_config",
                None,
            ),
        ]
        for config in configs:
            if config is None:
                continue
            value = getattr(config, "spatial_merge_size", None)
            if value is None and isinstance(config, dict):
                value = config.get("spatial_merge_size")
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return 1

    @staticmethod
    def _grid_token_lengths_for_merge(value: Any, merge: int) -> list[int]:
        lengths: list[int] = []
        merge = max(int(merge), 1)
        for row in _as_grid_list(value):
            if len(row) < 3:
                continue
            grid_t, grid_h, grid_w = (int(part) for part in row[:3])
            lengths.append(grid_t * (grid_h // merge) * (grid_w // merge))
        return [length for length in lengths if length > 0]

    def _grid_token_lengths(
        self,
        value: Any,
        *,
        expected_total: int | None = None,
    ) -> list[int]:
        merge = self._vision_spatial_merge_size()
        lengths = self._grid_token_lengths_for_merge(value, merge)
        if (
            expected_total is None
            or expected_total <= 0
            or sum(lengths) == int(expected_total)
        ):
            return lengths

        # Split-model metadata can occasionally omit thinker_config. In that case
        # the default merge=1 overestimates Qwen3.5 visual token spans by 4x. The
        # encoder output row count is authoritative for talker prompt repair, so
        # infer the merge that makes grid spans match the feature rows.
        candidates = [2, 4, 8, 1, 3, 5, 6, 7, 16]
        seen: set[int] = {merge}
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            inferred = self._grid_token_lengths_for_merge(value, candidate)
            if inferred and sum(inferred) == int(expected_total):
                return inferred
        return lengths

    def _media_slot_lengths(
        self,
        prompt_model_inputs: dict[str, Any],
        modality: str,
    ) -> list[int]:
        if modality == "audio":
            lengths = [
                length
                for length in self._int_sequence(
                    prompt_model_inputs.get("audio_feature_lengths")
                )
                if length > 0
            ]
            expected_total = self._feature_rows_for_prompt_model_inputs(
                prompt_model_inputs,
                "audio_embeds",
            )
            if (
                lengths
                and expected_total is not None
                and expected_total > 0
                and sum(lengths) != int(expected_total)
            ):
                # Qwen3.5 runtime metadata may carry pre-projector audio lengths
                # (for example 200 per 2s chunk) while audio_embeds has the
                # downsampled rows consumed by talker (14 per chunk). Preserve
                # per-item boundaries by distributing the authoritative row count.
                if int(expected_total) % len(lengths) == 0:
                    return [int(expected_total) // len(lengths)] * len(lengths)
                scaled: list[int] = []
                cumulative = 0.0
                scale = float(expected_total) / float(sum(lengths))
                emitted = 0
                for idx, length in enumerate(lengths):
                    cumulative += float(length) * scale
                    next_total = (
                        int(expected_total)
                        if idx == len(lengths) - 1
                        else int(round(cumulative))
                    )
                    scaled.append(max(0, next_total - emitted))
                    emitted = next_total
                return [length for length in scaled if length > 0]
            return lengths
        if modality == "image":
            return self._grid_token_lengths(
                prompt_model_inputs.get("image_grid_thw"),
                expected_total=self._feature_rows_for_prompt_model_inputs(
                    prompt_model_inputs,
                    "image_embeds",
                ),
            )
        if modality == "video":
            return self._grid_token_lengths(
                prompt_model_inputs.get("video_grid_thw"),
                expected_total=self._feature_rows_for_prompt_model_inputs(
                    prompt_model_inputs,
                    "video_embeds",
                ),
            )
        return []

    @staticmethod
    def _invalid_prompt_runs(invalid_mask: torch.Tensor) -> list[tuple[int, int, int]]:
        positions = torch.nonzero(invalid_mask, as_tuple=False).reshape(-1).tolist()
        if not positions:
            return []

        runs: list[tuple[int, int, int]] = []
        start = prev = int(positions[0])
        for raw_pos in positions[1:]:
            pos = int(raw_pos)
            if pos == prev + 1:
                prev = pos
                continue
            runs.append((start, prev + 1, prev - start + 1))
            start = prev = pos
        runs.append((start, prev + 1, prev - start + 1))
        return runs

    def _thinker_token_id(self, *names: str) -> int | None:
        aliases = {
            "vision_start_token_id": (
                "vision_start_token_id",
                "vision_bos_token_id",
                "vision_start_token",
                "vision_bos_token",
            ),
            "vision_end_token_id": (
                "vision_end_token_id",
                "vision_eos_token_id",
                "vision_end_token",
                "vision_eos_token",
            ),
            "video_end_token_id": (
                "video_end_token_id",
                "vision_end_token_id",
                "vision_eos_token_id",
                "vision_end_token",
                "vision_eos_token",
            ),
            "audio_start_token_id": (
                "audio_start_token_id",
                "audio_bos_token_id",
                "audio_start_token",
                "audio_bos_token",
            ),
            "audio_end_token_id": (
                "audio_end_token_id",
                "audio_eos_token_id",
                "audio_end_token",
                "audio_eos_token",
            ),
        }
        candidates: list[str] = []
        for name in names:
            candidates.extend(aliases.get(name, (name,)))

        configs = [self._thinker_config, self._tokenizer]
        convert_tokens_to_ids = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        for config in configs:
            if config is None:
                continue
            for name in candidates:
                value = getattr(config, name, None)
                if value is None and isinstance(config, dict):
                    value = config.get(name)
                if value is None:
                    continue
                if isinstance(value, str) and callable(convert_tokens_to_ids):
                    value = convert_tokens_to_ids(value)
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0:
                    return parsed
        return None

    def _context_modality_for_invalid_run(
        self,
        prompt_ids: torch.Tensor,
        *,
        start: int,
        end: int,
        candidate_modalities: set[str],
    ) -> str | None:
        prev_token = int(prompt_ids[start - 1].item()) if start > 0 else None
        next_token = int(prompt_ids[end].item()) if end < prompt_ids.numel() else None

        vision_start = self._thinker_token_id("vision_start_token_id")
        vision_end = self._thinker_token_id("vision_end_token_id", "video_end_token_id")
        if (
            (prev_token is not None and prev_token == vision_start)
            or (next_token is not None and next_token == vision_end)
        ):
            if "video" in candidate_modalities:
                return "video"
            if "image" in candidate_modalities:
                return "image"

        audio_start = self._thinker_token_id("audio_start_token_id")
        audio_end = self._thinker_token_id("audio_end_token_id")
        if (
            (prev_token is not None and prev_token == audio_start)
            or (next_token is not None and next_token == audio_end)
        ) and "audio" in candidate_modalities:
            return "audio"
        return None

    def _canonicalize_fully_compressed_media_prompt(
        self,
        prompt_ids: torch.Tensor,
        invalid_mask: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
        missing_modalities: list[tuple[str, int, int]],
    ) -> torch.Tensor | None:
        if not bool(invalid_mask.all()):
            return None

        token_by_modality = {
            modality: int(token_id) for modality, token_id, _ in missing_modalities
        }
        if not token_by_modality:
            return None

        slot_lengths = {
            modality: self._media_slot_lengths(prompt_model_inputs, modality)
            for modality in token_by_modality
        }
        slot_sums = {modality: sum(lengths) for modality, lengths in slot_lengths.items()}
        missing_by_modality = {
            modality: int(missing) for modality, _, missing in missing_modalities
        }
        prompt_len = int(prompt_ids.numel())
        if any(
            slot_sums.get(modality, 0) != missing
            for modality, missing in missing_by_modality.items()
        ):
            inferred_slot_lengths = self._infer_rtc_media_slot_lengths(
                prompt_len=prompt_len,
                missing_by_modality=missing_by_modality,
            )
            if inferred_slot_lengths is not None:
                slot_lengths = inferred_slot_lengths
                slot_sums = {
                    modality: sum(lengths)
                    for modality, lengths in slot_lengths.items()
                }
        total_media = sum(slot_sums.values())
        if any(
            slot_sums.get(modality, 0) != missing
            for modality, missing in missing_by_modality.items()
        ):
            return None
        if total_media > prompt_len:
            return None

        sequence: list[tuple[str, int]] = []
        fallback_modality = "__text_fallback__"
        if (
            "video" in slot_lengths
            and "audio" in slot_lengths
            and len(slot_lengths["video"]) == len(slot_lengths["audio"])
            and not slot_lengths.get("image")
        ):
            extras = self._distribute_prompt_extra(
                prompt_len - total_media,
                len(slot_lengths["video"]),
            )
            for index, (video_len, audio_len) in enumerate(
                zip(
                    slot_lengths["video"],
                    slot_lengths["audio"],
                )
            ):
                extra_len = extras[index] if index < len(extras) else 0
                if extra_len > 1:
                    sequence.append((fallback_modality, extra_len - 1))
                sequence.append(("video", video_len))
                if extra_len > 0:
                    sequence.append((fallback_modality, 1))
                sequence.append(("audio", audio_len))
        else:
            extra_len = prompt_len - total_media
            if extra_len > 0:
                sequence.append((fallback_modality, extra_len))
            for modality in ("image", "video", "audio"):
                for length in slot_lengths.get(modality, []):
                    sequence.append((modality, length))

        if sum(length for _, length in sequence) != prompt_len:
            return None

        original_prompt_ids = self._matching_original_prompt_ids(
            prompt_model_inputs,
            prompt_ids,
        )
        canonical = (
            original_prompt_ids.clone()
            if original_prompt_ids is not None
            else prompt_ids.clone()
        )
        cursor = 0
        fallback_token_id = self._fallback_prompt_text_token_id()
        for modality, length in sequence:
            if length <= 0:
                continue
            end = cursor + int(length)
            if modality == fallback_modality:
                if original_prompt_ids is None:
                    canonical[cursor:end] = fallback_token_id
                else:
                    segment = canonical[cursor:end]
                    invalid_segment = self._invalid_prompt_id_mask(segment)
                    if invalid_segment.any():
                        segment[invalid_segment] = fallback_token_id
            else:
                canonical[cursor:end] = token_by_modality[modality]
            cursor = end
        return canonical

    def _rtc_media_slot_sequence(
        self,
        *,
        prompt_len: int,
        prompt_model_inputs: dict[str, Any],
        token_by_modality: dict[str, int],
    ) -> list[tuple[str, int]] | None:
        if set(token_by_modality) != {"audio", "video"}:
            return None

        slot_lengths = {
            modality: self._media_slot_lengths(prompt_model_inputs, modality)
            for modality in token_by_modality
        }
        if (
            not slot_lengths.get("video")
            or not slot_lengths.get("audio")
            or len(slot_lengths["video"]) != len(slot_lengths["audio"])
        ):
            rows_by_modality = self._feature_rows_by_modality(prompt_model_inputs)
            inferred_slot_lengths = self._infer_rtc_media_slot_lengths(
                prompt_len=prompt_len,
                missing_by_modality=rows_by_modality,
            )
            if inferred_slot_lengths is None:
                return None
            slot_lengths = inferred_slot_lengths

        slot_sums = {modality: sum(lengths) for modality, lengths in slot_lengths.items()}
        rows_by_modality = self._feature_rows_by_modality(prompt_model_inputs)
        if any(
            slot_sums.get(modality, 0) != rows_by_modality.get(modality, 0)
            for modality in token_by_modality
        ):
            return None

        total_media = sum(slot_sums.values())
        if total_media > prompt_len:
            return None

        sequence: list[tuple[str, int]] = []
        fallback_modality = "__text_fallback__"
        extras = self._distribute_prompt_extra(
            prompt_len - total_media,
            len(slot_lengths["video"]),
        )
        for index, (video_len, audio_len) in enumerate(
            zip(slot_lengths["video"], slot_lengths["audio"])
        ):
            extra_len = extras[index] if index < len(extras) else 0
            if extra_len > 1:
                sequence.append((fallback_modality, extra_len - 1))
            sequence.append(("video", video_len))
            if extra_len > 0:
                sequence.append((fallback_modality, 1))
            sequence.append(("audio", audio_len))

        if sum(length for _, length in sequence) != prompt_len:
            return None
        return sequence

    def _canonicalize_partial_rtc_media_prompt(
        self,
        prompt_ids: torch.Tensor,
        invalid_mask: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
    ) -> torch.Tensor | None:
        rows_by_modality = self._feature_rows_by_modality(prompt_model_inputs)
        token_ids = {
            "audio": self._audio_token_id,
            "video": self._video_token_id,
        }
        token_by_modality = {
            modality: int(token_ids[modality])
            for modality in ("audio", "video")
            if rows_by_modality.get(modality, 0) > 0
            and token_ids.get(modality) is not None
        }
        if set(token_by_modality) != {"audio", "video"}:
            return None

        sequence = self._rtc_media_slot_sequence(
            prompt_len=int(prompt_ids.numel()),
            prompt_model_inputs=prompt_model_inputs,
            token_by_modality=token_by_modality,
        )
        if sequence is None:
            return None

        original_prompt_ids = self._matching_original_prompt_ids(
            prompt_model_inputs,
            prompt_ids,
        )
        canonical = (
            original_prompt_ids.clone()
            if original_prompt_ids is not None
            else prompt_ids.clone()
        )
        fallback_token_id = self._fallback_prompt_text_token_id()
        fallback_modality = "__text_fallback__"
        cursor = 0
        for modality, length in sequence:
            if length <= 0:
                continue
            end = cursor + int(length)
            invalid_segment = invalid_mask[cursor:end]
            if invalid_segment.any():
                segment = canonical[cursor:end]
                if modality == fallback_modality:
                    if original_prompt_ids is None:
                        segment[invalid_segment] = fallback_token_id
                    else:
                        still_invalid = self._invalid_prompt_id_mask(segment)
                        if still_invalid.any():
                            segment[still_invalid] = fallback_token_id
                else:
                    segment[invalid_segment] = token_by_modality[modality]
            cursor = end

        if self._invalid_prompt_id_mask(canonical).any():
            return None
        if not self._media_token_counts_match_feature_rows(
            canonical,
            prompt_model_inputs,
        ):
            return None
        return canonical

    def _unmapped_media_debug_summary(
        self,
        prompt_ids: torch.Tensor,
        invalid_mask: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
        missing_modalities: list[tuple[str, int, int]],
    ) -> str:
        runs = self._invalid_prompt_runs(invalid_mask)
        run_parts: list[str] = []
        for start, end, length in runs[:10]:
            prev_token = int(prompt_ids[start - 1].item()) if start > 0 else None
            next_token = int(prompt_ids[end].item()) if end < prompt_ids.numel() else None
            unique_count = int(torch.unique(prompt_ids[start:end]).numel())
            run_parts.append(
                f"{start}:{end}:len={length}:prev={prev_token}:next={next_token}:uniq={unique_count}"
            )

        slot_parts: list[str] = []
        for modality, _, _ in missing_modalities:
            lengths = self._media_slot_lengths(prompt_model_inputs, modality)
            unique_lengths = sorted(set(lengths))
            slot_parts.append(
                f"{modality}:n={len(lengths)}:sum={sum(lengths)}:uniq={unique_lengths[:8]}"
            )

        special_ids = {
            "vision_start": self._thinker_token_id("vision_start_token_id"),
            "vision_end": self._thinker_token_id(
                "vision_end_token_id", "video_end_token_id"
            ),
            "audio_start": self._thinker_token_id("audio_start_token_id"),
            "audio_end": self._thinker_token_id("audio_end_token_id"),
        }
        return (
            f"invalid_total={int(invalid_mask.sum().item())}; "
            f"run_count={len(runs)}; first_runs={run_parts}; "
            f"slots={slot_parts}; special_ids={special_ids}; "
            f"prompt_len={prompt_ids.numel()}"
        )

    def _canonicalize_unmapped_media_pad_ids(
        self,
        prompt_ids: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
    ) -> torch.Tensor:
        invalid_mask = self._invalid_prompt_id_mask(prompt_ids)
        if not invalid_mask.any():
            return prompt_ids

        original_prompt_ids = self._original_prompt_ids_if_valid_for_media(
            prompt_ids,
            prompt_model_inputs,
        )
        if original_prompt_ids is not None:
            return original_prompt_ids

        partial_rtc = self._canonicalize_partial_rtc_media_prompt(
            prompt_ids,
            invalid_mask,
            prompt_model_inputs,
        )
        if partial_rtc is not None:
            return partial_rtc

        missing_modalities: list[tuple[str, int, int]] = []
        for modality, feature_key, token_id in (
            ("audio", "audio_embeds", self._audio_token_id),
            ("image", "image_embeds", self._image_token_id),
            ("video", "video_embeds", self._video_token_id),
        ):
            if token_id is None:
                continue
            rows = self._feature_rows_for_prompt_model_inputs(
                prompt_model_inputs,
                feature_key,
            )
            if rows is None or rows <= 0:
                continue
            present = int((prompt_ids == int(token_id)).sum().item())
            missing = rows - present
            if missing > 0:
                missing_modalities.append((modality, int(token_id), missing))

        canonical = prompt_ids
        if len(missing_modalities) > 1:
            compressed = self._canonicalize_fully_compressed_media_prompt(
                prompt_ids,
                invalid_mask,
                prompt_model_inputs,
                missing_modalities,
            )
            if compressed is not None:
                return compressed

            canonical = prompt_ids.clone()
            invalid_values, invalid_counts = torch.unique(
                prompt_ids[invalid_mask],
                return_counts=True,
                sorted=False,
            )
            slot_lengths = {
                modality: self._media_slot_lengths(prompt_model_inputs, modality)
                for modality, _, _ in missing_modalities
            }
            remaining = {
                modality: int(missing) for modality, _, missing in missing_modalities
            }
            token_by_modality = {
                modality: int(token_id)
                for modality, token_id, _ in missing_modalities
            }

            run_mapped = False
            for start, end, length in self._invalid_prompt_runs(invalid_mask):
                context_modality = self._context_modality_for_invalid_run(
                    prompt_ids,
                    start=start,
                    end=end,
                    candidate_modalities=set(token_by_modality),
                )
                if (
                    context_modality is not None
                    and remaining.get(context_modality, 0) >= length
                ):
                    remaining[context_modality] -= length
                    canonical[start:end] = token_by_modality[context_modality]
                    run_mapped = True
            if run_mapped:
                invalid_mask = self._invalid_prompt_id_mask(canonical)
                if not invalid_mask.any():
                    return canonical
                prompt_ids = canonical

            mapped = False
            for value, count_tensor in zip(
                invalid_values.tolist(),
                invalid_counts.tolist(),
            ):
                count = int(count_tensor)
                matches = [
                    modality
                    for modality, lengths in slot_lengths.items()
                    if count in lengths and remaining.get(modality, 0) >= count
                ]
                if len(matches) != 1:
                    continue
                modality = matches[0]
                slot_lengths[modality].remove(count)
                remaining[modality] -= count
                canonical[canonical == int(value)] = token_by_modality[modality]
                mapped = True
            if mapped:
                invalid_mask = self._invalid_prompt_id_mask(canonical)
                if not invalid_mask.any():
                    return canonical
                prompt_ids = canonical

            run_mapped = False
            for start, end, length in self._invalid_prompt_runs(invalid_mask):
                context_modality = self._context_modality_for_invalid_run(
                    prompt_ids,
                    start=start,
                    end=end,
                    candidate_modalities=set(token_by_modality),
                )
                if (
                    context_modality is not None
                    and remaining.get(context_modality, 0) >= length
                ):
                    remaining[context_modality] -= length
                    canonical[start:end] = token_by_modality[context_modality]
                    run_mapped = True
                    continue
                matches = [
                    modality
                    for modality, lengths in slot_lengths.items()
                    if length in lengths and remaining.get(modality, 0) >= length
                ]
                if len(matches) != 1:
                    continue
                modality = matches[0]
                slot_lengths[modality].remove(length)
                remaining[modality] -= length
                canonical[start:end] = token_by_modality[modality]
                run_mapped = True
            if run_mapped:
                invalid_mask = self._invalid_prompt_id_mask(canonical)
                if not invalid_mask.any():
                    return canonical
                prompt_ids = canonical

        if len(missing_modalities) == 1:
            _, token_id, missing = missing_modalities[0]
            if int(invalid_mask.sum().item()) == missing:
                canonical = prompt_ids.clone()
                canonical[invalid_mask] = int(token_id)
                return canonical

        invalid_values = torch.unique(prompt_ids[invalid_mask]).tolist()
        formatted = ", ".join(str(int(value)) for value in invalid_values[:8])
        if len(invalid_values) > 8:
            formatted += ", ..."
        modality_state = ", ".join(
            f"{modality}:missing={missing}"
            for modality, _, missing in missing_modalities
        )
        debug_summary = self._unmapped_media_debug_summary(
            prompt_ids,
            invalid_mask,
            prompt_model_inputs,
            missing_modalities,
        )
        raise ValueError(
            "Qwen3.5 talker prompt contains unmapped media cache pad ids: "
            f"{formatted}; modality feature/token gaps: {modality_state or 'none'}; "
            f"{debug_summary}"
        )

    def _canonicalize_video_frame_placeholders(
        self,
        prompt_ids: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
    ) -> torch.Tensor:
        if self._video_token_id is None or self._image_token_id is None:
            return prompt_ids

        video_rows = self._feature_rows_for_prompt_model_inputs(
            prompt_model_inputs,
            "video_embeds",
        )
        if video_rows is None or video_rows <= 0:
            return prompt_ids
        image_rows = self._feature_rows_for_prompt_model_inputs(
            prompt_model_inputs,
            "image_embeds",
        )
        if image_rows not in (None, 0):
            return prompt_ids

        video_token_id = int(self._video_token_id)
        image_token_id = int(self._image_token_id)
        video_count = int((prompt_ids == video_token_id).sum().item())
        image_count = int((prompt_ids == image_token_id).sum().item())
        missing = int(video_rows) - video_count
        if missing <= 0 or image_count != missing:
            return prompt_ids

        canonical = prompt_ids.clone()
        canonical[canonical == image_token_id] = video_token_id
        return canonical

    def _canonicalize_prompt_ids_for_talker(
        self,
        prompt_ids: torch.Tensor,
        prompt_model_inputs: dict[str, Any],
    ) -> torch.Tensor:
        canonical = prompt_ids.clone()
        pad_values = prompt_model_inputs.get("pad_values")
        if isinstance(pad_values, dict):
            for modality, token_id in (
                ("audio", self._audio_token_id),
                ("image", self._image_token_id),
                ("video", self._video_token_id),
            ):
                if token_id is None or modality not in pad_values:
                    continue
                for pad_id in self._signed_int64_aliases(pad_values.get(modality)):
                    canonical[canonical == int(pad_id)] = int(token_id)
        canonical = self._canonicalize_unmapped_media_pad_ids(
            canonical,
            prompt_model_inputs,
        )
        return self._canonicalize_video_frame_placeholders(
            canonical,
            prompt_model_inputs,
        )

    def _reconstruct_prompt_states(
        self, state: Qwen3OmniPipelineState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        prompt = state.prompt or {}
        prompt_input_ids = prompt["input_ids"]
        if prompt_input_ids.dim() == 2:
            prompt_input_ids = prompt_input_ids[0]
        raw_prompt_ids = prompt_input_ids.to(dtype=torch.long).cpu()
        prompt_model_inputs = self._prompt_model_inputs(state)
        prompt_original_ids = prompt.get("original_input_ids")
        if (
            prompt_original_ids is not None
            and "original_input_ids" not in prompt_model_inputs
        ):
            prompt_model_inputs = dict(prompt_model_inputs)
            prompt_model_inputs["original_input_ids"] = prompt_original_ids
        prompt_ids = self._canonicalize_prompt_ids_for_talker(
            raw_prompt_ids,
            prompt_model_inputs,
        )

        prompt_embed = self._load_prompt_token_embeddings(prompt_ids)
        prompt_hidden = self._load_thinker_hidden_embeddings(prompt_ids)

        for modality_key, token_id in (
            ("audio_embeds", self._audio_token_id),
            ("image_embeds", self._image_token_id),
            ("video_embeds", self._video_token_id),
        ):
            if token_id is None:
                continue
            feature_tensor = coerce_feature_tensor(
                prompt_model_inputs.get(modality_key)
            )
            if feature_tensor is None:
                continue
            mask = prompt_ids == int(token_id)
            if not mask.any():
                continue
            prompt_hidden[mask] = feature_tensor.to(
                device=prompt_hidden.device,
                dtype=prompt_hidden.dtype,
            )

        return prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs

    def _filter_voice_style_prefix(
        self,
        thinker_chunks: list[Any],
        assistant_token_ids: torch.Tensor,
    ) -> tuple[list[Any], torch.Tensor, str]:
        raw_ids = [
            int(token_id)
            for token_id in assistant_token_ids.reshape(-1).detach().cpu().tolist()
        ]
        if not raw_ids or self._tokenizer is None:
            return thinker_chunks, assistant_token_ids, ""

        decoded = _decode_token_ids(self._tokenizer, raw_ids)
        if decoded is None:
            return thinker_chunks, assistant_token_ids, ""
        content, instruction = _parse_voice_style_prefix(decoded)
        if not instruction:
            return thinker_chunks, assistant_token_ids, ""

        drop_count = _voice_style_drop_count(
            self._tokenizer,
            raw_ids,
            content=content,
        )
        if drop_count is None or drop_count > len(thinker_chunks):
            return thinker_chunks, assistant_token_ids, ""

        # Qwen3.5 consumes the leading <voice_style> tag in the thinker output
        # and filters those tag tokens from talker text. Do the non-streaming
        # equivalent during full prefill so tokens, embeds, and hidden states
        # remain strictly chunk-aligned.
        return (
            thinker_chunks[drop_count:],
            assistant_token_ids[drop_count:],
            instruction,
        )

    def get_tts_special_embeds(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._tts_special_cache is None:
            token_ids = torch.tensor(
                [
                    self._tts_bos_token_id,
                    self._tts_eos_token_id,
                    self._tts_pad_token_id,
                ],
                dtype=torch.long,
                device=self._device,
            )
            projected = self._model.embed_text_ids(token_ids).to(
                device=self._device,
                dtype=self._dtype,
            )
            self._tts_special_cache = projected.chunk(3, dim=0)
        return self._tts_special_cache

    def build_prompt_prefill(
        self,
        payload: Any,
        thinker_chunks: list[Any],
        *,
        thinker_done: bool,
    ) -> dict[str, Any]:
        if not thinker_chunks:
            raise ValueError("prompt prefill requires thinker chunks")

        state = Qwen3OmniPipelineState.from_dict(payload.data)
        prompt_ids, prompt_embed, prompt_hidden, prompt_model_inputs = (
            self._reconstruct_prompt_states(state)
        )

        assistant_token_ids = self.extract_chunk_token_ids(thinker_chunks)
        thinker_chunks, assistant_token_ids, voice_style_instruction = (
            self._filter_voice_style_prefix(thinker_chunks, assistant_token_ids)
        )
        if assistant_token_ids.numel() > 0:
            assistant_embed = self._load_prompt_token_embeddings(assistant_token_ids)
        else:
            assistant_embed = prompt_embed.new_empty((0, prompt_embed.shape[-1]))
        assistant_hidden = prompt_hidden.new_empty(
            (assistant_token_ids.shape[0], prompt_hidden.shape[-1])
        )

        thinker_input_ids = torch.cat([prompt_ids, assistant_token_ids], dim=0)
        thinker_embed = torch.cat([prompt_embed, assistant_embed], dim=0)
        thinker_hidden = torch.cat([prompt_hidden, assistant_hidden], dim=0)
        multimodal_mask = self.build_multimodal_mask(thinker_input_ids)

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.get_tts_special_embeds()
        params = _apply_tts_generate_mode(
            _params_with_openai_audio_config(payload.request)
        )
        params = _params_with_request_language_fallback(payload.request, params)
        if (
            voice_style_instruction
            and not _has_any_value(params, _STYLE_PARAM_KEYS)
            and not _has_any_value(params, _INSTRUCTION_PARAM_KEYS)
            and not _has_any_value(params, _VOICE_CLONE_PARAM_KEYS)
        ):
            params["voice_style"] = voice_style_instruction
        params = self._normalize_voice_clone_params(params)
        if _should_ignore_voice(params):
            speaker_prefix = None
            speaker_id = None
        else:
            _, speaker_id = self._resolve_speaker_name_and_id(params)
            speaker_prefix = self._build_speaker_prefix(params)
        assistant_text = _decode_token_ids(
            self._tokenizer,
            assistant_token_ids.detach().cpu().tolist(),
        )
        language_id = self._resolve_language_id(params)
        if language_id is None and _looks_like_chinese_text(assistant_text):
            language_id = self._talker_language_id.get("chinese")
        assistant_instruct_ids = self._resolve_assistant_instruct_ids(params)
        assistant_instruct_embed = (
            self._text_embeds(assistant_instruct_ids)
            if assistant_instruct_ids
            else None
        )

        prefill = self._build_prefill_input(
            thinker_embed=thinker_embed,
            thinker_hidden=thinker_hidden,
            thinker_input_ids=thinker_input_ids,
            multimodal_mask=multimodal_mask,
            assistant_token_count=int(assistant_token_ids.numel()),
            text_projection=self._model.text_projection,
            hidden_projection=self._model.hidden_projection,
            codec_embed_fn=self._model.get_input_embeddings(),
            tts_bos_embed=tts_bos_embed,
            tts_eos_embed=tts_eos_embed,
            tts_pad_embed=tts_pad_embed,
            assistant_instruct_embed=assistant_instruct_embed,
            im_start_token_id=self._im_start_token_id,
            system_token_id=self._system_token_id,
            user_token_id=self._user_token_id,
            assistant_token_id=self._assistant_token_id,
            speaker_id=speaker_id,
            codec_nothink_id=self._codec_nothink_id,
            codec_think_id=self._codec_think_id,
            codec_think_bos_id=self._codec_think_bos_id,
            codec_think_eos_id=self._codec_think_eos_id,
            codec_pad_id=self._codec_pad_id,
            codec_bos_id=self._codec_bos_id,
            tts_pad_token_id=self._tts_pad_token_id,
            language_id=language_id,
            include_assistant_eos=thinker_done,
            im_end_token_id=self._im_end_token_id,
        )

        input_embeds = prefill["input_embeds"]
        input_ids = prefill["input_ids"]
        if speaker_prefix is not None:
            prefix_ids = torch.full(
                (speaker_prefix.shape[0],),
                self._tts_pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            input_embeds = torch.cat([speaker_prefix, input_embeds], dim=0)
            input_ids = torch.cat([prefix_ids, input_ids], dim=0)

        future_text_rows = prefill["future_text_rows"]
        return {
            "input_embeds": input_embeds,
            "input_ids": input_ids,
            "pending_text_queue": self.tensor_rows_to_queue(
                future_text_rows
            ),
            "tts_pad_embed": tts_pad_embed[0].detach(),
            "tts_eos_embed": tts_eos_embed[0].detach(),
            # Qwen3.5 talker uses prompt_embeds. Do not pass the thinker's
            # original grid to the talker for M-RoPE; after MM token downsampling
            # it would cause length mismatches.
            "prompt_model_inputs": {},
        }

    def _build_prefill_input(
        self,
        *,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        thinker_input_ids: torch.Tensor,
        multimodal_mask: torch.Tensor,
        assistant_token_count: int | None,
        text_projection,
        hidden_projection,
        codec_embed_fn,
        tts_bos_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        assistant_instruct_embed: torch.Tensor | None,
        im_start_token_id: int,
        system_token_id: int,
        user_token_id: int,
        assistant_token_id: int,
        speaker_id: int | None,
        codec_nothink_id: int,
        codec_think_id: int | None,
        codec_think_bos_id: int,
        codec_think_eos_id: int,
        codec_pad_id: int,
        codec_bos_id: int,
        tts_pad_token_id: int,
        language_id: int | None = None,
        include_assistant_eos: bool = True,
        im_end_token_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        segments = segment_chat_template(
            thinker_input_ids,
            im_start_token_id=im_start_token_id,
            system_token_id=system_token_id,
            user_token_id=user_token_id,
            assistant_token_id=assistant_token_id,
        )

        all_embeds = []
        all_ids = []
        future_text_rows = None
        assistant_indices = [
            idx for idx, seg in enumerate(segments) if seg["role"] == "assistant"
        ]
        last_assistant_idx = assistant_indices[-1] if assistant_indices else None

        for seg_idx, seg in enumerate(segments):
            if seg["role"] == "system":
                continue

            start, end = seg["start"], seg["end"]
            seg_embed = thinker_embed[start:end]
            seg_hidden = thinker_hidden[start:end]
            seg_mm_mask = multimodal_mask[start:end]
            seg_ids = thinker_input_ids[start:end]

            if seg["role"] == "user":
                continue
            elif seg["role"] == "assistant":
                if last_assistant_idx is not None and seg_idx != last_assistant_idx:
                    continue
                generated_count = (
                    max(0, int(assistant_token_count))
                    if assistant_token_count is not None
                    else None
                )
                if (
                    im_end_token_id is not None
                    and seg_embed.shape[0] > 0
                    and int(thinker_input_ids[end - 1].item()) == im_end_token_id
                ):
                    seg_embed = seg_embed[:-1]
                    if generated_count is not None:
                        generated_count = max(0, generated_count - 1)
                if generated_count is not None:
                    # Qwen3.5 chat templates may append static assistant prompt
                    # suffixes such as "<think>\n\n</think>\n\n" after the
                    # role prefix. Those prompt-only suffix tokens are not
                    # spoken text; keeping them makes code2wav speak a
                    # hallucinated word before the actual response.
                    role_rows = min(3, int(seg_embed.shape[0]))
                    available_generated = max(0, int(seg_embed.shape[0]) - role_rows)
                    generated_count = min(generated_count, available_generated)
                    generated_embed = (
                        seg_embed[-generated_count:]
                        if generated_count > 0
                        else seg_embed.new_empty((0, seg_embed.shape[-1]))
                    )
                    seg_embed = torch.cat(
                        [seg_embed[:role_rows], generated_embed],
                        dim=0,
                    )
                assistant_result = _build_qwen35_assistant_part(
                    assistant_embed=seg_embed,
                    assistant_instruct_embed=assistant_instruct_embed,
                    text_projection=text_projection,
                    codec_embed_fn=codec_embed_fn,
                    tts_bos_embed=tts_bos_embed,
                    tts_eos_embed=tts_eos_embed,
                    tts_pad_embed=tts_pad_embed,
                    speaker_id=speaker_id,
                    codec_nothink_id=codec_nothink_id,
                    codec_think_id=codec_think_id,
                    codec_think_bos_id=codec_think_bos_id,
                    codec_think_eos_id=codec_think_eos_id,
                    codec_pad_id=codec_pad_id,
                    codec_bos_id=codec_bos_id,
                    tts_pad_token_id=tts_pad_token_id,
                    language_id=language_id,
                )
                all_embeds.append(assistant_result["input_embeds"])
                all_ids.append(
                    assistant_result["input_ids"].to(
                        device=thinker_input_ids.device,
                        dtype=torch.long,
                    )
                )
                future_text_rows = assistant_result["future_text_rows"]
                if (
                    not include_assistant_eos
                    and future_text_rows is not None
                    and future_text_rows.shape[0] > 0
                ):
                    future_text_rows = future_text_rows[:-1]

        if not all_embeds:
            assistant_result = _build_qwen35_assistant_part(
                assistant_embed=thinker_embed.new_empty((0, thinker_embed.shape[-1])),
                assistant_instruct_embed=assistant_instruct_embed,
                text_projection=text_projection,
                codec_embed_fn=codec_embed_fn,
                tts_bos_embed=tts_bos_embed,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                speaker_id=speaker_id,
                codec_nothink_id=codec_nothink_id,
                codec_think_id=codec_think_id,
                codec_think_bos_id=codec_think_bos_id,
                codec_think_eos_id=codec_think_eos_id,
                codec_pad_id=codec_pad_id,
                codec_bos_id=codec_bos_id,
                tts_pad_token_id=tts_pad_token_id,
                language_id=language_id,
            )
            all_embeds.append(assistant_result["input_embeds"])
            all_ids.append(
                assistant_result["input_ids"].to(
                    device=thinker_input_ids.device,
                    dtype=torch.long,
                )
            )
            future_text_rows = assistant_result["future_text_rows"]
            if (
                not include_assistant_eos
                and future_text_rows is not None
                and future_text_rows.shape[0] > 0
            ):
                future_text_rows = future_text_rows[:-1]

        return {
            "input_embeds": torch.cat(all_embeds, dim=0),
            "input_ids": torch.cat(all_ids, dim=0),
            "future_text_rows": future_text_rows,
        }

    def _build_user_part(
        self,
        *,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        thinker_input_ids: torch.Tensor,
        multimodal_mask: torch.Tensor,
        text_projection,
        hidden_projection,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mm_pos = multimodal_mask.nonzero(as_tuple=True)[0]
        text_pos = (~multimodal_mask).nonzero(as_tuple=True)[0]
        limit = self._max_thinker_to_talker_mm_tokens
        if limit is not None and mm_pos.numel() > limit:
            if limit <= 0:
                mm_pos = mm_pos[:0]
            else:
                keep = torch.linspace(
                    0,
                    mm_pos.numel() - 1,
                    limit,
                    device=mm_pos.device,
                ).long()
                mm_pos = mm_pos.index_select(0, keep)

        selected_pos = torch.cat([mm_pos, text_pos], dim=0)
        if selected_pos.numel() == 0:
            out_size = text_projection(thinker_embed[:1]).shape[-1]
            return thinker_embed.new_empty((0, out_size)), thinker_input_ids[:0]

        selected_pos = selected_pos.index_select(0, selected_pos.argsort())
        selected_embed = thinker_embed.index_select(0, selected_pos)
        selected_hidden = thinker_hidden.index_select(0, selected_pos)
        selected_mask = multimodal_mask.index_select(0, selected_pos)
        selected_ids = thinker_input_ids.index_select(0, selected_pos)

        out_size = text_projection(selected_embed[:1]).shape[-1]
        output = torch.empty(
            (selected_embed.shape[0], out_size),
            device=selected_embed.device,
            dtype=selected_embed.dtype,
        )
        if selected_mask.any():
            # Qwen3.5 evenly samples overly long MM hidden states so long videos
            # do not blow up talker prefill. Text and role tokens keep their
            # original order and are not trimmed.
            output[selected_mask] = hidden_projection(selected_hidden[selected_mask])
        text_mask = ~selected_mask
        if text_mask.any():
            output[text_mask] = text_projection(selected_embed[text_mask])
        return output, selected_ids

    def _resolve_speaker_name_and_id(
        self,
        params: dict[str, Any],
    ) -> tuple[str | None, int]:
        speaker_name = _raw_voice_name(params)
        speaker_name = self._resolve_voice_code(speaker_name)

        if speaker_name is not None and speaker_name in self._speaker_map:
            return speaker_name, int(self._speaker_map[speaker_name])
        if speaker_name is not None and self._speaker_map:
            if "speaker_id" in params and params["speaker_id"] is not None:
                return speaker_name, int(params["speaker_id"])
            if self._resolve_prompt_speaker_codes(params) is not None:
                return speaker_name, int(params.get("speaker_id", 0))
            raise ValueError(
                f"Qwen3.5-Omni voice_type {speaker_name!r} is not in "
                f"available speakers: {sorted(self._speaker_map)}"
            )
        return speaker_name, int(params.get("speaker_id", 0))

    def _resolve_language_id(self, params: dict[str, Any]) -> int | None:
        value = params.get("language_id")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

        for key in (
            "language",
            "lang",
            "language_type",
            "target_language",
            "target_lang",
        ):
            value = params.get(key)
            if value is None:
                continue
            language = str(value).lower().strip()
            if not language or language == "auto":
                return None
            if language.lstrip("-").isdigit():
                return int(language)
            for candidate in _language_key_candidates(language):
                mapped = self._talker_language_id.get(candidate)
                if mapped is not None:
                    # Real requests often use zh-CN/en_US while configs usually
                    # provide zh/en. Try an exact match first, then fall back to
                    # the primary language to avoid requiring manual language_id.
                    return int(mapped)
        target_language = _translation_target_value(params)
        if target_language is not None:
            language = str(target_language).lower().strip()
            if not language or language == "auto":
                return None
            if language.lstrip("-").isdigit():
                return int(language)
            for candidate in _language_key_candidates(language):
                mapped = self._talker_language_id.get(candidate)
                if mapped is not None:
                    # Map translation_options.target_lang to the talker language
                    # token so translation requests produce target-language
                    # speech control instead of text only.
                    return int(mapped)
        return None

    def _resolve_assistant_instruct_ids(self, params: dict[str, Any]) -> list[int]:
        for key in (
            "assistant_instruct_ids",
            "talker_assistant_prompt_ids",
            "voice_style_ids",
            "instruct_ids",
        ):
            if key in params and params[key] is not None:
                return _coerce_int_list(params[key])

        for key in ("voice_style", "instruction", "talker_instruction", "style"):
            value = params.get(key)
            if not value:
                continue
            instruction = str(value).lower().strip()
            mapped = self._talker_assistant_prompt_id_mapping.get(instruction)
            if mapped is not None:
                # Mapping comes from Qwen3.5 config
                # talker_assistant_prompt_id_mapping and aligns with voice_style.
                return list(mapped)
        return []

    def _resolve_system_instruct_ids(self, params: dict[str, Any]) -> list[int]:
        for key in (
            "system_instruct_ids",
            "talker_system_instruct_ids",
            "speaker_system_instruct_ids",
        ):
            if key in params and params[key] is not None:
                return _coerce_int_list(params[key])
        for key in (
            "talker_system_instruct",
            "system_instruct",
            "voice_clone_system_instruct",
        ):
            value = params.get(key)
            if not value:
                continue
            if isinstance(value, (list, tuple, torch.Tensor)):
                return _coerce_int_list(value)
            if self._tokenizer is None or not hasattr(self._tokenizer, "encode"):
                raise ValueError(
                    "Qwen3.5 voice clone system instruct text requires a tokenizer"
                )
            return [int(token_id) for token_id in self._tokenizer.encode(str(value))]
        return []

    def _resolve_prompt_speaker_codes(self, params: dict[str, Any]) -> torch.Tensor | None:
        for key in (
            "prompt_speaker_codes",
            "speaker_codec_codes",
            "voice_clone_codes",
        ):
            value = params.get(key)
            codes = _coerce_long_tensor(value)
            if codes is not None and codes.numel() > 0:
                return codes
        return None

    def _resolve_voice_code(self, voice_name: str | None) -> str | None:
        if not voice_name:
            return self._default_voice_code()

        voice_name = _strip_voice_control_suffix(voice_name)
        if not voice_name:
            # Pure prefix_caching skips code2wav, but the talker still uses the
            # default speaker to build a reusable prefix.
            return self._default_voice_code()
        if voice_name == "default":
            return self._default_voice_code()
        mapped = self._voice_map.get(voice_name)
        if mapped is not None:
            return mapped
        if voice_name in self._voice_map.values():
            return voice_name
        if voice_name in _VOICE_TO_SPK_MAPPING:
            return _VOICE_TO_SPK_MAPPING[voice_name]
        if voice_name in _VOICE_TO_SPK_MAPPING.values():
            return voice_name
        return voice_name

    def _default_voice_code(self) -> str | None:
        for speaker_key in self._voice_map.values():
            if not self._speaker_map or speaker_key in self._speaker_map:
                return speaker_key
        # Qwen3.5 default voice selection: multilingual models prefer Tina,
        # dialect models commonly use Sunny, and otherwise fall back to the first
        # valid speaker_map entry.
        for voice_name in ("tina", "sunny"):
            speaker_key = _VOICE_TO_SPK_MAPPING[voice_name]
            if speaker_key in self._speaker_map:
                return speaker_key
        if self._speaker_map:
            return next(iter(self._speaker_map))
        return None

    def _text_embeds(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
        return self._model.embed_text_ids(ids).to(
            device=self._device,
            dtype=self._dtype,
        )

    def _codec_embeds(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
        return self._model.get_input_embeddings()(ids).to(
            device=self._device,
            dtype=self._dtype,
        )

    def _speaker_code_embeds(
        self,
        *,
        params: dict[str, Any],
        speaker_id: int,
    ) -> torch.Tensor | None:
        prompt_codes = self._resolve_prompt_speaker_codes(params)
        if prompt_codes is not None:
            codec_code_embeddings = getattr(self._model, "codec_code_embeddings", None)
            if codec_code_embeddings is None:
                return None
            return codec_code_embeddings(prompt_codes).to(
                device=self._device,
                dtype=self._dtype,
            )

        speaker_embed_fn = getattr(
            self._model,
            "speaker_codec_input_embeddings",
            None,
        )
        if speaker_embed_fn is None:
            return None
        return speaker_embed_fn(speaker_id)

    def _build_speaker_prefix(self, params: dict[str, Any]) -> torch.Tensor | None:
        if _should_ignore_voice(params):
            # When voice is None/null, do not inject a speaker codec prefix; the
            # absence of voice must not be interpreted as speaker_id=0.
            return None
        speaker_name, speaker_id = self._resolve_speaker_name_and_id(params)
        prompt_speaker_codes = self._resolve_prompt_speaker_codes(params)
        system_instruct_ids = self._resolve_system_instruct_ids(params)
        speaker_embeds = self._speaker_code_embeds(
            params=params,
            speaker_id=speaker_id,
        )
        if not isinstance(speaker_embeds, torch.Tensor) or speaker_embeds.numel() == 0:
            return None

        parts = [
            self._text_embeds(
                [self._im_start_token_id, self._system_token_id, self._nl_token_id]
            )
        ]
        if system_instruct_ids:
            parts.append(self._text_embeds(system_instruct_ids))

        prompt_ids = (
            []
            if prompt_speaker_codes is not None
            else self._speaker_system_prompt_id.get(speaker_name or "")
        )
        if prompt_ids:
            parts.append(self._text_embeds(prompt_ids))

        parts.append(self._codec_embeds([self._codec_bos_id]))
        parts.append(speaker_embeds.to(device=self._device, dtype=self._dtype))
        if self._codec_eos_id is not None:
            parts.append(self._codec_embeds([self._codec_eos_id]))
        parts.append(self._text_embeds([self._im_end_token_id, self._nl_token_id]))
        # This is the Qwen3.5 speaker/system conditioning prefix. Later
        # user/assistant parts still reuse sglang-omni's existing talker prefill
        # queue and streaming append logic.
        return torch.cat(parts, dim=0)


def make_talker_scheduler_adapters(
    *,
    tokenizer: Any,
    codec_vocab_size: int,
    model: Any,
    model_path: str,
    thinker_config: Any,
    required_aux_hidden_key: int,
    codec_bos_id: int = 2149,
    codec_eos_id: int | None = None,
    codec_nothink_id: int = 2155,
    codec_think_id: int | None = None,
    codec_think_bos_id: int = 2156,
    codec_think_eos_id: int = 2157,
    codec_pad_id: int = 2148,
    audio_token_id: int | None = None,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
    tts_bos_token_id: int = 151672,
    tts_eos_token_id: int = 151673,
    tts_pad_token_id: int = 151671,
    im_start_token_id: int = 151644,
    im_end_token_id: int = 151645,
    nl_token_id: int = 198,
    system_token_id: int = 8948,
    user_token_id: int = 872,
    assistant_token_id: int = 77091,
    speaker_map: dict[str, int] | None = None,
    talker_language_id: dict[str, int] | None = None,
    talker_assistant_prompt_id_mapping: dict[str, list[int]] | None = None,
    speaker_system_prompt_id: dict[str, list[int]] | None = None,
    max_thinker_to_talker_mm_tokens: int | None = None,
):
    """Build Qwen3.5 talker adapters with Next-style text embeddings."""
    # The aux hidden layer needed by the talker has already been selected by
    # accept_hidden_layer in the thinker stream builder. Keep this parameter for
    # interface alignment.
    prefill_builder = Qwen35TalkerPrefillBuilder(
        model=model,
        model_path=model_path,
        nl_token_id=nl_token_id,
        codec_eos_id=codec_eos_id,
        codec_think_id=codec_think_id,
        talker_language_id=talker_language_id,
        talker_assistant_prompt_id_mapping=talker_assistant_prompt_id_mapping,
        speaker_system_prompt_id=speaker_system_prompt_id,
        max_thinker_to_talker_mm_tokens=max_thinker_to_talker_mm_tokens,
        tokenizer=tokenizer,
        thinker_config=thinker_config,
        audio_token_id=audio_token_id,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        tts_bos_token_id=tts_bos_token_id,
        tts_eos_token_id=tts_eos_token_id,
        tts_pad_token_id=tts_pad_token_id,
        im_start_token_id=im_start_token_id,
        im_end_token_id=im_end_token_id,
        system_token_id=system_token_id,
        user_token_id=user_token_id,
        assistant_token_id=assistant_token_id,
        codec_bos_id=codec_bos_id,
        codec_nothink_id=codec_nothink_id,
        codec_think_bos_id=codec_think_bos_id,
        codec_think_eos_id=codec_think_eos_id,
        codec_pad_id=codec_pad_id,
        speaker_map=speaker_map,
    )

    def _resolve_talker_sampling_config(params: dict[str, Any]) -> dict[str, Any]:
        params = _as_params_mapping(params)
        talker_sampling = _resolve_nested_sampling_params(
            params,
            nested_key="talker_params",
            prefix="talker",
            defaults={
                # Use the steady-state Qwen3.5 talker sampling defaults;
                # audio-output requests default talker max_tokens to 2048.
                # Request-level talker_max_tokens/talker_params.max_tokens can
                # still override it explicitly.
                "max_new_tokens": 2048,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
                "min_p": 0.0,
                "repetition_penalty": 1.05,
            },
            fallback_base_seed=False,
        )
        model_codec_eos_id = getattr(model.config, "codec_eos_token_id", None)
        if model_codec_eos_id is None:
            resolved_codec_eos_id = (
                int(codec_eos_id) if codec_eos_id is not None else -1
            )
        else:
            resolved_codec_eos_id = int(model_codec_eos_id)
        max_new_tokens = _resolve_qwen35_talker_max_new_tokens(
            params,
            talker_sampling["max_new_tokens"],
        )
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": talker_sampling["temperature"],
            "top_k": talker_sampling["top_k"],
            "top_p": talker_sampling["top_p"],
            "min_p": talker_sampling["min_p"],
            "repetition_penalty": talker_sampling["repetition_penalty"],
            "codec_eos_id": (
                resolved_codec_eos_id if resolved_codec_eos_id >= 0 else None
            ),
            "suppress_tokens": [],
            "seed": talker_sampling["seed"],
        }

    def request_builder(payload):
        params = _as_params_mapping(payload.request.params)
        subtalker_sampling = _to_subtalker_sampling_namespace(
            _resolve_nested_sampling_params(
                params,
                nested_key="subtalker_params",
                prefix="subtalker",
                defaults={
                    # Residual codecs use independent sampling defaults. Do not
                    # inherit the main request or main talker sampling settings.
                    "max_new_tokens": 0,
                    "temperature": 0.1,
                    "top_k": 5,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "repetition_penalty": 1.05,
                },
                fallback_base_seed=False,
            )
        )
        req_data = _build_talker_request_data(
            payload,
            prefill_builder=prefill_builder,
            tokenizer=tokenizer,
            codec_vocab_size=codec_vocab_size,
            codec_bos_id=codec_bos_id,
            audio_token_id=audio_token_id,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            thinker_config=thinker_config,
            resolve_sampling_config=_resolve_talker_sampling_config,
            mrope_position_builder=_compute_qwen35_mrope_positions,
        )
        if hasattr(req_data, "req"):
            req_data.req._qwen35_subtalker_sampling_params = subtalker_sampling
            req_data.subtalker_sampling_params = subtalker_sampling
            req_data.talker_decode_input_mode = "qwen35_external"
            feedback_stride = _qwen35_talker_text_feedback_stride()
            req_data.talker_text_feedback_stride = feedback_stride
            req_data.talker_text_feedback_countdown = feedback_stride
            req_data.talker_text_chunk_size = _QWEN35_TALKER_NUM_OUTPUT_IN_CHUNK
            req_data.talker_text_chunk_remaining = 0
            req_data.talker_text_outputs_to_drop = 0
            req_data.last_talker_decode_should_emit = True
        return req_data

    def result_adapter(data):
        payload = data.stage_payload
        return payload.__class__(
            request_id=payload.request_id,
            request=payload.request,
            data=payload.data,
        )

    return (
        request_builder,
        result_adapter,
        prefill_builder.append_text_chunk,
        prefill_builder.mark_thinker_done,
    )
