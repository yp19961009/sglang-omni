# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for local Qwen3.5-Omni checkpoints."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from sglang_omni.models.qwen3_5_omni.components.code2wav_scheduler import (
    QWEN35_CODE2WAV_DIT_QUANTS,
    QWEN35_CODE2WAV_ODEINT_METHODS,
    _coerce_code2wav_frequency,
    _resolve_code2wav_files,
)
from sglang_omni.models.qwen3_5_omni.config import (
    QWEN3_5_OMNI_ARCH,
    QWEN3_5_OMNI_ARCH_ALIASES,
)

Severity = Literal["error", "warning", "info"]

_EXPECTED_MODEL_TYPE = "qwen3_omni_next"
_THINKER_MODEL_TYPES = (
    "qwen3_omni_next",
    "qwen3_omni_next_thinker",
    "qwen3_omni_next_thinker_mtp",
)
_TALKER_MODEL_TYPES = ("qwen3_omni_next", "qwen3_omni_next_talker")
_TALKER_ARCHES = {
    "Qwen3OmniNextTalkerModel",
    "Qwen3OmniNextTalkerForConditionalGeneration",
    "Qwen3OmniNextMoeTalkerForConditionalGeneration",
}
_ROOT_REQUIRED_FOR_SPEECH = (
    "tts_bos_token_id",
    "tts_eos_token_id",
    "tts_pad_token_id",
    "im_start_token_id",
    "im_end_token_id",
    "system_token_id",
    "user_token_id",
    "assistant_token_id",
)
_THINKER_REQUIRED_FOR_SPEECH = (
    "audio_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
)
_THINKER_END_TOKEN_ALIASES = ("vision_end_token_id", "video_end_token_id")
_THINKER_REQUIRED_CONFIGS_FOR_SPEECH = (
    "audio_config",
    "vision_config",
)
_AUDIO_REQUIRED_POSITIVE_INT_FIELDS = (
    "d_model",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "encoder_layers",
    "num_mel_bins",
    "max_source_positions",
    "n_window",
    "n_window_infer",
    "downsample_hidden_size",
    "output_dim",
)
_AUDIO_OPTIONAL_POSITIVE_INT_FIELDS = (
    "downsample_times",
    "downsample_chunk_size",
    "chunk_size",
    "conv_chunksize",
)
_AUDIO_ACTIVATIONS = ("gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "silu", "swish")
_THINKER_TEXT_REQUIRED_POSITIVE_INT_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
)
_VISION_REQUIRED_POSITIVE_INT_FIELDS = (
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "out_hidden_size",
    "num_position_embeddings",
)
_TALKER_TEXT_REQUIRED_POSITIVE_INT_FIELDS = (
    "vocab_size",
    "text_vocab_size",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
)
_CODE_PREDICTOR_REQUIRED_POSITIVE_INT_FIELDS = (
    "num_code_groups",
    "vocab_size",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_hidden_layers",
    "intermediate_size",
)
_CODE_PREDICTOR_OPTIONAL_POSITIVE_INT_FIELDS = (
    "head_dim",
    "talker_hidden_size",
)
_CODE_PREDICTOR_ACTIVATIONS = ("silu", "swish", "gelu", "relu")
_TALKER_CODEC_TOKEN_FIELDS = (
    "codec_bos_id",
    "codec_eos_token_id",
    "codec_nothink_id",
    "codec_think_bos_id",
    "codec_think_eos_id",
    "codec_pad_id",
)
_TALKER_OPTIONAL_CODEC_TOKEN_FIELDS = ("codec_think_id",)
_ROOT_TALKER_TEXT_TOKEN_FIELDS = (
    *_ROOT_REQUIRED_FOR_SPEECH,
    "nl_token_id",
)
_OPTIONAL_CODEC_ID_MAP_FIELDS = (
    "speaker_id",
    "talker_language_id",
)
_OPTIONAL_TEXT_TOKEN_ID_LIST_MAP_FIELDS = (
    "speaker_system_prompt_id",
    "talker_assistant_prompt_id_mapping",
)
_OPTIONAL_RUNTIME_POSITIVE_INT_FIELDS = ("max_thinker_to_talker_mm_tokens",)
_TALKER_REQUIRED_FOR_SPEECH = (
    "accept_hidden_layer",
    "codec_bos_id",
    "codec_eos_token_id",
    "codec_nothink_id",
    "codec_think_bos_id",
    "codec_think_eos_id",
    "codec_pad_id",
    "text_config",
)
_HF_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_HF_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_PROCESSOR_CONFIG_FILENAMES = ("processor_config.json", "preprocessor_config.json")
_TOKENIZER_CONFIG_FILENAMES = ("tokenizer_config.json",)
_TOKENIZER_VOCAB_FILENAMES = ("tokenizer.json", "tokenizer.model", "vocab.json")
_CHAT_TEMPLATE_FILENAMES = ("chat_template.json", "chat_template.jinja")
_HF_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_MTP_WEIGHT_PREFIXES = ("thinker.mtp.", "mtp.")
_THINKER_SUBDIR_CANDIDATES = ("thinker",)
_TALKER_SUBDIR_CANDIDATES = ("talker_lm", "talker")
_XVECTOR_PROMPT_CODE_KEYS = (
    "prompt_code",
    "prompt_speaker_codes",
    "prompt_codes",
    "ref_code",
    "speaker_codec_codes",
    "voice_clone_codes",
)
_XVECTOR_INFO_SYSTEM_INSTRUCT_KEYS = (
    "talker_system_instruct",
    "system_instruct",
    "voice_clone_system_instruct",
    "system_instruct_ids",
)
_XVECTOR_INFO_LANGUAGE_KEYS = (
    "language_type",
    "language",
    "lang",
    "target_language",
    "target_lang",
    "prompt_language",
    "ref_language",
)


@dataclass(frozen=True)
class PreflightIssue:
    severity: Severity
    path: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    model_path: str
    issues: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def run_qwen35_preflight(
    model_path: str,
    *,
    speech: bool = True,
    code2wav_model_path: str | None = None,
    xvector_info_paths: Sequence[str] | None = None,
    validate_xvector_pickle: bool = False,
) -> PreflightReport:
    """Validate a local Qwen3.5-Omni checkpoint layout before launching."""

    root = Path(model_path)
    issues: list[PreflightIssue] = []
    if not root.exists():
        issues.append(
            _issue(
                "error",
                root,
                "model path does not exist; preflight only supports local files",
            )
        )
        return PreflightReport(model_path=model_path, issues=tuple(issues))
    if not root.is_dir():
        issues.append(_issue("error", root, "model path must be a directory"))
        return PreflightReport(model_path=model_path, issues=tuple(issues))

    root_config = _load_json(root / "config.json", issues)
    if root_config is None:
        return PreflightReport(model_path=model_path, issues=tuple(issues))

    thinker_config_path = _sub_config_path(
        root,
        root_config,
        attr="thinker_config",
        subdirs=_THINKER_SUBDIR_CANDIDATES,
    )
    thinker_config = _resolve_sub_config(
        root,
        root_config,
        attr="thinker_config",
        subdirs=_THINKER_SUBDIR_CANDIDATES,
        issues=issues,
    )
    _check_root_identity(root, root_config, issues)
    if thinker_config_path != root / "config.json":
        _check_split_model_type(
            thinker_config_path,
            thinker_config,
            _THINKER_MODEL_TYPES,
            issues,
            label="split thinker config",
        )
    _check_processor_assets(
        _stage_weight_dir(root, _THINKER_SUBDIR_CANDIDATES),
        thinker_config if thinker_config_path != root / "config.json" else root_config,
        issues,
    )
    _check_thinker_config(thinker_config_path, thinker_config, issues)
    _check_hf_weight_files(
        _stage_weight_dir(root, _THINKER_SUBDIR_CANDIDATES),
        issues,
        label="thinker/SGLang AR weights",
    )
    _check_mtp_assets(root, thinker_config, issues, config_path=thinker_config_path)

    if speech:
        talker_config_path = _sub_config_path(
            root,
            root_config,
            attr="talker_config",
            subdirs=_TALKER_SUBDIR_CANDIDATES,
        )
        talker_config = _resolve_sub_config(
            root,
            root_config,
            attr="talker_config",
            subdirs=_TALKER_SUBDIR_CANDIDATES,
            issues=issues,
        )
        if talker_config_path != root / "config.json":
            _check_split_model_type(
                talker_config_path,
                talker_config,
                _TALKER_MODEL_TYPES,
                issues,
                label="split talker config",
            )
        _check_hf_weight_files(
            _stage_weight_dir(root, _TALKER_SUBDIR_CANDIDATES),
            issues,
            label="talker/SGLang AR weights",
        )
        _check_required_fields(
            root / "config.json",
            root_config,
            _ROOT_REQUIRED_FOR_SPEECH,
            issues,
            label="root speech metadata",
        )
        talker_vocab_size, talker_text_vocab_size = _resolve_talker_vocab_sizes(
            talker_config
        )
        _check_runtime_prompt_metadata(
            root / "config.json",
            root_config,
            issues,
            label="root speech metadata",
            codec_vocab_size=talker_vocab_size,
            text_vocab_size=talker_text_vocab_size,
        )
        _check_root_talker_text_token_ranges(
            root / "config.json",
            root_config,
            talker_text_vocab_size=talker_text_vocab_size,
            issues=issues,
        )
        _check_runtime_prompt_metadata(
            thinker_config_path,
            thinker_config,
            issues,
            label="thinker speech metadata",
            codec_vocab_size=talker_vocab_size,
            text_vocab_size=talker_text_vocab_size,
        )
        _check_required_fields(
            thinker_config_path,
            thinker_config,
            _THINKER_REQUIRED_FOR_SPEECH,
            issues,
            label="thinker speech metadata",
        )
        _check_required_any_field(
            thinker_config_path,
            thinker_config,
            _THINKER_END_TOKEN_ALIASES,
            issues,
            label="thinker speech metadata",
        )
        _check_required_dict_fields(
            thinker_config_path,
            thinker_config,
            _THINKER_REQUIRED_CONFIGS_FOR_SPEECH,
            issues,
            label="thinker speech metadata",
        )
        _check_audio_config(thinker_config_path, thinker_config, issues)
        _check_vision_config(thinker_config_path, thinker_config, issues)
        _check_talker_config(
            talker_config,
            thinker_config,
            issues,
            talker_config_path=talker_config_path,
            thinker_config_path=thinker_config_path,
        )
        _check_voice_map(
            _stage_weight_dir(root, _TALKER_SUBDIR_CANDIDATES),
            talker_config,
            issues,
        )
        _check_code2wav(root, code2wav_model_path, talker_config, issues)
    else:
        issues.append(_issue("info", root, "speech checks skipped"))

    if xvector_info_paths:
        _check_xvector_info_paths(
            xvector_info_paths,
            issues,
            validate_pickle=validate_xvector_pickle,
        )

    return PreflightReport(model_path=model_path, issues=tuple(issues))


def format_preflight_report(report: PreflightReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [f"Qwen3.5-Omni preflight {status}: {report.model_path}"]
    if not report.issues:
        lines.append("  [info] no issues found")
        return "\n".join(lines)
    for issue in report.issues:
        lines.append(f"  [{issue.severity}] {issue.path}: {issue.message}")
    return "\n".join(lines)


def _issue(severity: Severity, path: Path, message: str) -> PreflightIssue:
    return PreflightIssue(severity=severity, path=str(path), message=message)


def _load_json(path: Path, issues: list[PreflightIssue]) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(_issue("error", path, "missing config.json"))
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        issues.append(_issue("error", path, f"invalid JSON: {exc}"))
        return None
    if not isinstance(data, dict):
        issues.append(_issue("error", path, "config.json must contain an object"))
        return None
    return data


def _stage_weight_dir(root: Path, subdirs: tuple[str, ...]) -> Path:
    for subdir in subdirs:
        split_config = root / subdir / "config.json"
        if split_config.is_file():
            return split_config.parent
    return root


def _sub_config_path(
    root: Path,
    root_config: dict[str, Any],
    *,
    attr: str,
    subdirs: tuple[str, ...],
) -> Path:
    if isinstance(root_config.get(attr), dict):
        return root / "config.json"
    for subdir in subdirs:
        split_config = root / subdir / "config.json"
        if split_config.is_file():
            return split_config
    label = "/".join(subdirs)
    return root / f"{label}/config.json"


def _has_hf_weight_files(directory: Path) -> bool:
    for name in _HF_WEIGHT_FILENAMES:
        if (directory / name).is_file():
            return True
    if not directory.is_dir():
        return False
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("model-") and path.suffix in _HF_WEIGHT_SUFFIXES:
            return True
        if path.name.startswith("pytorch_model-") and path.suffix == ".bin":
            return True
    return False


def _has_any_file(root: Path, names: tuple[str, ...]) -> bool:
    return any((root / name).is_file() for name in names)


def _check_processor_assets(
    root: Path,
    root_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    if not _has_any_file(root, _PROCESSOR_CONFIG_FILENAMES):
        issues.append(
            _issue(
                "warning",
                root,
                "missing processor config; AutoProcessor.from_pretrained may fail "
                "without processor_config.json or preprocessor_config.json",
            )
        )
    if not _has_any_file(root, _TOKENIZER_CONFIG_FILENAMES):
        issues.append(
            _issue(
                "warning",
                root,
                "missing tokenizer_config.json; tokenizer startup may fail",
            )
        )
    if not _has_any_file(root, _TOKENIZER_VOCAB_FILENAMES):
        issues.append(
            _issue(
                "warning",
                root,
                "missing tokenizer vocabulary; expected tokenizer.json, "
                "tokenizer.model, or vocab.json",
            )
        )
    if not _has_chat_template(root, root_config, issues):
        issues.append(
            _issue(
                "warning",
                root,
                "missing chat template in config/processor/tokenizer assets; "
                "Qwen3.5 preprocessor has no Qwen3 fallback template",
            )
        )


def _has_chat_template(
    root: Path,
    root_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> bool:
    if _is_nonempty_text(root_config.get("chat_template")):
        return True
    if _has_any_nonempty_file(root, _CHAT_TEMPLATE_FILENAMES):
        return True

    for name in (*_PROCESSOR_CONFIG_FILENAMES, *_TOKENIZER_CONFIG_FILENAMES):
        path = root / name
        if not path.is_file():
            continue
        data = _load_optional_json(path, issues)
        if isinstance(data, dict) and _is_nonempty_text(data.get("chat_template")):
            return True
    return False


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_any_nonempty_file(root: Path, names: tuple[str, ...]) -> bool:
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _load_optional_json(
    path: Path,
    issues: list[PreflightIssue],
) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        issues.append(_issue("warning", path, f"could not parse JSON: {exc}"))
        return None
    return data if isinstance(data, dict) else None


def _check_hf_weight_files(
    directory: Path,
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    if _has_hf_weight_files(directory):
        _check_hf_index_shards(directory, issues, label=label)
        return
    issues.append(
        _issue(
            "error",
            directory,
            f"missing {label}; expected HF weight files such as "
            "model.safetensors, model-*.safetensors, or pytorch_model*.bin",
        )
    )


def _check_hf_index_shards(
    directory: Path,
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    for index_path in _iter_hf_index_paths(directory):
        try:
            data = json.loads(index_path.read_text())
        except Exception as exc:
            issues.append(_issue("error", index_path, f"invalid HF weight index: {exc}"))
            continue
        if not isinstance(data, dict):
            issues.append(
                _issue("error", index_path, "HF weight index must contain an object")
            )
            continue
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            issues.append(
                _issue("error", index_path, "HF weight index missing weight_map object")
            )
            continue

        invalid_values = [
            name
            for name, shard in weight_map.items()
            if not isinstance(name, str) or not isinstance(shard, str)
        ]
        if invalid_values:
            preview = ", ".join(str(name) for name in invalid_values[:3])
            issues.append(
                _issue(
                    "error",
                    index_path,
                    "HF weight index weight_map must map string weight names to "
                    f"string shard filenames; invalid entries: {preview}",
                )
            )
            continue

        unsafe_shards = sorted(
            {
                shard
                for shard in weight_map.values()
                if not shard
                or Path(shard).is_absolute()
                or ".." in Path(shard).parts
            }
        )
        if unsafe_shards:
            preview = ", ".join(unsafe_shards[:5])
            if len(unsafe_shards) > 5:
                preview += f", ... ({len(unsafe_shards)} total)"
            # 中文说明：HF index 里的 shard 应该是模型目录内部的相对文件名；
            # 绝对路径或 ../ 会让预检结果依赖外部路径，后续 loader 行为也不稳定。
            issues.append(
                _issue(
                    "error",
                    index_path,
                    "HF weight index shard filenames must be relative paths inside "
                    f"the model directory; invalid shards: {preview}",
                )
            )
            continue

        missing = sorted(
            {
                shard
                for shard in weight_map.values()
                if not (index_path.parent / shard).is_file()
            }
        )
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... ({len(missing)} total)"
            # 中文说明：大模型目录经常先传 index 再传 shard；如果少了某个
            # 分片，真正启动时会在 HF/SGLang loader 深处报错。preflight
            # 提前检查 weight_map 引用，直接指出缺哪个 shard。
            issues.append(
                _issue(
                    "error",
                    index_path,
                    f"{label} index references missing shard files: {preview}",
                )
            )


def _check_root_identity(
    root: Path,
    config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    archs = tuple(config.get("architectures") or ())
    known_archs = (QWEN3_5_OMNI_ARCH, *QWEN3_5_OMNI_ARCH_ALIASES)
    model_type = config.get("model_type")
    if model_type == _EXPECTED_MODEL_TYPE or any(arch in known_archs for arch in archs):
        return
    issues.append(
        _issue(
            "warning",
            root / "config.json",
            "not obviously Qwen3.5-Omni: expected model_type="
            f"{_EXPECTED_MODEL_TYPE!r} or architectures in {known_archs}",
        )
    )


def _check_split_model_type(
    path: Path,
    config: dict[str, Any],
    allowed: tuple[str, ...],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    model_type = config.get("model_type")
    if model_type in allowed:
        return
    if model_type is None:
        message = (
            f"{label} missing model_type; SGLang AutoConfig.from_pretrained "
            f"needs one of {allowed} for split AR stages"
        )
    else:
        message = (
            f"{label} has unsupported model_type={model_type!r}; expected "
            f"one of {allowed}"
        )
    issues.append(_issue("error", path, message))


def _resolve_sub_config(
    root: Path,
    root_config: dict[str, Any],
    *,
    attr: str,
    subdirs: tuple[str, ...],
    issues: list[PreflightIssue],
) -> dict[str, Any]:
    value = root_config.get(attr)
    if isinstance(value, dict):
        return value
    for subdir in subdirs:
        split_config = root / subdir / "config.json"
        if split_config.is_file():
            data = _load_json(split_config, issues)
            return data or {}
    # 中文说明：有些 split checkpoint 的子目录未拆出 config；这时先用 root
    # config 做 direct-subconfig fallback，后续字段检查会给出更具体的缺口。
    label = "/".join(subdirs)
    issues.append(
        _issue(
            "warning",
            root / f"{label}/config.json",
            f"missing split {label} config; falling back to root config",
        )
    )
    return root_config


def _check_thinker_config(
    path: Path,
    thinker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    text_config = thinker_config.get("text_config")
    if not isinstance(text_config, dict):
        issues.append(
            _issue(
                "error",
                path,
                "thinker_config must provide text_config for SGLang ModelConfig",
            )
        )
        return
    _check_text_config_positive_fields(
        path,
        text_config,
        _THINKER_TEXT_REQUIRED_POSITIVE_INT_FIELDS,
        issues,
        label="thinker text_config",
    )
    _check_qwen3_next_text_runtime_fields(
        path,
        text_config,
        issues,
        label="thinker text_config",
    )


def _check_mtp_assets(
    root: Path,
    thinker_config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    config_path: Path,
) -> None:
    mtp_layers = _resolve_mtp_num_hidden_layers(thinker_config)
    if mtp_layers and mtp_layers > 0:
        issues.append(
            _issue(
                "warning",
                config_path,
                "thinker text_config declares "
                f"mtp_num_hidden_layers={mtp_layers}; current sglang-omni "
                "Qwen3.5 path runs the base thinker AR model and ignores "
                "MTP/draft weights",
            )
        )

    index_paths = list(
        _iter_hf_index_paths(root, _stage_weight_dir(root, _THINKER_SUBDIR_CANDIDATES))
    )
    mtp_weight_names = _collect_mtp_weight_names(index_paths, issues)
    if mtp_weight_names:
        preview = ", ".join(mtp_weight_names[:3])
        if len(mtp_weight_names) > 3:
            preview += ", ..."
        issues.append(
            _issue(
                "warning",
                root,
                "detected Qwen3.5 MTP weights in HF index "
                f"({preview}); they are intentionally skipped by the current "
                "SGLang thinker loader",
            )
        )


def _resolve_mtp_num_hidden_layers(config: dict[str, Any]) -> int | None:
    text_config = config.get("text_config")
    for candidate in (text_config, config):
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("mtp_num_hidden_layers")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _iter_hf_index_paths(*directories: Path) -> Iterator[Path]:
    seen: set[Path] = set()
    for directory in directories:
        for name in _HF_INDEX_FILENAMES:
            path = directory / name
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                yield path


def _collect_mtp_weight_names(
    index_paths: list[Path],
    issues: list[PreflightIssue],
) -> list[str]:
    names: list[str] = []
    for path in index_paths:
        data = _load_optional_json(path, issues)
        if not isinstance(data, dict):
            continue
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            continue
        for name in weight_map:
            if isinstance(name, str) and name.startswith(_MTP_WEIGHT_PREFIXES):
                names.append(name)
    return names


def _check_talker_config(
    talker_config: dict[str, Any],
    thinker_config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    talker_config_path: Path,
    thinker_config_path: Path,
) -> None:
    archs = set(talker_config.get("architectures") or ())
    if archs and not (archs & _TALKER_ARCHES):
        issues.append(
            _issue(
                "warning",
                talker_config_path,
                f"unexpected talker architectures: {sorted(archs)}",
            )
        )
    _check_accept_hidden_layer(
        talker_config,
        thinker_config,
        issues,
        talker_config_path=talker_config_path,
        thinker_config_path=thinker_config_path,
    )
    _check_required_fields(
        talker_config_path,
        talker_config,
        _TALKER_REQUIRED_FOR_SPEECH,
        issues,
        label="talker speech metadata",
    )
    _check_talker_text_config(talker_config_path, talker_config, issues)
    _check_code_predictor_config(talker_config_path, talker_config, issues)
    _check_talker_num_code_groups(talker_config_path, talker_config, issues)
    codec_vocab_size, text_vocab_size = _resolve_talker_vocab_sizes(talker_config)
    _check_runtime_prompt_metadata(
        talker_config_path,
        talker_config,
        issues,
        label="talker speech metadata",
        codec_vocab_size=codec_vocab_size,
        text_vocab_size=text_vocab_size,
    )
    if not isinstance(talker_config.get("speaker_id"), dict):
        issues.append(
            _issue(
                "warning",
                talker_config_path,
                "speaker_id is missing; standard named voices may fail",
            )
        )


def _iter_voice_map_paths(talker_dir: Path) -> Iterator[Path]:
    seen: set[Path] = set()
    for path in (talker_dir / "voice_map.json", talker_dir.parent / "voice_map.json"):
        if path in seen:
            continue
        seen.add(path)
        yield path


def _check_voice_map(
    talker_dir: Path,
    talker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    speaker_map = talker_config.get("speaker_id")
    for path in _iter_voice_map_paths(talker_dir):
        if not path.is_file():
            continue
        data = _load_optional_json(path, issues)
        if data is None:
            continue
        if not isinstance(speaker_map, dict):
            issues.append(
                _issue(
                    "warning",
                    path,
                    "voice_map.json cannot be validated because speaker_id is missing",
                )
            )
            continue
        for voice_name, speaker_code in data.items():
            if not isinstance(voice_name, str) or not voice_name.strip():
                issues.append(
                    _issue(
                        "warning",
                        path,
                        f"voice_map.json key must be a non-empty string, got {voice_name!r}",
                    )
                )
                continue
            if not isinstance(speaker_code, str) or not speaker_code.strip():
                issues.append(
                    _issue(
                        "warning",
                        path,
                        f"voice_map.json[{voice_name!r}] must be a non-empty speaker string",
                    )
                )
                continue
            normalized = speaker_code.lower().strip()
            if normalized not in speaker_map:
                # The shipped Qwen3.5 voice_map may include aliases for voices
                # that are not present in a given smaller talker checkpoint. This
                # only affects requests that choose that alias, so keep preflight
                # useful without blocking otherwise runnable checkpoints.
                issues.append(
                    _issue(
                        "warning",
                        path,
                        f"voice_map.json[{voice_name!r}] maps to {normalized!r}, "
                        "which is not in talker_config.speaker_id",
                    )
                )


def _resolve_talker_num_code_groups(talker_config: dict[str, Any]) -> Any:
    value = talker_config.get("num_code_groups")
    if value is not None:
        return value
    code_predictor_config = talker_config.get("code_predictor_config")
    if isinstance(code_predictor_config, dict):
        return code_predictor_config.get("num_code_groups")
    return None


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _check_text_config_positive_fields(
    path: Path,
    text_config: dict[str, Any],
    fields: tuple[str, ...],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> dict[str, int]:
    missing = [field for field in fields if text_config.get(field) is None]
    if missing:
        issues.append(
            _issue(
                "error",
                path,
                f"{label} missing required fields: {', '.join(missing)}",
            )
        )

    values: dict[str, int] = {}
    for field in fields:
        value = text_config.get(field)
        if value is None:
            continue
        parsed = _parse_positive_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field} must be positive, got {value!r}",
                )
            )
        else:
            values[field] = parsed

    hidden_size = values.get("hidden_size")
    num_attention_heads = values.get("num_attention_heads")
    if (
        hidden_size is not None
        and num_attention_heads is not None
        and hidden_size % num_attention_heads != 0
    ):
        issues.append(
            _issue(
                "error",
                path,
                f"{label}.hidden_size must be divisible by num_attention_heads: "
                f"{hidden_size} % {num_attention_heads} != 0",
            )
        )

    num_key_value_heads = values.get("num_key_value_heads")
    if (
        num_attention_heads is not None
        and num_key_value_heads is not None
        and num_attention_heads % num_key_value_heads != 0
    ):
        # 中文说明：Qwen3Next 的 KV head 会被 tensor/pipeline runtime 用来
        # 推导 attention head 分组；不整除时晚到建模阶段才失败。
        issues.append(
            _issue(
                "error",
                path,
                f"{label}.num_attention_heads must be divisible by "
                "num_key_value_heads: "
                f"{num_attention_heads} % {num_key_value_heads} != 0",
            )
        )
    return values


def _check_qwen3_next_text_runtime_fields(
    path: Path,
    text_config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    # 中文说明：这些字段不是普通 shape metadata，而是当前 SGLang
    # Qwen3Next core 启动时会直接读取的运行期约定。提前预检可以避免
    # worker 初始化后才遇到 AttributeError/KeyError/IndexError。
    num_hidden_layers = _parse_positive_int(text_config.get("num_hidden_layers"))
    _check_qwen3_next_layer_types(
        path,
        text_config,
        issues,
        label=label,
        num_hidden_layers=num_hidden_layers,
    )
    _check_qwen3_next_rope_fields(path, text_config, issues, label=label)


def _check_qwen3_next_layer_types(
    path: Path,
    text_config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    label: str,
    num_hidden_layers: int | None,
) -> None:
    for field in ("layers_block_type", "layer_types"):
        value = text_config.get(field)
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            issues.append(
                _issue("error", path, f"{label}.{field} must be a sequence")
            )
            continue
        normalized = [_normalize_qwen3_next_layer_type(item) for item in value]
        unsupported = [
            str(item)
            for item, layer_type in zip(value, normalized)
            if layer_type not in {"attention", "linear_attention"}
        ]
        if unsupported:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field} has unsupported layer type(s): "
                    f"{', '.join(unsupported)}",
                )
            )
        if num_hidden_layers is not None and len(value) != num_hidden_layers:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field} length must equal num_hidden_layers: "
                    f"{len(value)} != {num_hidden_layers}",
                )
            )

    interval = text_config.get("full_attention_interval")
    if interval is None:
        return
    parsed = _parse_positive_int(interval)
    if parsed is None:
        issues.append(
            _issue(
                "error",
                path,
                f"{label}.full_attention_interval must be positive, got {interval!r}",
            )
        )


def _normalize_qwen3_next_layer_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized == "full_attention":
        return "attention"
    return normalized


def _check_qwen3_next_rope_fields(
    path: Path,
    text_config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    rope_source: dict[str, Any] = {}
    for field in ("rope_parameters", "rope_scaling"):
        value = text_config.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            issues.append(_issue("error", path, f"{label}.{field} must be an object"))
            continue
        rope_source.update(value)

    if "rope_theta" in rope_source:
        _check_positive_float(
            path,
            rope_source["rope_theta"],
            issues,
            label=f"{label}.rope_theta",
        )
    if "partial_rotary_factor" in text_config:
        partial_rotary_factor = text_config["partial_rotary_factor"]
    else:
        partial_rotary_factor = rope_source.get("partial_rotary_factor")
    if partial_rotary_factor is not None:
        _check_positive_float(
            path,
            partial_rotary_factor,
            issues,
            label=f"{label}.partial_rotary_factor",
            max_value=1.0,
        )


def _check_positive_float(
    path: Path,
    value: Any,
    issues: list[PreflightIssue],
    *,
    label: str,
    max_value: float | None = None,
) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        issues.append(_issue("error", path, f"{label} must be numeric, got {value!r}"))
        return
    if parsed <= 0:
        issues.append(_issue("error", path, f"{label} must be positive, got {value!r}"))
        return
    if max_value is not None and parsed > max_value:
        issues.append(
            _issue(
                "error",
                path,
                f"{label} must be <= {max_value:g}, got {value!r}",
            )
        )


def _check_talker_num_code_groups(
    path: Path,
    talker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    value = _resolve_talker_num_code_groups(talker_config)
    parsed = _parse_positive_int(value)
    if parsed is not None:
        return
    if value is None:
        message = (
            "talker_config missing num_code_groups; Qwen3.5 talker would "
            "fall back to 1 codec group, which is usually wrong for audio"
        )
    else:
        message = f"talker num_code_groups must be positive, got {value!r}"
    issues.append(_issue("error", path, message))


def _resolve_talker_vocab_sizes(
    talker_config: dict[str, Any],
) -> tuple[int | None, int | None]:
    text_config = talker_config.get("text_config")
    if not isinstance(text_config, dict):
        return None, None
    return (
        _parse_positive_int(text_config.get("vocab_size")),
        _parse_positive_int(text_config.get("text_vocab_size")),
    )


def _check_talker_text_config(
    path: Path,
    talker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    text_config = talker_config.get("text_config")
    if not isinstance(text_config, dict):
        return
    _check_text_config_positive_fields(
        path,
        text_config,
        _TALKER_TEXT_REQUIRED_POSITIVE_INT_FIELDS,
        issues,
        label="talker text_config",
    )
    _check_qwen3_next_text_runtime_fields(
        path,
        text_config,
        issues,
        label="talker text_config",
    )


def _check_code_predictor_config(
    path: Path,
    talker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    code_predictor_config = talker_config.get("code_predictor_config")
    if not isinstance(code_predictor_config, dict):
        issues.append(
            _issue(
                "error",
                path,
                "talker_config must provide code_predictor_config for "
                "Qwen3.5 residual codec groups",
            )
        )
        return

    values = _check_text_config_positive_fields(
        path,
        code_predictor_config,
        _CODE_PREDICTOR_REQUIRED_POSITIVE_INT_FIELDS,
        issues,
        label="talker code_predictor_config",
    )
    _check_qwen3_next_text_runtime_fields(
        path,
        code_predictor_config,
        issues,
        label="talker code_predictor_config",
    )
    for field in _CODE_PREDICTOR_OPTIONAL_POSITIVE_INT_FIELDS:
        value = code_predictor_config.get(field)
        if value is None:
            continue
        parsed = _parse_positive_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    "talker code_predictor_config."
                    f"{field} must be positive, got {value!r}",
                )
            )
        else:
            values[field] = parsed

    top_level_groups = _parse_positive_int(talker_config.get("num_code_groups"))
    code_groups = values.get("num_code_groups")
    if (
        top_level_groups is not None
        and code_groups is not None
        and top_level_groups != code_groups
    ):
        # 中文说明：talker 主链路和 residual predictor 会分别读这两个值；
        # 不一致会导致 buffer 维度和 residual codec group 数量错位。
        issues.append(
            _issue(
                "error",
                path,
                "talker num_code_groups does not match "
                "code_predictor_config.num_code_groups: "
                f"{top_level_groups} != {code_groups}",
            )
        )

    text_config = talker_config.get("text_config")
    talker_hidden_size = None
    talker_vocab_size = None
    if isinstance(text_config, dict):
        talker_hidden_size = _parse_positive_int(text_config.get("hidden_size"))
        talker_vocab_size = _parse_positive_int(text_config.get("vocab_size"))
    _check_code_predictor_talker_contract(
        path,
        values,
        talker_hidden_size=talker_hidden_size,
        talker_vocab_size=talker_vocab_size,
        issues=issues,
    )
    _check_talker_codec_token_ranges(
        path,
        talker_config,
        talker_vocab_size=talker_vocab_size,
        issues=issues,
    )

    hidden_act = code_predictor_config.get("hidden_act")
    if hidden_act is None:
        return
    normalized = str(hidden_act).lower().strip()
    if normalized not in _CODE_PREDICTOR_ACTIVATIONS:
        supported = ", ".join(_CODE_PREDICTOR_ACTIVATIONS)
        issues.append(
            _issue(
                "error",
                path,
                "talker code_predictor_config.hidden_act must be one of "
                f"{supported}, got {hidden_act!r}",
            )
        )


def _check_code_predictor_talker_contract(
    path: Path,
    values: dict[str, int],
    *,
    talker_hidden_size: int | None,
    talker_vocab_size: int | None,
    issues: list[PreflightIssue],
) -> None:
    predictor_hidden_size = values.get("talker_hidden_size") or values.get("hidden_size")
    if (
        talker_hidden_size is not None
        and predictor_hidden_size is not None
        and predictor_hidden_size != talker_hidden_size
    ):
        # 中文说明：subtalker 会把主 talker hidden 和 codec embedding 一起送入
        # predictor；输入维度必须和主 talker hidden_size 对齐。
        issues.append(
            _issue(
                "error",
                path,
                "talker code_predictor_config.talker_hidden_size/default "
                "hidden_size must match talker text_config.hidden_size: "
                f"{predictor_hidden_size} != {talker_hidden_size}",
            )
        )

    predictor_vocab_size = values.get("vocab_size")
    if (
        talker_vocab_size is not None
        and predictor_vocab_size is not None
        and predictor_vocab_size > talker_vocab_size
    ):
        # 中文说明：主 talker logits 会按 code_predictor_config.vocab_size
        # 构造合法 codec id mask；predictor vocab 不能超过主 talker vocab。
        issues.append(
            _issue(
                "error",
                path,
                "talker code_predictor_config.vocab_size must not exceed "
                "talker text_config.vocab_size: "
                f"{predictor_vocab_size} > {talker_vocab_size}",
            )
        )


def _check_talker_codec_token_ranges(
    path: Path,
    talker_config: dict[str, Any],
    *,
    talker_vocab_size: int | None,
    issues: list[PreflightIssue],
) -> None:
    if talker_vocab_size is None:
        return
    for field in (*_TALKER_CODEC_TOKEN_FIELDS, *_TALKER_OPTIONAL_CODEC_TOKEN_FIELDS):
        value = talker_config.get(field)
        if value is None:
            continue
        parsed = _parse_non_negative_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"talker {field} must be a non-negative integer, got {value!r}",
                )
            )
            continue
        if parsed >= talker_vocab_size:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"talker {field} must be in [0, vocab_size), got "
                    f"{parsed} for talker text_config.vocab_size={talker_vocab_size}",
                )
            )


def _check_runtime_prompt_metadata(
    path: Path,
    config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    label: str,
    codec_vocab_size: int | None = None,
    text_vocab_size: int | None = None,
) -> None:
    for field in _OPTIONAL_CODEC_ID_MAP_FIELDS:
        _check_non_negative_int_map(
            path,
            config,
            field,
            issues,
            label=label,
            max_token_id=codec_vocab_size,
            vocab_label="talker codec vocab_size",
        )
    for field in _OPTIONAL_TEXT_TOKEN_ID_LIST_MAP_FIELDS:
        _check_token_id_list_map(
            path,
            config,
            field,
            issues,
            label=label,
            max_token_id=text_vocab_size,
            vocab_label="talker text_vocab_size",
        )
    for field in _OPTIONAL_RUNTIME_POSITIVE_INT_FIELDS:
        value = config.get(field)
        if value is None:
            continue
        parsed = _parse_positive_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field} must be positive, got {value!r}",
                )
            )


def _check_root_talker_text_token_ranges(
    path: Path,
    root_config: dict[str, Any],
    *,
    talker_text_vocab_size: int | None,
    issues: list[PreflightIssue],
) -> None:
    if talker_text_vocab_size is None:
        return
    for field in _ROOT_TALKER_TEXT_TOKEN_FIELDS:
        value = root_config.get(field)
        if value is None:
            continue
        parsed = _parse_non_negative_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"root speech metadata.{field} must be a non-negative "
                    f"integer, got {value!r}",
                )
            )
            continue
        if parsed >= talker_text_vocab_size:
            # 中文说明：TTS/chat special token 会用于 Qwen3.5 talker 文本侧
            # embedding/prompt 构造，必须落在 text_vocab_size 内。
            issues.append(
                _issue(
                    "error",
                    path,
                    f"root speech metadata.{field} must be in "
                    "[0, talker text_vocab_size), got "
                    f"{parsed} for talker text_config.text_vocab_size="
                    f"{talker_text_vocab_size}",
                )
            )


def _check_non_negative_int_map(
    path: Path,
    config: dict[str, Any],
    field: str,
    issues: list[PreflightIssue],
    *,
    label: str,
    max_token_id: int | None = None,
    vocab_label: str = "vocab_size",
) -> None:
    value = config.get(field)
    if value is None:
        return
    if not isinstance(value, dict):
        issues.append(
            _issue(
                "error",
                path,
                f"{label}.{field} must be an object mapping names to token ids",
            )
        )
        return
    for key, token_id in value.items():
        parsed = _parse_non_negative_int(token_id)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field}[{key!r}] must be a non-negative integer, "
                    f"got {token_id!r}",
                )
            )
            continue
        if max_token_id is not None and parsed >= max_token_id:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field}[{key!r}] must be in [0, {vocab_label}), "
                    f"got {parsed} for {vocab_label}={max_token_id}",
                )
            )


def _check_token_id_list_map(
    path: Path,
    config: dict[str, Any],
    field: str,
    issues: list[PreflightIssue],
    *,
    label: str,
    max_token_id: int | None = None,
    vocab_label: str = "vocab_size",
) -> None:
    value = config.get(field)
    if value is None:
        return
    if not isinstance(value, dict):
        issues.append(
            _issue(
                "error",
                path,
                f"{label}.{field} must be an object mapping names to token id lists",
            )
        )
        return
    for key, token_ids in value.items():
        if not isinstance(token_ids, (list, tuple)) or not token_ids:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"{label}.{field}[{key!r}] must be a non-empty token id list",
                )
            )
            continue
        for index, token_id in enumerate(token_ids):
            parsed = _parse_non_negative_int(token_id)
            if parsed is None:
                # 中文说明：这些映射直接用于 talker prompt embedding lookup；
                # 提前检查可以避免请求时才因为非法 id 报错。
                issues.append(
                    _issue(
                        "error",
                        path,
                        f"{label}.{field}[{key!r}][{index}] must be a "
                        f"non-negative integer, got {token_id!r}",
                    )
                )
                continue
            if max_token_id is not None and parsed >= max_token_id:
                issues.append(
                    _issue(
                        "error",
                        path,
                        f"{label}.{field}[{key!r}][{index}] must be in "
                        f"[0, {vocab_label}), got {parsed} for "
                        f"{vocab_label}={max_token_id}",
                    )
                )


def _check_accept_hidden_layer(
    talker_config: dict[str, Any],
    thinker_config: dict[str, Any],
    issues: list[PreflightIssue],
    *,
    talker_config_path: Path,
    thinker_config_path: Path,
) -> None:
    raw_layers = talker_config.get("accept_hidden_layer")
    layers = _parse_accept_hidden_layers(raw_layers)
    if raw_layers is not None and layers is None:
        issues.append(
            _issue(
                "error",
                talker_config_path,
                f"accept_hidden_layer must be an int or int list, got {raw_layers!r}",
            )
        )
        return
    if not layers:
        return

    text_config = thinker_config.get("text_config")
    if not isinstance(text_config, dict):
        return
    num_layers = text_config.get("num_hidden_layers")
    if num_layers is None:
        return
    try:
        num_layers_int = int(num_layers)
    except (TypeError, ValueError):
        issues.append(
            _issue(
                "warning",
                thinker_config_path,
                "cannot validate accept_hidden_layer against "
                f"num_hidden_layers={num_layers!r}",
            )
        )
        return

    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers_int]
    if invalid:
        # 中文说明：SGLang/HF 都按 transformer layer 下标捕获 hidden；
        # 越界时真实启动可能到首个请求才暴露，这里提前报清楚。
        issues.append(
            _issue(
                "error",
                talker_config_path,
                "accept_hidden_layer out of thinker range "
                f"[0, {num_layers_int}): {invalid}",
            )
        )


def _parse_accept_hidden_layers(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            return None
        try:
            return [int(piece) for piece in pieces]
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        layers: list[int] = []
        for item in value:
            parsed = _parse_accept_hidden_layers(item)
            if parsed is None:
                return None
            layers.extend(parsed)
        return layers or None
    return None


def _check_code2wav(
    root: Path,
    code2wav_model_path: str | None,
    talker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    code2wav_root = Path(code2wav_model_path) if code2wav_model_path else root
    files = _resolve_code2wav_files(str(code2wav_root))
    if files is None:
        issues.append(
            _issue(
                "error",
                code2wav_root,
                "could not find code2wav config/checkpoint; pass "
                "--code2wav-model-path or add the codec decoder subdirectory",
            )
        )
        return
    _check_code2wav_codebook_alignment(files, talker_config, issues)
    issues.append(
        _issue(
            "info",
            files.model_dir,
            f"using code2wav config={files.config_path.name} "
            f"checkpoint={files.checkpoint_path.name}",
        )
    )


def _check_code2wav_codebook_alignment(
    files: Any,
    talker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    talker_groups = _parse_positive_int(_resolve_talker_num_code_groups(talker_config))
    if talker_groups is None:
        return
    codebook_nums = _read_code2wav_codebook_nums(files.config_path, issues)
    if codebook_nums is None:
        return
    if codebook_nums == talker_groups:
        return
    # 中文说明：talker 每步输出 K 组 codec code，code2wav DAC 也按 K 个
    # codebook 解码；两边不一致时通常要到首个音频 chunk 才会报维度错误。
    issues.append(
        _issue(
            "error",
            files.config_path,
            "code2wav dac.codebook_nums does not match talker num_code_groups: "
            f"{codebook_nums} != {talker_groups}",
        )
    )


def _read_code2wav_codebook_nums(
    config_path: Path,
    issues: list[PreflightIssue],
) -> int | None:
    data = _load_code2wav_config(config_path, issues)
    if not isinstance(data, dict):
        return None
    value = _get_nested_value(
        data,
        ("model", "dac", "codebook_nums"),
        ("dac", "codebook_nums"),
        ("codebook_nums",),
    )
    if value is None:
        return None
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        parsed_value = None
    if parsed_value == -1:
        issues.append(
            _issue(
                "warning",
                config_path,
                "code2wav dac.codebook_nums is -1; skipping static "
                "codebook alignment check",
            )
        )
        return None
    parsed = _parse_positive_int(value)
    if parsed is None:
        issues.append(
            _issue(
                "error",
                config_path,
                f"code2wav dac.codebook_nums must be positive, got {value!r}",
            )
        )
    return parsed


def _load_code2wav_config(
    config_path: Path,
    issues: list[PreflightIssue],
) -> dict[str, Any] | None:
    try:
        if config_path.suffix.lower() == ".json":
            data = json.loads(config_path.read_text())
        else:
            try:
                import yaml
            except Exception as exc:
                issues.append(
                    _issue(
                        "warning",
                        config_path,
                        f"cannot parse code2wav YAML without PyYAML: {exc}",
                    )
                )
                return None

            def _hparams_constructor(loader: Any, node: Any) -> dict[str, Any]:
                return loader.construct_mapping(node, deep=True)

            for tag in ("utils.utils.HParams", "!utils.utils.HParams"):
                yaml.SafeLoader.add_constructor(tag, _hparams_constructor)
            data = yaml.load(config_path.read_text(), Loader=yaml.SafeLoader)
    except Exception as exc:
        issues.append(
            _issue("warning", config_path, f"could not parse code2wav config: {exc}")
        )
        return None
    if isinstance(data, dict):
        return data
    issues.append(
        _issue(
            "warning",
            config_path,
            "code2wav config is not a mapping; cannot validate codebook count",
        )
    )
    return None


def _get_nested_value(data: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def _has_nonempty_info_key(info: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(info.get(key) not in (None, "") for key in keys)


def _check_xvector_info_paths(
    paths: Sequence[str],
    issues: list[PreflightIssue],
    *,
    validate_pickle: bool = False,
) -> None:
    seen: set[str] = set()
    for raw_path in paths:
        path_text = str(raw_path).strip()
        if not path_text:
            issues.append(
                PreflightIssue(
                    severity="error",
                    path="<empty-xvector-info-path>",
                    message="xvector_info path must not be empty",
                )
            )
            continue
        if path_text in seen:
            continue
        seen.add(path_text)
        _check_xvector_info_path(
            Path(path_text),
            issues,
            validate_pickle=validate_pickle,
        )


def _check_xvector_info_path(
    voice_dir: Path,
    issues: list[PreflightIssue],
    *,
    validate_pickle: bool = False,
) -> None:
    if not voice_dir.exists():
        issues.append(
            _issue(
                "error",
                voice_dir,
                "xvector_info path does not exist; voice clone expects a "
                "directory containing feat.pkl and info.json",
            )
        )
        return
    if not voice_dir.is_dir():
        issues.append(
            _issue(
                "error",
                voice_dir,
                "xvector_info path must be a directory containing feat.pkl "
                "and info.json",
            )
        )
        return

    path_errors = 0
    feat_path = voice_dir / "feat.pkl"
    info_path = voice_dir / "info.json"
    if not feat_path.is_file():
        path_errors += 1
        issues.append(
            _issue(
                "error",
                feat_path,
                "missing feat.pkl; request-time voice clone prompt codes "
                "will fail to load",
            )
        )
    else:
        try:
            if feat_path.stat().st_size == 0:
                issues.append(
                    _issue(
                        "warning",
                        feat_path,
                        "feat.pkl is empty; preflight does not unpickle this "
                        "file, but request-time voice clone loading is likely "
                        "to fail",
                    )
                )
            elif validate_pickle and not _check_xvector_feat_pickle(
                feat_path,
                issues,
            ):
                path_errors += 1
        except OSError as exc:
            issues.append(_issue("warning", feat_path, f"could not stat feat.pkl: {exc}"))

    if not info_path.is_file():
        path_errors += 1
        issues.append(
            _issue(
                "error",
                info_path,
                "missing info.json; voice clone metadata cannot be loaded",
            )
        )
    else:
        info = _load_xvector_info_json(info_path, issues)
        if info is None:
            path_errors += 1
        else:
            if not _has_nonempty_info_key(
                info,
                _XVECTOR_INFO_SYSTEM_INSTRUCT_KEYS,
            ):
                issues.append(
                    _issue(
                        "warning",
                        info_path,
                        "talker system instruct is missing; voice clone will "
                        "fall back to an empty talker system instruction",
                    )
                )
            if not _has_nonempty_info_key(info, _XVECTOR_INFO_LANGUAGE_KEYS):
                issues.append(
                    _issue(
                        "warning",
                        info_path,
                        "voice clone language is missing; voice clone will rely "
                        "on request/default language settings",
                    )
                )

    if path_errors == 0:
        # 中文说明：这里只做轻量存在性/JSON 检查，不 pickle.load(feat.pkl)；
        # 真正的 prompt code key 兼容性默认仍由请求构建阶段统一解析。
        # 用户显式传 --validate-xvector-pickle 时才解析本地 pickle，
        # 避免默认反序列化不受信任资产。
        issues.append(
            _issue(
                "info",
                voice_dir,
                (
                    "xvector_info assets found; feat.pkl content was validated"
                    if validate_pickle
                    else "xvector_info assets found; feat.pkl content will be "
                    "parsed lazily at request time"
                ),
            )
        )


def _check_xvector_feat_pickle(
    feat_path: Path,
    issues: list[PreflightIssue],
) -> bool:
    try:
        with feat_path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        issues.append(_issue("error", feat_path, f"invalid feat.pkl: {exc}"))
        return False
    if not isinstance(data, dict):
        issues.append(_issue("error", feat_path, "feat.pkl must contain a dict"))
        return False
    for key in _XVECTOR_PROMPT_CODE_KEYS:
        if data.get(key) is not None:
            issues.append(
                _issue(
                    "info",
                    feat_path,
                    f"validated voice clone prompt code key: {key}",
                )
            )
            return True
    issues.append(
        _issue(
            "error",
            feat_path,
            "feat.pkl missing prompt code; expected one of "
            + ", ".join(_XVECTOR_PROMPT_CODE_KEYS)
            + ". The current SGLang Qwen3.5 path does not implement the reference "
            "xvector-only zero-shot branch, so prompt codec codes are required.",
        )
    )
    return False


def _load_xvector_info_json(
    path: Path,
    issues: list[PreflightIssue],
) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(_issue("error", path, f"invalid info.json: {exc}"))
        return None
    if not isinstance(data, dict):
        issues.append(_issue("error", path, "info.json must contain an object"))
        return None
    return data


def _check_required_fields(
    path: Path,
    config: dict[str, Any],
    fields: tuple[str, ...],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    missing = [field for field in fields if config.get(field) is None]
    if missing:
        issues.append(
            _issue(
                "error",
                path,
                f"{label} missing required fields: {', '.join(missing)}",
            )
        )


def _check_required_any_field(
    path: Path,
    config: dict[str, Any],
    fields: tuple[str, ...],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    if any(config.get(field) is not None for field in fields):
        return
    issues.append(
        _issue(
            "error",
            path,
            f"{label} missing required field alias: {' or '.join(fields)}",
        )
    )


def _check_required_dict_fields(
    path: Path,
    config: dict[str, Any],
    fields: tuple[str, ...],
    issues: list[PreflightIssue],
    *,
    label: str,
) -> None:
    missing = [
        field for field in fields if not isinstance(config.get(field), dict)
    ]
    if missing:
        issues.append(
            _issue(
                "error",
                path,
                f"{label} missing required object fields: {', '.join(missing)}",
            )
        )


def _check_vision_config(
    path: Path,
    thinker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    vision_config = thinker_config.get("vision_config")
    if not isinstance(vision_config, dict):
        return

    missing = [
        field for field in _VISION_REQUIRED_POSITIVE_INT_FIELDS
        if vision_config.get(field) is None
    ]
    if missing:
        issues.append(
            _issue(
                "error",
                path,
                "thinker vision_config missing required fields: "
                + ", ".join(missing),
            )
        )

    values: dict[str, int] = {}
    for field in _VISION_REQUIRED_POSITIVE_INT_FIELDS:
        value = vision_config.get(field)
        if value is None:
            continue
        parsed = _parse_positive_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"thinker vision_config.{field} must be positive, got {value!r}",
                )
            )
        else:
            values[field] = parsed

    hidden_size = values.get("hidden_size")
    num_heads = values.get("num_heads")
    if hidden_size is not None and num_heads is not None and hidden_size % num_heads != 0:
        issues.append(
            _issue(
                "error",
                path,
                "thinker vision_config.hidden_size must be divisible by "
                f"num_heads: {hidden_size} % {num_heads} != 0",
            )
        )

    hidden_act = vision_config.get("hidden_act")
    if hidden_act is None or not str(hidden_act).strip():
        issues.append(
            _issue(
                "error",
                path,
                "thinker vision_config missing required fields: hidden_act",
            )
        )

    deepstack_indexes = vision_config.get("deepstack_visual_indexes")
    if deepstack_indexes is None:
        issues.append(
            _issue(
                "warning",
                path,
                "thinker vision_config.deepstack_visual_indexes is missing; "
                "deepstack visual features will be disabled unless the HF config "
                "class supplies defaults",
            )
        )
        return
    if not isinstance(deepstack_indexes, list):
        issues.append(
            _issue(
                "error",
                path,
                "thinker vision_config.deepstack_visual_indexes must be a list, "
                f"got {type(deepstack_indexes).__name__}",
            )
        )
        return

    depth = values.get("depth")
    for index, layer in enumerate(deepstack_indexes):
        try:
            layer_index = int(layer)
        except (TypeError, ValueError):
            issues.append(
                _issue(
                    "error",
                    path,
                    "thinker vision_config.deepstack_visual_indexes must contain "
                    f"integer layer indexes, got {layer!r} at position {index}",
                )
            )
            continue
        if layer_index < 0:
            issues.append(
                _issue(
                    "error",
                    path,
                    "thinker vision_config.deepstack_visual_indexes must contain "
                    f"non-negative indexes, got {layer_index} at position {index}",
                )
            )
        if depth is not None and layer_index >= depth:
            # 中文说明：Qwen reference 会按这些 index 从 vision blocks 里取 multiscale
            # hidden states；越界时要等到建模或推理才暴露，preflight 先拦住。
            issues.append(
                _issue(
                    "error",
                    path,
                    "thinker vision_config.deepstack_visual_indexes index "
                    f"{layer_index} is out of range for depth {depth}",
                )
            )


def _check_audio_config(
    path: Path,
    thinker_config: dict[str, Any],
    issues: list[PreflightIssue],
) -> None:
    audio_config = thinker_config.get("audio_config")
    if not isinstance(audio_config, dict):
        return
    missing = [
        field for field in _AUDIO_REQUIRED_POSITIVE_INT_FIELDS
        if audio_config.get(field) is None
    ]
    if missing:
        issues.append(
            _issue(
                "error",
                path,
                "thinker audio_config missing required fields: "
                + ", ".join(missing),
            )
        )

    values: dict[str, int] = {}
    for field in (
        *_AUDIO_REQUIRED_POSITIVE_INT_FIELDS,
        *_AUDIO_OPTIONAL_POSITIVE_INT_FIELDS,
    ):
        value = audio_config.get(field)
        if value is None:
            continue
        parsed = _parse_positive_int(value)
        if parsed is None:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"thinker audio_config.{field} must be positive, got {value!r}",
                )
            )
        else:
            values[field] = parsed

    d_model = values.get("d_model")
    heads = values.get("encoder_attention_heads")
    if d_model is not None:
        if d_model % 2 != 0:
            issues.append(
                _issue(
                    "error",
                    path,
                    f"thinker audio_config.d_model must be even, got {d_model}",
                )
            )
        if heads is not None and d_model % heads != 0:
            issues.append(
                _issue(
                    "error",
                    path,
                    "thinker audio_config.d_model must be divisible by "
                    f"encoder_attention_heads: {d_model} % {heads} != 0",
                )
            )

    activation = audio_config.get("activation_function")
    if activation is None:
        issues.append(
            _issue(
                "error",
                path,
                "thinker audio_config missing required fields: activation_function",
            )
        )
        return
    if str(activation).lower() not in _AUDIO_ACTIVATIONS:
        supported = ", ".join(_AUDIO_ACTIVATIONS)
        issues.append(
            _issue(
                "error",
                path,
                "thinker audio_config.activation_function must be one of "
                f"{supported}, got {activation!r}",
            )
        )
