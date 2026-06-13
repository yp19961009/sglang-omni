#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3.5-Omni offline inference and SGLang/vLLM alignment checks."""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import json
import os
import shlex
import shutil
import signal
import site
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    "/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315"
)
DEFAULT_AUDIO_URL = (
    "https://hci-wlcb.oss-cn-wulanchabu.aliyuncs.com/wumi/qwen-omni/"
    "badcase-two-audios/00004933-00000062.wav"
)
DEFAULT_PROMPT = "请描述这段音频的内容，并用一句话回答。"
DEFAULT_SYSTEM_PROMPT = (
    "You are a virtual voice assistant. Keep replies concise and natural. "
    "Answer in the same language as the user unless the user asks otherwise."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("compare", "sglang", "vllm"),
        default="compare",
        help="Which backend to run. compare runs vLLM then SGLang sequentially.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default="qwen3_5-omni")
    parser.add_argument("--code2wav-model-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "qwen35_omni_alignment"),
    )
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--voice-type", default="tina")
    parser.add_argument("--seed", type=int, default=3408)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--request-timeout", type=float, default=1800.0)

    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--vllm-root", default="/myapp/vllm")
    parser.add_argument("--vllm-docker-image", default=None)
    parser.add_argument("--vllm-docker-python", default="python")
    parser.add_argument("--vllm-docker-mount", default="/home/gangouyu:/myapp")
    parser.add_argument("--vllm-docker-gpus", default="all")
    parser.add_argument("--vllm-docker-extra-arg", action="append", default=[])
    parser.add_argument("--vllm-thinker-devices", default="[0]")
    parser.add_argument("--vllm-talker-devices", default="[1]")
    parser.add_argument("--vllm-code2wav-devices", default="[1]")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-block-size", type=int, default=256)
    parser.add_argument("--vllm-max-model-len", type=int, default=192000)
    parser.add_argument("--vllm-talker-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-talker-gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--disable-vllm-mtp", action="store_true")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--no-vllm-prefix-caching", action="store_true")
    parser.add_argument("--no-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--no-code2wav-torch-compile", action="store_true")
    parser.add_argument(
        "--vllm-default-talker-params",
        action="store_true",
        help="Let vLLM initialize talker SamplingParams from generation_config.json.",
    )
    parser.add_argument(
        "--vllm-thinker-only",
        action="store_true",
        help=(
            "Run only the vLLM thinker engine. This is useful for text/logit "
            "alignment diagnostics and does not produce vLLM audio."
        ),
    )

    parser.add_argument("--sglang-python", default=sys.executable)
    parser.add_argument("--sglang-base-url", default=None)
    parser.add_argument("--sglang-container", default=None)
    parser.add_argument("--sglang-workdir", default="/myapp/sglang-omni")
    parser.add_argument("--launch-sglang", action="store_true")
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--sglang-port", type=int, default=8101)
    parser.add_argument("--sglang-gpu-thinker", type=int, default=0)
    parser.add_argument("--sglang-gpu-talker", type=int, default=1)
    parser.add_argument("--sglang-gpu-code2wav", type=int, default=1)
    parser.add_argument("--sglang-thinker-max-seq-len", type=int, default=192000)
    parser.add_argument("--sglang-server-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--sglang-media-mode",
        choices=("content", "top-level", "both"),
        default="content",
    )
    parser.add_argument(
        "--sglang-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variables exported before launching the SGLang server.",
    )
    parser.add_argument("--sglang-extra-arg", action="append", default=[])
    parser.add_argument("--keep-sglang-server", action="store_true")

    parser.add_argument(
        "--asr-backend",
        choices=("auto", "whisper", "command", "none"),
        default="auto",
    )
    parser.add_argument("--asr-model", default="base")
    parser.add_argument(
        "--asr-language",
        default=None,
        help="Optional Whisper language hint, for example zh or en.",
    )
    parser.add_argument(
        "--asr-command",
        default=None,
        help="Optional shell command. Use {audio} as the wav path placeholder.",
    )
    parser.add_argument("--asr-container", default=None)
    parser.add_argument("--asr-container-mount", default="/home/gangouyu:/myapp")
    parser.add_argument("--wer-threshold", type=float, default=0.35)
    parser.add_argument("--cer-threshold", type=float, default=0.20)
    parser.add_argument(
        "--fallback-text-cer-threshold",
        type=float,
        default=0.50,
        help=(
            "Direct backend text CER threshold used when ASR comparison is "
            "unavailable."
        ),
    )
    parser.add_argument(
        "--duration-ratio-threshold",
        type=float,
        default=0.05,
        help=(
            "Relative output-audio duration threshold used when ASR comparison "
            "is unavailable."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_vllm_logprobs(logprobs: Any) -> list[dict[str, Any] | None]:
    if not logprobs:
        return []

    summary: list[dict[str, Any] | None] = []
    for position, item in enumerate(logprobs):
        if item is None:
            summary.append(None)
            continue
        if not isinstance(item, dict):
            summary.append({"position": position, "raw_type": type(item).__name__})
            continue

        candidates = []
        for token_id, info in item.items():
            candidate: dict[str, Any] = {"token_id": _json_int_or_str(token_id)}
            if hasattr(info, "logprob"):
                candidate["logprob"] = float(info.logprob)
            elif isinstance(info, (int, float)):
                candidate["logprob"] = float(info)
            if hasattr(info, "rank") and info.rank is not None:
                candidate["rank"] = int(info.rank)
            if hasattr(info, "decoded_token") and info.decoded_token is not None:
                candidate["decoded_token"] = str(info.decoded_token)
            candidates.append(candidate)
        candidates.sort(key=lambda x: x.get("logprob", float("-inf")), reverse=True)
        summary.append({"position": position, "candidates": candidates})
    return summary


def tensor_stats_for_debug(tensor: Any, sample_size: int = 8) -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    data = tensor.detach()
    stats_data = data.float()
    last = stats_data.reshape(-1, stats_data.shape[-1])[-1]
    return {
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


def tensor_values_for_debug(tensor: Any, max_items: int = 512) -> Any | None:
    try:
        import torch
    except Exception:
        return None
    if not isinstance(tensor, torch.Tensor) or tensor.numel() > max_items:
        return None
    values = tensor.detach().cpu().tolist()
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        return values
    return values


def append_jsonl_for_debug(path: str, record: dict[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def install_vllm_qwen35_debug_hooks() -> None:
    dump_path = os.getenv("VLLM_QWEN35_HIDDEN_DUMP")
    if not dump_path:
        return
    try:
        import torch
        import vllm.model_executor.models.qwen3_next as qwen3_next
    except Exception as exc:
        append_jsonl_for_debug(
            dump_path,
            {
                "stage": "install_error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return

    cls = qwen3_next.Qwen3NextModel
    if getattr(cls, "_qwen35_hidden_dump_installed", False):
        return
    original_forward = cls.forward

    try:
        expected_tokens = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_EXPECT_TOKENS", "0"))
    except ValueError:
        expected_tokens = 0
    try:
        max_calls = int(os.getenv("VLLM_QWEN35_HIDDEN_DUMP_MAX_CALLS", "1"))
    except ValueError:
        max_calls = 1

    def _forward_with_hidden_dump(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            return original_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        self._qwen35_hidden_dump_calls = dumped_calls + 1
        get_pp_group = qwen3_next.get_pp_group
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
            "source": "vllm",
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
            "input_ids": tensor_values_for_debug(input_ids),
            "positions": tensor_values_for_debug(positions),
        }
        append_jsonl_for_debug(
            dump_path,
            {**base_record, "stage": "embed", "hidden": tensor_stats_for_debug(hidden_states)},
        )

        for layer in qwen3_next.islice(self.layers, self.start_layer, self.end_layer):
            layer_id = getattr(layer, "layer_idx", None)
            if layer_id is None:
                layer_id = qwen3_next.extract_layer_index(getattr(layer, "prefix", ""))
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            record = {
                **base_record,
                "stage": "layer",
                "layer_id": int(layer_id),
                "hidden": tensor_stats_for_debug(hidden_states),
            }
            if isinstance(residual, torch.Tensor):
                record["residual"] = tensor_stats_for_debug(residual)
            append_jsonl_for_debug(dump_path, record)

        if not get_pp_group().is_last_rank:
            return qwen3_next.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        append_jsonl_for_debug(
            dump_path,
            {**base_record, "stage": "norm", "hidden": tensor_stats_for_debug(hidden_states)},
        )
        return hidden_states

    cls.forward = _forward_with_hidden_dump
    cls._qwen35_hidden_dump_installed = True
    append_jsonl_for_debug(dump_path, {"source": "vllm", "stage": "hook_installed"})


def _json_int_or_str(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def code2wav_path(args: argparse.Namespace) -> str:
    if args.code2wav_model_path:
        return args.code2wav_model_path
    return str(Path(args.model_path) / "qwen3_5_omni_codec_decode_online_0306")


def parse_devices(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, int) for item in parsed
    ):
        raise ValueError(f"Device list must be a JSON list of ints, got {value!r}")
    return parsed


def parse_bind_mount(spec: str) -> tuple[Path, str]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Docker mount must be HOST:CONTAINER, got {spec!r}")
    return Path(parts[0]).resolve(), parts[1].rstrip("/")


def map_host_path_to_container(path: str | Path, mount_spec: str) -> str:
    host_root, container_root = parse_bind_mount(mount_spec)
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(host_root)
    except ValueError:
        return str(path)
    return str(Path(container_root) / relative)


def same_python(candidate: str) -> bool:
    resolved = shutil.which(candidate) or candidate
    with contextlib.suppress(OSError):
        return Path(resolved).resolve() == Path(sys.executable).resolve()
    return candidate == sys.executable


def extend_vllm_binary_extension_path() -> None:
    try:
        import vllm
    except Exception:
        return
    package_path = getattr(vllm, "__path__", None)
    if package_path is None:
        return
    site_roots = list(site.getsitepackages())
    with contextlib.suppress(Exception):
        site_roots.append(site.getusersitepackages())
    for root in site_roots:
        candidate = Path(root) / "vllm"
        if candidate.is_dir() and str(candidate) not in package_path:
            package_path.append(str(candidate))


def normalize_voice_type(voice_type: str | None) -> str | None:
    if not voice_type:
        return voice_type
    mapping = {"tina": "f6009"}
    return mapping.get(voice_type.lower(), voice_type)


def ensure_audio(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")
        return audio_path

    audio_dir = output_dir / "inputs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(args.audio_url).suffix or ".wav"
    audio_path = audio_dir / f"default_input{suffix}"
    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    try:
        with urlopen(args.audio_url, timeout=120) as response:
            audio_path.write_bytes(response.read())
    except (OSError, URLError) as exc:
        fallback = audio_dir / "fallback_440hz.wav"
        write_fallback_wav(fallback)
        note = {
            "warning": "Failed to download default audio; generated a sine wave.",
            "audio_url": args.audio_url,
            "error": str(exc),
            "fallback": str(fallback),
        }
        save_json(audio_dir / "audio_download_warning.json", note)
        return fallback
    return audio_path


def write_fallback_wav(path: Path) -> None:
    import math
    import wave

    sample_rate = 16000
    seconds = 1.5
    samples = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for idx in range(samples):
            value = int(math.sin(2.0 * math.pi * 440.0 * idx / sample_rate) * 12000)
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    metrics: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    try:
        import numpy as np
        import soundfile as sf

        data, sample_rate = sf.read(str(path), always_2d=False)
        arr = np.asarray(data, dtype=np.float64)
        samples = int(arr.shape[0]) if arr.ndim else int(arr.size)
        metrics.update(
            {
                "sample_rate": int(sample_rate),
                "samples": samples,
                "duration_sec": samples / float(sample_rate) if sample_rate else None,
                "channels": int(arr.shape[1]) if arr.ndim > 1 else 1,
                "rms": float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0,
                "peak": float(np.max(np.abs(arr))) if arr.size else 0.0,
            }
        )
    except Exception as exc:
        wave_metrics = audio_metrics_with_wave(path)
        if wave_metrics:
            metrics.update(wave_metrics)
        else:
            metrics["error"] = str(exc)
    return metrics


def audio_metrics_with_wave(path: Path) -> dict[str, Any] | None:
    try:
        import array
        import math
        import wave

        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            samples = handle.getnframes()
            frames = handle.readframes(samples)
        values: list[int] | array.array[int]
        scale: float
        if sample_width == 1:
            values = [byte - 128 for byte in frames]
            scale = 128.0
        elif sample_width == 2:
            values = array.array("h")
            values.frombytes(frames)
            if sys.byteorder != "little":
                values.byteswap()
            scale = 32768.0
        elif sample_width == 4:
            values = array.array("i")
            values.frombytes(frames)
            if sys.byteorder != "little":
                values.byteswap()
            scale = 2147483648.0
        else:
            return None
        if not values:
            rms = 0.0
            peak = 0.0
        else:
            square_mean = sum(float(value) * float(value) for value in values) / len(
                values
            )
            rms = math.sqrt(square_mean) / scale
            peak = max(abs(value) for value in values) / scale
        return {
            "sample_rate": int(sample_rate),
            "samples": int(samples),
            "duration_sec": samples / float(sample_rate) if sample_rate else None,
            "channels": int(channels),
            "rms": float(rms),
            "peak": float(peak),
        }
    except Exception:
        return None


def data_url_for_audio(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def build_messages(args: argparse.Namespace, audio_ref: str, *, vllm: bool) -> list[dict[str, Any]]:
    if vllm:
        audio_item = {"type": "audio", "audio": audio_ref}
    else:
        if audio_ref.startswith("data:"):
            encoded = audio_ref.split(",", 1)[1]
            audio_item = {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            }
        else:
            audio_item = {"type": "audio_url", "audio_url": {"url": audio_ref}}
    content = [audio_item]
    if args.prompt:
        content.append({"type": "text", "text": args.prompt})
    return [
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": content,
        },
    ]


def run_vllm_backend(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not args.worker and args.vllm_docker_image:
        return run_vllm_docker_worker(args, audio_path, output_dir)
    if not args.worker and not same_python(args.vllm_python):
        return run_vllm_worker(args, audio_path, output_dir)
    return guarded_backend(
        "vllm",
        output_dir,
        lambda: run_vllm_local(args, audio_path, output_dir),
    )


def run_vllm_docker_worker(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    log_path = output_dir / "vllm_docker_worker.log"
    script_path = map_host_path_to_container(Path(__file__).resolve(), args.vllm_docker_mount)
    worker_output_dir = map_host_path_to_container(output_dir, args.vllm_docker_mount)
    worker_audio_path = map_host_path_to_container(audio_path, args.vllm_docker_mount)
    repo_workdir = map_host_path_to_container(REPO_ROOT, args.vllm_docker_mount)
    debug_hook_dir = map_host_path_to_container(
        REPO_ROOT / "scripts" / "vllm_qwen35_debug",
        args.vllm_docker_mount,
    )
    vllm_pythonpath = args.vllm_root
    if any(key.startswith("VLLM_QWEN35_") for key in os.environ):
        vllm_pythonpath = prepend_pythonpath(debug_hook_dir, vllm_pythonpath)
    cmd = ["docker", "run", "--rm", "--ipc=host", "--network=host"]
    if args.vllm_docker_gpus:
        cmd.extend(["--gpus", args.vllm_docker_gpus])
    for extra_arg in args.vllm_docker_extra_arg:
        cmd.extend(shlex.split(extra_arg))
    for key, value in sorted(os.environ.items()):
        if key.startswith("VLLM_QWEN35_"):
            cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(
        [
            "-v",
            args.vllm_docker_mount,
            "-w",
            repo_workdir,
            "-e",
            f"PYTHONPATH={vllm_pythonpath}",
            "-e",
            f"QWEN35_OMNI_MODEL_PATH={args.model_path}",
            "-e",
            "VLLM_FLASH_ATTN_USE_UPSTREAM=0",
            args.vllm_docker_image,
            args.vllm_docker_python,
            script_path,
            "--worker",
            "--backend",
            "vllm",
            "--model-path",
            args.model_path,
            "--model-name",
            args.model_name,
            "--code2wav-model-path",
            code2wav_path(args),
            "--output-dir",
            worker_output_dir,
            "--audio-path",
            worker_audio_path,
            "--prompt",
            args.prompt,
            "--system-prompt",
            args.system_prompt,
            "--voice-type",
            args.voice_type,
            "--seed",
            str(args.seed),
            "--max-tokens",
            str(args.max_tokens),
            "--sample-rate",
            str(args.sample_rate),
            "--request-timeout",
            str(args.request_timeout),
            "--vllm-root",
            args.vllm_root,
            "--vllm-thinker-devices",
            args.vllm_thinker_devices,
            "--vllm-talker-devices",
            args.vllm_talker_devices,
            "--vllm-code2wav-devices",
            args.vllm_code2wav_devices,
            "--vllm-max-num-seqs",
            str(args.vllm_max_num_seqs),
            "--vllm-block-size",
            str(args.vllm_block_size),
            "--vllm-max-model-len",
            str(args.vllm_max_model_len),
            "--vllm-talker-max-model-len",
            str(args.vllm_talker_max_model_len),
            "--vllm-max-num-batched-tokens",
            str(args.vllm_max_num_batched_tokens),
            "--vllm-gpu-memory-utilization",
            str(args.vllm_gpu_memory_utilization),
            "--vllm-talker-gpu-memory-utilization",
            str(args.vllm_talker_gpu_memory_utilization),
        ]
    )
    if args.disable_vllm_mtp:
        cmd.append("--disable-vllm-mtp")
    if args.vllm_enforce_eager:
        cmd.append("--vllm-enforce-eager")
    if args.no_vllm_prefix_caching:
        cmd.append("--no-vllm-prefix-caching")
    if args.no_vllm_chunked_prefill:
        cmd.append("--no-vllm-chunked-prefill")
    if args.no_code2wav_torch_compile:
        cmd.append("--no-code2wav-torch-compile")
    if args.vllm_default_talker_params:
        cmd.append("--vllm-default-talker-params")
    if args.vllm_thinker_only:
        cmd.append("--vllm-thinker-only")

    env = os.environ.copy()
    env["QWEN35_OMNI_MODEL_PATH"] = args.model_path
    if args.vllm_root:
        env["PYTHONPATH"] = prepend_pythonpath(args.vllm_root, env.get("PYTHONPATH"))
    return run_external_vllm_worker(cmd, log_path, output_dir, env=env)


def run_external_vllm_worker(
    cmd: list[str],
    log_path: Path,
    output_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result_path = output_dir / "vllm_result.json"
    if result_path.exists():
        result_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result_path.exists():
        result = load_json(result_path)
        result["worker_returncode"] = proc.returncode
        result["worker_log"] = str(log_path)
        result["worker_elapsed_sec"] = time.perf_counter() - started
        if proc.returncode != 0:
            result.setdefault("error", tail_text(log_path))
            result["ok"] = False
        save_backend_result(output_dir, "vllm", result)
        return result

    result = {
        "backend": "vllm",
        "ok": False,
        "error": tail_text(log_path) or f"vLLM worker exited {proc.returncode}",
        "worker_returncode": proc.returncode,
        "worker_log": str(log_path),
        "command": printable_command(cmd),
        "elapsed_sec": time.perf_counter() - started,
    }
    save_backend_result(output_dir, "vllm", result)
    return result


def prepend_pythonpath(path: str, current: str | None) -> str:
    return path if not current else path + os.pathsep + current


def guarded_backend(
    backend: str,
    output_dir: Path,
    fn: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = fn()
        result.setdefault("ok", True)
    except Exception as exc:
        result = {
            "backend": backend,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        }
    result.setdefault("backend", backend)
    result.setdefault("elapsed_sec", time.perf_counter() - started)
    result.setdefault("finished_at", utc_now())
    save_backend_result(output_dir, backend, result)
    return result


def save_backend_result(output_dir: Path, backend: str, result: dict[str, Any]) -> None:
    save_json(output_dir / f"{backend}_result.json", result)


def run_vllm_worker(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    log_path = output_dir / "vllm_worker.log"
    cmd = [
        args.vllm_python,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        "vllm",
        "--model-path",
        args.model_path,
        "--model-name",
        args.model_name,
        "--code2wav-model-path",
        code2wav_path(args),
        "--output-dir",
        str(output_dir),
        "--audio-path",
        str(audio_path),
        "--prompt",
        args.prompt,
        "--system-prompt",
        args.system_prompt,
        "--voice-type",
        args.voice_type,
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--sample-rate",
        str(args.sample_rate),
        "--request-timeout",
        str(args.request_timeout),
        "--vllm-root",
        args.vllm_root,
        "--vllm-thinker-devices",
        args.vllm_thinker_devices,
        "--vllm-talker-devices",
        args.vllm_talker_devices,
        "--vllm-code2wav-devices",
        args.vllm_code2wav_devices,
        "--vllm-max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--vllm-block-size",
        str(args.vllm_block_size),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--vllm-talker-max-model-len",
        str(args.vllm_talker_max_model_len),
        "--vllm-max-num-batched-tokens",
        str(args.vllm_max_num_batched_tokens),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-talker-gpu-memory-utilization",
        str(args.vllm_talker_gpu_memory_utilization),
    ]
    if args.disable_vllm_mtp:
        cmd.append("--disable-vllm-mtp")
    if args.vllm_enforce_eager:
        cmd.append("--vllm-enforce-eager")
    if args.no_vllm_prefix_caching:
        cmd.append("--no-vllm-prefix-caching")
    if args.no_vllm_chunked_prefill:
        cmd.append("--no-vllm-chunked-prefill")
    if args.no_code2wav_torch_compile:
        cmd.append("--no-code2wav-torch-compile")
    if args.vllm_default_talker_params:
        cmd.append("--vllm-default-talker-params")
    if args.vllm_thinker_only:
        cmd.append("--vllm-thinker-only")

    env = os.environ.copy()
    env["QWEN35_OMNI_MODEL_PATH"] = args.model_path
    if args.vllm_root:
        env["PYTHONPATH"] = prepend_pythonpath(args.vllm_root, env.get("PYTHONPATH"))
    return run_external_vllm_worker(cmd, log_path, output_dir, env=env)


def run_vllm_local(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if args.vllm_root and args.vllm_root not in sys.path:
        sys.path.insert(0, args.vllm_root)
    if args.vllm_root:
        extend_vllm_binary_extension_path()

    install_vllm_qwen35_debug_hooks()

    import numpy as np
    import psutil
    import soundfile as sf
    from qwen_omni_utils import process_mm_info
    from transformers import AutoProcessor, AutoTokenizer

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.omni3_5_llm_engine import OmniLLMEngine
    from vllm.inputs import TokensPrompt
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams

    model_path = resolve_stage_model(args.model_path, "thinker")
    talker_model_path = resolve_stage_model(args.model_path, "talker")
    c2w_model_path = resolve_stage_model(code2wav_path(args), "code2wav")
    messages = build_messages(args, str(audio_path), vllm=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        from transformers import Qwen3OmniNextProcessor

        processor = Qwen3OmniNextProcessor.from_pretrained(model_path)
    except (ImportError, AttributeError):
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    audios, images, videos, video_kwargs = process_mm_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
        use_audio_in_video=False,
    )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(prompt, list):
        prompt = prompt[0]

    multi_modal_data: dict[str, Any] = {}
    if audios:
        multi_modal_data["audio"] = audios
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    prompt_token_ids = tokenizer.encode(prompt)
    input_prompt = TokensPrompt(
        prompt_token_ids=prompt_token_ids,
        multi_modal_data=multi_modal_data,
        mm_processor_kwargs=video_kwargs,
    )
    thinker_sampling = SamplingParams(
        temperature=1e-6,
        top_k=1,
        top_p=0.8,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        max_tokens=args.max_tokens,
        detokenize=True,
        logprobs=20,
        seed=args.seed,
    )
    talker_sampling = SamplingParams(
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        max_tokens=2048,
        detokenize=False,
        seed=args.seed,
    )

    enable_chunked_prefill = not args.no_vllm_chunked_prefill
    thinker_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enable_chunked_prefill": enable_chunked_prefill,
        "tensor_parallel_size": len(parse_devices(args.vllm_thinker_devices)),
        "enforce_eager": args.vllm_enforce_eager,
        "distributed_executor_backend": "mp",
        "limit_mm_per_prompt": {"audio": 960, "image": 960, "video": 960},
        "max_model_len": args.vllm_max_model_len,
        "max_num_batched_tokens": (
            args.vllm_max_num_batched_tokens if enable_chunked_prefill else None
        ),
        "max_num_seqs": args.vllm_max_num_seqs,
        "block_size": args.vllm_block_size,
        "enable_prefix_caching": not args.no_vllm_prefix_caching,
        "enable_prompt_embeds": False,
        "disable_log_stats": False,
        "mm_processor_cache_type": "lru",
    }
    if not args.disable_vllm_mtp:
        thinker_kwargs["speculative_config"] = {
            "method": "qwen3_omni_next_thinker_mtp",
            "num_speculative_tokens": 4,
        }
    thinker_kwargs["compilation_config"] = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "use_inductor": False,
        "pass_config": {
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_attn_quant": False,
        },
    }
    thinker_engine_args = AsyncEngineArgs(**thinker_kwargs)
    if args.vllm_thinker_only:
        engine = OmniLLMEngine(
            thinker_engine_args,
            thinker_visible_devices=parse_devices(args.vllm_thinker_devices),
        )
    else:
        talker_engine_args = AsyncEngineArgs(
            model=talker_model_path,
            trust_remote_code=True,
            gpu_memory_utilization=args.vllm_talker_gpu_memory_utilization,
            enable_chunked_prefill=enable_chunked_prefill,
            tensor_parallel_size=1,
            enforce_eager=args.vllm_enforce_eager,
            distributed_executor_backend="mp",
            limit_mm_per_prompt={"audio": 32, "image": 96, "video": 32},
            max_model_len=args.vllm_talker_max_model_len,
            max_num_batched_tokens=(
                args.vllm_max_num_batched_tokens if enable_chunked_prefill else None
            ),
            max_num_seqs=args.vllm_max_num_seqs,
            block_size=args.vllm_block_size,
            enable_prefix_caching=not args.no_vllm_prefix_caching,
            enable_prompt_embeds=True,
        )

        engine = OmniLLMEngine(
            thinker_engine_args,
            talker_engine_args,
            c2w_model_path,
            code2wav_enable_torch_compile=not args.no_code2wav_torch_compile,
            thinker_visible_devices=parse_devices(args.vllm_thinker_devices),
            talker_visible_devices=parse_devices(args.vllm_talker_devices),
            code2wav_visible_devices=parse_devices(args.vllm_code2wav_devices),
        )
    try:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        output_queue = engine.add_request(
            request_id,
            copy.deepcopy(input_prompt),
            thinker_sampling,
            talker_params=(
                None
                if args.vllm_thinker_only or args.vllm_default_talker_params
                else talker_sampling
            ),
            voice_type=normalize_voice_type(args.voice_type),
        )
        last_output = None
        waveforms = []
        chunk_records = []
        while True:
            item = output_queue.get(timeout=args.request_timeout)
            if item is None:
                break
            if isinstance(item, RequestOutput):
                last_output = item
            elif isinstance(item, tuple) and isinstance(item[0], np.ndarray):
                chunk = np.asarray(item[0], dtype=np.float32)
                waveforms.append(chunk)
                chunk_records.append(
                    {
                        "index": len(chunk_records),
                        "samples": int(chunk.shape[0]) if chunk.ndim else int(chunk.size),
                        "min": float(np.min(chunk)) if chunk.size else 0.0,
                        "max": float(np.max(chunk)) if chunk.size else 0.0,
                        "rms": float(np.sqrt(np.mean(np.square(chunk))))
                        if chunk.size
                        else 0.0,
                        "output_tokens": item[1] if len(item) > 1 else None,
                    }
                )
            else:
                raise ValueError(f"Unexpected vLLM output item: {type(item).__name__}")

        response_path = output_dir / "vllm_response.json"
        chunk_path = output_dir / "vllm_audio_chunks.json"
        audio_output_path = output_dir / "vllm_output.wav"
        text = ""
        token_ids = []
        logprob_summary = []
        if last_output is not None and last_output.outputs:
            output = last_output.outputs[0]
            text = output.text or ""
            token_ids = list(output.token_ids or [])
            logprob_summary = summarize_vllm_logprobs(getattr(output, "logprobs", None))
        response_payload = {
            "request_id": request_id,
            "text": text,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "prompt_token_count": len(prompt_token_ids),
            "audio_chunks": len(chunk_records),
            "logprobs": logprob_summary,
        }
        save_json(response_path, response_payload)
        save_json(output_dir / "vllm_logprobs.json", logprob_summary)
        save_json(chunk_path, chunk_records)
        if waveforms:
            sf.write(
                str(audio_output_path),
                np.concatenate(waveforms),
                samplerate=args.sample_rate,
            )
        elapsed = time.perf_counter() - started
        return {
            "backend": "vllm",
            "ok": True,
            "text": text,
            "response_json": str(response_path),
            "audio_path": str(audio_output_path) if audio_output_path.exists() else None,
            "audio_metrics": audio_metrics(audio_output_path),
            "elapsed_sec": elapsed,
            "model_path": model_path,
            "talker_model_path": talker_model_path,
            "code2wav_model_path": c2w_model_path,
        }
    finally:
        with contextlib.suppress(Exception):
            engine.shutdown()
        current = psutil.Process()
        for child in current.children(recursive=True):
            with contextlib.suppress(Exception):
                os.kill(child.pid, signal.SIGTERM)


def resolve_stage_model(root: str, stage: str) -> str:
    path = Path(root)
    nested = path / stage
    if path.is_dir() and nested.is_dir():
        return str(nested)
    return str(path)


def run_sglang_backend(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return guarded_backend(
        "sglang",
        output_dir,
        lambda: run_sglang_local(args, audio_path, output_dir),
    )


def run_sglang_local(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    server_proc: subprocess.Popen[str] | None = None
    log_path = output_dir / "sglang_server.log"
    base_url = args.sglang_base_url or (
        f"http://{args.sglang_host}:{args.sglang_port}"
    )
    try:
        if args.launch_sglang:
            server_proc = launch_sglang_server(args, log_path)
            wait_for_server(base_url, server_proc, log_path, args.sglang_server_timeout)
        response = call_sglang_server(args, audio_path, base_url, output_dir)
        return response
    finally:
        if server_proc is not None and not args.keep_sglang_server:
            cleanup_sglang_container_server(args)
            terminate_process(server_proc)


def launch_sglang_server(
    args: argparse.Namespace,
    log_path: Path,
) -> subprocess.Popen[str]:
    server_args = [
        args.sglang_python,
        "examples/run_qwen3_5_omni_speech_server.py",
        "--model-path",
        args.model_path,
        "--model-name",
        args.model_name,
        "--host",
        args.sglang_host,
        "--port",
        str(args.sglang_port),
        "--voice-type",
        args.voice_type,
        "--max-tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--gpu-thinker",
        str(args.sglang_gpu_thinker),
        "--gpu-talker",
        str(args.sglang_gpu_talker),
        "--gpu-code2wav",
        str(args.sglang_gpu_code2wav),
        "--thinker-max-seq-len",
        str(args.sglang_thinker_max_seq_len),
    ]
    c2w_path = code2wav_path(args)
    if args.sglang_container or Path(c2w_path).exists():
        server_args.extend(["--code2wav-model-path", c2w_path])
    if args.no_code2wav_torch_compile:
        server_args.append("--no-code2wav-torch-compile")
    server_args.extend(args.sglang_extra_arg)

    if args.sglang_container:
        env_exports = " ".join(
            f"{shlex.quote(key)}={shlex.quote(value)}"
            for key, value in parse_env_assignments(args.sglang_env).items()
        )
        if env_exports:
            env_exports += " "
        inner_cmd = (
            f"cd {shlex.quote(args.sglang_workdir)} && "
            f"export PYTHONPATH={shlex.quote(args.sglang_workdir)}:"
            "${PYTHONPATH:-} TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1} "
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="
            "${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1} "
            f"{env_exports}&& "
            f"exec {printable_command(server_args)}"
        )
        cmd = ["docker", "exec", args.sglang_container, "/bin/sh", "-lc", inner_cmd]
        cwd = str(REPO_ROOT)
        env = None
    else:
        server_args[1] = str(
            REPO_ROOT / "examples" / "run_qwen3_5_omni_speech_server.py"
        )
        cmd = server_args
        cwd = str(REPO_ROOT)
        env = os.environ.copy()
        env["PYTHONPATH"] = prepend_pythonpath(str(REPO_ROOT), env.get("PYTHONPATH"))
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        env.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
        env.update(parse_env_assignments(args.sglang_env))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(proc, "_alignment_log_file", log_file)
    setattr(proc, "_alignment_command", printable_command(cmd))
    return proc


def parse_env_assignments(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--sglang-env must be KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--sglang-env has an empty key: {raw!r}")
        env[key] = value
    return env


def wait_for_server(
    base_url: str,
    proc: subprocess.Popen[str],
    log_path: Path,
    timeout: float,
) -> None:
    deadline = time.time() + timeout
    health_url = base_url.rstrip("/") + "/health"
    models_url = base_url.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"SGLang server exited with {proc.returncode}.\n{tail_text(log_path)}"
            )
        for url in (health_url, models_url):
            try:
                with urlopen(url, timeout=5) as response:
                    if 200 <= response.status < 500:
                        return
            except Exception:
                pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready at {base_url}")


def call_sglang_server(
    args: argparse.Namespace,
    audio_path: Path,
    base_url: str,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    audio_url = data_url_for_audio(audio_path)
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
    ]
    if args.sglang_media_mode in {"content", "both"}:
        messages = build_messages(args, audio_url, vllm=False)

    body: dict[str, Any] = {
        "model": args.model_name,
        "messages": messages,
        "modalities": ["text", "audio"],
        "audio": {"voice": args.voice_type, "format": "wav"},
        "max_tokens": args.max_tokens,
        "temperature": 1e-6,
        "top_k": 1,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "seed": args.seed,
        "stream": False,
    }
    if args.sglang_media_mode in {"top-level", "both"}:
        body["input_audio"] = {
            "data": audio_url.split(",", 1)[1],
            "format": "wav",
        }

    request_path = output_dir / "sglang_request.json"
    response_path = output_dir / "sglang_response.json"
    save_json(request_path, redact_audio_payload(body))
    response = post_json(
        base_url.rstrip("/") + "/v1/chat/completions",
        body,
        timeout=args.request_timeout,
    )
    save_json(response_path, response)

    message = response.get("choices", [{}])[0].get("message", {})
    text = message.get("content") or ""
    audio_payload = message.get("audio") or response.get("audio") or {}
    audio_output_path = output_dir / "sglang_output.wav"
    if audio_payload.get("data"):
        audio_output_path.write_bytes(decode_audio_data(str(audio_payload["data"])))

    return {
        "backend": "sglang",
        "ok": True,
        "text": text,
        "response_json": str(response_path),
        "request_json": str(request_path),
        "audio_path": str(audio_output_path) if audio_output_path.exists() else None,
        "audio_metrics": audio_metrics(audio_output_path),
        "elapsed_sec": time.perf_counter() - started,
        "base_url": base_url,
    }


def redact_audio_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(payload)
    marker = "<base64 audio omitted>"

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "data" in value and isinstance(value["data"], str):
                value["data"] = marker
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(redacted)
    return redacted


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def decode_audio_data(data: str) -> bytes:
    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
    log_file = getattr(proc, "_alignment_log_file", None)
    if log_file is not None:
        with contextlib.suppress(Exception):
            log_file.close()


def cleanup_sglang_container_server(args: argparse.Namespace) -> None:
    if not args.sglang_container:
        return
    server_pattern = (
        "[r]un_qwen3_5_omni_speech_server.py.*"
        f"--port {int(args.sglang_port)}"
    )
    worker_pattern = "[m]ultiprocessing.spawn import spawn_main"
    tracker_pattern = "[m]ultiprocessing.resource_tracker"
    command = (
        f"pkill -TERM -f {shlex.quote(server_pattern)} || true; "
        f"pkill -TERM -f {shlex.quote(worker_pattern)} || true; "
        f"pkill -TERM -f {shlex.quote(tracker_pattern)} || true; "
        "sleep 5; "
        f"pkill -KILL -f {shlex.quote(server_pattern)} || true; "
        f"pkill -KILL -f {shlex.quote(worker_pattern)} || true; "
        f"pkill -KILL -f {shlex.quote(tracker_pattern)} || true"
    )
    subprocess.run(
        ["docker", "exec", args.sglang_container, "/bin/sh", "-lc", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def printable_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_asr(args: argparse.Namespace, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    asr: dict[str, Any] = {"backend": args.asr_backend, "items": {}}
    if args.asr_backend == "none":
        asr["status"] = "skipped"
        return asr

    for backend, result in results.items():
        audio_path = result.get("audio_path")
        if not audio_path:
            asr["items"][backend] = {"ok": False, "error": "No audio output"}
            continue
        transcript = transcribe_audio(Path(audio_path), args)
        asr["items"][backend] = transcript

    if "vllm" in asr["items"] and "sglang" in asr["items"]:
        left = asr["items"]["vllm"].get("text") or ""
        right = asr["items"]["sglang"].get("text") or ""
        if left and right:
            asr["comparison"] = compare_text(left, right)
            asr["status"] = "ok"
        else:
            asr["status"] = "incomplete"
    else:
        asr["status"] = "single-backend"
    return asr


def transcribe_audio(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.asr_backend in {"auto", "command"} and args.asr_command:
        return transcribe_with_command(path, args.asr_command)
    if args.asr_backend == "command":
        return {"ok": False, "error": "--asr-command is required for command ASR"}
    if args.asr_backend in {"auto", "whisper"} and args.asr_container:
        return transcribe_with_whisper_container(path, args)
    if args.asr_backend in {"auto", "whisper"}:
        return transcribe_with_whisper(path, args)
    return {"ok": False, "error": f"Unsupported ASR backend: {args.asr_backend}"}


def transcribe_with_command(path: Path, command_template: str) -> dict[str, Any]:
    command = command_template.format(audio=shlex.quote(str(path)))
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "backend": "command",
            "command": command,
            "returncode": proc.returncode,
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "elapsed_sec": time.perf_counter() - started,
        }
    text = proc.stdout.strip()
    if not text:
        return {
            "ok": False,
            "backend": "command",
            "command": command,
            "error": "Empty ASR transcript",
            "elapsed_sec": time.perf_counter() - started,
        }
    return {
        "ok": True,
        "backend": "command",
        "command": command,
        "text": text,
        "elapsed_sec": time.perf_counter() - started,
    }


def transcribe_with_whisper_container(
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    audio_path = map_host_path_to_container(path, args.asr_container_mount)
    whisper_kwargs = {"task": "transcribe"}
    if args.asr_language:
        whisper_kwargs["language"] = args.asr_language
    code = (
        "import json, whisper; "
        f"model = whisper.load_model({json.dumps(args.asr_model)}); "
        "result = model.transcribe("
        f"{json.dumps(audio_path)}, **{json.dumps(whisper_kwargs)}); "
        "print(json.dumps({"
        "'text': (result.get('text') or '').strip(), "
        "'language': result.get('language')"
        "}, ensure_ascii=False))"
    )
    proc = subprocess.run(
        ["docker", "exec", args.asr_container, "python", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "backend": "whisper-container",
            "model": args.asr_model,
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "elapsed_sec": time.perf_counter() - started,
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "ok": False,
            "backend": "whisper-container",
            "model": args.asr_model,
            "error": f"Failed to parse ASR JSON: {exc}; stdout={proc.stdout!r}",
            "elapsed_sec": time.perf_counter() - started,
        }
    text = str(payload.get("text") or "").strip()
    if not text:
        return {
            "ok": False,
            "backend": "whisper-container",
            "model": args.asr_model,
            "text": "",
            "language": payload.get("language"),
            "error": "Empty ASR transcript",
            "elapsed_sec": time.perf_counter() - started,
        }
    return {
        "ok": True,
        "backend": "whisper-container",
        "model": args.asr_model,
        "text": text,
        "language": payload.get("language"),
        "elapsed_sec": time.perf_counter() - started,
    }


def transcribe_with_whisper(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import whisper

        model = whisper.load_model(args.asr_model)
        whisper_kwargs = {"task": "transcribe"}
        if args.asr_language:
            whisper_kwargs["language"] = args.asr_language
        result = model.transcribe(str(path), **whisper_kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "backend": "whisper",
            "model": args.asr_model,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        }
    text = str(result.get("text") or "").strip()
    if not text:
        return {
            "ok": False,
            "backend": "whisper",
            "model": args.asr_model,
            "text": "",
            "language": result.get("language"),
            "error": "Empty ASR transcript",
            "elapsed_sec": time.perf_counter() - started,
        }
    return {
        "ok": True,
        "backend": "whisper",
        "model": args.asr_model,
        "text": text,
        "language": result.get("language"),
        "elapsed_sec": time.perf_counter() - started,
    }


def compare_text(left: str, right: str) -> dict[str, Any]:
    left_norm = normalize_for_distance(left)
    right_norm = normalize_for_distance(right)
    cer_distance = levenshtein(left_norm, right_norm)
    cer_den = max(len(left_norm), len(right_norm), 1)
    left_words = left_norm.split()
    right_words = right_norm.split()
    if len(left_words) <= 1 and len(right_words) <= 1:
        wer_value = None
    else:
        wer_value = word_error_rate(left_words, right_words)
    return {
        "cer": cer_distance / cer_den,
        "cer_distance": cer_distance,
        "wer": wer_value,
        "left_normalized": left_norm,
        "right_normalized": right_norm,
    }


def normalize_for_distance(text: str) -> str:
    lowered = text.lower()
    kept = []
    for char in lowered:
        if char.isalnum() or char.isspace() or "\u4e00" <= char <= "\u9fff":
            kept.append(char)
    normalized = "".join(kept)
    return " ".join(normalized.split())


def word_error_rate(left_words: list[str], right_words: list[str]) -> float:
    distance = levenshtein_sequence(left_words, right_words)
    return distance / max(len(left_words), 1)


def levenshtein(left: str, right: str) -> int:
    return levenshtein_sequence(list(left), list(right))


def levenshtein_sequence(left: list[Any], right: list[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_item != right_item)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def decide_alignment(
    args: argparse.Namespace,
    results: dict[str, dict[str, Any]],
    asr: dict[str, Any],
) -> dict[str, Any]:
    if not {"vllm", "sglang"}.issubset(results):
        return {"status": "single-backend", "reason": "Only one backend was run."}
    failed = [name for name, result in results.items() if not result.get("ok")]
    if failed:
        return {"status": "failed", "reason": f"Backend failed: {', '.join(failed)}"}
    if args.asr_backend != "none":
        asr_failed = [
            name for name, item in (asr.get("items") or {}).items() if not item.get("ok")
        ]
        if asr_failed:
            return {
                "status": "failed",
                "reason": f"ASR failed or returned empty transcript: {', '.join(asr_failed)}",
            }
    comparison = asr.get("comparison")
    if comparison:
        cer = comparison.get("cer")
        wer = comparison.get("wer")
        cer_ok = cer is not None and cer <= args.cer_threshold
        wer_ok = wer is not None and wer <= args.wer_threshold
        if cer_ok or wer_ok:
            return {
                "status": "passed",
                "reason": "ASR distance is within threshold.",
                "asr_comparison": comparison,
            }
        return {
            "status": "failed",
            "reason": "ASR distance exceeded thresholds.",
            "cer_threshold": args.cer_threshold,
            "wer_threshold": args.wer_threshold,
            "asr_comparison": comparison,
        }

    fallback = compare_backend_metrics(results)
    text_cer = fallback.get("text_cer")
    duration_ratio = fallback.get("duration_ratio")
    text_ok = (
        text_cer is not None
        and text_cer <= args.fallback_text_cer_threshold
    )
    duration_ok = (
        duration_ratio is not None
        and duration_ratio <= args.duration_ratio_threshold
    )
    fallback.update(
        {
            "text_cer_threshold": args.fallback_text_cer_threshold,
            "duration_ratio_threshold": args.duration_ratio_threshold,
        }
    )
    if text_ok and duration_ok:
        return {
            "status": "passed",
            "reason": (
                "ASR unavailable; direct text and audio-duration metrics are "
                "within fallback thresholds."
            ),
            "fallback": fallback,
        }
    return {
        "status": "pending",
        "reason": "ASR comparison is unavailable and fallback metrics are not sufficient.",
        "fallback": fallback,
    }


def compare_backend_metrics(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = results.get("vllm") or {}
    right = results.get("sglang") or {}
    text_cmp = compare_text(str(left.get("text") or ""), str(right.get("text") or ""))
    left_metrics = left.get("audio_metrics") or {}
    right_metrics = right.get("audio_metrics") or {}
    left_duration = left_metrics.get("duration_sec")
    right_duration = right_metrics.get("duration_sec")
    duration_delta = None
    duration_ratio = None
    if left_duration is not None and right_duration is not None:
        duration_delta = abs(float(left_duration) - float(right_duration))
        denom = max(float(left_duration), float(right_duration), 1e-9)
        duration_ratio = duration_delta / denom
    return {
        "text_cer": text_cmp.get("cer"),
        "text_cer_distance": text_cmp.get("cer_distance"),
        "duration_delta_sec": duration_delta,
        "duration_ratio": duration_ratio,
        "vllm_duration_sec": left_duration,
        "sglang_duration_sec": right_duration,
    }


def write_report(
    args: argparse.Namespace,
    output_dir: Path,
    audio_path: Path,
    results: dict[str, dict[str, Any]],
    asr: dict[str, Any],
    decision: dict[str, Any],
) -> Path:
    report_path = output_dir / "alignment_report.md"
    lines = [
        "# Qwen3.5-Omni Alignment Report",
        "",
        f"- Generated: {utc_now()}",
        f"- Model: `{args.model_path}`",
        f"- Code2wav: `{code2wav_path(args)}`",
        f"- Input audio: `{audio_path}`",
        f"- Input SHA256: `{file_sha256(audio_path)}`",
        f"- Prompt: {args.prompt}",
        f"- Voice: `{args.voice_type}`",
        f"- Decision: **{decision.get('status', 'unknown')}**",
        f"- Reason: {decision.get('reason', '')}",
        "",
        "## Backend Outputs",
        "",
        "| Backend | Status | Text | Audio | Duration | RMS |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for backend in ("vllm", "sglang"):
        result = results.get(backend)
        if not result:
            continue
        metrics = result.get("audio_metrics") or {}
        duration = metrics.get("duration_sec")
        rms = metrics.get("rms")
        lines.append(
            "| {backend} | {status} | {text} | `{audio}` | {duration} | {rms} |".format(
                backend=backend,
                status="ok" if result.get("ok") else "failed",
                text=markdown_cell(result.get("text") or result.get("error") or ""),
                audio=result.get("audio_path") or "",
                duration=format_float(duration),
                rms=format_float(rms),
            )
        )
    lines.extend(render_decision_metrics(decision))
    lines.extend(["", "## ASR", ""])
    lines.extend(render_asr_section(asr))
    lines.extend(["", "## Artifacts", ""])
    for artifact in sorted(output_dir.glob("*")):
        if artifact.is_file():
            lines.append(f"- `{artifact}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def render_decision_metrics(decision: dict[str, Any]) -> list[str]:
    fallback = decision.get("fallback")
    if not fallback:
        return []
    return [
        "",
        "## Decision Metrics",
        "",
        f"- Direct text CER: `{format_float(fallback.get('text_cer'))}` "
        f"(threshold `{format_float(fallback.get('text_cer_threshold'))}`)",
        f"- Audio duration delta: `{format_float(fallback.get('duration_delta_sec'))}` sec",
        f"- Audio duration ratio: `{format_float(fallback.get('duration_ratio'))}` "
        f"(threshold `{format_float(fallback.get('duration_ratio_threshold'))}`)",
    ]


def render_asr_section(asr: dict[str, Any]) -> list[str]:
    lines = [
        f"- Backend: `{asr.get('backend')}`",
        f"- Status: `{asr.get('status')}`",
    ]
    comparison = asr.get("comparison")
    if comparison:
        lines.append(f"- CER: `{format_float(comparison.get('cer'))}`")
        lines.append(f"- WER: `{format_float(comparison.get('wer'))}`")
    items = asr.get("items") or {}
    if items:
        lines.extend(["", "| Backend | ASR status | Transcript |", "| --- | --- | --- |"])
        for backend, item in items.items():
            lines.append(
                "| {backend} | {status} | {text} |".format(
                    backend=backend,
                    status="ok" if item.get("ok") else "failed",
                    text=markdown_cell(item.get("text") or item.get("error") or ""),
                )
            )
    return lines


def markdown_cell(value: str, limit: int = 180) -> str:
    compact = " ".join(str(value).split())
    if len(compact) > limit:
        compact = compact[: limit - 3] + "..."
    return compact.replace("|", "\\|")


def format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = ensure_audio(args, output_dir)

    results: dict[str, dict[str, Any]] = {}
    if args.backend in {"compare", "vllm"}:
        results["vllm"] = run_vllm_backend(args, audio_path, output_dir)
    if args.backend in {"compare", "sglang"}:
        results["sglang"] = run_sglang_backend(args, audio_path, output_dir)

    asr = {"backend": args.asr_backend, "status": "skipped", "items": {}}
    if not args.worker:
        asr = run_asr(args, results)
        decision = decide_alignment(args, results, asr)
        save_json(
            output_dir / "alignment_summary.json",
            {
                "args": vars(args),
                "input_audio": str(audio_path),
                "input_audio_metrics": audio_metrics(audio_path),
                "results": results,
                "asr": asr,
                "decision": decision,
            },
        )
        report_path = write_report(args, output_dir, audio_path, results, asr, decision)
        print(f"Report: {report_path}")
        return 0 if decision.get("status") in {"passed", "single-backend", "pending"} else 1

    return 0 if all(result.get("ok") for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
