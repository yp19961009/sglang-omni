# SPDX-License-Identifier: Apache-2.0
"""Build reproducible Qwen3.5-Omni talker alignment contracts.

The SGLang and vLLM deployments used during Qwen3.5-Omni development do not
always load the same checkpoint. This tool records enough model and repository
identity to prevent a different-model comparison from being reported as an
exact numerical alignment.

Examples:

    python benchmarks/eval/qwen35_omni_talker_contract.py fingerprint \
        --engine sglang --model-path /myapp/models/qwen35 \
        --repo-path /myapp/sglang-omni --output sglang_fingerprint.json

    python benchmarks/eval/qwen35_omni_talker_contract.py compare \
        --candidate sglang_fingerprint.json --reference vllm_fingerprint.json \
        --output comparison_contract.json

    python benchmarks/eval/qwen35_omni_talker_contract.py replay \
        --fingerprint sglang_fingerprint.json --thinker-tokens tokens.json \
        --output talker_replay.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "qwen35-omni-talker-contract-v1"
_HASH_CHUNK_BYTES = 1024 * 1024
_IDENTITY_NAMES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "voice_map.json",
}
_WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_sha256_file(path: Path) -> str:
    """Hash file size plus the first and last MiB.

    Full checkpoint hashing is available through ``--full-weight-hash``. The
    sampled default keeps fingerprinting a multi-shard model quick while still
    distinguishing the development checkpoints used by the two engines.
    """

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(_HASH_CHUNK_BYTES))
        if size > _HASH_CHUNK_BYTES:
            handle.seek(max(0, size - _HASH_CHUNK_BYTES))
            digest.update(handle.read(_HASH_CHUNK_BYTES))
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _relative_records(
    model_path: Path,
    paths: Iterable[Path],
    *,
    full_hash: bool,
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths)):
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(model_path).as_posix(),
                "size": stat.st_size,
                "sha256": (
                    _sha256_file(path) if full_hash else _sampled_sha256_file(path)
                ),
            }
        )
    return records


def _identity_files(model_path: Path) -> list[Path]:
    files = []
    for path in model_path.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _IDENTITY_NAMES or path.name.endswith(".index.json"):
            files.append(path)
        elif path.suffix in {".yaml", ".yml"} and "codec" in path.as_posix().lower():
            files.append(path)
    return files


def _weight_files(model_path: Path) -> list[Path]:
    return [
        path
        for path in model_path.rglob("*")
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
    ]


def _git_output(repo_path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.decode("utf-8", errors="replace")


def build_repo_fingerprint(repo_path: Path | None) -> dict[str, Any] | None:
    if repo_path is None:
        return None
    commit = _git_output(repo_path, "rev-parse", "HEAD")
    status = _git_output(
        repo_path, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    diff = _git_output(repo_path, "diff", "--binary", "HEAD")
    return {
        "path": str(repo_path.resolve()),
        "commit": commit.strip() if commit else None,
        "dirty": bool(status and status.strip()),
        "status": status.splitlines() if status else [],
        "tracked_diff_sha256": _sha256_bytes(diff.encode("utf-8")) if diff else None,
    }


def build_model_fingerprint(
    model_path: Path,
    *,
    engine: str,
    repo_path: Path | None = None,
    full_weight_hash: bool = False,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise ValueError(f"model path is not a directory: {model_path}")

    identity_records = _relative_records(
        model_path,
        _identity_files(model_path),
        full_hash=True,
    )
    weight_records = _relative_records(
        model_path,
        _weight_files(model_path),
        full_hash=full_weight_hash,
    )
    if not identity_records:
        raise ValueError(f"no model identity files found under {model_path}")
    if not weight_records:
        raise ValueError(f"no model weights found under {model_path}")

    model_identity = {
        "identity_files": identity_records,
        "weight_files": weight_records,
        "weight_hash_mode": "full" if full_weight_hash else "sampled-first-last-1mib",
    }
    identity_set_id = _canonical_digest(identity_records)
    weight_set_id = _canonical_digest(weight_records)
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "model": {
            "path": str(model_path),
            "model_id": _canonical_digest(model_identity),
            "identity_set_id": identity_set_id,
            "weight_set_id": weight_set_id,
            **model_identity,
        },
        "repository": build_repo_fingerprint(
            repo_path.resolve() if repo_path else None
        ),
    }


def build_comparison_contract(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = candidate["model"]["model_id"]
    reference_id = reference["model"]["model_id"]
    same_model = candidate_id == reference_id
    candidate_weight_id = candidate["model"].get("weight_set_id")
    reference_weight_id = reference["model"].get("weight_set_id")
    same_weight_set = bool(candidate_weight_id) and (
        candidate_weight_id == reference_weight_id
    )
    if same_model:
        mode = "same_model_regression"
        allowed = [
            "thinker token ids under identical deterministic sampling",
            "talker codec ids from a fixed thinker-token replay",
            "decoded PCM hash when codec decoder and settings also match",
            "quality, latency, throughput, and memory metrics",
        ]
        forbidden: list[str] = []
    elif same_weight_set:
        mode = "same_weights_different_runtime_config"
        allowed = [
            "request and response protocol shape",
            "stage ordering and completion behavior",
            "thinker token equality only after tokenizer, processor, prompt, and sampling normalization",
            "talker codec equality only from fixed tokens after voice and talker config normalization",
            "quality, latency, throughput, and memory metrics with configuration labels",
        ]
        forbidden = [
            "unqualified exact thinker token equality",
            "unqualified exact talker codec or PCM equality",
            "attributing a delta solely to the serving engine before normalizing runtime config",
        ]
    else:
        mode = "different_model_reference"
        allowed = [
            "request and response protocol shape",
            "stage ordering and completion behavior",
            "audio presence, sample rate, duration, finiteness, and clipping checks",
            "quality metrics evaluated independently for each model",
            "latency, throughput, and memory metrics with configuration labels",
        ]
        forbidden = [
            "exact thinker token equality",
            "exact talker codec equality",
            "exact PCM equality",
            "attributing an answer or quality delta solely to the serving engine",
        ]
    candidate_identity = {
        record["path"]: record for record in candidate["model"]["identity_files"]
    }
    reference_identity = {
        record["path"]: record for record in reference["model"]["identity_files"]
    }
    identity_differences = []
    for path in sorted(set(candidate_identity) | set(reference_identity)):
        candidate_record = candidate_identity.get(path)
        reference_record = reference_identity.get(path)
        if candidate_record == reference_record:
            continue
        identity_differences.append(
            {
                "path": path,
                "candidate_sha256": (
                    candidate_record.get("sha256") if candidate_record else None
                ),
                "reference_sha256": (
                    reference_record.get("sha256") if reference_record else None
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": mode,
        "same_model": same_model,
        "same_weight_set": same_weight_set,
        "identity_differences": identity_differences,
        "candidate": {
            "engine": candidate.get("engine"),
            "model_id": candidate_id,
            "weight_set_id": candidate_weight_id,
            "repo_commit": (candidate.get("repository") or {}).get("commit"),
        },
        "reference": {
            "engine": reference.get("engine"),
            "model_id": reference_id,
            "weight_set_id": reference_weight_id,
            "repo_commit": (reference.get("repository") or {}).get("commit"),
        },
        "allowed_assertions": allowed,
        "forbidden_assertions": forbidden,
    }


def _load_token_ids(path: Path) -> list[int]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        for key in ("thinker_token_ids", "output_ids", "token_ids"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError("thinker token file must contain a JSON list of integers")
    return value


def extract_thinker_tokens(
    event_dir: Path,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    events_by_request: dict[str, list[tuple[int, int, int]]] = {}
    source_files = sorted(event_dir.glob("events_*.jsonl"))
    if not source_files:
        raise ValueError(f"no profiler event files found under {event_dir}")
    for path in source_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_name") != "thinker_token_emit":
                continue
            rid = str(event["request_id"])
            metadata = event.get("metadata") or {}
            events_by_request.setdefault(rid, []).append(
                (
                    int(metadata["token_index"]),
                    int(event.get("timestamp_ns") or 0),
                    int(metadata["token_id"]),
                )
            )

    if request_id is None:
        request_ids = sorted(events_by_request)
        if len(request_ids) != 1:
            raise ValueError(
                "event directory must contain exactly one token-emitting request "
                "when --request-id is omitted"
            )
        request_id = request_ids[0]
    if request_id not in events_by_request:
        raise ValueError(f"no thinker token events found for request {request_id}")

    ordered = sorted(events_by_request[request_id])
    indices = [item[0] for item in ordered]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"thinker token indices are incomplete or duplicated: {indices}"
        )
    token_ids = [item[2] for item in ordered]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "captured_thinker_tokens",
        "request_id": request_id,
        "thinker_token_ids": token_ids,
        "thinker_token_sha256": _canonical_digest(token_ids),
        "event_dir": str(event_dir.resolve()),
        "source_files": [path.name for path in source_files],
    }


def build_replay_contract(
    fingerprint: dict[str, Any],
    thinker_token_ids: list[int],
    *,
    voice: str,
    language: str,
    style: str | None,
    partial_start_min_tokens: int,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    if partial_start_min_tokens < 4:
        raise ValueError("Qwen3.5 talker partial start requires at least four tokens")
    if len(thinker_token_ids) < partial_start_min_tokens:
        raise ValueError(
            "thinker token replay is shorter than partial_start_min_tokens"
        )
    token_digest = _canonical_digest(thinker_token_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fixed_thinker_token_replay",
        "model_id": fingerprint["model"]["model_id"],
        "engine": fingerprint.get("engine"),
        "repo_commit": (fingerprint.get("repository") or {}).get("commit"),
        "thinker_token_ids": thinker_token_ids,
        "thinker_token_sha256": token_digest,
        "partial_start_min_tokens": partial_start_min_tokens,
        "audio": {
            "voice": voice,
            "language": language,
            "style": style,
        },
        "talker_sampling": {
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--engine", required=True)
    fingerprint.add_argument("--model-path", type=Path, required=True)
    fingerprint.add_argument("--repo-path", type=Path)
    fingerprint.add_argument("--full-weight-hash", action="store_true")
    fingerprint.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    extract = subparsers.add_parser("extract-tokens")
    extract.add_argument("--event-dir", type=Path, required=True)
    extract.add_argument("--request-id")
    extract.add_argument("--output", type=Path, required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--fingerprint", type=Path, required=True)
    replay.add_argument("--thinker-tokens", type=Path, required=True)
    replay.add_argument("--voice", default="Ethan")
    replay.add_argument("--language", default="auto")
    replay.add_argument("--style")
    replay.add_argument("--partial-start-min-tokens", type=int, default=4)
    replay.add_argument("--seed", type=int, default=1234)
    replay.add_argument("--temperature", type=float, default=0.9)
    replay.add_argument("--top-k", type=int, default=50)
    replay.add_argument("--top-p", type=float, default=1.0)
    replay.add_argument("--repetition-penalty", type=float, default=1.05)
    replay.add_argument("--max-new-tokens", type=int, default=4096)
    replay.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fingerprint":
        result = build_model_fingerprint(
            args.model_path,
            engine=args.engine,
            repo_path=args.repo_path,
            full_weight_hash=args.full_weight_hash,
        )
    elif args.command == "compare":
        result = build_comparison_contract(
            json.loads(args.candidate.read_text()),
            json.loads(args.reference.read_text()),
        )
    elif args.command == "extract-tokens":
        result = extract_thinker_tokens(
            args.event_dir,
            request_id=args.request_id,
        )
    else:
        result = build_replay_contract(
            json.loads(args.fingerprint.read_text()),
            _load_token_ids(args.thinker_tokens),
            voice=args.voice,
            language=args.language,
            style=args.style,
            partial_start_min_tokens=args.partial_start_min_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
