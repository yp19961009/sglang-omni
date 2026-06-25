# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for pipeline wiring."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class RelayConfig(BaseModel):
    """Relay configuration for stage data transfer."""

    model_config = ConfigDict(extra="forbid")

    slot_size_mb: int = 512
    credits: int = 2
    rank: int | None = None
    world_size: int | None = None
    device: str = "cpu"


class EndpointsConfig(BaseModel):
    """Endpoint allocation settings."""

    model_config = ConfigDict(extra="forbid")

    base_path: str = "/tmp/sglang_omni"


class ParallelismConfig(BaseModel):
    """Supported parallelism for one logical stage."""

    model_config = ConfigDict(extra="forbid")

    tp: int = 1

    def model_post_init(self, __context: Any = None) -> None:
        if self.tp < 1:
            raise ValueError("parallelism.tp must be >= 1")


class StageResourceConfig(BaseModel):
    """Placement-resource intent for one stage rank/process."""

    model_config = ConfigDict(extra="forbid")

    total_gpu_memory_fraction: float | None = Field(
        default=None,
        description=(
            "Per-rank/process budget as a fraction of total physical GPU "
            "memory. After TP expansion, each rank contributes this budget to "
            "its assigned GPU."
        ),
    )

    def model_post_init(self, __context: Any = None) -> None:
        value = self.total_gpu_memory_fraction
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(
                "runtime.resources.total_gpu_memory_fraction must be in (0, 1]"
            )


class SGLangServerArgsConfig(BaseModel):
    """Typed subset of SGLang ServerArgs exposed through pipeline config."""

    model_config = ConfigDict(extra="forbid")

    mem_fraction_static: float | None = None
    max_running_requests: int | None = None
    max_mamba_cache_size: int | None = None
    mamba_full_memory_ratio: float | None = None

    def model_post_init(self, __context: Any = None) -> None:
        mem_fraction_static = self.mem_fraction_static
        if (
            mem_fraction_static is not None
            and not 0.0 < mem_fraction_static < 1.0
        ):
            raise ValueError(
                "runtime.sglang_server_args.mem_fraction_static must be in (0, 1)"
            )
        if (
            self.max_running_requests is not None
            and self.max_running_requests < 1
        ):
            raise ValueError(
                "runtime.sglang_server_args.max_running_requests must be positive"
            )
        if (
            self.max_mamba_cache_size is not None
            and self.max_mamba_cache_size < 1
        ):
            raise ValueError(
                "runtime.sglang_server_args.max_mamba_cache_size must be positive"
            )
        if (
            self.mamba_full_memory_ratio is not None
            and not 0.0 < self.mamba_full_memory_ratio <= 1.0
        ):
            raise ValueError(
                "runtime.sglang_server_args.mamba_full_memory_ratio must be in (0, 1]"
            )


class StageRuntimeConfig(BaseModel):
    """Typed runtime intent for one stage.

    Backend-specific values stay namespaced. For example,
    sglang_server_args is translated into SGLang ServerArgs by the
    runtime adapter, not by placement planning.
    """

    model_config = ConfigDict(extra="forbid")

    resources: StageResourceConfig = Field(default_factory=StageResourceConfig)
    max_seq_len: int | None = None
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None
    video_override_max_pixels: bool | None = None
    video_seconds_per_chunk: float | None = None
    video_position_id_per_seconds: float | None = None
    audio_target_sr: int | None = None
    audio_sampling_rate: int | None = None
    sampling_rate: int | None = None
    audio_timestamp_interval: int | None = None
    timestamp_interval: int | None = None
    audio_downsample_times: int | None = None
    downsample_times: int | None = None
    audio_downsample_chunk_size: int | None = None
    downsample_chunk_size: int | None = None
    code2wav_stream_chunk_size: int | None = None
    send_chunk_size: int | None = None
    code2wav_codec_eos_token_id: int | None = None
    code2wav_sample_rate: int | None = None
    code2wav_left_context_size: int | None = None
    code2wav_enable_dynamic_chunk: bool | None = None
    enable_dynamic_chunk: bool | None = None
    code2wav_dynamic_batch: bool | None = None
    dynamic_batch: bool | None = None
    code2wav_dynamic_chunk_sizes: tuple[int, ...] | str | None = None
    code2wav_dynamic_chunk_steps: tuple[int, ...] | str | None = None
    code2wav_enable_torch_compile: bool | None = None
    code2wav_enable_torch_compile_first_chunk: bool | None = None
    enable_torch_compile_first_chunk: bool | None = None
    code2wav_odeint_method: str | None = None
    odeint_method: str | None = None
    code2wav_odeint_method_relaxed: bool | None = None
    odeint_method_relaxed: bool | None = None
    code2wav_batched_chunk: int | None = None
    batched_chunk: int | None = None
    code2wav_frequency: str | None = None
    frequency: str | None = None
    code2wav_dit_quant: str | None = None
    dit_quant: str | None = None
    sglang_server_args: SGLangServerArgsConfig = Field(
        default_factory=SGLangServerArgsConfig
    )

    def model_post_init(self, __context: Any = None) -> None:
        self._normalize_audio_sampling_rate_aliases()
        self._normalize_audio_processor_aliases()
        self._normalize_code2wav_aliases()
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError("runtime.max_seq_len must be positive")
        if self.video_fps is not None and self.video_fps <= 0:
            raise ValueError("runtime.video_fps must be positive")
        if (
            self.video_seconds_per_chunk is not None
            and self.video_seconds_per_chunk <= 0
        ):
            raise ValueError("runtime.video_seconds_per_chunk must be positive")
        if (
            self.video_position_id_per_seconds is not None
            and self.video_position_id_per_seconds <= 0
        ):
            raise ValueError(
                "runtime.video_position_id_per_seconds must be positive"
            )
        for field_name in (
            "image_min_pixels",
            "image_max_pixels",
            "video_max_frames",
            "video_min_frames",
            "video_min_pixels",
            "video_max_pixels",
            "video_total_pixels",
            "audio_target_sr",
            "audio_sampling_rate",
            "sampling_rate",
            "audio_timestamp_interval",
            "timestamp_interval",
            "audio_downsample_chunk_size",
            "downsample_chunk_size",
            "code2wav_stream_chunk_size",
            "send_chunk_size",
            "code2wav_sample_rate",
            "code2wav_batched_chunk",
            "batched_chunk",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"runtime.{field_name} must be positive")
        for field_name in (
            "audio_downsample_times",
            "downsample_times",
            "code2wav_codec_eos_token_id",
            "code2wav_left_context_size",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"runtime.{field_name} must be non-negative")

    def _normalize_audio_sampling_rate_aliases(self) -> None:
        self._normalize_int_aliases(
            canonical_name="audio_target_sr",
            alias_names=("audio_sampling_rate", "sampling_rate"),
            label="audio sampling rate",
        )

    def _normalize_audio_processor_aliases(self) -> None:
        self._normalize_int_aliases(
            canonical_name="audio_timestamp_interval",
            alias_names=("timestamp_interval",),
            label="audio processor timestamp interval",
        )
        self._normalize_int_aliases(
            canonical_name="audio_downsample_times",
            alias_names=("downsample_times",),
            label="audio processor downsample times",
        )
        self._normalize_int_aliases(
            canonical_name="audio_downsample_chunk_size",
            alias_names=("downsample_chunk_size",),
            label="audio processor downsample chunk size",
        )

    def _normalize_code2wav_aliases(self) -> None:
        self._normalize_int_aliases(
            canonical_name="code2wav_stream_chunk_size",
            alias_names=("send_chunk_size",),
            label="code2wav stream chunk size",
        )
        self._normalize_positive_int_sequence(
            field_name="code2wav_dynamic_chunk_sizes",
            label="code2wav dynamic chunk sizes",
        )
        self._normalize_positive_int_sequence(
            field_name="code2wav_dynamic_chunk_steps",
            label="code2wav dynamic chunk steps",
        )
        self._normalize_bool_aliases(
            canonical_name="code2wav_enable_dynamic_chunk",
            alias_names=(
                "enable_dynamic_chunk",
                "code2wav_dynamic_batch",
                "dynamic_batch",
            ),
            label="code2wav dynamic chunk",
        )
        self._normalize_bool_aliases(
            canonical_name="code2wav_enable_torch_compile_first_chunk",
            alias_names=("enable_torch_compile_first_chunk",),
            label="code2wav torch compile first chunk",
        )
        self._normalize_text_aliases(
            canonical_name="code2wav_odeint_method",
            alias_names=("odeint_method",),
            label="code2wav odeint method",
        )
        self._normalize_bool_aliases(
            canonical_name="code2wav_odeint_method_relaxed",
            alias_names=("odeint_method_relaxed",),
            label="code2wav odeint method relaxed",
        )
        self._normalize_int_aliases(
            canonical_name="code2wav_batched_chunk",
            alias_names=("batched_chunk",),
            label="code2wav batched chunk",
        )
        self._normalize_text_aliases(
            canonical_name="code2wav_frequency",
            alias_names=("frequency",),
            label="code2wav frequency",
        )
        self._normalize_text_aliases(
            canonical_name="code2wav_dit_quant",
            alias_names=("dit_quant",),
            label="code2wav dit quant",
        )

    def _normalize_int_aliases(
        self,
        *,
        canonical_name: str,
        alias_names: tuple[str, ...],
        label: str,
    ) -> None:
        field_names = (canonical_name, *alias_names)
        values = [
            (name, getattr(self, name))
            for name in field_names
            if getattr(self, name) is not None
        ]
        if not values:
            return
        canonical = int(values[0][1])
        for name, value in values[1:]:
            if int(value) != canonical:
                raise ValueError(
                    f"runtime {label} aliases disagree: "
                    f"{'/'.join(field_names)} include "
                    f"{values[0][0]}={canonical} and {name}={value}"
                )
        setattr(self, canonical_name, canonical)

    def _normalize_positive_int_sequence(
        self,
        *,
        field_name: str,
        label: str,
    ) -> None:
        value = getattr(self, field_name)
        if value is None:
            return
        values = self._parse_positive_int_sequence(value, label=label)
        setattr(self, field_name, values)

    def _parse_positive_int_sequence(
        self,
        value: tuple[int, ...] | str,
        *,
        label: str,
    ) -> tuple[int, ...]:
        if isinstance(value, str):
            pieces = [
                piece.strip()
                for piece in value.replace(",", " ").split()
                if piece.strip()
            ]
            values = tuple(int(piece) for piece in pieces)
        else:
            values = tuple(int(item) for item in value)
        if not values or any(item < 1 for item in values):
            raise ValueError(f"runtime {label} must contain positive integers")
        return values

    def _normalize_bool_aliases(
        self,
        *,
        canonical_name: str,
        alias_names: tuple[str, ...],
        label: str,
    ) -> None:
        field_names = (canonical_name, *alias_names)
        values = [
            (name, getattr(self, name))
            for name in field_names
            if getattr(self, name) is not None
        ]
        if not values:
            return
        canonical = bool(values[0][1])
        for name, value in values[1:]:
            if bool(value) != canonical:
                raise ValueError(
                    f"runtime {label} aliases disagree: "
                    f"{'/'.join(field_names)} include "
                    f"{values[0][0]}={canonical} and {name}={value}"
                )
        setattr(self, canonical_name, canonical)

    def _normalize_text_aliases(
        self,
        *,
        canonical_name: str,
        alias_names: tuple[str, ...],
        label: str,
    ) -> None:
        field_names = (canonical_name, *alias_names)
        values = [
            (name, getattr(self, name))
            for name in field_names
            if getattr(self, name) is not None
        ]
        if not values:
            return
        canonical = str(values[0][1]).strip()
        if not canonical:
            raise ValueError(f"runtime {label} must not be empty")
        comparable = canonical.casefold()
        for name, value in values[1:]:
            candidate = str(value).strip()
            if not candidate:
                raise ValueError(f"runtime {label} must not be empty")
            if candidate.casefold() != comparable:
                raise ValueError(
                    f"runtime {label} aliases disagree: "
                    f"{'/'.join(field_names)} include "
                    f"{values[0][0]}={canonical} and {name}={value}"
                )
        setattr(self, canonical_name, canonical)


class PlacementConfig(BaseModel):
    """Pipeline-level placement planning limits."""

    model_config = ConfigDict(extra="forbid")

    max_total_gpu_memory_fraction_per_gpu: float = 1.0
    require_memory_fraction_for_colocation: bool = True

    def model_post_init(self, __context: Any = None) -> None:
        value = self.max_total_gpu_memory_fraction_per_gpu
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "placement.max_total_gpu_memory_fraction_per_gpu must be in (0, 1]"
            )


class StageConfig(BaseModel):
    """Single pipeline stage configuration.

    Minimal example::

        StageConfig(name="decode", factory="...create_decode", terminal=True)

    Fan-in example::

        StageConfig(
            name="aggregate",
            factory="...create_aggregate",
            wait_for=["preprocessor", "image_enc", "audio_enc"],
            merge_fn="...merge_for_thinker",
            next="thinker",
        )
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity ---
    name: str

    # --- Factory ---
    factory: str
    factory_args: dict[str, Any] = Field(default_factory=dict)

    # --- Routing (set `next` for static routing or `terminal`) ---
    next: str | list[str] | None = None
    terminal: bool = False
    route_fn: str | None = None

    # --- GPU / parallelism ---
    gpu: int | list[int] | None = None
    tp_size: int = 1
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    process: str | None = None

    # --- Runtime intent ---
    runtime: StageRuntimeConfig = Field(default_factory=StageRuntimeConfig)
    runtime_arg_map: dict[str, str] = Field(default_factory=dict)

    # --- Fan-in ---
    wait_for: list[str] | None = None
    wait_for_fn: str | None = None
    merge_fn: str | None = None

    # --- Streaming ---
    stream_to: list[str] = Field(default_factory=list)
    stream_done_to_fn: str | None = None
    can_accept_stream_before_payload: bool = False

    # --- Route-specific payload projection ---
    project_payload: dict[str, str] = Field(default_factory=dict)

    # --- Relay (auto-inferred from gpu when None) ---
    relay: RelayConfig | None = None

    def model_post_init(self, __context: Any = None) -> None:
        fields_set = self.__pydantic_fields_set__
        tp_size_set = "tp_size" in fields_set
        parallelism_set = "parallelism" in fields_set
        if self.tp_size < 1:
            raise ValueError(f"Stage {self.name!r} must have tp_size >= 1")
        if self.process is not None:
            self.process = self.process.strip()
            if not self.process:
                raise ValueError(f"Stage {self.name!r} process must not be empty")
        if parallelism_set and tp_size_set and self.parallelism.tp != self.tp_size:
            raise ValueError(
                f"Stage {self.name!r}: tp_size={self.tp_size} conflicts with "
                f"parallelism.tp={self.parallelism.tp}"
            )
        if not parallelism_set and self.tp_size != self.parallelism.tp:
            self.parallelism.tp = self.tp_size
        elif (
            parallelism_set and not tp_size_set and self.tp_size != self.parallelism.tp
        ):
            self.tp_size = self.parallelism.tp


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    architecture_aliases: ClassVar[tuple[str, ...]] = ()

    model_path: str
    stages: list[StageConfig]
    name: str | None = None
    entry_stage: str | None = None
    relay_backend: Literal["shm", "nccl", "nixl", "mooncake"] = "shm"
    fused_stages: list[list[str]] = Field(default_factory=list)
    runtime_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_defaults: dict[str, str] = Field(default_factory=dict)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    placement_policy: str | None = None
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    terminal_stages_fn: str | None = None
    config_cls: str | None = None

    def model_post_init(self, __context: Any = None) -> None:
        self._validate_general()
        self._validate_fusion()
        self.config_cls = self.__class__.__name__
        if self.name is None:
            self.name = self.model_path

    @property
    def resolved_entry_stage(self) -> str:
        if self.entry_stage is not None:
            return self.entry_stage
        return self.stages[0].name

    @property
    def terminal_stages(self) -> list[str]:
        return [s.name for s in self.stages if s.terminal]

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for SGLang mem_fraction_static overrides."""
        return {}

    @classmethod
    def max_running_requests_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for SGLang max_running_requests overrides."""
        return {}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for encoder memory reserve overrides."""
        return {}

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for talker placement overrides."""
        return {}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for talker SGLang ServerArgs overrides."""
        return {}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        """Return the code2wav stage name when the pipeline supports it."""
        return None

    @classmethod
    def tensor_parallel_server_args_overrides(
        cls,
        *,
        stage_name: str,
        tp_size: int,
    ) -> dict[str, object]:
        """Return SGLang ServerArgs overrides implied by stage TP settings."""
        return {}

    @property
    def gpu_placement(self) -> dict[str, int | list[int]]:
        out: dict[str, int | list[int]] = {}
        for s in self.stages:
            if s.gpu is not None:
                out[s.name] = s.gpu
        return out

    def _validate_general(self) -> None:
        if not self.model_path:
            raise ValueError("Model path is required")

        names = [s.name for s in self.stages]
        if not names:
            raise ValueError("Pipeline must define at least one stage")
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")
        entry = self.resolved_entry_stage
        if entry not in names:
            raise ValueError(f"entry_stage {entry!r} is not defined")

        for s in self.stages:
            if not s.factory:
                raise ValueError(f"Stage {s.name!r} missing factory")
            has_next = s.next is not None
            if has_next == bool(s.terminal):
                raise ValueError(
                    f"Stage {s.name!r} must set exactly one of 'next' or 'terminal'"
                )
            if s.terminal and s.route_fn is not None:
                raise ValueError(
                    f"Stage {s.name!r} cannot set route_fn on a terminal stage"
                )
            if s.stream_done_to_fn is not None and not s.stream_to:
                raise ValueError(
                    f"Stage {s.name!r} cannot set stream_done_to_fn without stream_to"
                )
            if s.tp_size < 1:
                raise ValueError(f"Stage {s.name!r} must have tp_size >= 1")
            if s.parallelism.tp != s.tp_size:
                raise ValueError(
                    f"Stage {s.name!r}: tp_size={s.tp_size} conflicts with "
                    f"parallelism.tp={s.parallelism.tp}"
                )
            if isinstance(s.gpu, list) and len(s.gpu) != s.tp_size:
                raise ValueError(
                    f"Stage {s.name!r}: gpu has {len(s.gpu)} entries "
                    f"but tp_size={s.tp_size}"
                )
            if s.wait_for:
                if not s.merge_fn:
                    raise ValueError(f"Stage {s.name!r} has wait_for but no merge_fn")
                unknown = set(s.wait_for) - set(names)
                if unknown:
                    raise ValueError(
                        f"Stage {s.name!r} wait_for has unknown stages: {sorted(unknown)}"
                    )
            elif s.wait_for_fn is not None:
                raise ValueError(f"Stage {s.name!r} has wait_for_fn but no wait_for")
            if s.next is not None:
                targets = [s.next] if isinstance(s.next, str) else s.next
                unknown = set(targets) - set(names)
                if unknown:
                    raise ValueError(
                        f"Stage {s.name!r} next has unknown stages: {sorted(unknown)}"
                    )
            for t in s.stream_to:
                if t not in names:
                    raise ValueError(
                        f"Stage {s.name!r} stream_to references unknown stage {t!r}"
                    )
            for t in s.project_payload:
                if t not in names:
                    raise ValueError(
                        f"Stage {s.name!r} project_payload references unknown stage {t!r}"
                    )

        for stage_name in self.runtime_overrides:
            if stage_name not in names:
                raise ValueError(
                    f"runtime_overrides references unknown stage {stage_name!r}"
                )

        missing_process = [
            s.name for s in self.stages if s.tp_size == 1 and not s.process
        ]
        if missing_process:
            raise ValueError(
                "Non-TP stages must declare process; "
                f"missing process for {missing_process}"
            )

    def _validate_fusion(self) -> None:
        names = [s.name for s in self.stages]
        fused = self.fused_stages or []
        if not fused:
            return
        index_map = {n: i for i, n in enumerate(names)}
        stage_by_name = {s.name: s for s in self.stages}
        seen: set[str] = set()
        for group in fused:
            if not group or len(group) < 2:
                raise ValueError("fused_stages groups must have at least 2 stage names")
            for n in group:
                if n not in index_map:
                    raise ValueError(f"fused stage {n!r} is not defined")
                if n in seen:
                    raise ValueError(f"stage {n!r} appears in multiple fused groups")
                seen.add(n)
            indices = [index_map[n] for n in group]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError(f"fused group not adjacent/ordered: {group}")
            self._validate_fused_group_contract(group, stage_by_name)

    def _validate_fused_group_contract(
        self,
        group: list[str],
        stage_by_name: dict[str, StageConfig],
    ) -> None:
        stages = [stage_by_name[name] for name in group]

        for stage in stages:
            if stage.tp_size != 1:
                raise ValueError(
                    f"fused group {group} cannot include TP stage {stage.name!r}"
                )

        gpu_ids = {
            gpu_id for stage in stages for gpu_id in _stage_gpu_ids_for_fusion(stage)
        }
        if len(gpu_ids) > 1:
            raise ValueError(
                f"fused group {group} must fit on one GPU; got {sorted(gpu_ids)}"
            )

        for index, stage in enumerate(stages):
            if index > 0 and stage.wait_for:
                raise ValueError(
                    f"fused group {group} cannot include internal fan-in stage "
                    f"{stage.name!r}"
                )
            if index < len(stages) - 1:
                expected_next = group[index + 1]
                if stage.terminal or stage.route_fn is not None:
                    raise ValueError(
                        f"fused group {group} must be linear; internal stage "
                        f"{stage.name!r} cannot be terminal or dynamic-routed"
                    )
                if _target_list(stage.next) != [expected_next]:
                    raise ValueError(
                        f"fused group {group} must be linear; stage "
                        f"{stage.name!r} must route only to {expected_next!r}"
                    )

    def apply_fusion(self) -> tuple[list[StageConfig], dict[str, str], str]:
        name_map = {s.name: s.name for s in self.stages}
        return list(self.stages), name_map, self.resolved_entry_stage

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        return PipelineConfig(**data)


def _target_list(targets: str | list[str] | None) -> list[str]:
    if targets is None:
        return []
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _stage_gpu_ids_for_fusion(stage: StageConfig) -> tuple[int, ...]:
    gpu = stage.gpu
    if gpu is None:
        return ()
    if isinstance(gpu, int):
        return (gpu,)
    return tuple(int(gpu_id) for gpu_id in gpu)
