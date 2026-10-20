# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Build a ``RunConfig`` from a training manifest and a GPU.

Inputs: the validated manifest, a resource-class dict or GPU name, and the
checkpoint ``config.json``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.specs import (
    Algorithm,
    DeviceSpec,
    GenerationSettings,
    GiB,
    LoraTargetScope,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
    WeightDtype,
    WeightVariant,
)
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.algorithms.llmppo import LLMPPOSpec
from agilerl.arena.models.algorithms.llmreinforce import LLMREINFORCESpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.networks import LoraConfigDict, VLLMConfig

# Manifest algorithm names -> the estimator's algorithm identifiers. The
# estimator keys its reference/critic residency rules on these, so an
# unmapped LLM algorithm is a hard error.
ALGORITHM_NAMES: dict[str, Algorithm] = {
    "GRPO": "grpo",
    "GSPO": "gspo",
    "CISPO": "cispo",
    "LLMPPO": "ppo",
    "LLMREINFORCE": "reinforce",
    "DPO": "dpo",
    "SFT": "sft",
}

# Module names that mean an adapter touches attention projections only.
ATTENTION_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query", "key", "value"}
)

# Sequences the framework's colocated engine decodes when vllm_config sets none.
COLOCATED_MAX_NUM_SEQS = 8


class GpuInfo(BaseModel):
    """One accelerator model a resource class can provide."""

    model_config = ConfigDict(frozen=True)

    # Canonical CUDA device name, matching ``torch.cuda.get_device_name`` so
    # the per-device CUDA-context constants resolve.
    name: str
    total_gib: float
    cc_major: int
    cc_minor: int

    def device_spec(self) -> DeviceSpec:
        return DeviceSpec(
            total_bytes=int(self.total_gib * GiB),
            name=self.name,
            compute_capability=(self.cc_major, self.cc_minor),
        )


# The accelerators Arena resource classes are built from, in match order —
# longer tokens first, because "A100" contains "A10" and "L40S" contains
# "L4". Matched against the tier's ``gpu_type`` after normalisation.
GPU_CATALOGUE: tuple[tuple[str, GpuInfo], ...] = (
    ("H200", GpuInfo(name="NVIDIA H200", total_gib=141, cc_major=9, cc_minor=0)),
    (
        "H100",
        GpuInfo(name="NVIDIA H100 80GB HBM3", total_gib=80, cc_major=9, cc_minor=0),
    ),
    (
        "A100-80",
        GpuInfo(name="NVIDIA A100-SXM4-80GB", total_gib=80, cc_major=8, cc_minor=0),
    ),
    (
        "A100",
        GpuInfo(name="NVIDIA A100-SXM4-40GB", total_gib=40, cc_major=8, cc_minor=0),
    ),
    ("A10", GpuInfo(name="NVIDIA A10G", total_gib=24, cc_major=8, cc_minor=6)),
    ("L40", GpuInfo(name="NVIDIA L40S", total_gib=48, cc_major=8, cc_minor=9)),
    ("L4", GpuInfo(name="NVIDIA L4", total_gib=24, cc_major=8, cc_minor=9)),
    ("T4", GpuInfo(name="Tesla T4", total_gib=16, cc_major=7, cc_minor=5)),
    (
        "V100",
        GpuInfo(name="Tesla V100-SXM2-16GB", total_gib=16, cc_major=7, cc_minor=0),
    ),
)


def lookup_gpu(gpu_type: str) -> GpuInfo | None:
    """Resolve a free-form ``gpu_type`` ("NVIDIA L4", "a100 80gb") to a known GPU."""
    # Longest catalogue token first; "A100-80" matches any A100 string with an "80".
    collapsed = re.sub(r"[^A-Z0-9]", "", gpu_type.upper().replace("NVIDIA", ""))
    for token, info in GPU_CATALOGUE:
        if token == "A100-80":
            if "A100" in collapsed and "80" in collapsed:
                return info
        elif token in collapsed:
            return info
    return None


def device_spec_from_resource_class(
    resource_class: dict[str, Any],
    *,
    gpu_memory_gib: float | None = None,
) -> DeviceSpec:
    """The device one GPU of a resource-class dict provides.

    The estimate is per GPU, so ``num_gpus`` is ignored; data-parallel
    sharding comes from ``training_gpus_per_agent``.

    :param resource_class: The tier dict, e.g.
        ``{"name": "a100-1x", "gpu_type": "NVIDIA A100", "num_gpus": 1, ...}``.
    :param gpu_memory_gib: Per-GPU memory override for a ``gpu_type`` the
        catalogue does not know.
    :raises ValueError: A CPU-only tier, or an unknown ``gpu_type`` with no
        ``gpu_memory_gib`` override.
    """
    gpu_type = str(resource_class.get("gpu_type") or "")
    if not gpu_type and gpu_memory_gib is None:
        msg = (
            f"Resource class {resource_class.get('name')!r} provides no GPU; "
            "LLM memory estimation needs one."
        )
        raise ValueError(msg)

    info = lookup_gpu(gpu_type) if gpu_type else None
    if info is None:
        if gpu_memory_gib is None:
            known = ", ".join(entry.name for _, entry in GPU_CATALOGUE)
            msg = (
                f"Unknown gpu_type {gpu_type!r}; pass gpu_memory_gib explicitly. "
                f"Known accelerators: {known}."
            )
            raise ValueError(msg)
        return DeviceSpec(total_bytes=int(gpu_memory_gib * GiB), name=gpu_type or None)
    spec = info.device_spec()
    if gpu_memory_gib is not None:
        spec = spec.model_copy(update={"total_bytes": int(gpu_memory_gib * GiB)})
    return spec


def _validated(
    manifest: TrainingManifest | str | Path | dict[str, Any],
) -> TrainingManifest:
    if isinstance(manifest, TrainingManifest):
        return manifest
    return TrainingManifest.get_validated(manifest, mode="python")


def llm_spec(manifest: TrainingManifest) -> LLMAlgorithmSpec:
    """The manifest's LLM algorithm spec, if this estimator sizes it."""
    spec = manifest.algorithm
    if not isinstance(spec, LLMAlgorithmSpec):
        msg = (
            f"Memory estimation covers the LLM fine-tuning algorithms; "
            f"{spec.name} builds its own (small) networks and is not sized."
        )
        raise ValueError(msg)
    if spec.name not in ALGORITHM_NAMES:
        msg = f"No memory model for LLM algorithm {spec.name!r}."
        raise ValueError(msg)
    return spec


def _lora_config(spec: LLMAlgorithmSpec) -> LoraConfigDict:
    # Training is LoRA-only; an omitted section trains with the LoraConfigDict defaults.
    return spec.lora_config or LoraConfigDict()


def _lora_target_scope(lora: LoraConfigDict) -> LoraTargetScope:
    targets = lora.target_modules
    if isinstance(targets, str):
        return "all-linear"
    return "attention-only" if set(targets) <= ATTENTION_MODULES else "all-linear"


def _reject_trainer_quantization(quantization: str | dict[str, Any] | None) -> None:
    """Refuse trainer-side quantization: async runs unquantized bases."""
    if isinstance(quantization, dict):
        active = bool(
            quantization.get("load_in_4bit") or quantization.get("load_in_8bit")
        )
    else:
        active = quantization is not None and quantization.lower() != "none"
    if active:
        msg = (
            f"Trainer quantization ({quantization!r}) is not sized; async "
            "rollout runs unquantized bases."
        )
        raise ValueError(msg)


def _reject_eager_attn_implementation(value: str | None) -> None:
    """Refuse the eager backend: only tiled backends are sized."""
    if value == "eager":
        msg = (
            "attn_implementation='eager' materialises a rows x heads x S x S "
            "score matrix; this estimator sizes flash-style backends only."
        )
        raise ValueError(msg)


def _weight_dtype(value: str | None) -> WeightDtype:
    lowered = (value or "auto").lower()
    if lowered in {"auto", "bfloat16", "bf16"}:
        return "bf16"
    if lowered in {"float16", "fp16", "half"}:
        return "fp16"
    if lowered in {"float32", "fp32", "float"}:
        return "fp32"
    msg = f"Unknown vLLM dtype {value!r}; this estimator sizes bf16, fp16, and fp32."
    raise ValueError(msg)


def _reject_kv_cache_quantization(value: str | None) -> None:
    """Refuse KV cache quantization: the cache stays at weight dtype."""
    if value is not None and value.lower() != "auto":
        msg = (
            f"kv_cache_dtype ({value!r}) is not sized; this estimator sizes "
            "unquantized KV caches."
        )
        raise ValueError(msg)


def _reject_engine_quantization(value: str | None) -> None:
    """Refuse engine-side quantization: this estimator sizes unquantized engines."""
    if value is not None and value.lower() != "none":
        msg = (
            f"Engine quantization ({value!r}) is not sized; this estimator sizes "
            "unquantized engines."
        )
        raise ValueError(msg)


def _group_size(spec: LLMAlgorithmSpec) -> int:
    if isinstance(spec, GRPOSpec):
        return spec.group_size
    return 1


def _max_num_seqs(
    manifest: TrainingManifest, spec: LLMAlgorithmSpec, vllm: VLLMConfig
) -> int:
    """The manifest's ``max_num_seqs``, else the count the rollout runtime derives.

    An async engine runs one sequence per episode it keeps going:
    ``rollout_batch_size // rollout_engines_per_agent x group_size``. ``auto``
    counts one engine, the largest share any engine can get.
    """
    if vllm.max_num_seqs is not None:
        return vllm.max_num_seqs
    training = manifest.training
    if not training.async_rollout:
        return COLOCATED_MAX_NUM_SEQS
    engines = training.rollout_engines_per_agent
    n_engines = 1 if engines == "auto" else engines
    # Async validation sets an unset rollout_batch_size to 1.
    rollout_batch_size = training.rollout_batch_size or 1
    return rollout_batch_size // n_engines * _group_size(spec)


def _resolve_vllm(spec: LLMAlgorithmSpec) -> VLLMConfig:
    """The engine config the run starts, checked against what this estimator sizes."""
    # SFT and DPO start no engine; their generation bar is ignored.
    if not isinstance(spec, RolloutLLMSpec):
        return VLLMConfig()
    vllm = spec.vllm_config
    if vllm is None:
        msg = (
            f"{spec.name} names no vllm_config; this estimator sizes vLLM rollout "
            "only, so name the rollout engine config."
        )
        raise ValueError(msg)
    _reject_engine_quantization(vllm.quantization)
    model = vllm.vllm_model_name_or_path
    if model is not None and model != spec.pretrained_model_name_or_path:
        msg = (
            f"vllm_model_name_or_path ({model!r}) serves a different checkpoint "
            "than the trainer fine-tunes "
            f"({spec.pretrained_model_name_or_path!r}); this estimator sizes one arch."
        )
        raise ValueError(msg)
    if vllm.tensor_parallel_size != 1:
        msg = (
            f"tensor_parallel_size={vllm.tensor_parallel_size} is not modeled; "
            "this estimator sizes single-GPU engines."
        )
        raise ValueError(msg)
    if spec.vllm_engine_args:
        warnings.warn(
            "vllm_engine_args are forwarded to vLLM verbatim; this estimator sizes "
            "the manifest config and cannot see them.",
            stacklevel=2,
        )
    return vllm


def _stripped_towers(vllm: VLLMConfig) -> bool:
    towers = vllm.strip_multimodal_towers
    if towers is True:
        return True
    if isinstance(towers, list) and towers:
        warnings.warn(
            f"strip_multimodal_towers names {towers}; this estimator strips all "
            "towers or none, so it sizes them kept.",
            stacklevel=2,
        )
    return False


def _require_single_row_micro_batch(
    spec: LLMAlgorithmSpec, n_training_gpus: int
) -> None:
    """Refuse any resolved micro-batch but one row, resolved as the trainer does."""
    # Unset: the whole per-rank batch. mini_batch_size set alone is the micro-batch.
    micro = spec.micro_batch_size_per_gpu
    if micro is None:
        if n_training_gpus < 2:
            micro = spec.batch_size
        elif isinstance(spec, RolloutLLMSpec) and spec.mini_batch_size is not None:
            micro = spec.mini_batch_size
        else:
            micro = max(spec.batch_size // n_training_gpus, 1)
    if micro != 1:
        msg = (
            f"micro_batch_size_per_gpu resolves to {micro}; this estimator sizes "
            "single-row micro-batches only."
        )
        raise ValueError(msg)


def _reject_unbounded_liger_loss(spec: LLMAlgorithmSpec) -> None:
    """Refuse Liger loss outside token-level importance sampling."""
    # Above token level the kernel takes whole sequences per chunk, unbounded.
    # gspo forces trajectory level.
    if not spec.use_liger_loss:
        return
    is_gspo = isinstance(spec, GRPOSpec) and spec.loss_type == "gspo"
    level: str | None = None
    if isinstance(spec, (GRPOSpec, LLMPPOSpec, LLMREINFORCESpec)):
        level = spec.importance_sampling_level
    if is_gspo or level in ("turn", "trajectory"):
        msg = (
            "use_liger_loss with turn/trajectory importance sampling is not "
            "memory-bounded; this estimator sizes the chunked token-level path."
        )
        raise ValueError(msg)


def training_settings_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
) -> TrainingSettings:
    """Trainer-side settings, read as the trainer reads the manifest.

    Settings the estimator cannot size (quantization, multi-row micro-batches,
    the eager backend, Liger above token level) raise ``ValueError``.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    lora = _lora_config(spec)
    n_training_gpus = manifest.training.training_gpus_per_agent
    _reject_trainer_quantization(spec.quantization)
    _require_single_row_micro_batch(spec, n_training_gpus)
    _reject_eager_attn_implementation(spec.attn_implementation)
    _reject_unbounded_liger_loss(spec)
    return TrainingSettings(
        algorithm=ALGORITHM_NAMES[spec.name],
        group_size=_group_size(spec),
        trajectories_per_update=spec.batch_size * _group_size(spec),
        max_model_len=(
            spec.max_row_tokens
            if isinstance(spec, RolloutLLMSpec) and spec.max_row_tokens is not None
            else spec.max_model_len
        ),
        lora_rank=lora.lora_r,
        lora_target_scope=_lora_target_scope(lora),
        lora_dropout=lora.lora_dropout,
        lora_packed_target_matrices=len(lora.target_parameters or []),
        packed_moe_dispatch=(
            "contracted" if lora.target_parameters else "materialized"
        ),
        beta=spec.beta,
        use_separate_reference_adapter=spec.use_separate_reference_adapter,
        gradient_checkpointing=spec.gradient_checkpointing,
        activation_offload=spec.activation_offload,
        chunk_rows=spec.chunk_rows,
        n_training_gpus=n_training_gpus,
        fsdp=spec.fsdp,
        # Unset sizes separate passes: the trainer fuses only when that fits.
        fuse_actor_critic_pass=isinstance(spec, LLMPPOSpec)
        and spec.fuse_actor_critic_pass is True,
        checkpoint_optimizer=manifest.training.checkpoint_optimizer,
        async_rollout=manifest.training.async_rollout,
    )


def generation_settings_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
) -> GenerationSettings:
    """The engine-side settings the manifest's vLLM config pins down.

    A rollout without an engine section names no engine to size. SFT and
    DPO start no engine, so their settings are defaults the estimator
    ignores.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    vllm = _resolve_vllm(spec)
    lora = _lora_config(spec)
    stripped = _stripped_towers(vllm)
    _reject_kv_cache_quantization(vllm.kv_cache_dtype)
    return GenerationSettings(
        gpu_memory_utilization=vllm.gpu_memory_utilization,
        max_num_seqs=_max_num_seqs(manifest, spec, vllm),
        max_model_len=spec.max_model_len,
        max_num_batched_tokens=vllm.max_num_batched_tokens,
        kv_cache_memory_bytes=vllm.kv_cache_memory_bytes,
        enforce_eager=vllm.enforce_eager,
        max_lora_rank=max(vllm.max_lora_rank, lora.lora_r),
        max_loras=vllm.max_loras,
        weight_dtype=_weight_dtype(vllm.dtype),
        weight_variant="engine" if stripped else "base",
        concurrent_requests=spec.batch_size * _group_size(spec),
    )


def _model_spec(
    spec: LLMAlgorithmSpec,
    model_config: dict[str, Any],
    n_params: int | None,
    generation: GenerationSettings,
) -> ModelSpec:
    variants = [WeightVariant()]
    if generation.weight_variant == "engine":
        variants.append(WeightVariant(name="engine", stripped_multimodal=True))
    model_id = spec.pretrained_model_name_or_path
    if model_id is None:
        msg = "manifest names no pretrained model."
        raise ValueError(msg)
    return ModelSpec(
        model_id=model_id,
        arch=ModelArch.from_hf_config(model_config),
        n_params=n_params,
        variants=tuple(variants),
    )


def run_config_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
    device: DeviceSpec,
    model_config: dict[str, Any],
    *,
    n_params: int | None = None,
    gen_device: DeviceSpec | None = None,
) -> RunConfig:
    """Assemble the estimator's input from the three things a caller holds.

    :param manifest: The training manifest — a path, a raw dict, or an
        already-validated :class:`TrainingManifest`.
    :param device: The training GPU, usually from
        :func:`device_spec_from_resource_class`.
    :param model_config: The ``config.json`` of the checkpoint the manifest
        names. Fetched by the caller so the calculation core stays free of I/O.
    :param n_params: Exact parameter count when known, e.g. from safetensors
        metadata; ``None`` falls back to the analytic count.
    :param gen_device: The rollout engine's device. Defaults to ``device``.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    training = training_settings_from_manifest(manifest)
    if training.uses_generation_engine and not manifest.training.async_rollout:
        msg = (
            f"rollout_mode={manifest.training.rollout_mode!r} is not sized; "
            "this estimator sizes async rollout only."
        )
        raise ValueError(msg)
    generation = generation_settings_from_manifest(manifest)
    return RunConfig(
        model=_model_spec(spec, model_config, n_params, generation),
        train_device=device,
        gen_device=gen_device or device,
        training=training,
        generation=generation,
        orchestrated=True,
    )


def estimate_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
    device: DeviceSpec,
    model_config: dict[str, Any],
    *,
    n_params: int | None = None,
    gen_device: DeviceSpec | None = None,
) -> RunEstimate:
    """Estimate both phase peaks straight from the manifest."""
    return estimate_run(
        run_config_from_manifest(
            manifest, device, model_config, n_params=n_params, gen_device=gen_device
        )
    )
