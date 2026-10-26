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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.resolve.batch import (
    per_rank_trajectories,
    require_batch_split,
    resolve_micro_batch_size_per_gpu,
    resolve_mini_batch_size,
    rollout_group_size,
)
from agilerl.arena.memory.resolve.fuse import (
    resolve_fuse_actor_critic_pass,
    training_settings_with_fuse,
)
from agilerl.arena.memory.resolve.lora import (
    resolve_lora_rank_ceiling,
    resolve_max_lora_rank,
)
from agilerl.arena.memory.resolve.sequences import (
    prompts_per_engine,
    resolve_max_num_seqs,
)
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
from agilerl.arena.models.algorithms.rollout_llm import (
    RolloutLLMSpec,
    records_sampling_supports,
)
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
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

# The engine variant drops the multimodal towers; a model without towers sizes it as base.
MANIFEST_WEIGHT_VARIANTS = (
    WeightVariant(),
    WeightVariant(name="engine", stripped_multimodal=True),
)

# Must match max_num_batched_tokens in the framework's FAMILY_RUNTIME_CONFIGS.
FAMILY_MAX_NUM_BATCHED_TOKENS: dict[str, int] = {
    "nemotron_h": 8192,
    "nemotron_h_omni": 8192,
}


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
# ``total_gib`` is CUDA-visible memory (``torch.cuda.mem_get_info`` total):
# measured on A100-80 and L4, nominal on the rest.
GPU_CATALOGUE: tuple[tuple[str, GpuInfo], ...] = (
    ("H200", GpuInfo(name="NVIDIA H200", total_gib=141, cc_major=9, cc_minor=0)),
    (
        "H100",
        GpuInfo(name="NVIDIA H100 80GB HBM3", total_gib=80, cc_major=9, cc_minor=0),
    ),
    (
        "A100-80",
        GpuInfo(name="NVIDIA A100-SXM4-80GB", total_gib=79.25, cc_major=8, cc_minor=0),
    ),
    (
        "A100",
        GpuInfo(name="NVIDIA A100-SXM4-40GB", total_gib=40, cc_major=8, cc_minor=0),
    ),
    ("A10", GpuInfo(name="NVIDIA A10G", total_gib=24, cc_major=8, cc_minor=6)),
    ("L40", GpuInfo(name="NVIDIA L40S", total_gib=48, cc_major=8, cc_minor=9)),
    ("L4", GpuInfo(name="NVIDIA L4", total_gib=22.03, cc_major=8, cc_minor=9)),
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


def _engine_weight_quantization(value: str | None) -> Literal["none", "fp8"]:
    """``fp8`` engine weights are one byte. Anything else is unquantized or refused."""
    if value is None or value.lower() == "none":
        return "none"
    if value.lower() == "fp8":
        return "fp8"
    msg = (
        f"Engine quantization ({value!r}) is not sized; this estimator sizes "
        "bf16 and fp8 engines."
    )
    raise ValueError(msg)


FP8_NATIVE_MIN_CC = (8, 9)
"""Lowest CUDA capability with native FP8 (vLLM's Triton FP8 MoE needs it)."""

FP8_MOE_METHODS = frozenset(
    {"fp8", "fp8_per_block", "fp8_per_tensor", "fp8_per_channel"}
)
"""Online quantization methods that serve MoE experts in FP8."""


def check_fp8_moe_backend_for_device(
    quantization: str | None,
    moe_backend: str | None,
    is_moe: bool,
    cc_major: int,
    cc_minor: int,
) -> None:
    """Require Marlin for FP8 LoRA MoE engines below native-FP8 hardware.

    vLLM forces the Triton MoE backend on quantized LoRA engines, and its
    Triton FP8 kernels need SM89+ native FP8, so every fp8 scheme fails at
    engine startup on A100 and older. Marlin serves FP8 there. H100/L4 and
    dense models keep whatever backend they name.

    Callers check rollout engines, which serve LoRA in this stack.

    :param quantization: Engine ``quantization``; only fp8-family methods trigger.
    :param moe_backend: Explicit ``moe_backend`` engine arg, or None for default.
    :param is_moe: Whether the model has MoE layers (dense FP8 never consults
        the MoE backend oracle).
    :param cc_major: Engine CUDA compute capability major.
    :param cc_minor: Engine CUDA compute capability minor.
    :raises ValueError: If Marlin is required but another backend (or none) is set.
    """
    if (
        not is_moe
        or quantization is None
        or quantization.lower() not in FP8_MOE_METHODS
        or (cc_major, cc_minor) >= FP8_NATIVE_MIN_CC
    ):
        return
    backend = moe_backend if isinstance(moe_backend, str) else ""
    if backend.strip().lower().replace("-", "_") == "marlin":
        return
    msg = (
        f"Engine quantization {quantization!r} with LoRA needs "
        f"moe_backend='marlin' on SM{cc_major}{cc_minor} (A100 and older have "
        "no native FP8; vLLM's Triton FP8 MoE kernels need SM89+)."
    )
    raise ValueError(msg)


def _max_num_seqs(
    manifest: TrainingManifest,
    spec: LLMAlgorithmSpec,
    vllm: VLLMConfig,
    rollout_engines: int | None,
) -> int:
    """The manifest's ``max_num_seqs``, else sequences one engine decodes.

    Without ``rollout_engines``, ``auto`` counts as one engine.
    """
    training = manifest.training
    prompts = 0
    if vllm.max_num_seqs is None and training.async_rollout:
        engines = training.rollout_engines_per_agent
        if rollout_engines is not None:
            n_engines = rollout_engines
        else:
            n_engines = 1 if engines == "auto" else int(engines)
        # Async validation sets an unset rollout_batch_size to 1.
        rollout_batch_size = int(training.rollout_batch_size or 1)
        prompts = prompts_per_engine(rollout_batch_size, n_engines)
    return resolve_max_num_seqs(
        vllm.max_num_seqs,
        prompts,
        rollout_group_size(spec),
        not training.async_rollout,
    )


def _max_num_batched_tokens(spec: LLMAlgorithmSpec, vllm: VLLMConfig) -> int | None:
    """The manifest's value, else the model family's catalog default."""
    if vllm.max_num_batched_tokens is not None:
        return vllm.max_num_batched_tokens
    info = SUPPORTED_MODEL_INFO.get(spec.pretrained_model_name_or_path)
    if info is None:
        return None
    return FAMILY_MAX_NUM_BATCHED_TOKENS.get(info.config["model_type"])


def _resolve_vllm(spec: LLMAlgorithmSpec) -> VLLMConfig:
    """The engine config the run starts, checked against what this estimator sizes."""
    # SFT and DPO start no engine; their generation bar is ignored.
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return VLLMConfig()
    vllm = spec.vllm_config
    model = vllm.vllm_model_name_or_path
    if model is not None and model != spec.pretrained_model_name_or_path:
        msg = (
            f"vllm_model_name_or_path ({model!r}) serves a different checkpoint "
            "than the trainer fine-tunes "
            f"({spec.pretrained_model_name_or_path!r}); this estimator sizes one arch."
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


def resolve_micro_batch_rows(spec: LLMAlgorithmSpec, n_training_gpus: int) -> int:
    """Rows in one training forward, then the divisibility checks.

    :param spec: Validated LLM algorithm spec.
    :param n_training_gpus: Trainer ranks, ``training.training_gpus_per_agent``.
    :return: Rows in one forward.
    :raises ValueError: When the mini-batch and micro-batch do not divide.
    """
    micro = resolve_micro_batch_size_per_gpu(spec, n_training_gpus)
    _, per_rank = per_rank_trajectories(spec, n_training_gpus)
    mini = resolve_mini_batch_size(spec.mini_batch_size, per_rank)
    require_batch_split(mini, int(micro), per_rank)
    return int(micro)


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


def image_inputs(manifest: TrainingManifest) -> tuple[int, int, int]:
    """Images per completion row and the size the environment renders them at.

    :param manifest: Validated manifest.
    :return: ``(images_per_row, width, height)``; zeros without image fields.
    """
    env = manifest.environment
    if not isinstance(env, LLMEnvSpec) or not (
        env.screenshot_field or env.goal_images_field
    ):
        return 0, 0, 0
    # BrowserGym captures at the viewport the OpenEnv server reads from these.
    width = int(env.env_vars.get("BROWSERGYM_VIEWPORT_WIDTH", 1280))
    height = int(env.env_vars.get("BROWSERGYM_VIEWPORT_HEIGHT", 720))
    spec = llm_spec(manifest)
    engine_limit = None
    if isinstance(spec, RolloutLLMSpec) and spec.vllm_config is not None:
        engine_limit = (spec.vllm_config.limit_mm_per_prompt or {}).get("image")
    return env.segment_max_images or engine_limit or 1, width, height


def training_settings_from_spec(
    spec: LLMAlgorithmSpec,
    n_training_gpus: int,
    micro_batch_size: int,
    checkpoint_optimizer: bool = False,
    async_rollout: bool = False,
    images: tuple[int, int, int] = (0, 0, 0),
    segmented_rows: bool = False,
) -> TrainingSettings:
    """Trainer-side settings for one algorithm spec.

    :param spec: Validated LLM algorithm spec.
    :param n_training_gpus: Trainer ranks.
    :param micro_batch_size: Rows in one forward.
    :param checkpoint_optimizer: The checkpoint snapshot holds Adam moments.
    :param async_rollout: The next rollout batch stays in host memory during learn.
    :param images: ``(images_per_row, width, height)`` from :func:`image_inputs`.
    :param segmented_rows: Episodes train as segment rows.
    :return: Training settings.
    :raises ValueError: Quantization, eager attention, or unbounded Liger loss.
    """
    _reject_trainer_quantization(spec.quantization)
    _reject_eager_attn_implementation(spec.attn_implementation)
    _reject_unbounded_liger_loss(spec)
    lora = _lora_config(spec)
    images_per_row, image_width, image_height = images
    support_width = 0
    if isinstance(spec, RolloutLLMSpec) and records_sampling_supports(
        spec.temperature,
        spec.top_k,
        spec.top_p,
        spec.min_p,
        spec.constrain_answer_pattern,
    ):
        # Support frames are padded to at least two ids.
        support_width = max(spec.top_k, 2)
    return TrainingSettings(
        algorithm=ALGORITHM_NAMES[spec.name],
        group_size=rollout_group_size(spec),
        trajectories_per_update=spec.batch_size * rollout_group_size(spec),
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
        sampling_support_width=support_width,
        support_scored_loss=not (
            isinstance(spec, (LLMPPOSpec, LLMREINFORCESpec)) and spec.use_liger_loss
        ),
        n_training_gpus=n_training_gpus,
        fsdp=spec.fsdp,
        fuse_actor_critic_pass=(
            resolve_fuse_actor_critic_pass(spec.fuse_actor_critic_pass, None)
            if isinstance(spec, LLMPPOSpec)
            else False
        ),
        checkpoint_optimizer=checkpoint_optimizer,
        async_rollout=async_rollout,
        micro_batch_size=micro_batch_size,
        sequence_packing=spec.use_sequence_packing,
        images_per_row=images_per_row,
        image_width=image_width,
        image_height=image_height,
        vision_tower_lora=bool(lora.tower_target_modules),
        segmented_rows=segmented_rows,
    )


def training_settings_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
    micro_batch_size: int | None = None,
) -> TrainingSettings:
    """Trainer-side settings, read as the trainer reads the manifest.

    Settings the estimator cannot size (quantization, the eager backend,
    Liger above token level) raise ``ValueError``.

    :param manifest: Training manifest.
    :param micro_batch_size: Rows in one forward. ``None`` resolves an omitted
        field without a device.
    :return: Training settings.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    n_training_gpus = manifest.training.training_gpus_per_agent
    _reject_trainer_quantization(spec.quantization)
    if micro_batch_size is None:
        micro_batch_size = resolve_micro_batch_rows(spec, n_training_gpus)
    _reject_eager_attn_implementation(spec.attn_implementation)
    _reject_unbounded_liger_loss(spec)
    env = manifest.environment
    return training_settings_from_spec(
        spec,
        n_training_gpus,
        micro_batch_size,
        checkpoint_optimizer=manifest.training.checkpoint_optimizer,
        async_rollout=manifest.training.async_rollout,
        images=image_inputs(manifest),
        segmented_rows=isinstance(env, LLMEnvSpec)
        and (
            env.segment_prompt_tokens is not None or env.segment_max_images is not None
        ),
    )


def generation_settings_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
    tensor_parallel_size: int | None = None,
    rollout_engines: int | None = None,
) -> GenerationSettings:
    """The engine-side settings the manifest's vLLM config pins down.

    :param manifest: Training manifest.
    :param tensor_parallel_size: Width to size. Required when a rollout
        manifest leaves the width unset.
    :param rollout_engines: Engines per member the allocation starts, which
        split an omitted ``max_num_seqs``. ``None`` reads the manifest.
    :return: Generation settings for the estimator.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    vllm = _resolve_vllm(spec)
    lora = _lora_config(spec)
    stripped = _stripped_towers(vllm)
    _reject_kv_cache_quantization(vllm.kv_cache_dtype)
    weight_dtype = _weight_dtype(vllm.dtype)
    weight_quantization = _engine_weight_quantization(vllm.quantization)
    width = (
        tensor_parallel_size
        if tensor_parallel_size is not None
        else vllm.tensor_parallel_size
    )
    if width is None and isinstance(spec, RolloutLLMSpec):
        msg = "tensor_parallel_size is unset; pass the width to size."
        raise ValueError(msg)
    settings: dict[str, Any] = {
        "gpu_memory_utilization": vllm.gpu_memory_utilization,
        "max_num_seqs": _max_num_seqs(manifest, spec, vllm, rollout_engines),
        "max_model_len": spec.max_model_len,
        "max_num_batched_tokens": _max_num_batched_tokens(spec, vllm),
        "max_images_per_prompt": (vllm.limit_mm_per_prompt or {}).get("image"),
        "kv_cache_memory_bytes": vllm.kv_cache_memory_bytes,
        "enforce_eager": vllm.enforce_eager,
        "max_lora_rank": (
            resolve_max_lora_rank(vllm.max_lora_rank, lora.lora_r)
            if manifest.training.async_rollout
            else resolve_lora_rank_ceiling(vllm.max_lora_rank, lora.lora_r)
        ),
        "max_loras": vllm.max_loras,
        "lora_packed_target_matrices": len(lora.target_parameters or []),
        "weight_dtype": weight_dtype,
        "weight_quantization": weight_quantization,
        "weight_variant": "engine" if stripped else "base",
        "concurrent_requests": spec.batch_size * rollout_group_size(spec),
    }
    if width is not None:
        settings["tensor_parallel_size"] = width
    return GenerationSettings(**settings)


def _model_spec(
    spec: LLMAlgorithmSpec,
    model_config: dict[str, Any],
    n_params: int | None,
) -> ModelSpec:
    model_id = spec.pretrained_model_name_or_path
    if model_id is None:
        msg = "manifest names no pretrained model."
        raise ValueError(msg)
    return ModelSpec(
        model_id=model_id,
        arch=ModelArch.from_hf_config(model_config),
        n_params=n_params,
        variants=MANIFEST_WEIGHT_VARIANTS,
    )


def run_config_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
    device: DeviceSpec,
    model_config: dict[str, Any],
    *,
    n_params: int | None = None,
    gen_device: DeviceSpec | None = None,
    tensor_parallel_size: int | None = None,
    rollout_engines: int | None = None,
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
    :param tensor_parallel_size: Engine width to size. Required when a rollout
        manifest leaves the width unset.
    :param rollout_engines: Engines per member the allocation starts.
        ``None`` reads the manifest, counting ``auto`` as one.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    if isinstance(spec, RolloutLLMSpec):
        gpu = lookup_gpu((gen_device or device).name or "")
        if gpu is not None:
            vllm = _resolve_vllm(spec)
            check_fp8_moe_backend_for_device(
                vllm.quantization,
                spec.vllm_engine_args.get("moe_backend"),
                ModelArch.from_hf_config(model_config).is_moe,
                gpu.cc_major,
                gpu.cc_minor,
            )
    # Chosen on a copy so this call does not pin the caller's omitted field.
    # The run picks it on the catalog model, so the search does too.
    if spec.micro_batch_size_per_gpu is None:
        manifest = manifest.model_copy(deep=True)
        spec = llm_spec(manifest)
        n_training_gpus = manifest.training.training_gpus_per_agent
        base = training_settings_from_manifest(manifest, micro_batch_size=1)
        micro = resolve_micro_batch_size_per_gpu(
            spec, n_training_gpus, device, settings=base
        )
        _, per_rank = per_rank_trajectories(spec, n_training_gpus)
        spec.micro_batch_size_per_gpu = micro
        spec.mini_batch_size = resolve_mini_batch_size(spec.mini_batch_size, per_rank)
    training = training_settings_from_manifest(manifest)
    if training.uses_generation_engine and not manifest.training.async_rollout:
        msg = (
            f"rollout_mode={manifest.training.rollout_mode!r} is not sized; "
            "this estimator sizes async rollout only."
        )
        raise ValueError(msg)
    generation = generation_settings_from_manifest(
        manifest,
        tensor_parallel_size=tensor_parallel_size,
        rollout_engines=rollout_engines,
    )
    model = _model_spec(spec, model_config, n_params)
    training = training_settings_with_fuse(
        spec.fuse_actor_critic_pass if isinstance(spec, LLMPPOSpec) else None,
        training,
        model,
        device,
    )
    return RunConfig(
        model=model,
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
    tensor_parallel_size: int | None = None,
) -> RunEstimate:
    """Estimate both phase peaks straight from the manifest.

    :param tensor_parallel_size: Engine width to size. Required when a rollout
        manifest leaves the width unset.
    """
    return estimate_run(
        run_config_from_manifest(
            manifest,
            device,
            model_config,
            n_params=n_params,
            gen_device=gen_device,
            tensor_parallel_size=tensor_parallel_size,
        )
    )
