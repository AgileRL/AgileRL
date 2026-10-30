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

from agilerl.arena.memory.estimator import RunEstimate, estimate_run, training_breakdown
from agilerl.arena.memory.resolve.attention import resolve_attn_implementation
from agilerl.arena.memory.resolve.batch import (
    engine_group_size,
    per_rank_trajectories,
    require_batch_split,
    resolve_micro_batch_size_per_gpu,
    resolve_mini_batch_size,
    rollout_group_size,
)
from agilerl.arena.memory.resolve.checkpointing import resolve_gradient_checkpointing
from agilerl.arena.memory.resolve.fsdp import cpu_offload_blocker, resolve_fsdp_memory
from agilerl.arena.memory.resolve.liger import liger_head_grad, liger_loss_layout
from agilerl.arena.memory.resolve.lora import (
    resolve_max_lora_rank,
)
from agilerl.arena.memory.resolve.packing import (
    leaks_recurrent_state,
    sequence_packing_blocker,
)
from agilerl.arena.memory.resolve.rollout import (
    apply_run_rollout_mode,
    async_rollout_algorithm,
    run_rollout_engines,
)
from agilerl.arena.memory.resolve.sequences import (
    prompts_per_engine,
    resolve_max_num_seqs,
)
from agilerl.arena.memory.resolve.training import (
    lora_target_scope,
    packed_moe_dispatch,
    sampling_support_width,
    support_scored_loss,
)
from agilerl.arena.memory.specs import (
    ALGORITHM_NAMES,
    DeviceSpec,
    GenerationSettings,
    GiB,
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
from agilerl.arena.models.manifest import TrainingManifest, strips_multimodal_towers
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.networks import LoraConfigDict, VLLMConfig

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
# measured on H100, A100-80 and L4, nominal on the rest.
GPU_CATALOGUE: tuple[tuple[str, GpuInfo], ...] = (
    ("H200", GpuInfo(name="NVIDIA H200", total_gib=141, cc_major=9, cc_minor=0)),
    (
        "H100",
        GpuInfo(name="NVIDIA H100 80GB HBM3", total_gib=79.11, cc_major=9, cc_minor=0),
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
    # Sizes resolved manifests too, so the memory fields a run picks are allowed.
    return TrainingManifest.model_validate(TrainingManifest.load(manifest))


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


def trainer_packs_sequences(spec: LLMAlgorithmSpec) -> bool:
    """Whether an Arena trainer packs this spec's gradient forward.

    Arena trainers train on CUDA. A model outside :data:`SUPPORTED_MODEL_INFO`
    resolves with no family backend and no recurrent layers.

    :param spec: Validated LLM algorithm spec.
    :return: ``True`` when :func:`sequence_packing_blocker` passes.
    """
    entry = SUPPORTED_MODEL_INFO.get(spec.pretrained_model_name_or_path or "")
    model_type = None if entry is None else entry.config["model_type"]
    has_recurrent_layers = (
        entry is not None and ModelArch.from_hf_config(entry.config).is_hybrid_ssm
    )
    attn_implementation = resolve_attn_implementation(model_type, cuda=True)
    blocker = sequence_packing_blocker(
        spec.supports_sequence_packing,
        attn_implementation,
        spec.gradient_row_copies,
        leaks_recurrent_state(model_type, has_recurrent_layers),
    )
    return blocker is None


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


def async_rollout(manifest: TrainingManifest) -> bool:
    """Whether the run generates on async rollout engines, as :func:`async_rollout_algorithm` decides."""
    return async_rollout_algorithm(type(manifest.algorithm))


def explicit_colocated_rollout(manifest: TrainingManifest) -> bool:
    """Whether a rollout LLM manifest names ``rollout_mode: colocated``, which only local runs take."""
    return (
        async_rollout(manifest)
        and "rollout_mode" in manifest.training.model_fields_set
        and manifest.training.rollout_mode == "colocated"
    )


def configured_rollout_engines(manifest: TrainingManifest) -> int | Literal["auto"]:
    """Rollout engines per member as Ray reads the manifest, from :func:`run_rollout_engines`."""
    return run_rollout_engines(
        async_rollout(manifest), manifest.training.rollout_engines_per_agent
    )


def validated_run_manifest(manifest: TrainingManifest) -> TrainingManifest:
    """The manifest validated as Ray runs it, from :func:`apply_run_rollout_mode`.

    :param manifest: Training manifest, validated as written.
    :return: ``manifest`` itself when Ray reads it as written, else the run Ray starts.
    :raises pydantic.ValidationError: When the run cannot start as Ray runs it.
    """
    run_async = async_rollout(manifest)
    if not run_async and "rollout_mode" not in manifest.training.model_fields_set:
        return manifest
    payload = manifest.to_payload()
    apply_run_rollout_mode(payload["training"], run_async)
    return TrainingManifest.model_validate(payload)


def _async_engine_prompts(
    manifest: TrainingManifest, rollout_engines: int | None
) -> int:
    """Prompts one async engine receives from a job batch.

    ``rollout_engines`` is the engines per member the allocation starts;
    ``None`` reads the manifest. Without ``rollout_engines``, ``auto``
    counts as one engine.
    """
    engines = configured_rollout_engines(manifest)
    if rollout_engines is not None:
        n_engines = rollout_engines
    else:
        n_engines = 1 if engines == "auto" else engines
    # Async validation sets an unset rollout_batch_size to 1.
    rollout_batch_size = int(manifest.training.rollout_batch_size or 1)
    return prompts_per_engine(rollout_batch_size, n_engines)


MARLIN_MOE_TILE = 64
"""Marlin MoE pads a per-GPU expert width that is not a multiple of this."""


def marlin_pads_non_gated_experts(
    quantization: str | None,
    moe_backend: str | None,
    arch: ModelArch,
    tensor_parallel_size: int,
    expert_parallel: bool,
) -> bool:
    """Whether vLLM's Marlin FP8 MoE prep would fail on this engine's experts.

    vLLM 0.32 pads a misaligned Marlin expert width as if ``w13`` held gate
    and up halves, so non-gated (relu2) experts fail the reshape at load.
    Engine expert parallel keeps each expert whole; TP splits its width.

    :param quantization: Engine ``quantization``.
    :param moe_backend: Explicit ``moe_backend`` engine arg, or None.
    :param arch: Served model architecture.
    :param tensor_parallel_size: Engine tensor-parallel width.
    :param expert_parallel: Whether the engine sets ``enable_expert_parallel``.
    :return: True when the engine must not serve FP8 on Marlin.
    """
    backend = moe_backend if isinstance(moe_backend, str) else ""
    if (
        not arch.is_moe
        or arch.gated_mlp
        or quantization is None
        or quantization.lower() not in FP8_MOE_METHODS
        or backend.strip().lower().replace("-", "_") != "marlin"
    ):
        return False
    width = arch.expert_intermediate_size or arch.intermediate_size
    if not expert_parallel:
        width //= tensor_parallel_size
    return width % MARLIN_MOE_TILE != 0


def _max_num_seqs(
    manifest: TrainingManifest,
    spec: LLMAlgorithmSpec,
    vllm: VLLMConfig,
    rollout_engines: int | None,
) -> int:
    """The manifest's ``max_num_seqs``, else sequences one engine decodes.

    Async engines split the job batch (:func:`_async_engine_prompts`). Each
    prompt holds the group size HPO can reach.
    """
    is_async = async_rollout(manifest)
    prompts = 0
    if vllm.max_num_seqs is None and is_async:
        prompts = _async_engine_prompts(manifest, rollout_engines)
    return resolve_max_num_seqs(
        vllm.max_num_seqs,
        prompts,
        engine_group_size(spec, manifest.mutation),
        not is_async,
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


def _liger_loss_layout(
    spec: LLMAlgorithmSpec,
    records_supports: bool,
    multi_turn: bool,
) -> tuple[bool, bool]:
    """Whether the update runs the fused Liger loss, and whether it chunks whole rows.

    PPO's ``auto`` level pools at the GAE granularity under ``turn_level_clip``:
    ``turn`` on multi-turn batches, else ``token``. Every rollout algorithm's
    update weights tokens by the engines' sampling log-probs, so above token
    level it runs the standard loss.

    :param spec: Validated LLM algorithm spec.
    :param records_supports: Rollouts record vLLM sampling supports.
    :param multi_turn: Episodes can take more than one turn.
    :return: ``(liger_loss, whole_sequence_loss)``.
    """
    algorithm = (
        spec.loss_type if isinstance(spec, GRPOSpec) else ALGORITHM_NAMES[spec.name]
    )
    level = None
    if isinstance(spec, (GRPOSpec, LLMPPOSpec, LLMREINFORCESpec)):
        level = spec.importance_sampling_level
    if isinstance(spec, LLMPPOSpec) and level == "auto":
        turn = spec.advantage_granularity == "turn" or (
            spec.advantage_granularity == "auto" and multi_turn
        )
        level = "turn" if spec.turn_level_clip and turn else "token"
    hub_id = spec.pretrained_model_name_or_path
    model = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    return liger_loss_layout(
        algorithm,
        level,
        records_supports,
        isinstance(spec, GRPOSpec) and spec.use_kl_advantage_shaping,
        None if model is None else model.text_config.get("final_logit_softcapping"),
        isinstance(spec, RolloutLLMSpec),
        None,
    )


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
    multi_turn: bool = True,
) -> TrainingSettings:
    """Trainer-side settings for one algorithm spec.

    :param spec: Validated LLM algorithm spec.
    :param n_training_gpus: Trainer ranks.
    :param micro_batch_size: Rows in one forward.
    :param checkpoint_optimizer: The checkpoint snapshot holds Adam moments.
    :param async_rollout: The next rollout batch stays in host memory during learn.
    :param images: ``(images_per_row, width, height)`` from :func:`image_inputs`.
    :param segmented_rows: Episodes train as segment rows.
    :param multi_turn: Episodes can take more than one turn.
    :return: Training settings.
    :raises ValueError: Quantization.
    """
    _reject_trainer_quantization(spec.quantization)
    lora = _lora_config(spec)
    rollout = isinstance(spec, RolloutLLMSpec)
    group_size = rollout_group_size(spec)
    algorithm = ALGORITHM_NAMES[spec.name]
    records_supports = rollout and records_sampling_supports(
        spec.temperature,
        spec.top_k,
        spec.top_p,
        spec.min_p,
        spec.constrain_answer_pattern,
    )
    images_per_row, image_width, image_height = images
    liger_loss, whole_sequence_loss = _liger_loss_layout(
        spec, records_supports, multi_turn
    )
    return TrainingSettings(
        algorithm=algorithm,
        group_size=group_size,
        trajectories_per_update=spec.batch_size * group_size,
        max_model_len=(
            spec.max_row_tokens
            if rollout and spec.max_row_tokens is not None
            else spec.max_model_len
        ),
        micro_batch_size=micro_batch_size,
        lora_rank=lora.lora_r,
        lora_target_scope=lora_target_scope(lora.target_modules),
        lora_dropout=lora.lora_dropout,
        lora_packed_target_matrices=len(set(lora.target_parameters or ())),
        # Packed-expert adapters are sized on the split path.
        packed_moe_dispatch=packed_moe_dispatch(lora.target_parameters, False),
        beta=spec.beta,
        use_separate_reference_adapter=spec.use_separate_reference_adapter,
        weight_dtype="bf16",
        activation_offload=spec.fsdp is not None and bool(spec.fsdp.activation_offload),
        gradient_checkpointing=spec.gradient_checkpointing is not False,
        sampling_support_width=sampling_support_width(
            records_supports, spec.top_k if rollout else 0
        ),
        support_scored_loss=support_scored_loss(algorithm, liger_loss),
        whole_sequence_loss=whole_sequence_loss,
        liger_weight_grad=liger_head_grad(algorithm, liger_loss, whole_sequence_loss),
        n_training_gpus=n_training_gpus,
        fsdp=spec.fsdp,
        sequence_packing=trainer_packs_sequences(spec),
        images_per_row=images_per_row,
        image_width=image_width,
        image_height=image_height,
        vision_tower_lora=bool(lora.tower_target_modules),
        segmented_rows=segmented_rows,
        checkpoint_optimizer=checkpoint_optimizer,
        async_rollout=async_rollout,
    )


def training_settings_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
    micro_batch_size: int | None = None,
) -> TrainingSettings:
    """Trainer-side settings, read as the trainer reads the manifest.

    Settings the estimator cannot size (quantization) and settings the run cannot start (full FSDP CPU offload beside async rollout
    engines) raise ``ValueError``.

    :param manifest: Training manifest.
    :param micro_batch_size: Rows in one forward. ``None`` resolves an omitted
        field without a device.
    :return: Training settings.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    n_training_gpus = manifest.training.training_gpus_per_agent
    _reject_trainer_quantization(spec.quantization)
    blocker = cpu_offload_blocker(spec.fsdp, async_rollout(manifest))
    if blocker is not None:
        raise ValueError(blocker)
    if micro_batch_size is None:
        micro_batch_size = resolve_micro_batch_rows(spec, n_training_gpus)
    env = manifest.environment
    return training_settings_from_spec(
        spec,
        n_training_gpus,
        micro_batch_size,
        checkpoint_optimizer=manifest.training.checkpoint_optimizer,
        async_rollout=async_rollout(manifest),
        images=image_inputs(manifest),
        segmented_rows=isinstance(env, LLMEnvSpec)
        and (
            env.segment_prompt_tokens is not None or env.segment_max_images is not None
        ),
        multi_turn=not isinstance(env, LLMEnvSpec) or env.max_turns != 1,
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
        split the job batch. ``None`` reads the manifest.
    :return: Generation settings for the estimator.
    """
    manifest = _validated(manifest)
    spec = llm_spec(manifest)
    vllm = _resolve_vllm(spec)
    lora = _lora_config(spec)
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
        "max_lora_rank": resolve_max_lora_rank(lora.lora_r),
        "lora_packed_target_matrices": len(lora.target_parameters or []),
        "weight_dtype": weight_dtype,
        "weight_quantization": weight_quantization,
        "weight_variant": (
            "engine"
            if strips_multimodal_towers(
                spec,
                isinstance(manifest.environment, LLMEnvSpec)
                and manifest.environment.text_only,
            )
            else "base"
        ),
        "concurrent_requests": (
            _async_engine_prompts(manifest, rollout_engines)
            if async_rollout(manifest)
            else spec.batch_size
        )
        * engine_group_size(spec, manifest.mutation),
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


def resolve_training_memory(
    manifest: TrainingManifest, device: DeviceSpec, model: ModelSpec
) -> TrainingManifest:
    """Copy of ``manifest`` with unset trainer memory fields at the fastest values that fit.

    The micro-batch comes first, searched on the catalog model as the run
    does. The FSDP memory fields are priced at that micro-batch, then
    gradient checkpointing at both. Set values are kept.

    :param manifest: Validated training manifest.
    :param device: Training GPU.
    :param model: Training-side model spec the FSDP fields and checkpointing
        are priced on.
    :return: A resolved copy; ``manifest`` is not modified.
    """
    manifest = manifest.model_copy(deep=True)
    spec = llm_spec(manifest)
    if spec.micro_batch_size_per_gpu is None:
        spec.micro_batch_size_per_gpu = resolve_micro_batch_size_per_gpu(
            spec,
            manifest.training.training_gpus_per_agent,
            device,
            settings=training_settings_from_manifest(manifest, micro_batch_size=1),
        )
    if spec.fsdp is not None:
        spec.fsdp = resolve_fsdp_memory(
            model,
            device,
            training_settings_from_manifest(manifest),
            training_breakdown,
            orchestrated=True,
            micro_batch_floor=spec.micro_batch_size_per_gpu,
        )
    if spec.gradient_checkpointing is None:
        spec.gradient_checkpointing = resolve_gradient_checkpointing(
            model,
            device,
            training_settings_from_manifest(manifest),
            training_breakdown,
            orchestrated=True,
        )
    return manifest


def _engine_served_manifest(
    manifest: TrainingManifest,
    model_config: dict[str, Any],
    tensor_parallel_size: int | None,
) -> TrainingManifest:
    """The manifest as the rollout engine serves it.

    An fp8 Marlin engine on misaligned non-gated experts serves bf16 on Triton
    (:func:`marlin_pads_non_gated_experts`); a copy carries that change.
    """
    spec = llm_spec(manifest)
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return manifest
    width = tensor_parallel_size or spec.vllm_config.tensor_parallel_size
    if width is None or not marlin_pads_non_gated_experts(
        spec.vllm_config.quantization,
        spec.vllm_engine_args.get("moe_backend"),
        ModelArch.from_hf_config(model_config),
        width,
        bool(spec.vllm_engine_args.get("enable_expert_parallel")),
    ):
        return manifest
    served = manifest.model_copy(deep=True)
    served_spec = llm_spec(served)
    assert isinstance(served_spec, RolloutLLMSpec)
    assert served_spec.vllm_config is not None
    served_spec.vllm_config.quantization = None
    served_spec.vllm_engine_args["moe_backend"] = "triton"
    return served


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
        ``None`` reads the manifest, which pins the count.
    """
    manifest = _validated(manifest)
    if explicit_colocated_rollout(manifest):
        msg = (
            "rollout_mode='colocated' is not sized; "
            "this estimator sizes async rollout only."
        )
        raise ValueError(msg)
    manifest = _engine_served_manifest(manifest, model_config, tensor_parallel_size)
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
    model = _model_spec(spec, model_config, n_params)
    micro_batch_unset = spec.micro_batch_size_per_gpu is None
    manifest = resolve_training_memory(manifest, device, model)
    spec = llm_spec(manifest)
    if micro_batch_unset:
        _, per_rank = per_rank_trajectories(
            spec, manifest.training.training_gpus_per_agent
        )
        spec.mini_batch_size = resolve_mini_batch_size(spec.mini_batch_size, per_rank)
    training = training_settings_from_manifest(manifest)
    generation = generation_settings_from_manifest(
        manifest,
        tensor_parallel_size=tensor_parallel_size,
        rollout_engines=rollout_engines,
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
