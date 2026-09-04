# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Build a sizing question from the training manifest and a resource class.

The estimator's own inputs (:class:`~agilerl.arena.memory.specs.RunConfig`)
are the *derived* working form; nothing outside this module should have to
assemble them by hand. A caller holds exactly three things:

- the training manifest — the single source of truth for every setting,
- the resource class the run would be scheduled on (GPU type and count),
- the checkpoint's ``config.json``, fetched for the model the manifest names.

This module turns those into a :class:`RunConfig`. Every setting below is read
from the validated manifest — the same object the submission payload is built
from — so the gate can never disagree with the run it is gating.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast, get_args

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.specs import (
    Algorithm,
    AttnImplementation,
    DeviceSpec,
    DistributedBackend,
    GenerationKnobs,
    GiB,
    KVCacheDtype,
    LoraTargetScope,
    ModelArch,
    ModelSpec,
    QuantizationMethod,
    RunConfig,
    TrainerQuantization,
    TrainingKnobs,
    WeightDtype,
    WeightVariant,
)
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.networks import LoraConfigDict, VLLMConfig

#: Manifest algorithm names -> the estimator's algorithm identifiers. The
#: estimator keys its reference/critic residency rules on these, so an
#: unmapped LLM algorithm is a hard error rather than a silent "grpo".
ALGORITHM_NAMES: dict[str, Algorithm] = {
    "GRPO": "grpo",
    "GSPO": "gspo",
    "CISPO": "cispo",
    "LLMPPO": "ppo",
    "LLMREINFORCE": "reinforce",
    "DPO": "dpo",
    "SFT": "sft",
}

#: Module names that mean an adapter touches attention projections only.
ATTENTION_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query", "key", "value"}
)


class GpuInfo(BaseModel):
    """One accelerator model a resource class can provide."""

    model_config = ConfigDict(frozen=True)

    #: Canonical CUDA device name, matching ``torch.cuda.get_device_name`` so
    #: the measured per-device constants (CUDA context) resolve.
    name: str
    total_gib: float
    cc_major: int
    cc_minor: int

    def device_spec(self) -> DeviceSpec:
        return DeviceSpec.from_compute_capability(
            total_bytes=int(self.total_gib * GiB),
            major=self.cc_major,
            minor=self.cc_minor,
            name=self.name,
        )


#: The accelerators Arena resource classes are built from, in match order —
#: longer tokens first, because "A100" contains "A10" and "L40S" contains
#: "L4". Matched against the tier's ``gpu_type`` after normalisation.
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
    """Resolve a resource class's ``gpu_type`` string to a known accelerator.

    Tier strings are free-form ("NVIDIA L4", "A100-SXM4-40GB", "a100 80gb"),
    so matching is by token rather than equality: vendor words and separators
    are stripped, then the catalogue is scanned longest token first. The
    "A100-80" entry matches any A100 string that also carries an "80".
    """
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
    """The device one GPU of an Arena resource class provides.

    ``resource_class`` is a tier entry as the Arena API returns it
    (``client.list_resources()``): ``gpu_type`` names the accelerator and
    ``num_gpus`` how many the node carries. The estimate is per GPU — the
    model covers a single device, and multi-GPU placement (DP shards, TP,
    Ray overhead) is a stated gap — so ``num_gpus`` does not scale anything
    here.

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


def _llm_spec(manifest: TrainingManifest) -> LLMAlgorithmSpec:
    spec = manifest.algorithm
    if not isinstance(spec, LLMAlgorithmSpec):
        msg = (
            f"Memory estimation covers the LLM fine-tuning algorithms; "
            f"{spec.name} builds its own (small) networks and does not need a gate."
        )
        raise ValueError(msg)
    if spec.name not in ALGORITHM_NAMES:
        msg = f"No memory model for LLM algorithm {spec.name!r}."
        raise ValueError(msg)
    return spec


def _lora_config(spec: LLMAlgorithmSpec) -> LoraConfigDict:
    # The framework trains LoRA adapters only; a manifest that omits the
    # section runs the constructor defaults, which LoraConfigDict carries.
    return spec.lora_config or LoraConfigDict()


def _lora_target_scope(lora: LoraConfigDict) -> LoraTargetScope:
    targets = lora.target_modules
    if isinstance(targets, str):
        return "all-linear"
    return "attention-only" if set(targets) <= ATTENTION_MODULES else "all-linear"


def _trainer_quantization(
    quantization: str | dict[str, Any] | None,
) -> TrainerQuantization:
    """bitsandbytes preset or config -> what the trainer copy is stored as."""
    if isinstance(quantization, dict):
        if quantization.get("load_in_4bit"):
            return "nf4"
        if quantization.get("load_in_8bit"):
            return "int8"
        return "none"
    preset = (quantization or "none").lower()
    if preset in {"nf4", "4bit"}:
        return "nf4"
    if preset in {"int8", "8bit"}:
        return "int8"
    return "none"


def _attn_implementation(value: str | None) -> AttnImplementation:
    # Unset lets the trainer resolve the backend; the estimator's "auto"
    # mirrors that resolution rule.
    if value in get_args(AttnImplementation):
        return cast("AttnImplementation", value)
    return "auto"


def _weight_dtype(value: str | None) -> WeightDtype:
    lowered = (value or "").lower()
    if lowered in {"float16", "fp16", "half"}:
        return "fp16"
    if lowered in {"float32", "fp32"}:
        return "fp32"
    return "bf16"


def _kv_cache_dtype(value: str | None) -> KVCacheDtype:
    lowered = (value or "auto").lower()
    if lowered.startswith("fp8"):
        return "fp8"
    if lowered == "int8":
        return "int8"
    return "auto"


def _engine_quantization(value: str | None) -> QuantizationMethod:
    lowered = (value or "none").lower()
    if lowered in {"awq", "awq_marlin"}:
        return "awq"
    if lowered in {"gptq", "gptq_marlin"}:
        return "gptq"
    if lowered == "bitsandbytes":
        return "nf4"
    if lowered == "int8":
        return "int8"
    return "none"


def _group_size(spec: LLMAlgorithmSpec) -> int:
    return int(getattr(spec, "group_size", 1) or 1)


def training_knobs_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
) -> TrainingKnobs:
    """The trainer-side settings a validated manifest pins down.

    Field for field this mirrors how the trainer reads the same manifest:
    ``batch_size`` is prompts per update, the update carries
    ``batch_size x group_size`` completion rows, and the gradient micro-batch
    falls back to ``batch_size`` rows exactly as ``LLMAlgorithm`` does.
    """
    manifest = _validated(manifest)
    spec = _llm_spec(manifest)
    lora = _lora_config(spec)
    distributed: DistributedBackend = (
        "deepspeed"
        if spec.deepspeed is not None or manifest.training.training_gpus_per_agent > 1
        else "none"
    )
    return TrainingKnobs(
        algorithm=ALGORITHM_NAMES[spec.name],
        batch_size=spec.batch_size,
        micro_batch_size_per_gpu=spec.micro_batch_size_per_gpu,
        group_size=_group_size(spec),
        trajectories_per_update=spec.batch_size * _group_size(spec),
        max_model_len=spec.max_model_len,
        lora_rank=lora.lora_r,
        lora_target_scope=_lora_target_scope(lora),
        lora_packed_target_matrices=len(lora.target_parameters or []),
        beta=spec.beta,
        use_separate_reference_adapter=spec.use_separate_reference_adapter,
        quantization=_trainer_quantization(spec.quantization),
        attn_implementation=_attn_implementation(spec.attn_implementation),
        gradient_checkpointing=spec.gradient_checkpointing,
        activation_offload=spec.activation_offload,
        chunk_rows=spec.chunk_rows,
        distributed=distributed,
        n_training_gpus=manifest.training.training_gpus_per_agent,
        zero_stage=3 if spec.zero_stage == 3 else 2,
    )


def generation_knobs_from_manifest(
    manifest: TrainingManifest | str | Path | dict[str, Any],
) -> GenerationKnobs:
    """The engine-side settings the manifest's (resolved) vLLM config pins down.

    Manifest validation already resolves the engine config a colocated run
    gets when the document names none, so what this reads is exactly what the
    trainer would construct. Algorithms that never generate (SFT, DPO) size
    against the same defaults — their generation bar is the engine the run
    *would* start, which for them is none, so the conservative default keeps
    the gate simple rather than special-casing the phase away.
    """
    manifest = _validated(manifest)
    spec = _llm_spec(manifest)
    vllm: VLLMConfig = getattr(spec, "vllm_config", None) or VLLMConfig()
    lora = _lora_config(spec)
    quantization = _engine_quantization(vllm.quantization)
    stripped = bool(vllm.strip_multimodal_towers)
    return GenerationKnobs(
        gpu_memory_utilization=vllm.gpu_memory_utilization,
        max_num_seqs=vllm.max_num_seqs,
        max_model_len=spec.max_model_len,
        max_num_batched_tokens=vllm.max_num_batched_tokens,
        kv_cache_dtype=_kv_cache_dtype(vllm.kv_cache_dtype),
        kv_cache_memory_bytes=vllm.kv_cache_memory_bytes,
        enforce_eager=vllm.enforce_eager,
        max_lora_rank=max(vllm.max_lora_rank, lora.lora_r),
        max_loras=vllm.max_loras,
        weight_dtype=_weight_dtype(vllm.dtype),
        weight_variant=("engine" if quantization != "none" or stripped else "base"),
        concurrent_requests=spec.batch_size * _group_size(spec),
    )


def _model_spec(
    spec: LLMAlgorithmSpec,
    model_config: dict[str, Any],
    n_params: int | None,
    generation: GenerationKnobs,
    vllm: VLLMConfig,
) -> ModelSpec:
    variants = [WeightVariant()]
    if generation.weight_variant != "base":
        variants.append(
            WeightVariant(
                name="engine",
                quantization=_engine_quantization(vllm.quantization),
                stripped_multimodal=bool(vllm.strip_multimodal_towers),
            )
        )
    model_id = spec.pretrained_model_name_or_path or "unnamed-model"
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
        names. Fetched by the caller (the CLI fetches it from the Hub) so the
        calculation core itself stays free of I/O.
    :param n_params: Exact parameter count when known, e.g. from safetensors
        metadata; ``None`` falls back to the analytic count.
    :param gen_device: Where generation runs when the manifest's rollout mode
        moves it off the trainer GPU. Ignored for colocated runs; defaults to
        a device of the same class as ``device`` for async modes.
    """
    manifest = _validated(manifest)
    spec = _llm_spec(manifest)
    vllm: VLLMConfig = getattr(spec, "vllm_config", None) or VLLMConfig()
    generation = generation_knobs_from_manifest(manifest)
    colocated = not manifest.training.async_rollout
    return RunConfig(
        model=_model_spec(spec, model_config, n_params, generation, vllm),
        train_device=device,
        gen_device=None if colocated else (gen_device or device),
        training=training_knobs_from_manifest(manifest),
        generation=generation,
        # Every Arena submission executes under Ray orchestration.
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
    """Estimate both phase peaks straight from the manifest.

    Convenience wrapper: :func:`run_config_from_manifest` piped into
    :func:`~agilerl.arena.memory.estimator.estimate_run`.
    """
    return estimate_run(
        run_config_from_manifest(
            manifest, device, model_config, n_params=n_params, gen_device=gen_device
        )
    )
