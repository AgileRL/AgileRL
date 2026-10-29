# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose omitted vLLM engine width for rollout engines."""

from __future__ import annotations

from typing import Literal

from agilerl.arena.memory.estimator import estimate_generation, generation_can_serve
from agilerl.arena.memory.manifest import (
    MANIFEST_WEIGHT_VARIANTS,
    configured_rollout_engines,
    generation_settings_from_manifest,
)
from agilerl.arena.memory.solver import solve
from agilerl.arena.memory.specs import DeviceSpec, MiB, ModelArch, ModelSpec, RunConfig
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO, ModelInfo

# The predicted KV pool can run up to ~0.44 GiB above the pool vLLM allocates,
# so a predicted pool must clear the bytes it is relied on for by this much.
KV_MARGIN_BYTES = 512 * MiB


def _engine_model(hub_id: str, info: ModelInfo) -> ModelSpec:
    """Catalog model the rollout engine serves, with the manifest's weight variants."""
    counted = info.inspected["num_params"]
    n_params = counted if isinstance(counted, int) else None
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(info.config),
        n_params=n_params,
        variants=MANIFEST_WEIGHT_VARIANTS,
    )


def resolve_rollout_engines_per_agent(
    configured: int | Literal["auto"],
    total_gpus: int,
    pop_size: int,
    training_gpus_per_agent: int,
    gpus_per_engine: int,
) -> int:
    """Rollout engines each population member starts.

    :param configured: Manifest ``rollout_engines_per_agent``. A count is kept.
    :param total_gpus: GPUs in the allocation.
    :param pop_size: Population members.
    :param training_gpus_per_agent: Trainer GPUs per member.
    :param gpus_per_engine: GPUs per engine, its tensor-parallel width.
    :return: The set count, else the GPUs left after the trainers split
        evenly across members, rounded down.
    """
    if configured != "auto":
        return configured
    surplus = total_gpus - pop_size * training_gpus_per_agent
    return surplus // (pop_size * gpus_per_engine)


def _width_candidates(
    leftover: int,
    rollout_engines_per_agent: int | Literal["auto"],
    gpus_per_node: int,
) -> list[tuple[int, int]]:
    """Tensor-parallel widths that fit the leftover GPUs, with their engine counts.

    :param leftover: GPUs left after the trainers are placed.
    :param rollout_engines_per_agent: Engines per member, or ``auto``.
    :param gpus_per_node: GPUs on one node. One engine stays on one node.
    :return: ``(tensor_parallel_size, engines)`` pairs.
    """
    if rollout_engines_per_agent == "auto":
        return [
            (width, leftover // width)
            for width in range(1, leftover + 1)
            if leftover % width == 0 and width <= gpus_per_node
        ]
    engines = int(rollout_engines_per_agent)
    return [
        (width, engines)
        for width in range(1, leftover + 1)
        if engines * width <= leftover and width <= gpus_per_node
    ]


def resolve_engine_tensor_parallel(
    manifest: TrainingManifest,
    total_gpus: int,
    device: DeviceSpec,
    gpus_per_node: int | None = None,
) -> int | None:
    """Assign the omitted engine width that keeps the most sequences resident.

    :param manifest: Validated training manifest.
    :param total_gpus: GPUs in the allocation.
    :param device: One GPU of that allocation.
    :param gpus_per_node: GPUs on one node. Defaults to ``total_gpus``.
    :return: The assigned width, or None when the manifest is left unchanged.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return None
    vllm = spec.vllm_config
    if vllm.tensor_parallel_size is not None:
        return None
    hub_id = spec.pretrained_model_name_or_path
    info = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    if info is None:
        return None
    per_node = total_gpus if gpus_per_node is None else gpus_per_node
    leftover = (
        total_gpus
        - manifest.training.pop_size * manifest.training.training_gpus_per_agent
    )
    if leftover < 1:
        return None
    candidates = _width_candidates(
        leftover, configured_rollout_engines(manifest), per_node
    )
    if not candidates:
        return None

    model = _engine_model(hub_id, info)
    pinned = "max_num_seqs" in vllm.model_fields_set

    def resident_sequences(width: int, engines: int) -> int:
        try:
            trial = generation_settings_from_manifest(
                manifest, tensor_parallel_size=width, rollout_engines=engines
            )
            if pinned:
                breakdown = estimate_generation(model, device, trial)
                if not generation_can_serve(breakdown, KV_MARGIN_BYTES):
                    return 0
                return engines * trial.max_num_seqs
            run = RunConfig(
                model=model,
                train_device=device,
                gen_device=device,
                generation=trial,
            )
            solved = solve(
                run,
                "max_num_seqs",
                phases=("generation",),
                kv_margin_bytes=KV_MARGIN_BYTES,
            )
        except ValueError:
            # Illegal Mamba TP widths raise; a width that cannot be solved scores 0.
            return 0
        return engines * solved.value

    best: tuple[int, int, int] | None = None
    for width, engines in candidates:
        score = resident_sequences(width, engines)
        rank = (score, engines, width)
        if score > 0 and (best is None or rank > best):
            best = rank
    if best is None:
        return None
    chosen = best[2]
    vllm.tensor_parallel_size = chosen
    return chosen
