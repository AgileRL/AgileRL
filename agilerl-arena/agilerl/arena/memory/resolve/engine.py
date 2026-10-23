# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose an omitted vLLM tensor-parallel width for a rollout allocation."""

from __future__ import annotations

from typing import Literal

from agilerl.arena.memory.estimator import estimate_generation, generation_can_serve
from agilerl.arena.memory.manifest import generation_settings_from_manifest
from agilerl.arena.memory.solver import solve
from agilerl.arena.memory.specs import (
    DeviceSpec,
    ModelArch,
    ModelSpec,
    RunConfig,
    WeightVariant,
)
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


def _width_candidates(
    leftover: int,
    rollout_engines_per_agent: int | Literal["auto"],
    gpus_per_node: int,
) -> list[tuple[int, int]]:
    """Tensor-parallel widths that fit the leftover GPUs, with their engine counts.

    :param leftover: GPUs left after the trainers are placed.
    :param rollout_engines_per_agent: Engines per member, ``auto``, or ``0``.
    :param gpus_per_node: GPUs on one node. One engine stays on one node.
    :return: ``(tensor_parallel_size, engines)`` pairs.
    """
    if rollout_engines_per_agent in {"auto", 0}:
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
        leftover, manifest.training.rollout_engines_per_agent, per_node
    )
    if not candidates:
        return None

    settings = generation_settings_from_manifest(
        manifest, tensor_parallel_size=candidates[0][0]
    )
    counted = info.inspected["num_params"]
    n_params = counted if isinstance(counted, int) else None
    variants = [WeightVariant()]
    if settings.weight_variant == "engine":
        variants.append(WeightVariant(name="engine", stripped_multimodal=True))
    model = ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(info.config),
        n_params=n_params,
        variants=tuple(variants),
    )
    pinned = "max_num_seqs" in vllm.model_fields_set

    def resident_sequences(width: int, engines: int) -> int:
        trial = generation_settings_from_manifest(manifest, tensor_parallel_size=width)
        try:
            if pinned:
                breakdown = estimate_generation(model, device, trial)
                if not generation_can_serve(breakdown):
                    return 0
                return engines * trial.max_num_seqs
            run = RunConfig(
                model=model,
                train_device=device,
                gen_device=device,
                generation=trial,
            )
            solved = solve(run, "max_num_seqs", phases=("generation",))
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
