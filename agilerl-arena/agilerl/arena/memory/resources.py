# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Pick the cheapest resource tier whose node fits a training manifest."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.manifest import device_spec_from_resource_class, lookup_gpu
from agilerl.arena.memory.resolve.engine import resolve_engine_tensor_parallel
from agilerl.arena.memory.specs import DeviceSpec, RunConfig
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.training import TrainingSpec


@dataclass(frozen=True)
class TierCheck:
    """One tier's verdict for a manifest.

    ``config`` and ``estimate`` are ``None`` when the tier was ruled out
    before estimating: no catalogue GPU, or too few GPUs for the layout.
    """

    tier: dict[str, Any]
    reason: str | None
    config: RunConfig | None = None
    estimate: RunEstimate | None = None

    @property
    def fits(self) -> bool:
        return self.reason is None


def gpus_required(training: TrainingSpec, tensor_parallel_size: int = 1) -> int:
    """GPUs one node needs: each member's trainers plus its rollout engines.

    Each engine occupies ``tensor_parallel_size`` GPUs, matching
    ``vllm_config.tensor_parallel_size``. ``rollout_engines_per_agent="auto"``
    needs at least one engine per member.
    """
    engines = training.rollout_engines_per_agent
    engine_count = 1 if engines == "auto" else engines
    per_member = training.training_gpus_per_agent + engine_count * tensor_parallel_size
    return training.pop_size * per_member


def engine_width(manifest: TrainingManifest, num_gpus: int, device: DeviceSpec) -> int:
    """GPUs per rollout engine the run starts on one node of ``num_gpus``.

    :param manifest: Validated training manifest.
    :param num_gpus: GPUs on the node.
    :param device: One of those GPUs.
    :return: Engine tensor-parallel width.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return 1
    if spec.vllm_config.tensor_parallel_size is not None:
        return spec.vllm_config.tensor_parallel_size
    # An omitted width goes through the run's resolver; unresolved starts at 1.
    trial = manifest.model_copy(deep=True)
    return resolve_engine_tensor_parallel(trial, num_gpus, device, num_gpus) or 1


def engines_per_member(
    training: TrainingSpec, num_gpus: int, tensor_parallel_size: int
) -> int:
    """Rollout engines each member starts on ``num_gpus``, as the run splits them.

    :param training: Training section of the manifest.
    :param num_gpus: GPUs on the node.
    :param tensor_parallel_size: GPUs per rollout engine.
    :return: Engines per population member.
    :raises ValueError: The node has too few GPUs for the layout.
    """
    needed = gpus_required(training, tensor_parallel_size)
    if num_gpus < needed:
        msg = f"{num_gpus} GPUs, the job needs {needed}"
        raise ValueError(msg)
    engines = training.rollout_engines_per_agent
    if engines != "auto":
        return int(engines)
    surplus = num_gpus - training.pop_size * training.training_gpus_per_agent
    return surplus // (training.pop_size * tensor_parallel_size)


def check_tiers(
    tiers: Iterable[dict[str, Any]],
    manifest: TrainingManifest,
    build_config: Callable[[DeviceSpec, int, int], RunConfig],
) -> list[TierCheck]:
    """Every tier's verdict, cheapest first.

    :param tiers: Resource tiers as ``arena resources list`` returns them.
    :param manifest: Validated training manifest, for the GPU layout.
    :param build_config: The manifest's run config on one GPU of a tier, at
        the engine width and engines per member that tier starts.
    """
    training = manifest.training
    checks = []
    for tier in sorted(tiers, key=lambda t: t["price_per_node_hour"]):
        gpu_type = tier.get("gpu_type")
        if not gpu_type:
            checks.append(TierCheck(tier, "no GPU"))
            continue
        if lookup_gpu(gpu_type) is None:
            checks.append(TierCheck(tier, f"unknown GPU {gpu_type!r}"))
            continue
        device = device_spec_from_resource_class(tier)
        num_gpus = int(tier["num_gpus"])
        width = engine_width(manifest, num_gpus, device)
        needed = gpus_required(training, width)
        if num_gpus < needed:
            checks.append(TierCheck(tier, f"{num_gpus} GPUs, the job needs {needed}"))
            continue
        engines = engines_per_member(training, num_gpus, width)
        config = build_config(device, width, engines)
        estimate = estimate_run(config)
        over = [
            phase.phase
            for phase in (estimate.training, estimate.generation)
            if not phase.fits_with_buffer
        ]
        reason = f"{' and '.join(over)} over budget" if over else None
        checks.append(TierCheck(tier, reason, config, estimate))
    return checks
