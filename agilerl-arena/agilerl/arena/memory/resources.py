# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Pick the cheapest resource tier whose node fits a training manifest."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.manifest import device_spec_from_resource_class, lookup_gpu
from agilerl.arena.memory.specs import DeviceSpec, RunConfig
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


def gpus_required(training: TrainingSpec) -> int:
    """GPUs one node needs: each member's trainers plus its rollout engines.

    ``rollout_engines_per_agent="auto"`` needs at least one engine per member.
    """
    engines = training.rollout_engines_per_agent
    per_member = training.training_gpus_per_agent + (
        1 if engines == "auto" else engines
    )
    return training.pop_size * per_member


def check_tiers(
    tiers: Iterable[dict[str, Any]],
    training: TrainingSpec,
    build_config: Callable[[DeviceSpec], RunConfig],
) -> list[TierCheck]:
    """Every tier's verdict, cheapest first.

    :param tiers: Resource tiers as ``arena resources list`` returns them.
    :param training: The manifest's training section, for the GPU layout.
    :param build_config: The manifest's run config on one GPU of a tier.
    """
    needed = gpus_required(training)
    checks = []
    for tier in sorted(tiers, key=lambda t: t["price_per_node_hour"]):
        gpu_type = tier.get("gpu_type")
        if not gpu_type:
            checks.append(TierCheck(tier, "no GPU"))
            continue
        if lookup_gpu(gpu_type) is None:
            checks.append(TierCheck(tier, f"unknown GPU {gpu_type!r}"))
            continue
        if tier["num_gpus"] < needed:
            checks.append(
                TierCheck(tier, f"{tier['num_gpus']} GPUs, the job needs {needed}")
            )
            continue
        config = build_config(device_spec_from_resource_class(tier))
        estimate = estimate_run(config)
        over = [
            phase.phase
            for phase in (estimate.training, estimate.generation)
            if not phase.fits_with_buffer
        ]
        reason = f"{' and '.join(over)} over budget" if over else None
        checks.append(TierCheck(tier, reason, config, estimate))
    return checks
