# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Pick the cheapest resource tier whose node fits a training manifest."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.manifest import (
    async_rollout,
    configured_rollout_engines,
    device_spec_from_resource_class,
    lookup_gpu,
    unsupported_gpu_reason,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.engine import (
    resolve_engine_tensor_parallel,
    resolve_rollout_engines_per_agent,
)
from agilerl.arena.memory.resolve.split import resolve_run_memory
from agilerl.arena.memory.specs import DeviceSpec, RunConfig
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest

# Setting either count pins the split; the run searches it only when both are omitted.
SPLIT_COUNT_KEYS = frozenset({"training_gpus_per_agent", "rollout_engines_per_agent"})


@dataclass(frozen=True)
class TierCheck:
    """One tier's verdict for a manifest.

    ``config`` and ``estimate`` are ``None`` when the tier was ruled out
    before estimating: no catalogue GPU, a GPU LLM training cannot run on,
    or too few GPUs for the layout.
    """

    tier: dict[str, Any]
    reason: str | None
    config: RunConfig | None = None
    estimate: RunEstimate | None = None

    @property
    def fits(self) -> bool:
        return self.reason is None


def gpus_required(manifest: TrainingManifest, tensor_parallel_size: int = 1) -> int:
    """GPUs one node needs: each member's trainers plus its rollout engines.

    Each engine occupies ``tensor_parallel_size`` GPUs, matching
    ``vllm_config.tensor_parallel_size``. ``auto`` rollout engines need at
    least one engine per member.
    """
    training = manifest.training
    engines = configured_rollout_engines(manifest)
    engine_count = 1 if engines == "auto" else engines
    per_member = training.training_gpus_per_agent + engine_count * tensor_parallel_size
    return training.pop_size * per_member


def engine_width(
    manifest: TrainingManifest,
    num_gpus: int,
    device: DeviceSpec,
    gpus_per_node: int | None = None,
) -> int:
    """GPUs per rollout engine the run starts on ``num_gpus``.

    :param manifest: Validated training manifest.
    :param num_gpus: GPUs in the allocation.
    :param device: One of those GPUs.
    :param gpus_per_node: GPUs on one node; an engine stays on one node.
        ``None`` puts all ``num_gpus`` on one node.
    :return: Engine tensor-parallel width.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return 1
    if spec.vllm_config.tensor_parallel_size is not None:
        return spec.vllm_config.tensor_parallel_size
    # An omitted width goes through the run's resolver; unresolved starts at 1.
    trial = manifest.model_copy(deep=True)
    return resolve_engine_tensor_parallel(trial, num_gpus, device, gpus_per_node) or 1


def engines_per_member(
    manifest: TrainingManifest, num_gpus: int, tensor_parallel_size: int
) -> int:
    """Rollout engines each member starts on ``num_gpus``, as the run splits them.

    :param manifest: Validated training manifest.
    :param num_gpus: GPUs on the node.
    :param tensor_parallel_size: GPUs per rollout engine.
    :return: Engines per population member.
    :raises ValueError: The node has too few GPUs for the layout.
    """
    needed = gpus_required(manifest, tensor_parallel_size)
    if num_gpus < needed:
        msg = f"{num_gpus} GPUs, the job needs {needed}"
        raise ValueError(msg)
    training = manifest.training
    return resolve_rollout_engines_per_agent(
        configured_rollout_engines(manifest),
        num_gpus,
        training.pop_size,
        training.training_gpus_per_agent,
        tensor_parallel_size,
    )


def searches_gpu_split(manifest: TrainingManifest) -> bool:
    """Whether the run picks its trainer / engine split: async, with neither count set."""
    return async_rollout(manifest) and not (
        SPLIT_COUNT_KEYS & manifest.training.model_fields_set
    )


def with_run_gpu_split(
    manifest: TrainingManifest,
    num_gpus: int,
    device: DeviceSpec,
    gpus_per_node: int | None = None,
) -> TrainingManifest:
    """The manifest with the trainers, engines, engine width and rollout batch the run picks on ``num_gpus``.

    Runs :func:`resolve_run_memory`'s throughput search, as the Ray driver
    does when both counts are omitted.

    :param manifest: Validated async rollout manifest that sets neither count. Not written.
    :param num_gpus: GPUs in the allocation.
    :param device: One of those GPUs.
    :param gpus_per_node: GPUs on one node; an engine stays on one node.
        ``None`` puts all ``num_gpus`` on one node.
    :return: A copy with the split written into ``training`` and the width
        into ``vllm_config.tensor_parallel_size``.
    :raises ValueError: When the model is not in the catalogue or no split fits.
    """
    sized = manifest.model_copy(deep=True)
    spec = sized.algorithm
    model = catalog_model(spec)
    if model is None:
        msg = f"{spec.pretrained_model_name_or_path!r} is not in the memory estimator catalogue"
        raise ValueError(msg)
    split = resolve_run_memory(
        sized, model, device, num_gpus // sized.training.pop_size, gpus_per_node
    ).split
    if split is not None:
        sized.training.training_gpus_per_agent = split.trainers
        sized.training.rollout_engines_per_agent = split.engines
        sized.training.rollout_batch_size = split.rollout_batch_size
        if isinstance(spec, RolloutLLMSpec) and spec.vllm_config is not None:
            spec.vllm_config.tensor_parallel_size = split.engine_width
    return sized


def check_tiers(
    tiers: Iterable[dict[str, Any]],
    manifest: TrainingManifest,
    build_config: Callable[[TrainingManifest, DeviceSpec, int, int], RunConfig],
) -> list[TierCheck]:
    """Every tier's verdict, cheapest first.

    An async manifest that sets neither GPU count is sized at the split the
    run picks on each tier (:func:`with_run_gpu_split`). When no split fits,
    the tier is sized at the manifest's counts and keeps the search's reason.

    :param tiers: Resource tiers as ``arena resources list`` returns them. A
        tier's ``gpus_per_node``, when set, keeps each engine on one node.
    :param manifest: Validated training manifest, for the GPU layout.
    :param build_config: The run config of the manifest as laid out on a
        tier, on one GPU of it, at the engine width and engines per member
        that tier starts.
    """
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
        unsupported = unsupported_gpu_reason(device)
        if unsupported is not None:
            checks.append(TierCheck(tier, unsupported))
            continue
        num_gpus = int(tier["num_gpus"])
        gpus_per_node = tier.get("gpus_per_node")
        laid_out = manifest
        split_reason = None
        if searches_gpu_split(manifest):
            try:
                laid_out = with_run_gpu_split(manifest, num_gpus, device, gpus_per_node)
            except ValueError as err:
                split_reason = str(err)
        width = engine_width(laid_out, num_gpus, device, gpus_per_node)
        needed = gpus_required(laid_out, width)
        if num_gpus < needed:
            checks.append(TierCheck(tier, f"{num_gpus} GPUs, the job needs {needed}"))
            continue
        engines = engines_per_member(laid_out, num_gpus, width)
        config = build_config(laid_out, device, width, engines)
        estimate = estimate_run(config)
        over = [
            phase.phase
            for phase in (estimate.training, estimate.generation)
            if not phase.fits_with_buffer
        ]
        reason = f"{' and '.join(over)} over budget" if over else None
        checks.append(TierCheck(tier, split_reason or reason, config, estimate))
    return checks
