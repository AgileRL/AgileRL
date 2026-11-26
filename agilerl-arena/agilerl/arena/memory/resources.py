# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Pick the cheapest resource tier whose node fits a training manifest."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from agilerl.arena.memory.estimator import RunEstimate, estimate_run
from agilerl.arena.memory.formulas import recommendation_fits
from agilerl.arena.memory.manifest import (
    configured_rollout_engines,
    device_spec_from_resource_class,
    estimate_teacher_generation,
    lookup_gpu,
    training_memory_from_manifest,
    training_plan_model,
    unsupported_gpu_reason,
)
from agilerl.arena.memory.resolve.batch import TrainerSplitError, catalog_model
from agilerl.arena.memory.resolve.engine import (
    queue_warning,
    resolve_engine_tensor_parallel,
    resolve_rollout_engines_per_agent,
    resolve_teacher_tensor_parallel,
)
from agilerl.arena.memory.resolve.fsdp import TrainingMemoryUnfitError
from agilerl.arena.memory.resolve.lengths import ExpectedLengths
from agilerl.arena.memory.resolve.parallel import (
    ParallelLayout,
    cross_node_expert_bytes,
    layout_runs,
    manifest_at_layout,
    manifest_fsdp,
    off_node_reason,
    parallel_layouts,
    pinned_layout,
    trains_context_parallel,
    trains_expert_parallel,
)
from agilerl.arena.memory.resolve.run import (
    RunMemoryPlan,
    SearchCache,
    SplitSearchError,
    WeightSyncMode,
    kept_split_memory,
    resolve_run_memory,
    weight_sync_modes,
)
from agilerl.arena.memory.resolve.split import (
    RowTokens,
    TrainerCost,
    dataset_learn_seconds,
    price_gpu_split,
    split_workload_from_manifest,
)
from agilerl.arena.memory.resolve.weight_sync import SyncLink
from agilerl.arena.memory.solver import CannotSolve
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec, RunConfig
from agilerl.arena.memory.training_host import layout_free_host_bytes
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.dpo import DPOSpec
from agilerl.arena.models.algorithms.opd import (
    distillation_teacher,
    teacher_engine_gpus,
)
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.algorithms.sft import SFTSpec
from agilerl.arena.models.fsdp import parallel_layout_pins
from agilerl.arena.models.manifest import TrainingManifest, async_rollout

# Setting either count pins the split; the run searches it only when both are omitted.
SPLIT_COUNT_KEYS = frozenset({"training_gpus_per_agent", "rollout_engines_per_agent"})


class FailureKind(str, Enum):
    """Why a check does not fit: GPU memory, host memory, or anything else."""

    GPU_MEMORY = "gpu_memory"
    HOST_MEMORY = "host_memory"
    OTHER = "other"


@dataclass(frozen=True)
class TierCheck:
    """One tier's verdict for a manifest.

    ``config`` and ``estimate`` are ``None`` when the tier was ruled out
    before estimating: no catalogue GPU, a GPU LLM training cannot run on,
    a searched parallel group or an engine that leaves one node, or too few GPUs for
    the layout. ``layout`` is the pinned training
    layout, ``None`` when the manifest was priced as written.
    ``failure`` is ``None`` exactly when the check fits. ``host_limited`` is
    true when training missed the host share or only fitted after re-pricing
    for it, including on a fit. ``cross_node_expert_bytes`` is what a fitting
    check's trainer plan moves between nodes per optimizer step
    (:func:`cross_node_expert_bytes`); zero on a miss.
    """

    tier: dict[str, Any]
    reason: str | None
    config: RunConfig | None = None
    estimate: RunEstimate | None = None
    trainers: int | None = None
    engines: int | None = None
    num_nodes: int = 1
    layout: ParallelLayout | None = None
    engine_tensor_parallel: int | None = None
    # GPUs per separate OPD teacher engine; ``None`` without one.
    teacher_engine_tensor_parallel: int | None = None
    # Predicted step: the slower of learn and generation for a searched split,
    # the learn step for SFT and DPO, else ``None``.
    step_seconds: float | None = None
    # A searched split's step at the low and high tokens per turn of the
    # expected lengths; both equal ``step_seconds`` without a range.
    # ``None`` exactly when ``step_seconds`` is.
    step_seconds_low: float | None = None
    step_seconds_high: float | None = None
    failure: FailureKind | None = None
    host_limited: bool = False
    cross_node_expert_bytes: float = 0.0

    @property
    def fits(self) -> bool:
        return self.reason is None


def gpus_required(manifest: TrainingManifest, tensor_parallel_size: int = 1) -> int:
    """GPUs one node needs: each member's trainers plus its rollout engines.

    Each engine occupies ``tensor_parallel_size`` GPUs, matching
    ``vllm_config.tensor_parallel_size``, plus its teacher engine's when a
    separate model scores. ``auto`` rollout engines need at least one engine
    per member.
    """
    training = manifest.training
    engines = configured_rollout_engines(manifest)
    engine_count = 1 if engines == "auto" else engines
    engine_gpus = tensor_parallel_size + teacher_engine_gpus(manifest.algorithm)
    per_member = training.training_gpus_per_agent + engine_count * engine_gpus
    return training.pop_size * per_member


def tier_sync_link(num_gpus: int, gpus_per_node: int | None) -> SyncLink | None:
    """Link a tier's engines read synced weights over: CUDA IPC on one node, else unknown.

    :param num_gpus: GPUs in the allocation.
    :param gpus_per_node: GPUs on one node; ``None`` puts all ``num_gpus`` on one node.
    :return: ``cuda_ipc``, or ``None`` when the allocation spans nodes.
    """
    if gpus_per_node is None or num_gpus <= gpus_per_node:
        return "cuda_ipc"
    return None


def engine_width(
    manifest: TrainingManifest,
    num_gpus: int,
    device: DeviceSpec,
    gpus_per_node: int | None = None,
    adapter_sync: bool = False,
) -> int:
    """GPUs per rollout engine the run starts on ``num_gpus``.

    :param manifest: Validated training manifest.
    :param num_gpus: GPUs in the allocation.
    :param device: One of those GPUs.
    :param gpus_per_node: GPUs on one node; an engine stays on one node.
        ``None`` puts all ``num_gpus`` on one node.
    :param adapter_sync: Weight sync pushes the adapter even where the engines
        could serve merged weights.
    :return: Engine tensor-parallel width.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return 1
    if spec.vllm_config.tensor_parallel_size is not None:
        return spec.vllm_config.tensor_parallel_size
    # An omitted width goes through the run's resolver; unresolved starts at 1.
    trial = manifest.model_copy(deep=True)
    return (
        resolve_engine_tensor_parallel(
            trial, num_gpus, device, gpus_per_node, adapter_sync
        )
        or 1
    )


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
        tensor_parallel_size + teacher_engine_gpus(manifest.algorithm),
    )


def teacher_sized_manifest(
    manifest: TrainingManifest, device: DeviceSpec, num_gpus: int, gpus_per_node: int
) -> TrainingManifest:
    """``manifest`` with an omitted teacher engine width sized as the run sizes it on ``device``.

    :param manifest: Validated training manifest. Not written.
    :param device: One GPU of the allocation.
    :param num_gpus: GPUs in the allocation.
    :param gpus_per_node: GPUs on one node.
    :return: A copy with ``teacher.model.vllm_config.tensor_parallel_size``
        set, or ``manifest`` itself when nothing is sized.
    """
    teacher = distillation_teacher(manifest.algorithm)
    if (
        teacher is None
        or teacher.model is None
        or teacher.model.vllm_config.tensor_parallel_size is not None
    ):
        return manifest
    sized = manifest.model_copy(deep=True)
    member_gpus = num_gpus // manifest.training.pop_size
    if (
        resolve_teacher_tensor_parallel(sized, device, member_gpus, gpus_per_node)
        is None
    ):
        return manifest
    return sized


def colocated_manifest(manifest: TrainingManifest, num_gpus: int) -> TrainingManifest:
    """``manifest`` with each colocated member training on every GPU it owns, as Ray builds it.

    :param manifest: Validated training manifest.
    :param num_gpus: GPUs in the allocation.
    :return: A copy with ``training_gpus_per_agent`` set, or ``manifest`` itself under async rollout.
    """
    if async_rollout(manifest):
        return manifest
    laid_out = manifest.model_copy(deep=True)
    laid_out.training.training_gpus_per_agent = max(
        num_gpus // manifest.training.pop_size, 1
    )
    return laid_out


def searches_gpu_split(manifest: TrainingManifest) -> bool:
    """Whether the run picks its trainer / engine split: async, with neither count set."""
    return async_rollout(manifest) and not (
        SPLIT_COUNT_KEYS & manifest.training.model_fields_set
    )


def split_manifest(
    manifest: TrainingManifest,
    num_gpus: int,
    device: DeviceSpec,
    search_cache: SearchCache | None = None,
    sync_link: SyncLink | None = None,
    layout: ParallelLayout | None = None,
    expected_lengths: ExpectedLengths | None = None,
) -> tuple[TrainingManifest, RunMemoryPlan]:
    """The trainers, engines, engine width and rollout batch the run picks on ``num_gpus``, and its plan.

    Runs :func:`resolve_run_memory`'s throughput search, as the Ray driver
    does when both counts are omitted.

    :param manifest: Validated async rollout manifest that sets neither count. Not written.
    :param num_gpus: GPUs in the allocation.
    :param device: One of those GPUs. Its ``gpus_per_node`` keeps an engine
        on one node; ``None`` puts all ``num_gpus`` on one node.
    :param search_cache: Plans and widths already priced for this manifest.
    :param sync_link: Link the engines read synced weights over; ``None``
        leaves sync unpriced.
    :param layout: Training layout to search the split at alone; ``None``
        searches the layout too.
    :param expected_lengths: Expected row lengths to price the split at;
        ``None`` uses the generic defaults.
    :return: A copy with the split written into ``training``, the width into
        ``vllm_config.tensor_parallel_size`` and the plan's layout
        (``layout`` under DDP) pinned on ``fsdp``, and the plan: the
        predicted step and whether weight sync pushes merged weights.
    :raises ValueError: When the model is not in the catalogue or no split fits.
    """
    sized = manifest.model_copy(deep=True)
    spec = sized.algorithm
    model = catalog_model(spec)
    if model is None:
        msg = f"{spec.pretrained_model_name_or_path!r} is not in the memory estimator catalogue"
        raise ValueError(msg)
    plan = resolve_run_memory(
        sized,
        model,
        device,
        num_gpus // sized.training.pop_size,
        sync_link,
        True,
        search_cache,
        layout,
        expected_lengths,
    )
    split = plan.split
    if split is not None:
        sized.training.training_gpus_per_agent = split.trainers
        sized.training.rollout_engines_per_agent = split.engines
        sized.training.rollout_batch_size = split.rollout_batch_size
        if isinstance(spec, RolloutLLMSpec) and spec.vllm_config is not None:
            spec.vllm_config.tensor_parallel_size = split.engine_width
    fsdp = plan.training.fsdp
    planned = (
        layout
        if fsdp is None
        else ParallelLayout(fsdp.tp, fsdp.ep, fsdp.cp, fsdp.shard_group_size)
    )
    if planned is None:
        return sized, plan
    return manifest_at_layout(sized, planned), plan


def _unusable(tier: dict[str, Any]) -> TierCheck | None:
    """The verdict for a tier with no GPU LLM training can run on, else ``None``."""
    gpu_type = tier.get("gpu_type")
    if not gpu_type:
        return TierCheck(tier, "no GPU", failure=FailureKind.OTHER)
    if lookup_gpu(gpu_type) is None:
        return TierCheck(tier, f"unknown GPU {gpu_type!r}", failure=FailureKind.OTHER)
    unsupported = unsupported_gpu_reason(device_spec_from_resource_class(tier))
    if unsupported is not None:
        return TierCheck(tier, unsupported, failure=FailureKind.OTHER)
    return None


def _node_gpus(tier: dict[str, Any]) -> int:
    """GPUs on one node of ``tier``."""
    return int(tier.get("gpus_per_node") or tier["num_gpus"])


def _node_counts(tier: dict[str, Any]) -> list[int]:
    """Node counts ``tier`` offers, fewest first; ``num_gpus`` alone is one allocation."""
    listed = tier.get("node_counts")
    if listed:
        return sorted(int(count) for count in listed)
    return [max(math.ceil(int(tier["num_gpus"]) / _node_gpus(tier)), 1)]


def _allocation(tier: dict[str, Any], num_nodes: int) -> int:
    """GPUs in ``num_nodes`` of ``tier``; a tier without ``node_counts`` is its ``num_gpus``."""
    if tier.get("node_counts"):
        return num_nodes * _node_gpus(tier)
    return int(tier["num_gpus"])


def _most_trainers(manifest: TrainingManifest, allocation: int) -> int:
    """Trainer ranks one member's parallel groups span on ``allocation`` GPUs.

    :return: The kept trainer count, a colocated member's GPUs, or the most a
        searched split may use: a member's GPUs less one engine GPU.
    """
    if searches_gpu_split(manifest):
        engine_gpus = 1 + teacher_engine_gpus(manifest.algorithm)
        return max(allocation // manifest.training.pop_size - engine_gpus, 1)
    if not async_rollout(manifest):
        return max(allocation // manifest.training.pop_size, 1)
    return manifest.training.training_gpus_per_agent


def _pinned_engine_width(manifest: TrainingManifest) -> int:
    """The engine width the manifest pins; 1, the narrowest, when the run resolves it."""
    spec = manifest.algorithm
    if isinstance(spec, RolloutLLMSpec) and spec.vllm_config is not None:
        return spec.vllm_config.tensor_parallel_size or 1
    return 1


def _placement_shortfall(
    placed: TrainingManifest,
    layout: ParallelLayout | None,
    pinned: ParallelLayout,
    allocation: int,
    node_gpus: int,
) -> str | None:
    """Why ``allocation`` GPUs in ``node_gpus``-GPU nodes cannot hold ``placed``, read from its pins alone.

    :param placed: The manifest at ``layout``.
    :param layout: Training layout ``placed`` pins; ``None`` when priced as written.
    :param pinned: The manifest's own layout, which runs as written.
    :return: The reason, or ``None`` when only a memory search can tell.
    """
    if layout is not None:
        off_node = off_node_reason(
            layout,
            pinned,
            _most_trainers(placed, allocation),
            node_gpus,
            not searches_gpu_split(placed),
        )
        if off_node is not None:
            return off_node
    width = _pinned_engine_width(placed)
    if width > node_gpus:
        return (
            f"engine tensor_parallel_size {width} exceeds {node_gpus} GPUs on one node"
        )
    # One rollout group is packed onto one node, teacher included.
    bundle = width + teacher_engine_gpus(placed.algorithm)
    if bundle > node_gpus:
        return f"rollout bundle of {bundle} GPUs exceeds {node_gpus} GPUs on one node"
    if searches_gpu_split(placed):
        return None
    needed = gpus_required(placed, width)
    if allocation < needed:
        return f"{allocation} GPUs, the job needs {needed}"
    return None


def note_queued_sequences(estimate: RunEstimate, config: RunConfig) -> RunEstimate:
    """Attach :func:`queue_warning` to the generation bar when the engine holds fewer sequences than asked."""
    if not config.training.uses_generation_engine:
        return estimate
    message = queue_warning(
        estimate.generation,
        config.generation.max_num_seqs,
        config.generation.tensor_parallel_size,
    )
    if message is None:
        return estimate
    generation = estimate.generation.model_copy(
        update={"warnings": (*estimate.generation.warnings, message)}
    )
    return estimate.model_copy(update={"generation": generation})


def _step_range(
    laid_out: TrainingManifest,
    plan: RunMemoryPlan,
    model: ModelSpec,
    device: DeviceSpec,
    sync_link: SyncLink | None,
    expected_lengths: ExpectedLengths | None,
) -> tuple[float, float] | None:
    """The searched split's step at the low and high tokens per turn of ``expected_lengths``.

    Prices the counts, width, rollout batch and sync mode the search chose
    with :func:`price_gpu_split`; nothing is searched again. Without a low and
    high, or without ``expected_lengths``, both ends are the split's step.

    :param laid_out: The manifest :func:`split_manifest` wrote the split into.
    :param plan: The plan :func:`split_manifest` returned.
    :param model: The model :func:`split_manifest` priced the split on.
    :param expected_lengths: Expected row lengths the split was priced at.
    :return: The low and high step seconds; ``None`` when the counts were kept.
    """
    split = plan.split
    if split is None:
        return None
    if (
        expected_lengths is None
        or expected_lengths.generated_tokens_per_turn_low is None
        or expected_lengths.generated_tokens_per_turn_high is None
    ):
        return split.step_seconds, split.step_seconds
    cost = TrainerCost.of(plan.training)

    def step_at(tokens: float) -> float:
        lengths = expected_lengths.model_copy(
            update={"generated_tokens_per_turn": tokens}
        )
        workload = split_workload_from_manifest(
            laid_out, not plan.merged_lora_sync, lengths
        ).model_copy(
            update={
                "sync_link": sync_link,
                "sync_transport": plan.weight_sync_transport,
            }
        )
        priced = price_gpu_split(
            model,
            device,
            workload,
            split.trainers,
            split.engines,
            split.engine_width,
            lambda _: cost,
        )
        assert priced is not None, "the chosen counts priced"
        return priced.step_seconds

    return (
        step_at(expected_lengths.generated_tokens_per_turn_low),
        step_at(expected_lengths.generated_tokens_per_turn_high),
    )


@dataclass(frozen=True)
class SyncPlan:
    """The counts and weight-sync mode a run picks on one allocation."""

    laid_out: TrainingManifest
    mode: WeightSyncMode
    # The predicted step and its low and high: a searched split's step and
    # :func:`_step_range`, or an SFT or DPO learn step at both ends; ``None``
    # otherwise.
    step_seconds: float | None = None
    step_range: tuple[float, float] | None = None
    # Why no split fits, and that miss's kind; ``None`` when one did or none
    # was searched.
    split_reason: str | None = None
    split_failure: FailureKind | None = None


def _weight_sync_plan(
    manifest: TrainingManifest,
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    allocation: int,
    layout: ParallelLayout | None,
    cache: SearchCache,
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> SyncPlan:
    """The counts and weight-sync mode the run picks on ``allocation`` GPUs.

    A searched split runs :func:`split_manifest`; kept counts, or a split
    that does not fit, take :func:`kept_split_memory`'s mode on ``placed``,
    else the first of :func:`weight_sync_modes`. SFT and DPO price their
    learn step on the kept plan (:func:`dataset_learn_seconds`).

    :param placed: ``manifest`` at ``layout``.
    :param row_tokens: The SFT or DPO dataset's row lengths, or ``None``.
    """
    sync_link = tier_sync_link(allocation, device.gpus_per_node)
    plan_model = training_plan_model(placed, model)
    split_reason = None
    split_failure = None
    if searches_gpu_split(manifest):
        try:
            laid_out, plan = split_manifest(
                manifest, allocation, device, cache, sync_link, layout, expected_lengths
            )
        except SplitSearchError as err:
            split_reason = str(err)
            split_failure = FailureKind(err.kind)
        except ValueError as err:
            split_reason = str(err)
            split_failure = FailureKind.OTHER
        else:
            step_seconds = None if plan.split is None else plan.split.step_seconds
            mode = WeightSyncMode(plan.merged_lora_sync, plan.weight_sync_transport)
            step_range = _step_range(
                laid_out,
                plan,
                plan_model,
                device,
                sync_link,
                expected_lengths,
            )
            return SyncPlan(laid_out, mode, step_seconds, step_range)
    kept = kept_split_memory(
        placed, plan_model, device, allocation, sync_link, cache, expected_lengths
    )
    if kept is None:
        mode = weight_sync_modes(placed, device, sync_link)[0]
        return SyncPlan(
            placed, mode, split_reason=split_reason, split_failure=split_failure
        )
    mode = WeightSyncMode(kept.merged_lora_sync, kept.weight_sync_transport)
    spec = placed.algorithm
    step_seconds = None
    if isinstance(spec, SFTSpec | DPOSpec):
        step_seconds = dataset_learn_seconds(
            spec,
            placed.training.training_gpus_per_agent,
            plan_model,
            device,
            kept.training,
            row_tokens,
        )
    step_range = None if step_seconds is None else (step_seconds, step_seconds)
    return SyncPlan(
        placed,
        mode,
        step_seconds,
        step_range,
        split_reason=split_reason,
        split_failure=split_failure,
    )


def _exception_failure(
    err: CannotSolve | TrainingMemoryUnfitError | TrainerSplitError,
) -> FailureKind:
    """GPU memory for ``CannotSolve``, the unfit error's own kind, else other."""
    if isinstance(err, TrainingMemoryUnfitError):
        return FailureKind(err.failure)
    if isinstance(err, CannotSolve):
        return FailureKind.GPU_MEMORY
    return FailureKind.OTHER


def _other_miss(
    tier: dict[str, Any],
    reason: str,
    num_nodes: int,
    host_limited: bool = False,
) -> TierCheck:
    """A miss before a memory budget: placement, or too few GPUs for the layout."""
    return TierCheck(
        tier,
        reason,
        num_nodes=num_nodes,
        failure=FailureKind.OTHER,
        host_limited=host_limited,
    )


def _solve_miss(
    tier: dict[str, Any],
    num_nodes: int,
    err: CannotSolve | TrainingMemoryUnfitError | TrainerSplitError,
    sync: SyncPlan,
    host_limited: bool,
) -> TierCheck:
    """The check when the solve or the split search raises."""
    reason = sync.split_reason or str(err)
    failure = (
        sync.split_failure if sync.split_reason is not None else _exception_failure(err)
    )
    return TierCheck(
        tier,
        reason,
        num_nodes=num_nodes,
        failure=failure,
        host_limited=host_limited,
    )


def _budget_failure(
    reason: str | None,
    split_reason: str | None,
    split_failure: FailureKind | None,
    training_fits: bool,
    host_limited: bool,
) -> FailureKind | None:
    """``None`` when the tier fits. A split miss keeps its own kind."""
    if reason is None:
        return None
    if split_reason is not None:
        return split_failure
    if not training_fits and host_limited:
        return FailureKind.HOST_MEMORY
    return FailureKind.GPU_MEMORY


def _finish_price(
    tier: dict[str, Any],
    num_nodes: int,
    layout: ParallelLayout | None,
    sync: SyncPlan,
    device: DeviceSpec,
    config: RunConfig,
    engines: int,
    width: int,
    host_limited: bool,
) -> TierCheck:
    """Verdict after a config priced, including host RAM that changed the kind."""
    laid_out = sync.laid_out
    split_reason = sync.split_reason
    step_range = sync.step_range
    estimate = note_queued_sequences(estimate_run(config), config)
    over = [
        phase.phase
        for phase in (estimate.training, estimate.generation)
        if not phase.fits_with_buffer
    ]
    teacher = estimate_teacher_generation(laid_out, device, engines)
    if teacher is not None and not teacher.fits_with_buffer:
        over.append("teacher generation")
    over_reason = f"{' and '.join(over)} over budget" if over else None
    reason = split_reason or over_reason
    limited = host_limited or not estimate.training.host_fits_with_buffer
    failure = _budget_failure(
        reason,
        split_reason,
        sync.split_failure,
        estimate.training.fits_with_buffer,
        limited,
    )
    return TierCheck(
        tier,
        reason,
        config,
        estimate,
        trainers=laid_out.training.training_gpus_per_agent,
        engines=engines,
        num_nodes=num_nodes,
        layout=layout,
        engine_tensor_parallel=width,
        teacher_engine_tensor_parallel=teacher_engine_gpus(laid_out.algorithm) or None,
        step_seconds=sync.step_seconds,
        step_seconds_low=None if step_range is None else step_range[0],
        step_seconds_high=None if step_range is None else step_range[1],
        failure=failure,
        host_limited=limited,
    )


def _price(
    tier: dict[str, Any],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    layout: ParallelLayout | None,
    num_nodes: int,
    cache: SearchCache,
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> TierCheck:
    """``num_nodes`` of ``tier`` at ``layout``.

    :func:`_placement_shortfall` rules the allocation out before any memory
    search. Past it, the allocation syncs weights in the mode the run picks
    over :func:`tier_sync_link`; kept counts take :func:`kept_weight_sync_mode`.

    :param layout: Training layout to pin; ``None`` prices the manifest as written.
    """
    cache.host_limited = False
    node_gpus = _node_gpus(tier)
    allocation = _allocation(tier, num_nodes)
    device = _tier_device(tier, node_gpus)
    manifest = _colocated(manifest, device, allocation, node_gpus, cache)
    placed = manifest if layout is None else manifest_at_layout(manifest, layout)
    shortfall = _placement_shortfall(
        placed,
        layout,
        pinned_layout(manifest_fsdp(manifest)),
        allocation,
        node_gpus,
    )
    if shortfall is not None:
        return _other_miss(tier, shortfall, num_nodes)
    sync = _weight_sync_plan(
        manifest,
        placed,
        model,
        device,
        allocation,
        layout,
        cache,
        expected_lengths,
        row_tokens,
    )
    laid_out = sync.laid_out
    mode = sync.mode
    width = engine_width(laid_out, allocation, device, node_gpus, not mode.merged)
    needed = gpus_required(laid_out, width)
    if allocation < needed:
        return _other_miss(
            tier,
            f"{allocation} GPUs, the job needs {needed}",
            num_nodes,
            cache.host_limited,
        )
    engines = engines_per_member(laid_out, allocation, width)
    try:
        config = build_config(
            laid_out,
            device,
            width,
            engines,
            not mode.merged,
            mode.transport == "nccl_reshard",
        )
    except (CannotSolve, TrainingMemoryUnfitError, TrainerSplitError) as err:
        return _solve_miss(tier, num_nodes, err, sync, cache.host_limited)
    check = _finish_price(
        tier,
        num_nodes,
        layout,
        sync,
        device,
        config,
        engines,
        width,
        cache.host_limited,
    )
    return _with_cross_node_bytes(check, laid_out, model, device, mode, layout)


def _tier_device(tier: dict[str, Any], node_gpus: int) -> DeviceSpec:
    """The device of ``tier`` with ``node_gpus`` GPUs per node."""
    return device_spec_from_resource_class(tier).model_copy(
        update={"gpus_per_node": node_gpus}
    )


def _colocated(
    manifest: TrainingManifest,
    device: DeviceSpec,
    allocation: int,
    node_gpus: int,
    cache: SearchCache,
) -> TrainingManifest:
    """``manifest`` with its teacher sized for ``device`` and colocated on ``allocation`` GPUs.

    The teacher sizing is cached per device, allocation and GPUs per node.
    """
    sized_key = (device.name, allocation, node_gpus)
    if sized_key not in cache.teacher_sized:
        cache.teacher_sized[sized_key] = teacher_sized_manifest(
            manifest, device, allocation, node_gpus
        )
    return colocated_manifest(cache.teacher_sized[sized_key], allocation)


def _with_cross_node_bytes(
    check: TierCheck,
    laid_out: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    mode: WeightSyncMode,
    layout: ParallelLayout | None,
) -> TierCheck:
    """``check`` with the routed-expert bytes a fitting searched ``layout`` moves between nodes per step."""
    if not check.fits or layout is None:
        return check
    trainers = laid_out.training.training_gpus_per_agent
    node_gpus = device.gpus_per_node or trainers
    trained = training_plan_model(laid_out, model)
    if trainers <= node_gpus or not trained.arch.is_moe:
        return check
    try:
        plan = training_memory_from_manifest(
            laid_out, trained, device, not mode.merged, mode.transport == "nccl_reshard"
        )
    except TrainingMemoryUnfitError as unfit:
        # The tier priced this host-RAM miss at the lightest settings.
        plan = unfit.plan
    return replace(
        check,
        cross_node_expert_bytes=cross_node_expert_bytes(
            trained.arch, plan, layout, trainers, node_gpus
        ),
    )


@dataclass(frozen=True)
class NodeShardGroups:
    """``layout`` with one FSDP shard group per node of the tier it is priced on."""

    layout: ParallelLayout


SearchStep = ParallelLayout | NodeShardGroups


def _search_steps(
    manifest: TrainingManifest, tiers: list[dict[str, Any]]
) -> list[SearchStep]:
    """Layouts in search order; a :class:`NodeShardGroups` step shards at each tier's node size.

    The manifest's own pins come first and run as written, then with
    node-sized shard groups. A searched layout is kept when it is allowed
    (:func:`off_node_reason`) at some tier and node count. An expert group
    wider than the smallest node is tried with node-sized shard groups
    first. A set shard group adds no node-sized steps. Empty when the
    manifest has no layout to search: no catalogued LLM.
    """
    spec = manifest.algorithm
    model = catalog_model(spec) if isinstance(spec, LLMAlgorithmSpec) else None
    if model is None:
        return []
    fsdp = manifest_fsdp(manifest)
    ranks_fixed = not searches_gpu_split(manifest)
    spans = {
        (_node_gpus(tier), _most_trainers(manifest, _allocation(tier, count)))
        for tier in tiers
        for count in _node_counts(tier)
    }
    layouts = parallel_layouts(
        model.arch,
        max((trainers for _, trainers in spans), default=1),
        None,
        fsdp,
        trains_expert_parallel(manifest),
        trains_context_parallel(manifest),
        ranks_fixed,
    )
    if not layouts:
        return []
    pinned, *searched = layouts
    own = pinned_layout(fsdp)
    placeable = [
        layout
        for layout in searched
        if any(
            off_node_reason(layout, own, trainers, node, ranks_fixed) is None
            and layout_runs(trainers, layout, ranks_fixed)
            for node, trainers in spans
        )
    ]
    if "shard_group_size" in parallel_layout_pins(fsdp):
        return [pinned, *placeable]
    smallest = min((node for node, _ in spans), default=None)
    steps: list[SearchStep] = [pinned, NodeShardGroups(pinned)]
    for layout in placeable:
        # Routed tokens already cross nodes there; node-sized shard groups
        # keep the dense gathers on one node.
        if smallest is not None and layout.ep > smallest:
            steps.append(NodeShardGroups(layout))
        steps.append(layout)
    return steps


def _node_shard_layout(
    step: NodeShardGroups,
    manifest: TrainingManifest,
    node_gpus: int,
    allocation: int,
) -> ParallelLayout | None:
    """``step``'s layout with a shard group of one node, when the trainers can span more than one; else ``None``."""
    layout = replace(step.layout, shard_group_size=node_gpus)
    trainers = _most_trainers(manifest, allocation)
    ranks_fixed = not searches_gpu_split(manifest)
    if trainers <= node_gpus or not layout_runs(trainers, layout, ranks_fixed):
        return None
    return layout


def _max_allocation(tier: dict[str, Any]) -> int:
    """GPUs in ``tier``'s largest offered node count."""
    return _allocation(tier, _node_counts(tier)[-1])


def _covers(larger: dict[str, Any], smaller: dict[str, Any]) -> bool:
    """Same GPU and node size, at least as many GPUs, and no more host RAM per GPU on ``smaller``."""
    if (
        larger is smaller
        or larger.get("gpu_type") != smaller.get("gpu_type")
        or _node_gpus(larger) != _node_gpus(smaller)
        or _max_allocation(larger) < _max_allocation(smaller)
    ):
        return False
    large_host = device_spec_from_resource_class(larger).host_bytes_per_gpu
    small_host = device_spec_from_resource_class(smaller).host_bytes_per_gpu
    if large_host is None:
        return True
    if small_host is None:
        return False
    return small_host <= large_host


def _price_step(
    step: SearchStep | None,
    tier: dict[str, Any],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    num_nodes: int,
    cache: SearchCache,
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> TierCheck | None:
    """Price ``step`` on ``num_nodes`` of ``tier``. ``None`` when its node-sized shard groups cannot run there."""
    if not isinstance(step, NodeShardGroups):
        return _price(
            tier,
            manifest,
            model,
            build_config,
            step,
            num_nodes,
            cache,
            expected_lengths,
            row_tokens,
        )
    layout = _node_shard_layout(
        step, manifest, _node_gpus(tier), _allocation(tier, num_nodes)
    )
    if layout is None:
        return None
    return _price(
        tier,
        manifest,
        model,
        build_config,
        layout,
        num_nodes,
        cache,
        expected_lengths,
        row_tokens,
    )


def _searched_expert_spans_node(
    step: SearchStep | None,
    tier: dict[str, Any],
    pinned: SearchStep | None,
) -> bool:
    """Whether ``step`` is a searched expert group wider than ``tier``'s node.

    The manifest's own layout (``pinned``) runs as written, with or without
    node-sized shard groups.
    """
    layout = step.layout if isinstance(step, NodeShardGroups) else step
    return (
        layout is not pinned
        and isinstance(layout, ParallelLayout)
        and layout.ep > _node_gpus(tier)
    )


def _lock_layout(
    steps: list[SearchStep | None],
    usable: list[dict[str, Any]],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    cache: SearchCache,
    first_failure: dict[str, TierCheck],
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> TierCheck | None:
    """The first fitting step. A searched expert group wider than a tier's node waits until layouts on that node miss.

    :param steps: Layouts in search order. The first is the manifest's own.
    :param first_failure: Each tier's first failing check, written as tiers fail.
    """
    pinned = steps[0]

    def lock(spanning: bool) -> TierCheck | None:
        for step in steps:
            offered = [
                tier
                for tier in usable
                if _searched_expert_spans_node(step, tier, pinned) == spanning
            ]
            if not offered:
                continue
            found = _first_fit(
                step,
                offered,
                manifest,
                model,
                build_config,
                cache,
                first_failure,
                expected_lengths,
                row_tokens,
            )
            if found is not None:
                return found
        return None

    return lock(False) or lock(True)


def check_tiers(
    tiers: Iterable[dict[str, Any]],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    search_cache: SearchCache | None = None,
    expected_lengths: ExpectedLengths | None = None,
    row_tokens: RowTokens | None = None,
) -> list[TierCheck]:
    """Every tier's verdict, cheapest first.

    The search locks the least parallel training layout that fits, on the
    fewest nodes, on the cheapest tier: layout first, then node count, then
    price. A searched expert group wider than a tier's node is priced on
    that tier only after every layout on the node misses, so expert parallel
    on a node stays ahead of context and tensor parallel when the search
    covers several node sizes. When that lock moves routed-expert bytes
    between nodes, each later layout that differs only in a higher expert
    degree, and maybe its shard group, is priced at the lock's tier and node
    count, and the fit moving the fewest bytes takes the lock
    (:func:`_fewer_cross_node_bytes`). A layout that fails at a tier's largest node count is not priced
    at the counts in between, and a same-GPU tier with no more host RAM per
    GPU is not priced again for that layout. Tiers with at least as many
    GPUs per node and as much memory per GPU are priced at that layout
    and node count, each with its own engine width and trainer / engine
    split. A tier that does not fit there keeps its first failure; it is not
    retried with more parallelism or nodes. A layout whose searched expert
    or tensor x context groups leave one of a tier's nodes does not fit that
    tier; degrees the manifest pins are priced as written. A failed tier
    whose host share misses the buffer on :func:`layout_free_host_bytes`
    is not priced at later layouts; no layout fits it.

    An async manifest that sets neither GPU count is sized at the split the
    run picks (:func:`split_manifest`). When no split fits, the tier is
    sized at the manifest's counts and keeps the search's reason. Each
    allocation syncs weights in the mode the run picks over
    :func:`tier_sync_link`.

    :param tiers: Resource tiers as ``arena resources list`` returns them.
        ``gpus_per_node`` (else ``num_gpus``) is one node. ``node_counts``,
        when set, lists the node counts to search; otherwise ``num_gpus`` is
        the whole allocation. An engine stays on one node.
    :param manifest: Validated training manifest, for the GPU layout.
    :param model: Training-side model, to price kept counts' weight sync.
    :param build_config: The run config of the manifest as laid out on a
        tier, on one GPU of it, at the engine width and engines per member
        that tier starts, whether weight sync pushes the adapter, and whether
        it refits merged weights over ``nccl_reshard``.
    :param search_cache: Plans and widths already priced for this manifest.
        Calls for the same manifest, on other tiers or node counts, may share
        one; ``None`` starts a fresh cache.
    :param expected_lengths: Expected row lengths to price splits and steps
        at; ``None`` uses the generic defaults, with no step range.
    :param row_tokens: An SFT or DPO dataset's row lengths, for step seconds;
        ``None`` prices ``max_model_len`` rows.
    """
    ordered = sorted(tiers, key=lambda t: t["price_per_node_hour"])
    ruled_out = {tier["name"]: check for tier in ordered if (check := _unusable(tier))}
    usable = [tier for tier in ordered if tier["name"] not in ruled_out]
    steps: list[SearchStep | None] = [*_search_steps(manifest, usable)] or [None]
    cache = SearchCache() if search_cache is None else search_cache
    first_failure: dict[str, TierCheck] = {}
    lock = _lock_layout(
        steps,
        usable,
        manifest,
        model,
        build_config,
        cache,
        first_failure,
        expected_lengths,
        row_tokens,
    )
    if lock is not None:
        lock = _fewer_cross_node_bytes(
            lock,
            steps,
            manifest,
            model,
            build_config,
            cache,
            expected_lengths,
            row_tokens,
        )
    return [
        ruled_out.get(tier["name"])
        or _verdict(
            tier,
            lock,
            first_failure,
            manifest,
            model,
            build_config,
            cache,
            expected_lengths,
            row_tokens,
        )
        for tier in ordered
    ]


def _fewer_cross_node_bytes(
    lock: TierCheck,
    steps: list[SearchStep | None],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    cache: SearchCache,
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> TierCheck:
    """``lock``, or the fit at a higher expert degree on its tier and node count that moves fewer bytes between nodes.

    :param lock: The search's lock; kept when it moves nothing between nodes.
    :param steps: The search's steps, in order.
    """
    first = lock.layout
    if first is None or lock.cross_node_expert_bytes == 0:
        return lock
    best = lock
    node_gpus = _node_gpus(lock.tier)
    for step in steps:
        layout = (
            replace(step.layout, shard_group_size=node_gpus)
            if isinstance(step, NodeShardGroups)
            else step
        )
        if not (
            layout is not None
            and layout.ep > first.ep
            and replace(layout, ep=first.ep, shard_group_size=first.shard_group_size)
            == first
        ):
            continue
        check = _price_step(
            step,
            lock.tier,
            manifest,
            model,
            build_config,
            lock.num_nodes,
            cache,
            expected_lengths,
            row_tokens,
        )
        if (
            check is not None
            and check.fits
            and check.cross_node_expert_bytes < best.cross_node_expert_bytes
        ):
            best = check
    return best


def _first_fit(
    step: SearchStep | None,
    usable: list[dict[str, Any]],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    cache: SearchCache,
    first_failure: dict[str, TierCheck],
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> TierCheck | None:
    """The first tier that fits ``step``, fewest nodes then cheapest; ``None`` when none does.

    :param first_failure: Each tier's first failing check, written as tiers fail.
    """
    known: dict[tuple[str, int], TierCheck | None] = {}
    dead = {name for name, check in first_failure.items() if _host_bound(check)}

    def price(tier: dict[str, Any], num_nodes: int) -> TierCheck | None:
        key = (tier["name"], num_nodes)
        if key not in known:
            known[key] = _price_step(
                step,
                tier,
                manifest,
                model,
                build_config,
                num_nodes,
                cache,
                expected_lengths,
                row_tokens,
            )
        return known[key]

    def failed(key: tuple[str, int]) -> bool:
        check = known.get(key)
        return key in known and (check is None or not check.fits)

    counts = sorted({count for tier in usable for count in _node_counts(tier)})
    for num_nodes in counts:
        for tier in usable:
            name = tier["name"]
            offered = _node_counts(tier)
            if num_nodes not in offered or name in dead:
                continue
            smaller = [count for count in offered if count < num_nodes]
            if (
                smaller
                and num_nodes < offered[-1]
                and all(failed((name, count)) for count in smaller)
            ):
                largest = price(tier, offered[-1])
                if largest is None or not largest.fits:
                    _drop_after_largest_miss(tier, largest, usable, first_failure, dead)
                    continue
            check = price(tier, num_nodes)
            if check is None:
                continue
            if check.fits:
                return check
            first_failure.setdefault(name, check)
    return None


def _host_bound(check: TierCheck) -> bool:
    """Whether ``check``'s host share misses the buffer on bytes no layout or split changes."""
    if check.estimate is None:
        return False
    training = check.estimate.training
    return training.host_usable_bytes is not None and not recommendation_fits(
        layout_free_host_bytes(training), training.host_usable_bytes
    )


def _drop_after_largest_miss(
    tier: dict[str, Any],
    largest: TierCheck | None,
    usable: list[dict[str, Any]],
    first_failure: dict[str, TierCheck],
    dead: set[str],
) -> None:
    """Stop pricing ``tier`` and the same-GPU tiers it covers; they keep the miss as their failure."""
    covered = [other for other in usable if _covers(tier, other)]
    if largest is not None:
        first_failure.setdefault(tier["name"], largest)
        for other in covered:
            first_failure.setdefault(
                other["name"],
                TierCheck(
                    other,
                    largest.reason,
                    num_nodes=_node_counts(other)[-1],
                    failure=largest.failure,
                    host_limited=largest.host_limited,
                ),
            )
    dead.add(tier["name"])
    dead.update(other["name"] for other in covered)


def _verdict(
    tier: dict[str, Any],
    lock: TierCheck | None,
    first_failure: dict[str, TierCheck],
    manifest: TrainingManifest,
    model: ModelSpec,
    build_config: Callable[
        [TrainingManifest, DeviceSpec, int, int, bool, bool], RunConfig
    ],
    cache: SearchCache,
    expected_lengths: ExpectedLengths | None,
    row_tokens: RowTokens | None,
) -> TierCheck:
    """``tier``'s check: the lock, a price at the lock, or its first failure."""
    name = tier["name"]
    if lock is not None and tier is lock.tier:
        return lock
    if lock is not None and _holds_lock(tier, lock):
        return _price(
            tier,
            manifest,
            model,
            build_config,
            lock.layout,
            lock.num_nodes,
            cache,
            expected_lengths,
            row_tokens,
        )
    # With no lock, the first step priced and failed every usable tier.
    if lock is None or name in first_failure:
        return first_failure[name]
    if lock.num_nodes in _node_counts(tier):
        reason = f"fewer GPUs per node or less memory per GPU than {lock.tier['name']}"
    else:
        reason = f"no {lock.num_nodes}-node option, where {lock.tier['name']} fits"
    return TierCheck(tier, reason, failure=FailureKind.OTHER)


def _holds_lock(tier: dict[str, Any], lock: TierCheck) -> bool:
    """Whether ``tier`` has as many GPUs per node and memory per GPU as the locked tier, at its node count."""
    lock_bytes = device_spec_from_resource_class(lock.tier).total_bytes
    tier_bytes = device_spec_from_resource_class(tier).total_bytes
    # CUDA-visible memory for two cards sold as the same size can differ by a
    # fraction of a GiB (H100 80GB is 79.11, A100 80GB is 79.25).
    memory_slack_bytes = 512 * 1024 * 1024
    return (
        _node_gpus(tier) >= _node_gpus(lock.tier)
        and tier_bytes + memory_slack_bytes >= lock_bytes
        and lock.num_nodes in _node_counts(tier)
    )
