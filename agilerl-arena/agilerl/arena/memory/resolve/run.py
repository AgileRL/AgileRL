# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose a run's training layout, GPU split and weight-sync mode, and its trainer plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from agilerl.arena.memory.manifest import (
    async_rollout,
    can_sync_merged_lora,
    configured_rollout_engines,
    training_memory_from_manifest,
    unsupported_gpu_reason,
)
from agilerl.arena.memory.resolve.batch import TrainerSplitError
from agilerl.arena.memory.resolve.engine import (
    resolve_engine_tensor_parallel,
    resolve_rollout_engines_per_agent,
    servable_engine_widths,
)
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
)
from agilerl.arena.memory.resolve.parallel import (
    ParallelLayout,
    manifest_at_layout,
    manifest_fsdp,
    off_node_reason,
    parallel_layouts,
    pinned_layout,
    trains_context_parallel,
    trains_expert_parallel,
)
from agilerl.arena.memory.resolve.rollout import default_rollout_batch_size
from agilerl.arena.memory.resolve.split import (
    GpuSplit,
    choose_engine_width,
    choose_gpu_split,
    fullest_split_within,
    no_split_reason,
    price_gpu_split,
    split_workload_from_manifest,
)
from agilerl.arena.memory.resolve.weight_sync import SyncLink
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest

# Merged weight sync is kept while its predicted step is within this share of
# the adapter's. An adapter step the model prices learn-bound still waits on
# slow episodes. On 8x H100, merged was predicted 0.4% slower on AWM
# (Qwen3.6-35B-A3B) and ran 7% faster, and 6.7% slower on countdown
# (Qwen3-4B) and ran 1.6% faster. Applied over ``cuda_ipc`` only, the one link
# these runs measured.
WEIGHT_SYNC_TIE_SHARE = 0.1


@dataclass(frozen=True)
class RunMemoryPlan:
    """The GPU split and trainer memory plan a run starts with."""

    # ``None`` when the manifest's trainer and engine counts are kept.
    split: GpuSplit | None
    training: TrainingMemoryPlan
    # Async weight sync pushes merged weights to engines that run with LoRA
    # off. With no link it is merged whenever the engines can serve it and
    # the trainers fit.
    merged_lora_sync: bool


@dataclass
class SearchCache:
    """Plans and engine widths one search reuses across layouts, node counts and tiers."""

    # Keyed by :func:`_layout_plans_key`, then trainer count; ``None`` when it cannot run.
    plans: dict[tuple, dict[int, TrainingMemoryPlan | None]] = field(
        default_factory=dict
    )
    # Keyed by GPU and engine width: whether one engine holds a sequence.
    servable_widths: dict[tuple, bool] = field(default_factory=dict)


def merged_lora_sync_candidates(manifest: TrainingManifest) -> tuple[bool, ...]:
    """``merged_lora_sync`` values :func:`resolve_run_memory` prices, merged first.

    :param manifest: Validated run manifest.
    :return: Both for engines that can serve merged weights, adapter alone otherwise.
    """
    if not can_sync_merged_lora(manifest.algorithm):
        return (False,)
    return (True, False)


def pick_weight_sync(splits: list[GpuSplit], sync_link: SyncLink | None) -> GpuSplit:
    """The split of the weight-sync mode the run takes, from one split per mode that fits.

    :param splits: One split per :func:`merged_lora_sync_candidates` mode
        that fits, in candidate order.
    :param sync_link: Link the engines read synced weights over, or ``None``.
    :return: :func:`fullest_split_within` at :data:`WEIGHT_SYNC_TIE_SHARE`
        over ``cuda_ipc``, the fastest over other links; with sync unpriced,
        the first.
    """
    if sync_link is None:
        return splits[0]
    share = WEIGHT_SYNC_TIE_SHARE if sync_link == "cuda_ipc" else 0.0
    return fullest_split_within(splits, share)


def _layout_plans_key(
    device: DeviceSpec,
    placed: TrainingManifest,
    model: ModelSpec,
    merged_lora_sync: bool,
) -> tuple:
    """The model, GPU, node size, host RAM share, layout and sync mode one set of trainer plans is priced at."""
    fsdp = placed.algorithm.fsdp
    return (
        model,
        device.name,
        device.total_bytes,
        device.gpus_per_node,
        device.host_bytes_per_gpu,
        fsdp.tp,
        fsdp.ep,
        fsdp.cp,
        fsdp.shard_group_size,
        merged_lora_sync,
    )


def _cached_training_plan(
    cache: SearchCache,
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    trainers: int,
    merged_lora_sync: bool,
    require_buffer: bool,
) -> TrainingMemoryPlan | None:
    """The plan for this trainer count at ``placed``'s layout, or ``None`` when it does not fit.

    :param require_buffer: Only take a plan that fits with the underprediction buffer.
    """
    plans = cache.plans.setdefault(
        _layout_plans_key(device, placed, model, merged_lora_sync), {}
    )
    if trainers not in plans:
        trial = placed.model_copy(deep=True)
        trial.training.training_gpus_per_agent = trainers
        try:
            plans[trainers] = training_memory_from_manifest(
                trial, model, device, not merged_lora_sync
            )
        except (TrainingMemoryUnfitError, TrainerSplitError):
            plans[trainers] = None
    plan = plans[trainers]
    if plan is None or (require_buffer and not plan.breakdown.fits_with_buffer):
        return None
    return plan


def resolve_run_memory(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    sync_link: SyncLink | None = None,
    search_split: bool = True,
    search_cache: SearchCache | None = None,
    layout: ParallelLayout | None = None,
) -> RunMemoryPlan:
    """Choose the training layout, GPU split and weight-sync mode, then the trainer memory plan.

    Layouts are tried from least parallel to most (:func:`parallel_layouts`):
    the first with a split that fits with the underprediction buffer wins,
    else the first that fits at all. A layout stays fixed across its trainer
    counts, each priced with its own :func:`training_memory_from_manifest`
    plan; a count where a group the search raised over the manifest's own
    degrees straddles nodes is skipped (:func:`off_node_reason`). An omitted
    engine width is searched at every width one engine can
    serve a sequence at and picked by :func:`choose_engine_width`. At each
    layout each of :func:`merged_lora_sync_candidates` is searched with one
    sync priced over ``sync_link``, and :func:`pick_weight_sync` picks among
    the modes that fit. Kept trainer and engine counts take
    :func:`kept_split_memory`, the same search at those counts.

    :param manifest: Validated run manifest. Not written.
    :param model: Training-side catalog model.
    :param device: One GPU of the allocation. Its ``gpus_per_node`` keeps an
        engine and a parallel group on one node; ``None`` puts the member's
        GPUs on one node.
    :param member_gpus: GPUs one member owns.
    :param sync_link: Link the engines read synced weights over; ``None``
        leaves sync unpriced and syncs merged weights where the engines can
        serve them and the trainers fit.
    :param search_split: Search the trainer and engine counts; ``False``
        keeps the manifest's.
    :param search_cache: Plans and widths already priced for this manifest.
    :param layout: Training layout to search the split at alone; ``None``
        tries :func:`parallel_layouts`. Read only when the split is searched.
    :return: The plan.
    :raises ValueError: When the GPU cannot run LLM training, or the split is
        searched and none fits.
    :raises TypeError: When the split is searched and the algorithm does not
        generate rollouts.
    :raises TrainingMemoryUnfitError: When nothing fits one trainer GPU.
    """
    unsupported = unsupported_gpu_reason(device)
    if unsupported is not None:
        raise ValueError(unsupported)
    cache = SearchCache() if search_cache is None else search_cache
    if not search_split:
        return _kept_run_memory(manifest, model, device, member_gpus, sync_link, cache)
    engine_widths = _searched_engine_widths(manifest, device, member_gpus, cache)
    fsdp = manifest_fsdp(manifest)
    layouts = (
        parallel_layouts(
            model.arch,
            max(member_gpus - min(engine_widths, default=1), 1),
            device.gpus_per_node,
            fsdp,
            trains_expert_parallel(manifest),
            trains_context_parallel(manifest),
            ranks_fixed=False,
        )
        if layout is None
        else [layout]
    )
    placed = [manifest_at_layout(manifest, searched) for searched in layouts]
    for require_buffer in (True, False):
        for trial in placed:
            plan = _plan_at_layout(
                trial,
                pinned_layout(fsdp),
                model,
                device,
                member_gpus,
                engine_widths,
                sync_link,
                cache,
                require_buffer,
            )
            if plan is not None:
                return plan
    raise _no_split_error(
        manifest, placed, model, device, member_gpus, engine_widths, cache
    )


def _kept_run_memory(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    sync_link: SyncLink | None,
    cache: SearchCache,
) -> RunMemoryPlan:
    """:func:`kept_split_memory` over the whole population's GPUs.

    :raises TrainingMemoryUnfitError: When no mode's plan fits one trainer GPU.
    :raises TrainerSplitError: When no layout splits across the trainer ranks.
    """
    kept = kept_split_memory(
        manifest,
        model,
        device,
        member_gpus * manifest.training.pop_size,
        sync_link,
        cache,
    )
    if kept is not None:
        return kept
    # No mode fits at any layout, so the first mode's own search raises why.
    merged = merged_lora_sync_candidates(manifest)[0]
    return RunMemoryPlan(
        None, training_memory_from_manifest(manifest, model, device, not merged), merged
    )


def _searched_engine_widths(
    manifest: TrainingManifest, device: DeviceSpec, member_gpus: int, cache: SearchCache
) -> list[int]:
    """The pinned engine width, else every width up to one node that one engine serves a sequence at.

    :raises TypeError: When the algorithm does not generate rollouts.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        msg = f"{spec.name} does not generate rollouts; it has no GPU split"
        raise TypeError(msg)
    pinned_width = spec.vllm_config.tensor_parallel_size
    if pinned_width is not None:
        return [pinned_width]
    widest = min(member_gpus - 1, device.gpus_per_node or member_gpus)
    return servable_engine_widths(manifest, device, widest, cache.servable_widths)


def _no_split_error(
    manifest: TrainingManifest,
    placed: list[TrainingManifest],
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    engine_widths: list[int],
    cache: SearchCache,
) -> ValueError:
    """Why no split fits, read at the least parallel layout: the manifest's own pins.

    :param placed: The manifest at each searched layout, least parallel first.
    """
    merged = merged_lora_sync_candidates(manifest)[0]
    least = placed[0] if placed else manifest
    plans = (
        cache.plans.get(_layout_plans_key(device, least, model, merged), {})
        if placed
        else {}
    )
    trainer_counts = range(1, member_gpus - min(engine_widths, default=member_gpus) + 1)
    workload = split_workload_from_manifest(least, not merged)
    return ValueError(
        f"No trainer / rollout-engine split of {member_gpus} "
        f"{device.name} GPUs fits this run: "
        f"{no_split_reason(workload, trainer_counts, plans)}"
    )


def _plan_at_layout(
    placed: TrainingManifest,
    pinned: ParallelLayout,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    engine_widths: list[int],
    sync_link: SyncLink | None,
    cache: SearchCache,
    require_buffer: bool,
) -> RunMemoryPlan | None:
    """The plan in the sync mode :func:`pick_weight_sync` takes at ``placed``'s layout, or ``None``.

    With no link the first mode that fits wins, so later modes are not searched.

    :param placed: The manifest with its layout pinned by :func:`manifest_at_layout`.
    :param pinned: The manifest's own layout, before the search.
    :param cache: Plans already priced for this search.
    :param require_buffer: Only take trainer counts whose plan fits with the buffer.
    """
    by_mode: dict[bool, RunMemoryPlan] = {}
    for mode in merged_lora_sync_candidates(placed):
        found = _split_at_layout(
            placed,
            pinned,
            model,
            device,
            member_gpus,
            engine_widths,
            sync_link,
            mode,
            cache,
            require_buffer,
        )
        if found is None:
            continue
        if sync_link is None:
            return found
        by_mode[mode] = found
    if not by_mode:
        return None
    splits = [plan.split for plan in by_mode.values() if plan.split is not None]
    return by_mode[pick_weight_sync(splits, sync_link).merged_lora_sync]


def _split_at_layout(
    placed: TrainingManifest,
    pinned: ParallelLayout,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    engine_widths: list[int],
    sync_link: SyncLink | None,
    merged_lora_sync: bool,
    cache: SearchCache,
    require_buffer: bool,
) -> RunMemoryPlan | None:
    """The fastest split at ``placed``'s layout in one sync mode, or ``None`` when none fits.

    :param pinned: The manifest's own layout; groups raised over it stay on one node.
    """
    accepted: dict[int, TrainingMemoryPlan] = {}
    layout = pinned_layout(manifest_fsdp(placed))
    node = device.gpus_per_node or member_gpus

    def learn_cost_ratio(trainers: int) -> float | None:
        if off_node_reason(layout, pinned, trainers, node) is not None:
            return None
        plan = _cached_training_plan(
            cache, placed, model, device, trainers, merged_lora_sync, require_buffer
        )
        if plan is None:
            return None
        accepted[trainers] = plan
        return plan.learn_cost_ratio

    workload = split_workload_from_manifest(placed, not merged_lora_sync).model_copy(
        update={"sync_link": sync_link}
    )
    splits = [
        found
        for width in engine_widths
        if (
            found := choose_gpu_split(
                model, device, workload, member_gpus, width, learn_cost_ratio
            )
        )
        is not None
    ]
    split = choose_engine_width(splits)
    if split is None:
        return None
    return RunMemoryPlan(
        split=split,
        training=accepted[split.trainers],
        merged_lora_sync=merged_lora_sync,
    )


def kept_split_memory(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    total_gpus: int,
    sync_link: SyncLink | None,
    cache: SearchCache,
) -> RunMemoryPlan | None:
    """The trainer plan and weight-sync mode at the manifest's trainer and engine counts.

    Layouts for that trainer count are tried from least parallel to most, as
    :func:`training_memory_from_manifest` tries them; the first where a
    :func:`merged_lora_sync_candidates` mode's plan fits with the
    underprediction buffer wins, else the first where one fits at all. When
    both modes fit there, rollout is async and ``sync_link`` is set, each is
    priced at those counts by :func:`price_gpu_split`, with the engine width
    the run resolves for that mode, and picked as a searched split is
    (:func:`pick_weight_sync`). Otherwise, or when the counts cannot be
    priced, the first mode that fits wins.

    :param manifest: Validated run manifest. Not written.
    :param model: Training-side model.
    :param device: One GPU of the allocation. Its ``gpus_per_node`` keeps an
        engine and a parallel group on one node; ``None`` puts
        ``total_gpus`` on one node.
    :param total_gpus: GPUs in the allocation.
    :param sync_link: Link the engines read synced weights over, or ``None``.
    :param cache: Plans already priced for this manifest.
    :return: The plan with ``split`` ``None``, or ``None`` when no mode's plan
        fits at any layout.
    """
    layouts = parallel_layouts(
        model.arch,
        manifest.training.training_gpus_per_agent,
        device.gpus_per_node,
        manifest_fsdp(manifest),
        trains_expert_parallel(manifest),
        trains_context_parallel(manifest),
    )
    placed = [manifest_at_layout(manifest, layout) for layout in layouts]
    for require_buffer in (True, False):
        for trial in placed:
            plan = _kept_plan_at_layout(
                trial, model, device, total_gpus, sync_link, cache, require_buffer
            )
            if plan is not None:
                return plan
    return None


def kept_merged_lora_sync(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    total_gpus: int,
    sync_link: SyncLink | None,
    cache: SearchCache,
) -> bool:
    """Whether :func:`kept_split_memory` syncs merged weights; the first candidate when no mode fits.

    :return: ``True`` for merged sync, ``False`` for adapter sync.
    """
    kept = kept_split_memory(manifest, model, device, total_gpus, sync_link, cache)
    if kept is None:
        return merged_lora_sync_candidates(manifest)[0]
    return kept.merged_lora_sync


def _kept_plan_at_layout(
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    total_gpus: int,
    sync_link: SyncLink | None,
    cache: SearchCache,
    require_buffer: bool,
) -> RunMemoryPlan | None:
    """The plan in the sync mode the kept counts take at ``placed``'s layout, or ``None`` when no mode fits.

    :param placed: The manifest with its layout pinned by :func:`manifest_at_layout`.
    :param require_buffer: Only take a plan that fits with the buffer.
    """
    trainers = placed.training.training_gpus_per_agent
    prices_sync = sync_link is not None and async_rollout(placed)
    fitting: dict[bool, TrainingMemoryPlan] = {}
    for mode in merged_lora_sync_candidates(placed):
        plan = _cached_training_plan(
            cache, placed, model, device, trainers, mode, require_buffer
        )
        if plan is None:
            continue
        fitting[mode] = plan
        if not prices_sync:
            break
    if not fitting:
        return None
    merged = next(iter(fitting))
    if len(fitting) > 1:
        priced = [
            split
            for mode, plan in fitting.items()
            if (
                split := _kept_split(
                    placed,
                    model,
                    device,
                    total_gpus,
                    sync_link,
                    mode,
                    plan.learn_cost_ratio,
                )
            )
            is not None
        ]
        if priced:
            merged = pick_weight_sync(priced, sync_link).merged_lora_sync
    return RunMemoryPlan(split=None, training=fitting[merged], merged_lora_sync=merged)


def _kept_split(
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    total_gpus: int,
    sync_link: SyncLink | None,
    merged_lora_sync: bool,
    learn_cost_ratio: float,
) -> GpuSplit | None:
    """The kept counts at ``placed``'s layout priced with one sync in one mode, or None when they cannot be."""
    trial = placed.model_copy(deep=True)
    resolve_engine_tensor_parallel(
        trial, total_gpus, device, device.gpus_per_node, not merged_lora_sync
    )
    spec = trial.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return None
    # An unresolved width starts at 1, as the run does.
    width = spec.vllm_config.tensor_parallel_size or 1
    trainers = trial.training.training_gpus_per_agent
    engines = resolve_rollout_engines_per_agent(
        configured_rollout_engines(trial),
        total_gpus,
        trial.training.pop_size,
        trainers,
        width,
    )
    if engines < 1:
        return None
    workload = split_workload_from_manifest(trial, not merged_lora_sync)
    workload = workload.model_copy(
        update={
            "sync_link": sync_link,
            "rollout_batch_size": workload.rollout_batch_size
            or default_rollout_batch_size(workload.batch_size, trainers, engines),
        }
    )
    return price_gpu_split(
        model, device, workload, trainers, engines, width, lambda _: learn_cost_ratio
    )
