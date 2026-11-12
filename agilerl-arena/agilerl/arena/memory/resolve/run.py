# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose a run's training layout, GPU split and weight-sync mode, and its trainer plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from agilerl.arena.memory.manifest import (
    async_rollout,
    can_sync_merged_lora,
    configured_rollout_engines,
    nccl_reshard_layout_supported,
    training_memory_from_manifest,
    unsupported_gpu_reason,
)
from agilerl.arena.memory.resolve.batch import TrainerSplitError, catalog_model
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
    TrainerCost,
    choose_engine_width,
    choose_gpu_split,
    fullest_split_within,
    no_split_reason,
    price_gpu_split,
    split_workload_from_manifest,
)
from agilerl.arena.memory.resolve.weight_sync import (
    SyncLink,
    WeightSyncTransport,
    nccl_reshard_costs,
)
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
class WeightSyncMode:
    """One weight-sync candidate: merged or adapter weights, on one transport."""

    merged: bool
    transport: WeightSyncTransport


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
    # Transport the run syncs on.
    weight_sync_transport: WeightSyncTransport = "nixl"
    # The kept or searched split this plan's seconds come from.
    sync_quote: GpuSplit | None = None
    # Every mode priced at the chosen layout, in candidate order.
    quotes: tuple[GpuSplit, ...] = ()


@dataclass
class SearchCache:
    """Plans and engine widths one search reuses across layouts, node counts and tiers."""

    # Keyed by :func:`_layout_plans_key`, then trainer count. The host share is
    # not part of the key. ``None`` when the GPU search cannot run.
    plans: dict[tuple, dict[int, TrainingMemoryPlan | None]] = field(
        default_factory=dict
    )
    # Later rungs after the GPU plan missed a host share, keyed by
    # ``(layout key, trainer count, host share)``. A different share does not
    # reuse the entry.
    host_misses: dict[tuple, TrainingMemoryPlan | None] = field(default_factory=dict)
    # Keyed by GPU and engine width: whether one engine holds a sequence.
    servable_widths: dict[tuple, bool] = field(default_factory=dict)


def weight_sync_modes(
    manifest: TrainingManifest, device: DeviceSpec, sync_link: SyncLink | None
) -> tuple[WeightSyncMode, ...]:
    """Modes :func:`resolve_run_memory` prices.

    NIXL prices merged then adapter weights, or the adapter alone when the
    engines cannot serve merged weights. An async run of a dense model over
    ``cuda_ipc`` (one node) also prices merged ``nccl_reshard`` first when
    :func:`nccl_reshard_costs` has rates for the GPU and the layout passes
    :func:`~agilerl.arena.memory.manifest.nccl_reshard_layout_supported`.

    :param manifest: Validated run manifest.
    :param device: The training and rollout GPU.
    :param sync_link: Link the engines read synced weights over, or ``None``.
    :return: Candidates, merged ``nccl_reshard`` first when it is offered.
    """
    if not can_sync_merged_lora(manifest.algorithm):
        return (WeightSyncMode(False, "nixl"),)
    modes = [WeightSyncMode(True, "nixl"), WeightSyncMode(False, "nixl")]
    model = catalog_model(manifest.algorithm)
    if (
        sync_link == "cuda_ipc"
        and nccl_reshard_costs(device) is not None
        and async_rollout(manifest)
        and model is not None
        and not model.arch.is_moe
        and nccl_reshard_layout_supported(manifest)
    ):
        modes.insert(0, WeightSyncMode(True, "nccl_reshard"))
    return tuple(modes)


def pick_weight_sync(splits: list[GpuSplit], sync_link: SyncLink | None) -> GpuSplit:
    """The split of the weight-sync mode the run takes.

    Among merged transports the lowest :attr:`GpuSplit.synced_step_seconds`
    wins, then the split on more GPUs. That merged split is compared with
    the adapter splits by :func:`fullest_split_within` at
    :data:`WEIGHT_SYNC_TIE_SHARE` over ``cuda_ipc``, and by the fastest step
    over other links. With sync unpriced, the first split wins.

    :param splits: One split per mode that fits, in candidate order.
    :param sync_link: Link the engines read synced weights over, or ``None``.
    :return: The split the run takes.
    """
    if sync_link is None:
        return splits[0]
    share = WEIGHT_SYNC_TIE_SHARE if sync_link == "cuda_ipc" else 0.0
    merged = [split for split in splits if split.merged_lora_sync]
    adapters = [split for split in splits if not split.merged_lora_sync]
    if not merged:
        return fullest_split_within(adapters, share)
    chosen_merged = min(
        merged,
        key=lambda split: (
            split.synced_step_seconds,
            -(split.trainers + split.engines),
        ),
    )
    if not adapters:
        return chosen_merged
    return fullest_split_within([chosen_merged, *adapters], share)


def _layout_plans_key(
    device: DeviceSpec,
    placed: TrainingManifest,
    model: ModelSpec,
    merged_lora_sync: bool,
    transport: WeightSyncTransport,
) -> tuple:
    """The model, GPU, node size, layout and sync mode one set of trainer plans is priced at."""
    fsdp = placed.algorithm.fsdp
    return (
        model,
        device.name,
        device.total_bytes,
        device.gpus_per_node,
        fsdp.tp,
        fsdp.ep,
        fsdp.cp,
        fsdp.shard_group_size,
        merged_lora_sync,
        transport,
    )


def _price_training_plan(
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    trainers: int,
    mode: WeightSyncMode,
) -> TrainingMemoryPlan | None:
    """The trainer plan at ``placed`` on ``device``, or ``None`` when it does not fit.

    :param placed: Manifest at the layout being priced. Not written.
    :param model: Training-side model.
    :param device: GPU the plan is priced on, including its host share.
    :param trainers: Trainer ranks.
    :param mode: Weight-sync mode.
    :return: The plan, or ``None`` when training does not fit one GPU.
    """
    trial = placed.model_copy(deep=True)
    trial.training.training_gpus_per_agent = trainers
    try:
        return training_memory_from_manifest(
            trial,
            model,
            device,
            not mode.merged,
            mode.transport == "nccl_reshard",
        )
    except (TrainingMemoryUnfitError, TrainerSplitError):
        return None


def _host_share_fits(
    plan: TrainingMemoryPlan, host_bytes: int | None, require_buffer: bool
) -> bool:
    """Whether ``plan``'s stored host bytes fit this device's share.

    :param plan: GPU-priced plan. Its host components are already stored.
    :param host_bytes: Host RAM one trainer may hold, or ``None`` to skip.
    :param require_buffer: Require the underprediction buffer
        (:meth:`~agilerl.arena.memory.breakdown.PhaseBreakdown.host_fits_with_buffer`).
    :return: True when the share is unset or the host bytes fit it.
    """
    if host_bytes is None:
        return True
    priced = plan.breakdown.model_copy(update={"host_usable_bytes": host_bytes})
    if require_buffer:
        return priced.host_fits_with_buffer
    return priced.host_fits


def _host_miss_plan(
    cache: SearchCache,
    key: tuple,
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    trainers: int,
    mode: WeightSyncMode,
) -> TrainingMemoryPlan | None:
    """The plan priced on the host share the GPU plan missed.

    :param cache: Search cache for this request.
    :param key: :func:`_layout_plans_key` for this layout.
    :param placed: Manifest at the layout being priced. Not written.
    :param model: Training-side model.
    :param device: Device whose host share missed. That share is the cache key.
    :param trainers: Trainer ranks.
    :param mode: Weight-sync mode.
    :return: The plan for this share, or ``None`` when it cannot run.
    """
    misses = cache.host_misses
    miss_key = (key, trainers, device.host_bytes_per_gpu)
    if miss_key not in misses:
        misses[miss_key] = _price_training_plan(placed, model, device, trainers, mode)
    return misses[miss_key]


def _cached_training_plan(
    cache: SearchCache,
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    trainers: int,
    mode: WeightSyncMode,
    require_buffer: bool,
) -> TrainingMemoryPlan | None:
    """The plan for this trainer count at ``placed``'s layout, or ``None`` when it does not fit.

    :param require_buffer: Only take a plan that fits with the underprediction buffer.
    """
    key = _layout_plans_key(device, placed, model, mode.merged, mode.transport)
    plans = cache.plans.setdefault(key, {})
    if trainers not in plans:
        # No host share, so the rung walk is the GPU search. Host bytes stay
        # on the plan and are checked against this device below.
        plans[trainers] = _price_training_plan(
            placed,
            model,
            device.model_copy(update={"host_bytes_per_gpu": None}),
            trainers,
            mode,
        )
    plan = plans[trainers]
    if plan is None:
        return None
    if not _host_share_fits(plan, device.host_bytes_per_gpu, require_buffer):
        plan = _host_miss_plan(cache, key, placed, model, device, trainers, mode)
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
    layout each of :func:`weight_sync_modes` is searched with one
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
        manifest, placed, model, device, member_gpus, engine_widths, sync_link, cache
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
    mode = weight_sync_modes(manifest, device, sync_link)[0]
    return RunMemoryPlan(
        None,
        training_memory_from_manifest(
            manifest, model, device, not mode.merged, mode.transport == "nccl_reshard"
        ),
        mode.merged,
        mode.transport,
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
    sync_link: SyncLink | None,
    cache: SearchCache,
) -> ValueError:
    """Why no split fits, read at the least parallel layout: the manifest's own pins.

    :param placed: The manifest at each searched layout, least parallel first.
    """
    mode = weight_sync_modes(manifest, device, sync_link)[0]
    least = placed[0] if placed else manifest
    plans = (
        cache.plans.get(
            _layout_plans_key(device, least, model, mode.merged, mode.transport), {}
        )
        if placed
        else {}
    )
    trainer_counts = range(1, member_gpus - min(engine_widths, default=member_gpus) + 1)
    workload = split_workload_from_manifest(least, not mode.merged)
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
    by_mode: dict[WeightSyncMode, RunMemoryPlan] = {}
    for mode in weight_sync_modes(placed, device, sync_link):
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
    chosen = pick_weight_sync(splits, sync_link)
    plan = by_mode[WeightSyncMode(chosen.merged_lora_sync, chosen.sync_transport)]
    return RunMemoryPlan(
        split=chosen,
        training=plan.training,
        merged_lora_sync=chosen.merged_lora_sync,
        weight_sync_transport=chosen.sync_transport,
        sync_quote=chosen,
        quotes=tuple(splits),
    )


def _split_at_layout(
    placed: TrainingManifest,
    pinned: ParallelLayout,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    engine_widths: list[int],
    sync_link: SyncLink | None,
    mode: WeightSyncMode,
    cache: SearchCache,
    require_buffer: bool,
) -> RunMemoryPlan | None:
    """The fastest split at ``placed``'s layout in one sync mode, or ``None`` when none fits.

    :param pinned: The manifest's own layout; groups raised over it stay on one node.
    """
    accepted: dict[int, TrainingMemoryPlan] = {}
    layout = pinned_layout(manifest_fsdp(placed))
    node = device.gpus_per_node or member_gpus

    def trainer_cost(trainers: int) -> TrainerCost | None:
        if off_node_reason(layout, pinned, trainers, node) is not None:
            return None
        plan = _cached_training_plan(
            cache, placed, model, device, trainers, mode, require_buffer
        )
        if plan is None:
            return None
        accepted[trainers] = plan
        return TrainerCost.of(plan)

    workload = split_workload_from_manifest(placed, not mode.merged).model_copy(
        update={"sync_link": sync_link, "sync_transport": mode.transport}
    )
    splits = [
        found
        for width in engine_widths
        if (
            found := choose_gpu_split(
                model, device, workload, member_gpus, width, trainer_cost
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
        merged_lora_sync=mode.merged,
        weight_sync_transport=mode.transport,
        sync_quote=split,
        quotes=(split,),
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
    :func:`weight_sync_modes` mode's plan fits with the
    underprediction buffer wins, else the first where one fits at all. When
    rollout is async and ``sync_link`` is set, each mode that fits there is
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


def kept_weight_sync_mode(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    total_gpus: int,
    sync_link: SyncLink | None,
    cache: SearchCache,
) -> WeightSyncMode:
    """The weight-sync mode :func:`kept_split_memory` picks; the first of :func:`weight_sync_modes` when no mode fits.

    :return: Merged or adapter sync, and its transport.
    """
    kept = kept_split_memory(manifest, model, device, total_gpus, sync_link, cache)
    if kept is None:
        return weight_sync_modes(manifest, device, sync_link)[0]
    return WeightSyncMode(kept.merged_lora_sync, kept.weight_sync_transport)


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
    fitting: dict[WeightSyncMode, TrainingMemoryPlan] = {}
    for mode in weight_sync_modes(placed, device, sync_link):
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
    chosen_mode = next(iter(fitting))
    priced: list[GpuSplit] = []
    if prices_sync:
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
                    TrainerCost.of(plan),
                )
            )
            is not None
        ]
        if priced:
            chosen = pick_weight_sync(priced, sync_link)
            chosen_mode = WeightSyncMode(chosen.merged_lora_sync, chosen.sync_transport)
    quote = next(
        (
            split
            for split in priced
            if split.merged_lora_sync == chosen_mode.merged
            and split.sync_transport == chosen_mode.transport
        ),
        None,
    )
    return RunMemoryPlan(
        split=None,
        training=fitting[chosen_mode],
        merged_lora_sync=chosen_mode.merged,
        weight_sync_transport=chosen_mode.transport,
        sync_quote=quote,
        quotes=tuple(priced),
    )


def _kept_split(
    placed: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    total_gpus: int,
    sync_link: SyncLink | None,
    mode: WeightSyncMode,
    trainer_cost: TrainerCost,
) -> GpuSplit | None:
    """The kept counts at ``placed``'s layout priced with one sync in one mode, or None when they cannot be."""
    trial = placed.model_copy(deep=True)
    resolve_engine_tensor_parallel(
        trial, total_gpus, device, device.gpus_per_node, not mode.merged
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
    workload = split_workload_from_manifest(trial, not mode.merged)
    workload = workload.model_copy(
        update={
            "sync_link": sync_link,
            "sync_transport": mode.transport,
            "rollout_batch_size": workload.rollout_batch_size
            or default_rollout_batch_size(workload.batch_size, trainers, engines),
        }
    )
    return price_gpu_split(
        model, device, workload, trainers, engines, width, lambda _: trainer_cost
    )
