# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Least parallel training layout that fits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from agilerl.arena.memory.resolve.batch import TrainerSplitError
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
)
from agilerl.arena.memory.specs import ModelArch
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.fsdp import (
    PARALLEL_LAYOUT_FIELDS,
    FSDPConfig,
    parallel_layout_pins,
    record_parallel_pins,
)
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


@dataclass(frozen=True)
class ParallelLayout:
    """Tensor parallel, expert parallel, and the FSDP shard group."""

    tp: int = 1
    ep: int = 1
    shard_group_size: int | None = None


def _degrees(n_gpus: int, node: int, ranks_fixed: bool) -> list[int]:
    """Parallel degrees above 1, up to one node.

    A fixed trainer count only yields degrees that divide it. An open count
    yields every degree up to the node; the split keeps the ones that divide
    the trainer count it picks.
    """
    widest = min(n_gpus, node)
    if ranks_fixed:
        return [degree for degree in range(2, widest + 1) if n_gpus % degree == 0]
    return list(range(2, widest + 1))


def layout_runs(n_gpus: int, layout: ParallelLayout, ranks_fixed: bool) -> bool:
    """Whether ``layout`` can be placed on ``n_gpus`` trainer ranks.

    :param n_gpus: Trainer ranks. When ``ranks_fixed`` is false, the most
        trainers a split may use.
    :param layout: Tensor parallel, expert parallel, and the shard group.
    :param ranks_fixed: ``n_gpus`` is the trainer count, not an upper bound.
    :return: False when a group does not divide the ranks it has to cover.
    """
    shard = layout.shard_group_size
    if shard is not None and (shard % layout.tp or shard % layout.ep):
        return False
    if not ranks_fixed:
        return (
            layout.tp <= n_gpus
            and layout.ep <= n_gpus
            and (shard is None or shard <= n_gpus)
        )
    if n_gpus % layout.tp or n_gpus % layout.ep:
        return False
    return shard is None or n_gpus % shard == 0


def pinned_layout(fsdp: FSDPConfig) -> ParallelLayout:
    """The layout the manifest pins; an unpinned field is 1, or ``None`` for the shard group."""
    pins = parallel_layout_pins(fsdp)
    return ParallelLayout(
        fsdp.tp if "tp" in pins else 1,
        fsdp.ep if "ep" in pins else 1,
        fsdp.shard_group_size if "shard_group_size" in pins else None,
    )


def parallel_layouts(
    arch: ModelArch,
    n_gpus: int,
    gpus_per_node: int | None,
    fsdp: FSDPConfig,
    expert_parallel: bool,
    ranks_fixed: bool = True,
) -> list[ParallelLayout]:
    """Layouts from least parallel to most, skipping ones this model cannot run.

    A set ``tp``, ``ep``, or ``shard_group_size`` stays as the author wrote it.
    Tensor parallel is only offered for a family with a training plan, at
    degrees its heads split over, and
    expert parallel only for a MoE the trainer shards over experts, at degrees
    that divide its experts; both stay within one node. A searched MoE layout
    keeps ``ep >= tp``. HSDP is only offered when the trainer world is larger
    than one node.

    :param arch: Model architecture.
    :param n_gpus: Trainer ranks the degrees must divide, or the most trainers
        a split may use when ``ranks_fixed`` is false.
    :param gpus_per_node: GPUs on one node. ``None`` is one node of ``n_gpus``.
    :param fsdp: The manifest's FSDP config, read for pins.
    :param expert_parallel: The trainer can shard this model's experts
        (:func:`trains_expert_parallel`).
    :param ranks_fixed: ``n_gpus`` is the trainer count, not an upper bound.
    :return: Layouts in the order the search tries them.
    """
    pins = parallel_layout_pins(fsdp)
    pinned = pinned_layout(fsdp)
    node = n_gpus if gpus_per_node is None else gpus_per_node
    ep_degrees, tp_degrees = _searched_degrees(
        arch, pins, _degrees(n_gpus, node, ranks_fixed), expert_parallel
    )
    # Tensor parallel leaves routed experts whole, so a MoE needs ep >= tp.
    moe = arch.is_moe
    candidates = [pinned]
    if "shard_group_size" not in pins and node < n_gpus:
        candidates.append(replace(pinned, shard_group_size=node))
    candidates += [replace(pinned, ep=degree) for degree in ep_degrees]
    candidates += [
        replace(pinned, tp=degree)
        for degree in tp_degrees
        if not moe or degree <= pinned.ep
    ]
    pairs = sorted(
        (e * t, t, e)
        for e in ep_degrees
        for t in tp_degrees
        if e * t <= node and (not moe or e >= t)
    )
    candidates += [replace(pinned, tp=t, ep=e) for _, t, e in pairs]
    return [
        layout
        for layout in dict.fromkeys(candidates)
        if layout_runs(n_gpus, layout, ranks_fixed)
    ]


def _searched_degrees(
    arch: ModelArch, pins: frozenset[str], degrees: list[int], expert_parallel: bool
) -> tuple[list[int], list[int]]:
    """Expert- and tensor-parallel degrees the search tries; a pinned field tries none."""
    n_experts = arch.n_experts if arch.is_moe and arch.n_experts else 0
    ep_degrees = (
        []
        if "ep" in pins or not n_experts or not expert_parallel
        else [degree for degree in degrees if n_experts % degree == 0]
    )
    tp_degrees = (
        []
        if "tp" in pins or not arch.tensor_parallel_plan
        else [
            degree for degree in degrees if _splits_over_tensor_parallel(arch, degree)
        ]
    )
    return ep_degrees, tp_degrees


def _splits_over_tensor_parallel(arch: ModelArch, tp: int) -> bool:
    """Whether the trainer's tensor-parallel plan accepts ``tp`` for ``arch``.

    Query heads split evenly and a rank's heads stay within or span whole
    key-value groups; an untied ``lm_head`` splits its vocab rows; Mamba heads
    split evenly and Mamba groups and ``tp`` divide one another.
    """
    if arch.n_heads % tp or arch.n_heads % arch.n_kv_heads:
        return False
    local_heads = arch.n_heads // tp
    heads_per_kv = arch.n_heads // arch.n_kv_heads
    if heads_per_kv % local_heads and local_heads % heads_per_kv:
        return False
    if not arch.tied_embeddings and arch.vocab_size % tp:
        return False
    if not arch.is_hybrid_ssm:
        return True
    groups = arch.mamba_n_groups
    groups_split = not groups or groups % tp == 0 or tp % groups == 0
    return arch.mamba_n_heads % tp == 0 and groups_split


def trains_expert_parallel(manifest: TrainingManifest) -> bool:
    """Whether the trainer can shard the manifest's model over expert parallel."""
    spec = manifest.algorithm
    hub_id = (
        spec.pretrained_model_name_or_path
        if isinstance(spec, LLMAlgorithmSpec)
        else None
    )
    info = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    return info is not None and info.supports_expert_parallel


def manifest_fsdp(manifest: TrainingManifest) -> FSDPConfig:
    """The manifest's FSDP config; an omitted ``fsdp`` pins no layout, like ``fsdp: {}``."""
    spec = manifest.algorithm
    if isinstance(spec, LLMAlgorithmSpec) and spec.fsdp is not None:
        return spec.fsdp
    return record_parallel_pins(FSDPConfig(), ())


def manifest_at_layout(
    manifest: TrainingManifest, layout: ParallelLayout
) -> TrainingManifest:
    """A copy of ``manifest`` with ``layout`` written onto ``fsdp`` and pinned.

    :param manifest: Validated LLM manifest. Not written.
    :param layout: Tensor parallel, expert parallel, and the shard group.
    :return: The copy; its layout search tries ``layout`` alone.
    """
    trial = manifest.model_copy(deep=True)
    spec = trial.algorithm
    if not isinstance(spec, LLMAlgorithmSpec):
        msg = f"{spec.name} has no training layout"
        raise TypeError(msg)
    fsdp = spec.fsdp or FSDPConfig()
    fsdp.tp = layout.tp
    fsdp.ep = layout.ep
    fsdp.shard_group_size = layout.shard_group_size
    spec.fsdp = record_parallel_pins(fsdp, PARALLEL_LAYOUT_FIELDS)
    return trial


def first_fitting_plan(
    layouts: list[ParallelLayout],
    price: Callable[[ParallelLayout], TrainingMemoryPlan],
) -> TrainingMemoryPlan:
    """Least parallel layout whose plan fits with the underprediction buffer.

    A layout over the GPU or one the trainer ranks cannot split is skipped.
    The first plan that fits with the buffer wins. When none do, the first
    plan that fits without it is returned.

    :param layouts: Layouts in the order to try them.
    :param price: Prices one layout.
    :return: The chosen plan.
    :raises TrainingMemoryUnfitError: When no layout fits one GPU.
    :raises TrainerSplitError: When no layout splits across the trainer ranks.
    """
    bare: TrainingMemoryPlan | None = None
    last_unfit: TrainingMemoryUnfitError | None = None
    last_split: TrainerSplitError | None = None
    for layout in layouts:
        try:
            plan = price(layout)
        except TrainingMemoryUnfitError as err:
            last_unfit = err
            continue
        except TrainerSplitError as err:
            last_split = err
            continue
        if plan.breakdown.fits_with_buffer:
            return plan
        if bare is None and plan.breakdown.fits:
            bare = plan
    if bare is not None:
        return bare
    if last_unfit is not None:
        raise last_unfit
    if last_split is not None:
        raise last_split
    msg = "No training layout can run on this model"
    raise TrainerSplitError(msg)
