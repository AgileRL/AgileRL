# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Least parallel training layout that fits."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from math import lcm

from agilerl.arena.memory.resolve.batch import TrainerSplitError
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
    cpu_offload_blocker,
)
from agilerl.arena.memory.specs import ModelArch
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.fsdp import (
    PARALLEL_LAYOUT_FIELDS,
    FSDPConfig,
    parallel_layout_pins,
    record_parallel_pins,
)
from agilerl.arena.models.manifest import TrainingManifest, strips_multimodal_towers
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO, ModelInfo


@dataclass(frozen=True)
class ParallelLayout:
    """Tensor, expert, and context parallel, and the FSDP shard group."""

    tp: int = 1
    ep: int = 1
    cp: int = 1
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
    :param layout: Tensor, expert, and context parallel, and the shard group.
    :param ranks_fixed: ``n_gpus`` is the trainer count, not an upper bound.
    :return: False when a group does not divide the ranks it has to cover, or
        under an open count, when no trainer count up to ``n_gpus`` holds
        every group.
    """
    shard = layout.shard_group_size
    dense = layout.tp * layout.cp
    if shard is not None and (shard % dense or shard % layout.ep):
        return False
    if not ranks_fixed:
        return lcm(dense, layout.ep, shard or 1) <= n_gpus
    if n_gpus % dense or n_gpus % layout.ep:
        return False
    return shard is None or n_gpus % shard == 0


def pinned_layout(fsdp: FSDPConfig) -> ParallelLayout:
    """The layout the manifest pins; an unpinned field is 1, or ``None`` for the shard group."""
    pins = parallel_layout_pins(fsdp)
    return ParallelLayout(
        fsdp.tp if "tp" in pins else 1,
        fsdp.ep if "ep" in pins else 1,
        fsdp.cp if "cp" in pins else 1,
        fsdp.shard_group_size if "shard_group_size" in pins else None,
    )


def parallel_layouts(
    arch: ModelArch,
    n_gpus: int,
    gpus_per_node: int | None,
    fsdp: FSDPConfig,
    expert_parallel: bool,
    context_parallel: bool,
    ranks_fixed: bool = True,
) -> list[ParallelLayout]:
    """Layouts from least parallel to most, skipping ones this model cannot run.

    A set ``tp``, ``ep``, ``cp``, or ``shard_group_size`` stays as the author
    wrote it. HSDP comes first, and only when the trainer world is larger
    than one node. Then one raised axis, in the order expert, context, tensor
    parallel; then two axes, then three, smallest product first. Expert parallel is
    only offered for a MoE the trainer shards over experts, at degrees that
    divide its experts; context parallel only where the trainer splits the
    model's sequences; tensor parallel only for a family with a training plan.
    Each degree must split the model's heads. Searched expert groups and
    tensor x context groups stay within one node (:func:`off_node_reason`);
    under an open count the split drops trainer counts where they straddle
    nodes. A searched MoE layout keeps ``ep >= tp``, and a set
    ``cpu_offload`` keeps the degrees it runs at.

    :param arch: Model architecture.
    :param n_gpus: Trainer ranks the degrees must divide, or the most trainers
        a split may use when ``ranks_fixed`` is false.
    :param gpus_per_node: GPUs on one node. ``None`` is one node of ``n_gpus``.
    :param fsdp: The manifest's FSDP config, read for pins.
    :param expert_parallel: The trainer can shard this model's experts
        (:func:`trains_expert_parallel`).
    :param context_parallel: The trainer can split this run's sequences
        (:func:`trains_context_parallel`).
    :param ranks_fixed: ``n_gpus`` is the trainer count, not an upper bound.
    :return: Layouts in the order the search tries them.
    """
    pins = parallel_layout_pins(fsdp)
    pinned = pinned_layout(fsdp)
    node = n_gpus if gpus_per_node is None else gpus_per_node
    combos = product(
        *_axis_values(
            arch,
            pins,
            pinned,
            _degrees(n_gpus, node, ranks_fixed),
            expert_parallel,
            context_parallel,
        )
    )
    layouts = (replace(pinned, ep=ep, cp=cp, tp=tp) for ep, cp, tp in combos)
    searched = [
        layout
        for layout in layouts
        if off_node_reason(layout, pinned, n_gpus, node, ranks_fixed) is None
        and _trainer_accepts(arch, layout)
        and _offload_runs(fsdp, layout)
    ]
    candidates = [pinned]
    if "shard_group_size" not in pins and node < n_gpus:
        candidates.append(replace(pinned, shard_group_size=node))
    candidates += sorted(searched, key=lambda layout: _search_rank(layout, pinned))
    return [
        layout
        for layout in dict.fromkeys(candidates)
        if layout_runs(n_gpus, layout, ranks_fixed)
    ]


def _axis_values(
    arch: ModelArch,
    pins: frozenset[str],
    pinned: ParallelLayout,
    degrees: list[int],
    expert_parallel: bool,
    context_parallel: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Expert-, context- and tensor-parallel degrees to combine; a pinned or unsupported axis keeps its pinned degree."""
    n_experts = arch.n_experts if arch.is_moe and arch.n_experts else 0
    ep_values = [pinned.ep]
    if "ep" not in pins and expert_parallel and n_experts:
        ep_values += [degree for degree in degrees if n_experts % degree == 0]
    cp_values = [pinned.cp]
    if "cp" not in pins and context_parallel:
        cp_values += degrees
    tp_values = [pinned.tp]
    if "tp" not in pins and arch.tensor_parallel_plan:
        tp_values += degrees
    return ep_values, cp_values, tp_values


def _search_rank(layout: ParallelLayout, pinned: ParallelLayout) -> tuple[int, ...]:
    """Sort key: fewest raised axes, one raised axis as EP, CP, TP, then the smallest product, TP, CP."""
    raised = (layout.ep != pinned.ep, layout.cp != pinned.cp, layout.tp != pinned.tp)
    n_raised = sum(raised)
    axis = raised.index(True) if n_raised == 1 else 0
    return (n_raised, axis, layout.ep * layout.cp * layout.tp, layout.tp, layout.cp)


def off_node_reason(
    layout: ParallelLayout,
    pinned: ParallelLayout,
    n_gpus: int,
    node: int,
    ranks_fixed: bool = True,
) -> str | None:
    """Why a searched expert or tensor x context group of ``layout`` leaves one node, else ``None``.

    A group at the manifest's own degrees runs as written, even across
    nodes. A searched group wider than the node never fits on it. Past one
    node of trainers, a searched group whose size does not divide the node
    crosses into the next one; under an open count that turns on the trainer
    count the split picks, so only the width is checked.

    :param layout: Tensor, expert, and context parallel, and the shard group.
    :param pinned: The manifest's own layout (:func:`pinned_layout`).
    :param n_gpus: Trainer ranks, or the most trainers a split may use.
    :param node: GPUs on one node.
    :param ranks_fixed: ``n_gpus`` is the trainer count, not an upper bound.
    :return: The reason, or ``None`` when every searched group sits on one node.
    """
    groups: list[tuple[str, int]] = []
    if layout.ep != pinned.ep:
        groups.append(("expert parallel", layout.ep))
    if (layout.tp, layout.cp) != (pinned.tp, pinned.cp):
        groups.append(("tensor x context parallel", layout.tp * layout.cp))
    for axis, size in groups:
        if size > node:
            return f"{axis} group of {size} GPUs is wider than a {node}-GPU node"
        if ranks_fixed and n_gpus > node and node % size:
            return f"{axis} groups of {size} GPUs straddle {node}-GPU nodes"
    return None


def _offload_runs(fsdp: FSDPConfig, layout: ParallelLayout) -> bool:
    """Whether the manifest's ``cpu_offload`` runs at ``layout``'s degrees."""
    trial = FSDPConfig(
        cpu_offload=fsdp.cpu_offload, tp=layout.tp, ep=layout.ep, cp=layout.cp
    )
    # Colocated generation: the rollout reason does not depend on the layout.
    return cpu_offload_blocker(trial, False) is None


def _trainer_accepts(arch: ModelArch, layout: ParallelLayout) -> bool:
    """Whether the trainer runs ``arch`` at ``layout``'s degrees."""
    # Tensor parallel leaves routed experts whole, so a MoE needs ep >= tp.
    if arch.is_moe and layout.ep < layout.tp:
        return False
    if layout.tp > 1 and not _splits_over_tensor_parallel(arch, layout.tp):
        return False
    return layout.cp == 1 or _splits_over_context_parallel(arch, layout.tp, layout.cp)


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


def _splits_over_context_parallel(arch: ModelArch, tp: int, cp: int) -> bool:
    """Whether Ulysses context parallel accepts ``cp`` on each of ``tp``'s ranks.

    A tensor rank's query heads split evenly over ``cp``, and its key-value
    heads and ``cp`` divide one another. A Mamba-2 layer splits its heads over
    ``tp * cp``, and a tensor rank's groups and ``cp`` divide one another.
    Gated-delta-net layers stay on the sequence shard.
    """
    local_heads = arch.n_heads // tp
    local_kv = max(local_heads * arch.n_kv_heads // arch.n_heads, 1)
    if local_heads % cp or (local_kv % cp and cp % local_kv):
        return False
    if not arch.is_hybrid_ssm or arch.mamba_gated_delta:
        return True
    groups = max(arch.mamba_n_groups // tp, 1)
    groups_split = groups % cp == 0 or cp % groups == 0
    return arch.mamba_n_heads % (tp * cp) == 0 and groups_split


def _catalog_info(manifest: TrainingManifest) -> ModelInfo | None:
    """The manifest's catalogued model, or ``None`` for a non-LLM or uncatalogued run."""
    spec = manifest.algorithm
    hub_id = (
        spec.pretrained_model_name_or_path
        if isinstance(spec, LLMAlgorithmSpec)
        else None
    )
    return SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None


def trains_expert_parallel(manifest: TrainingManifest) -> bool:
    """Whether the trainer can shard the manifest's model over expert parallel."""
    info = _catalog_info(manifest)
    return info is not None and info.supports_expert_parallel


def trains_context_parallel(manifest: TrainingManifest) -> bool:
    """Whether the trainer can split the manifest's sequences over context parallel.

    A vision-language model needs a rollout run whose engine strips its towers:
    a text-only environment and no tower LoRA.
    """
    info = _catalog_info(manifest)
    if info is None:
        return False
    if "vision" not in info.modalities:
        return True
    spec = manifest.algorithm
    env = manifest.environment
    text_only = isinstance(env, LLMEnvSpec) and env.text_only
    return isinstance(spec, RolloutLLMSpec) and strips_multimodal_towers(
        spec, text_only
    )


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

    Only the manifest, its ``algorithm`` and ``fsdp`` are copied. The other
    sections are ``manifest``'s own objects, so a caller copies before
    writing them.

    :param manifest: Validated LLM manifest. Not written.
    :param layout: Tensor, expert, and context parallel, and the shard group.
    :return: The copy; its layout search tries ``layout`` alone.
    """
    spec = manifest.algorithm
    if not isinstance(spec, LLMAlgorithmSpec):
        msg = f"{spec.name} has no training layout"
        raise TypeError(msg)
    fsdp = FSDPConfig() if spec.fsdp is None else copy.copy(spec.fsdp)
    fsdp.tp = layout.tp
    fsdp.ep = layout.ep
    fsdp.cp = layout.cp
    fsdp.shard_group_size = layout.shard_group_size
    trial_spec = spec.model_copy()
    trial_spec.fsdp = record_parallel_pins(fsdp, PARALLEL_LAYOUT_FIELDS)
    trial = manifest.model_copy()
    trial.algorithm = trial_spec
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
