# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel mesh, ``Shard(0)`` placement, and torch all-to-all dispatch.

``ep > 1`` shards packed-expert weights on an ``ep`` mesh dim, moves tokens
with torch all-to-all, and FSDP-shards expert modules on the leftover
data-parallel axis. Non-expert parameters FSDP-shard inside a shard group
and replicate across groups (HSDP).

:func:`apply_expert_parallel` shards the experts and their LoRA adapters and
installs the EP ``forward`` from :mod:`agilerl.distributed.ep_forward`. Mesh
views live in :mod:`agilerl.distributed.ep_mesh`, the token all-to-alls in
:mod:`agilerl.distributed.ep_dispatch`, and expert placement in
:mod:`agilerl.distributed.ep_sharding`; their public names are re-exported here.
"""

from __future__ import annotations

from peft.tuners.lora.layer import ParamWrapper
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from agilerl.distributed.ep_dispatch import (
    AllToAllCtx,
    AllToAllVar,
    TokenDispatchState,
    all_to_all_single_autograd,
    exchange_expert_counts,
    reference_dispatch_combine,
    token_combine,
    token_dispatch,
)
from agilerl.distributed.ep_forward import (
    ROUTED_COUNTS_SCOPE,
    CombinedBlock,
    DispatchedBlock,
    LocalExperts,
    RoutedCounts,
    RoutedCountsScope,
    TokenAdapterIds,
    TokenExchange,
    install_ep_forward,
    routed_counts_contexts,
    scatter_scaled_expert_rows,
)
from agilerl.distributed.ep_mesh import (
    ParallelMesh,
    build_parallel_mesh,
    ep_data_parallel_size,
    tp_data_parallel_size,
    validate_ep_degree,
)
from agilerl.distributed.ep_sharding import (
    assert_packed_experts_ep_sharded,
    expert_local_tensor,
    expert_param_bytes_local,
    iter_packed_expert_modules,
    module_ep_degree,
    num_packed_experts,
    packed_expert_count,
    packed_expert_counts,
    shard_experts_on_ep,
    stash_ep,
    validate_actor_ep,
)
from agilerl.distributed.replicated_rows import (
    GatherReplicatedRows,
    ReplicatedRowsCtx,
    SliceReplicatedRows,
    replicated_row_span,
)
from agilerl.lora.fused import patch_lora_for_fused_forward

__all__ = [
    "ROUTED_COUNTS_SCOPE",
    "AllToAllCtx",
    "AllToAllVar",
    "CombinedBlock",
    "DispatchedBlock",
    "GatherReplicatedRows",
    "LocalExperts",
    "ParallelMesh",
    "ReplicatedRowsCtx",
    "RoutedCounts",
    "RoutedCountsScope",
    "SliceReplicatedRows",
    "TokenAdapterIds",
    "TokenDispatchState",
    "TokenExchange",
    "all_to_all_single_autograd",
    "apply_expert_parallel",
    "assert_packed_experts_ep_sharded",
    "build_parallel_mesh",
    "ep_data_parallel_size",
    "exchange_expert_counts",
    "expert_local_tensor",
    "expert_param_bytes_local",
    "iter_packed_expert_modules",
    "module_ep_degree",
    "num_packed_experts",
    "packed_expert_count",
    "packed_expert_counts",
    "reference_dispatch_combine",
    "replicated_row_span",
    "routed_counts_contexts",
    "scatter_scaled_expert_rows",
    "shard_experts_on_ep",
    "token_combine",
    "token_dispatch",
    "tp_data_parallel_size",
    "validate_actor_ep",
    "validate_ep_degree",
]


def _shard_lora_linear_on_ep(
    linear: nn.Module,
    ep_mesh: DeviceMesh,
    expert_dim: int,
    lora_rank: int | None = None,
    num_experts: int | None = None,
) -> None:
    """Shard a PEFT LoRA Linear on the stacked expert axis.

    ``A`` is expert-major ``[E*r, in]`` — contiguous ``Shard(0)``. ``B`` is PEFT
    ``[out, E*r]`` packed rank-major as ``[out, r, E]``; shard that last axis
    (``Shard(2)``) so each rank owns a contiguous expert block.
    """
    weight = getattr(linear, "weight", None)
    if weight is None or isinstance(weight, DTensor):
        return
    if expert_dim == 1:
        if weight.ndim != 2:
            msg = (
                "Expected 2D LoRA B weight before EP reshape, got shape "
                f"{tuple(weight.shape)}"
            )
            raise ValueError(msg)
        if lora_rank is None or num_experts is None:
            msg = "LoRA B EP shard requires lora_rank and num_experts."
            raise ValueError(msg)
        out_f, stacked = weight.shape
        if stacked != lora_rank * num_experts:
            msg = (
                f"LoRA B shape {tuple(weight.shape)} is not "
                f"[out, E*r] with E={num_experts}, r={lora_rank}."
            )
            raise ValueError(msg)
        if num_experts % ep_mesh.size() != 0:
            msg = (
                f"LoRA B expert count {num_experts} must be divisible by "
                f"ep ({ep_mesh.size()})."
            )
            raise ValueError(msg)
        weight_3d = (
            weight.detach().reshape(out_f, lora_rank, num_experts).contiguous().clone()
        )
        sharded = distribute_tensor(weight_3d, ep_mesh, [Shard(2)])
    else:
        if weight.ndim != 2:
            msg = (
                f"Expected 2D LoRA weight for EP shard, got shape {tuple(weight.shape)}"
            )
            raise ValueError(msg)
        if weight.shape[expert_dim] % ep_mesh.size() != 0:
            msg = (
                f"LoRA weight dim {expert_dim} size {weight.shape[expert_dim]} "
                f"must be divisible by ep ({ep_mesh.size()})."
            )
            raise ValueError(msg)
        sharded = distribute_tensor(weight, ep_mesh, [Shard(expert_dim)])
    linear.register_parameter(
        "weight", nn.Parameter(sharded, requires_grad=weight.requires_grad)
    )
    stash_ep(linear, ep_mesh)


def _outer_expert_wrappers(model: nn.Module, experts: nn.Module) -> list[nn.Module]:
    """PEFT wrappers whose base is ``experts`` and that are not inner chain links."""
    wrappers: dict[int, nn.Module] = {}
    for module in model.modules():
        get_base = getattr(module, "get_base_layer", None)
        if callable(get_base) and get_base() is experts:
            wrappers[id(module)] = module
        elif getattr(module, "base_layer", None) is experts:
            wrappers[id(module)] = module
    found = list(wrappers.values())
    inner_ids = {id(getattr(wrapper, "base_layer", None)) for wrapper in found}
    return [wrapper for wrapper in found if id(wrapper) not in inner_ids]


def _chain_links(wrapper: nn.Module) -> list[nn.Module]:
    """Every wrapper in a (possibly nested) PEFT wrapper chain, outermost first."""
    links: list[nn.Module] = []
    module: nn.Module | None = wrapper
    while isinstance(module, ParamWrapper):
        links.append(module)
        module = getattr(module, "base_layer", None)
    return links


def _shard_wrapper_adapters(
    wrapper: nn.Module, ep_mesh: DeviceMesh, num_experts: int, seen: set[int]
) -> None:
    """Shard one wrapper's LoRA adapters once (see :func:`apply_expert_parallel`)."""
    if id(wrapper) in seen:
        return
    seen.add(id(wrapper))
    stash_ep(wrapper, ep_mesh)
    lora_a = getattr(wrapper, "lora_A", None)
    lora_b = getattr(wrapper, "lora_B", None)
    adapter_ranks = getattr(wrapper, "r", {}) or {}
    if isinstance(lora_a, nn.ModuleDict):
        for adapter in lora_a.values():
            _shard_lora_linear_on_ep(adapter, ep_mesh, expert_dim=0)
    if isinstance(lora_b, nn.ModuleDict):
        for adapter_name, adapter in lora_b.items():
            rank = int(adapter_ranks[adapter_name])
            _shard_lora_linear_on_ep(
                adapter,
                ep_mesh,
                expert_dim=1,
                lora_rank=rank,
                num_experts=num_experts,
            )


def apply_expert_parallel(
    model: nn.Module,
    ep_mesh: DeviceMesh | None,
    tp_mesh: DeviceMesh | None = None,
    token_blocks: int = 1,
) -> int:
    """``Shard(0)`` packed-expert weights on ``ep_mesh`` and wrap expert ``forward``.

    PEFT stacked LoRA ``A`` is expert-major ``[E*r, in]`` (``Shard(0)``). ``B``
    is ``[out, E*r]`` viewed as ``[out, r, E]`` and ``Shard(2)``'d on the expert
    axis.

    :param model: Model holding packed-expert modules.
    :param ep_mesh: Expert-parallel mesh; ``None`` changes nothing.
    :param tp_mesh: Tensor-parallel mesh whose ranks hold the same tokens.
        Routed and sorted experts then dispatch each token from one of those
        ranks.
    :param token_blocks: Token blocks per routed MoE call; block ``b + 1``'s
        all-to-all overlaps block ``b``'s experts on CUDA.
    :return: How many packed-expert base modules were parallelized.
    """
    if ep_mesh is None:
        return 0
    tp_group = None if tp_mesh is None else tp_mesh.get_group()
    count = 0
    seen: set[int] = set()
    for _name, module in iter_packed_expert_modules(model):
        shard_experts_on_ep(module, ep_mesh)
        wrappers = _outer_expert_wrappers(model, module)
        # The upgrade nests one wrapper per targeted param; the split-LoRA
        # path reads every link, so shard adapters on all of them, not just
        # the outermost.
        num_experts = num_packed_experts(module) or 0
        for wrapper in wrappers:
            for link in _chain_links(wrapper):
                _shard_wrapper_adapters(link, ep_mesh, num_experts, seen)
        install_ep_forward(wrappers[0] if wrappers else module, token_blocks, tp_group)
        count += 1
    if count:
        patch_lora_for_fused_forward(model)
    return count
