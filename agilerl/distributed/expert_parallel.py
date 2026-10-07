# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel mesh, ``Shard(0)`` placement, and torch all-to-all dispatch.

``ep > 1`` shards packed-expert weights on an ``ep`` mesh dim, moves tokens
with torch all-to-all, and FSDP-shards expert modules on the leftover
data-parallel axis. Non-expert parameters FSDP-shard inside a shard group
and replicate across groups (HSDP).
Packed expert ``forward`` is wrapped here so ``agilerl.lora.moe`` stays EP-blind:
dispatch, remap ids to the local range, run the existing kernel, combine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from types import MethodType
from typing import Any, NamedTuple, Protocol, cast

import torch
import torch.distributed as dist
from peft.tuners.lora.layer import ParamWrapper
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.placement_types import Shard
from torch.func import functional_call

from agilerl.distributed.process import all_reduce_grads
from agilerl.lora.fused import ROUTING_STATE, patch_lora_for_fused_forward
from agilerl.lora.moe.adapters import mixed_routing
from agilerl.lora.moe.grouped_gemm import GROUPED_LINEAR_CHUNK_BYTES
from agilerl.lora.moe.layouts import (
    is_routed_experts_module,
    is_sorted_experts_module,
    routed_projection_names,
)
from agilerl.lora.moe.wrappers import RoutedExpertsLoraWrapper


def validate_ep_degree(ep: int) -> None:
    """Reject a non-positive or non-integer expert-parallel degree."""
    if not isinstance(ep, int) or isinstance(ep, bool):
        msg = f"ep must be an int, got {type(ep).__name__}"
        raise TypeError(msg)
    if ep < 1:
        msg = f"ep must be >= 1, got {ep}"
        raise ValueError(msg)


def ep_data_parallel_size(world_size: int, ep: int) -> int:
    """Leftover data-parallel size after folding ``world_size`` by ``ep``."""
    validate_ep_degree(ep)
    return _fold_world(world_size, ep, "ep", "Expert Parallel")


def tp_data_parallel_size(world_size: int, tp: int) -> int:
    """Distinct batch shards when each ``tp`` consecutive ranks share one."""
    if tp < 1:
        msg = f"tp must be >= 1, got {tp}"
        raise ValueError(msg)
    return _fold_world(world_size, tp, "tp", "Tensor Parallel")


def _fold_world(world_size: int, degree: int, name: str, label: str) -> int:
    if world_size < 1:
        msg = f"world_size must be >= 1, got {world_size}"
        raise ValueError(msg)
    if world_size % degree != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by {name} ({degree}) "
            f"for {label}."
        )
        raise ValueError(msg)
    return world_size // degree


def packed_expert_counts(model: nn.Module) -> list[int]:
    """Expert counts of each packed-expert module in ``model``.

    :param model: Model to scan for packed expert stacks.
    :return: Expert counts in module order, empty without packed experts.
    """
    counts: list[int] = []
    for module in model.modules():
        projections = routed_projection_names(module)
        if projections is not None:
            up_name, _ = projections
            up = getattr(module, up_name)
            counts.append(int(up.shape[0]))
        elif is_sorted_experts_module(module):
            counts.append(int(cast("torch.Tensor", module.weight).shape[0]))
    return counts


def validate_actor_ep(model: nn.Module, ep: int, world_size: int = 1) -> list[int]:
    """Fail fast when ``ep`` cannot split ``model``'s experts.

    ``ep == 1`` is today's path and always passes. Larger degrees need a
    packed-expert model, a world size divisible by ``ep``, and every expert
    stack divisible by ``ep``.

    :param model: Actor module to check.
    :param ep: Expert-parallel degree.
    :param world_size: Training world size.
    :return: Expert counts of each packed-expert module.
    """
    validate_ep_degree(ep)
    counts = packed_expert_counts(model)
    if ep == 1:
        return counts
    if world_size % ep != 0:
        msg = f"world size {world_size} is not divisible by ep {ep}"
        raise ValueError(msg)
    if not counts:
        msg = f"ep={ep} needs a packed-expert MoE model"
        raise ValueError(msg)
    for count in counts:
        if count % ep != 0:
            msg = f"expert count {count} is not divisible by ep {ep}"
            raise ValueError(msg)
    return counts


@dataclass
class ParallelMesh:
    """DeviceMesh views for HSDP with expert and tensor parallel.

    ``world`` is ``(replicate, shard)``. Weights shard inside one ``shard``
    group and replicate across groups.     ``ep`` and ``tp`` split a shard group
    into consecutive ranks; ``*_replicas`` join the ranks holding the same
    EP or TP shard. ``cp`` is the innermost split of that shard group, so
    context-parallel peers are consecutive and sit inside one expert group.
    EP views are ``None`` at ``ep == 1``, TP views at ``tp == 1``, and the
    CP view at ``cp == 1``. ``dp_mod_ep`` and ``fsdp_experts`` are ``None``
    unless experts also FSDP-shard (``ep`` smaller than the shard group).
    """

    world: DeviceMesh
    hsdp: DeviceMesh
    ep: DeviceMesh | None = None
    dp_mod_ep: DeviceMesh | None = None
    ep_replicas: DeviceMesh | None = None
    fsdp_experts: DeviceMesh | None = None
    tp: DeviceMesh | None = None
    tp_replicas: DeviceMesh | None = None
    cp: DeviceMesh | None = None

    @property
    def leftover_dp(self) -> int:
        """Ranks per shard group that hold the same experts."""
        ep = 1 if self.ep is None else int(self.ep.size())
        return int(self.world.size(1)) // ep

    @property
    def cp_group(self) -> dist.ProcessGroup:
        """Process group over the context-parallel ranks."""
        if self.cp is None:
            msg = "ParallelMesh has no 'cp' axis at cp == 1"
            raise ValueError(msg)
        return self.cp.get_group()

    def sync_grads(
        self, params: Sequence[nn.Parameter], reduce_dtype: torch.dtype
    ) -> None:
        """Average EP and TP shard grads over the ranks that hold each shard.

        An EP owner's grad already sums every EP-group rank's tokens, so the
        sum over replicas divides by the full world. TP ranks share tokens,
        so TP shard grads take a plain mean over replicas. FSDP reduces its
        own shards.

        :param params: Optimizer parameters; dense params are skipped.
        :param reduce_dtype: Dtype of the reduce buffer.
        :raises RuntimeError: A trainable DTensor is on no FSDP, EP or TP mesh.
        """
        fsdp_meshes = [
            mesh for mesh in (self.hsdp, self.fsdp_experts) if mesh is not None
        ]
        reductions: list[tuple[DeviceMesh, DeviceMesh, int, list[nn.Parameter]]] = []
        if self.ep is not None and self.ep_replicas is not None:
            reductions.append((self.ep, self.ep_replicas, self.world.size(), []))
        if self.tp is not None and self.tp_replicas is not None:
            reductions.append((self.tp, self.tp_replicas, self.tp_replicas.size(), []))
        for param in params:
            if not isinstance(param, DTensor) or not param.requires_grad:
                continue
            if param.device_mesh in fsdp_meshes:
                continue
            sharded = next(
                (
                    group
                    for mesh, _, _, group in reductions
                    if param.device_mesh == mesh
                ),
                None,
            )
            if sharded is None:
                msg = (
                    f"Trainable DTensor of shape {tuple(param.shape)} is on mesh "
                    f"{param.device_mesh.mesh_dim_names}, which is not an FSDP, "
                    "EP or TP mesh; its grad would never sync across replicas."
                )
                raise RuntimeError(msg)
            sharded.append(param)
        for _mesh, replicas, divisor, sharded in reductions:
            if sharded:
                all_reduce_grads(
                    sharded,
                    divisor=divisor,
                    group=replicas.get_group(),
                    reduce_dtype=reduce_dtype,
                )


def validate_cp_degree(cp: int) -> None:
    """Reject a non-positive or non-integer context-parallel degree."""
    if not isinstance(cp, int) or isinstance(cp, bool):
        msg = f"cp must be an int, got {type(cp).__name__}"
        raise TypeError(msg)
    if cp < 1:
        msg = f"cp must be >= 1, got {cp}"
        raise ValueError(msg)


def build_parallel_mesh(
    world_size: int | None = None,
    ep: int = 1,
    tp: int = 1,
    cp: int = 1,
    shard_group_size: int | None = None,
    device_type: str | None = None,
) -> ParallelMesh | None:
    """Build HSDP / EP / TP / CP mesh views, or ``None`` for plain FSDP.

    :param world_size: Trainer ranks; defaults to the process-group world.
    :param ep: Expert-parallel degree.
    :param tp: Tensor-parallel degree.
    :param cp: Context-parallel degree. Peers are the innermost consecutive
        ranks of a shard group and share one data shard.
    :param shard_group_size: Ranks per weight-shard group; ``None`` is the world.
    :param device_type: Mesh device type; defaults to CUDA when available.
    :return: Mesh views, or ``None`` for ``ep == tp == cp == 1`` with one
        shard group spanning the world.
    :raises ValueError: ``cp > 1`` combined with ``tp > 1``, or ``ep`` not
        divisible by ``cp``.
    """
    validate_cp_degree(cp)
    if cp > 1 and tp > 1:
        msg = f"cp={cp} is not composed with tp={tp}"
        raise ValueError(msg)
    if ep > 1 and ep % cp != 0:
        msg = (
            f"ep={ep} is not divisible by cp={cp}: the expert group must "
            "contain a whole number of context-parallel ranks."
        )
        raise ValueError(msg)
    if ep == 1 and tp == 1 and cp == 1 and shard_group_size is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        msg = (
            "Expert / tensor parallel and HSDP require an initialised process "
            "group. Launch with torchrun (or set rendezvous env vars) before "
            "building the mesh."
        )
        raise RuntimeError(msg)
    if world_size is None:
        world_size = dist.get_world_size()
    shard = world_size if shard_group_size is None else shard_group_size
    if world_size % shard != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by shard_group_size ({shard})"
        )
        raise ValueError(msg)
    leftover_dp = ep_data_parallel_size(shard, ep)
    tp_groups = tp_data_parallel_size(shard, tp)
    replicate = world_size // shard
    if replicate == 1 and ep == 1 and tp == 1 and cp == 1:
        return None
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    world = init_device_mesh(
        device_type, (replicate, shard), mesh_dim_names=("replicate", "shard")
    )
    mesh = ParallelMesh(world=world, hsdp=world if replicate > 1 else world["shard"])
    # Each unflatten / flatten creates process groups on every rank, so every
    # rank must take the same branches in the same order.
    if cp > 1:
        _attach_context_parallel(mesh, world, shard, ep, cp)
    elif ep > 1:
        mesh.ep, mesh.dp_mod_ep, mesh.ep_replicas = _split_shard_axis(
            world, leftover_dp, ep, ("dp", "ep")
        )
        if mesh.dp_mod_ep is not None:
            # The mesh fully_shard gives EP DTensors it shards on dp_mod_ep.
            mesh.fsdp_experts = DeviceMesh._concatenate([mesh.dp_mod_ep, mesh.ep])
    if tp > 1:
        mesh.tp, _, mesh.tp_replicas = _split_shard_axis(
            world, tp_groups, tp, ("tp_dp", "tp")
        )
    return mesh


def _split_shard_axis(
    world: DeviceMesh, outer: int, inner: int, names: tuple[str, str]
) -> tuple[DeviceMesh, DeviceMesh | None, DeviceMesh]:
    """Split ``world``'s shard axis into ``(outer, inner)`` consecutive ranks.

    No size-1 outer dim is created, and replicas reuse the outer or
    replicate group when one already holds exactly those ranks.

    :param world: ``(replicate, shard)`` root mesh.
    :param outer: Size of the outer split.
    :param inner: Size of the inner split (consecutive ranks).
    :param names: Outer and inner dim names.
    :return: Inner mesh; outer mesh with the replicate dim, ``None`` when
        ``outer == 1``; mesh joining the ranks that hold the same inner shard.
    """
    outer_name, inner_name = names
    if outer == 1:
        split = world._unflatten(1, (inner,), (inner_name,))
        return split[inner_name], None, world["replicate"]
    split = world._unflatten(1, (outer, inner), names)
    if world.size(0) == 1:
        return split[inner_name], split[outer_name], split[outer_name]
    outer_mesh = split["replicate", outer_name]
    return split[inner_name], outer_mesh, outer_mesh._flatten(f"{inner_name}_replicas")


def _attach_context_parallel(
    mesh: ParallelMesh, world: DeviceMesh, shard: int, ep: int, cp: int
) -> None:
    """Put ``cp`` innermost in the shard group, inside the expert group.

    Context-parallel peers stay consecutive, so they share one HSDP replica
    and one data shard. When ``ep > 1`` the expert group is that ``cp`` axis
    plus ``ep / cp`` expert owners. ``ep == cp`` makes the two groups the
    same ranks.

    :param mesh: Mesh being built. CP and, when ``ep > 1``, EP views are set.
    :param world: ``(replicate, shard)`` root mesh.
    :param shard: Ranks in one weight-shard group.
    :param ep: Expert-parallel degree.
    :param cp: Context-parallel degree.
    :raises ValueError: The shard group does not divide by ``cp``.
    """
    if shard % cp != 0:
        msg = f"shard group ({shard}) must be divisible by cp ({cp})"
        raise ValueError(msg)
    if ep <= 1:
        mesh.cp, _, _ = _split_shard_axis(world, shard // cp, cp, ("dp", "cp"))
        return
    dp_in_ep = ep // cp
    dp_mod_ep = shard // ep
    if dp_in_ep == 1:
        mesh.cp, mesh.dp_mod_ep, mesh.ep_replicas = _split_shard_axis(
            world, dp_mod_ep, cp, ("dp", "cp")
        )
        mesh.ep = mesh.cp
        if mesh.dp_mod_ep is not None:
            mesh.fsdp_experts = DeviceMesh._concatenate([mesh.dp_mod_ep, mesh.ep])
        return
    if dp_mod_ep == 1:
        split = world._unflatten(1, (dp_in_ep, cp), ("dp_in_ep", "cp"))
        mesh.cp = split["cp"]
        mesh.ep = split["dp_in_ep", "cp"]._flatten("ep")
        mesh.ep_replicas = world["replicate"]
        return
    split = world._unflatten(
        1, (dp_mod_ep, dp_in_ep, cp), ("dp_mod_ep", "dp_in_ep", "cp")
    )
    mesh.cp = split["cp"]
    mesh.ep = split["dp_in_ep", "cp"]._flatten("ep")
    if world.size(0) == 1:
        mesh.dp_mod_ep = split["dp_mod_ep"]
        mesh.ep_replicas = split["dp_mod_ep"]
    else:
        outer = split["replicate", "dp_mod_ep"]
        mesh.dp_mod_ep = outer
        mesh.ep_replicas = outer._flatten("ep_replicas")
    mesh.fsdp_experts = DeviceMesh._concatenate([mesh.dp_mod_ep, mesh.ep])


def _stash_ep(module: nn.Module, device_mesh: DeviceMesh) -> None:
    object.__setattr__(module, "_ep_group", device_mesh.get_group())
    object.__setattr__(module, "_ep_mesh", device_mesh)
    object.__setattr__(module, "_ep_degree", int(device_mesh.size()))


def _partition_experts_fn(_name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
    """``Shard(0)`` every direct parameter on the EP mesh; stash the EP group."""
    for param_name, param in list(mod.named_parameters(recurse=False)):
        sharded = nn.Parameter(
            distribute_tensor(param.detach(), device_mesh, [Shard(0)]),
            requires_grad=param.requires_grad,
        )
        mod.register_parameter(param_name, sharded)
    _stash_ep(mod, device_mesh)


def shard_experts_on_ep(module: nn.Module, ep_mesh: DeviceMesh | None) -> nn.Module:
    """Place packed expert weights as ``DTensor`` ``Shard(0)`` over ``ep_mesh``."""
    if ep_mesh is None:
        return module
    if getattr(module, "_ep_mesh", None) is not None:
        _partition_experts_fn("", module, ep_mesh)
        return module
    return distribute_module(module, ep_mesh, partition_fn=_partition_experts_fn)


def expert_local_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Dense local expert shard (``to_local`` for ``DTensor``, else identity)."""
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight


@contextmanager
def _gathered_expert_block(module: nn.Module) -> Iterator[bool]:
    """Gather leftover-dp expert shards for the duration of the local kernel.

    The EP-local block is split across the data-parallel pair. The token
    dispatch and the packed-expert kernel both index that whole block.
    Yields whether an FSDP gather happened.
    """
    base = module
    get_base = getattr(module, "get_base_layer", None)
    if callable(get_base):
        base = get_base()
    unshard = getattr(base, "unshard", None)
    reshard = getattr(base, "reshard", None)
    if not callable(unshard) or not callable(reshard):
        yield False
        return
    unshard()
    try:
        yield True
    finally:
        reshard()


def expert_param_bytes_local(module: nn.Module) -> int:
    """Sum of local (sharded) expert parameter bytes on this rank."""
    total = 0
    for param in module.parameters(recurse=True):
        local = expert_local_tensor(param)
        total += local.numel() * local.element_size()
    return total


@dataclass
class TokenDispatchState:
    """Metadata to reverse an EP token all-to-all after local expert compute."""

    input_splits: list[int]
    output_splits: list[int]
    ep_group: dist.ProcessGroup
    ep_degree: int
    num_local_experts: int
    permute_indices: torch.Tensor
    num_tokens_per_local_expert: torch.Tensor


def _all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Synchronous ``all_to_all_single`` returning ``output``."""
    dist.all_to_all_single(
        output,
        input.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


class AllToAllCtx(Protocol):
    output_splits: list[int]
    input_splits: list[int]
    group: dist.ProcessGroup


class _AllToAllVar(torch.autograd.Function):
    """Variable-split all-to-all with autograd."""

    @staticmethod
    def forward(
        ctx: AllToAllCtx,
        input: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        out_rows = sum(output_splits)
        output = input.new_empty((out_rows, *tuple(input.shape[1:])))
        _all_to_all_single(output, input, output_splits, input_splits, group)
        return output

    @staticmethod
    def backward(
        ctx: AllToAllCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_output,) = grad_outputs
        grad_input = grad_output.new_empty(
            (sum(ctx.input_splits), *tuple(grad_output.shape[1:]))
        )
        _all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
        )
        return grad_input, None, None, None


def all_to_all_single_autograd(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Autograd-aware variable-split all-to-all over ``group``."""
    return _AllToAllVar.apply(input, output_splits, input_splits, group)


def exchange_expert_counts(
    counts: torch.Tensor, ep_group: dist.ProcessGroup, ep_degree: int
) -> torch.Tensor:
    """Swap per-block expert row counts with every EP rank in one all-to-all.

    :param counts: ``[blocks, ep_degree * local_experts]`` rows this rank sends.
    :param ep_group: Expert-parallel process group.
    :param ep_degree: Ranks in ``ep_group``.
    :return: Same shape; row ``b`` is block ``b``'s counts this rank receives,
        source-rank major.
    """
    blocks, width = counts.shape
    send = counts.view(blocks, ep_degree, width // ep_degree).transpose(0, 1)
    send = send.contiguous()
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=ep_group)
    return received.transpose(0, 1).reshape(blocks, width)


def _dispatch_state(
    received_counts: torch.Tensor,
    send_list: list[int],
    received_list: list[int],
    ep_group: dist.ProcessGroup,
    ep_degree: int,
    num_local_experts: int,
) -> TokenDispatchState:
    """All-to-all splits and the local-expert-major row order for one block.

    :param received_counts: Source-rank-major counts this rank receives.
    :param send_list: Host copy of the counts this rank sends.
    :param received_list: Host copy of ``received_counts``.
    :param ep_group: Expert-parallel process group.
    :param ep_degree: Ranks in ``ep_group``.
    :param num_local_experts: Experts owned by this rank.
    :return: Dispatch metadata for the block.
    """
    input_splits = [
        sum(send_list[rank * num_local_experts : (rank + 1) * num_local_experts])
        for rank in range(ep_degree)
    ]
    output_splits = [
        sum(received_list[rank * num_local_experts : (rank + 1) * num_local_experts])
        for rank in range(ep_degree)
    ]
    slots = torch.arange(ep_degree * num_local_experts, device=received_counts.device)
    expert_major = (slots % num_local_experts) * ep_degree + slots // num_local_experts
    row_keys = torch.repeat_interleave(
        expert_major, received_counts, output_size=sum(output_splits)
    )
    return TokenDispatchState(
        input_splits=input_splits,
        output_splits=output_splits,
        ep_group=ep_group,
        ep_degree=ep_degree,
        num_local_experts=num_local_experts,
        permute_indices=torch.argsort(row_keys, stable=True),
        num_tokens_per_local_expert=received_counts.view(
            ep_degree, num_local_experts
        ).sum(dim=0),
    )


def _unpermute_from_local_expert_major(
    routed_output: torch.Tensor,
    permute_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Put local-expert-major rows back in all-to-all (source-rank) order."""
    out = routed_output.new_empty((num_rows, *tuple(routed_output.shape[1:])))
    out[permute_indices] = routed_output
    return out


def token_dispatch(
    routed_input: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, TokenDispatchState]:
    """All-to-all tokens to expert-owning ranks; return local-expert-major rows."""
    if num_tokens_per_expert.numel() != ep_degree * num_local_experts:
        msg = (
            f"num_tokens_per_expert length {num_tokens_per_expert.numel()} != "
            f"ep_degree * num_local_experts ({ep_degree} * {num_local_experts})"
        )
        raise ValueError(msg)

    counts = num_tokens_per_expert.view(1, -1)
    received = exchange_expert_counts(counts, ep_group, ep_degree)
    (send_list,), (received_list,) = torch.stack((counts, received)).tolist()
    state = _dispatch_state(
        received[0],
        send_list,
        received_list,
        ep_group=ep_group,
        ep_degree=ep_degree,
        num_local_experts=num_local_experts,
    )
    dispatched = all_to_all_single_autograd(
        routed_input, state.output_splits, state.input_splits, ep_group
    )
    return (
        dispatched[state.permute_indices],
        state.num_tokens_per_local_expert,
        state,
    )


def token_combine(
    routed_output: torch.Tensor,
    state: TokenDispatchState,
) -> torch.Tensor:
    """Reverse :func:`token_dispatch` — unpermute then all-to-all back."""
    num_rows = sum(state.output_splits)
    unpermuted = _unpermute_from_local_expert_major(
        routed_output, state.permute_indices, num_rows
    )
    return all_to_all_single_autograd(
        unpermuted,
        state.input_splits,
        state.output_splits,
        state.ep_group,
    )


class TokenAdapterIds(NamedTuple):
    """Adapter names under mixed fused routing and each token's index into them."""

    names: list[str]
    ids: torch.Tensor


def _token_adapter_ids(
    routing: list[str], n_tokens: int, device: torch.device
) -> TokenAdapterIds:
    """Expand fused routing (one adapter per batch row) to per-token adapter ids.

    :param routing: Adapter name per batch row.
    :param n_tokens: Leading dimension of the experts input.
    :param device: Device for the id tensor.
    :return: Adapter names and each token's index into them.
    """
    names = list(dict.fromkeys(routing))
    name_to_id = {name: index for index, name in enumerate(names)}
    factor, remainder = divmod(n_tokens, len(routing))
    if remainder:
        msg = (
            f"Fused adapter routing covers {len(routing)} rows but the "
            f"experts input's leading dimension is {n_tokens}."
        )
        raise ValueError(msg)
    ids = torch.tensor(
        [name_to_id[name] for name in routing for _ in range(factor)],
        device=device,
        dtype=torch.long,
    )
    return TokenAdapterIds(names, ids)


def _dispatch_adapter_ids(
    sorted_ids: torch.Tensor, state: TokenDispatchState
) -> torch.Tensor:
    """Send expert-sorted adapter ids with the tokens; local-expert-major result."""
    dispatched = sorted_ids.new_empty((sum(state.output_splits),))
    dist.all_to_all_single(
        dispatched,
        sorted_ids.contiguous(),
        state.output_splits,
        state.input_splits,
        group=state.ep_group,
    )
    return dispatched[state.permute_indices]


def module_ep_degree(module: nn.Module) -> int:
    """EP degree stashed by :func:`shard_experts_on_ep`, else ``1``."""
    return int(getattr(module, "_ep_degree", 1) or 1)


def iter_packed_expert_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Named modules that match the sorted or routed packed-expert layouts."""
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if is_sorted_experts_module(module) or is_routed_experts_module(module):
            found.append((name, module))
    return found


def num_packed_experts(module: nn.Module) -> int:
    """Stacked expert axis of a packed-expert module (global ``DTensor`` shape)."""
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 3:
        return int(weight.shape[0])
    for name in ("gate_up_proj", "up_proj", "down_proj"):
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            return int(tensor.shape[0])
    msg = "Packed-expert module has no stacked 3D expert weight."
    raise ValueError(msg)


def packed_expert_count(model: nn.Module) -> int | None:
    """Stacked expert count on the first packed-expert module, if any."""
    modules = iter_packed_expert_modules(model)
    if not modules:
        return None
    return num_packed_experts(modules[0][1])


def assert_packed_experts_ep_sharded(
    model: nn.Module,
    ep: int,
    modules: list[nn.Module] | None = None,
) -> None:
    """Raise if packed-expert weights are not local ``E/ep`` shards."""
    if ep <= 1:
        return
    named: list[tuple[str, nn.Module]]
    if modules is None:
        named = iter_packed_expert_modules(model)
    else:
        named = [(type(module).__name__, module) for module in modules]
    if not named:
        msg = "ep > 1 requires packed expert modules with local E/ep shards."
        raise RuntimeError(msg)
    for name, module in named:
        for param_name, param in module.named_parameters(recurse=False):
            if not isinstance(param, torch.Tensor) or param.ndim != 3:
                continue
            local = expert_local_tensor(param)
            global_e = int(param.shape[0])
            if local.shape[0] * ep != global_e:
                msg = (
                    f"{name}.{param_name} local expert dim {local.shape[0]} "
                    f"* ep {ep} != global {global_e}"
                )
                raise RuntimeError(msg)


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
    _stash_ep(linear, ep_mesh)


def _ep_param_grad_hook(param: nn.Parameter) -> Callable[[torch.Tensor], None]:
    """Accumulate a local shard grad onto an EP ``DTensor`` parameter.

    Local views of a ``DTensor`` cannot carry ``.grad`` back to it, so the
    detached leaf used for the local kernel copies its grad here as a
    same-placed ``DTensor``. Grad accumulation across micro-batches sums.
    """

    def hook(grad: torch.Tensor) -> None:
        assert isinstance(param, DTensor)
        shard = DTensor.from_local(
            grad, param.device_mesh, param.placements, run_check=False
        )
        param.grad = shard if param.grad is None else param.grad + shard

    return hook


def _local_param_dict(
    module: nn.Module, gathered: bool = False
) -> dict[str, torch.Tensor]:
    """Plain-tensor views of a module's params for the EP-blind local kernel.

    Frozen ``DTensor`` params pass as ``to_local()`` views. When ``gathered``,
    ``reshard()`` frees their storage before backward reads it, so they are
    copied. Trainable ones become detached leaf copies; a hook copies each
    leaf grad back onto the sharded parameter (see
    :func:`_ep_param_grad_hook`). Dense params pass through untouched.

    :param module: Module whose parameters the local kernel reads.
    :param gathered: Whether the parameters are an FSDP-gathered block.
    :return: Parameter and buffer tensors keyed by name.
    """
    params: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if not isinstance(param, DTensor):
            params[name] = param
        elif param.requires_grad:
            leaf = param.to_local().detach().clone().requires_grad_(True)
            leaf.register_hook(_ep_param_grad_hook(param))
            params[name] = leaf
        elif gathered:
            params[name] = param.to_local().clone()
        else:
            params[name] = param.to_local()
    params.update(dict(module.named_buffers()))
    return params


def _call_with_local_params(
    module: nn.Module,
    original: Callable[..., torch.Tensor],
    params: dict[str, torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``original`` as ``module.forward`` on ``params`` from :func:`_local_param_dict`."""
    saved = module.forward
    # Module.__call__ does not inject self into ``forward``; keep the bound method.
    module.forward = original
    try:
        return functional_call(module, params, args, kwargs)
    finally:
        module.forward = saved


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


def scatter_scaled_expert_rows(
    combined: torch.Tensor,
    router_weights: torch.Tensor,
    token_idx: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Scale expert-sorted rows and scatter them to token order in fp32."""
    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    _add_scaled_expert_rows(result, combined, router_weights, token_idx)
    return result.to(dtype=hidden_states.dtype)


def _add_scaled_expert_rows(
    result: torch.Tensor,
    combined: torch.Tensor,
    router_weights: torch.Tensor,
    token_idx: torch.Tensor,
) -> None:
    """Add router-scaled expert rows into the fp32 ``result`` at ``token_idx``."""
    router_weights = router_weights.to(dtype=torch.float32)
    # A full scaled copy of the combine buffer does not fit beside it.
    row_bytes = combined.shape[-1] * result.element_size()
    chunk_rows = max(1, GROUPED_LINEAR_CHUNK_BYTES // row_bytes)
    # Empty ``combined`` still adds one node, so its all-to-all runs in backward.
    for start in range(0, max(combined.shape[0], 1), chunk_rows):
        stop = min(start + chunk_rows, combined.shape[0])
        piece = (
            combined[start:stop].to(dtype=torch.float32) * router_weights[start:stop]
        )
        result.index_add_(0, token_idx[start:stop], piece)


def replicated_row_span(n_rows: int, group: dist.ProcessGroup) -> tuple[int, int]:
    """This rank's ``[start, stop)`` share of ``n_rows`` rows replicated over ``group``."""
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    return n_rows * rank // size, n_rows * (rank + 1) // size


def _all_gather_row_spans(
    part: torch.Tensor, n_rows: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Concatenate every rank's :func:`replicated_row_span` rows of ``n_rows``."""
    size = dist.get_world_size(group)
    spans = list(pairwise([n_rows * rank // size for rank in range(size + 1)]))
    width = max(stop - start for start, stop in spans)
    padded = part.new_zeros((width, *part.shape[1:]))
    padded[: part.shape[0]] = part
    pieces = [torch.empty_like(padded) for _ in range(size)]
    dist.all_gather(pieces, padded, group=group)
    return torch.cat(
        [
            piece[: stop - start]
            for piece, (start, stop) in zip(pieces, spans, strict=True)
        ]
    )


class ReplicatedRowsCtx(Protocol):
    group: dist.ProcessGroup
    n_rows: int


class SliceReplicatedRows(torch.autograd.Function):
    """Keep this rank's row span of a tensor replicated over ``group``.

    Backward all-gathers the span gradients and divides by the group size,
    which undoes the scale in :class:`GatherReplicatedRows`.
    """

    @staticmethod
    def forward(
        ctx: ReplicatedRowsCtx,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.n_rows = int(tensor.shape[0])
        start, stop = replicated_row_span(ctx.n_rows, group)
        return tensor[start:stop]

    @staticmethod
    def backward(
        ctx: ReplicatedRowsCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        full = _all_gather_row_spans(grad_output.contiguous(), ctx.n_rows, ctx.group)
        return full / dist.get_world_size(ctx.group), None


class GatherReplicatedRows(torch.autograd.Function):
    """All-gather row spans into the ``n_rows`` tensor replicated over ``group``.

    Every rank in ``group`` holds the same output gradient, and EP expert
    grads count each of those ranks' copy of a token (see
    :meth:`ParallelMesh.sync_grads`). Backward keeps this rank's span scaled
    by the group size so each token still counts once per rank.
    """

    @staticmethod
    def forward(
        ctx: ReplicatedRowsCtx,
        part: torch.Tensor,
        n_rows: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.n_rows = n_rows
        return _all_gather_row_spans(part.contiguous(), n_rows, group)

    @staticmethod
    def backward(
        ctx: ReplicatedRowsCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
        start, stop = replicated_row_span(ctx.n_rows, ctx.group)
        return grad_output[start:stop] * dist.get_world_size(ctx.group), None, None


@contextmanager
def _on_comm_stream(comm: torch.cuda.Stream | None) -> Iterator[None]:
    """Run the body on ``comm`` after the current stream's queued work."""
    if comm is None:
        yield
        return
    comm.wait_stream(torch.cuda.current_stream(comm.device))
    with torch.cuda.stream(comm):
        yield


def _hand_over(
    comm: torch.cuda.Stream | None,
    sent: Sequence[torch.Tensor],
    received: Sequence[torch.Tensor],
) -> torch.cuda.Event | None:
    """Tie ``sent`` to ``comm`` and ``received`` to the current stream.

    :return: Event that marks ``comm``'s queued work, or ``None`` without a side stream.
    """
    if comm is None:
        return None
    for tensor in sent:
        tensor.record_stream(comm)
    current = torch.cuda.current_stream(comm.device)
    for tensor in received:
        tensor.record_stream(current)
    return comm.record_event()


def _wait_for(event: torch.cuda.Event | None) -> None:
    if event is not None:
        torch.cuda.current_stream().wait_event(event)


class DispatchedBlock(NamedTuple):
    """A token block sent to the ranks that own its experts."""

    token_idx: torch.Tensor
    state: TokenDispatchState
    rows: torch.Tensor
    adapter_ids: torch.Tensor | None
    arrived: torch.cuda.Event | None


class CombinedBlock(NamedTuple):
    """A token block's expert output, back on this rank in expert-sorted order."""

    rows: torch.Tensor
    token_idx: torch.Tensor
    order: torch.Tensor
    arrived: torch.cuda.Event | None


@dataclass(frozen=True)
class LocalExperts:
    """An EP-blind expert ``forward`` bound to this rank's local parameters."""

    module: nn.Module
    forward: Callable[..., torch.Tensor]
    params: dict[str, torch.Tensor]
    kwargs: dict[str, Any]


def _base_experts(module: nn.Module) -> nn.Module:
    """The packed-experts module under a PEFT wrapper, or ``module`` itself."""
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


def _routed_up_weight(module: nn.Module) -> torch.Tensor:
    experts = _base_experts(module)
    projections = routed_projection_names(experts)
    if projections is None:
        msg = "Routed experts module does not match a supported packed layout."
        raise RuntimeError(msg)
    return getattr(experts, projections[0])


def _sorted_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(_base_experts(module), "weight", None)
    if not isinstance(weight, torch.Tensor):
        msg = "Sorted experts module has no stacked weight."
        raise RuntimeError(msg)
    return weight


def _sort_token_blocks(
    top_k_index: torch.Tensor, n_tokens: int, token_blocks: int, num_experts: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Expert-sorted flat-row order and ``[blocks, experts]`` row counts of each token block."""
    top_k = top_k_index.shape[-1]
    flat_experts = top_k_index.reshape(-1)
    bounds = [n_tokens * block // token_blocks for block in range(token_blocks + 1)]
    orders: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for start, stop in pairwise(bounds):
        block_experts = flat_experts[start * top_k : stop * top_k]
        orders.append(torch.argsort(block_experts, stable=True) + start * top_k)
        counts.append(torch.bincount(block_experts, minlength=num_experts))
    return orders, torch.stack(counts).to(torch.long)


def _dispatch_block(
    hidden_states: torch.Tensor,
    order: torch.Tensor,
    top_k: int,
    state: TokenDispatchState,
    adapter_ids: torch.Tensor | None,
    comm: torch.cuda.Stream | None,
) -> DispatchedBlock:
    """All-to-all one block's routed rows, and their adapter ids, to the expert owners."""
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    routed = hidden_states[token_idx]
    sent = [routed]
    local_ids = None
    with _on_comm_stream(comm):
        rows = all_to_all_single_autograd(
            routed, state.output_splits, state.input_splits, state.ep_group
        )
        if adapter_ids is not None:
            sorted_ids = adapter_ids[token_idx]
            sent += [sorted_ids, state.permute_indices]
            local_ids = _dispatch_adapter_ids(sorted_ids, state)
    arrived = [rows] if local_ids is None else [rows, local_ids]
    return DispatchedBlock(
        token_idx, state, rows, local_ids, _hand_over(comm, sent, arrived)
    )


@contextmanager
def _routing_override(module: nn.Module, routing: list[str] | None) -> Iterator[None]:
    """Route ``module``'s adapters by ``routing`` for the body; ``None`` keeps its routing."""
    if routing is None:
        yield
        return
    previous = ROUTING_STATE.get(module)
    ROUTING_STATE[module] = routing
    try:
        yield
    finally:
        if previous is None:
            ROUTING_STATE.pop(module, None)
        else:
            ROUTING_STATE[module] = previous


def _run_local_experts(
    experts: LocalExperts,
    state: TokenDispatchState,
    rows: torch.Tensor,
    local_ids: torch.Tensor | None,
    adapter_names: list[str],
    index_dtype: torch.dtype,
) -> torch.Tensor:
    """Run this rank's experts on a dispatched block, one expert per row in local-expert order."""
    local_rows = rows[state.permute_indices]
    n_rows = local_rows.shape[0]
    if n_rows == 0:
        # Peers with rows run the combine all-to-all in backward; tie
        # this rank's empty output to the same grad inputs so it joins.
        trainable = [p.sum() for p in experts.params.values() if p.requires_grad]
        return local_rows + 0 * sum(trainable, local_rows.sum())
    expert_index = torch.repeat_interleave(
        torch.arange(state.num_local_experts, device=rows.device, dtype=index_dtype),
        state.num_tokens_per_local_expert,
        output_size=n_rows,
    ).unsqueeze(-1)
    ones = torch.ones(n_rows, 1, dtype=rows.dtype, device=rows.device)
    # Mixed routing (e.g. PPO actor + critic rows) is per batch row; the
    # local wrapper needs routing aligned to its local rows.
    routing = (
        None if local_ids is None else [adapter_names[i] for i in local_ids.tolist()]
    )
    with _routing_override(experts.module, routing):
        return _call_with_local_params(
            experts.module,
            experts.forward,
            experts.params,
            local_rows,
            expert_index,
            ones,
            **experts.kwargs,
        )


def _scatter_block(
    result: torch.Tensor, combined: CombinedBlock, flat_weights: torch.Tensor
) -> None:
    """Add a combined block's router-scaled rows into ``result`` once they arrive."""
    _wait_for(combined.arrived)
    router_weights = flat_weights[combined.order].unsqueeze(-1)
    _add_scaled_expert_rows(result, combined.rows, router_weights, combined.token_idx)


def _routed_ep_blocks(
    module: nn.Module,
    inner: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    adapter_ids: TokenAdapterIds | None,
    ep_group: dist.ProcessGroup,
    ep_degree: int,
    token_blocks: int,
    expert_kwargs: dict[str, Any],
    comm: torch.cuda.Stream | None,
) -> torch.Tensor:
    """Dispatch, run local experts, and combine ``hidden_states`` in token blocks.

    One all-to-all swaps every block's expert counts. Block ``b + 1``'s
    dispatch runs on ``comm`` while block ``b``'s experts run on the current
    stream, and block ``b - 1`` scatters while block ``b``'s combine is in
    flight. With ``comm=None`` every step runs in order.

    :param module: EP-sharded routed experts, or their outer LoRA wrapper.
    :param inner: The module's own forward, run on this rank's expert rows.
    :param hidden_states: ``[tokens, hidden]`` experts input.
    :param top_k_index: ``[tokens, top_k]`` global expert ids.
    :param top_k_weights: ``[tokens, top_k]`` router weights.
    :param adapter_ids: Adapter names and per-token ids under mixed fused routing.
    :param ep_group: Expert-parallel process group.
    :param ep_degree: Ranks in ``ep_group``.
    :param token_blocks: Token blocks per call.
    :param expert_kwargs: Extra keyword arguments for ``inner``.
    :param comm: Side stream for the token all-to-alls, or ``None``.
    :return: ``[tokens, hidden]`` routed expert output.
    """
    with _gathered_expert_block(module) as gathered:
        local_e = expert_local_tensor(_routed_up_weight(module)).shape[0]
        top_k = top_k_index.shape[-1]
        orders, send = _sort_token_blocks(
            top_k_index, hidden_states.shape[0], token_blocks, local_e * ep_degree
        )
        received = exchange_expert_counts(send, ep_group, ep_degree)
        send_lists, received_lists = torch.stack((send, received)).tolist()
        experts = LocalExperts(
            module, inner, _local_param_dict(module, gathered), expert_kwargs
        )
        token_ids = None if adapter_ids is None else adapter_ids.ids
        names = [] if adapter_ids is None else adapter_ids.names
        flat_weights = top_k_weights.reshape(-1)
        result = torch.zeros_like(hidden_states, dtype=torch.float32)

        def dispatch(block: int) -> DispatchedBlock:
            state = _dispatch_state(
                received[block],
                send_lists[block],
                received_lists[block],
                ep_group,
                ep_degree,
                local_e,
            )
            return _dispatch_block(
                hidden_states, orders[block], top_k, state, token_ids, comm
            )

        pending = dispatch(0)
        previous: CombinedBlock | None = None
        for block in range(token_blocks):
            token_idx, state, rows, local_ids, arrived = pending
            if block + 1 < token_blocks:
                pending = dispatch(block + 1)
            _wait_for(arrived)
            expert_out = _run_local_experts(
                experts, state, rows, local_ids, names, top_k_index.dtype
            )
            del rows
            unpermuted = _unpermute_from_local_expert_major(
                expert_out, state.permute_indices, sum(state.output_splits)
            )
            del expert_out
            with _on_comm_stream(comm):
                combined = all_to_all_single_autograd(
                    unpermuted, state.input_splits, state.output_splits, ep_group
                )
            returned = _hand_over(comm, [unpermuted], [combined])
            del unpermuted
            if previous is not None:
                _scatter_block(result, previous, flat_weights)
            previous = CombinedBlock(combined, token_idx, orders[block], returned)
        assert previous is not None
        _scatter_block(result, previous, flat_weights)
        return result.to(dtype=hidden_states.dtype)


def _comm_stream(
    streams: dict[torch.device, torch.cuda.Stream], hidden_states: torch.Tensor
) -> torch.cuda.Stream | None:
    """Side stream for the token all-to-alls on CUDA inputs, one per device."""
    if not hidden_states.is_cuda:
        return None
    device = hidden_states.device
    if device not in streams:
        streams[device] = torch.cuda.Stream(device)
    return streams[device]


def _route_replicated_span(
    run_blocks: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    adapter_ids: TokenAdapterIds | None,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Route this rank's :func:`replicated_row_span` of tokens and all-gather every span's output."""
    n_tokens = hidden_states.shape[0]
    start, stop = replicated_row_span(n_tokens, tp_group)
    if adapter_ids is not None:
        adapter_ids = TokenAdapterIds(adapter_ids.names, adapter_ids.ids[start:stop])
    part = run_blocks(
        SliceReplicatedRows.apply(hidden_states, tp_group),
        top_k_index[start:stop],
        SliceReplicatedRows.apply(top_k_weights, tp_group),
        adapter_ids,
    )
    return GatherReplicatedRows.apply(part, n_tokens, tp_group)


def _install_routed_ep_forward(
    module: nn.Module, token_blocks: int, tp_group: dist.ProcessGroup | None
) -> None:
    """Dispatch → local experts → combine, in ``token_blocks`` token blocks.

    :param module: Routed experts module, or its outer LoRA wrapper.
    :param token_blocks: Token blocks per call; see :func:`_routed_ep_blocks`.
    :param tp_group: Ranks holding the same tokens, or ``None``. Each routes
        only its :func:`replicated_row_span` and the outputs are all-gathered.
    """
    if getattr(module, "_agilerl_ep_forward", False):
        return
    inner = module.forward
    # Local rows reach the kernel one expert per row, in expert order.
    expert_kwargs: dict[str, Any] = (
        {"already_grouped": True}
        if isinstance(module, RoutedExpertsLoraWrapper)
        else {}
    )
    comm_streams: dict[torch.device, torch.cuda.Stream] = {}

    def ep_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        ep_degree = module_ep_degree(self)
        ep_group = getattr(self, "_ep_group", None)
        if (
            args
            or kwargs
            or hidden_states.dim() != 2
            or ep_degree <= 1
            or ep_group is None
        ):
            return inner(hidden_states, top_k_index, top_k_weights, *args, **kwargs)
        routing = mixed_routing(self)
        adapter_ids = (
            None
            if routing is None
            else _token_adapter_ids(
                routing, hidden_states.shape[0], hidden_states.device
            )
        )
        comm = _comm_stream(comm_streams, hidden_states) if token_blocks > 1 else None
        run_blocks = partial(
            _routed_ep_blocks,
            self,
            inner,
            ep_group=ep_group,
            ep_degree=ep_degree,
            token_blocks=token_blocks,
            expert_kwargs=expert_kwargs,
            comm=comm,
        )
        if tp_group is None:
            return run_blocks(hidden_states, top_k_index, top_k_weights, adapter_ids)
        return _route_replicated_span(
            run_blocks, hidden_states, top_k_index, top_k_weights, adapter_ids, tp_group
        )

    module.forward = MethodType(ep_forward, module)
    object.__setattr__(module, "_agilerl_ep_forward", True)


def _install_sorted_ep_forward(module: nn.Module) -> None:
    """Dispatch sorted rows → existing grouped kernel with local counts → combine."""
    if getattr(module, "_agilerl_ep_forward", False):
        return
    inner = module.forward

    def ep_forward(
        self: nn.Module,
        inputs: torch.Tensor,
        expert_size: Sequence[int] | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        ep_degree = module_ep_degree(self)
        ep_group = getattr(self, "_ep_group", None)
        if ep_degree <= 1 or ep_group is None:
            return inner(inputs, expert_size, *args, **kwargs)
        with _gathered_expert_block(self) as gathered:
            local_e = expert_local_tensor(_sorted_weight(self)).shape[0]
            if isinstance(expert_size, torch.Tensor):
                counts = expert_size.to(dtype=torch.long)
            else:
                counts = torch.as_tensor(list(expert_size), dtype=torch.long)
            local_rows, local_counts, state = token_dispatch(
                inputs,
                counts,
                ep_group=ep_group,
                ep_degree=ep_degree,
                num_local_experts=local_e,
            )
            params = _local_param_dict(self, gathered=gathered)
            return token_combine(
                _call_with_local_params(self, inner, params, local_rows, local_counts),
                state,
            )

    module.forward = MethodType(ep_forward, module)
    object.__setattr__(module, "_agilerl_ep_forward", True)


def _install_ep_forward(
    module: nn.Module, token_blocks: int, tp_group: dist.ProcessGroup | None
) -> None:
    experts = _base_experts(module)
    if is_routed_experts_module(experts):
        _install_routed_ep_forward(module, token_blocks, tp_group)
    elif is_sorted_experts_module(experts):
        _install_sorted_ep_forward(module)


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
    _stash_ep(wrapper, ep_mesh)
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
        Routed experts then dispatch each token from one of those ranks.
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
        _install_ep_forward(wrappers[0] if wrappers else module, token_blocks, tp_group)
        count += 1
    if count:
        patch_lora_for_fused_forward(model)
    return count


def reference_dispatch_combine(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    ep_degree: int,
    num_experts: int,
) -> torch.Tensor:
    """Identity round-trip for dispatch/combine ordering in a single process."""
    if num_experts % ep_degree != 0:
        msg = f"num_experts ({num_experts}) must be divisible by ep ({ep_degree})"
        raise ValueError(msg)
    order = torch.argsort(expert_ids, stable=True)
    sorted_tokens = tokens[order]
    counts = torch.bincount(expert_ids, minlength=num_experts)
    num_local = num_experts // ep_degree
    pieces: list[torch.Tensor] = []
    for rank in range(ep_degree):
        start = rank * num_local
        end = start + num_local
        offsets = [int(counts[:start].sum().item()), int(counts[:end].sum().item())]
        pieces.append(sorted_tokens[offsets[0] : offsets[1]])
    restored_sorted = torch.cat(pieces, dim=0) if pieces else sorted_tokens
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return restored_sorted[inverse]
