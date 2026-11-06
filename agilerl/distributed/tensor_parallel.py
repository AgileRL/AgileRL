# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel dense layers on the tensor-parallel mesh.

Column-parallel weights are ``Shard(0)``. Row-parallel weights are ``Shard(1)``
and their outputs are all-reduced. Key/value heads that are shared by several
ranks stay whole on every rank; each rank's forward reads its own rows.

A block's hidden-state input enters through :class:`CopyToTPRegion` and its
output leaves through :class:`ReduceFromTPRegion`, so forward and backward
match the unsharded block. Grads of replicated params inside a region match
once :func:`sum_tp_partial_grads` sums them at the optimizer step.

Family-specific blocks are sharded by each catalog
:class:`~agilerl.architectures.runtime.TensorParallelPlan`, built from the
layers here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Protocol, cast

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.func import functional_call

from agilerl.architectures import family_tensor_parallel_plans
from agilerl.distributed.expert_parallel.dispatch import SAVED_COLLECTIVE_SCOPE
from agilerl.distributed.expert_parallel.forward import local_param_dict
from agilerl.distributed.mesh import MeshAxisName
from agilerl.distributed.process import all_reduce_grads


def stash_tp(module: nn.Module, device_mesh: DeviceMesh) -> None:
    """Record the tensor-parallel group, mesh, and degree on ``module``."""
    object.__setattr__(module, "_tp_group", device_mesh.get_group())
    object.__setattr__(module, "_tp_mesh", device_mesh)
    object.__setattr__(module, "_tp_degree", int(device_mesh.size()))


def tp_degree(module: nn.Module) -> int:
    """Tensor-parallel degree :func:`stash_tp` recorded on ``module``; 1 when unsharded."""
    return int(module.__dict__.get("_tp_degree", 1))


def base_linear(module: nn.Module) -> nn.Module:
    """The Linear under a PEFT wrapper, or ``module`` itself."""
    get_base = getattr(module, "get_base_layer", None)
    if callable(get_base):
        base = get_base()
        if isinstance(base, nn.Module) and hasattr(base, "weight"):
            return base
    base = getattr(module, "base_layer", None)
    if isinstance(base, nn.Module) and hasattr(base, "weight"):
        return base
    return module


def base_weight(module: nn.Module) -> nn.Parameter:
    """Weight of :func:`base_linear` of ``module``."""
    return base_linear(module).get_parameter("weight")


def is_linear(module: nn.Module) -> bool:
    """Whether :func:`base_linear` of ``module`` is an ``nn.Linear``."""
    return isinstance(base_linear(module), nn.Linear)


def replicate_on_tp(module: nn.Module, mesh: DeviceMesh, *names: str) -> None:
    """Make ``module``'s own params ``Replicate()`` DTensors on the TP mesh.

    :param module: Module that directly owns the parameters.
    :param mesh: Tensor-parallel mesh.
    :param names: Parameter names on ``module``.
    """
    for name in names:
        param = getattr(module, name, None)
        if not isinstance(param, nn.Parameter) or isinstance(param, DTensor):
            continue
        replicated = distribute_tensor(param.detach(), mesh, [Replicate()])
        module.register_parameter(
            name, nn.Parameter(replicated, requires_grad=param.requires_grad)
        )


def is_tp_replicated(param: torch.Tensor) -> bool:
    """Whether ``param`` is a whole copy on every rank of its mesh."""
    return isinstance(param, DTensor) and all(
        placement.is_replicate() for placement in param.placements
    )


def on_tp_mesh(param: DTensor) -> bool:
    """Whether ``param`` is sharded on a tensor-parallel axis, alone or under FSDP."""
    return MeshAxisName.TP in (param.device_mesh.mesh_dim_names or ())


def permute_rows(param: DTensor, order: torch.Tensor) -> None:
    """Reorder ``param``'s global dim-0 rows by ``order`` in place.

    :param param: Sharded parameter on any placements over its mesh.
    :param order: Source row of each output row.
    """
    full = param.full_tensor()
    ordered = full.index_select(0, order.to(full.device))
    local = distribute_tensor(
        ordered, param.device_mesh, param.placements, src_data_rank=None
    ).to_local()
    param.to_local().copy_(local)


def require_divisible(size: int, divisor: int, label: str, axis: str = "tp") -> None:
    """Raise unless ``label``'s ``size`` splits evenly over the ``axis`` degree ``divisor``."""
    if size % divisor != 0:
        msg = f"{label} ({size}) must be divisible by {axis} ({divisor})"
        raise ValueError(msg)


def register_shard(
    module: nn.Module, name: str, param: nn.Parameter, sharded: DTensor
) -> None:
    """Replace ``module.<name>`` with ``sharded``, keeping ``param``'s ``requires_grad``."""
    module.register_parameter(
        name, nn.Parameter(sharded, requires_grad=param.requires_grad)
    )


def shard_reordered_rows(
    module: nn.Module,
    name: str,
    row_order: torch.Tensor,
    device_mesh: DeviceMesh,
) -> None:
    """Reorder ``module.<name>`` rows by ``row_order``, then ``Shard(0)`` it on ``device_mesh``."""
    param = getattr(module, name)
    if not isinstance(param, nn.Parameter) or isinstance(param, DTensor):
        return
    if param.shape[0] != row_order.numel():
        msg = (
            f"{type(module).__name__}.{name} dim 0 is {param.shape[0]}, "
            f"expected {row_order.numel()}"
        )
        raise ValueError(msg)
    index = row_order
    if param.device.type != "meta":
        index = row_order.to(device=param.device)
    ordered = param.detach().index_select(0, index).contiguous()
    sharded = distribute_tensor(ordered, device_mesh, [Shard(0)])
    register_shard(module, name, param, sharded)


def shard_dim0(module: nn.Module, name: str, device_mesh: DeviceMesh) -> None:
    """``Shard(0)`` ``module.<name>`` on ``device_mesh``."""
    param = getattr(module, name)
    if not isinstance(param, nn.Parameter) or isinstance(param, DTensor):
        return
    require_divisible(int(param.shape[0]), int(device_mesh.size()), name)
    sharded = distribute_tensor(param.detach(), device_mesh, [Shard(0)])
    register_shard(module, name, param, sharded)


def _shard_weight(module: nn.Module, mesh: DeviceMesh, placement: Shard) -> None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter) or isinstance(weight, DTensor):
        return
    dim = int(placement.dim)
    require_divisible(int(weight.shape[dim]), int(mesh.size()), f"weight dim {dim}")
    sharded = distribute_tensor(weight.detach(), mesh, [placement])
    module.register_parameter(
        "weight",
        nn.Parameter(sharded, requires_grad=weight.requires_grad),
    )
    stash_tp(module, mesh)


def _shard_column_bias(module: nn.Module, mesh: DeviceMesh) -> None:
    bias = getattr(module, "bias", None)
    if not isinstance(bias, nn.Parameter) or isinstance(bias, DTensor):
        return
    require_divisible(int(bias.shape[0]), int(mesh.size()), "bias")
    sharded = distribute_tensor(bias.detach(), mesh, [Shard(0)])
    module.register_parameter(
        "bias",
        nn.Parameter(sharded, requires_grad=bias.requires_grad),
    )


def _lora_linears(wrapper: nn.Module, name: str) -> list[nn.Module]:
    bank = getattr(wrapper, name, None)
    if isinstance(bank, nn.ModuleDict):
        return list(bank.values())
    if isinstance(bank, dict):
        return [module for module in bank.values() if isinstance(module, nn.Module)]
    return []


def keep_lora_bank(wrapper: nn.Module, name: str, mesh: DeviceMesh) -> None:
    """Keep a LoRA ``lora_A`` / ``lora_B`` bank whole on every TP rank."""
    for linear in _lora_linears(wrapper, name):
        replicate_on_tp(linear, mesh, "weight")


def read_rows(
    module: nn.Module, mesh: DeviceMesh, ranges: tuple[tuple[int, int], ...]
) -> None:
    """Keep ``module``'s params whole on every TP rank; its region reads ``ranges``.

    :param module: Module whose weight (and bias) rows are selected in forward.
    :type module: nn.Module
    :param mesh: Tensor-parallel mesh.
    :type mesh: DeviceMesh
    :param ranges: ``(start, stop)`` row ranges of this rank, in local order.
    :type ranges: tuple[tuple[int, int], ...]
    """
    replicate_on_tp(module, mesh, "weight", "bias")
    object.__setattr__(module, "tp_row_ranges", ranges)


def read_lora_b_rows(
    wrapper: nn.Module, mesh: DeviceMesh, ranges: tuple[tuple[int, int], ...]
) -> None:
    """Keep LoRA B whole on every TP rank; its region reads ``ranges``.

    :param wrapper: PEFT LoRA layer.
    :type wrapper: nn.Module
    :param mesh: Tensor-parallel mesh.
    :type mesh: DeviceMesh
    :param ranges: ``(start, stop)`` row ranges of this rank, in local order.
    :type ranges: tuple[tuple[int, int], ...]
    """
    for linear in _lora_linears(wrapper, "lora_B"):
        read_rows(linear, mesh, ranges)


def shard_lora_b_rows(
    wrapper: nn.Module, row_order: torch.Tensor, mesh: DeviceMesh
) -> None:
    """``Shard(0)`` LoRA B in the base weight's ``row_order`` on ``mesh``.

    :param wrapper: PEFT LoRA layer whose base weight is stored in ``row_order``.
    :type wrapper: nn.Module
    :param row_order: Source row of each stored row.
    :type row_order: torch.Tensor
    :param mesh: Tensor-parallel mesh.
    :type mesh: DeviceMesh
    """
    for linear in _lora_linears(wrapper, "lora_B"):
        shard_reordered_rows(linear, "weight", row_order, mesh)


def is_lora_b_weight(relative: str, linear: str) -> bool:
    """Whether ``relative`` names a LoRA B weight of the PEFT layer ``linear``.

    :param relative: Parameter name relative to the layer's owner.
    :type relative: str
    :param linear: Attribute name of the PEFT layer on that owner.
    :type linear: str
    :return: ``True`` for ``<linear>.lora_B.<adapter>.weight``.
    :rtype: bool
    """
    head, found, adapter = relative.partition(".lora_B.")
    return (
        bool(found)
        and head == linear
        and adapter.endswith(".weight")
        and adapter.count(".") == 1
    )


class SharedSeedDropout(nn.Module):
    """Dropout that draws the same mask on every rank of a tensor-parallel group."""

    def __init__(self, p: float, seed: int) -> None:
        super().__init__()
        self.p = p
        self.seed = seed
        self.generators: dict[torch.device, torch.Generator] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero entries of ``x`` with probability ``p`` and rescale the rest."""
        if not self.training:
            return x
        generator = self.generators.get(x.device)
        if generator is None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(self.seed)
            self.generators[x.device] = generator
        keep = torch.empty_like(x).bernoulli_(1 - self.p, generator=generator)
        return x * keep / (1 - self.p)


def share_lora_dropout(wrapper: nn.Module, mesh: DeviceMesh) -> None:
    """Give LoRA dropout on a replicated input one mask across ``mesh``.

    :param wrapper: PEFT LoRA layer whose input is the same on every rank.
    :type wrapper: nn.Module
    :param mesh: Tensor-parallel mesh.
    :type mesh: DeviceMesh
    """
    bank = getattr(wrapper, "lora_dropout", None)
    if not isinstance(bank, nn.ModuleDict):
        return
    for name, dropout in bank.items():
        if not isinstance(dropout, nn.Dropout) or dropout.p == 0:
            continue
        seed = torch.randint(0, 2**62, (1,), device=mesh.device_type)
        dist.broadcast(seed, group=mesh.get_group(), group_src=0)
        bank[name] = SharedSeedDropout(dropout.p, int(seed.item()))


def _shard_lora(wrapper: nn.Module, mesh: DeviceMesh, column: bool) -> None:
    if column:
        for linear in _lora_linears(wrapper, "lora_B"):
            _shard_weight(linear, mesh, Shard(0))
        keep_lora_bank(wrapper, "lora_A", mesh)
        share_lora_dropout(wrapper, mesh)
        return
    for linear in _lora_linears(wrapper, "lora_A"):
        _shard_weight(linear, mesh, Shard(1))
    keep_lora_bank(wrapper, "lora_B", mesh)


def shard_column(linear: nn.Module, mesh: DeviceMesh) -> None:
    base = base_linear(linear)
    _shard_weight(base, mesh, Shard(0))
    _shard_column_bias(base, mesh)
    if base is not linear:
        stash_tp(linear, mesh)
    _shard_lora(linear, mesh, column=True)


def shard_row(linear: nn.Module, mesh: DeviceMesh) -> nn.Parameter | None:
    base = base_linear(linear)
    _shard_weight(base, mesh, Shard(1))
    if base is not linear:
        stash_tp(linear, mesh)
    _shard_lora(linear, mesh, column=False)
    replicate_on_tp(base, mesh, "bias")
    return live_row_bias(base)


def _validate_gqa(num_q: int, num_kv: int, tp: int) -> None:
    if num_q % tp != 0:
        msg = f"num_attention_heads ({num_q}) must be divisible by tp ({tp})"
        raise ValueError(msg)
    if num_kv <= 0 or num_q % num_kv != 0:
        msg = (
            f"num_attention_heads ({num_q}) must be divisible by "
            f"num_key_value_heads ({num_kv})"
        )
        raise ValueError(msg)
    local_q = num_q // tp
    heads_per_kv = num_q // num_kv
    if heads_per_kv % local_q != 0 and local_q % heads_per_kv != 0:
        msg = (
            f"query heads per rank ({local_q}) cross a key-value group "
            f"(num_attention_heads={num_q}, num_key_value_heads={num_kv}, "
            f"heads_per_kv={heads_per_kv}, tp={tp})"
        )
        raise ValueError(msg)


def kv_row_span(
    num_q: int, num_kv: int, head_dim: int, tp: int, rank: int
) -> tuple[int, int]:
    local_q = num_q // tp
    heads_per_kv = num_q // num_kv
    q0 = rank * local_q
    kv0 = q0 // heads_per_kv
    kv1 = (q0 + local_q - 1) // heads_per_kv + 1
    return kv0 * head_dim, kv1 * head_dim


def _shard_kv(
    linear: nn.Module,
    mesh: DeviceMesh,
    num_q: int,
    num_kv: int,
    head_dim: int,
) -> None:
    tp = int(mesh.size())
    base = base_linear(linear)
    weight = base.get_parameter("weight")
    if isinstance(weight, DTensor):
        return
    if num_kv % tp == 0:
        shard_column(linear, mesh)
        return
    full_rows = num_kv * head_dim
    if int(weight.shape[0]) != full_rows:
        msg = f"key/value weight rows {weight.shape[0]} do not match {full_rows} heads"
        raise ValueError(msg)
    # Several ranks read the same key/value head, so every rank keeps all of them.
    rows = (kv_row_span(num_q, num_kv, head_dim, tp, mesh.get_local_rank()),)
    read_rows(base, mesh, rows)
    read_lora_b_rows(linear, mesh, rows)
    keep_lora_bank(linear, "lora_A", mesh)
    share_lora_dropout(linear, mesh)


class GroupCtx(Protocol):
    group: dist.ProcessGroup


class GatherCtx(Protocol):
    rank: int
    width: int


class CopyToTPRegion(torch.autograd.Function):
    """Identity forward; all-reduce the gradient over the tensor-parallel group."""

    @staticmethod
    def forward(
        ctx: GroupCtx, tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group
        return tensor.view_as(tensor)

    @staticmethod
    def backward(
        ctx: GroupCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        grad = grad_output.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class ReduceFromTPRegion(torch.autograd.Function):
    """All-reduce forward over the tensor-parallel group; identity gradient."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        if SAVED_COLLECTIVE_SCOPE.depth:
            out = torch.ops._c10d_functional.all_reduce(tensor, "sum", group.group_name)
            return funcol.wait_tensor(out)
        # The all-reduce needs a contiguous buffer; a contiguous input reduces in place.
        reduced = tensor.contiguous()
        if reduced is tensor:
            ctx.mark_dirty(tensor)
        dist.all_reduce(reduced, group=group)
        return reduced

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        return grad_output, None


class SumOverTPRegion(torch.autograd.Function):
    """All-reduce forward and backward, for a value every rank's output reads."""

    @staticmethod
    def forward(
        ctx: GroupCtx, tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group
        reduced = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(reduced, group=group)
        return reduced

    @staticmethod
    def backward(
        ctx: GroupCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        grad = grad_output.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class RowGatherCtx(Protocol):
    group: dist.ProcessGroup
    rank: int
    rows: int


class GatherRowsFromTPRegion(torch.autograd.Function):
    """All-gather dim 0 forward; sum the gradient over the group and keep this rank's rows."""

    @staticmethod
    def forward(
        ctx: RowGatherCtx,
        tensor: torch.Tensor,
        rank: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.rank = rank
        ctx.rows = int(tensor.shape[0])
        local = tensor.contiguous()
        parts = [
            torch.empty_like(local) for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(parts, local, group=group)
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(
        ctx: RowGatherCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
        grad = grad_output.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(grad, group=ctx.group)
        start = ctx.rank * ctx.rows
        return grad[start : start + ctx.rows], None, None


def gather_rows_once(
    module: nn.Module,
    key: str,
    rows: torch.Tensor,
    trainable: bool,
    rank: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """All-gather ``rows`` over TP; frozen rows gather once and are cached on ``module``.

    :param module: Module that owns the cache.
    :param key: Cache key for this tensor within ``module``.
    :param rows: This rank's rows.
    :param trainable: Whether ``rows`` come from a trainable parameter, whose
        rows change every step, so they are never cached, even under ``no_grad``.
    :param rank: This rank's index in ``group``.
    :param group: Tensor-parallel process group.
    :return: Every rank's rows in rank order.
    """
    if trainable:
        return GatherRowsFromTPRegion.apply(rows, rank, group)
    cache = module.__dict__.setdefault("tp_frozen_rows", {})
    if key not in cache:
        cache[key] = GatherRowsFromTPRegion.apply(rows, rank, group)
    return cache[key]


def clear_gathered_rows(model: nn.Module) -> None:
    """Drop rows cached by :func:`gather_rows_once` once weights are rewritten.

    :param model: Model whose modules may hold cached rows.
    """
    for module in model.modules():
        module.__dict__.pop("tp_frozen_rows", None)


class GatherFromTPRegion(torch.autograd.Function):
    """All-gather the last dim forward; keep this rank's slice of the gradient."""

    @staticmethod
    def forward(
        ctx: GatherCtx, tensor: torch.Tensor, rank: int, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.width = int(tensor.shape[-1])
        local = tensor.contiguous()
        size = dist.get_world_size(group=group)
        if SAVED_COLLECTIVE_SCOPE.depth:
            gathered = torch.ops._c10d_functional.all_gather_into_tensor(
                local, size, group.group_name
            )
            return torch.cat(funcol.wait_tensor(gathered).chunk(size), dim=-1)
        parts = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(parts, local, group=group)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(
        ctx: GatherCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
        # The gathered output is replicated, so every rank already holds its full gradient.
        start = ctx.rank * ctx.width
        return grad_output[..., start : start + ctx.width].contiguous(), None, None


def copy_input_to_region(
    args: tuple[object, ...], kwargs: dict[str, object], group: dist.ProcessGroup
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Pass a block's hidden-state input through :class:`CopyToTPRegion`.

    :param args: Positional block arguments; the first is the hidden states.
    :type args: tuple[object, ...]
    :param kwargs: Keyword block arguments, holding ``hidden_states`` when ``args`` is empty.
    :type kwargs: dict[str, object]
    :param group: Tensor-parallel process group.
    :type group: dist.ProcessGroup
    :return: Arguments with the hidden states wrapped.
    :rtype: tuple[tuple[object, ...], dict[str, object]]
    """
    if args:
        return (CopyToTPRegion.apply(args[0], group), *args[1:]), kwargs
    hidden = CopyToTPRegion.apply(kwargs["hidden_states"], group)
    return args, {**kwargs, "hidden_states": hidden}


def local_region_params(
    module: nn.Module,
    skipped_bias: nn.Parameter | None,
) -> dict[str, torch.Tensor]:
    """Parameters for one rank's share of a tensor-parallel region.

    Sharded parameters become local leaves. A trainable replicated parameter
    only sees this rank's part of the gradient; its name is recorded on
    ``module`` so :func:`sum_tp_partial_grads` sums that gradient over TP once
    per optimizer step. Parameters marked by :func:`read_rows` keep this
    rank's rows.

    :param module: Module whose forward is the region.
    :type module: nn.Module
    :param skipped_bias: Row-parallel bias the caller adds after the reduce; zeroed here.
    :type skipped_bias: nn.Parameter | None
    :return: Name-to-tensor map for :func:`torch.func.functional_call`.
    :rtype: dict[str, torch.Tensor]
    """
    params = local_param_dict(module)
    row_ranges = {
        name: child.tp_row_ranges
        for name, child in module.named_modules()
        if hasattr(child, "tp_row_ranges")
    }
    partial: list[str] = []
    for name, param in module.named_parameters():
        tensor = params[name]
        if param is skipped_bias:
            params[name] = torch.zeros_like(tensor, requires_grad=False)
            continue
        if param.requires_grad and (
            not isinstance(param, DTensor) or is_tp_replicated(param)
        ):
            partial.append(name)
        ranges = row_ranges.get(name.rpartition(".")[0])
        if ranges is not None:
            tensor = torch.cat([tensor[start:stop] for start, stop in ranges])
        params[name] = tensor
    object.__setattr__(module, "tp_partial_grad_names", tuple(partial))
    return params


def sum_tp_partial_grads(
    model: nn.Module,
    group: dist.ProcessGroup,
    reduce_dtype: torch.dtype | None = None,
) -> None:
    """Sum the grads of replicated params in TP regions over ``group`` in one all-reduce.

    Each TP rank holds only its region's share of those grads, so this runs
    once per optimizer step, before clipping. FSDP grads reduce their local
    shard; the sum is linear, so it commutes with the reduce-scatter.

    :param model: Model whose TP regions have run a forward.
    :param group: Tensor-parallel process group.
    :param reduce_dtype: Dtype of the reduce buffer; ``None`` keeps the grad dtype.
    """
    params = [
        module.get_parameter(name)
        for module in model.modules()
        for name in module.__dict__.get("tp_partial_grad_names", ())
    ]
    all_reduce_grads(params, divisor=1, group=group, reduce_dtype=reduce_dtype)


def call_local_forward(
    module: nn.Module,
    original: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    skipped_bias: nn.Parameter | None,
) -> object:
    """Run ``original`` as ``module.forward`` on this rank's :func:`local_region_params`."""
    saved = module.forward
    module.forward = original
    try:
        params = local_region_params(module, skipped_bias)
        return functional_call(module, params, args, kwargs)
    finally:
        module.forward = saved


@contextmanager
def _local_head_counts(
    module: nn.Module, local_q: int, local_kv: int
) -> Iterator[None]:
    """Head counts read by the attention forward, restored on exit."""
    local_groups = local_q // local_kv
    updates = {
        "num_heads": local_q,
        "num_attention_heads": local_q,
        "num_key_value_heads": local_kv,
        "num_key_value_groups": local_groups,
    }
    missing = object()
    saved: list[tuple[str, object]] = []
    for name, value in updates.items():
        saved.append((name, module.__dict__.get(name, missing)))
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, previous in saved:
            if previous is missing:
                module.__dict__.pop(name, None)
            else:
                setattr(module, name, previous)


def _local_q_kv(module: nn.Module) -> tuple[int, int]:
    num_q = cast("int", module._tp_query_heads)
    num_kv = cast("int", module._tp_kv_heads)
    tp = cast("int", module._tp_degree)
    rank = cast("DeviceMesh", module._tp_mesh).get_local_rank()
    local_q = num_q // tp
    heads_per_kv = num_q // num_kv
    q0 = rank * local_q
    kv0 = q0 // heads_per_kv
    kv1 = (q0 + local_q - 1) // heads_per_kv + 1
    return local_q, kv1 - kv0


def live_row_bias(owner: nn.Module | None) -> nn.Parameter | None:
    """Replicated row-parallel bias registered on ``owner``."""
    if owner is None:
        return None
    bias = getattr(base_linear(owner), "bias", None)
    if isinstance(bias, nn.Parameter) and is_tp_replicated(bias):
        return bias
    return None


def add_row_bias(reduced: torch.Tensor, bias: nn.Parameter | None) -> torch.Tensor:
    """Add a replicated row-parallel bias after the region's all-reduce."""
    if bias is None:
        return reduced
    return reduced + cast("DTensor", bias).to_local().to(reduced.dtype)


def _all_reduce_hidden(
    output: object,
    group: dist.ProcessGroup,
    bias: nn.Parameter | None,
) -> object:
    if isinstance(output, tuple):
        hidden = output[0]
        if not isinstance(hidden, torch.Tensor):
            msg = "tensor-parallel forward did not return a tensor"
            raise TypeError(msg)
        reduced = ReduceFromTPRegion.apply(hidden, group)
        return (add_row_bias(reduced, bias), *output[1:])
    if not isinstance(output, torch.Tensor):
        msg = "tensor-parallel forward did not return a tensor"
        raise TypeError(msg)
    return add_row_bias(ReduceFromTPRegion.apply(output, group), bias)


def _install_block_forward(module: nn.Module, bias_owner: nn.Module | None) -> None:
    if getattr(module, "_agilerl_dense_tp", False):
        return
    original = module.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> object:
        group = cast("dist.ProcessGroup", self._tp_group)
        # Parameter objects are replaced when storage is allocated.
        bias = live_row_bias(bias_owner)
        local_q, local_kv = _local_q_kv(self)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        with _local_head_counts(self, local_q, local_kv):
            output = call_local_forward(self, original, args, kwargs, bias)
        return _all_reduce_hidden(output, group, bias)

    module.forward = MethodType(tp_forward, module)
    object.__setattr__(module, "_agilerl_dense_tp", True)


def install_reduce_forward(module: nn.Module, bias_owner: nn.Module | None) -> None:
    if getattr(module, "_agilerl_dense_tp", False):
        return
    original = module.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> object:
        group = cast("dist.ProcessGroup", self._tp_group)
        # Parameter objects are replaced when storage is allocated.
        bias = live_row_bias(bias_owner)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        output = call_local_forward(self, original, args, kwargs, bias)
        return _all_reduce_hidden(output, group, bias)

    module.forward = MethodType(tp_forward, module)
    object.__setattr__(module, "_agilerl_dense_tp", True)


class SliceLastDim(torch.autograd.Function):
    """Keep one rank's feature slice and all-gather its gradient."""

    @staticmethod
    def forward(
        ctx: GroupCtx,
        tensor: torch.Tensor,
        rank: int,
        chunks: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        local = int(tensor.shape[-1]) // chunks
        start = rank * local
        return tensor[..., start : start + local]

    @staticmethod
    def backward(
        ctx: GroupCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        grad_output = grad_outputs[0].contiguous()
        parts = [
            torch.empty_like(grad_output)
            for _ in range(dist.get_world_size(group=ctx.group))
        ]
        dist.all_gather(parts, grad_output, group=ctx.group)
        return torch.cat(parts, dim=-1), None, None, None


def _slice_last_dim(
    tensor: torch.Tensor, rank: int, chunks: int, group: dist.ProcessGroup
) -> torch.Tensor:
    return SliceLastDim.apply(tensor, rank, chunks, group)


def install_gather_forward(linear: nn.Module) -> None:
    """All-gather the column-parallel output of ``linear`` over TP."""
    if getattr(linear, "_agilerl_dense_tp", False):
        return
    original = linear.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> torch.Tensor:
        group = cast("dist.ProcessGroup", self._tp_group)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        output = call_local_forward(self, original, args, kwargs, None)
        if not isinstance(output, torch.Tensor):
            msg = "column-parallel gather expected a tensor"
            raise TypeError(msg)
        rank = cast("DeviceMesh", self._tp_mesh).get_local_rank()
        return GatherFromTPRegion.apply(output, rank, group)

    linear.forward = MethodType(tp_forward, linear)
    object.__setattr__(linear, "_agilerl_dense_tp", True)


def install_split_forward(linear: nn.Module, bias_owner: nn.Module | None) -> None:
    """Slice the input of row-parallel ``linear`` per rank and all-reduce its output."""
    if getattr(linear, "_agilerl_dense_tp", False):
        return
    original = linear.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> torch.Tensor:
        group = cast("dist.ProcessGroup", self._tp_group)
        tp = cast("int", self._tp_degree)
        rank = cast("DeviceMesh", self._tp_mesh).get_local_rank()
        incoming = args[0]
        if not isinstance(incoming, torch.Tensor):
            msg = "row-parallel linear expected a tensor input"
            raise TypeError(msg)
        # Parameter objects are replaced when storage is allocated.
        bias = live_row_bias(bias_owner)
        x = _slice_last_dim(incoming, rank, tp, group)
        output = call_local_forward(self, original, (x, *args[1:]), kwargs, bias)
        if not isinstance(output, torch.Tensor):
            msg = "row-parallel linear expected a tensor"
            raise TypeError(msg)
        return add_row_bias(ReduceFromTPRegion.apply(output, group), bias)

    linear.forward = MethodType(tp_forward, linear)
    object.__setattr__(linear, "_agilerl_dense_tp", True)


def _remember_heads(module: nn.Module, num_q: int, num_kv: int) -> None:
    object.__setattr__(module, "_tp_query_heads", num_q)
    object.__setattr__(module, "_tp_kv_heads", num_kv)


def shard_attention_projections(
    module: nn.Module,
    mesh: DeviceMesh,
    num_q: int,
    num_kv: int,
    head_dim: int,
    q_proj: nn.Module,
    k_proj: nn.Module | None,
    v_proj: nn.Module | None,
    o_proj: nn.Module,
) -> None:
    _validate_gqa(num_q, num_kv, int(mesh.size()))
    _remember_heads(module, num_q, num_kv)
    shard_column(q_proj, mesh)
    # A layer that reads another layer's keys and values has no k/v projection.
    for linear in (k_proj, v_proj):
        if linear is not None:
            _shard_kv(linear, mesh, num_q, num_kv, head_dim)
    bias = shard_row(o_proj, mesh)
    stash_tp(module, mesh)
    _install_block_forward(module, o_proj if bias is not None else None)


def _config_ties_embeddings(module: nn.Module) -> bool:
    config = getattr(module, "config", None)
    return bool(getattr(config, "tie_word_embeddings", False))


def _weight_is_embedding(model: nn.Module, weight: nn.Parameter) -> bool:
    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.weight is weight:
            return True
    return False


def _iter_lm_heads(model: nn.Module) -> list[tuple[str, nn.Module]]:
    found: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    direct = getattr(model, "lm_head", None)
    if isinstance(direct, nn.Module) and is_linear(direct):
        found.append(("lm_head", direct))
        seen.add(id(direct))
    for name, module in model.named_modules():
        if id(module) in seen or name.rsplit(".", 1)[-1] != "lm_head":
            continue
        if is_linear(module):
            found.append((name, module))
            seen.add(id(module))
    return found


def _lm_head_is_tied(model: nn.Module, name: str, head: nn.Module) -> bool:
    if _config_ties_embeddings(model):
        return True
    parent_name = name.rpartition(".")[0]
    parent = model.get_submodule(parent_name) if parent_name else model
    if _config_ties_embeddings(parent):
        return True
    return _weight_is_embedding(model, base_weight(head))


def _shard_lm_heads(model: nn.Module, mesh: DeviceMesh) -> int:
    count = 0
    for name, head in _iter_lm_heads(model):
        if _lm_head_is_tied(model, name, head):
            continue
        weight = base_weight(head)
        if isinstance(weight, DTensor):
            stash_tp(head, mesh)
            install_gather_forward(head)
            count += 1
            continue
        require_divisible(int(weight.shape[0]), int(mesh.size()), "vocab_size")
        shard_column(head, mesh)
        stash_tp(head, mesh)
        install_gather_forward(head)
        count += 1
    return count


class HasInputEmbeddings(Protocol):
    def get_input_embeddings(self) -> nn.Module: ...


def shard_tied_embeddings(model: nn.Module, mesh: DeviceMesh) -> int:
    """Split a tied token embedding and LM head by hidden column on ``mesh``.

    Head and embedding keep one shared weight. Each lookup is all-gathered;
    the head slices its input and all-reduces the logits.

    :param model: Model whose LM head is tied to its input embedding.
    :type model: nn.Module
    :param mesh: Tensor-parallel mesh.
    :type mesh: DeviceMesh
    :return: How many tied heads were sharded.
    :rtype: int
    """
    count = 0
    for name, head in _iter_lm_heads(model):
        if not _lm_head_is_tied(model, name, head):
            continue
        embedding = cast("HasInputEmbeddings", model).get_input_embeddings()
        _shard_weight(embedding, mesh, Shard(1))
        base = base_linear(head)
        base.weight = embedding.get_parameter("weight")
        stash_tp(head, mesh)
        install_gather_forward(embedding)
        install_split_forward(head, None)
        count += 1
    return count


def apply_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh | None) -> int:
    """Shard family blocks, then untied LM-head linears on ``tp_mesh``.

    :param model: Module tree that may contain those layers.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh. ``None`` shards nothing.
    :type tp_mesh: DeviceMesh | None
    :return: How many modules were sharded.
    :rtype: int
    :raises ValueError: If no family plan shards anything.
    """
    if tp_mesh is None or int(tp_mesh.size()) <= 1:
        return 0
    count = sum(plan.shard(model, tp_mesh) for plan in family_tensor_parallel_plans())
    if count == 0:
        msg = (
            f"tp={int(tp_mesh.size())} but there is no tensor-parallel plan for "
            f"{type(model).__name__}: no block would shard."
        )
        raise ValueError(msg)
    count += _shard_lm_heads(model, tp_mesh)
    return count


def restore_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Run each family's post-load step on shards filled with checkpoint weights.

    :param model: Model sharded by :func:`apply_tensor_parallel`.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """
    clear_gathered_rows(model)
    for plan in family_tensor_parallel_plans():
        plan.restore(model, tp_mesh)


def tensor_parallel_row_orders(model: nn.Module) -> dict[str, torch.Tensor]:
    """Source row of each stored dim-0 row, for every permuted TP parameter.

    :param model: Model sharded by :func:`apply_tensor_parallel`.
    :type model: nn.Module
    :return: Row order by live parameter FQN.
    :rtype: dict[str, torch.Tensor]
    """
    orders: dict[str, torch.Tensor] = {}
    for plan in family_tensor_parallel_plans():
        orders.update(plan.row_orders(model))
    return orders


def tensor_rows_in_order(
    value: torch.Tensor, order: torch.Tensor | None
) -> torch.Tensor:
    """Select ``value``'s dim-0 rows in ``order``; ``None`` or a scalar passes through.

    :param value: Tensor to reorder.
    :type value: torch.Tensor
    :param order: Row index for each output row.
    :type order: torch.Tensor | None
    :return: ``value`` with its rows in ``order``.
    :rtype: torch.Tensor
    """
    if order is None or value.dim() == 0:
        return value
    return value.index_select(0, order.to(value.device))
