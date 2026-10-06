# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel dense layers on the tensor-parallel mesh.

Column-parallel weights are ``Shard(0)``. Row-parallel weights are ``Shard(1)``
and their outputs are all-reduced. Key/value heads that are shared by several
ranks stay whole on every rank; each rank's forward reads its own rows.

A block's hidden-state input enters through :class:`CopyToTPRegion` and its
output leaves through :class:`ReduceFromTPRegion`, so forward and backward
match the unsharded block.

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
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.func import functional_call

from agilerl.architectures import family_tensor_parallel_plans
from agilerl.distributed.expert_parallel import _local_param_dict


def stash_tp(module: nn.Module, device_mesh: DeviceMesh) -> None:
    """Record the tensor-parallel group, mesh, and degree on ``module``."""
    object.__setattr__(module, "_tp_group", device_mesh.get_group())
    object.__setattr__(module, "_tp_mesh", device_mesh)
    object.__setattr__(module, "_tp_degree", int(device_mesh.size()))


def _base_linear(module: nn.Module) -> nn.Module:
    get_base = getattr(module, "get_base_layer", None)
    if callable(get_base):
        base = get_base()
        if isinstance(base, nn.Module) and hasattr(base, "weight"):
            return base
    base = getattr(module, "base_layer", None)
    if isinstance(base, nn.Module) and hasattr(base, "weight"):
        return base
    return module


def _base_weight(module: nn.Module) -> nn.Parameter:
    return _base_linear(module).get_parameter("weight")


def _is_linear(module: nn.Module) -> bool:
    return isinstance(_base_linear(module), nn.Linear)


def mark_tp_replicated(module: nn.Module, *names: str) -> None:
    """List ``module``'s own params that are full copies on every TP rank.

    FSDP leaves these unsharded. Names survive ``to_empty``; param objects do not.

    :param module: Module that directly owns the parameters.
    :param names: Parameter names on ``module``.
    """
    current = getattr(module, "tp_replicated_params", ())
    object.__setattr__(
        module, "tp_replicated_params", tuple(dict.fromkeys((*current, *names)))
    )


def require_divisible(size: int, tp: int, label: str) -> None:
    if size % tp != 0:
        msg = f"{label} ({size}) must be divisible by tp ({tp})"
        raise ValueError(msg)


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


def keep_lora_bank(wrapper: nn.Module, name: str) -> None:
    """Keep a LoRA ``lora_A`` / ``lora_B`` bank as full copies on every TP rank."""
    for linear in _lora_linears(wrapper, name):
        weight = getattr(linear, "weight", None)
        if isinstance(weight, nn.Parameter) and not isinstance(weight, DTensor):
            mark_tp_replicated(linear, "weight")


def read_rows(module: nn.Module, ranges: tuple[tuple[int, int], ...]) -> None:
    """Keep ``module``'s params whole on every TP rank; its region reads ``ranges``.

    :param module: Module whose weight (and bias) rows are selected in forward.
    :type module: nn.Module
    :param ranges: ``(start, stop)`` row ranges of this rank, in local order.
    :type ranges: tuple[tuple[int, int], ...]
    """
    for name in ("weight", "bias"):
        if isinstance(getattr(module, name, None), nn.Parameter):
            mark_tp_replicated(module, name)
    object.__setattr__(module, "tp_row_ranges", ranges)


def read_lora_b_rows(wrapper: nn.Module, ranges: tuple[tuple[int, int], ...]) -> None:
    """Keep LoRA B whole on every TP rank; its region reads ``ranges``.

    :param wrapper: PEFT LoRA layer.
    :type wrapper: nn.Module
    :param ranges: ``(start, stop)`` row ranges of this rank, in local order.
    :type ranges: tuple[tuple[int, int], ...]
    """
    for linear in _lora_linears(wrapper, "lora_B"):
        read_rows(linear, ranges)


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
        keep_lora_bank(wrapper, "lora_A")
        share_lora_dropout(wrapper, mesh)
        return
    for linear in _lora_linears(wrapper, "lora_A"):
        _shard_weight(linear, mesh, Shard(1))
    keep_lora_bank(wrapper, "lora_B")


def shard_column(linear: nn.Module, mesh: DeviceMesh) -> None:
    base = _base_linear(linear)
    _shard_weight(base, mesh, Shard(0))
    _shard_column_bias(base, mesh)
    if base is not linear:
        stash_tp(linear, mesh)
    _shard_lora(linear, mesh, column=True)


def shard_row(linear: nn.Module, mesh: DeviceMesh) -> nn.Parameter | None:
    base = _base_linear(linear)
    _shard_weight(base, mesh, Shard(1))
    if base is not linear:
        stash_tp(linear, mesh)
    _shard_lora(linear, mesh, column=False)
    bias = base.bias
    if not isinstance(bias, nn.Parameter) or isinstance(bias, DTensor):
        return None
    mark_tp_replicated(base, "bias")
    return bias


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


def _kv_row_span(
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
    base = _base_linear(linear)
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
    rows = (_kv_row_span(num_q, num_kv, head_dim, tp, mesh.get_local_rank()),)
    read_rows(base, rows)
    read_lora_b_rows(linear, rows)
    keep_lora_bank(linear, "lora_A")
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
        reduced = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(reduced, group=group)
        return reduced

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        return grad_output, None


class GatherFromTPRegion(torch.autograd.Function):
    """All-gather the last dim forward; keep this rank's slice of the gradient."""

    @staticmethod
    def forward(
        ctx: GatherCtx, tensor: torch.Tensor, rank: int, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.width = int(tensor.shape[-1])
        local = tensor.contiguous()
        parts = [
            torch.empty_like(local) for _ in range(dist.get_world_size(group=group))
        ]
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
    group: dist.ProcessGroup,
    skipped_bias: nn.Parameter | None,
) -> dict[str, torch.Tensor]:
    """Parameters for one rank's share of a tensor-parallel region.

    Sharded parameters become local leaves. A trainable replicated parameter
    only sees this rank's part of the gradient, so that gradient is summed over
    ``group``. Parameters marked by :func:`read_rows` keep this rank's rows.
    All parameters are cast to the region's ``tp_compute_dtype`` when set.

    :param module: Module whose forward is the region.
    :type module: nn.Module
    :param group: Tensor-parallel process group.
    :type group: dist.ProcessGroup
    :param skipped_bias: Row-parallel bias the caller adds after the reduce; zeroed here.
    :type skipped_bias: nn.Parameter | None
    :return: Name-to-tensor map for :func:`torch.func.functional_call`.
    :rtype: dict[str, torch.Tensor]
    """
    params = _local_param_dict(module)
    row_ranges = {
        name: child.tp_row_ranges
        for name, child in module.named_modules()
        if hasattr(child, "tp_row_ranges")
    }
    dtype = getattr(module, "tp_compute_dtype", None)
    for name, param in module.named_parameters():
        if param is skipped_bias:
            params[name] = torch.zeros_like(param, dtype=dtype, requires_grad=False)
            continue
        tensor = params[name]
        if param.requires_grad and not isinstance(param, DTensor):
            tensor = CopyToTPRegion.apply(param, group)
        ranges = row_ranges.get(name.rpartition(".")[0])
        if ranges is not None:
            tensor = torch.cat([tensor[start:stop] for start, stop in ranges])
        if dtype is not None:
            tensor = tensor.to(dtype)
        params[name] = tensor
    return params


def set_tp_compute_dtype(model: nn.Module, dtype: torch.dtype) -> set[nn.Parameter]:
    """Run every tensor-parallel region with its parameters cast to ``dtype``.

    Trainable parameters keep their stored dtype, so their gradients and
    optimizer state stay in it.

    :param model: Model with tensor-parallel modules.
    :type model: nn.Module
    :param dtype: Forward dtype, the FSDP ``param_dtype``.
    :type dtype: torch.dtype
    :return: Trainable parameters of tensor-parallel modules.
    :rtype: set[nn.Parameter]
    """
    trainable: set[nn.Parameter] = set()
    for module in model.modules():
        if not hasattr(module, "_tp_group"):
            continue
        object.__setattr__(module, "tp_compute_dtype", dtype)
        trainable.update(param for param in module.parameters() if param.requires_grad)
    return trainable


def _call_local_forward(
    module: nn.Module,
    original: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    group: dist.ProcessGroup,
    skipped_bias: nn.Parameter | None,
) -> object:
    saved = module.forward
    module.forward = original
    try:
        params = local_region_params(module, group, skipped_bias)
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


def _live_row_bias(owner: nn.Module | None) -> nn.Parameter | None:
    """Plain row-parallel bias registered on ``owner``."""
    if owner is None:
        return None
    bias = getattr(_base_linear(owner), "bias", None)
    if isinstance(bias, nn.Parameter) and not isinstance(bias, DTensor):
        return bias
    return None


def _tp_group(module: nn.Module) -> dist.ProcessGroup:
    group = getattr(module, "_tp_group", None)
    if not isinstance(group, dist.ProcessGroup):
        msg = "tensor-parallel module has no process group"
        raise RuntimeError(msg)
    return group


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
        if bias is not None:
            reduced = reduced + bias.to(reduced.dtype)
        return (reduced, *output[1:])
    if not isinstance(output, torch.Tensor):
        msg = "tensor-parallel forward did not return a tensor"
        raise TypeError(msg)
    reduced = ReduceFromTPRegion.apply(output, group)
    if bias is not None:
        reduced = reduced + bias.to(reduced.dtype)
    return reduced


def _install_block_forward(module: nn.Module, bias_owner: nn.Module | None) -> None:
    if getattr(module, "_agilerl_dense_tp", False):
        return
    original = module.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> object:
        group = _tp_group(self)
        # Parameter objects are replaced when storage is allocated.
        bias = _live_row_bias(bias_owner)
        local_q, local_kv = _local_q_kv(self)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        with _local_head_counts(self, local_q, local_kv):
            output = _call_local_forward(self, original, args, kwargs, group, bias)
        return _all_reduce_hidden(output, group, bias)

    module.forward = MethodType(tp_forward, module)
    object.__setattr__(module, "_agilerl_dense_tp", True)


def install_reduce_forward(module: nn.Module, bias_owner: nn.Module | None) -> None:
    if getattr(module, "_agilerl_dense_tp", False):
        return
    original = module.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> object:
        group = _tp_group(self)
        # Parameter objects are replaced when storage is allocated.
        bias = _live_row_bias(bias_owner)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        output = _call_local_forward(self, original, args, kwargs, group, bias)
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


def _install_gather_forward(linear: nn.Module) -> None:
    if getattr(linear, "_agilerl_dense_tp", False):
        return
    original = linear.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> torch.Tensor:
        group = _tp_group(self)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        output = _call_local_forward(self, original, args, kwargs, group, None)
        if not isinstance(output, torch.Tensor):
            msg = "column-parallel gather expected a tensor"
            raise TypeError(msg)
        rank = cast("DeviceMesh", self._tp_mesh).get_local_rank()
        return GatherFromTPRegion.apply(output, rank, group)

    linear.forward = MethodType(tp_forward, linear)
    object.__setattr__(linear, "_agilerl_dense_tp", True)


def _install_split_forward(linear: nn.Module, bias_owner: nn.Module | None) -> None:
    if getattr(linear, "_agilerl_dense_tp", False):
        return
    original = linear.forward

    def tp_forward(self: nn.Module, *args: object, **kwargs: object) -> torch.Tensor:
        group = _tp_group(self)
        tp = cast("int", self._tp_degree)
        rank = cast("DeviceMesh", self._tp_mesh).get_local_rank()
        incoming = args[0]
        if not isinstance(incoming, torch.Tensor):
            msg = "row-parallel linear expected a tensor input"
            raise TypeError(msg)
        # Parameter objects are replaced when storage is allocated.
        bias = _live_row_bias(bias_owner)
        x = _slice_last_dim(incoming, rank, tp, group)
        output = _call_local_forward(
            self, original, (x, *args[1:]), kwargs, group, bias
        )
        if not isinstance(output, torch.Tensor):
            msg = "row-parallel linear expected a tensor"
            raise TypeError(msg)
        reduced = ReduceFromTPRegion.apply(output, group)
        if bias is None:
            return reduced
        return reduced + bias.to(reduced.dtype)

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
    k_proj: nn.Module,
    v_proj: nn.Module,
    o_proj: nn.Module,
) -> None:
    _validate_gqa(num_q, num_kv, int(mesh.size()))
    _remember_heads(module, num_q, num_kv)
    shard_column(q_proj, mesh)
    _shard_kv(k_proj, mesh, num_q, num_kv, head_dim)
    _shard_kv(v_proj, mesh, num_q, num_kv, head_dim)
    bias = shard_row(o_proj, mesh)
    stash_tp(module, mesh)
    _install_block_forward(module, o_proj if bias is not None else None)


def _vision_head_count(module: nn.Module) -> int:
    for name in ("num_heads", "num_attention_heads"):
        value = module.__dict__.get(name)
        if isinstance(value, int):
            return value
    msg = f"{type(module).__name__} attention has no num_heads"
    raise ValueError(msg)


def _shard_vision_attention(module: nn.Module, mesh: DeviceMesh) -> None:
    num_q = _vision_head_count(module)
    num_kv_value = module.__dict__.get("num_key_value_heads", num_q)
    num_kv = int(num_kv_value)
    rows = int(_base_weight(module.get_submodule("query")).shape[0])
    if rows % num_q != 0:
        msg = f"query rows ({rows}) must be divisible by num_heads ({num_q})"
        raise ValueError(msg)
    head_dim = module.__dict__.get("head_dim", rows // num_q)
    shard_attention_projections(
        module,
        mesh,
        num_q,
        num_kv,
        int(head_dim),
        module.get_submodule("query"),
        module.get_submodule("key"),
        module.get_submodule("value"),
        module.get_submodule("proj"),
    )


def _shard_vision_mlp(module: nn.Module, mesh: DeviceMesh) -> None:
    fc1 = module.get_submodule("fc1")
    fc2 = module.get_submodule("fc2")
    intermediate = int(_base_weight(fc1).shape[0])
    require_divisible(intermediate, int(mesh.size()), "intermediate_size")
    shard_column(fc1, mesh)
    bias = shard_row(fc2, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, fc2 if bias is not None else None)


def _named_linear(module: nn.Module, name: str) -> bool:
    child = getattr(module, name, None)
    return isinstance(child, nn.Module) and _is_linear(child)


def _is_vision_attention(module: nn.Module) -> bool:
    return all(
        _named_linear(module, name) for name in ("query", "key", "value", "proj")
    )


def _is_vision_mlp(module: nn.Module) -> bool:
    return all(_named_linear(module, name) for name in ("fc1", "fc2"))


def _has_latent_projections(module: nn.Module) -> bool:
    return isinstance(
        getattr(module, "fc1_latent_proj", None), nn.Module
    ) and isinstance(getattr(module, "fc2_latent_proj", None), nn.Module)


def _shard_latent(module: nn.Module, mesh: DeviceMesh) -> bool:
    fc1 = module.get_submodule("fc1_latent_proj")
    fc2 = module.get_submodule("fc2_latent_proj")
    if not _is_linear(fc1) or not _is_linear(fc2):
        return False
    latent = int(_base_weight(fc1).shape[0])
    require_divisible(latent, int(mesh.size()), "moe_latent_size")
    shard_column(fc1, mesh)
    stash_tp(fc1, mesh)
    _install_gather_forward(fc1)
    bias = shard_row(fc2, mesh)
    stash_tp(fc2, mesh)
    _install_split_forward(fc2, fc2 if bias is not None else None)
    return True


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
    if isinstance(direct, nn.Module) and _is_linear(direct):
        found.append(("lm_head", direct))
        seen.add(id(direct))
    for name, module in model.named_modules():
        if id(module) in seen or name.rsplit(".", 1)[-1] != "lm_head":
            continue
        if _is_linear(module):
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
    return _weight_is_embedding(model, _base_weight(head))


def _shard_lm_heads(model: nn.Module, mesh: DeviceMesh) -> int:
    count = 0
    for name, head in _iter_lm_heads(model):
        if _lm_head_is_tied(model, name, head):
            continue
        weight = _base_weight(head)
        if isinstance(weight, DTensor):
            stash_tp(head, mesh)
            _install_gather_forward(head)
            count += 1
            continue
        require_divisible(int(weight.shape[0]), int(mesh.size()), "vocab_size")
        shard_column(head, mesh)
        stash_tp(head, mesh)
        _install_gather_forward(head)
        count += 1
    return count


def _shard_named_module(name: str, module: nn.Module, mesh: DeviceMesh) -> bool:
    if _has_latent_projections(module):
        return _shard_latent(module, mesh)
    if "vision_model" not in name:
        return False
    if _is_vision_attention(module):
        _shard_vision_attention(module, mesh)
        return True
    if _is_vision_mlp(module):
        _shard_vision_mlp(module, mesh)
        return True
    return False


def apply_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh | None) -> int:
    """Shard family blocks, then latent, vision, and LM-head linears on ``tp_mesh``.

    :param model: Module tree that may contain those layers.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh. ``None`` shards nothing.
    :type tp_mesh: DeviceMesh | None
    :return: How many modules were sharded.
    :rtype: int
    """
    if tp_mesh is None or int(tp_mesh.size()) <= 1:
        return 0
    count = sum(plan.shard(model, tp_mesh) for plan in family_tensor_parallel_plans())
    for name, module in model.named_modules():
        if _shard_named_module(name, module, tp_mesh):
            count += 1
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
    for plan in family_tensor_parallel_plans():
        plan.restore(model, tp_mesh)
