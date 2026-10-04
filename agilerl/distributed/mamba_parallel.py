# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel NemotronH Mamba2 on the tensor-parallel mesh.

Each rank runs a mixer over ``num_heads/tp`` heads and ``n_groups/tp`` groups,
then all-reduces the output. ``in_proj`` and ``conv1d`` rows are grouped per
rank as that rank's gate, conv channels, and dt so a ``Shard(0)`` chunk is one
local mixer. ``in_proj`` LoRA B stays whole, in Hugging Face row order, on
every rank; each rank's forward reads the same rows as its ``in_proj`` shard.

The input and output pass through the tensor-parallel copy and reduce
functions, so gradients match the unsharded mixer.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.func import functional_call
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from agilerl.architectures.nemotron_h.mamba import mark_mamba_tp_shard
from agilerl.distributed.tensor_parallel import (
    ReduceFromTPRegion,
    copy_input_to_region,
    keep_lora_bank,
    local_region_params,
    mark_tp_replicated,
    read_lora_b_rows,
    share_lora_dropout,
    stash_tp,
)

MambaProjKind = Literal["in_proj", "conv"]


def _mesh_rank(device_mesh: DeviceMesh) -> int:
    coordinate = device_mesh.get_coordinate()
    if coordinate is None:
        msg = "Rank is not part of the mamba tensor-parallel mesh"
        raise RuntimeError(msg)
    return int(coordinate[0])


def _mixer_dims(mixer: NemotronHMamba2Mixer) -> tuple[int, int, int, int]:
    return (
        int(mixer.num_heads),
        int(mixer.head_dim),
        int(mixer.n_groups),
        int(mixer.ssm_state_size),
    )


def _rank_row_ranges(
    num_heads: int,
    head_dim: int,
    n_groups: int,
    ssm_state_size: int,
    tp: int,
    rank: int,
    kind: MambaProjKind,
) -> list[tuple[int, int]]:
    """Rows of one rank's gate/x, B, C, and dt inside the full projection."""
    intermediate = num_heads * head_dim
    local_heads = num_heads // tp
    local_groups = n_groups // tp
    head0 = rank * local_heads
    group0 = rank * local_groups
    head_span = local_heads * head_dim
    group_span = local_groups * ssm_state_size
    if kind == "conv":
        x0 = 0
        b0 = intermediate
        c0 = intermediate + n_groups * ssm_state_size
        return [
            (x0 + head0 * head_dim, x0 + head0 * head_dim + head_span),
            (b0 + group0 * ssm_state_size, b0 + group0 * ssm_state_size + group_span),
            (c0 + group0 * ssm_state_size, c0 + group0 * ssm_state_size + group_span),
        ]
    gate0 = 0
    x0 = intermediate
    b0 = 2 * intermediate
    c0 = 2 * intermediate + n_groups * ssm_state_size
    dt0 = c0 + n_groups * ssm_state_size
    return [
        (gate0 + head0 * head_dim, gate0 + head0 * head_dim + head_span),
        (x0 + head0 * head_dim, x0 + head0 * head_dim + head_span),
        (b0 + group0 * ssm_state_size, b0 + group0 * ssm_state_size + group_span),
        (c0 + group0 * ssm_state_size, c0 + group0 * ssm_state_size + group_span),
        (dt0 + head0, dt0 + head0 + local_heads),
    ]


def _row_order(
    num_heads: int,
    head_dim: int,
    n_groups: int,
    ssm_state_size: int,
    tp: int,
    kind: MambaProjKind,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for rank in range(tp):
        for start, stop in _rank_row_ranges(
            num_heads, head_dim, n_groups, ssm_state_size, tp, rank, kind
        ):
            parts.append(torch.arange(start, stop))
    return torch.cat(parts)


def permuted_local_from_full(
    full: torch.Tensor,
    mixer: NemotronHMamba2Mixer,
    kind: MambaProjKind,
    device_mesh: DeviceMesh,
) -> torch.Tensor:
    """Rows of ``full`` that form this rank's local mixer projection.

    :param full: Projection-ordered tensor, shape ``[rows, ...]``.
    :type full: torch.Tensor
    :param mixer: Mixer whose head and group counts describe ``full``.
    :type mixer: NemotronHMamba2Mixer
    :param kind: ``in_proj`` includes gate and dt; ``conv`` is x, B, and C.
    :type kind: MambaProjKind
    :param device_mesh: One-dimensional tensor-parallel mesh.
    :type device_mesh: DeviceMesh
    :return: This rank's rows, concatenated as gate/x, B, C, and dt.
    :rtype: torch.Tensor
    """
    num_heads, head_dim, n_groups, ssm_state_size = _mixer_dims(mixer)
    tp = int(device_mesh.size())
    rank = _mesh_rank(device_mesh)
    pieces = [
        full[start:stop]
        for start, stop in _rank_row_ranges(
            num_heads, head_dim, n_groups, ssm_state_size, tp, rank, kind
        )
    ]
    return torch.cat(pieces, dim=0)


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


def _register_shard(
    module: nn.Module, name: str, param: nn.Parameter, sharded: DTensor
) -> None:
    module.register_parameter(
        name, nn.Parameter(sharded, requires_grad=param.requires_grad)
    )
    mark_mamba_tp_shard(getattr(module, name))


def _shard_reordered_dim0(
    module: nn.Module,
    name: str,
    row_order: torch.Tensor,
    device_mesh: DeviceMesh,
) -> None:
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
    _register_shard(module, name, param, sharded)


def _shard_dim0(module: nn.Module, name: str, device_mesh: DeviceMesh) -> None:
    param = getattr(module, name)
    if not isinstance(param, nn.Parameter) or isinstance(param, DTensor):
        return
    sharded = distribute_tensor(param.detach(), device_mesh, [Shard(0)])
    _register_shard(module, name, param, sharded)


def _keep_out_proj_bias(mixer: NemotronHMamba2Mixer) -> None:
    linear = _base_linear(mixer.out_proj)
    if linear.bias is not None:
        mark_tp_replicated(linear, "bias")


def _shard_out_proj(mixer: NemotronHMamba2Mixer, device_mesh: DeviceMesh) -> None:
    linear = mixer.out_proj
    weight = linear.weight
    if isinstance(weight, nn.Parameter) and not isinstance(weight, DTensor):
        sharded = distribute_tensor(weight.detach(), device_mesh, [Shard(1)])
        _register_shard(linear, "weight", weight, sharded)
    _keep_out_proj_bias(mixer)


def _require_divisible(mixer: NemotronHMamba2Mixer, tp: int) -> None:
    num_heads = int(mixer.num_heads)
    n_groups = int(mixer.n_groups)
    if num_heads % tp != 0:
        msg = f"mamba_num_heads ({num_heads}) must be divisible by tp ({tp})"
        raise ValueError(msg)
    if n_groups % tp != 0:
        msg = f"n_groups ({n_groups}) must be divisible by tp ({tp})"
        raise ValueError(msg)


def _in_proj_weight(mixer: NemotronHMamba2Mixer) -> nn.Parameter:
    return _base_linear(mixer.in_proj).weight


@contextmanager
def _local_mixer_sizes(mixer: NemotronHMamba2Mixer, tp: int) -> Iterator[None]:
    """Head, group, and conv sizes for one tensor-parallel forward."""
    num_heads = mixer.num_heads
    n_groups = mixer.n_groups
    intermediate_size = mixer.intermediate_size
    conv_dim = mixer.conv_dim
    conv_in = mixer.conv1d.in_channels
    conv_out = mixer.conv1d.out_channels
    conv_groups = mixer.conv1d.groups
    local_heads = int(num_heads) // tp
    local_groups = int(n_groups) // tp
    local_intermediate = local_heads * int(mixer.head_dim)
    local_conv = local_intermediate + 2 * local_groups * int(mixer.ssm_state_size)
    # torch_forward and cuda_kernels_forward split projections with these.
    mixer.num_heads = local_heads
    mixer.n_groups = local_groups
    mixer.intermediate_size = local_intermediate
    mixer.conv_dim = local_conv
    mixer.conv1d.in_channels = local_conv
    mixer.conv1d.out_channels = local_conv
    mixer.conv1d.groups = local_conv
    try:
        yield
    finally:
        mixer.num_heads = num_heads
        mixer.n_groups = n_groups
        mixer.intermediate_size = intermediate_size
        mixer.conv_dim = conv_dim
        mixer.conv1d.in_channels = conv_in
        mixer.conv1d.out_channels = conv_out
        mixer.conv1d.groups = conv_groups


def _call_local_forward(
    module: NemotronHMamba2Mixer,
    original: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    saved = module.forward
    module.forward = original
    try:
        # Local matmul omits out_proj.bias. The full bias is added after the all-reduce.
        params = local_region_params(module, group, module.out_proj.bias)
        return functional_call(module, params, args, kwargs)
    finally:
        module.forward = saved


def _install_mamba_tp_forward(mixer: NemotronHMamba2Mixer) -> None:
    if getattr(mixer, "_agilerl_mamba_tp", False):
        return
    original = mixer.forward

    def tp_forward(
        self: NemotronHMamba2Mixer, *args: object, **kwargs: object
    ) -> torch.Tensor:
        tp = int(getattr(self, "_tp_degree", 1) or 1)
        group = getattr(self, "_tp_group", None)
        if tp <= 1 or not isinstance(group, dist.ProcessGroup):
            return original(*args, **kwargs)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        with _local_mixer_sizes(self, tp):
            output = _call_local_forward(self, original, args, kwargs, group)
        reduced = ReduceFromTPRegion.apply(output, group)
        bias = self.out_proj.bias
        if bias is None:
            return reduced
        return reduced + bias.to(reduced.dtype)

    mixer.forward = MethodType(tp_forward, mixer)
    mixer._agilerl_mamba_tp = True


def _shard_mixer(mixer: NemotronHMamba2Mixer, device_mesh: DeviceMesh) -> None:
    tp = int(device_mesh.size())
    _require_divisible(mixer, tp)
    weight = _in_proj_weight(mixer)
    if isinstance(weight, DTensor):
        stash_tp(mixer, device_mesh)
        _install_mamba_tp_forward(mixer)
        _keep_out_proj_bias(mixer)
        return
    num_heads, head_dim, n_groups, ssm_state_size = _mixer_dims(mixer)
    in_order = _row_order(num_heads, head_dim, n_groups, ssm_state_size, tp, "in_proj")
    conv_order = _row_order(num_heads, head_dim, n_groups, ssm_state_size, tp, "conv")
    in_linear = _base_linear(mixer.in_proj)
    _shard_reordered_dim0(in_linear, "weight", in_order, device_mesh)
    if in_linear.bias is not None:
        _shard_reordered_dim0(in_linear, "bias", in_order, device_mesh)
    in_rows = _rank_row_ranges(
        num_heads,
        head_dim,
        n_groups,
        ssm_state_size,
        tp,
        _mesh_rank(device_mesh),
        "in_proj",
    )
    read_lora_b_rows(mixer.in_proj, tuple(in_rows))
    keep_lora_bank(mixer.in_proj, "lora_A")
    share_lora_dropout(mixer.in_proj, device_mesh)
    _shard_reordered_dim0(mixer.conv1d, "weight", conv_order, device_mesh)
    if mixer.conv1d.bias is not None:
        _shard_reordered_dim0(mixer.conv1d, "bias", conv_order, device_mesh)
    _shard_dim0(mixer, "dt_bias", device_mesh)
    _shard_dim0(mixer, "A_log", device_mesh)
    _shard_dim0(mixer, "D", device_mesh)
    _shard_dim0(mixer.norm, "weight", device_mesh)
    _shard_out_proj(mixer, device_mesh)
    stash_tp(mixer, device_mesh)
    _install_mamba_tp_forward(mixer)


def _permuted_kind(relative: str) -> MambaProjKind | None:
    if relative in {
        "in_proj.weight",
        "in_proj.bias",
        "in_proj.base_layer.weight",
        "in_proj.base_layer.bias",
    }:
        return "in_proj"
    if relative in {"conv1d.weight", "conv1d.bias"}:
        return "conv"
    return None


def iter_permuted_mamba_params(
    model: nn.Module,
) -> list[tuple[str, NemotronHMamba2Mixer, nn.Parameter, MambaProjKind]]:
    """Tensor-parallel mixer params whose rows are grouped per rank.

    :param model: Module tree that may contain sharded Mamba2 mixers.
    :type model: nn.Module
    :return: ``(fqn, mixer, parameter, kind)`` for ``in_proj`` and ``conv1d``.
    :rtype: list[tuple[str, NemotronHMamba2Mixer, nn.Parameter, MambaProjKind]]
    """
    mixers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, NemotronHMamba2Mixer)
        and int(getattr(module, "_tp_degree", 1) or 1) > 1
    }
    found: list[tuple[str, NemotronHMamba2Mixer, nn.Parameter, MambaProjKind]] = []
    for fqn, param in model.named_parameters():
        if not isinstance(param, DTensor):
            continue
        owner: tuple[str, NemotronHMamba2Mixer] | None = None
        for name, mixer in mixers.items():
            if name and not (fqn == name or fqn.startswith(name + ".")):
                continue
            if owner is None or len(name) > len(owner[0]):
                owner = (name, mixer)
        if owner is None:
            continue
        mixer_name, mixer = owner
        relative = fqn[len(mixer_name) + 1 :] if mixer_name else fqn
        kind = _permuted_kind(relative)
        if kind is None:
            continue
        if param.device_mesh is not getattr(mixer, "_tp_mesh", None):
            continue
        found.append((fqn, mixer, param, kind))
    return found


def realign_mamba_permuted_shards(model: nn.Module) -> None:
    """Copy per-rank gate, conv, and dt rows into tensor-parallel mixer shards.

    :param model: Model with tensor-parallel Mamba2 mixers.
    :type model: nn.Module
    :return: None
    :rtype: None
    """
    items = sorted(iter_permuted_mamba_params(model), key=lambda item: item[0])
    with torch.no_grad():
        for _fqn, mixer, param, kind in items:
            full = param.full_tensor()
            local = permuted_local_from_full(full, mixer, kind, param.device_mesh)
            param.to_local().copy_(
                local.to(device=param.to_local().device, dtype=param.dtype)
            )


def mark_mamba_tp_params(model: nn.Module) -> None:
    """Mark tensor-parallel mixer parameters; call again after they are replaced.

    :param model: Model that may contain sharded Mamba2 mixers.
    :type model: nn.Module
    :return: None
    :rtype: None
    """
    for module in model.modules():
        if not isinstance(module, NemotronHMamba2Mixer):
            continue
        mesh = getattr(module, "_tp_mesh", None)
        if mesh is None or int(getattr(module, "_tp_degree", 1) or 1) <= 1:
            continue
        for param in module.parameters():
            if isinstance(param, DTensor) and param.device_mesh is mesh:
                mark_mamba_tp_shard(param)


def apply_mamba_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh | None) -> int:
    """Shard NemotronH Mamba2 mixers on ``tp_mesh`` and all-reduce outputs.

    :param model: Module tree that may contain Mamba2 mixers.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh. ``None`` shards nothing.
    :type tp_mesh: DeviceMesh | None
    :return: How many mixers were sharded.
    :rtype: int
    """
    if tp_mesh is None or int(tp_mesh.size()) <= 1:
        return 0
    count = 0
    for module in model.modules():
        if not isinstance(module, NemotronHMamba2Mixer):
            continue
        _shard_mixer(module, tp_mesh)
        count += 1
    return count
