# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen3.5 and Qwen3.5-MoE mixers, attention, MLPs, and vision blocks.

Each rank runs a GatedDeltaNet over ``num_k_heads/tp`` and ``num_v_heads/tp``
heads, then all-reduces the output. ``in_proj_qkv`` and ``conv1d`` rows are
grouped per rank as that rank's Q, K, and V so a ``Shard(0)`` chunk is one
local mixer. ``in_proj_qkv`` LoRA B is sharded in the same row order.

Vision attention's fused ``qkv`` and its LoRA B are grouped the same way.
Attention, MLP, and vision MLP blocks use the column- and row-parallel layers
in :mod:`agilerl.distributed.tensor_parallel`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5VisionAttention,
    Qwen3_5VisionMLP,
    Qwen3_5VisionPatchMerger,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionMLP,
    Qwen3_5MoeVisionPatchMerger,
)

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    ReduceFromTPRegion,
    add_row_bias,
    base_linear,
    base_weight,
    call_local_forward,
    copy_input_to_region,
    install_reduce_forward,
    is_lora_b_weight,
    keep_lora_bank,
    live_row_bias,
    permute_rows,
    replicate_on_tp,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_dim0,
    shard_lora_b_rows,
    shard_reordered_rows,
    shard_row,
    share_lora_dropout,
    stash_tp,
    tp_degree,
)

QkvKind = Literal["gdn", "vision"]

# Qwen3.5-MoE ships copies of the Qwen3.5 classes, not subclasses.
GatedDeltaNet = Qwen3_5GatedDeltaNet | Qwen3_5MoeGatedDeltaNet
Attention = Qwen3_5Attention | Qwen3_5MoeAttention
VisionAttention = Qwen3_5VisionAttention | Qwen3_5MoeVisionAttention
VisionFc = (
    Qwen3_5VisionMLP
    | Qwen3_5VisionPatchMerger
    | Qwen3_5MoeVisionMLP
    | Qwen3_5MoeVisionPatchMerger
)
MLP_CLASS_NAMES = frozenset({"Qwen3_5MLP", "Qwen3_5MoeMLP"})

GDN_PERMUTED_PARAMS = frozenset(
    {
        "in_proj_qkv.weight",
        "in_proj_qkv.base_layer.weight",
        "conv1d.weight",
    }
)
VISION_PERMUTED_PARAMS = frozenset(
    {
        "qkv.weight",
        "qkv.bias",
        "qkv.base_layer.weight",
        "qkv.base_layer.bias",
    }
)


def _mark_full(module: nn.Module, mesh: DeviceMesh) -> None:
    names = [name for name, _ in module.named_parameters(recurse=False)]
    replicate_on_tp(module, mesh, *names)


def _gdn_rank_ranges(
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    tp: int,
    rank: int,
) -> list[tuple[int, int]]:
    """Q, K, and V rows of one rank inside the full ``[Q|K|V]`` projection."""
    local_k = num_k_heads // tp
    local_v = num_v_heads // tp
    key_dim = num_k_heads * head_k_dim
    k0 = rank * local_k
    v0 = rank * local_v
    q_start = k0 * head_k_dim
    k_start = key_dim + k0 * head_k_dim
    v_start = 2 * key_dim + v0 * head_v_dim
    return [
        (q_start, q_start + local_k * head_k_dim),
        (k_start, k_start + local_k * head_k_dim),
        (v_start, v_start + local_v * head_v_dim),
    ]


def _vision_rank_ranges(
    num_heads: int, head_dim: int, tp: int, rank: int
) -> list[tuple[int, int]]:
    """Q, K, and V rows of one rank inside a fused vision ``qkv``."""
    local = num_heads // tp
    section = num_heads * head_dim
    start = rank * local * head_dim
    span = local * head_dim
    return [
        (part * section + start, part * section + start + span) for part in range(3)
    ]


def _row_order(ranges_for_rank: list[list[tuple[int, int]]]) -> torch.Tensor:
    parts = [
        torch.arange(start, stop)
        for ranges in ranges_for_rank
        for start, stop in ranges
    ]
    return torch.cat(parts)


def _gdn_row_order(
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    tp: int,
) -> torch.Tensor:
    return _row_order(
        [
            _gdn_rank_ranges(num_k_heads, head_k_dim, num_v_heads, head_v_dim, tp, rank)
            for rank in range(tp)
        ]
    )


def _vision_row_order(num_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    return _row_order(
        [_vision_rank_ranges(num_heads, head_dim, tp, rank) for rank in range(tp)]
    )


@contextmanager
def _local_gdn_sizes(mixer: GatedDeltaNet, tp: int) -> Iterator[None]:
    """Head and conv sizes for one tensor-parallel GatedDeltaNet forward."""
    num_k = mixer.num_k_heads
    num_v = mixer.num_v_heads
    key_dim = mixer.key_dim
    value_dim = mixer.value_dim
    conv_dim = mixer.conv_dim
    conv_in = mixer.conv1d.in_channels
    conv_out = mixer.conv1d.out_channels
    conv_groups = mixer.conv1d.groups
    local_k = int(num_k) // tp
    local_v = int(num_v) // tp
    local_key = local_k * int(mixer.head_k_dim)
    local_value = local_v * int(mixer.head_v_dim)
    local_conv = local_key * 2 + local_value
    mixer.num_k_heads = local_k
    mixer.num_v_heads = local_v
    mixer.key_dim = local_key
    mixer.value_dim = local_value
    mixer.conv_dim = local_conv
    mixer.conv1d.in_channels = local_conv
    mixer.conv1d.out_channels = local_conv
    mixer.conv1d.groups = local_conv
    try:
        yield
    finally:
        mixer.num_k_heads = num_k
        mixer.num_v_heads = num_v
        mixer.key_dim = key_dim
        mixer.value_dim = value_dim
        mixer.conv_dim = conv_dim
        mixer.conv1d.in_channels = conv_in
        mixer.conv1d.out_channels = conv_out
        mixer.conv1d.groups = conv_groups


@contextmanager
def _local_vision_heads(attn: VisionAttention, tp: int) -> Iterator[None]:
    """Head count for one tensor-parallel vision-attention forward."""
    num_heads = attn.num_heads
    attn.num_heads = int(num_heads) // tp
    try:
        yield
    finally:
        attn.num_heads = num_heads


def _reduce(
    output: torch.Tensor, group: dist.ProcessGroup, bias: nn.Parameter | None
) -> torch.Tensor:
    return add_row_bias(ReduceFromTPRegion.apply(output, group), bias)


def _install_gdn_forward(mixer: GatedDeltaNet) -> None:
    if getattr(mixer, "_agilerl_gdn_tp", False):
        return
    original = mixer.forward

    def tp_forward(
        self: GatedDeltaNet, *args: object, **kwargs: object
    ) -> torch.Tensor:
        tp = tp_degree(self)
        group = cast("dist.ProcessGroup", self._tp_group)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        with _local_gdn_sizes(self, tp):
            output = call_local_forward(self, original, args, kwargs, None)
        return _reduce(cast("torch.Tensor", output), group, None)

    mixer.forward = MethodType(tp_forward, mixer)
    object.__setattr__(mixer, "_agilerl_gdn_tp", True)


def _install_vision_forward(attn: VisionAttention) -> None:
    if getattr(attn, "_agilerl_vision_tp", False):
        return
    original = cast("Callable[..., torch.Tensor]", attn.forward)

    def tp_forward(
        self: VisionAttention, *args: object, **kwargs: object
    ) -> torch.Tensor:
        tp = tp_degree(self)
        group = cast("dist.ProcessGroup", self._tp_group)
        bias = live_row_bias(self.proj)
        args, kwargs = copy_input_to_region(args, kwargs, group)
        with _local_vision_heads(self, tp):
            output = call_local_forward(self, original, args, kwargs, bias)
        return _reduce(cast("torch.Tensor", output), group, bias)

    object.__setattr__(attn, "forward", MethodType(tp_forward, attn))
    object.__setattr__(attn, "_agilerl_vision_tp", True)


def _shard_gdn(mixer: GatedDeltaNet, device_mesh: DeviceMesh) -> None:
    tp = int(device_mesh.size())
    num_k = int(mixer.num_k_heads)
    num_v = int(mixer.num_v_heads)
    require_divisible(num_k, tp, "linear_num_key_heads")
    require_divisible(num_v, tp, "linear_num_value_heads")
    weight = base_weight(mixer.in_proj_qkv)
    if isinstance(weight, DTensor):
        stash_tp(mixer, device_mesh)
        _install_gdn_forward(mixer)
        return
    head_k = int(mixer.head_k_dim)
    head_v = int(mixer.head_v_dim)
    order = _gdn_row_order(num_k, head_k, num_v, head_v, tp)
    qkv = base_linear(mixer.in_proj_qkv)
    shard_reordered_rows(qkv, "weight", order, device_mesh)
    shard_lora_b_rows(mixer.in_proj_qkv, order, device_mesh)
    keep_lora_bank(mixer.in_proj_qkv, "lora_A", device_mesh)
    share_lora_dropout(mixer.in_proj_qkv, device_mesh)
    shard_reordered_rows(mixer.conv1d, "weight", order, device_mesh)
    for name in ("in_proj_z", "in_proj_a", "in_proj_b"):
        shard_column(getattr(mixer, name), device_mesh)
    shard_dim0(mixer, "dt_bias", device_mesh)
    shard_dim0(mixer, "A_log", device_mesh)
    _mark_full(mixer.norm, device_mesh)
    shard_row(mixer.out_proj, device_mesh)
    stash_tp(mixer, device_mesh)
    _install_gdn_forward(mixer)


def _shard_attention(module: Attention, mesh: DeviceMesh) -> None:
    config = module.config
    shard_attention_projections(
        module,
        mesh,
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(module.head_dim),
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.o_proj,
    )
    _mark_full(module.q_norm, mesh)
    _mark_full(module.k_norm, mesh)


def _is_qwen3_5_mlp(module: nn.Module) -> bool:
    # Liger replaces modeling_qwen3_5.Qwen3_5MLP before this module is imported.
    return type(module).__name__ in MLP_CLASS_NAMES


def _shard_mlp(module: nn.Module, mesh: DeviceMesh) -> None:
    require_divisible(
        cast("int", module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    down_proj = module.get_submodule("down_proj")
    shard_column(module.get_submodule("gate_proj"), mesh)
    shard_column(module.get_submodule("up_proj"), mesh)
    bias = shard_row(down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, down_proj if bias is not None else None)


def _shard_vision_attention(module: VisionAttention, mesh: DeviceMesh) -> None:
    tp = int(mesh.size())
    num_heads = int(module.num_heads)
    require_divisible(num_heads, tp, "vision num_heads")
    if int(module.dim) % num_heads != 0:
        msg = (
            f"vision hidden ({module.dim}) must be divisible by num_heads ({num_heads})"
        )
        raise ValueError(msg)
    weight = base_weight(module.qkv)
    if isinstance(weight, DTensor):
        stash_tp(module, mesh)
        _install_vision_forward(module)
        return
    head_dim = int(module.head_dim)
    order = _vision_row_order(num_heads, head_dim, tp)
    qkv = base_linear(module.qkv)
    shard_reordered_rows(qkv, "weight", order, mesh)
    if qkv.bias is not None:
        shard_reordered_rows(qkv, "bias", order, mesh)
    shard_lora_b_rows(module.qkv, order, mesh)
    keep_lora_bank(module.qkv, "lora_A", mesh)
    share_lora_dropout(module.qkv, mesh)
    shard_row(module.proj, mesh)
    stash_tp(module, mesh)
    _install_vision_forward(module)


def _shard_fc_mlp(module: VisionFc, mesh: DeviceMesh) -> None:
    rows = int(base_weight(module.linear_fc1).shape[0])
    require_divisible(rows, int(mesh.size()), "linear_fc1")
    norm = getattr(module, "norm", None)
    if isinstance(norm, nn.Module):
        _mark_full(norm, mesh)
    shard_column(module.linear_fc1, mesh)
    bias = shard_row(module.linear_fc2, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, module.linear_fc2 if bias is not None else None)


def apply_qwen3_5_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard Qwen3.5 language and vision blocks on ``tp_mesh``.

    :param model: Module tree that may contain Qwen3.5 blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            _shard_gdn(module, tp_mesh)
        elif isinstance(module, Attention):
            _shard_attention(module, tp_mesh)
        elif _is_qwen3_5_mlp(module):
            _shard_mlp(module, tp_mesh)
        elif isinstance(module, VisionAttention):
            _shard_vision_attention(module, tp_mesh)
        elif isinstance(module, VisionFc):
            _shard_fc_mlp(module, tp_mesh)
        else:
            continue
        count += 1
    return count


def _permuted_kind(owner: nn.Module, relative: str) -> QkvKind | None:
    if isinstance(owner, GatedDeltaNet) and (
        relative in GDN_PERMUTED_PARAMS or is_lora_b_weight(relative, "in_proj_qkv")
    ):
        return "gdn"
    if isinstance(owner, VisionAttention) and (
        relative in VISION_PERMUTED_PARAMS or is_lora_b_weight(relative, "qkv")
    ):
        return "vision"
    return None


def qwen3_5_row_orders(model: nn.Module) -> dict[str, torch.Tensor]:
    """Source row of each stored dim-0 row of the permuted Q/K/V weights and LoRA B.

    A sharded block stores these rows per rank, so its params are listed also
    while :func:`~agilerl.distributed.fsdp.gather_params` holds them dense.

    :param model: Model with tensor-parallel Qwen3.5 blocks.
    :type model: nn.Module
    :return: Row order by live parameter FQN.
    :rtype: dict[str, torch.Tensor]
    """
    owners: list[tuple[str, GatedDeltaNet | VisionAttention]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, GatedDeltaNet | VisionAttention) and tp_degree(module) > 1
    ]
    orders: dict[str, torch.Tensor] = {}
    for fqn, _ in model.named_parameters():
        owner: tuple[str, GatedDeltaNet | VisionAttention] | None = None
        for name, module in owners:
            if name and not (fqn == name or fqn.startswith(name + ".")):
                continue
            if owner is None or len(name) > len(owner[0]):
                owner = (name, module)
        if owner is None:
            continue
        owner_name, module = owner
        relative = fqn[len(owner_name) + 1 :] if owner_name else fqn
        if _permuted_kind(module, relative) is None:
            continue
        tp = tp_degree(module)
        if isinstance(module, GatedDeltaNet):
            orders[fqn] = _gdn_row_order(
                int(module.num_k_heads),
                int(module.head_k_dim),
                int(module.num_v_heads),
                int(module.head_v_dim),
                tp,
            )
        else:
            orders[fqn] = _vision_row_order(
                int(module.num_heads), int(module.head_dim), tp
            )
    return orders


def _realign_qkv(model: nn.Module) -> None:
    with torch.no_grad():
        for fqn, order in sorted(qwen3_5_row_orders(model).items()):
            permute_rows(cast("DTensor", model.get_parameter(fqn)), order)


def restore_qwen3_5_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Regroup Q/K/V rows per rank after checkpoint weights load.

    :param model: Model with tensor-parallel Qwen3.5 blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """
    _realign_qkv(model)
    apply_qwen3_5_tensor_parallel(model, tp_mesh)


TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_qwen3_5_tensor_parallel,
    restore=restore_qwen3_5_tensor_parallel,
    row_orders=qwen3_5_row_orders,
)
