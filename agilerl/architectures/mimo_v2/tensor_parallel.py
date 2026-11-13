# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel MiMo-V2 attention, dense MLP, and vision blocks.

Each rank runs ``num_attention_heads/tp`` query heads on full-attention and
sliding-window layers, with that layer's key/value heads at the Q/K and V head
dims. Sliding-window sinks split with the query heads. Dense MLP ``gate_proj``
and ``up_proj`` split by column and ``down_proj`` by row, with an all-reduce
after it. Packed experts stay on the expert-parallel mesh and the router stays
whole. The untied LM head is split by the generic LM-head pass.

Vision attention runs ``num_heads/tp`` query and ``num_key_value_heads/tp``
key/value heads per rank: fused ``qkv`` rows are stored in rank order and
regrouped after the checkpoint loads, and sinks split with the query heads.
Vision MLPs and the merger MLP split by column then row. The merger's
RMSNorm stays whole; the patch embed and block norms run outside any
tensor-parallel region.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashMLP,
)

from agilerl.architectures.mimo_v2.modeling import (
    MiMoV2VisionAttention,
    MiMoV2VisionMerger,
    MiMoV2VisionMLP,
)
from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    base_linear,
    install_reduce_forward,
    is_lora_b_weight,
    keep_lora_bank,
    permute_rows,
    replicate_on_tp,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_lora_b_rows,
    shard_reordered_rows,
    shard_row,
    share_lora_dropout,
    stash_tp,
)


def _shard_sinks(module: nn.Module, mesh: DeviceMesh) -> None:
    sinks = cast("nn.Parameter | None", module.sinks)
    if sinks is None:
        return
    # Query heads split in contiguous chunks, so Shard(0) keeps each rank's sinks.
    module.register_parameter(
        "sinks",
        nn.Parameter(
            distribute_tensor(sinks.detach(), mesh, [Shard(0)]),
            requires_grad=sinks.requires_grad,
        ),
    )


def _vision_qkv_row_order(attn: MiMoV2VisionAttention, tp: int) -> torch.Tensor:
    """Source row of each stored ``qkv`` row: every rank's Q, then K, then V rows."""
    sizes = [
        attn.num_heads * attn.qk_channels,
        attn.num_kv_heads * attn.qk_channels,
        attn.num_kv_heads * attn.kv_channels,
    ]
    starts = [0, sizes[0], sizes[0] + sizes[1]]
    return torch.cat(
        [
            torch.arange(start + rank * size // tp, start + (rank + 1) * size // tp)
            for rank in range(tp)
            for start, size in zip(starts, sizes, strict=True)
        ]
    )


def _shard_attention(module: MiMoV2FlashAttention, mesh: DeviceMesh) -> None:
    num_q = int(module.config.num_attention_heads)
    shard_attention_projections(
        module,
        mesh,
        num_q,
        num_q // int(module.num_key_value_groups),
        int(module.head_dim),
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.o_proj,
        int(module.v_head_dim),
    )
    _shard_sinks(module, mesh)


def _shard_mlp(module: MiMoV2FlashMLP, mesh: DeviceMesh) -> None:
    require_divisible(
        int(module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    shard_column(module.gate_proj, mesh)
    shard_column(module.up_proj, mesh)
    shard_row(module.down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, None)


def _shard_vision_attention(attn: MiMoV2VisionAttention, mesh: DeviceMesh) -> None:
    tp = int(mesh.size())
    require_divisible(attn.num_kv_heads, tp, "vision num_key_value_heads")
    order = _vision_qkv_row_order(attn, tp)
    qkv = base_linear(attn.qkv)
    shard_reordered_rows(qkv, "weight", order, mesh)
    shard_reordered_rows(qkv, "bias", order, mesh)
    shard_lora_b_rows(attn.qkv, order, mesh)
    keep_lora_bank(attn.qkv, "lora_A", mesh)
    share_lora_dropout(attn.qkv, mesh)
    _shard_sinks(attn, mesh)
    shard_row(attn.proj, mesh)
    stash_tp(attn, mesh)
    install_reduce_forward(attn, attn.proj)


def _shard_vision_mlp(mlp: MiMoV2VisionMLP, mesh: DeviceMesh) -> None:
    shard_column(mlp.gate_proj, mesh)
    shard_column(mlp.up_proj, mesh)
    shard_row(mlp.down_proj, mesh)
    stash_tp(mlp, mesh)
    install_reduce_forward(mlp, mlp.down_proj)


def _shard_vision_merger(merger: MiMoV2VisionMerger, mesh: DeviceMesh) -> None:
    replicate_on_tp(merger.ln_q, mesh, "weight")
    shard_column(merger.mlp[0], mesh)
    shard_row(merger.mlp[2], mesh)
    stash_tp(merger, mesh)
    install_reduce_forward(merger, None)


def apply_mimo_v2_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard MiMo-V2 attention, sinks, dense MLPs and vision blocks on ``tp_mesh``.

    :param model: Module tree that may contain MiMo-V2 text or vision blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        # Match by name: Liger can reassign modeling classes after instances exist.
        name = type(module).__name__
        if name == "MiMoV2FlashAttention":
            _shard_attention(cast("MiMoV2FlashAttention", module), tp_mesh)
        elif name == "MiMoV2FlashMLP":
            _shard_mlp(cast("MiMoV2FlashMLP", module), tp_mesh)
        elif isinstance(module, MiMoV2VisionAttention):
            _shard_vision_attention(module, tp_mesh)
        elif isinstance(module, MiMoV2VisionMLP):
            _shard_vision_mlp(module, tp_mesh)
        elif isinstance(module, MiMoV2VisionMerger):
            _shard_vision_merger(module, tp_mesh)
        else:
            continue
        count += 1
    return count


def mimo_v2_row_orders(model: nn.Module) -> dict[str, torch.Tensor]:
    """Source row of each stored dim-0 row of every vision ``qkv`` and its LoRA B.

    :param model: Model sharded by :func:`apply_mimo_v2_tensor_parallel`.
    :type model: nn.Module
    :return: Row order by live parameter FQN.
    :rtype: dict[str, torch.Tensor]
    """
    base_rows = {
        "qkv.weight",
        "qkv.bias",
        "qkv.base_layer.weight",
        "qkv.base_layer.bias",
    }
    orders: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        tp = int(getattr(module, "_tp_degree", 1))
        if tp <= 1 or not isinstance(module, MiMoV2VisionAttention):
            continue
        order = _vision_qkv_row_order(module, tp)
        for relative, _ in module.named_parameters():
            if relative in base_rows or is_lora_b_weight(relative, "qkv"):
                orders[f"{name}.{relative}"] = order
    return orders


def restore_mimo_v2_tensor_parallel(model: nn.Module, _tp_mesh: DeviceMesh) -> None:
    """Regroup vision ``qkv`` rows per rank after checkpoint weights load.

    :param model: Model sharded by :func:`apply_mimo_v2_tensor_parallel`.
    :type model: nn.Module
    :param _tp_mesh: Tensor-parallel mesh.
    :type _tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """
    with torch.no_grad():
        for fqn, order in sorted(mimo_v2_row_orders(model).items()):
            permute_rows(cast("DTensor", model.get_parameter(fqn)), order)


TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_mimo_v2_tensor_parallel,
    restore=restore_mimo_v2_tensor_parallel,
    row_orders=mimo_v2_row_orders,
)
