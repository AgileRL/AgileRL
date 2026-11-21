# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel MiMo-V2 attention and dense MLP.

Each rank runs ``num_attention_heads/tp`` query heads on full-attention and
sliding-window layers, with that layer's key/value heads at the Q/K and V head
dims. Sliding-window sinks split with the query heads. Dense MLP ``gate_proj``
and ``up_proj`` split by column and ``down_proj`` by row, with an all-reduce
after it. Packed experts stay on the expert-parallel mesh and the router stays
whole. The untied LM head is split by the generic LM-head pass.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashMLP,
)

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    install_reduce_forward,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_row,
    stash_tp,
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
    sinks = module.sinks
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


def _shard_mlp(module: MiMoV2FlashMLP, mesh: DeviceMesh) -> None:
    require_divisible(
        int(module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    shard_column(module.gate_proj, mesh)
    shard_column(module.up_proj, mesh)
    shard_row(module.down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, None)


def apply_mimo_v2_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard MiMo-V2 attention, sinks and dense MLPs on ``tp_mesh``.

    :param model: Module tree that may contain MiMo-V2 blocks.
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
        else:
            continue
        count += 1
    return count


def restore_mimo_v2_tensor_parallel(_model: nn.Module, _tp_mesh: DeviceMesh) -> None:
    """Shards load in checkpoint row order, so nothing is regrouped.

    :param _model: Model sharded by :func:`apply_mimo_v2_tensor_parallel`.
    :type _model: nn.Module
    :param _tp_mesh: Tensor-parallel mesh.
    :type _tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """


def mimo_v2_row_orders(_model: nn.Module) -> dict[str, torch.Tensor]:
    """No MiMo-V2 parameter is stored in a per-rank row order.

    :param _model: Model sharded by :func:`apply_mimo_v2_tensor_parallel`.
    :type _model: nn.Module
    :return: An empty map.
    :rtype: dict[str, torch.Tensor]
    """
    return {}


MIMO_V2_TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_mimo_v2_tensor_parallel,
    restore=restore_mimo_v2_tensor_parallel,
    row_orders=mimo_v2_row_orders,
)
