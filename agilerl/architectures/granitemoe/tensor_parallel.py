# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Granite 3.1 MoE attention and tied embedding.

Each rank runs ``num_attention_heads/tp`` query heads. The tied token embedding
and LM head split their hidden columns and share one weight. Packed experts
stay on the expert-parallel mesh, which already spans the TP ranks; the router
and norms stay whole. The embedding, attention, residual and logit multipliers
scale full tensors after each gather or all-reduce, so they need no change.
"""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeAttention

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    shard_attention_projections,
    shard_tied_embeddings,
)


def _shard_attention(module: GraniteMoeAttention, mesh: DeviceMesh) -> None:
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
    )


def apply_granitemoe_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard Granite MoE attention and the tied head on ``tp_mesh``.

    :param model: Module tree that may contain Granite MoE attention blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        # Match by name: Liger can reassign modeling classes after instances exist.
        if type(module).__name__ != "GraniteMoeAttention":
            continue
        _shard_attention(cast("GraniteMoeAttention", module), tp_mesh)
        count += 1
    # Every family plan runs on every model; only a Granite MoE model splits its tied head.
    if count:
        count += shard_tied_embeddings(model, tp_mesh)
    return count


GRANITEMOE_TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_granitemoe_tensor_parallel
)
