# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen2 and Qwen3 attention and MLPs.

Each rank runs ``num_attention_heads/tp`` query heads; Qwen2's q/k/v biases
split with their heads. Qwen3's per-head ``q_norm`` and ``k_norm`` stay whole.
MLP ``gate_proj`` and ``up_proj`` split by column and ``down_proj`` by row,
with an all-reduce after it.
"""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    install_reduce_forward,
    replicate_on_tp,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_row,
    stash_tp,
)


def _shard_attention(module: Qwen2Attention | Qwen3Attention, mesh: DeviceMesh) -> None:
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
    if not isinstance(module, Qwen3Attention):
        return
    for norm in (module.q_norm, module.k_norm):
        names = [name for name, _ in norm.named_parameters(recurse=False)]
        replicate_on_tp(norm, mesh, *names)


def _shard_mlp(module: Qwen2MLP | Qwen3MLP, mesh: DeviceMesh) -> None:
    require_divisible(
        int(module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    shard_column(module.gate_proj, mesh)
    shard_column(module.up_proj, mesh)
    bias = shard_row(module.down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, module.down_proj if bias is not None else None)


def apply_qwen3_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard Qwen2 and Qwen3 attention and MLP blocks on ``tp_mesh``.

    :param model: Module tree that may contain Qwen2 or Qwen3 blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many blocks were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        # Match the decoder layer by name: Liger reassigns the modeling classes
        # after instances exist, and a model built after that holds
        # LigerSwiGLUMLP, with the same projections.
        if type(module).__name__ not in ("Qwen2DecoderLayer", "Qwen3DecoderLayer"):
            continue
        _shard_attention(
            cast("Qwen2Attention | Qwen3Attention", module.self_attn), tp_mesh
        )
        _shard_mlp(cast("Qwen2MLP | Qwen3MLP", module.mlp), tp_mesh)
        count += 2
    return count


QWEN3_TENSOR_PARALLEL_PLAN = TensorParallelPlan(shard=apply_qwen3_tensor_parallel)
