# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel gpt-oss attention.

Each rank runs ``num_attention_heads/tp`` query heads with their sinks. Packed
experts stay on the expert-parallel mesh and the router stays whole.
"""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import shard_attention_projections


def _shard_attention(module: GptOssAttention, mesh: DeviceMesh) -> None:
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
    # Query heads split in contiguous chunks, so Shard(0) keeps each rank's sinks.
    sinks = module.sinks
    module.register_parameter(
        "sinks",
        nn.Parameter(
            distribute_tensor(sinks.detach(), mesh, [Shard(0)]),
            requires_grad=sinks.requires_grad,
        ),
    )


def apply_gpt_oss_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard gpt-oss attention projections and sinks on ``tp_mesh``.

    :param model: Module tree that may contain gpt-oss attention blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many attention blocks were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        # Match by name: Liger can reassign modeling classes after instances exist.
        if type(module).__name__ != "GptOssAttention":
            continue
        _shard_attention(cast("GptOssAttention", module), tp_mesh)
        count += 1
    return count


TENSOR_PARALLEL_PLAN = TensorParallelPlan(shard=apply_gpt_oss_tensor_parallel)
