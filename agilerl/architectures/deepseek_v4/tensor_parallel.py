# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek-V4 attention and shared experts.

Each rank runs ``num_attention_heads/tp`` query heads with their sinks and
``o_groups/tp`` output groups. The single shared key/value head, the query
down-projection, the compressor, and the indexer stay whole on every rank.
Packed routed experts stay on the expert-parallel mesh and the routers stay
whole.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers import PretrainedConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4MLP,
)

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    install_reduce_forward,
    replicate_on_tp,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_row,
    share_lora_dropout,
    stash_tp,
)


class OutputGroupsConfig:
    """Live view of a model config with this rank's output-group count."""

    def __init__(self, config: PretrainedConfig, o_groups: int) -> None:
        self.config = config
        self.o_groups = o_groups

    def __getattr__(self, name: str) -> object:
        return getattr(self.config, name)


@contextmanager
def _local_output_groups(
    module: DeepseekV4Attention, local_config: OutputGroupsConfig
) -> Iterator[None]:
    """Output-group count read by the attention forward, restored on exit."""
    config = module.config
    grouped = module.o_a_proj
    n_groups = grouped.n_groups
    object.__setattr__(module, "config", local_config)
    grouped.n_groups = local_config.o_groups
    try:
        yield
    finally:
        module.config = config
        grouped.n_groups = n_groups


def _install_group_forward(module: DeepseekV4Attention, tp: int) -> None:
    if getattr(module, "_agilerl_deepseek_v4_tp", False):
        return
    head_split = cast("Callable[..., object]", module.forward)
    local_config = OutputGroupsConfig(module.config, int(module.config.o_groups) // tp)

    def tp_forward(
        self: DeepseekV4Attention, *args: object, **kwargs: object
    ) -> object:
        with _local_output_groups(self, local_config):
            return head_split(*args, **kwargs)

    object.__setattr__(module, "forward", MethodType(tp_forward, module))
    object.__setattr__(module, "_agilerl_deepseek_v4_tp", True)


def _shard_attention(module: DeepseekV4Attention, mesh: DeviceMesh) -> None:
    config = module.config
    tp = int(mesh.size())
    num_heads = int(config.num_attention_heads)
    require_divisible(num_heads, tp, "num_attention_heads")
    require_divisible(int(config.o_groups), tp, "o_groups")
    shard_attention_projections(
        module,
        mesh,
        num_heads,
        1,
        int(module.head_dim),
        module.q_b_proj,
        module.kv_proj,
        None,
        module.o_b_proj,
    )
    # Heads and output groups are contiguous, so Shard(0) keeps each rank's groups.
    shard_column(module.o_a_proj, mesh)
    sinks = module.sinks
    module.register_parameter(
        "sinks",
        nn.Parameter(
            distribute_tensor(sinks.detach(), mesh, [Shard(0)]),
            requires_grad=sinks.requires_grad,
        ),
    )
    # The query down-projection, compressor, and indexer feed every local head.
    share_lora_dropout(module.q_a_proj, mesh)
    for child in module.modules():
        names = [name for name, _ in child.named_parameters(recurse=False)]
        replicate_on_tp(child, mesh, *names)
    _install_group_forward(module, tp)


def _shard_mlp(module: DeepseekV4MLP, mesh: DeviceMesh) -> None:
    require_divisible(
        int(module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    shard_column(module.gate_proj, mesh)
    shard_column(module.up_proj, mesh)
    bias = shard_row(module.down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, module.down_proj if bias is not None else None)


def apply_deepseek_v4_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard DeepSeek-V4 attention and shared-expert MLPs on ``tp_mesh``.

    :param model: Module tree that may contain DeepSeek-V4 blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many blocks were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        # Match by name: Liger can reassign modeling classes after instances exist.
        name = type(module).__name__
        if name == "DeepseekV4Attention":
            _shard_attention(cast("DeepseekV4Attention", module), tp_mesh)
        elif name == "DeepseekV4MLP":
            _shard_mlp(cast("DeepseekV4MLP", module), tp_mesh)
        else:
            continue
        count += 1
    return count


def restore_deepseek_v4_tensor_parallel(
    _model: nn.Module, _tp_mesh: DeviceMesh
) -> None:
    """Shards load in checkpoint row order, so nothing is regrouped.

    :param _model: Model sharded by :func:`apply_deepseek_v4_tensor_parallel`.
    :type _model: nn.Module
    :param _tp_mesh: Tensor-parallel mesh.
    :type _tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """


def deepseek_v4_row_orders(_model: nn.Module) -> dict[str, torch.Tensor]:
    """No DeepSeek-V4 parameter is stored in a per-rank row order.

    :param _model: Model sharded by :func:`apply_deepseek_v4_tensor_parallel`.
    :type _model: nn.Module
    :return: An empty map.
    :rtype: dict[str, torch.Tensor]
    """
    return {}


TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_deepseek_v4_tensor_parallel,
    restore=restore_deepseek_v4_tensor_parallel,
    row_orders=deepseek_v4_row_orders,
)
