# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Granite 4 Mamba2 mixers, attention, shared MLPs, and tied embedding.

Each rank runs ``num_attention_heads/tp`` query heads and ``mamba_n_heads/tp``
Mamba2 heads. Granite 4 checkpoints have one SSM group, so every rank stores
``1/tp`` of the B and C rows and gathers them in forward; the gated norm spans
every head and sums its variance over TP. The shared MLP's fused
``input_linear`` and its LoRA B are grouped per rank as that rank's gate and
up rows. The tied token embedding and LM head split their hidden columns.
Packed experts stay on the expert-parallel mesh, which already spans the TP
ranks; the router and norms stay whole.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridAttention,
    GraniteMoeHybridMambaLayer,
    GraniteMoeHybridMLP,
)

from agilerl.architectures.nemotron_h.tensor_parallel import (
    mamba_row_orders,
    realign_mamba_permuted_shards,
    shard_mamba_mixer,
)
from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    base_linear,
    install_reduce_forward,
    is_lora_b_weight,
    keep_lora_bank,
    permute_rows,
    require_divisible,
    shard_attention_projections,
    shard_lora_b_rows,
    shard_reordered_rows,
    shard_row,
    shard_tied_embeddings,
    share_lora_dropout,
    stash_tp,
    tp_degree,
)

# Match by name: Liger can reassign modeling classes after instances exist.
ATTENTION_CLASS = "GraniteMoeHybridAttention"
MAMBA_CLASS = "GraniteMoeHybridMambaLayer"
MLP_CLASS = "GraniteMoeHybridMLP"
MLP_PERMUTED_PARAMS = frozenset(
    {"input_linear.weight", "input_linear.base_layer.weight"}
)


def is_granitemoehybrid_mixer(module: nn.Module) -> bool:
    """Whether ``module`` is a Granite 4 Mamba2 mixer."""
    return type(module).__name__ == MAMBA_CLASS


def _mlp_rank_ranges(hidden: int, tp: int, rank: int) -> list[tuple[int, int]]:
    """One rank's gate and up rows inside the fused ``[gate | up]`` projection."""
    local = hidden // tp
    return [
        (rank * local, (rank + 1) * local),
        (hidden + rank * local, hidden + (rank + 1) * local),
    ]


def _mlp_row_order(hidden: int, tp: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(start, stop)
            for rank in range(tp)
            for start, stop in _mlp_rank_ranges(hidden, tp, rank)
        ]
    )


def _shard_attention(module: GraniteMoeHybridAttention, mesh: DeviceMesh) -> None:
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


def _shard_mlp(module: GraniteMoeHybridMLP, mesh: DeviceMesh) -> None:
    tp = int(mesh.size())
    hidden = int(module.hidden_size)
    require_divisible(hidden, tp, "shared_intermediate_size")
    input_linear = module.input_linear
    base = base_linear(input_linear)
    order = _mlp_row_order(hidden, tp)
    shard_reordered_rows(base, "weight", order, mesh)
    stash_tp(base, mesh)
    stash_tp(input_linear, mesh)
    shard_lora_b_rows(input_linear, order, mesh)
    keep_lora_bank(input_linear, "lora_A", mesh)
    share_lora_dropout(input_linear, mesh)
    bias = shard_row(module.output_linear, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, module.output_linear if bias is not None else None)


def apply_granitemoehybrid_tensor_parallel(
    model: nn.Module, tp_mesh: DeviceMesh
) -> int:
    """Shard Granite 4 mixers, attention, shared MLPs, and the tied head on ``tp_mesh``.

    :param model: Module tree that may contain Granite 4 blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        name = type(module).__name__
        if name == ATTENTION_CLASS:
            _shard_attention(cast("GraniteMoeHybridAttention", module), tp_mesh)
        elif name == MAMBA_CLASS:
            mixer = cast("GraniteMoeHybridMambaLayer", module)
            # The gated norm averages over every Mamba2 head.
            shard_mamba_mixer(mixer, tp_mesh, int(mixer.intermediate_size))
        elif name == MLP_CLASS:
            _shard_mlp(cast("GraniteMoeHybridMLP", module), tp_mesh)
        else:
            continue
        count += 1
    # Every family plan runs on every model; only a Granite 4 model splits its tied head.
    if count:
        count += shard_tied_embeddings(model, tp_mesh)
    return count


def _mlp_row_orders(model: nn.Module) -> dict[str, torch.Tensor]:
    orders: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if type(module).__name__ != MLP_CLASS:
            continue
        tp = tp_degree(module)
        if tp <= 1:
            continue
        for relative, _ in module.named_parameters():
            if relative in MLP_PERMUTED_PARAMS or is_lora_b_weight(
                relative, "input_linear"
            ):
                prefix = f"{name}." if name else ""
                hidden = int(cast("GraniteMoeHybridMLP", module).hidden_size)
                orders[prefix + relative] = _mlp_row_order(hidden, tp)
    return orders


def granitemoehybrid_row_orders(model: nn.Module) -> dict[str, torch.Tensor]:
    """Source row of each stored dim-0 row of the permuted Granite 4 parameters.

    :param model: Model sharded by :func:`apply_granitemoehybrid_tensor_parallel`.
    :type model: nn.Module
    :return: Row order by live parameter FQN.
    :rtype: dict[str, torch.Tensor]
    """
    orders = mamba_row_orders(model, is_granitemoehybrid_mixer)
    orders.update(_mlp_row_orders(model))
    return orders


def restore_granitemoehybrid_tensor_parallel(
    model: nn.Module, tp_mesh: DeviceMesh
) -> None:
    """Regroup mixer and shared-MLP rows per rank after checkpoint weights load.

    :param model: Model sharded by :func:`apply_granitemoehybrid_tensor_parallel`.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """
    realign_mamba_permuted_shards(model, is_granitemoehybrid_mixer)
    with torch.no_grad():
        for fqn, order in sorted(_mlp_row_orders(model).items()):
            permute_rows(cast("DTensor", model.get_parameter(fqn)), order)
    for module in model.modules():
        if is_granitemoehybrid_mixer(module):
            mixer = cast("GraniteMoeHybridMambaLayer", module)
            shard_mamba_mixer(mixer, tp_mesh, int(mixer.intermediate_size))


TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_granitemoehybrid_tensor_parallel,
    restore=restore_granitemoehybrid_tensor_parallel,
    row_orders=granitemoehybrid_row_orders,
)
