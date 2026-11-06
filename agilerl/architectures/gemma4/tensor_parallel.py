# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Gemma 4 text attention and MLPs.

Each rank runs ``num_attention_heads/tp`` query heads at that layer's head dim.
A KV-shared layer has no k/v projection; it reads the local heads its source
layer stored on the same rank. ``q_norm`` and ``k_norm`` stay whole. MLP
``gate_proj`` and ``up_proj`` split by column and ``down_proj`` by row, with an
all-reduce after it. The per-layer embedding splits its packed
``num_layers * per_layer_dim`` columns and all-gathers each lookup. The tied
token embedding and LM head split their hidden columns and share one weight.
The per-layer input projections and the vision and audio towers stay whole.
"""

from __future__ import annotations

from types import MethodType
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.experimental import implicit_replication
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4Model,
    Gemma4TextAttention,
    Gemma4TextMLP,
    Gemma4TextScaledWordEmbedding,
)

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    GatherFromTPRegion,
    install_reduce_forward,
    replicate_on_tp,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_row,
    shard_tied_embeddings,
    stash_tp,
)

# Liger swaps Gemma4TextMLP for LigerGEGLUMLPForGemma4 on the modeling module.
MLP_CLASS_NAMES = ("Gemma4TextMLP", "LigerGEGLUMLPForGemma4")


def _shard_attention(module: Gemma4TextAttention, mesh: DeviceMesh) -> None:
    num_q = int(module.config.num_attention_heads)
    shared = module.is_kv_shared_layer
    shard_attention_projections(
        module,
        mesh,
        num_q,
        num_q // int(module.num_key_value_groups),
        int(module.head_dim),
        module.q_proj,
        None if shared else module.k_proj,
        None if shared else module.v_proj,
        module.o_proj,
    )
    norms = [module.q_norm] if shared else [module.q_norm, module.k_norm]
    for norm in norms:
        names = [name for name, _ in norm.named_parameters(recurse=False)]
        replicate_on_tp(norm, mesh, *names)


def _shard_mlp(module: Gemma4TextMLP, mesh: DeviceMesh) -> None:
    require_divisible(
        int(module.intermediate_size), int(mesh.size()), "intermediate_size"
    )
    shard_column(module.gate_proj, mesh)
    shard_column(module.up_proj, mesh)
    if shard_row(module.down_proj, mesh) is not None:
        msg = (
            "Gemma 4 MLP down_proj has a bias; its tensor-parallel all-reduce "
            "does not add one"
        )
        raise ValueError(msg)
    stash_tp(module, mesh)
    install_reduce_forward(module, None)


def _per_layer_embedding_forward(
    self: Gemma4TextScaledWordEmbedding, input_ids: torch.Tensor
) -> torch.Tensor:
    weight = cast("DTensor", self.weight).to_local()
    local = F.embedding(input_ids, weight, self.padding_idx)
    local = local * self.embed_scale.to(weight.dtype)
    rank = cast("DeviceMesh", self._tp_mesh).get_local_rank()
    return GatherFromTPRegion.apply(local, rank, self._tp_group)


def _shard_per_layer_embedding(module: nn.Module, mesh: DeviceMesh) -> None:
    weight = module.get_parameter("weight")
    require_divisible(int(weight.shape[1]), int(mesh.size()), "per-layer embedding dim")
    sharded = distribute_tensor(weight.detach(), mesh, [Shard(1)])
    module.register_parameter(
        "weight", nn.Parameter(sharded, requires_grad=weight.requires_grad)
    )
    stash_tp(module, mesh)
    # The module is its own FSDP unit, so the forward must not re-enter ``__call__``.
    module.forward = MethodType(_per_layer_embedding_forward, module)


def _multimodal_forward(self: Gemma4Model, *args: object, **kwargs: object) -> object:
    # HF reads a pad row of the split token embedding into a value it drops when
    # input_ids are given; mixed DTensor ops would raise there.
    with implicit_replication():
        return Gemma4Model.forward(self, *args, **kwargs)


def apply_gemma4_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard Gemma 4 text blocks, the per-layer embedding and the tied head on ``tp_mesh``.

    :param model: Module tree that may contain Gemma 4 text blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = 0
    for path, module in model.named_modules():
        # Match by name: Liger reassigns the modeling MLP class after instances exist.
        name = type(module).__name__
        if name == "Gemma4TextAttention":
            _shard_attention(cast("Gemma4TextAttention", module), tp_mesh)
        elif name in MLP_CLASS_NAMES:
            _shard_mlp(cast("Gemma4TextMLP", module), tp_mesh)
        elif path.rpartition(".")[2] == "embed_tokens_per_layer":
            _shard_per_layer_embedding(module, tp_mesh)
        else:
            continue
        count += 1
    # Every family plan runs on every model; only a Gemma 4 model splits its tied head.
    if count:
        count += shard_tied_embeddings(model, tp_mesh)
        for module in model.modules():
            if isinstance(module, Gemma4Model):
                module.forward = MethodType(_multimodal_forward, module)
    return count


GEMMA4_TENSOR_PARALLEL_PLAN = TensorParallelPlan(shard=apply_gemma4_tensor_parallel)
