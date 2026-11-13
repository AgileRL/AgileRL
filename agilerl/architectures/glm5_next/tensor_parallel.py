# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel GLM-5.3 KDA linear attention, DSA attention, MLPs, and vision blocks.

Each KDA rank runs ``linear_num_heads/tp`` heads, then all-reduces the output.
``q_proj``, ``k_proj``, ``v_proj``, ``f_b_proj``, ``g_b_proj``, ``b_proj``,
``dt_bias``, and ``A_log`` split by head. ``conv1d`` rows are grouped per rank
as that rank's Q, K, and V channels, so a ``Shard(0)`` chunk is one local conv.
The low-rank gate inputs ``f_a_proj`` and ``g_a_proj`` and the per-head output
norm stay whole.

DSA attention splits ``q_b_proj`` and ``kv_b_proj`` by head and ``o_proj`` by
input column. ``q_a_proj``, ``kv_a_proj_with_mqa``, their norms, and the
indexer stay whole on every rank, so each rank selects the same top-k keys.

Dense and shared-expert MLPs use the column- and row-parallel layers in
:mod:`agilerl.distributed.tensor_parallel`. Packed experts stay on the
expert-parallel mesh and the router stays whole.

Vision attention runs ``num_heads/tp`` heads per rank. Fused ``qkv`` rows and
bias are grouped per rank as that rank's Q, K, and V heads, and ``proj`` splits
by input column; the per-head ``q_norm`` and ``k_norm`` stay whole. Vision MLPs
and the patch merger's SwiGLU use the column- and row-parallel layers; the
merger's input ``proj`` and its LayerNorm stay whole. The patch embed, the
spatial-merge ``downsample`` conv, and the vision RMSNorms run outside any
region and stay whole.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from types import MethodType
from typing import TypeVar, cast

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextLinearAttention,
    Glm5NextTextMLP,
    Glm5NextVisionAttention,
    Glm5NextVisionMLP,
    Glm5NextVisionPatchMerger,
)

from agilerl.architectures.runtime import TensorParallelPlan
from agilerl.distributed.tensor_parallel import (
    install_reduce_forward,
    keep_lora_bank,
    permute_rows,
    read_lora_b_rows,
    replicate_on_tp,
    require_divisible,
    shard_attention_projections,
    shard_column,
    shard_row,
    share_lora_dropout,
    stash_tp,
)

# Match by name: Liger can reassign modeling classes after instances exist.
LINEAR_ATTENTION_CLASS = "Glm5NextTextLinearAttention"
ATTENTION_CLASS = "Glm5NextTextAttention"
MLP_CLASS = "Glm5NextTextMLP"

ModuleT = TypeVar("ModuleT", bound=nn.Module)
SwiGluMlp = Glm5NextTextMLP | Glm5NextVisionMLP | Glm5NextVisionPatchMerger


def _qkv_rank_ranges(
    num_heads: int, head_dim: int, tp: int, rank: int
) -> list[tuple[int, int]]:
    """Q, K, and V rows of one rank inside a fused ``[Q|K|V]`` projection."""
    section = num_heads * head_dim
    span = section // tp
    start = rank * span
    return [
        (part * section + start, part * section + start + span) for part in range(3)
    ]


def _qkv_row_order(num_heads: int, head_dim: int, tp: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(start, stop)
            for rank in range(tp)
            for start, stop in _qkv_rank_ranges(num_heads, head_dim, tp, rank)
        ]
    )


def _base_linear(linear: nn.Module) -> nn.Linear:
    base = linear.get_base_layer() if isinstance(linear, BaseTunerLayer) else linear
    return cast("nn.Linear", base)


def _shard_rows(
    module: nn.Module,
    name: str,
    mesh: DeviceMesh,
    row_order: torch.Tensor | None = None,
) -> None:
    """``Shard(0)`` ``module.<name>`` on ``mesh``, after reordering its rows by ``row_order``."""
    param = module.get_parameter(name)
    if isinstance(param, DTensor):
        return
    require_divisible(int(param.shape[0]), int(mesh.size()), name)
    rows = param.detach()
    if row_order is not None:
        index = row_order if rows.is_meta else row_order.to(rows.device)
        rows = rows.index_select(0, index).contiguous()
    module.register_parameter(
        name,
        nn.Parameter(
            distribute_tensor(rows, mesh, [Shard(0)]),
            requires_grad=param.requires_grad,
        ),
    )


def _replicate_params(module: nn.Module, mesh: DeviceMesh) -> None:
    """Make every parameter under ``module`` a whole copy on each TP rank."""
    for child in module.modules():
        names = [name for name, _ in child.named_parameters(recurse=False)]
        replicate_on_tp(child, mesh, *names)


def _replicate_linear(linear: nn.Module, mesh: DeviceMesh) -> None:
    """Keep a linear and its LoRA banks whole on each TP rank, with one dropout mask."""
    replicate_on_tp(_base_linear(linear), mesh, "weight", "bias")
    keep_lora_bank(linear, "lora_A", mesh)
    keep_lora_bank(linear, "lora_B", mesh)
    share_lora_dropout(linear, mesh)


@contextmanager
def _local_kda_heads(mixer: Glm5NextTextLinearAttention, tp: int) -> Iterator[None]:
    """Head count and Q/K/V width read by one tensor-parallel KDA forward."""
    forget_gate = mixer.forget_gate
    num_heads = mixer.num_heads
    qkv_dim = mixer.qkv_dim
    mixer.num_heads = forget_gate.num_heads = num_heads // tp
    mixer.qkv_dim = forget_gate.qkv_dim = qkv_dim // tp
    try:
        yield
    finally:
        mixer.num_heads = forget_gate.num_heads = num_heads
        mixer.qkv_dim = forget_gate.qkv_dim = qkv_dim


@contextmanager
def _local_vision_heads(attn: Glm5NextVisionAttention, tp: int) -> Iterator[None]:
    """Head count read by one tensor-parallel vision-attention forward."""
    num_heads = attn.num_heads
    attn.num_heads = num_heads // tp
    try:
        yield
    finally:
        attn.num_heads = num_heads


def _install_local_forward(
    module: ModuleT,
    local_sizes: Callable[[ModuleT, int], AbstractContextManager[None]],
    bias_owner: nn.Module | None,
) -> None:
    """Run ``module`` with this rank's head sizes inside an all-reduced region."""
    if getattr(module, "_agilerl_local_heads_tp", False):
        return
    hf_forward = module.forward

    def local_forward(self: ModuleT, *args: object, **kwargs: object) -> torch.Tensor:
        with local_sizes(self, cast("int", self._tp_degree)):
            return hf_forward(*args, **kwargs)

    object.__setattr__(module, "forward", MethodType(local_forward, module))
    object.__setattr__(module, "_agilerl_local_heads_tp", True)
    install_reduce_forward(module, bias_owner)


def _shard_linear_attention(
    mixer: Glm5NextTextLinearAttention, mesh: DeviceMesh
) -> None:
    tp = int(mesh.size())
    require_divisible(mixer.num_heads, tp, "linear_num_heads")
    for linear in (
        mixer.q_proj,
        mixer.k_proj,
        mixer.v_proj,
        mixer.b_proj,
        mixer.g_b_proj,
        mixer.forget_gate.f_b_proj,
    ):
        shard_column(linear, mesh)
    order = _qkv_row_order(mixer.num_heads, mixer.head_dim, tp)
    _shard_rows(mixer.conv1d, "weight", mesh, order)
    _shard_rows(mixer.forget_gate, "dt_bias", mesh)
    _shard_rows(mixer.forget_gate, "A_log", mesh)
    _replicate_linear(mixer.forget_gate.f_a_proj, mesh)
    _replicate_linear(mixer.g_a_proj, mesh)
    _replicate_params(mixer.o_norm, mesh)
    shard_row(mixer.o_proj, mesh)
    stash_tp(mixer, mesh)
    _install_local_forward(mixer, _local_kda_heads, None)


def _shard_attention(module: Glm5NextTextAttention, mesh: DeviceMesh) -> None:
    _replicate_linear(module.q_a_proj, mesh)
    _replicate_params(module.q_a_layernorm, mesh)
    _replicate_linear(module.kv_a_proj_with_mqa, mesh)
    _replicate_params(module.kv_a_layernorm, mesh)
    # Shared-index layers reuse the previous layer's top-k and have no indexer.
    if module.indexer is not None:
        _replicate_params(module.indexer, mesh)
    config = module.config
    shard_attention_projections(
        module,
        mesh,
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        module.qk_nope_head_dim + module.v_head_dim,
        module.q_b_proj,
        module.kv_b_proj,
        None,
        module.o_proj,
    )


def _shard_mlp(module: SwiGluMlp, mesh: DeviceMesh) -> None:
    intermediate = _base_linear(module.gate_proj).out_features
    require_divisible(intermediate, int(mesh.size()), "intermediate_size")
    shard_column(module.gate_proj, mesh)
    shard_column(module.up_proj, mesh)
    bias = shard_row(module.down_proj, mesh)
    stash_tp(module, mesh)
    install_reduce_forward(module, module.down_proj if bias is not None else None)


def _shard_vision_attention(attn: Glm5NextVisionAttention, mesh: DeviceMesh) -> None:
    tp = int(mesh.size())
    require_divisible(attn.num_heads, tp, "vision num_heads")
    order = _qkv_row_order(attn.num_heads, attn.head_dim, tp)
    qkv = _base_linear(attn.qkv)
    _shard_rows(qkv, "weight", mesh, order)
    if qkv.bias is not None:
        _shard_rows(qkv, "bias", mesh, order)
    rows = _qkv_rank_ranges(attn.num_heads, attn.head_dim, tp, mesh.get_local_rank())
    read_lora_b_rows(attn.qkv, mesh, tuple(rows))
    keep_lora_bank(attn.qkv, "lora_A", mesh)
    share_lora_dropout(attn.qkv, mesh)
    _replicate_params(attn.q_norm, mesh)
    _replicate_params(attn.k_norm, mesh)
    bias = shard_row(attn.proj, mesh)
    stash_tp(attn, mesh)
    _install_local_forward(
        attn, _local_vision_heads, attn.proj if bias is not None else None
    )


def _shard_vision_merger(merger: Glm5NextVisionPatchMerger, mesh: DeviceMesh) -> None:
    _replicate_linear(merger.proj, mesh)
    _replicate_params(merger.post_projection_norm, mesh)
    _shard_mlp(merger, mesh)


def apply_glm5_next_tensor_parallel(model: nn.Module, tp_mesh: DeviceMesh) -> int:
    """Shard GLM-5.3 KDA, DSA attention, MLPs, shared experts, and vision blocks on ``tp_mesh``.

    :param model: Module tree that may contain GLM-5.3 text or vision blocks.
    :type model: nn.Module
    :param tp_mesh: Tensor-parallel mesh.
    :type tp_mesh: DeviceMesh
    :return: How many modules were sharded.
    :rtype: int
    """
    count = 0
    for module in model.modules():
        name = type(module).__name__
        if name == LINEAR_ATTENTION_CLASS:
            _shard_linear_attention(
                cast("Glm5NextTextLinearAttention", module), tp_mesh
            )
        elif name == ATTENTION_CLASS:
            _shard_attention(cast("Glm5NextTextAttention", module), tp_mesh)
        elif name == MLP_CLASS or isinstance(module, Glm5NextVisionMLP):
            _shard_mlp(cast("SwiGluMlp", module), tp_mesh)
        elif isinstance(module, Glm5NextVisionAttention):
            _shard_vision_attention(module, tp_mesh)
        elif isinstance(module, Glm5NextVisionPatchMerger):
            _shard_vision_merger(module, tp_mesh)
        else:
            continue
        count += 1
    return count


def glm5_next_row_orders(model: nn.Module) -> dict[str, torch.Tensor]:
    """Source row of each stored dim-0 row of every KDA ``conv1d`` and vision ``qkv``.

    :param model: Model sharded by :func:`apply_glm5_next_tensor_parallel`.
    :type model: nn.Module
    :return: Row order by live parameter FQN.
    :rtype: dict[str, torch.Tensor]
    """
    orders: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        tp = int(getattr(module, "_tp_degree", 1))
        if tp <= 1:
            continue
        prefix = f"{name}." if name else ""
        if type(module).__name__ == LINEAR_ATTENTION_CLASS:
            mixer = cast("Glm5NextTextLinearAttention", module)
            orders[f"{prefix}conv1d.weight"] = _qkv_row_order(
                mixer.num_heads, mixer.head_dim, tp
            )
        elif isinstance(module, Glm5NextVisionAttention):
            order = _qkv_row_order(module.num_heads, module.head_dim, tp)
            qkv = "qkv.base_layer" if isinstance(module.qkv, BaseTunerLayer) else "qkv"
            orders[f"{prefix}{qkv}.weight"] = order
            if _base_linear(module.qkv).bias is not None:
                orders[f"{prefix}{qkv}.bias"] = order
    return orders


def restore_glm5_next_tensor_parallel(model: nn.Module, _tp_mesh: DeviceMesh) -> None:
    """Regroup KDA ``conv1d`` and vision ``qkv`` rows per rank after checkpoint weights load.

    :param model: Model sharded by :func:`apply_glm5_next_tensor_parallel`.
    :type model: nn.Module
    :param _tp_mesh: Tensor-parallel mesh.
    :type _tp_mesh: DeviceMesh
    :return: None
    :rtype: None
    """
    with torch.no_grad():
        for fqn, order in sorted(glm5_next_row_orders(model).items()):
            permute_rows(cast("DTensor", model.get_parameter(fqn)), order)


TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    shard=apply_glm5_next_tensor_parallel,
    restore=restore_glm5_next_tensor_parallel,
    row_orders=glm5_next_row_orders,
)
