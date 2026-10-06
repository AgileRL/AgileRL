# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Adapter selection and split low-rank deltas for packed-expert LoRA wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper
from torch.distributed.tensor import DTensor

from agilerl.lora.fused import ROUTING_STATE, uniform_routed_adapter
from agilerl.lora.moe.grouped_gemm import (
    add_grouped_linear,
    counts_list,
    counts_tensor,
    grouped_linear,
)


@dataclass(frozen=True)
class ExpertLora:
    """One adapter on a routed-experts projection, with weights prepared once per forward."""

    wrapper: ParamWrapper
    adapter: str
    # ``None`` while the LoRA weights are partitioned DTensors.
    stacked: tuple[torch.Tensor, torch.Tensor] | None
    # Adapter id in the per-row ids under mixed routing, else ``None``.
    row_id: int | None


def mixed_routing(layer: nn.Module) -> list[str] | None:
    """Per-row fused routing on *layer* when it mixes adapters, else ``None``."""
    routing = ROUTING_STATE.get(layer)
    return routing if routing and len(set(routing)) > 1 else None


def adapters_in_routing(wrapper: ParamWrapper, routing: Sequence[str]) -> list[str]:
    """Adapter names from *routing* that this wrapper actually hosts."""
    return [name for name in dict.fromkeys(routing) if name in wrapper.lora_A]


def token_adapter_ids(
    routing: Sequence[str], n_rows: int, token_idx: torch.Tensor
) -> tuple[torch.Tensor, dict[str, int]]:
    """Expand per-sample fused routing to tokens, then permute into expert-sorted order.

    ``token_idx`` is the gate's ``batch_index``: ``token_idx[i]`` is the original
    token for expert-sorted row ``i``.
    """
    factor, remainder = divmod(n_rows, len(routing))
    if remainder:
        msg = (
            f"Fused adapter routing covers {len(routing)} rows but the "
            f"experts input's leading dimension is {n_rows}."
        )
        raise ValueError(msg)
    expanded = [name for name in routing for _ in range(factor)]
    name_to_id = {name: index for index, name in enumerate(dict.fromkeys(expanded))}
    table = torch.tensor(
        [name_to_id[name] for name in expanded],
        device=token_idx.device,
        dtype=torch.int64,
    )
    return table[token_idx], name_to_id


def resolve_adapters(wrapper: ParamWrapper) -> list[str]:
    """Adapter names to apply on this forward, honoring fused routing and adapter state."""
    routed = uniform_routed_adapter(wrapper)
    if routed is not None:
        return [routed] if routed in wrapper.lora_A else []
    if wrapper.disable_adapters:
        if wrapper.merged:
            wrapper.unmerge()
        return []
    return [
        name
        for name in wrapper.active_adapters
        if name in wrapper.lora_A and name not in wrapper.merged_adapters
    ]


def expert_counts(
    expert_size: Sequence[int] | torch.Tensor, num_experts: int
) -> list[int]:
    """Normalize a per-expert row-count spec, validating the calling convention."""
    counts = counts_list(expert_size)
    if len(counts) != num_experts:
        msg = (
            f"Expected {num_experts} per-expert counts, got {len(counts)}; "
            "the wrapped experts module does not follow the sorted-rows "
            "calling convention."
        )
        raise ValueError(msg)
    return counts


def wrapper_chain(wrapper: ParamWrapper) -> dict[str, ParamWrapper]:
    """Map targeted parameter name to wrapper for a (possibly nested) wrapper chain."""
    chain: dict[str, ParamWrapper] = {}
    module: nn.Module = wrapper
    while isinstance(module, ParamWrapper):
        chain[module.parameter_name] = module
        module = module.base_layer
    return chain


def stacked_lora_weights(
    wrapper: ParamWrapper,
    adapter: str,
    dtype: torch.dtype,
    num_experts: int | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """LoRA A as ``[E, r, in]`` and contiguous B as ``[E, out, r]`` in ``dtype``, or ``None`` for DTensors."""
    weight_a = wrapper.lora_A[adapter].weight
    weight_b = wrapper.lora_B[adapter].weight
    assert isinstance(weight_a, torch.Tensor)
    assert isinstance(weight_b, torch.Tensor)
    if isinstance(weight_a, DTensor) or isinstance(weight_b, DTensor):
        return None
    rank = wrapper.r[adapter]
    # Stacked PEFT layouts: A is ``[E*r, in]`` or ``[E, r, in]``;
    # B is ``[out, E*r]`` or ``[out, r, E]``.
    if weight_a.ndim == 3:
        a3 = weight_a
    else:
        experts = num_experts if num_experts is not None else wrapper.num_experts
        a3 = weight_a.view(experts, rank, weight_a.shape[1])
    experts = a3.shape[0]
    if weight_b.ndim == 3:
        if weight_b.shape[0] == experts:
            b_grouped = weight_b
        else:
            # ``[out, r, E]`` with experts on the last dim
            b_grouped = weight_b.permute(2, 0, 1)
    else:
        b_grouped = weight_b.view(weight_b.shape[0], rank, experts).permute(2, 0, 1)
    # The multiply stays in the activation dtype. A wider LoRA weight is cast.
    # The permuted B has strides no GEMM accepts; one copy serves every chunk.
    return (
        a3.to(dtype=dtype),
        b_grouped.to(dtype=dtype, memory_format=torch.contiguous_format),
    )


def partitioned_lora_delta(
    wrapper: ParamWrapper,
    x: torch.Tensor,
    adapter: str,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """LoRA delta through the adapter ``Linear`` modules, for still-partitioned DTensor weights."""
    rank = wrapper.r[adapter]
    total = x.shape[0]
    rows = torch.arange(total, device=x.device)
    a_full = wrapper.lora_A[adapter](x).view(total, num_experts, rank)
    gated = torch.zeros(total, rank, num_experts, dtype=a_full.dtype, device=x.device)
    gated[rows, :, expert_ids] = a_full[rows, expert_ids]
    delta = wrapper.lora_B[adapter](gated.reshape(total, rank * num_experts))
    return delta * wrapper.scaling[adapter]


def low_rank_delta(
    rows: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    counts: list[int],
    offs: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Scaled ``rows @ A[e]^T @ B[e]^T`` for expert-sorted rows with stacked ``[E, r, in]`` / ``[E, out, r]`` factors."""
    down = grouped_linear(rows, lora_a, counts, offs)
    return grouped_linear(down, lora_b, counts, offs).mul_(scaling)


def split_lora_delta(
    wrapper: ParamWrapper,
    x: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    adapter: str,
    offs: torch.Tensor | None = None,
    num_experts: int | None = None,
    destination: torch.Tensor | None = None,
) -> torch.Tensor:
    """Low-rank delta for expert-sorted rows without materializing per-expert full-rank weights."""
    stacked = stacked_lora_weights(wrapper, adapter, x.dtype, num_experts)
    # Prefer the grouped GEMM on dense weights. The Linear fallback is for
    # still-partitioned DTensors (FSDP leftover outside a rooted forward).
    if stacked is None:
        experts = num_experts if num_experts is not None else wrapper.num_experts
        expert_ids = torch.repeat_interleave(
            torch.arange(experts, device=x.device),
            counts_tensor(counts, x.device),
        )
        delta = partitioned_lora_delta(wrapper, x, adapter, expert_ids, experts)
        if destination is None:
            return delta
        if delta.dtype != destination.dtype:
            delta = delta.to(dtype=destination.dtype)
        destination.add_(delta)
        return destination

    a3, b3 = stacked
    scaling = wrapper.scaling[adapter]
    down = grouped_linear(x, a3, counts, offs)
    if destination is None:
        up = grouped_linear(down, b3, counts, offs)
        # ``up`` is a fresh GEMM output. Scaling it in place skips a second full copy.
        return up.mul_(scaling)
    add_grouped_linear(destination, down, b3, counts, scaling)
    return destination
