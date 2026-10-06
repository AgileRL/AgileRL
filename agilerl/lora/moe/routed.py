# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Self-routing packed-experts forward with split-LoRA deltas, in expert row chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper
from torch.distributed.tensor import DTensor

from agilerl.lora.moe.adapters import (
    ExpertLora,
    low_rank_delta,
    partitioned_lora_delta,
    resolve_adapters,
    stacked_lora_weights,
    token_adapter_ids,
)
from agilerl.lora.moe.grouped_gemm import (
    chunk_offsets,
    grouped_linear,
    iter_expert_row_chunks,
)
from agilerl.lora.moe.layouts import (
    expert_activation,
    routed_experts_act_fn,
    routed_projection_names,
)
from agilerl.lora.moe.recompute import (
    AdapterSlot,
    LoraExpertsConfig,
    LoraExpertsFunction,
)

# Widest [rows, features] activation of one routed-expert row chunk. A chunk's
# backward holds about a dozen buffers that size (peak near 0.8 GiB). Each
# chunk adds a fixed set of small kernel launches that bound the step on the host.
ROUTED_EXPERT_CHUNK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RoutedRows:
    """Expert-sorted ``tokens * top_k`` rows: where each comes from and how it is weighted."""

    # Rows per expert.
    counts: torch.Tensor
    # Source token of each row.
    token_idx: torch.Tensor
    # ``[rows, 1]`` router weight of each row.
    routed_weights: torch.Tensor
    # Adapter id of each row under mixed routing, else ``None``.
    row_ids: torch.Tensor | None


def _add_expert_loras(
    destination: torch.Tensor,
    rows: torch.Tensor,
    loras: Sequence[ExpertLora],
    counts: list[int],
    offs: torch.Tensor,
    experts: slice,
    num_experts: int,
    row_ids: torch.Tensor | None,
) -> None:
    """Add each adapter's low-rank delta for one chunk of expert-sorted rows."""
    for lora in loras:
        if lora.stacked is None:
            expert_ids = torch.repeat_interleave(
                torch.arange(experts.start, experts.stop), torch.tensor(counts)
            ).to(rows.device)
            delta = partitioned_lora_delta(
                lora.wrapper, rows, lora.adapter, expert_ids, num_experts
            )
        else:
            a3, b3 = lora.stacked
            delta = low_rank_delta(
                rows,
                a3[experts],
                b3[experts],
                counts,
                offs,
                lora.wrapper.scaling[lora.adapter],
            )
        if lora.row_id is not None:
            assert row_ids is not None
            delta.mul_((row_ids == lora.row_id).to(delta.dtype).unsqueeze(-1))
        if delta.dtype != destination.dtype:
            delta = delta.to(dtype=destination.dtype)
        destination.add_(delta)


class ScatterRowsCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...

    def mark_dirty(self, *args: torch.Tensor) -> None: ...


class ScatterRows(torch.autograd.Function):
    """Add source rows into a buffer."""

    @staticmethod
    def forward(
        ctx: ScatterRowsCtx,
        base: torch.Tensor,
        index: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.mark_dirty(base)
        base.index_add_(0, index, source)
        return base

    @staticmethod
    def backward(
        ctx: ScatterRowsCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        (grad_out,) = grad_outputs
        (index,) = ctx.saved_tensors
        return grad_out, None, grad_out.index_select(0, index)


def _scatter_routed_expert_chunks(
    hidden_states: torch.Tensor,
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
    routing: RoutedRows,
    up_loras: Sequence[ExpertLora],
    down_loras: Sequence[ExpertLora],
) -> torch.Tensor:
    """Run the routed expert forward in row chunks and scatter into the layer output.

    Expanded expert rows are ``top_k`` times the token count, so the full up
    and down activations do not fit beside the gathered hidden states.
    """
    result = torch.zeros_like(hidden_states)
    num_experts = up_weight.shape[0]
    row_bytes = max(up_weight.shape[1], down_weight.shape[1]) * x.element_size()
    max_rows = max(1, ROUTED_EXPERT_CHUNK_BYTES // max(row_bytes, 1))
    plan = list(iter_expert_row_chunks(routing.counts, max_rows))
    sizes = [taken for _, taken, _, _, _ in plan]
    # Chunk offsets stay on device so the loop issues no host sync.
    group_ends = torch.cumsum(routing.counts, dim=0)
    # Split once so backward concatenates the chunk grads into one buffer.
    chunks = zip(plan, x.split(sizes), routing.routed_weights.split(sizes), strict=True)
    for (
        start_row,
        taken,
        local_counts,
        start_expert,
        end_expert,
    ), rows, weights in chunks:
        stop = start_row + taken
        experts = slice(start_expert, end_expert)
        offs = chunk_offsets(group_ends, experts, start_row, stop)
        chunk_ids = None if routing.row_ids is None else routing.row_ids[start_row:stop]
        projected = grouped_linear(rows, up_weight[experts], local_counts, offs)
        _add_expert_loras(
            projected,
            rows,
            up_loras,
            local_counts,
            offs,
            experts,
            num_experts,
            chunk_ids,
        )
        intermediate = expert_activation(projected, act_fn, gated)
        del projected
        down = grouped_linear(intermediate, down_weight[experts], local_counts, offs)
        _add_expert_loras(
            down,
            intermediate,
            down_loras,
            local_counts,
            offs,
            experts,
            num_experts,
            chunk_ids,
        )
        del intermediate
        down.mul_(weights)
        if down.dtype != result.dtype:
            down = down.to(dtype=result.dtype)
        result = ScatterRows.apply(result, routing.token_idx[start_row:stop], down)
    return result


def _recompute_routed_experts(
    hidden_states: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
    routing: RoutedRows,
    up_loras: Sequence[ExpertLora],
    down_loras: Sequence[ExpertLora],
) -> torch.Tensor:
    """Run :class:`LoraExpertsFunction` over the routed rows with stacked LoRA factors."""
    if isinstance(up_weight, DTensor):
        up_weight = up_weight.to_local()
    if isinstance(down_weight, DTensor):
        down_weight = down_weight.to_local()
    row_bytes = (
        max(up_weight.shape[1], down_weight.shape[1]) * hidden_states.element_size()
    )
    max_rows = max(1, ROUTED_EXPERT_CHUNK_BYTES // row_bytes)
    config = LoraExpertsConfig(
        act_fn,
        gated,
        list(iter_expert_row_chunks(routing.counts, max_rows)),
        tuple(
            AdapterSlot(lora.wrapper.scaling[lora.adapter], lora.row_id)
            for lora in up_loras
        ),
        tuple(
            AdapterSlot(lora.wrapper.scaling[lora.adapter], lora.row_id)
            for lora in down_loras
        ),
    )
    factors = []
    for lora in (*up_loras, *down_loras):
        assert lora.stacked is not None
        factors.extend(lora.stacked)
    return LoraExpertsFunction.apply(
        hidden_states,
        routing.token_idx,
        routing.routed_weights,
        torch.cumsum(routing.counts, dim=0),
        routing.row_ids,
        up_weight,
        down_weight,
        config,
        *factors,
    )


def routed_experts_local_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    chain: dict[str, ParamWrapper] | None = None,
    adapters: dict[str, list[str]] | None = None,
    routing: Sequence[str] | None = None,
    already_grouped: bool | None = None,
    recompute: bool = False,
) -> torch.Tensor:
    """Packed routed-experts forward with optional split-LoRA deltas.

    :param already_grouped: ``True`` when each row has one expert and rows are
        in expert order, which skips the sort and gather; ``None`` detects it.
    :param recompute: Run :class:`LoraExpertsFunction`, which saves only the
        token rows, routing and LoRA factors and recomputes the up-projection in
        backward. Applies when both base weights are frozen and the LoRA factors
        are not partitioned DTensors.
    """
    projections = routed_projection_names(experts)
    if projections is None:
        msg = "Routed experts module does not match a supported packed layout."
        raise RuntimeError(msg)
    up_name, gated = projections
    up_weight = getattr(experts, up_name)
    down_weight = experts.down_proj
    act_fn = routed_experts_act_fn(experts)
    assert isinstance(up_weight, torch.Tensor)
    assert isinstance(down_weight, torch.Tensor)

    num_experts = up_weight.shape[0]
    if chain is not None and adapters is None:
        adapters = {name: resolve_adapters(w) for name, w in chain.items()}
    adapters = adapters or {}
    wrappers = chain or {}

    top_k = top_k_index.shape[-1]
    if already_grouped and top_k != 1:
        msg = f"already_grouped needs one expert per row, got top_k={top_k}."
        raise ValueError(msg)
    flat_experts = top_k_index.reshape(-1)
    order = (
        torch.arange(flat_experts.shape[0], device=flat_experts.device)
        if already_grouped
        else torch.argsort(flat_experts, stable=True)
    )
    if already_grouped is None:
        # Rows already sit in expert order. Indexing them would clone the activation.
        already_grouped = top_k == 1 and torch.equal(
            order, torch.arange(order.shape[0], device=order.device)
        )
    counts = torch.bincount(flat_experts, minlength=num_experts)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    routed_weights = top_k_weights.reshape(-1)[order].unsqueeze(-1)
    row_ids: torch.Tensor | None = None
    id_map: dict[str, int] | None = None
    if routing is not None and len(set(routing)) > 1:
        row_ids, id_map = token_adapter_ids(routing, hidden_states.shape[0], token_idx)

    def expert_loras(param_name: str) -> list[ExpertLora]:
        return [
            ExpertLora(
                wrappers[param_name],
                name,
                stacked_lora_weights(
                    wrappers[param_name], name, hidden_states.dtype, num_experts
                ),
                None if id_map is None else id_map[name],
            )
            for name in adapters.get(param_name, [])
        ]

    up_loras = expert_loras(up_name)
    down_loras = expert_loras("down_proj")
    routed_rows = RoutedRows(counts, token_idx, routed_weights, row_ids)
    if (
        recompute
        and not up_weight.requires_grad
        and not down_weight.requires_grad
        and all(lora.stacked is not None for lora in (*up_loras, *down_loras))
    ):
        return _recompute_routed_experts(
            hidden_states,
            up_weight,
            down_weight,
            act_fn,
            gated,
            routed_rows,
            up_loras,
            down_loras,
        )
    x = hidden_states if already_grouped else hidden_states[token_idx]
    return _scatter_routed_expert_chunks(
        hidden_states,
        x,
        up_weight,
        down_weight,
        act_fn,
        gated,
        routed_rows,
        up_loras,
        down_loras,
    )
