# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Routed experts with split LoRA that recompute the up-projection in backward."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.autograd.function import once_differentiable

from agilerl.lora.moe.adapters import low_rank_delta
from agilerl.lora.moe.grouped_gemm import chunk_offsets, grouped_linear, grouped_matmul
from agilerl.lora.moe.layouts import expert_activation


@dataclass(frozen=True)
class AdapterSlot:
    """Scaling and mixed-routing row id of one adapter inside :class:`LoraExpertsFunction`."""

    scaling: float
    # Adapter id in the per-row ids under mixed routing, else ``None``.
    row_id: int | None


def _slot_factors(
    factors: Sequence[torch.Tensor],
    slots: Sequence[AdapterSlot],
    experts: slice,
    row_ids: torch.Tensor | None,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]]:
    """Per-adapter ``(A, B, scaling, row_mask)`` for one chunk; ``factors`` alternate A and B."""
    adapters = []
    for slot, lora_a, lora_b in zip(slots, factors[::2], factors[1::2], strict=True):
        mask = None
        if slot.row_id is not None:
            assert row_ids is not None
            mask = (row_ids == slot.row_id).to(dtype).unsqueeze(-1)
        adapters.append((lora_a[experts], lora_b[experts], slot.scaling, mask))
    return adapters


def _adapter_delta(
    rows: torch.Tensor,
    adapters: Sequence[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]],
    counts: list[int],
    offs: torch.Tensor,
) -> torch.Tensor | None:
    """Summed low-rank delta of every adapter for one chunk, or ``None`` without adapters."""
    total: torch.Tensor | None = None
    for lora_a, lora_b, scaling, mask in adapters:
        delta = low_rank_delta(rows, lora_a, lora_b, counts, offs, scaling)
        if mask is not None:
            delta.mul_(mask)
        total = delta if total is None else total.add_(delta)
    return total


class LoraExpertsCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    needs_input_grad: tuple[bool, ...]
    act_fn: Callable[[torch.Tensor], torch.Tensor]
    gated: bool
    plan: list[tuple[int, int, list[int], int, int]]
    up_slots: tuple[AdapterSlot, ...]
    down_slots: tuple[AdapterSlot, ...]

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None: ...


class LoraExpertsFunction(torch.autograd.Function):
    """Routed experts on frozen packed weights with split LoRA, recomputing the up-projection in backward.

    Saves the token rows, routing, and LoRA factors only. Rows are gathered and
    outputs scattered one expert chunk at a time, so neither the expert-sorted
    copy nor any ``[rows, intermediate]`` activation outlives its chunk.
    """

    @staticmethod
    def forward(
        ctx: LoraExpertsCtx,
        hidden_states: torch.Tensor,
        token_idx: torch.Tensor,
        routed_weights: torch.Tensor,
        group_ends: torch.Tensor,
        row_ids: torch.Tensor | None,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        gated: bool,
        plan: list[tuple[int, int, list[int], int, int]],
        up_slots: tuple[AdapterSlot, ...],
        down_slots: tuple[AdapterSlot, ...],
        *factors: torch.Tensor,
    ) -> torch.Tensor:
        up_factors = factors[: 2 * len(up_slots)]
        down_factors = factors[2 * len(up_slots) :]
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        for start_row, taken, counts, start_expert, end_expert in plan:
            stop = start_row + taken
            experts = slice(start_expert, end_expert)
            offs = chunk_offsets(group_ends, experts, start_row, stop)
            chunk_ids = None if row_ids is None else row_ids[start_row:stop]
            index = token_idx[start_row:stop]
            rows = hidden_states.index_select(0, index)
            projected = grouped_linear(rows, up_weight[experts], counts, offs)
            up_delta = _adapter_delta(
                rows,
                _slot_factors(up_factors, up_slots, experts, chunk_ids, rows.dtype),
                counts,
                offs,
            )
            if up_delta is not None:
                projected.add_(up_delta)
            intermediate = expert_activation(projected, act_fn, gated)
            out = grouped_linear(intermediate, down_weight[experts], counts, offs)
            down_delta = _adapter_delta(
                intermediate,
                _slot_factors(down_factors, down_slots, experts, chunk_ids, rows.dtype),
                counts,
                offs,
            )
            if down_delta is not None:
                out.add_(down_delta)
            out.mul_(routed_weights[start_row:stop])
            result.index_add_(0, index, out.to(torch.float32))
        ctx.save_for_backward(
            hidden_states,
            token_idx,
            routed_weights,
            group_ends,
            row_ids,
            up_weight,
            down_weight,
            *factors,
        )
        ctx.act_fn = act_fn
        ctx.gated = gated
        ctx.plan = plan
        ctx.up_slots = up_slots
        ctx.down_slots = down_slots
        return result.to(hidden_states.dtype)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: LoraExpertsCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            hidden_states,
            token_idx,
            routed_weights,
            group_ends,
            row_ids,
            up_weight,
            down_weight,
            *factors,
        ) = ctx.saved_tensors
        # Forward inputs ahead of ``*factors``.
        n_fixed = 12
        needs_rows = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[2]
        needs_factors = ctx.needs_input_grad[n_fixed:]
        n_up = 2 * len(ctx.up_slots)
        grad_hidden = (
            torch.zeros_like(hidden_states, dtype=torch.float32) if needs_rows else None
        )
        grad_weights = torch.zeros_like(routed_weights) if needs_weights else None
        grad_factors = [
            torch.zeros_like(factor, dtype=torch.float32) if needs else None
            for factor, needs in zip(factors, needs_factors, strict=True)
        ]
        for start_row, taken, counts, start_expert, end_expert in ctx.plan:
            stop = start_row + taken
            experts = slice(start_expert, end_expert)
            offs = chunk_offsets(group_ends, experts, start_row, stop)
            chunk_ids = None if row_ids is None else row_ids[start_row:stop]
            index = token_idx[start_row:stop]
            weights = routed_weights[start_row:stop]
            rows = hidden_states.index_select(0, index)
            grad_rows = grad_output.index_select(0, index).to(rows.dtype)
            # Leaves hold only this chunk's experts, so the local graph and its
            # factor grads stay chunk-sized.
            leaves = [
                factor[experts].detach().requires_grad_(needs)
                for factor, needs in zip(factors, needs_factors, strict=True)
            ]
            whole = slice(None)
            with torch.enable_grad():
                rows.requires_grad_(needs_rows)
                projected = grouped_linear(rows, up_weight[experts], counts, offs)
                up_delta = _adapter_delta(
                    rows,
                    _slot_factors(
                        leaves[:n_up], ctx.up_slots, whole, chunk_ids, rows.dtype
                    ),
                    counts,
                    offs,
                )
                if up_delta is not None:
                    projected = projected + up_delta
                intermediate = expert_activation(projected, ctx.act_fn, ctx.gated)
                down_delta = _adapter_delta(
                    intermediate,
                    _slot_factors(
                        leaves[n_up:], ctx.down_slots, whole, chunk_ids, rows.dtype
                    ),
                    counts,
                    offs,
                )
            # The output is linear in ``intermediate`` through the frozen base,
            # so its transpose gives both the intermediate grad and the
            # router-weight grad without recomputing the down projection.
            base_grad = grouped_matmul(grad_rows, down_weight[experts], counts, offs)
            if grad_weights is not None:
                score = (intermediate.detach().float() * base_grad.float()).sum(
                    -1, keepdim=True
                )
                if down_delta is not None:
                    score += (down_delta.detach().float() * grad_rows.float()).sum(
                        -1, keepdim=True
                    )
                grad_weights[start_row:stop] = score.to(grad_weights.dtype)
            scale = weights.to(grad_rows.dtype)
            outputs: list[torch.Tensor] = []
            output_grads: list[torch.Tensor] = []
            if intermediate.requires_grad:
                outputs.append(intermediate)
                output_grads.append(base_grad.mul_(scale))
            if down_delta is not None and down_delta.requires_grad:
                outputs.append(down_delta)
                output_grads.append(grad_rows * scale)
            targets = [
                (leaf, position)
                for position, leaf in enumerate([rows, *leaves])
                if leaf.requires_grad
            ]
            if not outputs or not targets:
                continue
            grads = torch.autograd.grad(
                outputs,
                [leaf for leaf, _ in targets],
                output_grads,
                allow_unused=True,
            )
            for (_, position), grad in zip(targets, grads, strict=True):
                if grad is None:
                    continue
                if position == 0:
                    assert grad_hidden is not None
                    grad_hidden.index_add_(0, index, grad.float())
                    continue
                target = grad_factors[position - 1]
                assert target is not None
                target[experts] += grad.float()
        return (
            None if grad_hidden is None else grad_hidden.to(hidden_states.dtype),
            None,
            grad_weights,
            *([None] * (n_fixed - 3)),
            *(
                None if grad is None else grad.to(factor.dtype)
                for grad, factor in zip(grad_factors, factors, strict=True)
            ),
        )
