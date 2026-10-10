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
from agilerl.lora.moe.grouped_gemm import grouped_matmul, grouped_operand
from agilerl.lora.moe.layouts import expert_activation


@dataclass(frozen=True)
class LoraExpertsConfig:
    """Non-tensor inputs of :class:`LoraExpertsFunction`."""

    act_fn: Callable[[torch.Tensor], torch.Tensor]
    gated: bool
    # Rows per chunk of the expert-sorted rows.
    sizes: tuple[int, ...]
    # Mixed-routing row id of each up / down adapter, ``None`` when unrouted.
    up_row_ids: tuple[int | None, ...]
    down_row_ids: tuple[int | None, ...]


def _chunk_masks(
    row_ids: torch.Tensor | None,
    adapter_row_ids: Sequence[int | None],
    sizes: Sequence[int],
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, ...] | None]:
    """Per-adapter ``[rows, 1]`` row masks split into chunks, ``None`` when unrouted."""
    masks: list[tuple[torch.Tensor, ...] | None] = []
    for row_id in adapter_row_ids:
        if row_id is None:
            masks.append(None)
            continue
        assert row_ids is not None
        masks.append((row_ids == row_id).to(dtype).unsqueeze(-1).split(sizes))
    return masks


def _adapter_deltas(
    rows: torch.Tensor,
    operands: Sequence[torch.Tensor],
    masks: Sequence[tuple[torch.Tensor, ...] | None],
    chunk: int,
    offs: torch.Tensor,
) -> list[torch.Tensor]:
    """Low-rank delta of every adapter for one chunk; ``operands`` alternate scaled A and B."""
    deltas = []
    for a_operand, b_operand, mask in zip(
        operands[::2], operands[1::2], masks, strict=True
    ):
        delta = low_rank_delta(rows, a_operand, b_operand, offs)
        if mask is not None:
            delta.mul_(mask[chunk])
        deltas.append(delta)
    return deltas


def _sum_deltas(deltas: list[torch.Tensor]) -> torch.Tensor | None:
    """Sum of the adapter deltas, or ``None`` without adapters."""
    if not deltas:
        return None
    total = deltas[0]
    for delta in deltas[1:]:
        total = total.add_(delta)
    return total


class LoraExpertsCtx(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    needs_input_grad: tuple[bool, ...]
    config: LoraExpertsConfig

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None: ...


class LoraExpertsFunction(torch.autograd.Function):
    """Routed experts on frozen packed weights with split LoRA, recomputing the up-projection in backward.

    Saves the token rows, routing, and LoRA operands only. Rows are gathered
    and outputs scattered one row chunk at a time, so neither the
    expert-sorted copy nor any ``[rows, intermediate]`` activation outlives its
    chunk. Each chunk's grouped GEMMs see every local expert, empty groups
    included, through that chunk's row of ``offsets``.
    """

    @staticmethod
    def forward(
        ctx: LoraExpertsCtx,
        hidden_states: torch.Tensor,
        token_idx: torch.Tensor,
        routed_weights: torch.Tensor,
        offsets: torch.Tensor,
        row_ids: torch.Tensor | None,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        config: LoraExpertsConfig,
        *operands: torch.Tensor,
    ) -> torch.Tensor:
        n_up = 2 * len(config.up_row_ids)
        dtype = hidden_states.dtype
        up_masks = _chunk_masks(row_ids, config.up_row_ids, config.sizes, dtype)
        down_masks = _chunk_masks(row_ids, config.down_row_ids, config.sizes, dtype)
        up_operand = grouped_operand(up_weight)
        down_operand = grouped_operand(down_weight)
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        chunks = zip(
            token_idx.split(config.sizes),
            routed_weights.split(config.sizes),
            offsets.unbind(),
            strict=True,
        )
        for chunk, (index, weights, offs) in enumerate(chunks):
            rows = hidden_states.index_select(0, index)
            projected = grouped_matmul(rows, up_operand, offs)
            for delta in _adapter_deltas(rows, operands[:n_up], up_masks, chunk, offs):
                projected.add_(delta)
            intermediate = expert_activation(projected, config.act_fn, config.gated)
            out = grouped_matmul(intermediate, down_operand, offs)
            for delta in _adapter_deltas(
                intermediate, operands[n_up:], down_masks, chunk, offs
            ):
                out.add_(delta)
            out.mul_(weights)
            result.index_add_(0, index, out.to(torch.float32))
        ctx.save_for_backward(
            hidden_states,
            token_idx,
            routed_weights,
            offsets,
            row_ids,
            up_weight,
            down_weight,
            *operands,
        )
        ctx.config = config
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
            offsets,
            row_ids,
            up_weight,
            down_weight,
            *operands,
        ) = ctx.saved_tensors
        config = ctx.config
        # Forward inputs ahead of ``*operands``.
        n_fixed = 8
        needs_rows = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[2]
        needs_operands = ctx.needs_input_grad[n_fixed:]
        n_up = 2 * len(config.up_row_ids)
        dtype = hidden_states.dtype
        up_masks = _chunk_masks(row_ids, config.up_row_ids, config.sizes, dtype)
        down_masks = _chunk_masks(row_ids, config.down_row_ids, config.sizes, dtype)
        up_operand = grouped_operand(up_weight)
        # ``grad_rows @ down[e]`` is the down projection's transpose.
        down_transposed = grouped_operand(down_weight).transpose(-2, -1)
        grad_hidden = (
            torch.zeros_like(hidden_states, dtype=torch.float32) if needs_rows else None
        )
        grad_weights = torch.zeros_like(routed_weights) if needs_weights else None
        chunk_weight_grads = (
            None if grad_weights is None else grad_weights.split(config.sizes)
        )
        grad_operands = [
            torch.zeros_like(operand, dtype=torch.float32) if needs else None
            for operand, needs in zip(operands, needs_operands, strict=True)
        ]
        # Leaves span every local expert; each chunk's grads add into the full stack.
        leaves = [
            operand.detach().requires_grad_(needs)
            for operand, needs in zip(operands, needs_operands, strict=True)
        ]
        scales = routed_weights.to(grad_output.dtype)
        chunks = zip(
            token_idx.split(config.sizes),
            scales.split(config.sizes),
            offsets.unbind(),
            strict=True,
        )
        for chunk, (index, scale, offs) in enumerate(chunks):
            rows = hidden_states.index_select(0, index)
            grad_rows = grad_output.index_select(0, index)
            with torch.enable_grad():
                rows.requires_grad_(needs_rows)
                projected = grouped_matmul(rows, up_operand, offs)
                for delta in _adapter_deltas(
                    rows, leaves[:n_up], up_masks, chunk, offs
                ):
                    projected.add_(delta)
                intermediate = expert_activation(projected, config.act_fn, config.gated)
                down_delta = _sum_deltas(
                    _adapter_deltas(
                        intermediate, leaves[n_up:], down_masks, chunk, offs
                    )
                )
            # The output is linear in ``intermediate`` through the frozen base,
            # so its transpose gives both the intermediate grad and the
            # router-weight grad without recomputing the down projection.
            base_grad = grouped_matmul(grad_rows, down_transposed, offs)
            if chunk_weight_grads is not None:
                score = (intermediate.detach().float() * base_grad.float()).sum(
                    -1, keepdim=True
                )
                if down_delta is not None:
                    score += (down_delta.detach().float() * grad_rows.float()).sum(
                        -1, keepdim=True
                    )
                chunk_weight_grads[chunk].copy_(score)
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
                target = grad_operands[position - 1]
                assert target is not None
                target.add_(grad)
        return (
            None if grad_hidden is None else grad_hidden.to(hidden_states.dtype),
            None,
            grad_weights,
            *([None] * (n_fixed - 3)),
            *(
                None if grad is None else grad.to(operand.dtype)
                for grad, operand in zip(grad_operands, operands, strict=True)
            ),
        )
