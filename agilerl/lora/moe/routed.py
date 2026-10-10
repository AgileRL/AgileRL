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
    lora_operands,
    low_rank_delta,
    partitioned_lora_delta,
    resolve_adapters,
    stacked_lora_weights,
    token_adapter_ids,
)
from agilerl.lora.moe.grouped_gemm import (
    ROUTED_EXPERT_CHUNK_BYTES,
    expert_row_counts,
    grouped_matmul,
    grouped_operand,
    row_chunk_offsets,
)
from agilerl.lora.moe.layouts import (
    expert_activation,
    routed_experts_act_fn,
    routed_projection_names,
)
from agilerl.lora.moe.recompute import LoraExpertsConfig, LoraExpertsFunction


@dataclass(frozen=True)
class RoutedRows:
    """Expert-sorted ``tokens * top_k`` rows: where each comes from and how it is weighted."""

    # Rows per expert, on device.
    counts: torch.Tensor
    # Source token of each row.
    token_idx: torch.Tensor
    # ``[rows, 1]`` router weight of each row.
    routed_weights: torch.Tensor
    # Adapter id of each row under mixed routing, else ``None``.
    row_ids: torch.Tensor | None


@dataclass(frozen=True)
class ChunkedLora:
    """One adapter on a routed-experts projection, prepared once per layer for every row chunk."""

    lora: ExpertLora
    # Scaled ``A`` and ``B`` grouped-GEMM operands; ``None`` while partitioned.
    operands: tuple[torch.Tensor, torch.Tensor] | None
    # ``[rows, 1]`` row mask of each chunk under mixed routing, else ``None``.
    masks: tuple[torch.Tensor, ...] | None
    # Expert id of each row of each chunk, for partitioned weights only.
    expert_ids: tuple[torch.Tensor, ...] | None


def _chunked_loras(
    loras: Sequence[ExpertLora],
    routing: RoutedRows,
    sizes: list[int],
    dtype: torch.dtype,
) -> list[ChunkedLora]:
    """Per-layer operands and per-chunk row tensors of each adapter."""
    num_experts = routing.counts.shape[0]
    expert_ids: tuple[torch.Tensor, ...] | None = None
    chunked = []
    for lora in loras:
        masks = None
        if lora.row_id is not None:
            assert routing.row_ids is not None
            mask = (routing.row_ids == lora.row_id).to(dtype).unsqueeze(-1)
            masks = mask.split(sizes)
        if lora.stacked is not None:
            scaling = lora.wrapper.scaling[lora.adapter]
            chunked.append(
                ChunkedLora(lora, lora_operands(*lora.stacked, scaling), masks, None)
            )
            continue
        if expert_ids is None:
            expert_ids = torch.repeat_interleave(
                torch.arange(num_experts, device=routing.counts.device),
                routing.counts,
                output_size=sum(sizes),
            ).split(sizes)
        chunked.append(ChunkedLora(lora, None, masks, expert_ids))
    return chunked


def _add_expert_loras(
    destination: torch.Tensor,
    rows: torch.Tensor,
    loras: Sequence[ChunkedLora],
    chunk: int,
    offs: torch.Tensor,
) -> None:
    """Add each adapter's low-rank delta for one chunk of expert-sorted rows."""
    for chunked in loras:
        lora = chunked.lora
        if chunked.operands is None:
            assert chunked.expert_ids is not None
            delta = partitioned_lora_delta(
                lora.wrapper,
                rows,
                lora.adapter,
                chunked.expert_ids[chunk],
                offs.shape[0],
            )
        else:
            delta = low_rank_delta(rows, *chunked.operands, offs)
        if chunked.masks is not None:
            delta.mul_(chunked.masks[chunk])
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


def _chunk_rows(
    up_weight: torch.Tensor, down_weight: torch.Tensor, itemsize: int, chunk_bytes: int
) -> int:
    """Rows per chunk so the widest ``[rows, features]`` activation fits ``chunk_bytes``."""
    row_bytes = max(up_weight.shape[1], down_weight.shape[1]) * itemsize
    return max(1, chunk_bytes // row_bytes)


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
    chunk_bytes: int,
) -> torch.Tensor:
    """Run the routed expert forward in row chunks and scatter into the layer output.

    Expanded expert rows are ``top_k`` times the token count, so the full up
    and down activations do not fit beside the gathered hidden states.
    """
    result = torch.zeros_like(hidden_states)
    max_rows = _chunk_rows(up_weight, down_weight, x.element_size(), chunk_bytes)
    sizes, offsets = row_chunk_offsets(routing.counts, x.shape[0], max_rows)
    up_operand = grouped_operand(up_weight)
    down_operand = grouped_operand(down_weight)
    up_chunked = _chunked_loras(up_loras, routing, sizes, x.dtype)
    down_chunked = _chunked_loras(down_loras, routing, sizes, x.dtype)
    # Split once so backward concatenates the chunk grads into one buffer.
    chunks = zip(
        x.split(sizes),
        routing.routed_weights.split(sizes),
        routing.token_idx.split(sizes),
        offsets.unbind(),
        strict=True,
    )
    for chunk, (rows, weights, index, offs) in enumerate(chunks):
        projected = grouped_matmul(rows, up_operand, offs)
        _add_expert_loras(projected, rows, up_chunked, chunk, offs)
        intermediate = expert_activation(projected, act_fn, gated)
        del projected
        down = grouped_matmul(intermediate, down_operand, offs)
        _add_expert_loras(down, intermediate, down_chunked, chunk, offs)
        del intermediate
        down.mul_(weights)
        if down.dtype != result.dtype:
            down = down.to(dtype=result.dtype)
        result = ScatterRows.apply(result, index, down)
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
    chunk_bytes: int,
) -> torch.Tensor:
    """Run :class:`LoraExpertsFunction` over the routed rows with stacked LoRA factors."""
    if isinstance(up_weight, DTensor):
        up_weight = up_weight.to_local()
    if isinstance(down_weight, DTensor):
        down_weight = down_weight.to_local()
    max_rows = _chunk_rows(
        up_weight, down_weight, hidden_states.element_size(), chunk_bytes
    )
    sizes, offsets = row_chunk_offsets(
        routing.counts, routing.token_idx.shape[0], max_rows
    )
    config = LoraExpertsConfig(
        act_fn,
        gated,
        tuple(sizes),
        tuple(lora.row_id for lora in up_loras),
        tuple(lora.row_id for lora in down_loras),
    )
    operands = []
    for lora in (*up_loras, *down_loras):
        assert lora.stacked is not None
        operands.extend(
            lora_operands(*lora.stacked, lora.wrapper.scaling[lora.adapter])
        )
    return LoraExpertsFunction.apply(
        hidden_states,
        routing.token_idx,
        routing.routed_weights,
        offsets,
        routing.row_ids,
        up_weight,
        down_weight,
        config,
        *operands,
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
    chunk_bytes: int = ROUTED_EXPERT_CHUNK_BYTES,
) -> torch.Tensor:
    """Packed routed-experts forward with optional split-LoRA deltas.

    :param already_grouped: ``True`` when each row has one expert and rows are
        in expert order, which skips the sort and gather; ``None`` detects it.
    :param recompute: Run :class:`LoraExpertsFunction`, which saves only the
        token rows, routing and LoRA factors and recomputes the up-projection in
        backward. Applies when both base weights are frozen and the LoRA factors
        are not partitioned DTensors.
    :param chunk_bytes: Widest ``[rows, features]`` activation of one row
        chunk; see :data:`ROUTED_EXPERT_CHUNK_BYTES`.
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
    counts = expert_row_counts(flat_experts, num_experts)
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
            chunk_bytes,
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
        chunk_bytes,
    )
