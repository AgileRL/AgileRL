# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Variable-split token all-to-all with autograd, for expert-parallel dispatch and combine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist


@dataclass
class TokenDispatchState:
    """Metadata to reverse an EP token all-to-all after local expert compute."""

    input_splits: list[int]
    output_splits: list[int]
    ep_group: dist.ProcessGroup
    ep_degree: int
    num_local_experts: int
    permute_indices: torch.Tensor
    num_tokens_per_local_expert: torch.Tensor


def _all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Synchronous ``all_to_all_single`` returning ``output``."""
    dist.all_to_all_single(
        output,
        input.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


class AllToAllCtx(Protocol):
    output_splits: list[int]
    input_splits: list[int]
    group: dist.ProcessGroup


class AllToAllVar(torch.autograd.Function):
    """Variable-split all-to-all with autograd."""

    @staticmethod
    def forward(
        ctx: AllToAllCtx,
        input: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        out_rows = sum(output_splits)
        output = input.new_empty((out_rows, *tuple(input.shape[1:])))
        _all_to_all_single(output, input, output_splits, input_splits, group)
        return output

    @staticmethod
    def backward(
        ctx: AllToAllCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_output,) = grad_outputs
        grad_input = grad_output.new_empty(
            (sum(ctx.input_splits), *tuple(grad_output.shape[1:]))
        )
        _all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
        )
        return grad_input, None, None, None


def all_to_all_single_autograd(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Autograd-aware variable-split all-to-all over ``group``."""
    return AllToAllVar.apply(input, output_splits, input_splits, group)


def exchange_expert_counts(
    counts: torch.Tensor, ep_group: dist.ProcessGroup, ep_degree: int
) -> torch.Tensor:
    """Swap per-block expert row counts with every EP rank in one all-to-all.

    :param counts: ``[blocks, ep_degree * local_experts]`` rows this rank sends.
    :param ep_group: Expert-parallel process group.
    :param ep_degree: Ranks in ``ep_group``.
    :return: Same shape; row ``b`` is block ``b``'s counts this rank receives,
        source-rank major.
    """
    blocks, width = counts.shape
    send = counts.view(blocks, ep_degree, width // ep_degree).transpose(0, 1)
    send = send.contiguous()
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=ep_group)
    return received.transpose(0, 1).reshape(blocks, width)


def dispatch_state(
    received_counts: torch.Tensor,
    send_list: list[int],
    received_list: list[int],
    ep_group: dist.ProcessGroup,
    ep_degree: int,
    num_local_experts: int,
) -> TokenDispatchState:
    """All-to-all splits and the local-expert-major row order for one block.

    :param received_counts: Source-rank-major counts this rank receives.
    :param send_list: Host copy of the counts this rank sends.
    :param received_list: Host copy of ``received_counts``.
    :param ep_group: Expert-parallel process group.
    :param ep_degree: Ranks in ``ep_group``.
    :param num_local_experts: Experts owned by this rank.
    :return: Dispatch metadata for the block.
    """
    input_splits = [
        sum(send_list[rank * num_local_experts : (rank + 1) * num_local_experts])
        for rank in range(ep_degree)
    ]
    output_splits = [
        sum(received_list[rank * num_local_experts : (rank + 1) * num_local_experts])
        for rank in range(ep_degree)
    ]
    slots = torch.arange(ep_degree * num_local_experts, device=received_counts.device)
    expert_major = (slots % num_local_experts) * ep_degree + slots // num_local_experts
    row_keys = torch.repeat_interleave(
        expert_major, received_counts, output_size=sum(output_splits)
    )
    return TokenDispatchState(
        input_splits=input_splits,
        output_splits=output_splits,
        ep_group=ep_group,
        ep_degree=ep_degree,
        num_local_experts=num_local_experts,
        permute_indices=torch.argsort(row_keys, stable=True),
        num_tokens_per_local_expert=received_counts.view(
            ep_degree, num_local_experts
        ).sum(dim=0),
    )


def unpermute_from_local_expert_major(
    routed_output: torch.Tensor,
    permute_indices: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Put local-expert-major rows back in all-to-all (source-rank) order."""
    out = routed_output.new_empty((num_rows, *tuple(routed_output.shape[1:])))
    out[permute_indices] = routed_output
    return out


def token_dispatch(
    routed_input: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_group: dist.ProcessGroup,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, TokenDispatchState]:
    """All-to-all tokens to expert-owning ranks; return local-expert-major rows."""
    if num_tokens_per_expert.numel() != ep_degree * num_local_experts:
        msg = (
            f"num_tokens_per_expert length {num_tokens_per_expert.numel()} != "
            f"ep_degree * num_local_experts ({ep_degree} * {num_local_experts})"
        )
        raise ValueError(msg)

    counts = num_tokens_per_expert.view(1, -1)
    received = exchange_expert_counts(counts, ep_group, ep_degree)
    (send_list,), (received_list,) = torch.stack((counts, received)).tolist()
    state = dispatch_state(
        received[0],
        send_list,
        received_list,
        ep_group=ep_group,
        ep_degree=ep_degree,
        num_local_experts=num_local_experts,
    )
    dispatched = all_to_all_single_autograd(
        routed_input, state.output_splits, state.input_splits, ep_group
    )
    return (
        dispatched[state.permute_indices],
        state.num_tokens_per_local_expert,
        state,
    )


def token_combine(
    routed_output: torch.Tensor,
    state: TokenDispatchState,
) -> torch.Tensor:
    """Reverse :func:`token_dispatch` — unpermute then all-to-all back."""
    num_rows = sum(state.output_splits)
    unpermuted = unpermute_from_local_expert_major(
        routed_output, state.permute_indices, num_rows
    )
    return all_to_all_single_autograd(
        unpermuted,
        state.input_splits,
        state.output_splits,
        state.ep_group,
    )


def reference_dispatch_combine(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    ep_degree: int,
    num_experts: int,
) -> torch.Tensor:
    """Identity round-trip for dispatch/combine ordering in a single process."""
    if num_experts % ep_degree != 0:
        msg = f"num_experts ({num_experts}) must be divisible by ep ({ep_degree})"
        raise ValueError(msg)
    order = torch.argsort(expert_ids, stable=True)
    sorted_tokens = tokens[order]
    counts = torch.bincount(expert_ids, minlength=num_experts)
    num_local = num_experts // ep_degree
    pieces: list[torch.Tensor] = []
    for rank in range(ep_degree):
        start = rank * num_local
        end = start + num_local
        offsets = [int(counts[:start].sum().item()), int(counts[:end].sum().item())]
        pieces.append(sorted_tokens[offsets[0] : offsets[1]])
    restored_sorted = torch.cat(pieces, dim=0) if pieces else sorted_tokens
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return restored_sorted[inverse]
