# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel sequence pad, shard, gather, and mixer all-to-all.

``cp`` splits the sequence: ranks in a CP group hold the same batch rows and
one even ``1/cp`` slice of each sequence. Ulysses attention restores full
sequence attention; Mamba/GDN all-to-all to ``[B, L, D/cp]`` for the SSM.
Log-probs are gathered before the loss. Inert at ``cp == 1``.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F


class GatherCpCtx(Protocol):
    """Autograd ctx for a context-parallel all-gather."""

    cp_rank: int
    shard_len: int


def validate_cp_degree(cp: int) -> None:
    """Reject a non-positive or non-integer context-parallel degree."""
    if not isinstance(cp, int) or isinstance(cp, bool):
        msg = f"cp must be an int, got {type(cp).__name__}"
        raise TypeError(msg)
    if cp < 1:
        msg = f"cp must be >= 1, got {cp}"
        raise ValueError(msg)


def validate_cp_style(cp_style: str) -> str:
    """Return the CP style, rejecting anything but Ulysses."""
    if cp_style != "ulysses":
        msg = (
            f"cp_style={cp_style!r} is not supported: only 'ulysses' is "
            "supported. Ulysses redistributes sequence shards to head shards "
            "with two all-to-alls around local flash attention."
        )
        raise ValueError(msg)
    return cp_style


def validate_cp_heads(
    num_attention_heads: int,
    num_key_value_heads: int | None,
    cp: int,
) -> None:
    """Validate Ulysses head sharding for ``cp``.

    Query heads split evenly; key/value heads either split evenly or replicate
    when fewer than ``cp``.
    """
    validate_cp_degree(cp)
    if cp <= 1:
        return
    if num_attention_heads % cp != 0:
        msg = (
            f"Ulysses CP requires num_attention_heads ({num_attention_heads}) "
            f"divisible by cp ({cp})"
        )
        raise ValueError(msg)
    kv_heads = (
        num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    )
    if kv_heads % cp != 0 and cp % kv_heads != 0:
        msg = (
            f"Ulysses GQA requires num_key_value_heads ({kv_heads}) divisible "
            f"by cp ({cp}) or cp divisible by num_key_value_heads "
            "(KV replicate path); got neither"
        )
        raise ValueError(msg)


def validate_cp_seq_len(seq_len: int, cp: int) -> None:
    """Fail when a sequence length cannot split evenly over ``cp``."""
    validate_cp_degree(cp)
    if cp <= 1:
        return
    if seq_len % cp != 0:
        msg = (
            f"sequence length ({seq_len}) must be divisible by cp ({cp}): "
            "uneven shards deadlock CP collectives. Pad so the action-frame "
            "length is a multiple of cp."
        )
        raise ValueError(msg)


def pad_right_for_cp(
    token_ids: torch.Tensor,
    cp: int,
    pad_token_id: int,
    *tensors: torch.Tensor | None,
    pad_values: tuple[int | float, ...] = (),
) -> tuple[torch.Tensor, ...]:
    """Pad a right-aligned batch so the action frame is a multiple of ``cp``.

    :param token_ids: ``(B, T)`` token ids.
    :param tensors: Optional ``(B, T-1)`` (or ``(B, T)``) tensors padded on
        the same tail. Length-``T`` tensors take ``extra`` columns; length
        ``T-1`` take ``extra`` as well (action frame).
    :param pad_values: Pad value per extra tensor; 0 when omitted.
    """
    validate_cp_degree(cp)
    if cp <= 1:
        return (token_ids, *tensors)
    remainder = (token_ids.shape[1] - 1) % cp
    if remainder == 0:
        return (token_ids, *tensors)
    extra = cp - remainder
    token_ids = F.pad(token_ids, (0, extra), value=pad_token_id)
    padded: list[torch.Tensor | None] = []
    for i, tensor in enumerate(tensors):
        if tensor is None:
            padded.append(None)
            continue
        value = pad_values[i] if i < len(pad_values) else 0
        padded.append(F.pad(tensor, (0, extra), value=value))
    return (token_ids, *padded)


def shard_for_cp(
    data: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Return this rank's even ``1/cp`` slice of ``data`` along ``seq_dim``."""
    validate_cp_degree(cp_size)
    if cp_size <= 1:
        return data
    length = data.shape[seq_dim]
    if length % cp_size != 0:
        msg = (
            f"CP requires sequence dimension {seq_dim} ({length}) divisible by "
            f"cp size ({cp_size}): uneven shards deadlock CP collectives."
        )
        raise ValueError(msg)
    return torch.chunk(data, cp_size, dim=seq_dim)[cp_rank]


class GatherForCp(torch.autograd.Function):
    """All-gather on dim 1; backward keeps this rank's slice and drops the rest."""

    @staticmethod
    def forward(
        ctx: GatherCpCtx,
        data: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        cp_size = dist.get_world_size(group)
        ctx.cp_rank = dist.get_rank(group)
        ctx.shard_len = data.shape[1]
        gathered = [torch.empty_like(data) for _ in range(cp_size)]
        dist.all_gather(gathered, data.contiguous(), group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(
        ctx: GatherCpCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        start = ctx.cp_rank * ctx.shard_len
        shard_grad = grad_output.narrow(1, start, ctx.shard_len).contiguous()
        return shard_grad, None


def gather_for_cp(data: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """Differentiable all-gather of per-rank slices, concatenated on dim 1."""
    return GatherForCp.apply(data, cp_group)


def gather_for_cp_wo_grad(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Non-differentiable all-gather of per-rank slices, concatenated on dim 1."""
    gathered = [torch.empty_like(data) for _ in range(cp_size)]
    dist.all_gather(gathered, data, group=cp_group)
    return torch.cat(gathered, dim=1)


def all_to_all_seq_to_hidden(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Redistribute ``[B, L_local, D] -> [B, L, D/cp]`` for mixer CP."""
    if cp_size <= 1:
        return data
    batch, seq_local, hidden = data.shape
    if hidden % cp_size != 0:
        msg = (
            f"hidden dim ({hidden}) must be divisible by cp_size ({cp_size}) "
            "for mixer sequence-to-hidden all-to-all"
        )
        raise ValueError(msg)
    hidden_local = hidden // cp_size
    swapped = (
        data.reshape(batch, seq_local, cp_size, hidden_local)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    exchanged = dist_nn.all_to_all_single(
        torch.empty_like(swapped), swapped, group=cp_group
    )
    return (
        exchanged.permute(1, 0, 2, 3)
        .contiguous()
        .reshape(batch, cp_size * seq_local, hidden_local)
    )


def all_to_all_hidden_to_seq(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Inverse of :func:`all_to_all_seq_to_hidden`."""
    if cp_size <= 1:
        return data
    batch, seq_global, hidden_local = data.shape
    if seq_global % cp_size != 0:
        msg = (
            f"global sequence length ({seq_global}) must be divisible by "
            f"cp_size ({cp_size})"
        )
        raise ValueError(msg)
    seq_local = seq_global // cp_size
    swapped = (
        data.reshape(batch, cp_size, seq_local, hidden_local)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    exchanged = dist_nn.all_to_all_single(
        torch.empty_like(swapped), swapped, group=cp_group
    )
    return (
        exchanged.permute(1, 2, 0, 3)
        .contiguous()
        .reshape(batch, seq_local, hidden_local * cp_size)
    )
