# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Split rows replicated over a process group into per-rank spans, and gather them back."""

from __future__ import annotations

from itertools import pairwise
from typing import Protocol

import torch
import torch.distributed as dist


def replicated_row_span(n_rows: int, group: dist.ProcessGroup) -> tuple[int, int]:
    """This rank's ``[start, stop)`` share of ``n_rows`` rows replicated over ``group``."""
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    return n_rows * rank // size, n_rows * (rank + 1) // size


def _all_gather_row_spans(
    part: torch.Tensor, n_rows: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """Concatenate every rank's :func:`replicated_row_span` rows of ``n_rows``."""
    size = dist.get_world_size(group)
    spans = list(pairwise([n_rows * rank // size for rank in range(size + 1)]))
    width = max(stop - start for start, stop in spans)
    padded = part.new_zeros((width, *part.shape[1:]))
    padded[: part.shape[0]] = part
    pieces = [torch.empty_like(padded) for _ in range(size)]
    dist.all_gather(pieces, padded, group=group)
    return torch.cat(
        [
            piece[: stop - start]
            for piece, (start, stop) in zip(pieces, spans, strict=True)
        ]
    )


class ReplicatedRowsCtx(Protocol):
    group: dist.ProcessGroup
    n_rows: int


class SliceReplicatedRows(torch.autograd.Function):
    """Keep this rank's row span of a tensor replicated over ``group``.

    Backward all-gathers the span gradients and divides by the group size,
    which undoes the scale in :class:`GatherReplicatedRows`.
    """

    @staticmethod
    def forward(
        ctx: ReplicatedRowsCtx,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.n_rows = int(tensor.shape[0])
        start, stop = replicated_row_span(ctx.n_rows, group)
        return tensor[start:stop]

    @staticmethod
    def backward(
        ctx: ReplicatedRowsCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        full = _all_gather_row_spans(grad_output.contiguous(), ctx.n_rows, ctx.group)
        return full / dist.get_world_size(ctx.group), None


class GatherReplicatedRows(torch.autograd.Function):
    """All-gather row spans into the ``n_rows`` tensor replicated over ``group``.

    Every rank in ``group`` holds the same output gradient, and EP expert
    grads count each of those ranks' copy of a token (see
    :meth:`ParallelMesh.sync_grads`). Backward keeps this rank's span scaled
    by the group size so each token still counts once per rank.
    """

    @staticmethod
    def forward(
        ctx: ReplicatedRowsCtx,
        part: torch.Tensor,
        n_rows: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.n_rows = n_rows
        return _all_gather_row_spans(part.contiguous(), n_rows, group)

    @staticmethod
    def backward(
        ctx: ReplicatedRowsCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (grad_output,) = grad_outputs
        start, stop = replicated_row_span(ctx.n_rows, ctx.group)
        return grad_output[start:stop] * dist.get_world_size(ctx.group), None, None
