# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Deal segment rows out across data-parallel ranks so each runs about the same rows.

Ranks run their micro-batches in lockstep, so a micro-batch takes as long as
the longest row any rank runs in it. Balanced ranks run the same number of
rows, and every rank's i-th row is about as long as every other rank's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from agilerl.distributed import collective_device, is_distributed
from agilerl.utils.llm_utils import attention_mask_from_padded_ids
from agilerl.utils.segment_rows import SegmentRows, is_per_token

TOKENS, ACTION_MASK, SAMPLING_LOGPS, TURN_IDS = range(4)
"""Positions of a row's first flat parts; ``TURN_IDS`` only when the rows carry
turn ids. Row values and vision rows follow."""


@dataclass(frozen=True)
class RankRows:
    """Layout of one rank's training rows: enough to receive them without their tensors.

    :param costs: Real tokens per row.
    :param lengths: Episode tokens each row spans.
    :param episodes: Source episode per row.
    :param starts: First episode token per row.
    :param image_counts: Vision rows per row.
    :param logp_sizes: Sampling log-probs per row, ``-1`` where a row has none.
    :param num_episodes: Episodes in the rank's batch.
    :param value_shapes: Shape of one row of each row value, ``None`` for a
        value per action position.
    :param value_dtypes: Dtype of each row value.
    :param has_turn_ids: Whether the rows carry turn ids.
    :param has_logps: Whether the rows carry sampling log-probs.
    :param pixel_shape: Shape of one vision row, or ``None`` in a text batch.
    :param pixel_dtype: Dtype of the vision rows, or ``None`` in a text batch.
    """

    costs: list[int]
    lengths: list[int]
    episodes: list[int]
    starts: list[int]
    image_counts: list[int]
    logp_sizes: list[int]
    num_episodes: int
    value_shapes: tuple[tuple[int, ...] | None, ...]
    value_dtypes: tuple[torch.dtype, ...]
    has_turn_ids: bool
    has_logps: bool
    pixel_shape: tuple[int, ...] | None
    pixel_dtype: torch.dtype | None

    @property
    def first_value_part(self) -> int:
        """Position of the first row value among a row's flat parts."""
        return TURN_IDS + int(self.has_turn_ids)

    def part_dtypes(self) -> list[torch.dtype]:
        """Dtype of each flat part a row travels as.

        :return: Dtypes of tokens, action mask and sampling log-probs, then
            turn ids when carried, each row value, and vision rows when carried.
        """
        dtypes = [torch.long, torch.uint8, torch.float32]
        if self.has_turn_ids:
            dtypes.append(torch.long)
        dtypes.extend(self.value_dtypes)
        if self.pixel_dtype is not None:
            dtypes.append(self.pixel_dtype)
        return dtypes

    def part_sizes(self, row: int) -> list[int]:
        """Elements of each flat part of one row, in :meth:`part_dtypes` order.

        :param row: Row index within this rank's training rows.
        :return: Element count per part.
        """
        length = self.lengths[row]
        sizes = [length, length - 1, max(self.logp_sizes[row], 0)]
        if self.has_turn_ids:
            sizes.append(length - 1)
        sizes.extend(
            length - 1 if shape is None else int(np.prod(shape))
            for shape in self.value_shapes
        )
        if self.pixel_shape is not None:
            sizes.append(self.image_counts[row] * int(np.prod(self.pixel_shape)))
        return sizes


def balanced_row_plan(
    row_costs: Sequence[Sequence[int]],
    ranks_per_group: int,
) -> list[list[tuple[int, int]]]:
    """Rows each rank runs, in run order, as ``(source rank, source row)``.

    Rows sorted most tokens first are cut into rounds of one row per rank, so
    each rank's i-th row costs about as much as every other rank's. Ranks in
    one lockstep group (a contiguous block of ``ranks_per_group``) wait on
    their costliest row every micro-batch, and groups meet only at the
    gradient sync, so a group's time is the sum of its round maxima. Each
    round's costliest rows go one per group, the costliest to the group with
    the least time so far. Every other row stays on its own rank when that
    rank has no row yet.

    :param row_costs: Tokens of each training row, per rank.
    :param ranks_per_group: Ranks per lockstep group.
    :return: Per destination rank, its rows in run order.
    :raises ValueError: If ``ranks_per_group`` does not divide the world.
    """
    world_size = len(row_costs)
    if world_size % ranks_per_group != 0:
        msg = (
            f"ranks_per_group ({ranks_per_group}) must divide the world ({world_size})"
        )
        raise ValueError(msg)
    num_groups = world_size // ranks_per_group
    ordered = sorted(
        (
            (rank, row)
            for rank, costs in enumerate(row_costs)
            for row in range(len(costs))
        ),
        key=lambda item: (-row_costs[item[0]][item[1]], item),
    )
    plan: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
    group_time = [0] * num_groups
    for start in range(0, len(ordered), world_size):
        round_rows = ordered[start : start + world_size]
        round_sources = {source for source, _ in round_rows}
        free = set(range(world_size))
        lightest_first = sorted(range(num_groups), key=lambda g: (group_time[g], g))
        for (source, row), group in zip(
            round_rows[:num_groups], lightest_first, strict=False
        ):
            members = range(group * ranks_per_group, (group + 1) * ranks_per_group)
            if source in members:
                destination = source
            else:
                # A member with no row of its own this round keeps its peers' rows local.
                destination = next(
                    (rank for rank in members if rank not in round_sources),
                    members[0],
                )
            plan[destination].append((source, row))
            free.remove(destination)
            group_time[group] += row_costs[source][row]
        moved = []
        for source, row in round_rows[num_groups:]:
            if source in free:
                plan[source].append((source, row))
                free.remove(source)
            else:
                moved.append((source, row))
        for item, destination in zip(moved, sorted(free), strict=False):
            plan[destination].append(item)
    return plan


def balance_rows_across_ranks(
    rows: SegmentRows,
    train_rows: npt.NDArray[np.intp],
    row_values: Sequence[tuple[torch.Tensor, float]],
    pad_token_id: int,
    shard_group_size: int | None,
) -> tuple[SegmentRows, list[torch.Tensor]] | None:
    """Deal every rank's training rows out so each rank runs about the same rows.

    A row trains on whichever rank it lands on with its own values (its
    episode's advantages, returns, log-probs, ...). Rows outside
    ``train_rows`` are dropped. Rows run in :func:`balanced_row_plan` order,
    and ``row_episodes`` numbers episodes across ranks, rank by rank.

    :param rows: This rank's split rows, before filler rows.
    :param train_rows: Rows that survive the advantage filter.
    :param row_values: Values of ``rows``, each ``(R, ...)`` per row or
        ``(R, W - 1)`` per action position, with the value a per-position
        value takes past its row's end.
    :param pad_token_id: Token id that pads rows to a common width.
    :param shard_group_size: Ranks per FSDP weight-shard group, which run in
        lockstep; ``None`` is the world.
    :return: The rows this rank runs with each of their ``row_values``, or
        ``None`` on one process or when there are fewer training rows than ranks.
    :raises ValueError: If ranks disagree on which row fields they carry.
    """
    if not is_distributed() or dist.get_world_size() == 1:
        return None
    world_size, rank = dist.get_world_size(), dist.get_rank()
    values = [value for value, _ in row_values]
    local = _rank_rows(rows, train_rows, values, pad_token_id)
    layouts: list[RankRows] = [local] * world_size
    dist.all_gather_object(layouts, local)
    if sum(len(layout.costs) for layout in layouts) < world_size:
        return None
    fields = {
        (
            layout.value_shapes,
            layout.value_dtypes,
            layout.has_turn_ids,
            layout.pixel_shape,
            layout.pixel_dtype,
        )
        for layout in layouts
    }
    if len(fields) > 1:
        msg = f"Ranks carry different segment row fields: {sorted(map(str, fields))}."
        raise ValueError(msg)
    plan = balanced_row_plan(
        [layout.costs for layout in layouts],
        world_size if shard_group_size is None else shard_group_size,
    )
    local_parts = _local_parts(rows, train_rows, values, local)
    received = _exchange_parts(local_parts, layouts, plan, rank)
    parts = [
        local_parts[row] if source == rank else received[source].pop(0)
        for source, row in plan[rank]
    ]
    offsets = np.cumsum([0, *(layout.num_episodes for layout in layouts)])
    return _assemble(
        rows,
        [pad for _, pad in row_values],
        plan[rank],
        layouts,
        offsets,
        parts,
        pad_token_id,
    )


def _rank_rows(
    rows: SegmentRows,
    train_rows: npt.NDArray[np.intp],
    values: Sequence[torch.Tensor],
    pad_token_id: int,
) -> RankRows:
    """Layout of this rank's training rows.

    :param rows: This rank's split rows.
    :param train_rows: Rows that survive the advantage filter.
    :param values: Row values of ``rows``.
    :param pad_token_id: Token id of the trailing padding.
    :return: The layout.
    """
    counts = rows.pixel_image_counts or [0] * len(rows.row_episodes)
    logps = rows.sampling_logps
    pixels = rows.pixel_values
    real_tokens = attention_mask_from_padded_ids(
        rows.token_ids[train_rows], pad_token_id
    ).sum(-1)
    return RankRows(
        costs=real_tokens.tolist(),
        lengths=(rows.row_ends - rows.row_starts)[train_rows].tolist(),
        episodes=rows.row_episodes[train_rows].tolist(),
        starts=rows.row_starts[train_rows].tolist(),
        image_counts=[int(counts[row]) for row in train_rows],
        logp_sizes=[
            -1 if logps is None or logps[row] is None else int(logps[row].numel())
            for row in train_rows
        ],
        num_episodes=int(rows.row_episodes.max(initial=-1)) + 1,
        value_shapes=tuple(
            None if is_per_token(value) else tuple(value.shape[1:]) for value in values
        ),
        value_dtypes=tuple(value.dtype for value in values),
        has_turn_ids=rows.turn_ids is not None,
        has_logps=logps is not None,
        pixel_shape=tuple(pixels.shape[1:]) if pixels is not None else None,
        pixel_dtype=pixels.dtype if pixels is not None else None,
    )


def _local_parts(
    rows: SegmentRows,
    train_rows: npt.NDArray[np.intp],
    values: Sequence[torch.Tensor],
    layout: RankRows,
) -> list[list[torch.Tensor]]:
    """Each training row as flat parts, in :meth:`RankRows.part_dtypes` order.

    :param rows: This rank's split rows.
    :param train_rows: Rows that survive the advantage filter.
    :param values: Row values of ``rows``.
    :param layout: This rank's layout.
    :return: Per training row, its flat parts.
    """
    counts = rows.pixel_image_counts or [0] * len(rows.row_episodes)
    image_offsets = np.cumsum([0, *counts])
    no_logps = rows.action_masks.new_zeros(0, dtype=torch.float32)
    all_parts = []
    for row, length in zip(train_rows, layout.lengths, strict=True):
        logps = rows.sampling_logps[row] if rows.sampling_logps is not None else None
        parts = [
            rows.token_ids[row, :length].long(),
            rows.action_masks[row, : length - 1].to(torch.uint8),
            logps.float() if logps is not None else no_logps,
        ]
        if rows.turn_ids is not None:
            parts.append(rows.turn_ids[row, : length - 1].long())
        parts.extend(
            value[row, : length - 1] if shape is None else value[row].reshape(-1)
            for value, shape in zip(values, layout.value_shapes, strict=True)
        )
        if rows.pixel_values is not None:
            images = rows.pixel_values[image_offsets[row] : image_offsets[row + 1]]
            parts.append(images.reshape(-1))
        all_parts.append(parts)
    return all_parts


def _exchange_parts(
    local_parts: list[list[torch.Tensor]],
    layouts: list[RankRows],
    plan: list[list[tuple[int, int]]],
    rank: int,
) -> dict[int, list[list[torch.Tensor]]]:
    """Send each row to the rank that runs it and receive this rank's rows.

    One point-to-point message per part kind and peer; both sides size it
    from the layouts, and skip it when it is empty.

    :param local_parts: Flat parts of this rank's training rows.
    :param layouts: Every rank's layout.
    :param plan: Per destination rank, its rows as ``(source rank, source row)``.
    :param rank: This rank.
    :return: Per source rank, the flat parts of the rows it sent, in ``plan`` order.
    """
    device = collective_device()
    dtypes = layouts[rank].part_dtypes()
    ops = []
    received: dict[int, list[list[torch.Tensor]]] = {}
    for peer in range(len(layouts)):
        if peer == rank:
            continue
        sent = [local_parts[row] for source, row in plan[peer] if source == rank]
        for kind in range(len(dtypes)):
            if sent:
                buffer = torch.cat([parts[kind].to(device) for parts in sent])
                if buffer.numel():
                    ops.append(dist.P2POp(dist.isend, buffer, peer))
        sizes = [
            layouts[peer].part_sizes(row)
            for source, row in plan[rank]
            if source == peer
        ]
        if not sizes:
            continue
        buffers = []
        for kind, dtype in enumerate(dtypes):
            kind_sizes = [row_sizes[kind] for row_sizes in sizes]
            buffer = torch.empty(sum(kind_sizes), dtype=dtype, device=device)
            if buffer.numel():
                ops.append(dist.P2POp(dist.irecv, buffer, peer))
            buffers.append(buffer.split(kind_sizes))
        received[peer] = [list(row_parts) for row_parts in zip(*buffers, strict=True)]
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    return received


def _assemble(
    rows: SegmentRows,
    value_pads: Sequence[float],
    plan: list[tuple[int, int]],
    layouts: list[RankRows],
    offsets: npt.NDArray[np.intp],
    parts: list[list[torch.Tensor]],
    pad_token_id: int,
) -> tuple[SegmentRows, list[torch.Tensor]]:
    """Rows this rank runs, built from each row's flat parts.

    :param rows: This rank's split rows, for devices and dtypes.
    :param value_pads: Value past a row's end of each per-position row value.
    :param plan: This rank's rows as ``(source rank, source row)``.
    :param layouts: Every rank's layout.
    :param offsets: First cross-rank episode number of each rank.
    :param parts: Flat parts of each row in ``plan`` order.
    :param pad_token_id: Token id that pads rows to a common width.
    :return: The rows and each of their row values.
    """
    device = rows.token_ids.device
    columns = list(zip(*parts, strict=True))
    layout = layouts[0]

    def padded(kind: int, value: float) -> torch.Tensor:
        return pad_sequence(
            [part.to(device) for part in columns[kind]],
            batch_first=True,
            padding_value=value,
        )

    values = [
        padded(layout.first_value_part + index, pad)
        if shape is None
        else torch.stack(
            [
                part.to(device).reshape(shape)
                for part in columns[layout.first_value_part + index]
            ]
        )
        for index, (shape, pad) in enumerate(
            zip(layout.value_shapes, value_pads, strict=True)
        )
    ]
    turn_ids = None
    if rows.turn_ids is not None:
        turn_ids = padded(TURN_IDS, -1).to(rows.turn_ids.dtype)
    pixel_values = None
    image_counts = None
    if rows.pixel_values is not None:
        pixel_shape = rows.pixel_values.shape[1:]
        pixel_values = torch.cat(
            [
                part.to(rows.pixel_values.device).reshape(-1, *pixel_shape)
                for part in columns[-1]
            ]
        )
        image_counts = [layouts[source].image_counts[row] for source, row in plan]
    sampling_logps = None
    if any(rank_rows.has_logps for rank_rows in layouts):
        sampling_logps = [
            part if layouts[source].logp_sizes[row] >= 0 else None
            for part, (source, row) in zip(columns[SAMPLING_LOGPS], plan, strict=True)
        ]
    starts = np.array([layouts[source].starts[row] for source, row in plan])
    lengths = np.array([layouts[source].lengths[row] for source, row in plan])
    balanced = SegmentRows(
        token_ids=padded(TOKENS, pad_token_id).to(rows.token_ids.dtype),
        action_masks=padded(ACTION_MASK, 0).to(rows.action_masks.dtype),
        row_episodes=np.array(
            [offsets[source] + layouts[source].episodes[row] for source, row in plan],
            dtype=np.intp,
        ),
        row_starts=starts.astype(np.intp),
        row_ends=(starts + lengths).astype(np.intp),
        turn_ids=turn_ids,
        sampling_logps=sampling_logps,
        pixel_values=pixel_values,
        pixel_image_counts=image_counts,
    )
    return balanced, values
